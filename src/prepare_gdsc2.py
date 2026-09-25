r"""Create auditable GDSC2 tables and train-only features, without fitting models.

Run from the repository root:
    .venv\Scripts\python.exe src\prepare_gdsc2.py

The input pickle is loaded only after its known SHA-256 is checked. Never load
an untrusted replacement pickle without independently verifying its origin.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn

from preprocessing import assign_cell_lines, feature_catalog, fit_expression_transform, morgan_bit_matrix


ROOT = Path(os.environ.get("DRP_RUN_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW = Path(os.environ.get("DRP_RAW_DIR", ROOT / "data" / "raw")).resolve()
OUT = ROOT / "data" / "processed"
CONFIG = ROOT / "configs" / "preprocessing.json"
MANIFEST = ROOT / "results" / "preprocessing_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.17g", lineterminator="\n")


def build() -> dict[str, object]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for filename, expected in config["input_sha256"].items():
        actual = sha256(RAW / filename)
        if actual != expected:
            raise ValueError(f"Input checksum mismatch: {filename}, {actual}")

    frame = pd.read_pickle(RAW / "gdsc2.pkl")
    if list(frame.columns) != ["ID1", "ID2", "X1", "X2", "Y"]:
        raise ValueError(f"Unexpected raw columns: {list(frame.columns)}")
    if frame[list(frame.columns)].isna().any().any():
        raise ValueError("Missing raw value")
    if not np.isfinite(frame["Y"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite raw Y")
    frame = frame.reset_index(drop=True)
    frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int32))
    frame["ID1"] = frame["ID1"].astype(str)
    frame["ID2"] = frame["ID2"].astype(str)
    frame["X1"] = frame["X1"].astype(str)

    # One profile per cell line. Compare all source references by value, not
    # merely by Python object identity, before separating features from labels.
    cell_ids = sorted(frame["ID2"].unique().tolist())
    first_profile: dict[str, np.ndarray] = {}
    first_smiles: dict[str, str] = {}
    for drug, cell, smiles, profile in frame[["ID1", "ID2", "X1", "X2"]].itertuples(index=False, name=None):
        array = np.asarray(profile, dtype=np.float64)
        if cell in first_profile:
            if not np.array_equal(array, first_profile[cell]):
                raise ValueError(f"Conflicting expression representations for {cell}")
        else:
            first_profile[cell] = array
        if drug in first_smiles and smiles != first_smiles[drug]:
            raise ValueError(f"Conflicting SMILES for {drug}")
        first_smiles[drug] = smiles
    gene_symbols = pd.read_csv(RAW / "gdsc_gene_symbols.tab", sep="\t").iloc[:, 0].astype(str).tolist()
    expression = np.stack([first_profile[cell] for cell in cell_ids])
    if expression.shape[1] != len(gene_symbols) or not np.isfinite(expression).all():
        raise ValueError("Expression length, gene-symbol alignment, or finiteness failed")
    catalog = feature_catalog(gene_symbols, config["expression"]["feature_id_prefix"])
    if not catalog["feature_id"].is_unique or not catalog["position"].is_unique:
        raise ValueError("Expression feature IDs must be unique")

    # The sole assignment is made from identifiers before constructing either
    # response table. No Y values participate in the split.
    split = config["split"]
    assignments = assign_cell_lines(
        cell_ids, seed=split["seed"], train_fraction=split["train_fraction"],
        validation_fraction=split["validation_fraction"],
    )
    if not assignments["cell_line_id"].is_unique or set(assignments["cell_line_id"]) != set(cell_ids):
        raise ValueError("Incomplete cell-line assignment")
    split_by_cell = dict(zip(assignments["cell_line_id"], assignments["split"]))
    cell_index = {cell: index for index, cell in enumerate(cell_ids)}

    # Explicitly inspect both representations in each source group before an
    # analytical average. Drug/experiment IDs beyond the drug name are absent.
    sensitivity_rows: list[dict[str, object]] = []
    source_to_pair = np.full(len(frame), -1, dtype=np.int32)
    duplicate_groups = 0
    duplicate_source_rows = 0
    duplicate_non_uprosertib = 0
    for (drug, cell), group in frame.groupby(["ID1", "ID2"], sort=True):
        smiles = first_smiles[drug]
        profile = first_profile[cell]
        for source_smiles, source_profile in group[["X1", "X2"]].itertuples(index=False, name=None):
            if source_smiles != smiles or not np.array_equal(np.asarray(source_profile, dtype=np.float64), profile):
                raise ValueError(f"Nonidentical features within drug-cell group: {drug}, {cell}")
        source_ids = group["source_row_id"].to_numpy(dtype=np.int32)
        values = group["Y"].to_numpy(dtype=np.float64)
        if len(group) > 1:
            duplicate_groups += 1
            duplicate_source_rows += len(group)
            duplicate_non_uprosertib += int(drug != config["primary_excluded_drug"])
        pair_id = len(sensitivity_rows)
        source_to_pair[source_ids] = pair_id
        sensitivity_rows.append({
            "pair_id": pair_id,
            "drug_id": drug,
            "cell_line_id": cell,
            "drug_index": None,  # filled after the stable drug catalog is created
            "cell_index": cell_index[cell],
            "split": split_by_cell[cell],
            "y": float(np.mean(values)),
            "source_row_ids": "|".join(map(str, source_ids.tolist())),
            "n_measurements": len(values),
            "y_min": float(np.min(values)),
            "y_max": float(np.max(values)),
            "y_range": float(np.max(values) - np.min(values)),
        })
    if np.any(source_to_pair < 0) or duplicate_non_uprosertib:
        raise ValueError("Source mapping incomplete or unexpected repeated drug")

    drugs = sorted(first_smiles)
    drug_index = {drug: index for index, drug in enumerate(drugs)}
    drug_table = pd.DataFrame({"drug_index": np.arange(len(drugs)), "drug_id": drugs, "smiles": [first_smiles[d] for d in drugs]})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity["drug_index"] = sensitivity["drug_id"].map(drug_index).astype(np.int32)
    if sensitivity[["drug_id", "cell_line_id"]].duplicated().any():
        raise ValueError("Sensitivity pairs are not unique")

    primary_raw = frame.loc[frame["ID1"] != config["primary_excluded_drug"]]
    primary = pd.DataFrame({
        "source_row_id": primary_raw["source_row_id"].to_numpy(),
        "drug_id": primary_raw["ID1"].to_numpy(),
        "cell_line_id": primary_raw["ID2"].to_numpy(),
        "drug_index": primary_raw["ID1"].map(drug_index).to_numpy(dtype=np.int32),
        "cell_index": primary_raw["ID2"].map(cell_index).to_numpy(dtype=np.int32),
        "split": primary_raw["ID2"].map(split_by_cell).to_numpy(),
        "y": primary_raw["Y"].to_numpy(dtype=np.float64),
    })
    if primary[["drug_id", "cell_line_id"]].duplicated().any():
        raise ValueError("Primary table has duplicated drug-cell pairs")
    if not np.array_equal(primary["y"].to_numpy(), frame.loc[primary["source_row_id"], "Y"].to_numpy()):
        raise ValueError("Primary Y was altered")
    excluded = int((frame["ID1"] == config["primary_excluded_drug"]).sum())

    mapping = pd.DataFrame({
        "source_row_id": frame["source_row_id"],
        "drug_id": frame["ID1"],
        "cell_line_id": frame["ID2"],
        "split": frame["ID2"].map(split_by_cell),
        "primary_row_id": frame["source_row_id"].where(frame["ID1"] != config["primary_excluded_drug"], pd.NA),
        "sensitivity_pair_id": source_to_pair,
    })
    if mapping["split"].isna().any() or primary["split"].isna().any() or sensitivity["split"].isna().any():
        raise ValueError("Unassigned split")
    for name, table in (("primary", primary), ("sensitivity", sensitivity)):
        if not table.groupby("drug_id")["split"].apply(lambda x: (x == "train").any()).all():
            raise ValueError(f"Training-drug coverage failed in {name}")

    transform = fit_expression_transform(
        expression, cell_ids,
        sorted(assignments.loc[assignments["split"] == "train", "cell_line_id"].tolist()),
        catalog["feature_id"].tolist(), config["expression"],
    )
    transformed = transform.transform(expression)
    if not np.isfinite(transformed).all():
        raise ValueError("Nonfinite transformed expression")
    fingerprints = morgan_bit_matrix(drug_table["smiles"].tolist(), config["fingerprint"])
    if not np.isin(fingerprints, [0, 1]).all():
        raise ValueError("Fingerprints are not bit vectors")

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(assignments, "cell_line_assignments.csv")
    write_csv(catalog, "expression_features.csv")
    write_csv(drug_table, "drugs.csv")
    write_csv(primary, "primary_responses.csv")
    write_csv(sensitivity, "sensitivity_responses.csv")
    write_csv(mapping, "source_row_mapping.csv")
    np.save(OUT / "expression_raw.npy", expression, allow_pickle=False)
    np.save(OUT / "expression_pca128.npy", transformed, allow_pickle=False)
    np.save(OUT / "morgan_radius2_2048.npy", fingerprints, allow_pickle=False)
    joblib.dump(transform, OUT / "expression_transform.joblib", compress=0)

    file_names = sorted(path.name for path in OUT.iterdir() if path.is_file())
    manifest: dict[str, object] = {
        "config_sha256": sha256(CONFIG),
        "raw_sha256": config["input_sha256"],
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__, "joblib": joblib.__version__,
        },
        "counts": {
            "raw_rows": len(frame), "primary_rows": len(primary), "primary_excluded_rows": excluded,
            "primary_drugs": int(primary["drug_id"].nunique()),
            "sensitivity_pairs": len(sensitivity), "sensitivity_drugs": int(sensitivity["drug_id"].nunique()),
            "uprosertib_sensitivity_pairs": int((sensitivity["drug_id"] == config["primary_excluded_drug"]).sum()),
            "repeated_source_groups": duplicate_groups, "source_rows_in_repeated_groups": duplicate_source_rows,
            "duplicate_gene_symbol_positions": int(catalog["gene_symbol"].duplicated().sum()),
            "split_cell_lines": {str(k): int(v) for k, v in assignments["split"].value_counts().sort_index().items()},
            "primary_rows_by_split": {str(k): int(v) for k, v in primary["split"].value_counts().sort_index().items()},
            "sensitivity_pairs_by_split": {str(k): int(v) for k, v in sensitivity["split"].value_counts().sort_index().items()},
        },
        "shapes": {
            "expression_raw": list(expression.shape),
            "expression_pca128": list(transformed.shape),
            "morgan_radius2_2048": list(fingerprints.shape),
        },
        "expression_preprocessing": {
            "fit_unique_training_cells": len(transform.train_cell_ids),
            "original_features": expression.shape[1],
            "zero_variance_removed": int(expression.shape[1] - transform.selector.get_support().sum()),
            "features_after_filter": int(transform.selector.get_support().sum()),
            "pca_components": int(transform.pca.n_components_),
            "retained_training_variance_fraction": float(transform.pca.explained_variance_ratio_.sum()),
            "train_cell_ids_sha256": hashlib.sha256("\n".join(transform.train_cell_ids).encode()).hexdigest(),
        },
        "artifacts_sha256": {name: sha256(OUT / name) for name in file_names},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build()["counts"], indent=2))
