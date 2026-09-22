"""Verify saved GDSC2 row alignment, split, and training-only features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import morgan_bit_matrix


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check() -> None:
    manifest = json.loads((ROOT / "results" / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs" / "preprocessing.json").read_text(encoding="utf-8"))
    assert digest(ROOT / "configs" / "preprocessing.json") == manifest["config_sha256"]
    for filename, expected in manifest["artifacts_sha256"].items():
        assert digest(OUT / filename) == expected, filename
    raw = pd.read_pickle(ROOT / "data" / "raw" / "gdsc2.pkl").reset_index(drop=True)
    assignment = pd.read_csv(OUT / "cell_line_assignments.csv")
    features = pd.read_csv(OUT / "expression_features.csv")
    drugs = pd.read_csv(OUT / "drugs.csv")
    primary = pd.read_csv(OUT / "primary_responses.csv", float_precision="round_trip")
    sensitivity = pd.read_csv(OUT / "sensitivity_responses.csv", dtype={"source_row_ids": str}, float_precision="round_trip")
    mapping = pd.read_csv(OUT / "source_row_mapping.csv")
    expression = np.load(OUT / "expression_raw.npy", allow_pickle=False)
    pca_expression = np.load(OUT / "expression_pca128.npy", allow_pickle=False)
    fingerprint = np.load(OUT / "morgan_radius2_2048.npy", allow_pickle=False)
    transform = joblib.load(OUT / "expression_transform.joblib")

    assert assignment["cell_line_id"].is_unique and assignment["cell_line_id"].is_monotonic_increasing
    assert set(assignment["split"]) == {"train", "validation", "test"}
    assert assignment.groupby("cell_line_id")["split"].nunique().max() == 1
    assert len(assignment) == raw["ID2"].nunique()
    assert features["feature_id"].is_unique and len(features) == expression.shape[1]
    assert features["gene_symbol"].duplicated().sum() == 317
    assert features["position"].tolist() == list(range(expression.shape[1]))
    assert pca_expression.shape == (len(assignment), 128)
    assert fingerprint.shape == (len(drugs), 2048)
    assert np.isfinite(expression).all() and np.isfinite(pca_expression).all()
    assert np.isin(fingerprint, [0, 1]).all()
    assert np.array_equal(fingerprint, morgan_bit_matrix(drugs["smiles"].tolist(), config["fingerprint"]))
    assert np.allclose(transform.transform(expression), pca_expression, atol=1e-10)

    train_ids = tuple(sorted(assignment.loc[assignment["split"] == "train", "cell_line_id"].tolist()))
    assert transform.train_cell_ids == train_ids
    assert transform.feature_ids == tuple(features["feature_id"])
    train_rows = assignment.index[assignment["split"] == "train"].to_numpy()
    selected_train = expression[train_rows][:, transform.selector.get_support()]
    np.testing.assert_allclose(transform.scaler.mean_, selected_train.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(transform.pca.mean_, transform.scaler.transform(selected_train).mean(axis=0), rtol=1e-10, atol=1e-10)

    assert mapping["source_row_id"].tolist() == list(range(len(raw)))
    assert mapping["split"].tolist() == assignment.set_index("cell_line_id").loc[raw["ID2"], "split"].tolist()
    assert primary["source_row_id"].is_unique and len(primary) == (raw["ID1"] != "Uprosertib").sum()
    assert not (primary["drug_id"] == "Uprosertib").any()
    assert np.array_equal(primary["y"].to_numpy(), raw.loc[primary["source_row_id"], "Y"].to_numpy())
    assert primary["cell_line_id"].tolist() == raw.loc[primary["source_row_id"], "ID2"].tolist()
    assert primary["drug_id"].tolist() == raw.loc[primary["source_row_id"], "ID1"].tolist()
    assert mapping["primary_row_id"].notna().sum() == len(primary)
    assert sensitivity[["drug_id", "cell_line_id"]].duplicated().sum() == 0
    assert len(sensitivity) == raw.groupby(["ID1", "ID2"]).ngroups
    assert sensitivity["n_measurements"].sum() == len(raw)
    assert mapping["sensitivity_pair_id"].between(0, len(sensitivity)-1).all()
    for row in sensitivity.itertuples(index=False):
        ids = [int(item) for item in row.source_row_ids.split("|")]
        group = raw.iloc[ids]
        assert (group["ID1"] == row.drug_id).all() and (group["ID2"] == row.cell_line_id).all()
        assert len(ids) == row.n_measurements
        assert np.all(mapping.loc[ids, "sensitivity_pair_id"].to_numpy() == row.pair_id)
        values = group["Y"].to_numpy()
        assert np.isclose(row.y, values.mean(), rtol=1e-14, atol=1e-14)
        assert np.isclose(row.y_min, values.min()) and np.isclose(row.y_max, values.max())
        assert np.isclose(row.y_range, values.max() - values.min())
    for table in (primary, sensitivity):
        assert table["split"].tolist() == assignment.set_index("cell_line_id").loc[table["cell_line_id"], "split"].tolist()
        assert table["drug_id"].tolist() == drugs.loc[table["drug_index"], "drug_id"].tolist()
        assert table["cell_line_id"].tolist() == assignment.loc[table["cell_index"], "cell_line_id"].tolist()
        assert table.groupby("drug_id")["split"].apply(lambda s: (s == "train").any()).all()
    print("PASS: checksums; row/label/feature alignment; split; duplicate symbols; finite outputs; train-only fit; molecular bits")


if __name__ == "__main__":
    check()
