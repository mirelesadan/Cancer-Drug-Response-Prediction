"""Audit the unmodified TDC GDSC2 file; no filtering, splitting, or fitting.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\audit_gdsc2.py

The input is a pickle from Harvard Dataverse. Load it only after verifying its
source and checksum; pickle is not safe for untrusted files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.error")


def distribution(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    series = pd.Series(values).dropna()
    if series.empty:
        return {"count": 0}
    return {
        "count": int(series.size),
        "min": float(series.min()),
        "p01": float(series.quantile(0.01)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std()),
    }


def missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def audit(data_path: Path, genes_path: Path) -> dict[str, object]:
    frame = pd.read_pickle(data_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(frame).__name__}")
    expected = {"ID1", "ID2", "X1", "X2", "Y"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Missing expected columns: {sorted(expected - set(frame.columns))}")

    missing = {column: int(frame[column].map(missing_scalar).sum()) for column in expected}
    y = pd.to_numeric(frame["Y"], errors="coerce").to_numpy(dtype=float)
    finite_y = np.isfinite(y)
    drug_counts = frame.groupby("ID1", dropna=False).size()
    cell_counts = frame.groupby("ID2", dropna=False).size()

    # Check both identifier-to-representation directions. No row is modified.
    drug_to_smiles: dict[str, set[str]] = defaultdict(set)
    smiles_to_drugs: dict[str, set[str]] = defaultdict(set)
    cell_to_objects: dict[str, set[int]] = defaultdict(set)
    object_to_cells: dict[int, set[str]] = defaultdict(set)
    objects: dict[int, object] = {}
    for drug, cell, smiles, expression in frame[["ID1", "ID2", "X1", "X2"]].itertuples(index=False, name=None):
        drug_key, cell_key = str(drug), str(cell)
        if not missing_scalar(smiles):
            smiles_text = str(smiles)
            drug_to_smiles[drug_key].add(smiles_text)
            smiles_to_drugs[smiles_text].add(drug_key)
        if not missing_scalar(expression):
            object_id = id(expression)
            objects[object_id] = expression
            cell_to_objects[cell_key].add(object_id)
            object_to_cells[object_id].add(cell_key)

    invalid_smiles = []
    canonical_to_drugs: dict[str, set[str]] = defaultdict(set)
    for smiles, drugs in smiles_to_drugs.items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid_smiles.append({"drugs": sorted(drugs), "smiles": smiles})
        else:
            canonical_to_drugs[Chem.MolToSmiles(molecule, canonical=True)].update(drugs)

    expression_shapes: Counter[str] = Counter()
    expression_dtypes: Counter[str] = Counter()
    bad_expression_objects = 0
    nonfinite_expression_entries = 0
    objects_with_nonfinite = 0
    hash_to_cells: dict[str, set[str]] = defaultdict(set)
    for object_id, value in objects.items():
        try:
            array = np.asarray(value)
            expression_shapes[str(array.shape)] += 1
            expression_dtypes[str(array.dtype)] += 1
            numeric = np.asarray(array, dtype=np.float64)
            bad = int((~np.isfinite(numeric)).sum())
            nonfinite_expression_entries += bad
            objects_with_nonfinite += int(bad > 0)
            digest = hashlib.sha256(np.ascontiguousarray(numeric).tobytes()).hexdigest()
            hash_to_cells[digest].update(object_to_cells[object_id])
        except (TypeError, ValueError):
            bad_expression_objects += 1

    pair = frame.groupby(["ID1", "ID2"], dropna=False)["Y"].agg(["size", "nunique", "min", "max"])
    repeated = pair[pair["size"] > 1]
    conflicts = repeated[repeated["nunique"] > 1]
    conflict_ranges = conflicts["max"] - conflicts["min"]
    repeated_by_drug = repeated.reset_index().groupby("ID1").size().sort_values(ascending=False)
    genes = pd.read_csv(genes_path, sep="\t")
    gene_symbols = genes.iloc[:, 0].astype(str)

    # Feasibility only: these are independent trial draws, not saved model splits.
    all_cells = sorted(cell_to_objects)
    all_drugs = set(drug_to_smiles)
    drug_cell_sets = {
        drug: set(group["ID2"].astype(str))
        for drug, group in frame.groupby("ID1", dropna=False)
    }
    rng = np.random.default_rng(20260921)
    train_coverage_failures: Counter[str] = Counter()
    validation_coverage_failures: Counter[str] = Counter()
    test_coverage_failures: Counter[str] = Counter()
    trials_train_all_covered = 0
    trials_all_partitions_covered = 0
    for _ in range(100):
        shuffled = rng.permutation(all_cells)
        train_end = round(0.70 * len(all_cells))
        validation_end = train_end + round(0.15 * len(all_cells))
        partitions = {
            "train": set(shuffled[:train_end]),
            "validation": set(shuffled[train_end:validation_end]),
            "test": set(shuffled[validation_end:]),
        }
        missing_by_partition = {
            name: [drug for drug in all_drugs if not (drug_cell_sets[drug] & cells)]
            for name, cells in partitions.items()
        }
        trials_train_all_covered += int(not missing_by_partition["train"])
        trials_all_partitions_covered += int(all(not missing for missing in missing_by_partition.values()))
        train_coverage_failures.update(missing_by_partition["train"])
        validation_coverage_failures.update(missing_by_partition["validation"])
        test_coverage_failures.update(missing_by_partition["test"])

    result: dict[str, object] = {
        "input": {"data": str(data_path), "genes": str(genes_path)},
        "schema": {"rows": len(frame), "columns": list(frame.columns), "dtypes": frame.dtypes.astype(str).to_dict()},
        "identities": {"unique_drugs": len(drug_to_smiles), "unique_cell_lines": len(cell_to_objects), "unique_smiles": len(smiles_to_drugs)},
        "missing_by_column": missing,
        "nonfinite_y": int((~finite_y).sum()),
        "response": distribution(y[finite_y]),
        "rows_per_drug": distribution(drug_counts),
        "rows_per_cell_line": distribution(cell_counts),
        "least_measured_drugs": [[str(k), int(v)] for k, v in drug_counts.nsmallest(5).items()],
        "most_measured_drugs": [[str(k), int(v)] for k, v in drug_counts.nlargest(5).items()],
        "unique_cell_lines_per_drug": distribution(pd.Series({k: len(v) for k, v in drug_cell_sets.items()})),
        "drug_mapping": {
            "drugs_with_multiple_smiles": {k: len(v) for k, v in drug_to_smiles.items() if len(v) > 1},
            "smiles_shared_by_multiple_drug_names": {k: sorted(v) for k, v in smiles_to_drugs.items() if len(v) > 1},
            "canonical_structures_shared_by_multiple_drug_names": {k: sorted(v) for k, v in canonical_to_drugs.items() if len(v) > 1},
            "invalid_smiles": invalid_smiles,
        },
        "expression": {
            "unique_python_objects": len(objects),
            "shapes": dict(expression_shapes),
            "dtypes": dict(expression_dtypes),
            "bad_objects": bad_expression_objects,
            "nonfinite_entries_in_unique_objects": nonfinite_expression_entries,
            "objects_with_nonfinite": objects_with_nonfinite,
            "cells_with_multiple_expression_objects": {k: len(v) for k, v in cell_to_objects.items() if len(v) > 1},
            "expression_objects_shared_by_multiple_cells": sum(len(v) > 1 for v in object_to_cells.values()),
            "identical_expression_values_shared_by_multiple_cells": sum(len(v) > 1 for v in hash_to_cells.values()),
        },
        "genes": {
            "table_rows": len(genes), "columns": list(genes.columns),
            "blank_symbols": int(gene_symbols.str.strip().eq("").sum()),
            "duplicated_symbols": int(gene_symbols.duplicated().sum()),
            "length_matches_all_expression_objects": all(str((len(genes),)) == shape for shape in expression_shapes),
        },
        "duplicate_measurements": {
            "unique_drug_cell_pairs": len(pair),
            "extra_rows_beyond_unique_pairs": int(len(frame) - len(pair)),
            "pairs_with_repeated_rows": len(repeated),
            "pairs_with_conflicting_y": len(conflicts),
            "exact_duplicate_key_and_y_rows": int(frame.duplicated(["ID1", "ID2", "Y"]).sum()),
            "conflicting_pair_y_range": distribution(conflict_ranges),
            "drugs_with_repeated_pairs": len(repeated_by_drug),
            "most_repeated_drugs": [[str(k), int(v)] for k, v in repeated_by_drug.head(10).items()],
        },
        "split_feasibility": {
            "trial_train_fraction_of_cells": 0.70,
            "independent_trial_draws": 100,
            "trial_seed": 20260921,
            "all_drugs_covered_in_train_trial_count": trials_train_all_covered,
            "all_drugs_covered_in_all_partitions_trial_count": trials_all_partitions_covered,
            "most_often_uncovered_in_train": train_coverage_failures.most_common(10),
            "most_often_uncovered_in_validation": validation_coverage_failures.most_common(10),
            "most_often_uncovered_in_test": test_coverage_failures.most_common(10),
            "minimum_cell_lines_for_any_drug": min(map(len, drug_cell_sets.values())),
            "note": "Feasibility diagnostic only; no train, validation, or test split is selected or saved.",
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/gdsc2.pkl"))
    parser.add_argument("--genes", type=Path, default=Path("data/raw/gdsc_gene_symbols.tab"))
    parser.add_argument("--output", type=Path, default=Path("results/data_audit_stats.json"))
    args = parser.parse_args()
    result = audit(args.data, args.genes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
