r"""Replay the frozen exploratory GNN test predictions and paired bootstrap.

Run from the repository root: .venv\Scripts\python.exe src\check_gnn_final.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from gnn import build_drug_graphs, predict_rows
from run_gnn import make_model
from run_neural import configure_determinism, sha256


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def _assert_close(actual: float, expected: float, *, tolerance: float = 1e-10) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"Saved metric differs from replay: {actual} versus {expected}")


def check() -> None:
    lock_path = ROOT / "configs" / "gnn_final_evaluation.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "gnn_final_manifest.json").read_text(encoding="utf-8"))
    validation_manifest = json.loads((RESULTS / "gnn_validation_manifest.json").read_text(encoding="utf-8"))
    if manifest["lock_sha256"] != sha256(lock_path):
        raise ValueError("GNN final protocol lock changed")
    for relative, expected in lock["frozen_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Final artifact changed: {relative}")

    config = json.loads((ROOT / "configs" / "gnn_extension.json").read_text(encoding="utf-8"))
    configure_determinism(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drugs = pd.read_csv(PROCESSED / "drugs.csv")
    graphs, graph_hash = build_drug_graphs(drugs)
    if graph_hash != validation_manifest["graph_sha256"]:
        raise ValueError("Drug graphs changed")
    graphs = graphs.to(device)
    scores = torch.as_tensor(np.load(PROCESSED / "expression_pca128_scaled.npy", allow_pickle=False),
                             dtype=torch.float32, device=device)
    primary = pd.read_csv(PROCESSED / "primary_responses.csv", float_precision="round_trip")
    test = primary.loc[primary["split"].eq("test")].reset_index(drop=True)
    saved = pd.read_csv(RESULTS / "tmp" / "gnn_test_predictions.csv", float_precision="round_trip")
    comparison = pd.read_csv(RESULTS / "gnn_test_comparison.csv")
    detail = pd.read_csv(RESULTS / "gnn_test_by_drug.csv")
    intervals = pd.read_csv(RESULTS / "gnn_bootstrap_intervals.csv")
    runs = pd.read_csv(RESULTS / "gnn_validation_runs.csv")
    if len(test) != 13622 or len(saved) != len(test) or test["cell_line_id"].nunique() != 120:
        raise ValueError("Frozen GNN test rows changed")
    for column in ("source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "split"):
        if saved[column].tolist() != test[column].tolist():
            raise ValueError(f"Saved GNN test row alignment failed: {column}")
    if not np.array_equal(saved["y"].to_numpy(), test["y"].to_numpy()) or not np.isfinite(saved["y"]).all():
        raise ValueError("Saved test labels changed or are nonfinite")
    assignments = pd.read_csv(PROCESSED / "cell_line_assignments.csv")
    if saved["cell_line_id"].tolist() != assignments.loc[saved["cell_index"], "cell_line_id"].tolist():
        raise ValueError("Saved test cell indices changed")
    if saved["drug_id"].tolist() != drugs.loc[saved["drug_index"], "drug_id"].tolist():
        raise ValueError("Saved test drug indices changed")
    if not assignments.loc[saved["cell_index"], "split"].eq("test").all():
        raise ValueError("GNN test contains non-test cells")
    if saved["drug_id"].eq("Uprosertib").any():
        raise ValueError("Excluded drug entered the GNN primary test")

    cell_index = torch.as_tensor(saved["cell_index"].to_numpy(dtype=np.int64), device=device)
    drug_index = torch.as_tensor(saved["drug_index"].to_numpy(dtype=np.int64), device=device)
    max_prediction_difference = 0.0
    for run in runs.itertuples(index=False):
        checkpoint = torch.load(ROOT / run.checkpoint, map_location=device, weights_only=True)
        if checkpoint["seed"] != run.seed or checkpoint["best_epoch"] != run.best_epoch:
            raise ValueError("Checkpoint seed or epoch changed")
        _assert_close(checkpoint["validation_rmse"], run.validation_rmse, tolerance=1e-5)
        if checkpoint["config_sha256"] != validation_manifest["config_sha256"] or checkpoint["graph_sha256"] != graph_hash:
            raise ValueError("Checkpoint configuration or graph identity changed")
        model = make_model(config, len(drugs), device)
        model.load_state_dict(checkpoint["state_dict"])
        replay = predict_rows(model, cell_index, drug_index, scores, graphs,
                              int(config["training"]["evaluation_batch_size"])).numpy().astype(np.float64)
        difference = float(np.max(np.abs(replay - saved[run.run_id].to_numpy(dtype=np.float64))))
        max_prediction_difference = max(max_prediction_difference, difference)
        if difference > 1e-5:
            raise ValueError(f"Checkpoint test replay exceeds float32 tolerance for {run.run_id}: {difference}")

    y = saved["y"].to_numpy(dtype=np.float64)
    for row in comparison.itertuples(index=False):
        if row.model == "gnn_three_seed_mean":
            members = comparison.loc[comparison["model"].str.startswith("gnn_seed_")]
            for metric in ("rmse", "mae", "equal_drug_mean_rmse", "median_within_drug_spearman"):
                _assert_close(float(members[metric].mean()), float(getattr(row, metric)))
                _assert_close(float(members[metric].std(ddof=1)), float(getattr(row, metric + "_sample_sd")))
            continue
        residual = saved[row.model].to_numpy(dtype=np.float64) - y
        _assert_close(float(np.sqrt(np.mean(residual ** 2))), row.rmse)
        _assert_close(float(np.mean(np.abs(residual))), row.mae)
        one_drug = detail.loc[detail["model"].eq(row.model)]
        if one_drug["test_n"].sum() != len(saved):
            raise ValueError("Per-drug test counts do not sum to test rows")
        _assert_close(float(one_drug["rmse"].mean()), row.equal_drug_mean_rmse)
        defined = one_drug["spearman"].dropna()
        if len(defined) != row.spearman_defined_drugs:
            raise ValueError("Defined within-drug correlations changed")
        _assert_close(float(defined.median()), row.median_within_drug_spearman)
        for drug in one_drug.itertuples(index=False):
            subset = saved.loc[saved["drug_id"].eq(drug.drug_id)]
            if len(subset) != drug.test_n:
                raise ValueError("Drug test count changed")
            _assert_close(float(np.sqrt(np.mean((subset[row.model] - subset["y"]) ** 2))), drug.rmse)
            if drug.spearman_status == "defined":
                rho = float(spearmanr(subset["y"], subset[row.model]).statistic)
                _assert_close(rho, drug.spearman)
            elif np.isfinite(drug.spearman):
                raise ValueError("Undefined correlation was saved as finite")

    old = np.load(RESULTS / "tmp" / "final_bootstrap_replicates.npz", allow_pickle=False)
    new = np.load(RESULTS / "tmp" / "gnn_bootstrap_replicates.npz", allow_pickle=False)
    cell_ids, group = np.unique(saved["cell_line_id"].to_numpy(), return_inverse=True)
    counts = np.bincount(group, minlength=len(cell_ids))
    draws = new["draws"]
    if draws.shape != (2000, 120) or not np.array_equal(draws, old["draws"]):
        raise ValueError("Paired bootstrap draws changed")
    if not np.array_equal(cell_ids.astype(str), new["cell_ids"].astype(str)) or not np.array_equal(counts, new["row_counts_per_cell"]):
        raise ValueError("Bootstrap cell-line mapping changed")
    if not np.all(draws.sum(axis=1) == 120) or not np.all(draws >= 0):
        raise ValueError("Bootstrap cluster multiplicities changed")
    denominator = draws @ counts
    for model_name in ("per_drug_ridge", "gnn_seed_17", "gnn_seed_29", "gnn_seed_43"):
        residual = saved[model_name].to_numpy(dtype=np.float64) - y
        sse = np.bincount(group, weights=residual ** 2, minlength=120)
        sae = np.bincount(group, weights=np.abs(residual), minlength=120)
        expected_rmse = np.sqrt((draws @ sse) / denominator)
        expected_mae = (draws @ sae) / denominator
        if not np.allclose(new[f"{model_name}_rmse"], expected_rmse, rtol=0, atol=1e-12):
            raise ValueError("Bootstrap RMSE changed")
        if not np.allclose(new[f"{model_name}_mae"], expected_mae, rtol=0, atol=1e-12):
            raise ValueError("Bootstrap MAE changed")
    for metric in ("rmse", "mae"):
        seed_mean = np.mean([new[f"gnn_seed_{seed}_{metric}"] for seed in (17, 29, 43)], axis=0)
        if not np.allclose(seed_mean, new[f"gnn_three_seed_mean_{metric}"], rtol=0, atol=1e-12):
            raise ValueError("GNN seed-metric bootstrap mean changed")
    for interval in intervals.itertuples(index=False):
        for metric in ("rmse", "mae"):
            low, high = np.percentile(new[f"{interval.model}_{metric}"], [2.5, 97.5])
            _assert_close(float(low), getattr(interval, metric + "_ci_low"))
            _assert_close(float(high), getattr(interval, metric + "_ci_high"))
        delta = new[f"{interval.model}_rmse"] - new["per_drug_ridge_rmse"]
        low, high = np.percentile(delta, [2.5, 97.5])
        _assert_close(float(low), interval.paired_rmse_delta_ci_low)
        _assert_close(float(high), interval.paired_rmse_delta_ci_high)

    outcome = {
        "check": "passed", "test_rows": len(saved), "test_cell_lines": len(cell_ids),
        "checkpoint_count": len(runs), "bootstrap_draws": len(draws),
        "max_abs_prediction_difference": max_prediction_difference, "prediction_tolerance": 1e-5,
        "scope": "exploratory_GNN_extension_original_test_results_already_public",
    }
    (RESULTS / "gnn_final_replay_check.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(outcome, sort_keys=True))


if __name__ == "__main__":
    check()
