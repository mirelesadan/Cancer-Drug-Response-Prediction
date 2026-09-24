r"""Evaluate the frozen exploratory GNN extension on the existing test rows.

Run only after configs/gnn_final_evaluation.json is frozen:
    .venv\Scripts\python.exe src\evaluate_gnn.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch

from baselines import by_drug_validation, errors
from gnn import build_drug_graphs, predict_rows
from run_gnn import make_model
from run_neural import configure_determinism, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TEMP = RESULTS / "tmp"
LOCK_PATH = ROOT / "configs" / "gnn_final_evaluation.json"


def check_lock() -> dict:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if not lock["no_further_training_or_tuning"] or lock["reference_model"] != "unchanged_per_drug_ridge":
        raise ValueError("GNN test protocol is not frozen")
    for relative, expected in lock["frozen_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Locked GNN input changed: {relative}")
    return lock


def summarize(name: str, frame: pd.DataFrame, train_counts: dict[str, int]) -> tuple[dict, pd.DataFrame]:
    y = frame["y"].to_numpy(dtype=np.float64)
    estimate = frame[name].to_numpy(dtype=np.float64)
    rmse, mae = errors(y, estimate)
    detail = by_drug_validation(frame["drug_id"].to_numpy(), y, estimate, train_counts,
                                minimum_spearman_n=3).rename(columns={"validation_n": "test_n"})
    if detail["test_n"].sum() != len(frame):
        raise ValueError("GNN test drug counts do not sum to response rows")
    detail.insert(0, "model", name)
    eligible = int((detail["test_n"] >= 3).sum())
    defined = int(detail["spearman"].notna().sum())
    return {
        "model": name, "test_n": len(frame), "test_cell_lines": frame["cell_line_id"].nunique(),
        "test_drugs": len(detail), "rmse": rmse, "mae": mae,
        "equal_drug_mean_rmse": float(detail["rmse"].mean()),
        "median_within_drug_spearman": float(detail["spearman"].median()) if defined else np.nan,
        "spearman_eligible_drugs": eligible, "spearman_defined_drugs": defined,
    }, detail


def paired_bootstrap(frame: pd.DataFrame, original_bootstrap_path: Path) -> tuple[pd.DataFrame, dict]:
    """Apply the original 2,000 whole-cell draws to new GNN residuals."""
    old = np.load(original_bootstrap_path, allow_pickle=False)
    draws = old["draws"]
    reference = old["per_drug_ridge_rmse"]
    cell_ids, row_cluster = np.unique(frame["cell_line_id"].to_numpy(), return_inverse=True)
    counts = np.bincount(row_cluster, minlength=len(cell_ids))
    if draws.shape != (2000, 120) or not np.array_equal(cell_ids.astype(str), old["cell_ids"].astype(str)):
        raise ValueError("Original bootstrap cell-line catalog changed")
    if not np.array_equal(counts, old["row_counts_per_cell"]) or not np.all(draws.sum(axis=1) == 120):
        raise ValueError("Bootstrap cluster counts or multiplicities changed")
    denominator = draws @ counts
    if not np.all(denominator > 0):
        raise ValueError("Empty GNN bootstrap draw")
    models = ["per_drug_ridge", "gnn_seed_17", "gnn_seed_29", "gnn_seed_43"]
    y = frame["y"].to_numpy(dtype=np.float64)
    replicates: dict[str, np.ndarray] = {}
    for name in models:
        residual = frame[name].to_numpy(dtype=np.float64) - y
        sse = np.bincount(row_cluster, weights=residual * residual, minlength=120)
        sae = np.bincount(row_cluster, weights=np.abs(residual), minlength=120)
        replicates[f"{name}_rmse"] = np.sqrt((draws @ sse) / denominator)
        replicates[f"{name}_mae"] = (draws @ sae) / denominator
    if not np.allclose(replicates["per_drug_ridge_rmse"], reference, rtol=0, atol=1e-12):
        raise ValueError("GNN comparison did not reproduce prior paired Ridge draws")
    for metric in ("rmse", "mae"):
        replicates[f"gnn_three_seed_mean_{metric}"] = np.mean(
            [replicates[f"gnn_seed_{seed}_{metric}"] for seed in (17, 29, 43)], axis=0)
    rows = []
    for name in models + ["gnn_three_seed_mean"]:
        rmse_draws = replicates[f"{name}_rmse"]
        mae_draws = replicates[f"{name}_mae"]
        rmse_low, rmse_high = np.percentile(rmse_draws, [2.5, 97.5])
        mae_low, mae_high = np.percentile(mae_draws, [2.5, 97.5])
        delta_low, delta_high = np.percentile(rmse_draws - reference, [2.5, 97.5])
        rows.append({
            "model": name, "rmse_ci_low": rmse_low, "rmse_ci_high": rmse_high,
            "mae_ci_low": mae_low, "mae_ci_high": mae_high,
            "paired_rmse_delta_ci_low": delta_low, "paired_rmse_delta_ci_high": delta_high,
        })
    replicates["draws"] = draws
    replicates["cell_ids"] = cell_ids.astype(str)
    replicates["row_counts_per_cell"] = counts
    return pd.DataFrame(rows), replicates


def main() -> None:
    started = perf_counter()
    lock = check_lock()  # hash check precedes reading any test response value
    config = json.loads((ROOT / "configs" / "gnn_extension.json").read_text(encoding="utf-8"))
    configure_determinism(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((RESULTS / "gnn_validation_manifest.json").read_text(encoding="utf-8"))
    drugs = pd.read_csv(OUT / "drugs.csv")
    graphs, graph_hash = build_drug_graphs(drugs)
    if graph_hash != manifest["graph_sha256"]:
        raise ValueError("GNN drug graph identity changed")
    graphs = graphs.to(device)
    expression = torch.as_tensor(np.load(OUT / "expression_pca128_scaled.npy", allow_pickle=False), dtype=torch.float32, device=device)
    primary = pd.read_csv(OUT / "primary_responses.csv", float_precision="round_trip")
    test = primary.loc[primary["split"].eq("test")].reset_index(drop=True)
    train = primary.loc[primary["split"].eq("train")]
    old = pd.read_csv(TEMP / "primary_test_predictions.csv", float_precision="round_trip")
    if len(test) != len(old) or len(test) != 13622 or test["cell_line_id"].nunique() != 120:
        raise ValueError("GNN test composition changed")
    for column in ("source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "split"):
        if test[column].tolist() != old[column].tolist():
            raise ValueError(f"GNN and baseline test rows differ: {column}")
    if not np.array_equal(test["y"].to_numpy(), old["y"].to_numpy()):
        raise ValueError("GNN and baseline test targets differ")
    assignments = pd.read_csv(OUT / "cell_line_assignments.csv")
    if test["cell_line_id"].tolist() != assignments.loc[test["cell_index"], "cell_line_id"].tolist():
        raise ValueError("GNN test cell index misaligned")
    if test["drug_id"].tolist() != drugs.loc[test["drug_index"], "drug_id"].tolist():
        raise ValueError("GNN test drug index misaligned")
    if not assignments.loc[test["cell_index"], "split"].eq("test").all():
        raise ValueError("GNN test contains non-test cell line")
    if test["drug_id"].eq("Uprosertib").any() or test["drug_id"].nunique() != 135:
        raise ValueError("GNN primary test exclusion or drug count changed")
    if not set(test["drug_id"]).issubset(set(train["drug_id"])):
        raise ValueError("GNN test drug not represented in training")
    if set(test["cell_line_id"]) & set(train["cell_line_id"]):
        raise ValueError("GNN train/test cell-line overlap")
    cell_index = torch.as_tensor(test["cell_index"].to_numpy(dtype=np.int64), device=device)
    drug_index = torch.as_tensor(test["drug_index"].to_numpy(dtype=np.int64), device=device)
    result = test.copy()
    result["per_drug_ridge"] = old["per_drug_ridge"].to_numpy(dtype=np.float64)
    for seed in (17, 29, 43):
        checkpoint = torch.load(RESULTS / "checkpoints" / f"gnn_seed_{seed}.pt", map_location=device, weights_only=True)
        if checkpoint["seed"] != seed or checkpoint["config_sha256"] != manifest["config_sha256"] or checkpoint["graph_sha256"] != graph_hash:
            raise ValueError("GNN checkpoint metadata changed after lock")
        model = make_model(config, len(drugs), device)
        model.load_state_dict(checkpoint["state_dict"])
        result[f"gnn_seed_{seed}"] = predict_rows(model, cell_index, drug_index, expression, graphs,
                                                  int(config["training"]["evaluation_batch_size"])).numpy().astype(np.float64)
    if not np.isfinite(result[["per_drug_ridge", "gnn_seed_17", "gnn_seed_29", "gnn_seed_43"]].to_numpy()).all():
        raise ValueError("Nonfinite GNN test prediction")
    train_counts = train.groupby("drug_id").size().to_dict()
    summaries, details = [], []
    for name in ("per_drug_ridge", "gnn_seed_17", "gnn_seed_29", "gnn_seed_43"):
        summary, detail = summarize(name, result, train_counts)
        summaries.append(summary)
        details.append(detail)
    comparison = pd.DataFrame(summaries)
    old_metrics = pd.read_csv(RESULTS / "final_test_comparison.csv")
    old_ridge = old_metrics.loc[old_metrics["model"].eq("per_drug_ridge")].iloc[0]
    if abs(comparison.iloc[0]["rmse"] - old_ridge["rmse"]) > 1e-12:
        raise ValueError("Original per-drug Ridge test metric changed")
    mean_row = {"model": "gnn_three_seed_mean", "test_n": len(test), "test_cell_lines": 120, "test_drugs": 135}
    for metric in ("rmse", "mae", "equal_drug_mean_rmse", "median_within_drug_spearman"):
        mean_row[metric] = float(comparison.loc[comparison["model"].str.startswith("gnn_seed_"), metric].mean())
        mean_row[f"{metric}_sample_sd"] = float(comparison.loc[comparison["model"].str.startswith("gnn_seed_"), metric].std(ddof=1))
    for count_name in ("spearman_eligible_drugs", "spearman_defined_drugs"):
        count_values = comparison.loc[comparison["model"].str.startswith("gnn_seed_"), count_name]
        if count_values.nunique() != 1:
            raise ValueError(f"GNN seeds disagree on {count_name}")
        mean_row[count_name] = int(count_values.iloc[0])
    comparison = pd.concat([comparison, pd.DataFrame([mean_row])], ignore_index=True)
    intervals, replicates = paired_bootstrap(result, TEMP / "final_bootstrap_replicates.npz")
    comparison_path = RESULTS / "gnn_test_comparison.csv"
    drug_path = RESULTS / "gnn_test_by_drug.csv"
    interval_path = RESULTS / "gnn_bootstrap_intervals.csv"
    prediction_path = TEMP / "gnn_test_predictions.csv"
    bootstrap_path = TEMP / "gnn_bootstrap_replicates.npz"
    comparison.to_csv(comparison_path, index=False, float_format="%.17g", lineterminator="\n")
    pd.concat(details, ignore_index=True).to_csv(drug_path, index=False, float_format="%.17g", lineterminator="\n")
    intervals.to_csv(interval_path, index=False, float_format="%.17g", lineterminator="\n")
    result.to_csv(prediction_path, index=False, float_format="%.17g", lineterminator="\n")
    np.savez_compressed(bootstrap_path, **replicates)
    artifacts = [comparison_path, drug_path, interval_path, prediction_path, bootstrap_path]
    final_manifest = {
        "scope": lock["scope"], "lock_sha256": sha256(LOCK_PATH),
        "validation_manifest_sha256": sha256(RESULTS / "gnn_validation_manifest.json"),
        "device": device.type, "runtime_seconds": perf_counter() - started,
        "test_rows": len(test), "test_cell_lines": 120, "test_drugs": 135,
        "artifact_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in artifacts},
        "checks": ["pretest_frozen_hashes", "baseline_row_and_target_alignment", "checkpoint_identity",
                   "known_drug_training_coverage", "finite_predictions", "paired_original_bootstrap_draws",
                   "original_ridge_reproduction", "no_seed_selection_or_prediction_ensemble"],
    }
    (RESULTS / "gnn_final_manifest.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(comparison[["model", "rmse", "mae", "equal_drug_mean_rmse", "median_within_drug_spearman"]].to_string(index=False))
    print(intervals[["model", "paired_rmse_delta_ci_low", "paired_rmse_delta_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
