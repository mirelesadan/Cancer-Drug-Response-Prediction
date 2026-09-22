r"""Evaluate locked primary and sensitivity models on the frozen test cells."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch

from baselines import by_drug_validation, errors
from neural import DrugResponseMLP, predict_rows
from run_sensitivity import sha256, verify_locked


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TEMP = RESULTS / "tmp"
CHECKPOINTS = RESULTS / "checkpoints"
NAMES = ["per_drug_mean", "pooled_ridge", "per_drug_ridge", "mlp_seed_17", "mlp_seed_29", "mlp_seed_43"]


def validate_rows(frame: pd.DataFrame, assignments: pd.DataFrame, drugs: pd.DataFrame) -> None:
    if not frame["cell_line_id"].tolist() == assignments.loc[frame["cell_index"], "cell_line_id"].tolist():
        raise ValueError("Cell-line index/label misalignment")
    if not frame["drug_id"].tolist() == drugs.loc[frame["drug_index"], "drug_id"].tolist():
        raise ValueError("Drug index/label misalignment")
    if not frame["split"].eq("test").all() or not assignments.loc[frame["cell_index"], "split"].eq("test").all():
        raise ValueError("Test split misalignment")
    if not np.isfinite(frame["y"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite test targets")


def predict_all(frame: pd.DataFrame, baseline_path: Path, mlp_paths: dict[int, Path],
                scores: np.ndarray, fingerprints: np.ndarray, neural_config: dict,
                device: torch.device, *, sensitivity: bool) -> pd.DataFrame:
    models = joblib.load(baseline_path)
    cell = frame["cell_index"].to_numpy(dtype=np.int32)
    drug = frame["drug_index"].to_numpy(dtype=np.int32)
    result = frame.copy()
    result["per_drug_mean"] = models["per_drug_mean"].predict(drug)
    result["pooled_ridge"] = models["pooled_ridge"].predict(cell, drug, scores, fingerprints)
    result["per_drug_ridge"] = models["per_drug_ridge"].predict(cell, drug, scores)
    cell_features = torch.as_tensor(scores, dtype=torch.float32, device=device)
    drug_features = torch.as_tensor(fingerprints, dtype=torch.float32, device=device)
    cell_index = torch.as_tensor(cell.astype(np.int64), device=device)
    drug_index = torch.as_tensor(drug.astype(np.int64), device=device)
    for seed, path in mlp_paths.items():
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["seed"] != seed or checkpoint["learning_rate"] != 0.0003:
            raise ValueError("MLP checkpoint identity changed")
        if sensitivity:
            if checkpoint["epochs"] != {17: 3, 29: 4, 43: 4}[seed]:
                raise ValueError("Sensitivity MLP epoch count changed")
        elif checkpoint["best_epoch"] != {17: 3, 29: 4, 43: 4}[seed]:
            raise ValueError("Primary MLP best epoch changed")
        model = DrugResponseMLP(
            int(neural_config["input_dim"]), tuple(neural_config["hidden_layers"]),
            float(neural_config["dropout_after_each_hidden_activation"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        result[f"mlp_seed_{seed}"] = predict_rows(
            model, cell_index, drug_index, cell_features, drug_features,
            int(neural_config["evaluation_batch_size"]),
        ).numpy().astype(np.float64)
    if not np.isfinite(result[NAMES].to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite test prediction")
    return result


def summarize(frame: pd.DataFrame, source: str, subset: str, train_counts: dict[str, int]) -> tuple[list[dict], pd.DataFrame]:
    y = frame["y"].to_numpy(dtype=np.float64)
    summary: list[dict] = []
    per_drug = []
    for name in NAMES:
        estimate = frame[name].to_numpy(dtype=np.float64)
        rmse, mae = errors(y, estimate)
        detail = by_drug_validation(frame["drug_id"].to_numpy(), y, estimate, train_counts, minimum_spearman_n=3)
        if sum(detail["validation_n"]) != len(frame):
            raise ValueError("Drug-level rows do not sum to evaluated set")
        detail = detail.rename(columns={"validation_n": "test_n"})
        detail.insert(0, "model", name)
        detail.insert(0, "subset", subset)
        detail.insert(0, "source", source)
        per_drug.append(detail)
        eligible = int((detail["test_n"] >= 3).sum())
        defined = int(detail["spearman"].notna().sum())
        summary.append({
            "source": source, "subset": subset, "model": name,
            "n_rows": len(frame), "n_drugs": len(detail), "rmse": rmse, "mae": mae,
            "equal_drug_mean_rmse": float(detail["rmse"].mean()),
            "median_within_drug_spearman": float(detail["spearman"].median()) if defined else np.nan,
            "spearman_eligible_drugs": eligible, "spearman_defined_drugs": defined,
        })
    return summary, pd.concat(per_drug, ignore_index=True)


def bootstrap_clusters(frame: pd.DataFrame, *, resamples: int, seed: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Reuse the same whole-cell draws for every model and every paired delta."""
    cell_ids, cell_index = np.unique(frame["cell_line_id"].to_numpy(), return_inverse=True)
    n_cells = len(cell_ids)
    if n_cells != 120:
        raise ValueError(f"Expected 120 test cell lines, got {n_cells}")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_cells, size=(resamples, n_cells), endpoint=False)
    draws = np.zeros((resamples, n_cells), dtype=np.int16)
    for i, one_draw in enumerate(sampled):
        draws[i] = np.bincount(one_draw, minlength=n_cells)
    if not np.all(draws.sum(axis=1) == n_cells):
        raise ValueError("Bootstrap draw omitted a cell-line cluster")
    row_counts = np.bincount(cell_index, minlength=n_cells)
    denominator = draws @ row_counts
    if not np.all(denominator > 0):
        raise ValueError("Empty bootstrap response draw")
    y = frame["y"].to_numpy(dtype=np.float64)
    replicate: dict[str, np.ndarray] = {}
    for name in NAMES:
        residual = frame[name].to_numpy(dtype=np.float64) - y
        cluster_sse = np.bincount(cell_index, weights=residual * residual, minlength=n_cells)
        cluster_sae = np.bincount(cell_index, weights=np.abs(residual), minlength=n_cells)
        replicate[f"{name}_rmse"] = np.sqrt((draws @ cluster_sse) / denominator)
        replicate[f"{name}_mae"] = (draws @ cluster_sae) / denominator
        if not np.isclose(np.sqrt(cluster_sse.sum() / len(frame)), errors(y, frame[name].to_numpy())[0], atol=1e-12):
            raise ValueError("Cluster error aggregation mismatch")
    replicate["mlp_three_seed_mean_rmse"] = np.mean([replicate[f"mlp_seed_{seed}_rmse"] for seed in (17, 29, 43)], axis=0)
    replicate["mlp_three_seed_mean_mae"] = np.mean([replicate[f"mlp_seed_{seed}_mae"] for seed in (17, 29, 43)], axis=0)
    rows = []
    for name in NAMES + ["mlp_three_seed_mean"]:
        rmse_draws = replicate[f"{name}_rmse"]
        mae_draws = replicate[f"{name}_mae"]
        reference = replicate["per_drug_ridge_rmse"]
        low_rmse, high_rmse = np.percentile(rmse_draws, [2.5, 97.5])
        low_mae, high_mae = np.percentile(mae_draws, [2.5, 97.5])
        delta_low, delta_high = np.percentile(rmse_draws - reference, [2.5, 97.5])
        rows.append({"model": name, "rmse_ci_low": low_rmse, "rmse_ci_high": high_rmse,
                     "mae_ci_low": low_mae, "mae_ci_high": high_mae,
                     "paired_rmse_delta_ci_low": delta_low, "paired_rmse_delta_ci_high": delta_high})
    replicate["draws"] = draws
    replicate["cell_ids"] = cell_ids.astype(str)
    replicate["row_counts_per_cell"] = row_counts
    return pd.DataFrame(rows), replicate


def plot_figures(comparison: pd.DataFrame, intervals: pd.DataFrame, by_drug: pd.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = RESULTS / "figures"
    folder.mkdir(exist_ok=True)
    out = []
    plot_names = ["per_drug_mean", "pooled_ridge", "per_drug_ridge", "mlp_three_seed_mean"]
    labels = ["Per-drug mean", "Pooled Ridge", "Per-drug Ridge", "MLP (3-seed mean)"]
    metric = comparison.set_index("model")
    interval = intervals.set_index("model")
    points = [metric.loc[n, "rmse"] for n in plot_names]
    lo = [interval.loc[n, "rmse_ci_low"] for n in plot_names]
    hi = [interval.loc[n, "rmse_ci_high"] for n in plot_names]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(plot_names))
    ax.errorbar(x, points, yerr=[np.subtract(points, lo), np.subtract(hi, points)], fmt="o", capsize=5, color="#184b68")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Test RMSE on provided Y scale")
    ax.set_title("Held-out cell lines; 95% cell-line bootstrap intervals")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = folder / "test_performance.png"; fig.savefig(path, dpi=180); plt.close(fig); out.append(path)

    details = by_drug.loc[by_drug["model"].isin(["per_drug_ridge", "pooled_ridge", "mlp_seed_17", "mlp_seed_29", "mlp_seed_43"])].copy()
    pivot = details.pivot(index="drug_id", columns="model", values="rmse")
    counts = details.loc[details["model"] == "per_drug_ridge"].set_index("drug_id")["test_n"]
    mlp_mean = pivot[[f"mlp_seed_{seed}" for seed in (17, 29, 43)]].mean(axis=1)
    fig, ax = plt.subplots(figsize=(6.7, 5.5))
    for eligible, color, label in [(True, "#176b82", "at least 10 test rows"), (False, "#ba6832", "fewer than 10 test rows")]:
        mask = (counts >= 10) if eligible else (counts < 10)
        ax.scatter(pivot.loc[mask, "per_drug_ridge"], mlp_mean.loc[mask], s=np.clip(counts.loc[mask], 12, 100), alpha=0.7, color=color, label=label)
    lim = max(float(pivot["per_drug_ridge"].max()), float(mlp_mean.max())) * 1.05
    ax.plot([0, lim], [0, lim], color="gray", linewidth=1, linestyle="--")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Per-drug Ridge test RMSE", ylabel="Mean of MLP seed-level drug RMSEs",
           title="Drug-level errors on the same held-out pairs")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout()
    path = folder / "drug_level_errors.png"; fig.savefig(path, dpi=180); plt.close(fig); out.append(path)

    curves = pd.read_csv(TEMP / "neural_learning_curves.csv")
    curves = curves.loc[np.isclose(curves["learning_rate"], 0.0003)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, seed in zip(axes, (17, 29, 43)):
        one = curves.loc[curves["seed"] == seed]
        ax.plot(one["epoch"], one["train_eval_rmse"], label="Train", color="#24759a")
        ax.plot(one["epoch"], one["validation_rmse"], label="Validation", color="#c36f25")
        ax.axvline({17: 3, 29: 4, 43: 4}[seed], color="#555", linestyle="--", linewidth=1, label="Best checkpoint")
        ax.set(xlabel="Epoch", title=f"Seed {seed}"); ax.grid(alpha=0.2)
    axes[0].set_ylabel("RMSE on provided Y scale")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Selected-rate MLP: training improves after validation peaks")
    fig.tight_layout()
    path = folder / "mlp_learning_curves.png"; fig.savefig(path, dpi=180); plt.close(fig); out.append(path)
    return out


def run() -> None:
    started = perf_counter()
    config_path = ROOT / "configs" / "final_evaluation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_locked(config)
    sensitivity_manifest = json.loads((RESULTS / "sensitivity_validation_manifest.json").read_text(encoding="utf-8"))
    if sensitivity_manifest["final_config_sha256"] != sha256(config_path):
        raise ValueError("Sensitivity run used a different frozen protocol")
    for relative, expected in sensitivity_manifest["artifact_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Sensitivity artifact changed: {relative}")
    neural_config = json.loads((ROOT / "configs" / "neural_mlp.json").read_text(encoding="utf-8"))
    assignments = pd.read_csv(PROCESSED / "cell_line_assignments.csv")
    drugs = pd.read_csv(PROCESSED / "drugs.csv")
    primary_all = pd.read_csv(PROCESSED / "primary_responses.csv", float_precision="round_trip")
    sensitivity_all = pd.read_csv(PROCESSED / "sensitivity_responses.csv", dtype={"source_row_ids": str}, float_precision="round_trip")
    primary = primary_all.loc[primary_all["split"] == "test"].reset_index(drop=True)
    sensitivity = sensitivity_all.loc[sensitivity_all["split"] == "test"].reset_index(drop=True)
    validate_rows(primary, assignments, drugs); validate_rows(sensitivity, assignments, drugs)
    if len(primary) != 13622 or len(sensitivity) != 13741 or primary["drug_id"].eq("Uprosertib").any():
        raise ValueError("Frozen test counts or exclusion changed")
    common = sensitivity.loc[sensitivity["drug_id"].ne("Uprosertib")].copy()
    common["source_row_id"] = common["source_row_ids"].astype(np.int64)
    if len(common) != len(primary) or not common["n_measurements"].eq(1).all():
        raise ValueError("Common test-pair composition changed")
    joined = common.merge(primary[["source_row_id", "drug_id", "cell_line_id", "y"]], on="source_row_id", suffixes=("_sensitivity", "_primary"), validate="one_to_one")
    if len(joined) != len(primary) or not joined["drug_id_sensitivity"].eq(joined["drug_id_primary"]).all() or not joined["cell_line_id_sensitivity"].eq(joined["cell_line_id_primary"]).all() or not np.array_equal(joined["y_sensitivity"].to_numpy(), joined["y_primary"].to_numpy()):
        raise ValueError("Common test labels and identities misaligned")
    scores = np.load(PROCESSED / "expression_pca128_scaled.npy", allow_pickle=False)
    fingerprints = np.load(PROCESSED / "morgan_radius2_2048.npy", allow_pickle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary_paths = {int(s): ROOT / f"results/checkpoints/mlp_{info['run_id']}.pt" for s, info in config["primary_models"]["mlp_selected_runs"].items()}
    sensitivity_paths = {s: CHECKPOINTS / f"sensitivity_mlp_seed_{s}.pt" for s in (17, 29, 43)}
    primary_pred = predict_all(primary, CHECKPOINTS / "baseline_models.joblib", primary_paths, scores, fingerprints, neural_config, device, sensitivity=False)
    sensitivity_pred = predict_all(sensitivity, CHECKPOINTS / "sensitivity_baselines.joblib", sensitivity_paths, scores, fingerprints, neural_config, device, sensitivity=True)
    primary_pred.to_csv(TEMP / "primary_test_predictions.csv", index=False, float_format="%.17g", lineterminator="\n")
    sensitivity_pred.to_csv(TEMP / "sensitivity_test_predictions.csv", index=False, float_format="%.17g", lineterminator="\n")
    common_pred = sensitivity_pred.loc[sensitivity_pred["drug_id"].ne("Uprosertib")].copy()
    common_pred["source_row_id"] = common_pred["source_row_ids"].astype(np.int64)
    aligned = common_pred.merge(primary_pred[["source_row_id"] + NAMES], on="source_row_id", suffixes=("_sensitivity", "_primary"), validate="one_to_one")
    if len(aligned) != len(primary_pred):
        raise ValueError("Test predictions do not align on common pairs")
    for name in ("per_drug_mean", "per_drug_ridge"):
        np.testing.assert_allclose(aligned[f"{name}_sensitivity"], aligned[f"{name}_primary"], rtol=0, atol=1e-8)
    primary_train_counts = primary_all.loc[primary_all["split"] == "train"].groupby("drug_id").size().to_dict()
    sensitivity_train_counts = sensitivity_all.loc[sensitivity_all["split"] == "train"].groupby("drug_id").size().to_dict()
    primary_summary, primary_drug = summarize(primary_pred, "primary", "common_non_uprosertib", primary_train_counts)
    sensitivity_common_summary, _ = summarize(common_pred, "sensitivity", "common_non_uprosertib", sensitivity_train_counts)
    sensitivity_up_summary, _ = summarize(sensitivity_pred.loc[sensitivity_pred["drug_id"].eq("Uprosertib")], "sensitivity", "uprosertib", sensitivity_train_counts)
    if len(sensitivity_pred.loc[sensitivity_pred["drug_id"].eq("Uprosertib")]) != 119:
        raise ValueError("Uprosertib test-pair count changed")
    comparison = pd.DataFrame(primary_summary)
    mlp_rows = comparison.loc[comparison["model"].str.startswith("mlp_seed_")]
    summary_row = {"source": "primary", "subset": "common_non_uprosertib", "model": "mlp_three_seed_mean", "n_rows": len(primary), "n_drugs": len(primary_drug["drug_id"].unique())}
    for key in ("rmse", "mae", "equal_drug_mean_rmse", "median_within_drug_spearman"):
        summary_row[key] = float(mlp_rows[key].mean())
        summary_row[key + "_sample_sd"] = float(mlp_rows[key].std(ddof=1))
    summary_row["spearman_eligible_drugs"] = int(mlp_rows["spearman_eligible_drugs"].iloc[0])
    summary_row["spearman_defined_drugs"] = int(mlp_rows["spearman_defined_drugs"].iloc[0])
    comparison = pd.concat([comparison, pd.DataFrame([summary_row])], ignore_index=True)
    comparison.to_csv(RESULTS / "final_test_comparison.csv", index=False, float_format="%.17g", lineterminator="\n")
    primary_drug.to_csv(RESULTS / "final_test_by_drug.csv", index=False, float_format="%.17g", lineterminator="\n")
    pd.DataFrame(primary_summary + sensitivity_common_summary + sensitivity_up_summary).to_csv(RESULTS / "final_sensitivity_comparison.csv", index=False, float_format="%.17g", lineterminator="\n")
    intervals, replicates = bootstrap_clusters(primary_pred, resamples=int(config["bootstrap"]["resamples"]), seed=int(config["bootstrap"]["seed"]))
    intervals.to_csv(RESULTS / "final_bootstrap_intervals.csv", index=False, float_format="%.17g", lineterminator="\n")
    np.savez_compressed(TEMP / "final_bootstrap_replicates.npz", **replicates)
    figure_paths = plot_figures(comparison, intervals, primary_drug)
    artifacts = [TEMP / "primary_test_predictions.csv", TEMP / "sensitivity_test_predictions.csv", TEMP / "final_bootstrap_replicates.npz",
                 RESULTS / "final_test_comparison.csv", RESULTS / "final_test_by_drug.csv", RESULTS / "final_sensitivity_comparison.csv", RESULTS / "final_bootstrap_intervals.csv"] + figure_paths
    manifest = {"final_config_sha256": sha256(config_path), "sensitivity_manifest_sha256": sha256(RESULTS / "sensitivity_validation_manifest.json"),
                "device": str(device), "primary_test_rows": len(primary), "test_cell_lines": primary["cell_line_id"].nunique(),
                "sensitivity_common_test_rows": len(common_pred), "sensitivity_uprosertib_test_pairs": 119,
                "bi_2536_test_rows": int(primary["drug_id"].eq("BI-2536").sum()), "runtime_seconds": perf_counter() - started,
                "artifact_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in artifacts},
                "checks": ["frozen_hashes", "test_identifier_alignment", "same_common_test_pairs", "checkpoint_identity", "independent_model_common_prediction_invariance", "finite_predictions", "per_drug_count_sum", "cell_cluster_bootstrap_multiplicity", "same_bootstrap_draws_for_all_models"]}
    (RESULTS / "final_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(comparison[["model", "rmse", "mae", "equal_drug_mean_rmse", "median_within_drug_spearman", "spearman_defined_drugs"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Final evaluation done in {manifest['runtime_seconds']:.1f} seconds; {len(primary)} primary test rows, {len(common_pred)} common sensitivity test rows")


if __name__ == "__main__":
    run()
