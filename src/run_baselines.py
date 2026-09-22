r"""Fit training-only GDSC2 baselines and choose Ridge alpha by validation RMSE.

From the repository root: .venv\Scripts\python.exe src\run_baselines.py
No test rows are scored or predicted in this script.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baselines import (
    PerDrugMean, PooledRidge, PerDrugRidge, by_drug_validation,
    errors, fingerprint_row_basis, ridge_path,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
TEMP = RESULTS / "tmp"
CONFIG_PATH = ROOT / "configs" / "baseline_ridge.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(
    rows: pd.DataFrame, assignments: pd.DataFrame, drugs: pd.DataFrame,
    scores: np.ndarray, fingerprints: np.ndarray, manifest: dict,
) -> None:
    for name in ("primary_responses.csv", "cell_line_assignments.csv", "drugs.csv", "expression_pca128.npy", "morgan_radius2_2048.npy"):
        if sha256(PROCESSED / name) != manifest["artifacts_sha256"][name]:
            raise ValueError(f"Preprocessing artifact changed: {name}")
    if rows["source_row_id"].duplicated().any() or (rows["drug_id"] == "Uprosertib").any():
        raise ValueError("Primary source-row identity or exclusion failed")
    if list(assignments["cell_line_id"]) != sorted(assignments["cell_line_id"]) or not assignments["cell_line_id"].is_unique:
        raise ValueError("Cell-line catalog order failed")
    if scores.shape != (len(assignments), 128) or fingerprints.shape != (len(drugs), 2048):
        raise ValueError("Input feature shapes changed")
    if not np.isfinite(scores).all() or not np.isin(fingerprints, [0, 1]).all():
        raise ValueError("Nonfinite expression or nonbinary fingerprints")
    if rows["cell_line_id"].tolist() != assignments.loc[rows["cell_index"], "cell_line_id"].tolist():
        raise ValueError("Cell feature/label index misalignment")
    if rows["drug_id"].tolist() != drugs.loc[rows["drug_index"], "drug_id"].tolist():
        raise ValueError("Drug feature/label index misalignment")
    if rows["split"].tolist() != assignments.loc[rows["cell_index"], "split"].tolist():
        raise ValueError("Frozen split mapping changed")
    if not np.isfinite(rows["y"].to_numpy()).all():
        raise ValueError("Nonfinite target")


def fit_and_validate() -> dict[str, object]:
    started = perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    prep_manifest = json.loads((RESULTS / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    alpha_grid = [float(value) for value in config["alpha_grid"]]
    if alpha_grid != [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        raise ValueError("Unexpected alpha grid")
    assignments = pd.read_csv(PROCESSED / "cell_line_assignments.csv")
    drugs = pd.read_csv(PROCESSED / "drugs.csv")
    rows = pd.read_csv(PROCESSED / "primary_responses.csv", float_precision="round_trip")
    pca_scores = np.load(PROCESSED / "expression_pca128.npy", allow_pickle=False)
    fingerprints = np.load(PROCESSED / "morgan_radius2_2048.npy", allow_pickle=False)
    validate_inputs(rows, assignments, drugs, pca_scores, fingerprints, prep_manifest)
    train = rows.loc[rows["split"] == "train"].reset_index(drop=True)
    validation = rows.loc[rows["split"] == "validation"].reset_index(drop=True)
    if len(train) != prep_manifest["counts"]["primary_rows_by_split"]["train"] or len(validation) != prep_manifest["counts"]["primary_rows_by_split"]["validation"]:
        raise ValueError("Frozen train/validation counts changed")
    y_train = train["y"].to_numpy(dtype=np.float64)
    y_validation = validation["y"].to_numpy(dtype=np.float64)
    train_cell = train["cell_index"].to_numpy(dtype=np.int32)
    val_cell = validation["cell_index"].to_numpy(dtype=np.int32)
    train_drug = train["drug_index"].to_numpy(dtype=np.int32)
    val_drug = validation["drug_index"].to_numpy(dtype=np.int32)
    trained_drugs = np.bincount(train_drug, minlength=len(drugs)) > 0
    if not trained_drugs[val_drug].all():
        raise ValueError("Validation contains drug absent from training")

    # One PCA-score vector per training cell, not one per response row.
    scaling_started = perf_counter()
    train_cell_positions = assignments.index[assignments["split"] == "train"].to_numpy(dtype=np.int32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(pca_scores[train_cell_positions])
    scaled_scores = scaler.transform(pca_scores)
    if not np.isfinite(scaled_scores).all():
        raise ValueError("Nonfinite standardized PCA score")
    np.testing.assert_allclose(scaler.mean_, pca_scores[train_cell_positions].mean(axis=0), rtol=1e-13, atol=1e-13)
    scaling_seconds = perf_counter() - scaling_started

    mean_started = perf_counter()
    counts = np.bincount(train_drug, minlength=len(drugs))
    sums = np.bincount(train_drug, weights=y_train, minlength=len(drugs))
    means = np.full(len(drugs), np.nan, dtype=np.float64)
    means[trained_drugs] = sums[trained_drugs] / counts[trained_drugs]
    mean_model = PerDrugMean(means=means, trained_drugs=trained_drugs)
    mean_train = mean_model.predict(train_drug)
    mean_val = mean_model.predict(val_drug)
    train_counts = dict(zip(drugs["drug_id"], counts))
    mean_seconds = perf_counter() - mean_started

    # Project fingerprints onto their training-drug row space. This is an exact
    # orthonormal reparameterization of the additive linear fingerprint term:
    # it preserves both predictions and the Ridge coefficient L2 penalty.
    pooled_started = perf_counter()
    with threadpool_limits(limits=1):
        basis = fingerprint_row_basis(
            fingerprints[np.flatnonzero(trained_drugs)],
            float(config["fingerprint_basis_relative_tolerance"]),
        )
        drug_scores = fingerprints.astype(np.float64) @ basis
        x_train_pooled = np.column_stack((scaled_scores[train_cell], drug_scores[train_drug]))
        x_val_pooled = np.column_stack((scaled_scores[val_cell], drug_scores[val_drug]))
        pooled_coefficients, pooled_intercepts = ridge_path(x_train_pooled, y_train, alpha_grid)
        pooled_val_grid = np.stack([
            x_val_pooled @ pooled_coefficients[index] + pooled_intercepts[index]
            for index in range(len(alpha_grid))
        ])
        pooled_search = [errors(y_validation, prediction) for prediction in pooled_val_grid]
        pooled_selected = int(np.argmin([item[0] for item in pooled_search]))
        pooled_alpha = alpha_grid[pooled_selected]
        selected_coef = pooled_coefficients[pooled_selected]
        pooled_model = PooledRidge(
            expression_coef=selected_coef[:128].copy(),
            fingerprint_coef=(basis @ selected_coef[128:]).copy(),
            intercept=float(pooled_intercepts[pooled_selected]),
            alpha=pooled_alpha,
            trained_drugs=trained_drugs.copy(),
        )
        pooled_train = pooled_model.predict(train_cell, train_drug, scaled_scores, fingerprints)
        pooled_val = pooled_model.predict(val_cell, val_drug, scaled_scores, fingerprints)
    np.testing.assert_allclose(pooled_val, pooled_val_grid[pooled_selected], rtol=0, atol=1e-8)
    pooled_seconds = perf_counter() - pooled_started
    del x_train_pooled, x_val_pooled, pooled_val_grid

    # All drugs share one selected alpha. Fit each drug's coefficients from its
    # training response rows; only validation labels select the shared alpha.
    separate_started = perf_counter()
    perdrug_val_grid = np.empty((len(alpha_grid), len(validation)), dtype=np.float64)
    perdrug_coef_grid = np.zeros((len(alpha_grid), len(drugs), 128), dtype=np.float64)
    perdrug_intercept_grid = np.full((len(alpha_grid), len(drugs)), np.nan, dtype=np.float64)
    fallback_drugs: list[str] = []
    constant_train_target_drugs: list[str] = []
    with threadpool_limits(limits=1):
        for drug_index in np.flatnonzero(trained_drugs):
            train_positions = np.flatnonzero(train_drug == drug_index)
            val_positions = np.flatnonzero(val_drug == drug_index)
            subset_y = y_train[train_positions]
            if np.unique(subset_y).size < 2:
                constant_train_target_drugs.append(str(drugs.loc[drug_index, "drug_id"]))
            if len(train_positions) < 2:
                fallback_drugs.append(str(drugs.loc[drug_index, "drug_id"]))
                perdrug_intercept_grid[:, drug_index] = float(subset_y.mean())
            else:
                coefficients, intercepts = ridge_path(
                    scaled_scores[train_cell[train_positions]], subset_y, alpha_grid,
                )
                perdrug_coef_grid[:, drug_index] = coefficients
                perdrug_intercept_grid[:, drug_index] = intercepts
            if len(val_positions):
                x_val_drug = scaled_scores[val_cell[val_positions]]
                perdrug_val_grid[:, val_positions] = (
                    perdrug_coef_grid[:, drug_index] @ x_val_drug.T
                    + perdrug_intercept_grid[:, drug_index, None]
                )
    if not np.isfinite(perdrug_val_grid).all():
        raise ValueError("Incomplete per-drug validation predictions")
    perdrug_search = [errors(y_validation, prediction) for prediction in perdrug_val_grid]
    perdrug_selected = int(np.argmin([item[0] for item in perdrug_search]))
    perdrug_alpha = alpha_grid[perdrug_selected]
    perdrug_model = PerDrugRidge(
        expression_coef=perdrug_coef_grid[perdrug_selected].copy(),
        intercept=perdrug_intercept_grid[perdrug_selected].copy(),
        alpha=perdrug_alpha,
        trained_drugs=trained_drugs.copy(),
    )
    perdrug_train = perdrug_model.predict(train_cell, train_drug, scaled_scores)
    perdrug_val = perdrug_model.predict(val_cell, val_drug, scaled_scores)
    np.testing.assert_allclose(perdrug_val, perdrug_val_grid[perdrug_selected], rtol=0, atol=1e-9)
    separate_seconds = perf_counter() - separate_started

    # Response-weighted overall metrics and equal-drug-weight macro metrics
    # are separate summaries of the same validation predictions.
    evaluation_started = perf_counter()
    candidates = [
        ("per_drug_mean", None, mean_train, mean_val),
        ("pooled_ridge", pooled_alpha, pooled_train, pooled_val),
        ("per_drug_ridge", perdrug_alpha, perdrug_train, perdrug_val),
    ]
    comparison_rows = []
    drug_tables = []
    for model_name, alpha, train_pred, val_pred in candidates:
        train_rmse, train_mae = errors(y_train, train_pred)
        val_rmse, val_mae = errors(y_validation, val_pred)
        by_drug = by_drug_validation(
            validation["drug_id"].to_numpy(), y_validation, val_pred,
            train_counts, minimum_spearman_n=int(config["minimum_within_drug_spearman_n"]),
        )
        by_drug.insert(0, "model", model_name)
        drug_tables.append(by_drug)
        eligible = by_drug["spearman_status"].isin(["defined", "constant_prediction", "undefined_numeric"])
        comparison_rows.append({
            "model": model_name, "selected_alpha": alpha,
            "train_n": len(train), "validation_n": len(validation),
            "train_rmse": train_rmse, "train_mae": train_mae,
            "validation_rmse": val_rmse, "validation_mae": val_mae,
            "validation_macro_drug_rmse": float(by_drug["rmse"].mean()),
            "validation_macro_drug_mae": float(by_drug["mae"].mean()),
            "validation_median_defined_drug_spearman": float(by_drug["spearman"].median()) if by_drug["spearman"].notna().any() else np.nan,
            "validation_drugs": len(by_drug),
            "spearman_eligible_drugs": int(eligible.sum()),
            "spearman_defined_drugs": int(by_drug["spearman"].notna().sum()),
        })
    comparison = pd.DataFrame(comparison_rows)
    by_drug_all = pd.concat(drug_tables, ignore_index=True)
    alpha_rows = []
    for model_name, search in (("pooled_ridge", pooled_search), ("per_drug_ridge", perdrug_search)):
        for alpha, (rmse, mae) in zip(alpha_grid, search):
            alpha_rows.append({"model": model_name, "alpha": alpha, "validation_rmse": rmse, "validation_mae": mae})
    alpha_table = pd.DataFrame(alpha_rows)
    predictions = validation[["source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "split", "y"]].copy()
    predictions["per_drug_mean"] = mean_val
    predictions["pooled_ridge"] = pooled_val
    predictions["per_drug_ridge"] = perdrug_val
    evaluation_seconds = perf_counter() - evaluation_started

    CHECKPOINTS.mkdir(exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    scaler_path = PROCESSED / "pca_score_scaler.joblib"
    scaled_path = PROCESSED / "expression_pca128_scaled.npy"
    models_path = CHECKPOINTS / "baseline_models.joblib"
    prediction_path = TEMP / "baseline_validation_predictions.csv"
    joblib.dump(scaler, scaler_path, compress=0)
    np.save(scaled_path, scaled_scores, allow_pickle=False)
    joblib.dump({"per_drug_mean": mean_model, "pooled_ridge": pooled_model, "per_drug_ridge": perdrug_model}, models_path, compress=0)
    predictions.to_csv(prediction_path, index=False, float_format="%.17g", lineterminator="\n")
    comparison.to_csv(RESULTS / "baseline_comparison.csv", index=False, float_format="%.17g", lineterminator="\n")
    by_drug_all.to_csv(RESULTS / "baseline_validation_by_drug.csv", index=False, float_format="%.17g", lineterminator="\n")
    alpha_table.to_csv(RESULTS / "baseline_alpha_search.csv", index=False, float_format="%.17g", lineterminator="\n")

    # Replay predictions only on validation rows after loading persisted models.
    loaded_scaler = joblib.load(scaler_path)
    loaded_scores = loaded_scaler.transform(pca_scores)
    np.testing.assert_allclose(loaded_scores, np.load(scaled_path, allow_pickle=False), rtol=0, atol=0)
    loaded = joblib.load(models_path)
    np.testing.assert_allclose(loaded["per_drug_mean"].predict(val_drug), mean_val, rtol=0, atol=0)
    np.testing.assert_allclose(loaded["pooled_ridge"].predict(val_cell, val_drug, loaded_scores, fingerprints), pooled_val, rtol=0, atol=1e-10)
    np.testing.assert_allclose(loaded["per_drug_ridge"].predict(val_cell, val_drug, loaded_scores), perdrug_val, rtol=0, atol=1e-10)
    if not np.array_equal(predictions["y"].to_numpy(), y_validation):
        raise ValueError("Validation prediction/label alignment failed")

    artifact_paths = [scaler_path, scaled_path, models_path, prediction_path,
                      RESULTS / "baseline_comparison.csv", RESULTS / "baseline_validation_by_drug.csv",
                      RESULTS / "baseline_alpha_search.csv"]
    summary: dict[str, object] = {
        "config_sha256": sha256(CONFIG_PATH),
        "preprocessing_manifest_sha256": sha256(RESULTS / "preprocessing_manifest.json"),
        "input_primary_sha256": prep_manifest["artifacts_sha256"]["primary_responses.csv"],
        "alpha_grid": alpha_grid,
        "selected_alpha": {"pooled_ridge": pooled_alpha, "per_drug_ridge": perdrug_alpha},
        "training_counts": {
            "drugs": int(trained_drugs.sum()), "minimum": int(counts[trained_drugs].min()),
            "median": float(np.median(counts[trained_drugs])), "maximum": int(counts[trained_drugs].max()),
            "fallback_drugs": fallback_drugs, "constant_target_drugs": constant_train_target_drugs,
        },
        "pooled_fingerprint_basis_rank": int(basis.shape[1]),
        "scores_fit_unique_training_cells": len(train_cell_positions),
        "runtime_seconds": {
            "score_scaling": scaling_seconds, "per_drug_mean": mean_seconds,
            "pooled_ridge_search": pooled_seconds, "per_drug_ridge_search": separate_seconds,
            "metrics_and_tables": evaluation_seconds, "total": perf_counter() - started,
        },
        "runtime_context": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "cpu_count": os.cpu_count(), "blas_threads_during_ridge": 1,
        },
        "artifact_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in artifact_paths},
        "checks": [
            "frozen_preprocessing_hashes", "cell_and_drug_alignment", "training_drug_coverage",
            "score_scaler_fit_on_unique_training_cells", "positive_alpha_cholesky",
            "fingerprint_basis_exact_on_training_drugs", "validation_replay_after_model_reload",
            "no_test_prediction_or_metric",
        ],
    }
    (RESULTS / "baseline_validation_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(json.dumps({"selected_alpha": summary["selected_alpha"], "runtime_seconds": summary["runtime_seconds"]}, indent=2))
    return summary


if __name__ == "__main__":
    fit_and_validate()
