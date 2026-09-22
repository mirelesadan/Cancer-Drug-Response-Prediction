r"""Fit fixed sensitivity models on the pre-existing mean-aggregated table.

Run after configs/final_evaluation.json has been frozen. This script uses only
training and validation labels; the test subset is reserved for final evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from baselines import (
    PerDrugMean, PooledRidge, PerDrugRidge, by_drug_validation,
    errors, fingerprint_row_basis, ridge_path,
)
from neural import DrugResponseMLP, assemble_features, predict_rows
from run_neural import configure_determinism, indexed_tensors, set_seed


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
TEMP = RESULTS / "tmp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_locked(config: dict) -> None:
    if config["state"] != "locked_before_test_outcomes":
        raise ValueError("Final protocol is not locked")
    for relative, expected in {**config["frozen_inputs_sha256"], **config["primary_model_artifacts_sha256"]}.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")


def metrics_by_subset(frame: pd.DataFrame, predictions: dict[str, np.ndarray], train_counts: dict[str, int], source: str) -> list[dict[str, object]]:
    output = []
    for subset_name, selected in (
        ("common_non_uprosertib", frame["drug_id"].ne("Uprosertib").to_numpy()),
        ("uprosertib", frame["drug_id"].eq("Uprosertib").to_numpy()),
    ):
        if not selected.any():
            continue
        subset = frame.loc[selected]
        y = subset["y"].to_numpy(dtype=np.float64)
        for model_name, all_pred in predictions.items():
            predicted = all_pred[selected]
            rmse, mae = errors(y, predicted)
            detail = by_drug_validation(subset["drug_id"].to_numpy(), y, predicted, train_counts, minimum_spearman_n=3)
            output.append({
                "source": source, "subset": subset_name, "model": model_name,
                "n_rows": len(subset), "n_drugs": len(detail), "rmse": rmse, "mae": mae,
                "equal_drug_mean_rmse": float(detail["rmse"].mean()),
                "median_defined_within_drug_spearman": float(detail["spearman"].median()) if detail["spearman"].notna().any() else np.nan,
                "spearman_defined_drugs": int(detail["spearman"].notna().sum()),
            })
    return output


def fit_sensitivity_baselines(
    train: pd.DataFrame, validation: pd.DataFrame, scores: np.ndarray,
    fingerprints: np.ndarray, n_drugs: int, alpha: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    train_y = train["y"].to_numpy(dtype=np.float64)
    train_cell = train["cell_index"].to_numpy(dtype=np.int32)
    train_drug = train["drug_index"].to_numpy(dtype=np.int32)
    val_cell = validation["cell_index"].to_numpy(dtype=np.int32)
    val_drug = validation["drug_index"].to_numpy(dtype=np.int32)
    counts = np.bincount(train_drug, minlength=n_drugs)
    if np.any(counts == 0):
        raise ValueError("Sensitivity training lacks a drug")
    trained = counts > 0
    means = np.bincount(train_drug, weights=train_y, minlength=n_drugs) / counts
    mean_model = PerDrugMean(means, trained)
    per_coef = np.zeros((n_drugs, 128), dtype=np.float64)
    per_intercept = np.zeros(n_drugs, dtype=np.float64)
    with threadpool_limits(limits=1):
        # One fixed alpha for every drug; no sensitivity validation tuning.
        for drug in range(n_drugs):
            use = train_drug == drug
            coefficient, intercept = ridge_path(scores[train_cell[use]], train_y[use], [alpha])
            per_coef[drug] = coefficient[0]
            per_intercept[drug] = intercept[0]
        per_model = PerDrugRidge(per_coef, per_intercept, alpha, trained)
        basis = fingerprint_row_basis(fingerprints[trained], 1e-12)
        projected = fingerprints.astype(np.float64) @ basis
        x_train = np.column_stack((scores[train_cell], projected[train_drug]))
        coefficient, intercept = ridge_path(x_train, train_y, [alpha])
        pooled_model = PooledRidge(
            coefficient[0, :128].copy(), (basis @ coefficient[0, 128:]).copy(),
            float(intercept[0]), alpha, trained,
        )
    models = {"per_drug_mean": mean_model, "pooled_ridge": pooled_model, "per_drug_ridge": per_model}
    predictions = {
        "per_drug_mean": mean_model.predict(val_drug),
        "pooled_ridge": pooled_model.predict(val_cell, val_drug, scores, fingerprints),
        "per_drug_ridge": per_model.predict(val_cell, val_drug, scores),
    }
    return models, predictions


def fit_fixed_epoch_mlp(
    train: pd.DataFrame, validation: pd.DataFrame, scores: np.ndarray,
    fingerprints: np.ndarray, neural_config: dict, seed: int, epochs: int,
    device: torch.device,
) -> tuple[np.ndarray, Path, float]:
    started = perf_counter()
    order_generator = set_seed(seed)
    cell_features = torch.as_tensor(np.ascontiguousarray(scores, dtype=np.float32), device=device)
    drug_features = torch.as_tensor(np.ascontiguousarray(fingerprints, dtype=np.float32), device=device)
    train_cell, train_drug, train_y = indexed_tensors(train, device)
    val_cell, val_drug, _ = indexed_tensors(validation, device)
    model = DrugResponseMLP(
        int(neural_config["input_dim"]), tuple(neural_config["hidden_layers"]),
        float(neural_config["dropout_after_each_hidden_activation"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0003, weight_decay=float(neural_config["weight_decay"]),
    )
    batch_size = int(neural_config["batch_size"])
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=order_generator).to(device)
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            x = assemble_features(train_cell[rows], train_drug[rows], cell_features, drug_features)
            target = train_y[rows]
            predicted = model(x)
            if predicted.shape != target.shape:
                raise ValueError("Sensitivity MLP target/feature row mismatch")
            loss = nn.functional.mse_loss(predicted, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite sensitivity loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    checkpoint = CHECKPOINTS / f"sensitivity_mlp_seed_{seed}.pt"
    torch.save({
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "seed": seed, "learning_rate": 0.0003, "epochs": epochs,
        "input_dim": int(neural_config["input_dim"]),
        "hidden_layers": list(neural_config["hidden_layers"]),
        "dropout": float(neural_config["dropout_after_each_hidden_activation"]),
    }, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved["seed"] != seed or saved["epochs"] != epochs:
        raise ValueError("Sensitivity checkpoint metadata mismatch")
    model.load_state_dict(saved["state_dict"])
    prediction = predict_rows(
        model, val_cell, val_drug, cell_features, drug_features,
        int(neural_config["evaluation_batch_size"]),
    ).numpy().astype(np.float64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return prediction, checkpoint, perf_counter() - started


def run() -> None:
    started = perf_counter()
    config = json.loads((ROOT / "configs" / "final_evaluation.json").read_text(encoding="utf-8"))
    verify_locked(config)
    neural_config = json.loads((ROOT / "configs" / "neural_mlp.json").read_text(encoding="utf-8"))
    configure_determinism(neural_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignments = pd.read_csv(OUT / "cell_line_assignments.csv")
    drugs = pd.read_csv(OUT / "drugs.csv")
    sensitivity = pd.read_csv(OUT / "sensitivity_responses.csv", dtype={"source_row_ids": str}, float_precision="round_trip")
    # Test rows remain unscored here; all training uses sensitivity train rows.
    use = sensitivity.loc[sensitivity["split"].isin(["train", "validation"])]
    if use["cell_line_id"].tolist() != assignments.loc[use["cell_index"], "cell_line_id"].tolist():
        raise ValueError("Sensitivity cell-line alignment failed")
    if use["drug_id"].tolist() != drugs.loc[use["drug_index"], "drug_id"].tolist():
        raise ValueError("Sensitivity drug alignment failed")
    if use["split"].tolist() != assignments.loc[use["cell_index"], "split"].tolist():
        raise ValueError("Sensitivity split alignment failed")
    train = use.loc[use["split"] == "train"].reset_index(drop=True)
    validation = use.loc[use["split"] == "validation"].reset_index(drop=True)
    if len(train) != 64506 or len(validation) != 13782:
        raise ValueError("Sensitivity training/validation counts changed")
    scores = np.load(OUT / "expression_pca128_scaled.npy", allow_pickle=False)
    fingerprints = np.load(OUT / "morgan_radius2_2048.npy", allow_pickle=False)
    CHECKPOINTS.mkdir(exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    alpha = float(config["primary_models"]["per_drug_ridge_alpha"])
    if alpha != 10 or float(config["primary_models"]["mlp_learning_rate"]) != 0.0003:
        raise ValueError("Sensitivity hyperparameters changed")
    models, estimates = fit_sensitivity_baselines(train, validation, scores, fingerprints, len(drugs), alpha)
    baseline_checkpoint = CHECKPOINTS / "sensitivity_baselines.joblib"
    joblib.dump(models, baseline_checkpoint, compress=0)

    runtime_by_seed = {}
    for seed_text, info in config["primary_models"]["mlp_selected_runs"].items():
        seed = int(seed_text)
        prediction, checkpoint, seconds = fit_fixed_epoch_mlp(
            train, validation, scores, fingerprints, neural_config, seed,
            int(info["best_epoch"]), device,
        )
        estimates[f"mlp_seed_{seed}"] = prediction
        runtime_by_seed[str(seed)] = {"seconds": seconds, "fixed_epochs": int(info["best_epoch"]), "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/")}
        print(f"Sensitivity MLP seed {seed}: fixed {info['best_epoch']} epochs, {seconds:.1f}s", flush=True)

    saved = validation[["pair_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "split", "y", "source_row_ids", "n_measurements", "y_min", "y_max", "y_range"]].copy()
    for name, predicted in estimates.items():
        if len(predicted) != len(validation) or not np.isfinite(predicted).all():
            raise ValueError(f"Incomplete sensitivity validation predictions: {name}")
        saved[name] = predicted
    prediction_path = TEMP / "sensitivity_validation_predictions.csv"
    saved.to_csv(prediction_path, index=False, float_format="%.17g", lineterminator="\n")

    # Non-Uprosertib pairs are exactly the same source observations in both
    # tables. Align on the source-row identity, never on accidental row order.
    common = saved.loc[saved["drug_id"].ne("Uprosertib")].copy()
    if not common["n_measurements"].eq(1).all():
        raise ValueError("Common sensitivity pair has multiple measurements")
    common["source_row_id"] = common["source_row_ids"].astype(np.int64)
    if not common["source_row_id"].is_unique:
        raise ValueError("Common source IDs are not unique")
    primary_baseline = pd.read_csv(TEMP / "baseline_validation_predictions.csv", float_precision="round_trip")
    primary_neural = pd.read_csv(TEMP / "neural_validation_predictions.csv", float_precision="round_trip")
    primary_predictions = primary_baseline.merge(primary_neural[["source_row_id"] + [info["run_id"] for info in config["primary_models"]["mlp_selected_runs"].values()]], on="source_row_id", validate="one_to_one")
    aligned = common.merge(primary_predictions, on="source_row_id", suffixes=("_sensitivity", "_primary"), validate="one_to_one")
    if len(aligned) != len(primary_baseline) or not np.array_equal(aligned["y_sensitivity"].to_numpy(), aligned["y_primary"].to_numpy()):
        raise ValueError("Primary/sensitivity common validation labels do not align")
    for name in ("per_drug_mean", "per_drug_ridge"):
        np.testing.assert_allclose(aligned[f"{name}_sensitivity"], aligned[f"{name}_primary"], rtol=0, atol=1e-8)

    train_counts = train.groupby("drug_id").size().to_dict()
    sensitivity_summary = metrics_by_subset(validation, estimates, train_counts, "sensitivity")
    primary_aligned = common[["drug_id", "cell_line_id", "y"]].copy()
    primary_columns = {
        "per_drug_mean": "per_drug_mean",
        "pooled_ridge": "pooled_ridge",
        "per_drug_ridge": "per_drug_ridge",
    }
    primary_columns.update({f"mlp_seed_{seed}": info["run_id"] for seed, info in config["primary_models"]["mlp_selected_runs"].items()})
    indexed = primary_predictions.set_index("source_row_id").loc[common["source_row_id"]]
    primary_estimates = {name: indexed[column].to_numpy(dtype=np.float64) for name, column in primary_columns.items()}
    primary_summary = metrics_by_subset(primary_aligned, primary_estimates, train_counts, "primary")
    summary_path = RESULTS / "sensitivity_validation_comparison.csv"
    pd.DataFrame(primary_summary + sensitivity_summary).to_csv(summary_path, index=False, float_format="%.17g", lineterminator="\n")

    paths = [baseline_checkpoint, prediction_path, summary_path]
    paths.extend(CHECKPOINTS / f"sensitivity_mlp_seed_{seed}.pt" for seed in (17, 29, 43))
    manifest = {
        "final_config_sha256": sha256(ROOT / "configs" / "final_evaluation.json"),
        "primary_model_hashes_unchanged": all(sha256(ROOT / name) == expected for name, expected in config["primary_model_artifacts_sha256"].items()),
        "device": device.type,
        "training_rows": len(train), "validation_rows": len(validation),
        "common_validation_rows": len(common), "uprosertib_validation_rows": int(validation["drug_id"].eq("Uprosertib").sum()),
        "selected_alpha": alpha, "selected_learning_rate": 0.0003,
        "mlp_fixed_epoch_runtime": runtime_by_seed,
        "runtime_seconds_total": perf_counter() - started,
        "artifact_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths},
        "checks": ["frozen_hashes", "training_only_fits", "common_pair_alignment", "independent_model_invariance", "fixed_mlp_epochs", "no_test_scoring"],
    }
    (RESULTS / "sensitivity_validation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Sensitivity validation complete: {len(common)} common rows and {manifest['uprosertib_validation_rows']} Uprosertib rows; independent common predictions unchanged", flush=True)


if __name__ == "__main__":
    run()
