r"""Train the six fixed MLP runs using only frozen training and validation rows.

From the repository root:
    .venv\Scripts\python.exe src\run_neural.py --smoke-only
    .venv\Scripts\python.exe src\run_neural.py

No test prediction, metric, or sensitivity analysis is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from time import perf_counter

# cuBLAS requires this before CUDA is initialized for deterministic matrix ops.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from torch import nn

from baselines import by_drug_validation, errors
from neural import DrugResponseMLP, assemble_features, predict_rows


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
TEMP = RESULTS / "tmp"
CONFIG_PATH = ROOT / "configs" / "neural_mlp.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(seed)
    return order_generator


def configure_determinism(config: dict) -> None:
    settings = config["determinism"]
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != settings["cublas_workspace_config"]:
        raise ValueError("Unexpected cuBLAS workspace setting")
    torch.use_deterministic_algorithms(bool(settings["deterministic_algorithms"]))
    torch.backends.cudnn.benchmark = bool(settings["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(settings["cudnn_deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = bool(settings["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(settings["allow_tf32"])
    torch.set_float32_matmul_precision("highest")


def load_primary(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    baseline_manifest = json.loads((RESULTS / "baseline_validation_manifest.json").read_text(encoding="utf-8"))
    prep_manifest = json.loads((RESULTS / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    required = {
        OUT / "primary_responses.csv": baseline_manifest["input_primary_sha256"],
        OUT / "cell_line_assignments.csv": prep_manifest["artifacts_sha256"]["cell_line_assignments.csv"],
        OUT / "drugs.csv": prep_manifest["artifacts_sha256"]["drugs.csv"],
        OUT / "expression_pca128_scaled.npy": baseline_manifest["artifact_sha256"]["data/processed/expression_pca128_scaled.npy"],
        OUT / "morgan_radius2_2048.npy": prep_manifest["artifacts_sha256"]["morgan_radius2_2048.npy"],
        OUT / "pca_score_scaler.joblib": baseline_manifest["artifact_sha256"]["data/processed/pca_score_scaler.joblib"],
    }
    for path, expected in required.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path.name}")
    if config["input_table"] != "data/processed/primary_responses.csv":
        raise ValueError("Neural input must be the frozen primary table")
    assignments = pd.read_csv(OUT / "cell_line_assignments.csv")
    drugs = pd.read_csv(OUT / "drugs.csv")
    rows = pd.read_csv(OUT / "primary_responses.csv", float_precision="round_trip")
    scores = np.load(OUT / "expression_pca128_scaled.npy", allow_pickle=False)
    fingerprints = np.load(OUT / "morgan_radius2_2048.npy", allow_pickle=False)
    if scores.shape != (len(assignments), 128) or fingerprints.shape != (len(drugs), 2048):
        raise ValueError("Frozen feature shapes changed")
    if not np.isfinite(scores).all() or not np.isin(fingerprints, [0, 1]).all():
        raise ValueError("Nonfinite or nonbinary inputs")
    if (rows["drug_id"] == "Uprosertib").any():
        raise ValueError("Uprosertib exclusion changed")
    # Check only training and validation response rows; test Y is never scored.
    use = rows.loc[rows["split"].isin(["train", "validation"])]
    if use["cell_line_id"].tolist() != assignments.loc[use["cell_index"], "cell_line_id"].tolist():
        raise ValueError("Cell feature/label alignment failed")
    if use["drug_id"].tolist() != drugs.loc[use["drug_index"], "drug_id"].tolist():
        raise ValueError("Drug feature/label alignment failed")
    if use["split"].tolist() != assignments.loc[use["cell_index"], "split"].tolist():
        raise ValueError("Frozen split alignment failed")
    train = use.loc[use["split"] == "train"].reset_index(drop=True)
    validation = use.loc[use["split"] == "validation"].reset_index(drop=True)
    if len(train) != 63946 or len(validation) != 13661:
        raise ValueError("Frozen train/validation row counts changed")
    if not np.isfinite(train["y"].to_numpy()).all() or not np.isfinite(validation["y"].to_numpy()).all():
        raise ValueError("Nonfinite training/validation target")
    return train, validation, drugs, scores, fingerprints, baseline_manifest


def indexed_tensors(frame: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cells = torch.as_tensor(frame["cell_index"].to_numpy(dtype=np.int64), device=device)
    drugs = torch.as_tensor(frame["drug_index"].to_numpy(dtype=np.int64), device=device)
    targets = torch.as_tensor(frame["y"].to_numpy(dtype=np.float32), device=device)
    if cells.shape != drugs.shape or cells.shape != targets.shape:
        raise ValueError("Prediction/target row shape mismatch")
    return cells, drugs, targets


def smoke_check(
    config: dict, device: torch.device, cell_features: torch.Tensor, drug_features: torch.Tensor,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """One small training batch, device and shape check; no study metric."""
    generator = set_seed(int(config["seeds"][0]))
    del generator
    model = DrugResponseMLP(
        int(config["input_dim"]), tuple(config["hidden_layers"]),
        float(config["dropout_after_each_hidden_activation"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rates"][0]), weight_decay=float(config["weight_decay"]))
    cells, drugs, targets = (tensor[:32] for tensor in train_tensors)
    features = assemble_features(cells, drugs, cell_features, drug_features)
    if features.shape != (32, 2176) or features.device.type != device.type:
        raise ValueError("Smoke batch feature shape or device mismatch")
    model.train()
    predicted = model(features)
    if predicted.shape != targets.shape:
        raise ValueError("Smoke prediction/target shape mismatch")
    loss = nn.functional.mse_loss(predicted, targets)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite smoke loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    repeat = predict_rows(model, cells, drugs, cell_features, drug_features, 32)
    if repeat.shape != (32,) or not torch.isfinite(repeat).all():
        raise ValueError("Nonfinite smoke inference")
    print(f"SMOKE PASS ({device.type}): 32 aligned rows, finite step, eval-mode output shape (32,)", flush=True)


def train_one(
    config: dict, learning_rate: float, seed: int, device: torch.device,
    cell_features: torch.Tensor, drug_features: torch.Tensor,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    y_train: np.ndarray, y_validation: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray, Path]:
    started = perf_counter()
    order_generator = set_seed(seed)
    model = DrugResponseMLP(
        int(config["input_dim"]), tuple(config["hidden_layers"]),
        float(config["dropout_after_each_hidden_activation"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(config["weight_decay"]))
    train_cell, train_drug, train_y = train_tensors
    val_cell, val_drug, _ = val_tensors
    batch_size = int(config["batch_size"])
    eval_batch_size = int(config["evaluation_batch_size"])
    patience = int(config["early_stopping_patience"])
    max_epochs = int(config["maximum_epochs"])
    run_id = f"lr_{learning_rate:.0e}_seed_{seed}"
    checkpoint = CHECKPOINTS / f"mlp_{run_id}.pt"
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_validation_prediction: np.ndarray | None = None
    curves = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=order_generator).to(device)
        weighted_loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch_ids = order[start:start + batch_size]
            features = assemble_features(train_cell[batch_ids], train_drug[batch_ids], cell_features, drug_features)
            target = train_y[batch_ids]
            predicted = model(features)
            if predicted.shape != target.shape:
                raise ValueError(f"Misaligned batch target in {run_id}")
            loss = nn.functional.mse_loss(predicted, target)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss in {run_id} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weighted_loss_sum += float(loss.detach().item()) * len(batch_ids)
        train_prediction = predict_rows(model, train_cell, train_drug, cell_features, drug_features, eval_batch_size).numpy().astype(np.float64)
        val_prediction = predict_rows(model, val_cell, val_drug, cell_features, drug_features, eval_batch_size).numpy().astype(np.float64)
        train_rmse, train_mae = errors(y_train, train_prediction)
        val_rmse, val_mae = errors(y_validation, val_prediction)
        if not np.isfinite([train_rmse, train_mae, val_rmse, val_mae]).all():
            raise ValueError(f"Nonfinite evaluation metric in {run_id}")
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            best_validation_prediction = val_prediction.copy()
            torch.save({
                "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "seed": seed, "learning_rate": learning_rate, "best_epoch": epoch,
                "validation_rmse": best_rmse, "input_dim": int(config["input_dim"]),
                "hidden_layers": list(config["hidden_layers"]),
                "dropout": float(config["dropout_after_each_hidden_activation"]),
            }, checkpoint)
        else:
            epochs_without_improvement += 1
        curves.append({
            "run_id": run_id, "learning_rate": learning_rate, "seed": seed, "epoch": epoch,
            "batch_training_mse": weighted_loss_sum / len(train_y),
            "train_eval_rmse": train_rmse, "train_eval_mae": train_mae,
            "validation_rmse": val_rmse, "validation_mae": val_mae,
            "best_epoch_so_far": best_epoch, "epochs_without_improvement": epochs_without_improvement,
        })
        if epoch % 5 == 0 or epoch == 1:
            print(f"{run_id} epoch {epoch}: train {train_rmse:.4f}, val {val_rmse:.4f}, best {best_rmse:.4f} at {best_epoch}", flush=True)
        if epochs_without_improvement >= patience:
            break
    if best_validation_prediction is None:
        raise ValueError("No finite best checkpoint")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved["seed"] != seed or saved["best_epoch"] != best_epoch or saved["learning_rate"] != learning_rate:
        raise ValueError("Checkpoint metadata mismatch")
    model.load_state_dict(saved["state_dict"])
    restored_train = predict_rows(model, train_cell, train_drug, cell_features, drug_features, eval_batch_size).numpy().astype(np.float64)
    restored_val = predict_rows(model, val_cell, val_drug, cell_features, drug_features, eval_batch_size).numpy().astype(np.float64)
    np.testing.assert_allclose(restored_val, best_validation_prediction, rtol=0, atol=1e-6)
    train_rmse, train_mae = errors(y_train, restored_train)
    val_rmse, val_mae = errors(y_validation, restored_val)
    if abs(val_rmse - best_rmse) > 1e-6:
        raise ValueError("Reloaded checkpoint metric mismatch")
    if device.type == "cuda":
        torch.cuda.synchronize()
    run_result: dict[str, object] = {
        "run_id": run_id, "learning_rate": learning_rate, "seed": seed,
        "best_epoch": best_epoch, "last_epoch": epoch,
        "early_stopped": bool(epochs_without_improvement >= patience),
        "hit_epoch_cap": bool(epoch == max_epochs),
        "train_rmse": train_rmse, "train_mae": train_mae,
        "validation_rmse": val_rmse, "validation_mae": val_mae,
        "runtime_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
    }
    print(f"DONE {run_id}: best epoch {best_epoch}, val RMSE {val_rmse:.4f}, seconds {run_result['runtime_seconds']:.1f}", flush=True)
    return run_result, curves, restored_train, restored_val, checkpoint


def run(*, smoke_only: bool = False) -> None:
    started = perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["learning_rates"] != [0.001, 0.0003] or config["seeds"] != [17, 29, 43]:
        raise ValueError("Fixed six-run experiment configuration changed")
    configure_determinism(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, drugs, scores, fingerprints, baseline_manifest = load_primary(config)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        gpu_memory_bytes = properties.total_memory
    else:
        gpu_name, gpu_memory_bytes = None, None
    cell_features = torch.as_tensor(np.ascontiguousarray(scores, dtype=np.float32), device=device)
    drug_features = torch.as_tensor(np.ascontiguousarray(fingerprints, dtype=np.float32), device=device)
    train_tensors = indexed_tensors(train, device)
    val_tensors = indexed_tensors(validation, device)
    y_train = train["y"].to_numpy(dtype=np.float64)
    y_validation = validation["y"].to_numpy(dtype=np.float64)
    smoke_check(config, device, cell_features, drug_features, train_tensors)
    if smoke_only:
        return

    CHECKPOINTS.mkdir(exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    run_rows = []
    curve_rows = []
    by_drug_rows = []
    predictions = validation[["source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "split", "y"]].copy()
    train_counts = train.groupby("drug_id").size().to_dict()
    for learning_rate in config["learning_rates"]:
        for seed in config["seeds"]:
            result, curves, _, val_prediction, _ = train_one(
                config, float(learning_rate), int(seed), device,
                cell_features, drug_features, train_tensors, val_tensors,
                y_train, y_validation,
            )
            by_drug = by_drug_validation(
                validation["drug_id"].to_numpy(), y_validation, val_prediction,
                train_counts, minimum_spearman_n=int(config["minimum_within_drug_spearman_n"]),
            )
            by_drug.insert(0, "run_id", result["run_id"])
            by_drug_rows.append(by_drug)
            result["validation_macro_drug_rmse"] = float(by_drug["rmse"].mean())
            result["validation_macro_drug_mae"] = float(by_drug["mae"].mean())
            result["validation_median_defined_drug_spearman"] = float(by_drug["spearman"].median()) if by_drug["spearman"].notna().any() else np.nan
            result["spearman_eligible_drugs"] = int(by_drug["spearman_status"].isin(["defined", "constant_prediction", "undefined_numeric"]).sum())
            result["spearman_defined_drugs"] = int(by_drug["spearman"].notna().sum())
            predictions[result["run_id"]] = val_prediction
            curve_rows.extend(curves)
            run_rows.append(result)
    runs = pd.DataFrame(run_rows)
    if len(runs) != 6:
        raise ValueError("Expected exactly six runs")
    summary_rows = []
    for learning_rate, group in runs.groupby("learning_rate", sort=False):
        if sorted(group["seed"].tolist()) != [17, 29, 43]:
            raise ValueError("Incomplete seeds for learning rate")
        summary_rows.append({
            "learning_rate": learning_rate, "runs": len(group),
            "validation_rmse_mean": float(group["validation_rmse"].mean()),
            "validation_rmse_sample_sd": float(group["validation_rmse"].std(ddof=1)),
            "validation_mae_mean": float(group["validation_mae"].mean()),
            "validation_mae_sample_sd": float(group["validation_mae"].std(ddof=1)),
            "median_within_drug_spearman_mean": float(group["validation_median_defined_drug_spearman"].mean()),
            "median_within_drug_spearman_sample_sd": float(group["validation_median_defined_drug_spearman"].std(ddof=1)),
            "best_epoch_min": int(group["best_epoch"].min()),
            "best_epoch_max": int(group["best_epoch"].max()),
            "runs_hitting_epoch_cap": int(group["hit_epoch_cap"].sum()),
            "runtime_seconds_sum": float(group["runtime_seconds"].sum()),
        })
    summary = pd.DataFrame(summary_rows)
    selected_index = int(summary["validation_rmse_mean"].argmin())
    selected_lr = float(summary.iloc[selected_index]["learning_rate"])
    runs["selected_learning_rate"] = runs["learning_rate"].eq(selected_lr)
    summary["selected_learning_rate"] = summary["learning_rate"].eq(selected_lr)
    selected_run_ids = runs.loc[runs["selected_learning_rate"], "run_id"].tolist()
    if len(selected_run_ids) != 3:
        raise ValueError("Must retain all three selected-rate seeds")

    runs_path = RESULTS / "neural_runs.csv"
    summary_path = RESULTS / "neural_summary.csv"
    drug_path = RESULTS / "neural_validation_by_drug.csv"
    curves_path = TEMP / "neural_learning_curves.csv"
    prediction_path = TEMP / "neural_validation_predictions.csv"
    runs.to_csv(runs_path, index=False, float_format="%.17g", lineterminator="\n")
    summary.to_csv(summary_path, index=False, float_format="%.17g", lineterminator="\n")
    pd.concat(by_drug_rows, ignore_index=True).to_csv(drug_path, index=False, float_format="%.17g", lineterminator="\n")
    pd.DataFrame(curve_rows).to_csv(curves_path, index=False, float_format="%.17g", lineterminator="\n")
    predictions.to_csv(prediction_path, index=False, float_format="%.17g", lineterminator="\n")
    baseline = pd.read_csv(RESULTS / "baseline_comparison.csv")
    driver = None
    if device.type == "cuda":
        command = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        driver = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    artifact_paths = [runs_path, summary_path, drug_path, curves_path, prediction_path]
    artifact_paths.extend(CHECKPOINTS / f"mlp_{run_id}.pt" for run_id in runs["run_id"])
    manifest = {
        "config_sha256": sha256(CONFIG_PATH),
        "baseline_manifest_sha256": sha256(RESULTS / "baseline_validation_manifest.json"),
        "input_primary_sha256": baseline_manifest["input_primary_sha256"],
        "selected_learning_rate": selected_lr,
        "selected_run_ids": selected_run_ids,
        "selection_rule": config["learning_rate_selection"],
        "baseline_validation_rmse": dict(zip(baseline["model"], baseline["validation_rmse"])),
        "hardware": {"device": device.type, "gpu_name": gpu_name, "gpu_memory_bytes": gpu_memory_bytes, "driver": driver, "cpu_count": os.cpu_count()},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "determinism": config["determinism"],
        "runtime_seconds_total_including_smoke_and_artifacts": perf_counter() - started,
        "artifact_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in artifact_paths},
        "checks": ["frozen_input_hashes", "row_alignment", "finite_losses", "smoke_device_and_shape",
                   "eval_mode_metrics", "checkpoint_reload", "selected_all_three_seeds", "no_test_predictions"],
    }
    (RESULTS / "neural_validation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", flush=True)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    print(f"Selected learning rate: {selected_lr:g}; selected run IDs: {selected_run_ids}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    run(smoke_only=args.smoke_only)
