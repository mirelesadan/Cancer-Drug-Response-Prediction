r"""Run the predeclared molecular GNN extension on training/validation only.

From the repository root:
    .venv\Scripts\python.exe src\run_gnn.py --smoke-only
    .venv\Scripts\python.exe src\run_gnn.py

No test outcomes are scored here. The historical MLP and final-evaluation
artifacts are left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
from rdkit import rdBase
import torch
import torch_geometric
from torch.nn import functional as F

from baselines import by_drug_validation, errors
from gnn import ATOM_DIM, BOND_DIM, MolecularResponseGNN, build_drug_graphs, predict_rows
from run_neural import configure_determinism, indexed_tensors, load_primary, set_seed, sha256


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
TEMP = RESULTS / "tmp"
CONFIG_PATH = ROOT / "configs" / "gnn_extension.json"


def make_model(config: dict, n_drugs: int, device: torch.device) -> MolecularResponseGNN:
    settings = config["architecture"]
    if settings["graph_layers"] != 2 or settings["graph_operator"] != "GINEConv_with_edge_attributes":
        raise ValueError("GNN architecture differs from fixed extension protocol")
    model = MolecularResponseGNN(
        n_drugs=n_drugs,
        expression_dim=128,
        graph_dim=int(settings["graph_hidden_dim"]),
        fusion_hidden=tuple(settings["fusion_hidden_layers"]),
        dropout=float(settings["dropout"]),
    )
    return model.to(device)


def load_context(config: dict, device: torch.device):
    train, validation, drugs, scores, _fingerprints_for_input_audit, baseline_manifest = load_primary(config)
    graphs, graph_hash = build_drug_graphs(drugs)
    if len(drugs) != 137 or graphs.num_graphs != 137 or scores.shape != (805, 128):
        raise ValueError("Frozen drug/cell feature catalogs changed")
    if train["drug_id"].nunique() != 136 or validation["drug_id"].nunique() != 136:
        raise ValueError("Primary known-drug coverage changed")
    if train["cell_line_id"].nunique() != 564 or validation["cell_line_id"].nunique() != 121:
        raise ValueError("Frozen cell-line split changed")
    if set(train["cell_line_id"]) & set(validation["cell_line_id"]):
        raise ValueError("Training and validation cell lines overlap")
    if train["drug_id"].eq("Uprosertib").any() or validation["drug_id"].eq("Uprosertib").any():
        raise ValueError("Primary Uprosertib exclusion changed")
    if not set(validation["drug_id"]).issubset(set(train["drug_id"])):
        raise ValueError("Validation drug absent from training")
    cell_features = torch.as_tensor(np.ascontiguousarray(scores, dtype=np.float32), device=device)
    train_tensors = indexed_tensors(train, device)
    val_tensors = indexed_tensors(validation, device)
    return train, validation, drugs, cell_features, graphs.to(device), graph_hash, train_tensors, val_tensors, baseline_manifest


def smoke_check(config: dict, device: torch.device, n_drugs: int, cell_features: torch.Tensor,
                graphs, train_tensors) -> None:
    set_seed(int(config["training"]["seeds"][0]))
    model = make_model(config, n_drugs, device)
    model.train()
    cell, drug, y = (tensor[:32] for tensor in train_tensors)
    output = model(cell_features[cell], drug, graphs)
    if output.shape != y.shape or output.device != y.device or not torch.isfinite(output).all():
        raise ValueError("GNN smoke shape, device, or finiteness failed")
    loss = F.mse_loss(output, y)
    if not torch.isfinite(loss):
        raise ValueError("GNN smoke loss is not finite")
    loss.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.graph_layers[0].parameters()):
        raise ValueError("Graph encoder did not receive finite gradients")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    optimizer.step()
    estimate = predict_rows(model, cell, drug, cell_features, graphs, 32)
    if estimate.shape != (32,) or not torch.isfinite(estimate).all():
        raise ValueError("GNN smoke evaluation failed")
    print(f"SMOKE PASS ({device.type}): 32 aligned rows, finite graph gradients, eval-mode shape (32,)", flush=True)


def train_one(config: dict, seed: int, device: torch.device, n_drugs: int, cell_features: torch.Tensor,
              graphs, train_tensors, val_tensors, train: pd.DataFrame, validation: pd.DataFrame,
              graph_hash: str):
    started = perf_counter()
    settings = config["training"]
    generator = set_seed(seed)
    model = make_model(config, n_drugs, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    train_cell, train_drug, train_y = train_tensors
    val_cell, val_drug, _ = val_tensors
    y_train = train["y"].to_numpy(dtype=np.float64)
    y_val = validation["y"].to_numpy(dtype=np.float64)
    batch_size = int(settings["batch_size"])
    eval_batch_size = int(settings["evaluation_batch_size"])
    patience = int(settings["early_stopping_patience"])
    max_epochs = int(settings["maximum_epochs"])
    run_id = f"gnn_seed_{seed}"
    checkpoint = CHECKPOINTS / f"{run_id}.pt"
    best_rmse = float("inf")
    best_epoch = 0
    best_validation_prediction = None
    epochs_without_improvement = 0
    curve = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).to(device)
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch_ids = order[start:start + batch_size]
            target = train_y[batch_ids]
            predicted = model(cell_features[train_cell[batch_ids]], train_drug[batch_ids], graphs)
            if predicted.shape != target.shape:
                raise ValueError("Training prediction/target shape mismatch")
            loss = F.mse_loss(predicted, target)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss in {run_id}, epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(batch_ids)
        train_prediction = predict_rows(model, train_cell, train_drug, cell_features, graphs, eval_batch_size).numpy().astype(np.float64)
        validation_prediction = predict_rows(model, val_cell, val_drug, cell_features, graphs, eval_batch_size).numpy().astype(np.float64)
        train_rmse, train_mae = errors(y_train, train_prediction)
        val_rmse, val_mae = errors(y_val, validation_prediction)
        if not np.isfinite([train_rmse, train_mae, val_rmse, val_mae]).all():
            raise ValueError("Nonfinite GNN evaluation metric")
        if val_rmse < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch
            best_validation_prediction = validation_prediction.copy()
            epochs_without_improvement = 0
            torch.save({
                "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "seed": seed, "best_epoch": epoch, "validation_rmse": val_rmse,
                "config_sha256": sha256(CONFIG_PATH), "graph_sha256": graph_hash,
            }, checkpoint)
        else:
            epochs_without_improvement += 1
        curve.append({
            "run_id": run_id, "seed": seed, "epoch": epoch, "train_batch_mse": loss_sum / len(train),
            "train_eval_rmse": train_rmse, "train_eval_mae": train_mae,
            "validation_rmse": val_rmse, "validation_mae": val_mae,
            "best_epoch_so_far": best_epoch,
        })
        if epoch == 1 or epoch % 5 == 0:
            print(f"{run_id} epoch {epoch}: train {train_rmse:.4f}, val {val_rmse:.4f}, best {best_rmse:.4f} at {best_epoch}", flush=True)
        if epochs_without_improvement >= patience:
            break
    if best_validation_prediction is None:
        raise ValueError("No finite GNN validation checkpoint")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved["seed"] != seed or saved["best_epoch"] != best_epoch or saved["config_sha256"] != sha256(CONFIG_PATH) or saved["graph_sha256"] != graph_hash:
        raise ValueError("GNN checkpoint metadata mismatch")
    model.load_state_dict(saved["state_dict"])
    restored_train = predict_rows(model, train_cell, train_drug, cell_features, graphs, eval_batch_size).numpy().astype(np.float64)
    restored_val = predict_rows(model, val_cell, val_drug, cell_features, graphs, eval_batch_size).numpy().astype(np.float64)
    # CUDA graph aggregation can differ by a few float32 ulps on replay.
    np.testing.assert_allclose(restored_val, best_validation_prediction, rtol=0, atol=1e-5)
    train_rmse, train_mae = errors(y_train, restored_train)
    val_rmse, val_mae = errors(y_val, restored_val)
    if abs(val_rmse - best_rmse) > 1e-5:
        raise ValueError("Reloaded GNN validation metric mismatch")
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {
        "run_id": run_id, "seed": seed, "best_epoch": best_epoch, "last_epoch": epoch,
        "early_stopped": bool(epochs_without_improvement >= patience),
        "hit_epoch_cap": bool(epoch == max_epochs),
        "train_rmse": train_rmse, "train_mae": train_mae,
        "validation_rmse": val_rmse, "validation_mae": val_mae,
        "runtime_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
    }
    print(f"DONE {run_id}: epoch {best_epoch}, validation RMSE {val_rmse:.4f}, {result['runtime_seconds']:.1f}s", flush=True)
    return result, curve, restored_val


def run(smoke_only: bool = False) -> None:
    started = perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    settings = config["training"]
    if settings["seeds"] != [17, 29, 43] or settings["learning_rate"] != 0.0003 or config["test_policy"] != "no_test_access_until_validation_runs_checks_and_separate_extension_lock":
        raise ValueError("GNN extension differs from predeclared protocol")
    configure_determinism(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, drugs, cell_features, graphs, graph_hash, train_tensors, val_tensors, baseline_manifest = load_context(config, device)
    smoke_check(config, device, len(drugs), cell_features, graphs, train_tensors)
    if smoke_only:
        return
    CHECKPOINTS.mkdir(exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    run_rows, curve_rows, drug_rows = [], [], []
    prediction_frame = validation[["source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "split", "y"]].copy()
    train_counts = train.groupby("drug_id").size().to_dict()
    for seed in settings["seeds"]:
        result, curves, estimate = train_one(
            config, int(seed), device, len(drugs), cell_features, graphs,
            train_tensors, val_tensors, train, validation, graph_hash,
        )
        drug_metrics = by_drug_validation(validation["drug_id"].to_numpy(), validation["y"].to_numpy(dtype=np.float64),
                                          estimate, train_counts, minimum_spearman_n=int(config["minimum_within_drug_spearman_n"]))
        drug_metrics.insert(0, "run_id", result["run_id"])
        result["validation_equal_drug_mean_rmse"] = float(drug_metrics["rmse"].mean())
        result["validation_median_within_drug_spearman"] = float(drug_metrics["spearman"].median())
        result["spearman_eligible_drugs"] = int(drug_metrics["spearman_status"].isin(["defined", "constant_prediction", "undefined_numeric"]).sum())
        result["spearman_defined_drugs"] = int(drug_metrics["spearman"].notna().sum())
        prediction_frame[result["run_id"]] = estimate
        run_rows.append(result)
        curve_rows.extend(curves)
        drug_rows.append(drug_metrics)
    runs = pd.DataFrame(run_rows)
    if runs["seed"].tolist() != settings["seeds"]:
        raise ValueError("Incomplete GNN seed set")
    runs_path = RESULTS / "gnn_validation_runs.csv"
    drug_path = RESULTS / "gnn_validation_by_drug.csv"
    curve_path = TEMP / "gnn_learning_curves.csv"
    prediction_path = TEMP / "gnn_validation_predictions.csv"
    runs.to_csv(runs_path, index=False, float_format="%.17g", lineterminator="\n")
    pd.concat(drug_rows, ignore_index=True).to_csv(drug_path, index=False, float_format="%.17g", lineterminator="\n")
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False, float_format="%.17g", lineterminator="\n")
    prediction_frame.to_csv(prediction_path, index=False, float_format="%.17g", lineterminator="\n")
    if device.type == "cuda":
        torch.cuda.synchronize()
    driver = None
    if device.type == "cuda":
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    artifact_paths = [runs_path, drug_path, curve_path, prediction_path]
    artifact_paths.extend(ROOT / item for item in runs["checkpoint"])
    manifest = {
        "scope": "exploratory_validation_only_no_test_predictions",
        "config_sha256": sha256(CONFIG_PATH),
        "baseline_manifest_sha256": sha256(RESULTS / "baseline_validation_manifest.json"),
        "input_primary_sha256": baseline_manifest["input_primary_sha256"],
        "graph_sha256": graph_hash,
        "drug_catalog_sha256": sha256(ROOT / "data" / "processed" / "drugs.csv"),
        "expression_scores_sha256": sha256(ROOT / "data" / "processed" / "expression_pca128_scaled.npy"),
        "atom_feature_dim": ATOM_DIM, "bond_feature_dim": BOND_DIM,
        "hardware": {"device": device.type, "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                     "driver": driver},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "pyg": torch_geometric.__version__, "rdkit": rdBase.rdkitVersion,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "determinism": config["determinism"],
        "seeds": settings["seeds"],
        "runtime_seconds_total": perf_counter() - started,
        "artifact_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in artifact_paths},
        "checks": ["frozen_input_hashes", "split_and_row_alignment", "training_drug_coverage", "deterministic_graph_featurization",
                   "finite_smoke_and_gradients", "finite_losses", "checkpoint_reload", "no_test_predictions"],
    }
    (RESULTS / "gnn_validation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(runs[["run_id", "best_epoch", "validation_rmse", "validation_mae", "runtime_seconds"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    run(smoke_only=arguments.smoke_only)
