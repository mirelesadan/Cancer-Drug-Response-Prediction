r"""Independently reload all GNN checkpoints and replay validation rows.

From the repository root: .venv\Scripts\python.exe src\check_gnn.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch

from baselines import errors
from gnn import build_drug_graphs, predict_rows
from run_gnn import make_model
from run_neural import configure_determinism, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def check() -> None:
    config_path = ROOT / "configs" / "gnn_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "gnn_validation_manifest.json").read_text(encoding="utf-8"))
    if sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("GNN configuration differs from trained run")
    if sha256(RESULTS / "baseline_validation_manifest.json") != manifest["baseline_manifest_sha256"]:
        raise ValueError("Primary baseline manifest changed")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"GNN artifact hash changed: {relative}")
    if sha256(OUT / "primary_responses.csv") != manifest["input_primary_sha256"]:
        raise ValueError("Frozen response table changed")
    if sha256(OUT / "expression_pca128_scaled.npy") != manifest["expression_scores_sha256"]:
        raise ValueError("Frozen expression transform changed")
    if sha256(OUT / "drugs.csv") != manifest["drug_catalog_sha256"]:
        raise ValueError("Frozen drug catalog changed")
    drugs = pd.read_csv(OUT / "drugs.csv")
    graphs, graph_hash = build_drug_graphs(drugs)
    if graph_hash != manifest["graph_sha256"]:
        raise ValueError("Drug graph representation changed")
    configure_determinism(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != manifest["hardware"]["device"]:
        raise ValueError("Replay device differs from validation run")
    graphs = graphs.to(device)
    scores = torch.as_tensor(np.load(OUT / "expression_pca128_scaled.npy", allow_pickle=False), dtype=torch.float32, device=device)
    primary = pd.read_csv(OUT / "primary_responses.csv", float_precision="round_trip")
    validation = primary.loc[primary["split"].eq("validation")].reset_index(drop=True)
    saved = pd.read_csv(RESULTS / "tmp" / "gnn_validation_predictions.csv", float_precision="round_trip")
    runs = pd.read_csv(RESULTS / "gnn_validation_runs.csv")
    if len(validation) != 13661 or len(saved) != 13661 or runs["seed"].tolist() != [17, 29, 43]:
        raise ValueError("GNN validation rows or seeds changed")
    for column in ("source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "split"):
        if saved[column].tolist() != validation[column].tolist():
            raise ValueError(f"GNN saved prediction row misalignment: {column}")
    if not np.array_equal(saved["y"].to_numpy(), validation["y"].to_numpy()):
        raise ValueError("GNN saved target misalignment")
    if validation["drug_id"].eq("Uprosertib").any():
        raise ValueError("Uprosertib entered primary validation")
    if validation["drug_id"].tolist() != drugs.loc[validation["drug_index"], "drug_id"].tolist():
        raise ValueError("GNN drug-index mapping changed")
    assignments = pd.read_csv(OUT / "cell_line_assignments.csv")
    if validation["cell_line_id"].tolist() != assignments.loc[validation["cell_index"], "cell_line_id"].tolist():
        raise ValueError("GNN cell-index mapping changed")
    cell_index = torch.as_tensor(saved["cell_index"].to_numpy(dtype=np.int64), device=device)
    drug_index = torch.as_tensor(saved["drug_index"].to_numpy(dtype=np.int64), device=device)
    y = saved["y"].to_numpy(dtype=np.float64)
    max_prediction_difference = 0.0
    for run in runs.itertuples(index=False):
        checkpoint = torch.load(ROOT / run.checkpoint, map_location=device, weights_only=True)
        if checkpoint["seed"] != run.seed or checkpoint["best_epoch"] != run.best_epoch:
            raise ValueError("GNN checkpoint seed/epoch mismatch")
        if checkpoint["config_sha256"] != manifest["config_sha256"] or checkpoint["graph_sha256"] != graph_hash:
            raise ValueError("GNN checkpoint configuration/graph identity mismatch")
        model = make_model(config, len(drugs), device)
        model.load_state_dict(checkpoint["state_dict"])
        replay = predict_rows(model, cell_index, drug_index, scores, graphs, int(config["training"]["evaluation_batch_size"])).numpy().astype(np.float64)
        original = saved[run.run_id].to_numpy(dtype=np.float64)
        difference = float(np.max(np.abs(replay - original)))
        max_prediction_difference = max(max_prediction_difference, difference)
        if difference > 1e-5:
            raise ValueError(f"GNN validation replay exceeds float32 tolerance: {run.run_id}: {difference}")
        rmse, mae = errors(y, replay)
        if abs(rmse - run.validation_rmse) > 1e-5 or abs(mae - run.validation_mae) > 1e-5:
            raise ValueError("GNN validation metrics differ after checkpoint reload")
    result = {
        "check": "passed",
        "checkpoint_count": len(runs),
        "validation_rows": len(validation),
        "max_abs_prediction_difference": max_prediction_difference,
        "tolerance": 1e-5,
        "test_predictions_computed": False,
    }
    (RESULTS / "gnn_replay_check.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    check()
