r"""Verify saved neural validation artifacts by reloading all six checkpoints.

From the repository root: .venv\Scripts\python.exe src\check_neural.py
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
from neural import DrugResponseMLP, predict_rows
from run_neural import configure_determinism, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def check() -> None:
    manifest = json.loads((RESULTS / "neural_validation_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs" / "neural_mlp.json").read_text(encoding="utf-8"))
    assert sha256(ROOT / "configs" / "neural_mlp.json") == manifest["config_sha256"]
    assert sha256(RESULTS / "baseline_validation_manifest.json") == manifest["baseline_manifest_sha256"]
    for relative, expected in manifest["artifact_sha256"].items():
        assert sha256(ROOT / relative) == expected, relative
    configure_determinism(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == manifest["hardware"]["device"]

    primary = pd.read_csv(OUT / "primary_responses.csv", float_precision="round_trip")
    validation = primary.loc[primary["split"] == "validation"].reset_index(drop=True)
    saved = pd.read_csv(RESULTS / "tmp" / "neural_validation_predictions.csv", float_precision="round_trip")
    runs = pd.read_csv(RESULTS / "neural_runs.csv")
    summary = pd.read_csv(RESULTS / "neural_summary.csv")
    by_drug = pd.read_csv(RESULTS / "neural_validation_by_drug.csv")
    assert len(validation) == len(saved) == 13661
    assert saved["split"].eq("validation").all()
    for column in ("source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index"):
        assert saved[column].tolist() == validation[column].tolist(), column
    assert np.array_equal(saved["y"].to_numpy(), validation["y"].to_numpy())
    assert not saved["drug_id"].eq("Uprosertib").any()
    assert len(runs) == 6 and len(summary) == 2
    assert sorted(runs["seed"].unique().tolist()) == [17, 29, 43]
    np.testing.assert_allclose(sorted(runs["learning_rate"].unique()), [0.0003, 0.001], rtol=0, atol=1e-15)
    assert len(by_drug) == 6 * 136
    assert by_drug.groupby("run_id").size().eq(136).all()
    assert runs.loc[runs["selected_learning_rate"], "run_id"].tolist() == manifest["selected_run_ids"]
    selected_from_summary = float(summary.loc[summary["validation_rmse_mean"].idxmin(), "learning_rate"])
    assert np.isclose(selected_from_summary, manifest["selected_learning_rate"])

    scores = torch.as_tensor(np.load(OUT / "expression_pca128_scaled.npy", allow_pickle=False), dtype=torch.float32, device=device)
    fingerprints = torch.as_tensor(np.load(OUT / "morgan_radius2_2048.npy", allow_pickle=False), dtype=torch.float32, device=device)
    cell_index = torch.as_tensor(saved["cell_index"].to_numpy(dtype=np.int64), device=device)
    drug_index = torch.as_tensor(saved["drug_index"].to_numpy(dtype=np.int64), device=device)
    actual = saved["y"].to_numpy(dtype=np.float64)
    for row in runs.itertuples(index=False):
        checkpoint = torch.load(ROOT / row.checkpoint, map_location=device, weights_only=True)
        assert checkpoint["seed"] == row.seed
        assert checkpoint["best_epoch"] == row.best_epoch
        assert np.isclose(checkpoint["learning_rate"], row.learning_rate)
        model = DrugResponseMLP(
            checkpoint["input_dim"], tuple(checkpoint["hidden_layers"]), checkpoint["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        replay = predict_rows(model, cell_index, drug_index, scores, fingerprints, int(config["evaluation_batch_size"])).numpy().astype(np.float64)
        np.testing.assert_allclose(replay, saved[row.run_id].to_numpy(), rtol=0, atol=1e-6)
        rmse, mae = errors(actual, replay)
        assert np.isclose(rmse, row.validation_rmse, rtol=0, atol=1e-6)
        assert np.isclose(mae, row.validation_mae, rtol=0, atol=1e-6)
        assert row.best_epoch <= row.last_epoch <= 100
    print("PASS: six checkpoint hashes/reloads, validation row alignment and replay, selected three-seed configuration, no test predictions")


if __name__ == "__main__":
    check()
