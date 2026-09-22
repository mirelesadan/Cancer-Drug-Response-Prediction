r"""Replay saved baseline validation predictions without fitting or test scoring.

From the repository root: .venv\Scripts\python.exe src\check_baselines.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from baselines import errors


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check() -> None:
    manifest = json.loads((RESULTS / "baseline_validation_manifest.json").read_text(encoding="utf-8"))
    prep = json.loads((RESULTS / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    assert sha256(ROOT / "configs" / "baseline_ridge.json") == manifest["config_sha256"]
    assert sha256(RESULTS / "preprocessing_manifest.json") == manifest["preprocessing_manifest_sha256"]
    for relative, expected in manifest["artifact_sha256"].items():
        assert sha256(ROOT / relative) == expected, relative
    assert sha256(OUT / "primary_responses.csv") == prep["artifacts_sha256"]["primary_responses.csv"]

    assignments = pd.read_csv(OUT / "cell_line_assignments.csv")
    drugs = pd.read_csv(OUT / "drugs.csv")
    primary = pd.read_csv(OUT / "primary_responses.csv", float_precision="round_trip")
    train = primary.loc[primary["split"] == "train"]
    validation = primary.loc[primary["split"] == "validation"].reset_index(drop=True)
    saved = pd.read_csv(RESULTS / "tmp" / "baseline_validation_predictions.csv", float_precision="round_trip")
    comparison = pd.read_csv(RESULTS / "baseline_comparison.csv")
    score_raw = np.load(OUT / "expression_pca128.npy", allow_pickle=False)
    score_saved = np.load(OUT / "expression_pca128_scaled.npy", allow_pickle=False)
    fingerprints = np.load(OUT / "morgan_radius2_2048.npy", allow_pickle=False)
    scaler = joblib.load(OUT / "pca_score_scaler.joblib")
    models = joblib.load(RESULTS / "checkpoints" / "baseline_models.joblib")

    assert len(saved) == len(validation) == 13661
    assert saved["split"].eq("validation").all()
    assert saved["source_row_id"].tolist() == validation["source_row_id"].tolist()
    assert saved["drug_id"].tolist() == validation["drug_id"].tolist()
    assert saved["cell_line_id"].tolist() == validation["cell_line_id"].tolist()
    assert np.array_equal(saved["y"].to_numpy(), validation["y"].to_numpy())
    assert saved["drug_id"].tolist() == drugs.loc[saved["drug_index"], "drug_id"].tolist()
    assert saved["cell_line_id"].tolist() == assignments.loc[saved["cell_index"], "cell_line_id"].tolist()

    train_cell_positions = assignments.index[assignments["split"] == "train"].to_numpy()
    np.testing.assert_allclose(scaler.mean_, score_raw[train_cell_positions].mean(axis=0), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(scaler.var_, score_raw[train_cell_positions].var(axis=0), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(score_saved, scaler.transform(score_raw), rtol=0, atol=0)
    means = models["per_drug_mean"].means
    for drug_index, subset in train.groupby("drug_index"):
        assert np.isclose(means[drug_index], subset["y"].mean(), rtol=1e-13, atol=1e-13)
    assert train["drug_id"].nunique() == 136 and not primary["drug_id"].eq("Uprosertib").any()

    cells = saved["cell_index"].to_numpy(dtype=np.int32)
    drug_ids = saved["drug_index"].to_numpy(dtype=np.int32)
    replay = {
        "per_drug_mean": models["per_drug_mean"].predict(drug_ids),
        "pooled_ridge": models["pooled_ridge"].predict(cells, drug_ids, score_saved, fingerprints),
        "per_drug_ridge": models["per_drug_ridge"].predict(cells, drug_ids, score_saved),
    }
    for name, predicted in replay.items():
        np.testing.assert_allclose(predicted, saved[name].to_numpy(), rtol=0, atol=1e-10)
        rmse, mae = errors(saved["y"].to_numpy(), predicted)
        row = comparison.loc[comparison["model"] == name].iloc[0]
        assert np.isclose(rmse, row["validation_rmse"], rtol=1e-13, atol=1e-13)
        assert np.isclose(mae, row["validation_mae"], rtol=1e-13, atol=1e-13)
    assert models["pooled_ridge"].alpha == manifest["selected_alpha"]["pooled_ridge"]
    assert models["per_drug_ridge"].alpha == manifest["selected_alpha"]["per_drug_ridge"]
    print("PASS: hashes, frozen split, training-only means/scaler, row alignment, and validation replay after reload")


if __name__ == "__main__":
    check()
