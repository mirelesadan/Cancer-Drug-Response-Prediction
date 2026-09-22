r"""Replay saved held-out predictions and verify locked final summaries."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from baselines import errors
from evaluate_final import (
    CHECKPOINTS, NAMES, PROCESSED, RESULTS, ROOT, TEMP, predict_all,
)
from run_sensitivity import sha256, verify_locked


def run() -> None:
    config = json.loads((ROOT / "configs" / "final_evaluation.json").read_text(encoding="utf-8"))
    verify_locked(config)
    manifest = json.loads((RESULTS / "final_evaluation_manifest.json").read_text(encoding="utf-8"))
    if manifest["final_config_sha256"] != sha256(ROOT / "configs" / "final_evaluation.json"):
        raise ValueError("Final manifest references different lock")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Final artifact hash changed: {relative}")
    neural_config = json.loads((ROOT / "configs" / "neural_mlp.json").read_text(encoding="utf-8"))
    scores = np.load(PROCESSED / "expression_pca128_scaled.npy", allow_pickle=False)
    fingerprints = np.load(PROCESSED / "morgan_radius2_2048.npy", allow_pickle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for source, dataset, baseline, paths in [
        ("primary", "primary_responses.csv", "baseline_models.joblib",
         {int(seed): ROOT / f"results/checkpoints/mlp_{info['run_id']}.pt" for seed, info in config["primary_models"]["mlp_selected_runs"].items()}),
        ("sensitivity", "sensitivity_responses.csv", "sensitivity_baselines.joblib",
         {seed: CHECKPOINTS / f"sensitivity_mlp_seed_{seed}.pt" for seed in (17, 29, 43)}),
    ]:
        all_rows = pd.read_csv(PROCESSED / dataset, dtype={"source_row_ids": str} if source == "sensitivity" else None, float_precision="round_trip")
        test = all_rows.loc[all_rows["split"] == "test"].reset_index(drop=True)
        saved = pd.read_csv(TEMP / f"{source}_test_predictions.csv", dtype={"source_row_ids": str} if source == "sensitivity" else None, float_precision="round_trip")
        id_column = "source_row_id" if source == "primary" else "pair_id"
        if not test[id_column].equals(saved[id_column]) or not np.array_equal(test["y"].to_numpy(), saved["y"].to_numpy()):
            raise ValueError(f"{source} saved row identity/target changed")
        replay = predict_all(test, CHECKPOINTS / baseline, paths, scores, fingerprints, neural_config, device, sensitivity=source == "sensitivity")
        for name in NAMES:
            np.testing.assert_allclose(replay[name], saved[name], rtol=0, atol=1e-6,
                                       err_msg=f"Reloaded {source}/{name} predictions differ")
        if source == "primary":
            comparison = pd.read_csv(RESULTS / "final_test_comparison.csv")
            for name in NAMES:
                reported = comparison.loc[comparison["model"] == name].iloc[0]
                rmse, mae = errors(saved["y"].to_numpy(dtype=np.float64), saved[name].to_numpy(dtype=np.float64))
                np.testing.assert_allclose([rmse, mae], [reported["rmse"], reported["mae"]], rtol=0, atol=1e-12)
    with np.load(TEMP / "final_bootstrap_replicates.npz", allow_pickle=False) as stored:
        draws = stored["draws"]
        ids = stored["cell_ids"]
        if draws.shape != (2000, 120) or not np.all(draws.sum(axis=1) == 120):
            raise ValueError("Bootstrap cluster-draw shape/multiplicity changed")
        saved = pd.read_csv(TEMP / "primary_test_predictions.csv", float_precision="round_trip")
        multiplicity = saved["cell_line_id"].map(dict(zip(ids, draws[0]))).to_numpy(dtype=int)
        if int(multiplicity.sum()) != int(draws[0] @ stored["row_counts_per_cell"]):
            raise ValueError("Bootstrap row multiplicity changed")
        for name in NAMES:
            residual = np.repeat((saved[name] - saved["y"]).to_numpy(), multiplicity)
            rmse = float(np.sqrt(np.mean(residual * residual)))
            mae = float(np.mean(np.abs(residual)))
            np.testing.assert_allclose([rmse, mae], [stored[f"{name}_rmse"][0], stored[f"{name}_mae"][0]], rtol=0, atol=1e-12)
        seed_mean = np.mean([stored[f"mlp_seed_{seed}_rmse"][0] for seed in (17, 29, 43)])
        np.testing.assert_allclose(seed_mean, stored["mlp_three_seed_mean_rmse"][0], rtol=0, atol=1e-12)
    outcome = {"status": "passed", "frozen_config_sha256": manifest["final_config_sha256"],
               "primary_and_sensitivity_models_reloaded": 12, "test_prediction_tolerance": 1e-6,
               "checked": ["artifact_hashes", "row_identity", "saved_prediction_replay", "overall_metrics", "bootstrap_cluster_multiplicity", "per_draw_seed_metric_average"]}
    (RESULTS / "final_replay_check.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Final replay passed: 12 model predictions, metrics, hashes, and whole-cell bootstrap multiplicities")


if __name__ == "__main__":
    run()
