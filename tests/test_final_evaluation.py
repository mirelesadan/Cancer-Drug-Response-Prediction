"""Checks for paired whole-cell uncertainty and reported metric arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_final import NAMES, bootstrap_clusters


def test_bootstrap_duplicates_whole_clusters_and_averages_seed_metrics() -> None:
    cells = np.repeat(np.arange(120), 2)
    y = np.zeros(len(cells))
    frame = pd.DataFrame({"cell_line_id": cells.astype(str), "y": y})
    for index, name in enumerate(NAMES):
        frame[name] = cells.astype(float) / 20 + (index + 1) * np.tile([1.0, -0.5], 120)
    intervals, replica = bootstrap_clusters(frame, resamples=10, seed=2026)
    assert len(intervals) == 7
    assert (replica["draws"].sum(axis=1) == 120).all()
    draws = replica["draws"][0]
    row_counts = frame["cell_line_id"].map(dict(zip(replica["cell_ids"], draws))).to_numpy(dtype=int)
    for name in NAMES:
        explicit_residual = np.repeat((frame[name] - frame["y"]).to_numpy(), row_counts)
        np.testing.assert_allclose(replica[f"{name}_rmse"][0], np.sqrt(np.mean(explicit_residual ** 2)))
        np.testing.assert_allclose(replica[f"{name}_mae"][0], np.mean(np.abs(explicit_residual)))
    mean_seed_rmse = np.mean([replica[f"mlp_seed_{s}_rmse"][0] for s in (17, 29, 43)])
    np.testing.assert_allclose(replica["mlp_three_seed_mean_rmse"][0], mean_seed_rmse)


def test_bootstrap_seed_reproduces_identical_paired_draws() -> None:
    cells = np.repeat(np.arange(120), 2)
    frame = pd.DataFrame({"cell_line_id": cells.astype(str), "y": np.zeros(len(cells))})
    for name in NAMES:
        frame[name] = 1.0
    _, first = bootstrap_clusters(frame, resamples=3, seed=2026)
    _, second = bootstrap_clusters(frame, resamples=3, seed=2026)
    np.testing.assert_array_equal(first["draws"], second["draws"])


if __name__ == "__main__":
    test_bootstrap_duplicates_whole_clusters_and_averages_seed_metrics()
    test_bootstrap_seed_reproduces_identical_paired_draws()
    print("Two focused bootstrap checks passed")
