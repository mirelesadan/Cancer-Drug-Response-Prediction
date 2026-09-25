"""Plot compact, exploratory grouped-split and PCA sensitivity results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLITS = (19, 23, 29, 31, 37)
LABELS = {
    "per_drug_mean": "Per-drug mean",
    "pooled_ridge": "Pooled Ridge",
    "mlp_three_seed_mean": "MLP (three-seed metric mean)",
}
COLORS = {
    "per_drug_mean": "#b95c3a",
    "pooled_ridge": "#3976ab",
    "mlp_three_seed_mean": "#7350a5",
}


def plot_split_differences(frame: pd.DataFrame, output: Path) -> None:
    expected = {(seed, model) for seed in SPLITS for model in (*LABELS, "per_drug_ridge")}
    observed = frame.loc[frame.model.isin((*LABELS, "per_drug_ridge"))]
    if len(observed) != len(expected) or set(zip(observed.split_seed, observed.model)) != expected:
        raise ValueError("Grouped-split comparison is incomplete or contains unexpected rows")
    fig, ax = plt.subplots(figsize=(8.6, 4.9), layout="constrained")
    ax.axhline(0, color="#444444", linewidth=1.2, label="Per-drug Ridge reference")
    for offset, (model, label) in zip((-0.16, 0, 0.16), LABELS.items()):
        one = frame.loc[frame.model.eq(model)].set_index("split_seed").loc[list(SPLITS)]
        x = np.arange(len(SPLITS), dtype=float) + offset
        delta = one.rmse.to_numpy() - frame.loc[frame.model.eq("per_drug_ridge")].set_index("split_seed").loc[list(SPLITS), "rmse"].to_numpy()
        low = one.paired_rmse_delta_vs_per_drug_ridge_ci_low.to_numpy()
        high = one.paired_rmse_delta_vs_per_drug_ridge_ci_high.to_numpy()
        ax.errorbar(
            x, delta, yerr=np.stack((delta - low, high - delta)),
            fmt="o", capsize=2.5, markersize=4.5, linewidth=1.4,
            color=COLORS[model], label=label,
        )
    ax.set_xticks(np.arange(len(SPLITS)), [str(seed) for seed in SPLITS])
    ax.set_xlabel("Prespecified cell-line split seed")
    ax.set_ylabel("Test RMSE − per-drug Ridge RMSE (provided Y)")
    ax.set_title("Paired model differences across five overlapping GDSC2 splits")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pca_dimensions(frame: pd.DataFrame, output: Path) -> None:
    dimensions = (64, 128, 256)
    models = ("pooled_ridge", "per_drug_ridge")
    expected = {(seed, dimension, model) for seed in SPLITS for dimension in dimensions for model in models}
    if set(zip(frame.split_seed, frame.pca_components, frame.model)) != expected:
        raise ValueError("PCA comparison is incomplete or contains unexpected rows")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), sharey=True, layout="constrained")
    palette = plt.get_cmap("viridis")
    for ax, model in zip(axes, models):
        for number, seed in enumerate(SPLITS):
            one = frame.loc[frame.model.eq(model) & frame.split_seed.eq(seed)].set_index("pca_components").loc[list(dimensions)]
            ax.plot(dimensions, one.rmse.to_numpy(), "o-", linewidth=1.5,
                    markersize=4, color=palette(number / 4), label=f"Split {seed}")
        ax.set_title("Pooled Ridge" if model == "pooled_ridge" else "Per-drug Ridge")
        ax.set_xticks(dimensions)
        ax.set_xlabel("Training-fitted PCA components")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Test RMSE (provided Y)")
    axes[1].legend(title="Cell-line split", fontsize=8, loc="upper right")
    fig.suptitle("PCA pipeline sensitivity; separately fitted randomized bases", fontsize=11)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--pca-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(args.split_csv)
    pca = pd.read_csv(args.pca_csv)
    plot_split_differences(split, args.output_dir / "multisplit_paired_rmse.png")
    plot_pca_dimensions(pca, args.output_dir / "pca_dimension_sensitivity.png")


if __name__ == "__main__":
    main()
