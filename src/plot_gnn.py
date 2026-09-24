"""Plot the saved exploratory GNN extension results without recomputing metrics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURE = RESULTS / "figures" / "gnn_extension.png"
SEEDS = (17, 29, 43)


def run() -> Path:
    comparison = pd.read_csv(RESULTS / "gnn_test_comparison.csv").set_index("model")
    intervals = pd.read_csv(RESULTS / "gnn_bootstrap_intervals.csv").set_index("model")
    curves = pd.read_csv(RESULTS / "gnn_learning_curves.csv")
    runs = pd.read_csv(RESULTS / "gnn_validation_runs.csv").set_index("seed")

    names = ("per_drug_ridge", "gnn_three_seed_mean")
    labels = ("Per-drug Ridge", "GNN (3-seed metric mean)")
    colors = ("#22577a", "#c65c3e")
    for name in names:
        assert name in comparison.index and name in intervals.index
        point = comparison.loc[name, "rmse"]
        assert intervals.loc[name, "rmse_ci_low"] <= point <= intervals.loc[name, "rmse_ci_high"]
    assert set(SEEDS) == set(runs.index)
    assert set(SEEDS) == set(curves["seed"].unique())

    fig, (ax_score, ax_curve) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.suptitle("Exploratory molecular GNN extension on held-out cell lines", fontsize=13, fontweight="bold")

    for y, (name, label, color) in enumerate(zip(names, labels, colors)):
        value = float(comparison.loc[name, "rmse"])
        low = float(intervals.loc[name, "rmse_ci_low"])
        high = float(intervals.loc[name, "rmse_ci_high"])
        ax_score.errorbar(
            value, y, xerr=[[value - low], [high - value]], fmt="o", markersize=7,
            capsize=5, elinewidth=2, color=color,
        )
        ax_score.text(high + 0.015, y, f"{value:.3f}", va="center", fontsize=9, color=color)
    ax_score.set_yticks([0, 1], labels)
    ax_score.invert_yaxis()
    ax_score.set_xlim(1.12, 1.87)
    ax_score.set_ylim(1.65, -0.7)
    ax_score.set_xlabel("Test RMSE (provided Y scale)")
    ax_score.set_title("Response-weighted test error", fontsize=11)
    ax_score.grid(axis="x", alpha=0.2)
    ax_score.spines[["top", "right"]].set_visible(False)
    delta = float(comparison.loc[names[1], "rmse"] - comparison.loc[names[0], "rmse"])
    delta_low = float(intervals.loc[names[1], "paired_rmse_delta_ci_low"])
    delta_high = float(intervals.loc[names[1], "paired_rmse_delta_ci_high"])
    ax_score.text(
        0.02, 0.04,
        f"GNN − Ridge: +{delta:.3f} RMSE\nPaired 95% CI: [{delta_low:.3f}, {delta_high:.3f}]",
        transform=ax_score.transAxes, ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f3f5f6", "edgecolor": "none"},
    )

    seed_colors = {17: "#1d6e92", 29: "#6a51a3", 43: "#b45d3b"}
    for seed in SEEDS:
        one = curves.loc[curves["seed"] == seed].sort_values("epoch")
        epoch = int(runs.loc[seed, "best_epoch"])
        best = one.loc[one["epoch"] == epoch]
        assert len(best) == 1
        ax_curve.plot(one["epoch"], one["train_eval_rmse"], color=seed_colors[seed], alpha=0.25, linewidth=1.1)
        ax_curve.plot(one["epoch"], one["validation_rmse"], color=seed_colors[seed], linewidth=1.7, label=f"Validation, seed {seed}")
        ax_curve.scatter([epoch], best["validation_rmse"], color=seed_colors[seed], s=30, zorder=3)
    ax_curve.plot([], [], color="#777777", alpha=0.45, linewidth=1.1, label="Training (same seed colors)")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("RMSE (provided Y scale)")
    ax_curve.set_title("Validation-selected checkpoints", fontsize=11)
    ax_curve.set_xlim(1, curves["epoch"].max())
    ax_curve.set_ylim(1.2, 2.85)
    ax_curve.grid(alpha=0.2)
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.legend(loc="upper right", frameon=False, fontsize=8)

    fig.text(
        0.5, 0.025,
        "Bars: 95% intervals from 2,000 paired whole-cell bootstrap draws; conditional on this split and fitted models. "
        "GNN point averages seed-level metrics, not predictions.",
        ha="center", fontsize=8, color="#444444",
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.83, bottom=0.18, wspace=0.38)
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE, dpi=190, facecolor="white")
    plt.close(fig)
    return FIGURE


if __name__ == "__main__":
    print(run())
