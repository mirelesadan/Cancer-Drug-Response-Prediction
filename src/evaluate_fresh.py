"""Evaluate one isolated fresh primary run after its validation-only lock exists.

This is a methodological replay of the primary known-drug study. It deliberately
does not run the historical Uprosertib sensitivity analysis or the optional GNN.
The lock is checked before the response table (and hence test targets) is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from baselines import by_drug_validation, errors


BASELINE_NAMES = ("per_drug_mean", "pooled_ridge", "per_drug_ridge")
LOCKED_INPUTS = {
    "configs/preprocessing.json",
    "configs/baseline_ridge.json",
    "data/processed/primary_responses.csv",
    "data/processed/cell_line_assignments.csv",
    "data/processed/drugs.csv",
    "data/processed/expression_pca128_scaled.npy",
    "data/processed/morgan_radius2_2048.npy",
    "results/preprocessing_manifest.json",
    "results/baseline_validation_manifest.json",
    "results/checkpoints/baseline_models.joblib",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
        raise ValueError(f"Invalid locked artifact path: {relative}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Locked artifact escapes run directory: {relative}")
    return resolved


def verify_lock(root: Path) -> dict:
    """Verify the frozen inputs without parsing response labels."""
    lock_path = root / "results" / "fresh_evaluation_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("mode") not in {"baselines", "primary"}:
        raise ValueError("Unsupported fresh-run evaluation lock")
    artifacts = lock.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not LOCKED_INPUTS.issubset(artifacts):
        raise ValueError("Fresh lock is missing required input hashes")
    if lock["mode"] == "primary" and not {
        "configs/neural_mlp.json", "results/neural_validation_manifest.json", "results/neural_runs.csv",
    }.issubset(artifacts):
        raise ValueError("Fresh primary lock is missing neural-selection inputs")
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Malformed locked artifact checksum")
        if sha256(verified_path(root, relative)) != expected:
            raise ValueError(f"Locked artifact changed: {relative}")
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = lock.get("source_sha256", {})
    if "src/evaluate_fresh.py" not in source_hashes:
        raise ValueError("Evaluator source was not frozen before test access")
    for relative, expected in source_hashes.items():
        if sha256(verified_path(source_root, relative)) != expected:
            raise ValueError(f"Fresh-run source changed since lock: {relative}")
    prep_manifest = json.loads((root / "results/preprocessing_manifest.json").read_text(encoding="utf-8"))
    if lock.get("raw_sha256") != prep_manifest.get("raw_sha256"):
        raise ValueError("Raw-input provenance differs from preprocessing manifest")
    baseline_manifest = json.loads((root / "results/baseline_validation_manifest.json").read_text(encoding="utf-8"))
    selected_alpha = lock.get("baseline_selected_alpha")
    if selected_alpha != baseline_manifest.get("selected_alpha"):
        raise ValueError("Locked Ridge selection differs from validation manifest")
    baseline_models = joblib.load(root / "results/checkpoints/baseline_models.joblib")
    if set(baseline_models) != set(BASELINE_NAMES):
        raise ValueError("Incomplete baseline checkpoint")
    for model in ("pooled_ridge", "per_drug_ridge"):
        if baseline_models[model].alpha != selected_alpha[model]:
            raise ValueError(f"{model} checkpoint alpha differs from validation lock")
    settings = lock.get("bootstrap")
    if not isinstance(settings, dict) or int(settings.get("resamples", 0)) < 1:
        raise ValueError("Invalid bootstrap configuration")
    int(settings["seed"])

    selected_runs = lock.get("selected_mlp_runs", [])
    if lock["mode"] == "baselines":
        if selected_runs:
            raise ValueError("Baseline-only lock unexpectedly names MLP checkpoints")
    else:
        if len(selected_runs) != 3 or sorted(int(run["seed"]) for run in selected_runs) != [17, 29, 43]:
            raise ValueError("Primary lock must retain all three selected-rate MLP seeds")
        neural_manifest = json.loads((root / "results/neural_validation_manifest.json").read_text(encoding="utf-8"))
        runs = pd.read_csv(root / "results/neural_runs.csv")
        selected_ids = set(neural_manifest["selected_run_ids"])
        if set(run["run_id"] for run in selected_runs) != selected_ids:
            raise ValueError("Locked MLP runs differ from validation selection")
        for run in selected_runs:
            run_id = run["run_id"]
            relative = f"results/checkpoints/mlp_{run_id}.pt"
            if relative not in artifacts or run["checkpoint_sha256"] != artifacts[relative]:
                raise ValueError("Selected MLP checkpoint is absent from artifact lock")
            row = runs.loc[runs["run_id"] == run_id]
            if len(row) != 1 or not bool(row.iloc[0]["selected_learning_rate"]):
                raise ValueError("Locked MLP run was not selected by validation")
            if (int(row.iloc[0]["seed"]) != int(run["seed"])
                    or int(row.iloc[0]["best_epoch"]) != int(run["best_epoch"])
                    or not np.isclose(float(row.iloc[0]["learning_rate"]), float(run["learning_rate"]), rtol=0, atol=1e-12)
                    or not np.isclose(float(run["learning_rate"]), float(neural_manifest["selected_learning_rate"]), rtol=0, atol=1e-12)):
                raise ValueError("Locked MLP identity or validation selection differs")
    return lock


def load_test_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, int]]:
    processed = root / "data/processed"
    assignments = pd.read_csv(processed / "cell_line_assignments.csv")
    drugs = pd.read_csv(processed / "drugs.csv")
    rows = pd.read_csv(processed / "primary_responses.csv", float_precision="round_trip")
    if not assignments["cell_line_id"].is_unique or set(assignments["split"]) != {"train", "validation", "test"}:
        raise ValueError("Cell-line split identities are incomplete or nonunique")
    if not drugs["drug_id"].is_unique or rows["source_row_id"].duplicated().any():
        raise ValueError("Drug or source-row identities are nonunique")
    if rows["drug_id"].eq("Uprosertib").any():
        raise ValueError("Primary exclusion changed")
    cell = rows["cell_index"].to_numpy(dtype=np.int64)
    drug = rows["drug_index"].to_numpy(dtype=np.int64)
    if (np.any(cell < 0) or np.any(cell >= len(assignments))
            or np.any(drug < 0) or np.any(drug >= len(drugs))):
        raise ValueError("Response index out of bounds")
    if not np.array_equal(rows["cell_line_id"].to_numpy(), assignments.iloc[cell]["cell_line_id"].to_numpy()):
        raise ValueError("Cell-line feature/label alignment changed")
    if not np.array_equal(rows["drug_id"].to_numpy(), drugs.iloc[drug]["drug_id"].to_numpy()):
        raise ValueError("Drug feature/label alignment changed")
    if not np.array_equal(rows["split"].to_numpy(), assignments.iloc[cell]["split"].to_numpy()):
        raise ValueError("Response split alignment changed")
    if not np.isfinite(rows["y"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite response target")
    test = rows.loc[rows["split"] == "test"].reset_index(drop=True)
    if test.empty or test["cell_line_id"].nunique() != int(assignments["split"].eq("test").sum()):
        raise ValueError("Test response coverage incomplete")
    train_counts = rows.loc[rows["split"] == "train"].groupby("drug_id").size().to_dict()
    if not set(test["drug_id"]).issubset(train_counts):
        raise ValueError("Test includes drug absent from training")
    scores = np.load(processed / "expression_pca128_scaled.npy", allow_pickle=False)
    fingerprints = np.load(processed / "morgan_radius2_2048.npy", allow_pickle=False)
    if scores.shape != (len(assignments), 128) or fingerprints.shape != (len(drugs), 2048):
        raise ValueError("Fresh feature-table shape changed")
    if not np.isfinite(scores).all() or not np.isin(fingerprints, [0, 1]).all():
        raise ValueError("Nonfinite expression or nonbinary drug features")
    return test, drugs, scores, fingerprints, train_counts


def baseline_predictions(root: Path, frame: pd.DataFrame, scores: np.ndarray,
                         fingerprints: np.ndarray) -> dict[str, np.ndarray]:
    path = root / "results/checkpoints/baseline_models.joblib"
    cell = frame["cell_index"].to_numpy(dtype=np.int64)
    drug = frame["drug_index"].to_numpy(dtype=np.int64)

    def predict(models: dict) -> dict[str, np.ndarray]:
        return {
            "per_drug_mean": models["per_drug_mean"].predict(drug),
            "pooled_ridge": models["pooled_ridge"].predict(cell, drug, scores, fingerprints),
            "per_drug_ridge": models["per_drug_ridge"].predict(cell, drug, scores),
        }

    first = predict(joblib.load(path))
    replay = predict(joblib.load(path))
    for name in BASELINE_NAMES:
        if first[name].shape != (len(frame),) or not np.isfinite(first[name]).all():
            raise ValueError(f"Invalid {name} test prediction")
        np.testing.assert_allclose(replay[name], first[name], rtol=0, atol=1e-10)
    return first


def mlp_predictions(root: Path, lock: dict, frame: pd.DataFrame,
                    scores: np.ndarray, fingerprints: np.ndarray) -> tuple[dict[str, np.ndarray], str]:
    # Torch is optional for the baseline-only synthetic CI path.
    import torch
    from neural import DrugResponseMLP, predict_rows

    config = json.loads((root / "configs/neural_mlp.json").read_text(encoding="utf-8"))
    if int(config["input_dim"]) != scores.shape[1] + fingerprints.shape[1]:
        raise ValueError("MLP feature dimensions changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell_features = torch.as_tensor(np.ascontiguousarray(scores, dtype=np.float32), device=device)
    drug_features = torch.as_tensor(np.ascontiguousarray(fingerprints, dtype=np.float32), device=device)
    cell = torch.as_tensor(frame["cell_index"].to_numpy(dtype=np.int64), device=device)
    drug = torch.as_tensor(frame["drug_index"].to_numpy(dtype=np.int64), device=device)
    result = {}
    for run in sorted(lock["selected_mlp_runs"], key=lambda item: int(item["seed"])):
        path = root / f"results/checkpoints/mlp_{run['run_id']}.pt"

        def predict_reload() -> np.ndarray:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            if (int(checkpoint["seed"]) != int(run["seed"])
                    or int(checkpoint["best_epoch"]) != int(run["best_epoch"])
                    or not np.isclose(float(checkpoint["learning_rate"]), float(run["learning_rate"]), rtol=0, atol=1e-12)
                    or int(checkpoint["input_dim"]) != int(config["input_dim"])
                    or tuple(checkpoint["hidden_layers"]) != tuple(config["hidden_layers"])
                    or float(checkpoint["dropout"]) != float(config["dropout_after_each_hidden_activation"])):
                raise ValueError("MLP checkpoint identity or architecture changed")
            model = DrugResponseMLP(
                int(config["input_dim"]), tuple(config["hidden_layers"]),
                float(config["dropout_after_each_hidden_activation"]),
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            prediction = predict_rows(
                model, cell, drug, cell_features, drug_features,
                int(config["evaluation_batch_size"]),
            ).numpy().astype(np.float64)
            del model
            return prediction

        name = f"mlp_seed_{int(run['seed'])}"
        first = predict_reload()
        replay = predict_reload()
        if first.shape != (len(frame),) or not np.isfinite(first).all():
            raise ValueError(f"Invalid {name} test prediction")
        np.testing.assert_allclose(replay, first, rtol=0, atol=1e-6)
        result[name] = first
    return result, device.type


def summarize(frame: pd.DataFrame, names: list[str], train_counts: dict[str, int],
              minimum_spearman_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["y"].to_numpy(dtype=np.float64)
    comparison = []
    drug_rows = []
    for name in names:
        predicted = frame[name].to_numpy(dtype=np.float64)
        rmse, mae = errors(y, predicted)
        detail = by_drug_validation(
            frame["drug_id"].to_numpy(), y, predicted, train_counts,
            minimum_spearman_n=minimum_spearman_n,
        ).rename(columns={"validation_n": "test_n"})
        if int(detail["test_n"].sum()) != len(frame):
            raise ValueError("Drug-level counts do not sum to test rows")
        detail.insert(0, "model", name)
        drug_rows.append(detail)
        defined = int(detail["spearman"].notna().sum())
        comparison.append({
            "model": name, "n_rows": len(frame), "n_cell_lines": frame["cell_line_id"].nunique(),
            "n_drugs": len(detail), "rmse": rmse, "mae": mae,
            "equal_drug_mean_rmse": float(detail["rmse"].mean()),
            "median_within_drug_spearman": float(detail["spearman"].median()) if defined else np.nan,
            "spearman_n_eligible_drugs": int((detail["test_n"] >= minimum_spearman_n).sum()),
            "spearman_defined_drugs": defined,
        })
    result = pd.DataFrame(comparison)
    if len(names) > len(BASELINE_NAMES):
        mlp = result.loc[result["model"].str.startswith("mlp_seed_")]
        if len(mlp) != 3:
            raise ValueError("Three selected MLP seeds required for mean summary")
        row = {
            "model": "mlp_three_seed_mean", "n_rows": len(frame),
            "n_cell_lines": frame["cell_line_id"].nunique(), "n_drugs": frame["drug_id"].nunique(),
            "spearman_n_eligible_drugs": int(mlp["spearman_n_eligible_drugs"].iloc[0]),
            "spearman_defined_drugs": int(mlp["spearman_defined_drugs"].iloc[0]),
        }
        for metric in ("rmse", "mae", "equal_drug_mean_rmse", "median_within_drug_spearman"):
            row[metric] = float(mlp[metric].mean())
            row[f"{metric}_sample_sd"] = float(mlp[metric].std(ddof=1))
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result, pd.concat(drug_rows, ignore_index=True)


def bootstrap_clusters(frame: pd.DataFrame, names: list[str], *, resamples: int,
                       seed: int) -> pd.DataFrame:
    """Use identical whole-cell draws for every model and paired RMSE delta."""
    cell_ids, row_cell = np.unique(frame["cell_line_id"].to_numpy(), return_inverse=True)
    n_cells = len(cell_ids)
    rng = np.random.default_rng(seed)
    sampled_cells = rng.integers(0, n_cells, size=(resamples, n_cells), endpoint=False)
    draws = np.zeros((resamples, n_cells), dtype=np.int32)
    for draw_index, one_draw in enumerate(sampled_cells):
        draws[draw_index] = np.bincount(one_draw, minlength=n_cells)
    if not np.array_equal(draws.sum(axis=1), np.full(resamples, n_cells)):
        raise ValueError("Bootstrap cluster multiplicities changed")
    rows_per_cell = np.bincount(row_cell, minlength=n_cells)
    denominator = draws @ rows_per_cell
    if np.any(denominator < 1):
        raise ValueError("Bootstrap produced an empty response draw")
    # Explicitly check one draw against repeated whole-cell rows, including
    # multiplicity when the same cell line is sampled more than once.
    repeated_rows = np.concatenate([np.flatnonzero(row_cell == cell) for cell in sampled_cells[0]])
    if len(repeated_rows) != int(denominator[0]):
        raise ValueError("Bootstrap failed to preserve repeated-cluster rows")
    y = frame["y"].to_numpy(dtype=np.float64)
    replicate = {}
    for name in names:
        residual = frame[name].to_numpy(dtype=np.float64) - y
        sse = np.bincount(row_cell, weights=np.square(residual), minlength=n_cells)
        sae = np.bincount(row_cell, weights=np.abs(residual), minlength=n_cells)
        replicate[f"{name}_rmse"] = np.sqrt((draws @ sse) / denominator)
        replicate[f"{name}_mae"] = (draws @ sae) / denominator
        np.testing.assert_allclose(
            replicate[f"{name}_rmse"][0],
            np.sqrt(np.mean(np.square(residual[repeated_rows]))), rtol=0, atol=1e-12,
        )
        np.testing.assert_allclose(
            replicate[f"{name}_mae"][0],
            np.mean(np.abs(residual[repeated_rows])), rtol=0, atol=1e-12,
        )
    report_names = names.copy()
    mlp_names = [name for name in names if name.startswith("mlp_seed_")]
    if mlp_names:
        if len(mlp_names) != 3:
            raise ValueError("MLP bootstrap summary requires three seeds")
        for metric in ("rmse", "mae"):
            replicate[f"mlp_three_seed_mean_{metric}"] = np.mean(
                [replicate[f"{name}_{metric}"] for name in mlp_names], axis=0,
            )
        report_names.append("mlp_three_seed_mean")
    rows = []
    reference = replicate["per_drug_ridge_rmse"]
    for name in report_names:
        rmse = replicate[f"{name}_rmse"]
        mae = replicate[f"{name}_mae"]
        delta = rmse - reference
        rows.append({
            "model": name,
            "rmse_ci_low": float(np.percentile(rmse, 2.5)),
            "rmse_ci_high": float(np.percentile(rmse, 97.5)),
            "mae_ci_low": float(np.percentile(mae, 2.5)),
            "mae_ci_high": float(np.percentile(mae, 97.5)),
            "paired_rmse_delta_vs_per_drug_ridge_ci_low": float(np.percentile(delta, 2.5)),
            "paired_rmse_delta_vs_per_drug_ridge_ci_high": float(np.percentile(delta, 97.5)),
        })
    return pd.DataFrame(rows)


def run(root: Path) -> dict:
    started = perf_counter()
    root = root.resolve()
    lock = verify_lock(root)  # Do not open test outcomes before this call.
    frame, _, scores, fingerprints, train_counts = load_test_inputs(root)
    predictions = frame[["source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "split", "y"]].copy()
    predictions_by_model = baseline_predictions(root, frame, scores, fingerprints)
    device = "none"
    if lock["mode"] == "primary":
        neural_predictions, device = mlp_predictions(root, lock, frame, scores, fingerprints)
        predictions_by_model.update(neural_predictions)
    names = list(BASELINE_NAMES) + [f"mlp_seed_{int(run['seed'])}" for run in sorted(lock["selected_mlp_runs"], key=lambda item: int(item["seed"]))]
    for name in names:
        predictions[name] = predictions_by_model[name]
    if not np.isfinite(predictions[names].to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite saved prediction")

    baseline_config = json.loads((root / "configs/baseline_ridge.json").read_text(encoding="utf-8"))
    minimum_n = int(baseline_config["minimum_within_drug_spearman_n"])
    comparison, by_drug = summarize(predictions, names, train_counts, minimum_n)
    intervals = bootstrap_clusters(
        predictions, names,
        resamples=int(lock["bootstrap"]["resamples"]),
        seed=int(lock["bootstrap"]["seed"]),
    )
    results = root / "results"
    temp = results / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results/tmp/fresh_test_predictions.csv": predictions,
        "results/fresh_test_comparison.csv": comparison,
        "results/fresh_test_by_drug.csv": by_drug,
        "results/fresh_bootstrap_intervals.csv": intervals,
    }
    for relative, table in outputs.items():
        table.to_csv(root / relative, index=False, float_format="%.17g", lineterminator="\n")
    replay = pd.read_csv(temp / "fresh_test_predictions.csv", float_precision="round_trip")
    if (not np.array_equal(replay["source_row_id"].to_numpy(), frame["source_row_id"].to_numpy())
            or not np.array_equal(replay["y"].to_numpy(), frame["y"].to_numpy())):
        raise ValueError("Saved test prediction labels or order changed")
    for name in names:
        np.testing.assert_allclose(replay[name], predictions[name], rtol=0, atol=1e-12)
        original = comparison.loc[comparison["model"] == name].iloc[0]
        rmse, mae = errors(replay["y"].to_numpy(), replay[name].to_numpy())
        np.testing.assert_allclose([rmse, mae], [original["rmse"], original["mae"]], rtol=0, atol=1e-12)

    manifest = {
        "schema_version": 1,
        "lock_sha256": sha256(results / "fresh_evaluation_lock.json"),
        "mode": lock["mode"],
        "test_rows": len(frame),
        "test_cell_lines": int(frame["cell_line_id"].nunique()),
        "test_drugs": int(frame["drug_id"].nunique()),
        "device": device,
        "bootstrap": lock["bootstrap"],
        "runtime_seconds": perf_counter() - started,
        "artifact_sha256": {relative: sha256(root / relative) for relative in outputs},
        "checks": [
            "frozen_hashes_before_test_outcomes", "validation_selected_checkpoint_identity",
            "test_feature_label_alignment", "cell_line_split_disjointness", "training_drug_coverage",
            "finite_predictions", "checkpoint_prediction_reload", "saved_prediction_replay",
            "per_drug_counts_sum", "paired_whole_cell_bootstrap_multiplicities",
        ],
        "interpretation": "Conditional on the fitted models and this split; MLP seed mean is a mean of metrics, not an ensemble.",
    }
    (results / "fresh_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(comparison[["model", "rmse", "mae", "equal_drug_mean_rmse", "median_within_drug_spearman"]].to_string(index=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Isolated fresh-run directory containing a pretest lock")
    arguments = parser.parse_args()
    run(arguments.run_dir)
