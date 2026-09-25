"""Prespecified PCA-dimension sensitivity on five fresh GDSC2 cell-line splits.

Validation selects a Ridge alpha separately for each model, dimension, and
split. All fitted artifacts are locked before this script scores any 64- or
256-component test prediction. Outputs belong in a new Git-ignored directory.
The original 128-component checkpoint in each fresh run is only a reference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baselines import (
    PerDrugRidge, PooledRidge, by_drug_validation, errors,
    fingerprint_row_basis, ridge_path,
)
from evaluate_fresh import verify_lock as verify_fresh_lock
from preprocessing import assign_cell_lines


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/robustness_pca.json"
DIMENSIONS = (64, 128, 256)
MODEL_NAMES = ("pooled_ridge", "per_drug_ridge")
IDENTITY_COLUMNS = ("source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "y")
INPUT_FILES = (
    "configs/preprocessing.json", "configs/baseline_ridge.json",
    "data/processed/cell_line_assignments.csv", "data/processed/drugs.csv",
    "data/processed/expression_features.csv", "data/processed/expression_raw.npy",
    "data/processed/expression_transform.joblib", "data/processed/expression_pca128_scaled.npy",
    "data/processed/morgan_radius2_2048.npy", "data/processed/primary_responses.csv",
    "results/preprocessing_manifest.json", "results/baseline_validation_manifest.json",
    "results/fresh_evaluation_lock.json", "results/fresh_evaluation_manifest.json",
    "results/checkpoints/baseline_models.joblib",
    "results/tmp/baseline_validation_predictions.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def primary_identity_sha256(rows: pd.DataFrame) -> str:
    """Hash the prespecified response identity while omitting split labels."""
    buffer = io.StringIO()
    rows.loc[:, list(IDENTITY_COLUMNS)].to_csv(
        buffer, index=False, float_format="%.17g", lineterminator="\n",
    )
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def validate_protocol(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported PCA study protocol")
    if config["split"]["seeds"] != [19, 23, 29, 31, 37]:
        raise ValueError("The prespecified split seeds changed")
    pca = config["pca_sensitivity"]
    if pca["components"] != list(DIMENSIONS) or pca["models"] != list(MODEL_NAMES):
        raise ValueError("The prespecified PCA arms changed")
    if pca["alpha_grid"] != [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        raise ValueError("The prespecified Ridge grid changed")
    if config["bootstrap"]["per_split_resamples"] != 2000 or config["bootstrap"]["seed"] != 2026:
        raise ValueError("The prespecified bootstrap changed")
    if config["fixed_models"]["data_table"] != "primary_non_Uprosertib" or config["fixed_models"]["target_transform"] != "none":
        raise ValueError("Primary table or target scale changed")


def verify_study_manifest(study_dir: Path, config: dict) -> dict:
    """Require the five completed fresh runs recorded by the split wrapper."""
    manifest = read_json(study_dir / "study_manifest.json")
    if (manifest.get("schema_version") != 1
            or manifest.get("protocol_sha256") != sha256(CONFIG_PATH)
            or manifest.get("split_seeds") != config["split"]["seeds"]
            or manifest.get("n_completed_splits") != len(config["split"]["seeds"])
            or manifest.get("primary_identity_sha256_excluding_split")
            != config["seed_invariant_primary_identity"]["sha256"]):
        raise ValueError("Completed split-study manifest differs from the prespecified protocol")
    if sha256(study_dir / "prefit_protocol_lock.json") != manifest["prefit_protocol_lock_sha256"]:
        raise ValueError("Split-study prefit lock changed")
    details = manifest.get("per_split")
    if not isinstance(details, list) or [item.get("seed") for item in details] != config["split"]["seeds"]:
        raise ValueError("Incomplete ordered split-study details")
    for item in details:
        seed = int(item["seed"])
        if item.get("run_directory") != f"seed_{seed}":
            raise ValueError("Unexpected split-study workspace path")
        run = study_dir / item["run_directory"]
        expected = {
            "preprocessing_config_sha256": "configs/preprocessing.json",
            "preprocessing_manifest_sha256": "results/preprocessing_manifest.json",
            "fresh_evaluation_lock_sha256": "results/fresh_evaluation_lock.json",
            "fresh_evaluation_manifest_sha256": "results/fresh_evaluation_manifest.json",
            "fresh_test_comparison_sha256": "results/fresh_test_comparison.csv",
            "fresh_bootstrap_intervals_sha256": "results/fresh_bootstrap_intervals.csv",
            "primary_response_sha256": "data/processed/primary_responses.csv",
        }
        for key, relative in expected.items():
            if sha256(run / relative) != item.get(key):
                raise ValueError(f"Seed {seed}: split-study manifest hash changed: {relative}")
        if item.get("primary_identity_sha256_excluding_split") != config["seed_invariant_primary_identity"]["sha256"]:
            raise ValueError(f"Seed {seed}: split-study source identity changed")
    return manifest


def validate_seed_input(run: Path, seed: int, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Check source identities, feature mappings, and saved fresh-run hashes."""
    verify_fresh_lock(run)  # verifies all generated inputs before fitted objects are loaded
    processed = run / "data/processed"
    manifest = read_json(run / "results/preprocessing_manifest.json")
    if manifest["raw_sha256"] != config["input_sha256"]:
        raise ValueError(f"Seed {seed}: raw source hashes differ")
    for name in ("expression_raw.npy", "expression_transform.joblib", "expression_features.csv", "primary_responses.csv"):
        if sha256(processed / name) != manifest["artifacts_sha256"][name]:
            raise ValueError(f"Seed {seed}: preprocessing artifact changed: {name}")
    assignments = pd.read_csv(processed / "cell_line_assignments.csv")
    drugs = pd.read_csv(processed / "drugs.csv")
    rows = pd.read_csv(processed / "primary_responses.csv", float_precision="round_trip")
    if primary_identity_sha256(rows) != config["seed_invariant_primary_identity"]["sha256"]:
        raise ValueError(f"Seed {seed}: split-independent primary response identity changed")
    if rows["source_row_id"].duplicated().any() or rows["drug_id"].eq("Uprosertib").any():
        raise ValueError(f"Seed {seed}: primary exclusion or row uniqueness changed")
    if not assignments["cell_line_id"].is_unique or list(assignments["cell_line_id"]) != sorted(assignments["cell_line_id"]):
        raise ValueError(f"Seed {seed}: cell-line catalog changed")
    if not drugs["drug_id"].is_unique:
        raise ValueError(f"Seed {seed}: drug catalog changed")
    expected = assign_cell_lines(
        assignments["cell_line_id"].tolist(), seed=seed,
        train_fraction=float(config["split"]["train_fraction"]),
        validation_fraction=float(config["split"]["validation_fraction"]),
    )
    if not assignments.equals(expected):
        raise ValueError(f"Seed {seed}: cell-line assignment differs from the prespecified rule")
    cell = rows["cell_index"].to_numpy(dtype=np.int64)
    drug = rows["drug_index"].to_numpy(dtype=np.int64)
    if (np.any(cell < 0) or np.any(cell >= len(assignments))
            or np.any(drug < 0) or np.any(drug >= len(drugs))):
        raise ValueError(f"Seed {seed}: response feature index out of bounds")
    if not np.array_equal(rows["cell_line_id"].to_numpy(), assignments.iloc[cell]["cell_line_id"].to_numpy()):
        raise ValueError(f"Seed {seed}: cell feature-label alignment failed")
    if not np.array_equal(rows["drug_id"].to_numpy(), drugs.iloc[drug]["drug_id"].to_numpy()):
        raise ValueError(f"Seed {seed}: drug feature-label alignment failed")
    if not np.array_equal(rows["split"].to_numpy(), assignments.iloc[cell]["split"].to_numpy()):
        raise ValueError(f"Seed {seed}: response split alignment failed")
    if not np.isfinite(rows["y"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Seed {seed}: nonfinite target")
    train_drugs = set(rows.loc[rows["split"].eq("train"), "drug_id"])
    if not set(rows["drug_id"]).issubset(train_drugs):
        raise ValueError(f"Seed {seed}: training-drug coverage failed")
    return assignments, drugs, rows, manifest


def fit_dimension_scores(
    standardized_genes: np.ndarray, train_cell_positions: np.ndarray,
    components: int, pca_settings: dict,
) -> tuple[np.ndarray, PCA, StandardScaler]:
    """Fit PCA and score scaling once per unique training cell line."""
    if standardized_genes.ndim != 2 or not np.isfinite(standardized_genes).all():
        raise ValueError("Nonfinite or invalid standardized gene matrix")
    if (train_cell_positions.ndim != 1 or len(np.unique(train_cell_positions)) != len(train_cell_positions)
            or np.any(train_cell_positions < 0) or np.any(train_cell_positions >= len(standardized_genes))):
        raise ValueError("PCA fit cell positions must be unique and valid")
    if components >= min(len(train_cell_positions), standardized_genes.shape[1]):
        raise ValueError("Insufficient train-cell count or genes for requested PCA")
    pca = PCA(
        n_components=components, svd_solver="randomized",
        random_state=int(pca_settings["pca_random_state"]),
        n_oversamples=int(pca_settings["pca_n_oversamples"]),
        iterated_power=int(pca_settings["pca_iterated_power"]),
        power_iteration_normalizer=str(pca_settings["pca_power_iteration_normalizer"]),
        whiten=False,
    )
    with threadpool_limits(limits=1):
        pca.fit(standardized_genes[train_cell_positions])
        scores = pca.transform(standardized_genes)
    score_scaler = StandardScaler(with_mean=True, with_std=True)
    score_scaler.fit(scores[train_cell_positions])
    scaled = score_scaler.transform(scores)
    np.testing.assert_allclose(
        score_scaler.mean_, scores[train_cell_positions].mean(axis=0), rtol=1e-13, atol=1e-13,
    )
    if scaled.shape != (len(standardized_genes), components) or not np.isfinite(scaled).all():
        raise ValueError("PCA score shape or finiteness failed")
    return scaled, pca, score_scaler


def fit_ridge_models(
    train: pd.DataFrame, validation: pd.DataFrame, scaled_scores: np.ndarray,
    fingerprints: np.ndarray, drug_scores: np.ndarray, basis: np.ndarray,
    alpha_grid: list[float], n_drugs: int,
) -> tuple[dict[str, object], dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Training-only Ridge fits; one validation-selected alpha per approach."""
    d = scaled_scores.shape[1]
    train_cell = train["cell_index"].to_numpy(dtype=np.int32)
    val_cell = validation["cell_index"].to_numpy(dtype=np.int32)
    train_drug = train["drug_index"].to_numpy(dtype=np.int32)
    val_drug = validation["drug_index"].to_numpy(dtype=np.int32)
    y_train = train["y"].to_numpy(dtype=np.float64)
    y_val = validation["y"].to_numpy(dtype=np.float64)
    trained = np.bincount(train_drug, minlength=n_drugs) > 0
    if not trained[val_drug].all():
        raise ValueError("Validation drug absent from training")
    with threadpool_limits(limits=1):
        x_train = np.column_stack((scaled_scores[train_cell], drug_scores[train_drug]))
        x_val = np.column_stack((scaled_scores[val_cell], drug_scores[val_drug]))
        coefficients, intercepts = ridge_path(x_train, y_train, alpha_grid)
        pooled_grid = x_val @ coefficients.T + intercepts
        pooled_index = int(np.argmin(np.sqrt(np.mean(np.square(pooled_grid - y_val[:, None]), axis=0))))
        pooled = PooledRidge(
            expression_coef=coefficients[pooled_index, :d].copy(),
            fingerprint_coef=(basis @ coefficients[pooled_index, d:]).copy(),
            intercept=float(intercepts[pooled_index]), alpha=float(alpha_grid[pooled_index]),
            trained_drugs=trained.copy(),
        )
        pooled_train = pooled.predict(train_cell, train_drug, scaled_scores, fingerprints)
        pooled_val = pooled.predict(val_cell, val_drug, scaled_scores, fingerprints)
        np.testing.assert_allclose(pooled_val, pooled_grid[:, pooled_index], rtol=0, atol=1e-8)
        del x_train, x_val, pooled_grid

        per_coef = np.zeros((len(alpha_grid), n_drugs, d), dtype=np.float64)
        per_intercept = np.full((len(alpha_grid), n_drugs), np.nan, dtype=np.float64)
        per_grid = np.full((len(alpha_grid), len(validation)), np.nan, dtype=np.float64)
        fallbacks: list[str] = []
        for drug_index in np.flatnonzero(trained):
            one_train = np.flatnonzero(train_drug == drug_index)
            one_val = np.flatnonzero(val_drug == drug_index)
            if len(one_train) < 2:
                fallbacks.append(str(train.iloc[one_train[0]]["drug_id"]))
                per_intercept[:, drug_index] = float(y_train[one_train].mean())
            else:
                one_coef, one_intercept = ridge_path(
                    scaled_scores[train_cell[one_train]], y_train[one_train], alpha_grid,
                )
                per_coef[:, drug_index] = one_coef
                per_intercept[:, drug_index] = one_intercept
            if len(one_val):
                per_grid[:, one_val] = (
                    per_coef[:, drug_index] @ scaled_scores[val_cell[one_val]].T
                    + per_intercept[:, drug_index, None]
                )
        if not np.isfinite(per_grid).all():
            raise ValueError("Incomplete per-drug validation predictions")
        per_index = int(np.argmin(np.sqrt(np.mean(np.square(per_grid - y_val[None, :]), axis=1))))
        per_drug = PerDrugRidge(
            expression_coef=per_coef[per_index].copy(),
            intercept=per_intercept[per_index].copy(),
            alpha=float(alpha_grid[per_index]), trained_drugs=trained.copy(),
        )
        per_train = per_drug.predict(train_cell, train_drug, scaled_scores)
        per_val = per_drug.predict(val_cell, val_drug, scaled_scores)
        np.testing.assert_allclose(per_val, per_grid[per_index], rtol=0, atol=1e-9)
    return (
        {"pooled_ridge": pooled, "per_drug_ridge": per_drug},
        {"pooled_ridge": float(alpha_grid[pooled_index]), "per_drug_ridge": float(alpha_grid[per_index])},
        {"pooled_ridge": pooled_train, "per_drug_ridge": per_train},
        {"pooled_ridge": pooled_val, "per_drug_ridge": per_val},
        fallbacks,
    )


def arm_metrics(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray], train_counts: dict[str, int],
    *, seed: int, components: int, phase: str, minimum_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["y"].to_numpy(dtype=np.float64)
    summary_rows = []
    detail_rows = []
    for model in MODEL_NAMES:
        predicted = predictions[model]
        rmse, mae = errors(y, predicted)
        detail = by_drug_validation(
            frame["drug_id"].to_numpy(), y, predicted, train_counts,
            minimum_spearman_n=minimum_n,
        )
        if int(detail["validation_n"].sum()) != len(frame):
            raise ValueError("Drug-level counts do not sum to evaluated rows")
        detail = detail.rename(columns={"validation_n": "evaluation_n"})
        detail.insert(0, "phase", phase)
        detail.insert(0, "model", model)
        detail.insert(0, "pca_components", components)
        detail.insert(0, "split_seed", seed)
        detail_rows.append(detail)
        defined = int(detail["spearman"].notna().sum())
        summary_rows.append({
            "split_seed": seed, "pca_components": components, "phase": phase,
            "model": model, "n_rows": len(frame), "n_cell_lines": frame["cell_line_id"].nunique(),
            "n_drugs": len(detail), "rmse": rmse, "mae": mae,
            "equal_drug_mean_rmse": float(detail["rmse"].mean()),
            "median_within_drug_spearman": float(detail["spearman"].median()) if defined else np.nan,
            "spearman_eligible_drugs": int((detail["evaluation_n"] >= minimum_n).sum()),
            "spearman_defined_drugs": defined,
        })
    return pd.DataFrame(summary_rows), pd.concat(detail_rows, ignore_index=True)


def paired_bootstrap(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray], *, resamples: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired whole-cell intervals for each dimension minus 128, per model."""
    cells, row_cell = np.unique(frame["cell_line_id"].to_numpy(), return_inverse=True)
    n_cells = len(cells)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_cells, size=(resamples, n_cells), endpoint=False)
    draws = np.zeros((resamples, n_cells), dtype=np.int32)
    for index, one in enumerate(sampled):
        draws[index] = np.bincount(one, minlength=n_cells)
    if not np.array_equal(draws.sum(axis=1), np.full(resamples, n_cells)):
        raise ValueError("Bootstrap whole-cell multiplicities changed")
    row_counts = np.bincount(row_cell, minlength=n_cells)
    denominator = draws @ row_counts
    if np.any(denominator < 1):
        raise ValueError("Empty bootstrap draw")
    first_rows = np.concatenate([np.flatnonzero(row_cell == cell) for cell in sampled[0]])
    if len(first_rows) != int(denominator[0]):
        raise ValueError("Bootstrap first-draw cluster expansion failed")
    y = frame["y"].to_numpy(dtype=np.float64)
    replicate = {}
    interval_rows = []
    for name, estimate in predictions.items():
        residual = estimate - y
        if residual.shape != y.shape or not np.isfinite(residual).all():
            raise ValueError("Bootstrap prediction alignment or finiteness failed")
        cluster_sse = np.bincount(row_cell, weights=np.square(residual), minlength=n_cells)
        cluster_sae = np.bincount(row_cell, weights=np.abs(residual), minlength=n_cells)
        replicate[f"{name}_rmse"] = np.sqrt((draws @ cluster_sse) / denominator)
        replicate[f"{name}_mae"] = (draws @ cluster_sae) / denominator
        np.testing.assert_allclose(
            replicate[f"{name}_rmse"][0],
            np.sqrt(np.mean(np.square(residual[first_rows]))), rtol=0, atol=1e-12,
        )
    for model in MODEL_NAMES:
        reference = replicate[f"{model}_pca128_rmse"]
        for dimension in DIMENSIONS:
            name = f"{model}_pca{dimension}"
            rmse = replicate[f"{name}_rmse"]
            mae = replicate[f"{name}_mae"]
            delta = rmse - reference
            interval_rows.append({
                "model": model, "pca_components": dimension,
                "rmse_ci_low": float(np.percentile(rmse, 2.5)),
                "rmse_ci_high": float(np.percentile(rmse, 97.5)),
                "mae_ci_low": float(np.percentile(mae, 2.5)),
                "mae_ci_high": float(np.percentile(mae, 97.5)),
                "paired_rmse_delta_vs_pca128_ci_low": float(np.percentile(delta, 2.5)),
                "paired_rmse_delta_vs_pca128_ci_high": float(np.percentile(delta, 97.5)),
            })
    return pd.DataFrame(interval_rows), pd.DataFrame(draws, columns=cells)


def artifact_map(study_dir: Path, output_dir: Path, config: dict) -> tuple[dict[str, str], dict[str, str]]:
    inputs = {"configs/robustness_pca.json": sha256(CONFIG_PATH)}
    study_manifest = study_dir / "study_manifest.json"
    if study_manifest.exists():
        inputs["study_manifest.json"] = sha256(study_manifest)
    generated = {}
    for seed in config["split"]["seeds"]:
        prefix = f"seed_{seed}"
        for relative in INPUT_FILES:
            inputs[f"{prefix}/{relative}"] = sha256(study_dir / prefix / relative)
        for dimension in (64, 256):
            for filename in ("scores.npy", "transforms.joblib", "models.joblib", "validation_predictions.csv"):
                relative = f"{prefix}/pca_{dimension}/{filename}"
                generated[relative] = sha256(output_dir / relative)
    for filename in ("validation_metrics.csv", "validation_by_drug.csv", "pca_fit_summary.csv"):
        generated[filename] = sha256(output_dir / filename)
    return inputs, generated


def lock_validation(study_dir: Path, output_dir: Path, config: dict, selection: list[dict]) -> Path:
    input_hashes, generated_hashes = artifact_map(study_dir, output_dir, config)
    source_files = ("src/run_pca_sensitivity.py", "src/baselines.py", "src/preprocessing.py", "src/evaluate_fresh.py")
    lock = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": config["study_id"],
        "split_seeds": config["split"]["seeds"],
        "pca_components": config["pca_sensitivity"]["components"],
        "validation_selected_alpha": selection,
        "input_sha256": input_hashes,
        "generated_sha256": generated_hashes,
        "source_sha256": {relative: sha256(ROOT / relative) for relative in source_files},
        "test_status": "not_scored_by_pca_sensitivity_at_lock_creation",
        "interpretation": "Exploratory internal robustness on overlapping cell-line splits of an already public GDSC2 snapshot.",
    }
    path = output_dir / "pretest_lock.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def verify_pretest_lock(study_dir: Path, output_dir: Path, config: dict) -> dict:
    lock = read_json(output_dir / "pretest_lock.json")
    if (lock.get("schema_version") != 1 or lock.get("study_id") != config["study_id"]
            or lock.get("split_seeds") != config["split"]["seeds"]
            or lock.get("pca_components") != config["pca_sensitivity"]["components"]):
        raise ValueError("PCA pretest lock protocol identity changed")
    for relative, expected in lock["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"PCA source changed after validation lock: {relative}")
    for relative, expected in lock["input_sha256"].items():
        path = CONFIG_PATH if relative == "configs/robustness_pca.json" else study_dir / relative
        if sha256(path) != expected:
            raise ValueError(f"PCA study input changed after validation lock: {relative}")
    for relative, expected in lock["generated_sha256"].items():
        if sha256(output_dir / relative) != expected:
            raise ValueError(f"PCA fitted artifact changed after validation lock: {relative}")
    return lock


def validation_phase(study_dir: Path, output_dir: Path, config: dict) -> Path:
    all_metrics = []
    all_details = []
    fit_rows = []
    selections = []
    for seed in config["split"]["seeds"]:
        started = perf_counter()
        run = study_dir / f"seed_{seed}"
        if not run.is_dir():
            raise FileNotFoundError(f"Missing prespecified fresh-run workspace: {run}")
        assignments, drugs, rows, prep_manifest = validate_seed_input(run, seed, config)
        processed = run / "data/processed"
        train = rows.loc[rows["split"].eq("train")].reset_index(drop=True)
        validation = rows.loc[rows["split"].eq("validation")].reset_index(drop=True)
        train_cell_positions = np.flatnonzero(assignments["split"].eq("train").to_numpy())
        train_cell_ids = tuple(assignments.iloc[train_cell_positions]["cell_line_id"].tolist())
        gene_transform = joblib.load(processed / "expression_transform.joblib")
        if tuple(sorted(gene_transform.train_cell_ids)) != train_cell_ids:
            raise ValueError(f"Seed {seed}: gene preprocessing fitted on different cells")
        catalog = pd.read_csv(processed / "expression_features.csv")
        if tuple(catalog["feature_id"]) != gene_transform.feature_ids:
            raise ValueError(f"Seed {seed}: positional gene-feature ordering changed")
        raw = np.load(processed / "expression_raw.npy", allow_pickle=False, mmap_mode="r")
        if raw.shape != (len(assignments), len(catalog)):
            raise ValueError(f"Seed {seed}: raw expression shape changed")
        standardized_genes = gene_transform.scaler.transform(gene_transform.selector.transform(raw))
        if not np.isfinite(standardized_genes).all():
            raise ValueError(f"Seed {seed}: nonfinite standardized expression")
        fingerprints = np.load(processed / "morgan_radius2_2048.npy", allow_pickle=False)
        if fingerprints.shape != (len(drugs), 2048) or not np.isin(fingerprints, [0, 1]).all():
            raise ValueError(f"Seed {seed}: fingerprint representation changed")
        train_drug_index = train["drug_index"].to_numpy(dtype=np.int32)
        trained = np.bincount(train_drug_index, minlength=len(drugs)) > 0
        baseline_config = read_json(run / "configs/baseline_ridge.json")
        alpha_grid = [float(value) for value in config["pca_sensitivity"]["alpha_grid"]]
        if alpha_grid != [float(value) for value in baseline_config["alpha_grid"]]:
            raise ValueError(f"Seed {seed}: Ridge alpha grid differs from prespecified grid")
        with threadpool_limits(limits=1):
            basis = fingerprint_row_basis(
                fingerprints[np.flatnonzero(trained)],
                float(baseline_config["fingerprint_basis_relative_tolerance"]),
            )
            drug_scores = fingerprints.astype(np.float64) @ basis
        train_counts = train.groupby("drug_id").size().to_dict()
        minimum_n = int(baseline_config["minimum_within_drug_spearman_n"])

        # The 128-component arm is the actual saved fit for this seed.
        scores128 = np.load(processed / "expression_pca128_scaled.npy", allow_pickle=False)
        baseline_manifest = read_json(run / "results/baseline_validation_manifest.json")
        baseline_models = joblib.load(run / "results/checkpoints/baseline_models.joblib")
        saved_val = pd.read_csv(run / "results/tmp/baseline_validation_predictions.csv", float_precision="round_trip")
        if (not np.array_equal(saved_val["source_row_id"].to_numpy(), validation["source_row_id"].to_numpy())
                or not np.array_equal(saved_val["y"].to_numpy(), validation["y"].to_numpy())):
            raise ValueError(f"Seed {seed}: saved 128-component validation rows changed")
        train_cell = train["cell_index"].to_numpy(dtype=np.int32)
        train_drug = train["drug_index"].to_numpy(dtype=np.int32)
        val_cell = validation["cell_index"].to_numpy(dtype=np.int32)
        val_drug = validation["drug_index"].to_numpy(dtype=np.int32)
        anchor_train = {
            "pooled_ridge": baseline_models["pooled_ridge"].predict(train_cell, train_drug, scores128, fingerprints),
            "per_drug_ridge": baseline_models["per_drug_ridge"].predict(train_cell, train_drug, scores128),
        }
        anchor_val = {
            "pooled_ridge": baseline_models["pooled_ridge"].predict(val_cell, val_drug, scores128, fingerprints),
            "per_drug_ridge": baseline_models["per_drug_ridge"].predict(val_cell, val_drug, scores128),
        }
        for model in MODEL_NAMES:
            np.testing.assert_allclose(anchor_val[model], saved_val[model], rtol=0, atol=1e-9)
            if baseline_models[model].alpha != baseline_manifest["selected_alpha"][model]:
                raise ValueError(f"Seed {seed}: 128-component alpha identity changed")
            selections.append({"split_seed": seed, "pca_components": 128, "model": model,
                               "alpha": float(baseline_models[model].alpha)})
        for phase, frame, estimates in (("train", train, anchor_train), ("validation", validation, anchor_val)):
            summary, detail = arm_metrics(
                frame, estimates, train_counts, seed=seed, components=128,
                phase=phase, minimum_n=minimum_n,
            )
            all_metrics.append(summary)
            if phase == "validation":
                all_details.append(detail)
        fit_rows.append({"split_seed": seed, "pca_components": 128,
                         "retained_training_variance_fraction": prep_manifest["expression_preprocessing"]["retained_training_variance_fraction"],
                         "fitted_unique_training_cells": len(train_cell_positions),
                         "zero_variance_removed": prep_manifest["expression_preprocessing"]["zero_variance_removed"],
                         "source": "saved_fresh_run_baseline", "runtime_seconds": np.nan})

        for dimension in (64, 256):
            dimension_started = perf_counter()
            dimension_dir = output_dir / f"seed_{seed}/pca_{dimension}"
            dimension_dir.mkdir(parents=True)
            scaled, pca, score_scaler = fit_dimension_scores(
                standardized_genes, train_cell_positions, dimension,
                config["pca_sensitivity"],
            )
            scores_path = dimension_dir / "scores.npy"
            transform_path = dimension_dir / "transforms.joblib"
            models_path = dimension_dir / "models.joblib"
            validation_path = dimension_dir / "validation_predictions.csv"
            np.save(scores_path, scaled, allow_pickle=False)
            joblib.dump({"pca": pca, "score_scaler": score_scaler,
                         "training_cell_ids": train_cell_ids,
                         "source_gene_transform_sha256": sha256(processed / "expression_transform.joblib")},
                        transform_path, compress=0)
            models, alphas, train_predictions, val_predictions, fallbacks = fit_ridge_models(
                train, validation, scaled, fingerprints, drug_scores, basis, alpha_grid, len(drugs),
            )
            joblib.dump(models, models_path, compress=0)
            loaded_scores = np.load(scores_path, allow_pickle=False)
            loaded_models = joblib.load(models_path)
            for model in MODEL_NAMES:
                replay = (
                    loaded_models[model].predict(val_cell, val_drug, loaded_scores, fingerprints)
                    if model == "pooled_ridge" else
                    loaded_models[model].predict(val_cell, val_drug, loaded_scores)
                )
                np.testing.assert_allclose(replay, val_predictions[model], rtol=0, atol=1e-9)
                selections.append({"split_seed": seed, "pca_components": dimension,
                                   "model": model, "alpha": alphas[model]})
            saved = validation[list(IDENTITY_COLUMNS) + ["split"]].copy()
            for model in MODEL_NAMES:
                saved[model] = val_predictions[model]
            write_csv(saved, validation_path)
            replay = pd.read_csv(validation_path, float_precision="round_trip")
            for column in IDENTITY_COLUMNS + ("split",):
                if not np.array_equal(replay[column].to_numpy(), validation[column].to_numpy()):
                    raise ValueError(f"Seed {seed}, PCA {dimension}: validation {column} replay changed")
            for model in MODEL_NAMES:
                np.testing.assert_allclose(replay[model], val_predictions[model], rtol=0, atol=1e-12)
            for phase, frame, estimates in (("train", train, train_predictions), ("validation", validation, val_predictions)):
                summary, detail = arm_metrics(
                    frame, estimates, train_counts, seed=seed, components=dimension,
                    phase=phase, minimum_n=minimum_n,
                )
                all_metrics.append(summary)
                if phase == "validation":
                    all_details.append(detail)
            fit_rows.append({"split_seed": seed, "pca_components": dimension,
                             "retained_training_variance_fraction": float(pca.explained_variance_ratio_.sum()),
                             "fitted_unique_training_cells": len(train_cell_positions),
                             "zero_variance_removed": int(raw.shape[1] - gene_transform.selector.get_support().sum()),
                             "source": "new_train_only_fit", "runtime_seconds": perf_counter() - dimension_started,
                             "fallback_drugs": "|".join(fallbacks)})
            print(f"Seed {seed}, PCA {dimension}: validation fits complete in {perf_counter()-dimension_started:.1f}s", flush=True)
        print(f"Seed {seed}: all PCA validation arms complete in {perf_counter()-started:.1f}s", flush=True)
        del raw, standardized_genes

    write_csv(pd.concat(all_metrics, ignore_index=True), output_dir / "validation_metrics.csv")
    write_csv(pd.concat(all_details, ignore_index=True), output_dir / "validation_by_drug.csv")
    write_csv(pd.DataFrame(fit_rows), output_dir / "pca_fit_summary.csv")
    return lock_validation(study_dir, output_dir, config, selections)


def test_phase(study_dir: Path, output_dir: Path, config: dict) -> dict:
    started = perf_counter()
    lock = verify_pretest_lock(study_dir, output_dir, config)
    all_metrics = []
    all_details = []
    all_intervals = []
    paired_rows = []
    output_hashes = {}
    for seed in config["split"]["seeds"]:
        run = study_dir / f"seed_{seed}"
        assignments, drugs, rows, _ = validate_seed_input(run, seed, config)
        test = rows.loc[rows["split"].eq("test")].reset_index(drop=True)
        train_counts = rows.loc[rows["split"].eq("train")].groupby("drug_id").size().to_dict()
        if test.empty or test["cell_line_id"].nunique() != int(assignments["split"].eq("test").sum()):
            raise ValueError(f"Seed {seed}: test cluster coverage changed")
        processed = run / "data/processed"
        fingerprints = np.load(processed / "morgan_radius2_2048.npy", allow_pickle=False)
        baseline_models = joblib.load(run / "results/checkpoints/baseline_models.joblib")
        cell = test["cell_index"].to_numpy(dtype=np.int32)
        drug = test["drug_index"].to_numpy(dtype=np.int32)
        predictions = {}
        seed_output = output_dir / f"seed_{seed}"
        for dimension in DIMENSIONS:
            if dimension == 128:
                scores = np.load(processed / "expression_pca128_scaled.npy", allow_pickle=False)
                models = baseline_models
            else:
                dimension_dir = seed_output / f"pca_{dimension}"
                scores = np.load(dimension_dir / "scores.npy", allow_pickle=False)
                models = joblib.load(dimension_dir / "models.joblib")
            if scores.shape != (len(assignments), dimension) or not np.isfinite(scores).all():
                raise ValueError(f"Seed {seed}, PCA {dimension}: test feature shape or finiteness failed")
            estimates = {
                "pooled_ridge": models["pooled_ridge"].predict(cell, drug, scores, fingerprints),
                "per_drug_ridge": models["per_drug_ridge"].predict(cell, drug, scores),
            }
            for model in MODEL_NAMES:
                predictions[f"{model}_pca{dimension}"] = estimates[model]
            baseline_config = read_json(run / "configs/baseline_ridge.json")
            summary, detail = arm_metrics(
                test, estimates, train_counts, seed=seed, components=dimension,
                phase="test", minimum_n=int(baseline_config["minimum_within_drug_spearman_n"]),
            )
            all_metrics.append(summary)
            all_details.append(detail)
        saved = test[list(IDENTITY_COLUMNS) + ["split"]].copy()
        for name, prediction in predictions.items():
            saved[name] = prediction
        prediction_path = seed_output / "test_predictions.csv"
        write_csv(saved, prediction_path)
        replay = pd.read_csv(prediction_path, float_precision="round_trip")
        for column in IDENTITY_COLUMNS + ("split",):
            if not np.array_equal(replay[column].to_numpy(), test[column].to_numpy()):
                raise ValueError(f"Seed {seed}: saved test {column} replay changed")
        for name, prediction in predictions.items():
            np.testing.assert_allclose(replay[name], prediction, rtol=0, atol=1e-12)
        intervals, _ = paired_bootstrap(
            test, predictions, resamples=int(config["bootstrap"]["per_split_resamples"]),
            seed=int(config["bootstrap"]["seed"]),
        )
        intervals.insert(0, "split_seed", seed)
        all_intervals.append(intervals)
        one_metrics = pd.concat(all_metrics, ignore_index=True)
        one_metrics = one_metrics.loc[one_metrics["split_seed"].eq(seed)]
        for model in MODEL_NAMES:
            for dimension in DIMENSIONS:
                name = f"{model}_pca{dimension}"
                reported = one_metrics.loc[(one_metrics["model"] == model)
                                           & (one_metrics["pca_components"] == dimension)].iloc[0]
                np.testing.assert_allclose(
                    errors(replay["y"].to_numpy(dtype=np.float64), replay[name].to_numpy(dtype=np.float64)),
                    [reported["rmse"], reported["mae"]], rtol=0, atol=1e-12,
                )
        for model in MODEL_NAMES:
            reference = float(one_metrics.loc[(one_metrics["model"] == model)
                                             & (one_metrics["pca_components"] == 128), "rmse"].iloc[0])
            for dimension in (64, 256):
                candidate = float(one_metrics.loc[(one_metrics["model"] == model)
                                                  & (one_metrics["pca_components"] == dimension), "rmse"].iloc[0])
                paired_rows.append({"split_seed": seed, "model": model,
                                    "pca_components": dimension,
                                    "rmse_difference_minus_pca128": candidate - reference})
        output_hashes[f"seed_{seed}/test_predictions.csv"] = sha256(prediction_path)
        print(f"Seed {seed}: paired PCA test scores complete", flush=True)

    metrics = pd.concat(all_metrics, ignore_index=True)
    details = pd.concat(all_details, ignore_index=True)
    intervals = pd.concat(all_intervals, ignore_index=True)
    differences = pd.DataFrame(paired_rows)
    summaries = []
    for (model, dimension), group in metrics.groupby(["model", "pca_components"], sort=True):
        summaries.append({
            "model": model, "pca_components": int(dimension), "n_splits": len(group),
            "rmse_median": float(group["rmse"].median()),
            "rmse_min": float(group["rmse"].min()),
            "rmse_max": float(group["rmse"].max()),
            "mae_median": float(group["mae"].median()),
            "equal_drug_mean_rmse_median": float(group["equal_drug_mean_rmse"].median()),
            "median_within_drug_spearman_median": float(group["median_within_drug_spearman"].median()),
            "wins_vs_pca128": int((differences.loc[
                differences["model"].eq(model) & differences["pca_components"].eq(dimension),
                "rmse_difference_minus_pca128"] < 0).sum()) if dimension != 128 else np.nan,
        })
    outputs = {
        "test_metrics.csv": metrics,
        "test_by_drug.csv": details,
        "test_bootstrap_intervals.csv": intervals,
        "test_paired_differences.csv": differences,
        "test_across_split_summary.csv": pd.DataFrame(summaries),
    }
    for relative, table in outputs.items():
        write_csv(table, output_dir / relative)
        output_hashes[relative] = sha256(output_dir / relative)
    manifest = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "pretest_lock_sha256": sha256(output_dir / "pretest_lock.json"),
        "split_seeds": config["split"]["seeds"],
        "dimensions": list(DIMENSIONS),
        "bootstrap": config["bootstrap"],
        "runtime_seconds": perf_counter() - started,
        "test_output_sha256": output_hashes,
        "checks": [
            "all_fitted_artifact_hashes_before_test_scoring", "prespecified_split_assignments",
            "seed_invariant_response_identity", "training_only_gene_and_PCA_score_scaling",
            "validation_only_alpha_selection", "128_component_saved_checkpoint_replay",
            "test_row_and_feature_alignment", "saved_test_prediction_replay",
            "paired_whole_cell_bootstrap_multiplicity",
        ],
        "interpretation": "Exploratory internal robustness: five overlapping splits of the same public GDSC2 snapshot; intervals are conditional on each split and fitted models, not independent study replication.",
    }
    (output_dir / "pca_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True,
                        help="Completed multisplit workspace containing seed_19 through seed_37")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for ignored PCA artifacts; must not exist")
    args = parser.parse_args()
    study_dir = args.study_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    data_root = (ROOT / "data").resolve()
    if not output_dir.is_relative_to(data_root) or output_dir == data_root:
        parser.error("PCA output directory must be a new Git-ignored directory under repository data/")
    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}")
    if output_dir == study_dir or output_dir.is_relative_to(study_dir):
        parser.error("PCA output directory must be separate from the multisplit workspace")
    config = read_json(CONFIG_PATH)
    validate_protocol(config)
    verify_study_manifest(study_dir, config)
    output_dir.mkdir(parents=True)
    lock = validation_phase(study_dir, output_dir, config)
    print(f"PCA validation selections locked: {lock}", flush=True)
    manifest = test_phase(study_dir, output_dir, config)
    print(json.dumps({"study_id": manifest["study_id"],
                      "test_metrics": str(output_dir / "test_metrics.csv"),
                      "runtime_seconds": manifest["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
