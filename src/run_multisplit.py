"""Run the prespecified five additional cell-line splits in isolated workspaces.

The original seed-17 study is a historical reference. These overlapping splits
of the same public snapshot are exploratory robustness checks, not independent
external validation. The tracked protocol is locked before any new fit, while
each fresh run locks validation selections before its own test scoring.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from run_fresh import RAW_NAMES, SOURCE_ROOT, json_file, sha256


PROTOCOL_PATH = SOURCE_ROOT / "configs/robustness_pca.json"
PREPROCESSING_PATH = SOURCE_ROOT / "configs/preprocessing.json"
SPLIT_SEEDS = (19, 23, 29, 31, 37)
BASELINE_MODELS = ("per_drug_mean", "pooled_ridge", "per_drug_ridge")
PRIMARY_MODELS = BASELINE_MODELS + (
    "mlp_seed_17", "mlp_seed_29", "mlp_seed_43", "mlp_three_seed_mean",
)
IDENTITY_COLUMNS = (
    "source_row_id", "drug_id", "cell_line_id", "drug_index", "cell_index", "y",
)
SOURCE_NAMES = (
    "run_multisplit.py", "run_fresh.py", "prepare_gdsc2.py", "run_baselines.py",
    "run_neural.py", "evaluate_fresh.py", "preprocessing.py", "baselines.py", "neural.py",
)


def source_hashes() -> dict[str, str]:
    return {f"src/{name}": sha256(SOURCE_ROOT / "src" / name) for name in SOURCE_NAMES}


def primary_identity_digest(path: Path) -> tuple[str, int]:
    """Hash seed-invariant row identity and labels; the split column differs."""
    frame = pd.read_csv(path, float_precision="round_trip")
    if not set(IDENTITY_COLUMNS).issubset(frame):
        raise ValueError("Primary table lacks invariant identity columns")
    if frame["source_row_id"].duplicated().any():
        raise ValueError("Primary source-row identifiers are not unique")
    columns = frame.loc[:, list(IDENTITY_COLUMNS)]
    payload = columns.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(frame)


def verify_protocol(protocol: dict, preprocessing: dict, *, synthetic_smoke: bool) -> None:
    if protocol.get("schema_version") != 1 or tuple(protocol["split"]["seeds"]) != SPLIT_SEEDS:
        raise ValueError("Protocol does not name the five prespecified split seeds")
    if protocol["input_sha256"] != preprocessing["input_sha256"]:
        raise ValueError("Protocol and preprocessing config disagree on raw input hashes")
    base = json_file(PREPROCESSING_PATH)
    if {key: value for key, value in preprocessing.items() if key != "input_sha256"} != {
        key: value for key, value in base.items() if key != "input_sha256"
    }:
        raise ValueError("Preprocessing config changes more than raw input hashes")
    if (protocol["fixed_models"]["expression_pca_components"] != 128
            or preprocessing["expression"]["pca_components"] != 128
            or preprocessing["split"]["seed"] != 17):
        raise ValueError("Fixed PCA dimension or starting split changed")
    if protocol["fixed_models"]["data_table"] != "primary_non_Uprosertib":
        raise ValueError("Unexpected analysis table in protocol")
    identity = protocol.get("seed_invariant_primary_identity")
    if not isinstance(identity, dict) or tuple(identity.get("columns", ())) != IDENTITY_COLUMNS:
        raise ValueError("Protocol does not define the seed-invariant source-row identity")
    if not synthetic_smoke and (
        not isinstance(identity.get("sha256"), str) or len(identity["sha256"]) != 64
    ):
        raise ValueError("GDSC2 protocol lacks the source-row identity checksum")
    if synthetic_smoke:
        if protocol["input_sha256"] == base["input_sha256"]:
            raise ValueError("Synthetic smoke must use invented data, not the published raw files")
    elif protocol["input_sha256"] != base["input_sha256"]:
        raise ValueError("Real robustness study must use the prespecified GDSC2 snapshot")


def verify_split_workspace(run_dir: Path, *, seed: int, protocol: dict, models: str,
                           expected_identity_digest: str | None,
                           bootstrap_draws: int) -> tuple[dict, pd.DataFrame, str]:
    config = json_file(run_dir / "configs/preprocessing.json")
    lock_path = run_dir / "results/fresh_evaluation_lock.json"
    lock = json_file(lock_path)
    eval_manifest_path = run_dir / "results/fresh_evaluation_manifest.json"
    eval_manifest = json_file(eval_manifest_path)
    prep = json_file(run_dir / "results/preprocessing_manifest.json")
    if config["split"]["seed"] != seed or lock.get("split_seed") != seed:
        raise ValueError(f"Run-local split seed does not match requested seed {seed}")
    if config["input_sha256"] != protocol["input_sha256"] or lock["raw_sha256"] != protocol["input_sha256"]:
        raise ValueError(f"Raw provenance changed for split seed {seed}")
    if prep["config_sha256"] != sha256(run_dir / "configs/preprocessing.json"):
        raise ValueError(f"Prepared configuration changed for split seed {seed}")
    if lock["mode"] != models or eval_manifest["lock_sha256"] != sha256(lock_path):
        raise ValueError(f"Model mode or evaluation lock differs for split seed {seed}")
    if int(lock["bootstrap"]["resamples"]) != bootstrap_draws:
        raise ValueError("Split bootstrap differs from the prefit protocol")
    if prep["counts"]["primary_excluded_rows"] < 1 or prep["counts"]["primary_drugs"] < 1:
        raise ValueError("Primary exclusion or drug coverage is missing")
    if prep["counts"]["primary_rows"] != sum(prep["counts"]["primary_rows_by_split"].values()):
        raise ValueError("Primary response counts do not cover all splits")
    if models == "primary" and prep["counts"]["split_cell_lines"] != {
        "test": 120, "train": 564, "validation": 121,
    }:
        raise ValueError("GDSC2 grouped split cell counts changed")
    identity_digest, n_rows = primary_identity_digest(run_dir / "data/processed/primary_responses.csv")
    if expected_identity_digest is not None and identity_digest != expected_identity_digest:
        raise ValueError(f"Seed-invariant source/target rows differ for split seed {seed}")
    protocol_digest = protocol["seed_invariant_primary_identity"].get("sha256")
    if protocol_digest is not None and identity_digest != protocol_digest:
        raise ValueError(f"Primary source-row identity differs from the prespecified protocol for seed {seed}")
    if n_rows != prep["counts"]["primary_rows"]:
        raise ValueError("Primary table count differs from preparation manifest")

    assignments = pd.read_csv(run_dir / "data/processed/cell_line_assignments.csv")
    primary = pd.read_csv(
        run_dir / "data/processed/primary_responses.csv",
        usecols=["drug_id", "cell_line_id", "split"],
    )
    if not assignments["cell_line_id"].is_unique or set(assignments["split"]) != {"train", "validation", "test"}:
        raise ValueError("Cell-line split assignment is incomplete")
    ordered = np.array(sorted(assignments["cell_line_id"].tolist()))
    n_train = round(float(config["split"]["train_fraction"]) * len(ordered))
    n_validation = round(float(config["split"]["validation_fraction"]) * len(ordered))
    shuffled = np.random.default_rng(seed).permutation(len(ordered))
    expected_split = np.empty(len(ordered), dtype=object)
    expected_split[shuffled[:n_train]] = "train"
    expected_split[shuffled[n_train:n_train + n_validation]] = "validation"
    expected_split[shuffled[n_train + n_validation:]] = "test"
    if (assignments["cell_line_id"].tolist() != ordered.tolist()
            or assignments["split"].tolist() != expected_split.tolist()):
        raise ValueError("Saved grouped assignments differ from the fixed ID-only split rule")
    if not (primary.groupby("cell_line_id")["split"].nunique() == 1).all():
        raise ValueError("Responses from a cell line cross splits")
    if not (primary.groupby("drug_id")["split"].apply(lambda values: "train" in set(values))).all():
        raise ValueError("A drug lacks training representation")
    counts = primary.groupby(["drug_id", "split"]).size().unstack(fill_value=0)
    comparison_path = run_dir / "results/fresh_test_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    intervals_path = run_dir / "results/fresh_bootstrap_intervals.csv"
    intervals = pd.read_csv(intervals_path)
    expected_models = set(PRIMARY_MODELS if models == "primary" else BASELINE_MODELS)
    if set(comparison["model"]) != expected_models or len(comparison) != len(expected_models):
        raise ValueError(f"Incomplete test model comparison for split seed {seed}")
    if set(intervals["model"]) != expected_models or len(intervals) != len(expected_models):
        raise ValueError(f"Incomplete paired cell-line intervals for split seed {seed}")
    required_interval_columns = {
        "rmse_ci_low", "rmse_ci_high", "mae_ci_low", "mae_ci_high",
        "paired_rmse_delta_vs_per_drug_ridge_ci_low",
        "paired_rmse_delta_vs_per_drug_ridge_ci_high",
    }
    if not required_interval_columns.issubset(intervals):
        raise ValueError("Missing paired bootstrap interval columns")
    comparison = comparison.merge(intervals, on="model", validate="one_to_one")
    if not np.isfinite(comparison[["rmse", "mae"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite split test metric")
    if set(comparison["n_rows"]) != {int(prep["counts"]["primary_rows_by_split"]["test"])}:
        raise ValueError("Test comparison count differs from prepared rows")
    detail = {
        "seed": seed,
        "run_directory": run_dir.name,
        "preprocessing_config_sha256": sha256(run_dir / "configs/preprocessing.json"),
        "preprocessing_manifest_sha256": sha256(run_dir / "results/preprocessing_manifest.json"),
        "fresh_evaluation_lock_sha256": sha256(lock_path),
        "fresh_evaluation_manifest_sha256": sha256(eval_manifest_path),
        "fresh_test_comparison_sha256": sha256(comparison_path),
        "fresh_bootstrap_intervals_sha256": sha256(intervals_path),
        "primary_response_sha256": sha256(run_dir / "data/processed/primary_responses.csv"),
        "primary_identity_sha256_excluding_split": identity_digest,
        "n_primary_rows": n_rows,
        "rows_by_split": prep["counts"]["primary_rows_by_split"],
        "cells_by_split": prep["counts"]["split_cell_lines"],
        "drugs_by_split": {split: int((counts[split] > 0).sum()) for split in ("train", "validation", "test")},
        "minimum_drug_rows_by_split": {split: int(counts[split].min()) for split in ("train", "validation", "test")},
        "selected_alpha": lock["baseline_selected_alpha"],
        "selected_mlp_learning_rate": (
            float(lock["selected_mlp_runs"][0]["learning_rate"]) if models == "primary" else None
        ),
        "validation_selected_model": lock["validation_selected_model"],
    }
    return detail, comparison, identity_digest


def aggregate(seed_tables: list[tuple[int, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    differences = []
    for seed, table in seed_tables:
        model_rows = table.set_index("model")
        reference = float(model_rows.loc["per_drug_ridge", "rmse"])
        for record in table.to_dict(orient="records"):
            record["split_seed"] = seed
            rows.append(record)
            differences.append({
                "split_seed": seed,
                "model": str(record["model"]),
                "rmse_delta_vs_per_drug_ridge": float(record["rmse"]) - reference,
                "paired_rmse_delta_vs_per_drug_ridge_ci_low": float(
                    record["paired_rmse_delta_vs_per_drug_ridge_ci_low"]
                ),
                "paired_rmse_delta_vs_per_drug_ridge_ci_high": float(
                    record["paired_rmse_delta_vs_per_drug_ridge_ci_high"]
                ),
            })
    comparison = pd.DataFrame(rows)
    differences_table = pd.DataFrame(differences)
    summary_rows = []
    for model, group in comparison.groupby("model", sort=True):
        deltas = differences_table.loc[differences_table["model"] == model, "rmse_delta_vs_per_drug_ridge"]
        summary_rows.append({
            "model": model,
            "n_splits": len(group),
            "rmse_median": float(group["rmse"].median()),
            "rmse_min": float(group["rmse"].min()),
            "rmse_max": float(group["rmse"].max()),
            "paired_rmse_delta_median_vs_per_drug_ridge": float(deltas.median()),
            "paired_rmse_delta_min_vs_per_drug_ridge": float(deltas.min()),
            "paired_rmse_delta_max_vs_per_drug_ridge": float(deltas.max()),
            "splits_with_lower_rmse_than_per_drug_ridge": int((deltas < 0).sum()),
        })
    return comparison, differences_table, pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True, help="New ignored directory under data/")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--preprocessing-config", type=Path, default=PREPROCESSING_PATH)
    parser.add_argument("--models", choices=("primary", "baselines"), default="primary")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Permit baseline-only check on invented data")
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    args = parser.parse_args()
    if args.synthetic_smoke != (args.models == "baselines"):
        parser.error("Baseline-only mode is for --synthetic-smoke; GDSC2 study uses --models primary")
    if not args.synthetic_smoke and (
        args.protocol.resolve() != PROTOCOL_PATH.resolve()
        or args.preprocessing_config.resolve() != PREPROCESSING_PATH.resolve()
    ):
        parser.error("GDSC2 robustness must use the tracked protocol and preprocessing config")
    input_dir = args.input_dir.resolve(strict=True)
    study_dir = args.study_dir.resolve()
    data_root = (SOURCE_ROOT / "data").resolve()
    if not study_dir.is_relative_to(data_root) or study_dir == data_root:
        parser.error("--study-dir must be a new ignored directory under repository data/")
    if study_dir.is_relative_to(input_dir) or study_dir.exists():
        parser.error("--study-dir must not exist or sit inside the input directory")
    protocol = json_file(args.protocol)
    preprocessing = json_file(args.preprocessing_config)
    verify_protocol(protocol, preprocessing, synthetic_smoke=args.synthetic_smoke)
    bootstrap_draws = int(
        args.bootstrap_draws if args.bootstrap_draws is not None
        else protocol["bootstrap"]["per_split_resamples"]
    )
    if bootstrap_draws < 2 or (not args.synthetic_smoke
                               and bootstrap_draws != protocol["bootstrap"]["per_split_resamples"]):
        parser.error("GDSC2 bootstrap draws must match the prespecified protocol")
    actual_hashes = {name: sha256(input_dir / name) for name in RAW_NAMES}
    if actual_hashes != protocol["input_sha256"]:
        parser.error("Raw input files differ from the prespecified SHA-256 hashes")

    tracked_sources = source_hashes()
    config_hashes = {
        name: sha256(SOURCE_ROOT / "configs" / name)
        for name in ("baseline_ridge.json", "neural_mlp.json")
    }
    study_dir.mkdir(parents=True)
    prefit = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": "synthetic_pipeline_check" if args.synthetic_smoke else "exploratory_grouped_split_robustness",
        "protocol_sha256": sha256(args.protocol),
        "preprocessing_template_sha256": sha256(args.preprocessing_config),
        "input_sha256": actual_hashes,
        "source_sha256": tracked_sources,
        "model_config_sha256": config_hashes,
        "split_seeds": list(SPLIT_SEEDS),
        "models": args.models,
        "bootstrap_draws_per_split": bootstrap_draws,
        "historical_seed_17": "reference only; excluded from new-split summary",
    }
    prefit_path = study_dir / "prefit_protocol_lock.json"
    with prefit_path.open("x", encoding="utf-8") as output:
        json.dump(prefit, output, indent=2, sort_keys=True)
        output.write("\n")

    details = []
    seed_tables = []
    identity_digest = None
    for seed in SPLIT_SEEDS:
        if (source_hashes() != tracked_sources
                or sha256(args.protocol) != prefit["protocol_sha256"]
                or any(sha256(SOURCE_ROOT / "configs" / name) != digest
                       for name, digest in config_hashes.items())):
            raise ValueError("Source or study protocol changed during the five-split experiment")
        run_dir = study_dir / f"seed_{seed}"
        print(f"Running prespecified grouped split seed {seed}", flush=True)
        started = perf_counter()
        subprocess.run([
            sys.executable, str(SOURCE_ROOT / "src/run_fresh.py"),
            "--input-dir", str(input_dir),
            "--run-dir", str(run_dir),
            "--preprocessing-config", str(args.preprocessing_config),
            "--models", args.models,
            "--split-seed", str(seed),
            "--bootstrap-draws", str(bootstrap_draws),
        ], check=True)
        detail, comparison, identity_digest = verify_split_workspace(
            run_dir, seed=seed, protocol=protocol, models=args.models,
            expected_identity_digest=identity_digest, bootstrap_draws=bootstrap_draws,
        )
        detail["runtime_seconds"] = perf_counter() - started
        details.append(detail)
        seed_tables.append((seed, comparison))
    if (source_hashes() != tracked_sources
            or sha256(args.protocol) != prefit["protocol_sha256"]
            or any(sha256(SOURCE_ROOT / "configs" / name) != digest
                   for name, digest in config_hashes.items())):
        raise ValueError("Source or study protocol changed before aggregation")

    comparison, differences, summary = aggregate(seed_tables)
    outputs = {
        "multisplit_comparison.csv": comparison,
        "multisplit_paired_differences.csv": differences,
        "multisplit_summary.csv": summary,
    }
    for name, table in outputs.items():
        table.to_csv(study_dir / name, index=False, float_format="%.17g", lineterminator="\n")
    manifest = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "prefit_protocol_lock_sha256": sha256(prefit_path),
        "protocol_sha256": prefit["protocol_sha256"],
        "model_mode": args.models,
        "split_seeds": list(SPLIT_SEEDS),
        "n_completed_splits": len(details),
        "primary_identity_sha256_excluding_split": identity_digest,
        "per_split": details,
        "aggregate_sha256": {name: sha256(study_dir / name) for name in outputs},
        "interpretation_boundary": (
            "Same public GDSC2 snapshot and overlapping cell-line splits; descriptive "
            "internal robustness only, not independent confirmation or a five-sample "
            "population confidence interval."
        ),
    }
    with (study_dir / "study_manifest.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"Five-split summary: {study_dir / 'multisplit_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
