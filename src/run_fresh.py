"""Run the primary study in a new, isolated workspace.

This is a methodological fresh run, not a replay of the published byte-locked
checkpoint. Validation finishes and a run-specific lock is written before the
separate evaluator reads test outcomes. Uprosertib sensitivity and the later
exploratory GNN extension are outside this runner.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = ("preprocessing.json", "baseline_ridge.json", "neural_mlp.json")
RAW_NAMES = ("gdsc2.pkl", "gdsc_gene_symbols.tab")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest_artifacts(run_dir: Path, artifacts: dict[str, str], *, prefix: str) -> None:
    """Reject files altered between fitting/validation and evaluation lock."""
    if not artifacts:
        raise ValueError(f"Empty {prefix} artifact manifest")
    for name, expected in artifacts.items():
        relative = f"{prefix}/{name}" if prefix else name
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise ValueError(f"Missing or unsafe manifest artifact: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Artifact differs from validation manifest: {relative}")


def make_lock(run_dir: Path, *, mode: str, bootstrap_draws: int) -> Path:
    """Freeze inputs and validation selections without reading test labels."""
    results = run_dir / "results"
    baseline_manifest = json_file(results / "baseline_validation_manifest.json")
    prep_manifest = json_file(results / "preprocessing_manifest.json")
    preprocessing_config = json_file(run_dir / "configs/preprocessing.json")
    split_seed = int(preprocessing_config["split"]["seed"])
    baseline_comparison = pd.read_csv(results / "baseline_comparison.csv")
    required = [
        "configs/preprocessing.json", "configs/baseline_ridge.json",
        "data/processed/primary_responses.csv",
        "data/processed/cell_line_assignments.csv", "data/processed/drugs.csv",
        "data/processed/expression_pca128_scaled.npy",
        "data/processed/morgan_radius2_2048.npy",
        "data/processed/pca_score_scaler.joblib",
        "results/preprocessing_manifest.json",
        "results/baseline_validation_manifest.json",
        "results/baseline_comparison.csv",
        "results/checkpoints/baseline_models.joblib",
    ]
    if baseline_manifest["preprocessing_manifest_sha256"] != sha256(results / "preprocessing_manifest.json"):
        raise ValueError("Baseline/preprocessing manifest mismatch")
    if prep_manifest["config_sha256"] != sha256(run_dir / "configs/preprocessing.json"):
        raise ValueError("Preprocessing configuration mismatch")
    if baseline_manifest["config_sha256"] != sha256(run_dir / "configs/baseline_ridge.json"):
        raise ValueError("Baseline configuration mismatch")
    if baseline_manifest["input_primary_sha256"] != sha256(run_dir / "data/processed/primary_responses.csv"):
        raise ValueError("Primary response table changed after baseline fitting")
    verify_manifest_artifacts(run_dir, prep_manifest["artifacts_sha256"], prefix="data/processed")
    verify_manifest_artifacts(run_dir, baseline_manifest["artifact_sha256"], prefix="")

    selected_mlp_runs: list[dict[str, object]] = []
    validation = {str(row.model): float(row.validation_rmse) for row in baseline_comparison.itertuples()}
    if mode == "primary":
        neural_manifest = json_file(results / "neural_validation_manifest.json")
        if neural_manifest["baseline_manifest_sha256"] != sha256(results / "baseline_validation_manifest.json"):
            raise ValueError("Neural/baseline manifest mismatch")
        if neural_manifest["config_sha256"] != sha256(run_dir / "configs/neural_mlp.json"):
            raise ValueError("Neural configuration mismatch")
        verify_manifest_artifacts(run_dir, neural_manifest["artifact_sha256"], prefix="")
        runs = pd.read_csv(results / "neural_runs.csv")
        selected_ids = neural_manifest["selected_run_ids"]
        selected = runs.set_index("run_id").loc[selected_ids].reset_index()
        if len(selected) != 3 or selected["seed"].nunique() != 3:
            raise ValueError("Expected three independently trained selected-rate MLP runs")
        for row in selected.itertuples():
            relative = f"results/checkpoints/mlp_{row.run_id}.pt"
            selected_mlp_runs.append({
                "seed": int(row.seed), "run_id": str(row.run_id),
                "learning_rate": float(row.learning_rate),
                "best_epoch": int(row.best_epoch),
                "checkpoint_sha256": sha256(run_dir / relative),
            })
            required.append(relative)
        if not all(abs(run["learning_rate"] - neural_manifest["selected_learning_rate"]) < 1e-12 for run in selected_mlp_runs):
            raise ValueError("Selected MLP checkpoint rate mismatch")
        required += ["configs/neural_mlp.json", "results/neural_validation_manifest.json", "results/neural_runs.csv", "results/neural_summary.csv"]
        neural_summary = pd.read_csv(results / "neural_summary.csv")
        selected_summary = neural_summary.loc[neural_summary["selected_learning_rate"].astype(bool)]
        if len(selected_summary) != 1:
            raise ValueError("Ambiguous MLP validation selection")
        validation["mlp_three_seed_metric_mean"] = float(selected_summary.iloc[0]["validation_rmse_mean"])

    chosen = min(validation, key=validation.get)
    lock = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "split_seed": split_seed,
        "protocol": {
            "question": "known drugs in previously unseen cancer cell lines",
            "target": "provided Y; no additional transform",
            "split": f"seed-{split_seed} disjoint cell lines from preprocessing configuration",
            "fitting": "training rows only; expression transforms fitted on unique training cells",
            "selection": "validation RMSE; no train-plus-validation refit",
            "test": "evaluate the locked models once after this file is written",
            "sensitivity": "not included in fresh primary runner; see historical locked study",
        },
        "validation_rmse": validation,
        "validation_selected_model": chosen,
        "baseline_selected_alpha": baseline_manifest["selected_alpha"],
        "selected_mlp_runs": selected_mlp_runs,
        "bootstrap": {"resamples": bootstrap_draws, "seed": 2026, "unit": "whole_cell_line", "interval": "95_percentile_paired"},
        "raw_sha256": prep_manifest["raw_sha256"],
        "artifact_sha256": {relative: sha256(run_dir / relative) for relative in required},
        "source_sha256": {
            str(path.relative_to(SOURCE_ROOT)).replace("\\", "/"): sha256(path)
            for path in [SOURCE_ROOT / "src" / name for name in (
                "run_fresh.py", "prepare_gdsc2.py", "run_baselines.py", "run_neural.py",
                "preprocessing.py", "baselines.py", "neural.py", "evaluate_fresh.py",
            )] if path.exists()
        },
    }
    path = results / "fresh_evaluation_lock.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(lock, output, indent=2, sort_keys=True)
        output.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing verified gdsc2.pkl and gdsc_gene_symbols.tab")
    parser.add_argument("--run-dir", type=Path, required=True, help="New directory for generated artifacts; must not exist")
    parser.add_argument("--preprocessing-config", type=Path, default=SOURCE_ROOT / "configs/preprocessing.json")
    parser.add_argument("--models", choices=("baselines", "primary"), default="primary", help="Primary adds the six fixed MLP validation runs")
    parser.add_argument("--split-seed", type=int, default=17, help="Grouped cell-line split seed; default preserves the original seed-17 method")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if args.split_seed < 0:
        parser.error("--split-seed must be nonnegative")
    if args.bootstrap_draws < 2:
        parser.error("--bootstrap-draws must be at least 2")
    input_dir = args.input_dir.resolve(strict=True)
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        parser.error(f"Run directory already exists: {run_dir}")
    config = json_file(args.preprocessing_config)
    fixed_config = json_file(SOURCE_ROOT / "configs/preprocessing.json")
    if {key: value for key, value in config.items() if key != "input_sha256"} != {
        key: value for key, value in fixed_config.items() if key != "input_sha256"
    }:
        parser.error("Preprocessing configuration differs from the fixed study protocol; only input_sha256 may change")
    for name in RAW_NAMES:
        path = input_dir / name
        if not path.is_file():
            parser.error(f"Missing input: {path}")
        if sha256(path) != config["input_sha256"][name]:
            parser.error(f"Input SHA-256 mismatch for {name}; check source and configuration")

    (run_dir / "configs").mkdir(parents=True)
    (run_dir / "data/processed").mkdir(parents=True)
    (run_dir / "results/checkpoints").mkdir(parents=True)
    (run_dir / "results/tmp").mkdir(parents=True)
    if args.split_seed == int(config["split"]["seed"]):
        # Preserve the original config bytes for the default seed-17 path.
        shutil.copyfile(args.preprocessing_config, run_dir / "configs/preprocessing.json")
    else:
        derived_config = json.loads(json.dumps(config))
        derived_config["split"]["seed"] = args.split_seed
        (run_dir / "configs/preprocessing.json").write_text(
            json.dumps(derived_config, indent=2) + "\n", encoding="utf-8",
        )
    for name in CONFIG_NAMES[1:]:
        shutil.copyfile(SOURCE_ROOT / "configs" / name, run_dir / "configs" / name)
    run_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "source_root": str(SOURCE_ROOT),
        "python_executable": sys.executable,
        "mode": args.models,
        "split_seed": args.split_seed,
        "raw_sha256": config["input_sha256"],
    }
    (run_dir / "results/run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["DRP_RUN_ROOT"] = str(run_dir)
    env["DRP_RAW_DIR"] = str(input_dir)
    for name in ("prepare_gdsc2.py", "run_baselines.py") + (("run_neural.py",) if args.models == "primary" else ()):
        print(f"Running {name} in {run_dir}", flush=True)
        subprocess.run([sys.executable, str(SOURCE_ROOT / "src" / name)], env=env, check=True)
    lock_path = make_lock(run_dir, mode=args.models, bootstrap_draws=args.bootstrap_draws)
    print(f"Locked validation selections at {lock_path}", flush=True)
    subprocess.run([sys.executable, str(SOURCE_ROOT / "src/evaluate_fresh.py"), "--run-dir", str(run_dir)], check=True)
    print(f"Fresh run complete: {run_dir / 'results/fresh_test_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
