"""Export only aggregate robustness results from ignored experiment workspaces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPLITS = (19, 23, 29, 31, 37)
SPLIT_TABLES = {
    "multisplit_comparison.csv": "robustness_multisplit_comparison.csv",
    "multisplit_paired_differences.csv": "robustness_multisplit_paired_differences.csv",
    "multisplit_summary.csv": "robustness_multisplit_summary.csv",
}
PCA_TABLES = {
    "validation_metrics.csv": "robustness_pca_validation_metrics.csv",
    "pca_fit_summary.csv": "robustness_pca_fit_summary.csv",
    "test_metrics.csv": "robustness_pca_test_metrics.csv",
    "test_bootstrap_intervals.csv": "robustness_pca_bootstrap_intervals.csv",
    "test_paired_differences.csv": "robustness_pca_paired_differences.csv",
    "test_across_split_summary.csv": "robustness_pca_summary.csv",
    "test_by_drug.csv": "robustness_pca_test_by_drug.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--pca-dir", type=Path, required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve(strict=True)
    pca = args.pca_dir.resolve(strict=True)
    data_root = (ROOT / "data").resolve()
    if not study.is_relative_to(data_root) or not pca.is_relative_to(data_root):
        parser.error("Source workspaces must be Git-ignored directories under data/")
    destination = ROOT / "results"
    split_manifest = load_json(study / "study_manifest.json")
    pca_manifest = load_json(pca / "pca_sensitivity_manifest.json")
    pca_lock = load_json(pca / "pretest_lock.json")
    if split_manifest["split_seeds"] != list(SPLITS) or pca_manifest["split_seeds"] != list(SPLITS):
        raise ValueError("Study manifests do not contain the five prespecified splits")
    if pca_manifest["pretest_lock_sha256"] != sha256(pca / "pretest_lock.json"):
        raise ValueError("PCA validation lock changed")
    exported = {}
    for source_name, public_name in SPLIT_TABLES.items():
        source = study / source_name
        if sha256(source) != split_manifest["aggregate_sha256"][source_name]:
            raise ValueError(f"Split aggregate changed: {source_name}")
        shutil.copyfile(source, destination / public_name)
        exported[public_name] = sha256(destination / public_name)
    for source_name, public_name in PCA_TABLES.items():
        source = pca / source_name
        expected = (
            pca_manifest.get("test_output_sha256", {}).get(source_name)
            or pca_lock["generated_sha256"].get(source_name)
        )
        if expected is None or sha256(source) != expected:
            raise ValueError(f"PCA aggregate changed: {source_name}")
        shutil.copyfile(source, destination / public_name)
        exported[public_name] = sha256(destination / public_name)

    provenance_files = (
        (study / "prefit_protocol_lock.json", "robustness_multisplit_prefit_lock.json"),
        (study / "study_manifest.json", "robustness_multisplit_manifest.json"),
        (pca / "pretest_lock.json", "robustness_pca_pretest_lock.json"),
        (pca / "pca_sensitivity_manifest.json", "robustness_pca_manifest.json"),
    )
    for source, public_name in provenance_files:
        shutil.copyfile(source, destination / public_name)
        exported[public_name] = sha256(destination / public_name)

    drug_tables = []
    for seed in SPLITS:
        run = study / f"seed_{seed}"
        manifest = load_json(run / "results/fresh_evaluation_manifest.json")
        source = run / "results/fresh_test_by_drug.csv"
        if sha256(source) != manifest["artifact_sha256"]["results/fresh_test_by_drug.csv"]:
            raise ValueError(f"Seed {seed}: drug-level table changed")
        one = pd.read_csv(source)
        # The MLP summary averages seed-specific metrics; it has no ensemble
        # prediction and therefore no separate drug-level prediction table.
        if one["model"].nunique() != 6 or len(one) != 6 * 136:
            raise ValueError(f"Seed {seed}: incomplete drug-level table")
        one.insert(0, "split_seed", seed)
        drug_tables.append(one)
    by_drug = pd.concat(drug_tables, ignore_index=True)
    by_drug_path = destination / "robustness_multisplit_test_by_drug.csv"
    by_drug.to_csv(by_drug_path, index=False, float_format="%.17g", lineterminator="\n")
    exported[by_drug_path.name] = sha256(by_drug_path)

    provenance = {
        "schema_version": 1,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(ROOT / "configs/robustness_pca.json"),
        "split_study_manifest_sha256": sha256(study / "study_manifest.json"),
        "pca_pretest_lock_sha256": sha256(pca / "pretest_lock.json"),
        "pca_manifest_sha256": sha256(pca / "pca_sensitivity_manifest.json"),
        "split_seeds": list(SPLITS),
        "exported_sha256": exported,
        "exclusions": "No response rows, raw expression/fingerprints, checkpoints, or row-level predictions exported.",
    }
    path = destination / "robustness_release_manifest.json"
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported {len(exported)} compact result/provenance files with manifest {path}")


if __name__ == "__main__":
    main()
