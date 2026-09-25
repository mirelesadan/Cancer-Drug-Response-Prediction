"""Synthetic five-split check; scores are pipeline diagnostics only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np
import pandas as pd

from synthetic_fixture import write_synthetic_gdsc2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_multisplit import SPLIT_SEEDS, primary_identity_digest, verify_protocol  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MultiSplitIntegrationTests(unittest.TestCase):
    def test_synthetic_five_split_baseline_study(self):
        data_root = (ROOT / "data").resolve()
        work = (data_root / f"synthetic-multisplit-{uuid.uuid4().hex}").resolve()
        self.assertEqual(work.parent, data_root)
        work.mkdir()
        try:
            input_dir = work / "input"
            write_synthetic_gdsc2(input_dir)
            input_hashes = {
                name: sha256(input_dir / name)
                for name in ("gdsc2.pkl", "gdsc_gene_symbols.tab")
            }
            config = json.loads((ROOT / "configs/preprocessing.json").read_text(encoding="utf-8"))
            config["input_sha256"] = input_hashes
            config_path = work / "synthetic_preprocessing.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            protocol = json.loads((ROOT / "configs/robustness_pca.json").read_text(encoding="utf-8"))
            protocol["input_sha256"] = input_hashes
            protocol["historical_seed17_primary_response_sha256"] = None
            protocol["seed_invariant_primary_identity"]["sha256"] = None
            protocol_path = work / "synthetic_protocol.json"
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
            verify_protocol(protocol, config, synthetic_smoke=True)
            bad_protocol = json.loads(json.dumps(protocol))
            bad_protocol["split"]["seeds"] = [19, 23, 29, 31, 41]
            with self.assertRaisesRegex(ValueError, "prespecified split seeds"):
                verify_protocol(bad_protocol, config, synthetic_smoke=True)

            published_manifest = ROOT / "results/preprocessing_manifest.json"
            before_published_hash = sha256(published_manifest)
            study_dir = work / "study"
            command = [
                sys.executable, str(ROOT / "src/run_multisplit.py"),
                "--input-dir", str(input_dir),
                "--study-dir", str(study_dir),
                "--protocol", str(protocol_path),
                "--preprocessing-config", str(config_path),
                "--models", "baselines", "--synthetic-smoke",
                "--bootstrap-draws", "32",
            ]
            environment = os.environ.copy()
            environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            result = subprocess.run(
                command, cwd=ROOT, env=environment,
                capture_output=True, text=True, timeout=900,
            )
            self.assertEqual(result.returncode, 0, (result.stdout + "\n" + result.stderr)[-5000:])
            self.assertEqual(sha256(published_manifest), before_published_hash)
            self.assertEqual(
                {name: sha256(input_dir / name) for name in input_hashes}, input_hashes,
            )
            manifest = json.loads((study_dir / "study_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_seeds"], list(SPLIT_SEEDS))
            self.assertEqual(manifest["n_completed_splits"], len(SPLIT_SEEDS))
            self.assertEqual(manifest["model_mode"], "baselines")
            self.assertEqual(len(manifest["per_split"]), len(SPLIT_SEEDS))
            self.assertEqual(
                manifest["prefit_protocol_lock_sha256"],
                sha256(study_dir / "prefit_protocol_lock.json"),
            )
            identities = set()
            for detail in manifest["per_split"]:
                seed = detail["seed"]
                run_dir = study_dir / f"seed_{seed}"
                self.assertTrue(run_dir.is_dir())
                self.assertEqual(detail["run_directory"], run_dir.name)
                self.assertEqual(detail["fresh_evaluation_lock_sha256"],
                                 sha256(run_dir / "results/fresh_evaluation_lock.json"))
                lock = json.loads((run_dir / "results/fresh_evaluation_lock.json").read_text(encoding="utf-8"))
                self.assertEqual(lock["split_seed"], seed)
                self.assertEqual(lock["bootstrap"]["resamples"], 32)
                self.assertEqual(detail["drugs_by_split"], {"train": 3, "validation": 3, "test": 3})
                identity, n_rows = primary_identity_digest(run_dir / "data/processed/primary_responses.csv")
                self.assertEqual(detail["primary_identity_sha256_excluding_split"], identity)
                self.assertEqual(detail["n_primary_rows"], n_rows)
                identities.add(identity)
            self.assertEqual(len(identities), 1)
            self.assertEqual(manifest["primary_identity_sha256_excluding_split"], identities.pop())
            comparison = pd.read_csv(study_dir / "multisplit_comparison.csv")
            differences = pd.read_csv(study_dir / "multisplit_paired_differences.csv")
            summary = pd.read_csv(study_dir / "multisplit_summary.csv")
            self.assertEqual(set(comparison["split_seed"]), set(SPLIT_SEEDS))
            self.assertEqual(set(comparison["model"]), {"per_drug_mean", "pooled_ridge", "per_drug_ridge"})
            self.assertEqual(len(comparison), len(SPLIT_SEEDS) * 3)
            self.assertTrue(np.isfinite(comparison[["rmse", "mae", "rmse_ci_low", "rmse_ci_high"]]).all().all())
            self.assertEqual(len(differences), len(comparison))
            self.assertTrue((differences.loc[
                differences["model"].eq("per_drug_ridge"), "rmse_delta_vs_per_drug_ridge"
            ] == 0).all())
            self.assertEqual(set(summary["n_splits"]), {len(SPLIT_SEEDS)})
            for name, digest in manifest["aggregate_sha256"].items():
                self.assertEqual(sha256(study_dir / name), digest)

            repeat = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("must not exist", repeat.stderr)
        finally:
            if work.parent != data_root or not work.name.startswith("synthetic-multisplit-"):
                raise AssertionError("Refusing cleanup outside generated synthetic workspace")
            shutil.rmtree(work)


if __name__ == "__main__":
    unittest.main()
