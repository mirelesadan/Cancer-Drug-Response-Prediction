"""End-to-end CPU check of the public fresh-run path on invented data.

The synthetic scores here are pipeline diagnostics, not scientific results.
No TDC download, historical checkpoint, or GPU is needed.
"""

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

import joblib
import numpy as np
import pandas as pd

from synthetic_fixture import (
    N_CELLS, N_EXPRESSION_FEATURES, PRIMARY_RESPONSE_ROWS,
    UPROSERTIB_SOURCE_ROWS, write_synthetic_gdsc2,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # joblib models use the baselines module name


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreshRunIntegrationTests(unittest.TestCase):
    def test_synthetic_baseline_run_is_isolated_and_aligned(self):
        # The checked base ensures the generated tree is removed only inside
        # this workspace. Windows restricted tokens can create inaccessible
        # mode-0700 tempfile directories, so use the default workspace ACL.
        temporary_base = (ROOT / "data").resolve()
        self.assertTrue(temporary_base.is_relative_to(ROOT.resolve()))
        work = (temporary_base / f"synthetic-ci-{uuid.uuid4().hex}").resolve()
        self.assertEqual(work.parent, temporary_base)
        work.mkdir()
        try:
            raw_dir = work / "input"
            run_dir = work / "fresh"
            raw = write_synthetic_gdsc2(raw_dir)
            self.assertEqual(len(raw), PRIMARY_RESPONSE_ROWS + UPROSERTIB_SOURCE_ROWS)
            config = json.loads((ROOT / "configs" / "preprocessing.json").read_text(encoding="utf-8"))
            raw_hashes = {name: sha256(raw_dir / name) for name in ("gdsc2.pkl", "gdsc_gene_symbols.tab")}
            config["input_sha256"] = raw_hashes
            synthetic_config = work / "synthetic_preprocessing.json"
            synthetic_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            public_manifest_paths = (
                ROOT / "results" / "preprocessing_manifest.json",
                ROOT / "results" / "baseline_validation_manifest.json",
            )
            original_hashes = {path: sha256(path) for path in public_manifest_paths}
            environment = os.environ.copy()
            environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            command = [
                sys.executable, str(ROOT / "src" / "run_fresh.py"),
                "--input-dir", str(raw_dir),
                "--run-dir", str(run_dir),
                "--preprocessing-config", str(synthetic_config),
                "--models", "baselines",
                "--bootstrap-draws", "32",
            ]
            completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=600)
            self.assertEqual(completed.returncode, 0, (completed.stdout + "\n" + completed.stderr)[-4000:])
            self.assertEqual({name: sha256(raw_dir / name) for name in raw_hashes}, raw_hashes)
            self.assertEqual({path: sha256(path) for path in public_manifest_paths}, original_hashes)

            processed = run_dir / "data" / "processed"
            results = run_dir / "results"
            prep_manifest = json.loads((results / "preprocessing_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(prep_manifest["counts"]["raw_rows"], PRIMARY_RESPONSE_ROWS + UPROSERTIB_SOURCE_ROWS)
            self.assertEqual(prep_manifest["counts"]["primary_rows"], PRIMARY_RESPONSE_ROWS)
            self.assertEqual(prep_manifest["counts"]["primary_excluded_rows"], UPROSERTIB_SOURCE_ROWS)
            self.assertEqual(prep_manifest["counts"]["split_cell_lines"], {"train": 133, "validation": 28, "test": 29})
            self.assertEqual(prep_manifest["shapes"]["expression_raw"], [N_CELLS, N_EXPRESSION_FEATURES])
            self.assertEqual(prep_manifest["shapes"]["expression_pca128"], [N_CELLS, 128])
            self.assertEqual(prep_manifest["expression_preprocessing"]["fit_unique_training_cells"], 133)
            self.assertEqual(prep_manifest["expression_preprocessing"]["zero_variance_removed"], 1)
            self.assertEqual(prep_manifest["counts"]["duplicate_gene_symbol_positions"], 1)

            assignments = pd.read_csv(processed / "cell_line_assignments.csv")
            catalog = pd.read_csv(processed / "expression_features.csv")
            drugs = pd.read_csv(processed / "drugs.csv")
            primary = pd.read_csv(processed / "primary_responses.csv", float_precision="round_trip")
            sensitivity = pd.read_csv(processed / "sensitivity_responses.csv", float_precision="round_trip")
            self.assertEqual(len(assignments), N_CELLS)
            self.assertEqual(assignments.groupby("cell_line_id")["split"].nunique().max(), 1)
            self.assertEqual(catalog.loc[10, "gene_symbol"], catalog.loc[11, "gene_symbol"])
            self.assertNotEqual(catalog.loc[10, "feature_id"], catalog.loc[11, "feature_id"])
            self.assertEqual(len(primary), PRIMARY_RESPONSE_ROWS)
            self.assertFalse(primary["drug_id"].eq("Uprosertib").any())
            self.assertEqual(len(sensitivity.loc[sensitivity["drug_id"].eq("Uprosertib")]), 12)
            self.assertEqual(
                primary["cell_line_id"].tolist(),
                assignments.loc[primary["cell_index"], "cell_line_id"].tolist(),
            )
            self.assertEqual(
                primary["drug_id"].tolist(),
                drugs.loc[primary["drug_index"], "drug_id"].tolist(),
            )
            self.assertEqual(
                primary["split"].tolist(),
                assignments.loc[primary["cell_index"], "split"].tolist(),
            )
            source_positions = primary["source_row_id"].to_numpy(dtype=np.int64)
            np.testing.assert_allclose(primary["y"], raw.loc[source_positions, "Y"], rtol=0, atol=0)
            self.assertTrue(
                primary.groupby("drug_id")["split"].apply(lambda splits: "train" in set(splits)).all()
            )

            pca = np.load(processed / "expression_pca128.npy", allow_pickle=False)
            scaled = np.load(processed / "expression_pca128_scaled.npy", allow_pickle=False)
            bits = np.load(processed / "morgan_radius2_2048.npy", allow_pickle=False)
            scaler = joblib.load(processed / "pca_score_scaler.joblib")
            train_cells = assignments.index[assignments["split"].eq("train")].to_numpy()
            np.testing.assert_allclose(scaler.mean_, pca[train_cells].mean(axis=0), atol=1e-12, rtol=0)
            np.testing.assert_allclose(scaled, scaler.transform(pca), atol=1e-12, rtol=0)
            self.assertEqual(bits.shape, (4, 2048))
            self.assertTrue(np.isin(bits, [0, 1]).all())
            self.assertTrue(np.isfinite(scaled).all())
            # Unequal drug coverage makes row-weighted and unique-cell means
            # differ, so this assertion guards a consequential leakage mistake.
            weighted = pca[primary.loc[primary["split"].eq("train"), "cell_index"]].mean(axis=0)
            self.assertGreater(float(np.linalg.norm(weighted - scaler.mean_)), 0.01)

            validation = pd.read_csv(results / "baseline_comparison.csv")
            self.assertEqual(set(validation["model"]), {"per_drug_mean", "pooled_ridge", "per_drug_ridge"})
            self.assertTrue(np.isfinite(validation[["train_rmse", "validation_rmse"]].to_numpy()).all())
            self.assertEqual(set(validation["train_n"]), {int(primary["split"].eq("train").sum())})
            self.assertEqual(set(validation["validation_n"]), {int(primary["split"].eq("validation").sum())})

            # Verify saved validation predictions from reloaded models.
            models = joblib.load(results / "checkpoints" / "baseline_models.joblib")
            saved = pd.read_csv(results / "tmp" / "baseline_validation_predictions.csv", float_precision="round_trip")
            self.assertEqual(len(saved), int(primary["split"].eq("validation").sum()))
            self.assertEqual(saved["source_row_id"].tolist(), primary.loc[primary["split"].eq("validation"), "source_row_id"].tolist())
            cells = saved["cell_index"].to_numpy(dtype=np.int32)
            drug_index = saved["drug_index"].to_numpy(dtype=np.int32)
            replay = {
                "per_drug_mean": models["per_drug_mean"].predict(drug_index),
                "pooled_ridge": models["pooled_ridge"].predict(cells, drug_index, scaled, bits),
                "per_drug_ridge": models["per_drug_ridge"].predict(cells, drug_index, scaled),
            }
            for name, prediction in replay.items():
                np.testing.assert_allclose(saved[name], prediction, rtol=0, atol=1e-9)

            lock = json.loads((results / "fresh_evaluation_lock.json").read_text(encoding="utf-8"))
            evaluation_manifest = json.loads((results / "fresh_evaluation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(lock["mode"], "baselines")
            self.assertEqual(lock["raw_sha256"], raw_hashes)
            self.assertEqual(lock["bootstrap"], {
                "resamples": 32, "seed": 2026,
                "unit": "whole_cell_line", "interval": "95_percentile_paired",
            })
            self.assertIn(lock["validation_selected_model"], set(replay))
            self.assertEqual(lock["selected_mlp_runs"], [])
            self.assertEqual(lock["artifact_sha256"]["results/checkpoints/baseline_models.joblib"],
                             sha256(results / "checkpoints" / "baseline_models.joblib"))
            self.assertEqual(evaluation_manifest["lock_sha256"], sha256(results / "fresh_evaluation_lock.json"))
            self.assertEqual(evaluation_manifest["test_cell_lines"], 29)
            self.assertEqual(evaluation_manifest["device"], "none")
            for relative, expected in evaluation_manifest["artifact_sha256"].items():
                self.assertEqual(sha256(run_dir / relative), expected)

            final = pd.read_csv(results / "fresh_test_comparison.csv")
            self.assertEqual(set(final["model"]), set(replay))
            self.assertTrue(np.isfinite(final[["rmse", "mae"]].to_numpy()).all())
            self.assertEqual(set(final["n_rows"]), {int(primary["split"].eq("test").sum())})
            heldout = pd.read_csv(results / "tmp" / "fresh_test_predictions.csv", float_precision="round_trip")
            test_rows = primary.loc[primary["split"].eq("test")]
            self.assertEqual(heldout["source_row_id"].tolist(), test_rows["source_row_id"].tolist())
            np.testing.assert_array_equal(heldout["y"].to_numpy(), test_rows["y"].to_numpy())
            test_cells = heldout["cell_index"].to_numpy(dtype=np.int32)
            test_drugs = heldout["drug_index"].to_numpy(dtype=np.int32)
            expected_test = {
                "per_drug_mean": models["per_drug_mean"].predict(test_drugs),
                "pooled_ridge": models["pooled_ridge"].predict(test_cells, test_drugs, scaled, bits),
                "per_drug_ridge": models["per_drug_ridge"].predict(test_cells, test_drugs, scaled),
            }
            for name, prediction in expected_test.items():
                np.testing.assert_allclose(heldout[name], prediction, rtol=0, atol=1e-9)
                result = final.loc[final["model"].eq(name)].iloc[0]
                residual = prediction - heldout["y"].to_numpy(dtype=np.float64)
                self.assertAlmostEqual(float(result["rmse"]), float(np.sqrt(np.mean(residual ** 2))), places=10)
                self.assertAlmostEqual(float(result["mae"]), float(np.mean(np.abs(residual))), places=10)
            intervals = pd.read_csv(results / "fresh_bootstrap_intervals.csv")
            self.assertEqual(set(intervals["model"]), set(replay))
            self.assertTrue(np.isfinite(intervals.drop(columns="model").to_numpy()).all())
            self.assertTrue((intervals["rmse_ci_low"] <= intervals["rmse_ci_high"]).all())
            self.assertTrue((intervals["mae_ci_low"] <= intervals["mae_ci_high"]).all())
            reference = intervals.loc[intervals["model"].eq("per_drug_ridge")].iloc[0]
            self.assertEqual(float(reference["paired_rmse_delta_vs_per_drug_ridge_ci_low"]), 0.0)
            self.assertEqual(float(reference["paired_rmse_delta_vs_per_drug_ridge_ci_high"]), 0.0)

            # A post-validation checkpoint change must fail before a new lock
            # can be written, even if its metadata still looks plausible.
            from run_fresh import make_lock
            checkpoint_path = results / "checkpoints" / "baseline_models.joblib"
            original_checkpoint = checkpoint_path.read_bytes()
            checkpoint_path.write_bytes(original_checkpoint + b"changed")
            with self.assertRaisesRegex(ValueError, "differs from validation manifest"):
                make_lock(run_dir, mode="baselines", bootstrap_draws=32)
            checkpoint_path.write_bytes(original_checkpoint)

            # An input change after the lock must be caught before test labels
            # are loaded by the evaluator.
            from evaluate_fresh import verify_lock
            baseline_config = run_dir / "configs" / "baseline_ridge.json"
            baseline_config.write_bytes(baseline_config.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Locked artifact changed"):
                verify_lock(run_dir)
        finally:
            if work.parent != temporary_base or not work.name.startswith("synthetic-ci-"):
                raise AssertionError("Refusing cleanup outside the generated test directory")
            shutil.rmtree(work)


if __name__ == "__main__":
    unittest.main()
