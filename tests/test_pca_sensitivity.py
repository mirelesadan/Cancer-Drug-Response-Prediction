"""Focused synthetic checks for the prespecified PCA sensitivity helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import fingerprint_row_basis  # noqa: E402
from run_pca_sensitivity import (  # noqa: E402
    DIMENSIONS, MODEL_NAMES, fit_dimension_scores, fit_ridge_models,
    paired_bootstrap, primary_identity_sha256, validate_protocol,
)


class PcaSensitivityTests(unittest.TestCase):
    def test_pca_and_score_scaler_fit_only_unique_training_cells(self):
        rng = np.random.default_rng(2718)
        expression = rng.normal(size=(22, 14))
        train_positions = np.array([0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 18, 20])
        settings = {
            "pca_random_state": 17,
            "pca_n_oversamples": 4,
            "pca_iterated_power": 3,
            "pca_power_iteration_normalizer": "QR",
        }
        original, first_pca, first_scaler = fit_dimension_scores(expression, train_positions, 4, settings)
        altered = expression.copy()
        held_out = np.setdiff1d(np.arange(len(expression)), train_positions)
        altered[held_out] += 5000
        changed, second_pca, second_scaler = fit_dimension_scores(altered, train_positions, 4, settings)
        np.testing.assert_array_equal(first_pca.components_, second_pca.components_)
        np.testing.assert_array_equal(first_scaler.mean_, second_scaler.mean_)
        np.testing.assert_allclose(original[train_positions], changed[train_positions], rtol=0, atol=0)
        self.assertGreater(float(np.linalg.norm(original[held_out] - changed[held_out])), 100)
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_dimension_scores(expression, np.array([0, 0, 1, 2, 3]), 4, settings)

    def test_ridge_models_match_reference_and_share_per_drug_alpha(self):
        rng = np.random.default_rng(31415)
        n_cells, n_drugs, components = 34, 3, 6
        scores = rng.normal(size=(n_cells, components))
        fingerprints = np.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 1]], dtype=np.uint8)
        basis = fingerprint_row_basis(fingerprints, 1e-12)
        drug_scores = fingerprints.astype(float) @ basis
        records = []
        for cell in range(n_cells):
            for drug in range(n_drugs):
                y = 1.0 + drug + (0.3 + 0.15 * drug) * scores[cell, 0] - 0.4 * scores[cell, 1]
                records.append({"cell_index": cell, "drug_index": drug,
                                "drug_id": f"Drug{drug}", "y": y + rng.normal(scale=0.01)})
        rows = pd.DataFrame(records)
        train = rows.loc[rows["cell_index"] < 23].reset_index(drop=True)
        validation = rows.loc[rows["cell_index"] >= 23].reset_index(drop=True)
        grid = [0.01, 0.1, 1.0, 10.0]
        models, alphas, train_predictions, val_predictions, fallbacks = fit_ridge_models(
            train, validation, scores, fingerprints, drug_scores, basis, grid, n_drugs,
        )
        self.assertEqual(fallbacks, [])
        self.assertEqual(set(models), set(MODEL_NAMES))
        self.assertTrue(all(alpha in grid for alpha in alphas.values()))
        self.assertTrue(all(prediction.shape == (len(validation),) for prediction in val_predictions.values()))
        self.assertTrue(all(prediction.shape == (len(train),) for prediction in train_predictions.values()))
        x_train = np.column_stack((scores[train["cell_index"]], fingerprints[train["drug_index"]]))
        x_val = np.column_stack((scores[validation["cell_index"]], fingerprints[validation["drug_index"]]))
        reference = Ridge(alpha=alphas["pooled_ridge"], solver="svd").fit(x_train, train["y"])
        np.testing.assert_allclose(reference.predict(x_val), val_predictions["pooled_ridge"], atol=1e-9, rtol=0)
        for drug in range(n_drugs):
            one_train = train.loc[train["drug_index"].eq(drug)]
            one_val = validation.loc[validation["drug_index"].eq(drug)]
            reference = Ridge(alpha=alphas["per_drug_ridge"], solver="svd").fit(
                scores[one_train["cell_index"]], one_train["y"],
            )
            observed = val_predictions["per_drug_ridge"][validation["drug_index"].eq(drug).to_numpy()]
            np.testing.assert_allclose(reference.predict(scores[one_val["cell_index"]]), observed, atol=1e-9, rtol=0)

    def test_paired_bootstrap_keeps_cluster_multiplicity(self):
        frame = pd.DataFrame({"cell_line_id": ["A", "A", "B", "B", "B", "C"],
                              "y": [1.0, 2.0, 3.0, 2.5, 4.0, 5.0]})
        predictions = {}
        for model in MODEL_NAMES:
            for dimension in DIMENSIONS:
                predictions[f"{model}_pca{dimension}"] = (
                    frame["y"].to_numpy() + (dimension - 128) / 1000 + 0.1
                )
        intervals, draws = paired_bootstrap(frame, predictions, resamples=50, seed=2026)
        self.assertEqual(len(intervals), len(DIMENSIONS) * len(MODEL_NAMES))
        self.assertTrue((draws.sum(axis=1) == 3).all())
        self.assertTrue((draws.to_numpy() >= 0).all())
        reference = intervals.loc[intervals["pca_components"].eq(128)]
        np.testing.assert_array_equal(reference["paired_rmse_delta_vs_pca128_ci_low"], np.zeros(len(MODEL_NAMES)))
        np.testing.assert_array_equal(reference["paired_rmse_delta_vs_pca128_ci_high"], np.zeros(len(MODEL_NAMES)))

    def test_primary_identity_ignores_split_but_not_response(self):
        frame = pd.DataFrame({
            "source_row_id": [0, 1], "drug_id": ["A", "B"],
            "cell_line_id": ["C1", "C2"], "drug_index": [0, 1],
            "cell_index": [0, 1], "split": ["train", "test"], "y": [1.2, 3.4],
        })
        digest = primary_identity_sha256(frame)
        changed_split = frame.copy()
        changed_split["split"] = ["test", "train"]
        self.assertEqual(primary_identity_sha256(changed_split), digest)
        changed_response = frame.copy()
        changed_response.loc[0, "y"] += 0.1
        self.assertNotEqual(primary_identity_sha256(changed_response), digest)

    def test_locked_protocol_matches_known_dimensions(self):
        protocol = json.loads((ROOT / "configs/robustness_pca.json").read_text(encoding="utf-8"))
        validate_protocol(protocol)
        changed = json.loads(json.dumps(protocol))
        changed["pca_sensitivity"]["components"] = [64, 128, 512]
        with self.assertRaisesRegex(ValueError, "PCA arms changed"):
            validate_protocol(changed)


if __name__ == "__main__":
    unittest.main()
