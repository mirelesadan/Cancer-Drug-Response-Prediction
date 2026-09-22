"""Small debugging checks; results here are not study performance."""

import sys
from pathlib import Path
import unittest

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import by_drug_validation, fingerprint_row_basis, ridge_path


class BaselineTests(unittest.TestCase):
    def test_ridge_path_matches_reference_svd(self):
        rng = np.random.default_rng(17)
        expression = rng.normal(size=(42, 3))
        drug_fingerprints = np.array([[1, 0, 1, 0, 0], [0, 1, 1, 0, 1], [1, 1, 0, 1, 0]], dtype=float)
        drug_index = np.arange(42) % 3
        bits = drug_fingerprints[drug_index]
        y = 2 + expression @ np.array([.5, -.3, .7]) + bits @ np.array([.2, 0, -.1, .4, .3])
        basis = fingerprint_row_basis(drug_fingerprints, 1e-12)
        compressed = np.column_stack([expression, bits @ basis])
        full = np.column_stack([expression, bits])
        alphas = [.01, .1, 1, 10]
        coefficients, intercepts = ridge_path(compressed, y, alphas)
        for index, alpha in enumerate(alphas):
            reference = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(full, y)
            predicted = compressed @ coefficients[index] + intercepts[index]
            np.testing.assert_allclose(predicted, reference.predict(full), rtol=0, atol=1e-9)
            expanded = np.concatenate([coefficients[index, :3], basis @ coefficients[index, 3:]])
            np.testing.assert_allclose(full @ expanded + intercepts[index], reference.predict(full), rtol=0, atol=1e-9)

    def test_undefined_within_drug_correlations_are_missing(self):
        drugs = np.array(["A"] * 3 + ["B"] * 2 + ["C"] * 3)
        y = np.array([1, 2, 3, 1, 2, 5, 5, 5], dtype=float)
        predicted = np.array([2, 2, 2, 1, 2, 4, 5, 6], dtype=float)
        table = by_drug_validation(drugs, y, predicted, {"A": 4, "B": 5, "C": 6}, minimum_spearman_n=3)
        self.assertEqual(table["spearman_status"].tolist(), ["constant_prediction", "insufficient_n", "constant_target"])
        self.assertTrue(table["spearman"].isna().all())


if __name__ == "__main__":
    unittest.main()
