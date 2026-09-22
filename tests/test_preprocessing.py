"""Focused synthetic checks for consequential preparation failure modes."""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import assign_cell_lines, feature_catalog, fit_expression_transform, morgan_bit_matrix


class PreprocessingTests(unittest.TestCase):
    def test_split_uses_sorted_unique_ids_and_is_disjoint(self):
        ids = [f"cell_{i:03d}" for i in range(20)]
        a = assign_cell_lines(ids[::-1], seed=17, train_fraction=.7, validation_fraction=.15)
        b = assign_cell_lines(ids, seed=17, train_fraction=.7, validation_fraction=.15)
        self.assertTrue(a.equals(b))
        self.assertEqual(a["split"].value_counts().to_dict(), {"train": 14, "validation": 3, "test": 3})
        self.assertEqual(a.groupby("cell_line_id")["split"].nunique().max(), 1)

    def test_duplicate_symbols_keep_distinct_positions(self):
        catalog = feature_catalog(["A", "B", "A"])
        self.assertEqual(catalog["gene_symbol"].tolist(), ["A", "B", "A"])
        self.assertEqual(catalog["feature_id"].tolist(), ["expr_00000", "expr_00001", "expr_00002"])

    def test_fit_uses_training_profiles_once_and_is_finite(self):
        ids = ["a", "b", "c", "d", "heldout"]
        x = np.array([[1, 2, 3, 7], [2, 3, 5, 7], [3, 5, 7, 7], [5, 8, 11, 7], [9, 99, 100, 7]], dtype=float)
        settings = dict(variance_threshold=0, scaler_with_mean=True, scaler_with_std=True,
                        pca_components=2, pca_solver="randomized", pca_random_state=17,
                        pca_n_oversamples=2, pca_iterated_power=3,
                        pca_power_iteration_normalizer="QR", pca_whiten=False)
        first = fit_expression_transform(x, ids, ids[:4], ["f0", "f1", "f2", "f3"], settings)
        changed = x.copy()
        changed[4, :3] = 1e9
        second = fit_expression_transform(changed, ids, ids[:4], ["f0", "f1", "f2", "f3"], settings)
        np.testing.assert_array_equal(first.selector.get_support(), [True, True, True, False])
        np.testing.assert_allclose(first.scaler.mean_, second.scaler.mean_)
        np.testing.assert_allclose(first.pca.components_, second.pca.components_)
        np.testing.assert_allclose(first.transform(x[:4]), second.transform(changed[:4]))
        self.assertTrue(np.isfinite(first.transform(x)).all())

    def test_morgan_bits_are_deterministic(self):
        settings = dict(radius=2, n_bits=2048, count_simulation=False, include_chirality=True,
                        use_bond_types=True, only_nonzero_invariants=False,
                        include_ring_membership=True, include_redundant_environments=False)
        a = morgan_bit_matrix(["CCO", "CCN"], settings)
        b = morgan_bit_matrix(["CCO", "CCN"], settings)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (2, 2048))
        self.assertTrue(np.isin(a, [0, 1]).all())
        self.assertFalse(np.array_equal(a[0], a[1]))


if __name__ == "__main__":
    unittest.main()
