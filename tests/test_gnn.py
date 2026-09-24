"""Focused graph-order and numerical checks for the molecular extension."""

import sys
from pathlib import Path
import unittest

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn import ATOM_DIM, BOND_DIM, MolecularResponseGNN, build_drug_graphs, predict_rows, smiles_graph


class GNNTests(unittest.TestCase):
    def test_graph_order_and_deterministic_representation(self):
        drugs = pd.DataFrame({"drug_index": [0, 1], "drug_id": ["methane", "ethanol"], "smiles": ["C", "CCO"]})
        first, first_hash = build_drug_graphs(drugs)
        second, second_hash = build_drug_graphs(drugs.copy())
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first.num_graphs, 2)
        self.assertEqual(first.ptr.tolist(), [0, 1, 4])
        self.assertEqual(first.x.shape[1], ATOM_DIM)
        self.assertEqual(first.edge_attr.shape[1], BOND_DIM)
        torch.testing.assert_close(first.x, second.x, rtol=0, atol=0)
        torch.testing.assert_close(first.edge_index, second.edge_index, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            build_drug_graphs(drugs.iloc[::-1].reset_index(drop=True))
        with self.assertRaises(ValueError):
            smiles_graph("not_a_molecule")

    def test_finite_graph_gradient_and_aligned_inference(self):
        drugs = pd.DataFrame({"drug_index": [0, 1], "drug_id": ["ethanol", "ethylamine"], "smiles": ["CCO", "CCN"]})
        graphs, _ = build_drug_graphs(drugs)
        model = MolecularResponseGNN(n_drugs=2, expression_dim=3, graph_dim=8, fusion_hidden=(12, 6), dropout=0.1)
        cells = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
        cell_index = torch.tensor([0, 1, 0])
        drug_index = torch.tensor([0, 1, 1])
        target = torch.tensor([1.0, 2.0, -1.0])
        model.train()
        estimate = model(cells[cell_index], drug_index, graphs)
        self.assertEqual(tuple(estimate.shape), (3,))
        loss = torch.nn.functional.mse_loss(estimate, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.graph_layers[0].parameters()))
        first = predict_rows(model, cell_index, drug_index, cells, graphs, 2)
        second = predict_rows(model, cell_index, drug_index, cells, graphs, 3)
        torch.testing.assert_close(first, second, rtol=0, atol=1e-7)
        self.assertFalse(model.training)
        with self.assertRaises(ValueError):
            predict_rows(model, cell_index[:2], drug_index, cells, graphs, 2)


if __name__ == "__main__":
    unittest.main()
