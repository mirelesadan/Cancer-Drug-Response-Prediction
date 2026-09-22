"""Focused architecture and row-assembly checks; no study training here."""

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neural import DrugResponseMLP, assemble_features, predict_rows


class NeuralTests(unittest.TestCase):
    def test_architecture_and_unrestricted_scalar_output(self):
        model = DrugResponseMLP(2176, (256, 128), .1)
        linear = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]
        dropout = [layer for layer in model.network if isinstance(layer, torch.nn.Dropout)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linear], [(2176, 256), (256, 128), (128, 1)])
        self.assertEqual([layer.p for layer in dropout], [.1, .1])
        self.assertIsInstance(model.network[-1], torch.nn.Linear)
        output = model(torch.zeros((3, 2176)))
        self.assertEqual(tuple(output.shape), (3,))

    def test_feature_order_and_eval_replay(self):
        cells = torch.tensor([[1., 2.], [3., 4.]])
        drugs = torch.tensor([[0., 1., 0.], [1., 0., 1.]])
        cell_index = torch.tensor([1, 0, 1])
        drug_index = torch.tensor([0, 1, 1])
        expected = torch.tensor([[3., 4., 0., 1., 0.], [1., 2., 1., 0., 1.], [3., 4., 1., 0., 1.]])
        torch.testing.assert_close(assemble_features(cell_index, drug_index, cells, drugs), expected)
        model = DrugResponseMLP(5, (4, 2), .1)
        first = predict_rows(model, cell_index, drug_index, cells, drugs, 2)
        second = predict_rows(model, cell_index, drug_index, cells, drugs, 2)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
