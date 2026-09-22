"""Small MLP and index-based feature assembly for known-drug response."""

from __future__ import annotations

import torch
from torch import nn


class DrugResponseMLP(nn.Module):
    def __init__(self, input_dim: int = 2176, hidden: tuple[int, int] = (256, 128), dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.network[0].in_features:
            raise ValueError("Expected a two-dimensional aligned feature batch")
        return self.network(features).squeeze(-1)


def assemble_features(
    cell_index: torch.Tensor, drug_index: torch.Tensor,
    cell_features: torch.Tensor, drug_features: torch.Tensor,
) -> torch.Tensor:
    """Build only the current batch; never duplicate full feature tables per response."""
    if cell_index.ndim != 1 or drug_index.ndim != 1 or len(cell_index) != len(drug_index):
        raise ValueError("Cell and drug row indexes must align")
    return torch.cat((cell_features[cell_index], drug_features[drug_index]), dim=1)


@torch.inference_mode()
def predict_rows(
    model: DrugResponseMLP, cell_index: torch.Tensor, drug_index: torch.Tensor,
    cell_features: torch.Tensor, drug_features: torch.Tensor, batch_size: int,
) -> torch.Tensor:
    """Return one prediction per input row with dropout disabled."""
    model.eval()
    parts = []
    for start in range(0, len(cell_index), batch_size):
        features = assemble_features(
            cell_index[start:start + batch_size], drug_index[start:start + batch_size],
            cell_features, drug_features,
        )
        estimate = model(features)
        if estimate.shape != (len(features),):
            raise ValueError("Prediction/target shape mismatch")
        parts.append(estimate.detach().cpu())
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)
