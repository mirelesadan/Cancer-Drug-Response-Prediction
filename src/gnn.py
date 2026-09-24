"""Deterministic RDKit graphs and a small expression-plus-molecule GNN."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_mean_pool


HYBRIDIZATIONS = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
)
CHIRALITIES = (
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
BOND_TYPES = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)
BOND_STEREO = (
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOANY,
)
ATOM_DIM = 119 + 8 + 8 + 8 + 6 + 4 + 2
BOND_DIM = 5 + 5 + 2


def _one_hot(index: int, size: int) -> list[float]:
    if not 0 <= index < size:
        raise ValueError("Feature category out of range")
    return [1.0 if position == index else 0.0 for position in range(size)]


def _known_or_other(value: object, known: tuple) -> int:
    return known.index(value) if value in known else len(known)


def atom_features(atom: Chem.Atom) -> list[float]:
    atomic_number = int(atom.GetAtomicNum())
    degree = int(atom.GetDegree())
    charge = int(atom.GetFormalCharge())
    hydrogens = int(atom.GetTotalNumHs())
    return (
        _one_hot(atomic_number if 0 <= atomic_number <= 118 else 0, 119)
        + _one_hot(degree if degree <= 6 else 7, 8)
        + _one_hot(charge + 3 if -3 <= charge <= 3 else 7, 8)
        + _one_hot(_known_or_other(atom.GetHybridization(), HYBRIDIZATIONS), 8)
        + _one_hot(hydrogens if hydrogens <= 4 else 5, 6)
        + _one_hot(_known_or_other(atom.GetChiralTag(), CHIRALITIES), 4)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(_known_or_other(bond.GetBondType(), BOND_TYPES), 5)
        + _one_hot(_known_or_other(bond.GetStereo(), BOND_STEREO), 5)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )


def smiles_graph(smiles: str) -> Data:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles}")
    atoms = torch.tensor([atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.float32)
    edge_pairs: list[tuple[int, int]] = []
    edge_values: list[list[float]] = []
    for bond in molecule.GetBonds():
        a, b = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        features = bond_features(bond)
        edge_pairs.extend(((a, b), (b, a)))
        edge_values.extend((features, features))
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).T.contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_values, dtype=torch.float32) if edge_values else torch.empty((0, BOND_DIM), dtype=torch.float32)
    if atoms.shape != (molecule.GetNumAtoms(), ATOM_DIM) or edge_attr.shape != (2 * molecule.GetNumBonds(), BOND_DIM):
        raise ValueError("Molecular graph feature shape mismatch")
    return Data(x=atoms, edge_index=edge_index, edge_attr=edge_attr)


def build_drug_graphs(drugs: pd.DataFrame) -> tuple[Batch, str]:
    """Preserve frozen drug_index order and hash the complete graph representation."""
    if list(drugs.columns) != ["drug_index", "drug_id", "smiles"]:
        raise ValueError("Unexpected frozen drug catalog schema")
    if drugs["drug_index"].tolist() != list(range(len(drugs))) or drugs["drug_id"].duplicated().any():
        raise ValueError("Drug index/order is not a unique zero-based catalog")
    graphs = []
    digest = hashlib.sha256()
    for drug_index, drug_id, smiles in drugs.itertuples(index=False, name=None):
        graph = smiles_graph(str(smiles))
        graphs.append(graph)
        for item in (str(drug_index), str(drug_id), str(smiles)):
            digest.update(item.encode("utf-8") + b"\0")
        for item in (graph.x, graph.edge_index, graph.edge_attr):
            array = item.numpy()
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    result = Batch.from_data_list(graphs)
    if result.num_graphs != len(drugs) or result.x.shape[1] != ATOM_DIM or result.edge_attr.shape[1] != BOND_DIM:
        raise ValueError("Batched graph order or shape mismatch")
    return result, digest.hexdigest()


class MolecularResponseGNN(nn.Module):
    def __init__(self, n_drugs: int, expression_dim: int = 128, graph_dim: int = 64,
                 fusion_hidden: tuple[int, int] = (256, 128), dropout: float = 0.1):
        super().__init__()
        self.n_drugs = n_drugs
        self.expression_dim = expression_dim
        self.atom_projection = nn.Linear(ATOM_DIM, graph_dim)
        self.graph_layers = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(graph_dim, graph_dim), nn.ReLU(), nn.Linear(graph_dim, graph_dim)),
                     edge_dim=BOND_DIM, train_eps=True)
            for _ in range(2)
        ])
        self.dropout = float(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(expression_dim + graph_dim, fusion_hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fusion_hidden[0], fusion_hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fusion_hidden[1], 1),
        )

    def encode_molecules(self, graphs: Batch) -> torch.Tensor:
        if graphs.num_graphs != self.n_drugs or graphs.x.shape[1] != ATOM_DIM or graphs.edge_attr.shape[1] != BOND_DIM:
            raise ValueError("Molecular graph catalog mismatch")
        node = F.relu(self.atom_projection(graphs.x))
        for convolution in self.graph_layers:
            node = F.relu(convolution(node, graphs.edge_index, graphs.edge_attr))
            node = F.dropout(node, p=self.dropout, training=self.training)
        pooled = global_mean_pool(node, graphs.batch, size=self.n_drugs)
        if pooled.shape != (self.n_drugs, self.atom_projection.out_features):
            raise ValueError("Molecular embedding shape mismatch")
        return pooled

    def predict_encoded(self, expression: torch.Tensor, drug_index: torch.Tensor,
                        drug_embeddings: torch.Tensor) -> torch.Tensor:
        if expression.ndim != 2 or expression.shape[1] != self.expression_dim or drug_index.shape != (len(expression),):
            raise ValueError("Expression/drug row alignment failed")
        if torch.any(drug_index < 0) or torch.any(drug_index >= self.n_drugs):
            raise ValueError("Unknown drug index")
        estimate = self.fusion(torch.cat((expression, drug_embeddings[drug_index]), dim=1)).squeeze(-1)
        if estimate.shape != (len(expression),):
            raise ValueError("Prediction/target shape mismatch")
        return estimate

    def forward(self, expression: torch.Tensor, drug_index: torch.Tensor, graphs: Batch) -> torch.Tensor:
        return self.predict_encoded(expression, drug_index, self.encode_molecules(graphs))


@torch.inference_mode()
def predict_rows(model: MolecularResponseGNN, cell_index: torch.Tensor, drug_index: torch.Tensor,
                 expression_scores: torch.Tensor, graphs: Batch, batch_size: int) -> torch.Tensor:
    """Evaluate one graph encoding for the catalog, then aligned response rows."""
    if cell_index.shape != drug_index.shape or cell_index.ndim != 1:
        raise ValueError("Cell and drug row indexes must align")
    model.eval()
    embeddings = model.encode_molecules(graphs)
    parts = []
    for start in range(0, len(cell_index), batch_size):
        cells = cell_index[start:start + batch_size]
        drugs = drug_index[start:start + batch_size]
        parts.append(model.predict_encoded(expression_scores[cells], drugs, embeddings).cpu())
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)
