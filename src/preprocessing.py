"""Small reusable, label-free helpers for the GDSC2 preparation milestone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def assign_cell_lines(cell_ids: list[str], *, seed: int, train_fraction: float, validation_fraction: float) -> pd.DataFrame:
    """Freeze a grouped split using only sorted unique cell-line identifiers."""
    ordered = sorted(cell_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Cell IDs must be unique")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("Invalid split fractions")
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    n_train = round(train_fraction * len(ordered))
    n_validation = round(validation_fraction * len(ordered))
    split_by_position = [""] * len(ordered)
    rank_by_position = [0] * len(ordered)
    for rank, position in enumerate(permutation):
        split = "train" if rank < n_train else "validation" if rank < n_train + n_validation else "test"
        split_by_position[int(position)] = split
        rank_by_position[int(position)] = rank
    return pd.DataFrame({"cell_line_id": ordered, "split": split_by_position, "permutation_rank": rank_by_position})


def feature_catalog(symbols: list[str], prefix: str = "expr_") -> pd.DataFrame:
    """Give every input position a stable ID, including repeated symbols."""
    return pd.DataFrame({
        "position": np.arange(len(symbols), dtype=np.int32),
        "feature_id": [f"{prefix}{index:05d}" for index in range(len(symbols))],
        "gene_symbol": symbols,
    })


@dataclass
class ExpressionTransform:
    selector: VarianceThreshold
    scaler: StandardScaler
    pca: PCA
    train_cell_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]

    def transform(self, x: np.ndarray) -> np.ndarray:
        selected = self.selector.transform(x)
        scaled = self.scaler.transform(selected)
        with threadpool_limits(limits=1):
            return self.pca.transform(scaled)


def fit_expression_transform(
    expression: np.ndarray,
    cell_ids: list[str],
    train_cell_ids: list[str],
    feature_ids: list[str],
    settings: dict[str, object],
) -> ExpressionTransform:
    """Fit only to one profile per explicitly named training cell line."""
    if expression.shape != (len(cell_ids), len(feature_ids)):
        raise ValueError("Expression shape does not match ordered IDs/features")
    if len(cell_ids) != len(set(cell_ids)) or len(train_cell_ids) != len(set(train_cell_ids)):
        raise ValueError("Cell IDs are not unique")
    positions = {cell: index for index, cell in enumerate(cell_ids)}
    if not set(train_cell_ids).issubset(positions):
        raise ValueError("Unknown training cell ID")
    train = expression[[positions[cell] for cell in train_cell_ids]]
    if not np.isfinite(train).all():
        raise ValueError("Nonfinite training expression")
    selector = VarianceThreshold(threshold=float(settings["variance_threshold"]))
    selected = selector.fit_transform(train)
    if min(selected.shape) <= int(settings["pca_components"]):
        raise ValueError("Insufficient training rank/features for requested PCA")
    scaler = StandardScaler(with_mean=bool(settings["scaler_with_mean"]), with_std=bool(settings["scaler_with_std"]))
    scaled = scaler.fit_transform(selected)
    pca = PCA(
        n_components=int(settings["pca_components"]),
        svd_solver=str(settings["pca_solver"]),
        random_state=int(settings["pca_random_state"]),
        n_oversamples=int(settings["pca_n_oversamples"]),
        iterated_power=int(settings["pca_iterated_power"]),
        power_iteration_normalizer=str(settings["pca_power_iteration_normalizer"]),
        whiten=bool(settings["pca_whiten"]),
    )
    # Single BLAS thread improves reproducibility across repeat runs on this host.
    with threadpool_limits(limits=1):
        pca.fit(scaled)
    return ExpressionTransform(selector, scaler, pca, tuple(train_cell_ids), tuple(feature_ids))


def morgan_bit_matrix(smiles: list[str], settings: dict[str, object]) -> np.ndarray:
    """Create one deterministic bit vector per ordered drug SMILES."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(settings["radius"]),
        fpSize=int(settings["n_bits"]),
        countSimulation=bool(settings["count_simulation"]),
        includeChirality=bool(settings["include_chirality"]),
        useBondTypes=bool(settings["use_bond_types"]),
        onlyNonzeroInvariants=bool(settings["only_nonzero_invariants"]),
        includeRingMembership=bool(settings["include_ring_membership"]),
        includeRedundantEnvironments=bool(settings["include_redundant_environments"]),
    )
    bits = np.zeros((len(smiles), int(settings["n_bits"])), dtype=np.uint8)
    for index, structure in enumerate(smiles):
        molecule = Chem.MolFromSmiles(structure)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at position {index}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), bits[index])
    return bits
