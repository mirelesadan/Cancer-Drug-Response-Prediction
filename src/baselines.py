"""Reusable known-drug Ridge models and validation summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, svd
from scipy.stats import spearmanr


def ridge_path(x: np.ndarray, y: np.ndarray, alphas: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Solve Ridge with an unpenalized intercept; reuse centered sufficient statistics.

    Returns coefficients (n_alpha, n_features) and intercepts (n_alpha,).
    Positive alpha makes the centered Gram system positive definite, including
    when a drug has fewer rows than expression features.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("Misaligned or empty Ridge inputs")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite Ridge input")
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("Ridge alphas must be positive")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered = x - x_mean
    gram = centered.T @ centered
    rhs = centered.T @ (y - y_mean)
    identity = np.eye(x.shape[1], dtype=np.float64)
    coefficients = np.empty((len(alphas), x.shape[1]), dtype=np.float64)
    intercepts = np.empty(len(alphas), dtype=np.float64)
    for index, alpha in enumerate(alphas):
        factor = cho_factor(gram + float(alpha) * identity, lower=True, check_finite=False)
        coef = cho_solve(factor, rhs, check_finite=False)
        coefficients[index] = coef
        intercepts[index] = y_mean - float(x_mean @ coef)
    return coefficients, intercepts


def fingerprint_row_basis(training_fingerprints: np.ndarray, relative_tolerance: float) -> np.ndarray:
    """Orthonormal basis preserving linear Ridge predictions and L2 penalty."""
    matrix = np.asarray(training_fingerprints, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("Invalid training fingerprint matrix")
    _, singular_values, vt = svd(matrix, full_matrices=False, lapack_driver="gesdd")
    rank = int(np.sum(singular_values > singular_values[0] * relative_tolerance))
    if not rank:
        raise ValueError("No fingerprint rank")
    basis = vt[:rank].T.copy()
    if not np.allclose(matrix @ basis @ basis.T, matrix, rtol=0, atol=1e-8):
        raise ValueError("Fingerprint basis lost training information")
    return basis


@dataclass
class PerDrugMean:
    means: np.ndarray
    trained_drugs: np.ndarray

    def predict(self, drug_index: np.ndarray) -> np.ndarray:
        ids = np.asarray(drug_index, dtype=np.int64)
        if not self.trained_drugs[ids].all():
            raise ValueError("Drug absent from training")
        return self.means[ids]


@dataclass
class PooledRidge:
    expression_coef: np.ndarray
    fingerprint_coef: np.ndarray
    intercept: float
    alpha: float
    trained_drugs: np.ndarray

    def predict(self, cell_index: np.ndarray, drug_index: np.ndarray, scores: np.ndarray, fingerprints: np.ndarray) -> np.ndarray:
        cells = np.asarray(cell_index, dtype=np.int64)
        drugs = np.asarray(drug_index, dtype=np.int64)
        if not self.trained_drugs[drugs].all():
            raise ValueError("Drug absent from training")
        return scores[cells] @ self.expression_coef + fingerprints[drugs] @ self.fingerprint_coef + self.intercept


@dataclass
class PerDrugRidge:
    expression_coef: np.ndarray
    intercept: np.ndarray
    alpha: float
    trained_drugs: np.ndarray

    def predict(self, cell_index: np.ndarray, drug_index: np.ndarray, scores: np.ndarray) -> np.ndarray:
        cells = np.asarray(cell_index, dtype=np.int64)
        drugs = np.asarray(drug_index, dtype=np.int64)
        if not self.trained_drugs[drugs].all():
            raise ValueError("Drug absent from training")
        return np.einsum("ij,ij->i", scores[cells], self.expression_coef[drugs]) + self.intercept[drugs]


def errors(y: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    actual = np.asarray(y, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if actual.shape != estimate.shape or not np.isfinite(estimate).all():
        raise ValueError("Prediction shape or finiteness failure")
    residual = estimate - actual
    return float(np.sqrt(np.mean(residual * residual))), float(np.mean(np.abs(residual)))


def by_drug_validation(
    drug_names: np.ndarray, y: np.ndarray, predicted: np.ndarray,
    train_counts: dict[str, int], *, minimum_spearman_n: int,
) -> pd.DataFrame:
    """Each drug receives one metric row; undefined correlations stay missing."""
    frame = pd.DataFrame({"drug_id": drug_names, "y": y, "predicted": predicted})
    rows = []
    for drug, group in frame.groupby("drug_id", sort=True):
        observed = group["y"].to_numpy(dtype=np.float64)
        estimated = group["predicted"].to_numpy(dtype=np.float64)
        rmse, mae = errors(observed, estimated)
        if len(group) < minimum_spearman_n:
            status, rho = "insufficient_n", np.nan
        elif np.unique(observed).size < 2:
            status, rho = "constant_target", np.nan
        elif np.unique(estimated).size < 2:
            status, rho = "constant_prediction", np.nan
        else:
            rho = float(spearmanr(observed, estimated).statistic)
            status = "defined" if np.isfinite(rho) else "undefined_numeric"
            if status != "defined":
                rho = np.nan
        rows.append({
            "drug_id": drug, "train_n": int(train_counts[drug]), "validation_n": len(group),
            "rmse": rmse, "mae": mae, "spearman": rho, "spearman_status": status,
        })
    return pd.DataFrame(rows)
