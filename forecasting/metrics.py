"""Regression error measures, reported in the series' own price units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Errors:
    mse: float
    rmse: float
    mae: float
    mape: float

    def format(self) -> str:
        return (
            f"MSE {self.mse:10.2f}  RMSE {self.rmse:8.2f}  "
            f"MAE {self.mae:8.2f}  MAPE {self.mape:6.2%}"
        )


def errors(y_true: np.ndarray, y_predicted: np.ndarray) -> Errors:
    """Compare predictions to truth.

    MAPE skips any point where the true value is zero rather than dividing by it;
    an undefined percentage should not become an infinity that swallows the mean.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_predicted = np.asarray(y_predicted, dtype=np.float64).ravel()
    if y_true.shape != y_predicted.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} against {y_predicted.shape}")

    residual = y_true - y_predicted
    mse = float(np.mean(residual**2))

    nonzero = y_true != 0
    mape = float(np.mean(np.abs(residual[nonzero] / y_true[nonzero]))) if nonzero.any() else float("nan")

    return Errors(mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(residual))), mape=mape)
