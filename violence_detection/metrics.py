"""Classification metrics accumulated over a whole split.

Counts are summed across every sample before any ratio is taken. Precision,
recall and F1 are ratios of counts, so a mean of per-batch ratios is a different
and misleading quantity; only accuracy survives batch averaging, and then only
when the batches are equal-sized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @property
    def total(self) -> int:
        return (
            self.true_negatives + self.false_positives
            + self.false_negatives + self.true_positives
        )

    @property
    def accuracy(self) -> float:
        return self._ratio(self.true_positives + self.true_negatives, self.total)

    @property
    def precision(self) -> float:
        return self._ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        return self._ratio(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 0.0 if denominator == 0 else 2 * self.precision * self.recall / denominator

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else numerator / denominator

    def format(self) -> str:
        return (
            f"accuracy {self.accuracy:.3f}  precision {self.precision:.3f}  "
            f"recall {self.recall:.3f}  f1 {self.f1:.3f}"
        )


def from_confusion_matrix(matrix: np.ndarray) -> BinaryMetrics:
    """Read metrics off a 2x2 sklearn-ordered confusion matrix."""
    matrix = np.asarray(matrix)
    if matrix.shape != (2, 2):
        raise ValueError(f"expected a 2x2 matrix, got {matrix.shape}")
    (true_negatives, false_positives), (false_negatives, true_positives) = matrix
    return BinaryMetrics(
        int(true_negatives), int(false_positives), int(false_negatives), int(true_positives)
    )


def from_predictions(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> BinaryMetrics:
    """Score raw sigmoid outputs against labels, over the whole split at once."""
    y_true = np.asarray(y_true).ravel().astype(int)
    predicted = (np.asarray(probabilities).ravel() >= threshold).astype(int)
    if y_true.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} against {predicted.shape}")

    return BinaryMetrics(
        true_negatives=int(np.sum((y_true == 0) & (predicted == 0))),
        false_positives=int(np.sum((y_true == 0) & (predicted == 1))),
        false_negatives=int(np.sum((y_true == 1) & (predicted == 0))),
        true_positives=int(np.sum((y_true == 1) & (predicted == 1))),
    )
