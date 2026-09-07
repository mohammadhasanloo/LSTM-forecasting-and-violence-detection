"""Tests for metrics accumulated over a whole split."""

from __future__ import annotations

import numpy as np
import pytest

from violence_detection.metrics import from_confusion_matrix, from_predictions

# The confusion matrix measured on the held-out test set.
ORIGINAL_MATRIX = np.array([[72, 3], [5, 70]])


def test_metrics_match_the_measured_confusion_matrix():
    """Every metric must follow from the counts, not from any per-batch average."""
    metrics = from_confusion_matrix(ORIGINAL_MATRIX)
    assert metrics.total == 150
    assert metrics.accuracy == pytest.approx(142 / 150)
    assert metrics.precision == pytest.approx(70 / 73)
    assert metrics.recall == pytest.approx(70 / 75)
    assert metrics.f1 == pytest.approx(2 * (70 / 73) * (70 / 75) / ((70 / 73) + (70 / 75)))


def test_batch_averaged_precision_differs_from_accumulated_precision():
    """A mean of per-batch ratios is not the ratio of summed counts."""
    y_true = np.array([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.9, 0.9, 0.1])

    accumulated = from_predictions(y_true, probabilities).precision  # 2 TP, 1 FP -> 2/3
    batch_means = np.mean([
        from_predictions(y_true[:2], probabilities[:2]).precision,  # 1 TP, 1 FP -> 1/2
        from_predictions(y_true[2:], probabilities[2:]).precision,  # 1 TP, 0 FP -> 1
    ])
    assert accumulated == pytest.approx(2 / 3)
    assert batch_means == pytest.approx(0.75)
    assert accumulated != pytest.approx(batch_means)


def test_perfect_and_useless_predictions():
    perfect = from_predictions([0, 1], [0.1, 0.9])
    assert (perfect.accuracy, perfect.precision, perfect.recall, perfect.f1) == (1.0,) * 4

    inverted = from_predictions([0, 1], [0.9, 0.1])
    assert inverted.accuracy == 0.0
    assert inverted.f1 == 0.0


def test_metrics_are_zero_rather_than_undefined_when_nothing_is_predicted():
    metrics = from_predictions([0, 1], [0.1, 0.1])
    assert metrics.precision == 0.0
    assert metrics.f1 == 0.0


def test_confusion_matrix_must_be_two_by_two():
    with pytest.raises(ValueError):
        from_confusion_matrix(np.eye(3))


def test_threshold_shifts_the_operating_point():
    y_true = np.array([0, 1, 1])
    probabilities = np.array([0.4, 0.45, 0.9])
    assert from_predictions(y_true, probabilities, threshold=0.5).recall == pytest.approx(0.5)
    assert from_predictions(y_true, probabilities, threshold=0.42).recall == pytest.approx(1.0)
