"""Tests for windowing, splitting and error measures."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.data import make_windows, split
from forecasting.metrics import errors


@pytest.fixture
def series():
    """A rising series, so leakage across the split is easy to detect."""
    return np.arange(200, dtype=np.float64)


def test_windows_pair_history_with_what_follows():
    x, y = make_windows(np.arange(10), window=3, horizon=1)
    assert x.shape == (7, 3, 1)
    assert y.shape == (7, 1)
    assert list(x[0].ravel()) == [0, 1, 2]
    assert y[0][0] == 3


def test_windows_support_multi_step_horizons():
    x, y = make_windows(np.arange(10), window=3, horizon=2)
    assert x.shape == (6, 3, 1) and y.shape == (6, 2)
    assert list(y[0]) == [3, 4]


def test_windows_reject_a_series_that_is_too_short():
    with pytest.raises(ValueError):
        make_windows(np.arange(3), window=30)


def test_split_is_chronological_not_random(series):
    """Test targets must all come after the training targets, or the model is
    being asked to predict a past it was already shown."""
    windows = split(series, window=10, horizon=1, test_fraction=0.2)
    assert windows.y_train.max() < windows.y_test.min()


def test_scaler_is_fitted_on_the_training_section_only(series):
    """Training values scale into [0, 1]; the unseen tail must exceed 1."""
    windows = split(series, window=10, horizon=1, test_fraction=0.2)
    assert windows.x_train.max() <= 1.0 + 1e-9
    assert windows.y_test.max() > 1.0


def test_split_covers_the_series_without_gaps(series):
    windows = split(series, window=10, horizon=1, test_fraction=0.2)
    assert len(windows.x_train) > 0 and len(windows.x_test) > 0
    restored = windows.scaler.inverse_transform(windows.y_test.reshape(-1, 1)).ravel()
    assert restored[-1] == pytest.approx(series[-1])


def test_errors_are_zero_for_a_perfect_prediction():
    truth = np.array([1.0, 2.0, 3.0])
    result = errors(truth, truth)
    assert (result.mse, result.rmse, result.mae, result.mape) == (0.0, 0.0, 0.0, 0.0)


def test_error_measures_agree_with_hand_calculation():
    result = errors([10.0, 20.0], [12.0, 16.0])
    assert result.mse == pytest.approx(10.0)      # (4 + 16) / 2
    assert result.rmse == pytest.approx(np.sqrt(10.0))
    assert result.mae == pytest.approx(3.0)       # (2 + 4) / 2
    assert result.mape == pytest.approx(0.2)      # (0.2 + 0.2) / 2


def test_mape_skips_zero_valued_targets():
    """A zero true value has no defined percentage error; it must not become inf."""
    result = errors([0.0, 10.0], [1.0, 11.0])
    assert np.isfinite(result.mape)
    assert result.mape == pytest.approx(0.1)


def test_errors_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        errors([1.0, 2.0], [1.0])
