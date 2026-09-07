"""Turning a price series into supervised windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler

WINDOW = 30


@dataclass(frozen=True)
class Windows:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler


def make_windows(series: np.ndarray, window: int = WINDOW, horizon: int = 1):
    """Slide a window over the series, pairing each window with what follows it."""
    series = np.asarray(series, dtype=np.float64).reshape(-1, 1)
    last_start = len(series) - window - horizon + 1
    if last_start <= 0:
        raise ValueError(
            f"series of {len(series)} is too short for window {window} and horizon {horizon}"
        )
    x = np.stack([series[i : i + window] for i in range(last_start)])
    y = np.stack([series[i + window : i + window + horizon, 0] for i in range(last_start)])
    return x, y


def split(series: np.ndarray, window: int = WINDOW, horizon: int = 1,
          test_fraction: float = 0.2) -> Windows:
    """Chronological train/test split, with the scaler fitted on the training part.

    The cut is a point in time: everything before it trains, everything after it
    tests. A random split would draw training windows from after test windows and
    ask the model to predict a past it had already seen, and fitting the scaler on
    the full series would carry the test period's range into training features.
    """
    series = np.asarray(series, dtype=np.float64).reshape(-1, 1)
    cut = int(len(series) * (1 - test_fraction))
    if cut <= window + horizon:
        raise ValueError("training section is too short for the requested window")

    scaler = MinMaxScaler().fit(series[:cut])
    scaled = scaler.transform(series)

    # Overlap the test section by `window` so the first test window has its history.
    x_train, y_train = make_windows(scaled[:cut], window, horizon)
    x_test, y_test = make_windows(scaled[cut - window:], window, horizon)
    return Windows(x_train, y_train, x_test, y_test, scaler)
