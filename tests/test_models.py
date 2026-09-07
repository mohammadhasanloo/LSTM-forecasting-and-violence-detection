"""Model wiring tests. Built without pretrained weights so they need no download."""

from __future__ import annotations

import numpy as np

from forecasting.models import build_baseline_model, build_hybrid_model
from violence_detection.model import build_model, classification_metrics

WINDOW = 30


def test_forecasting_models_map_a_window_to_a_prediction():
    batch = np.random.default_rng(0).random((4, WINDOW, 1)).astype("float32")
    for builder in (build_hybrid_model, build_baseline_model):
        model = builder(window=WINDOW, horizon=1)
        assert model.predict(batch, verbose=0).shape == (4, 1)


def test_hybrid_model_supports_multi_step_horizons():
    model = build_hybrid_model(window=WINDOW, horizon=5)
    batch = np.random.default_rng(1).random((2, WINDOW, 1)).astype("float32")
    assert model.predict(batch, verbose=0).shape == (2, 5)


def test_hybrid_output_is_linear_not_relu():
    """A ReLU head cannot predict below zero, so an under-prediction has no gradient."""
    model = build_hybrid_model(window=WINDOW)
    assert model.layers[-1].activation.__name__ == "linear"


def test_hybrid_model_has_both_recurrent_branches():
    names = {type(layer).__name__ for layer in build_hybrid_model(window=WINDOW).layers}
    assert {"LSTM", "GRU", "Concatenate"} <= names


def test_violence_model_scores_a_clip():
    model = build_model(frames=2, height=32, width=32, weights=None)
    clip = np.random.default_rng(2).random((1, 2, 32, 32, 3)).astype("float32")
    probability = model.predict(clip, verbose=0)
    assert probability.shape == (1, 1)
    assert 0.0 <= probability.min() and probability.max() <= 1.0


def test_violence_model_uses_accumulating_metric_classes():
    """Precision and recall must be stateful Metric objects, not per-batch functions.

    A function metric is averaged over batches by Keras, and a mean of ratios is
    not the ratio of sums.
    """
    import keras

    from violence_detection.model import classification_metrics

    metrics = classification_metrics()
    stateful = {m.name: m for m in metrics if isinstance(m, keras.metrics.Metric)}
    assert {"precision", "recall"} <= set(stateful)
    assert not any(callable(m) and not isinstance(m, keras.metrics.Metric) for m in metrics
                   if not isinstance(m, str))
