"""Sequence models for next-step price prediction."""

from __future__ import annotations

import keras
from keras import layers

from forecasting.data import WINDOW


def build_hybrid_model(window: int = WINDOW, horizon: int = 1) -> keras.Model:
    """Parallel LSTM and GRU branches over the same window, concatenated.

    The two recurrent families see identical input and their summaries are joined
    before the output, so the head can lean on whichever branch is carrying more
    signal rather than being committed to one gate design.
    """
    inputs = keras.Input(shape=(window, 1))

    lstm = layers.LSTM(30, return_sequences=True)(inputs)
    lstm = layers.Dropout(0.5)(lstm)
    lstm = layers.LSTM(50)(lstm)
    lstm = layers.Dense(10, activation="relu")(lstm)

    gru = layers.GRU(30)(inputs)
    gru = layers.Dropout(0.5)(gru)
    gru = layers.Dense(10, activation="relu")(gru)

    merged = layers.Concatenate()([lstm, gru])
    # Linear output: the series is min-max scaled, but a ReLU here would clamp
    # every under-prediction at zero and kill the gradient that corrects it.
    outputs = layers.Dense(horizon)(merged)

    model = keras.Model(inputs, outputs, name="hybrid_lstm_gru")
    model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
    return model


def build_baseline_model(window: int = WINDOW, horizon: int = 1) -> keras.Model:
    """A single LSTM layer, the thing the hybrid model is measured against."""
    inputs = keras.Input(shape=(window, 1))
    x = layers.LSTM(50)(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(horizon)(x)

    model = keras.Model(inputs, outputs, name="baseline_lstm")
    model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
    return model
