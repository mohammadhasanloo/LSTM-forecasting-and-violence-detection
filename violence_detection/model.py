"""ResNet50 features per frame, then a ConvLSTM over time."""

from __future__ import annotations

import keras
from keras import layers

FRAMES = 20
FRAME_SIZE = 224


def classification_metrics() -> list:
    """Stateful metric objects, which accumulate counts across batches.

    Metric classes rather than plain functions, because Keras averages a function
    metric per batch and a mean of ratios is not the ratio of sums.
    """
    return [
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]


def build_model(
    frames: int = FRAMES,
    height: int = FRAME_SIZE,
    width: int = FRAME_SIZE,
    weights: str | None = "imagenet",
    learning_rate: float = 1e-4,
) -> keras.Model:
    """Classify a clip as violent or not.

    ResNet50 is applied to every frame through ``TimeDistributed``, giving a
    sequence of spatial feature maps. ``ConvLSTM2D`` then reads that sequence
    while keeping the spatial layout, so motion across frames stays localised
    rather than being flattened away before the temporal model sees it.

    Metrics are the built-in Keras classes, which accumulate counts across
    batches rather than averaging a ratio per batch.
    """
    backbone = keras.applications.ResNet50(
        weights=weights, include_top=False, input_shape=(height, width, 3)
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(frames, height, width, 3))
    x = layers.TimeDistributed(backbone)(inputs)
    x = layers.ConvLSTM2D(256, 3, strides=1, padding="same", return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1000, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(10, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="violence")(x)

    model = keras.Model(inputs, outputs, name="violence_detector")
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=classification_metrics(),
    )
    return model
