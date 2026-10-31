"""Shared fixtures for the DeepDefend test-suite."""

import os

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

tf = pytest.importorskip("tensorflow")

NUM_CLASSES = 3
IMAGE_SHAPE = (8, 8, 3)
NUM_SAMPLES = 48


def _make_dataset(seed=0):
    """Linearly separable synthetic images so a tiny CNN can actually learn."""
    rng = np.random.default_rng(seed)
    x = rng.random((NUM_SAMPLES,) + IMAGE_SHAPE, dtype=np.float32) * 0.25
    y = rng.integers(0, NUM_CLASSES, size=NUM_SAMPLES)
    for i, label in enumerate(y):
        # A bright block whose position encodes the class.
        x[i, label * 2:label * 2 + 2, :, :] += 0.75
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return x, y.astype(np.int32)


def _small_cnn(activation="softmax"):
    return tf.keras.Sequential(
        [
            tf.keras.Input(IMAGE_SHAPE),
            tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_CLASSES, activation=activation),
        ]
    )


@pytest.fixture(scope="session")
def dataset():
    return _make_dataset()


@pytest.fixture(scope="session")
def trained_model(dataset):
    """A small CNN that is accurate enough for attacks to succeed."""
    x, y = dataset
    tf.keras.utils.set_random_seed(0)
    model = _small_cnn()
    # Label smoothing keeps the toy model from saturating to a float32
    # probability of exactly 1.0, which would make gradient-free attacks
    # (SPSA) unable to measure any loss difference at all.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.fit(x, np.eye(NUM_CLASSES)[y], epochs=6, batch_size=16, verbose=0)
    return model


@pytest.fixture(scope="session")
def linear_model(dataset):
    """A linear softmax classifier — easy to fool, so attacks are measurable."""
    x, y = dataset
    tf.keras.utils.set_random_seed(2)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(IMAGE_SHAPE),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(x, np.eye(NUM_CLASSES)[y], epochs=4, batch_size=16, verbose=0)
    return model


@pytest.fixture(scope="session")
def batch(dataset):
    """A small batch with one-hot and sparse labels."""
    x, y = dataset
    return x[:4], y[:4]


@pytest.fixture(scope="session")
def untrained_model():
    tf.keras.utils.set_random_seed(1)
    model = _small_cnn()
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model
