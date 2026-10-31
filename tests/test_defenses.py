"""Tests for :mod:`deepdefend.defenses`."""

import numpy as np
import pytest

import deepdefend
from deepdefend import defenses

tf = pytest.importorskip("tensorflow")

# name -> factory taking a model
PREPROCESSING_DEFENSES = {
    "feature_squeezing": lambda m: defenses.feature_squeezing(m, bit_depth=4),
    "input_transformation": lambda m: defenses.input_transformation(
        m, tf.keras.layers.Rescaling(0.5)
    ),
    "randomized_smoothing": lambda m: defenses.randomized_smoothing(m, noise_level=0.05, seed=0),
    "feature_denoising": lambda m: defenses.feature_denoising(m, method="median"),
    "thermometer_encoding": lambda m: defenses.thermometer_encoding(m, num_bins=8),
    "spatial_smoothing": lambda m: defenses.spatial_smoothing(m, kernel_size=3),
    "median_smoothing": lambda m: defenses.median_smoothing(m, kernel_size=3),
    "gaussian_blur": lambda m: defenses.gaussian_blur(m, kernel_size=3, sigma=1.0),
    "jpeg_compression": lambda m: defenses.jpeg_compression(m, quality=75),
    "pixel_deflection": lambda m: defenses.pixel_deflection(m, deflection_count=10, window_size=2, seed=0),
    "total_variation_minimization": lambda m: defenses.total_variation_minimization(m, iterations=2),
}


@pytest.mark.parametrize("name", sorted(PREPROCESSING_DEFENSES))
def test_defense_builds_and_predicts(name, untrained_model, batch):
    x, _ = batch
    defended = PREPROCESSING_DEFENSES[name](untrained_model)
    outputs = defended(x)
    assert tuple(outputs.shape) == (x.shape[0], 3)
    assert np.isfinite(np.asarray(outputs)).all()


@pytest.mark.parametrize("name", sorted(PREPROCESSING_DEFENSES))
def test_defense_models_can_be_saved_and_reloaded(name, untrained_model, batch, tmp_path):
    """Every wrapped defense used to fail at save/load (layers defined inline)."""
    x, _ = batch
    defended = PREPROCESSING_DEFENSES[name](untrained_model)
    path = tmp_path / f"{name}.keras"
    defended.save(path)
    reloaded = tf.keras.models.load_model(path)
    np.testing.assert_allclose(
        np.asarray(reloaded(x)), np.asarray(defended(x)), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    "name,builder",
    [
        ("feature_squeezing", lambda m: defenses.feature_squeezing(m)),
        ("thermometer_encoding", lambda m: defenses.thermometer_encoding(m)),
        ("gaussian_blur", lambda m: defenses.gaussian_blur(m)),
        ("median_smoothing", lambda m: defenses.median_smoothing(m)),
        ("spatial_smoothing", lambda m: defenses.spatial_smoothing(m)),
        ("jpeg_compression", lambda m: defenses.jpeg_compression(m)),
        ("pixel_deflection", lambda m: defenses.pixel_deflection(m, seed=0)),
        ("total_variation_minimization", lambda m: defenses.total_variation_minimization(m)),
        ("randomized_smoothing", lambda m: defenses.randomized_smoothing(m, seed=0)),
        ("feature_denoising", lambda m: defenses.feature_denoising(m)),
        ("input_transformation", lambda m: defenses.input_transformation(
            m, tf.keras.layers.Rescaling(1.0)
        )),
    ],
)
def test_gradient_flows_through_preprocessing(name, builder, untrained_model, batch):
    """Attacks against a defended model need a usable gradient."""
    x, _ = batch
    defended = builder(untrained_model)
    inputs = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        # A non-degenerate loss: summing a softmax output would be constant.
        loss = tf.reduce_sum(tf.square(defended(inputs)))
    gradient = tape.gradient(loss, inputs)
    assert gradient is not None, f"{name} blocks the gradient entirely"
    assert float(tf.reduce_sum(tf.abs(gradient))) > 0.0, f"{name} yields a zero gradient"


def test_attacks_work_against_jpeg_defended_model(trained_model, batch):
    """This used to raise `ValueError: Attempt to convert a value (None)`."""
    from deepdefend import attacks

    x, y = batch
    defended = defenses.jpeg_compression(trained_model)
    adversarial = attacks.fgsm(defended, x, np.eye(3)[y], epsilon=0.1)
    assert adversarial.shape == x.shape


# ---------------------------------------------------------------------------
# Behavioural correctness of the individual filters
# ---------------------------------------------------------------------------


def test_median_smoothing_is_not_an_average_pool(untrained_model, batch):
    x, _ = batch
    median = defenses.median_smoothing(untrained_model)
    average = untrained_model(tf.nn.avg_pool2d(x, 3, 1, "SAME"))
    assert not np.allclose(np.asarray(median(x)), np.asarray(average))


def test_gaussian_blur_uses_sigma(untrained_model, batch):
    x, _ = batch
    narrow = defenses.gaussian_blur(untrained_model, sigma=0.2)
    wide = defenses.gaussian_blur(untrained_model, sigma=5.0)
    assert not np.allclose(np.asarray(narrow(x)), np.asarray(wide(x)))


def test_feature_squeezing_output_range(untrained_model, batch):
    x, _ = batch
    layer = defenses.SqueezeLayer(bit_depth=2)
    squeezed = np.asarray(layer(x))
    assert squeezed.min() >= 0.0 and squeezed.max() <= 1.0
    values = np.unique(np.round(squeezed, 6))
    assert len(values) <= 4  # 2 bits -> levels 0, 1/3, 2/3, 1


def test_thermometer_encoding_output_range(untrained_model, batch):
    x, _ = batch
    layer = defenses.QuantizationLayer(num_bins=4)
    quantised = np.asarray(layer(x))
    assert quantised.min() >= 0.0 and quantised.max() <= 1.0


def test_randomized_smoothing_is_stochastic(untrained_model, batch):
    x, _ = batch
    defended = defenses.randomized_smoothing(untrained_model, noise_level=0.2, seed=None)
    assert not np.allclose(np.asarray(defended(x)), np.asarray(defended(x)))


def test_pixel_deflection_is_seeded(untrained_model, batch):
    x, _ = batch
    defended = defenses.pixel_deflection(untrained_model, seed=3)
    np.testing.assert_allclose(np.asarray(defended(x)), np.asarray(defended(x)))


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (defenses.feature_squeezing, {"bit_depth": 0}),
        (defenses.feature_squeezing, {"bit_depth": 9}),
        (defenses.thermometer_encoding, {"num_bins": 1}),
        (defenses.jpeg_compression, {"quality": 0}),
        (defenses.gaussian_blur, {"kernel_size": 4}),
        (defenses.median_smoothing, {"kernel_size": 2}),
        (defenses.randomized_smoothing, {"noise_level": -0.1}),
        (defenses.pixel_deflection, {"deflection_count": -1}),
    ],
)
def test_invalid_arguments_raise(builder, kwargs, untrained_model):
    with pytest.raises((ValueError, TypeError)):
        builder(untrained_model, **kwargs)


def test_input_transformation_requires_a_callable(untrained_model):
    with pytest.raises(ValueError):
        defenses.input_transformation(untrained_model)


@pytest.mark.parametrize(
    "builder",
    [
        defenses.gaussian_blur,
        defenses.median_smoothing,
        defenses.spatial_smoothing,
        defenses.jpeg_compression,
        defenses.pixel_deflection,
        defenses.total_variation_minimization,
        defenses.feature_denoising,
    ],
)
def test_image_only_defenses_reject_non_image_inputs(builder):
    """Silently returning the input unchanged would be a fake defense."""
    model = tf.keras.Sequential(
        [tf.keras.Input((10,)), tf.keras.layers.Dense(3, activation="softmax")]
    )
    with pytest.raises(ValueError, match="4-D"):
        builder(model)


def test_multi_input_models_are_rejected():
    left = tf.keras.Input((4,))
    right = tf.keras.Input((4,))
    merged = tf.keras.layers.Concatenate()([left, right])
    model = tf.keras.Model([left, right], tf.keras.layers.Dense(3)(merged))
    with pytest.raises(TypeError):
        defenses.gaussian_blur(model)


def test_subclassed_models_get_a_clear_error():
    class Subclassed(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(3)

        def call(self, inputs):
            return self.dense(inputs)

    with pytest.raises(TypeError):
        defenses.gaussian_blur(Subclassed())


# ---------------------------------------------------------------------------
# Training-time defenses
# ---------------------------------------------------------------------------


def test_adversarial_training_returns_a_defended_model(dataset):
    x, y = dataset
    model = tf.keras.Sequential(
        [
            tf.keras.Input((8, 8, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    defended = defenses.adversarial_training(
        model, x[:16], y[:16], epsilon=0.05, epochs=1, batch_size=8, verbose=0
    )
    assert isinstance(defended, tf.keras.Model)
    assert defended is not model
    assert defended.predict(x[:4], verbose=0).shape == (4, 3)
    # The caller's loss must be preserved (it used to be forced to categorical).
    assert defended.loss == "sparse_categorical_crossentropy"


def test_adversarial_training_generates_examples_in_batches(dataset, monkeypatch):
    """Adversarial examples must be generated per batch, not per sample."""
    x, y = dataset
    calls = []

    def spy(model, batch_x, batch_y, epsilon=0.01):
        calls.append(len(batch_x))
        return np.asarray(batch_x) + 0.01

    model = tf.keras.Sequential(
        [tf.keras.Input((8, 8, 3)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(3, activation="softmax")]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    defenses.adversarial_training(
        model, x[:16], y[:16], epochs=1, batch_size=8, attack=spy, verbose=0
    )
    assert calls == [8, 8]


def test_gradient_masking_warns_and_fits(untrained_model, batch):
    x, y = batch
    with pytest.warns(RuntimeWarning):
        wrapped = defenses.gradient_masking(untrained_model, mask_threshold=0.0)
    wrapped.compile(optimizer="adam", loss="categorical_crossentropy")
    wrapped.fit(x, np.eye(3)[y], epochs=1, verbose=0)


def test_defensive_distillation_fits(untrained_model, batch):
    x, y = batch
    distilled = defenses.defensive_distillation(untrained_model, untrained_model, temperature=3)
    distilled.compile(optimizer="adam", loss="categorical_crossentropy")
    history = distilled.fit(x, np.eye(3)[y], epochs=1, verbose=0)
    assert "loss" in history.history


def test_adversarial_logit_pairing_uses_adversarial_examples(untrained_model, batch):
    x, y = batch
    wrapped = defenses.adversarial_logit_pairing(untrained_model, epsilon=0.05, num_steps=1)

    generated = np.asarray(wrapped._adversarial(tf.constant(x), tf.constant(np.eye(3)[y])))
    # ALP must pair clean and *adversarial* logits; the old implementation
    # paired two models on the same clean batch.
    assert np.abs(generated - x).max() > 0.0

    wrapped.compile(optimizer="adam", loss="categorical_crossentropy")
    wrapped.fit(x, np.eye(3)[y], epochs=1, verbose=0)


def test_smooth_predict_returns_certified_radius(untrained_model, batch):
    x, _ = batch
    defended = defenses.randomized_smoothing(untrained_model, noise_level=0.1, seed=0)
    result = defenses.smooth_predict(defended, x, n_samples=5, seed=0)
    assert set(result) == {"predictions", "probabilities", "certified_radius"}
    assert result["predictions"].shape == (x.shape[0],)
    assert np.all(result["certified_radius"] >= 0.0)


def test_smooth_predict_requires_a_noise_level(untrained_model, batch):
    x, _ = batch
    with pytest.raises(ValueError):
        defenses.smooth_predict(untrained_model, x, n_samples=2)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def test_word_masking_preserves_whitespace():
    # The old implementation collapsed double spaces and newlines via str.split().
    masked = defenses.word_masking("a  b\nc", mask_prob=1.0)
    assert masked == "[MASK]  [MASK]\n[MASK]"


def test_word_masking_is_seeded():
    text = "one two three four five six"
    first = defenses.word_masking(text, mask_prob=0.5, seed=1)
    second = defenses.word_masking(text, mask_prob=0.5, seed=1)
    assert first == second


def test_word_masking_validates_probability():
    with pytest.raises(ValueError):
        defenses.word_masking("text", mask_prob=2.0)


def test_defenses_are_exported_from_the_package():
    for name in defenses.__all__:
        assert getattr(deepdefend, name) is getattr(defenses, name)
