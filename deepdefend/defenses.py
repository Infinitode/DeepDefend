# defenses.py
"""
Functions to apply adversarial defense mechanisms to deep learning models.

Each model-level defense returns a new ``tf.keras.Model`` that applies a
preprocessing step before the original model.  The preprocessing layers are
module level, serialisable and registered with Keras, so defended models can be
saved and reloaded with ``model.save()`` / ``tf.keras.models.load_model()``.

Preprocessing defenses are *not* a guarantee of robustness.  The docstrings of
each function state what the technique does and does not buy you; see also the
"Limitations" section of the README.

Available functions:
- `adversarial_training(model, x, y, epsilon=0.01)`: Adversarial Training.
- `feature_squeezing(model, bit_depth=4)`: Feature Squeezing.
- `input_transformation(model, transformation_function)`: Input Transformation.
- `defensive_distillation(model, teacher_model, temperature=2)`: Defensive Distillation.
- `randomized_smoothing(model, noise_level=0.1)`: Randomized Smoothing (inference-time noise).
- `smooth_predict(model, x, n_samples=100)`: Monte-Carlo prediction + certified radius.
- `feature_denoising(model, method="median")`: Feature/input Denoising.
- `thermometer_encoding(model, num_bins=10)`: shape-preserving quantisation.
- `adversarial_logit_pairing(model, paired_model)`: Adversarial Logit Pairing.
- `spatial_smoothing(model, kernel_size=3)`: Spatial Smoothing (median filter).
- `jpeg_compression(model, quality=75)`: JPEG Compression.
- `pixel_deflection(model, deflection_count=100, window_size=10)`: Pixel Deflection.
- `gaussian_blur(model, kernel_size=3, sigma=1.0)`: Gaussian Blur.
- `total_variation_minimization(model, iterations=10)`: Total Variation Minimization.
- `word_masking(text, mask_token="[MASK]", mask_prob=0.1)`: Word masking for text.
- `median_smoothing(model, kernel_size=3)`: Median Smoothing (alias of `spatial_smoothing`).
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from . import attacks
from ._utils import (
    check_float,
    check_int,
    infer_from_logits,
    straight_through,
    tokenize_text,
    wrap_preprocessing,
)

__all__ = [
    "adversarial_logit_pairing",
    "adversarial_training",
    "defensive_distillation",
    "feature_denoising",
    "feature_squeezing",
    "gaussian_blur",
    "input_transformation",
    "jpeg_compression",
    "median_smoothing",
    "pixel_deflection",
    "randomized_smoothing",
    "smooth_predict",
    "spatial_smoothing",
    "thermometer_encoding",
    "total_variation_minimization",
    "word_masking",
]

_DEFAULT_RANGE = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Serialisable preprocessing layers
# ---------------------------------------------------------------------------


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class SqueezeLayer(tf.keras.layers.Layer):
    """Uniform quantisation of the input ("feature squeezing")."""

    def __init__(self, bit_depth=4, straight_through_gradient=True, value_range=_DEFAULT_RANGE, **kwargs):
        super().__init__(**kwargs)
        self.bit_depth = check_int("bit_depth", bit_depth, minimum=1, maximum=8)
        self.straight_through_gradient = bool(straight_through_gradient)
        self.value_range = tuple(float(v) for v in value_range)

    def call(self, inputs):
        lo, hi = self.value_range
        clipped = tf.clip_by_value(inputs, lo, hi)
        levels = float(2 ** self.bit_depth - 1)
        squeezed = tf.round(clipped * levels) / levels
        if self.straight_through_gradient:
            return straight_through(squeezed, inputs)
        return tf.stop_gradient(squeezed)

    def get_config(self):
        config = super().get_config()
        config.update(
            bit_depth=self.bit_depth,
            straight_through_gradient=self.straight_through_gradient,
            value_range=self.value_range,
        )
        return config


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class QuantizationLayer(tf.keras.layers.Layer):
    """Uniform quantisation into ``num_bins`` levels (shape preserving)."""

    def __init__(self, num_bins=10, straight_through_gradient=True, value_range=_DEFAULT_RANGE, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = check_int("num_bins", num_bins, minimum=2)
        self.straight_through_gradient = bool(straight_through_gradient)
        self.value_range = tuple(float(v) for v in value_range)

    def call(self, inputs):
        lo, hi = self.value_range
        clipped = tf.clip_by_value(inputs, lo, hi)
        bins = float(self.num_bins - 1)
        quantised = tf.round((clipped - lo) / (hi - lo) * bins) / bins * (hi - lo) + lo
        if self.straight_through_gradient:
            return straight_through(quantised, inputs)
        return tf.stop_gradient(quantised)

    def get_config(self):
        config = super().get_config()
        config.update(
            num_bins=self.num_bins,
            straight_through_gradient=self.straight_through_gradient,
            value_range=self.value_range,
        )
        return config


def _gaussian_kernel_1d(kernel_size, sigma, dtype=tf.float32):
    radius = kernel_size // 2
    coords = tf.cast(tf.range(-radius, radius + 1), dtype)
    kernel = tf.exp(-tf.square(coords) / (2.0 * sigma * sigma))
    return kernel / tf.reduce_sum(kernel)


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class GaussianBlurLayer(tf.keras.layers.Layer):
    """Separable Gaussian blur (per channel) applied to 4-D image tensors."""

    def __init__(self, kernel_size=3, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        kernel_size = check_int("kernel_size", kernel_size, minimum=1)
        if kernel_size % 2 == 0:
            raise ValueError(f"`kernel_size` must be odd, got {kernel_size}.")
        self.kernel_size = kernel_size
        self.sigma = check_float("sigma", sigma, minimum=1e-6)

    def call(self, inputs):
        if inputs.shape.rank != 4:
            raise ValueError(
                "GaussianBlurLayer requires 4-D image inputs (batch, height, width, channels); "
                f"received shape {inputs.shape}."
            )
        kernel = _gaussian_kernel_1d(self.kernel_size, self.sigma, inputs.dtype)
        kernel = tf.tensordot(kernel, kernel, axes=0)
        kernel = tf.reshape(kernel, (self.kernel_size, self.kernel_size, 1, 1))
        kernel = tf.tile(kernel, [1, 1, tf.shape(inputs)[-1], 1])
        return tf.nn.depthwise_conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding="SAME")

    def get_config(self):
        config = super().get_config()
        config.update(kernel_size=self.kernel_size, sigma=self.sigma)
        return config


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class MedianFilterLayer(tf.keras.layers.Layer):
    """True median filter over a ``kernel_size x kernel_size`` window."""

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        kernel_size = check_int("kernel_size", kernel_size, minimum=3)
        if kernel_size % 2 == 0:
            raise ValueError(f"`kernel_size` must be odd, got {kernel_size}.")
        self.kernel_size = kernel_size

    def call(self, inputs):
        if inputs.shape.rank != 4:
            raise ValueError(
                "MedianFilterLayer requires 4-D image inputs (batch, height, width, channels); "
                f"received shape {inputs.shape}."
            )
        k = self.kernel_size
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, k, k, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        shape = tf.shape(inputs)
        reshaped = tf.reshape(patches, tf.stack([shape[0], shape[1], shape[2], k * k, shape[3]]))
        ordered = tf.sort(reshaped, axis=3)
        return ordered[:, :, :, (k * k) // 2, :]

    def get_config(self):
        config = super().get_config()
        config.update(kernel_size=self.kernel_size)
        return config


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class JPEGCompressionLayer(tf.keras.layers.Layer):
    """JPEG encode/decode round-trip with a straight-through gradient."""

    def __init__(self, quality=75, straight_through_gradient=True, value_range=_DEFAULT_RANGE, **kwargs):
        super().__init__(**kwargs)
        self.quality = check_int("quality", quality, minimum=1, maximum=100)
        self.straight_through_gradient = bool(straight_through_gradient)
        self.value_range = tuple(float(v) for v in value_range)

    def _roundtrip(self, image):
        lo, hi = self.value_range
        image = tf.clip_by_value(image, lo, hi)
        data = tf.cast(tf.round((image - lo) / (hi - lo) * 255.0), tf.uint8)
        channels = data.shape[-1]
        fmt = "grayscale" if channels == 1 else "rgb"
        encoded = tf.io.encode_jpeg(data, quality=self.quality, format=fmt)
        decoded = tf.io.decode_jpeg(encoded, channels=channels)
        return tf.cast(decoded, tf.float32) / 255.0 * (hi - lo) + lo

    def call(self, inputs):
        if inputs.shape.rank != 4:
            raise ValueError(
                "JPEGCompressionLayer requires 4-D image inputs (batch, height, width, channels); "
                f"received shape {inputs.shape}."
            )
        channels = inputs.shape[-1]
        if channels not in (1, 3):
            raise ValueError(
                "JPEG compression requires 1 (grayscale) or 3 (RGB) channels, "
                f"got {channels}."
            )
        compressed = tf.map_fn(self._roundtrip, inputs, fn_output_signature=inputs.dtype)
        if self.straight_through_gradient:
            return straight_through(compressed, inputs)
        return tf.stop_gradient(compressed)

    def get_config(self):
        config = super().get_config()
        config.update(
            quality=self.quality,
            straight_through_gradient=self.straight_through_gradient,
            value_range=self.value_range,
        )
        return config


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class PixelDeflectionLayer(tf.keras.layers.Layer):
    """Pixel deflection: random pixels are replaced by random nearby pixels."""

    def __init__(self, deflection_count=100, window_size=10, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.deflection_count = check_int("deflection_count", deflection_count, minimum=0)
        self.window_size = check_int("window_size", window_size, minimum=1)
        self.seed = seed

    def _uniform(self, shape, minval, maxval, tag):
        """Random ints; stateless (and therefore reproducible) when seeded."""
        if self.seed is None:
            return tf.random.uniform(shape, minval, maxval, dtype=tf.int32)
        return tf.random.stateless_uniform(
            shape, seed=(self.seed, tag), minval=minval, maxval=maxval,
            dtype=tf.int32,
        )

    def _deflect(self, image):
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        count = self.deflection_count
        rows = self._uniform([count], 0, height, 0)
        cols = self._uniform([count], 0, width, 1)
        row_shift = self._uniform([count], -self.window_size, self.window_size + 1, 2)
        col_shift = self._uniform([count], -self.window_size, self.window_size + 1, 3)
        source = tf.stack(
            [tf.clip_by_value(rows + row_shift, 0, height - 1),
             tf.clip_by_value(cols + col_shift, 0, width - 1)],
            axis=1,
        )
        destination = tf.stack([rows, cols], axis=1)
        values = tf.gather_nd(image, source)
        return tf.tensor_scatter_nd_update(image, destination, values)

    def call(self, inputs):
        if inputs.shape.rank != 4:
            raise ValueError(
                "PixelDeflectionLayer requires 4-D image inputs (batch, height, width, channels); "
                f"received shape {inputs.shape}."
            )
        return tf.map_fn(self._deflect, inputs, fn_output_signature=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            deflection_count=self.deflection_count,
            window_size=self.window_size,
            seed=self.seed,
        )
        return config


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class TotalVariationLayer(tf.keras.layers.Layer):
    """Gradient-descent total-variation denoising with a straight-through gradient."""

    def __init__(self, iterations=10, regularization_parameter=0.1,
                 straight_through_gradient=True, value_range=_DEFAULT_RANGE, **kwargs):
        super().__init__(**kwargs)
        self.iterations = check_int("iterations", iterations, minimum=1)
        self.regularization_parameter = check_float(
            "regularization_parameter", regularization_parameter, minimum=0.0
        )
        self.straight_through_gradient = bool(straight_through_gradient)
        self.value_range = tuple(float(v) for v in value_range)

    def call(self, inputs):
        if inputs.shape.rank != 4:
            raise ValueError(
                "TotalVariationLayer requires 4-D image inputs (batch, height, width, channels); "
                f"received shape {inputs.shape}."
            )
        lo, hi = self.value_range
        image = tf.stop_gradient(inputs)
        for _ in range(self.iterations):
            with tf.GradientTape() as tape:
                tape.watch(image)
                tv = tf.reduce_sum(tf.image.total_variation(image))
            grad = tape.gradient(tv, image)
            if grad is None:
                break
            image = tf.clip_by_value(image - self.regularization_parameter * grad, lo, hi)
            image = tf.stop_gradient(image)
        if self.straight_through_gradient:
            return straight_through(image, inputs)
        return image

    def get_config(self):
        config = super().get_config()
        config.update(
            iterations=self.iterations,
            regularization_parameter=self.regularization_parameter,
            straight_through_gradient=self.straight_through_gradient,
            value_range=self.value_range,
        )
        return config


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class NoiseLayer(tf.keras.layers.Layer):
    """Additive Gaussian noise (optionally disabled at inference time)."""

    def __init__(self, noise_level=0.1, seed=None, add_noise_at_inference=True,
                 value_range=_DEFAULT_RANGE, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = check_float("noise_level", noise_level, minimum=0.0)
        self.seed = seed
        self.add_noise_at_inference = bool(add_noise_at_inference)
        self.value_range = tuple(float(v) for v in value_range)

    def call(self, inputs, training=None):
        if training is False and not self.add_noise_at_inference:
            return inputs
        if self.seed is None:
            noise = tf.random.normal(
                tf.shape(inputs), mean=0.0, stddev=self.noise_level,
                dtype=inputs.dtype,
            )
        else:
            # Stateless noise keeps a seeded layer reproducible across calls
            # and across a save/load round-trip.
            noise = tf.random.stateless_normal(
                tf.shape(inputs), seed=(self.seed, 0), mean=0.0,
                stddev=self.noise_level, dtype=inputs.dtype,
            )
        lo, hi = self.value_range
        return tf.clip_by_value(inputs + noise, lo, hi)

    def get_config(self):
        config = super().get_config()
        config.update(
            noise_level=self.noise_level,
            seed=self.seed,
            add_noise_at_inference=self.add_noise_at_inference,
            value_range=self.value_range,
        )
        return config


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class InputTransformationLayer(tf.keras.layers.Layer):
    """Applies a user supplied callable or Keras layer to the input."""

    def __init__(self, transformation=None, **kwargs):
        super().__init__(**kwargs)
        if transformation is None or not callable(transformation):
            raise ValueError("`transformation` must be a callable or a Keras layer.")
        self.transformation = transformation

    def call(self, inputs):
        return self.transformation(inputs)

    def get_config(self):
        config = super().get_config()
        if isinstance(self.transformation, tf.keras.layers.Layer):
            # Keras layers serialise properly, so the wrapped model stays
            # saveable.
            config["transformation"] = {
                "kind": "layer",
                "config": tf.keras.utils.serialize_keras_object(self.transformation),
            }
        else:
            config["transformation"] = {
                "kind": "callable",
                "config": getattr(self.transformation, "__name__", None),
            }
        return config

    @classmethod
    def from_config(cls, config):
        payload = config.pop("transformation", None) or {}
        kind = payload.get("kind")
        if kind == "layer":
            transformation = tf.keras.utils.deserialize_keras_object(payload["config"])
        else:
            name = payload.get("config")
            transformation = (
                tf.keras.utils.get_registered_object(name) if name else None
            )
            if transformation is None:
                raise ValueError(
                    f"Could not resolve the transformation {name!r}. Plain "
                    "Python callables cannot be serialised; pass a Keras layer "
                    "(for example `tf.keras.layers.Rescaling(0.5)`) or register "
                    "the function with "
                    "`tf.keras.utils.register_keras_serializable` and pass it "
                    "through `custom_objects` when loading."
                )
        return cls(transformation=transformation, **config)


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class FeatureDenoisingBlock(tf.keras.layers.Layer):
    """Learnable residual smoothing block, meant to be inserted into a network.

    This is the practical, lighter-weight variant of the Feature Denoising
    block (Xie et al., 2019): a learned local average is added to the feature
    map through a zero-initialised scalar, so the block starts as an identity
    and can be trained to denoise intermediate features.
    """

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = check_int("kernel_size", kernel_size, minimum=1)

    def build(self, input_shape):
        # A fixed box filter: every tap is 1 / (k * k), so the block starts as
        # a local average and the learnable scale decides how much to subtract.
        self.average = tf.keras.layers.DepthwiseConv2D(
            self.kernel_size, padding="same", use_bias=False, trainable=False,
            depthwise_initializer=tf.keras.initializers.Constant(
                1.0 / (self.kernel_size * self.kernel_size)
            ),
        )
        self.scale = self.add_weight(
            name="scale", shape=(), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.scale * self.average(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(kernel_size=self.kernel_size)
        return config


# ---------------------------------------------------------------------------
# Training-time defenses
# ---------------------------------------------------------------------------


def _infer_loss(y):
    """Pick a sensible Keras loss string for the given labels."""
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.floating):
        return "sparse_categorical_crossentropy"
    if y.ndim == 1:
        return "sparse_categorical_crossentropy"
    if y.ndim == 2 and y.shape[-1] == 1:
        return "binary_crossentropy"
    return "categorical_crossentropy"


def _clone_for_training(model):
    try:
        clone = tf.keras.models.clone_model(model)
    except Exception as exc:  # pragma: no cover - depends on the user model
        raise TypeError(
            "Adversarial training needs a Functional or Sequential model so it "
            "can be cloned. Subclassed models are not supported."
        ) from exc
    clone.set_weights(model.get_weights())
    return clone


def adversarial_training(model, x, y, epsilon=0.01, epochs=10, batch_size=32,
                         attack=None, optimizer=None, loss=None, verbose=1,
                         shuffle=True):
    """
    Adversarial Training defense.

    Augments the training set with adversarial examples generated in batches and
    fine-tunes a clone of ``model``.  The caller's optimiser, loss and metrics
    are preserved when the model was already compiled.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        x (numpy.ndarray): The input training examples.
        y (numpy.ndarray): The true labels of the training examples.
        epsilon (float): The attack strength used to generate the examples.
        epochs (int): Training epochs (default: 10).
        batch_size (int): Batch size for both attack generation and training.
        attack (callable): ``fn(model, x, y, epsilon=...)`` returning adversarial
            examples. Defaults to :func:`deepdefend.attacks.fgsm`.
        optimizer: Optional Keras optimiser (defaults to the model's optimiser).
        loss: Optional Keras loss (defaults to the model's loss, then inferred
            from the label encoding).
        verbose (int): Keras ``fit`` verbosity (default: 1).
        shuffle (bool): Shuffle the augmented training set.

    Returns:
        defended_model (tensorflow.keras.Model): The adversarially trained model.
    """
    epsilon = check_float("epsilon", epsilon, minimum=0.0)
    epochs = check_int("epochs", epochs, minimum=1)
    batch_size = check_int("batch_size", batch_size, minimum=1)

    x = np.asarray(x)
    y = np.asarray(y)
    if attack is None:
        attack = attacks.fgsm
    if not callable(attack):
        raise TypeError("`attack` must be callable.")

    defended_model = _clone_for_training(model)

    adversarial_batches = []
    for start in range(0, len(x), batch_size):
        chunk_x = x[start:start + batch_size]
        chunk_y = y[start:start + batch_size]
        adversarial_batches.append(attack(model, chunk_x, chunk_y, epsilon=epsilon))
    x_adversarial = np.concatenate(adversarial_batches, axis=0)

    x_train = np.concatenate([x, x_adversarial], axis=0)
    y_train = np.concatenate([y, np.copy(y)], axis=0)

    if optimizer is None:
        optimizer = getattr(model, "optimizer", None)
    if loss is None:
        loss = getattr(model, "loss", None)
    if loss is None:
        loss = _infer_loss(y)
    if optimizer is None:
        optimizer = "adam"

    defended_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    defended_model.fit(
        x_train, y_train, epochs=epochs, batch_size=batch_size,
        verbose=verbose, shuffle=shuffle,
    )
    return defended_model


def _temperature_softmax(outputs, temperature, from_logits):
    """Temperature-scaled softmax that works for logits and probabilities."""
    if from_logits:
        return tf.nn.softmax(outputs / temperature)
    probabilities = tf.clip_by_value(outputs, 1e-7, 1.0)
    return tf.nn.softmax(tf.math.log(probabilities) / temperature)


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class DistillationModel(tf.keras.Model):
    """Knowledge distillation from a (fixed) teacher into a student model."""

    def __init__(self, student=None, teacher=None, temperature=2.0,
                 from_logits=None, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.from_logits = infer_from_logits(student) if from_logits is None else bool(from_logits)

    def call(self, inputs, training=None):
        return self.student(inputs, training=training)

    def train_step(self, data):
        x, y = data[0], data[1]
        teacher_outputs = self.teacher(x, training=False)
        soft_targets = _temperature_softmax(teacher_outputs, self.temperature, self.from_logits)

        with tf.GradientTape() as tape:
            student_outputs = self.student(x, training=True)
            soft_student = _temperature_softmax(student_outputs, self.temperature, self.from_logits)
            distillation_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(soft_targets, soft_student)
            ) * (self.temperature ** 2)
            task_loss = self.compiled_loss(y, student_outputs)
            loss = task_loss + distillation_loss

        trainable = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable)
        pairs = [(g, v) for g, v in zip(gradients, trainable) if g is not None]
        self.optimizer.apply_gradients(pairs)
        self.compiled_metrics.update_state(y, student_outputs)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update(
            student=tf.keras.utils.serialize_keras_object(self.student),
            teacher=tf.keras.utils.serialize_keras_object(self.teacher),
            temperature=self.temperature,
            from_logits=self.from_logits,
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        for key in ("student", "teacher"):
            config[key] = tf.keras.utils.deserialize_keras_object(config[key])
        return cls(**config)


def defensive_distillation(model, teacher_model, temperature=2):
    """
    Defensive Distillation defense.

    Trains the student to match the temperature-softened predictions of a
    teacher, in addition to the ordinary task loss.  The distillation term is
    scaled by ``temperature ** 2`` and the teacher is evaluated in inference
    mode, so its batch-norm statistics and dropout do not corrupt the targets.

    Parameters:
        model (tensorflow.keras.Model): The student model to defend.
        teacher_model (tensorflow.keras.Model): The teacher model.
        temperature (float): Distillation temperature (default: 2).

    Returns:
        defended_model (tensorflow.keras.Model): The distilled student model.
    """
    temperature = check_float("temperature", temperature, minimum=1e-6)
    return DistillationModel(student=model, teacher=teacher_model, temperature=temperature)


@tf.keras.utils.register_keras_serializable(package="deepdefend")
class AdversarialLogitPairingModel(tf.keras.Model):
    """Adversarial Logit Pairing: clean and adversarial logits must agree."""

    def __init__(self, base_model=None, paired_model=None, epsilon=0.01,
                 num_steps=1, alpha=None, from_logits=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.paired_model = paired_model if paired_model is not None else base_model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha
        self.from_logits = infer_from_logits(base_model) if from_logits is None else bool(from_logits)

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def _adversarial(self, x, y):
        return attacks._iterative_gradient_attack(
            self.base_model, x, y, self.epsilon, self.alpha, self.num_steps,
            random_start=False, seed=None, targeted=False,
            from_logits=self.from_logits, clip_min=0.0, clip_max=1.0,
        )

    def train_step(self, data):
        x, y = data[0], data[1]
        adversarial = self._adversarial(x, y)

        with tf.GradientTape() as tape:
            clean_logits = self.base_model(x, training=True)
            adversarial_logits = self.paired_model(adversarial, training=True)
            classification_loss = self.compiled_loss(y, clean_logits)
            pairing_loss = tf.reduce_mean(tf.square(clean_logits - adversarial_logits))
            loss = classification_loss + pairing_loss

        trainable = self.base_model.trainable_variables
        gradients = tape.gradient(loss, trainable)
        pairs = [(g, v) for g, v in zip(gradients, trainable) if g is not None]
        self.optimizer.apply_gradients(pairs)
        self.compiled_metrics.update_state(y, clean_logits)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update(
            base_model=tf.keras.utils.serialize_keras_object(self.base_model),
            paired_model=tf.keras.utils.serialize_keras_object(self.paired_model),
            epsilon=self.epsilon,
            num_steps=self.num_steps,
            alpha=self.alpha,
            from_logits=self.from_logits,
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        for key in ("base_model", "paired_model"):
            if config.get(key) is not None:
                config[key] = tf.keras.utils.deserialize_keras_object(config[key])
        return cls(**config)


def adversarial_logit_pairing(model, paired_model=None, epsilon=0.01, num_steps=1,
                              alpha=None):
    """
    Adversarial Logit Pairing (ALP) defense.

    In each training step an adversarial example is generated and the clean and
    adversarial logits of the model are pulled together.  (The previous
    implementation paired two models on *clean* inputs, which is not ALP.)

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        paired_model (tensorflow.keras.Model): Optional second model; defaults
            to ``model``.
        epsilon (float): Strength of the inner attack (default: 0.01).
        num_steps (int): Iterations of the inner attack (default: 1).
        alpha (float): Step size of the inner attack.

    Returns:
        defended_model (tensorflow.keras.Model): The wrapped model.
    """
    epsilon = check_float("epsilon", epsilon, minimum=0.0)
    num_steps = check_int("num_steps", num_steps, minimum=1)
    alpha = check_float("alpha", alpha, minimum=0.0, allow_none=True)
    return AdversarialLogitPairingModel(
        base_model=model, paired_model=paired_model, epsilon=epsilon,
        num_steps=num_steps, alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Inference-time preprocessing defenses
# ---------------------------------------------------------------------------


def feature_squeezing(model, bit_depth=4):
    """
    Feature Squeezing defense.

    Reduces the colour depth of the input to ``bit_depth`` bits, removing small
    adversarial perturbations.  A straight-through gradient is used so gradient
    based attacks against the defended model still work.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        bit_depth (int): Number of bits per feature, 1-8 (default: 4).

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    return wrap_preprocessing(model, SqueezeLayer(bit_depth=bit_depth, name="feature_squeezing"))


def input_transformation(model, transformation_function=None):
    """
    Input Transformation defense.

    Applies ``transformation_function`` to the input before the model.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        transformation_function (callable): The transformation to apply. It is
            required — the previous default silently produced a model with no
            defense at all.

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    if transformation_function is None:
        raise ValueError(
            "`transformation_function` is required. Pass a callable such as "
            "`lambda x: tf.image.resize(x, (224, 224))`."
        )
    return wrap_preprocessing(
        model, InputTransformationLayer(transformation_function, name="input_transformation")
    )


def randomized_smoothing(model, noise_level=0.1, seed=None, add_noise_at_inference=True):
    """
    Randomized Smoothing (inference-time noise).

    Adds Gaussian noise to the input.  This alone does **not** make a model
    certifiably robust: the base model should be trained with the same noise
    (see :func:`adversarial_training` with a noise-augmented dataset) and
    predictions should be averaged over many samples with :func:`smooth_predict`.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        noise_level (float): Standard deviation of the noise (default: 0.1).
        seed (int): Optional seed.
        add_noise_at_inference (bool): Whether to add noise outside training
            (default: True). Set to False to use the noise only as training
            augmentation.

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    layer = NoiseLayer(
        noise_level=noise_level, seed=seed,
        add_noise_at_inference=add_noise_at_inference, name="randomized_smoothing",
    )
    return wrap_preprocessing(model, layer)


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    lower = 0.02425
    if p < lower:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - lower:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def smooth_predict(model, x, n_samples=100, noise_level=None, seed=None):
    """
    Monte-Carlo prediction for a noise-augmented model.

    Averages ``n_samples`` stochastic forward passes and estimates the certified
    L2 radius ``sigma * Phi^-1(p)`` of the majority prediction.

    Parameters:
        model (tf.keras.Model): A model wrapped by :func:`randomized_smoothing`.
        x (numpy.ndarray): Input batch.
        n_samples (int): Number of Monte-Carlo samples (default: 100).
        noise_level (float): Noise standard deviation. Read from the model's
            first ``NoiseLayer`` when omitted.
        seed (int): Optional seed.

    Returns:
        dict: ``{"predictions", "probabilities", "certified_radius"}``.
    """
    n_samples = check_int("n_samples", n_samples, minimum=1)
    x = np.asarray(x, dtype="float32")

    if noise_level is None:
        for layer in getattr(model, "layers", []):
            if isinstance(layer, NoiseLayer):
                noise_level = layer.noise_level
                break
    if noise_level is None:
        raise ValueError(
            "Could not find a NoiseLayer in the model; pass `noise_level` "
            "explicitly."
        )
    noise_level = check_float("noise_level", noise_level, minimum=0.0)

    rng = np.random.default_rng(seed)
    votes = np.zeros((n_samples,) + (x.shape[0],), dtype="int32")
    probabilities = np.zeros((n_samples, x.shape[0], int(np.shape(model(x[:1]))[-1])), dtype="float32")

    for i in range(n_samples):
        noisy = x + rng.normal(0.0, noise_level, size=x.shape).astype("float32")
        outputs = np.asarray(model.predict(noisy, verbose=0))
        probabilities[i] = outputs
        votes[i] = np.argmax(outputs, axis=-1)

    counts = np.zeros((x.shape[0], probabilities.shape[-1]), dtype="float32")
    for i in range(x.shape[0]):
        counts[i] = np.bincount(votes[:, i], minlength=counts.shape[1])
    predictions = np.argmax(counts, axis=-1)
    p_hat = counts.max(axis=-1) / float(n_samples)
    radius = np.where(p_hat > 0.5, noise_level * np.array([_norm_ppf(p) for p in p_hat]), 0.0)

    return {
        "predictions": predictions,
        "probabilities": probabilities.mean(axis=0),
        "certified_radius": radius,
    }


def feature_denoising(model, method="median", kernel_size=3, sigma=1.0):
    """
    Feature Denoising defense (input-space variant).

    Applies a denoising filter to the input.  For real feature-level denoising,
    insert :class:`FeatureDenoisingBlock` inside your network and train it.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        method (str): ``"median"`` (default) or ``"gaussian"``.
        kernel_size (int): Filter size (odd, default: 3).
        sigma (float): Standard deviation for the Gaussian filter.

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    return wrap_preprocessing(model, _make_smoothing_layer(method, kernel_size, sigma, "feature_denoising"))


def _make_smoothing_layer(method, kernel_size, sigma, name):
    method = str(method).lower()
    if method == "median":
        return MedianFilterLayer(kernel_size=kernel_size, name=name)
    if method == "gaussian":
        return GaussianBlurLayer(kernel_size=kernel_size, sigma=sigma, name=name)
    raise ValueError(f"`method` must be 'median' or 'gaussian', got {method!r}.")


def thermometer_encoding(model, num_bins=10):
    """
    Thermometer-encoding style discretisation.

    The input is quantised into ``num_bins`` levels.  Note that canonical
    thermometer encoding *expands* each feature into ``num_bins`` binary
    features, which changes the input shape and requires retraining the first
    layer; this shape-preserving variant is the quantisation equivalent and is
    documented as such instead of pretending otherwise.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        num_bins (int): Number of quantisation levels (default: 10).

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    return wrap_preprocessing(model, QuantizationLayer(num_bins=num_bins, name="thermometer_encoding"))


def spatial_smoothing(model, kernel_size=3, method="median", sigma=1.0):
    """
    Spatial Smoothing defense.

    Applies a median (default) or Gaussian smoothing filter to the input.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        kernel_size (int): Filter size (odd, default: 3).
        method (str): ``"median"`` (default) or ``"gaussian"``.
        sigma (float): Standard deviation for the Gaussian filter.

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    return wrap_preprocessing(model, _make_smoothing_layer(method, kernel_size, sigma, "spatial_smoothing"))


def median_smoothing(model, kernel_size=3):
    """
    Median Smoothing defense.

    Applies a true median filter (this used to call ``avg_pool2d``).  Kept as a
    named alias of :func:`spatial_smoothing` with ``method="median"``.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        kernel_size (int): Filter size (odd, default: 3).

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    return wrap_preprocessing(model, MedianFilterLayer(kernel_size=kernel_size, name="median_smoothing"))


def gaussian_blur(model, kernel_size=3, sigma=1.0):
    """
    Gaussian Blur defense.

    Convolves the input with a separable Gaussian kernel of standard deviation
    ``sigma`` (the parameter is now actually used).

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        kernel_size (int): Kernel size (odd, default: 3).
        sigma (float): Standard deviation of the kernel (default: 1.0).

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    return wrap_preprocessing(
        model, GaussianBlurLayer(kernel_size=kernel_size, sigma=sigma, name="gaussian_blur")
    )


def jpeg_compression(model, quality=75):
    """
    JPEG Compression defense.

    Compresses and decompresses the input with JPEG.  A straight-through
    gradient is used, so gradient based attacks against the defended model still
    produce a signal instead of ``None``.

    Requires static spatial dimensions and 1 or 3 channels.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        quality (int): JPEG quality, 1-100 (default: 75).

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    return wrap_preprocessing(
        model, JPEGCompressionLayer(quality=quality, name="jpeg_compression")
    )


def pixel_deflection(model, deflection_count=100, window_size=10, seed=None):
    """
    Pixel Deflection defense.

    Randomly replaces ``deflection_count`` pixels by random neighbours within
    ``window_size``.  The previous implementation raised a ``ValueError`` on
    every call.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        deflection_count (int): Number of pixels to deflect (default: 100).
        window_size (int): Maximum displacement in pixels (default: 10).
        seed (int): Optional seed.

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    return wrap_preprocessing(
        model,
        PixelDeflectionLayer(
            deflection_count=deflection_count, window_size=window_size,
            seed=seed, name="pixel_deflection",
        ),
    )


def total_variation_minimization(model, iterations=10, regularization_parameter=0.1):
    """
    Total Variation Minimization defense.

    Denoises the input with ``iterations`` gradient steps on the total variation
    norm.  A straight-through gradient is used so attacks against the defended
    model do not silently produce a zero perturbation.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        iterations (int): Number of denoising steps (default: 10).
        regularization_parameter (float): Step size of the denoiser (default: 0.1).

    Returns:
        defended_model (tensorflow.keras.Model): The defended model.
    """
    return wrap_preprocessing(
        model,
        TotalVariationLayer(
            iterations=iterations,
            regularization_parameter=regularization_parameter,
            name="total_variation_minimization",
        ),
    )


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def word_masking(text, mask_token="[MASK]", mask_prob=0.1, seed=None):
    """
    Word masking for text.  Whitespace and line breaks are preserved.

    Parameters:
        text (str): The input text.
        mask_token (str): The token used for masking (default: "[MASK]").
        mask_prob (float): Probability of masking a word (default: 0.1).
        seed (int): Optional seed for reproducibility.

    Returns:
        defended_text (str): The text with randomly masked words.
    """
    if not isinstance(text, str):
        raise TypeError(f"`text` must be a string, got {type(text).__name__}.")
    mask_prob = check_float("mask_prob", mask_prob, minimum=0.0, maximum=1.0)
    rng = np.random.default_rng(seed)

    tokens = tokenize_text(text)
    for i, (_token, is_word) in enumerate(tokens):
        if not is_word:
            continue
        if rng.random() < mask_prob:
            tokens[i] = (mask_token, False)
    return "".join(token for token, _ in tokens)
