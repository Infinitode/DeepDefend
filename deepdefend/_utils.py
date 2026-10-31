"""Internal helpers shared by :mod:`deepdefend.attacks` and :mod:`deepdefend.defenses`.

This module is private (leading underscore) and its API may change between
releases.  It centralises the things that the public functions used to
re-implement (badly) in every function body:

* argument validation, so a security tool never silently emits ``NaN``;
* label handling for one-hot, sparse-integer and binary targets;
* ``from_logits`` detection, so attacks work on models that do not end in a
  ``softmax``/``sigmoid``;
* projection onto the ``[clip_min, clip_max]`` box and the ``epsilon`` ball;
* straight-through gradients for quantisation / compression based defenses.
"""

from __future__ import annotations

import math
import numbers
import re

import numpy as np
import tensorflow as tf

__all__ = [
    "AttackContext",
    "as_numpy",
    "build_layer_model",
    "call_model",
    "check_float",
    "check_int",
    "infer_from_logits",
    "infer_num_classes",
    "normalise_labels",
    "prepare_attack_context",
    "project",
    "resolve_clip_range",
    "straight_through",
    "tokenize_text",
    "to_tensor",
    "wrap_preprocessing",
]

# A "word" for the text helpers: letters, digits and inner apostrophes.
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_TOKEN_RE = re.compile(r"([A-Za-z0-9']+)")

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def check_float(name, value, minimum=None, maximum=None, allow_none=False):
    """Validate a real-valued hyper-parameter and return it as ``float``."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"`{name}` must be a number, got None.")
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"`{name}` must be a real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"`{name}` must be finite, got {value}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"`{name}` must be <= {maximum}, got {value}.")
    return value


def check_int(name, value, minimum=None, maximum=None):
    """Validate an integer hyper-parameter and return it as ``int``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"`{name}` must be an integer, got {type(value).__name__}.")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"`{name}` must be <= {maximum}, got {value}.")
    return value


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def tokenize_text(text):
    """Split ``text`` into ``(token, is_word)`` pairs.

    Whitespace runs, newlines and punctuation are returned as their own tokens,
    so a caller can rebuild the text verbatim with ``"".join(tokens)``.  The
    naive ``text.split()`` / ``" ".join(...)`` round-trip used previously
    collapsed double spaces and removed line breaks.
    """
    parts = _TOKEN_RE.split(text)
    return [(part, bool(_WORD_RE.fullmatch(part))) for part in parts if part]


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def to_tensor(x, dtype=tf.float32):
    """Convert ``x`` to a float tensor without changing its values."""
    tensor = tf.convert_to_tensor(x)
    if not tensor.dtype.is_floating:
        tensor = tf.cast(tensor, dtype)
    elif tensor.dtype != dtype:
        tensor = tf.cast(tensor, dtype)
    return tensor


def as_numpy(tensor):
    """Return a NumPy view/copy of ``tensor`` (keeps the original dtype)."""
    return np.asarray(tensor.numpy() if tf.is_tensor(tensor) else tensor)


def call_model(model, x):
    """Call ``model`` in inference mode, tolerating callables without ``training``."""
    try:
        return model(x, training=False)
    except TypeError:
        return model(x)


def straight_through(value, reference):
    """Forward ``value``, but back-propagate the identity gradient of ``reference``.

    Used by non-differentiable defenses (quantisation, JPEG, denoisers) so that
    gradient based attacks against a *defended* model still produce a useful
    signal instead of ``None``.
    """
    return reference + tf.stop_gradient(value - reference)


# ---------------------------------------------------------------------------
# Output / label introspection
# ---------------------------------------------------------------------------


def _model_output_shape(model):
    shape = getattr(model, "output_shape", None)
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        if len(shape) != 1:
            return None  # multi-output models are not supported
        shape = shape[0]
    if shape is None or not isinstance(shape, (list, tuple)):
        return None
    return tuple(shape)


def infer_num_classes(model, y=None):
    """Best-effort detection of the number of output units."""
    shape = _model_output_shape(model)
    if shape is not None:
        last = shape[-1]
        if isinstance(last, numbers.Integral) and last > 0:
            return int(last)
    if y is not None:
        # ``y`` may be a symbolic tensor inside a tf.function; only inspect it
        # when it can be converted without executing the graph.
        try:
            array = np.asarray(y)
        except (TypeError, ValueError, tf.errors.OperatorNotAllowedInGraphError):
            array = None
        if array is not None:
            if array.ndim >= 2 and array.shape[-1] > 1:
                return int(array.shape[-1])
            if array.ndim >= 2 and array.shape[-1] == 1:
                return 1
    raise ValueError(
        "Could not infer the number of classes; pass a model with a known "
        "`output_shape` or one-hot encoded labels."
    )


def infer_from_logits(model):
    """Guess whether ``model`` returns logits (``True``) or probabilities.

    Returns ``True`` for a final ``linear``/``relu`` layer and ``False`` for a
    final ``softmax``/``sigmoid``.  Unknown activation functions default to
    ``False`` (probabilities), which matches the historical behaviour of the
    package and is the safer assumption for the cross-entropy losses used here.
    """
    layers = getattr(model, "layers", None)
    if not layers:
        return False
    last = layers[-1]
    activation = getattr(last, "activation", None)
    name = getattr(activation, "__name__", None)
    if name in ("softmax", "sigmoid"):
        return False
    if name == "linear":
        return True
    # A defense wrapper ends with the wrapped model itself: look inside it.
    if getattr(last, "layers", None):
        return infer_from_logits(last)
    return False


def normalise_labels(y, num_classes):
    """Convert labels to ``(float32 labels, string mode)``.

    ``mode`` is ``"binary"`` (shape ``(N, 1)``) or ``"categorical"``
    (one-hot, shape ``(N, num_classes)``).  Accepts one-hot float arrays,
    sparse integer arrays of shape ``(N,)`` or ``(N, 1)``, and float arrays of
    integral class indices.
    """
    tensor = tf.convert_to_tensor(y)
    if tensor.dtype.is_integer:
        if num_classes == 1:
            return tf.reshape(tf.cast(tensor, tf.float32), (-1, 1)), "binary"
        indices = tf.reshape(tf.cast(tensor, tf.int32), (-1,))
        return tf.one_hot(indices, num_classes, dtype=tf.float32), "categorical"

    tensor = tf.cast(tensor, tf.float32)
    rank = tensor.shape.rank
    if num_classes == 1:
        return tf.reshape(tensor, (-1, 1)), "binary"
    if rank == 1:
        # Float-encoded class indices, e.g. np.array([0., 2., 1.]).
        rounded = tf.round(tensor)
        if tf.executing_eagerly() and not bool(tf.reduce_all(tf.abs(tensor - rounded) < 1e-6)):
            raise ValueError(
                "1-D float labels must be integral class indices, or pass "
                "one-hot labels of shape (N, num_classes)."
            )
        return tf.one_hot(tf.cast(rounded, tf.int32), num_classes, dtype=tf.float32), "categorical"
    if rank >= 2 and tensor.shape[-1] == 1:
        return tf.reshape(tensor, (-1, 1)), "binary"
    return tensor, "categorical"


# ---------------------------------------------------------------------------
# Attack context
# ---------------------------------------------------------------------------


class AttackContext:
    """Resolved attack inputs: tensors, labels, losses and clipping bounds."""

    __slots__ = ("x", "labels", "y_idx", "num_classes", "mode", "from_logits", "clip_min", "clip_max")

    def __init__(self, x, labels, y_idx, num_classes, mode, from_logits, clip_min, clip_max):
        self.x = x
        self.labels = labels
        self.y_idx = y_idx
        self.num_classes = num_classes
        self.mode = mode
        self.from_logits = from_logits
        self.clip_min = clip_min
        self.clip_max = clip_max

    # -- losses ---------------------------------------------------------
    def loss_vec(self, outputs):
        """Per-example cross-entropy (shape ``(batch,)``)."""
        if self.mode == "binary":
            return tf.keras.losses.binary_crossentropy(
                self.labels, outputs, from_logits=self.from_logits
            )
        return tf.keras.losses.categorical_crossentropy(
            self.labels, outputs, from_logits=self.from_logits
        )

    def loss(self, outputs):
        """Mean cross-entropy over the batch (scalar)."""
        return tf.reduce_mean(self.loss_vec(outputs))

    # -- projection -----------------------------------------------------
    def project(self, adv, epsilon=None):
        return project(adv, self.x, epsilon=epsilon, clip_min=self.clip_min, clip_max=self.clip_max)


def prepare_attack_context(model, x, y, from_logits=None, clip_min=None, clip_max=None):
    """Validate inputs and build an :class:`AttackContext`."""
    x_t = to_tensor(x)
    if x_t.shape.rank is None or x_t.shape.rank < 1:
        raise ValueError("`x` must be a tensor with at least one dimension (batch included).")

    num_classes = infer_num_classes(model, y)
    labels, mode = normalise_labels(y, num_classes)

    if labels.shape[0] is not None and x_t.shape[0] is not None and labels.shape[0] != x_t.shape[0]:
        raise ValueError(
            f"`x` and `y` have different batch sizes: {int(x_t.shape[0])} vs "
            f"{int(labels.shape[0])}."
        )

    if from_logits is None:
        from_logits = infer_from_logits(model)
    from_logits = bool(from_logits)

    if mode == "binary":
        y_idx = tf.cast(tf.reshape(labels, (-1,)) >= 0.5, tf.int32)
    else:
        y_idx = tf.cast(tf.argmax(labels, axis=-1), tf.int32)

    lo, hi = resolve_clip_range(x_t, clip_min, clip_max)

    return AttackContext(
        x=x_t,
        labels=labels,
        y_idx=y_idx,
        num_classes=num_classes,
        mode=mode,
        from_logits=from_logits,
        clip_min=lo,
        clip_max=hi,
    )


# ---------------------------------------------------------------------------
# Clipping / projection
# ---------------------------------------------------------------------------


def resolve_clip_range(x, clip_min=None, clip_max=None):
    """Decide the valid data range.

    * both given  -> use them;
    * both ``None`` and the data already lies in ``[0, 1]`` -> ``(0.0, 1.0)``;
    * both ``None`` and the data is outside ``[0, 1]`` (e.g. ImageNet-style
      normalised inputs) -> no clipping, so the input is not silently
      destroyed.  Pass an explicit range if you want one.
    """
    if (clip_min is None) != (clip_max is None):
        raise ValueError("`clip_min` and `clip_max` must be provided together.")
    if clip_min is None:
        if not tf.executing_eagerly():
            # Inside a tf.function the data range cannot be inspected; assume
            # image data in [0, 1] rather than guessing from a symbolic tensor.
            return 0.0, 1.0
        lo = float(tf.reduce_min(x))
        hi = float(tf.reduce_max(x))
        if lo >= 0.0 and hi <= 1.0:
            return 0.0, 1.0
        return None, None
    lo = check_float("clip_min", clip_min)
    hi = check_float("clip_max", clip_max)
    if hi <= lo:
        raise ValueError(f"`clip_max` ({hi}) must be greater than `clip_min` ({lo}).")
    return lo, hi


def project(adv, x, epsilon=None, clip_min=None, clip_max=None):
    """Project ``adv`` onto the ``epsilon`` L-infinity ball around ``x`` and the data range."""
    if epsilon is not None:
        adv = tf.clip_by_value(adv, x - epsilon, x + epsilon)
    if clip_min is not None and clip_max is not None:
        adv = tf.clip_by_value(adv, clip_min, clip_max)
    return adv


# ---------------------------------------------------------------------------
# Model wrapping used by the defense module
# ---------------------------------------------------------------------------


def _single_input_spec(model):
    """Return ``(shape, dtype)`` for a single-input Keras model."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        raise TypeError(
            "DeepDefend defenses only support single-input models; this model "
            f"has {len(shape)} inputs."
        )
    if shape is None:
        raise TypeError(
            "DeepDefend defenses require a model with a known `input_shape` "
            "(a Functional or Sequential model). Subclassed models must be "
            "built from an explicit `tf.keras.Input` before they can be wrapped."
        )
    dtype = getattr(model, "input_dtype", None) or "float32"
    return tuple(shape[1:]), dtype


def build_layer_model(model, layer, name=None):
    """Wrap ``model`` in a new Functional model that applies ``layer`` first."""
    shape, dtype = _single_input_spec(model)
    inputs = tf.keras.Input(shape=shape, dtype=dtype, name="deepdefend_input")
    outputs = call_model(model, layer(inputs))
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def wrap_preprocessing(model, layer):
    """Alias of :func:`build_layer_model` kept for readability in call sites."""
    layer_name = getattr(layer, "name", None)
    return build_layer_model(model, layer, name=f"{model.name}_{layer_name}" if layer_name else None)
