"""Tests for :mod:`deepdefend.attacks`."""

import numpy as np
import pytest

import deepdefend
from deepdefend import attacks

IMAGE_ATTACKS = ["fgsm", "pgd", "bim", "mim", "cw", "deepfool", "jsma", "spsa", "ead"]

# Cheap keyword overrides so the suite stays fast.
FAST_KWARGS = {
    "cw": {"num_steps": 5},
    "deepfool": {"num_steps": 3},
    "jsma": {"num_steps": 3},
    "spsa": {"num_steps": 3, "spsa_samples": 4},
    "ead": {"num_steps": 5},
    "pgd": {"num_steps": 3},
    "bim": {"num_steps": 3},
    "mim": {"num_steps": 3},
}


def _one_hot(y, n=3):
    return np.eye(n)[y].astype(np.float32)


@pytest.mark.parametrize("name", IMAGE_ATTACKS)
def test_attack_output_shape_and_range(name, trained_model, batch):
    x, y = batch
    attack = getattr(attacks, name)
    adversarial = attack(trained_model, x, _one_hot(y), **FAST_KWARGS.get(name, {}))

    assert isinstance(adversarial, np.ndarray)
    assert adversarial.shape == x.shape
    assert adversarial.dtype == np.float32
    assert np.isfinite(adversarial).all()
    # Inputs live in [0, 1], so the output must too (this used to fail for
    # fgsm and deepfool, which returned values such as -0.9 and 2.08).
    assert adversarial.min() >= 0.0
    assert adversarial.max() <= 1.0


@pytest.mark.parametrize("name", IMAGE_ATTACKS)
def test_attack_produces_a_perturbation(name, trained_model, batch):
    x, y = batch
    attack = getattr(attacks, name)
    adversarial = attack(trained_model, x, _one_hot(y), **FAST_KWARGS.get(name, {}))
    # `ead` used to be a complete no-op with its default parameters.
    assert np.abs(adversarial - x).max() > 0.0


@pytest.mark.parametrize("name", IMAGE_ATTACKS)
def test_attack_accepts_sparse_labels(name, trained_model, batch):
    x, y = batch
    attack = getattr(attacks, name)
    adversarial = attack(trained_model, x, y, **FAST_KWARGS.get(name, {}))
    assert adversarial.shape == x.shape


@pytest.mark.parametrize("name", IMAGE_ATTACKS)
def test_attack_accepts_binary_labels(name, batch):
    tf = pytest.importorskip("tensorflow")
    x, _ = batch
    binary_model = tf.keras.Sequential(
        [
            tf.keras.Input((8, 8, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    labels = (x[:, 0, 0, 0] > 0.5).astype("float32")
    adversarial = getattr(attacks, name)(binary_model, x, labels, **FAST_KWARGS.get(name, {}))
    assert adversarial.shape == x.shape


@pytest.mark.parametrize("name", ["fgsm", "pgd", "bim", "mim", "ead"])
def test_epsilon_bound_is_respected(name, trained_model, batch):
    x, y = batch
    epsilon = 0.05
    adversarial = getattr(attacks, name)(
        trained_model, x, _one_hot(y), epsilon=epsilon, **FAST_KWARGS.get(name, {})
    )
    assert np.abs(adversarial - x).max() <= epsilon + 1e-5


def test_pgd_random_start_in_epsilon_ball(trained_model, batch):
    x, y = batch
    epsilon = 0.05
    adversarial = attacks.pgd(
        trained_model, x, _one_hot(y), epsilon=epsilon, num_steps=2, seed=0
    )
    assert np.abs(adversarial - x).max() <= epsilon + 1e-5


def test_pgd_and_bim_differ(trained_model, batch):
    x, y = batch
    y = _one_hot(y)
    pgd = attacks.pgd(trained_model, x, y, epsilon=0.05, num_steps=3, seed=7)
    bim = attacks.bim(trained_model, x, y, epsilon=0.05, num_steps=3)
    # PGD starts from a random point, BIM does not.
    assert not np.allclose(pgd, bim)


def test_cw_parameters_are_effective(trained_model, batch):
    x, y = batch
    y = _one_hot(y)
    weak = attacks.cw(trained_model, x, y, c=0.01, kappa=0.0, num_steps=5)
    strong = attacks.cw(trained_model, x, y, c=100.0, kappa=5.0, num_steps=5)
    # These used to be bit-identical because the L2 term and kappa were inert.
    assert not np.allclose(weak, strong)


def test_ead_is_not_a_noop_with_defaults(trained_model, batch):
    x, y = batch
    adversarial = attacks.ead(trained_model, x, _one_hot(y))
    assert np.abs(adversarial - x).max() > 0.0


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("fgsm", {}),
        ("pgd", {"num_steps": 10}),
        ("bim", {"num_steps": 10}),
        ("mim", {"num_steps": 10}),
        ("cw", {"num_steps": 100}),
        ("deepfool", {"num_steps": 5}),
    ],
)
def test_attacks_reduce_accuracy(name, kwargs, linear_model, dataset):
    """A sanity check that the attacks actually attack."""
    x, y = dataset
    x, y = x[:16], y[:16]
    base = np.mean(np.argmax(linear_model.predict(x, verbose=0), axis=-1) == y)
    assert base > 0.8

    adversarial = getattr(attacks, name)(linear_model, x, _one_hot(y), epsilon=0.3, **kwargs)
    accuracy = np.mean(
        np.argmax(linear_model.predict(adversarial, verbose=0), axis=-1) == y
    )
    assert accuracy < base, f"{name} did not reduce accuracy"


def test_targeted_attack_moves_to_the_target(linear_model, dataset):
    x, y = dataset
    x, y = x[:8], y[:8]
    targets = (y + 1) % 3
    adversarial = attacks.pgd(
        linear_model, x, _one_hot(targets), epsilon=0.3, alpha=0.05,
        num_steps=10, targeted=True, random_start=False,
    )
    predictions = np.argmax(linear_model.predict(adversarial, verbose=0), axis=-1)
    assert np.mean(predictions == targets) >= 0.5


def test_deepfool_with_epsilon_bound(trained_model, batch):
    x, y = batch
    adversarial = attacks.deepfool(trained_model, x, _one_hot(y), epsilon=0.02, num_steps=3)
    assert np.abs(adversarial - x).max() <= 0.02 + 1e-5


def test_from_logits_is_auto_detected(batch):
    tf = pytest.importorskip("tensorflow")
    from deepdefend._utils import infer_from_logits

    x, y = batch
    softmax_model = tf.keras.Sequential(
        [tf.keras.Input((8, 8, 3)), tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(3, activation="softmax")]
    )
    logits_model = tf.keras.Sequential(
        [tf.keras.Input((8, 8, 3)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(3)]
    )
    assert infer_from_logits(softmax_model) is False
    assert infer_from_logits(logits_model) is True

    # Auto-detection must match an explicit value, and the two assumptions must
    # produce different attacks (the old code always assumed probabilities).
    auto = attacks.fgsm(logits_model, x, _one_hot(y), epsilon=0.05)
    explicit = attacks.fgsm(logits_model, x, _one_hot(y), epsilon=0.05, from_logits=True)
    np.testing.assert_allclose(auto, explicit)

    as_probabilities = attacks.fgsm(
        logits_model, x, _one_hot(y), epsilon=0.05, from_logits=False
    )
    assert not np.allclose(auto, as_probabilities)


def test_explicit_clip_range_for_normalised_inputs(trained_model, batch):
    x, y = batch
    normalised = (x - 0.5) * 2.0
    adversarial = attacks.pgd(
        trained_model, normalised, _one_hot(y), epsilon=0.1, clip_min=-1.0, clip_max=1.0
    )
    assert adversarial.min() >= -1.0 - 1e-6
    assert adversarial.max() <= 1.0 + 1e-6


def test_clip_range_is_not_applied_to_normalised_inputs(trained_model, batch):
    """Without an explicit range the [0, 1] clip must not destroy the input."""
    x, y = batch
    normalised = (x - 0.5) * 4.0  # roughly in [-2, 2]
    adversarial = attacks.fgsm(trained_model, normalised, _one_hot(y), epsilon=0.01)
    assert np.abs(adversarial - normalised).max() <= 0.01 + 1e-6


@pytest.mark.parametrize(
    "call,kwargs",
    [
        ("fgsm", {"epsilon": -1.0}),
        ("pgd", {"num_steps": 0}),
        ("jsma", {"gamma": 0.0}),
        ("spsa", {"spsa_samples": 1}),
        ("cw", {"num_steps": 0}),
        ("ead", {"alpha": 0.0}),
    ],
)
def test_invalid_arguments_raise(call, kwargs, trained_model, batch):
    x, y = batch
    with pytest.raises((ValueError, TypeError)):
        getattr(attacks, call)(trained_model, x, _one_hot(y), **kwargs)


def test_mismatched_batch_sizes_raise(trained_model, batch):
    x, y = batch
    with pytest.raises(ValueError):
        attacks.fgsm(trained_model, x, _one_hot(y)[:2])


# ---------------------------------------------------------------------------
# Text attacks
# ---------------------------------------------------------------------------


def test_word_swap_preserves_whitespace():
    text = "The  movie was\ngreat, really\tgreat."
    swapped = attacks.word_swap(text, {"great": "awful"})
    assert swapped == "The  movie was\nawful, really\tawful."
    assert len(swapped.split("\n")) == 2


def test_word_swap_requires_a_dictionary():
    with pytest.raises(ValueError):
        attacks.word_swap("nothing to see")


def test_char_swap_preserves_layout():
    text = "line one\nline two"
    swapped = attacks.char_swap(text, swap_prob=1.0, seed=0)
    assert swapped.count("\n") == text.count("\n")
    assert swapped.count(" ") == text.count(" ")
    assert swapped != text


def test_char_swap_is_reproducible():
    text = "deterministic results please"
    first = attacks.char_swap(text, swap_prob=1.0, seed=42)
    second = attacks.char_swap(text, swap_prob=1.0, seed=42)
    assert first == second
    assert attacks.char_swap(text, swap_prob=1.0, seed=7) != first or first == text


def test_char_swap_validates_probability():
    with pytest.raises(ValueError):
        attacks.char_swap("text", swap_prob=1.5)


def test_attacks_are_exported_from_the_package():
    for name in IMAGE_ATTACKS + ["word_swap", "char_swap"]:
        assert getattr(deepdefend, name) is getattr(attacks, name)
