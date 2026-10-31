"""Tests for the public package surface."""

import subprocess
import sys

import pytest

import deepdefend

EXPECTED_ATTACKS = [
    "fgsm", "pgd", "bim", "mim", "cw", "deepfool", "jsma", "spsa", "ead",
    "word_swap", "char_swap",
]
EXPECTED_DEFENSES = [
    "adversarial_training", "feature_squeezing", "gradient_masking",
    "input_transformation", "defensive_distillation", "randomized_smoothing",
    "feature_denoising", "thermometer_encoding", "adversarial_logit_pairing",
    "spatial_smoothing", "jpeg_compression", "pixel_deflection", "gaussian_blur",
    "total_variation_minimization", "word_masking", "median_smoothing",
    "smooth_predict",
]


def test_version_is_exposed():
    assert isinstance(deepdefend.__version__, str)
    assert deepdefend.__version__.count(".") >= 1


@pytest.mark.parametrize("name", EXPECTED_ATTACKS + EXPECTED_DEFENSES)
def test_public_exports_exist(name):
    # These were all missing from the package namespace before 0.2.0.
    assert hasattr(deepdefend, name)
    assert name in deepdefend.__all__


def test_all_names_are_resolvable():
    for name in deepdefend.__all__:
        assert getattr(deepdefend, name) is not None


def test_submodules_are_exposed():
    assert deepdefend.attacks is not None
    assert deepdefend.defenses is not None


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = deepdefend.this_does_not_exist


def test_importing_the_package_does_not_import_tensorflow():
    """`import deepdefend` must stay cheap and TensorFlow-free."""
    code = (
        "import sys\n"
        "import deepdefend\n"
        "assert 'tensorflow' not in sys.modules, 'tensorflow imported eagerly'\n"
        "print(deepdefend.__version__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
