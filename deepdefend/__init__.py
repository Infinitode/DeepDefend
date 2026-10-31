"""
DeepDefend — adversarial attacks and defenses for deep learning models.

Submodules and their public functions are loaded lazily, so ``import
deepdefend`` does not pay the cost of importing TensorFlow::

    import deepdefend
    deepdefend.fgsm                      # imported on first access
    from deepdefend.attacks import pgd   # or import the submodule directly
"""

from __future__ import annotations

import importlib

try:  # pragma: no cover - depends on the installation method
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("deepdefend")
    except PackageNotFoundError:
        __version__ = "0.2.0"
except ImportError:  # pragma: no cover - Python < 3.8
    __version__ = "0.2.0"

_ATTACKS = frozenset(
    {
        "bim",
        "char_swap",
        "cw",
        "deepfool",
        "ead",
        "fgsm",
        "jsma",
        "mim",
        "pgd",
        "spsa",
        "word_swap",
    }
)

_DEFENSES = frozenset(
    {
        "adversarial_logit_pairing",
        "adversarial_training",
        "defensive_distillation",
        "feature_denoising",
        "feature_squeezing",
        "gaussian_blur",
        "gradient_masking",
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
    }
)

_SUBMODULES = frozenset({"attacks", "defenses"})

__all__ = sorted(_ATTACKS | _DEFENSES | _SUBMODULES | {"__version__"})


def __getattr__(name):
    """Lazily import submodules and their public functions (PEP 562)."""
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _ATTACKS or name in _DEFENSES:
        module_name = "attacks" if name in _ATTACKS else "defenses"
        module = importlib.import_module(f".{module_name}", __name__)
        attribute = getattr(module, name)
        globals()[name] = attribute
        return attribute
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
