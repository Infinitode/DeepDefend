# DeepDefend
![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)
![Code Size](https://img.shields.io/github/languages/code-size/infinitode/deepdefend)
![Downloads](https://pepy.tech/badge/deepdefend)
![License Compliance](https://img.shields.io/badge/license-compliance-brightgreen.svg)
![PyPI Version](https://img.shields.io/pypi/v/deepdefend)

[Documentation](https://infinitode-docs.gitbook.io/documentation/package-documentation/deepdefend-package-documentation)

An open-source Python library for adversarial attacks and defenses in deep learning models, enhancing the security and robustness of AI systems.

## Changes in 0.2.0

This release is a correctness pass over the whole package: several functions
either crashed on every call or returned silently wrong results. See
[CHANGELOG.md](CHANGELOG.md) for the full list.

**Fixed**
- Fixed `jsma`, `spsa` and `pixel_deflection`, which raised an exception on every call.
- Fixed `ead`, which returned the input unchanged with its default arguments (a silent "no vulnerability found").
- Fixed `cw`, where `c` and `kappa` had no effect on the result.
- Fixed `deepfool`, which returned out-of-range values (e.g. `-0.90` and `2.08` for `[0, 1]` inputs) and ignored any budget.
- Fixed `fgsm`, which returned unclipped, invalid inputs.
- Fixed all attacks rejecting sparse integer labels, the default label format in Keras tutorials.
- Fixed all attacks ignoring `from_logits`, so models not ending in `softmax` were attacked with the wrong loss.
- Fixed targeted attacks freezing at the first step.
- Fixed `median_smoothing`, which called `avg_pool2d` instead of a median filter.
- Fixed `gaussian_blur`, which ignored `sigma` and average-pooled instead.
- Fixed `jpeg_compression`, `feature_squeezing` and `total_variation_minimization` breaking the gradient, so attacks against a defended model crashed or silently did nothing.
- Fixed all preprocessing defenses producing models that could not be saved or reloaded.
- Fixed `adversarial_training` generating examples one sample at a time and discarding the caller's optimiser and loss.
- Fixed `defensive_distillation` applying softmax twice and dropping the `T²` scaling.
- Fixed `adversarial_logit_pairing`, which paired two models on clean inputs instead of clean and adversarial logits.
- Fixed `randomized_smoothing`, `pixel_deflection` and the text helpers being non-reproducible and destroying whitespace/newlines.
- Fixed silent `NaN`s from invalid parameters such as `bit_depth=0` or `num_bins=0`.
- Fixed `pip install .` from a checkout, which failed because the version was a placeholder.
- Fixed the release pipeline: SHA-pinned actions, a tag/version check, a test gate, and PyPI Trusted Publishing instead of a long-lived API token.

**Added** — `deepdefend.__version__`, lazy imports, `smooth_predict()` with a
certified-radius estimate, a 171-test suite, CI, `SECURITY.md`,
`CODE_OF_CONDUCT.md`, Dependabot, pre-commit and `pyproject.toml`.

**Breaking** — the numerical output of `cw`, `deepfool`, `ead`, `spsa` and
`jsma` changed (they are now the algorithms they claim to be), several defaults
changed, and image-only defenses now raise on non-image inputs instead of
silently returning the input. See [CHANGELOG.md](CHANGELOG.md).

## Installation

You can install DeepDefend using pip:

```bash
pip install deepdefend
```

For development (tests and linting):

```bash
git clone https://github.com/Infinitode/DeepDefend.git
cd DeepDefend
pip install -e ".[dev]"
pytest
```

## Supported Python Versions

DeepDefend supports Python 3.9 – 3.12, matching the versions supported by
TensorFlow 2.x. Python 3.6 – 3.8 are no longer supported because TensorFlow does
not publish wheels for them.

## Features

- Adversarial Attacks: Generate adversarial examples to evaluate model vulnerabilities.
- Adversarial Defenses: Employ various methods to protect models against adversarial attacks.

## Supported Model Types

Legend: ✅ supported · ❌ unsupported (raises a clear error)

| Feature | Image | Text | Numeric | Classification |
|---------|:-----:|:----:|:-------:|:--------------:|
| **Attacks** | | | | |
| FGSM | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| PGD | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| BIM | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| MIM | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| CW (L2) | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| EAD | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| DeepFool | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| JSMA | ✅ | ❌ | ✅ | ✅ |
| SPSA | ✅ | ❌ | ✅ | ✅ |
| Word Swap | ❌ | ✅ | ❌ | — |
| Char Swap | ❌ | ✅ | ❌ | — |
| **Defenses** | | | | |
| Adversarial Training | ✅ | ✅ | ✅ | ✅ |
| Feature Squeezing | ✅ | ✅ | ✅ | ✅ |
| Gradient Masking (⚠️ not a defense) | ✅ | ✅ | ✅ | ✅ |
| Input Transformation | ✅ | ✅ | ✅ | ✅ |
| Defensive Distillation | ✅ | ✅ | ✅ | ✅ |
| Randomized Smoothing | ✅ | ✅ | ✅ | ✅ |
| Thermometer Encoding | ✅ | ✅ | ✅ | ✅ |
| ALP | ✅ | ✅ | ✅ | ✅ |
| Feature Denoising | ✅ | ❌ | ❌ | ✅ |
| Spatial Smoothing | ✅ | ❌ | ❌ | ✅ |
| Median Smoothing | ✅ | ❌ | ❌ | ✅ |
| JPEG Compression | ✅ | ❌ | ❌ | ✅ |
| Pixel Deflection | ✅ | ❌ | ❌ | ✅ |
| Gaussian Blur | ✅ | ❌ | ❌ | ✅ |
| TV Minimization | ✅ | ❌ | ❌ | ✅ |
| Word Masking | ❌ | ✅ | ❌ | — |

## Usage

### Adversarial Attacks

```python
import numpy as np
import tensorflow as tf
from deepdefend.attacks import fgsm, pgd, bim, cw, deepfool, jsma, spsa, mim, ead

# Load a pre-trained TensorFlow model
model = ...

# Load example input and label data (replace this with your own data loading code)
x_example = ...  # shape (batch, ...), values in [0, 1]
y_example = ...  # one-hot (batch, classes), sparse indices (batch,) or binary (batch, 1)

# Single-step attack
adversarial_fgsm = fgsm(model, x_example, y_example, epsilon=0.01)

# Iterative attacks (alpha defaults to epsilon / 4)
adversarial_pgd = pgd(model, x_example, y_example, epsilon=0.01, num_steps=10)
adversarial_bim = bim(model, x_example, y_example, epsilon=0.01, num_steps=10)
adversarial_mim = mim(model, x_example, y_example, epsilon=0.01, num_steps=10)

# Optimisation based attacks
adversarial_cw = cw(model, x_example, y_example, c=10.0, num_steps=100)
adversarial_ead = ead(model, x_example, y_example, epsilon=0.01, num_steps=50)
adversarial_deepfool = deepfool(model, x_example, y_example, num_steps=50)
adversarial_spsa = spsa(model, x_example, y_example, epsilon=0.01, num_steps=50)

# Saliency map attack
adversarial_jsma = jsma(model, x_example, y_example, num_steps=10)

# Targeted attack: y_example is the class you want the model to predict
adversarial_targeted = pgd(model, x_example, target_labels, epsilon=0.03, targeted=True)

# Models that return logits are auto-detected; override with from_logits=True/False.
# Normalised (e.g. ImageNet) inputs: tell the attack the valid range so it does
# not clip your input to [0, 1].
adversarial = pgd(
    model, x_normalised, y_example, epsilon=0.01,
    clip_min=-2.2, clip_max=2.7,
)

# Text attacks preserve whitespace, newlines and punctuation
from deepdefend.attacks import word_swap, char_swap

perturbed_text = word_swap("The movie was great", {"great": "terrible"})
perturbed_text = char_swap("The movie was great", swap_prob=0.1, seed=0)
```

### Adversarial Defenses

```python
import tensorflow as tf
from deepdefend.defenses import *

# Load a pre-trained TensorFlow model
model = ...

# Teacher model for distillation
teacher_model = ...

# Load training data
x_train, y_train = ...
```

```python
# Training-time defenses
defended_model = adversarial_training(
    model, x_train, y_train, epsilon=0.01, epochs=10, batch_size=32
)
defended_model = defensive_distillation(model, teacher_model, temperature=2)
defended_model = adversarial_logit_pairing(model, epsilon=0.01)
defended_model = gradient_masking(model, mask_threshold=0.1)  # ⚠️ see below
```

```python
# Inference-time preprocessing defenses (these return a new Keras model)
defended_model = feature_squeezing(model, bit_depth=4)
defended_model = gaussian_blur(model, kernel_size=3, sigma=1.0)
defended_model = median_smoothing(model, kernel_size=3)
defended_model = spatial_smoothing(model, kernel_size=3, method="median")
defended_model = jpeg_compression(model, quality=75)
defended_model = pixel_deflection(model, deflection_count=100, window_size=10)
defended_model = total_variation_minimization(model, iterations=10)
defended_model = thermometer_encoding(model, num_bins=10)
defended_model = feature_denoising(model, method="median")
defended_model = input_transformation(model, tf.keras.layers.Rescaling(1.0 / 255))
defended_model = randomized_smoothing(model, noise_level=0.1)

# ... and they can be saved and reloaded like any other Keras model
defended_model.save("defended.keras")
restored = tf.keras.models.load_model("defended.keras")

# Monte-Carlo prediction with a certified L2 radius (randomized smoothing)
result = smooth_predict(defended_model, x_test, n_samples=100)
print(result["predictions"], result["certified_radius"])
```

```python
# Text
defended_text = word_masking("The movie was great", mask_prob=0.2)
```

## Limitations

Adversarial defenses are an active research area, and most preprocessing
defenses can be broken by an adaptive attacker. Do not treat a defended model as
certified robust unless you have measured it:

- **Gradient masking is not a defense.** It only alters training gradients. It
  has no effect on the gradients an attacker sees, and `gradient_masking()` emits
  a `RuntimeWarning` for this reason.
- **Randomized smoothing** only provides a certified radius if the base model is
  trained on noise-augmented data and predictions are averaged over many
  samples (`smooth_predict`). Adding noise at inference time alone does not make
  a model robust.
- **Thermometer encoding** here is the shape-preserving quantisation variant.
  Canonical thermometer encoding expands each feature into `num_bins` binary
  features, which requires retraining the first layer.
- **Preprocessing defenses** (blur, JPEG, squeezing, denoising, TV, deflection)
  are often defeated by a small step-size attack (Athalye et al., 2018). Always
  evaluate them with a strong, adaptive attack (PGD or CW with several restarts).
- **Text helpers** (`word_swap`, `char_swap`, `word_masking`) are simple
  baselines, not semantically-aware attacks.

## Responsible use

The attack implementations exist so that defenders can measure and improve the
robustness of models they are authorised to test. Do not use this library
against systems you do not own or have explicit permission to assess. To report
a security issue in DeepDefend itself, see [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome! If you encounter any issues, have suggestions, or want to contribute to DeepDefend, please open an issue or submit a pull request on [GitHub](https://github.com/infinitode/deepdefend). Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first.

## License

DeepDefend is released under the terms of the **MIT License (Modified)**. Please see the [LICENSE](https://github.com/infinitode/deepdefend/blob/master/LICENSE) file for the full text.

**Modified License Clause**

The modified license clause grants users the permission to make derivative works based on the DeepDefend software. However, it requires any substantial changes to the software to be clearly distinguished from the original work and distributed under a different name.

By enforcing this distinction, it aims to prevent direct publishing of the source code without changes while allowing users to create derivative works that incorporate the code but are not exactly the same.

Please read the full license terms in the [LICENSE](https://github.com/infinitode/deepdefend/blob/master/LICENSE) file for complete details.
