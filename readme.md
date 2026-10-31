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
certified-radius estimate, a 170-test suite, CI, `SECURITY.md`,
`CODE_OF_CONDUCT.md`, Dependabot, pre-commit and `pyproject.toml`.

**Removed — `gradient_masking`.** The function has been deleted from the
package. It only altered *training* gradients, never the gradients an attacker
computes at inference time, so it had no measurable defensive effect (we
verified that `fgsm(model)` and `fgsm(gradient_masking(model))` produced
bit-identical adversarial examples). Gradient masking is a well documented
**false defense** — it makes a model *look* robust while leaving it vulnerable,
which is exactly the failure mode a robustness tool must not encourage. If you
need it, mask gradients in your own `train_step`; do not report it as a security
measure. Everything else about that class of techniques is covered in
[Limitations](#limitations).

**Breaking** — the numerical output of `cw`, `deepfool`, `ead`, `spsa` and
`jsma` changed (they are now the algorithms they claim to be), several defaults
changed, `gradient_masking` was removed, and image-only defenses now raise on
non-image inputs instead of silently returning the input. See
[CHANGELOG.md](CHANGELOG.md).

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

> `gradient_masking` used to be listed here. It has been removed in 0.2.0 — see
> the note above and [Limitations](#limitations).

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
certified robust unless you have measured it.

- **There is no gradient-masking defense in this package, by design.** Zeroing
  or obfuscating gradients during training does not change the gradients an
  attacker sees; it only produces a false sense of security (Athalye et al.,
  "Obfuscated Gradients Give a False Sense of Security", 2018). The old
  `gradient_masking()` helper was removed in 0.2.0 for that reason. If you want
  gradient regularisation as a *training* technique, implement it in your own
  `train_step` — and never report it as robustness.
- **Randomized smoothing** only provides a certified radius if the base model is
  trained on noise-augmented data and predictions are averaged over many
  samples (`smooth_predict`). Adding noise at inference time alone does not make
  a model robust.
- **Thermometer encoding** here is the shape-preserving quantisation variant.
  Canonical thermometer encoding expands each feature into `num_bins` binary
  features, which requires retraining the first layer.
- **Preprocessing defenses** (blur, JPEG, squeezing, denoising, TV, deflection)
  are often defeated by a small step-size attack. Always evaluate them with a
  strong, adaptive attack (PGD or CW with several restarts).
- **Text helpers** (`word_swap`, `char_swap`, `word_masking`) are simple
  baselines, not semantically-aware attacks.
- **Iterative attacks stop early.** A sample the model already misclassifies is
  returned unchanged (for targeted attacks, a sample already classified as the
  target is returned unchanged).
- **Inputs are not silently clipped.** If your data does not lie in `[0, 1]`,
  pass `clip_min` / `clip_max`; otherwise attacks will not clip at all rather
  than destroy your normalised inputs.

## What was wrong before 0.2.0

0.2.0 fixed defects that made earlier versions actively misleading. This section
is kept so that results produced with 0.1.x can be interpreted correctly, and so
that the reasons behind the API changes are documented.

Every entry below was reproduced by executing the released code against a real
Keras model, not inferred from reading it.

### Attacks

| Function | What it did in 0.1.x | Consequence |
|---|---|---|
| `jsma` | Indexed a rank-4 Jacobian with five indices and mutated an eager tensor in place | Raised on **every** call |
| `spsa` | Passed a tensor (not a `tf.Variable`) to the optimiser and hard-coded rank-4 inputs | Raised on **every** call |
| `ead` | The default `alpha == beta` made the L1 shrinkage cancel every step | Returned the input **unchanged** (`max\|δ\| = 0.0000`) — "model is robust" |
| `cw` | Clipped the input before measuring `‖x − clip(x)‖²`, so the L2 term was always 0 | `c` and `kappa` had **no effect**; not C&W |
| `deepfool` | Added an unnormalised unit step with no budget and no clipping | Returned invalid images in `[-0.90, 2.08]` |
| `fgsm` | Did not clip the output at all | Returned values such as `-0.010` / `1.007` |
| all attacks | Used `CategoricalCrossentropy` only | Rejected sparse integer labels (the default in Keras tutorials) |
| all attacks | Never exposed `from_logits` | Attacked logit models with the wrong loss |
| targeted attacks | Inverted the early-stop mask | Froze at step 0, producing no perturbation |
| `word_swap`, `char_swap` | Round-tripped through `str.split()` / `" ".join()` | Destroyed double spaces, tabs and newlines |

### Defenses

| Function | What it did in 0.1.x | Consequence |
|---|---|---|
| all preprocessing defenses | Defined their layer classes inside the function body | **No defended model could be saved or reloaded** |
| `jpeg_compression` | Relied on non-differentiable JPEG decode | `gradient is None`; attacking the defended model raised `ValueError` |
| `total_variation_minimization` | Nested `GradientTape` inside `Layer.call` | Gradient through the defense was **all zeros** |
| `feature_squeezing` | `floor`/`cast` with no straight-through gradient | No gradient; attacks silently did nothing |
| `median_smoothing` | Called `avg_pool2d` | Was not a median filter at all |
| `gaussian_blur` | Called `avg_pool2d` and ignored `sigma` | `sigma=0.1` and `sigma=9.0` were bit-identical |
| `pixel_deflection` | `tf.random.uniform` with a non-scalar `maxval` | Raised on **every** call |
| `gradient_masking` | Masked training gradients only | Bit-identical to having no defense; **removed in 0.2.0** |
| `adversarial_training` | Generated examples one sample at a time with fixed `Adam` + categorical CE | ~500× more model calls; discarded the caller's compile config |
| `defensive_distillation` | Applied `softmax` to an already-softmaxed model, dropped `T²`, ran the teacher in train mode | Not distillation; noisy soft targets |
| `adversarial_logit_pairing` | Paired two models on the *same clean batch* | Not ALP |
| `randomized_smoothing` | Added unclipped inference-time noise only | No certification; non-deterministic `predict` |
| `feature_squeezing`, `thermometer_encoding` | `bit_depth=0` / `num_bins=0` divided by zero | Returned `NaN` instead of raising |

### Packaging and CI

- `pip install .` from a checkout **failed** — `setup.py` shipped
  `version='{{VERSION_PLACEHOLDER}}'`.
- Dependencies were unbounded, and `python_requires='>=3.6'` claimed support for
  interpreters on which TensorFlow does not exist.
- The publish workflow used `actions/checkout@master` and
  `pypa/gh-action-pypi-publish@master` (mutable refs) with a long-lived
  `PYPI_API_TOKEN`, the removed `::set-output` command, no tag guard, and no
  tests before publishing.
- There were **no tests** and no CI checks at all.

How the fixes are guarded today: 170 tests (`pytest`), `ruff check .`, a CI
workflow on Python 3.10 – 3.12, and a release workflow that refuses to publish
when the tag and `pyproject.toml` version disagree or the tests fail.

## Documentation update brief (for the docs agent)

The block below is a self-contained brief. It is a code cell so it renders as
plain text: paste it into the docs agent as-is after any API change.

```text
DOCS AGENT BRIEF — DeepDefend 0.2.0
===================================

CONTEXT
-------
Repository: Infinitode/DeepDefend (PyPI package: deepdefend)
Branch merged: arena/b52ef87b-deepdefend, released as version 0.2.0.
Published docs live at:
https://infinitode-docs.gitbook.io/documentation/package-documentation/deepdefend-package-documentation

0.2.0 is a correctness release. Several public functions were broken or silently
wrong before it; the API changed accordingly. The docs must describe the 0.2.0
behaviour only. Do not keep pages that document 0.1.x behaviour, except in the
"Migration from 0.1.x" page described below.

TASKS
-----
1. Add a "What's new in 0.2.0" page, linking to CHANGELOG.md in the repo.
2. Create a "Migration from 0.1.x" page (breaking changes; see section below).
3. Update every per-function page for the attacks and defenses listed.
4. Remove the gradient_masking page entirely and redirect it to the new
   "Gradient masking is not a defense" note under Limitations.
5. Update the supported-model-types table so it matches the README exactly.
6. Update the Python version support statement to 3.9-3.12.
7. Add the "Limitations" and "Responsible use" sections verbatim from the README.
8. Add code samples for every function, using 0.2.0 signatures (see below).

REMOVED IN 0.2.0
----------------
gradient_masking(model, mask_threshold=0.1)
  - Deleted from the package. hasattr(deepdefend, "gradient_masking") is False.
  - Reason: it only masked training gradients, never the gradients an attacker
    computes at inference, so it provided no protection (verified: attacks
    against a wrapped model were bit-identical to attacks against the base
    model). Gradient masking is a documented false defense (Athalye et al.,
    2018, "Obfuscated Gradients Give a False Sense of Security").
  - Docs action: delete the page; state clearly that the technique creates a
    false sense of security and that gradient regularisation belongs in the
    user's own train_step, never in a robustness claim.

CHANGED SIGNATURES AND DEFAULTS (attacks)
-----------------------------------------
All attacks additionally accept: from_logits=None, clip_min=None, clip_max=None
and return a NumPy array with the same shape and dtype as the input.

fgsm(model, x, y, epsilon=0.01, *, targeted=False, from_logits=None,
     clip_min=None, clip_max=None)
  - Output is clipped to the valid data range (used to be unclipped).
  - New: targeted.

pgd(model, x, y, epsilon=0.01, alpha=None, num_steps=10, *, random_start=True,
    seed=None, targeted=False, from_logits=None, clip_min=None, clip_max=None)
  - alpha now defaults to epsilon / 4 (used to be a fixed 0.01).
  - New: random_start, seed, targeted.

bim(model, x, y, epsilon=0.01, alpha=None, num_steps=10, *, seed=None,
    targeted=False, from_logits=None, clip_min=None, clip_max=None)
  - Now documented as PGD without a random start (they used to be identical).

mim(model, x, y, epsilon=0.01, alpha=None, num_steps=10, decay_factor=1.0, *,
    seed=None, targeted=False, from_logits=None, clip_min=None, clip_max=None)
  - Momentum is now normalised per sample, not across the batch.

cw(model, x, y, epsilon=None, c=10.0, kappa=0.0, num_steps=100, alpha=0.01, *,
   targeted=False, from_logits=None, clip_min=None, clip_max=None)
  - This is now the real Carlini & Wagner L2 attack (tanh-space optimisation of
    ||delta||^2 + c * margin).
  - epsilon defaults to None (unconstrained). If set, it is an L2 budget
    expressed as an L-infinity equivalent: radius = epsilon * sqrt(num_features).
  - c and kappa now genuinely change the result; document that c trades
    distortion against the margin and should be swept to minimise distortion.
  - The untargeted target class is the runner-up class of the original input.

deepfool(model, x, y, num_steps=50, epsilon=None, overshoot=0.02, *,
         from_logits=None, clip_min=None, clip_max=None)
  - Now computes per-class decision boundaries and respects epsilon / clip range
    (used to return out-of-range images and ignore any budget).

jsma(model, x, y, theta=0.0, gamma=0.2, num_steps=10, *, targeted=False,
     from_logits=None, clip_min=None, clip_max=None)
  - theta defaults to 0.0 (always pick the most salient feature); raw saliency
    values are products of two gradients and therefore tiny.

spsa(model, x, y, epsilon=0.01, num_steps=50, learning_rate=0.05, delta=None,
     spsa_samples=32, *, seed=None, targeted=False, from_logits=None,
     clip_min=None, clip_max=None)
  - Accepts any batch size and any input rank.
  - delta defaults to epsilon / 5.

ead(model, x, y, epsilon=0.01, beta=0.001, num_steps=50, alpha=0.01, *, c=0.1,
    kappa=0.0, targeted=False, from_logits=None, clip_min=None, clip_max=None)
  - beta defaults to 0.001 (0.01 made the default attack a no-op).

word_swap(text, swap_dict=None, max_swaps=None, seed=None)
  - swap_dict is now required (ValueError when missing).
  - Whitespace, newlines and punctuation are preserved.

char_swap(text, swap_prob=0.1, seed=None)
  - Whitespace/punctuation preserved; adds seed.

LABEL HANDLING (document as a shared concept)
---------------------------------------------
Every attack accepts one-hot (batch, classes), sparse integer (batch,) and
binary (batch, 1) targets. from_logits is auto-detected from the last layer and
can be overridden. clip_min/clip_max are inferred: [0, 1] when the batch already
lies in that range, otherwise no clipping (so normalised inputs are not
destroyed). Iterative attacks stop working on a sample once it reaches the goal,
so already-misclassified samples are returned unchanged.

NEW FUNCTIONS (defenses)
------------------------
smooth_predict(model, x, n_samples=100, noise_level=None, seed=None)
  - Monte-Carlo prediction for randomized_smoothing models.
  - Returns dict with keys: predictions, probabilities, certified_radius.
  - certified_radius = sigma * Phi^-1(p_hat) for the majority class, or 0.
  - Document that it requires a noise-trained base model.

FeatureDenoisingBlock(kernel_size=3)
  - Keras layer to insert inside a network for feature-level denoising
    (learned residual box filter, starts as an identity).

defenses.FeatureDenoisingBlock is importable as
`from deepdefend.defenses import FeatureDenoisingBlock`.

CHANGED DEFENSE SIGNATURES
--------------------------
adversarial_training(model, x, y, epsilon=0.01, epochs=10, batch_size=32,
                     attack=None, optimizer=None, loss=None, verbose=1,
                     shuffle=True)
  - Generates adversarial examples in batches, keeps the caller's optimiser and
    loss, returns a NEW model (a clone).

defensive_distillation(model, teacher_model, temperature=2)
  - Distillation loss is scaled by T^2; the teacher runs in inference mode.

adversarial_logit_pairing(model, paired_model=None, epsilon=0.01, num_steps=1,
                          alpha=None)
  - Generates adversarial examples inside the training step.

feature_denoising(model, method="median", kernel_size=3, sigma=1.0)
spatial_smoothing(model, kernel_size=3, method="median", sigma=1.0)
median_smoothing(model, kernel_size=3)   # alias of spatial_smoothing(median)
gaussian_blur(model, kernel_size=3, sigma=1.0)  # sigma is now used
randomized_smoothing(model, noise_level=0.1, seed=None,
                     add_noise_at_inference=True)
pixel_deflection(model, deflection_count=100, window_size=10, seed=None)
thermometer_encoding(model, num_bins=10)
word_masking(text, mask_token="[MASK]", mask_prob=0.1, seed=None)
input_transformation(model, transformation_function)  # now REQUIRED

BEHAVIOUR THE DOCS MUST STATE
-----------------------------
1. All preprocessing defenses return a tf.keras.Model that can be saved with
   model.save() and reloaded with tf.keras.models.load_model() without
   custom_objects.
2. Image-only defenses (gaussian_blur, median_smoothing, spatial_smoothing,
   jpeg_compression, pixel_deflection, total_variation_minimization,
   feature_denoising) raise ValueError on non-4-D inputs instead of silently
   returning the input. JPEG requires 1 or 3 channels.
3. Subclassed or multi-input models raise TypeError: these defenses need a
   Functional/Sequential model with a known input_shape.
4. Invalid parameters raise ValueError instead of producing NaN:
   bit_depth must be 1-8, num_bins >= 2, quality 1-100, kernel sizes odd,
   noise levels >= 0, epsilon >= 0, num_steps >= 1.
5. feature_squeezing, jpeg_compression and total_variation_minimization use a
   straight-through gradient so gradient-based attacks against a defended model
   still work.
6. word_masking, word_swap and char_swap preserve whitespace, newlines and
   punctuation exactly.
7. All randomized helpers accept seed= and are reproducible when it is set.

MIGRATION FROM 0.1.x (new page)
-------------------------------
- gradient_masking() no longer exists; remove it from user code.
- Results produced with cw, deepfool, ead, spsa or jsma under 0.1.x are invalid
  and should be re-run.
- cw: the first positional argument after `y` is still epsilon, but its default
  is now None and its meaning is an optional L2 budget.
- jsma: theta now defaults to 0.0.
- pgd/bim/mim: alpha now defaults to epsilon / 4; pass alpha explicitly to keep
  the old step size.
- spsa: no longer restricted to batch size 1 or rank-4 inputs.
- Attacks now accept sparse labels, so `y.argmax(-1)` no longer needs
  converting to one-hot.
- Defended models saved with 0.1.x cannot be loaded; retrain or re-wrap them.
- Python 3.9+ is required.

TONE / FORMAT
-------------
- Each function page: one-line description, signature block, parameters table,
  returns, a runnable example, and a "Caveats" note.
- Be explicit that preprocessing defenses are not robustness guarantees; link to
  the Limitations page.
- Never present an attack's success rate as a defense's guarantee.
- Code samples must be copy-pasteable and use only public API.
```

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
