# Changelog

All notable changes to DeepDefend are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

This release is a correctness and security pass over the whole package. Many
functions that previously returned *silently wrong* results (or crashed) now do
what their name and documentation claim. Because the numerical output of several
attacks and defenses changes, the minor version was bumped.

### Fixed — attacks

- **`jsma`** no longer raises on every call (it indexed a rank-4 Jacobian with
  five indices and mutated an eager tensor in place). It is now a real
  saliency-map attack that works on any input rank.
- **`spsa`** no longer raises on every call (it passed a tensor, not a
  `tf.Variable`, to the optimizer and hard-coded rank-4 inputs). The gradient
  estimate now uses the correct finite difference
  `(L(x+δd) − L(x−δd)) / (2δ)`, the optimiser is Adam, and any number of
  samples and dimensions is accepted.
- **`ead`** no longer returns the input unchanged with its default arguments.
  It now performs a proper ISTA/proximal step on the elastic-net objective.
- **`cw`** is now the Carlini & Wagner L2 attack: the tanh-space optimisation
  actually minimises distortion and the `c` / `kappa` parameters change the
  result (previously they were inert). The untargeted target class is fixed at
  the runner-up class instead of being recomputed every step.
- **`deepfool`** is now DeepFool: per-class decision boundaries, an optional
  L∞ budget, and output that stays inside the valid data range (it used to
  return values such as `-0.90` and `2.08` for `[0, 1]` inputs).
- **`fgsm`** output is clipped to the valid data range instead of overshooting
  to `-0.01` / `1.007`.
- **All attacks** now accept sparse integer labels, one-hot labels and binary
  targets; previously every attack rejected the sparse-label format used by most
  Keras examples.
- **All attacks** now auto-detect `from_logits` (with an override), call the
  model with `training=False`, and support an explicit `clip_min` / `clip_max`
  for normalised inputs.
- **Targeted attacks** no longer freeze at step 0 (the early-stop mask had the
  logic inverted for targeted mode).
- **`pgd`** gained `random_start` and no longer duplicates `bim`.
- **DeepFool / JSMA / SPSA** no longer project against the whole batch inside
  their per-sample loops.
- **`word_swap` / `char_swap`** preserve whitespace, newlines and punctuation,
  accept a `seed`, and `word_swap` no longer silently returns the input when
  `swap_dict` is missing.

### Fixed — defenses

- **All preprocessing defenses** can now be saved and reloaded. Their layers
  were previously defined inside the function body, so every defended model
  failed at `model.save()` / `load_model()`.
- **`jpeg_compression`**, **`feature_squeezing`** and
  **`total_variation_minimization`** no longer break the gradient: attacks
  against a defended model used to raise `ValueError` (JPEG) or silently return
  a zero perturbation (TV, quantisation). They now use a straight-through
  gradient.
- **`median_smoothing`** is a real median filter again (it was calling
  `avg_pool2d`).
- **`gaussian_blur`** builds an actual Gaussian kernel and honours `sigma`
  (the parameter was previously ignored).
- **`pixel_deflection`** no longer raises on every call, works inside
  `tf.function`, and is reproducible when a `seed` is given.
- **`adversarial_training`** generates adversarial examples in batches instead
  of one sample at a time (~500× fewer model calls), honours the caller's
  optimiser and loss instead of forcing Adam + categorical cross-entropy, and
  accepts `epochs`, `batch_size`, `verbose` and a custom `attack`.
- **`defensive_distillation`** no longer applies softmax twice, scales the
  distillation term by `T²`, evaluates the teacher in inference mode, and works
  for both probability and logit outputs.
- **`adversarial_logit_pairing`** generates adversarial examples inside the
  training step; the previous implementation paired two models on clean inputs.
- **`randomized_smoothing`** clips the noisy input, is reproducible when
  seeded, and ships with `smooth_predict()` for Monte-Carlo prediction and a
  certified-radius estimate.
- **`feature_denoising`** has an explicit `method` (`median` / `gaussian`) and a
  new `FeatureDenoisingBlock` layer for feature-level denoising.
- **`input_transformation`** raises instead of silently returning an
  undefended model when no transformation is given.
- **`word_masking`** preserves whitespace and accepts a `seed`.
- **All defenses** raise a clear `TypeError` for subclassed or multi-input
  models instead of an obscure `AttributeError`, and reject invalid parameters
  (`bit_depth` outside 1-8, `num_bins < 2`, `quality` outside 1-100, even kernel
  sizes, negative noise levels, ...) instead of producing `NaN`.

### Added

- `deepdefend.__version__`, `deepdefend.__all__`, and lazy imports: every public
  function is reachable as `deepdefend.<name>`, and `import deepdefend` no
  longer imports TensorFlow.
- `deepdefend/smooth_predict` for Monte-Carlo randomized smoothing.
- 170 tests covering attacks, defenses, serialisation, gradients, label
  formats, input validation and the public API.
- CI: lint + test matrix (3.10-3.12) + reproducible build on every push and PR.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, Dependabot, pre-commit and
  `pyproject.toml`.

### Changed

- `setup.py` was replaced by `pyproject.toml`; the version is now static and
  `pip install .` works from a checkout (it previously failed because the
  version was a `{{VERSION_PLACEHOLDER}}`).
- Dependencies are bounded (`numpy<3`, `tensorflow>=2.13,<3`) and
  `python_requires` is `>=3.9` (TensorFlow 2.x is not available on 3.6-3.8).
- `pgd` / `bim` / `mim` default `alpha` is now `epsilon / 4` (previously
  `0.01`, which equals `epsilon` for the default settings).
- `jsma` defaults are `theta=0.0`, `gamma=0.2`.
- `cw` defaults are `epsilon=None` (unconstrained), `c=10.0`.
- `spsa` defaults are `num_steps=50`, `learning_rate=0.05`, `spsa_samples=32`
  probe directions, `delta=epsilon/5`.
- `median_smoothing` is documented as an alias of `spatial_smoothing`.
- The publish workflow uses SHA-pinned actions, a tag/version consistency check,
  a test gate, `python -m build`, `twine check`, the `PYPI_API_TOKEN` secret and
  a `pypi` environment. CI runs the full Python matrix only on `master`/`main`.

### Removed

- **`gradient_masking`** (and the `GradientMaskingModel` wrapper) has been
  deleted. It only masked *training* gradients and had no effect on the
  gradients an attacker computes at inference time — attacks against a wrapped
  model were bit-identical to attacks against the base model. Gradient masking
  is a documented false defense (Athalye et al., 2018) that makes a model look
  robust while leaving it vulnerable, so it does not belong in a robustness
  library. Mask gradients in your own `train_step` if you want the training
  regulariser, and never report it as a security measure.
- The `setup.py`-based `{{VERSION_PLACEHOLDER}}` release mechanism.

## [0.1.5]

- Added MIM and EAD attacks, word/character swap attacks, pixel deflection,
  Gaussian blur, TV minimisation and median smoothing defenses, word masking,
  and a support table.

## [0.1.4]

- Added the SPSA attack and the JPEG compression defense.

## [0.1.3]

- Added randomized smoothing, feature denoising, thermometer encoding,
  adversarial logit pairing and spatial smoothing.

## [0.1.2]

- Three new attacks and three new defenses.
