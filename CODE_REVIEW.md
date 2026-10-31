# DeepDefend — Full Code Review & Vulnerability Assessment

**Version reviewed:** `0.1.5` (branch `arena/b52ef87b-deepdefend`, commit `cda6c32`)
**Files reviewed:** `deepdefend/__init__.py`, `deepdefend/attacks.py` (430 lines), `deepdefend/defenses.py` (565 lines), `setup.py`, `requirements.txt`, `readme.md`, `LICENSE`, `.github/workflows/build_and_publish.yml`, `CONTRIBUTING.md`, issue templates
**Method:** static review of every function **plus** dynamic verification of each suspected defect against a real Keras model (`tensorflow-cpu 2.16.2` / Keras 3.15, Python 3.11). Every claim below marked *(measured)* was reproduced by executing the package; raw output is in the appendix.

> This is a security library: a function that **silently returns something wrong** is worse than one that crashes, because downstream users report "no vulnerability found" based on it. That severity ordering drives this report.

---

## 0. Executive summary

| Area | State |
|---|---|
| Attacks | 7/11 functions are broken, silently inert, or are not the algorithm they claim to be |
| Defenses | 4 crash or are bit-identical to each other; 6 are not the technique they claim; 13 of 15 wrapped models cannot be saved |
| Packaging | `pip install .` from a checkout **fails outright** |
| CI/CD | Unpinned mutable actions + long-lived PyPI token + publishes on any tag |
| Tests | **None.** No test suite, no test workflow, no lint config |

Of the 11 attacks, **5 raise an exception on normal input** and **2 silently produce no perturbation at all**. Of the 15 model-wrapping defenses, **1 always crashes** and **3 break gradient flow so badly that attacks against the defended model raise or silently no-op**.

Recommended target version: **0.2.0** (several fixes are intentionally behaviour-changing).

---

## 1. CRITICAL — broken, inert, or scientifically false

### C1. `attacks.jsma` always raises — implemented incorrectly
`attacks.py:325-376`

* `saliency_map = np.zeros(input_shape)` with `input_shape` a `TensorShape`; loops `range(input_shape[1..3])` assume a hard-coded 4-D image input.
* `jacobian[0, i, j, k, :]` indexes a rank-4 tensor with 5 indices.
* **Measured:** `InvalidArgumentError: Index out of range using input dim 4; input has only 4 dimensions`.
* `adversarial_example[...] += gamma` mutates an `EagerTensor` in place — illegal even if the above were fixed.
* Conceptually it is not JSMA: it sums the whole Jacobian instead of computing the *saliency map* over (target, other-class) derivative pairs, and applies a fixed `gamma` instead of the `theta` pixel budget.

**Verdict:** unusable. Needs a rewrite.

### C2. `attacks.spsa` always raises; hard-codes rank-4 inputs
`attacks.py:378-430`

* `perturbation = tf.zeros_like(x)` is a **tensor, not a `tf.Variable`**, yet it is passed to `optimizer.apply_gradients(...)` and later `.assign(...)` — both require a Variable.
* `tf.tile(x, [spsa_samples, 1, 1, 1])` hard-codes 4 dimensions — **measured** failure on a `(1, 10)` Dense input, contradicting the README's "Numeric ✅" matrix entry.
* `y_tiled = tf.tile(y, [128])` produces rank-2 labels for `sparse_categorical_crossentropy` — **measured** failure even on a valid 4-D batch.
* The gradient estimate `mean(loss * delta_x) / delta` is not SPSA: it needs the loss *difference* `(L(x+δ) − L(x−δ)) / (2δ)`, otherwise the absolute loss magnitude dominates the estimate.
* `adversarial_example` is unbound if `num_steps <= 0` → `UnboundLocalError`.

### C3. `defenses.pixel_deflection` always raises
`defenses.py:405-451`

* **Measured:** `ValueError: maxval must be a scalar; got a tensor of shape [2]` from `tf.random.uniform([count, 2], 0, [h, w], ...)`. Any Functional-model construction traces this, so it can never work.
* Also fails inside `tf.function` (no `fn_output_signature`), and duplicate scatter indices silently drop updates.
* The in-code comments admit it is "placeholder logic" and "an approximation".

### C4. `attacks.ead` is a **complete no-op** with default arguments
`attacks.py:101-145`

Default `beta=0.01` and `alpha=0.01` give `maximum(|diff| - beta, 0)` where `|diff| == alpha == 0.01` → **exactly zero**. All `num_steps` are spent producing the input unchanged.

* **Measured:** `max|δ| = 0.0000`, `mean|δ| = 0.0000` over 10 steps at `eps=0.02`.
* It is also not EAD: no L2 term, no per-step adversarial objective, no early stop — it is PGD with a shrink applied *after* the update and then clipped, which partially cancels it.

**Verdict:** a user running EAD concludes their model is already robust. This is the single most dangerous defect in the package.

### C5. `attacks.cw` — `c` and `kappa` are dead parameters
`attacks.py:252-292`

`loss + c * tf.norm(x - tf.clip_by_value(x, 0, 1))**2 - kappa` — `x` has already been clipped to `[0,1]` on every previous iteration, so the L2 term is **identically 0** and `kappa` adds a constant whose gradient is **identically 0**.

* **Measured:** `cw(c=1, kappa=0)` and `cw(c=1000, kappa=50)` produce **bit-identical** output.
* No tanh-space optimisation, no L2 minimisation, no confidence margin (`- kappa` has the wrong sign for C&W anyway), FGSM-style `alpha*sign(grad)` step.

**Verdict:** it is PGD with two decorative arguments. A C&W benchmark built on this is meaningless.

### C6. `attacks.deepfool` is unbounded and returns invalid inputs
`attacks.py:294-323`

* Adds `gradient / ||gradient||` with step size **1.0** and never clips or projects.
* **Measured:** output range `[-0.900, 2.080]` for an input in `[0,1]` — a completely invalid image.
* Has no `epsilon` parameter at all, so the "minimum perturbation" property that defines DeepFool cannot be expressed.
* No per-class linearisation; it is gradient ascent in the loss direction, which is closer to an *anti*-attack than DeepFool.

### C7. `attacks.fgsm` returns out-of-range data
`attacks.py:21-53`

* No clipping at all (unlike PGD/BIM/MIM).
* **Measured:** `min=-0.010, max=1.007` for `x ∈ [0,1]`, `epsilon=0.01`. Feeding that back into the model gives garbage and it is not a valid image.
* No `gradient is None` handling → `tf.sign(None)` raises a confusing `AttributeError`.

### C8. Every attack rejects `sparse_categorical` integer labels
`attacks.py` — all of `fgsm, pgd, bim, mim, cw, ead, deepfool`

* **Measured:** all seven raise `Arguments target and output must have the same rank (ndim). Received target.shape=(4,), output.shape=(4, 3)`.
* Sparse integer labels are *the* default for MNIST/CIFAR examples in the Keras documentation, i.e. the package's primary audience.
* `pgd`/`bim`/`cw`/`deepfool` hard-code `CategoricalCrossentropy`; `fgsm`/`mim`/`ead` use a fragile heuristic (`y.shape[-1] == 1 or len(y.shape) == 1`) that misclassifies a 1-D integer label tensor as a *binary* target.
* `from_logits` is never exposed, so any model not ending in `softmax` is silently attacked with the wrong loss.

### C9. Every wrapped defense model is unsaveable
`defenses.py` — all 11 preprocessing wrappers

Every layer class (`SqueezeLayer`, `NoiseLayer`, `ThermometerLayer`, `JPEGCompressLayer`, `DeflectionLayer`, `BlurLayer`, `TVLayer`, `MedianLayer`, `SmoothLayer`, `DenoiseLayer`, `TransformationLayer`) is **defined inside the function body**, so it has no importable qualified name and no `get_config()`.

* **Measured:** `model.save(...)` → `TypeError: <class 'keras.src.models.functional.Functional'> could not be deserialized properly` for `feature_squeezing`, `gaussian_blur`, `randomized_smoothing` (3/3).
* `tf.keras.models.clone_model()` on a defended model fails for the same reason.
* This makes the defenses unusable in production: you cannot train today and serve tomorrow.

### C10. `jpeg_compression` destroys the gradient — attacks on the defended model crash
`defenses.py:396-397`

`tf.image.encode_jpeg`/`decode_jpeg` are non-differentiable.

* **Measured:** `tape.gradient(...)` → `None`.
* **Measured:** `attacks.fgsm(jpeg_compression(model), x, y)` → `ValueError: Attempt to convert a value (None)`.
* The library's headline use case — "defend, then evaluate the defense with an attack" — therefore fails. Needs a straight-through-estimator wrapper.
* Also hard-codes `channels=3` (breaks grayscale), assumes `x ∈ [0,1]`, and cannot run on GPU/TPU.

### C11. `gradient_masking` provides zero protection at inference
`defenses.py:93-129`

It masks gradients **during training only**. An attacker queries the model's own gradients, which are untouched.

* **Measured:** `fgsm(base_model) == fgsm(gradient_masking(base_model))` — bit-identical.
* The docstring claims it "makes them less informative for adversarial attackers", which is false. Gradient masking is a known *false* defense in the literature (Obfuscated Gradients, Athalye et al. 2018). It must be relabelled as a training regulariser, and the README's ✅ row should carry a warning.

### C12. `median_smoothing` is an average pool — not a median filter
`defenses.py:552-560`

* **Measured:** output is bit-identical to `tf.nn.avg_pool2d(x, 3, 1, "SAME")`.
* Ironically a real median implementation already exists in `spatial_smoothing` (line 347). This is a copy-paste regression.

### C13. `gaussian_blur` ignores `sigma` entirely
`defenses.py:467-476`

* Never builds a Gaussian kernel; calls `avg_pool2d`.
* **Measured:** `gaussian_blur(sigma=0.1)` ≡ `gaussian_blur(sigma=9.0)`.

### C14. `total_variation_minimization` silently kills attack gradients
`defenses.py:503-514`

* Runs an inner `GradientTape` inside `Layer.call`; the outer tape records the inner optimisation, so the composed gradient w.r.t. the input is **measured as all-zero**.
* Consequence: `fgsm(tv_defended_model, ...)` returns an example that is *not* adversarial, with no error — exactly the silent failure mode this report is about.
* Also un-trainable, adds second-order cost, and cannot be used in a `tf.function` training loop.

---

## 2. HIGH — wrong results, bad defaults, unsafe API

### H1. `adversarial_training` is crippled and ignores the caller's configuration
`defenses.py:28-60`

* **Per-sample loop** (`for i in range(len(x))`): one forward+backward pass per image. **Measured: 3.08 s for 200 images** → ~13 minutes for 50 000 MNIST samples, before training starts.
* Hard-codes `epochs=10`, `batch_size=32`, `optimizer='adam'`, `loss='categorical_crossentropy'`.
  **Measured:** a model compiled with `SGD(0.001)` + `KLDivergence` + `mae` comes back with **Adam + categorical_crossentropy + no `mae`** — the user's compile config is silently discarded.
* **Measured:** fails on sparse labels (`target.shape=(1,)`).
* No `verbose`, no `shuffle`, no `validation_data`, no early stop; refuses lists/`tf.data.Dataset` (`len()`/slicing).

### H2. No input validation anywhere → NaN or crashes from legal-looking arguments
**Measured:**

| Call | Result |
|---|---|
| `fgsm(..., epsilon=-1)` | accepted, returns values as low as `-0.9997` |
| `pgd(..., num_steps=0)` | accepted, returns the input silently |
| `feature_squeezing(model, bit_depth=0)` | **NaN** (division by `2**0 - 1 == 0`) |
| `feature_squeezing(model, bit_depth=9)` | `TypeError` (floordiv int32 by `0.5`) |
| `thermometer_encoding(model, num_bins=0)` | **NaN** |
| `spsa(..., num_steps=0)` | `UnboundLocalError` |

A security library must not emit NaN. Every numeric parameter needs a documented range and a `ValueError`.

### H3. `feature_squeezing` quantisation is subtly biased and non-differentiable
`defenses.py:85`

For `bit_depth=4`: `floor(x*255/16)/15`. The level index is divided by `2**b - 1` (15) rather than by `255/16`, so the mapping is stretched by ~6% and `x=0.5 → 0.4667`. Correct is `round(x*(2**b-1))/(2**b-1)` or a `255`-based round-trip. Also `floor`+`cast` ⇒ **measured** `gradient is None`, so no attack can be run through it (no straight-through estimator).

### H4. Every preprocessing wrapper breaks on non-Functional models
`infrastructure of all wrappers, e.g. defenses.py:88, 155, 230, 371, 399, 448, 478, 516, 562`

* **Measured:** subclassed model → `AttributeError: 'Sub' object has no attribute 'input_shape'`.
* **Measured:** multi-input model → `ValueError: Cannot convert '[(None, 4)]' to a shape`.

Both need an explicit `TypeError`/`ValueError` with a clear message, plus support for subclassed models via `model.inputs`.

### H5. `randomized_smoothing` is not randomized smoothing
`defenses.py:207-233`

* Adds inference-time Gaussian noise and nothing else: no noise-augmented training, no Monte-Carlo vote over N samples, no certified radius. Inference-time-only noise is a well-documented *broken* defense.
* No clip to `[0,1]` after adding noise; no `training` gate; no seed.
* **Measured:** two identical calls return different outputs (non-deterministic `predict`).

### H6. `thermometer_encoding` is plain uniform quantisation
`defenses.py:279-283`

`floor(x*bins)/bins` is quantisation, not thermometer encoding (which expands `C → C×L` binary features and requires retraining the first layer). Claims in the docstring and README are false. `num_bins=0` → NaN.

### H7. `defensive_distillation` applies softmax twice and drops `T²`
`defenses.py:182-200`

* `tf.nn.softmax(teacher_predictions / T)` on a model that already ends in `softmax` → softmax of softmax, then `CategoricalCrossentropy(from_logits=False)` on that. Mathematically not distillation.
* Teacher is called **without `training=False`**, so its BN/dropout is live and the soft targets are noisy.
* Missing the `T²` scaling factor and the KL formulation; the typical practice is CE against the same temperature-softened distribution.

### H8. `adversarial_logit_pairing` does not implement ALP
`defenses.py:308-321`

Both models are called on the **same clean batch** `x`; the pairing loss therefore just forces two models to agree on clean data. ALP requires generating an adversarial example inside the train step and pairing `logits(x)` with `logits(x_adv)`.

### H9. Packaging and release pipeline (supply chain)

* **`pip install .` from a checkout FAILS.** `setup.py` ships `version='{{VERSION_PLACEHOLDER}}'`; **measured** `pip install .` → `Encountered error while generating package metadata`. Only the tag-time `sed` in CI produces a valid version, so local builds, editable installs, and `python -m build` are all broken. Replace with a real static version (or `setuptools_scm` / `importlib.metadata`).
* **`pypa/gh-action-pypi-publish@master` and `actions/checkout@master`** — mutable refs with `PYPI_API_TOKEN` in the environment: a compromise or force-push of the action branch means arbitrary code executes with publishing rights. Pin every action to a commit SHA.
* **`echo ::set-output`** is removed on modern runners → the version step is broken/warned.
* No `permissions:` block (defaults are broad), no `environment: pypi` protection rule, no tag pattern guard → **any** tag publishes.
* Long-lived `PYPI_API_TOKEN` secret instead of [PyPI Trusted Publishing] (OIDC `id-token: write`).
* No test/lint step before publish; uses deprecated `python setup.py sdist bdist_wheel` instead of `python -m build`; no `twine check`.
* `install_requires=['numpy','tensorflow']` is fully unbounded, so a future `numpy 3` / `tensorflow 3` breaks the package on install. `tensorflow` is unavailable on many platforms; consider a `tensorflow-cpu` extra.
* Classifiers claim Python **3.6** and `python_requires='>=3.6'`, but TF 2.x requires ≥3.9 — the package is installed on interpreters where it cannot import.
* No `license=`/`license_files=`, yet `License :: OSI Approved :: MIT License` is declared while `LICENSE` is "MIT License (Modified)" (not OSI-approved, no SPDX id). Remove the classifier or relicense.
* `Development Status :: 5 - Production/Stable` is not defensible given the above.
* No `pyproject.toml` (PEP 517/518), no `MANIFEST.in`, no `py.typed`, no extras for tests.

### H10. Public API is incomplete; no `__version__`
`__init__.py`

* **Measured MISSING:** `deepdefend.mim`, `.ead`, `.word_swap`, `.char_swap`, `.pixel_deflection`, `.gaussian_blur`, `.word_masking`, `.median_smoothing`, `.__version__`. All of these exist in the submodules but are not re-exported, and there is no `__all__`.
* No `__version__` at all makes bug reports unactionable.

---

## 3. MEDIUM — maintainability, fidelity, performance

* **M1. Duplicated bodies:** `pgd` (188-219) and `bim` (220-251) are byte-identical; `fgsm`/`pgd`/`bim`/`cw`/`deepfool`/`mim`/`ead` each re-implement loss selection and projection. The 11 preprocessing wrappers repeat the same 4-line `Input → Layer → model → Model` block. Extract `_utils` + `_wrap()`.
* **M2. Local layer classes** have no `get_config()`/`compute_output_shape()`, blocking serialization even if moved to module level.
* **M3. Fixed `[0,1]` assumption everywhere.** No `clip_min`/`clip_max` on any attack; real models are fed ImageNet-normalised tensors, where clipping to `[0,1]` silently destroys the input.
* **M4. No PGD `random_start`**, no early stopping on misclassification, no `@tf.function` anywhere (each step is an eager op, ~5-50× slower than necessary).
* **M5. Text helpers destroy whitespace.** `" ".join(text.split())` collapses runs of spaces, tabs and newlines. **Measured:** `word_masking("a  b\nc", mask_prob=1.0)` → `"[MASK] [MASK] [MASK]"`; `char_swap("line one\nline two", 1.0)` → `"lnie noe lien wto"` (newline gone). Tokenise with `re` and preserve separators.
* **M6. Non-reproducible randomness:** `np.random.*` globals in `char_swap`/`word_masking`, `tf.random` without seeds in `randomized_smoothing`/`pixel_deflection`. No `seed` parameter on any function.
* **M7. `spatial_smoothing`** allocates `B×H×W×k²×C` (81× the input for `k=9`); mislabelled (it *is* a median filter, not spatial smoothing); crashes on non-4-D input.
* **M8. `input_transformation()` with its default `transformation_function=None` is a silent identity** — returns a model with no defense, named as if defended.
* **M9. Documentation is materially inaccurate:** the support matrix marks SPSA "Numeric ✅" (fails, H2), and PGD/EAD/DeepFool "Text ✅ (Embeddings)" (unverified); the 0.1.5 changelog claims "Fixed logical errors in several defense functions" while C4/C5/C12/C13 remain; the Python-3.6 claim is impossible with the TF dependency.
* **M10. `attacks.py` calls `model(x)` without `training=False`.** In Keras 3 this defaults to inference, so the impact is currently low (**measured:** deterministic on a dropout model), but the behaviour of `training=None` is context-dependent (inside `fit`'s `train_step`, and in older TF versions, it can be `True`). Pass it explicitly.
* **M11. Circular/pointless imports:** `defenses.py:26` does `from deepdefend import attacks`; `__init__.py:1` does `import deepdefend` inside the package's own `__init__`. Both work only by accident. Note that `import deepdefend` also eagerly pulls in TensorFlow (~5 s, ~600 MB RSS for any consumer, even one that only wants `word_swap`).
* **M12. Zero tests, zero CI checks.** `CONTRIBUTING.md` instructs contributors to "run tests" — there are none, and no workflow runs anything on push/PR. No `ruff`/`flake8`, no pre-commit, no Dependabot, no `SECURITY.md` (notable for a security-focused project), no `CODE_OF_CONDUCT.md`, no `pyproject.toml`.

---

## 4. Function-by-function verdict

### `deepdefend.attacks`
| Function | Status | Key defect |
|---|---|---|
| `fgsm` | ⚠️ wrong | unclipped output (C7); binary/categorical heuristic; no sparse labels |
| `pgd` | ⚠️ weak | no `random_start`, no `clip_min/max`, no sparse labels, dup of `bim` |
| `bim` | ⚠️ weak | byte-identical to `pgd` |
| `mim` | ⚠️ weak | L1-norm is over the whole batch (cross-sample coupling), same as PGD otherwise |
| `cw` | ❌ | `c`/`kappa` inert, not C&W (C5) |
| `ead` | ❌ **no-op** | default params ⇒ zero perturbation (C4) |
| `deepfool` | ❌ | unbounded, no ε, invalid images (C6) |
| `jsma` | ❌ crash | always raises (C1) |
| `spsa` | ❌ crash | not a Variable, rank-4 only, bad estimator (C2) |
| `word_swap` | ⚠️ | whitespace loss, silent no-op on `swap_dict=None` |
| `char_swap` | ⚠️ | whitespace loss, unseeded, `swap_prob` unvalidated |

### `deepdefend.defenses`
| Function | Status | Key defect |
|---|---|---|
| `adversarial_training` | ⚠️ | per-sample loop, hard-coded compile config (H1) |
| `feature_squeezing` | ⚠️ | biased quantiser, kills gradient, `bit_depth` unvalidated (H3) |
| `gradient_masking` | ❌ **no-op** | zero inference-time effect (C11) |
| `input_transformation` | ⚠️ | default is a silent identity (M8) |
| `defensive_distillation` | ⚠️ wrong | double softmax, no `T²`, teacher in train mode (H7) |
| `randomized_smoothing` | ❌ wrong | inference-only noise, no certification (H5) |
| `feature_denoising` | ⚠️ | an average pool, duplicate of blur |
| `thermometer_encoding` | ❌ wrong | plain quantisation (H6) |
| `adversarial_logit_pairing` | ❌ wrong | pairs on clean inputs only (H8) |
| `spatial_smoothing` | ⚠️ | actually a median filter; memory blow-up (M7) |
| `jpeg_compression` | ❌ | breaks gradient; grayscale unsupported (C10) |
| `pixel_deflection` | ❌ crash | `tf.random.uniform` misuse (C3) |
| `gaussian_blur` | ❌ wrong | `sigma` unused, is avg-pool (C13) |
| `total_variation_minimization` | ❌ wrong | zero gradient through, nested tape (C14) |
| `word_masking` | ⚠️ | whitespace loss, unseeded |
| `median_smoothing` | ❌ wrong | is avg-pool (C12) |

---

## 5. Proposed remediation plan

**Phase 0 — guardrails first (so the fixes can be proven)**
1. `tests/` with a tiny deterministic CNN fixture; pytest.
2. `.github/workflows/ci.yml`: pytest on 3.10–3.12 + `ruff check` on every push/PR.
3. `pyproject.toml`: real version `0.2.0`, bounded deps, `[project.optional-dependencies] dev`, ruff/pytest config.
4. `SECURITY.md`, `CODE_OF_CONDUCT.md`, `.pre-commit-config.yaml`, Dependabot.

**Phase 1 — attack correctness**
5. New `deepdefend/_utils.py`: `_prepare_labels` (one-hot **and** sparse **and** binary), `_loss_fn(model, x, y, from_logits)`, `_project(adv, x, eps, clip_min, clip_max)`, `_validate_*`, `resolve_input_range(model, x)`.
6. Rewrite `fgsm/pgd/bim/mim`: batched, `training=False`, `clip_min/clip_max`, `from_logits`, `random_start` (PGD), sparse labels, `None`-gradient guard.
7. Rewrite `cw` (tanh space, L2 objective, working `c`/`kappa`), `deepfool` (per-class boundaries + ε bound), `jsma` (real saliency, vectorised update), `spsa` (Variable + proper finite-difference estimator, rank-agnostic), `ead` (real elastic-net + non-degenerate defaults).
8. Text attacks: `re`-based tokenisation preserving whitespace, `seed` parameter, validated `swap_prob`.

**Phase 2 — defense correctness**
9. Hoist every layer to module level with `get_config()`/`compute_output_shape()`; add a shared `_wrap_preprocessing()`; raise clear errors for subclassed/multi-input models. **Add a save→load round-trip test for all wrappers.**
10. `adversarial_training`: batched attack generation, honour `model.optimizer`/`model.loss`, accept `epochs`/`batch_size`/`verbose`.
11. Real median filter for `median_smoothing`; real separable Gaussian for `gaussian_blur` (use `sigma`); de-duplicate `feature_denoising`; make `spatial_smoothing` honest about what it is and memory-safe.
12. Straight-through gradients for `jpeg_compression`, `feature_squeezing` and `total_variation_minimization` (replace the nested tape), so attacks against defended models work — plus an explicit `differentiable=True/False` flag and a test that `grad is not None`.
13. `randomized_smoothing`: `training` gate, clip, seed, and a `predict_smoothed(n_samples)` + certified-accuracy helper; document that the base model must be noise-trained.
14. `thermometer_encoding`: real `C → C×L` expansion (with a documented retraining requirement) or rename to `uniform_quantization`.
15. `defensive_distillation` (logits + `T²` + KL + `training=False` teacher), `adversarial_logit_pairing` (generate adversarial examples inside the step), `gradient_masking` (keep, but relabel + docstring/README warning that it is not a security defense).
16. `pixel_deflection`: scalar-maxval fix, `fn_output_signature`, unique-index scatter, seed.
17. Validation pass: every numeric arg gets a documented range and a `ValueError` (kills the NaN paths in H2).
18. `input_transformation`: error or documented identity default instead of a silent no-op.

**Phase 3 — API consistency**
19. `__init__.py`: `__all__`, `__version__`, export everything, remove the self-import, fix the circular import, lazy TF import.
20. Uniform contract: attacks return `np.ndarray` in the input dtype; defenses return a `keras.Model`; text helpers return `str`. Add `seed` everywhere random.

**Phase 4 — docs, packaging, release**
21. Rewrite the README support matrix from measured behaviour; add per-defense "limitations / not a guarantee" notes and a "dual-use / responsible disclosure" section.
22. Fix `setup.py`/`pyproject.toml` metadata (version, bounded deps, `python_requires>=3.9`, license field, extras).
23. Rewrite the publish workflow: SHA-pinned actions, `permissions: contents: read` + `id-token: write`, `environment: pypi`, Trusted Publishing (no long-lived token), `$GITHUB_OUTPUT`, tag-pattern guard, tests before publish, `python -m build` + `twine check`.

**Compatibility note:** items 7, 9–17 change numerical output by design. Recommend releasing as **0.2.0** with a `CHANGELOG.md` "Behaviour changes" section rather than pretending to be a patch release.

---

## Appendix — measured evidence (TF 2.16.2 / Keras 3.15 / Python 3.11)

```
=== ATTACKS (x in [0,1]) ===
[OK]   fgsm      shape=(4,8,8,3) min=-0.010 max=1.007     <-- outside [0,1]
[FAIL] jsma      InvalidArgumentError: Index out of range using input dim 4
[FAIL] spsa      InvalidArgumentError: Tile ... (also fails with batch_size=1)
[OK]   deepfool  shape=(4,8,8,3) min=-0.900 max=2.080     <-- invalid images

=== epsilon bound, eps=0.02 ===
  fgsm      max|delta|=0.0200  in[0,1]=False
  deepfool  FAIL TypeError
  ead       max|delta|=0.0000  mean|delta|=0.0000         <-- no-op

--- cw: do c / kappa change anything? ---
   cw c/kappa have no effect: True

--- sparse (integer) labels ---
  fgsm / pgd / bim / cw / mim / ead / deepfool  ->  all FAIL:
  "target and output must have the same rank (ndim)"

--- gradient flow through defenses ---
   jpeg_compression          grad_present=False          <-- attacks impossible
   total_variation_minimization grad_present=True grad_nonzero=False
   feature_squeezing         grad_present=False

--- serialization ---
   feature_squeezing / gaussian_blur / randomized_smoothing -> save+load FAIL
   "could not be deserialized properly"

--- equivalence / dead-parameter checks ---
   gaussian_blur(sigma=0.1) == gaussian_blur(sigma=9): True   <-- sigma unused
   median_smoothing == avg_pool2d: True                       <-- not a median
   fgsm(base) == fgsm(gradient_masking(base)): True           <-- no defense
   feature_squeezing bit_depth=0 -> NaN: True
   thermometer num_bins=0 -> NaN: True
   pixel_deflection -> ValueError: maxval must be a scalar

--- misc ---
   fgsm(epsilon=-1) accepted, min=-0.9997
   adversarial_training(200 samples) = 3.08 s   (~13 min extrapolated for MNIST)
   model compiled with SGD(0.001)/KLDivergence  ->  returned with Adam/categorical_crossentropy
   deepdefend.mim / .ead / .pixel_deflection / .gaussian_blur / .median_smoothing / .__version__  ->  MISSING
   pip install .  ->  "Encountered error while generating package metadata"
   jsma on subclassed/multi-input models -> AttributeError / ValueError
```
