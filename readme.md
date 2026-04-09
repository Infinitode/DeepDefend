# DeepDefend
![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)
![Code Size](https://img.shields.io/github/languages/code-size/infinitode/deepdefend)
![Downloads](https://pepy.tech/badge/deepdefend)
![License Compliance](https://img.shields.io/badge/license-compliance-brightgreen.svg)
![PyPI Version](https://img.shields.io/pypi/v/deepdefend)

[Documentation](https://infinitode-docs.gitbook.io/documentation/package-documentation/deepdefend-package-documentation)

An open-source Python library for adversarial attacks and defenses in deep learning models, enhancing the security and robustness of AI systems.

## Changes in 0.1.5:
- Added MIM (Momentum Iterative Method) and EAD (Elastic Net Attack) attacks.
- Added Word Swap and Character Swap attacks for text-based models.
- Added Pixel Deflection, Gaussian Blur, Total Variation Minimization, and Median Smoothing defenses.
- Added Word Masking defense for text-based models.
- Added a comprehensive support table for different model types.
- Fixed logical errors in several defense functions.
- Improved Keras compatibility for training-time defenses.

## Changes in 0.1.4:
- Added SPSA (Simultaneous Perturbation Stochastic Approximation) attack.
- Added JPEG Compression defense.

## Changes in 0.1.3:
- 5 new functions in `defenses.py`, including: Randomized Smoothing, Feature Denoising, Thermometer Encoding, Adversarial Logit Pairing (ALP), and Spatial Smoothing.

> [!NOTE]
> Updates to DeepDefend's attack module will be less frequent due to the potential misuse of adversarial attacks on AI models.

## Changes in 0.1.2:
We've updated DeepDefend, here's what's new:
- 3 new functions under `deepdefend.attacks`
- 3 new functions under `deepdefend.defenses`

## Installation

You can install DeepDefend using pip:

```bash
pip install deepdefend
```

## Supported Python Versions

DeepDefend supports the following Python versions:

- Python 3.6
- Python 3.7
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11 or later

Please ensure that you have one of these Python versions installed before using DeepDefend. DeepDefend may not work as expected on lower versions of Python than the supported.

## Features

- Adversarial Attacks: Generate adversarial examples to evaluate model vulnerabilities.
- Adversarial Defenses: Employ various methods to protect models against adversarial attacks.

## Supported Model Types

| Feature | Image | Text | Numeric | Classification |
|---------|:-----:|:----:|:-------:|:--------------:|
| **Attacks** | | | | |
| FGSM | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| PGD | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| BIM | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| CW | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| DeepFool | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| JSMA | ✅ | ❌ | ❌ | ✅ |
| SPSA | ✅ | ❌ | ✅ | ✅ |
| MIM | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| EAD | ✅ | ✅ (Embeddings) | ✅ | ✅ |
| Word Swap | ❌ | ✅ | ❌ | ✅ |
| Char Swap | ❌ | ✅ | ❌ | ✅ |
| **Defenses** | | | | |
| Adversarial Training | ✅ | ✅ | ✅ | ✅ |
| Feature Squeezing | ✅ | ❌ | ✅ | ✅ |
| Gradient Masking | ✅ | ✅ | ✅ | ✅ |
| Input Transformation | ✅ | ✅ | ✅ | ✅ |
| Defensive Distillation| ✅ | ✅ | ✅ | ✅ |
| Randomized Smoothing | ✅ | ❌ | ✅ | ✅ |
| Feature Denoising | ✅ | ❌ | ❌ | ✅ |
| Thermometer Encoding | ✅ | ❌ | ✅ | ✅ |
| ALP | ✅ | ✅ | ✅ | ✅ |
| Spatial Smoothing | ✅ | ❌ | ❌ | ✅ |
| JPEG Compression | ✅ | ❌ | ❌ | ✅ |
| Pixel Deflection | ✅ | ❌ | ❌ | ✅ |
| Gaussian Blur | ✅ | ❌ | ❌ | ✅ |
| TV Minimization | ✅ | ❌ | ❌ | ✅ |
| Word Masking | ❌ | ✅ | ❌ | ✅ |
| Median Smoothing | ✅ | ❌ | ❌ | ✅ |

## Usage

### Adversarial Attacks

```python
import tensorflow as tf
from deepdefend.attacks import fgsm, pgd, bim, cw, deepfool, jsma

# Load a pre-trained TensorFlow model
model = ...

# Load example input and label data (replace this with your own data loading code)
x_example = ...  # example input data
y_example = ...  # true label

# Perform FGSM attack on the example data
adversarial_example_fgsm = fgsm(model, x_example, y_example, epsilon=0.01)

# Perform PGD attack on the example data
adversarial_example_pgd = pgd(model, x_example, y_example, epsilon=0.01, alpha=0.01, num_steps=10)

# Perform BIM attack on the example data
adversarial_example_bim = bim(model, x_example, y_example, epsilon=0.01, alpha=0.01, num_steps=10)

# Perform CW attack on the example data
adversarial_example_cw = cw(model, x_example, y_example, epsilon=0.01, c=1, kappa=0, num_steps=10, alpha=0.01)

# Perform Deepfool attack on the example data
adversarial_example_deepfool = deepfool(model, x_example, y_example, num_steps=10)

# Perform JSMA attack on the example data
adversarial_example_jsma = jsma(model, x_example, y_example, theta=0.1, gamma=0.1, num_steps=10)

# Perform SPSA attack on the example data
adversarial_example_spsa = spsa(model, x_example, y_example, epsilon=0.01, num_steps=10)

# Perform MIM attack on the example data
adversarial_example_mim = mim(model, x_example, y_example, epsilon=0.01, alpha=0.01, num_steps=10)

# Perform EAD attack on the example data
adversarial_example_ead = ead(model, x_example, y_example, epsilon=0.01, beta=0.01, num_steps=10)

# Perform Word Swap attack on text data
text_data = "The movie was great"
swaps = {"great": "terrible"}
perturbed_text = word_swap(text_data, swap_dict=swaps)

# Perform Character Swap attack on text data
perturbed_text_char = char_swap(text_data, swap_prob=0.1)
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
x_train, y_train = ...  # training data and labels

# Adversarial training to defend against attacks
defended_model = adversarial_training(model, x_train, y_train, epsilon=0.01)

# Feature squeezing defense
defended_model_squeezed = feature_squeezing(model, bit_depth=4)

# Gradient masking defense
defended_model_masking = gradient_masking(model, mask_threshold=0.1)

# Input transformation defense
defended_model_transformation = input_transformation(model, transformation_function=None)

# Defensive distillation defense
defended_model_distillation = defensive_distillation(model, teacher_model, temperature=2)

# JPEG compression defense
defended_model_jpeg = jpeg_compression(model, quality=75)

# Randomized smoothing defense
defended_model_smoothing = randomized_smoothing(model, noise_level=0.1)

# Feature denoising defense
defended_model_denoising = feature_denoising(model)

# Thermometer encoding defense
defended_model_thermometer = thermometer_encoding(model, num_bins=10)

# Adversarial Logit Pairing (ALP) defense
defended_model_alp = adversarial_logit_pairing(model, paired_model=model)

# Spatial smoothing defense
defended_model_spatial = spatial_smoothing(model, kernel_size=3)

# Pixel deflection defense
defended_model_deflection = pixel_deflection(model, deflection_count=100, window_size=10)

# Gaussian blur defense
defended_model_blur = gaussian_blur(model, kernel_size=3, sigma=1.0)

# TV Minimization defense
defended_model_tv = total_variation_minimization(model, iterations=10)

# Median smoothing defense
defended_model_median = median_smoothing(model, kernel_size=3)

# Word masking defense for text
text_data = "The movie was great"
defended_text = word_masking(text_data, mask_prob=0.2)
```

## Contributing

Contributions are welcome! If you encounter any issues, have suggestions, or want to contribute to DeepDefend, please open an issue or submit a pull request on [GitHub](https://github.com/infinitode/deepdefend).

## License

DeepDefend is released under the terms of the **MIT License (Modified)**. Please see the [LICENSE](https://github.com/infinitode/deepdefend/blob/master/LICENSE) file for the full text.

**Modified License Clause**



The modified license clause grants users the permission to make derivative works based on the DeepDefend software. However, it requires any substantial changes to the software to be clearly distinguished from the original work and distributed under a different name.

By enforcing this distinction, it aims to prevent direct publishing of the source code without changes while allowing users to create derivative works that incorporate the code but are not exactly the same.

Please read the full license terms in the [LICENSE](https://github.com/infinitode/deepdefend/blob/master/LICENSE) file for complete details.
