# DeepDefend 0.1.0
![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![Code Size](https://img.shields.io/github/languages/code-size/infinitode/deepdefend)
![Downloads](https://pepy.tech/badge/deepdefend)
![License Compliance](https://img.shields.io/badge/license-compliance-brightgreen.svg)
![PyPI Version](https://img.shields.io/pypi/v/deepdefend)

An open-source Python library for adversarial attacks and defenses in deep learning models, enhancing the security and robustness of AI systems.

## Notice

DeepDefend has not yet been fully tested. Please report any issues you may encounter when using DeepDefend.

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
- Python 3.11

Please ensure that you have one of these Python versions installed before using DeepDefend. DeepDefend may not work as expected on lower versions of Python than the supported.

## Features

- Adversarial Attacks: Generate adversarial examples to evaluate model vulnerabilities.
- Adversarial Defenses: Employ various methods to protect models against adversarial attacks.

## Usage

### Adversarial Attacks

```python
import tensorflow as tf
from deepdefend.attacks import fgsm, pgd, bim

# Load a pre-trained TensorFlow model
model = ...

# Load example input and label data (replace this with your own data loading code)
x_example = ...  # example input data
y_example = ...  # true label

# Perform FGSM attack on the example data
adversarial_example_fgsm = fgsm(model, x_example, y_example, epsilon=0.01)

# Perform PGD attack on the example data
adversarial_example_pgd = pgd(model, x_example, y_example, epsilon=0.01, alpha=0.01, num_steps=10)

# Perfrom BIM attack on the example data
adversarial_example_bim = bim(model, x_example, y_example, epsilon=0.01, alpha=0.01, num_steps=10)
```

### Adversarial Defenses

```python
import tensorflow as tf
from deepdefend.defenses import adversarial_training, feature_squeezing

# Load a pre-trained TensorFlow model
model = ...

# Load training data
x_train, y_train = ...  # training data and labels

# Adversarial training to defend against attacks
defended_model = adversarial_training(model, x_train, y_train, epsilon=0.01)

# Feature squeezing defense
defended_model_squeezed = feature_squeezing(model, bit_depth=4)
```

## Contributing

Contributions are welcome! If you encounter any issues, have suggestions, or want to contribute to DeepDefend, please open an issue or submit a pull request on [GitHub](https://github.com/infinitode/deepdefend).

## License

DeepDefend is released under the terms of the **MIT License (Modified)**. Please see the [LICENSE](https://github.com/infinitode/deepdefend/blob/main/LICENSE) file for the full text.

**Modified License Clause**



The modified license clause grants users the permission to make derivative works based on the DeepDefend software. However, it requires any substantial changes to the software to be clearly distinguished from the original work and distributed under a different name.

By enforcing this distinction, it aims to prevent direct publishing of the source code without changes while allowing users to create derivative works that incorporate the code but are not exactly the same.

Please read the full license terms in the [LICENSE](https://github.com/infinitode/deepdefend/blob/main/LICENSE) file for complete details.
