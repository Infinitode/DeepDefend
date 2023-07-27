"""
Functions to apply adversarial defense mechanisms to deep learning models.

Available functions:
- `adversarial_training(model, x, y, epsilon=0.01)`: Adversarial Training defense.
- `feature_squeezing(model, bit_depth=4)`: Feature Squeezing defense.

"""

import numpy as np
import tensorflow as tf
from deepdefend import attacks

def adversarial_training(model, x, y, epsilon=0.01):
    """
    Adversarial Training defense.

    Adversarial training is a method where the model is trained on both the original
    and adversarial examples, aiming to make the model more robust to adversarial attacks.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        x (numpy.ndarray): The input training examples.
        y (numpy.ndarray): The true labels of the training examples.
        epsilon (float): The magnitude of the perturbation (default: 0.01).

    Returns:
        defended_model (tensorflow.keras.Model): The adversarially trained model.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    adversarial_examples = []
    for i in range(len(x)):
        adversarial_example = attacks.fgsm(model, x[i:i+1], y[i:i+1], epsilon)
        adversarial_examples.append(adversarial_example)
        
    x_adversarial = np.concatenate(adversarial_examples, axis=0)
    y_adversarial = np.copy(y)
    x_train = np.concatenate([x, x_adversarial], axis=0)
    y_train = np.concatenate([y, y_adversarial], axis=0)
    
    defended_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    defended_model.fit(x_train, y_train, epochs=10, batch_size=32)
    
    return defended_model

def feature_squeezing(model, bit_depth=4):
    """
    Feature Squeezing defense.

    Feature squeezing reduces the number of bits used to represent the input features,
    which can remove certain adversarial perturbations.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        bit_depth (int): The number of bits per feature (default: 4).

    Returns:
        defended_model (tensorflow.keras.Model): The model with feature squeezing defense.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    for layer in defended_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer_weights = layer.get_weights()
            squeezed_weights = [np.clip(np.round(w * (2**bit_depth) / np.max(np.abs(w))), -2**(bit_depth - 1), 2**(bit_depth - 1) - 1) / (2**(bit_depth) / np.max(np.abs(w))) for w in layer_weights]
            layer.set_weights(squeezed_weights)
    
    return defended_model