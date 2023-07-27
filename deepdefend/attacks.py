"""
Functions to run adversarial attacks on deep learning models.

Available functions:
- `fgsm(model, x, y, epsilon=0.01)`: Fast Gradient Sign Method (FGSM) attack.
- `pgd(model, x, y, epsilon=0.01, alpha=0.01, num_steps=10)`: Projected Gradient Descent (PGD) attack.
- `bim(model, x, y, epsilon=0.01, alpha=0.01, num_steps=10)`: Basic Iterative Method (BIM) attack.

"""

import numpy as np
import tensorflow as tf

def fgsm(model, x, y, epsilon=0.01):
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example to attack.
        y (numpy.ndarray): The true labels of the input example.
        epsilon (float): The magnitude of the perturbation (default: 0.01).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example.
    """
    # Determine the loss function based on the number of classes
    if y.shape[-1] == 1 or len(y.shape) == 1:
        loss_object = tf.keras.losses.BinaryCrossentropy()
    else:
        loss_object = tf.keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = loss_object(y, prediction)
        
    gradient = tape.gradient(loss, x)

    # Generate adversarial example
    perturbation = epsilon * tf.sign(gradient)
    adversarial_example = x + perturbation
    return adversarial_example.numpy()

def pgd(model, x, y, epsilon=0.01, alpha=0.01, num_steps=10):
    """
    Projected Gradient Descent (PGD) attack.
    
    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example to attack.
        y (numpy.ndarray): The true labels of the input example.
        epsilon (float): The maximum magnitude of the perturbation (default: 0.01).
        alpha (float): The step size for each iteration (default: 0.01).
        num_steps (int): The number of PGD iterations (default: 10).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example.
    """
    adversarial_example = tf.identity(x)

    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_example)
            prediction = model(adversarial_example)
            loss = tf.keras.losses.CategoricalCrossentropy()(y, prediction)

        gradient = tape.gradient(loss, adversarial_example)
        perturbation = alpha * tf.sign(gradient)
        adversarial_example = tf.clip_by_value(adversarial_example + perturbation, 0, 1)
        adversarial_example = tf.clip_by_value(adversarial_example, x - epsilon, x + epsilon)

    return adversarial_example.numpy()

def bim(model, x, y, epsilon=0.01, alpha=0.01, num_steps=10):
    """
    Basic Iterative Method (BIM) attack.
    
    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example to attack.
        y (numpy.ndarray): The true labels of the input example.
        epsilon (float): The maximum magnitude of the perturbation (default: 0.01).
        alpha (float): The step size for each iteration (default: 0.01).
        num_steps (int): The number of BIM iterations (default: 10).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example.
    """
    adversarial_example = tf.identity(x)

    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_example)
            prediction = model(adversarial_example)
            loss = tf.keras.losses.CategoricalCrossentropy()(y, prediction)

        gradient = tape.gradient(loss, adversarial_example)
        perturbation = alpha * tf.sign(gradient)
        adversarial_example = tf.clip_by_value(adversarial_example + perturbation, 0, 1)
        adversarial_example = tf.clip_by_value(adversarial_example, x - epsilon, x + epsilon)

    return adversarial_example.numpy()