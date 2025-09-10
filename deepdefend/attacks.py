"""
Functions to run adversarial attacks on deep learning models.

Available functions:
- `fgsm(model, x, y, epsilon=0.01)`: Fast Gradient Sign Method (FGSM) attack.
- `pgd(model, x, y, epsilon=0.01, alpha=0.01, num_steps=10)`: Projected Gradient Descent (PGD) attack.
- `bim(model, x, y, epsilon=0.01, alpha=0.01, num_steps=10)`: Basic Iterative Method (BIM) attack.
- `cw(model, x, y, epsilon=0.01, c=1, kappa=0, num_steps=10, alpha=0.01)`: Carlini & Wagner (C&W) attack.
- `deepfool(model, x, y, num_steps=10)`: DeepFool attack.
- `jsma(model, x, y, theta=0.1, gamma=0.1, num_steps=10)`: Jacobian-based Saliency Map Attack (JSMA).
- `spsa(model, x, y, epsilon=0.01, num_steps=10)`: Simultaneous Perturbation Stochastic Approximation (SPSA) attack.
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

def cw(model, x, y, epsilon=0.01, c=1, kappa=0, num_steps=10, alpha=0.01):
    """
    Carlini & Wagner (C&W) attack.
    
    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example to attack.
        y (numpy.ndarray): The true labels of the input example.
        epsilon (float): The maximum magnitude of the perturbation (default: 0.01).
        c (float): The weight of the L2 norm of the perturbation (default: 1).
        kappa (float): The confidence parameter (default: 0).
        num_steps (int): The number of C&W iterations (default: 10).
        alpha (float): The step size for each iteration (default: 0.01).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example.
    """
    # Define the loss function
    def loss_function(x, y, model, c, kappa):
        prediction = model(x)
        loss = tf.keras.losses.CategoricalCrossentropy()(y, prediction)
        return loss + c * tf.norm(x - tf.clip_by_value(x, 0, 1)) ** 2 - kappa

    # Initialize the adversarial example
    adversarial_example = tf.identity(x)

    # Perform the C&W attack
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_example)
            loss = loss_function(adversarial_example, y, model, c, kappa)

        gradient = tape.gradient(loss, adversarial_example)
        perturbation = alpha * tf.sign(gradient)
        adversarial_example = tf.clip_by_value(adversarial_example + perturbation, 0, 1)
        adversarial_example = tf.clip_by_value(adversarial_example, x - epsilon, x + epsilon)

    return adversarial_example.numpy()

def deepfool(model, x, y, num_steps=10):
    """
    DeepFool attack.
    
    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example to attack.
        y (numpy.ndarray): The true labels of the input example.
        num_steps (int): The number of DeepFool iterations (default: 10).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example.
    """
    # Initialize the adversarial example
    adversarial_example = tf.identity(x)

    # Perform the DeepFool attack
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_example)
            prediction = model(adversarial_example)
            loss = tf.keras.losses.CategoricalCrossentropy()(y, prediction)

        gradient = tape.gradient(loss, adversarial_example)
        perturbation = gradient / tf.norm(gradient)
        adversarial_example = adversarial_example + perturbation

    return adversarial_example.numpy()

def jsma(model, x, y, theta=0.1, gamma=0.1, num_steps=10):
    """
    Jacobian-based Saliency Map Attack (JSMA) attack.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example to attack.
        y (numpy.ndarray): The true labels of the input example.
        theta (float): The threshold for selecting pixels (default: 0.1).
        gamma (float): The step size for each iteration (default: 0.1).
        num_steps (int): The number of JSMA iterations (default: 10).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example.
    """
    # Initialize the adversarial example
    adversarial_example = tf.identity(x)

    # Get the input shape
    input_shape = x.shape

    # Perform the JSMA attack
    for _ in range(num_steps):
        # Calculate the Jacobian matrix
        with tf.GradientTape() as tape:
            tape.watch(adversarial_example)
            prediction = model(adversarial_example)
            loss = tf.keras.losses.CategoricalCrossentropy()(y, prediction)

        jacobian = tape.jacobian(loss, adversarial_example)

        # Calculate the saliency map
        saliency_map = np.zeros(input_shape)
        for i in range(input_shape[1]):
            for j in range(input_shape[2]):
                for k in range(input_shape[3]):
                    saliency_map[0, i, j, k] = np.sum(jacobian[0, i, j, k, :])

        # Select the pixels to perturb
        perturbed_pixels = np.where(saliency_map > theta)
        
        # Perturb the selected pixels
        for i in range(len(perturbed_pixels[0])):
            if adversarial_example[0, perturbed_pixels[0][i], perturbed_pixels[1][i], perturbed_pixels[2][i]] < 1:
                adversarial_example[0, perturbed_pixels[0][i], perturbed_pixels[1][i], perturbed_pixels[2][i]] += gamma
            else:
                adversarial_example[0, perturbed_pixels[0][i], perturbed_pixels[1][i], perturbed_pixels[2][i]] -= gamma

    return adversarial_example.numpy()


def spsa(model, x, y, epsilon=0.01, num_steps=10, learning_rate=0.01, delta=0.01, spsa_samples=128):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) attack.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example to attack.
        y (numpy.ndarray): The true labels of the input example.
        epsilon (float): The magnitude of the perturbation (default: 0.01).
        num_steps (int): The number of SPSA iterations (default: 10).
        learning_rate (float): The learning rate for the ADAM optimizer (default: 0.01).
        delta (float): The perturbation size for SPSA (default: 0.01).
        spsa_samples (int): The number of samples for SPSA (default: 128).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example.
    """
    if x.shape[0] != 1:
        raise ValueError("For SPSA, input tensor x must have batch_size of 1.")

    tf_dtype = tf.as_dtype("float32")
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def loss_fn(data, label):
        logits = model(data)
        label = tf.cast(label, tf.int64)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, logits)
        return loss

    perturbation = tf.zeros_like(x, dtype=tf_dtype)

    for _ in range(num_steps):
        # SPSA gradient approximation
        x_shape = x.shape
        delta_x = delta * tf.sign(tf.random.uniform([spsa_samples // 2] + list(x_shape[1:]), minval=-1.0, maxval=1.0, dtype=tf_dtype))
        delta_x = tf.concat([delta_x, -delta_x], axis=0)

        y_tiled = tf.tile(y, [spsa_samples])

        loss_vals = loss_fn(tf.tile(x, [spsa_samples, 1, 1, 1]) + delta_x, y_tiled)
        loss_vals = tf.reshape(loss_vals, [spsa_samples] + [1] * (len(x_shape) - 1))

        grad = tf.reduce_mean(loss_vals * delta_x, axis=0, keepdims=True) / delta

        # ADAM update
        optimizer.apply_gradients([(grad[0], perturbation)])

        # Projection
        adversarial_example = x + tf.clip_by_value(perturbation, -epsilon, epsilon)
        adversarial_example = tf.clip_by_value(adversarial_example, 0, 1)
        perturbation.assign(adversarial_example - x)

    return adversarial_example.numpy()
