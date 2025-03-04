# defenses.py
"""
Functions to apply adversarial defense mechanisms to deep learning models.

Available functions:
- `adversarial_training(model, x, y, epsilon=0.01)`: Adversarial Training defense.
- `feature_squeezing(model, bit_depth=4)`: Feature Squeezing defense.
- `gradient_masking(model, mask_threshold=0.1)`: Gradient Masking defense.
- `input_transformation(model, transformation_function=None)`: Input Transformation defense.
- `defensive_distillation(model, teacher_model, temperature=2)`: Defensive Distillation defense.
- `randomized_smoothing(model, noise_level=0.1)`: Randomized Smoothing defense.
- `feature_denoising(model)`: Feature Denoising defense.
- `thermometer_encoding(model, num_bins=10)`: Thermometer Encoding defense.
- `adversarial_logit_pairing(model, paired_model)`: Adversarial Logit Pairing defense.
- `spatial_smoothing(model, kernel_size=3)`: Spatial Smoothing defense.
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

def gradient_masking(model, mask_threshold=0.1):
    """
    Gradient Masking defense.

    Gradient masking modifies the gradients during training to make them less informative
    for adversarial attackers.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        mask_threshold (float): The threshold for masking gradients (default: 0.1).

    Returns:
        defended_model (tensorflow.keras.Model): The model with gradient masking defense.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    def masked_loss(y_true, y_pred):
        loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        gradients = tf.gradients(loss, defended_model.trainable_variables)
        masked_gradients = [tf.where(tf.abs(g) > mask_threshold, g, tf.zeros_like(g)) for g in gradients]
        return loss, masked_gradients

    defended_model.compile(optimizer='adam', loss=masked_loss, metrics=['accuracy'])
    return defended_model

def input_transformation(model, transformation_function=None):
    """
    Input Transformation defense.

    Input transformation applies a transformation to the input data before feeding it
    to the model, aiming to remove adversarial perturbations.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        transformation_function (function): The transformation function to apply (default: None).

    Returns:
        defended_model (tensorflow.keras.Model): The model with input transformation defense.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    def transformed_input(x):
        if transformation_function is not None:
            return transformation_function(x)
        else:
            return x

    defended_model.layers[0].input = tf.keras.Input(shape=model.input_shape[1:])
    defended_model.layers[0].input = transformed_input(defended_model.layers[0].input)
    return defended_model

def defensive_distillation(model, teacher_model, temperature=2):
    """
    Defensive Distillation defense.

    Defensive distillation trains a student model to mimic the predictions of a
    teacher model, which is often a more robust model.

    Parameters:
        model (tensorflow.keras.Model): The student model to defend.
        teacher_model (tensorflow.keras.Model): The teacher model.
        temperature (float): The temperature parameter for distillation (default: 2).

    Returns:
        defended_model (tensorflow.keras.Model): The distilled student model.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    def distilled_loss(y_true, y_pred):
        teacher_predictions = teacher_model(y_true)
        return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred) + temperature**2 * tf.keras.losses.CategoricalCrossentropy()(teacher_predictions, y_pred)

    defended_model.compile(optimizer='adam', loss=distilled_loss, metrics=['accuracy'])
    return defended_model

def randomized_smoothing(model, noise_level=0.1):
    """
    Randomized Smoothing defense.

    Randomized smoothing adds random noise to the input data to make the model more robust
    to adversarial attacks.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        noise_level (float): The standard deviation of the Gaussian noise (default: 0.1).

    Returns:
        defended_model (tensorflow.keras.Model): The model with randomized smoothing defense.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    def add_noise(x):
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_level, dtype=tf.float32)
        return x + noise

    defended_model.layers[0].input = tf.keras.Input(shape=model.input_shape[1:])
    defended_model.layers[0].input = add_noise(defended_model.layers[0].input)
    return defended_model

def feature_denoising(model):
    """
    Feature Denoising defense.

    Feature denoising applies denoising operations to the input data to remove adversarial perturbations.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.

    Returns:
        defended_model (tensorflow.keras.Model): The model with feature denoising defense.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    def denoise(x):
        return tf.image.total_variation(x)

    defended_model.layers[0].input = tf.keras.Input(shape=model.input_shape[1:])
    defended_model.layers[0].input = denoise(defended_model.layers[0].input)
    return defended_model

def thermometer_encoding(model, num_bins=10):
    """
    Thermometer Encoding defense.

    Thermometer encoding discretizes the input features into bins, making it harder for adversarial perturbations
    to affect the model.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        num_bins (int): The number of bins for encoding (default: 10).

    Returns:
        defended_model (tensorflow.keras.Model): The model with thermometer encoding defense.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    def encode(x):
        x = tf.clip_by_value(x, 0, 1)
        x = tf.floor(x * num_bins) / num_bins
        return x

    defended_model.layers[0].input = tf.keras.Input(shape=model.input_shape[1:])
    defended_model.layers[0].input = encode(defended_model.layers[0].input)
    return defended_model

def adversarial_logit_pairing(model, paired_model):
    """
    Adversarial Logit Pairing (ALP) defense.

    Adversarial logit pairing encourages the logits of adversarial examples to be similar to those of clean examples.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        paired_model (tensorflow.keras.Model): The paired model for logit pairing.

    Returns:
        defended_model (tensorflow.keras.Model): The model with adversarial logit pairing defense.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    def alp_loss(y_true, y_pred):
        clean_logits = model(y_true)
        adv_logits = paired_model(y_true)
        return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred) + tf.reduce_mean(tf.square(clean_logits - adv_logits))

    defended_model.compile(optimizer='adam', loss=alp_loss, metrics=['accuracy'])
    return defended_model

def spatial_smoothing(model, kernel_size=3):
    """
    Spatial Smoothing defense.

    Spatial smoothing applies a smoothing filter to the input data to remove adversarial perturbations.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        kernel_size (int): The size of the smoothing kernel (default: 3).

    Returns:
        defended_model (tensorflow.keras.Model): The model with spatial smoothing defense.
    """
    defended_model = tf.keras.models.clone_model(model)
    defended_model.set_weights(model.get_weights())

    def smooth(x):
        return tf.nn.avg_pool2d(x, ksize=kernel_size, strides=1, padding='SAME')

    defended_model.layers[0].input = tf.keras.Input(shape=model.input_shape[1:])
    defended_model.layers[0].input = smooth(defended_model.layers[0].input)
    return defended_model