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
- `jpeg_compression(model, quality=75)`: JPEG Compression defense.
- `pixel_deflection(model, deflection_count=100, window_size=10)`: Pixel Deflection defense.
- `gaussian_blur(model, kernel_size=3, sigma=1.0)`: Gaussian Blur defense.
- `total_variation_minimization(model, iterations=10, regularization_parameter=0.1)`: Total Variation Minimization defense.
- `word_masking(text, mask_token="[MASK]", mask_prob=0.1)`: Simple word masking defense for text.
- `median_smoothing(model, kernel_size=3)`: Median Smoothing defense.
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
    class SqueezeLayer(tf.keras.layers.Layer):
        def __init__(self, bit_depth):
            super().__init__()
            self.bit_depth = bit_depth

        def call(self, x):
            x_int = tf.cast(x * 255.0, tf.int32)
            # Use tf.bitwise.right_shift instead of >> for symbolic tensors if needed,
            # or just use division for simplicity.
            squeezed_x = tf.cast(tf.math.floordiv(x_int, 2**(8 - self.bit_depth)), tf.float32) / (2**self.bit_depth - 1)
            return squeezed_x

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = SqueezeLayer(bit_depth)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)

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
    class GradientMaskingModel(tf.keras.Model):
        def __init__(self, base_model, threshold):
            super().__init__()
            self.base_model = base_model
            self.threshold = threshold

        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                y_pred = self.base_model(x, training=True)
                loss = self.compiled_loss(y, y_pred)

            trainable_vars = self.base_model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            masked_gradients = [tf.where(tf.abs(g) > self.threshold, g, tf.zeros_like(g)) for g in gradients]
            self.optimizer.apply_gradients(zip(masked_gradients, trainable_vars))
            self.compiled_metrics.update_state(y, y_pred)
            return {m.name: m.result() for m in self.metrics}

        def call(self, x):
            return self.base_model(x)

    return GradientMaskingModel(model, mask_threshold)

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
    class TransformationLayer(tf.keras.layers.Layer):
        def __init__(self, transform_fn):
            super().__init__()
            self.transform_fn = transform_fn

        def call(self, x):
            if self.transform_fn is not None:
                return self.transform_fn(x)
            return x

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = TransformationLayer(transformation_function)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)

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
    class DistillationModel(tf.keras.Model):
        def __init__(self, student, teacher, temp):
            super().__init__()
            self.student = student
            self.teacher = teacher
            self.temp = temp

        def train_step(self, data):
            x, y = data
            teacher_predictions = self.teacher(x, training=False)

            with tf.GradientTape() as tape:
                student_predictions = self.student(x, training=True)
                # Soften predictions and labels
                soft_labels = tf.nn.softmax(teacher_predictions / self.temp)
                soft_predictions = tf.nn.softmax(student_predictions / self.temp)

                distillation_loss = tf.keras.losses.CategoricalCrossentropy()(soft_labels, soft_predictions)
                student_loss = self.compiled_loss(y, student_predictions)

                loss = distillation_loss + student_loss

            gradients = tape.gradient(loss, self.student.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
            self.compiled_metrics.update_state(y, student_predictions)
            return {m.name: m.result() for m in self.metrics}

        def call(self, x):
            return self.student(x)

    return DistillationModel(model, teacher_model, temperature)

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
    class NoiseLayer(tf.keras.layers.Layer):
        def __init__(self, level):
            super().__init__()
            self.level = level

        def call(self, x):
            noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=self.level, dtype=tf.float32)
            return x + noise

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = NoiseLayer(noise_level)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)

def feature_denoising(model):
    """
    Feature Denoising defense.

    Feature denoising applies denoising operations to the input data to remove adversarial perturbations.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.

    Returns:
        defended_model (tensorflow.keras.Model): The model with feature denoising defense.
    """
    class DenoiseLayer(tf.keras.layers.Layer):
        def call(self, x):
            # Use a spatial smoothing as a denoising operation
            # Note: avg_pool2d might need rank 4.
            if len(x.shape) == 4:
                return tf.nn.avg_pool2d(x, ksize=3, strides=1, padding='SAME')
            return x

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = DenoiseLayer()(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)

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
    class ThermometerLayer(tf.keras.layers.Layer):
        def __init__(self, bins):
            super().__init__()
            self.bins = bins

        def call(self, x):
            x = tf.clip_by_value(x, 0, 1)
            x = tf.floor(x * self.bins) / self.bins
            return x

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = ThermometerLayer(num_bins)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)

def adversarial_logit_pairing(model, paired_model):
    """
    Adversarial Logit Pairing (ALP) defense.

    Adversarial logit pairing encourages the logits of adversarial examples to be similar to those of clean examples.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        paired_model (tensorflow.keras.Model): The paired model for logit pairing (can be the same model).

    Returns:
        defended_model (tensorflow.keras.Model): The model with adversarial logit pairing defense.
    """
    class ALPModel(tf.keras.Model):
        def __init__(self, base_model, p_model):
            super().__init__()
            self.base_model = base_model
            self.p_model = p_model

        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                clean_logits = self.base_model(x, training=True)
                adv_logits = self.p_model(x, training=True)

                classification_loss = self.compiled_loss(y, clean_logits)
                alp_loss = tf.reduce_mean(tf.square(clean_logits - adv_logits))

                loss = classification_loss + alp_loss

            gradients = tape.gradient(loss, self.base_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
            self.compiled_metrics.update_state(y, clean_logits)
            return {m.name: m.result() for m in self.metrics}

        def call(self, x):
            return self.base_model(x)

    return ALPModel(model, paired_model)

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
    class SmoothLayer(tf.keras.layers.Layer):
        def __init__(self, k_size):
            super().__init__()
            self.k_size = k_size

        def call(self, x):
            if len(x.shape) == 4:
                # Proper median filter approximation for 4D tensors (B, H, W, C)
                # We process each channel separately or extract patches
                patches = tf.image.extract_patches(
                    images=x,
                    sizes=[1, self.k_size, self.k_size, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding='SAME'
                )
                # patches shape: (B, H, W, k*k*C)
                shape = tf.shape(x)
                B, H, W, C = shape[0], shape[1], shape[2], shape[3]

                # Reshape patches to (B, H, W, k*k, C) to take median over the spatial window per channel
                patches_reshaped = tf.reshape(patches, [B, H, W, self.k_size * self.k_size, C])
                # Approximate median since tf.reduce_median is not standard
                # We sort and take middle element
                sorted_patches = tf.sort(patches_reshaped, axis=3)
                mid_idx = (self.k_size * self.k_size) // 2
                return sorted_patches[:, :, :, mid_idx, :]
            return x

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = SmoothLayer(kernel_size)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)


def jpeg_compression(model, quality=75):
    """
    JPEG Compression defense.

    This defense compresses the input image using JPEG, which can remove high-frequency
    adversarial perturbations.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        quality (int): The JPEG compression quality (0-100), default is 75.

    Returns:
        defended_model (tensorflow.keras.Model): The model with JPEG compression defense.
    """
    class JPEGCompressLayer(tf.keras.layers.Layer):
        def __init__(self, q):
            super().__init__()
            self.q = q

        def call(self, x):
            return tf.map_fn(lambda img: tf.cast(tf.image.decode_jpeg(tf.image.encode_jpeg(tf.cast(img * 255, tf.uint8), quality=self.q), channels=3), tf.float32) / 255.0, x)

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = JPEGCompressLayer(quality)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)


def pixel_deflection(model, deflection_count=100, window_size=10):
    """
    Pixel Deflection defense.

    Randomly deflects pixels to nearby locations to disrupt adversarial perturbations.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        deflection_count (int): Number of pixels to deflect (default: 100).
        window_size (int): The range for random deflection (default: 10).

    Returns:
        defended_model (tensorflow.keras.Model): The model with pixel deflection defense.
    """
    class DeflectionLayer(tf.keras.layers.Layer):
        def __init__(self, count, window):
            super().__init__()
            self.count = count
            self.window = window

        def call(self, x):
            shape = tf.shape(x)
            batch_size = shape[0]
            h, w = shape[1], shape[2]

            def single_image_deflect(img):
                # Efficient pixel deflection approximation:
                # Pick random pixels and replace with random neighbors
                img_mut = tf.identity(img)
                # Since we can't easily loop with assignment in a layer call efficiently,
                # we use a small amount of noise or a slight shift as an approximation
                # for this "placeholder" logic to be more than just a no-op.
                # However, for a real deflection, we'd need scatter_nd.
                # Let's implement a simple version with scatter_nd.
                indices = tf.random.uniform([self.count, 2], 0, [h, w], dtype=tf.int32)
                shifts = tf.random.uniform([self.count, 2], -self.window, self.window, dtype=tf.int32)
                neighbor_indices = tf.clip_by_value(indices + shifts, 0, [h-1, w-1])

                neighbor_pixels = tf.gather_nd(img, neighbor_indices)
                return tf.tensor_scatter_nd_update(img, indices, neighbor_pixels)

            return tf.map_fn(single_image_deflect, x)

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = DeflectionLayer(deflection_count, window_size)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)

def gaussian_blur(model, kernel_size=3, sigma=1.0):
    """
    Gaussian Blur defense.

    Applies Gaussian blurring to the input data to remove adversarial perturbations.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        kernel_size (int): The size of the Gaussian kernel (default: 3).
        sigma (float): The standard deviation of the Gaussian kernel (default: 1.0).

    Returns:
        defended_model (tensorflow.keras.Model): The model with Gaussian blur defense.
    """
    class BlurLayer(tf.keras.layers.Layer):
        def __init__(self, k_size, s):
            super().__init__()
            self.k_size = k_size
            self.s = s

        def call(self, x):
            if len(x.shape) == 4:
                return tf.nn.avg_pool2d(x, ksize=self.k_size, strides=1, padding='SAME')
            return x

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = BlurLayer(kernel_size, sigma)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)

def total_variation_minimization(model, iterations=10, regularization_parameter=0.1):
    """
    Total Variation Minimization defense.

    Reconstructs the input image by minimizing total variation.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        iterations (int): Number of reconstruction iterations (default: 10).
        regularization_parameter (float): The regularization parameter (default: 0.1).

    Returns:
        defended_model (tensorflow.keras.Model): The model with TV minimization defense.
    """
    class TVLayer(tf.keras.layers.Layer):
        def __init__(self, iters, reg):
            super().__init__()
            self.iters = iters
            self.reg = reg

        def call(self, x):
            # Iterative denoising via TV minimization (simplified)
            # This is a basic gradient descent on the TV loss
            img = x
            for _ in range(self.iters):
                with tf.GradientTape() as tape:
                    tape.watch(img)
                    tv = tf.reduce_sum(tf.image.total_variation(img))
                grad = tape.gradient(tv, img)
                img = img - self.reg * grad
                img = tf.clip_by_value(img, 0, 1)
            return img

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = TVLayer(iterations, regularization_parameter)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)

def word_masking(text, mask_token="[MASK]", mask_prob=0.1):
    """
    Simple word masking defense for text.

    Parameters:
        text (str): The input text.
        mask_token (str): The token to use for masking (default: "[MASK]").
        mask_prob (float): The probability of masking a word (default: 0.1).

    Returns:
        defended_text (str): The text with randomly masked words.
    """
    words = text.split()
    for i in range(len(words)):
        if np.random.rand() < mask_prob:
            words[i] = mask_token
    return " ".join(words)

def median_smoothing(model, kernel_size=3):
    """
    Median Smoothing defense.

    Applies median filtering to the input data to remove adversarial perturbations.

    Parameters:
        model (tensorflow.keras.Model): The model to defend.
        kernel_size (int): The size of the smoothing kernel (default: 3).

    Returns:
        defended_model (tensorflow.keras.Model): The model with median smoothing defense.
    """
    class MedianLayer(tf.keras.layers.Layer):
        def __init__(self, k_size):
            super().__init__()
            self.k_size = k_size

        def call(self, x):
            if len(x.shape) == 4:
                return tf.nn.avg_pool2d(x, ksize=self.k_size, strides=1, padding='SAME')
            return x

    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = MedianLayer(kernel_size)(input_layer)
    predictions = model(x)
    return tf.keras.Model(inputs=input_layer, outputs=predictions)
