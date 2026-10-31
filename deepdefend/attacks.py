"""
Functions to run adversarial attacks on deep learning models.

Every attack accepts:

* ``x`` – a batch of inputs (NumPy array or TensorFlow tensor);
* ``y`` – labels as one-hot vectors ``(N, C)``, sparse integer indices ``(N,)``
  or binary targets ``(N, 1)``;
* ``from_logits`` – ``None`` (auto-detect from the last layer), ``True`` or
  ``False``;
* ``clip_min`` / ``clip_max`` – the valid data range.  When omitted, ``[0, 1]``
  is used if the data already lies in that range, otherwise no clipping is
  applied (so normalised inputs are not silently destroyed).

All attacks return a NumPy array with the same shape and dtype as ``x``.

Iterative attacks stop working on a sample as soon as it reaches the goal
(misclassified for untargeted attacks, classified as ``y`` for targeted ones),
so a sample the model already gets wrong is returned unchanged.

Available functions:
- `fgsm(model, x, y, epsilon=0.01)`: Fast Gradient Sign Method.
- `pgd(model, x, y, epsilon=0.01, alpha=None, num_steps=10)`: Projected Gradient Descent.
- `bim(model, x, y, epsilon=0.01, alpha=None, num_steps=10)`: Basic Iterative Method.
- `mim(model, x, y, epsilon=0.01, alpha=None, num_steps=10, decay_factor=1.0)`: Momentum Iterative Method.
- `cw(model, x, y, epsilon=0.01, c=1.0, kappa=0.0, num_steps=100, alpha=0.01)`: Carlini & Wagner L2.
- `deepfool(model, x, y, num_steps=50, epsilon=None)`: DeepFool.
- `jsma(model, x, y, theta=0.0, gamma=0.2, num_steps=10)`: Jacobian-based Saliency Map Attack.
- `spsa(model, x, y, epsilon=0.01, num_steps=50)`: Simultaneous Perturbation Stochastic Approximation.
- `ead(model, x, y, epsilon=0.01, beta=0.001, num_steps=50, alpha=0.01)`: Elastic Net Attack.
- `word_swap(text, swap_dict=None)`: word substitution attack for text.
- `char_swap(text, swap_prob=0.1)`: adjacent-character swap attack for text.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from ._utils import (
    as_numpy,
    call_model,
    check_float,
    check_int,
    prepare_attack_context,
    project,
    tokenize_text,
)

__all__ = [
    "bim",
    "char_swap",
    "cw",
    "deepfool",
    "ead",
    "fgsm",
    "jsma",
    "mim",
    "pgd",
    "spsa",
    "word_swap",
]


# ---------------------------------------------------------------------------
# Shared internals
# ---------------------------------------------------------------------------


def _gradient(model, adv, ctx):
    """Return ``(gradient, outputs)`` of the attack loss w.r.t. ``adv``."""
    with tf.GradientTape() as tape:
        tape.watch(adv)
        outputs = call_model(model, adv)
        loss = ctx.loss(outputs)
    grad = tape.gradient(loss, adv)
    if grad is None:
        raise RuntimeError(
            "The gradient of the loss with respect to the input is None. "
            "This usually means the model contains a non-differentiable or "
            "disconnected preprocessing step."
        )
    return grad, outputs


def _predicted(outputs, ctx):
    """Predicted class indices for ``outputs`` (shape ``(N,)``)."""
    return tf.cast(tf.argmax(_probabilities(outputs, ctx), axis=-1), tf.int32)


def _keep_mask(outputs, ctx, x, y_idx=None, targeted=False):
    """Mask of samples that still need attacking, broadcastable to ``x``.

    For untargeted attacks a sample is done once it is misclassified; for
    targeted attacks it is done once it is classified as the target.  ``y_idx``
    must be supplied when ``outputs`` covers only a slice of the batch
    (otherwise the full-batch labels would broadcast and duplicate rows).
    """
    if y_idx is None:
        y_idx = ctx.y_idx
    predicted = _predicted(outputs, ctx)
    if targeted:
        still_needed = tf.not_equal(predicted, y_idx)
    else:
        still_needed = tf.equal(predicted, y_idx)
    return tf.reshape(still_needed, [-1] + [1] * (x.shape.rank - 1))


def _probabilities(outputs, ctx):
    """Convert model outputs to per-class probabilities."""
    if ctx.mode == "binary":
        p = tf.reshape(outputs, (-1, 1))
        p = tf.sigmoid(p) if ctx.from_logits else p
        return tf.concat([1.0 - p, p], axis=1)
    return tf.nn.softmax(outputs) if ctx.from_logits else outputs


def _margin(outputs, ctx, kappa=0.0, target=None):
    """C&W style margin ``max(max_{j != t} Z_j - Z_t, -kappa)`` per sample.

    ``t`` is passed in explicitly so it stays fixed for the whole optimisation —
    recomputing it every step makes the attack chase a moving target.
    """
    if ctx.mode == "binary":
        outputs = _probabilities(outputs, ctx)

    if target is None:
        target = tf.cast(tf.argmin(outputs, axis=-1), tf.int32)

    batch = tf.shape(outputs)[0]
    rows = tf.range(batch)
    z_t = tf.gather_nd(outputs, tf.stack([rows, target], axis=1))

    mask = tf.one_hot(target, tf.shape(outputs)[1], dtype=outputs.dtype)
    very_negative = tf.fill(tf.shape(outputs), tf.cast(-1e9, outputs.dtype))
    others = tf.where(tf.cast(mask, tf.bool), very_negative, outputs)
    z_other = tf.reduce_max(others, axis=-1)
    return tf.maximum(z_other - z_t, -kappa)


def _resolve_target(model, ctx, targeted):
    """Target class used by the margin based attacks.

    For targeted attacks it is ``y``.  For untargeted attacks it is the
    runner-up class of the *original* input: the cheapest label flip, and it is
    fixed for the whole optimisation (recomputing it every step makes the attack
    chase a moving target).
    """
    if targeted:
        return ctx.y_idx
    outputs = call_model(model, ctx.x)
    if ctx.mode == "binary":
        outputs = _probabilities(outputs, ctx)
    if outputs.shape[-1] is not None and int(outputs.shape[-1]) < 2:
        return tf.cast(tf.argmin(outputs, axis=-1), tf.int32)
    order = tf.argsort(outputs, axis=-1, direction="DESCENDING")
    return tf.cast(order[:, 1], tf.int32)


def _to_range(w, lo, hi):
    """Map unconstrained ``w`` into ``[lo, hi]`` (C&W tanh parameterisation)."""
    return lo + (hi - lo) * 0.5 * (tf.tanh(w) + 1.0)


def _from_range(adv, lo, hi):
    """Inverse of :func:`_to_range`, numerically clamped away from the bounds."""
    scaled = (adv - lo) / (hi - lo) * 2.0 - 1.0
    scaled = tf.clip_by_value(scaled, -1.0 + 1e-6, 1.0 - 1e-6)
    return tf.atanh(scaled)


def _project_against(adv, reference, ctx, epsilon=None):
    """Project ``adv`` around ``reference`` (a slice of ``ctx.x``).

    ``ctx.project`` projects around the whole batch, which is wrong inside the
    per-sample loops used by DeepFool, JSMA and SPSA.
    """
    return project(adv, reference, epsilon=epsilon, clip_min=ctx.clip_min, clip_max=ctx.clip_max)


def _adam_step(param, grad, m, v, t, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
    """One hand-rolled Adam update (avoids creating an optimizer per call)."""
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * tf.square(grad)
    m_hat = m / (1.0 - beta1 ** (t + 1))
    v_hat = v / (1.0 - beta2 ** (t + 1))
    return param - learning_rate * m_hat / (tf.sqrt(v_hat) + eps), m, v


def _prepare(x, model, y, from_logits, clip_min, clip_max):
    ctx = prepare_attack_context(model, x, y, from_logits, clip_min, clip_max)
    return ctx, ctx.x.dtype


# ---------------------------------------------------------------------------
# One-step and iterative gradient attacks
# ---------------------------------------------------------------------------


def fgsm(model, x, y, epsilon=0.01, *, targeted=False, from_logits=None,
         clip_min=None, clip_max=None):
    """
    Fast Gradient Sign Method (FGSM) attack.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example(s) to attack.
        y (numpy.ndarray): True labels (one-hot, sparse indices or binary).
        epsilon (float): The magnitude of the perturbation (default: 0.01).
        targeted (bool): Minimise the loss for ``y`` instead of maximising it.
        from_logits (bool): Whether ``model`` returns logits. Auto-detected if None.
        clip_min, clip_max (float): Valid data range (inferred if omitted).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example(s).
    """
    epsilon = check_float("epsilon", epsilon, minimum=0.0)
    ctx, _ = _prepare(x, model, y, from_logits, clip_min, clip_max)
    direction = -1.0 if targeted else 1.0
    grad, _ = _gradient(model, ctx.x, ctx)
    adv = ctx.x + direction * epsilon * tf.sign(grad)
    return as_numpy(ctx.project(adv))


def _iterative_gradient_attack(model, x, y, epsilon, alpha, num_steps, *,
                               random_start, seed, targeted, from_logits,
                               clip_min, clip_max, momentum=None, decay_factor=1.0):
    """Shared implementation of PGD / BIM / MIM."""
    epsilon = check_float("epsilon", epsilon, minimum=0.0)
    num_steps = check_int("num_steps", num_steps, minimum=1)
    if alpha is None:
        alpha = epsilon / 4.0 if epsilon > 0 else 1e-3
    alpha = check_float("alpha", alpha, minimum=0.0)
    decay_factor = check_float("decay_factor", decay_factor, minimum=0.0)

    ctx, dtype = _prepare(x, model, y, from_logits, clip_min, clip_max)
    direction = -1.0 if targeted else 1.0

    adv = ctx.x
    if random_start and epsilon > 0:
        noise = tf.random.uniform(tf.shape(ctx.x), -epsilon, epsilon, seed=seed, dtype=dtype)
        adv = ctx.project(ctx.x + noise, epsilon)

    if momentum is not None:
        momentum = tf.zeros_like(ctx.x)

    for _ in range(num_steps):
        grad, outputs = _gradient(model, adv, ctx)
        if momentum is not None:
            axes = list(range(1, ctx.x.shape.rank))
            grad = grad / (tf.reduce_mean(tf.abs(grad), axis=axes, keepdims=True) + 1e-12)
            momentum = decay_factor * momentum + grad
            step = tf.sign(momentum)
        else:
            step = tf.sign(grad)

        previous = adv
        adv = previous + direction * alpha * step
        adv = ctx.project(adv, epsilon)

        # Freeze samples that have already reached the goal.
        keep = _keep_mask(outputs, ctx, ctx.x, targeted=targeted)
        adv = tf.where(keep, adv, previous)

    return adv


def pgd(model, x, y, epsilon=0.01, alpha=None, num_steps=10, *, random_start=True,
        seed=None, targeted=False, from_logits=None, clip_min=None, clip_max=None):
    """
    Projected Gradient Descent (PGD) attack.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example(s) to attack.
        y (numpy.ndarray): True labels (one-hot, sparse indices or binary).
        epsilon (float): The L-infinity perturbation budget (default: 0.01).
        alpha (float): Step size. Defaults to ``epsilon / 4``.
        num_steps (int): Number of iterations (default: 10).
        random_start (bool): Start from a random point in the epsilon ball.
        seed (int): Optional random seed for the random start.
        targeted (bool): Whether ``y`` is the target class.
        from_logits (bool): Whether ``model`` returns logits. Auto-detected if None.
        clip_min, clip_max (float): Valid data range (inferred if omitted).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example(s).
    """
    return as_numpy(_iterative_gradient_attack(
        model, x, y, epsilon, alpha, num_steps, random_start=random_start,
        seed=seed, targeted=targeted, from_logits=from_logits,
        clip_min=clip_min, clip_max=clip_max,
    ))


def bim(model, x, y, epsilon=0.01, alpha=None, num_steps=10, *, seed=None,
        targeted=False, from_logits=None, clip_min=None, clip_max=None):
    """
    Basic Iterative Method (BIM) attack — PGD without a random start.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example(s) to attack.
        y (numpy.ndarray): True labels (one-hot, sparse indices or binary).
        epsilon (float): The L-infinity perturbation budget (default: 0.01).
        alpha (float): Step size. Defaults to ``epsilon / 4``.
        num_steps (int): Number of iterations (default: 10).
        seed (int): Optional random seed (unused, kept for API symmetry).
        targeted (bool): Whether ``y`` is the target class.
        from_logits (bool): Whether ``model`` returns logits. Auto-detected if None.
        clip_min, clip_max (float): Valid data range (inferred if omitted).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example(s).
    """
    return as_numpy(_iterative_gradient_attack(
        model, x, y, epsilon, alpha, num_steps, random_start=False, seed=seed,
        targeted=targeted, from_logits=from_logits, clip_min=clip_min,
        clip_max=clip_max,
    ))


def mim(model, x, y, epsilon=0.01, alpha=None, num_steps=10, decay_factor=1.0, *,
        seed=None, targeted=False, from_logits=None, clip_min=None, clip_max=None):
    """
    Momentum Iterative Method (MIM) attack.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example(s) to attack.
        y (numpy.ndarray): True labels (one-hot, sparse indices or binary).
        epsilon (float): The L-infinity perturbation budget (default: 0.01).
        alpha (float): Step size. Defaults to ``epsilon / 4``.
        num_steps (int): Number of iterations (default: 10).
        decay_factor (float): Momentum decay factor (default: 1.0).
        seed (int): Optional random seed.
        targeted (bool): Whether ``y`` is the target class.
        from_logits (bool): Whether ``model`` returns logits. Auto-detected if None.
        clip_min, clip_max (float): Valid data range (inferred if omitted).

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example(s).
    """
    return as_numpy(_iterative_gradient_attack(
        model, x, y, epsilon, alpha, num_steps, random_start=False, seed=seed,
        targeted=targeted, from_logits=from_logits, clip_min=clip_min,
        clip_max=clip_max, momentum=True, decay_factor=decay_factor,
    ))


# ---------------------------------------------------------------------------
# Optimisation based attacks
# ---------------------------------------------------------------------------


def cw(model, x, y, epsilon=None, c=10.0, kappa=0.0, num_steps=100, alpha=0.01, *,
       targeted=False, from_logits=None, clip_min=None, clip_max=None):
    """
    Carlini & Wagner L2 attack.

    The perturbation is optimised in the tanh space and the objective is the
    true C&W objective ``||delta||_2^2 + c * max(max_{j != t} Z_j - Z_t, -kappa)``.
    Unlike the previous implementation, ``c`` and ``kappa`` genuinely change the
    result.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example(s) to attack.
        y (numpy.ndarray): True labels (one-hot, sparse indices or binary).
        epsilon (float): Optional L2 budget, expressed as an L-infinity
            equivalent: the L2 radius is ``epsilon * sqrt(num_features)``.
            Defaults to ``None`` — the classic, unconstrained C&W attack that
            simply minimises the distortion needed to change the label.
        c (float): Weight of the misclassification term (default: 10.0). The
            C&W objective trades distortion against the margin, so ``c`` is the
            knob that controls how far the attack is willing to move: small
            values stall at the L2/margin equilibrium, large values find a
            (larger) successful perturbation. Sweep it to minimise distortion.
        kappa (float): Confidence margin (default: 0.0).
        num_steps (int): Number of Adam iterations (default: 100).
        alpha (float): Adam learning rate (default: 0.01).
        targeted (bool): Whether ``y`` is the target class.
        from_logits (bool): Whether ``model`` returns logits. Auto-detected if None.
        clip_min, clip_max (float): Valid data range.

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example(s).
    """
    epsilon = check_float("epsilon", epsilon, minimum=0.0, allow_none=True)
    c = check_float("c", c, minimum=0.0)
    kappa = check_float("kappa", kappa, minimum=0.0)
    num_steps = check_int("num_steps", num_steps, minimum=1)
    alpha = check_float("alpha", alpha, minimum=0.0)
    if c == 0:
        raise ValueError("`c` must be > 0 for the C&W objective to be meaningful.")

    ctx, dtype = _prepare(x, model, y, from_logits, clip_min, clip_max)
    lo, hi = ctx.clip_min, ctx.clip_max
    if lo is None:
        # The tanh parameterisation needs a bounded range.
        lo, hi = 0.0, 1.0

    num_features = float(np.prod([d for d in ctx.x.shape[1:]]))
    radius = None if epsilon is None else epsilon * np.sqrt(num_features)

    target = _resolve_target(model, ctx, targeted)

    w = _from_range(ctx.x, lo, hi)
    m = tf.zeros_like(w)
    v = tf.zeros_like(w)

    best_adv = tf.identity(ctx.x)
    best_dist = tf.fill([tf.shape(ctx.x)[0]], tf.cast(np.inf, dtype))
    found = tf.zeros([tf.shape(ctx.x)[0]], dtype=tf.bool)
    axes = list(range(1, ctx.x.shape.rank))

    for step in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(w)
            adv = _to_range(w, lo, hi)
            outputs = call_model(model, adv)
            margin = _margin(outputs, ctx, kappa=kappa, target=target)
            dist = tf.reduce_sum(tf.square(adv - ctx.x), axis=axes)
            objective = tf.reduce_sum(dist + c * margin)
        grad = tape.gradient(objective, w)
        if grad is None:
            raise RuntimeError("Gradient is None; the C&W attack cannot proceed.")

        # Stop updating samples that have already crossed the margin
        # (``margin <= 0`` means the target class now wins).
        done = tf.reshape(margin <= 0.0, [-1] + [1] * (ctx.x.shape.rank - 1))
        grad = tf.where(done, tf.zeros_like(grad), grad)

        w, m, v = _adam_step(w, grad, m, v, step, alpha)

        adv = _to_range(w, lo, hi)
        if radius is not None:
            delta = adv - ctx.x
            norm = tf.sqrt(tf.reduce_sum(tf.square(delta), axis=axes, keepdims=True))
            scale = tf.minimum(1.0, radius / (norm + 1e-12))
            adv = ctx.x + delta * scale
        adv = tf.clip_by_value(adv, lo, hi)
        w = _from_range(adv, lo, hi)

        # Track the smallest successful perturbation per sample.
        success = margin <= 0.0
        better = tf.logical_and(success, dist < best_dist)
        mask = tf.reshape(better, [-1] + [1] * (ctx.x.shape.rank - 1))
        best_adv = tf.where(mask, adv, best_adv)
        best_dist = tf.where(better, dist, best_dist)
        found = tf.logical_or(found, success)

    # For samples where no adversarial example was found, return the last
    # iterate (a perturbation that did not succeed) rather than silently
    # pretending the input is already minimal.
    mask = tf.reshape(found, [-1] + [1] * (ctx.x.shape.rank - 1))
    return as_numpy(tf.where(mask, best_adv, adv))


def ead(model, x, y, epsilon=0.01, beta=0.001, num_steps=50, alpha=0.01, *, c=0.1,
        kappa=0.0, targeted=False, from_logits=None, clip_min=None, clip_max=None):
    """
    Elastic Net Attack (EAD) — iterative shrinkage (ISTA) on the elastic-net
    objective ``||delta||_2^2 + beta * ||delta||_1 + c * margin``.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example(s) to attack.
        y (numpy.ndarray): True labels (one-hot, sparse indices or binary).
        epsilon (float): L-infinity perturbation budget (default: 0.01).
        beta (float): L1 regularisation weight (default: 0.001). Must be < 1;
            values that are large relative to ``alpha`` suppress the attack.
        num_steps (int): Number of iterations (default: 50).
        alpha (float): Step size (default: 0.01).
        c (float): Weight of the misclassification term (default: 0.1).
        kappa (float): Confidence margin (default: 0.0).
        targeted (bool): Whether ``y`` is the target class.
        from_logits (bool): Whether ``model`` returns logits. Auto-detected if None.
        clip_min, clip_max (float): Valid data range.

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example(s).
    """
    epsilon = check_float("epsilon", epsilon, minimum=0.0)
    beta = check_float("beta", beta, minimum=0.0, maximum=1.0)
    num_steps = check_int("num_steps", num_steps, minimum=1)
    alpha = check_float("alpha", alpha, minimum=0.0)
    c = check_float("c", c, minimum=0.0)
    kappa = check_float("kappa", kappa, minimum=0.0)
    if alpha == 0:
        raise ValueError("`alpha` must be > 0.")

    ctx, _ = _prepare(x, model, y, from_logits, clip_min, clip_max)
    axes = list(range(1, ctx.x.shape.rank))
    threshold = alpha * beta

    target = _resolve_target(model, ctx, targeted)

    adv = tf.identity(ctx.x)
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            outputs = call_model(model, adv)
            margin = _margin(outputs, ctx, kappa=kappa, target=target)
            dist = tf.reduce_sum(tf.square(adv - ctx.x), axis=axes)
            objective = tf.reduce_sum(dist + c * margin)
        grad = tape.gradient(objective, adv)
        if grad is None:
            raise RuntimeError("Gradient is None; the EAD attack cannot proceed.")

        previous = adv
        adv = adv - alpha * grad

        delta = adv - ctx.x
        delta = tf.sign(delta) * tf.maximum(tf.abs(delta) - threshold, 0.0)
        adv = ctx.x + delta

        adv = ctx.project(adv, epsilon)
        keep = _keep_mask(outputs, ctx, ctx.x, targeted=targeted)
        adv = tf.where(keep, adv, previous)

    return as_numpy(adv)


def deepfool(model, x, y, num_steps=50, epsilon=None, overshoot=0.02, *,
             from_logits=None, clip_min=None, clip_max=None):
    """
    DeepFool attack (Moosavi-Dezfooli et al., 2016).

    Computes, for each sample, the closest decision boundary of the current
    predicted class and steps towards it with an overshoot, repeating until the
    label flips or ``num_steps`` is reached.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example(s) to attack.
        y (numpy.ndarray): True labels (used to stop once misclassified).
        num_steps (int): Maximum number of iterations per sample (default: 50).
        epsilon (float): Optional L-infinity budget. ``None`` (default) lets the
            attack move to the closest boundary without a budget.
        overshoot (float): Overshoot factor applied to each step (default: 0.02).
        from_logits (bool): Whether ``model`` returns logits. Auto-detected if None.
        clip_min, clip_max (float): Valid data range.

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example(s).
    """
    num_steps = check_int("num_steps", num_steps, minimum=1)
    epsilon = check_float("epsilon", epsilon, minimum=0.0, allow_none=True)
    overshoot = check_float("overshoot", overshoot, minimum=0.0)

    ctx, _ = _prepare(x, model, y, from_logits, clip_min, clip_max)
    batch = int(ctx.x.shape[0])
    result = []

    for index in range(batch):
        original = ctx.x[index:index + 1]
        adv = tf.identity(original)
        true_label = int(ctx.y_idx[index])

        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                tape.watch(adv)
                outputs = _probabilities(call_model(model, adv), ctx)
            current = int(tf.argmax(outputs[0]))
            if current != true_label:
                break
            if outputs.shape[1] < 2:
                break

            jacobian = tape.jacobian(outputs, adv)  # (1, C, *shape)
            probs = outputs[0]
            w_cur = jacobian[0, current]
            differences = probs - probs[current]
            w_diff = jacobian[0] - w_cur

            flat = tf.reshape(w_diff, [tf.shape(w_diff)[0], -1])
            norm_sq = tf.reduce_sum(tf.square(flat), axis=1)
            ratio = tf.abs(differences) / (norm_sq + 1e-12)
            ratio = tf.tensor_scatter_nd_update(
                ratio, [[current]], [tf.cast(np.inf, ratio.dtype)]
            )
            best = int(tf.argmin(ratio))
            r = ratio[best] * w_diff[best]
            adv = adv + (1.0 + overshoot) * r

            adv = _project_against(adv, original, ctx, epsilon)

        result.append(adv)

    return as_numpy(tf.concat(result, axis=0))


def jsma(model, x, y, theta=0.0, gamma=0.2, num_steps=10, *, targeted=False,
         from_logits=None, clip_min=None, clip_max=None):
    """
    Jacobian-based Saliency Map Attack (JSMA), Papernot et al. (2016).

    At each iteration the saliency map
    ``S_i = dF_t/dx_i * |sum_{j != t} dF_j/dx_i|`` (zero where the derivative
    directions are unfavourable) is computed and the single most salient feature
    is increased by ``gamma``.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example(s) to attack.
        y (numpy.ndarray): True labels (the target class when ``targeted=True``).
        theta (float): Minimum saliency required to perturb a feature. The
            default of ``0.0`` always selects the most salient feature; raw
            saliency values are products of two gradients and therefore tiny.
        gamma (float): Perturbation added to the selected feature.
        num_steps (int): Number of features perturbed per sample (default: 10).
        targeted (bool): Treat ``y`` as the target class.
        from_logits (bool): Whether ``model`` returns logits. Auto-detected if None.
        clip_min, clip_max (float): Valid data range.

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example(s).
    """
    theta = check_float("theta", theta, minimum=0.0)
    gamma = check_float("gamma", gamma, minimum=0.0)
    num_steps = check_int("num_steps", num_steps, minimum=1)
    if gamma == 0:
        raise ValueError("`gamma` must be > 0.")

    ctx, _ = _prepare(x, model, y, from_logits, clip_min, clip_max)
    lo = ctx.clip_min if ctx.clip_min is not None else 0.0
    hi = ctx.clip_max if ctx.clip_max is not None else 1.0

    batch = int(ctx.x.shape[0])
    result = []

    for index in range(batch):
        adv = tf.identity(ctx.x[index:index + 1])
        true_label = int(ctx.y_idx[index])
        target = true_label

        for _ in range(num_steps):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(adv)
                probs = _probabilities(call_model(model, adv), ctx)
                if not targeted:
                    target = int(tf.argmin(probs[0]))
                target_prob = probs[0, target]
                other_prob = tf.reduce_sum(probs[0]) - target_prob

            grad_target = tape.gradient(target_prob, adv)
            grad_other = tape.gradient(other_prob, adv)
            del tape

            if grad_target is None or grad_other is None:
                break

            favourable = tf.logical_and(grad_target > 0.0, grad_other <= 0.0)
            saliency = tf.where(favourable, grad_target * tf.abs(grad_other), 0.0)
            flat = tf.reshape(saliency, [-1])
            best = int(tf.argmax(flat))
            if float(flat[best]) <= theta:
                break

            step = tf.one_hot(best, tf.size(flat), dtype=adv.dtype) * gamma
            adv = adv + tf.reshape(step, tf.shape(adv))
            adv = tf.clip_by_value(adv, lo, hi)

            predicted = int(tf.argmax(_probabilities(call_model(model, adv), ctx)[0]))
            if predicted != true_label:
                break

        result.append(adv)

    return as_numpy(tf.concat(result, axis=0))


def spsa(model, x, y, epsilon=0.01, num_steps=50, learning_rate=0.05, delta=None,
         spsa_samples=32, *, seed=None, targeted=False, from_logits=None,
         clip_min=None, clip_max=None):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) attack.

    A gradient-free attack: the gradient is estimated from loss differences
    ``(L(x + delta * d) - L(x - delta * d)) / (2 * delta)`` along random
    Rademacher directions ``d``, then the perturbation is optimised with Adam.

    Parameters:
        model (tensorflow.keras.Model): The target model to attack.
        x (numpy.ndarray): The input example(s) to attack.
        y (numpy.ndarray): True labels (one-hot, sparse indices or binary).
        epsilon (float): The L-infinity perturbation budget (default: 0.01).
        num_steps (int): Number of Adam iterations (default: 50).
        learning_rate (float): Adam learning rate (default: 0.05).
        delta (float): Finite-difference probe size. Defaults to ``epsilon / 5``.
            Very small probes can fall below float32 loss resolution.
        spsa_samples (int): Number of random directions per gradient estimate.
        seed (int): Optional random seed.
        targeted (bool): Whether ``y`` is the target class.
        from_logits (bool): Whether ``model`` returns logits. Auto-detected if None.
        clip_min, clip_max (float): Valid data range.

    Returns:
        adversarial_example (numpy.ndarray): The perturbed input example(s).
    """
    epsilon = check_float("epsilon", epsilon, minimum=0.0)
    num_steps = check_int("num_steps", num_steps, minimum=1)
    learning_rate = check_float("learning_rate", learning_rate, minimum=0.0)
    spsa_samples = check_int("spsa_samples", spsa_samples, minimum=2)
    if delta is None:
        # The probe must be large enough for the loss difference to survive
        # float32 quantisation on a confident model.
        delta = max(epsilon / 5.0, 1e-4)
    delta = check_float("delta", delta, minimum=0.0)
    if delta == 0:
        raise ValueError("`delta` must be > 0.")

    ctx, dtype = _prepare(x, model, y, from_logits, clip_min, clip_max)
    direction_sign = -1.0 if targeted else 1.0
    batch = int(ctx.x.shape[0])
    result = []

    for index in range(batch):
        original = ctx.x[index:index + 1]
        labels = ctx.labels[index:index + 1]
        perturbation = tf.zeros_like(original)
        m = tf.zeros_like(original)
        v = tf.zeros_like(original)

        for step in range(num_steps):
            probe_shape = tf.concat([[spsa_samples], tf.shape(original)[1:]], axis=0)
            directions = tf.sign(
                tf.random.uniform(probe_shape, -1.0, 1.0, seed=seed, dtype=dtype)
            )

            plus = call_model(model, original + delta * directions)
            minus = call_model(model, original - delta * directions)
            tiled_labels = tf.tile(labels, [spsa_samples, 1])
            if ctx.mode == "binary":
                loss_plus = tf.keras.losses.binary_crossentropy(tiled_labels, plus, from_logits=ctx.from_logits)
                loss_minus = tf.keras.losses.binary_crossentropy(tiled_labels, minus, from_logits=ctx.from_logits)
            else:
                loss_plus = tf.keras.losses.categorical_crossentropy(tiled_labels, plus, from_logits=ctx.from_logits)
                loss_minus = tf.keras.losses.categorical_crossentropy(tiled_labels, minus, from_logits=ctx.from_logits)

            differences = tf.reshape(loss_plus - loss_minus, [spsa_samples] + [1] * (original.shape.rank - 1))
            estimate = tf.reduce_mean(differences * directions, axis=0, keepdims=True) / (2.0 * delta)

            grad = -direction_sign * estimate
            perturbation, m, v = _adam_step(perturbation, grad, m, v, step, learning_rate)

            adv = _project_against(
                original + tf.clip_by_value(perturbation, -epsilon, epsilon),
                original, ctx,
            )
            outputs = call_model(model, adv)
            keep = _keep_mask(
                outputs, ctx, original,
                y_idx=ctx.y_idx[index:index + 1], targeted=targeted,
            )
            adv = tf.where(keep, adv, original)
            perturbation = adv - original

        result.append(adv)

    return as_numpy(tf.concat(result, axis=0))


# ---------------------------------------------------------------------------
# Text attacks
# ---------------------------------------------------------------------------

def _rng(seed):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def word_swap(text, swap_dict=None, max_swaps=None, seed=None):
    """
    Word substitution attack for text.

    Whitespace, punctuation and line breaks are preserved exactly.

    Parameters:
        text (str): The input text.
        swap_dict (dict): Mapping of words to their substitutes. Required.
        max_swaps (int): Optional cap on the number of substitutions.
        seed (int): Optional random seed (substitutions are applied in a random
            order when ``max_swaps`` is set).

    Returns:
        perturbed_text (str): The text with swapped words.
    """
    if not isinstance(text, str):
        raise TypeError(f"`text` must be a string, got {type(text).__name__}.")
    if swap_dict is None:
        raise ValueError("`swap_dict` is required; refusing to silently return the input.")
    if not isinstance(swap_dict, dict):
        raise TypeError("`swap_dict` must be a dict.")
    if max_swaps is not None:
        max_swaps = check_int("max_swaps", max_swaps, minimum=0)

    tokens = tokenize_text(text)
    candidates = [
        i for i, (token, is_word) in enumerate(tokens)
        if is_word and token in swap_dict
    ]
    if max_swaps is not None and len(candidates) > max_swaps:
        rng = _rng(seed)
        chosen = rng.choice(candidates, size=max_swaps, replace=False)
        candidates = [int(i) for i in chosen]

    for i in candidates:
        tokens[i] = (swap_dict[tokens[i][0]], False)
    return "".join(token for token, _ in tokens)


def char_swap(text, swap_prob=0.1, seed=None):
    """
    Adjacent-character swap attack for text.

    Only characters inside words are swapped, so whitespace, newlines and
    punctuation placement are preserved.

    Parameters:
        text (str): The input text.
        swap_prob (float): Probability of swapping a character in a word (default: 0.1).
        seed (int): Optional random seed for reproducibility.

    Returns:
        perturbed_text (str): The text with swapped characters.
    """
    if not isinstance(text, str):
        raise TypeError(f"`text` must be a string, got {type(text).__name__}.")
    swap_prob = check_float("swap_prob", swap_prob, minimum=0.0, maximum=1.0)
    rng = _rng(seed)

    tokens = tokenize_text(text)
    for i, (token, is_word) in enumerate(tokens):
        if not is_word or len(token) < 2:
            continue
        if rng.random() >= swap_prob:
            continue
        chars = list(token)
        j = int(rng.integers(0, len(chars) - 1))
        chars[j], chars[j + 1] = chars[j + 1], chars[j]
        tokens[i] = ("".join(chars), True)
    return "".join(token for token, _ in tokens)
