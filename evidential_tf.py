"""
Evidential / Dirichlet head utilities (TensorFlow), aligned with EviDTI (Sensoy et al. style loss).
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="DDI")
class DirichletAlphasFromLogits(layers.Layer):
    """Maps unconstrained logits to Dirichlet concentration parameters alpha > 1."""

    def call(self, logits):
        return tf.nn.softplus(logits) + 1.0


@tf.keras.utils.register_keras_serializable(package="DDI")
class DirichletEvidentialLoss(keras.losses.Loss):
    """
    Multiclass evidential loss: type-II maximum likelihood + KL to uniform Dirichlet
    on non-target classes (masked alpha_hat), following EviDTI / Chemprop-style formulation.
    y_true: integer sparse labels shape (batch,) or (batch, 1).
    y_pred: Dirichlet alphas shape (batch, num_classes), each alpha > 1.
    """

    def __init__(self, num_classes=4, lam=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.lam = float(lam)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "lam": self.lam})
        return cfg

    @staticmethod
    def _kl_dirichlet_to_uniform(alpha):
        beta = tf.ones_like(alpha)
        s_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        s_beta = tf.reduce_sum(beta, axis=-1, keepdims=True)

        ln_alpha = tf.math.lgamma(s_alpha) - tf.reduce_sum(
            tf.math.lgamma(alpha), axis=-1, keepdims=True
        )
        ln_beta = tf.reduce_sum(tf.math.lgamma(beta), axis=-1, keepdims=True) - tf.math.lgamma(
            s_beta
        )

        dg_alpha = tf.math.digamma(alpha)
        dg_s_alpha = tf.math.digamma(s_alpha)

        kl = ln_alpha + ln_beta + tf.reduce_sum(
            (alpha - beta) * (dg_alpha - dg_s_alpha), axis=-1, keepdims=True
        )
        return kl

    def call(self, y_true, alphas):
        y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_one_hot = tf.one_hot(y, depth=self.num_classes, dtype=alphas.dtype)

        s = tf.reduce_sum(alphas, axis=-1, keepdims=True)
        p = alphas / s
        a = tf.reduce_sum(tf.square(y_one_hot - p), axis=-1, keepdims=True)
        b = tf.reduce_sum((p * (1.0 - p)) / (s + 1.0), axis=-1, keepdims=True)
        sos = a + b

        alpha_hat = y_one_hot + (1.0 - y_one_hot) * alphas
        kl = self.lam * self._kl_dirichlet_to_uniform(alpha_hat)

        return tf.squeeze(sos + kl, axis=-1)
