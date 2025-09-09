""" Going to store my MDN stuff in here
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
import keras


tf.executing_eagerly()

def softmax(w):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature.
    Arguments:
    w -- a list or numpy array of logits
    Keyword arguments:
    t -- the temperature for to adjust the distribution (default 1.0)
    """
    e = np.array(w)  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist

@keras.saving.register_keras_serializable()
class MDN(layers.Layer):
    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        with tf.name_scope('MDN'):
            self.mdn_mus = layers.Dense(self.num_mix * self.output_dim,
                                        name='mdn_mus')  # mix*output vals, no activation
            self.mdn_sigmas = layers.Dense(self.num_mix * self.output_dim, activation="softplus",
                                         name='mdn_sigmas')  # mix*output vals exp activation
            # self.mdn_sigmas = layers.Dense(self.num_mix * self.output_dim, activation='softplus', name='mdn_sigmas')

            self.mdn_pi = layers.Dense(self.num_mix, name='mdn_pi')  # mix vals, logits
        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.name_scope('mus'):
            self.mdn_mus.build(input_shape)
        with tf.name_scope('sigmas'):
            self.mdn_sigmas.build(input_shape)
        with tf.name_scope('pis'):
            self.mdn_pi.build(input_shape)
        super(MDN, self).build(input_shape)

    @property
    def trainable_weights(self):
        return self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights

    def call(self, inputs):
        with tf.name_scope('MDN'):
            mu = self.mdn_mus(inputs)
            sigma = self.mdn_sigmas(inputs)
            pi_logits = self.mdn_pi(inputs)
            pi = tf.nn.softmax(pi_logits, axis=-1)  # Ensure probs during inference

            mdn_out = layers.concatenate([mu, sigma, pi], name='mdn_outputs')
            return mdn_out

    def compute_output_shape(self, input_shape):
        """Returns output shape, showing the number of mixture parameters."""
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return K.elu(x) + 1 + K.epsilon()

@keras.saving.register_keras_serializable()
class MDNLoss(tf.keras.losses.Loss):
    def __init__(self, output_dim, num_mixes, name="mdn_loss"):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.num_mixes = num_mixes

    def call(self, y_true, y_pred):
        # Split y_pred into mu, sigma, pi
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                self.num_mixes * self.output_dim,
                self.num_mixes * self.output_dim,
                self.num_mixes
            ],
            axis=-1
        )

        # Reshape to (batch_size, num_mixes, output_dim)
        out_mu = tf.reshape(out_mu, [-1, self.num_mixes, self.output_dim])
        out_sigma = tf.reshape(out_sigma, [-1, self.num_mixes, self.output_dim])
        out_pi = tf.nn.softmax(out_pi, axis=-1)

        # Create the mixture model
        cat = tfd.Categorical(probs=out_pi)
        components = tfd.MultivariateNormalDiag(loc=out_mu, scale_diag=out_sigma)
        mixture = tfd.MixtureSameFamily(mixture_distribution=cat, components_distribution=components)

        log_likelihood = mixture.log_prob(y_true)
        return -tf.reduce_mean(log_likelihood)

    def get_config(self):
        return {
            "output_dim": self.output_dim,
            "num_mixes": self.num_mixes,
            "name": self.name
        }

@keras.saving.register_keras_serializable()
class MDNLossWithEntropy(tf.keras.losses.Loss):
    def __init__(self, output_dim, num_mixes, entropy_weight=1e-3, name="mdn_loss_with_entropy"):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.num_mixes = num_mixes
        self.entropy_weight = entropy_weight

    def call(self, y_true, y_pred):
        num_mixes = self.num_mixes
        output_dim = self.output_dim

        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            [num_mixes * output_dim, num_mixes * output_dim, num_mixes],
            axis=-1
        )

        out_mu = tf.reshape(out_mu, [-1, num_mixes, output_dim])
        out_sigma = tf.reshape(out_sigma, [-1, num_mixes, output_dim])
        out_pi = tf.nn.softmax(out_pi)

        cat = tfd.Categorical(probs=out_pi)
        components = tfd.MultivariateNormalDiag(loc=out_mu, scale_diag=out_sigma)
        mixture = tfd.MixtureSameFamily(mixture_distribution=cat, components_distribution=components)

        log_likelihood = mixture.log_prob(y_true)
        entropy = -tf.reduce_mean(tf.reduce_sum(out_pi * tf.math.log(out_pi + 1e-8), axis=-1))
        return -tf.reduce_mean(log_likelihood) - self.entropy_weight * entropy

    def get_config(self):
        return {
            "output_dim": self.output_dim,
            "num_mixes": self.num_mixes,
            "entropy_weight": self.entropy_weight
        }