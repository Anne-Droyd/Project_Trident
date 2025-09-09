"""
Going to store my MDN stuff in here
"""
import keras
from keras import backend as K
from keras.activations import elu
from keras.layers import Dense
from keras.layers import Layer
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return elu(x) + 1 + K.epsilon()

def entropy_regularization(pi, weight=0.01):
    """
    A function to add the sum of the pis * log of the pis along each row (shannon entropy)
    while adding a small value to stop log(0). This should discourage uncertain distributions
    """
    entropy = -tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=-1)
    return -weight * entropy

def softmax(w,temperature=1.0):
    """
    My edit: I added the temperature variable which was excluded from the github.
    Higher temperature means more uniform distribution(less certainty),
    lower temperature encourages selection of most likely gaussians as pdf

    Softmax function for a list or numpy array of logits. Also adjusts temperature.
    Arguments:
    w -- a list or numpy array of logits
    Keyword arguments:
    t -- the temperature for to adjust the distribution (default 1.0)
    """
    e = np.array(w/temperature)  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist

class MDN(Layer):
    """
    Using EXOMDN as jump off point
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        #name scope is kinda like a directory for variables so the variables here would be MDN/mdn_mus ect
        with tf.name_scope('MDN'):
            #We need to essentially have a output that is (num_mix * num_outs, num_mix * num_outs, num_mix)
            #this is because our confidence in a certain output gaussian will be the same across all variables
            #ie if the model is really confident in one gaussian for a certain output say zatm, then
            #the model will be really confident in the same gaussian for other outputs
            self.mdn_mus = Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_mus')
            self.mdn_sigmas = Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')
            #tried to softmax here but think doing it after is more sensical.
            self.mdn_pi = Dense(self.num_mix, name='mdn_pi')
        super(MDN, self).__init__(**kwargs)

    #removed the build definition as it should be done automatically

    #don't need mask
    def call(self, x):
        with tf.name_scope('MDN'):
            mdn_out = keras.layers.concatenate([self.mdn_mus(x),
                                                self.mdn_sigmas(x),
                                                self.mdn_pi(x)],
                                               name='mdn_outputs')
        return mdn_out

    #removed compute_output_shape as it should be done automatically

    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    #removed the properties and class method wrappers becuase they should be handled automatically

def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    # Construct a loss function with the right number of mixtures and outputs
    def mdn_loss_func(y_true, y_pred):
        # Split the inputs into parameters
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')
        # Construct the mixture models

        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_func

def get_winner_takes_all_loss(output_dim, num_mixes):
    def loss(y_true, y_pred):
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[num_mixes * output_dim,
                                num_mixes * output_dim,
                                num_mixes],
            axis=-1
        )

        mus = tf.reshape(out_mu, [-1, num_mixes, output_dim])
        sigs = tf.nn.softplus(tf.reshape(out_sigma, [-1, num_mixes, output_dim])) + 1e-6
        pi = tf.nn.softmax(out_pi)

        mvn = tfd.MultivariateNormalDiag(loc=mus, scale_diag=sigs)
        log_probs = mvn.log_prob(tf.expand_dims(y_true, 1))  # [batch, num_mixes]

        # Posterior responsibility: p(component | y_true)
        log_pi = tf.math.log(pi + 1e-8)
        log_responsibility = log_pi + log_probs
        top_component = tf.argmax(log_responsibility, axis=1, output_type=tf.int32)
        batch_idx = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
        idx = tf.stack([batch_idx, top_component], axis=1)
        chosen_mu = tf.gather_nd(mus, idx)
        chosen_sigma = tf.gather_nd(sigs, idx)
        chosen_dist = tfd.MultivariateNormalDiag(loc=chosen_mu, scale_diag=chosen_sigma)
        return tf.reduce_mean(-chosen_dist.log_prob(y_true))
    return loss

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