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

def softmax(w,temperature=1.0):
    """
    My edit: I added the temperature variable which was excluded from the github.
    Higher temperature means more uniform distribution(less certainty),
    lower temperature encourages selection of most likely gaussians as pdf

    Softmax function for a list or numpy array of logits. Also adjusts temperature.
    Arguments:
    w -- a list or numpy array of logits
    Keyword arguments:
    temperature -- the temperature for to adjust the distribution (default 1.0)
    """
    e = np.array(w)/temperature  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist

def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    """Sample from an MDN output with temperature adjustment.
    This calculation is done outside of the Keras model using
    Numpy.

    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented

    Keyword arguments:
    temp -- the temperature for sampling between mixture components (default 1.0)
    sigma_temp -- the temperature for sampling from the normal distribution (default 1.0)

    Returns:
    One sample from the the mixture model, that is a numpy array of length output_dim
    """
    assert len(params) == num_mixes + (output_dim * 2 * num_mixes), "The size of params needs to match the mixture configuration"
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, temperature=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim:(m + 1) * output_dim]
    sig_vector = sigs[m * output_dim:(m + 1) * output_dim]
    scale_matrix = np.identity(output_dim) * sig_vector  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample[0]

class MDN(Layer):
    """
    A Mixture Density Layer for Keras
    cpmpercussion: Charles Martin (University of Oslo) 2018
    https://github.com/cpmpercussion/keras-mdn-layer
    Hat tip to [Omimo's Keras MDN layer](https://github.com/omimo/Keras-MDN)
    for a starting point for this code.
    Provided under MIT License
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
        #this is the tensorflow version of what I did in my main script
        #isolate pi, mu, sigma
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        #rebuild gaussians
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        #blend them into mixture
        mixture = tfd.Mixture(cat=cat, components=coll)
        #negative log likelihood loss
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_func

#had a winner takes all loss function but wasn't really an MDN after so went on to include entropy

def mdn_loss_with_entropy(output_dim,num_mixes,entropy_weight=1e-3,temperature=1):
    """
    the MDN loss function but the certainty of the model can be altered through entropy weight
    """
    def mdn_loss_entropy_func(y_true,y_pred):
        # Split the inputs into parameters
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')
        # Construct the mixture models
        # this is the tensorflow version of what I did with numpy in my main script
        # isolate pi, mu, sigma
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        # rebuild gaussians
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        # blend them into mixture
        mixture = tfd.Mixture(cat=cat, components=coll)

        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        # adding a term to penalise uncertainty, so 1 gaussian is most likely
        pi_after_softmax = np.apply_along_axis(softmax, 1, out_pi, temperature=temperature)
        entropy = -tf.reduce_mean(tf.reduce_sum(pi_after_softmax * tf.math.log(pi_after_softmax + 1e-8), axis=-1))
        loss = -tf.reduce_mean(loss) - entropy_weight * entropy
        return loss

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_entropy_func
