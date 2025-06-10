"""The plan here is to use an MDN."""
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras

import Plotting as plots
import numpy as np
import matplotlib.pyplot as plt

from keras import layers, callbacks, ops
from Data_Options import data_options
from tensorflow.keras.utils import plot_model
from tensorflow_probability import optimizer
from tensorflow_probability.substrates.jax import distributions as tfd

@keras.saving.register_keras_serializable(package="MyLayers")
class MDN(layers.Layer):
    def __init__(self,output_dim,num_mixtures,**kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_mixtures=num_mixtures
        #mean/mu
        self.mdn_mus=layers.Dense(self.num_mixtures*self.output_dim,name="mdn_mus")
        #sigma
        self.mdn_sigmas=layers.Dense(self.num_mixtures*self.output_dim,activation=self.elu_plus_one_plus_epsilon,name="mdn_sigmas",)
        #pi
        self.mdn_pi=layers.Dense(self.num_mixtures,name="mdn_pi")


    def elu_plus_one_plus_epsilon(self,x):
        return keras.activations.elu(x) + 1 + keras.backend.epsilon()

    def build(self,input_dim):
        self.mdn_mus.build(input_dim)
        self.mdn_sigmas.build(input_dim)
        self.mdn_pi.build(input_dim)
        super().build(input_dim)

    @property
    def trainable_weights(self):
        return (
            self.mdn_mus.trainable_weights
            + self.mdn_sigmas.trainable_weights
            + self.mdn_pi.trainable_weights
        )

    @property
    def non_trainable_weights(self):
        return (
            self.mdn_mus.non_trainable_weights
            + self.mdn_sigmas.non_trainable_weights
            + self.mdn_pi.non_trainable_weights
        )

    def call(self, x, mask=None):
        return layers.concatenate(
            [self.mdn_mus(x), self.mdn_sigmas(x), self.mdn_pi(x)], name="mdn_outputs"
        )


data_ops=data_options()

def model_maker(input,output):
    inputs=keras.Input(shape=(input,))
    x = layers.Dense(64,activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(output)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="M_R_Teq_model")
    print(model.summary())
    plot_model(model, "my_first_model.png")
    return model

@keras.saving.register_keras_serializable(package="MyLosses")
def get_mixture_loss_func(output_dim, num_mixes):
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistributed layer
        y_pred = ops.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes])
        y_true = ops.reshape(y_true, [-1, output_dim])
        # Split the inputs into parameters
        out_mu, out_sigma, out_pi = ops.split(y_pred, 3, axis=-1)
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        mus = ops.split(out_mu, num_mixes, axis=1)
        sigs = ops.split(out_sigma, num_mixes, axis=1)
        coll = [
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            for loc, scale in zip(mus, sigs)
        ]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = ops.negative(loss)
        loss = ops.mean(loss)
        return loss

    return mdn_loss_func

def split_mixture_params(params, output_dim, num_mixes):
    mus = params[: num_mixes * output_dim]
    sigs = params[num_mixes * output_dim : 2 * num_mixes * output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist


def sample_from_categorical(dist):
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    print("Error sampling categorical model.")
    return -1


def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim : (m + 1) * output_dim]
    sig_vector = sigs[m * output_dim : (m + 1) * output_dim]
    scale_matrix = np.identity(output_dim) * sig_vector  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample


def main():
    data=data_ops.get_data()
    train, test, valid = data_ops.partition_data(data)
    train_x, train_y = data_ops.get_xy(train, "m_core", ["mass", "req", "Teq"])
    test_x, test_y = data_ops.get_xy(test, "m_core", ["mass", "req", "Teq"])
    valid_x, valid_y = data_ops.get_xy(valid, "m_core", ["mass", "req", "Teq"])
    print(test_x)

    model=model_maker(train_x.shape[1],train_y.shape[1])
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=keras.optimizers.RMSprop(),metrics=["accuracy"])
    history = model.fit(train_x,train_y,batch_size=64,epochs=2,validation_split=0.2)
    test_scores = model.evaluate(test_x,test_y,verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    user="n"
    if user == "y":
        mdn_network = keras.models.load_model("mdn_model.keras")
    else:
        OUTPUT_DIMS = train_y.shape[1]
        N_MIXES = 30
        N_HIDDEN = 128
        mdn_network = keras.Sequential(
            [
                layers.Dense(N_HIDDEN, activation="relu"),
                layers.Dense(N_HIDDEN, activation="relu"),
                MDN(OUTPUT_DIMS, N_MIXES),
            ]
        )

        mdn_network.compile(loss=get_mixture_loss_func(OUTPUT_DIMS, N_MIXES), optimizer="adam")
        mdn_network.fit(
            train_x,
            train_y,
            epochs=150,
            batch_size=128,
            validation_split=0.15,
            callbacks=[
                callbacks.EarlyStopping(monitor="loss", patience=30, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="loss", patience=5),
            ],
        )

        mdn_network.save("mdn_model.keras")

    y_pred_mixture = mdn_network.predict(test_x)
    # Sample from the predicted distributions
    y_samples = np.apply_along_axis(
        sample_from_output, 1, y_pred_mixture, 1, N_MIXES, temp=1.0
    )



    plt.scatter(
        test_y[:100],
        y_samples[:100, :, 0],
        color="green",
        alpha=0.05,
        label="Mixture Density Network prediction",
    )
    plt.scatter(
        test_y[:100],
        test_y[:100],
        color="red",
        alpha=0.05,
        label="Real values",
    )
    plt.show()


main()