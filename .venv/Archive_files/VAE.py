
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

from keras import ops
from keras import layers

class VAE(keras.Model):
    def __init__(self, latent_dim, input_dim ,output_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.regressor = self._build_regressor()

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def _build_regressor(self):
        reg_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(16, activation='relu')(reg_inputs)
        x = layers.Dense(8, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)  # assuming 1D regression target
        return keras.Model(reg_inputs, outputs, name='regressor')

    def _build_encoder(self):
        encoder_inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(64, activation='relu')(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = layers.Lambda(self.sampling, name='z')([z_mean, z_log_var])

        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    def _build_decoder(self):
        decoder_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32, activation='relu')(decoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        decoder_outputs = layers.Dense(self.input_dim, activation='linear') \
            (x)  # Use 'linear' if your inputs are unbounded

        return keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        x, y_true, _ = tf.keras.utils.unpack_x_y_sample_weight(data)


        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            x_hat = self.decoder(z)

            # Reconstruction loss (x → x_hat)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(x, x_hat)
            )

            # KL divergence
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
            kl_loss = tf.reduce_mean(kl_loss)


            y_pred = self.regressor(z)
            regression_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(y_true, y_pred)
            )

            # Weighted total loss
            beta = tf.minimum(1.0, tf.cast(self.optimizer.iterations, tf.float32) / 1000.0)
            alpha = 10.0  # <- you can tune this
            total_loss = reconstruction_loss + beta * kl_loss + alpha * regression_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y_true, _ = tf.keras.utils.unpack_x_y_sample_weight(data)


        z_mean, z_log_var, z = self.encoder(x)
        y_pred = self.regressor(z)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            keras.losses.mean_squared_error(x, reconstruction)
        )

        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        kl_loss = tf.reduce_mean(kl_loss)

        regression_loss = tf.reduce_mean(
            keras.losses.mean_squared_error(y_true, y_pred)
        )


        beta = tf.minimum(1.0, tf.cast(self.optimizer.iterations, tf.float32) / 1000.0)
        alpha = 10.0
        total_loss = reconstruction_loss + beta * kl_loss + alpha * regression_loss


        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
