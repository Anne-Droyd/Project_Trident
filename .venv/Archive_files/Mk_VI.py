""" Hopefully this run works
How do I build an MDN???
Going to try make a simple double mixture network
"""
import keras
import pickle
import keras_tuner as kt
import MDN as mdn
import numpy as np
import pandas as pd
import tensorflow as tf
import Plotting as plots
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler

from MDN import MDN
from scipy.stats import norm
from keras.layers import Dropout
from Data_Options import data_options
from keras import callbacks, layers, ops
from Linear_Methods import Linear_methods as linear


tfd = tfp.distributions

data_ops = data_options()
linear_methods = linear()


print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("GPU is available and will be used.")
else:
    print("No GPU found. Using CPU.")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

def sample_from_mixture(pi, mu, sigma, n_samples=100):
    """
    Sample from an MDN's mixture for each test point.
    - pi: shape (batch_size, num_mixtures)
    - mu: shape (batch_size, num_mixtures, output_dim)
    - sigma: shape (batch_size, num_mixtures, output_dim)
    """
    batch_size, num_mixtures, output_dim = mu.shape
    samples = np.zeros((batch_size, n_samples, output_dim))

    for i in range(batch_size):
        # Choose mixture indices for sampling
        mixture_idx = np.random.choice(num_mixtures, size=n_samples, p=pi[i])

        for j, mix in enumerate(mixture_idx):
            mean = mu[i, mix]
            std = sigma[i, mix]
            samples[i, j] = np.random.normal(loc=mean, scale=std)

    return samples  # shape (batch_size, n_samples, output_dim)


def main():

    data = data_ops.get_data()
    train, test, valid = data_ops.partition_data(data)

    # Index(['m_core', 'zatm', 'zatm0', 'zatm1', 'zdeep', 'zdeep0', 'zdeep1',
    #        'p_ppt', 'req', 'mass', 'lum', 'Teq', 'p_rot', 'k2', "test"],
    #can add Teq, k2, zatm, zdeep, zatm0/1 , zdeep0/1
	
    #mkIII = mass, req, mcore, zatm, zdeep
    #mkiv = mcore, zatm, zdeep, mass, req, teq
    #mkv= mcore, mass, req
    #mkvi = mcore, mass, req, teq

    path = "C:/Users/Matth/PycharmProjects/Project_Trident/.venv/Archive_files/models"
    Model_save_name = "MDN_model_MKVI.keras"
    history_name = "Training_history_MDN_MKVI"
    hyper_parameter ="Hyper_params_MKVI"
    train_new_model = "n"
    overwrite=True
    y_col = ["m_core","test"]
    x_col = ["mass","req","Teq"]
    epoch = 300


    train_x, train_y = data_ops.get_xy(train, y_col, x_col)
    test_x, test_y = data_ops.get_xy(test, y_col, x_col)
    valid_x, valid_y = data_ops.get_xy(valid, y_col, x_col)

    print("Train_y shape:", train_y.shape)

    # Get the correlation of the training data to see if its worth looking at
    # df = pd.DataFrame(train_x, columns=x_col)
    # df["m_core"] = train_y
    # print("correlation", df.corr())

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_x)
    train_y = y_scaler.fit_transform(train_y)
    valid_x = x_scaler.transform(valid_x)
    valid_y = y_scaler.transform(valid_y)

    #Do not refit the transform
    test_x = x_scaler.transform(test_x)

    #linear.L_reg(train_x,train_y,test_x,test_y)
    #linear.R_for(train_x,train_y,test_x,test_y)

    input_dim = train_x.shape[1]
    output_dim = train_y.shape[1]

    try:
        with open(path+hyper_parameter,"rb") as file:
            best_hp = pickle.load(file)

        N_MIXES = best_hp.get('n_mixtures')
    except FileNotFoundError:
        N_MIXES=20


    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

    loss_func = mdn.MDNLoss(output_dim, N_MIXES)

    def build_mdn_model(hp):
        input_dim = train_x.shape[1]
        output_dim = train_y.shape[1]

        n_hidden = hp.Int('n_hidden', min_value=64, max_value=1024, step=128)
        n_layers = hp.Int('n_layers', min_value=1, max_value=5)
        n_mixtures = hp.Int('n_mixtures', min_value=5, max_value=30, step=5)
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        model.add(layers.Dropout(0.3))

        for _ in range(n_layers):
            model.add(layers.Dense(n_hidden, activation='relu'))

        model.add(MDN(output_dim, n_mixtures))

        loss = mdn.MDNLoss(output_dim, n_mixtures)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=loss)
        return model



    if train_new_model.lower() == "y" or train_new_model.lower() == "yes":
        with tf.device('/GPU:0'):
            tuner = kt.BayesianOptimization(
                build_mdn_model,
                objective='val_loss',
                max_trials=20,
                directory='tuner_dir',
                project_name='mdn_search',
                overwrite = overwrite
            )

            tuner.search(train_x, train_y,
                         validation_split=0.15,
                         epochs=epoch,
                         batch_size=128,
                         callbacks=[
                             callbacks.EarlyStopping(monitor='val_loss', patience=10),
                             callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5),
                         ])

            best_model = tuner.get_best_models(num_models=1)[0]
            best_hp = tuner.get_best_hyperparameters(1)[0]
            N_MIXES = best_hp.get('n_mixtures')
            loss_func = mdn.MDNLoss(output_dim, N_MIXES)

            # Optional: re-train best model on full training data
            history = best_model.fit(train_x, train_y,
                                     validation_split=0.15,
                                     epochs=epoch,
                                     batch_size=128,
                                     callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10),
		                     callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)
		                     ])

            best_model.save(path+Model_save_name)

            mdn_network = best_model

            with open(path+hyper_parameter, "wb") as file:
                pickle.dump(best_hp,file)
                
            print(best_hp)


            with open(path+history_name,"wb") as file:
                pickle.dump(history,file)




    else:
        mdn_network = tf.keras.models.load_model(path+Model_save_name,
                                                      custom_objects={"MDN": MDN,"MDNLoss":loss_func})

        with open(path+history_name,"rb") as file:
            history = pickle.load(file)
            



    plots.plot_history_loss(history)

	
    y_pred_mixture = mdn_network.predict(test_x)
    train_y = y_scaler.inverse_transform(train_y)

    # Extract mixture parameters
    mu, sigma, pi = np.split(y_pred_mixture, [N_MIXES * output_dim, 2 * N_MIXES * output_dim], axis=1)

    # Inverse-transform means only
    # Reshape to (batch_size, N_MIXES, 1) for consistency (especially important for multi-output)
    mu = mu.reshape(-1, N_MIXES, output_dim)
    sigma = sigma.reshape(-1, N_MIXES, output_dim)

    # Flatten to 2D for inverse transform (samples * N_MIXES, output_dim)
    mu_flat = mu.reshape(-1, output_dim)
    mu_unscaled = y_scaler.inverse_transform(mu_flat)
    mu_unscaled = mu_unscaled.reshape(-1, N_MIXES, output_dim)

    # Same for sigma (optional, rough scaling)
    sigma = sigma * y_scaler.scale_.reshape(1, 1, -1)
    sigma = np.clip(sigma, 1e-3, None)

    max_pis = np.max(pi, axis=1)
    plt.hist(max_pis, bins=100)
    plt.title("Histogram of Maximum Pi per Sample")
    plt.show()

    weighted_mu = pi[:, :, np.newaxis] * mu_unscaled
    mean_pred = np.sum(weighted_mu, axis=1)


    # Convert to DataFrame
    singular_prediction = pd.DataFrame(
        data=mean_pred,  # (samples, output_dim), using mixture mean approx
        columns=y_col
    )

    train_x = pd.DataFrame(train_x,columns=x_col)
    train_y = pd.DataFrame(train_y,columns=y_col)
    test_x = pd.DataFrame(test_x,columns=x_col)
    test_y = pd.DataFrame(test_y, columns=y_col)
    valid_x = pd.DataFrame(valid_x, columns= x_col)
    valid_y = pd.DataFrame(valid_y, columns= y_col)
    print(test_y)

    plots.plot_hist_pred_mean_vs_real(test_y,singular_prediction,"m_core")
    plots.plot_mdn_prediction(pi,mu,sigma,y_scaler)

    # Compute mean: E[y] = sum(pi * mu)
    mean_pred = np.sum(pi[:, :, np.newaxis] * mu_unscaled, axis=1)

    y_true_unscaled=test_y


    for i, name in enumerate(y_col):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_unscaled.iloc[:, i], mean_pred[:, i], alpha=0.03, label="Predictions")
        plt.plot(
            [y_true_unscaled.iloc[:, i].min(), y_true_unscaled.iloc[:, i].max()],
            [y_true_unscaled.iloc[:, i].min(), y_true_unscaled.iloc[:, i].max()],
            'r--', label="1:1 line"
        )
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"MDN Predictions for {name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    samples = sample_from_mixture(pi, mu, sigma, n_samples=100)
    print("samples shape:", samples.shape)
    # Get percentiles
    y_lower = np.percentile(samples, 5, axis=1)
    y_upper = np.percentile(samples, 95, axis=1)
    y_median = np.percentile(samples, 50, axis=1)

    for i, name in enumerate(y_col):
        y_true_i = y_true_unscaled.iloc[:, i] if hasattr(y_true_unscaled, 'iloc') else y_true_unscaled[:, i]
        y_lower_i = y_lower[:, i]
        y_upper_i = y_upper[:, i]
        y_median_i = y_median[:, i]

        plt.figure(figsize=(6, 6))
        plt.errorbar(
            y_true_i, y_median_i,
            yerr=[y_median_i - y_lower_i, y_upper_i - y_median_i],
            fmt='o', alpha=0.02, label="90% CI"
        )
        plt.plot(
            [y_true_i.min(), y_true_i.max()],
            [y_true_i.min(), y_true_i.max()],
            'r--', label="1:1 line"
        )
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"MDN 90% CI for {name}")
        plt.legend()
        plt.grid(True)
        plt.show()

main()
