"""
This attempt will focus on variational auto encoders
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

import numpy as np
import pandas as pd
import tensorflow as tf
import Plotting as plots
import matplotlib.pyplot as plt

from VAE import VAE
from keras import ops
from keras import layers
from Data_Options import data_options
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data_ops = data_options()

def main():

    data = data_ops.get_data()
    train, test, valid = data_ops.get_partitioned_data(data)
    #can add Teq
    y_col = ["m_core"]
    x_col = ["mass","req","Teq","k2"]
    train_x, train_y = data_ops.get_xy(train, y_col, x_col)
    test_x, test_y = data_ops.get_xy(test, y_col, x_col)
    valid_x, valid_y = data_ops.get_xy(valid, y_col, x_col)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_x)
    train_y = y_scaler.fit_transform(train_y)
    # Do not refit the transform
    test_x = x_scaler.transform(test_x)

    input_dim = train_x.shape[1]
    output_dim = train_y.shape[1]
    no_epochs = 1
    batch_size = 32


    latent_dim = 2
    vae = VAE(latent_dim=latent_dim,input_dim=input_dim, output_dim= output_dim)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.encoder.summary()
    vae.decoder.summary()


    vae.fit(train_x, train_y, epochs=no_epochs, batch_size=batch_size, validation_data=(test_x, test_y))
    plots.plot_latent_space(vae,test_x,test_y)

    _, _, train_z = vae.encoder.predict(train_x)
    _, _, test_z = vae.encoder.predict(test_x)


    z_mean, z_log_var, z = vae.encoder.predict(test_x)


    reg = LinearRegression()
    reg.fit(train_z, train_y)

    # Step 3: Predict
    pred_y = reg.predict(test_z)
    pred_y = y_scaler.inverse_transform(pred_y)

    pred_y = pd.DataFrame(pred_y,columns=y_col)
    test_y = pd.DataFrame(test_y, columns=y_col)

    plots.plot_real_vs_pred(pred_y,test_y)

    print(f"pred_y: {pred_y}")
    print(f"test_y: {test_y}")

main()