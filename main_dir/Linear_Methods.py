""" Going to store my linear methods that work here
"""
import os
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers, layers, ops, Sequential
import Plotting as plots

from Data_Options import data_options

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

data_ops=data_options()

class Linear_methods:

    def __init__(self):
        self.save_model_path = "C:/Users/Matth/Documents/GitHub/first_research_project/.venv/Project_first_model/models/"

    def plot_loss(self, history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error [MPG]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def L_reg(self, train_x, train_y, test_x, test_y):
        user="n"
        train_x,train_y,test_x,test_y,x_scaler,y_scaler = data_ops.scale_data(train_x,train_y,test_x,test_y)
        if user=="y" and os.path.isfile(self.save_model_path+"Linear_regression_model.keras"):
            model = tf.keras.models.load_model(self.save_model_path+"Linear_regression_model.keras")
        else:

            normalizer = layers.Normalization()
            normalizer.adapt(train_x)
            model = Sequential([normalizer,layers.Dense(1)])
            model.summary()
            model.compile(optimizers.Adam(learning_rate=0.1),loss="mse",metrics=["mae"])
            history = model.fit(train_x,train_y,epochs=10,validation_split=0.2)
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            hist.tail()
            self.plot_loss(history)
            model.save(self.save_model_path + "Linear_regression_model.keras")
        print("Predicting...")
        preds=model.predict(test_x).flatten()
        plots.plot_real_vs_pred(pd.DataFrame(preds),pd.DataFrame(test_y))
        # Evaluate on test set
        loss, mae = model.evaluate(test_x, test_y, verbose=0)
        print(f"Test MAE: {mae:.4f}")
        return

    def R_for(self, train_x, train_y, test_x, test_y):
        rf = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
        rf.fit(train_x,np.ravel(train_y))
        preds = rf.predict(test_x)
        plots.plot_real_vs_pred(pd.DataFrame(preds), pd.DataFrame(test_y))
        oob_score = rf.oob_score_
        print(f'Out-of-Bag Score: {oob_score}')
        return