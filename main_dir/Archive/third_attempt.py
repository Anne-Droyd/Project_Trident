#Going to bench mark against the paper model
import os
import torch
import os
import copy
import pyro

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyro.distributions as dist
import torch.nn as nn
import pyro.optim as optim
import tensorflow as tf

from tkinter.filedialog import askdirectory, askopenfile
from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from pyro.nn import PyroModule, PyroSample
from exomdn.plotting import cornerplot, cornerplot_logratios
from keras.layers import TFSMLayer
from torch.distributions import constraints

class data_options:

    def __init__(self):
        self.save_dir = None

    def check_if_exists(self, path):
        return os.path.isfile(path)

    def get_save_folder(self):
        if self.save_dir:
            return self.save_dir
        else:
            self.save_dir = askdirectory(title="Choose Saving Directory")
            return self.save_dir

    def get_data(self):
        save_dir = self.get_save_folder()
        saved_file = str(save_dir + "/initial_data.pk")
        if self.check_if_exists(saved_file) == True:
            # user = input(f"Do you want to use {saved_file}? :")
            user = "y"
            if user.lower() == "yes" or user.lower() == "y":
                df = pd.read_pickle(saved_file)
                return df
            elif user.lower() == "no" or user.lower() == "n":
                print("Answered no, so a new file will be made")
                path = askopenfile(title="Select File")
                df = pd.read_csv(path, delimiter="\t")
                data_name = "/initial_data.pk"
                with open(str(save_dir + data_name), "wb") as file:
                    df.to_pickle(file)
                file.close()
                return df
            else:
                print(f"What do you want from me? Your answer: {user}")
                return None
        else:
            path = askopenfile(title="Select File")
            df = pd.read_csv(path, delimiter="\t")
            data_name = "/initial_data.pk"
            with open(str(save_dir + data_name), "wb") as file:
                df.to_pickle(file)
            file.close()
            return df

    def partition_data(self, data):
        if not self.save_dir:
            self.get_save_folder()
        # Define these variables before using them
        train_name = "training_data.pk"
        test_name = "testing_data.pk"
        valid_name = "validation_data.pk"

        train, valid, test = np.split(data.sample(frac=1), [int(0.6 * len(data)), int(0.8 * len(data))])
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)
        valid = pd.DataFrame(valid)
        with open(str(self.save_dir + "/" + train_name), "wb") as file:
            train.to_pickle(file)
        file.close()
        with open(str(self.save_dir + "/" + test_name), "wb") as file:
            test.to_pickle(file)
        file.close()
        with open(str(self.save_dir + "/" + valid_name), "wb") as file:
            valid.to_pickle(file)
        file.close()
        return train, test, valid

    def get_partitioned_data(self, data):
        train_name = "training_data.pk"
        test_name = "testing_data.pk"
        valid_name = "validation_data.pk"
        if self.check_if_exists(str(self.save_dir + "/" + train_name)) and \
                self.check_if_exists(str(self.save_dir + "/" + test_name)) and \
                self.check_if_exists(str(self.save_dir + "/" + valid_name)):
            # user = input("Partitioned data found, would you like to use it?: ")
            user = "y"
            if user.lower() == "yes" or user.lower() == "y":
                train = pd.read_pickle(str(self.save_dir + "/" + train_name))
                test = pd.read_pickle(str(self.save_dir + "/" + test_name))
                valid = pd.read_pickle(str(self.save_dir + "/" + valid_name))
                return train, test, valid
            elif user.lower() == "info":
                print(f"Training data: {str(self.save_dir + "/" + train_name)}")
                print(f"Testing data: {str(self.save_dir + "/" + test_name)}")
                print(f"Validation data: {str(self.save_dir + "/" + valid_name)}")
            else:
                train, test, valid = self.partition_data(data)
                return train, test, valid
        else:
            train, test, valid = self.partition_data(data)
            return train, test, valid

    def get_xy(self, dataframe, y_label, x_labels=None):
        dataframe = copy.deepcopy(dataframe)  # Avoid modifying the original DF

        if x_labels is None:
            X = dataframe.drop(columns=[y_label]).values  # Keep original shape
        else:
            X = dataframe[x_labels].values  # Keep original shape

        y = dataframe[y_label].values.reshape(-1, 1)  # Ensure y is 2D

        data = np.hstack((X, y))  # Ensure both arrays have matching row counts

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return  X, y

class Guide(PyroModule):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x, y=None):
        fc1w_loc = pyro.param("fc1w_loc", torch.randn_like(fc1.weight))
        fc1w_scale = pyro.param("fc1w_scale", torch.ones_like(fc1.weight), constraint=constraints.positive)
        pyro.sample("fc1.weight", dist.Normal(fc1w_loc, fc1w_scale).to_event(2))
        return


class BayesianNN(PyroModule):
    def __init__(self, in_features, hidden1_features, hidden2_features, hidden3_features, out_features):
        super().__init__()

        # First hidden layer
        self.fc1 = PyroModule[nn.Linear](in_features, hidden1_features)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden1_features, in_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden1_features]).to_event(1))

        # Second hidden layer
        self.fc2 = PyroModule[nn.Linear](hidden1_features, hidden2_features)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden2_features, hidden1_features]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden2_features]).to_event(1))

        # # Second hidden layer
        self.fc3 = PyroModule[nn.Linear](hidden2_features, hidden3_features)
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([hidden3_features, hidden2_features]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([hidden3_features]).to_event(1))

        # Output layer
        self.out = PyroModule[nn.Linear](hidden2_features, out_features)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, hidden2_features]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return self.out(x)


class Model(PyroModule):
    def __init__(self, in_size, out_size):
        super().__init__()
        hidden1 = 32
        hidden2 = 32
        hidden3 = 32
        self.net = BayesianNN(in_size, hidden1, hidden2, hidden3, out_size)
        self.obs_scale = PyroSample(prior=dist.LogNormal(-1., 0.5))

    def forward(self, input, output=None):
        obs_loc = self.net(input)
        obs_scale = self.obs_scale
        with pyro.plate("instances", len(input)):
            return pyro.sample("m_core", dist.Normal(obs_loc, obs_scale).to_event(1), obs=output)


def simple_model(data):
    data_ops = data_options()
    train, test, valid = data_ops.partition_data(data)
    train_x, train_y = data_ops.get_xy(train, "m_core", ["mass", "req", "Teq"])
    test_x, test_y = data_ops.get_xy(test, "m_core", ["mass", "req", "Teq"])

    train_x = train_x[:1000]
    train_y = train_y[:1000]
    test_x = test_x[:1000]
    test_y = test_y[:1000]

    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    x_scaler = StandardScaler()
    y_scaler = MinMaxScaler(feature_range=(1e-8, 1))

    # Convert to NumPy if Torch Tensors
    train_x_np = train_x.numpy() if isinstance(train_x, torch.Tensor) else train_x
    test_x_np = test_x.numpy() if isinstance(test_x, torch.Tensor) else test_x
    train_y_np = train_y.numpy() if isinstance(train_y, torch.Tensor) else train_y
    test_y_np = test_y.numpy() if isinstance(test_y, torch.Tensor) else test_y

    # Fit scalers on training data
    train_x_scaled = x_scaler.fit_transform(train_x_np)
    train_y_scaled_np = y_scaler.fit_transform(train_y_np.reshape(-1, 1))

    # Apply scaling to test data
    test_x_scaled = x_scaler.transform(test_x_np)
    test_y_scaled_np = y_scaler.transform(test_y_np.reshape(-1, 1))

    # Convert scaled data back to tensors
    train_x_scaled = torch.tensor(train_x_scaled, dtype=torch.float32)
    train_y_scaled = torch.tensor(train_y_scaled_np, dtype=torch.float32)
    test_x_scaled = torch.tensor(test_x_scaled, dtype=torch.float32)
    test_y_scaled = torch.tensor(test_y_scaled_np, dtype=torch.float32)

    # Define and train the model
    model = Model(train_x_scaled.shape[1],1)
    guide = AutoNormal(model)
    optimizer = pyro.optim.ClippedAdam({'lr': 1e-3, 'clip_norm': 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Training loop
    for step in range(10000):
        loss = svi.step(train_x_scaled, train_y_scaled) / len(train_y_scaled)
        if step % 100 == 0:
            test_loss = svi.evaluate_loss(test_x_scaled, test_y_scaled) / len(test_y_scaled)
            print(f"[Step {step}] Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")

    # Posterior predictive sampling
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000)
    samples = predictive(test_x_scaled)
    predictions = samples["m_core"].detach()
    mean_prediction = predictions.mean(dim=0)
    std_prediction = predictions.std(dim=0)

    # Inverse transform to original scale
    mean_prediction_original = y_scaler.inverse_transform(mean_prediction.numpy())
    std_prediction_original = std_prediction.numpy() * y_scaler.scale_

    # Calculate metrics
    test_y_actual = test_y_np.reshape(-1, 1)
    test_pred_mean = mean_prediction_original

    mse = np.mean((test_pred_mean - test_y_actual) ** 2)
    r2 = 1 - mse / np.var(test_y_actual)

    print(f"R² on test set: {r2:.4f}")
    print(f"MSE on test set: {mse:.4f}")

    # Sample prediction printout
    print("\nSample predictions vs actual values:")
    for i in range(min(10, len(test_y_actual))):
        actual = test_y_actual[i][0]
        predicted = test_pred_mean[i][0]
        std_dev = std_prediction_original[i][0]
        print(f"Actual: {actual:.4f}, Predicted: {predicted:.4f}, Std: {std_dev:.4f}")

    plt.scatter(test_y_np, mean_prediction_original)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Predictions vs Actuals")
    plt.show()

    return model, mean_prediction_original, std_prediction_original, x_scaler


def main():
    data_ops = data_options()
    data = data_ops.get_data()
    data_point = data.iloc[0][["mass", "req", "Teq", "m_core"]]

    print("Single data point:")
    print(data_point)

    model, mean_pred, std_pred, x_scaler = simple_model(data)

    print("Mean prediction (test set) shape:", mean_pred.shape)
    print("First few predictions:")
    for i in range(min(5, mean_pred.shape[0])):
        print(f"  {i}: {float(mean_pred[i][0]):.4f} ± {float(std_pred[i][0]):.4f}")


main()