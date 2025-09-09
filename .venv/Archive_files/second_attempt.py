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

from pyro.nn import PyroModule, PyroSample
from tkinter.filedialog import askdirectory, askopenfile
from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO


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

        return data, X, y


class BNN(PyroModule):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.activation = nn.Tanh()
        self.layer_sizes = None
        self.layers = None

    def load_model(self):
        saved_file = askopenfile()
        model = torch.load(saved_file.name, map_location=torch.device(self.device))
        saved_file.close()
        return model

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def train_bnn_model(self, x, y, hid_dim, n_hid_layers, prior_scale, num_epochs=10):
        in_dim = x.shape[1]
        out_dim = y.shape[1]
        # init the layers
        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]

        layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_scale).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))

        guide = pyro.infer.autoguide.AutoDiagonalNormal(self)

        # Optimizer and loss
        pyro.clear_param_store()
        optimizer = optim.Adam({"lr": 0.01})
        svi = SVI(self, guide, optimizer, loss=Trace_ELBO())

        loss_history = []

        # Training loop
        for epoch in range(num_epochs):
            loss = svi.step(x, y)
            loss_history.append(loss)
            if epoch % 1 == 0:
                print(f"[Epoch {epoch}] Loss: {loss:.4f}")

        return self, guide, loss_history

    def forward(self, x, y=None):
        x = x.reshape(-1, self.layer_sizes[0])  # Ensure correct input shape
        x = self.activation(self.layers[0](x))  # input --> hidden
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))  # hidden --> hidden
        mu = self.layers[-1](x).squeeze()  # hidden --> output
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1))  # infer the response noise

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu

    def basic_bnn(self, x_train, y_train, data_ops):
        user = input("Do you want to create a new model? :")
        if user.lower() == "yes" or user.lower() == "y":
            model_path = "basic_bnn_model.pt"
            best_model = None
            best_guide = None
            best_loss = np.inf
            best_mcmc = None
            hid_dims = [1, 2, 4, 8, 16, 32]
            num_hid_layers = [1, 2, 4, 8]  # Reduced the search space for efficiency
            prior_scales = [0.5, 1, 2, 4]  # Reduced the search space for efficiency

            for prior_scale in prior_scales:
                for hid_dim in hid_dims:
                    for n_hid_layers in num_hid_layers:
                        print(
                            f"Training with prior_scale={prior_scale}, hid_dim={hid_dim}, n_hid_layers={n_hid_layers}")

                        # Create a new model instance for each configuration
                        model = BNN()
                        model, guide, history = model.train_bnn_model(x_train, y_train, hid_dim, n_hid_layers,
                                                                      prior_scale)
                        loss = history[-1]

                        # Create and run MCMC
                        nuts_kernel = NUTS(model, jit_compile=False)
                        mcmc = MCMC(nuts_kernel, num_samples=40)
                        mcmc.run(x_train, y_train)

                        if loss < best_loss:
                            best_loss = loss
                            best_model = model
                            best_guide = guide
                            best_mcmc = mcmc
                            print(f"New best model found! Loss: {best_loss:.4f}")

            save_path = data_ops.get_save_folder()
            best_model.save_model(save_path + "/" + model_path)
            return best_model, best_mcmc
        else:
            model = self.load_model()
            # Note: We would need to run MCMC again for this loaded model
            nuts_kernel = NUTS(model, jit_compile=False)
            mcmc = MCMC(nuts_kernel, num_samples=40)
            mcmc.run(x_train, y_train)
            return model, mcmc


def plot_predictions(preds, x_test, y_test):
    """
    Plot the predictions from the Bayesian neural network.

    Args:
        preds: Dictionary containing predictions
        x_test: Test input data
        y_test: True test output data
    """
    # Extract predictions
    y_pred = preds['obs'].mean(0)
    y_pred_std = preds['obs'].std(0)

    # Sort by x for better visualization
    sorted_indices = torch.argsort(x_test[:, 0])
    x_sorted = x_test[sorted_indices]
    y_true_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    y_pred_std_sorted = y_pred_std[sorted_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_sorted[:, 0].numpy(), y_true_sorted.numpy(), alpha=0.6, label='True values')
    plt.plot(x_sorted[:, 0].numpy(), y_pred_sorted.numpy(), 'r-', label='Predicted mean')
    plt.fill_between(
        x_sorted[:, 0].numpy(),
        (y_pred_sorted - 2 * y_pred_std_sorted).numpy(),
        (y_pred_sorted + 2 * y_pred_std_sorted).numpy(),
        alpha=0.2, color='r', label='95% confidence interval'
    )
    plt.legend()
    plt.xlabel('Input feature')
    plt.ylabel('Target value')
    plt.title('BNN Predictions with Uncertainty')
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    mods = BNN()
    data_ops = data_options()
    data = data_ops.get_data()
    train, test, valid = data_ops.get_partitioned_data(data)

    _, x_train_simple, y_train_simple = data_ops.get_xy(train, "m_core", ["mass", "req"])
    _, x_valid_simple, y_valid_simple = data_ops.get_xy(valid, "m_core", ["mass", "req"])
    _, x_test_simple, y_test_simple = data_ops.get_xy(test, "m_core", ["mass", "req"])

    x_train_simple = torch.from_numpy(x_train_simple).float()
    y_train_simple = torch.from_numpy(y_train_simple).float()
    x_valid_simple = torch.from_numpy(x_valid_simple).float()
    y_valid_simple = torch.from_numpy(y_valid_simple).float()
    x_test_simple = torch.from_numpy(x_test_simple).float()
    y_test_simple = torch.from_numpy(y_test_simple).float()

    best_model, mcmc = mods.basic_bnn(x_train_simple, y_train_simple, data_ops)

    # Make predictions
    predictive = Predictive(model=best_model, posterior_samples=mcmc.get_samples())
    preds = predictive(x_test_simple)

    # Plot the results
    plot_predictions(preds, x_test_simple, y_test_simple)


if __name__ == "__main__":
    main()