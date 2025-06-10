#Going to bench mark against the paper model

import torch
import pyro

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyro.distributions as dist
import torch.nn as nn
import pyro.optim as optim
import torch.nn.functional as F
import Plotting as plots
import tensorflow as tf

from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
from pyro.optim import Adam
from pyro.nn import PyroModule, PyroSample
from torch.distributions import constraints
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Data_Options import data_options
from scipy import stats



class Guide(PyroModule):
    #this is for later use when optimization is needed
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

        # Second hidden layer
        self.fc3 = PyroModule[nn.Linear](hidden2_features, hidden3_features)
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([hidden3_features, hidden2_features]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([hidden3_features]).to_event(1))

        # Output layer
        self.out = PyroModule[nn.Linear](hidden3_features, out_features)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, hidden3_features]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return self.out(x)

# Custom ELBO loss class using MSE
class MSE_ELBO(Trace_ELBO):
    def loss(self, model, guide, num_particles, *args, **kwargs):
        # Compute the regular ELBO loss
        elbo_loss = super().loss(model, guide, num_particles, *args, **kwargs)

        # Compute Mean Squared Error (MSE) between predicted and true values
        predicted = model(*args, **kwargs)
        true_values = kwargs.get('true_values')  # assuming 'true_values' are passed as kwargs
        mse_loss = F.mse_loss(predicted, true_values)

        # Combine the ELBO loss and MSE loss
        return elbo_loss + 0.1 * mse_loss  # Scaling factor for MSE

class Model(PyroModule):
    def __init__(self, in_size, out_size):
        super().__init__()
        hidden1 = 128
        hidden2 = 128
        hidden3 = 128
        #maybe a beta distribution is what I should use
        self.net = BayesianNN(in_size, hidden1, hidden2, hidden3, out_size)
        self.obs_scale = PyroSample(prior=dist.Uniform(1e-8, 1))

    def forward(self, input, output=None):
        obs_loc = self.net(input)
        obs_scale = self.obs_scale
        with pyro.plate("instances", len(input)):
            sample = pyro.sample("m_core", dist.LogNormal(obs_loc, obs_scale).to_event(1), obs=output)

            return sample

def simple_model(data):
    train, test, valid = data_ops.partition_data(data)
    train_x, train_y = data_ops.get_xy(train, "m_core", ["mass", "req", "Teq"])
    test_x, test_y = data_ops.get_xy(test, "m_core", ["mass", "req", "Teq"])
    valid_x, valid_y = data_ops.get_xy(valid, "m_core", ["mass", "req", "Teq"])

    train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, x_scaler, y_scaler = data_ops.scale_data(train_x,train_y,test_x,test_y)

    model = Model(train_x_scaled.shape[1], 1)
    guide = AutoDiagonalNormal(model)
    optimizer = Adam({'lr': 1e-3})
    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=1))

    train_x_scaled=torch.tensor(train_x_scaled,dtype=torch.float32)
    train_y_scaled=torch.tensor(train_y_scaled,dtype=torch.float32)
    test_x_scaled=torch.tensor(test_x_scaled,dtype=torch.float32)
    test_y_scaled=torch.tensor(test_y_scaled,dtype=torch.float32)


    num_steps=100
    n_samples = train_x_scaled.shape[0]
    losses=[]
    for step in range(num_steps):
        loss = svi.step(train_x_scaled,train_y_scaled)/n_samples
        losses.append(loss)
        if step % 100 == 0:
            print("step {} loss = {:0.4g}".format(step, loss))
        if loss<=1.1:
            num_steps=step+1
            break


    plots.plot_loss(range(num_steps),losses)

    # Posterior predictive sampling
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=2000)
    samples = predictive(test_x_scaled)
    predictions = samples["m_core"].detach().squeeze(-1)

    # Inverse transform to original scale
    predictions = y_scaler.inverse_transform(predictions.numpy())

    mean_prediction = pd.DataFrame(predictions.mean(axis=0),columns=["m_core_mean"])
    median_prediction = pd.DataFrame(np.median(predictions,axis=0),columns=["m_core_median"])
    mode_prediction = pd.DataFrame(stats.mode(predictions,keepdims=False).mode)
    print("mode",mode_prediction)

    predictions = pd.DataFrame(predictions)

    test_x = pd.DataFrame(test_x)
    test_y = pd.DataFrame(test_y)

    idx = 0
    data_point = data.iloc[idx][["mass", "req", "Teq", "m_core"]]
    plots.plot_posterior(predictions,test_y,data_point)
    plots.plot_real_vs_pred(median_prediction,test_y)
    plots.plot_real_vs_pred(median_prediction,test_y)
    plots.plot_real_vs_pred(mode_prediction,test_y)

    return model

data_ops=data_options()
def main():
    data = data_ops.get_data()

    model = simple_model(data)

main()