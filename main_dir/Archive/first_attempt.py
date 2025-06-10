import torch
import os
import copy
import pyro

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyro.distributions as dist
import torch.nn as nn

from pyro.nn import PyroModule, PyroSample
from tkinter.filedialog import askdirectory, askopenfile
from pyro.infer import MCMC, NUTS, Predictive


class data_options:

    def __init__(self):
        self.save_dir=None

    def check_if_exists(self,path):
        return os.path.isfile(path)

    def get_save_folder(self):
        if self.save_dir:
            return self.save_dir
        else:
            self.save_dir = askdirectory(title="Choose Saving Directory")
            return self.save_dir

    def get_data(self):
        save_dir=self.get_save_folder()
        saved_file = str(save_dir+"/initial_data.pk")
        if self.check_if_exists(saved_file)==True:
            # user = input(f"Do you want to use {saved_file}? :")
            user="y"
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
            df =pd.read_csv(path,delimiter="\t")
            data_name= "/initial_data.pk"
            with open(str(save_dir+data_name),"wb") as file:
                df.to_pickle(file)
            file.close()
            return df

    def partition_data(self,data):
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

    def get_partitioned_data(self,data):
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
                train,test,valid = self.partition_data(data)
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

#basemodule is PyroModule
class BNN_models(PyroModule):

    #initialising the calss
    #2-5-1, prior scale being lower means the belief that the weights are close to 0, makes the network less flexible
    #large prior scale of 10 means the weights can be larger in magnitude
    #can result in more vairance and overfitting or unstable training
    def __init__(self, in_dim=3, out_dim=1, hid_dim=16, n_hid_layers=2, prior_scale=5):
        #super() calls the constructor of pyromodule to init the internal stuff for pyromodule
        #calls pyromodule.__init__()
        super().__init__()

        #okay to use this for in between layers
        #a non linear activation in hidden layers
        #squeezes values from -1->1
        #may want to change this to RELU as is more commonly used
        #or leaky relu to handle the dead nueron problem(no idea)
        #tanh gives better error than ReLU
        self.activation = nn.Tanh()

        # Input layer
        #creates a bayesian linear layer by wrapping nn.linear(in_dim,hid_dim) in pyromodules probablistic framework
        #this goes from the input to the hidden layer
        # self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)
        # #grabbing the weights and bias from a normal distribution of mean 0 and std prior scale
        # #to_event tells model how many dimensions are dependent
        # #expand reshapes the distribution to match the layers dimensions
        # self.layer1.weight = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, in_dim]).to_event(2))
        # self.layer1.bias = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))

        # Define the layer sizes and the PyroModule layer list
        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]
        layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_scale).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))

        # Output layer
        #this goes from the hidden layer to the output
        self.out = PyroModule[nn.Linear](hid_dim, out_dim)
        self.out.weight = PyroSample(dist.Normal(0., prior_scale).expand([out_dim, hid_dim]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., prior_scale).expand([out_dim]).to_event(1))

    # x is input data
    # y is target values used for training, could set this as y_train
    def forward(self, x, y=None):
        # apply bayesian linear layer to x, then applies the non linear tanh activation to this layer to make it non linear
        # x = self.activation(self.layers[0](x))
        #this takes the output of the hidden layer and passes it through to the output layer
        #squeeze flattens the data
        # mean = self.out(x).squeeze(-1)
        # #pick a random sigma from a uniform distribution
        # #I think this is a variable used later to train the system
        # #if we wanted to make this data dependent we would use
        # #sigma = torch.exp(self.sigma_layer(x))
        # #this is currently a simple model one sigma for all x
        # sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
        #
        # #tells pyro we're modelling a bunch of independent observations 1 per data point
        # with pyro.plate("data", x.shape[0]):
        #     # the true output of y is drawn from a normal distribution
        #     #if y is provided this is training,
        #     #if not this is sampling from the learned distribution
        #     obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        # return mean
        
        x = self.activation(self.layers[0](x))  # input --> hidden
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))  # hidden --> hidden
        mu = torch.nn.functional.softplus(self.out(x).squeeze(-1)) # hidden --> output
        sigma = pyro.sample("sigma", dist.HalfNormal(0.5))

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.LogNormal(mu, sigma), obs=y)
        return mu

    def load_model(self):
        saved_file = askopenfile()
        model = torch.load(saved_file, map_location=torch.device('cpu'))
        saved_file.close()
        return model

    def train_bnn_model(self,x_train,y_train):
        h=0

    def basic_bnn(self,x_train, y_train):
        user = input("Do you want to create a new model? :")
        if user.lower() == "yes" or user.lower() == "y":
            model_path = "basic_bnn_model.pt"
            model, history = self.train_bnn_model(x_train,y_train)
            save_path = data_ops.get_save_folder()
            model.save(save_path + "/" + model_path)
        else:
            model = self.load_model()
        return model

class plotting:

    def plot_predictions(self, x_test, y_test, preds, n=100):
        # Convert tensors to numpy
        x_test = x_test.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()

        # Use the first feature (e.g., 'mass') for the x-axis
        x_vals = sorted(x_test[:n, 0])

        y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
        y_std = preds['obs'].T.detach().numpy().std(axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.xlabel("mass", fontsize=30)
        plt.ylabel("m_core", fontsize=30)

        ax.scatter(x_vals, y_test[:n], marker='o',color="black", label="test")
        ax.errorbar(x_vals, y_pred[:n], yerr =2*y_std[:n],fmt="o", label="pred")

        plt.ylim(bottom=-0.5,top=2)
        plt.legend(loc=4, fontsize=15, frameon=False)
        plt.tight_layout()
        plt.show()

def main():
    mods = BNN_models()
    data_ops = data_options()
    data = data_ops.get_data()
    plots = plotting()
    train, test, valid = data_ops.get_partitioned_data(data)

    _, x_train_simple, y_train_simple = data_ops.get_xy(train,"m_core", ["mass","req","Teq"])
    _, x_valid_simple, y_valid_simple = data_ops.get_xy(valid,"m_core", ["mass","req","Teq"])
    _, x_test_simple, y_test_simple = data_ops.get_xy(test,"m_core", ["mass","req","Teq"])

    x_train_simple = torch.from_numpy(x_train_simple).float()
    y_train_simple = torch.from_numpy(y_train_simple).float()
    x_valid_simple = torch.from_numpy(x_valid_simple).float()
    y_valid_simple = torch.from_numpy(y_valid_simple).float()
    x_test_simple = torch.from_numpy(x_test_simple).float()
    y_test_simple = torch.from_numpy(y_test_simple).float()
    y_train_simple = torch.log(y_train_simple + 1e-6)
    # model = mods.basic_bnn(x_train_simple,y_train_simple)

    # Set Pyro random seed
    pyro.set_rng_seed(42)

    # Define Hamiltonian Monte Carlo (HMC) kernel
    # NUTS = "No-U-Turn Sampler" (https://arxiv.org/abs/1111.4246), gives HMC an adaptive step size
    nuts_kernel = NUTS(mods, jit_compile=True)  # jit_compile=True is faster but requires PyTorch 1.6+

    # Define MCMC sampler, get 10 posterior samples
    mcmc = MCMC(nuts_kernel, num_samples=10,warmup_steps=10)

    device = torch.device("cpu")
    # Move tensors to device
    x_train_simple = x_train_simple.to(device)

    # Run MCMC
    mcmc.run(x_train_simple[:1000], y_train_simple[:1000])

    predictive = Predictive(model=mods, posterior_samples=mcmc.get_samples())

    preds = predictive(x_test_simple)

    print(preds)

    plots.plot_predictions(x_test_simple,y_test_simple,preds)

    # Pick a test index
    test_idx = 0

    # Extract posterior samples for this point
    posterior_samples = preds["obs"][:, test_idx].detach().cpu().numpy()

    # Plot histogram of posterior
    plt.figure(figsize=(8, 5))
    plt.hist(posterior_samples, bins=30, color='skyblue', edgecolor='black', density=True)
    plt.title(f"Posterior distribution for test input {test_idx}")
    plt.xlabel("Predicted m_core")
    plt.ylabel("Density")

    # Optional: add true value line
    true_val = y_test_simple[test_idx].item()
    plt.axvline(true_val, color='red', linestyle='--', label=f"True: {true_val:.2f}")
    plt.legend()

    plt.tight_layout()
    plt.show()

main()