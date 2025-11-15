"""
Clean version
"""
import warnings
from importlib.metadata import metadata
from tkinter.filedialog import askopenfile, askopenfilename
from sklearn.metrics import mean_squared_error, r2_score

import h5py
import json
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from main_dir.Methods import Plotting as plots
from keras import callbacks, Sequential, regularizers
from main_dir.Methods.Data_Options import data_options
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from main_dir.Methods.MDN import get_mixture_loss_func as MDNLoss
from main_dir.Methods.MDN import mdn_loss_with_entropy as EntropyLoss
from main_dir.Methods.MDN import MDN, softmax, sample_from_output

def unpack_mdn_predictions(predictions, output_dim, num_mixtures):
    out_mu, out_sigma, out_pi = np.split(predictions,indices_or_sections=[
                                        num_mixtures * output_dim,
                                        2 * num_mixtures * output_dim],axis=-1)

    # Reshape:
    mus = np.reshape(out_mu, (-1, num_mixtures, output_dim))        # (samples, mixtures, outputs)
    sigmas = np.reshape(out_sigma, (-1, num_mixtures, output_dim))  # (samples, mixtures, outputs)
    pi = np.reshape(out_pi, (-1, num_mixtures))                      # (samples, mixtures)

    return mus, sigmas, pi

def sample_from_mixture(mu, sigma, pi, num_samples=1):
    """
    gather N samples per predicted output
    pi must be softmaxed first not in logits
    """
    all_samples = []
    for i in range(mu.shape[0]):  # for each test point
        samples =[]
        for _ in range(num_samples):
            #pick a weighted random mu from the row given the certainty of the gaussian
            component_idx = np.random.choice(mu.shape[1], p=pi[i])
            sample = np.random.normal(loc=mu[i, component_idx], scale=sigma[i, component_idx])
            samples.append(sample)
        all_samples.append(samples)
    return np.array(all_samples)

class MDN_trainer():
    """
    Made this a seperate class so I can create new instances without recreating the base data class allowing for simpler
    experimentation and iterating. Initializing creates the base model, train
    """
    def __init__(self,
                 input_dimension,
                 output_dimension,
                 number_mixtures=20,
                 number_layers=4,
                 number_nodes=60,
                 activation="relu",
                 bias_initalizer = None,
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 activity_regularizer = None,
                 optimizer=Adam,
                 learning_rate=0.001,
                 loss_function=MDNLoss,
                 **kwargs):

        self.input_dim =    input_dimension
        self.output_dim =   output_dimension
        self.num_mixes =    number_mixtures
        self.num_layers =   number_layers
        self.num_nodes =    number_nodes
        self.activ =        activation
        self.bias_init =    bias_initalizer
        self.bias_reg =     bias_regularizer
        self.kernel_reg =   kernel_regularizer
        self.act_reg =      activity_regularizer
        self.lr =           learning_rate
        self.optim =        optimizer(learning_rate=self.lr)
        self.loss =         loss_function(self.output_dim,self.num_mixes)

        self.create_model()


    def create_model(self):
        self.model = Sequential()
        for layer_num in range(self.num_layers):
            self.model.add(Dense(self.num_nodes, input_shape=(self.input_dim,), activation=self.activ,
                            bias_initializer=self.bias_init, kernel_regularizer=self.kernel_reg,
                            bias_regularizer=self.bias_reg, activity_regularizer=self.act_reg))
        self.model.add(MDN(self.output_dim, self.num_mixes))
        self.model.compile(loss=self.loss, optimizer=self.optim)


    def fit_model(self,
                    train_x,
                    train_y,
                    valid_x,
                    valid_y,
                    epochs=50,
                    batch_size=100,
                    **kwargs):

        self.history = self.model.fit(train_x,
                                      train_y,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=(valid_x, valid_y),
                                      callbacks=[
                                callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                                callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4)
                                ])

        return self.history, self.model

class MK_VI_MDN():

    def __init__(self,
                 input_names,
                 output_names,
                 default_data=True,
                 dataset="VR_DATA", #currently only support "VR_DATA" or "BAU", will include "live" later as default
                 scaler = StandardScaler(),
                 file_prefix="MDN", # What the model file name will contain
                 select_model_path=False, #option to select the directory for the model, if false its default
                 select_history_path=False, #option to select the directory for the history, if false its default
                 **kwargs):

        x_col_no_k2 = ["mass","radius","temp"]
        x_col_k2 = ["mass","radius","temp","k2"]
        #if dataset="live":

        self.scaler= scaler
        self.file_prefix = file_prefix
        assert dataset == "VR_DATA" or dataset == "BAU", "Unknown dataset"

        self.data_ops = data_options(type=dataset,default_data=default_data)
        self.data = self.data_ops.get_data()
        #smoke test
        # self.data=self.data[:50]


        if dataset=="VR_DATA":
            self.data.rename(columns={"req": "radius", "Teq": "temp"})
        elif dataset=="BAU":
            self.data.rename(columns={'planet_mass': 'mass', 'planet_radius': 'radius', 'T_eq': 'temp'})

        self.train, self.test, self.valid = self.data_ops.partition_data(self.data,
                                                                         train_frac=0.8,
                                                                         test_frac=0.1,
                                                                         valid_frac=0.1)
        self.x_col, self.y_col = input_names, output_names
        self.train_x, self.train_y = self.data_ops.get_xy(self.train, self.y_col, self.x_col)
        self.test_x, self.test_y = self.data_ops.get_xy(self.test, self.y_col, self.x_col)
        self.valid_x, self.valid_y = self.data_ops.get_xy(self.valid, self.y_col, self.x_col)

        self.input_dimension, self.output_dimension = self.train_x.shape[1], self.train_y.shape[1]

        if dataset != "live":
            self.x_scaler = self.scaler.__class__()
            self.y_scaler = self.scaler.__class__()
            self.train_x = self.x_scaler.fit_transform(self.train_x)
            self.train_y = self.y_scaler.fit_transform(self.train_y)
            # Do not refit the transform
            self.valid_x = self.x_scaler.transform(self.valid_x)
            self.valid_y = self.y_scaler.transform(self.valid_y)
            self.test_x = self.x_scaler.transform(self.test_x)
            self.test_y = self.y_scaler.transform(self.test_y)
        else:
            if x_col==x_col_no_k2:
                self.x_scaler = load_no_k2_scaler()
                self.test_x = self.x_scaler.transform(self.test_x)
            elif x_col==x_col_k2:
                self.x_scaler = load_k2_scaler()
                self.test_x = self.x_scaler.transform(self.test_x)


        if select_model_path == False:
            self.model_folder_path = "C:/Users/Matth/Documents/Leiden University/Project/models/MDN/"

        elif select_model_path == True:
            self.model_folder_path = askdirectory()

        if select_history_path == False:
            self.history_path = "C:/Users/Matth\Documents/Leiden University/Project/Histories/MDN/"

        elif select_history_path == True:
            self.history_path = askdirectory()

        self.current_iteration = self.data_ops.get_iteration(self.model_folder_path, file_prefix=self.file_prefix)


    def train_model(self,
                    number_mixtures=10,
                    number_layers=1,
                    number_nodes=60,
                    activation="relu",
                    bias_initalizer = None,
                    kernel_regularizer = None,
                    bias_regularizer = None,
                    activity_regularizer = None,
                    optimizer=Adam,
                    learning_rate=0.001,
                    loss_function=MDNLoss,
                    epochs = 50,
                    batch_size = 100,
                    **kwargs):
        self.num_mixes = number_mixtures
        trainer = MDN_trainer(input_dimension = self.input_dimension,
                                output_dimension = self.output_dimension,
                                number_mixtures = number_mixtures,
                                number_layers = number_layers,
                                number_nodes = number_nodes,
                                activation = activation,
                                bias_initalizer = bias_initalizer,
                                kernel_regularizer = kernel_regularizer,
                                bias_regularizer = bias_regularizer,
                                activity_regularizer = activity_regularizer,
                                optimizer = optimizer,
                                learning_rate = learning_rate,
                                loss_function = loss_function)
        history, model =trainer.fit_model(self.train_x,
                                                self.train_y,
                                                self.valid_x,
                                                self.valid_y,
                                                epochs,
                                                batch_size)
        Model_save_name = f"{self.file_prefix}_model_{self.current_iteration}.h5"
        history_name = f"{self.file_prefix}_history_model_{self.current_iteration}.csv"
        #add metadata to file for easy retracability
        model.save(self.model_folder_path + Model_save_name,include_optimizer=False)
        # then open and add metadata
        with h5py.File(self.model_folder_path + Model_save_name, "a") as file:
            file.attrs["input_dimension"] = self.input_dimension
            file.attrs["output_dimension"] = self.output_dimension
            file.attrs["number_of_mixtures"] = number_mixtures
            file.attrs["input_parameters"] = json.dumps(self.x_col)
            file.attrs["output_parameters"] = json.dumps(self.y_col)
            file.attrs["number_of_layers"] = number_layers
            file.attrs["number_of_nodes"] = number_nodes
            file.attrs["activation"] = str(activation)
            file.attrs["optimizer"] = str(optimizer)
            file.attrs["learning_rate"] = learning_rate
            file.attrs["loss_function"] = str(loss_function)
            file.attrs["bias_initalizer"] = str(bias_initalizer)
            file.attrs["kernel_regularizer"] = str(kernel_regularizer)
            file.attrs["bias_regularizer"] = str(bias_regularizer)
            file.attrs["activity_regularizer"] = str(activity_regularizer)


        history_df = pd.DataFrame(history.history)
        # Save to CSV
        history_df.to_csv(self.history_path + history_name, index=False)

        return model, history_df

    def load_model(self,input_names,output_names,model_name = "Select"):
        """
        Model 1 will be mass, radius, temp as inputs and whatever outputs for live service
        Model 2 will include K2 in the inputs as well as whatever outputs for model 1
        Select will allow the user to pick which model and history from their files
        """
        model_name = model_name.lower()
        if model_name == "mdn_model_1":
            Model_save_name = ""
            history_name = ""

        elif model_name == "mdn_model_2":
            Model_save_name = ""
            history_name = ""

        elif model_name == "select":
            Model_save_name = askopenfilename()
            history_name = askopenfilename()

        elif model_name =="latest":
            if self.current_iteration > 1:
                latest_iteration = self.current_iteration-1
            Model_save_name = f"{self.file_prefix}_model_{latest_iteration}.h5"
            history_name = f"{self.file_prefix}_history_model_{latest_iteration}.csv"

        with h5py.File(self.model_folder_path + Model_save_name, "r") as file:
            if not file.attrs["input_dimension"]:
                raise warnings.warn("Metadata missing from the model file.")
                input_dim, output_dim = len(input_names), len(output_names)
                metadata_indicator = False
            else:
                metadata_indicator = True
                input_dim = file.attrs["input_dimension"]
                output_dim = file.attrs["output_dimension"]
                self.num_mixes = file.attrs["number_of_mixtures"]
                x_col = json.loads(file.attrs["input_parameters"])
                y_col = json.loads(file.attrs["output_parameters"])
                loss = file.attrs["loss_function"]
                optim = file.attrs["optimizer"]
                lr = file.attrs["learning_rate"]
                assert x_col == input_names, f"Inputs don't match:\n data input {x_col},\n model input {input_names}"
                assert y_col == output_names, f"Outputs don't match:\n data output {y_col},\n model output {output_names}"

        optim=optim(learning_rate=lr)

        model = keras.models.load_model(
            self.model_folder_path + Model_save_name,
            custom_objects={"MDN": MDN}
        )
        model.compile(loss=loss(self.output_dimension, self.num_mixes), optimizer=optim)
        #load metadata if available

        if not metadata_indicator:
            print(model.summary())
            print("Input shape :", model.input_shape())
            print("Output shape :", model.output_shape())
            print("input dim: ", input_dim)
            print("output dim: ", output_dim)
            if input_dim == model.input_shape()[1]:
                print("Input dimensions match")
            else:
                raise ValueError("Input dimensions do not match")
            if output_dim == model.output_shape()[1]:
                print("Output dimensions match")
            else:
                raise ValueError("Output dimensions do not match")

        history_df = pd.read_csv(self.history_path + history_name)

        return model, history_df

    def prediction(self,model):
        predictions = model.predict(self.test_x)
        mu, sigma, pi = unpack_mdn_predictions(predictions, self.output_dimension, self.num_mixes)
        mu_flat = mu.reshape(-1, self.output_dimension)
        mu = self.y_scaler.inverse_transform(mu_flat)
        mu = mu.reshape(-1, self.num_mixes, self.output_dimension)

        scale = self.y_scaler.scale_
        sigma = sigma * scale.reshape(1, 1, -1)

        pi = np.apply_along_axis(softmax, 1, pi, temperature=1)
        return mu, sigma, pi

    def plot_pdf(self,mu,sigma,pi,means,mu_mixture,mu_top5_avg,var_mixture,top5_mus):
        test=pd.DataFrame(self.y_scaler.inverse_transform(self.test_y))
        plots.plot_single_pdf(mu, sigma, pi, self.y_col,
                              test_y=test,
                              mean_pred=means,
                              means=mu_mixture,
                              mu_map=mu_top5_avg,
                              vars=var_mixture,
                              top5_mus=top5_mus,
                              idx=np.random.randint(0, mu.shape[0]))
    def plot_real_v_pred(self,mean):
        test = pd.DataFrame(self.y_scaler.inverse_transform(self.test_y))
        plots.plot_predicted_vs_real_scatter(mean, test)

    def plot_predicted_vs_real_hist(self,predictions,plot_path):

        for idx, col in enumerate(self.y_col):
            iter = self.data_ops.get_iteration(plot_path, file_prefix=col)
            fig = plt.figure(figsize=(6, 4))
            mse = mean_squared_error(self.test_y[:, idx], predictions[:, idx])
            print(f'{col} Mean Squared Error: {mse}')

            r2 = r2_score(self.test_y[:, idx], predictions[:, idx])
            print(f'{col} R-squared: {r2}')
            min_val = self.test_y[:, idx].min()
            max_val = self.test_y[:, idx].max()
            h = plt.hist2d(self.test_y[:, idx], predictions[:, idx], bins=100, cmap="jet", cmax=100, density=True)
            fig.colorbar(h[3], label="Density")
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='Perfect prediction')
            plt.title(f'Real vs predicted values for {col}')
            image_path = plot_path + f'real_vs_pred_{col}_{iter}.png'
            plt.text(
                0.05, 0.95, f'$R^2 = {r2:.3f}$',
                transform=plt.gca().transAxes,  # place relative to axes (0–1)
                fontsize=12,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
            )
            plt.text(
                0.05, 0.85, f'$MSE = {mse:.3f}$',
                transform=plt.gca().transAxes,  # place relative to axes (0–1)
                fontsize=12,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
            )
            plt.ylim(min_val, max_val)
            plt.xlim(min_val, max_val)
            plt.legend()
            plt.tight_layout()
            plt.savefig(image_path)
            plt.show()

def main():

    #smoke test
    epochs = 100
    lr = 0.001
    num_mixtures_1 = 20
    batch_size_1 = 128
    num_hidden_nodes_1 = 60

    #model params
    bias_init="zeros"
    activation="relu"

    y_col = ["m_core","ice_mass","rock_mass","h_he_mass"]
    x_col = ["mass","radius","temp"]

    optimizer = Adam(learning_rate=lr)

    main_mdn_class = MK_VI_MDN(input_names=x_col,output_names=y_col,default_data=True)



    model, history = main_mdn_class.train_model(number_mixture = num_mixtures_1,batch_size=batch_size_1,
                                                number_layers=1,number_nodes=num_hidden_nodes_1,
                                                epochs=epochs,learning_rate=0.001)

    mu, sigma, pi = main_mdn_class.prediction(model)
    plot_path = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/plots/MDN/'
    #smoke test
    samples = plots.sample_from_mixture(mu, sigma, pi, 500)
    pred_mean = np.mean(samples, axis=1)
    main_mdn_class.plot_predicted_vs_real_hist(pred_mean, plot_path)
    #
    #
    #
    #
    # means = np.sum(pi[..., np.newaxis] * mu, axis=1)
    # map_indices = np.argsort(pi, axis=1)[:,-1]
    #
    # top5_indices = np.argsort(pi, axis=1)[:, -5:]
    # batch_indices = np.arange(mu.shape[0])[:, None]  # shape (n_samples, 1)
    # top5_mus = mu[batch_indices, top5_indices]  # shape (n_samples, 5, output_dim)
    #
    # # Step 3: Average them
    # mu_top5_avg = np.mean(top5_mus, axis=1)
    # top5_pis = pi[batch_indices, top5_indices]  # shape (n_samples, 5)
    # weighted_avg = np.sum(top5_mus * top5_pis[..., None], axis=1) / np.sum(top5_pis, axis=1, keepdims=True)
    #
    # mu_map = np.array([mu[i, idx, :] for i, idx in enumerate(map_indices)])
    # mu_mixture = np.sum(pi[:, :, np.newaxis] * mu, axis=1)  # Shape: (17506, 5)
    #
    # # Broadcasting mu_mixture to (17506, 120, 5) for element-wise subtraction with mu
    # mu_mixture_broadcasted = mu_mixture[:, np.newaxis, :]  # Shape: (17506, 1, 5)
    #
    # # Now calculate variance
    # var_mixture = np.sum(pi[:, :, np.newaxis] * (sigma ** 2 + (mu - mu_mixture_broadcasted) ** 2), axis=1)

    # main_mdn_class.plot_pdf(mu,sigma,pi,means,mu_mixture,mu_top5_avg,var_mixture,top5_mus)
    # main_mdn_class.plot_real_v_pred(means)
    # plt.scatter(range(num_mixtures_1),pi[np.random.randint(0, mu.shape[0]),:])
    # plt.ylabel()
    # plt.show()

    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, means, mu_mixture, var_mixture, idx=0)
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=1)
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=2)
    #
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=3)
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=4)
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=5)

    # plots.plot_mu_vs_real_with_errorbars(mu_mixture, var_mixture, test_y, y_col)


    # plots.plot_predicted_vs_real_scatter(best_mean,test_y)
    # plots.plot_predicted_vs_real_scatter(samples,test_y)

main()