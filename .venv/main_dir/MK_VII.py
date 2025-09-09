"""
This attempt will focus on variational auto encoders
"""

import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from main_dir.Methods import Plotting as plots
from main_dir.Methods.Data_Options import data_options
from main_dir.Methods.VAE import VAE, Encoder, Decoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


data_ops = data_options(type="VR_DATA")
model_folder_path = "C:/Users/Matth/Documents/Leiden University/Project/models/VAE/"
history_path = "C:/Users/Matth/Documents/Leiden University/Project/Histories/VAE/"

def plot_reconstruction(original, reconstructed, feature_names):
    n_features = original.shape[1]
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))

    if n_features == 1:
        axes = [axes]  # Handle single feature case

    for i in range(n_features):
        axes[i].scatter(original[:, i], reconstructed[:, i], alpha=0.5)
        axes[i].plot([original[:, i].min(), original[:, i].max()],
                     [original[:, i].min(), original[:, i].max()],
                     'r--', linewidth=2)
        axes[i].set_xlabel(f"Real {feature_names[i]}")
        axes[i].set_ylabel(f"Reconstructed {feature_names[i]}")
        axes[i].set_title(f"{feature_names[i]} Reconstruction")

    plt.tight_layout()
    plt.show()

def MSE_loss_function(x, x_hat, mu, logvar, beta=0.01):
    # reconstruction loss
    MSE = nn.functional.mse_loss(x_hat, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

def BCE_loss_function(x, x_hat, mu, logvar, beta=0.001):
    BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

def train_VAE_model(model,train_loader,val_loader,optimizer,device,epochs=10):
    model.train()
    history_list=[]
    for epoch in range(epochs):
        dictionary = {}
        dictionary["Epoch"] = epoch
        training_loss = 0
        validation_loss = 0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss = MSE_loss_function(x, x_hat, mu, logvar)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                loss = MSE_loss_function(x, x_hat, mu, logvar)
                validation_loss += loss.item()
        avg_train_loss = training_loss / len(train_loader.dataset)
        avg_val_loss = validation_loss / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        dictionary["Training loss"] = avg_train_loss
        dictionary["Validation loss"] = avg_val_loss
        history_list.append(dictionary)
    history = pd.DataFrame(history_list)

    return history, model


class LatentRegressor(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, z):
        return self.net(z)

def train_NN_model(model,latent_space_1,latent_space_2,optimizer,epochs=50):
    h=0

def train_latent_regressor(input_model, output_model, regressor,
                           train_x_tensor, train_y_tensor, optimizer, device, epochs=50):
    regressor.train()
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Encode inputs
        mu_x, logvar_x = input_model.encoder(train_x_tensor.to(device))
        z_x = mu_x  # deterministic option

        # Encode outputs
        mu_y, logvar_y = output_model.encoder(train_y_tensor.to(device))
        z_y = mu_y

        # Predict z_y from z_x
        z_y_pred = regressor(z_x)

        # Loss = MSE in latent space
        loss = nn.functional.mse_loss(z_y_pred, z_y)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Latent regression loss: {loss.item():.4f}")
    return history

def predict_from_x(input_model, output_model, regressor, x_tensor, device):
    with torch.no_grad():
        mu_x, logvar_x = input_model.encoder(x_tensor.to(device))
        z_x = mu_x
        z_y_pred = regressor(z_x)
        y_hat = output_model.decoder(z_y_pred)
    return y_hat.cpu().numpy()

def main():
    data = data_ops.get_data()
    # could also use data = data.sample(frac=1).reset_index(drop=True)
    data = shuffle(data,random_state=42)
    data = data.rename(columns={"req": "radius", "Teq": "temp"})
    train, test, valid = data_ops.partition_data(data,train_frac=0.8,test_frac=0.1,valid_frac=0.1)



    #automatically label models
    input_iteration = data_ops.get_iteration(model_folder_path,"input")
    NN_iteration = data_ops.get_iteration(model_folder_path,"NN")
    output_iteration = data_ops.get_iteration(model_folder_path,"output")
    print("Current input model iteration #",input_iteration)
    print("Current NN model iteration #",NN_iteration)
    print("Current output model iteration #",output_iteration)

    #setting paths for saving
    input_model_save_name = f"VAE_input_model_{input_iteration}.h5"
    NN_model_save_name = f"VAE_NN_model_{NN_iteration}.h5"
    output_model_save_name = f"VAE_output_model_{output_iteration}.h5"
    input_history_name = f"input_history_{input_iteration}.csv"
    NN_history_name = f"NN_history_{NN_iteration}.csv"
    output_history_name = f"output_history_{output_iteration}.csv"

    y_col = ["m_core", 'zatm', 'zdeep']
    x_col = ["mass","radius","temp"]

    train_x, train_y = data_ops.get_xy(train, y_col, x_col)
    test_x, test_y = data_ops.get_xy(test, y_col, x_col)
    valid_x, valid_y = data_ops.get_xy(valid, y_col, x_col)

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    train_x = x_scaler.fit_transform(train_x)
    train_y = y_scaler.fit_transform(train_y)
    valid_x = x_scaler.transform(valid_x)
    valid_y = y_scaler.transform(valid_y)
    test_x = x_scaler.transform(test_x)
    test_y = y_scaler.transform(test_y)

    # Hyperparameters
    input_dim = train_x.shape[1]
    output_dim = train_y.shape[1]
    load_input_model = "n"
    load_NN_model = "n"
    load_output_model = "n"
    hidden_dim = 400
    latent_dim = 20
    lr = 1e-3
    batch_size = 128
    epochs = 10
    device = "cpu"

    # Convert pandas DataFrames -> numpy -> torch tensors
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
    valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32)
    valid_y_tensor = torch.tensor(valid_y, dtype=torch.float32)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

    train_x_loader = DataLoader(train_x_tensor, batch_size=batch_size,shuffle=True)
    train_y_loader = DataLoader(train_y_tensor, batch_size=batch_size,shuffle=True)
    valid_x_loader = DataLoader(valid_x_tensor, batch_size=batch_size,shuffle=True)
    valid_y_loader = DataLoader(valid_y_tensor, batch_size=batch_size,shuffle=True)
    test_x_loader = DataLoader(test_x_tensor, batch_size=batch_size,shuffle=True)
    test_y_loader = DataLoader(test_y_tensor, batch_size=batch_size,shuffle=True)


    input_vae = VAE(input_dim,hidden_dim,latent_dim)
    output_vae = VAE(output_dim,hidden_dim,latent_dim)

    optimizer_x = optim.Adam(input_vae.parameters(), lr=lr)
    optimizer_y = optim.Adam(output_vae.parameters(), lr=lr)

    X_history, input_model = train_VAE_model(input_vae,train_x_loader,valid_x_loader,optimizer_x,device,epochs)
    Y_history, output_model = train_VAE_model(output_vae,train_y_loader,valid_y_loader,optimizer_y,device,epochs)

    with torch.no_grad():
        x_hat, _, _ = input_model(test_x_tensor.to(device))
        y_hat ,_, _ = output_model(test_y_tensor.to(device))
        reconstructed_x = x_hat.cpu().numpy()
        reconstructed_y = y_hat.cpu().numpy()
    plot_reconstruction(test_x, reconstructed_x, feature_names=x_col)
    plot_reconstruction(test_y, reconstructed_y, feature_names=y_col)

    regressor = LatentRegressor(latent_dim).to(device)
    optimizer_reg = optim.Adam(regressor.parameters(), lr=1e-3)

    print("Training latent regressor (z_x -> z_y)...")
    latent_history = train_latent_regressor(input_model, output_model, regressor,
                                            train_x_tensor, train_y_tensor,
                                            optimizer_reg, device, epochs=50)

    # === Stage 3: Prediction ===
    print("Making predictions from inputs...")
    y_hat = predict_from_x(input_model, output_model, regressor, test_x_tensor, device)

    # Unscale back to physical units
    y_hat_unscaled = y_scaler.inverse_transform(y_hat)
    test_y_unscaled = y_scaler.inverse_transform(test_y)

    # Plot results
    plot_reconstruction(test_y_unscaled, y_hat_unscaled, feature_names=y_col)
main()