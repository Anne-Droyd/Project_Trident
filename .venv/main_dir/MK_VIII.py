"""
Normalizing flows approach
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from main_dir.Methods.Data_Options import data_options
from sklearn.preprocessing import StandardScaler

data_ops=data_options(type="VR_DATA",default_data=True)
model_folder_path = "C:/Users/Matth/Documents/Leiden University/Project/models/NF/"
history_path = "C:/Users/Matth/Documents/Leiden University/Project/Histories/NF/"

import torch
import torch.nn as nn
import torch.optim as optim
from FrEIA.framework import InputNode, ConditionNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

class EarlyStoppingAndCheckpoint:
    def __init__(self, patience=10, save_path=None):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1

        # return True if training should stop
        return self.counter >= self.patience

class ConditionalINN(nn.Module):
    def __init__(self, y_dim, x_dim, hidden_dim=128, n_blocks=4, lr=1e-3, device=None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_dim = y_dim
        self.x_dim = x_dim

        # --- build the conditional INN ---
        self.model = self.build_cinn(y_dim, x_dim, hidden_dim, n_blocks).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def subnet_fc(self, c_in, c_out):
        """Small MLP used inside each coupling block."""
        return nn.Sequential(
            nn.Linear(c_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, c_out)
        )

    def build_cinn(self, y_dim, x_dim, hidden_dim, n_blocks):
        nodes = []
        cond = ConditionNode(x_dim, name='condition')
        nodes.append(InputNode(y_dim, name='y_input'))

        for i in range(n_blocks):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor': self.subnet_fc, 'clamp': 2.0},
                              conditions=cond,
                              name=f'coupling_{i}'))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed': i},
                              name=f'permute_{i}'))

        nodes.append(OutputNode(nodes[-1], name='output'))
        return ReversibleGraphNet(nodes + [cond])

    def train_step(self, x_cond, y_target):
        self.optimizer.zero_grad()
        y_target = y_target.to(self.device)
        x_cond = torch.tensor(x_cond, dtype=torch.float32).to(self.device)

        z, log_jac_det = self.model(y_target, c=[x_cond])
        loss = 0.5 * torch.sum(z**2) - torch.sum(log_jac_det)
        loss = loss / y_target.shape[0]
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def val_step(self, x_cond, y_target):
        self.model.eval()
        y_target = y_target.to(self.device)
        x_cond = torch.tensor(x_cond, dtype=torch.float32).to(self.device)

        z, log_jac_det = self.model(y_target, c=[x_cond])
        loss = 0.5 * torch.sum(z ** 2) - torch.sum(log_jac_det)
        loss = loss / y_target.shape[0]

        self.model.train()  # switch back to train mode
        return loss.item()

    def predict(self, x_cond, n_samples=1):
        """Generate samples of y given x."""
        with torch.no_grad():
            x_cond = torch.tensor(x_cond, dtype=torch.float32).to(self.device)
            z = torch.randn(n_samples, self.y_dim).to(self.device)

            # repeat condition to match batch size
            x_rep = x_cond.repeat(n_samples, 1)

            # universal inverse call returns tuple (y, log_jac_det)
            y_pred, _ = self.model(z, c=[x_rep], rev=True)

        return y_pred.cpu()

    def predict_deterministic(self, x_cond):
        """Most likely prediction (mean of the conditional)."""
        y_samples = self.predict(x_cond, n_samples=20)
        return y_samples.mean(dim=0)

from sklearn.metrics import mean_squared_error, r2_score

def plot_2d_hist(predictions,test_y,y_col,plot_path):

    for idx, col in enumerate(y_col):
        iter = data_ops.get_iteration(plot_path,file_prefix=col)
        fig = plt.figure(figsize=(6,4))
        print("predictions shape:", predictions.shape)
        print("test_y shape:", test_y.shape)
        mse = mean_squared_error(test_y[:,idx], predictions[:,idx])
        print(f'{col} Mean Squared Error: {mse}')

        r2 = r2_score(test_y[:,idx], predictions[:,idx])
        print(f'{col} R-squared: {r2}')
        min_val = min(test_y[:,idx].min())
        max_val = max(test_y[:,idx].max())
        h = plt.hist2d(test_y[:,idx], predictions[:,idx], bins=100, cmap="jet", cmax=100, density=True)
        fig.colorbar(h[3], label="Density")
        plt.plot([min_val, max_val], [min_val, max_val],'k--',lw=1,label='Perfect prediction')
        plt.title(f'Real vs predicted values for {col}')
        image_path = plot_path+f'real_vs_pred_{col}_{iter}.png'
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
        plt.ylim(min_val,max_val)
        plt.xlim(min_val, max_val)
        plt.legend()
        plt.tight_layout()
        plt.savefig(image_path)
        plt.show()

def main():
    data = data_ops.get_data()
    data = shuffle(data, random_state=42)
    # data = data[:1500]
    train, test, valid = data_ops.partition_data(data, train_frac=0.8, test_frac=0.1, valid_frac=0.1)
    iteration = data_ops.get_iteration(model_folder_path, "NF")
    print("Current input model iteration #",iteration)

    model_save_name = f"NF_input_model_{iteration}.h5"
    history_name = f"NF_history_{iteration}.csv"
    plot_path = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/plots/Normalizing_flow/'
    y_col = ["m_core", "ice_mass", "rock_mass", 'h_he_mass']
    x_col = ["mass", "radius", "temp"]
    train_x, train_y = data_ops.get_xy(train, y_col, x_col)
    test_x, test_y = data_ops.get_xy(test, y_col, x_col)
    valid_x, valid_y = data_ops.get_xy(valid, y_col, x_col)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_x)
    train_y = y_scaler.fit_transform(train_y)
    valid_x = x_scaler.transform(valid_x)
    valid_y = y_scaler.transform(valid_y)
    test_x = x_scaler.transform(test_x)
    test_y = y_scaler.transform(test_y)
    input_dim = train_x.shape[1]
    output_dim = train_y.shape[1]

    X_train = torch.tensor(train_x, dtype=torch.float32)
    Y_train = torch.tensor(train_y, dtype=torch.float32)
    X_valid = torch.tensor(valid_x, dtype=torch.float32)
    Y_valid = torch.tensor(valid_y, dtype=torch.float32)

    cinn = ConditionalINN(y_dim=Y_train.shape[1], x_dim=X_train.shape[1])

    n_epochs = 100
    callback = EarlyStoppingAndCheckpoint(
        patience=10,
        save_path=model_folder_path + model_save_name  # save best model
    )

    for epoch in range(1, n_epochs + 1):
        train_loss = cinn.train_step(X_train, Y_train)
        val_loss = cinn.val_step(X_valid, Y_valid)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        stop = callback.step(cinn, val_loss)
        if stop:
            print("Early stopping triggered.")
            break

    torch.save(cinn.state_dict(),model_folder_path+model_save_name)

    # Predict deterministically
    y_pred = cinn.predict_deterministic(X_valid[:1])
    y_pred_2d = y_pred.reshape(1, -1)  # 1 sample, 4 features
    y_pred_original = y_scaler.inverse_transform(y_pred_2d)
    print("Predicted y (deterministic):", y_pred_original)

    y_true_original = y_scaler.inverse_transform(test_y)  # scale test_y back

    y_pred = []
    for x in test_x:
        y_p = cinn.predict_deterministic(x)  # returns 1x4
        y_pred.append(y_p)
    y_pred = np.vstack(y_pred)  # shape: [n_test_samples, 4]

    # Scale back
    y_pred_original = y_scaler.inverse_transform(y_pred)
    y_true_original = y_scaler.inverse_transform(test_y)

    # Now plot real vs predicted
    fig, axes = plt.subplots(1, len(y_col), figsize=(16, 4))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.scatter(y_true_original[:, i], y_pred_original[:, i], alpha=0.7)
        ax.plot([y_true_original[:, i].min(), y_true_original[:, i].max()],
                [y_true_original[:, i].min(), y_true_original[:, i].max()], 'r--')
        ax.set_xlabel("True " + y_col[i])
        ax.set_ylabel("Predicted " + y_col[i])
    plt.tight_layout()
    plt.show()


    # Predict probabilistically (samples)
    y_samples = cinn.predict(X_valid[:1], n_samples=10)
    y_samples = y_scaler.inverse_transform(y_samples.numpy())
    print("Sampled ys:", y_samples)

    plot_2d_hist(y_pred_original,y_true_original,y_col,plot_path=plot_path)


main()