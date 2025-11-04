import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt

tfb = tfp.bijectors
tfd = tfp.distributions


class ConditionalCouplingLayer(layers.Layer):
    """Conditional affine coupling layer that takes external conditioning"""

    def __init__(self, split_dim, hidden_units=[128, 128], **kwargs):
        super(ConditionalCouplingLayer, self).__init__(**kwargs)
        self.split_dim = split_dim
        self.hidden_units = hidden_units

    def build(self, input_shape):
        dim = input_shape[-1]

        # Scale and translation networks (will receive concatenated input)
        self.scale_net = keras.Sequential([
                                              layers.Dense(units, activation='relu')
                                              for units in self.hidden_units
                                          ] + [layers.Dense(dim - self.split_dim, activation='tanh')])

        self.translate_net = keras.Sequential([
                                                  layers.Dense(units, activation='relu')
                                                  for units in self.hidden_units
                                              ] + [layers.Dense(dim - self.split_dim)])

        super(ConditionalCouplingLayer, self).build(input_shape)

    def call(self, x, condition, forward=True):
        x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]

        # Concatenate first part with condition for transformation
        conditioned_input = tf.concat([x1, condition], axis=-1)

        if forward:
            s = self.scale_net(conditioned_input)
            t = self.translate_net(conditioned_input)
            z2 = x2 * tf.exp(s) + t
            z = tf.concat([x1, z2], axis=-1)
            log_det = tf.reduce_sum(s, axis=-1)
            return z, log_det
        else:
            s = self.scale_net(conditioned_input)
            t = self.translate_net(conditioned_input)
            z2 = (x2 - t) * tf.exp(-s)
            z = tf.concat([x1, z2], axis=-1)
            log_det = -tf.reduce_sum(s, axis=-1)
            return z, log_det


class NormalizingFlowModel(keras.Model):
    """Conditional Normalizing Flow for planetary composition prediction"""

    def __init__(self, input_dim, output_dim, num_coupling_layers=8, hidden_units=[256, 256], **kwargs):
        super(NormalizingFlowModel, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_coupling_layers = num_coupling_layers

        # Conditioning network (processes input features)
        self.condition_net = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu')
        ])

        # Coupling layers with alternating splits
        self.coupling_layers = []
        self.permutations = []

        for i in range(num_coupling_layers):
            # Alternate split dimension
            split_dim = output_dim // 2 if i % 2 == 0 else (output_dim + 1) // 2
            self.coupling_layers.append(
                ConditionalCouplingLayer(split_dim, hidden_units)
            )

            # Add permutation layer (simple reversal)
            if i < num_coupling_layers - 1:
                perm = tf.constant(list(reversed(range(output_dim))))
                self.permutations.append(perm)

        # Base distribution (standard normal)
        self.base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(output_dim),
            scale_diag=tf.ones(output_dim)
        )

    def forward(self, y, x, training=False):
        """Transform y to base distribution given condition x"""
        # Process condition
        condition = self.condition_net(x, training=training)

        z = y
        log_det_sum = 0.0

        for i, layer in enumerate(self.coupling_layers):
            z, log_det = layer(z, condition, forward=True)
            log_det_sum += log_det

            # Apply permutation
            if i < len(self.permutations):
                z = tf.gather(z, self.permutations[i], axis=-1)

        return z, log_det_sum

    def inverse(self, z, x, training=False):
        """Transform from base distribution to data space given condition x"""
        condition = self.condition_net(x, training=training)

        y = z
        log_det_sum = 0.0

        # Reverse through layers
        for i, layer in enumerate(reversed(self.coupling_layers)):
            # Reverse permutation first
            perm_idx = len(self.coupling_layers) - i - 2
            if perm_idx >= 0:
                # Inverse permutation
                inv_perm = tf.argsort(self.permutations[perm_idx])
                y = tf.gather(y, inv_perm, axis=-1)

            y, log_det = layer(y, condition, forward=False)
            log_det_sum += log_det

        return y, log_det_sum

    def call(self, inputs, training=False):
        """Forward pass for training"""
        x, y = inputs
        z, log_det = self.forward(y, x, training=training)
        return z, log_det

    def sample(self, x, num_samples=1):
        """Sample from the conditional distribution p(y|x)"""
        batch_size = tf.shape(x)[0]

        # Sample from base distribution
        z = self.base_dist.sample((num_samples, batch_size))

        samples = []
        for i in range(num_samples):
            y, _ = self.inverse(z[i], x, training=False)
            samples.append(y)

        return tf.stack(samples, axis=0)

    def log_prob(self, y, x):
        """Compute log probability of y given x"""
        z, log_det = self.forward(y, x, training=False)
        log_pz = self.base_dist.log_prob(z)
        return log_pz + log_det


def nf_loss(model, x, y):
    """Negative log-likelihood loss"""
    log_prob = model.log_prob(y, x)
    return -tf.reduce_mean(log_prob)


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        loss = nf_loss(model, x, y)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train_normalizing_flow(model, train_x, train_y, valid_x, valid_y,
                           epochs=200, batch_size=64, learning_rate=5e-4):
    """Train the normalizing flow model"""
    # Use learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.96
    )
    optimizer = keras.optimizers.Adam(lr_schedule)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        train_losses = []
        for x_batch, y_batch in train_dataset:
            loss = train_step(model, optimizer, x_batch, y_batch)
            train_losses.append(loss.numpy())

        train_loss = np.mean(train_losses)

        # Validation
        val_loss = nf_loss(model, valid_x, valid_y).numpy()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return history


def plot_predictions(model, test_x, test_y, y_scaler, y_col, num_samples=50, save_path=None):
    """Plot actual vs predicted values for each output dimension"""
    # Generate predictions
    predictions = model.sample(test_x, num_samples=num_samples)
    mean_pred = tf.reduce_mean(predictions, axis=0).numpy()
    std_pred = tf.math.reduce_std(predictions, axis=0).numpy()

    # Inverse transform to original scale
    mean_pred_original = y_scaler.inverse_transform(mean_pred)
    test_y_original = y_scaler.inverse_transform(test_y)

    # Calculate prediction intervals (±2 std)
    std_pred_original = std_pred * y_scaler.scale_

    # Create subplots
    n_outputs = test_y.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, col_name in enumerate(y_col):
        ax = axes[i]

        actual = test_y_original[:, i]
        predicted = mean_pred_original[:, i]
        uncertainty = std_pred_original[:, i]

        # Scatter plot with error bars
        ax.errorbar(actual, predicted, yerr=2 * uncertainty, fmt='o',
                    alpha=0.5, markersize=4, capsize=2,
                    label='Predictions ± 2σ')

        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect Prediction')

        # Calculate R² score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Calculate MAE
        mae = np.mean(np.abs(actual - predicted))

        ax.set_xlabel(f'Actual {col_name}', fontsize=11)
        ax.set_ylabel(f'Predicted {col_name}', fontsize=11)
        ax.set_title(f'{col_name}\nR² = {r2:.3f}, MAE = {mae:.3f}', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

    return fig


def plot_training_history(history, save_path=None):
    """Plot training and validation loss over epochs"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax.set_title('Training History', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()

    return fig

from main_dir.Methods.Data_Options import data_options
from sklearn.preprocessing import StandardScaler

data_ops=data_options(type="VR_DATA",default_data=True)
model_folder_path = "C:/Users/Matth/Documents/Leiden University/Project/models/NF/"
history_path = "C:/Users/Matth/Documents/Leiden University/Project/Histories/NF/"

def main():
    # Assuming data is loaded (replace with your data loading)
    # data = pd.read_csv("your_data.csv")
    # data = shuffle(data, random_state=42)

    # For demonstration, create synthetic data
    np.random.seed(42)
    n_samples = 10000
    data = data_ops.get_data()
    data = shuffle(data, random_state=42)

    # Partition data
    train_size = int(0.8 * len(data))
    test_size = int(0.1 * len(data))

    train = data[:train_size]
    test = data[train_size:train_size + test_size]
    valid = data[train_size + test_size:]

    y_col = ["m_core", "ice_mass", "rock_mass", 'h_he_mass']
    x_col = ["mass", "radius", "temp", 'k2']

    train_x, train_y = train[x_col].values, train[y_col].values
    test_x, test_y = test[x_col].values, test[y_col].values
    valid_x, valid_y = valid[x_col].values, valid[y_col].values

    # Standardization
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_x = x_scaler.fit_transform(train_x).astype(np.float32)
    train_y = y_scaler.fit_transform(train_y).astype(np.float32)
    valid_x = x_scaler.transform(valid_x).astype(np.float32)
    valid_y = y_scaler.transform(valid_y).astype(np.float32)
    test_x = x_scaler.transform(test_x).astype(np.float32)
    test_y = y_scaler.transform(test_y).astype(np.float32)

    input_dim = train_x.shape[1]
    output_dim = train_y.shape[1]

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Training samples: {len(train_x)}")

    # Create model
    model = NormalizingFlowModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_coupling_layers=10,
        hidden_units=[256, 256]
    )

    # Train model
    print("\nTraining Normalizing Flow...")
    history = train_normalizing_flow(
        model, train_x, train_y, valid_x, valid_y,
        epochs=20, batch_size=64, learning_rate=5e-4
    )

    # Generate predictions (samples from conditional distribution)
    print("\nGenerating predictions...")
    num_samples = 10
    predictions = model.sample(test_x[:5], num_samples=num_samples)

    # Mean prediction
    mean_pred = tf.reduce_mean(predictions, axis=0)
    std_pred = tf.math.reduce_std(predictions, axis=0)

    print("\nSample predictions (first 5 test samples):")
    print("Mean predictions shape:", mean_pred.shape)
    print("\nActual vs Predicted (mean ± std):")
    for i in range(5):
        print(f"\nSample {i + 1}:")
        print(f"Actual: {test_y[i]}")
        print(f"Predicted: {mean_pred[i].numpy()}")
        print(f"Std: {std_pred[i].numpy()}")

    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, save_path="NF_training_history.png")

    # Plot predictions vs actual
    print("\nPlotting predictions vs actual values...")
    plot_predictions(model, test_x, test_y, y_scaler, y_col,
                     num_samples=50, save_path="NF_predictions.png")

    # Save model
    # model.save_weights("NF_input_model_0.h5")
    # pd.DataFrame(history).to_csv("NF_history_0.csv", index=False)

    return model, history, x_scaler, y_scaler


if __name__ == "__main__":
    main()