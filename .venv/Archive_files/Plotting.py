#my plotting module
import torch
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_posterior(predictions, y_test, data_point, prediction_col="m_core", real_col="m_core", index=0):
    #plot dataframes only
    if not isinstance(predictions,pd.DataFrame):
        raise TypeError("Prediction data must be pd.DataFrame")
    if not isinstance(y_test,pd.DataFrame):
        raise TypeError("Test data must be pd.DataFrame")
    # Pick a test index
    test_idx = index

    # Extract posterior samples for this point
    posterior_samples = predictions[test_idx]

    # Plot histogram of posterior
    plt.figure(figsize=(8, 5))
    sns.kdeplot(posterior_samples, fill=True)
    plt.title(f"Posterior distribution for test input {test_idx}")
    plt.xlabel(f"Predicted {prediction_col}")
    plt.ylabel("Density")

    # Optional: add true value line
    true_val = y_test[0][test_idx]
    plt.axvline(true_val, color='red', linestyle='--', label=f"True: {true_val:.2f}")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_real_vs_pred(predictions, y_test, col="m_core"):
    # Ensure inputs are DataFrames
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("Prediction data must be a pd.DataFrame")
    if col not in predictions.columns:
        raise Exception(f"{col} not in prediction DataFrame")
    if not isinstance(y_test, pd.DataFrame):
        raise TypeError("Test data must be a pd.DataFrame")
    if col not in y_test.columns:
        raise Exception(f"{col} not in test y DataFrame")

    y_true = y_test[col]
    y_pred = predictions[col]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, color="blue", alpha=0.05,label="Prediction")
    plt.scatter(y_true, y_true, color='red', alpha=0.05, label="Ground Truth")
    plt.title("Real values vs Predicted")
    plt.xlabel(f"True values of {col}")
    plt.ylabel(f"Predicted values of {col}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_real_vs_pred_error(predictions, y_test, pred_error, pred_col="m_core"):

    if not isinstance(predictions,pd.DataFrame):
        raise TypeError("Prediction data must be pd.DataFrame")
    if not isinstance(y_test,pd.DataFrame):
        raise TypeError("Test data must be pd.DataFrame")
    # Get sorting indices based on y_test
    sort_idx = y_test[:, 0].argsort()

    # Sort y_test, predictions, and pred_error using the same indices
    y_sorted = y_test[sort_idx]
    y_pred_sorted = predictions[sort_idx]
    pred_error_sorted = pred_error[sort_idx]

    # Plot with error bars
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        y_sorted.squeeze().cpu().numpy(),
        y_pred_sorted.squeeze().cpu().numpy(),
        yerr=pred_error_sorted.squeeze().cpu().numpy(),
        fmt='o'
    )
    plt.title("Real values vs Predicted")
    plt.xlabel(f"Real values of {pred_col}")
    plt.ylabel(f"Predicted values of {pred_col}")
    plt.tight_layout()
    plt.show()

def random_sample_violin_plot(predictions, y_test, pred_col="m_core", num_samples=5):

    if not isinstance(predictions,pd.DataFrame):
        raise TypeError("Prediction data must be pd.DataFrame")
    if not isinstance(y_test,pd.DataFrame):
        raise TypeError("Test data must be pd.DataFrame")
    # Randomly sample some indices
    indices = random.sample(range(predictions[pred_col].shape[1]), num_samples)

    # Prepare data for seaborn
    data = []
    for i in indices:
        samples = predictions[pred_col][:, i].detach().cpu().numpy()
        for val in samples:
            data.append({
                "Prediction": val,
                "Sample Index": f"Sample {i}",
                "True Value": y_test[i].item()
            })

    df = pd.DataFrame(data)

    # Plot violin plot
    plt.figure(figsize=(10, 5))
    sns.violinplot(x="Sample Index", y="Prediction", data=df, inner="quartile", palette="pastel")

    # Add true values as red dots
    for i, idx in enumerate(indices):
        plt.scatter(i, y_test[idx].item(), color='red', label="True Value" if i == 0 else "", zorder=3)

    plt.ylabel("Predicted Value")
    plt.title("Posterior Predictions (Violin Plot)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mixture(y_pred, x_input):
    pi, mu, std = mdn.extract_mixture_params(y_pred, output_dim=1, num_mixes=N_MIXES)
    idx = 0  # choose a test sample
    x = np.linspace(-3, 3, 500)  # or suitable range for your target

    total_pdf = np.zeros_like(x)
    for j in range(N_MIXES):
        total_pdf += pi[idx, j] * norm.pdf(x, mu[idx, j], std[idx, j])
    plt.plot(x, total_pdf)
    plt.title("Predicted PDF for one test example")
    plt.show()


def plot_mdn_prediction(pi, mu, sigma, y_scaler, idx=0,target_dim=0):
    """Visualize the predicted PDF of the MDN for one test sample."""
    # Inverse transform
    # Select the desired dimension
    mu_i = mu[idx][:, target_dim]
    sigma_i = sigma[idx][:, target_dim]
    pi_i = pi[idx]

    # Inverse-transform mu manually for that dimension
    mu_i_unscaled = mu_i * y_scaler.scale_[target_dim] + y_scaler.mean_[target_dim]
    sigma_i_unscaled = sigma_i * y_scaler.scale_[target_dim]

    # Create x-range for PDF
    x = np.linspace(mu_i_unscaled.min() - 3 * sigma_i_unscaled.max(),
                    mu_i_unscaled.max() + 3 * sigma_i_unscaled.max(), 1000)

    # Weighted sum of Gaussians
    pdf = sum(p * norm.pdf(x, m, s) for p, m, s in zip(pi_i, mu_i_unscaled, sigma_i_unscaled))

    plt.plot(x, pdf)
    plt.title(f"Predicted PDF for sample {idx}, target {target_dim}")
    plt.xlabel("Target value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

def plot_history_accuracy(history):
    #  for use in classification problems
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def plot_history_loss(history):
    # can be used in both regression and classification I believe
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def plot_hist_pred_mean_vs_real(test_y,prediction,col):
    if not isinstance(test_y,pd.DataFrame):
        raise TypeError("Test Y data-set must be pd.DataFrame")
    if col not in test_y.columns:
        raise Exception(f"{col} not in test y DataFrame")
    if not isinstance(prediction,pd.DataFrame):
        raise TypeError("Predicted mean data-set must be pd.DataFrame")
    if col not in prediction.columns:
        raise Exception(f"{col} not in predicted mean DataFrame")
    plt.hist(test_y[col],color="red",bins=100,alpha=0.5,edgecolor="None",label="Real values")
    plt.hist(prediction[col],color="blue", bins=100,alpha=0.5,edgecolor="None",label="Predicted values")
    plt.title(f"Histogram of Real vs Predicted for {col}")
    plt.xlabel(f"{col}")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualization functions
def plot_latent_space(vae,test_x,test_y, n=30, figsize=8):
    # Display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(test_x)
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=test_y)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()