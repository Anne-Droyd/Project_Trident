#my plotting module
import torch
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_single_pdf(mu, sigma, pi, y_col, test_y=None, mean_pred=None, mu_map=None,
                    means=None, vars=None, top5_mus=None, idx=0, plot_individual=False):
    mu = mu[idx, :, :]        # (K, D)
    sigma = sigma[idx, :, :]  # (K, D)
    pi = pi[idx, :]           # (K,)

    K, D = mu.shape

    for dim, col in enumerate(y_col):

        # Clip μ to [0, 1] to avoid out-of-range Gaussians
        mu_d = np.clip(mu[:, dim], 0, 1)        # (K,)
        sigma_d = sigma[:, dim]                 # (K,)

        # Define x-axis range fixed to [0, 1]
        x = np.linspace(0, 1, 1000)              # (1000,)

        # Vectorized Gaussian PDF computation
        x_expanded = x[:, np.newaxis]  # (1000, 1)
        pdfs = norm.pdf(x_expanded, loc=mu_d, scale=sigma_d)  # (1000, K)
        weighted_pdfs = pdfs * pi
        total_pdf = np.sum(weighted_pdfs, axis=1)

        plt.figure(figsize=(8, 4))
        plt.plot(x, total_pdf, label="Mixture PDF", linewidth=2)

        if plot_individual:
            plt.plot(x, weighted_pdfs, color="gray", alpha=0.2)

        if test_y is not None:
            real_val = np.clip(test_y.iloc[idx, dim], 0, 1)
            plt.axvline(x=real_val, color='r', linestyle='--', label='True y')

        if mean_pred is not None:
            pred = np.clip(mean_pred[idx, dim], 0, 1)
            plt.axvline(x=pred, color='b', linestyle='-', label='Prediction')

        if mu_map is not None:
            pred_1 = np.clip(mu_map[idx, dim], 0, 1)
            plt.axvline(x=pred_1, color='b', linestyle='--', label='2nd prediction')

        if top5_mus is not None:
            # Get weights for top5 mus in the same order as top5_mus
            pi_d = pi  # (K,)
            mus_for_dim = top5_mus[idx, :, dim]  # shape: (5,)

            # Sort top5 by pi value for this dimension (descending)
            pi_values_for_top5 = []
            for m in mus_for_dim:
                # Find the matching component index in mu_d
                comp_idx = np.where(mu_d == m)[0]
                if len(comp_idx) > 0:
                    pi_values_for_top5.append(pi_d[comp_idx[0]])
                else:
                    pi_values_for_top5.append(0.0)

            sorted_indices = np.argsort(pi_values_for_top5)[::-1]
            sorted_mus = mus_for_dim[sorted_indices]

            colors = ['red', 'orange', 'green', 'blue', 'purple']

            for rank, m in enumerate(sorted_mus):
                label = f"Top-{rank + 1} μ" if rank > 0 else "Top-1 μ (most likely)"
                plt.axvline(x=m, color=colors[rank], linestyle=':', alpha=0.8, label=label)

        if means is not None and vars is not None:
            mean = np.clip(means[idx, dim], 0, 1)
            std_dev = np.sqrt(vars[idx, dim])
            lower_bound = np.clip(mean - std_dev, 0, 1)
            upper_bound = np.clip(mean + std_dev, 0, 1)
            plt.fill_between(x, total_pdf, where=(x >= lower_bound) & (x <= upper_bound),
                             color="blue", alpha=0.2, label="±1 Std Dev")
            lower_bound_2 = np.clip(mean - 2 * std_dev, 0, 1)
            upper_bound_2 = np.clip(mean + 2 * std_dev, 0, 1)
            plt.fill_between(x, total_pdf, where=(x >= lower_bound_2) & (x <= upper_bound_2),
                             color="blue", alpha=0.1, label="±2 Std Dev")

        plt.title(f"PDF for output: {col}")
        plt.xlim(0, 1)  # Force x-axis range
        plt.xlabel("y")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

# def plot_single_pdf(mu,sigma,pi,y_col,test_y = None,idx=0):
#
#     mu = mu[idx, :, :]
#     sigma = sigma[idx, :, :]
#     pi = pi[idx, :]
#
#     for dim, col in enumerate(y_col):
#         mu_per_dim = mu[:,dim]
#         sigma_per_dim = sigma[:,dim]
#         x = np.linspace(mu_per_dim.min() - 2*sigma_per_dim, mu_per_dim.max() + 2*sigma_per_dim, 1000)
#         pdf = np.zeros_like(x)
#         for i in range(len(mu_per_dim)):
#             pdf += pi[i]*norm.pdf(x,loc=mu_per_dim[i],scale=sigma_per_dim[i])
#         plt.figure(figsize=(10,5))
#         plt.plot(x,pdf, label="PDF of all 50 gaussians")
#
#         for i in range(len(mu_per_dim)):
#             plt.plot(x,pi[i]*norm.pdf(x, loc=mu_per_dim[i], scale=sigma_per_dim[i]),color="gray",alpha=0.3)
#
#         if test_y is None:
#             real_val = None
#         else:
#             real_val = test_y.iloc[idx,dim]
#             plt.axvline(x=real_val, color='r', label='Real value')
#         plt.title(f"Micture of Gaussians PDF for {col}")
#         plt.xlabel("x")
#         plt.ylabel("Density")
#         plt.legend()
#         plt.show()

def plot_random_multiple_pdf(mu,sigma,pi):
    h=0

def sample_from_mixture(mu,sigma,pi,number_of_samples=1):
    N, K, D = mu.shape
    samples = np.zeros((N, number_of_samples, D))

    for i in range(N):
        for num in range(number_of_samples):
            # 1. Sample a component index from the categorical distribution
            k = np.random.choice(K, p=pi[i])

            # 2. Sample from the selected Gaussian
            mu_sample = mu[i, k]
            sigma_sample = sigma[i, k]
            sample = np.random.normal(mu_sample, sigma_sample)  # Draw from N(mu, sigma)
            samples[i,num] = sample

    return np.array(samples)

def sample_from_all_components(mu, sigma, pi, n_samples=1):
    N, K, D = mu.shape
    samples = np.zeros((N, n_samples, D))

    for i in range(N):
        weights = np.random.choice(K, size=n_samples, p=pi[i])
        for s, k in enumerate(weights):
            samples[i, s] = np.random.normal(mu[i, k], sigma[i, k])
    return samples



def plot_predicted_vs_real_hist(samples, test_y,y_col=None):
    N, M, D = samples.shape

    # Create a 2 x 3 subplot grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()  # make indexing easier

    for dim in range(D):
        if y_col is not None:
            label = y_col[dim]
        else:
            label = dim
        ax = axes[dim]

        # Flatten for plotting
        true_vals = np.repeat(test_y.iloc[:, dim].values, M)
        sampled_vals = samples[:, :, dim].reshape(-1)

        h = ax.hist2d(true_vals, sampled_vals, bins=100, cmap="hot", cmax=100, density=True)
        fig.colorbar(h[3], ax=ax, label="Density")

        ax.scatter(true_vals, true_vals, s=5, c="blue", label="y=x")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Sampled Prediction")
        ax.set_title(f"Predicted vs Real {label}")
        ax.legend()

    # Hide unused subplots if D < 6
    for j in range(D, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_predicted_vs_real_scatter(samples,test_y):
    # samples = np.mean(samples,axis=1)
    for dim in range(samples.shape[1]):
        plt.figure(figsize=(6, 5))
        plt.scatter(test_y.iloc[:,dim], samples[:,dim],c="blue",label="Predictions",alpha=0.01)
        plt.scatter(test_y.iloc[:,dim],test_y.iloc[:,dim],c="red",label="Real values",alpha=0.5)
        plt.xlabel("True Value")
        plt.ylabel("Sampled Prediction")
        plt.title(f"Predicted vs Real (dim {dim})")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_mu_vs_real_with_errorbars(mu_mixture, var_mixture, test_y, y_col):
    """
    Plots real values vs predicted mixture means with error bars.

    Parameters:
    - mu_mixture: (num_points, output_dim) – predicted means
    - var_mixture: (num_points, output_dim) – predicted variances
    - test_y: DataFrame – true values
    - y_col: list – column names (same order as output_dim)
    """

    std_mixture = np.sqrt(var_mixture)
    output_dim = mu_mixture.shape[1]

    for dim in range(output_dim):
        plt.figure(figsize=(6, 5))
        plt.errorbar(test_y.iloc[:, dim], mu_mixture[:, dim],
                     yerr=std_mixture[:, dim], fmt='o', alpha=0.2,
                     ecolor='gray', label='Prediction ±1σ')

        # Line y = x (perfect prediction)
        min_val = min(test_y.iloc[:, dim].min(), mu_mixture[:, dim].min())
        max_val = max(test_y.iloc[:, dim].max(), mu_mixture[:, dim].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.xlabel('True Value')
        plt.ylabel('Predicted Mean (μ)')
        plt.title(f'Predicted μ vs Real: {y_col[dim]}')
        plt.legend()
        plt.tight_layout()
        plt.show()