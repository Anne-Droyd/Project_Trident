import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from main_dir.Methods.Data_Options import data_options

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def prepare_data(data, features=None, target_col=None, scaling = "standard"):
    """
    Prepare data for dimensionality reduction

    Parameters:
    -----------
    data : pd.DataFrame
        The input data
    features : list, optional
        List of feature columns to use. If None, uses all numeric columns
    target_col : str, optional
        Column to use for coloring points
    """
    if features is None:
        # Use all numeric columns except target
        features = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in features:
            features.remove(target_col)

    X = data[features].values

    # Standardize features
    scaling = scaling.lower()
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Get target for coloring if specified
    y = data[target_col].values if target_col in data.columns else None

    return X_scaled, y, features


def apply_dimensionality_reduction(X, method='pca', n_components=2, **kwargs):
    """
    Apply dimensionality reduction method

    Parameters:
    -----------
    X : array-like
        Input data (should be scaled)
    method : str
        Method to use: 'pca', 'kernel_pca', 'umap', 'tsne'
    n_components : int
        Number of dimensions to reduce to
    **kwargs : dict
        Additional parameters for the method
    """


    if method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
        X_reduced = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_
        return X_reduced, {'explained_variance': explained_var, 'reducer': reducer}

    elif method == 'kernel_pca':
        kernel = kwargs.pop('kernel', 'rbf')
        gamma = kwargs.pop('gamma', None)
        reducer = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, **kwargs)
        X_reduced = reducer.fit_transform(X)
        return X_reduced, {'reducer': reducer, 'kernel': kernel}

    elif method == 'umap':
        n_neighbors = kwargs.pop('n_neighbors', 15)
        min_dist = kwargs.pop('min_dist', 0.1)
        metric = kwargs.pop('metric', 'euclidean')
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
            **kwargs
        )
        X_reduced = reducer.fit_transform(X)
        return X_reduced, {'reducer': reducer}

    elif method == 'tsne':
        perplexity = kwargs.pop('perplexity', 30)
        learning_rate = kwargs.pop('learning_rate', 200)
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=42,
            **kwargs
        )
        X_reduced = reducer.fit_transform(X)
        return X_reduced, {'reducer': reducer}

    else:
        raise ValueError(f"Unknown method: {method}")


def plot_comparison(data, features=None, target_col=None, save_path=None, sample_size=None):
    """
    Create a 2x2 comparison plot of all four methods

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    features : list, optional
        Features to use for reduction
    target_col : str, optional
        Column to use for coloring
    save_path : str, optional
        Path to save the figure
    sample_size : int, optional
        Number of samples to use (for speed with large datasets)
    """
    # Sample data if needed
    if sample_size and len(data) > sample_size:
        data_sample = data.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} points from {len(data)} total")
    else:
        data_sample = data

    # Prepare data
    X_scaled, y, feature_names = prepare_data(data_sample, features, target_col)

    print(f"Using {len(feature_names)} features: {feature_names}")
    print(f"Data shape: {X_scaled.shape}")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    methods = [
        ('pca', 'PCA', {}),
        ('kernel_pca', 'Kernel PCA (RBF)', {'kernel': 'rbf', 'gamma': 1 / X_scaled.shape[1]}),
        ('umap', 'UMAP', {'n_neighbors': 15, 'min_dist': 0.1}),
        ('tsne', 't-SNE', {'perplexity': 30})
    ]

    for idx, (method, title, params) in enumerate(methods):
        ax = axes[idx]

        print(f"\nApplying {title}...")
        X_reduced, info = apply_dimensionality_reduction(X_scaled, method=method, **params)

        # Create scatter plot
        if y is not None:
            scatter = ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                c=y,
                cmap='viridis',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(target_col, fontsize=10)
        else:
            ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                alpha=0.6,
                s=20,
                edgecolors='none'
            )

        # Set title with additional info
        title_text = title
        if method == 'pca' and 'explained_variance' in info:
            var_explained = info['explained_variance'][:2].sum() * 100
            title_text += f"\n(Explained Variance: {var_explained:.1f}%)"

        ax.set_title(title_text, fontsize=14, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=11)
        ax.set_ylabel('Component 2', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Dimensionality Reduction Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    plt.show()

    return fig


def plot_pca_variance(X_scaled, n_components=10, save_path=None):
    """
    Plot explained variance for PCA components

    Parameters:
    -----------
    X_scaled : array-like
        Scaled input data
    n_components : int
        Number of components to analyze
    save_path : str, optional
        Path to save figure
    """
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    pca.fit(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_ * 100)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained (%)', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_ * 100)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, 'o-', linewidth=2, markersize=8)
    ax2.axhline(y=90, color='r', linestyle='--', label='90% threshold')
    ax2.axhline(y=95, color='orange', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA variance plot saved to {save_path}")

    plt.show()

    return fig


def analyze_feature_importance(data, features, target_col, n_components=2):
    """
    Analyze feature importance using PCA loadings

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    features : list
        Feature columns
    target_col : str
        Target column
    n_components : int
        Number of PCA components
    """
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Create loadings dataframe
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i + 1}' for i in range(n_components)],
        index=features
    )

    print("\nPCA Loadings (Feature Contributions):")
    print(loadings)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.5)))
    sns.heatmap(loadings, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, ax=ax, cbar_kws={'label': 'Loading'})
    ax.set_title('PCA Feature Loadings', fontsize=14, fontweight='bold')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

    return loadings


def main():
    # Load data
    data_ops = data_options(type="VR_DATA",default_data=True)
    data = data_ops.get_data()

    print(f"Total samples: {len(data)}")
    print(f"\nData columns: {data.columns.tolist()}")

    # Define features and target
    input_features = ["mass", "radius", "temp"]
    output_features = ["m_core", "ice_mass", "rock_mass", "h_he_mass"]

    # Analyze input space (X)
    print("\n" + "=" * 60)
    print("ANALYZING INPUT SPACE (mass, radius, temp)")
    print("=" * 60)

    X_data = data[input_features].copy()
    X_scaled, _, _ = prepare_data(data, input_features, None)

    # Plot PCA variance for input space
    print("\nPCA Variance Analysis for Input Features:")
    plot_pca_variance(X_scaled, n_components=3,
                      save_path="pca_variance_input.png")

    # Visualize input space colored by total mass
    print("\nVisualizing input space (colored by total mass)...")
    plot_comparison(data, features=input_features, target_col="mass",
                    save_path="dimred_comparison_input.png",
                    sample_size=5000)  # Sample for speed

    # Analyze output space (Y)
    print("\n" + "=" * 60)
    print("ANALYZING OUTPUT SPACE (compositions)")
    print("=" * 60)

    Y_data = data[output_features].copy()
    Y_scaled, _, _ = prepare_data(data, output_features, None)

    # Plot PCA variance for output space
    print("\nPCA Variance Analysis for Output Features:")
    plot_pca_variance(Y_scaled, n_components=4,
                      save_path="pca_variance_output.png")

    # Visualize output space colored by core mass
    print("\nVisualizing output space (colored by core mass)...")
    plot_comparison(data, features=output_features, target_col="m_core",
                    save_path="dimred_comparison_output.png",
                    sample_size=5000)

    # Analyze combined space
    print("\n" + "=" * 60)
    print("ANALYZING COMBINED SPACE (X + Y)")
    print("=" * 60)

    all_features = input_features + output_features
    combined_scaled, _, _ = prepare_data(data, all_features, None)

    # Plot PCA variance for combined space
    print("\nPCA Variance Analysis for Combined Features:")
    plot_pca_variance(combined_scaled, n_components=7,
                      save_path="pca_variance_combined.png")

    # Visualize combined space colored by mass
    print("\nVisualizing combined space (colored by planet mass)...")
    plot_comparison(data, features=all_features, target_col="mass",
                    save_path="dimred_comparison_combined.png",
                    sample_size=5000)

    # Feature importance analysis
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    analyze_feature_importance(data, input_features, "mass", n_components=3)

    print("\n" + "=" * 60)
    print("Analysis complete! Check saved plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()