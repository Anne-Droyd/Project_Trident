
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main_dir.Methods.Data_Options import data_options

data_ops = data_options(type="VR_DATA")
data = data_ops.get_data()


# ['planet_mass', 'planet_radius', 'T_eq', 'log_d_mantle_core','log_m_mantle_core', 'log_d_atmosphere_core',
#  'log_m_atmosphere_core','log_d_water_core', 'log_m_water_core']

df1 = data[data.isna().any(axis=1)]
data = data.dropna()

# print(data.info())
# print(data.isnull().sum())

earth_mass = 5.97e24
phobos_mass = 1.06e16
Grav_cons = 6.6743e-11
# data = data.rename(columns={'planet_mass':'mass','planet_radius':'radius','T_eq':'temp'})
data = data.rename(columns={'req':'radius','Teq':'temp'})
data["ice_mass"] = (data["zatm0"] * data["matm"] + data["zdeep0"] * data["mdeep"])
data["rock_mass"] = (data["zatm1"] * data["matm"] + data["zdeep1"] * data["mdeep"])
data["h_he_mass"] = (1 - (data['m_core'] + data["rock_mass"] + data["ice_mass"]))

# data["ice_mass"]  = np.log(data['ice_mass']/(data['m_core']*data['mass']))
# data["rock_mass"] = np.log(data['rock_mass']/(data['m_core']*data['mass']))
# data["h_he_mass"] = np.log(data['h_he_mass']/(data['m_core']*data['mass']))

# data['m_core']  = data['mass']/(1+(10**data['log_m_mantle_core']+10**data['log_m_atmosphere_core']+10**data['log_m_water_core']))
# data['m_mantle']= 10**data['log_m_mantle_core']*data['m_core']
# data['matm']    = 10**data['log_m_atmosphere_core']*data['m_core']
# data['m_water'] = 10**data['log_m_water_core']*data['m_core']

# data['density'] = data['mass'] / ((4/3)*np.pi*data['radius']**3)  # Bulk density
# data['surface_gravity'] = Grav_cons*data['mass'] / (data['radius']**2)
# data['mass_radius_ratio'] = 0.56*data['mass']**(0.67) / data['radius']
# data['temp_density_interaction'] = data['temp'] * data['density']

# columns=['mass','radius','temp','ice_mass','rock_mass','h_he_mass','density','surface_gravity','mass_radius_ratio','temp_density_interaction']

columns = data.columns.tolist()

file_outname= "C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/Data/T0_data_set.dat"
# file_outname= "C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/ExoMDN-main/ExoMDN-main/data/training_demo/default.dat"
data.to_csv(file_outname,index=False,sep="\t",columns=columns,header=True)

# for index, row in data.iterrows():
#     if row["m_core"]+row['matm']+row['m_mantle']+row['m_water'] > row['mass']:
#         print(row["m_core"]+row['matm']+row['m_mantle']+row['m_water']-row['mass'])


def plot_distributions(data, columns, bins=30):
    """
    Plot distributions for multiple columns in a grid layout

    Parameters:
    -----------
    data : pandas DataFrame
        The dataframe containing the columns to plot
    columns : list
        List of column names to plot
    bins : int
        Number of bins for histograms (default: 30)
    """

    # Filter columns that actually exist in the dataframe
    available_columns = [col for col in columns if col in data.columns]

    if not available_columns:
        print("None of the specified columns found in dataframe!")
        print(f"Available columns: {list(data.columns)}")
        return

    # Calculate grid dimensions
    n_cols = len(available_columns)
    n_rows = int(np.ceil(n_cols / 3))
    n_plot_cols = min(3, n_cols)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(15, 4 * n_rows))

    # Flatten axes array for easier iteration
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_cols > 3 else axes

    # Plot each column
    for idx, col in enumerate(available_columns):
        ax = axes[idx]

        # Remove NaN and infinite values
        clean_data = data[col].replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean_data) == 0:
            ax.text(0.5, 0.5, f'No valid data for {col}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot histogram
        ax.hist(clean_data, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')

        # Add KDE overlay if data is suitable
        if len(clean_data) > 10 and clean_data.std() > 0:
            ax2 = ax.twinx()
            clean_data.plot.kde(ax=ax2, color='red', linewidth=2, alpha=0.7)
            ax2.set_ylabel('Density (KDE)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(False)

        # Calculate statistics
        mean_val = clean_data.mean()
        median_val = clean_data.median()
        std_val = clean_data.std()

        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3e}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3e}')

        # Set labels and title
        ax.set_xlabel(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Distribution of {col}\nStd: {std_val:.3e}, n={len(clean_data)}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(len(available_columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for col in available_columns:
        clean_data = data[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_data) > 0:
            print(f"\n{col}:")
            print(f"  Count:    {len(clean_data)}")
            print(f"  Mean:     {clean_data.mean():.6e}")
            print(f"  Median:   {clean_data.median():.6e}")
            print(f"  Std Dev:  {clean_data.std():.6e}")
            print(f"  Min:      {clean_data.min():.6e}")
            print(f"  Max:      {clean_data.max():.6e}")
            print(f"  Q1 (25%): {clean_data.quantile(0.25):.6e}")
            print(f"  Q3 (75%): {clean_data.quantile(0.75):.6e}")
            print(f"  Skewness: {clean_data.skew():.6f}")
            print(f"  Kurtosis: {clean_data.kurtosis():.6f}")

# plot_distributions(data, columns)



def plot_correlation_matrix(data, columns):
    """
    Plot correlation matrix for the specified columns
    """
    available_columns = [col for col in columns if col in data.columns]

    if len(available_columns) < 2:
        print("Need at least 2 columns for correlation matrix")
        return

    plt.figure(figsize=(12, 10))

    # Calculate correlation matrix
    corr = data[available_columns].corr()

    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})

    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
# plot_correlation_matrix(data, columns)