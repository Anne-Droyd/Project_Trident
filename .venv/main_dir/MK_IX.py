'''
probabilistic random forest
'''
import os
import json
import joblib
import math

from quantile_forest import RandomForestQuantileRegressor # best choice

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.utils import shuffle
from main_dir.Methods.Data_Options import data_options
from sklearn.metrics import mean_squared_error, r2_score

data_ops=data_options(type="VR_DATA",default_data=True)
model_folder_path = "C:/Users/Matth/Documents/Leiden University/Project/models/RF/"
history_path = "C:/Users/Matth/Documents/Leiden University/Project/Histories/RF/"
plot_path = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/plots/Random_forest/'

def train_model(train_x,train_y,min_samples_leaf=10):
    model = RandomForestQuantileRegressor(bootstrap=True,
                                          max_samples=0.8,
                                          n_estimators=1,
                                          # criterion='poisson', #minor improvement for mcore
                                          default_quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                                          oob_score=True,
                                          n_jobs=-1,
                                          max_features = None,
                                          min_samples_leaf=min_samples_leaf,
                                          min_samples_split=2*min_samples_leaf,
                                          max_samples_leaf=None,
                                          # max_leaf_nodes=1000, #only negatively impacts results
                                          # max_depth=500, #only negatively impacts results
                                          # min_impurity_decrease=1e-9, # negatively impacts results
                                          # ccp_alpha=1e-5, #untested
                                          random_state=42,
                                          verbose=0,
                                          warm_start=True)
    model.fit(X=train_x, y=train_y)
    print(f'Trees: {model.n_estimators}, OOB: {model.oob_score_}')

    model.n_estimators+=9
    model.fit(train_x,train_y)
    print(f'Trees: {model.n_estimators}, OOB: {model.oob_score_}')

    for _ in range(6):
        model.n_estimators += 10
        model.fit(train_x, train_y)
        print(f'Trees: {model.n_estimators}, OOB: {model.oob_score_}')


    return model

def plot_2d_hist(y_col,test_y,predictions):
    for idx, col in enumerate(y_col):

        fig = plt.figure(figsize=(6,4))
        mse = mean_squared_error(test_y[col], predictions[:,idx,2])
        print(f'{col} Mean Squared Error: {mse}')

        r2 = r2_score(test_y[col], predictions[:,idx,2])
        print(f'{col} R-squared: {r2}')
        min_val = min(test_y[col].min(), predictions[:, idx,2].min())
        max_val = max(test_y[col].max(), predictions[:, idx,2].max())
        h = plt.hist2d(test_y[col], predictions[:,idx,2], bins=100, cmap="viridis", cmax=100,density=False)
        fig.colorbar(h[3], label="Density")
        plt.plot([min_val, max_val], [min_val, max_val],'k--',lw=1,label='Perfect prediction')
        plt.title(f'Real vs predicted values for {col}')
        image_path = plot_path+f'real_vs_pred_{col}.png'
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
        # plt.show()

def plot_2d_hist_subplots(y_col, test_y, predictions, model_num):

    n = len(y_col)

    # Choose grid layout automatically
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,

    )

    axes = axes.flatten()
    h_last = None

    for idx, col in enumerate(y_col):
        ax = axes[idx]

        y_true = test_y[col]
        y_pred = predictions[:, idx, 2]  # median

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())

        h = ax.hist2d(
            y_true,
            y_pred,
            bins=100,
            cmap="viridis",
            cmax=100,
            density=False
        )
        h_last = h  # save for colorbar

        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
        ax.set_title(col,fontsize=24)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=14)

        ax.text(
            0.05, 0.97, f'$R^2 = {r2:.3f}$',
            transform=ax.transAxes,
            fontsize=24,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7)
        )
        ax.text(
            0.05, 0.83, f'$MSE = {mse:.3f}$',
            transform=ax.transAxes,
            fontsize=24,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7)
        )
        if idx + ncols >= len(y_col):
            ax.set_xlabel("True value",fontsize=24)
        current_col = idx % ncols
        if current_col == 0:
            ax.set_ylabel("Predicted value",fontsize=24)

    # Remove unused axes
    for j in range(len(y_col), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout for shared colorbar
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])  # leave space on the right

    # Add shared colorbar
    cbar = fig.colorbar(h_last[3], ax=axes[:len(y_col)], fraction=0.046, pad=0.04)
    cbar.set_label("Density",fontsize=24)

    fig.suptitle("Real vs Predicted Values", fontsize=28)
    plot_path = 'Plots/'
    plt.savefig(plot_path + f"real_vs_pred_all_targets_model_{model_num}.png", bbox_inches='tight')
    # plt.show()

def plot_distance_from_median(y_col,test_y,predictions,model_num):
    n = len(y_col)

    # Auto grid
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False
    )

    axes = axes.flatten()

    for idx, col in enumerate(y_col):
        ax = axes[idx]

        y_true = test_y[col].values

        q_low = predictions[:, idx, 0]
        q_med = predictions[:, idx, 2]
        q_high = predictions[:, idx, 4]

        distance = y_true - q_med

        low80, high80 = np.percentile(distance, [10, 90])

        # Center by interval midpoint
        center = 0.5 * (q_low + q_high)

        ax.hist(distance,bins=100)
        ax.axvspan(low80, high80, alpha=0.3)
        ax.set_ylim(0,1000)
        ax.set_title(f"{col}",fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=14)

    # Remove unused axes
    for j in range(len(y_col), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "distance from median",
        fontsize=28,
        y=0.93
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = 'Plots/'
    plt.savefig(plot_path + f"distance_from_median_{model_num}.png")
    # plt.show()

def plot_sorted_error(y_col,test_y,predictions,model_num,q_low_idx=0,q_med_idx=2,q_high_idx=4):
    n = len(y_col)

    # Auto grid
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for idx, col in enumerate(y_col):
        ax = axes[idx]

        y_true = test_y[col].values

        q_low = predictions[:, idx, q_low_idx]
        q_med = predictions[:, idx, q_med_idx]
        q_high = predictions[:, idx, q_high_idx]

        # Sort by interval width
        width = q_high - q_low
        order = np.argsort(width)

        q_low = q_low[order]
        q_med = q_med[order]
        q_high = q_high[order]
        y_true = y_true[order]

        # Center by interval midpoint
        center = 0.5 * (q_low + q_high)

        q_low_c = q_low - center
        q_high_c = q_high - center
        q_med_c = q_med - center
        y_true_c = y_true - center

        x = np.arange(len(y_true))

        # Interval bars
        ax.bar(
            x,
            q_high_c - q_low_c,
            bottom=q_low_c,
            width=1.0,
            color="#e0f2ff",
            edgecolor="none",
            label="Quantile interval",
            alpha = 1
        )

        # Median ticks
        # ax.plot(
        #     x,
        #     q_med_c,
        #     linestyle="None",
        #     marker="_",
        #     markersize=6,
        #     color="#006aff",
        #     label="Median"
        # )

        # True values
        ax.scatter(
            x,
            y_true_c,
            s=10,
            color="#f2a619",
            alpha=0.005,
            label="True value"
        )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

        # Empirical coverage
        coverage = np.mean((y_true >= q_low) & (y_true <= q_high))

        ax.set_title(f"{col}\nCoverage: {coverage:.2f}",fontsize=24)
        if idx + ncols >= len(y_col):
            ax.set_xlabel("Ordered samples",fontsize=24)
        current_col = idx % ncols
        if current_col == 0:
            ax.set_ylabel("Centered value",fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=14)

    # Remove unused axes
    for j in range(len(y_col), len(axes)):
        fig.delaxes(axes[j])

    # Shared legend (only once)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),  # x=0.5 (center), y=1.05 (above figure)
        ncol=3,
        fontsize=24
    )

    fig.suptitle(
        "Centered Prediction Intervals (sorted by uncertainty)",
        fontsize=28,
        y=0.93
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = 'Plots/'
    plt.savefig(plot_path + f"error_coverage_all_targets_model_{model_num}.png")
    # plt.show()

def plot_real_planets(y_col,y,predictions,model_name):
    planets = y["planet_name"]

    n = len(y_col)

    # Auto grid
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False
    )

    axes = axes.flatten()

    for idx, col in enumerate(y_col):
        ax = axes[idx]

        if col in y.columns:
            true = pd.to_numeric(y[col]).values
            has_true = True
        else:
            has_true = False

        q_low = predictions[:, idx, 0]
        q_med = predictions[:, idx, 2]
        q_high = predictions[:, idx, 4]

        # Center by interval midpoint
        center = 0.5 * (q_low + q_high)
        center = 0

        q_low_c = q_low - center
        q_high_c = q_high - center
        q_med_c = q_med - center

        x = np.arange(len(planets))
        labels = planets.values if hasattr(planets, "values") else planets

        # Interval bars
        ax.bar(
            x,
            q_high_c - q_low_c,
            bottom=q_low_c,
            width=1.0,
            color="#e0f2ff",
            edgecolor="none",
            label="Quantile interval"
        )

        # Median ticks
        ax.plot(
            x,
            q_med_c,
            linestyle="None",
            marker="_",
            markersize=8,
            color="#006aff",
            label="Median prediction"
        )

        if has_true:
            ax.plot(
                x,
                true,
                linestyle="None",
                marker="o",
                markersize=4,
                color="r",
                label="Literature value"
            )

        current_row = idx // ncols
        current_col = idx % ncols

        ax.set_title(f"{col}",fontsize=24)
        ax.set_ylim(0,1)

        if col == 'm_core' or col == 'ice_mass' or col == 'rock_mass' or col == 'h_he_mass':
            ax.set_ylabel("Mass fraction",fontsize=24)
        else:
            ax.set_ylabel("Metallicity",fontsize=24)


        ax.set_xticks(x)
        if idx + ncols >= len(y_col):
            ax.set_xticklabels(labels, rotation=45, ha="right",fontsize=24)
        else:
            ax.set_xticklabels([])



    # Remove unused axes
    for j in range(len(y_col), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),  # x=0.5 (center), y=1.05 (above figure)
        ncol=3,
        fontsize=24
    )

    fig.suptitle(
        "known planet predictions",
        fontsize=28,
        y=0.93
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = 'Plots/'
    plt.savefig(plot_path + f"known_planets_{model_name}.png")
    # plt.show()

def main(default_model = False,
         train_new_model = False,
         save_tree_image = False,
         k2 = False,
         extra_params = False,
         results = False,
         cepam_results = False,
         iteration = None):
    data = data_ops.get_data()
    data = shuffle(data, random_state=42)
    # train, test, valid = data_ops.partition_data(data, train_frac=0.8, test_frac=0.2, valid_frac=0)
    train, test, valid = data_ops.partition_data(data, train_frac=0.99, test_frac=0.01, valid_frac=0)
    if iteration == None:
        iteration = data_ops.get_iteration(model_folder_path, "RF")
    else: iteration = iteration
    print("Current input model iteration #", iteration)

    current_model_save_name = f"RF_input_model_{iteration}.joblib"
    current_metadata_save_name = f"RF_input_model_{iteration}_metadata.json"
    history_name = f"RF_history_{iteration}.csv"

    if extra_params:
        y_col = ["m_core", 'zatm0','zatm1','zdeep','zdeep0','zdeep1',"ice_mass", "rock_mass",'h_he_mass'] #,'matm','mdeep',"ice_mass", "rock_mass",'h_he_mass']
    else:
        y_col = ["m_core",  'zatm0', 'zatm1', 'zdeep', 'zdeep0', 'zdeep1']
    if k2:
        x_col = ["mass", "radius", 'k2']
    else:
        x_col = ["mass", "radius"]
    train_x, train_y = data_ops.get_xy(train, y_col, x_col)
    test_x, test_y = data_ops.get_xy(test, y_col, x_col)

    if train_new_model == True:
        model = train_model(train_x,train_y,min_samples_leaf=3)
        model_path = os.path.join(model_folder_path, current_model_save_name)
        metadata_path = os.path.join(model_folder_path, current_metadata_save_name)
        joblib.dump(model, model_path)
        metadata = {
            "model_type": "RandomForestRegressor",
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "train_x_columns": list(train_x.columns),
            "train_y_columns": list(train_y.columns),
            "params": model.get_params(),
            "oob_score": getattr(model, "oob_score_", None)
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
    else:
        if default_model == True:
            if k2:
                model_number = 83
            else:
                model_number = 84
            model_name = f"RF_input_model_{model_number}.joblib"
            meta_name = f"RF_input_model_{model_number}_metadata.json"
        else:
            model_name = f"RF_input_model_{iteration-1}.joblib"
            meta_name = f"RF_input_model_{iteration-1}_metadata.json"
        try:
            model = joblib.load(os.path.join(model_folder_path, model_name))
            with open(os.path.join(model_folder_path, meta_name), "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f'Failed to load a model: {e}')
            model = train_model(train_x, train_y)
            model_path = os.path.join(model_folder_path, current_model_save_name)
            metadata_path = os.path.join(model_folder_path, current_metadata_save_name)
            joblib.dump(model, model_path)
            metadata = {
                "model_type": "RandomForestRegressor",
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "train_x_columns": list(train_x.columns),
                "train_y_columns": list(train_y.columns),
                "params": model.get_params(),
                "oob_score": getattr(model, "oob_score_", None)
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
    if save_tree_image == True:
        fn = train_x.columns
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        tree.plot_tree(
            model.estimators_[0],
            feature_names=fn,
            filled=True,
            rounded=True,
            fontsize=6,
            max_depth=4
        )
        tree_path = plot_path+f'rf_individualtree_model_{iteration}.png'
        fig.savefig(tree_path)
        # plt.show()
        print(f'Individual tree image created and saved to :{tree_path}')

    oob_score = getattr(model, "oob_score_", None)
    if oob_score is not None:
        print(f'Out-of-Bag Score: {oob_score}')

    predictions = model.predict(test_x)

    plot_2d_hist_subplots(y_col,test_y,predictions,iteration)
    plot_sorted_error(y_col,test_y,predictions,iteration)
    plot_distance_from_median(y_col,test_y,predictions,iteration)

    if results:
        result_iter = data_ops.get_iteration('./results', file_prefix='MK_IX')
        quantile_names = ["q05", "q25", "median", "q75", "q95"]
        columns = [f"{col}_{q}" for col in y_col for q in quantile_names]
        flat_pred = predictions.reshape(-1, len(y_col) * 5)
        results = pd.DataFrame(flat_pred, columns=columns)
        results.to_csv(f'./results/MK_IX_results_include_mass_radius_T0_ppt_{result_iter}.csv')
    if cepam_results:
        result_iter = data_ops.get_iteration('./results', file_prefix='MK_IX')
        quantile_names = ["q05", "q25", "median", "q75", "q95"]
        columns = [f"{col}_{q}" for col in y_col for q in quantile_names]
        flat_pred = predictions.reshape(-1, len(y_col) * 5)
        results = pd.DataFrame(flat_pred, columns=columns)
        results['mass'] = test_x['mass'].values
        results['radius'] = test_x['radius'].values
        results['T0'] = test_x_stored['T0'].values
        results['p_ppt'] = test_x_stored['p_ppt'].values

        results.to_csv(f'./results/MK_IX_results_include_mass_radius_T0_ppt_{result_iter}.csv')


def test_on_known_planets(k2=False,extra_params = False,base=None):
    ester_path = os.path.join('./data', 'Ester_planets.dat')
    solar_path = os.path.join('./data', 'Solar_system_planets.dat')
    data = pd.read_csv(solar_path, delimiter="\s+")

    # y_col = ["m_core",'zatm','zdeep']
    if extra_params:
        y_col = ["m_core", 'zatm', 'zatm0', 'zatm1', 'zdeep', 'zdeep0', 'zdeep1','ice_mass','rock_mass','h_he_mass']
        y = ['planet_name','m_core','zatm','zdeep','ice_mass','rock_mass','h_he_mass']
    else:
        y_col = ["m_core", 'zatm', 'zatm0', 'zatm1', 'zdeep', 'zdeep0', 'zdeep1']
        y = ['planet_name', 'm_core', 'zatm', 'zdeep']
    if not k2:
        x_col = ["mass", "radius"]
    else:
        x_col = ["mass", "radius",'k2']
    x, y = data_ops.get_xy(data, y, x_col)


    if not k2:
        if base is None:
            base = 76
        model_file = f"RF_input_model_{base}.joblib"
        model_name = f"MR_{base}"
    else:
        if base is None:
            base = 77
        model_file = f"RF_input_model_{base}.joblib"
        model_name = f"MRK2_{base}"

    model = joblib.load(os.path.join(model_folder_path, model_file))

    predictions = model.predict(x)

    plot_real_planets(y_col,y,predictions,model_name)

def plot_known_planet_with_error_medians(y_col, predictions, model_name):

    n = len(y_col)
    n_mc = predictions.shape[0]

    # Auto grid
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        sharey=False
    )

    axes = axes.flatten()

    for idx, col in enumerate(y_col):
        ax = axes[idx]

        q_med = predictions[:, idx, 2]

        # Histogram of median predictions
        ax.hist(
            q_med,
            bins=30,
            density=True,
            color="#e0f2ff",
            edgecolor="none"
        )

        # Median marker
        ax.axvline(
            np.median(q_med),
            color="#006aff",
            linestyle="--",
            linewidth=2,
            label="Median"
        )

        ax.set_title(col, fontsize=28)

        if idx + ncols >= len(y_col):
            ax.set_xlabel("Median predicted value", fontsize=24)

        if idx % ncols == 0:
            ax.set_ylabel("Probability density", fontsize=24)

        ax.tick_params(axis="both", labelsize=18)

    # Remove unused axes
    for j in range(len(y_col), len(axes)):
        fig.delaxes(axes[j])

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            fontsize=24
        )

    fig.suptitle(
        "GJ436b — Distribution of Median Interior Predictions",
        fontsize=28,
        y=0.93
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    plot_path = "Plots/"
    plt.savefig(plot_path + f"GJ436b_medians_{model_name}.png")

def plot_known_planet_with_error_sorted_uncertain(y_col, predictions, model_name):

    n = len(y_col)
    n_mc = predictions.shape[0]

    # Auto grid
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for idx, col in enumerate(y_col):
        ax = axes[idx]

        q_low  = predictions[:, idx, 0]
        q_med  = predictions[:, idx, 2]
        q_high = predictions[:, idx, 4]

        # Sort by interval width (uncertainty)
        width = q_high - q_low
        order = np.argsort(width)

        q_low  = q_low[order]
        q_med  = q_med[order]
        q_high = q_high[order]

        # Center by interval midpoint
        center = 0.5 * (q_low + q_high)

        q_low_c  = q_low  - center
        q_high_c = q_high - center
        q_med_c  = q_med  - center

        x = np.arange(n_mc)

        # Interval bars
        ax.bar(
            x,
            q_high_c - q_low_c,
            bottom=q_low_c,
            width=1.0,
            color="#e0f2ff",
            edgecolor="none",
            alpha=1.0,
            label="Quantile interval"
        )

        # Median ticks
        ax.plot(
            x,
            q_med_c,
            linestyle="None",
            marker="_",
            markersize=6,
            color="#006aff",
            label="Median prediction"
        )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

        ax.set_title(col, fontsize=24)

        if idx + ncols >= len(y_col):
            ax.set_xlabel("MC samples (sorted by uncertainty)", fontsize=28)

        if idx % ncols == 0:
            ax.set_ylabel("Centered value", fontsize=28)

        ax.tick_params(axis="both", labelsize=14)

    # Remove unused axes
    for j in range(len(y_col), len(axes)):
        fig.delaxes(axes[j])

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        fontsize=28
    )

    fig.suptitle(
        "GJ436b — Interior Predictions (Monte Carlo)",
        fontsize=28,
        y=0.93
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    plot_path = "Plots/"
    plt.savefig(plot_path + f"GJ436b_sorted_errors{model_name}.png")

def sample_asymmetric_normal(mu, sigma_minus, sigma_plus, size=1):
    """
    Draw samples from a two-sided normal distribution.
    """
    u = np.random.uniform(0, 1, size)
    samples = np.empty(size)

    left = u < (sigma_minus / (sigma_minus + sigma_plus))
    right = ~left

    samples[left] = mu - np.abs(
        np.random.normal(0, sigma_minus, left.sum())
    )
    samples[right] = mu + np.abs(
        np.random.normal(0, sigma_plus, right.sum())
    )

    return samples

def run_planet_with_uncertainty(k2=False,extra_params=False,base=76):
    GJ436b = os.path.join('./data', 'known_planet_constraints.dat')
    data = pd.read_csv(GJ436b, delimiter="\s+")
    #selecting one uncertainty
    data = data[0:1]
    # data = data[1:]
    # y_col = ["m_core",'zatm','zdeep']
    if extra_params:
        y_col = ["m_core", 'zatm', 'zatm0', 'zatm1', 'zdeep', 'zdeep0', 'zdeep1', 'ice_mass', 'rock_mass', 'h_he_mass']
        y = []
    else:
        y_col = ["m_core", 'zatm', 'zatm0', 'zatm1', 'zdeep', 'zdeep0', 'zdeep1']
        y = []
    if not k2:
        x_col = ["mass", 'mass_err1','mass_err2','radius','radius_err1','radius_err2']
    else:
        x_col = ["mass", 'mass_err1','mass_err2','radius','radius_err1','radius_err2', 'k2', 'k2_err1', 'k2_err2']
    x_distributions, y = data_ops.get_xy(data, y, x_col)

    if not k2:
        if base is None:
            base = 76
        model_file = f"RF_input_model_{base}.joblib"
        model_name = f"MR_{base}"
    else:
        if base is None:
            base = 77
        model_file = f"RF_input_model_{base}.joblib"
        model_name = f"MRK2_{base}"

    #make random samples
    x = pd.DataFrame(columns=['mass','radius'])
    mu_m = x_distributions['mass'].iloc[0]
    sigm_m = x_distributions['mass_err2'].iloc[0]
    sigp_m = x_distributions['mass_err1'].iloc[0]

    mu_r = x_distributions['radius'].iloc[0]
    sigm_r = x_distributions['radius_err2'].iloc[0]
    sigp_r = x_distributions['radius_err1'].iloc[0]




    x['mass'] = sample_asymmetric_normal(mu_m, sigm_m, sigp_m, size=1000)
    x['radius'] = sample_asymmetric_normal(mu_r, sigm_r, sigp_r, size=1000)
    if k2:
        mu_k2 = x_distributions['k2'].iloc[0]
        sigm_k2 = x_distributions['k2_err2'].iloc[0]
        sigp_k2 = x_distributions['k2_err1'].iloc[0]
        # x['k2'] = sample_asymmetric_normal(mu_k2, sigm_k2, sigp_k2, size=1000)
        x['k2'] = np.random.uniform(0, 0.8, size=1000)

    print(x)

    model = joblib.load(os.path.join(model_folder_path, model_file))

    predictions = model.predict(x)

    plot_known_planet_with_error_sorted_uncertain(y_col, predictions, model_name)
    plot_known_planet_with_error_medians(y_col,predictions,model_name)

# test_on_known_planets(k2=True,extra_params=False,base=77)
# test_on_known_planets(k2=False,extra_params=False,base=76)
# main(train_new_model=False,default_model=True,k2=True,extra_params=True,iteration=83)
# main(train_new_model=False,default_model=True,k2=False,extra_params=True,iteration=84)
# test_on_known_planets(k2=True,extra_params=True,base=83)
# test_on_known_planets(k2=False,extra_params=True,base=84)
#
# run_planet_with_uncertainty(k2=False,extra_params=True,base=84)
# run_planet_with_uncertainty(k2=True,extra_params=True,base=83)
main(train_new_model=True,default_model=True,k2=True,extra_params=True)
main(train_new_model=True,default_model=True,k2=False,extra_params=True)