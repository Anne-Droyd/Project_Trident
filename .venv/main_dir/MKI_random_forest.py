import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from main_dir.Methods.Data_Options import data_options
from sklearn.metrics import mean_squared_error, r2_score
from main_dir.Methods import Plotting as plots

data_ops=data_options(type="VR_DATA",default_data=True)
model_folder_path = "C:/Users/Matth/Documents/Leiden University/Project/models/RF/"
history_path = "C:/Users/Matth/Documents/Leiden University/Project/Histories/RF/"
plot_path = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/plots/Random_forest/'

def train_model(train_x,train_y):
    regressor = RandomForestRegressor(n_estimators=200, max_depth=None, bootstrap=True, oob_score=True)
    model = regressor.fit(train_x, train_y)
    return model

def main(default_model = False,train_new_model = True, save_tree_image = True):
    data = data_ops.get_data()
    data = shuffle(data, random_state=42)
    # data = data[:15000]
    train, test, valid = data_ops.partition_data(data, train_frac=0.8, test_frac=0.1, valid_frac=0.1)
    iteration = data_ops.get_iteration(model_folder_path, "RF")
    print("Current input model iteration #", iteration)

    current_model_save_name = f"RF_input_model_{iteration}.joblib"
    current_metadata_save_name = f"RF_input_model_{iteration}_metadata.json"
    history_name = f"RF_history_{iteration}.csv"

    y_col = ["m_core", "ice_mass", "rock_mass", 'h_he_mass','k2']
    x_col = ["mass", "radius", "temp"]
    train_x, train_y = data_ops.get_xy(train, y_col, x_col)
    test_x, test_y = data_ops.get_xy(test, y_col, x_col)
    valid_x, valid_y = data_ops.get_xy(valid, y_col, x_col)

    if train_new_model == True:
        model = train_model(train_x,train_y)
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
            model_name = f"RF_input_model_{1}.joblib"
            meta_name = f"RF_input_model_{1}_metadata.json"
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
        plt.show()
        print(f'Individual tree image created and saved to :{tree_path}')

    oob_score = model.oob_score_
    print(f'Out-of-Bag Score: {oob_score}')

    predictions = model.predict(test_x)

    for idx, col in enumerate(y_col):
        fig = plt.figure(figsize=(6,4))
        mse = mean_squared_error(test_y[col], predictions[:,idx])
        print(f'{col} Mean Squared Error: {mse}')

        r2 = r2_score(test_y[col], predictions[:,idx])
        print(f'{col} R-squared: {r2}')
        min_val = min(test_y[col].min(), predictions[:, idx].min())
        max_val = max(test_y[col].max(), predictions[:, idx].max())
        h = plt.hist2d(test_y[col], predictions[:,idx], bins=100, cmap="jet", cmax=100, density=True)
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
        plt.show()

main()