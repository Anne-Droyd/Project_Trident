"""
Rebuilding from scratch to make sure everything makes sense
"""
import keras
import keras.backend as K
import gc
import pickle
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from keras import callbacks, Sequential, regularizers
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from Methods import Plotting as plots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Methods.EXOMDN_MDN_layer import MDN, softmax, elu_plus_one_plus_epsilon, get_mixture_loss_func
from Methods.Data_Options import data_options
from Methods.EXOMDN_MDN_layer import get_mixture_loss_func as MDNLoss

data_ops = data_options(type="VR_DATA", computer="laptop")


def unpack_mdn_predictions(predictions, output_dim, num_mixtures,scaler,y_col):
    out_mu, out_sigma, out_pi = np.split(predictions, indices_or_sections=[
        num_mixtures * output_dim,
        2 * num_mixtures * output_dim], axis=-1)

    # Reshape:
    mus = np.reshape(out_mu, (-1, num_mixtures, output_dim))  # (samples, mixtures, outputs)
    sigmas = np.reshape(out_sigma, (-1, num_mixtures, output_dim))  # (samples, mixtures, outputs)
    pi = np.reshape(out_pi, (-1, num_mixtures))  # (samples, mixtures)

    mus_flat = mus.reshape(-1, output_dim)
    mus = scaler.inverse_transform(mus_flat)
    mus = mus.reshape(-1, num_mixtures, output_dim)

    scale = scaler.scale_
    sigmas = sigmas * scale.reshape(1, 1, -1)

    pi = np.apply_along_axis(softmax, 1, pi, temperature=1)

    return mus, sigmas, pi

def main():

    # laptop
    model_folder_path = "C:/Users/Matth/Documents/GitHub/first_research_project/.venv/Project_first_model/models/"
    models=["MDN_model_210.h5",
            "MDN_model_211.h5",
            "MDN_model_212.h5",
            "MDN_model_213.h5",
            "MDN_model_214.h5",
            "MDN_model_215.h5",
            "MDN_model_216.h5",
            "MDN_model_217.h5",
            "MDN_model_218.h5",
            "MDN_model_219.h5",
            "MDN_model_220.h5",
            "MDN_model_221.h5",
            "MDN_model_222.h5",
            "MDN_model_223.h5",
            "MDN_model_224.h5",
            "MDN_model_225.h5",
            "MDN_model_226.h5",
            "MDN_model_227.h5",
            "MDN_model_228.h5",
            "MDN_model_229.h5"]
    num_mixtures = 120
    # lr = 0.001
    lr = 0.001
    optimizer = Adam(learning_rate=lr)

    y_col = ["m_core", "zatm", "zdeep"]
    x_col = ["mass", "req", "Teq"]

    data = data_ops.get_data()
    expansion_factor = 0.01
    data = data.sample(n=(int(expansion_factor * len(data))), replace=True, random_state=42)

    train, test, valid = data_ops.partition_data(data, train_frac=0.8, test_frac=0.1, valid_frac=0.1)

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

    input_dim = train_x.shape[1]
    output_dim = train_y.shape[1]

    mean_sigs = pd.DataFrame(columns=y_col)
    for Model_save_name in models:
        model = keras.models.load_model(
            model_folder_path + Model_save_name,
            custom_objects={"MDN": MDN}
        )
        model.compile(loss=MDNLoss(output_dim, num_mixtures), optimizer=optimizer)

        predictions = model.predict(test_x)
        mu, sigma, pi = unpack_mdn_predictions(predictions, output_dim, num_mixtures,y_scaler,y_col)

        mean_values = np.mean(sigma, axis=(0, 1))
        print(mean_values)
        mean_sigs = pd.concat([mean_sigs, pd.DataFrame([mean_values], columns=y_col)], ignore_index=True)

    print(mean_sigs)
    fig,axs=plt.subplots(3,2,sharex=True,sharey=True)
    row=0
    col=0
    for column in y_col:
        axs[row,col].scatter([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0],mean_sigs[column])
        axs[row,col].set_title(column)
        col +=1
        if col ==2:
            col=0
            row+=1
    fig.suptitle("Mean deviation vs sample size (120 mixtures, 60 nodes)")

    for ax in axs.flat:
        ax.set(xlabel="Percentage of dataset (%)", ylabel="Mean deviation $\sigma$")
    for ax in axs.flat:
        ax.label_outer()
    plt.tight_layout()
    plt.show()

def fix_data():
    earth_mass_kg = 5.972e24
    data_ops = data_options(type="VR_DATA")
    data = data_ops.get_data()

    data_ops = data_options(type="BAU")
    bau = data_ops.get_data()

    df_a = data.rename(columns={"req": "radius", "Teq": "temp"})
    df_b = bau.rename(columns={'planet_mass': 'mass', 'planet_radius': 'radius', 'T_eq': 'temp'})

    df_a = df_a.loc[:,["mass","radius","temp","k2","lum","m_core","zatm","zatm0","zatm1","zdeep","zdeep0","zdeep1",
                        "p_ppt","p_rot","test","density"]]

    df_a["log_m_core"]=np.log10(df_a["m_core"])
    df_a["log_zatm"] = np.log10(df_a["zatm"])
    df_a["log_zatm0"] = np.log10(df_a["zatm0"])
    df_a["log_zatm1"] = np.log10(df_a["zatm1"])
    df_a["log_zdeep"] = np.log10(df_a["zdeep"])
    df_a["log_zdeep0"] = np.log10(df_a["zdeep0"])
    df_a["log_zdeep1"] = np.log10(df_a["zdeep1"])
    print(df_a.loc[:,["log_m_core","log_zatm","log_zdeep"]])
    df_b["m_core"] = df_b["mass"]/((10**df_b["log_m_mantle_core"]+10**df_b["log_m_water_core"]+10**df_b["log_m_atmosphere_core"])+1)
    df_b["m_core"] = df_b["m_core"]/df_b["mass"]

    def run_pca(df, features, n_components=2):
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(
            pcs,
            columns=[f"PC{i + 1}" for i in range(n_components)],
            index=X.index
        )
        return pca_df, pca.explained_variance_ratio_, pca.components_

    # Features to use
    features = ["m_core", "mass", "radius", "temp"]

    # Run PCA for df_a
    pca_a_df, var_a, comp_a = run_pca(df_a, features)

    # Run PCA for df_b
    pca_b_df, var_b, comp_b = run_pca(df_b, features)

    # Show explained variance ratios
    print("df_a explained variance ratio:", var_a)
    print("df_b explained variance ratio:", var_b)

    plt.scatter(pca_a_df["PC1"], pca_a_df["PC2"], alpha=0.01, label="VR_DATA")
    plt.scatter(pca_a_df["PC1"], pca_a_df["PC2"], alpha=0.01, label="BAU_DATA")
    plt.title("PC1 vs PC2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()


    plt.scatter(df_a["m_core"], df_a["mass"], alpha=0.1, label="VR_DATA")
    plt.scatter(df_b["m_core"], df_b["mass"], alpha=0.1, label="BAU_DATA")
    plt.title("m_core vs mass")
    plt.xlabel("m_core")
    plt.ylabel("mass")
    plt.legend()
    plt.show()

    plt.scatter(df_a["m_core"], df_a["radius"], alpha=0.1, label="VR_DATA")
    plt.scatter(df_b["m_core"], df_b["radius"], alpha=0.1, label="BAU_DATA")
    plt.title("m_core vs radius")
    plt.xlabel("m_core")
    plt.ylabel("radius")
    plt.legend()
    plt.show()
    plt.scatter(df_a["m_core"], df_a["temp"], alpha=0.1, label="VR_DATA")
    plt.scatter(df_b["m_core"], df_b["temp"], alpha=0.1, label="BAU_DATA")
    plt.title("m_core vs temp")
    plt.xlabel("m_core")
    plt.ylabel("temp")
    plt.legend()
    plt.show()
    # plt.scatter(df_a["mass"],df_a["radius"],alpha=0.1,label="VR_DATA")
    # plt.scatter(df_b["mass"],df_b["radius"],alpha=0.1,label="BAU_DATA")
    # plt.title("mass vs radius")
    # plt.xlabel("mass")
    # plt.ylabel("radius")
    # plt.legend()
    # plt.show()
    # plt.scatter(df_a["mass"], df_a["temp"], alpha=0.1, label="VR_DATA")
    # plt.scatter(df_b["mass"], df_b["temp"], alpha=0.1, label="BAU_DATA")
    # plt.title("mass vs temp")
    # plt.xlabel("mass")
    # plt.ylabel("temp")
    # plt.legend()
    # plt.show()
    # plt.scatter(df_a["radius"], df_a["temp"], alpha=0.1, label="VR_DATA")
    # plt.scatter(df_b["radius"], df_b["temp"], alpha=0.1, label="BAU_DATA")
    # plt.title("radius vs temp")
    # plt.xlabel("radius")
    # plt.ylabel("temp")
    # plt.legend()
    # plt.show()
    # plt.scatter(df_a["log_m_core"],df_a["radius"],alpha=0.1,label="VR_DATA")
    # plt.scatter(df_b["log_m_mantle_core"],df_b["radius"],alpha=0.1,label="BAU_DATA")
    # plt.title("m_core vs radius")
    # plt.xlabel("m_core")
    # plt.ylabel("radius")
    # plt.legend()
    # plt.show()
    # plt.scatter(df_a["log_m_core"],df_a["mass"],alpha=0.1,label="VR_DATA")
    # plt.scatter(df_b["log_m_mantle_core"],df_b["mass"],alpha=0.1,label="BAU_DATA")
    # plt.title("m_core vs mass")
    # plt.xlabel("m_core")
    # plt.ylabel("mass")
    # plt.legend()
    # plt.show()

def nasa_data_check():
    import pandas as pd
    import matplotlib.pyplot as plt

    # --- Load CSV, skipping metadata before the header ---
    file_path = r"C:\Users\Matth\Downloads\PS_2025.08.18_12.38.22.csv"
    df = pd.read_csv(file_path, skiprows=96)
    df = df.groupby("pl_name", as_index=False).agg({"pl_bmasse": "mean"})

    # --- Define bins and labels ---
    bins = [0, 0.1, 0.5, 2, 10, 50, 5000]
    labels = ["mercurians", "sub-earths", "earths", "super-Earths", "Neptunians", "Jovians"]

    # --- Bin data ---
    df["mass_category"] = pd.cut(df["pl_bmasse"], bins=bins, labels=labels, include_lowest=True)

    # # --- Add NaN/blank category ---
    # df["mass_category"] = df["mass_category"].cat.add_categories(["NaN/blank"])
    # df["mass_category"] = df["mass_category"].fillna("NaN/blank")
    #
    # # --- Count values in each category ---
    # labels_with_nan = labels + ["NaN/blank"]
    counts = df["mass_category"].value_counts().reindex(labels)

    # --- Compute percentages ---
    percentages = counts / counts.sum() * 100

    # --- Print counts and percentages ---
    print(pd.DataFrame({"Count": counts, "Percentage": percentages.round(2)}))

    # --- Plot bar chart with percentages ---
    plt.figure(figsize=(9, 5))
    ax = counts.plot(kind="bar", color="skyblue", edgecolor="black")

    plt.title("Planet Mass Categories (pl_bmasse)")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # Add percentage labels on top of bars
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        ax.text(i, count + 1, f"{pct:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

# nasa_data_check()
fix_data()

# main()