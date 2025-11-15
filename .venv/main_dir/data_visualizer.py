

import umap
import corner

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from random import shuffle

from sklearn.manifold import TSNE
from matplotlib.ticker import MaxNLocator
from tkinter.filedialog import askdirectory
from sklearn.metrics import explained_variance_score
from main_dir.Methods.Data_Options import data_options
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def data_prep(dataset, input_columns = None, target_column = None, scaling = "standard"):

    if target_column == None:
        raise TypeError("Need a target column")

    if input_columns is None:
        input_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in input_columns:
            input_columns.remove(target_column)

    if scaling is None:
        target_data = dataset[target_column]
        return input_columns, target_data, input_columns, target_column

    scaling = scaling.lower()
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        raise TypeError("Unknown scaling method")



    input_data_scaled = scaler.fit_transform(dataset[input_columns])
    target_data = dataset[target_column]

    return input_data_scaled, target_data, input_columns, target_column

def data_reduction_method(X,method = "pca", n_components=2, **kwargs):

    method = method.lower()
    if method == "pca":
        reducer=PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
        return X_reduced

    elif method == "k_pca":
        reducer = KernelPCA(n_components=n_components,
                            kernel='rbf',
                            gamma=0.01)
        X_reduced = reducer.fit_transform(X)
        return X_reduced

    elif method == "t-sne":
        reducer = TSNE( n_components=n_components,
                        perplexity=15,
                        early_exaggeration=12,
                        learning_rate='auto',
                        random_state=42)
        X_reduced = reducer.fit_transform(X)
        return X_reduced

    elif method == "umap":

        reducer = umap.UMAP(n_neighbors=15,
                            n_components=n_components,
                            min_dist=0.1,
                            n_epochs=None,
                            learning_rate=1.0,
                            metric = 'euclidean',
                            random_state=42)
        X_reduced = reducer.fit_transform(X)
        return X_reduced

    else:
        raise TypeError(f"Unknown M-E-T-H-O-D MAN: {method}")


def plot_comparison_of_reductions(dict, y, label,save_fig=True,save_dir=None,show_fig=False):
    keys = list(dict.keys())
    if len(keys) > 4:
        raise ValueError(f'Too many keys {keys}')
    if len(dict) == 0:
        raise TypeError('Method tracker is empty, what are you doing??')

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for idx, key in enumerate(keys):
        dataframe = pd.DataFrame(dict[key])
        scatter = axs[idx].scatter(dataframe[0], dataframe[1], c=y, cmap='jet', alpha=0.6, vmax=2,vmin=-2) #, vmax=2,vmin=-2 add to fix scale
        axs[idx].set_title(f'{key.upper()}')
        axs[idx].set_xlabel('Component 1')
        axs[idx].set_ylabel('Component 2')

    # Hide unused subplots
    for idx in range(len(keys), 4):
        axs[idx].axis('off')

    fig.subplots_adjust(right=0.9)
    cbar = fig.colorbar(scatter, ax=axs, fraction=0.046, pad=0.04)

    cbar.set_label(label=label)
    fig.suptitle(f'Dimensionality Reduction Comparison - {label}')
    if save_dir == None:
        save_dir=askdirectory()
    if save_fig == True:
        iter_op = data_options()
        iteration=iter_op.get_iteration(path=save_dir,file_prefix=f'reduction_{label}')
        file_name=f'/data_reduction_{label}_{iteration}.png'
        path = save_dir+file_name
        plt.savefig(path)
    if show_fig == True:
        plt.show()


def plot_3D_rep_data(X, y, X_cols, y_col,save_fig=True,save_dir=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    color_map=ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap="jet")
    ax.set_xlabel(X_cols[0])
    ax.set_ylabel(X_cols[1])
    ax.set_zlabel(X_cols[2])
    fig.suptitle("3D representation of data")
    fig.colorbar(color_map,label = y_col)
    fig.tight_layout()
    if save_dir == None:
        save_dir=askdirectory()
    if save_fig == True:
        iter_op = data_options()
        iteration=iter_op.get_iteration(path=save_dir,file_prefix='3D')
        file_name=f'/3D_rep_{iteration}.png'
        path = save_dir+file_name
        plt.savefig(path)
    plt.show()

def plot_explained_var_score(X,in_cols):
    pca = PCA(n_components=len(in_cols))
    pca.fit(X)
    pca.fit_transform(X)
    var_score = pca.explained_variance_ratio_

    fig,axs = plt.subplots(1,2)
    fig.suptitle("Principle Component Analysis")
    axs[0].bar(range(1,len(var_score)+1),var_score*100)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set(xlabel="Principle Component",ylabel="Explained variance (%)")

    axs[1].plot(range(1,len(var_score)+1),np.cumsum(var_score)*100,'o-',linewidth=2,markersize=8)
    axs[1].axhline(y=90, color='r', linestyle='--', label='90% threshold')
    axs[1].axhline(y=95, color='orange', linestyle='--', label='95% threshold')
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set(xlabel="Principle Component",ylabel="Cumulative Explained variance (%)")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_histogram(X,x_labels):
    X = np.asarray(X)
    save_dir = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/plots/Simple_data'
    for idx, label in enumerate(x_labels):
        plt.hist(X[:,idx],bins=50,density=True)
        plt.xlabel(label)
        plt.ylabel('Probability')
        plt.title(f"Distribution of {label}")

        file_name = f'/Hist_rep_{label}.png'
        path = save_dir + file_name
        plt.savefig(path)

def corner_plot(X_Y_df):
    savedir = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/plots/corner_plot.jpg'
    corner.corner(X_Y_df,labels=X_Y_df.columns,show_titles=True)
    plt.savefig(savedir)
    plt.show()

def plot_X_v_Y_subplots(x,y,x_label=None,y_label=None,title=None,save_dir=None):
    #automatically make a 4 plot figure, where there are 4 y values 1 x value
    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(8,8))
    axes=axes.flatten()
    for plot in range(4):
        axes[plot].scatter(x,y[y_label[plot]])
        axes[plot].set(xlabel=x_label,ylabel=y_label[plot])
        axes[plot].set_title(y_label[plot])
    fig.suptitle(title)
    save_dir += f'{title}.jpg'
    fig.tight_layout()
    plt.savefig(save_dir)
    plt.show()

def main():
    data_ops = data_options(type="VR_DATA",default_data=True)
    data = data_ops.get_data()
    data = data.sample(frac=1)
    # corner_plot(data)
    data = data.loc[data['p_ppt']>=4.9]
    data = data.loc[data['p_ppt']<=5.1]
    # data = data[:2500]
    x_col_1 = ["mass","radius","temp"]
    x_col_2 = ["mass","radius","temp","k2"]
    x_col_3 = ['mass', 'radius', 'temp', 'density', 'surface_gravity','mass_radius_ratio', 'temp_density_interaction']
    x_col_4 = ['mass', 'radius', 'temp', 'k2', 'density', 'surface_gravity','mass_radius_ratio', 'temp_density_interaction']
    x_col_5 = ["mass", "radius", 'surface_gravity','mass_radius_ratio',
               'temp_density_interaction','k2']
    y_col_0 = ["m_core","ice_mass","rock_mass","h_he_mass"]
    y_col_1 = ["m_core","ice_mass","rock_mass","h_he_mass",'matm','mdeep'] #new data
    y_col_2 = ['m_core','m_mantle','matm','m_water'] # baumeister
    y_col_3 = ['m_core','zatm','zatm0','zatm1','zdeep','zdeep0','zdeep1'] #original data
    methods_1 = ["pca","k_pca","t-sne","umap"]

    methods = methods_1

    save_dir = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/plots/'

    #just storing column names for lazy reasons
    x_col = x_col_1
    y_columns = y_col_0

    n=0

    X, Y, in_cols, out_cols = data_prep(data, x_col, y_columns)
    cols=in_cols+out_cols
    # joined = pd.DataFrame(np.hstack((X,Y)),columns=cols)
    # print(X[:,1])
    # plot_X_v_Y_subplots(X[:, 1],Y,x_label='radius',y_label=y_columns,title='radius vs parameters of interest reduced ppt',save_dir=save_dir)
    # plot_X_v_Y_subplots(X[:, 3], Y, x_label='k2', y_label=y_columns, title='k2 vs parameters of interest reduced ppt',
    #                     save_dir=save_dir)

    #simple rep
    # plot_histogram(data.loc[:,x_col],x_col)
    # plot_histogram(Y,out_cols)

    #

    #dimension reduction and higher dimension rep
    scaler = RobustScaler()
    Y = scaler.fit_transform(Y)
    Y = pd.DataFrame(Y,columns=y_columns)
    method_tracker = {}
    for method in methods:
        two_d_rep = data_reduction_method(X, method)
        method_tracker[method] = two_d_rep
    for y_col in y_columns:
        y = Y[y_col]

        plot_comparison_of_reductions(method_tracker,y,y_col,save_dir=save_dir)

        # plot_3D_rep_data(X[:,:3],y,in_cols[:3],y_col)

if __name__ == '__main__':
    main()