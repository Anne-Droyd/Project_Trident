"""
Rebuilding from scratch to make sure everything makes sense
"""
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from main_dir.Methods import Plotting as plots
from keras import callbacks, Sequential, regularizers
from main_dir.Methods.Data_Options import data_options
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from main_dir.Methods.EXOMDN_MDN_layer import get_mixture_loss_func as MDNLoss
# from main_dir.Methods.EXOMDN_MDN_layer import get_winner_takes_all_loss as Winner_Loss
# from main_dir.Methods.EXOMDN_MDN_layer import MDN, softmax, elu_plus_one_plus_epsilon, get_mixture_loss_func



def unpack_mdn_predictions(predictions, output_dim, num_mixtures):
    out_mu, out_sigma, out_pi = np.split(predictions,indices_or_sections=[
                                        num_mixtures * output_dim,
                                        2 * num_mixtures * output_dim],axis=-1)

    # Reshape:
    mus = np.reshape(out_mu, (-1, num_mixtures, output_dim))        # (samples, mixtures, outputs)
    sigmas = np.reshape(out_sigma, (-1, num_mixtures, output_dim))  # (samples, mixtures, outputs)
    pi = np.reshape(out_pi, (-1, num_mixtures))                      # (samples, mixtures)

    return mus, sigmas, pi

def sample_from_mixture(mu, sigma, pi, num_samples=1):
    samples = []
    for i in range(mu.shape[0]):  # for each test point
        component = np.random.choice(mu.shape[1], p=pi[i])
        sample = np.random.normal(loc=mu[i, component], scale=sigma[i, component])
        samples.append(sample)
    return np.array(samples)

def main():
    data_ops = data_options(type="VR_DATA")
    data = data_ops.get_data()
    train, test, valid = data_ops.partition_data(data,train_frac=0.8,test_frac=0.1,valid_frac=0.1)

    data_ops = data_options(type="BAU")
    bau=data_ops.get_data()

    df_a = data.rename(columns={"req":"radius","Teq":"temp"})
    df_b = bau.rename(columns={'planet_mass': 'mass', 'planet_radius': 'radius', 'T_eq': 'temp'})

    # expansion_factor = 0.9
    # train = train.sample(n=(int(expansion_factor * len(train))), replace=True, random_state=42)

    #Index(['planet_mass', 'planet_radius', 'T_eq', 'log_d_mantle_core',
      #  'log_m_mantle_core', 'log_d_atmosphere_core', 'log_m_atmosphere_core',
      #  'log_d_water_core', 'log_m_water_core'],
      # dtype='object')

    # Index(['m_core', 'zatm', 'zatm0', 'zatm1', 'zdeep', 'zdeep0', 'zdeep1',
    #        'p_ppt', 'req', 'mass', 'lum', 'Teq', 'p_rot', 'k2',"test","density"],

    #BAU data
        #MKVIII relu, scaled
        #MKVII gelu instead of relu
        #MKVI relu
    #VR data
        # MK_I 120 mix 50 batch 60 nodes 4 layers relu6 random normal bias init no kern/bias/act reg
            # loss 3.81 not great flat predictions on multiple
        # MK_II act reg 1e-2
            # loss 5.19 not good all predictions flat
        # MK_III act reg 1e-7
            # loss 3.82 same as without
        # MK_IV act reg 1e-4
            # loss 4.12 mid
        # MK_V bias reg 1e-2
            # loss 3.83
        # MK_VI bias reg 1e-7
            # loss 3.79
        # MK_VII bias reg 1e-4
            # loss 3.81
        # MK_VIII kernel reg 1e-2
            # loss
        # MK_IX kernel reg 1e-7
            # loss
        # MK_X kernel reg 1e-4
            # loss
        # MK_XI
        # MK_XII
        # MK_XIII
        # MK_XIV
        # MK_XVI
        #   mcore, zatm, zdeep, no lum
        #   5 mix, 60 nodes
        #biggest_MKXII
            #1.7 loss
            # 120 mix, 60 nodes
            #     y_col = ["m_core",'zatm0', 'zatm1', 'zdeep0', 'zdeep1']
            #     x_col = ["mass","req","Teq","lum"]
    model_folder_path = "C:/Users/Matth/Documents/Leiden University/Project/models/MDN/"
    history_path = "C:/Users/Matth\Documents/Leiden University/Project/Histories/MDN/"

    # expansion_factor = 0.9
    # train = train.sample(n=(int(expansion_factor * len(train))), replace=False)

    iteration = "230"
    Model_save_name = f"MDN_model_{iteration}.h5"
    history_name = f"history_MK_{iteration}.csv"
    train_new_model = "n"

    epochs = 500

    lr = 0.001

    num_mixtures_1 = 120
    batch_size_1 = 1000
    num_hidden_nodes_1 = 60

    num_mixtures_2 = 50
    batch_size_2 = 1000
    num_hidden_nodes_2 = 64

    num_hidden_nodes_3 = 32

    #model params
    bias_init="zeros"
    activation="relu"
    kernel_reg = None
    bias_reg = None
    act_reg = None
    # kernel_reg = regularizers.L1L2(l1=2e-2, l2=1e-2)
    L1=[0,1e-6,1e-5,1e-4,1e-3,1e-2]
    L2=[0,1e-6,1e-5,1e-4,1e-3,1e-2]
    # bias_reg=regularizers.L2(1e-2)
    # act_reg=regularizers.L2(1e-2)
    # bias_reg=[None,regularizers.L1(l1=0.01),regularizers.L2(1e-7),regularizers.L1L2(l1=0.0, l2=0.0)]
    # act_reg=[None,regularizers.L1(l1=0.01),regularizers.L2(1e-7),regularizers.L1L2(l1=0.0, l2=0.0)]

    # y_col = ["log_d_mantle_core","log_d_atmosphere_core","log_d_water_core","log_m_mantle_core","log_m_atmosphere_core","log_m_water_core"]
    # x_col = ["planet_mass","planet_radius","T_eq"]

    y_col = ["m_core", 'zatm', 'zdeep']
    x_col = ["mass","req","Teq","k2"]

    train_x, train_y = data_ops.get_xy(train, y_col, x_col)
    test_x, test_y = data_ops.get_xy(test, y_col, x_col)
    valid_x, valid_y = data_ops.get_xy(valid, y_col, x_col)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_x)
    train_y = y_scaler.fit_transform(train_y)
    # Do not refit the transform
    valid_x = x_scaler.transform(valid_x)
    valid_y = y_scaler.transform(valid_y)

    # test_x = pd.DataFrame([[3.2,1.2,876]],columns=x_col)
    # test_y = pd.DataFrame([[-7.22,-7.6,-2.11,-15.27,-2.73,-4.76]],columns=y_col)
    test_x = x_scaler.transform(test_x)

    input_dim = train_x.shape[1]
    output_dim = train_y.shape[1]
    optimizer = Adam(learning_rate=lr)

    #architecture 4 layers
    if train_new_model == "yes" or train_new_model == "y":
        model = Sequential()
        model.add(Dense(num_hidden_nodes_1,input_shape=(input_dim,),activation=activation,
                        bias_initializer=bias_init, kernel_regularizer=kernel_reg,
                        bias_regularizer=bias_reg, activity_regularizer=act_reg))
        model.add(Dense(num_hidden_nodes_1,activation=activation,
                        bias_initializer=bias_init, kernel_regularizer=kernel_reg,
                        bias_regularizer=bias_reg, activity_regularizer=act_reg))
        model.add(Dense(num_hidden_nodes_1,activation=activation,
                        bias_initializer=bias_init, kernel_regularizer=kernel_reg,
                        bias_regularizer=bias_reg, activity_regularizer=act_reg))
        model.add(Dense(num_hidden_nodes_1,activation=activation,
                        bias_initializer=bias_init, kernel_regularizer=kernel_reg,
                        bias_regularizer=bias_reg, activity_regularizer=act_reg))
        model.add(MDN(output_dim,num_mixtures_1))
        model.compile(loss=MDNLoss(output_dim,num_mixtures_1), optimizer=optimizer)

        history = model.fit(train_x,train_y,epochs=epochs,batch_size=batch_size_1, validation_data = (valid_x,valid_y),
                callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss",patience=4)
                ])
        model.save(model_folder_path+Model_save_name, include_optimizer=False)

        history_df = pd.DataFrame(history.history)

        # Save to CSV
        history_df.to_csv(history_path + history_name, index=False)

    else:
        model = keras.models.load_model(
            model_folder_path + Model_save_name,
            custom_objects={"MDN":MDN}
        )
        model.compile(loss=MDNLoss(output_dim, num_mixtures_1), optimizer=optimizer)



    predictions = model.predict(test_x)
    mu, sigma, pi = unpack_mdn_predictions(predictions, output_dim, num_mixtures_1)

    mu_flat = mu.reshape(-1, output_dim)
    mu = y_scaler.inverse_transform(mu_flat)
    mu = mu.reshape(-1, num_mixtures_1, output_dim)

    scale = y_scaler.scale_
    sigma = sigma * scale.reshape(1, 1, -1)

    # mu = np.clip(mu, 0, 1)
    # mu[:, :, 0] = np.clip(mu[:, :, 0], 0, 0.5)

    # scale = y_scaler.data_max_ - y_scaler.data_min_
    # sigma = sigma * scale.reshape(1, 1, -1)

    pi = np.apply_along_axis(softmax, 1, pi, temperature=1)

    # mask = (pi >= 0.01) & np.all(sigma <= 0.5, axis=2) & np.all(sigma >= 0.005, axis=2)  # shape (batch_size, num_mixes)

    # Apply the mask
    # We keep the same shape but set invalid components' pi to zero
    # pi = np.where(mask, pi, 0.0)

    # # Optional: renormalize pi so it sums to 1 for each sample
    # pi_sum = np.sum(pi, axis=1, keepdims=True)
    # pi_sum = np.where(pi_sum == 0, 1.0, pi_sum)  # avoid divide-by-zero
    # pi /= pi_sum

    means = np.sum(pi[..., np.newaxis] * mu, axis=1)
    # map_indices = np.argmax(pi, axis=1)
    map_indices = np.argsort(pi, axis=1)[:,-1]
    #
    top5_indices = np.argsort(pi, axis=1)[:, -5:]  # last 5 = top 5 since argsort is ascending
    # sigma_reduced = sigma.mean(axis=2)  # shape (batch_size, num_mixtures)

    # Get index of mixture with lowest sigma for each sample
    # lowest_sig_idx = np.argmin(sigma_reduced, axis=1)
    # best_mean = mu[np.arange(mu.shape[0]), lowest_sig_idx, :]

    # Step 2: Gather μ for those components
    # This indexing will be a bit tricky with numpy
    batch_indices = np.arange(mu.shape[0])[:, None]  # shape (n_samples, 1)
    top5_mus = mu[batch_indices, top5_indices]  # shape (n_samples, 5, output_dim)

    # Step 3: Average them
    mu_top5_avg = np.mean(top5_mus, axis=1)
    top5_pis = pi[batch_indices, top5_indices]  # shape (n_samples, 5)
    weighted_avg = np.sum(top5_mus * top5_pis[..., None], axis=1) / np.sum(top5_pis, axis=1, keepdims=True)

    mu_map = np.array([mu[i, idx, :] for i, idx in enumerate(map_indices)])
    mu_mixture = np.sum(pi[:, :, np.newaxis] * mu, axis=1)  # Shape: (17506, 5)

    # Broadcasting mu_mixture to (17506, 120, 5) for element-wise subtraction with mu
    mu_mixture_broadcasted = mu_mixture[:, np.newaxis, :]  # Shape: (17506, 1, 5)

    # Now calculate variance
    var_mixture = np.sum(pi[:, :, np.newaxis] * (sigma ** 2 + (mu - mu_mixture_broadcasted) ** 2), axis=1)

    plots.plot_single_pdf(mu, sigma, pi, y_col,
                          test_y= test_y,
                          mean_pred= means,
                          means= mu_mixture,
                          mu_map=mu_top5_avg,
                          vars= var_mixture,
                          top5_mus=top5_mus,
                          idx= np.random.randint(0, mu.shape[0]))

    # plt.scatter(range(num_mixtures_1),pi[np.random.randint(0, mu.shape[0]),:])
    # plt.ylabel()
    # plt.show()

    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, means, mu_mixture, var_mixture, idx=0)
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=1)
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=2)
    #
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=3)
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=4)
    # plots.plot_single_pdf(mu, sigma, pi, y_col, test_y, mu_mixture, var_mixture, idx=5)

    # plots.plot_mu_vs_real_with_errorbars(mu_mixture, var_mixture, test_y, y_col)
    # samples = plots.sample_from_mixture(mu,sigma, pi,100)
    # plots.plot_predicted_vs_real_hist(samples, test_y,y_col)

    # plots.plot_predicted_vs_real_scatter(best_mean,test_y)
    # plots.plot_predicted_vs_real_scatter(samples,test_y)

main()