#my data options module

import os
import copy
import torch

import pandas as pd
import numpy as np

from tkinter.filedialog import askdirectory, askopenfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class data_options:

    def __init__(self,user="y"):
        if user == "y":
            self.save_dir = "C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/Data/"
        else:
            self.save_dir = None

    def check_if_exists(self, path):
        return os.path.isfile(path)

    def get_save_folder(self):
        if self.save_dir:
            return self.save_dir
        else:
            self.save_dir = askdirectory(title="Choose Saving Directory")
            return self.save_dir

    def get_data(self):
        if self.save_dir is None:
            self.save_dir = self.get_save_folder()
        saved_file = str(self.save_dir + "/initial_data.pk")
        if self.check_if_exists(saved_file) == True:
            # user = input(f"Do you want to use {saved_file}? :")
            user = "y"
            if user.lower() == "yes" or user.lower() == "y":
                df = pd.read_pickle(saved_file)
                return df
            elif user.lower() == "no" or user.lower() == "n":
                print("Answered no, so a new file will be made")
                path = askopenfile(title="Select File")
                df = pd.read_csv(path, delimiter="\t")
                data_name = "/initial_data.pk"
                with open(str(self.save_dir + data_name), "wb") as file:
                    df.to_pickle(file)
                file.close()
                return df
            else:
                print(f"What do you want from me? Your answer: {user}")
                return None
        else:
            path = askopenfile(title="Select File")
            df = pd.read_csv(path, delimiter="\t")
            data_name = "/initial_data.pk"
            with open(str(self.save_dir + data_name), "wb") as file:
                df.to_pickle(file)
            file.close()
            return df

    def partition_data(self, data):
        if self.save_dir is None:
            self.save_dir = self.get_save_folder()
        # Define these variables before using them
        train_name = "training_data.pk"
        test_name = "testing_data.pk"
        valid_name = "validation_data.pk"

        train, valid, test = np.split(data.sample(frac=1), [int(0.6 * len(data)), int(0.8 * len(data))])
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)
        valid = pd.DataFrame(valid)

        with open(str(self.save_dir + "/" + train_name), "wb") as file:
            train.to_pickle(file)
        file.close()
        with open(str(self.save_dir + "/" + test_name), "wb") as file:
            test.to_pickle(file)
        file.close()
        with open(str(self.save_dir + "/" + valid_name), "wb") as file:
            valid.to_pickle(file)
        file.close()
        return train, test, valid

    def get_partitioned_data(self, data):
        train_name = "training_data.pk"
        test_name = "testing_data.pk"
        valid_name = "validation_data.pk"
        if self.check_if_exists(str(self.save_dir + "/" + train_name)) and \
                self.check_if_exists(str(self.save_dir + "/" + test_name)) and \
                self.check_if_exists(str(self.save_dir + "/" + valid_name)):
            # user = input("Partitioned data found, would you like to use it?: ")
            user = "y"
            if user.lower() == "yes" or user.lower() == "y":
                train = pd.read_pickle(str(self.save_dir + "/" + train_name))
                test = pd.read_pickle(str(self.save_dir + "/" + test_name))
                valid = pd.read_pickle(str(self.save_dir + "/" + valid_name))
                return train, test, valid
            elif user.lower() == "info":
                print(f"Training data: {str(self.save_dir + "/" + train_name)}")
                print(f"Testing data: {str(self.save_dir + "/" + test_name)}")
                print(f"Validation data: {str(self.save_dir + "/" + valid_name)}")
            else:
                train, test, valid = self.partition_data(data)
                return train, test, valid
        else:
            train, test, valid = self.partition_data(data)
            return train, test, valid

    def get_xy(self, dataframe, y_label, x_labels=None):
        dataframe = copy.deepcopy(dataframe)  # Avoid modifying the original DF

        if x_labels is None:
            X = dataframe.drop(columns=[y_label]).values  # Keep original shape
        else:
            X = dataframe[x_labels].values  # Keep original shape

        y = dataframe[y_label].values  # Ensure y is 2D

        return  X, y

    def scale_data(self,train_x,train_y,test_x,test_y):
        x_scaler = StandardScaler()
        y_scaler = MinMaxScaler(feature_range=(1e-8, 1))

        # Fit scalers on training data
        train_x_scaled = x_scaler.fit_transform(train_x)
        train_y_scaled = y_scaler.fit_transform(train_y)

        # Apply scaling to test data
        test_x_scaled = x_scaler.transform(test_x)
        test_y_scaled = y_scaler.transform(test_y)

        return train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, x_scaler, y_scaler

    def scale_single(self, data, scaler=None):
        if scaler == None:
            scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data, scaler
