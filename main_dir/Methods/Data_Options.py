#my data options module

import os
import copy
# import torch

import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from tkinter.filedialog import askdirectory, askopenfilename
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

class data_options:

    def get_iteration(self,path,file_prefix=None):

        files = [f for f in listdir(path) if isfile(join(path, f))]
        iterations = []
        if file_prefix is not None:
            files = [file for file in files if file_prefix in file]

        if not files:
            return 1

        for file in files:
            file = file.split(".", 1)[0]
            try:
                iteration = int(file.split("_")[-1])
                iterations.append(iteration)
            except ValueError:
                continue
        iterations.sort(reverse=True)
        return iterations[0] + 1

    def __init__(self,use_default_folder="y",type="VR_DATA",default_data=False,computer="laptop"):
        self.default_data=default_data
        if computer == "laptop":
            if type == "VR_DATA":
                if use_default_folder == "y":
                    self.save_dir = "C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/Data/"
                else:
                    self.save_dir = None
            elif type == "BAU":
                if use_default_folder == "y":
                    self.save_dir = "C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/ExoMDN-main/ExoMDN-main/data/training_demo/"
                else:
                    self.save_dir = None
        elif computer == "main_pc":
            if type == "VR_DATA":
                if use_default_folder == "y":
                    self.save_dir = ""
                else:
                    self.save_dir = None
            elif type == "BAU":
                if use_default_folder == "y":
                    self.save_dir = ""
                else:
                    self.save_dir = None
        else:
            self.save_dir = None

    def get_save_dir(self):
        if self.save_dir is None:
            self.save_dir = askdirectory()

    def get_data(self):
        files_list=[]
        if self.default_data == True:
            for files in os.listdir(self.save_dir):
                if files == "default.dat":
                    files_list.append(files)
            if len(files_list) > 1:
                raise ValueError("Too many default files, returning none")
                return None
            file = files_list[0]
            path = os.path.join(self.save_dir, file)
            data = pd.read_csv(path, delimiter="\t")
            return data
        for files in os.listdir(self.save_dir):
            if (files.endswith(".csv") or files.endswith(".dat")) and not (files.endswith("copy.csv") or files.endswith("copy.dat")):
                files_list.append(files)
        if len(files_list) == 1:
            file = files_list[0]
            path = os.path.join(self.save_dir, file)
        else:
            file = askopenfilename(initialdir=self.save_dir,
                                   filetypes=[("Data files", "*.dat *.csv"), ("DAT files", "*.dat"), ("CSV files", "*.csv")])
            path = file
        if file.endswith(".csv"):
            data = pd.read_csv(path)
        elif file.endswith(".dat"):
            data = pd.read_csv(path, delimiter="\t")
        return data

    def partition_data(self,data,train_frac=0.6,test_frac=0.4,valid_frac=None,seed=42):
        total = train_frac + test_frac + valid_frac
        if not np.isclose(total, 1.0):
            raise ValueError(f"Fractions must sum to 1. Got {total}")

        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(data)

        n_train = int(train_frac * n)
        n_test = int(test_frac * n)

        train = data.iloc[:n_train]
        test = data.iloc[n_train:n_train + n_test]
        valid = data.iloc[n_train + n_test:]

        return train, test, valid


    def get_xy(self, dataframe, y_label, x_labels=None):
        dataframe = copy.deepcopy(dataframe)  # Avoid modifying the original DF

        if x_labels is None:
            x = dataframe.drop(columns=[y_label])  # Keep original shape
        else:
            x = dataframe[x_labels]  # Keep original shape

        y = dataframe[y_label]  # Ensure y is 2D

        return  x, y
