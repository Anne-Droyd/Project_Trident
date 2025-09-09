#my data options module

import os
import copy
# import torch

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tkinter.filedialog import askdirectory, askopenfile
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

class data_options:

    def __init__(self,use_default_folder="y",type="VR_DATA",computer="laptop"):
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
        for files in os.listdir(self.save_dir):
            if (files.endswith(".csv") or files.endswith(".dat")) and not (files.endswith("copy.csv") or files.endswith("copy.dat")):
                files_list.append(files)
        if len(files_list) == 1:
            file = files_list[0]
            if file.endswith(".csv"):
                type = "csv"
            elif file.endswith(".dat"):
                type = "dat"
        else:
            file = askopenfile(mode="r",filetypes=[("*.dat","*.csv")])
        path = self.save_dir+file
        if type == "csv":
            data = pd.read_csv(path)
        elif type == "dat":
            data = pd.read_csv(path, delimiter="\t")
        return data

    def partition_data(self,data,train_frac=0.6,test_frac=0.2,valid_frac=0.2,seed=42):
        train, test, valid = np.split(data.sample(frac=1,random_state=seed),[int(train_frac*len(data)),int((1-test_frac)*len(data))])

        return train, test, valid


    def get_xy(self, dataframe, y_label, x_labels=None):
        dataframe = copy.deepcopy(dataframe)  # Avoid modifying the original DF

        if x_labels is None:
            x = dataframe.drop(columns=[y_label])  # Keep original shape
        else:
            x = dataframe[x_labels]  # Keep original shape

        y = dataframe[y_label]  # Ensure y is 2D

        return  x, y
