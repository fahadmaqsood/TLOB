import numpy as np
import constants as cst
import os
from torch.utils import data
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

  
def databento_fi_2010_load(path, seq_size, horizon, all_features):
    # dec_data = np.loadtxt(path + "/Train_Dst_NoAuction_ZScore_CF_7.txt")
    # full_train = dec_data[:, :int(dec_data.shape[1] * cst.SPLIT_RATES[0])]
    # full_val = dec_data[:, int(dec_data.shape[1] * cst.SPLIT_RATES[0]):]
    # dec_test1 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_7.txt')
    # dec_test2 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_8.txt')
    # dec_test3 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_9.txt')
    # full_test = np.hstack((dec_test1, dec_test2, dec_test3))

    # Load dataset using pandas
    df = pd.read_csv(path + "/databento_fi2010_2025-09-15_2025-09-15.txt", 
                    sep="\t")

    # Separate features and labels
    data = df.iloc[:, :-1].values       # time × features
    labels = df.iloc[:, -1].values.astype(int) - 1

    # Stratified split: train 80%, val+test 20%
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Split remaining 20% into val and test (10% each)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Transpose back to original FI-2010 style (features × time)
    full_train = X_train
    full_val   = X_val
    full_test  = X_test

    train_input = torch.from_numpy(X_train).float()
    train_labels = torch.from_numpy(y_train).long()
    val_input = torch.from_numpy(full_val).float()
    val_labels = torch.from_numpy(y_val).long()
    test_input = torch.from_numpy(full_test).float()
    test_labels = torch.from_numpy(y_test).long()

    print("Unique labels in train:", np.unique(train_labels))
    print("Unique labels in val:", np.unique(val_labels))
    print("Unique labels in test:", np.unique(test_labels))

    return train_input, train_labels, val_input, val_labels, test_input, test_labels
    
    
