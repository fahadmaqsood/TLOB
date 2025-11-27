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
    dec_data = np.loadtxt("data/DATABENTO/databento_fi2010_2025-09-15_2025-09-15.txt")

    full_train = dec_data[:, :int(dec_data.shape[1] * cst.SPLIT_RATES[0])]
    full_val = dec_data[:, int(dec_data.shape[1] * cst.SPLIT_RATES[0]):]
    full_test = full_val

    horizon = 10
    seq_size = 128
    all_features = True

    if horizon == 10:
        tmp = 5
    elif horizon == 20:
        tmp = 4
    elif horizon == 30:
        tmp = 3
    elif horizon == 50:
        tmp = 2
    elif horizon == 100:
        tmp = 1
    else:
        raise ValueError("Horizon not found")

    train_labels = full_train[-tmp, :].flatten()
    val_labels = full_val[-tmp, :].flatten()
    test_labels = full_test[-tmp, :].flatten()

    print(np.unique(train_labels))

    train_labels = train_labels[seq_size-1:] - 1
    val_labels = val_labels[seq_size-1:] - 1
    test_labels = test_labels[seq_size-1:] - 1
    if all_features:
        train_input = full_train[:144, :].T
        val_input = full_val[:144, :].T
        test_input = full_test[:144, :].T
    else:
        train_input = full_train[:40, :].T
        val_input = full_val[:40, :].T
        test_input = full_test[:40, :].T
    train_input = torch.from_numpy(train_input).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_input = torch.from_numpy(val_input).float()
    val_labels = torch.from_numpy(val_labels).long()
    test_input = torch.from_numpy(test_input).float()
    test_labels = torch.from_numpy(test_labels).long()

    print("Unique labels in train:", np.unique(train_labels))
    print("Unique labels in val:", np.unique(val_labels))
    print("Unique labels in test:", np.unique(test_labels))

    return train_input.shape, train_labels.shape, val_input.shape, val_labels.shape, test_input.shape, test_labels.shape
    
