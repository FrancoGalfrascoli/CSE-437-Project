
import numpy as np
from scipy.io import loadmat
import torch

__all__ = [
    "load_tok_data_struct",
    "load_norm_and_datasets",
    "load_flux_mats"
]

def load_tok_data_struct(mat_path):
    data = loadmat(mat_path)
    return data["tok_data_struct"]

def load_norm_and_datasets(base="Data_generation"):
    train_data = np.load(f"{base}/data_train.npz")
    test_data  = np.load(f"{base}/data_test.npz")
    val_data   = np.load(f"{base}/data_validation.npz")
    norm_values = np.load(f"{base}/norm_values.npz")
    return train_data, test_data, val_data, norm_values

def load_flux_mats(base="Data_generation"):
    fluxPF_Bnd = np.load(f"{base}/fluxPF_Bnd.npy")
    A_mat = np.load(f"{base}/A_matrix.npy")
    return fluxPF_Bnd, A_mat
