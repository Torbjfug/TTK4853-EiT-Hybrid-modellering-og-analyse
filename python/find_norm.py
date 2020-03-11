import os
import numpy as np
import hdf5storage
import pickle
import h5py
from tqdm import tqdm
from fileHandling import *
import pandas as pd


def calculate_means(data):
    means = {}
    for key in data.keys():
        means[key] = np.nanmean(data[key])
    return means


def calculate_std(data):
    stds = {}
    for key in data.keys():
        stds[key] = np.nanstd(data[key])
    return stds


def normalize_dataset(data, means, std):
    norm_data = {}
    for key in data.keys():
        val = data[key]
        val[np.isnan(val)] = 0
        norm_data[key] = (val -
                          means[key].item()) / std[key].item()
    return norm_data


def compute_individual_means(path):
    """
    For debuging of means calculations
    """

    for filename in tqdm(os.listdir(path)):
        if filename.endswith('.mat'):
            data = hdf5storage.loadmat(path + filename)
            means = calculate_means(data)
            stds = calculate_std(data)
            # if any(np.isnan(list(stds.values()))) or \
            #     any(np.isnan(list(means.   values()))) or \
            #         np.mean(list(means.values())) > 1e6 or \
            #         np.mean(list(stds.values())) > 1e6:
            #     os.remove(path+filename)


def normalize_datasets(original_folder, dest_folder, means, stds):
    for filename in tqdm(os.listdir(original_folder)):
        if filename.endswith('.mat'):
            data = hdf5storage.loadmat(original_folder + filename)
            data = normalize_dataset(data, means, stds)
            save_hdf5(data, dest_folder+filename)



    


if __name__ == "__main__":

    # load_data = False
    # if load_data:
    #     data = load_hdf5('all_files.hdf5')
    #     means, stds = load_hdf5('means.hdf5'), load_hdf5('stds.hdf5')

    # else:
    #     input("Press enter to overwrite file")
    #     filename = 'all_files.hdf5'
    #     if filename in os.listdir('./'):
    #         os.remove(filename)
    #     data = concatenate_files_in_folder('data/calibration/', filename)
    #     means, stds = calculate_means(data), calculate_std(data)
    #     save_hdf5(means, 'means.hdf5')
    #     save_hdf5(stds, 'stds.hdf5')
    #     print(means)
    #     print(stds)

    # test_file = 'data/calibration/2017_11_05.mat'
    # test_data = load_hdf5(test_file)
    # test_data_norm = normalize_dataset(test_data, means, stds)
    # test_mean = calculate_means(test_data_norm)
    # test_std = calculate_std(test_data_norm)
    # print(test_mean)
    # print(test_std)
    #means, stds = load_hdf5('means.hdf5'), load_hdf5('stds.hdf5')
    #data = load_hdf5("data/calibration/2017_12_05.mat")

    #normalize_datasets('data/calibration/', 'data/normalized/', means, stds)
