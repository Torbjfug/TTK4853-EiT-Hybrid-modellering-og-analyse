import os
import numpy as np
import hdf5storage
import pickle
import h5py
from tqdm import tqdm


def save_hdf5(data, filename):
    with h5py.File(filename, 'w') as data_file:
        for key in data.keys():
            data_file.create_dataset(key, data=data[key],)


def load_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        data = {}
        for key in list(f.keys()):
            data[key] = np.array(f[key])
    return data


def concatinate_files(data1, data2):
    for key in data2.keys():
        data1[key] = np.concatenate((data1[key], data2[key]))
    return data1


def concatenate_hdf5(filename, data):
    with h5py.File(filename, 'a') as f:
        dataset_keys = list(f.keys())
        for key in data.keys():
            if key in dataset_keys:
                dset = f[key]
                dset.resize(dset.shape[0]+data[key].shape[0], axis=0)
                dset[-data[key].shape[0]:] = data[key]
            else:
                max_size = (None,)+data[key].shape[1:]
                f.create_dataset(key, data=data[key], maxshape=max_size)
    return data


def concatenate_files_in_folder(source_dir, save_file):
    for filename in tqdm(os.listdir(source_dir)):
        if filename.endswith('.mat'):
            data = hdf5storage.loadmat(source_dir + filename)
            concatenate_hdf5(save_file, data)
    all_data = load_hdf5(save_file)
    return all_data


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
        norm_data[key] = (data[key] - means[key]) / std[key]
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


if __name__ == "__main__":
    load_data = False
    if load_data:
        #data = load_hdf5('all_files.hdf5')
        means, stds = load_hdf5('means.hdf5'), load_hdf5('stds.hdf5')

    else:
        input("Press enter to overwrite file")
        filename = 'all_files.hdf5'
        if filename in os.listdir('./'):
            os.remove(filename)
        data = concatenate_files_in_folder('data/calibration/', filename)
        means, stds = calculate_means(data), calculate_std(data)
        save_hdf5(means, 'means.hdf5')
        save_hdf5(stds, 'stds.hdf5')

    test_file = 'data/calibration/2017_11_05.mat'
    test_data = load_hdf5(test_file)
    test_data_norm = normalize_dataset(test_data, means, stds)
    test_mean = calculate_means(test_data_norm)
    test_std = calculate_std(test_data_norm)
    print(test_mean)
    print(test_std)
