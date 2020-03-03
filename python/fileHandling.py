import h5py
import os
import numpy as np
import hdf5storage
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


def delete_bad_data(folder):
    for filename in tqdm(os.listdir(folder)):
        if filename.endswith('.mat'):
            data = load_hdf5(folder + filename)
            for key in data.keys():
                if np.isnan(np.mean(data[key])):
                    print(key, filename)
                    os.remove(folder+filename)
                    break


if __name__ == "__main__":
    # delete_bad_data('data/validation/')
    data = load_hdf5('data/train/2018_01_01.mat')
    print(data.keys())
    print(data['x_wind_ml'].shape)
