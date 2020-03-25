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


def update_dataset(filename, prev_v, next_v):
    if prev_v == 1 and next_v == 2:
        norm_metrics = {}
        norm_metrics['std'] = np.nanstd
        norm_metrics['mean'] = np.nanmean
        norm_metrics['min'] = np.nanmax
        norm_metrics['max'] = np.nanmin

        with h5py.File(filename, 'a') as f:
            for key in f.keys():
                data = f[key]
                group = f.create_group(key+'_metrics')
                #group.create_dataset('data', data=h5py.SoftLink(data))
                for metric in norm_metrics.keys():
                    print(norm_metrics[metric](data.value))
                    group.create_dataset(
                        metric, data=norm_metrics[metric](data.value))
            f.create_dataset('version', data=2)


def delete_old_files(path):
    for filename in tqdm(os.listdir(path)):
        if filename.endswith('.mat'):
            data = hdf5storage.loadmat(path + filename)
            if data['x_wind_ml'].shape[:-1] != (132, 132, 32):
                os.remove(path+filename)


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
    data = load_hdf5('..\TTK4853-EiT-Hybrid-modellering-og-analyse\SR\\2019_08_05.mat')
    print(data.keys())
    print(data['geopotential_height_ml'].shape)
