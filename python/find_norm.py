import os
import numpy as np
import hdf5storage
import pickle
import h5py
from tqdm import tqdm


def save_hdf5(data,filename):
    with h5py.File(filename, 'w') as data_file:
        for key in data.keys():
            data_file.create_dataset(key, data=data[key],)


def load_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        data = {}
        for key in list(f.keys()):
            print(key)
            data[key] = np.array(f[key])
    return data


def load_data(year,month,day,filepath):
    filename = f"{year}_{month:02d}_{day:02d}.mat"
    return hdf5storage.loadmat(filepath + filename)

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
                f.create_dataset(key,data=data[key],maxshape=max_size)
    return data

def concatenate_files_in_folder(source_dir,save_file):
    for filename in tqdm(os.listdir(source_dir)):
        if filename.endswith('.mat'):
            #print(filename)
            data = hdf5storage.loadmat(source_dir + filename)
            concatenate_hdf5(save_file, data)
            
    all_data = load_hdf5(save_file)
    return all_data


def calculate_get_means(data):
    Print("Calculate mean")
    means = {}
    for key in data.keys():
        print(key)
        means[key] = np.mean(data[key])
    return means

def calculate_std(data):
    print("Calculate STD")
    stds = {}
    for key in data.keys():
        print(key)
        stds[key] = np.std(data[key])
    return stds

def normalize_dataset(data, means, std):
    Print("Normalizing")
    norm_data = {}
    for key in data.keys():
        print(key)
        norm_data[key] = (data[key] - means[key]) / std[key]
    return norm_data

def comute_normalization_parameters(data):
    means = calculate_get_means(data)
    stds = calculate_std(data)  
    return means, stds


load_data = True
if load_data:
    data = load_hdf5('all_files.hdf5')
    
else:
    filename = 'all_files.hdf5'
    if filename in os.listdir('./'):
        os.remove(filename)
    data = concatenate_files_in_folder('data/calibration/',filename)
means,std = comute_normalization_parameters(data)
save_hdf5(means, 'means.hdf5')
save_hdf5(std, 'stds.hdf5')    

