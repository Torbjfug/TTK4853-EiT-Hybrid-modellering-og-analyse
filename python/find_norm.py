import os
import numpy as np
import hdf5storage
import pickle
import h5py


def save_hdf5(data,filename):
    with h5py.File(filename, 'w') as data_file:
        for key in data.keys():
            data_file.create_dataset(key, data=data[key])


def load_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        data = {}
        for key in list(f.keys()):
            data[key] = np.array(f[key])
    return data


def load_data(year,month,day,filepath):
    filename = f"{year}_{month:02d}_{day:02d}.mat"
    mat = hdf5storage.loadmat(filepath + filename)
    return mat

def concatinate_files(data1, data2):
    for key in data2.keys():
        data1[key] = np.concatenate((data1[key], data2[key]))
    return data1

def concatenate_htf5(filename, data):
    with h5py.File(filename, 'a') as f:
        dataset_keys = list(f.keys())[0]
        for key in dataset_keys:
            dset = f[keys]
            dset.resize(dset.shape[0],data[key].shape[0],axis=0)
            f[key] = np.array(f[key])
    return data
def concatenate_files_in_folder(directory):
    with h5py.File(filename, 'w') as data_file:

    all_data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            print(filename)

            try:
                data = hdf5storage.loadmat(directory + filename)
                if not bool(all_data):
                    all_data = data
                else:
                    all_data = concatinate_files(all_data, data)
            except Exception as e:
                print(e)
                pass
    return all_data


def calculate_get_means(data):
    means = {}
    for key in data.keys():
        means[key] = np.mean(data[key])
    return means

def calculate_std(data):
    stds = {}
    for key in data.keys():
        stds[key] = np.std(data[key])
    return stds

def normalize_dataset(data, means, std):
    norm_data = {}
    for key in data.keys():
        norm_data[key] = (data[key] - means[key]) / std[key]
    return norm_data

def comute_normalization_parameters(sorce_file_dir,save_path='./'):
    data = concatenate_files_in_folder(sorce_file_dir)
    save_hdf5(data, save_path+'all_data.hdf5')
    means = calculate_get_means(data)
    stds = calculate_std(data)
    save_hdf5(means, save_path+'means.hdf5')
    save_hdf5(stds,save_path+'stds.hdf5')

load_data = False
if load_data:
    data = load_hdf5('all_data.hdf5')
    
else:
    data = concatenate_files_in_folder('data/calibration/')
    save_hdf5(data, 'all_data.hdf5')
    



means = calculate_get_means(data)
stds = calculate_std(data)
save_hdf5(means, 'means.hdf5')
save_hdf5(stds,'stds.hdf5')

for key in means.keys():

    print(key)
    print("Mean: ", means[key])
    print("Standard_dev", stds[key])
    print(data[key].shape)
    print("")