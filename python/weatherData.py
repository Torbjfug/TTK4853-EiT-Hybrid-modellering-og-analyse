from torch.utils.data import Dataset
import os
from fileHandling import load_hdf5, save_hdf5
import h5py
import numpy as np
from torch.utils.data import DataLoader
import torch


class weatherDataSet(Dataset):
    def __init__(self, x_size, y_size, z_size, folder):
        assert (x_size) % 2 == 0
        assert (y_size) % 2 == 0
        assert (z_size) % 2 == 0
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.folder = folder
        self.filenames = [f for f in os.listdir(folder)
                          if f.endswith('.mat') and os.path.isfile(os.path.join(folder, f))]
        self.x_quadrants = 128//x_size
        self.y_quadrants = 128//y_size
        self.z_quadrants = 32//z_size
        self.qubes = self.x_quadrants*self.y_quadrants*self.z_quadrants

    def __len__(self):
        return len(self.filenames)*13*self.qubes

    def __getitem__(self, idx):
        hour = idx % 13
        x_quadrant = (idx//13) % self.x_quadrants
        y_quadrant = (idx//(13*self.x_quadrants)) % self.y_quadrants
        z_quadrant = (idx//(13*self.x_quadrants *
                            self.y_quadrants)) % self.z_quadrants
        x_range = [self.x_size*x_quadrant,  self.x_size*(x_quadrant+1)]
        y_range = [self.y_size*y_quadrant,  self.y_size*(y_quadrant+1)]
        z_range = [self.z_size*z_quadrant,  self.z_size*(z_quadrant+1)]
        filename = self.filenames[idx//(13*self.x_quadrants *
                                        self.y_quadrants*self.z_quadrants)]
        data, norm_params = load_file_tensor(
            self.folder + filename, x_range, y_range, z_range, hour)
        return data, norm_params


def load_file_tensor(filename, x_range, y_range, z_range, hour):
        # print(filename)
    with h5py.File(filename, 'r') as f:

        keys = list(f.keys())
        if 'geopotential_height_ml' in keys:
            keys.remove('geopotential_height_ml')
        shape = (z_range[1]-z_range[0], y_range[1] -
                 y_range[0], x_range[1]-x_range[0])
        tensor_data = torch.empty((len(keys),) + shape)
        norm_params = np.empty((len(keys), 2))
        for i, key in enumerate(keys):
            val = f[key][hour, z_range[0]:z_range[1], y_range[0]:y_range[1],
                         x_range[0]:x_range[1]]
            # val = f[key][x_range[0]:x_range[1], y_range[0]:y_range[1],
            #              z_range[0]:z_range[1], time]
            min_val = np.nanmin(val)
            max_val = np.nanmax(val)
            norm_params[i, 0] = min_val
            norm_params[i, 1] = max_val
            val = (val - min_val) / (max_val-min_val)
            tensor_data[i, :, :, :] = torch.from_numpy(val)
            if np.any(np.isnan(val)):
                print(filename)

    return tensor_data, norm_params


def load_day(filename):
    data = torch.empty(12, 4, 32, 128, 128)
    norm = torch.empty(12, 4, 2)
    for hour in range(12):
        hour_data, norm_params = load_file_tensor(
            filename, [0, 128], [0, 128], [0, 32], hour)
        data[hour, :, :, :] = hour_data
        norm[hour, :, :] = torch.from_numpy(norm_params)
    return data, norm


def reconstruct_data(data, norm_params):
    for i, norm_params_hour in enumerate(norm_params):
        for j, params in enumerate(norm_params_hour):
            max_val = params[1]
            min_val = params[0]
            data[i, j, :, :, :] *= float((max_val-min_val))
            data[i, j, :, :, :] += float(min_val)
    return data


def save_batch(filename, data):
    keys = ['air_pressure_ml', 'upward_air_velocity_ml',
            'x_wind_ml', 'y_wind_ml']
    data_map = {}
    for i, key in enumerate(keys):
        data_map[key] = data[:, i, :, :, :, ]
    save_hdf5(data_map, filename)
