from torch.utils.data import Dataset
import os
from python.fileHandling import load_hdf5
import h5py
import numpy as np
from torch.utils.data import DataLoader
import torch


def load_file_tensor(filename, x_range=[0, 130], y_range=[0, 130], z_range=[0, 30], time=0):
    with h5py.File(filename, 'r') as f:

        keys = list(f.keys())

        shape = (z_range[1]-z_range[0],y_range[1]-y_range[0],x_range[1]-x_range[0])
        tensor_data = torch.empty((len(keys),) + shape)

        for i,key in enumerate(keys):
            tensor_data[i,:,:,:] = torch.from_numpy(f[key][time,z_range[0]:z_range[1],y_range[0]:y_range[1], x_range[0]:x_range[1]])
    return tensor_data


class weatherDataSet(Dataset):
    def __init__(self,x_range, y_range, z_range,folder):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.folder = folder
        self.filenames = [f for f in os.listdir(folder) 
                        if f.endswith('.mat') and os.path.isfile(os.path.join(folder, f))]
    
    def __len__(self):
        return len(self.filenames)*13
    
    def __getitem__(self, idx):
        hour = idx%13
        data = load_file_tensor(self.folder + self.filenames[idx // 13], self.x_range, self.y_range, self.z_range, time=hour)
        return data
        
        


if __name__ == "__main__":
    dataset = weatherDataSet(x_range = [10,40], y_range = [10,40], z_range = [20,30], folder = 'data/calibration/')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    for i, batch in enumerate(dataloader):
        print(i, batch.shape)