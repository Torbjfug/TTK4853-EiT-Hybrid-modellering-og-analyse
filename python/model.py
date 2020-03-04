import pathlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import time
import typing
import collections
import utils
from weatherData import weatherDataSet
from torch.utils.data import DataLoader
from utils import compute_loss
from trainer import create_plots
import trainer

import plotting


def compute_dims(dims, out_dims, P, F, S):
    a = (dims - F[0] + 2 * P[0]) / S[0] + 1
    out_dims.append(a)
    print(f"A{a}")
    print(f"Out dims {out_dims}")

    if len(P) == 1:
        return out_dims
    else:
        return compute_dims(dims, out_dims, P[1:], F[1:], S[1:])


class FullyConnectedModel(nn.Module):
    def __init__(self,
                 input_chanels,
                 input_dimentions):
        super().__init__()
        self.encoded = None
        self.input_dimentions = tuple(input_dimentions)
        print(self.input_dimentions)
        self.layers = [np.prod(input_dimentions)*input_chanels, 500, 50]
        print(self.layers)
        self.encoder = nn.Sequential(
            nn.Linear(self.layers[0], self.layers[1]),
            nn.ReLU(),
            nn.Linear(self.layers[1], self.layers[2]),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.layers[2], self.layers[1]),
            nn.ReLU(),
            nn.Linear(self.layers[1], self.layers[0]),
        )

    def forward(self, x):
        x = x.view(-1, self.layers[0])
        self.encoded = self.encoder(x)
        out = self.decoder(self.encoded)
        assert out.shape == x.shape
        return out

class Model(nn.Module):

    def __init__(self,
                 image_channels,
                 input_dimentions):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        #Encoder
        stride_dim = (2, 2, 2)
        kernel_dim = (4, 4, 2)
        self.num_filters = [2, 2, 2]
        self.paddings = [0,0,0]
        self.strides = [stride_dim,stride_dim,stride_dim]
        self.kernels = [kernel_dim,kernel_dim,kernel_dim]
        self.encoded = None

        self.conv1 = nn.Conv3d(image_channels, self.num_filters[0], 3, padding=1)  

        self.conv2 = nn.Conv3d(self.num_filters[0], self.num_filters[1], 3, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=2,stride=2,return_indices=True)
        self.unpool = nn.MaxUnpool3d(kernel_size=2,stride=2)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose3d(
            in_channels = self.num_filters[1],
            out_channels = self.num_filters[0],
            kernel_size = 2,
            stride = 1,)
        
        self.t_conv1 = nn.ConvTranspose3d(
            in_channels = self.num_filters[0],
            out_channels = image_channels,
            kernel_size = 2,
            stride = 1)
        
        self.pool_indecies = [()]

    def encode(self,x):
        x = self.conv1(x)
        (x,pool_idx) = self.pool(x)
        x = self.conv2(x)
        self.pool_indecies[0] = pool_idx
        return x

    def decode(self,x):
        x1 = self.t_conv1(x)
        x = self.unpool(x1,self.pool_indecies[0])
        x = self.t_conv2(x)
        return x

    def forward(self, x):
        self.encoded = self.encode(x)
        out = self.decode(self.encoded)
        print("X: ", x.shape)
        print("Encode: ", self.encoded.shape)
        print("out", out.shape)
        assert out.shape == x.shape
        return out


if __name__ == "__main__":
    x_dim = 125
    dataset = weatherDataSet(x_range=[0, x_dim], y_range=[0, x_dim], z_range=[
        0, 30], folder='data/train/')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    val_dataset = weatherDataSet(x_range=[0, x_dim], y_range=[0, x_dim], z_range=[0, 30], folder='data/validation/')
    validation_dataloader = DataLoader(val_dataset, batch_size=64,
                                       shuffle=True, num_workers=4)
    dataloaders = (dataloader, validation_dataloader, validation_dataloader)
    model = Model(3, [31, x_dim, x_dim])

    epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    early_stop_count = 5

    trainer = trainer.Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )

    print(torch.cuda.is_available())
    train = True
    if train:
        trainer.train()
        create_plots(trainer, "test")
    else:
        trainer.load_best_model()
        data_sample = next(iter(validation_dataloader))
        #reconstructed = model(data_sample.view((1,) + tuple(data_sample.shape)))
        reconstructed = model(data_sample)
        print(data_sample.shape)

        data = data_sample.detach().numpy()
        reconstructed = reconstructed.detach().numpy()
        #plotting.plot_histogram(data_sample[:,0,:,:,:], reconstructed[:,0,:,:,:] ,title='x',bins=20)

        #plotting.plot_histogram(data_sample[:,1,:,:,:], reconstructed[:,1,:,:,:] ,title='y',bins=20)

        #plotting.plot_histogram(data_sample[:, 2,:,:,:], reconstructed[:, 2,:,:,:], title='up', bins=20)
        plotting.plot_contour(data, reconstructed ,title='x')

        
        plt.show(block=False)
        input("Press key to exit")