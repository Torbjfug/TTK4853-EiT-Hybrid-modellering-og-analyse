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


class Model(nn.Module):

    def __init__(self,
                 image_channels,
                 input_dimentions):

        super().__init__()
        # Encoder
        stride_dim = (2, 2, 2)
        kernel_dim = (4, 4, 2)
        self.num_filters = [64, 128, 256]
        self.paddings = [0, 0, 0]
        self.strides = [stride_dim, stride_dim, stride_dim]
        self.kernels = [kernel_dim, kernel_dim, kernel_dim]
        self.encoded = None

        self.conv1 = nn.Conv3d(
            in_channels=image_channels,
            out_channels=self.num_filters[0],
            kernel_size=3,
            padding=1)

        self.conv2 = nn.Conv3d(
            in_channels=self.num_filters[0],
            out_channels=self.num_filters[1],
            kernel_size=3,
            padding=1)

        self.conv3 = nn.Conv3d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[2],
            kernel_size=3,
            padding=1)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool3d(kernel_size=2, stride=2)

        # decoder layers
        self.t_conv1 = nn.ConvTranspose3d(
            in_channels=self.num_filters[2],
            out_channels=self.num_filters[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.t_conv2 = nn.ConvTranspose3d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[0],
            kernel_size=3,
            padding=1,
            stride=1
        )

        self.t_conv3 = nn.ConvTranspose3d(
            in_channels=self.num_filters[0],
            out_channels=image_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )

        self.pool_indecies = [(), ()]
        self.activation = nn.ReLU(inplace=True)

    def encode(self, x):
        x = self.conv1(x)
        self.activation(x)
        (x, self.pool_indecies[0]) = self.pool(x)
        x = self.conv2(x)
        self.activation(x)
        (x, self.pool_indecies[1]) = self.pool(x)
        x = self.conv3(x)
        self.activation(x)
        return x

    def decode(self, x):
        x = self.t_conv1(x)
        self.activation(x)
        x = self.unpool(x, self.pool_indecies[1])
        x = self.t_conv2(x)
        self.activation(x)
        x = self.unpool(x, self.pool_indecies[0])
        x = self.t_conv3(x)
        return x

    def forward(self, x):
        self.encoded = self.encode(x)
        out = self.decode(self.encoded)
        # print("X: ", x.shape)
        # print("Xec: ", x_ec.shape)
        # print("Encode: ", self.encoded.shape)
        # print("out", out.shape)
        assert out.shape == x.shape
        return out


if __name__ == "__main__":
    x_dim = 32
    batch_size = 32
    epochs = 5
    learning_rate = 1e-3
    early_stop_count = 4
    dataset = weatherDataSet(x_range=[0, x_dim],
                             y_range=[0, x_dim],
                             z_range=[0, 32],
                             folder='data/train/')
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
    val_dataset = weatherDataSet(x_range=[0, x_dim],
                                 y_range=[0, x_dim],
                                 z_range=[0, 32],
                                 folder='data/validation/')
    validation_dataloader = DataLoader(val_dataset,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=4)
    dataloaders = (dataloader, validation_dataloader, validation_dataloader)
    model = Model(3, [32, x_dim, x_dim])

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
    data_sample = utils.to_cuda(next(iter(validation_dataloader)))
    #reconstructed = model(data_sample.view((1,) + tuple(data_sample.shape)))
    trainer.model.eval()
    reconstructed = model(data_sample)

    print(data_sample.shape)

    data = data_sample.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    plotting.plot_histogram(
        data[:, 0, :, :, :], reconstructed[:, 0, :, :, :], title='x', bins=20)
    plt.savefig('plots/x_hist.png')
    plotting.plot_histogram(
        data[:, 1, :, :, :], reconstructed[:, 1, :, :, :], title='y', bins=20)
    plt.savefig('plots/y_hist.png')
    plotting.plot_histogram(
        data[:, 2, :, :, :], reconstructed[:, 2, :, :, :], title='up', bins=20)
    plt.savefig('plots/up_hist.png')
    plotting.plot_contour(data, reconstructed, title='x')
    plt.savefig('plots/contour_hist.png')
    plt.show(block=False)
    input("Press key to exit")
