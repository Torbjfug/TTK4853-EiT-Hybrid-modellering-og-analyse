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
        # Encoder
        stride_dim = (2, 2, 2)
        kernel_dim = (3, 3, 3)
        self.num_filters = [10, 10, 10]
        self.paddings = [0, 0, 0]
        self.strides = [stride_dim, stride_dim, stride_dim]
        self.kernels = [kernel_dim, kernel_dim, kernel_dim]
        self.encoded = None

        self.encoder_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=image_channels,
                out_channels=self.num_filters[0],
                kernel_size=self.kernels[0],
                stride=1,
                padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_filters[0]),
            nn.Conv3d(
                in_channels=self.num_filters[0],
                out_channels=self.num_filters[0],
                kernel_size=self.kernels[0],
                stride=1,
                padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_filters[0]),
            nn.MaxPool3d((2, 2, 2),return_indices =True),

            nn.Conv3d(
                in_channels=self.num_filters[0],
                out_channels=self.num_filters[1],
                kernel_size=self.kernels[1],
                stride=1,
                padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_filters[1]),
            nn.Conv3d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[1],
                kernel_size=self.kernels[1],
                stride=self.strides[1],
                padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_filters[1]),
            nn.MaxPool3d((2, 2, 2),return_indices =True),

            nn.Conv3d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[2],
                kernel_size=(3, 3, 3),
                stride=1,
                padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_filters[2]),
            nn.Conv3d(
                in_channels=self.num_filters[2],
                out_channels=self.num_filters[2],
                kernel_size=(3, 3, 3),
                stride=1,
                padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_filters[2]),
        )
        in_shape = (1, image_channels)+tuple(input_dimentions)
        print(in_shape)
        dummy_data = torch.rand(in_shape)
        print(dummy_data.shape)
        dummy_compressed = self.encoder_conv(dummy_data)
        self.cov_end_shape = tuple(dummy_compressed.shape[1:])
        self.fc_input = np.prod(list(dummy_compressed.shape[1:]))
        print(dummy_compressed.shape)
        print(self.fc_input)

        fc_dims = [self.fc_input, 10]

        self.encoder_dense = nn.Sequential(
            nn.Linear( in_features=fc_dims[0], out_features = fc_dims[1]),
            nn.BatchNorm1d(fc_dims[1]),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.decoder_dense = nn.Sequential(
            nn.Linear(in_features = fc_dims[1], out_features = fc_dims[0]),
            nn.BatchNorm1d(fc_dims[0]),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.num_filters[2],
                out_channels=self.num_filters[1],
                kernel_size=(3, 3, 3),
                stride=(1,1, 1),
                output_padding=(0, 0, 0)
            ),
            nn.Upsample3d(2),
            nn.ConvTranspose3d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[0],
                kernel_size=(3, 3, 3),
                stride=(1,1, 1),
                output_padding=(0, 0, 0)
            ),
            nn.Upsample(2),
            nn.ConvTranspose3d(
                in_channels=self.num_filters[0],
                out_channels=image_channels,
                kernel_size=(3, 3, 3),
                stride=(1,1, 1),
                output_padding=(0, 0, 0)
            )
        )
        # self.decoder_conv = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         in_channels=self.num_filters[2],
        #         out_channels=self.num_filters[1],
        #         kernel_size=(3, 3, 3),
        #         stride=(1,1, 1),
        #         output_padding=(0, 0, 0)
        #     ),
        #     nn.ReLU(),
        #     nn.BatchNorm3d(self.num_filters[1]),
        #     nn.ConvTranspose3d(
        #         in_channels=self.num_filters[1],
        #         out_channels=self.num_filters[1],
        #         kernel_size=(1, 4, 4),
        #         stride=(1,2, 2),
        #         output_padding=(0, 0, 0)
        #     ),

        #     nn.ConvTranspose3d(
        #         in_channels=self.num_filters[1],
        #         out_channels=self.num_filters[0],
        #         kernel_size=(3, 3, 3),
        #         stride=(1,1, 1),
        #         output_padding=(0, 0, 0)
        #     ),
        #     nn.ReLU(),
        #     nn.BatchNorm3d(self.num_filters[0]),
        #     nn.ConvTranspose3d(
        #         in_channels=self.num_filters[0],
        #         out_channels=self.num_filters[0],
        #         kernel_size=(1, 4, 4),
        #         stride=(1,2, 2),
        #         output_padding=(0, 0, 0)
        #     ),

        #     nn.ConvTranspose3d(
        #         in_channels=self.num_filters[0],
        #         out_channels=image_channels,
        #         kernel_size=(3, 3, 3),
        #         stride=(1,1, 1),
        #         output_padding=(0, 0, 0)
        #     ),
            
        # )

    def forward(self, x):
        print("X: ", x.shape)
        x_ec = self.encoder_conv(x)
        print("Xec: ", x_ec.shape)
        self.encoded = self.encoder_dense(x_ec.view((-1, self.fc_input)))
        print("Encode: ", self.encoded.shape)
        x_dc = self.decoder_dense(self.encoded)
        print("Xdc: ", x_dc.shape)
        print("Xdc_2",x_dc.view((-1,)+ self.cov_end_shape).shape)
        out = self.decoder_conv(x_dc.view((-1,)+ self.cov_end_shape))
        # print("X: ", x.shape)
        # print("Xec: ", x_ec.shape)
        # print("Encode: ", self.encoded.shape)
        # print("Xdc: ", x_dc.shape)
        print("out", out.shape)
        assert out.shape == x.shape
        return out


if __name__ == "__main__":
    xy_max = 126
    dataset = weatherDataSet(x_range=[0, xy_max], y_range=[0, xy_max], z_range=[
        0, 30], folder='data/train/')
    dataloader = DataLoader(dataset, batch_size=2,
                            shuffle=True, num_workers=0)
    val_dataset = weatherDataSet(x_range=[0, xy_max], y_range=[0, xy_max], z_range=[
        0, 30], folder='data/validation/')
    validation_dataloader = DataLoader(val_dataset, batch_size=64,
                                       shuffle=True, num_workers=16)
    dataloaders = (dataloader, validation_dataloader, validation_dataloader)
    model = Model(3, [31, xy_max, xy_max])

    epochs = 2
    batch_size = 1
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
        # reconstructed = model(data_sample.view((1,) + tuple(data_sample.shape)))
        reconstructed = model(data_sample)
        print(data_sample.shape)

        data = data_sample.detach().numpy()
        reconstructed = reconstructed.detach().numpy()
        # plotting.plot_histogram(data_sample[:,0,:,:,:], reconstructed[:,0,:,:,:] ,title='x',bins=20)
        # plotting.plot_histogram(data_sample[:,1,:,:,:], reconstructed[:,1,:,:,:] ,title='y',bins=20)
        # plotting.plot_histogram(data_sample[:, 2,:,:,:], reconstructed[:, 2,:,:,:], title='up', bins=20)
        plotting.plot_contour(data, reconstructed, title='x')

        plt.show(block=False)
        input("Press key to exit")
