import pathlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import time
import typing
import collections
import utils
import weatherData
from torchsummary import summary
from weatherData import weatherDataSet, load_day, save_batch, reconstruct_data
from torch.utils.data import DataLoader
from utils import compute_loss, to_cuda
from trainer import create_plots
import trainer
import plotting
from tqdm import tqdm


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
        self.num_filters = [16, 32, 64]
        self.encoded = None
        cov_layers = len(self.num_filters)
        input_data_points = image_channels*np.prod(input_dimentions)
        self.conv_output_shape = (self.num_filters[-1], input_dimentions[0]//(
            2**(cov_layers-0)), input_dimentions[1]//(2**(cov_layers-0)), input_dimentions[2]//(2**(cov_layers-0)))
        #self.conv_output_shape = (self.num_filters[-1],4,4,4)
        # self.dense_neurons = [
        # np.prod(list(self.conv_output_shape)), int(input_data_points*self.compression_rate)]

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=image_channels,
                out_channels=self.num_filters[0],
                kernel_size=3,
                padding=1),
            nn.BatchNorm3d(num_features=self.num_filters[0]),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.num_filters[0],
                out_channels=self.num_filters[1],
                kernel_size=3,
                padding=1),
            nn.BatchNorm3d(num_features=self.num_filters[1]),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[2],
                kernel_size=3,
                padding=1),
            nn.BatchNorm3d(num_features=self.num_filters[2]),
            nn.ReLU(),
        )
        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.upSize1 = nn.ConvTranspose3d(
            in_channels=self.num_filters[2],
            out_channels=self.num_filters[2],
            kernel_size=4,
            padding=1,
            stride=2,
        )
        self.upSize2 = nn.ConvTranspose3d(
            in_channels=self.num_filters[2],
            out_channels=self.num_filters[1],
            kernel_size=4,
            padding=1,
            stride=2,
        )
        self.upSize3 = nn.ConvTranspose3d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[0],
            kernel_size=4,
            padding=1,
            stride=2,
        )

        # decoder layers
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.num_filters[2],
                out_channels=self.num_filters[2],
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm3d(num_features=self.num_filters[2]),
            nn.ReLU(),
        )

        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[1],
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm3d(num_features=self.num_filters[1]),
            nn.ReLU(),
        )

        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.num_filters[0],
                out_channels=image_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            # nn.LeakyReLU(),
        )

        self.pool_indecies = [(), (), ()]
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0)

    def encode(self, x):
        x = self.conv1(x)
        self.dropout(x)
        (x, self.pool_indecies[0]) = self.pool(x)
        x = self.conv2(x)
        self.dropout(x)
        (x, self.pool_indecies[1]) = self.pool(x)
        x = self.conv3(x)
        (x, self.pool_indecies[2]) = self.pool(x)
        #x = x.view((-1, self.dense_neurons[0]))
        #x = self.encode_linear(x)
        return x

    def decode(self, x):
        #x = self.decode_linear(x)
        #x = x.view((-1,)+self.conv_output_shape)
        #x = self.unpool(x, self.pool_indecies[2])
        x = self.upSize1(x)
        x = self.t_conv1(x)
        # self.dropout(x)
        #x = self.unpool(x, self.pool_indecies[1])
        x = self.upSize2(x)
        x = self.t_conv2(x)
        # self.dropout(x)
        #x = self.unpool(x, self.pool_indecies[0])
        x = self.upSize3(x)
        x = self.t_conv3(x)
        return x

    def forward(self, x):

        x = self.encode(x)
        x = self.decode(x)
        # print("X: ", x.shape)
        # print("Xec: ", x_ec.shape)
        # print("Encode: ", self.encoded.shape)
        # print("out", out.shape)
        #assert out.shape == x.shape
        return x


def error_mean_std(model, dataloader):
    total_sum = np.zeros(4)
    total_samples = np.zeros(4)
    model.eval()
    print("Computing mean")
    for data, norm_params in tqdm(dataloader):
        #data = utils.to_cuda(data)
        with torch.no_grad():
            output = model(data)
        data = weatherData.reconstruct_data(
            data, norm_params).detach().numpy()
        output = weatherData.reconstruct_data(
            output, norm_params).detach().numpy()
        error = data-output
        n = np.prod(error.shape)//error.shape[1]
        for i in range(error.shape[1]):
            total_sum[i] += np.sum(error[:, i, :, :, :])
        total_samples += n

    means = total_sum/total_samples
    print(means)
    total_sum = np.zeros(4)
    print("Computing std")
    for data, norm_params in tqdm(dataloader):
        with torch.no_grad():
            output = model(data)
        data = weatherData.reconstruct_data(
            data, norm_params).detach().numpy()
        output = weatherData.reconstruct_data(
            output, norm_params).detach().numpy()
        error = data-output
        for i in range(error.shape[1]):
            total_sum[i] += np.sum(np.square(error[:,
                                                   i, :, :, :]-means[i]))
    stds = total_sum/(total_samples-1)
    return means, stds


if __name__ == "__main__":
    test_name = "24_times_163264"
    x_dim = 32
    y_dim = 32
    z_dim = 32
    batch_size = 32
    epochs = 5
    learning_rate = 1e-4
    early_stop_count = 5
    max_steps = 200000
    dataset = weatherDataSet(x_size=x_dim,
                             y_size=y_dim,
                             z_size=z_dim,
                             folder='data/validation/')
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8)
    val_dataset = weatherDataSet(x_size=128,
                                 y_size=128,
                                 z_size=32,
                                 folder='data/test2/')
    validation_dataloader = DataLoader(val_dataset,
                                       batch_size=32,
                                       shuffle=False,
                                       num_workers=8)
    dataloaders = (dataloader, validation_dataloader, validation_dataloader)
    model = Model(4, [z_dim, x_dim, x_dim])
    trainer = trainer.Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        max_steps,
        model,
        dataloaders,
        test_name
    )
    summary(model, (4, z_dim, y_dim, x_dim))
    print(torch.cuda.is_available())
    train = False
    if train:
        trainer.train()
        create_plots(trainer, test_name)
        trainer.load_best_model()
    else:
        trainer.load_best_model()
        # trainer.load_statistic(test_name)
        #create_plots(trainer, test_name)
    #original, norm_params = load_day('data/train/2018_01_01.mat')
    #original = to_cuda(original)
    # trainer.model.eval()
    # with torch.no_grad():
    #decomp = trainer.model(original)
    #reconstructed = reconstruct_data(decomp, to_cuda(norm_params))
    #reconstructed = reconstructed.cpu().detach().numpy()
    #save_batch('data/decomp/2018_01_01.mat', reconstructed)
    my, sigma = error_mean_std(trainer.model, validation_dataloader)
    print(my)
    print(sigma)
    exit()
    # data_sample, norm_params = next(iter(validation_dataloader))
    # data_sample = utils.to_cuda(data_sample)
    # # reconstructed = model(data_sample.view((1,) + tuple(data_sample.shape)))
    # trainer.model.eval()
    # with torch.no_grad():
    #     reconstructed = trainer.model(data_sample)

    # print(data_sample.shape)

    # data = data_sample.cpu().detach().numpy()
    # reconstructed = reconstructed.cpu().detach().numpy()
    # plotting.plot_histogram(
    #     data[:, 2, :, :, :], reconstructed[:, 2, :, :, :], title='x', bins=100)
    # plt.savefig('plots/x_hist_norm.png')
    # plotting.plot_histogram(
    #     data[:, 3, :, :, :], reconstructed[:, 3, :, :, :], title='y', bins=100)
    # plt.savefig('plots/y_hist_norm.png')
    # plotting.plot_histogram(
    #     data[:, 1, :, :, :], reconstructed[:, 1, :, :, :], title='up', bins=100)
    # plt.savefig('plots/up_hist_norm.png')

    # plotting.plot_histogram(
    #     data[:, 0, :, :, :], reconstructed[:, 0, :, :, :], title='preasure_hist', bins=100)
    # plt.savefig('plots/preasure_hist_norm.png')

    # data = weatherData.reconstruct_data(data, norm_params)
    # reconstructed = weatherData.reconstruct_data(reconstructed, norm_params)

    # plotting.plot_arrows3D(data[0, :], reconstructed[0, :], 5)
    # plt.savefig('plots/arrows.png')
    # plotting.plot_histogram(
    #     data[:, 2, :, :, :], reconstructed[:, 2, :, :, :], title='x', bins=100)
    # plt.savefig('plots/x_hist.png')
    # plotting.plot_histogram(
    #     data[:, 3, :, :, :], reconstructed[:, 3, :, :, :], title='y', bins=100)
    # plt.savefig('plots/y_hist.png')
    # plotting.plot_histogram(
    #     data[:, 1, :, :, :], reconstructed[:, 1, :, :, :], title='up', bins=100)
    # plt.savefig('plots/up_hist.png')

    # plotting.plot_histogram(
    #     data[:, 0, :, :, :], reconstructed[:, 0, :, :, :], title='preasure_hist', bins=100)
    # plt.savefig('plots/preasure_hist.png')
    # plotting.plot_contour(data, reconstructed, title='x')
    # plt.savefig('plots/contour_hist.png')
    # plt.show(block=False)

    # input("Press key to exit")
