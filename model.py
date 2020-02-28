import torch
import torch.nn as nn
import numpy as np
from weatherData import weatherDataSet
from torch.utils.data import DataLoader
import trainer


def compute_dims(dims, out_dims, P, F, S):
    a = (dims - F[0] + 2 * P[0]) / S[0] + 1
    out_dims.append(a)
    print(f"A{a}")
    print(f"Out dims {out_dims}")

    if len(P) == 1:
        return out_dims
    else:
        return compute_dims(dims,out_dims,P[1:],F[1:],S[1:])

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
        kernel_dim = (2, 2, 2)
        self.num_filters = [16, 32, 64]
        self.paddings = [0,0,0]
        self.strides = [stride_dim,stride_dim,stride_dim]
        self.kernels = [kernel_dim,kernel_dim,kernel_dim]
        self.encoded = None

        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=image_channels,
                out_channels=self.num_filters[0],
                kernel_size=self.kernels[0],
                stride=self.strides[0],
                padding=0
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.num_filters[0],
                out_channels=self.num_filters[1],
                kernel_size=self.kernels[1],
                stride=self.strides[1],
                padding=0
            ),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.num_filters[1],
                out_channels=self.num_filters[0],
                kernel_size=self.kernels[1],
                stride = self.strides[1],
                output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                in_channels=self.num_filters[0],
                out_channels=image_channels,
                kernel_size=self.kernels[0],
                stride=self.strides[0],
            ),

        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        self.encoded = self.encoder(x)
        out = self.decoder(self.encoded)
        
        print("X",x.shape)
        print("enx",self.encoded.shape)
        print("out",out.shape)
        assert out.shape == x.shape
        return out  

if __name__ == "__main__":
    dataset = weatherDataSet(x_range = [0,30], y_range = [0,30], z_range = [0,30], folder = 'data/calibration/')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    model = Model(3, [30, 30, 30])
    print(model)
    trainer 

    for i, batch in enumerate(dataloader):
         output = model.forward(batch)
    print(output.shape)
    