import torch
import torch.nn as nn
import numpy as np
from weatherData import weatherDataSet
from torch.utils.data import DataLoader


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
        self.num_filters = [32]
        self.paddings = [0]
        self.strides = [2]
        self.kernels = [2]
        self.encoded = None


        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=image_channels,
                out_channels=self.num_filters[0],
                kernel_size=self.kernels[0],
                stride=self.strides[0],
                padding=self.kernels[0]
            ),
            nn.ReLU(),
        )
        spatial_dims = compute_dims(np.array(input_dimentions), [], self.paddings, self.kernels, self.strides)
        self.num_output_features = int(np.prod(spatial_dims[-1]) * self.num_filters[-1])
        
        

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.num_filters[0],
                out_channels=image_channels,
                kernel_size=self.kernels[0],
                stride=self.strides[0],
                padding=self.paddings[0]
            )
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        self.encoded = self.encoder(x)
        out = self.decoder(self.encoded)
        
        print(x.shape)
        print(self.encoded.shape)
        print(out.shape)
        assert out.shape == x.shape
        return out  

if __name__ == "__main__":
    dataset = weatherDataSet(x_range = [10,20], y_range = [10,20], z_range = [20,30], folder = 'data/calibration/')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    model = Model(7, [10, 10, 10])
    for i, batch in enumerate(dataloader):
         print(i, batch.shape)
         output = model.forward(batch)

    

    
    print(output.shape)
    