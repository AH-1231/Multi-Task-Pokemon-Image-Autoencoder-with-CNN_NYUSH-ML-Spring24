import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=4):
        super().__init__()
        # Make sure the layers are consistent with your checkpoint weights
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def preprocess(self, x):
        # The input data is always on value range [0, 1], you may do normalization here
        return 2 * x - 1

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        # Make sure you only return the latent tensor, not a tuple
        return mean

    def decode(self, z):
        h = self.post_quant_conv(z)
        x_recon = self.decoder(h)
        # Make sure you only return the recon tensor, not a tuple
        return x_recon
