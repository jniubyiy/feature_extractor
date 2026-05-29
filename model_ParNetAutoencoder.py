# model_ParNetAutoencoder.py
"""
ParNetAutoencoder: автоэнкодер для сжатых парнетов (compressed parnet).
Энкодер сжимает до bottleneck_channels с Tanh на выходе (диапазон [-1,1]).
Декодер восстанавливает исходный сжатый парнет.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_Autoencoder import GlobalContextScaleBlock, DynamicContextResidualBlock

class ParNetEncoder(nn.Module):
    def __init__(self, input_channels=4, bottleneck_channels=4, base_dim=128, num_blocks=2):
        super().__init__()
        self.init_conv = nn.Conv2d(input_channels, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, bottleneck_channels, kernel_size=3, padding=1)
        self.output_activation = nn.Tanh()   # гарантирует [-1,1]

    def forward(self, x):
        x = self.init_conv(x)
        ctx = self.global_blocks(x)
        bottleneck = self.compress(ctx)
        bottleneck = self.output_activation(bottleneck)
        return bottleneck

class ParNetDecoder(nn.Module):
    def __init__(self, bottleneck_channels=4, output_channels=4, base_dim=128, num_blocks=2):
        super().__init__()
        self.expand = nn.Conv2d(bottleneck_channels, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, output_channels, kernel_size=3, padding=1)
        self.dynamic_refine = DynamicContextResidualBlock(output_channels, base_dim)

    def forward(self, x):
        x = self.expand(x)
        ctx = self.global_blocks(x)
        out = self.compress(ctx)
        out = self.dynamic_refine(out, ctx)
        return out

class ParNetAutoencoder(nn.Module):
    def __init__(self, input_channels=4, bottleneck_channels=4, base_dim=128, num_blocks=2):
        super().__init__()
        self.encoder = ParNetEncoder(input_channels, bottleneck_channels, base_dim, num_blocks)
        self.decoder = ParNetDecoder(bottleneck_channels, input_channels, base_dim, num_blocks)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon