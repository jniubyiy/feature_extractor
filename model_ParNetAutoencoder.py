# model_ParNetAutoencoder.py
"""
ParNetAutoencoder: автоэнкодер для сжатых парнетов.
Промежуточное представление – структурированный парнет (structured parnet) в [-1,1].
"""
import torch
import torch.nn as nn
from model_Autoencoder import GlobalContextScaleBlock, DynamicContextResidualBlock

class ParNetEncoder(nn.Module):
    def __init__(self, input_channels=4, bottleneck_channels=4, base_dim=128, num_blocks=2):
        super().__init__()
        self.init_conv = nn.Conv2d(input_channels, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, bottleneck_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, compressed_parnet):
        x = self.init_conv(compressed_parnet)
        ctx = self.global_blocks(x)
        structured_parnet = self.compress(ctx)
        structured_parnet = self.tanh(structured_parnet)
        return structured_parnet

class ParNetDecoder(nn.Module):
    def __init__(self, bottleneck_channels=4, output_channels=4, base_dim=128, num_blocks=2):
        super().__init__()
        self.expand = nn.Conv2d(bottleneck_channels, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, output_channels, kernel_size=3, padding=1)
        self.dynamic_refine = DynamicContextResidualBlock(output_channels, base_dim)

    def forward(self, structured_parnet):
        x = self.expand(structured_parnet)
        ctx = self.global_blocks(x)
        out = self.compress(ctx)
        out = self.dynamic_refine(out, ctx)
        return out

class ParNetAutoencoder(nn.Module):
    def __init__(self, input_channels=4, bottleneck_channels=4, base_dim=128, num_blocks=2,
                 **kwargs):  # **kwargs для совместимости с разными конфигами
        super().__init__()
        self.encoder = ParNetEncoder(input_channels, bottleneck_channels, base_dim, num_blocks)
        self.decoder = ParNetDecoder(bottleneck_channels, input_channels, base_dim, num_blocks)

    def forward(self, x):
        structured_parnet = self.encoder(x)
        reconstructed = self.decoder(structured_parnet)
        return reconstructed