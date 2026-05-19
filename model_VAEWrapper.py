# model_VAEWrapper.py
"""
VAE-надстройка над сжатым парнетом с увеличенной ёмкостью.
Добавлены остаточные блоки 1x1 и увеличена скрытая размерность.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1x1(nn.Module):
    """Остаточный блок для поканальной обработки (kernel_size=1)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + r)

class VAEWrapper(nn.Module):
    """
    Параметры:
        compressed_channels: число каналов во входном сжатом парнете.
        stochastic_parnet_dim: размерность стохастического парнета (каналов).
        hidden_dim: число каналов в скрытых слоях (по умолчанию 128).
        num_res_blocks: количество остаточных блоков в head и tail (по умолчанию 3).
    """
    def __init__(self, compressed_channels: int, stochastic_parnet_dim: int,
                 hidden_dim: int = 128, num_res_blocks: int = 3):
        super().__init__()
        self.stochastic_parnet_dim = stochastic_parnet_dim
        self.compressed_channels = compressed_channels

        # Head: сжатый парнет -> параметры стохастического парнета (μ, log σ²)
        head_layers = [
            nn.Conv2d(compressed_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_res_blocks):
            head_layers.append(ResidualBlock1x1(hidden_dim))
        head_layers.append(nn.Conv2d(hidden_dim, 2 * stochastic_parnet_dim, kernel_size=1))
        self.head = nn.Sequential(*head_layers)

        # Tail: стохастический парнет z -> восстановленный сжатый парнет
        tail_layers = [
            nn.Conv2d(stochastic_parnet_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_res_blocks):
            tail_layers.append(ResidualBlock1x1(hidden_dim))
        tail_layers.append(nn.Conv2d(hidden_dim, compressed_channels, kernel_size=1))
        self.tail = nn.Sequential(*tail_layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, c: torch.Tensor):
        params = self.head(c)
        mu, logvar = params.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        c_hat = self.tail(z)
        return c_hat, mu, logvar

    def loss(self, c, c_hat, mu, logvar, kld_weight=0.001):
        recon_loss = F.l1_loss(c_hat, c)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / c.size(0)
        total = recon_loss + kld_weight * kld_loss
        return total, recon_loss, kld_loss

    @torch.no_grad()
    def encode_deterministic(self, c: torch.Tensor) -> torch.Tensor:
        params = self.head(c)
        mu, _ = params.chunk(2, dim=1)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.tail(z)