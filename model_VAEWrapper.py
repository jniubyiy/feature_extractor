# model_VAEWrapper.py
"""
Отдельный модуль для добавления стохастичности и KL-регуляризации
поверх детерминированного сжатого парнета (выход ParnetCompressor).

Используется как надстройка:
    c = compressor(parnet)          # [B, C, H_c, W_c] – сжатый парнет
    vae = VAEWrapper(C, latent_dim)
    c_hat, mu, logvar = vae(c)      # c_hat – восстановленный сжатый парнет
    loss = vae.loss(c, c_hat, mu, logvar)

При инференсе:
    - Детерминированный режим: z = mu (можно модифицировать forward)
    - Генеративный режим: z ~ N(0, I) -> tail(z) -> c_sampled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEWrapper(nn.Module):
    """
    VAE-надстройка над сжатым парнетом.

    Параметры:
        compressed_channels: число каналов во входном сжатом парнете.
        latent_dim: размерность латентного пространства (каналов).
        hidden_dim: промежуточное число каналов в head/tail (по умолчанию 32).
    """
    def __init__(self, compressed_channels: int, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.compressed_channels = compressed_channels

        # Голова: сжатый парнет -> параметры распределения (μ и log σ²)
        self.head = nn.Sequential(
            nn.Conv2d(compressed_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2 * latent_dim, kernel_size=1)  # удвоенное число каналов
        )

        # Хвост: латентная переменная z -> восстановленный сжатый парнет
        self.tail = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, compressed_channels, kernel_size=1)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Репараметризация: z = μ + ε * exp(0.5 * log σ²), ε ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, c: torch.Tensor):
        """
        Принимает сжатый парнет c [B, C, H, W] и возвращает:
            c_hat: восстановленный сжатый парнет
            mu, logvar: параметры распределения
        """
        params = self.head(c)
        mu, logvar = params.chunk(2, dim=1)  # разделяем на μ и logvar по каналам
        z = self.reparameterize(mu, logvar)
        c_hat = self.tail(z)
        return c_hat, mu, logvar

    def loss(self, c: torch.Tensor, c_hat: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kld_weight: float = 0.001):
        """
        Вычисляет суммарную потерю:
            loss = L1(c_hat, c) + kld_weight * KL(N(μ,σ²) || N(0,I))
        Возвращает:
            total_loss, recon_loss, kld_loss
        """
        recon_loss = F.l1_loss(c_hat, c)
        # KL-дивергенция для нормального распределения
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / c.size(0)  # усреднение по батчу
        total = recon_loss + kld_weight * kld_loss
        return total, recon_loss, kld_loss

    @torch.no_grad()
    def encode_deterministic(self, c: torch.Tensor) -> torch.Tensor:
        """Детерминированное кодирование: возвращает μ."""
        params = self.head(c)
        mu, _ = params.chunk(2, dim=1)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Декодирует латентный вектор в сжатый парнет."""
        return self.tail(z)