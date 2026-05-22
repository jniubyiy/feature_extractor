# model_VAEWrapper.py
"""
Разделённый VAE для сжатого парнета.
StochasticEncoder: улучшен – использует пространственные ResidualBlock 3x3
                   для лучшего улавливания локальных зависимостей.
StochasticDecoder: принимает стохастический парнет, восстанавливает сжатый парнет.
                   Внутренне обогащает вход локальной KL-дивергенцией
                   (оценённой по соседним пикселям) для лучшего использования корреляций.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------- Базовые блоки ---------------------
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


class SpatialResidualBlock(nn.Module):
    """Остаточный блок с пространственными свёртками 3x3 для учёта соседей."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + r)


# --------------------- Энкодер (улучшенный) ---------------------
class StochasticEncoder(nn.Module):
    """
    Преобразует сжатый парнет (B, C, H, W) в стохастический парнет той же формы.
    Улучшенная версия: вместо 1x1 блоков используются пространственные ResidualBlock 3x3,
    что позволяет улавливать локальные корреляции и генерировать более информативные mu, logvar.
    Параметры:
        compressed_channels: число каналов входного сжатого парнета
        stochastic_parnet_dim: размерность стохастического парнета (каналов)
        hidden_dim: число каналов в скрытых слоях
        num_res_blocks: количество пространственных остаточных блоков
    """
    def __init__(self, compressed_channels: int, stochastic_parnet_dim: int,
                 hidden_dim: int = 128, num_res_blocks: int = 3):
        super().__init__()
        # Проекция на скрытое пространство
        self.input_proj = nn.Conv2d(compressed_channels, hidden_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

        # Пространственные остаточные блоки
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(SpatialResidualBlock(hidden_dim))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Финальная проекция на mu и logvar
        self.output_proj = nn.Conv2d(hidden_dim, 2 * stochastic_parnet_dim, kernel_size=1)

    def forward(self, c):
        x = self.act(self.input_proj(c))
        x = self.res_blocks(x)
        params = self.output_proj(x)
        mu, logvar = params.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + strength * eps * std

    @torch.no_grad()
    def sample(self, c: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        mu, logvar = self.forward(c)
        return self.reparameterize(mu, logvar, strength)

    @torch.no_grad()
    def encode_deterministic(self, c: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(c)
        return mu


# --------------------- Декодер (KL‑признаки всегда активны) ---------------------
class StochasticDecoder(nn.Module):
    """
    Преобразует стохастический парнет (B, C, H, W) обратно в сжатый парнет.
    Вход: z (стохастический парнет) размерности (B, stochastic_parnet_dim, H, W).
    Внутренне:
      1. Вычисляет локальное среднее и дисперсию по окну 3x3.
      2. Считает пиксельную KL-дивергенцию N(μ_local, σ²_local) || N(0,1).
      3. Конкатенирует исходный z с картой KL-признаков (каналов: stochastic_parnet_dim * 2).
      4. Пропускает через пространственные ResidualBlock 3x3.
    Параметры:
        compressed_channels: число каналов выходного сжатого парнета
        stochastic_parnet_dim: размерность стохастического парнета (каналов)
        hidden_dim: число каналов в скрытых слоях
        num_res_blocks: количество пространственных остаточных блоков
    """
    def __init__(self, compressed_channels: int, stochastic_parnet_dim: int,
                 hidden_dim: int = 128, num_res_blocks: int = 3):
        super().__init__()
        input_channels = stochastic_parnet_dim * 2   # z + KL features

        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(SpatialResidualBlock(hidden_dim))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.output_proj = nn.Conv2d(hidden_dim, compressed_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Локальные статистики: среднее и дисперсия по окну 3x3
        mu_local = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        z_sq = z.pow(2)
        var_local = F.avg_pool2d(z_sq, kernel_size=3, stride=1, padding=1) - mu_local.pow(2)
        var_local = var_local.clamp(min=1e-8)  # стабильность логарифма

        # KL(N(μ_local, σ²_local) || N(0,1)) = 0.5 * (μ² + σ² - 1 - ln σ²)
        kl_feat = 0.5 * (mu_local.pow(2) + var_local - 1.0 - torch.log(var_local))

        # Объединяем с исходным z
        x = torch.cat([z, kl_feat], dim=1)

        x = self.act(self.input_proj(x))
        x = self.res_blocks(x)
        x = self.output_proj(x)
        return x