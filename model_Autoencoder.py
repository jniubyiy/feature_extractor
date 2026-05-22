# model_Autoencoder.py
"""
Улучшенный Decoder с блоками модуляции (FiLM-подобные).
Позволяет гибко менять отдельные пиксели без влияния на соседей
и управлять цветом, насыщенностью, яркостью и другими поканальными характеристиками.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Остаточный блок 3x3 с dropout (используется в Encoder)."""
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        r = x
        x = self.act(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return self.act(x + r)


class ModulatedResBlock(nn.Module):
    """
    Остаточный блок с поканальной модуляцией (FiLM).
    Основная ветвь: две свёртки 3x3 с ReLU.
    Модуляция: две свёртки 1x1 (поканальные) предсказывают gamma и beta,
    которые поэлементно умножают/смещают выход основной ветви.
    Позволяет гибко управлять каждым пикселем независимо.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

        # Модуляция: предсказываем gamma и beta (размерность 2*channels -> делим пополам)
        self.mod_conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.mod_conv2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)

    def forward(self, x):
        # Основная ветвь
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)

        # Модуляция
        mod = self.act(self.mod_conv1(x))
        mod = self.mod_conv2(mod)
        gamma, beta = mod.chunk(2, dim=1)

        out = out * gamma + beta
        out = self.act(out + residual)
        return out


class Decoder(nn.Module):
    """
    Парнет [B,3,W,H] -> изображение [B,3,W,H] в [-1,1].
    Улучшенная версия:
      - начальная поканальная проекция 1x1
      - несколько ModulatedResBlock (пространственные свёртки + поканальная модуляция)
      - финальная проекция с Tanh
    """
    def __init__(self, base_dim=64, num_blocks=3, parnet_channels=3, dropout_rate=0.1):
        super().__init__()
        # Начальная проекция без перемешивания пространства (1x1)
        self.init_conv = nn.Conv2d(parnet_channels, base_dim, kernel_size=1)

        # Блоки с модуляцией
        self.res_blocks = nn.Sequential(*[
            ModulatedResBlock(base_dim) for _ in range(num_blocks)
        ])

        # Финальная проекция: можно добавить несколько 1x1 для гибкости
        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, parnet):
        x = self.init_conv(parnet)
        x = self.res_blocks(x)
        img = self.to_rgb(x)
        return img


class Encoder(nn.Module):
    """Изображение [B,3,W,H] -> парнет [B,3,W,H] (без постобработки)."""
    def __init__(self, base_dim=64, num_blocks=3, parnet_channels=3, dropout_rate=0.1):
        super().__init__()
        self.init_conv = nn.Conv2d(3, base_dim, 3, padding=1)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(base_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.to_parnet = nn.Conv2d(base_dim, parnet_channels, 3, padding=1)

    def forward(self, image):
        x = F.relu(self.init_conv(image))
        x = self.res_blocks(x)
        parnet = self.to_parnet(x)
        return parnet


class Autoencoder(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)

    def forward(self, image, encoder_device, decoder_device):
        parnet = self.encoder(image.to(encoder_device))
        parnet = parnet.to(decoder_device)
        rec = self.decoder(parnet)
        return rec