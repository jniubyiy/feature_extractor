# model_ParnetCompressor.py
"""
Модели для плавного сжатия и разжатия парнета.
ParnetCompressor:  [B,3,H,W] -> [B,4,H/2,W/2]   (увеличение каналов, затем уменьшение размера)
ParnetDecompressor: [B,4,H/2,W/2] -> [B,3,H,W] (восстановление)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + r)


class ParnetCompressor(nn.Module):
    """
    Плавное сжатие:
    1. Подъём каналов: 3 -> base_dim
    2. Обработка на base_dim каналах
    3. Расширение каналов: base_dim -> base_dim * expansion_factor (перед сжатием)
    4. Обработка на расширенных каналах
    5. Сжатие пространства вдвое (stride=2)
    6. Обработка на низком разрешении
    7. Финальная проекция в compressed_channels (4) каналов
    """
    def __init__(self, base_dim=64, num_blocks=2, expansion_factor=2, compressed_channels=4):
        super().__init__()
        self.base_dim = base_dim
        self.expansion_factor = expansion_factor
        self.high_dim = base_dim * expansion_factor

        # Этап 1: вход -> base_dim
        self.init_conv = nn.Conv2d(3, base_dim, 3, padding=1)
        self.res_blocks_1 = nn.Sequential(*[ResidualBlock(base_dim) for _ in range(num_blocks)])

        # Этап 2: расширение каналов (без изменения разрешения)
        self.expand_channels = nn.Conv2d(base_dim, self.high_dim, 1)
        self.res_blocks_2 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

        # Этап 3: уменьшение разрешения вдвое
        self.down = nn.Conv2d(self.high_dim, self.high_dim, 3, stride=2, padding=1)
        self.res_blocks_3 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

        # Этап 4: проекция на выходное число каналов
        self.to_parnet = nn.Sequential(
            nn.Conv2d(self.high_dim, compressed_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, parnet):
        x = F.relu(self.init_conv(parnet))
        x = self.res_blocks_1(x)
        x = F.relu(self.expand_channels(x))
        x = self.res_blocks_2(x)
        x = F.relu(self.down(x))
        x = self.res_blocks_3(x)
        return self.to_parnet(x)


class ParnetDecompressor(nn.Module):
    """
    Плавное разжатие:
    1. Вход: compressed_channels (4) -> high_dim на низком разрешении
    2. Обработка на низком разрешении
    3. Повышение разрешения вдвое
    4. Обработка на высоком разрешении (high_dim)
    5. Сужение каналов: high_dim -> base_dim
    6. Обработка на base_dim
    7. Финальная проекция в 3 канала
    """
    def __init__(self, base_dim=64, num_blocks=2, expansion_factor=2, compressed_channels=4):
        super().__init__()
        self.high_dim = base_dim * expansion_factor

        # Этап 1: низкое разрешение, high_dim каналов
        self.init_conv = nn.Conv2d(compressed_channels, self.high_dim, 3, padding=1)
        self.res_blocks_1 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

        # Этап 2: повышение разрешения
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.high_dim, self.high_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_blocks_2 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

        # Этап 3: уменьшение каналов до base_dim
        self.reduce_channels = nn.Conv2d(self.high_dim, base_dim, 1)
        self.res_blocks_3 = nn.Sequential(*[ResidualBlock(base_dim) for _ in range(num_blocks)])

        # Этап 4: проекция в 3 канала
        self.to_parnet = nn.Sequential(
            nn.Conv2d(base_dim, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, compressed_parnet):
        x = F.relu(self.init_conv(compressed_parnet))
        x = self.res_blocks_1(x)
        x = self.up(x)
        x = self.res_blocks_2(x)
        x = F.relu(self.reduce_channels(x))
        x = self.res_blocks_3(x)
        return self.to_parnet(x)