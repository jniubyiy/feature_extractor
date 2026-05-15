# model_ParnetCompressorLevel2.py
"""
Модели второго уровня сжатия парнетов.
ParnetCompressorLevel2:  [B,4,H/2,W/2] -> [B,5,H/4,W/4]
ParnetDecompressorLevel2: [B,5,H/4,W/4] -> [B,4,H/2,W/2]
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

class ParnetCompressorLevel2(nn.Module):
    """
    Плавное сжатие второго уровня:
    1. Подъём каналов: 4 -> base_dim
    2. Обработка
    3. Расширение каналов: base_dim -> base_dim * expansion_factor
    4. Обработка
    5. Сжатие пространства вдвое (H/2,W/2 -> H/4,W/4)
    6. Обработка
    7. Финальная проекция в compressed_channels (5) каналов
    """
    def __init__(self, base_dim=64, num_blocks=2, expansion_factor=2, compressed_channels=5):
        super().__init__()
        self.high_dim = base_dim * expansion_factor

        self.init_conv = nn.Conv2d(4, base_dim, 3, padding=1)
        self.res_blocks_1 = nn.Sequential(*[ResidualBlock(base_dim) for _ in range(num_blocks)])

        self.expand_channels = nn.Conv2d(base_dim, self.high_dim, 1)
        self.res_blocks_2 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

        self.down = nn.Conv2d(self.high_dim, self.high_dim, 3, stride=2, padding=1)
        self.res_blocks_3 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

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

class ParnetDecompressorLevel2(nn.Module):
    """
    Плавное разжатие второго уровня:
    1. Вход: compressed_channels (5) -> high_dim (низкое разрешение)
    2. Обработка
    3. Повышение разрешения вдвое (H/4,W/4 -> H/2,W/2)
    4. Обработка
    5. Сужение каналов: high_dim -> base_dim
    6. Обработка
    7. Финальная проекция в 4 канала
    """
    def __init__(self, base_dim=64, num_blocks=2, expansion_factor=2, compressed_channels=5):
        super().__init__()
        self.high_dim = base_dim * expansion_factor

        self.init_conv = nn.Conv2d(compressed_channels, self.high_dim, 3, padding=1)
        self.res_blocks_1 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.high_dim, self.high_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_blocks_2 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

        self.reduce_channels = nn.Conv2d(self.high_dim, base_dim, 1)
        self.res_blocks_3 = nn.Sequential(*[ResidualBlock(base_dim) for _ in range(num_blocks)])

        self.to_parnet = nn.Sequential(
            nn.Conv2d(base_dim, 4, 3, padding=1),
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