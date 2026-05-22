# model_ParnetCompressor.py
"""
Модели для плавного сжатия и разжатия парнета.
ParnetCompressor:   [B,3,H,W] -> [B,4,H/2,W/2]
    - добавлены поканальные (1x1) блоки для независимого анализа каждого вектора.
ParnetDecompressor: [B,4,H/2,W/2] -> [B,3,H,W]
    - добавлены блоки модуляции (FiLM) для независимого изменения каждого пикселя.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------- Базовые блоки ---------------------
class ResidualBlock(nn.Module):
    """Остаточный блок 3x3 (пространственное смешивание)."""
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


class ResidualBlock1x1(nn.Module):
    """Остаточный блок 1x1 – поканальная обработка без учёта соседей."""
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


class ModulatedResBlock(nn.Module):
    """
    Остаточный блок с поканальной модуляцией (FiLM).
    Основная ветвь: две свёртки 3x3.
    Модуляция: две свёртки 1x1 предсказывают gamma и beta,
    которые поэлементно управляют выходом основной ветви.
    Позволяет точно менять каждый пиксель независимо.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

        # Предсказание gamma и beta
        self.mod_conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.mod_conv2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)

        mod = self.act(self.mod_conv1(x))
        mod = self.mod_conv2(mod)
        gamma, beta = mod.chunk(2, dim=1)

        out = out * gamma + beta
        out = self.act(out + residual)
        return out


# --------------------- Компрессор ---------------------
class ParnetCompressor(nn.Module):
    """
    Плавное сжатие парнета.
    Добавлены поканальные блоки 1x1 на начальном и конечном этапах,
    чтобы компрессор мог анализировать каждый вектор независимо.
    """
    def __init__(self, base_dim=64, num_blocks=2, expansion_factor=2, compressed_channels=4):
        super().__init__()
        self.high_dim = base_dim * expansion_factor

        # Этап 0: поканальный анализ (1x1)
        self.pre_1x1 = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            ResidualBlock1x1(base_dim)
        )

        # Этап 1: вход -> base_dim, пространственная обработка
        self.init_conv = nn.Conv2d(base_dim, base_dim, 3, padding=1)
        self.res_blocks_1 = nn.Sequential(*[ResidualBlock(base_dim) for _ in range(num_blocks)])

        # Этап 2: расширение каналов (1x1)
        self.expand_channels = nn.Conv2d(base_dim, self.high_dim, 1)
        self.res_blocks_2 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

        # Этап 3: уменьшение разрешения
        self.down = nn.Conv2d(self.high_dim, self.high_dim, 3, stride=2, padding=1)
        self.res_blocks_3 = nn.Sequential(*[ResidualBlock(self.high_dim) for _ in range(num_blocks)])

        # Этап 4: поканальная обработка перед финальной проекцией (1x1)
        self.post_1x1 = nn.Sequential(
            ResidualBlock1x1(self.high_dim),
            nn.Conv2d(self.high_dim, self.high_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Финальная проекция
        self.to_parnet = nn.Conv2d(self.high_dim, compressed_channels, 3, padding=1)

    def forward(self, parnet):
        # Поканальный анализ
        x = self.pre_1x1(parnet)

        # Пространственная обработка
        x = F.relu(self.init_conv(x))
        x = self.res_blocks_1(x)

        # Расширение каналов и обработка
        x = F.relu(self.expand_channels(x))
        x = self.res_blocks_2(x)

        # Сжатие
        x = F.relu(self.down(x))
        x = self.res_blocks_3(x)

        # Поканальное уточнение
        x = self.post_1x1(x)

        # Выход
        x = self.to_parnet(x)
        return x


# --------------------- Декомпрессор ---------------------
class ParnetDecompressor(nn.Module):
    """
    Плавное разжатие парнета.
    Обычные ResidualBlock заменены на ModulatedResBlock,
    чтобы можно было независимо менять каждый пиксель восстанавливаемого парнета.
    """
    def __init__(self, base_dim=64, num_blocks=2, expansion_factor=2, compressed_channels=4):
        super().__init__()
        self.high_dim = base_dim * expansion_factor

        # Низкое разрешение, high_dim каналов
        self.init_conv = nn.Conv2d(compressed_channels, self.high_dim, 3, padding=1)
        self.res_blocks_1 = nn.Sequential(*[ModulatedResBlock(self.high_dim) for _ in range(num_blocks)])

        # Повышение разрешения
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.high_dim, self.high_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_blocks_2 = nn.Sequential(*[ModulatedResBlock(self.high_dim) for _ in range(num_blocks)])

        # Уменьшение каналов
        self.reduce_channels = nn.Conv2d(self.high_dim, base_dim, 1)
        self.res_blocks_3 = nn.Sequential(*[ModulatedResBlock(base_dim) for _ in range(num_blocks)])

        # Финальная проекция
        self.to_parnet = nn.Conv2d(base_dim, 3, 3, padding=1)

    def forward(self, compressed_parnet):
        x = F.relu(self.init_conv(compressed_parnet))
        x = self.res_blocks_1(x)
        x = self.up(x)
        x = self.res_blocks_2(x)
        x = F.relu(self.reduce_channels(x))
        x = self.res_blocks_3(x)
        x = self.to_parnet(x)
        return x