# model_ParnetCompressor.py
"""
ParnetCompressor и ParnetDecompressor с глобальным контекстом и динамической
попиксельной коррекцией. Никаких ReLU на выходе, нет статической модуляции.
Архитектура аналогична улучшенному model_Autoencoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalContextScaleBlock(nn.Module):
    """Inverted bottleneck + SE-подобное глобальное перевзвешивание."""
    def __init__(self, channels, expand_ratio=2, reduction=4):
        super().__init__()
        hidden = channels * expand_ratio
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden)
        self.compress = nn.Conv2d(hidden, channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.se_fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        out = self.act(self.expand(x))
        out = self.act(self.depthwise(out))
        out = self.compress(out)
        se = self.global_pool(out)
        se = self.act(self.se_fc1(se))
        se = self.sigmoid(self.se_fc2(se))
        out = out * se + residual
        return self.act(out)


class DynamicContextResidualBlock(nn.Module):
    """
    Для каждого пикселя генерирует персональную матрицу (x_channels × x_channels)
    и применяет её к соответствующему вектору. Не содержит ReLU/Tanh.
    """
    def __init__(self, x_channels, ctx_channels, reduction=4):
        super().__init__()
        self.weight_generator = nn.Sequential(
            nn.Conv2d(ctx_channels, ctx_channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(ctx_channels // reduction, x_channels * x_channels, kernel_size=1)
        )
        self.bias_generator = nn.Sequential(
            nn.Conv2d(ctx_channels, ctx_channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(ctx_channels // reduction, x_channels, kernel_size=1)
        )

    def forward(self, x, ctx):
        B, C, H, W = x.shape
        weight_raw = self.weight_generator(ctx)          # [B, C*C, H, W]
        weight_mat = weight_raw.view(B, C, C, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, C, C]
        bias_raw = self.bias_generator(ctx)              # [B, C, H, W]
        bias = bias_raw.permute(0, 2, 3, 1)              # [B, H, W, C]
        x_perm = x.permute(0, 2, 3, 1)                   # [B, H, W, C]
        delta = torch.einsum('bhwij,bhwj->bhwi', weight_mat, x_perm) + bias  # [B, H, W, C]
        delta = delta.permute(0, 3, 1, 2)                # [B, C, H, W]
        return x + delta


class ParnetCompressor(nn.Module):
    """
    Парнет [B, 3, H, W] -> сжатый парнет [B, compressed_channels, H/2, W/2].
    """
    def __init__(self, base_dim=64, num_blocks=4, compressed_channels=6, **kwargs):
        super().__init__()
        # Поднимаем число каналов
        self.init_conv = nn.Conv2d(3, base_dim, kernel_size=3, padding=1)

        # Глобальные блоки на полном разрешении
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim) for _ in range(num_blocks)
        ])

        # Downsample контекста: base_dim -> base_dim с уменьшением в 2 раза
        self.ctx_down = nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1)

        # Сжатие основного потока: base_dim -> compressed_channels, H/2, W/2
        self.compress = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(base_dim, compressed_channels, kernel_size=1)
        )

        # Динамическая коррекция сжатого парнета на основе контекста
        self.dynamic_refine = DynamicContextResidualBlock(compressed_channels, base_dim)

    def forward(self, x):
        x = self.init_conv(x)                       # [B, base_dim, H, W]
        ctx_full = self.global_blocks(x)            # [B, base_dim, H, W]

        # Контекст для низкого разрешения
        ctx_low = self.ctx_down(ctx_full)           # [B, base_dim, H/2, W/2]

        # Сжатое представление
        compressed = self.compress(ctx_full)        # [B, compressed_channels, H/2, W/2]

        # Точечная коррекция с использованием контекста
        compressed = self.dynamic_refine(compressed, ctx_low)
        return compressed


class ParnetDecompressor(nn.Module):
    """
    Сжатый парнет [B, compressed_channels, H/2, W/2] -> парнет [B, 3, H, W].
    """
    def __init__(self, base_dim=64, num_blocks=4, compressed_channels=6, **kwargs):
        super().__init__()
        # Поднимаем число каналов на низком разрешении
        self.init_conv = nn.Conv2d(compressed_channels, base_dim, kernel_size=3, padding=1)

        # Глобальные блоки на низком разрешении
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim) for _ in range(num_blocks)
        ])

        # Upsample контекста: base_dim -> base_dim с увеличением в 2 раза
        self.ctx_up = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)

        # Расширение основного потока: base_dim -> 3 канала с увеличением разрешения
        self.expand = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_dim, 3, kernel_size=1)
        )

        # Динамическая коррекция восстановленного парнета
        self.dynamic_refine = DynamicContextResidualBlock(3, base_dim)

    def forward(self, x):
        x = self.init_conv(x)                       # [B, base_dim, H/2, W/2]
        ctx_low = self.global_blocks(x)             # [B, base_dim, H/2, W/2]

        # Контекст для полного разрешения
        ctx_full = self.ctx_up(ctx_low)             # [B, base_dim, H, W]

        # Восстановление парнета
        parnet = self.expand(ctx_low)               # [B, 3, H, W]

        # Точечная коррекция
        parnet = self.dynamic_refine(parnet, ctx_full)
        return parnet