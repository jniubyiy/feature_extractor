# model_Autoencoder.py
"""
Encoder и Decoder с динамическим контекстным рефайн-блоком.
Для каждого пикселя предсказывается своя матрица 1x1-преобразования,
что позволяет точно корректировать цвета без увеличения размера ядра.
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
        # Генератор матриц: из контекста предсказывает веса для каждого пикселя
        self.weight_generator = nn.Sequential(
            nn.Conv2d(ctx_channels, ctx_channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(ctx_channels // reduction, x_channels * x_channels, kernel_size=1)
        )
        # Генератор смещения (bias), зависящего от контекста
        self.bias_generator = nn.Sequential(
            nn.Conv2d(ctx_channels, ctx_channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(ctx_channels // reduction, x_channels, kernel_size=1)
        )

    def forward(self, x, ctx):
        B, C, H, W = x.shape                     # размеры основного тензора
        # Генерируем матрицы весов (B, C*C, H, W) и преобразуем в (B, H, W, C, C)
        weight_raw = self.weight_generator(ctx)  # [B, C*C, H, W]
        weight_mat = weight_raw.view(B, C, C, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, C, C]

        # Смещение: (B, C, H, W) -> (B, H, W, C)
        bias_raw = self.bias_generator(ctx)      # [B, C, H, W]
        bias = bias_raw.permute(0, 2, 3, 1)      # [B, H, W, C]

        # Применяем линейное преобразование к каждому пространственному вектору
        x_perm = x.permute(0, 2, 3, 1)           # [B, H, W, C]
        delta = torch.einsum('bhwij,bhwj->bhwi', weight_mat, x_perm) + bias  # [B, H, W, C]
        delta = delta.permute(0, 3, 1, 2)        # [B, C, H, W]

        return x + delta                         # остаточная связь


class Encoder(nn.Module):
    def __init__(self, base_dim=64, parnet_channels=6, num_blocks=4, expand_ratio=2, **kwargs):
        super().__init__()
        self.init_conv = nn.Conv2d(3, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim, expand_ratio) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, parnet_channels, kernel_size=3, padding=1)
        # Динамический блок: каждому пикселю парнета своя матрица на основе контекста
        self.dynamic_refine = DynamicContextResidualBlock(parnet_channels, base_dim)

    def forward(self, image):
        x = self.init_conv(image)
        ctx = self.global_blocks(x)
        parnet = self.compress(ctx)                 # [B, parnet_channels, H, W]
        parnet = self.dynamic_refine(parnet, ctx)   # персональная коррекция
        return parnet


class Decoder(nn.Module):
    def __init__(self, base_dim=64, parnet_channels=6, num_blocks=4, expand_ratio=2, **kwargs):
        super().__init__()
        self.expand = nn.Conv2d(parnet_channels, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim, expand_ratio) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)
        # В декодере тоже динамическая коррекция перед to_rgb
        self.dynamic_refine = DynamicContextResidualBlock(3, base_dim)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(base_dim, 3, kernel_size=1)
        )

    def forward(self, parnet):
        x = self.expand(parnet)
        ctx = self.global_blocks(x)
        mid = self.compress(ctx)
        mid = self.dynamic_refine(mid, ctx)         # персональная коррекция
        img = self.to_rgb(mid)
        return img


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