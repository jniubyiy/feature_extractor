# model_ParNetAutoencoder.py
"""
ParNetAutoencoder: автоэнкодер для сжатых парнетов.
Промежуточное представление – структурированный парнет (structured parnet) в [-1,1].
ParNetDecoder идентичен Decoder из model_Autoencoder.py, но сохраняет разрешение.
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


class ParNetEncoder(nn.Module):
    """
    Структурирует сжатый парнет -> структурированный парнет в [-1,1].
    Аналог ParnetCompressor, но без даунсэмплинга.
    """
    def __init__(self, input_channels=4, bottleneck_channels=4, base_dim=128, num_blocks=2):
        super().__init__()
        self.init_conv = nn.Conv2d(input_channels, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, bottleneck_channels, kernel_size=3, padding=1)

        # Явное преобразование контекста перед динамической коррекцией
        self.ctx_proj = nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=1, padding=1)

        self.dynamic_refine = DynamicContextResidualBlock(bottleneck_channels, base_dim)
        self.tanh = nn.Tanh()

    def forward(self, compressed_parnet):
        x = self.init_conv(compressed_parnet)          # [B, base_dim, H/2, W/2]
        ctx_full = self.global_blocks(x)               # [B, base_dim, H/2, W/2] — контекст

        # Подготовленный контекст для динамического блока
        ctx_refined = self.ctx_proj(ctx_full)          # [B, base_dim, H/2, W/2]

        # Проекция в bottleneck_channels и первое ограничение (базовый структурированный парнет)
        structured = self.compress(ctx_full)           # [B, 4, H/2, W/2]
        structured = self.tanh(structured)             # [B, 4, H/2, W/2] строго в [-1,1]

        # Динамическая коррекция уже структурированного парнета
        structured = self.dynamic_refine(structured, ctx_refined)   # может выйти за границы
        structured = self.tanh(structured)             # повторно в [-1,1]
        return structured


class ParNetDecoder(nn.Module):
    """
    Восстанавливает сжатый парнет из структурированного.
    Полностью аналогичен Decoder из model_Autoencoder.py, но без изменения разрешения.
    """
    def __init__(self, bottleneck_channels=4, output_channels=4, base_dim=128, num_blocks=2):
        super().__init__()
        self.expand = nn.Conv2d(bottleneck_channels, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            GlobalContextScaleBlock(base_dim) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, output_channels, kernel_size=3, padding=1)
        self.dynamic_refine = DynamicContextResidualBlock(output_channels, base_dim)
        # Постобработка, аналогичная to_rgb в Decoder
        self.to_output = nn.Sequential(
            nn.Conv2d(output_channels, base_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(base_dim, output_channels, kernel_size=1)
        )

    def forward(self, structured_parnet):
        x = self.expand(structured_parnet)
        ctx = self.global_blocks(x)
        mid = self.compress(ctx)
        mid = self.dynamic_refine(mid, ctx)   # персональная коррекция
        out = self.to_output(mid)             # финальное преобразование
        return out


class ParNetAutoencoder(nn.Module):
    def __init__(self, input_channels=4, bottleneck_channels=4, base_dim=128, num_blocks=2,
                 **kwargs):
        super().__init__()
        self.encoder = ParNetEncoder(input_channels, bottleneck_channels, base_dim, num_blocks)
        self.decoder = ParNetDecoder(bottleneck_channels, input_channels, base_dim, num_blocks)

    def forward(self, x):
        structured_parnet = self.encoder(x)
        reconstructed = self.decoder(structured_parnet)
        return reconstructed