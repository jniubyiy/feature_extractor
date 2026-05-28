# model_ParnetCompressor.py
"""
ParnetCompressor и ParnetDecompressor с явными блоками сжатия/расширения.
Все слои определены в __init__, чтобы избежать ошибок с torch.compile.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7,
                                   padding=3, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.depthwise(x))
        x = self.pointwise(x)
        x = self.act(x + residual)
        return x


class ModulationBlock(nn.Module):
    def __init__(self, hint_channels, target_channels):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Conv2d(hint_channels, target_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_channels, target_channels, kernel_size=1)
        )
        self.beta_net = nn.Sequential(
            nn.Conv2d(hint_channels, target_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_channels, target_channels, kernel_size=1)
        )

    def forward(self, hint, target):
        gamma = self.gamma_net(hint)
        beta = self.beta_net(hint)
        return target * gamma + beta


class ResidualBlock1x1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.act(x + residual)
        return x


class ParnetCompressor(nn.Module):
    """
    Парнет [B,3,H,W] -> сжатый парнет [B, compressed_channels, H/2, W/2].
    """
    def __init__(self, base_dim=64, num_blocks=4, compressed_channels=4, **kwargs):
        super().__init__()
        # 1. Поднятие числа каналов без изменения разрешения
        self.init_conv = nn.Conv2d(3, base_dim, kernel_size=3, padding=1)

        # 2. StageBlock1: GlobalScaleBlock на полном разрешении
        self.global_blocks = nn.Sequential(*[
            GlobalScaleBlock(base_dim) for _ in range(num_blocks)
        ])

        # 3. Блоки сжатия для global_hint, gamma и beta
        self.hint_down = nn.Conv2d(base_dim, compressed_channels, kernel_size=3, stride=2, padding=1)
        self.gamma_down = nn.Conv2d(base_dim, compressed_channels, kernel_size=3, stride=2, padding=1)
        self.beta_down  = nn.Conv2d(base_dim, compressed_channels, kernel_size=3, stride=2, padding=1)

        # Дополнительный блок для сжатия основного потока: base_dim -> compressed_channels с уменьшением размера
        self.compress_main = nn.Sequential(
            nn.AvgPool2d(2),                                    # уменьшает размер вдвое
            nn.Conv2d(base_dim, compressed_channels, kernel_size=1)
        )

        # 4. StageBlock2: модуляция + refine
        self.mod_block = ModulationBlock(hint_channels=compressed_channels, target_channels=compressed_channels)
        self.refine_blocks = nn.Sequential(*[
            ResidualBlock1x1(compressed_channels) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.init_conv(x)                        # [B, base_dim, H, W]
        global_hint = self.global_blocks(x)          # [B, base_dim, H, W]

        # Сжимаем глобальный хинт и получаем gamma/beta
        hint = self.hint_down(global_hint)           # [B, compressed_channels, H/2, W/2]
        gamma = self.gamma_down(global_hint)         # [B, compressed_channels, H/2, W/2]
        beta  = self.beta_down(global_hint)          # [B, compressed_channels, H/2, W/2]

        # Сжимаем основной поток (base_dim -> compressed_channels + уменьшение разрешения)
        compressed = self.compress_main(x)           # [B, compressed_channels, H/2, W/2]

        # Модуляция с готовыми gamma и beta
        compressed = compressed * gamma + beta
        compressed = self.refine_blocks(compressed)
        return compressed


class ParnetDecompressor(nn.Module):
    """
    Сжатый парнет [B, compressed_channels, H/2, W/2] -> парнет [B,3,H,W].
    """
    def __init__(self, base_dim=64, num_blocks=4, compressed_channels=4, **kwargs):
        super().__init__()
        # 1. Поднятие числа каналов без изменения разрешения
        self.init_conv = nn.Conv2d(compressed_channels, base_dim, kernel_size=3, padding=1)

        # 2. StageBlock1: GlobalScaleBlock на низком разрешении
        self.global_blocks = nn.Sequential(*[
            GlobalScaleBlock(base_dim) for _ in range(num_blocks)
        ])

        # 3. Блоки расширения для global_hint, gamma и beta
        self.hint_up = nn.ConvTranspose2d(base_dim, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gamma_up = nn.ConvTranspose2d(base_dim, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.beta_up  = nn.ConvTranspose2d(base_dim, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Дополнительный блок для расширения основного потока: base_dim -> 3 с увеличением размера
        self.expand_main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_dim, 3, kernel_size=1)
        )

        # 4. StageBlock2: модуляция + refine
        self.mod_block = ModulationBlock(hint_channels=3, target_channels=3)
        self.refine_blocks = nn.Sequential(*[
            ResidualBlock1x1(3) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.init_conv(x)                        # [B, base_dim, H/2, W/2]
        global_hint = self.global_blocks(x)          # [B, base_dim, H/2, W/2]

        # Расширяем глобальный хинт и получаем gamma/beta
        hint = self.hint_up(global_hint)             # [B, 3, H, W]
        gamma = self.gamma_up(global_hint)           # [B, 3, H, W]
        beta  = self.beta_up(global_hint)            # [B, 3, H, W]

        # Расширяем основной поток (base_dim -> 3 + увеличение разрешения)
        parnet = self.expand_main(x)                 # [B, 3, H, W]

        # Модуляция
        parnet = parnet * gamma + beta
        parnet = self.refine_blocks(parnet)
        return parnet