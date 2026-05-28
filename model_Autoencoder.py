# model_Autoencoder.py
"""
Encoder и Decoder с управлением глубины через num_blocks.
StageBlock1: цепочка InvertedGlobalScaleBlock (expand_ratio).
StageBlock2: InvertedModulationBlock + цепочка InvertedResidualBlock1x1.
Все блоки с расширением каналов внутри для увеличения параметров без роста памяти.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedGlobalScaleBlock(nn.Module):
    """
    Аналог GlobalScaleBlock с inverted bottleneck.
    Pointwise 1x1 → depthwise 7x7 → pointwise 1x1.
    channels * expand_ratio → channels * expand_ratio (depthwise) → channels.
    """
    def __init__(self, channels, expand_ratio=2):
        super().__init__()
        hidden = channels * expand_ratio
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden)
        self.compress = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.ReLU()  # исправлено

    def forward(self, x):
        residual = x
        x = self.act(self.expand(x))
        x = self.act(self.depthwise(x))
        x = self.compress(x)
        x = self.act(x + residual)
        return x


class InvertedModulationBlock(nn.Module):
    """
    ModulationBlock с расширением внутри.
    Генерирует gamma и beta из hint с промежуточной размерностью.
    """
    def __init__(self, hint_channels, target_channels, expand_ratio=2):
        super().__init__()
        hidden = hint_channels * expand_ratio
        self.hint_expand = nn.Conv2d(hint_channels, hidden, kernel_size=1)

        self.gamma_net = nn.Sequential(
            nn.ReLU(),  # исправлено
            nn.Conv2d(hidden, target_channels, kernel_size=1),
            nn.ReLU(),  # исправлено
            nn.Conv2d(target_channels, target_channels, kernel_size=1)
        )
        self.beta_net = nn.Sequential(
            nn.ReLU(),  # исправлено
            nn.Conv2d(hidden, target_channels, kernel_size=1),
            nn.ReLU(),  # исправлено
            nn.Conv2d(target_channels, target_channels, kernel_size=1)
        )

    def forward(self, hint, target):
        h = self.hint_expand(hint)
        gamma = self.gamma_net(h)
        beta = self.beta_net(h)
        return target * gamma + beta


class InvertedResidualBlock1x1(nn.Module):
    """
    ResidualBlock1x1 с inverted bottleneck.
    Pointwise 1x1 (channels → hidden) → Pointwise 1x1 (hidden → channels).
    """
    def __init__(self, channels, expand_ratio=2):
        super().__init__()
        hidden = channels * expand_ratio
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.ReLU()  # исправлено

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.act(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, base_dim=64, parnet_channels=3, num_blocks=4, expand_ratio=2, **kwargs):
        super().__init__()
        self.init_conv = nn.Conv2d(3, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            InvertedGlobalScaleBlock(base_dim, expand_ratio) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, parnet_channels, kernel_size=3, padding=1)
        self.mod_block = InvertedModulationBlock(base_dim, parnet_channels, expand_ratio)
        self.refine_blocks = nn.Sequential(*[
            InvertedResidualBlock1x1(parnet_channels, expand_ratio) for _ in range(num_blocks)
        ])

    def forward(self, image):
        x = self.init_conv(image)
        global_hint = self.global_blocks(x)
        parnet = self.compress(x)
        parnet = self.mod_block(global_hint, parnet)
        parnet = self.refine_blocks(parnet)
        return parnet


class Decoder(nn.Module):
    def __init__(self, base_dim=64, parnet_channels=3, num_blocks=4, expand_ratio=2, **kwargs):
        super().__init__()
        self.expand = nn.Conv2d(parnet_channels, base_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            InvertedGlobalScaleBlock(base_dim, expand_ratio) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)
        self.mod_block = InvertedModulationBlock(base_dim, 3, expand_ratio)
        self.refine_blocks = nn.Sequential(*[
            InvertedResidualBlock1x1(3, expand_ratio) for _ in range(num_blocks)
        ])
        self.to_rgb = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=1),
            nn.ReLU(),  # исправлено
            nn.Conv2d(base_dim, 3, kernel_size=1)
        )

    def forward(self, parnet):
        x = self.expand(parnet)
        global_hint = self.global_blocks(x)
        mid = self.compress(x)
        mid = self.mod_block(global_hint, mid)
        mid = self.refine_blocks(mid)
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