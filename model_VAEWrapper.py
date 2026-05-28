# model_VAEWrapper.py
"""
StochasticEncoder и StochasticDecoder без ограничений диапазона.
Архитектура как у model_Autoencoder, но с SE-блоками для устойчивости к шуму.
Диапазон значений не ограничен (нет Tanh/clamp).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- Базовые блоки ---------------------

class SELayer(nn.Module):
    """Squeeze-and-Excitation: перекалибровка каналов для подавления шума."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


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


# ---------- Специализированные блоки для VAE ----------

class EncoderGlobalScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = GlobalScaleBlock(channels)
        self.se = SELayer(channels)

    def forward(self, x):
        x = self.block(x)
        x = self.se(x)
        return x


class DecoderGlobalScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = GlobalScaleBlock(channels)
        self.se = SELayer(channels)

    def forward(self, x):
        x = self.block(x)
        x = self.se(x)
        return x


class EncoderModulationBlock(nn.Module):
    def __init__(self, hint_channels, target_channels):
        super().__init__()
        self.mod = ModulationBlock(hint_channels, target_channels)
        self.se = SELayer(target_channels)

    def forward(self, hint, target):
        out = self.mod(hint, target)
        out = self.se(out)
        return out


class DecoderModulationBlock(nn.Module):
    def __init__(self, hint_channels, target_channels):
        super().__init__()
        self.mod = ModulationBlock(hint_channels, target_channels)
        self.se = SELayer(target_channels)

    def forward(self, hint, target):
        out = self.mod(hint, target)
        out = self.se(out)
        return out


class EncoderResidualBlock1x1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res = ResidualBlock1x1(channels)
        self.se = SELayer(channels)

    def forward(self, x):
        x = self.res(x)
        x = self.se(x)
        return x


class DecoderResidualBlock1x1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res = ResidualBlock1x1(channels)
        self.se = SELayer(channels)

    def forward(self, x):
        x = self.res(x)
        x = self.se(x)
        return x


# ---------- Основные модели ----------

class StochasticEncoder(nn.Module):
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, num_res_blocks=4, **kwargs):
        super().__init__()
        num_blocks = num_res_blocks

        self.init_conv = nn.Conv2d(compressed_channels, hidden_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            EncoderGlobalScaleBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(hidden_dim, stochastic_parnet_dim, kernel_size=3, padding=1)
        self.mod_block = EncoderModulationBlock(hidden_dim, stochastic_parnet_dim)
        self.refine_blocks = nn.Sequential(*[
            EncoderResidualBlock1x1(stochastic_parnet_dim) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.init_conv(x)
        global_hint = self.global_blocks(x)
        parnet = self.compress(x)
        parnet = self.mod_block(global_hint, parnet)
        parnet = self.refine_blocks(parnet)
        mu = parnet                                    # БЕЗ Tanh
        return mu

    def reparameterize(self, mu, strength=1.0):
        noise = torch.empty_like(mu).uniform_(-1.0, 1.0)
        z = mu + noise * strength                       # БЕЗ clamp
        return z

    def kl_divergence(self, mu):
        return 0.5 * torch.sum(mu.pow(2))


class StochasticDecoder(nn.Module):
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, num_res_blocks=6, **kwargs):
        super().__init__()
        num_blocks = num_res_blocks

        self.expand = nn.Conv2d(stochastic_parnet_dim, hidden_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            DecoderGlobalScaleBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.compress = nn.Conv2d(hidden_dim, compressed_channels, kernel_size=3, padding=1)
        self.mod_block = DecoderModulationBlock(hidden_dim, compressed_channels)
        self.refine_blocks = nn.Sequential(*[
            DecoderResidualBlock1x1(compressed_channels) for _ in range(num_blocks)
        ])

    def forward(self, z):
        x = self.expand(z)
        global_hint = self.global_blocks(x)
        mid = self.compress(x)
        mid = self.mod_block(global_hint, mid)
        mid = self.refine_blocks(mid)
        out = mid                                       # БЕЗ Tanh
        return out


class VAEWrapper(nn.Module):
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, **kwargs):
        super().__init__()
        self.encoder = StochasticEncoder(compressed_channels, stochastic_parnet_dim,
                                         hidden_dim, **kwargs)
        self.decoder = StochasticDecoder(compressed_channels, stochastic_parnet_dim,
                                         hidden_dim, **kwargs)

    def forward(self, x):
        mu = self.encoder(x)
        z = mu
        out = self.decoder(z)
        return out