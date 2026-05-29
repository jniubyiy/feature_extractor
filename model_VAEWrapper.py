# model_VAEWrapper.py
"""
StochasticEncoder и StochasticDecoder с фиксированным диапазоном mu (Tanh) и шумом.
mask_seed из входного сжатого парнета, noise_seed – для конкретной реализации шума.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import zlib

# ========== хеш от входного парнета для mask_seed ==========
def compute_mask_seed(compressed_parnet: torch.Tensor) -> int:
    with torch.no_grad():
        data = compressed_parnet.detach().cpu().numpy().tobytes()
        crc1 = zlib.crc32(data)
        crc2 = zlib.crc32(data, crc1)
        return (crc1 << 32) | crc2

# ---------- Перестановочный слой ----------
class PermutationMask(nn.Module):
    def __init__(self, channels, height, width, seed):
        super().__init__()
        total = channels * height * width
        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(total, generator=generator)
        inv_perm = torch.argsort(perm)
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', inv_perm)
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        B = x.size(0)
        x_flat = x.view(B, -1)
        return x_flat[:, self.perm].view(B, self.channels, self.height, self.width)

    def inverse(self, x):
        B = x.size(0)
        x_flat = x.view(B, -1)
        return x_flat[:, self.inv_perm].view(B, self.channels, self.height, self.width)


# ---------- Базовые блоки ----------
class GlobalContextScaleBlock(nn.Module):
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
        weight_raw = self.weight_generator(ctx)
        weight_mat = weight_raw.view(B, C, C, H, W).permute(0, 3, 4, 1, 2)
        bias_raw = self.bias_generator(ctx)
        bias = bias_raw.permute(0, 2, 3, 1)
        x_perm = x.permute(0, 2, 3, 1)
        delta = torch.einsum('bhwij,bhwj->bhwi', weight_mat, x_perm) + bias
        delta = delta.permute(0, 3, 1, 2)
        return x + delta


class StochasticGlobalContextBlock(GlobalContextScaleBlock):
    def __init__(self, channels, expand_ratio=2):
        super().__init__(channels, expand_ratio, reduction=2)


class StochasticDynamicContextResidualBlock(nn.Module):
    def __init__(self, x_channels, ctx_channels, reduction=4):
        super().__init__()
        self.core = DynamicContextResidualBlock(x_channels, ctx_channels, reduction)
        self.delta_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(x_channels, x_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(x_channels // reduction, x_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, ctx):
        delta = self.core(x, ctx) - x
        scale = self.delta_se(delta)
        delta = delta * scale
        return x + delta


# ---------- Основные модели ----------
class StochasticEncoder(nn.Module):
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, num_res_blocks=4, H=256, W=256, **kwargs):
        super().__init__()
        self.H = H
        self.W = W
        self.stochastic_parnet_dim = stochastic_parnet_dim
        self.init_conv = nn.Conv2d(compressed_channels, hidden_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            StochasticGlobalContextBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        self.compress = nn.Conv2d(hidden_dim, stochastic_parnet_dim, kernel_size=3, padding=1)
        self.dynamic_refine = StochasticDynamicContextResidualBlock(stochastic_parnet_dim, hidden_dim)
        self.tanh = nn.Tanh()   # mu в [-1, 1]

    def forward(self, x):
        self.input_parnet = x.detach().clone()
        x = self.init_conv(x)
        ctx = self.global_blocks(x)
        parnet = self.compress(ctx)
        mu_raw = self.dynamic_refine(parnet, ctx)
        mu = self.tanh(mu_raw)   # ограничиваем диапазон
        return mu

    def reparameterize(self, mu, strength=1.0):
        """
        Возвращает (z, noise_seed, mask_seed).
        Шум равномерный в [-strength, strength].
        """
        mask_seed = compute_mask_seed(self.input_parnet)
        noise_seed = torch.randint(0, 2**31-1, (1,)).item()

        gen = torch.Generator(device=mu.device).manual_seed(noise_seed)
        noise = (torch.rand_like(mu, generator=gen) * 2 - 1) * strength
        z_raw = mu + noise

        perm_mask = PermutationMask(mu.size(1), mu.size(2), mu.size(3), mask_seed)
        z = perm_mask(z_raw)
        return z, noise_seed, mask_seed

    def kl_divergence(self, mu):
        # KL для ограниченного mu можно вычислять как обычно, но теперь mu не распределён как N(0,1)
        # Оставляем стандартную форму, она всё ещё будет работать как регуляризатор.
        return 0.5 * torch.sum(mu.pow(2))


class StochasticDecoder(nn.Module):
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, num_res_blocks=6, **kwargs):
        super().__init__()
        self.expand = nn.Conv2d(stochastic_parnet_dim, hidden_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            StochasticGlobalContextBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        self.compress = nn.Conv2d(hidden_dim, compressed_channels, kernel_size=3, padding=1)
        self.dynamic_refine = StochasticDynamicContextResidualBlock(compressed_channels, hidden_dim)

    def forward(self, z, mask_seed: int):
        B, C, H, W = z.shape
        perm_mask = PermutationMask(C, H, W, mask_seed)
        z_restored = perm_mask.inverse(z)
        x = self.expand(z_restored)
        ctx = self.global_blocks(x)
        mid = self.compress(ctx)
        out = self.dynamic_refine(mid, ctx)
        return out


class VAEWrapper(nn.Module):
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, H=256, W=256, **kwargs):
        super().__init__()
        self.encoder = StochasticEncoder(compressed_channels, stochastic_parnet_dim,
                                         hidden_dim, H=H, W=W, **kwargs)
        self.decoder = StochasticDecoder(compressed_channels, stochastic_parnet_dim,
                                         hidden_dim, **kwargs)

    def forward(self, x):
        mu = self.encoder(x)
        return mu