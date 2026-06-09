# model_VAEWrapper.py
"""
StochasticEncoder (mu, seed_tensor) и StochasticDecoder.
seed_tensor – тензор [B, 16, 1, 1] со значениями в [-1, 1],
детерминированно порождаемый из входного сжатого парнета
по математической формуле (без обучаемых параметров).
Он не зависит от обучаемых весов энкодера, поэтому для одного и того же
входного сжатого парнета всегда одинаков.

Из seed_tensor напрямую вычисляется целочисленный ключ (хеш),
который используется для инициализации генератора шума.
Это гарантирует, что для одного и того же входа шум всегда одинаков.

Адаптивное масштабирование шума:
- Если mu_abs_mean > noise_abs_mean: scale = 1 + noise/mu (z ~ mu).
- Иначе (шум доминирует): scale = (mu + noise)/noise (z ~ noise).

Шум равномерный в диапазоне [-NOISE_RANGE, NOISE_RANGE], умноженный на STOCHASTIC_STRENGTH.
StochasticDecoder использует seed_tensor как специализированную подсказку
через FiLM-слой.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Хеш тензора для генератора ----------
def seed_tensor_hash(seed_tensor: torch.Tensor) -> torch.Tensor:
    raw = seed_tensor.sum(dim=(1, 2, 3))  # [B]
    hash_int = (raw.abs() * 1e9).long() % (2**63 - 1)
    return hash_int.to(seed_tensor.device)


def reparameterize(mu, noise_range, strength, seed_tensor):
    """
    Добавляет к mu равномерный шум с адаптивным масштабированием.
    
    Параметры:
        mu: [B, C, H, W]
        noise_range: float – граница равномерного шума (шум из U[-range, range])
        strength: float – множитель шума
        seed_tensor: [B, 16, 1, 1] – для детерминированного шума
    
    Возвращает:
        z: [B, C, H, W] = (mu + noise*strength) / scale
        raw_noise: [B, C, H, W] – сгенерированный шум (без множителя и scale)
        scale: [B] – применённый масштаб (для информации)
    """
    hash_int = seed_tensor_hash(seed_tensor)
    B = mu.size(0)
    z_list, noise_list, scale_list = [], [], []

    for i in range(B):
        gen = torch.Generator(device=mu.device)
        gen.manual_seed(hash_int[i].item())

        # Равномерный шум в [-noise_range, noise_range]
        raw_noise = (torch.rand(mu[i].shape, generator=gen, device=mu.device) * 2 - 1) * noise_range
        effective_noise = raw_noise * strength

        mu_abs_mean = mu[i].abs().mean() + 1e-8
        noise_abs_mean = effective_noise.abs().mean()

        if mu_abs_mean > noise_abs_mean:
            scale = 1.0 + noise_abs_mean / mu_abs_mean
        else:
            scale = (mu_abs_mean + noise_abs_mean) / noise_abs_mean

        z_i = (mu[i] + effective_noise) / scale

        z_list.append(z_i.unsqueeze(0))
        noise_list.append(raw_noise.unsqueeze(0))
        scale_list.append(scale)

    z = torch.cat(z_list, dim=0)
    noise_batch = torch.cat(noise_list, dim=0)
    scale_batch = torch.stack(scale_list, dim=0)  # [B]
    return z, noise_batch, scale_batch


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


# ---------- Вспомогательные модули (SWT, Sobel, Laplace, ResBlock) ----------
class SWTHaar(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        lp_h = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
        hp_h = torch.tensor([[[0.5, -0.5]]], dtype=torch.float32)
        lp_v = torch.tensor([[[0.5], [0.5]]], dtype=torch.float32)
        hp_v = torch.tensor([[[0.5], [-0.5]]], dtype=torch.float32)
        self.register_buffer('lp_h', lp_h.repeat(channels, 1, 1, 1))
        self.register_buffer('hp_h', hp_h.repeat(channels, 1, 1, 1))
        self.register_buffer('lp_v', lp_v.repeat(channels, 1, 1, 1))
        self.register_buffer('hp_v', hp_v.repeat(channels, 1, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        x_pad_h = F.pad(x, (0, 1, 0, 0), mode='reflect')
        L = F.conv2d(x_pad_h, self.lp_h, groups=C)
        H = F.conv2d(x_pad_h, self.hp_h, groups=C)
        L_pad_v = F.pad(L, (0, 0, 0, 1), mode='reflect')
        H_pad_v = F.pad(H, (0, 0, 0, 1), mode='reflect')
        LL = F.conv2d(L_pad_v, self.lp_v, groups=C)
        LH = F.conv2d(L_pad_v, self.hp_v, groups=C)
        HL = F.conv2d(H_pad_v, self.lp_v, groups=C)
        HH = F.conv2d(H_pad_v, self.hp_v, groups=C)
        return LL, LH, HL, HH


class SobelFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', sobel_x.repeat(channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=self.channels)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=self.channels)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)


class LaplacianFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kernel', laplacian_kernel.repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=self.channels)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(out + residual)


# ---------- Основные модели ----------
class StochasticEncoder(nn.Module):
    """Генерирует mu (без ограничения диапазона) и seed_tensor [B, 16, 1, 1]
       по чисто математической формуле от исходного входа (не зависит от весов)."""
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, num_res_blocks=4, **kwargs):
        super().__init__()
        self.stochastic_parnet_dim = stochastic_parnet_dim
        self.init_conv = nn.Conv2d(compressed_channels, hidden_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            StochasticGlobalContextBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        self.compress_mu = nn.Conv2d(hidden_dim, stochastic_parnet_dim, kernel_size=3, padding=1)
        self.dynamic_refine_mu = StochasticDynamicContextResidualBlock(stochastic_parnet_dim, hidden_dim)

    def forward(self, x):
        # Вычисляем seed_tensor из исходного входа (не зависит от весов)
        abs_mean = x.abs().mean(dim=(2,3))
        freqs = torch.linspace(1.0, 10.0, 16, device=x.device).view(1, 16)
        phases = torch.linspace(0.0, 3.1415, 16, device=x.device).view(1, 16)
        weights = torch.ones(x.shape[1], 16, device=x.device) / x.shape[1]
        mixed = torch.matmul(abs_mean, weights)
        seed_raw = torch.sin(mixed * freqs + phases)
        seed_tensor = seed_raw.unsqueeze(-1).unsqueeze(-1)  # [B, 16, 1, 1]

        # Обучаемая часть
        x = self.init_conv(x)
        ctx = self.global_blocks(x)
        mu = self.compress_mu(ctx)
        mu = self.dynamic_refine_mu(mu, ctx)

        return mu, seed_tensor

    def mu_regularization(self, mu):
        return mu.pow(2).mean()


class StochasticDecoder(nn.Module):
    """Декодер с аналитическими признаками.
       Принимает z [B, C, H, W] и seed_tensor [B, 16, 1, 1].
    """
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, num_res_blocks=6, **kwargs):
        super().__init__()
        C = stochastic_parnet_dim
        self.swt = SWTHaar(C)
        self.sobel = SobelFilter(C)
        self.laplacian = LaplacianFilter(C)

        total_in = C + 4*C + 3*C + 2*C + 2*C + 2*C
        self.preprocess = nn.Sequential(
            nn.Conv2d(total_in, hidden_dim, kernel_size=3, padding=1),
            ResidualConvBlock(hidden_dim),
            ResidualConvBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )

        self.film_generator = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim * 2, kernel_size=1)
        )

        self.global_blocks = nn.Sequential(*[
            StochasticGlobalContextBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        self.compress = nn.Conv2d(hidden_dim, compressed_channels, kernel_size=3, padding=1)
        self.dynamic_refine = StochasticDynamicContextResidualBlock(compressed_channels, hidden_dim)

    def forward(self, z, seed_tensor):
        C = z.shape[1]
        LL, LH, HL, HH = self.swt(z)

        def local_var(x):
            mu = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            mu2 = F.avg_pool2d(x**2, kernel_size=3, stride=1, padding=1)
            return torch.relu(mu2 - mu**2)

        var_LH = local_var(LH)
        var_HL = local_var(HL)
        var_HH = local_var(HH)

        grad_z = self.sobel(z)
        grad_LL = self.sobel(LL)

        lap_z = self.laplacian(z)
        lap_LL = self.laplacian(LL)

        avg3 = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        avg5 = F.avg_pool2d(z, kernel_size=5, stride=1, padding=2)

        features = torch.cat([
            z,
            LL, LH, HL, HH,
            var_LH, var_HL, var_HH,
            grad_z, grad_LL,
            lap_z, lap_LL,
            avg3, avg5
        ], dim=1)

        x = self.preprocess(features)

        film_params = self.film_generator(seed_tensor)
        scale, bias = torch.chunk(film_params, 2, dim=1)
        x = x * scale + bias

        ctx = self.global_blocks(x)
        mid = self.compress(ctx)
        out = self.dynamic_refine(mid, ctx)
        return out


class VAEWrapper(nn.Module):
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, **kwargs):
        super().__init__()
        self.encoder = StochasticEncoder(compressed_channels, stochastic_parnet_dim, hidden_dim, **kwargs)
        self.decoder = StochasticDecoder(compressed_channels, stochastic_parnet_dim, hidden_dim, **kwargs)

    def forward(self, x):
        mu, seed_tensor = self.encoder(x)
        return mu, seed_tensor