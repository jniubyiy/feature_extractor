# model_VAEWrapper.py
"""
StochasticEncoder (mu, noise_seed) и StochasticDecoder.
Ручное управление шумом через LOG_VAR_VALUE.
Семплирование: z = mu + ε * exp(0.5 * log_var), ε ~ N(0,1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, log_var, strength=1.0):
    """Семплирует z = mu + ε * exp(0.5 * log_var). Возвращает z, ε, σ."""
    std = torch.exp(0.5 * log_var) * strength
    eps = torch.randn_like(mu)
    z = mu + eps * std
    return z, eps, std


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


# ---------- Вспомогательные модули для извлечения признаков из z ----------
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
    """Генерирует mu (в [-1,1]) и noise_seed (неограниченный)."""
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, num_res_blocks=4, **kwargs):
        super().__init__()
        self.stochastic_parnet_dim = stochastic_parnet_dim
        self.init_conv = nn.Conv2d(compressed_channels, hidden_dim, kernel_size=3, padding=1)
        self.global_blocks = nn.Sequential(*[
            StochasticGlobalContextBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        # Голова для mu
        self.compress_mu = nn.Conv2d(hidden_dim, stochastic_parnet_dim, kernel_size=3, padding=1)
        self.dynamic_refine_mu = StochasticDynamicContextResidualBlock(stochastic_parnet_dim, hidden_dim)
        # Голова для noise_seed
        self.compress_seed = nn.Conv2d(hidden_dim, stochastic_parnet_dim, kernel_size=3, padding=1)
        self.dynamic_refine_seed = StochasticDynamicContextResidualBlock(stochastic_parnet_dim, hidden_dim)

    def forward(self, x):
        x = self.init_conv(x)
        ctx = self.global_blocks(x)
        mu = self.compress_mu(ctx)
        mu = self.dynamic_refine_mu(mu, ctx)
        mu = torch.tanh(mu)
        noise_seed = self.compress_seed(ctx)
        noise_seed = self.dynamic_refine_seed(noise_seed, ctx)
        return mu, noise_seed

    def mu_regularization(self, mu):
        """L2-регуляризация mu (притягивает к нулю)."""
        return mu.pow(2).mean()


class StochasticDecoder(nn.Module):
    """Декодер с аналитическими признаками.
       Принимает z и noise_seed, возвращает структурированный парнет без tanh."""
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, num_res_blocks=6, **kwargs):
        super().__init__()
        C = stochastic_parnet_dim
        # Проекция объединённого входа до C каналов для совместимости с анализом признаков
        self.input_proj = nn.Conv2d(2 * C, C, kernel_size=1)
        self.swt = SWTHaar(C)
        self.sobel = SobelFilter(C)
        self.laplacian = LaplacianFilter(C)

        total_in = C + 4*C + 3*C + 2*C + 2*C + 2*C   # 14*C
        self.preprocess = nn.Sequential(
            nn.Conv2d(total_in, hidden_dim, kernel_size=3, padding=1),
            ResidualConvBlock(hidden_dim),
            ResidualConvBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        self.global_blocks = nn.Sequential(*[
            StochasticGlobalContextBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        self.compress = nn.Conv2d(hidden_dim, compressed_channels, kernel_size=3, padding=1)
        self.dynamic_refine = StochasticDynamicContextResidualBlock(compressed_channels, hidden_dim)
        # Без финального tanh

    def forward(self, z, noise_seed):
        # Объединяем z и noise_seed по каналам и проецируем до C
        combined = torch.cat([z, noise_seed], dim=1)   # B, 2C, H, W
        combined = self.input_proj(combined)           # B, C, H, W

        B, C, H, W = combined.shape
        LL, LH, HL, HH = self.swt(combined)

        def local_var(x):
            mu = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            mu2 = F.avg_pool2d(x**2, kernel_size=3, stride=1, padding=1)
            return torch.relu(mu2 - mu**2)

        var_LH = local_var(LH)
        var_HL = local_var(HL)
        var_HH = local_var(HH)

        grad_z = self.sobel(combined)
        grad_LL = self.sobel(LL)

        lap_z = self.laplacian(combined)
        lap_LL = self.laplacian(LL)

        avg3 = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
        avg5 = F.avg_pool2d(combined, kernel_size=5, stride=1, padding=2)

        features = torch.cat([
            combined,
            LL, LH, HL, HH,
            var_LH, var_HL, var_HH,
            grad_z, grad_LL,
            lap_z, lap_LL,
            avg3, avg5
        ], dim=1)

        x = self.preprocess(features)
        ctx = self.global_blocks(x)
        mid = self.compress(ctx)
        out = self.dynamic_refine(mid, ctx)
        return out   # без tanh


class VAEWrapper(nn.Module):
    """Обёртка, содержащая encoder и decoder."""
    def __init__(self, compressed_channels=4, stochastic_parnet_dim=4,
                 hidden_dim=128, **kwargs):
        super().__init__()
        self.encoder = StochasticEncoder(compressed_channels, stochastic_parnet_dim, hidden_dim, **kwargs)
        self.decoder = StochasticDecoder(compressed_channels, stochastic_parnet_dim, hidden_dim, **kwargs)

    def forward(self, x):
        mu, noise_seed = self.encoder(x)
        # Возвращаем mu, noise_seed и фиктивную log_var для совместимости
        return mu, noise_seed, torch.zeros_like(mu)