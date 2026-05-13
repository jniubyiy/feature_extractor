# model_Decoder.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_sinusoidal_position_embeddings(num_positions: int, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Возвращает синусоидальные позиционные эмбеддинги для заданного количества позиций.
    Форма: (1, num_positions, d_model).
    """
    position = torch.arange(num_positions, dtype=torch.float, device=device).unsqueeze(1)  # (N, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(1, num_positions, d_model, device=device)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerDecoderBlock(nn.Module):
    """
    Блок трансформера с AdaLN (обусловлен pooled-вектором) и Pre‑LN.
    key_padding_mask позволяет игнорировать паддинговые патчи.
    """
    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int = 8,
                 ff_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
            nn.Dropout(dropout)
        )
        self.adaln_attn = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_dim))
        self.adaln_ff = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift, scale = self.adaln_attn(cond).chunk(2, dim=-1)
        norm_x = F.layer_norm(x, x.shape[-1:])
        norm_x = norm_x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
        x = x + attn_out

        shift, scale = self.adaln_ff(cond).chunk(2, dim=-1)
        norm_x = F.layer_norm(x, x.shape[-1:])
        norm_x = norm_x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        ff_out = self.ff(norm_x)
        x = x + ff_out
        return x


class Decoder(nn.Module):
    """
    Генератор изображения из pooled‑вектора и маски паддинга.
    Принимает:
        pooled : (B, hidden_dim)
        padding_mask : (B, 1, H, W) или (B, H, W), значения 0/1 (0 = паддинг, 1 = реальный пиксель)
    Возвращает изображение (B, 3, H, W), где пиксели паддинга занулены.
    """
    def __init__(self,
                 hidden_dim: int = 1024,
                 cond_dim: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_multiplier: int = 4,
                 dropout: float = 0.1,
                 patch_size: int = 32,
                 out_channels: int = 3,
                 use_checkpoint: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.use_checkpoint = use_checkpoint

        self.pooled_to_cond = nn.Linear(hidden_dim, cond_dim) if hidden_dim != cond_dim else nn.Identity()

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(hidden_dim, cond_dim, num_heads, ff_multiplier, dropout)
            for _ in range(num_layers)
        ])

        self.patch_proj = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)

    def forward(self, pooled: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if padding_mask.dim() == 3:
            padding_mask = padding_mask.unsqueeze(1)

        B, _, H, W = padding_mask.shape
        device = pooled.device

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Разрешение ({H}, {W}) должно быть кратно patch_size={self.patch_size}")

        Hp, Wp = H // self.patch_size, W // self.patch_size
        N = Hp * Wp

        mask_patches = F.avg_pool2d(padding_mask.float(), kernel_size=self.patch_size, stride=self.patch_size)
        mask_patches = mask_patches.flatten(2).squeeze(1)  # (B, N)
        mask_patches = mask_patches > 0.5

        cond = self.pooled_to_cond(pooled)  # (B, cond_dim)

        pos_embed = get_sinusoidal_position_embeddings(N, self.hidden_dim, device)
        x = pos_embed.expand(B, -1, -1)  # (B, N, D)

        key_padding_mask = ~mask_patches

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    lambda _x, _cond, _mask: layer(_x, _cond, key_padding_mask=_mask),
                    x, cond, key_padding_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, cond, key_padding_mask=key_padding_mask)

        patches = self.patch_proj(x)  # (B, N, patch_size*patch_size*3)
        patches = patches.reshape(B, Hp, Wp, self.patch_size, self.patch_size, self.out_channels)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = patches.reshape(B, self.out_channels, H, W)

        img = img * padding_mask
        return img