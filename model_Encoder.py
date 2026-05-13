# model_Encoder.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def get_sinusoidal_position_embeddings(num_patches: int, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Генерирует синусоидальные позиционные эмбеддинги для любого количества патчей.
    Возвращает тензор формы (1, num_patches, d_model).
    """
    position = torch.arange(num_patches, dtype=torch.float, device=device).unsqueeze(1)  # (N, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, num_patches, d_model, device=device)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEmbed(nn.Module):
    """
    Разбивает изображение на патчи и линейно проецирует их в эмбеддинги.
    Возвращает:
        x : (B, N, D)  — эмбеддинги патчей
        mask_patches : (B, N)  — bool-маска валидных патчей (True = реальный патч, False = паддинг)
    """
    def __init__(self, patch_size: int = 32, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if padding_mask.dim() == 3:
            padding_mask = padding_mask.unsqueeze(1)  # (B, 1, H, W)

        x = self.proj(image)  # (B, D, H', W')
        B, D, Hp, Wp = x.shape
        N = Hp * Wp
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        mask = F.avg_pool2d(padding_mask.float(), kernel_size=self.patch_size, stride=self.patch_size)  # (B, 1, H', W')
        mask = mask.flatten(2).squeeze(1)  # (B, N)
        mask_patches = mask > 0.5  # Патч считается валидным, если внутри больше половины реальных пикселей
        return x, mask_patches


class AttentionPooling(nn.Module):
    """Обучаемый attention‑pooling: один запрос собирает информацию со всех позиций."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        key_padding_mask = ~mask if mask is not None else None
        pooled, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)


class TransformerBlock(nn.Module):
    """
    Блок трансформера с опциональным AdaLN (если передан cond) или обычным Pre‑LN.
    key_padding_mask позволяет игнорировать паддинговые позиции.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, ff_multiplier: int = 4, dropout: float = 0.1,
                 use_adaln: bool = False, cond_dim: Optional[int] = None):
        super().__init__()
        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln1 = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * hidden_dim)
            )
            self.adaln2 = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * hidden_dim)
            )
        else:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
            nn.Dropout(dropout)
        )

    def _norm(self, x: torch.Tensor, cond: Optional[torch.Tensor], adaln_layer: Optional[nn.Module] = None) -> torch.Tensor:
        if self.use_adaln and cond is not None:
            assert adaln_layer is not None
            shift, scale = adaln_layer(cond).chunk(2, dim=-1)
            x = F.layer_norm(x, x.shape[-1:])
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            return self.norm1(x) if adaln_layer is self.adaln1 else self.norm2(x)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_adaln:
            norm_x = self._norm(x, cond, self.adaln1)
        else:
            norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
        x = x + attn_out

        if self.use_adaln:
            norm_x = self._norm(x, cond, self.adaln2)
        else:
            norm_x = self.norm2(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        return x


class Encoder(nn.Module):
    """
    Принимает изображение (B, 3, H, W) и бинарную маску паддинга (B, 1, H, W) [0=паддинг, 1=изображение].
    Возвращает один вектор (B, hidden_dim) — агрегированное представление, игнорируя паддинг.
    """
    def __init__(self,
                 hidden_dim: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_multiplier: int = 4,
                 dropout: float = 0.1,
                 patch_size: int = 32,
                 in_channels: int = 3,
                 use_adaln: bool = False,
                 cond_dim: Optional[int] = None,
                 use_checkpoint: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.use_adaln = use_adaln
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_multiplier, dropout,
                             use_adaln=use_adaln, cond_dim=cond_dim)
            for _ in range(num_layers)
        ])

        self.pooling = AttentionPooling(hidden_dim)

    def forward(self,
                image: torch.Tensor,
                padding_mask: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, mask_patches = self.patch_embed(image, padding_mask)  # (B, N, D), (B, N) bool
        device = x.device

        pos_embed = get_sinusoidal_position_embeddings(x.size(1), self.hidden_dim, device)
        x = x + pos_embed

        key_padding_mask = ~mask_patches

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    lambda _x, _cond, _mask: layer(_x, key_padding_mask=_mask, cond=_cond),
                    x, cond, key_padding_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, key_padding_mask=key_padding_mask, cond=cond)

        pooled = self.pooling(x, mask_patches)  # (B, hidden_dim)
        return pooled