# model_Autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        r = x
        x = self.act(self.conv1(x))
        x = self.dropout(x)         # dropout после активации
        x = self.conv2(x)
        return self.act(x + r)

class Encoder(nn.Module):
    """Изображение [B,3,W,H] -> парнет [B,3,W,H] (без ограничения диапазона, без нулей)."""
    def __init__(self, base_dim=64, num_blocks=3, parnet_channels=3, dropout_rate=0.1):
        super().__init__()
        self.init_conv = nn.Conv2d(3, base_dim, 3, padding=1)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(base_dim, dropout_rate) for _ in range(num_blocks)
        ])
        # Убрали Tanh – теперь выход не ограничен
        self.to_parnet = nn.Conv2d(base_dim, parnet_channels, 3, padding=1)

    def forward(self, image):
        x = F.relu(self.init_conv(image))
        x = self.res_blocks(x)
        parnet = self.to_parnet(x)
        # Гарантируем, что ни одно значение не равно 0 (прибавляем эпсилон)
        parnet = parnet + 1e-8
        return parnet

class Decoder(nn.Module):
    """Парнет [B,3,W,H] (любой диапазон) -> изображение [B,3,W,H] в [-1,1]."""
    def __init__(self, base_dim=64, num_blocks=3, parnet_channels=3, dropout_rate=0.1):
        super().__init__()
        self.init_conv = nn.Conv2d(parnet_channels, base_dim, 3, padding=1)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(base_dim, dropout_rate) for _ in range(num_blocks)
        ])
        # Выход декодера остаётся в [-1,1] благодаря Tanh
        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_dim, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, parnet):
        x = F.relu(self.init_conv(parnet))
        x = self.res_blocks(x)
        img = self.to_rgb(x)
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