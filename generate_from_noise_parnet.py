# generate_from_noise_parnet.py
"""
Генерация изображений из случайных шумовых парнетов с помощью Decoder.
Поддерживает два режима:
  1) Обычный: полный парнет [3, H, W] → Decoder → изображение.
  2) Сжатый: сжатый парнет [compressed_channels, H/2, W/2] → Decompressor → парнет → Decoder → изображение.

Создаёт NUM_SAMPLES случайных тензоров, пропускает через цепочку и сохраняет в OUTPUT_DIR.
"""
import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image

from model_Autoencoder import Decoder
from model_ParnetCompressor import ParnetDecompressor
from config_training_models_Encoder_Decoder import DECODER_CONFIG, IMAGE_SIZE
from config_training_models_Compressor_Decompressor import DECOMPRESSOR_CONFIG, COMPRESSOR_CONFIG

# ------------------------ НАСТРОЙКИ ------------------------
NUM_SAMPLES = 10
OUTPUT_DIR = "generated_images"

# Путь к чекпоинту декодера (обязательно)
DECODER_CHECKPOINT = os.path.join("models", "decoder_epoch46.pth")

# Путь к чекпоинту декомпрессора (если не None, используется сжатый режим)
DECOMPRESSOR_CHECKPOINT = os.path.join("models_compressor", "decompressor_epoch82.pth")
# Если DECOMPRESSOR_CHECKPOINT = None, то генерируются полные парнеты (3, H, W)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Создаём выходную папку
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_decoder(checkpoint_path: str) -> Decoder:
    model = Decoder(**DECODER_CONFIG).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Decoder loaded from {checkpoint_path}")
    return model


def load_decompressor(checkpoint_path: str) -> ParnetDecompressor:
    model = ParnetDecompressor(**DECOMPRESSOR_CONFIG).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Decompressor loaded from {checkpoint_path}")
    return model


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Преобразует тензор [1,3,H,W] (в диапазоне [-1,1]) в PIL Image."""
    arr = tensor.squeeze(0).clamp(-1, 1).cpu().detach()
    arr = ((arr + 1) / 2).permute(1, 2, 0).numpy() * 255
    return Image.fromarray(arr.astype(np.uint8))


def main():
    # Загружаем декодер (всегда нужен)
    decoder = load_decoder(DECODER_CHECKPOINT)

    # Определяем режим: сжатый или полный
    use_compressed = DECOMPRESSOR_CHECKPOINT is not None

    if use_compressed:
        decompressor = load_decompressor(DECOMPRESSOR_CHECKPOINT)
        # Размеры сжатого парнета
        compressed_channels = COMPRESSOR_CONFIG["compressed_channels"]  # обычно 4
        spatial_size = IMAGE_SIZE // 2  # H/2, W/2
        print(f"Compressed mode: generating random parnets of shape [1, {compressed_channels}, {spatial_size}, {spatial_size}]")
    else:
        decompressor = None
        print(f"Full parnet mode: generating random parnets of shape [1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")

    print(f"Generating {NUM_SAMPLES} samples...")
    with torch.no_grad():
        for i in range(NUM_SAMPLES):
            if use_compressed:
                # Создаём сжатый шумовой парнет
                compressed = torch.randn(1, compressed_channels, spatial_size, spatial_size, device=DEVICE) * 5.0
                # Разжимаем до полного парнета
                full_parnet = decompressor(compressed)  # [1, 3, H, W]
            else:
                # Создаём полный шумовой парнет напрямую
                full_parnet = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 5.0

            # Декодируем в изображение
            image_tensor = decoder(full_parnet)  # [1, 3, H, W] в [-1, 1]

            img = tensor_to_image(image_tensor)
            img.save(os.path.join(OUTPUT_DIR, f"gen_{i+1:02d}.png"))
            print(f"Saved gen_{i+1:02d}.png")

    print(f"Done. Images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()