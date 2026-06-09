# inference_example.py
"""
Демонстрация использования всех экспортированных моделей (TorchScript).

Все модели хранятся в подпапках:
  ./models/                  – Encoder, Decoder основного автоэнкодера
  ./models_compressor/       – ParnetCompressor, ParnetDecompressor
  ./models_vae_wrapper/      – StochasticEncoder, StochasticDecoder (VAE)

Каждая модель загружается как автономный TorchScript-модуль.
Входы и выходы описаны перед каждым примером.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# Общие вспомогательные функции
# ----------------------------------------------------------------------
def load_inference_model(path: str) -> torch.jit.ScriptModule:
    """Загружает TorchScript-модель из указанного пути."""
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model

def image_to_tensor(pil_image: Image.Image, size: int = 512) -> torch.Tensor:
    """Преобразует PIL RGB-изображение в тензор [1, 3, size, size] в диапазоне [-1, 1]."""
    img = pil_image.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return t.to(DEVICE)

def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """Преобразует тензор [1, 3, H, W] в диапазоне [-1, 1] в PIL Image."""
    arr = t.squeeze(0).clamp(-1, 1).cpu()
    arr = ((arr + 1) / 2).permute(1, 2, 0).numpy() * 255
    return Image.fromarray(arr.astype(np.uint8))

# ======================================================================
# Детальное описание каждой модели
# ======================================================================

# ====================== 1. Encoder (основной) ==========================
"""
Модель: Encoder
Файл: ./models/encoder_inference.pt
Назначение: преобразует RGB-изображение в парнет – промежуточное представление,
            сохраняющее информацию об изображении в 3 каналах без потери разрешения.
Вход:  [1, 3, H, W] – изображение в диапазоне [-1, 1]
Выход: [1, 3, H, W] – парнет (значения не ограничены, гарантированно без нулей)
"""
def demo_encoder():
    print("=== 1. Encoder (изображение -> парнет) ===")
    encoder = load_inference_model("./models/encoder_inference.pt")
    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    x = image_to_tensor(img)
    with torch.no_grad():
        parnet = encoder(x)
    print(f"Вход:  {tuple(x.shape)}")
    print(f"Выход: {tuple(parnet.shape)}")
    print(f"Диапазон парнета: [{parnet.min().item():.3f}, {parnet.max().item():.3f}]")
    print(f"Нулей в парнете: {(parnet == 0).sum().item()}\n")

# ====================== 2. Decoder (основной) ==========================
"""
Модель: Decoder
Файл: ./models/decoder_inference.pt
Назначение: восстанавливает RGB-изображение из парнета.
Вход:  [1, 3, H, W] – парнет (любые значения)
Выход: [1, 3, H, W] – восстановленное изображение в [-1, 1]
"""
def demo_decoder():
    print("=== 2. Decoder (парнет -> изображение) ===")
    decoder = load_inference_model("./models/decoder_inference.pt")
    dummy_parnet = torch.randn(1, 3, 512, 512, device=DEVICE)
    with torch.no_grad():
        rec = decoder(dummy_parnet)
    print(f"Вход:  {tuple(dummy_parnet.shape)}")
    print(f"Выход: {tuple(rec.shape)}")
    tensor_to_image(rec).save("demo_decoder_random_parnet.jpg")
    print("Сохранено demo_decoder_random_parnet.jpg\n")

# ====================== 3. ParnetCompressor ============================
"""
Модель: ParnetCompressor
Файл: ./models_compressor/compressor_inference.pt
Назначение: сжимает парнет в 2 раза по высоте и ширине, увеличивая число каналов до 4.
            Это первый уровень сжатия представления.
Вход:  [1, 3, H, W]   – парнет (любые значения)
Выход: [1, 4, H/2, W/2] – сжатый парнет (значения любые, без нулей)
"""
def demo_compressor():
    print("=== 3. ParnetCompressor (парнет -> сжатый парнет) ===")
    compressor = load_inference_model("./models_compressor/compressor_inference.pt")
    encoder = load_inference_model("./models/encoder_inference.pt")
    img = Image.new("RGB", (512, 512), color=(80, 120, 200))
    x = image_to_tensor(img)
    with torch.no_grad():
        parnet = encoder(x)
        compressed = compressor(parnet)
    print(f"Вход:  {tuple(parnet.shape)}")
    print(f"Выход: {tuple(compressed.shape)}")
    print(f"Диапазон сжатого парнета: [{compressed.min().item():.3f}, {compressed.max().item():.3f}]")
    print(f"Нулей: {(compressed == 0).sum().item()}\n")

# ====================== 4. ParnetDecompressor ==========================
"""
Модель: ParnetDecompressor
Файл: ./models_compressor/decompressor_inference.pt
Назначение: восстанавливает парнет из сжатого представления (обратная операция к Compressor).
Вход:  [1, 4, H/2, W/2] – сжатый парнет
Выход: [1, 3, H, W]     – восстановленный парнет
"""
def demo_decompressor():
    print("=== 4. ParnetDecompressor (сжатый парнет -> парнет) ===")
    decompressor = load_inference_model("./models_compressor/decompressor_inference.pt")
    compressor = load_inference_model("./models_compressor/compressor_inference.pt")
    encoder = load_inference_model("./models/encoder_inference.pt")
    img = Image.new("RGB", (512, 512), color=(50, 200, 100))
    x = image_to_tensor(img)
    with torch.no_grad():
        parnet = encoder(x)
        comp = compressor(parnet)
        restored = decompressor(comp)
    print(f"Вход:  {tuple(comp.shape)}")
    print(f"Выход: {tuple(restored.shape)}")
    print(f"Диапазон восстановленного парнета: [{restored.min().item():.3f}, {restored.max().item():.3f}]\n")

# ====================== 5. StochasticEncoder ===========================
"""
Модель: StochasticEncoder (с фиксированными параметрами шума)
Файл: ./models_vae_wrapper/encoder_inference.pt
Назначение: принимает сжатый парнет (4 канала) и генерирует латентный вектор z
            (mu + равномерный шум из [-NOISE_RANGE, NOISE_RANGE], масштабированный адаптивно)
            и опорный шум noise_seed.
            Параметры шума зафиксированы: NOISE_RANGE и STOCHASTIC_STRENGTH вшиты в модель.
Вход:  [1, 4, H/2, W/2] – сжатый парнет
Выход: z:         [1, 4, H/2, W/2] – стохастическое представление
       noise_seed:[1, 4, H/2, W/2] – опорный шум для декодера
"""
def demo_stochastic_encoder():
    print("=== 5. StochasticEncoder (сжатый парнет -> z + noise_seed) ===")
    stoch_enc = load_inference_model("./models_vae_wrapper/encoder_inference.pt")
    compressor = load_inference_model("./models_compressor/compressor_inference.pt")
    encoder = load_inference_model("./models/encoder_inference.pt")
    img = Image.new("RGB", (512, 512), color=(200, 100, 50))
    x = image_to_tensor(img)
    with torch.no_grad():
        parnet = encoder(x)
        comp = compressor(parnet)
        z, noise_seed = stoch_enc(comp)
    print(f"Вход:       {tuple(comp.shape)}")
    print(f"Выход z:    {tuple(z.shape)}")
    print(f"Выход seed: {tuple(noise_seed.shape)}")
    print(f"z диапазон: [{z.min().item():.3f}, {z.max().item():.3f}]\n")

# ====================== 6. StochasticDecoder ===========================
"""
Модель: StochasticDecoder
Файл: ./models_vae_wrapper/decoder_inference.pt
Назначение: принимает конкатенацию [z, noise_seed] (2*4 каналов) и восстанавливает
            сжатый парнет (4 канала). Выход не ограничен по диапазону.
Вход:  [1, 8, H/2, W/2] – объединённые z и noise_seed (оба по 4 канала)
Выход: [1, 4, H/2, W/2] – декодированный сжатый парнет
"""
def demo_stochastic_decoder():
    print("=== 6. StochasticDecoder (z+noise_seed -> сжатый парнет) ===")
    stoch_dec = load_inference_model("./models_vae_wrapper/decoder_inference.pt")
    stoch_enc = load_inference_model("./models_vae_wrapper/encoder_inference.pt")
    compressor = load_inference_model("./models_compressor/compressor_inference.pt")
    encoder = load_inference_model("./models/encoder_inference.pt")
    img = Image.new("RGB", (512, 512), color=(150, 70, 200))
    x = image_to_tensor(img)
    with torch.no_grad():
        parnet = encoder(x)
        comp = compressor(parnet)
        z, noise_seed = stoch_enc(comp)
        combined = torch.cat([z, noise_seed], dim=1)
        decoded_comp = stoch_dec(combined)
    print(f"Вход:  {tuple(combined.shape)}")
    print(f"Выход: {tuple(decoded_comp.shape)}")
    print(f"Диапазон декодированного: [{decoded_comp.min().item():.3f}, {decoded_comp.max().item():.3f}]\n")

# ====================== 7. Сквозной VAE пайплайн =======================
"""
Полный VAE-пайплайн: изображение -> парнет -> сжатый парнет -> z + noise_seed ->
декодированный сжатый парнет -> восстановленный парнет -> изображение.
"""
def demo_full_vae_pipeline():
    print("=== 7. Полный VAE-пайплайн (изображение -> VAE -> изображение) ===")
    encoder = load_inference_model("./models/encoder_inference.pt")
    compressor = load_inference_model("./models_compressor/compressor_inference.pt")
    stoch_enc = load_inference_model("./models_vae_wrapper/encoder_inference.pt")
    stoch_dec = load_inference_model("./models_vae_wrapper/decoder_inference.pt")
    decompressor = load_inference_model("./models_compressor/decompressor_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")

    try:
        img = Image.open("test.jpg").convert("RGB")
    except FileNotFoundError:
        img = Image.new("RGB", (512, 512), color=(70, 130, 180))
    x = image_to_tensor(img)

    with torch.no_grad():
        parnet = encoder(x)                             # [1,3,512,512]
        comp = compressor(parnet)                       # [1,4,256,256]
        z, noise_seed = stoch_enc(comp)                 # каждый [1,4,256,256]
        combined = torch.cat([z, noise_seed], dim=1)    # [1,8,256,256]
        decoded_comp = stoch_dec(combined)              # [1,4,256,256]
        rest_parnet = decompressor(decoded_comp)        # [1,3,512,512]
        reconstructed = decoder(rest_parnet)            # [1,3,512,512]

    tensor_to_image(reconstructed).save("pipeline_vae_reconstructed.jpg")
    print("Результат сохранён в pipeline_vae_reconstructed.jpg\n")

# ======================================================================
# Запуск всех демонстраций
# ======================================================================
if __name__ == "__main__":
    demo_encoder()
    demo_decoder()
    demo_compressor()
    demo_decompressor()
    demo_stochastic_encoder()
    demo_stochastic_decoder()
    demo_full_vae_pipeline()