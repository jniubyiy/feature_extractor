# inference_example.py
"""
Демонстрация использования экспортированных моделей (TorchScript).

Файлы моделей: *_inference.pt — самодостаточные, не требуют импорта архитектуры.
Все модели обновлены: парнет не ограничен по диапазону и не содержит нулей.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_inference_model(path: str) -> torch.jit.ScriptModule:
    """ Загружает модель из TorchScript файла.
    Возвращает модуль, готовый к инференсу (eval mode). """
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model

# ──────────────────────────────────────────────────────────────────────
# Описание данных для каждой модели
# ──────────────────────────────────────────────────────────────────────
# Encoder:
#   Вход:  [B, 3, H, W] — изображение RGB в [-1, 1]
#   Выход: [B, 3, H, W] — парнет, значения любые, гарантированно без нулей.
#
# Decoder:
#   Вход:  [B, 3, H, W] — парнет с любыми значениями
#   Выход: [B, 3, H, W] — восстановленное изображение RGB в [-1, 1]
#
# Compressor (уровень 1):
#   Вход:  [B, 3, H, W] — парнет (любой диапазон)
#   Выход: [B, 4, H/2, W/2] — сжатый парнет, значения любые, без нулей.
#
# Decompressor (уровень 1):
#   Вход:  [B, 4, H/2, W/2] — сжатый парнет
#   Выход: [B, 3, H, W] — восстановленный парнет, без нулей.
#
# Compressor Level 2 / Decompressor Level 2 — аналогично (пока не обновлены).
# ──────────────────────────────────────────────────────────────────────

def image_to_tensor(pil_image: Image.Image, size: int = 512) -> torch.Tensor:
    """Преобразует PIL RGB изображение в тензор [1,3,size,size] в диапазоне [-1,1]."""
    img = pil_image.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return t.to(DEVICE)

def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """Преобразует тензор [1,3,H,W] в диапазоне [-1,1] в PIL Image."""
    arr = t.squeeze(0).clamp(-1, 1).cpu()
    arr = ((arr + 1) / 2).permute(1, 2, 0).numpy() * 255
    return Image.fromarray(arr.astype(np.uint8))

# ──────────────────────────────────────────────────────────────────────
# 1. Примеры использования каждой модели по отдельности
# ──────────────────────────────────────────────────────────────────────
def demo_individual_models():
    print("=== Демонстрация отдельных моделей ===")

    encoder = load_inference_model("./models/encoder_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")
    compressor1 = load_inference_model("./models_compressor/compressor_inference.pt")
    decompressor1 = load_inference_model("./models_compressor/decompressor_inference.pt")
    # Второй уровень при необходимости

    test_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    x = image_to_tensor(test_img)   # [1,3,512,512]

    with torch.no_grad():
        # Энкодер
        parnet = encoder(x)
        print(f"Encoder: in {tuple(x.shape)} -> out {tuple(parnet.shape)}")
        print(f"  parnet min: {parnet.min().item():.4f}, max: {parnet.max().item():.4f}")
        zero_count = (parnet == 0).sum().item()
        print(f"  zeros: {zero_count}")

        # Декодер
        rec_img = decoder(parnet)
        print(f"Decoder: in {tuple(parnet.shape)} -> out {tuple(rec_img.shape)}")

        # Компрессор 1
        comp1 = compressor1(parnet)
        print(f"Compressor1: in {tuple(parnet.shape)} -> out {tuple(comp1.shape)}")
        print(f"  comp1 min: {comp1.min().item():.4f}, max: {comp1.max().item():.4f}, zeros: {(comp1==0).sum().item()}")

        # Декомпрессор 1
        decomp1 = decompressor1(comp1)
        print(f"Decompressor1: in {tuple(comp1.shape)} -> out {tuple(decomp1.shape)}")
        print(f"  decomp1 min: {decomp1.min().item():.4f}, max: {decomp1.max().item():.4f}, zeros: {(decomp1==0).sum().item()}")

    tensor_to_image(rec_img).save("demo_individual_decoder_output.jpg")
    print("Сохранено demo_individual_decoder_output.jpg\n")

# ──────────────────────────────────────────────────────────────────────
# 2. Полный пайплайн сжатия и восстановления изображения
# ──────────────────────────────────────────────────────────────────────
def demo_full_pipeline():
    print("=== Полный пайплайн изображение → парнет → сжатие → восстановление ===")

    encoder = load_inference_model("./models/encoder_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")
    compressor1 = load_inference_model("./models_compressor/compressor_inference.pt")
    decompressor1 = load_inference_model("./models_compressor/decompressor_inference.pt")

    try:
        img = Image.open("test.jpg").convert("RGB")
    except FileNotFoundError:
        print("test.jpg не найден, используется серый квадрат.")
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))

    x = image_to_tensor(img)

    with torch.no_grad():
        parnet = encoder(x)                     # [1,3,512,512]
        comp1 = compressor1(parnet)             # [1,4,256,256]
        rest_parnet = decompressor1(comp1)      # [1,3,512,512]
        reconstructed = decoder(rest_parnet)    # [1,3,512,512]

    tensor_to_image(reconstructed).save("pipeline_reconstructed.jpg")
    print("Результат сохранён в pipeline_reconstructed.jpg")

    print("\nРазмеры тензоров:")
    print(f"  Исходное изображение: {tuple(x.shape)}")
    print(f"  Парнет:               {tuple(parnet.shape)}")
    print(f"  Сжатый уровень 1:     {tuple(comp1.shape)}")
    print(f"  Восстановленный парнет:{tuple(rest_parnet.shape)}")
    print(f"  Восстановленное изобр.:{tuple(reconstructed.shape)}")

# ──────────────────────────────────────────────────────────────────────
# 3. Модульное использование
# ──────────────────────────────────────────────────────────────────────
def demo_modular_usage():
    print("\n=== Модульное использование моделей ===")
    encoder = load_inference_model("./models/encoder_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")
    compressor1 = load_inference_model("./models_compressor/compressor_inference.pt")
    decompressor1 = load_inference_model("./models_compressor/decompressor_inference.pt")

    # Работа с готовым парнетом из файла
    # loaded = torch.load("prepared_dataset_parnet/0.pt")
    # parnet_from_disk = loaded['parnet'].unsqueeze(0).to(DEVICE)
    # img = decoder(parnet_from_disk)
    # tensor_to_image(img).save("from_saved_parnet.jpg")

    # Генерация случайного парнета с большим разбросом
    dummy_parnet = torch.randn(1, 3, 512, 512, device=DEVICE) * 5
    print(f"Случайный парнет: min={dummy_parnet.min().item():.2f}, max={dummy_parnet.max().item():.2f}")
    img = decoder(dummy_parnet)
    print(f"Декодированное изображение: форма {img.shape}, min={img.min().item():.2f}, max={img.max().item():.2f}")

if __name__ == "__main__":
    demo_individual_models()
    demo_full_pipeline()
    demo_modular_usage()