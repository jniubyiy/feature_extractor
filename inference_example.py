# inference_example.py
"""
Демонстрация использования экспортированных моделей (TorchScript).
Файлы моделей: *_inference.pt — самодостаточные, не требуют импорта архитектуры.

Каждая модель загружается функцией load_inference_model() и может использоваться
как обычный nn.Module. Ниже приведено описание входов/выходов для всех моделей и примеры.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_inference_model(path: str) -> torch.jit.ScriptModule:
    """
    Загружает модель из TorchScript файла.
    Возвращает модуль, готовый к инференсу (eval mode).
    """
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────
# Описание данных для каждой модели
# ──────────────────────────────────────────────────────────────────────

# ------------------------------------------------------------
# Encoder (энкодер основного автоэнкодера)
# ------------------------------------------------------------
#   Вход:  [B, 3, H, W] — изображение RGB, нормализованное в [-1, 1]
#   Выход: [B, 3, H, W] — парнет первого уровня (не сжатый),
#          тоже в диапазоне [-1, 1]
#
#   Пример использования: получение парнета из изображения
#   (затем парнет можно сжимать, передавать и т.д.)

# ------------------------------------------------------------
# Decoder (декодер основного автоэнкодера)
# ------------------------------------------------------------
#   Вход:  [B, 3, H, W] — парнет в [-1, 1]
#   Выход: [B, 3, H, W] — восстановленное изображение RGB в [-1, 1]
#
#   Пример использования: превращение парнета обратно в изображение

# ------------------------------------------------------------
# Compressor (первый уровень сжатия парнета)
# ------------------------------------------------------------
#   Вход:  [B, 3, H, W] — парнет (3 канала, полный размер)
#   Выход: [B, 4, H/2, W/2] — сжатый парнет, 4 канала, половинный размер,
#          значения в [-1, 1]
#
#   Пример использования: уменьшение размера парнета перед хранением/передачей

# ------------------------------------------------------------
# Decompressor (первый уровень разжатия)
# ------------------------------------------------------------
#   Вход:  [B, 4, H/2, W/2] — сжатый парнет (4 канала)
#   Выход: [B, 3, H, W] — восстановленный парнет (3 канала, полный размер)
#
#   Пример использования: восстановление парнета после сжатия

# ------------------------------------------------------------
# Compressor Level 2 (второй уровень сжатия)
# ------------------------------------------------------------
#   Вход:  [B, 4, H/2, W/2] — уже сжатый парнет первого уровня
#   Выход: [B, 5, H/4, W/4] — ещё более сжатый парнет, 5 каналов, четверть размера
#
#   Пример использования: дальнейшее сжатие, если нужен более компактный код

# ------------------------------------------------------------
# Decompressor Level 2 (второй уровень разжатия)
# ------------------------------------------------------------
#   Вход:  [B, 5, H/4, W/4] — сжатый парнет второго уровня
#   Выход: [B, 4, H/2, W/2] — восстановленный парнет первого уровня (4 канала)
#
#   Пример использования: частичное разжатие для последующего полного восстановления


# ──────────────────────────────────────────────────────────────────────
# Вспомогательные функции для преобразования изображений/тензоров
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

    # Загружаем модели (укажите актуальные пути)
    encoder = load_inference_model("./models/encoder_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")
    compressor1 = load_inference_model("./models_compressor/compressor_inference.pt")
    decompressor1 = load_inference_model("./models_compressor/decompressor_inference.pt")
    # Второй уровень (раскомментируйте при наличии)
    # compressor2 = load_inference_model("./models_compressor_level2/compressor_level2_inference.pt")
    # decompressor2 = load_inference_model("./models_compressor_level2/decompressor_level2_inference.pt")

    # Подготовим тестовое изображение (чёрный квадрат или загрузите свой файл)
    test_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    x = image_to_tensor(test_img)   # [1,3,512,512] в [-1,1]

    with torch.no_grad():
        # --- Только энкодер ---
        parnet = encoder(x)                 # [1,3,512,512]
        print(f"Encoder input:  {tuple(x.shape)}")
        print(f"Encoder output: {tuple(parnet.shape)}")

        # --- Только декодер (подаём любой парнет) ---
        rec_img = decoder(parnet)           # [1,3,512,512]
        print(f"Decoder input:  {tuple(parnet.shape)}")
        print(f"Decoder output: {tuple(rec_img.shape)}")

        # --- Компрессор первого уровня ---
        comp1 = compressor1(parnet)         # [1,4,256,256]
        print(f"Compressor1 input:  {tuple(parnet.shape)}")
        print(f"Compressor1 output: {tuple(comp1.shape)}")

        # --- Декомпрессор первого уровня ---
        decomp1 = decompressor1(comp1)      # [1,3,512,512]
        print(f"Decompressor1 input:  {tuple(comp1.shape)}")
        print(f"Decompressor1 output: {tuple(decomp1.shape)}")

        # --- Второй уровень (если доступен) ---
        # comp2 = compressor2(comp1)         # [1,5,128,128]
        # decomp2 = decompressor2(comp2)      # [1,4,256,256]
        # print(f"Compressor2 input:  {tuple(comp1.shape)}")
        # print(f"Compressor2 output: {tuple(comp2.shape)}")
        # print(f"Decompressor2 input:  {tuple(comp2.shape)}")
        # print(f"Decompressor2 output: {tuple(decomp2.shape)}")

    # Сохраняем результат декодирования для проверки
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
    # Второй уровень (опционально)
    # compressor2 = load_inference_model("./models_compressor_level2/compressor_level2_inference.pt")
    # decompressor2 = load_inference_model("./models_compressor_level2/decompressor_level2_inference.pt")

    # Загружаем реальное изображение или создаём тестовое
    try:
        img = Image.open("test.jpg").convert("RGB")
    except FileNotFoundError:
        print("test.jpg не найден, используется серый квадрат.")
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))

    x = image_to_tensor(img)

    with torch.no_grad():
        # Прямой проход
        parnet = encoder(x)                       # парнет, 3 канала, полный размер
        comp1 = compressor1(parnet)               # сжатый парнет, 4 канала, половина
        # comp2 = compressor2(comp1)              # ещё более сжатый (опционально)

        # Обратный проход
        # decomp1_partial = decompressor2(comp2)  # восстановление до 4 каналов
        rest_parnet = decompressor1(comp1)        # восстановление до 3 каналов, полный размер
        reconstructed = decoder(rest_parnet)      # итоговое изображение

    # Сохраняем
    tensor_to_image(reconstructed).save("pipeline_reconstructed.jpg")
    print("Результат сохранён в pipeline_reconstructed.jpg")

    # Дополнительно можно сравнить размеры сжатых представлений
    print("\nРазмеры тензоров:")
    print(f"  Исходное изображение:  {tuple(x.shape)}")
    print(f"  Парнет (3 канала):      {tuple(parnet.shape)}")
    print(f"  Сжатый уровень 1:       {tuple(comp1.shape)}")
    # print(f"  Сжатый уровень 2:       {tuple(comp2.shape)}")
    print(f"  Восстановленное изобр.:  {tuple(reconstructed.shape)}")


# ──────────────────────────────────────────────────────────────────────
# 3. Использование моделей в других "пакетах" (просто вызов по отдельности)
#    Показано, как можно строить произвольные графы обработки.
# ──────────────────────────────────────────────────────────────────────
def demo_modular_usage():
    print("\n=== Модульное использование моделей ===")

    # Здесь демонстрируется, что каждую модель можно применять
    # независимо и комбинировать в любом порядке.
    # Например, можно получить сжатый парнет, а затем сразу его передать
    # в декомпрессор, минуя энкодер, если парнет уже есть.

    encoder = load_inference_model("./models/encoder_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")
    compressor1 = load_inference_model("./models_compressor/compressor_inference.pt")
    decompressor1 = load_inference_model("./models_compressor/decompressor_inference.pt")

    # Пример: имеется готовый парнет (например, загруженный из файла)
    # и мы хотим его сжать и восстановить без участия изображений.
    dummy_parnet = torch.randn(1, 3, 512, 512, device=DEVICE).clamp(-1, 1)
    with torch.no_grad():
        comp = compressor1(dummy_parnet)
        restored_parnet = decompressor1(comp)
    print(f"Парнет -> сжатие -> восстановление парнета: форма {restored_parnet.shape}")

    # Если есть файл с сохранённым парнетом (например, из prepared_dataset_parnet)
    # можно загрузить его и передать в декодер для визуализации.
    # loaded = torch.load("prepared_dataset_parnet/0.pt")  # содержит ключ 'parnet'
    # parnet_from_disk = loaded['parnet'].unsqueeze(0).to(DEVICE)
    # img = decoder(parnet_from_disk)
    # tensor_to_image(img).save("from_saved_parnet.jpg")


# ──────────────────────────────────────────────────────────────────────
# Главная точка входа
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_individual_models()
    demo_full_pipeline()
    demo_modular_usage()