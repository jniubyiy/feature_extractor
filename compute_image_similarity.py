# compute_image_similarity.py
"""
Вычисляет попарную схожесть изображений из prepared_dataset,
комбинируя MSE и SSIM, и сохраняет для каждого изображения
список ближайших соседей в файл similarities.pt.
"""

import torch
import torch.nn.functional as F
import os
from pathlib import Path
import math

# ---------------------- Конфигурация ----------------------
DATASET_DIR = "./prepared_dataset"
OUTPUT_PATH = os.path.join(DATASET_DIR, "similarities.pt")
TOP_K = 10               # сколько соседей сохранить для каждого изображения
TEMPERATURE = 0.1        # температура для преобразования расстояния в сходство

# Параметры SSIM
SSIM_WINDOW_SIZE = 11
SSIM_C1 = 0.01 ** 2
SSIM_C2 = 0.03 ** 2

# ---------------------- Функции SSIM ----------------------
def gaussian_window(size, sigma=1.5):
    """Одномерное гауссово окно."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g

def create_window(window_size, channels):
    """Двумерное гауссово окно для всех каналов."""
    _1d = gaussian_window(window_size)
    _2d = _1d[:, None] * _1d[None, :]
    window = _2d.expand(channels, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window, window_size, C1, C2):
    """Вычисляет SSIM между двумя батчами изображений [N, C, H, W]."""
    # Убедимся, что окно на том же устройстве
    window = window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3])  # усреднение по каналам и пространству

# ---------------------- Загрузка данных ----------------------
def load_images(directory):
    files = sorted(
        [f for f in os.listdir(directory) if f.endswith('.pt')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    images = []
    for f in files:
        data = torch.load(os.path.join(directory, f), map_location='cpu', weights_only=False)
        img = data['image']                 # [C, H, W] в [0, 1]
        images.append(img)
    # Стек в батч [N, C, H, W]
    images = torch.stack(images, dim=0)
    return images, files

# ---------------------- Основной расчёт ----------------------
def main():
    print("Загрузка изображений...")
    imgs, file_names = load_images(DATASET_DIR)
    N = imgs.shape[0]
    print(f"Загружено {N} изображений.")

    # Подготовка окна для SSIM
    window = create_window(SSIM_WINDOW_SIZE, imgs.shape[1])

    # Для больших датасетов вычислять всю матрицу N×N может быть затратно по памяти.
    # Будем обрабатывать батчами по строкам (по 16 изображений за раз).
    BATCH_SIZE = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgs = imgs.to(device)
    window = window.to(device)

    all_mse = torch.zeros(N, N, device='cpu')
    all_ssim = torch.zeros(N, N, device='cpu')

    print("Вычисление попарных метрик...")
    for i in range(0, N, BATCH_SIZE):
        i_end = min(i + BATCH_SIZE, N)
        batch_i = imgs[i:i_end]  # [b, C, H, W]
        b = batch_i.shape[0]

        # MSE: ||A - B||^2
        # Разница между batch_i и всеми изображениями
        # (b, C, H, W) vs (N, C, H, W) -> (b, N, C, H, W) разница
        diff = batch_i.unsqueeze(1) - imgs.unsqueeze(0)  # [b, N, C, H, W]
        mse = (diff ** 2).mean(dim=(2, 3, 4))            # [b, N]
        all_mse[i:i_end] = mse.cpu()

        # SSIM: вычисляем для каждой пары (i, j)
        for j in range(b):
            # повторяем j-е изображение из batch_i N раз
            img1 = batch_i[j:j+1].expand(N, -1, -1, -1)  # [N, C, H, W]
            img2 = imgs
            ssim_vals = ssim(img1, img2, window, SSIM_WINDOW_SIZE, SSIM_C1, SSIM_C2)
            all_ssim[i + j] = ssim_vals.cpu()

        print(f"  обработаны строки {i}-{i_end-1} из {N}")

    # Переносим обратно на CPU для дальнейшей обработки
    print("Комбинирование метрик...")
    # Расстояния
    dist_mse = all_mse
    dist_ssim = 1.0 - all_ssim

    # Нормировка: делим каждую матрицу на её среднее (исключая диагональ)
    mask = ~torch.eye(N, dtype=bool)  # не учитываем сам-себя
    mean_mse = dist_mse[mask].mean()
    mean_ssim = dist_ssim[mask].mean()

    norm_mse = dist_mse / mean_mse
    norm_ssim = dist_ssim / mean_ssim

    # Комбинированное расстояние (равные веса)
    combined_dist = 0.5 * norm_mse + 0.5 * norm_ssim

    # Преобразование в сходство
    similarity = torch.exp(-combined_dist / TEMPERATURE)
    # Обнуляем диагональ (сходство самого с собой не нужно)
    similarity.fill_diagonal_(0.0)

    print("Выбор ближайших соседей...")
    neighbors = []
    for i in range(N):
        # Берём top_k значений
        vals, idx = similarity[i].topk(min(TOP_K, N-1))
        neighbors.append({
            'file': file_names[i],
            'index': i,
            'neighbors': [(file_names[j], vals[k].item()) for k, j in enumerate(idx)]
        })

    # Сохранение
    torch.save(neighbors, OUTPUT_PATH)
    print(f"Результат сохранён в {OUTPUT_PATH}")

if __name__ == "__main__":
    main()