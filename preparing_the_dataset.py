# preparing_the_dataset.py

import os
import argparse
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import concurrent.futures
import multiprocessing

import config_preparing_the_dataset as cfg


def process_single_image(args_tuple):
    """
    Обрабатывает одно изображение в дочернем процессе.
    Принимает кортеж (file_path_str, target_resolution, output_dir_str)
    и сохраняет .pt файл.
    """
    file_path_str, target_resolution, output_dir_str = args_tuple
    file_path = Path(file_path_str)
    output_path = Path(output_dir_str)

    try:
        # Открываем изображение
        img = Image.open(file_path).convert("RGB")
        w, h = img.size
        max_side = max(w, h)

        # Масштабирование, если нужно
        if max_side > target_resolution:
            ratio = target_resolution / max_side
            new_w = int(round(w * ratio))
            new_h = int(round(h * ratio))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        # Центрирование с чёрным паддингом
        canvas = Image.new("RGB", (target_resolution, target_resolution), (0, 0, 0))
        offset_x = (target_resolution - w) // 2
        offset_y = (target_resolution - h) // 2
        canvas.paste(img, (offset_x, offset_y))

        # Бинарная маска
        mask = np.zeros((target_resolution, target_resolution), dtype=np.uint8)
        mask[offset_y:offset_y + h, offset_x:offset_x + w] = 1

        # Тензоры
        img_tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float()

        # Сохранение
        number = file_path.stem
        save_path = output_path / f"{number}.pt"
        torch.save({"image": img_tensor, "mask": mask_tensor}, save_path)

        return (file_path.name, "OK")
    except Exception as e:
        return (file_path.name, f"ERROR: {e}")


def prepare_dataset(target_resolution: int, dataset_dir: str = "./dataset", output_dir: str = "./prepared_dataset"):
    """
    Параллельно (через multiprocessing) загружает, обрабатывает и сохраняет
    все подходящие изображения из dataset_dir.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg'}
    file_paths = []
    for f in sorted(dataset_path.iterdir()):
        if f.suffix.lower() in image_extensions:
            try:
                int(f.stem)
            except ValueError:
                print(f"Пропущен файл {f}: имя не является целым числом")
                continue
            file_paths.append(f)

    if not file_paths:
        print("Нет подходящих изображений в папке dataset.")
        return

    print(f"Найдено {len(file_paths)} изображений. Целевое разрешение: {target_resolution}")
    print(f"Запуск параллельной обработки в {cfg.NUM_WORKERS} процессов...")

    # Готовим аргументы для каждого процесса: (путь_к_файлу, разрешение, выходная_папка)
    tasks = [(str(p), target_resolution, str(output_path)) for p in file_paths]

    # Используем ProcessPoolExecutor для параллельной обработки
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.NUM_WORKERS) as executor:
        # Запускаем все задачи и получаем результаты по мере завершения
        futures = {executor.submit(process_single_image, task): task[0] for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            fname = Path(futures[future]).name
            try:
                result = future.result()
                if result[1] == "OK":
                    print(f"OK: {result[0]}")
                else:
                    print(f"FAIL: {result[0]} – {result[1]}")
            except Exception as e:
                print(f"FAIL: {fname} – исключение в процессе: {e}")

    print(f"Готово. Тензоры сохранены в {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Подготовка датасета: параллельная обработка изображений с паддингом и масками."
    )
    parser.add_argument(
        "--resolution", type=int, default=cfg.TARGET_RESOLUTION,
        help="Целевое разрешение (сторона квадрата), должно быть кратно 32."
    )
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--output_dir", type=str, default="./prepared_dataset")
    args = parser.parse_args()

    if args.resolution % 32 != 0:
        print("Предупреждение: разрешение не кратно 32, архитектура рассчитана на кратность 32.")

    prepare_dataset(args.resolution, args.dataset_dir, args.output_dir)