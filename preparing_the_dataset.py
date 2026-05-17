# preparing_the_dataset.py
import os
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import concurrent.futures
import config_preparing_the_dataset as cfg

def process_single_image(args_tuple):
    """
    Обрабатывает одно изображение в дочернем процессе.
    Принимает кортеж (file_path_str, target_resolution, output_dir_str)
    и сохраняет .pt файл с ключами 'image', 'mask', 'tags'.
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
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
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

        # Загружаем теги из соответствующего .txt файла
        number = file_path.stem  # теперь это любое имя файла
        tags = []
        tag_file = file_path.with_suffix('.txt')
        if tag_file.exists():
            try:
                with open(tag_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # Разделяем запятыми, удаляем пробелы, фильтруем пустые
                        tags = [tag.strip() for tag in content.split(',') if tag.strip()]
            except Exception:
                # Если не удалось прочитать, оставляем пустой список
                pass

        # Сохранение
        save_path = output_path / f"{number}.pt"
        torch.save({"image": img_tensor, "mask": mask_tensor, "tags": tags}, save_path)

        return (file_path.name, "OK")

    except Exception as e:
        return (file_path.name, f"ERROR: {e}")

def prepare_dataset(target_resolution: int, dataset_dir: str, output_dir: str):
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
            # Больше не требуем, чтобы имя было целым числом
            file_paths.append(f)

    if not file_paths:
        print("Нет подходящих изображений в папке dataset.")
        return

    print(f"Найдено {len(file_paths)} изображений. Целевое разрешение: {target_resolution}")
    print(f"Запуск параллельной обработки в {cfg.NUM_WORKERS} процессов...")

    # Готовим аргументы для каждого процесса
    tasks = [(str(p), target_resolution, str(output_path)) for p in file_paths]

    # Используем ProcessPoolExecutor для параллельной обработки
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.NUM_WORKERS) as executor:
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
    if cfg.TARGET_RESOLUTION % 32 != 0:
        print("Предупреждение: разрешение не кратно 32, архитектура рассчитана на кратность 32.")
    prepare_dataset(cfg.TARGET_RESOLUTION, cfg.DATASET_DIR, cfg.OUTPUT_DIR)