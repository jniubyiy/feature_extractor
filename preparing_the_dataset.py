# preparing_the_dataset.py

import os
import argparse
import numpy as np
from PIL import Image
import torch
from pathlib import Path

# Импорт конфигурации
import config_preparing_the_dataset as cfg


def prepare_dataset(target_resolution: int, dataset_dir: str = "./dataset", output_dir: str = "./prepared_dataset"):
    """
    Загружает все изображения из dataset_dir, приводит к квадратному размеру target_resolution
    с центрированием и паддингом (чёрный), создаёт бинарные маски (0 – паддинг, 1 – изображение)
    и сохраняет тензоры в output_dir с именами {n}.pt.

    Типы изображений:
      a) Хотя бы одна сторона > target_resolution: масштабируется так, чтобы большая сторона стала target_resolution,
         затем центрируется и дополняется паддингом до target_resolution.
      b) Обе стороны ≤ target_resolution: сразу центрируется и дополняется паддингом.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Собираем все изображения (расширения png/jpg/jpeg)
    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_files = []
    for f in dataset_path.iterdir():
        if f.suffix.lower() in image_extensions:
            try:
                int(f.stem)
            except ValueError:
                print(f"Пропущен файл {f}: имя не является целым числом")
                continue
            image_files.append(f)

    if not image_files:
        print("Нет подходящих изображений в папке dataset.")
        return

    print(f"Найдено {len(image_files)} изображений. Целевое разрешение: {target_resolution}")

    images = {}
    for f in image_files:
        try:
            img = Image.open(f).convert("RGB")
            images[f.stem] = img
        except Exception as e:
            print(f"Ошибка при загрузке {f}: {e}")

    for number, pil_img in images.items():
        w, h = pil_img.size
        max_side = max(w, h)

        if max_side > target_resolution:
            ratio = target_resolution / max_side
            new_w = int(round(w * ratio))
            new_h = int(round(h * ratio))
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        canvas = Image.new("RGB", (target_resolution, target_resolution), (0, 0, 0))
        offset_x = (target_resolution - w) // 2
        offset_y = (target_resolution - h) // 2
        canvas.paste(pil_img, (offset_x, offset_y))

        mask = np.zeros((target_resolution, target_resolution), dtype=np.uint8)
        mask[offset_y:offset_y + h, offset_x:offset_x + w] = 1

        img_tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float()

        save_path = output_path / f"{number}.pt"
        torch.save({"image": img_tensor, "mask": mask_tensor}, save_path)

    print(f"Готово. Тензоры сохранены в {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Подготовка датасета: приведение изображений к единому разрешению с паддингом и масками."
    )
    parser.add_argument(
        "--resolution", type=int, default=cfg.TARGET_RESOLUTION,
        help="Целевое разрешение (сторона квадрата), должно быть кратно 32. По умолчанию из конфига."
    )
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--output_dir", type=str, default="./prepared_dataset")
    args = parser.parse_args()

    if args.resolution % 32 != 0:
        print("Предупреждение: разрешение не кратно 32, архитектура рассчитана на кратность 32.")

    prepare_dataset(args.resolution, args.dataset_dir, args.output_dir)