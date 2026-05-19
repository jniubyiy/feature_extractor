# preparing_the_dataset_parnet.py
import torch
import concurrent.futures
from pathlib import Path
from model_Autoencoder import Encoder
from config_training_models_Encoder_Decoder import ENCODER_CONFIG
import config_preparing_the_dataset_parnet as cfg

def process_single_image(args_tuple):
    """
    Загружает .pt файл (изображение), прогоняет через энкодер и сохраняет парнет.
    Аргументы: (file_path_str, encoder_state_path, output_dir_str, device_str).
    """
    file_path_str, encoder_state_path, output_dir_str, device_str = args_tuple
    file_path = Path(file_path_str)
    output_path = Path(output_dir_str)
    device = torch.device(device_str)

    try:
        # Загружаем тензор изображения
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        image = data['image']  # [0,1] (C, H, W)
        image = image.unsqueeze(0) * 2 - 1  # [0,1] -> [-1,1], батч размерности 1

        # Создаём энкодер и загружаем веса
        model = Encoder(**ENCODER_CONFIG).to(device)
        state_dict = torch.load(encoder_state_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        # Инференс – парнет теперь без ограничения диапазона, не содержит нулей
        with torch.no_grad():
            parnet = model(image.to(device))  # [1, 3, H, W]

        # Сохраняем парнет (убираем размерность батча)
        save_path = output_path / f"{file_path.stem}.pt"
        torch.save({"parnet": parnet.squeeze(0).cpu()}, save_path)
        return (file_path.name, "OK")

    except Exception as e:
        return (file_path.name, f"ERROR: {e}")

def prepare_dataset_parnet():
    dataset_path = Path(cfg.DATASET_DIR)
    output_path = Path(cfg.OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Проверка чекпоинта
    encoder_path = Path(cfg.ENCODER_CHECKPOINT)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    file_paths = []
    for f in sorted(dataset_path.iterdir()):
        if f.suffix.lower() == '.pt':
            # Пропускаем файл similarities.pt
            if f.name == "similarities.pt":
                print(f"Пропущен файл {f.name} (служебный файл схожести)")
                continue
            # Больше не требуем, чтобы имя было целым числом
            file_paths.append(f)

    if not file_paths:
        print("Нет подходящих .pt файлов в папке dataset.")
        return

    print(f"Найдено {len(file_paths)} изображений для конвертации.")
    print(f"Используется энкодер: {encoder_path}")
    print(f"Запуск параллельной обработки в {cfg.NUM_WORKERS} процессов...")

    tasks = [
        (str(p), cfg.ENCODER_CHECKPOINT, str(output_path), cfg.DEVICE)
        for p in file_paths
    ]

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

    print(f"Готово. Парнеты сохранены в {output_path}")

if __name__ == "__main__":
    prepare_dataset_parnet()