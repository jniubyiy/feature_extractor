# preparing_the_dataset_compressed.py
import torch
import concurrent.futures
from pathlib import Path
from model_ParnetCompressor import ParnetCompressor
from config_training_models_Compressor_Decompressor import COMPRESSOR_CONFIG
import config_preparing_the_dataset_compressed as cfg

def process_single_parnet(args_tuple):
    """
    Загружает .pt файл с парнетом, прогоняет через компрессор и сохраняет сжатый парнет.
    Аргументы: (file_path_str, compressor_state_path, output_dir_str, device_str).
    """
    file_path_str, compressor_state_path, output_dir_str, device_str = args_tuple
    file_path = Path(file_path_str)
    output_path = Path(output_dir_str)
    device = torch.device(device_str)

    try:
        # Загружаем парнет (ожидаем ключ 'parnet')
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        parnet = data['parnet']  # [3, H, W], значения не ограничены
        parnet = parnet.unsqueeze(0)  # батч размерности 1

        # Создаём компрессор и загружаем веса
        model = ParnetCompressor(**COMPRESSOR_CONFIG).to(device)
        checkpoint = torch.load(compressor_state_path, map_location=device, weights_only=False)

        # Поддержка как чистого state_dict, так и полного чекпоинта
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

        # Инференс
        with torch.no_grad():
            compressed = model(parnet.to(device))  # [1, 4, H/2, W/2]

        # Сохраняем сжатый парнет (убираем размерность батча)
        save_path = output_path / f"{file_path.stem}.pt"
        compressed_cpu = compressed.squeeze(0).cpu()
        torch.save({"compressed_parnet": compressed_cpu}, save_path)

        # Проверка нулей
        zero_count = (compressed_cpu == 0).sum().item()
        if zero_count > 0:
            return (file_path.name, f"WARNING: {zero_count} zeros found")
        return (file_path.name, "OK")

    except Exception as e:
        return (file_path.name, f"ERROR: {e}")

def prepare_dataset_compressed():
    dataset_path = Path(cfg.DATASET_DIR)
    output_path = Path(cfg.OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Проверка чекпоинта
    compressor_path = Path(cfg.COMPRESSOR_CHECKPOINT)
    if not compressor_path.exists():
        raise FileNotFoundError(f"Compressor checkpoint not found: {compressor_path}")

    file_paths = []
    for f in sorted(dataset_path.iterdir()):
        if f.suffix.lower() == '.pt':
            # Пропускаем служебный файл similarities.pt
            if f.name == "similarities.pt":
                print(f"Пропущен файл {f.name} (служебный файл схожести)")
                continue
            # Поддерживаем любые имена файлов (цифры, буквы и т.д.)
            file_paths.append(f)

    if not file_paths:
        print("Нет подходящих .pt файлов в папке dataset.")
        return

    print(f"Найдено {len(file_paths)} парнетов для сжатия.")
    print(f"Используется компрессор: {compressor_path}")
    print(f"Запуск параллельной обработки в {cfg.NUM_WORKERS} процессов...")

    tasks = [
        (str(p), cfg.COMPRESSOR_CHECKPOINT, str(output_path), cfg.DEVICE)
        for p in file_paths
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_parnet, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            fname = Path(futures[future]).name
            try:
                result = future.result()
                if result[1].startswith("WARNING"):
                    print(f"WARNING: {result[0]} – {result[1]}")
                elif result[1] == "OK":
                    print(f"OK: {result[0]}")
                else:
                    print(f"FAIL: {result[0]} – {result[1]}")
            except Exception as e:
                print(f"FAIL: {fname} – исключение в процессе: {e}")

    print(f"Готово. Сжатые парнеты сохранены в {output_path}")

if __name__ == "__main__":
    prepare_dataset_compressed()