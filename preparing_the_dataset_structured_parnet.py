# preparing_the_dataset_structured_parnet.py
"""
Загружает сжатые парнеты, пропускает через ParNetEncoder и сохраняет структурированные парнеты.
"""
import torch
import concurrent.futures
from pathlib import Path
from model_ParNetAutoencoder import ParNetEncoder
from config_preparing_the_dataset_structured_parnet import *

def process_single_parnet(args_tuple):
    file_path_str, encoder_checkpoint, output_dir_str, device_str = args_tuple
    file_path = Path(file_path_str)
    output_path = Path(output_dir_str)
    device = torch.device(device_str)

    try:
        # Загружаем сжатый парнет
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        compressed = data['compressed_parnet'].unsqueeze(0)   # добавляем батч

        # Создаём энкодер и загружаем веса
        model = ParNetEncoder(input_channels=4, bottleneck_channels=4, base_dim=128, num_blocks=2).to(device)
        checkpoint = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            structured_parnet = model(compressed.to(device)).squeeze(0).cpu()

        save_path = output_path / f"{file_path.stem}.pt"
        torch.save({"structured_parnet": structured_parnet}, save_path)
        return (file_path.name, "OK")

    except Exception as e:
        return (file_path.name, f"ERROR: {e}")

def main():
    dataset_path = Path(DATASET_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    encoder_path = Path(ENCODER_CHECKPOINT)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    file_paths = []
    for f in sorted(dataset_path.iterdir()):
        if f.suffix.lower() == '.pt':
            if f.name == "similarities.pt":
                print(f"Пропущен {f.name}")
                continue
            file_paths.append(f)

    if not file_paths:
        print("Нет .pt файлов в исходной папке.")
        return

    print(f"Найдено {len(file_paths)} сжатых парнетов.")
    print(f"Используется энкодер: {encoder_path}")
    print(f"Запуск обработки в {NUM_WORKERS} процессов (CPU)...")

    tasks = [(str(p), ENCODER_CHECKPOINT, str(output_path), DEVICE) for p in file_paths]

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_parnet, task): task[0] for task in tasks}
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

    print(f"Готово. Структурированные парнеты сохранены в {output_path}")

if __name__ == "__main__":
    main()