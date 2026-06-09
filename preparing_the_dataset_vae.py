# preparing_the_dataset_vae.py
"""
Скрипт подготовки датасета для VAE: применяет StochasticEncoder к сжатым парнетам,
генерирует z и noise_seed с фиксированным равномерным шумом, сохраняет combined (z+noise_seed) и целевой сжатый парнет.
"""
import torch
import concurrent.futures
from pathlib import Path
from model_VAEWrapper import StochasticEncoder, reparameterize
from config_preparing_the_dataset_vae import *


def process_single_item(args_tuple):
    """
    Загружает сжатый парнет (теперь напрямую, без структурированного), пропускает через StochasticEncoder,
    вычисляет z и noise_seed, объединяет их и сохраняет вместе с целевым сжатым парнетом.
    """
    (comp_path_str, output_dir_str, device_str,
     encoder_config, encoder_checkpoint, noise_range, strength) = args_tuple

    comp_path = Path(comp_path_str)
    output_path = Path(output_dir_str)
    device = torch.device(device_str)

    try:
        # Загружаем сжатый парнет
        comp_data = torch.load(comp_path, map_location='cpu', weights_only=False)
        compressed = comp_data['compressed_parnet'].unsqueeze(0)  # [1, C, H, W]

        # Создаём энкодер и загружаем веса
        encoder = StochasticEncoder(**encoder_config).to(device)
        checkpoint = torch.load(encoder_checkpoint, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Убираем префикс _orig_mod., если модель была сохранена после torch.compile
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        encoder.load_state_dict(state_dict)
        encoder.eval()

        # Инференс энкодера
        with torch.no_grad():
            mu, noise_seed = encoder(compressed.to(device))

            # Равномерный шум с заданными параметрами
            z, _, _ = reparameterize(mu, noise_range, strength, noise_seed)

            # Объединяем z и noise_seed в один тензор (как ожидает StochasticDecoder)
            combined = torch.cat([z, noise_seed], dim=1)  # [1, 2*C, H, W]

        # Сохраняем результат
        save_path = output_path / f"{comp_path.stem}.pt"
        torch.save({
            "combined": combined.squeeze(0).cpu(),          # вход для декодера
            "compressed_parnet": compressed.squeeze(0).cpu()  # целевой сжатый парнет (копия исходного)
        }, save_path)

        return (comp_path.name, "OK")

    except Exception as e:
        return (comp_path.name, f"ERROR: {e}")


def main():
    comp_dir = Path(DATASET_COMP_DIR)   # используем DATASET_COMP_DIR (сжатые парнеты)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Проверяем наличие чекпоинта энкодера
    encoder_checkpoint_path = Path(ENCODER_CHECKPOINT)
    if not encoder_checkpoint_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_checkpoint_path}")

    # Собираем все сжатые парнеты
    comp_files = sorted(comp_dir.glob("*.pt"))
    valid_files = [f for f in comp_files if f.name != "similarities.pt"]

    if not valid_files:
        print("Нет подходящих сжатых парнетов.")
        return

    print(f"Найдено {len(valid_files)} сжатых парнетов для обработки.")
    print(f"Энкодер: {encoder_checkpoint_path}")
    print(f"Параметры шума: range={NOISE_RANGE}, strength={STOCHASTIC_STRENGTH}")
    print(f"Запуск параллельной обработки в {NUM_WORKERS} процессов ({DEVICE})...")

    # Подготавливаем задания
    tasks = []
    for comp_f in valid_files:
        tasks.append((
            str(comp_f),
            str(output_dir),
            DEVICE,
            ENCODER_CONFIG,
            str(encoder_checkpoint_path),
            NOISE_RANGE,
            STOCHASTIC_STRENGTH
        ))

    # Параллельное выполнение
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_item, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            fname = Path(futures[future]).name
            try:
                result = future.result()
                if result[1] == "OK":
                    print(f"OK: {result[0]}")
                else:
                    print(f"FAIL: {result[0]} - {result[1]}")
            except Exception as e:
                print(f"FAIL: {fname} - исключение в процессе: {e}")

    print(f"Готово. VAE-датасет сохранён в {output_dir}")


if __name__ == "__main__":
    main()