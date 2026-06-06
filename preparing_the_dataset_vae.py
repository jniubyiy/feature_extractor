# preparing_the_dataset_vae.py
"""
Скрипт подготовки датасета для VAE: применяет StochasticEncoder к структурированным парнетам,
генерирует z и noise_seed с фиксированным шумом, сохраняет combined (z+noise_seed) и целевой сжатый парнет.
"""
import torch
import concurrent.futures
from pathlib import Path
from model_VAEWrapper import StochasticEncoder, reparameterize
from config_preparing_the_dataset_vae import *


def process_single_item(args_tuple):
    """
    Загружает структурированный и сжатый парнеты, пропускает через StochasticEncoder,
    вычисляет z и noise_seed, объединяет их и сохраняет вместе с целевым сжатым парнетом.
    """
    (struct_path_str, comp_path_str, output_dir_str, device_str,
     encoder_config, encoder_checkpoint, log_var_val, strength) = args_tuple

    struct_path = Path(struct_path_str)
    comp_path = Path(comp_path_str)
    output_path = Path(output_dir_str)
    device = torch.device(device_str)

    try:
        # Загружаем структурированный парнет
        struct_data = torch.load(struct_path, map_location='cpu', weights_only=False)
        structured = struct_data['structured_parnet'].unsqueeze(0)  # [1, C, H, W]

        # Загружаем целевой сжатый парнет
        comp_data = torch.load(comp_path, map_location='cpu', weights_only=False)
        compressed_target = comp_data['compressed_parnet']  # [C, H, W] – цель

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
            mu, noise_seed = encoder(structured.to(device))

            # Фиксированная репараметризация (как в экспортированной модели)
            log_var = torch.full_like(mu, log_var_val)
            z, _, _ = reparameterize(mu, log_var, strength)

            # Объединяем z и noise_seed в один тензор (как ожидает StochasticDecoder)
            combined = torch.cat([z, noise_seed], dim=1)  # [1, 2*C, H, W]

        # Сохраняем результат
        save_path = output_path / f"{struct_path.stem}.pt"
        torch.save({
            "combined": combined.squeeze(0).cpu(),          # вход для декодера
            "compressed_parnet": compressed_target.cpu()    # целевой сжатый парнет
        }, save_path)

        return (struct_path.name, "OK")

    except Exception as e:
        return (struct_path.name, f"ERROR: {e}")


def main():
    struct_dir = Path(DATASET_STRUCT_DIR)
    comp_dir = Path(DATASET_COMP_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Проверяем наличие чекпоинта энкодера
    encoder_checkpoint_path = Path(ENCODER_CHECKPOINT)
    if not encoder_checkpoint_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_checkpoint_path}")

    # Собираем пары структурированный/сжатый парнет
    struct_files = sorted(struct_dir.glob("*.pt"))
    valid_pairs = []
    for sf in struct_files:
        if sf.name == "similarities.pt":
            continue
        cf = comp_dir / sf.name
        if cf.exists():
            valid_pairs.append((sf, cf))
        else:
            print(f"Пропущен {sf.name}: нет сжатого парнета {cf}")

    if not valid_pairs:
        print("Нет подходящих пар структурированный/сжатый парнет.")
        return

    print(f"Найдено {len(valid_pairs)} пар для обработки.")
    print(f"Энкодер: {encoder_checkpoint_path}")
    print(f"Запуск параллельной обработки в {NUM_WORKERS} процессов ({DEVICE})...")

    # Подготавливаем задания
    tasks = []
    for struct_f, comp_f in valid_pairs:
        tasks.append((
            str(struct_f),
            str(comp_f),
            str(output_dir),
            DEVICE,
            ENCODER_CONFIG,
            str(encoder_checkpoint_path),
            LOG_VAR_VALUE,
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