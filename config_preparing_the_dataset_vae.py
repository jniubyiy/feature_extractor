# config_preparing_the_dataset_vae.py
"""
Конфигурация для подготовки VAE-датасета (предварительный инференс энкодера).
Генерирует z и noise_seed из структурированных парнетов и сохраняет их вместе с целевым сжатым парнетом.
"""
import torch

# Пути к исходным данным
DATASET_STRUCT_DIR = "./prepared_dataset_structured_parnet"   # структурированные парнеты
DATASET_COMP_DIR = "./prepared_dataset_parnet_compressed"     # сжатые парнеты (цель)

# Папка для сохранения результатов
OUTPUT_DIR = "./prepared_dataset_vae"

# Чекпоинт StochasticEncoder
ENCODER_CHECKPOINT = "./models_vae_wrapper/encoder_epoch100.pth"  # указать актуальный чекпоинт

# Параметры параллельной обработки
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Параметры энкодера (должны совпадать с чекпоинтом)
ENCODER_CONFIG = {
    "compressed_channels": 4,
    "stochastic_parnet_dim": 4,
    "hidden_dim": 512,
    "num_res_blocks": 1,
}

# Параметры шума (фиксированы, как в обучении)
LOG_VAR_VALUE = 1.0
STOCHASTIC_STRENGTH = 1.0