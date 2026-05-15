# config_preparing_the_dataset_compressed.py
""" Конфигурация для создания датасета сжатых парнетов. """
import torch

# Папка с исходными парнетами (из prepared_dataset_parnet)
DATASET_DIR = "./prepared_dataset_parnet"
# Папка для сохранения сжатых парнетов
OUTPUT_DIR = "./prepared_dataset_parnet_compressed"
# Путь к чекпоинту компрессора (например, "models_compressor/compressor_epoch100.pth")
COMPRESSOR_CHECKPOINT = "./models_compressor/compressor_epoch100.pth"
# Количество параллельных процессов
NUM_WORKERS = 1
# Устройство для инференса
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"