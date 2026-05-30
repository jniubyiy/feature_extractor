# config_preparing_the_dataset_structured_parnet.py
"""Конфигурация для создания датасета структурированных парнетов."""

DATASET_DIR = "./prepared_dataset_parnet_compressed"   # откуда берём сжатые парнеты
OUTPUT_DIR = "./prepared_dataset_structured_parnet"             # куда сохраняем структурированные парнеты
ENCODER_CHECKPOINT = "./models_parnet_ae/encoder_epoch10.pth"  # указать актуальный чекпоинт
NUM_WORKERS = 4
DEVICE = "cpu"   # для параллельной обработки безопаснее CPU