# config_preparing_the_dataset_structured_parnet.py
"""Конфигурация для создания датасета структурированных парнетов."""

DATASET_DIR = "./prepared_dataset_parnet_compressed"   # откуда берём сжатые парнеты
OUTPUT_DIR = "./prepared_dataset_structured_parnet"             # куда сохраняем структурированные парнеты
ENCODER_CHECKPOINT = "./models_parnet_ae/encoder_epoch234.pth"  # указать актуальный чекпоинт
NUM_WORKERS = 4
DEVICE = "cuda"   # для параллельной обработки безопаснее CPU

# Параметры архитектуры энкодера (должны совпадать с чекпоинтом)
ENCODER_BASE_DIM = 256      # base_dim, с которым обучался энкодер
ENCODER_NUM_BLOCKS = 1      # число GlobalContextScaleBlock в энкодере