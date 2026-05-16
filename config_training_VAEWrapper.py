# config_training_VAEWrapper.py
"""Конфигурация для обучения VAEWrapper."""

import torch

# Размеры
IMAGE_SIZE = 512                      # ожидаемое разрешение

# Параметры VAEWrapper
COMPRESSED_CHANNELS = 12              # должно совпадать с compressor_config['compressed_channels']
LATENT_DIM = 8                        # размерность вариационного латента (каналов)
HIDDEN_DIM = 32                       # промежуточные каналы в head/tail

# Устройства
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Пути к чекпоинтам (замороженные модели)
ENCODER_CHECKPOINT = "./models/encoder_epoch200.pth"
DECODER_CHECKPOINT = "./models/decoder_epoch200.pth"     # для визуализации
COMPRESSOR_CHECKPOINT = "./models_compressor/compressor_epoch100.pth"

# Датасет – используем подготовленные изображения
DATASET_DIR = "./prepared_dataset"

# Куда сохранять обученный VAEWrapper
MODELS_DIR = "./models_vae_wrapper"
TESTS_DIR = "./tests_vae_wrapper"
VAL_TESTS_DIR = "./val_tests_vae_wrapper"

# Параметры обучения
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
MAX_TRAIN_IMAGES = 200               # сколько изображений использовать для обучения
VALIDATION_SPLIT = 10                # количество валидационных примеров (если >0)

KLD_WEIGHT = 0.001                    # вес KL-дивергенции

# Логирование и сохранение
SAVE_EVERY_EPOCHS = 10
MAX_CHECKPOINTS = 5
VAL_EVERY_EPOCHS = 10
TEST_EVERY_EPOCHS = 20
NUM_TEST_EXAMPLES = 5
TEST_SEED = 123
RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True