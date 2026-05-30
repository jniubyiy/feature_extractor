# config_training_ParNetAutoencoder.py
"""Конфигурация обучения ParNetAutoencoder. Выход энкодера – структурированный парнет."""

INPUT_CHANNELS = 4
BOTTLENECK_CHANNELS = 4          # размерность структурированного парнета
BASE_DIM = 64
NUM_BLOCKS = 1

ENCODER_CONFIG = {
    "input_channels": INPUT_CHANNELS,
    "bottleneck_channels": BOTTLENECK_CHANNELS,
    "base_dim": BASE_DIM,
    "num_blocks": NUM_BLOCKS
}
DECODER_CONFIG = {
    "bottleneck_channels": BOTTLENECK_CHANNELS,
    "output_channels": INPUT_CHANNELS,
    "base_dim": BASE_DIM,
    "num_blocks": NUM_BLOCKS
}

ENCODER_DEVICE_STR = "cuda:0"
DECODER_DEVICE_STR = "cuda:0"

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10000
MAX_TRAIN_IMAGES = 427

DATASET_DIR = "./prepared_dataset_parnet_compressed"   # сжатые парнеты (вход)
MODELS_DIR = "./models_parnet_ae"
TESTS_DIR = "./tests_parnet_ae"
VAL_TESTS_DIR = "./val_tests_parnet_ae"

# Пути к замороженным моделям для визуализации
DECOMPRESSOR_CHECKPOINT = "./models_compressor/decompressor_epoch85.pth"
DECODER_CHECKPOINT = "./models/decoder_epoch73.pth"

SAVE_EVERY_EPOCHS = 1
MAX_CHECKPOINTS = 5
VALIDATION_SPLIT = 10
VAL_EVERY_EPOCHS = 20
TEST_EVERY_EPOCHS = 10
NUM_TEST_EXAMPLES = 10
TEST_SEED = 123                   # <-- добавлено
RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True

DIFF_LOSS_WEIGHT = 10.0
DIFF_SMOOTH_LOSS_WEIGHT = 0.00001
NOISE_STRENGTH = 0.0