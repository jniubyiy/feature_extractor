# config_training_ParNetAutoencoder.py
"""Конфигурация обучения ParNetAutoencoder."""

# Параметры модели
INPUT_CHANNELS = 4          # число каналов сжатого парнета (обычно 4)
BOTTLENECK_CHANNELS = 4     # размерность бутылочного горлышка (латента)
BASE_DIM = 128
NUM_BLOCKS = 2

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

# Устройства
ENCODER_DEVICE_STR = "cuda:0"
DECODER_DEVICE_STR = "cuda:0"

BATCH_SIZE = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 10000
MAX_TRAIN_IMAGES = 427

DATASET_DIR = "./prepared_dataset_parnet_compressed"
MODELS_DIR = "./models_parnet_ae"
TESTS_DIR = "./tests_parnet_ae"
VAL_TESTS_DIR = "./val_tests_parnet_ae"

SAVE_EVERY_EPOCHS = 1
MAX_CHECKPOINTS = 5
VALIDATION_SPLIT = 10
VAL_EVERY_EPOCHS = 2
TEST_EVERY_EPOCHS = 1
NUM_TEST_EXAMPLES = 10
RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True

# Веса потерь
DIFF_LOSS_WEIGHT = 100.0
DIFF_SMOOTH_LOSS_WEIGHT = 200.0   # можно снизить для начала

# Шум на латенте (можно отключить, т.к. латент уже Tanh)
NOISE_STRENGTH = 0.0