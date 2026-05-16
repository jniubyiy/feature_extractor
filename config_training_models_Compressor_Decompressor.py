IMAGE_SIZE = 512

COMPRESSOR_CONFIG = {
    "base_dim": 32,
    "num_blocks": 2,
    "expansion_factor": 2,
    "compressed_channels": 12          # 6 каналов
}

DECOMPRESSOR_CONFIG = {
    "base_dim": 64,
    "num_blocks": 2,
    "expansion_factor": 2,
    "compressed_channels": 12          # 6 каналов на входе
}

# Декодер для визуализации парнетов (из основного автоэнкодера)
DECODER_CONFIG = {
    "base_dim": 64,
    "num_blocks": 3,
    "parnet_channels": 3,
    "dropout_rate": 0.1
}

COMPRESSOR_DEVICE_STR = "cuda:0"
DECOMPRESSOR_DEVICE_STR = "cuda:0"
DECODER_DEVICE_STR = "cuda:0"          # устройство для декодера при тестах

BATCH_SIZE = 1
LEARNING_RATE = 0.00001
NUM_EPOCHS = 10000

MAX_TRAIN_IMAGES = 1

DATASET_DIR = "./prepared_dataset_parnet"
MODELS_DIR = "./models_compressor"
TESTS_DIR = "./tests_compressor"
VAL_TESTS_DIR = "./val_tests_compressor"

# Путь к чекпоинту декодера для визуализации
DECODER_CHECKPOINT = "./models/decoder_epoch200.pth"

SAVE_EVERY_EPOCHS = 1
MAX_CHECKPOINTS = 5

VALIDATION_SPLIT = 10
VAL_EVERY_EPOCHS = 200

TEST_EVERY_EPOCHS = 400
NUM_TEST_EXAMPLES = 1
TEST_SEED = 123

RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True

PARNET_DIFF_LOSS_WEIGHT = 100.0