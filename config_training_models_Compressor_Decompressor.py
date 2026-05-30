IMAGE_SIZE = 512

COMPRESSOR_CONFIG = {
    "base_dim": 128,
    "num_blocks": 1,
    "compressed_channels": 4,
    "expand_ratio": 2
}
DECOMPRESSOR_CONFIG = {
    "base_dim": 128,
    "num_blocks": 1,
    "compressed_channels": 4,
    "expand_ratio": 2
}

# Декодер для визуализации парнетов (из основного автоэнкодера)
DECODER_CONFIG = {
    "base_dim": 256,
    "num_blocks": 1,
    "parnet_channels": 3,
    "dropout_rate": 0.1,
    "expand_ratio": 2
}

COMPRESSOR_DEVICE_STR = "cuda:0"
DECOMPRESSOR_DEVICE_STR = "cuda:0"
DECODER_DEVICE_STR = "cuda:0"          # устройство для декодера при тестах

BATCH_SIZE = 2
LEARNING_RATE = 0.000001
NUM_EPOCHS = 10000

MAX_TRAIN_IMAGES = 427

DATASET_DIR = "./prepared_dataset_parnet"
MODELS_DIR = "./models_compressor"
TESTS_DIR = "./tests_compressor"
VAL_TESTS_DIR = "./val_tests_compressor"

# Путь к чекпоинту декодера для визуализации
DECODER_CHECKPOINT = "./models/decoder_epoch73.pth"

SAVE_EVERY_EPOCHS = 1
MAX_CHECKPOINTS = 5

VALIDATION_SPLIT = 10
VAL_EVERY_EPOCHS = 5

TEST_EVERY_EPOCHS = 2
NUM_TEST_EXAMPLES = 10
TEST_SEED = 123

RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True

PARNET_DIFF_LOSS_WEIGHT = 1000.0

# Потеря гладкости разностного парнета (аналог diff_smooth_loss)
DIFF_SMOOTH_LOSS_WEIGHT = 10000.0
