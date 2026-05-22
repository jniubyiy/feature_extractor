IMAGE_SIZE = 512

COMPRESSOR_CONFIG = {
    "base_dim": 64,
    "num_blocks": 4,
    "expansion_factor": 1,
    "compressed_channels": 4          # 6 каналов
}

DECOMPRESSOR_CONFIG = {
    "base_dim": 64,
    "num_blocks": 4,
    "expansion_factor": 1,
    "compressed_channels": 4          # 6 каналов на входе
}

# Декодер для визуализации парнетов (из основного автоэнкодера)
DECODER_CONFIG = {
    "base_dim": 64,
    "num_blocks": 4,
    "parnet_channels": 3,
    "dropout_rate": 0.1
}

COMPRESSOR_DEVICE_STR = "cuda:0"
DECOMPRESSOR_DEVICE_STR = "cuda:0"
DECODER_DEVICE_STR = "cuda:0"          # устройство для декодера при тестах

BATCH_SIZE = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10000

MAX_TRAIN_IMAGES = 427

DATASET_DIR = "./prepared_dataset_parnet"
MODELS_DIR = "./models_compressor"
TESTS_DIR = "./tests_compressor"
VAL_TESTS_DIR = "./val_tests_compressor"

# Путь к чекпоинту декодера для визуализации
DECODER_CHECKPOINT = "./models/decoder_epoch39.pth"

SAVE_EVERY_EPOCHS = 1
MAX_CHECKPOINTS = 5

VALIDATION_SPLIT = 10
VAL_EVERY_EPOCHS = 1

TEST_EVERY_EPOCHS = 1
NUM_TEST_EXAMPLES = 10
TEST_SEED = 123

RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True

PARNET_DIFF_LOSS_WEIGHT = 1000.0

# Потеря гладкости разностного парнета (аналог diff_smooth_loss)
DIFF_SMOOTH_LOSS_WEIGHT = 10000.0

# Параметры потери качества сжатого парнета
QUALITY_LOSS_WEIGHT = 0.000001

QUALITY_SMOOTH_WEIGHT = 1.0
QUALITY_MEAN_WEIGHT = 1.0
QUALITY_STD_WEIGHT = 1.0
QUALITY_MAX_WEIGHT = 1.0
QUALITY_HIST_WEIGHT = 1.0