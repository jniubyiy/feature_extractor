IMAGE_SIZE = 512

ENCODER_CONFIG = {
    "base_dim": 64,
    "num_blocks": 4,
    "parnet_channels": 3,
    "dropout_rate": 0.1
}
DECODER_CONFIG = {
    "base_dim": 64,
    "num_blocks": 4,
    "parnet_channels": 3,
    "dropout_rate": 0.1
}

ENCODER_DEVICE_STR = "cuda:0"
DECODER_DEVICE_STR = "cuda:0"

BATCH_SIZE = 2
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100000
MAX_TRAIN_IMAGES = 427
DATASET_DIR = "./prepared_dataset"
MODELS_DIR = "./models"
TESTS_DIR = "./tests"
VAL_TESTS_DIR = "./val_tests"
SAVE_EVERY_EPOCHS = 1
MAX_CHECKPOINTS = 5
VALIDATION_SPLIT = 10
VAL_EVERY_EPOCHS = 1
TEST_EVERY_EPOCHS = 2
NUM_TEST_EXAMPLES = 10
RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True
DIFF_LOSS_WEIGHT = 1000.0

# Потеря гладкости разностного изображения
DIFF_SMOOTH_LOSS_WEIGHT = 10000.0

# Параметры потери качества парнета (плавность, нормализация, отсутствие выбросов)
QUALITY_LOSS_WEIGHT = 100.0       # общий коэффициент перед качеством в loss

QUALITY_SMOOTH_WEIGHT = 1.0
QUALITY_MEAN_WEIGHT = 1.0
QUALITY_STD_WEIGHT = 1.0
QUALITY_MAX_WEIGHT = 1.0
QUALITY_HIST_WEIGHT = 1.0