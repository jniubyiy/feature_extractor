# config_training_models_Encoder_Decoder.py

# Размер изображения (должен совпадать с TARGET_RESOLUTION из подготовки датасета)
IMAGE_SIZE = 512

# Параметры моделей
ENCODER_CONFIG = {
    "hidden_dim": 1024,
    "num_layers": 6,
    "num_heads": 8,
    "ff_multiplier": 4,
    "dropout": 0.1,
    "patch_size": 32,
    "in_channels": 3,
    "use_adaln": False,
    "use_checkpoint": False
}

DECODER_CONFIG = {
    "hidden_dim": 1024,
    "cond_dim": 1024,
    "num_layers": 6,
    "num_heads": 8,
    "ff_multiplier": 4,
    "dropout": 0.1,
    "patch_size": 32,
    "out_channels": 3,
    "use_checkpoint": False
}

# Устройства обучения
ENCODER_DEVICE_STR = "cuda:0"
DECODER_DEVICE_STR = "cuda:0"

# Обучение
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100

# Директории
DATASET_DIR = "./prepared_dataset"
MODELS_DIR = "./models"
TESTS_DIR = "./tests"

# Чекпоинты
SAVE_EVERY_EPOCHS = 5
MAX_CHECKPOINTS = 3

# Валидация
VALIDATION_SPLIT = 0.1      # доля данных для валидации (берутся из конца датасета)
VAL_EVERY_EPOCHS = 5        # каждые сколько эпох делать валидацию

# Тестирование (сохраняется в ./tests/epoch_N/)
TEST_EVERY_EPOCHS = 5       # как часто запускать тестовые примеры
NUM_TEST_EXAMPLES = 3       # сколько случайных примеров из train_dataset сохранять

# Другие
RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True