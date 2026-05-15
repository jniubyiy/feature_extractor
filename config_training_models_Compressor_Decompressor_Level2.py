# config_training_models_Compressor_Decompressor_Level2.py
IMAGE_SIZE = 512   # исходное изображение, для сжатого парнета размер H/2=256

COMPRESSOR_CONFIG = {
    "base_dim": 64,
    "num_blocks": 2,
    "expansion_factor": 2,
    "compressed_channels": 5
}

DECOMPRESSOR_CONFIG = {
    "base_dim": 64,
    "num_blocks": 2,
    "expansion_factor": 2,
    "compressed_channels": 5
}

# Декодер для визуализации (основного автоэнкодера, принимает 3 канала)
DECODER_CONFIG = {
    "base_dim": 64,
    "num_blocks": 3,
    "parnet_channels": 3,
    "dropout_rate": 0.1
}

COMPRESSOR_DEVICE_STR = "cuda:0"
DECOMPRESSOR_DEVICE_STR = "cuda:0"
DECODER_DEVICE_STR = "cuda:0"

BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000

MAX_TRAIN_IMAGES = 1

DATASET_DIR = "./prepared_dataset_parnet_compressed"   # содержит сжатые парнеты [4,256,256]
MODELS_DIR = "./models_compressor_level2"
TESTS_DIR = "./tests_compressor_level2"
VAL_TESTS_DIR = "./val_tests_compressor_level2"

# Декодер основного автоэнкодера
DECODER_CHECKPOINT = "./models/decoder_epoch178.pth"
# Декомпрессор первого уровня (из models_compressor)
LEVEL1_DECOMPRESSOR_CHECKPOINT = "./models_compressor/decompressor_epoch100.pth"

SAVE_EVERY_EPOCHS = 50
MAX_CHECKPOINTS = 3

VALIDATION_SPLIT = 1
VAL_EVERY_EPOCHS = 50

TEST_EVERY_EPOCHS = 100
NUM_TEST_EXAMPLES = 1
TEST_SEED = 123

RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True

PARNET_DIFF_LOSS_WEIGHT = 1.0