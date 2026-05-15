IMAGE_SIZE = 512

ENCODER_CONFIG = {
    "base_dim": 64,
    "num_blocks": 3,
    "parnet_channels": 3,
    "dropout_rate": 0.1            # добавлен dropout
}

DECODER_CONFIG = {
    "base_dim": 64,
    "num_blocks": 3,
    "parnet_channels": 3,
    "dropout_rate": 0.1
}

ENCODER_DEVICE_STR = "cuda:0"
DECODER_DEVICE_STR = "cuda:0"

BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100000

MAX_TRAIN_IMAGES = 200

DATASET_DIR = "./prepared_dataset"
MODELS_DIR = "./models"
TESTS_DIR = "./tests"
VAL_TESTS_DIR = "./val_tests"

SAVE_EVERY_EPOCHS = 1
MAX_CHECKPOINTS = 5

VALIDATION_SPLIT = 10
VAL_EVERY_EPOCHS = 1

TEST_EVERY_EPOCHS = 1
NUM_TEST_EXAMPLES = 10

RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True


TV_LOSS_WEIGHT = 100.0
MSE_LOSS_WEIGHT = 100.0 
EDGE_LOSS_WEIGHT = 100.0 
SSIM_LOSS_WEIGHT = 100.0     
NOISE_LOSS_WEIGHT = 100.0