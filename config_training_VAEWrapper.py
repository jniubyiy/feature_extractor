# config_training_VAEWrapper.py
"""Конфигурация для обучения VAEWrapper."""
import torch

IMAGE_SIZE = 512
COMPRESSED_CHANNELS = 4
STOCHASTIC_PARNET_DIM = 4
HIDDEN_DIM = 64                # увеличено с 32
NUM_RES_BLOCKS = 4               # количество остаточных блоков в head и tail
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Пути к замороженным моделям (только для визуализации)
DECOMPRESSOR_CHECKPOINT = "./models_compressor/decompressor_epoch47.pth"
DECODER_CHECKPOINT = "./models/decoder_epoch1576.pth"

# Датасет – сжатые парнеты
DATASET_DIR = "./prepared_dataset_parnet_compressed"

MODELS_DIR = "./models_vae_wrapper"
TESTS_DIR = "./tests_vae_wrapper"
VAL_TESTS_DIR = "./val_tests_vae_wrapper"

BATCH_SIZE = 1
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100000
MAX_TRAIN_IMAGES = 427
VALIDATION_SPLIT = 10

# ----- Параметры включения и прогрева KL -----
USE_KL_LOSS = True                    # главный переключатель: False – KL никогда не используется
KL_ENABLE_RECON_THRESHOLD = 4000.0       # KL включается, когда средний W_Recon за эпоху опустится ниже этого порога
KL_WARMUP_EPOCHS = 100                  # число эпох, за которое множитель KL растёт от стартового до конечного
KL_START_WEIGHT = 0.00001             # начальный множитель KL при включении
KL_END_WEIGHT = 0.00001                   # конечный множитель KL после прогрева

# ----- Защита от низкого / коллапсирующего KL (действует после прогрева) -----
KL_TARGET_MIN = 0.5                   # если KL батча ниже, множитель уменьшается (защита от коллапса)
KL_ADAPTIVE_POWER = 2.0               # степень для ослабления веса: (KL / KL_TARGET_MIN) ** power
KL_WEIGHT_MIN = 0.000001              # абсолютный минимум, до которого может упасть множитель KL
KL_WEIGHT_MAX = 10.0                  # (не используется, оставлен для совместимости)
KL_ZERO_THRESHOLD = 0.09              # если KL батча <= этому значению, вес KL обнуляется полностью

# ----- Веса потерь -----
RECON_LOSS_WEIGHT = 100.0               # множитель L1-потери реконструкции сжатого парнета

SAVE_EVERY_EPOCHS = 2
MAX_CHECKPOINTS = 5
VAL_EVERY_EPOCHS = 20
TEST_EVERY_EPOCHS = 40
NUM_TEST_EXAMPLES = 10
TEST_SEED = 123
RANDOM_SEED = 42
CLEAR_CACHE_EACH_BATCH = True

# Параметры регуляризации схожести
SIMILARITY_LOSS_WEIGHT = 10.0
SIMILARITIES_FILE = "./prepared_dataset/similarities.pt"
SIMILARITY_BUFFER_BATCHES = 8
SIMILARITY_WARMUP_BATCHES = 4
