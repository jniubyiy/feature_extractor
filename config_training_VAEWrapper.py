# config_training_VAEWrapper.py
"""
Конфигурация для трёхфазного обучения:
  StochasticEncoder (mu, noise_seed)
  StochasticDecoder (z + noise_seed → сжатый парнет)
  RefinerDecoder (выход StochasticDecoder + noise_seed → улучшенный сжатый парнет)
"""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================== Общие настройки ===========================
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Конфигурации моделей ===
STOCHASTIC_ENCODER_CONFIG = {
    "compressed_channels": 4,
    "stochastic_parnet_dim": 4,
    "hidden_dim": 512,
    "num_res_blocks": 1,
}

STOCHASTIC_DECODER_CONFIG = {
    "compressed_channels": 4,
    "stochastic_parnet_dim": 4,
    "hidden_dim": 512,
    "num_res_blocks": 4,
}

# ========================= Пути и имена директорий =========================
STRUCTURED_DATASET_DIR = "./prepared_dataset_structured_parnet"   # вход для энкодера
COMPRESSED_DATASET_DIR = "./prepared_dataset_parnet_compressed"   # целевые сжатые парнеты
MODELS_DIR = "./models_vae_wrapper"
TESTS_DIR = "./tests_vae_wrapper"
VAL_TESTS_DIR = "./val_tests_vae_wrapper"

# Пути к замороженным моделям (для визуализации)
AE_CHECKPOINT = "./models_parnet_ae/decoder_epoch234.pth"        # ParNetDecoder
DECOMPRESSOR_CHECKPOINT = "./models_compressor/decompressor_epoch85.pth"
DECODER_CHECKPOINT = "./models/decoder_epoch73.pth"

# ========================= Параметры обучения =========================
BATCH_SIZE = 1
LEARNING_RATE_ENCODER = 0.0000001
LEARNING_RATE_DECODER = 0.000001
NUM_EPOCHS = 100000
MAX_TRAIN_IMAGES = 1
VALIDATION_SPLIT = 10

# ----------------------- Веса потерь ---------------------------------
RECON_LOSS_WEIGHT = 100.0
DIFF_SMOOTH_LOSS_WEIGHT = 100.0
MU_LOSS_WEIGHT = 0.0
MSE_LOSS_WEIGHT = 0.0   # или другое значение
HYBRID_LOSS_WEIGHT = 0.0

# ----------------------- Ручное управление шумом -----------------------
STOCHASTIC_MODE = True
STOCHASTIC_STRENGTH = 1.0
# Диапазон равномерного шума: шум ~ U[-NOISE_RANGE, NOISE_RANGE]
NOISE_RANGE = 2.0

# ----------------------- Сохранение и тестирование ---------------------
SAVE_EVERY_EPOCHS = 5
MAX_CHECKPOINTS = 5
VAL_EVERY_EPOCHS = 50
TEST_EVERY_EPOCHS = 25
NUM_TEST_EXAMPLES = 1
TEST_SEED = 123
RANDOM_SEED = 42

CLEAR_CACHE_EACH_BATCH = True
NUM_MC_SAMPLES = 1