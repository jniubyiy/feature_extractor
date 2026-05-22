# config_training_VAEWrapper.py
"""
Конфигурация для двухфазного обучения StochasticEncoder и StochasticDecoder.
Все настройки сгруппированы по смыслу.
"""
import torch

# =========================== Общие настройки ===========================
IMAGE_SIZE = 512                  # исходное разрешение изображений (для справки)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== Параметры архитектуры моделей ====================
COMPRESSED_CHANNELS = 4           # число каналов во входном/выходном сжатом парнете
STOCHASTIC_PARNET_DIM = 4         # размерность стохастического парнета (каналов) – одинакова для обеих моделей

# --- Энкодер (StochasticEncoder) ---
ENCODER_HIDDEN_DIM = 128           # каналов в скрытых слоях энкодера
ENCODER_NUM_RES_BLOCKS = 4        # количество ResidualBlock1x1 в энкодере

# --- Декодер (StochasticDecoder) ---
DECODER_HIDDEN_DIM = 128           # каналов в скрытых слоях декодера
DECODER_NUM_RES_BLOCKS = 6        # количество ResidualBlock1x1 в декодере

# ========================= Пути и имена директорий =========================
DATASET_DIR = "./prepared_dataset_parnet_compressed"   # сжатые парнеты
MODELS_DIR = "./models_vae_wrapper"                    # чекпоинты моделей
TESTS_DIR = "./tests_vae_wrapper"                      # визуализации на train
VAL_TESTS_DIR = "./val_tests_vae_wrapper"              # визуализации на val

# Пути к замороженным моделям (только для визуализации)
DECOMPRESSOR_CHECKPOINT = "./models_compressor/decompressor_epoch47.pth"
DECODER_CHECKPOINT = "./models/decoder_epoch1576.pth"

# ========================= Параметры обучения =========================
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100000
MAX_TRAIN_IMAGES = 427            # сколько первых изображений использовать для обучения
VALIDATION_SPLIT = 10             # сколько последних изображений отвести под валидацию

# ----------------------- Веса потерь ---------------------------------
RECON_LOSS_WEIGHT = 1000.0         # множитель L1 потери реконструкции сжатого парнета

# Штраф за негладкую разность сжатых парнетов
DIFF_SMOOTH_LOSS_WEIGHT = 10000.0

# ----------------------- Стохастичность и KL-регуляризация -----------------------
STOCHASTIC_MODE = True          # True = энкодер выдаёт стохастический z (mu + eps*std),
                                 # False = детерминированный mu, KL отключён принудительно
STOCHASTIC_STRENGTH = 1        # сила стохастичности (0.0 – без шума, 1.0 – полный шум)
USE_KL_LOSS = True               # Работает только при STOCHASTIC_MODE = True
KL_WEIGHT = 0.000001             # постоянный базовый множитель KL-лосса

# Защита от низкого / коллапсирующего KL (действует только при STOCHASTIC_MODE = True)
KL_TARGET_MIN = 0.5
KL_ADAPTIVE_POWER = 1.0
KL_WEIGHT_MIN = 0.000001
KL_ZERO_THRESHOLD = 0.09

# ----------------------- Параметры регуляризации схожести ---------------
# (в текущей версии не задействованы, но оставлены для совместимости)
SIMILARITY_LOSS_WEIGHT = 100.0
SIMILARITIES_FILE = "./prepared_dataset/similarities.pt"
SIMILARITY_BUFFER_BATCHES = 8
SIMILARITY_WARMUP_BATCHES = 4

# ======================== Сохранение и тестирование =====================
SAVE_EVERY_EPOCHS = 2
MAX_CHECKPOINTS = 5               # сколько последних чекпоинтов хранить
VAL_EVERY_EPOCHS = 2             # каждые сколько эпох запускать валидацию
TEST_EVERY_EPOCHS = 5            # каждые сколько эпох генерировать тестовые примеры
NUM_TEST_EXAMPLES = 10            # сколько примеров визуализировать
TEST_SEED = 123                   # фиксированный seed для тестовых выборок
RANDOM_SEED = 42                  # общий seed для воспроизводимости

# ==================== Оптимизация и память ============================
CLEAR_CACHE_EACH_BATCH = True     # очищать кэш CUDA после каждого батча


# ----------------------- Потеря качества стохастического парнета -----------------------
QUALITY_LOSS_WEIGHT = 0.00001

QUALITY_SMOOTH_WEIGHT = 1.0
QUALITY_MEAN_WEIGHT = 1.0
QUALITY_STD_WEIGHT = 1.0
QUALITY_MAX_WEIGHT = 1.0
QUALITY_HIST_WEIGHT = 1.0