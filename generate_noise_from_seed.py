# generate_noise_from_seed.py
"""
Генерирует тензор, неотличимый от стохастического парнета z (с Tanh на mu).
Распределение: сумма двух равномерных на [-1,1] -> треугольное на [-2,2].
Настройки в начале файла.
"""
import torch
from model_VAEWrapper import PermutationMask

# ================== НАСТРОЙКИ ==================
NOISE_SEED = 42                     # ключ шума
MASK_SEED  = 9876543210987654321    # ключ перестановки (из входного парнета)
STRENGTH   = 1.0                    # амплитуда шума (должна совпадать с обучением)

CHANNELS = 4        # stochastic_parnet_dim
HEIGHT   = 256
WIDTH    = 256
BATCH_SIZE = 1
OUTPUT = "noise_z.pt"
# ===============================================

def generate_z_noise(noise_seed: int, mask_seed: int, strength: float,
                     channels: int, height: int, width: int,
                     batch_size: int) -> torch.Tensor:
    # Генерируем две независимые равномерные величины на [-strength, strength]
    gen = torch.Generator().manual_seed(noise_seed)
    shape = (batch_size, channels, height, width)
    u1 = (torch.rand(shape, generator=gen) * 2 - 1) * strength
    u2 = (torch.rand(shape, generator=gen) * 2 - 1) * strength
    z_raw = u1 + u2   # имитация mu + noise

    # Перестановка
    perm = PermutationMask(channels, height, width, mask_seed)
    z = perm(z_raw)
    return z

if __name__ == "__main__":
    z = generate_z_noise(NOISE_SEED, MASK_SEED, STRENGTH, CHANNELS, HEIGHT, WIDTH, BATCH_SIZE)
    torch.save(z, OUTPUT)
    print(f"Сохранён тензор {tuple(z.shape)} в {OUTPUT}")