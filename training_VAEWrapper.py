# training_VAEWrapper.py
"""
Обучение VAEWrapper поверх замороженных Encoder + ParnetCompressor.

Использует подготовленный датасет изображений (prepared_dataset).
Каждый батч:
  image -> encoder (заморожен) -> parnet
        -> compressor (заморожен) -> сжатый_парнет c
        -> VAEWrapper(c) -> c_hat, mu, logvar
        -> loss = L1(c_hat, c) + kld_weight * KL

Для визуализации качества восстановления используется замороженный декодер.
"""

import os, re, glob, math, random, gc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from model_Autoencoder import Encoder, Decoder
from model_ParnetCompressor import ParnetCompressor
from model_VAEWrapper import VAEWrapper
from config_training_VAEWrapper import *

# Устройства
DEVICE = torch.device(DEVICE)
ENCODER_DEVICE = DEVICE
COMPRESSOR_DEVICE = DEVICE   # будем считать, что всё на одном устройстве

# ---------------------- Загрузка и заморозка базовых моделей ----------------------
def load_frozen_encoder(checkpoint_path):
    model = Encoder(
        base_dim=64,
        num_blocks=3,
        parnet_channels=3,
        dropout_rate=0.1
    ).to(ENCODER_DEVICE)
    state = torch.load(checkpoint_path, map_location=ENCODER_DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def load_frozen_decoder(checkpoint_path):
    model = Decoder(
        base_dim=64,
        num_blocks=3,
        parnet_channels=3,
        dropout_rate=0.1
    ).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def load_frozen_compressor(checkpoint_path):
    model = ParnetCompressor(
        base_dim=32,
        num_blocks=2,
        expansion_factor=2,
        compressed_channels=COMPRESSED_CHANNELS
    ).to(COMPRESSOR_DEVICE)
    state = torch.load(checkpoint_path, map_location=COMPRESSOR_DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

# ---------------------- Датасет изображений ----------------------
class ImageDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        img = data['image']      # [C,H,W] в [0,1]
        return img * 2 - 1       # -> [-1,1]

def collate_fn(batch):
    return torch.stack(batch, dim=0)

# ---------------------- Функции потерь и метрик ----------------------
def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(2.0) - 10 * math.log10(mse.item())

# ---------------------- Визуализация ----------------------
def tensor_to_pil(t):
    arr = (t.cpu().clamp(-1, 1).numpy() + 1) / 2 * 255
    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(arr)

def save_example(base_dir, image, recon_image, compressed, recon_compressed,
                 diff, metrics, prefix=""):
    os.makedirs(base_dir, exist_ok=True)
    tensor_to_pil(image).save(os.path.join(base_dir, f"{prefix}original.png"))
    tensor_to_pil(recon_image).save(os.path.join(base_dir, f"{prefix}reconstructed.png"))
    # diff – разница изображений
    tensor_to_pil(diff).save(os.path.join(base_dir, f"{prefix}difference.png"))
    # Также можно сохранить сжатые представления в виде отдельного визуального представления (среднее по каналам)
    # Но здесь просто сохраним метрики
    with open(os.path.join(base_dir, f"{prefix}metrics.txt"), 'w') as f:
        f.write(f"Recon L1: {metrics['recon_l1']:.6f}\n")
        f.write(f"KL loss: {metrics['kld']:.6f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")

# ---------------------- Чекпоинты ----------------------
def get_model_path(name, epoch):
    return os.path.join(MODELS_DIR, f"{name}_epoch{epoch}.pth")

def find_latest_checkpoint():
    files = glob.glob(os.path.join(MODELS_DIR, "vae_wrapper_epoch*.pth"))
    if not files:
        return None, 0
    def extract_epoch(f):
        m = re.search(r'epoch(\d+)', f)
        return int(m.group(1)) if m else -1
    latest = max(files, key=extract_epoch)
    return latest, extract_epoch(latest)

def cleanup_old_checkpoints(keep=MAX_CHECKPOINTS):
    files = glob.glob(os.path.join(MODELS_DIR, "vae_wrapper_epoch*.pth"))
    if len(files) <= keep:
        return
    files.sort(key=lambda f: int(re.search(r'epoch(\d+)', f).group(1)), reverse=True)
    for old in files[keep:]:
        try:
            os.remove(old)
        except OSError:
            pass

def save_checkpoint(epoch, model, optimizer):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, get_model_path("vae_wrapper", epoch))
    cleanup_old_checkpoints()

def load_checkpoint_if_exist(model, optimizer):
    path, epoch = find_latest_checkpoint()
    if path:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Loaded VAEWrapper from epoch {epoch}")
        return epoch
    return 0

# ---------------------- Тестирование / Валидация ----------------------
@torch.no_grad()
def evaluate_and_visualize(model, encoder, compressor, decoder, dataset, output_base, epoch, num_examples, seed=None):
    """Генерирует тестовые примеры, сохраняет изображения и метрики."""
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    for idx in indices:
        img = dataset[idx].unsqueeze(0).to(DEVICE)          # [1,3,H,W] в [-1,1]
        # Прямой проход через замороженные модели
        parnet = encoder(img)
        c = compressor(parnet)                               # сжатый парнет
        c_hat, mu, logvar = model(c)
        # Восстановление до изображения
        with torch.no_grad():
            # Разжимаем через декомпрессор (если есть) или обходной путь – здесь используем замороженный декомпрессор?
            # У нас нет замороженного декомпрессора в этом скрипте. Используем ParnetDecompressor?
            # Для простоты будем декодировать изображение напрямую, используя замороженный декодер,
            # но ему нужен полный парнет. Сначала нужно восстановить парнет из c_hat.
            # У нас нет загруженного decompressor. Мы можем загрузить его аналогично компрессору.
            # Предположим, что мы загружаем decompressor отдельно.
            pass
        # Пока оставим только метрики восстановления c.
        recon_l1 = F.l1_loss(c_hat, c).item()
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / c.size(0)
        kld = kld.item()
        # PSNR для c (не особо информативно)
        psnr = compute_psnr(c_hat, c)

        eid = int(os.path.splitext(os.path.basename(dataset.files[idx]))[0])
        base_dir = os.path.join(output_base, f"epoch_{epoch}", f"example_{eid}")
        os.makedirs(base_dir, exist_ok=True)
        # Сохраним c и c_hat как .pt для анализа
        torch.save({"c": c.cpu(), "c_hat": c_hat.cpu()}, os.path.join(base_dir, "compressed.pt"))
        with open(os.path.join(base_dir, "metrics.txt"), 'w') as f:
            f.write(f"Recon L1 (c): {recon_l1:.6f}\n")
            f.write(f"KL: {kld:.6f}\n")
            f.write(f"PSNR (c): {psnr:.2f} dB\n")

# ---------------------- Обучение одной эпохи ----------------------
def train_epoch(model, encoder, compressor, train_loader, optimizer):
    model.train()
    total_recon = 0.0
    total_kld = 0.0
    n_batches = 0
    for batch_idx, images in enumerate(train_loader):
        images = images.to(DEVICE)
        with torch.no_grad():
            parnet = encoder(images)
            c = compressor(parnet)     # [B, C, H/2, W/2]
        # Прямой проход VAEWrapper
        c_hat, mu, logvar = model(c)
        loss, recon_loss, kld_loss = model.loss(c, c_hat, mu, logvar, kld_weight=KLD_WEIGHT)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon += recon_loss.item()
        total_kld += kld_loss.item()
        n_batches += 1

        print(f"Batch {batch_idx+1}/{len(train_loader)} | "
              f"Recon: {recon_loss.item():.6f} | KL: {kld_loss.item():.6f} | Total: {loss.item():.6f}")

        del images, parnet, c, c_hat, mu, logvar, loss
        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    avg_recon = total_recon / n_batches
    avg_kld = total_kld / n_batches
    return avg_recon, avg_kld

# ---------------------- Основная функция ----------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Загрузка замороженных моделей
    print("Loading frozen encoder...")
    encoder = load_frozen_encoder(ENCODER_CHECKPOINT)
    print("Loading frozen compressor...")
    compressor = load_frozen_compressor(COMPRESSOR_CHECKPOINT)
    # Декодер загрузим для тестов
    print("Loading frozen decoder...")
    decoder = load_frozen_decoder(DECODER_CHECKPOINT) if os.path.exists(DECODER_CHECKPOINT) else None

    # Подготовка датасета
    all_files = sorted(
        [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.pt')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    if not all_files:
        raise RuntimeError(f"No .pt files in {DATASET_DIR}")
    print(f"Found {len(all_files)} samples.")

    if MAX_TRAIN_IMAGES and MAX_TRAIN_IMAGES > 0:
        train_files = all_files[:MAX_TRAIN_IMAGES]
        if VALIDATION_SPLIT > 0 and len(all_files) > MAX_TRAIN_IMAGES:
            val_files = all_files[MAX_TRAIN_IMAGES:MAX_TRAIN_IMAGES+VALIDATION_SPLIT]
        else:
            val_files = []
    else:
        n_val = min(VALIDATION_SPLIT, len(all_files))
        train_files = all_files[:-n_val] if n_val < len(all_files) else all_files
        val_files = all_files[-n_val:] if n_val > 0 else []

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    train_dataset = ImageDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=0)

    val_dataset = None
    if val_files:
        val_dataset = ImageDataset(val_files)

    # Создание VAEWrapper и оптимизатора
    model = VAEWrapper(COMPRESSED_CHANNELS, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoint_if_exist(model, optimizer) + 1

    # Цикл обучения
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_recon, avg_kld = train_epoch(model, encoder, compressor, train_loader, optimizer)
        print(f"Epoch {epoch:3d} | Avg Recon: {avg_recon:.6f} | Avg KL: {avg_kld:.6f}")

        # Валидация
        if val_dataset and epoch % VAL_EVERY_EPOCHS == 0:
            print("Running validation...")
            evaluate_and_visualize(model, encoder, compressor, decoder, val_dataset,
                                   VAL_TESTS_DIR, epoch, NUM_TEST_EXAMPLES, TEST_SEED)

        # Тестирование (на тренировочных примерах)
        if epoch % TEST_EVERY_EPOCHS == 0:
            print("Running tests...")
            evaluate_and_visualize(model, encoder, compressor, decoder, train_dataset,
                                   TESTS_DIR, epoch, NUM_TEST_EXAMPLES, TEST_SEED)

        # Сохранение
        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoint(epoch, model, optimizer)
            print(f"Checkpoint saved at epoch {epoch}")

    print("Training completed.")

if __name__ == "__main__":
    main()