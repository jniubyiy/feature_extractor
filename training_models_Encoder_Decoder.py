# training_models_Encoder_Decoder.py

import os
import re
import glob
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc

from model_Encoder import Encoder
from model_Decoder import Decoder
from config_training_models_Encoder_Decoder import *

# Устройства
ENCODER_DEVICE = torch.device(ENCODER_DEVICE_STR if torch.cuda.is_available() else "cpu")
DECODER_DEVICE = torch.device(DECODER_DEVICE_STR if torch.cuda.is_available() else "cpu")
print(f"Encoder device: {ENCODER_DEVICE}, Decoder device: {DECODER_DEVICE}")


# ==================== Датасет ====================
class ImageDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        image = data['image']  # (3, H, W)
        mask = data['mask']    # (H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask


def collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks


# ==================== Потери и метрики ====================
def masked_mse_loss(pred, target, mask):
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    loss = masked_diff.sum() / mask.sum().clamp(min=1)
    return loss


def compute_psnr(pred, target, mask):
    mse = masked_mse_loss(pred, target, mask)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0) - 10 * math.log10(mse.item())
    return psnr


# ==================== Чекпоинты ====================
def get_model_path(model_name, epoch):
    return os.path.join(MODELS_DIR, f"{model_name}_epoch{epoch}.pth")


def find_latest_checkpoint(model_name):
    pattern = f"{model_name}_epoch*.pth"
    files = glob.glob(os.path.join(MODELS_DIR, pattern))
    if not files:
        return None, 0
    def extract_epoch(fname):
        match = re.search(r'epoch(\d+)', fname)
        return int(match.group(1)) if match else -1
    latest = max(files, key=extract_epoch)
    return latest, extract_epoch(latest)


def cleanup_old_checkpoints(model_name, keep=MAX_CHECKPOINTS):
    pattern = f"{model_name}_epoch*.pth"
    files = glob.glob(os.path.join(MODELS_DIR, pattern))
    if len(files) <= keep:
        return
    def extract_epoch(fname):
        match = re.search(r'epoch(\d+)', fname)
        return int(match.group(1)) if match else -1
    files.sort(key=extract_epoch, reverse=True)
    for old_file in files[keep:]:
        try:
            os.remove(old_file)
        except OSError:
            pass


def save_checkpoints(epoch, encoder, decoder):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(encoder.state_dict(), get_model_path('encoder', epoch))
    torch.save(decoder.state_dict(), get_model_path('decoder', epoch))
    cleanup_old_checkpoints('encoder')
    cleanup_old_checkpoints('decoder')


def load_checkpoints_if_exist(encoder, decoder):
    loaded_epoch = 0
    for name, model, dev in [('encoder', encoder, ENCODER_DEVICE),
                             ('decoder', decoder, DECODER_DEVICE)]:
        path, epoch = find_latest_checkpoint(name)
        if path:
            state_dict = torch.load(path, map_location=dev, weights_only=False)
            model.load_state_dict(state_dict)
            print(f"Loaded {name} from epoch {epoch} on {dev}")
            if loaded_epoch == 0:
                loaded_epoch = epoch
            else:
                assert epoch == loaded_epoch, f"Epoch mismatch for {name}"
    return loaded_epoch


def apply_saved_gradients(model, saved_grads, optimizer):
    for name, param in model.named_parameters():
        if name in saved_grads and saved_grads[name] is not None:
            param.grad = saved_grads[name].to(param.device)
        else:
            param.grad = None
    optimizer.step()
    optimizer.zero_grad()


# ==================== Обучение с двумя фазами ====================
def train_epoch(encoder, decoder, train_loader, opt_enc, opt_dec, epoch):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (images, masks) in enumerate(train_loader):
        images_enc = images.to(ENCODER_DEVICE, non_blocking=True)
        masks_enc = masks.to(ENCODER_DEVICE, non_blocking=True)
        # Копии для декодера (на его устройстве)
        images_dec = images.to(DECODER_DEVICE, non_blocking=True)
        masks_dec = masks.to(DECODER_DEVICE, non_blocking=True)

        # ------------------------------------------------------------------
        # Фаза 1: обучение декодера (энкодер заморожен)
        # ------------------------------------------------------------------
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = True

        opt_dec.zero_grad()
        with torch.no_grad():
            pooled_enc = encoder(images_enc, masks_enc)               # на ENCODER_DEVICE
            pooled_dec = pooled_enc.to(DECODER_DEVICE)                # перенос на декодер
        reconstructed_1 = decoder(pooled_dec, masks_dec)
        loss_dec = masked_mse_loss(reconstructed_1, images_dec, masks_dec)

        loss_dec.backward()
        # Сохраняем градиенты декодера
        saved_grads_dec = {name: param.grad.clone().cpu() if param.grad is not None else None
                           for name, param in decoder.named_parameters()}
        opt_dec.zero_grad()

        # ------------------------------------------------------------------
        # Фаза 2: обучение энкодера (декодер заморожен)
        # ------------------------------------------------------------------
        for p in encoder.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = False

        opt_enc.zero_grad()
        pooled_enc = encoder(images_enc, masks_enc)
        pooled_dec = pooled_enc.to(DECODER_DEVICE)
        reconstructed_2 = decoder(pooled_dec, masks_dec)   # декодер не получит градиенты
        loss_enc = masked_mse_loss(reconstructed_2, images_dec, masks_dec)

        loss_enc.backward()

        # Применяем градиенты:
        # 1) восстанавливаем градиенты декодера и делаем шаг его оптимизатором
        apply_saved_gradients(decoder, saved_grads_dec, opt_dec)
        # 2) шаг энкодера (градиенты уже вычислены на ENCODER_DEVICE)
        opt_enc.step()
        opt_enc.zero_grad()

        total_loss += (loss_dec.item() + loss_enc.item()) / 2

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch}, batch {batch_idx+1}/{num_batches}, "
                  f"loss_dec: {loss_dec.item():.6f}, loss_enc: {loss_enc.item():.6f}")

        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return total_loss / num_batches


def validate(encoder, decoder, val_loader):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images_enc = images.to(ENCODER_DEVICE)
            masks_enc = masks.to(ENCODER_DEVICE)
            images_dec = images.to(DECODER_DEVICE)
            masks_dec = masks.to(DECODER_DEVICE)

            pooled_enc = encoder(images_enc, masks_enc)
            pooled_dec = pooled_enc.to(DECODER_DEVICE)
            reconstructed = decoder(pooled_dec, masks_dec)
            loss = masked_mse_loss(reconstructed, images_dec, masks_dec)
            total_loss += loss.item()
            total_psnr += compute_psnr(reconstructed, images_dec, masks_dec)
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    return avg_loss, avg_psnr


# ==================== Главный цикл ====================
def train():
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    all_files = sorted(
        [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.pt')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    if not all_files:
        raise RuntimeError(f"No .pt files in {DATASET_DIR}")
    print(f"Found {len(all_files)} samples.")

    if VALIDATION_SPLIT > 0:
        random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - VALIDATION_SPLIT))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
    else:
        train_files = all_files
        val_files = []
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_dataset = ImageDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = None
    if val_files:
        val_dataset = ImageDataset(val_files)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, pin_memory=True)

    encoder = Encoder(**ENCODER_CONFIG).to(ENCODER_DEVICE)
    decoder = Decoder(**DECODER_CONFIG).to(DECODER_DEVICE)

    opt_enc = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    opt_dec = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoints_if_exist(encoder, decoder) + 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        train_loss = train_epoch(encoder, decoder, train_loader, opt_enc, opt_dec, epoch)
        print(f"Epoch {epoch} average train loss: {train_loss:.6f}")

        if val_loader is not None and epoch % SAVE_EVERY_EPOCHS == 0:
            val_loss, val_psnr = validate(encoder, decoder, val_loader)
            print(f"  Validation loss: {val_loss:.6f}, PSNR: {val_psnr:.2f} dB")

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoints(epoch, encoder, decoder)

    print("Training completed.")


if __name__ == "__main__":
    train()