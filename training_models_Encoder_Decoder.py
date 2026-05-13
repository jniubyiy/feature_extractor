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
import numpy as np
from PIL import Image

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
    return 20 * math.log10(1.0) - 10 * math.log10(mse.item())


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


# ==================== Функции валидации (как в примере) ====================
def evaluate_reconstruction(encoder, decoder, loader):
    """Считает средние MSE и PSNR по всему loader без градиентов."""
    encoder.eval()
    decoder.eval()
    total_mse = 0.0
    total_psnr = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, masks in loader:
            images_enc = images.to(ENCODER_DEVICE)
            masks_enc = masks.to(ENCODER_DEVICE)
            images_dec = images.to(DECODER_DEVICE)
            masks_dec = masks.to(DECODER_DEVICE)

            pooled_enc = encoder(images_enc, masks_enc)
            pooled_dec = pooled_enc.to(DECODER_DEVICE)
            reconstructed = decoder(pooled_dec, masks_dec)

            mse = masked_mse_loss(reconstructed, images_dec, masks_dec)
            psnr = compute_psnr(reconstructed, images_dec, masks_dec)

            total_mse += mse.item()
            total_psnr += psnr
            num_batches += 1

    encoder.train()
    decoder.train()
    return total_mse / num_batches, total_psnr / num_batches


def validate(encoder, decoder, val_loader):
    """Возвращает словарь с метриками валидации в стиле примера."""
    mse, psnr = evaluate_reconstruction(encoder, decoder, val_loader)
    return {'mse': mse, 'psnr': psnr}


# ==================== Сохранение тестовых примеров ====================
def tensor_to_pil(img_tensor):
    """Преобразует тензор (C, H, W) со значениями [0,1] в PIL Image RGB."""
    arr = (img_tensor.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


def save_test_example(epoch, example_idx, original_image, mask, reconstructed, metrics):
    """
    Сохраняет в ./tests/epoch_N/example_M/ изображения и метрики.
    original_image, mask, reconstructed: тензоры (3,H,W) или (1,H,W) в [0,1].
    metrics: dict с ключами 'mse', 'psnr'.
    """
    base_dir = os.path.join(TESTS_DIR, f"epoch_{epoch}", f"example_{example_idx}")
    os.makedirs(base_dir, exist_ok=True)

    # Сохраняем оригинал
    orig_pil = tensor_to_pil(original_image)
    orig_pil.save(os.path.join(base_dir, "original.png"))

    # Маска в grayscale (сохраняем как изображение, для наглядности)
    mask_np = mask.cpu().squeeze().numpy()
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
    mask_img.save(os.path.join(base_dir, "mask.png"))

    # Реконструированное изображение
    recon_pil = tensor_to_pil(reconstructed)
    recon_pil.save(os.path.join(base_dir, "reconstructed.png"))

    # Разница (усиленная для видимости)
    diff = (reconstructed - original_image).abs()
    diff_pil = tensor_to_pil(diff)
    diff_pil.save(os.path.join(base_dir, "difference.png"))

    # Метрики в текстовый файл
    with open(os.path.join(base_dir, "metrics.txt"), 'w') as f:
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")


def run_tests(encoder, decoder, dataset, epoch):
    """Выбирает NUM_TEST_EXAMPLES случайных индексов из dataset, прогоняет и сохраняет."""
    encoder.eval()
    decoder.eval()
    indices = random.sample(range(len(dataset)), min(NUM_TEST_EXAMPLES, len(dataset)))

    for idx in indices:
        image, mask = dataset[idx]  # (3,H,W), (1,H,W) на CPU
        example_id = int(os.path.splitext(os.path.basename(dataset.files[idx]))[0])

        # Перенос на устройства
        img_enc = image.unsqueeze(0).to(ENCODER_DEVICE)
        msk_enc = mask.unsqueeze(0).to(ENCODER_DEVICE)
        img_dec = image.unsqueeze(0).to(DECODER_DEVICE)
        msk_dec = mask.unsqueeze(0).to(DECODER_DEVICE)

        with torch.no_grad():
            pooled = encoder(img_enc, msk_enc)
            pooled_dec = pooled.to(DECODER_DEVICE)
            recon = decoder(pooled_dec, msk_dec)  # (1,3,H,W)

        mse_val = masked_mse_loss(recon, img_dec, msk_dec).item()
        psnr_val = compute_psnr(recon, img_dec, msk_dec)

        # Сохраняем с оригинальными тензорами на CPU (убираем batch)
        save_test_example(epoch, example_id, image, mask.squeeze(0), recon.squeeze(0).cpu(),
                          {'mse': mse_val, 'psnr': psnr_val})

    encoder.train()
    decoder.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== Обучение с двумя фазами ====================
def train_epoch(encoder, decoder, train_loader, opt_enc, opt_dec, epoch):
    encoder.train()
    decoder.train()
    total_loss_dec = 0.0
    total_loss_enc = 0.0
    num_batches = len(train_loader)

    for batch_idx, (images, masks) in enumerate(train_loader):
        images_enc = images.to(ENCODER_DEVICE, non_blocking=True)
        masks_enc = masks.to(ENCODER_DEVICE, non_blocking=True)
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
            pooled_enc = encoder(images_enc, masks_enc)
            pooled_dec = pooled_enc.to(DECODER_DEVICE)
        reconstructed_1 = decoder(pooled_dec, masks_dec)
        loss_dec = masked_mse_loss(reconstructed_1, images_dec, masks_dec)

        loss_dec.backward()
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
        reconstructed_2 = decoder(pooled_dec, masks_dec)
        loss_enc = masked_mse_loss(reconstructed_2, images_dec, masks_dec)

        loss_enc.backward()

        # Применяем градиенты
        apply_saved_gradients(decoder, saved_grads_dec, opt_dec)
        opt_enc.step()
        opt_enc.zero_grad()

        total_loss_dec += loss_dec.item()
        total_loss_enc += loss_enc.item()

        print(f"Batch {batch_idx+1}/{num_batches}")
        print(f"LossDec: {loss_dec.item():.6f}")
        print(f"LossEnc: {loss_enc.item():.6f}")

        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return {'dec': total_loss_dec / num_batches, 'enc': total_loss_enc / num_batches}


# ==================== Главный цикл ====================
def train():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
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
        num_val = max(1, int(len(all_files) * VALIDATION_SPLIT))
        train_files = all_files[:-num_val] if num_val < len(all_files) else []
        val_files = all_files[-num_val:]
    else:
        train_files = all_files
        val_files = []
    print(f"Train: {len(train_files)}, Val: {len(val_files)} (last {len(val_files)} files)")

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
        print(f"\n--- Epoch {epoch} ---")
        losses = train_epoch(encoder, decoder, train_loader, opt_enc, opt_dec, epoch)
        print(f"Epoch {epoch:3d} TRAIN SUMMARY")
        print(f"LossDec: {losses['dec']:.6f}")
        print(f"LossEnc: {losses['enc']:.6f}")

        if val_loader is not None and epoch % VAL_EVERY_EPOCHS == 0:
            val_metrics = validate(encoder, decoder, val_loader)
            print(f"Epoch {epoch:3d} VAL")
            print(f"MSE: {val_metrics['mse']:.6f}")
            print(f"PSNR: {val_metrics['psnr']:.2f} dB")

        if epoch % TEST_EVERY_EPOCHS == 0:
            print(f"Running test examples for epoch {epoch}...")
            run_tests(encoder, decoder, train_dataset, epoch)

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoints(epoch, encoder, decoder)

    print("Training completed.")


if __name__ == "__main__":
    train()