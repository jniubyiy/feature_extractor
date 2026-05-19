# training_VAEWrapper.py
"""
Обучение VAEWrapper напрямую на сжатых парнетах.
Оптимизируется восстановление сжатого парнета (L1).
После достижения порога W_Recon < KL_ENABLE_RECON_THRESHOLD
плавно добавляется KL‑регуляризация.
Множитель KL стартует с KL_START_WEIGHT и линейно растёт до KL_END_WEIGHT
в течение KL_WARMUP_EPOCHS, начиная со следующей эпохи после включения.
Первая эпоха прогрева (current_epoch == kl_enable_epoch) даёт множитель KL_START_WEIGHT,
последняя (current_epoch == kl_enable_epoch + KL_WARMUP_EPOCHS - 1) даёт KL_END_WEIGHT.
После прогрева действуют защиты:
- если KL батча <= KL_ZERO_THRESHOLD, вес KL обнуляется
- если KL батча < KL_TARGET_MIN, вес дополнительно снижается
"""

import os, re, glob, math, random, gc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from model_ParnetCompressor import ParnetDecompressor
from model_Autoencoder import Decoder
from model_VAEWrapper import VAEWrapper
from config_training_VAEWrapper import *
from config_training_models_Encoder_Decoder import DECODER_CONFIG
from config_training_models_Compressor_Decompressor import DECOMPRESSOR_CONFIG

DEVICE = torch.device(DEVICE)

# ---------------------- Загрузка замороженных моделей ----------------------
def _load_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint

def load_frozen_decompressor(checkpoint_path):
    model = ParnetDecompressor(**DECOMPRESSOR_CONFIG).to(DEVICE)
    state = _load_state_dict_from_checkpoint(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    )
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def load_frozen_decoder(checkpoint_path):
    model = Decoder(**DECODER_CONFIG).to(DEVICE)
    state = _load_state_dict_from_checkpoint(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    )
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

# ---------------------- Датасет ----------------------
class CompressedDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        c = data['compressed_parnet']
        return c, idx

def collate_fn(batch):
    compressed, indices = zip(*batch)
    compressed = torch.stack(compressed, dim=0)
    indices = torch.tensor(indices, dtype=torch.long)
    return compressed, indices

# ---------------------- Визуализация ----------------------
def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(2.0) - 10 * math.log10(mse.item())

def tensor_to_pil(t):
    arr = (t.cpu().clamp(-1, 1).numpy() + 1) / 2 * 255
    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(arr)

@torch.no_grad()
def save_visualization(c_hat, c, decompressor, decoder, out_dir, eid):
    os.makedirs(out_dir, exist_ok=True)
    parnet_hat = decompressor(c_hat)
    parnet_orig = decompressor(c)
    img_hat = decoder(parnet_hat)
    img_orig = decoder(parnet_orig)
    tensor_to_pil(img_hat.squeeze(0)).save(os.path.join(out_dir, f"reconstructed_{eid}.png"))
    tensor_to_pil(img_orig.squeeze(0)).save(os.path.join(out_dir, f"original_{eid}.png"))
    diff = (img_hat - img_orig).abs()
    tensor_to_pil(diff.squeeze(0)).save(os.path.join(out_dir, f"difference_{eid}.png"))
    l1_img = F.l1_loss(img_hat, img_orig).item()
    psnr_img = compute_psnr(img_hat, img_orig)
    with open(os.path.join(out_dir, f"metrics_{eid}.txt"), 'w') as f:
        f.write(f"Image L1: {l1_img:.6f}\nPSNR: {psnr_img:.2f} dB\n")

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

def save_checkpoint(epoch, model, optimizer, kl_enabled, kl_enable_epoch):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'kl_enabled': kl_enabled,
        'kl_enable_epoch': kl_enable_epoch,
    }, get_model_path("vae_wrapper", epoch))
    cleanup_old_checkpoints()

def load_checkpoint_if_exist(model, optimizer):
    path, epoch = find_latest_checkpoint()
    kl_enabled = False
    kl_enable_epoch = 0
    if path:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        kl_enabled = ckpt.get('kl_enabled', False)
        kl_enable_epoch = ckpt.get('kl_enable_epoch', 0)
        print(f"Loaded VAEWrapper from epoch {epoch}, KL enabled: {kl_enabled}, enable epoch: {kl_enable_epoch}")
        return epoch, kl_enabled, kl_enable_epoch
    return 0, kl_enabled, kl_enable_epoch

# ---------------------- Валидация ----------------------
@torch.no_grad()
def evaluate_and_visualize(model, decompressor, decoder, dataset, output_base, epoch, num_examples, seed=None):
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    for idx in indices:
        c, _ = dataset[idx]
        c = c.unsqueeze(0).to(DEVICE)
        c_hat, mu, logvar = model(c)
        recon_l1 = F.l1_loss(c_hat, c).item()
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / c.size(0)
        kld = kld.item()
        eid = os.path.splitext(os.path.basename(dataset.files[idx]))[0]
        base_dir = os.path.join(output_base, f"epoch_{epoch}", f"example_{eid}")
        save_visualization(c_hat, c, decompressor, decoder, base_dir, eid)
        with open(os.path.join(base_dir, f"metrics_{eid}.txt"), 'a') as f:
            f.write(f"Compressed L1: {recon_l1:.6f}\n")
            f.write(f"KL: {kld:.6f}\n")

# ---------------------- Обучение эпохи ----------------------
def train_epoch(model, train_loader, optimizer, kl_enabled, kl_enable_epoch, current_epoch):
    model.train()
    total_w_recon = 0.0
    total_w_kld = 0.0
    total_loss_sum = 0.0
    n_batches = 0

    for batch_idx, (c, indices) in enumerate(train_loader):
        c = c.to(DEVICE)

        params = model.head(c)
        mu, logvar = params.chunk(2, dim=1)
        z = model.reparameterize(mu, logvar)
        c_hat = model.tail(z)

        recon_loss = F.l1_loss(c_hat, c)
        w_recon = RECON_LOSS_WEIGHT * recon_loss

        if USE_KL_LOSS and kl_enabled:
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / c.size(0)
            kld_value = kld_loss.item()

            if kld_value <= KL_ZERO_THRESHOLD:
                kl_multiplier = 0.0
                if KL_ZERO_THRESHOLD > 0:
                    print(f"KL {kld_value:.6f} <= {KL_ZERO_THRESHOLD}, KL loss disabled for this batch")
            else:
                # Прогресс от 0 до 1 за (KL_WARMUP_EPOCHS - 1) шагов, если KL_WARMUP_EPOCHS > 1
                if KL_WARMUP_EPOCHS > 1:
                    progress = min(1.0, (current_epoch - kl_enable_epoch) / (KL_WARMUP_EPOCHS - 1))
                else:
                    progress = 1.0
                base_multiplier = KL_START_WEIGHT + (KL_END_WEIGHT - KL_START_WEIGHT) * progress

                # Защита от низкого KL после завершения прогрева
                if progress >= 1.0 and kld_value < KL_TARGET_MIN:
                    ratio = kld_value / KL_TARGET_MIN
                    protective_multiplier = max(KL_WEIGHT_MIN, ratio ** KL_ADAPTIVE_POWER)
                    kl_multiplier = base_multiplier * protective_multiplier
                else:
                    kl_multiplier = base_multiplier

            w_kld = kl_multiplier * kld_loss
            total_loss = w_recon + w_kld
        else:
            w_kld = torch.tensor(0.0, device=DEVICE)
            total_loss = w_recon

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_w_recon += w_recon.item()
        total_w_kld += w_kld.item()
        total_loss_sum += total_loss.item()
        n_batches += 1

        if USE_KL_LOSS and kl_enabled:
            print(f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"W_Recon: {w_recon.item():.6f} | W_KL: {w_kld.item():.6f} (mult: {kl_multiplier:.6f}) | "
                  f"Loss: {total_loss.item():.6f}")
        else:
            print(f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"W_Recon: {w_recon.item():.6f} | Loss: {total_loss.item():.6f}")

        del c, mu, logvar, z, c_hat, total_loss
        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    avg_w_recon = total_w_recon / n_batches
    avg_w_kld = total_w_kld / n_batches
    avg_loss = total_loss_sum / n_batches
    return avg_w_recon, avg_w_kld, avg_loss

# ---------------------- main ----------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print("Loading frozen decompressor...")
    decompressor = load_frozen_decompressor(DECOMPRESSOR_CHECKPOINT)
    print("Loading frozen decoder...")
    decoder = load_frozen_decoder(DECODER_CHECKPOINT)

    all_files = sorted(
        [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.pt')],
        key=lambda x: os.path.basename(x)
    )
    if not all_files:
        raise RuntimeError(f"No .pt files in {DATASET_DIR}")
    print(f"Found {len(all_files)} compressed samples.")

    if MAX_TRAIN_IMAGES and MAX_TRAIN_IMAGES > 0:
        train_files = all_files[:MAX_TRAIN_IMAGES]
        start_val = len(train_files)
        val_files = all_files[start_val:start_val + VALIDATION_SPLIT] if start_val < len(all_files) else []
    else:
        n_val = min(VALIDATION_SPLIT, len(all_files))
        train_files = all_files[:-n_val] if n_val < len(all_files) else []
        val_files = all_files[-n_val:] if n_val > 0 else []

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    train_dataset = CompressedDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=0)
    val_dataset = None
    if val_files:
        val_dataset = CompressedDataset(val_files)

    model = VAEWrapper(COMPRESSED_CHANNELS, STOCHASTIC_PARNET_DIM,
                       hidden_dim=HIDDEN_DIM, num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_epoch, kl_enabled, kl_enable_epoch = load_checkpoint_if_exist(model, optimizer)
    start_epoch += 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_w_recon, avg_w_kld, avg_loss = train_epoch(
            model, train_loader, optimizer, kl_enabled, kl_enable_epoch, epoch
        )

        if not kl_enabled and USE_KL_LOSS:
            if avg_w_recon < KL_ENABLE_RECON_THRESHOLD:
                kl_enabled = True
                kl_enable_epoch = epoch + 1
                print(f">>> KL loss enabled at epoch {epoch} (W_Recon={avg_w_recon:.4f}), "
                      f"warmup starts at epoch {kl_enable_epoch}")

        if USE_KL_LOSS and kl_enabled:
            print(f"Epoch {epoch:3d} | W_Recon: {avg_w_recon:.6f} | W_KL: {avg_w_kld:.6f} | Loss: {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch:3d} | W_Recon: {avg_w_recon:.6f} | Loss: {avg_loss:.6f}")

        if val_dataset and epoch % VAL_EVERY_EPOCHS == 0:
            print("Running validation...")
            evaluate_and_visualize(model, decompressor, decoder, val_dataset,
                                   VAL_TESTS_DIR, epoch, NUM_TEST_EXAMPLES, TEST_SEED)

        if epoch % TEST_EVERY_EPOCHS == 0:
            print("Running tests...")
            evaluate_and_visualize(model, decompressor, decoder, train_dataset,
                                   TESTS_DIR, epoch, NUM_TEST_EXAMPLES, TEST_SEED)

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoint(epoch, model, optimizer, kl_enabled, kl_enable_epoch)
            print(f"Checkpoint saved at epoch {epoch}")

    print("Training completed.")

if __name__ == "__main__":
    main()