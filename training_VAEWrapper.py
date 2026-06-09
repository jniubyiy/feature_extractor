# training_VAEWrapper.py
"""
Двухфазное обучение на сжатых парнетах с контролем детерминированности seed_tensor.
При замороженном энкодере seed_tensor для одного и того же файла не должен меняться.
"""

import os, re, glob, math, random, gc, json
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from model_ParnetCompressor import ParnetDecompressor
from model_Autoencoder import Decoder
from model_VAEWrapper import (
    StochasticEncoder, StochasticDecoder,
    reparameterize
)
from config_training_VAEWrapper import *
from config_training_models_Encoder_Decoder import DECODER_CONFIG
from config_training_models_Compressor_Decompressor import DECOMPRESSOR_CONFIG

# Резервные значения для отсутствующих параметров
try:
    MSE_LOSS_WEIGHT
except NameError:
    MSE_LOSS_WEIGHT = 1.0
try:
    HYBRID_LOSS_WEIGHT
except NameError:
    HYBRID_LOSS_WEIGHT = 1.0

DEVICE = torch.device(DEVICE)

# Глобальный реестр для проверки стабильности seed_tensor
SEED_REGISTRY = {}

def check_seed_consistency(file_path, seed_tensor, registry, phase="train"):
    """
    Сравнивает текущий seed_tensor с сохранённым для этого же файла.
    Если значение изменилось (абсолютная разница > 1e-6), выводит предупреждение.
    """
    if file_path in registry:
        ref = registry[file_path]
        if not torch.allclose(seed_tensor, ref, atol=1e-6):
            max_diff = (seed_tensor - ref).abs().max().item()
            print(f"⚠️ WARNING [{phase}]: seed_tensor изменился для {os.path.basename(file_path)}! "
                  f"Макс. разница: {max_diff:.6e}")
            registry[file_path] = seed_tensor
    else:
        registry[file_path] = seed_tensor

# ----------------------------------------------------------------------
# Загрузка замороженных моделей (для визуализации)
# ----------------------------------------------------------------------
def _load_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint

def load_frozen_decompressor(checkpoint_path):
    model = ParnetDecompressor(**DECOMPRESSOR_CONFIG).to(DEVICE)
    state = _load_state_dict_from_checkpoint(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False))
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model

def load_frozen_decoder(checkpoint_path):
    model = Decoder(**DECODER_CONFIG).to(DEVICE)
    state = _load_state_dict_from_checkpoint(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False))
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model

# ----------------------------------------------------------------------
# Датасет (только сжатые парнеты)
# ----------------------------------------------------------------------
class CompressedOnlyDataset(Dataset):
    """Загружает сжатые парнеты, возвращает (compressed, compressed) как вход и цель."""
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        compressed = data['compressed_parnet']   # [4, H, W]
        return compressed, compressed, idx

def collate_fn(batch):
    inputs, targets, indices = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    indices = torch.tensor(indices, dtype=torch.long)
    return inputs, targets, indices

# ----------------------------------------------------------------------
# Функции потерь
# ----------------------------------------------------------------------
def difference_loss(pred, target):
    return torch.mean(torch.log(1.0 + torch.abs(pred - target)))

def diff_smooth_loss(pred, target):
    diff = torch.abs(pred - target)
    d_h = diff[:, :, 1:, :] - diff[:, :, :-1, :]
    d_w = diff[:, :, :, 1:] - diff[:, :, :, :-1]
    return d_h.abs().mean() + d_w.abs().mean()

def mse_loss(pred, target):
    return F.mse_loss(pred, target)

def hybrid_loss(pred, target):
    """Гибридная потеря: (x^2 * ln(1 + |x|)) / 2, усреднённая по всем элементам."""
    diff = pred - target
    loss = (diff ** 2) * torch.log(1.0 + torch.abs(diff)) / 2.0
    return loss.mean()

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0: return float('inf')
    return 20 * math.log10(2.0) - 10 * math.log10(mse.item())

# ----------------------------------------------------------------------
# Визуализация
# ----------------------------------------------------------------------
def tensor_to_pil(t):
    arr = (t.cpu().clamp(-1, 1).numpy() + 1) / 2 * 255
    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(arr)

def save_json(tensor, path):
    with open(path, 'w') as f:
        json.dump(tensor.squeeze(0).cpu().tolist(), f)

@torch.no_grad()
def save_visualizations(eid, out_dir, compressed, mu, seed_tensor, z, decoded,
                        decompressor, frozen_decoder):
    os.makedirs(out_dir, exist_ok=True)

    if compressed is not None:
        img_orig = frozen_decoder(decompressor(compressed.to(DEVICE)))
        tensor_to_pil(img_orig[0]).save(os.path.join(out_dir, "original.png"))

    img_z = frozen_decoder(decompressor(z.to(DEVICE)))
    tensor_to_pil(img_z[0]).save(os.path.join(out_dir, "z.png"))

    img_dec = frozen_decoder(decompressor(decoded.to(DEVICE)))
    tensor_to_pil(img_dec[0]).save(os.path.join(out_dir, "decoded.png"))

    img_mu = frozen_decoder(decompressor(mu.to(DEVICE)))
    tensor_to_pil(img_mu[0]).save(os.path.join(out_dir, "mu.png"))

    if compressed is not None:
        diff_dec = (img_dec - img_orig).abs()
        tensor_to_pil(diff_dec[0]).save(os.path.join(out_dir, "diff_decoded.png"))

    save_json(mu, os.path.join(out_dir, "mu.json"))
    save_json(z, os.path.join(out_dir, "z.json"))
    seed_list = seed_tensor.squeeze(0).squeeze(-1).squeeze(-1).cpu().tolist()
    with open(os.path.join(out_dir, "seed_tensor.json"), 'w') as f:
        json.dump(seed_list, f)

    with open(os.path.join(out_dir, "metrics.txt"), 'w') as f:
        f.write(f"Seed mean: {seed_tensor.mean().item():.4f}\n")
        if compressed is not None:
            l1_dec = F.l1_loss(img_dec, img_orig).item()
            psnr_dec = compute_psnr(img_dec, img_orig)
            f.write(f"Decoded L1: {l1_dec:.6f}, PSNR: {psnr_dec:.2f} dB\n")

# ----------------------------------------------------------------------
# Чекпоинты
# ----------------------------------------------------------------------
def get_model_path(name, epoch):
    return os.path.join(MODELS_DIR, f"{name}_epoch{epoch}.pth")

def find_latest_checkpoint():
    enc_files = glob.glob(os.path.join(MODELS_DIR, "encoder_epoch*.pth"))
    if not enc_files:
        return None
    def extract_epoch(f):
        m = re.search(r'epoch(\d+)', f)
        return int(m.group(1)) if m else -1
    epochs = sorted([extract_epoch(f) for f in enc_files], reverse=True)
    for epoch in epochs:
        dec_path = get_model_path("decoder", epoch)
        if os.path.exists(dec_path):
            return epoch
    return None

def cleanup_old_checkpoints(keep=MAX_CHECKPOINTS):
    for name in ["encoder", "decoder"]:
        files = glob.glob(os.path.join(MODELS_DIR, f"{name}_epoch*.pth"))
        if len(files) <= keep: continue
        files.sort(key=lambda f: int(re.search(r'epoch(\d+)', f).group(1)), reverse=True)
        for old in files[keep:]:
            try: os.remove(old)
            except OSError: pass

def save_checkpoint(epoch, encoder, decoder, opt_enc, opt_dec):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder._orig_mod.state_dict(),
        'optimizer_state_dict': opt_enc.state_dict(),
    }, get_model_path("encoder", epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder._orig_mod.state_dict(),
        'optimizer_state_dict': opt_dec.state_dict(),
    }, get_model_path("decoder", epoch))
    cleanup_old_checkpoints()

def load_checkpoint_if_exist(encoder, decoder, opt_enc, opt_dec):
    epoch = find_latest_checkpoint()
    if epoch is not None:
        enc_path = get_model_path("encoder", epoch)
        enc_ckpt = torch.load(enc_path, map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(enc_ckpt['model_state_dict'])
        opt_enc.load_state_dict(enc_ckpt['optimizer_state_dict'])
        dec_path = get_model_path("decoder", epoch)
        dec_ckpt = torch.load(dec_path, map_location=DEVICE, weights_only=False)
        decoder.load_state_dict(dec_ckpt['model_state_dict'])
        opt_dec.load_state_dict(dec_ckpt['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch
    else:
        print("No checkpoints found, starting from scratch.")
        return 0

# ----------------------------------------------------------------------
# Валидация / тесты с проверкой seed_tensor
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_and_visualize(encoder, decoder, decompressor, frozen_decoder,
                           dataset, output_base, epoch, num_examples, seed=None):
    if seed is not None: random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    for idx in indices:
        compressed, _, _ = dataset[idx]
        file_path = dataset.files[idx]
        compressed = compressed.unsqueeze(0).to(DEVICE)

        mu, seed_tensor = encoder(compressed)
        check_seed_consistency(file_path, seed_tensor.detach().cpu(), SEED_REGISTRY, phase="eval")

        if STOCHASTIC_MODE:
            z, _, _ = reparameterize(mu, NOISE_RANGE, STOCHASTIC_STRENGTH, seed_tensor)
        else:
            z = mu
        decoded = decoder(z, seed_tensor)

        eid = os.path.splitext(os.path.basename(file_path))[0]
        base_dir = os.path.join(output_base, f"epoch_{epoch}", f"example_{eid}")
        save_visualizations(eid, base_dir, compressed, mu, seed_tensor, z, decoded,
                            decompressor, frozen_decoder)

# ----------------------------------------------------------------------
# Тренировочная эпоха с проверкой seed_tensor в фазе 1
# ----------------------------------------------------------------------
def train_epoch(encoder, decoder, train_loader, opt_enc, opt_dec):
    encoder.train(); decoder.train()
    total_loss1 = 0.0; total_loss2 = 0.0
    total_ds1 = 0.0; total_ds2 = 0.0
    total_mse1 = 0.0; total_mse2 = 0.0
    total_hyb1 = 0.0; total_hyb2 = 0.0
    total_mu_reg = 0.0
    n_batches = 0

    for batch_idx, (inputs, targets, indices) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        saved_grads_enc = {}; saved_grads_dec = {}

        # ===== Фаза 1: Декодер (энкодер заморожен) =====
        for p in encoder.parameters(): p.requires_grad = False
        for p in decoder.parameters(): p.requires_grad = True
        opt_dec.zero_grad()

        with torch.no_grad():
            mu, seed_tensor = encoder(inputs)

        # Проверка стабильности seed_tensor для каждого элемента батча
        for i in range(len(indices)):
            file_path = train_loader.dataset.files[indices[i].item()]
            check_seed_consistency(file_path, seed_tensor[i:i+1].detach().cpu(),
                                   SEED_REGISTRY, phase="train_phase1")

        loss1_sum = 0.0; ds1_sum = 0.0; mse1_sum = 0.0; hyb1_sum = 0.0
        for _ in range(NUM_MC_SAMPLES):
            if STOCHASTIC_MODE:
                z, _, _ = reparameterize(mu, NOISE_RANGE, STOCHASTIC_STRENGTH, seed_tensor)
            else:
                z = mu
            decoded = decoder(z, seed_tensor)
            rec = RECON_LOSS_WEIGHT * difference_loss(decoded, targets)
            smooth = DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(decoded, targets)
            mse = MSE_LOSS_WEIGHT * mse_loss(decoded, targets)
            hyb = HYBRID_LOSS_WEIGHT * hybrid_loss(decoded, targets)
            loss1_sum += rec + smooth + mse + hyb
            ds1_sum += smooth
            mse1_sum += mse
            hyb1_sum += hyb

        loss1_avg = loss1_sum / NUM_MC_SAMPLES
        loss1_avg.backward()
        for name, param in decoder.named_parameters():
            if param.grad is not None: saved_grads_dec[name] = param.grad.clone().cpu()
        opt_dec.zero_grad()
        loss1_val = loss1_avg.item()
        ds1_val = ds1_sum.item() / NUM_MC_SAMPLES
        mse1_val = mse1_sum.item() / NUM_MC_SAMPLES
        hyb1_val = hyb1_sum.item() / NUM_MC_SAMPLES

        # ===== Фаза 2: Энкодер =====
        for p in encoder.parameters(): p.requires_grad = True
        for p in decoder.parameters(): p.requires_grad = False
        opt_enc.zero_grad()

        mu, seed_tensor = encoder(inputs)
        loss2_sum = 0.0; ds2_sum = 0.0; mse2_sum = 0.0; hyb2_sum = 0.0
        for _ in range(NUM_MC_SAMPLES):
            if STOCHASTIC_MODE:
                z, _, _ = reparameterize(mu, NOISE_RANGE, STOCHASTIC_STRENGTH, seed_tensor)
            else:
                z = mu
            decoded = decoder(z, seed_tensor)
            rec = RECON_LOSS_WEIGHT * difference_loss(decoded, targets)
            smooth = DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(decoded, targets)
            mse = MSE_LOSS_WEIGHT * mse_loss(decoded, targets)
            hyb = HYBRID_LOSS_WEIGHT * hybrid_loss(decoded, targets)
            loss2_sum += rec + smooth + mse + hyb
            ds2_sum += smooth
            mse2_sum += mse
            hyb2_sum += hyb

        mu_reg = encoder._orig_mod.mu_regularization(mu)
        mu_weighted = MU_LOSS_WEIGHT * mu_reg
        loss2_avg = loss2_sum / NUM_MC_SAMPLES + mu_weighted
        loss2_avg.backward()
        for name, param in encoder.named_parameters():
            if param.grad is not None: saved_grads_enc[name] = param.grad.clone().cpu()
        opt_enc.zero_grad()
        loss2_val = loss2_avg.item()
        ds2_val = ds2_sum.item() / NUM_MC_SAMPLES
        mse2_val = mse2_sum.item() / NUM_MC_SAMPLES
        hyb2_val = hyb2_sum.item() / NUM_MC_SAMPLES
        mu_val = mu_weighted.item()

        # Применяем градиенты фаз 1 и 2
        for name, param in decoder.named_parameters():
            if name in saved_grads_dec: param.grad = saved_grads_dec[name].to(param.device)
            else: param.grad = None
        opt_dec.step(); opt_dec.zero_grad()

        for name, param in encoder.named_parameters():
            if name in saved_grads_enc: param.grad = saved_grads_enc[name].to(param.device)
            else: param.grad = None
        opt_enc.step(); opt_enc.zero_grad()

        saved_grads_dec.clear(); saved_grads_enc.clear()

        total_loss1 += loss1_val; total_loss2 += loss2_val
        total_ds1 += ds1_val; total_ds2 += ds2_val
        total_mse1 += mse1_val; total_mse2 += mse2_val
        total_hyb1 += hyb1_val; total_hyb2 += hyb2_val
        total_mu_reg += mu_val
        n_batches += 1

        print(f"Batch {batch_idx+1}/{len(train_loader)} | "
              f"Ph1(Dec): {loss1_val:.6f} (MSE:{mse1_val:.6f} HYB:{hyb1_val:.6f}) | "
              f"Ph2(Enc): {loss2_val:.6f} (MSE:{mse2_val:.6f} HYB:{hyb2_val:.6f}) | "
              f"Mu: {mu_val:.6f} | DSm1: {ds1_val:.4f} DSm2: {ds2_val:.4f}")

        del inputs, targets, mu, seed_tensor, z, decoded
        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache(); gc.collect()

    return (total_loss1/n_batches, total_loss2/n_batches,
            total_ds1/n_batches, total_ds2/n_batches,
            total_mse1/n_batches, total_mse2/n_batches,
            total_hyb1/n_batches, total_hyb2/n_batches,
            total_mu_reg/n_batches)

# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------
def main():
    global SEED_REGISTRY
    torch.manual_seed(RANDOM_SEED); random.seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)

    print("Loading frozen decompressor...")
    decompressor = load_frozen_decompressor(DECOMPRESSOR_CHECKPOINT)
    print("Loading frozen decoder...")
    frozen_decoder = load_frozen_decoder(DECODER_CHECKPOINT)

    all_files = sorted(Path(COMPRESSED_DATASET_DIR).glob("*.pt"))
    all_files = [f for f in all_files if f.name != "similarities.pt"]
    print(f"Found {len(all_files)} compressed parnet samples.")
    if not all_files:
        raise RuntimeError("No compressed parnet files found.")

    if MAX_TRAIN_IMAGES and MAX_TRAIN_IMAGES > 0:
        train_files = all_files[:MAX_TRAIN_IMAGES]
        start_val = len(train_files)
        val_files = all_files[start_val:start_val + VALIDATION_SPLIT] if start_val < len(all_files) else []
    else:
        n_val = min(VALIDATION_SPLIT, len(all_files))
        train_files = all_files[:-n_val] if n_val < len(all_files) else []
        val_files = all_files[-n_val:] if n_val > 0 else []

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    train_dataset = CompressedOnlyDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=0)
    val_dataset = CompressedOnlyDataset(val_files) if val_files else None

    encoder = StochasticEncoder(**STOCHASTIC_ENCODER_CONFIG).to(DEVICE)
    decoder = StochasticDecoder(**STOCHASTIC_DECODER_CONFIG).to(DEVICE)

    opt_enc = optim.Adam(encoder.parameters(), lr=LEARNING_RATE_ENCODER)
    opt_dec = optim.Adam(decoder.parameters(), lr=LEARNING_RATE_DECODER)

    start_epoch = load_checkpoint_if_exist(encoder, decoder, opt_enc, opt_dec) + 1

    encoder = torch.compile(encoder, backend="aot_eager")
    decoder = torch.compile(decoder, backend="aot_eager")

    print(f"STOCHASTIC_MODE = {STOCHASTIC_MODE} | "
          f"STOCHASTIC_STRENGTH = {STOCHASTIC_STRENGTH} | "
          f"NOISE_RANGE = {NOISE_RANGE} | "
          f"Effective noise limit = ±{NOISE_RANGE * STOCHASTIC_STRENGTH:.4f}")
    print(f"MU_LOSS_WEIGHT = {MU_LOSS_WEIGHT} | NUM_MC_SAMPLES = {NUM_MC_SAMPLES} | "
          f"MSE_LOSS_WEIGHT = {MSE_LOSS_WEIGHT} | HYBRID_LOSS_WEIGHT = {HYBRID_LOSS_WEIGHT}")
    print("Seed consistency check enabled for frozen encoder phases.")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_l1, avg_l2, avg_ds1, avg_ds2, avg_mse1, avg_mse2, avg_hyb1, avg_hyb2, avg_mu = train_epoch(
            encoder, decoder, train_loader, opt_enc, opt_dec
        )
        print(f"Epoch {epoch:3d} | "
              f"Ph1: {avg_l1:.6f} (MSE:{avg_mse1:.6f} HYB:{avg_hyb1:.6f}) | "
              f"Ph2: {avg_l2:.6f} (MSE:{avg_mse2:.6f} HYB:{avg_hyb2:.6f}) | "
              f"Mu: {avg_mu:.6f}")

        if val_dataset and epoch % VAL_EVERY_EPOCHS == 0:
            print("Running validation...")
            evaluate_and_visualize(encoder, decoder, decompressor, frozen_decoder,
                                   val_dataset, VAL_TESTS_DIR, epoch, NUM_TEST_EXAMPLES, TEST_SEED)

        if epoch % TEST_EVERY_EPOCHS == 0:
            print("Running tests...")
            evaluate_and_visualize(encoder, decoder, decompressor, frozen_decoder,
                                   train_dataset, TESTS_DIR, epoch, NUM_TEST_EXAMPLES, TEST_SEED)

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoint(epoch, encoder, decoder, opt_enc, opt_dec)
            print(f"Checkpoint saved at epoch {epoch}")

    print("Training completed.")

if __name__ == "__main__":
    main()