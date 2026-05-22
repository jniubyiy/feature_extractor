# training_VAEWrapper.py
"""
Двухфазное обучение StochasticEncoder и StochasticDecoder.
Градиенты сохраняются в RAM и применяются после обеих фаз.
Потери:
  - difference_loss (log(1+|diff|)) – реконструкция сжатого парнета
  - diff_smooth_loss – штраф за неоднородность разности сжатых парнетов
  - parnet_quality_loss – качество стохастического парнета (плавность, нормализация и т.д.)
  - KL-регуляризация (при STOCHASTIC_MODE и USE_KL_LOSS)
Визуализация включает стохастическое изображение (z -> decompressor -> decoder).
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
from model_VAEWrapper import StochasticEncoder, StochasticDecoder
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


# ---------------------- Функции потерь ----------------------
def difference_loss(pred, target):
    """Устойчивая потеря: log(1 + |pred - target|)."""
    return torch.mean(torch.log(1.0 + torch.abs(pred - target)))


def diff_smooth_loss(pred, target):
    """Штраф за резкие перепады в карте разности (аналог для изображений)."""
    diff = torch.abs(pred - target)
    d_h = diff[:, :, 1:, :] - diff[:, :, :-1, :]
    d_w = diff[:, :, :, 1:] - diff[:, :, :, :-1]
    return d_h.abs().mean() + d_w.abs().mean()


def parnet_quality_loss(parnet, weights, num_hist_bins=100):
    """Качество парнета: плавность, нормализация, отсутствие выбросов."""
    diff_h = parnet[:, :, 1:, :] - parnet[:, :, :-1, :]
    diff_w = parnet[:, :, :, 1:] - parnet[:, :, :, :-1]
    smooth_loss = diff_h.abs().mean() + diff_w.abs().mean()

    mean_loss = parnet.mean().abs()
    std_loss = (parnet.std() - 1.0).abs()
    max_loss = parnet.abs().max()

    flat = parnet.flatten()
    flat_min = flat.min()
    flat_max = flat.max()
    if flat_max - flat_min > 1e-8:
        hist = torch.histc(flat, bins=num_hist_bins, min=flat_min.item(), max=flat_max.item())
        hist = hist / (hist.sum() + 1e-8)
        hist_rough = (hist[1:] - hist[:-1]).abs().mean()
        hist_loss = hist_rough
    else:
        hist_loss = torch.tensor(0.0, device=parnet.device)

    total = (weights['smooth'] * smooth_loss +
             weights['mean']   * mean_loss +
             weights['std']    * std_loss +
             weights['max']    * max_loss +
             weights['hist']   * hist_loss)
    return total


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
def save_visualization(c_hat, c, z, decompressor, decoder, out_dir, eid):
    os.makedirs(out_dir, exist_ok=True)

    parnet_hat = decompressor(c_hat)
    parnet_orig = decompressor(c)
    parnet_stoch = decompressor(z) if z is not None else None

    img_hat = decoder(parnet_hat)
    img_orig = decoder(parnet_orig)

    tensor_to_pil(img_hat.squeeze(0)).save(os.path.join(out_dir, f"reconstructed_{eid}.png"))
    tensor_to_pil(img_orig.squeeze(0)).save(os.path.join(out_dir, f"original_{eid}.png"))

    if parnet_stoch is not None:
        img_stoch = decoder(parnet_stoch)
        tensor_to_pil(img_stoch.squeeze(0)).save(os.path.join(out_dir, f"stochastic_{eid}.png"))
        l1_stoch = F.l1_loss(img_stoch, img_orig).item()
        psnr_stoch = compute_psnr(img_stoch, img_orig)
    else:
        l1_stoch = 0.0
        psnr_stoch = 0.0

    diff = (img_hat - img_orig).abs()
    tensor_to_pil(diff.squeeze(0)).save(os.path.join(out_dir, f"difference_{eid}.png"))

    l1_img = F.l1_loss(img_hat, img_orig).item()
    psnr_img = compute_psnr(img_hat, img_orig)

    with open(os.path.join(out_dir, f"metrics_{eid}.txt"), 'w') as f:
        f.write(f"Reconstructed L1: {l1_img:.6f}\nReconstructed PSNR: {psnr_img:.2f} dB\n")
        f.write(f"Stochastic L1: {l1_stoch:.6f}\nStochastic PSNR: {psnr_stoch:.2f} dB\n")


# ---------------------- Чекпоинты ----------------------
def get_model_path(name, epoch):
    return os.path.join(MODELS_DIR, f"{name}_epoch{epoch}.pth")


def find_latest_checkpoint():
    files = glob.glob(os.path.join(MODELS_DIR, "encoder_epoch*.pth"))
    if not files:
        return None
    def extract_epoch(f):
        m = re.search(r'epoch(\d+)', f)
        return int(m.group(1)) if m else -1
    latest_enc = max(files, key=extract_epoch)
    epoch = extract_epoch(latest_enc)
    dec_path = get_model_path("decoder", epoch)
    return epoch if os.path.exists(dec_path) else None


def cleanup_old_checkpoints(keep=MAX_CHECKPOINTS):
    for name in ["encoder", "decoder"]:
        files = glob.glob(os.path.join(MODELS_DIR, f"{name}_epoch*.pth"))
        if len(files) <= keep:
            continue
        files.sort(key=lambda f: int(re.search(r'epoch(\d+)', f).group(1)), reverse=True)
        for old in files[keep:]:
            try:
                os.remove(old)
            except OSError:
                pass


def save_checkpoint(epoch, encoder, decoder, opt_enc, opt_dec):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': opt_enc.state_dict(),
    }, get_model_path("encoder", epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': opt_dec.state_dict(),
    }, get_model_path("decoder", epoch))
    cleanup_old_checkpoints()


def load_checkpoint_if_exist(encoder, decoder, opt_enc, opt_dec):
    epoch = find_latest_checkpoint()
    if epoch is not None:
        enc_path = get_model_path("encoder", epoch)
        dec_path = get_model_path("decoder", epoch)
        enc_ckpt = torch.load(enc_path, map_location=DEVICE, weights_only=False)
        dec_ckpt = torch.load(dec_path, map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(enc_ckpt['model_state_dict'])
        decoder.load_state_dict(dec_ckpt['model_state_dict'])
        opt_enc.load_state_dict(enc_ckpt['optimizer_state_dict'])
        opt_dec.load_state_dict(dec_ckpt['optimizer_state_dict'])
        print(f"Loaded checkpoint epoch {epoch}")
    else:
        epoch = 0
        print("No checkpoint found, starting from scratch.")
    return epoch


# ---------------------- Валидация ----------------------
@torch.no_grad()
def evaluate_and_visualize(encoder, decoder, decompressor, frozen_decoder, dataset, output_base, epoch, num_examples, seed=None):
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    for idx in indices:
        c, _ = dataset[idx]
        c = c.unsqueeze(0).to(DEVICE)
        mu, logvar = encoder(c)

        if STOCHASTIC_MODE:
            z = encoder.reparameterize(mu, logvar, STOCHASTIC_STRENGTH)
        else:
            z = mu

        c_hat = decoder(z)

        recon_l1 = F.l1_loss(c_hat, c).item()
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / c.size(0)
        kld = kld.item()

        eid = os.path.splitext(os.path.basename(dataset.files[idx]))[0]
        base_dir = os.path.join(output_base, f"epoch_{epoch}", f"example_{eid}")
        save_visualization(c_hat, c, z if STOCHASTIC_MODE else None, decompressor, frozen_decoder, base_dir, eid)

        with open(os.path.join(base_dir, f"metrics_{eid}.txt"), 'a') as f:
            f.write(f"Compressed L1: {recon_l1:.6f}\n")
            f.write(f"KL: {kld:.6f}\n")


# ---------------------- Обучение эпохи ----------------------
def train_epoch(encoder, decoder, train_loader, opt_enc, opt_dec, quality_weights):
    encoder.train()
    decoder.train()
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_w_kld = 0.0
    total_quality = 0.0
    total_diff_smooth_1 = 0.0
    total_diff_smooth_2 = 0.0
    n_batches = 0

    for batch_idx, (c, indices) in enumerate(train_loader):
        c = c.to(DEVICE)

        saved_grads_dec = {}
        saved_grads_enc = {}

        # ================= Фаза 1: декодер =================
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = True

        opt_dec.zero_grad()
        opt_enc.zero_grad()

        with torch.no_grad():
            mu, logvar = encoder(c)
            if STOCHASTIC_MODE:
                z = encoder.reparameterize(mu, logvar, STOCHASTIC_STRENGTH)
            else:
                z = mu

        c_hat = decoder(z)
        loss_1 = (RECON_LOSS_WEIGHT * difference_loss(c_hat, c) +
                  DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(c_hat, c))
        loss_1.backward()

        for name, param in decoder.named_parameters():
            if param.grad is not None:
                saved_grads_dec[name] = param.grad.clone().cpu()
        opt_dec.zero_grad()
        loss1_val = loss_1.item()
        diff_smooth_val_1 = diff_smooth_loss(c_hat, c).item()
        del c_hat, loss_1

        # ================= Фаза 2: энкодер =================
        for p in encoder.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = False

        opt_enc.zero_grad()

        mu, logvar = encoder(c)
        if STOCHASTIC_MODE:
            z = encoder.reparameterize(mu, logvar, STOCHASTIC_STRENGTH)
        else:
            z = mu
        c_hat = decoder(z)
        w_recon = RECON_LOSS_WEIGHT * difference_loss(c_hat, c)

        # Качество стохастического парнета
        q_loss = parnet_quality_loss(z, quality_weights)
        weighted_q = QUALITY_LOSS_WEIGHT * q_loss
        weighted_q_val = weighted_q.item()

        # Гладкость разности
        diff_smooth_val_2 = diff_smooth_loss(c_hat, c)

        # KL
        kl_active = STOCHASTIC_MODE and USE_KL_LOSS
        if kl_active:
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / c.size(0)
            kld_value = kld_loss.item()

            if kld_value <= KL_ZERO_THRESHOLD:
                kl_multiplier = 0.0
                if KL_ZERO_THRESHOLD > 0:
                    print(f"KL {kld_value:.6f} <= {KL_ZERO_THRESHOLD}, KL loss disabled for this batch")
            else:
                kl_multiplier = KL_WEIGHT
                if kld_value < KL_TARGET_MIN:
                    ratio = kld_value / KL_TARGET_MIN
                    protective_multiplier = max(KL_WEIGHT_MIN, ratio ** KL_ADAPTIVE_POWER)
                    kl_multiplier *= protective_multiplier

            w_kld = kl_multiplier * kld_loss
            loss_enc = w_recon + w_kld + weighted_q + DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_val_2
        else:
            w_kld = torch.tensor(0.0, device=DEVICE)
            loss_enc = w_recon + weighted_q + DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_val_2

        loss_enc.backward()

        for name, param in encoder.named_parameters():
            if param.grad is not None:
                saved_grads_enc[name] = param.grad.clone().cpu()
        opt_enc.zero_grad()
        loss2_val = loss_enc.item()

        # ================= Применение сохранённых градиентов =================
        for name, param in decoder.named_parameters():
            if name in saved_grads_dec:
                param.grad = saved_grads_dec[name].to(param.device)
            else:
                param.grad = None
        opt_dec.step()
        opt_dec.zero_grad()

        for name, param in encoder.named_parameters():
            if name in saved_grads_enc:
                param.grad = saved_grads_enc[name].to(param.device)
            else:
                param.grad = None
        opt_enc.step()
        opt_enc.zero_grad()

        saved_grads_dec.clear()
        saved_grads_enc.clear()

        total_loss1 += loss1_val
        total_loss2 += loss2_val
        total_w_kld += w_kld.item()
        total_quality += weighted_q_val
        total_diff_smooth_1 += diff_smooth_val_1
        total_diff_smooth_2 += diff_smooth_val_2.item() if torch.is_tensor(diff_smooth_val_2) else diff_smooth_val_2
        n_batches += 1

        if kl_active:
            print(f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Ph1: {loss1_val:.6f} | Ph2: {loss2_val:.6f} | "
                  f"Qual: {weighted_q_val:.6f} | KL: {w_kld.item():.6f} | "
                  f"DSm1: {diff_smooth_val_1:.4f} | DSm2: {diff_smooth_val_2:.4f}")
        else:
            print(f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Ph1: {loss1_val:.6f} | Ph2: {loss2_val:.6f} | "
                  f"Qual: {weighted_q_val:.6f} | "
                  f"DSm1: {diff_smooth_val_1:.4f} | DSm2: {diff_smooth_val_2:.4f}")

        del c, mu, logvar, z, c_hat, loss_enc, w_recon, w_kld, q_loss
        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss1 = total_loss1 / n_batches
    avg_loss2 = total_loss2 / n_batches
    avg_kld = total_w_kld / n_batches
    avg_quality = total_quality / n_batches
    avg_diff_smooth_1 = total_diff_smooth_1 / n_batches
    avg_diff_smooth_2 = total_diff_smooth_2 / n_batches
    return avg_loss1, avg_loss2, avg_kld, avg_quality, avg_diff_smooth_1, avg_diff_smooth_2


# ---------------------- main ----------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print("Loading frozen decompressor...")
    decompressor = load_frozen_decompressor(DECOMPRESSOR_CHECKPOINT)
    print("Loading frozen decoder...")
    frozen_decoder = load_frozen_decoder(DECODER_CHECKPOINT)

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

    encoder = StochasticEncoder(COMPRESSED_CHANNELS, STOCHASTIC_PARNET_DIM,
                                hidden_dim=ENCODER_HIDDEN_DIM, num_res_blocks=ENCODER_NUM_RES_BLOCKS).to(DEVICE)
    decoder = StochasticDecoder(COMPRESSED_CHANNELS, STOCHASTIC_PARNET_DIM,
                                hidden_dim=DECODER_HIDDEN_DIM, num_res_blocks=DECODER_NUM_RES_BLOCKS).to(DEVICE)

    opt_enc = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    opt_dec = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoint_if_exist(encoder, decoder, opt_enc, opt_dec) + 1

    quality_weights = {
        'smooth': QUALITY_SMOOTH_WEIGHT,
        'mean': QUALITY_MEAN_WEIGHT,
        'std': QUALITY_STD_WEIGHT,
        'max': QUALITY_MAX_WEIGHT,
        'hist': QUALITY_HIST_WEIGHT
    }

    print(f"STOCHASTIC_MODE = {STOCHASTIC_MODE} | STOCHASTIC_STRENGTH = {STOCHASTIC_STRENGTH} | "
          f"USE_KL_LOSS = {USE_KL_LOSS} (active = {STOCHASTIC_MODE and USE_KL_LOSS})")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_l1, avg_l2, avg_kld, avg_qual, avg_ds1, avg_ds2 = train_epoch(
            encoder, decoder, train_loader, opt_enc, opt_dec, quality_weights
        )

        kl_active = STOCHASTIC_MODE and USE_KL_LOSS
        if kl_active:
            print(f"Epoch {epoch:3d} | Ph1 (Dec): {avg_l1:.6f} | Ph2 (Enc): {avg_l2:.6f} | "
                  f"Qual: {avg_qual:.6f} | KL: {avg_kld:.6f}")
        else:
            print(f"Epoch {epoch:3d} | Ph1 (Dec): {avg_l1:.6f} | Ph2 (Enc): {avg_l2:.6f} | "
                  f"Qual: {avg_qual:.6f}")
        print(f"          | Avg DiffSmooth1: {avg_ds1:.4f} | Avg DiffSmooth2: {avg_ds2:.4f}")

        if val_dataset and epoch % VAL_EVERY_EPOCHS == 0:
            print("Running validation...")
            evaluate_and_visualize(encoder, decoder, decompressor, frozen_decoder, val_dataset,
                                   VAL_TESTS_DIR, epoch, NUM_TEST_EXAMPLES, TEST_SEED)

        if epoch % TEST_EVERY_EPOCHS == 0:
            print("Running tests...")
            evaluate_and_visualize(encoder, decoder, decompressor, frozen_decoder, train_dataset,
                                   TESTS_DIR, epoch, NUM_TEST_EXAMPLES, TEST_SEED)

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoint(epoch, encoder, decoder, opt_enc, opt_dec)
            print(f"Checkpoint saved at epoch {epoch}")

    print("Training completed.")


if __name__ == "__main__":
    main()