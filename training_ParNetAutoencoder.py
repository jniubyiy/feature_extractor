# training_ParNetAutoencoder.py
"""
Двухфазное обучение ParNetAutoencoder (структурирование сжатого парнета).
Фаза 1 – обучается только декодер (энкодер заморожен).
Фаза 2 – обучается только энкодер (декодер заморожен).
Градиенты сохраняются и применяются после обеих фаз.
Визуализация: сжатый парнет → decompressor → decoder → изображение.
"""
import os, re, glob, math, random, gc, json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from model_ParNetAutoencoder import ParNetAutoencoder
from model_ParnetCompressor import ParnetDecompressor
from model_Autoencoder import Decoder
from config_training_ParNetAutoencoder import *

DEVICE = torch.device(ENCODER_DEVICE_STR if torch.cuda.is_available() else "cpu")

# ------------------------ Загрузка замороженных моделей ------------------------
def _load_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_frozen_decompressor(checkpoint_path):
    from config_training_models_Compressor_Decompressor import DECOMPRESSOR_CONFIG
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
    from config_training_models_Encoder_Decoder import DECODER_CONFIG
    model = Decoder(**DECODER_CONFIG).to(DEVICE)
    state = _load_state_dict_from_checkpoint(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    )
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ------------------------ Датасет ------------------------
class CompressedParNetDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        return data['compressed_parnet'], idx


def collate_fn(batch):
    tensors, indices = zip(*batch)
    return torch.stack(tensors, dim=0), torch.tensor(indices, dtype=torch.long)


# ------------------------ Функции потерь ------------------------
def difference_loss(pred, target):
    return torch.mean(torch.log(1.0 + torch.abs(pred - target)))


def diff_smooth_loss(pred, target):
    diff = torch.abs(pred - target)
    d_h = diff[:, :, 1:, :] - diff[:, :, :-1, :]
    d_w = diff[:, :, :, 1:] - diff[:, :, :, :-1]
    return d_h.abs().mean() + d_w.abs().mean()


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(2.0) - 10 * math.log10(mse.item())


# ------------------------ Сохранение / загрузка чекпоинтов ------------------------
def get_model_path(name, epoch):
    return os.path.join(MODELS_DIR, f"{name}_epoch{epoch}.pth")


def find_latest_checkpoint(name):
    files = glob.glob(os.path.join(MODELS_DIR, f"{name}_epoch*.pth"))
    if not files:
        return None, 0
    def extract_epoch(f):
        m = re.search(r'epoch(\d+)', f)
        return int(m.group(1)) if m else -1
    latest = max(files, key=extract_epoch)
    return latest, extract_epoch(latest)


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
        'model_state_dict': encoder._orig_mod.state_dict(),
        'optimizer_state_dict': opt_enc.state_dict(),
    }, get_model_path("encoder", epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder._orig_mod.state_dict(),
        'optimizer_state_dict': opt_dec.state_dict(),
    }, get_model_path("decoder", epoch))
    cleanup_old_checkpoints()


def load_checkpoints_if_exist(encoder, decoder, opt_enc, opt_dec):
    epoch_enc_path, epoch_enc = find_latest_checkpoint("encoder")
    epoch_dec_path, epoch_dec = find_latest_checkpoint("decoder")
    if epoch_enc_path and epoch_dec_path:
        enc_ckpt = torch.load(epoch_enc_path, map_location=DEVICE, weights_only=False)
        dec_ckpt = torch.load(epoch_dec_path, map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(enc_ckpt['model_state_dict'])
        decoder.load_state_dict(dec_ckpt['model_state_dict'])
        opt_enc.load_state_dict(enc_ckpt['optimizer_state_dict'])
        opt_dec.load_state_dict(dec_ckpt['optimizer_state_dict'])
        print(f"Loaded checkpoint epoch {epoch_enc}")
        return epoch_enc
    else:
        print("No checkpoints found, starting from scratch.")
        return 0


# ------------------------ Визуализация ------------------------
def tensor_to_pil(t):
    arr = (t.cpu().clamp(-1, 1).numpy() + 1) / 2 * 255
    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def save_examples(encoder, decoder, dataset, epoch, output_dir, num_examples, seed,
                  decompressor, frozen_decoder):
    encoder.eval()
    decoder.eval()
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    for idx in indices:
        compressed, _ = dataset[idx]
        eid = os.path.splitext(os.path.basename(dataset.files[idx]))[0]
        inp = compressed.unsqueeze(0).to(DEVICE)

        structured = encoder(inp)
        rec_compressed = decoder(structured)

        img_orig = frozen_decoder(decompressor(inp))
        img_rec = frozen_decoder(decompressor(rec_compressed))
        img_diff = (img_rec - img_orig).abs()

        base = os.path.join(output_dir, f"epoch_{epoch}", f"example_{eid}")
        os.makedirs(base, exist_ok=True)
        tensor_to_pil(img_orig[0]).save(os.path.join(base, "original.png"))
        tensor_to_pil(img_rec[0]).save(os.path.join(base, "reconstructed.png"))
        tensor_to_pil(img_diff[0]).save(os.path.join(base, "difference.png"))

        l1 = F.l1_loss(img_rec, img_orig).item()
        psnr = compute_psnr(img_rec, img_orig)
        with open(os.path.join(base, "metrics.txt"), 'w') as f:
            f.write(f"Image L1: {l1:.6f}\nImage PSNR: {psnr:.2f} dB\n")
    encoder.train()
    decoder.train()


# ------------------------ Двухфазное обучение ------------------------
def train_epoch(encoder, decoder, train_loader, opt_enc, opt_dec):
    encoder.train()
    decoder.train()
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_ds1 = 0.0
    total_ds2 = 0.0
    n_batches = len(train_loader)

    for batch_idx, (compressed, indices) in enumerate(train_loader):
        compressed = compressed.to(DEVICE)
        saved_grads_enc = {}
        saved_grads_dec = {}

        # ---- Фаза 1: обучаем декодер, энкодер заморожен ----
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = True
        opt_dec.zero_grad()
        opt_enc.zero_grad()

        with torch.no_grad():
            structured = encoder(compressed)
        rec = decoder(structured)

        loss1 = (DIFF_LOSS_WEIGHT * difference_loss(rec, compressed) +
                 DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(rec, compressed))
        loss1.backward()
        # Сохраняем градиенты декодера
        for name, param in decoder.named_parameters():
            if param.grad is not None:
                saved_grads_dec[name] = param.grad.clone().cpu()
        opt_dec.zero_grad()
        loss1_val = loss1.item()
        ds1_val = (DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(rec, compressed)).item()
        del rec, loss1

        # ---- Фаза 2: обучаем энкодер, декодер заморожен ----
        for p in encoder.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = False
        opt_enc.zero_grad()

        structured = encoder(compressed)
        rec = decoder(structured)
        loss2 = (DIFF_LOSS_WEIGHT * difference_loss(rec, compressed) +
                 DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(rec, compressed))
        loss2.backward()
        # Сохраняем градиенты энкодера
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                saved_grads_enc[name] = param.grad.clone().cpu()
        opt_enc.zero_grad()
        loss2_val = loss2.item()
        ds2_val = (DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(rec, compressed)).item()
        del rec, loss2, structured

        # ---- Применяем сохранённые градиенты ----
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

        total_loss1 += loss1_val
        total_loss2 += loss2_val
        total_ds1 += ds1_val
        total_ds2 += ds2_val

        # Лог КАЖДОГО батча (убрано условие batch_idx % 10)
        print(f"Batch {batch_idx+1}/{n_batches} | Ph1: {loss1_val:.6f} | Ph2: {loss2_val:.6f} | "
              f"DS1: {ds1_val:.4f} | DS2: {ds2_val:.4f}")

        del compressed
        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss1 = total_loss1 / n_batches
    avg_loss2 = total_loss2 / n_batches
    avg_ds1 = total_ds1 / n_batches
    avg_ds2 = total_ds2 / n_batches
    return avg_loss1, avg_loss2, avg_ds1, avg_ds2


# ------------------------ Главный цикл ------------------------
def train():
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
    print(f"Found {len(all_files)} compressed parnet samples.")

    if MAX_TRAIN_IMAGES and MAX_TRAIN_IMAGES > 0:
        train_files = all_files[:MAX_TRAIN_IMAGES]
        start_val = len(train_files)
        val_files = all_files[start_val:start_val + VALIDATION_SPLIT] if start_val < len(all_files) else []
    else:
        n_val = min(VALIDATION_SPLIT, len(all_files))
        train_files = all_files[:-n_val] if n_val < len(all_files) else []
        val_files = all_files[-n_val:] if n_val > 0 else []

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    train_dataset = CompressedParNetDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=0)
    val_dataset = CompressedParNetDataset(val_files) if val_files else None

    # Создаём модель и разделяем энкодер / декодер
    autoencoder = ParNetAutoencoder(
        input_channels=INPUT_CHANNELS,
        bottleneck_channels=BOTTLENECK_CHANNELS,
        base_dim=BASE_DIM,
        num_blocks=NUM_BLOCKS
    ).to(DEVICE)
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    opt_enc = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    opt_dec = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoints_if_exist(encoder, decoder, opt_enc, opt_dec) + 1

    encoder = torch.compile(encoder, backend="aot_eager")
    decoder = torch.compile(decoder, backend="aot_eager")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_l1, avg_l2, avg_ds1, avg_ds2 = train_epoch(encoder, decoder, train_loader, opt_enc, opt_dec)
        print(f"Epoch {epoch:3d} | Ph1 (Dec): {avg_l1:.6f} | Ph2 (Enc): {avg_l2:.6f}")
        print(f" | DiffSmooth1: {avg_ds1:.4f} | DiffSmooth2: {avg_ds2:.4f}")

        if val_dataset and epoch % VAL_EVERY_EPOCHS == 0:
            print(f"Running validation for epoch {epoch}...")
            save_examples(encoder, decoder, val_dataset, epoch, VAL_TESTS_DIR, NUM_TEST_EXAMPLES,
                          TEST_SEED, decompressor, frozen_decoder)

        if epoch % TEST_EVERY_EPOCHS == 0:
            print(f"Running test visualization for epoch {epoch}...")
            save_examples(encoder, decoder, train_dataset, epoch, TESTS_DIR, NUM_TEST_EXAMPLES,
                          TEST_SEED, decompressor, frozen_decoder)

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoint(epoch, encoder, decoder, opt_enc, opt_dec)
            print(f"Checkpoints saved at epoch {epoch}")

    print("Training completed.")


if __name__ == "__main__":
    train()