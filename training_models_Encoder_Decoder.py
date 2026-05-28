# training_models_Encoder_Decoder.py
"""
Обучение Encoder и Decoder с двухфазным обновлением градиентов.
К парнету может добавляться гауссов шум (регулируется NOISE_STRENGTH).
Потери: difference_loss (линейная L1), diff_smooth_loss.
"""
import os, re, glob, math, random, gc, json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from model_Autoencoder import Autoencoder
from config_training_models_Encoder_Decoder import *

ENCODER_DEVICE = torch.device(ENCODER_DEVICE_STR if torch.cuda.is_available() else "cpu")
DECODER_DEVICE = torch.device(DECODER_DEVICE_STR if torch.cuda.is_available() else "cpu")

print(f"Encoder device: {ENCODER_DEVICE}, Decoder device: {DECODER_DEVICE}")


class IndexedImageDataset(Dataset):
    def __init__(self, file_list):
        self.files = []
        for f in file_list:
            if os.path.basename(f) == "similarities.pt":
                print(f"Skipping service file: {f}")
                continue
            try:
                data = torch.load(f, map_location='cpu', weights_only=False)
                _ = data['image']
                self.files.append(f)
            except Exception as e:
                print(f"Skipping corrupted/unreadable file {f}: {e}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        img = data['image']  # [0,1]
        return img * 2 - 1, idx  # -> [-1,1]


def collate_fn(batch):
    images, indices = zip(*batch)
    images = torch.stack(images, dim=0)
    indices = torch.tensor(indices, dtype=torch.long)
    return images, indices


# ==== Изменённая loss-функция (линейная L1 вместо log) ====
def difference_loss(pred, target):
    """Линейная L1-потеря: среднее абсолютных отклонений."""
    return torch.mean(torch.abs(pred - target))
# ===========================================================

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


def cleanup_old_checkpoints(name, keep=MAX_CHECKPOINTS):
    files = glob.glob(os.path.join(MODELS_DIR, f"{name}_epoch*.pth"))
    if len(files) <= keep:
        return
    files.sort(key=lambda f: int(re.search(r'epoch(\d+)', f).group(1)), reverse=True)
    for old in files[keep:]:
        try: os.remove(old)
        except OSError: pass


def save_checkpoints(epoch, model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.encoder._orig_mod.state_dict(), get_model_path('encoder', epoch))
    torch.save(model.decoder._orig_mod.state_dict(), get_model_path('decoder', epoch))
    cleanup_old_checkpoints('encoder')
    cleanup_old_checkpoints('decoder')


def load_checkpoints_if_exist(model):
    loaded_epoch = 0
    for name, sub, dev in [('encoder', model.encoder, ENCODER_DEVICE),
                           ('decoder', model.decoder, DECODER_DEVICE)]:
        path, epoch = find_latest_checkpoint(name)
        if path:
            sub.load_state_dict(torch.load(path, map_location=dev, weights_only=False))
            print(f"Loaded {name} from epoch {epoch}")
            if loaded_epoch == 0:
                loaded_epoch = epoch
            else:
                assert epoch == loaded_epoch, f"Epoch mismatch for {name}"
    return loaded_epoch


@torch.no_grad()
def evaluate_reconstruction(model, loader):
    model.encoder.eval()
    model.decoder.eval()
    sum_diff = 0.0
    sum_total = 0.0
    sum_psnr = 0.0
    n_batches = 0
    for images, _ in loader:
        if ENCODER_DEVICE == DECODER_DEVICE:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = im_enc
        else:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = images.to(DECODER_DEVICE)
        parnet = model.encoder(im_enc)
        rec = model.decoder(parnet)
        loss_diff = difference_loss(rec, im_dec)
        total = DIFF_LOSS_WEIGHT * loss_diff
        sum_diff += loss_diff.item()
        sum_total += total.item()
        sum_psnr += compute_psnr(rec, im_dec)
        n_batches += 1
    model.encoder.train()
    model.decoder.train()
    return sum_diff / n_batches, sum_total / n_batches, sum_psnr / n_batches


def tensor_to_pil(t):
    arr = (t.cpu().clamp(-1, 1).numpy() + 1) / 2 * 255
    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(arr)


def save_example(base_dir, orig, rec, parnet, metrics):
    os.makedirs(base_dir, exist_ok=True)
    tensor_to_pil(orig).save(os.path.join(base_dir, "original.png"))
    tensor_to_pil(rec).save(os.path.join(base_dir, "reconstructed.png"))
    diff = (rec - orig).abs()
    tensor_to_pil(diff).save(os.path.join(base_dir, "difference.png"))
    with open(os.path.join(base_dir, "metrics.txt"), 'w') as f:
        f.write(f"Diff: {metrics['diff']:.6f}\nPSNR: {metrics['psnr']:.2f} dB\n")
    parnet_cpu = parnet.cpu()
    with open(os.path.join(base_dir, "parnet_values.json"), 'w') as f:
        json.dump(parnet_cpu.tolist(), f)


def run_tests(model, dataset, epoch):
    model.encoder.eval()
    model.decoder.eval()
    indices = random.sample(range(len(dataset)), min(NUM_TEST_EXAMPLES, len(dataset)))
    for idx in indices:
        img, _ = dataset[idx]
        eid = os.path.splitext(os.path.basename(dataset.files[idx]))[0]
        im_enc = img.unsqueeze(0).to(ENCODER_DEVICE)
        im_dec = im_enc if ENCODER_DEVICE == DECODER_DEVICE else img.unsqueeze(0).to(DECODER_DEVICE)
        with torch.no_grad():
            parnet = model.encoder(im_enc)
            rec = model.decoder(parnet)
            diff_v = difference_loss(rec, im_dec).item()
            psnr_v = compute_psnr(rec, im_dec)
        base = os.path.join(TESTS_DIR, f"epoch_{epoch}", f"example_{eid}")
        save_example(base, img, rec.squeeze(0).cpu(), parnet.squeeze(0).cpu(),
                     {'diff': diff_v, 'psnr': psnr_v})
    model.encoder.train()
    model.decoder.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_all_val_examples(model, val_dataset, epoch):
    model.encoder.eval()
    model.decoder.eval()
    for idx in range(len(val_dataset)):
        img, _ = val_dataset[idx]
        eid = os.path.splitext(os.path.basename(val_dataset.files[idx]))[0]
        im_enc = img.unsqueeze(0).to(ENCODER_DEVICE)
        im_dec = im_enc if ENCODER_DEVICE == DECODER_DEVICE else img.unsqueeze(0).to(DECODER_DEVICE)
        with torch.no_grad():
            parnet = model.encoder(im_enc)
            rec = model.decoder(parnet)
            diff_v = difference_loss(rec, im_dec).item()
            psnr_v = compute_psnr(rec, im_dec)
        base = os.path.join(VAL_TESTS_DIR, f"epoch_{epoch}", f"example_{eid}")
        save_example(base, img, rec.squeeze(0).cpu(), parnet.squeeze(0).cpu(),
                     {'diff': diff_v, 'psnr': psnr_v})
    model.encoder.train()
    model.decoder.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_epoch(model, train_loader, opt_enc, opt_dec):
    model.encoder.train()
    model.decoder.train()
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_diff_smooth_1 = 0.0
    total_diff_smooth_2 = 0.0
    n_batches = len(train_loader)

    for batch_idx, (images, indices) in enumerate(train_loader):
        saved_grads_enc = {}
        saved_grads_dec = {}

        # Фаза 1: декодер
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = True
        opt_dec.zero_grad()

        if ENCODER_DEVICE == DECODER_DEVICE:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = im_enc
        else:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = images.to(DECODER_DEVICE)

        with torch.no_grad():
            parnet = model.encoder(im_enc)
            if ENCODER_DEVICE != DECODER_DEVICE:
                parnet = parnet.to(DECODER_DEVICE)
            if NOISE_STRENGTH > 0:
                noise = torch.randn_like(parnet) * NOISE_STRENGTH
                parnet = parnet + noise

        rec = model.decoder(parnet)
        loss_1 = (DIFF_LOSS_WEIGHT * difference_loss(rec, im_dec) +
                  DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(rec, im_dec))
        loss_1.backward()

        for name, param in model.decoder.named_parameters():
            if param.grad is not None:
                saved_grads_dec[name] = param.grad.clone().cpu()
        opt_dec.zero_grad()

        loss1_val = loss_1.item()
        diff_smooth_val_1 = (DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(rec, im_dec)).item()
        del parnet, rec, loss_1

        # Фаза 2: энкодер
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = False
        opt_enc.zero_grad()

        parnet = model.encoder(im_enc)
        if ENCODER_DEVICE != DECODER_DEVICE:
            parnet = parnet.to(DECODER_DEVICE)
        if NOISE_STRENGTH > 0:
            noise = torch.randn_like(parnet) * NOISE_STRENGTH
            parnet = parnet + noise

        rec = model.decoder(parnet)
        loss_2 = (DIFF_LOSS_WEIGHT * difference_loss(rec, im_dec) +
                  DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(rec, im_dec))
        loss_2.backward()

        for name, param in model.encoder.named_parameters():
            if param.grad is not None:
                saved_grads_enc[name] = param.grad.clone().cpu()
        opt_enc.zero_grad()

        loss2_val = loss_2.item()
        diff_smooth_val_2 = (DIFF_SMOOTH_LOSS_WEIGHT * diff_smooth_loss(rec, im_dec)).item()
        del rec, loss_2, parnet

        # Применяем градиенты
        for name, param in model.decoder.named_parameters():
            if name in saved_grads_dec:
                param.grad = saved_grads_dec[name].to(param.device)
            else:
                param.grad = None
        opt_dec.step()
        opt_dec.zero_grad()
        saved_grads_dec.clear()

        for name, param in model.encoder.named_parameters():
            if name in saved_grads_enc:
                param.grad = saved_grads_enc[name].to(param.device)
            else:
                param.grad = None
        opt_enc.step()
        opt_enc.zero_grad()
        saved_grads_enc.clear()

        total_loss1 += loss1_val
        total_loss2 += loss2_val
        total_diff_smooth_1 += diff_smooth_val_1
        total_diff_smooth_2 += diff_smooth_val_2

        print(f"Batch {batch_idx+1}/{n_batches} | "
              f"Ph1: {loss1_val:.6f} | Ph2: {loss2_val:.6f} | "
              f"DiffSm1: {diff_smooth_val_1:.4f} | DiffSm2: {diff_smooth_val_2:.4f}")

        del images, indices, im_enc, im_dec
        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss1 = total_loss1 / n_batches
    avg_loss2 = total_loss2 / n_batches
    avg_ds1 = total_diff_smooth_1 / n_batches
    avg_ds2 = total_diff_smooth_2 / n_batches
    return avg_loss1, avg_loss2, avg_ds1, avg_ds2


def train():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    all_files = sorted(
        [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.pt')],
        key=lambda x: os.path.basename(x)
    )
    if not all_files:
        raise RuntimeError(f"No .pt files in {DATASET_DIR}")

    print(f"Found {len(all_files)} files. Filtering...")

    if MAX_TRAIN_IMAGES and MAX_TRAIN_IMAGES > 0:
        train_files = all_files[:MAX_TRAIN_IMAGES]
        start_val = len(train_files)
        val_files = all_files[start_val:start_val + VALIDATION_SPLIT] if start_val < len(all_files) else []
    else:
        n_val = min(VALIDATION_SPLIT, len(all_files))
        train_files = all_files[:-n_val] if n_val < len(all_files) else []
        val_files = all_files[-n_val:] if n_val > 0 else []

    train_dataset = IndexedImageDataset(train_files)
    val_dataset = IndexedImageDataset(val_files) if val_files else None
    print(f"Train files (valid): {len(train_dataset)}, Val files (valid): {len(val_dataset) if val_dataset else 0}")
    if len(train_dataset) == 0:
        raise RuntimeError("No valid training files found.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=0)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, pin_memory=True, num_workers=0)

    model = Autoencoder(ENCODER_CONFIG, DECODER_CONFIG)
    model.encoder.to(ENCODER_DEVICE)
    model.decoder.to(DECODER_DEVICE)
    opt_enc = optim.Adam(model.encoder.parameters(), lr=LEARNING_RATE)
    opt_dec = optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoints_if_exist(model) + 1

    model.encoder = torch.compile(model.encoder, backend="aot_eager")
    model.decoder = torch.compile(model.decoder, backend="aot_eager")

    print(f"Noise strength: {NOISE_STRENGTH}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_loss1, avg_loss2, avg_ds1, avg_ds2 = train_epoch(model, train_loader, opt_enc, opt_dec)
        print(f"Epoch {epoch:3d} | Phase1 (Decoder): {avg_loss1:.6f} | "
              f"Phase2 (Encoder): {avg_loss2:.6f}")
        print(f" | Avg DiffSmooth1: {avg_ds1:.4f} | Avg DiffSmooth2: {avg_ds2:.4f}")

        if val_loader and epoch % VAL_EVERY_EPOCHS == 0:
            val_diff, val_total, val_psnr = evaluate_reconstruction(model, val_loader)
            print(f"Epoch {epoch:3d} VAL Diff: {val_diff:.6f} Total: {val_total:.6f} PSNR: {val_psnr:.2f} dB")
            save_all_val_examples(model, val_dataset, epoch)

        if epoch % TEST_EVERY_EPOCHS == 0:
            run_tests(model, train_dataset, epoch)

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoints(epoch, model)

    print("Training completed.")


if __name__ == "__main__":
    train()