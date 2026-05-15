# training_models_Encoder_Decoder.py
import os, re, glob, math, random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
import numpy as np
from PIL import Image
from model_Autoencoder import Autoencoder
from config_training_models_Encoder_Decoder import *

ENCODER_DEVICE = torch.device(ENCODER_DEVICE_STR if torch.cuda.is_available() else "cpu")
DECODER_DEVICE = torch.device(DECODER_DEVICE_STR if torch.cuda.is_available() else "cpu")
print(f"Encoder device: {ENCODER_DEVICE}, Decoder device: {DECODER_DEVICE}")

# ---------------------- Датасет ----------------------
class ImageDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        img = data['image']         # [0,1]
        return img * 2 - 1          # -> [-1,1]

def collate_fn(batch):
    return torch.stack(batch, dim=0)

# ---------------------- Потери ----------------------
def difference_loss(pred, target):
    """Гипотетическая потеря: log(1 + |pred - target|)."""
    return torch.mean(torch.log(1.0 + torch.abs(pred - target)))

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(2.0) - 10 * math.log10(mse.item())

# ---------------------- Чекпоинты ----------------------
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
        try:
            os.remove(old)
        except OSError:
            pass

def save_checkpoints(epoch, model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.encoder.state_dict(), get_model_path('encoder', epoch))
    torch.save(model.decoder.state_dict(), get_model_path('decoder', epoch))
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

# ---------------------- Валидация / Тесты ----------------------
@torch.no_grad()
def evaluate_reconstruction(model, loader):
    model.encoder.eval()
    model.decoder.eval()
    sum_diff = 0.0
    sum_total = 0.0
    sum_psnr = 0.0
    n_batches = 0

    for images in loader:
        if ENCODER_DEVICE == DECODER_DEVICE:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = im_enc
        else:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = images.to(DECODER_DEVICE)
        rec = model(im_enc, ENCODER_DEVICE, DECODER_DEVICE)

        loss_diff = difference_loss(rec, im_dec)
        total = DIFF_LOSS_WEIGHT * loss_diff

        sum_diff += loss_diff.item()
        sum_total += total.item()
        sum_psnr += compute_psnr(rec, im_dec)
        n_batches += 1

    model.encoder.train()
    model.decoder.train()
    return (sum_diff / n_batches, sum_total / n_batches, sum_psnr / n_batches)

def tensor_to_pil(t):
    arr = (t.cpu().clamp(-1, 1).numpy() + 1) / 2 * 255
    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(arr)

def save_example(base_dir, orig, rec, metrics):
    os.makedirs(base_dir, exist_ok=True)
    tensor_to_pil(orig).save(os.path.join(base_dir, "original.png"))
    tensor_to_pil(rec).save(os.path.join(base_dir, "reconstructed.png"))
    diff = (rec - orig).abs()
    tensor_to_pil(diff).save(os.path.join(base_dir, "difference.png"))
    with open(os.path.join(base_dir, "metrics.txt"), 'w') as f:
        f.write(f"Diff: {metrics['diff']:.6f}\nPSNR: {metrics['psnr']:.2f} dB\n")

def run_tests(model, dataset, epoch):
    model.encoder.eval()
    model.decoder.eval()
    indices = random.sample(range(len(dataset)), min(NUM_TEST_EXAMPLES, len(dataset)))
    for idx in indices:
        img = dataset[idx]
        eid = int(os.path.splitext(os.path.basename(dataset.files[idx]))[0])
        im_enc = img.unsqueeze(0).to(ENCODER_DEVICE)
        im_dec = im_enc if ENCODER_DEVICE == DECODER_DEVICE else img.unsqueeze(0).to(DECODER_DEVICE)
        with torch.no_grad():
            rec = model(im_enc, ENCODER_DEVICE, DECODER_DEVICE)
            diff_v = difference_loss(rec, im_dec).item()
            psnr_v = compute_psnr(rec, im_dec)
        base = os.path.join(TESTS_DIR, f"epoch_{epoch}", f"example_{eid}")
        save_example(base, img, rec.squeeze(0).cpu(), {'diff': diff_v, 'psnr': psnr_v})
    model.encoder.train()
    model.decoder.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_all_val_examples(model, val_dataset, epoch):
    model.encoder.eval()
    model.decoder.eval()
    for idx in range(len(val_dataset)):
        img = val_dataset[idx]
        eid = int(os.path.splitext(os.path.basename(val_dataset.files[idx]))[0])
        im_enc = img.unsqueeze(0).to(ENCODER_DEVICE)
        im_dec = im_enc if ENCODER_DEVICE == DECODER_DEVICE else img.unsqueeze(0).to(DECODER_DEVICE)
        with torch.no_grad():
            rec = model(im_enc, ENCODER_DEVICE, DECODER_DEVICE)
            diff_v = difference_loss(rec, im_dec).item()
            psnr_v = compute_psnr(rec, im_dec)
        base = os.path.join(VAL_TESTS_DIR, f"epoch_{epoch}", f"example_{eid}")
        save_example(base, img, rec.squeeze(0).cpu(), {'diff': diff_v, 'psnr': psnr_v})
    model.encoder.train()
    model.decoder.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------- Обучение (две фазы) ----------------------
def train_epoch(model, train_loader, opt_enc, opt_dec):
    model.encoder.train()
    model.decoder.train()
    total_loss_epoch = 0.0
    n_batches = len(train_loader)

    for batch_idx, images in enumerate(train_loader):
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

        rec = model.decoder(parnet)
        loss_1 = DIFF_LOSS_WEIGHT * difference_loss(rec, im_dec)
        loss_1.backward()

        for name, param in model.decoder.named_parameters():
            if param.grad is not None:
                saved_grads_dec[name] = param.grad.clone().cpu()
        opt_dec.zero_grad()
        loss1_val = loss_1.item()
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
        rec = model.decoder(parnet)
        loss_2 = DIFF_LOSS_WEIGHT * difference_loss(rec, im_dec)
        loss_2.backward()

        for name, param in model.encoder.named_parameters():
            if param.grad is not None:
                saved_grads_enc[name] = param.grad.clone().cpu()
        opt_enc.zero_grad()
        loss2_val = loss_2.item()
        del parnet, rec, loss_2

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

        total_loss_epoch += loss1_val + loss2_val

        print(f"Batch {batch_idx+1}/{n_batches} | "
              f"Ph1 Diff: {loss1_val:.6f} | Ph2 Diff: {loss2_val:.6f}")

        del images, im_enc, im_dec
        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return total_loss_epoch / (2 * n_batches)

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

    if MAX_TRAIN_IMAGES and MAX_TRAIN_IMAGES > 0:
        train_files = all_files[:MAX_TRAIN_IMAGES]
        start_val = len(train_files)
        val_files = all_files[start_val:start_val + VALIDATION_SPLIT] if start_val < len(all_files) else []
    else:
        n_val = min(VALIDATION_SPLIT, len(all_files))
        train_files = all_files[:-n_val] if n_val < len(all_files) else []
        val_files = all_files[-n_val:] if n_val > 0 else []

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    train_dataset = ImageDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=0)
    val_loader = None
    val_dataset = None
    if val_files:
        val_dataset = ImageDataset(val_files)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, pin_memory=True, num_workers=0)

    model = Autoencoder(ENCODER_CONFIG, DECODER_CONFIG)
    model.encoder.to(ENCODER_DEVICE)
    model.decoder.to(DECODER_DEVICE)

    opt_enc = optim.Adam(model.encoder.parameters(), lr=LEARNING_RATE)
    opt_dec = optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoints_if_exist(model) + 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_total = train_epoch(model, train_loader, opt_enc, opt_dec)
        print(f"Epoch {epoch:3d}  Average Total: {avg_total:.6f}")

        if val_loader and epoch % VAL_EVERY_EPOCHS == 0:
            val_diff, val_total, val_psnr = evaluate_reconstruction(model, val_loader)
            print(f"Epoch {epoch:3d} VAL  Diff: {val_diff:.6f}  Total: {val_total:.6f}  PSNR: {val_psnr:.2f} dB")
            save_all_val_examples(model, val_dataset, epoch)

        if epoch % TEST_EVERY_EPOCHS == 0:
            run_tests(model, train_dataset, epoch)

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoints(epoch, model)

    print("Training completed.")

if __name__ == "__main__":
    train()