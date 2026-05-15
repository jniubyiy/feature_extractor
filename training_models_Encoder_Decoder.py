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
def mse_loss(pred, target):
    return F.mse_loss(pred, target)

def tv_loss(img):
    """Total Variation Loss: сумма абсолютных разностей соседних пикселей по H и W."""
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()

def laplacian(x):
    """Применяет Лапласиан (3x3) к каждому каналу."""
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, kernel.expand(x.size(1), 1, 3, 3), padding=1, groups=x.size(1))

def edge_loss(pred, target):
    """Сравнивает Лапласианы предсказанного и целевого изображений (L1)."""
    pred_lap = laplacian(pred)
    target_lap = laplacian(target)
    return F.l1_loss(pred_lap, target_lap)

def ssim_loss(pred, target):
    """Потеря 1 - SSIM. Используем torchvision, если доступна, иначе простую реализацию."""
    try:
        from torchvision.transforms.functional import ssim as tv_ssim
        # data_range=2.0 для изображений в [-1,1]
        val = tv_ssim(pred, target, data_range=2.0, win_size=11)
        return 1 - val
    except ImportError:
        # Запасная реализация через свёртку с гауссовым окном
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        # Приводим к диапазону [0,1] для стабильности констант
        pred_01 = (pred + 1) / 2
        target_01 = (target + 1) / 2

        # Простое гауссово окно 11x11
        def gaussian_window(window_size, sigma=1.5, channels=1):
            coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            g = g.unsqueeze(0) * g.unsqueeze(1)   # (11,11)
            g = g.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size)
            return g

        win = gaussian_window(11, 1.5, pred.size(1))
        mu1 = F.conv2d(pred_01, win, padding=5, groups=pred.size(1))
        mu2 = F.conv2d(target_01, win, padding=5, groups=pred.size(1))
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(pred_01 * pred_01, win, padding=5, groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target_01 * target_01, win, padding=5, groups=pred.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred_01 * target_01, win, padding=5, groups=pred.size(1)) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

def noise_loss(pred, target):
    """
    Оценка зашумлённости в гладких областях целевого изображения.
    Вычисляем дисперсию Лапласиана предсказания внутри маски «гладких» пикселей target.
    """
    # Градиенты target для определения гладкости
    dh = target[:, :, 1:, :] - target[:, :, :-1, :]  # (B, C, H-1, W)
    dw = target[:, :, :, 1:] - target[:, :, :, :-1]  # (B, C, H, W-1)
    # Обрезаем до одинакового размера (B, C, H-1, W-1)
    grad_mag = torch.sqrt(dh[:, :, :, :-1] ** 2 + dw[:, :, :-1, :] ** 2)  # (B, C, H-1, W-1)
    # Усредняем по каналам, чтобы получить одноканальную маску
    grad_mag = grad_mag.mean(dim=1, keepdim=True)  # (B, 1, H-1, W-1)
    # Порог для гладкости – настраивается
    threshold = 0.05  # подходит для [-1,1]
    smooth_mask = grad_mag < threshold

    # Лапласиан предсказания (размер совпадает с pred, H, W)
    pred_lap = laplacian(pred)
    # Приводим маску к размеру pred_lap (отбрасываем края, потеря 1 пиксель по H и W)
    pred_lap_cut = pred_lap[:, :, :-1, :-1]
    # Применяем маску
    if smooth_mask.sum() > 0:
        noise_var = pred_lap_cut[smooth_mask.expand_as(pred_lap_cut)].var()
    else:
        noise_var = torch.tensor(0.0, device=pred.device)
    return noise_var

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
    total_mse = 0.0
    total_psnr = 0.0
    n = 0
    for images in loader:
        if ENCODER_DEVICE == DECODER_DEVICE:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = im_enc
        else:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = images.to(DECODER_DEVICE)
        rec = model(im_enc, ENCODER_DEVICE, DECODER_DEVICE)
        mse = mse_loss(rec, im_dec)
        total_mse += mse.item()
        total_psnr += compute_psnr(rec, im_dec)
        n += 1
    model.encoder.train()
    model.decoder.train()
    return total_mse / n, total_psnr / n

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
        f.write(f"MSE: {metrics['mse']:.6f}\nPSNR: {metrics['psnr']:.2f} dB\n")

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
            mse_v = mse_loss(rec, im_dec).item()
            psnr_v = compute_psnr(rec, im_dec)
        base = os.path.join(TESTS_DIR, f"epoch_{epoch}", f"example_{eid}")
        save_example(base, img, rec.squeeze(0).cpu(), {'mse': mse_v, 'psnr': psnr_v})
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
            mse_v = mse_loss(rec, im_dec).item()
            psnr_v = compute_psnr(rec, im_dec)
        base = os.path.join(VAL_TESTS_DIR, f"epoch_{epoch}", f"example_{eid}")
        save_example(base, img, rec.squeeze(0).cpu(), {'mse': mse_v, 'psnr': psnr_v})
    model.encoder.train()
    model.decoder.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------- Обучение ----------------------
def train_epoch(model, train_loader, optimizer):
    model.encoder.train()
    model.decoder.train()
    total_loss = 0.0
    n_batches = len(train_loader)

    for batch_idx, images in enumerate(train_loader):
        optimizer.zero_grad()

        if ENCODER_DEVICE == DECODER_DEVICE:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = im_enc
        else:
            im_enc = images.to(ENCODER_DEVICE)
            im_dec = images.to(DECODER_DEVICE)

        rec = model(im_enc, ENCODER_DEVICE, DECODER_DEVICE)

        loss_mse = mse_loss(rec, im_dec)
        loss_tv = tv_loss(rec)
        loss_edge = edge_loss(rec, im_dec)
        loss_ssim = ssim_loss(rec, im_dec)
        loss_noise = noise_loss(rec, im_dec)

        loss = (MSE_LOSS_WEIGHT * loss_mse +
                TV_LOSS_WEIGHT * loss_tv +
                EDGE_LOSS_WEIGHT * loss_edge +
                SSIM_LOSS_WEIGHT * loss_ssim +
                NOISE_LOSS_WEIGHT * loss_noise)

        loss.backward()
        optimizer.step()
        total_loss += loss_mse.item()

        print(f"Batch {batch_idx+1}/{n_batches} | "
              f"MSE: {loss_mse.item():.6f} | TV: {loss_tv.item():.6f} | Edge: {loss_edge.item():.6f} | "
              f"SSIM: {loss_ssim.item():.6f} | Noise: {loss_noise.item():.6f} | Total: {loss.item():.6f}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del images, rec, im_enc, im_dec
        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return total_loss / n_batches

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

    optimizer = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=LEARNING_RATE)
    start_epoch = load_checkpoints_if_exist(model) + 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        loss = train_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch:3d}  Average MSE: {loss:.6f}")

        if val_loader and epoch % VAL_EVERY_EPOCHS == 0:
            val_mse, val_psnr = evaluate_reconstruction(model, val_loader)
            print(f"Epoch {epoch:3d} VAL  MSE: {val_mse:.6f}  PSNR: {val_psnr:.2f} dB")
            save_all_val_examples(model, val_dataset, epoch)

        if epoch % TEST_EVERY_EPOCHS == 0:
            run_tests(model, train_dataset, epoch)

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoints(epoch, model)

    print("Training completed.")

if __name__ == "__main__":
    train()