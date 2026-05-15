# training_models_Compressor_Decompressor.py
import os, re, glob, math, random, gc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from model_ParnetCompressor import ParnetCompressor, ParnetDecompressor
from model_Autoencoder import Decoder
from config_training_models_Compressor_Decompressor import *

COMPRESSOR_DEVICE = torch.device(COMPRESSOR_DEVICE_STR if torch.cuda.is_available() else "cpu")
DECOMPRESSOR_DEVICE = torch.device(DECOMPRESSOR_DEVICE_STR if torch.cuda.is_available() else "cpu")
DECODER_DEVICE = torch.device(DECODER_DEVICE_STR if torch.cuda.is_available() else "cpu")
print(f"Compressor: {COMPRESSOR_DEVICE}, Decompressor: {DECOMPRESSOR_DEVICE}, Decoder: {DECODER_DEVICE}")

COMPRESSOR_NAME = "compressor"
DECOMPRESSOR_NAME = "decompressor"

# ---------------------- Датасет ----------------------
class ParnetDataset(Dataset):
    """Загружает парнеты из .pt файлов (содержат ключ 'parnet')."""
    def __init__(self, file_list):
        self.files = file_list
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        return data['parnet']  # [3, H, W] в [-1,1]

def collate_fn(batch):
    return torch.stack(batch, dim=0)

# ---------------------- Потери (ГИПОТЕТИЧЕСКАЯ) ----------------------
def difference_loss(pred, target):
    """
    Гипотетическая потеря: log(1 + |pred - target|).
    Высокая чувствительность к малым отклонениям, низкая к большим.
    """
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

def save_checkpoints(epoch, compressor, opt_comp, decompressor, opt_decomp):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model, opt in [(COMPRESSOR_NAME, compressor, opt_comp),
                             (DECOMPRESSOR_NAME, decompressor, opt_decomp)]:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, get_model_path(name, epoch))
    cleanup_old_checkpoints(COMPRESSOR_NAME)
    cleanup_old_checkpoints(DECOMPRESSOR_NAME)

def load_checkpoints_if_exist(compressor, opt_comp, decompressor, opt_decomp):
    loaded_epoch = 0
    for name, model, opt in [(COMPRESSOR_NAME, compressor, opt_comp),
                             (DECOMPRESSOR_NAME, decompressor, opt_decomp)]:
        path, epoch = find_latest_checkpoint(name)
        if path:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"Loaded {name} from epoch {epoch}")
            if loaded_epoch == 0:
                loaded_epoch = epoch
            else:
                assert epoch == loaded_epoch, f"Epoch mismatch for {name}"
    return loaded_epoch

# ---------------------- Валидация (по полной схеме) ----------------------
def run_validation(compressor, opt_comp, decompressor, opt_decomp, val_loader, epoch):
    compressor.eval()
    decompressor.eval()
    sum_diff = 0.0
    sum_total = 0.0
    sum_psnr = 0.0
    n_batches = 0

    with torch.no_grad():
        for parnets in val_loader:
            if COMPRESSOR_DEVICE == DECOMPRESSOR_DEVICE:
                parnet_comp = parnets.to(COMPRESSOR_DEVICE)
                parnet_decomp = parnet_comp
            else:
                parnet_comp = parnets.to(COMPRESSOR_DEVICE)
                parnet_decomp = parnets.to(DECOMPRESSOR_DEVICE)

            compressed = compressor(parnet_comp)
            if COMPRESSOR_DEVICE != DECOMPRESSOR_DEVICE:
                compressed = compressed.to(DECOMPRESSOR_DEVICE)
            reconstructed = decompressor(compressed)

            loss_diff = difference_loss(reconstructed, parnet_decomp)
            total = PARNET_DIFF_LOSS_WEIGHT * loss_diff

            sum_diff += loss_diff.item()
            sum_total += total.item()
            sum_psnr += compute_psnr(reconstructed, parnet_decomp)
            n_batches += 1

    avg_diff = sum_diff / n_batches
    avg_total = sum_total / n_batches
    avg_psnr = sum_psnr / n_batches

    # Сохраняем несколько примеров
    val_dataset = val_loader.dataset
    indices = random.sample(range(len(val_dataset)), min(NUM_TEST_EXAMPLES, len(val_dataset)))

    examples = []
    for idx in indices:
        parnet = val_dataset[idx]
        eid = int(os.path.splitext(os.path.basename(val_dataset.files[idx]))[0])
        if COMPRESSOR_DEVICE == DECOMPRESSOR_DEVICE:
            parnet_comp = parnet.unsqueeze(0).to(COMPRESSOR_DEVICE)
            parnet_decomp = parnet_comp
        else:
            parnet_comp = parnet.unsqueeze(0).to(COMPRESSOR_DEVICE)
            parnet_decomp = parnet.unsqueeze(0).to(DECOMPRESSOR_DEVICE)
        with torch.no_grad():
            compressed = compressor(parnet_comp)
            if COMPRESSOR_DEVICE != DECOMPRESSOR_DEVICE:
                compressed = compressed.to(DECOMPRESSOR_DEVICE)
            rec = decompressor(compressed)
            diff_val = difference_loss(rec, parnet_decomp).item()
            psnr_val = compute_psnr(rec, parnet_decomp)
        examples.append({
            'example_id': eid,
            'original_parnet': parnet_decomp.squeeze(0).cpu(),
            'reconstructed_parnet': rec.squeeze(0).cpu(),
            'diff': diff_val,
            'psnr': psnr_val,
        })

    # Временно сохраняем модели и оптимизаторы
    temp_paths = []
    for name, model, opt in [(COMPRESSOR_NAME, compressor, opt_comp),
                             (DECOMPRESSOR_NAME, decompressor, opt_decomp)]:
        path = os.path.join(MODELS_DIR, f"temp_val_{name}_restore.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, path)
        temp_paths.append(path)

    del compressor, opt_comp, decompressor, opt_decomp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Загружаем декодер
    decoder = Decoder(**DECODER_CONFIG).to(DECODER_DEVICE)
    decoder_ckpt = torch.load(DECODER_CHECKPOINT, map_location=DECODER_DEVICE, weights_only=False)
    decoder.load_state_dict(decoder_ckpt)
    decoder.eval()

    for item in examples:
        base_dir = os.path.join(VAL_TESTS_DIR, f"epoch_{epoch}", f"example_{item['example_id']}")
        os.makedirs(base_dir, exist_ok=True)

        def parnet_to_pil(t):
            arr = (t.clamp(-1, 1).numpy() + 1) / 2 * 255
            arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
            return Image.fromarray(arr)

        orig_parnet = item['original_parnet']
        rec_parnet = item['reconstructed_parnet']
        parnet_to_pil(orig_parnet).save(os.path.join(base_dir, "original_parnet.png"))
        parnet_to_pil(rec_parnet).save(os.path.join(base_dir, "reconstructed_parnet.png"))
        diff_img = (rec_parnet - orig_parnet).abs()
        parnet_to_pil(diff_img).save(os.path.join(base_dir, "difference_parnet.png"))

        with torch.no_grad():
            orig_dec = decoder(orig_parnet.unsqueeze(0).to(DECODER_DEVICE)).squeeze(0).cpu()
            rec_dec = decoder(rec_parnet.unsqueeze(0).to(DECODER_DEVICE)).squeeze(0).cpu()
        parnet_to_pil(orig_dec).save(os.path.join(base_dir, "original_decoded.png"))
        parnet_to_pil(rec_dec).save(os.path.join(base_dir, "reconstructed_decoded.png"))
        diff_decoded = (rec_dec - orig_dec).abs()
        parnet_to_pil(diff_decoded).save(os.path.join(base_dir, "difference_decoded.png"))

        with open(os.path.join(base_dir, "metrics.txt"), 'w') as f:
            f.write(f"Diff (hypo): {item['diff']:.6f}\nPSNR: {item['psnr']:.2f} dB\n")

    del decoder, examples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Восстанавливаем модели
    compressor = ParnetCompressor(**COMPRESSOR_CONFIG).to(COMPRESSOR_DEVICE)
    decompressor = ParnetDecompressor(**DECOMPRESSOR_CONFIG).to(DECOMPRESSOR_DEVICE)
    opt_comp = optim.Adam(compressor.parameters(), lr=LEARNING_RATE)
    opt_decomp = optim.Adam(decompressor.parameters(), lr=LEARNING_RATE)

    for (name, model, opt), path in zip(
        [(COMPRESSOR_NAME, compressor, opt_comp), (DECOMPRESSOR_NAME, decompressor, opt_decomp)],
        temp_paths
    ):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        os.remove(path)

    compressor.train()
    decompressor.train()
    return compressor, opt_comp, decompressor, opt_decomp, avg_diff, avg_total, avg_psnr

# ---------------------- Тестирование ----------------------
def collect_test_data(compressor, decompressor, dataset):
    compressor.eval()
    decompressor.eval()
    random.seed(TEST_SEED)
    indices = random.sample(range(len(dataset)), min(NUM_TEST_EXAMPLES, len(dataset)))
    results = []

    with torch.no_grad():
        for idx in indices:
            parnet = dataset[idx]
            eid = int(os.path.splitext(os.path.basename(dataset.files[idx]))[0])

            if COMPRESSOR_DEVICE == DECOMPRESSOR_DEVICE:
                parnet_comp = parnet.unsqueeze(0).to(COMPRESSOR_DEVICE)
                parnet_decomp = parnet_comp
            else:
                parnet_comp = parnet.unsqueeze(0).to(COMPRESSOR_DEVICE)
                parnet_decomp = parnet.unsqueeze(0).to(DECOMPRESSOR_DEVICE)

            compressed = compressor(parnet_comp)
            if COMPRESSOR_DEVICE != DECOMPRESSOR_DEVICE:
                compressed = compressed.to(DECOMPRESSOR_DEVICE)
            reconstructed = decompressor(compressed)

            diff_val = difference_loss(reconstructed, parnet_decomp).item()
            psnr_val = compute_psnr(reconstructed, parnet_decomp)

            results.append({
                'example_id': eid,
                'original_parnet': parnet_decomp.squeeze(0).cpu(),
                'reconstructed_parnet': reconstructed.squeeze(0).cpu(),
                'diff': diff_val,
                'psnr': psnr_val,
            })

    compressor.train()
    decompressor.train()
    return results

def save_example_images(base_dir, item, decoder):
    os.makedirs(base_dir, exist_ok=True)

    def parnet_to_pil(t):
        arr = (t.clamp(-1, 1).numpy() + 1) / 2 * 255
        arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
        return Image.fromarray(arr)

    orig_parnet = item['original_parnet']
    rec_parnet = item['reconstructed_parnet']
    parnet_to_pil(orig_parnet).save(os.path.join(base_dir, "original_parnet.png"))
    parnet_to_pil(rec_parnet).save(os.path.join(base_dir, "reconstructed_parnet.png"))
    diff_img = (rec_parnet - orig_parnet).abs()
    parnet_to_pil(diff_img).save(os.path.join(base_dir, "difference_parnet.png"))

    decoder.eval()
    with torch.no_grad():
        orig_img = decoder(orig_parnet.unsqueeze(0).to(DECODER_DEVICE)).squeeze(0).cpu()
        rec_img = decoder(rec_parnet.unsqueeze(0).to(DECODER_DEVICE)).squeeze(0).cpu()
    parnet_to_pil(orig_img).save(os.path.join(base_dir, "original_decoded.png"))
    parnet_to_pil(rec_img).save(os.path.join(base_dir, "reconstructed_decoded.png"))
    diff_decoded = (rec_img - orig_img).abs()
    parnet_to_pil(diff_decoded).save(os.path.join(base_dir, "difference_decoded.png"))

    with open(os.path.join(base_dir, "metrics.txt"), 'w') as f:
        f.write(f"Diff (hypo): {item['diff']:.6f}\nPSNR: {item['psnr']:.2f} dB\n")

def run_tests(compressor, opt_comp, decompressor, opt_decomp, dataset, epoch):
    print("Collecting test data...")
    test_data = collect_test_data(compressor, decompressor, dataset)

    temp_paths = []
    for name, model, opt in [(COMPRESSOR_NAME, compressor, opt_comp),
                             (DECOMPRESSOR_NAME, decompressor, opt_decomp)]:
        path = os.path.join(MODELS_DIR, f"temp_test_{name}_restore.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, path)
        temp_paths.append(path)

    del compressor, opt_comp, decompressor, opt_decomp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    decoder = Decoder(**DECODER_CONFIG).to(DECODER_DEVICE)
    decoder_ckpt = torch.load(DECODER_CHECKPOINT, map_location=DECODER_DEVICE, weights_only=False)
    decoder.load_state_dict(decoder_ckpt)
    decoder.eval()

    for item in test_data:
        base_dir = os.path.join(TESTS_DIR, f"epoch_{epoch}", f"example_{item['example_id']}")
        save_example_images(base_dir, item, decoder)

    del decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    compressor = ParnetCompressor(**COMPRESSOR_CONFIG).to(COMPRESSOR_DEVICE)
    decompressor = ParnetDecompressor(**DECOMPRESSOR_CONFIG).to(DECOMPRESSOR_DEVICE)
    opt_comp = optim.Adam(compressor.parameters(), lr=LEARNING_RATE)
    opt_decomp = optim.Adam(decompressor.parameters(), lr=LEARNING_RATE)

    for (name, model, opt), path in zip(
        [(COMPRESSOR_NAME, compressor, opt_comp), (DECOMPRESSOR_NAME, decompressor, opt_decomp)],
        temp_paths
    ):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        os.remove(path)

    compressor.train()
    decompressor.train()
    return compressor, opt_comp, decompressor, opt_decomp

# ---------------------- Обучение (две фазы) ----------------------
def train_epoch(compressor, decompressor, train_loader, opt_comp, opt_decomp):
    compressor.train()
    decompressor.train()
    total_loss_epoch = 0.0
    n_batches = len(train_loader)

    for batch_idx, parnets in enumerate(train_loader):
        saved_grads_comp = {}
        saved_grads_decomp = {}

        # Фаза 1: декомпрессор
        for p in compressor.parameters():
            p.requires_grad = False
        for p in decompressor.parameters():
            p.requires_grad = True
        opt_decomp.zero_grad()

        if COMPRESSOR_DEVICE == DECOMPRESSOR_DEVICE:
            parnet_comp = parnets.to(COMPRESSOR_DEVICE)
            parnet_decomp = parnet_comp
        else:
            parnet_comp = parnets.to(COMPRESSOR_DEVICE)
            parnet_decomp = parnets.to(DECOMPRESSOR_DEVICE)

        with torch.no_grad():
            compressed = compressor(parnet_comp)
            if COMPRESSOR_DEVICE != DECOMPRESSOR_DEVICE:
                compressed = compressed.to(DECOMPRESSOR_DEVICE)

        rec = decompressor(compressed)
        loss_1 = PARNET_DIFF_LOSS_WEIGHT * difference_loss(rec, parnet_decomp)
        loss_1.backward()

        for name, param in decompressor.named_parameters():
            if param.grad is not None:
                saved_grads_decomp[name] = param.grad.clone().cpu()
        opt_decomp.zero_grad()
        loss1_val = loss_1.item()
        del compressed, rec, loss_1

        # Фаза 2: компрессор
        for p in compressor.parameters():
            p.requires_grad = True
        for p in decompressor.parameters():
            p.requires_grad = False
        opt_comp.zero_grad()

        compressed = compressor(parnet_comp)
        if COMPRESSOR_DEVICE != DECOMPRESSOR_DEVICE:
            compressed = compressed.to(DECOMPRESSOR_DEVICE)
        rec = decompressor(compressed)
        loss_2 = PARNET_DIFF_LOSS_WEIGHT * difference_loss(rec, parnet_decomp)
        loss_2.backward()

        for name, param in compressor.named_parameters():
            if param.grad is not None:
                saved_grads_comp[name] = param.grad.clone().cpu()
        opt_comp.zero_grad()
        loss2_val = loss_2.item()
        del compressed, rec, loss_2

        # Применяем градиенты
        for name, param in decompressor.named_parameters():
            if name in saved_grads_decomp:
                param.grad = saved_grads_decomp[name].to(param.device)
            else:
                param.grad = None
        opt_decomp.step()
        opt_decomp.zero_grad()
        saved_grads_decomp.clear()

        for name, param in compressor.named_parameters():
            if name in saved_grads_comp:
                param.grad = saved_grads_comp[name].to(param.device)
            else:
                param.grad = None
        opt_comp.step()
        opt_comp.zero_grad()
        saved_grads_comp.clear()

        total_loss_epoch += loss1_val + loss2_val

        print(f"Batch {batch_idx+1}/{n_batches} | "
              f"Ph1 Diff: {loss1_val:.6f} | Ph2 Diff: {loss2_val:.6f}")

        del parnets, parnet_comp, parnet_decomp
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
    print(f"Found {len(all_files)} parnet samples.")

    if MAX_TRAIN_IMAGES and MAX_TRAIN_IMAGES > 0:
        train_files = all_files[:MAX_TRAIN_IMAGES]
        start_val = len(train_files)
        val_files = all_files[start_val:start_val + VALIDATION_SPLIT] if start_val < len(all_files) else []
    else:
        n_val = min(VALIDATION_SPLIT, len(all_files))
        train_files = all_files[:-n_val] if n_val < len(all_files) else []
        val_files = all_files[-n_val:] if n_val > 0 else []

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    train_dataset = ParnetDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=0)
    val_loader = None
    if val_files:
        val_dataset = ParnetDataset(val_files)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, pin_memory=True, num_workers=0)

    compressor = ParnetCompressor(**COMPRESSOR_CONFIG).to(COMPRESSOR_DEVICE)
    decompressor = ParnetDecompressor(**DECOMPRESSOR_CONFIG).to(DECOMPRESSOR_DEVICE)

    opt_comp = optim.Adam(compressor.parameters(), lr=LEARNING_RATE)
    opt_decomp = optim.Adam(decompressor.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoints_if_exist(compressor, opt_comp, decompressor, opt_decomp) + 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_total = train_epoch(compressor, decompressor, train_loader, opt_comp, opt_decomp)
        print(f"Epoch {epoch:3d}  Average Total: {avg_total:.6f}")

        if val_loader and epoch % VAL_EVERY_EPOCHS == 0:
            print(f"Running validation for epoch {epoch}...")
            compressor, opt_comp, decompressor, opt_decomp, val_diff, val_total, val_psnr = run_validation(
                compressor, opt_comp, decompressor, opt_decomp, val_loader, epoch
            )
            print(f"Epoch {epoch:3d} VAL  Diff: {val_diff:.6f}  Total: {val_total:.6f}  PSNR: {val_psnr:.2f} dB")

        if epoch % TEST_EVERY_EPOCHS == 0:
            print(f"Running test examples for epoch {epoch}...")
            compressor, opt_comp, decompressor, opt_decomp = run_tests(
                compressor, opt_comp, decompressor, opt_decomp, train_dataset, epoch
            )

        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoints(epoch, compressor, opt_comp, decompressor, opt_decomp)
            print(f"Checkpoints saved at epoch {epoch}")

    print("Training completed.")

if __name__ == "__main__":
    train()