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

# ---------------------- Датасет (с индексами) ----------------------
class IndexedParnetDataset(Dataset):
    """Загружает парнеты из .pt файлов (ключ 'parnet') и возвращает индекс."""
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        return data['parnet'], idx   # [3, H, W], индекс

def collate_fn(batch):
    parnets, indices = zip(*batch)
    parnets = torch.stack(parnets, dim=0)
    indices = torch.tensor(indices, dtype=torch.long)
    return parnets, indices

# ---------------------- Потери ----------------------
def difference_loss(pred, target):
    """Устойчивая L1 потеря (средняя абсолютная ошибка) с множителем 2."""
    return 2.0 * torch.mean(torch.abs(pred - target))

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(2.0) - 10 * math.log10(mse.item())

def compute_span(tensor):
    """Средний размах (max-min) по батчу."""
    B = tensor.shape[0]
    flat = tensor.view(B, -1)
    span_per_image = flat.max(dim=1).values - flat.min(dim=1).values
    return span_per_image.mean().item()

# ---------------------- Буфер сравнения (на CPU) ----------------------
class SimilarityLossBuffer:
    """
    Хранит фиксированное количество последних сжатых парнетов и их индексов в RAM (CPU).
    При сравнении текущий батч переносится на CPU, вычисляется потеря, результат возвращается на устройство обучения.
    """
    def __init__(self, max_elements, neighbor_map):
        self.max_elements = max_elements
        self.neighbor_map = neighbor_map
        self.buffer_parnets = None   # CPU тензор [N, D]
        self.buffer_indices = None   # CPU тензор [N]

    def add(self, parnet_flat, indices):
        """
        Добавляет отсоединённый тензор в буфер (на CPU).
        parnet_flat: тензор на GPU (или CPU), indices: тензор на CPU.
        """
        detached = parnet_flat.detach().cpu()
        if self.buffer_parnets is None:
            self.buffer_parnets = detached
            self.buffer_indices = indices.cpu() if indices.is_cuda else indices
        else:
            self.buffer_parnets = torch.cat([self.buffer_parnets, detached], dim=0)
            self.buffer_indices = torch.cat([self.buffer_indices, indices.cpu() if indices.is_cuda else indices], dim=0)
        # Ограничиваем размер
        if self.buffer_parnets.shape[0] > self.max_elements:
            self.buffer_parnets = self.buffer_parnets[-self.max_elements:]
            self.buffer_indices = self.buffer_indices[-self.max_elements:]

    def compute_loss(self, parnet_flat, indices):
        """
        Вычисляет косинусное расстояние между текущим батчем и релевантными элементами буфера.
        Все операции проводятся на CPU, результат возвращается на устройство parnet_flat.
        Возвращает (loss_tensor_on_device, count).
        """
        B = parnet_flat.shape[0]
        device = parnet_flat.device
        loss = torch.tensor(0.0, device=device)
        count = 0
        if self.buffer_parnets is None or self.buffer_parnets.shape[0] == 0:
            return loss, count
        # Переносим текущий батч на CPU
        parnet_flat_cpu = parnet_flat.detach().cpu()
        indices_cpu = indices.cpu() if indices.is_cuda else indices
        # Нормализуем на CPU
        parnet_norm = F.normalize(parnet_flat_cpu, dim=1, eps=1e-8)
        buffer_norm = F.normalize(self.buffer_parnets, dim=1, eps=1e-8)
        cos_sim = torch.mm(parnet_norm, buffer_norm.t())  # [B, N_buf] на CPU
        cos_dist = 1.0 - cos_sim
        # Проходим по всем парам (i в батче, j в буфере), которые являются соседями
        sum_loss = 0.0
        for i in range(B):
            idx_i = indices_cpu[i].item()
            if idx_i not in self.neighbor_map:
                continue
            neighbors = self.neighbor_map[idx_i]
            for j in range(self.buffer_indices.shape[0]):
                if self.buffer_indices[j].item() in neighbors:
                    sum_loss += cos_dist[i, j].item()
                    count += 1
        if count > 0:
            loss = torch.tensor(sum_loss / count, device=device)
        return loss, count

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
    sum_span_compressed = 0.0
    sum_span_reconstructed = 0.0
    n_batches = 0
    with torch.no_grad():
        for parnets, _ in val_loader:          # индексы не нужны
            if COMPRESSOR_DEVICE == DECOMPRESSOR_DEVICE:
                parnet_comp = parnets.to(COMPRESSOR_DEVICE)
                parnet_decomp = parnet_comp
            else:
                parnet_comp = parnets.to(COMPRESSOR_DEVICE)
                parnet_decomp = parnets.to(DECOMPRESSOR_DEVICE)
            compressed = compressor(parnet_comp)  # [B,4,H/2,W/2]
            if COMPRESSOR_DEVICE != DECOMPRESSOR_DEVICE:
                compressed = compressed.to(DECOMPRESSOR_DEVICE)
            reconstructed = decompressor(compressed)
            loss_diff = difference_loss(reconstructed, parnet_decomp)
            total = PARNET_DIFF_LOSS_WEIGHT * loss_diff
            sum_diff += loss_diff.item()
            sum_total += total.item()
            sum_psnr += compute_psnr(reconstructed, parnet_decomp)
            sum_span_compressed += compute_span(compressed)
            sum_span_reconstructed += compute_span(reconstructed)
            n_batches += 1
    avg_diff = sum_diff / n_batches
    avg_total = sum_total / n_batches
    avg_psnr = sum_psnr / n_batches
    avg_span_comp = sum_span_compressed / n_batches
    avg_span_rec = sum_span_reconstructed / n_batches

    # Сохраняем несколько примеров
    val_dataset = val_loader.dataset
    indices = random.sample(range(len(val_dataset)), min(NUM_TEST_EXAMPLES, len(val_dataset)))
    examples = []
    for idx in indices:
        parnet, _ = val_dataset[idx]   # распаковка (parnet, index)
        # Для совместимости с буквами имена могут быть любыми, используем просто индекс
        fname = os.path.basename(val_dataset.files[idx])
        eid = os.path.splitext(fname)[0]
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
            span_comp_val = compute_span(compressed)
            span_rec_val = compute_span(rec)
            examples.append({
                'example_id': eid,
                'original_parnet': parnet_decomp.squeeze(0).cpu(),
                'reconstructed_parnet': rec.squeeze(0).cpu(),
                'diff': diff_val,
                'psnr': psnr_val,
                'span_compressed': span_comp_val,
                'span_reconstructed': span_rec_val,
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
            f.write(f"Span compressed: {item['span_compressed']:.4f}\n")
            f.write(f"Span reconstructed: {item['span_reconstructed']:.4f}\n")

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
    return compressor, opt_comp, decompressor, opt_decomp, avg_diff, avg_total, avg_psnr, avg_span_comp, avg_span_rec

# ---------------------- Тестирование ----------------------
def collect_test_data(compressor, decompressor, dataset):
    compressor.eval()
    decompressor.eval()
    random.seed(TEST_SEED)
    indices = random.sample(range(len(dataset)), min(NUM_TEST_EXAMPLES, len(dataset)))
    results = []
    with torch.no_grad():
        for idx in indices:
            parnet, _ = dataset[idx]   # распаковка (parnet, index)
            fname = os.path.basename(dataset.files[idx])
            eid = os.path.splitext(fname)[0]
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
            span_comp_val = compute_span(compressed)
            span_rec_val = compute_span(reconstructed)
            results.append({
                'example_id': eid,
                'original_parnet': parnet_decomp.squeeze(0).cpu(),
                'reconstructed_parnet': reconstructed.squeeze(0).cpu(),
                'diff': diff_val,
                'psnr': psnr_val,
                'span_compressed': span_comp_val,
                'span_reconstructed': span_rec_val,
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
        f.write(f"Span compressed: {item['span_compressed']:.4f}\n")
        f.write(f"Span reconstructed: {item['span_reconstructed']:.4f}\n")

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

# ---------------------- Обучение (с буфером схожести) ----------------------
def train_epoch(compressor, decompressor, train_loader, opt_comp, opt_decomp, buffer):
    compressor.train()
    decompressor.train()
    total_loss_epoch = 0.0
    total_span_comp_epoch = 0.0
    total_span_rec_epoch = 0.0
    total_sim_epoch = 0.0
    n_batches = len(train_loader)

    for batch_idx, (parnets, indices) in enumerate(train_loader):
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
            compressed = compressor(parnet_comp)  # [B,4,H/2,W/2]
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
        span_comp_batch = compute_span(compressed)
        span_rec_batch = compute_span(rec)
        del compressed, rec, loss_1

        # Фаза 2: компрессор + схожесть
        for p in compressor.parameters():
            p.requires_grad = True
        for p in decompressor.parameters():
            p.requires_grad = False
        opt_comp.zero_grad()

        compressed = compressor(parnet_comp)
        compressed_flat = compressed.view(compressed.shape[0], -1)
        sim_loss, sim_pairs = buffer.compute_loss(compressed_flat, indices)

        if COMPRESSOR_DEVICE != DECOMPRESSOR_DEVICE:
            compressed = compressed.to(DECOMPRESSOR_DEVICE)
        rec = decompressor(compressed)
        loss_2 = PARNET_DIFF_LOSS_WEIGHT * difference_loss(rec, parnet_decomp) + SIMILARITY_LOSS_WEIGHT * sim_loss
        loss_2.backward()
        for name, param in compressor.named_parameters():
            if param.grad is not None:
                saved_grads_comp[name] = param.grad.clone().cpu()
        opt_comp.zero_grad()
        loss2_val = loss_2.item()
        sim_loss_val = sim_loss.item()
        compressed_flat_cpu = compressed_flat.detach().cpu()
        del compressed, rec, loss_2, compressed_flat

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

        # Добавляем текущий батч в буфер (CPU)
        buffer.add(compressed_flat_cpu, indices)

        total_loss_epoch += loss1_val + loss2_val
        total_span_comp_epoch += span_comp_batch
        total_span_rec_epoch += span_rec_batch
        total_sim_epoch += sim_loss_val

        print(f"Batch {batch_idx+1}/{n_batches} | "
              f"Ph1 Diff: {loss1_val:.6f} | Ph2 Diff: {loss2_val:.6f} | "
              f"Span comp: {span_comp_batch:.4f} | Span rec: {span_rec_batch:.4f} | "
              f"Sim: {sim_loss_val:.6f} (pairs: {sim_pairs})")

        del parnets, indices, parnet_comp, parnet_decomp, compressed_flat_cpu
        if CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss_epoch / (2 * n_batches)
    avg_span_comp = total_span_comp_epoch / n_batches
    avg_span_rec = total_span_rec_epoch / n_batches
    avg_sim = total_sim_epoch / n_batches
    return avg_loss, avg_span_comp, avg_span_rec, avg_sim

def train():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Список файлов – сортировка по имени (лексикографически), поддерживает буквы и цифры
    all_files = sorted(
        [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.pt')],
        key=lambda x: os.path.basename(x)
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
    train_dataset = IndexedParnetDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=0)
    val_loader = None
    if val_files:
        val_dataset = IndexedParnetDataset(val_files)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, pin_memory=True, num_workers=0)

    # Загрузка карты соседей
    neighbor_map = {}
    if os.path.exists(SIMILARITIES_FILE):
        print("Загрузка данных о схожести изображений...")
        data = torch.load(SIMILARITIES_FILE, map_location='cpu', weights_only=False)
        file_to_neighbors = {}
        for item in data:
            file = item['file']
            file_to_neighbors[file] = [n[0] for n in item['neighbors']]
        train_file_to_idx = {os.path.basename(f): i for i, f in enumerate(train_files)}
        for f in train_files:
            fname = os.path.basename(f)
            if fname in file_to_neighbors:
                neighbor_files = file_to_neighbors[fname]
                local_idx = train_file_to_idx[fname]
                neighbor_indices = set()
                for nf in neighbor_files:
                    if nf in train_file_to_idx:
                        neighbor_indices.add(train_file_to_idx[nf])
                if neighbor_indices:
                    neighbor_map[local_idx] = neighbor_indices
        print(f"Построена карта соседей для {len(neighbor_map)} тренировочных изображений.")
    else:
        print("Файл similarities.pt не найден. Потеря схожести будет отключена.")

    compressor = ParnetCompressor(**COMPRESSOR_CONFIG).to(COMPRESSOR_DEVICE)
    decompressor = ParnetDecompressor(**DECOMPRESSOR_CONFIG).to(DECOMPRESSOR_DEVICE)
    opt_comp = optim.Adam(compressor.parameters(), lr=LEARNING_RATE)
    opt_decomp = optim.Adam(decompressor.parameters(), lr=LEARNING_RATE)
    start_epoch = load_checkpoints_if_exist(compressor, opt_comp, decompressor, opt_decomp) + 1

    # Создаём буфер сравнения (CPU)
    max_buffer_elements = SIMILARITY_BUFFER_BATCHES * BATCH_SIZE
    buffer = SimilarityLossBuffer(max_buffer_elements, neighbor_map)

    # Прогрев буфера
    warmup_batches = SIMILARITY_WARMUP_BATCHES
    if warmup_batches > 0 and len(train_dataset) > 0:
        print(f"Прогрев буфера на {warmup_batches} батчах...")
        compressor.eval()
        warmup_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   collate_fn=collate_fn, pin_memory=True, num_workers=0)
        with torch.no_grad():
            for idx, (parnets, indices) in enumerate(warmup_loader):
                if idx >= warmup_batches:
                    break
                parnet_comp = parnets.to(COMPRESSOR_DEVICE)
                compressed = compressor(parnet_comp)
                flat_compressed = compressed.view(compressed.shape[0], -1)
                buffer.add(flat_compressed, indices)
                print(f"  добавлен батч {idx+1}/{warmup_batches}")
        compressor.train()
        print("Прогрев завершён.")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_total, avg_span_comp, avg_span_rec, avg_sim = train_epoch(
            compressor, decompressor, train_loader, opt_comp, opt_decomp, buffer
        )
        print(f"Epoch {epoch:3d} Average Total: {avg_total:.6f} | Avg Span comp: {avg_span_comp:.4f} | "
              f"Avg Span rec: {avg_span_rec:.4f} | Avg Sim: {avg_sim:.6f}")

        if val_loader and epoch % VAL_EVERY_EPOCHS == 0:
            print(f"Running validation for epoch {epoch}...")
            compressor, opt_comp, decompressor, opt_decomp, val_diff, val_total, val_psnr, val_span_comp, val_span_rec = run_validation(
                compressor, opt_comp, decompressor, opt_decomp, val_loader, epoch
            )
            print(f"Epoch {epoch:3d} VAL Diff: {val_diff:.6f} Total: {val_total:.6f} PSNR: {val_psnr:.2f} dB | "
                  f"Span comp: {val_span_comp:.4f} Span rec: {val_span_rec:.4f}")

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