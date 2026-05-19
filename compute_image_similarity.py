# compute_image_similarity.py
"""
Вычисляет попарную схожесть изображений из prepared_dataset,
используя MSE, SSIM и Jaccard-метрику по тегам.

Конвейерная обработка с параллельной подготовкой батчей на CPU
и контролем памяти (воркеры блокируются на полной очереди q_prepare).
Логируются только ключевые этапы сравнения.
"""

import torch
import torch.nn.functional as F
import os
import gc
import time
import threading
import queue
from datetime import datetime

# ---------------------- Конфигурация ----------------------
DATASET_DIR = "./prepared_dataset"
OUTPUT_PATH = os.path.join(DATASET_DIR, "similarities.pt")
TOP_K = 50
TEMPERATURE = 1.0

WEIGHT_MSE = 1.0
WEIGHT_SSIM = 1.0
WEIGHT_TAGS = 1.0

SSIM_WINDOW_SIZE = 11
SSIM_C1 = 0.01 ** 2
SSIM_C2 = 0.03 ** 2

BATCH_SIZE = 64            # якорь + до BATCH_SIZE-1 целей
USE_CPU = False

# Размеры очередей
QUEUE_PREPARE_MAXSIZE = 8   # очередь готовых CPU-батчей
QUEUE_DEVICE_MAXSIZE = 4    # очередь батчей на устройстве
QUEUE_TASKS_MAXSIZE = 16     # очередь заданий для воркеров (лёгкие кортежи)

# Количество потоков-воркеров для подготовки батчей
NUM_PREPARE_WORKERS = 2

# ---------------------- Вспомогательные функции ----------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} сек"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins} мин {secs:.1f} сек"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours} ч {mins} мин"

# ---------------------- SSIM ----------------------
def gaussian_window(size, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g

def create_window(window_size, channels):
    _1d = gaussian_window(window_size)
    _2d = _1d[:, None] * _1d[None, :]
    window = _2d.expand(channels, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window, window_size, C1, C2):
    B1, C, H, W = img1.shape
    B2 = img2.shape[0]
    img1_rep = img1.unsqueeze(1).expand(-1, B2, -1, -1, -1).reshape(B1 * B2, C, H, W)
    img2_rep = img2.unsqueeze(0).expand(B1, -1, -1, -1, -1).reshape(B1 * B2, C, H, W)
    window = window.to(img1.device)
    mu1 = F.conv2d(img1_rep, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2_rep, window, padding=window_size//2, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1_rep * img1_rep, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2_rep * img2_rep, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1_rep * img2_rep, window, padding=window_size//2, groups=C) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_val = ssim_map.mean(dim=[1, 2, 3])
    return ssim_val.view(B1, B2)

# ---------------------- Поток загрузки на устройство ----------------------
class BatchLoader(threading.Thread):
    def __init__(self, queue_in, queue_out, device):
        super().__init__(daemon=True)
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.device = device

    def run(self):
        while True:
            item = self.queue_in.get()
            if item is None:
                self.queue_out.put(None)
                break
            i, j_start, j_end, batch_cpu, batch_num, total_batches = item
            batch_dev = batch_cpu.to(self.device, non_blocking=True)
            self.queue_out.put((i, j_start, j_end, batch_dev, batch_num, total_batches))

# ---------------------- Поток вычисления ----------------------
class Computer(threading.Thread):
    def __init__(self, queue_in, N, all_mse, all_ssim, window, device, total_pairs, start_time):
        super().__init__()
        self.queue_in = queue_in
        self.N = N
        self.all_mse = all_mse
        self.all_ssim = all_ssim
        self.window = window
        self.device = device
        self.total_pairs = total_pairs
        self.start_time = start_time
        self.pairs_processed = 0
        self.current_anchor = -1
        self.anchor_batch_counter = 0

    def run(self):
        while True:
            item = self.queue_in.get()
            if item is None:
                break
            i, j_start, j_end, batch_dev, batch_num, total_batches = item

            if i != self.current_anchor:
                if self.current_anchor != -1:
                    log(f"  [якорь {self.current_anchor}] завершён, переход к изображению {i}")
                self.current_anchor = i
                self.anchor_batch_counter = 0

            anchor = batch_dev[0:1]
            targets = batch_dev[1:]

            diff = anchor - targets
            mse = (diff ** 2).mean(dim=(1,2,3))
            for idx, j in enumerate(range(j_start, j_end)):
                self.all_mse[i, j] = mse[idx].cpu()
                self.all_mse[j, i] = self.all_mse[i, j]

            ssim_vals = ssim(anchor, targets, self.window, SSIM_WINDOW_SIZE, SSIM_C1, SSIM_C2)
            ssim_vals = ssim_vals.squeeze(0)
            for idx, j in enumerate(range(j_start, j_end)):
                self.all_ssim[i, j] = ssim_vals[idx].cpu()
                self.all_ssim[j, i] = self.all_ssim[i, j]

            del batch_dev, anchor, targets, diff, mse, ssim_vals

            self.pairs_processed += (j_end - j_start)
            self.anchor_batch_counter += 1

            if self.anchor_batch_counter % 10 == 0:
                remaining = total_batches - self.anchor_batch_counter
                log(f"  [якорь {i}] батч {self.anchor_batch_counter}/{total_batches}, осталось батчей: {remaining}")

            if self.pairs_processed % max(1, self.total_pairs // 20) == 0 or self.pairs_processed == self.total_pairs:
                progress = self.pairs_processed / self.total_pairs * 100
                elapsed = time.time() - self.start_time
                eta = elapsed / progress * (100 - progress) if progress > 0 else 0
                log(f"Общий прогресс: {self.pairs_processed}/{self.total_pairs} пар ({progress:.1f}%), "
                    f"прошло {format_time(elapsed)}, осталось {format_time(eta)}")

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if self.current_anchor != -1:
            log(f"  [якорь {self.current_anchor}] завершён (последний)")

# ---------------------- Загрузка данных ----------------------
def load_images_and_tags(directory):
    files = sorted([f for f in os.listdir(directory) if f.endswith('.pt')], key=lambda x: x)
    images = []
    tags_list = []
    file_names = []
    for f in files:
        data = torch.load(os.path.join(directory, f), map_location='cpu', weights_only=False)
        img = data['image']
        tags = data.get('tags', [])
        images.append(img)
        tags_list.append(tags)
        file_names.append(f)
    return images, file_names, tags_list

# ---------------------- Основной расчёт ----------------------
def main():
    start_time = time.time()
    log("Загрузка изображений и тегов в RAM...")
    img_list, file_names, tags_list = load_images_and_tags(DATASET_DIR)
    N = len(img_list)
    log(f"Загружено {N} изображений.")

    imgs_cpu = torch.stack(img_list, dim=0)
    del img_list
    gc.collect()
    log(f"Общий тензор: {imgs_cpu.shape}")

    if USE_CPU:
        device = torch.device('cpu')
        log("Режим: CPU")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log(f"Режим: {device}")

    window = create_window(SSIM_WINDOW_SIZE, imgs_cpu.shape[1])

    all_mse = torch.zeros(N, N, device='cpu')
    all_ssim = torch.zeros(N, N, device='cpu')

    total_pairs = N * (N - 1) // 2
    log(f"Начало сравнений (BATCH_SIZE={BATCH_SIZE}, всего пар {total_pairs})")

    # Очереди
    q_prepare = queue.Queue(maxsize=QUEUE_PREPARE_MAXSIZE)
    q_device = queue.Queue(maxsize=QUEUE_DEVICE_MAXSIZE)
    task_queue = queue.Queue(maxsize=QUEUE_TASKS_MAXSIZE)

    # Генерация всех заданий (лёгкие кортежи)
    tasks = []
    for i in range(N - 1):
        j_start = i + 1
        targets_left = N - (i + 1)
        batches_for_anchor = (targets_left + BATCH_SIZE - 2) // (BATCH_SIZE - 1)
        batch_num = 0
        while j_start < N:
            j_end = min(j_start + (BATCH_SIZE - 1), N)
            batch_indices = [i] + list(range(j_start, j_end))
            batch_num += 1
            tasks.append((i, j_start, j_end, batch_indices, batch_num, batches_for_anchor))
            j_start = j_end

    # Функция воркера: берёт задание, готовит тензор и кладёт в q_prepare
    def worker():
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                break
            i, j_start, j_end, indices, bnum, btotal = task
            batch_cpu = imgs_cpu[indices]  # CPU тензор
            q_prepare.put((i, j_start, j_end, batch_cpu, bnum, btotal))
            task_queue.task_done()

    # Запускаем воркеры
    workers = []
    for _ in range(NUM_PREPARE_WORKERS):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        workers.append(t)

    # Запускаем загрузчик и вычислитель
    loader = BatchLoader(q_prepare, q_device, device)
    computer = Computer(q_device, N, all_mse, all_ssim, window, device, total_pairs, start_time)
    loader.start()
    computer.start()

    # Подаём задания в task_queue
    for task in tasks:
        task_queue.put(task)  # блокируется, если очередь заданий полна
    # Сигнал завершения воркерам
    for _ in workers:
        task_queue.put(None)
    # Ждём завершения всех заданий
    task_queue.join()
    # Сигнал завершения в q_prepare
    q_prepare.put(None)

    loader.join()
    computer.join()

    del imgs_cpu
    gc.collect()

    # Теги
    log("Вычисление Jaccard-расстояний...")
    dist_tags = torch.zeros(N, N, device='cpu')
    for i in range(N):
        tags_i = set(tags_list[i])
        for j in range(i+1, N):
            tags_j = set(tags_list[j])
            if not tags_i and not tags_j:
                d = 0.0
            elif not tags_i or not tags_j:
                d = 1.0
            else:
                inter = len(tags_i & tags_j)
                union = len(tags_i | tags_j)
                d = 1.0 - (inter / union) if union > 0 else 0.0
            dist_tags[i, j] = d
            dist_tags[j, i] = d

    log("Нормировка и комбинирование...")
    dist_mse = all_mse
    dist_ssim = 1.0 - all_ssim
    mask = ~torch.eye(N, dtype=bool)
    mean_mse = dist_mse[mask].mean()
    mean_ssim = dist_ssim[mask].mean()
    mean_tags = dist_tags[mask].mean()
    log(f"  Средние расстояния: MSE={mean_mse:.6f}, SSIM={mean_ssim:.6f}, Tags={mean_tags:.6f}")

    norm_mse = dist_mse / mean_mse
    norm_ssim = dist_ssim / mean_ssim
    norm_tags = dist_tags / mean_tags
    combined_dist = (WEIGHT_MSE * norm_mse +
                     WEIGHT_SSIM * norm_ssim +
                     WEIGHT_TAGS * norm_tags)
    similarity = torch.exp(-combined_dist / TEMPERATURE)
    similarity.fill_diagonal_(0.0)

    log("Выбор ближайших соседей...")
    neighbors = []
    for i in range(N):
        if i % max(1, N // 10) == 0:
            log(f"  Отбор соседей: {i}/{N}")
        vals, idx = similarity[i].topk(min(TOP_K, N-1))
        neighbors.append({
            'file': file_names[i],
            'index': i,
            'neighbors': [(file_names[j], vals[k].item()) for k, j in enumerate(idx)]
        })

    torch.save(neighbors, OUTPUT_PATH)
    total_time = time.time() - start_time
    log(f"Результат сохранён в {OUTPUT_PATH}")
    log(f"Общее время: {format_time(total_time)}")

if __name__ == "__main__":
    main()