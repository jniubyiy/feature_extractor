# analyze_dataset.py
"""
Анализирует similarities.pt и .pt файлы изображений из prepared_dataset.
Выводит статистику в консоль:
1. Общая информация о схожести.
2. Топ популярных тегов.
3. Топ изображений, наиболее похожих на остальные.
4. Топ самых редких тегов.
5. (пропущен)
6. Топ изображений, наименее похожих на остальные.
"""

import os
import torch
from collections import Counter
from pathlib import Path
import sys

# ---------------------- Конфигурация ----------------------
PREPARED_DATASET_DIR = "./prepared_dataset"
SIMILARITIES_PATH = os.path.join(PREPARED_DATASET_DIR, "similarities.pt")
TOP_N = 20  # сколько элементов показывать в топах

def load_similarities(path):
    """Загружает данные о схожести из similarities.pt"""
    if not os.path.exists(path):
        print(f"Ошибка: файл {path} не найден.")
        sys.exit(1)
    data = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Загружено {len(data)} записей о схожести.")
    return data

def load_tags(directory):
    """Извлекает теги из всех .pt файлов в директории.
    Возвращает словарь {file_name: list_of_tags} и множество всех уникальных тегов."""
    tags_dict = {}
    all_tags = []
    for f in sorted(os.listdir(directory)):
        if not f.endswith('.pt'):
            continue
        path = os.path.join(directory, f)
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            tags = data.get('tags', [])
            tags_dict[f] = tags
            all_tags.extend(tags)
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить {f}: {e}")
    print(f"Загружено тегов: {len(tags_dict)} изображений, всего тегов: {len(all_tags)}")
    return tags_dict, all_tags

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def main():
    print("Загрузка данных о схожести...")
    sim_data = load_similarities(SIMILARITIES_PATH)
    print("Загрузка тегов...")
    tags_dict, all_tags = load_tags(PREPARED_DATASET_DIR)

    # 1. Общая статистика схожести
    print_section("1. Общая статистика схожести")
    all_scores = []
    total_pairs = 0
    for entry in sim_data:
        for _, score in entry['neighbors']:
            all_scores.append(score)
            total_pairs += 1
    if all_scores:
        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        mean_sim = scores_t.mean().item()
        std_sim = scores_t.std().item()
        min_sim = scores_t.min().item()
        max_sim = scores_t.max().item()
        median_sim = scores_t.median().item()
        # перцентили
        q25 = scores_t.quantile(0.25).item()
        q75 = scores_t.quantile(0.75).item()
        q90 = scores_t.quantile(0.9).item()
        print(f"Всего пар (top-K): {total_pairs}")
        print(f"Средняя схожесть: {mean_sim:.4f} ± {std_sim:.4f}")
        print(f"Медиана: {median_sim:.4f}")
        print(f"Минимальная: {min_sim:.4f}, Максимальная: {max_sim:.4f}")
        print(f"Квартили: 25% = {q25:.4f}, 75% = {q75:.4f}, 90% = {q90:.4f}")
        # гистограмма (текстовая)
        bins = 10
        hist = torch.histc(scores_t, bins=bins, min=0, max=1)
        print("\nРаспределение (приблизительное):")
        bin_width = 1.0 / bins
        for i in range(bins):
            left = i * bin_width
            right = (i+1) * bin_width
            count = hist[i].item()
            bar = '#' * int(count / max(1, max(hist).item()) * 40)
            print(f"  [{left:.2f}, {right:.2f}]: {int(count):5d} {bar}")
    else:
        print("Нет данных о схожести.")

    # 2. Топ популярных тегов
    print_section(f"2. Топ {TOP_N} популярных тегов")
    tag_counter = Counter(all_tags)
    if tag_counter:
        print(f"Всего уникальных тегов: {len(tag_counter)}")
        for i, (tag, count) in enumerate(tag_counter.most_common(TOP_N), 1):
            print(f"  {i:2}. {tag:30s} – {count} раз(а)")
    else:
        print("Теги отсутствуют.")

    # 3. Топ изображений, наиболее похожих на остальные
    print_section(f"3. Топ {TOP_N} изображений, наиболее похожих на остальные")
    # Считаем суммарную схожесть по соседям
    image_sum_sim = {}
    for entry in sim_data:
        fname = entry['file']
        total = sum(score for _, score in entry['neighbors'])
        image_sum_sim[fname] = total
    sorted_by_sim = sorted(image_sum_sim.items(), key=lambda x: x[1], reverse=True)
    for i, (fname, total) in enumerate(sorted_by_sim[:TOP_N], 1):
        # также выведем топ-теги этого изображения
        tags = tags_dict.get(fname, [])
        tag_str = ", ".join(tags[:5]) if tags else "нет тегов"
        print(f"  {i:2}. {fname:20s} суммарная схожесть: {total:.4f}  | теги: {tag_str}")

    # 4. Топ самых редких тегов
    print_section(f"4. Топ {TOP_N} самых редких тегов")
    if tag_counter:
        # редкие – наименьшая частота
        rare_tags = sorted(tag_counter.items(), key=lambda x: x[1])
        # показываем теги с наименьшей частотой, если их много, ограничим TOP_N
        min_freq = rare_tags[0][1] if rare_tags else 0
        rare_selected = [item for item in rare_tags if item[1] == min_freq][:TOP_N]
        if len(rare_selected) < TOP_N:
            # дополним следующими по частоте
            current_freq = min_freq + 1
            while len(rare_selected) < TOP_N and current_freq <= rare_tags[-1][1]:
                extras = [item for item in rare_tags if item[1] == current_freq]
                rare_selected.extend(extras[:TOP_N - len(rare_selected)])
                current_freq += 1
        for i, (tag, count) in enumerate(rare_selected[:TOP_N], 1):
            print(f"  {i:2}. {tag:30s} – {count} раз(а)")
        if len(rare_selected) < TOP_N:
            print("  (больше тегов нет)")
    else:
        print("Теги отсутствуют.")

    # 6. Топ изображений, наименее похожих на остальные
    print_section(f"6. Топ {TOP_N} изображений, наименее похожих на остальные")
    sorted_least_sim = sorted(image_sum_sim.items(), key=lambda x: x[1])
    for i, (fname, total) in enumerate(sorted_least_sim[:TOP_N], 1):
        tags = tags_dict.get(fname, [])
        tag_str = ", ".join(tags[:5]) if tags else "нет тегов"
        print(f"  {i:2}. {fname:20s} суммарная схожесть: {total:.4f}  | теги: {tag_str}")

if __name__ == "__main__":
    main()