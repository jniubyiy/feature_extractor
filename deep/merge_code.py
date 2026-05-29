import os
from pathlib import Path

# ============================================================
# 1. Укажите директорию, в которой нужно искать .py файлы
# ============================================================
SOURCE_DIR = r"D:\feature_extractor"   # "." — текущая папка; можно любой путь

# ============================================================
# 2. Папки, которые нужно ПРОПУСКАТЬ (по именам)
# ============================================================
EXCLUDE_DIR_NAMES = [
    "venv",         # виртуальное окружение
    ".git",         # служебная папка Git
    "__pycache__",  # кэш Python
    "node_modules", # если есть
    "archive",      # пример пользовательской папки
    ".vscode",
    "dataset",
    "deep",
    "generated_images",
    "models",
    "models_compressor",
    "models_vae_wrapper",
    "prepared_dataset",
    "prepared_dataset_parnet",
    "prepared_dataset_parnet_compressed",
    "tests",
    "tests_compressor",
    "tests_vae_wrapper",
    "val_tests",
    "val_tests_compressor",
    "val_tests_vae_wrapper",
    ".gitignore"
]

# Имя выходного файла
OUTPUT_FILE = "merged_code.txt"

# Файл с содержимым для вставки в начало (ищется в папке скрипта)
CONTENTS_FILE = "contents.txt"

# Разделители
SEPARATOR = "=" * 80
SUB_SEPARATOR = "-" * 80


def collect_py_files(root_dir, exclude_files=None, exclude_dir_names=None):
    """
    Рекурсивно собирает все .py файлы в root_dir и подпапках,
    исключая папки, имена которых есть в exclude_dir_names.
    """
    if exclude_files is None:
        exclude_files = set()
    if exclude_dir_names is None:
        exclude_dir_names = set()
    else:
        exclude_dir_names = set(exclude_dir_names)

    root = Path(root_dir).resolve()
    py_files = []

    for current_dir, dirs, filenames in os.walk(root):
        # Удаляем исключённые папки из списка обхода
        dirs[:] = [d for d in dirs if d not in exclude_dir_names]

        for fname in filenames:
            if fname.lower().endswith(".py"):
                full_path = (Path(current_dir) / fname).resolve()
                if full_path in exclude_files:
                    continue
                py_files.append(full_path)

    py_files.sort()
    return py_files


def merge_files(py_files, output_path, prologue=""):
    """
    Записывает код всех собранных файлов в output_path с разделителями.
    Если prologue не пуст, он вставляется в самое начало файла.
    """
    processed = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        # Вставка дополнительного содержимого из contents.txt
        if prologue:
            out.write(SEPARATOR + "\n")
            out.write("СОДЕРЖИМОЕ contents.txt\n")
            out.write(SEPARATOR + "\n")
            out.write(prologue.rstrip() + "\n\n")

        # Основной заголовок (без даты, количества файлов и исключённых папок)
        out.write(SEPARATOR + "\n")
        out.write("СБОРКА КОДА ВСЕХ .py ФАЙЛОВ\n")
        out.write(f"Директория-источник: {Path(SOURCE_DIR).resolve()}\n")
        out.write(SEPARATOR + "\n\n")

        for idx, file_path in enumerate(py_files, start=1):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                print(f"[ОШИБКА] Не удалось прочитать {file_path.name}: {e}")
                out.write(SEPARATOR + "\n")
                out.write(f"ФАЙЛ {idx}: {file_path.name}\n")
                out.write(f"Полный путь: {file_path}\n")
                out.write(f"!!! ОШИБКА ЧТЕНИЯ: {e} !!!\n")
                out.write(SEPARATOR + "\n\n")
                skipped += 1
                continue

            if not code.strip():
                code = "<!-- Файл пуст -->"

            out.write(SEPARATOR + "\n")
            out.write(f"ФАЙЛ {idx}: {file_path.name}\n")
            out.write(f"Полный путь: {file_path}\n")
            out.write(SUB_SEPARATOR + "\n")
            out.write(code.rstrip() + "\n")
            out.write(SEPARATOR + "\n")
            out.write(f"КОНЕЦ ФАЙЛА: {file_path.name}\n")
            out.write(SEPARATOR + "\n\n")

            try:
                rel = file_path.relative_to(Path(SOURCE_DIR).resolve())
            except ValueError:
                rel = file_path
            print(f"[OK] Добавлен: {rel}")
            processed += 1

        # Итоговая статистика остаётся
        out.write(SEPARATOR + "\n")
        out.write(f"Обработано успешно: {processed}\n")
        out.write(f"Пропущено / с ошибками: {skipped}\n")
        out.write(SEPARATOR + "\n")

    print(f"\nГотово! Результат сохранён в: {output_path}")
    print(f"Успешно обработано: {processed}, пропущено: {skipped}")


if __name__ == "__main__":
    # Исключаем из сборки сам скрипт и выходной файл
    this_script = Path(__file__).resolve()
    output_abs = Path(OUTPUT_FILE).resolve()
    exclude_files = {this_script, output_abs}

    # Проверяем наличие contents.txt в папке со скриптом
    script_dir = this_script.parent
    contents_path = script_dir / CONTENTS_FILE
    prologue_text = ""
    if contents_path.is_file():
        try:
            prologue_text = contents_path.read_text(encoding="utf-8")
            print(f"[INFO] Найден {CONTENTS_FILE}, его содержимое будет добавлено в начало файла.")
        except Exception as e:
            print(f"[ОШИБКА] Не удалось прочитать {CONTENTS_FILE}: {e}")

    # Собираем .py файлы
    py_files = collect_py_files(
        SOURCE_DIR,
        exclude_files=exclude_files,
        exclude_dir_names=EXCLUDE_DIR_NAMES
    )

    if not py_files:
        print("В указанной директории не найдено ни одного .py файла (с учётом исключений).")
    else:
        merge_files(py_files, OUTPUT_FILE, prologue=prologue_text)