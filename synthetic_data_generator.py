"""
Генератор синтетических данных для A/B-теста с нейромаркетингом.
1. Скачивает и распаковывает реальные датасеты (если их нет).
2. Анализирует UEyes для параметров фиксаций.
3. Генерирует синтетические фиксации и клики, имитируя A/B-тест рекламных баннеров
   с разными макетами (A – кнопка внизу, B – кнопка справа).
"""

import pandas as pd
import numpy as np
import os
import json
import zipfile
import requests
import shutil
import sys
import subprocess
from config import *

# ------------------------------------------------------------
# 0. Загрузка реальных датасетов
# ------------------------------------------------------------
def download_file(url, dest_path, chunk_size=8192):
    """Скачивает файл с индикатором прогресса."""
    print(f"  Скачиваю {os.path.basename(dest_path)} ...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r    {downloaded/total*100:.1f}%", end='')
            print("\n  Готово.")
    except Exception as e:
        print(f"\n  Ошибка при скачивании: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

def ensure_ueyes_dataset():
    """Проверяет наличие UEyes, при необходимости скачивает и распаковывает."""
    target_dir = "data/ueyes_raw/UEyes_dataset"
    zip_path = "data/UEyes_dataset.zip"

    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"  UEyes уже распакован: {target_dir}")
        return

    if os.path.isdir(target_dir) and not os.listdir(target_dir):
        os.rmdir(target_dir)

    if not os.path.exists(zip_path):
        print("  UEyes не найден. Загрузка с Zenodo (~12.9 ГБ)...")
        url = "https://zenodo.org/records/8010312/files/UEyes_dataset.zip?download=1"
        os.makedirs("data", exist_ok=True)
        download_file(url, zip_path)
    else:
        print(f"  Найден архив: {zip_path}")

    print(f"  Распаковываю {zip_path} в data/ueyes_raw/ ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall("data/ueyes_raw/")
    print("  UEyes готов.")

def ensure_ab_dataset():
    """
    Проверяет наличие датасета A/B-теста рекламных кампаний.
    При необходимости скачивает с Kaggle (amirmotefaker/ab-testing-dataset).
    """
    target_csv = AB_REAL_DATASET
    if os.path.exists(target_csv):
        print(f"  Датасет рекламных кампаний уже загружен: {target_csv}")
        return

    print("  Скачиваю A/B Testing Dataset (Control vs Test Campaign) с Kaggle ...")
    try:
        import kagglehub
    except ImportError:
        print("    Устанавливаю kagglehub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    try:
        path = kagglehub.dataset_download("amirmotefaker/ab-testing-dataset")
        print(f"    Данные сохранены: {path}")
    except Exception as e:
        print(f"    Ошибка загрузки с Kaggle: {e}")
        return

    os.makedirs("data/ab_real", exist_ok=True)
    csv_found = None
    for f in os.listdir(path):
        if f.endswith('.csv'):
            csv_found = os.path.join(path, f)
            break
    if csv_found:
        shutil.copy(csv_found, target_csv)
        print(f"    Скопирован {os.path.basename(csv_found)} в {target_csv}")
    else:
        print("    CSV-файл не найден в скачанном датасете.")

# ------------------------------------------------------------
# 1. Анализ UEyes
# ------------------------------------------------------------
def load_real_ueyes_stats(ueyes_log_dir=UEYES_LOG_DIR):
    """
    Загружает реальные фиксации из UEyes и возвращает словарь со средними и std.
    Если данных нет – возвращает реалистичные умолчания.
    """
    defaults = {
        "fix_duration_mean": 0.28,
        "fix_duration_std": 0.12,
        "fix_count_mean": 22,
        "fix_count_std": 8,
    }

    if not os.path.isdir(ueyes_log_dir):
        print(f"  UEyes не найден ({ueyes_log_dir}), использую умолчания.")
        return defaults

    fixation_files = []
    for root, _, files in os.walk(ueyes_log_dir):
        for f in files:
            if f.endswith(".csv") and "fixation" in f.lower():
                path = os.path.join(root, f)
                if os.path.getsize(path) >= 100:
                    fixation_files.append(path)

    if not fixation_files:
        print(f"  В {ueyes_log_dir} нет файлов фиксаций, использую умолчания.")
        return defaults

    durations = []
    counts_per_stimulus = []
    for fp in fixation_files[:30]:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        dur_col = None
        for col in df.columns:
            if col.upper().strip() in ['FPOGD', 'FPOG_DURATION', 'FIXATION_DURATION']:
                dur_col = col
                break
        if dur_col:
            d = pd.to_numeric(df[dur_col], errors='coerce').dropna()
            if len(d) > 0:
                durations.extend(d.tolist())
                counts_per_stimulus.append(len(d))

    if durations:
        defaults["fix_duration_mean"] = np.mean(durations)
        defaults["fix_duration_std"] = np.std(durations)
        defaults["fix_count_mean"] = np.mean(counts_per_stimulus)
        defaults["fix_count_std"] = np.std(counts_per_stimulus)
        print(f"  UEyes: обработано {len(durations)} фиксаций.")
    else:
        print("  Не удалось извлечь длительности из UEyes, использую умолчания.")

    return defaults

# ------------------------------------------------------------
# 2. Генерация синтетических данных
# ------------------------------------------------------------
def generate_users(n_users=N_USERS):
    """Создаёт синтетических пользователей."""
    users = pd.DataFrame({
        'participant_id': [f"P{i:04d}" for i in range(n_users)],
        'variant': np.random.choice(['A', 'B'], size=n_users, p=[0.5, 0.5])
    })
    users['task_id'] = np.where(
        users['variant'] == 'A',
        np.random.choice([f"banner_A_{i}" for i in range(4)], size=n_users),
        np.random.choice([f"banner_B_{i}" for i in range(4)], size=n_users)
    )
    return users

def generate_fixations(users, aoi_bboxes, ueyes_stats, ttff_delay=0):
    """
    Генерирует фиксации взгляда.
    ttff_delay добавляет задержку (сек) к первой фиксации на CTA для каждого пользователя.
    """
    aoi_names = list(aoi_bboxes.keys())
    fixations = []
    delay_applied = {}  # для каждого участника: была ли уже задержка

    for _, user in users.iterrows():
        variant = user['variant']
        task = user['task_id']
        n_fix = max(1, int(np.random.lognormal(
            mean=np.log(ueyes_stats['fix_count_mean']), sigma=0.3)))
        if variant == 'B':
            n_fix = int(n_fix * EFFECT_FIXATION_COUNT_LIFT)

        for i in range(n_fix):
            # Вероятности попадания в зоны (для B кнопка справа менее заметна)
            if variant == 'B':
                probs = [0.25, 0.35, 0.15, 0.25]  # headline, image, cta_button, background
            else:
                probs = [0.25, 0.35, 0.20, 0.20]  # классическое распределение

            aoi = np.random.choice(aoi_names, p=probs)
            bbox = aoi_bboxes[aoi]
            x = np.random.uniform(bbox[0], bbox[0] + bbox[2])
            y = np.random.uniform(bbox[1], bbox[1] + bbox[3])
            base_dur = np.random.lognormal(
                mean=np.log(ueyes_stats['fix_duration_mean']), sigma=0.3)
            if variant == 'B' and aoi == 'cta_button':
                base_dur *= EFFECT_DWELL_CTA_LIFT
            duration = max(0.05, base_dur)
            timestamp = np.random.uniform(0, 30)

            # Задержка первого взгляда на CTA (для варианта B)
            if ttff_delay > 0 and aoi == 'cta_button':
                user_key = user['participant_id']
                if user_key not in delay_applied:
                    timestamp += ttff_delay
                    delay_applied[user_key] = True

            fixations.append({
                'participant_id': user['participant_id'],
                'task_id': task,
                'variant': variant,
                'fixation_id': i,
                'gaze_x': x,
                'gaze_y': y,
                'fixation_duration': duration,
                'timestamp': timestamp,
                'aoi': aoi
            })
    return pd.DataFrame(fixations)

def generate_clicks(users, aoi_bboxes, base_conv):
    """Генерирует события кликов. Координаты мыши берутся из aoi_bboxes['cta_button']."""
    clicks = []
    for _, user in users.iterrows():
        variant = user['variant']
        conv_prob = base_conv + (EFFECT_CONVERSION_LIFT if variant == 'B' else 0)
        if np.random.random() < conv_prob:
            bbox = aoi_bboxes['cta_button']
            x = np.random.uniform(bbox[0], bbox[0] + bbox[2])
            y = np.random.uniform(bbox[1], bbox[1] + bbox[3])
            clicks.append({
                'participant_id': user['participant_id'],
                'task_id': user['task_id'],
                'variant': variant,
                'event_type': 'click',
                'mouse_x': x,
                'mouse_y': y,
                'timestamp': np.random.uniform(5, 30)
            })
    return pd.DataFrame(clicks)

# ------------------------------------------------------------
# 3. Главная функция
# ------------------------------------------------------------
def main():
    print("=== Генератор синтетического A/B-теста с нейромаркетингом ===\n")
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[0/3] Проверка и загрузка реальных датасетов...")
    ensure_ueyes_dataset()
    ensure_ab_dataset()          # датасет скачивается, но конверсию берём из BASE_CONVERSION

    print("\n[1/3] Анализ UEyes...")
    ueyes_stats = load_real_ueyes_stats()

    print(f"\n[2/3] Создание синтетических пользователей (конверсия A ≈ {BASE_CONVERSION:.0%}, B ≈ {BASE_CONVERSION+EFFECT_CONVERSION_LIFT:.0%})...")
    users = generate_users()
    users.to_csv(SYNTHETIC_USERS, index=False)
    n_A = (users['variant'] == 'A').sum()
    n_B = (users['variant'] == 'B').sum()
    print(f"  Пользователей: {len(users)} (A: {n_A}, B: {n_B})")

    print("\n[3/3] Генерация фиксаций и кликов...")
    users_A = users[users['variant'] == 'A']
    users_B = users[users['variant'] == 'B']

    fix_A = generate_fixations(users_A, AOI_BBOXES_A, ueyes_stats, ttff_delay=0)
    fix_B = generate_fixations(users_B, AOI_BBOXES_B, ueyes_stats, ttff_delay=3.0)
    fix_df = pd.concat([fix_A, fix_B], ignore_index=True)
    fix_df.to_csv(SYNTHETIC_FIXATIONS, index=False)
    print(f"  Фиксаций: {len(fix_df)} (A: {len(fix_A)}, B: {len(fix_B)})")

    click_df = generate_clicks(users, AOI_BBOXES_A, BASE_CONVERSION)
    click_df.to_csv(SYNTHETIC_CLICKS, index=False)
    print(f"  Кликов: {len(click_df)}")

    with open(AOI_DEFINITIONS, 'w') as f:
        json.dump({"A": AOI_BBOXES_A, "B": AOI_BBOXES_B}, f, indent=2)

    print("\n--- Сводка синтетического датасета ---")
    for var in ['A', 'B']:
        n_usr = (users['variant'] == var).sum()
        n_clk = click_df[click_df['variant'] == var].shape[0]
        print(f"  Вариант {var}: конверсия {n_clk/n_usr:.3f}")

    print("\nСинтетический датасет сохранён в папку data/")

if __name__ == "__main__":
    main()