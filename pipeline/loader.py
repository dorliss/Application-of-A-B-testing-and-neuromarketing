import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
from config import SYNTHETIC_FIXATIONS, SYNTHETIC_CLICKS, AOI_DEFINITIONS

def load_synthetic_data():
    print("[1/5] Загрузка синтетических данных (A/B тест + нейромаркетинг)...")
    fixations_df = pd.read_csv(SYNTHETIC_FIXATIONS)
    clicks_df = pd.read_csv(SYNTHETIC_CLICKS)

    try:
        with open(AOI_DEFINITIONS, 'r') as f:
            aoi_data = json.load(f)
        if 'A' in aoi_data and 'B' in aoi_data:
            aoi_dict = aoi_data
            print("  Разметка AOI для вариантов A и B загружена.")
        else:
            aoi_dict = {'A': aoi_data, 'B': aoi_data}
            print("  Разметка AOI загружена (единая).")
    except FileNotFoundError:
        print("  Файл разметки AOI не найден. Будет использована заглушка.")
        aoi_dict = None

    print(f"  Загружено фиксаций: {len(fixations_df)}")
    print(f"  Загружено событий кликов: {len(clicks_df)}")
    return fixations_df, clicks_df, aoi_dict