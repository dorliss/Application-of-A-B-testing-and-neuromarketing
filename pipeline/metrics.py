import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from config import SYNTHETIC_USERS

def compute_static_metrics(clicks_df, users_df=None):
    """
    Статические A/B-метрики: конверсия (conversion rate) по вариантам.
    Возвращает:
      - conv_rates: Series с конверсией по вариантам (A, B)
      - user_conv: DataFrame с колонками [participant_id, variant, converted]
    """
    if users_df is None:
        users_df = pd.read_csv(SYNTHETIC_USERS)

    # Флаг конверсии: есть ли хотя бы один клик у пользователя
    converted = clicks_df.groupby('participant_id').size().reset_index(name='click_count')
    converted['converted'] = 1

    # Присоединяем к полному списку пользователей
    user_conv = users_df.merge(converted[['participant_id', 'converted']], on='participant_id', how='left')
    user_conv['converted'] = user_conv['converted'].fillna(0).astype(int)

    conv_rates = user_conv.groupby('variant')['converted'].mean()
    return conv_rates, user_conv[['participant_id', 'variant', 'converted']]


def compute_neurometrics(fixations_df, aoi='cta_button'):
    """
    Нейромаркетинговые метрики для заданной AOI (по умолчанию CTA-кнопка).
    Для каждой пары (участник, задание) считает:
      - ttff: время до первой фиксации (сек)
      - dwell_time: суммарная длительность фиксаций (сек)
      - fixation_count: количество фиксаций
      - avg_fixation_duration: средняя длительность фиксации (сек)
    Возвращает DataFrame с колонками:
      participant_id, task_id, variant, ttff, dwell_time, fixation_count, avg_fixation_duration
    """
    aoi_fix = fixations_df[fixations_df['aoi'] == aoi].copy()
    if aoi_fix.empty:
        print(f"  Внимание: нет фиксаций для AOI '{aoi}'. Возвращается пустой DataFrame.")
        return pd.DataFrame(columns=['participant_id', 'task_id', 'variant',
                                     'ttff', 'dwell_time', 'fixation_count', 'avg_fixation_duration'])

    # Группируем по участнику, заданию и варианту
    metrics = aoi_fix.groupby(['participant_id', 'task_id', 'variant']).agg(
        ttff=('timestamp', 'min'),                     # первый взгляд на кнопку
        dwell_time=('fixation_duration', 'sum'),       # общее время просмотра
        fixation_count=('fixation_id', 'nunique'),     # число фиксаций
        avg_fixation_duration=('fixation_duration', 'mean')  # средняя длительность
    ).reset_index()

    return metrics


# Быстрая проверка при прямом запуске
if __name__ == "__main__":
    from pipeline.loader import load_synthetic_data
    fix_df, click_df, _ = load_synthetic_data()

    conv_rates, user_conv = compute_static_metrics(click_df)
    print("\nСтатические метрики (конверсия):")
    print(conv_rates)

    neuro = compute_neurometrics(fix_df)
    print("\nНейрометрики (CTA):")
    print(neuro.head())