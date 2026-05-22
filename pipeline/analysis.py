import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import OUTPUT_DIR, SCREEN_WIDTH, SCREEN_HEIGHT
from matplotlib import colors

# ------------------------------------------------------------
# Статистические тесты
# ------------------------------------------------------------
def run_ttest(data, metric, group_col='variant'):
    a = data[data[group_col]=='A'][metric]
    b = data[data[group_col]=='B'][metric]
    if len(a) < 2 or len(b) < 2:
        print(f"  Недостаточно данных для теста {metric} (A: {len(a)}, B: {len(b)})")
        return None, None
    t_stat, p_val = stats.ttest_ind(a, b)
    if p_val < 0.001:
        print(f"  T-тест {metric}: t={t_stat:.3f}, p < 0.001")
    else:
        print(f"  T-тест {metric}: t={t_stat:.3f}, p = {p_val:.4f}")
    return t_stat, p_val

def run_proportions_test(conv_rates, n_A, n_B):
    p_A = conv_rates['A']
    p_B = conv_rates['B']
    p_pool = (p_A * n_A + p_B * n_B) / (n_A + n_B)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
    if se == 0:
        print("  Стандартная ошибка равна нулю, тест не выполнен.")
        return None, None
    z_stat = (p_B - p_A) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"  Z-тест конверсий: z={z_stat:.3f}, p={p_val:.4f}")
    return z_stat, p_val

# ------------------------------------------------------------
# Визуализации
# ------------------------------------------------------------
def plot_conversion_bar(conv_rates):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(6,4))
    bars = plt.bar(conv_rates.index, conv_rates.values, color=['gray', 'green'])
    plt.title('Конверсия по вариантам')
    plt.ylabel('Conversion Rate')
    for bar, val in zip(bars, conv_rates.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha='center', va='bottom')
    plt.ylim(0, max(conv_rates.values)*1.2)
    path = os.path.join(OUTPUT_DIR, 'conversion_bar.png')
    plt.savefig(path)
    plt.close()
    print(f"  Сохранён: {path}")

def plot_neurometrics_boxplots(neuro_df):
    if neuro_df.empty:
        print("  Нет данных для нейрометрик.")
        return
    metrics = ['dwell_time', 'fixation_count', 'avg_fixation_duration']
    titles = ['Время на CTA (сек)', 'Число фиксаций на CTA', 'Средняя длительность фиксации (сек)']
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    for ax, metric, title in zip(axes, metrics, titles):
        sns.boxplot(x='variant', y=metric, data=neuro_df, ax=ax,
                    hue='variant', palette=['gray', 'green'], legend=False)
        ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'neurometrics_boxplots.png')
    plt.savefig(path)
    plt.close()
    print(f"  Сохранены боксплоты: {path}")

def plot_neurometrics_histograms(neuro_df):
    if neuro_df.empty:
        return
    metrics = ['dwell_time', 'fixation_count', 'avg_fixation_duration']
    titles = ['Время на CTA (сек)', 'Число фиксаций на CTA', 'Средняя длительность фиксации (сек)']
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    for ax, metric, title in zip(axes, metrics, titles):
        for var, color in [('A', 'gray'), ('B', 'green')]:
            subset = neuro_df[neuro_df['variant']==var][metric]
            sns.histplot(subset, ax=ax, color=color, label=f'Вариант {var}',
                         kde=True, alpha=0.5, bins=20)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'neurometrics_histograms.png')
    plt.savefig(path)
    plt.close()
    print(f"  Сохранены гистограммы: {path}")

def plot_neurometrics_bars(neuro_df):
    if neuro_df.empty:
        return
    metrics = ['dwell_time', 'fixation_count', 'avg_fixation_duration']
    titles = ['Время на CTA (сек)', 'Число фиксаций на CTA', 'Средняя длительность фиксации (сек)']
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    for ax, metric, title in zip(axes, metrics, titles):
        summary = neuro_df.groupby('variant')[metric].agg(['mean', 'std'])
        means = summary['mean']
        stds = summary['std']
        bars = ax.bar(means.index, means.values, yerr=stds.values,
                      color=['gray', 'green'], capsize=5)
        ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'neurometrics_bars.png')
    plt.savefig(path)
    plt.close()
    print(f"  Сохранены столбчатые графики: {path}")

def plot_scatter_dwell_vs_fixcount(neuro_df):
    if neuro_df.empty:
        return
    plt.figure(figsize=(8,6))
    for var, color in [('A', 'gray'), ('B', 'green')]:
        subset = neuro_df[neuro_df['variant']==var]
        plt.scatter(subset['dwell_time'], subset['fixation_count'],
                    c=color, label=f'Вариант {var}', alpha=0.6, edgecolors='k')
    plt.xlabel('Общее время на CTA (сек)')
    plt.ylabel('Число фиксаций на CTA')
    plt.title('Взаимосвязь времени и числа фиксаций на CTA')
    plt.legend()
    path = os.path.join(OUTPUT_DIR, 'scatter_dwell_vs_fixcount.png')
    plt.savefig(path)
    plt.close()
    print(f"  Сохранён scatter plot: {path}")

def plot_ecdf(neuro_df, metric='dwell_time'):
    if neuro_df.empty:
        return
    plt.figure(figsize=(8,6))
    for var, color in [('A', 'gray'), ('B', 'green')]:
        subset = neuro_df[neuro_df['variant']==var][metric]
        sns.ecdfplot(subset, label=f'Вариант {var}', color=color)
    plt.title(f'ECDF: {metric}')
    plt.xlabel(metric)
    plt.ylabel('Доля наблюдений')
    plt.legend()
    path = os.path.join(OUTPUT_DIR, f'ecdf_{metric}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Сохранён ECDF: {path}")

# ------------------------------------------------------------
# Тепловые карты (только общие)
# ------------------------------------------------------------
def generate_heatmap(fixations_df, variant_label, aoi_bboxes):
    """Общая тепловая карта с контурами AOI (линейная шкала)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x = fixations_df['gaze_x'].values
    y = fixations_df['gaze_y'].values
    plt.figure(figsize=(10,6))
    heatmap, _, _ = np.histogram2d(x, y, bins=50,
                                   range=[[0, SCREEN_WIDTH], [0, SCREEN_HEIGHT]])
    plt.imshow(heatmap.T, origin='upper', cmap='inferno', aspect='auto',
               extent=[0, SCREEN_WIDTH, SCREEN_HEIGHT, 0],
               vmin=0, vmax=heatmap.max())
    for name, bbox in aoi_bboxes.items():
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                             linewidth=2, edgecolor='cyan', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(bbox[0]+2, bbox[1]+4, name, color='cyan', fontsize=8, va='top')
    plt.title(f'Тепловая карта внимания — вариант {variant_label}')
    plt.colorbar(label='Число фиксаций (шт.)')
    path = os.path.join(OUTPUT_DIR, f'heatmap_{variant_label}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Сохранена общая тепловая карта: {path}")