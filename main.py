import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline.loader import load_synthetic_data
from pipeline.metrics import compute_static_metrics, compute_neurometrics
from pipeline.analysis import (
    run_ttest, run_proportions_test, plot_conversion_bar,
    plot_neurometrics_boxplots, plot_neurometrics_histograms,
    plot_neurometrics_bars, plot_scatter_dwell_vs_fixcount,
    plot_ecdf, generate_heatmap
)
from config import OUTPUT_DIR

def main():
    print("=== A/B-тест с нейромаркетингом: полный анализ ===\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fix_df, click_df, aoi_dict = load_synthetic_data()
    bbox_A = aoi_dict['A'] if aoi_dict and 'A' in aoi_dict else None
    bbox_B = aoi_dict['B'] if aoi_dict and 'B' in aoi_dict else None

    print("\n[2/5] Статические A/B-метрики...")
    conv_rates, user_conv = compute_static_metrics(click_df)
    print(conv_rates)
    plot_conversion_bar(conv_rates)
    n_A = (user_conv['variant'] == 'A').sum()
    n_B = (user_conv['variant'] == 'B').sum()
    run_proportions_test(conv_rates, n_A, n_B)

    print("\n[3/5] Нейромаркетинговые метрики (CTA)...")
    neuro_df = compute_neurometrics(fix_df)
    if not neuro_df.empty:
        print("  Средние значения по вариантам:")
        print(neuro_df.groupby('variant')[['dwell_time', 'fixation_count', 'avg_fixation_duration']].mean())
        plot_neurometrics_boxplots(neuro_df)
        plot_neurometrics_histograms(neuro_df)
        plot_neurometrics_bars(neuro_df)
        plot_scatter_dwell_vs_fixcount(neuro_df)
        plot_ecdf(neuro_df, 'dwell_time')
        plot_ecdf(neuro_df, 'fixation_count')
        for metric in ['dwell_time', 'fixation_count', 'avg_fixation_duration']:
            run_ttest(neuro_df, metric)
    else:
        print("  Нет данных по CTA.")

    print("\n[4/5] Тепловые карты внимания...")
    for var, bbox in [('A', bbox_A), ('B', bbox_B)]:
        subset = fix_df[fix_df['variant'] == var]
        if not subset.empty and bbox is not None:
            generate_heatmap(subset, var, bbox)
        else:
            print(f"  Нет данных или разметки для варианта {var}")

    print(f"\n[5/5] Готово! Все результаты в папке {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()