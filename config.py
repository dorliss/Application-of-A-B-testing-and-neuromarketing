import os

DATA_DIR = "data"
OUTPUT_DIR = "output"

UEYES_LOG_DIR = "data/ueyes_raw/UEyes_dataset/eyetracker_logs"
AB_REAL_DATASET = "data/ab_real/ab_campaigns.csv"

SYNTHETIC_FIXATIONS = os.path.join(DATA_DIR, "synthetic_fixations.csv")
SYNTHETIC_CLICKS = os.path.join(DATA_DIR, "synthetic_clicks.csv")
SYNTHETIC_USERS = os.path.join(DATA_DIR, "synthetic_users.csv")
AOI_DEFINITIONS = os.path.join(DATA_DIR, "aoi_definitions.json")

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 250
N_USERS = 500

AOI_NAMES = ["headline", "image", "cta_button", "background"]

# Макет A (классический) — кнопка внизу
AOI_BBOXES_A = {
    "headline":    [10, 10, 200, 30],
    "image":       [10, 45, 200, 160],
    "cta_button":  [60, 210, 180, 30],
    "background":  [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT]
}

# Макет B (экспериментальный) — кнопка справа
AOI_BBOXES_B = {
    "headline":    [10, 10, 200, 30],
    "image":       [10, 45, 200, 160],
    "cta_button":  [220, 45, 70, 160],
    "background":  [0, 0, SCREEN_WIDTH, SCREEN_HEIGHT]
}

# Эффекты для варианта B (умеренное снижение)
EFFECT_CONVERSION_LIFT = -0.03        # конверсия B на 3 п.п. ниже
EFFECT_DWELL_CTA_LIFT = 1.0           # длительность фиксаций не меняем
EFFECT_FIXATION_COUNT_LIFT = 1.0      # количество фиксаций не меняем

# Фиксированная базовая конверсия (используется напрямую)
BASE_CONVERSION = 0.30