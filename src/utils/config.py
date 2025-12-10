# src/utils/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

INITIAL_DATASET_PATH = DATA_RAW_DIR / "initial_dataset.csv"
ML_DATASET_PATH = DATA_PROCESSED_DIR / "ml_training_dataset.csv"

MODELS_DIR = PROJECT_ROOT / "models"

# Uplift-модели
UPLIFT_TREATMENT_MODEL_PATH = MODELS_DIR / "uplift_treatment.cbm"
UPLIFT_CONTROL_MODEL_PATH = MODELS_DIR / "uplift_control.cbm"
UPLIFT_TREATMENT_META_PATH = MODELS_DIR / "uplift_treatment_meta.json"
UPLIFT_CONTROL_META_PATH = MODELS_DIR / "uplift_control_meta.json"

MIN_CLIENT_OBS = 20
MIN_OFFER_OBS = 20

DEFAULT_TOP_N = 3
EXPECTED_GAIN_THRESHOLD = 0.0