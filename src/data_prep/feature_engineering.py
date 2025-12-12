# src/data_prep/feature_engineering.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd

from src.utils.config import ML_DATASET_PATH

TARGET_COL = "conversion"
ID_COLS = ["client_id", "offer_id"]

# Базовые числовые признаки
BASE_NUMERIC_FEATURES: List[str] = [
    "cost",
    "offer_AOV",
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "avg_order_value_lifetime",
    "total_orders_lifetime",
    "days_since_last_promo",
    "discounts_used_90d",
    "avg_discount_percent_90d",
    "email_open_rate_30d",
    "is_mobile_user",
    "city_tier",
    "push_enabled",
    "age",
    "treatment_dow",
    "treatment_month",
    "treatment_hour",
    "time_morning",
    "time_afternoon",
    "time_evening",
    "time_night",
]

# Категориальные признаки (строки, а не числа!)
CATEGORICAL_FEATURES: List[str] = [
    "offer_type",
    "offer_category",
    "channel",
    "favorite_category",
    "visited_category_14d",
    "category_affinity_top1",
    "gender",
    "price_segment",
]

# Производные числовые признаки
DERIVED_NUMERIC_FEATURES: List[str] = [
    "discount_share",            # размер скидки относительно AOV оффера
    "lf_check_to_offer_ratio",   # отношение lifetime среднего чека к AOV оффера
]

# Производные бинарные признаки (мы тоже считаем их числовыми)
BINARY_DERIVED_FEATURES: List[str] = [
    "is_favorite_category",
    "is_top_affinity_category",
    "is_recent_promo",
]

ALL_NUMERIC_FEATURES: List[str] = (
    BASE_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES + BINARY_DERIVED_FEATURES
)

FEATURE_COLS: List[str] = ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------- Загрузка ml_training_dataset ----------


def load_ml_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Загружает подготовленный ml_training_dataset.csv.
    """
    csv_path = path or ML_DATASET_PATH
    return pd.read_csv(csv_path)


# ---------- Feature engineering ----------


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет производные признаки:
    - discount_share
    - lf_check_to_offer_ratio
    - is_favorite_category
    - is_top_affinity_category
    - is_recent_promo
    """
    df = df.copy()

    # избегаем деления на ноль для AOV
    df["discount_share"] = df["cost"] / df["offer_AOV"].replace(0, pd.NA)
    df["discount_share"] = df["discount_share"].fillna(0.0)

    df["lf_check_to_offer_ratio"] = (
        df["avg_order_value_lifetime"] / df["offer_AOV"].replace(0, pd.NA)
    )
    df["lf_check_to_offer_ratio"] = df["lf_check_to_offer_ratio"].fillna(1.0)

    df["is_favorite_category"] = (
        df["offer_category"] == df["favorite_category"]
    ).astype("int8")

    df["is_top_affinity_category"] = (
        df["offer_category"] == df["category_affinity_top1"]
    ).astype("int8")

    df["is_recent_promo"] = (df["days_since_last_promo"] <= 30).astype("int8")

    return df


def ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит типы:
    - числовые фичи → float32
    - бинарные фичи → int8
    - категориальные фичи → string (object), чтобы CatBoost видел их как категориальные
    """
    df = df.copy()

    # Бинарные колонки (включая производные)
    binary_cols = [
        "treatment",
        "conversion",
        "is_mobile_user",
        "push_enabled",
        "time_morning",
        "time_afternoon",
        "time_evening",
        "time_night",
        "is_favorite_category",
        "is_top_affinity_category",
        "is_recent_promo",
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    # Числовые фичи (все, кроме явно бинарных)
    for col in ALL_NUMERIC_FEATURES:
        if col in df.columns and col not in binary_cols:
            df[col] = df[col].astype("float32")

    # Категориальные фичи — приводим к строкам
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


# ---------- Общий интерфейс для моделей ----------


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, List[str]]]:
    """
    Высокоуровневый интерфейс:
    - добавляет derived-фичи
    - приводит типы
    - формирует X, y, ids и метаинформацию.

    Возвращает:
      X: матрица признаков
      y: таргет (conversion)
      ids: DataFrame с client_id, offer_id
      meta: словарь со списками фич:
            {
              "numeric_features": [...],
              "categorical_features": [...],
              "feature_cols": [...]
            }
    """
    df_proc = add_derived_features(df)
    df_proc = ensure_dtypes(df_proc)

    missing = [col for col in FEATURE_COLS if col not in df_proc.columns]
    if missing:
        raise ValueError(f"Отсутствуют ожидаемые фичи: {missing}")

    X = df_proc[FEATURE_COLS].copy()
    y = df_proc[TARGET_COL].astype("int8")

    if all(c in df_proc.columns for c in ID_COLS):
        ids = df_proc[ID_COLS].copy()
    else:
        ids = pd.DataFrame()

    meta = {
        "numeric_features": ALL_NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_cols": FEATURE_COLS,
    }

    return X, y, ids, meta