# src/data_prep/build_ml_dataset.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import (
    INITIAL_DATASET_PATH,
    ML_DATASET_PATH,
    DATA_PROCESSED_DIR,
)


# ---------- Загрузка исходного датасета ----------


def load_initial_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Загружает исходный датасет initial_dataset.csv.
    Ожидается стандартный CSV с разделителем ','.
    """
    csv_path = path or INITIAL_DATASET_PATH
    df = pd.read_csv(csv_path)
    return df


# ---------- Чистка ----------


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Минимальная чистка:
    - удаление дублей
    - удаление строк без conversion
    - приведение conversion/treatment к int (0/1)
    """
    df = df.drop_duplicates().copy()

    # conversion — основной таргет, должен быть 0/1 и не NaN
    df = df[df["conversion"].notna()].copy()
    df["conversion"] = df["conversion"].astype(int)

    # treatment — бинарный флаг воздействия
    if "treatment" in df.columns:
        df = df[df["treatment"].isin([0, 1])].copy()
        df["treatment"] = df["treatment"].astype(int)

    return df


def remove_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляем строки с некорректными значениями:
    - cost < 0
    - offer_AOV <= 0
    - monetary_90d < 0
    - avg_order_value_lifetime <= 0
    - revenue_14d < 0
    """
    mask = (
        (df["cost"] >= 0)
        & (df["offer_AOV"] > 0)
        & (df["monetary_90d"] >= 0)
        & (df["avg_order_value_lifetime"] > 0)
        & (df["revenue_14d"] >= 0)
    )
    return df[mask].copy()


def add_time_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет базовые временные признаки на основе treatment_date:
    - treatment_dow — день недели (0–6)
    - treatment_month — месяц (1–12)
    - treatment_hour — час (0–23)
    - time_morning / afternoon / evening / night — one-hot время суток
    """
    df = df.copy()

    dt = pd.to_datetime(df["treatment_date"], errors="coerce")
    df["treatment_dow"] = dt.dt.dayofweek
    df["treatment_month"] = dt.dt.month
    df["treatment_hour"] = dt.dt.hour

    df["time_morning"] = ((df["treatment_hour"] >= 6) & (df["treatment_hour"] < 12)).astype(int)
    df["time_afternoon"] = ((df["treatment_hour"] >= 12) & (df["treatment_hour"] < 18)).astype(int)
    df["time_evening"] = ((df["treatment_hour"] >= 18) & (df["treatment_hour"] < 24)).astype(int)
    df["time_night"] = ((df["treatment_hour"] >= 0) & (df["treatment_hour"] < 6)).astype(int)

    return df


def remove_target_leaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет потенциальные утечки таргета.
    Сейчас это:
    - revenue_14d (зависит от события после оффера)
    """
    df = df.copy()
    leak_cols = [c for c in ["revenue_14d"] if c in df.columns]
    if leak_cols:
        df = df.drop(columns=leak_cols)
    return df


def drop_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет служебные поля, которые не нужны для моделирования.
    Сейчас это:
    - other
    """
    df = df.copy()
    df = df.drop(columns=["other"], errors="ignore")
    return df


# ---------- Основной конвейер ----------


def build_ml_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит ml_training_dataset из исходного initial_dataset.
    Здесь только общая чистка + простые контекстные признаки.
    Более тонкий feature engineering будет в feature_engineering.py.
    """
    df = raw_df.copy()

    df = basic_clean(df)
    df = remove_invalid_values(df)
    df = add_time_context(df)
    df = remove_target_leaks(df)
    df = drop_service_columns(df)

    return df


def save_ml_dataset(df: pd.DataFrame, path: Path | None = None) -> None:
    """
    Сохраняет итоговый ml_training_dataset.csv.
    """
    out_path = path or ML_DATASET_PATH
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


# ---------- CLI ----------


def main():
    raw = load_initial_dataset()
    ml = build_ml_dataset(raw)
    save_ml_dataset(ml)
    print(f"Saved ML dataset to: {ML_DATASET_PATH}  shape={ml.shape}")


if __name__ == "__main__":
    main()