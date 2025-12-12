# src/data_prep/build_ml_dataset.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.config import (
    INITIAL_DATASET_PATH,
    ML_DATASET_PATH,
    DATA_PROCESSED_DIR,
)


# ---------- Загрузка исходного датасета ----------

def load_initial_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or INITIAL_DATASET_PATH
    return pd.read_csv(csv_path)


# ---------- Чистка ----------

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

    df = df[df["conversion"].notna()].copy()
    df["conversion"] = df["conversion"].astype(int)

    if "treatment" in df.columns:
        df = df[df["treatment"].isin([0, 1])].copy()
        df["treatment"] = df["treatment"].astype(int)

    return df


def remove_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["cost"] >= 0)
        & (df["offer_AOV"] > 0)
        & (df["monetary_90d"] >= 0)
        & (df["avg_order_value_lifetime"] > 0)
        & (df["revenue_14d"] >= 0)
    )
    return df[mask].copy()


def add_time_context(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    if "revenue_14d" in df.columns:
        df = df.drop(columns=["revenue_14d"])
    return df


def drop_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["other"], errors="ignore")
    return df


def add_random_control_conversions(
    df: pd.DataFrame,
    control_share_of_treat: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:

    df = df.copy()

    mask_treat = df["treatment"] == 1
    p_treat = df.loc[mask_treat, "conversion"].mean()

    if p_treat == 0 or pd.isna(p_treat):
        p_treat = 0.05

    p_control = float(p_treat * control_share_of_treat)
    p_control = max(0.0001, min(p_control, 0.5))

    rng = np.random.default_rng(seed)
    mask_control = df["treatment"] == 0
    n_control = int(mask_control.sum())

    if n_control > 0:
        df.loc[mask_control, "conversion"] = rng.binomial(1, p_control, size=n_control).astype("int8")

    return df


# ---------- Основной конвейер ----------

def build_ml_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df = basic_clean(df)
    df = remove_invalid_values(df)
    df = add_time_context(df)

    df = add_random_control_conversions(df)

    df = remove_target_leaks(df)
    df = drop_service_columns(df)

    return df


def save_ml_dataset(df: pd.DataFrame, path: Path | None = None) -> None:
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