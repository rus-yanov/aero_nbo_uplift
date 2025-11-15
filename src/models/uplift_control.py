from typing import Tuple, List
import os

import pandas as pd
from catboost import CatBoostClassifier

FEATURE_COLS: List[str] = [
    "recency_days",
    "frequency_30d",
    "frequency_90d",
    "monetary_90d",
    "avg_purchase_value",
    "category_encoded",
    "channel_encoded",
    "time_morning",
    "time_afternoon",
    "time_evening",
    "time_night",
]

TARGET_COL = "outcome_click"
MODEL_PATH = os.path.join("models", "control_model.cbm")


def _prepare_control_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df_ctrl = df[df["treatment"] == 0].copy()
    X = df_ctrl[FEATURE_COLS]
    y = df_ctrl[TARGET_COL]
    return X, y


def train_control_model(df: pd.DataFrame) -> CatBoostClassifier:
    X, y = _prepare_control_data(df)

    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
    )

    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)

    return model


def load_control_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


def predict_control_proba(model: CatBoostClassifier, X: pd.DataFrame) -> pd.Series:
    proba = model.predict_proba(X[FEATURE_COLS])[:, 1]
    return pd.Series(proba, index=X.index, name="p_control")