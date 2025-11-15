import os
from typing import List, Tuple

import pandas as pd
from catboost import CatBoostClassifier


# Используем те же фичи, что и в uplift-моделях
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
MODEL_PATH = os.path.join("models", "ctr_model.cbm")


def _prepare_ctr_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Для CTR-модели берём только записи, где оффер был реально показан (treatment=1)
    """
    df_treat = df[df["treatment"] == 1].copy()
    X = df_treat[FEATURE_COLS]
    y = df_treat[TARGET_COL]
    return X, y


def train_ctr_model(df: pd.DataFrame) -> CatBoostClassifier:
    """
    Тренировка CTR-модели и сохранение её в models/ctr_model.cbm
    """
    X, y = _prepare_ctr_data(df)

    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
    )

    model.fit(X, y)

    # Создаём папку models/, если её нет
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)

    return model


def load_ctr_model() -> CatBoostClassifier:
    """
    Загружает модель CTR из файла models/ctr_model.cbm
    """
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


def predict_click_proba(model: CatBoostClassifier, X: pd.DataFrame) -> pd.Series:
    """
    Возвращает вероятности клика по CTR-модели
    """
    proba = model.predict_proba(X[FEATURE_COLS])[:, 1]
    return pd.Series(proba, index=X.index, name="p_ctr")