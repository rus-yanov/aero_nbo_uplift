# src/models/uplift_treatment.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.data_prep.feature_engineering import prepare_features
from src.utils.config import (
    UPLIFT_TREATMENT_MODEL_PATH,
    UPLIFT_TREATMENT_META_PATH,
    MODELS_DIR,
)


DEFAULT_TREATMENT_PARAMS: Dict = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "iterations": 500,
    "random_seed": 42,
    "verbose": False,
}


def _ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_treatment_model(
    df_ml: pd.DataFrame,
    params: Dict | None = None,
) -> Tuple[CatBoostClassifier, Dict]:
    """
    Обучает модель p(click | treatment=1) на срезе treatment == 1.

    Возвращает:
      model — обученный CatBoostClassifier,
      meta  — словарь с описанием фичей (из prepare_features).
    """
    df_t = df_ml[df_ml["treatment"] == 1].copy()
    if df_t.empty:
        raise ValueError("Нет строк с treatment == 1 для обучения treatment-модели")

    X, y, ids, meta = prepare_features(df_t)

    cat_features = [
        col for col in meta["categorical_features"]
        if col in X.columns
    ]

    model_params = DEFAULT_TREATMENT_PARAMS.copy()
    if params:
        model_params.update(params)

    model = CatBoostClassifier(**model_params)
    model.fit(
        X,
        y,
        cat_features=cat_features,
    )

    return model, meta


def save_treatment_model(
    model: CatBoostClassifier,
    meta: Dict,
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> None:
    """
    Сохраняет модель и мета-информацию о фичах.
    """
    _ensure_models_dir()

    m_path = model_path or UPLIFT_TREATMENT_MODEL_PATH
    meta_p = meta_path or UPLIFT_TREATMENT_META_PATH

    model.save_model(str(m_path))

    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_treatment_model(
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> Tuple[CatBoostClassifier, Dict]:
    """
    Загружает модель и мета-информацию.
    """
    m_path = model_path or UPLIFT_TREATMENT_MODEL_PATH
    meta_p = meta_path or UPLIFT_TREATMENT_META_PATH

    model = CatBoostClassifier()
    model.load_model(str(m_path))

    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


def predict_treatment_proba(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    meta: Dict,
) -> np.ndarray:
    """
    Возвращает вектор p_treat = P(click | treatment=1).

    X — уже подготовленная матрица признаков
        (через prepare_features на том же наборе фичей).
    """
    # на всякий случай: порядок колонок должен совпадать с обучением
    feature_cols = meta["feature_cols"]
    X = X[feature_cols]

    proba = model.predict_proba(X)[:, 1]
    return proba