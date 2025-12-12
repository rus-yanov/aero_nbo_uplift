# src/models/uplift_control.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from catboost import CatBoostClassifier, CatBoostError

from src.utils.config import UPLIFT_CONTROL_MODEL_PATH, UPLIFT_CONTROL_META_PATH


def save_control_model(
    model: Any,
    meta: Dict,
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> None:
    """
    Сохраняет control-модель и метаинформацию.

    Поддерживает:
      - CatBoostClassifier (через .save_model)
      - любые sklearn-модели (через joblib.dump)
    """
    m_path = model_path or UPLIFT_CONTROL_MODEL_PATH
    meta_p = meta_path or UPLIFT_CONTROL_META_PATH

    m_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, CatBoostClassifier):
        model.save_model(str(m_path))
    else:
        joblib.dump(model, m_path)

    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_control_model(
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> Tuple[Any, Dict]:
    """
    Загружает control-модель и метаинформацию.

    Пытается:
      1) загрузить как CatBoost (load_model),
      2) при ошибке — как joblib (sklearn/прочие модели).
    """
    m_path = model_path or UPLIFT_CONTROL_MODEL_PATH
    meta_p = meta_path or UPLIFT_CONTROL_META_PATH

    if not m_path.exists():
        raise FileNotFoundError(f"Control model file not found: {m_path}")
    if not meta_p.exists():
        raise FileNotFoundError(f"Control meta file not found: {meta_p}")

    try:
        cb = CatBoostClassifier()
        cb.load_model(str(m_path))
        model: Any = cb
    except CatBoostError:
        model = joblib.load(m_path)

    with open(meta_p, "r", encoding="utf-8") as f:
        meta: Dict = json.load(f)

    return model, meta


def predict_control_proba(model: Any, X, meta: Dict) -> np.ndarray:
    """
    Возвращает p_control = P(conversion | treatment=0).

    - Выравнивает порядок/набор колонок по meta["feature_cols"] (как в treatment).
    - Для моделей с predict_proba.
    - Если predict_proba возвращает только один столбец (DummyClassifier/один класс),
      считаем p_control = 0 для всех объектов.
    """
    feature_cols = meta.get("feature_cols")
    if feature_cols is not None:
        X = X[feature_cols]

    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model of type {type(model)} has no predict_proba")

    proba = model.predict_proba(X)

    if proba.ndim == 2 and proba.shape[1] == 1:
        return np.zeros(X.shape[0], dtype=float)

    return proba[:, 1].astype(float)