# src/models/uplift_scoring.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data_prep.feature_engineering import prepare_features
from src.models.uplift_treatment import load_treatment_model, predict_treatment_proba
from src.models.uplift_control import load_control_model, predict_control_proba


@dataclass(frozen=True)
class UpliftModels:
    """
    Единый контейнер для uplift-моделей.

    treatment — CatBoostClassifier
    control   — CatBoostClassifier ИЛИ sklearn-модель (например, DummyClassifier)
    meta_t/meta_c — мета с feature_cols/categorical_features и т.п.
    """
    treatment: Any
    control: Any
    meta_t: Dict
    meta_c: Dict


def load_uplift_models() -> UpliftModels:
    """
    Загружает обе модели через их "официальные" модули.
    Это исключает рассинхрон путей к model/meta и ошибки FileNotFound по meta-json.
    """
    model_t, meta_t = load_treatment_model()
    model_c, meta_c = load_control_model()
    return UpliftModels(treatment=model_t, control=model_c, meta_t=meta_t, meta_c=meta_c)


def add_uplift_scores(
    df: pd.DataFrame,
    models: Optional[UpliftModels] = None,
    cost_col: str = "cost",
    aov_col: str = "offer_AOV",
) -> pd.DataFrame:
    """
    Добавляет в датафрейм:
      - p_treat
      - p_control
      - uplift = p_treat - p_control
      - expected_gain_uplift = uplift * offer_AOV - cost
    """
    if cost_col not in df.columns:
        raise ValueError(f"В df нет колонки cost_col='{cost_col}'")
    if aov_col not in df.columns:
        raise ValueError(f"В df нет колонки aov_col='{aov_col}'")

    df_out = df.copy()
    models = models or load_uplift_models()

    # Готовим признаки (важно: порядок и типы должны совпадать с meta/prepare_features)
    X, _, _, _ = prepare_features(df_out)

    p_treat = predict_treatment_proba(models.treatment, X, models.meta_t)
    p_control = predict_control_proba(models.control, X, models.meta_c)

    df_out["p_treat"] = p_treat.astype(float)
    df_out["p_control"] = p_control.astype(float)
    df_out["uplift"] = df_out["p_treat"] - df_out["p_control"]

    df_out["expected_gain_uplift"] = (
        df_out["uplift"] * df_out[aov_col].astype(float) - df_out[cost_col].astype(float)
    )

    return df_out


def recommend_top_offers(
    df_scored: pd.DataFrame,
    client_id: int,
    top_n: int = 3,
    min_expected_gain: float = 0.0,
    sort_by: str = "expected_gain_uplift",
) -> pd.DataFrame:
    """
    Возвращает top-N офферов для одного client_id по выбранному скору.
    """
    if "client_id" not in df_scored.columns:
        raise ValueError("В df_scored нет client_id")

    d = df_scored[df_scored["client_id"] == client_id].copy()
    if d.empty:
        return d

    if sort_by not in d.columns:
        raise ValueError(f"Нет колонки для сортировки: {sort_by}")

    if min_expected_gain is not None and sort_by in d.columns:
        d = d[d[sort_by] >= float(min_expected_gain)].copy()

    return d.sort_values(sort_by, ascending=False).head(int(top_n))


def recommend_for_all_clients(
    df_scored: pd.DataFrame,
    top_n: int = 3,
    min_expected_gain: float = 0.0,
    sort_by: str = "expected_gain_uplift",
) -> pd.DataFrame:
    """
    Возвращает рекомендации top-N для каждого client_id.
    """
    if "client_id" not in df_scored.columns:
        raise ValueError("В df_scored нет client_id")

    if sort_by not in df_scored.columns:
        raise ValueError(f"Нет колонки для сортировки: {sort_by}")

    parts = []
    for cid, grp in df_scored.groupby("client_id", sort=False):
        g = grp.copy()
        if min_expected_gain is not None:
            g = g[g[sort_by] >= float(min_expected_gain)].copy()
        if g.empty:
            continue
        parts.append(g.sort_values(sort_by, ascending=False).head(int(top_n)))

    return pd.concat(parts, axis=0, ignore_index=True) if parts else df_scored.iloc[0:0].copy()