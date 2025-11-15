from typing import Optional, Tuple

import pandas as pd

from src.models.uplift_treatment import (
    load_treatment_model,
    predict_treatment_proba,
    FEATURE_COLS as FEATURE_COLS_TREAT,
)
from src.models.uplift_control import (
    load_control_model,
    predict_control_proba,
    FEATURE_COLS as FEATURE_COLS_CTRL,
)


def add_uplift_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    На вход: датафрейм в формате nbo_dataset (любой treatment/control).
    На выход: df с колонками p_treat, p_control, uplift.
    """
    df = df.copy()

    model_treat = load_treatment_model()
    model_ctrl = load_control_model()

    # предполагаем, что FEATURE_COLS_TREAT == FEATURE_COLS_CTRL
    X = df[FEATURE_COLS_TREAT]

    p_treat = predict_treatment_proba(model_treat, X)
    p_ctrl = predict_control_proba(model_ctrl, X)

    df["p_treat"] = p_treat
    df["p_control"] = p_ctrl
    df["uplift"] = df["p_treat"] - df["p_control"]

    return df


def recommend_best_offer_for_user_uplift(df: pd.DataFrame, user_id) -> Optional[Tuple[str, float]]:
    """
    df — датафрейм-кандидаты (user × offer); user_id — конкретный пользователь.
    Возвращает (offer_id, uplift).
    """
    user_df = df[df["user_id"] == user_id]
    if user_df.empty:
        return None

    user_scored = add_uplift_scores(user_df)
    best_row = user_scored.sort_values("uplift", ascending=False).iloc[0]
    return str(best_row["offer_id"]), float(best_row["uplift"])