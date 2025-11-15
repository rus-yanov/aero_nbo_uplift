from typing import Dict

import pandas as pd

from src.models.rule_based import add_rule_score, evaluate_rule_based_ctr_at_1
from src.models.ctr_model import load_ctr_model, predict_click_proba
from src.models.scoring import add_uplift_scores


def evaluate_rule_based(df: pd.DataFrame) -> float:
    df_scored = add_rule_score(df)
    return evaluate_rule_based_ctr_at_1(df_scored)


def evaluate_ctr_model(df: pd.DataFrame) -> float:
    """
    Оценка CTR-модели по схеме CTR@1:
    - считаем p_click для каждой строки (user×offer)
    - для каждого user_id берём offer с max p_click
    - считаем долю кликов по этим офферам
    """
    model = load_ctr_model()

    feature_cols = [
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

    df = df.copy()
    df["p_ctr"] = predict_click_proba(model, df[feature_cols])

    best_per_user = (
        df.sort_values("p_ctr", ascending=False)
          .groupby("user_id", as_index=False)
          .first()
    )

    ctr_at_1 = best_per_user["outcome_click"].mean()
    return float(ctr_at_1)


def evaluate_uplift_model(df: pd.DataFrame) -> float:
    """
    Грубая offline-оценка uplift-модели по CTR@1:
    - считаем p_treat, p_control, uplift
    - для каждого user_id выбираем offer с max uplift
    - смотрим фактический outcome_click по этим офферам
    """
    df_scored = add_uplift_scores(df)

    best_per_user = (
        df_scored.sort_values("uplift", ascending=False)
                 .groupby("user_id", as_index=False)
                 .first()
    )

    ctr_at_1 = best_per_user["outcome_click"].mean()
    return float(ctr_at_1)


def compare_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает табличку с CTR@1 для трёх подходов.
    """
    results: Dict[str, float] = {}

    results["rule_based_ctr@1"] = evaluate_rule_based(df)
    results["ctr_model_ctr@1"] = evaluate_ctr_model(df)
    results["uplift_model_ctr@1"] = evaluate_uplift_model(df)

    summary = (
        pd.Series(results, name="value")
        .reset_index()
        .rename(columns={"index": "model"})
    )

    return summary