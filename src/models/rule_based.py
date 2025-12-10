# src/models/rule_based.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# Диапазон для эвристического CTR (минимальный/максимальный)
MIN_CTR = 0.01
MAX_CTR = 0.25


def compute_rule_score(df: pd.DataFrame) -> pd.Series:
    """
    Векторно считает rule_score для каждой строки (client × offer)
    на основе простых бизнес-правил.

    Чем выше score, тем более «обещающим» считается оффер для клиента.
    """
    s = pd.Series(0.0, index=df.index, dtype="float32")

    # 1. Давность покупок (recency_days)
    rec = df["recency_days"]
    s += np.where(rec <= 7, 3.0,
         np.where(rec <= 30, 2.0,
         np.where(rec <= 90, 1.0, 0.0)))

    # 2. Частота и денежный объём (поведение за 90 дней)
    freq = df["frequency_90d"]
    s += np.where(freq >= 10, 2.0,
         np.where(freq >= 3, 1.0, 0.0))

    mon = df["monetary_90d"]
    s += np.where(mon >= 5000, 2.0,
         np.where(mon >= 1000, 1.0, 0.0))

    # 3. Скидочная активность (чувствительность к дисконту)
    disc_cnt = df["discounts_used_90d"]
    s += np.where(disc_cnt >= 5, 1.5,
         np.where(disc_cnt >= 1, 0.5, 0.0))

    # 4. Категориальные предпочтения
    #    – оффер в любимой категории / top1 affinity
    fav_match = (df["offer_category"] == df["favorite_category"])
    top_match = (df["offer_category"] == df["category_affinity_top1"])
    s += fav_match.astype("float32") * 2.0
    s += top_match.astype("float32") * 1.0

    # 5. Канал и мобильность
    #    app / push считаем более вовлекающими
    channel = df["channel"].astype(str)
    s += np.where(channel.isin(["app", "push"]), 1.0, 0.0)

    is_mobile = df["is_mobile_user"]
    s += np.where(is_mobile == 1, 0.5, 0.0)

    # 6. Email open rate (прокси вовлечённости в коммуникации)
    open_rate = df["email_open_rate_30d"]
    s += np.where(open_rate >= 0.5, 1.0,
         np.where(open_rate >= 0.2, 0.5, 0.0))

    # 7. Тип оффера — усиливаем скидочные офферы для budget-сегмента
    offer_type = df["offer_type"].astype(str)
    price_segment = df["price_segment"].astype(str)

    is_discount_offer = offer_type.str.startswith("discount")
    is_budget = (price_segment == "budget")

    s += (is_discount_offer & is_budget).astype("float32") * 1.0

    return s


def score_to_p_click(rule_score: pd.Series) -> pd.Series:
    """
    Переводит rule_score в эвристическую вероятность клика.

    Нормализуем score в [0,1] и растягиваем в диапазон [MIN_CTR, MAX_CTR].
    """
    if rule_score.empty:
        return rule_score.astype("float32")

    score_min = float(rule_score.min())
    score_max = float(rule_score.max())
    denom = max(score_max - score_min, 1e-6)

    normalized = (rule_score - score_min) / denom  # [0,1]
    p_click = MIN_CTR + normalized * (MAX_CTR - MIN_CTR)

    return p_click.astype("float32")


def add_rule_based_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в датафрейм:
      - rule_score
      - p_click_rule
      - expected_gain_rule = p_click_rule * offer_AOV - cost
    Возвращает новый df (копию), не модифицируя исходный.
    """
    df_out = df.copy()

    rule_score = compute_rule_score(df_out)
    p_click_rule = score_to_p_click(rule_score)

    df_out["rule_score"] = rule_score
    df_out["p_click_rule"] = p_click_rule
    df_out["expected_gain_rule"] = df_out["p_click_rule"] * df_out["offer_AOV"] - df_out["cost"]

    return df_out


def score_rule_based(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Высокоуровневая функция:
    принимает датафрейм с фичами (ml_training_dataset или его срез),
    добавляет rule-based скор и экономику.

    Используется и в оффлайн-оценке (03_rule_based.ipynb),
    и может быть использована в API как rule-based fallback.
    """
    return add_rule_based_columns(df)