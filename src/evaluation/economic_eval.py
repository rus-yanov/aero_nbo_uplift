# src/evaluation/economic_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyReport:
    name: str
    n_rows: int
    n_clients: int
    mean_expected_gain: float
    sum_expected_gain: float
    observed_conversion_rate: float


def _select_topn_per_client(
    df: pd.DataFrame,
    score_col: str,
    top_n: int,
    min_score: float = 0.0,
) -> pd.DataFrame:
    parts = []
    for cid, grp in df.groupby("client_id", sort=False):
        g = grp.copy()
        g = g[g[score_col] >= float(min_score)].copy()
        if g.empty:
            continue
        g = g.sort_values(score_col, ascending=False).head(int(top_n))
        parts.append(g)
    return pd.concat(parts, axis=0, ignore_index=True) if parts else df.iloc[0:0].copy()


def evaluate_strategy(
    df_scored: pd.DataFrame,
    name: str,
    score_col: str,
    top_n: int = 3,
    min_score: float = 0.0,
    target_col: str = "conversion",
) -> StrategyReport:
    """
    Экономическая оценка через expected_gain (модельный показатель),
    плюс proxy-метрика observed_conversion_rate на выбранных строках.
    """
    if "client_id" not in df_scored.columns:
        raise ValueError("Нет client_id в датасете")
    if score_col not in df_scored.columns:
        raise ValueError(f"Нет score_col={score_col}")
    if target_col not in df_scored.columns:
        raise ValueError(f"Нет target_col={target_col}")

    chosen = _select_topn_per_client(df_scored, score_col=score_col, top_n=top_n, min_score=min_score)

    if chosen.empty:
        return StrategyReport(
            name=name,
            n_rows=0,
            n_clients=0,
            mean_expected_gain=float("nan"),
            sum_expected_gain=0.0,
            observed_conversion_rate=float("nan"),
        )

    n_rows = int(len(chosen))
    n_clients = int(chosen["client_id"].nunique())
    mean_gain = float(chosen[score_col].mean())
    sum_gain = float(chosen[score_col].sum())
    obs_conv = float(chosen[target_col].mean())

    return StrategyReport(
        name=name,
        n_rows=n_rows,
        n_clients=n_clients,
        mean_expected_gain=mean_gain,
        sum_expected_gain=sum_gain,
        observed_conversion_rate=obs_conv,
    )


def compare_rule_vs_uplift(
    df_rule_scored: pd.DataFrame,
    df_uplift_scored: pd.DataFrame,
    top_n: int = 3,
    min_expected_gain: float = 0.0,
) -> pd.DataFrame:
    """
    Ожидается:
      df_rule_scored: содержит expected_gain_rule
      df_uplift_scored: содержит expected_gain_uplift
    """
    rep_rule = evaluate_strategy(
        df_rule_scored,
        name="rule_based",
        score_col="expected_gain_rule",
        top_n=top_n,
        min_score=min_expected_gain,
    )
    rep_uplift = evaluate_strategy(
        df_uplift_scored,
        name="uplift",
        score_col="expected_gain_uplift",
        top_n=top_n,
        min_score=min_expected_gain,
    )

    return pd.DataFrame(
        [
            rep_rule.__dict__,
            rep_uplift.__dict__,
        ]
    )