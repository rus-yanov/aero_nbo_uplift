# src/evaluation/uplift_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UpliftCurveResult:
    curve: pd.DataFrame  # columns: frac, n, conv_treat, conv_control, uplift
    auuc: float


def uplift_curve(
    df: pd.DataFrame,
    uplift_col: str = "uplift",
    treatment_col: str = "treatment",
    target_col: str = "conversion",
    n_bins: int = 20,
) -> UpliftCurveResult:
    """
    Простейшая uplift-curve:
    - сортируем по uplift убыв.
    - режем на бины по доле населения.
    - считаем uplift как (mean(conv|treat=1) - mean(conv|treat=0)) * frac
      (это не “true causal”, но для прототипа как диагностическая кривая подходит)
    """
    d = df[[uplift_col, treatment_col, target_col]].dropna().copy()
    d = d.sort_values(uplift_col, ascending=False).reset_index(drop=True)

    n = len(d)
    if n == 0:
        empty = pd.DataFrame(columns=["frac", "n", "conv_treat", "conv_control", "uplift"])
        return UpliftCurveResult(curve=empty, auuc=float("nan"))

    bins = np.linspace(0, n, n_bins + 1).astype(int)

    rows = []
    for i in range(1, len(bins)):
        end = bins[i]
        chunk = d.iloc[:end]

        treat = chunk[chunk[treatment_col] == 1][target_col]
        ctrl = chunk[chunk[treatment_col] == 0][target_col]

        conv_t = float(treat.mean()) if len(treat) else float("nan")
        conv_c = float(ctrl.mean()) if len(ctrl) else float("nan")

        upl = (conv_t - conv_c) if (not np.isnan(conv_t) and not np.isnan(conv_c)) else float("nan")

        rows.append(
            {
                "frac": end / n,
                "n": end,
                "conv_treat": conv_t,
                "conv_control": conv_c,
                "uplift": upl,
            }
        )

    curve = pd.DataFrame(rows)

    # AUUC = площадь под uplift(fraction) по трапециям
    auuc = float(np.trapz(curve["uplift"].fillna(0.0).values, curve["frac"].values))
    return UpliftCurveResult(curve=curve, auuc=auuc)


def uplift_at_k(
    df: pd.DataFrame,
    uplift_col: str = "uplift",
    treatment_col: str = "treatment",
    target_col: str = "conversion",
    k_frac: float = 0.1,
) -> float:
    """
    Uplift@k%: разница конверсий treat-control среди top-k% по uplift.
    """
    d = df[[uplift_col, treatment_col, target_col]].dropna().copy()
    d = d.sort_values(uplift_col, ascending=False)

    n = len(d)
    if n == 0:
        return float("nan")

    k = max(1, int(round(n * float(k_frac))))
    chunk = d.iloc[:k]

    treat = chunk[chunk[treatment_col] == 1][target_col]
    ctrl = chunk[chunk[treatment_col] == 0][target_col]

    if len(treat) == 0 or len(ctrl) == 0:
        return float("nan")

    return float(treat.mean() - ctrl.mean())