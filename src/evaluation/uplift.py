import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _check_inputs(df: pd.DataFrame):
    required_cols = ["treatment", "outcome_click", "uplift"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from dataframe")


def compute_qini(df: pd.DataFrame, n_bins: int = 10) -> float:
    """
    Вычисление Qini coefficient (Klimov & Zadrozny).
    df должен содержать: treatment, outcome_click, uplift.
    """

    _check_inputs(df)

    # сортируем по uplift по убыванию
    df_sorted = df.sort_values("uplift", ascending=False).reset_index(drop=True)

    # бины по квантилям
    df_sorted["bucket"] = pd.qcut(df_sorted["uplift"], q=n_bins, labels=False, duplicates="drop")

    qini_values = []
    cum_treat = 0
    cum_control = 0

    for b in range(n_bins):
        bucket_df = df_sorted[df_sorted["bucket"] == b]

        treat_outcomes = bucket_df[bucket_df["treatment"] == 1]["outcome_click"].sum()
        control_outcomes = bucket_df[bucket_df["treatment"] == 0]["outcome_click"].sum()

        cum_treat += treat_outcomes
        cum_control += control_outcomes

        qini_values.append(cum_treat - cum_control)

    # разница между лучшим сценарием и baseline
    qini = qini_values[-1]
    return qini


def compute_auuc(df: pd.DataFrame) -> float:
    """
    Area Under Uplift Curve — интеграл uplift-curve.
    """

    _check_inputs(df)

    df_sorted = df.sort_values("uplift", ascending=False).reset_index(drop=True)

    # накапливаем uplift по мере добавления пользователей
    treat = df_sorted["treatment"]
    outcome = df_sorted["outcome_click"]

    incremental_gain = (treat * outcome) - ((1 - treat) * outcome)
    cumulative_gain = incremental_gain.cumsum()

    # площадь под кривой
    auuc = cumulative_gain.sum() / len(df_sorted)
    return float(auuc)


def plot_uplift_curve(df: pd.DataFrame):
    """
    Строит uplift curve: cumulative gain по top-k пользователей.
    """

    _check_inputs(df)

    df_sorted = df.sort_values("uplift", ascending=False).reset_index(drop=True)

    treat = df_sorted["treatment"]
    outcome = df_sorted["outcome_click"]

    incremental_gain = (treat * outcome) - ((1 - treat) * outcome)
    cumulative_gain = incremental_gain.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_gain, label="Uplift curve", lw=2)

    plt.xlabel("Number of users (sorted by uplift)")
    plt.ylabel("Cumulative gain")
    plt.title("Uplift Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()