from typing import Optional, Tuple

import pandas as pd

def _rule_score(row: pd.Series) -> float:
    """
    Чем выше score, тем привлекательнее оффер для пользователя.
    Правила можно потом подкрутить под конкретный бизнес-кейс.
    """
    score = 0.0

    # недавняя активность
    if row["recency_days"] <= 7:
        score += 3.0
    elif row["recency_days"] <= 30:
        score += 1.5

    # частые покупки → можно предлагать что-то активнее
    if row["frequency_30d"] >= 3:
        score += 2.0
    elif row["frequency_30d"] >= 1:
        score += 1.0

    # высокая сумма трат за 90 дней
    if row["monetary_90d"] >= 20000:
        score += 2.5
    elif row["monetary_90d"] >= 5000:
        score += 1.0

    # средний чек
    if row["avg_purchase_value"] >= 5000:
        score += 1.0

    # время суток: пример — вечер считается более конверсионным
    # (коды времени суток нужно согласовать с данными)
    if row["time_of_day"] in (2, 3):  # условно: 2 = вечер, 3 = ночь
        score += 0.5

    # канал: пуш / приложение условно считаем более эффективными, чем e-mail
    # (коды каналов нужно согласовать с данными)
    if row["channel_encoded"] == 1:  # допустим, 1 = mobile push
        score += 1.0
    elif row["channel_encoded"] == 2:  # допустим, 2 = in-app
        score += 0.5

    return score


def add_rule_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет колонку rule_score к датафрейму.
    Ожидается структура nbo_dataset.csv.
    """
    df = df.copy()
    df["rule_score"] = df.apply(_rule_score, axis=1)
    return df


def recommend_best_offer_for_user(df: pd.DataFrame, user_id) -> Optional[Tuple[str, float]]:
    """
    Возвращает (offer_id, rule_score) для лучшего оффера по rule-based логике
    для заданного пользователя.
    """
    user_df = df[df["user_id"] == user_id]
    if user_df.empty:
        return None

    user_scored = add_rule_score(user_df)
    best_row = user_scored.sort_values("rule_score", ascending=False).iloc[0]
    return best_row["offer_id"], float(best_row["rule_score"])


def evaluate_rule_based_ctr_at_1(df: pd.DataFrame) -> float:
    """
    Простейшая offline-оценка:
    - для каждого пользователя выбираем оффер с максимальным rule_score
    - смотрим, был ли по нему клик в исторических данных (outcome_click == 1)
    Возвращает CTR@1 (долю пользователей, которые кликнули по рекомендованному офферу).
    """
    scored = add_rule_score(df)

    # выбираем по одному "лучшему" офферу на пользователя
    best_per_user = (
        scored.sort_values("rule_score", ascending=False)
        .groupby("user_id", as_index=False)
        .first()
    )

    # считаем долю кликов
    ctr_at_1 = best_per_user["outcome_click"].mean()
    return float(ctr_at_1)


if __name__ == "__main__":
    # пример использования (offline)
    path = "data/processed/nbo_dataset.csv"
    nbo_df = pd.read_csv(path)

    ctr = evaluate_rule_based_ctr_at_1(nbo_df)
    print(f"Rule-based CTR@1: {ctr:.4f}")

    example_user = nbo_df["user_id"].iloc[0]
    best_offer = recommend_best_offer_for_user(nbo_df, example_user)
    print(f"Best offer for user {example_user}: {best_offer}")