Aero NBO — Прототип 1 (ML + Uplift)

Overview

Данный прототип реализует подход Next Best Offer (NBO) на основе классических ML-моделей и методологии uplift modeling.
Цель — оценить, какое коммерческое предложение (оффер) создаст максимальный прирост вероятности отклика пользователя, по сравнению с ситуацией, когда оффер не показывается.

Система обучается на исторических данных и сравнивает три стратегии:
	1.	Rule-based baseline (простые бизнес-правила)
	2.	CTR-модель (классическая бинарная классификация)
	3.	Uplift-модель (T-learner: две раздельные модели treatment/control)

⸻

Project Structure

aero_nbo_uplift/
│
├── data/
│   ├── raw/                  # исходные данные (interactions_raw.csv)
│   ├── processed/            # подготовленные датасеты
│   └── external/             # дополнительные датасеты
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ctr_model_training.ipynb
│   ├── 04_uplift_models_training.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── data_prep/            # ETL, сборка nbo_dataset
│   ├── models/               # baseline, CTR, treatment/control
│   ├── evaluation/           # метрики, сравнение моделей
│   └── utils/                # утилиты, конфигурации
│
├── models/                   # сохранённые модели (.pkl)
│
├── reports/
│   └── final_report.docx
│
├── requirements.txt
└── README.md


⸻

Data

Основным входным датасетом является nbo_dataset.csv, где каждая строка — взаимодействие пользователь × оффер.

Используемые признаки:
	•	recency_days — дни с последней покупки
	•	frequency_30d, frequency_90d — частотные метрики
	•	monetary_90d — сумма затрат
	•	avg_purchase_value — средний чек
	•	category_encoded, channel_encoded, time_of_day — категориальные признаки
	•	treatment — оффер показан (1) или нет (0)
	•	outcome_click — целевая переменная

⸻

Models

1. Rule-based baseline

Простейший набор бизнес-правил для выбора оффера.
Используется для сравнения с ML-моделями.

2. CTR-модель

Классическая бинарная классификация (CatBoost/XGBoost).
Обучается на данных treatment=1 и прогнозирует:

p(click | user, offer)

3. Uplift-модель (T-learner)

Состоит из двух ML-моделей:
	•	model_treat: обучается на treatment-группе
	•	model_control: обучается на control-группе

Uplift рассчитывается как:

uplift = p_treat - p_control

Эта величина оценивает прирост отклика при показе оффера.

⸻

Pipeline
	1.	Подготовка данных (RFM, кодирование категорий, объединение данных).
	2.	Обучение rule-based baseline и фиксация его CTR.
	3.	Обучение CTR-модели на treatment-записях.
	4.	Обучение моделей treatment и control.
	5.	Расчёт uplift для всех пар user × offer.
	6.	Ранжирование офферов и выбор NBO.
	7.	Сравнение результатов всех подходов.

⸻

Installation

pip install -r requirements.txt


⸻

Usage

Пример запуска обучения:

python src/data_prep/build_dataset.py
python src/models/ctr_model.py
python src/models/uplift_treatment.py
python src/models/uplift_control.py

Демонстрация работы доступна в notebooks/05_evaluation.ipynb.

⸻

License

Проект разработан в рамках исследовательской задачи NBO и не содержит производственных данных.

⸻

Если хочешь — сделаю такие же README для второго варианта или их объединённую версию.
