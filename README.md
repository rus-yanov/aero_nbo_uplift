<body>

  <h1>Aero NBO — Прототип 1 (Uplift)</h1>

  <p>
    Данный прототип реализует Next Best Offer (NBO) подход с использованием
    <b>uplift-моделирования (T-learner)</b> и сравнивает его с базовыми
    <b>rule-based</b> стратегиями подбора офферов.
    Основной фокус — оптимизация <b>инкрементального эффекта</b> и
    <b>ожидаемой бизнес-прибыли</b>, а не только вероятности клика.
  </p>

  <h2>1. Цели прототипа</h2>
  <ul>
    <li>Подготовить единый candidate set клиент–оффер для NBO-сценария.</li>
    <li>Реализовать rule-based baseline стратегии.</li>
    <li>Обучить CTR-модель как reference-подход.</li>
    <li>Построить uplift-модель (T-learner: treatment + control).</li>
    <li>Рассчитать uplift и profit-aware uplift.</li>
    <li>Сравнить стратегии по CTR, ожидаемой прибыли и causal-метрикам.</li>
  </ul>

  <h2>2. Стек технологий</h2>

  <h3>Язык / среда</h3>
  <ul>
    <li>Python 3.10+</li>
    <li>Jupyter Notebook</li>
  </ul>

  <h3>Библиотеки</h3>
  <ul>
    <li>pandas, numpy</li>
    <li>scikit-learn</li>
    <li>catboost</li>
    <li>matplotlib, seaborn</li>
  </ul>

  <h2>3. Структура проекта</h2>

  <pre><code>aero_nbo_uplift/
├── data/
│   ├── raw/                  # исходные данные
│   ├── processed/            # ml_training_dataset.csv
│   └── external/
│
├── notebooks/
│   ├── 01_eda_initial.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_rule_based_baseline.ipynb
│   ├── 04_uplift_training.ipynb
│   ├── 05_uplift_vs_rule_based.ipynb
│   └── 06_nbo_demo.ipynb
│
├── src/
│   ├── data_prep/            # очистка и feature engineering
│   ├── models/               # uplift / rule-based / scoring
│   ├── evaluation/           # метрики и сравнение стратегий
│   └── utils/
│
├── models/                   # сохранённые модели и meta
├── reports/
└── README.md
  </code></pre>

  <h2>4. Описание подхода</h2>
  <ol>
    <li>
      Формируется <b>candidate set</b>:
      <ul>
        <li>15 307 клиент–оффер пар</li>
        <li>4 338 уникальных клиентов</li>
        <li>в среднем 3–4 оффера на клиента</li>
      </ul>
      Это приближено к реальному NBO-сценарию, где офферы уже предварительно отфильтрованы бизнес-правилами.
    </li>
    <li>
      Строятся rule-based стратегии:
      <ul>
        <li>rule_based — фиксированный скор без учёта стоимости,</li>
        <li>rule_based_profit — эвристика с учётом AOV и cost.</li>
      </ul>
    </li>
    <li>
      Обучается uplift-модель (T-learner):
      <ul>
        <li><b>treatment-модель</b>: P(click | offer shown), AUC = <b>0.736</b></li>
        <li><b>control-модель</b>: P(click | offer not shown), AUC = <b>0.552</b></li>
      </ul>
    </li>
    <li>
      Uplift рассчитывается как:
      <pre><code>uplift = P(click | treatment) − P(click | control)</code></pre>
    </li>
    <li>
      Profit-aware uplift:
      <pre><code>expected_gain = uplift × offer_AOV − cost</code></pre>
    </li>
    <li>
      Для каждого клиента выбираются top-k офферов по различным стратегиям
      и проводится offline-сравнение.
    </li>
  </ol>

  <h2>5. Метрики оценки</h2>

  <ul>
    <li><b>CTR@k</b> — observed click rate (sanity-check, не causal).</li>
    <li><b>Mean expected gain per client @k</b> — ключевая бизнес-метрика.</li>
    <li><b>Reject rate @k</b> — доля клиентов без показанных офферов.</li>
    <li><b>Qini curve и AUUC</b> — наличие реального uplift-сигнала.</li>
  </ul>

  <p>
    Для uplift-модели получено:
    <ul>
      <li><b>AUUC ≈ 1970</b></li>
      <li>монотонно растущая Qini-кривая без провалов</li>
    </ul>
  </p>

  <h2>6. Ключевые результаты</h2>

  <ul>
    <li>
      <b>uplift_profit</b> стабильно даёт:
      <ul>
        <li>CTR@k ≈ <b>0.60</b></li>
        <li>Mean expected gain per client @k ≈ <b>+2.5 – 2.8</b></li>
      </ul>
    </li>
    <li>
      Чистый uplift (без cost):
      <ul>
        <li>может иметь хороший uplift-сигнал,</li>
        <li>но даёт <b>отрицательный ожидаемый доход</b>.</li>
      </ul>
    </li>
    <li>
      Rule-based стратегии:
      <ul>
        <li>могут показывать приемлемый CTR,</li>
        <li>но проигрывают по прибыли и инкрементальности.</li>
      </ul>
    </li>
  </ul>

  <h2>7. Выводы</h2>

  <ul>
    <li>Uplift ≠ CTR: высокий CTR не гарантирует бизнес-эффект.</li>
    <li>Uplift-модель оптимизирует именно <b>инкрементальные клики</b>.</li>
    <li>Profit-aware uplift — обязательное условие для production.</li>
    <li>Qini / AUUC подтверждают наличие causal-сигнала.</li>
    <li>Прототип готов к online A/B-тестированию.</li>
  </ul>

</body>
