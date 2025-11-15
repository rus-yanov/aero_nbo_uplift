<body>

  <h1>Aero NBO — Прототип 1 (ML + Uplift)</h1>

  <p>
    Этот прототип реализует классический подход к Next Best Offer (NBO) на основе моделей отклика 
    и uplift-моделирования. Основная цель — сравнить эффективность rule-based правил с ML- и 
    uplift-подходами на открытом датасете.
  </p>

  <h2>1. Цели прототипа</h2>
  <ul>
    <li>Подготовить единый датасет для моделирования поведения пользователей.</li>
    <li>Построить baseline на основе простых rule-based правил.</li>
    <li>Обучить CTR-модель для оценки вероятности отклика на оффер.</li>
    <li>Обучить T-learner uplift-модель (treatment + control).</li>
    <li>Рассчитать uplift для пользователей и ранжировать офферы.</li>
    <li>Оценить прирост качества относительно baseline.</li>
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
    <li>catboost или xgboost</li>
    <li>matplotlib, seaborn</li>
    <li>(опционально) causalml / econml</li>
  </ul>

  <h2>3. Структура проекта</h2>

  <pre><code>aero_nbo_uplift/
├── data/
│   ├── raw/                  # исходные данные
│   ├── processed/            # подготовленный nbo_dataset.csv
│   └── external/
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ctr_model_training.ipynb
│   ├── 04_uplift_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_nbo_demo.ipynb
│
├── src/
│   ├── data_prep/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── models/                   # сохранённые модели
├── reports/
└── README.md
  </code></pre>

  <h2>4. Описание подхода</h2>
  <ol>
    <li>
      На основе исторических данных формируется датасет 
      <code>nbo_dataset.csv</code> с RFM-признаками, контекстом показа и таргетами.
    </li>
    <li>
      Строится rule-based baseline для первичного сравнения.
    </li>
    <li>
      Обучается CTR-модель на treatment-записях (оффер был показан).
    </li>
    <li>
      Обучаются две независимые модели:
      <ul>
        <li>treatment-модель: прогноз отклика при показе оффера,</li>
        <li>control-модель: отклик в сценарии «оффер не показан».</li>
      </ul>
    </li>
    <li>
      Uplift рассчитывается как разница прогнозов treatment и control.
    </li>
    <li>
      Для каждого пользователя офферы ранжируются по uplift, выбирается наилучший.
    </li>
  </ol>

  <h2>5. Как запустить</h2>

  <ol>
    <li>
      Установить зависимости:
      <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>
      Последовательно пройти через ноутбуки в каталоге <code>notebooks/</code>:
      <ul>
        <li>01–02: подготовка данных</li>
        <li>03: обучение CTR-модели</li>
        <li>04: обучение uplift-моделей</li>
        <li>05: сравнение baseline / CTR / uplift</li>
        <li>06: демонстрация выбора NBO</li>
      </ul>
    </li>
  </ol>

  <h2>6. Результат</h2>
  <p>Прототип предоставляет:</p>
  <ul>
    <li>обученные модели отклика и uplift;</li>
    <li>сравнительный анализ rule-based / CTR / uplift;</li>
    <li>механизм вычисления uplift и алгоритм выбора NBO;</li>
    <li>отчёт и визуализации, готовые для включения в аналитическую записку.</li>
  </ul>

</body>
