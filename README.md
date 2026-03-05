# advanced_experiments

Библиотека для прикладного каузального анализа на табличных данных:
- `Synthetic Control` (с ограничениями на веса),
- подбор контрольной группы (`feature matching`, `propensity score matching`),
- `Conformal Inference` для Synthetic Control (p-value, CI, dashboard),
- инфраструктура для пакетных запусков экспериментов (`Experiment`),
- утилиты препроцессинга панельных данных + подготовка к RDD.

## Что внутри

### 1) Synthetic Control
- Обучение синтетического ряда для treated-unit.
- Получение весов доноров и fit error.
- Графики:
  - `real vs synthetic`,
  - распределение весов,
  - `effect/gap = real - synthetic`.

### 2) Matching
- `select_control_group_by_features(...)`: exact + distance matching.
- `propensity_score_match(...)`: низкоуровневый PSM.
- `select_control_group_by_propensity_from_panel(...)`: PSM прямо на panel-данных.

### 3) Conformal Inference
- Остатки под null-гипотезой.
- Permutation p-value.
- Pointwise p-value grid по сетке `H0`.
- Инверсия теста в confidence intervals.
- Готовые визуализации + единый conformal dashboard.

### 4) Experiment API
- `fit_all(...)` для фитинга по treated-unit.
- `evaluate(...)` для фильтрации/инференса/метрик.
- `run_experiment(...)` как end-to-end пайплайн.
- Поддержка методов инференса:
  - `independent_ttest`,
  - `paired_ttest`,
  - `permutation`.

### 5) Preprocessing API
- `prepare_panel_data(...)` — балансировка panel.
- `build_conformal_frame(...)` — pre/post блоки.
- `prepare_rdd_data(...)` — подготовка таблицы под sharp RDD.

---

## Установка

Требования: Python `>= 3.10.11`.

### Через `uv` (рекомендуется)
```bash
uv sync
```

### Через `pip`
```bash
pip install -e .
```

---

## Быстрый старт

```python
import pandas as pd
from cib_advanced_experiments import SyntheticControl

df = pd.read_csv("notebooks/data/smoking.csv")
df["state"] = df["state"].astype(str)
df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")

treated_state = df.loc[df["california"] == True, "state"].iloc[0]
treatment_date = pd.Timestamp("1989-01-01")

sc = SyntheticControl(
    data=df,
    unit_column="state",
    date_column="date",
    features=["cigsale"],
    treated_unit=treated_state,
    treatment_date=treatment_date,
    number_of_units_for_matching=10,
    matching_method="feature",
    numeric_matching=["retprice", "age15to24"],
    date_frequency="Y",
    aggregation_statistic="mean",
    epochs=600,
    lr=0.05,
)
sc.fit()

sc.plot_results(feature="cigsale", top_weights_to_plot=8, with_effect=True)
```

---

## Conformal Inference (пример)

```python
import numpy as np
import pandas as pd

null_grid = np.linspace(-20, 10, 121)

# Важно: period должен существовать в агрегированной частоте модели.
# Если date_frequency='Y', берите период из фактических дат sc:
post_dates = sc.get_data_on_features("cigsale").query(
    "date >= @sc.treatment_date"
)["date"].unique()
period = pd.Timestamp(post_dates[0])

p_grid = sc.get_conformal_p_value_grid_for_period(
    feature="cigsale",
    period=period,
    nulls=null_grid,
    q=1,
)

ci = sc.get_conformal_confidence_interval(
    feature="cigsale",
    nulls=null_grid,
    window=6,
    alpha=0.1,
    q=1,
)

sc.plot_conformal_dashboard(
    feature="cigsale",
    nulls=null_grid,
    window=6,
    period=period,
    alpha=0.1,
)
```

---

## Experiment API (новый workflow)

```python
import pandas as pd
from cib_advanced_experiments import Experiment

treatments = pd.DataFrame(
    {
        "state": ["3", "1"],  # пример: California + ещё один treated-unit
        "treat_date": [pd.Timestamp("1989-01-01"), pd.Timestamp("1989-01-01")],
    }
)

experiment = Experiment(
    data=df,
    treatments_dataframe=treatments,
    unit_column="state",
    date_column="date",
    features=["cigsale"],
    treatment_date_column="treat_date",
    date_frequency="Y",
    aggregation_statistic="mean",
)

# split workflow
fitted = experiment.fit_all(
    feature="cigsale",
    to_plot=False,
    control_group=sorted(df.loc[df["california"] == False, "state"].unique().tolist()),
    epochs=300,
    lr=0.05,
)
result = experiment.evaluate(
    fitted_units=fitted,
    feature="cigsale",
    inference_method="paired_ttest",
    to_plot=False,
)

# one-shot workflow
run_result = experiment.run_experiment(
    feature="cigsale",
    to_plot=False,
    to_plot_individual=False,
    control_group=sorted(df.loc[df["california"] == False, "state"].unique().tolist()),
    inference_method="permutation",
    permutation_iterations=500,
    epochs=300,
    lr=0.05,
)

metrics = experiment.get_metrics()
```

---

## Демонстрационный ноутбук

Полный walkthrough на `smoking.csv`:

- [`notebooks/smoking_library_demo_ru.ipynb`](notebooks/smoking_library_demo_ru.ipynb)

В нём есть:
- preprocessing,
- matching,
- synthetic control,
- conformal inference + dashboard,
- новый `Experiment`.

---

## Запуск тестов

```bash
PYTHONPATH=. uv run pytest -q
```

---

## Основные модули

- `cib_advanced_experiments/synthetic_control.py`
- `cib_advanced_experiments/experiment.py`
- `cib_advanced_experiments/conformal_inference.py`
- `cib_advanced_experiments/matching.py`
- `cib_advanced_experiments/propensity_score_matching.py`
- `cib_advanced_experiments/preprocessing.py`
- `cib_advanced_experiments/visualization.py`

