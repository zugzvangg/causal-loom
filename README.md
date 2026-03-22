# advanced_experiments

`advanced_experiments` — библиотека для прикладного каузального анализа на табличных и панельных данных.

## Возможности

- `Synthetic Control`:
  - fitting синтетического ряда,
  - анализ весов,
  - эффект `real - synthetic`,
  - placebo-анализ,
  - conformal inference (p-value, CI, dashboard).
- `Difference-in-Differences`:
  - 2x2 DiD через class API и one-shot функцию,
  - инференс (SE, p-value, CI),
  - 2x2 график и parallel trends.
- `Synthetic Difference-in-Differences`:
  - SDID для block-treatment,
  - residualization по ковариатам,
  - clustered SE,
  - period-by-period эффекты,
  - staggered adoption (`fit_staggered_sdid`).
- Matching:
  - feature matching,
  - propensity score matching.
- `Experiment` API:
  - пакетный запуск synthetic control по множеству treated units,
  - агрегированные метрики и inference.

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

## Быстрый старт: Synthetic Control

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

## Conformal Inference (пример)

```python
import numpy as np

null_grid = np.linspace(-20, 10, 121)

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

## Difference-in-Differences (пример)

```python
from cib_advanced_experiments import DifferenceInDifferences

did_data = df.copy()
did_data["treated"] = (did_data["state"] == treated_state).astype(int)
did_data["post"] = (did_data["date"] >= treatment_date).astype(int)

did = DifferenceInDifferences(
    data=did_data,
    outcome_column="cigsale",
    treatment_column="treated",
    post_column="post",
)

did_result = did.fit(alpha=0.05)
breakdown = did.get_estimators_breakdown()
did.plot_2x2()
did.plot_parallel_trends(time_column="date", treatment_start=treatment_date)
```

## Synthetic Difference-in-Differences (пример)

```python
from cib_advanced_experiments import (
    SyntheticDifferenceInDifferences,
    fit_sdid,
    fit_staggered_sdid,
)

sdid = SyntheticDifferenceInDifferences(
    data=did_data,
    outcome_column="cigsale",
    unit_column="state",
    time_column="date",
    treatment_column="treated",
    post_column="post",
)

# Для реальных данных иногда полезно ослабить tolerance SLSQP.
sdid_result = sdid.fit(alpha=0.05, optimizer_tolerance=1e-5)

print("SDID ATT:", sdid_result.estimate)
print("Unit weights sum:", sdid_result.unit_weights.sum())
print("Time weights sum:", sdid_result.time_weights.sum())

period_effects = sdid.fit_effects_by_period(alpha=0.05, optimizer_tolerance=1e-5)

payload = fit_sdid(
    panel=did_data,
    outcome_col="cigsale",
    unit_column="state",
    time_column="date",
    treatment_column="treated",
    post_column="post",
    optimizer_tolerance=1e-5,
)

sdid.plot_2x2()
sdid.plot_parallel_trends(treatment_start=treatment_date, pre_period_only=True)

# Staggered adoption
staggered_res = fit_staggered_sdid(
    data=panel_with_staggered_treatment,
    outcome_col="y",
    unit_column="unit",
    time_column="date",
    treatment_column="treated",
    post_column="post",
)
print(staggered_res.aggregate_estimate)
```

## Experiment API (пример)

```python
from cib_advanced_experiments import Experiment

treatments = pd.DataFrame(
    {
        "state": ["3", "1"],
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
```

## Демонстрационные ноутбуки

- [`notebooks/smoking_library_demo.ipynb`](notebooks/smoking_library_demo.ipynb)
  - полный walkthrough по Synthetic Control, preprocessing, matching, conformal inference, placebo, experiment.
- [`notebooks/billboard_did_library_demo.ipynb`](notebooks/billboard_did_library_demo.ipynb)
  - подробный разбор классического Difference-in-Differences.
- [`notebooks/synthetic_did_library_demo.ipynb`](notebooks/synthetic_did_library_demo.ipynb)
  - полный walkthrough по Synthetic Difference-in-Differences:
    - базовый SDID,
    - residualization + clustered SE,
    - period-by-period effects,
    - low-level helpers (`fit_unit_weights`, `fit_time_weights`, `join_sdid_weights`),
    - wrapper (`fit_sdid`) и one-shot API,
    - staggered SDID (`fit_staggered_sdid`),
    - диагностические графики.

## Запуск тестов

```bash
PYTHONPATH=. uv run pytest -q
```

## Основные модули

- `cib_advanced_experiments/synthetic_control.py`
- `cib_advanced_experiments/difference_in_differences.py`
- `cib_advanced_experiments/synthetic_difference_in_differences.py`
- `cib_advanced_experiments/experiment.py`
- `cib_advanced_experiments/conformal_inference.py`
- `cib_advanced_experiments/matching.py`
- `cib_advanced_experiments/propensity_score_matching.py`
- `cib_advanced_experiments/preprocessing.py`
- `cib_advanced_experiments/visualization.py`
