# CausalLoom

[![CI](https://github.com/zugzvangg/causal-loom/actions/workflows/ci.yml/badge.svg)](https://github.com/zugzvangg/causal-loom/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Лицензия: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Статус: experimental](https://img.shields.io/badge/status-experimental-orange.svg)](#статус-проекта)

[English version](README.md)

CausalLoom — компактный тестируемый Python-инструментарий для изучения и
прототипирования каузального анализа табличных и панельных данных. Проект
превращает несколько учебных notebook-реализаций в повторно используемые API с
явными объектами результатов, диагностикой, инференсом и визуализациями.

Сейчас реализованы Synthetic Control, классический Difference-in-Differences
(DiD) для дизайна 2×2, Synthetic Difference-in-Differences (SDID), conformal- и
placebo-инференс, подбор контрольной группы и пакетный запуск synthetic control.

> [!IMPORTANT]
> Это экспериментальный образовательный проект, а не готовая замена
> рецензируемой библиотеке каузального анализа. Успешная оптимизация модели сама
> по себе не доказывает причинную идентификацию. Ответственность за дизайн
> исследования, правдоподобие предпосылок, качество данных и интерпретацию
> результата остаётся на исследователе.

## Зачем нужен этот проект

Многие разборы каузальных методов остаются одноразовыми ноутбуками. Этот
репозиторий показывает инженерную сторону превращения таких разборов в небольшую
библиотеку:

- единообразные class API и one-shot функции;
- неизменяемые dataclass-объекты с оценками, неопределённостью и диагностикой;
- проверки распространённых ошибок в структуре панели и treatment-дизайне;
- неотрицательные веса synthetic control, сумма которых равна единице;
- подбор доноров по признакам или propensity score;
- placebo-траектории, conformal p-value и построение доверительных множеств;
- диагностика на Plotly и воспроизводимые выполненные ноутбуки;
- тесты API, ошибок валидации и восстановления известных эффектов на симуляциях.

## Возможности

| Направление | Основной API | Результат |
| --- | --- | --- |
| Synthetic Control | `SyntheticControl`, `SynthControlModel` | Веса доноров на симплексе, фактический и синтетический ряды, траектория эффекта, ошибка подгонки |
| Инференс для Synthetic Control | методы `SyntheticControl` | In-space placebo и randomization-style p-value; conformal residuals, сетки p-value, доверительные множества и dashboard |
| Difference-in-Differences | `DifferenceInDifferences`, `difference_in_differences` | 2×2 DiD-оценка, обычная OLS standard error, p-value, доверительный интервал, декомпозиция ячеек и трендов |
| Synthetic DiD | `SyntheticDifferenceInDifferences`, `synthetic_difference_in_differences`, `fit_sdid` | Веса юнитов и периодов, взвешенная оценка, residualization, fixed effects, кластерные standard errors, эффекты по периодам |
| Staggered SDID | `fit_staggered_sdid` | Отдельные модели по когортам и настраиваемая взвешенная агрегация |
| Matching | `select_control_group_by_features`, `propensity_score_match` | Идентификаторы подобранных контролей и метаданные matching |
| Пакетные эксперименты | `Experiment` | Набор synthetic-control моделей, фильтрация по качеству fit, агрегированный эффект, paired-test или permutation-инференс |
| Визуализация | `causal_loom.visualization` | Plotly-графики рядов, эффектов, весов, placebo, DiD-трендов, conformal-результатов и агрегированных экспериментов |

## Установка

Нужен Python 3.10.11 или новее. Сейчас проект устанавливается из исходного кода
и не заявлен как опубликованный пакет на PyPI.

Через [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/zugzvangg/causal-loom.git
cd causal-loom
uv sync --locked --all-groups
```

Через `pip` в виртуальном окружении:

```bash
git clone https://github.com/zugzvangg/causal-loom.git
cd causal-loom
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Быстрый старт

Пример сам генерирует небольшую панель, строит synthetic control и оценивает
средний post-treatment gap, поэтому не зависит от учебных датасетов.

```python
import numpy as np
import pandas as pd

from causal_loom import SyntheticControl

rng = np.random.default_rng(7)
dates = pd.date_range("2024-01-01", periods=24, freq="D")
controls = [f"control_{i}" for i in range(10)]
treatment_date = dates[16]

common_path = 10 + 0.12 * np.arange(len(dates)) + np.sin(np.arange(len(dates)) / 3)
rows = []

for i, unit in enumerate(controls):
    outcome = common_path + np.linspace(-0.8, 0.8, len(controls))[i]
    outcome = outcome + rng.normal(0, 0.05, len(dates))
    rows.extend(
        {"unit": unit, "date": date, "outcome": value}
        for date, value in zip(dates, outcome)
    )

untreated_path = common_path + rng.normal(0, 0.04, len(dates))
treated_path = untreated_path + np.where(dates >= treatment_date, 2.5, 0.0)
rows.extend(
    {"unit": "treated", "date": date, "outcome": value}
    for date, value in zip(dates, treated_path)
)

panel = pd.DataFrame(rows)

model = SyntheticControl(
    data=panel,
    unit_column="unit",
    date_column="date",
    features=["outcome"],
    treated_unit="treated",
    treatment_date=treatment_date,
    control_group=controls,
    max_iter=2_000,
    tolerance=1e-8,
)
model.fit()

effect = model.get_effect_data("outcome")
post_effect = effect.loc[effect["date"] >= treatment_date, "effect"].mean()

print(f"Estimated mean post-treatment effect: {post_effect:.2f}")
print(f"Donor weights sum: {model.get_weights().sum():.6f}")

# Открывает графики рядов, весов доноров и эффекта.
model.plot_results("outcome", top_weights_to_plot=5, with_effect=True)
```

`max_iter` — максимальное число итераций SLSQP, а `tolerance` — точность
сходимости оптимизатора. Старые имена `epochs` и `lr` временно сохранены как
deprecated aliases.

### Difference-in-Differences 2×2

```python
from causal_loom import DifferenceInDifferences

did = DifferenceInDifferences(
    data=did_frame,
    outcome_column="outcome",
    treatment_column="treated",
    post_column="post",
)
result = did.fit(alpha=0.05)

print(result.estimate, result.ci_lower, result.ci_upper, result.pvalue)
did.plot_parallel_trends(
    time_column="date",
    treatment_start="2024-01-17",
    pre_period_only=True,
)
```

В `did_frame` должны быть числовой `outcome`, бинарные `treated` и `post`, а
также наблюдения во всех четырёх ячейках «группа × период».

### Synthetic Difference-in-Differences

```python
from causal_loom import SyntheticDifferenceInDifferences

sdid = SyntheticDifferenceInDifferences(
    data=did_frame,
    outcome_column="outcome",
    unit_column="unit",
    time_column="date",
    treatment_column="treated",
    post_column="post",
)
result = sdid.fit(
    alpha=0.05,
    use_fixed_effects=True,
    cluster_by_unit=True,
    optimizer_tolerance=1e-9,
)

print(result.estimate, result.ci_lower, result.ci_upper)
print(result.unit_weights.head())
```

Полные примеры подготовки данных и диагностики находятся в ноутбуках.

## Демонстрационные ноутбуки

- [`notebooks/smoking_library_demo.ipynb`](notebooks/smoking_library_demo.ipynb)
  — Synthetic Control, matching, траектории эффекта, placebo, conformal inference
  и пакетный API `Experiment`.
- [`notebooks/billboard_did_library_demo.ipynb`](notebooks/billboard_did_library_demo.ipynb)
  — пошаговый классический Difference-in-Differences 2×2.
- [`notebooks/synthetic_did_library_demo.ipynb`](notebooks/synthetic_did_library_demo.ipynb)
  — веса SDID, residualization, кластерный инференс, эффекты по периодам,
  низкоуровневые функции и staggered adoption.

Запуск:

```bash
uv run jupyter notebook
```

## Предпосылки идентификации и ограничения реализации

Таблица — начальный чек-лист, а не замена обоснованию конкретного
исследовательского дизайна.

| Метод | Что необходимо обосновать | Ограничения текущей реализации |
| --- | --- | --- |
| Synthetic Control | Нет anticipation и spillover; доноры не получили воздействие; их выпуклая комбинация приближает потенциальный исход treated unit без воздействия; pre-treatment связь информативна после интервенции | Требуется минимум 10 доноров; необходимо анализировать качество pre-fit и чувствительность к составу доноров; экстраполяции за пределы их convex hull нет |
| DiD 2×2 | Параллельные тренды потенциальных исходов без воздействия; нет anticipation и interference; состав групп стабилен; timing не определяется неучтёнными шоками outcome | Реализована классическая регрессия с двумя группами и двумя периодами; standard errors — обычные гомоскедастичные OLS, без кластерной или heteroskedasticity-robust поправки; сам по себе API не подходит для современного staggered DiD |
| SDID | Нет anticipation и spillover; контролей и pre-периодов достаточно для балансировки латентной структуры; есть адекватный overlap и стабильное измерение | Веса находятся SLSQP, инференс основан на взвешенной регрессии; кластерные SE не воспроизводят полный jackknife/placebo-инференс статьи; при малом числе кластеров или слабом pre-fit неопределённость может быть ненадёжной |
| Staggered SDID wrapper | Предпосылки SDID выполняются внутри когорт, а выбранные веса агрегации соответствуют целевому estimand | Когорты оцениваются отдельно и затем агрегируются; это не исчерпывающий modern staggered-treatment framework и он автоматически не устраняет гетерогенность эффектов |
| Feature/propensity matching | Conditional exchangeability по наблюдаемым ковариатам, overlap/positivity, корректные pre-treatment признаки | Matching только выбирает контролей: он не устраняет скрытый confounding и не исправляет leakage post-treatment признаков |
| Conformal inference | Выбранная схема перестановок residuals допустима при null, а их поведение достаточно стабильно/exchangeable | Доверительные множества зависят от сетки null-значений и окна; грубая сетка и временная зависимость могут существенно менять ответ |

Для любого метода до запуска модели стоит явно определить estimand, treatment,
дату интервенции, допустимый donor pool, outcome, окно anticipation и возможные
spillover-эффекты.

## Воспроизводимость и проверки

Lock-файл включён в репозиторий. Воспроизвести окружение и запустить все проверки:

```bash
uv sync --locked --all-groups
PYTHONPATH=. uv run --locked pytest -q
uv run --locked ruff check causal_loom tests
uv run --locked ruff format --check causal_loom tests
uv build
```

Тесты покрывают публичные API, валидацию входных данных, конструкторы графиков,
варианты matching, восстановление известных эффектов на симулированных данных и
опубликованный DiD-пример из handbook. Это полезная защита от регрессий, но не
независимый научный аудит и не заявление о полном совпадении со всеми эталонными
реализациями.

Для воспроизводимого анализа:

1. зафиксируйте release tag или commit SHA;
2. сохраните snapshot данных и описание их происхождения;
3. запишите аргументы оценивания, seed, исключения доноров и фильтры дат;
4. публикуйте качество pre-fit и sensitivity diagnostics вместе с эффектом;
5. сравнивайте выводы между заранее обоснованными спецификациями, а не только по
   модели с лучшим fit.

## Статус проекта

API имеет версию `0.x` и может меняться до стабильного релиза. Ближайшие
направления развития:

- более широкая сверка reference values с опубликованными примерами и
  каноническими пакетами;
- более богатый robust inference, включая bootstrap/jackknife там, где он
  методологически оправдан;
- современные event-study и staggered-adoption DiD estimators;
- API-документация и небольшие end-to-end примеры;
- профилирование производительности и более широкая CI-матрица;
- публикация пакета только после стабилизации API и документации происхождения.

Issue и сфокусированные pull request приветствуются. См.
[CONTRIBUTING.md](CONTRIBUTING.md), [Code of Conduct](CODE_OF_CONDUCT.md) и
[политику безопасности](SECURITY.md).

## Научные источники

- Matheus Facure,
  [*Causal Inference for the Brave and True*](https://matheusfacure.github.io/python-causality-handbook/landing-page.html):
  [Synthetic Control](https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html),
  [Difference-in-Differences](https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html)
  и [Synthetic Difference-in-Differences](https://matheusfacure.github.io/python-causality-handbook/25-Synthetic-Diff-in-Diff.html).
- Abadie, Diamond, Hainmueller (2010),
  [“Synthetic Control Methods for Comparative Case Studies”](https://doi.org/10.1198/jasa.2009.ap08746).
- Arkhangelsky, Athey, Hirshberg, Imbens, Wager (2021),
  [“Synthetic Difference-in-Differences”](https://doi.org/10.1257/aer.20190159).
- Chernozhukov, Wüthrich, Zhu,
  [“An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls”](https://arxiv.org/abs/1712.09089).
- Rosenbaum, Rubin (1983),
  [“The Central Role of the Propensity Score in Observational Studies for Causal Effects”](https://doi.org/10.1093/biomet/70.1.41).

При использовании методов в исследовании цитируйте в первую очередь
оригинальные научные работы, а не только этот репозиторий.

Для цитирования самой программы используйте метаданные из
[`CITATION.cff`](CITATION.cff). GitHub также показывает их через кнопку
**Cite this repository**.

## Авторство, лицензии и происхождение данных

Это независимая образовательная реализация, не аффилированная с авторами
handbook или перечисленных статей и не одобренная ими.

Проект вдохновлён открытым handbook Matheus Facure
[*Causal Inference for the Brave and True*](https://github.com/matheusfacure/python-causality-handbook).
Части образовательной структуры и реализации вдохновлены этой работой или
адаптированы из неё на условиях MIT. Необходимая атрибуция и текст исходной
лицензии сохранены в [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), а краткая
атрибуция для дистрибутива — в [`NOTICE`](NOTICE).

Учебные данные находятся в `notebooks/data/`:

| Локальный файл | Происхождение и назначение |
| --- | --- |
| [`smoking.csv`](notebooks/data/smoking.csv) | Побайтово совпадает с закреплённой версией [`smoking.csv`](https://github.com/matheusfacure/python-causality-handbook/blob/2edbd69f2e3629dd2f6ec805c72f0dbe651d713a/causal-inference-for-the-brave-and-true/data/smoking.csv) из handbook; используется в главах 15 и 25 для кейса California Proposition 99, связанного с работой Abadie, Diamond, Hainmueller (2010) |
| [`billboard_impact.csv`](notebooks/data/billboard_impact.csv) | Побайтово совпадает с закреплённой версией [`billboard_impact.csv`](https://github.com/matheusfacure/python-causality-handbook/blob/2edbd69f2e3629dd2f6ec805c72f0dbe651d713a/causal-inference-for-the-brave-and-true/data/billboard_impact.csv); это синтетические данные, сгенерированные upstream-проектом для DiD-примера из главы 13 |

Файлы включены для воспроизводимости примеров. Для них сохранены условия и
атрибуция upstream-проекта; лицензия Apache-2.0 этого репозитория не отменяет
отдельные условия для сторонних материалов. Контрольные суммы, поля и подробная
provenance приведены в
[`notebooks/data/README.md`](notebooks/data/README.md), условия — в
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Для `smoking.csv` известная цепочка источников задокументирована, но ни handbook,
ни цитируемая статья не дают отдельной лицензионной гарантии от каждого
поставщика исходных скомпилированных наблюдений. Для коммерческого
распространения или сценария, где необходима формальная гарантия прав на данные,
получите отдельно лицензированную копию у первичных поставщиков либо обратитесь
за квалифицированной юридической консультацией.

Оригинальный код проекта распространяется по
[Apache License 2.0](LICENSE). [`NOTICE`](NOTICE) и
[third-party notices](THIRD_PARTY_NOTICES.md) должны сохраняться в дистрибутиве;
сторонние материалы остаются под своими лицензиями. История изменений — в
[CHANGELOG.md](CHANGELOG.md).
