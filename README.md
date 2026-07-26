# CausalLoom

[![CI](https://github.com/zugzvangg/causal-loom/actions/workflows/ci.yml/badge.svg)](https://github.com/zugzvangg/causal-loom/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)](#project-status)

[Русская версия](README.ru.md)

CausalLoom is a compact, tested Python toolkit for learning and prototyping
causal analyses on tabular and panel data. It turns several notebook-style
estimators into reusable APIs with explicit result objects, diagnostics,
inference helpers, and visualizations.

The project currently covers Synthetic Control, classic 2×2
Difference-in-Differences (DiD), Synthetic Difference-in-Differences (SDID),
conformal and placebo inference, control-group matching, and batch
synthetic-control experiments.

> [!IMPORTANT]
> This is an experimental educational project, not a drop-in replacement for a
> peer-reviewed causal inference package. A successful fit does not establish
> identification. You remain responsible for the research design, assumptions,
> data quality, and interpretation.

## Why this project

Many causal inference examples end as one-off notebooks. This repository explores
the engineering work required to turn those examples into a small library:

- consistent class and one-shot function APIs;
- immutable result dataclasses with estimates, uncertainty, and diagnostics;
- validation for common panel-shape and treatment-design errors;
- constrained synthetic weights estimated on the probability simplex;
- feature-based and propensity-score control selection;
- placebo paths, conformal p-values, and confidence-set helpers;
- Plotly diagnostics and reproducible executable notebooks;
- unit tests for API behavior, failure modes, and recovery of known simulated
  effects.

## Capabilities

| Area | Main API | What it returns or supports |
| --- | --- | --- |
| Synthetic Control | `SyntheticControl`, `SynthControlModel` | Non-negative donor weights summing to one, treated-vs-synthetic series, effect paths, fit loss |
| Synthetic Control inference | methods on `SyntheticControl` | In-space placebo paths and randomization-style p-values; conformal residuals, p-value grids, confidence sets, dashboards |
| Difference-in-Differences | `DifferenceInDifferences`, `difference_in_differences` | 2×2 DiD estimate, conventional OLS standard error, p-value, confidence interval, cell decomposition, trend diagnostics |
| Synthetic DiD | `SyntheticDifferenceInDifferences`, `synthetic_difference_in_differences`, `fit_sdid` | Unit/time weights, weighted estimate, optional residualization, fixed effects, clustered standard errors, period effects |
| Staggered SDID | `fit_staggered_sdid` | Cohort-specific fits and a configurable weighted aggregate |
| Matching | `select_control_group_by_features`, `propensity_score_match` | Matched control identifiers and matching metadata |
| Batch experiments | `Experiment` | Multiple synthetic-control fits, fit-quality filtering, aggregated effects, paired-test or permutation inference |
| Visualization | `causal_loom.visualization` | Plotly figures for series, effects, weights, placebos, DiD trends, conformal outputs, and aggregate experiments |

## Installation

Python 3.10.11 or newer is required. The project is currently installed from
source; it is not advertised as a PyPI release.

Using [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/zugzvangg/causal-loom.git
cd causal-loom
uv sync --locked --all-groups
```

Using `pip` in a virtual environment:

```bash
git clone https://github.com/zugzvangg/causal-loom.git
cd causal-loom
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Quickstart

The following self-contained example generates a small panel, fits a synthetic
control, and measures the post-treatment gap. It does not depend on a bundled
dataset.

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

# Opens treated-vs-synthetic, donor-weight, and effect figures.
model.plot_results("outcome", top_weights_to_plot=5, with_effect=True)
```

`max_iter` is the SLSQP iteration budget and `tolerance` is its convergence
tolerance. The legacy names `epochs` and `lr` remain temporarily available as
deprecated aliases.

### 2×2 Difference-in-Differences

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

`did_frame` must contain numeric `outcome` values and binary `treated` and `post`
columns, with observations in all four treatment-by-period cells.

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

See the notebooks for complete data preparation and diagnostic workflows.

## Demonstration notebooks

- [`notebooks/smoking_library_demo.ipynb`](notebooks/smoking_library_demo.ipynb)
  — Synthetic Control, matching, effect paths, placebo analysis, conformal
  inference, and the batch `Experiment` API.
- [`notebooks/billboard_did_library_demo.ipynb`](notebooks/billboard_did_library_demo.ipynb)
  — a guided classic 2×2 Difference-in-Differences analysis.
- [`notebooks/synthetic_did_library_demo.ipynb`](notebooks/synthetic_did_library_demo.ipynb)
  — SDID unit/time weights, residualization, clustered inference, period effects,
  low-level helpers, and staggered adoption.

Launch them with:

```bash
uv run jupyter notebook
```

## Identification assumptions and implementation limits

The table below is a starting checklist, not a substitute for a design-specific
argument.

| Method | Core assumptions to defend | Important implementation limits |
| --- | --- | --- |
| Synthetic Control | No anticipation or spillovers; donors are untreated; a convex combination of donors approximates the treated unit's untreated path; the pre-period relationship remains informative after treatment | Requires at least 10 donor units in this implementation; fit quality and donor sensitivity must be inspected; extrapolation outside the donor convex hull is not supported |
| 2×2 DiD | Parallel untreated trends; no anticipation; stable treatment composition; no interference; treatment timing is not driven by unmodeled outcome shocks | Implements a classic two-group/two-period interaction regression; standard errors are conventional homoskedastic OLS, not clustered or heteroskedasticity-robust; not suitable for modern staggered-adoption DiD by itself |
| SDID | No anticipation or spillovers; control outcomes and pre-periods contain enough information to balance latent unit/time structure; adequate overlap and stable measurement | Uses SLSQP weights and weighted-regression inference; clustered SE is not the paper's full jackknife/placebo inference procedure; small numbers of clusters or weak pre-fit can make uncertainty unreliable |
| Staggered SDID wrapper | The cohort-specific SDID assumptions hold and aggregation weights target a meaningful estimand | Fits cohorts separately and aggregates them; it is not a comprehensive modern staggered-treatment framework and does not resolve treatment-effect heterogeneity automatically |
| Feature/propensity matching | Conditional exchangeability given observed covariates, overlap/positivity, correctly measured pre-treatment covariates | Matching only selects controls; it cannot remove hidden confounding or repair post-treatment covariate leakage |
| Conformal inference | The chosen residual permutation scheme is justified under the null, and residual behavior is sufficiently stable/exchangeable for the application | Confidence sets depend on the null grid and window; coarse grids and serial dependence can materially change results |

Across all methods, explicitly define the estimand, treatment, intervention date,
eligible donor population, outcome, anticipation window, and potential spillovers
before fitting a model.

## Reproducibility and verification

The lockfile is committed. To recreate the development environment and run the
checks:

```bash
uv sync --locked --all-groups
PYTHONPATH=. uv run --locked pytest -q
uv run --locked ruff check causal_loom tests
uv run --locked ruff format --check causal_loom tests
uv build
```

The tests exercise public APIs, input validation, visualization constructors,
matching paths, numerical recovery on simulated data with known effects, and a
published handbook DiD example. They are useful regression checks, but they are
not an independent scientific audit or a claim of parity with every reference
implementation.

For a reproducible analysis:

1. pin this repository to a release tag or commit SHA;
2. retain the input-data snapshot and its provenance;
3. record every estimator argument, random seed, donor exclusion, and date filter;
4. report pre-treatment fit and sensitivity diagnostics alongside the effect;
5. compare conclusions across defensible specifications, not only the best fit.

## Project status

The API is at `0.x` and may change before a stable release. Current priorities are:

- broader reference-value comparisons against published examples and canonical
  implementations;
- richer robust inference, including bootstrap/jackknife options where justified;
- modern event-study and staggered-adoption DiD estimators;
- API documentation and smaller end-to-end examples;
- performance profiling and broader Python/platform CI coverage;
- a packaged release only after the public API and provenance documentation settle.

Issues and focused pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md),
the [Code of Conduct](CODE_OF_CONDUCT.md), and the
[security policy](SECURITY.md).

## Academic references

The implementation is informed by the following educational and research
materials:

- Matheus Facure,
  [*Causal Inference for the Brave and True*](https://matheusfacure.github.io/python-causality-handbook/landing-page.html):
  [Synthetic Control](https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html),
  [Difference-in-Differences](https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html),
  and [Synthetic Difference-in-Differences](https://matheusfacure.github.io/python-causality-handbook/25-Synthetic-Diff-in-Diff.html).
- Abadie, Diamond, and Hainmueller (2010),
  [“Synthetic Control Methods for Comparative Case Studies”](https://doi.org/10.1198/jasa.2009.ap08746).
- Arkhangelsky, Athey, Hirshberg, Imbens, and Wager (2021),
  [“Synthetic Difference-in-Differences”](https://doi.org/10.1257/aer.20190159).
- Chernozhukov, Wüthrich, and Zhu,
  [“An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls”](https://arxiv.org/abs/1712.09089).
- Rosenbaum and Rubin (1983),
  [“The Central Role of the Propensity Score in Observational Studies for Causal Effects”](https://doi.org/10.1093/biomet/70.1.41).

Please cite the original methodological papers—not only this repository—when
using these estimators in research.

To cite the software itself, use the metadata in
[`CITATION.cff`](CITATION.cff). GitHub also exposes it through the repository's
**Cite this repository** control.

## Credits, licensing, and data provenance

This is an independent educational implementation. It is not affiliated with or
endorsed by the authors of the handbook or papers above.

The project was inspired by Matheus Facure's open-source
[*Causal Inference for the Brave and True*](https://github.com/matheusfacure/python-causality-handbook).
Parts of the educational framing and implementation were inspired by or adapted
from that MIT-licensed work. Required attribution and the upstream MIT license
are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); the compact
distribution attribution is in [`NOTICE`](NOTICE).

The demonstration datasets are kept under `notebooks/data/`:

| Local file | Provenance and use |
| --- | --- |
| [`smoking.csv`](notebooks/data/smoking.csv) | Byte-identical to the handbook's pinned [`smoking.csv`](https://github.com/matheusfacure/python-causality-handbook/blob/2edbd69f2e3629dd2f6ec805c72f0dbe651d713a/causal-inference-for-the-brave-and-true/data/smoking.csv); used in Chapters 15 and 25 for the California Proposition 99 case study associated with Abadie, Diamond, and Hainmueller (2010) |
| [`billboard_impact.csv`](notebooks/data/billboard_impact.csv) | Byte-identical to the handbook's pinned [`billboard_impact.csv`](https://github.com/matheusfacure/python-causality-handbook/blob/2edbd69f2e3629dd2f6ec805c72f0dbe651d713a/causal-inference-for-the-brave-and-true/data/billboard_impact.csv); synthetic data generated by the upstream project for Chapter 13's billboard/deposit DiD example |

These files are included for reproducible demonstrations. They retain the
upstream attribution and licensing notice; the project's Apache-2.0 license does
not erase separate third-party terms. Before reusing a bundled dataset outside
these examples, review the detailed checksums, fields, and provenance in
[`notebooks/data/README.md`](notebooks/data/README.md) and the terms in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For `smoking.csv`, the known chain of sources is documented, but neither the
handbook nor the cited paper provides a separate license warranty from every
provider of the underlying compiled observations. For commercial redistribution
or any workflow requiring a formal data-rights warranty, obtain a separately
licensed copy from the original providers or seek qualified legal advice.

Original project code is licensed under the
[Apache License 2.0](LICENSE). The [`NOTICE`](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md) travel with the distribution;
third-party materials remain under their respective terms. See
[CHANGELOG.md](CHANGELOG.md) for release notes.
