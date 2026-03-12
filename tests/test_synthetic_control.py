import numpy as np
import pandas as pd
import pytest

from cib_advanced_experiments.matching import select_control_group_by_features
from cib_advanced_experiments.propensity_score_matching import propensity_score_match
from cib_advanced_experiments.synthetic_control import SyntheticControl


import numpy as np
import pandas as pd

from cib_advanced_experiments.synthetic_control import SyntheticControl


def test_synthetic_control_uses_external_feature_matching():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    units = [f"u{i}" for i in range(12)]

    rows = []
    for idx, unit in enumerate(units):
        segment = "A" if idx < 11 else "B"
        base = 10.0 + idx * 0.1
        for date in dates:
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "y": base + rng.normal(0.0, 0.1),
                    "segment": segment,
                    "pre_metric": base,
                }
            )
    data = pd.DataFrame(rows)

    sc = SyntheticControl(
        data=data,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treated_unit="u0",
        treatment_date="2024-01-06",
        number_of_units_for_matching=10,
        categorical_matching=["segment"],
        numeric_matching=["pre_metric"],
        epochs=5,
        lr=0.01,
    )
    sc.fit()

    weights = sc.get_weights()
    assert weights.shape == (10,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)


def test_synthetic_control_defaults_to_feature_matching():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    for idx, unit in enumerate(units):
        for date in dates:
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "y": 5.0 + idx * 0.1 + rng.normal(0.0, 0.1),
                    "pre_metric": float(idx),
                }
            )
    data = pd.DataFrame(rows)

    sc = SyntheticControl(
        data=data,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treated_unit="u0",
        treatment_date="2024-01-06",
        number_of_units_for_matching=10,
        numeric_matching=["pre_metric"],
        epochs=5,
        lr=0.01,
    )
    sc.fit()
    assert sc.get_weights().shape == (10,)


def test_synthetic_control_can_switch_to_propensity_matching():
    rng = np.random.default_rng(13)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    for idx, unit in enumerate(units):
        income = 40 + idx * 2
        age = 20 + idx
        for date in dates:
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "y": income * 0.1 + age * 0.05 + rng.normal(0.0, 0.05),
                    "income": float(income),
                    "age": float(age),
                }
            )
    data = pd.DataFrame(rows)

    sc = SyntheticControl(
        data=data,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treated_unit="u0",
        treatment_date="2024-01-08",
        number_of_units_for_matching=10,
        matching_method="propensity",
        propensity_covariates=["income", "age"],
        epochs=5,
        lr=0.01,
    )
    sc.fit()

    weights = sc.get_weights()
    assert weights.shape == (10,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)


def test_synthetic_control_get_effect_data_contains_effect_column():
    rng = np.random.default_rng(17)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    for idx, unit in enumerate(units):
        for date in dates:
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "y": 3.0 + idx * 0.2 + rng.normal(0.0, 0.05),
                }
            )
    data = pd.DataFrame(rows)

    sc = SyntheticControl(
        data=data,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treated_unit="u0",
        treatment_date="2024-01-06",
        control_group=[f"u{i}" for i in range(1, 11)],
        epochs=5,
        lr=0.01,
    )
    sc.fit()
    effect_data = sc.get_effect_data("y")
    assert {"date", "real", "synthetic", "effect"}.issubset(effect_data.columns)


def test_synthetic_control_raises_for_conflicting_control_group_settings():
    rng = np.random.default_rng(19)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    for idx, unit in enumerate(units):
        for date in dates:
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "y": 3.0 + idx * 0.2 + rng.normal(0.0, 0.05),
                    "segment": "A",
                }
            )
    data = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="Pass only one of them"):
        SyntheticControl(
            data=data,
            unit_column="unit",
            date_column="date",
            features=["y"],
            treated_unit="u0",
            treatment_date="2024-01-06",
            control_group=[f"u{i}" for i in range(1, 11)],
            number_of_units_for_matching=10,
            numeric_matching=["y"],
            epochs=5,
            lr=0.01,
        )


def test_synthetic_control_requires_fit_before_series_access():
    rng = np.random.default_rng(20)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    for idx, unit in enumerate(units):
        for date in dates:
            rows.append({"unit": unit, "date": date, "y": 1.0 + idx + rng.normal()})
    data = pd.DataFrame(rows)

    sc = SyntheticControl(
        data=data,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treated_unit="u0",
        treatment_date="2024-01-06",
        control_group=[f"u{i}" for i in range(1, 11)],
        epochs=5,
        lr=0.01,
    )
    with pytest.raises(AssertionError, match="Fit model first with .fit"):
        sc.get_data_on_features("y")


def test_synthetic_control_raises_on_too_small_control_group():
    rng = np.random.default_rng(21)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    for idx, unit in enumerate(units):
        for date in dates:
            rows.append({"unit": unit, "date": date, "y": 1.0 + idx + rng.normal()})
    data = pd.DataFrame(rows)

    with pytest.raises(AssertionError, match="To small 'control_group'"):
        SyntheticControl(
            data=data,
            unit_column="unit",
            date_column="date",
            features=["y"],
            treated_unit="u0",
            treatment_date="2024-01-06",
            control_group=[f"u{i}" for i in range(1, 6)],
            epochs=5,
            lr=0.01,
        )


def test_synthetic_control_placebo_result_api():
    rng = np.random.default_rng(22)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    for idx, unit in enumerate(units):
        for date in dates:
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "y": 2.0 + idx * 0.15 + rng.normal(0.0, 0.05),
                }
            )
    data = pd.DataFrame(rows)

    sc = SyntheticControl(
        data=data,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treated_unit="u0",
        treatment_date="2024-01-06",
        control_group=[f"u{i}" for i in range(1, 11)],
        epochs=5,
        lr=0.01,
    )
    sc.fit()

    placebo = sc.get_placebo_result(
        feature="y",
        effect_evaluation_aggregation="last_date",
        pre_error_quantile=0.9,
        alternative="two-sided",
    )

    assert {"treated_unit", "date", "effect"}.issubset(placebo.placebo_effects.columns)
    assert {"pre_treatment_mse", "effect"}.issubset(placebo.unit_level_summary.columns)
    assert 0.0 <= placebo.placebo_pvalue <= 1.0
    assert placebo.filtered_unit_level_summary["is_actual_treated"].any()
