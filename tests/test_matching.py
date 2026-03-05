import numpy as np
import pandas as pd

from cib_advanced_experiments.matching import select_control_group_by_features
from cib_advanced_experiments.propensity_score_matching import propensity_score_match
from cib_advanced_experiments.synthetic_control import SyntheticControl


def test_select_control_group_by_features_returns_closest_controls():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    rows = []
    for unit_id, base in [("u0", 10.0), ("u1", 10.1), ("u2", 9.8), ("u3", 15.0), ("u4", 10.3)]:
        for date in dates:
            rows.append(
                {
                    "unit": unit_id,
                    "date": date,
                    "revenue": base,
                    "segment": "A",
                }
            )
    data = pd.DataFrame(rows)

    result = select_control_group_by_features(
        data=data,
        unit_column="unit",
        date_column="date",
        treated_unit="u0",
        treatment_date="2024-01-04",
        number_of_units=2,
        categorical_features=["segment"],
        numeric_features=["revenue"],
    )

    assert len(result.control_group) == 2
    assert result.control_group == ["u1", "u2"]
    assert result.used_exact_matching is True
    assert result.used_numeric_matching is True


def test_propensity_score_match_returns_requested_control_group():
    data = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(10)],
            "treated": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "x1": [0.9, 0.88, 0.7, 0.2, 0.1, 0.8, 0.3, 0.4, 0.6, 0.5],
            "x2": [1.0, 0.95, 0.65, 0.1, 0.2, 0.75, 0.25, 0.35, 0.55, 0.45],
        }
    )

    result = propensity_score_match(
        data=data,
        unit_column="unit",
        treatment_column="treated",
        covariates=["x1", "x2"],
        treated_unit="u0",
        number_of_controls=3,
    )

    assert result.treated_unit == "u0"
    assert len(result.control_group) == 3
    assert "propensity_score" in result.scores.columns


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
