import numpy as np
import pandas as pd

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
    assert weights.shape == (10,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
