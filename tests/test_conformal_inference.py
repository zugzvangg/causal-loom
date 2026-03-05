import numpy as np
import pandas as pd

from cib_advanced_experiments.synthetic_control import SyntheticControl


def _build_panel_data(seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    for idx, unit in enumerate(units):
        baseline = 5 + idx * 0.2
        for date in dates:
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "y": baseline + rng.normal(0.0, 0.05),
                }
            )
    return pd.DataFrame(rows)


def test_conformal_residuals_and_p_value_api():
    data = _build_panel_data()
    sc = SyntheticControl(
        data=data,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treated_unit="u0",
        treatment_date="2024-01-09",
        control_group=[f"u{i}" for i in range(1, 11)],
        epochs=10,
        lr=0.05,
    )

    residuals = sc.get_conformal_residuals(feature="y", null_hypothesis=0.0, window=2)
    assert {"date", "y0", "y0_est", "residuals", "post_intervention"}.issubset(
        residuals.columns
    )
    assert int(residuals["post_intervention"].sum()) == 2

    p_value = sc.get_conformal_p_value(feature="y", null_hypothesis=0.0, window=2)
    assert 0.0 <= p_value <= 1.0


def test_conformal_grid_and_confidence_interval_api():
    data = _build_panel_data(seed=22)
    sc = SyntheticControl(
        data=data,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treated_unit="u0",
        treatment_date="2024-01-09",
        control_group=[f"u{i}" for i in range(1, 11)],
        epochs=8,
        lr=0.05,
    )

    nulls = np.linspace(-0.5, 0.5, 11)
    period = pd.Timestamp("2024-01-10")
    p_grid = sc.get_conformal_p_value_grid_for_period(
        feature="y",
        period=period,
        nulls=nulls,
    )
    assert p_grid.shape == (len(nulls), 1)
    assert p_grid.columns[0] == period

    ci = sc.get_conformal_confidence_interval(
        feature="y",
        nulls=nulls,
        window=2,
        alpha=0.1,
    )
    assert ci.shape[0] == 2
    assert "90_ci_lower" in ci.columns
    assert "90_ci_upper" in ci.columns
