import numpy as np
import pandas as pd
import pytest

from cib_advanced_experiments.difference_in_differences import (
    DifferenceInDifferences,
    DifferenceInDifferencesResult,
    difference_in_differences,
)


def _build_did_data(seed: int = 202) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    rows = []
    for unit, treated in [("t0", 1), ("t1", 1), ("c0", 0), ("c1", 0)]:
        for date in dates:
            post = int(date >= pd.Timestamp("2024-01-05"))
            base = 10.0 + treated * 1.0 + post * 0.5 + treated * post * 2.0
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "treated": treated,
                    "post": post,
                    "y": base + rng.normal(0.0, 0.05),
                }
            )
    return pd.DataFrame(rows)


def test_difference_in_differences_estimates_known_effect():
    data = _build_did_data()
    did = DifferenceInDifferences(
        data=data,
        outcome_column="y",
        treatment_column="treated",
        post_column="post",
    )
    result = did.fit(alpha=0.1)

    assert isinstance(result, DifferenceInDifferencesResult)
    assert abs(result.estimate - 2.0) < 0.25
    assert result.n_obs == len(data)
    assert "mean_outcome" in result.group_means.columns
    assert np.isfinite(result.naive_before_after)
    assert np.isfinite(result.naive_post_period_comparison)


def test_difference_in_differences_function_api():
    data = _build_did_data(seed=203)
    result = difference_in_differences(
        data=data,
        outcome_column="y",
        treatment_column="treated",
        post_column="post",
    )
    assert isinstance(result, DifferenceInDifferencesResult)
    assert np.isfinite(result.estimate)


def test_difference_in_differences_raises_on_invalid_binary_columns():
    bad = _build_did_data(seed=204)
    bad["treated"] = 2
    with pytest.raises(ValueError, match="should be binary"):
        DifferenceInDifferences(
            data=bad,
            outcome_column="y",
            treatment_column="treated",
            post_column="post",
        )


def test_difference_in_differences_raises_when_cell_is_empty():
    data = _build_did_data(seed=205)
    data = data[~((data["treated"] == 1) & (data["post"] == 0))]
    did = DifferenceInDifferences(
        data=data,
        outcome_column="y",
        treatment_column="treated",
        post_column="post",
    )
    with pytest.raises(
        ValueError, match="Each DiD cell should have at least one observation"
    ):
        did.fit()


def test_difference_in_differences_parallel_trends_frame():
    data = _build_did_data(seed=206)
    did = DifferenceInDifferences(
        data=data,
        outcome_column="y",
        treatment_column="treated",
        post_column="post",
    )
    trends = did.get_parallel_trends_frame(time_column="date", pre_period_only=True)
    assert {"time", "treated_mean", "control_mean", "gap"}.issubset(trends.columns)
    assert len(trends) > 0


def test_difference_in_differences_estimators_breakdown():
    data = _build_did_data(seed=207)
    did = DifferenceInDifferences(
        data=data,
        outcome_column="y",
        treatment_column="treated",
        post_column="post",
    )
    result = did.fit()
    breakdown = did.get_estimators_breakdown()

    assert {"estimator", "value"}.issubset(breakdown.columns)
    did_value = float(
        breakdown.loc[
            breakdown["estimator"] == "did_controlled_difference", "value"
        ].iloc[0]
    )
    assert abs(did_value - result.estimate) < 1e-9
