import numpy as np
import pandas as pd
import pytest

from cib_advanced_experiments.synthetic_difference_in_differences import (
    StaggeredSyntheticDifferenceInDifferencesResult,
    SyntheticDifferenceInDifferences,
    SyntheticDifferenceInDifferencesResult,
    calculate_regularization,
    calculate_sigma_hat,
    calculate_time_regularization,
    fit_sdid,
    fit_staggered_sdid,
    fit_time_weights,
    fit_unit_weights,
    join_sdid_weights,
    residualize_outcome,
    synthetic_difference_in_differences,
)


def _build_sdid_data(seed: int = 301, true_effect: float = 1.8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2024-01-01", periods=10, freq="D")
    treated_units = ["t0", "t1", "t2"]
    control_units = [f"c{i}" for i in range(8)]

    rows = []
    for unit in treated_units + control_units:
        treated = int(unit.startswith("t"))
        unit_fe = rng.normal(0.0, 0.35)
        for idx, date in enumerate(periods):
            post = int(date >= pd.Timestamp("2024-01-07"))
            y = (
                10.0
                + unit_fe
                + 0.15 * idx
                + 0.2 * treated
                + 0.35 * post
                + true_effect * treated * post
                + rng.normal(0.0, 0.05)
            )
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "treated": treated,
                    "post": post,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def _build_staggered_sdid_data(seed: int = 401, true_effect: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2024-01-01", periods=12, freq="D")
    never_treated_units = [f"c{i}" for i in range(8)]
    cohort_a = ["t0", "t1"]
    cohort_b = ["t2", "t3"]
    cohort_start = {
        "t0": pd.Timestamp("2024-01-06"),
        "t1": pd.Timestamp("2024-01-06"),
        "t2": pd.Timestamp("2024-01-09"),
        "t3": pd.Timestamp("2024-01-09"),
    }

    rows = []
    for unit in never_treated_units + cohort_a + cohort_b:
        treated = int(unit.startswith("t"))
        unit_fe = rng.normal(0.0, 0.3)
        start = cohort_start.get(unit)
        for idx, date in enumerate(periods):
            post = int(treated and date >= start)
            y = (
                15.0
                + unit_fe
                + 0.1 * idx
                + 0.25 * treated
                + 0.2 * post
                + true_effect * post
                + rng.normal(0.0, 0.08)
            )
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "treated": treated,
                    "post": post,
                    "x_num": float(idx + (1 if treated else 0)),
                    "segment": "A" if unit in cohort_a else "B",
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def test_sdid_estimates_known_effect():
    data = _build_sdid_data()
    sdid = SyntheticDifferenceInDifferences(
        data=data,
        outcome_column="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    result = sdid.fit(alpha=0.1)

    assert isinstance(result, SyntheticDifferenceInDifferencesResult)
    assert abs(result.estimate - 1.8) < 0.3
    assert np.isfinite(result.std_error)
    assert np.isclose(result.unit_weights.sum(), 1.0, atol=1e-6)
    assert np.isclose(result.time_weights.sum(), 1.0, atol=1e-6)
    assert (result.unit_weights >= -1e-10).all()
    assert (result.time_weights >= -1e-10).all()


def test_sdid_function_api():
    data = _build_sdid_data(seed=302, true_effect=1.5)
    result = synthetic_difference_in_differences(
        data=data,
        outcome_column="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    assert isinstance(result, SyntheticDifferenceInDifferencesResult)
    assert np.isfinite(result.estimate)


def test_sdid_raises_on_invalid_binary_columns():
    bad = _build_sdid_data(seed=303)
    bad["post"] = 2
    with pytest.raises(ValueError, match="should be binary"):
        SyntheticDifferenceInDifferences(
            data=bad,
            outcome_column="y",
            unit_column="unit",
            time_column="date",
            treatment_column="treated",
            post_column="post",
        )


def test_sdid_raises_when_no_control_pre_period():
    data = _build_sdid_data(seed=304)
    no_control_pre = data[~((data["treated"] == 0) & (data["post"] == 0))]
    sdid = SyntheticDifferenceInDifferences(
        data=no_control_pre,
        outcome_column="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    with pytest.raises(ValueError, match="No control units in pre-period"):
        sdid.fit()


def test_sdid_parallel_trends_and_breakdown():
    data = _build_sdid_data(seed=305)
    sdid = SyntheticDifferenceInDifferences(
        data=data,
        outcome_column="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    result = sdid.fit()
    trends = sdid.get_parallel_trends_frame(pre_period_only=True)
    breakdown = sdid.get_estimators_breakdown()

    assert {"time", "treated_mean", "control_mean", "gap"}.issubset(trends.columns)
    assert len(trends) > 0
    assert {"estimator", "value"}.issubset(breakdown.columns)
    sdid_value = float(
        breakdown.loc[
            breakdown["estimator"] == "sdid_controlled_difference", "value"
        ].iloc[0]
    )
    assert abs(sdid_value - result.estimate) < 1e-9


def test_sdid_supports_covariate_residualization_and_clustered_se():
    data = _build_sdid_data(seed=306, true_effect=1.6)
    data["x_num"] = np.linspace(0.0, 1.0, len(data))
    data["segment"] = np.where(data["treated"] == 1, "treated", "control")

    sdid = SyntheticDifferenceInDifferences(
        data=data,
        outcome_column="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    result = sdid.fit(
        covariate_columns=["x_num", "segment"],
        cluster_by_unit=True,
    )
    assert result.residualized is True
    assert result.model_outcome_column == "y_resid"
    assert result.residualization_model is not None
    assert result.used_clustered_se is True
    assert result.inference_df >= 1


def test_sdid_can_estimate_period_effects():
    data = _build_sdid_data(seed=307, true_effect=1.7)
    sdid = SyntheticDifferenceInDifferences(
        data=data,
        outcome_column="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    period_effects = sdid.fit_effects_by_period()
    expected_periods = data.loc[data["post"] == 1, "date"].nunique()
    assert len(period_effects) == expected_periods
    assert {"period", "estimate", "pvalue"}.issubset(period_effects.columns)


def test_fit_sdid_wrapper_returns_expected_payload():
    data = _build_sdid_data(seed=308, true_effect=1.4)
    payload = fit_sdid(
        panel=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    assert {"att", "pvalue", "unit_weights", "time_weights", "did_data"}.issubset(
        payload.keys()
    )
    assert np.isfinite(payload["att"])
    assert isinstance(payload["did_data"], pd.DataFrame)


def test_fit_staggered_sdid_returns_cohort_and_aggregate_effects():
    data = _build_staggered_sdid_data()
    result = fit_staggered_sdid(
        data=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
        alpha=0.1,
    )
    assert isinstance(result, StaggeredSyntheticDifferenceInDifferencesResult)
    assert len(result.cohort_results) == 2
    assert np.isfinite(result.aggregate_estimate)
    assert abs(result.aggregate_estimate - 2.0) < 0.6


def test_calculation_helpers_return_finite_values():
    data = _build_sdid_data(seed=309, true_effect=1.9)
    sigma = calculate_sigma_hat(
        panel=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    zeta = calculate_regularization(
        panel=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    zeta_time = calculate_time_regularization(
        panel=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
        ridge_scale=1e-6,
    )
    assert np.isfinite(sigma)
    assert np.isfinite(zeta)
    assert np.isfinite(zeta_time)
    assert sigma > 0
    assert zeta > 0
    assert zeta_time >= 0


def test_calculate_time_regularization_zero_scale_returns_zero():
    data = _build_sdid_data(seed=314, true_effect=1.4)
    zeta_time = calculate_time_regularization(
        panel=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
        ridge_scale=0.0,
    )
    assert zeta_time == 0.0


def test_residualize_outcome_direct_api():
    data = _build_sdid_data(seed=310, true_effect=1.5)
    data["x_num"] = np.linspace(0.0, 1.0, len(data))
    data["segment"] = np.where(data["treated"] == 1, "treated", "control")
    residualized, residual_col, model_info = residualize_outcome(
        panel=data,
        outcome_col="y",
        covariate_cols=["x_num", "segment"],
        treatment_column="treated",
        post_column="post",
    )
    assert residual_col == "y_resid"
    assert residual_col in residualized.columns
    assert np.isfinite(residualized[residual_col]).all()
    assert "coefficients" in model_info
    assert model_info["n_fit"] > 0


def test_residualize_outcome_raises_on_missing_covariates():
    data = _build_sdid_data(seed=315, true_effect=1.5)
    with pytest.raises(ValueError, match="Missing covariate columns"):
        residualize_outcome(
            panel=data,
            outcome_col="y",
            covariate_cols=["missing_cov"],
            treatment_column="treated",
            post_column="post",
        )


def test_low_level_weight_helpers_and_join():
    data = _build_sdid_data(seed=311, true_effect=1.2)
    unit_w = fit_unit_weights(
        panel=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    time_w = fit_time_weights(
        panel=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
        zeta_time_small=1e-8,
    )
    joined = join_sdid_weights(
        panel=data,
        unit_weights=unit_w,
        time_weights=time_w,
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
        outcome_column="y",
    )
    assert np.isclose(unit_w.sum(), 1.0, atol=1e-6)
    assert np.isclose(time_w.sum(), 1.0, atol=1e-6)
    assert {"unit_weight", "time_weight", "sdid_weight"}.issubset(joined.columns)
    assert np.isfinite(joined["sdid_weight"]).all()
    assert (joined["sdid_weight"] > 0).all()


def test_join_sdid_weights_allows_missing_outcome_column():
    data = _build_sdid_data(seed=316, true_effect=1.2)
    unit_w = fit_unit_weights(
        panel=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    time_w = fit_time_weights(
        panel=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    panel_without_outcome = data.drop(columns=["y"])
    joined = join_sdid_weights(
        panel=panel_without_outcome,
        unit_weights=unit_w,
        time_weights=time_w,
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
        outcome_column="outcome",
    )
    assert {"sdid_weight", "unit_weight", "time_weight"}.issubset(joined.columns)


def test_sdid_fit_without_fe_and_without_clustered_se():
    data = _build_sdid_data(seed=312, true_effect=1.3)
    sdid = SyntheticDifferenceInDifferences(
        data=data,
        outcome_column="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    result = sdid.fit(
        use_fixed_effects=False,
        cluster_by_unit=False,
    )
    assert result.used_fixed_effects is False
    assert result.used_clustered_se is False
    assert result.cluster_column is None
    assert result.inference_df > 0


def test_sdid_fit_supports_custom_cluster_column():
    data = _build_sdid_data(seed=313, true_effect=1.5)
    data["cluster_id"] = data["unit"].str.slice(0, 1)
    sdid = SyntheticDifferenceInDifferences(
        data=data,
        outcome_column="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
    )
    result = sdid.fit(cluster_by_unit=True, cluster_column="cluster_id")
    assert result.cluster_column == "cluster_id"


def test_fit_staggered_sdid_with_explicit_treatment_start_column():
    data = _build_staggered_sdid_data(seed=402, true_effect=1.8)
    treatment_starts = (
        data[(data["treated"] == 1) & (data["post"] == 1)]
        .groupby("unit")["date"]
        .min()
        .rename("treatment_start")
    )
    data = data.merge(treatment_starts, left_on="unit", right_index=True, how="left")
    result = fit_staggered_sdid(
        data=data,
        outcome_col="y",
        unit_column="unit",
        time_column="date",
        treatment_column="treated",
        post_column="post",
        treatment_start_column="treatment_start",
        include_not_yet_treated_as_controls=False,
    )
    assert isinstance(result, StaggeredSyntheticDifferenceInDifferencesResult)
    assert len(result.cohort_results) >= 1
    assert np.isfinite(result.aggregate_estimate)


def test_fit_staggered_sdid_raises_on_invalid_weighting_column():
    data = _build_staggered_sdid_data(seed=403, true_effect=2.0)
    with pytest.raises(ValueError, match="weighting_column"):
        fit_staggered_sdid(
            data=data,
            outcome_col="y",
            unit_column="unit",
            time_column="date",
            treatment_column="treated",
            post_column="post",
            weighting_column="missing_weight",
        )


def test_fit_staggered_sdid_raises_when_no_treated_cohorts():
    data = _build_sdid_data(seed=317, true_effect=1.1)
    data["treated"] = 0
    data["post"] = 0
    with pytest.raises(ValueError, match="No treated cohorts"):
        fit_staggered_sdid(
            data=data,
            outcome_col="y",
            unit_column="unit",
            time_column="date",
            treatment_column="treated",
            post_column="post",
        )
