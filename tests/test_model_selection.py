import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from market_insights.data import real_weekly, simulate_pricing
from market_insights.forecasting import select_ensemble, scores
from market_insights.features import build_features
from market_insights.quality import clean_transactions, validate_weekly
from market_insights.pricing import evaluate_pricing
from market_insights.visualization import plot_forecast


def test_final_holdout_cannot_change_ensemble_selection(weekly):
    cutoff = pd.Timestamp("2021-12-06")
    first, folds, search, _ = select_ensemble(
        weekly, cutoff, folds=2, validation_weeks=4, rf_trees=8, xgb_trees=10
    )
    changed = weekly.copy()
    changed.loc[changed.week_start >= cutoff, "units_sold"] = 1e8
    second, other_folds, other_search, _ = select_ensemble(
        changed, cutoff, folds=2, validation_weeks=4, rf_trees=8, xgb_trees=10
    )
    assert first == second
    assert_frame_equal(folds, other_folds)
    assert_frame_equal(search, other_search)
    assert pd.to_datetime(folds.validation_end).max() < cutoff
    assert sum(first["ensemble_weights"]) == pytest.approx(1)
    assert min(first["ensemble_weights"][:2]) > 0 and first["ensemble_weights"][2] >= 0


def test_cohort_selection_precedes_every_validation_week():
    weeks = pd.date_range("2021-01-04", periods=104, freq="W-MON")
    rows = [
        dict(
            order_date=week,
            product_id=sku,
            country="United Kingdom",
            quantity=float(qty),
            gross_sales=float(qty * 2),
            product_name=sku,
        )
        for week in weeks
        for sku, qty in [("10001", 30), ("10002", 20), ("10003", 10)]
    ]
    rows.append({**rows[-3], "order_date": weeks[-1] + pd.Timedelta(days=6)})
    data = pd.DataFrame(rows)
    _, cutoff, scope = real_weekly(data, top_n=2)
    origin = pd.Timestamp(scope["selection_cutoff"])
    assert origin == cutoff - pd.Timedelta(weeks=24)
    data.loc[data.order_date.ge(origin) & data.product_id.eq("10003"), "quantity"] = 1e8
    _, _, changed_scope = real_weekly(data, top_n=2)
    assert (
        scope["selected_series"]
        == changed_scope["selected_series"]
        == ["10001", "10002"]
    )


def test_long_history_features_use_past_values(weekly):
    frame, names = build_features(weekly, "2021-12-06")
    row = frame.loc[frame.segment.eq("A")].iloc[-1]
    assert row.lag_52 == 21
    assert row.rolling_avg_13wk == pytest.approx(66)
    assert "high_demand_rd" not in names


@pytest.mark.parametrize("column", ["Quantity", "Price"])
def test_nonfinite_transaction_values_are_audited(transactions, column):
    transactions[column] = transactions[column].astype(float)
    transactions.loc[0, column] = np.inf
    clean, audit = clean_transactions(transactions)
    assert audit["invalid_required_rows"] == 1 and len(clean) == 2


def test_weekly_targets_and_coverage_are_checked(weekly):
    changed = weekly.copy()
    changed.loc[0, "units_sold"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_weekly(changed)
    with pytest.raises(ValueError, match="covered calendar"):
        validate_weekly(weekly.drop(index=0))


def test_scores_reject_invalid_shapes_and_infinities():
    with pytest.raises(ValueError, match="matching"):
        scores([1, 2], [1])
    with pytest.raises(ValueError, match="finite"):
        scores([1, np.inf], [1, 2])


def test_pricing_rejects_invalid_numeric_outcomes():
    experiment, costs = simulate_pricing()
    experiment["units_sold"] = experiment.units_sold.astype(float)
    experiment.loc[0, "units_sold"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        evaluate_pricing(experiment, costs)


def test_forecast_figure_cannot_mix_countries(tmp_path):
    frame = pd.DataFrame(
        dict(
            segment=["A", "A"],
            country=["UK", "US"],
            week_start=pd.to_datetime(["2024-01-01"] * 2),
        )
    )
    with pytest.raises(ValueError, match="one nonempty"):
        plot_forecast(frame, "A", "test", tmp_path)
