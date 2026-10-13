import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from market_insights.operations import (
    daily_orders,
    daily_features,
    select_daily_ensemble,
    fit_daily,
)
from market_insights.forecasting import scores


def example_daily():
    dates = pd.date_range("2021-01-04", periods=168, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "orders": [float(12 + (i % 7) * 3 + i // 14) for i in range(168)],
        }
    )


def test_daily_features_use_only_past_observations():
    daily = example_daily()
    cut = daily.date.iloc[-21]
    first, _ = daily_features(daily)
    changed = daily.copy()
    changed.loc[changed.date >= cut, "orders"] = 1e7
    second, _ = daily_features(changed)
    assert_frame_equal(first[first.date < cut], second[second.date < cut])
    row = first.iloc[0]
    assert row.lag_7 == daily.orders.iloc[21]
    assert row.lag_28 == daily.orders.iloc[0]


def test_daily_selection_is_invariant_to_final_test():
    daily = example_daily()
    cut = daily.date.iloc[-21]
    a, folds, search, _ = select_daily_ensemble(
        daily, cut, folds=2, validation_days=14, rf_trees=8, xgb_trees=10
    )
    changed = daily.copy()
    changed.loc[changed.date >= cut, "orders"] = 1e7
    b, other_folds, other_search, _ = select_daily_ensemble(
        changed, cut, folds=2, validation_days=14, rf_trees=8, xgb_trees=10
    )
    assert {k: v for k, v in a.items() if k != "validation_ensemble"} == {
        k: v for k, v in b.items() if k != "validation_ensemble"
    }
    for metric in a["validation_ensemble"]:
        assert a["validation_ensemble"][metric] == pytest.approx(
            b["validation_ensemble"][metric], rel=1e-12, abs=1e-12
        )
    assert_frame_equal(folds, other_folds)
    assert_frame_equal(search, other_search)
    assert pd.to_datetime(folds.validation_end).max() < cut
    frame, numeric = daily_features(daily)
    _, pred, _ = fit_daily(
        frame[frame.date < cut],
        frame[frame.date >= cut],
        numeric,
        a["config"],
        a["ensemble_weights"],
        8,
        10,
    )
    np.testing.assert_allclose(
        pred.ensemble,
        pred[["random_forest", "xgboost", "ridge"]].to_numpy()
        @ np.asarray(a["ensemble_weights"]),
    )
    assert pred[["random_forest", "xgboost", "ridge", "ensemble"]].ge(0).all().all()


def test_daily_counts_distinct_orders_instead_of_lines_or_units():
    dates = pd.date_range("2021-01-04", periods=210, freq="D")
    clean = pd.DataFrame(
        [
            {
                "order_date": date,
                "order_id": f"{i}-{order}",
                "country": "United Kingdom",
                "quantity": 20,
            }
            for i, date in enumerate(dates)
            for order in ["a", "a", "b"]
        ]
    )
    daily, scope = daily_orders(clean)
    assert daily.orders.eq(2).all() and len(daily) == 210
    assert scope["calendar_days"] == 210


def test_daily_calendar_contract():
    daily = example_daily()
    with pytest.raises(ValueError, match="contiguous"):
        daily_features(daily.drop(index=3))
    with pytest.raises(ValueError, match="unique"):
        daily_features(pd.concat([daily, daily.iloc[[0]]]))


def test_constant_target_does_not_claim_defined_r2():
    result = scores([0, 0], [1, 1])
    assert result["r2"] is None and result["wape"] is None and result["mae"] == 1


def test_daily_shap_includes_nonnegative_forecast_correction(tmp_path):
    pytest.importorskip("shap")
    from market_insights.explainability import explain

    daily = example_daily()
    cut = daily.date.iloc[-21]
    frame, numeric = daily_features(daily)
    _, pred, fitted = fit_daily(
        frame[frame.date < cut],
        frame[frame.date >= cut],
        numeric,
        weights=(0.2, 0.3, 0.5),
        rf_trees=8,
        xgb_trees=10,
    )
    metrics, figures = tmp_path / "metrics", tmp_path / "figures"
    metrics.mkdir()
    figures.mkdir()
    explain(fitted, "test", figures, metrics)
    audit = pd.read_csv(metrics / "shap_additivity.csv")
    expected = pred.loc[audit.feature_row_index, "ensemble"].to_numpy()
    np.testing.assert_allclose(audit.prediction, expected, atol=0.05)
    np.testing.assert_allclose(audit.reconstructed, expected, atol=0.05)
