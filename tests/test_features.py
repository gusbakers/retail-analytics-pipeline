import pandas as pd
from pandas.testing import assert_frame_equal
from market_insights.features import build_features, temporal_split
from market_insights.forecasting import backtest


def test_future_sales_cannot_change_training_features(weekly):
    cutoff = pd.Timestamp("2021-12-06")
    before, names = build_features(weekly, cutoff)
    changed = weekly.copy()
    changed.loc[changed.week_start >= cutoff, "units_sold"] = 1e9
    after, _ = build_features(changed, cutoff)
    assert_frame_equal(
        before.loc[before.week_start < cutoff], after.loc[after.week_start < cutoff]
    )


def test_lags_use_the_prior_week_and_are_grouped(weekly):
    frame, _ = build_features(weekly, "2021-12-06")
    first = frame.loc[frame.segment.eq("A")].iloc[0]
    assert first.units_sold == 14 and first.lag_1 == 13 and first.lag_4 == 10
    assert first.rolling_avg_4wk == 11.5
    second = frame.loc[frame.segment.eq("B")].iloc[0]
    assert second.lag_1 == 18


def test_temporal_backtests_never_enter_final_holdout(weekly):
    cutoff = pd.Timestamp("2021-12-06")
    frame, _ = build_features(weekly, cutoff)
    train, test = temporal_split(frame, cutoff)
    assert train.week_start.max() < test.week_start.min()
    scores = backtest(
        weekly, cutoff, folds=2, validation_weeks=4, rf_trees=8, xgb_trees=10
    )
    assert pd.to_datetime(scores.validation_end).max() < cutoff
    assert scores.model.value_counts().eq(2).all()
