"""One-week-ahead features, with observed history updated between test weeks."""

import numpy as np
import pandas as pd
from .quality import validate_weekly

NUMERIC = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_avg_2wk",
    "rolling_avg_4wk",
    "lag_8",
    "lag_13",
    "lag_26",
    "lag_52",
    "rolling_avg_8wk",
    "rolling_avg_13wk",
    "rolling_avg_26wk",
    "rolling_median_13wk",
    "rolling_std_13wk",
    "zero_share_13wk",
    "trend_4_to_13",
    "week_sin",
    "week_cos",
    "price_lag_1",
    "is_high_demand",
]
EXOGENOUS = [
    "promo_intensity",
    "holiday_flag",
    "competitor_price_index",
    "economic_index",
]


def build_features(
    weekly: pd.DataFrame, training_cutoff
) -> tuple[pd.DataFrame, list[str]]:
    data = weekly.copy()
    data["week_start"] = pd.to_datetime(data.week_start)
    validate_weekly(data)
    data = data.sort_values(["country", "segment", "week_start"]).reset_index(drop=True)
    grouped = data.groupby(["country", "segment"], sort=False)
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        data[f"lag_{lag}"] = grouped.units_sold.shift(lag)
    for width in [2, 4, 8, 13, 26]:
        data[f"rolling_avg_{width}wk"] = grouped.units_sold.transform(
            lambda s: s.shift(1).rolling(width, min_periods=min(width, 4)).mean()
        )
    data["rolling_median_13wk"] = grouped.units_sold.transform(
        lambda s: s.shift(1).rolling(13, min_periods=4).median()
    )
    data["rolling_std_13wk"] = grouped.units_sold.transform(
        lambda s: s.shift(1).rolling(13, min_periods=4).std()
    )
    data["zero_share_13wk"] = grouped.units_sold.transform(
        lambda s: s.eq(0).astype(float).shift(1).rolling(13, min_periods=4).mean()
    )
    data["trend_4_to_13"] = data.rolling_avg_4wk / data.rolling_avg_13wk.clip(lower=1)
    data["price_lag_1"] = grouped.avg_price.transform(lambda s: s.ffill().shift(1))
    # Calendar variables are known before prediction; use an annual continuous phase.
    phase = (data.week_start.dt.dayofyear - 1) / 365.25
    data["week_sin"], data["week_cos"] = (
        np.sin(2 * np.pi * phase),
        np.cos(2 * np.pi * phase),
    )
    cutoff = pd.Timestamp(training_cutoff)
    history = data.loc[data.week_start < cutoff, "lag_1"].dropna()
    if history.empty:
        raise ValueError("No training history to estimate the high-demand threshold.")
    threshold = history.quantile(0.95)
    data["is_high_demand"] = (data.lag_1 > threshold).astype(int)
    features = NUMERIC.copy()
    if data.country.eq("Dominican Republic").any():
        data["high_demand_rd"] = data.is_high_demand * data.country.eq(
            "Dominican Republic"
        ).astype(int)
        features.append("high_demand_rd")
    for c in EXOGENOUS:
        if c in data:
            features.append(
                c
            )  # Demo only: generator assumes these are planned/available.
    if {"promo_intensity", "holiday_flag"}.issubset(data):
        data["promo_x_holiday"] = data.promo_intensity * data.holiday_flag
        features.append("promo_x_holiday")
    data = data.dropna(subset=["lag_1", "lag_2", "lag_4", "rolling_avg_4wk"]).copy()
    data.attrs["high_demand_threshold"] = float(threshold)
    return data, features


def temporal_split(features: pd.DataFrame, cutoff):
    train = features.loc[features.week_start < pd.Timestamp(cutoff)].copy()
    test = features.loc[features.week_start >= pd.Timestamp(cutoff)].copy()
    if train.empty or test.empty or train.week_start.max() >= test.week_start.min():
        raise ValueError("The temporal split must have nonempty, disjoint periods.")
    return train, test
