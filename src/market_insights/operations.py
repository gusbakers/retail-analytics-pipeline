"""Daily paid-order forecasts for historical UK fulfilment capacity planning.

This is a different business target from the weekly SKU-units experiment.
Every model and blend is selected with folds strictly before the holdout.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from .forecasting import scores

CONFIGS = {"balanced": (4, 3, 10), "smooth": (12, 2, 30)}
MODELS = ["weekly_naive", "random_forest", "xgboost", "ridge", "ensemble"]


def daily_orders(clean):
    minimum, maximum = (
        clean.order_date.min().normalize(),
        clean.order_date.max().normalize(),
    )
    first = minimum + pd.Timedelta(days=(7 - minimum.dayofweek) % 7)
    last_monday = maximum - pd.Timedelta(days=maximum.dayofweek)
    if maximum.dayofweek != 6:
        last_monday -= pd.Timedelta(days=7)
    last = last_monday + pd.Timedelta(days=6)
    calendar = pd.date_range(first, last, freq="D")
    uk = clean.loc[clean.country.eq("United Kingdom")].copy()
    if uk.empty or len(calendar) < 196:
        raise ValueError(
            "Daily forecasting requires UK orders and enough complete covered days."
        )
    uk["date"] = uk.order_date.dt.normalize()
    recorded = uk.groupby("date").order_id.nunique()
    panel = (
        recorded.reindex(calendar, fill_value=0)
        .rename("orders")
        .rename_axis("date")
        .reset_index()
    )
    panel["orders"] = panel.orders.astype(float)
    return panel, {
        "target": "Distinct retained paid UK order IDs within each recorded date",
        "country": "United Kingdom",
        "source": "UCI Online Retail II",
        "covered_first_day": str(first.date()),
        "covered_last_day": str(last.date()),
        "calendar_days": len(panel),
        "zero_policy": "No retained order recorded in this covered day; actual opening hours unknown",
        "horizon": "Next day; observations update after each completed day",
    }


def daily_features(daily):
    if not {"date", "orders"}.issubset(daily) or daily.empty:
        raise ValueError("Daily data must contain date and orders.")
    data = daily.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    if data.date.isna().any() or data.date.duplicated().any():
        raise ValueError("Daily dates must be valid and unique.")
    data = data.sort_values("date").reset_index(drop=True)
    if not data.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Daily dates must be contiguous.")
    if (
        not pd.api.types.is_numeric_dtype(data.orders)
        or not np.isfinite(data.orders).all()
        or data.orders.lt(0).any()
    ):
        raise ValueError("Daily orders must be finite nonnegative counts.")
    for lag in [1, 2, 7, 14, 28, 364]:
        data[f"lag_{lag}"] = data.orders.shift(lag)
    for width in [7, 28]:
        data[f"mean_{width}"] = (
            data.orders.shift(1).rolling(width, min_periods=7).mean()
        )
    phase = (data.date.dt.dayofyear - 1) / 365.25
    for harmonic in [1, 2]:
        data[f"sin{harmonic}"] = np.sin(2 * np.pi * harmonic * phase)
        data[f"cos{harmonic}"] = np.cos(2 * np.pi * harmonic * phase)
    data["dow"] = data.date.dt.dayofweek.astype(str)
    data["trend"] = (data.date - data.date.min()).dt.days / 365.25
    data["year_end"] = (
        data.date.dt.month.eq(12) & data.date.dt.day.between(24, 27)
    ).astype(float)
    data["new_year"] = (data.date.dt.month.eq(1) & data.date.dt.day.eq(1)).astype(float)
    numeric = [c for c in data if c not in ["date", "orders", "dow"]]
    return data.loc[data.lag_28.notna()].copy(), numeric


def fit_daily(
    train,
    test,
    numeric,
    config="balanced",
    weights=(0.8, 0.1, 0.1),
    rf_trees=200,
    xgb_trees=300,
):
    if (
        config not in CONFIGS
        or len(weights) != 3
        or not np.isfinite(weights).all()
        or min(weights) < 0
        or not np.isclose(sum(weights), 1)
    ):
        raise ValueError("Invalid daily model configuration or blend weights.")
    if train.empty or test.empty or train.date.max() >= test.date.min():
        raise ValueError("Daily train and test periods must be disjoint and nonempty.")
    leaf, depth, alpha = CONFIGS[config]
    prep = ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(
                    strategy="median", keep_empty_features=True, add_indicator=True
                ),
                numeric,
            ),
            (
                "dow",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["dow"],
            ),
        ]
    )
    xt, xv = prep.fit_transform(train), prep.transform(test)
    names = prep.get_feature_names_out().tolist()
    xt, xv = (
        pd.DataFrame(xt, columns=names, index=train.index),
        pd.DataFrame(xv, columns=names, index=test.index),
    )
    for column in [c for c in names if c.startswith("dow__")]:
        for phase in ["sin1", "cos1"]:
            for frame in [xt, xv]:
                frame[f"{column}_{phase}"] = frame[column] * frame[f"num__{phase}"]
    rf = RandomForestRegressor(
        n_estimators=rf_trees,
        min_samples_leaf=leaf,
        max_depth=10,
        random_state=42,
        n_jobs=2,
    )
    xgb = XGBRegressor(
        n_estimators=xgb_trees,
        max_depth=depth,
        min_child_weight=5,
        reg_lambda=10,
        learning_rate=0.05,
        random_state=42,
        n_jobs=2,
    )
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    pred = test[["date", "orders"]].copy()
    pred["weekly_naive"] = test.lag_7
    for name, model in [("random_forest", rf), ("xgboost", xgb), ("ridge", ridge)]:
        model.fit(xt, train.orders)
        pred[name] = np.maximum(0, model.predict(xv))
    pred["ensemble"] = pred[
        ["random_forest", "xgboost", "ridge"]
    ].to_numpy() @ np.asarray(weights)
    metrics = pd.DataFrame(
        [
            {"model": name, **scores(pred.orders.to_numpy(), pred[name].to_numpy())}
            for name in MODELS
        ]
    )
    fitted = {
        "rf": rf,
        "xgb": xgb,
        "ridge": ridge,
        "preprocessor": prep,
        "x_train": xt,
        "x_test": xv,
        "ensemble_weights": list(weights),
        "clip_components": True,
    }
    return metrics, pred, fitted


def select_daily_ensemble(
    daily, cutoff, folds=3, validation_days=28, rf_trees=200, xgb_trees=300
):
    cutoff = pd.Timestamp(cutoff)
    history = daily.loc[pd.to_datetime(daily.date) < cutoff].copy()
    frame, numeric = daily_features(history)
    if len(frame) < folds * validation_days + 56:
        raise ValueError("Not enough historical days for daily validation.")
    candidates, pooled_by_config = [], {}
    for config in CONFIGS:
        parts = []
        for fold in range(folds):
            start = cutoff - pd.Timedelta(days=(folds - fold) * validation_days)
            end = start + pd.Timedelta(days=validation_days)
            train, val = (
                frame.loc[frame.date < start],
                frame.loc[frame.date.between(start, end, inclusive="left")],
            )
            _, pred, _ = fit_daily(
                train, val, numeric, config, rf_trees=rf_trees, xgb_trees=xgb_trees
            )
            pred["fold"] = fold + 1
            parts.append(pred)
        pooled = pd.concat(parts, ignore_index=True)
        pooled_by_config[config] = pooled
        for a in range(1, 9):
            for b in range(1, 10 - a):
                weights = np.array([a, b, 10 - a - b]) / 10
                predicted = (
                    pooled[["random_forest", "xgboost", "ridge"]].to_numpy() @ weights
                )
                candidates.append(
                    {
                        "config": config,
                        "rf_weight": weights[0],
                        "xgb_weight": weights[1],
                        "ridge_weight": weights[2],
                        "mse": float(
                            np.mean((pooled.orders.to_numpy() - predicted) ** 2)
                        ),
                        **scores(pooled.orders.to_numpy(), predicted),
                    }
                )
    search = (
        pd.DataFrame(candidates)
        .sort_values(["mse", "mae", "config", "rf_weight", "xgb_weight"])
        .reset_index(drop=True)
    )
    best = search.iloc[0]
    weights = [float(best.rf_weight), float(best.xgb_weight), float(best.ridge_weight)]
    pooled = pooled_by_config[best.config].copy()
    pooled["ensemble"] = pooled[
        ["random_forest", "xgboost", "ridge"]
    ].to_numpy() @ np.asarray(weights)
    rows = [
        {
            "fold": int(fold),
            "model": name,
            "validation_start": part.date.min(),
            "validation_end": part.date.max(),
            **scores(part.orders.to_numpy(), part[name].to_numpy()),
        }
        for fold, part in pooled.groupby("fold")
        for name in MODELS
    ]
    selected = {
        "config": best.config,
        "ensemble_weights": weights,
        "selection_metric": "Pooled pre-test MSE",
        "validation_start": str(pooled.date.min().date()),
        "validation_end": str(pooled.date.max().date()),
        "validation_rows": len(pooled),
        "validation_ensemble": scores(
            pooled.orders.to_numpy(), pooled.ensemble.to_numpy()
        ),
    }
    return selected, pd.DataFrame(rows), search, pooled


def run_daily_forecast(daily, cutoff, rf_trees=200, xgb_trees=300):
    selected, validation, search, validation_predictions = select_daily_ensemble(
        daily, cutoff, rf_trees=rf_trees, xgb_trees=xgb_trees
    )
    frame, numeric = daily_features(daily)
    metrics, predictions, fitted = fit_daily(
        frame.loc[frame.date < cutoff],
        frame.loc[frame.date >= cutoff],
        numeric,
        selected["config"],
        selected["ensemble_weights"],
        rf_trees,
        xgb_trees,
    )
    return (
        metrics,
        predictions,
        fitted,
        selected,
        validation,
        search,
        validation_predictions,
    )
