"""Fixed RF/XGBoost/ensemble comparison and temporal backtests."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from .features import build_features, temporal_split


def scores(actual, predicted) -> dict:
    denominator = np.abs(actual).sum()
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "wape": float(np.abs(actual - predicted).sum() / denominator)
        if denominator
        else None,
    }


def fit_forecasters(train, test, numeric, rf_trees=200, xgb_trees=300):
    categorical = ["country", "segment"]
    prep = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median")), numeric),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    xt = prep.fit_transform(train)
    xv = prep.transform(test)
    names = prep.get_feature_names_out().tolist()
    # DataFrame preserves names for XGBoost and explanation artifacts.
    xt = pd.DataFrame(xt, columns=names, index=train.index)
    xv = pd.DataFrame(xv, columns=names, index=test.index)
    rf = RandomForestRegressor(
        n_estimators=rf_trees, max_depth=12, random_state=42, n_jobs=2
    )
    xgb = XGBRegressor(
        n_estimators=xgb_trees,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    rf.fit(xt, train.units_sold)
    xgb.fit(xt, train.units_sold)
    pred = test[["week_start", "country", "segment", "units_sold"]].copy()
    pred["naive"] = test.lag_1
    pred["random_forest"] = rf.predict(xv)
    pred["xgboost"] = xgb.predict(xv)
    pred["ensemble"] = (pred.random_forest + pred.xgboost) / 2
    metrics = pd.DataFrame(
        [
            {"model": name, **scores(pred.units_sold.to_numpy(), pred[name].to_numpy())}
            for name in ["naive", "random_forest", "xgboost", "ensemble"]
        ]
    )
    fitted = {
        "preprocessor": prep,
        "rf": rf,
        "xgb": xgb,
        "x_train": xt,
        "x_test": xv,
        "train": train,
        "test": test,
    }
    return metrics, pred, fitted


def backtest(
    weekly, test_cutoff, folds=3, validation_weeks=8, rf_trees=200, xgb_trees=300
):
    """Only dates before the final holdout enter any training or validation fold."""
    weeks = np.sort(pd.to_datetime(weekly.week_start.unique()))
    history = weeks[weeks < np.datetime64(pd.Timestamp(test_cutoff))]
    if len(history) < (folds * validation_weeks) + 24:
        raise ValueError("Not enough pre-test weeks for the configured backtest.")
    rows = []
    for fold in range(folds):
        start = len(history) - (folds - fold) * validation_weeks
        cutoff = pd.Timestamp(history[start])
        end = pd.Timestamp(history[start + validation_weeks - 1])
        frame, numeric = build_features(
            weekly.loc[pd.to_datetime(weekly.week_start) <= end], cutoff
        )
        train, validation = temporal_split(frame, cutoff)
        metrics, _, _ = fit_forecasters(train, validation, numeric, rf_trees, xgb_trees)
        metrics["fold"] = fold + 1
        metrics["validation_start"] = cutoff
        metrics["validation_end"] = end
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)
