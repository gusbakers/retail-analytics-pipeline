"""Temporal model selection and an ensemble fitted without final-test tuning."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from .features import build_features, temporal_split

CONFIGS = {
    "legacy": {"rf": {"max_depth": 12}, "xgb": {"max_depth": 6}},
    "balanced": {
        "rf": {"max_depth": 10, "min_samples_leaf": 4},
        "xgb": {"max_depth": 3, "min_child_weight": 5, "reg_lambda": 10},
    },
    "smooth": {
        "rf": {"max_depth": 8, "min_samples_leaf": 12, "max_features": 0.8},
        "xgb": {"max_depth": 2, "min_child_weight": 10, "reg_lambda": 20},
    },
    "conservative": {
        "rf": {"max_depth": 6, "min_samples_leaf": 24, "max_features": 0.8},
        "xgb": {"max_depth": 2, "min_child_weight": 20, "reg_lambda": 40},
    },
}
MODEL_NAMES = [
    "naive",
    "rolling_4",
    "rolling_8",
    "rolling_13",
    "rolling_26",
    "random_forest",
    "xgboost",
    "ridge",
    "ensemble",
]


def scores(actual, predicted) -> dict:
    actual, predicted = (
        np.asarray(actual, dtype=float),
        np.asarray(predicted, dtype=float),
    )
    if actual.ndim != 1 or actual.shape != predicted.shape or len(actual) < 2:
        raise ValueError(
            "Scores require matching one-dimensional arrays with at least two rows."
        )
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actuals and predictions must be finite.")
    denominator = np.abs(actual).sum()
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if np.ptp(actual) > 0 else None,
        "wape": float(np.abs(actual - predicted).sum() / denominator)
        if denominator
        else None,
    }


def fit_forecasters(
    train,
    test,
    numeric,
    rf_trees=200,
    xgb_trees=300,
    config="legacy",
    rf_weight=0.5,
    weights=None,
):
    if config not in CONFIGS or not 0 <= rf_weight <= 1:
        raise ValueError("Unknown model configuration or invalid ensemble weight.")
    categorical = ["country", "segment", "series_key"]
    train, test = train.copy(), test.copy()
    for data in [train, test]:
        data["series_key"] = data.country.astype(str) + "/" + data.segment.astype(str)
    prep = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(
                        strategy="median", keep_empty_features=True, add_indicator=True
                    )
                ),
                numeric,
            ),
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
    # Each country/SKU can have its own seasonal amplitude and phase.
    # Calendar values are known before forecasting; dummy columns fit only train.
    for column in [c for c in names if c.startswith("category__series_key_")]:
        for phase in ["sin", "cos"]:
            name = f"seasonal__{column.removeprefix('category__')}_{phase}"
            for data in [xt, xv]:
                data[name] = data[column] * data[f"numeric__week_{phase}"]
    rf = RandomForestRegressor(
        n_estimators=rf_trees, random_state=42, n_jobs=2, **CONFIGS[config]["rf"]
    )
    xgb = XGBRegressor(
        n_estimators=xgb_trees,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
        **CONFIGS[config]["xgb"],
    )
    rf.fit(xt, train.units_sold)
    xgb.fit(xt, train.units_sold)
    ridge = make_pipeline(
        StandardScaler(),
        Ridge(
            alpha={"legacy": 30, "balanced": 10, "smooth": 30, "conservative": 100}[
                config
            ]
        ),
    )
    ridge.fit(xt, train.units_sold)
    pred = test[["week_start", "country", "segment", "units_sold"]].copy()
    pred["naive"] = test.lag_1
    for width in [4, 8, 13, 26]:
        column = f"rolling_avg_{width}wk"
        pred[f"rolling_{width}"] = test[column]
    pred["random_forest"] = np.maximum(0, rf.predict(xv))
    pred["xgboost"] = np.maximum(0, xgb.predict(xv))
    pred["ridge"] = np.maximum(0, ridge.predict(xv))
    weights = list(weights) if weights is not None else [rf_weight, 1 - rf_weight, 0.0]
    if (
        len(weights) != 3
        or not np.isfinite(weights).all()
        or min(weights) < 0
        or not np.isclose(sum(weights), 1)
    ):
        raise ValueError("Three nonnegative ensemble weights must sum to one.")
    pred["ensemble"] = pred[
        ["random_forest", "xgboost", "ridge"]
    ].to_numpy() @ np.asarray(weights)
    metrics = pd.DataFrame(
        [
            {"model": name, **scores(pred.units_sold.to_numpy(), pred[name].to_numpy())}
            for name in MODEL_NAMES
        ]
    )
    fitted = {
        "preprocessor": prep,
        "rf": rf,
        "xgb": xgb,
        "ridge": ridge,
        "x_train": xt,
        "x_test": xv,
        "train": train,
        "test": test,
        "config": config,
        "ensemble_weights": [float(w) for w in weights],
        "clip_components": True,
    }
    return metrics, pred, fitted


def backtest(
    weekly,
    test_cutoff,
    folds=3,
    validation_weeks=8,
    rf_trees=200,
    xgb_trees=300,
    config="legacy",
    rf_weight=0.5,
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
        metrics, _, _ = fit_forecasters(
            train, validation, numeric, rf_trees, xgb_trees, config, rf_weight
        )
        metrics["fold"] = fold + 1
        metrics["validation_start"] = cutoff
        metrics["validation_end"] = end
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def select_ensemble(
    weekly, test_cutoff, folds=3, validation_weeks=8, rf_trees=200, xgb_trees=300
):
    """Select regularization and blend weights by pooled pre-test squared error.

    The final holdout is never used here. Both components retain positive weight.
    Selection scores are development scores; they are not an independent test.
    """
    cutoff = pd.Timestamp(test_cutoff)
    history = weekly.loc[pd.to_datetime(weekly.week_start) < cutoff].copy()
    history["week_start"] = pd.to_datetime(history.week_start)
    weeks = np.sort(pd.to_datetime(history.week_start.unique()))
    if len(weeks) < folds * validation_weeks + 24:
        raise ValueError("Not enough pre-test weeks for ensemble selection.")
    candidates, predictions_by_config = [], {}
    for config in ["balanced", "smooth", "conservative"]:
        pieces = []
        for fold in range(folds):
            start = len(weeks) - (folds - fold) * validation_weeks
            origin, end = (
                pd.Timestamp(weeks[start]),
                pd.Timestamp(weeks[start + validation_weeks - 1]),
            )
            frame, numeric = build_features(
                history.loc[history.week_start <= end], origin
            )
            train, validation = temporal_split(frame, origin)
            _, pred, _ = fit_forecasters(
                train, validation, numeric, rf_trees, xgb_trees, config
            )
            pred["fold"] = fold + 1
            pieces.append(pred)
        pooled = pd.concat(pieces, ignore_index=True)
        predictions_by_config[config] = pooled
        weight_grid = [(float(w), float(1 - w), 0.0) for w in np.linspace(0.1, 0.9, 9)]
        weight_grid += [
            (a / 10, b / 10, (10 - a - b) / 10)
            for a in range(1, 9)
            for b in range(1, 10 - a)
        ]
        for weight, xgb_weight, ridge_weight in weight_grid:
            prediction = (
                weight * pooled.random_forest
                + xgb_weight * pooled.xgboost
                + ridge_weight * pooled.ridge
            )
            actual = pooled.units_sold.to_numpy()
            candidates.append(
                {
                    "config": config,
                    "rf_weight": weight,
                    "xgb_weight": xgb_weight,
                    "ridge_weight": ridge_weight,
                    "mse": float(np.mean((actual - prediction.to_numpy()) ** 2)),
                    **scores(actual, prediction.to_numpy()),
                }
            )
    search = (
        pd.DataFrame(candidates)
        .sort_values(["mse", "mae", "config", "rf_weight"])
        .reset_index(drop=True)
    )
    best = search.iloc[0]
    pooled = predictions_by_config[best.config].copy()
    pooled["ensemble"] = (
        best.rf_weight * pooled.random_forest
        + best.xgb_weight * pooled.xgboost
        + best.ridge_weight * pooled.ridge
    )
    rows = []
    for fold, part in pooled.groupby("fold"):
        for name in MODEL_NAMES:
            rows.append(
                {
                    "fold": int(fold),
                    "model": name,
                    "validation_start": part.week_start.min(),
                    "validation_end": part.week_start.max(),
                    **scores(part.units_sold.to_numpy(), part[name].to_numpy()),
                }
            )
    baseline_scores = {
        name: scores(pooled.units_sold.to_numpy(), pooled[name].to_numpy())
        for name in MODEL_NAMES[:5]
    }
    best_baseline = min(baseline_scores, key=lambda name: baseline_scores[name]["mae"])
    selected = {
        "config": best.config,
        "ensemble_weights": [
            float(best.rf_weight),
            float(best.xgb_weight),
            float(best.ridge_weight),
        ],
        "selection_metric": "pooled pre-test MSE",
        "validation_rows": len(pooled),
        "validation_start": str(pooled.week_start.min().date()),
        "validation_end": str(pooled.week_start.max().date()),
        "best_validation_baseline": best_baseline,
        "validation_ensemble": scores(
            pooled.units_sold.to_numpy(), pooled.ensemble.to_numpy()
        ),
    }
    return selected, pd.DataFrame(rows), search, pooled


def series_scores(predictions):
    rows = []
    for (country, segment), part in predictions.groupby(["country", "segment"]):
        for name in MODEL_NAMES:
            rows.append(
                {
                    "country": country,
                    "segment": segment,
                    "model": name,
                    "rows": len(part),
                    **scores(part.units_sold.to_numpy(), part[name].to_numpy()),
                }
            )
    return pd.DataFrame(rows)
