from pathlib import Path
import json
import numpy as np
import pandas as pd
from market_insights.pipeline import run_pipeline


def test_end_to_end_demo_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run = run_pipeline(root, "demo", tmp_path, rf_trees=12, xgb_trees=15)
    scores = pd.read_csv(tmp_path / "metrics/forecast_scores.csv")
    pred = pd.read_csv(tmp_path / "metrics/forecast_predictions.csv")
    assert set(scores.model) == {
        "naive",
        "rolling_4",
        "rolling_8",
        "rolling_13",
        "rolling_26",
        "random_forest",
        "xgboost",
        "ridge",
        "ensemble",
    }
    assert np.isfinite(scores.mae).all()
    weights = run["forecast"]["ensemble_weights"]
    np.testing.assert_allclose(
        pred.ensemble,
        weights[0] * pred.random_forest
        + weights[1] * pred.xgboost
        + weights[2] * pred.ridge,
    )
    assert pd.Timestamp(run["forecast"]["selection"]["validation_end"]) < pd.Timestamp(
        "2022-01-03"
    )
    assert (tmp_path / "metrics/ensemble_selection.csv").is_file()
    assert (tmp_path / "metrics/forecast_by_series.csv").is_file()
    assert run["weekly_quality"]["rows"] == 3744
    assert run["forecast"]["train_rows"] > 0 and run["forecast"]["test_rows"] > 0
    saved = json.loads((tmp_path / "metrics/run.json").read_text())
    assert saved["source_mode"] == "demo"
    assert (tmp_path / "figures/sentiment_confusion.png").stat().st_size > 1000
    assert (tmp_path / "figures/pricing_effects.png").is_file()
