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
    assert set(scores.model) == {"naive", "random_forest", "xgboost", "ensemble"}
    assert np.isfinite(scores.mae).all()
    np.testing.assert_allclose(pred.ensemble, (pred.random_forest + pred.xgboost) / 2)
    assert run["weekly_quality"]["rows"] == 3744
    assert run["forecast"]["train_rows"] > 0 and run["forecast"]["test_rows"] > 0
    saved = json.loads((tmp_path / "metrics/run.json").read_text())
    assert saved["source_mode"] == "demo"
    assert (tmp_path / "figures/sentiment_confusion.png").stat().st_size > 1000
    assert (tmp_path / "figures/pricing_effects.png").is_file()
