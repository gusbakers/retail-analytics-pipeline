from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pytest
from market_insights.data import simulate_pricing, simulate_feedback, demo_demand
from market_insights.quality import clean_transactions
from market_insights.pricing import evaluate_pricing
from market_insights.sentiment import clean_text, evaluate_sentiment
from market_insights.queries import export_queries


def test_sql_day_and_hour_match_calendar(transactions, tmp_path):
    clean, _ = clean_transactions(transactions)
    root = Path(__file__).resolve().parents[1]
    results = export_queries(
        clean, tmp_path / "sales.db", root / "sql", tmp_path / "metrics"
    )
    days = results["03_orders_by_dow"].set_index("day_name")
    assert days.loc["Mon", "total_orders"] == 2
    assert days.loc["Tue", "total_orders"] == 1
    hours = results["04_orders_by_hour"].set_index("order_hour")
    assert hours.loc[10, "total_orders"] == 2
    assert results["01_top_products"].units_sold.sum() == 6
    assert results["05_dow_x_hour_heatmap"].total_orders.sum() == 3
    with duckdb.connect(str(tmp_path / "sales.db"), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sales_clean").fetchone()[0] == 3


def test_scenarios_have_reproducible_independent_units():
    experiment, costs = simulate_pricing()
    repeated, _ = simulate_pricing()
    pd.testing.assert_frame_equal(experiment, repeated)
    assert experiment.experimental_unit_id.is_unique
    assert experiment.group.value_counts().to_dict() == {
        "control": 400,
        "treatment": 400,
    }
    result, _ = evaluate_pricing(experiment, costs)
    assert np.isfinite(result.difference).all()
    assert (result.p_holm >= result.p_value).all()
    assert result.p_holm.between(0, 1).all()
    assert (result.ci95_low <= result.ci95_high).all()


def test_missing_cost_is_rejected():
    experiment, costs = simulate_pricing()
    with pytest.raises(ValueError, match="Missing unit cost"):
        evaluate_pricing(experiment, costs.iloc[:-1])


def test_sentiment_keeps_negation_and_checks_unseen_vocabulary():
    assert clean_text("No, NO me gustó!") == "no no me gustó"
    rows = []
    for family in range(30):
        label = ["positive", "negative", "neutral"][family % 3]
        for i in range(4):
            rows.append(
                {
                    "feedback_id": f"{family}-{i}",
                    "template_id": f"family{family}",
                    "review_text": f"uniqueword{family} generic review detail{i}",
                    "sentiment_label": label,
                }
            )
    result, heldout, cm, model = evaluate_sentiment(pd.DataFrame(rows))
    vocabulary = model.named_steps["tfidfvectorizer"].vocabulary_
    assert all(
        f"uniqueword{int(family.removeprefix('family'))}" not in vocabulary
        for family in heldout.template_id.unique()
    )
    assert cm.sum() == len(heldout) and len(result) == 6


def test_original_demand_generator_is_preserved():
    data = demo_demand()
    assert data.shape == (3744, 11)
    assert data.competitor_price_index.isna().sum() == 56
    assert data.week_start.nunique() == 104
    feedback = simulate_feedback()
    assert len(feedback) == 650
    assert feedback.sentiment_label.value_counts().to_dict() == {
        "positive": 273,
        "negative": 215,
        "neutral": 162,
    }
