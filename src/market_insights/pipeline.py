"""End-to-end execution with source-specific artifacts and run provenance."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import logging
import pandas as pd
from .data import (
    load_transactions,
    real_weekly,
    demo_demand,
    simulate_pricing,
    simulate_feedback,
)
from .quality import clean_transactions, validate_weekly
from .features import build_features, temporal_split
from .forecasting import fit_forecasters, backtest
from .pricing import evaluate_pricing
from .sentiment import evaluate_sentiment
from .queries import export_queries
from .visualization import (
    plot_weekly,
    plot_forecast,
    plot_models,
    plot_errors,
    plot_sql,
    plot_pricing,
    plot_sentiment,
)

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    root: Path,
    source="real",
    output_dir=None,
    with_explanations=False,
    rf_trees=200,
    xgb_trees=300,
):
    root = Path(root).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir
        else root / "reports" / ("demo" if source == "demo" else "")
    )
    figures = output / "figures"
    metrics = output / "metrics"
    for folder in [figures, metrics, root / "data/processed"]:
        folder.mkdir(parents=True, exist_ok=True)
    LOGGER.info("1/6 Data and quality (%s)", source)
    metadata = {
        "source_mode": source,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if source == "real":
        raw, provenance = load_transactions(root / "data/raw")
        clean, audit = clean_transactions(raw)
        weekly, cutoff, scope = real_weekly(clean)
        metadata.update(
            {
                "sales_source": provenance,
                "transaction_quality": audit,
                "forecast_scope": scope,
            }
        )
        pd.DataFrame([audit]).to_csv(metrics / "transaction_quality.csv", index=False)
        # Real fixture, separate from the legacy grocery demo, with UCI attribution.
        sample = root / "data/samples/uci_transactions_sample.csv"
        if not sample.exists():
            raw.sample(n=min(300, len(raw)), random_state=42).to_csv(
                sample, index=False
            )
        queries = export_queries(
            clean, root / "data/processed/retail.duckdb", root / "sql", metrics
        )
        label = "UCI Online Retail II | United Kingdom | 12 training-selected SKUs"
        plot_sql(
            queries,
            "UCI Online Retail II | retained product sales | 2009-2011",
            figures,
        )
        del raw, clean
    elif source == "demo":
        weekly = demo_demand()
        weekly["week_start"] = pd.to_datetime(weekly.week_start)
        cutoff = pd.Timestamp("2022-01-03")
        scope = {
            "example_series": "Beverages",
            "cutoff": str(cutoff.date()),
            "selection": "All 36 synthetic country/category series",
            "target": "Simulated units",
            "covariate_assumption": "Demo promotion, holiday and market indices known at prediction time",
        }
        metadata.update(
            {
                "sales_source": {
                    "source": "Week 2 demand simulation",
                    "seed": 42,
                    "currency": "USD",
                },
                "forecast_scope": scope,
            }
        )
        label = "SIMULATION | 36 country/category series | Week 2 generator"
    else:
        raise ValueError("source must be 'real' or 'demo'")
    metadata["weekly_quality"] = validate_weekly(weekly)
    weekly.to_csv(root / "data/processed" / f"weekly_{source}.csv", index=False)
    metadata["weekly_csv_sha256"] = hashlib.sha256(
        weekly.to_csv(index=False).encode()
    ).hexdigest()
    LOGGER.info("2/6 Features and temporal validation")
    frame, numeric = build_features(weekly, cutoff)
    train, test = temporal_split(frame, cutoff)
    validation = backtest(weekly, cutoff, rf_trees=rf_trees, xgb_trees=xgb_trees)
    validation.to_csv(metrics / "temporal_backtest.csv", index=False)
    LOGGER.info("3/6 Final forecast evaluation")
    model_scores, predictions, fitted = fit_forecasters(
        train, test, numeric, rf_trees, xgb_trees
    )
    model_scores.to_csv(metrics / "forecast_scores.csv", index=False)
    predictions.to_csv(metrics / "forecast_predictions.csv", index=False)
    metadata["forecast"] = {
        "train_rows": len(train),
        "test_rows": len(test),
        "numeric_features": numeric,
        "high_demand_threshold": frame.attrs["high_demand_threshold"],
        "rf_trees": rf_trees,
        "xgb_trees": xgb_trees,
        "ensemble_weights": [0.5, 0.5],
        "horizon": "one week, observed history updated between predictions",
    }
    plot_weekly(weekly, cutoff, label, figures)
    # In real mode the example SKU is chosen using training volume.
    plot_forecast(predictions, scope["example_series"], label, figures)
    plot_models(model_scores, label, figures)
    plot_errors(predictions, label, figures)
    LOGGER.info("4/6 Pricing simulation")
    experiment, costs = simulate_pricing()
    pricing, experiment_enriched = evaluate_pricing(experiment, costs)
    pricing.to_csv(metrics / "pricing_simulation.csv", index=False)
    experiment_enriched.to_csv(
        root / "data/processed/pricing_simulation.csv", index=False
    )
    plot_pricing(pricing, figures)
    LOGGER.info("5/6 Feedback simulation and unseen-template evaluation")
    feedback = simulate_feedback()
    sentiment, heldout, matrix, _ = evaluate_sentiment(feedback)
    feedback.to_csv(root / "data/processed/feedback_simulation.csv", index=False)
    sentiment.to_csv(metrics / "sentiment_simulation.csv", index=False)
    heldout.to_csv(metrics / "sentiment_heldout_predictions.csv", index=False)
    plot_sentiment(sentiment, matrix, figures)
    metadata["simulations"] = {
        "pricing": {
            "seed": 2427,
            "rows": len(experiment),
            "cost_assumption": "45% of regular unit price",
            "currency": "USD",
        },
        "feedback": {
            "seed": 2240,
            "rows": len(feedback),
            "template_families": int(feedback.template_id.nunique()),
            "evaluation": "random stratified and group-held-out templates",
        },
    }
    if with_explanations:
        LOGGER.info("6/6 Ensemble SHAP")
        from .explainability import explain

        metadata["shap"] = explain(fitted, label, figures, metrics)
    else:
        LOGGER.info("6/6 SHAP skipped; use --explain to generate it")
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "openpyxl",
        "duckdb",
        "xgboost-cpu",
        "xgboost",
        "shap",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata["package_versions"] = versions
    (metrics / "run.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Complete. Figures: %s | metrics: %s", figures, metrics)
    return metadata
