"""Run from the repository root after installing requirements.txt."""

import argparse
import logging
from pathlib import Path
from market_insights.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Market Insights & Forecasting")
    parser.add_argument(
        "--source",
        choices=["real", "demo"],
        default="real",
        help="real: public UCI sales; demo: synthetic Week 2 panel",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Optional separate report directory"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Calculate ensemble SHAP; install the explain extra",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_pipeline(
        Path(__file__).resolve().parent, args.source, args.output_dir, args.explain
    )


if __name__ == "__main__":
    main()
