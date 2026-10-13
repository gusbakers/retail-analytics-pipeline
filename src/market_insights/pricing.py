"""Welch intervals and Holm correction for a simulated pricing experiment."""

import numpy as np
import pandas as pd
from scipy import stats


def evaluate_pricing(experiment: pd.DataFrame, costs: pd.DataFrame):
    if not costs.product_category.is_unique:
        raise ValueError("Cost table must have unique category keys.")
    data = experiment.merge(
        costs, on="product_category", how="left", validate="many_to_one"
    )
    if data.unit_cost_usd.isna().any():
        raise ValueError("Missing unit cost for an experimental category.")
    if not data.experimental_unit_id.is_unique or set(data.group) != {
        "control",
        "treatment",
    }:
        raise ValueError("Expected independent, unique units in two groups.")
    data["margin_usd"] = data.revenue_usd - data.units_sold * data.unit_cost_usd
    control, treatment = [data.loc[data.group.eq(g)] for g in ["control", "treatment"]]
    rows = []
    for metric in ["revenue_usd", "units_sold", "margin_usd"]:
        result = stats.ttest_ind(treatment[metric], control[metric], equal_var=False)
        interval = result.confidence_interval()
        mean = control[metric].mean()
        difference = treatment[metric].mean() - mean
        rows.append(
            {
                "metric": metric,
                "control_mean": mean,
                "treatment_mean": treatment[metric].mean(),
                "difference": difference,
                "lift_pct": difference / mean * 100,
                "ci95_low": interval.low,
                "ci95_high": interval.high,
                "p_value": result.pvalue,
            }
        )
    results = pd.DataFrame(rows)
    order = np.argsort(results.p_value.to_numpy())
    adjusted = np.minimum(
        1,
        np.maximum.accumulate(
            (len(order) - np.arange(len(order))) * results.p_value.to_numpy()[order]
        ),
    )
    corrected = np.empty_like(adjusted)
    corrected[order] = adjusted
    results["p_holm"] = corrected
    return results, data
