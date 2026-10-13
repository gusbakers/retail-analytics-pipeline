"""Reconstruct the selected weighted ensemble with tree SHAP contributions."""

import numpy as np
import pandas as pd
from .visualization import plot_shap


def explain(fitted, scope, figure_dir, metric_dir):
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "Install explanations with: pip install -e '.[explain]'"
        ) from error
    test = fitted["x_test"].sample(n=min(80, len(fitted["x_test"])), random_state=42)
    outputs = []
    for key in ["rf", "xgb"]:
        explainer = shap.TreeExplainer(
            fitted[key], feature_perturbation="tree_path_dependent", model_output="raw"
        )
        outputs.append(explainer(test))
    rf_weight, xgb_weight, ridge_weight = fitted["ensemble_weights"]
    values = rf_weight * outputs[0].values + xgb_weight * outputs[1].values
    base = rf_weight * outputs[0].base_values + xgb_weight * outputs[1].base_values
    linear = fitted["ridge"].named_steps["ridge"]
    scaler = fitted["ridge"].named_steps["standardscaler"]
    coefficients = linear.coef_ / scaler.scale_
    reference = fitted["x_train"].mean().to_numpy()
    linear_base = (
        linear.intercept_ - coefficients @ scaler.mean_ + coefficients @ reference
    )
    linear_values = (test.to_numpy() - reference) * coefficients
    values += ridge_weight * linear_values
    base += ridge_weight * linear_base
    predicted = (
        rf_weight * fitted["rf"].predict(test)
        + xgb_weight * fitted["xgb"].predict(test)
        + ridge_weight * fitted["ridge"].predict(test)
    )
    adjustment = np.zeros(len(test))
    if fitted.get("clip_components", False):
        for weight, key in zip(
            [rf_weight, xgb_weight, ridge_weight], ["rf", "xgb", "ridge"]
        ):
            raw = fitted[key].predict(test)
            adjustment += weight * (np.maximum(0, raw) - raw)
    predicted += adjustment
    reconstructed = base + values.sum(axis=1) + adjustment
    if not np.allclose(predicted, reconstructed, atol=0.05, rtol=1e-4):
        raise AssertionError("SHAP contributions do not reconstruct the ensemble.")
    importance = pd.DataFrame(
        {"feature": test.columns, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(metric_dir / "shap_importance.csv", index=False)
    audit = pd.DataFrame(
        {
            "prediction": predicted,
            "base": base,
            "nonnegative_adjustment": adjustment,
            "reconstructed": reconstructed,
            "absolute_difference": np.abs(predicted - reconstructed),
        },
        index=test.index,
    )
    audit.to_csv(metric_dir / "shap_additivity.csv", index_label="feature_row_index")
    plot_shap(importance, scope, figure_dir)
    return {
        "sample_n": len(test),
        "reference": "training tree-path counts for trees; training feature means for linear terms",
        "ensemble_weights": [rf_weight, xgb_weight, ridge_weight],
        "postprocessing": "Component forecasts floored at zero; exact correction exported separately"
        if fitted.get("clip_components", False)
        else "None",
        "max_additivity_error": float(np.abs(predicted - reconstructed).max()),
    }
