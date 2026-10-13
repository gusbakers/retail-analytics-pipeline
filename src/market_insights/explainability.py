"""Explain the fixed average of two fitted tree models."""

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
    values = (outputs[0].values + outputs[1].values) / 2
    base = (outputs[0].base_values + outputs[1].base_values) / 2
    predicted = (fitted["rf"].predict(test) + fitted["xgb"].predict(test)) / 2
    reconstructed = base + values.sum(axis=1)
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
            "reconstructed": reconstructed,
            "absolute_difference": np.abs(predicted - reconstructed),
        },
        index=test.index,
    )
    audit.to_csv(metric_dir / "shap_additivity.csv", index_label="feature_row_index")
    plot_shap(importance, scope, figure_dir)
    return {
        "sample_n": len(test),
        "reference": "training tree-path counts",
        "max_additivity_error": float(np.abs(predicted - reconstructed).max()),
    }
