from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import shap


@dataclass
class ShapExplainerArtifacts:
    explainer: shap.Explainer
    feature_names: List[str]


def build_tree_explainer(model, feature_names: List[str]) -> ShapExplainerArtifacts:
    explainer = shap.TreeExplainer(model)
    return ShapExplainerArtifacts(explainer=explainer, feature_names=feature_names)


def explain_single(
    artifacts: ShapExplainerArtifacts, features: Dict[str, float], top_k: int = 5
) -> List[Dict[str, float]]:
    row = np.array([[features[f] for f in artifacts.feature_names]], dtype=float)
    shap_values = artifacts.explainer.shap_values(row)

    # shap_values can be list (for multiclass) or array
    if isinstance(shap_values, list):
        shap_row = np.array(shap_values[0]).flatten()
    else:
        shap_row = np.array(shap_values).flatten()

    results = []
    for name, value, shap_val in zip(artifacts.feature_names, row[0], shap_row):
        results.append(
            {
                "feature": name,
                "value": float(value),
                "shap_value": float(shap_val.item()),
            }
        )

    # Sort by absolute shap value and return top_k
    results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return results[:top_k]

