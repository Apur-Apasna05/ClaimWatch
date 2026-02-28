from __future__ import annotations

from typing import Dict

from fastapi import HTTPException

from backend.models.anomaly_model import AnomalyModelArtifacts, anomaly_score
from backend.models.fraud_model import FraudModelArtifacts, FEATURE_COLUMNS
from backend.schemas import ClaimInput, PredictionResponse, FeatureImportance
from backend.services.explainability import ShapExplainerArtifacts, explain_single
from backend.services.fraud_persona import classify_fraud_persona
from backend.services.generative_reporting import generate_template_summary
from backend.services.risk_fusion import (
    fuse_risk,
    normalize_anomaly_to_unit,
)


def predict_insurance(
    claim: ClaimInput,
    *,
    fraud_artifacts: FraudModelArtifacts,
    anomaly_artifacts: AnomalyModelArtifacts,
    shap_artifacts: ShapExplainerArtifacts,
) -> PredictionResponse:
    features: Dict[str, float] = {
        col: float(getattr(claim, col))
        for col in FEATURE_COLUMNS
    }

    try:
        fraud_prob = fraud_artifacts.model.predict_proba(
            [[features[c] for c in FEATURE_COLUMNS]]
        )[0, 1]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fraud prediction failed: {exc}")

    fraud_prob = max(0.0, min(1.0, float(fraud_prob)))

    try:
        raw_score = anomaly_score(anomaly_artifacts, features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly scoring failed: {exc}")

    if not isinstance(raw_score, (int, float)) or (raw_score != raw_score):
        raw_score = 0.0
    raw_score = float(raw_score)

    # Normalize anomaly to [0, 1] using training-time bounds (or defaults)
    bounds = getattr(anomaly_artifacts, "score_bounds", None)
    norm_anomaly = normalize_anomaly_to_unit(raw_score, bounds=bounds)

    # Fused risk: single calibrated score, monotonic in both signals
    fused_risk_score = fuse_risk(
        fraud_probability=fraud_prob,
        norm_anomaly=norm_anomaly,
        method="convex",
        alpha=0.65,
    )

    # Downstream logic uses fused risk so decisions are consistent
    is_anomalous = fused_risk_score >= 0.5
    # Display anomaly on 0–10 scale for UI
    anomaly_display_0_10 = round(norm_anomaly * 10.0, 2)

    top_features_dicts = explain_single(shap_artifacts, features, top_k=5)
    top_features = [
        FeatureImportance(
            feature=f["feature"],
            value=f["value"],
            shap_value=f["shap_value"],
        )
        for f in top_features_dicts
    ]

    persona = classify_fraud_persona(
        fraud_probability=fused_risk_score,
        anomaly_score=anomaly_display_0_10,
        features=features,
    )

    summary, actions = generate_template_summary(
        fraud_probability=fused_risk_score,
        anomaly_score=anomaly_display_0_10,
        top_features=top_features_dicts,
    )

    return PredictionResponse(
        fraud_type="insurance",
        fraud_probability=fraud_prob,
        fused_risk=round(fused_risk_score, 4),
        trust_score=float(1.0 - fused_risk_score),
        anomaly_score=anomaly_display_0_10,
        is_anomalous=bool(is_anomalous),
        fraud_persona=persona.label,
        top_features=top_features,
        important_keywords=[],
        summary=summary,
        recommended_actions=actions,
        raw_features=features,
    )

