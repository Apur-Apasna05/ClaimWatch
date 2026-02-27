from __future__ import annotations

from typing import Dict

from fastapi import HTTPException

from backend.models.anomaly_model import AnomalyModelArtifacts, anomaly_score
from backend.models.fraud_model import FraudModelArtifacts, FEATURE_COLUMNS
from backend.schemas import ClaimInput, PredictionResponse, FeatureImportance
from backend.services.explainability import ShapExplainerArtifacts, explain_single
from backend.services.fraud_persona import classify_fraud_persona
from backend.services.generative_reporting import generate_template_summary


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

    try:
        a_score = anomaly_score(anomaly_artifacts, features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly scoring failed: {exc}")

    is_anomalous = a_score < -0.2

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
        fraud_probability=fraud_prob,
        anomaly_score=a_score,
        features=features,
    )

    summary, actions = generate_template_summary(
        fraud_probability=fraud_prob,
        anomaly_score=a_score,
        top_features=top_features_dicts,
    )

    return PredictionResponse(
        fraud_type="insurance",
        fraud_probability=fraud_prob,
        trust_score=float(1.0 - fraud_prob),
        anomaly_score=a_score,
        is_anomalous=bool(is_anomalous),
        fraud_persona=persona.label,
        top_features=top_features,
        important_keywords=[],
        summary=summary,
        recommended_actions=actions,
        raw_features=features,
    )

