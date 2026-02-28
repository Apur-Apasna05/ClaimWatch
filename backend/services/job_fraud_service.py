from __future__ import annotations

from typing import Dict, List

from backend.models.job_fraud_model import (
    JobFraudArtifacts,
    predict_job_proba,
    top_keywords,
)
from backend.schemas import ClaimInput, PredictionResponse, KeywordImportance
from backend.services.generative_reporting import generate_template_summary


def trust_score_from_prob(prob: float) -> float:
    # Simple inverse mapping: trust = 1 - fraud probability
    return float(1.0 - prob)


def predict_job_fraud(
    claim: ClaimInput,
    *,
    job_artifacts: JobFraudArtifacts,
) -> PredictionResponse:
    text = (claim.job_text or "").strip()
    if not text:
        raise ValueError("job_text is required for job_fraud predictions")

    prob = predict_job_proba(job_artifacts, text)
    trust = trust_score_from_prob(prob)

    keywords: List[KeywordImportance] = []
    for word, score in top_keywords(job_artifacts, text, top_k=10):
        keywords.append(KeywordImportance(keyword=word, score=score))

    # Anomaly score 0-10 aligned with fraud probability (so they match and don't contradict).
    anomaly_0_10 = round(float(prob) * 10.0, 2)
    anomaly_0_10 = max(0.0, min(10.0, anomaly_0_10))
    is_anomalous = prob >= 0.5

    # Simple persona for job fraud so we never return N/A.
    if prob >= 0.7:
        fraud_persona_label = "High risk – Likely fake posting"
    elif prob >= 0.4:
        fraud_persona_label = "Medium risk – Needs review"
    else:
        fraud_persona_label = "Low risk – Normal posting"

    summary, actions = generate_template_summary(
        fraud_probability=prob,
        anomaly_score=anomaly_0_10,
        top_features=[],
    )

    raw_features: Dict[str, str] = {"job_text": text}

    return PredictionResponse(
        fraud_type="job_fraud",
        fraud_probability=prob,
        trust_score=trust,
        anomaly_score=anomaly_0_10,
        is_anomalous=is_anomalous,
        fraud_persona=fraud_persona_label,
        top_features=[],
        important_keywords=keywords,
        summary=summary,
        recommended_actions=actions,
        raw_features=raw_features,
    )

