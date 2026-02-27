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

    # Reuse the same summary generator with no anomaly score / features.
    summary, actions = generate_template_summary(
        fraud_probability=prob,
        anomaly_score=0.0,
        top_features=[],
    )

    raw_features: Dict[str, str] = {"job_text": text}

    return PredictionResponse(
        fraud_type="job_fraud",
        fraud_probability=prob,
        trust_score=trust,
        anomaly_score=None,
        is_anomalous=None,
        fraud_persona=None,
        top_features=[],
        important_keywords=keywords,
        summary=summary,
        recommended_actions=actions,
        raw_features=raw_features,
    )

