from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class FraudPersona:
    """Simple container for persona label and an internal code."""

    code: str
    label: str


def classify_fraud_persona(
    *,
    fraud_probability: float,
    anomaly_score: float,
    features: Dict[str, float],
) -> FraudPersona:
    """
    Rule-based fraud persona classification.

    Uses a combination of model outputs and core features:
    - fraud_probability
    - anomaly_score
    - claim_amount
    - policy_tenure_days
    - num_prior_claims

    The rules are deliberately simple and easy to extend.
    """
    claim_amount = float(features.get("claim_amount", 0.0))
    policy_tenure = float(features.get("policy_tenure_days", 0.0))
    num_prior = float(features.get("num_prior_claims", 0.0))

    is_high_risk = fraud_probability >= 0.8
    is_medium_risk = 0.4 <= fraud_probability < 0.8
    is_low_risk = fraud_probability < 0.4

    # anomaly_score is 0-10 (higher = more anomalous)
    is_strong_anomaly = anomaly_score >= 7.0

    # 1. Repeat offender pattern: very high fraud risk and many prior claims.
    if is_high_risk and num_prior >= 3:
        return FraudPersona(
            code="repeat_offender",
            label="Repeat Offender Pattern",
        )

    # 2. Policy manipulation risk: new/young policy with elevated risk or anomaly.
    if (is_high_risk or is_medium_risk) and policy_tenure < 60 and (num_prior <= 1):
        return FraudPersona(
            code="policy_manipulation",
            label="Policy Manipulation Risk",
        )

    # 3. Opportunistic high-value claim: very high amount with at least medium risk.
    if (is_high_risk or is_medium_risk) and claim_amount >= 25000:
        return FraudPersona(
            code="opportunistic_high_value",
            label="Opportunistic High-Value Claim",
        )

    # 4. Financial distress pattern: several prior claims and moderate risk/anomaly.
    if is_medium_risk and (num_prior >= 2 or is_strong_anomaly):
        return FraudPersona(
            code="financial_distress",
            label="Financial Distress Pattern",
        )

    # 5. Low risk – normal behavior: low risk and not strongly anomalous.
    if is_low_risk and not is_strong_anomaly and num_prior <= 1 and claim_amount < 20000:
        return FraudPersona(
            code="low_risk_normal",
            label="Low Risk – Normal Behavior",
        )

    # Fallback bucket for anything not covered explicitly.
    return FraudPersona(
        code="needs_review",
        label="Needs Analyst Review",
    )

