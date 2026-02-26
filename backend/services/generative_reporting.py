from __future__ import annotations

from typing import Dict, List


def generate_template_summary(
    fraud_probability: float,
    anomaly_score: float,
    top_features: List[Dict[str, float]],
) -> tuple[str, List[str]]:
    """
    Lightweight, deterministic "generative-style" summary.
    This avoids external dependencies while still giving a
    natural-language investigation summary and next steps.
    """
    risk_level = (
        "HIGH"
        if fraud_probability >= 0.8
        else "MEDIUM"
        if fraud_probability >= 0.4
        else "LOW"
    )

    summary_lines: List[str] = []
    summary_lines.append(
        f"The claim is assessed as {risk_level} fraud risk "
        f"with an estimated fraud probability of {fraud_probability:.2f}."
    )
    summary_lines.append(
        f"The anomaly detector returned a score of {anomaly_score:.3f}, "
        "where lower scores indicate more unusual patterns."
    )

    if top_features:
        feature_descriptions = []
        for f in top_features:
            direction = "increases" if f["shap_value"] > 0 else "decreases"
            feature_descriptions.append(
                f"{f['feature']} (value={f['value']:.2f}, impact={direction} fraud risk)"
            )
        joined = "; ".join(feature_descriptions)
        summary_lines.append(
            f"Key factors influencing the decision include: {joined}."
        )

    summary = " ".join(summary_lines)

    actions: List[str] = []
    if risk_level == "HIGH":
        actions.append("Escalate to manual investigation before approval.")
        actions.append("Verify customer identity and policy history.")
        actions.append("Request supporting documents (invoices, medical reports, police reports).")
    elif risk_level == "MEDIUM":
        actions.append("Perform targeted checks on the highest-impact risk factors.")
        actions.append("Cross-check claim details against prior claim history.")
    else:
        actions.append("Proceed with standard automated checks.")
        actions.append("Spot-audit a random sample of low-risk claims for quality control.")

    return summary, actions

