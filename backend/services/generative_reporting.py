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

    # Keep summary to 2–3 short sentences.
    summary_lines: List[str] = []
    summary_lines.append(
        f"Overall this is assessed as {risk_level} fraud risk "
        f"(estimated fraud probability {fraud_probability:.2f})."
    )

    # Anomaly score is 0-10 (higher = more anomalous).
    if anomaly_score is not None and anomaly_score > 0:
        summary_lines.append(
            f"Anomaly score is {anomaly_score:.1f}/10 (higher means more unusual behaviour)."
        )

    if top_features:
        # Call out only the top 2 drivers.
        top2 = top_features[:2]
        parts: List[str] = []
        for f in top2:
            direction = "increases" if f["shap_value"] > 0 else "reduces"
            parts.append(f"{f['feature']} {direction} risk")
        joined = ", ".join(parts)
        summary_lines.append(f"Key drivers include: {joined}.")

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

