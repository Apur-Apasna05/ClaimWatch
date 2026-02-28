"""
Risk fusion layer: combine supervised fraud probability and unsupervised anomaly
into a single calibrated risk score without retraining either model.

- Normalizes raw anomaly scores to [0, 1] (1 = most anomalous).
- Fuses with fraud_probability via convex combination or logistic stacking.
- Guarantees: output in [0, 1], monotonic in both inputs, no contradiction.

Why this is statistically sound
------------------------------
1. Normalization: Linear map of raw scores using (min, max) preserves order;
   using empirical bounds from the training set makes the scale data-driven
   without retraining the anomaly model.
2. Convex combination (default): risk = α * P(fraud) + (1-α) * norm_anomaly is
   a valid expectation, bounded [0,1], and strictly increasing in both
   arguments—so higher fraud probability or higher anomaly never decreases risk.
   α reflects relative trust in the two signals (e.g. α=0.65 favours the
   supervised model when labels are reliable).
3. Logistic stacking (optional): sigmoid(β0 + β1*p + β2*a) with β1,β2 > 0 is
   monotonic in both inputs and bounded (0,1). Fixed coefficients avoid
   retraining while allowing a nonlinear blend.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

# Default range for IsolationForest.score_samples when no training bounds available
DEFAULT_RAW_MIN = -0.6
DEFAULT_RAW_MAX = 0.2


def normalize_anomaly_to_unit(
    raw_score: float,
    bounds: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Map raw anomaly score to [0, 1] with 1 = most anomalous.

    IsolationForest.score_samples: higher raw = more "normal", lower raw = more
    anomalous. We invert and linearly scale so that:
    - raw at upper bound (normal) -> 0
    - raw at lower bound (anomalous) -> 1

    Statistically valid: linear transformation of the raw score using
    empirical or fixed bounds; preserves order and is deterministic.

    Parameters
    ----------
    raw_score : float
        Raw score from model.score_samples (or decision_function if needed).
    bounds : (min, max) or None
        Empirical min/max from training set. None => use DEFAULT_RAW_*.

    Returns
    -------
    float in [0, 1]
        Normalized anomaly; 1 = most anomalous.
    """
    if bounds is not None:
        raw_min, raw_max = bounds
    else:
        raw_min, raw_max = DEFAULT_RAW_MIN, DEFAULT_RAW_MAX

    if raw_max <= raw_min:
        raw_max = raw_min + 1e-6

    # Invert: high raw (normal) -> low norm, low raw (anomalous) -> high norm
    norm = (raw_max - raw_score) / (raw_max - raw_min)
    return float(max(0.0, min(1.0, norm)))


def fuse_risk_convex(
    fraud_probability: float,
    norm_anomaly: float,
    alpha: float = 0.65,
) -> float:
    """
    Convex combination: risk = alpha * fraud_prob + (1 - alpha) * norm_anomaly.

    - Bounded [0, 1] since both inputs are in [0, 1] and alpha in [0, 1].
    - Monotonic in both: increasing either input cannot decrease risk.
    - Statistically sound: weighted average of two calibrated signals; alpha
      can reflect prior trust in the supervised vs unsupervised signal (e.g.
      alpha > 0.5 when labels are reliable).
    """
    alpha = max(0.0, min(1.0, alpha))
    p = max(0.0, min(1.0, float(fraud_probability)))
    a = max(0.0, min(1.0, float(norm_anomaly)))
    return alpha * p + (1.0 - alpha) * a


def fuse_risk_logistic(
    fraud_probability: float,
    norm_anomaly: float,
    beta0: float = -2.0,
    beta1: float = 2.0,
    beta2: float = 2.0,
) -> float:
    """
    Logistic stacking: risk = sigmoid(beta0 + beta1 * fraud_prob + beta2 * norm_anomaly).

    With beta1, beta2 > 0, the score is strictly increasing in both inputs
    (monotonic). sigmoid maps to (0, 1); in practice we clamp to [0, 1].
    No retraining: fix beta0, beta1, beta2 (e.g. beta1=beta2=2, beta0=-2
    so that equal inputs 0.5 give risk ≈ 0.5).
    """
    p = max(0.0, min(1.0, float(fraud_probability)))
    a = max(0.0, min(1.0, float(norm_anomaly)))
    z = beta0 + beta1 * p + beta2 * a
    try:
        risk = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        risk = 0.0 if z < 0 else 1.0
    return max(0.0, min(1.0, risk))


def fuse_risk(
    fraud_probability: float,
    norm_anomaly: float,
    method: str = "convex",
    alpha: float = 0.65,
    beta0: float = -2.0,
    beta1: float = 2.0,
    beta2: float = 2.0,
) -> float:
    """
    Single entry point for risk fusion.

    Parameters
    ----------
    fraud_probability : float in [0, 1]
        Supervised classifier output.
    norm_anomaly : float in [0, 1]
        Normalized anomaly (1 = most anomalous).
    method : "convex" | "logistic"
        Fusion strategy.
    alpha : float
        Weight for supervised signal in convex combination (method="convex").
    beta0, beta1, beta2 : float
        Logistic coefficients (method="logistic"); beta1, beta2 > 0 for monotonicity.

    Returns
    -------
    float in [0, 1]
        Fused risk score; use for decisions and downstream (persona, summary).
    """
    if method == "logistic":
        return fuse_risk_logistic(
            fraud_probability,
            norm_anomaly,
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
        )
    return fuse_risk_convex(fraud_probability, norm_anomaly, alpha=alpha)
