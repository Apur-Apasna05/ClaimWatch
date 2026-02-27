from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict

from backend.schemas import ClaimInput, PredictionResponse, FraudType
from backend.services.insurance_service import predict_insurance
from backend.services.job_fraud_service import predict_job_fraud
from backend.models.fraud_model import FraudModelArtifacts
from backend.models.anomaly_model import AnomalyModelArtifacts
from backend.services.explainability import ShapExplainerArtifacts
from backend.models.job_fraud_model import JobFraudArtifacts


Handler = Callable[[ClaimInput], PredictionResponse]


@dataclass
class RouterContext:
    handlers: Dict[FraudType, Handler]


_context: RouterContext | None = None


def init_router(
    *,
    fraud_artifacts: FraudModelArtifacts,
    anomaly_artifacts: AnomalyModelArtifacts,
    shap_artifacts: ShapExplainerArtifacts,
    job_artifacts: JobFraudArtifacts | None,
) -> None:
    """
    Initialize handlers for each fraud_type.

    This is called from FastAPI startup after models are loaded, so that
    each handler is a closure with its own dependencies.
    """
    global _context

    handlers: Dict[FraudType, Handler] = {
        "insurance": partial(
            predict_insurance,
            fraud_artifacts=fraud_artifacts,
            anomaly_artifacts=anomaly_artifacts,
            shap_artifacts=shap_artifacts,
        )
    }

    if job_artifacts is not None:
        handlers["job_fraud"] = partial(
            predict_job_fraud,
            job_artifacts=job_artifacts,
        )

    _context = RouterContext(handlers=handlers)


def route_prediction(claim: ClaimInput) -> PredictionResponse:
    if _context is None:
        raise RuntimeError("Model router not initialized")

    fraud_type: FraudType = claim.fraud_type or "insurance"
    if fraud_type not in _context.handlers:
        raise ValueError(f"Unsupported fraud_type: {fraud_type}")

    handler = _context.handlers[fraud_type]
    return handler(claim)

