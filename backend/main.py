from __future__ import annotations

from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.models.anomaly_model import AnomalyModelArtifacts, anomaly_score, load_anomaly_model
from backend.models.fraud_model import FraudModelArtifacts, FEATURE_COLUMNS, load_fraud_model
from backend.schemas import ClaimInput, HealthResponse, PredictionResponse, FeatureImportance
from backend.services.explainability import ShapExplainerArtifacts, build_tree_explainer, explain_single
from backend.services.generative_reporting import generate_template_summary


app = FastAPI(title=settings.project_name)

fraud_artifacts: FraudModelArtifacts | None = None
anomaly_artifacts: AnomalyModelArtifacts | None = None
shap_artifacts: ShapExplainerArtifacts | None = None


def _load_artifacts() -> None:
    global fraud_artifacts, anomaly_artifacts, shap_artifacts

    model_dir: Path = settings.model_dir
    fraud_path = model_dir / "fraud_model.joblib"
    anomaly_path = model_dir / "anomaly_model.joblib"

    if not fraud_path.exists() or not anomaly_path.exists():
        raise RuntimeError(
            "Model artifacts not found. "
            "Run `python backend/train.py` first to train and save models."
        )

    fraud_artifacts = load_fraud_model(fraud_path)
    anomaly_artifacts = load_anomaly_model(anomaly_path)
    shap_artifacts = build_tree_explainer(fraud_artifacts.model, fraud_artifacts.feature_columns)


@app.on_event("startup")
def startup_event() -> None:
    _load_artifacts()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect to dashboard."""
    return RedirectResponse(url="/dashboard/")


@app.get("/dashboard", include_in_schema=False)
def dashboard_redirect() -> RedirectResponse:
    return RedirectResponse(url="/dashboard/")


# Serve HTML/CSS/JS frontend (must be after routes that take precedence)
_frontend_dir = settings.base_dir / "frontend"
if _frontend_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_frontend_dir), html=True), name="dashboard")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        if fraud_artifacts is None or anomaly_artifacts is None:
            _load_artifacts()
        return HealthResponse(status="ok", detail="Models loaded")
    except Exception as exc:  # pragma: no cover - defensive
        return HealthResponse(status="error", detail=str(exc))


@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimInput) -> PredictionResponse:
    if fraud_artifacts is None or anomaly_artifacts is None or shap_artifacts is None:
        _load_artifacts()

    # Convert input to feature dict
    features: Dict[str, float] = {col: float(getattr(claim, col)) for col in FEATURE_COLUMNS}

    try:
        fraud_prob = fraud_artifacts.model.predict_proba([[features[c] for c in FEATURE_COLUMNS]])[0, 1]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fraud prediction failed: {exc}")

    try:
        a_score = anomaly_score(anomaly_artifacts, features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly scoring failed: {exc}")

    # Heuristic threshold: treat very low scores as anomalous
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

    summary, actions = generate_template_summary(
        fraud_probability=fraud_prob,
        anomaly_score=a_score,
        top_features=top_features_dicts,
    )

    return PredictionResponse(
        fraud_probability=fraud_prob,
        anomaly_score=a_score,
        is_anomalous=bool(is_anomalous),
        top_features=top_features,
        summary=summary,
        recommended_actions=actions,
        raw_features=features,
    )

