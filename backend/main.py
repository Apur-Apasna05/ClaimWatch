from __future__ import annotations

from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.models.anomaly_model import AnomalyModelArtifacts, load_anomaly_model
from backend.models.fraud_model import FraudModelArtifacts, load_fraud_model
from backend.models.job_fraud_model import JobFraudArtifacts, load_job_fraud_model
from backend.schemas import (
    ClaimInput,
    HealthResponse,
    PredictionResponse,
    FileUploadResponse,
    FeedbackRequest,
    FeedbackResponse,
    FraudType,
)
from backend.services.explainability import ShapExplainerArtifacts, build_tree_explainer
from backend.services.model_router import init_router, route_prediction
from backend.services.feedback_service import log_feedback, should_retrain
from backend.utils.file_processor import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_docx,
    extract_text_from_image,
)


app = FastAPI(title=settings.project_name)

fraud_artifacts: FraudModelArtifacts | None = None
anomaly_artifacts: AnomalyModelArtifacts | None = None
shap_artifacts: ShapExplainerArtifacts | None = None
job_artifacts: JobFraudArtifacts | None = None


def _load_artifacts() -> None:
    global fraud_artifacts, anomaly_artifacts, shap_artifacts, job_artifacts

    model_dir: Path = settings.model_dir
    fraud_path = model_dir / "fraud_model.joblib"
    anomaly_path = model_dir / "anomaly_model.joblib"
    job_path = model_dir / "job_fraud_model.joblib"

    if not fraud_path.exists() or not anomaly_path.exists():
        raise RuntimeError(
            "Model artifacts not found. "
            "Run `python backend/train.py` first to train and save models."
        )

    fraud_artifacts = load_fraud_model(fraud_path)
    anomaly_artifacts = load_anomaly_model(anomaly_path)
    shap_artifacts = build_tree_explainer(fraud_artifacts.model, fraud_artifacts.feature_columns)

    if job_path.exists():
        job_artifacts = load_job_fraud_model(job_path)
    else:
        job_artifacts = None

    init_router(
        fraud_artifacts=fraud_artifacts,
        anomaly_artifacts=anomaly_artifacts,
        shap_artifacts=shap_artifacts,
        job_artifacts=job_artifacts,
    )


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
        if fraud_artifacts is None or anomaly_artifacts is None or shap_artifacts is None:
            _load_artifacts()
        return HealthResponse(status="ok", detail="Models loaded")
    except Exception as exc:  # pragma: no cover - defensive
        return HealthResponse(status="error", detail=str(exc))


@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimInput) -> PredictionResponse:
    if fraud_artifacts is None or anomaly_artifacts is None or shap_artifacts is None:
        _load_artifacts()
    try:
        return route_prediction(claim)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict-from-file", response_model=FileUploadResponse)
async def predict_from_file(
    fraud_type: FraudType = "job_fraud",
    file: UploadFile = File(...),
) -> FileUploadResponse:
    if fraud_type != "job_fraud":
        raise HTTPException(status_code=400, detail="File-based prediction is currently supported for job_fraud only.")

    content_type = (file.content_type or "").lower()

    if "pdf" in content_type:
        text = extract_text_from_pdf(file.file)
    elif "text" in content_type:
        text = extract_text_from_txt(file.file)
    elif "word" in content_type or file.filename.lower().endswith(".docx"):
        text = extract_text_from_docx(file.file)
    elif "image" in content_type:
        text = extract_text_from_image(file.file)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    claim = ClaimInput(fraud_type="job_fraud", job_text=text)
    prediction = predict(claim)

    return FileUploadResponse(
        fraud_type="job_fraud",
        extracted_text=text,
        prediction=prediction,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    log_feedback(req)
    retrain = should_retrain()
    return FeedbackResponse(status="logged", retrain_triggered=retrain)

