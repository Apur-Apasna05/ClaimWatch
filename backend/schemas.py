from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


FraudType = Literal["insurance", "job_fraud"]


class ClaimInput(BaseModel):
    fraud_type: FraudType = Field("insurance", description="Fraud scenario type")

    # Insurance tabular features
    claim_amount: Optional[float] = Field(
        None, description="Total claim amount in policy currency"
    )
    policy_tenure_days: Optional[int] = Field(
        None, description="How long the policy has been active, in days"
    )
    num_prior_claims: Optional[int] = Field(
        None, description="Number of prior claims by this customer"
    )
    customer_age: Optional[int] = Field(
        None, description="Age of the policy holder"
    )

    # Job fraud features (text-based)
    job_text: Optional[str] = Field(
        None, description="Full text of job posting or extracted file text"
    )


class FeatureImportance(BaseModel):
    feature: str
    value: float
    shap_value: float


class KeywordImportance(BaseModel):
    keyword: str
    score: float


class PredictionResponse(BaseModel):
    fraud_type: FraudType
    fraud_probability: float
    """Supervised classifier output in [0, 1]."""
    fused_risk: Optional[float] = None
    """Fused risk in [0, 1] (supervised + anomaly); use for decisions. Insurance only."""
    trust_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    """Anomaly on 0–10 scale for display (insurance: from normalized anomaly; job: derived)."""
    is_anomalous: Optional[bool] = None
    fraud_persona: Optional[str] = None
    top_features: List[FeatureImportance] = []
    important_keywords: List[KeywordImportance] = []
    summary: str
    recommended_actions: List[str]
    raw_features: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    detail: Optional[str] = None


class FileUploadResponse(BaseModel):
    fraud_type: FraudType
    extracted_text: str
    prediction: PredictionResponse


class FeedbackRequest(BaseModel):
    fraud_type: FraudType
    input_payload: Dict[str, Any]
    predicted_label: str
    predicted_probability: float
    user_feedback: Literal["yes", "no"]
    timestamp: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    retrain_triggered: bool = False

