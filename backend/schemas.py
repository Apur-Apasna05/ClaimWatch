from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ClaimInput(BaseModel):
    claim_amount: float = Field(..., description="Total claim amount in policy currency")
    policy_tenure_days: int = Field(..., description="How long the policy has been active, in days")
    num_prior_claims: int = Field(..., description="Number of prior claims by this customer")
    customer_age: int = Field(..., description="Age of the policy holder")

    # Extra features can be added here as optional fields
    # example: region: Optional[str] = None


class FeatureImportance(BaseModel):
    feature: str
    value: float
    shap_value: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    anomaly_score: float
    is_anomalous: bool
    top_features: List[FeatureImportance]
    summary: str
    recommended_actions: List[str]
    raw_features: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    detail: Optional[str] = None

