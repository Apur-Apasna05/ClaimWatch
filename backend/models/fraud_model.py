from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - xgboost may be unavailable
    XGBClassifier = None  # type: ignore


FEATURE_COLUMNS: List[str] = [
    "claim_amount",
    "policy_tenure_days",
    "num_prior_claims",
    "customer_age",
]

TARGET_COLUMN = "is_fraud"


@dataclass
class FraudModelArtifacts:
    model: object
    feature_columns: List[str]


def train_fraud_model(df: pd.DataFrame) -> FraudModelArtifacts:
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        )

    model.fit(X, y)
    return FraudModelArtifacts(model=model, feature_columns=FEATURE_COLUMNS)


def save_fraud_model(artifacts: FraudModelArtifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": artifacts.model,
            "feature_columns": artifacts.feature_columns,
        },
        path,
    )


def load_fraud_model(path: Path) -> FraudModelArtifacts:
    obj = joblib.load(path)
    return FraudModelArtifacts(
        model=obj["model"],
        feature_columns=list(obj["feature_columns"]),
    )


def predict_proba(artifacts: FraudModelArtifacts, features: dict) -> float:
    row = np.array([[features[c] for c in artifacts.feature_columns]], dtype=float)
    proba = artifacts.model.predict_proba(row)[0, 1]
    return float(proba)

