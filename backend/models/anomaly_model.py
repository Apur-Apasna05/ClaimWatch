from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .fraud_model import FEATURE_COLUMNS


@dataclass
class AnomalyModelArtifacts:
    model: IsolationForest
    feature_columns: List[str]


def train_anomaly_model(df: pd.DataFrame) -> AnomalyModelArtifacts:
    df = df.dropna(subset=FEATURE_COLUMNS)
    X = df[FEATURE_COLUMNS]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )
    model.fit(X)
    return AnomalyModelArtifacts(model=model, feature_columns=FEATURE_COLUMNS)


def save_anomaly_model(artifacts: AnomalyModelArtifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": artifacts.model,
            "feature_columns": artifacts.feature_columns,
        },
        path,
    )


def load_anomaly_model(path: Path) -> AnomalyModelArtifacts:
    obj = joblib.load(path)
    return AnomalyModelArtifacts(
        model=obj["model"],
        feature_columns=list(obj["feature_columns"]),
    )


def anomaly_score(artifacts: AnomalyModelArtifacts, features: dict) -> float:
    row = np.array([[features[c] for c in artifacts.feature_columns]], dtype=float)
    # IsolationForest score_samples: higher scores = less anomalous
    score = artifacts.model.score_samples(row)[0]
    return float(score)

