from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .fraud_model import FEATURE_COLUMNS

# Default range for IsolationForest.score_samples (empirically typical).
# Used when no training-time bounds are stored (backward compatibility).
DEFAULT_RAW_MIN = -0.6
DEFAULT_RAW_MAX = 0.2


@dataclass
class AnomalyModelArtifacts:
    model: IsolationForest
    feature_columns: List[str]
    # (min, max) of score_samples on training data for normalization. None = use defaults.
    score_bounds: Optional[Tuple[float, float]] = None


def train_anomaly_model(df: pd.DataFrame) -> AnomalyModelArtifacts:
    df = df.dropna(subset=FEATURE_COLUMNS)
    X = df[FEATURE_COLUMNS]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )
    model.fit(X)

    # Empirical bounds for calibration: score_samples on training set
    scores = model.score_samples(X)
    raw_min = float(np.nanmin(scores))
    raw_max = float(np.nanmax(scores))
    if raw_max <= raw_min:
        raw_max = raw_min + 1e-6
    score_bounds = (raw_min, raw_max)

    return AnomalyModelArtifacts(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        score_bounds=score_bounds,
    )


def save_anomaly_model(artifacts: AnomalyModelArtifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": artifacts.model,
            "feature_columns": artifacts.feature_columns,
            "score_bounds": getattr(artifacts, "score_bounds", None),
        },
        path,
    )


def load_anomaly_model(path: Path) -> AnomalyModelArtifacts:
    obj = joblib.load(path)
    return AnomalyModelArtifacts(
        model=obj["model"],
        feature_columns=list(obj["feature_columns"]),
        score_bounds=obj.get("score_bounds"),
    )


def anomaly_score(artifacts: AnomalyModelArtifacts, features: dict) -> float:
    row = np.array([[features[c] for c in artifacts.feature_columns]], dtype=float)
    # IsolationForest score_samples: higher scores = less anomalous (often negative for anomalies)
    raw = artifacts.model.score_samples(row)[0]
    score = float(raw)
    if np.isnan(score) or np.isinf(score):
        return 0.0
    return score

