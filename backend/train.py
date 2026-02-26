from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running as script (e.g. python backend/train.py)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd

from backend.config import settings
from backend.models.anomaly_model import (
    AnomalyModelArtifacts,
    save_anomaly_model,
    train_anomaly_model,
)
from backend.models.fraud_model import (
    FraudModelArtifacts,
    save_fraud_model,
    train_fraud_model,
)


def main() -> None:
    data_path: Path = settings.data_path
    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected training data at {data_path}. "
            "Please place claims_sample.csv there (see README)."
        )

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print("Training supervised fraud model...")
    fraud_artifacts: FraudModelArtifacts = train_fraud_model(df)

    print("Training anomaly detection model...")
    anomaly_artifacts: AnomalyModelArtifacts = train_anomaly_model(df)

    artifacts_dir = settings.model_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fraud_path = artifacts_dir / "fraud_model.joblib"
    anomaly_path = artifacts_dir / "anomaly_model.joblib"

    print(f"Saving fraud model to {fraud_path}...")
    save_fraud_model(fraud_artifacts, fraud_path)

    print(f"Saving anomaly model to {anomaly_path}...")
    save_anomaly_model(anomaly_artifacts, anomaly_path)

    print("Training complete.")


if __name__ == "__main__":
    main()

