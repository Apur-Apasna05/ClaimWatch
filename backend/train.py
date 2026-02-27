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


def _load_insurance_dataframe() -> pd.DataFrame:
    """
    Load insurance training data.

    Priority:
    1) data/claims_sample.csv               (simple sample file)
    2) data/insurance_claims.csv           (common Kaggle dataset name)
    3) data/insurance_fraud.csv or similar (user-provided mapping point)
    """
    # 1) Preferred path (existing sample)
    if settings.data_path.exists():
        print(f"Loading insurance data from {settings.data_path}...")
        return pd.read_csv(settings.data_path)

    base = settings.base_dir / "data"
    kaggle_candidates = [
        base / "insurance_claims.csv",
        base / "insurance_fraud.csv",
    ]
    for path in kaggle_candidates:
        if path.exists():
            print(f"Loading Kaggle-style insurance data from {path}...")
            df_raw = pd.read_csv(path)
            # Minimal mapping example for typical Kaggle \"insurance_claims\" dataset:
            # - 'total_claim_amount' -> claim_amount
            # - 'policy_duration' or 'policy_bind_date' derived -> policy_tenure_days
            # - 'umbrella_limit' or 'incident_hour_of_the_day' etc. can be ignored or engineered.
            mapping: dict[str, str] = {}
            if "total_claim_amount" in df_raw.columns:
                mapping["claim_amount"] = "total_claim_amount"
            if "months_as_customer" in df_raw.columns:
                # months_as_customer * 30 ≈ tenure days
                df_raw["policy_tenure_days"] = df_raw["months_as_customer"] * 30
                mapping["policy_tenure_days"] = "policy_tenure_days"
            if "number_of_open_claims" in df_raw.columns:
                mapping["num_prior_claims"] = "number_of_open_claims"
            if "age" in df_raw.columns:
                mapping["customer_age"] = "age"
            if "fraud_reported" in df_raw.columns:
                # Typical 'Y'/'N' → 1/0
                df_raw["is_fraud"] = (df_raw["fraud_reported"].astype(str).str.upper() == "Y").astype(int)

            required = [
                "claim_amount",
                "policy_tenure_days",
                "num_prior_claims",
                "customer_age",
                "is_fraud",
            ]
            missing = [c for c in required if c not in df_raw.columns and c not in mapping]
            if missing:
                raise ValueError(
                    f"Found insurance dataset at {path}, but could not map required columns: {missing}. "
                    "Update backend/train.py mapping for your specific Kaggle columns."
                )

            df_mapped = pd.DataFrame()
            for col in required:
                if col in df_raw.columns:
                    df_mapped[col] = df_raw[col]
                else:
                    source = mapping[col]
                    df_mapped[col] = df_raw[source]

            return df_mapped

    raise FileNotFoundError(
        "No suitable insurance training data found. "
        "Place either 'claims_sample.csv' or a Kaggle-style 'insurance_claims.csv' in the data/ folder."
    )


def main() -> None:
    df = _load_insurance_dataframe()

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

