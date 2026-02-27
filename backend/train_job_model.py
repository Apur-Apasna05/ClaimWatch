from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path when running as script (e.g. python backend/train_job_model.py)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backend.config import settings
from backend.models.job_fraud_model import (
    JobFraudArtifacts,
    save_job_fraud_model,
    train_job_fraud_model,
)


def _load_job_dataframe() -> pd.DataFrame:
    """
    Load job fraud training data.

    Priority:
    1) data/job_posts_sample.csv            (simple sample format text,label)
    2) data/fake_job_postings.csv          (Kaggle 'Fake Job Postings Dataset')
    """
    base = settings.base_dir / "data"
    simple_path = base / "job_posts_sample.csv"
    if simple_path.exists():
        print(f"Loading job fraud data from {simple_path}...")
        df = pd.read_csv(simple_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("job_posts_sample.csv must contain 'text' and 'label' columns")
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        df = df[df["text"].str.len() > 0].copy()
        df["label"] = df["label"].astype(int)
        return df

    kaggle_path = base / "fake_job_postings.csv"
    if kaggle_path.exists():
        print(f"Loading Kaggle-style job fraud data from {kaggle_path}...")
        df_raw = pd.read_csv(kaggle_path)
        # Typical Kaggle 'fake_job_postings' columns: 'description', 'fraudulent'
        if "description" not in df_raw.columns or "fraudulent" not in df_raw.columns:
            raise ValueError(
                "fake_job_postings.csv must contain 'description' and 'fraudulent' columns, "
                "or adjust _load_job_dataframe mapping."
            )
        df_mapped = pd.DataFrame()
        df_mapped["text"] = df_raw["description"].fillna("").astype(str).str.strip()
        df_mapped["label"] = df_raw["fraudulent"].astype(int)
        df_mapped = df_mapped[df_mapped["text"].str.len() > 0].copy()
        return df_mapped

    raise FileNotFoundError(
        "No suitable job fraud training data found. "
        "Place either 'job_posts_sample.csv' or Kaggle's 'fake_job_postings.csv' in the data/ folder."
    )


def main() -> None:
    df = _load_job_dataframe()

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    print("Training job fraud text model...")
    artifacts: JobFraudArtifacts = train_job_fraud_model(texts, labels)

    artifacts_dir = settings.model_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / "job_fraud_model.joblib"

    print(f"Saving job fraud model to {path}...")
    save_job_fraud_model(artifacts, path)
    print("Job fraud training complete.")


if __name__ == "__main__":
    main()

