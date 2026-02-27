from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class JobFraudArtifacts:
    vectorizer: TfidfVectorizer
    model: LogisticRegression
    feature_names: List[str]


def train_job_fraud_model(
    texts: List[str],
    labels: List[int],
    *,
    max_features: int = 10000,
) -> JobFraudArtifacts:
    # Clean bad/empty documents while keeping labels aligned
    cleaned_texts: List[str] = []
    cleaned_labels: List[int] = []
    for t, y in zip(texts, labels):
        if not isinstance(t, str):
            continue
        t2 = t.strip()
        if not t2:
            continue
        cleaned_texts.append(t2)
        cleaned_labels.append(int(y))

    if not cleaned_texts:
        raise ValueError("No valid training documents found after cleaning (empty/NaN texts).")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
    )
    X = vectorizer.fit_transform(cleaned_texts)

    model = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
    )
    model.fit(X, cleaned_labels)

    feature_names = list(vectorizer.get_feature_names_out())
    return JobFraudArtifacts(
        vectorizer=vectorizer,
        model=model,
        feature_names=feature_names,
    )


def save_job_fraud_model(artifacts: JobFraudArtifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": artifacts.vectorizer,
            "model": artifacts.model,
            "feature_names": artifacts.feature_names,
        },
        path,
    )


def load_job_fraud_model(path: Path) -> JobFraudArtifacts:
    obj = joblib.load(path)
    return JobFraudArtifacts(
        vectorizer=obj["vectorizer"],
        model=obj["model"],
        feature_names=list(obj["feature_names"]),
    )


def predict_job_proba(artifacts: JobFraudArtifacts, text: str) -> float:
    X = artifacts.vectorizer.transform([text])
    proba = artifacts.model.predict_proba(X)[0, 1]
    return float(proba)


def top_keywords(
    artifacts: JobFraudArtifacts,
    text: str,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    X = artifacts.vectorizer.transform([text])
    scores = X.toarray()[0]
    idxs = np.argsort(scores)[::-1][:top_k]
    return [
        (artifacts.feature_names[i], float(scores[i]))
        for i in idxs
        if scores[i] > 0
    ]

