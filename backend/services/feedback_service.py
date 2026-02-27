from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from backend.config import settings
from backend.schemas import FeedbackRequest


FEEDBACK_PATH = settings.base_dir / "data" / "feedback_log.csv"
RETRAIN_THRESHOLD = 100


def log_feedback(req: FeedbackRequest) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                [
                    "timestamp",
                    "fraud_type",
                    "predicted_label",
                    "predicted_probability",
                    "user_feedback",
                    "input_payload",
                ]
            )
        ts = req.timestamp or datetime.utcnow().isoformat()
        writer.writerow(
            [
                ts,
                req.fraud_type,
                req.predicted_label,
                req.predicted_probability,
                req.user_feedback,
                str(req.input_payload),
            ]
        )


def should_retrain() -> bool:
    if not FEEDBACK_PATH.exists():
        return False
    with FEEDBACK_PATH.open("r", encoding="utf-8") as f:
        count = sum(1 for _ in f) - 1  # minus header
    return count >= RETRAIN_THRESHOLD

