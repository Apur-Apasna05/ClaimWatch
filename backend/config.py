import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


class Settings:
    project_name: str = "ClaimWatch AI"
    base_dir: Path = BASE_DIR
    model_dir: Path = BASE_DIR / "backend" / "models" / "artifacts"
    data_path: Path = BASE_DIR / "data" / "claims_sample.csv"

    use_openai_summaries: bool = os.getenv("USE_OPENAI_SUMMARIES", "false").lower() == "true"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")


settings = Settings()

