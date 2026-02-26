## ClaimWatch AI – Insurance Fraud Detection Prototype

ClaimWatch AI is an intelligent fraud detection prototype designed for insurance claims. It combines **supervised ML fraud prediction**, **anomaly detection**, **explainable AI**, and a **generative-style investigation summary** layer, exposed via a FastAPI backend and a simple dashboard.

### 1. Project Structure

- **backend/**
  - `main.py` – FastAPI app with prediction APIs
  - `train.py` – offline training script for fraud + anomaly models
  - `config.py` – configuration and paths
  - `schemas.py` – Pydantic request/response models
  - `models/` – saved model logic
  - `services/` – explainability + reporting utilities
- **frontend/**
  - `index.html` – dashboard page
  - `css/styles.css` – styles
  - `js/app.js` – calls `/predict` and renders results
  - `dashboard.py` – legacy Streamlit option (not required)
- **data/**
  - `claims_sample.csv` – your training data (you provide)
- `requirements.txt` – Python dependencies
- `.env.example` – example for API keys and config

### 2. Python & Environment Setup

1. Install Python 3.10+ if you don’t have it.
2. In a terminal at the project root:

```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare Sample Data

Place a CSV file at `data/claims_sample.csv` with at least:

- `claim_amount`
- `policy_tenure_days`
- `num_prior_claims`
- `customer_age`
- `is_fraud` (0 = genuine, 1 = fraud)

You can extend this with more features; just keep the target column named `is_fraud`.

### 4. Train Models (Fraud + Anomaly)

From the project root:

```bash
.venv\Scripts\activate
python backend/train.py
```

This will:

- Load `data/claims_sample.csv`
- Train a supervised fraud classifier (XGBoost if available, fallback to RandomForest)
- Train an `IsolationForest` anomaly detector
- Save models and feature metadata to `backend/models/artifacts/`

### 5. Run the FastAPI Backend

```bash
.venv\Scripts\activate
uvicorn backend.main:app --reload --port 8000
```

Key endpoints:

- `GET /health` – health check
- `POST /predict` – send a single claim JSON, receive:
  - fraud probability
  - anomaly score
  - top important features (SHAP-based)
  - an auto-generated investigation summary + recommended actions

### 6. Open the dashboard (HTML/CSS/JS)

The same FastAPI server serves a static frontend. With the backend running, open in your browser:

- **http://127.0.0.1:8000/** or **http://127.0.0.1:8000/dashboard/**

You can:

- Enter claim details in the form
- Click **Evaluate claim** to call the API
- See fraud probability, anomaly score, top factors (SHAP), summary, and recommended actions

### 7. Configure Generative AI (Optional)

The prototype includes a deterministic, template-based summary generator by default (no external API). If you want to integrate an LLM (e.g., OpenAI):

1. Copy `.env.example` to `.env`.
2. Fill in `OPENAI_API_KEY` and set `USE_OPENAI_SUMMARIES=true`.
3. Restart the backend; the reporting service will switch to LLM-based summaries.

### 8. Hackathon Demo Flow

1. Explain the architecture using this README.
2. Show the training script and model artifacts.
3. Call the `/predict` endpoint from:
   - Streamlit dashboard, or
   - `curl` / Postman.
4. Highlight:
   - Fraud probability and anomaly score
   - Explainable features from SHAP
   - Auto-generated investigation summary and next steps.

