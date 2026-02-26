# ClaimWatch AI – Steps to Run

Follow these steps **in order** from the project root:  
`c:\Users\apple\OneDrive\Desktop\hackathon\insurance fraud`

---

## Step 1: Open terminal at project root

```powershell
cd "c:\Users\apple\OneDrive\Desktop\hackathon\insurance fraud"
```

---

## Step 2: Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

You should see `(.venv)` in your prompt.

---

## Step 3: Install dependencies

**If you previously saw a numpy "metadata-generation-failed" or "meson" error**, use this order:

```powershell
pip install --upgrade pip
pip install numpy
pip install -r requirements.txt
```

Installing `numpy` first makes pip use a pre-built wheel instead of building from source.

**Otherwise**, you can run:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If you get a **permission error** on Windows, try:
- Closing other programs that might be using Python, or
- Running the terminal **as Administrator**, or
- Installing into your user folder: `pip install --user -r requirements.txt`

---

## Step 4: Add training data

Create a file **`data/claims_sample.csv`** with at least these columns:

| Column               | Description                    |
|----------------------|--------------------------------|
| `claim_amount`       | Total claim amount (number)    |
| `policy_tenure_days` | Days since policy start        |
| `num_prior_claims`   | Number of prior claims         |
| `customer_age`       | Age of policyholder            |
| `is_fraud`           | 0 = genuine, 1 = fraud         |

**Example (copy into `data/claims_sample.csv`):**

```csv
claim_amount,policy_tenure_days,num_prior_claims,customer_age,is_fraud
5000,365,1,35,0
12000,30,3,29,1
3500,180,0,45,0
25000,60,5,28,1
8000,400,2,52,0
```

---

## Step 5: Train the models

From the project root (with venv activated):

```powershell
python backend/train.py
```

If you see `ModuleNotFoundError: No module named 'backend'`, run instead:

```powershell
python -m backend.train
```

Wait until you see **"Training complete."**  
Models are saved in `backend/models/artifacts/`.

---

## Step 6: Start the backend (Terminal 1)

Use the **same Python** that has your packages (avoids "No module named 'shap'" in the worker):

```powershell
.\.venv\Scripts\activate
python -m uvicorn backend.main:app --reload --port 8000
```

Leave this terminal open. You should see **"Application startup complete"** (no traceback).

---

## Step 7: Open the dashboard in your browser

The frontend is **HTML/CSS/JS** served by the same backend. No second terminal needed.

Open in your browser:

- **Dashboard:** **http://127.0.0.1:8000/** or **http://127.0.0.1:8000/dashboard/**
- **Health check:** http://127.0.0.1:8000/health
- **API docs:** http://127.0.0.1:8000/docs

On the dashboard, enter claim details and click **Evaluate claim** to see fraud probability, anomaly score, explanation, and recommended actions.

---

## Quick reference

| What              | Command / URL                                                |
|-------------------|--------------------------------------------------------------|
| Activate venv     | `.\.venv\Scripts\activate`                                   |
| Train models      | `python backend/train.py`                                    |
| Run backend       | `python -m uvicorn backend.main:app --reload --port 8000`     |
| Dashboard         | http://127.0.0.1:8000/ or http://127.0.0.1:8000/dashboard/    |
| Health check      | http://127.0.0.1:8000/health                                 |
| API docs          | http://127.0.0.1:8000/docs                                   |
