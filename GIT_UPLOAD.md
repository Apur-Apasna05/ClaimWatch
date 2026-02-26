# Upload ClaimWatch AI to Git (GitHub / GitLab / etc.)

A `.gitignore` is already in the project. Run these in **PowerShell** from the project root.

---

## 1. Open terminal at project root

```powershell
cd "c:\Users\apple\OneDrive\Desktop\hackathon\insurance fraud"
```

---

## 2. Initialize Git (if not already done)

```powershell
git init
```

---

## 3. Add all files and commit

```powershell
git add .
git status
git commit -m "Initial commit: ClaimWatch AI fraud detection prototype"
```

---

## 4. Create a repo on GitHub (or your host)

- **GitHub:** Go to [github.com/new](https://github.com/new), create a new repository (e.g. `claimwatch-ai`). **Do not** add a README, .gitignore, or license if you want to push this code.
- Copy the repo URL (e.g. `https://github.com/YourUsername/claimwatch-ai.git` or `git@github.com:YourUsername/claimwatch-ai.git`).

---

## 5. Add remote and push

Replace `YOUR_REPO_URL` with your actual repo URL:

```powershell
git remote add origin YOUR_REPO_URL
git branch -M main
git push -u origin main
```

**Example (HTTPS):**

```powershell
git remote add origin https://github.com/YourUsername/claimwatch-ai.git
git branch -M main
git push -u origin main
```

**Example (SSH):**

```powershell
git remote add origin git@github.com:YourUsername/claimwatch-ai.git
git branch -M main
git push -u origin main
```

---

## 6. If you already had a repo and hit "permission denied"

- Close any program that might be using the folder (e.g. another VS Code window, OneDrive sync).
- Run PowerShell **as Administrator** and try again from step 2.
- If the project is in **OneDrive**, try moving it to a non-synced folder (e.g. `C:\Projects\insurance-fraud`) and run the Git commands there.

---

## What’s in `.gitignore`

- `.venv/` (virtual environment)
- `__pycache__/`, `.env`, IDE/OS junk
- Trained model files are **included** by default so clones can run the app; to exclude them, uncomment the `backend/models/artifacts/*.joblib` line in `.gitignore`.
