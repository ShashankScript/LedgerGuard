# LedgerGuard — AI Finance Controller

Reconcile money movement. Explain every decision. Escalate uncertainty.

Reconciles payment gateway transactions against bank statement records —
exact matching, fuzzy matching, and an ML classifier compared honestly
against each other, with a full audit trail and human-review workflow.

---

## Setup (Windows + VS Code)

**Step 1 — Open the project**
Open the extracted `LedgerGuard` folder in VS Code (`File → Open Folder...`).

**Step 2 — Open the terminal**
`View → Terminal` (or `` Ctrl+` ``).

**Step 3 — Create a virtual environment**
```
python -m venv .venv
```

**Step 4 — Activate it**
```
.venv\Scripts\activate
```
Your terminal prompt should now start with `(.venv)`.

**Step 5 — Install dependencies**
```
pip install -r requirements.txt
```

**Step 6 — Launch**
```
streamlit run app.py
```
This opens `http://localhost:8501` in your browser automatically.

---

## Using it

Go to **Data / Run → Demo Mode → Generate & Run Demo** for a one-click,
fully offline walkthrough (no CSVs to prepare). See `DEMO_GUIDE.md` for a
suggested tour of every page.

To use your own data instead, go to **Data / Run → Upload CSVs**. Required
columns: `gateway_txn_id, reference, amount, date` for the gateway file, and
`bank_txn_id, reference, amount, date` for the bank file.

---

## Troubleshooting

**"'python' is not recognized as an internal or external command"**
Python isn't installed or isn't on your PATH. Install it from
[python.org](https://python.org) — during install, check **"Add python.exe to
PATH"**. Restart VS Code's terminal afterward. Try `python --version` to confirm.

**`pip install` fails / times out / SSL errors**
- Confirm you're connected to the internet.
- Try `python -m pip install --upgrade pip` first, then re-run
  `pip install -r requirements.txt`.
- If behind a corporate proxy/VPN, you may need
  `pip install -r requirements.txt --proxy http://your-proxy:port`.

**"ModuleNotFoundError: No module named 'streamlit'" (or pandas/sklearn/numpy)**
Your virtual environment isn't activated, or dependencies weren't installed
into it. Confirm your prompt shows `(.venv)`, then re-run:
```
.venv\Scripts\activate
pip install -r requirements.txt
```

**Streamlit doesn't open a browser window**
Manually go to `http://localhost:8501` in any browser. If that doesn't load,
check the terminal output for the actual URL/port it bound to.

**"Port 8501 is already in use"**
Another Streamlit instance (or something else) is using that port. Either
close it, or launch on a different port:
```
streamlit run app.py --server.port 8502
```

**App runs but pages show "No data loaded yet"**
Expected on first launch — go to **Data / Run** and either generate demo
data or upload CSVs first.

**Import errors after moving/renaming folders**
This project uses Python package imports (`from matching import
rule_matcher`, etc.) which require the folder structure to stay intact and
`streamlit run app.py` to be launched **from the `LedgerGuard` root folder**
— not from inside a subfolder. If you renamed the root folder, that's fine;
just don't rename or move the subfolders (`data/`, `matching/`, etc.)
relative to `app.py`.

---

## Project layout

See `PROJECT_STRUCTURE.md` for what each folder does, and `ARCHITECTURE.md`
for how the pipeline fits together.

## What this doesn't do (yet)

See `LIMITATIONS.md` for the full, honest list — no LLM agent, no
calibration, no Razorpay API integration, synthetic data only unless you
upload your own.

## Running tests

```
python -m unittest tests.test_core -v
```

## Running pipeline steps individually (optional — the UI does all of this for you)

```
python -m data.generate_data --n 500 --seed 42
python -m matching.rule_matcher
python -m evaluation.evaluate
python -m evaluation.sweep
python -m matching.build_training_data
python -m matching.ml_matcher
```
All must be run from the `LedgerGuard` root folder.
