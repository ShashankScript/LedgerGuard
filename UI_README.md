# UI_README.md

## Pages

| Page | Purpose |
|---|---|
| Data / Run | Generate synthetic demo data or upload real CSVs; runs the pipeline. |
| Command Center | Headline KPIs — records, matched/unresolved, automation rate, measured accuracy (if available). |
| Reconciliation | Full transaction table with status and filters. |
| Transaction Investigator | Click any transaction, see every candidate considered and why the final decision was made. |
| Exception Center | Unresolved records grouped by reason. |
| Human Review | Approve/reject/mark-unresolved for medium-confidence cases; every action is audit-logged. |
| Model / Evaluation | Rule-based vs. ML classifier comparison on held-out data, honestly reported. |
| Audit Log | Every automated decision and human override, in order. |

## Honesty guarantees enforced in the UI layer

- If ground truth isn't available (real uploaded data), the Command Center
  and Model/Evaluation pages explicitly show **"Not evaluated"** rather than
  a number — see `train_ml_if_possible()` in `app.py`.
- The Model/Evaluation page states plainly when the ML model does **not**
  outperform the rule baseline, rather than only surfacing favorable numbers.
- All numbers shown are recomputed from the actual pipeline output each run —
  nothing in the UI is hardcoded.

## Known UI limitations (be upfront about these if asked)

- This was built and syntax-checked in an offline sandbox without internet
  access, so Streamlit itself could not be installed and run live during
  development — every underlying function it calls was tested directly and
  passes, but you should do one live `streamlit run app.py` pass yourself
  before your demo to catch any rendering-level issues.
- Human Review overrides are stored in Streamlit session state (reset if the
  app restarts) plus logged permanently to `audit_log.csv` — the log persists,
  the in-session "reviewed" badges do not survive a restart.
- No pagination on large tables — fine through ~5,000 records, would need it
  for larger production data.
