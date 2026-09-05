# DEMO_GUIDE.md

## Launch

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. No other commands needed — the app
generates and processes data itself.

## Suggested walkthrough (≈5 minutes)

1. **Data / Run → Demo Mode.** Leave the default (500 records), click
   "Generate & Run Demo." This generates a synthetic batch (with fee
   deductions, refunds, mangled references, duplicates, and deliberately
   ambiguous decoy transactions), runs the full reconciliation pipeline, and
   trains the ML comparison model — all in a few seconds.

2. **Command Center.** Shows the headline numbers: records processed,
   matched/unresolved counts, automation rate, and (since this is demo data
   with ground truth) real precision/recall/F1 — not invented numbers.

3. **Reconciliation.** The full transaction table, filterable by match method
   (exact / fuzzy / unresolved). Point out the `amount_diff` and `reason`
   columns — every row is explainable at a glance.

4. **Transaction Investigator.** Pick any gateway transaction. Shows every
   candidate bank record considered, its similarity/amount/date features,
   rule score, and ML probability side by side, plus the final decision and
   why. This is the "click any transaction, understand exactly why" feature.

5. **Exception Center.** Unresolved records grouped by *reason* (missing
   bank record, low confidence, duplicate, etc.) rather than dumped in one
   pile.

6. **Human Review.** Pick a medium-confidence case, show the evidence and
   candidates, click Approve / Reject / Mark Unresolved — watch it get
   logged.

7. **Model / Evaluation.** The honest rule-vs-ML comparison on a held-out
   test set. If asked "why does this need AI" — this page is the answer,
   including the honest finding about where ML does and doesn't help.

8. **Audit Log.** Show the entry created by the human review action above —
   full provenance: who, what, when, why.

## If ML doesn't outperform the rule baseline

This is expected and reported honestly on the Model / Evaluation page — do
not be thrown by it. The finding itself (a well-designed 6-feature rule
formula ties a trained classifier) is a legitimate, defensible result. See
`matching/ml_matcher.py`'s feature importance output for why: reference similarity
dominates both approaches.

## Offline guarantee

The core demo (steps 1–8 above) requires no external API calls and no
internet connection — everything runs from the bundled synthetic data
generator and local scikit-learn training.
# Difficult-case walkthrough

In **Transaction Investigator**, choose a record with a `credit_after_refund`
candidate to show the amount relationship, reference evidence, and settlement
date. Then open **Human Review** and select a duplicate or close-margin case.
LedgerGuard deliberately presents it for review rather than inventing a match.

In **Model / Evaluation**, describe pairwise diagnostics separately from the
full reconciliation pipeline: the former measures candidate classification on
a held-out split; the latter measures actual record assignment and unresolved
decisions on the batch.
