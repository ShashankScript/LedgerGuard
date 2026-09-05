# LIMITATIONS.md

Stated plainly, so nobody — including you, in an interview — is caught off
guard by a question this doesn't already answer.

## Data

- **Synthetic only.** `data/generate_data.py` produces realistic but
  artificial data. Real bank statement formats, real fee structures, and real
  reference-mangling patterns will differ from what this generator produces,
  likely by more than the tolerances tuned here assume.
- **Single currency, single merchant assumption.** No multi-currency, no
  multi-entity reconciliation.

## Matching

- **Two hand-picked matching approaches** (rule-based weighted formula, and a
  logistic regression classifier on 6 features). On the current synthetic
  dataset they are roughly tied (see Model / Evaluation page) — this is
  reported honestly rather than forced into a "ML wins" narrative.
- **Six features only.** Reference similarity, exact-reference flag, absolute
  amount difference, percentage amount difference, amount-lower ratio, date
  difference. No merchant/entity-name similarity, no transaction-type
  compatibility feature, no historical pattern features.
- **Blocking assumes settlement lag is bounded** (currently a 4-day window)
  and amounts fall within adjacent ₹500 buckets. A real system with longer
  settlement windows or wider fee variance would need these re-tuned.

## ML

- **Logistic regression only.** No gradient boosting / tree-based comparison
  was built — deliberately scoped down given time constraints (see
  PROJECT_AUDIT.md's prioritization reasoning) rather than added superficially.
- **No calibration.** The ML model's `predict_proba()` output is used as a
  ranking/threshold signal, not verified to be a calibrated probability (i.e.
  "70% confidence" is not verified to mean "correct 70% of the time"). No
  calibration curve or Brier score has been computed.
- **Threshold tuned on one validation split**, not cross-validated. With more
  time, k-fold cross-validation on the threshold choice would be more robust.

## Anomaly detection

- **Statistical baselines only** (z-score on amount, duplicate reference
  detection). No Isolation Forest or other trained anomaly model — this was
  intentionally kept simple and interpretable rather than added as an
  unexplained black box.
- **Not fraud detection.** Deliberately not labeled as such anywhere in the
  code or UI — flags are phrased as "unusual, needs review."

## AI agent

- **No LLM-based investigation agent is included.** The original spec called
  for one; it was explicitly deprioritized (see PROJECT_AUDIT.md) because a
  poorly-guardrailed agent that could hallucinate evidence would be worse for
  credibility than no agent at all, and building one properly (tool-scoped,
  evidence-grounded, tested against hallucination) was judged to need more
  time than was available. `agent/investigate.py` is fully deterministic —
  no LLM calls — and covers the "explain why" requirement without that risk.

## Human-in-the-loop

- **Session-scoped overrides.** Human review decisions are stored in
  Streamlit session state and reset if the app restarts. The permanent record
  is the audit log (`outputs/audit_log.csv`), which does persist, but the
  live "already reviewed" UI badges do not survive a restart.
- **No user authentication.** Any user of the running app can approve/reject
  any case; there's no reviewer identity tracked beyond "a human reviewed
  this."

## Evaluation

- **Difficulty-stratified breakdown is available via `ground_truth.csv`'s
  `match_type` column** but not yet surfaced as a dedicated report — the
  Model/Evaluation page shows aggregate metrics, not per-subset (exact vs.
  fuzzy vs. ambiguous vs. fee-adjusted) breakdowns.
- **No formal robustness/stress-test report** beyond confirming the pipeline
  runs correctly at 5,000 records (see PROJECT_AUDIT.md) — a systematic sweep
  of corruption level / duplicate density / volume with degradation curves
  was scoped as a stretch item and not built in this session.

## Integration

- **No live Razorpay API integration.** All data is synthetic or
  user-uploaded CSVs; nothing connects to a real payment gateway or bank feed.

## Testing

- **`tests/test_core.py` covers the core pipeline** (data generation,
  blocking, matching, ground-truth isolation, ML training) but not the
  Streamlit UI layer itself — Streamlit apps are hard to unit test
  meaningfully without a browser-driving tool, which wasn't set up here.
# Reconciliation limitations

Refunds previously failed because the old amount-only blocker admitted only
nearby ₹500 bands, while synthetic refunds reduce settlement by 10–50%. Fee
adjustments also failed at larger values when a legitimate percentage fee
crossed that fixed band, and the old scorer gave no amount credit beyond 3%.

The current matcher uses bank `record_type`, reference, amount relationship,
and date only. It does not infer refunds from arbitrary large discrepancies.
It cannot safely resolve a refund if a real bank export omits both a reliable
reference relationship and a refund/credit marker. Those cases remain for
human review.

`HUMAN_REVIEW` is a deliberate outcome, not a failed match: indistinguishable
duplicate gateway records and candidates with too-small a score margin are
not assigned a bank record automatically. A reviewer must choose, reject, or
keep the case under investigation.
