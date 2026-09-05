# ARCHITECTURE.md — LedgerGuard

## Pipeline

```
data/generate_data.py ──┐
                         ├──> matching/rule_matcher.py (blocking + exact + fuzzy)
uploaded CSVs ───────────┘             │
                                        ├──> matching/build_training_data.py
                                        │      ──> matching/ml_matcher.py
                                        │      (labeled pairs from blocking +
                                        │       ground truth, then train +
                                        │       evaluate ML vs rule, honestly)
                                        │
                                        ├──> agent/investigate.py (per-transaction
                                        │      drill-down, recomputes features/scores
                                        │      live, never cached stale)
                                        │
                                        ├──> anomaly/detector.py (statistical flags:
                                        │      amount z-score, duplicate references)
                                        │
                                        └──> audit/audit_log.py (every decision +
                                               human override logged)

app.py orchestrates all of the above as the Streamlit presentation layer.
It does not reimplement any matching, scoring, or evaluation logic itself.
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `data/generate_data.py` | Synthetic data generation (scalable 50–5,000+ records), with hidden ground truth kept separate from the matcher inputs. Writes to `data/sample/`. |
| `matching/normalize.py` | Shared reference/date normalization, used by both the rule matcher and ML feature extraction. |
| `matching/blocking.py` | Candidate generation via date+amount bucketing — keeps matching fast at scale (O(n×m) → near-linear in practice). |
| `matching/rule_matcher.py` | Rule-based baseline: exact-match pass, then weighted fuzzy-score pass on blocked candidates. Also defines `extract_features()`, shared with the ML model for a fair comparison. |
| `matching/build_training_data.py` | Converts blocking's candidate pairs into labeled rows (using ground truth ONLY here, never fed to the matcher itself). |
| `matching/ml_matcher.py` | Trains a logistic regression classifier, group-split by gateway record to avoid leakage, threshold tuned on validation only, evaluated once on test. Reports precision/recall/F1/confusion matrix for both the ML model and the rule baseline on the identical test set. |
| `evaluation/evaluate.py` | Scores matcher output against ground truth; writes the exception report to `outputs/`. |
| `evaluation/sweep.py` | Sweeps the confidence threshold and reports the accuracy/false-match trade-off. |
| `agent/investigate.py` | Given one transaction, recomputes its full candidate list with feature breakdown, rule score, and ML probability — powers the drill-down UI. Deterministic, no LLM, no invented evidence. |
| `anomaly/detector.py` | Interpretable statistical flags (amount z-score outliers, duplicate references). Phrased as "unusual, needs review" — never labeled "fraud." |
| `audit/audit_log.py` | Append-only audit log in `outputs/audit_log.csv`: every automated decision and every human override, with timestamp, candidates considered, model version, threshold, and reason. |
| `app.py` | Streamlit orchestration/presentation only. |

## Data integrity guarantees

- **Ground truth isolation:** `matching/rule_matcher.py` never reads
  `ground_truth.csv` / `ground_truth_records`. It is used only in
  `matching/build_training_data.py` (to build labels) and `evaluation/evaluate.py`
  (to score results) — both clearly separate call sites from the matcher itself.
  Verified by `tests/test_core.py::TestGroundTruthIsolation`.
- **No fabricated metrics:** if ground truth isn't available (uploaded data
  has none), the app shows "Not evaluated" rather than inventing a number.
- **Leakage-aware ML split:** train/val/test split is grouped by
  `gateway_txn_id`, not by row, so different candidate pairs for the same
  transaction never cross split boundaries.
- **Threshold fairness:** the ML model's decision threshold is tuned on the
  validation set only, not the test set — same standard the rule-based
  threshold was held to via `evaluation/sweep.py`.

## Known limitations

See `LIMITATIONS.md` for the full, honest list.

## Reconciliation policy (v3)

- Ordinary payments use a bounded date/amount candidate route. The amount span
  scales to the documented 0.5–2.5% synthetic gateway-fee model rather than
  assuming every deduction fits one ₹500 bucket.
- A bank row whose observable `record_type` is `credit_after_refund` also uses
  a sparse refund date/reference route. This prevents a 10–50% post-refund
  deduction from being discarded before scoring; no label is consulted.
- Amount evidence is explicitly classified as exact, fee-adjusted,
  refund-adjusted, or other adjustment. Large differences without a refund
  record type are not auto-matched.
- Identical gateway reference/amount/date records and close top candidates are
  emitted as `human_review`. This conservative state is intentional: it avoids
  claiming a bank record when the evidence cannot distinguish alternatives.
