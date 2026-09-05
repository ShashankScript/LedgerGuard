# PROJECT_AUDIT.md — LedgerGuard Baseline Audit

Audit of the existing reconciliation project before any redesign. No code changed
as part of writing this file.

## Current architecture

```
generate_data.py  → gateway_transactions.csv, bank_statement.csv, ground_truth.csv
matcher.py        → match_results.csv   (reads only the two transaction CSVs)
evaluate.py        → exception_report.csv, printed metrics (reads ground_truth.csv
                      SEPARATELY from the matcher — matcher never sees it)
sweep.py           → threshold sweep across matcher.FUZZY_CONFIDENCE_THRESHOLD
```

Two-stage matcher:
- **Stage 1 (exact):** normalize reference (strip punctuation, uppercase), require
  amount within ₹5 and date within 3 days.
- **Stage 2 (fuzzy):** for unmatched records, compare against every remaining
  candidate, score on `0.6*ref_similarity + 0.25*amount_score + 0.15*date_score`
  (difflib.SequenceMatcher for string similarity), accept if score ≥ threshold.

## Strengths

- **Ground truth is properly isolated.** `matcher.py` only ever reads the two
  transaction files, never `ground_truth.csv`. Evaluation happens in a separate
  module that loads truth independently. No leakage into the matcher itself.
- **Metrics are computed against ground truth, not self-reported.** Accuracy,
  false-match rate, and match rate are all measured by comparing matcher output
  to known-correct pairs, not just "% of records the matcher touched."
- **Exceptions have real reasons**, not just a dump of unmatched IDs — each
  unresolved record states why (no candidate in window vs. below-threshold score).
- **A threshold sweep already exists** and was used to move the default from an
  arbitrary 0.72 to a data-justified 0.85 (peak accuracy + minimum false-match
  rate simultaneously — see `sweep.py` output).
- **Reproducible generation** via a fixed random seed.

## Weaknesses / bugs found

1. **[FIXED during this session] Match-rate calculation bug.** The original
   `evaluate.py` computed match rate from *any* gateway ID appearing in the
   output file, including ones marked `unresolved`. This made the metric
   meaningless — it measured "did this ID appear in the results" rather than
   "did we actually find its match." Fixed to only count IDs from
   `made_matches` (exact/fuzzy), not all results. **Lesson for the audit:** the
   original number (1.0 / 100%) was silently wrong for two full runs before
   this was caught — worth mentioning in an interview as an example of catching
   your own measurement bugs.

2. **Dataset is too small for the spec's evaluation bar.** Currently ~91
   gateway + 85 bank records. The spec (and generally, credible evaluation)
   wants 500+ for the main run, ideally scalable to 5,000+. At 90 records the
   O(n×m) fuzzy pass is fine; at 5,000 it will not be — see #3.

3. **No blocking / candidate restriction — fuzzy pass is O(n×m).** Every
   unmatched gateway record is compared against every remaining bank record.
   Fine at ~90 records (~90×85 ≈ 7,650 comparisons). At 5,000 records this
   becomes ~25M comparisons of string-similarity calls, which will be slow.
   Needs a blocking step (e.g. bucket by date window and amount band first,
   only fuzzy-score within-bucket candidates) before scaling the dataset.

4. **The "fuzzy" scorer is a hand-weighted formula, not ML.** The 0.6/0.25/0.15
   weights were chosen by intuition, not fit to data. This is fine as a
   rule-based baseline, but it should never be described as "AI" or "ML" — it's
   a scoring heuristic. The spec explicitly warns against this exact
   mislabeling, and the current README already avoids calling it AI, which is
   good — but there is currently no actual ML component to compare it against.

5. **No train/validation/test split — there's nothing to leak yet, but also
   nothing learned.** Because there's no trained model, there's no leakage risk
   today, but also no way to claim "ML-assisted matching," which the spec
   wants as a genuine comparison arm (baseline vs. rule/fuzzy vs. ML).

6. **Confidence scores are not calibrated probabilities.** The `confidence`
   field in `match_results.csv` is the same ad hoc weighted combination used
   for the accept/reject decision — it has no probabilistic interpretation
   ("0.85 confidence" doesn't mean "85% chance this is correct"). No
   calibration curve or Brier score currently exists.

7. **Data generator lacks several realistic record types.** Currently: exact
   matches, fee-adjusted matches, reference mangling, date lag, gateway-only,
   bank-only, and injected duplicates. Missing: refunds, adjustments, ambiguous
   multi-candidate cases (two bank records that both look plausible for one
   gateway record — currently the generator doesn't deliberately create these,
   though the matcher may encounter them incidentally).

8. **No difficulty-stratified evaluation.** Metrics are currently reported as
   one aggregate number. The spec wants performance broken out separately for
   exact / fuzzy / duplicate / missing-record / fee-adjusted subsets, which the
   current `evaluate.py` doesn't do — `ground_truth.csv` already has a
   `match_type` column that makes this straightforward to add.

9. **No audit trail with full decision provenance.** `match_results.csv` has
   the decision and confidence, but not: which candidates were considered and
   rejected, model/threshold version, or a timestamp. Not needed for the
   current CLI-only pipeline, but required before any human-review or dashboard
   layer is added.

10. **Nothing beyond the matcher yet** — no ML classifier, no anomaly
    detection, no LLM investigation agent, no human-in-the-loop mechanism, no
    dashboard, no Razorpay API integration. This is expected at this stage,
    not a "bug," just noting current scope for the recommendation below.

## Recommended architecture (target state)

```
config.py              → thresholds, paths, model version, random seed
data/
  generate_data.py      → scaled to 500+ (parameterized up to 5,000), adds
                           refunds/adjustments and deliberate ambiguous cases
matching/
  normalize.py           → shared normalization (already exists inline, extract it)
  blocking.py            → candidate generation (date+amount buckets) — NEW,
                            required before scaling the dataset
  rule_matcher.py         → today's exact + weighted-fuzzy logic, relabeled
                            explicitly as the rule-based baseline
  ml_matcher.py           → NEW: logistic regression / gradient boosting on
                            [ref_similarity, amount_diff, date_diff, ...],
                            proper train/val/test split, compared honestly
                            against the rule baseline
  calibration.py          → NEW: calibration curve + Brier score on the ML
                            model's output probabilities; sets the 3
                            confidence bands (auto / investigate / human-review)
anomaly/
  detector.py              → NEW: simple statistical baseline first (z-score on
                            amount/frequency), then Isolation Forest compared
                            against it
agent/
  investigate.py           → NEW: LLM tool-calling agent, invoked ONLY for
                            medium-confidence cases, tools scoped to
                            read/compare operations, never does arithmetic itself
audit/
  audit_log.py             → NEW: structured per-decision log (candidates
                            considered, scores, threshold/model version, human
                            override if any, timestamp)
evaluation/
  evaluate.py               → extend today's version with per-subset breakdown
                            and precision/recall/F1/automation-rate
  stress_test.py            → NEW: sweep corruption level / duplicate density /
                            volume and show degradation curves
dashboard/
  app.py                    → Streamlit, sections per spec
docs/
  README.md, ARCHITECTURE.md, EVALUATION.md, LIMITATIONS.md
```

## Prioritized improvement list

Ranked by "value per hour of work," not by spec section order:

| # | Item | Why it matters | Est. effort |
|---|---|---|---|
| 1 | Fix match-rate bug (done) | Wrong metrics are worse than no metrics | done |
| 2 | Scale dataset to 500+, add blocking | Spec's explicit minimum; blocking is required to keep runtime sane at that scale | Medium |
| 3 | Per-subset evaluation breakdown | Cheap to add (data already has `match_type`), high credibility payoff | Small |
| 4 | Genuine ML classifier + honest baseline comparison | This is the difference between "we filtered/scored" and "we built ML" — directly addresses the spec's #4 and #21 ("why does this need AI?") | Medium-Large |
| 5 | Calibration + 3-band confidence | Makes "confidence" mean something, enables the human-in-the-loop story cleanly | Medium |
| 6 | Audit trail | Needed before dashboard/human-review has any credibility | Small-Medium |
| 7 | Human-in-the-loop review (even CLI-level) | Core differentiator per spec's philosophy ("safe automation, not blind automation") | Medium |
| 8 | Anomaly detection (statistical baseline first) | Real but scoped — don't over-invest here | Small-Medium |
| 9 | Streamlit dashboard | High demo value, but only pays off once 1–8 exist to show | Medium-Large |
| 10 | LLM investigation agent | Highest risk/effort item — a bad implementation (hallucinated evidence, LLM doing arithmetic) actively hurts credibility more than no agent at all | Large |
| 11 | Razorpay sandbox integration | Spec says only add if it adds genuine value — needs a concrete justification first | Medium, contingent |
| 12 | Robustness/stress-test mode | Good evidence of engineering rigor, cheap once blocking exists | Small |

## Honest scope reality check

This full spec — as written — is a multi-week project for an experienced
engineer, not a one-week solo buildathon build. Items 4, 5, 9, and 10 alone
(genuine ML model with proper evaluation, calibration, a real dashboard, and a
guardrailed LLM agent) are each independently a few days of focused work if
done properly rather than superficially — and "superficially" is explicitly
what the spec tells us to avoid.

I'd rather tell you this now than have you find it out on day 5. The good
news: items 1–3 and 6–8 and 12 are individually cheap and each directly
strengthens your ability to defend the project. Items 4, 5, 9, 10 are where
the real judgment calls are — doing 2 of them well beats doing all 4 shallowly.
# Reconciliation audit update

The v3 matcher was checked against the deterministic `n=500, seed=42` batch.
Ground truth remains isolated to training and evaluation. Inference imports no
ground-truth file and `run_matcher` accepts no truth argument. The old baseline
reported 79.0% match precision and 21.0% false-match rate, but incorrectly
scored fee/refund pairs as false even when their bank ids were selected. The
current evaluator reports pair precision/recall for every legitimate pair type
and separately credits unresolved expected-unmatched records.

The primary safety trade-off is lower automation for duplicates/close cases in
return for substantially fewer automated false matches. This is appropriate for
financial reconciliation; unresolved cases are visible and auditable.

| Full pipeline metric (n=500, seed=42) | Before | After |
| --- | ---: | ---: |
| Reported match precision | 79.0% | 99.0% |
| False-match rate | 21.0% | 1.0% |
| True-pair match rate | 89.4% | 78.1% |
| Automation rate | not reported | 66.6% |
| Unresolved / human-review output rows | not reported | 324 |

The prior figures only counted `true_match` as a correct automatic pairing,
so fee/refund outcomes were unfairly treated as failures. The after precision,
recall, and F1 (99.0%, 78.1%, 87.3%) use every legitimate bank-pair type.
For that reason the two precision definitions are not perfectly comparable;
the table intentionally retains the regression in true-pair recall caused by
the new duplicate/ambiguity safety policy.

| Subset | Before | After correct rate |
| --- | ---: | ---: |
| true_match | not separately supplied | 81.9% |
| fee_adjusted_match | 10.2% | 59.2% |
| refund_match | 0.0% | 39.3% |
| ambiguous_true_match | 98.0% | 98.0% |
| gateway_only | 100.0% | 96.3% |
| duplicate_gateway_only | 61.1% | 100.0% |
