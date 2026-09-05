# PROJECT_STRUCTURE.md

```
LedgerGuard/
├── app.py                  Entry point. Streamlit UI — orchestration only,
│                            no business logic lives here.
├── requirements.txt         Every Python dependency.
├── README.md                 Setup + Windows/VS Code instructions.
├── ARCHITECTURE.md            How the pipeline fits together.
├── DEMO_GUIDE.md               Suggested walkthrough for a demo/interview.
├── UI_README.md                 What each dashboard page does.
├── LIMITATIONS.md                Honest list of what this does NOT do.
├── PROJECT_AUDIT.md                Technical audit of the original build.
├── PROJECT_STRUCTURE.md              This file.
├── .gitignore
│
├── data/
│   ├── generate_data.py      Synthetic data generator (scalable, seeded,
│   │                          writes to data/sample/ by default).
│   └── sample/                Generated CSVs land here when run via CLI.
│                               Gitignored — regenerate anytime.
│
├── matching/
│   ├── normalize.py            Shared string/date normalization helpers.
│   ├── blocking.py               Candidate generation (date+amount buckets)
│   │                             — keeps matching fast at scale.
│   ├── rule_matcher.py             The rule-based baseline matcher. Also
│   │                               defines extract_features(), shared with
│   │                               the ML model for a fair comparison.
│   ├── build_training_data.py       Turns blocking's candidates into labeled
│   │                                 training rows using ground truth —
│   │                                 ONLY used here, never by the matcher.
│   └── ml_matcher.py                  Trains + evaluates the ML classifier
│                                       against the rule baseline, honestly.
│
├── evaluation/
│   ├── evaluate.py             Scores matcher output against ground truth;
│   │                            writes the exception report.
│   └── sweep.py                  Sweeps the matcher's confidence threshold
│                                  to justify the production value.
│
├── anomaly/
│   └── detector.py             Statistical anomaly flags (amount z-score,
│                                duplicate references). Never labeled "fraud."
│
├── agent/
│   └── investigate.py          Per-transaction drill-down: recomputes every
│                                candidate + feature + score live. No LLM,
│                                nothing invented — fully deterministic.
│
├── audit/
│   └── audit_log.py            Append-only audit trail: every automated
│                                decision and human override, with reasons.
│
├── tests/
│   └── test_core.py            Data generation, blocking, matching, ground-
│                                truth isolation, and ML pipeline tests.
│
└── outputs/                    match_results.csv, exception_report.csv,
                                 labeled_pairs.csv, audit_log.csv all land
                                 here. Gitignored — regenerate anytime.
```

## Why this layout

Each folder is one responsibility, matching the finance-ops loop the product
implements: **ingest (data/) → match (matching/) → evaluate (evaluation/) →
flag (anomaly/) → explain (agent/) → record (audit/)**, with `app.py` as the
thin layer that wires them together for the UI. Nothing in `app.py`
reimplements logic that belongs in a package — if you're looking for how a
number is computed, it's in the package, not in `app.py`.
