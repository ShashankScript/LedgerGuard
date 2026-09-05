"""
investigate.py

Given one gateway transaction, recompute its candidate bank matches with full
feature breakdown, rule score, and (if a trained model is available) ML
probability — this powers the Transaction Investigator page so a judge can
click any transaction and see exactly why it was decided the way it was.

Deterministic and evidence-based: every number shown is recomputed from the
actual records, nothing is invented or looked up from a cache that could go
stale.
"""

import numpy as np
from matching import rule_matcher as matcher
from matching import blocking
FEATURE_NAMES = matcher.MODEL_FEATURE_NAMES


def investigate_transaction(gateway_record, bank_records, model=None, scaler=None,
                             ml_threshold=None, rule_threshold=None):
    rule_threshold = rule_threshold if rule_threshold is not None else matcher.FUZZY_CONFIDENCE_THRESHOLD

    bank_index = blocking.build_bank_index(bank_records)
    candidates = blocking.candidates_for(gateway_record, bank_index)

    rows = []
    for b in candidates:
        features = matcher.extract_features(gateway_record, b)
        rule_score = matcher.rule_based_score(features)
        ml_prob = None
        if model is not None and scaler is not None:
            X = np.array([[features[f] for f in FEATURE_NAMES]])
            ml_prob = float(model.predict_proba(scaler.transform(X))[:, 1][0])
        rows.append({
            "bank_txn_id": b["bank_txn_id"],
            "bank_reference": b["reference"],
            "bank_amount": b["amount"],
            "bank_date": b["date"],
            "ref_similarity": round(features["ref_similarity"], 3),
            "amount_abs_diff": round(features["amount_abs_diff"], 2),
            "amount_relationship_score": round(features["amount_relationship_score"], 3),
            "amount_relationship": matcher.amount_relationship(gateway_record, b)[0],
            "date_diff_days": int(features["date_diff_days"]),
            "rule_score": round(rule_score, 3),
            "ml_probability": round(ml_prob, 3) if ml_prob is not None else None,
        })

    rows.sort(key=lambda r: r["rule_score"], reverse=True)

    if not rows:
        decision, reason = "UNRESOLVED", "No candidates found within the blocking window (date/amount range)."
    elif rows[0]["rule_score"] >= rule_threshold:
        decision = "AUTO_MATCHED"
        reason = f"Top candidate scored {rows[0]['rule_score']} which clears the {rule_threshold} threshold."
        # Flag ambiguity even on an auto-match: if a close second candidate exists
        if len(rows) > 1 and (rows[0]["rule_score"] - rows[1]["rule_score"]) < 0.05:
            decision = "AUTO_MATCHED_AMBIGUOUS"
            reason += f" However, candidate {rows[1]['bank_txn_id']} scored nearly as close " \
                      f"({rows[1]['rule_score']}) — flagged for review rather than fully automatic."
    else:
        decision = "HUMAN_REVIEW"
        reason = f"Best candidate ({rows[0]['bank_txn_id']}) only scored {rows[0]['rule_score']}, " \
                  f"below the {rule_threshold} threshold."

    return {
        "gateway_record": gateway_record,
        "candidates": rows,
        "decision": decision,
        "reason": reason,
        "threshold": rule_threshold,
    }
