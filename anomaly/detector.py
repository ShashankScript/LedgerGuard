"""
anomaly.py

Interpretable, statistical anomaly flags — not "fraud detection." Every flag
is phrased as "unusual, needs review," never as an accusation, and every flag
states WHY it was raised so it's checkable, not a black box.
"""

import statistics
from collections import Counter


def detect_amount_anomalies(records, amount_key="amount", z_threshold=3.0):
    """Flag records whose amount is a statistical outlier vs. the rest of the batch."""
    amounts = [r[amount_key] for r in records]
    if len(amounts) < 3:
        return []
    mean = statistics.mean(amounts)
    stdev = statistics.pstdev(amounts) or 1e-9
    flagged = []
    for r in records:
        z = (r[amount_key] - mean) / stdev
        if abs(z) >= z_threshold:
            flagged.append({**r, "anomaly_reason": f"Unusual transaction amount (z-score {z:.2f})", "z_score": round(z, 2)})
    return flagged


def detect_duplicate_references(records, ref_key="reference"):
    """Flag records whose reference string appears more than once in the batch."""
    counts = Counter(r[ref_key] for r in records)
    dupes = {ref: c for ref, c in counts.items() if c > 1}
    flagged = []
    for r in records:
        if r[ref_key] in dupes:
            flagged.append({**r, "anomaly_reason": f"Reference '{r[ref_key]}' used {dupes[r[ref_key]]} times in this batch"})
    return flagged


def run_all(records, amount_key="amount", ref_key="reference"):
    amount_flags = detect_amount_anomalies(records, amount_key)
    dup_flags = detect_duplicate_references(records, ref_key)
    return {"amount_anomalies": amount_flags, "duplicate_references": dup_flags}
