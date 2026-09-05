"""Inference-only candidate generation for reconciliation.

Payments and settlements normally belong in a narrow amount/date window. A
post-refund settlement does not: its visible bank amount is the original
payment less a partial refund. This module combines a fast amount/date index
with a reference/date index for observable refund records; it never reads
labels or ground truth.
"""
from collections import defaultdict
from datetime import datetime, timedelta
import math
from .normalize import normalize_ref

DATE_WINDOW_DAYS = 4
AMOUNT_BAND_SIZE = 500.0


def amount_bucket(amount):
    return int(float(amount) // AMOUNT_BAND_SIZE)


def reference_key(reference):
    """Tolerant key for prefix-less/truncated transaction references."""
    digits = "".join(ch for ch in normalize_ref(reference) if ch.isdigit())
    return digits[:5] if len(digits) >= 5 else normalize_ref(reference)[:8]


def build_bank_index(bank_records):
    """Build amount/date and refund-reference/date indexes in one pass."""
    amount_index, refund_ref_index, refund_by_date = defaultdict(list), defaultdict(list), defaultdict(list)
    for b in bank_records:
        d = datetime.strptime(b["date"], "%Y-%m-%d")
        amount_index[(d, amount_bucket(b["amount"]))].append(b)
        if "refund" in str(b.get("record_type", "")).lower():
            refund_ref_index[(d, reference_key(b.get("reference", "")))].append(b)
            refund_by_date[d].append(b)
    return {"amount": amount_index, "refund_ref": refund_ref_index, "refund_by_date": refund_by_date}


def candidates_for(gateway_record, bank_index):
    """Return viable candidates from observable amount, date, type and ref data.

    Standard settlement candidates use +/- one ₹500 amount band. Explicit
    post-refund bank records additionally use a tolerant reference/date route,
    deliberately bypassing amount blocking because the refund can be 10–50%.
    """
    g_date = datetime.strptime(gateway_record["date"], "%Y-%m-%d")
    g_bucket = amount_bucket(gateway_record["amount"])
    g_ref_key = reference_key(gateway_record.get("reference", ""))
    seen_ids, candidates = set(), []
    for day_offset in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1):
        d = g_date + timedelta(days=day_offset)
        # A fee can be 2.5% of a large payment, which is more than one ₹500
        # bucket. Bound the range by the documented 3% operational fee cap.
        fee_bands = max(1, math.ceil(float(gateway_record["amount"]) * 0.03 / AMOUNT_BAND_SIZE))
        for bucket in range(g_bucket - fee_bands, g_bucket + fee_bands + 1):
            for b in bank_index["amount"].get((d, bucket), []):
                if b["bank_txn_id"] not in seen_ids:
                    seen_ids.add(b["bank_txn_id"]); candidates.append(b)
        for b in bank_index["refund_ref"].get((d, g_ref_key), []):
            if b["bank_txn_id"] not in seen_ids:
                seen_ids.add(b["bank_txn_id"]); candidates.append(b)
        # Refund rows are a sparse, explicitly typed population. Date-scanning
        # just these rows admits a reference typo without widening payment
        # blocking to the entire bank statement.
        for b in bank_index["refund_by_date"].get(d, []):
            if b["bank_txn_id"] not in seen_ids:
                seen_ids.add(b["bank_txn_id"]); candidates.append(b)
    return candidates
