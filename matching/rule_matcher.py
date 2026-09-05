"""Inference-only, conservative rule reconciliation.

Rules recognize observable exact bookings, small gateway-fee settlements, and
bank rows explicitly marked as post-refund settlements. Ambiguity is escalated
to a human instead of being converted into a financial guess.
"""
import csv
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from .normalize import normalize_ref, date_diff_days
from . import blocking

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GATEWAY_PATH = str(_ROOT / "data" / "sample" / "gateway_transactions.csv")
DEFAULT_BANK_PATH = str(_ROOT / "data" / "sample" / "bank_statement.csv")
DEFAULT_RESULTS_PATH = str(_ROOT / "outputs" / "match_results.csv")
AMOUNT_TOLERANCE_EXACT = 5.00
FUZZY_CONFIDENCE_THRESHOLD = 0.85
AMBIGUITY_MARGIN = 0.06
# Shared order for every model feature vector. Keeping it with feature
# extraction prevents the investigator and model scaler from drifting apart.
MODEL_FEATURE_NAMES = ("ref_similarity", "ref_exact", "amount_abs_diff",
                       "amount_pct_diff", "amount_lower_ratio", "date_diff_days",
                       "amount_relationship_score", "is_fee_adjusted", "is_refund_adjusted")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def amount_relationship(gateway_record, bank_record):
    """Classify the visible amount evidence.

    Exact means within ₹5. Fee settlement accepts a 0.3–3% deduction (the
    generator's documented fee model is 0.5–2.5%). A 5–60% deduction is a
    refund only when the observable bank record type says so. Everything else
    is a mismatch, not a special case.
    """
    g, b = float(gateway_record["amount"]), float(bank_record["amount"])
    diff = g - b
    lower_ratio = diff / g if g else 0.0
    record_type = str(bank_record.get("record_type", "")).lower()
    if abs(diff) <= AMOUNT_TOLERANCE_EXACT:
        return "exact", 1.0
    if 0.003 <= lower_ratio <= 0.03:
        return "fee_adjusted", max(0.80, 1.0 - abs(lower_ratio - 0.015) / 0.03)
    if "refund" in record_type and 0.05 <= lower_ratio <= 0.60:
        return "refund_adjusted", max(0.82, 1.0 - abs(lower_ratio - 0.30) / 0.60)
    return "other_adjustment", 0.0


def extract_features(gateway_record, bank_record):
    g_ref, b_ref = normalize_ref(gateway_record["reference"]), normalize_ref(bank_record["reference"])
    g_amount, b_amount = float(gateway_record["amount"]), float(bank_record["amount"])
    relationship, relationship_score = amount_relationship(gateway_record, bank_record)
    amount_abs_diff = abs(g_amount - b_amount)
    amount_pct_diff = amount_abs_diff / g_amount if g_amount else 1.0
    return {
        "ref_similarity": SequenceMatcher(None, g_ref, b_ref).ratio(),
        "ref_exact": 1.0 if g_ref == b_ref else 0.0,
        "amount_abs_diff": amount_abs_diff,
        "amount_pct_diff": amount_pct_diff,
        "amount_lower_ratio": (g_amount - b_amount) / g_amount if g_amount else 0.0,
        "date_diff_days": float(date_diff_days(gateway_record["date"], bank_record["date"])),
        "amount_relationship_score": relationship_score,
        "is_fee_adjusted": 1.0 if relationship == "fee_adjusted" else 0.0,
        "is_refund_adjusted": 1.0 if relationship == "refund_adjusted" else 0.0,
    }


def rule_based_score(features):
    """Reference-led score with relationship and settlement-date evidence."""
    date_score = max(0.0, 1 - features["date_diff_days"] / 4)
    return (0.62 * features["ref_similarity"] + 0.25 * features["amount_relationship_score"] +
            0.13 * date_score)


def _duplicate_gateway_ids(gateway):
    keys = [(normalize_ref(g["reference"]), round(float(g["amount"]), 2), g["date"]) for g in gateway]
    counts = Counter(keys)
    return {g["gateway_txn_id"] for g, key in zip(gateway, keys) if counts[key] > 1}


def run_matcher(gateway_path=None, bank_path=None, threshold=None, gateway_records=None, bank_records=None):
    threshold = threshold if threshold is not None else FUZZY_CONFIDENCE_THRESHOLD
    gateway = gateway_records if gateway_records is not None else load_csv(gateway_path or DEFAULT_GATEWAY_PATH)
    bank = bank_records if bank_records is not None else load_csv(bank_path or DEFAULT_BANK_PATH)
    gateway = [dict(row, amount=float(row["amount"])) for row in gateway]
    bank = [dict(row, amount=float(row["amount"])) for row in bank]
    bank_index, matched_bank_ids, results = blocking.build_bank_index(bank), set(), []
    duplicate_ids = _duplicate_gateway_ids(gateway)

    for g in gateway:
        if g["gateway_txn_id"] in duplicate_ids:
            results.append({"gateway_txn_id": g["gateway_txn_id"], "bank_txn_id": "", "match_type": "human_review", "confidence": 0.0,
                            "reason": "Duplicate gateway record has indistinguishable reference, amount, and date; no bank record was claimed."})
            continue
        scored = []
        for b in blocking.candidates_for(g, bank_index):
            if b["bank_txn_id"] not in matched_bank_ids:
                features = extract_features(g, b)
                scored.append((rule_based_score(features), b, features))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            results.append({"gateway_txn_id": g["gateway_txn_id"], "bank_txn_id": "", "match_type": "unresolved", "confidence": 0.0,
                            "reason": "No candidate found within amount/date or refund reference/date window."})
            continue
        score, best_bank, features = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if score >= threshold and score - second_score >= AMBIGUITY_MARGIN:
            matched_bank_ids.add(best_bank["bank_txn_id"])
            relationship, _ = amount_relationship(g, best_bank)
            method = "exact" if relationship == "exact" and features["ref_exact"] else relationship
            results.append({"gateway_txn_id": g["gateway_txn_id"], "bank_txn_id": best_bank["bank_txn_id"], "match_type": method,
                            "confidence": round(score, 3), "reason": f"{relationship.replace('_', ' ')} amount relationship; reference and date evidence passed auto-match policy."})
        elif score >= threshold:
            results.append({"gateway_txn_id": g["gateway_txn_id"], "bank_txn_id": "", "match_type": "human_review", "confidence": round(score, 3),
                            "reason": f"Top candidate margin {score - second_score:.3f} is below {AMBIGUITY_MARGIN:.2f}; candidates are too close to choose safely."})
        else:
            results.append({"gateway_txn_id": g["gateway_txn_id"], "bank_txn_id": "", "match_type": "unresolved", "confidence": round(score, 3),
                            "reason": f"Best candidate scored {score:.3f}, below {threshold:.2f} auto-match threshold."})
    for b in bank:
        if b["bank_txn_id"] not in matched_bank_ids:
            results.append({"gateway_txn_id": "", "bank_txn_id": b["bank_txn_id"], "match_type": "unresolved", "confidence": 0.0,
                            "reason": "No gateway transaction claimed this bank record."})
    return results


def save_results(results, path=None):
    path = path or DEFAULT_RESULTS_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gateway_txn_id", "bank_txn_id", "match_type", "confidence", "reason"])
        writer.writeheader(); writer.writerows(results)


if __name__ == "__main__":
    save_results(run_matcher())
