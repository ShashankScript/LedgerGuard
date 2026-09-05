"""
build_training_data.py

Turns every candidate pair that blocking.py would consider into a labeled
training row: features (from matcher.extract_features) + label (1 if this
pair is a true match per ground_truth.csv, else 0).

CRITICAL: ground_truth.csv is used HERE ONLY, to build labels for training.
It is never available to the matcher or the ML model at inference time — at
inference time the model only sees features computed from the two raw CSVs,
exactly like the rule-based matcher does.

This also gives us realistic NEGATIVE examples "for free": every candidate
pair that blocking considers but ISN'T the true match (e.g. the ambiguous
decoy sitting right next to a real match) becomes a hard negative, which is
exactly what teaches the model to do better than the rule-based formula.
"""

import csv
from pathlib import Path
from . import blocking
from . import rule_matcher as matcher

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELED_PAIRS_PATH = str(_ROOT / "outputs" / "labeled_pairs.csv")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_labeled_pairs(gateway_path=None, bank_path=None, ground_truth_path=None,
                         gateway_records=None, bank_records=None, ground_truth_records=None):
    gateway_path = gateway_path or matcher.DEFAULT_GATEWAY_PATH
    bank_path = bank_path or matcher.DEFAULT_BANK_PATH
    ground_truth_path = ground_truth_path or str(_ROOT / "data" / "sample" / "ground_truth.csv")

    gateway = gateway_records if gateway_records is not None else load_csv(gateway_path)
    bank = bank_records if bank_records is not None else load_csv(bank_path)
    gateway = [dict(row, amount=float(row["amount"])) for row in gateway]
    bank = [dict(row, amount=float(row["amount"])) for row in bank]

    true_gateway_to_bank = {}
    gt_rows = ground_truth_records if ground_truth_records is not None \
        else list(csv.DictReader(open(ground_truth_path, newline="")))
    for row in gt_rows:
        if row["gateway_txn_id"] and row["bank_txn_id"]:
            true_gateway_to_bank[row["gateway_txn_id"]] = row["bank_txn_id"]

    bank_index = blocking.build_bank_index(bank)

    rows = []
    for g in gateway:
        candidates = blocking.candidates_for(g, bank_index)
        true_bank_id = true_gateway_to_bank.get(g["gateway_txn_id"])
        for b in candidates:
            features = matcher.extract_features(g, b)
            label = 1 if (true_bank_id is not None and b["bank_txn_id"] == true_bank_id) else 0
            rows.append({
                "gateway_txn_id": g["gateway_txn_id"],
                "bank_txn_id": b["bank_txn_id"],
                **features,
                "label": label,
            })
    return rows


def save_labeled_pairs(rows, path=None):
    path = path or DEFAULT_LABELED_PAIRS_PATH
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run via: python -m matching.build_training_data   (from the project root)
    rows = build_labeled_pairs()
    save_labeled_pairs(rows)
    n_pos = sum(r["label"] for r in rows)
    print(f"Built {len(rows)} candidate pairs from blocking ({n_pos} positive, {len(rows)-n_pos} negative)")
