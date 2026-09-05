"""
generate_data.py (v2 — scaled)

Produces a configurable-size synthetic reconciliation batch. Default is 500
records; safe up to 5,000+ (generation itself is O(n), the matcher's blocking
step is what keeps matching fast at that scale, not this file).

Record types generated (each gateway record's fate is decided independently):
  - true_match             : exists cleanly (or messily) on both sides
  - fee_adjusted_match      : true match, but bank amount = gateway amount - fee
  - refund_match            : a matched pair where a partial refund was later
                              issued, so the "settled" amount differs from the
                              original charge for a legitimate business reason
                              (not a data error)
  - gateway_only            : pending settlement, no bank record yet
  - bank_only               : unrelated bank activity (charges, other transfers)
  - duplicate_gateway_only  : near-identical extra gateway record, no bank match
  - ambiguous               : TWO bank records that both plausibly match one
                              gateway record (similar amount/date/reference) —
                              deliberately hard, this is what should separate a
                              good matcher from a naive one

ground_truth.csv is written separately and must only be used for training
labels / evaluation, never as a matcher input feature.
"""

import random
import string
import csv
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta

SAMPLE_DIR = Path(__file__).resolve().parent / "sample"
START_DATE = datetime(2026, 6, 1)


def rand_amount(rng):
    return round(rng.uniform(150, 45000), 2)


def rand_ref(txn_id):
    return f"TXN-{txn_id:06d}"


def mangle_ref(ref, rng):
    choice = rng.random()
    if choice < 0.35:
        return ref
    if choice < 0.55:
        return ref.replace("TXN-", "")
    if choice < 0.70:
        return ref.replace("-", "")
    if choice < 0.85:
        return ref[:-2]
    ref = list(ref)
    pos = rng.randrange(len(ref))
    if ref[pos].isdigit():
        ref[pos] = rng.choice(string.digits)
    return "".join(ref)


def rand_date_offset(base_date, rng, max_days=25):
    return base_date + timedelta(days=rng.randint(0, max_days))


def build_dataset(n_records=500, seed=42):
    rng = random.Random(seed)

    p_true_match = 0.55
    p_fee_adjusted = 0.10
    p_refund = 0.05
    p_gateway_only = 0.10
    p_bank_only = 0.08
    p_duplicate = 0.04
    # remainder -> ambiguous

    gateway_rows = []
    bank_rows = []
    ground_truth = []

    txn_counter = 1

    def new_ids():
        nonlocal txn_counter
        gid, bid = f"GW{txn_counter:06d}", f"BK{txn_counter:06d}"
        txn_counter += 1
        return gid, bid

    n_generated = 0
    while n_generated < n_records:
        roll = rng.random()
        gw_date = rand_date_offset(START_DATE, rng)
        amount = rand_amount(rng)
        ref = rand_ref(txn_counter)

        if roll < p_true_match:
            gid, bid = new_ids()
            bank_date = gw_date + timedelta(days=rng.choice([0, 0, 0, 1, 2]))
            gateway_rows.append({"gateway_txn_id": gid, "reference": ref, "amount": amount,
                                  "date": gw_date.strftime("%Y-%m-%d"), "record_type": "payment"})
            bank_rows.append({"bank_txn_id": bid, "reference": mangle_ref(ref, rng), "amount": amount,
                               "date": bank_date.strftime("%Y-%m-%d"), "record_type": "credit"})
            ground_truth.append({"gateway_txn_id": gid, "bank_txn_id": bid, "match_type": "true_match"})

        elif roll < p_true_match + p_fee_adjusted:
            gid, bid = new_ids()
            fee = round(amount * rng.uniform(0.005, 0.025), 2)
            bank_amount = round(amount - fee, 2)
            bank_date = gw_date + timedelta(days=rng.choice([0, 1, 1, 2]))
            gateway_rows.append({"gateway_txn_id": gid, "reference": ref, "amount": amount,
                                  "date": gw_date.strftime("%Y-%m-%d"), "record_type": "payment"})
            bank_rows.append({"bank_txn_id": bid, "reference": mangle_ref(ref, rng), "amount": bank_amount,
                               "date": bank_date.strftime("%Y-%m-%d"), "record_type": "credit"})
            ground_truth.append({"gateway_txn_id": gid, "bank_txn_id": bid, "match_type": "fee_adjusted_match"})

        elif roll < p_true_match + p_fee_adjusted + p_refund:
            gid, bid = new_ids()
            refund_amount = round(amount * rng.uniform(0.1, 0.5), 2)
            bank_amount = round(amount - refund_amount, 2)
            bank_date = gw_date + timedelta(days=rng.randint(1, 4))
            gateway_rows.append({"gateway_txn_id": gid, "reference": ref, "amount": amount,
                                  "date": gw_date.strftime("%Y-%m-%d"), "record_type": "payment"})
            bank_rows.append({"bank_txn_id": bid, "reference": mangle_ref(ref, rng), "amount": bank_amount,
                               "date": bank_date.strftime("%Y-%m-%d"), "record_type": "credit_after_refund"})
            ground_truth.append({"gateway_txn_id": gid, "bank_txn_id": bid, "match_type": "refund_match"})

        elif roll < p_true_match + p_fee_adjusted + p_refund + p_gateway_only:
            gid, _ = new_ids()
            gateway_rows.append({"gateway_txn_id": gid, "reference": ref, "amount": amount,
                                  "date": gw_date.strftime("%Y-%m-%d"), "record_type": "payment"})
            ground_truth.append({"gateway_txn_id": gid, "bank_txn_id": "", "match_type": "gateway_only"})

        elif roll < p_true_match + p_fee_adjusted + p_refund + p_gateway_only + p_bank_only:
            _, bid = new_ids()
            bank_rows.append({"bank_txn_id": bid, "reference": f"MISC-{rng.randint(1000,9999)}", "amount": amount,
                               "date": gw_date.strftime("%Y-%m-%d"), "record_type": "misc"})
            ground_truth.append({"gateway_txn_id": "", "bank_txn_id": bid, "match_type": "bank_only"})

        elif roll < p_true_match + p_fee_adjusted + p_refund + p_gateway_only + p_bank_only + p_duplicate:
            if gateway_rows:
                src = rng.choice(gateway_rows)
                gid, _ = new_ids()
                gateway_rows.append({"gateway_txn_id": gid, "reference": src["reference"], "amount": src["amount"],
                                      "date": src["date"], "record_type": "payment"})
                ground_truth.append({"gateway_txn_id": gid, "bank_txn_id": "", "match_type": "duplicate_gateway_only"})
            else:
                continue

        else:  # ambiguous: two bank records that both plausibly match one gateway record
            gid, bid_true = new_ids()
            _, bid_decoy = new_ids()
            bank_date = gw_date + timedelta(days=rng.choice([0, 1]))
            gateway_rows.append({"gateway_txn_id": gid, "reference": ref, "amount": amount,
                                  "date": gw_date.strftime("%Y-%m-%d"), "record_type": "payment"})
            bank_rows.append({"bank_txn_id": bid_true, "reference": mangle_ref(ref, rng), "amount": amount,
                               "date": bank_date.strftime("%Y-%m-%d"), "record_type": "credit"})
            decoy_amount = round(amount + rng.choice([-1, 1]) * rng.uniform(1, 15), 2)
            bank_rows.append({"bank_txn_id": bid_decoy, "reference": f"TXN-{rng.randint(100000,999999)}",
                               "amount": decoy_amount, "date": bank_date.strftime("%Y-%m-%d"), "record_type": "credit"})
            ground_truth.append({"gateway_txn_id": gid, "bank_txn_id": bid_true, "match_type": "ambiguous_true_match"})
            ground_truth.append({"gateway_txn_id": "", "bank_txn_id": bid_decoy, "match_type": "ambiguous_decoy"})

        n_generated += 1

    rng.shuffle(gateway_rows)
    rng.shuffle(bank_rows)
    return gateway_rows, bank_rows, ground_truth


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of underlying records to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gateway_rows, bank_rows, ground_truth = build_dataset(n_records=args.n, seed=args.seed)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(gateway_rows, str(SAMPLE_DIR / "gateway_transactions.csv"),
              ["gateway_txn_id", "reference", "amount", "date", "record_type"])
    write_csv(bank_rows, str(SAMPLE_DIR / "bank_statement.csv"),
              ["bank_txn_id", "reference", "amount", "date", "record_type"])
    write_csv(ground_truth, str(SAMPLE_DIR / "ground_truth.csv"),
              ["gateway_txn_id", "bank_txn_id", "match_type"])

    print(f"Generated {len(gateway_rows)} gateway records, {len(bank_rows)} bank records "
          f"(n={args.n}, seed={args.seed})")
    print(f"Written to: {SAMPLE_DIR}")
    from collections import Counter
    print("Ground truth breakdown:", dict(Counter(r["match_type"] for r in ground_truth)))


if __name__ == "__main__":
    main()
