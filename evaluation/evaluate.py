"""
evaluate.py

Scores the matcher's output against ground_truth.csv (used ONLY here, never
inside matcher.py itself — the matcher must not see the answer key).

Reports:
  - Match rate:        % of gateway transactions that were auto-resolved (matched or
                        correctly identified as gateway-only/duplicate)
  - Accuracy:          of the matches the system MADE, what % were correct
  - False-match rate:  % of made matches that were WRONG (paired the wrong two records)
  - Exception quality: how many unresolved records got a real reason vs a blank one

Also writes exception_report.csv, a plain-English breakdown of every record the
system could not confidently resolve.
"""

import csv
from pathlib import Path
from matching.rule_matcher import run_matcher, save_results, DEFAULT_RESULTS_PATH

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH_PATH = str(_ROOT / "data" / "sample" / "ground_truth.csv")
DEFAULT_EXCEPTION_REPORT_PATH = str(_ROOT / "outputs" / "exception_report.csv")


def load_ground_truth(path=None):
    path = path or DEFAULT_GROUND_TRUTH_PATH
    truth = {}
    for row in csv.DictReader(open(path, newline="")):
        gw = row["gateway_txn_id"]
        if gw:
            truth[gw] = {"bank_txn_id": row["bank_txn_id"], "match_type": row["match_type"]}
    return truth


AUTO_MATCH_TYPES = {"exact", "fee_adjusted", "refund_adjusted", "fuzzy"}


def evaluate(results, truth):
    """Evaluate the full decision, not just ordinary exact transactions.

    Ground truth is used only after inference to score outputs. A valid
    fee/refund pair is credited when its actual bank id is selected; an
    expected-unmatched gateway record is credited only when it remains
    unclaimed. Human-review is deliberately neither an automatic true match
    nor a false match.
    """
    gateway_results = {r["gateway_txn_id"]: r for r in results if r["gateway_txn_id"]}
    true_pairs = {gw: item for gw, item in truth.items() if item["bank_txn_id"]}
    made_matches = [r for r in gateway_results.values() if r["match_type"] in AUTO_MATCH_TYPES]
    correct_matches, false_matches = 0, []
    decision_correct = 0
    subset = {}
    for gw, expected in truth.items():
        result = gateway_results.get(gw, {})
        selected = result.get("bank_txn_id", "")
        is_auto = result.get("match_type") in AUTO_MATCH_TYPES
        if expected["bank_txn_id"]:
            correct = is_auto and selected == expected["bank_txn_id"]
            if correct:
                correct_matches += 1
        else:
            correct = not selected
        decision_correct += int(correct)
        bucket = subset.setdefault(expected["match_type"], {"total": 0, "correct": 0, "auto_matched": 0})
        bucket["total"] += 1; bucket["correct"] += int(correct); bucket["auto_matched"] += int(is_auto)
    for r in made_matches:
        expected = truth.get(r["gateway_txn_id"], {})
        if expected.get("bank_txn_id") != r["bank_txn_id"]:
            false_matches.append(r)
    n_made, n_true_pairs = len(made_matches), len(true_pairs)
    precision = correct_matches / n_made if n_made else 0.0
    recall = correct_matches / n_true_pairs if n_true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    for values in subset.values():
        values["correct_rate"] = round(values["correct"] / values["total"], 4) if values["total"] else 0.0
    unresolved = [r for r in results if r["match_type"] in ("unresolved", "human_review")]
    return {
        "total_gateway_records": len(truth), "total_true_matches_available": n_true_pairs,
        "matches_made": n_made, "correct_matches": correct_matches,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "accuracy": round(decision_correct / len(truth), 4) if truth else 0.0,
        "false_matches": len(false_matches), "false_match_rate": round(len(false_matches) / n_made, 4) if n_made else 0.0,
        "match_rate_on_true_pairs": round(recall, 4), "automation_rate": round(n_made / len(truth), 4) if truth else 0.0,
        "unresolved_count": len(unresolved), "subset_metrics": subset,
    }, false_matches, unresolved


def write_exception_report(unresolved, false_matches, path=None):
    path = path or DEFAULT_EXCEPTION_REPORT_PATH
    rows = []
    for r in unresolved:
        side = "gateway" if r["gateway_txn_id"] else "bank"
        record_id = r["gateway_txn_id"] or r["bank_txn_id"]
        rows.append({
            "record_id": record_id,
            "side": side,
            "issue": "unresolved",
            "explanation": r["reason"],
        })
    for r in false_matches:
        rows.append({
            "record_id": r["gateway_txn_id"],
            "side": "gateway",
            "issue": "incorrect_match",
            "explanation": f"Matched to {r['bank_txn_id']} (confidence {r['confidence']}) but this pairing is wrong",
        })

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "side", "issue", "explanation"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    results = run_matcher()
    save_results(results)

    truth = load_ground_truth()
    metrics, false_matches, unresolved = evaluate(results, truth)
    n_exceptions = write_exception_report(unresolved, false_matches)

    print("=" * 50)
    print("RECONCILIATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"{k:32s}: {v}")
    print("-" * 50)
    print(f"Exception report written: {DEFAULT_EXCEPTION_REPORT_PATH} ({n_exceptions} rows)")
    print(f"Full match results written: {DEFAULT_RESULTS_PATH} ({len(results)} rows)")


if __name__ == "__main__":
    # Run via: python -m evaluation.evaluate   (from the project root)
    main()
