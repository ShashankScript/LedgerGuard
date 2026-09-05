"""
tests/test_core.py

Smoke tests for the core pipeline. Run via:
    python -m unittest tests.test_core -v      (from the project root)

Covers: reproducible data generation, blocking correctness, rule-based
matching on known cases, ground-truth isolation, and ML training/evaluation
running without error.

No pytest dependency required — uses the standard library's unittest so this
runs with nothing beyond requirements.txt already installed.
"""

import unittest

from data import generate_data
from matching import blocking
from matching import rule_matcher as matcher
from matching import build_training_data
try:
    from matching import ml_matcher
except ModuleNotFoundError:  # allows core rules to run before optional ML deps are installed
    ml_matcher = None


class TestDataGeneration(unittest.TestCase):
    def test_reproducible_with_same_seed(self):
        gw1, bk1, gt1 = generate_data.build_dataset(n_records=100, seed=7)
        gw2, bk2, gt2 = generate_data.build_dataset(n_records=100, seed=7)
        self.assertEqual([g["gateway_txn_id"] for g in gw1], [g["gateway_txn_id"] for g in gw2])
        self.assertEqual([b["amount"] for b in bk1], [b["amount"] for b in bk2])

    def test_different_seeds_differ(self):
        gw1, _, _ = generate_data.build_dataset(n_records=100, seed=1)
        gw2, _, _ = generate_data.build_dataset(n_records=100, seed=2)
        self.assertNotEqual([g["amount"] for g in gw1], [g["amount"] for g in gw2])

    def test_ground_truth_covers_every_gateway_and_bank_record(self):
        gw, bk, gt = generate_data.build_dataset(n_records=200, seed=42)
        gt_gateway_ids = {r["gateway_txn_id"] for r in gt if r["gateway_txn_id"]}
        gt_bank_ids = {r["bank_txn_id"] for r in gt if r["bank_txn_id"]}
        gw_ids = {g["gateway_txn_id"] for g in gw}
        bk_ids = {b["bank_txn_id"] for b in bk}
        self.assertEqual(gt_gateway_ids, gw_ids)
        self.assertEqual(gt_bank_ids, bk_ids)


class TestBlocking(unittest.TestCase):
    def test_true_match_is_always_a_candidate(self):
        """Every true match pair must survive blocking, or the matcher can
        never find it no matter how good the scoring logic is."""
        gw, bk, gt = generate_data.build_dataset(n_records=300, seed=42)
        bk_by_id = {b["bank_txn_id"]: b for b in bk}
        gw_by_id = {g["gateway_txn_id"]: g for g in gw}
        bank_index = blocking.build_bank_index([dict(b, amount=float(b["amount"])) for b in bk])

        checked = 0
        for row in gt:
            if row["match_type"] in {"true_match", "fee_adjusted_match", "refund_match", "ambiguous_true_match"} and row["gateway_txn_id"] and row["bank_txn_id"]:
                g = dict(gw_by_id[row["gateway_txn_id"]], amount=float(gw_by_id[row["gateway_txn_id"]]["amount"]))
                candidates = blocking.candidates_for(g, bank_index)
                candidate_ids = {c["bank_txn_id"] for c in candidates}
                self.assertIn(row["bank_txn_id"], candidate_ids,
                              f"True match {row['bank_txn_id']} was blocked out for {row['gateway_txn_id']}")
                checked += 1
        self.assertGreater(checked, 0, "No true_match rows found to check — test data generation may have changed")


class TestRuleMatcher(unittest.TestCase):
    def test_exact_match_found(self):
        gw = [{"gateway_txn_id": "GW1", "reference": "TXN-000001", "amount": "1000.00", "date": "2026-06-01"}]
        bk = [{"bank_txn_id": "BK1", "reference": "TXN-000001", "amount": "1000.00", "date": "2026-06-01"}]
        results = matcher.run_matcher(gateway_records=gw, bank_records=bk)
        matched = [r for r in results if r["match_type"] == "exact"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["bank_txn_id"], "BK1")

    def test_no_match_when_nothing_close(self):
        gw = [{"gateway_txn_id": "GW1", "reference": "TXN-000001", "amount": "1000.00", "date": "2026-06-01"}]
        bk = [{"bank_txn_id": "BK1", "reference": "MISC-9999", "amount": "50000.00", "date": "2026-08-01"}]
        results = matcher.run_matcher(gateway_records=gw, bank_records=bk)
        self.assertTrue(all(r["match_type"] == "unresolved" for r in results))

    def test_fee_adjusted_match_found_by_fuzzy_pass(self):
        gw = [{"gateway_txn_id": "GW1", "reference": "TXN-000001", "amount": "1000.00", "date": "2026-06-01"}]
        bk = [{"bank_txn_id": "BK1", "reference": "TXN-000001", "amount": "990.00", "date": "2026-06-01"}]
        results = matcher.run_matcher(gateway_records=gw, bank_records=bk)
        matched = [r for r in results if r["match_type"] in ("exact", "fuzzy", "fee_adjusted")]
        self.assertEqual(len(matched), 1)

    def test_typo_in_reference_is_matched(self):
        gw = [{"gateway_txn_id": "GW1", "reference": "TXN-000001", "amount": "1000", "date": "2026-06-01"}]
        bk = [{"bank_txn_id": "BK1", "reference": "TXN-000002", "amount": "1000", "date": "2026-06-01", "record_type": "credit"}]
        self.assertTrue(any(r["bank_txn_id"] == "BK1" for r in matcher.run_matcher(gateway_records=gw, bank_records=bk)))

    def test_date_lag_is_matched(self):
        gw = [{"gateway_txn_id": "GW1", "reference": "TXN-000001", "amount": "1000", "date": "2026-06-01"}]
        bk = [{"bank_txn_id": "BK1", "reference": "TXN-000001", "amount": "1000", "date": "2026-06-04", "record_type": "credit"}]
        self.assertTrue(any(r["bank_txn_id"] == "BK1" for r in matcher.run_matcher(gateway_records=gw, bank_records=bk)))

    def test_refund_is_matched_without_amount_blocking(self):
        gw = [{"gateway_txn_id": "GW1", "reference": "TXN-000001", "amount": "10000", "date": "2026-06-01"}]
        bk = [{"bank_txn_id": "BK1", "reference": "TXN-000001", "amount": "7000", "date": "2026-06-03", "record_type": "credit_after_refund"}]
        result = matcher.run_matcher(gateway_records=gw, bank_records=bk)
        self.assertEqual(next(r for r in result if r["gateway_txn_id"] == "GW1")["match_type"], "refund_adjusted")

    def test_duplicate_gateway_records_are_escalated(self):
        gw = [{"gateway_txn_id": x, "reference": "TXN-000001", "amount": "1000", "date": "2026-06-01"} for x in ("GW1", "GW2")]
        bk = [{"bank_txn_id": "BK1", "reference": "TXN-000001", "amount": "1000", "date": "2026-06-01", "record_type": "credit"}]
        self.assertTrue(all(r["match_type"] == "human_review" for r in matcher.run_matcher(gateway_records=gw, bank_records=bk) if r["gateway_txn_id"]))

    def test_ambiguous_candidates_are_escalated(self):
        gw = [{"gateway_txn_id": "GW1", "reference": "TXN-000001", "amount": "1000", "date": "2026-06-01"}]
        bk = [{"bank_txn_id": x, "reference": "TXN-000001", "amount": amount, "date": "2026-06-01", "record_type": "credit"} for x, amount in (("BK1", "1000"), ("BK2", "1001"))]
        result = matcher.run_matcher(gateway_records=gw, bank_records=bk)
        self.assertEqual(next(r for r in result if r["gateway_txn_id"] == "GW1")["match_type"], "human_review")

    def test_gateway_only_and_unrelated_bank_stay_unresolved(self):
        gw = [{"gateway_txn_id": "GW1", "reference": "TXN-000001", "amount": "1000", "date": "2026-06-01"}]
        bk = [{"bank_txn_id": "BK1", "reference": "MISC-9999", "amount": "1000", "date": "2026-06-01", "record_type": "misc"}]
        self.assertTrue(all(r["match_type"] == "unresolved" for r in matcher.run_matcher(gateway_records=gw, bank_records=bk)))

    def test_near_match_with_wrong_amount_is_not_matched(self):
        gw = [{"gateway_txn_id": "GW1", "reference": "TXN-000001", "amount": "1000", "date": "2026-06-01"}]
        bk = [{"bank_txn_id": "BK1", "reference": "TXN-000001", "amount": "800", "date": "2026-06-01", "record_type": "credit"}]
        result = matcher.run_matcher(gateway_records=gw, bank_records=bk)
        self.assertEqual(next(r for r in result if r["gateway_txn_id"] == "GW1")["match_type"], "unresolved")


class TestGroundTruthIsolation(unittest.TestCase):
    def test_run_matcher_signature_has_no_ground_truth_param(self):
        """The matcher must structurally be unable to see ground truth —
        checked by confirming its signature has no such parameter."""
        import inspect
        sig = inspect.signature(matcher.run_matcher)
        self.assertNotIn("ground_truth", sig.parameters)
        self.assertNotIn("truth", sig.parameters)


class TestMLPipeline(unittest.TestCase):
    def test_training_and_evaluation_runs_without_error(self):
        if ml_matcher is None:
            self.skipTest("scikit-learn is not installed in this interpreter; install requirements.txt")
        gw, bk, gt = generate_data.build_dataset(n_records=300, seed=42)
        rows = build_training_data.build_labeled_pairs(
            gateway_records=gw, bank_records=bk, ground_truth_records=gt,
        )
        self.assertGreater(len(rows), 0)
        self.assertGreater(sum(r["label"] for r in rows), 10)  # need positives to train meaningfully

        result = ml_matcher.train_and_evaluate(rows, seed=42)
        self.assertIn("ml_metrics", result)
        self.assertIn("rule_metrics", result)
        self.assertGreaterEqual(result["ml_metrics"]["f1"], 0.0)
        self.assertLessEqual(result["ml_metrics"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
