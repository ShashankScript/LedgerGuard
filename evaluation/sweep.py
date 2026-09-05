"""
evaluation/sweep.py

Sweeps matcher.FUZZY_CONFIDENCE_THRESHOLD across a range and reports how
accuracy / false-match rate / unresolved count trade off — used to justify
the production threshold rather than picking one arbitrarily.

Run via: python -m evaluation.sweep   (from the project root)
"""

from matching import rule_matcher as matcher
from evaluation import evaluate


def main():
    thresholds = [round(x * 0.05 + 0.5, 2) for x in range(11)]  # 0.50 to 1.00
    truth = evaluate.load_ground_truth()

    print(f"{'threshold':>10} {'accuracy':>10} {'false_match_rate':>18} {'unresolved':>12}")
    for t in thresholds:
        results = matcher.run_matcher(threshold=t)
        metrics, _, _ = evaluate.evaluate(results, truth)
        print(f"{t:>10} {metrics['accuracy']:>10} {metrics['false_match_rate']:>18} {metrics['unresolved_count']:>12}")


if __name__ == "__main__":
    main()
