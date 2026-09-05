"""
ml_matcher.py

Trains a logistic regression classifier to predict whether a (gateway, bank)
candidate pair is a true match, using the features from matcher.extract_features().

LEAKAGE NOTE: rows are split by gateway_txn_id (GroupShuffleSplit), not by row.
Reason: multiple candidate rows share the same gateway_txn_id (one true
candidate + several false ones from blocking). Splitting by row could put
different candidates for the SAME gateway record into both train and test,
which would let the model implicitly learn "this exact gateway record" rather
than generalizing — a subtle form of leakage. Splitting by group prevents that.

Reports precision/recall/F1/confusion matrix on a held-out TEST set that is
touched exactly once, at the end. Also runs the rule-based scorer on the
identical test set for a fair side-by-side comparison.
"""

import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from . import rule_matcher as matcher
from . import build_training_data as _btd

FEATURE_NAMES = matcher.MODEL_FEATURE_NAMES


def load_labeled_pairs(path=None):
    path = path or _btd.DEFAULT_LABELED_PAIRS_PATH
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        for k in FEATURE_NAMES + ["label"]:
            r[k] = float(r[k])
    return rows


def to_arrays(rows):
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["gateway_txn_id"] for r in rows])
    return X, y, groups


def group_split(rows, seed=42):
    """60/20/20 train/val/test, split by gateway_txn_id group."""
    X, y, groups = to_arrays(rows)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx_rel, test_idx_rel = next(gss2.split(X[temp_idx], y[temp_idx], groups[temp_idx]))
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    # Sanity check: no gateway_txn_id should appear in more than one split
    train_groups, val_groups, test_groups = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
    assert not (train_groups & val_groups), "LEAKAGE: train/val group overlap"
    assert not (train_groups & test_groups), "LEAKAGE: train/test group overlap"
    assert not (val_groups & test_groups), "LEAKAGE: val/test group overlap"

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate_predictions(y_true, y_pred, label=""):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"Precision: {p:.4f}   Recall: {r:.4f}   F1: {f1:.4f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tn": int(cm[0][0]), "fp": int(cm[0][1]), "fn": int(cm[1][0]), "tp": int(cm[1][1])}


def rule_based_predictions(rows_subset, threshold=matcher.FUZZY_CONFIDENCE_THRESHOLD):
    """Apply the existing hand-weighted formula to the same rows, for comparison."""
    preds = []
    for r in rows_subset:
        features = {f: r[f] for f in FEATURE_NAMES}
        score = matcher.rule_based_score(features)
        preds.append(1 if score >= threshold else 0)
    return np.array(preds)


def tune_threshold_on_val(model, scaler, X_val, y_val):
    """
    Sweep the ML model's decision threshold on the VALIDATION set only, and
    pick the one that maximizes F1. This makes the eventual rule-vs-ML
    comparison fair: both approaches get a tuned threshold, neither uses an
    arbitrary default. Test set is never touched here.
    """
    X_val_scaled = scaler.transform(X_val)
    probs = model.predict_proba(X_val_scaled)[:, 1]

    best_threshold, best_f1 = 0.5, -1.0
    for t in [round(x * 0.02, 2) for x in range(10, 50)]:  # 0.20 to 0.98
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold, best_f1


def train_and_evaluate(rows, seed=42):
    """
    Full train -> tune -> evaluate pipeline as a callable function, so both the
    CLI (__main__ below) and the Streamlit app can use identical logic instead
    of duplicating it. Returns a dict with the trained model, scaler, chosen
    threshold, and honest metrics for both approaches on the same test set.
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = group_split(rows, seed=seed)

    model, scaler = train_model(X_train, y_train)
    ml_threshold, val_f1 = tune_threshold_on_val(model, scaler, X_val, y_val)

    Xa, ya, ga = to_arrays(rows)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, temp_idx = next(gss1.split(Xa, ya, ga))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx_rel, test_idx_rel = next(gss2.split(Xa[temp_idx], ya[temp_idx], ga[temp_idx]))
    test_idx = temp_idx[test_idx_rel]
    test_rows = [rows[i] for i in test_idx]

    X_test_scaled = scaler.transform(X_test)
    y_proba_ml = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_ml = (y_proba_ml >= ml_threshold).astype(int)
    y_pred_rule = rule_based_predictions(test_rows)

    ml_metrics = evaluate_predictions(y_test, y_pred_ml, label="ML classifier (logistic regression)")
    rule_metrics = evaluate_predictions(y_test, y_pred_rule, label="Rule-based baseline (weighted formula)")

    feature_importance = sorted(zip(FEATURE_NAMES, model.coef_[0]), key=lambda x: -abs(x[1]))

    return {
        "model": model, "scaler": scaler, "ml_threshold": ml_threshold,
        "val_f1": val_f1, "ml_metrics": ml_metrics, "rule_metrics": rule_metrics,
        "feature_importance": feature_importance,
        "split_sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "positive_rates": {"train": float(y_train.mean()), "val": float(y_val.mean()), "test": float(y_test.mean())},
    }


def main():
    rows = load_labeled_pairs()
    print(f"Loaded {len(rows)} labeled pairs")
    result = train_and_evaluate(rows)
    print(f"\nSplit sizes -> train: {result['split_sizes']['train']}  "
          f"val: {result['split_sizes']['val']}  test: {result['split_sizes']['test']}")
    print(f"ML decision threshold tuned on VALIDATION set: {result['ml_threshold']} (val F1={result['val_f1']:.4f})")
    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET COMPARISON (touched once, here)")
    print("=" * 60)
    print(f"\n--- ML classifier ---\n{result['ml_metrics']}")
    print(f"\n--- Rule-based baseline ---\n{result['rule_metrics']}")
    print("\n--- Feature importance ---")
    for name, coef in result["feature_importance"]:
        print(f"{name:20s}: {coef:+.3f}")


if __name__ == "__main__":
    main()
