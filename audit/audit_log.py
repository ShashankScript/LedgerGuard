"""
audit.py

Structured audit trail. Every automated decision AND every human override
gets logged here, so a judge (or a real ops team) can click a transaction and
answer "why did the system decide this?"

Persisted to audit_log.csv (append-only) so it survives across app reruns
within a demo session.
"""

import csv
import json
import os
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG_PATH = str(_ROOT / "outputs" / "audit_log.csv")
FIELDS = ["timestamp", "gateway_txn_id", "bank_txn_id", "candidates_considered",
          "selected_candidate", "decision", "confidence", "model_version",
          "threshold", "human_override", "reason"]


def _ensure_file():
    Path(AUDIT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def log_decision(gateway_txn_id, bank_txn_id, candidates_considered, selected_candidate,
                  decision, confidence, model_version="rule_v1", threshold=None,
                  human_override=False, reason=""):
    _ensure_file()
    with open(AUDIT_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gateway_txn_id": gateway_txn_id,
            "bank_txn_id": bank_txn_id or "",
            "candidates_considered": json.dumps(candidates_considered),
            "selected_candidate": selected_candidate or "",
            "decision": decision,
            "confidence": confidence,
            "model_version": model_version,
            "threshold": threshold,
            "human_override": human_override,
            "reason": reason,
        })


def load_audit_log():
    _ensure_file()
    with open(AUDIT_LOG_PATH, newline="") as f:
        return list(csv.DictReader(f))


def clear_audit_log():
    if os.path.exists(AUDIT_LOG_PATH):
        os.remove(AUDIT_LOG_PATH)
    _ensure_file()
