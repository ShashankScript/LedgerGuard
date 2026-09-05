"""
matching/normalize.py

Shared normalization helpers used by both the rule-based matcher and the ML
feature extraction, so both approaches see identically-normalized inputs.
"""

import re
from datetime import datetime


def normalize_ref(ref):
    return re.sub(r"[^A-Za-z0-9]", "", ref).upper()


def date_diff_days(d1, d2):
    return abs((datetime.strptime(d1, "%Y-%m-%d") - datetime.strptime(d2, "%Y-%m-%d")).days)
