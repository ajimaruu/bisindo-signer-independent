"""Success rule of the live per-sign test (manuscript Table 8; scripts/live_test.py).

A trial is a SUCCESS when the target gloss is the top label above the threshold on at least NEED consecutive frames
of the 3-second recording window; otherwise it is a MISS. For a miss, label_on_miss is the most frequent label other
than the target ("none" when no other label appeared at all; ties: the label that appeared first).
Frame predictions are gloss names or "none".
"""
from collections import Counter

NEED = 5
NONE = "none"


def max_consecutive(preds, target):
    best = run = 0
    for p in preds:
        run = run + 1 if p == target else 0
        best = max(best, run)
    return best


def judge(preds, target, need=NEED):
    """Returns (success: bool, max_consecutive_target: int, label_on_miss: str; "" on success)."""
    mc = max_consecutive(preds, target)
    if mc >= need:
        return True, mc, ""
    others = [p for p in preds if p not in (target, NONE)]
    if not others:
        return False, mc, NONE
    counts = Counter(others)
    top = max(counts.values())
    return False, mc, next(p for p in others if counts[p] == top)   # tie: first to appear
