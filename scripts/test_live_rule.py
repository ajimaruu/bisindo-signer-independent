"""Unit test of the >= 5 consecutive frames rule (scripts/live_rule.py).
Usage: python scripts/test_live_rule.py   (or: python -m pytest scripts/test_live_rule.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_rule import judge  # noqa: E402

T, N = "Cari", "none"


def test_four_in_a_row_is_miss():
    assert judge([N, T, T, T, T, N, N], T) == (False, 4, N)


def test_five_in_a_row_is_success():
    assert judge([N, T, T, T, T, T, N], T) == (True, 5, "")


def test_six_interrupted_is_miss():
    preds = [T, T, N, T, T, "Hari", T, T, N]          # 6 target frames, longest run 2
    assert judge(preds, T) == (False, 2, "Hari")


def test_five_at_window_end_is_success():
    assert judge([N] * 20 + [T] * 5, T) == (True, 5, "")


def test_all_none_is_miss_with_none():
    assert judge([N] * 30, T) == (False, 0, N)


def test_most_frequent_wrong_label():
    assert judge(["Hari", "Makan", "Makan", T, "Makan", "Hari", N], T) == (False, 1, "Makan")


def test_tie_takes_first_to_appear():
    assert judge(["Pagi", "Saya", "Saya", "Pagi"], T) == (False, 0, "Pagi")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
