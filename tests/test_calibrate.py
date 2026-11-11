"""Tests for the calibration GUI click filtering (duplicates and overflow)."""

from __future__ import annotations

import sys
from pathlib import Path

# ``scripts/`` is not a package: make it importable so the calibrate helpers get unit-tested.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate import _accepts_click  # pyright: ignore[reportMissingImports]


def test_accepts_first_click() -> None:
    assert _accepts_click([], 100.0, 200.0) is True


def test_rejects_exact_duplicate_of_double_click() -> None:
    clicks = [(100.0, 200.0)]
    assert _accepts_click(clicks, 100.0, 200.0) is False


def test_rejects_click_within_separation_threshold() -> None:
    clicks = [(100.0, 200.0)]
    assert _accepts_click(clicks, 104.0, 203.0) is False  # 5 px away


def test_accepts_click_beyond_separation_threshold() -> None:
    clicks = [(100.0, 200.0)]
    assert _accepts_click(clicks, 109.0, 200.0) is True  # 9 px away


def test_rejects_fifth_click() -> None:
    clicks = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert _accepts_click(clicks, 500.0, 500.0) is False
