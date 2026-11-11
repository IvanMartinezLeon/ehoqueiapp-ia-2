"""Tests for the dataset split helpers (group / temporal split)."""

from __future__ import annotations

import sys
from pathlib import Path

# ``scripts/`` is not a package: make it importable so the split helpers get unit-tested.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split_dataset import group_of, split_by_group  # pyright: ignore[reportMissingImports]


def _items(prefix: str, count: int) -> list[tuple[Path, Path]]:
    return [
        (Path(f"images/{prefix}_{index:06d}.jpg"), Path(f"labels/{prefix}_{index:06d}.txt"))
        for index in range(1, count + 1)
    ]


def _numbers(pairs: list[tuple[Path, Path]]) -> list[int]:
    return sorted(int(image.stem.rsplit("_", 1)[1]) for image, _ in pairs)


def test_group_of_strips_the_frame_number() -> None:
    assert group_of("noa_000010") == "noa"
    assert group_of("frame_000001") == "frame"
    assert group_of("p2403_000123") == "p2403"
    assert group_of("sin_numero") == "sin_numero"


def test_group_split_keeps_videos_on_both_sides_without_leaking_frames() -> None:
    items = _items("frame", 20) + _items("noa", 20)

    train, val = split_by_group(items, 0.2)

    assert len(train) + len(val) == 40
    for prefix in ("frame", "noa"):
        train_numbers = [n for n in _numbers(train) if str(n) in {str(i) for i in range(1, 21)}]
        val_numbers = [
            n
            for image, _ in val
            if image.stem.startswith(prefix) and (n := int(image.stem.rsplit("_", 1)[1]))
        ]
        assert val_numbers, f"{prefix} should contribute validation frames"
        assert max(train_numbers) < min(val_numbers), f"{prefix}: validation must be the tail"
    assert _numbers(train) != []
    assert len(val) == 8  # 20 % of each group


def test_group_split_leaves_tiny_groups_in_train() -> None:
    items = _items("frame", 20) + _items("corto", 3)

    train, val = split_by_group(items, 0.2)

    assert all("corto" not in image.stem for image, _ in val)
    assert any("corto" in image.stem for image, _ in train)


def test_group_split_is_deterministic() -> None:
    items = _items("noa", 30)

    assert split_by_group(items, 0.2) == split_by_group(items, 0.2)


def test_group_split_never_puts_a_group_entirely_in_validation() -> None:
    items = _items("frame", 5)

    train, val = split_by_group(items, 0.9)

    assert train, "at least one frame per group must stay in train"
    assert len(val) <= len(items) - 1
