"""Tests for the JSON export module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from player_detect.export import ExportCollector
from player_detect.tracking import Track


def _track(track_id: int, x: float, conf: float = 0.9, class_id: int = -1) -> Track:
    return Track(track_id=track_id, bbox=(x, 10.0, x + 20.0, 50.0), conf=conf, class_id=class_id)


def _collector(class_names: dict[int, str] | None = None) -> ExportCollector:
    return ExportCollector(
        video="videos/x.mp4",
        fps=25.0,
        width=640,
        height=360,
        class_names=class_names,
    )


def test_detections_payload_has_expected_shape() -> None:
    collector = _collector()
    collector.add(0, 0.0, [_track(1, 0.0), _track(2, 100.0)])
    collector.add(2, 0.08, [_track(1, 5.0)])

    payload = collector.detections_payload()

    assert payload["video"] == "videos/x.mp4"
    assert payload["fps"] == 25.0
    assert len(payload["frames"]) == 2
    first = payload["frames"][0]
    assert first["frame"] == 0
    assert [det["track_id"] for det in first["detections"]] == [1, 2]
    assert first["detections"][0]["center"] == [10.0, 30.0]


def test_tracks_payload_aggregates_path_length() -> None:
    collector = _collector()
    collector.add(0, 0.0, [_track(7, 0.0)])
    collector.add(1, 0.04, [_track(7, 30.0)])
    collector.add(2, 0.08, [_track(7, 30.0, conf=0.5)])

    payload = collector.tracks_payload()
    summary = payload["tracks"][0]

    assert payload["track_count"] == 1
    assert summary["track_id"] == 7
    assert summary["frames"] == 3
    assert summary["first_t"] == 0.0
    assert summary["last_t"] == 0.08
    assert summary["path_length_px"] == pytest.approx(30.0)
    assert summary["mean_conf"] == pytest.approx((0.9 + 0.9 + 0.5) / 3, abs=1e-4)


def test_write_creates_both_files(tmp_path: Path) -> None:
    collector = _collector()
    collector.add(0, 0.0, [_track(3, 0.0)])

    detections_path = tmp_path / "detections.json"
    tracks_path = tmp_path / "tracks.json"
    track_count = collector.write(detections_path, tracks_path)

    assert track_count == 1
    assert json.loads(detections_path.read_text(encoding="utf-8"))["frames"][0]["frame"] == 0
    assert json.loads(tracks_path.read_text(encoding="utf-8"))["tracks"][0]["track_id"] == 3
    assert collector.detection_count == 1


def test_empty_frames_are_recorded() -> None:
    collector = _collector()
    collector.add(5, 0.2, [])

    payload = collector.detections_payload()

    assert payload["frames"][0]["detections"] == []
    assert payload["frames"][0]["frame"] == 5
    assert collector.tracks_payload()["track_count"] == 0


def test_detections_carry_class_id_and_name() -> None:
    collector = _collector({0: "player", 1: "goalkeeper", 2: "referee"})
    collector.add(0, 0.0, [_track(1, 0.0, class_id=1), _track(2, 100.0, class_id=2)])

    payload = collector.detections_payload()
    detections = payload["frames"][0]["detections"]

    assert [det["class_id"] for det in detections] == [1, 2]
    assert [det["class_name"] for det in detections] == ["goalkeeper", "referee"]
    assert payload["classes"] == {"player": 0, "goalkeeper": 1, "referee": 2}
    assert payload["class_counts"] == {"goalkeeper": 1, "referee": 1}


def test_track_summary_reports_dominant_class() -> None:
    collector = _collector({0: "player", 2: "referee"})
    collector.add(0, 0.0, [_track(9, 0.0, class_id=2)])
    collector.add(1, 0.04, [_track(9, 5.0, class_id=2), _track(9, 5.0, class_id=0)])

    summary = collector.tracks_payload()["tracks"][0]

    assert summary["class_name"] == "referee"
    assert summary["class_counts"] == {"player": 1, "referee": 2}


def test_track_without_class_is_unknown() -> None:
    collector = _collector({0: "player"})
    collector.add(0, 0.0, [_track(4, 0.0)])

    payload = collector.detections_payload()

    assert payload["frames"][0]["detections"][0]["class_name"] == "unknown"
    assert payload["class_counts"] == {"unknown": 1}
