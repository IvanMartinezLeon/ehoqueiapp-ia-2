"""Tests for jersey-colour team classification."""

from __future__ import annotations

import numpy as np
import pytest

from player_detect.teams import (
    REFEREE,
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    ColorFeature,
    TeamClassifier,
    kmeans,
    sample_jersey_color,
    torso_region,
)
from player_detect.tracking import Track

FRAME_H, FRAME_W = 240, 320


def _frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _track(track_id: int, bbox: tuple[float, float, float, float]) -> Track:
    return Track(track_id=track_id, bbox=bbox, conf=0.9)


def _feature(degrees: float, saturation: float, value: float) -> ColorFeature:
    angle = np.deg2rad(degrees)
    return ColorFeature(
        cos_h=float(np.cos(angle)),
        sin_h=float(np.sin(angle)),
        saturation=saturation,
        value=value,
    )


def test_torso_region_rejects_small_boxes() -> None:
    assert torso_region((10.0, 10.0, 30.0, 30.0), (FRAME_H, FRAME_W), 40) is None
    assert torso_region((10.0, 10.0, 40.0, 90.0), (FRAME_H, FRAME_W), 40) is not None


def test_torso_region_is_inside_the_box() -> None:
    region = torso_region((100.0, 50.0, 180.0, 150.0), (FRAME_H, FRAME_W), 40)

    assert region is not None
    rows, cols = region
    assert 50 <= rows.start < rows.stop <= 150
    assert 100 <= cols.start < cols.stop <= 180


@pytest.mark.parametrize(
    ("bgr", "expected_hue"),
    [((0, 200, 255), 48.0), ((255, 0, 0), 240.0), ((0, 0, 0), 0.0)],
)
def test_sample_jersey_color_reads_hue(
    bgr: tuple[int, int, int], expected_hue: float | None
) -> None:
    frame = _frame(bgr)

    feature = sample_jersey_color(frame, (100.0, 80.0, 160.0, 200.0), min_box_height=40)

    assert feature is not None
    assert feature.hue == pytest.approx(expected_hue, abs=6.0)


def test_sample_jersey_color_handles_black_jerseys() -> None:
    feature = sample_jersey_color(_frame((10, 10, 10)), (100.0, 80.0, 160.0, 200.0))

    assert feature is not None
    assert feature.value == pytest.approx(10.0, abs=1.0)


def test_kmeans_is_deterministic() -> None:
    points = np.array(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0], [0.0, 10.0], [0.1, 10.1]]
    )

    first = kmeans(points, 3, seed=0)
    second = kmeans(points, 3, seed=0)

    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 3


def _classifier() -> TeamClassifier:
    return TeamClassifier(min_color_samples=3, classify_every=5, min_box_height=40)


def test_classifier_keeps_tracks_unknown_until_enough_samples() -> None:
    classifier = _classifier()
    for _ in range(3):
        classifier.observe(_frame((0, 200, 255)), [_track(1, (100.0, 80.0, 160.0, 200.0))])

    labels = classifier.classify()

    assert labels == {}


def test_classifier_assigns_three_labels_and_referee_is_darkest() -> None:
    classifier = _classifier()
    colors = {
        11: (0, 200, 255),  # yellow
        12: (0, 210, 255),  # yellow
        21: (230, 230, 230),  # white / pale
        22: (240, 240, 240),  # white / pale
        31: (10, 10, 10),  # black jersey (referee)
        32: (15, 15, 15),  # black jersey (referee)
    }
    bbox = (100.0, 80.0, 160.0, 200.0)
    for _ in range(4):
        for track_id, bgr in colors.items():
            classifier.observe(_frame(bgr), [_track(track_id, bbox)])

    labels = classifier.classify()

    assert set(labels.values()) == {REFEREE, TEAM_A, TEAM_B}
    assert {labels[31], labels[32]} == {REFEREE}
    assert labels[11] == labels[12]
    assert labels[21] == labels[22]
    assert labels[11] != labels[21]


def test_labels_are_sticky() -> None:
    classifier = _classifier()
    bbox = (100.0, 80.0, 160.0, 200.0)
    palette = {1: (0, 200, 255), 2: (230, 230, 230), 3: (10, 10, 10)}
    for _ in range(4):
        for track_id, bgr in palette.items():
            classifier.observe(_frame(bgr), [_track(track_id, bbox)])
    first = classifier.classify()

    for _ in range(4):
        for track_id, bgr in palette.items():
            classifier.observe(_frame(bgr), [_track(track_id, bbox)])
    second = classifier.classify()

    assert first == second


def test_box_colors_map_by_label() -> None:
    classifier = _classifier()
    bbox = (100.0, 80.0, 160.0, 200.0)
    palette = {1: (0, 200, 255), 2: (230, 230, 230), 3: (10, 10, 10)}
    for _ in range(4):
        for track_id, bgr in palette.items():
            classifier.observe(_frame(bgr), [_track(track_id, bbox)])
    classifier.classify()

    colors = classifier.box_colors()

    assert colors[3] == classifier.colors[REFEREE]
    assert colors[1] != colors[2]


def test_unknown_label_for_unseen_track() -> None:
    classifier = _classifier()

    assert classifier.label_of(999) == UNKNOWN
    assert classifier.counts()[UNKNOWN] == 0
    assert classifier.track_colors() == {}


def test_track_colors_report_contains_team_and_hsv() -> None:
    classifier = _classifier()
    bbox = (100.0, 80.0, 160.0, 200.0)
    palette = {1: (0, 200, 255), 2: (230, 230, 230), 3: (10, 10, 10)}
    for _ in range(4):
        for track_id, bgr in palette.items():
            classifier.observe(_frame(bgr), [_track(track_id, bbox)])
    classifier.classify()

    report = classifier.track_colors()

    assert set(report) == {1, 2, 3}
    assert report[3]["team"] == REFEREE
    mean_hsv = report[1]["mean_hsv"]
    assert isinstance(mean_hsv, list)
    assert len(mean_hsv) == 3
    assert report[1]["color_samples"] == 4


def test_should_classify_every_n_frames() -> None:
    classifier = TeamClassifier(classify_every=15)

    assert classifier.should_classify(0)
    assert classifier.should_classify(15)
    assert not classifier.should_classify(7)


def test_feature_helpers() -> None:
    feature = _feature(120.0, 100.0, 50.0)

    assert feature.hue == pytest.approx(120.0, abs=1e-6)
    assert feature.vector.shape == (4,)
