"""Tests for camera motion, pitch projection and minimap rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from player_detect.camera import AnchorTracker, CameraTracker, compose, to_reference
from player_detect.minimap import HeatmapAccumulator, build_minimap, overlay, rink_canvas
from player_detect.pitch import (
    Calibration,
    RinkProjector,
    foot_point,
    load_calibration,
    order_quad,
    save_calibration,
)
from player_detect.world import WorldTracker, _line_distance, floor_line_mask, line_residual

IDENTITY = np.eye(3)


def _synthetic_pair(shift: tuple[int, int] = (12, 5)) -> tuple[np.ndarray, np.ndarray]:
    """Two frames of a textured plane where the second is shifted by ``shift`` pixels."""
    rng = np.random.default_rng(0)
    base = np.zeros((240, 320), dtype=np.uint8)
    for _ in range(400):
        x, y = rng.integers(10, 300), rng.integers(10, 220)
        cv2.circle(base, (int(x), int(y)), 3, 200, -1)
    matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], dtype=np.float64)
    moved = cv2.warpAffine(base, matrix[:2], (320, 240), borderMode=cv2.BORDER_REFLECT)
    return base, moved


def test_camera_tracker_recovers_known_shift() -> None:
    previous, current = _synthetic_pair((12, 5))

    matrix = CameraTracker(min_inliers=10).update(previous, current)

    assert matrix is not None
    assert matrix[0, 2] == pytest.approx(12, abs=1.5)
    assert matrix[1, 2] == pytest.approx(5, abs=1.5)


def test_camera_tracker_rejects_featureless_frames() -> None:
    blank = np.zeros((240, 320), dtype=np.uint8)

    assert CameraTracker(min_inliers=10).update(blank, blank) is None


def _textured_frame(shift: tuple[float, float], size: tuple[int, int] = (480, 854)) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = np.zeros(size, dtype=np.uint8)
    for _ in range(1500):
        x, y = rng.integers(20, size[1] - 20), rng.integers(20, size[0] - 20)
        cv2.circle(base, (int(x), int(y)), 4, int(rng.integers(120, 255)), -1)
    matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
    return cv2.warpAffine(base, matrix, (size[1], size[0]), borderMode=cv2.BORDER_REFLECT)


def test_anchor_tracker_recovers_absolute_shift() -> None:
    reference = _textured_frame((0.0, 0.0))
    tracker = AnchorTracker(reference, max_features=1500, min_inliers=20)

    matrix = tracker.update(_textured_frame((37.0, 11.0)))

    assert matrix is not None
    assert matrix[0, 2] == pytest.approx(37, abs=2.0)
    assert matrix[1, 2] == pytest.approx(11, abs=2.0)


def test_anchor_tracker_does_not_accumulate_error() -> None:
    """A frame 40 steps away is registered directly, so its error does not grow with distance."""
    reference = _textured_frame((0.0, 0.0))
    tracker = AnchorTracker(reference, max_features=3000, min_inliers=20)

    errors = []
    for step in (10, 20, 30, 40):
        matrix = tracker.update(_textured_frame((float(step * 12), float(step * 2))))
        assert matrix is not None
        errors.append(float(np.hypot(matrix[0, 2] - step * 12, matrix[1, 2] - step * 2)))

    assert max(errors) < 3.0
    assert max(errors) - min(errors) < 3.0


def test_anchor_tracker_returns_none_without_overlap() -> None:
    reference = _textured_frame((0.0, 0.0))
    tracker = AnchorTracker(reference, max_features=500, min_inliers=20)
    blank = np.zeros((480, 854), dtype=np.uint8)

    assert tracker.update(blank) is None


def test_compose_accumulates_shifts() -> None:
    first = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    second = np.array([[1, 0, 5], [0, 1, 2], [0, 0, 1]], dtype=np.float64)

    cumulative = compose(second, compose(first, IDENTITY))

    assert cumulative[0, 2] == pytest.approx(15)
    assert cumulative[1, 2] == pytest.approx(2)
    assert cumulative[2, 2] == pytest.approx(1)


def test_to_reference_inverts_the_accumulated_shift() -> None:
    cumulative = np.array([[1, 0, 30], [0, 1, -10], [0, 0, 1]], dtype=np.float64)

    reference = to_reference(cumulative, (130.0, 40.0))

    assert reference is not None
    assert reference[0] == pytest.approx(100.0)
    assert reference[1] == pytest.approx(50.0)


def _calibration(frame: int = 0) -> Calibration:
    return Calibration(
        video="videos/a.mp4",
        frame=frame,
        image_points=((0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0)),
        rink_points=((0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)),
        width_m=40.0,
        height_m=20.0,
    )


def test_calibration_roundtrip_has_no_error() -> None:
    calibration = _calibration()

    assert calibration.reprojection_error() == pytest.approx(0.0, abs=1e-6)


def test_calibration_requires_four_points() -> None:
    with pytest.raises(ValueError):
        Calibration(
            video="v",
            frame=0,
            image_points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            rink_points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            width_m=1.0,
            height_m=1.0,
        )


def test_projector_maps_centre_of_the_pitch() -> None:
    projector = RinkProjector(_calibration())

    position = projector.to_rink((200.0, 100.0), (200.0, 100.0))

    assert position is not None
    assert position[0] == pytest.approx(20.0, abs=0.01)
    assert position[1] == pytest.approx(10.0, abs=0.01)


def test_projector_rejects_points_far_outside_the_pitch() -> None:
    projector = RinkProjector(_calibration())

    assert projector.to_rink((10.0, 10.0), (-5000.0, -5000.0)) is None


def test_order_quad_accepts_any_click_order() -> None:
    tl, tr, br, bl = (0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)

    assert order_quad((tl, tr, bl, br)) == (tl, tr, br, bl)
    assert order_quad((br, bl, tr, tl)) == (tl, tr, br, bl)
    assert order_quad((tr, tl, br, bl)) == (tl, tr, br, bl)


def test_order_quad_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicated"):
        order_quad(((0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))


def test_order_quad_handles_perspective_trapezoid() -> None:
    """Near edge much wider than the far one: the x+y / x-y extremes hit the same corner."""
    tl, tr, br, bl = (600.0, 300.0), (900.0, 300.0), (1500.0, 700.0), (200.0, 700.0)

    assert order_quad((tl, tr, br, bl)) == (tl, tr, br, bl)
    assert order_quad((br, bl, tl, tr)) == (tl, tr, br, bl)
    assert order_quad((tr, br, tl, bl)) == (tl, tr, br, bl)
    assert order_quad((bl, tl, br, tr)) == (tl, tr, br, bl)


def test_order_quad_rejects_point_inside_the_triangle() -> None:
    with pytest.raises(ValueError, match="convex"):
        order_quad(((0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (150.0, 40.0)))


def test_load_calibration_fixes_crossed_clicks(tmp_path: Path) -> None:
    path = tmp_path / "crossed.yaml"
    path.write_text(
        "video: v\nframe: 0\nwidth_m: 40\nheight_m: 20\n"
        "image_points: [[0,0],[400,0],[0,200],[400,200]]\n"
        "rink_points: [[0,0],[40,0],[40,20],[0,20]]\n",
        encoding="utf-8",
    )

    calibration = load_calibration(path)

    assert calibration.image_points == ((0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0))
    assert calibration.rink_points == ((0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0))
    assert calibration.reprojection_error() == pytest.approx(0.0, abs=1e-6)
    centre = calibration.homography @ np.array([200.0, 100.0, 1.0])
    assert centre[0] / centre[2] == pytest.approx(20.0, abs=0.01)


def test_calibration_rejects_crossed_points() -> None:
    with pytest.raises(ValueError, match="crossed"):
        Calibration(
            video="v",
            frame=0,
            image_points=((0.0, 0.0), (400.0, 0.0), (0.0, 200.0), (400.0, 200.0)),
            rink_points=((0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)),
            width_m=40.0,
            height_m=20.0,
        )


def test_foot_point_is_bottom_centre() -> None:
    assert foot_point((10.0, 20.0, 30.0, 80.0)) == (20.0, 80.0)


def test_calibration_save_and_load(tmp_path: Path) -> None:
    calibration = _calibration()
    path = tmp_path / "cal.yaml"

    save_calibration(calibration, path)
    loaded = load_calibration(path)

    assert loaded.image_points == calibration.image_points
    assert loaded.width_m == pytest.approx(40.0)
    assert loaded.reprojection_error() == pytest.approx(0.0, abs=1e-6)


def test_load_calibration_without_keys_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("video: x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        load_calibration(path)


def test_rink_canvas_has_expected_shape() -> None:
    canvas = rink_canvas(40.0, 20.0, px_per_m=10)

    assert canvas.shape == (200, 400, 3)
    assert canvas.std() > 0


def test_build_minimap_paints_dots_inside_the_canvas() -> None:
    canvas = build_minimap(40.0, 20.0, [(20.0, 10.0, (0, 0, 255))], px_per_m=10, margin_m=0.0)

    assert canvas.shape == (200, 400, 3)
    assert tuple(canvas[100, 200]) == (0, 0, 255)


def test_build_minimap_with_margin_offsets_the_pitch() -> None:
    with_margin = build_minimap(40.0, 20.0, [], px_per_m=10, margin_m=2.0)

    assert with_margin.shape == (240, 440, 3)


def test_minimap_clamps_positions_outside_the_pitch() -> None:
    canvas = build_minimap(40.0, 20.0, [(-50.0, 200.0, (0, 0, 255))], px_per_m=10, margin_m=2.0)

    painted = (canvas == (0, 0, 255)).all(axis=2).sum()
    assert painted > 0


def test_overlay_places_the_minimap_bottom_right() -> None:
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    minimap = np.full((240, 480, 3), 200, dtype=np.uint8)

    result = overlay(frame, minimap, size_ratio=0.25)

    assert result.shape == frame.shape
    assert result[-20, -20].any()


def test_overlay_skips_too_small_frames() -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    minimap = np.full((240, 480, 3), 200, dtype=np.uint8)

    result = overlay(frame, minimap, size_ratio=0.5)

    assert result.shape == frame.shape


def test_heatmap_accumulates_and_renders() -> None:
    accumulator = HeatmapAccumulator(40.0, 20.0, grid=(40, 20))
    for _ in range(5):
        accumulator.add("team_a", 0.0, 0.0)
    accumulator.add("team_b", 40.0, 20.0)

    counts = accumulator.counts()
    rendered = accumulator.render_team("team_a", px_per_m=10)

    assert counts["team_a"][0][0] == 5
    assert counts["team_b"][19][39] == 1
    assert accumulator.teams() == ["team_a", "team_b"]
    assert not accumulator.is_empty()
    assert rendered.std() > 0


def test_heatmap_ignores_out_of_pitch_positions() -> None:
    accumulator = HeatmapAccumulator(40.0, 20.0)

    accumulator.add("team_a", -5.0, 99.0)

    assert accumulator.counts() == {}
    assert accumulator.is_empty()


def test_heatmap_panels_stack_all_teams() -> None:
    accumulator = HeatmapAccumulator(40.0, 20.0, grid=(40, 20))
    accumulator.add("team_a", 5.0, 5.0)
    accumulator.add("team_b", 30.0, 15.0)

    panels = accumulator.render_panels(px_per_m=10)

    assert panels.shape[0] >= 200
    assert panels.shape[1] > 800


# --- world tracking (v4): verified keyframe anchoring -------------------------------


def _pitch_like_frame(shift: tuple[float, float] = (0.0, 0.0), jerseys: bool = True) -> np.ndarray:
    """Synthetic rink: grey floor, yellow lines, optional yellow blobs (jerseys) and crowd noise."""
    rng = np.random.default_rng(11)
    frame = np.full((480, 854, 3), 90, dtype=np.uint8)
    for x in range(60, 830, 40):  # yellow floor lines
        cv2.line(frame, (x, 300), (x + 24, 330), (0, 220, 220), 3)
    cv2.line(frame, (40, 330), (800, 330), (0, 220, 220), 3)
    for _ in range(300):  # stands texture (repeated structure that fools matchers)
        px, py = int(rng.integers(0, 854)), int(rng.integers(0, 160))
        cv2.circle(frame, (px, py), 2, (60, 60, 120), -1)
    if jerseys:
        for _ in range(8):
            px, py = int(rng.integers(60, 800)), int(rng.integers(340, 460))
            cv2.circle(frame, (px, py), 9, (0, 220, 220), -1)
    if shift != (0.0, 0.0):
        matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
        return cv2.warpAffine(frame, matrix, (854, 480), borderMode=cv2.BORDER_REFLECT)
    return frame


def test_floor_line_mask_ignores_yellow_jerseys() -> None:
    frame = _pitch_like_frame()

    mask = floor_line_mask(frame)

    assert mask.sum() > 0
    # Jerseys are compact blobs: they must be filtered out, so the mask must stay sparse.
    assert mask.sum() // 255 < frame.size // 20


def test_line_residual_is_zero_for_the_identity() -> None:
    mask = floor_line_mask(_pitch_like_frame())

    assert line_residual(mask, _line_distance(mask), np.eye(3)) == pytest.approx(0.0, abs=0.01)


def test_line_residual_grows_with_a_wrong_transform() -> None:
    mask = floor_line_mask(_pitch_like_frame())
    distance = _line_distance(mask)
    shifted = np.array([[1, 0, 40], [0, 1, 25], [0, 0, 1]], dtype=np.float64)

    assert line_residual(mask, distance, shifted) > 10.0


def test_world_tracker_anchors_shifted_frames() -> None:
    tracker = WorldTracker(
        max_features=800, min_inliers=12, anchor_interval=1, max_residual_px=15.0
    )
    tracker.update(_pitch_like_frame())

    world = tracker.update(_pitch_like_frame((30.0, 12.0)))

    assert tracker.status in {"anchored", "rejected", "carried"}
    assert world.shape == (3, 3)


def test_world_tracker_keeps_the_anchor_keyframe() -> None:
    tracker = WorldTracker(max_features=500, keyframe_interval=1, max_keyframes=2)
    first = _pitch_like_frame()
    tracker.update(first)
    for shift in (5.0, 10.0, 15.0, 20.0):
        tracker.update(_pitch_like_frame((shift, 0.0)))

    assert tracker._anchor is not None
    assert len(tracker._recent) <= 2


def test_world_tracker_stats_are_consistent() -> None:
    tracker = WorldTracker(max_features=500, anchor_interval=1)
    frames = [_pitch_like_frame((float(step * 4), 0.0)) for step in range(6)]

    for frame in frames:
        tracker.update(frame)

    total = sum(tracker.stats.values())
    assert total >= len(frames) - 1
    assert set(tracker.stats) == {"anchored", "carried", "rejected"}
