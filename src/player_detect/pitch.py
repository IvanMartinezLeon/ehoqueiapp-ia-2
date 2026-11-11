"""Pitch calibration and image -> pitch coordinates projection."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

MIN_CALIBRATION_POINTS = 4


def order_quad(
    points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Return the four points ordered top-left, top-right, bottom-right, bottom-left.

    Works regardless of the order the user clicked. The corners are sorted by angle around
    their centroid, which is stable for perspective views: the classic ``x + y`` / ``x - y``
    extremes collapse onto the same corner when the near edge is much wider than the far one
    (a pitch filmed from one end), so they are only used to pick which corner goes first.
    """
    if len(points) != 4:
        raise ValueError(f"order_quad expects 4 points, got {len(points)}")
    unique: list[tuple[float, float]] = []
    for x, y in points:
        corner = (float(x), float(y))
        if corner not in unique:
            unique.append(corner)
    if len(unique) != 4:
        raise ValueError("degenerate calibration: duplicated points")
    centre_x = sum(x for x, _ in unique) / 4.0
    centre_y = sum(y for _, y in unique) / 4.0
    ordered = sorted(unique, key=lambda p: atan2(p[1] - centre_y, p[0] - centre_x))
    start = min(range(4), key=lambda i: ordered[i][0] + ordered[i][1])
    ordered = ordered[start:] + ordered[:start]
    if _signed_area(ordered) < 0:
        ordered = [ordered[0], *reversed(ordered[1:])]
    quad = (ordered[0], ordered[1], ordered[2], ordered[3])
    if _is_twisted(quad):
        raise ValueError("degenerate calibration: points do not form a convex quadrilateral")
    return quad


def _signed_area(quad: list[tuple[float, float]]) -> float:
    """Shoelace area; positive means TL -> TR -> BR -> BL in image coordinates (y down)."""
    return 0.5 * sum(
        quad[i][0] * quad[(i + 1) % 4][1] - quad[(i + 1) % 4][0] * quad[i][1] for i in range(4)
    )


def _is_twisted(quad: tuple[tuple[float, float], ...]) -> bool:
    """True when the quadrilateral in the given vertex order self-intersects."""

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = [cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) for i in range(4)]
    return not (all(sign > 0 for sign in signs) or all(sign < 0 for sign in signs))


@dataclass(frozen=True, slots=True)
class Calibration:
    """Maps pixels of the reference frame to pitch metres."""

    video: str
    frame: int
    image_points: tuple[tuple[float, float], ...]
    rink_points: tuple[tuple[float, float], ...]
    width_m: float
    height_m: float

    def __post_init__(self) -> None:
        if len(self.image_points) != len(self.rink_points):
            raise ValueError("image_points and rink_points must have the same length")
        if len(self.image_points) < MIN_CALIBRATION_POINTS:
            raise ValueError(f"at least {MIN_CALIBRATION_POINTS} points are required")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("pitch dimensions must be positive")
        if _is_twisted(tuple(self.image_points)):
            raise ValueError(
                "calibration image points form a crossed quadrilateral; "
                "click the corners in order TL, TR, BR, BL (or let order_quad fix them)"
            )

    @property
    def homography(self) -> np.ndarray:
        """3x3 matrix mapping reference-frame pixels to pitch metres."""
        image = np.array(self.image_points, dtype=np.float64)
        rink = np.array(self.rink_points, dtype=np.float64)
        matrix, _ = cv2.findHomography(image, rink, 0)
        if matrix is None:
            raise ValueError("degenerate calibration points: homography cannot be estimated")
        return matrix

    def reprojection_error(self) -> float:
        """Mean round-trip error in pixels over the calibration points."""
        matrix = self.homography
        errors = []
        for image_point, rink_point in zip(self.image_points, self.rink_points, strict=True):
            projected = _apply(matrix, image_point)
            errors.append(
                float(np.hypot(projected[0] - rink_point[0], projected[1] - rink_point[1]))
            )
        return float(np.mean(errors))


def _apply(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    homogeneous = matrix @ np.array([point[0], point[1], 1.0])
    return (float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2]))


def load_calibration(path: Path) -> Calibration:
    """Read a calibration YAML file."""
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = {"image_points", "rink_points"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"calibration {path} is missing keys: {sorted(missing)}")
    raw_image_points = tuple((float(x), float(y)) for x, y in data["image_points"])
    raw_rink_points = tuple((float(x), float(y)) for x, y in data["rink_points"])
    if _is_twisted(raw_image_points):
        # Click order was not TL, TR, BR, BL: normalise both quads the same way.
        image_points = order_quad(raw_image_points)
        rink_points = order_quad(raw_rink_points)
    else:
        image_points, rink_points = raw_image_points, raw_rink_points
    width_m, height_m = _infer_size(rink_points, data)
    return Calibration(
        video=str(data.get("video", "")),
        frame=int(data.get("frame", 0)),
        image_points=image_points,
        rink_points=rink_points,
        width_m=width_m,
        height_m=height_m,
    )


def _infer_size(
    rink_points: tuple[tuple[float, float], ...], data: dict[str, Any]
) -> tuple[float, float]:
    if "width_m" in data and "height_m" in data:
        return float(data["width_m"]), float(data["height_m"])
    xs = [point[0] for point in rink_points]
    ys = [point[1] for point in rink_points]
    return float(max(xs)), float(max(ys))


def save_calibration(calibration: Calibration, path: Path) -> None:
    """Write a calibration YAML file."""
    payload = {
        "video": calibration.video,
        "frame": calibration.frame,
        "width_m": calibration.width_m,
        "height_m": calibration.height_m,
        "image_points": [list(point) for point in calibration.image_points],
        "rink_points": [list(point) for point in calibration.rink_points],
    }
    Path(path).write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def foot_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-centre of a bounding box: the best cheap approximation of where a player stands."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


class RinkProjector:
    """Projects player positions from image space to pitch metres."""

    def __init__(
        self, calibration: Calibration, margin_m: float = 2.0, max_offset_m: float = 15.0
    ) -> None:
        self.calibration = calibration
        self.margin_m = margin_m
        self.max_offset_m = max_offset_m
        self._homography = calibration.homography

    def is_inside(self, x_m: float, y_m: float) -> bool:
        """True when the position falls within the pitch plus the drawing margin."""
        return (
            -self.margin_m <= x_m <= self.calibration.width_m + self.margin_m
            and -self.margin_m <= y_m <= self.calibration.height_m + self.margin_m
        )

    def to_rink(
        self, point: tuple[float, float], reference_point: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Project a pixel position (already mapped into the reference frame) to pitch metres.

        Positions slightly outside the pitch are kept (the caller draws them on the margin ring);
        only wildly divergent projections (camera-tracking drift, bad calibration) are dropped.
        """
        if reference_point is None:
            return None
        x_m, y_m = _apply(self._homography, reference_point)
        limit_x = self.calibration.width_m + self.max_offset_m
        limit_y = self.calibration.height_m + self.max_offset_m
        if not (-self.max_offset_m <= x_m <= limit_x and -self.max_offset_m <= y_m <= limit_y):
            return None
        return (x_m, y_m)
