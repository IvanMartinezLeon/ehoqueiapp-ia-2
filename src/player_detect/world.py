"""Drift-controlled world tracking: keyframe registration verified against the painted lines.

Why this exists
---------------
Chaining frame-to-frame homographies accumulates error (48 px after 4 s on the test clip).
Registering every frame against frame 0 has the opposite problem: when the camera pans away there
is no overlap, and when it does overlap the matcher sometimes locks onto repeated structure
(crowd, LED boards) and returns a confidently wrong homography.

This tracker keeps several keyframes, registers each new frame against the most promising ones and
**verifies every candidate** by measuring how well the painted floor lines of the current frame
line up with the keyframe's lines. Only verified transforms are accepted; otherwise the previous
transform is carried forward with frame-to-frame tracking, which is safe for short distances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

FLOOR_TOP_RATIO = 0.33
LINE_HSV_LOW = (18, 50, 80)
LINE_HSV_HIGH = (42, 255, 255)
MIN_LINE_AREA = 150
MIN_LINE_LENGTH = 70
MIN_LINE_ASPECT = 3.5


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Accept grayscale or BGR frames and always return grayscale."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def floor_line_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask with the long painted floor lines, ignoring yellow jerseys (blobby, short)."""
    height = frame.shape[0]
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LINE_HSV_LOW, LINE_HSV_HIGH)
    mask[: int(height * FLOOR_TOP_RATIO), :] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        width = stats[index, cv2.CC_STAT_WIDTH]
        height_box = stats[index, cv2.CC_STAT_HEIGHT]
        longest = max(width, height_box)
        aspect = longest / max(min(width, height_box), 1)
        if area >= MIN_LINE_AREA and longest >= MIN_LINE_LENGTH and aspect >= MIN_LINE_ASPECT:
            keep[labels == index] = 255
    return keep


def _line_distance(mask: np.ndarray) -> np.ndarray:
    """Distance transform to the nearest line pixel (the reference for verification)."""
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)


def line_residual(mask: np.ndarray, distance: np.ndarray, transform: np.ndarray) -> float:
    """Mean distance (px) from the transformed mask pixels to the reference lines.

    ``transform`` maps current-frame pixels into the reference frame. Lower is better.
    """
    pixels = int(mask.sum() // 255)
    if pixels < 120:
        return float("nan")
    inverse = np.linalg.inv(transform)
    ys, xs = np.nonzero(mask)
    homogeneous = inverse @ np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    denominator = homogeneous[2]
    denominator[np.abs(denominator) < 1e-9] = 1e-9
    mapped_x = np.clip((homogeneous[0] / denominator).round().astype(int), 0, distance.shape[1] - 1)
    mapped_y = np.clip((homogeneous[1] / denominator).round().astype(int), 0, distance.shape[0] - 1)
    return float(distance[mapped_y, mapped_x].mean())


@dataclass
class Keyframe:
    """A reference view with a known transform to the world (calibration) frame."""

    gray: np.ndarray
    transform: np.ndarray
    points: list = field(default_factory=list)
    descriptors: np.ndarray | None = None


class WorldTracker:
    """Registers frames against verified keyframes to keep the world transform drift-free."""

    def __init__(
        self,
        max_features: int = 1500,
        min_inliers: int = 40,
        max_residual_px: float = 20.0,
        keyframe_interval: int = 25,
        max_keyframes: int = 3,
        max_missed: int = 5,
        anchor_interval: int = 5,
        strong_inliers: int = 120,
    ) -> None:
        self.max_residual_px = max_residual_px
        self.keyframe_interval = max(keyframe_interval, 1)
        self.max_keyframes = max(max_keyframes, 1)
        self._orb = cv2.ORB.create(max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._min_inliers = min_inliers
        self._anchor: Keyframe | None = None
        self._recent: list[Keyframe] = []
        self._reference_distance: np.ndarray | None = None
        self._missed = 0
        self._max_missed = max_missed
        self._anchor_interval = max(anchor_interval, 1)
        self._strong_inliers = strong_inliers
        self._since_anchor = 0
        self._previous_gray: np.ndarray | None = None
        self._world = np.eye(3)
        self._frames_since_keyframe = 0
        self.status = "init"
        self.last_residual = float("nan")
        self.last_inliers = 0
        self.stats = {"anchored": 0, "carried": 0, "rejected": 0}

    @property
    def world(self) -> np.ndarray:
        return self._world

    def _empty_keyframe(self, frame: np.ndarray, transform: np.ndarray) -> Keyframe:
        gray = _to_gray(frame)
        points, descriptors = self._orb.detectAndCompute(gray, None)
        return Keyframe(
            gray=gray,
            transform=transform,
            points=list(points) if points is not None else [],
            descriptors=descriptors,
        )

    def add_keyframe(self, frame: np.ndarray, transform: np.ndarray) -> None:
        """Register a reference view (``frame`` is BGR). The first one is pinned as the anchor."""
        keyframe = self._empty_keyframe(frame, transform)
        if self._anchor is None:
            self._anchor = keyframe
            return
        self._recent.append(keyframe)
        if len(self._recent) > max(self.max_keyframes - 1, 1):
            self._recent.pop(0)
        self._frames_since_keyframe = 0

    @property
    def _candidates(self) -> list[Keyframe]:
        """Anchor first (absolute reference), then the most recent verified views."""
        candidates = list(reversed(self._recent))
        if self._anchor is not None:
            candidates.append(self._anchor)
        return candidates

    def _match(self, keyframe: Keyframe, frame: np.ndarray) -> tuple[np.ndarray | None, int]:
        points, descriptors = self._orb.detectAndCompute(_to_gray(frame), None)
        if descriptors is None or points is None or keyframe.descriptors is None:
            return None, 0
        matches = self._matcher.knnMatch(keyframe.descriptors, descriptors, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < max(self._min_inliers, 10):
            return None, 0
        source = np.array([keyframe.points[m.queryIdx].pt for m in good], dtype=np.float32).reshape(
            -1, 2
        )
        target = np.array([points[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 2)
        matrix, inlier_mask = cv2.findHomography(
            source, target, cv2.RANSAC, 3.0, maxIters=3000, confidence=0.999
        )
        if matrix is None or inlier_mask is None:
            return None, 0
        return matrix, int(inlier_mask.sum())

    def update(self, frame: np.ndarray, previous_frame: np.ndarray | None = None) -> np.ndarray:
        """Return the world transform for the current frame (maps calibration pixels to this frame).

        Every candidate is verified against the painted lines of the calibration frame, so a
        confidently wrong match is rejected instead of silently corrupting the projection.
        """
        mask = floor_line_mask(frame)
        if self._anchor is None or self._reference_distance is None:
            self.add_keyframe(frame, np.eye(3))
            self._reference_distance = _line_distance(mask)
            self._previous_gray = _to_gray(frame)
            self.status = "keyframe"
            return self._world

        # Anchoring every frame is expensive and unnecessary: between anchors the transform is
        # carried with frame-to-frame tracking, which stays accurate for a handful of frames.
        if self._since_anchor < self._anchor_interval and self.status != "keyframe":
            self._since_anchor += 1
            self.stats["carried"] += 1
            self.status = "carried"
            self._carry(
                _to_gray(frame),
                previous_frame if previous_frame is not None else self._previous_gray,
            )
            self._previous_gray = _to_gray(frame)
            self._frames_since_keyframe += 1
            return self._world

        best: tuple[float, np.ndarray] | None = None
        any_candidate = False
        for keyframe in self._candidates:
            matrix, inliers = self._match(keyframe, frame)
            if matrix is None or inliers < self._min_inliers:
                continue
            any_candidate = True
            world = matrix @ keyframe.transform
            residual = line_residual(mask, self._reference_distance, world)
            unverifiable = residual != residual  # no floor lines visible in this frame
            if unverifiable:
                # No lines to compare against: only trust a match that is itself very strong.
                if inliers < self._strong_inliers:
                    continue
                residual = float(
                    self.max_residual_px
                )  # neutral score, never beats a verified match
            elif residual > self.max_residual_px:
                continue
            if best is None or residual < best[0]:
                best = (residual, world)

        if best is not None:
            self.last_residual, self._world = best
            self.status = "anchored"
            self.stats["anchored"] += 1
            self._missed = 0
            self._since_anchor = 0
        else:
            self.status = "rejected" if any_candidate else "carried"
            self.stats[self.status] += 1
            self._missed += 1
            fallback = previous_frame if previous_frame is not None else self._previous_gray
            self._carry(_to_gray(frame), fallback)

        self._previous_gray = _to_gray(frame)
        self._frames_since_keyframe += 1
        should_add = (
            self.status == "anchored" and self._frames_since_keyframe >= self.keyframe_interval
        )
        if should_add or self._missed >= self._max_missed:
            self.add_keyframe(frame, self._world)
            self._missed = 0
        return self._world

    def _carry(self, gray: np.ndarray, previous_frame: np.ndarray | None) -> None:
        """Hold the last world transform, optionally refining it with frame-to-frame features."""
        if previous_frame is None:
            return
        previous = _to_gray(previous_frame)
        tracker = cv2.goodFeaturesToTrack(
            previous, maxCorners=300, qualityLevel=0.01, minDistance=12
        )
        if tracker is None or len(tracker) < 8:
            return
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, tracker, tracker.copy())
        if tracked is None or status is None:
            return
        valid = status.ravel() == 1
        if int(valid.sum()) < 8:
            return
        source = tracker[valid].reshape(-1, 2)
        target = tracked[valid].reshape(-1, 2)
        matrix, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
        if matrix is None or inlier_mask is None or int(inlier_mask.sum()) < 20:
            return
        composed = matrix @ self._world
        if abs(composed[2, 2]) > 1e-9:
            composed = composed / composed[2, 2]
        self._world = composed
