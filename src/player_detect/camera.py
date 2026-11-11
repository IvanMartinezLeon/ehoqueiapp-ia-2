"""Camera motion estimation between consecutive frames."""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_MAX_FEATURES = 400
DEFAULT_MIN_INLIERS = 40
DEFAULT_RANSAC_THRESHOLD = 2.0


class CameraTracker:
    """Estimates the frame-to-frame image homography caused by camera motion (pan/tilt/zoom).

    Robustness: feature tracking with pyramidal Lucas-Kanade + RANSAC homography.
    When the estimate is unreliable (too few inliers) ``update`` returns ``None`` so the
    caller can keep the last valid transform.
    """

    def __init__(
        self,
        max_features: int = DEFAULT_MAX_FEATURES,
        min_inliers: int = DEFAULT_MIN_INLIERS,
        ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD,
    ) -> None:
        self.max_features = max_features
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        self.last_inliers = 0

    def update(self, previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray | None:
        """Return the 3x3 homography mapping previous-frame pixels to current-frame pixels."""
        corners = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=12
        )
        if corners is None or len(corners) < 8:
            self.last_inliers = 0
            return None
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, corners, corners.copy()
        )
        if tracked is None or status is None:
            self.last_inliers = 0
            return None
        valid = status.ravel() == 1
        if int(valid.sum()) < 8:
            self.last_inliers = 0
            return None
        source = corners[valid].reshape(-1, 2)
        destination = tracked[valid].reshape(-1, 2)
        matrix, mask = cv2.findHomography(
            source, destination, cv2.RANSAC, self.ransac_threshold, maxIters=2000, confidence=0.995
        )
        self.last_inliers = int(mask.sum()) if mask is not None else 0
        if matrix is None or self.last_inliers < self.min_inliers:
            return None
        return matrix


def compose(camera_matrix: np.ndarray, cumulative: np.ndarray) -> np.ndarray:
    """Accumulate a new frame-to-frame homography on top of the calibration-to-current one."""
    composed = camera_matrix @ cumulative
    norm = composed[2, 2]
    if abs(norm) > 1e-9:
        composed = composed / norm
    return composed


def to_reference(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float] | None:
    """Map a point from the current frame into the reference (calibration) frame."""
    inverse = np.linalg.inv(matrix)
    homogeneous = inverse @ np.array([point[0], point[1], 1.0])
    if abs(homogeneous[2]) < 1e-9:
        return None
    return (float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2]))


class AnchorTracker:
    """Registers every frame directly against a fixed reference frame.

    Unlike chaining frame-to-frame homographies, this does NOT accumulate error: each estimate is
    independent, so the drift of frame *n* does not depend on the drift of frame *n-1*.
    It requires the reference frame and the current frame to share enough static scene, which holds
    for panning cameras that keep returning to a similar view.
    """

    def __init__(
        self,
        reference_gray: np.ndarray,
        max_features: int = 3000,
        min_inliers: int = 40,
        ratio: float = 0.75,
        ransac_threshold: float = 3.0,
    ) -> None:
        self.min_inliers = min_inliers
        self.ratio = ratio
        self.ransac_threshold = ransac_threshold
        self.last_inliers = 0
        self._orb = cv2.ORB.create(max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._reference_points, self._reference_desc = self._orb.detectAndCompute(
            reference_gray, None
        )

    def update(self, current_gray: np.ndarray) -> np.ndarray | None:
        """Return the homography mapping reference-frame pixels to current-frame pixels."""
        if self._reference_desc is None or self._reference_points is None:
            return None
        points, desc = self._orb.detectAndCompute(current_gray, None)
        if desc is None or points is None or len(points) < 10:
            self.last_inliers = 0
            return None
        matches = self._matcher.knnMatch(self._reference_desc, desc, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < self.ratio * m[1].distance]
        if len(good) < max(self.min_inliers, 10):
            self.last_inliers = 0
            return None
        source = np.array(
            [self._reference_points[m.queryIdx].pt for m in good], dtype=np.float32
        ).reshape(-1, 2)
        target = np.array([points[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 2)
        matrix, mask = cv2.findHomography(
            source, target, cv2.RANSAC, self.ransac_threshold, maxIters=3000, confidence=0.999
        )
        self.last_inliers = int(mask.sum()) if mask is not None else 0
        if matrix is None or self.last_inliers < self.min_inliers:
            return None
        return matrix
