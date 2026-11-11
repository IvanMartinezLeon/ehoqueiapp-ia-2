"""Jersey-colour based team classification."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np

REFEREE = "referee"
TEAM_A = "team_a"
TEAM_B = "team_b"
UNKNOWN = "unknown"
TEAM_LABELS = (REFEREE, TEAM_A, TEAM_B, UNKNOWN)

# Default boxes: BGR tuples.
DEFAULT_TEAM_COLORS: dict[str, tuple[int, int, int]] = {
    TEAM_A: (46, 190, 240),
    TEAM_B: (235, 160, 60),
    REFEREE: (20, 20, 20),
    UNKNOWN: (180, 180, 180),
}

# Torso window as fractions of the bounding box.
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45
TORSO_LEFT = 0.25
TORSO_RIGHT = 0.75
MIN_SATURATED_RATIO = 0.2
SATURATION_THRESHOLD = 40
EMA_ALPHA = 0.3


@dataclass(frozen=True, slots=True)
class ColorFeature:
    """Hue as a unit vector (wrap-around safe) plus saturation and value."""

    cos_h: float
    sin_h: float
    saturation: float
    value: float

    @property
    def hue(self) -> float:
        """Circular mean hue in degrees (0-360)."""
        return degrees(atan2(self.sin_h, self.cos_h)) % 360.0

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.cos_h, self.sin_h, self.saturation, self.value], dtype=np.float64)

    @property
    def hsv(self) -> tuple[float, float, float]:
        return (round(self.hue, 1), round(self.saturation, 1), round(self.value, 1))


def torso_region(
    bbox: tuple[float, float, float, float], frame_shape: tuple[int, ...], min_box_height: int
) -> tuple[slice, slice] | None:
    """Return the torso slices for a bbox, or None when the box is too small."""
    x1, y1, x2, y2 = bbox
    height, width = y2 - y1, x2 - x1
    if height < min_box_height or width < 8:
        return None
    frame_h, frame_w = frame_shape[0], frame_shape[1]
    top = int(y1 + height * TORSO_TOP)
    bottom = int(y1 + height * TORSO_BOTTOM)
    left = int(x1 + width * TORSO_LEFT)
    right = int(x1 + width * TORSO_RIGHT)
    top, left = max(top, 0), max(left, 0)
    bottom, right = min(bottom, frame_h), min(right, frame_w)
    if bottom - top < 2 or right - left < 2:
        return None
    return slice(top, bottom), slice(left, right)


def sample_jersey_color(
    frame: np.ndarray, bbox: tuple[float, float, float, float], min_box_height: int = 40
) -> ColorFeature | None:
    """Average HSV of the torso window, ignoring very dark and unsaturated pixels when possible."""
    region = torso_region(bbox, frame.shape, min_box_height)
    if region is None:
        return None
    crop = frame[region]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    visible = hsv[hsv[:, 2] >= 15]
    if len(visible):
        hsv = visible
    saturated = hsv[hsv[:, 1] >= SATURATION_THRESHOLD]
    if len(saturated) >= max(int(len(hsv) * MIN_SATURATED_RATIO), 1):
        hsv = saturated
    angles = np.deg2rad(hsv[:, 0] * 2.0)  # OpenCV hue 0-179 -> degrees
    cos_values, sin_values = np.cos(angles), np.sin(angles)
    cos_mean, sin_mean = float(np.median(cos_values)), float(np.median(sin_values))
    norm = max(np.hypot(cos_mean, sin_mean), 1e-6)
    return ColorFeature(
        cos_h=cos_mean / norm,
        sin_h=sin_mean / norm,
        saturation=float(np.median(hsv[:, 1])),
        value=float(np.median(hsv[:, 2])),
    )


def _blend(previous: ColorFeature, new: ColorFeature, alpha: float) -> ColorFeature:
    return ColorFeature(
        cos_h=previous.cos_h * (1 - alpha) + new.cos_h * alpha,
        sin_h=previous.sin_h * (1 - alpha) + new.sin_h * alpha,
        saturation=previous.saturation * (1 - alpha) + new.saturation * alpha,
        value=previous.value * (1 - alpha) + new.value * alpha,
    )


def kmeans(
    points: np.ndarray, k: int, seed: int = 0, iterations: int = 50, n_init: int = 3
) -> np.ndarray:
    """Deterministic k-means (best inertia over ``n_init`` seeded starts)."""
    if len(points) < k:
        return np.zeros(len(points), dtype=int)
    best_labels: np.ndarray | None = None
    best_inertia = float("inf")
    for init in range(n_init):
        rng = np.random.default_rng(seed + init)
        centers = points[rng.choice(len(points), k, replace=False)].copy()
        labels = np.zeros(len(points), dtype=int)
        for _ in range(iterations):
            distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            for index in range(k):
                members = points[labels == index]
                if len(members):
                    centers[index] = members.mean(axis=0)
        inertia = float(((points - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    assert best_labels is not None
    return best_labels


class TeamClassifier:
    """Accumulates per-track jersey colours and assigns team labels."""

    def __init__(
        self,
        min_color_samples: int = 8,
        min_box_height: int = 40,
        classify_every: int = 15,
        colors: dict[str, tuple[int, int, int]] | None = None,
    ) -> None:
        self.min_color_samples = max(min_color_samples, 1)
        self.min_box_height = min_box_height
        self.classify_every = max(classify_every, 1)
        self.colors = {**DEFAULT_TEAM_COLORS, **(colors or {})}
        self._features: dict[int, ColorFeature] = {}
        self._samples: dict[int, int] = {}
        self._labels: dict[int, str] = {}

    def observe(self, frame: np.ndarray, tracks: list) -> None:
        """Feed one frame of tracks into the colour statistics."""
        for track in tracks:
            feature = sample_jersey_color(frame, track.bbox, self.min_box_height)
            if feature is None:
                continue
            previous = self._features.get(track.track_id)
            self._features[track.track_id] = (
                feature if previous is None else _blend(previous, feature, EMA_ALPHA)
            )
            self._samples[track.track_id] = self._samples.get(track.track_id, 0) + 1

    def should_classify(self, frame_index: int) -> bool:
        return frame_index % self.classify_every == 0

    def classify(self) -> dict[int, str]:
        """Recompute labels. Existing labels are sticky and never reassigned."""
        candidates = sorted(
            track_id for track_id, count in self._samples.items() if count >= self.min_color_samples
        )
        if len(candidates) < 3:
            return dict(self._labels)
        points = np.vstack([self._features[track_id].vector for track_id in candidates])
        labels = kmeans(points, 3, seed=0)
        groups: dict[int, list[int]] = {}
        for index, cluster in enumerate(labels):
            groups.setdefault(int(cluster), []).append(candidates[index])
        if len(groups) < 3:
            return dict(self._labels)
        referee_cluster = min(groups, key=lambda c: float(points[labels == c][:, 3].mean()))
        remaining = [c for c in groups if c != referee_cluster]

        def team_sort_key(cluster: int) -> tuple[float, float]:
            members = points[labels == cluster]
            hue = degrees(atan2(members[:, 1].mean(), members[:, 0].mean())) % 360.0
            return (-float(members[:, 2].mean()), hue)

        ordered = sorted(remaining, key=team_sort_key)
        assignment = {referee_cluster: REFEREE, ordered[0]: TEAM_A, ordered[1]: TEAM_B}
        for cluster, label in assignment.items():
            for track_id in groups[cluster]:
                self._labels.setdefault(track_id, label)
        return dict(self._labels)

    @property
    def labels(self) -> dict[int, str]:
        return dict(self._labels)

    def label_of(self, track_id: int) -> str:
        return self._labels.get(track_id, UNKNOWN)

    def box_colors(self) -> dict[int, tuple[int, int, int]]:
        return {track_id: self.colors[label] for track_id, label in self._labels.items()}

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {label: 0 for label in TEAM_LABELS}
        for label in self._labels.values():
            counts[label] = counts.get(label, 0) + 1
        return counts

    def track_colors(self) -> dict[int, dict[str, object]]:
        """Per-track colour diagnostics for the JSON export."""
        report: dict[int, dict[str, object]] = {}
        for track_id, label in self._labels.items():
            feature = self._features.get(track_id)
            report[track_id] = {
                "team": label,
                "mean_hsv": list(feature.hsv) if feature else None,
                "color_samples": self._samples.get(track_id, 0),
            }
        return report
