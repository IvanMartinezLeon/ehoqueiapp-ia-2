"""Top-down pitch minimap and per-team occupancy heatmap."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

PITCH_BGR = (34, 92, 60)
LINE_BGR = (235, 235, 235)
DOT_RADIUS = 4
GOAL_DEPTH_M = 1.0
DEFAULT_MARGIN_M = 2.0
TEAM_PANEL_ORDER = ("team_a", "team_b", "referee", "unknown")
DEFAULT_PANEL_COLORS: dict[str, tuple[int, int, int]] = {
    "team_a": (46, 190, 240),
    "team_b": (235, 160, 60),
    "referee": (20, 20, 20),
    "unknown": (180, 180, 180),
}


def rink_canvas(
    width_m: float, height_m: float, px_per_m: int = 12, margin_m: float = 0.0
) -> np.ndarray:
    """Top-down pitch with painted lines, optionally surrounded by a margin ring."""
    total_w_m = width_m + 2 * margin_m
    total_h_m = height_m + 2 * margin_m
    width_px = max(int(total_w_m * px_per_m), 8)
    height_px = max(int(total_h_m * px_per_m), 8)
    canvas = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    canvas[:, :] = PITCH_BGR
    offset_x = int(margin_m * px_per_m)
    offset_y = int(margin_m * px_per_m)
    pitch_w = max(int(width_m * px_per_m), 4)
    pitch_h = max(int(height_m * px_per_m), 4)
    thickness = max(1, px_per_m // 6)
    cv2.rectangle(
        canvas,
        (offset_x, offset_y),
        (offset_x + pitch_w - 1, offset_y + pitch_h - 1),
        LINE_BGR,
        thickness,
    )
    centre_x = offset_x + pitch_w // 2
    cv2.line(
        canvas,
        (centre_x, offset_y),
        (centre_x, offset_y + pitch_h - 1),
        LINE_BGR,
        thickness,
    )
    cv2.circle(
        canvas,
        (centre_x, offset_y + pitch_h // 2),
        int(3.0 * px_per_m),
        LINE_BGR,
        thickness,
        cv2.LINE_AA,
    )
    goal_depth = max(int(GOAL_DEPTH_M * px_per_m), 2)
    goal_height = max(int(3.0 * px_per_m), 4)
    top = offset_y + (pitch_h - goal_height) // 2
    cv2.rectangle(
        canvas, (offset_x, top), (offset_x + goal_depth, top + goal_height), LINE_BGR, thickness
    )
    cv2.rectangle(
        canvas,
        (offset_x + pitch_w - goal_depth - 1, top),
        (offset_x + pitch_w - 1, top + goal_height),
        LINE_BGR,
        thickness,
    )
    return canvas


def plot_positions(
    canvas: np.ndarray,
    positions: list[tuple[float, float, tuple[int, int, int]]],
    width_m: float,
    height_m: float,
    margin_m: float = 0.0,
    radius: int = DOT_RADIUS,
) -> np.ndarray:
    """Draw one dot per position. Positions beyond the pitch are clamped to the canvas edge."""
    height_px, width_px = canvas.shape[:2]
    total_w_m = width_m + 2 * margin_m
    total_h_m = height_m + 2 * margin_m
    for x_m, y_m, color in positions:
        px = round((x_m + margin_m) / total_w_m * (width_px - 1))
        py = round((y_m + margin_m) / total_h_m * (height_px - 1))
        px = min(max(px, radius), width_px - radius - 1)
        py = min(max(py, radius), height_px - radius - 1)
        cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def build_minimap(
    width_m: float,
    height_m: float,
    positions: list[tuple[float, float, tuple[int, int, int]]],
    px_per_m: int = 12,
    margin_m: float = DEFAULT_MARGIN_M,
) -> np.ndarray:
    """Pitch canvas with the current player positions."""
    canvas = rink_canvas(width_m, height_m, px_per_m, margin_m)
    return plot_positions(canvas, positions, width_m, height_m, margin_m)


def overlay(
    frame: np.ndarray, minimap: np.ndarray, size_ratio: float = 0.22, margin: int = 12
) -> np.ndarray:
    """Paste the minimap in the bottom-right corner of the frame (in place)."""
    frame_h, frame_w = frame.shape[:2]
    if frame_h <= 6 * margin or frame_w <= 6 * margin:
        return frame
    target_w = max(int(frame_w * size_ratio), 40)
    scale = target_w / minimap.shape[1]
    target_h = int(minimap.shape[0] * scale)
    if target_h + 2 * margin >= frame_h:
        scale = (frame_h - 3 * margin) / minimap.shape[0]
        target_w, target_h = int(minimap.shape[1] * scale), int(minimap.shape[0] * scale)
    if target_w < 8 or target_h < 8:
        return frame
    resized = cv2.resize(minimap, (target_w, target_h), interpolation=cv2.INTER_AREA)
    x0, y0 = frame_w - target_w - margin, frame_h - target_h - margin
    if x0 < 0 or y0 < 0:
        return frame
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + target_w + 2, y0 + target_h + 2), (0, 0, 0), -1)
    frame[y0 : y0 + target_h, x0 : x0 + target_w] = resized
    return frame


class HeatmapAccumulator:
    """Occupancy histogram per team over the pitch, in a fixed grid."""

    def __init__(self, width_m: float, height_m: float, grid: tuple[int, int] = (80, 40)) -> None:
        self.width_m = width_m
        self.height_m = height_m
        self.grid = grid
        self._counts: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros((self.grid[1], self.grid[0]), dtype=np.float64)
        )

    def ensure(self, labels: list[str] | tuple[str, ...]) -> None:
        """Create empty buckets so every team appears in the output, even without samples."""
        for label in labels:
            _ = self._counts[label]

    def add(self, team: str, x_m: float, y_m: float) -> None:
        column = int(x_m / self.width_m * (self.grid[0] - 1))
        row = int(y_m / self.height_m * (self.grid[1] - 1))
        if 0 <= column < self.grid[0] and 0 <= row < self.grid[1]:
            self._counts[team][row, column] += 1.0

    def teams(self) -> list[str]:
        return sorted(self._counts)

    def counts(self) -> dict[str, list[list[int]]]:
        return {team: grid.astype(int).tolist() for team, grid in self._counts.items()}

    def totals(self) -> dict[str, int]:
        return {team: int(grid.sum()) for team, grid in self._counts.items()}

    def is_empty(self) -> bool:
        return all(grid.sum() == 0 for grid in self._counts.values())

    def render_team(
        self,
        team: str,
        color: tuple[int, int, int] | None = None,
        px_per_m: int = 20,
        blur: int = 9,
    ) -> np.ndarray:
        """Density in the team's own colour over the pitch outline."""
        grid = self._counts.get(team)
        base = rink_canvas(self.width_m, self.height_m, px_per_m, DEFAULT_MARGIN_M)
        if grid is None or grid.sum() == 0:
            return base
        normalized = (grid / grid.max() * 255).astype(np.uint8)
        pitch_h = int(self.height_m * px_per_m)
        pitch_w = int(self.width_m * px_per_m)
        offset = int(DEFAULT_MARGIN_M * px_per_m)
        resized = cv2.resize(normalized, (pitch_w, pitch_h), interpolation=cv2.INTER_LINEAR)
        if blur > 1:
            resized = cv2.GaussianBlur(resized, (blur, blur), 0)
        tint = np.zeros_like(base)
        tint[:, :] = color or (255, 255, 255)
        alpha = np.zeros(base.shape[:2], dtype=np.float32)
        alpha[offset : offset + pitch_h, offset : offset + pitch_w] = (
            resized.astype(np.float32) / 255.0
        )
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)[..., None] * 0.85
        blended = base.astype(np.float32) * (1 - alpha) + tint.astype(np.float32) * alpha
        return blended.astype(np.uint8)

    def render_panels(
        self,
        colors: dict[str, tuple[int, int, int]] | None = None,
        labels: tuple[str, ...] = TEAM_PANEL_ORDER,
        px_per_m: int = 20,
        pad: int = 8,
    ) -> np.ndarray:
        """One labelled panel per team, in that team's colour."""
        palette = {**DEFAULT_PANEL_COLORS, **(colors or {})}
        totals = self.totals()
        panels = []
        for label in labels:
            panel = self.render_team(label, palette.get(label), px_per_m)
            cv2.rectangle(panel, (6, 8), (22, 24), palette.get(label, (255, 255, 255)), -1)
            cv2.putText(
                panel,
                f"{label}  ({totals.get(label, 0)} puntos)",
                (30, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(
                cv2.copyMakeBorder(
                    panel, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(15, 15, 15)
                )
            )
        return np.hstack(panels)


def heatmap_payload(accumulator: HeatmapAccumulator) -> dict[str, Any]:
    """Serialisable heatmap metadata."""
    return {
        "width_m": accumulator.width_m,
        "height_m": accumulator.height_m,
        "grid": list(accumulator.grid),
        "totals": accumulator.totals(),
        "counts": accumulator.counts(),
    }
