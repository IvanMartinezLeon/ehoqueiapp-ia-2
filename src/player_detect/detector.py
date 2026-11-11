"""YOLO-based person detection with optional tiled inference."""

from __future__ import annotations

from itertools import product

import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .config import Settings

# Box format used across the pipeline: (x1, y1, x2, y2, conf, cls)
BOX_COLUMNS = 6


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` to the best available backend."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _tiles(height: int, width: int, grid: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return the pixel windows of a grid x grid tiling with relative overlap."""
    step_h, step_w = height / grid, width / grid
    pad_h, pad_w = int(step_h * overlap), int(step_w * overlap)
    windows: list[tuple[int, int, int, int]] = []
    for row, col in product(range(grid), range(grid)):
        y1 = max(int(row * step_h) - pad_h, 0)
        y2 = min(int((row + 1) * step_h) + pad_h, height)
        x1 = max(int(col * step_w) - pad_w, 0)
        x2 = min(int((col + 1) * step_w) + pad_w, width)
        windows.append((x1, y1, x2, y2))
    return windows


class YoloDetector:
    """Wraps an Ultralytics model and returns person boxes per frame."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.model = YOLO(settings.model)
        self._tile: int | None = None

    @property
    def tile(self) -> int:
        """Effective tiling grid; resolved from the first frame when ``tile_auto`` is on."""
        return self.settings.tile if self._tile is None else self._tile

    @property
    def class_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", None) or {}
        return {int(key): str(name) for key, name in names.items()}

    def _resolve_tile(self, frame: np.ndarray) -> int:
        if self._tile is None:
            if self.settings.tile_auto:
                longest = max(frame.shape[:2])
                self._tile = 2 if longest >= self.settings.tile_auto_min_side else 1
            else:
                self._tile = self.settings.tile
        return self._tile

    def _predict(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        results = self.model.predict(
            source=frames,
            conf=self.settings.conf,
            iou=self.settings.iou,
            imgsz=self.settings.imgsz,
            classes=list(self.settings.detector_classes),
            device=self.device,
            verbose=False,
        )
        boxes: list[np.ndarray] = []
        for result in results:
            if not isinstance(result, Results):
                raise TypeError(f"unexpected prediction result: {type(result)!r}")
            raw_boxes = result.boxes
            if raw_boxes is None:
                boxes.append(np.empty((0, BOX_COLUMNS), dtype=np.float32))
                continue
            data = raw_boxes.cpu().numpy()
            boxes.append(np.hstack([data.xyxy, data.conf[:, None], data.cls[:, None]]))
        return boxes

    def _predict_tiled(self, frame: np.ndarray, tile: int) -> np.ndarray:
        height, width = frame.shape[:2]
        windows = _tiles(height, width, tile, self.settings.tile_overlap)
        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]
        per_crop = self._predict(crops)
        offsets = np.array(
            [[x1, y1, x1, y1, 0.0, 0.0] for x1, y1, _, _ in windows], dtype=np.float32
        )
        stacked = [
            boxes + offset for boxes, offset in zip(per_crop, offsets, strict=True) if len(boxes)
        ]
        if not stacked:
            return np.empty((0, BOX_COLUMNS), dtype=np.float32)
        fused = np.vstack(stacked).astype(np.float32)
        keep = nms(
            torch.from_numpy(fused[:, :4]),
            torch.from_numpy(fused[:, 4]),
            self.settings.iou,
        )
        return fused[keep.numpy()]

    def detect(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Detect persons in each frame, returning ``(N, 6)`` arrays."""
        if not frames:
            return []
        tile = self._resolve_tile(frames[0])
        if tile == 1:
            return self._predict(frames)
        return [self._predict_tiled(frame, tile) for frame in frames]
