"""Multi-object tracking on top of the Ultralytics tracker implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ultralytics
from ultralytics.engine.results import Boxes
from ultralytics.trackers import BOTSORT, OCSORT, BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace

SUPPORTED_TRACKERS = ("bytetrack.yaml", "botsort.yaml", "ocsort.yaml")
TRACKER_YAML_DIR = Path(ultralytics.__file__).parent / "cfg" / "trackers"

# Ultralytics tracker rows are ``[x1, y1, x2, y2, track_id, score, cls, idx]``.
CLS_COLUMN = 6


@dataclass(frozen=True, slots=True)
class Track:
    """A detection associated with a persistent track id."""

    track_id: int
    bbox: tuple[float, float, float, float]
    conf: float
    class_id: int = -1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _create_tracker(tracker_yaml: str, device: str, params: dict[str, Any] | None = None) -> Any:
    config_file = TRACKER_YAML_DIR / tracker_yaml
    if not config_file.exists():
        raise ValueError(f"unknown tracker {tracker_yaml!r}; expected one of {SUPPORTED_TRACKERS}")
    raw: dict[str, Any] = YAML.load(str(config_file))
    overrides = dict(params or {})
    unknown = set(overrides) - set(raw)
    if unknown:
        raise ValueError(
            f"unknown tracker options for {tracker_yaml}: {sorted(unknown)}; "
            f"valid keys: {sorted(raw)}"
        )
    cfg = IterableSimpleNamespace(**{**raw, **overrides})
    cfg.device = device
    if tracker_yaml == "bytetrack.yaml":
        return BYTETracker(cfg)
    if tracker_yaml == "botsort.yaml":
        return BOTSORT(cfg)
    if tracker_yaml == "ocsort.yaml":
        return OCSORT(cfg)
    raise ValueError(f"unknown tracker {tracker_yaml!r}; expected one of {SUPPORTED_TRACKERS}")


class ObjectTracker:
    """Feeds per-frame detections to a tracker and returns boxes with ids."""

    def __init__(
        self, tracker_yaml: str, device: str = "cpu", params: dict[str, Any] | None = None
    ) -> None:
        self.params = dict(params or {})
        self._tracker = _create_tracker(tracker_yaml, device, self.params)

    def update(self, dets: np.ndarray, frame: np.ndarray) -> list[Track]:
        """Update the tracker with one frame of ``N x 6`` detections (xyxy, conf, cls)."""
        if dets.size == 0:
            data = torch.zeros((0, 6), dtype=torch.float32)
        else:
            data = torch.from_numpy(np.ascontiguousarray(dets, dtype=np.float32))
        boxes = Boxes(data, orig_shape=frame.shape[:2]).cpu().numpy()
        rows = self._tracker.update(boxes, frame)
        if rows is None or len(rows) == 0:
            return []
        return [
            Track(
                track_id=int(row[4]),
                bbox=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                conf=float(row[5]),
                class_id=int(row[CLS_COLUMN]) if len(row) > CLS_COLUMN else -1,
            )
            for row in rows
        ]
