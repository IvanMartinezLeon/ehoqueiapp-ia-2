"""Configuration model and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PERSON_CLASS_ID = 0
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable run configuration."""

    video: Path
    out_dir: Path
    model: str = "yolo11n.pt"
    conf: float = 0.35
    iou: float = 0.55
    imgsz: int = 960
    batch: int = 8
    tile: int = 1
    tile_overlap: float = 0.2
    tile_auto: bool = False
    tile_auto_min_side: int = 1920
    tracker: str = "bytetrack.yaml"
    tracker_params: dict[str, Any] = field(default_factory=dict)
    device: str = "auto"
    max_seconds: float | None = None
    stride: int = 1
    save_video: bool = True
    save_json: bool = True
    draw_conf: bool = True
    detector_classes: tuple[int, ...] = (PERSON_CLASS_ID,)
    classify_teams: bool = True
    min_color_samples: int = 8
    classify_every: int = 15
    min_box_height: int = 40
    calibration: Path | None = None
    minimap: bool = True
    heatmap: bool = True
    minimap_size: float = 0.22
    min_inliers: int = 40
    pitch_margin_m: float = 2.0
    pitch_max_offset_m: float = 15.0
    anchor_camera: bool = True
    world_anchor: bool = True
    world_max_residual_px: float = 20.0
    world_anchor_interval: int = 5
    anchor_features: int = 3000
    team_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "team_a": (46, 190, 240),
            "team_b": (235, 160, 60),
            "referee": (20, 20, 20),
            "unknown": (180, 180, 180),
        }
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.conf < 1.0:
            raise ValueError("conf must be in (0, 1)")
        if not 0.0 < self.iou <= 1.0:
            raise ValueError("iou must be in (0, 1]")
        if self.tile < 1:
            raise ValueError("tile must be >= 1")
        if not 0.0 <= self.tile_overlap < 0.5:
            raise ValueError("tile_overlap must be in [0, 0.5)")
        if self.tile_auto_min_side < 640:
            raise ValueError("tile_auto_min_side must be >= 640")
        if not isinstance(self.tracker_params, dict):
            raise ValueError("tracker_params must be a mapping of option -> value")
        object.__setattr__(self, "tracker_params", dict(self.tracker_params))
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.imgsz < 64:
            raise ValueError("imgsz must be >= 64")
        if self.min_color_samples < 1:
            raise ValueError("min_color_samples must be >= 1")
        if self.classify_every < 1:
            raise ValueError("classify_every must be >= 1")
        if not 0.05 <= self.minimap_size <= 0.6:
            raise ValueError("minimap_size must be in [0.05, 0.6]")
        if self.pitch_max_offset_m <= self.pitch_margin_m:
            raise ValueError("pitch_max_offset_m must be greater than pitch_margin_m")
        if self.world_max_residual_px <= 0:
            raise ValueError("world_max_residual_px must be positive")
        if self.world_anchor_interval < 1:
            raise ValueError("world_anchor_interval must be >= 1")
        if self.anchor_features < 100:
            raise ValueError("anchor_features must be >= 100")
        if self.min_inliers < 8:
            raise ValueError("min_inliers must be >= 8")
        if self.calibration is not None:
            object.__setattr__(self, "calibration", Path(self.calibration))
        classes = tuple(int(value) for value in self.detector_classes)
        if not classes or any(value < 0 for value in classes):
            raise ValueError("detector_classes must be a non-empty list of non-negative ints")
        object.__setattr__(self, "detector_classes", classes)
        normalized: dict[str, tuple[int, int, int]] = {}
        for label, color in self.team_colors.items():
            channels = tuple(int(c) for c in color)
            if len(channels) != 3 or any(not 0 <= c <= 255 for c in channels):
                raise ValueError(f"invalid BGR color for {label!r}: {color!r}")
            normalized[label] = (channels[0], channels[1], channels[2])
        object.__setattr__(self, "team_colors", normalized)

    @property
    def effective_batch(self) -> int:
        """Batch size capped to 1 when tiling, to keep memory bounded."""
        return 1 if self.tile > 1 or self.tile_auto else self.batch


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    """Read a YAML config, returning an empty dict when absent."""
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping")
    return data


def build_settings(
    video: Path, out_dir: Path, config: dict[str, Any], overrides: dict[str, Any]
) -> Settings:
    """Merge YAML config with explicit CLI overrides (overrides win)."""
    known = {f.name for f in fields(Settings)} - {"video", "out_dir"}
    unknown = set(config) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    merged = {k: v for k, v in config.items() if v is not None}
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(video=Path(video), out_dir=Path(out_dir), **merged)
