"""Player detection and tracking for futsal videos."""

from .camera import CameraTracker, compose, to_reference
from .config import Settings
from .detector import YoloDetector, resolve_device
from .export import ExportCollector
from .minimap import HeatmapAccumulator, build_minimap, overlay
from .pipeline import RunSummary, run
from .pitch import Calibration, RinkProjector, load_calibration, save_calibration
from .teams import TeamClassifier
from .tracking import ObjectTracker, Track

__all__ = [
    "Calibration",
    "CameraTracker",
    "ExportCollector",
    "HeatmapAccumulator",
    "ObjectTracker",
    "RinkProjector",
    "RunSummary",
    "Settings",
    "TeamClassifier",
    "Track",
    "YoloDetector",
    "build_minimap",
    "compose",
    "load_calibration",
    "overlay",
    "resolve_device",
    "run",
    "save_calibration",
    "to_reference",
]

__version__ = "0.1.0"
