"""JSON export of detections and per-track summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from typing import Any

from .tracking import Track


@dataclass(slots=True)
class _TrackAccumulator:
    track_id: int
    frames: int = 0
    first_t: float = 0.0
    last_t: float = 0.0
    path_length: float = 0.0
    conf_sum: float = 0.0
    bbox_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    last_center: tuple[float, float] | None = None
    rink_path: float = 0.0
    rink_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])
    rink_frames: int = 0
    last_rink: tuple[float, float] | None = None
    class_counts: dict[int, int] = field(default_factory=dict)

    def add_rink(self, position: tuple[float, float]) -> None:
        if self.last_rink is not None:
            self.rink_path += hypot(
                position[0] - self.last_rink[0], position[1] - self.last_rink[1]
            )
        self.last_rink = position
        self.rink_sum[0] += position[0]
        self.rink_sum[1] += position[1]
        self.rink_frames += 1

    def add(self, t: float, track: Track) -> None:
        center = track.center
        if self.frames == 0:
            self.first_t = t
        elif self.last_center is not None:
            self.path_length += hypot(
                center[0] - self.last_center[0], center[1] - self.last_center[1]
            )
        self.last_center = center
        self.last_t = t
        self.conf_sum += track.conf
        self.class_counts[track.class_id] = self.class_counts.get(track.class_id, 0) + 1
        for i, value in enumerate(track.bbox):
            self.bbox_sum[i] += value
        self.frames += 1

    def to_dict(self) -> dict[str, object]:
        frames = max(self.frames, 1)
        payload: dict[str, object] = {
            "track_id": self.track_id,
            "frames": self.frames,
            "first_t": round(self.first_t, 3),
            "last_t": round(self.last_t, 3),
            "path_length_px": round(self.path_length, 1),
            "mean_conf": round(self.conf_sum / frames, 4),
            "mean_bbox": [round(v / frames, 1) for v in self.bbox_sum],
        }
        if self.rink_frames:
            payload["rink_path_m"] = round(self.rink_path, 2)
            payload["mean_position_m"] = [
                round(self.rink_sum[0] / self.rink_frames, 2),
                round(self.rink_sum[1] / self.rink_frames, 2),
            ]
        return payload


class ExportCollector:
    """Accumulates per-frame detections and writes the JSON artefacts."""

    def __init__(
        self,
        video: str,
        fps: float,
        width: int,
        height: int,
        stride: int = 1,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.video = video
        self.fps = fps
        self.width = width
        self.height = height
        self.stride = stride
        self.class_names = dict(class_names or {})
        self.frames: list[dict[str, Any]] = []
        self._tracks: dict[int, _TrackAccumulator] = {}
        self._teams: dict[int, dict[str, Any]] = {}
        self._detection_count = 0

    def class_name(self, class_id: int) -> str:
        """Readable class label; ``unknown`` when the tracker reported no class."""
        if class_id < 0:
            return "unknown"
        return self.class_names.get(class_id, f"class_{class_id}")

    def set_teams(self, report: dict[int, dict[str, Any]]) -> None:
        """Attach the per-track team report produced by the classifier."""
        self._teams = report

    def _team_of(self, track_id: int) -> str:
        return str(self._teams.get(track_id, {}).get("team", "unknown"))

    @property
    def detection_count(self) -> int:
        return self._detection_count

    def add(
        self,
        frame_index: int,
        t: float,
        tracks: list[Track],
        rink: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        self._detection_count += len(tracks)
        rink = rink or {}
        self.frames.append(
            {
                "frame": frame_index,
                "t": round(t, 3),
                "detections": [
                    {
                        "track_id": track.track_id,
                        "class_id": track.class_id,
                        "class_name": self.class_name(track.class_id),
                        "bbox": [round(v, 1) for v in track.bbox],
                        "conf": round(track.conf, 4),
                        "center": [round(v, 1) for v in track.center],
                        "rink_m": (
                            [round(v, 2) for v in rink[track.track_id]]
                            if track.track_id in rink
                            else None
                        ),
                    }
                    for track in tracks
                ],
            }
        )
        for track in tracks:
            accumulator = self._tracks.setdefault(track.track_id, _TrackAccumulator(track.track_id))
            accumulator.add(t, track)
            if track.track_id in rink:
                accumulator.add_rink(rink[track.track_id])

    def detections_payload(self) -> dict[str, Any]:
        frames = [
            {
                **frame,
                "detections": [
                    {**det, "team": self._team_of(int(det["track_id"]))}
                    for det in frame["detections"]
                ],
            }
            for frame in self.frames
        ]
        return {
            "video": self.video,
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
            "stride": self.stride,
            "classes": {self.class_name(k): k for k in sorted(self.class_names)},
            "class_counts": self.class_counts(),
            "frames": frames,
        }

    def tracks_payload(self) -> dict[str, Any]:
        summaries = sorted(self._tracks.values(), key=lambda acc: acc.track_id)
        tracks = []
        for acc in summaries:
            payload = {**acc.to_dict(), **self._teams.get(acc.track_id, {"team": "unknown"})}
            if acc.class_counts:
                dominant = max(acc.class_counts.items(), key=lambda item: item[1])[0]
                payload["class_name"] = self.class_name(dominant)
                payload["class_counts"] = {
                    self.class_name(class_id): count
                    for class_id, count in sorted(acc.class_counts.items())
                }
            tracks.append(payload)
        return {
            "video": self.video,
            "track_count": len(tracks),
            "team_counts": self.team_counts(),
            "tracks": tracks,
        }

    def class_counts(self) -> dict[str, int]:
        """Total detections per class label across the whole run."""
        counts: dict[str, int] = {}
        for frame in self.frames:
            for det in frame["detections"]:
                label = str(det["class_name"])
                counts[label] = counts.get(label, 0) + 1
        return counts

    def team_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for track in self._tracks:
            label = self._team_of(track)
            counts[label] = counts.get(label, 0) + 1
        return counts

    def positions_payload(self) -> dict[str, Any]:
        frames = []
        for frame in self.frames:
            positions = [
                {
                    "track_id": det["track_id"],
                    "team": self._team_of(int(det["track_id"])),
                    "x_m": det["rink_m"][0],
                    "y_m": det["rink_m"][1],
                }
                for det in frame["detections"]
                if det.get("rink_m") is not None
            ]
            frames.append({"frame": frame["frame"], "t": frame["t"], "positions": positions})
        return {"video": self.video, "fps": round(self.fps, 3), "frames": frames}

    def write_positions(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.positions_payload()
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return sum(len(frame["positions"]) for frame in payload["frames"])

    def write(self, detections_path: Path, tracks_path: Path) -> int:
        detections_path.parent.mkdir(parents=True, exist_ok=True)
        detections_path.write_text(
            json.dumps(self.detections_payload(), ensure_ascii=False), encoding="utf-8"
        )
        tracks_path.write_text(
            json.dumps(self.tracks_payload(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return len(self._tracks)
