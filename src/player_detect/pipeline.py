"""End-to-end processing pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .camera import AnchorTracker, CameraTracker, compose, to_reference
from .config import Settings
from .detector import YoloDetector
from .draw import annotate
from .export import ExportCollector
from .minimap import (
    TEAM_PANEL_ORDER,
    HeatmapAccumulator,
    build_minimap,
    heatmap_payload,
    overlay,
)
from .pitch import RinkProjector, foot_point, load_calibration
from .teams import UNKNOWN, TeamClassifier
from .tracking import ObjectTracker, Track
from .video import VideoInfo, VideoSink, VideoSource
from .world import WorldTracker

MINIMAP_PX_PER_M = 12
MINIMAP_VIDEO_PX_PER_M = 20
HEATMAP_GRID = (80, 40)


@dataclass(frozen=True, slots=True)
class RunSummary:
    video: Path
    out_dir: Path
    frames_processed: int
    frames_read: int
    elapsed_s: float
    detections: int
    tracks: int
    annotated_video: Path | None
    detections_json: Path | None
    tracks_json: Path | None
    tile: int = 1
    team_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    minimap_video: Path | None = None
    heatmap_png: Path | None = None
    positions_json: Path | None = None
    projected_points: int = 0
    rejected_points: int = 0
    outside_points: int = 0
    no_reference_points: int = 0
    anchored_frames: int = 0
    fallback_frames: int = 0
    line_residual_px: float = float("nan")

    @property
    def fps(self) -> float:
        return self.frames_processed / self.elapsed_s if self.elapsed_s > 0 else 0.0


class _RinkState:
    """Mutable state for pitch projection across frames."""

    def __init__(
        self,
        projector: RinkProjector,
        min_inliers: int,
        with_heatmap: bool,
        anchor: bool = True,
        anchor_features: int = 3000,
        world: bool = True,
        world_max_residual_px: float = 20.0,
        world_anchor_interval: int = 5,
    ) -> None:
        self.projector = projector
        self.camera = CameraTracker(min_inliers=min_inliers)
        self.cumulative = np.eye(3)
        self.previous_gray: np.ndarray | None = None
        self.anchor: AnchorTracker | None = None
        self.world: WorldTracker | None = None
        self._world_enabled = world
        self._world_max_residual_px = world_max_residual_px
        self._world_anchor_interval = world_anchor_interval
        self.anchor_enabled = anchor
        self.anchor_features = anchor_features
        self.anchor_min_inliers = min_inliers
        self.anchored = 0
        self.fallback = 0
        self.heatmap = (
            HeatmapAccumulator(
                projector.calibration.width_m, projector.calibration.height_m, HEATMAP_GRID
            )
            if with_heatmap
            else None
        )
        self.projected = 0
        self.rejected = 0
        self.outside = 0
        self.no_reference = 0
        if self.heatmap is not None:
            self.heatmap.ensure(TEAM_PANEL_ORDER)

    def update_camera(self, frame: np.ndarray) -> None:
        if self._world_enabled:
            if self.world is None:
                self.world = WorldTracker(
                    max_features=self.anchor_features,
                    min_inliers=self.anchor_min_inliers,
                    max_residual_px=self._world_max_residual_px,
                    anchor_interval=self._world_anchor_interval,
                )
            self.cumulative = self.world.update(frame)
            self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.world.status == "anchored":
                self.anchored += 1
            elif self.world.status in {"carried", "rejected"}:
                self.fallback += 1
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            # First processed frame == calibration reference frame.
            self.previous_gray = gray
            if self.anchor_enabled:
                self.anchor = AnchorTracker(
                    gray,
                    max_features=self.anchor_features,
                    min_inliers=self.anchor_min_inliers,
                )
            return
        if self.anchor is not None:
            matrix = self.anchor.update(gray)
            if matrix is not None:
                self.cumulative = matrix
                self.anchored += 1
                self.previous_gray = gray
                return
        chained = self.camera.update(self.previous_gray, gray)
        if chained is not None:
            self.cumulative = compose(chained, self.cumulative)
        self.fallback += 1
        self.previous_gray = gray

    def project(self, tracks: list[Track]) -> dict[int, tuple[float, float]]:
        positions: dict[int, tuple[float, float]] = {}
        for track in tracks:
            foot = foot_point(track.bbox)
            reference = to_reference(self.cumulative, foot)
            if reference is None:
                self.no_reference += 1
                continue
            rink = self.projector.to_rink(foot, reference)
            if rink is None:
                self.rejected += 1
                continue
            if not self.projector.is_inside(*rink):
                self.outside += 1
            positions[track.track_id] = rink
        self.projected += len(positions)
        return positions


def _annotated_name(video: Path) -> str:
    return f"{video.stem}_annotated.mp4"


def _build_classifier(settings: Settings) -> TeamClassifier | None:
    if not settings.classify_teams:
        return None
    return TeamClassifier(
        min_color_samples=settings.min_color_samples,
        min_box_height=settings.min_box_height,
        classify_every=settings.classify_every,
        colors=settings.team_colors,
    )


def run(settings: Settings) -> RunSummary:
    """Process one video and write the annotated video plus JSON artefacts."""
    start = time.perf_counter()
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    annotated_name = _annotated_name(settings.video)

    collector: ExportCollector | None = None
    sink: VideoSink | None = None
    minimap_sink: VideoSink | None = None
    rink: _RinkState | None = None
    processed = 0
    read = 0
    detection_total = 0
    class_counts_live: dict[str, int] = {}
    seen_track_ids: set[int] = set()

    with VideoSource(settings.video) as source:
        info = source.info
        detector = YoloDetector(settings)
        tracker = ObjectTracker(
            settings.tracker, device=detector.device, params=settings.tracker_params
        )
        classifier = _build_classifier(settings)

        if settings.calibration is not None:
            calibration = load_calibration(settings.calibration)
            if calibration.frame > 0:
                raise ValueError(
                    f"calibration reference frame is {calibration.frame}; use frame 0 "
                    "or cut the video so the reference frame is first"
                )
            rink = _RinkState(
                RinkProjector(calibration, settings.pitch_margin_m, settings.pitch_max_offset_m),
                settings.min_inliers,
                settings.heatmap,
                anchor=settings.anchor_camera,
                anchor_features=settings.anchor_features,
                world=settings.world_anchor,
                world_max_residual_px=settings.world_max_residual_px,
                world_anchor_interval=settings.world_anchor_interval,
            )
            if settings.minimap:
                canvas_margin = settings.pitch_margin_m
                canvas_height = int(
                    (calibration.height_m + 2 * canvas_margin) * MINIMAP_VIDEO_PX_PER_M
                )
                canvas_width = int(
                    (calibration.width_m + 2 * canvas_margin) * MINIMAP_VIDEO_PX_PER_M
                )
                minimap_sink = VideoSink(
                    settings.out_dir / "minimap.mp4",
                    VideoInfo(
                        path=settings.out_dir / "minimap.mp4",
                        fps=info.fps,
                        width=canvas_width,
                        height=canvas_height,
                        frame_count=0,
                    ),
                )

        if settings.save_json:
            collector = ExportCollector(
                video=str(settings.video),
                fps=info.fps,
                width=info.width,
                height=info.height,
                stride=settings.stride,
                class_names=detector.class_names,
            )
        if settings.save_video:
            sink = VideoSink(settings.out_dir / annotated_name, info)

        progress = tqdm(
            total=source.estimated_output_frames(settings.stride, settings.max_seconds),
            desc=settings.video.name,
            unit="frame",
            file=None,
        )
        batch_frames: list[np.ndarray] = []
        batch_meta: list[tuple[int, float]] = []

        def flush() -> None:
            nonlocal processed, detection_total
            if not batch_frames:
                return
            detections = detector.detect(batch_frames)
            for (frame_index, timestamp), frame, dets in zip(
                batch_meta, batch_frames, detections, strict=True
            ):
                if rink is not None:
                    rink.update_camera(frame)
                tracks: list[Track] = tracker.update(dets, frame)
                detection_total += len(tracks)
                for track in tracks:
                    seen_track_ids.add(track.track_id)
                    label = _class_label(detector.class_names, track.class_id)
                    class_counts_live[label] = class_counts_live.get(label, 0) + 1
                if classifier is not None:
                    classifier.observe(frame, tracks)
                    if classifier.should_classify(frame_index):
                        classifier.classify()
                positions: dict[int, tuple[float, float]] = {}
                if rink is not None:
                    positions = rink.project(tracks)
                    if rink.heatmap is not None:
                        for track_id, (x_m, y_m) in positions.items():
                            rink.heatmap.add(_team_label(classifier, track_id), x_m, y_m)
                if sink is not None:
                    colors = classifier.box_colors() if classifier is not None else None
                    canvas = annotate(frame, tracks, settings.draw_conf, colors)
                    if rink is not None and settings.minimap:
                        calibration = rink.projector.calibration
                        dots = [
                            (x_m, y_m, _dot_color(classifier, track_id))
                            for track_id, (x_m, y_m) in positions.items()
                        ]
                        margin_m = settings.pitch_margin_m
                        mini = build_minimap(
                            calibration.width_m,
                            calibration.height_m,
                            dots,
                            MINIMAP_PX_PER_M,
                            margin_m,
                        )
                        relative = build_minimap(
                            calibration.width_m,
                            calibration.height_m,
                            dots,
                            MINIMAP_VIDEO_PX_PER_M,
                            margin_m,
                        )
                        if minimap_sink is not None:
                            minimap_sink.write(relative, repeat=settings.stride)
                        overlay(canvas, mini, settings.minimap_size)
                    sink.write(canvas, repeat=settings.stride)
                if collector is not None:
                    collector.add(frame_index, timestamp, tracks, positions)
                processed += 1
            progress.update(len(batch_frames))
            batch_frames.clear()
            batch_meta.clear()

        try:
            for frame_index, timestamp, frame in source.frames(
                settings.stride, settings.max_seconds
            ):
                read += 1
                batch_frames.append(frame)
                batch_meta.append((frame_index, timestamp))
                if len(batch_frames) >= settings.effective_batch:
                    flush()
            flush()
        finally:
            progress.close()
            if sink is not None:
                sink.close()
            if minimap_sink is not None:
                minimap_sink.close()

    elapsed = time.perf_counter() - start
    detections_json: Path | None = None
    tracks_json: Path | None = None
    positions_json: Path | None = None
    heatmap_png: Path | None = None
    minimap_video: Path | None = None
    track_count: int
    team_counts: dict[str, int] = {}
    class_counts: dict[str, int]
    projected_points = 0
    rejected_points = 0
    outside_points = 0
    no_reference_points = 0
    anchored_frames = 0
    fallback_frames = 0
    line_residual_px = float("nan")

    if classifier is not None:
        classifier.classify()
        team_counts = classifier.counts()
        unlabelled = len(seen_track_ids) - len(classifier.track_colors())
        if unlabelled > 0:
            team_counts[UNKNOWN] = team_counts.get(UNKNOWN, 0) + unlabelled
    if collector is not None:
        if classifier is not None:
            collector.set_teams(classifier.track_colors())
        detections_json = settings.out_dir / "detections.json"
        tracks_json = settings.out_dir / "tracks.json"
        collector.write(detections_json, tracks_json)
    track_count = len(seen_track_ids)
    class_counts = dict(class_counts_live)
    if rink is not None:
        projected_points = rink.projected
        rejected_points = rink.rejected
        outside_points = rink.outside
        no_reference_points = rink.no_reference
        anchored_frames = rink.anchored
        fallback_frames = rink.fallback
        if rink.world is not None:
            line_residual_px = rink.world.last_residual
        if collector is not None:
            positions_json = settings.out_dir / "positions.json"
            collector.write_positions(positions_json)
        if minimap_sink is not None:
            minimap_video = minimap_sink.path
        if rink.heatmap is not None and not rink.heatmap.is_empty():
            heatmap_png = settings.out_dir / "heatmap.png"
            cv2.imwrite(
                str(heatmap_png),
                rink.heatmap.render_panels(classifier.colors if classifier is not None else None),
            )
            (settings.out_dir / "heatmap_counts.json").write_text(
                _json_dumps(heatmap_payload(rink.heatmap)), encoding="utf-8"
            )

    return RunSummary(
        video=settings.video,
        out_dir=settings.out_dir,
        frames_processed=processed,
        frames_read=read,
        elapsed_s=elapsed,
        detections=detection_total,
        tracks=track_count,
        tile=detector.tile,
        annotated_video=(settings.out_dir / annotated_name) if sink is not None else None,
        detections_json=detections_json,
        tracks_json=tracks_json,
        team_counts=team_counts,
        class_counts=class_counts,
        minimap_video=minimap_video,
        heatmap_png=heatmap_png,
        positions_json=positions_json,
        projected_points=projected_points,
        rejected_points=rejected_points,
        outside_points=outside_points,
        no_reference_points=no_reference_points,
        anchored_frames=anchored_frames,
        fallback_frames=fallback_frames,
        line_residual_px=line_residual_px,
    )


def _class_label(class_names: dict[int, str], class_id: int) -> str:
    """Readable class label; ``unknown`` when the tracker reported no class."""
    if class_id < 0:
        return "unknown"
    return class_names.get(class_id, f"class_{class_id}")


def _team_label(classifier: TeamClassifier | None, track_id: int) -> str:
    return classifier.label_of(track_id) if classifier is not None else UNKNOWN


def _dot_color(classifier: TeamClassifier | None, track_id: int) -> tuple[int, int, int]:
    label = _team_label(classifier, track_id)
    colors = classifier.colors if classifier is not None else {}
    return colors.get(label, (200, 200, 200))


def _json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)
