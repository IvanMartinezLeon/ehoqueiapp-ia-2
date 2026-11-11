"""Command line interface for player detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import DEFAULT_CONFIG_PATH, Settings, build_settings, load_yaml_config
from .pipeline import RunSummary, run
from .tracking import SUPPORTED_TRACKERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="player-detect",
        description="Detecta y sigue jugadores en un vídeo (YOLO11 + tracking).",
    )
    parser.add_argument("--video", type=Path, required=True, help="vídeo de entrada")
    parser.add_argument(
        "--out", type=Path, default=None, help="carpeta de salida (def. runs/<video>)"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help=f"YAML de config (def. {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument("--model", default=None, help="pesos YOLO (def. yolo11m.pt)")
    parser.add_argument("--conf", type=float, default=None, help="confianza mínima")
    parser.add_argument(
        "--detector-classes",
        default=None,
        help="índices de clase del modelo a conservar, ej. '0,1,2' (def. 0)",
    )
    parser.add_argument("--iou", type=float, default=None, help="IoU de NMS")
    parser.add_argument("--imgsz", type=int, default=None, help="resolución de inferencia")
    parser.add_argument("--batch", type=int, default=None, help="frames por inferencia")
    parser.add_argument(
        "--tile",
        type=str,
        default=None,
        help="rejilla NxN para gran angular, o 'auto' (2 si el lado mayor >= tile_auto_min_side)",
    )
    parser.add_argument(
        "--tile-overlap", type=float, default=None, help="solape relativo entre tiles"
    )
    parser.add_argument("--tracker", default=None, choices=SUPPORTED_TRACKERS, help="tracker YAML")
    parser.add_argument(
        "--tracker-opt",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="opción del tracker, repetible (ej. --tracker-opt track_high_thresh=0.15)",
    )
    parser.add_argument("--device", default=None, help="auto | mps | cuda | cuda:0 | cpu")
    parser.add_argument(
        "--max-seconds", type=float, default=None, help="procesar solo los primeros N segundos"
    )
    parser.add_argument("--stride", type=int, default=None, help="procesar 1 de cada N frames")
    parser.add_argument(
        "--no-teams", action="store_true", help="no clasificar por equipo (color por track id)"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="YAML de calibración de pista (activa el minimapa)",
    )
    parser.add_argument("--no-minimap", action="store_true", help="no incrustar el minimapa")
    parser.add_argument("--no-heatmap", action="store_true", help="no generar heatmap.png")
    parser.add_argument(
        "--minimap-size", type=float, default=None, help="ancho del minimapa / ancho del frame"
    )
    parser.add_argument("--no-video", action="store_true", help="no escribir vídeo anotado")
    parser.add_argument("--no-json", action="store_true", help="no escribir JSON")
    parser.add_argument(
        "--no-conf", action="store_true", help="ocultar la confianza en las etiquetas"
    )
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "model": args.model,
        "conf": args.conf,
        "detector_classes": _parse_classes(args.detector_classes),
        "iou": args.iou,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "tile_overlap": args.tile_overlap,
        "tracker": args.tracker,
        "tracker_params": _parse_tracker_opts(args.tracker_opt),
        "device": args.device,
        "max_seconds": args.max_seconds,
        "stride": args.stride,
        "save_video": False if args.no_video else None,
        "save_json": False if args.no_json else None,
        "draw_conf": False if args.no_conf else None,
        "classify_teams": False if args.no_teams else None,
        "calibration": args.calibration,
        "minimap": False if args.no_minimap else None,
        "heatmap": False if args.no_heatmap else None,
        "minimap_size": args.minimap_size,
    }
    overrides = {key: value for key, value in mapping.items() if value is not None}
    tile = _parse_tile(args.tile)
    if tile is not None:
        overrides.update(tile)
    return overrides


def _parse_classes(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    try:
        return tuple(int(value) for value in raw.replace(" ", "").split(",") if value)
    except ValueError as error:
        raise ValueError(f"--detector-classes debe ser una lista de enteros: {raw!r}") from error


def _parse_tile(raw: str | None) -> dict[str, Any] | None:
    """Accept an integer grid or ``auto``; returns the config fields to override."""
    if raw is None:
        return None
    if raw.strip().lower() == "auto":
        return {"tile": 1, "tile_auto": True}
    try:
        return {"tile": int(raw), "tile_auto": False}
    except ValueError as error:
        raise ValueError(f"--tile debe ser un entero o 'auto': {raw!r}") from error


def _parse_tracker_opts(raw: list[str] | None) -> dict[str, Any] | None:
    """Parse repeated ``KEY=VALUE`` options, using YAML semantics for the value."""
    if not raw:
        return None
    params: dict[str, Any] = {}
    for item in raw:
        key, separator, value = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"--tracker-opt espera KEY=VALUE: {item!r}")
        params[key] = yaml.safe_load(value.strip())
    return params


def resolve_settings(args: argparse.Namespace) -> Settings:
    config_path = args.config if args.config is not None else DEFAULT_CONFIG_PATH
    out_dir = args.out if args.out is not None else Path("runs") / args.video.stem
    return build_settings(
        video=args.video,
        out_dir=out_dir,
        config=load_yaml_config(config_path),
        overrides=_overrides(args),
    )


def format_summary(summary: RunSummary, settings: Settings) -> str:
    lines = [
        f"vídeo            : {summary.video}",
        f"salida           : {summary.out_dir}",
        f"modelo           : {settings.model} (tile={summary.tile}, stride={settings.stride})",
        f"frames           : {summary.frames_processed} procesados de {summary.frames_read} leídos",
        f"tiempo           : {summary.elapsed_s:.1f} s ({summary.fps:.1f} fps)",
        f"detecciones      : {summary.detections}",
        f"trayectorias     : {summary.tracks}",
    ]
    if settings.tracker_params:
        options = ", ".join(f"{key}={value}" for key, value in settings.tracker_params.items())
        lines.append(f"tracker          : {settings.tracker} ({options})")
    if summary.class_counts:
        counts = ", ".join(f"{label}={count}" for label, count in summary.class_counts.items())
        lines.append(f"por clase        : {counts}")
    if summary.team_counts:
        counts = ", ".join(f"{label}={count}" for label, count in summary.team_counts.items())
        lines.append(f"equipos          : {counts}")
    if summary.detections:
        coverage = summary.projected_points / summary.detections
        if summary.projected_points and coverage < 0.6:
            lines.append(
                f"aviso            : solo el {coverage:.0%} de las detecciones llegan al minimapa. "
                "Revisa la calibración: el cuadrilátero debe cubrir la zona de juego visible."
            )
    if summary.projected_points:
        lines.append(
            f"puntos en pista  : {summary.projected_points} "
            f"(en el anillo: {summary.outside_points}, "
            f"descartados: {summary.rejected_points}, sin referencia: {summary.no_reference_points})"
        )
    if summary.anchored_frames:
        lines.append(
            f"anclaje cámara   : {summary.anchored_frames} anclados / "
            f"{summary.fallback_frames} encadenados"
        )
    if summary.minimap_video:
        lines.append(f"minimapa vídeo   : {summary.minimap_video}")
    if summary.heatmap_png:
        lines.append(f"heatmap          : {summary.heatmap_png}")
    if summary.positions_json:
        lines.append(f"posiciones json  : {summary.positions_json}")
    if summary.annotated_video:
        lines.append(f"vídeo anotado    : {summary.annotated_video}")
    if summary.detections_json:
        lines.append(f"detecciones json : {summary.detections_json}")
    if summary.tracks_json:
        lines.append(f"tracks json      : {summary.tracks_json}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.video.exists():
        print(f"error: no existe el vídeo {args.video}", file=sys.stderr)
        return 2
    try:
        settings = resolve_settings(args)
    except ValueError as exc:
        print(f"error de configuración: {exc}", file=sys.stderr)
        return 2
    summary = run(settings)
    print(format_summary(summary, settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
