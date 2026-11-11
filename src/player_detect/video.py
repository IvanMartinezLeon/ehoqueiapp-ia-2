"""Video reading and writing."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FPS = 25.0


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoSource:
    """Streaming video reader (never loads the whole file into memory)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open video: {self._path}")
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.info = VideoInfo(
            path=self._path,
            fps=float(fps) if fps and fps > 1 else DEFAULT_FPS,
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self._capture.release()

    def frames(
        self, stride: int = 1, max_seconds: float | None = None
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield ``(frame_index, timestamp, frame)`` honouring stride and duration limit."""
        max_frames = int(max_seconds * self.info.fps) if max_seconds else None
        index = 0
        while True:
            if max_frames is not None and index >= max_frames:
                break
            ok, frame = self._capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, index / self.info.fps, frame
            index += 1

    def estimated_output_frames(self, stride: int = 1, max_seconds: float | None = None) -> int:
        total = self.info.frame_count
        if max_seconds:
            total = min(total, int(max_seconds * self.info.fps))
        return max(total // stride, 1)


class VideoSink:
    """Writes an annotated video, re-encoding to H.264 when ffmpeg is available."""

    def __init__(self, path: Path, info: VideoInfo) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temp = Path(tempfile.mkstemp(suffix=".mp4", dir=self.path.parent)[1])
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._temp), fourcc, info.fps, (info.width, info.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {self.path}")
        self._size = (info.width, info.height)
        self._frames = 0

    def __enter__(self) -> VideoSink:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        if (frame.shape[1], frame.shape[0]) != self._size:
            raise ValueError(
                f"frame size {frame.shape[1]}x{frame.shape[0]} does not match "
                f"the writer {self._size[0]}x{self._size[1]} ({self.path})"
            )
        for _ in range(repeat):
            self._writer.write(frame)
            self._frames += 1

    def close(self) -> None:
        self._writer.release()
        if self._frames == 0:
            self._temp.unlink(missing_ok=True)
            return
        if not self._reencode():
            self._temp.replace(self.path)

    def _reencode(self) -> bool:
        if shutil.which("ffmpeg") is None:
            return False
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(self._temp),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0:
            return False
        self._temp.unlink(missing_ok=True)
        return True
