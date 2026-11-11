#!/usr/bin/env python3
"""Patch labelImg 1.8.6 so it runs with PyQt5 >= 5.15.11.

labelImg passes float values where newer PyQt5 requires int (QPainter.drawLine/drawRect,
QScrollBar.setValue, QSpinBox.setValue...). The result is a hard crash on mouse move or scroll.
This script casts those values to int, which preserves the original behaviour. Idempotent:
running it twice changes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# (file to patch, original code, patched code)
PATCHES: tuple[tuple[str, str, str], ...] = (
    # --- libs/canvas.py: float coordinates passed to QPainter ---
    (
        "libs/canvas.py",
        "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())",
        "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))",
    ),
    (
        "libs/canvas.py",
        "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
        "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))",
    ),
    (
        "libs/canvas.py",
        "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
        "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))",
    ),
    (
        "libs/canvas.py",
        "p.drawLine(self.current_point.x(), 0, self.current_point.x(), self.pixmap.height())",
        "p.drawLine(int(self.current_point.x()), 0, int(self.current_point.x()), int(self.pixmap.height()))",
    ),
    (
        "libs/canvas.py",
        "p.drawLine(0, self.current_point.y(), self.pixmap.width(), self.current_point.y())",
        "p.drawLine(0, int(self.current_point.y()), int(self.pixmap.width()), int(self.current_point.y()))",
    ),
    # --- labelImg/labelImg.py: float values passed to scroll bars and the zoom spin box ---
    (
        "labelImg/labelImg.py",
        "bar.setValue(bar.value() + bar.singleStep() * units)",
        "bar.setValue(int(bar.value() + bar.singleStep() * units))",
    ),
    (
        "labelImg/labelImg.py",
        "self.zoom_widget.setValue(value)",
        "self.zoom_widget.setValue(int(value))",
    ),
    (
        "labelImg/labelImg.py",
        "h_bar.setValue(new_h_bar_value)",
        "h_bar.setValue(int(new_h_bar_value))",
    ),
    (
        "labelImg/labelImg.py",
        "v_bar.setValue(new_v_bar_value)",
        "v_bar.setValue(int(new_v_bar_value))",
    ),
)


def site_packages(venv_root: Path) -> Path:
    candidates = sorted((venv_root / "lib").glob("python*/site-packages"))
    if not candidates:
        raise FileNotFoundError(f"no se encuentra site-packages en {venv_root}")
    return candidates[0]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    packages = site_packages(root / ".venv")
    touched: dict[str, int] = {}
    pending: list[str] = []

    for relative, old, new in PATCHES:
        target = packages / relative
        if not target.exists():
            pending.append(f"{relative} (no existe)")
            continue
        text = target.read_text(encoding="utf-8")
        if new in text:
            continue
        if old not in text:
            pending.append(f"{relative}: patrón no encontrado")
            continue
        target.write_text(text.replace(old, new), encoding="utf-8")
        touched[relative] = touched.get(relative, 0) + 1

    if touched:
        detail = ", ".join(f"{name} ({count})" for name, count in sorted(touched.items()))
        print(f"labelImg parcheado: {detail}")
    else:
        print("labelImg ya estaba parcheado")
    for warning in pending:
        print(f"aviso: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
