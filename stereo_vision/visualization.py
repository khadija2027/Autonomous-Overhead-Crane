"""Drawing helpers for the live display."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from stereo_vision.detection import Detection

GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_HEIGHT_PX = 22


def draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.6, color=GREEN) -> None:
    cv2.putText(frame, text, origin, FONT, scale, color, 2)


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
        cv2.circle(frame, det.center, 5, RED, -1)
        draw_text(frame, f"{det.class_name} {det.conf:.2f}", (x1, y1 - 10), scale=0.8)


def draw_tracking_lost(frame: np.ndarray) -> None:
    draw_text(frame, "TRACKING LOST", (75, 50), scale=1.0, color=RED)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    draw_text(frame, f"FPS: {int(fps)}", (20, 450), scale=1.2)


def draw_measurement(frame: np.ndarray, lines: Sequence[str], top: int) -> int:
    """Draw a block of measurement lines and return the top of the next block."""
    if len(lines) == 1:
        draw_text(frame, lines[0], (50, top), scale=0.8)
        return top + 30

    for i, line in enumerate(lines):
        draw_text(frame, line, (50, top + i * LINE_HEIGHT_PX))
    return top + len(lines) * LINE_HEIGHT_PX + 4
