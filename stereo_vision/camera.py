"""Stereo camera capture."""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

from stereo_vision.config import CameraConfig

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or read."""


def _open_capture(index: int, width: int | None, height: int | None) -> cv2.VideoCapture:
    # DirectShow opens USB cameras faster and more reliably on Windows.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        raise CameraError(f"Cannot open camera {index}.")

    if width and height:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


class StereoCamera:
    """Left/right camera pair, usable as a context manager.

    With ``set_resolution=False`` the cameras keep their default resolution.
    """

    def __init__(self, config: CameraConfig, set_resolution: bool = True):
        self.config = config
        self._set_resolution = set_resolution
        self._left: cv2.VideoCapture | None = None
        self._right: cv2.VideoCapture | None = None

    def open(self) -> StereoCamera:
        width = self.config.width if self._set_resolution else None
        height = self.config.height if self._set_resolution else None
        try:
            self._right = _open_capture(self.config.right_index, width, height)
            self._left = _open_capture(self.config.left_index, width, height)
        except CameraError:
            self.release()
            raise
        logger.info(
            "Opened cameras: left=%d, right=%d", self.config.left_index, self.config.right_index
        )
        return self

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(left, right)`` frames, or raise :class:`CameraError`."""
        if self._left is None or self._right is None:
            raise CameraError("Cameras are not open.")

        ok_right, frame_right = self._right.read()
        ok_left, frame_left = self._left.read()
        if not ok_left:
            raise CameraError(f"Cannot read from left camera {self.config.left_index}.")
        if not ok_right:
            raise CameraError(f"Cannot read from right camera {self.config.right_index}.")
        return frame_left, frame_right

    def release(self) -> None:
        for capture in (self._left, self._right):
            if capture is not None:
                capture.release()
        self._left = self._right = None

    def __enter__(self) -> StereoCamera:
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.release()
