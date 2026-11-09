"""Load the stereo calibration file and rectify frames."""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path(__file__).resolve().parent / "stereoMap.xml"

CALIBRATION_KEYS = (
    "stereoMapL_x",
    "stereoMapL_y",
    "stereoMapR_x",
    "stereoMapR_y",
    "cameraMatrixL",
    "cameraMatrixR",
    "newCameraMatrixL",
    "newCameraMatrixR",
    "projMatrixL",
    "projMatrixR",
    "Q",
)


def _read_matrix(file_storage: cv2.FileStorage, name: str) -> np.ndarray | None:
    node = file_storage.getNode(name)
    if node.empty():
        return None

    matrix = node.mat()
    if matrix is None or matrix.size == 0:
        return None
    return matrix


def write_matrices(path: Path | str, matrices: dict[str, np.ndarray]) -> None:
    """Write matrices in OpenCV XML format.

    The file is written by Python rather than by ``cv2.FileStorage`` because
    OpenCV cannot open paths with non-ASCII characters on Windows.
    """
    storage = cv2.FileStorage(".xml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
    for name, value in matrices.items():
        storage.write(name, value)
    Path(path).write_text(storage.releaseAndGetString(), encoding="utf-8")


@functools.lru_cache(maxsize=None)
def load_calibration(path: Path | str = CALIBRATION_FILE) -> dict[str, np.ndarray | None]:
    """Read every known matrix from ``stereoMap.xml``; missing ones are ``None``."""
    data: dict[str, np.ndarray | None] = dict.fromkeys(CALIBRATION_KEYS)

    try:
        # Read with Python: cv2.FileStorage cannot open non-ASCII paths on Windows.
        content = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        logger.warning("Calibration file not readable: %s (%s)", path, error)
        return data

    cv_file = cv2.FileStorage(content, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    if not cv_file.isOpened():
        logger.warning("Calibration file is not valid OpenCV storage: %s", path)
        return data

    try:
        for key in CALIBRATION_KEYS:
            data[key] = _read_matrix(cv_file, key)
    finally:
        cv_file.release()
    return data


def undistort_rectify(frame_right: np.ndarray, frame_left: np.ndarray, path=CALIBRATION_FILE):
    """Rectify a stereo pair; returns the raw frames if the maps are missing."""
    data = load_calibration(path)
    maps = [data[k] for k in ("stereoMapL_x", "stereoMapL_y", "stereoMapR_x", "stereoMapR_y")]

    if any(m is None for m in maps):
        _warn_once("Stereo maps missing from the calibration file; using raw frames.")
        return frame_right, frame_left

    left_x, left_y, right_x, right_y = maps
    rectified_left = cv2.remap(frame_left, left_x, left_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    rectified_right = cv2.remap(frame_right, right_x, right_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return rectified_right, rectified_left


def get_q_matrix(path=CALIBRATION_FILE) -> np.ndarray | None:
    return load_calibration(path)["Q"]


@functools.lru_cache(maxsize=None)
def _warn_once(message: str) -> None:
    logger.warning(message)
