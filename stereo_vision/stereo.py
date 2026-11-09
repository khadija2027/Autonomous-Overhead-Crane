"""Stereo matching, triangulation and coordinate conversion."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from stereo_vision.calibration import CALIBRATION_FILE, _warn_once, load_calibration
from stereo_vision.config import MatchingConfig, StereoConfig
from stereo_vision.detection import Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    source: str


def intrinsics_from_matrix(matrix: np.ndarray | None, source: str) -> Intrinsics | None:
    if matrix is None or matrix.shape[0] < 3 or matrix.shape[1] < 3:
        return None
    return Intrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        source=source,
    )


def intrinsics_from_fov(frame: np.ndarray, horizontal_fov_deg: float) -> Intrinsics:
    """Approximate pinhole intrinsics from the horizontal field of view."""
    height, width = frame.shape[:2]
    fx = (width * 0.5) / np.tan(np.deg2rad(horizontal_fov_deg * 0.5))
    return Intrinsics(fx=float(fx), fy=float(fx), cx=width * 0.5, cy=height * 0.5, source="fov")


def raw_frame_intrinsics(
    side: str,
    frame: np.ndarray,
    horizontal_fov_deg: float,
    calibration_path=CALIBRATION_FILE,
) -> Intrinsics:
    """Intrinsics suited to unrectified frames.

    Prefers ``cameraMatrix{L,R}``; falls back to the projection or new camera
    matrix, then to an approximation from the field of view.
    """
    side = side.lower()
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")

    suffix = "R" if side == "right" else "L"
    data = load_calibration(calibration_path)
    for name in (f"cameraMatrix{suffix}", f"projMatrix{suffix}", f"newCameraMatrix{suffix}"):
        intrinsics = intrinsics_from_matrix(data.get(name), name)
        if intrinsics is not None:
            return intrinsics

    _warn_once("Camera intrinsics missing from the calibration file; using the field-of-view approximation.")
    return intrinsics_from_fov(frame, horizontal_fov_deg)


def find_match(
    target: Detection,
    candidates: Sequence[Detection],
    config: MatchingConfig,
) -> Detection | None:
    """Return the detection in the other image that best matches ``target``.

    Detections of the same class at a similar image position are preferred.
    """
    best_match = None
    best_score = float("inf")
    target_x, target_y = target.center

    for candidate in candidates:
        cand_x, cand_y = candidate.center
        y_diff = abs(target_y - cand_y)
        if y_diff > config.max_y_diff_px:
            continue
        # Close objects have a large disparity, so X only contributes up to the cap.
        x_diff = min(abs(target_x - cand_x), config.max_x_diff_px)

        class_penalty = 0.0 if target.match_class_name == candidate.match_class_name else config.class_mismatch_penalty
        score = class_penalty + y_diff + config.x_diff_weight * x_diff
        if score < best_score:
            best_score = score
            best_match = candidate

    return best_match


def find_depth(
    right_point: tuple[float, float],
    left_point: tuple[float, float],
    frame_right: np.ndarray,
    frame_left: np.ndarray,
    fx_pixels: float,
    stereo: StereoConfig,
) -> float | None:
    """Depth Z in cm from the horizontal disparity: ``Z = baseline * fx / disparity``.

    Returns ``None`` when the depth is undefined or outside the configured range.
    """
    if frame_right.shape[1] != frame_left.shape[1]:
        logger.error("Left and right frames do not have the same pixel width.")
        return None

    disparity = abs(float(right_point[0]) - float(left_point[0]))
    if disparity <= 0:
        return None

    depth = (stereo.baseline_cm * fx_pixels / disparity) * stereo.distance_scale
    if not np.isfinite(depth) or not stereo.min_depth_cm <= depth <= stereo.max_depth_cm:
        return None
    return float(depth)


def pixel_to_xy(point: tuple[float, float], depth: float, intrinsics: Intrinsics) -> tuple[float, float]:
    """Back-project a pixel at a known depth to camera-frame X and Y."""
    u, v = point
    x = (u - intrinsics.cx) * depth / intrinsics.fx
    y = (v - intrinsics.cy) * depth / intrinsics.fy
    return x, y
