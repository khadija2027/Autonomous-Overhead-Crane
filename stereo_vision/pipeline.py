"""Live stereo detection: detect objects in both cameras and measure their position."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from stereo_vision import visualization as viz
from stereo_vision.camera import StereoCamera
from stereo_vision.config import CameraConfig, CraneFrame, MatchingConfig, Profile, StereoConfig
from stereo_vision.detection import Detection, MultiModelDetector
from stereo_vision.stereo import (
    find_depth,
    find_match,
    intrinsics_from_fov,
    pixel_to_xy,
    raw_frame_intrinsics,
)

logger = logging.getLogger(__name__)

QUIT_KEY = ord("q")


@dataclass(frozen=True)
class Measurement:
    """Position of one detected object, in cm."""

    detection: Detection
    z: float
    x: float | None = None
    y: float | None = None
    crane_x: float | None = None
    crane_y: float | None = None

    def lines(self) -> list[str]:
        name = self.detection.class_name
        if self.x is None or self.y is None:
            return [f"{name}: {self.z:.1f} cm"]

        lines = [f"{name}: Z={self.z:.1f} cm", f"X={self.x:.1f} cm, Y={self.y:.1f} cm"]
        if self.crane_x is not None and self.crane_y is not None:
            lines.append(f"X_crane={self.crane_x:.1f} cm, Y_crane={self.crane_y:.1f} cm")
        return lines

    def __str__(self) -> str:
        return " | ".join(self.lines())


class StereoMeasurer:
    """Turns per-camera detections into 3-D measurements."""

    def __init__(
        self,
        profile: Profile,
        stereo: StereoConfig | None = None,
        matching: MatchingConfig | None = None,
        crane_frame: CraneFrame | None = None,
    ):
        self.profile = profile
        self.stereo = stereo or StereoConfig()
        self.matching = matching or MatchingConfig()
        self.crane_frame = crane_frame or CraneFrame()

    def measure(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        detections_left: list[Detection],
        detections_right: list[Detection],
    ) -> list[Measurement]:
        if self.profile.use_calibrated_intrinsics:
            intrinsics = raw_frame_intrinsics("right", frame_right, self.stereo.horizontal_fov_deg)
        else:
            intrinsics = intrinsics_from_fov(frame_right, self.stereo.horizontal_fov_deg)

        measurements = []
        for det_right in detections_right:
            det_left = find_match(det_right, detections_left, self.matching)
            if det_left is None:
                continue

            depth = find_depth(
                det_right.center, det_left.center, frame_right, frame_left, intrinsics.fx, self.stereo
            )
            if depth is None:
                continue

            if not self.profile.show_xy:
                measurements.append(Measurement(det_right, z=depth))
                continue

            x, y = pixel_to_xy(det_right.center, depth, intrinsics)
            crane_x = crane_y = None
            if self.profile.show_crane_frame:
                crane_x, crane_y = self.crane_frame.from_camera(x, y)
            measurements.append(Measurement(det_right, z=depth, x=x, y=y, crane_x=crane_x, crane_y=crane_y))

        return measurements


def _annotate(frame: np.ndarray, detections, measurements, tracking_lost: bool, fps: float) -> None:
    viz.draw_detections(frame, detections)
    if tracking_lost:
        viz.draw_tracking_lost(frame)
    else:
        top = 50
        for measurement in measurements:
            top = viz.draw_measurement(frame, measurement.lines(), top)
    viz.draw_fps(frame, fps)


def run(profile: Profile, camera_config: CameraConfig | None = None, rectify: bool = False) -> None:
    """Run live detection until the user presses ``q``.

    ``rectify`` applies the calibration maps before detection. It is off by
    default to match the behaviour the matching tolerances were tuned for.
    """
    camera_config = camera_config or CameraConfig()
    logger.info("Loading models for profile '%s': %s", profile.name, profile.description)
    detector = MultiModelDetector(profile)
    measurer = StereoMeasurer(profile)

    if rectify:
        from stereo_vision.calibration import undistort_rectify

    try:
        with StereoCamera(camera_config) as camera:
            while True:
                frame_left, frame_right = camera.read()
                start = time.time()

                if rectify:
                    frame_right, frame_left = undistort_rectify(frame_right, frame_left)

                detections_right = detector.detect(frame_right)
                detections_left = detector.detect(frame_left)

                tracking_lost = not detections_right or not detections_left
                measurements = (
                    []
                    if tracking_lost
                    else measurer.measure(frame_left, frame_right, detections_left, detections_right)
                )
                for measurement in measurements:
                    logger.info("%s", measurement)

                elapsed = time.time() - start
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                for frame, detections in ((frame_right, detections_right), (frame_left, detections_left)):
                    _annotate(frame, detections, measurements, tracking_lost, fps)

                cv2.imshow("frame right", frame_right)
                cv2.imshow("frame left", frame_left)
                if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                    break
    finally:
        cv2.destroyAllWindows()
