"""Capture synchronized left/right chessboard images for calibration.

Press ``s`` to save a pair and ``Esc`` to quit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from stereo_vision.camera import StereoCamera
from stereo_vision.config import CameraConfig
from stereo_vision.image_io import write_image

logger = logging.getLogger(__name__)

IMAGES_DIR = Path(__file__).resolve().parent / "images"
SAVE_KEY = ord("s")
ESCAPE_KEY = 27


def capture(camera_config: CameraConfig | None = None, images_dir: Path = IMAGES_DIR) -> int:
    """Show both cameras and save image pairs on demand. Returns the number saved."""
    left_dir = images_dir / "StereoLeft"
    right_dir = images_dir / "StereoRight"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    try:
        # Keep the cameras' default resolution, as the calibration expects.
        with StereoCamera(camera_config or CameraConfig(), set_resolution=False) as camera:
            while True:
                frame_left, frame_right = camera.read()
                cv2.imshow("left camera", frame_left)
                cv2.imshow("right camera", frame_right)

                key = cv2.waitKey(5)
                if key == ESCAPE_KEY:
                    break
                if key == SAVE_KEY:
                    write_image(left_dir / f"imageL{saved}.png", frame_left)
                    write_image(right_dir / f"imageR{saved}.png", frame_right)
                    logger.info("Saved image pair %d.", saved)
                    saved += 1
    finally:
        cv2.destroyAllWindows()
    return saved


def main() -> None:
    from stereo_vision.cli import main as cli_main

    cli_main(["capture"])


if __name__ == "__main__":
    main()
