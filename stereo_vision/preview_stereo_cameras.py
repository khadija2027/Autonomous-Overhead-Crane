"""Show both cameras side by side to check indexes and alignment. Press ``q`` to quit."""

from __future__ import annotations

import cv2

from stereo_vision.camera import StereoCamera
from stereo_vision.config import CameraConfig

QUIT_KEY = ord("q")


def preview(camera_config: CameraConfig | None = None) -> None:
    config = camera_config or CameraConfig()
    size = (config.width, config.height)
    try:
        with StereoCamera(config, set_resolution=False) as camera:
            while True:
                frame_left, frame_right = camera.read()
                # Resize so both frames fit side by side.
                combined = cv2.hconcat([cv2.resize(frame_left, size), cv2.resize(frame_right, size)])
                cv2.imshow("stereo preview (left | right)", combined)
                if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                    break
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    from stereo_vision.cli import main as cli_main

    cli_main(["preview"])


if __name__ == "__main__":
    main()
