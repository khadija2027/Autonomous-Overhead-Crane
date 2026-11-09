"""Command-line interface: ``python -m stereo_vision <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from stereo_vision.config import DEFAULT_PROFILE, PROFILES, CameraConfig

logger = logging.getLogger("stereo_vision")


def _build_parser() -> argparse.ArgumentParser:
    defaults = CameraConfig()
    parser = argparse.ArgumentParser(
        prog="python -m stereo_vision",
        description="Stereo perception tools for the autonomous overhead crane.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug messages.")

    camera = argparse.ArgumentParser(add_help=False)
    camera.add_argument("--left", type=int, default=defaults.left_index, help="Left camera index (default: %(default)s).")
    camera.add_argument("--right", type=int, default=defaults.right_index, help="Right camera index (default: %(default)s).")

    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("preview", parents=[camera], help="Show both cameras side by side.")
    commands.add_parser("capture", parents=[camera], help="Capture chessboard image pairs for calibration.")

    calibrate = commands.add_parser("calibrate", help="Compute stereoMap.xml from the captured image pairs.")
    calibrate.add_argument("--no-display", action="store_true", help="Do not show detected corners.")

    detect = commands.add_parser("detect", parents=[camera], help="Run live stereo detection.")
    detect.add_argument(
        "--mode",
        choices=sorted(PROFILES),
        default=DEFAULT_PROFILE,
        help="; ".join(f"{name}: {p.description.rstrip('.')}" for name, p in PROFILES.items())
        + " (default: %(default)s).",
    )
    detect.add_argument("--rectify", action="store_true", help="Rectify frames with stereoMap.xml before detection.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Imported lazily so that `--help` does not load OpenCV/Ultralytics models.
    from stereo_vision.camera import CameraError

    camera_config = None
    if hasattr(args, "left"):
        camera_config = CameraConfig(left_index=args.left, right_index=args.right)

    try:
        if args.command == "preview":
            from stereo_vision.preview_stereo_cameras import preview

            preview(camera_config)
        elif args.command == "capture":
            from stereo_vision.capture_calibration_images import capture

            count = capture(camera_config)
            logger.info("Captured %d image pair(s).", count)
        elif args.command == "calibrate":
            from stereo_vision.calibrate_stereo import calibrate

            calibrate(show=not args.no_display)
        elif args.command == "detect":
            from stereo_vision.pipeline import run

            run(PROFILES[args.mode], camera_config, rectify=args.rectify)
    except (CameraError, FileNotFoundError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
