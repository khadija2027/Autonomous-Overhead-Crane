"""Compute stereo calibration and rectification maps from chessboard images.

Reads paired images from ``images/StereoLeft`` and ``images/StereoRight`` and
writes ``stereoMap.xml`` next to this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2 as cv
import numpy as np

from stereo_vision.calibration import CALIBRATION_FILE, load_calibration, write_matrices
from stereo_vision.image_io import read_image

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PACKAGE_DIR / "images"

CHESSBOARD_SIZE = (7, 7)  # Inner corners per row and column.
FRAME_SIZE = (640, 480)
SUBPIX_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def _chessboard_object_points(chessboard_size: tuple[int, int]) -> np.ndarray:
    """Corner coordinates on the board plane: (0,0,0), (1,0,0), ..."""
    cols, rows = chessboard_size
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return points


def find_corners(
    images_dir: Path = IMAGES_DIR,
    chessboard_size: tuple[int, int] = CHESSBOARD_SIZE,
    show: bool = True,
):
    """Detect chessboard corners in every left/right image pair."""
    left_paths = sorted((images_dir / "StereoLeft").glob("*.png"))
    right_paths = sorted((images_dir / "StereoRight").glob("*.png"))
    if not left_paths or len(left_paths) != len(right_paths):
        raise FileNotFoundError(
            f"Capture matching left/right chessboard images in {images_dir / 'StereoLeft'} "
            f"and {images_dir / 'StereoRight'} first."
        )

    object_template = _chessboard_object_points(chessboard_size)
    object_points, points_left, points_right = [], [], []
    gray_left = gray_right = None

    for left_path, right_path in zip(left_paths, right_paths):
        image_left = read_image(left_path)
        image_right = read_image(right_path)
        gray_left = cv.cvtColor(image_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(image_right, cv.COLOR_BGR2GRAY)

        found_left, corners_left = cv.findChessboardCorners(gray_left, chessboard_size, None)
        found_right, corners_right = cv.findChessboardCorners(gray_right, chessboard_size, None)
        if not (found_left and found_right):
            logger.info("Chessboard not found in both images: %s / %s", left_path.name, right_path.name)
            continue

        object_points.append(object_template)
        corners_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        corners_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        points_left.append(corners_left)
        points_right.append(corners_right)

        if show:
            cv.drawChessboardCorners(image_left, chessboard_size, corners_left, found_left)
            cv.drawChessboardCorners(image_right, chessboard_size, corners_right, found_right)
            cv.imshow("left corners", image_left)
            cv.imshow("right corners", image_right)
            cv.waitKey(1000)

    if show:
        cv.destroyAllWindows()

    if not object_points:
        raise RuntimeError(f"No {chessboard_size[0]}x{chessboard_size[1]} chessboard corners were found.")

    logger.info("Using %d of %d image pairs.", len(object_points), len(left_paths))
    return object_points, points_left, points_right, gray_left.shape[::-1]


def calibrate(
    images_dir: Path = IMAGES_DIR,
    output_path: Path = CALIBRATION_FILE,
    chessboard_size: tuple[int, int] = CHESSBOARD_SIZE,
    frame_size: tuple[int, int] = FRAME_SIZE,
    show: bool = True,
) -> Path:
    """Run single-camera and stereo calibration and save ``stereoMap.xml``."""
    object_points, points_left, points_right, image_size = find_corners(images_dir, chessboard_size, show)

    # Calibrate each camera on its own.
    _, camera_matrix_left, dist_left, _, _ = cv.calibrateCamera(object_points, points_left, frame_size, None, None)
    new_camera_matrix_left, _ = cv.getOptimalNewCameraMatrix(camera_matrix_left, dist_left, image_size, 1, image_size)

    _, camera_matrix_right, dist_right, _, _ = cv.calibrateCamera(object_points, points_right, frame_size, None, None)
    new_camera_matrix_right, _ = cv.getOptimalNewCameraMatrix(camera_matrix_right, dist_right, image_size, 1, image_size)

    # Stereo calibration with fixed intrinsics: only R, T, E and F are estimated.
    # OpenCV's default termination criteria and rectification alpha are used;
    # these are the values the existing stereoMap.xml was produced with.
    (
        stereo_error,
        new_camera_matrix_left,
        dist_left,
        new_camera_matrix_right,
        dist_right,
        rotation,
        translation,
        _,
        _,
    ) = cv.stereoCalibrate(
        object_points,
        points_left,
        points_right,
        new_camera_matrix_left,
        dist_left,
        new_camera_matrix_right,
        dist_right,
        image_size,
        flags=cv.CALIB_FIX_INTRINSIC,
    )
    logger.info("Stereo reprojection error: %.4f px", stereo_error)

    # Rectification.
    rect_left, rect_right, proj_left, proj_right, q_matrix, _, _ = cv.stereoRectify(
        new_camera_matrix_left, dist_left, new_camera_matrix_right, dist_right,
        image_size, rotation, translation,
    )
    map_left = cv.initUndistortRectifyMap(
        new_camera_matrix_left, dist_left, rect_left, proj_left, image_size, cv.CV_16SC2
    )
    map_right = cv.initUndistortRectifyMap(
        new_camera_matrix_right, dist_right, rect_right, proj_right, image_size, cv.CV_16SC2
    )

    write_matrices(
        output_path,
        {
            "stereoMapL_x": map_left[0],
            "stereoMapL_y": map_left[1],
            "stereoMapR_x": map_right[0],
            "stereoMapR_y": map_right[1],
            "cameraMatrixL": camera_matrix_left,
            "cameraMatrixR": camera_matrix_right,
            "newCameraMatrixL": new_camera_matrix_left,
            "newCameraMatrixR": new_camera_matrix_right,
            "projMatrixL": proj_left,
            "projMatrixR": proj_right,
            "Q": q_matrix,
        },
    )
    load_calibration.cache_clear()

    logger.info("Saved calibration to %s", output_path)
    return output_path


def main() -> None:
    from stereo_vision.cli import main as cli_main

    cli_main(["calibrate"])


if __name__ == "__main__":
    main()
