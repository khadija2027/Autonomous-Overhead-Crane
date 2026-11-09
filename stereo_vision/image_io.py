"""Image file I/O that works with non-ASCII paths.

``cv2.imread`` and ``cv2.imwrite`` cannot open paths containing characters such
as "é" on Windows, so files are read and written with NumPy and only the
encoding/decoding is done by OpenCV.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path | str) -> np.ndarray:
    """Read a BGR image, raising ``FileNotFoundError`` if it cannot be decoded."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def write_image(path: Path | str, image: np.ndarray) -> None:
    """Write an image; the format is chosen from the file extension."""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode image as {path.suffix}: {path}")
    encoded.tofile(str(path))
