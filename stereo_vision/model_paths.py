"""Resolve model checkpoint paths from environment variables or defaults."""

from __future__ import annotations

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_MODEL_PATHS = {
    "truck_cargo": (
        "TRUCK_CARGO_MODEL_PATH",
        REPOSITORY_ROOT / "Detection" / "models" / "truck-cargo" / "best.pt",
    ),
    "cone": (
        "CONE_MODEL_PATH",
        REPOSITORY_ROOT / "Detection" / "models" / "cone" / "best.pt",
    ),
    "person": (
        "PERSON_MODEL_PATH",
        REPOSITORY_ROOT / "stereo_vision" / "models" / "person.pt",
    ),
    "box": (
        "BOX_MODEL_PATH",
        REPOSITORY_ROOT / "stereo_vision" / "models" / "box.pt",
    ),
    "coco": (
        "COCO_MODEL_PATH",
        REPOSITORY_ROOT / "stereo_vision" / "yolov8n.pt",
    ),
}

# Official Ultralytics weights that may be downloaded automatically when absent.
_DOWNLOADABLE_FALLBACKS = {"coco": "yolov8n.pt"}


def get_model_path(name: str) -> str:
    """Return the checkpoint path for a registered model name.

    The matching ``*_MODEL_PATH`` environment variable overrides the default.
    Relative paths are resolved from the repository root.
    """
    try:
        environment_variable, default_path = _MODEL_PATHS[name]
    except KeyError:
        raise KeyError(f"Unknown model name: {name}") from None

    configured_path = Path(os.environ.get(environment_variable, default_path)).expanduser()
    if not configured_path.is_absolute():
        configured_path = REPOSITORY_ROOT / configured_path

    if configured_path.is_file():
        return str(configured_path)

    if environment_variable not in os.environ and name in _DOWNLOADABLE_FALLBACKS:
        # Let Ultralytics download the official checkpoint.
        return _DOWNLOADABLE_FALLBACKS[name]

    raise FileNotFoundError(
        f"Model file not found: {configured_path}. "
        f"Set {environment_variable} to the model file path."
    )
