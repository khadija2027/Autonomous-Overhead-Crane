"""Configuration for the stereo perception pipeline.

All hardware-specific values and detection thresholds live here so that the
pipeline code stays free of magic numbers. Three detection profiles are
provided; each one reproduces one of the original experiment scripts:

* ``depth`` - depth (Z) only, focal length estimated from the field of view.
* ``xyz``   - X/Y/Z using the calibrated camera matrix.
* ``crane`` - X/Y/Z plus coordinates expressed in the crane frame, with
  aspect-ratio filters to reject implausible truck and cone boxes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _frozen_map(values: Mapping | None = None) -> Mapping:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True)
class CameraConfig:
    """Stereo camera pair settings."""

    left_index: int = 0
    right_index: int = 2
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class StereoConfig:
    """Geometry of the stereo rig and depth limits."""

    baseline_cm: float = 7.0
    horizontal_fov_deg: float = 84.0
    # Empirical correction: a real distance of 90 cm was measured as 50 cm.
    known_distance_cm: float = 90.0
    measured_distance_cm: float = 50.0
    min_depth_cm: float = 10.0
    max_depth_cm: float = 500.0

    @property
    def distance_scale(self) -> float:
        return self.known_distance_cm / self.measured_distance_cm


@dataclass(frozen=True)
class MatchingConfig:
    """Rules for pairing a right-camera detection with a left-camera one."""

    # Tolerances are generous because the frames are not rectified.
    max_x_diff_px: float = 80.0
    max_y_diff_px: float = 60.0
    class_mismatch_penalty: float = 100.0
    x_diff_weight: float = 0.25


@dataclass(frozen=True)
class CraneFrame:
    """Transform from camera X/Y to the crane coordinate frame (cm).

    The camera X axis maps to the crane Y axis and vice versa; the offsets are
    the position of the camera in the crane frame.
    """

    offset_x_cm: float = 47.5
    offset_y_cm: float = 17.5

    def from_camera(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        return self.offset_x_cm + y_cm, self.offset_y_cm + x_cm


@dataclass(frozen=True)
class ModelConfig:
    """One YOLO model and its confidence thresholds.

    ``weights`` is a key understood by :func:`stereo_vision.model_paths.get_model_path`.
    """

    name: str
    weights: str
    conf: float
    class_conf: Mapping[str, float] = field(default_factory=_frozen_map)
    exclude_classes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_conf", _frozen_map(self.class_conf))


@dataclass(frozen=True)
class Profile:
    """A complete detection-and-measurement configuration."""

    name: str
    description: str
    models: tuple[ModelConfig, ...]
    # ``None`` keeps every class the models produce.
    allowed_classes: frozenset[str] | None = None
    display_aliases: Mapping[str, str] = field(default_factory=_frozen_map)
    match_aliases: Mapping[str, str] = field(default_factory=_frozen_map)
    # Pairs of (match) class names allowed to overlap, e.g. cargo on a truck.
    coexisting_pairs: frozenset[frozenset[str]] = frozenset()
    min_aspect_ratio: Mapping[str, float] = field(default_factory=_frozen_map)
    max_aspect_ratio: Mapping[str, float] = field(default_factory=_frozen_map)
    # False: estimate the focal length from the horizontal field of view.
    use_calibrated_intrinsics: bool = True
    show_xy: bool = True
    show_crane_frame: bool = False

    def __post_init__(self) -> None:
        for name in ("display_aliases", "match_aliases", "min_aspect_ratio", "max_aspect_ratio"):
            object.__setattr__(self, name, _frozen_map(getattr(self, name)))


_TRUCK_CARGO_CLASSES = frozenset({"truck", "mytruck", "fardeau", "fardeaux", "plot", "cone"})


def _models(*, cargo_conf: Mapping[str, float], cone_conf: float, coco_exclude: set[str]):
    return (
        ModelConfig("person_custom", "person", conf=0.50),
        ModelConfig("fardeaux_truck_custom", "truck_cargo", conf=0.30, class_conf=cargo_conf),
        ModelConfig("cone_custom", "cone", conf=cone_conf),
        ModelConfig("box_custom", "box", conf=0.40),
        ModelConfig("coco_obstacles", "coco", conf=0.50, exclude_classes=frozenset(coco_exclude)),
    )


PROFILES: dict[str, Profile] = {
    "depth": Profile(
        name="depth",
        description="Depth (Z) only; focal length estimated from the field of view.",
        models=_models(
            cargo_conf={"fardeaux": 0.05},
            cone_conf=0.25,
            coco_exclude={"frisbee", "truck"},
        ),
        coexisting_pairs=frozenset({frozenset({"mytruck", "fardeaux"})}),
        use_calibrated_intrinsics=False,
        show_xy=False,
    ),
    "xyz": Profile(
        name="xyz",
        description="X/Y/Z in the right-camera frame using calibrated intrinsics.",
        models=_models(
            cargo_conf={"fardeau": 0.04, "fardeaux": 0.04},
            cone_conf=0.25,
            coco_exclude={"frisbee"},
        ),
        allowed_classes=_TRUCK_CARGO_CLASSES,
        display_aliases={"mytruck": "truck", "cone": "plot"},
        match_aliases={"mytruck": "truck", "cone": "plot", "fardeau": "fardeaux"},
        coexisting_pairs=frozenset({frozenset({"truck", "fardeaux"})}),
    ),
    "crane": Profile(
        name="crane",
        description="X/Y/Z plus crane-frame coordinates, with box-shape filters.",
        models=_models(
            cargo_conf={"fardeau": 0.04, "fardeaux": 0.04},
            cone_conf=0.10,
            coco_exclude={"frisbee"},
        ),
        allowed_classes=_TRUCK_CARGO_CLASSES,
        display_aliases={"mytruck": "truck", "plot": "cone"},
        match_aliases={"mytruck": "truck", "plot": "cone", "fardeau": "fardeaux"},
        coexisting_pairs=frozenset({frozenset({"truck", "fardeaux"})}),
        min_aspect_ratio={"truck": 1.15},
        max_aspect_ratio={"cone": 1.45},
        show_crane_frame=True,
    ),
}

DEFAULT_PROFILE = "xyz"
