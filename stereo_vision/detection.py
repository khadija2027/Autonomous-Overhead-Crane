"""Multi-model YOLO detection with class aliasing and overlap suppression."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from stereo_vision.config import Profile
from stereo_vision.model_paths import get_model_path

Box = tuple[int, int, int, int]

IOU_THRESHOLD = 0.25
OVERLAP_THRESHOLD = 0.60


@dataclass(frozen=True)
class Detection:
    """A single bounding box produced by one of the models."""

    box: Box
    conf: float
    class_name: str  # Name shown to the user.
    raw_class_name: str  # Name reported by the model.
    match_class_name: str  # Name used for stereo matching and overlap rules.
    model_name: str

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


def normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower()


def box_iou(box1: Box, box2: Box) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if inter_w < 0 or inter_h < 0:
        return 0.0

    inter_area = inter_w * inter_h
    union_area = _area(box1) + _area(box2) - inter_area
    return inter_area / union_area if union_area else 0.0


def overlap_ratio(box1: Box, box2: Box) -> float:
    """Fraction of the smaller box covered by the intersection."""
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    min_area = min(_area(box1), _area(box2))
    return (inter_w * inter_h) / min_area if min_area > 0 else 0.0


def aspect_ratio(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(x2 - x1, 1) / max(y2 - y1, 1)


def _area(box: Box) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def suppress_overlaps(
    detections: Iterable[Detection],
    can_coexist: Callable[[Detection, Detection], bool],
    iou_threshold: float = IOU_THRESHOLD,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> list[Detection]:
    """Keep the highest-confidence detection among overlapping boxes.

    Boxes from different models often describe the same object; only one class
    is kept per object unless ``can_coexist`` allows both (e.g. cargo on a truck).
    """
    ordered = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept: list[Detection] = []
    suppressed: set[int] = set()

    for i, det in enumerate(ordered):
        if i in suppressed:
            continue
        for j in range(i + 1, len(ordered)):
            if j in suppressed:
                continue
            other = ordered[j]
            overlapping = (
                box_iou(det.box, other.box) > iou_threshold
                or overlap_ratio(det.box, other.box) > overlap_threshold
            )
            if overlapping and not can_coexist(det, other):
                suppressed.add(j)
        kept.append(det)

    return kept


class MultiModelDetector:
    """Runs every model of a profile on a frame and merges the results."""

    def __init__(self, profile: Profile, model_factory: Callable[[str], object] | None = None):
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO

        self.profile = profile
        self._models = [
            (config, model_factory(get_model_path(config.weights)))
            for config in profile.models
        ]
        self._class_conf = [
            {normalize_class_name(k): v for k, v in config.class_conf.items()}
            for config, _ in self._models
        ]
        self._exclude = [
            {normalize_class_name(c) for c in config.exclude_classes}
            for config, _ in self._models
        ]

    def display_name(self, class_key: str) -> str:
        return self.profile.display_aliases.get(class_key, class_key)

    def match_name(self, class_key: str) -> str:
        return self.profile.match_aliases.get(class_key, self.display_name(class_key))

    def can_coexist(self, det1: Detection, det2: Detection) -> bool:
        pair = frozenset((det1.match_class_name, det2.match_class_name))
        return pair in self.profile.coexisting_pairs

    def _shape_is_valid(self, match_class: str, box: Box) -> bool:
        ratio = aspect_ratio(box)
        minimum = self.profile.min_aspect_ratio.get(match_class)
        maximum = self.profile.max_aspect_ratio.get(match_class)
        return (minimum is None or ratio >= minimum) and (maximum is None or ratio <= maximum)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        allowed = self.profile.allowed_classes

        for (config, model), class_conf, exclude in zip(self._models, self._class_conf, self._exclude):
            # Run the model at the lowest threshold, then apply per-class ones.
            model_conf = min([config.conf, *class_conf.values()])
            result = model(frame, conf=model_conf, verbose=False)[0]
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                raw_name = model.names[cls_id] if cls_id in model.names else "object"
                key = normalize_class_name(raw_name)

                if allowed is not None and key not in allowed:
                    continue
                if key in exclude:
                    continue
                if conf < class_conf.get(key, config.conf):
                    continue

                match_name = self.match_name(key)
                if not self._shape_is_valid(match_name, (x1, y1, x2, y2)):
                    continue

                detections.append(
                    Detection(
                        box=(x1, y1, x2, y2),
                        conf=conf,
                        class_name=self.display_name(key),
                        raw_class_name=raw_name,
                        match_class_name=match_name,
                        model_name=config.name,
                    )
                )

        return suppress_overlaps(detections, self.can_coexist)
