"""Detection: segmentation, the physics gate, and characterisation (stages 1-3)."""

from .gate import (
    FEATURE_NAMES,
    LookAlikeClassifier,
    PatchFeatures,
    extract_features,
    lane_density,
    outside_detectability_window,
)
from .injection import InjectedSlick, inject, read_scene, render_damping
from .pipeline import DEFAULT_MODEL_PATH, DetectionDiagnostics, detect
from .segment import (
    DarkPatch,
    cell_area_km2,
    segment_dark_patches,
    segmentation_iou,
    segmentation_recall,
)

__all__ = [
    "inject", "render_damping", "read_scene", "InjectedSlick",
    "segment_dark_patches", "DarkPatch", "segmentation_iou", "segmentation_recall",
    "cell_area_km2",
    "extract_features", "PatchFeatures", "FEATURE_NAMES", "lane_density",
    "outside_detectability_window", "LookAlikeClassifier",
    "detect", "DetectionDiagnostics", "DEFAULT_MODEL_PATH",
]
