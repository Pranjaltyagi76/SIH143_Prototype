"""Vessel attribution.

Each candidate vessel's AIS track is a generative hypothesis, forward-simulated
and scored by the same observation operator the inversion uses, normalised
against an explicit dark-vessel hypothesis.
See Context/technical_design.md section 5.
"""

from .ais import (
    GAP_THRESHOLD_HOURS,
    MAX_PLAUSIBLE_SPEED_KN,
    TRACK_SPLIT_GAP_HOURS,
    CleaningReport,
    VesselTrack,
    baseline_gap_hours,
    clean_and_reconstruct,
    load_ais,
)
from .engine import AttributionConfig, AttributionDiagnostics, attribute, prefilter
from .priors import PriorConfig, TYPE_PRIOR, behavioural_prior

__all__ = [
    "load_ais", "clean_and_reconstruct", "VesselTrack", "CleaningReport",
    "baseline_gap_hours", "MAX_PLAUSIBLE_SPEED_KN", "TRACK_SPLIT_GAP_HOURS",
    "GAP_THRESHOLD_HOURS",
    "behavioural_prior", "PriorConfig", "TYPE_PRIOR",
    "attribute", "prefilter", "AttributionConfig", "AttributionDiagnostics",
]
