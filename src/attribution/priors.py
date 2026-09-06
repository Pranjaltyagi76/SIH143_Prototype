"""The behavioural prior: few factors, each interpretable, none hidden.

    P(v | obs)  proportional to  L(v) * pi(v)

``L(v)`` is a likelihood computed by forward-simulating the vessel's own track.
``pi(v)`` is this: a small set of multiplicative factors expressing how the
vessel was *behaving*, independent of where the oil went.

Every factor is displayed individually in the interface. None is folded into a
weighted sum, because a weight nobody can justify is exactly what makes a
ranking indefensible -- and there is no labelled spill-to-vessel dataset from
which weights could be learned.

Each factor returns a value and a plain-language note. The note is rendered in
the evidence panel and is validated against the banned-language list, because
the system produces investigative leads and not determinations of
responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attribution.ais import VesselTrack
from src.contracts import EvidenceFactor

# Neutral factor. A signal that says nothing must not move the ranking.
NEUTRAL = 1.0

# Ceilings. No single behavioural signal may dominate the likelihood, because
# none of these is strong enough evidence on its own to carry an accusation.
MAX_GAP_FACTOR = 3.0
MAX_SPEED_FACTOR = 2.0
MAX_COURSE_FACTOR = 1.5
MAX_ALIGNMENT_FACTOR = 3.0

# Type priors. Coarse, but a discharge of mineral oil is not equally likely
# across vessel classes.
TYPE_PRIOR = {
    "Tanker": 1.4,
    "Bulk Carrier": 1.2,
    "Cargo": 1.0,
    "Tug": 1.0,
    "Fishing": 0.9,
    "Passenger": 0.7,
}


@dataclass(frozen=True)
class PriorConfig:
    baseline_gap_hours: float = 0.5
    slick_axis_deg: float = 0.0
    alignment_tolerance_deg: float = 25.0


def gap_factor(
    track: VesselTrack, t_from: float, t_to: float, baseline_hours: float
) -> EvidenceFactor:
    """AIS silence during the candidate release window, relative to local baseline.

    **This factor is the most dangerous one in the system.** Gaps are
    overwhelmingly common for benign reasons -- shore receiver coverage, not
    concealment -- so it is measured against the gap rate that vessels in this
    area normally show, never against an absolute threshold. In a poorly covered
    area an absolute threshold would flag every vessel present and manufacture
    suspicion out of a property of the receiver network (W-07).
    """
    observed = track.gap_hours_overlapping(t_from, t_to)
    if observed <= 0.0:
        return EvidenceFactor(value=NEUTRAL, note="no AIS gap in the candidate release window")

    reference = max(baseline_hours, 0.25)
    ratio = observed / reference
    value = float(np.clip(1.0 + 0.6 * np.log1p(max(ratio - 1.0, 0.0)), NEUTRAL, MAX_GAP_FACTOR))
    if value <= NEUTRAL + 1e-9:
        return EvidenceFactor(
            value=NEUTRAL,
            note=(
                f"{observed:.1f} h AIS gap, at or below the {reference:.1f} h local "
                f"baseline for this area"
            ),
        )
    return EvidenceFactor(
        value=value,
        note=(
            f"{observed:.1f} h AIS gap against a {reference:.1f} h local baseline "
            f"gap rate ({ratio:.1f}x)"
        ),
    )


def speed_factor(track: VesselTrack, t_from: float, t_to: float) -> EvidenceFactor:
    """Slowing relative to the vessel's own transit speed.

    Compared against the vessel's own median rather than a fleet average, so a
    fishing boat is not flagged simply for being slower than a container ship.
    """
    window = track.speed_near(t_from, t_to)
    transit = track.transit_speed_kn
    if window is None or transit <= 0.5:
        return EvidenceFactor(value=NEUTRAL, note="insufficient speed data in the window")

    ratio = window / transit
    if ratio >= 0.75:
        return EvidenceFactor(
            value=NEUTRAL,
            note=f"{window:.1f} kn against a {transit:.1f} kn transit median; no anomaly",
        )
    value = float(np.clip(1.0 + 1.4 * (0.75 - ratio), NEUTRAL, MAX_SPEED_FACTOR))
    return EvidenceFactor(
        value=value,
        note=f"{window:.1f} kn against a {transit:.1f} kn transit median for this vessel",
    )


def course_factor(track: VesselTrack, t_from: float, t_to: float) -> EvidenceFactor:
    """Course alteration near the candidate release window. Weak on its own."""
    change = track.course_change_near(t_from, t_to)
    if change < 20.0:
        return EvidenceFactor(value=NEUTRAL, note="no significant course alteration")
    value = float(np.clip(1.0 + change / 240.0, NEUTRAL, MAX_COURSE_FACTOR))
    return EvidenceFactor(value=value, note=f"{change:.0f} deg course alteration in the window")


def type_factor(track: VesselTrack) -> EvidenceFactor:
    value = TYPE_PRIOR.get(track.ship_type, 1.0)
    return EvidenceFactor(
        value=float(value),
        note=f"{track.ship_type.lower()} prior" if value != 1.0 else f"{track.ship_type.lower()}, neutral prior",
    )


def alignment_factor(
    track: VesselTrack, t_mid: float, slick_axis_deg: float, tolerance_deg: float = 25.0
) -> EvidenceFactor:
    """Angle between the slick's long axis and the vessel's heading.

    Physically motivated and comparatively strong: a continuous discharge from a
    vessel under way leaves a slick *along* the track it sailed. This is the one
    behavioural factor that comes from the shape of the oil rather than from the
    vessel's conduct.
    """
    heading = track.heading_near(t_mid)
    # The slick axis is undirected, so fold the difference into [0, 90].
    offset = abs((heading - slick_axis_deg + 90.0) % 180.0 - 90.0)
    if offset > 3.0 * tolerance_deg:
        return EvidenceFactor(
            value=NEUTRAL,
            note=f"slick major axis {offset:.0f} deg from vessel heading; not aligned",
        )
    value = float(
        np.clip(1.0 + (MAX_ALIGNMENT_FACTOR - 1.0) * np.exp(-((offset / tolerance_deg) ** 2)),
                NEUTRAL, MAX_ALIGNMENT_FACTOR)
    )
    return EvidenceFactor(
        value=value,
        note=f"slick major axis within {offset:.0f} deg of vessel heading",
    )


def behavioural_prior(
    track: VesselTrack, t_from: float, t_to: float, config: PriorConfig
) -> tuple[float, dict[str, EvidenceFactor]]:
    """All factors, plus their product. Returns ``(pi, factors)``."""
    factors = {
        "ais_gap_factor": gap_factor(track, t_from, t_to, config.baseline_gap_hours),
        "speed_anomaly_factor": speed_factor(track, t_from, t_to),
        "course_change_factor": course_factor(track, t_from, t_to),
        "vessel_type_factor": type_factor(track),
        "axis_alignment_factor": alignment_factor(
            track, 0.5 * (t_from + t_to), config.slick_axis_deg, config.alignment_tolerance_deg
        ),
    }
    prior = float(np.prod([f.value for f in factors.values()]))
    return prior, factors


__all__ = [
    "PriorConfig",
    "behavioural_prior",
    "gap_factor",
    "speed_factor",
    "course_factor",
    "type_factor",
    "alignment_factor",
    "TYPE_PRIOR",
    "NEUTRAL",
]
