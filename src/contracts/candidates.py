"""Contract 4 of 4: RankedCandidates -- output of the attribution engine (stage 7).

See Context/architecture.md section 3.4 and technical_design.md section 5.

The single most important thing in this file is that
``RankedCandidates.dark_vessel_hypothesis`` is a REQUIRED field with no default
and no Optional wrapper.

If the true polluter had AIS switched off it is not in the candidate set at all,
and a system that normalises only over observed vessels will confidently name an
innocent ship. Carrying the dark hypothesis is what makes every other number in
this contract meaningful. It is control SC-1 in Context/security_review.md.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from .common import StrictModel, require_utc

# Probabilities including the dark hypothesis must sum to 1 within this tolerance.
PROB_SUM_TOLERANCE = 1e-6

# Language that must never appear in any evidence note, reason string or UI copy.
# The system produces investigative leads, not determinations of responsibility.
# See security_review.md control SC-2.
BANNED_TERMS = (
    "guilty",
    "culprit",
    "responsible for",
    "the polluter is",
    "identified the vessel",
    "confirmed spill by",
)


def reject_accusatory_language(text: str) -> str:
    lowered = text.lower()
    for term in BANNED_TERMS:
        if term in lowered:
            raise ValueError(
                f"accusatory language {term!r} is not permitted in system output; "
                f"this system produces investigative leads, not determinations (SC-2)"
            )
    return text


class EvidenceFactor(StrictModel):
    """One interpretable component of the behavioural prior.

    Each factor is displayed individually in the UI. None is hidden inside a
    weighted sum, because a weight nobody can justify is the thing that makes a
    ranking indefensible.
    """

    value: float = Field(gt=0.0, description="Multiplicative factor; 1.0 is neutral")
    note: str = Field(min_length=1, description="Plain-language justification, shown in the UI")

    @field_validator("note")
    @classmethod
    def no_accusation(cls, v: str) -> str:
        return reject_accusatory_language(v)


class VesselEvidence(StrictModel):
    """The full evidence breakdown for one candidate.

    ``track_overlap_likelihood`` is the log-likelihood from vessel-conditioned
    forward simulation: particles seeded along this vessel's actual AIS track,
    propagated, and scored against the observed mask with the same observation
    operator used everywhere else. It is a likelihood, not an invented score.
    """

    track_overlap_likelihood: float = Field(description="log p(observed slick | this vessel)")
    ais_gap_factor: EvidenceFactor
    speed_anomaly_factor: EvidenceFactor
    course_change_factor: EvidenceFactor
    vessel_type_factor: EvidenceFactor
    axis_alignment_factor: EvidenceFactor


class Candidate(StrictModel):
    """One AIS-observed vessel, scored as a generative hypothesis."""

    rank: int = Field(ge=1)
    mmsi: str = Field(min_length=1)
    vessel_name: str = Field(
        min_length=1,
        description=(
            "Redacted as 'REDACTED_IN_DEMO' in slides, video and any published "
            "screenshot. See security_review.md control SC-4."
        ),
    )
    vessel_type: str = Field(min_length=1)

    posterior_probability: float = Field(ge=0.0, le=1.0)
    log_likelihood: float
    best_discharge_window_utc: tuple[datetime, datetime]

    evidence: VesselEvidence
    track_geojson: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def window_ordered(self) -> Candidate:
        start, end = self.best_discharge_window_utc
        require_utc(start)
        require_utc(end)
        if start >= end:
            raise ValueError("discharge window start must be before end")
        return self


class DarkVesselHypothesis(StrictModel):
    """H_dark: the source was a vessel not present in AIS.

    Its likelihood is computed by the identical observation operator used for
    named vessels -- seed the blind proposal region excluding tubes around
    observed tracks, forward-simulate, evaluate. That is what makes it
    comparable on the same scale rather than a fudge factor.
    """

    posterior_probability: float = Field(ge=0.0, le=1.0)
    prior_used: float = Field(
        gt=0.0,
        lt=1.0,
        description=(
            "Prior probability of AIS non-compliance in this region. An assumption, "
            "not a measurement -- report the sensitivity. Debt item D5."
        ),
    )
    note: str = Field(min_length=1)


class TrafficReduction(StrictModel):
    """The headline metric: how much the search space actually narrowed.

    Reducing 214 vessels to 3 is the deliverable. It is the one number a
    non-specialist judge will remember.
    """

    vessels_in_window: int = Field(ge=0)
    after_prefilter: int = Field(ge=0)
    reported: int = Field(ge=0)

    @model_validator(mode="after")
    def monotonic(self) -> TrafficReduction:
        if not (self.reported <= self.after_prefilter <= self.vessels_in_window):
            raise ValueError(
                "must satisfy reported <= after_prefilter <= vessels_in_window "
                f"(got {self.reported}, {self.after_prefilter}, {self.vessels_in_window})"
            )
        return self

    @property
    def reduction_factor(self) -> float:
        return self.vessels_in_window / max(self.reported, 1)


class RankedCandidates(StrictModel):
    """Ranked investigative leads, normalised against the dark-vessel hypothesis."""

    case_id: str = Field(min_length=1)
    traffic_reduction: TrafficReduction
    candidates: list[Candidate]

    dark_vessel_hypothesis: DarkVesselHypothesis = Field(
        description=(
            "REQUIRED. Not Optional, no default. Without this term the system will "
            "confidently name an innocent ship whenever the true polluter was dark. "
            "See security_review.md control SC-1."
        )
    )

    disclaimer: str = Field(
        default=(
            "Investigative leads under stated model assumptions. "
            "Not a determination of responsibility."
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def ranks_are_contiguous(self) -> RankedCandidates:
        ranks = [c.rank for c in self.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"ranks must be 1..n in order, got {ranks}")
        return self

    @model_validator(mode="after")
    def probabilities_normalise(self) -> RankedCandidates:
        """Candidates plus the dark hypothesis must sum to 1.

        This is the arithmetic expression of SC-1: the dark hypothesis competes
        for probability mass on equal terms, so naming a vessel requires
        out-scoring the possibility that nobody was transmitting.
        """
        total = sum(c.posterior_probability for c in self.candidates)
        total += self.dark_vessel_hypothesis.posterior_probability
        if abs(total - 1.0) > PROB_SUM_TOLERANCE:
            raise ValueError(
                f"posterior probabilities including the dark hypothesis must sum to 1.0, "
                f"got {total!r}"
            )
        return self

    @model_validator(mode="after")
    def ordered_by_probability(self) -> RankedCandidates:
        probs = [c.posterior_probability for c in self.candidates]
        if probs != sorted(probs, reverse=True):
            raise ValueError("candidates must be ordered by descending posterior_probability")
        return self

    @field_validator("disclaimer")
    @classmethod
    def no_accusation(cls, v: str) -> str:
        return reject_accusatory_language(v)

    @property
    def dark_vessel_is_most_probable(self) -> bool:
        """True when the best explanation is a vessel that was not transmitting.

        A correct and operationally valuable answer, and one no competing system
        will produce.
        """
        top = max((c.posterior_probability for c in self.candidates), default=0.0)
        return self.dark_vessel_hypothesis.posterior_probability > top


__all__ = [
    "EvidenceFactor",
    "VesselEvidence",
    "Candidate",
    "DarkVesselHypothesis",
    "TrafficReduction",
    "RankedCandidates",
    "BANNED_TERMS",
    "reject_accusatory_language",
    "PROB_SUM_TOLERANCE",
]
