"""Contract 3 of 4: SourcePosterior -- output of the inversion (stage 5).

See Context/architecture.md section 3.3 and technical_design.md section 4.

Two fields here exist purely to keep us honest:

``effective_sample_size``
    If it collapses, the posterior is Monte Carlo noise and the UI must say so
    rather than draw a confident-looking blob (watch-list W-08).

``assumptions``
    Carried all the way to the UI and the evidence export. Every probability we
    display appears next to the list of things that would have to be true for it
    to mean anything. Non-empty by construction.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from .common import Polygon, StrictModel, require_utc, validate_polygon

# Beyond this lookback the envelope grows super-linearly (shear dispersion
# separates neighbouring particles roughly exponentially at short lags) and
# stops being operationally useful. Reported as a system limit, not hidden.
# See technical_design.md section 4.6 and FR-10a.
MAX_USEFUL_LOOKBACK_HOURS = 72.0


class PosteriorGrid(StrictModel):
    """Axes of the 3-D posterior. Density itself lives in an .npz sidecar."""

    lon: list[float] = Field(min_length=2)
    lat: list[float] = Field(min_length=2)
    t0: list[datetime] = Field(min_length=2, description="Release-time bin edges (UTC)")
    density_path: str = Field(description="Path to the .npz holding the normalised density")
    normalised: bool

    _utc = field_validator("t0")(lambda v: [require_utc(t) for t in v])

    @field_validator("normalised")
    @classmethod
    def must_be_normalised(cls, v: bool) -> bool:
        if not v:
            raise ValueError("posterior density must be normalised before serialisation")
        return v


class CredibleRegion(StrictModel):
    """A highest-density region and its area.

    The area in km^2 is the honest measure of how much we actually narrowed the
    search. It is the number that goes on screen.
    """

    polygon_wgs84: Polygon
    area_km2: float = Field(gt=0.0)

    _poly = field_validator("polygon_wgs84")(validate_polygon)


class TimeMarginal(StrictModel):
    """Posterior over release time alone.

    The width of this is our statement about slick age. We do not estimate age
    and then drift; we drift and thereby infer age, with error bars.
    See technical_design.md section 2 ("Age").
    """

    bins_utc: list[datetime] = Field(min_length=2)
    density: list[float] = Field(min_length=1)
    hpd_95: tuple[datetime, datetime] = Field(description="95% highest posterior density interval")
    width_hours: float = Field(gt=0.0)

    _utc_bins = field_validator("bins_utc")(lambda v: [require_utc(t) for t in v])

    @field_validator("density")
    @classmethod
    def non_negative(cls, v: list[float]) -> list[float]:
        if any(d < 0.0 for d in v):
            raise ValueError("density values must be non-negative")
        return v

    @model_validator(mode="after")
    def hpd_ordered(self) -> TimeMarginal:
        lo, hi = self.hpd_95
        require_utc(lo)
        require_utc(hi)
        if lo >= hi:
            raise ValueError("hpd_95 lower bound must be < upper bound")
        return self


class SourcePosterior(StrictModel):
    """p(x0, t0 | observed slick), with credible regions and stated assumptions."""

    case_id: str = Field(min_length=1)
    detection_id: str = Field(min_length=1)

    grid: PosteriorGrid
    credible_regions: dict[str, CredibleRegion] = Field(
        description="Keyed by nominal level as a string, e.g. '50' and '95'"
    )
    t0_marginal: TimeMarginal

    lookback_hours: float = Field(gt=0.0)
    within_operating_envelope: bool

    n_hypotheses: int = Field(gt=0)
    n_particles: int = Field(gt=0)
    effective_sample_size: float = Field(
        gt=0.0,
        description=(
            "ESS of the weighted ensemble. If this collapses relative to n_particles "
            "the posterior is Monte Carlo noise and the UI must say so (W-08)."
        ),
    )

    assumptions: list[str] = Field(
        min_length=1,
        description=(
            "Stated model assumptions, carried to the UI and the evidence export. "
            "Every displayed probability is qualified by these. Non-empty by design: "
            "an unqualified probability is a marketing claim, not a scientific one."
        ),
    )

    @field_validator("credible_regions")
    @classmethod
    def levels_are_required(cls, v: dict[str, CredibleRegion]) -> dict[str, CredibleRegion]:
        for level in ("50", "95"):
            if level not in v:
                raise ValueError(f"credible_regions must include the '{level}' level")
        return v

    @model_validator(mode="after")
    def regions_nest(self) -> SourcePosterior:
        """A 50% region cannot be larger than the 95% region containing it."""
        if self.credible_regions["50"].area_km2 >= self.credible_regions["95"].area_km2:
            raise ValueError("50% credible region area must be smaller than the 95% region")
        return self

    @model_validator(mode="after")
    def ess_is_bounded(self) -> SourcePosterior:
        if self.effective_sample_size > self.n_particles:
            raise ValueError("effective_sample_size cannot exceed n_particles")
        return self

    @model_validator(mode="after")
    def operating_envelope_is_honest(self) -> SourcePosterior:
        """Past the horizon, the envelope flag must admit it (FR-10a)."""
        if self.lookback_hours > MAX_USEFUL_LOOKBACK_HOURS and self.within_operating_envelope:
            raise ValueError(
                f"lookback {self.lookback_hours} h exceeds the "
                f"{MAX_USEFUL_LOOKBACK_HOURS} h horizon, so within_operating_envelope "
                f"must be False"
            )
        return self

    @property
    def ess_fraction(self) -> float:
        """Below roughly 0.01 the posterior should be treated as unreliable."""
        return self.effective_sample_size / self.n_particles


__all__ = [
    "PosteriorGrid",
    "CredibleRegion",
    "TimeMarginal",
    "SourcePosterior",
    "MAX_USEFUL_LOOKBACK_HOURS",
]
