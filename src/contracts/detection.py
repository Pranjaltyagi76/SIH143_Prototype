"""Contract 2 of 4: SlickDetection -- output of stages 1 to 3.

See Context/architecture.md section 3.2 and technical_design.md sections 1 and 2.

The field that matters most here is ``classification_reason``. Showing what the
system correctly REFUSES to flag is the strongest beat in the demo, and it only
works if the reason survives all the way to the frontend. It is mandatory and
non-empty by construction.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from .common import (
    Classification,
    LonLat,
    Polygon,
    ReleaseMode,
    StrictModel,
    validate_polygon,
)


class SlickGeometry(StrictModel):
    """Geometric properties (FR-4). Areas in km^2, computed in projected metres."""

    area_km2: float = Field(gt=0.0)
    perimeter_km: float = Field(gt=0.0)
    centroid: LonLat
    elongation: float = Field(
        ge=1.0, description="Major/minor axis ratio. >4 with track alignment implies continuous discharge."
    )
    orientation_deg: float = Field(ge=0.0, lt=180.0, description="Major axis azimuth, 0-180")
    n_components: int = Field(ge=1, description="Fragmentation: connected component count")
    complexity: float = Field(gt=0.0, description="P^2/(4*pi*A); 1.0 is a perfect circle")
    solidity: float = Field(gt=0.0, le=1.0, description="area / convex hull area")


class SlickRadiometry(StrictModel):
    """Backscatter measurements feeding the look-alike classifier."""

    damping_ratio_db: float = Field(
        description="sigma0(slick) - sigma0(background), dB. Negative for oil."
    )
    damping_norm_incidence: float = Field(
        description=(
            "Damping ratio normalised for incidence angle. Backscatter falls steeply "
            "across the swath; without this, near-range and far-range patches are not "
            "comparable and a naive classifier learns the swath position instead of the oil."
        )
    )
    sigma0_mean_db: float
    sigma0_background_db: float
    incidence_deg: float = Field(gt=0.0, lt=90.0)


class SlickEnvironment(StrictModel):
    """ERA5 conditions at the patch. This is the physics gate input."""

    wind_speed_ms: float = Field(ge=0.0)
    wind_dir_deg: float = Field(ge=0.0, lt=360.0, description="Derived for display only")
    wind_gradient_ms_per_km: float = Field(ge=0.0)


# The physical detectability window, in m/s of 10 m wind.
# Below WIND_MIN the sea surface is already smooth, so no oil-water contrast is
# possible. Above WIND_MAX wave action disperses and submerges the slick.
# See technical_design.md section 1.4.
WIND_MIN_MS = 3.0
WIND_MAX_MS = 12.0


class SlickDetection(StrictModel):
    """One candidate dark patch, classified and characterised."""

    detection_id: str = Field(min_length=1, description="Format: '<case_id>:d<nn>'")
    classification: Classification
    classification_reason: str = Field(
        min_length=1,
        description=(
            "Human-readable justification. MANDATORY and rendered in the UI. "
            "For an abstention this states the wind speed that put the patch outside "
            "the detectability window."
        ),
    )
    abstained: bool
    confidence_segmentation: float = Field(ge=0.0, le=1.0)

    polygon_wgs84: Polygon
    geometry: SlickGeometry
    radiometry: SlickRadiometry
    environment: SlickEnvironment

    release_mode: ReleaseMode
    release_mode_justification: str = Field(min_length=1)

    _poly = field_validator("polygon_wgs84")(validate_polygon)

    @model_validator(mode="after")
    def abstention_is_consistent(self) -> SlickDetection:
        """``abstained`` and ``UNDETERMINED`` must agree.

        Letting these diverge would allow the UI to show a confident label for a
        patch the system actually declined to judge.
        """
        is_undetermined = self.classification is Classification.UNDETERMINED
        if is_undetermined != self.abstained:
            raise ValueError(
                f"abstained={self.abstained} contradicts classification={self.classification.value}"
            )
        return self

    @model_validator(mode="after")
    def hard_wind_gate_is_enforced(self) -> SlickDetection:
        """The physics gate cannot be overridden by the classifier (FR-3a).

        Outside the detectability window no oil-water contrast is physically
        possible, so any confident label there is unsupportable regardless of
        what LightGBM produced. Enforced in the contract so that no downstream
        stage, and no future refactor, can quietly bypass it.
        """
        w = self.environment.wind_speed_ms
        if (w < WIND_MIN_MS or w > WIND_MAX_MS) and not self.abstained:
            raise ValueError(
                f"wind {w} m/s is outside the detectability window "
                f"[{WIND_MIN_MS}, {WIND_MAX_MS}] so classification must be "
                f"'undetermined', got '{self.classification.value}' (FR-3a)"
            )
        return self

    @property
    def is_actionable(self) -> bool:
        """Only confirmed oil is passed to the inversion."""
        return self.classification is Classification.OIL


__all__ = [
    "SlickGeometry",
    "SlickRadiometry",
    "SlickEnvironment",
    "SlickDetection",
    "WIND_MIN_MS",
    "WIND_MAX_MS",
]
