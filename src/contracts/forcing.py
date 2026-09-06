"""Contract 1 of 4: ForcingBundle -- the offline data guarantee.

See Context/architecture.md section 3.1.

This contract exists for two reasons:

1. It guarantees NFR-1 (full offline operation) structurally. Nothing downstream
   is permitted to fetch data; it reads paths from here or it fails.
2. It carries ``includes_stokes``, which prevents the double-counting bug
   recorded as P-05 in Context/problems_faced_and_bugs_encountered.md.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from .common import BoundingBox, StrictModel, require_utc


class CurrentsSpec(StrictModel):
    """Ocean surface current forcing.

    Default product is CMEMS SMOC (``cmems_mod_glo_phy_anfc_merged-uv_PT1H-i``),
    which CMEMS explicitly recommends for Lagrangian drift applications.
    """

    path: str = Field(description="Path to NetCDF, relative to the case folder")
    product: str
    vars: list[str] = Field(min_length=2, description="Velocity component names, e.g. ['uo','vo']")

    includes_tides: bool = Field(
        description="True if tidal currents are already merged into the velocity field."
    )
    includes_stokes: bool = Field(
        description=(
            "True if Stokes drift is ALREADY MERGED into the velocity field. "
            "CMEMS SMOC merges it; adding a separate Stokes parameterisation on top "
            "would double-count a real physical term and bias every trajectory "
            "downwind, silently, with no error raised. The transport kernel MUST "
            "read this flag rather than assume. There is deliberately no default: "
            "whoever builds a case has to look it up in the product QUID. "
            "See problems_faced_and_bugs_encountered.md P-05."
        )
    )

    resolution_deg: float = Field(gt=0.0, description="Nominal grid spacing in degrees")
    rms_error_ms: float = Field(
        ge=0.0,
        description=(
            "Published RMS velocity error (CMEMS QUID). Used to scale the correlated "
            "noise perturbation of the current field in the ensemble, so that MODEL "
            "error enters the posterior rather than being ignored."
        ),
    )


class WindSpec(StrictModel):
    """10 m wind forcing, normally ERA5 single-levels.

    Components only, never a bearing. Meteorological 'wind from' versus
    oceanographic 'wind to' is watch-list item W-02: the sign error makes the
    slick drift in exactly the wrong direction. Direction is derived once, at
    the UI boundary.
    """

    path: str
    product: str
    vars: list[str] = Field(min_length=2, description="Component names, e.g. ['u10','v10']")
    resolution_deg: float = Field(gt=0.0)

    @field_validator("vars")
    @classmethod
    def components_not_bearing(cls, v: list[str]) -> list[str]:
        banned = {"wind_dir", "wdir", "direction", "bearing", "wind_speed", "wspd"}
        if any(name.lower() in banned for name in v):
            raise ValueError(
                "wind must be supplied as u/v components, not speed or bearing (see W-02)"
            )
        return v


class ForcingBundle(StrictModel):
    """Everything the transport kernel needs, cached on disk, for one case."""

    case_id: str = Field(min_length=1)
    aoi: BoundingBox
    t_start: datetime = Field(description="Earliest forcing time available (UTC)")
    t_obs: datetime = Field(description="Satellite acquisition time (UTC)")

    currents: CurrentsSpec
    wind: WindSpec
    land_mask: str = Field(description="Path to the land raster, relative to the case folder")
    crs_working: str = Field(
        pattern=r"^EPSG:\d{4,5}$",
        description="Local projected CRS (UTM). ALL computation happens here, in metres.",
    )

    _utc = field_validator("t_start", "t_obs")(require_utc)

    @model_validator(mode="after")
    def check_window(self) -> ForcingBundle:
        if self.t_start >= self.t_obs:
            raise ValueError("t_start must be strictly before t_obs")
        lookback_h = (self.t_obs - self.t_start).total_seconds() / 3600.0
        if lookback_h < 6.0:
            raise ValueError(
                f"forcing window is only {lookback_h:.1f} h; the inversion needs at least 6 h"
            )
        return self

    @property
    def available_lookback_hours(self) -> float:
        return (self.t_obs - self.t_start).total_seconds() / 3600.0


__all__ = ["CurrentsSpec", "WindSpec", "ForcingBundle"]
