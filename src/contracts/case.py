"""The Case manifest -- the architectural spine.

See Context/architecture.md section 1.

A Case is a self-contained, on-disk bundle holding everything needed to
reproduce one analysis. The system is not a service; it is a batch pipeline over
immutable Case folders. That single decision is what makes the demo offline by
construction, reproducible, cacheable, and parallelisable across six people.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from .common import BoundingBox, StrictModel, require_utc

CASE_MANIFEST_NAME = "case.json"

# Subdirectories every case folder carries.
CASE_LAYOUT = ("scene", "forcing", "ais", "out")


class SceneSpec(StrictModel):
    """The SAR observation.

    The prototype consumes pre-calibrated sigma0 in dB (Zenodo tiles), which is
    why no SNAP dependency exists. Round 3 adds raw GRD ingest behind this same
    field. See engineering_review.md debt item D2.
    """

    sigma0_path: str = Field(description="Calibrated sigma0 VV in dB, georeferenced")
    incidence_path: str = Field(description="Per-pixel incidence angle raster")
    platform: str = Field(default="Sentinel-1", min_length=1)
    polarisation: str = Field(default="VV", pattern=r"^(VV|VH|HH|HV)$")
    product_type: str = Field(default="GRD", pattern=r"^(GRD|SLC)$")
    pixel_spacing_m: float = Field(default=10.0, gt=0.0)

    @field_validator("polarisation")
    @classmethod
    def prefer_co_pol(cls, v: str) -> str:
        """VH sits near the noise floor over low-backscatter water, so
        slick-versus-sea contrast collapses exactly where we need it. Bragg
        scattering, which oil damps, is a co-pol phenomenon. VH is kept only as
        a noise-floor sanity channel, never as the detection channel.
        """
        return v


class CaseManifest(StrictModel):
    """``case.json`` -- the complete, reproducible input to one analysis."""

    case_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    description: str = Field(min_length=1)
    region: str = Field(min_length=1)

    aoi: BoundingBox
    t_obs: datetime = Field(description="Satellite acquisition time (UTC)")
    t_start: datetime = Field(description="Earliest forcing time; sets the lookback limit (UTC)")

    scene: SceneSpec

    seed: int = Field(
        ge=0,
        description=(
            "Threaded to every RNG in the pipeline. Same case plus same seed gives a "
            "bit-identical posterior (NFR-3). No bare np.random calls anywhere."
        ),
    )

    ais_source: str = Field(
        min_length=1,
        description="e.g. 'danish_dma' (real) or 'synthetic' (generated, and disclosed)",
    )
    ais_is_synthetic: bool = Field(
        description=(
            "Disclosed synthetic data is a methodological choice; discovered synthetic "
            "data is a credibility collapse. This flag is surfaced in the UI and on the "
            "data slide. The PS explicitly permits synthetic AIS where real is unavailable."
        )
    )

    notes: list[str] = Field(default_factory=list)

    _utc = field_validator("t_obs", "t_start")(require_utc)

    @model_validator(mode="after")
    def window_ordered(self) -> CaseManifest:
        if self.t_start >= self.t_obs:
            raise ValueError("t_start must be strictly before t_obs")
        return self

    @property
    def max_lookback_hours(self) -> float:
        return (self.t_obs - self.t_start).total_seconds() / 3600.0

    @property
    def crs_working(self) -> str:
        """Local UTM zone. All computation happens here, in metres."""
        return self.aoi.utm_epsg()

    @classmethod
    def load(cls, case_dir: Path | str) -> CaseManifest:
        path = Path(case_dir) / CASE_MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(f"no {CASE_MANIFEST_NAME} in {case_dir}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, case_dir: Path | str) -> Path:
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        for sub in CASE_LAYOUT:
            (case_dir / sub).mkdir(exist_ok=True)
        path = case_dir / CASE_MANIFEST_NAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


__all__ = ["SceneSpec", "CaseManifest", "CASE_MANIFEST_NAME", "CASE_LAYOUT"]
