"""Assemble a complete, self-contained Case folder.

Phase 1. Produces the same folder layout the real ingest will produce at Phase 8,
so every downstream stage is written once and never rewritten:

    <case_dir>/
      case.json                 CaseManifest
      forcing_bundle.json       ForcingBundle
      scene/sigma0_vv_db.tif    calibrated sigma-0 VV, dB
      scene/incidence.tif       per-pixel incidence angle
      forcing/currents.nc       currents + tides + Stokes, merged
      forcing/wind.nc           10 m u/v components
      forcing/land.tif          1 = land, 0 = sea
      ais/tracks.parquet        AIS message log
      out/                      pipeline output, written by run_case.py

Everything is written once, offline afterwards, and reproducible from
``(case_id, seed)`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.contracts import (
    BoundingBox,
    CaseManifest,
    CurrentsSpec,
    ForcingBundle,
    SceneSpec,
    WindSpec,
)
from src.ingest.synthetic_ais import AISConfig, write_ais
from src.ingest.synthetic_forcing import SyntheticForcingConfig, write_forcing
from src.ingest.synthetic_scene import SceneConfig, write_scene

FORCING_BUNDLE_NAME = "forcing_bundle.json"

# Enough history for the inversion to look back over, plus margin. The
# operating envelope caps useful lookback at 72 h; 60 h of forcing gives room to
# demonstrate both a comfortable inversion and one at the edge.
DEFAULT_LOOKBACK_HOURS = 60


@dataclass(frozen=True)
class SyntheticCaseSpec:
    """Everything needed to generate one synthetic case."""

    case_id: str
    description: str
    region: str
    aoi: BoundingBox
    t_obs: datetime
    seed: int
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS

    forcing: SyntheticForcingConfig = field(default_factory=SyntheticForcingConfig)
    ais: AISConfig = field(default_factory=AISConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    notes: tuple[str, ...] = ()

    @property
    def t_start(self) -> datetime:
        return self.t_obs - timedelta(hours=self.lookback_hours)


def build_synthetic_case(spec: SyntheticCaseSpec, root: Path) -> Path:
    """Generate a complete Case folder. Returns its path."""
    case_dir = root / spec.case_id

    manifest = CaseManifest(
        case_id=spec.case_id,
        description=spec.description,
        region=spec.region,
        aoi=spec.aoi,
        t_obs=spec.t_obs,
        t_start=spec.t_start,
        scene=SceneSpec(
            sigma0_path="scene/sigma0_vv_db.tif",
            incidence_path="scene/incidence.tif",
            platform="Synthetic (Sentinel-1 IW GRD analogue)",
            polarisation="VV",
            product_type="GRD",
            pixel_spacing_m=spec.scene.pixel_spacing_m,
        ),
        seed=spec.seed,
        ais_source="synthetic",
        # Disclosed synthetic data is a methodological choice; discovered
        # synthetic data is a credibility collapse. This flag is surfaced by
        # run_case.py and in the UI.
        ais_is_synthetic=True,
        notes=[
            "SYNTHETIC CASE. Forcing, scene and AIS are all analytically generated.",
            "Used for development and for the synthetic-truth evaluation harness.",
            "Real cases follow the identical folder layout and contracts.",
            *spec.notes,
        ],
    )
    manifest.save(case_dir)

    write_forcing(case_dir / "forcing", spec.aoi, spec.t_start, spec.t_obs, spec.forcing)
    write_scene(
        case_dir / "scene",
        spec.aoi,
        spec.t_obs,
        spec.t_start,
        spec.seed,
        spec.forcing,
        spec.scene,
    )
    write_ais(case_dir / "ais", spec.aoi, spec.t_start, spec.t_obs, spec.seed, spec.ais)

    bundle = ForcingBundle(
        case_id=spec.case_id,
        aoi=spec.aoi,
        t_start=spec.t_start,
        t_obs=spec.t_obs,
        currents=CurrentsSpec(
            path="forcing/currents.nc",
            product="synthetic (CMEMS SMOC analogue)",
            vars=["uo", "vo"],
            includes_tides=True,
            # The synthetic field merges Stokes drift into uo/vo exactly as
            # SMOC does, so the transport kernel must not add it again (P-05).
            includes_stokes=True,
            resolution_deg=spec.forcing.grid.currents_resolution_deg,
            rms_error_ms=spec.forcing.grid.currents_rms_error_ms,
        ),
        wind=WindSpec(
            path="forcing/wind.nc",
            product="synthetic (ERA5 analogue), 10 m u/v",
            vars=["u10", "v10"],
            resolution_deg=spec.forcing.grid.wind_resolution_deg,
        ),
        land_mask="forcing/land.tif",
        crs_working=spec.aoi.utm_epsg(),
    )
    (case_dir / FORCING_BUNDLE_NAME).write_text(
        bundle.model_dump_json(indent=2), encoding="utf-8"
    )

    return case_dir


def load_forcing_bundle(case_dir: Path) -> ForcingBundle:
    path = Path(case_dir) / FORCING_BUNDLE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {FORCING_BUNDLE_NAME} in {case_dir}")
    return ForcingBundle.model_validate_json(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------- named cases
#
# Two development cases. The first mirrors the primary validation region; the
# second mirrors the Indian-waters demonstration case, where synthetic AIS is
# what the problem statement explicitly permits.

SYNTH_KATTEGAT = SyntheticCaseSpec(
    case_id="synth_kattegat",
    description=(
        "Synthetic shelf sea with a mesoscale eddy, M2 tide, sheared background "
        "flow and a low-wind pocket that presents as a genuine look-alike."
    ),
    region="Synthetic analogue of the Kattegat / Skagerrak",
    aoi=BoundingBox(min_lon=10.2, min_lat=56.4, max_lon=12.8, max_lat=58.1),
    t_obs=datetime(2024, 3, 11, 5, 42, 13, tzinfo=timezone.utc),
    seed=20260906,
    notes=("Primary development case.",),
)

SYNTH_KUTCH = SyntheticCaseSpec(
    case_id="synth_kutch",
    description=(
        "Synthetic Arabian Sea case for the Indian-waters demonstration. "
        "No free bulk historical AIS source exists for this region, so AIS is "
        "generated -- which the problem statement explicitly permits."
    ),
    region="Synthetic analogue of the Gulf of Kutch, Arabian Sea",
    aoi=BoundingBox(min_lon=67.8, min_lat=21.4, max_lon=70.4, max_lat=23.1),
    t_obs=datetime(2024, 4, 2, 1, 18, 44, tzinfo=timezone.utc),
    seed=20260907,
    notes=("Indian-waters demonstration case. Transferability, not validation.",),
)

NAMED_CASES: dict[str, SyntheticCaseSpec] = {
    SYNTH_KATTEGAT.case_id: SYNTH_KATTEGAT,
    SYNTH_KUTCH.case_id: SYNTH_KUTCH,
}


__all__ = [
    "SyntheticCaseSpec",
    "build_synthetic_case",
    "load_forcing_bundle",
    "NAMED_CASES",
    "SYNTH_KATTEGAT",
    "SYNTH_KUTCH",
    "FORCING_BUNDLE_NAME",
]
