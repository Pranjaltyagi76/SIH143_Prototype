"""Phase 1 tests for the synthetic case generator.

These do not merely check that files appear. They assert that the synthetic
world is *physically usable*, because everything in Phases 2 to 7 is developed
against it. A synthetic ocean that is too smooth, too steady, or too calm would
silently flatter the inversion and we would not find out until real data
arrived at Phase 8.

Specifically:

- the flow must be **sheared**, or particles never separate and backward
  uncertainty is unrealistically small
- the flow must be **time-dependent**, or t0 is unidentifiable and the time
  marginal is flat no matter how good the inversion is
- the wind must contain a region **below 3 m/s**, or the physics gate has
  nothing to reject and the strongest demo beat cannot be shown
- no generated MMSI may collide with the real ship-station range (W-14)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.contracts import CaseManifest, ForcingBundle
from src.contracts.detection import WIND_MAX_MS, WIND_MIN_MS
from src.ingest.case_builder import (
    SYNTH_KATTEGAT,
    build_synthetic_case,
    load_forcing_bundle,
)
from src.ingest.synthetic_ais import (
    REAL_MID_MAX,
    REAL_MID_MIN,
    generate_ais,
    is_synthetic_mmsi,
    synthetic_mmsi,
)
from src.ingest.synthetic_forcing import (
    SyntheticForcingConfig,
    current_field,
    km_offsets,
    land_mask,
    wind_field,
)

CFG = SyntheticForcingConfig()
AOI = SYNTH_KATTEGAT.aoi
T_START = SYNTH_KATTEGAT.t_start
T_OBS = SYNTH_KATTEGAT.t_obs


@pytest.fixture(scope="module")
def built_case(tmp_path_factory) -> Path:
    """Build the primary development case once for the whole module."""
    root = tmp_path_factory.mktemp("cases")
    return build_synthetic_case(SYNTH_KATTEGAT, root)


def _grid(step: float = 0.0833):
    lon = np.arange(AOI.min_lon, AOI.max_lon, step)
    lat = np.arange(AOI.min_lat, AOI.max_lat, step)
    return lon, lat


# ------------------------------------------------------------- ocean physics


def test_current_speeds_are_physically_plausible():
    lon, lat = _grid()
    u, v = current_field(lon, lat, np.arange(0.0, 60.0), AOI, CFG.ocean)
    speed = np.hypot(u, v)
    assert not np.isnan(speed).any(), "NaN velocity silently freezes a particle"
    assert 0.05 < speed.mean() < 0.6, f"mean {speed.mean():.3f} m/s is not a shelf-sea regime"
    assert speed.max() < 1.5


def test_flow_is_sheared_so_particles_separate():
    """Shear dispersion is what makes backward uncertainty grow super-linearly.
    Without it the synthetic ocean would make the inversion look better than it is."""
    lon, lat = _grid()
    u, _ = current_field(lon, lat, np.array([0.0]), AOI, CFG.ocean)
    south = u[0, : len(lat) // 4, :].mean()
    north = u[0, -len(lat) // 4 :, :].mean()
    assert abs(north - south) > 0.05, "background flow has no meaningful meridional shear"


def test_flow_is_time_dependent_so_t0_is_identifiable():
    """In a steady flow many (x0, t0) pairs give an identical observed slick and
    the time marginal is flat regardless of how good the inversion is. The M2
    tide is what breaks that degeneracy."""
    lon, lat = _grid()
    hours = np.array([0.0, CFG.ocean.tide_period_hours / 2.0])
    u, v = current_field(lon, lat, hours, AOI, CFG.ocean)
    change = np.hypot(u[1] - u[0], v[1] - v[0]).mean()
    assert change > 0.05, f"flow barely changes over half a tidal cycle ({change:.3f} m/s)"


def test_eddy_produces_rotation():
    """Curved trajectories, so backward advection is not a straight line."""
    lon, lat = _grid(0.02)
    u, v = current_field(lon, lat, np.array([0.0]), AOI, CFG.ocean)
    # Curl of the horizontal flow should be clearly non-zero somewhere.
    dv_dx = np.gradient(v[0], axis=1)
    du_dy = np.gradient(u[0], axis=0)
    assert np.abs(dv_dx - du_dy).max() > 1e-4


# -------------------------------------------------------------- wind physics


def test_wind_contains_a_region_below_the_detectability_threshold():
    """Without this the physics gate has nothing to reject, and the strongest
    beat in the demo cannot be demonstrated on synthetic data."""
    lon, lat = _grid(0.25)
    u, v = wind_field(lon, lat, np.array([0.0]), AOI, CFG.wind)
    speed = np.hypot(u[0], v[0])
    assert speed.min() < WIND_MIN_MS, (
        f"minimum sampled wind {speed.min():.2f} m/s is not below the "
        f"{WIND_MIN_MS} m/s gate threshold; the calm pocket is under-resolved"
    )


def test_most_of_the_domain_is_inside_the_detectability_window():
    """A domain that is mostly ungateable would make abstention the default
    rather than the exception. Target abstention is 5-15%, not 90%."""
    lon, lat = _grid(0.25)
    u, v = wind_field(lon, lat, np.arange(0.0, 60.0), AOI, CFG.wind)
    speed = np.hypot(u, v)
    inside = ((speed >= WIND_MIN_MS) & (speed <= WIND_MAX_MS)).mean()
    assert inside > 0.85, f"only {inside:.0%} of the domain is inside the window"


def test_wind_direction_rotates_over_time():
    lon, lat = _grid(0.25)
    u, v = wind_field(lon, lat, np.array([0.0, 48.0]), AOI, CFG.wind)
    bearing = lambda uu, vv: np.rad2deg(np.arctan2(uu.mean(), vv.mean()))  # noqa: E731
    assert abs(bearing(u[1], v[1]) - bearing(u[0], v[0])) > 5.0


def test_wind_components_encode_from_direction_correctly():
    """For wind FROM the south-west (225 deg) the vector must point north-east:
    u > 0 and v > 0. Getting this backwards is W-02, and it makes the slick
    drift in exactly the wrong direction."""
    from src.ingest.synthetic_forcing import WindConfig

    cfg = WindConfig(base_from_direction_deg=225.0, rotation_deg_per_day=0.0,
                     calm_min_speed_ms=6.0)
    lon, lat = _grid(0.25)
    u, v = wind_field(lon, lat, np.array([0.0]), AOI, cfg)
    assert u.mean() > 0.0 and v.mean() > 0.0


# ---------------------------------------------------------------------- land


def test_land_mask_has_both_land_and_sea():
    lon, lat = _grid(0.01)
    mask = land_mask(lon, lat, AOI, CFG.land)
    frac = mask.mean()
    assert 0.02 < frac < 0.40, f"land fraction {frac:.2%} is degenerate"


def test_land_is_in_the_north():
    lon, lat = _grid(0.01)
    mask = land_mask(lon, lat, AOI, CFG.land)
    northern_half = mask[len(lat) // 2 :, :].mean()
    southern_half = mask[: len(lat) // 2, :].mean()
    assert northern_half > southern_half


# ----------------------------------------------------------------- geometry


def test_km_offsets_are_symmetric_and_scaled():
    lon = np.array([11.0, 11.0])
    lat = np.array([57.0, 57.9])
    dx, dy = km_offsets(lon, lat, 11.0, 57.0)
    assert abs(dx[0]) < 1e-9
    # 0.9 degrees of latitude is very close to 100 km.
    assert 99.0 < dy[1] < 101.0


# ---------------------------------------------------------------------- AIS


def test_synthetic_mmsi_cannot_collide_with_a_real_vessel():
    """A real ship appearing in an accusation demo would be a serious problem.
    Ship-station MMSIs carry a MID prefix in 201-775 (W-14)."""
    df = generate_ais(AOI, T_START, T_OBS, seed=42)
    for mmsi in df["mmsi"].unique():
        assert len(mmsi) == 9 and mmsi.isdigit()
        assert not (REAL_MID_MIN <= int(mmsi[:3]) <= REAL_MID_MAX), f"{mmsi} is in the real MID range"
        assert is_synthetic_mmsi(mmsi)


def test_is_synthetic_mmsi_rejects_real_looking_identifiers():
    assert not is_synthetic_mmsi("219000000")  # Denmark
    assert not is_synthetic_mmsi("419000000")  # India
    assert not is_synthetic_mmsi("not-a-mmsi")
    assert is_synthetic_mmsi(synthetic_mmsi(1))


def test_ais_traffic_is_dense_enough_to_be_a_real_filtering_problem():
    """The prefilter must earn its reduction factor against genuine traffic
    density. Reducing 12 vessels to 3 is not a headline."""
    df = generate_ais(AOI, T_START, T_OBS, seed=42)
    assert df["mmsi"].nunique() > 100
    assert len(df) > 10_000


def test_ais_is_within_the_aoi_and_time_window():
    df = generate_ais(AOI, T_START, T_OBS, seed=42)
    assert df["lon"].between(AOI.min_lon, AOI.max_lon).all()
    assert df["lat"].between(AOI.min_lat, AOI.max_lat).all()
    assert df["timestamp"].min() >= T_START
    assert df["timestamp"].max() <= T_OBS


def test_ais_timestamps_are_utc_aware():
    """W-04: a naive timestamp assumed local shifts every attribution by a
    constant number of hours."""
    df = generate_ais(AOI, T_START, T_OBS, seed=42)
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_ais_has_a_tight_baseline_and_a_minority_with_gaps():
    """The gap prior must be baseline-relative. That requires a baseline to be
    relative to, and a genuine minority of benign gaps (W-07)."""
    df = generate_ais(AOI, T_START, T_OBS, seed=42).sort_values("timestamp")
    max_gap_h = df.groupby("mmsi")["timestamp"].apply(
        lambda s: s.diff().dt.total_seconds().max() / 3600.0
    )
    assert max_gap_h.median() < 0.25, "baseline reporting interval is not tight"
    with_gaps = (max_gap_h > 0.5).sum()
    assert 5 <= with_gaps <= 0.5 * len(max_gap_h), (
        f"{with_gaps} of {len(max_gap_h)} vessels have gaps; need a clear minority"
    )


def test_ais_has_speed_variation_for_the_anomaly_factor():
    df = generate_ais(AOI, T_START, T_OBS, seed=42)
    per_vessel = df.groupby("mmsi")["sog_kn"].median()
    assert per_vessel.min() < 5.0, "no slow vessels; speed anomaly has nothing to find"
    assert per_vessel.max() > 12.0


def test_ais_generation_is_deterministic():
    """NFR-3: same case plus same seed gives identical output."""
    a = generate_ais(AOI, T_START, T_OBS, seed=7)
    b = generate_ais(AOI, T_START, T_OBS, seed=7)
    c = generate_ais(AOI, T_START, T_OBS, seed=8)
    assert a.equals(b)
    assert not a.equals(c)


# ------------------------------------------------------------- case assembly


def test_case_folder_is_complete(built_case: Path):
    for rel in (
        "case.json",
        "forcing_bundle.json",
        "scene/sigma0_vv_db.tif",
        "scene/incidence.tif",
        "forcing/currents.nc",
        "forcing/wind.nc",
        "forcing/land.tif",
        "ais/tracks.parquet",
    ):
        assert (built_case / rel).is_file(), f"missing {rel}"
    assert (built_case / "out").is_dir()


def test_built_case_validates_against_the_contracts(built_case: Path):
    case = CaseManifest.load(built_case)
    bundle = load_forcing_bundle(built_case)
    assert isinstance(case, CaseManifest)
    assert isinstance(bundle, ForcingBundle)
    assert case.case_id == bundle.case_id
    assert case.aoi == bundle.aoi
    assert case.crs_working == bundle.crs_working


def test_synthetic_ais_is_disclosed_not_hidden(built_case: Path):
    """Disclosed synthetic data is a methodological choice; discovered synthetic
    data is a credibility collapse."""
    case = CaseManifest.load(built_case)
    assert case.ais_is_synthetic is True
    assert case.ais_source == "synthetic"
    assert any("SYNTHETIC" in n.upper() for n in case.notes)


def test_forcing_bundle_declares_stokes_as_merged(built_case: Path):
    """The synthetic current field merges Stokes drift exactly as SMOC does, so
    the transport kernel must not add it again (P-05)."""
    bundle = load_forcing_bundle(built_case)
    assert bundle.currents.includes_stokes is True
    assert bundle.currents.includes_tides is True
    assert bundle.currents.rms_error_ms > 0.0


def test_forcing_time_axis_covers_the_whole_window_with_padding(built_case: Path):
    """Interpolating at the exact endpoint of a time axis is a reliable source
    of NaN, and a NaN velocity silently freezes a particle."""
    import xarray as xr

    case = CaseManifest.load(built_case)
    with xr.open_dataset(built_case / "forcing" / "currents.nc") as ds:
        t0 = ds.time.values[0].astype("datetime64[s]").astype(object).replace(tzinfo=timezone.utc)
        t1 = ds.time.values[-1].astype("datetime64[s]").astype(object).replace(tzinfo=timezone.utc)
    assert t0 < case.t_start
    assert t1 > case.t_obs


def test_scene_contains_a_genuine_low_wind_dark_patch(built_case: Path):
    """The calm pocket must actually appear dark in the SAR scene. This is a
    real look-alike arising from the same physics that makes the problem hard --
    it is not painted on -- so the detector has something honest to be fooled by
    and the gate has something honest to reject."""
    import rasterio

    with rasterio.open(built_case / "scene" / "sigma0_vv_db.tif") as src:
        sigma0 = src.read(1)
    h, w = sigma0.shape

    def patch(lon_frac: float, lat_frac: float) -> float:
        r, c = int((1.0 - lat_frac) * h), int(lon_frac * w)
        return float(sigma0[max(0, r - 30) : r + 30, max(0, c - 30) : c + 30].mean())

    calm = patch(CFG.wind.calm_offset_lon_frac, CFG.wind.calm_offset_lat_frac)
    windy = patch(0.30, 0.30)
    assert calm < windy - 3.0, (
        f"calm pocket {calm:.1f} dB vs windy {windy:.1f} dB -- not a convincing look-alike"
    )


def test_scene_incidence_ramps_across_range(built_case: Path):
    """Backscatter falls steeply across the swath. Normalising damping ratio for
    incidence is what makes near-range and far-range patches comparable."""
    import rasterio

    with rasterio.open(built_case / "scene" / "incidence.tif") as src:
        inc = src.read(1)
    assert inc[:, 0].mean() < inc[:, -1].mean() - 10.0
    assert 25.0 < inc.min() < 35.0 and 40.0 < inc.max() < 50.0


def test_case_build_is_reproducible(tmp_path: Path):
    """NFR-3, at the whole-case level."""
    import pandas as pd

    a = build_synthetic_case(SYNTH_KATTEGAT, tmp_path / "a")
    b = build_synthetic_case(SYNTH_KATTEGAT, tmp_path / "b")
    assert (a / "case.json").read_text() == (b / "case.json").read_text()
    assert pd.read_parquet(a / "ais" / "tracks.parquet").equals(
        pd.read_parquet(b / "ais" / "tracks.parquet")
    )


def test_kutch_case_uses_the_correct_utm_zone():
    """Gulf of Kutch sits in zone 42, not 43. Getting the working CRS wrong puts
    every distance and area computation quietly out by a scale factor."""
    from src.ingest.case_builder import SYNTH_KUTCH

    assert SYNTH_KUTCH.aoi.utm_epsg() == "EPSG:32642"
    assert SYNTH_KUTCH.t_obs > datetime(2024, 1, 1, tzinfo=timezone.utc)
