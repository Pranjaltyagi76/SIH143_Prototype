"""Analytic ocean and atmosphere fields, written as CF-style NetCDF.

Phase 1. These stand in for CMEMS SMOC currents and ERA5 10 m wind so that the
transport kernel, the inversion and the attribution engine can all be built and
tested before a single byte is downloaded. Real forcing slots in at Phase 8
behind the same ``ForcingBundle`` contract and the same file layout.

The fields are not arbitrary. Each term exists because the inversion needs it:

``shear`` and ``jet``
    A meridional gradient in the background flow, plus a narrow zonal jet whose
    flanks carry strong lateral shear. Two particles a few hundred metres apart
    end up in slightly different velocity cells and separate -- measured at
    2.1-3.1x over 48 h. This is the mechanism that makes backward uncertainty
    grow super-linearly, and without it the synthetic ocean would flatter the
    inversion. The large-scale shear alone gave only 1.1-1.3x, which is why the
    jet is there.

``eddy``
    A Gaussian vortex, so trajectories curve and backward advection is not a
    straight line. Constructed from a streamfunction, hence non-divergent.

``tide``
    A rotary M2 oscillation. Crucially this makes the flow TIME-DEPENDENT, which
    is what makes t0 identifiable at all: in a steady flow, many (x0, t0) pairs
    produce an identical observed slick and the time marginal is flat.

``calm pocket``
    A region where 10 m wind drops below 3 m/s. This is what gives the physics
    gate something real to reject, and it is the strongest beat in the demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from src.contracts import BoundingBox

# Mean Earth radius, metres. Used only for the local equirectangular mapping
# between degrees and kilometres when synthesising fields.
EARTH_RADIUS_M = 6_371_000.0

# Principal lunar semi-diurnal tidal constituent, hours.
M2_PERIOD_HOURS = 12.4206


@dataclass(frozen=True)
class OceanConfig:
    """Surface current field.

    Speeds are m/s. The defaults give a ~0.2-0.4 m/s regime, which is typical of
    a shelf sea and produces a plausible 20-60 km of drift over 36 hours.
    """

    background_u_ms: float = 0.14
    background_v_ms: float = 0.06

    # Fractional change in background u per 100 km of northward offset.
    # This is the shear that drives particle separation.
    shear_per_100km: float = 0.45

    eddy_offset_lon_frac: float = 0.35
    eddy_offset_lat_frac: float = 0.55
    eddy_radius_km: float = 28.0
    eddy_strength_ms: float = 0.22

    # A narrow zonal jet. Measured on the first build, the eddy plus large-scale
    # shear alone separated a 250 m particle cloud by only 1.1-1.3x over 48 h --
    # an effective strain rate of ~1.5e-6 /s, at the very bottom of the
    # realistic range. Fronts and jets are where sub-mesoscale separation
    # actually happens, and this is the term that makes the design's claim about
    # shear dispersion demonstrable rather than asserted.
    #
    # Measured after adding it: 2.1-3.1x separation over 48 h depending on
    # where the cloud starts, an effective strain of ~6.5e-6 /s. Realistic for
    # a shelf sea, and enough that shear dispersion is demonstrable rather than
    # asserted. u depends on y only, so the jet is divergence-free.
    jet_lat_frac: float = 0.45
    jet_width_km: float = 18.0
    jet_speed_ms: float = 0.35

    tide_amplitude_ms: float = 0.18
    tide_period_hours: float = M2_PERIOD_HOURS
    tide_ellipticity: float = 0.45


@dataclass(frozen=True)
class WindConfig:
    """10 m wind field.

    Stored as u/v components only, never as speed and bearing. Meteorological
    "wind from" versus oceanographic "wind to" is watch-list item W-02: the sign
    error makes the slick drift in exactly the wrong direction.
    """

    base_speed_ms: float = 6.8
    base_from_direction_deg: float = 235.0
    rotation_deg_per_day: float = 18.0

    # A pocket of near-calm water. Inside it no oil-water radar contrast is
    # physically possible, so any dark patch here must be reported as
    # 'undetermined' rather than as oil.
    calm_offset_lon_frac: float = 0.72
    calm_offset_lat_frac: float = 0.78
    # 35 km, not 22 km. At the ERA5 0.25 deg spacing a 22 km pocket falls
    # between grid points and the sampled minimum came out at 2.0 m/s rather
    # than the analytic 1.3 -- close enough to the 3 m/s threshold to be fragile.
    # This is a real effect, not an artefact: coarse reanalysis genuinely
    # under-resolves small calm patches, which is one reason look-alike
    # rejection is hard. We keep the effect but give the gate a clear signal.
    calm_radius_km: float = 35.0
    calm_min_speed_ms: float = 1.3


@dataclass(frozen=True)
class GridSpec:
    """Grid resolutions, chosen to mirror the real products we will swap in."""

    currents_resolution_deg: float = 1.0 / 12.0  # CMEMS SMOC
    wind_resolution_deg: float = 0.25  # ERA5
    time_step_hours: int = 1

    # Published RMS velocity error for the real product. Carried into the
    # ForcingBundle so the ensemble can perturb the current field by MODEL
    # error, not merely by resolution.
    currents_rms_error_ms: float = 0.12


@dataclass(frozen=True)
class LandConfig:
    """A wiggly coastline across the northern edge of the AOI.

    Particles that reach it are marked beached and frozen, never deleted --
    deleting them silently biases the posterior away from the coast, which is
    exactly where spills matter most (W-03).
    """

    coast_lat_frac: float = 0.88
    wiggle_amplitude_frac: float = 0.045
    wiggle_cycles: float = 3.0
    resolution_deg: float = 0.01


@dataclass(frozen=True)
class SyntheticForcingConfig:
    ocean: OceanConfig = field(default_factory=OceanConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    grid: GridSpec = field(default_factory=GridSpec)
    land: LandConfig = field(default_factory=LandConfig)


# ------------------------------------------------------------------ geometry


def km_offsets(
    lon: np.ndarray, lat: np.ndarray, ref_lon: float, ref_lat: float
) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular offsets in kilometres from a reference point.

    Adequate for synthesising fields over a few hundred kilometres. The
    transport kernel does NOT use this -- it projects to UTM properly, because
    computing distance in degrees is W-11.
    """
    m_per_deg_lat = EARTH_RADIUS_M * np.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(ref_lat))
    return (
        (lon - ref_lon) * m_per_deg_lon / 1000.0,
        (lat - ref_lat) * m_per_deg_lat / 1000.0,
    )


def _axis(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(np.floor((hi - lo) / step)) + 1
    return (lo + step * np.arange(n)).astype("float64")


def _hours_since(times: list[datetime], origin: datetime) -> np.ndarray:
    return np.array([(t - origin).total_seconds() / 3600.0 for t in times], dtype="float64")


def as_naive_utc(times: list[datetime]) -> np.ndarray:
    """Timezone-aware UTC stamps -> naive ``datetime64[ns]`` for NetCDF.

    NetCDF and CF have no timezone concept; time is stored numerically against a
    reference epoch. Everything in this project is UTC (enforced at the contract
    boundary by ``require_utc``), so dropping the tzinfo loses no information --
    but readers must therefore re-attach UTC on load rather than assume local
    time, which is watch-list item W-04.
    """
    import pandas as pd

    return pd.DatetimeIndex(times).tz_convert("UTC").tz_localize(None).to_numpy()


def _time_axis(t_start: datetime, t_end: datetime, step_hours: int) -> list[datetime]:
    """Hourly stamps spanning [t_start, t_end], with one hour of padding.

    The padding matters: interpolation at the exact endpoint of a time axis is a
    reliable source of NaN, and a NaN velocity silently freezes a particle.
    """
    start = t_start - timedelta(hours=step_hours)
    end = t_end + timedelta(hours=step_hours)
    out, cursor = [], start
    while cursor <= end:
        out.append(cursor)
        cursor += timedelta(hours=step_hours)
    return out


# -------------------------------------------------------------------- fields


def current_field(
    lon: np.ndarray, lat: np.ndarray, hours: np.ndarray, aoi: BoundingBox, cfg: OceanConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Surface current (u, v) in m/s, shaped (time, lat, lon).

    Represents currents + tides + Stokes drift merged, exactly as CMEMS SMOC
    supplies them. ``ForcingBundle.currents.includes_stokes`` is therefore set
    True for synthetic cases, and the transport kernel must NOT add a separate
    Stokes term on top (P-05).
    """
    ref_lon, ref_lat = aoi.centroid
    lon2d, lat2d = np.meshgrid(lon, lat)
    dx_km, dy_km = km_offsets(lon2d, lat2d, ref_lon, ref_lat)

    # Sheared background. u grows northward, so neighbouring particles separate.
    u_bg = cfg.background_u_ms * (1.0 + cfg.shear_per_100km * dy_km / 100.0)
    v_bg = np.full_like(u_bg, cfg.background_v_ms)

    # Gaussian vortex from a streamfunction psi = strength * R * exp(-r^2/2R^2),
    # so (u, v) = (-d/dy, +d/dx) is non-divergent by construction.
    eddy_lon = aoi.min_lon + cfg.eddy_offset_lon_frac * (aoi.max_lon - aoi.min_lon)
    eddy_lat = aoi.min_lat + cfg.eddy_offset_lat_frac * (aoi.max_lat - aoi.min_lat)
    ex_km, ey_km = km_offsets(lon2d, lat2d, eddy_lon, eddy_lat)
    r = cfg.eddy_radius_km
    decay = np.exp(-(ex_km**2 + ey_km**2) / (2.0 * r**2))
    u_eddy = -cfg.eddy_strength_ms * (ey_km / r) * decay
    v_eddy = cfg.eddy_strength_ms * (ex_km / r) * decay

    # Zonal jet: u is a function of y alone, so it adds no divergence. Strong
    # lateral shear on its flanks is what produces realistic particle
    # separation over a day or two.
    jet_lat = aoi.min_lat + cfg.jet_lat_frac * (aoi.max_lat - aoi.min_lat)
    _, jy_km = km_offsets(lon2d, lat2d, aoi.centroid[0], jet_lat)
    u_jet = cfg.jet_speed_ms * np.exp(-((jy_km / cfg.jet_width_km) ** 2))

    # Rotary tide: spatially uniform, purely time-dependent. This is what makes
    # the release time recoverable.
    omega = 2.0 * np.pi / cfg.tide_period_hours
    u_tide = cfg.tide_amplitude_ms * np.cos(omega * hours)
    v_tide = cfg.tide_amplitude_ms * cfg.tide_ellipticity * np.sin(omega * hours)

    steady_u = (u_bg + u_eddy + u_jet)[None, :, :]
    steady_v = (v_bg + v_eddy)[None, :, :]
    return (
        (steady_u + u_tide[:, None, None]).astype("float32"),
        (steady_v + v_tide[:, None, None]).astype("float32"),
    )


def wind_field(
    lon: np.ndarray, lat: np.ndarray, hours: np.ndarray, aoi: BoundingBox, cfg: WindConfig
) -> tuple[np.ndarray, np.ndarray]:
    """10 m wind (u10, v10) in m/s, shaped (time, lat, lon).

    Convention: for wind blowing FROM bearing theta, the vector components point
    to where it is going, so u = -speed*sin(theta), v = -speed*cos(theta).
    """
    ref_lon, ref_lat = aoi.centroid
    lon2d, lat2d = np.meshgrid(lon, lat)

    calm_lon = aoi.min_lon + cfg.calm_offset_lon_frac * (aoi.max_lon - aoi.min_lon)
    calm_lat = aoi.min_lat + cfg.calm_offset_lat_frac * (aoi.max_lat - aoi.min_lat)
    cx_km, cy_km = km_offsets(lon2d, lat2d, calm_lon, calm_lat)
    dip = np.exp(-(cx_km**2 + cy_km**2) / (2.0 * cfg.calm_radius_km**2))

    # Interpolate between the base speed and the calm floor.
    speed = cfg.base_speed_ms - (cfg.base_speed_ms - cfg.calm_min_speed_ms) * dip

    # Mild large-scale gradient so the field is not piecewise-uniform.
    _, dy_km = km_offsets(lon2d, lat2d, ref_lon, ref_lat)
    speed = speed * (1.0 + 0.05 * dy_km / 100.0)
    speed = np.clip(speed, 0.2, None)

    theta = np.deg2rad(cfg.base_from_direction_deg + cfg.rotation_deg_per_day * hours / 24.0)
    u = -speed[None, :, :] * np.sin(theta)[:, None, None]
    v = -speed[None, :, :] * np.cos(theta)[:, None, None]
    return u.astype("float32"), v.astype("float32")


def land_mask(lon: np.ndarray, lat: np.ndarray, aoi: BoundingBox, cfg: LandConfig) -> np.ndarray:
    """1 where land, 0 where sea. Shaped (lat, lon)."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    span_lat = aoi.max_lat - aoi.min_lat
    frac_lon = (lon2d - aoi.min_lon) / (aoi.max_lon - aoi.min_lon)
    coast = aoi.min_lat + span_lat * (
        cfg.coast_lat_frac
        + cfg.wiggle_amplitude_frac * np.sin(2.0 * np.pi * cfg.wiggle_cycles * frac_lon)
    )
    return (lat2d > coast).astype("uint8")


# --------------------------------------------------------------------- write


def write_forcing(
    out_dir: Path,
    aoi: BoundingBox,
    t_start: datetime,
    t_obs: datetime,
    cfg: SyntheticForcingConfig | None = None,
) -> dict[str, Path]:
    """Write currents.nc, wind.nc and land.tif into ``out_dir``.

    Returns the three paths. File names and variable names match what the real
    CMEMS and ERA5 subsets will use, so Phase 8 is a swap of the producer with
    no change to any consumer.
    """
    import rasterio
    from rasterio.transform import from_origin

    cfg = cfg or SyntheticForcingConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    times = _time_axis(t_start, t_obs, cfg.grid.time_step_hours)
    hours = _hours_since(times, t_start)
    time_axis = as_naive_utc(times)

    # --- currents -----------------------------------------------------------
    step = cfg.grid.currents_resolution_deg
    clon = _axis(aoi.min_lon, aoi.max_lon, step)
    clat = _axis(aoi.min_lat, aoi.max_lat, step)
    uo, vo = current_field(clon, clat, hours, aoi, cfg.ocean)

    currents = xr.Dataset(
        {
            "uo": (("time", "latitude", "longitude"), uo, {"units": "m s-1",
                    "standard_name": "eastward_sea_water_velocity"}),
            "vo": (("time", "latitude", "longitude"), vo, {"units": "m s-1",
                    "standard_name": "northward_sea_water_velocity"}),
        },
        coords={"time": time_axis, "latitude": clat, "longitude": clon},
        attrs={
            "title": "Synthetic surface currents (Phase 1 stand-in for CMEMS SMOC)",
            "source": "src/ingest/synthetic_forcing.py",
            "synthetic": "true",
            "includes_tides": "true",
            "includes_stokes": "true",
            "comment": (
                "Currents, tides and Stokes drift are merged into uo/vo, matching "
                "CMEMS SMOC. Consumers must NOT add a separate Stokes term."
            ),
        },
    )
    currents_path = out_dir / "currents.nc"
    currents.to_netcdf(currents_path)

    # --- wind ---------------------------------------------------------------
    step = cfg.grid.wind_resolution_deg
    wlon = _axis(aoi.min_lon - step, aoi.max_lon + step, step)
    wlat = _axis(aoi.min_lat - step, aoi.max_lat + step, step)
    u10, v10 = wind_field(wlon, wlat, hours, aoi, cfg.wind)

    wind = xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), u10, {"units": "m s-1",
                     "standard_name": "eastward_wind"}),
            "v10": (("time", "latitude", "longitude"), v10, {"units": "m s-1",
                     "standard_name": "northward_wind"}),
        },
        coords={"time": time_axis, "latitude": wlat, "longitude": wlon},
        attrs={
            "title": "Synthetic 10 m wind (Phase 1 stand-in for ERA5)",
            "source": "src/ingest/synthetic_forcing.py",
            "synthetic": "true",
            "comment": (
                "Components only. For wind FROM bearing theta, "
                "u = -speed*sin(theta), v = -speed*cos(theta)."
            ),
        },
    )
    wind_path = out_dir / "wind.nc"
    wind.to_netcdf(wind_path)

    # --- land ---------------------------------------------------------------
    step = cfg.land.resolution_deg
    llon = _axis(aoi.min_lon, aoi.max_lon, step)
    llat = _axis(aoi.min_lat, aoi.max_lat, step)
    mask = land_mask(llon, llat, aoi, cfg.land)

    land_path = out_dir / "land.tif"
    # GeoTIFF rows run north to south, so flip the latitude axis.
    with rasterio.open(
        land_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(llon[0] - step / 2, llat[-1] + step / 2, step, step),
        compress="deflate",
    ) as dst:
        dst.write(np.flipud(mask), 1)
        dst.update_tags(synthetic="true", description="1 = land, 0 = sea")

    return {"currents": currents_path, "wind": wind_path, "land": land_path}


__all__ = [
    "OceanConfig",
    "WindConfig",
    "GridSpec",
    "LandConfig",
    "SyntheticForcingConfig",
    "current_field",
    "wind_field",
    "land_mask",
    "km_offsets",
    "write_forcing",
    "M2_PERIOD_HOURS",
]
