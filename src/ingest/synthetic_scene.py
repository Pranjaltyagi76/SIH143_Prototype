"""A synthetic Sentinel-1 style SAR scene.

Phase 1. Produces calibrated sigma-0 VV in dB plus a per-pixel incidence angle,
standing in for the Zenodo tiles until Phase 8.

The backscatter is generated from the same wind field the physics gate reads,
using a CMOD-like proxy:

    sigma0_dB = a + b*log10(wind) - c*(incidence - 35)

That has a consequence worth stating plainly: **the calm pocket in the wind
field shows up in the scene as a genuine dark patch, with no oil in it.** It is
not painted on. It is a real low-wind look-alike, arising from the same physics
that makes look-alike rejection hard in the first place -- below about 3 m/s the
sea surface is already smooth, so everything is dark and no oil-water contrast
is possible.

That gives the detector something honest to be fooled by, and gives the physics
gate something honest to reject. Phase 7 injects actual oil slicks on top.

Speckle is multiplicative (Gamma-distributed intensity), not additive Gaussian.
SAR speckle is multiplicative, and getting that wrong is a detail a remote
sensing scientist will notice immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.contracts import BoundingBox
from src.ingest.synthetic_forcing import SyntheticForcingConfig, land_mask, wind_field

# Sentinel-1 IW GRD is multi-looked to roughly this many equivalent looks.
EQUIVALENT_NUMBER_OF_LOOKS = 4.4


@dataclass(frozen=True)
class SceneConfig:
    """Scene raster and radiometry settings."""

    # 100 m rather than the real 10 m. A 10 m raster over a degree-scale AOI is
    # hundreds of millions of pixels, which buys nothing for algorithm
    # development. Phase 8 uses real 10 m tiles over a much smaller footprint.
    pixel_spacing_m: float = 100.0

    incidence_near_deg: float = 30.5
    incidence_far_deg: float = 45.5

    # sigma0_dB = base + wind_gain*log10(wind) - incidence_slope*(inc - ref)
    base_db: float = -21.0
    wind_gain_db: float = 8.0
    incidence_slope_db_per_deg: float = 0.20
    incidence_ref_deg: float = 35.0

    land_sigma0_db: float = -7.5
    looks: float = EQUIVALENT_NUMBER_OF_LOOKS


def _raster_axes(aoi: BoundingBox, pixel_spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude axes at approximately the requested spacing."""
    from src.ingest.synthetic_forcing import EARTH_RADIUS_M

    m_per_deg_lat = EARTH_RADIUS_M * np.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(aoi.centroid[1]))

    step_lat = pixel_spacing_m / m_per_deg_lat
    step_lon = pixel_spacing_m / m_per_deg_lon

    n_lon = int((aoi.max_lon - aoi.min_lon) / step_lon)
    n_lat = int((aoi.max_lat - aoi.min_lat) / step_lat)
    lon = aoi.min_lon + step_lon * np.arange(n_lon)
    lat = aoi.min_lat + step_lat * np.arange(n_lat)
    return lon, lat


def incidence_grid(lon: np.ndarray, lat: np.ndarray, cfg: SceneConfig) -> np.ndarray:
    """Incidence angle ramp across the range direction (west to east).

    Backscatter falls steeply across the swath. Normalising the damping ratio
    for this is what makes near-range and far-range patches comparable; without
    it a classifier learns swath position rather than oil.
    """
    ramp = np.linspace(cfg.incidence_near_deg, cfg.incidence_far_deg, lon.size)
    return np.broadcast_to(ramp[None, :], (lat.size, lon.size)).astype("float32")


def sigma0_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    wind_speed_ms: np.ndarray,
    incidence_deg: np.ndarray,
    land: np.ndarray,
    rng: np.random.Generator,
    cfg: SceneConfig,
) -> np.ndarray:
    """Calibrated sigma-0 VV in dB, with multiplicative speckle."""
    clean_db = (
        cfg.base_db
        + cfg.wind_gain_db * np.log10(np.clip(wind_speed_ms, 0.4, None))
        - cfg.incidence_slope_db_per_deg * (incidence_deg - cfg.incidence_ref_deg)
    )
    clean_db = np.where(land.astype(bool), cfg.land_sigma0_db, clean_db)

    # Multiplicative speckle: multi-look intensity is Gamma(L, 1/L), mean 1.
    linear = 10.0 ** (clean_db / 10.0)
    speckle = rng.gamma(shape=cfg.looks, scale=1.0 / cfg.looks, size=clean_db.shape)
    return (10.0 * np.log10(np.clip(linear * speckle, 1e-12, None))).astype("float32")


def write_scene(
    out_dir: Path,
    aoi: BoundingBox,
    t_obs: datetime,
    t_start: datetime,
    seed: int,
    forcing_cfg: SyntheticForcingConfig | None = None,
    cfg: SceneConfig | None = None,
) -> dict[str, Path]:
    """Write ``sigma0_vv_db.tif`` and ``incidence.tif`` for the acquisition time."""
    import rasterio
    from rasterio.transform import from_origin

    forcing_cfg = forcing_cfg or SyntheticForcingConfig()
    cfg = cfg or SceneConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    lon, lat = _raster_axes(aoi, cfg.pixel_spacing_m)

    # Wind at the acquisition instant, on the scene grid.
    hours = np.array([(t_obs - t_start).total_seconds() / 3600.0])
    u10, v10 = wind_field(lon, lat, hours, aoi, forcing_cfg.wind)
    wind_speed = np.hypot(u10[0], v10[0])

    land = land_mask(lon, lat, aoi, forcing_cfg.land)
    incidence = incidence_grid(lon, lat, cfg)
    sigma0 = sigma0_grid(lon, lat, wind_speed, incidence, land, rng, cfg)

    step_lon = float(lon[1] - lon[0])
    step_lat = float(lat[1] - lat[0])
    transform = from_origin(lon[0] - step_lon / 2, lat[-1] + step_lat / 2, step_lon, step_lat)

    paths: dict[str, Path] = {}
    for name, data, tags in (
        (
            "sigma0_vv_db",
            sigma0,
            {
                "units": "dB",
                "polarisation": "VV",
                "synthetic": "true",
                "acquisition_utc": t_obs.isoformat(),
                "comment": (
                    "Backscatter derived from the same wind field the physics gate "
                    "reads. The low-wind pocket is a genuine look-alike, not painted on."
                ),
            },
        ),
        ("incidence", incidence, {"units": "degrees", "synthetic": "true"}),
    ):
        path = out_dir / f"{name}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            compress="deflate",
            predictor=3,
        ) as dst:
            # GeoTIFF rows run north to south.
            dst.write(np.flipud(data), 1)
            dst.update_tags(**tags)
        paths[name] = path

    return paths


__all__ = [
    "SceneConfig",
    "incidence_grid",
    "sigma0_grid",
    "write_scene",
    "EQUIVALENT_NUMBER_OF_LOOKS",
]
