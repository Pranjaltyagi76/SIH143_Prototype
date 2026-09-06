"""Synthetic AIS traffic.

Phase 1. Stands in for Danish Maritime Authority CSV so the attribution engine
can be built before any download, and it is what the Arabian Sea case uses in
the demo -- the problem statement explicitly permits synthetic AIS where real
data is unavailable, and we disclose it on the slide.

The traffic is not random noise. It is shaped so that every signal the
attribution engine claims to use actually exists in the data:

- **lanes** give realistic clustering, so the spatio-temporal prefilter has to
  earn its reduction factor against genuine traffic density rather than against
  a uniform scatter
- **AIS gaps** occur at a realistic base rate, so the gap prior has a *baseline*
  to be relative to. Gaps are overwhelmingly benign; a prior that treats them
  absolutely manufactures suspicion out of poor receiver coverage (W-07)
- **loiterers** slow well below their transit speed, so the speed-anomaly factor
  has something to find that is not the culprit

MMSI safety
-----------
Every generated MMSI begins with ``00``. Ship-station MMSIs carry a Maritime
Identification Digit prefix in 201-775, so a leading zero cannot collide with a
real vessel. This is watch-list item W-14, and it is asserted in the tests: a
real ship appearing in an accusation demo would be a serious problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import BoundingBox
from src.ingest.synthetic_forcing import EARTH_RADIUS_M

KNOTS_TO_MS = 0.514444

# Ship-station MMSIs use a Maritime Identification Digit prefix in this range.
# Our synthetic identifiers sit deliberately outside it.
REAL_MID_MIN, REAL_MID_MAX = 201, 775
SYNTHETIC_MMSI_PREFIX = "00"


@dataclass(frozen=True)
class VesselType:
    name: str
    code: int
    speed_kn: tuple[float, float]
    length_m: tuple[float, float]
    weight: float


VESSEL_TYPES: tuple[VesselType, ...] = (
    VesselType("Tanker", 80, (10.0, 14.5), (180.0, 330.0), 0.18),
    VesselType("Cargo", 70, (11.0, 16.0), (90.0, 230.0), 0.32),
    VesselType("Bulk Carrier", 79, (10.0, 14.0), (150.0, 290.0), 0.12),
    VesselType("Fishing", 30, (3.0, 9.0), (15.0, 45.0), 0.20),
    VesselType("Passenger", 60, (14.0, 22.0), (60.0, 200.0), 0.08),
    VesselType("Tug", 52, (6.0, 12.0), (20.0, 40.0), 0.10),
)


@dataclass(frozen=True)
class AISConfig:
    n_vessels: int = 180

    # Shipping lanes as (start, end) fractional AOI coordinates. Traffic
    # concentrates along these, as it does in any real sea area.
    lanes: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
        ((-0.05, 0.22), (1.05, 0.48)),
        ((0.18, -0.05), (0.62, 1.05)),
        ((-0.05, 0.80), (1.05, 0.62)),
    )
    lane_fraction: float = 0.72
    lane_width_km: float = 9.0

    report_interval_s: float = 120.0
    report_jitter_s: float = 45.0
    position_noise_m: float = 25.0

    # Base rate of AIS gaps. These are benign: receiver coverage, not
    # concealment. Their purpose is to give the gap prior a baseline.
    gap_probability: float = 0.22
    gap_duration_hours: tuple[float, float] = (0.4, 3.5)

    # Vessels that slow well below their transit speed, for benign reasons.
    loiter_probability: float = 0.12
    loiter_speed_factor: tuple[float, float] = (0.25, 0.5)

    types: tuple[VesselType, ...] = field(default_factory=lambda: VESSEL_TYPES)


def synthetic_mmsi(index: int) -> str:
    """A 9-digit identifier that cannot be a real ship station."""
    return f"{SYNTHETIC_MMSI_PREFIX}{index:07d}"


def is_synthetic_mmsi(mmsi: str) -> bool:
    """True only if this identifier is provably not a real ship station."""
    if len(mmsi) != 9 or not mmsi.isdigit():
        return False
    return not (REAL_MID_MIN <= int(mmsi[:3]) <= REAL_MID_MAX)


def km_to_deg(dx_km: float | np.ndarray, dy_km: float | np.ndarray, ref_lat: float):
    m_per_deg_lat = EARTH_RADIUS_M * np.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(ref_lat))
    return dx_km * 1000.0 / m_per_deg_lon, dy_km * 1000.0 / m_per_deg_lat


def _frac_to_lonlat(frac: tuple[float, float], aoi: BoundingBox) -> tuple[float, float]:
    fx, fy = frac
    return (
        aoi.min_lon + fx * (aoi.max_lon - aoi.min_lon),
        aoi.min_lat + fy * (aoi.max_lat - aoi.min_lat),
    )


def _bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Course over ground, degrees clockwise from north."""
    dlon = np.deg2rad(lon2 - lon1)
    y = np.sin(dlon) * np.cos(np.deg2rad(lat2))
    x = np.cos(np.deg2rad(lat1)) * np.sin(np.deg2rad(lat2)) - np.sin(
        np.deg2rad(lat1)
    ) * np.cos(np.deg2rad(lat2)) * np.cos(dlon)
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def _pick_type(rng: np.random.Generator, cfg: AISConfig) -> VesselType:
    weights = np.array([t.weight for t in cfg.types], dtype="float64")
    return cfg.types[int(rng.choice(len(cfg.types), p=weights / weights.sum()))]


def _one_vessel(
    rng: np.random.Generator,
    index: int,
    aoi: BoundingBox,
    t_start: datetime,
    t_obs: datetime,
    cfg: AISConfig,
) -> list[dict]:
    vtype = _pick_type(rng, cfg)
    mmsi = synthetic_mmsi(index)
    ref_lat = aoi.centroid[1]

    # Route: mostly along a lane, otherwise a random crossing.
    if rng.random() < cfg.lane_fraction:
        lane = cfg.lanes[int(rng.integers(len(cfg.lanes)))]
        a, b = _frac_to_lonlat(lane[0], aoi), _frac_to_lonlat(lane[1], aoi)
        offset_km = rng.normal(0.0, cfg.lane_width_km / 3.0)
    else:
        a = _frac_to_lonlat((rng.uniform(-0.05, 1.05), rng.uniform(-0.05, 1.05)), aoi)
        b = _frac_to_lonlat((rng.uniform(-0.05, 1.05), rng.uniform(-0.05, 1.05)), aoi)
        offset_km = 0.0

    if rng.random() < 0.5:
        a, b = b, a

    # Displace perpendicular to the route by the lane offset.
    heading = _bearing_deg(a[0], a[1], b[0], b[1])
    perp = np.deg2rad(heading + 90.0)
    d_lon, d_lat = km_to_deg(offset_km * np.sin(perp), offset_km * np.cos(perp), ref_lat)
    a = (a[0] + d_lon, a[1] + d_lat)
    b = (b[0] + d_lon, b[1] + d_lat)

    speed_kn = float(rng.uniform(*vtype.speed_kn))
    loitering = rng.random() < cfg.loiter_probability
    if loitering:
        speed_kn *= float(rng.uniform(*cfg.loiter_speed_factor))

    # Route length, then transit duration.
    dx_km = (b[0] - a[0]) * np.cos(np.deg2rad(ref_lat)) * EARTH_RADIUS_M * np.pi / 180.0 / 1000.0
    dy_km = (b[1] - a[1]) * EARTH_RADIUS_M * np.pi / 180.0 / 1000.0
    route_km = float(np.hypot(dx_km, dy_km))
    if route_km < 1.0:
        return []
    transit_s = route_km * 1000.0 / max(speed_kn * KNOTS_TO_MS, 0.1)

    window_s = (t_obs - t_start).total_seconds()
    t_enter = t_start + timedelta(seconds=float(rng.uniform(-transit_s * 0.5, window_s)))

    # An AIS gap: benign, and the reason the gap prior must be baseline-relative.
    gap_start = gap_end = None
    if rng.random() < cfg.gap_probability:
        gap_hours = float(rng.uniform(*cfg.gap_duration_hours))
        gap_start = t_enter + timedelta(seconds=float(rng.uniform(0, max(transit_s - 60, 60))))
        gap_end = gap_start + timedelta(hours=gap_hours)

    rows: list[dict] = []
    elapsed = 0.0
    while elapsed <= transit_s:
        stamp = t_enter + timedelta(seconds=elapsed)
        elapsed += max(15.0, float(rng.normal(cfg.report_interval_s, cfg.report_jitter_s)))

        if stamp < t_start or stamp > t_obs:
            continue
        if gap_start is not None and gap_start <= stamp <= gap_end:
            continue

        f = (stamp - t_enter).total_seconds() / transit_s
        lon = a[0] + f * (b[0] - a[0])
        lat = a[1] + f * (b[1] - a[1])

        nx, ny = rng.normal(0.0, cfg.position_noise_m, 2) / 1000.0
        n_lon, n_lat = km_to_deg(nx, ny, ref_lat)
        lon, lat = lon + n_lon, lat + n_lat

        if not (aoi.min_lon <= lon <= aoi.max_lon and aoi.min_lat <= lat <= aoi.max_lat):
            continue

        cog = _bearing_deg(a[0], a[1], b[0], b[1])
        rows.append(
            {
                "timestamp": stamp,
                "mmsi": mmsi,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "sog_kn": round(speed_kn + float(rng.normal(0.0, 0.25)), 2),
                "cog_deg": round((cog + float(rng.normal(0.0, 2.0))) % 360.0, 1),
                "heading_deg": round(cog % 360.0, 0),
                "nav_status": "Under way using engine",
                "ship_name": f"SYNTH-{index:04d}",
                "ship_type": vtype.name,
                "ship_type_code": vtype.code,
                "length_m": round(float(rng.uniform(*vtype.length_m)), 1),
            }
        )
    return rows


def generate_ais(
    aoi: BoundingBox,
    t_start: datetime,
    t_obs: datetime,
    seed: int,
    cfg: AISConfig | None = None,
) -> pd.DataFrame:
    """Generate an AIS message log for the AOI and time window.

    Deterministic in ``seed`` (NFR-3).
    """
    cfg = cfg or AISConfig()
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for i in range(1, cfg.n_vessels + 1):
        rows.extend(_one_vessel(rng, i, aoi, t_start, t_obs, cfg))

    if not rows:
        raise RuntimeError("generated no AIS messages; check AOI and time window")

    df = pd.DataFrame(rows).sort_values(["timestamp", "mmsi"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def write_ais(
    out_dir: Path,
    aoi: BoundingBox,
    t_start: datetime,
    t_obs: datetime,
    seed: int,
    cfg: AISConfig | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Write ``tracks.parquet``. Parquet, never CSV, so the run path never parses text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_ais(aoi, t_start, t_obs, seed, cfg)
    path = out_dir / "tracks.parquet"
    df.to_parquet(path, index=False)
    return path, df


__all__ = [
    "AISConfig",
    "VesselType",
    "VESSEL_TYPES",
    "generate_ais",
    "write_ais",
    "synthetic_mmsi",
    "is_synthetic_mmsi",
    "km_to_deg",
    "KNOTS_TO_MS",
]
