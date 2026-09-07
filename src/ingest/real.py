"""Adapters for real data sources.

Everything here produces exactly the same Case folder layout the synthetic
generator produces, so nothing downstream changes when real data arrives. That
was the point of freezing the contracts in Phase 0.

    Copernicus Data Space   Sentinel-1 GRD IW VV  ->  scene/sigma0_vv_db.tif
    CMEMS SMOC              currents + tides + Stokes -> forcing/currents.nc
    ERA5                    10 m u/v              ->  forcing/wind.nc
    Danish Maritime Auth.   AIS CSV               ->  ais/tracks.parquet

Why normalisation is a module and not three lines
-------------------------------------------------
Real gridded products disagree with each other in ways that do not raise:

- **ERA5 latitude descends.** North to south, 90 to -90. Our interpolators
  require ascending axes; handed a descending one they return values from the
  wrong hemisphere-ward direction without complaint.
- **CMEMS carries a depth dimension** even for a surface product, so the array
  is 4-D where the kernel expects 3-D.
- **Coordinate names differ** -- ``latitude``/``longitude`` versus ``lat``/``lon``,
  and recent CDS deliveries use ``valid_time`` rather than ``time``.
- **Longitude conventions differ**: 0..360 in some products, -180..180 in others.

Each of those produces plausible, silently wrong trajectories. ``normalise``
fixes all four and asserts the result, so a real case either loads correctly or
fails loudly.

Status
------
Written against published format documentation. **Not yet exercised against live
downloads** -- no credentials are configured in this repository and the archives
run to ~96 GB. ``scripts/fetch_real_data.py`` performs the fetch; every
adapter here validates its input on load rather than trusting the format, so
the first real file will produce a clear error rather than a quiet wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Danish Maritime Authority AIS CSV. Column names as published; the loader
# matches case-insensitively and reports what it could not find rather than
# raising a bare KeyError.
DMA_COLUMNS = {
    "timestamp": "# Timestamp",
    "mmsi": "MMSI",
    "lat": "Latitude",
    "lon": "Longitude",
    "sog_kn": "SOG",
    "cog_deg": "COG",
    "heading_deg": "Heading",
    "nav_status": "Navigational status",
    "ship_name": "Name",
    "ship_type": "Ship type",
    "length_m": "Length",
}

# DMA timestamps are day-first and UTC.
DMA_TIME_FORMAT = "%d/%m/%Y %H:%M:%S"

# Candidate coordinate names across CMEMS, ERA5 and CDS deliveries.
LAT_NAMES = ("latitude", "lat", "nav_lat", "y")
LON_NAMES = ("longitude", "lon", "nav_lon", "x")
TIME_NAMES = ("time", "valid_time", "t")
DEPTH_NAMES = ("depth", "deptht", "lev", "level")


@dataclass
class IngestReport:
    """What an adapter actually found, so surprises surface immediately."""

    source: str
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict = None
    notes: list = None

    def __post_init__(self):
        self.dropped = self.dropped or {}
        self.notes = self.notes or []


# ------------------------------------------------------------------ gridded


def normalise(ds, *, lon_name: str | None = None):
    """Coerce a real gridded product into the layout the kernel assumes.

    Ascending latitude, ascending longitude in -180..180, a plain ``time``
    dimension, and no depth axis. Asserts the result.
    """
    import xarray as xr

    if not isinstance(ds, xr.Dataset):
        raise TypeError("normalise expects an xarray.Dataset")

    rename = {}
    for names, target in ((LAT_NAMES, "latitude"), (LON_NAMES, "longitude"),
                          (TIME_NAMES, "time")):
        for candidate in names:
            if candidate in ds.coords or candidate in ds.dims:
                if candidate != target:
                    rename[candidate] = target
                break
    if rename:
        ds = ds.rename(rename)

    for missing in ("latitude", "longitude", "time"):
        if missing not in ds.coords:
            raise ValueError(
                f"cannot find a '{missing}' coordinate; available: {list(ds.coords)}"
            )

    # A surface product still ships a depth axis. Take the shallowest level.
    for depth in DEPTH_NAMES:
        if depth in ds.dims:
            ds = ds.isel({depth: 0}, drop=True)

    # ERA5 latitude descends north to south. Interpolators need ascending axes,
    # and handed a descending one they do not complain -- they just interpolate
    # in the wrong direction.
    if ds["latitude"].size > 1 and float(ds["latitude"][0]) > float(ds["latitude"][-1]):
        ds = ds.isel(latitude=slice(None, None, -1))

    # Some products publish longitude in 0..360.
    lons = ds["longitude"].values
    if lons.size and float(np.nanmax(lons)) > 180.0:
        ds = ds.assign_coords(longitude=(((ds["longitude"] + 180.0) % 360.0) - 180.0))
        ds = ds.sortby("longitude")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    if lat.size > 1 and not np.all(np.diff(lat) > 0):
        raise ValueError("latitude is not strictly ascending after normalisation")
    if lon.size > 1 and not np.all(np.diff(lon) > 0):
        raise ValueError("longitude is not strictly ascending after normalisation")

    # NetCDF has no timezone concept; everything in this project is UTC.
    ds["time"] = ds["time"].values.astype("datetime64[s]")
    return ds


def subset(ds, aoi, t_start: datetime, t_obs: datetime, margin_deg: float = 0.5):
    """Cut a global product down to one case, with margin for drifting particles."""
    return ds.sel(
        latitude=slice(aoi.min_lat - margin_deg, aoi.max_lat + margin_deg),
        longitude=slice(aoi.min_lon - margin_deg, aoi.max_lon + margin_deg),
        time=slice(
            np.datetime64(t_start.replace(tzinfo=None) - timedelta(hours=2), "s"),
            np.datetime64(t_obs.replace(tzinfo=None) + timedelta(hours=2), "s"),
        ),
    )


def write_currents(ds, path: Path, u: str = "uo", v: str = "vo") -> IngestReport:
    """Write a normalised CMEMS subset as ``forcing/currents.nc``."""
    report = IngestReport(source="CMEMS")
    missing = [name for name in (u, v) if name not in ds.data_vars]
    if missing:
        raise ValueError(f"currents product has no {missing}; found {list(ds.data_vars)}")

    out = ds[[u, v]].rename({u: "uo", v: "vo"})
    nan_fraction = float(np.isnan(out["uo"].values).mean())
    if nan_fraction > 0.5:
        report.notes.append(
            f"{nan_fraction:.0%} of current cells are NaN -- mostly land, or the "
            f"AOI is outside the product domain"
        )
    # Land cells are NaN in ocean models. A NaN velocity silently freezes a
    # particle instead of beaching it, so fill and rely on the land mask.
    out = out.fillna(0.0)
    out.attrs["comment"] = (
        "Currents from CMEMS. Whether tides and Stokes drift are merged is a "
        "property of the product and is recorded in ForcingBundle, not here."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)
    report.rows_out = int(out["uo"].size)
    return report


def write_wind(ds, path: Path, u: str = "u10", v: str = "v10") -> IngestReport:
    """Write a normalised ERA5 subset as ``forcing/wind.nc``."""
    report = IngestReport(source="ERA5")
    missing = [name for name in (u, v) if name not in ds.data_vars]
    if missing:
        raise ValueError(f"wind product has no {missing}; found {list(ds.data_vars)}")

    out = ds[[u, v]].rename({u: "u10", v: "v10"})
    speed = np.hypot(out["u10"].values, out["v10"].values)
    if np.nanmax(speed) > 60.0:
        report.notes.append(
            f"peak wind {np.nanmax(speed):.0f} m/s is implausible; check units"
        )
    report.notes.append(
        f"wind {np.nanmin(speed):.1f}-{np.nanmax(speed):.1f} m/s; "
        f"{float((speed < 3.0).mean()):.0%} of the field is below the detectability gate"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)
    report.rows_out = int(out["u10"].size)
    return report


# ---------------------------------------------------------------------- AIS


def load_dma_ais(path: Path | str, aoi=None, t_start=None, t_obs=None):
    """Read a Danish Maritime Authority AIS CSV into our schema.

    Returns a DataFrame matching what ``src.attribution.load_ais`` expects, so
    cleaning and track reconstruction are shared with the synthetic path.
    """
    import pandas as pd

    path = Path(path)
    raw = pd.read_csv(path, low_memory=False)
    report = IngestReport(source="Danish Maritime Authority", rows_in=len(raw))

    lookup = {c.strip().lower(): c for c in raw.columns}
    resolved, missing = {}, []
    for target, published in DMA_COLUMNS.items():
        col = lookup.get(published.strip().lower())
        if col is None:
            missing.append(published)
        else:
            resolved[target] = col
    required = {"timestamp", "mmsi", "lat", "lon", "sog_kn", "cog_deg"}
    if not required <= set(resolved):
        raise ValueError(
            f"DMA CSV is missing required columns {sorted(missing)}; "
            f"found {list(raw.columns)[:12]}"
        )

    df = pd.DataFrame({target: raw[col] for target, col in resolved.items()})

    # DMA timestamps are day-first and UTC. Parsing them as month-first would
    # silently reorder a third of the year's fixes.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format=DMA_TIME_FORMAT, utc=True, errors="coerce"
    )
    bad_time = int(df["timestamp"].isna().sum())
    if bad_time:
        report.dropped["unparseable_timestamp"] = bad_time
        df = df[df["timestamp"].notna()]

    df["mmsi"] = df["mmsi"].astype("Int64").astype(str)
    for col in ("lat", "lon", "sog_kn", "cog_deg"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, default in (("heading_deg", 0.0), ("nav_status", "Unknown"),
                         ("ship_name", ""), ("ship_type", "Unknown"), ("length_m", 0.0)):
        if col not in df:
            df[col] = default
    df["ship_name"] = df["ship_name"].fillna("").astype(str)
    df["ship_type"] = df["ship_type"].fillna("Unknown").astype(str)
    df["heading_deg"] = pd.to_numeric(df.get("heading_deg"), errors="coerce").fillna(
        df["cog_deg"]
    )
    df["length_m"] = pd.to_numeric(df.get("length_m"), errors="coerce").fillna(0.0)

    before = len(df)
    df = df.dropna(subset=["lat", "lon", "sog_kn", "cog_deg"])
    report.dropped["incomplete_fix"] = before - len(df)

    if aoi is not None:
        before = len(df)
        df = df[
            df["lat"].between(aoi.min_lat, aoi.max_lat)
            & df["lon"].between(aoi.min_lon, aoi.max_lon)
        ]
        report.dropped["outside_aoi"] = before - len(df)
    if t_start is not None and t_obs is not None:
        before = len(df)
        df = df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_obs)]
        report.dropped["outside_window"] = before - len(df)

    report.rows_out = len(df)
    return df.reset_index(drop=True), report


# -------------------------------------------------------------------- scene


def write_scene_from_grd(src_path: Path | str, dst: Path, band: int = 1) -> IngestReport:
    """Copy a calibrated, geocoded Sentinel-1 GRD band into a Case folder.

    Expects sigma-0 **in dB** and a valid CRS. Both are checked: an
    ungeoreferenced scene cannot be drifted or correlated with AIS, and linear
    sigma-0 mistaken for dB would make every damping ratio meaningless.
    """
    import rasterio

    report = IngestReport(source="Sentinel-1 GRD")
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(
                f"{Path(src_path).name} has no CRS. An ungeoreferenced scene cannot "
                f"be drifted or correlated with AIS -- geocode it first (Ellipsoid "
                f"Correction is enough over open ocean; terrain correction is not "
                f"needed on a flat sea)."
            )
        data = src.read(band).astype("float32")
        finite = data[np.isfinite(data)]
        if finite.size:
            p1, p50, p99 = (float(np.percentile(finite, q)) for q in (1, 50, 99))
            # Linear sigma-0 over water sits in roughly 0.001-1.0, which falls
            # entirely inside any plausible dB range -- so a median test alone
            # cannot tell the two apart. What separates them is sign and spread:
            # sigma-0 in dB over water is negative and spans tens of dB, while
            # linear sigma-0 is positive and bounded well below 2.
            if p1 >= 0.0 and p99 < 2.0:
                raise ValueError(
                    f"band {band} spans {p1:.3f}..{p99:.3f}, all positive and small. "
                    f"That is linear sigma-0, not decibels. Convert with 10*log10 "
                    f"before ingest -- read as dB it would make every damping ratio "
                    f"meaningless."
                )
            if not (-60.0 < p50 < 10.0):
                raise ValueError(
                    f"band {band} median is {p50:.2f}, outside any plausible "
                    f"sigma-0 dB range."
                )
        profile = src.profile | {"count": 1, "dtype": "float32", "compress": "deflate"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst, "w", **profile) as out:
            out.write(data, 1)
            out.update_tags(units="dB", polarisation="VV", source=str(src_path))
        report.notes.append(f"{src.width}x{src.height} px, CRS {src.crs}")
    report.rows_out = data.size
    return report


__all__ = [
    "IngestReport",
    "normalise",
    "subset",
    "write_currents",
    "write_wind",
    "load_dma_ais",
    "write_scene_from_grd",
    "DMA_COLUMNS",
    "DMA_TIME_FORMAT",
]
