"""Forcing fields, resampled once onto a projected grid and held in memory.

Two decisions in this module shape everything downstream.

**Everything is resampled to UTM metres at load time.**
The particle loop then never touches a projection, never converts degrees to
metres, and never opens a NetCDF file. Computing distance or area in degrees is
watch-list item W-11 -- it produces values wrong by a latitude-dependent factor,
silently -- and re-reading NetCDF inside the integration loop is the single
easiest way to turn a 10-second run into a 10-minute one.

**Velocity vectors are rotated into the grid frame.**
``uo``/``vo`` are eastward/northward in the geographic frame. Grid north in a
UTM projection is not true north; the difference is the meridian convergence,
up to about 3 degrees at a zone edge. Rather than derive the rotation
analytically and risk a sign error, we measure it numerically from the
projection itself: project each cell centre and a point one metre due
geographic north of it, and read off the resulting grid direction. Unambiguous,
computed once, and covered by a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.contracts import ForcingBundle

# Resolution of the in-memory projected grids, metres.
# Velocity is resampled from a ~9 km source, so 2 km is comfortably finer than
# the information content and costs only a few MB. Land needs to be finer,
# because beaching resolution is the resolution of the coastline.
VELOCITY_GRID_M = 2_000.0
LAND_GRID_M = 500.0

# Margin added around the AOI so particles leaving the area still find valid
# forcing for a step or two before being marked as exited.
GRID_MARGIN_M = 20_000.0

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def to_epoch_seconds(t: datetime) -> float:
    return (t - EPOCH).total_seconds()


@dataclass
class ForcingField:
    """Currents, wind and land on a common projected grid, fully in memory.

    Velocities are in the grid frame (metres per second along +x and +y of the
    projected CRS), so the integrator is pure arithmetic.
    """

    crs: str
    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int

    times: np.ndarray  # (nt,) epoch seconds, uniform
    u: np.ndarray  # (nt, ny, nx) float32, grid frame
    v: np.ndarray
    uw: np.ndarray  # (nt, ny, nx) float32, 10 m wind, grid frame
    vw: np.ndarray

    land_x0: float
    land_y0: float
    land_dx: float
    land_dy: float
    land: np.ndarray  # (lny, lnx) bool, True = land

    includes_stokes: bool
    includes_tides: bool
    rms_error_ms: float

    _to_lonlat: object = None
    _to_grid: object = None

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_case(
        cls,
        case_dir: Path | str,
        bundle: ForcingBundle,
        velocity_grid_m: float = VELOCITY_GRID_M,
        land_grid_m: float = LAND_GRID_M,
    ) -> ForcingField:
        import rasterio
        import xarray as xr
        from pyproj import Transformer
        from scipy.interpolate import RegularGridInterpolator

        case_dir = Path(case_dir)
        to_grid = Transformer.from_crs("EPSG:4326", bundle.crs_working, always_xy=True)
        to_lonlat = Transformer.from_crs(bundle.crs_working, "EPSG:4326", always_xy=True)

        # Projected bounding box of the AOI, plus margin.
        corners_lon = [bundle.aoi.min_lon, bundle.aoi.max_lon] * 2
        corners_lat = [bundle.aoi.min_lat] * 2 + [bundle.aoi.max_lat] * 2
        cx, cy = to_grid.transform(corners_lon, corners_lat)
        x0 = min(cx) - GRID_MARGIN_M
        y0 = min(cy) - GRID_MARGIN_M
        nx = int((max(cx) + GRID_MARGIN_M - x0) / velocity_grid_m) + 1
        ny = int((max(cy) + GRID_MARGIN_M - y0) / velocity_grid_m) + 1

        gx = x0 + velocity_grid_m * np.arange(nx)
        gy = y0 + velocity_grid_m * np.arange(ny)
        gxx, gyy = np.meshgrid(gx, gy)
        glon, glat = to_lonlat.transform(gxx.ravel(), gyy.ravel())
        sample_pts = np.column_stack([glat, glon])  # source grids are (lat, lon)

        rotation = cls._grid_north_rotation(gxx, gyy, to_grid, to_lonlat)
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)

        def resample(ds: xr.Dataset, uname: str, vname: str) -> tuple[np.ndarray, np.ndarray]:
            lat = ds["latitude"].values.astype("float64")
            lon = ds["longitude"].values.astype("float64")
            nt = ds.sizes["time"]
            ug = np.empty((nt, ny, nx), dtype="float32")
            vg = np.empty((nt, ny, nx), dtype="float32")
            for i in range(nt):
                ue = RegularGridInterpolator(
                    (lat, lon), ds[uname].values[i], bounds_error=False, fill_value=None
                )(sample_pts).reshape(ny, nx)
                vn = RegularGridInterpolator(
                    (lat, lon), ds[vname].values[i], bounds_error=False, fill_value=None
                )(sample_pts).reshape(ny, nx)
                # Geographic (east, north) -> grid (+x, +y).
                ug[i] = (ue * cos_r - vn * sin_r).astype("float32")
                vg[i] = (ue * sin_r + vn * cos_r).astype("float32")
            return ug, vg

        with xr.open_dataset(case_dir / bundle.currents.path) as ds:
            # NetCDF stores naive timestamps; they are UTC by construction
            # (require_utc at the contract boundary). Re-attaching UTC here
            # rather than assuming local time is watch-list item W-04.
            times = (
                ds["time"].values.astype("datetime64[s]").astype("int64").astype("float64")
            )
            u, v = resample(ds, bundle.currents.vars[0], bundle.currents.vars[1])

        with xr.open_dataset(case_dir / bundle.wind.path) as ds:
            wind_times = (
                ds["time"].values.astype("datetime64[s]").astype("int64").astype("float64")
            )
            if not np.array_equal(wind_times, times):
                raise ValueError(
                    "wind and current time axes differ; the kernel assumes a shared axis"
                )
            uw, vw = resample(ds, bundle.wind.vars[0], bundle.wind.vars[1])

        # Land, on its own finer grid.
        lnx = int((max(cx) + GRID_MARGIN_M - x0) / land_grid_m) + 1
        lny = int((max(cy) + GRID_MARGIN_M - y0) / land_grid_m) + 1
        lx = x0 + land_grid_m * np.arange(lnx)
        ly = y0 + land_grid_m * np.arange(lny)
        lxx, lyy = np.meshgrid(lx, ly)
        llon, llat = to_lonlat.transform(lxx.ravel(), lyy.ravel())

        with rasterio.open(case_dir / bundle.land_mask) as src:
            raster = src.read(1)
            # rasterio's src.index() is scalar-only here, so apply the inverse
            # affine directly -- it broadcasts over arrays.
            inv = ~src.transform
            cols, rows = inv * (np.asarray(llon), np.asarray(llat))
            rows = np.clip(rows.astype(np.intp), 0, raster.shape[0] - 1)
            cols = np.clip(cols.astype(np.intp), 0, raster.shape[1] - 1)
            land = raster[rows, cols].reshape(lny, lnx).astype(bool)

        return cls(
            crs=bundle.crs_working,
            x0=x0, y0=y0, dx=velocity_grid_m, dy=velocity_grid_m, nx=nx, ny=ny,
            times=times, u=u, v=v, uw=uw, vw=vw,
            land_x0=x0, land_y0=y0, land_dx=land_grid_m, land_dy=land_grid_m, land=land,
            includes_stokes=bundle.currents.includes_stokes,
            includes_tides=bundle.currents.includes_tides,
            rms_error_ms=bundle.currents.rms_error_ms,
            _to_lonlat=to_lonlat,
            _to_grid=to_grid,
        )

    @staticmethod
    def _grid_north_rotation(gxx, gyy, to_grid, to_lonlat) -> np.ndarray:
        """Angle to rotate geographic (east, north) into grid (+x, +y).

        Measured, not derived: project each cell centre and a point 1 m due
        geographic north, then read the grid-frame direction of that step.
        """
        lon, lat = to_lonlat.transform(gxx.ravel(), gyy.ravel())
        dlat = 1.0 / 111_320.0  # ~1 metre of latitude, in degrees
        nx_, ny_ = to_grid.transform(lon, np.asarray(lat) + dlat)
        step_x = np.asarray(nx_) - gxx.ravel()
        step_y = np.asarray(ny_) - gyy.ravel()
        # True north points along (step_x, step_y) in the grid frame. Rotating
        # geographic north (0, 1) onto that direction is a rotation by this angle.
        return np.arctan2(step_x, step_y).reshape(gxx.shape)

    # ------------------------------------------------------------ conversion

    def to_lonlat(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self._to_lonlat.transform(x, y)
        return np.asarray(lon), np.asarray(lat)

    def to_grid(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._to_grid.transform(lon, lat)
        return np.asarray(x), np.asarray(y)

    @property
    def t_min(self) -> float:
        return float(self.times[0])

    @property
    def t_max(self) -> float:
        return float(self.times[-1])

    def covers(self, t: float) -> bool:
        return self.t_min <= t <= self.t_max

    # --------------------------------------------------------- interpolation

    def _time_slice(self, arr: np.ndarray, t: float) -> np.ndarray:
        """Linearly interpolate a (nt, ny, nx) stack to one 2-D field at time t.

        Done once per velocity evaluation rather than per particle: the grid has
        a few thousand cells and there are 1e5 particles, so collapsing time
        first and then sampling space is far cheaper than the reverse.
        """
        idx = np.searchsorted(self.times, t) - 1
        idx = int(np.clip(idx, 0, len(self.times) - 2))
        t0, t1 = self.times[idx], self.times[idx + 1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        w = float(np.clip(w, 0.0, 1.0))
        return arr[idx] * (1.0 - w) + arr[idx + 1] * w

    def _bilinear(self, field2d: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        fx = np.clip((x - self.x0) / self.dx, 0.0, self.nx - 1.001)
        fy = np.clip((y - self.y0) / self.dy, 0.0, self.ny - 1.001)
        ix = fx.astype(np.intp)
        iy = fy.astype(np.intp)
        wx = fx - ix
        wy = fy - iy
        return (
            field2d[iy, ix] * (1.0 - wx) * (1.0 - wy)
            + field2d[iy, ix + 1] * wx * (1.0 - wy)
            + field2d[iy + 1, ix] * (1.0 - wx) * wy
            + field2d[iy + 1, ix + 1] * wx * wy
        )

    def current(self, x: np.ndarray, y: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Surface current in the grid frame, m/s.

        This already contains tides and Stokes drift when the source product
        merges them (CMEMS SMOC does). Callers must not add a separate Stokes
        term -- see ``includes_stokes`` and P-05.
        """
        return (
            self._bilinear(self._time_slice(self.u, t), x, y),
            self._bilinear(self._time_slice(self.v, t), x, y),
        )

    def wind(self, x: np.ndarray, y: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._bilinear(self._time_slice(self.uw, t), x, y),
            self._bilinear(self._time_slice(self.vw, t), x, y),
        )

    def wind_speed(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        uw, vw = self.wind(x, y, t)
        return np.hypot(uw, vw)

    def is_land(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = np.clip(((x - self.land_x0) / self.land_dx).astype(np.intp), 0, self.land.shape[1] - 1)
        iy = np.clip(((y - self.land_y0) / self.land_dy).astype(np.intp), 0, self.land.shape[0] - 1)
        return self.land[iy, ix]

    def in_domain(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (
            (x >= self.x0)
            & (x <= self.x0 + self.dx * (self.nx - 1))
            & (y >= self.y0)
            & (y <= self.y0 + self.dy * (self.ny - 1))
        )

    @property
    def memory_mb(self) -> float:
        arrays = (self.u, self.v, self.uw, self.vw, self.land)
        return sum(a.nbytes for a in arrays) / 1e6


__all__ = [
    "ForcingField",
    "to_epoch_seconds",
    "VELOCITY_GRID_M",
    "LAND_GRID_M",
    "GRID_MARGIN_M",
    "EPOCH",
]
