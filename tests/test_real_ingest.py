"""Phase 8 tests for the real-data adapters.

No live downloads happen here. Instead each test constructs a file in the
**published real format** -- ERA5 with descending latitude, CMEMS with a depth
axis, DMA AIS with day-first timestamps, a two-band Zenodo tile with no CRS --
and checks the adapter handles it.

That is deliberately not the same as verifying against the real bytes, and the
distinction is recorded rather than blurred: these tests prove the adapters
handle the format *as documented*. Whether the documentation matches the
archives is unknown until someone downloads 96 GB, and every adapter validates
its input on load so the first real file fails loudly rather than quietly.

The formats reproduced here are the ones that break silently:

- ERA5 latitude runs north to south. Interpolators need ascending axes and do
  not complain when handed a descending one.
- CMEMS ships a depth axis even for a surface product.
- DMA timestamps are day-first; parsing them month-first silently reorders a
  third of the year.
- Zenodo tiles carry no CRS, which decides what they can be used for at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.contracts import BoundingBox
from src.ingest.real import (
    DMA_COLUMNS,
    load_dma_ais,
    normalise,
    subset,
    write_currents,
    write_scene_from_grd,
    write_wind,
)
from src.ingest.zenodo import (
    EXPECTED_TILE_PX,
    VH_BAND,
    VV_BAND,
    TileClass,
    inspect_tile,
    load_tile,
    verify_dataset,
)

UTC = timezone.utc
AOI = BoundingBox(min_lon=10.2, min_lat=56.4, max_lon=12.8, max_lat=58.1)


def _times(n: int = 6):
    return np.array([np.datetime64("2024-03-10T00", "s") + np.timedelta64(i, "h")
                     for i in range(n)])


def era5_like() -> xr.Dataset:
    """ERA5 as delivered: latitude DESCENDING, coordinate called valid_time."""
    lat = np.arange(59.0, 55.9, -0.25)   # north to south, as ERA5 ships it
    lon = np.arange(9.0, 14.01, 0.25)
    t = _times()
    shape = (t.size, lat.size, lon.size)
    return xr.Dataset(
        {"u10": (("valid_time", "latitude", "longitude"), np.full(shape, 4.0, "float32")),
         "v10": (("valid_time", "latitude", "longitude"), np.full(shape, -3.0, "float32"))},
        coords={"valid_time": t, "latitude": lat, "longitude": lon},
    )


def cmems_like() -> xr.Dataset:
    """CMEMS as delivered: a depth axis even for a surface product."""
    lat = np.arange(56.0, 58.51, 1 / 12)
    lon = np.arange(10.0, 13.01, 1 / 12)
    t = _times()
    shape = (t.size, 1, lat.size, lon.size)
    return xr.Dataset(
        {"uo": (("time", "depth", "latitude", "longitude"), np.full(shape, 0.2, "float32")),
         "vo": (("time", "depth", "latitude", "longitude"), np.full(shape, 0.1, "float32"))},
        coords={"time": t, "depth": [0.494], "latitude": lat, "longitude": lon},
    )


# ------------------------------------------------------------- normalisation


def test_descending_latitude_is_flipped():
    """The failure this prevents is silent: an interpolator handed a descending
    axis returns values from the wrong direction without raising."""
    raw = era5_like()
    assert raw["latitude"][0] > raw["latitude"][-1], "fixture should start descending"
    out = normalise(raw)
    assert np.all(np.diff(out["latitude"].values) > 0)


def test_depth_axis_is_dropped():
    out = normalise(cmems_like())
    assert "depth" not in out.dims
    assert out["uo"].dims == ("time", "latitude", "longitude")


def test_valid_time_is_renamed_to_time():
    """Recent CDS deliveries use valid_time; the kernel expects time."""
    out = normalise(era5_like())
    assert "time" in out.coords and "valid_time" not in out.coords


def test_longitude_is_converted_from_0_360():
    lat = np.arange(56.0, 58.1, 0.5)
    lon = np.arange(350.0, 360.0, 0.5)  # 0..360 convention
    t = _times(3)
    ds = xr.Dataset(
        {"u10": (("time", "latitude", "longitude"),
                 np.zeros((t.size, lat.size, lon.size), "float32"))},
        coords={"time": t, "latitude": lat, "longitude": lon},
    )
    out = normalise(ds)
    assert float(out["longitude"].max()) <= 180.0
    assert np.all(np.diff(out["longitude"].values) > 0)


def test_missing_coordinate_is_a_clear_error():
    ds = xr.Dataset({"u10": (("a", "b"), np.zeros((2, 2), "float32"))},
                    coords={"a": [1, 2], "b": [1, 2]})
    with pytest.raises(ValueError, match="cannot find a 'latitude'"):
        normalise(ds)


def test_subset_cuts_to_the_case_with_margin():
    out = subset(normalise(cmems_like()), AOI,
                 datetime(2024, 3, 10, 1, tzinfo=UTC), datetime(2024, 3, 10, 4, tzinfo=UTC))
    assert out["latitude"].size > 0 and out["longitude"].size > 0
    assert float(out["latitude"].min()) >= AOI.min_lat - 0.75


# ------------------------------------------------------------------ writers


def test_currents_writer_produces_what_the_kernel_reads(tmp_path: Path):
    path = tmp_path / "currents.nc"
    write_currents(normalise(cmems_like()), path)
    with xr.open_dataset(path) as ds:
        assert set(ds.data_vars) == {"uo", "vo"}
        assert ds["uo"].dims == ("time", "latitude", "longitude")


def test_currents_writer_fills_nan_so_particles_beach_rather_than_freeze(tmp_path: Path):
    """A NaN velocity silently freezes a particle instead of beaching it."""
    ds = normalise(cmems_like())
    ds["uo"][:, :3, :3] = np.nan
    write_currents(ds, tmp_path / "c.nc")
    with xr.open_dataset(tmp_path / "c.nc") as out:
        assert not np.isnan(out["uo"].values).any()


def test_wind_writer_reports_the_gated_fraction(tmp_path: Path):
    """The share of the field below 3 m/s predicts the abstention rate."""
    report = write_wind(normalise(era5_like()), tmp_path / "wind.nc")
    assert any("detectability gate" in n for n in report.notes)


def test_missing_variable_names_the_available_ones(tmp_path: Path):
    ds = normalise(cmems_like()).rename({"uo": "eastward_sea_water_velocity"})
    with pytest.raises(ValueError, match="found"):
        write_currents(ds, tmp_path / "c.nc")


# ---------------------------------------------------------------- DMA AIS


def _dma_csv(path: Path, rows: int = 40) -> Path:
    """A CSV in the Danish Maritime Authority's published column layout."""
    base = datetime(2024, 3, 10, 6, 0, tzinfo=UTC)
    data = {
        DMA_COLUMNS["timestamp"]: [
            (base.replace(minute=i % 60)).strftime("%d/%m/%Y %H:%M:%S") for i in range(rows)
        ],
        DMA_COLUMNS["mmsi"]: [219000001 + (i % 3) for i in range(rows)],
        DMA_COLUMNS["lat"]: 57.0 + np.linspace(0, 0.2, rows),
        DMA_COLUMNS["lon"]: 11.3 + np.linspace(0, 0.3, rows),
        DMA_COLUMNS["sog_kn"]: np.full(rows, 11.4),
        DMA_COLUMNS["cog_deg"]: np.full(rows, 62.0),
        DMA_COLUMNS["heading_deg"]: np.full(rows, 61.0),
        DMA_COLUMNS["nav_status"]: ["Under way using engine"] * rows,
        DMA_COLUMNS["ship_name"]: ["VESSEL A"] * rows,
        DMA_COLUMNS["ship_type"]: ["Tanker"] * rows,
        DMA_COLUMNS["length_m"]: np.full(rows, 180.0),
        "Type of mobile": ["Class A"] * rows,
    }
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_dma_timestamps_are_parsed_day_first_and_utc(tmp_path: Path):
    """Parsing day-first dates as month-first silently reorders a third of the
    year, and every attribution downstream inherits the error."""
    df, _ = load_dma_ais(_dma_csv(tmp_path / "ais.csv"))
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].dt.day.unique().tolist() == [10]
    assert df["timestamp"].dt.month.unique().tolist() == [3]


def test_dma_output_is_accepted_by_the_shared_cleaner(tmp_path: Path):
    """Real and synthetic AIS share one cleaning path, so the adapter's job is
    to produce the same schema, not a parallel one."""
    from src.attribution import clean_and_reconstruct

    df, _ = load_dma_ais(_dma_csv(tmp_path / "ais.csv", rows=60))
    tracks, report = clean_and_reconstruct(df)
    assert report.rows_in == 60
    assert tracks


def test_dma_reports_everything_it_drops(tmp_path: Path):
    path = _dma_csv(tmp_path / "ais.csv", rows=30)
    df_raw = pd.read_csv(path)
    df_raw.loc[0, DMA_COLUMNS["lat"]] = np.nan
    df_raw.loc[1, DMA_COLUMNS["timestamp"]] = "not-a-date"
    df_raw.to_csv(path, index=False)

    _, report = load_dma_ais(path)
    assert report.dropped["incomplete_fix"] >= 1
    assert report.dropped["unparseable_timestamp"] >= 1


def test_dma_filters_to_aoi_and_window(tmp_path: Path):
    df, report = load_dma_ais(
        _dma_csv(tmp_path / "ais.csv", rows=40),
        aoi=BoundingBox(min_lon=11.0, min_lat=56.9, max_lon=11.35, max_lat=57.05),
        t_start=datetime(2024, 3, 10, 6, 0, tzinfo=UTC),
        t_obs=datetime(2024, 3, 10, 6, 30, tzinfo=UTC),
    )
    assert "outside_aoi" in report.dropped and "outside_window" in report.dropped
    assert len(df) < 40


def test_unrecognisable_csv_names_what_it_found(tmp_path: Path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dma_ais(path)


# ------------------------------------------------------------------ scene


def _geotiff(path: Path, data: np.ndarray, crs: str | None = "EPSG:4326") -> Path:
    import rasterio
    from rasterio.transform import from_origin

    bands = data.shape[0] if data.ndim == 3 else 1
    arr = data if data.ndim == 3 else data[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=bands, dtype="float32", crs=crs,
        transform=from_origin(10.0, 58.0, 0.001, 0.001),
    ) as dst:
        for i in range(bands):
            dst.write(arr[i].astype("float32"), i + 1)
    return path


def test_ungeoreferenced_scene_is_refused(tmp_path: Path):
    """An ungeoreferenced scene cannot be drifted or correlated with AIS."""
    src = _geotiff(tmp_path / "raw.tif", np.full((1, 40, 40), -15.0, "float32"), crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        write_scene_from_grd(src, tmp_path / "out.tif")


def test_linear_sigma0_mistaken_for_db_is_refused(tmp_path: Path):
    """Linear sigma-0 read as dB would make every damping ratio meaningless.

    A median test alone cannot catch this: linear values (0.001-1.0) sit
    entirely inside any plausible dB range. Sign and spread are what separate
    them.
    """
    rng = np.random.default_rng(0)
    linear = rng.uniform(0.005, 0.08, (1, 40, 40)).astype("float32")
    src = _geotiff(tmp_path / "linear.tif", linear)
    with pytest.raises(ValueError, match="linear sigma-0, not decibels"):
        write_scene_from_grd(src, tmp_path / "out.tif")


def test_valid_grd_scene_is_written(tmp_path: Path):
    src = _geotiff(tmp_path / "grd.tif", np.full((1, 40, 40), -15.0, "float32"))
    report = write_scene_from_grd(src, tmp_path / "scene" / "sigma0_vv_db.tif")
    assert (tmp_path / "scene" / "sigma0_vv_db.tif").is_file()
    assert report.notes


# ----------------------------------------------------------------- Zenodo


def _zenodo_tile(path: Path, px: int = 64) -> Path:
    """A two-band tile with no CRS, as the archives ship them.

    VV brighter than VH, which is how the band-order check works.
    """
    rng = np.random.default_rng(0)
    vv = rng.normal(-14.0, 1.0, (px, px)).astype("float32")
    vh = rng.normal(-22.0, 1.0, (px, px)).astype("float32")
    return _geotiff(path, np.stack([vv, vh]), crs=None)


def test_zenodo_tile_is_recognised_as_ungeoreferenced(tmp_path: Path):
    """This is what decides the dataset can only be used for segmentation."""
    info = inspect_tile(_zenodo_tile(tmp_path / "0001.tif"))
    assert info.has_crs is False
    assert info.n_bands == 2
    assert info.looks_like_db


def test_band_order_check_catches_a_swap(tmp_path: Path):
    """A swapped VV/VH would not crash; it would quietly halve the contrast and
    depress every IoU we report."""
    rng = np.random.default_rng(1)
    swapped = np.stack([
        rng.normal(-22.0, 1.0, (64, 64)).astype("float32"),   # VH first
        rng.normal(-14.0, 1.0, (64, 64)).astype("float32"),
    ])
    ok = inspect_tile(_zenodo_tile(tmp_path / "ok.tif"))
    bad = inspect_tile(_geotiff(tmp_path / "swapped.tif", swapped, crs=None))
    assert ok.band_order_looks_correct
    assert not bad.band_order_looks_correct


def test_load_tile_defaults_to_vv(tmp_path: Path):
    path = _zenodo_tile(tmp_path / "0001.tif")
    assert VV_BAND == 1 and VH_BAND == 2
    assert np.median(load_tile(path, VV_BAND)) > np.median(load_tile(path, VH_BAND))


def test_requesting_a_missing_band_is_a_clear_error(tmp_path: Path):
    with pytest.raises(ValueError, match="inspect_tile"):
        load_tile(_zenodo_tile(tmp_path / "t.tif"), band=3)


def test_verify_dataset_flags_wrong_tile_size(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()
    _zenodo_tile(root / "0001.tif", px=64)
    result = verify_dataset(root)
    assert result["checked"] == 1
    assert any("expected 2048" in p for p in result["problems"])
    assert "note" in result and "segmentation" in result["note"].lower()
    assert EXPECTED_TILE_PX == 2048


def test_verify_dataset_reports_an_empty_directory(tmp_path: Path):
    result = verify_dataset(tmp_path)
    assert any("no .tif files" in p for p in result["problems"])


def test_class_directories_match_the_published_layout():
    from src.ingest.zenodo import CLASS_DIRS

    assert CLASS_DIRS[TileClass.OIL] == "Oil"
    assert CLASS_DIRS[TileClass.LOOKALIKE] == "Lookalike"
    assert CLASS_DIRS[TileClass.NO_OIL] == "No oil"


# ------------------------------------------------------------ the fetcher


def test_readiness_check_downloads_nothing_and_reports_honestly():
    from scripts.fetch_real_data import ZENODO_PARTS, check

    r = check()
    assert set(r.credentials)
    assert "py7zr" in r.packages
    # No credentials are configured in this repository, so a real end-to-end
    # case is not buildable and the tool must say so rather than imply it is.
    assert r.can_build_real_case is False
    assert sum(p["gb"] for p in ZENODO_PARTS.values()) > 90.0
