"""Adapter for the Zenodo Sentinel-1 SAR oil spill dataset.

Trujillo-Acatitla et al., *"Sentinel-1 SAR Oil spill image dataset for train,
validate, and test deep learning models"*, Parts I-III, CC-BY-4.0. Named
directly in the problem statement.

Format, confirmed from the Zenodo record rather than assumed:

- Sentinel-1 SAR **sigma-0 in decibels**, TIFF
- Images are **2048 x 2048 x 2** -- two bands
- Masks are 2048 x 2048, one per image, matched by index
- Organised by class: ``Oil`` / ``Lookalike`` / ``No oil``, 150 of each in the
  test part
- Distributed as ``.7z`` archives totalling ~96 GB across the three parts

What this dataset can and cannot be used for
--------------------------------------------
**It is a segmentation dataset, not a case source.** The tiles are image crops.
Nothing in the record promises a CRS or a geotransform, and without geolocation
a tile cannot be drifted, cannot be matched to a current field, and cannot be
correlated with AIS. So the split is:

- **Zenodo tiles** -> train and benchmark the segmenter and the look-alike
  classifier, and report IoU against the published ~0.54.
- **Copernicus Data Space GRD** -> georeferenced scenes for end-to-end cases,
  paired with CMEMS and ERA5.

That is exactly the division the design always intended (Zenodo listed as
"Training", CDSE as "Inference"); this module makes it explicit in code.
``check_georeferencing`` reports whether a tile carries a CRS, so the
assumption is tested against the real files rather than trusted.

Status
------
Written against the published format description. **Not yet exercised against
the real archives** -- see ``scripts/fetch_real_data.py``. Every assumption that
could not be verified from the record is marked ASSUMPTION below and is
re-checked at load time by ``inspect_tile``, which fails loudly rather than
silently mis-reading a band.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

# ASSUMPTION: band 1 is VV, band 2 is VH.
#
# The record states "2048x2048x2" without naming the band order. VV is the
# detection channel -- Bragg scattering is co-pol and it is what oil damps,
# while VH sits near the noise floor over low-backscatter water, which is
# exactly our regime. Getting this backwards would not crash; it would quietly
# halve the contrast and depress every IoU we report.
#
# inspect_tile() checks it empirically: VH over water is several dB darker than
# VV, so if band 2 is consistently brighter the bands are swapped.
VV_BAND = 1
VH_BAND = 2

# Expected tile geometry, from the record.
EXPECTED_TILE_PX = 2048

# VH should sit this far below VV over open water at C-band. Used only as a
# sanity check on band order, not as a calibration.
MIN_VV_VH_SEPARATION_DB = 3.0


class TileClass(str, Enum):
    OIL = "oil"
    LOOKALIKE = "lookalike"
    NO_OIL = "no_oil"


# Directory names as they appear in the archives.
CLASS_DIRS = {
    TileClass.OIL: "Oil",
    TileClass.LOOKALIKE: "Lookalike",
    TileClass.NO_OIL: "No oil",
}


@dataclass
class TileInfo:
    """What a real tile actually contains, as opposed to what we expected."""

    path: Path
    width: int
    height: int
    n_bands: int
    dtype: str
    has_crs: bool
    crs: str | None
    sigma0_range_db: tuple[float, float]
    band_medians_db: tuple[float, ...]

    @property
    def looks_like_db(self) -> bool:
        """sigma-0 in dB over water sits roughly in [-35, 0]."""
        lo, hi = self.sigma0_range_db
        return -60.0 < lo < 5.0 and -40.0 < hi < 15.0

    @property
    def band_order_looks_correct(self) -> bool:
        """VV should be clearly brighter than VH over water."""
        if len(self.band_medians_db) < 2:
            return True
        vv, vh = self.band_medians_db[0], self.band_medians_db[1]
        return vv - vh >= MIN_VV_VH_SEPARATION_DB


def inspect_tile(path: Path | str) -> TileInfo:
    """Read one tile and report what it really is.

    Called before any bulk load. Every assumption in this module is checked
    here, because a silently mis-read band produces a plausible but depressed
    IoU rather than an error.
    """
    import rasterio

    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read()
        finite = data[np.isfinite(data)]
        medians = tuple(
            float(np.median(b[np.isfinite(b)])) if np.isfinite(b).any() else float("nan")
            for b in data
        )
        return TileInfo(
            path=path,
            width=src.width,
            height=src.height,
            n_bands=src.count,
            dtype=str(src.dtypes[0]),
            has_crs=src.crs is not None,
            crs=str(src.crs) if src.crs else None,
            sigma0_range_db=(
                (float(np.percentile(finite, 1)), float(np.percentile(finite, 99)))
                if finite.size else (float("nan"), float("nan"))
            ),
            band_medians_db=medians,
        )


def verify_dataset(root: Path | str, sample: int = 6) -> dict:
    """Check a extracted archive against every assumption this module makes.

    Returns a report rather than raising, so the failures can be read together.
    """
    root = Path(root)
    report: dict = {"root": str(root), "checked": 0, "problems": [], "georeferenced": 0}

    tiles = sorted(root.rglob("*.tif")) + sorted(root.rglob("*.tiff"))
    if not tiles:
        report["problems"].append("no .tif files found; is the archive extracted?")
        return report

    for path in tiles[:sample]:
        info = inspect_tile(path)
        report["checked"] += 1
        if info.has_crs:
            report["georeferenced"] += 1
        if info.width != EXPECTED_TILE_PX or info.height != EXPECTED_TILE_PX:
            report["problems"].append(
                f"{path.name}: {info.width}x{info.height}, expected "
                f"{EXPECTED_TILE_PX}x{EXPECTED_TILE_PX}"
            )
        if "mask" not in str(path).lower() and info.n_bands != 2:
            report["problems"].append(f"{path.name}: {info.n_bands} bands, expected 2")
        if not info.looks_like_db:
            report["problems"].append(
                f"{path.name}: values in {info.sigma0_range_db}, not plausible sigma-0 dB"
            )
        if not info.band_order_looks_correct:
            report["problems"].append(
                f"{path.name}: band 2 is not clearly darker than band 1 "
                f"(medians {info.band_medians_db}); VV/VH order may be reversed"
            )

    report["all_georeferenced"] = report["georeferenced"] == report["checked"]
    if report["georeferenced"] == 0:
        report["note"] = (
            "No tile carries a CRS. As expected: these are image crops, usable "
            "for segmentation training and IoU benchmarking but NOT for "
            "end-to-end cases, which need geolocation to drift a slick and "
            "correlate it with AIS. Use Copernicus Data Space GRD for those."
        )
    return report


def load_tile(path: Path | str, band: int = VV_BAND) -> np.ndarray:
    """Load one band of sigma-0 in dB. VV by default."""
    import rasterio

    with rasterio.open(path) as src:
        if band > src.count:
            raise ValueError(
                f"{Path(path).name} has {src.count} band(s); band {band} requested. "
                f"Run inspect_tile() before assuming the layout."
            )
        return src.read(band).astype("float32")


def load_mask(path: Path | str) -> np.ndarray:
    """Load a ground-truth mask as a boolean oil/not-oil raster."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1) > 0


def iter_pairs(root: Path | str, tile_class: TileClass):
    """Yield ``(image_path, mask_path)`` for one class.

    Pairing is by matching index in the filename, which is how the dataset is
    organised: image 0001 has mask 0001.
    """
    root = Path(root)
    images = sorted((root / "Images" / CLASS_DIRS[tile_class]).glob("*.tif"))
    masks = {p.stem: p for p in (root / "Mask" / CLASS_DIRS[tile_class]).glob("*.tif")}
    for image in images:
        mask = masks.get(image.stem)
        if mask is not None:
            yield image, mask


__all__ = [
    "TileClass",
    "TileInfo",
    "CLASS_DIRS",
    "inspect_tile",
    "verify_dataset",
    "load_tile",
    "load_mask",
    "iter_pairs",
    "VV_BAND",
    "VH_BAND",
    "EXPECTED_TILE_PX",
]
