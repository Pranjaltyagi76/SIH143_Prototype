"""Stage 1: dark-patch segmentation.

Deliberately **high recall**. Precision is Stage 2's job -- a dark patch that
turns out to be a low-wind artefact should be caught by the physics gate and
reported with a reason, not silently dropped here. Missing a slick at this stage
is unrecoverable; over-detecting is not.

Why this is classical and not a U-Net
-------------------------------------
The design calls for a U-Net with a ResNet34 encoder, and that remains the plan
for real Sentinel-1 tiles. It is deliberately **not** used on the synthetic
scenes, for a reason worth stating plainly:

Our synthetic sigma-0 is generated from a smooth analytic wind field with
multiplicative Gamma speckle. A CNN trained on that would reach an IoU near
1.0 -- and the number would be meaningless, because the texture it learned does
not exist in real SAR. Reporting it would be self-deception, and reporting it on
a slide would be worse.

So the split is honest:

- **synthetic path (now):** an adaptive-threshold segmenter. Fast, no training,
  no false precision, and entirely adequate for developing everything
  downstream.
- **real-data path (Phase 5b/8):** the U-Net, trained on the Zenodo tiles, with
  a geographic holdout, reported against the published benchmark of IoU ~0.54.

Both produce the same label raster, so nothing downstream changes when the
model arrives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Window over which the local background backscatter is estimated, metres.
#
# This is the most sensitive parameter in Stage 1, and its value is a real
# trade-off rather than a tuned constant. The window must be LARGER than the
# largest dark feature you want to detect, or the feature contaminates its own
# reference level and becomes invisible. Measured on the development scene,
# whose low-wind pocket is ~70 km across:
#
#     20 km -> 1 patch,  low-wind pocket INVISIBLE (window sits inside it)
#     40 km -> 1 patch,  still invisible
#     60 km -> 6 patches, pocket found at 257 km^2   <- default
#     90 km -> 2 patches, pocket over-merged to 1104 km^2
#
# 60 km is defensible on physical grounds and not just because it works here:
# low-wind regions in a reanalysis wind field are synoptic-scale, tens to
# hundreds of km, while vessel slicks are at most tens of km. A window between
# the two separates them. But the sensitivity above is a genuine limitation of
# threshold-based segmentation, and it is one of the reasons the real-data path
# uses a trained segmenter instead.
BACKGROUND_WINDOW_M = 60_000.0

# How far below local background a cell must sit to be a candidate, dB.
# Deliberately permissive: Stage 2 removes false positives with physics.
DARK_THRESHOLD_DB = 1.6

# Speckle suppression, metres. Small enough to keep slick edges sharp.
DESPECKLE_M = 300.0

# Patches smaller than this are speckle residue, not slicks.
MIN_PATCH_KM2 = 1.5


@dataclass
class DarkPatch:
    """One candidate dark region, before any judgement about what it is."""

    label: int
    pixels: np.ndarray  # (N, 2) array of (row, col)
    area_km2: float
    centroid_lonlat: tuple[float, float]

    @property
    def n_pixels(self) -> int:
        return int(self.pixels.shape[0])


def segment_dark_patches(
    sigma0_db: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    land: np.ndarray | None = None,
    background_window_m: float = BACKGROUND_WINDOW_M,
    dark_threshold_db: float = DARK_THRESHOLD_DB,
    despeckle_m: float = DESPECKLE_M,
    min_patch_km2: float = MIN_PATCH_KM2,
) -> tuple[np.ndarray, list[DarkPatch]]:
    """Find dark regions relative to a locally estimated background.

    Returns ``(labels, patches)`` where ``labels`` is 0 for background and
    1..N for patches.
    """
    from scipy.ndimage import gaussian_filter, uniform_filter
    from skimage.measure import label as cc_label
    from skimage.morphology import closing, disk, opening

    cell_m = _cell_size_m(lon, lat)
    cell_km2 = (cell_m[0] * cell_m[1]) / 1e6

    sea = np.ones(sigma0_db.shape, dtype=bool) if land is None else ~land.astype(bool)

    # Despeckle before thresholding. Speckle is multiplicative, so smoothing in
    # dB (log domain) is the right place to do it -- it averages the log of a
    # multiplicative process rather than distorting the linear power.
    sigma_cells = max(despeckle_m / cell_m[1], 0.8)
    smooth = gaussian_filter(np.where(sea, sigma0_db, np.nan), sigma=sigma_cells, mode="nearest")
    smooth = np.where(np.isfinite(smooth), smooth, np.nanmedian(sigma0_db[sea]))

    # Local background: a broad mean over sea cells only, so the coastline does
    # not drag the reference level and create a false dark rim.
    win = max(int(background_window_m / cell_m[1]), 5)
    filled = np.where(sea, smooth, 0.0)
    weights = uniform_filter(sea.astype("float32"), size=win, mode="nearest")
    background = uniform_filter(filled, size=win, mode="nearest") / np.clip(weights, 1e-3, None)

    candidate = sea & ((background - smooth) > dark_threshold_db)

    # Open then close: remove speckle specks, then fill pinholes inside slicks.
    radius = max(int(round(400.0 / cell_m[1])), 1)
    candidate = opening(candidate, disk(radius))
    candidate = closing(candidate, disk(radius))
    # Closing dilates, so it can push a coastal patch onto the shore. Re-apply
    # the sea mask: a patch that extends onto land corrupts its own geometry and
    # would put "oil" on a beach.
    candidate &= sea

    labels = cc_label(candidate, connectivity=2)

    patches: list[DarkPatch] = []
    keep = np.zeros(labels.max() + 1, dtype=bool)
    for lbl in range(1, labels.max() + 1):
        rows, cols = np.nonzero(labels == lbl)
        if rows.size * cell_km2 < min_patch_km2:
            continue
        keep[lbl] = True
        patches.append(
            DarkPatch(
                label=lbl,
                pixels=np.column_stack([rows, cols]),
                area_km2=float(rows.size * cell_km2),
                centroid_lonlat=(float(lon[cols].mean()), float(lat[rows].mean())),
            )
        )

    labels = np.where(keep[labels], labels, 0)
    return labels, patches


def _cell_size_m(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    """Cell width and height in metres. Never compute distance in degrees (W-11)."""
    m_lat = 111_320.0
    m_lon = m_lat * np.cos(np.deg2rad(float(np.mean(lat))))
    return (
        abs(float(lon[1] - lon[0])) * m_lon,
        abs(float(lat[1] - lat[0])) * m_lat,
    )


def cell_area_km2(lon: np.ndarray, lat: np.ndarray) -> float:
    w, h = _cell_size_m(lon, lat)
    return w * h / 1e6


def segmentation_iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    """IoU on the oil class specifically.

    Never overall pixel accuracy. Oil is a tiny fraction of any scene, so
    "predict all sea" scores 99% accuracy -- quoting it is a tell that the class
    imbalance has not been understood.
    """
    p = predicted.astype(bool)
    t = truth.astype(bool)
    union = (p | t).sum()
    return float((p & t).sum() / union) if union else 0.0


def segmentation_recall(predicted: np.ndarray, truth: np.ndarray) -> float:
    t = truth.astype(bool)
    return float((predicted.astype(bool) & t).sum() / t.sum()) if t.any() else 0.0


__all__ = [
    "DarkPatch",
    "segment_dark_patches",
    "segmentation_iou",
    "segmentation_recall",
    "cell_area_km2",
    "BACKGROUND_WINDOW_M",
    "DARK_THRESHOLD_DB",
    "MIN_PATCH_KM2",
]
