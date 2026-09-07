"""Inject an oil slick into a SAR scene.

Phase 5. The synthetic scene from Phase 1 contains a genuine low-wind
look-alike but no actual oil, so there is nothing for a detector to find. This
module renders a simulated particle cloud as a damping patch in sigma-0, which
gives us scenes containing **both** real oil and a real look-alike -- the exact
discrimination problem the physics gate exists to solve.

The damping is applied in dB, i.e. multiplicatively in linear power. That is
physically right and it also preserves the speckle statistics already in the
scene: SAR speckle is multiplicative, so scaling an already-speckled image by a
damping factor gives the same distribution as damping the clean field and then
speckling it. Adding a dB offset is therefore correct, whereas subtracting a
linear constant would not be.

Damping saturates with film thickness. A thin sheen suppresses Bragg waves
partially; beyond a certain surface concentration the short-wave spectrum is
already flattened and more oil changes nothing. Hence the exponential
saturation rather than a linear ramp -- and it is why damping ratio cannot be
inverted for thickness from a single scene, which is the reason we refuse to
report slick age radiometrically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Maximum Bragg suppression a thick slick produces, dB. Literature values for
# mineral oil on the sea surface sit around 6-12 dB at C-band VV.
MAX_DAMPING_DB = 9.0

# Surface concentration at which damping is ~63% saturated, in particles per
# scene cell. Calibrated so a few thousand particles produce a visible slick.
SATURATION_DENSITY = 1.2

# Smoothing applied to the damping field, metres. Real slick edges are sharper
# than wind fronts but not pixel-sharp.
EDGE_SMOOTHING_M = 250.0


@dataclass
class InjectedSlick:
    """A slick rendered into a scene, with the ground truth kept alongside.

    ``truth_mask`` is what the detector is supposed to find. It is never handed
    to the detector -- only to the scorer.
    """

    sigma0_db: np.ndarray
    truth_mask: np.ndarray
    damping_db: np.ndarray
    area_km2: float
    n_particles: int

    @property
    def peak_damping_db(self) -> float:
        return float(self.damping_db.min())


def render_damping(
    lon: np.ndarray,
    lat: np.ndarray,
    scene_lon: np.ndarray,
    scene_lat: np.ndarray,
    max_damping_db: float = MAX_DAMPING_DB,
    saturation_density: float = SATURATION_DENSITY,
    smoothing_m: float = EDGE_SMOOTHING_M,
) -> np.ndarray:
    """Damping field in dB (negative) for a particle cloud on a scene grid."""
    from scipy.ndimage import gaussian_filter

    dlon = float(scene_lon[1] - scene_lon[0])
    dlat = float(scene_lat[1] - scene_lat[0])

    ix = np.floor((lon - scene_lon[0]) / dlon).astype(np.intp)
    iy = np.floor((lat - scene_lat[0]) / dlat).astype(np.intp)
    inside = (ix >= 0) & (ix < scene_lon.size) & (iy >= 0) & (iy < scene_lat.size)

    density = np.zeros((scene_lat.size, scene_lon.size), dtype="float32")
    if inside.any():
        np.add.at(density, (iy[inside], ix[inside]), 1.0)

    # Metres per cell, for a smoothing sigma expressed in physical units.
    m_per_deg_lat = 111_320.0
    cell_m = abs(dlat) * m_per_deg_lat
    sigma_cells = max(smoothing_m / max(cell_m, 1e-6), 0.8)
    density = gaussian_filter(density, sigma=sigma_cells)

    # Saturating damping: more oil past a point changes nothing, which is why
    # damping ratio cannot be inverted for film thickness.
    return -max_damping_db * (1.0 - np.exp(-density / saturation_density))


def inject(
    sigma0_db: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    scene_lon: np.ndarray,
    scene_lat: np.ndarray,
    land: np.ndarray | None = None,
    truth_threshold_db: float = 1.5,
    **kwargs,
) -> InjectedSlick:
    """Add a slick to a scene and return it with its ground-truth mask.

    ``truth_threshold_db`` sets what counts as oil for scoring: cells damped by
    less than this are a sheen too faint to be detectable, and calling them
    ground truth would make recall look worse than the detector deserves.
    """
    damping = render_damping(lon, lat, scene_lon, scene_lat, **kwargs)
    if land is not None:
        damping = np.where(land.astype(bool), 0.0, damping)

    truth = damping <= -truth_threshold_db
    cell_km2 = _cell_area_km2(scene_lon, scene_lat)

    return InjectedSlick(
        sigma0_db=(sigma0_db + damping).astype("float32"),
        truth_mask=truth,
        damping_db=damping.astype("float32"),
        area_km2=float(truth.sum() * cell_km2),
        n_particles=int(lon.size),
    )


def _cell_area_km2(scene_lon: np.ndarray, scene_lat: np.ndarray) -> float:
    dlon = abs(float(scene_lon[1] - scene_lon[0]))
    dlat = abs(float(scene_lat[1] - scene_lat[0]))
    mid_lat = float(np.mean(scene_lat))
    m_lat = 111_320.0
    m_lon = m_lat * np.cos(np.deg2rad(mid_lat))
    return (dlon * m_lon) * (dlat * m_lat) / 1e6


def read_scene(case_dir: Path | str, name: str = "sigma0_vv_db"):
    """Load a scene raster and its axes. Returns ``(data, lon, lat)``."""
    import rasterio

    with rasterio.open(Path(case_dir) / "scene" / f"{name}.tif") as src:
        data = src.read(1)
        t = src.transform
        lon = t.c + t.a * (np.arange(src.width) + 0.5)
        lat = t.f + t.e * (np.arange(src.height) + 0.5)

    # GeoTIFF rows run north to south; flip so row 0 is the southern edge and
    # index arithmetic matches every other grid in the project.
    if lat[0] > lat[-1]:
        data = np.flipud(data)
        lat = lat[::-1]
    return data, lon, lat


__all__ = [
    "InjectedSlick",
    "inject",
    "render_damping",
    "read_scene",
    "MAX_DAMPING_DB",
    "SATURATION_DENSITY",
]
