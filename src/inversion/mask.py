"""The observed slick, rasterised onto a projected grid.

This is the thing the inversion conditions on. Two details matter more than
they look.

**The mask is a grid, not a polygon.**
The likelihood is evaluated per cell, so the polygon is rasterised once, at
load, in projected metres. Doing point-in-polygon per particle per hypothesis
would be orders of magnitude slower and no more correct.

**There is a third state between oil and not-oil.**
Segmentation boundaries are uncertain to a pixel or two, plus geolocation
error. Cells inside that annulus are marked ``unobserved`` and contribute to
neither side of the likelihood. Without it, a hypothesis is punished for a
one-cell disagreement at the slick edge, which is noise rather than evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.contracts import Polygon, SlickDetection
from src.transport import ForcingField

# Cell size for the observation grid, metres. A typical slick of a few tens of
# km^2 spans a few hundred cells at this resolution -- fine enough to carry
# shape information, coarse enough that the likelihood stays cheap.
MASK_GRID_M = 500.0

# Extra margin around the slick bounding box, metres. Particles beyond this are
# simply "outside the mask" and need no grid cell of their own.
MASK_MARGIN_M = 2_000.0


@dataclass
class ObservedMask:
    """A rasterised slick observation in the working (projected) CRS."""

    x0: float
    y0: float
    dx: float
    dy: float
    mask: np.ndarray  # (ny, nx) bool -- observed oil
    unobserved: np.ndarray  # (ny, nx) bool -- boundary annulus, scores nothing
    crs: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def n_mask_cells(self) -> int:
        return int(self.mask.sum())

    @property
    def area_km2(self) -> float:
        return self.n_mask_cells * self.dx * self.dy / 1e6

    @property
    def centroid_xy(self) -> tuple[float, float]:
        iy, ix = np.nonzero(self.mask)
        return (
            float(self.x0 + self.dx * (ix.mean() + 0.5)),
            float(self.y0 + self.dy * (iy.mean() + 0.5)),
        )

    # ------------------------------------------------------------- building

    @classmethod
    def from_polygon(
        cls,
        polygon_wgs84: Polygon,
        field: ForcingField,
        boundary_uncertainty_m: float = 1_000.0,
        grid_m: float = MASK_GRID_M,
        margin_m: float = MASK_MARGIN_M,
    ) -> ObservedMask:
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        from scipy.ndimage import binary_dilation, binary_erosion
        from shapely.geometry import Polygon as ShapelyPolygon

        lon = np.array([p[0] for p in polygon_wgs84], dtype="float64")
        lat = np.array([p[1] for p in polygon_wgs84], dtype="float64")
        px, py = field.to_grid(lon, lat)

        x0 = float(px.min() - margin_m)
        y0 = float(py.min() - margin_m)
        nx = int((px.max() + margin_m - x0) / grid_m) + 1
        ny = int((py.max() + margin_m - y0) / grid_m) + 1

        # Rasterise with rows running north to south, then flip so row 0 is the
        # southern edge and index arithmetic matches the rest of the codebase.
        transform = from_origin(x0, y0 + grid_m * ny, grid_m, grid_m)
        burned = rasterize(
            [(ShapelyPolygon(zip(px, py)), 1)],
            out_shape=(ny, nx),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
        mask = np.flipud(burned).astype(bool)

        if not mask.any():
            raise ValueError(
                "slick polygon rasterised to zero cells; it is smaller than one "
                f"{grid_m:.0f} m grid cell"
            )

        # The annulus straddles the boundary: cells just inside and just outside.
        pad = max(1, int(round(boundary_uncertainty_m / grid_m)))
        outer = binary_dilation(mask, iterations=pad)
        inner = binary_erosion(mask, iterations=pad)
        unobserved = outer & ~inner
        # Cells in the annulus are excluded from BOTH terms of the likelihood.
        mask = mask & ~unobserved

        if not mask.any():
            raise ValueError(
                "boundary uncertainty consumed the entire slick; reduce "
                "boundary_uncertainty_m or use a finer grid"
            )

        return cls(x0=x0, y0=y0, dx=grid_m, dy=grid_m, mask=mask,
                   unobserved=unobserved, crs=field.crs)

    @classmethod
    def from_points(
        cls,
        lon: np.ndarray,
        lat: np.ndarray,
        field: ForcingField,
        keep_fraction: float = 0.85,
        boundary_uncertainty_m: float = 1_000.0,
        grid_m: float = MASK_GRID_M,
        margin_m: float = MASK_MARGIN_M,
        smooth_cells: float = 1.5,
    ) -> ObservedMask:
        """Build a slick by thresholding particle density.

        Used to turn a simulated release into a synthetic observation. The
        obvious shortcut -- taking the convex hull of the particle cloud -- is
        wrong and measurably so: a hull is far larger than the plume it
        encloses, and it fills in the concave gaps a real drifting slick has.
        Scoring against a hull rewards hypotheses that over-spread, which biases
        the posterior away from the true compact source.

        Thresholding density keeps the shape the transport actually produced,
        which is what a satellite would have seen.
        """
        from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

        px, py = field.to_grid(np.asarray(lon), np.asarray(lat))
        x0 = float(px.min() - margin_m)
        y0 = float(py.min() - margin_m)
        nx = int((px.max() + margin_m - x0) / grid_m) + 1
        ny = int((py.max() + margin_m - y0) / grid_m) + 1

        ix = np.clip(((px - x0) / grid_m).astype(np.intp), 0, nx - 1)
        iy = np.clip(((py - y0) / grid_m).astype(np.intp), 0, ny - 1)
        density = np.zeros((ny, nx), dtype="float64")
        np.add.at(density, (iy, ix), 1.0)
        density = gaussian_filter(density, sigma=smooth_cells)

        flat = density.ravel()
        order = np.argsort(flat)[::-1]
        cumulative = np.cumsum(flat[order])
        if cumulative[-1] <= 0:
            raise ValueError("particle cloud produced no density")
        n_keep = int(np.searchsorted(cumulative, keep_fraction * cumulative[-1]) + 1)
        mask = np.zeros(flat.size, dtype=bool)
        mask[order[:n_keep]] = True
        mask = mask.reshape(density.shape)

        pad = max(1, int(round(boundary_uncertainty_m / grid_m)))
        unobserved = binary_dilation(mask, iterations=pad) & ~binary_erosion(mask, iterations=pad)
        mask = mask & ~unobserved
        if not mask.any():
            raise ValueError("boundary uncertainty consumed the entire slick")

        return cls(x0=x0, y0=y0, dx=grid_m, dy=grid_m, mask=mask,
                   unobserved=unobserved, crs=field.crs)

    @classmethod
    def from_detection(
        cls, detection: SlickDetection, field: ForcingField, **kwargs
    ) -> ObservedMask:
        """Build from a Stage 1-3 detection. Only confirmed oil may be inverted."""
        if not detection.is_actionable:
            raise ValueError(
                f"detection {detection.detection_id} is classified "
                f"'{detection.classification.value}', not 'oil'; the inversion "
                f"runs only on confirmed oil"
            )
        return cls.from_polygon(detection.polygon_wgs84, field, **kwargs)

    # ------------------------------------------------------------ membership

    def classify_points(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(inside_mask, in_annulus)`` for each point.

        Points outside the grid entirely are neither -- they count as clean sea,
        which is exactly right: predicting oil out there should cost you.
        """
        ix = ((x - self.x0) / self.dx).astype(np.intp)
        iy = ((y - self.y0) / self.dy).astype(np.intp)
        ny, nx = self.mask.shape
        on_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

        inside = np.zeros(x.shape, dtype=bool)
        annulus = np.zeros(x.shape, dtype=bool)
        if on_grid.any():
            gi, gj = iy[on_grid], ix[on_grid]
            inside[on_grid] = self.mask[gi, gj]
            annulus[on_grid] = self.unobserved[gi, gj]
        return inside, annulus

    def cell_rank(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Index of each point's mask cell in ``[0, n_mask_cells)``; -1 if not oil.

        Precomputing this ranking is what lets the likelihood bin particles by
        (hypothesis, mask cell) with a single ``bincount``.
        """
        ranks = np.full(self.mask.shape, -1, dtype=np.int64)
        ranks[self.mask] = np.arange(self.n_mask_cells)

        ix = ((x - self.x0) / self.dx).astype(np.intp)
        iy = ((y - self.y0) / self.dy).astype(np.intp)
        ny, nx = self.mask.shape
        on_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

        out = np.full(x.shape, -1, dtype=np.int64)
        if on_grid.any():
            out[on_grid] = ranks[iy[on_grid], ix[on_grid]]
        return out

    def sample_points(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` points uniformly over the observed slick.

        Used to seed the backward proposal pass.
        """
        iy, ix = np.nonzero(self.mask)
        pick = rng.integers(0, iy.size, n)
        return (
            self.x0 + self.dx * (ix[pick] + rng.random(n)),
            self.y0 + self.dy * (iy[pick] + rng.random(n)),
        )


__all__ = ["ObservedMask", "MASK_GRID_M", "MASK_MARGIN_M"]
