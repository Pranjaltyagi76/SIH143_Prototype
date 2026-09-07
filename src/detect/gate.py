"""Stage 2: the physics gate and look-alike discrimination.

Everyone else attacks look-alikes with a bigger CNN. **The classes are not
separable in image space**, because a low-wind zone and an oil slick genuinely
look identical on a radar image. Oil damps short capillary-gravity waves, which
suppresses Bragg scattering and makes it dark -- but below about 3 m/s the sea
surface is already smooth, so *everything* is dark and no oil-water contrast is
physically possible.

The information that separates the classes is **not in the image**. So we inject
it from the wind field.

The hard gate
-------------
Outside 3-12 m/s, the patch is returned as ``undetermined`` with the wind speed
as the stated reason, and the classifier cannot overturn it. Below the lower
bound no contrast is possible; above the upper bound wave action disperses and
submerges the slick so a coherent surface film is unlikely to persist.

This is enforced twice: here, and again in the ``SlickDetection`` contract, so
no future refactor can quietly route around it.

Adding ``undetermined`` is the maturity signal. An operational system that
abstains under low wind is more trustworthy than one that always answers, and
showing what the system correctly *refuses* to flag is more persuasive than
anything it does flag.

Inside the window
-----------------
A LightGBM classifier over ~14 physical and geometric features. Gradient-boosted
trees rather than a neural network because with this few features GBDT wins,
trains in seconds, and -- critically -- yields feature importances that can be
put on a slide. Explainability is a scored asset here, not a nicety.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.contracts.detection import WIND_MAX_MS, WIND_MIN_MS

# Backscatter falls roughly this steeply across the swath at C-band VV.
INCIDENCE_SLOPE_DB_PER_DEG = 0.20
INCIDENCE_REF_DEG = 35.0

# Feature order is fixed: the model file stores it, and a silent reordering
# would corrupt every prediction without raising anything.
FEATURE_NAMES = (
    "wind_speed_ms",
    "wind_gradient_ms_per_km",
    "damping_ratio_db",
    "damping_norm_incidence",
    "sigma0_mean_db",
    "sigma0_std_db",
    "edge_gradient_db",
    "gradient_ratio",
    "area_km2",
    "elongation",
    "complexity",
    "solidity",
    "hole_ratio",
    "lane_distance_km",
)


@dataclass
class PatchFeatures:
    """The ~14 features Stage 2 decides on. Each is here because it discriminates."""

    wind_speed_ms: float
    wind_gradient_ms_per_km: float
    damping_ratio_db: float
    damping_norm_incidence: float
    sigma0_mean_db: float
    sigma0_std_db: float
    edge_gradient_db: float
    gradient_ratio: float
    area_km2: float
    elongation: float
    complexity: float
    solidity: float
    hole_ratio: float
    lane_distance_km: float

    # Carried for the contract, not fed to the classifier.
    sigma0_background_db: float = 0.0
    incidence_deg: float = 35.0
    wind_dir_deg: float = 0.0
    orientation_deg: float = 0.0
    n_components: int = 1
    perimeter_km: float = 0.0
    centroid_lonlat: tuple[float, float] = (0.0, 0.0)

    def vector(self) -> np.ndarray:
        d = asdict(self)
        return np.array([d[n] for n in FEATURE_NAMES], dtype="float64")


def lane_density(
    ais_lon: np.ndarray,
    ais_lat: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    coarsen: int = 20,
) -> np.ndarray:
    """Distance from every scene cell to the nearest busy shipping lane, km.

    A contextual prior: mineral-oil slicks cluster along traffic, biogenic films
    do not. Weak on its own, which is why it is one of fourteen features rather
    than a rule.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    clon = lon[::coarsen]
    clat = lat[::coarsen]
    hist, _, _ = np.histogram2d(
        ais_lat, ais_lon,
        bins=[
            np.append(clat, clat[-1] + (clat[1] - clat[0])),
            np.append(clon, clon[-1] + (clon[1] - clon[0])),
        ],
    )
    hist = gaussian_filter(hist, sigma=1.5)
    busy = hist > np.percentile(hist[hist > 0], 70) if (hist > 0).any() else hist > 0
    if not busy.any():
        return np.full((clat.size, clon.size), 999.0, dtype="float32")

    cell_km = abs(float(clat[1] - clat[0])) * 111.32
    return (distance_transform_edt(~busy) * cell_km).astype("float32")


def extract_features(
    patch_pixels: np.ndarray,
    sigma0_db: np.ndarray,
    incidence_deg: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    lane_distance: np.ndarray | None,
    cell_km2: float,
    background_db: float,
) -> PatchFeatures:
    """Compute the feature vector for one dark patch."""
    from skimage.measure import label as cc_label
    from skimage.morphology import convex_hull_image

    rows, cols = patch_pixels[:, 0], patch_pixels[:, 1]
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    local[rows - r0, cols - c0] = True

    values = sigma0_db[rows, cols]
    inc = float(np.mean(incidence_deg[rows, cols]))

    # --- radiometry ----------------------------------------------------------
    damping = float(background_db - values.mean())
    # Contrast is measured against a background that itself falls ~0.2 dB per
    # degree of incidence, so an absolute dB difference is not comparable across
    # the swath. Expressing damping as a fraction of the background level makes
    # near-range and far-range patches comparable -- without this, a classifier
    # learns swath position instead of oil.
    damping_norm = damping / max(abs(background_db), 1e-3)

    # --- edges ---------------------------------------------------------------
    gy, gx = np.gradient(sigma0_db[max(r0 - 3, 0):r1 + 3, max(c0 - 3, 0):c1 + 3])
    grad = np.hypot(gx, gy)
    pad_r = r0 - max(r0 - 3, 0)
    pad_c = c0 - max(c0 - 3, 0)
    inner = np.zeros(grad.shape, dtype=bool)
    inner[pad_r:pad_r + local.shape[0], pad_c:pad_c + local.shape[1]] = local
    from scipy.ndimage import binary_dilation, binary_erosion

    boundary = binary_dilation(inner, iterations=2) & ~binary_erosion(inner, iterations=2)
    edge_grad = float(grad[boundary].mean()) if boundary.any() else 0.0
    interior_grad = float(grad[binary_erosion(inner, iterations=2)].mean()) if inner.sum() > 20 else edge_grad
    gradient_ratio = float(edge_grad / max(interior_grad, 1e-6))

    # --- geometry ------------------------------------------------------------
    area_km2 = float(rows.size * cell_km2)
    py = (rows - rows.mean()) * np.sqrt(cell_km2)
    px = (cols - cols.mean()) * np.sqrt(cell_km2)
    cov = np.cov(np.vstack([px, py])) if rows.size > 2 else np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-9, None)
    elongation = float(np.sqrt(vals[-1] / vals[0]))
    ex, ey = vecs[:, -1]
    orientation = float(np.rad2deg(np.arctan2(ex, ey)) % 180.0)

    perim_cells = int((binary_dilation(local, iterations=1) & ~local).sum())
    perimeter_km = perim_cells * np.sqrt(cell_km2)
    complexity = float(perimeter_km**2 / max(4.0 * np.pi * area_km2, 1e-9))
    hull = convex_hull_image(local)
    solidity = float(local.sum() / max(hull.sum(), 1))

    filled = binary_dilation(binary_erosion(local, iterations=1), iterations=1) | local
    hole_ratio = float(filled.sum() / max(local.sum(), 1))
    n_components = int(cc_label(local, connectivity=2).max())

    # --- environment ---------------------------------------------------------
    ri = int(np.clip(rows.mean() * wind_u.shape[0] / sigma0_db.shape[0], 0, wind_u.shape[0] - 1))
    ci = int(np.clip(cols.mean() * wind_u.shape[1] / sigma0_db.shape[1], 0, wind_u.shape[1] - 1))
    speed_field = np.hypot(wind_u, wind_v)
    wind_speed = float(speed_field[ri, ci])
    wind_dir = float(np.rad2deg(np.arctan2(-wind_u[ri, ci], -wind_v[ri, ci])) % 360.0)

    rr0 = int(np.clip(r0 * wind_u.shape[0] / sigma0_db.shape[0], 0, wind_u.shape[0] - 1))
    rr1 = int(np.clip(r1 * wind_u.shape[0] / sigma0_db.shape[0], rr0 + 1, wind_u.shape[0]))
    cc0 = int(np.clip(c0 * wind_u.shape[1] / sigma0_db.shape[1], 0, wind_u.shape[1] - 1))
    cc1 = int(np.clip(c1 * wind_u.shape[1] / sigma0_db.shape[1], cc0 + 1, wind_u.shape[1]))
    block = speed_field[rr0:rr1, cc0:cc1]
    extent_km = max(np.sqrt(area_km2), 1.0)
    wind_gradient = float((block.max() - block.min()) / extent_km) if block.size else 0.0

    if lane_distance is not None:
        li = int(np.clip(rows.mean() * lane_distance.shape[0] / sigma0_db.shape[0],
                         0, lane_distance.shape[0] - 1))
        lj = int(np.clip(cols.mean() * lane_distance.shape[1] / sigma0_db.shape[1],
                         0, lane_distance.shape[1] - 1))
        lane_km = float(lane_distance[li, lj])
    else:
        lane_km = 999.0

    return PatchFeatures(
        wind_speed_ms=wind_speed,
        wind_gradient_ms_per_km=wind_gradient,
        damping_ratio_db=damping,
        damping_norm_incidence=damping_norm,
        sigma0_mean_db=float(values.mean()),
        sigma0_std_db=float(values.std()),
        edge_gradient_db=edge_grad,
        gradient_ratio=gradient_ratio,
        area_km2=area_km2,
        elongation=elongation,
        complexity=complexity,
        solidity=solidity,
        hole_ratio=hole_ratio,
        lane_distance_km=lane_km,
        sigma0_background_db=background_db,
        incidence_deg=inc,
        wind_dir_deg=wind_dir,
        orientation_deg=orientation,
        n_components=n_components,
        perimeter_km=float(perimeter_km),
        centroid_lonlat=(float(lon[cols].mean()), float(lat[rows].mean())),
    )


# ------------------------------------------------------------------ the gate


def outside_detectability_window(wind_speed_ms: float) -> str | None:
    """Return a reason string if physics forbids a judgement, else ``None``."""
    if wind_speed_ms < WIND_MIN_MS:
        return (
            f"Wind {wind_speed_ms:.1f} m/s is below the {WIND_MIN_MS:.0f} m/s "
            f"detectability threshold; at this wind speed the sea surface is already "
            f"smooth and oil cannot produce radar contrast. Outside detectability window."
        )
    if wind_speed_ms > WIND_MAX_MS:
        return (
            f"Wind {wind_speed_ms:.1f} m/s is above the {WIND_MAX_MS:.0f} m/s "
            f"threshold; wave action disperses and submerges surface films, so a "
            f"coherent slick is unlikely to persist. Outside detectability window."
        )
    return None


class LookAlikeClassifier:
    """LightGBM over the physical features, with a rule-based fallback.

    The fallback is not a placeholder to be embarrassed about: until a model is
    trained it makes the same decision on explicit, inspectable grounds, and it
    is what runs if the model file is missing on the demo machine.
    """

    def __init__(self, booster=None):
        self.booster = booster

    @classmethod
    def load(cls, path: Path | str) -> LookAlikeClassifier:
        import lightgbm as lgb

        return cls(lgb.Booster(model_file=str(path)))

    @property
    def is_trained(self) -> bool:
        return self.booster is not None

    def predict_oil_probability(self, features: PatchFeatures) -> float:
        if self.booster is not None:
            return float(self.booster.predict(features.vector()[None, :])[0])
        return self._rule_based(features)

    @staticmethod
    def _rule_based(f: PatchFeatures) -> float:
        """Explicit, inspectable fallback.

        Oil boundaries are sharper than low-wind transitions, oil is more
        homogeneous inside, and vessel discharges are elongated. A wind field
        with a strong local gradient across the patch points to a wind feature.
        """
        score = 0.0
        score += 1.2 * np.tanh(max(f.damping_norm_incidence, 0.0) / 0.35)
        score += 0.9 * np.tanh(f.gradient_ratio - 1.0)
        score += 0.6 * np.tanh((f.elongation - 2.0) / 2.0)
        score -= 1.1 * np.tanh(f.wind_gradient_ms_per_km / 0.05)
        score -= 0.5 * np.tanh((f.solidity - 0.85) / 0.1)
        return float(1.0 / (1.0 + np.exp(-score)))

    def feature_importance(self) -> dict[str, float]:
        if self.booster is None:
            return {}
        gains = self.booster.feature_importance(importance_type="gain")
        return dict(sorted(zip(FEATURE_NAMES, gains), key=lambda kv: -kv[1]))


__all__ = [
    "PatchFeatures",
    "FEATURE_NAMES",
    "extract_features",
    "lane_density",
    "outside_detectability_window",
    "LookAlikeClassifier",
    "INCIDENCE_SLOPE_DB_PER_DEG",
    "INCIDENCE_REF_DEG",
]
