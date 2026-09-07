"""Stages 1-3 assembled: segmentation, physics gate, characterisation.

Produces the ``SlickDetection`` contract objects the inversion consumes.

Characterisation extracts only what feeds a downstream decision. Features
computed to pad a slide are cut. The one that earns its place most clearly is
**elongation combined with orientation**: a long thin slick whose major axis
lines up with a nearby vessel track means a continuous discharge under way,
which constrains the release time to an *interval* rather than an instant. That
is real information, and it materially narrows the inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.contracts import (
    Classification,
    ReleaseMode,
    SlickDetection,
    SlickEnvironment,
    SlickGeometry,
    SlickRadiometry,
)
from src.detect.gate import (
    LookAlikeClassifier,
    extract_features,
    lane_density,
    outside_detectability_window,
)
from src.detect.injection import read_scene
from src.detect.segment import cell_area_km2, segment_dark_patches
from src.transport import ForcingField

# Probability above which a patch inside the detectability window is called oil.
OIL_THRESHOLD = 0.5

# Release-mode thresholds. See Context/technical_design.md section 2.
CONTINUOUS_MIN_ELONGATION = 4.0
CONTINUOUS_MAX_HEADING_OFFSET_DEG = 15.0
INSTANTANEOUS_MAX_ELONGATION = 2.5

DEFAULT_MODEL_PATH = Path("models/gate_lgbm.txt")


@dataclass
class DetectionDiagnostics:
    n_patches: int = 0
    n_oil: int = 0
    n_look_alike: int = 0
    n_undetermined: int = 0
    model_trained: bool = False
    elapsed_s: float = 0.0

    @property
    def abstention_rate(self) -> float:
        """Target is 5-15%. Too low means the gate is not gating; too high means
        the system is useless."""
        return self.n_undetermined / self.n_patches if self.n_patches else 0.0


def detect(
    case_dir: Path | str,
    field: ForcingField,
    t_obs: datetime,
    case_id: str,
    sigma0_db: np.ndarray | None = None,
    tracks: list | None = None,
    classifier: LookAlikeClassifier | None = None,
    oil_threshold: float = OIL_THRESHOLD,
) -> tuple[list[SlickDetection], DetectionDiagnostics]:
    """Run stages 1-3 over one scene.

    ``sigma0_db`` overrides the on-disk scene, which is how an injected slick is
    fed through the real detector rather than around it.
    """
    import time as _time

    started = _time.perf_counter()
    case_dir = Path(case_dir)
    classifier = classifier or _load_classifier()

    disk_sigma0, lon, lat = read_scene(case_dir)
    sigma0 = disk_sigma0 if sigma0_db is None else sigma0_db
    incidence, _, _ = read_scene(case_dir, "incidence")

    land = _land_on_scene(field, lon, lat)
    cell_km2 = cell_area_km2(lon, lat)

    labels, patches = segment_dark_patches(sigma0, lon, lat, land=land)

    wind_u, wind_v = _wind_on_scene(field, lon, lat, t_obs)
    lanes = None
    if tracks:
        all_lon = np.concatenate([t.lon for t in tracks])
        all_lat = np.concatenate([t.lat for t in tracks])
        lanes = lane_density(all_lon, all_lat, lon, lat)

    sea = ~land.astype(bool)
    background_db = float(np.median(sigma0[sea])) if sea.any() else float(np.median(sigma0))

    detections: list[SlickDetection] = []
    diag = DetectionDiagnostics(n_patches=len(patches), model_trained=classifier.is_trained)

    for i, patch in enumerate(sorted(patches, key=lambda p: -p.area_km2), start=1):
        feats = extract_features(
            patch.pixels, sigma0, incidence, lon, lat, wind_u, wind_v,
            lanes, cell_km2, background_db,
        )

        # The hard gate. Physics first, and the classifier cannot overturn it.
        reason = outside_detectability_window(feats.wind_speed_ms)
        if reason is not None:
            classification, abstained = Classification.UNDETERMINED, True
            confidence = 0.0
            diag.n_undetermined += 1
        else:
            p_oil = classifier.predict_oil_probability(feats)
            confidence = float(p_oil)
            if p_oil >= oil_threshold:
                classification, abstained = Classification.OIL, False
                diag.n_oil += 1
                reason = (
                    f"Wind {feats.wind_speed_ms:.1f} m/s is inside the detectability "
                    f"window; normalised damping {feats.damping_norm_incidence:.2f} with "
                    f"edge-gradient ratio {feats.gradient_ratio:.2f} and elongation "
                    f"{feats.elongation:.1f}."
                )
            else:
                classification, abstained = Classification.LOOK_ALIKE, False
                diag.n_look_alike += 1
                reason = (
                    f"Wind {feats.wind_speed_ms:.1f} m/s is inside the detectability "
                    f"window, but the boundary is diffuse (gradient ratio "
                    f"{feats.gradient_ratio:.2f}) and the shape is amorphous "
                    f"(solidity {feats.solidity:.2f}, elongation {feats.elongation:.1f}); "
                    f"consistent with a surface feature other than mineral oil."
                )

        mode, justification = _release_mode(feats, tracks, t_obs)
        polygon = _patch_polygon(labels == patch.label, lon, lat)

        detections.append(
            SlickDetection(
                detection_id=f"{case_id}:d{i:02d}",
                classification=classification,
                classification_reason=reason,
                abstained=abstained,
                confidence_segmentation=float(np.clip(confidence, 0.0, 1.0)),
                polygon_wgs84=polygon,
                geometry=SlickGeometry(
                    area_km2=max(feats.area_km2, 1e-6),
                    perimeter_km=max(feats.perimeter_km, 1e-6),
                    centroid=feats.centroid_lonlat,
                    elongation=max(feats.elongation, 1.0),
                    orientation_deg=feats.orientation_deg % 180.0,
                    n_components=max(feats.n_components, 1),
                    complexity=max(feats.complexity, 1e-6),
                    solidity=float(np.clip(feats.solidity, 1e-6, 1.0)),
                ),
                radiometry=SlickRadiometry(
                    damping_ratio_db=-abs(feats.damping_ratio_db),
                    damping_norm_incidence=-abs(feats.damping_norm_incidence),
                    sigma0_mean_db=feats.sigma0_mean_db,
                    sigma0_background_db=feats.sigma0_background_db,
                    incidence_deg=float(np.clip(feats.incidence_deg, 0.1, 89.9)),
                ),
                environment=SlickEnvironment(
                    wind_speed_ms=max(feats.wind_speed_ms, 0.0),
                    wind_dir_deg=feats.wind_dir_deg % 360.0,
                    wind_gradient_ms_per_km=max(feats.wind_gradient_ms_per_km, 0.0),
                ),
                release_mode=mode,
                release_mode_justification=justification,
            )
        )

    diag.elapsed_s = _time.perf_counter() - started
    return detections, diag


# ------------------------------------------------------------------ helpers


def _load_classifier(path: Path | str = DEFAULT_MODEL_PATH) -> LookAlikeClassifier:
    path = Path(path)
    if path.is_file():
        return LookAlikeClassifier.load(path)
    return LookAlikeClassifier()


def _release_mode(feats, tracks, t_obs) -> tuple[ReleaseMode, str]:
    """Continuous discharge, instantaneous release, or not determinable.

    A continuous discharge constrains t0 to an interval rather than an instant,
    so this is not a decorative label -- it changes the inversion prior.
    """
    if feats.elongation >= CONTINUOUS_MIN_ELONGATION and tracks:
        offset = _nearest_track_heading_offset(feats, tracks, t_obs)
        if offset is not None and offset <= CONTINUOUS_MAX_HEADING_OFFSET_DEG:
            return ReleaseMode.CONTINUOUS, (
                f"Elongation {feats.elongation:.1f} with major axis "
                f"{feats.orientation_deg:.0f} deg aligned within {offset:.0f} deg of a "
                f"nearby track heading; consistent with continuous discharge under way, "
                f"which constrains the release time to an interval rather than an instant."
            )
        return ReleaseMode.INDETERMINATE, (
            f"Elongation {feats.elongation:.1f} but no nearby track heading within "
            f"{CONTINUOUS_MAX_HEADING_OFFSET_DEG:.0f} deg of the major axis."
        )
    if feats.elongation <= INSTANTANEOUS_MAX_ELONGATION and feats.n_components == 1:
        return ReleaseMode.INSTANTANEOUS, (
            f"Compact and single-component (elongation {feats.elongation:.1f}); "
            f"consistent with an instantaneous release."
        )
    return ReleaseMode.INDETERMINATE, (
        f"Elongation {feats.elongation:.1f} with {feats.n_components} component(s); "
        f"neither clearly track-aligned nor clearly compact."
    )


def _nearest_track_heading_offset(feats, tracks, t_obs) -> float | None:
    lon0, lat0 = feats.centroid_lonlat
    best, best_d = None, np.inf
    for track in tracks:
        d = np.hypot((track.lon - lon0) * 0.6, track.lat - lat0)
        j = int(np.argmin(d))
        if d[j] < best_d:
            best_d, best = d[j], track.heading_near(float(track.time[j]))
    if best is None or best_d > 0.35:  # ~35 km
        return None
    return float(abs((best - feats.orientation_deg + 90.0) % 180.0 - 90.0))


def _patch_polygon(patch_mask: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> list:
    """Outline of a patch as a closed lon/lat ring."""
    from skimage.measure import find_contours

    contours = find_contours(patch_mask.astype(float), 0.5)
    if not contours:
        rows, cols = np.nonzero(patch_mask)
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        ring = [(lon[c0], lat[r0]), (lon[c1], lat[r0]), (lon[c1], lat[r1]), (lon[c0], lat[r1])]
        return [*ring, ring[0]]

    contour = max(contours, key=len)
    step = max(1, len(contour) // 120)
    contour = contour[::step]

    ring = [
        (float(np.interp(c, np.arange(lon.size), lon)),
         float(np.interp(r, np.arange(lat.size), lat)))
        for r, c in contour
    ]
    if len(ring) < 3:
        ring = ring * 3
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _wind_on_scene(field: ForcingField, lon, lat, t_obs) -> tuple[np.ndarray, np.ndarray]:
    """Wind on a coarse grid covering the scene. Coarse is right: ERA5 is 0.25 deg."""
    from src.transport.field import to_epoch_seconds

    n = 60
    gl = np.linspace(lon[0], lon[-1], n)
    ga = np.linspace(lat[0], lat[-1], n)
    gx, gy = np.meshgrid(gl, ga)
    px, py = field.to_grid(gx.ravel(), gy.ravel())
    u, v = field.wind(px, py, to_epoch_seconds(t_obs))
    return u.reshape(n, n), v.reshape(n, n)


def _land_on_scene(field: ForcingField, lon, lat) -> np.ndarray:
    gx, gy = np.meshgrid(lon, lat)
    px, py = field.to_grid(gx.ravel(), gy.ravel())
    return field.is_land(px, py).reshape(gx.shape)


__all__ = ["detect", "DetectionDiagnostics", "OIL_THRESHOLD", "DEFAULT_MODEL_PATH"]
