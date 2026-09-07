"""Build the assets the interface renders.

The pipeline's contract objects are the source of truth, but a browser cannot
usefully consume a 3-megapixel GeoTIFF or a 200,000-particle ensemble. This
stage turns ``out/`` into a small, static bundle under ``out/ui/``:

    scene.png        the SAR scene, 8-bit, clipped to a sensible dB range
    coastline.json   land outlines, vectorised from the case's own land mask
    particles.json   ~2,000 backward trails, for the animation
    posterior.json   density cells above a threshold, as weighted points
    manifest.json    bounds, timings, and everything the page needs to lay out

Two decisions worth stating.

**Coastline comes from the case's own land mask, not from a downloaded
basemap.** Natural Earth is the plan for real cases, but deriving it from the
mask the transport kernel actually beached particles against means the
coastline the viewer sees is the coastline the physics used. It also keeps the
demo free of any network dependency, which NFR-1 requires.

**The particle animation shows the BACKWARD proposal pass.** That is the honest
visual: it is the pass that narrows the search, and it runs with diffusion
disabled because backward diffusion is ill-posed. Animating a "backward
diffusion cloud" would be showing the exact thing this project refuses to do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Backscatter range mapped to 0-255 in the scene PNG. Clipping rather than
# auto-scaling keeps brightness comparable between cases.
SIGMA0_CLIP_DB = (-26.0, -6.0)

# Longest edge of the exported scene PNG, pixels. The full raster is ~3 Mpx and
# encodes to 5 MB, which is dead weight over a loopback socket and slow to
# decode. At 1100 px the slick and the low-wind pocket are still obvious, and
# the bundle drops by roughly 5x.
SCENE_MAX_PX = 1100

# Trails kept for the animation. deck.gl holds 60 fps well past this; the limit
# is JSON payload size, not rendering.
MAX_TRAILS = 900

# Posterior cells below this fraction of the peak are dropped from the payload.
POSTERIOR_FLOOR = 0.02


@dataclass
class ExportDiagnostics:
    scene_px: tuple[int, int] = (0, 0)
    n_trails: int = 0
    n_posterior_cells: int = 0
    n_tracks: int = 0
    bundle_kb: float = 0.0
    elapsed_s: float = 0.0


def export_ui(
    case_dir: Path | str,
    field,
    case,
    seed: int = 0,
    max_trails: int = MAX_TRAILS,
) -> ExportDiagnostics:
    """Write ``out/ui/`` for one case."""
    import time as _time

    started = _time.perf_counter()
    case_dir = Path(case_dir)
    out = case_dir / "out"
    ui = out / "ui"
    ui.mkdir(parents=True, exist_ok=True)
    diag = ExportDiagnostics()

    from src.detect.injection import read_scene
    from src.detect.pipeline import _land_on_scene

    sigma0, lon, lat = read_scene(case_dir)
    land = _land_on_scene(field, lon, lat)
    diag.scene_px = (int(sigma0.shape[1]), int(sigma0.shape[0]))

    bounds = [float(lon[0]), float(lat[0]), float(lon[-1]), float(lat[-1])]

    _write_scene_png(ui / "scene.png", sigma0, land)
    _write_coastline(ui / "coastline.json", land, lon, lat)

    detections = _read_json(out / "detections.json", default=[])
    posterior = _read_json(out / "posterior.json", default={})
    candidates = _read_json(out / "candidates.json", default={})

    diag.n_posterior_cells = _write_posterior(ui / "posterior.json", out, posterior)
    diag.n_trails = _write_particles(
        ui / "particles.json", case_dir, field, case, posterior, seed, max_trails
    )
    diag.n_tracks = len(candidates.get("candidates", []))

    manifest = {
        "case_id": case.case_id,
        "region": case.region,
        "description": case.description,
        "t_obs_utc": case.t_obs.isoformat(),
        "lookback_hours": round(case.max_lookback_hours, 1),
        "ais_is_synthetic": case.ais_is_synthetic,
        "bounds": bounds,
        "centre": [(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0],
        "scene_px": diag.scene_px,
        "has_posterior": "credible_regions" in posterior,
        "n_detections": len(detections),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ui / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    diag.bundle_kb = sum(p.stat().st_size for p in ui.iterdir() if p.is_file()) / 1024.0
    diag.elapsed_s = _time.perf_counter() - started
    return diag


# --------------------------------------------------------------------- assets


def _write_scene_png(path: Path, sigma0: np.ndarray, land: np.ndarray) -> None:
    """8-bit greyscale of the SAR scene, land tinted so the coast reads."""
    from PIL import Image

    lo, hi = SIGMA0_CLIP_DB
    grey = np.clip((sigma0 - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.dstack([grey, grey, grey])

    # Land in a cool slate so it is obviously not sea, without competing with
    # the slick overlays for attention.
    mask = land.astype(bool)
    rgb[mask] = np.dstack([
        0.16 + 0.25 * grey[mask], 0.19 + 0.25 * grey[mask], 0.26 + 0.25 * grey[mask]
    ])[0]

    # PNG rows run top to bottom; our arrays are south-up.
    img = Image.fromarray((np.flipud(rgb) * 255).astype("uint8"), mode="RGB")
    if max(img.size) > SCENE_MAX_PX:
        scale = SCENE_MAX_PX / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.LANCZOS)
    img.save(path, optimize=True)


def _write_coastline(path: Path, land: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> None:
    """Vectorise the land mask the physics actually used."""
    from skimage.measure import find_contours

    features = []
    for contour in find_contours(land.astype(float), 0.5):
        if len(contour) < 20:
            continue
        step = max(1, len(contour) // 250)
        coords = [
            [float(np.interp(c, np.arange(lon.size), lon)),
             float(np.interp(r, np.arange(lat.size), lat))]
            for r, c in contour[::step]
        ]
        if len(coords) >= 3:
            features.append({
                "type": "Feature", "properties": {},
                "geometry": {"type": "LineString", "coordinates": coords},
            })
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )


def _write_posterior(path: Path, out: Path, posterior: dict) -> int:
    """Density cells above a floor, as weighted points for a heatmap."""
    npz = out / "posterior.npz"
    if not npz.is_file() or "credible_regions" not in posterior:
        path.write_text(json.dumps({"cells": [], "t0": []}), encoding="utf-8")
        return 0

    data = np.load(npz)
    density = data["density"]
    lon, lat = data["lon"], data["lat"]

    spatial = density.sum(axis=0)
    peak = spatial.max()
    cells = []
    if peak > 0:
        iy, ix = np.nonzero(spatial >= POSTERIOR_FLOOR * peak)
        for r, c in zip(iy, ix):
            cells.append([round(float(lon[c]), 5), round(float(lat[r]), 5),
                          round(float(spatial[r, c] / peak), 4)])

    time_marginal = density.sum(axis=(1, 2))
    total = time_marginal.sum()
    t0 = [
        {"t": datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat(),
         "w": round(float(w / total), 5) if total else 0.0}
        for t, w in zip(data["t0"], time_marginal)
    ]
    path.write_text(json.dumps({"cells": cells, "t0": t0}), encoding="utf-8")
    return len(cells)


def _write_particles(
    path: Path, case_dir: Path, field, case, posterior: dict, seed: int, max_trails: int
) -> int:
    """Backward proposal trails, for the animation.

    Deliberately the backward ADVECTION-ONLY pass. Diffusion is disabled because
    running it backwards is ill-posed, and animating a backward diffusion cloud
    would be showing exactly the mistake this project exists to avoid.
    """
    from src.contracts import SlickDetection
    from src.inversion import ObservedMask
    from src.inversion.hypotheses import PROPOSAL_PARTICLES
    from src.transport import Seeds, TransportParams, simulate

    raw = _read_json(case_dir / "out" / "detections.json", default=[])
    oil = [d for d in (SlickDetection.model_validate(x) for x in raw) if d.is_actionable]
    if not oil:
        path.write_text(json.dumps({"trails": [], "t_start": None, "t_end": None}),
                        encoding="utf-8")
        return 0

    target = max(oil, key=lambda d: d.geometry.area_km2)
    mask = ObservedMask.from_detection(target, field)
    lookback = float(posterior.get("lookback_hours", 24.0))

    rng = np.random.default_rng(seed)
    px, py = mask.sample_points(min(PROPOSAL_PARTICLES, max_trails), rng)
    lon, lat = field.to_lonlat(px, py)

    from datetime import timedelta

    params = TransportParams.proposal(history_stride=2, history_max_particles=max_trails)
    traj = simulate(
        field, Seeds.at_time(lon, lat, case.t_obs), case.t_obs,
        case.t_obs - timedelta(hours=lookback), params, seed=seed, record_history=True,
    )

    hx, hy = traj.history_x, traj.history_y
    n_frames, n_tracked = hx.shape
    trails = []
    for i in range(n_tracked):
        glon, glat = field.to_lonlat(hx[:, i].astype("float64"), hy[:, i].astype("float64"))
        trails.append([[round(float(a), 4), round(float(b), 4)] for a, b in zip(glon, glat)])

    path.write_text(json.dumps({
        "trails": trails,
        "n_frames": int(n_frames),
        "t_start": case.t_obs.isoformat(),
        "lookback_hours": lookback,
        "note": "backward advection-only proposal; diffusion is not run backwards",
    }), encoding="utf-8")
    return len(trails)


def _read_json(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


__all__ = ["export_ui", "ExportDiagnostics", "SIGMA0_CLIP_DB", "MAX_TRAILS"]
