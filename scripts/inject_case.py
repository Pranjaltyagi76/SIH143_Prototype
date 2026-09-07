"""Derive a spill case from a clean one.

    python scripts/inject_case.py --case synth_kattegat --lookback 18

Copies a Case folder, seeds a discharge along one of its real AIS tracks,
forward-simulates it to the acquisition time, and bakes the resulting damping
into the scene raster. The clean case is left untouched.

The ground truth -- which vessel, which window, which cells -- is written to
``truth.json`` **outside** the folders the pipeline reads. The pipeline is given
the scene and nothing else, so recovering the vessel is a real recovery and not
a lookup.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attribution import clean_and_reconstruct, load_ais  # noqa: E402
from src.contracts import CaseManifest  # noqa: E402
from src.detect import inject, read_scene  # noqa: E402
from src.detect.pipeline import _land_on_scene  # noqa: E402
from src.ingest.case_builder import load_forcing_bundle  # noqa: E402
from src.transport import ForcingField, Seeds, TransportParams, simulate  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="synth_kattegat")
    ap.add_argument("--suffix", default="_spill")
    ap.add_argument("--lookback", type=float, default=18.0, help="hours before acquisition")
    ap.add_argument("--window", type=float, default=2.5, help="discharge duration, hours")
    ap.add_argument("--vessel-index", type=int, default=3)
    ap.add_argument("--particles", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    cases = ROOT / "data" / "cases"
    src_dir = cases / args.case
    dst_dir = cases / f"{args.case}{args.suffix}"

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("out"))
    (dst_dir / "out").mkdir(exist_ok=True)

    case = CaseManifest.load(src_dir)
    field = ForcingField.from_case(src_dir, load_forcing_bundle(src_dir))
    tracks, _ = clean_and_reconstruct(load_ais(src_dir / "ais" / "tracks.parquet"))

    window_s = args.window * 3600.0
    t0 = case.t_obs.timestamp() - args.lookback * 3600.0
    usable = [t for t in tracks if t.covers(t0) and t.covers(t0 + window_s) and len(t) > 150]
    if not usable:
        print("error: no AIS track covers the requested discharge window")
        return 1
    culprit = usable[args.vessel_index % len(usable)]

    rng = np.random.default_rng(args.seed)
    times = t0 + rng.random(args.particles) * window_s
    lon, lat, _ = culprit.interpolate(times)
    traj = simulate(
        field,
        Seeds(lon=lon, lat=lat, seed_time=times,
              origin_marker=np.zeros(args.particles, "int64")),
        datetime.fromtimestamp(t0, tz=timezone.utc), case.t_obs,
        TransportParams(), seed=args.seed,
    )
    ok = traj.usable

    sigma0, slon, slat = read_scene(src_dir)
    land = _land_on_scene(field, slon, slat)
    injected = inject(sigma0, traj.lon[ok], traj.lat[ok], slon, slat, land=land)

    _write_scene(dst_dir / "scene" / "sigma0_vv_db.tif",
                 src_dir / "scene" / "sigma0_vv_db.tif", injected.sigma0_db)

    manifest = case.model_copy(update={
        "case_id": dst_dir.name,
        "description": f"{case.description} Contains one injected discharge.",
        "notes": [*case.notes,
                  "Scene contains a synthetic oil discharge injected from a real AIS track.",
                  "Ground truth is in truth.json, which the pipeline never reads."],
    })
    manifest.save(dst_dir)

    truth = {
        "case_id": dst_dir.name,
        "culprit_mmsi": culprit.mmsi,
        "culprit_type": culprit.ship_type,
        "discharge_start_utc": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "discharge_end_utc": datetime.fromtimestamp(t0 + window_s, tz=timezone.utc).isoformat(),
        "lookback_hours": args.lookback,
        "slick_area_km2": round(injected.area_km2, 1),
        "peak_damping_db": round(injected.peak_damping_db, 2),
        "particles_released": args.particles,
        "particles_usable": int(ok.sum()),
    }
    (dst_dir / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    print(f"built {dst_dir.name}")
    print(f"  culprit        {culprit.mmsi} ({culprit.ship_type})")
    print(f"  discharge      {truth['discharge_start_utc']} +{args.window} h")
    print(f"  lookback       {args.lookback} h before acquisition")
    print(f"  slick          {injected.area_km2:.0f} km2, peak damping "
          f"{injected.peak_damping_db:.1f} dB")
    print(f"\n  ground truth in truth.json -- the pipeline never reads it")
    print(f"\nnext: python scripts/run_case.py --case data/cases/{dst_dir.name}")
    return 0


def _write_scene(dst: Path, template: Path, data: np.ndarray) -> None:
    """Write a raster reusing the template's georeferencing."""
    import rasterio

    with rasterio.open(template) as src:
        profile = src.profile
        tags = src.tags()
    with rasterio.open(dst, "w", **profile) as out:
        # Scene arrays are south-up internally; GeoTIFF rows run north to south.
        out.write(np.flipud(data).astype(profile["dtype"]), 1)
        out.update_tags(**tags, injected="true")


if __name__ == "__main__":
    raise SystemExit(main())
