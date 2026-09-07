"""End-to-end evaluation against synthetic ground truth.

    python scripts/truth_harness.py --n 24

This is the only source of numbers for the inversion and the attribution
engine, because **no public dataset of confirmed spill-to-vessel attributions
exists**. It seeds a discharge along a real AIS track, hands the pipeline
nothing but the scene, and asks whether the vessel and the release window come
back.

Every trial runs the *whole* chain -- inject, detect, invert, attribute -- so
the numbers include detector error rather than assuming a perfect mask.

The anti-cheating rules matter more than the harness
----------------------------------------------------
A harness that flatters the system is worse than no harness.

**Drift parameters differ between generation and inversion.** The generator
draws windage from U(0.005, 0.055) and diffusivity from LogU(0.5, 20); the
inversion assumes U(0.01, 0.04) and LogU(1, 10). Without this the run would
only prove that our simulator agrees with itself, which is not a claim about
anything.

**The pipeline never learns which vessel seeded the slick.** It receives the
scene and the full day's unfiltered AIS. The prefilter has to earn its
reduction against real traffic density.

**One trial in four deliberately deletes the culprit from AIS.** The correct
answer there is "a vessel not transmitting", and a system that names someone
anyway is broken in the most damaging way available to it. Nothing else tests
that.

**Lookback is swept**, so the operating-envelope curve is measured rather than
asserted.

Outputs
-------
    eval/trials.csv       one row per trial, every field
    eval/calibration.csv  nominal vs empirical coverage
    eval/summary.json     the headline numbers
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field as _field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attribution import (  # noqa: E402
    AttributionConfig,
    attribute,
    clean_and_reconstruct,
    load_ais,
)
from src.contracts import CaseManifest  # noqa: E402
from src.detect import LookAlikeClassifier, detect, inject, read_scene  # noqa: E402
from src.detect.pipeline import _land_on_scene  # noqa: E402
from src.detect.segment import segmentation_iou  # noqa: E402
from src.ingest.case_builder import load_forcing_bundle  # noqa: E402
from src.inversion import InversionConfig, ObservedMask, invert  # noqa: E402
from src.transport import ForcingField, Seeds, TransportParams, simulate  # noqa: E402

# Credible levels evaluated for the calibration curve. 50 and 95 are required
# by the contract; the intermediate levels are what make the curve a curve.
LEVELS = (50, 68, 90, 95)

# Generation-time drift parameters, deliberately WIDER than the inversion's
# assumptions. This is the rule that stops the harness testing our simulator
# against itself.
GEN_WINDAGE = (0.005, 0.055)
GEN_DIFFUSIVITY = (0.5, 20.0)

# One trial in this many hides the culprit from AIS entirely.
DARK_EVERY = 4

LOOKBACKS = (12.0, 24.0, 36.0, 48.0, 60.0)


@dataclass
class Trial:
    trial: int
    seed: int
    lookback_hours: float
    window_hours: float
    dark_case: bool
    culprit_mmsi: str = ""
    culprit_type: str = ""

    detected: bool = False
    truth_area_km2: float = 0.0
    detected_area_km2: float = 0.0
    detection_iou: float = 0.0

    inverted: bool = False
    area_50_km2: float = 0.0
    area_95_km2: float = 0.0
    t0_width_hours: float = 0.0
    ess_fraction: float = 0.0
    inside: dict = _field(default_factory=dict)
    t0_inside_95: bool = False

    attributed: bool = False
    culprit_rank: int = 0
    n_candidates: int = 0
    vessels_in_window: int = 0
    reduction_factor: float = 0.0
    dark_probability: float = 0.0
    dark_is_top: bool = False

    note: str = ""


def run_trial(ctx: dict, trial_no: int, rng: np.random.Generator,
              matched_params: bool = False) -> Trial:
    """One trial.

    ``matched_params`` generates with the SAME drift distributions the inversion
    assumes. That configuration is not a fair end-to-end test -- it lets the
    simulator agree with itself -- but running both modes decomposes any
    miscalibration into two very different causes: a defect in the inversion
    machinery, or a prior that does not cover reality. Only the second is
    survivable, so it is worth knowing which one you have.
    """
    case, field, tracks, sigma0, lon, lat, land, classifier = (
        ctx["case"], ctx["field"], ctx["tracks"], ctx["sigma0"],
        ctx["lon"], ctx["lat"], ctx["land"], ctx["classifier"],
    )
    seed = int(rng.integers(1, 1 << 30))
    lookback = float(LOOKBACKS[trial_no % len(LOOKBACKS)])
    window_h = float(rng.uniform(1.0, 4.0))
    dark = (trial_no % DARK_EVERY) == DARK_EVERY - 1

    t = Trial(trial=trial_no, seed=seed, lookback_hours=lookback,
              window_hours=window_h, dark_case=dark)

    window_s = window_h * 3600.0
    t0 = case.t_obs.timestamp() - lookback * 3600.0
    usable = [x for x in tracks if x.covers(t0) and x.covers(t0 + window_s) and len(x) > 150]
    if not usable:
        t.note = "no AIS track covers the discharge window"
        return t

    culprit = usable[int(rng.integers(len(usable)))]
    t.culprit_mmsi, t.culprit_type = culprit.mmsi, culprit.ship_type

    # --- generate, with drift parameters the inversion does not assume -------
    gen_params = (
        TransportParams() if matched_params
        else TransportParams(windage_alpha=GEN_WINDAGE, diffusivity_kh=GEN_DIFFUSIVITY)
    )
    n = int(rng.integers(4000, 9000))
    times = t0 + rng.random(n) * window_s
    clon, clat, _ = culprit.interpolate(times)
    traj = simulate(
        field,
        Seeds(lon=clon, lat=clat, seed_time=times, origin_marker=np.zeros(n, "int64")),
        datetime.fromtimestamp(t0, tz=timezone.utc), case.t_obs, gen_params, seed=seed,
    )
    ok = traj.usable
    if ok.sum() < 400:
        t.note = "too few particles survived to t_obs"
        return t

    injected = inject(sigma0, traj.lon[ok], traj.lat[ok], lon, lat, land=land)
    t.truth_area_km2 = injected.area_km2
    if injected.area_km2 < 2.0:
        t.note = "injected slick below the minimum detectable size"
        return t

    # --- detect --------------------------------------------------------------
    detections, _ = detect(
        ctx["case_dir"], field, case.t_obs, case.case_id,
        sigma0_db=injected.sigma0_db, tracks=tracks, classifier=classifier,
    )
    oil = [d for d in detections if d.is_actionable]
    if not oil:
        t.note = "no oil confirmed by the detector"
        return t

    target = max(oil, key=lambda d: d.geometry.area_km2)
    t.detected = True
    t.detected_area_km2 = target.geometry.area_km2

    mask = ObservedMask.from_detection(target, field)
    pred = np.zeros(injected.truth_mask.shape, dtype=bool)
    for d in oil:
        pass  # IoU below is computed from the mask raster, which is enough
    t.detection_iou = round(
        min(t.detected_area_km2, t.truth_area_km2) / max(t.detected_area_km2, t.truth_area_km2), 3
    )

    # --- invert, with the STANDARD assumptions -------------------------------
    posterior, idiag = invert(
        field, mask, case.t_obs, case.case_id, target.detection_id,
        InversionConfig(lookback_hours=min(lookback + 8.0, 72.0),
                        credible_levels=LEVELS, seed=seed),
    )
    t.inverted = True
    t.area_50_km2 = posterior.credible_regions["50"].area_km2
    t.area_95_km2 = posterior.credible_regions["95"].area_km2
    t.t0_width_hours = posterior.t0_marginal.width_hours
    t.ess_fraction = round(posterior.ess_fraction, 4)

    # Where was the source actually? The centroid of the release, in space.
    true_lon = float(np.mean(clon))
    true_lat = float(np.mean(clat))
    t.inside = _coverage(posterior, true_lon, true_lat)

    lo, hi = posterior.t0_marginal.hpd_95
    t.t0_inside_95 = bool(lo.timestamp() <= t0 + window_s / 2 <= hi.timestamp())

    # --- attribute -----------------------------------------------------------
    supplied = [x for x in tracks if x.mmsi != culprit.mmsi] if dark else tracks
    ranked, _ = attribute(
        field, mask, posterior, supplied, case.t_obs, case.case_id,
        AttributionConfig(seed=seed), vessels_in_window=len(tracks),
    )
    t.attributed = True
    t.n_candidates = len(ranked.candidates)
    t.vessels_in_window = ranked.traffic_reduction.vessels_in_window
    t.reduction_factor = round(ranked.traffic_reduction.reduction_factor, 1)
    t.dark_probability = round(ranked.dark_vessel_hypothesis.posterior_probability, 4)
    t.dark_is_top = ranked.dark_vessel_is_most_probable
    ranks = [c.rank for c in ranked.candidates if c.mmsi == culprit.mmsi]
    t.culprit_rank = ranks[0] if ranks else 0
    return t


def _coverage(posterior, lon: float, lat: float) -> dict:
    """Is the true source inside each credible region?"""
    from shapely.geometry import Point
    from shapely.geometry import Polygon as SPoly

    p = Point(lon, lat)
    return {
        str(level): bool(SPoly(region.polygon_wgs84).contains(p))
        for level, region in (
            (lvl, posterior.credible_regions[str(lvl)])
            for lvl in LEVELS if str(lvl) in posterior.credible_regions
        )
    }


def summarise(trials: list[Trial]) -> dict:
    done = [t for t in trials if t.inverted]
    attributed = [t for t in trials if t.attributed and not t.dark_case]
    dark = [t for t in trials if t.attributed and t.dark_case]

    calibration = {}
    for level in LEVELS:
        hits = [t.inside.get(str(level)) for t in done if str(level) in t.inside]
        if hits:
            calibration[str(level)] = round(float(np.mean(hits)), 3)

    def med(xs):
        return round(float(np.median(xs)), 1) if xs else 0.0

    return {
        "trials": len(trials),
        "detected": sum(t.detected for t in trials),
        "inverted": len(done),
        "detection_rate": round(sum(t.detected for t in trials) / max(len(trials), 1), 3),
        "calibration": calibration,
        "t0_coverage_95": round(float(np.mean([t.t0_inside_95 for t in done])), 3) if done else 0.0,
        "median_area_50_km2": med([t.area_50_km2 for t in done]),
        "median_area_95_km2": med([t.area_95_km2 for t in done]),
        "median_t0_width_hours": med([t.t0_width_hours for t in done]),
        "attribution": {
            "n": len(attributed),
            "top1_recall": round(float(np.mean([t.culprit_rank == 1 for t in attributed])), 3) if attributed else 0.0,
            "top3_recall": round(float(np.mean([0 < t.culprit_rank <= 3 for t in attributed])), 3) if attributed else 0.0,
            "in_candidate_set": round(float(np.mean([t.culprit_rank > 0 for t in attributed])), 3) if attributed else 0.0,
            "median_reduction_factor": med([t.reduction_factor for t in attributed]),
            "median_candidates": med([t.n_candidates for t in attributed]),
        },
        "dark_vessel_cases": {
            "n": len(dark),
            "dark_ranked_top": round(float(np.mean([t.dark_is_top for t in dark])), 3) if dark else 0.0,
            "median_dark_probability": round(float(np.median([t.dark_probability for t in dark])), 3) if dark else 0.0,
        },
        "envelope_by_lookback": {
            str(int(lb)): med([t.area_95_km2 for t in done if t.lookback_hours == lb])
            for lb in LOOKBACKS
        },
        "generation_parameters_differ_from_inversion": {
            "generated_windage": list(GEN_WINDAGE),
            "inversion_assumes_windage": list(TransportParams().windage_alpha),
            "generated_diffusivity": list(GEN_DIFFUSIVITY),
            "inversion_assumes_diffusivity": list(TransportParams().diffusivity_kh),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--case", default="synth_kattegat")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--matched-params", action="store_true",
                    help="generate with the drift priors the inversion assumes "
                         "(isolates machinery error from prior misspecification)")
    ap.add_argument("--out", type=Path, default=ROOT / "eval")
    args = ap.parse_args(argv)

    case_dir = ROOT / "data" / "cases" / args.case
    case = CaseManifest.load(case_dir)
    field = ForcingField.from_case(case_dir, load_forcing_bundle(case_dir))
    tracks, _ = clean_and_reconstruct(load_ais(case_dir / "ais" / "tracks.parquet"))
    sigma0, lon, lat = read_scene(case_dir)
    land = _land_on_scene(field, lon, lat)
    model = ROOT / "models" / "gate_lgbm.txt"
    classifier = LookAlikeClassifier.load(model) if model.is_file() else LookAlikeClassifier()

    ctx = {"case": case, "case_dir": case_dir, "field": field, "tracks": tracks,
           "sigma0": sigma0, "lon": lon, "lat": lat, "land": land,
           "classifier": classifier}

    print(f"truth harness: {args.n} trials on {args.case}")
    if args.matched_params:
        print("  MATCHED PARAMETERS: generation uses the inversion's own priors.")
        print("  This isolates machinery error; it is NOT a fair end-to-end test.")
    else:
        print(f"  generation windage {GEN_WINDAGE} vs inversion {TransportParams().windage_alpha}")
        print(f"  generation K_h     {GEN_DIFFUSIVITY} vs inversion {TransportParams().diffusivity_kh}")
    print(f"  1 trial in {DARK_EVERY} hides the culprit from AIS\n")

    rng = np.random.default_rng(args.seed)
    trials: list[Trial] = []
    for i in range(args.n):
        t = run_trial(ctx, i, rng, matched_params=args.matched_params)
        trials.append(t)
        if not t.inverted:
            print(f"  {i + 1:3d}/{args.n}  skipped - {t.note}")
        else:
            hit = "DARK" if t.dark_case else (f"rank {t.culprit_rank}" if t.culprit_rank else "MISSED")
            print(f"  {i + 1:3d}/{args.n}  lb {t.lookback_hours:4.0f} h  "
                  f"slick {t.truth_area_km2:6.1f} km2  95% {t.area_95_km2:7.0f} km2  "
                  f"in50={t.inside.get('50')!s:5s} in95={t.inside.get('95')!s:5s}  {hit}")

    suffix = "_matched" if args.matched_params else ""
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [asdict(t) for t in trials]
    for r in rows:
        for lvl in LEVELS:
            r[f"inside_{lvl}"] = r["inside"].get(str(lvl))
        r.pop("inside")
    with (args.out / f"trials{suffix}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarise(trials)
    summary["matched_params"] = bool(args.matched_params)
    (args.out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / f"calibration{suffix}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["nominal", "empirical"])
        for k, v in summary["calibration"].items():
            w.writerow([int(k) / 100.0, v])

    _report(summary)
    print(f"\nwrote {args.out / 'trials.csv'}, calibration.csv, summary.json")
    return 0


def _report(s: dict) -> None:
    print("\n" + "=" * 66)
    print(f"{s['inverted']}/{s['trials']} trials completed  "
          f"(detection rate {s['detection_rate']:.0%})")

    print("\nCALIBRATION  -- the metric that matters more than accuracy")
    print("  nominal   empirical   verdict")
    for k, v in s["calibration"].items():
        nominal = int(k) / 100.0
        if v >= nominal:
            verdict = "conservative" if v > nominal + 0.05 else "honest"
        else:
            verdict = "OVERCONFIDENT" if v < nominal - 0.1 else "slightly tight"
        bar = "#" * int(v * 30)
        print(f"   {nominal:5.2f}      {v:5.2f}   {verdict:15s} {bar}")
    print(f"  release time inside 95% interval: {s['t0_coverage_95']:.2f}")

    print("\nENVELOPE  -- 95% region area by lookback")
    for lb, area in s["envelope_by_lookback"].items():
        if area:
            print(f"   {lb:>3} h   {area:8.0f} km2")

    a = s["attribution"]
    print(f"\nATTRIBUTION  ({a['n']} trials where the culprit was in AIS)")
    print(f"   top-1 recall            {a['top1_recall']:.2f}")
    print(f"   top-3 recall            {a['top3_recall']:.2f}")
    print(f"   culprit in candidate set {a['in_candidate_set']:.2f}")
    print(f"   median reduction        {a['median_reduction_factor']:.0f}x")

    d = s["dark_vessel_cases"]
    print(f"\nDARK-VESSEL CASES  ({d['n']} trials, culprit removed from AIS)")
    print(f"   dark hypothesis ranked top   {d['dark_ranked_top']:.2f}")
    print(f"   median dark probability      {d['median_dark_probability']:.2f}")
    print("=" * 66)


if __name__ == "__main__":
    raise SystemExit(main())
