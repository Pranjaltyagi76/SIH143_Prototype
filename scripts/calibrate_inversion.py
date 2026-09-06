"""Measure whether the inversion's stated uncertainty is honest.

    python scripts/calibrate_inversion.py --n 20
    python scripts/calibrate_inversion.py --n 12 --sweep

**Calibration, not accuracy, is the headline metric.** If we say 95% and the
true source falls inside the 95% region 95% of the time, our uncertainty means
something. A tight region that misses is worse than a wide one that contains,
because a confident wrong answer is what puts an innocent ship in front of an
investigator.

Each trial releases oil from a known position at a known time, drifts it forward
with the real kernel, converts the resulting cloud into an observed slick, then
throws the ground truth away and asks the inversion to recover it.

This is deliberately circular in one respect and we say so: the simulator both
generates the observation and evaluates the hypotheses, so it tests the
inversion machinery, not the physics. Validating the physics needs real drifter
trajectories with no simulator in the loop. What it *can* test -- and what
nothing else can -- is whether the reported credible regions have the coverage
they claim.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import CaseManifest  # noqa: E402
from src.ingest.case_builder import load_forcing_bundle  # noqa: E402
from src.inversion import InversionConfig, ObservedMask, invert  # noqa: E402
from src.transport import ForcingField, Seeds, TransportParams, simulate  # noqa: E402

DEFAULT_CASE = ROOT / "data" / "cases" / "synth_kattegat"


@dataclass
class Trial:
    ok: bool
    inside_50: bool = False
    inside_95: bool = False
    t0_inside: bool = False
    area_50: float = 0.0
    area_95: float = 0.0
    t0_width_h: float = 0.0
    ess_fraction: float = 0.0
    slick_km2: float = 0.0
    note: str = ""


def run_trial(
    field: ForcingField,
    case: CaseManifest,
    rng: np.random.Generator,
    lookback_h: float,
    correlation_m: float | None,
    seed: int,
) -> Trial:
    from shapely.geometry import Point
    from shapely.geometry import Polygon as SPoly

    # --- ground truth: a release we will pretend not to know -----------------
    true_lon = float(rng.uniform(case.aoi.min_lon + 0.4, case.aoi.max_lon - 0.7))
    true_lat = float(rng.uniform(case.aoi.min_lat + 0.3, case.aoi.max_lat - 0.6))
    t0_true = case.t_obs - timedelta(hours=lookback_h)

    n = 3000
    seeds = Seeds.at_time(
        true_lon + rng.normal(0, 0.010, n), true_lat + rng.normal(0, 0.008, n), t0_true
    )
    try:
        truth = simulate(field, seeds, t0_true, case.t_obs, TransportParams(), seed=seed)
    except ValueError as exc:
        return Trial(ok=False, note=str(exc)[:60])

    usable = truth.usable
    if usable.sum() < 0.5 * n:
        return Trial(ok=False, note="most particles beached or left the domain")

    # Threshold the particle density, do NOT take a convex hull: a hull is far
    # larger than the plume it encloses and rewards hypotheses that over-spread.
    try:
        mask = ObservedMask.from_points(truth.lon[usable], truth.lat[usable], field)
    except ValueError as exc:
        return Trial(ok=False, note=str(exc)[:60])

    # --- forget the truth, invert -------------------------------------------
    try:
        posterior, diag = invert(
            field, mask, case.t_obs, case.case_id, f"{case.case_id}:cal",
            InversionConfig(
                lookback_hours=lookback_h,
                likelihood_correlation_m=correlation_m,
                seed=seed,
            ),
        )
    except ValueError as exc:
        return Trial(ok=False, note=str(exc)[:60])

    truth_point = Point(true_lon, true_lat)
    lo, hi = posterior.t0_marginal.hpd_95
    return Trial(
        ok=True,
        inside_50=SPoly(posterior.credible_regions["50"].polygon_wgs84).contains(truth_point),
        inside_95=SPoly(posterior.credible_regions["95"].polygon_wgs84).contains(truth_point),
        t0_inside=lo <= t0_true <= hi,
        area_50=posterior.credible_regions["50"].area_km2,
        area_95=posterior.credible_regions["95"].area_km2,
        t0_width_h=posterior.t0_marginal.width_hours,
        ess_fraction=diag.ess_fraction,
        slick_km2=mask.area_km2,
    )


def summarise(trials: list[Trial], label: str) -> dict:
    good = [t for t in trials if t.ok]
    if not good:
        print(f"  {label:>26}: no usable trials")
        return {}
    row = {
        "label": label,
        "n": len(good),
        "cov50": np.mean([t.inside_50 for t in good]),
        "cov95": np.mean([t.inside_95 for t in good]),
        "covt0": np.mean([t.t0_inside for t in good]),
        "a50": np.median([t.area_50 for t in good]),
        "a95": np.median([t.area_95 for t in good]),
        "tw": np.median([t.t0_width_h for t in good]),
        "ess": np.median([t.ess_fraction for t in good]),
    }
    print(
        f"  {label:>26}: n={row['n']:2d}  cov50={row['cov50']:5.0%}  cov95={row['cov95']:5.0%}"
        f"  covT0={row['covt0']:5.0%}  area95={row['a95']:7.0f} km2"
        f"  t0width={row['tw']:4.1f} h  ESS={row['ess']:5.1%}"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=DEFAULT_CASE)
    ap.add_argument("--n", type=int, default=20, help="trials per setting")
    ap.add_argument("--lookback", type=float, default=24.0)
    ap.add_argument("--sweep", action="store_true", help="sweep the decorrelation length")
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args(argv)

    case = CaseManifest.load(args.case)
    field = ForcingField.from_case(args.case, load_forcing_bundle(args.case))

    print(f"case {case.case_id}   lookback {args.lookback:.0f} h   {args.n} trials per setting")
    print("target: cov95 >= 90%. A region that is tight but misses is worse than")
    print("a wide one that contains -- a confident wrong answer is the failure mode.\n")

    if args.sweep:
        settings = [
            ("none (every cell independent)", 0.0),
            ("2 km", 2_000.0),
            ("5 km", 5_000.0),
            ("9 km (forcing resolution)", 9_000.0),
            ("15 km", 15_000.0),
        ]
    else:
        settings = [("default", None)]

    rows = []
    for label, correlation in settings:
        trials = []
        rng = np.random.default_rng(args.seed)
        for i in range(args.n):
            trials.append(run_trial(field, case, rng, args.lookback, correlation, args.seed + i))
        row = summarise(trials, label)
        if row:
            rows.append(row)

    if args.sweep and rows:
        best = min(rows, key=lambda r: abs(r["cov95"] - 0.95) + 0.3 * abs(r["cov50"] - 0.50))
        print(f"\nbest calibrated setting: {best['label']}")
        print("  report this number honestly, including if it is poor -- the calibration")
        print("  curve is the point, not the accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
