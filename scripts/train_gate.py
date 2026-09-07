"""Train the Stage 2 look-alike classifier.

    python scripts/train_gate.py --trials 24

Training data is generated, not collected: slicks are injected into scenes at
random places and times, the real segmenter is run over the result, and each
resulting patch is labelled oil or look-alike by its overlap with the injected
truth. Negatives therefore come from whatever the segmenter *actually* gets
wrong, rather than from a hand-made caricature of a look-alike.

Two protocol choices matter.

**Geographic holdout, never a random split.**
Train on one region, test on another, and report the drop. Patches from the same
scene share speckle statistics, sea state and incidence geometry; a random split
leaks between train and test and inflates the number by an amount nobody can
estimate afterwards (W-05).

**Patches outside the wind window are excluded from training entirely.**
They are decided by the hard physics gate, which the classifier cannot overturn,
so training on them would teach the model to reproduce a rule it is not allowed
to influence -- and would inflate its apparent accuracy with free wins.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attribution import clean_and_reconstruct, load_ais  # noqa: E402
from src.contracts import CaseManifest  # noqa: E402
from src.contracts.detection import WIND_MAX_MS, WIND_MIN_MS  # noqa: E402
from src.detect import inject, read_scene  # noqa: E402
from src.detect.gate import FEATURE_NAMES, extract_features, lane_density  # noqa: E402
from src.detect.pipeline import _land_on_scene, _wind_on_scene  # noqa: E402
from src.detect.segment import cell_area_km2, segment_dark_patches  # noqa: E402
from src.ingest.case_builder import load_forcing_bundle  # noqa: E402
from src.transport import ForcingField, Seeds, TransportParams, simulate  # noqa: E402

# A patch overlapping this much of the injected slick is oil. Below it, the
# patch is something the segmenter found that is not the slick.
OIL_OVERLAP = 0.30


def harvest(case_dir: Path, trials: int, seed: int = 0):
    """Generate labelled patches from injected slicks in one case."""
    rng = np.random.default_rng(seed)
    case = CaseManifest.load(case_dir)
    field = ForcingField.from_case(case_dir, load_forcing_bundle(case_dir))
    tracks, _ = clean_and_reconstruct(load_ais(case_dir / "ais" / "tracks.parquet"))

    sigma0, lon, lat = read_scene(case_dir)
    incidence, _, _ = read_scene(case_dir, "incidence")
    land = _land_on_scene(field, lon, lat)
    wind_u, wind_v = _wind_on_scene(field, lon, lat, case.t_obs)
    cell_km2 = cell_area_km2(lon, lat)
    lanes = lane_density(
        np.concatenate([t.lon for t in tracks]),
        np.concatenate([t.lat for t in tracks]),
        lon, lat,
    )
    sea = ~land.astype(bool)
    background_db = float(np.median(sigma0[sea]))

    X, y = [], []
    for trial in range(trials):
        # A discharge along a real vessel track, at a random lookback.
        lookback = float(rng.uniform(8.0, 30.0))
        window = float(rng.uniform(1.0, 4.0)) * 3600.0
        t0 = case.t_obs.timestamp() - lookback * 3600.0
        usable_tracks = [t for t in tracks if t.covers(t0) and t.covers(t0 + window) and len(t) > 120]
        if not usable_tracks:
            continue
        track = usable_tracks[int(rng.integers(len(usable_tracks)))]

        n = int(rng.integers(3000, 9000))
        times = t0 + rng.random(n) * window
        clon, clat, _ = track.interpolate(times)
        traj = simulate(
            field,
            Seeds(lon=clon, lat=clat, seed_time=times, origin_marker=np.zeros(n, "int64")),
            datetime.fromtimestamp(t0, tz=timezone.utc), case.t_obs,
            TransportParams(), seed=int(rng.integers(1 << 30)),
        )
        ok = traj.usable
        if ok.sum() < 500:
            continue

        injected = inject(sigma0, traj.lon[ok], traj.lat[ok], lon, lat, land=land)
        labels, patches = segment_dark_patches(injected.sigma0_db, lon, lat, land=land)

        for patch in patches:
            feats = extract_features(
                patch.pixels, injected.sigma0_db, incidence, lon, lat,
                wind_u, wind_v, lanes, cell_km2, background_db,
            )
            # The hard gate decides these; the classifier never sees them.
            if not (WIND_MIN_MS <= feats.wind_speed_ms <= WIND_MAX_MS):
                continue
            hit = injected.truth_mask[patch.pixels[:, 0], patch.pixels[:, 1]].mean()
            X.append(feats.vector())
            y.append(1 if hit >= OIL_OVERLAP else 0)

        print(f"  trial {trial + 1:2d}/{trials}  lookback {lookback:4.1f} h  "
              f"slick {injected.area_km2:6.1f} km2  patches {len(patches):2d}  "
              f"labelled {len(y)}")

    return np.asarray(X, dtype="float64"), np.asarray(y, dtype="int32")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=24, help="injections in the training region")
    ap.add_argument("--holdout-trials", type=int, default=10)
    ap.add_argument("--train-case", default="synth_kattegat")
    ap.add_argument("--holdout-case", default="synth_kutch")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "gate_lgbm.txt")
    args = ap.parse_args(argv)

    cases = ROOT / "data" / "cases"
    print(f"harvesting training patches from {args.train_case} ...")
    Xtr, ytr = harvest(cases / args.train_case, args.trials, seed=1)
    print(f"\nharvesting HOLDOUT patches from {args.holdout_case} (different region) ...")
    Xte, yte = harvest(cases / args.holdout_case, args.holdout_trials, seed=2)

    if Xtr.size == 0 or Xte.size == 0:
        print("error: no labelled patches harvested")
        return 1

    print(f"\ntrain {len(ytr)} patches ({ytr.sum()} oil, {len(ytr) - ytr.sum()} look-alike)")
    print(f"holdout {len(yte)} patches ({yte.sum()} oil, {len(yte) - yte.sum()} look-alike)")

    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=220, learning_rate=0.06, num_leaves=12,
        min_child_samples=6, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=1.0, verbose=-1, random_state=0,
    )
    model.fit(Xtr, ytr, feature_name=list(FEATURE_NAMES))

    def report(X, y, label):
        if not len(y):
            return
        p = model.predict_proba(X)[:, 1]
        pred = p >= 0.5
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        neg = max(fp + tn, 1)
        print(f"\n{label}:")
        print(f"  accuracy         {(tp + tn) / len(y):.3f}")
        print(f"  recall (oil)     {tp / max(tp + fn, 1):.3f}")
        print(f"  FALSE POSITIVE   {fp / neg:.3f}   <- the headline number for Stage 2")
        print(f"  specificity      {tn / neg:.3f}")
        print(f"  confusion        tp={tp} fp={fp} tn={tn} fn={fn}")
        return fp / neg

    fpr_in = report(Xtr, ytr, "in-domain (train region)")
    fpr_out = report(Xte, yte, f"CROSS-REGION HOLDOUT ({args.holdout_case})")
    if fpr_in is not None and fpr_out is not None:
        print(f"\ncross-region FPR change: {fpr_in:.3f} -> {fpr_out:.3f} "
              f"({fpr_out - fpr_in:+.3f}). Reporting the drop is the point.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.out))
    print(f"\nsaved {args.out}")

    gains = model.booster_.feature_importance(importance_type="gain")
    print("\nfeature importance (gain) -- this goes on a slide:")
    for name, g in sorted(zip(FEATURE_NAMES, gains), key=lambda kv: -kv[1]):
        bar = "#" * int(40 * g / max(gains.max(), 1e-9))
        print(f"  {name:26s} {g:9.1f}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
