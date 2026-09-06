"""Generate a synthetic Case folder.

    python scripts/build_synthetic_case.py --case synth_kattegat
    python scripts/build_synthetic_case.py --all

Phase 1. Everything is analytic, so this needs no network and no credentials.
Phases 2 to 7 are built and tested entirely against these cases; real data slots
in at Phase 8 behind the same contracts and the same folder layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.case_builder import NAMED_CASES, build_synthetic_case  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "cases"


def summarise(case_dir: Path) -> None:
    import pandas as pd

    from src.contracts import CaseManifest
    from src.ingest.case_builder import load_forcing_bundle

    case = CaseManifest.load(case_dir)
    bundle = load_forcing_bundle(case_dir)
    ais = pd.read_parquet(case_dir / "ais" / "tracks.parquet")

    total_mb = sum(p.stat().st_size for p in case_dir.rglob("*") if p.is_file()) / 1e6

    print(f"  region        {case.region}")
    print(f"  aoi           {case.aoi.min_lon}..{case.aoi.max_lon} E, "
          f"{case.aoi.min_lat}..{case.aoi.max_lat} N   ({case.crs_working})")
    print(f"  t_obs         {case.t_obs.isoformat()}")
    print(f"  lookback      {case.max_lookback_hours:.0f} h")
    print(f"  currents      {bundle.currents.resolution_deg:.4f} deg, "
          f"stokes_merged={bundle.currents.includes_stokes}")
    print(f"  wind          {bundle.wind.resolution_deg:.2f} deg")
    print(f"  ais           {len(ais):,} messages from {ais['mmsi'].nunique()} vessels")
    print(f"  ais synthetic {case.ais_is_synthetic}")
    print(f"  size on disk  {total_mb:.1f} MB")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=sorted(NAMED_CASES), help="which case to build")
    ap.add_argument("--all", action="store_true", help="build every named case")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="output root")
    args = ap.parse_args(argv)

    if not args.case and not args.all:
        ap.error("give --case NAME or --all")

    names = sorted(NAMED_CASES) if args.all else [args.case]
    for name in names:
        print(f"\nbuilding {name} ...")
        case_dir = build_synthetic_case(NAMED_CASES[name], args.root)
        summarise(case_dir)
        print(f"  -> {case_dir}")

    print("\nnext:")
    print(f"  python scripts/run_case.py --case {args.root / names[0]} --fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
