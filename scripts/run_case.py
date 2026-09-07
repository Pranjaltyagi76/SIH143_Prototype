"""Execute the pipeline over one Case folder.

    python scripts/run_case.py --case data/cases/kattegat_2024_03_11

The system is a batch pipeline over immutable Case folders, not a service. Every
stage reads and writes contract objects and nothing else, so the six workstreams
integrate by construction rather than by negotiation.

Phase 0 status: every stage is a stub. The skeleton, the stage ordering, the
skip/resume logic and the timing log are real and working, so stages can be
filled in one at a time without touching the harness.

Useful now:

    --fixtures    populate out/ from tests/fixtures instead of computing.
                  This gives the frontend a complete, valid out/ folder on day
                  one, so UI work never blocks on the pipeline.
    --stage NAME  run a single stage
    --force       recompute even if outputs are up to date
    --seed N      override the case seed (NFR-3: same case + seed = same output)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import CaseManifest  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


# --------------------------------------------------------------------- context


@dataclass
class RunContext:
    """Everything a stage is allowed to see."""

    case_dir: Path
    case: CaseManifest
    seed: int
    force: bool
    use_fixtures: bool

    @property
    def out_dir(self) -> Path:
        return self.case_dir / "out"


# ---------------------------------------------------------------------- stages
#
# Each stage declares the artefact it produces. That declaration drives both the
# skip/resume logic and the --fixtures shortcut, so adding a stage means adding
# one entry here and nothing else.


def stage_detect(ctx: RunContext) -> None:
    """Stages 1-3: segmentation, physics gate, characterisation.  Owner: M1.

    Emits list[SlickDetection]. Stage 1 is tuned for recall; precision is the
    physics gate's job. The gate is a hard override -- outside 3-12 m/s the
    result is 'undetermined' regardless of what the classifier produced.
    """
    import json

    from src.attribution import clean_and_reconstruct, load_ais
    from src.detect import detect
    from src.ingest.case_builder import load_forcing_bundle
    from src.transport import ForcingField

    field = ForcingField.from_case(ctx.case_dir, load_forcing_bundle(ctx.case_dir))
    ais_path = ctx.case_dir / "ais" / "tracks.parquet"
    tracks = None
    if ais_path.is_file():
        tracks, _ = clean_and_reconstruct(load_ais(ais_path))

    detections, diag = detect(
        ctx.case_dir, field, ctx.case.t_obs, ctx.case.case_id, tracks=tracks
    )
    body = ",\n".join(d.model_dump_json(indent=2) for d in detections)
    (ctx.out_dir / "detections.json").write_text(f"[\n{body}\n]\n", encoding="utf-8")
    log_event(ctx, {"stage": "detect", **diag.__dict__})


def stage_invert(ctx: RunContext) -> None:
    """Stage 5: Bayesian source inversion.  Owner: M2.

    Backward advection-only proposal, then a forward ensemble conditioned on
    reproducing the observed mask. Physics only ever runs forward: turbulent
    diffusion is not time-reversible, so backward integration narrows the
    search and forward simulation computes the answer.

    Emits SourcePosterior plus the density .npz.
    """
    from src.contracts import SlickDetection
    from src.ingest.case_builder import load_forcing_bundle
    from src.inversion import InversionConfig, ObservedMask, invert
    from src.transport import ForcingField

    raw = json.loads((ctx.out_dir / "detections.json").read_text(encoding="utf-8"))
    detections = [SlickDetection.model_validate(d) for d in raw]
    oil = [d for d in detections if d.is_actionable]
    if not oil:
        # Not a failure. A scene where every dark patch was a look-alike or fell
        # outside the detectability window has nothing to invert, and saying so
        # is the correct output.
        (ctx.out_dir / "posterior.json").write_text(
            json.dumps({"case_id": ctx.case.case_id, "detections_actionable": 0,
                        "note": "no confirmed oil in this scene; nothing to invert"},
                       indent=2), encoding="utf-8")
        return

    field = ForcingField.from_case(ctx.case_dir, load_forcing_bundle(ctx.case_dir))
    target = max(oil, key=lambda d: d.geometry.area_km2)
    mask = ObservedMask.from_detection(target, field)
    posterior, diag = invert(
        field, mask, ctx.case.t_obs, ctx.case.case_id, target.detection_id,
        InversionConfig(seed=ctx.seed), out_dir=ctx.out_dir,
    )
    (ctx.out_dir / "posterior.json").write_text(
        posterior.model_dump_json(indent=2), encoding="utf-8")
    log_event(ctx, {"stage": "invert", **{k: v for k, v in diag.__dict__.items()
                                          if isinstance(v, (int, float, str, bool))}})


def stage_forecast(ctx: RunContext) -> None:
    """Stage 6: forward prediction at +6/+12/+24 h.  Owner: M2.

    Reuses the same kernel and the same per-particle parameter sampling.
    """
    raise NotImplementedError("Phase 3: forecast (M2)")


def stage_attribute(ctx: RunContext) -> None:
    """Stage 7: vessel attribution.  Owner: M4.

    Each candidate vessel's AIS track is a generative hypothesis: seed particles
    along the track it actually sailed, forward-simulate, and score against the
    observed mask with the same observation operator used by the inversion.
    Normalised against an explicit dark-vessel hypothesis.

    Emits RankedCandidates.
    """
    from src.attribution import AttributionConfig, attribute, clean_and_reconstruct, load_ais
    from src.contracts import SlickDetection, SourcePosterior
    from src.ingest.case_builder import load_forcing_bundle
    from src.inversion import ObservedMask
    from src.transport import ForcingField

    raw_post = json.loads((ctx.out_dir / "posterior.json").read_text(encoding="utf-8"))
    if "credible_regions" not in raw_post:
        print("           no posterior to attribute against; skipping")
        return
    posterior = SourcePosterior.model_validate(raw_post)

    raw = json.loads((ctx.out_dir / "detections.json").read_text(encoding="utf-8"))
    target = next(
        d for d in (SlickDetection.model_validate(x) for x in raw)
        if d.detection_id == posterior.detection_id
    )

    field = ForcingField.from_case(ctx.case_dir, load_forcing_bundle(ctx.case_dir))
    mask = ObservedMask.from_detection(target, field)
    tracks, cleaning = clean_and_reconstruct(load_ais(ctx.case_dir / "ais" / "tracks.parquet"))

    ranked, diag = attribute(
        field, mask, posterior, tracks, ctx.case.t_obs, ctx.case.case_id,
        AttributionConfig(seed=ctx.seed), cleaning=cleaning.as_dict(),
        vessels_in_window=cleaning.vessels_out,
    )
    (ctx.out_dir / "candidates.json").write_text(
        ranked.model_dump_json(indent=2), encoding="utf-8")
    log_event(ctx, {"stage": "attribute", **{k: v for k, v in diag.__dict__.items()
                                             if isinstance(v, (int, float, str, bool))}})


def stage_export(ctx: RunContext) -> None:
    """Build the static bundle the interface renders.  Owner: M5/M6.

    The full ensemble stays in .npz; the page gets a scene PNG, a vectorised
    coastline, weighted posterior cells and ~2k particle trails.
    """
    from src.api.export import export_ui
    from src.ingest.case_builder import load_forcing_bundle
    from src.transport import ForcingField

    field = ForcingField.from_case(ctx.case_dir, load_forcing_bundle(ctx.case_dir))
    diag = export_ui(ctx.case_dir, field, ctx.case, seed=ctx.seed)
    log_event(ctx, {"stage": "export", **{k: v for k, v in diag.__dict__.items()
                                          if isinstance(v, (int, float, str, bool))}})


@dataclass(frozen=True)
class Stage:
    name: str
    output: str
    fixture: str | None
    owner: str
    run: Callable[[RunContext], None]


STAGES: tuple[Stage, ...] = (
    Stage("detect", "detections.json", "slick_detections.json", "M1", stage_detect),
    Stage("invert", "posterior.json", "source_posterior.json", "M2", stage_invert),
    Stage("forecast", "forecast.json", None, "M2", stage_forecast),
    Stage("attribute", "candidates.json", "ranked_candidates.json", "M4", stage_attribute),
    Stage("export", "ui/manifest.json", None, "M5", stage_export),
)


# ------------------------------------------------------------------ harness


def is_up_to_date(ctx: RunContext, stage: Stage) -> bool:
    """A stage is skipped when its output is newer than the case manifest."""
    out = ctx.out_dir / stage.output
    if not out.is_file():
        return False
    return out.stat().st_mtime >= (ctx.case_dir / "case.json").stat().st_mtime


def copy_fixture(ctx: RunContext, stage: Stage) -> bool:
    if stage.fixture is None:
        return False
    src = FIXTURES / stage.fixture
    if not src.is_file():
        return False
    shutil.copyfile(src, ctx.out_dir / stage.output)
    return True


def log_event(ctx: RunContext, record: dict) -> None:
    """Structured timings to out/log.jsonl.

    Written from Phase 0 so the performance data accumulates for free across the
    whole sprint, rather than being reconstructed under pressure on Day 11.
    """
    with (ctx.out_dir / "log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def run(ctx: RunContext, only: str | None) -> int:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    selected = [s for s in STAGES if only is None or s.name == only]
    if only and not selected:
        print(f"error: unknown stage {only!r}; choose from {[s.name for s in STAGES]}")
        return 2

    failures = 0
    for stage in selected:
        label = f"[{stage.name:<10}] ({stage.owner})"

        if not ctx.force and is_up_to_date(ctx, stage):
            print(f"{label} skip - up to date")
            continue

        if ctx.use_fixtures:
            if copy_fixture(ctx, stage):
                print(f"{label} fixture -> out/{stage.output}")
            else:
                print(f"{label} no fixture available")
            continue

        started = time.perf_counter()
        try:
            stage.run(ctx)
        except NotImplementedError as exc:
            print(f"{label} STUB - {exc}")
            failures += 1
            continue
        except FileNotFoundError as exc:
            # A Case missing an input should say which file, not emit a
            # traceback. This runs on a stage in front of judges.
            print(f"{label} MISSING INPUT - {exc}")
            failures += 1
            continue
        elapsed = time.perf_counter() - started

        log_event(ctx, {"stage": stage.name, "seconds": round(elapsed, 3), "seed": ctx.seed})
        print(f"{label} done in {elapsed:.2f}s -> out/{stage.output}")

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, type=Path, help="path to a Case folder")
    ap.add_argument("--stage", default=None, help=f"run one of: {[s.name for s in STAGES]}")
    ap.add_argument("--force", action="store_true", help="recompute even if up to date")
    ap.add_argument("--seed", type=int, default=None, help="override the case seed")
    ap.add_argument(
        "--fixtures",
        action="store_true",
        help="populate out/ from tests/fixtures instead of computing",
    )
    args = ap.parse_args(argv)

    try:
        case = CaseManifest.load(args.case)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2

    ctx = RunContext(
        case_dir=args.case,
        case=case,
        seed=args.seed if args.seed is not None else case.seed,
        force=args.force,
        use_fixtures=args.fixtures,
    )

    print(f"case   {case.case_id}  ({case.region})")
    print(f"t_obs  {case.t_obs.isoformat()}   lookback {case.max_lookback_hours:.0f} h")
    print(f"crs    {case.crs_working}   seed {ctx.seed}")
    if case.ais_is_synthetic:
        print("note   AIS for this case is SYNTHETIC (disclosed)")
    print()

    rc = run(ctx, args.stage)

    if rc:
        print("\nsome stages are still stubs - expected during Phase 0.")
        print("run with --fixtures to populate out/ for frontend work.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
