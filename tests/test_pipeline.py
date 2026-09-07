"""Phase 0 tests for the pipeline harness.

The stages are stubs, but the harness around them is real and is what every
later phase plugs into: stage ordering, skip/resume, fixture mode, seed
override, and the timing log. Getting it wrong would waste time in every
subsequent phase, so it is tested now.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_case import STAGES, main
from src.contracts import CaseManifest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def case_dir(tmp_path: Path) -> Path:
    """An isolated Case folder built from the reference manifest."""
    case = CaseManifest.model_validate_json(
        (FIXTURES / "case.json").read_text(encoding="utf-8")
    )
    return Path(case.save(tmp_path / case.case_id)).parent


def test_case_folder_has_the_standard_layout(case_dir: Path):
    for sub in ("scene", "forcing", "ais", "out"):
        assert (case_dir / sub).is_dir(), f"missing {sub}/"


def test_stub_run_reports_failure_not_silence(case_dir: Path, capsys):
    """A stage that cannot run must be loudly visible, never a silent no-op.

    The fixture case carries a manifest but no scene, forcing or AIS files, so
    the implemented stages hit missing inputs and the unimplemented ones report
    as stubs. Both must surface as a readable line and a non-zero exit -- this
    runs on a stage in front of judges, where a traceback is not an option.
    """
    assert main(["--case", str(case_dir)]) == 1
    out = capsys.readouterr().out
    assert "MISSING INPUT" in out or "STUB" in out


def test_fixtures_mode_populates_out(case_dir: Path):
    """This is what lets the frontend work from day one without the pipeline."""
    assert main(["--case", str(case_dir), "--fixtures"]) == 0
    out = case_dir / "out"
    for produced in ("detections.json", "posterior.json", "candidates.json"):
        assert (out / produced).is_file(), f"missing out/{produced}"
        json.loads((out / produced).read_text(encoding="utf-8"))


def test_completed_stages_are_skipped_on_rerun(case_dir: Path, capsys):
    main(["--case", str(case_dir), "--fixtures"])
    capsys.readouterr()
    main(["--case", str(case_dir)])
    captured = capsys.readouterr().out
    assert "skip - up to date" in captured
    # The two stages with no fixture are still stubs, so they must still report.
    assert "STUB" in captured


def test_force_recomputes(case_dir: Path, capsys):
    main(["--case", str(case_dir), "--fixtures"])
    capsys.readouterr()
    main(["--case", str(case_dir), "--fixtures", "--force"])
    assert "skip" not in capsys.readouterr().out


def test_single_stage_selection(case_dir: Path, capsys):
    main(["--case", str(case_dir), "--stage", "invert", "--fixtures"])
    captured = capsys.readouterr().out
    assert "[invert" in captured
    assert "[detect" not in captured


def test_unknown_stage_is_an_error(case_dir: Path):
    assert main(["--case", str(case_dir), "--stage", "nonsense"]) == 2


def test_missing_case_is_an_error(tmp_path: Path):
    assert main(["--case", str(tmp_path / "does_not_exist")]) == 2


def test_seed_override_is_reported(case_dir: Path, capsys):
    """NFR-3: the seed is part of the reproducible input and must be visible."""
    main(["--case", str(case_dir), "--seed", "12345", "--fixtures"])
    assert "seed 12345" in capsys.readouterr().out


def test_stage_registry_is_coherent():
    names = [s.name for s in STAGES]
    assert names == ["detect", "invert", "forecast", "attribute", "particles"]
    assert len(set(names)) == len(names), "stage names must be unique"
    outputs = [s.output for s in STAGES]
    assert len(set(outputs)) == len(outputs), "stage outputs must be unique"
    for stage in STAGES:
        assert stage.owner, f"stage {stage.name} has no owner"
        assert stage.run.__doc__, f"stage {stage.name} has no docstring"


def test_synthetic_ais_is_disclosed_in_output(tmp_path: Path, capsys):
    """Disclosed synthetic data is a methodological choice; discovered synthetic
    data is a credibility collapse. The flag must surface, not hide."""
    case = CaseManifest.model_validate_json(
        (FIXTURES / "case.json").read_text(encoding="utf-8")
    )
    synthetic = case.model_copy(update={"ais_is_synthetic": True, "ais_source": "synthetic"})
    synthetic.save(tmp_path / "synth")
    main(["--case", str(tmp_path / "synth"), "--fixtures"])
    assert "SYNTHETIC" in capsys.readouterr().out
