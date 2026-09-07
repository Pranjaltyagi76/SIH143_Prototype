"""Phase 7 tests for the evaluation harness.

These test the *harness*, not the pipeline. That distinction matters: the
harness is the only source of numbers for the inversion and the attribution
engine, so a harness that quietly flatters the system would corrupt every
metric we report and there would be nothing left to catch it.

So the tests here are mostly about the anti-cheating rules holding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.truth_harness import (
    DARK_EVERY,
    GEN_DIFFUSIVITY,
    GEN_WINDAGE,
    LEVELS,
    LOOKBACKS,
    Trial,
    summarise,
)
from src.transport import TransportParams

EVAL = Path(__file__).resolve().parents[1] / "eval"


# ---------------------------------------------------- the anti-cheating rules


def test_generation_parameters_differ_from_the_inversions_priors():
    """Without this the run would only prove our simulator agrees with itself.

    The generator must be able to produce drift the inversion's prior does not
    cover, in both directions, or the calibration figure is meaningless.
    """
    assumed = TransportParams()
    assert GEN_WINDAGE[0] < assumed.windage_alpha[0], "generator cannot go below the prior"
    assert GEN_WINDAGE[1] > assumed.windage_alpha[1], "generator cannot go above the prior"
    assert GEN_DIFFUSIVITY[0] < assumed.diffusivity_kh[0]
    assert GEN_DIFFUSIVITY[1] > assumed.diffusivity_kh[1]


def test_dark_cases_are_a_meaningful_fraction():
    """A system that names a vessel when the real polluter was not transmitting
    is broken in the most damaging way available to it, and nothing except these
    trials tests for it."""
    assert 2 <= DARK_EVERY <= 5
    flags = [(i % DARK_EVERY) == DARK_EVERY - 1 for i in range(24)]
    assert 0.15 < np.mean(flags) < 0.35


def test_lookback_is_swept_so_the_envelope_is_measured():
    assert len(LOOKBACKS) >= 4
    assert min(LOOKBACKS) <= 12.0 and max(LOOKBACKS) >= 48.0


def test_calibration_levels_include_the_contract_levels_and_intermediates():
    """50 and 95 are required by the contract; the rest make it a curve."""
    assert 50 in LEVELS and 95 in LEVELS
    assert len(LEVELS) >= 4


# ------------------------------------------------------------- summarising


def _trial(**kw) -> Trial:
    base = dict(trial=0, seed=1, lookback_hours=24.0, window_hours=2.0, dark_case=False)
    return Trial(**{**base, **kw})


def test_summary_separates_dark_cases_from_named_attribution():
    """Scoring a dark case for 'culprit rank' would be nonsense: the culprit was
    deliberately removed from the candidate set."""
    trials = [
        _trial(inverted=True, attributed=True, culprit_rank=1, inside={"95": True}),
        _trial(inverted=True, attributed=True, culprit_rank=4, inside={"95": True}),
        _trial(dark_case=True, inverted=True, attributed=True, dark_is_top=True,
               dark_probability=0.4, inside={"95": True}),
    ]
    s = summarise(trials)
    assert s["attribution"]["n"] == 2
    assert s["dark_vessel_cases"]["n"] == 1
    assert s["attribution"]["top1_recall"] == 0.5
    assert s["dark_vessel_cases"]["dark_ranked_top"] == 1.0


def test_summary_counts_a_missed_culprit_as_a_miss_not_a_skip():
    """rank 0 means the culprit was not in the candidate set at all. Dropping
    those rows would silently inflate recall."""
    trials = [
        _trial(inverted=True, attributed=True, culprit_rank=0, inside={"95": False}),
        _trial(inverted=True, attributed=True, culprit_rank=1, inside={"95": True}),
    ]
    s = summarise(trials)
    assert s["attribution"]["n"] == 2
    assert s["attribution"]["top3_recall"] == 0.5
    assert s["attribution"]["in_candidate_set"] == 0.5


def test_summary_records_that_the_priors_were_mismatched():
    """The reader must be able to tell which mode produced a number."""
    s = summarise([_trial(inverted=True, inside={"95": True})])
    gen = s["generation_parameters_differ_from_inversion"]
    assert gen["generated_windage"] != gen["inversion_assumes_windage"]


def test_calibration_is_computed_per_level():
    trials = [
        _trial(inverted=True, inside={"50": True, "95": True}),
        _trial(inverted=True, inside={"50": False, "95": True}),
        _trial(inverted=True, inside={"50": False, "95": False}),
    ]
    s = summarise(trials)
    assert s["calibration"]["50"] == pytest.approx(1 / 3, abs=0.01)
    assert s["calibration"]["95"] == pytest.approx(2 / 3, abs=0.01)


def test_summary_survives_an_all_skipped_run():
    s = summarise([_trial(note="no oil confirmed by the detector")])
    assert s["inverted"] == 0
    assert s["attribution"]["top3_recall"] == 0.0


# ------------------------------------------------------ recorded results


@pytest.fixture(scope="module")
def results():
    if not (EVAL / "summary.json").is_file():
        pytest.skip("run scripts/truth_harness.py first")
    return json.loads((EVAL / "summary.json").read_text(encoding="utf-8"))


def test_the_prefilter_never_drops_the_true_culprit(results):
    """The point of filtering is to remove irrelevant traffic, not the answer.
    This is the one attribution number that must not degrade."""
    assert results["attribution"]["in_candidate_set"] == 1.0


def test_dark_vessel_hypothesis_wins_when_the_culprit_is_absent(results):
    """The single most important behaviour in the system."""
    d = results["dark_vessel_cases"]
    assert d["n"] > 0
    assert d["dark_ranked_top"] >= 0.9


def test_traffic_reduction_meets_the_headline_target(results):
    assert results["attribution"]["median_reduction_factor"] >= 50.0


def test_the_envelope_grows_with_lookback(results):
    """Uncertainty must grow as you look further back. A flat or shrinking
    envelope would mean the inversion is not using the physics at all."""
    env = {int(k): v for k, v in results["envelope_by_lookback"].items() if v}
    keys = sorted(env)
    assert len(keys) >= 3
    assert env[keys[-1]] > env[keys[0]], f"envelope did not grow: {env}"


def test_low_credible_levels_are_honestly_calibrated(results):
    """The 50% region should contain the truth about half the time."""
    assert abs(results["calibration"]["50"] - 0.50) <= 0.25


def test_tail_miscalibration_is_recorded_not_hidden(results):
    """The 95% coverage falls short of 0.95 under mismatched priors.

    This assertion documents the known shortfall rather than asserting the
    target is met. Changing it to demand 0.95 would be tuning the test to the
    hoped-for answer; the honest response is to report the number and the reason
    (see P-22), not to move the goalposts.
    """
    coverage = results["calibration"]["95"]
    assert 0.6 <= coverage <= 1.0
    if coverage < 0.90:
        matched = EVAL / "summary_matched.json"
        assert matched.is_file(), (
            "95% coverage is below target, so the matched-prior run must exist "
            "to show whether the cause is the machinery or the prior"
        )


def test_matched_prior_run_isolates_the_cause(results):
    """With priors that cover the truth, the machinery must calibrate."""
    path = EVAL / "summary_matched.json"
    if not path.is_file():
        pytest.skip("matched-prior run not present")
    matched = json.loads(path.read_text(encoding="utf-8"))
    assert matched["matched_params"] is True
    assert matched["calibration"]["95"] >= results["calibration"]["95"], (
        "matched priors should calibrate at least as well as mismatched ones; "
        "if not, the fault is in the inversion machinery, not the prior"
    )
