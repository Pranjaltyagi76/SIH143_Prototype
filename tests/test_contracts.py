"""Phase 0 contract tests.

Two jobs:

1. Every committed fixture validates against its model. This is what lets the
   frontend and every stage owner build in parallel against a known-good shape.

2. Every design decision encoded as a validator actually FIRES. A guard that
   silently does nothing is worse than no guard, because it buys false
   confidence. Each test below names the requirement or the logged near-miss it
   defends.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.contracts import (
    BoundingBox,
    CaseManifest,
    Classification,
    ForcingBundle,
    RankedCandidates,
    SlickDetection,
    SourcePosterior,
    reject_accusatory_language,
)

FIXTURES = Path(__file__).parent / "fixtures"
UTC = timezone.utc


def load(name: str) -> dict | list:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- fixtures load


def test_case_fixture_validates():
    case = CaseManifest.model_validate(load("case.json"))
    assert case.case_id == "kattegat_2024_03_11"
    assert case.max_lookback_hours > 6.0
    assert case.crs_working.startswith("EPSG:")


def test_forcing_fixture_validates():
    fb = ForcingBundle.model_validate(load("forcing_bundle.json"))
    assert fb.currents.includes_stokes is True
    assert fb.available_lookback_hours > 6.0


def test_detection_fixtures_validate():
    raw = load("slick_detections.json")
    dets = [SlickDetection.model_validate(d) for d in raw]
    assert len(dets) == 3
    by_class = {d.classification for d in dets}
    # The fixture set deliberately exercises all three outcomes, because the UI
    # must render all three from day one.
    assert by_class == {
        Classification.OIL,
        Classification.LOOK_ALIKE,
        Classification.UNDETERMINED,
    }
    assert sum(d.is_actionable for d in dets) == 1


def test_posterior_fixture_validates():
    p = SourcePosterior.model_validate(load("source_posterior.json"))
    assert p.credible_regions["50"].area_km2 < p.credible_regions["95"].area_km2
    assert p.within_operating_envelope
    assert p.assumptions, "assumptions must never be empty"


def test_candidates_fixture_validates():
    rc = RankedCandidates.model_validate(load("ranked_candidates.json"))
    assert rc.traffic_reduction.reduction_factor > 50.0
    assert rc.dark_vessel_hypothesis.posterior_probability > 0.0
    assert not rc.dark_vessel_is_most_probable


# ------------------------------------------------------ strictness and time (W-04)


def test_extra_fields_are_rejected():
    """Contract drift must fail loudly at the boundary (W-13)."""
    payload = load("case.json")
    payload["totally_new_field"] = 1
    with pytest.raises(ValidationError, match="totally_new_field"):
        CaseManifest.model_validate(payload)


def test_naive_datetime_is_rejected():
    """AIS timestamps assumed local instead of UTC put the answer out by a
    constant whole number of hours -- plausible, and it survives review (W-04)."""
    payload = load("case.json")
    payload["t_obs"] = "2024-03-11T05:42:13"  # no timezone
    with pytest.raises(ValidationError, match="UTC"):
        CaseManifest.model_validate(payload)


def test_non_utc_offset_is_rejected():
    payload = load("case.json")
    payload["t_obs"] = "2024-03-11T05:42:13+05:30"
    with pytest.raises(ValidationError, match="UTC"):
        CaseManifest.model_validate(payload)


def test_posterior_list_datetimes_are_utc_checked():
    """Guards the list-valued time validators, which are easy to get wrong."""
    payload = load("source_posterior.json")
    payload["grid"]["t0"][0] = "2024-03-10T00:00:00"  # naive
    with pytest.raises(ValidationError, match="UTC"):
        SourcePosterior.model_validate(payload)


# ----------------------------------------------------- the Stokes guard (P-05)


def test_includes_stokes_has_no_default():
    """CMEMS SMOC already merges Stokes drift into uo/vo. Adding a separate
    parameterisation double-counts a real physical term and biases every
    trajectory downwind, silently. Whoever builds a case must look it up."""
    payload = load("forcing_bundle.json")
    del payload["currents"]["includes_stokes"]
    with pytest.raises(ValidationError, match="includes_stokes"):
        ForcingBundle.model_validate(payload)


def test_wind_must_be_components_not_bearing():
    """Meteorological 'from' vs oceanographic 'to' makes the slick drift exactly
    the wrong way (W-02). Components only."""
    payload = load("forcing_bundle.json")
    payload["wind"]["vars"] = ["wind_speed", "wind_dir"]
    with pytest.raises(ValidationError, match="components"):
        ForcingBundle.model_validate(payload)


def test_forcing_window_must_be_ordered_and_long_enough():
    payload = load("forcing_bundle.json")
    payload["t_start"] = payload["t_obs"]
    with pytest.raises(ValidationError):
        ForcingBundle.model_validate(payload)


# -------------------------------------------------- the physics gate (FR-3a)


def test_low_wind_cannot_be_classified_as_oil():
    """Below ~3 m/s the sea surface is already smooth, so no oil-water contrast
    is physically possible. The classifier must not be able to overturn this."""
    payload = load("slick_detections.json")[0]  # the 1.4 m/s abstention
    assert payload["environment"]["wind_speed_ms"] == 1.4
    payload["classification"] = "oil"
    payload["abstained"] = False
    with pytest.raises(ValidationError, match="detectability window"):
        SlickDetection.model_validate(payload)


def test_high_wind_cannot_be_classified_as_oil():
    """Above ~12 m/s wave action disperses and submerges the slick."""
    payload = load("slick_detections.json")[1]  # the confirmed oil patch
    payload["environment"]["wind_speed_ms"] = 15.5
    with pytest.raises(ValidationError, match="detectability window"):
        SlickDetection.model_validate(payload)


def test_abstained_flag_must_match_classification():
    payload = load("slick_detections.json")[1]
    payload["abstained"] = True
    with pytest.raises(ValidationError, match="contradicts"):
        SlickDetection.model_validate(payload)


def test_classification_reason_cannot_be_empty():
    """The reason is what the UI renders when the system refuses to judge.
    Showing what we correctly decline to flag is the strongest demo beat."""
    payload = load("slick_detections.json")[0]
    payload["classification_reason"] = ""
    with pytest.raises(ValidationError):
        SlickDetection.model_validate(payload)


# ------------------------------------------------- posterior honesty (FR-10a, W-08)


def test_credible_regions_must_nest():
    payload = load("source_posterior.json")
    payload["credible_regions"]["50"]["area_km2"] = 99_999.0
    with pytest.raises(ValidationError, match="smaller"):
        SourcePosterior.model_validate(payload)


def test_both_credible_levels_are_required():
    payload = load("source_posterior.json")
    del payload["credible_regions"]["50"]
    with pytest.raises(ValidationError, match="'50'"):
        SourcePosterior.model_validate(payload)


def test_ess_cannot_exceed_particle_count():
    payload = load("source_posterior.json")
    payload["effective_sample_size"] = payload["n_particles"] + 1.0
    with pytest.raises(ValidationError, match="effective_sample_size"):
        SourcePosterior.model_validate(payload)


def test_lookback_past_horizon_must_admit_it():
    """Beyond ~72 h the envelope stops being operationally useful. The system
    reports its own operating limit rather than producing envelopes nobody can
    act on."""
    payload = load("source_posterior.json")
    payload["lookback_hours"] = 96.0
    payload["within_operating_envelope"] = True
    with pytest.raises(ValidationError, match="horizon"):
        SourcePosterior.model_validate(payload)

    payload["within_operating_envelope"] = False
    assert SourcePosterior.model_validate(payload).lookback_hours == 96.0


def test_assumptions_cannot_be_empty():
    """An unqualified probability is a marketing claim, not a scientific one."""
    payload = load("source_posterior.json")
    payload["assumptions"] = []
    with pytest.raises(ValidationError):
        SourcePosterior.model_validate(payload)


def test_unnormalised_posterior_is_rejected():
    payload = load("source_posterior.json")
    payload["grid"]["normalised"] = False
    with pytest.raises(ValidationError, match="normalised"):
        SourcePosterior.model_validate(payload)


# ------------------------------------------ the dark-vessel hypothesis (SC-1)


def test_dark_vessel_hypothesis_is_required():
    """Without it, a system normalising only over observed vessels will
    confidently name an innocent ship whenever the true polluter was dark."""
    payload = load("ranked_candidates.json")
    del payload["dark_vessel_hypothesis"]
    with pytest.raises(ValidationError, match="dark_vessel_hypothesis"):
        RankedCandidates.model_validate(payload)


def test_probabilities_including_dark_must_sum_to_one():
    payload = load("ranked_candidates.json")
    payload["dark_vessel_hypothesis"]["posterior_probability"] = 0.05
    with pytest.raises(ValidationError, match="sum to 1"):
        RankedCandidates.model_validate(payload)


def test_dark_vessel_can_be_the_most_probable_explanation():
    """'Most probable explanation: a vessel not transmitting AIS' is a correct
    and operationally valuable answer, and the contract must permit it."""
    payload = load("ranked_candidates.json")
    payload["candidates"] = payload["candidates"][:1]
    payload["candidates"][0]["posterior_probability"] = 0.30
    payload["dark_vessel_hypothesis"]["posterior_probability"] = 0.70
    payload["traffic_reduction"]["reported"] = 1
    rc = RankedCandidates.model_validate(payload)
    assert rc.dark_vessel_is_most_probable


def test_zero_candidates_is_valid_when_dark_takes_all_mass():
    payload = load("ranked_candidates.json")
    payload["candidates"] = []
    payload["dark_vessel_hypothesis"]["posterior_probability"] = 1.0
    payload["traffic_reduction"] = {
        "vessels_in_window": 214,
        "after_prefilter": 0,
        "reported": 0,
    }
    rc = RankedCandidates.model_validate(payload)
    assert rc.dark_vessel_is_most_probable


# ------------------------------------------------------- ranking integrity


def test_ranks_must_be_contiguous():
    payload = load("ranked_candidates.json")
    payload["candidates"][2]["rank"] = 7
    with pytest.raises(ValidationError, match="ranks must be"):
        RankedCandidates.model_validate(payload)


def test_candidates_must_be_ordered_by_probability():
    payload = load("ranked_candidates.json")
    p = payload["candidates"]
    p[0]["posterior_probability"], p[1]["posterior_probability"] = (
        p[1]["posterior_probability"],
        p[0]["posterior_probability"],
    )
    with pytest.raises(ValidationError, match="descending"):
        RankedCandidates.model_validate(payload)


def test_traffic_reduction_must_be_monotonic():
    payload = load("ranked_candidates.json")
    payload["traffic_reduction"]["after_prefilter"] = 500
    with pytest.raises(ValidationError, match="reported <="):
        RankedCandidates.model_validate(payload)


# --------------------------------------------------- accusatory language (SC-2)


@pytest.mark.parametrize(
    "text",
    [
        "this vessel is guilty",
        "the culprit was identified",
        "confirmed spill by MMSI 219000000",
        "The polluter is this tanker",
    ],
)
def test_accusatory_language_is_rejected(text: str):
    """The system produces investigative leads, not determinations of
    responsibility. Enforced in code so it cannot drift into a UI string."""
    with pytest.raises(ValueError, match="investigative leads"):
        reject_accusatory_language(text)


def test_permitted_language_passes():
    for text in [
        "candidate vessel, consistent with the posterior",
        "investigative lead under stated model assumptions",
        "3.2 h AIS gap against a 0.4 h local baseline",
    ]:
        assert reject_accusatory_language(text) == text


def test_evidence_note_rejects_accusation():
    payload = load("ranked_candidates.json")
    payload["candidates"][0]["evidence"]["ais_gap_factor"]["note"] = "clearly guilty"
    with pytest.raises(ValidationError, match="investigative leads"):
        RankedCandidates.model_validate(payload)


# ------------------------------------------------------------- geometry (W-11)


def test_bounding_box_ordering_is_enforced():
    with pytest.raises(ValidationError, match="min_lon"):
        BoundingBox(min_lon=12.0, min_lat=56.0, max_lon=10.0, max_lat=58.0)


@pytest.mark.parametrize(
    "lon,lat,expected",
    [
        (11.5, 57.0, "EPSG:32632"),  # Kattegat, northern hemisphere, zone 32
        (69.0, 22.0, "EPSG:32642"),  # Gulf of Kutch: zone 42 spans 66-72 E
        (-92.0, 27.0, "EPSG:32615"),  # Gulf of Mexico, zone 15
        (11.5, -57.0, "EPSG:32732"),  # southern hemisphere flips the prefix
    ],
)
def test_utm_zone_selection(lon: float, lat: float, expected: str):
    """All computation happens in projected metres. Computing area in degrees
    gives values wrong by a latitude-dependent factor, silently (W-11)."""
    box = BoundingBox(min_lon=lon - 0.5, min_lat=lat - 0.5, max_lon=lon + 0.5, max_lat=lat + 0.5)
    assert box.utm_epsg() == expected


def test_unclosed_polygon_is_rejected():
    payload = load("slick_detections.json")[1]
    payload["polygon_wgs84"] = payload["polygon_wgs84"][:-1]
    with pytest.raises(ValidationError, match="closed"):
        SlickDetection.model_validate(payload)


def test_degenerate_polygon_is_rejected():
    payload = load("slick_detections.json")[1]
    payload["polygon_wgs84"] = [[11.0, 57.0], [11.1, 57.0], [11.0, 57.0]]
    with pytest.raises(ValidationError, match=">= 4 vertices"):
        SlickDetection.model_validate(payload)


# ----------------------------------------------------------- round trip


def test_case_manifest_round_trips_on_disk(tmp_path: Path):
    """Reproducibility (NFR-3) depends on the manifest being the complete input."""
    case = CaseManifest.model_validate(load("case.json"))
    case.save(tmp_path / case.case_id)
    reloaded = CaseManifest.load(tmp_path / case.case_id)
    assert reloaded == case
    for sub in ("scene", "forcing", "ais", "out"):
        assert (tmp_path / case.case_id / sub).is_dir()


def test_all_contracts_round_trip_through_json():
    for name, model in [
        ("case.json", CaseManifest),
        ("forcing_bundle.json", ForcingBundle),
        ("source_posterior.json", SourcePosterior),
        ("ranked_candidates.json", RankedCandidates),
    ]:
        obj = model.model_validate(load(name))
        assert model.model_validate_json(obj.model_dump_json()) == obj


def test_seeded_case_is_deterministic_input():
    """Same case plus same seed must be the complete, reproducible input."""
    a = CaseManifest.model_validate(load("case.json"))
    b = CaseManifest.model_validate(load("case.json"))
    assert a.seed == b.seed and a == b
