"""Phase 4 tests for the attribution engine.

The tests that matter most here are not the ones checking that ranking works.
They are:

``test_timestamp_resolution_is_not_assumed``
    guards P-18, a unit error that silently discarded 99.6% of the AIS while
    reporting success.

``test_dark_vessel_takes_all_mass_when_no_vessel_explains_the_slick``
    the property that stops the system naming an innocent ship when the real
    polluter was not transmitting.

``test_gap_prior_is_relative_to_the_local_baseline``
    the factor most likely to manufacture suspicion out of poor receiver
    coverage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attribution import (
    AttributionConfig,
    VesselTrack,
    attribute,
    baseline_gap_hours,
    behavioural_prior,
    clean_and_reconstruct,
    load_ais,
    prefilter,
)
from src.attribution.priors import (
    NEUTRAL,
    PriorConfig,
    alignment_factor,
    gap_factor,
    speed_factor,
    type_factor,
)
from src.contracts import CaseManifest, RankedCandidates
from src.ingest.case_builder import load_forcing_bundle
from src.inversion import InversionConfig, ObservedMask, invert
from src.transport import ForcingField, Seeds, TransportParams, simulate

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "cases" / "synth_kattegat"
UTC = timezone.utc


@pytest.fixture(scope="module")
def case() -> CaseManifest:
    if not CASE_DIR.is_dir():
        pytest.skip("run scripts/build_synthetic_case.py --all first")
    return CaseManifest.load(CASE_DIR)


@pytest.fixture(scope="module")
def field() -> ForcingField:
    if not CASE_DIR.is_dir():
        pytest.skip("run scripts/build_synthetic_case.py --all first")
    return ForcingField.from_case(CASE_DIR, load_forcing_bundle(CASE_DIR))


@pytest.fixture(scope="module")
def cleaned():
    tracks, report = clean_and_reconstruct(load_ais(CASE_DIR / "ais" / "tracks.parquet"))
    return tracks, report


def _synthetic_track(mmsi="001234567", n=200, start=0.0, step=120.0,
                     lon0=11.3, lat0=57.0, speed_kn=12.0, ship_type="Cargo"):
    t = start + step * np.arange(n)
    return VesselTrack(
        mmsi=mmsi, name=f"T-{mmsi}", ship_type=ship_type,
        time=t,
        lon=lon0 + 0.002 * np.arange(n),
        lat=lat0 + 0.001 * np.arange(n),
        sog_kn=np.full(n, speed_kn),
        cog_deg=np.full(n, 60.0),
    )


# ------------------------------------------------------------------ AIS


def test_timestamp_resolution_is_not_assumed(cleaned):
    """P-18. pandas stores this column as datetime64[us], so the common idiom
    astype("int64") / 1e9 yields seconds/1000. Every dt then comes out 1000x too
    small, every implied speed 1000x too large, and the speed filter discards
    almost the whole dataset while the report still looks plausible."""
    tracks, report = cleaned
    assert report.rows_out > 0.95 * report.rows_in, (
        f"only {report.rows_out}/{report.rows_in} fixes survived cleaning; "
        f"check the timestamp units"
    )
    assert report.dropped_speed_jump < 0.02 * report.rows_in


def test_cleaning_refuses_to_return_a_gutted_dataset():
    """A filter that drops most of the data is a unit error, not dirty data."""
    n = 400
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            [datetime(2024, 3, 10, tzinfo=UTC) + timedelta(seconds=2 * i) for i in range(n)],
            utc=True,
        ),
        "mmsi": ["001111111"] * n,
        # Teleporting every fix: implied speeds far above any real vessel.
        "lat": 57.0 + 0.5 * (np.arange(n) % 2),
        "lon": 11.0 + 0.5 * (np.arange(n) % 2),
        "sog_kn": np.full(n, 10.0),
        "cog_deg": np.full(n, 90.0),
    })
    with pytest.raises(ValueError, match="check the timestamp units"):
        clean_and_reconstruct(df)
    tracks, report = clean_and_reconstruct(df, strict=False)
    assert report.dropped_speed_jump > 0


def test_naive_timestamps_are_refused(tmp_path):
    """W-04: a naive timestamp assumed local shifts every attribution by a
    constant number of hours."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-03-10T00:00:00", "2024-03-10T00:02:00"]),
        "mmsi": ["1", "1"], "lat": [57.0, 57.001], "lon": [11.0, 11.001],
        "sog_kn": [10.0, 10.0], "cog_deg": [90.0, 90.0],
    })
    path = tmp_path / "naive.parquet"
    df.to_parquet(path)
    with pytest.raises(ValueError, match="timezone-naive"):
        load_ais(path)


def test_cleaning_counts_everything_it_drops(cleaned):
    """An undisclosed filter is a hidden assumption."""
    _, report = cleaned
    d = report.as_dict()
    for key in ("rows_in", "rows_out", "dropped_bad_coords", "dropped_duplicate_fix",
                "dropped_speed_jump", "vessels_in", "vessels_out", "tracks_out"):
        assert key in d


def test_tracks_are_split_on_long_silence():
    a = _synthetic_track(n=60, start=0.0)
    b = _synthetic_track(n=60, start=60 * 120.0 + 9 * 3600.0)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(np.concatenate([a.time, b.time]), unit="s", utc=True),
        "mmsi": ["001234567"] * 120,
        "lat": np.concatenate([a.lat, b.lat]), "lon": np.concatenate([a.lon, b.lon]),
        "sog_kn": np.concatenate([a.sog_kn, b.sog_kn]),
        "cog_deg": np.concatenate([a.cog_deg, b.cog_deg]),
        "ship_name": ["X"] * 120, "ship_type": ["Cargo"] * 120,
    })
    tracks, report = clean_and_reconstruct(df)
    assert len(tracks) == 2
    assert report.tracks_split_on_gap == 1


def test_interpolation_across_a_gap_is_flagged():
    """W-09: interpolated positions must never quietly become evidence."""
    track = _synthetic_track(n=100)
    track.gaps = [(track.time[40], track.time[60])]
    times = np.array([track.time[10], track.time[50], track.time[90]])
    _, _, crossed = track.interpolate(times)
    assert not crossed[0] and crossed[1] and not crossed[2]


def test_gap_hours_overlapping_window():
    track = _synthetic_track(n=100)
    track.gaps = [(1000.0, 1000.0 + 2 * 3600.0)]
    assert track.gap_hours_overlapping(0.0, 1e9) == pytest.approx(2.0)
    assert track.gap_hours_overlapping(0.0, 1000.0 + 3600.0) == pytest.approx(1.0)
    assert track.gap_hours_overlapping(1e8, 1e9) == 0.0


def test_baseline_gap_is_a_local_median(cleaned):
    tracks, _ = cleaned
    assert baseline_gap_hours(tracks) >= 0.0
    assert baseline_gap_hours([]) == 0.0


# ------------------------------------------------------------- the priors


def test_gap_prior_is_relative_to_the_local_baseline():
    """W-07, the most dangerous factor in the system.

    In a poorly covered area every vessel has gaps. An absolute threshold would
    flag them all and manufacture suspicion out of a property of the receiver
    network, so the gap is measured against what vessels here normally show.
    """
    track = _synthetic_track(n=200)
    track.gaps = [(track.time[50], track.time[50] + 2 * 3600.0)]
    window = (track.time[50], track.time[50] + 2 * 3600.0)

    tight = gap_factor(track, *window, baseline_hours=0.2)
    loose = gap_factor(track, *window, baseline_hours=3.0)

    assert tight.value > loose.value, "a 2 h gap must matter less where gaps are normal"
    assert loose.value == NEUTRAL, "a gap at or below the local baseline is not a signal"
    assert "baseline" in tight.note


def test_no_gap_is_neutral():
    track = _synthetic_track(n=100)
    assert gap_factor(track, track.time[0], track.time[-1], 0.5).value == NEUTRAL


def test_speed_prior_compares_against_the_vessels_own_median():
    """A fishing boat must not be flagged simply for being slower than a
    container ship."""
    slow = _synthetic_track(n=120, speed_kn=12.0)
    slow.sog_kn[40:80] = 3.0
    anomalous = speed_factor(slow, slow.time[40], slow.time[80])
    steady = speed_factor(_synthetic_track(n=120, speed_kn=4.0),
                          slow.time[40], slow.time[80])
    assert anomalous.value > NEUTRAL
    assert steady.value == NEUTRAL, "a consistently slow vessel is not anomalous"


def test_alignment_prior_rewards_track_aligned_slicks():
    """A continuous discharge from a vessel under way leaves a slick along the
    track it sailed -- the one behavioural factor that comes from the shape of
    the oil rather than the vessel's conduct."""
    track = _synthetic_track(n=100)  # cog 60 deg
    aligned = alignment_factor(track, track.time[50], slick_axis_deg=60.0)
    crossed = alignment_factor(track, track.time[50], slick_axis_deg=150.0)
    assert aligned.value > crossed.value
    assert crossed.value == NEUTRAL


def test_alignment_is_undirected():
    """A slick axis has no direction, so a vessel on a reciprocal heading aligns
    just as well."""
    track = _synthetic_track(n=100)
    assert alignment_factor(track, track.time[50], 60.0).value == pytest.approx(
        alignment_factor(track, track.time[50], 240.0).value
    )


def test_type_prior_ranks_tankers_above_passenger_vessels():
    assert type_factor(_synthetic_track(ship_type="Tanker")).value > \
           type_factor(_synthetic_track(ship_type="Passenger")).value


def test_prior_factors_are_individually_visible():
    """No factor is folded into a weighted sum. A weight nobody can justify is
    what makes a ranking indefensible."""
    track = _synthetic_track(n=150)
    prior, factors = behavioural_prior(
        track, track.time[10], track.time[60], PriorConfig(slick_axis_deg=60.0)
    )
    assert set(factors) == {
        "ais_gap_factor", "speed_anomaly_factor", "course_change_factor",
        "vessel_type_factor", "axis_alignment_factor",
    }
    assert prior == pytest.approx(np.prod([f.value for f in factors.values()]))
    for f in factors.values():
        assert f.note, "every factor must carry a plain-language justification"


def test_no_factor_can_dominate_on_its_own():
    """None of these signals is strong enough to carry an accusation alone."""
    track = _synthetic_track(n=300, speed_kn=12.0)
    track.sog_kn[:] = 0.5
    track.gaps = [(track.time[0], track.time[-1])]
    _, factors = behavioural_prior(
        track, track.time[0], track.time[-1], PriorConfig(slick_axis_deg=60.0)
    )
    for name, f in factors.items():
        assert f.value <= 3.0, f"{name} reached {f.value}"


# ------------------------------------------------- full attribution run


@pytest.fixture(scope="module")
def attributed(field, case, cleaned):
    """Seed a discharge along a real vessel's track, then try to recover it."""
    tracks, report = cleaned
    lookback, window = 20.0, 2.0 * 3600.0
    t0 = case.t_obs.timestamp() - lookback * 3600.0

    candidates = [t for t in tracks if t.covers(t0) and t.covers(t0 + window) and len(t) > 150]
    culprit = candidates[3]

    rng = np.random.default_rng(11)
    n = 3000
    times = t0 + rng.random(n) * window
    lon, lat, _ = culprit.interpolate(times)
    truth = simulate(
        field,
        Seeds(lon=lon, lat=lat, seed_time=times, origin_marker=np.zeros(n, "int64")),
        datetime.fromtimestamp(t0, tz=UTC), case.t_obs, TransportParams(), seed=11,
    )
    ok = truth.usable
    mask = ObservedMask.from_points(truth.lon[ok], truth.lat[ok], field)
    posterior, _ = invert(field, mask, case.t_obs, case.case_id, "x",
                          InversionConfig(lookback_hours=lookback, seed=3))
    ranked, diag = attribute(
        field, mask, posterior, tracks, case.t_obs, case.case_id,
        AttributionConfig(seed=5), cleaning=report.as_dict(),
        vessels_in_window=report.vessels_out,
    )
    return ranked, diag, culprit, mask, posterior, tracks


def test_output_validates_against_the_contract(attributed):
    ranked, *_ = attributed
    RankedCandidates.model_validate(ranked.model_dump())


def test_probabilities_including_dark_sum_to_one(attributed):
    ranked, *_ = attributed
    total = sum(c.posterior_probability for c in ranked.candidates)
    total += ranked.dark_vessel_hypothesis.posterior_probability
    assert total == pytest.approx(1.0, abs=1e-9)


def test_prefilter_removes_irrelevant_traffic(attributed):
    """Reducing 179 vessels to a handful is the deliverable."""
    ranked, _, _, _, _, _ = attributed
    tr = ranked.traffic_reduction
    assert tr.after_prefilter < tr.vessels_in_window
    assert tr.reduction_factor > 10.0


def test_prefilter_does_not_drop_the_true_culprit(attributed):
    """The point of filtering is to remove irrelevant traffic, not the answer."""
    ranked, _, culprit, _, _, _ = attributed
    assert any(c.mmsi == culprit.mmsi for c in ranked.candidates)


def test_true_culprit_is_ranked_in_the_top_three(attributed):
    """Top-3 recall is the target. Top-1 is deliberately not the goal."""
    ranked, _, culprit, _, _, _ = attributed
    rank = next(c.rank for c in ranked.candidates if c.mmsi == culprit.mmsi)
    assert rank <= 3, f"true culprit ranked {rank}"


def test_dark_vessel_hypothesis_is_always_present(attributed):
    """Without it, a system normalising only over observed vessels will
    confidently name an innocent ship whenever the polluter was dark."""
    ranked, *_ = attributed
    dark = ranked.dark_vessel_hypothesis
    assert dark.posterior_probability > 0.0
    assert 0.0 < dark.prior_used < 1.0
    assert "not explained by any AIS-observed track" in dark.note


def test_dark_vessel_takes_all_mass_when_no_vessel_is_supplied(field, case, attributed):
    """The correct answer when the polluter was not transmitting."""
    _, _, _, mask, posterior, _ = attributed
    ranked, _ = attribute(field, mask, posterior, [], case.t_obs, case.case_id,
                          AttributionConfig(seed=5), vessels_in_window=179)
    assert ranked.candidates == []
    assert ranked.dark_vessel_hypothesis.posterior_probability == pytest.approx(1.0)
    assert ranked.dark_vessel_is_most_probable


def test_evidence_is_broken_out_per_candidate(attributed):
    ranked, *_ = attributed
    top = ranked.candidates[0]
    ev = top.evidence
    assert np.isfinite(ev.track_overlap_likelihood)
    for factor in (ev.ais_gap_factor, ev.speed_anomaly_factor, ev.course_change_factor,
                   ev.vessel_type_factor, ev.axis_alignment_factor):
        assert factor.value > 0.0 and factor.note


def test_best_discharge_window_is_reported(attributed):
    ranked, _, culprit, _, _, _ = attributed
    c = next(c for c in ranked.candidates if c.mmsi == culprit.mmsi)
    start, end = c.best_discharge_window_utc
    assert start < end
    assert start.tzinfo is not None


def test_track_geojson_accompanies_each_candidate(attributed):
    ranked, *_ = attributed
    geo = ranked.candidates[0].track_geojson
    assert geo["geometry"]["type"] == "LineString"
    assert len(geo["geometry"]["coordinates"]) > 1


def test_slick_axis_is_computed_from_the_mask(attributed):
    _, diag, _, mask, _, _ = attributed
    assert 0.0 <= diag.slick_axis_deg < 180.0
    assert mask.elongation >= 1.0


def test_attribution_is_deterministic(field, case, attributed):
    _, _, _, mask, posterior, tracks = attributed
    cfg = AttributionConfig(seed=9)
    a, _ = attribute(field, mask, posterior, tracks, case.t_obs, case.case_id, cfg)
    b, _ = attribute(field, mask, posterior, tracks, case.t_obs, case.case_id, cfg)
    assert [c.mmsi for c in a.candidates] == [c.mmsi for c in b.candidates]
    assert a.dark_vessel_hypothesis.posterior_probability == pytest.approx(
        b.dark_vessel_hypothesis.posterior_probability
    )


def test_raising_the_dark_prior_raises_the_dark_probability(field, case, attributed):
    """The dark prior is an assumption, so its sensitivity must be visible."""
    _, _, _, mask, posterior, tracks = attributed
    low, _ = attribute(field, mask, posterior, tracks, case.t_obs, case.case_id,
                       AttributionConfig(seed=5, dark_vessel_prior=0.05))
    high, _ = attribute(field, mask, posterior, tracks, case.t_obs, case.case_id,
                        AttributionConfig(seed=5, dark_vessel_prior=0.40))
    assert (high.dark_vessel_hypothesis.posterior_probability
            > low.dark_vessel_hypothesis.posterior_probability)


def test_one_simulation_covers_every_hypothesis(attributed):
    """All candidates, all release windows and the dark hypothesis share a
    single forward run, distinguished by origin_marker."""
    _, diag, _, _, _, _ = attributed
    assert diag.n_hypotheses > diag.after_prefilter
    assert diag.n_dark_samples > 0
    assert diag.n_particles == diag.n_hypotheses * AttributionConfig().particles_per_hypothesis


def test_attribution_config_loads_from_physics_yaml():
    cfg = AttributionConfig.from_yaml()
    assert cfg.prefilter_spatial_margin_km == 25.0
    assert cfg.dark_vessel_prior == 0.15


@pytest.mark.slow
def test_attribution_runtime_budget(attributed):
    _, diag, _, _, _, _ = attributed
    assert diag.elapsed_s < 20.0, f"attribution took {diag.elapsed_s:.1f}s"
