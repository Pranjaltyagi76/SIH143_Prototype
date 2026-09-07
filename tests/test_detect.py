"""Phase 5 tests for detection: injection, segmentation, the physics gate.

The test that matters most is ``test_low_wind_patch_always_abstains``. Showing
what the system correctly *refuses* to flag is the strongest thing it does, and
the hard gate is enforced in two places -- here and in the contract -- precisely
so that no future change can quietly route around it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.attribution import clean_and_reconstruct, load_ais
from src.contracts import CaseManifest, Classification, SlickDetection
from src.contracts.detection import WIND_MAX_MS, WIND_MIN_MS
from src.detect import (
    LookAlikeClassifier,
    detect,
    inject,
    read_scene,
    render_damping,
    segment_dark_patches,
    segmentation_iou,
    segmentation_recall,
)
from src.detect.gate import FEATURE_NAMES, PatchFeatures, outside_detectability_window
from src.detect.pipeline import _land_on_scene
from src.ingest.case_builder import load_forcing_bundle
from src.transport import ForcingField, Seeds, TransportParams, simulate

CASES = Path(__file__).resolve().parents[1] / "data" / "cases"
CASE_DIR = CASES / "synth_kattegat"
UTC = timezone.utc


@pytest.fixture(scope="module")
def scene():
    if not CASE_DIR.is_dir():
        pytest.skip("run scripts/build_synthetic_case.py --all first")
    case = CaseManifest.load(CASE_DIR)
    field = ForcingField.from_case(CASE_DIR, load_forcing_bundle(CASE_DIR))
    sigma0, lon, lat = read_scene(CASE_DIR)
    land = _land_on_scene(field, lon, lat)
    tracks, _ = clean_and_reconstruct(load_ais(CASE_DIR / "ais" / "tracks.parquet"))
    return case, field, sigma0, lon, lat, land, tracks


@pytest.fixture(scope="module")
def injected(scene):
    """A discharge seeded along a real vessel track, baked into the scene."""
    case, field, sigma0, lon, lat, land, tracks = scene
    window = 2.5 * 3600.0
    t0 = case.t_obs.timestamp() - 18.0 * 3600.0
    culprit = [t for t in tracks if t.covers(t0) and t.covers(t0 + window) and len(t) > 150][3]

    rng = np.random.default_rng(7)
    n = 7000
    times = t0 + rng.random(n) * window
    clon, clat, _ = culprit.interpolate(times)
    traj = simulate(
        field,
        Seeds(lon=clon, lat=clat, seed_time=times, origin_marker=np.zeros(n, "int64")),
        datetime.fromtimestamp(t0, tz=UTC), case.t_obs, TransportParams(), seed=7,
    )
    ok = traj.usable
    return inject(sigma0, traj.lon[ok], traj.lat[ok], lon, lat, land=land), culprit


# ------------------------------------------------------------------ injection


def test_damping_is_negative_and_bounded(scene):
    _, _, sigma0, lon, lat, _, _ = scene
    rng = np.random.default_rng(0)
    d = render_damping(
        rng.uniform(11.3, 11.5, 4000), rng.uniform(56.9, 57.1, 4000), lon, lat
    )
    assert d.max() <= 0.0, "damping must darken, never brighten"
    assert d.min() >= -12.0, "damping beyond ~12 dB is not physical at C-band VV"


def test_damping_saturates_with_concentration(scene):
    """More oil past a point changes nothing, which is exactly why damping ratio
    cannot be inverted for film thickness from a single scene -- and why we
    refuse to report slick age radiometrically."""
    _, _, _, lon, lat, _, _ = scene
    rng = np.random.default_rng(0)
    thin = render_damping(rng.normal(11.4, 0.01, 2000), rng.normal(57.0, 0.01, 2000), lon, lat)
    thick = render_damping(rng.normal(11.4, 0.01, 20000), rng.normal(57.0, 0.01, 20000), lon, lat)
    assert thick.min() < thin.min()
    assert thick.min() / thin.min() < 3.0, "10x the oil must not give 10x the damping"


def test_injection_darkens_the_scene_where_the_slick_is(injected, scene):
    inj, _ = injected
    _, _, sigma0, _, _, _, _ = scene
    assert inj.truth_mask.any()
    assert inj.sigma0_db[inj.truth_mask].mean() < sigma0[inj.truth_mask].mean() - 1.0
    untouched = ~inj.truth_mask & (inj.damping_db == 0.0)
    assert np.allclose(inj.sigma0_db[untouched], sigma0[untouched])


def test_injection_does_not_place_oil_on_land(injected, scene):
    inj, _ = injected
    _, _, _, _, _, land, _ = scene
    assert not (inj.truth_mask & land.astype(bool)).any()


# --------------------------------------------------------------- segmentation


def test_segmenter_finds_the_injected_slick(injected, scene):
    """Stage 1 is tuned for recall; a missed slick is unrecoverable."""
    inj, _ = injected
    _, _, _, lon, lat, land, _ = scene
    labels, _ = segment_dark_patches(inj.sigma0_db, lon, lat, land=land)
    recall = segmentation_recall(labels > 0, inj.truth_mask)
    assert recall > 0.75, f"recall {recall:.2f} is below the 0.75 target"


def test_segmentation_iou_is_reported_on_the_oil_class(injected, scene):
    """IoU is measured AFTER the gate, on the oil class only.

    Never overall pixel accuracy: oil is a tiny fraction of any scene, so
    'predict all sea' scores 99%. And never over the raw Stage 1 output either:
    Stage 1 is deliberately high-recall and returns look-alikes on purpose, so
    scoring its union against oil-only truth would blame it for doing its job.
    The meaningful number is the union of patches the gate confirmed as oil.
    """
    inj, culprit = injected
    case, field, _, _, _, _, tracks = scene
    labels, patches = segment_dark_patches(inj.sigma0_db, *read_scene(CASE_DIR)[1:],
                                           land=scene[5])
    detections, _ = detect(CASE_DIR, field, case.t_obs, case.case_id,
                           sigma0_db=inj.sigma0_db, tracks=tracks)
    oil_ids = {d.detection_id for d in detections if d.is_actionable}
    assert oil_ids, "no oil confirmed, so IoU is undefined"

    # Patch order in `detect` is by descending area, matching detection_id order.
    by_area = sorted(patches, key=lambda p: -p.area_km2)
    oil_mask = np.zeros(labels.shape, dtype=bool)
    for i, patch in enumerate(by_area, start=1):
        if f"{case.case_id}:d{i:02d}" in oil_ids:
            oil_mask[patch.pixels[:, 0], patch.pixels[:, 1]] = True

    iou = segmentation_iou(oil_mask, inj.truth_mask)
    assert iou > 0.35, f"post-gate IoU on the oil class is {iou:.2f}"


def test_background_window_must_exceed_the_feature_size(scene):
    """The documented sensitivity, asserted so it cannot regress silently.

    A window smaller than a broad dark feature sits inside it, so the feature
    contaminates its own reference level and becomes invisible.
    """
    _, _, sigma0, lon, lat, land, _ = scene
    _, narrow = segment_dark_patches(sigma0, lon, lat, land=land, background_window_m=20_000.0)
    _, wide = segment_dark_patches(sigma0, lon, lat, land=land, background_window_m=60_000.0)
    assert max((p.area_km2 for p in wide), default=0) > 100.0
    assert max((p.area_km2 for p in narrow), default=0) < 100.0


def test_segmenter_ignores_land(scene):
    _, _, sigma0, lon, lat, land, _ = scene
    labels, _ = segment_dark_patches(sigma0, lon, lat, land=land)
    assert not (labels[land.astype(bool)] > 0).any()


# ------------------------------------------------------------- the hard gate


@pytest.mark.parametrize("wind", [0.0, 1.4, 2.9])
def test_wind_below_the_window_is_refused(wind):
    reason = outside_detectability_window(wind)
    assert reason is not None
    assert "below" in reason and "smooth" in reason


@pytest.mark.parametrize("wind", [12.1, 18.0])
def test_wind_above_the_window_is_refused(wind):
    reason = outside_detectability_window(wind)
    assert reason is not None
    assert "above" in reason


@pytest.mark.parametrize("wind", [WIND_MIN_MS, 6.4, WIND_MAX_MS])
def test_wind_inside_the_window_is_allowed(wind):
    assert outside_detectability_window(wind) is None


def test_low_wind_patch_always_abstains(scene):
    """The strongest thing the system does.

    Below ~3 m/s the sea surface is already smooth, so no oil-water contrast is
    physically possible and any confident label is unsupportable. The classifier
    cannot overturn this.
    """
    case, field, sigma0, _, _, _, tracks = scene
    detections, diag = detect(CASE_DIR, field, case.t_obs, case.case_id, tracks=tracks)
    low = [d for d in detections if d.environment.wind_speed_ms < WIND_MIN_MS]
    assert low, "the development scene should contain low-wind dark patches"
    for d in low:
        assert d.classification is Classification.UNDETERMINED
        assert d.abstained
        assert "detectability window" in d.classification_reason


def test_the_calm_pocket_is_found_and_refused(scene):
    """The demo beat: a large dark patch that is correctly not called oil."""
    case, field, _, _, _, _, tracks = scene
    detections, _ = detect(CASE_DIR, field, case.t_obs, case.case_id, tracks=tracks)
    big = max(detections, key=lambda d: d.geometry.area_km2)
    assert big.geometry.area_km2 > 100.0
    assert big.abstained
    assert f"{big.environment.wind_speed_ms:.1f}" in big.classification_reason


def test_no_false_positives_on_a_clean_scene(scene):
    """The stored scene contains no oil at all."""
    case, field, _, _, _, _, tracks = scene
    detections, diag = detect(CASE_DIR, field, case.t_obs, case.case_id, tracks=tracks)
    assert diag.n_oil == 0, "called oil on a scene containing none"


def test_feature_vector_order_is_stable():
    """The model file stores this order; a silent reordering would corrupt every
    prediction without raising anything."""
    f = PatchFeatures(**{n: 1.0 for n in FEATURE_NAMES})
    assert f.vector().shape == (len(FEATURE_NAMES),)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_classifier_falls_back_when_no_model_is_present():
    """The fallback runs if the model file is missing on the demo machine."""
    clf = LookAlikeClassifier()
    assert not clf.is_trained
    p = clf.predict_oil_probability(PatchFeatures(**{n: 1.0 for n in FEATURE_NAMES}))
    assert 0.0 <= p <= 1.0
    assert clf.feature_importance() == {}


# ------------------------------------------------------------------ pipeline


def test_detection_output_validates_against_the_contract(scene):
    case, field, _, _, _, _, tracks = scene
    detections, _ = detect(CASE_DIR, field, case.t_obs, case.case_id, tracks=tracks)
    for d in detections:
        SlickDetection.model_validate(d.model_dump())


def test_every_detection_carries_a_reason(scene):
    """The UI is required to render it; an abstention with no reason is useless."""
    case, field, _, _, _, _, tracks = scene
    detections, _ = detect(CASE_DIR, field, case.t_obs, case.case_id, tracks=tracks)
    for d in detections:
        assert len(d.classification_reason) > 30
        assert "wind" in d.classification_reason.lower()


def test_injected_slick_is_classified_as_oil(injected, scene):
    case, field, _, _, _, _, tracks = scene
    inj, _ = injected
    detections, diag = detect(
        CASE_DIR, field, case.t_obs, case.case_id,
        sigma0_db=inj.sigma0_db, tracks=tracks,
    )
    oil = [d for d in detections if d.is_actionable]
    assert oil, "the injected slick was not recovered"
    best = max(oil, key=lambda d: d.geometry.area_km2)
    assert 0.4 * inj.area_km2 < best.geometry.area_km2 < 2.5 * inj.area_km2
    assert best.geometry.elongation > 3.0, "a line-source discharge should be elongated"


def test_abstention_rate_is_reported(scene):
    case, field, _, _, _, _, tracks = scene
    _, diag = detect(CASE_DIR, field, case.t_obs, case.case_id, tracks=tracks)
    assert 0.0 <= diag.abstention_rate <= 1.0
    assert diag.n_patches == diag.n_oil + diag.n_look_alike + diag.n_undetermined


def test_polygons_are_closed_rings(scene):
    case, field, _, _, _, _, tracks = scene
    detections, _ = detect(CASE_DIR, field, case.t_obs, case.case_id, tracks=tracks)
    for d in detections:
        assert d.polygon_wgs84[0] == d.polygon_wgs84[-1]
        assert len(d.polygon_wgs84) >= 4


@pytest.mark.slow
def test_spill_case_runs_end_to_end():
    """The whole vertical slice: raw scene to ranked vessels."""
    import json

    spill = CASES / "synth_kattegat_spill"
    if not spill.is_dir():
        pytest.skip("run scripts/inject_case.py first")

    truth = json.loads((spill / "truth.json").read_text(encoding="utf-8"))
    candidates = json.loads((spill / "out" / "candidates.json").read_text(encoding="utf-8"))
    ranks = [c["rank"] for c in candidates["candidates"] if c["mmsi"] == truth["culprit_mmsi"]]
    assert ranks, "the true culprit is not in the candidate list at all"
    assert ranks[0] <= 3, f"true culprit ranked {ranks[0]}"
    assert candidates["dark_vessel_hypothesis"]["posterior_probability"] > 0.0
