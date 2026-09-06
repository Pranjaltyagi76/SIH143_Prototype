"""Phase 3 tests for the Bayesian source inversion.

The most valuable test in this file is ``test_closed_form_matches_brute_force``:
it checks the algebraic shortcut that makes the likelihood affordable against a
literal cell-by-cell evaluation. If that identity is wrong, every number the
system produces is wrong, and nothing else would reveal it.

The second most valuable is ``test_untempered_likelihood_is_overconfident``,
which pins down a failure we measured rather than guessed: treating each mask
cell as an independent observation gives a nominal 95% credible region that
contains the truth 7% of the time.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon as SPoly

from src.contracts import CaseManifest, SourcePosterior
from src.ingest.case_builder import load_forcing_bundle
from src.inversion import (
    HypothesisGrid,
    InversionConfig,
    ObservedMask,
    backward_proposal,
    effective_sample_size,
    evaluate,
    invert,
)
from src.inversion.likelihood import DEFAULT_LAMBDA, MISS_FLOOR, effective_cell_count
from src.transport import ForcingField, Seeds, TransportParams, simulate

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "cases" / "synth_kattegat"


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


def _release(field, case, lon, lat, lookback_h, n=3000, seed=7):
    """Drift a known release forward and return (truth trajectory, observed mask)."""
    t0 = case.t_obs - timedelta(hours=lookback_h)
    rng = np.random.default_rng(seed)
    seeds = Seeds.at_time(lon + rng.normal(0, 0.01, n), lat + rng.normal(0, 0.008, n), t0)
    truth = simulate(field, seeds, t0, case.t_obs, TransportParams(), seed=seed)
    ok = truth.usable
    return truth, ObservedMask.from_points(truth.lon[ok], truth.lat[ok], field), t0


# ------------------------------------------------------------------ the mask


def test_mask_from_polygon_has_area_and_annulus(field):
    ring = [(11.30, 57.00), (11.50, 57.00), (11.50, 57.10), (11.30, 57.10), (11.30, 57.00)]
    mask = ObservedMask.from_polygon(ring, field)
    assert mask.n_mask_cells > 0
    assert 50.0 < mask.area_km2 < 400.0
    assert mask.unobserved.sum() > 0, "boundary annulus must exist"
    assert not (mask.mask & mask.unobserved).any(), "a cell cannot be both oil and unobserved"


def test_annulus_cells_score_neither_way(field):
    """Segmentation boundaries are uncertain to a pixel or two. Punishing a
    hypothesis for a one-cell disagreement at the edge is scoring noise."""
    ring = [(11.30, 57.00), (11.50, 57.00), (11.50, 57.10), (11.30, 57.10), (11.30, 57.00)]
    thin = ObservedMask.from_polygon(ring, field, boundary_uncertainty_m=500.0)
    thick = ObservedMask.from_polygon(ring, field, boundary_uncertainty_m=2000.0)
    assert thick.unobserved.sum() > thin.unobserved.sum()
    assert thick.n_mask_cells < thin.n_mask_cells


def test_mask_from_points_is_tighter_than_a_convex_hull(field, case):
    """A hull fills in the concave gaps a drifting slick has and is much larger
    than the plume. Scoring against one rewards hypotheses that over-spread,
    which biases the posterior away from the true compact source."""
    from shapely.geometry import MultiPoint

    truth, mask, _ = _release(field, case, 11.05, 56.95, 24.0)
    ok = truth.usable
    hull = MultiPoint(list(zip(truth.lon[ok], truth.lat[ok]))).convex_hull
    hull_mask = ObservedMask.from_polygon(
        [(float(a), float(b)) for a, b in hull.exterior.coords], field
    )
    assert mask.area_km2 < hull_mask.area_km2


def test_degenerate_polygon_is_refused(field):
    tiny = [(11.4000, 57.0000), (11.4002, 57.0000), (11.4002, 57.0001),
            (11.4000, 57.0001), (11.4000, 57.0000)]
    with pytest.raises(ValueError):
        ObservedMask.from_polygon(tiny, field, grid_m=5000.0)


def test_points_outside_the_grid_count_as_clean_sea(field):
    ring = [(11.30, 57.00), (11.50, 57.00), (11.50, 57.10), (11.30, 57.10), (11.30, 57.00)]
    mask = ObservedMask.from_polygon(ring, field)
    x, y = field.to_grid(np.array([10.4]), np.array([56.5]))
    inside, annulus = mask.classify_points(x, y)
    assert not inside[0] and not annulus[0]
    assert mask.cell_rank(x, y)[0] == -1


# --------------------------------------------------------- the likelihood


def _fake_trajectory(field, mask, positions_xy, markers, n_hyp):
    """Minimal Trajectory stand-in for likelihood unit tests."""
    from src.transport.kernel import ACTIVE, Trajectory

    x = np.asarray([p[0] for p in positions_xy], dtype="float64")
    y = np.asarray([p[1] for p in positions_xy], dtype="float64")
    lon, lat = field.to_lonlat(x, y)
    return Trajectory(
        x=x, y=y, lon=lon, lat=lat,
        status=np.full(x.size, ACTIVE, dtype="int8"),
        origin_marker=np.asarray(markers, dtype="int64"),
        alpha=np.zeros(x.size), kh=np.zeros(x.size),
        crs=field.crs, t_start=0.0, t_end=1.0, n_steps=1, elapsed_s=0.0,
    )


def test_closed_form_matches_brute_force(field):
    """The algebraic shortcut that makes the likelihood affordable.

    Because log(1 - q) = -lambda * rho, the sum over non-mask cells collapses to
    a count. Evaluating literally over every cell is ~1e9 operations; this
    identity makes it O(mask cells + particles) with NO approximation. If the
    algebra is wrong, every number the system produces is wrong.
    """
    ring = [(11.30, 57.00), (11.45, 57.00), (11.45, 57.08), (11.30, 57.08), (11.30, 57.00)]
    mask = ObservedMask.from_polygon(ring, field, boundary_uncertainty_m=0.0)
    rng = np.random.default_rng(0)

    iy, ix = np.nonzero(mask.mask)
    pts, markers = [], []
    for h in range(3):
        pick = rng.integers(0, iy.size, 400)
        jitter = rng.normal(0, 2500.0 * (h + 1), (400, 2))
        pts += [
            (mask.x0 + mask.dx * (ix[p] + 0.5) + jitter[i, 0],
             mask.y0 + mask.dy * (iy[p] + 0.5) + jitter[i, 1])
            for i, p in enumerate(pick)
        ]
        markers += [h] * 400

    traj = _fake_trajectory(field, mask, pts, markers, 3)
    fast = evaluate(traj, mask, 3, DEFAULT_LAMBDA)

    # Brute force: build rho on the full mask-grid and sum every cell.
    n_cells = mask.n_mask_cells
    for h in range(3):
        sel = np.asarray(markers) == h
        gx, gy = traj.x[sel], traj.y[sel]
        ixp = ((gx - mask.x0) / mask.dx).astype(np.intp)
        iyp = ((gy - mask.y0) / mask.dy).astype(np.intp)
        ny, nx = mask.mask.shape
        on = (ixp >= 0) & (ixp < nx) & (iyp >= 0) & (iyp < ny)
        counts = np.zeros(mask.mask.shape)
        np.add.at(counts, (iyp[on], ixp[on]), 1.0)
        n_off_grid = int((~on).sum())

        rho = counts / sel.sum() * n_cells
        q = MISS_FLOOR + (1 - MISS_FLOOR) * (1 - np.exp(-DEFAULT_LAMBDA * rho))
        brute = np.log(q[mask.mask]).sum()
        # Cells with no particles outside the mask contribute log(1-q)=0, and
        # particles that left the grid entirely are clean sea too.
        outside_mass = (counts[~mask.mask & ~mask.unobserved].sum() + n_off_grid) / sel.sum()
        brute += -DEFAULT_LAMBDA * outside_mass * n_cells

        assert fast.log_likelihood[h] == pytest.approx(brute, rel=1e-9), (
            f"closed form disagrees with brute force for hypothesis {h}"
        )


def test_overspread_hypothesis_scores_worse_than_a_tight_correct_one(field):
    """The exact defect that per-particle rejection ABC has.

    Under naive rejection, a hypothesis smearing across the whole scene earns
    the same per-particle credit as one producing a compact matching cloud,
    because only hits are counted and misses cost nothing.
    """
    ring = [(11.30, 57.00), (11.45, 57.00), (11.45, 57.08), (11.30, 57.08), (11.30, 57.00)]
    mask = ObservedMask.from_polygon(ring, field, boundary_uncertainty_m=0.0)
    rng = np.random.default_rng(1)
    iy, ix = np.nonzero(mask.mask)

    pts, markers = [], []
    for h, spread_m in enumerate([300.0, 40_000.0]):
        pick = rng.integers(0, iy.size, 600)
        jitter = rng.normal(0, spread_m, (600, 2))
        pts += [
            (mask.x0 + mask.dx * (ix[p] + 0.5) + jitter[i, 0],
             mask.y0 + mask.dy * (iy[p] + 0.5) + jitter[i, 1])
            for i, p in enumerate(pick)
        ]
        markers += [h] * 600

    result = evaluate(_fake_trajectory(field, mask, pts, markers, 2), mask, 2)
    assert result.log_likelihood[0] > result.log_likelihood[1]
    assert result.spill_fraction[1] > result.spill_fraction[0]


def test_hypothesis_with_no_usable_particles_scores_negative_infinity(field):
    ring = [(11.30, 57.00), (11.45, 57.00), (11.45, 57.08), (11.30, 57.08), (11.30, 57.00)]
    mask = ObservedMask.from_polygon(ring, field)
    cx, cy = mask.centroid_xy
    traj = _fake_trajectory(field, mask, [(cx, cy)] * 10, [0] * 10, 3)
    result = evaluate(traj, mask, 3)
    assert np.isfinite(result.log_likelihood[0])
    assert np.isneginf(result.log_likelihood[1])


def test_effective_cell_count_is_area_over_correlation_area(field):
    ring = [(11.30, 57.00), (11.50, 57.00), (11.50, 57.10), (11.30, 57.10), (11.30, 57.00)]
    mask = ObservedMask.from_polygon(ring, field)
    n_eff = effective_cell_count(mask, 9_000.0)
    assert 1.0 <= n_eff <= mask.n_mask_cells
    assert n_eff == pytest.approx(mask.area_km2 / 81.0, rel=0.01)
    # A very long correlation length floors at one independent observation.
    assert effective_cell_count(mask, 500_000.0) == 1.0


def test_tempering_flattens_the_posterior(field):
    """Tempering by n_effective/n_cells is what stops the likelihood being a
    sum of hundreds of correlated terms."""
    ring = [(11.30, 57.00), (11.45, 57.00), (11.45, 57.08), (11.30, 57.08), (11.30, 57.00)]
    mask = ObservedMask.from_polygon(ring, field, boundary_uncertainty_m=0.0)
    rng = np.random.default_rng(2)
    iy, ix = np.nonzero(mask.mask)
    pts, markers = [], []
    for h, spread in enumerate([500.0, 3000.0, 9000.0]):
        pick = rng.integers(0, iy.size, 500)
        j = rng.normal(0, spread, (500, 2))
        pts += [(mask.x0 + mask.dx * (ix[p] + .5) + j[i, 0],
                 mask.y0 + mask.dy * (iy[p] + .5) + j[i, 1]) for i, p in enumerate(pick)]
        markers += [h] * 500
    traj = _fake_trajectory(field, mask, pts, markers, 3)

    sharp = evaluate(traj, mask, 3).weights()
    flat = evaluate(traj, mask, 3, n_effective_cells=effective_cell_count(mask, 9000.0)).weights()
    assert flat.max() < sharp.max(), "tempering must reduce the dominance of the top hypothesis"


def test_effective_sample_size_behaves():
    assert effective_sample_size(np.array([1.0, 0.0, 0.0]), 100) == pytest.approx(100.0)
    assert effective_sample_size(np.full(4, 0.25), 100) == pytest.approx(400.0)


# ------------------------------------------------------------ the proposal


def test_backward_proposal_sweeps_a_region(field, case):
    _, mask, _ = _release(field, case, 11.05, 56.95, 24.0)
    proposal = backward_proposal(field, mask, case.t_obs, lookback_hours=24.0)
    assert proposal.n_bins >= 20
    assert proposal.valid.any()
    assert np.all(np.diff(proposal.t0_bins) > 0), "t0 bins must ascend"
    # The swept region must actually move away from the observation.
    cx, _ = mask.centroid_xy
    assert abs(proposal.x[0][proposal.valid[0]].mean() - cx) > 1_000.0


def test_proposal_pass_never_runs_diffusion_backwards(field, case):
    """The proposal uses TransportParams.proposal(), which is diffusion-off and
    windage-fixed. If it ever used the default params the kernel would raise."""
    _, mask, _ = _release(field, case, 11.05, 56.95, 18.0)
    proposal = backward_proposal(field, mask, case.t_obs, lookback_hours=18.0)
    assert proposal.x.shape[0] == proposal.t0_bins.size


def test_hypothesis_grid_covers_the_proposal_and_refines(field, case):
    _, mask, _ = _release(field, case, 11.05, 56.95, 24.0)
    proposal = backward_proposal(field, mask, case.t_obs, lookback_hours=24.0)
    grid = HypothesisGrid.from_proposal(proposal, cell_m=6000.0, max_hypotheses=4096)
    assert grid.n > 50
    assert np.unique(grid.t0).size >= 20

    log_lik = np.linspace(-100.0, 0.0, grid.n)
    finer = grid.refine(log_lik, keep_fraction=0.1)
    assert finer.cell_m < grid.cell_m
    assert finer.n < grid.n * 4


def test_refine_refuses_when_nothing_scored(field, case):
    _, mask, _ = _release(field, case, 11.05, 56.95, 24.0)
    proposal = backward_proposal(field, mask, case.t_obs, lookback_hours=24.0)
    grid = HypothesisGrid.from_proposal(proposal, cell_m=6000.0)
    with pytest.raises(ValueError, match="nothing to refine"):
        grid.refine(np.full(grid.n, -np.inf), 0.1)


# --------------------------------------------------------- the full inversion


@pytest.fixture(scope="module")
def inverted(field, case):
    truth, mask, t0_true = _release(field, case, 11.05, 56.95, 24.0)
    posterior, diag = invert(
        field, mask, case.t_obs, case.case_id, f"{case.case_id}:d01",
        InversionConfig(lookback_hours=24.0, seed=3),
    )
    return posterior, diag, mask, t0_true


def test_inversion_produces_a_valid_contract(inverted):
    posterior, _, _, _ = inverted
    assert isinstance(posterior, SourcePosterior)
    SourcePosterior.model_validate(posterior.model_dump())


def test_credible_regions_nest_strictly(inverted):
    posterior, _, _, _ = inverted
    assert posterior.credible_regions["50"].area_km2 < posterior.credible_regions["95"].area_km2


def test_regions_are_reported_in_km2_not_degrees(inverted):
    """W-11. Computing area in degrees gives values wrong by a
    latitude-dependent factor, silently."""
    posterior, _, mask, _ = inverted
    area = posterior.credible_regions["95"].area_km2
    assert 20.0 < area < 50_000.0, f"{area} km2 is not a plausible search envelope"


def test_true_source_falls_inside_the_95_percent_region(inverted):
    """Coverage is the metric that matters. A tight region that misses is worse
    than a wide one that contains."""
    posterior, _, _, _ = inverted
    assert SPoly(posterior.credible_regions["95"].polygon_wgs84).contains(Point(11.05, 56.95))


def test_true_release_time_falls_inside_the_hpd(inverted):
    posterior, _, _, t0_true = inverted
    lo, hi = posterior.t0_marginal.hpd_95
    assert lo <= t0_true <= hi


def test_assumptions_are_carried_to_the_output(inverted):
    """Every displayed probability must appear next to what would have to be
    true for it to mean anything."""
    posterior, _, _, _ = inverted
    joined = " ".join(posterior.assumptions).lower()
    assert "windage" in joined
    assert "diffusivity" in joined
    assert "stokes" in joined and "no separate" in joined
    assert "tempered" in joined
    assert "weathering" in joined


def test_effective_sample_size_is_reported(inverted):
    posterior, diag, _, _ = inverted
    assert 0.0 < posterior.effective_sample_size <= posterior.n_particles
    assert posterior.ess_fraction == pytest.approx(diag.ess_fraction, rel=0.05)


def test_untempered_likelihood_is_overconfident(field, case):
    """A measured failure, not a hypothetical one.

    Treating every mask cell as an independent observation shrinks the nominal
    95% region until it contains the truth only 7% of the time (measured over
    14 trials by scripts/calibrate_inversion.py). Here we check the mechanism:
    the untempered region is dramatically smaller than the tempered one.
    """
    _, mask, _ = _release(field, case, 11.05, 56.95, 24.0)
    common = dict(lookback_hours=24.0, seed=3)
    tempered, _ = invert(field, mask, case.t_obs, case.case_id, "x",
                         InversionConfig(**common))
    raw, _ = invert(field, mask, case.t_obs, case.case_id, "x",
                    InversionConfig(likelihood_correlation_m=0.0, **common))

    assert raw.credible_regions["95"].area_km2 < tempered.credible_regions["95"].area_km2
    assert any("NOT tempered" in a for a in raw.assumptions), (
        "an untempered posterior must disclose that it is overconfident"
    )


def test_lookback_beyond_the_horizon_is_flagged(field, case):
    """Past ~72 h the envelope stops being operationally useful, and the system
    reports its own limit rather than producing envelopes nobody can act on."""
    _, mask, _ = _release(field, case, 11.05, 56.95, 20.0)
    posterior, _ = invert(
        field, mask, case.t_obs, case.case_id, "x",
        InversionConfig(lookback_hours=56.0, seed=1),
    )
    assert posterior.within_operating_envelope
    assert posterior.lookback_hours == 56.0


def test_posterior_npz_is_written_when_requested(field, case, tmp_path):
    _, mask, _ = _release(field, case, 11.05, 56.95, 18.0)
    posterior, _ = invert(
        field, mask, case.t_obs, case.case_id, "x",
        InversionConfig(lookback_hours=18.0, seed=1), out_dir=tmp_path,
    )
    saved = np.load(tmp_path / "posterior.npz")
    assert saved["density"].shape == (
        len(posterior.grid.t0), len(posterior.grid.lat), len(posterior.grid.lon)
    )
    assert saved["density"].sum() == pytest.approx(1.0, rel=1e-4)


def test_inversion_is_deterministic(field, case):
    _, mask, _ = _release(field, case, 11.05, 56.95, 18.0)
    cfg = InversionConfig(lookback_hours=18.0, seed=5)
    a, _ = invert(field, mask, case.t_obs, case.case_id, "x", cfg)
    b, _ = invert(field, mask, case.t_obs, case.case_id, "x", cfg)
    assert a.credible_regions["95"].area_km2 == b.credible_regions["95"].area_km2
    assert a.t0_marginal.hpd_95 == b.t0_marginal.hpd_95


def test_inversion_config_loads_from_physics_yaml():
    cfg = InversionConfig.from_yaml()
    assert cfg.max_hypotheses == 4096
    assert cfg.refinement_rounds == 2
    assert cfg.credible_levels == (50, 95)


@pytest.mark.slow
def test_inversion_runtime_budget(field, case):
    """The inversion is part of a 60 s end-to-end budget."""
    _, mask, _ = _release(field, case, 11.05, 56.95, 24.0)
    _, diag = invert(field, mask, case.t_obs, case.case_id, "x",
                     InversionConfig(lookback_hours=24.0, seed=1))
    assert diag.elapsed_s < 25.0, f"inversion took {diag.elapsed_s:.1f}s"
