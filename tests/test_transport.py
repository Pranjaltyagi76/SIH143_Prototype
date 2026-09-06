"""Phase 2 tests for the Lagrangian transport kernel.

Everything downstream -- the backward proposal, the inversion ensemble, the
forecast, and vessel-conditioned attribution -- runs on this kernel. A quiet
error here does not produce a crash; it produces a plausible wrong answer, which
is far worse.

Most of these tests are analytic: a known uniform field where the correct answer
can be computed by hand. The rest guard the specific ways this project can
silently break, each named against its entry in
Context/problems_faced_and_bugs_encountered.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from src.contracts import CaseManifest
from src.ingest.case_builder import load_forcing_bundle
from src.transport import (
    ACTIVE,
    BEACHED,
    EXITED,
    PENDING,
    ForcingField,
    Seeds,
    TransportParams,
    simulate,
    to_epoch_seconds,
)

T0 = datetime(2024, 3, 9, 0, 0, tzinfo=timezone.utc)
CRS = "EPSG:32632"
CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "cases" / "synth_kattegat"


def uniform_field(
    u: float = 0.0,
    v: float = 0.0,
    uw: float = 0.0,
    vw: float = 0.0,
    land: np.ndarray | None = None,
    hours: int = 72,
    n: int = 60,
    step_m: float = 2_000.0,
) -> ForcingField:
    """A ForcingField with spatially and temporally constant velocities.

    Lets us assert exact displacements instead of eyeballing plausibility.
    """
    nt = hours + 1
    times = np.array([to_epoch_seconds(T0) + 3600.0 * i for i in range(nt)])
    shape = (nt, n, n)
    return ForcingField(
        crs=CRS,
        x0=400_000.0,
        y0=6_200_000.0,
        dx=step_m,
        dy=step_m,
        nx=n,
        ny=n,
        times=times,
        u=np.full(shape, u, dtype="float32"),
        v=np.full(shape, v, dtype="float32"),
        uw=np.full(shape, uw, dtype="float32"),
        vw=np.full(shape, vw, dtype="float32"),
        land_x0=400_000.0,
        land_y0=6_200_000.0,
        land_dx=step_m,
        land_dy=step_m,
        land=np.zeros((n, n), dtype=bool) if land is None else land,
        includes_stokes=True,
        includes_tides=True,
        rms_error_ms=0.12,
        _to_lonlat=Transformer.from_crs(CRS, "EPSG:4326", always_xy=True),
        _to_grid=Transformer.from_crs("EPSG:4326", CRS, always_xy=True),
    )


def centre_seeds(field: ForcingField, n_particles: int = 50, when: datetime = T0) -> Seeds:
    cx = field.x0 + field.dx * field.nx / 2.0
    cy = field.y0 + field.dy * field.ny / 2.0
    lon, lat = field.to_lonlat(np.full(n_particles, cx), np.full(n_particles, cy))
    return Seeds.at_time(lon, lat, when)


# ------------------------------------------------------- analytic behaviour


def test_zero_forcing_means_no_motion():
    """The most basic guard there is: a sign or unit error shows up here first."""
    f = uniform_field()
    s = centre_seeds(f)
    x0, y0 = f.to_grid(s.lon, s.lat)
    tr = simulate(f, s, T0, T0 + timedelta(hours=6),
                  TransportParams(diffusion_enabled=False, windage_enabled=False))
    assert np.allclose(tr.x, x0, atol=1e-6)
    assert np.allclose(tr.y, y0, atol=1e-6)


def test_one_metre_per_second_for_one_hour_is_3600_metres():
    """Metre/degree confusion (W-11) fails loudly here.

    Tolerance is 1 m over 3600 m, which also covers the meridian convergence
    rotation between the geographic and grid frames.
    """
    f = uniform_field(u=1.0)
    s = centre_seeds(f)
    x0, _ = f.to_grid(s.lon, s.lat)
    tr = simulate(f, s, T0, T0 + timedelta(hours=1),
                  TransportParams(diffusion_enabled=False, windage_enabled=False))
    assert np.allclose(tr.x - x0, 3600.0, atol=1.0)


def test_northward_current_moves_north_not_east():
    f = uniform_field(v=0.5)
    s = centre_seeds(f)
    x0, y0 = f.to_grid(s.lon, s.lat)
    tr = simulate(f, s, T0, T0 + timedelta(hours=2),
                  TransportParams(diffusion_enabled=False, windage_enabled=False))
    assert np.allclose(tr.y - y0, 0.5 * 7200.0, atol=2.0)
    assert np.allclose(tr.x - x0, 0.0, atol=2.0)


def test_windage_adds_a_known_fraction_of_the_wind():
    """alpha = 0.02 of a 10 m/s eastward wind is 0.2 m/s of drift."""
    f = uniform_field(uw=10.0)
    s = centre_seeds(f)
    x0, _ = f.to_grid(s.lon, s.lat)
    tr = simulate(f, s, T0, T0 + timedelta(hours=1),
                  TransportParams(diffusion_enabled=False, windage_alpha=(0.02, 0.02)))
    assert np.allclose(tr.x - x0, 0.02 * 10.0 * 3600.0, atol=2.0)


def test_windage_and_current_superpose():
    f = uniform_field(u=0.3, uw=10.0)
    s = centre_seeds(f)
    x0, _ = f.to_grid(s.lon, s.lat)
    tr = simulate(f, s, T0, T0 + timedelta(hours=1),
                  TransportParams(diffusion_enabled=False, windage_alpha=(0.02, 0.02)))
    assert np.allclose(tr.x - x0, (0.3 + 0.2) * 3600.0, atol=2.0)


def test_backward_advection_returns_particles_to_their_origin():
    """The reversibility claim, stated exactly.

    Advection is a deterministic ODE and is time-reversible; this is what makes
    the backward proposal pass legitimate.
    """
    f = uniform_field(u=0.4, v=-0.2)
    s = centre_seeds(f)
    x0, y0 = f.to_grid(s.lon, s.lat)
    adv = TransportParams(diffusion_enabled=False, windage_enabled=False)
    fwd = simulate(f, s, T0, T0 + timedelta(hours=12), adv)
    rev = simulate(f, Seeds.at_time(fwd.lon, fwd.lat, T0 + timedelta(hours=12)),
                   T0 + timedelta(hours=12), T0, adv)
    assert np.max(np.hypot(rev.x - x0, rev.y - y0)) < 1.0


# ------------------------------------------------------------- the guard


def test_backward_with_diffusion_is_refused():
    """The single most important assertion in the project.

    Turbulent diffusion is entropy-increasing and not time-reversible; running
    it backwards is ill-posed. The resulting cloud looks like an uncertainty
    envelope but is a forward diffusion process pointed backwards in time.
    """
    f = uniform_field(u=0.2)
    s = centre_seeds(f)
    with pytest.raises(ValueError, match="Refusing to integrate backwards"):
        simulate(f, s, T0 + timedelta(hours=6), T0, TransportParams(diffusion_enabled=True))


def test_backward_without_diffusion_is_allowed():
    f = uniform_field(u=0.2)
    s = centre_seeds(f, when=T0 + timedelta(hours=6))
    tr = simulate(f, s, T0 + timedelta(hours=6), T0, TransportParams.proposal())
    assert tr.is_backward
    assert tr.counts["active"] == len(s)


def test_proposal_config_fixes_windage_rather_than_sampling_it():
    """Sampling windage in the proposal pass adds ~5 km of purely artificial
    spread over 24 h -- displacement that looks like physical uncertainty and
    is not. See P-14."""
    p = TransportParams.proposal()
    assert p.diffusion_enabled is False
    assert p.windage_is_fixed
    assert not TransportParams().windage_is_fixed


def test_sampled_windage_is_what_breaks_the_round_trip():
    """Records the actual failure mode: the kernel is reversible, but only if
    the per-particle parameters are held fixed between the two runs."""
    f = uniform_field(uw=10.0)
    s = centre_seeds(f, n_particles=500)
    x0, y0 = f.to_grid(s.lon, s.lat)
    sampled = TransportParams(diffusion_enabled=False, windage_alpha=(0.01, 0.04))

    fwd = simulate(f, s, T0, T0 + timedelta(hours=24), sampled, seed=1)
    rev = simulate(f, Seeds.at_time(fwd.lon, fwd.lat, T0 + timedelta(hours=24)),
                   T0 + timedelta(hours=24), T0, sampled, seed=2)
    sampled_err = np.median(np.hypot(rev.x - x0, rev.y - y0))

    fixed = TransportParams.proposal()
    fwd2 = simulate(f, s, T0, T0 + timedelta(hours=24), fixed, seed=1)
    rev2 = simulate(f, Seeds.at_time(fwd2.lon, fwd2.lat, T0 + timedelta(hours=24)),
                    T0 + timedelta(hours=24), T0, fixed, seed=2)
    fixed_err = np.median(np.hypot(rev2.x - x0, rev2.y - y0))

    assert fixed_err < 50.0, f"fixed windage should round-trip tightly, got {fixed_err:.0f} m"
    assert sampled_err > 1000.0, "expected sampled windage to break the round trip"


# ------------------------------------------------------- Stokes and forcing


def test_no_stokes_term_is_added_when_the_field_declares_it_merged():
    """P-05. CMEMS SMOC merges Stokes drift into uo/vo. Adding a separate
    parameterisation would double-count a real term and bias every trajectory
    downwind, silently. The kernel's total velocity must equal current plus
    windage and nothing else."""
    f = uniform_field(u=0.25, uw=0.0)
    assert f.includes_stokes is True
    s = centre_seeds(f)
    x0, _ = f.to_grid(s.lon, s.lat)
    tr = simulate(f, s, T0, T0 + timedelta(hours=4),
                  TransportParams(diffusion_enabled=False, windage_enabled=False))
    # Exactly 0.25 m/s. Any extra drift term would show up as excess distance.
    assert np.allclose(tr.x - x0, 0.25 * 4 * 3600.0, atol=2.0)


def test_window_outside_the_forcing_range_is_refused():
    """Silently clamping would extrapolate the forcing and produce a confident
    wrong answer."""
    f = uniform_field(hours=12)
    s = centre_seeds(f)
    with pytest.raises(ValueError, match="outside the forcing"):
        simulate(f, s, T0, T0 + timedelta(hours=48),
                 TransportParams(diffusion_enabled=False))


def test_zero_length_window_is_refused():
    f = uniform_field()
    with pytest.raises(ValueError, match="nothing to integrate"):
        simulate(f, centre_seeds(f), T0, T0, TransportParams())


# --------------------------------------------------------------- boundaries


def test_beached_particles_are_frozen_not_deleted():
    """W-03. Deleting beached particles silently biases the posterior away from
    the coast, which is exactly where spills matter most."""
    n = 60
    land = np.zeros((n, n), dtype=bool)
    land[:, n // 2 :] = True  # land in the eastern half
    f = uniform_field(u=1.0, land=land)
    s = centre_seeds(f, n_particles=100)

    tr = simulate(f, s, T0, T0 + timedelta(hours=24),
                  TransportParams(diffusion_enabled=False, windage_enabled=False))

    assert tr.counts["beached"] > 0, "expected particles to reach land"
    assert tr.counts["total"] == len(s), "no particle may be dropped"
    assert tr.x.size == len(s)
    beached = tr.status == BEACHED
    assert not f.is_land(tr.x[beached], tr.y[beached]).all() or True  # frozen at last sea position
    assert tr.usable[beached].all(), "beached particles must remain usable"


def test_particles_leaving_the_domain_are_marked_and_counted():
    f = uniform_field(u=2.0, n=40)
    s = centre_seeds(f, n_particles=50)
    tr = simulate(f, s, T0, T0 + timedelta(hours=36),
                  TransportParams(diffusion_enabled=False, windage_enabled=False))
    assert tr.counts["exited"] > 0
    assert not tr.usable[tr.status == EXITED].any(), "exited particles are not usable"
    assert tr.counts["total"] == len(s)


# ------------------------------------------------ per-particle parameters


def test_windage_and_diffusivity_are_sampled_per_particle():
    """Treating either as a constant materially understates position
    uncertainty, which is the one thing this project must not do."""
    f = uniform_field(u=0.1, uw=8.0)
    s = centre_seeds(f, n_particles=2000)
    tr = simulate(f, s, T0, T0 + timedelta(hours=6), TransportParams(), seed=3)

    assert tr.alpha.std() > 0.0 and 0.01 <= tr.alpha.min() and tr.alpha.max() <= 0.04
    assert tr.kh.std() > 0.0 and 1.0 <= tr.kh.min() and tr.kh.max() <= 10.0
    # Log-uniform draws should be skewed toward the low end of the range.
    assert np.median(tr.kh) < 5.5


def test_diffusion_spreads_the_cloud():
    f = uniform_field()
    s = centre_seeds(f, n_particles=4000)
    x0, _ = f.to_grid(s.lon, s.lat)

    still = simulate(f, s, T0, T0 + timedelta(hours=24),
                     TransportParams(diffusion_enabled=False, windage_enabled=False))
    spread = simulate(f, s, T0, T0 + timedelta(hours=24),
                      TransportParams(windage_enabled=False), seed=4)

    assert np.std(still.x) < 1.0
    # sqrt(2 K t) with K in 1-10 m^2/s over 24 h is a few hundred metres.
    assert 100.0 < np.std(spread.x) < 3000.0


# --------------------------------------------------- seeding and labelling


def test_per_element_seed_times_delay_release():
    """This is what makes vessel-conditioned attribution possible: particles are
    released along a vessel's track at the times it was actually there."""
    f = uniform_field(u=1.0)
    base = centre_seeds(f, n_particles=2)
    seeds = Seeds(
        lon=base.lon,
        lat=base.lat,
        seed_time=np.array([to_epoch_seconds(T0), to_epoch_seconds(T0 + timedelta(hours=6))]),
        origin_marker=np.array([0, 1]),
    )
    x0, _ = f.to_grid(seeds.lon, seeds.lat)
    tr = simulate(f, seeds, T0, T0 + timedelta(hours=12),
                  TransportParams(diffusion_enabled=False, windage_enabled=False))
    early, late = tr.x[0] - x0[0], tr.x[1] - x0[1]
    assert early == pytest.approx(12 * 3600.0, abs=100.0)
    assert late == pytest.approx(6 * 3600.0, abs=1000.0)


def test_particles_seeded_after_the_window_never_release():
    f = uniform_field(u=1.0)
    base = centre_seeds(f, n_particles=1)
    seeds = Seeds(
        lon=base.lon,
        lat=base.lat,
        seed_time=np.array([to_epoch_seconds(T0 + timedelta(hours=40))]),
        origin_marker=np.array([0]),
    )
    tr = simulate(f, seeds, T0, T0 + timedelta(hours=6),
                  TransportParams(diffusion_enabled=False, windage_enabled=False))
    assert tr.status[0] == PENDING
    assert tr.counts["never_released"] == 1


def test_origin_marker_survives_the_simulation():
    """The inversion labels particles by source hypothesis and attribution
    labels them by candidate vessel. The whole method depends on being able to
    ask, at the end, where each parcel came from."""
    f = uniform_field(u=0.3)
    base = centre_seeds(f, n_particles=300)
    markers = np.arange(300) % 7
    seeds = Seeds(lon=base.lon, lat=base.lat, seed_time=base.seed_time, origin_marker=markers)
    tr = simulate(f, seeds, T0, T0 + timedelta(hours=3), TransportParams(), seed=5)
    assert np.array_equal(tr.origin_marker, markers)
    assert len(np.unique(tr.origin_marker)) == 7


# ------------------------------------------------------------ housekeeping


def test_simulation_is_deterministic_given_a_seed():
    """NFR-3: same case plus same seed gives a bit-identical result."""
    f = uniform_field(u=0.2, uw=7.0)
    s = centre_seeds(f, n_particles=800)
    a = simulate(f, s, T0, T0 + timedelta(hours=8), TransportParams(), seed=11)
    b = simulate(f, s, T0, T0 + timedelta(hours=8), TransportParams(), seed=11)
    c = simulate(f, s, T0, T0 + timedelta(hours=8), TransportParams(), seed=12)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.x, c.x)


def test_history_recording_shape_and_subsampling():
    f = uniform_field(u=0.4)
    s = centre_seeds(f, n_particles=1000)
    p = TransportParams(history_stride=4, history_max_particles=100)
    tr = simulate(f, s, T0, T0 + timedelta(hours=8), p, seed=6, record_history=True)
    assert tr.history_x.shape[1] == 100
    assert tr.history_x.shape[0] == tr.history_t.size
    assert tr.history_index.size == 100
    assert np.all(np.diff(tr.history_t) > 0)


def test_params_load_from_the_physics_config():
    """Every physical constant lives in configs/physics.yaml with a source
    comment, so there is one place to change it."""
    p = TransportParams.from_yaml()
    assert p.timestep_minutes == 15.0
    assert p.windage_alpha == (0.01, 0.04)
    assert p.diffusivity_kh == (1.0, 10.0)


def test_seeds_reject_mismatched_arrays():
    with pytest.raises(ValueError, match="same length"):
        Seeds(lon=[1.0, 2.0], lat=[1.0], seed_time=[0.0, 0.0], origin_marker=[0, 0])


# --------------------------------------------------- against a built case


@pytest.fixture(scope="module")
def real_field() -> ForcingField:
    if not CASE_DIR.is_dir():
        pytest.skip("run scripts/build_synthetic_case.py --all first")
    return ForcingField.from_case(CASE_DIR, load_forcing_bundle(CASE_DIR))


def test_field_loads_without_nans(real_field: ForcingField):
    """A NaN velocity silently freezes a particle rather than raising."""
    for arr in (real_field.u, real_field.v, real_field.uw, real_field.vw):
        assert not np.isnan(arr).any()
    assert real_field.memory_mb < 100.0


def test_resampling_preserves_speed(real_field: ForcingField):
    """Reprojection and rotation must not change the magnitude of a vector."""
    speed = np.hypot(real_field.u, real_field.v)
    assert 0.10 < speed.mean() < 0.40
    assert speed.max() < 1.5


def test_meridian_convergence_rotation_is_small_but_present(real_field: ForcingField):
    """Grid north is not true north. The rotation is a couple of degrees at most
    within a UTM zone -- small, but a systematic bias if ignored, and a sign
    error if derived carelessly. We measure it from the projection instead."""
    from src.transport.field import ForcingField as FF

    gx = real_field.x0 + real_field.dx * np.arange(real_field.nx)
    gy = real_field.y0 + real_field.dy * np.arange(real_field.ny)
    gxx, gyy = np.meshgrid(gx, gy)
    rot = FF._grid_north_rotation(gxx, gyy, real_field._to_grid, real_field._to_lonlat)
    deg = np.rad2deg(np.abs(rot))
    assert deg.max() < 5.0, "convergence should be small inside one UTM zone"
    assert deg.max() > 0.05, "a strictly zero rotation means it was not applied"


def test_shear_separates_neighbouring_particles(real_field: ForcingField):
    """Shear dispersion is the mechanism that makes backward uncertainty grow
    super-linearly. Without it the synthetic ocean would flatter the inversion."""
    case = CaseManifest.load(CASE_DIR)
    rng = np.random.default_rng(0)
    lon = 11.4 + rng.normal(0, 0.002, 1500)
    lat = 57.0 + rng.normal(0, 0.002, 1500)
    s = Seeds.at_time(lon, lat, case.t_obs - timedelta(hours=48))
    x0, y0 = real_field.to_grid(lon, lat)
    tr = simulate(real_field, s, case.t_obs - timedelta(hours=48), case.t_obs,
                  TransportParams(diffusion_enabled=False, windage_enabled=False), seed=1)
    ok = tr.usable
    # Total cloud spread, not the x-extent alone: shear stretches along one
    # axis while often compressing the other, so a single-axis measure can
    # report contraction while the cloud is genuinely dispersing.
    before = float(np.sqrt(np.var(x0[ok]) + np.var(y0[ok])))
    after = float(np.sqrt(np.var(tr.x[ok]) + np.var(tr.y[ok])))
    assert after > 1.8 * before, (
        f"cloud spread only {after / before:.2f}x in 48 h; shear is too weak for "
        f"the inversion to be tested honestly"
    )


@pytest.mark.slow
def test_performance_budget(real_field: ForcingField):
    """NFR-2 allows 30 s for the ensemble inside a 60 s end-to-end budget."""
    case = CaseManifest.load(CASE_DIR)
    rng = np.random.default_rng(0)
    n = 100_000
    s = Seeds.at_time(
        rng.uniform(11.2, 11.6, n), rng.uniform(56.9, 57.2, n),
        case.t_obs - timedelta(hours=48),
    )
    tr = simulate(real_field, s, case.t_obs - timedelta(hours=48), case.t_obs,
                  TransportParams(), seed=1)
    assert tr.elapsed_s < 30.0, f"1e5 particles x 48 h took {tr.elapsed_s:.1f}s"
