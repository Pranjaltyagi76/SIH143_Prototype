"""The inversion: backward proposal, forward ensemble, conditioning, posterior.

    1. PROPOSAL    Backward advection-only sweep from the observed slick.
                   Legitimate, because pure advection is time-reversible.
                   Yields the space-time region worth searching.

    2. HYPOTHESES  Partition that region into (position, release-time) cells.

    3. PROPAGATE   Seed particles per hypothesis, tagged by origin_marker, and
                   run ONE forward simulation with full stochastic physics.

    4. CONDITION   Score each hypothesis with the Bernoulli observation
                   operator against the observed mask.

    5. REFINE      Subdivide the best cells and repeat once. Two coarse-to-fine
                   rounds cost less than one fine blind pass.

    6. POSTERIOR   Normalise, smooth, extract credible regions in km^2 and the
                   release-time marginal.

Physics only ever runs forward. Turbulent diffusion is entropy-increasing and
not time-reversible; the kernel refuses to integrate it backwards. Backward
integration narrows the search, forward simulation computes the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.contracts import (
    MAX_USEFUL_LOOKBACK_HOURS,
    CredibleRegion,
    PosteriorGrid,
    SourcePosterior,
    TimeMarginal,
)
from src.inversion.hypotheses import HypothesisGrid, backward_proposal
from src.inversion.likelihood import (
    DEFAULT_LAMBDA,
    effective_cell_count,
    effective_sample_size,
    evaluate,
)
from src.inversion.mask import ObservedMask
from src.transport import ForcingField, TransportParams, simulate

EARTH_RADIUS_M = 6_371_000.0
M_PER_DEG_LAT = EARTH_RADIUS_M * np.pi / 180.0


@dataclass(frozen=True)
class InversionConfig:
    """Knobs for the inversion. Defaults mirror ``configs/physics.yaml``."""

    lookback_hours: float = 36.0
    t0_bin_hours: float = 1.0

    # Initial hypothesis cell size, metres. Coarsened automatically if the
    # proposal region is large enough to blow the hypothesis budget -- reporting
    # a coarser posterior is honest, silently shrinking the search is not.
    cell_m: float = 6_000.0
    max_hypotheses: int = 4_096

    particles_round1: int = 40
    particles_round2: int = 110
    refinement_rounds: int = 2
    refinement_keep_fraction: float = 0.10

    detectability_lambda: float = DEFAULT_LAMBDA
    min_ess_fraction: float = 0.01

    # Spatial decorrelation scale of the observation, metres. Mask cells closer
    # together than this carry no independent information about a source
    # hypothesis, because the drift model cannot resolve structure below the
    # forcing grid scale. Defaults to the forcing resolution when None.
    # See likelihood.effective_cell_count -- this is what keeps the posterior
    # from being wildly overconfident.
    likelihood_correlation_m: float | None = None

    # Posterior smoothing, in grid cells / time bins. A separable Gaussian on
    # the histogram, not a full kernel density estimate: orders of magnitude
    # faster and adequate at this resolution (debt item D8).
    smooth_cells: float = 1.0
    smooth_bins: float = 1.0

    credible_levels: tuple[int, ...] = (50, 95)
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: Path | str = "configs/physics.yaml", **overrides) -> InversionConfig:
        import yaml

        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["inversion"]
        base = dict(
            lookback_hours=float(cfg["default_lookback_hours"]),
            max_hypotheses=int(cfg["n_hypotheses"]),
            refinement_rounds=int(cfg["refinement_rounds"]),
            refinement_keep_fraction=float(cfg["refinement_keep_fraction"]),
            detectability_lambda=float(cfg["detectability_lambda"]),
            min_ess_fraction=float(cfg["min_ess_fraction"]),
            credible_levels=tuple(cfg["credible_levels"]),
        )
        return cls(**{**base, **overrides})


@dataclass
class InversionDiagnostics:
    """What actually happened, for the log and for honest reporting."""

    n_proposal_particles: int
    rounds: list[dict] = _field(default_factory=list)
    elapsed_s: float = 0.0
    ess_fraction: float = 0.0
    posterior_is_reliable: bool = True
    n_region_components: dict[str, int] = _field(default_factory=dict)


# ------------------------------------------------------------------- gridding


def _lonlat_axes(lon: np.ndarray, lat: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Regular lon/lat axes at roughly ``cell_m`` spacing, covering the points.

    The posterior is exported on a regular geographic grid because that is what
    the contract carries and what the interface consumes. Areas are still
    computed metrically, per cell, with the cos(latitude) factor -- computing
    area in degrees is watch-list item W-11.
    """
    mid_lat = float(np.mean(lat))
    d_lat = cell_m / M_PER_DEG_LAT
    d_lon = cell_m / (M_PER_DEG_LAT * np.cos(np.deg2rad(mid_lat)))

    lon_axis = np.arange(lon.min() - d_lon, lon.max() + 2 * d_lon, d_lon)
    lat_axis = np.arange(lat.min() - d_lat, lat.max() + 2 * d_lat, d_lat)
    return lon_axis, lat_axis


def _cell_area_km2(lat_axis: np.ndarray, d_lon: float, d_lat: float) -> np.ndarray:
    """Area of one grid cell at each latitude, km^2."""
    return (d_lat * M_PER_DEG_LAT) * (
        d_lon * M_PER_DEG_LAT * np.cos(np.deg2rad(lat_axis))
    ) / 1e6


def _highest_density_region(
    density2d: np.ndarray, level: float
) -> tuple[np.ndarray, float]:
    """Smallest cell set holding ``level`` of the mass. Returns (mask, threshold)."""
    flat = density2d.ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    total = cumulative[-1]
    if total <= 0:
        return np.zeros_like(density2d, dtype=bool), 0.0
    n_needed = int(np.searchsorted(cumulative, level * total) + 1)
    keep = np.zeros(flat.size, dtype=bool)
    keep[order[:n_needed]] = True
    return keep.reshape(density2d.shape), float(flat[order[n_needed - 1]])


def _region_polygon(
    region: np.ndarray, lon_axis: np.ndarray, lat_axis: np.ndarray
) -> tuple[list[tuple[float, float]], int]:
    """Vectorise a boolean region into its largest connected ring.

    Returns the ring and the number of connected components. When a posterior is
    multi-modal the reported polygon is the largest lobe, while the reported
    area covers all of them -- the discrepancy is disclosed in ``assumptions``
    rather than smoothed over.
    """
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from shapely.geometry import shape

    d_lon = float(lon_axis[1] - lon_axis[0])
    d_lat = float(lat_axis[1] - lat_axis[0])
    transform = from_origin(
        lon_axis[0] - d_lon / 2, lat_axis[-1] + d_lat / 2, d_lon, d_lat
    )

    geoms = [
        shape(geom)
        for geom, value in shapes(
            np.flipud(region).astype("uint8"), mask=np.flipud(region), transform=transform
        )
        if value == 1
    ]
    if not geoms:
        raise ValueError("credible region vectorised to nothing")

    biggest = max(geoms, key=lambda g: g.area)
    ring = [(float(x), float(y)) for x, y in biggest.exterior.simplify(d_lon / 4).coords]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    if len(ring) < 4:
        raise ValueError("credible region ring degenerated")
    return ring, len(geoms)


# ---------------------------------------------------------------- the entry point


def invert(
    field: ForcingField,
    mask: ObservedMask,
    t_obs: datetime,
    case_id: str,
    detection_id: str,
    config: InversionConfig | None = None,
    transport: TransportParams | None = None,
    out_dir: Path | None = None,
) -> tuple[SourcePosterior, InversionDiagnostics]:
    """Solve for where and when the observed slick was released."""
    import time as _time

    from scipy.ndimage import gaussian_filter

    config = config or InversionConfig()
    transport = transport or TransportParams()
    rng = np.random.default_rng(config.seed)
    started = _time.perf_counter()

    if config.lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive")

    # --- 1. backward proposal ------------------------------------------------
    proposal = backward_proposal(
        field, mask, t_obs, config.lookback_hours, config.t0_bin_hours, seed=config.seed
    )
    diag = InversionDiagnostics(n_proposal_particles=int(proposal.x.shape[1]))

    grid = HypothesisGrid.from_proposal(
        proposal, cell_m=config.cell_m, max_hypotheses=config.max_hypotheses
    )

    # --- 2-5. forward ensemble, condition, refine ----------------------------
    t_begin = t_obs - timedelta(hours=config.lookback_hours)
    result = None
    per_hypothesis = config.particles_round1

    correlation_m = config.likelihood_correlation_m
    if correlation_m is None:
        # The forcing resolution is the scale below which the drift model has
        # no information, so it is the natural decorrelation length.
        correlation_m = field.dx * 4.5
    # A correlation length of zero means "treat every cell as independent",
    # which is wrong and badly overconfident. It exists only so the failure can
    # be demonstrated rather than asserted.
    n_eff_cells = None if correlation_m <= 0 else effective_cell_count(mask, correlation_m)

    for round_index in range(max(1, config.refinement_rounds)):
        if round_index > 0:
            grid = grid.refine(result.log_likelihood, config.refinement_keep_fraction)
            per_hypothesis = config.particles_round2

        seeds = grid.seeds(field, per_hypothesis, rng)
        traj = simulate(
            field, seeds, t_begin, t_obs, transport,
            seed=config.seed + round_index,
        )
        result = evaluate(
            traj, mask, grid.n, config.detectability_lambda,
            n_effective_cells=n_eff_cells,
        )

        diag.rounds.append(
            {
                "round": round_index + 1,
                "hypotheses": grid.n,
                "particles": int(len(seeds)),
                "cell_m": round(grid.cell_m, 1),
                "seconds": round(traj.elapsed_s, 2),
                "best_log_likelihood": float(np.nanmax(result.log_likelihood)),
                "scored": int(np.isfinite(result.log_likelihood).sum()),
            }
        )

    weights = result.weights()
    ess = effective_sample_size(weights, per_hypothesis)
    total_particles = grid.n * per_hypothesis
    diag.ess_fraction = ess / max(total_particles, 1)
    diag.posterior_is_reliable = diag.ess_fraction >= config.min_ess_fraction

    # --- 6. posterior --------------------------------------------------------
    h_lon, h_lat = field.to_lonlat(grid.x, grid.y)
    lon_axis, lat_axis = _lonlat_axes(h_lon, h_lat, grid.cell_m)
    t_axis = np.unique(grid.t0)
    if t_axis.size < 2:
        t_axis = np.array([t_axis[0] - 3600.0, t_axis[0]])

    d_lon = float(lon_axis[1] - lon_axis[0])
    d_lat = float(lat_axis[1] - lat_axis[0])

    ix = np.clip(((h_lon - lon_axis[0]) / d_lon).round().astype(np.intp), 0, lon_axis.size - 1)
    iy = np.clip(((h_lat - lat_axis[0]) / d_lat).round().astype(np.intp), 0, lat_axis.size - 1)
    it = np.clip(np.searchsorted(t_axis, grid.t0), 0, t_axis.size - 1)

    volume = np.zeros((t_axis.size, lat_axis.size, lon_axis.size), dtype="float64")
    np.add.at(volume, (it, iy, ix), weights)

    volume = gaussian_filter(
        volume, sigma=(config.smooth_bins, config.smooth_cells, config.smooth_cells)
    )
    total = volume.sum()
    if total <= 0:
        raise ValueError("posterior collapsed to zero mass; the inversion found nothing")
    volume /= total

    spatial = volume.sum(axis=0)
    areas = _cell_area_km2(lat_axis, d_lon, d_lat)[:, None]

    regions: dict[str, CredibleRegion] = {}
    previous_cells = None
    # Widest level first, so each successive (inner) level can be checked
    # against the one enclosing it. Iterating the other way would trim the
    # OUTER region down to the inner one, which is backwards.
    for level in sorted(config.credible_levels, reverse=True):
        hdr, _ = _highest_density_region(spatial, level / 100.0)
        # A very peaked posterior can put the 50% and 95% regions in the same
        # cells, which would make the reported areas equal. Nesting must be
        # strict, so the inner level is trimmed to its densest cells.
        if previous_cells is not None and hdr.sum() >= previous_cells:
            keep = max(1, previous_cells - 1)
            order = np.argsort(spatial.ravel())[::-1][:keep]
            hdr = np.zeros(spatial.size, dtype=bool)
            hdr[order] = True
            hdr = hdr.reshape(spatial.shape)
        ring, n_parts = _region_polygon(hdr, lon_axis, lat_axis)
        regions[str(level)] = CredibleRegion(
            polygon_wgs84=ring,
            area_km2=float(np.broadcast_to(areas, spatial.shape)[hdr].sum()),
        )
        diag.n_region_components[str(level)] = n_parts
        previous_cells = int(hdr.sum())

    # Nesting is enforced by construction above, but assert it: the contract
    # rejects a 50% region that is not smaller than the 95% region, and a
    # violation here would mean the trimming logic is wrong.
    if regions["50"].area_km2 >= regions["95"].area_km2:
        raise ValueError("credible regions failed to nest; check the trimming logic")

    # --- release-time marginal ----------------------------------------------
    t_density = volume.sum(axis=(1, 2))
    t_bins = [datetime.fromtimestamp(float(t), tz=timezone.utc) for t in t_axis]
    order = np.argsort(t_density)[::-1]
    cum = np.cumsum(t_density[order])
    n_in = int(np.searchsorted(cum, 0.95) + 1)
    chosen = np.sort(order[:n_in])
    lo, hi = t_bins[int(chosen[0])], t_bins[int(chosen[-1])]
    if lo >= hi:
        hi = lo + timedelta(seconds=float(config.t0_bin_hours * 3600.0))

    assumptions = [
        f"windage alpha ~ U({transport.windage_alpha[0]}, {transport.windage_alpha[1]}), "
        "sampled per particle",
        f"horizontal diffusivity K_h ~ LogU({transport.diffusivity_kh[0]}, "
        f"{transport.diffusivity_kh[1]}) m^2/s, sampled per particle",
        f"forcing RMS velocity error {field.rms_error_ms} m/s (product QUID)",
        "Stokes drift taken from the merged current product; no separate "
        "parameterisation applied",
        f"detectability constant lambda = {config.detectability_lambda} "
        "(calibrated, not derived)",
        (
            f"likelihood tempered for spatial correlation: {n_eff_cells:.1f} effective "
            f"independent observations from {mask.n_mask_cells} mask cells, "
            f"decorrelation length {correlation_m / 1000:.1f} km"
            if n_eff_cells is not None
            else "likelihood NOT tempered: every mask cell treated as an independent "
                 "observation, which makes this posterior overconfident"
        ),
        f"hypothesis cell {grid.cell_m:.0f} m, release-time bin "
        f"{config.t0_bin_hours:.1f} h",
        "prototype transport kernel has no weathering module",
    ]
    if not diag.posterior_is_reliable:
        assumptions.append(
            f"WARNING: effective sample size is {diag.ess_fraction:.3%} of the "
            "ensemble; this posterior is Monte Carlo noise and must not be read "
            "as a confident localisation"
        )
    for level, parts in diag.n_region_components.items():
        if parts > 1:
            assumptions.append(
                f"the {level}% credible region has {parts} disconnected lobes; the "
                f"reported area covers all of them, the drawn polygon is the largest"
            )

    density_path = "out/posterior.npz"
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "posterior.npz",
            density=volume.astype("float32"),
            lon=lon_axis, lat=lat_axis, t0=t_axis,
        )

    posterior = SourcePosterior(
        case_id=case_id,
        detection_id=detection_id,
        grid=PosteriorGrid(
            lon=[float(v) for v in lon_axis],
            lat=[float(v) for v in lat_axis],
            t0=t_bins,
            density_path=density_path,
            normalised=True,
        ),
        credible_regions=regions,
        t0_marginal=TimeMarginal(
            bins_utc=t_bins,
            density=[float(v) for v in t_density],
            hpd_95=(lo, hi),
            width_hours=max((hi - lo).total_seconds() / 3600.0, 1e-3),
        ),
        lookback_hours=config.lookback_hours,
        within_operating_envelope=config.lookback_hours <= MAX_USEFUL_LOOKBACK_HOURS,
        n_hypotheses=grid.n,
        n_particles=total_particles,
        effective_sample_size=max(ess, 1e-6),
        assumptions=assumptions,
    )

    diag.elapsed_s = _time.perf_counter() - started
    return posterior, diag


__all__ = ["InversionConfig", "InversionDiagnostics", "invert"]
