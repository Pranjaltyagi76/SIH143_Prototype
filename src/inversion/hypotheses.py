"""Source hypotheses: the backward proposal pass and the space-time grid.

    Backward integration narrows the search. Forward simulation computes
    the answer.

That sentence is the intellectual core of the project, and this module is the
first half of it.

A blind prior over a 60 x 60 km box and 48 hours wastes almost every particle,
because the vast majority of (x0, t0) pairs cannot possibly produce the observed
slick. So we first run the mask **backwards with advection only** -- which is
legitimate, because pure advection is a deterministic ODE and is time-reversible
-- and keep only the space-time cells that pass through. Diffusion is never run
backwards; the kernel refuses.

The proposal pass uses ``TransportParams.proposal()``: diffusion off, and
windage **fixed at the mean rather than sampled**. That second part is not
cosmetic. Sampling windage across the two directions adds ~5 km of purely
artificial spread over 24 hours (P-14) -- displacement that looks like physical
uncertainty and is not. The proposal exists to narrow the search, so widening it
with a parameter we control would defeat the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from src.inversion.mask import ObservedMask
from src.transport import ForcingField, Seeds, TransportParams, simulate, to_epoch_seconds

# Particles used for the backward proposal. Cheap: this pass takes ~1 s.
PROPOSAL_PARTICLES = 3_000

# How far to dilate the proposal region before building hypotheses, metres.
# The backward pass uses mean windage and no diffusion, so the true source can
# legitimately sit outside the swept region. This margin is what stops the
# proposal from excluding the answer.
PROPOSAL_MARGIN_M = 15_000.0


@dataclass
class ProposalRegion:
    """Where the observed oil could plausibly have come from, per release time."""

    t0_bins: np.ndarray  # (T,) epoch seconds, candidate release times
    x: np.ndarray  # (T, P) swept positions
    y: np.ndarray
    valid: np.ndarray  # (T, P) bool -- particle still usable at that time

    @property
    def n_bins(self) -> int:
        return int(self.t0_bins.size)


def backward_proposal(
    field: ForcingField,
    mask: ObservedMask,
    t_obs: datetime,
    lookback_hours: float,
    t0_bin_hours: float = 1.0,
    n_particles: int = PROPOSAL_PARTICLES,
    seed: int = 0,
) -> ProposalRegion:
    """Sweep the observed slick backwards to find where the oil could have started."""
    rng = np.random.default_rng(seed)
    px, py = mask.sample_points(n_particles, rng)
    lon, lat = field.to_lonlat(px, py)

    params = TransportParams.proposal()
    stride = max(1, int(round(t0_bin_hours * 60.0 / params.timestep_minutes)))
    params = TransportParams.proposal(
        history_stride=stride, history_max_particles=n_particles
    )

    traj = simulate(
        field,
        Seeds.at_time(lon, lat, t_obs),
        t_obs,
        t_obs - timedelta(hours=lookback_hours),
        params,
        seed=seed,
        record_history=True,
    )

    # history_t runs backwards from t_obs; the earliest time is the deepest
    # lookback. Reverse so t0 bins ascend, which is how everything else reads.
    order = np.argsort(traj.history_t)
    hx = traj.history_x[order].astype("float64")
    hy = traj.history_y[order].astype("float64")
    t0_bins = traj.history_t[order]

    # A particle that beached or left the domain is frozen; its recorded
    # position stops being informative from that point backwards. Marking it
    # invalid keeps the proposal honest rather than piling mass on the boundary.
    valid = field.in_domain(hx, hy) & ~field.is_land(hx, hy)

    return ProposalRegion(t0_bins=t0_bins, x=hx, y=hy, valid=valid)


@dataclass
class HypothesisGrid:
    """A discrete set of candidate sources, each a (position, release-time) cell.

    Hypotheses live only where the backward proposal says oil could have come
    from, which is what makes the forward ensemble affordable.
    """

    x: np.ndarray  # (H,) cell-centre easting
    y: np.ndarray  # (H,) cell-centre northing
    t0: np.ndarray  # (H,) release time, epoch seconds
    cell_m: float
    t0_bin_s: float

    # Axes retained so the posterior can be assembled back onto a regular grid.
    x_axis: np.ndarray
    y_axis: np.ndarray
    t_axis: np.ndarray

    @property
    def n(self) -> int:
        return int(self.x.size)

    @classmethod
    def from_proposal(
        cls,
        proposal: ProposalRegion,
        cell_m: float,
        margin_m: float = PROPOSAL_MARGIN_M,
        max_hypotheses: int = 4096,
    ) -> HypothesisGrid:
        """Build hypotheses covering the swept region, dilated by a safety margin."""
        from scipy.ndimage import binary_dilation

        ok = proposal.valid
        if not ok.any():
            raise ValueError("backward proposal produced no valid positions")

        x0 = float(proposal.x[ok].min() - margin_m)
        x1 = float(proposal.x[ok].max() + margin_m)
        y0 = float(proposal.y[ok].min() - margin_m)
        y1 = float(proposal.y[ok].max() + margin_m)

        # Coarsen the cell size if the region is large enough that a fine grid
        # would blow the hypothesis budget. Reporting a coarser posterior is
        # honest; silently truncating the search region is not.
        n_bins = proposal.n_bins
        while True:
            nx = max(2, int((x1 - x0) / cell_m) + 1)
            ny = max(2, int((y1 - y0) / cell_m) + 1)
            if nx * ny * n_bins <= max_hypotheses * 6 or cell_m > 60_000.0:
                break
            cell_m *= 1.35

        x_axis = x0 + cell_m * np.arange(nx)
        y_axis = y0 + cell_m * np.arange(ny)
        pad = max(1, int(round(margin_m / cell_m)))

        xs, ys, ts = [], [], []
        for k in range(n_bins):
            sel = proposal.valid[k]
            if not sel.any():
                continue
            occupied = np.zeros((ny, nx), dtype=bool)
            ix = np.clip(((proposal.x[k][sel] - x0) / cell_m).astype(np.intp), 0, nx - 1)
            iy = np.clip(((proposal.y[k][sel] - y0) / cell_m).astype(np.intp), 0, ny - 1)
            occupied[iy, ix] = True
            occupied = binary_dilation(occupied, iterations=pad)

            cy, cx = np.nonzero(occupied)
            xs.append(x_axis[cx] + cell_m / 2.0)
            ys.append(y_axis[cy] + cell_m / 2.0)
            ts.append(np.full(cx.size, proposal.t0_bins[k]))

        if not xs:
            raise ValueError("no hypotheses generated from the proposal region")

        return cls(
            x=np.concatenate(xs),
            y=np.concatenate(ys),
            t0=np.concatenate(ts),
            cell_m=cell_m,
            t0_bin_s=float(np.diff(proposal.t0_bins).mean()) if n_bins > 1 else 3600.0,
            x_axis=x_axis,
            y_axis=y_axis,
            t_axis=proposal.t0_bins,
        )

    def seeds(
        self, field: ForcingField, particles_per_hypothesis: int, rng: np.random.Generator
    ) -> Seeds:
        """Expand to particles, jittered within each cell and time bin.

        Jitter matters: seeding every particle at the exact cell centre would
        make the predicted plume artificially compact and would flatter the
        likelihood of whichever cell happened to line up.
        """
        m = particles_per_hypothesis
        marker = np.repeat(np.arange(self.n), m)
        jx = np.repeat(self.x, m) + (rng.random(self.n * m) - 0.5) * self.cell_m
        jy = np.repeat(self.y, m) + (rng.random(self.n * m) - 0.5) * self.cell_m
        jt = np.repeat(self.t0, m) + (rng.random(self.n * m) - 0.5) * self.t0_bin_s

        lon, lat = field.to_lonlat(jx, jy)
        return Seeds(lon=lon, lat=lat, seed_time=jt, origin_marker=marker)

    def refine(
        self, log_likelihood: np.ndarray, keep_fraction: float, split: int = 2
    ) -> HypothesisGrid:
        """Subdivide the best-scoring hypotheses for a second round.

        Two coarse-to-fine rounds cost less than one fine blind pass and put the
        particles where the posterior actually is.
        """
        finite = np.isfinite(log_likelihood)
        if not finite.any():
            raise ValueError("every hypothesis scored -inf; nothing to refine")

        n_keep = max(4, int(round(finite.sum() * keep_fraction)))
        order = np.argsort(log_likelihood)[::-1]
        keep = order[:n_keep]

        new_cell = self.cell_m / split
        offsets = (np.arange(split) - (split - 1) / 2.0) * new_cell
        ox, oy = np.meshgrid(offsets, offsets)
        ox, oy = ox.ravel(), oy.ravel()

        xs = (self.x[keep][:, None] + ox[None, :]).ravel()
        ys = (self.y[keep][:, None] + oy[None, :]).ravel()
        ts = np.repeat(self.t0[keep], ox.size)

        return HypothesisGrid(
            x=xs, y=ys, t0=ts,
            cell_m=new_cell,
            t0_bin_s=self.t0_bin_s,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            t_axis=self.t_axis,
        )


def to_datetime(epoch_s: float) -> datetime:
    from datetime import timezone

    return datetime.fromtimestamp(float(epoch_s), tz=timezone.utc)


__all__ = [
    "ProposalRegion",
    "backward_proposal",
    "HypothesisGrid",
    "to_datetime",
    "PROPOSAL_PARTICLES",
    "PROPOSAL_MARGIN_M",
]
