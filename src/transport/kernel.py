"""Lagrangian transport kernel.

One implementation, used by the backward proposal, the forward inversion
ensemble, the forecast, and the vessel-conditioned attribution run. Everything
downstream depends on this being right.

    dx = [ u_current(x,t) + alpha * u_wind10(x,t) ] dt  +  sqrt(2 * K_h * dt) * xi

- ``u_current`` already contains tides and Stokes drift when the source product
  merges them, which CMEMS SMOC does. **No separate Stokes term is ever added**;
  the field carries an ``includes_stokes`` flag and the kernel honours it. Adding
  it twice would bias every trajectory downwind, silently, with no error raised.
- ``alpha`` (windage) and ``K_h`` (diffusivity) are **sampled per particle**, not
  fixed. Treating either as a constant materially understates position
  uncertainty, which is the one thing this project must not do.

The non-negotiable rule
-----------------------
**The kernel refuses to run backwards with diffusion enabled.**

Advection is a deterministic ODE and is time-reversible. Turbulent diffusion is
entropy-increasing and is not: integrating it backwards is ill-posed, the same
error as un-stirring milk out of coffee. Run backwards with diffusion on and you
get a spreading cloud that *looks* like an uncertainty envelope but is a forward
diffusion process pointed backwards in time. It is not a posterior.

Backward integration narrows the search. Forward simulation computes the answer.

That rule is enforced here as an exception rather than documented as a
convention, because it is the claim the whole project rests on.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field as _field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.transport.field import ForcingField, to_epoch_seconds

# Particle status codes.
PENDING = 0  # seeded, but its release time has not been reached yet
ACTIVE = 1
BEACHED = 2  # reached land: frozen, never deleted (W-03)
EXITED = 3  # left the forcing domain: frozen, and counted


@dataclass(frozen=True)
class TransportParams:
    """Integration and stochastic parameters.

    Defaults mirror ``configs/physics.yaml``; use ``from_yaml`` to load that
    file so there is a single source of truth for the physical constants.
    """

    timestep_minutes: float = 15.0

    # Windage: fraction of 10 m wind speed imparted to surface oil.
    # Literature range roughly 1-4%. Sampled per particle.
    windage_alpha: tuple[float, float] = (0.01, 0.04)

    # Horizontal diffusivity, m^2/s. Order-of-magnitude uncertain, so
    # log-uniform. Sampled per particle.
    diffusivity_kh: tuple[float, float] = (1.0, 10.0)

    diffusion_enabled: bool = True
    windage_enabled: bool = True

    # Trajectory recording. Storing every step for 1e5 particles is ~150 MB;
    # the UI needs a few thousand trails, so subsample by default.
    history_stride: int = 4
    history_max_particles: int = 5_000

    @classmethod
    def from_yaml(cls, path: Path | str = "configs/physics.yaml", **overrides) -> TransportParams:
        import yaml

        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["transport"]
        base = dict(
            timestep_minutes=float(cfg["timestep_minutes"]),
            windage_alpha=(float(cfg["windage_alpha"]["low"]), float(cfg["windage_alpha"]["high"])),
            diffusivity_kh=(
                float(cfg["diffusivity_kh"]["low"]),
                float(cfg["diffusivity_kh"]["high"]),
            ),
        )
        return cls(**{**base, **overrides})

    @classmethod
    def proposal(cls, **overrides) -> TransportParams:
        """Configuration for the backward proposal pass.

        Diffusion off, because backward diffusion is ill-posed and the kernel
        refuses it. **Windage fixed at the mean, not sampled** -- and that second
        part is not cosmetic.

        Windage spans 1-4% of a ~6 m/s wind, so two independent draws differ by
        roughly 0.09 m/s. Over 24 hours that is ~7.8 km of purely artificial
        displacement. Measured on the development case, a forward-then-backward
        round trip returns to within 0.0 m with windage off and 13 m with
        windage fixed, but **5.1 km median with windage sampled** -- spread that
        looks like physical uncertainty and is not.

        The proposal pass exists to narrow the search region. Sampling windage
        there would widen it by kilometres for no reason, so this factory exists
        to make the correct configuration the easy one.
        """
        mid = sum(cls.__dataclass_fields__["windage_alpha"].default) / 2.0
        base = dict(diffusion_enabled=False, windage_alpha=(mid, mid))
        return cls(**{**base, **overrides})

    @property
    def timestep_seconds(self) -> float:
        return self.timestep_minutes * 60.0

    @property
    def windage_is_fixed(self) -> bool:
        return self.windage_alpha[0] == self.windage_alpha[1]


@dataclass
class Seeds:
    """Particle release specification.

    ``seed_time`` is per element, which is what makes vessel-conditioned
    attribution possible: particles are released along a vessel's actual AIS
    track, at the times it was actually there, in a single simulation.

    ``origin_marker`` labels each particle with what released it -- a source
    hypothesis during inversion, a candidate vessel during attribution. The
    whole method depends on being able to ask, at the end, where each parcel
    came from.
    """

    lon: np.ndarray
    lat: np.ndarray
    seed_time: np.ndarray  # epoch seconds, per particle
    origin_marker: np.ndarray  # int label, per particle

    def __post_init__(self) -> None:
        self.lon = np.asarray(self.lon, dtype="float64").ravel()
        self.lat = np.asarray(self.lat, dtype="float64").ravel()
        self.seed_time = np.asarray(self.seed_time, dtype="float64").ravel()
        self.origin_marker = np.asarray(self.origin_marker, dtype="int64").ravel()
        n = self.lon.size
        if not (self.lat.size == self.seed_time.size == self.origin_marker.size == n):
            raise ValueError("Seeds arrays must all be the same length")
        if n == 0:
            raise ValueError("Seeds must contain at least one particle")

    def __len__(self) -> int:
        return int(self.lon.size)

    @classmethod
    def at_time(
        cls,
        lon: np.ndarray,
        lat: np.ndarray,
        when: datetime,
        origin_marker: int | np.ndarray = 0,
    ) -> Seeds:
        """All particles released at the same instant."""
        lon = np.asarray(lon, dtype="float64").ravel()
        t = np.full(lon.size, to_epoch_seconds(when), dtype="float64")
        marker = (
            np.full(lon.size, origin_marker, dtype="int64")
            if np.isscalar(origin_marker)
            else np.asarray(origin_marker, dtype="int64").ravel()
        )
        return cls(lon=lon, lat=lat, seed_time=t, origin_marker=marker)


@dataclass
class Trajectory:
    """Result of one simulation."""

    x: np.ndarray  # final position, projected metres
    y: np.ndarray
    lon: np.ndarray  # final position, degrees
    lat: np.ndarray
    status: np.ndarray  # PENDING / ACTIVE / BEACHED / EXITED
    origin_marker: np.ndarray
    alpha: np.ndarray  # windage sampled for each particle
    kh: np.ndarray  # diffusivity sampled for each particle

    crs: str
    t_start: float
    t_end: float
    n_steps: int
    elapsed_s: float

    history_t: np.ndarray | None = None  # (n_frames,)
    history_x: np.ndarray | None = None  # (n_frames, n_tracked)
    history_y: np.ndarray | None = None
    history_index: np.ndarray | None = None  # which particles were tracked

    counts: dict[str, int] = _field(default_factory=dict)

    @property
    def n_particles(self) -> int:
        return int(self.x.size)

    @property
    def usable(self) -> np.ndarray:
        """Particles that were released and did not leave the domain.

        Beached particles remain usable: they are frozen at the coast, and
        discarding them would bias the posterior away from the shoreline (W-03).
        """
        return (self.status == ACTIVE) | (self.status == BEACHED)

    @property
    def is_backward(self) -> bool:
        return self.t_end < self.t_start


def simulate(
    field: ForcingField,
    seeds: Seeds,
    t_start: datetime,
    t_end: datetime,
    params: TransportParams | None = None,
    seed: int = 0,
    record_history: bool = False,
) -> Trajectory:
    """Integrate particles from ``t_start`` to ``t_end``.

    Backward simply means ``t_end < t_start``. Raises if diffusion is enabled in
    that case -- see the module docstring.
    """
    params = params or TransportParams()
    ts = to_epoch_seconds(t_start)
    te = to_epoch_seconds(t_end)

    if ts == te:
        raise ValueError("t_start and t_end are identical; nothing to integrate")

    backward = te < ts

    if backward and params.diffusion_enabled:
        raise ValueError(
            "Refusing to integrate backwards with diffusion enabled.\n"
            "Advection is time-reversible; turbulent diffusion is not -- running it "
            "backwards is ill-posed, the same error as un-stirring milk out of coffee. "
            "The resulting cloud looks like an uncertainty envelope but is a forward "
            "diffusion process pointed backwards in time, and is not a posterior.\n"
            "Use TransportParams(diffusion_enabled=False) for the backward proposal "
            "pass, then run the ensemble FORWARD and condition on the observation. "
            "Backward integration narrows the search; forward simulation computes "
            "the answer."
        )

    if not (field.covers(min(ts, te)) and field.covers(max(ts, te))):
        raise ValueError(
            f"requested window [{min(ts, te)}, {max(ts, te)}] is outside the forcing "
            f"range [{field.t_min}, {field.t_max}]"
        )

    rng = np.random.default_rng(seed)
    n = len(seeds)

    x, y = field.to_grid(seeds.lon, seeds.lat)
    x = x.astype("float64")
    y = y.astype("float64")

    alpha = (
        rng.uniform(*params.windage_alpha, size=n)
        if params.windage_enabled
        else np.zeros(n, dtype="float64")
    )
    lo, hi = params.diffusivity_kh
    kh = (
        np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
        if params.diffusion_enabled
        else np.zeros(n, dtype="float64")
    )

    status = np.full(n, PENDING, dtype="int8")
    # Particles whose release time is already reached at t_start start active.
    status[(seeds.seed_time <= ts) if not backward else (seeds.seed_time >= ts)] = ACTIVE

    dt = params.timestep_seconds * (-1.0 if backward else 1.0)
    n_steps = int(np.ceil(abs(te - ts) / params.timestep_seconds))

    # Trajectory recording, on a deterministic subsample.
    track_idx = hist_t = hist_x = hist_y = None
    if record_history:
        k = min(params.history_max_particles, n)
        track_idx = np.linspace(0, n - 1, k).astype(np.intp)
        n_frames = n_steps // params.history_stride + 1
        hist_t = np.empty(n_frames, dtype="float64")
        hist_x = np.empty((n_frames, k), dtype="float32")
        hist_y = np.empty((n_frames, k), dtype="float32")

    def total_velocity(px, py, t, wind_u, wind_v, a):
        cu, cv = field.current(px, py, t)
        return cu + a * wind_u, cv + a * wind_v

    started = _time.perf_counter()
    frame = 0

    for step in range(n_steps):
        t = ts + step * dt
        t_next = t + dt
        # Do not overshoot the requested end time on the final step.
        if (not backward and t_next > te) or (backward and t_next < te):
            t_next = te
        h = t_next - t
        if h == 0.0:
            break
        t_mid = t + 0.5 * h

        if record_history and step % params.history_stride == 0:
            hist_t[frame] = t
            hist_x[frame] = x[track_idx]
            hist_y[frame] = y[track_idx]
            frame += 1

        # Release particles whose seed time falls in this step.
        pending = status == PENDING
        if pending.any():
            reached = (
                (seeds.seed_time <= t_next) if not backward else (seeds.seed_time >= t_next)
            )
            status[pending & reached] = ACTIVE

        act = status == ACTIVE
        if not act.any():
            continue

        ax, ay, aa, akh = x[act], y[act], alpha[act], kh[act]
        prev_x, prev_y = ax.copy(), ay.copy()

        # Wind is evaluated ONCE per step, at the midpoint, rather than at every
        # RK4 stage. The wind field varies hourly over ~25 km cells while a
        # particle moves under a kilometre per step, so sub-step wind
        # interpolation is arithmetic without information. Saves 3 of 4 lookups.
        wu, wv = field.wind(ax, ay, t_mid) if params.windage_enabled else (0.0, 0.0)

        # --- RK4 on the deterministic part -------------------------------
        k1u, k1v = total_velocity(ax, ay, t, wu, wv, aa)
        k2u, k2v = total_velocity(ax + 0.5 * h * k1u, ay + 0.5 * h * k1v, t_mid, wu, wv, aa)
        k3u, k3v = total_velocity(ax + 0.5 * h * k2u, ay + 0.5 * h * k2v, t_mid, wu, wv, aa)
        k4u, k4v = total_velocity(ax + h * k3u, ay + h * k3v, t_next, wu, wv, aa)

        ax = ax + (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        ay = ay + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        # --- Euler-Maruyama on the stochastic part ------------------------
        # Operator splitting: RK4 for advection, Euler-Maruyama for diffusion.
        # abs(h) because the random-walk magnitude does not depend on the sign
        # of time -- though backward with diffusion is refused above anyway.
        if params.diffusion_enabled:
            sigma = np.sqrt(2.0 * akh * abs(h))
            ax = ax + sigma * rng.standard_normal(ax.size)
            ay = ay + sigma * rng.standard_normal(ay.size)

        # --- boundaries ----------------------------------------------------
        # Beached and exited particles are FROZEN at their last valid position,
        # never deleted. Deleting them silently biases the posterior away from
        # the coast, which is exactly where spills matter most (W-03).
        new_status = np.full(ax.size, ACTIVE, dtype="int8")
        outside = ~field.in_domain(ax, ay)
        beached = field.is_land(ax, ay) & ~outside

        ax = np.where(beached | outside, prev_x, ax)
        ay = np.where(beached | outside, prev_y, ay)
        new_status[beached] = BEACHED
        new_status[outside] = EXITED

        x[act], y[act] = ax, ay
        idx = np.flatnonzero(act)
        status[idx] = new_status

    if record_history and frame < len(hist_t):
        hist_t[frame] = ts + n_steps * dt
        hist_x[frame] = x[track_idx]
        hist_y[frame] = y[track_idx]
        frame += 1
        hist_t, hist_x, hist_y = hist_t[:frame], hist_x[:frame], hist_y[:frame]

    lon, lat = field.to_lonlat(x, y)

    return Trajectory(
        x=x,
        y=y,
        lon=lon,
        lat=lat,
        status=status,
        origin_marker=seeds.origin_marker,
        alpha=alpha,
        kh=kh,
        crs=field.crs,
        t_start=ts,
        t_end=te,
        n_steps=n_steps,
        elapsed_s=_time.perf_counter() - started,
        history_t=hist_t,
        history_x=hist_x,
        history_y=hist_y,
        history_index=track_idx,
        counts={
            "total": n,
            "active": int((status == ACTIVE).sum()),
            "beached": int((status == BEACHED).sum()),
            "exited": int((status == EXITED).sum()),
            "never_released": int((status == PENDING).sum()),
        },
    )


__all__ = [
    "TransportParams",
    "Seeds",
    "Trajectory",
    "simulate",
    "PENDING",
    "ACTIVE",
    "BEACHED",
    "EXITED",
]
