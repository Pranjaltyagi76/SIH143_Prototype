"""Attribution: each vessel's AIS track as a generative hypothesis.

The obvious approach -- ``score = w1*proximity + w2*gap + w3*anomaly`` -- dies on
one question: *where did w1 come from?* There is no answer. No labelled
spill-to-vessel dataset exists, so any weights not learned from labels are
invented.

A better approach integrates the source posterior along each vessel's path.
That has no free weights, which is a real advance, but it is still a **proxy**:
``p(x0, t0 | obs)`` is a marginal over point sources, while a real discharge from
a moving vessel is a *line source in space-time*. Evaluating a point-source
marginal along a line is not ``p(obs | vessel v)``.

What we do instead
------------------
A vessel's AIS track **is** the parameterisation of that line source. We already
have it, so we do not need to invert for it -- we can simulate it directly.

For each candidate vessel and each candidate release window, seed particles
along the track it actually sailed, at the times it was actually there, run them
forward, and score the result with the **same Bernoulli observation operator**
the inversion uses. Every candidate, every window, and the dark-vessel
hypothesis all go into a single forward simulation, distinguished by
``origin_marker``.

The result is a genuine likelihood, comparable across candidates on one scale,
with no free parameters.

The dark-vessel hypothesis
--------------------------
                       L(v) * pi(v)
    P(v | obs) = -------------------------------------
                 SUM_u L(u) pi(u)  +  L_dark * pi_dark

If the true polluter had AIS switched off it is **not in the candidate set at
all**, and a system that normalises only over observed vessels will confidently
name an innocent ship. So we carry an explicit hypothesis for it, computed with
the identical operator over source points that no observed track passes near.

The system can therefore output *"most probable explanation: a vessel not
transmitting AIS"*, which is a correct and operationally valuable answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.attribution.ais import VesselTrack, baseline_gap_hours
from src.attribution.priors import PriorConfig, behavioural_prior
from src.contracts import (
    Candidate,
    DarkVesselHypothesis,
    RankedCandidates,
    SourcePosterior,
    TrafficReduction,
    VesselEvidence,
)
from src.inversion import ObservedMask
from src.inversion.likelihood import DEFAULT_LAMBDA, effective_cell_count, evaluate
from src.transport import ForcingField, Seeds, TransportParams, simulate


@dataclass(frozen=True)
class AttributionConfig:
    """Knobs for attribution. Defaults mirror ``configs/physics.yaml``."""

    prefilter_spatial_margin_km: float = 25.0
    prefilter_temporal_margin_hours: float = 6.0

    # Prior probability that the source vessel was not transmitting AIS. An
    # ASSUMPTION, not a measurement -- report sensitivity across the range
    # rather than a single value (debt item D5).
    dark_vessel_prior: float = 0.15

    # A candidate source point closer than this to an observed track at the
    # same time is explained by that vessel, so it does not belong to the dark
    # hypothesis.
    dark_exclusion_radius_km: float = 3.0
    dark_samples: int = 320

    particles_per_hypothesis: int = 60
    release_window_hours: float = 2.0

    # Candidates below this probability are still returned -- dropping them
    # would break normalisation and inflate the leaders -- but they are not
    # counted in the headline reduction figure.
    reporting_threshold: float = 0.02

    detectability_lambda: float = DEFAULT_LAMBDA
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: Path | str = "configs/physics.yaml", **overrides):
        import yaml

        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["attribution"]
        base = dict(
            prefilter_spatial_margin_km=float(cfg["prefilter_spatial_margin_km"]),
            prefilter_temporal_margin_hours=float(cfg["prefilter_temporal_margin_hours"]),
            dark_vessel_prior=float(cfg["dark_vessel_prior"]),
        )
        return cls(**{**base, **overrides})


@dataclass
class AttributionDiagnostics:
    vessels_in_window: int = 0
    after_prefilter: int = 0
    n_hypotheses: int = 0
    n_particles: int = 0
    n_dark_samples: int = 0
    baseline_gap_hours: float = 0.0
    slick_axis_deg: float = 0.0
    elapsed_s: float = 0.0
    cleaning: dict = _field(default_factory=dict)


# ------------------------------------------------------------------ prefilter


def prefilter(
    tracks: list[VesselTrack],
    posterior: SourcePosterior,
    config: AttributionConfig,
) -> list[VesselTrack]:
    """Drop traffic that cannot have produced the slick.

    A vessel survives if any of its fixes falls inside the 95% credible region,
    buffered by a spatial margin, during the release-time window, extended by a
    temporal margin. The margins matter: the point of filtering is to remove
    irrelevant traffic, not to remove the answer.
    """
    from shapely.geometry import Point
    from shapely.geometry import Polygon as SPoly
    from shapely.prepared import prep

    region = SPoly(posterior.credible_regions["95"].polygon_wgs84)
    # Degrees are only ever used here, for a coarse containment test with a
    # generous margin -- never for distance or area (W-11).
    mid_lat = float(np.mean([p[1] for p in posterior.credible_regions["95"].polygon_wgs84]))
    buffer_deg = config.prefilter_spatial_margin_km / (111.32 * max(np.cos(np.deg2rad(mid_lat)), 0.2))
    region = prep(region.buffer(buffer_deg))

    margin = config.prefilter_temporal_margin_hours * 3600.0
    t_lo = posterior.t0_marginal.hpd_95[0].timestamp() - margin
    t_hi = posterior.t0_marginal.hpd_95[1].timestamp() + margin

    survivors = []
    for track in tracks:
        in_time = (track.time >= t_lo) & (track.time <= t_hi)
        if not in_time.any():
            continue
        if any(
            region.contains(Point(lo, la))
            for lo, la in zip(track.lon[in_time], track.lat[in_time])
        ):
            survivors.append(track)
    return survivors


# ------------------------------------------------------------------ the engine


def _release_bins(posterior: SourcePosterior, config: AttributionConfig) -> np.ndarray:
    """Candidate release-window start times, spanning the t0 marginal support."""
    lo = posterior.t0_marginal.hpd_95[0].timestamp()
    hi = posterior.t0_marginal.hpd_95[1].timestamp()
    step = config.release_window_hours * 3600.0
    n = max(1, int(np.ceil((hi - lo) / step)))
    return lo + step * np.arange(n)


def attribute(
    field: ForcingField,
    mask: ObservedMask,
    posterior: SourcePosterior,
    tracks: list[VesselTrack],
    t_obs: datetime,
    case_id: str,
    config: AttributionConfig | None = None,
    transport: TransportParams | None = None,
    cleaning: dict | None = None,
    vessels_in_window: int | None = None,
) -> tuple[RankedCandidates, AttributionDiagnostics]:
    """Rank candidate vessels against the observed slick."""
    import time as _time

    config = config or AttributionConfig()
    transport = transport or TransportParams()
    rng = np.random.default_rng(config.seed)
    started = _time.perf_counter()

    diag = AttributionDiagnostics(
        vessels_in_window=vessels_in_window if vessels_in_window is not None else len(tracks),
        cleaning=cleaning or {},
        baseline_gap_hours=baseline_gap_hours(tracks),
        slick_axis_deg=mask.major_axis_deg,
    )

    candidates = prefilter(tracks, posterior, config)
    diag.after_prefilter = len(candidates)

    bins = _release_bins(posterior, config)
    window_s = config.release_window_hours * 3600.0
    n_bins = bins.size
    m = config.particles_per_hypothesis

    t_begin = min(bins.min(), posterior.t0_marginal.hpd_95[0].timestamp())
    t_begin = max(t_begin, field.t_min + 60.0)
    t_begin_dt = datetime.fromtimestamp(t_begin, tz=timezone.utc)

    # --- vessel hypotheses: (vessel, release window) -------------------------
    lon_parts, lat_parts, time_parts, marker_parts = [], [], [], []
    for vi, track in enumerate(candidates):
        for bi, b0 in enumerate(bins):
            times = b0 + rng.random(m) * window_s
            lon, lat, _ = track.interpolate(times)
            lon_parts.append(lon)
            lat_parts.append(lat)
            time_parts.append(times)
            marker_parts.append(np.full(m, vi * n_bins + bi, dtype="int64"))

    n_vessel_hyp = len(candidates) * n_bins

    # --- the dark-vessel hypothesis ------------------------------------------
    # Source points inside the plausible region that no observed track passes
    # near. Scored by the identical operator, so it competes on one scale.
    dark_lon, dark_lat, dark_time = _sample_dark_sources(
        posterior, candidates, bins, window_s, config, rng
    )
    n_dark = dark_lon.size
    diag.n_dark_samples = n_dark
    if n_dark:
        jitter_t = rng.random(n_dark * m) * window_s
        lon_parts.append(np.repeat(dark_lon, m))
        lat_parts.append(np.repeat(dark_lat, m))
        time_parts.append(np.repeat(dark_time, m) + jitter_t)
        marker_parts.append(
            n_vessel_hyp + np.repeat(np.arange(n_dark), m).astype("int64")
        )

    if not lon_parts:
        raise ValueError("no candidate vessels and no dark samples; nothing to attribute")

    seeds = Seeds(
        lon=np.concatenate(lon_parts),
        lat=np.concatenate(lat_parts),
        seed_time=np.clip(np.concatenate(time_parts), field.t_min + 1.0, t_obs.timestamp() - 1.0),
        origin_marker=np.concatenate(marker_parts),
    )
    n_hyp = n_vessel_hyp + n_dark
    diag.n_hypotheses = n_hyp
    diag.n_particles = len(seeds)

    # ONE forward run covers every candidate, every release window, and the
    # dark hypothesis.
    traj = simulate(field, seeds, t_begin_dt, t_obs, transport, seed=config.seed)
    result = evaluate(
        traj, mask, n_hyp, config.detectability_lambda,
        n_effective_cells=effective_cell_count(mask, field.dx * 4.5),
    )

    log_lik = result.log_likelihood
    vessel_ll = log_lik[:n_vessel_hyp].reshape(len(candidates), n_bins) if candidates else np.empty((0, n_bins))
    dark_ll = log_lik[n_vessel_hyp:] if n_dark else np.array([-np.inf])

    # Marginalise over release windows with a uniform prior; report the
    # best-fitting window as evidence.
    marginal = _logsumexp(vessel_ll, axis=1) - np.log(n_bins) if candidates else np.array([])
    best_bin = np.argmax(vessel_ll, axis=1) if candidates else np.array([], dtype=int)
    dark_marginal = _logsumexp(dark_ll[None, :], axis=1)[0] - np.log(max(n_dark, 1))

    # --- behavioural priors ---------------------------------------------------
    prior_cfg = PriorConfig(
        baseline_gap_hours=diag.baseline_gap_hours, slick_axis_deg=diag.slick_axis_deg
    )
    priors, evidences, windows = [], [], []
    for vi, track in enumerate(candidates):
        b0 = float(bins[best_bin[vi]])
        b1 = b0 + window_s
        pi, factors = behavioural_prior(track, b0, b1, prior_cfg)
        priors.append(pi)
        evidences.append(factors)
        windows.append((b0, b1))
    priors = np.asarray(priors, dtype="float64")

    # --- combine -------------------------------------------------------------
    # Prior mass splits between "some AIS-observed vessel" and "a vessel not
    # transmitting", then divides among candidates by behavioural prior.
    pd = float(np.clip(config.dark_vessel_prior, 1e-6, 1.0 - 1e-6))
    all_ll = np.concatenate([marginal, [dark_marginal]])
    finite = all_ll[np.isfinite(all_ll)]
    shift = finite.max() if finite.size else 0.0

    if candidates and priors.sum() > 0:
        vessel_w = (1.0 - pd) * (priors / priors.sum()) * np.exp(marginal - shift)
    else:
        vessel_w = np.zeros(len(candidates))
    dark_w = pd * np.exp(dark_marginal - shift)

    total = vessel_w.sum() + dark_w
    if not np.isfinite(total) or total <= 0:
        vessel_w = np.zeros(len(candidates))
        dark_w = 1.0
        total = 1.0
    probs = vessel_w / total
    dark_prob = float(dark_w / total)

    # --- assemble the contract ------------------------------------------------
    order = np.argsort(probs)[::-1]
    out: list[Candidate] = []
    for rank, vi in enumerate(order, start=1):
        track = candidates[vi]
        b0, b1 = windows[vi]
        out.append(
            Candidate(
                rank=rank,
                mmsi=track.mmsi,
                vessel_name=track.name or track.mmsi,
                vessel_type=track.ship_type or "Unknown",
                posterior_probability=float(probs[vi]),
                log_likelihood=float(marginal[vi]) if np.isfinite(marginal[vi]) else -1e9,
                best_discharge_window_utc=(
                    datetime.fromtimestamp(b0, tz=timezone.utc),
                    datetime.fromtimestamp(b1, tz=timezone.utc),
                ),
                evidence=VesselEvidence(
                    track_overlap_likelihood=float(marginal[vi])
                    if np.isfinite(marginal[vi])
                    else -1e9,
                    **evidences[vi],
                ),
                track_geojson=_track_geojson(track, b0 - 6 * 3600.0, b1 + 6 * 3600.0),
            )
        )

    # Renormalise for float drift; the contract requires an exact sum.
    _rebalance(out, dark_prob)

    n_reported = int(sum(c.posterior_probability >= config.reporting_threshold for c in out))
    ranked = RankedCandidates(
        case_id=case_id,
        traffic_reduction=TrafficReduction(
            vessels_in_window=max(diag.vessels_in_window, len(candidates)),
            after_prefilter=len(candidates),
            reported=min(n_reported, len(candidates)),
        ),
        candidates=out,
        dark_vessel_hypothesis=DarkVesselHypothesis(
            posterior_probability=dark_prob,
            prior_used=pd,
            note=(
                f"Posterior mass not explained by any AIS-observed track, evaluated "
                f"over {n_dark} source points that no candidate passed within "
                f"{config.dark_exclusion_radius_km:.0f} km of. This hypothesis "
                f"competes on equal terms with the named candidates."
            ),
        ),
    )
    diag.elapsed_s = _time.perf_counter() - started
    return ranked, diag


# ------------------------------------------------------------------ helpers


def _sample_dark_sources(posterior, candidates, bins, window_s, config, rng):
    """Source points inside the plausible region that no observed track passes near."""
    from shapely.geometry import Point
    from shapely.geometry import Polygon as SPoly
    from shapely.prepared import prep

    region = SPoly(posterior.credible_regions["95"].polygon_wgs84)
    prepared = prep(region)
    minx, miny, maxx, maxy = region.bounds
    mid_lat = 0.5 * (miny + maxy)
    excl_deg = config.dark_exclusion_radius_km / (
        111.32 * max(np.cos(np.deg2rad(mid_lat)), 0.2)
    )

    lons, lats, times = [], [], []
    attempts = 0
    while len(lons) < config.dark_samples and attempts < config.dark_samples * 40:
        attempts += 1
        lo = rng.uniform(minx, maxx)
        la = rng.uniform(miny, maxy)
        if not prepared.contains(Point(lo, la)):
            continue
        t = float(bins[rng.integers(bins.size)] + rng.random() * window_s)

        explained = False
        for track in candidates:
            if not track.covers(t):
                continue
            vlon, vlat, _ = track.interpolate(np.array([t]))
            if abs(vlon[0] - lo) < excl_deg and abs(vlat[0] - la) < excl_deg:
                explained = True
                break
        if explained:
            continue

        lons.append(lo)
        lats.append(la)
        times.append(t)

    return (
        np.asarray(lons, dtype="float64"),
        np.asarray(lats, dtype="float64"),
        np.asarray(times, dtype="float64"),
    )


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    finite = np.where(np.isfinite(a), a, -np.inf)
    peak = np.max(finite, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    out = peak.squeeze(axis) + np.log(np.sum(np.exp(finite - peak), axis=axis))
    return np.where(np.isfinite(out), out, -np.inf)


def _rebalance(candidates: list[Candidate], dark_prob: float) -> None:
    """Force an exact sum of 1 across candidates plus the dark hypothesis."""
    total = sum(c.posterior_probability for c in candidates) + dark_prob
    if total <= 0:
        return
    drift = 1.0 - total
    if abs(drift) < 1e-12:
        return
    if candidates:
        top = max(candidates, key=lambda c: c.posterior_probability)
        top.posterior_probability = float(np.clip(top.posterior_probability + drift, 0.0, 1.0))


def _track_geojson(track: VesselTrack, t_from: float, t_to: float) -> dict:
    sel = (track.time >= t_from) & (track.time <= t_to)
    if not sel.any():
        sel = np.ones(track.time.size, dtype=bool)
    return {
        "type": "Feature",
        "properties": {"mmsi": track.mmsi},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [round(float(lo), 5), round(float(la), 5)]
                for lo, la in zip(track.lon[sel], track.lat[sel])
            ],
        },
    }


__all__ = ["AttributionConfig", "AttributionDiagnostics", "attribute", "prefilter"]
