"""The observation operator: p(observed slick | source hypothesis).

This is where a Bayesian method is actually right or wrong, and it is the part
that is easiest to get subtly wrong.

What we do NOT do
-----------------
The tempting implementation is per-particle rejection: *accept particle i if it
lands inside the observed slick; the origins of accepted particles are the
posterior.* That computes ``p(x0, t0 | one oil parcel ended up somewhere in the
slick)``, which is not ``p(source | observed slick shape)``. It has two defects:

1. **False coverage is never penalised.** A hypothesis whose particles smear
   across the whole scene earns the same per-particle credit as one producing a
   compact cloud that matches the slick precisely, because only hits are counted
   and misses cost nothing.
2. **Shape and extent are discarded** -- the very quantities characterisation
   just computed.

What we do instead
------------------
A hypothesis-level Bernoulli likelihood over the whole area. Rasterise each
hypothesis's particle cloud to a predicted oil-presence probability

    q_h(x) = 1 - exp(-lambda * rho_h(x))

and score it against the observed binary mask m(x):

    log L(h) = SUM_x [ m(x) log q_h(x) + (1 - m(x)) log(1 - q_h(x)) ]

Predicted oil where the satellite saw none now costs you, which is exactly the
property the naive version lacks.

The closed form
---------------
Evaluated literally over every cell for every hypothesis this is ~1e9
operations. It does not need to be. Since ``log(1 - q_h) = -lambda * rho_h``,
the negative term collapses to a count:

    SUM over x NOT in mask [ log(1 - q_h) ] = -lambda * (mass landing outside)

so the whole thing becomes

    log L(h) = SUM over MASK cells [ log(1 - exp(-lambda rho_h)) ]
               - lambda * (mass outside the mask)

which costs O(mask cells + particles) instead of O(all cells). Roughly 2 s
instead of 40 s, with no approximation -- it is algebraically identical.

It also makes the physics legible in one line: **reward for covering the
observed slick, minus a linear penalty for every particle predicted where the
satellite saw nothing.**

Normalisation of rho
--------------------
``rho_h`` is scaled so a hypothesis whose plume exactly covers the observed
slick has rho = 1 in each mask cell. Then ``lambda`` is a genuine dimensionless
detectability constant rather than a number that has to be retuned whenever the
slick changes size:

    rho_h(x) = (particles of h in cell x / usable particles of h) * n_mask_cells
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inversion.mask import ObservedMask
from src.transport import Trajectory

# Dimensionless detectability constant. lambda = 2.5 means a hypothesis whose
# plume exactly reproduces the observed slick reaches q = 1 - exp(-2.5) = 0.92
# oil-presence probability in each mask cell. An ASSUMPTION, calibrated rather
# than derived -- debt item D4.
DEFAULT_LAMBDA = 2.5

# Floor on predicted oil-presence probability. Segmentation misses happen, so a
# hypothesis that covers no mask cell should score very badly but not
# negative-infinitely badly.
MISS_FLOOR = 1e-3


@dataclass
class LikelihoodResult:
    """Per-hypothesis evaluation."""

    hypothesis_id: np.ndarray  # (H,) the origin_marker values, sorted
    log_likelihood: np.ndarray  # (H,)
    n_particles: np.ndarray  # (H,) usable particles per hypothesis
    coverage: np.ndarray  # (H,) fraction of mask cells the plume reaches
    spill_fraction: np.ndarray  # (H,) fraction of mass landing outside the mask

    @property
    def n_hypotheses(self) -> int:
        return int(self.hypothesis_id.size)

    def weights(self, prior: np.ndarray | None = None) -> np.ndarray:
        """Normalised posterior weights, computed in log space."""
        log_w = self.log_likelihood.astype("float64")
        if prior is not None:
            with np.errstate(divide="ignore"):
                log_w = log_w + np.log(np.clip(prior, 1e-300, None))
        log_w -= log_w.max()
        w = np.exp(log_w)
        total = w.sum()
        return w / total if total > 0 else np.full_like(w, 1.0 / w.size)


def effective_cell_count(mask: ObservedMask, correlation_length_m: float) -> float:
    """How many genuinely independent observations the slick actually carries.

    This is the correction that stops the posterior being wildly overconfident,
    and it is not optional.

    The Bernoulli likelihood sums one term per mask cell. At a 500 m grid a
    128 km^2 slick is ~500 cells, so the log-likelihood is a sum of 500 terms
    and any small difference in fit gets multiplied by 500. Measured on the
    development case, that produced a 95% credible region of 306 km^2 -- an
    order of magnitude tighter than the design expects -- with the true source
    outside the 50% region and the true release time outside the 95% interval.
    Textbook overconfidence, and the single easiest way to ship a system that
    confidently names the wrong ship.

    The cells are not independent. Neighbouring cells are perfectly correlated
    at scales the drift model cannot resolve, and the binding scale is the
    forcing resolution: at CMEMS 1/12 degree the model has no information about
    structure below ~9 km, so two mask cells 2 km apart say the same thing
    about a source hypothesis. The number of independent observations is
    therefore the slick area divided by the correlation area, not the cell
    count.

    The likelihood is then tempered by ``n_effective / n_cells``, which is the
    standard treatment for a pseudo-likelihood over correlated observations.
    """
    correlation_area_km2 = (correlation_length_m / 1000.0) ** 2
    return float(np.clip(mask.area_km2 / max(correlation_area_km2, 1e-9), 1.0, mask.n_mask_cells))


def evaluate(
    traj: Trajectory,
    mask: ObservedMask,
    n_hypotheses: int,
    detectability_lambda: float = DEFAULT_LAMBDA,
    miss_floor: float = MISS_FLOOR,
    n_effective_cells: float | None = None,
) -> LikelihoodResult:
    """Score every source hypothesis against the observed slick.

    ``traj.origin_marker`` labels each particle with the hypothesis that
    released it, in ``[0, n_hypotheses)``.

    ``n_effective_cells`` tempers the likelihood for spatial correlation between
    mask cells -- see ``effective_cell_count``. Leaving it ``None`` treats every
    cell as an independent observation, which is wrong and produces a badly
    overconfident posterior; it exists only so the failure can be demonstrated
    in a test.
    """
    usable = traj.usable
    marker = traj.origin_marker
    n_cells = mask.n_mask_cells

    # Usable particles per hypothesis. A hypothesis whose particles all left the
    # domain or never released cannot be scored, and gets -inf rather than a
    # divide-by-zero.
    counts = np.bincount(marker[usable], minlength=n_hypotheses).astype("float64")
    alive = counts > 0

    x, y, m = traj.x[usable], traj.y[usable], marker[usable]
    rank = mask.cell_rank(x, y)
    inside = rank >= 0
    _, annulus = mask.classify_points(x, y)

    # --- negative term: closed form, just a weighted count -------------------
    # Cells in the boundary annulus score nothing, so they are excluded from the
    # outside mass rather than counted against the hypothesis.
    outside = ~inside & ~annulus
    mass_outside = np.bincount(m[outside], minlength=n_hypotheses).astype("float64")
    spill_fraction = np.divide(
        mass_outside, counts, out=np.zeros(n_hypotheses), where=alive
    )
    negative_term = -detectability_lambda * spill_fraction * n_cells

    # --- positive term: only over mask cells ---------------------------------
    log_lik = np.full(n_hypotheses, -np.inf, dtype="float64")
    coverage = np.zeros(n_hypotheses, dtype="float64")

    if inside.any():
        key = m[inside] * n_cells + rank[inside]
        binned = np.bincount(key, minlength=n_hypotheses * n_cells).astype("float64")
        binned = binned.reshape(n_hypotheses, n_cells)
    else:
        binned = np.zeros((n_hypotheses, n_cells), dtype="float64")

    scale = np.divide(
        np.full(n_hypotheses, float(n_cells)), counts, out=np.zeros(n_hypotheses), where=alive
    )
    rho = binned * scale[:, None]

    q = miss_floor + (1.0 - miss_floor) * (1.0 - np.exp(-detectability_lambda * rho))
    positive_term = np.log(q).sum(axis=1)
    coverage = (binned > 0).sum(axis=1) / max(n_cells, 1)

    log_lik[alive] = positive_term[alive] + negative_term[alive]

    # Temper for spatial correlation between mask cells. Without this the
    # posterior is overconfident by roughly the ratio below.
    if n_effective_cells is not None and n_cells > 0:
        log_lik[alive] *= float(n_effective_cells) / float(n_cells)

    return LikelihoodResult(
        hypothesis_id=np.arange(n_hypotheses),
        log_likelihood=log_lik,
        n_particles=counts,
        coverage=coverage,
        spill_fraction=spill_fraction,
    )


def effective_sample_size(weights: np.ndarray, particles_per_hypothesis: float) -> float:
    """Effective number of particles contributing to the posterior.

    If this collapses relative to the total, the posterior is Monte Carlo noise
    and the interface must say so rather than draw a confident-looking blob
    (watch-list W-08).
    """
    ss = float((weights**2).sum())
    if ss <= 0.0:
        return 0.0
    return (1.0 / ss) * particles_per_hypothesis


__all__ = [
    "evaluate",
    "LikelihoodResult",
    "effective_sample_size",
    "DEFAULT_LAMBDA",
    "MISS_FLOOR",
]
