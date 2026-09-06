"""Bayesian source inversion.

Backward integration narrows the search. Forward simulation computes the answer.
See Context/technical_design.md section 4.
"""

from .engine import InversionConfig, InversionDiagnostics, invert
from .hypotheses import (
    PROPOSAL_MARGIN_M,
    PROPOSAL_PARTICLES,
    HypothesisGrid,
    ProposalRegion,
    backward_proposal,
)
from .likelihood import (
    DEFAULT_LAMBDA,
    MISS_FLOOR,
    LikelihoodResult,
    effective_sample_size,
    evaluate,
)
from .mask import MASK_GRID_M, MASK_MARGIN_M, ObservedMask

__all__ = [
    "ObservedMask", "MASK_GRID_M", "MASK_MARGIN_M",
    "backward_proposal", "ProposalRegion", "HypothesisGrid",
    "PROPOSAL_PARTICLES", "PROPOSAL_MARGIN_M",
    "evaluate", "LikelihoodResult", "effective_sample_size",
    "DEFAULT_LAMBDA", "MISS_FLOOR",
    "invert", "InversionConfig", "InversionDiagnostics",
]
