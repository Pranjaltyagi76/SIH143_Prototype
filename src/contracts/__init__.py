"""The frozen contracts.

Four inter-stage contracts plus the Case manifest that binds them. These are the
ONLY interface between pipeline stages: no stage reaches into another's
internals. Frozen in Phase 0; changing them is a deliberate, announced act.

See Context/architecture.md section 3.

    ForcingBundle     cached ocean + wind, and the offline guarantee
    SlickDetection    stages 1-3: segmentation, physics gate, characterisation
    SourcePosterior   stage 5: Bayesian source inversion
    RankedCandidates  stage 7: vessel attribution
"""

from .candidates import (
    BANNED_TERMS,
    Candidate,
    DarkVesselHypothesis,
    EvidenceFactor,
    RankedCandidates,
    TrafficReduction,
    VesselEvidence,
    reject_accusatory_language,
)
from .case import CASE_LAYOUT, CASE_MANIFEST_NAME, CaseManifest, SceneSpec
from .common import (
    BoundingBox,
    Classification,
    LonLat,
    Polygon,
    ReleaseMode,
    StrictModel,
    require_utc,
    validate_polygon,
)
from .detection import (
    WIND_MAX_MS,
    WIND_MIN_MS,
    SlickDetection,
    SlickEnvironment,
    SlickGeometry,
    SlickRadiometry,
)
from .forcing import CurrentsSpec, ForcingBundle, WindSpec
from .posterior import (
    MAX_USEFUL_LOOKBACK_HOURS,
    CredibleRegion,
    PosteriorGrid,
    SourcePosterior,
    TimeMarginal,
)

__all__ = [
    # common
    "BoundingBox",
    "Classification",
    "LonLat",
    "Polygon",
    "ReleaseMode",
    "StrictModel",
    "require_utc",
    "validate_polygon",
    # case
    "CaseManifest",
    "SceneSpec",
    "CASE_MANIFEST_NAME",
    "CASE_LAYOUT",
    # forcing
    "ForcingBundle",
    "CurrentsSpec",
    "WindSpec",
    # detection
    "SlickDetection",
    "SlickGeometry",
    "SlickRadiometry",
    "SlickEnvironment",
    "WIND_MIN_MS",
    "WIND_MAX_MS",
    # posterior
    "SourcePosterior",
    "PosteriorGrid",
    "CredibleRegion",
    "TimeMarginal",
    "MAX_USEFUL_LOOKBACK_HOURS",
    # candidates
    "RankedCandidates",
    "Candidate",
    "DarkVesselHypothesis",
    "VesselEvidence",
    "EvidenceFactor",
    "TrafficReduction",
    "BANNED_TERMS",
    "reject_accusatory_language",
]
