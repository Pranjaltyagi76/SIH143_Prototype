"""Shared types used across all four frozen contracts.

See Context/architecture.md section 3. These types are frozen as of Phase 0 and
must not change: every stage of the pipeline is written against them, and six
people build in parallel only because they can rely on that.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A single (longitude, latitude) pair in EPSG:4326, degrees.
LonLat = tuple[float, float]

# A closed ring of LonLat vertices.
Polygon = list[LonLat]


class StrictModel(BaseModel):
    """Base for every contract model.

    ``extra="forbid"`` is the whole point: a typo in a field name, or a stage
    quietly adding an undeclared field, fails loudly at the boundary instead of
    silently vanishing downstream. This is the mechanism that makes "contracts
    frozen on Day 2" mean something.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Classification(str, Enum):
    """Stage 2 output. See Context/technical_design.md section 1.4.

    ``UNDETERMINED`` is not a failure state. Abstaining when the 10 m wind is
    outside the detectability window is the correct answer, because below about
    3 m/s no oil-water contrast is physically possible.
    """

    OIL = "oil"
    LOOK_ALIKE = "look_alike"
    UNDETERMINED = "undetermined"


class ReleaseMode(str, Enum):
    """Inferred discharge geometry. See Context/technical_design.md section 2.

    ``CONTINUOUS`` constrains t0 to an interval rather than an instant, which
    materially narrows the inversion.
    """

    CONTINUOUS = "continuous"
    INSTANTANEOUS = "instantaneous"
    INDETERMINATE = "indeterminate"


def require_utc(value: datetime) -> datetime:
    """Reject naive or non-UTC datetimes.

    Timezone drift is watch-list item W-04 in
    Context/problems_faced_and_bugs_encountered.md: AIS timestamps assumed local
    instead of UTC produce an attribution answer that is wrong by a constant
    whole number of hours, which is exactly the kind of error that looks
    plausible and survives review. We refuse the input instead.
    """
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware and UTC (got naive datetime)")
    if value.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(f"datetime must be UTC (got offset {value.utcoffset()})")
    return value


class BoundingBox(StrictModel):
    """Area of interest in EPSG:4326 degrees."""

    min_lon: float = Field(ge=-180.0, le=180.0)
    min_lat: float = Field(ge=-90.0, le=90.0)
    max_lon: float = Field(ge=-180.0, le=180.0)
    max_lat: float = Field(ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def check_ordering(self) -> BoundingBox:
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be < max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be < max_lat")
        return self

    @property
    def centroid(self) -> LonLat:
        return ((self.min_lon + self.max_lon) / 2.0, (self.min_lat + self.max_lat) / 2.0)

    def utm_epsg(self) -> str:
        """Local UTM zone for this AOI, as an EPSG code string.

        All computation happens in projected metres, never in degrees. Computing
        area or distance in degrees is watch-list item W-11: it produces values
        wrong by a latitude-dependent factor, silently.
        """
        lon, lat = self.centroid
        zone = int((lon + 180.0) / 6.0) + 1
        return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def validate_polygon(poly: Polygon) -> Polygon:
    """A polygon needs at least three distinct vertices and a closed ring."""
    if len(poly) < 4:
        raise ValueError("polygon needs >= 4 vertices (3 distinct plus closure)")
    if poly[0] != poly[-1]:
        raise ValueError("polygon ring must be closed (first vertex == last vertex)")
    for lon, lat in poly:
        if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
            raise ValueError(f"vertex out of range: ({lon}, {lat})")
    return poly


__all__ = [
    "LonLat",
    "Polygon",
    "StrictModel",
    "Classification",
    "ReleaseMode",
    "BoundingBox",
    "require_utc",
    "validate_polygon",
    "field_validator",
    "model_validator",
    "Field",
]
