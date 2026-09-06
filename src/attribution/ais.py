"""AIS ingest, cleaning and track reconstruction.

Real AIS is dirty: shared MMSI, position jumps, irregular reporting, missing
static data. We filter hard and **count everything dropped**, because an
undisclosed filter is a hidden assumption, and a hidden assumption in a system
that produces accusations is not acceptable.

Two things here are load-bearing for the attribution that follows.

**Gaps are annotated, never repaired silently.**
A gap is the single most suggestive-looking signal in AIS and the one most
likely to be benign -- shore receiver coverage, not concealment. Every track
carries its gap list, and any position derived by interpolating across one is
flagged, so interpolated positions can never quietly become evidence (W-09).

**Timestamps are UTC or the load fails.**
A naive timestamp assumed local shifts every attribution by a constant whole
number of hours: plausible, consistent, and completely wrong (W-04).
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

KNOTS_TO_MS = 0.514444

# A fix implying more than this speed from the previous one is not a real
# position report; it is a shared MMSI or a decoding error.
MAX_PLAUSIBLE_SPEED_KN = 40.0

# A silence longer than this ends one track and starts another. Bridging it
# would invent a trajectory across hours of unknown movement.
TRACK_SPLIT_GAP_HOURS = 6.0

# A silence longer than this is recorded as a gap worth reasoning about.
GAP_THRESHOLD_HOURS = 0.5


@dataclass
class CleaningReport:
    """What was thrown away. Reported, never silent."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_bad_coords: int = 0
    dropped_duplicate_fix: int = 0
    dropped_speed_jump: int = 0
    vessels_in: int = 0
    vessels_out: int = 0
    tracks_out: int = 0
    tracks_split_on_gap: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class VesselTrack:
    """One continuous run of position reports from one vessel."""

    mmsi: str
    name: str
    ship_type: str
    time: np.ndarray  # (N,) epoch seconds, ascending
    lon: np.ndarray
    lat: np.ndarray
    sog_kn: np.ndarray
    cog_deg: np.ndarray
    gaps: list[tuple[float, float]] = _field(default_factory=list)  # (start_s, end_s)

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def t_start(self) -> float:
        return float(self.time[0])

    @property
    def t_end(self) -> float:
        return float(self.time[-1])

    @property
    def transit_speed_kn(self) -> float:
        """This vessel's own typical speed, used as the anomaly baseline.

        Comparing against the vessel's own median rather than a fleet average
        avoids flagging every fishing boat as anomalous next to a container ship.
        """
        return float(np.median(self.sog_kn)) if self.sog_kn.size else 0.0

    def covers(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end

    def gap_hours_overlapping(self, t_from: float, t_to: float) -> float:
        """Total silence, in hours, inside the given window."""
        total = 0.0
        for g0, g1 in self.gaps:
            overlap = min(g1, t_to) - max(g0, t_from)
            if overlap > 0:
                total += overlap
        return total / 3600.0

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position at arbitrary times, plus a flag for interpolation across a gap.

        Positions inside a gap are still produced -- a vessel that went silent is
        precisely the case worth simulating -- but they are flagged so they can
        never be presented as observed fact.
        """
        lon = np.interp(times, self.time, self.lon)
        lat = np.interp(times, self.time, self.lat)
        crossed = np.zeros(times.shape, dtype=bool)
        for g0, g1 in self.gaps:
            crossed |= (times > g0) & (times < g1)
        return lon, lat, crossed

    def heading_near(self, t: float, window_s: float = 3600.0) -> float:
        """Mean course over ground around time ``t``, degrees."""
        sel = np.abs(self.time - t) <= window_s
        if not sel.any():
            sel = np.abs(self.time - t) == np.abs(self.time - t).min()
        ang = np.deg2rad(self.cog_deg[sel])
        return float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0)

    def speed_near(self, t_from: float, t_to: float) -> float | None:
        sel = (self.time >= t_from) & (self.time <= t_to)
        return float(np.median(self.sog_kn[sel])) if sel.any() else None

    def course_change_near(self, t_from: float, t_to: float) -> float:
        """Largest course alteration inside the window, degrees."""
        sel = (self.time >= t_from) & (self.time <= t_to)
        if sel.sum() < 3:
            return 0.0
        c = np.unwrap(np.deg2rad(self.cog_deg[sel]))
        return float(np.rad2deg(np.abs(np.diff(c)).max()))


def load_ais(path: Path | str) -> pd.DataFrame:
    """Read an AIS message log and assert its timestamps are UTC."""
    df = pd.read_parquet(path)
    required = {"timestamp", "mmsi", "lat", "lon", "sog_kn", "cog_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"AIS log is missing columns: {sorted(missing)}")
    if getattr(df["timestamp"].dt, "tz", None) is None:
        raise ValueError(
            "AIS timestamps are timezone-naive. Assuming local time shifts every "
            "attribution by a constant number of hours (W-04); the source must "
            "state its timezone."
        )
    df["mmsi"] = df["mmsi"].astype(str)
    return df


# A cleaning step that discards most of the data is almost never dirty data --
# it is a unit error in the filter. Refusing loudly beats reporting success.
MAX_SANE_DROP_FRACTION = 0.5


def clean_and_reconstruct(
    df: pd.DataFrame,
    max_speed_kn: float = MAX_PLAUSIBLE_SPEED_KN,
    split_gap_hours: float = TRACK_SPLIT_GAP_HOURS,
    gap_threshold_hours: float = GAP_THRESHOLD_HOURS,
    strict: bool = True,
) -> tuple[list[VesselTrack], CleaningReport]:
    """Clean the message log and reconstruct continuous per-vessel tracks.

    ``strict`` refuses to return a dataset the speed filter has gutted. That is
    not defensive padding: a timestamp-resolution bug once made this filter drop
    99.6% of fixes while the report still looked plausible (P-18).
    """
    report = CleaningReport(rows_in=len(df), vessels_in=int(df["mmsi"].nunique()))

    valid = df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    report.dropped_bad_coords = int((~valid).sum())
    df = df[valid].copy()

    # Resolution-independent epoch seconds. NOT astype("int64") / 1e9: pandas
    # stores this column as datetime64[us], so that idiom divides microseconds
    # by 1e9 and yields seconds/1000. Every dt then comes out 1000x too small,
    # every implied speed 1000x too large, and the speed filter silently
    # discards almost the entire dataset while reporting success. See P-18.
    df["t"] = (df["timestamp"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    df = df.sort_values(["mmsi", "t"])

    before = len(df)
    df = df.drop_duplicates(subset=["mmsi", "t"], keep="first")
    report.dropped_duplicate_fix = before - len(df)

    tracks: list[VesselTrack] = []
    for mmsi, group in df.groupby("mmsi", sort=False):
        t = group["t"].to_numpy(dtype="float64")
        lon = group["lon"].to_numpy(dtype="float64")
        lat = group["lat"].to_numpy(dtype="float64")

        # Reject fixes implying an impossible speed. A shared MMSI shows up as a
        # sequence of teleports between two real vessels.
        keep = np.ones(t.size, dtype=bool)
        if t.size > 1:
            dt = np.diff(t)
            dist_m = _haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
            with np.errstate(divide="ignore", invalid="ignore"):
                implied_kn = np.where(dt > 0, dist_m / np.maximum(dt, 1e-9), 0.0) / KNOTS_TO_MS
            keep[1:] = implied_kn <= max_speed_kn
        report.dropped_speed_jump += int((~keep).sum())

        t, lon, lat = t[keep], lon[keep], lat[keep]
        sog = group["sog_kn"].to_numpy(dtype="float64")[keep]
        cog = group["cog_deg"].to_numpy(dtype="float64")[keep]
        if t.size < 3:
            continue

        name = str(group["ship_name"].iloc[0]) if "ship_name" in group else str(mmsi)
        ship_type = str(group["ship_type"].iloc[0]) if "ship_type" in group else "Unknown"

        # Split on long silences; annotate short ones as gaps.
        breaks = np.flatnonzero(np.diff(t) > split_gap_hours * 3600.0) + 1
        segments = np.split(np.arange(t.size), breaks)
        if len(segments) > 1:
            report.tracks_split_on_gap += len(segments) - 1

        for seg in segments:
            if seg.size < 3:
                continue
            st, slon, slat = t[seg], lon[seg], lat[seg]
            dt = np.diff(st)
            gap_idx = np.flatnonzero(dt > gap_threshold_hours * 3600.0)
            gaps = [(float(st[i]), float(st[i + 1])) for i in gap_idx]
            tracks.append(
                VesselTrack(
                    mmsi=str(mmsi), name=name, ship_type=ship_type,
                    time=st, lon=slon, lat=slat,
                    sog_kn=sog[seg], cog_deg=cog[seg], gaps=gaps,
                )
            )

    if strict and report.rows_in and (
        report.dropped_speed_jump / report.rows_in > MAX_SANE_DROP_FRACTION
    ):
        raise ValueError(
            f"speed filter dropped {report.dropped_speed_jump} of {report.rows_in} "
            f"fixes ({report.dropped_speed_jump / report.rows_in:.1%}). Real AIS is "
            f"dirty but not this dirty -- check the timestamp units before trusting "
            f"this. Pass strict=False to override."
        )

    report.rows_out = sum(len(t) for t in tracks)
    report.tracks_out = len(tracks)
    report.vessels_out = len({t.mmsi for t in tracks})
    return tracks, report


def baseline_gap_hours(tracks: list[VesselTrack]) -> float:
    """The local, benign gap rate this area normally shows.

    The AIS-gap prior is measured against this, never against an absolute
    threshold. In an area with poor shore-receiver coverage every vessel has
    gaps, and an absolute threshold would manufacture suspicion out of a
    property of the receiver network (W-07).
    """
    longest = [
        max((g1 - g0 for g0, g1 in t.gaps), default=0.0) / 3600.0 for t in tracks
    ]
    return float(np.median(longest)) if longest else 0.0


def _haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    r = 6_371_000.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def to_epoch(t: datetime) -> float:
    return t.timestamp()


__all__ = [
    "VesselTrack",
    "CleaningReport",
    "load_ais",
    "clean_and_reconstruct",
    "baseline_gap_hours",
    "MAX_PLAUSIBLE_SPEED_KN",
    "TRACK_SPLIT_GAP_HOURS",
    "GAP_THRESHOLD_HOURS",
    "KNOTS_TO_MS",
]
