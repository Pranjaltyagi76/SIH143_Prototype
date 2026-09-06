"""Generate the reference fixtures for every frozen contract.

Run once in Phase 0, then commit the JSON. The fixtures are what let the whole
team work in parallel: M5 builds the entire deck.gl interface against these and
never blocks on the pipeline, and every stage owner has a concrete example of
what they must emit.

    python scripts/make_fixtures.py

The case modelled here is the primary demo case -- Kattegat, three dark patches,
one of which is correctly rejected because the 10 m wind was 1.4 m/s. That
rejection is the strongest beat in the demo, so it is in the fixtures from day
one and the UI is built to render it.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import (  # noqa: E402
    BoundingBox,
    Candidate,
    CaseManifest,
    Classification,
    CredibleRegion,
    CurrentsSpec,
    DarkVesselHypothesis,
    EvidenceFactor,
    ForcingBundle,
    PosteriorGrid,
    RankedCandidates,
    ReleaseMode,
    SceneSpec,
    SlickDetection,
    SlickEnvironment,
    SlickGeometry,
    SlickRadiometry,
    SourcePosterior,
    TimeMarginal,
    TrafficReduction,
    VesselEvidence,
    WindSpec,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

CASE_ID = "kattegat_2024_03_11"
T_OBS = datetime(2024, 3, 11, 5, 42, 13, tzinfo=timezone.utc)
T_START = datetime(2024, 3, 9, 0, 0, 0, tzinfo=timezone.utc)
AOI = BoundingBox(min_lon=10.2, min_lat=56.4, max_lon=12.8, max_lat=58.1)


def _ring(lon: float, lat: float, dx: float, dy: float) -> list[tuple[float, float]]:
    """A closed rectangular ring, for fixture geometry."""
    return [
        (lon - dx, lat - dy),
        (lon + dx, lat - dy),
        (lon + dx, lat + dy),
        (lon - dx, lat + dy),
        (lon - dx, lat - dy),
    ]


def build_case() -> CaseManifest:
    return CaseManifest(
        case_id=CASE_ID,
        description="Kattegat, elongated track-aligned slick plus one low-wind look-alike.",
        region="Kattegat / Skagerrak, Danish waters",
        aoi=AOI,
        t_obs=T_OBS,
        t_start=T_START,
        scene=SceneSpec(
            sigma0_path="scene/sigma0_vv_db.tif",
            incidence_path="scene/incidence.tif",
        ),
        seed=20260906,
        ais_source="danish_dma",
        ais_is_synthetic=False,
        notes=[
            "Real AIS from the Danish Maritime Authority.",
            "Primary validation region: free real AIS plus a resolved ocean model.",
        ],
    )


def build_forcing() -> ForcingBundle:
    return ForcingBundle(
        case_id=CASE_ID,
        aoi=AOI,
        t_start=T_START,
        t_obs=T_OBS,
        currents=CurrentsSpec(
            path="forcing/currents.nc",
            product="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
            vars=["uo", "vo"],
            includes_tides=True,
            # SMOC merges Stokes drift into uo/vo. The transport kernel reads this
            # flag and must NOT add a separate Stokes term. See P-05.
            includes_stokes=True,
            resolution_deg=0.0833,
            rms_error_ms=0.12,
        ),
        wind=WindSpec(
            path="forcing/wind.nc",
            product="ERA5 single-levels, 10 m u/v",
            vars=["u10", "v10"],
            resolution_deg=0.25,
        ),
        land_mask="forcing/land.tif",
        crs_working=AOI.utm_epsg(),
    )


def build_detections() -> list[SlickDetection]:
    """Three patches: one abstention, one oil, one look-alike.

    d01 is the demo beat. At 1.4 m/s the sea surface is already smooth, so no
    oil-water contrast is physically possible and the system refuses to judge.
    Showing what we correctly decline to flag is more persuasive than anything
    we do flag.
    """
    d01 = SlickDetection(
        detection_id=f"{CASE_ID}:d01",
        classification=Classification.UNDETERMINED,
        classification_reason=(
            "Wind 1.4 m/s is below the 3.0 m/s detectability threshold; at this wind "
            "speed the sea surface is already smooth and oil cannot produce radar "
            "contrast. Outside detectability window."
        ),
        abstained=True,
        confidence_segmentation=0.74,
        polygon_wgs84=_ring(10.95, 57.62, 0.09, 0.07),
        geometry=SlickGeometry(
            area_km2=41.7,
            perimeter_km=29.8,
            centroid=(10.95, 57.62),
            elongation=1.6,
            orientation_deg=118.0,
            n_components=1,
            complexity=1.7,
            solidity=0.88,
        ),
        radiometry=SlickRadiometry(
            damping_ratio_db=-6.2,
            damping_norm_incidence=-5.9,
            sigma0_mean_db=-22.8,
            sigma0_background_db=-16.6,
            incidence_deg=41.5,
        ),
        environment=SlickEnvironment(
            wind_speed_ms=1.4, wind_dir_deg=192.0, wind_gradient_ms_per_km=0.11
        ),
        release_mode=ReleaseMode.INDETERMINATE,
        release_mode_justification="Not assessed; patch did not pass the detectability gate.",
    )

    d02 = SlickDetection(
        detection_id=f"{CASE_ID}:d02",
        classification=Classification.OIL,
        classification_reason=(
            "Wind 6.4 m/s is inside the detectability window; incidence-normalised "
            "damping -7.6 dB with sharp boundary gradient and elongation 7.3."
        ),
        abstained=False,
        confidence_segmentation=0.81,
        polygon_wgs84=_ring(11.44, 57.12, 0.14, 0.03),
        geometry=SlickGeometry(
            area_km2=18.4,
            perimeter_km=41.2,
            centroid=(11.44, 57.12),
            elongation=7.3,
            orientation_deg=62.5,
            n_components=2,
            complexity=3.1,
            solidity=0.62,
        ),
        radiometry=SlickRadiometry(
            damping_ratio_db=-8.1,
            damping_norm_incidence=-7.6,
            sigma0_mean_db=-21.4,
            sigma0_background_db=-13.3,
            incidence_deg=38.2,
        ),
        environment=SlickEnvironment(
            wind_speed_ms=6.4, wind_dir_deg=245.0, wind_gradient_ms_per_km=0.03
        ),
        release_mode=ReleaseMode.CONTINUOUS,
        release_mode_justification=(
            "Elongation 7.3 with major axis 62.5 deg aligned within 8 deg of nearby "
            "track headings; consistent with continuous discharge under way, which "
            "constrains t0 to an interval rather than an instant."
        ),
    )

    d03 = SlickDetection(
        detection_id=f"{CASE_ID}:d03",
        classification=Classification.LOOK_ALIKE,
        classification_reason=(
            "Wind 7.1 m/s is inside the detectability window, but the boundary "
            "gradient is diffuse and shape is amorphous (solidity 0.94, elongation "
            "1.2); consistent with a biogenic film rather than mineral oil."
        ),
        abstained=False,
        confidence_segmentation=0.66,
        polygon_wgs84=_ring(12.10, 56.78, 0.11, 0.10),
        geometry=SlickGeometry(
            area_km2=63.2,
            perimeter_km=33.4,
            centroid=(12.10, 56.78),
            elongation=1.2,
            orientation_deg=15.0,
            n_components=1,
            complexity=1.4,
            solidity=0.94,
        ),
        radiometry=SlickRadiometry(
            damping_ratio_db=-4.3,
            damping_norm_incidence=-4.0,
            sigma0_mean_db=-19.9,
            sigma0_background_db=-15.6,
            incidence_deg=35.8,
        ),
        environment=SlickEnvironment(
            wind_speed_ms=7.1, wind_dir_deg=238.0, wind_gradient_ms_per_km=0.02
        ),
        release_mode=ReleaseMode.INDETERMINATE,
        release_mode_justification="Not applicable; classified as a look-alike.",
    )

    return [d01, d02, d03]


def build_posterior() -> SourcePosterior:
    t_bins = [T_OBS - timedelta(hours=h) for h in range(36, 5, -2)]
    density = [0.01, 0.02, 0.04, 0.07, 0.11, 0.15, 0.17, 0.15, 0.11, 0.09, 0.05, 0.02, 0.01, 0.00, 0.00, 0.00]
    return SourcePosterior(
        case_id=CASE_ID,
        detection_id=f"{CASE_ID}:d02",
        grid=PosteriorGrid(
            lon=[10.4 + 0.05 * i for i in range(40)],
            lat=[56.6 + 0.04 * i for i in range(35)],
            t0=t_bins,
            density_path="out/posterior.npz",
            normalised=True,
        ),
        credible_regions={
            "50": CredibleRegion(polygon_wgs84=_ring(11.18, 57.02, 0.16, 0.12), area_km2=980.0),
            "95": CredibleRegion(polygon_wgs84=_ring(11.15, 57.00, 0.34, 0.26), area_km2=4210.0),
        },
        t0_marginal=TimeMarginal(
            bins_utc=t_bins,
            density=density,
            hpd_95=(
                datetime(2024, 3, 10, 2, 0, tzinfo=timezone.utc),
                datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc),
            ),
            width_hours=14.0,
        ),
        lookback_hours=36.0,
        within_operating_envelope=True,
        n_hypotheses=4096,
        n_particles=100_000,
        effective_sample_size=7412.0,
        assumptions=[
            "windage alpha ~ U(0.01, 0.04), sampled per particle",
            "horizontal diffusivity K_h ~ LogU(1, 10) m^2/s, sampled per particle",
            "current field perturbed with correlated noise, sigma = 0.12 m/s (CMEMS QUID)",
            "CMEMS 1/12 deg cannot resolve sub-mesoscale structure below ~9 km",
            "Stokes drift taken from SMOC; no separate parameterisation applied",
            "prototype transport kernel has no weathering module",
        ],
    )


def build_candidates() -> RankedCandidates:
    def ev(overlap: float, gap: tuple[float, str], spd: tuple[float, str],
           crs: tuple[float, str], typ: tuple[float, str], axis: tuple[float, str]) -> VesselEvidence:
        return VesselEvidence(
            track_overlap_likelihood=overlap,
            ais_gap_factor=EvidenceFactor(value=gap[0], note=gap[1]),
            speed_anomaly_factor=EvidenceFactor(value=spd[0], note=spd[1]),
            course_change_factor=EvidenceFactor(value=crs[0], note=crs[1]),
            vessel_type_factor=EvidenceFactor(value=typ[0], note=typ[1]),
            axis_alignment_factor=EvidenceFactor(value=axis[0], note=axis[1]),
        )

    c1 = Candidate(
        rank=1,
        mmsi="219000000",
        vessel_name="REDACTED_IN_DEMO",
        vessel_type="Tanker",
        posterior_probability=0.41,
        log_likelihood=-142.6,
        best_discharge_window_utc=(
            datetime(2024, 3, 10, 4, 10, tzinfo=timezone.utc),
            datetime(2024, 3, 10, 7, 35, tzinfo=timezone.utc),
        ),
        evidence=ev(
            -142.6,
            (2.1, "3.2 h AIS gap against a 0.4 h local baseline gap rate"),
            (1.6, "4.1 kn against a 12.3 kn transit median for this vessel"),
            (1.0, "no significant course alteration"),
            (1.4, "tanker prior"),
            (2.8, "slick major axis within 8 deg of vessel heading"),
        ),
    )
    c2 = Candidate(
        rank=2,
        mmsi="219000001",
        vessel_name="REDACTED_IN_DEMO",
        vessel_type="Bulk Carrier",
        posterior_probability=0.24,
        log_likelihood=-147.9,
        best_discharge_window_utc=(
            datetime(2024, 3, 10, 6, 0, tzinfo=timezone.utc),
            datetime(2024, 3, 10, 9, 20, tzinfo=timezone.utc),
        ),
        evidence=ev(
            -147.9,
            (1.0, "no AIS gap in the posterior time window"),
            (1.3, "9.8 kn against an 11.9 kn transit median"),
            (1.2, "moderate course alteration near the posterior region"),
            (1.2, "bulk carrier prior"),
            (1.5, "slick major axis within 27 deg of vessel heading"),
        ),
    )
    c3 = Candidate(
        rank=3,
        mmsi="219000002",
        vessel_name="REDACTED_IN_DEMO",
        vessel_type="Cargo",
        posterior_probability=0.13,
        log_likelihood=-153.2,
        best_discharge_window_utc=(
            datetime(2024, 3, 10, 9, 45, tzinfo=timezone.utc),
            datetime(2024, 3, 10, 12, 5, tzinfo=timezone.utc),
        ),
        evidence=ev(
            -153.2,
            (1.4, "1.1 h AIS gap against a 0.4 h local baseline gap rate"),
            (1.0, "no speed anomaly"),
            (1.0, "no significant course alteration"),
            (1.0, "general cargo, neutral prior"),
            (1.1, "slick major axis within 44 deg of vessel heading"),
        ),
    )

    return RankedCandidates(
        case_id=CASE_ID,
        traffic_reduction=TrafficReduction(vessels_in_window=214, after_prefilter=19, reported=3),
        candidates=[c1, c2, c3],
        dark_vessel_hypothesis=DarkVesselHypothesis(
            posterior_probability=0.22,
            prior_used=0.15,
            note=(
                "Posterior mass not explained by any AIS-observed track. The source may "
                "have been a vessel not transmitting AIS; this hypothesis competes on "
                "equal terms with the named candidates."
            ),
        ),
    )


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    artefacts = {
        "case.json": build_case(),
        "forcing_bundle.json": build_forcing(),
        "source_posterior.json": build_posterior(),
        "ranked_candidates.json": build_candidates(),
    }

    for name, model in artefacts.items():
        (FIXTURES / name).write_text(model.model_dump_json(indent=2), encoding="utf-8")
        print(f"  wrote {name}")

    detections = build_detections()
    payload = "[\n" + ",\n".join(d.model_dump_json(indent=2) for d in detections) + "\n]\n"
    (FIXTURES / "slick_detections.json").write_text(payload, encoding="utf-8")
    print(f"  wrote slick_detections.json ({len(detections)} detections)")

    print(f"\nFixtures written to {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
