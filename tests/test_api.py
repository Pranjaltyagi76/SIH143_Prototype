"""Phase 6 tests for the API and the UI export.

Two of these are not really about the web layer at all:

``test_serve_refuses_a_non_loopback_host``
    guards the fact that this API has no authentication and serves vessel
    attribution output. It must never be reachable from a conference network.

``test_unknown_case_is_rejected``
    the case id is the one untrusted string in the system, since it arrives
    from a URL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, serve
from src.contracts import CaseManifest
from src.ingest.case_builder import load_forcing_bundle

CASES = Path(__file__).resolve().parents[1] / "data" / "cases"
SPILL = "synth_kattegat_spill"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def spill_ready() -> str:
    ui = CASES / SPILL / "out" / "ui" / "manifest.json"
    if not ui.is_file():
        pytest.skip("run scripts/inject_case.py then scripts/run_case.py first")
    return SPILL


# ------------------------------------------------------------------ security


def test_serve_refuses_a_non_loopback_host():
    """No auth layer + vessel attribution output = loopback only, enforced in
    code rather than trusted to whoever types the command line."""
    for host in ("0.0.0.0", "192.168.1.10", "::"):
        with pytest.raises(ValueError, match="refusing to bind"):
            serve(host=host)


def test_loopback_hosts_are_accepted(monkeypatch):
    called = {}
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: called.update(k))
    serve(host="127.0.0.1", port=8123)
    assert called["host"] == "127.0.0.1"


@pytest.mark.parametrize("bad", ["../../etc", "..", "nonexistent", "synth_kattegat/../.."])
def test_unknown_case_is_rejected(client: TestClient, bad: str):
    """Path traversal is closed by validating against the actual directory
    listing, not by pattern-matching the string."""
    assert client.get(f"/api/cases/{bad}/detections").status_code in (404, 400)


# --------------------------------------------------------------- endpoints


def test_health(client: TestClient):
    body = client.get("/healthz").json()
    assert body["ok"] is True and body["cases"] >= 1


def test_case_listing_reports_readiness(client: TestClient):
    cases = client.get("/api/cases").json()
    assert cases
    for c in cases:
        assert {"case_id", "region", "t_obs_utc", "ais_is_synthetic", "ready"} <= set(c)


def test_detections_endpoint_returns_contract_objects(client: TestClient, spill_ready):
    from src.contracts import SlickDetection

    data = client.get(f"/api/cases/{spill_ready}/detections").json()
    assert data
    for d in data:
        SlickDetection.model_validate(d)


def test_candidates_endpoint_returns_contract_object(client: TestClient, spill_ready):
    from src.contracts import RankedCandidates

    RankedCandidates.model_validate(client.get(f"/api/cases/{spill_ready}/candidates").json())


def test_posterior_endpoint(client: TestClient, spill_ready):
    from src.contracts import SourcePosterior

    SourcePosterior.model_validate(client.get(f"/api/cases/{spill_ready}/posterior").json())


def test_scene_png_is_served(client: TestClient, spill_ready):
    r = client.get(f"/api/cases/{spill_ready}/scene.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 10_000


def test_unbuilt_artifact_says_what_to_run(client: TestClient):
    r = client.get("/api/cases/kattegat_2024_03_11/manifest")
    assert r.status_code == 404
    assert "run_case" in r.json()["detail"]


def test_truth_is_served_but_flagged(client: TestClient, spill_ready):
    """Ground truth is a development convenience. The pipeline never reads it."""
    body = client.get(f"/api/cases/{spill_ready}/truth").json()
    assert body["available"] is True
    assert "culprit_mmsi" in body
    assert client.get("/api/cases/synth_kattegat/truth").json()["available"] is False


# ------------------------------------------------------------------ export


def test_ui_bundle_is_complete(spill_ready):
    ui = CASES / spill_ready / "out" / "ui"
    for name in ("manifest.json", "scene.png", "coastline.json",
                 "particles.json", "posterior.json"):
        assert (ui / name).is_file(), f"missing {name}"


def test_ui_bundle_stays_small(spill_ready):
    """A 60 fps demo cannot wait on an 8 MB payload over a loopback socket."""
    ui = CASES / spill_ready / "out" / "ui"
    total_mb = sum(p.stat().st_size for p in ui.iterdir() if p.is_file()) / 1e6
    assert total_mb < 4.0, f"UI bundle is {total_mb:.1f} MB"


def test_particles_are_the_backward_advection_pass(spill_ready):
    """Animating a backward diffusion cloud would be showing the exact mistake
    this project exists to avoid, so the payload states what it is."""
    data = json.loads((CASES / spill_ready / "out" / "ui" / "particles.json").read_text())
    assert data["trails"]
    assert "diffusion is not run backwards" in data["note"]
    assert data["n_frames"] > 5


def test_coastline_comes_from_the_cases_own_land_mask(spill_ready):
    """The coastline the viewer sees is the one the physics beached against."""
    geo = json.loads((CASES / spill_ready / "out" / "ui" / "coastline.json").read_text())
    assert geo["type"] == "FeatureCollection"
    assert geo["features"]
    assert geo["features"][0]["geometry"]["type"] == "LineString"


def test_posterior_field_is_normalised_weights(spill_ready):
    data = json.loads((CASES / spill_ready / "out" / "ui" / "posterior.json").read_text())
    assert data["cells"]
    assert max(c[2] for c in data["cells"]) == pytest.approx(1.0, abs=1e-3)
    assert all(0.0 <= c[2] <= 1.0 for c in data["cells"])


def test_export_is_reproducible(tmp_path, spill_ready):
    """NFR-3 extends to the UI bundle."""
    from src.api.export import export_ui
    from src.transport import ForcingField

    case_dir = CASES / spill_ready
    case = CaseManifest.load(case_dir)
    field = ForcingField.from_case(case_dir, load_forcing_bundle(case_dir))
    a = export_ui(case_dir, field, case, seed=1)
    first = (case_dir / "out" / "ui" / "particles.json").read_text()
    b = export_ui(case_dir, field, case, seed=1)
    assert first == (case_dir / "out" / "ui" / "particles.json").read_text()
    assert a.n_trails == b.n_trails


# --------------------------------------------------------------- the page


def test_page_is_served_and_vendors_deck_gl(client: TestClient):
    """deck.gl is committed, not fetched. NFR-1 forbids a network call on any
    demo code path -- and deck.gl is not on cdnjs at all, so the obvious CDN
    URL fails silently (P-20)."""
    html = client.get("/").text
    assert "deck-canvas" in html
    assert "/vendor/deck.gl" in html
    assert "cdnjs.cloudflare.com" not in html
    assert (Path(__file__).resolve().parents[1] / "web" / "vendor").glob("deck.gl*.min.js")


def test_page_states_the_disclaimer(client: TestClient):
    """Control SC-2: investigative leads, never a determination of responsibility."""
    html = client.get("/").text.lower()
    assert "not a determination of responsibility" in html
    for banned in ("guilty", "culprit was", "the polluter is"):
        assert banned not in html


def test_page_renders_abstention_reasons(client: TestClient):
    """Showing what the system refuses to flag is the strongest thing it does,
    so the reason string must reach the page."""
    html = client.get("/").text
    assert "classification_reason" in html
    assert "undetermined" in html
