"""Phase 9: the offline guarantee, enforced rather than asserted.

NFR-1 requires that no code path touches the network at demo time. The roadmap
originally planned to verify this by disabling the network adapter and running
the cases by hand -- which is a one-off check by a person who might forget, on a
machine that might differ.

These tests replace the socket layer with one that raises, then run the entire
pipeline through it. Any attempt to open a connection fails loudly and names the
address it wanted, so the guarantee is repeatable, runs in CI, and cannot rot.

The distinction matters on demo day: venue WiFi is the single most reliable
thing to fail, and the whole Case architecture exists so that it cannot matter.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "cases"
SPILL = CASES / "synth_kattegat_spill"


class NetworkAccessAttempted(AssertionError):
    """Raised when offline code tries to open a connection."""


# Loopback is not "the network". The API itself binds to 127.0.0.1, and on
# Windows asyncio builds its self-pipe from a loopback socketpair -- blocking
# those would fail the API tests for reasons that have nothing to do with NFR-1.
# What the requirement forbids is reaching OFF this machine.
LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


def _is_loopback(address) -> bool:
    host = address[0] if isinstance(address, (tuple, list)) and address else address
    return isinstance(host, str) and host in LOOPBACK


@pytest.fixture()
def no_network(monkeypatch):
    """Make every connection that leaves this machine raise, naming the host."""
    real_socket = socket.socket
    real_create = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def guard(address):
        if not _is_loopback(address):
            raise NetworkAccessAttempted(
                f"offline code attempted to reach {address!r}. "
                f"NFR-1 forbids any network call on a demo code path."
            )

    class BlockedSocket(real_socket):
        def connect(self, address):
            guard(address)
            return super().connect(address)

        def connect_ex(self, address):
            guard(address)
            return super().connect_ex(address)

    def create_connection(address, *a, **kw):
        guard(address)
        return real_create(address, *a, **kw)

    def getaddrinfo(host, *a, **kw):
        guard(host)
        return real_getaddrinfo(host, *a, **kw)

    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return guard


@pytest.fixture()
def spill_case():
    if not (SPILL / "out" / "candidates.json").is_file():
        pytest.skip("run scripts/inject_case.py then scripts/run_case.py first")
    return SPILL


# ------------------------------------------------------- the guard itself


def test_the_network_block_actually_blocks(no_network):
    """A test that cannot fail proves nothing, so prove the fixture bites."""
    with pytest.raises(NetworkAccessAttempted):
        socket.create_connection(("example.com", 80))
    with pytest.raises(NetworkAccessAttempted):
        socket.getaddrinfo("zenodo.org", 443)


def test_the_network_block_permits_loopback(no_network):
    """Loopback is not the network: the API binds to it, and on Windows asyncio
    builds its self-pipe from a loopback socketpair."""
    socket.getaddrinfo("127.0.0.1", 8000)
    assert _is_loopback(("127.0.0.1", 8000))
    assert not _is_loopback(("zenodo.org", 443))


# --------------------------------------------------- the pipeline offline


def test_full_pipeline_runs_with_no_network(no_network, spill_case, tmp_path):
    """Detection, inversion, attribution and export, all with sockets blocked."""
    import shutil

    from scripts.run_case import main

    work = tmp_path / "offline_case"
    shutil.copytree(spill_case, work, ignore=shutil.ignore_patterns("out"))
    (work / "out").mkdir(exist_ok=True)

    assert main(["--case", str(work), "--stage", "detect"]) == 0
    assert main(["--case", str(work), "--stage", "invert"]) == 0
    assert main(["--case", str(work), "--stage", "attribute"]) == 0
    assert main(["--case", str(work), "--stage", "export"]) == 0

    for produced in ("detections.json", "posterior.json", "candidates.json"):
        assert (work / "out" / produced).is_file()
    assert (work / "out" / "ui" / "manifest.json").is_file()


def test_api_serves_every_endpoint_with_no_network(no_network, spill_case):
    """The interface reads from disk only."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)
    case_id = spill_case.name
    for path in ("/healthz", "/api/cases",
                 f"/api/cases/{case_id}/detections",
                 f"/api/cases/{case_id}/posterior",
                 f"/api/cases/{case_id}/candidates",
                 f"/api/cases/{case_id}/manifest",
                 f"/api/cases/{case_id}/particles",
                 f"/api/cases/{case_id}/coastline"):
        assert client.get(path).status_code == 200, path
    assert client.get(f"/api/cases/{case_id}/scene.png").status_code == 200


def test_the_page_loads_no_remote_assets():
    """deck.gl is vendored. A CDN reference would be a network call the page
    makes on its own, which no Python-side check would catch."""
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    for remote in ("http://", "https://"):
        for line in html.splitlines():
            if remote in line and ("src=" in line or "href=" in line):
                pytest.fail(f"page references a remote asset: {line.strip()[:120]}")
    assert "/vendor/deck.gl" in html
    assert list((ROOT / "web" / "vendor").glob("deck.gl*.min.js")), "deck.gl is not vendored"


def test_synthetic_case_generation_is_offline(no_network, tmp_path):
    """Cases can be rebuilt on the demo machine without a connection."""
    from src.ingest.case_builder import SYNTH_KATTEGAT, build_synthetic_case

    case_dir = build_synthetic_case(SYNTH_KATTEGAT, tmp_path)
    assert (case_dir / "forcing" / "currents.nc").is_file()
    assert (case_dir / "ais" / "tracks.parquet").is_file()


# ------------------------------------------------------------ determinism


def test_pipeline_is_bit_identical_on_rerun(spill_case, tmp_path):
    """NFR-3. A demo that produces different numbers on the second run is a
    demo nobody can rehearse."""
    import shutil

    from scripts.run_case import main

    outputs = []
    for run in ("a", "b"):
        work = tmp_path / run
        shutil.copytree(spill_case, work, ignore=shutil.ignore_patterns("out"))
        (work / "out").mkdir(exist_ok=True)
        for stage in ("detect", "invert", "attribute"):
            main(["--case", str(work), "--stage", stage])
        outputs.append({
            name: (work / "out" / name).read_text(encoding="utf-8")
            for name in ("detections.json", "posterior.json", "candidates.json")
        })

    for name in outputs[0]:
        assert outputs[0][name] == outputs[1][name], f"{name} differs between runs"


def test_a_different_seed_changes_the_result(spill_case, tmp_path):
    """Determinism must come from the seed, not from the pipeline ignoring it."""
    import shutil

    from scripts.run_case import main

    results = []
    for seed in (11, 22):
        work = tmp_path / f"seed{seed}"
        shutil.copytree(spill_case, work, ignore=shutil.ignore_patterns("out"))
        (work / "out").mkdir(exist_ok=True)
        for stage in ("detect", "invert"):
            main(["--case", str(work), "--stage", stage, "--seed", str(seed)])
        results.append((work / "out" / "posterior.json").read_text(encoding="utf-8"))
    assert results[0] != results[1], "the seed had no effect on the posterior"


# ---------------------------------------------- no network in the run path


NETWORK_MODULES = ("requests", "urllib.request", "urllib3", "httpx", "aiohttp", "ftplib")

# Modules allowed to reach the network. Everything else runs offline.
ONLINE_ALLOWED = {"src/ingest/real.py", "src/ingest/zenodo.py"}


def test_no_run_path_module_imports_a_network_library():
    """A structural check, complementing the socket block.

    The fetchers are permitted to use the network -- that is their job -- but
    nothing the demo executes may even import a client library, because an
    import is how a network dependency creeps back in unnoticed.
    """
    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ONLINE_ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        for module in NETWORK_MODULES:
            root_name = module.split(".")[0]
            if f"import {module}" in text or f"from {module}" in text or \
               f"import {root_name}\n" in text:
                offenders.append(f"{rel} imports {module}")
    assert not offenders, "network libraries in the run path: " + "; ".join(offenders)


def test_case_folders_are_self_contained(spill_case):
    """Everything a case needs is inside its own folder."""
    required = [
        "case.json", "forcing_bundle.json",
        "scene/sigma0_vv_db.tif", "scene/incidence.tif",
        "forcing/currents.nc", "forcing/wind.nc", "forcing/land.tif",
        "ais/tracks.parquet",
    ]
    for rel in required:
        assert (spill_case / rel).is_file(), f"missing {rel}"

    bundle = json.loads((spill_case / "forcing_bundle.json").read_text(encoding="utf-8"))
    for key in ("path", "land_mask"):
        for value in _walk_strings(bundle, key):
            assert not value.startswith(("http://", "https://", "/", "\\")), (
                f"{value} is not a path relative to the case folder"
            )


def _walk_strings(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, str):
                yield v
            else:
                yield from _walk_strings(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_strings(item, key)
