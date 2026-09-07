"""FastAPI application serving the interface and the pipeline's output.

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

**Bound to loopback, deliberately.** There is no authentication layer -- correct
for a single-user offline tool -- which is exactly why the API must never be
reachable from a conference network. ``serve()`` refuses a non-loopback host
rather than trusting whoever types the command line.

Every response is read from a Case folder on disk. Nothing here fetches
anything, so the demo satisfies NFR-1 structurally rather than by discipline.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = ROOT / "data" / "cases"
WEB_DIR = ROOT / "web"

app = FastAPI(
    title="SIH26143 — oil spill detection and vessel attribution",
    description="Investigative leads under stated model assumptions. "
                "Not a determination of responsibility.",
    version="0.6.0",
)


def _case_dir(case_id: str) -> Path:
    """Resolve a case id to a folder, refusing anything outside the cases root.

    The case id arrives from the URL, so it is the one untrusted string in the
    system. Validating against the actual directory listing closes path
    traversal by construction rather than by pattern-matching.
    """
    if case_id not in {p.name for p in CASES_DIR.iterdir() if p.is_dir()}:
        raise HTTPException(status_code=404, detail=f"unknown case {case_id!r}")
    return CASES_DIR / case_id


def _artifact(case_id: str, *parts: str) -> JSONResponse:
    path = _case_dir(case_id).joinpath(*parts)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{'/'.join(parts)} not built for {case_id}; run scripts/run_case.py",
        )
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/cases")
def list_cases() -> list[dict]:
    """Cases available on disk, newest-looking first."""
    from src.contracts import CaseManifest

    out = []
    for folder in sorted(CASES_DIR.iterdir()):
        if not (folder / "case.json").is_file():
            continue
        try:
            case = CaseManifest.load(folder)
        except Exception:
            continue
        out.append({
            "case_id": case.case_id,
            "region": case.region,
            "description": case.description,
            "t_obs_utc": case.t_obs.isoformat(),
            "ais_is_synthetic": case.ais_is_synthetic,
            "ready": (folder / "out" / "ui" / "manifest.json").is_file(),
        })
    return out


@app.get("/api/cases/{case_id}/manifest")
def manifest(case_id: str):
    return _artifact(case_id, "out", "ui", "manifest.json")


@app.get("/api/cases/{case_id}/detections")
def detections(case_id: str):
    return _artifact(case_id, "out", "detections.json")


@app.get("/api/cases/{case_id}/posterior")
def posterior(case_id: str):
    return _artifact(case_id, "out", "posterior.json")


@app.get("/api/cases/{case_id}/candidates")
def candidates(case_id: str):
    return _artifact(case_id, "out", "candidates.json")


@app.get("/api/cases/{case_id}/particles")
def particles(case_id: str):
    return _artifact(case_id, "out", "ui", "particles.json")


@app.get("/api/cases/{case_id}/posterior-field")
def posterior_field(case_id: str):
    return _artifact(case_id, "out", "ui", "posterior.json")


@app.get("/api/cases/{case_id}/coastline")
def coastline(case_id: str):
    return _artifact(case_id, "out", "ui", "coastline.json")


@app.get("/api/cases/{case_id}/scene.png")
def scene(case_id: str) -> FileResponse:
    path = _case_dir(case_id) / "out" / "ui" / "scene.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="scene not exported; run run_case.py")
    return FileResponse(path, media_type="image/png")


@app.get("/api/cases/{case_id}/truth")
def truth(case_id: str):
    """Ground truth for an injected case, shown only after a result is revealed.

    Serving this is a development convenience. It is never read by the pipeline,
    and the page requests it only to score itself.
    """
    path = _case_dir(case_id) / "truth.json"
    if not path.is_file():
        return JSONResponse({"available": False})
    return JSONResponse({"available": True, **json.loads(path.read_text(encoding="utf-8"))})


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "cases": len(list_cases())}


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the server, refusing to expose it beyond this machine."""
    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            f"refusing to bind to {host!r}. This API has no authentication layer "
            f"and serves vessel attribution output; it must not be reachable from "
            f"a conference network. Use 127.0.0.1."
        )
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["app", "serve"]
