# Deployment

**Status:** 🔧 Live document — written during development, finalised after the demo · **Owner:** M6 · **Last updated:** 2026-09-05

---

## 0. What "deployment" means here

There is no cloud, no cluster, no CI/CD pipeline. **The deployment target is one laptop, on a stage, with the WiFi off.**

Every decision below follows from that. The system is a batch pipeline over immutable Case folders (see [architecture.md](architecture.md) §1), which means "deploying" is: copy a folder, create a venv, run a script.

> **Design rule:** if any code path can touch the network at demo time, the architecture is wrong. Offline operation is structural, not disciplinary.

---

## 1. Environments

| Environment | Where | Purpose | Network |
|---|---|---|---|
| **Build** | Any dev machine | Fetching data, assembling Case folders, training | ✅ required |
| **Dev** | Each member's machine | Component work against fixtures | ✅ optional |
| **Demo** | One designated laptop | Round 1 presentation | ❌ **disabled** |
| **Backup demo** | A second laptop, identically prepared | If the primary dies | ❌ disabled |

**The demo machine is designated on Day 10 and frozen on Day 14.** After freeze, nothing is installed on it.

---

## 2. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11.x preferred | 3.12+ has occasional wheel gaps in the geospatial stack. **See §2.2 — the dev machine currently has 3.14 and 3.13 only** |
| NVIDIA GPU + CUDA | driver ≥ 535 | Training only. **Inference must also work on CPU** — the demo machine may not be the training machine |
| Disk | ~40 GB | Zenodo dataset ~15 GB, forcing ~5 GB, AIS ~5 GB, cases ~5 GB, working room |
| RAM | 16 GB min | Transport at 1e5 particles is comfortable; 1e6 is not |
| Browser | Chrome / Edge, current | WebGL2 required by deck.gl |
| OS | Windows 11 | **No WSL2, no Docker, no conda required** — this is the payoff of dropping SNAP and OpenDrift |

### 2.1 Actual dev/demo machine (verified 2026-09-05)

| Resource | Measured | Verdict |
|---|---|---|
| Disk free | 161 GB | ✅ Ample |
| RAM | 15.3 GB | ✅ Meets target. Keep the ensemble at 1e5–4e5 particles, not 1e6 |
| GPU | RTX 3050 Laptop, **4 GB VRAM**, driver 591.84 | ⚠️ **Tight.** See §2.3 |
| Python | 3.14.6 (default), 3.13 | ⚠️ **Needs resolution.** See §2.2 |

### 2.2 Python version — ✅ RESOLVED in Phase 0 (2026-09-06)

The machine has Python 3.14 and 3.13; neither is the previously recommended 3.11, and PyTorch plus the geospatial wheels historically lag new CPython releases by months. Rather than assume, we resolved it empirically with a `pip install --dry-run` sweep of the full dependency set.

**Result: Python 3.13.14 works for everything. Python 3.11 is not needed.**

| Package | Resolved on 3.13 |
|---|---|
| torch (CUDA) | ✅ **2.14.0+cu126** — `--index-url https://download.pytorch.org/whl/cu126` |
| torch (CPU, PyPI) | ✅ 2.14.0 |
| segmentation_models_pytorch | ✅ 0.5.0 |
| numpy / scipy / pandas | ✅ 2.5.2 / 1.18.1 / 3.0.5 |
| xarray / netCDF4 | ✅ 2026.7.0 / 1.7.4 |
| rasterio / shapely / pyproj / geopandas | ✅ 1.5.1 / 2.1.2 / 3.8.0 / 1.1.4 |
| lightgbm / scikit-learn / scikit-image | ✅ 4.7.0 / 1.9.0 / 0.26.0 |
| duckdb / fastapi / uvicorn | ✅ 1.5.5 / 0.141.1 / 0.52.4 |

Note `cu121` has no 3.13 build; use **cu126**. Exact pins are in `requirements.txt`.

The venv is created with `py -3.13 -m venv .venv`. Python 3.14 remains the system default and is not used by this project.

### 2.3 The 4 GB VRAM constraint

4 GB is enough for this project but not by much. Plan for it rather than discovering it mid-training:

| Setting | Value | Why |
|---|---|---|
| Tile size | 256 x 256 | Already the plan |
| Batch size | 8, falling back to 4 | A ResNet34 U-Net at 256² in fp32 with batch 16 will OOM at 4 GB |
| **Mixed precision** | **`torch.amp` enabled** | Roughly halves activation memory. Effectively mandatory here, not an optimisation |
| Gradient accumulation | 2–4 steps | Recovers an effective batch of 16–32 without the memory |
| Encoder | ResNet34, **not** ResNet50+ | Another reason the architecture choice was right |
| Inference | Also verified on CPU | The demo path must not depend on the GPU |

This costs nothing and is well within reach. It does mean **U-Net training is the one phase that cannot be rushed** — expect a few hours per full run, so start it early and let it run overnight rather than blocking a work session on it.

### The Windows decision, recorded

Dropping SNAP (Java/snappy) and OpenDrift (conda/GDAL) removes the entire class of Windows environment failure from the critical path. The remaining stack — PyTorch, NumPy/SciPy, xarray, rasterio, shapely, pyproj, LightGBM, FastAPI — all ship prebuilt Windows wheels on PyPI and install with plain `pip` into a plain `venv`.

This was the single largest schedule risk in the original plan. It is now zero. See [architecture.md](architecture.md) §6 for the trade-off accepted in exchange.

---

## 3. Installation

```bash
git clone <repo> SIH143 && cd SIH143
```

```bash
python -3.11 -m venv .venv
```

```bash
.venv\Scripts\activate && python -m pip install --upgrade pip && pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

### requirements.txt policy

- **Every version pinned exactly.** No `>=`, no unpinned transitive surprises on Day 13.
- Split into `requirements.txt` (runtime) and `requirements-dev.txt` (pytest, matplotlib, jupyter).
- PyTorch installed from the CUDA index URL, documented as a comment in the file.
- **A wheel cache (`pip download -d vendor/`) is committed to the backup drive on Day 12** so the demo machine can be rebuilt from scratch with no internet.

---

## 4. Configuration and secrets

| Item | Where | Committed? |
|---|---|---|
| CDSE / CMEMS / CDS credentials | `.env`, loaded by `python-dotenv` | ❌ **never** — `.gitignore`d on Day 1 |
| Case parameters (AOI, t_obs, seed) | `data/cases/<id>/case.json` | ✅ yes — they are the reproducibility record |
| Model hyperparameters | `configs/*.yaml` | ✅ |
| Trained weights | `models/*.pt` | Git LFS, or the backup drive if LFS is awkward |

`.env.example` is committed with empty values so a new machine knows what it needs. See [security_review.md](security_review.md).

---

## 5. Build phase — assembling Cases (online, done days early)

```bash
python scripts/build_case.py --config configs/cases/kattegat_2024_03_11.yaml
```

This is the **only** script permitted to touch the network. It fetches the SAR tile, subsets CMEMS and ERA5 to the AOI and time window, pulls and filters AIS, writes `case.json`, and validates the result against the `ForcingBundle` contract.

**Cases are frozen on Day 7.** No new data after that — a case assembled on Day 13 has not been tested.

### Training

```bash
python scripts/train_unet.py --config configs/unet_resnet34.yaml
```

Runs on the GPU machine. Exports `models/unet_resnet34.pt`, which is copied to the demo machine. **Inference is verified on CPU** before the demo, because the demo laptop may not have the training GPU.

---

## 6. Run phase — executing a case (fully offline)

```bash
python scripts/run_case.py --case data/cases/kattegat_2024_03_11
```

Stages skip when their output exists and is newer than their inputs. `--force` recomputes; `--stage detect` runs one stage; `--seed N` overrides the case seed.

Writes `out/detections.json`, `out/posterior.npz`, `out/candidates.json`, `out/particles.json`, and `out/log.jsonl` (structured timings — the raw material for [performance_review.md](performance_review.md)).

---

## 7. Serve phase

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**Bound to `127.0.0.1` deliberately, never `0.0.0.0`.** There is no authentication layer, so the API must not be reachable from a conference WiFi network. This is enforced in code, not left to a command-line flag someone might change.

Serves `web/index.html` as static content plus a small JSON API over `out/`. deck.gl loads from CDN in development — **and is vendored locally into `web/vendor/` on Day 12**, because a CDN is a network dependency and NFR-1 forbids one.

**There is no tile basemap.** Land is drawn from Natural Earth 1:10m coastline polygons (public domain, ~20 MB, committed once to `web/assets/`) as a deck.gl `GeoJsonLayer` over a solid ocean fill. Every tile service — Mapbox, MapTiler, OSM — requires an API key, a network connection, or both, and NFR-1 rules all of them out regardless of price. This is not a compromise: an ocean application needs coastlines, not street-level raster tiles.

| Endpoint | Returns |
|---|---|
| `GET /api/cases` | List of available cases |
| `GET /api/cases/{id}/detections` | `SlickDetection[]` |
| `GET /api/cases/{id}/posterior` | `SourcePosterior` + heatmap grid |
| `GET /api/cases/{id}/candidates` | `RankedCandidates` |
| `GET /api/cases/{id}/particles` | Downsampled trajectories for animation |
| `GET /api/cases/{id}/evidence.pdf` | Evidence bundle (Tier B) |

---

## 8. Demo-day runbook

### T−1 day (Day 13)

- [ ] All three cases run end to end on the demo machine, network disabled
- [ ] `out/` folders cached and backed up
- [ ] Browser bookmark set; zoom level and window size fixed
- [ ] Screen resolution matched to the projector; **text legibility checked from 5 m**
- [ ] Battery charged; charger packed; **laptop set to never sleep**
- [ ] Backup laptop prepared identically and verified independently
- [ ] Two USB drives + one cloud folder with the full repo, models, cases and cached outputs
- [ ] Demo video exported and playable offline — the ultimate fallback

### T−30 min

- [ ] Cold boot the demo machine
- [ ] **Disable the network adapter**
- [ ] Activate venv, start uvicorn, open the browser, load each case once to warm caches
- [ ] Close every other application — Slack, email, notifications off
- [ ] Confirm the projector shows what you expect, at the resolution you tested

### Failure drills — rehearsed on Day 14, not improvised on stage

| Failure | Response | Recovery |
|---|---|---|
| A stage crashes mid-demo | The UI loads from cached `out/` — **say nothing, keep talking** | Cached outputs are the primary demo path, not the fallback |
| Browser hangs | Ctrl-Shift-R; second browser window already open on the same page | < 10 s |
| uvicorn dies | Second terminal already running an identical server on port 8001 | < 5 s |
| Laptop dies | Switch to the backup laptop | < 60 s |
| Everything dies | Play the demo video from a USB drive | Immediate |
| Judge asks to run an unprepared scene | *"The pipeline runs on any Sentinel-1 scene, but building a case requires downloading forcing data for that region and window — a few minutes online. Here are three we prepared."* Then offer to run one live from cache | Honest and prepared |

> **The cached-output path is the primary demo path.** Live computation on stage is a risk with no reward — nobody in the audience can tell the difference, and everybody can tell when it hangs.

---

## 9. Packaging (Round 3, not now)

Docker Compose with two services (API, worker) and a volume for cases is the obvious Round 3 packaging. **It is deliberately not built for Round 1** — it adds an install dependency and a failure mode to a demo that runs fine from a venv, and it earns zero marks. Recorded here so Round 2 can point at the plan.

---

## 10. Deployment log

*Filled in during the sprint.*

| Date | Event | Notes |
|---|---|---|
| 2026-09-05 | Document created; deployment target and offline constraint fixed | Pre-coding |
| | Demo machine designated | *(Day 10)* |
| | CDN assets vendored locally | *(Day 12)* |
| | Offline run verified on demo machine | *(Day 12)* |
| | Code freeze and backups | *(Day 14)* |
