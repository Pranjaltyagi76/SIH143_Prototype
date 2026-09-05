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
| Python | 3.11.x | 3.12 has occasional wheel gaps in the geospatial stack; 3.11 is the safe choice |
| NVIDIA GPU + CUDA | driver ≥ 535 | Training only. **Inference must also work on CPU** — the demo machine may not be the training machine |
| Disk | ~40 GB | Zenodo dataset ~15 GB, forcing ~5 GB, AIS ~5 GB, cases ~5 GB, working room |
| RAM | 16 GB min | Transport at 1e5 particles is comfortable; 1e6 is not |
| Browser | Chrome / Edge, current | WebGL2 required by deck.gl |
| OS | Windows 11 | **No WSL2, no Docker, no conda required** — this is the payoff of dropping SNAP and OpenDrift |

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

Serves `web/index.html` as static content plus a small JSON API over `out/`. deck.gl and MapLibre load from CDN in development — **and are vendored locally into `web/vendor/` on Day 12**, because a CDN is a network dependency and NFR-1 forbids one.

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
