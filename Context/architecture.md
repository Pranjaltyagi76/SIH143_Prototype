# Architecture

**Status:** Frozen before coding · **Owner:** M6 · **Last updated:** 2026-09-05

---

## 1. The organising idea: everything is a `Case`

The single most important architectural decision is that the system is **not a service**. It is a **batch pipeline over immutable Case folders.**

A `Case` is a self-contained, on-disk bundle holding everything needed to reproduce one analysis:

```
data/cases/kattegat_2024_03_11/
├── case.json              ← the Case manifest (AOI, t_obs, provenance, config)
├── scene/
│   ├── sigma0_vv_db.tif   ← calibrated backscatter, georeferenced
│   └── incidence.tif      ← incidence angle per pixel
├── forcing/
│   ├── currents.nc        ← CMEMS SMOC, subset to AOI + time window
│   └── wind.nc            ← ERA5 10 m u/v, subset
├── ais/
│   └── tracks.parquet     ← AIS, prefiltered to AOI + time window
└── out/                   ← everything the pipeline produces
    ├── detections.json
    ├── posterior.npz
    ├── candidates.json
    └── particles.json     ← downsampled, for the UI
```

Why this matters more than it looks:

| Property | Consequence |
|---|---|
| **Offline by construction** | No code path touches the network at demo time. NFR-1 is satisfied structurally, not by discipline |
| **Reproducible** | A case + a seed is the complete input. Bit-identical reruns |
| **Parallelisable across the team** | M5 builds the UI against `out/` fixtures while M2 is still writing the inversion |
| **Cacheable** | Slow stages write their output; the demo can run from cached `out/` if anything breaks live |
| **Debuggable** | Every intermediate is a file you can open |

`scripts/run_case.py --case <path>` executes the whole pipeline. Each stage skips if its output exists and is newer than its inputs, unless `--force`.

---

## 2. System diagram

```
┌─── INGEST (offline, run once per case) ─────────────────────────┐
│  Sentinel-1 GRD IW VV sigma-0 dB   (Zenodo tiles / CDSE)        │
│  CMEMS SMOC  — currents + tides + Stokes, hourly, 1/12°         │
│  ERA5 10 m wind — hourly, 0.25°                                 │
│  AIS — Danish DMA (real) / generator (synthetic Arabian Sea)    │
│                          ↓ writes a Case folder                 │
└─────────────────────────────────────────────────────────────────┘
                            │
   ═══════════ everything below runs fully offline ═══════════
                            │
┌─── STAGE 1 · SEGMENTATION ──────────────────────────────────────┐
│  U-Net (ResNet34 encoder, ImageNet init), Dice + Focal loss     │
│  Deliberately HIGH RECALL — precision is Stage 2's job          │
│  OUT: candidate dark-patch polygons                             │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─── STAGE 2 · PHYSICS GATE + LOOK-ALIKE REJECTION ★ ─────────────┐
│  hard gate:  ERA5 wind ∉ [3, 12] m/s  ⇒  UNDETERMINED           │
│  then LightGBM on ~14 physical + geometric features             │
│  OUT: oil | look-alike | undetermined,  each with a reason      │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─── STAGE 3 · CHARACTERISATION ──────────────────────────────────┐
│  area · perimeter · elongation · orientation · fragmentation    │
│  complexity · solidity · damping stats                          │
│  → release-mode inference (continuous vs instantaneous)         │
│  → constrains the t0 prior for Stage 5                          │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─── STAGE 4 · TRANSPORT KERNEL (shared engine) ──────────────────┐
│  RK4 advection + sampled windage + random-walk diffusion        │
│  Runs FORWARD and BACKWARD. Particles carry origin_marker.      │
│  ONE implementation, used by stages 5, 6 and 7                  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─── STAGE 5 · SOURCE INVERSION ★★  (the scientific core) ────────┐
│  (i)   backward advection-only pass  → proposal region R(t0)    │
│  (ii)  partition R x T into source hypotheses h                 │
│  (iii) forward-simulate all h in ONE run, tagged by origin      │
│  (iv)  HYPOTHESIS-LEVEL likelihood vs observed mask             │
│  (v)   KDE → posterior p(x0, t0 | obs) + credible regions       │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
           ▼                              ▼
┌─── STAGE 6 · FORECAST ─────┐  ┌─── STAGE 7 · ATTRIBUTION ★★★ ──┐
│  seed from observed mask   │  │  AIS prefilter (214 → ~20)     │
│  forward +6 / +12 / +24 h  │  │  each vessel track = a         │
│  OUT: forecast envelopes   │  │    GENERATIVE HYPOTHESIS       │
└────────────────────────────┘  │  one forward run, all vessels  │
                                │  + DARK-VESSEL hypothesis      │
                                │  x behavioural prior           │
                                │  OUT: ranked candidates        │
                                └────────────┬───────────────────┘
                                             ▼
┌─── PRESENTATION ────────────────────────────────────────────────┐
│  FastAPI (127.0.0.1) serving out/*.json                         │
│  Single HTML page · deck.gl from CDN · MapLibre basemap         │
│  Evidence bundle export (Tier B)                                │
└─────────────────────────────────────────────────────────────────┘
```

`★` = differentiator we implement.

---

## 3. The four frozen contracts

**These are frozen on Day 2 and do not change.** With them, six people build against stubs in parallel. Without them, integration happens in week 12 or never.

Each lives as a Pydantic model in `src/contracts/` and is the *only* interface between stages.

### 3.1 `ForcingBundle` — the offline data guarantee

The contract that makes the demo possible. Nothing downstream is allowed to fetch data.

```json
{
  "case_id": "kattegat_2024_03_11",
  "aoi": { "min_lon": 10.2, "min_lat": 56.4, "max_lon": 12.8, "max_lat": 58.1 },
  "t_start": "2024-03-09T00:00:00Z",
  "t_obs":   "2024-03-11T05:42:13Z",
  "currents": {
    "path": "forcing/currents.nc",
    "product": "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
    "vars": ["uo", "vo"],
    "includes_tides": true,
    "includes_stokes": true,
    "resolution_deg": 0.0833,
    "rms_error_ms": 0.12
  },
  "wind": {
    "path": "forcing/wind.nc",
    "product": "ERA5 single-levels, 10m u/v",
    "vars": ["u10", "v10"],
    "resolution_deg": 0.25
  },
  "land_mask": "forcing/land.tif",
  "crs_working": "EPSG:32632"
}
```

> **`includes_stokes: true` is not decoration.** It is the flag the transport kernel reads to decide whether to add a Stokes parameterisation. CMEMS SMOC already merges Stokes drift into `uo`/`vo`; adding it again silently doubles a real physical term. This flag is how we prevent that bug by construction rather than by remembering.

`crs_working` is the local UTM zone. All integration happens in metres in this projection, never in degrees.

### 3.2 `SlickDetection` — Stage 1–3 output

```json
{
  "detection_id": "kattegat_2024_03_11:d02",
  "classification": "oil",
  "classification_reason": "wind 6.4 m/s within detectability window; damping -8.1 dB; elongation 7.3",
  "abstained": false,
  "confidence_segmentation": 0.81,
  "polygon_wgs84": [[11.42, 57.10], "..."],
  "geometry": {
    "area_km2": 18.4,
    "perimeter_km": 41.2,
    "centroid": [11.44, 57.12],
    "elongation": 7.3,
    "orientation_deg": 62.5,
    "n_components": 2,
    "complexity": 3.1,
    "solidity": 0.62
  },
  "radiometry": {
    "damping_ratio_db": -8.1,
    "damping_norm_incidence": -7.6,
    "sigma0_mean_db": -21.4,
    "sigma0_background_db": -13.3,
    "incidence_deg": 38.2
  },
  "environment": {
    "wind_speed_ms": 6.4,
    "wind_dir_deg": 245.0,
    "wind_gradient_ms_per_km": 0.03
  },
  "release_mode": "continuous",
  "release_mode_justification": "elongation 7.3 with major axis 62.5° aligned within 8° of nearby track headings"
}
```

`abstained` and `classification_reason` are mandatory. **The UI is required to render the reason.** Showing what we correctly refuse to flag is the strongest demo beat we have, and it only works if the reason survives to the frontend.

### 3.3 `SourcePosterior` — Stage 5 output

```json
{
  "case_id": "kattegat_2024_03_11",
  "detection_id": "kattegat_2024_03_11:d02",
  "grid": {
    "lon": [], "lat": [], "t0": [],
    "density_path": "out/posterior.npz",
    "normalised": true
  },
  "credible_regions": {
    "50": { "polygon_wgs84": [], "area_km2": 980 },
    "95": { "polygon_wgs84": [], "area_km2": 4210 }
  },
  "t0_marginal": {
    "bins_utc": [], "density": [],
    "hpd_95": ["2024-03-10T02:00:00Z", "2024-03-10T16:00:00Z"],
    "width_hours": 14
  },
  "lookback_hours": 36,
  "within_operating_envelope": true,
  "n_hypotheses": 4096,
  "n_particles": 100000,
  "effective_sample_size": 7412,
  "assumptions": [
    "windage alpha ~ U(0.01, 0.04)",
    "horizontal diffusivity K_h ~ LogU(1, 10) m^2/s",
    "current field perturbed with correlated noise, sigma = 0.12 m/s (CMEMS QUID)",
    "CMEMS 1/12 deg cannot resolve sub-mesoscale structure below ~9 km"
  ]
}
```

`effective_sample_size` is a self-check: if it collapses, the posterior is Monte Carlo noise and the UI must say so rather than draw a confident-looking blob.

`assumptions` is carried all the way to the UI and the evidence export. Every probability we display is accompanied by the list of things that would have to be true for it to mean anything.

### 3.4 `RankedCandidates` — Stage 7 output

```json
{
  "case_id": "kattegat_2024_03_11",
  "traffic_reduction": { "vessels_in_window": 214, "after_prefilter": 19, "reported": 3 },
  "candidates": [
    {
      "rank": 1,
      "mmsi": "219000000",
      "vessel_name": "REDACTED_IN_DEMO",
      "vessel_type": "Tanker",
      "posterior_probability": 0.41,
      "log_likelihood": -142.6,
      "best_discharge_window_utc": ["2024-03-10T04:10:00Z", "2024-03-10T07:35:00Z"],
      "evidence": {
        "track_overlap_likelihood": -142.6,
        "ais_gap_factor":      { "value": 2.1, "note": "3.2 h gap vs 0.4 h local baseline" },
        "speed_anomaly_factor":{ "value": 1.6, "note": "4.1 kn vs 12.3 kn transit median" },
        "course_change_factor":{ "value": 1.0, "note": "no significant alteration" },
        "vessel_type_factor":  { "value": 1.4, "note": "tanker prior" },
        "axis_alignment_factor":{ "value": 2.8, "note": "slick axis within 8 deg of heading" }
      },
      "track_geojson": {}
    }
  ],
  "dark_vessel_hypothesis": {
    "posterior_probability": 0.22,
    "prior_used": 0.15,
    "note": "posterior mass not explained by any AIS-observed track"
  },
  "disclaimer": "Investigative leads under stated model assumptions. Not a determination of responsibility."
}
```

> The `dark_vessel_hypothesis` block is **mandatory and non-removable**. If the true polluter had AIS off, it is not in the candidate set at all, and a system that normalises only over observed vessels will confidently name an innocent ship. Carrying this term is what makes the other numbers meaningful.

---

## 4. Module layout and ownership

| Module | Contents | Owner | Depends on |
|---|---|---|---|
| `src/contracts/` | The four Pydantic models above. **Written first, Day 2** | M6 | — |
| `src/ingest/` | CDSE / CMEMS / CDS / DMA fetchers, Case builder, synthetic AIS generator | M3 | contracts |
| `src/detect/` | Tiling, U-Net inference, physics gate, LightGBM, characterisation | M1 | contracts |
| `src/transport/` | Forcing readers, interpolators, the Lagrangian kernel | M2 | contracts |
| `src/inversion/` | Proposal pass, hypothesis grid, observation operator, posterior + KDE | M2 | transport, detect |
| `src/attribution/` | AIS cleaning, track reconstruction, vessel hypotheses, dark term, priors, ranking | M4 | transport, inversion |
| `src/eval/` | Synthetic truth generator, calibration, metrics | M4 + M2 | everything |
| `src/api/` | FastAPI, static serving, case listing | M6 | contracts |
| `web/` | `index.html`, deck.gl layers, panels | M5 | API fixtures |
| `scripts/` | `run_case.py`, `build_case.py`, `build_synthetic_truth.py`, `train_unet.py` | M6 | all |

**Fixture-first rule:** before any stage is implemented, its owner commits a hand-written example of its output contract to `tests/fixtures/`. M5 builds the entire UI against fixtures and never blocks on the pipeline.

---

## 5. Tech stack and the reasoning

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| SAR preprocessing | **Skipped in prototype**; SNAP / pyroSAR in R3 | Zenodo data is already calibrated sigma-0 dB. SNAP is Java-backed and the highest-probability install failure in the project. It buys zero innovation credit |
| Segmentation | **PyTorch + `segmentation_models_pytorch`**, U-Net / ResNet34 | Not YOLO (slicks are amorphous, bounding boxes are meaningless). Not SegFormer (transformers are data-hungry; ~2k labelled scenes will underperform and we cannot afford the ablation to prove otherwise) |
| Look-alike classifier | **LightGBM** on ~14 features | With this few features GBDT beats a neural net, trains in seconds, and gives feature importances you can put on a slide. Explainability is a scored asset here |
| Ocean / wind | **CMEMS SMOC + ERA5**, pre-cached NetCDF | Building an ocean model is a PhD, not a hackathon |
| Drift solver | **Custom ~150-line Lagrangian kernel** (R3: OpenDrift / OpenOil) | See §6 — this is the biggest scoping decision in the project |
| Numerics | NumPy + SciPy, float32, fully vectorised | 1e5 particles x 192 steps is a few seconds vectorised, minutes in a Python loop |
| Geospatial | rasterio, shapely, geopandas, pyproj | Standard. All area computation in projected metres, never degrees |
| AIS store | **Parquet + DuckDB** | No server. Columnar, fast over CSV. PostGIS only if we needed concurrency — we do not |
| Backend | **FastAPI**, bound to `127.0.0.1` | Async, auto OpenAPI docs, trivial static serving |
| Frontend | **Single HTML page + deck.gl from CDN + MapLibre GL** | No npm, no Vite, no build step, no `node_modules` on demo day. Identical GPU rendering to a React build |
| Packaging | `requirements.txt` with pinned versions; Docker optional | A venv on the demo laptop is the primary path. Docker is the backup, not the plan |

### Why deck.gl and not Leaflet

Not a style preference. Animating 1e5 particles is the visual core of the demo. Leaflet renders DOM/canvas markers and will collapse well before 1e4. deck.gl's `ScatterplotLayer` and `TripsLayer` are WebGL and hold 60 fps at this scale. `HeatmapLayer` renders the posterior directly.

---

## 6. The drift-solver decision, recorded

**Decision:** the prototype uses a custom Lagrangian kernel, not OpenDrift.

| | Custom kernel | OpenDrift / OpenOil |
|---|---|---|
| Install risk on Windows | None (NumPy + xarray) | High — conda, GDAL, likely WSL2 or Docker |
| Weathering | ✗ none | ✓ NOAA ADIOS, peer-reviewed |
| Citable in front of judges | Weak | Strong (MET Norway) |
| Control over per-particle origin tagging | Total | Via `origin_marker`, workable |
| Time to first result | ~half a day | 1–3 days, possibly never on this platform |

**Rationale.** The differentiator is the *inversion method* — hypothesis-level Bayesian conditioning on forward ensembles — not the ODE solver underneath it. The solver is interchangeable, and the method is provably identical whichever we use. With 14 days and a Windows fleet, a solver that might not install is an unacceptable dependency for the critical path.

**The honest sentence we say out loud:** *"The prototype uses a minimal Lagrangian transport kernel — RK4 advection, sampled windage, random-walk diffusion. It has no weathering module. Round 3 substitutes OpenOil for validated weathering and parameterisations; the inversion above it does not change."*

**Reversibility:** the kernel sits behind a single interface, `transport.simulate(seeds, t_start, t_end, forcing, params) -> trajectories`. Swapping in OpenDrift is one adapter class. This is recorded in [engineering_review.md](engineering_review.md) as deliberate, time-boxed debt.

---

## 7. Data flow and execution modes

| Mode | Command | Purpose |
|---|---|---|
| **Build** | `scripts/build_case.py` | Online. Fetches everything, writes a Case folder. Run once, days before the demo |
| **Run** | `scripts/run_case.py --case <path>` | Offline. Executes stages 1–7, writes `out/` |
| **Serve** | `uvicorn src.api.main:app` | Offline. Serves `out/` to the browser |
| **Fixture** | `uvicorn ... --fixtures` | Serves `tests/fixtures/` instead of real output. Lets M5 work from Day 2 |
| **Eval** | `scripts/build_synthetic_truth.py --n 40` | Generates synthetic cases with known ground truth, runs the pipeline, emits calibration and top-K metrics |

---

## 8. Cross-cutting conventions

Small, boring, and each one is a bug that will otherwise cost half a day.

| Concern | Convention |
|---|---|
| **Time** | UTC everywhere, ISO 8601 with explicit `Z`, `numpy.datetime64[s]` internally. AIS timestamps from DMA are UTC — assert it on load |
| **CRS** | Storage and contracts in EPSG:4326. **All computation** in the case's local UTM zone (`crs_working`). Areas in km² are only ever computed in projected metres |
| **Wind convention** | Meteorological "from" vs oceanographic "to" is a classic sign error. We store `u10`, `v10` components only, never a bearing, and derive direction once at the UI boundary |
| **Stokes** | Read `ForcingBundle.currents.includes_stokes`. Never hard-code |
| **Randomness** | One seed per case in `case.json`, threaded to every RNG. NFR-3 depends on it |
| **Units** | Metres, seconds, m/s internally. Conversion to km / knots / hours happens only at the UI boundary |
| **Land** | Particles that intersect land are marked `beached` and frozen, never deleted — deleting them silently biases the posterior away from the coast |
| **Logging** | Structured JSON per stage into `out/log.jsonl`, including timings. This is the raw material for [performance_review.md](performance_review.md) |

---

## 9. What this architecture explicitly refuses

- **No microservices.** One process, one pipeline, one machine.
- **No database server.** Parquet files and DuckDB over them.
- **No message queue.** Stages are functions called in order. Celery/Redis was considered and rejected — inversion takes 30 s, not 30 min.
- **No Kubernetes, no cloud deployment.** It adds risk and zero marks.
- **No live API calls at demo time.** Structurally prevented by the Case design.
- **No authentication layer.** The API binds to `127.0.0.1` and is never exposed. See [security_review.md](security_review.md).
