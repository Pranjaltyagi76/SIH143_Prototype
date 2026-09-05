# Performance Review

**Status:** 🔧 Framework and budgets set before coding; measurements filled in after the application works (Day 11–12) · **Owner:** M2 + M6 · **Last updated:** 2026-09-05

---

## 0. Why performance matters here at all

It is a 14-day prototype on one laptop, so most performance work would be waste. Exactly two things make it worth a document:

1. **NFR-2: end-to-end in under 60 s.** Not for throughput — for *iteration speed*. A pipeline that takes 10 minutes gets run five times a day; one that takes 40 seconds gets run fifty times, and the team finds five times as many bugs.
2. **The inversion has one genuine algorithmic cliff** (§3.1). Implemented naively it is 4 × 10⁹ operations per case; implemented correctly it is closed-form. Getting this right on Day 5 is the difference between an interactive demo and a slideshow.

Everything else is left alone deliberately.

---

## 1. Budget

Target: **< 60 s** per case, cached forcing, offline.

| Stage | Budget | Basis |
|---|---|---|
| Load case + forcing into memory | 3 s | ~500 MB NetCDF → float32 arrays |
| U-Net inference (GPU) | 3 s | ~50 tiles at 256² |
| U-Net inference (CPU fallback) | 15 s | Demo machine may lack the training GPU |
| Physics gate + LightGBM + characterisation | 1 s | 14 features over ~10 patches |
| Backward proposal (1e4 particles, advection only) | 2 s | |
| **Forward ensemble (4e5 particles × 192 steps)** | **20 s** | The dominant cost — see §2 |
| **Likelihood evaluation (4,096 hypotheses)** | **2 s** | Only if §3.1 is implemented correctly. **40+ s if not** |
| KDE + credible regions | 3 s | |
| AIS load + prefilter + track reconstruction | 3 s | DuckDB over parquet |
| Vessel-conditioned run (~20 vessels + dark, one simulation) | 8 s | Reuses the same kernel |
| Ranking + evidence assembly | 1 s | |
| Serialisation of `out/` | 2 s | Particles downsampled to 5k trails for the UI |
| **Total** | **~48 s** | ~20% headroom |

**Frontend budget (separate):** first paint < 2 s · particle animation ≥ 30 fps at 1e4 rendered trails · case switch < 1 s.

---

## 2. The transport kernel — where the time actually goes

Cost model per timestep:

```
4 RK4 stages x 2 current components x N particles      (bilinear space + linear time)
+ 1 wind lookup x 2 components x N particles
+ 1 RNG draw x 2 dims x N particles
```

At N = 4×10⁵ that is ~4×10⁶ interpolated lookups per step, ~7.7×10⁸ over 192 steps.

**The only optimisation that matters is vectorisation.** All N particles advance as one NumPy array operation per step. A per-particle Python loop is roughly 100× slower and is the single most likely reason this prototype would feel broken.

| Decision | Effect |
|---|---|
| Fully vectorised NumPy, no particle loop | ~100× vs the naive version |
| **float32 throughout** | ~2× memory bandwidth; precision is irrelevant against 9 km grid error |
| Forcing preloaded into one contiguous in-memory array | Eliminates repeated NetCDF/HDF5 reads inside the loop — easily 10× on its own |
| Hand-written bilinear indexing rather than `scipy.map_coordinates` | Avoids per-call overhead at 192 calls |
| Wind interpolated once per step, not per RK4 stage | Wind varies hourly; sub-step interpolation is wasted work. 4× saving on the wind path |
| Beached particles masked out of the update | Cheap, and grows as the run proceeds |

**Not done, deliberately:** Numba, Cython, multiprocessing, GPU. Each adds an install dependency or a debugging burden to save time we do not need. Recorded so the choice is visible rather than accidental.

---

## 3. The one algorithmic cliff

### 3.1 Likelihood evaluation — naive is O(hypotheses × pixels)

The observation operator ([technical_design.md](technical_design.md) §4.3) is

```
log L(h) = SUM over ALL AOI pixels [ m(x) log q_h(x) + (1 − m(x)) log(1 − q_h(x)) ]
```

Implemented literally, that is 4,096 hypotheses × 10⁶ pixels = **4 × 10⁹ operations per case.** Tens of seconds at best, and it would dominate everything.

**It does not need to be computed that way.** With `q_h(x) = 1 − exp(−λ ρ_h(x))`, the negative term collapses:

```
log(1 − q_h(x)) = −λ ρ_h(x)
```

so

```
SUM over x NOT in mask  log(1 − q_h(x))  =  −λ * SUM over x NOT in mask  rho_h(x)
                                          =  −λ * (particle mass landing outside the mask)
```

which is **a closed form: just a count.** The remaining positive term runs only over mask pixels, a small set. The whole evaluation becomes:

```
log L(h) = SUM over MASK pixels [ log(1 − exp(−λ rho_h)) ]  −  lambda * (mass outside mask)
```

**Cost: O(mask pixels + particles) per hypothesis** instead of O(all pixels). Roughly 2 s instead of 40 s, with **no approximation** — it is algebraically identical.

It also makes the physics legible: the likelihood is *reward for covering the observed slick, minus a linear penalty for every particle predicted where the satellite saw nothing.* That is exactly the property naive rejection ABC lacks, and now it is one subtraction.

> **This derivation is worth a slide.** It is the kind of thing that separates a team that implemented a method from a team that understood it.

### 3.2 Fallback if it is still slow

In order: coarsen the likelihood grid to ~1 km cells · reduce hypotheses via sequential refinement (two coarse-to-fine rounds beat one fine pass) · cut the lookback window and *state the assumption* · precompute demo cases and serve from cache.

---

## 4. Other predicted hotspots

| Suspect | Prediction | Pre-planned mitigation |
|---|---|---|
| NetCDF reads inside the particle loop | 🔴 Would dominate everything | Preload to memory. Architectural, not an optimisation |
| KDE over a 3-D posterior grid | 🟠 Possibly slow at fine resolution | Use a separable Gaussian filter on the histogram rather than `gaussian_kde` — orders of magnitude faster and adequate |
| AIS load (a region-month of CSV) | 🟠 Slow once | Convert to parquet at build time. Never parse CSV in the run path |
| Shapely polygon ops on complex slicks | 🟡 | Simplify to ~1 m tolerance before any set operation |
| Particle serialisation to JSON | 🟡 4e5 particles is a large JSON | Downsample to 5k trails for the UI. The full set stays in `.npz` |
| deck.gl re-render on data change | 🟡 | Use layer `updateTriggers`, do not recreate layers |
| U-Net on CPU | 🟠 If the demo machine has no GPU | Measure early. Batch tiles; fall back to cached detections |

---

## 5. Measurement method

`out/log.jsonl` receives a structured record per stage — name, wall time, particle count, hypothesis count, peak RSS. Written from Day 2 so the data accumulates for free across the whole sprint.

```bash
python scripts/run_case.py --case data/cases/kattegat_2024_03_11 --profile
```

`--profile` adds `cProfile` around each stage and writes `out/profile_<stage>.prof`. Inspect with `snakeviz`. Use `memory_profiler` only if RSS approaches 12 GB.

**Rule: profile before optimising.** The predictions in §4 are predictions. Every optimisation in the log below must cite a measurement.

---

## 6. Frontend performance

| Concern | Approach |
|---|---|
| 1e4+ animated particles | deck.gl `TripsLayer` — WebGL, not DOM. Verified on **Day 2** with 1e5 random points before anything depends on it |
| Posterior heatmap | Pre-rasterised PNG or `BitmapLayer`, not `HeatmapLayer` recomputing per frame |
| AIS tracks | `PathLayer`, simplified server-side |
| SAR raster | Pre-tiled PNG pyramid, not a full-resolution GeoTIFF in the browser |
| Case switching | Prefetch the other two cases' JSON on idle |

Day 2 is deliberately early for the rendering-budget check. Discovering on Day 12 that the visual centrepiece cannot hold frame rate would be unrecoverable.

---

## 7. What we are not optimising, and why

Multi-case throughput (we run three) · concurrent users (there is one) · cold-start time beyond NFR-7 · memory below 16 GB (the target machine has it) · model inference latency past the budget (it is not the bottleneck) · database query planning (parquet scans of one region-month are already fast).

---

## 8. Measurements

*Filled in Day 11–12, after the pipeline works end to end. Empty cells are honest.*

| Stage | Budget | Measured | Machine | Notes |
|---|---|---|---|---|
| Load case + forcing | 3 s | | | |
| U-Net inference (GPU) | 3 s | | | |
| U-Net inference (CPU) | 15 s | | | |
| Gate + characterisation | 1 s | | | |
| Backward proposal | 2 s | | | |
| Forward ensemble | 20 s | | | |
| Likelihood evaluation | 2 s | | | |
| KDE + credible regions | 3 s | | | |
| AIS + tracks | 3 s | | | |
| Vessel-conditioned run | 8 s | | | |
| Ranking | 1 s | | | |
| Serialisation | 2 s | | | |
| **Total** | **48 s** | | | |

## 9. Optimisation log

*Each entry must cite the measurement that motivated it.*

| Date | Hotspot | Measured before | Change | Measured after |
|---|---|---|---|---|
| | | | | |
