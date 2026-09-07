# Phasewise Roadmap

**Status:** Frozen before coding · **Owner:** M6 · **Last updated:** 2026-09-05

**Sprint window:** Sun 6 Sep → Sat 19 Sep 2026 (14 days) · **Round 1 demo:** 19 Sep · **SIH idea submission:** 20 Sep

---

## 0. The framing that decides everything

You are **not** building the system described in [technical_design.md](technical_design.md). You are building a **thin vertical slice that runs end to end**, plus the documents that prove you know what the full system is.

> **One case running badly, end to end, beats four components running beautifully in isolation.**

Every prioritisation call in this sprint resolves in favour of "the demo runs". A component at 60% that is wired in beats a component at 95% that is not.

### Tiering

| Tier | Meaning | Rule |
|---|---|---|
| **A** | The demo does not exist without it | Build first, in dependency order |
| **B** | Makes the demo materially better | Only after all of Tier A is wired end to end |
| **C** | Round 3 | Do not touch. Write it in the docs instead |

---

## 1. Team and ownership

| | Role | Owns | Tier A deliverable |
|---|---|---|---|
| **M1** | SAR / ML | Tiling, U-Net training + inference, physics gate, LightGBM, characterisation | `SlickDetection` for 3 cases |
| **M2** | Physics / inversion ★ | Transport kernel, proposal pass, hypothesis grid, observation operator, posterior | `SourcePosterior` with credible regions |
| **M3** | Data engineering | CDSE / CMEMS / CDS / DMA ingest, Case builder, synthetic AIS generator, caching | 3 complete Case folders |
| **M4** | Attribution ★ | AIS cleaning, track reconstruction, vessel hypotheses, dark term, priors, ranking, **synthetic truth harness** | `RankedCandidates` + calibration numbers |
| **M5** | Frontend | Single-page deck.gl UI, all panels, animation | The interface the PS requires |
| **M6** | Integration + narrative | Contracts, FastAPI, `run_case.py`, **the slides, the video, the submission** | Submitted PDF + video |

**M6 is not a spare part.** The documented failure mode of this team is the submission, not the code. M6 owns slides from Day 1 and has standing authority to demand a screenshot from anyone at any time.

---

## 2. The critical path

```
M3: Case #1 exists  ──┬──► M1: detection  ──┐
   (Day 3, HARD GATE) │                     ├──► M6: integration ──► M5: UI ──► DEMO
                      └──► M2: transport ───┴──► M2: inversion ──► M4: attribution
```

**The long pole is M2's inversion.** It must start on Day 2 against a **hand-drawn mask**, not wait for M1's model. This is the single most important scheduling instruction in the document — the two most common ways this sprint fails are (a) M2 waiting on M1, and (b) M1 chasing IoU past the benchmark.

---

## 3. Phase 1 · Foundation (Days 1–3, Sun 6 – Tue 8 Sep)

### Day 1 — Sun 6 Sep · Access and accounts

Everything here has external latency. It all happens on Day 1 or it blocks the sprint.

| Who | Task | Done when |
|---|---|---|
| M3 | Register Copernicus Data Space, CMEMS, CDS. **Accept the ERA5 licence** | Three credentials in `.env`, one test download each |
| M3 | Pull one Danish DMA AIS day, inspect columns, confirm UTC | A CSV on disk, columns documented |
| M1 | Download Zenodo SAR oil spill Parts I–III | Data on disk, one tile rendered |
| M1 | **Email a faculty supervisor to request the Krestenitis benchmark** | Email sent. Long lead time — today or never |
| M2 | Environment: numpy, scipy, xarray, netCDF4, rasterio, pyproj | `import` smoke test passes |
| M4 | Environment: pandas, duckdb, geopandas, shapely | Same |
| M5 | Blank HTML page with deck.gl from CDN + MapLibre basemap rendering | A map appears in a browser |
| M6 | Repo skeleton, `.gitignore` (secrets, data), `requirements.txt` pinned | Everyone can clone and run |
| M6 | **Verify Sentinel-1 constellation status against ESA** | Written answer in the log. Do not put 1C/1D on a slide until this is checked |

### Day 2 — Mon 7 Sep · Contracts frozen 🔴

| Who | Task |
|---|---|
| **M6** | **Write all four Pydantic contracts and commit hand-written fixtures for each.** Nothing else on Day 2 matters as much |
| M6 | `scripts/run_case.py` skeleton — stages as no-op functions that read and write contracts |
| M3 | Download CMEMS SMOC + ERA5 for the Kattegat AOI. **Read the SMOC QUID and confirm whether Stokes is included** — record the answer in `ForcingBundle` |
| M1 | Tiling + dataloader; first U-Net epoch running on the GPU |
| M2 | Transport kernel: RK4 advection over the cached NetCDF, forward only, no diffusion |
| M4 | AIS cleaning: duplicate MMSI, speed-jump filter, gap detection. **Count everything dropped** |
| M5 | deck.gl `ScatterplotLayer` animating 1e5 random points at 60 fps — prove the rendering budget now, not on Day 12 |

> 🔴 **Contracts frozen end of Day 2. They do not change after this.** Every subsequent day assumes all six people are building against stubs in parallel.

### Day 3 — Tue 8 Sep · 🔴 HARD GATE: Case #1 exists

| Who | Task |
|---|---|
| M3 | **Assemble Case #1 (Kattegat) completely:** scene, forcing, AIS, `case.json`. Commit the folder layout |
| M2 | Diffusion + per-particle windage sampling. **Backward advection-only pass working.** Assert that backward + diffusion raises an error |
| M2 | **Hand-draw a slick mask on Case #1 and start the inversion against it.** Do not wait for M1 |
| M1 | U-Net training run underway; geographic holdout split defined |
| M4 | Track reconstruction + spatio-temporal prefilter against a stub posterior |
| M5 | Map + SAR raster overlay + polygon layer, driven by fixtures |
| M6 | First six slides drafted from [requirements.md](requirements.md), with gaps marked `[SCREENSHOT]` |

**Gate criteria — end of Day 3.** All four must be true:

- [ ] Case #1 folder is complete and loads offline
- [ ] Transport kernel runs forward and backward on real forcing
- [ ] A U-Net epoch completes on the GPU
- [ ] Contracts are frozen and fixtures exist

❌ **If the gate fails:** cut to a single case, drop the Arabian Sea case entirely, and drop Tier B for the whole sprint. Do not push the gate right.

---

## 4. Phase 2 · Components (Days 4–7, Wed 9 – Sat 12 Sep)

Four tracks in parallel. Nobody blocks anybody, because contracts are frozen and fixtures exist.

| Day | M1 detection | M2 inversion ★ | M3 data | M4 attribution ★ | M5 UI | M6 integration |
|---|---|---|---|---|---|---|
| **4** Wed | U-Net converging; IoU on holdout | Hypothesis grid + rasterisation `rho_h` | Case #2 (2nd Danish scene, contains a **look-alike**) | Vessel-conditioned seeding along AIS tracks | Segmentation overlay + classification colouring | FastAPI serving `out/` |
| **5** Thu | Physics gate: ERA5 sampling + hard override | **Bernoulli observation operator** (§4.3) | Case #3 (Arabian Sea, synthetic AIS) | `origin_marker` tagging; one run, all vessels | Particle animation from fixtures | `run_case.py` chains stages 1–3 |
| **6** Fri | LightGBM on 14 features; feature importances plotted | Posterior + KDE + credible regions in km² | Land mask; AIS density raster for the lane feature | Dark-vessel hypothesis + likelihood | Posterior heatmap + credible-region outlines | `run_case.py` chains stages 4–5 |
| **7** Sat | Characterisation + release-mode inference | 2-round sequential refinement; `effective_sample_size` | Cases frozen. **No new data after today** | Behavioural priors + ranking + evidence blocks | Candidate panel + evidence breakdown | `run_case.py` chains 6–7. **End-to-end skeleton** |

🔴 **Day 7 gate: `run_case.py` produces all four contract objects for Case #1**, even if the numbers are poor. Quality comes next week; wiring comes now.

**Standing risks this phase:**

| Risk | Trigger | Response |
|---|---|---|
| M1 chases IoU | IoU stuck below 0.5 on Day 6 | **0.55 is the published benchmark. Hit it and stop.** The gate matters more than three more IoU points |
| Inversion posterior is uniform / uninformative | Day 6 | Shrink lookback to 24 h, coarsen the hypothesis grid, verify the observation operator on a synthetic case with a known source before blaming the physics |
| Rejection efficiency collapses | Day 5–6 | Sequential refinement (§4.4). If still bad, reduce the prior volume and *state the assumption* |
| AIS volume | Day 4 | Subset to one region-month before optimising anything |

---

## 5. Phase 3 · Integration and evaluation (Days 8–11, Sun 13 – Wed 16 Sep)

### Day 8 — Sun 13 Sep · First true end-to-end run

Everyone stops feature work. One case, real model output, real posterior, real ranking, real UI. **Expect it to be broken and ugly.** Log every failure into [problems_faced_and_bugs_encountered.md](problems_faced_and_bugs_encountered.md) — that log is Round 2 material, not overhead.

### Day 9 — Mon 14 Sep · Synthetic truth harness 🔴

**M4 + M2, highest priority day of the sprint after the Day 3 gate.**

`scripts/build_synthetic_truth.py`: take a real DMA AIS track, seed a synthetic release along it, forward-simulate with real forcing, threshold to a synthetic mask, then run the pipeline from that mask and check whether the true vessel and the true (x0, t0) are recovered. Run 30–40 times.

This is the **only** source of numbers for inversion and attribution. Without it there is no calibration curve, no top-K recall, and no headline reduction factor — which means the three strongest slides have nothing on them.

### Day 10 — Tue 15 Sep · Numbers

| Who | Deliverable |
|---|---|
| M4 + M2 | Calibration curve (nominal vs empirical coverage), top-1 / top-3 recall, candidate reduction factor |
| M1 | Final metrics table: IoU in-domain, IoU cross-region, **the drop**, gate FPR, abstention rate |
| M2 | 95% region area at 12 / 24 / 48 h lookback — the operating-envelope plot |
| M6 | Every number onto a slide **as measured**. Bad numbers go on the slide too |

> **If the numbers are bad, report them anyway.** Honest weak numbers beat absent numbers, and the calibration curve is the point — not the accuracy.

### Day 11 — Wed 16 Sep · Cases 2 and 3 + Tier B

All three cases run end to end. **Case #2 must contain the look-alike that gets correctly rejected** — that is the demo's strongest beat and it needs to be real, not staged.

Tier B, strictly in this order, and only if Tier A is green:
1. CFAR ship detection + AIS matching (dark-vessel visual)
2. Evidence bundle export
3. Forecast envelopes at +6 / +12 / +24 h

---

## 6. Phase 4 · Hardening and rehearsal (Days 12–14, Thu 17 – Sat 19 Sep)

### Day 12 — Thu 17 Sep · Offline hardening 🔴

| Who | Task |
|---|---|
| M6 | **Disable the network adapter and run all three cases.** Any failure is a P0 bug (NFR-1) |
| M6 | Cache every `out/` folder. The demo must be able to run from cache if live execution breaks |
| M5 | UI polish, loading states, and the **assumptions panel** — every probability displayed next to what would have to be true for it to mean anything |
| All | **Overclaiming audit:** read every screen aloud. Ban "87% confident", "identified the vessel", "guilty", "responsible". Replace with "candidate", "investigative lead", "under stated model assumptions" |

### Day 13 — Fri 18 Sep · Video and slides

M6 records the demo video. Everyone else fixes only what the recording exposes. **No new features.**

### Day 14 — Sat 19 Sep · Rehearsal and freeze

- Three full run-throughs on the demo machine, network off, from a cold boot
- Two people rehearse the judge Q&A (see the twenty prepared questions in the source design doc)
- **Code freeze at 18:00.** Submission package assembled and checked
- Backup: full repo + cached outputs on two USB drives and one cloud folder

**Sun 20 Sep:** SIH idea submission uploaded. **The PS cannot be changed after submission.**

---

## 7. Daily rhythm

| When | What | Duration |
|---|---|---|
| 10:00 | Standup: yesterday / today / blocked | 10 min, hard stop |
| 18:00 | Integration check — does `run_case.py` still work? | 15 min |
| 22:00 | M6 updates the roadmap status table and the problems log | — |

**Blocked-for-two-hours rule:** anyone blocked for more than two hours says so in the channel immediately. Silent blocking is what turns a one-day slip into a four-day slip.

---

## 8. Cut list, pre-agreed

When the schedule slips — and it will — cut from the bottom. **This order is agreed now, while nobody is panicking, so it is not renegotiated at 2 a.m. on Day 12.**

1. Forecast envelopes (+6/+12/+24 h)
2. Evidence bundle export
3. CFAR ship detection
4. Case #3 (Arabian Sea) — keep the *slide*, cut the running case
5. Case #2 → but **never** cut the look-alike rejection beat; move it into Case #1 if you must
6. Sequential refinement in the inversion — accept a coarser posterior and say so
7. Release-mode inference
8. Trained U-Net → hand-corrected masks for demo cases, **disclosed on the slide**

**Never cut:** the physics gate, the forward-ensemble inversion, the dark-vessel hypothesis, the calibration number, or the submission.

---

## 9. Status tracker

Update at 22:00 daily. `⬜` not started · `🟡` in progress · `✅` done · `🔴` blocked

| Day | Date | Milestone | Status |
|---|---|---|---|
| 1 | Sun 6 Sep | Accounts, data, Krestenitis request sent | ⬜ **not started — external latency, do today** |
| 2 | Mon 7 Sep | 🔴 **Contracts frozen** | ✅ **DONE (early, 6 Sep)** — see below |
| 3 | Tue 8 Sep | 🔴 **GATE: Case #1 exists, kernel runs, epoch completes** | ⬜ |
| 4 | Wed 9 Sep | Hypothesis grid; U-Net converging | ⬜ |
| 5 | Thu 10 Sep | Observation operator; physics gate | ⬜ |
| 6 | Fri 11 Sep | Posterior + credible regions; LightGBM | ⬜ |
| 7 | Sat 12 Sep | 🔴 **GATE: all four contracts produced for Case #1** | ⬜ |
| 8 | Sun 13 Sep | First true end-to-end run | ⬜ |
| 9 | Mon 14 Sep | 🔴 **Synthetic truth harness** | ⬜ |
| 10 | Tue 15 Sep | All metrics measured and on slides | ⬜ |
| 11 | Wed 16 Sep | Three cases running; Tier B | ⬜ |
| 12 | Thu 17 Sep | 🔴 **Offline hardening + overclaiming audit** | ⬜ |
| 13 | Fri 18 Sep | Video recorded | ⬜ |
| 14 | Sat 19 Sep | Rehearsal, freeze, backups | ⬜ |
| — | Sun 20 Sep | **SIH idea submission uploaded** | ⬜ |

### Phase 0 completion note — 6 Sep 2026

Delivered ahead of the Day 2 gate:

- **Four contracts frozen** in `src/contracts/`, plus the `CaseManifest` spine. `extra="forbid"` on every model, so contract drift fails loudly at the boundary.
- **Design decisions enforced in code, not convention** — `includes_stokes` is required with no default (P-05); the wind gate is a contract-level override the classifier cannot beat (FR-3a); `dark_vessel_hypothesis` is required with no default (SC-1); probabilities including the dark term must sum to 1; accusatory language is rejected at the field level (SC-2); lookback past 72 h must set `within_operating_envelope=False` (FR-10a).
- **Reference fixtures** for all five artefacts in `tests/fixtures/`, covering all three classification outcomes including the 1.4 m/s abstention that is the demo's strongest beat. The frontend can now be built without the pipeline.
- **`scripts/run_case.py`** with real stage ordering, skip/resume, `--force`, `--stage`, `--seed`, `--fixtures`, and the `out/log.jsonl` timing log. All five stages are honest stubs that report loudly.
- **56 tests passing.** Every guard has a test that constructs the invalid input and asserts rejection.
- **Environment resolved:** Python 3.13.14, all 20 dependencies including `torch 2.14.0+cu126`. No 3.11 needed, no conda, no Docker. Exact pins in `requirements.txt`.
- **Three bugs logged** as P-08 / P-09 / P-10 in [problems_faced_and_bugs_encountered.md](problems_faced_and_bugs_encountered.md).

Still outstanding from Day 1–2, all owned outside Phase 0: accounts and downloads (M3), the Krestenitis request (M1), CMEMS QUID Stokes confirmation (M2), and the deck.gl 1e5-point rendering budget check (M5).

### Phase 1 completion note — 6 Sep 2026

**Two complete synthetic cases exist on disk and validate against every contract.** Phases 2–4 are now unblocked with zero downloads.

- **`src/ingest/synthetic_forcing.py`** — analytic currents (sheared background + Gaussian eddy from a streamfunction + rotary M2 tide) and 10 m wind, written as CF-style NetCDF at CMEMS 1/12° and ERA5 0.25° spacing. Every term is there because the inversion needs it: shear so particles separate, an eddy so trajectories curve, and time-dependence so **t0 is identifiable at all** — in a steady flow the time marginal is flat no matter how good the inversion is.
- **`src/ingest/synthetic_scene.py`** — sigma-0 VV in dB from a CMOD-like wind proxy with multiplicative Gamma speckle, plus an incidence ramp. **The low-wind pocket produces a genuine dark patch in the scene** (−6.7 dB against surrounding water) rather than a painted-on one, so the detector has something honest to be fooled by.
- **`src/ingest/synthetic_ais.py`** — ~180 vessels, 43k–60k messages, lane-clustered, with a tight reporting baseline plus a benign minority carrying gaps, and loiterers for the speed-anomaly factor. Every MMSI is provably outside the real ship-station MID range (W-14).
- **`src/ingest/case_builder.py`** + `scripts/build_synthetic_case.py` — assembles the full Case folder in the exact layout Phase 8 will produce.

**Measured properties of the synthetic world:**

| Property | Value | Why it matters |
|---|---|---|
| Mean current speed | 0.23 m/s | ~30 km of drift over 36 h — a realistic shelf-sea regime |
| Wind, domain minimum | 1.61 m/s | Below the 3 m/s gate, so abstention is demonstrable |
| Domain below 3 m/s | **5.1%** | Lands inside the 5–15% target abstention band, unforced |
| Calm-pocket contrast | −6.7 dB | A convincing look-alike |
| AIS vessels / messages | 179 / 43,309 | Dense enough that the prefilter must earn its reduction factor |
| MMSI collisions with real range | **0** | Verified by test |
| Case size on disk | 11–18 MB | Fully offline, committable-adjacent |

**84 tests passing** (28 new). Two bugs logged: P-12 (xarray rejects tz-aware datetimes) and P-13 (the calm pocket was under-resolved by the ERA5-spaced grid — kept as a real effect rather than hidden).

> Amended during Phase 2: a zonal jet was added to the ocean config after measurement showed the original field sheared far too weakly (P-15). Mean current rose from 0.19 to 0.23 m/s.

### Phase 2 completion note — 6 Sep 2026

**The transport kernel is done and is the spine every later stage runs on.**

- **`src/transport/field.py`** — forcing resampled once onto a projected grid and held in memory. The particle loop never touches a projection, never converts degrees to metres, and never opens a NetCDF file. Velocity vectors are rotated into the grid frame by a meridian convergence **measured from the projection itself** rather than derived analytically, which removes any chance of a sign error.
- **`src/transport/kernel.py`** — RK4 advection with Euler–Maruyama diffusion by operator splitting, per-particle windage and diffusivity, per-element seed times, `origin_marker` labelling, and beaching that freezes rather than deletes.

**Measured:**

| Property | Value | Budget |
|---|---|---|
| 1e5 particles × 48 h, full physics | **9.4 s** (49 ms/step) | 30 s ✅ |
| Forcing load + resample | 1.2 s | — |
| Field memory | 12.3 MB | — |
| Round trip, advection only, windage off | **0.0 m** | exact reversibility |
| Round trip, windage fixed | 13 m | explainable |
| Shear-only separation over 48 h | 2.1–3.1× | realistic strain |
| Displacement over 48 h | 57 km mean | plausible |

**The guard works.** `simulate()` raises rather than integrating backwards with diffusion enabled, with an error message that explains why. This is the claim the whole project rests on, and it is now enforced in code rather than documented as a convention.

**113 tests passing** (29 new). Two significant findings logged: P-14 (reversibility holds only with per-particle parameters held fixed — led to `TransportParams.proposal()`) and P-15 (a "64× shear growth" figure that turned out to be windage variance, withdrawn and corrected).

---

## 10. Round 3 outline (post-selection, → December)

Not scheduled here, but recorded so Round 2 can point at it: SNAP raw-scene ingest · OpenOil weathering · adaptive SMC inversion at 1e6 particles · repeat-offender aggregation across spills · GDP drifter validation · multi-scene temporal linking · external review by an oceanographer · regional then EEZ-scale coverage.

### Phase 3 completion note — 6 Sep 2026

**The inversion works and, more importantly, its uncertainty was measured rather than asserted.**

Backward proposal → hypothesis grid → forward ensemble → Bernoulli conditioning → posterior. Runs in **1.7 s**. The closed-form likelihood is verified against a literal cell-by-cell evaluation, because if that algebra were wrong every number the system produces would be wrong and nothing else would reveal it.

The headline finding is P-17. The first working inversion produced a 95% credible region of 306 km² with correct code, correct algebra and good runtime — and measuring coverage showed that region contained the truth **7% of the time**. Tempering the likelihood for spatial correlation between mask cells fixed it:

| Decorrelation length | 95% coverage | t0 coverage | median 95% area |
|---|---|---|---|
| none (cells independent) | **7%** | 0% | 180 km² |
| 2 km | 43% | 29% | 284 km² |
| 5 km | 100% | 93% | 563 km² |
| **9 km (forcing resolution, default)** | **100%** | 93% | 630 km² |

This is exactly why calibration rather than accuracy is the headline metric — accuracy would have looked excellent.

**Known deficiency, reported not tuned away:** the 50% region does not contain the truth at anything like 50%. The forcing-error perturbation term is not implemented, so the ensemble treats the ocean model as perfect. Phase 7.

**141 tests passing** (28 new). Findings: P-16 (a convex hull is not a slick), P-17 (overconfidence).

Reusable tool: `scripts/calibrate_inversion.py --n 20 --sweep`.

### Phase 4 completion note — 6 Sep 2026

**Attribution runs, and the true culprit was recovered from realistic traffic.**

Seeding a discharge along a real vessel's AIS track, then asking the system to find it:

| | |
|---|---|
| Traffic reduction | **179 vessels → 61 prefiltered → 3 reported** (60×) |
| True culprit rank | **1 of 61** |
| Dark-vessel hypothesis | 41.3% |
| Runtime | 0.8 s — 930 hypotheses, 55,800 particles, **one forward run** |

Every candidate, every release window and the dark hypothesis share a single simulation, distinguished by `origin_marker`. Each vessel's AIS track is used as a generative line source and scored by the *same* observation operator as the inversion, so the numbers are comparable on one scale with no invented weights.

The dark hypothesis at 41% against a top candidate at 8.6% is the honest answer for a 576 km² region containing 61 vessels: no single ship dominates, and the system says so rather than manufacturing a name.

Behavioural priors are five interpretable factors, each individually visible with a plain-language note. The AIS-gap factor is measured against the local baseline gap rate — in a poorly covered area an absolute threshold would flag every vessel present.

**173 tests passing** (32 new). One critical finding: **P-18**, a timestamp-resolution error that silently discarded 99.6% of the AIS while reporting success.

### Phase 5 completion note — 6 Sep 2026

**The vertical slice is closed. A raw SAR scene now goes to a ranked vessel list in 12 s.**

```
python scripts/inject_case.py --case synth_kattegat
python scripts/run_case.py --case data/cases/synth_kattegat_spill
```

| Stage | Result |
|---|---|
| Detection | 4 patches → 1 oil (89 km² vs 87.7 km² truth), 1 look-alike, **2 abstentions** |
| Inversion | 95% region 1,188 km², t0 width 26 h |
| Attribution | 179 vessels → 102 prefiltered → 2 reported; **true culprit rank 2** |
| Runtime | 4.3 s + 5.4 s + 3.1 s = **12.8 s**, against a 60 s budget |

**The demo beat works and is real, not staged.** The 219 km² low-wind pocket is found by the segmenter and then refused by the gate with its wind speed stated as the reason — and that patch exists because the scene's backscatter is generated from the same wind field the gate reads, not because it was painted on.

**Two scoping decisions, both recorded rather than hidden.**

*Stage 1 is classical, not the U-Net.* Our synthetic sigma-0 comes from a smooth analytic wind field with Gamma speckle; a CNN would reach IoU near 1.0 on it and the number would be meaningless. The U-Net remains the real-data path (Phase 5b/8), trained on Zenodo with a geographic holdout and reported against the published ~0.54 benchmark. Both emit the same label raster, so nothing downstream changes.

*Attribution degrades when fed the detector's mask rather than the true particle cloud* — the culprit drops from rank 1 to rank 2. That is the honest end-to-end number, and mask error propagating into attribution is a real property of the system worth stating.

**Known limitation, reported not tuned away.** The background-window parameter in Stage 1 is genuinely sensitive: a window smaller than a broad dark feature sits inside it and the feature vanishes. Measured — 20 km and 40 km miss the calm pocket entirely; 60 km finds it at 257 km²; 90 km over-merges it to 1,104 km². 60 km is defensible physically (low-wind regions are synoptic-scale, slicks are not), but the sensitivity is a real weakness of threshold-based segmentation and one more reason the real-data path uses a trained model.

**200 tests passing** (27 new). One significant finding: **P-19**, a classifier that scored 100% by learning a shortcut, now shipped with a degeneracy warning attached to its own metrics.

### Phase 6 completion note — 7 Sep 2026

**The pipeline has a face, and after this phase the project has no network dependency at all.**

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The export stage turns `out/` into a 2.5 MB static bundle: scene PNG capped at 1100 px, coastline vectorised from the case's own land mask, posterior density as weighted cells, ~900 backward particle trails.

**Verified rendering in a real browser**, not assumed: SAR scene with speckle, coastline, the 219 km² low-wind pocket outlined as *undetermined*, the 89 km² confirmed oil, the rejected look-alike, posterior density, animated backward trails, candidate tracks, and the dark-vessel row at 27.2%.

| Requirement | Status |
|---|---|
| FR-17 map with classification **including rejected patches and their reason** | ✅ |
| FR-18 animated particle playback | ✅ backward proposal, labelled as such |
| FR-19 posterior with credible regions | ✅ 50% and 95%, areas in km² |
| FR-20 ranked candidates + per-factor evidence + dark row | ✅ |
| NFR-1 full offline operation | ✅ deck.gl vendored and committed |

`serve()` refuses any non-loopback host in code — no auth layer plus vessel attribution output means it must never be reachable from a conference network.

**223 tests passing** (23 new). One significant finding: **P-20**, a missing SAR backdrop caused by three stacked defects, none of which raised an error.

### Phase 7 completion note — 7 Sep 2026

**The numbers exist, they were measured honestly, and two of them miss target.**

```bash
python scripts/truth_harness.py --n 24                    # fair test
python scripts/truth_harness.py --n 24 --matched-params   # isolates the cause
```

Every trial runs the whole chain — inject, detect, invert, attribute — so the figures include detector error.

#### Calibration — the metric that matters more than accuracy

| Nominal | Matched priors | **Mismatched (fair test)** |
|---|---|---|
| 50% | 0.50 | **0.50** ✅ |
| 68% | 0.83 | **0.67** ✅ |
| 90% | 1.00 | **0.75** ❌ |
| 95% | 1.00 | **0.83** ❌ (target ≥ 0.90) |

**The machinery calibrates correctly when its priors cover reality; the miscalibration is prior misspecification and lives entirely in the tails.** Not fixed — widening the drift prior would lift the number, but the literature windage range *is* 1–4%, and widening it because our own generator went outside it would be tuning to the test. See P-22.

#### Attribution

| Metric | Measured | Target | |
|---|---|---|---|
| Culprit in candidate set | **1.00** | — | ✅ the prefilter never drops the answer |
| Top-1 recall | 0.33 | 0.30–0.50 | ✅ |
| Top-3 recall | **0.44** | 0.60–0.80 | ❌ below target |
| Median traffic reduction | **179×** | 50–100× | ✅ exceeds |
| **Dark hypothesis ranked top when culprit removed** | **1.00** | — | ✅ the most important result |

#### Envelope grows with lookback, as it must

643 km² @ 12 h → 711 @ 24 h → 1,053 @ 36 h → 1,656 @ 48 h.

#### Completion rate

12 of 24 trials completed. Breakdown: 5 no oil confirmed by the detector, 4 no AIS track spanning the discharge window (a harness limitation — synthetic tracks are short), 2 slick below detectable size, 1 too few particles surviving. Excluding the harness artefact, 12/20 = 60%.

**239 tests passing** (16 new). Two bugs found by the harness *before* it produced a single metric: **P-21** (slick brightness scaled with Monte Carlo particle count) and a thin slick being consumed entirely by the boundary annulus — the signature shape of a continuous discharge.
