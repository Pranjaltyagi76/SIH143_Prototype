# Requirements

**Status:** Frozen before coding · **Owner:** M6 (Integration + Narrative) · **Last updated:** 2026-09-05

---

## 1. Problem statement traceability

Every requirement below traces to a clause of the official PS. Anything that traces to nothing is scope creep and gets cut.

| PS clause | Verbatim requirement | Our requirement IDs |
|---|---|---|
| (a) | "Detect and characterise the oil spill and calculating geometric properties and **age if feasible**" | FR-1 … FR-5 |
| (b) | "Using oceanographic and meteorological data … trace the slick towards the **origin point and time**, predict the future flow" | FR-6 … FR-10 |
| (c) | "Attribute the spill to a vessel using historic AIS … reconstruct vessel traffic around the **origin window in space and time** … filter irrelevant traffic … score suspects on proximity, trajectory, behavioural anomalies" | FR-11 … FR-16 |
| UI | "A suitable visual interface is also to be developed" | FR-17 … FR-21 |

### The clause (b) / clause (c) tension — and why it settles our design

Clause (b) says **"origin point and time."** Clause (c) says **"origin window in space and time."**

These conflict. When they do, **(c) governs**, because (c) is the clause that describes what the output is actually *used for* — reconstructing vessel traffic over a region and an interval. A point has no traffic in it.

This is not us softening the requirement to protect ourselves. **The PS itself asks for a window.** Our uncertainty-envelope output is literal compliance with clause (c), and this paragraph is the prepared defence if anyone claims we under-delivered on "point".

The PS also explicitly names our data sources: MarineCadastre AIS for format, real AIS *"if available, else synthetic data can be prepared"*, and the Zenodo Sentinel-1 SAR oil spill dataset.

---

## 2. Functional requirements

`P` = in prototype scope (Round 1) · `R3` = Round 3 only · `P-B` = prototype Tier B (build if time)

### (a) Detection and characterisation

| ID | Requirement | Scope | Acceptance criterion |
|---|---|---|---|
| FR-1 | Ingest a calibrated Sentinel-1 GRD IW VV scene as sigma-0 in dB with geolocation | **P** | Scene loads, renders, every pixel maps to lat/lon within 50 m |
| FR-1b | Ingest and preprocess *raw* S-1 GRD (orbit, thermal noise, calibration, speckle, land mask) | R3 | SNAP/pyroSAR chain reproduces reference sigma-0 within 0.5 dB |
| FR-2 | Segment candidate dark patches with high recall | **P** | Recall (oil class) > 0.75 on held-out geographic region |
| FR-3 | Classify each patch as `oil` / `look-alike` / `undetermined` using a physical wind-detectability gate plus learned features | **P** | False-positive rate < 0.15; abstention rate between 5% and 15% |
| FR-3a | System **must abstain** when 10 m wind is outside 3–12 m/s, regardless of classifier output | **P** | Hard override is unit-tested; a 1.4 m/s patch is always `undetermined` |
| FR-4 | Compute geometric properties per confirmed slick: area, perimeter, centroid, boundary polygon, elongation, orientation, fragmentation, complexity, solidity, damping-ratio statistics | **P** | All values present in the `SlickDetection` contract; area in km² via equal-area projection |
| FR-5 | Infer **release mode** (continuous discharge vs instantaneous) from morphology and track alignment | **P** | Outputs a mode label plus the geometric justification string |
| FR-5a | Estimate slick **age** | see §4 | Delivered indirectly as the t0 posterior marginal (FR-9), not as a radiometric estimate |

### (b) Drift, hindcast and forecast

| ID | Requirement | Scope | Acceptance criterion |
|---|---|---|---|
| FR-6 | Ingest ocean surface currents (CMEMS SMOC: currents + tides + Stokes, hourly, 1/12°) and 10 m wind (ERA5, hourly, 0.25°) for a case AOI and time window | **P** | Cached as a `ForcingBundle`; runs with no network access |
| FR-7 | Lagrangian transport: RK4 advection + sampled windage + random-walk diffusion, with land beaching | **P** | 1e5 particles x 48 h completes in < 30 s |
| FR-8 | **Backward advection-only** pass producing a source proposal region | **P** | Bounded region R(t0) per candidate release time in < 5 s |
| FR-9 | **Forward-ensemble Bayesian source inversion** producing a posterior over (lat, lon, t0) with 50% and 95% credible regions and their areas in km² | **P** | Posterior is a normalised 3-D grid; areas reported in km²; empirical coverage of the 95% region >= 0.90 on synthetic truth |
| FR-10 | **Forward prediction** of slick position at +6 / +12 / +24 h with an uncertainty envelope | **P** | Three forecast polygons rendered with their areas |
| FR-10a | Report and enforce a **lookback horizon limit** beyond which the envelope is declared not operationally useful | **P** | System flags or refuses inversions beyond 72 h |
| FR-10b | Oil weathering (evaporation, emulsification, dispersion) via OpenOil / ADIOS | R3 | Mass balance closes; weathered fraction reported |

### (c) AIS reconstruction and attribution

| ID | Requirement | Scope | Acceptance criterion |
|---|---|---|---|
| FR-11 | Ingest historic AIS (Danish DMA CSV real; generator for synthetic) into a queryable store | **P** | One region-month loads; duplicate MMSI, position jumps and impossible speeds filtered and **counted** |
| FR-12 | Reconstruct per-vessel continuous tracks with gap annotation | **P** | Each track carries its gap list with start/end/duration |
| FR-13 | **Filter irrelevant traffic** by spatio-temporal intersection with the posterior support | **P** | Reduction factor reported (target 50–100x, e.g. 214 vessels to 3) |
| FR-14 | Score each surviving vessel by **vessel-conditioned forward simulation** — seed particles along its actual AIS track, forward-simulate, evaluate likelihood against the observed mask | **P** | Every candidate has a log-likelihood from the same observation operator |
| FR-15 | Carry an explicit **dark-vessel hypothesis** in the normalisation, so the system can output "most probable explanation: a vessel not transmitting AIS" | **P** | Appears as a row in the ranked output with its own probability |
| FR-16 | Apply an interpretable behavioural prior: AIS gap (baseline-relative), speed anomaly, course change, vessel type, track–slick axis alignment | **P** | Each factor individually visible per vessel; no hidden weights |
| FR-16a | Repeat-offender aggregation across multiple spills | R3 | A vessel in top-3 across N independent spills is surfaced |
| FR-16b | Dark-vessel *detection* from SAR (CFAR ship detection minus AIS matches) | **P-B** | Radar-detected ships with no AIS match within 500 m are flagged |

### UI

| ID | Requirement | Scope | Acceptance criterion |
|---|---|---|---|
| FR-17 | Map showing SAR scene, segmentation overlay, and classification result **including rejected patches with the reason** | **P** | A rejected look-alike is visibly greyed with its wind speed shown |
| FR-18 | Animated particle playback of backward proposal and forward ensemble | **P** | >= 1e4 particles animate at >= 30 fps |
| FR-19 | Posterior heat map with 50% / 95% credible region outlines, areas labelled in km² | **P** | Areas readable on screen without interaction |
| FR-20 | Ranked candidate panel: vessel identity, probability, per-factor evidence breakdown, AIS track overlay, dark-vessel row | **P** | Clicking a candidate highlights its track and shows its evidence |
| FR-21 | Evidence bundle export (PDF / JSON) with all assumptions stated | **P-B** | Export contains inputs, parameters, posterior summary, ranking, limitations statement |

---

## 3. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | **Full offline operation** during demo — no network call on any code path | Hard requirement. Verified with the network adapter disabled |
| NFR-2 | End-to-end runtime, cached case to ranked output | < 60 s (prototype); < 10 min (full system on a raw scene) |
| NFR-3 | Reproducibility — same case + same seed gives identical output | Bit-identical posterior grid |
| NFR-4 | Every numeric claim in the UI traceable to a computation, not a constant | Audited before demo |
| NFR-5 | Runs on a single laptop with one NVIDIA GPU. No cloud, no Kubernetes | Verified on the demo machine |
| NFR-6 | Every stated probability carries the phrase "under stated model assumptions" | Enforced in UI copy review |
| NFR-7 | Cold start to first pixel on screen | < 10 s |

---

## 4. Explicit non-goals — what we will NOT deliver, and why

Stating these is a maturity signal, not a weakness. Each has a prepared one-line defence.

| Non-goal | Why | What we deliver instead |
|---|---|---|
| **Slick age in hours from a single SAR scene** | Damping ratio confounds film thickness, oil type, wind, incidence angle and sea state simultaneously — not decoupleable from one image. The PS says *"if feasible"*; that is the PS telling you it may not be | The **posterior marginal over t0** from the inversion, with error bars. We do not estimate age and then drift; we drift and thereby infer age |
| **A named guilty vessel** | The system produces investigative leads, not proof. Naming a ship on a probabilistic posterior is both a defamation exposure and a scientific overclaim | A **ranked shortlist** with per-factor evidence and an explicit dark-vessel alternative |
| **A single "confidence %" of guilt** | Meaningless without a reference class. "87% confident" is the exact language that destroys a team under questioning | Posterior probability *conditional on the candidate set including the dark hypothesis*, with assumptions listed alongside |
| **Inversion beyond 72 h lookback** | Shear dispersion separates neighbouring particles super-linearly; the envelope stops being actionable | An explicit **operating envelope**, reported as a system limit |
| **Real-time operational monitoring** | Out of scope for a prototype | Batch processing of prepared cases |
| **Indian-waters real AIS validation** | No free bulk historical source exists. We looked | Synthetic AIS for the Arabian Sea case, **as the PS explicitly permits**, disclosed on the slide |
| **Sub-mesoscale accuracy** | CMEMS at 1/12° (~8–9 km) physically cannot resolve eddies below that scale | An irreducible error floor, stated, and folded into the ensemble |
| Our own SAR calibration chain, ocean model, drift solver or weathering model | Weeks of work, zero innovation credit | Reuse SNAP / CMEMS / ERA5; Round 3 adds OpenOil |
| A supervised attribution ranker | **No labelled spill-to-vessel dataset exists.** Any such model learns our own simulator's biases | Bayesian formulation needing no labels — turning a data limitation into an architectural advantage |
| Chatbot, blockchain evidence ledger, mobile app, "AI agent" layer | Adds risk and zero marks | — |

---

## 5. Acceptance criteria for Round 1 (the demo gate)

The prototype is **done** when all of the following hold on the demo machine with the network disabled:

- [ ] Three prepared cases load and run end-to-end in under 60 s each
- [ ] At least one case contains a dark patch that is **correctly rejected** as a look-alike, with the wind speed shown as the reason
- [ ] The posterior renders with 50% and 95% credible regions and their areas in km²
- [ ] The candidate panel shows a reduction from >100 vessels to <= 5, with the dark-vessel row present
- [ ] A calibration number exists from the synthetic-truth harness — even a bad one; an honest bad number beats an absent one
- [ ] Every screen has been read aloud once, specifically hunting for overclaiming language

---

## 6. Assumptions and dependencies

| Assumption | Risk if wrong | Verification owner |
|---|---|---|
| Zenodo SAR oil spill dataset (Trujillo-Acatitla et al., Parts I–III) is openly downloadable, sigma-0 dB TIFF, with a look-alike class | Training data gone; fall back to Krestenitis or hand-labelled masks | M1, Day 1 |
| Krestenitis benchmark needs an institutional-email request with lead time | Cannot benchmark against published SOTA | M1, request sent Day 1 |
| CMEMS SMOC `cmems_mod_glo_phy_anfc_merged-uv_PT1H-i` already includes currents + tides + **Stokes drift** | **Stokes double-counting bug** if we also parameterise it separately | M2, Day 2 — read the product QUID |
| ERA5 via CDS API requires licence acceptance at the current endpoint | Wind data blocked, which kills the entire physics gate | M3, Day 1 |
| Danish DMA AIS at `web.ais.dk/aisdata/` is free CSV | No real AIS; everything becomes synthetic | M3, Day 1 |
| Sentinel-1 constellation is currently **1C + 1D** (1A terminated 2026) | ⚠️ **UNVERIFIED — post-knowledge-cutoff claim.** Being confidently wrong about mission status in front of NTRO is worse than being vague | M6, verify against ESA before it goes on any slide |
| Global Drifter Program data is accessible for drift validation | Lose the strongest independent validation | M2, Day 3 — **currently unverified** |
