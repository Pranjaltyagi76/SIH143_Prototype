# Testing Strategy

**Status:** Basic plan frozen before coding; expanded during development · **Owners:** M4 (harness), M2 (physics validation), M1 (detection metrics) · **Last updated:** 2026-09-05

---

## 0. The central problem: there is no ground truth

**No public dataset of confirmed spill-to-vessel attributions exists.** We looked. This is not a gap in our research; it is the state of the field, and it is why this problem statement is hard and why a working system would be valuable.

The consequence is that **we cannot validate the system end to end.** Pretending otherwise would be the single worst thing we could do. So we decompose validation into four independent claims, each of which *can* be tested, and we are explicit about the seam between them.

| # | Claim | How it is tested | Data is real? |
|---|---|---|---|
| 1 | We can find slicks in SAR | Segmentation metrics on labelled SAR with **geographic holdout** | ✅ fully real |
| 2 | Our transport physics is right | Separation distance vs **real drifter buoy trajectories** | ✅ fully real |
| 3 | Our inversion machinery is correct and *calibrated* | **Synthetic truth harness** — releases seeded from real AIS, drifted with real forcing | ⚠️ real inputs, synthetic event |
| 4 | Our attribution ranks the right vessel | Same harness — the seeding vessel is the known answer | ⚠️ real inputs, synthetic event |

**Claims 3 and 4 are partly circular and we say so out loud.** The simulator generates the observation *and* evaluates the candidates, so it tests the inversion *machinery*, not the *physics*. That is exactly why claim 2 must be validated separately against real drifters, with no simulator in the loop. The prepared answer to "isn't your validation circular?" is that sentence.

---

## 1. The synthetic truth harness — the spine of all evaluation

`scripts/build_synthetic_truth.py`. **Tier A. Scheduled for Day 9.** Without it, three of our strongest slides are blank.

### Design

```
1. SELECT     Pick a real Danish DMA AIS day and a real vessel track.
2. RELEASE    Choose a discharge window W along that track and a release
              mode (continuous | instantaneous).  ← this is the ground truth
3. DRIFT      Seed particles along the track over W. Forward-simulate to
              t_obs with REAL CMEMS + ERA5 forcing.
4. OBSERVE    Threshold particle density into a slick polygon. Perturb the
              boundary to mimic segmentation error. Optionally inject as a
              damping patch into a real SAR tile.
5. FORGET     Discard the ground truth. Hand the pipeline only the mask,
              t_obs, the forcing, and the FULL unfiltered AIS for that day.
6. RUN        Execute stages 4–7 exactly as in the demo.
7. SCORE      Was the true (x0, t0) inside the 50% / 95% credible region?
              Was the true vessel in top-1 / top-3? What was the reduction?
```

Repeat 30–40 times across different vessels, days, lookback times and release modes.

### Anti-cheating rules

These matter more than the harness itself. A harness that flatters the system is worse than no harness.

| Rule | Why |
|---|---|
| **The pipeline never sees the seeding vessel's identity** | Otherwise you are testing a lookup, not an inversion |
| **The full day's AIS goes in, unfiltered** | The prefilter must earn its reduction factor against real traffic density |
| **Vary the drift parameters between generation and inversion** | Generate with `alpha` drawn from a *different* distribution than the inversion assumes. Otherwise you are testing that your simulator agrees with itself |
| **Include cases where the true vessel is deliberately removed from the AIS** | The only way to test the dark-vessel hypothesis. The correct answer is "dark vessel", and the system must produce it |
| **Include a lookback sweep: 12 / 24 / 48 / 72 h** | Produces the operating-envelope plot and proves we know where our method dies |
| **Fixed seeds, logged per run** | Reproducible failures |

> The deliberately-dark cases are the most valuable in the whole harness. They are the only quantitative evidence that the system does not confidently name innocent ships.

### Outputs

- `eval/calibration.csv` — nominal vs empirical coverage, per credible level
- `eval/attribution.csv` — top-1, top-3, reduction factor, per case
- `eval/envelope.csv` — 95% region area vs lookback hours
- Three plots for the slides

---

## 2. Calibration — the metric that wins

**Calibration, not accuracy, is the headline.**

Bin the synthetic runs by nominal credible level (50%, 68%, 90%, 95%) and compute the fraction of runs in which the true source actually fell inside. Plot nominal on x, empirical on y, with the diagonal drawn.

| Result | Interpretation | What we say |
|---|---|---|
| On the diagonal | Uncertainty is honest | "When we say 95%, we mean it" — the strongest claim available to us |
| Below the diagonal | **Overconfident** — regions too small | A serious defect. Widen the parameter priors and re-run. Never ship overconfident |
| Above the diagonal | Conservative — regions too large | Acceptable and disclosable. Say "our envelopes are conservative by roughly X%" |

**Acceptance: empirical coverage of the nominal 95% region ≥ 0.90.**

If the curve is bad, **we put the bad curve on the slide.** A team that measures its own calibration and reports that it is conservative is far more credible than one that never measured. Almost no student team can produce this plot at all.

---

## 3. Test levels

### 3.1 Unit tests (`pytest`, run on every commit)

Small, fast, and each one guards a specific way we know this project can silently break.

| Area | Test | Guards against |
|---|---|---|
| Transport | Zero current + zero wind + zero diffusion ⇒ particles do not move | Sign and unit errors |
| Transport | Constant 1 m/s eastward current for 1 h ⇒ displacement is 3600 m ± 1 m | Metre/degree confusion |
| Transport | Forward then backward with diffusion off returns to the origin within tolerance | The reversibility claim in §4.1 of the design |
| Transport | **Backward with `K_h > 0` raises an error** | The single most important scientific claim in the project, enforced in code |
| Transport | **`includes_stokes: true` ⇒ no Stokes term is added** | Silent double-counting of a real physical term |
| Transport | Particle hitting land is `beached`, not deleted | Posterior bias away from the coast |
| Geometry | Area of a known polygon in km² matches an independent computation | Computing area in degrees |
| Geometry | Elongation of a synthetic ellipse matches its axis ratio | |
| Gate | Wind 1.4 m/s ⇒ `undetermined` regardless of every other feature | The hard override being overridable |
| Gate | Wind 14 m/s ⇒ `undetermined` | Same, upper bound |
| Observation op | Perfect overlap scores higher than partial overlap | |
| Observation op | **A hypothesis covering the whole AOI scores *worse* than a tight correct one** | The exact failure of naive rejection ABC that §4.3 exists to fix |
| Posterior | Density integrates to 1; 50% region area < 95% region area | |
| AIS | A 60 kn implied speed is rejected | |
| AIS | A shared-MMSI track is split | |
| AIS | Interpolated segments are flagged | Interpolation silently becoming evidence |
| Attribution | Probabilities including the dark hypothesis sum to 1 | |
| Attribution | With zero AIS vessels supplied, the dark hypothesis gets probability 1 | Degenerate normalisation |
| Contracts | Every fixture validates against its Pydantic model | Contract drift |

### 3.2 Integration tests

| Test | Assertion |
|---|---|
| `run_case.py` on the tiny fixture case | All four contract objects produced and valid |
| Stage skip/resume logic | Re-running does not recompute; `--force` does |
| **Determinism** | Same case + same seed ⇒ bit-identical `posterior.npz` (NFR-3) |
| **Offline** | Full run with the network adapter disabled (NFR-1) |
| API | Every endpoint returns valid contract JSON |

### 3.3 Detection metrics (M1)

**Geographic holdout, never random tiles.** Adjacent tiles from one scene share speckle statistics and sea state; a random split leaks and inflates the number by a large and unknowable margin.

| Metric | Target | Report |
|---|---|---|
| IoU (oil class), in-domain | 0.50–0.65 | Published SOTA ~0.54 |
| IoU (oil class), cross-region | ~0.10 lower | **Report the drop.** This number is the credibility |
| Recall (oil) | > 0.75 | Stage 1 favours recall by design |
| Gate false-positive rate | < 0.15 | The headline for Stage 2 |
| Gate abstention rate | 5–15% | Too low = not gating; too high = useless |
| Overall pixel accuracy | **Never reported** | Quoting it is a tell that you do not understand the class imbalance |

### 3.4 Physics validation (M2)

Against **Global Drifter Program** trajectories — real ocean, real paths, no simulator in the loop. *(Access unverified as of 2026-09-05; verify Day 3. If unavailable, this claim is dropped and we say so rather than substituting something synthetic.)*

| Metric | Target |
|---|---|
| Separation distance at 24 h | 10–25 km. **Anyone claiming < 5 km against a 1/12° field is not measuring honestly** |
| Skill score vs persistence | > 0.5 — beat "assume it does not move" |

### 3.5 Demo-day tests

Run on the actual demo machine, from a cold boot, with the network adapter physically disabled. See [deployment.md](deployment.md) for the runbook.

---

## 4. What we deliberately do not test

| Not tested | Why | How we handle it |
|---|---|---|
| End-to-end on a real attributed spill | No such public dataset exists | Stated as an open problem — and as something a system like this would help create |
| Weathering | Not implemented in the prototype | Round 3 |
| Sub-mesoscale accuracy | Physically unresolvable at 1/12° | Stated as an irreducible error floor |
| Load / concurrency | Single-user prototype on one laptop | Not a requirement |
| Cross-browser | Demo runs on one known machine and browser | Chrome only, verified |

---

## 5. Test data inventory

| Purpose | Source | Real? |
|---|---|---|
| Segmentation train/val | Zenodo SAR oil spill Parts I–III | ✅ |
| Segmentation benchmark | Krestenitis (pending institutional request) | ✅ if granted |
| Forcing | CMEMS SMOC + ERA5 | ✅ |
| AIS, Danish cases | Danish Maritime Authority | ✅ |
| AIS, Arabian Sea case | Our generator, seeded from real traffic-lane geometry | ❌ **synthetic — disclosed, and permitted by the PS** |
| Synthetic spill events | Our harness | ❌ synthetic by design, that is the point |
| Drift validation | Global Drifter Program | ⚠️ unverified |

---

## 6. Definition of done, per component

A component is done when: its unit tests pass · it emits a valid contract object · its output renders in the UI · its failure mode is handled explicitly (not by an exception) · and its known limitations are written into [problems_faced_and_bugs_encountered.md](problems_faced_and_bugs_encountered.md).

---

## 7. Expansion log

*Appended as testing develops during the sprint.*

| Date | Change | Why |
|---|---|---|
| 2026-09-05 | Initial strategy written | Pre-coding |
