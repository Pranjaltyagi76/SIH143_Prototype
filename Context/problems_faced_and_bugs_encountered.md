# Problems Faced and Bugs Encountered

**Status:** ✍️ Continuous log — appended whenever a problem is solved · **Owner:** everyone; M6 curates · **Last updated:** 2026-09-05

---

## How to use this file

**Append an entry the moment you solve something, not at the end of the day.** By evening you will have forgotten the detail that made it interesting, and the detail is the whole value.

This is not overhead. It is **Round 2 material.** "Here is the bug we found in our own method and how we fixed it" is a stronger slide than "here is our architecture", because it demonstrates the one thing an architecture diagram cannot: that you understood what you were building well enough to catch yourself being wrong.

### Entry format

```
### [P-nn] Short title
**Date** · **Who** · **Severity:** blocker | major | minor | near-miss
**Symptom** — what was observed
**Root cause** — what was actually wrong
**Fix** — what was changed
**Lesson** — what we would tell another team
```

**Near-misses count.** A problem caught in design review before it reached code is worth logging — often more than a bug that reached code, because it is evidence of a working review process.

---

## Design-phase problems (before any code)

### [P-01] Naive backward drift is scientifically wrong
**2026-09-04** · M2 + design review · **Severity:** near-miss (would have been fatal)

**Symptom** — The obvious reading of PS clause (b) — "trace the slick towards the origin" — suggests running a particle model backwards from the observed slick. Our first sketch did exactly that.

**Root cause** — Oil transport is advection **plus** turbulent diffusion. Advection is a deterministic ODE and is time-reversible. **Diffusion is not** — it is entropy-increasing, and integrating it backwards is ill-posed, mathematically equivalent to un-stirring milk out of coffee. Run backwards with diffusion on, the particle cloud spreads and *looks* like an uncertainty envelope, but it is a forward diffusion process pointed backwards in time. It is not a posterior, and any number derived from it is meaningless.

**Fix** — Restructured the entire inversion. Backward integration is used **only** with diffusion disabled, and **only** to narrow the search region. The actual answer comes from forward simulation of candidate sources, conditioned on reproducing the observed slick — Approximate Bayesian Computation. Physics only ever runs forward. A code-level assertion now makes the kernel refuse to run backwards with `K_h > 0`.

**Lesson** — The intuitive reading of the requirement was the scientifically wrong one. *Backward integration narrows the search; forward simulation computes the answer.* We expect most teams attempting this problem statement to get this wrong, and to be unable to defend it when asked.

---

### [P-02] Per-particle rejection ABC computes the wrong likelihood
**2026-09-05** · design review · **Severity:** near-miss (major)

**Symptom** — Our corrected design said: *accept particle i if its position at t_obs falls inside the observed slick polygon; the origins of accepted particles are the posterior.* This felt rigorous. It is not.

**Root cause** — That computes `p(x0, t0 | one oil parcel ended up somewhere in the slick)`, which is **not** `p(source | observed slick shape)`. Two consequences, both bad:
1. **False coverage is never penalised.** A hypothesis whose particles smear across the entire scene earns exactly the same per-particle credit as one producing a compact cloud matching the slick precisely — because only hits are counted and misses cost nothing.
2. **All shape and extent information is discarded** — the very quantities the characterisation stage had just computed.

**Fix** — Replaced per-particle acceptance with a **hypothesis-level Bernoulli likelihood evaluated over the whole AOI**. Each source hypothesis is rasterised to a predicted oil-presence probability `q_h(x) = 1 − exp(−λρ_h(x))` and scored against the observed mask with `Σ [ m log q + (1−m) log(1−q) ]`. Predicted oil where the satellite saw none now costs you.

**Lesson** — "We used ABC" is not the same as "we specified the right observation operator." The likelihood is where a Bayesian method is actually right or wrong, and it is the part everyone skips. Ask of any likelihood: *what does a hypothesis have to do to score badly?* If the answer is "nothing", it is not a likelihood.

---

### [P-03] The correct likelihood is 4×10⁹ operations per case
**2026-09-05** · design review · **Severity:** near-miss (major)

**Symptom** — The fix in P-02 requires evaluating over every AOI pixel for every hypothesis: ~4,096 × 10⁶ = 4×10⁹ operations. That would have dominated the entire 60 s runtime budget and probably broken it.

**Root cause** — Naive literal implementation of the summation.

**Fix** — The negative term has a closed form. Since `log(1 − q_h(x)) = −λ ρ_h(x)`, the sum over non-mask pixels collapses to `−λ × (particle mass landing outside the mask)` — a count, not a sum over pixels. The positive term runs only over mask pixels, a small set. Cost drops from O(hypotheses × pixels) to O(mask pixels + particles), roughly 2 s instead of 40 s, **with no approximation.**

**Lesson** — Do the algebra before writing the loop. It also made the method *easier to explain*: the likelihood is now literally "reward for covering the observed slick, minus a penalty for every particle predicted where the satellite saw nothing." Correctness and clarity moved in the same direction, which is usually a sign you found the right form.

---

### [P-04] Track-integral scoring is a proxy, not a likelihood
**2026-09-05** · design review · **Severity:** near-miss (major)

**Symptom** — Our attribution design scored vessels by integrating the source posterior along each vessel's AIS track: `L(v) = ∫ p(x_v(t), t) dt`. This is a real improvement on invented weights, and we nearly shipped it.

**Root cause** — It is still a *proxy*. `p(x0, t0 | obs)` is a marginal over **point** sources, but a real discharge from a moving vessel is a **line source in space-time**. Evaluating a point-source marginal along a line does not give you `p(obs | vessel v)`. It also inherits whatever smoothing the KDE applied, and it throws away the slick's shape a second time.

**Fix** — Realised that a vessel's AIS track **is** the parameterisation of the line source — we already have it, so we do not need to invert for it. We can simulate it directly. Each vessel becomes a **generative hypothesis**: seed particles along its actual track at its actual times, forward-simulate, and evaluate the *same* observation operator from P-02. All candidates are seeded into one simulation, distinguished by `origin_marker`, so ~20 vessel hypotheses cost one run rather than twenty.

**Lesson** — The best version of this was cheaper *and* more correct than the version we nearly built. When the principled approach looks more expensive, check whether you are solving a harder problem than the one you have — we had been treating the discharge location as unknown when AIS already tells us exactly where each candidate was.

---

### [P-05] Stokes drift would have been double-counted
**2026-09-05** · design review · **Severity:** near-miss (major)

**Symptom** — The transport equation was drafted as `currents + tides + Stokes + windage`, with each term added explicitly. Standard, and it appears in most drift-modelling write-ups.

**Root cause** — We source currents from **CMEMS SMOC**, which *already merges* geostrophic currents, tides and Stokes drift into its `uo`/`vo` fields — that is precisely why CMEMS recommends it for Lagrangian applications. Adding a separate Stokes parameterisation on top would have counted a real physical term twice, biasing every trajectory downwind by a systematic amount, silently, with no error raised.

**Fix** — `ForcingBundle.currents.includes_stokes` is now a required contract field, and the transport kernel reads it. A unit test asserts no Stokes term is added when it is `true`. The docstring of the windage function states it explicitly.

**Lesson** — A bug that produces *plausible* wrong answers is far more dangerous than one that crashes. We would never have noticed this from the output. **Encode the assumption in the data contract, not in someone's memory** — that is the only fix that survives a tired teammate on Day 11.

---

### [P-06] SNAP and OpenDrift on Windows put the demo at risk
**2026-09-05** · planning · **Severity:** blocker (avoided by scoping)

**Symptom** — The reference stack requires ESA SNAP (Java, via `snappy`) for SAR preprocessing and OpenDrift (conda, GDAL) for drift. The team is on Windows 11. Both are well-documented sources of multi-day environment failure, and both sat on the critical path.

**Root cause** — Adopting a research-grade stack wholesale without asking which parts the *prototype* actually needs.

**Fix** — Removed both from the prototype. The Zenodo training data is already calibrated σ⁰ in dB, so SNAP is unnecessary until we ingest raw scenes (Round 3). OpenDrift was replaced with a ~150-line Lagrangian kernel — RK4 advection, sampled windage, random-walk diffusion — because **the differentiator is the inversion method, not the ODE solver.** The remaining stack installs with plain `pip` into a plain `venv`: no conda, no WSL2, no Docker, no Java.

**Lesson** — Ask what a dependency actually buys *this* deliverable. OpenDrift buys weathering and citability; we need neither for Round 1, and we pay for both with the largest schedule risk in the project. Recorded as debt items D1 and D2 in [engineering_review.md](engineering_review.md), each with a one-class exit path.

---

### [P-07] A mission-status claim we cannot verify
**2026-09-05** · M6 · **Severity:** minor (open)

**Symptom** — Our source design document asserts that Sentinel-1A was terminated on 29 June 2026 and that Sentinel-1D has been open-access since 17 April 2026, and recommends saying "1C/1D" to signal currency.

**Root cause** — Both claims post-date the knowledge available to us at drafting time and neither has been checked against ESA directly.

**Fix** — ⬜ **Open.** M6 to verify against ESA mission documentation before this appears on any slide. Until verified, say "the current Sentinel-1 constellation" rather than naming satellites.

**Lesson** — The advice was right — knowing current mission status signals you read the documentation — but it cuts both ways. Being confidently *wrong* about a satellite's status in front of NTRO is far worse than being unspecific. A detail included to demonstrate currency is exactly the detail that must be verified.

---

## Watch list — traps predicted but not yet hit

Logged in advance so they are recognised in seconds rather than debugged for hours. Move an entry up into the log when it actually bites.

| # | Predicted problem | Recognise it by | Pre-planned response |
|---|---|---|---|
| W-01 | Metre/degree confusion in the transport kernel | Trajectories look right near the equator, wrong at high latitude | Integrate in projected UTM metres, never degrees. Unit test: 1 m/s for 1 h = 3600 m |
| W-02 | Meteorological "wind from" vs oceanographic "wind to" sign error | Slick drifts exactly the wrong way | Store `u10`/`v10` components only, never a bearing. Derive direction once, at the UI boundary |
| W-03 | Deleting beached particles biases the posterior away from the coast | Coastal sources under-represented; posterior hugs open water | Mark `beached` and freeze. **Never delete.** Unit tested |
| W-04 | AIS timestamps assumed local instead of UTC | Attribution off by a fixed whole number of hours | Assert UTC on load. A constant offset in the answer is the tell |
| W-05 | Random tile split leaks between train and test | IoU suspiciously high, ~0.8+ | Geographic holdout only. If IoU exceeds 0.7, assume leakage before assuming success |
| W-06 | Reporting overall pixel accuracy instead of oil-class IoU | A number near 0.99 | Never report overall accuracy. It measures "predicted all sea" |
| W-07 | AIS gaps treated as evidence of concealment | Every vessel in a poorly-covered area looks suspicious | Baseline-relative gap prior. Gaps are overwhelmingly benign |
| W-08 | Effective sample size collapses; posterior is Monte Carlo noise | A confident-looking blob that moves between runs with different seeds | Report ESS. If it collapses, **the UI must say so** |
| W-09 | Interpolated AIS segments silently become evidence | A vessel scores well on a stretch where we invented its positions | Flag interpolated segments in the contract; exclude or downweight |
| W-10 | The demo depends on a CDN | Works in dev, fails offline on Day 12 | Vendor deck.gl and MapLibre locally on Day 12 |
| W-11 | Area computed in degrees | km² values wrong by a latitude-dependent factor | Equal-area or UTM projection. Unit tested against a known polygon |
| W-12 | NetCDF reads inside the particle loop | Ensemble takes minutes instead of seconds | Preload forcing into one in-memory float32 array |
| W-13 | Contract drift after Day 2 | Integration breaks on Day 8 in a way nobody can localise | Contracts frozen Day 2. Fixtures validated in tests |
| W-14 | Synthetic AIS uses a real MMSI | A real vessel appears in our accusation demo | Generator restricted to reserved/invalid MMSI ranges. Unit tested |

---

## Build-phase log

### [P-08] Two-argument lambda validators silently receive the wrong arguments
**2026-09-06** · Phase 0 · **Severity:** major (caught before it shipped)

**Symptom** — Two Pydantic field validators for list-valued UTC datetime fields were written compactly as `field_validator("t0")(lambda cls, v: [require_utc(t) for t in v])`, mirroring the `(cls, value)` signature of the decorated classmethods elsewhere in the file.

**Root cause** — Pydantic v2 inspects the arity of a plain (non-classmethod) validator function. A two-argument function is interpreted as `(value, info)`, **not** `(cls, value)`. So `cls` would have been bound to the list of datetimes and `v` to a `ValidationInfo` object — meaning the loop would have iterated the wrong thing and the UTC check would never have run as intended on `PosteriorGrid.t0` or `TimeMarginal.bins_utc`.

**Fix** — Reduced both to single-argument lambdas, which Pydantic treats unambiguously as `(value)`. Added `test_posterior_list_datetimes_are_utc_checked`, which feeds a naive datetime into `grid.t0` and asserts the rejection.

**Lesson** — A validator that silently does nothing is worse than no validator, because it buys false confidence — and this one guarded W-04, the timezone error that shifts every attribution by a constant number of hours. **Every guard needs a test that proves it fires.** All 45 contract tests in Phase 0 are written that way: each one deliberately constructs the invalid input and asserts the rejection.

---

### [P-09] Python 3.14 was the system default; the stack needed a decision
**2026-09-06** · Phase 0 · **Severity:** minor (resolved)

**Symptom** — The dev machine had Python 3.14.6 as default and 3.13 available, but not the 3.11 our deployment plan recommended. PyTorch and the geospatial wheels historically lag new CPython releases.

**Root cause** — The 3.11 recommendation was written from general caution, not from a check against this machine.

**Fix** — Resolved empirically instead of by assumption: a `pip install --dry-run` sweep over the entire dependency set on 3.13 resolved **all 20 packages**, including `torch 2.14.0+cu126`. No 3.11 install needed. Exact versions pinned into `requirements.txt`; `deployment.md` §2.2 updated with the evidence table. Note `cu121` has no 3.13 build — use **cu126**.

**Lesson** — Thirty minutes of empirical checking beat a plausible assumption in both directions here. Had we followed the doc blindly we would have installed a redundant Python; had we assumed 3.14 was fine we would have hit a wall later. **Check the actual machine before trusting a general recommendation.**

---

### [P-23] The Zenodo dataset cannot be used for end-to-end cases
**2026-09-07** · Phase 8 · **Severity:** major (an architectural finding, not a defect)

**Symptom** — The problem statement names the Zenodo Sentinel-1 oil spill dataset as our data source. The plan assumed it would carry us end to end. Reading the record's own format description shows it cannot.

**Root cause** — The dataset is **image crops**: 2048×2048×2 TIFF tiles of σ⁰ in dB with matching masks, organised as `Oil` / `Lookalike` / `No oil`. Nothing in the record promises a CRS or a geotransform, and without geolocation a tile **cannot be drifted, cannot be matched to a current field, and cannot be correlated with AIS**. Three of the four pipeline stages are unavailable on it.

**Fix** — Split the sourcing explicitly in code rather than discovering it at integration time:

| Source | Unlocks |
|---|---|
| Zenodo tiles | Segmentation training + IoU benchmarking against the published ~0.54 **only** |
| CDSE GRD + CMEMS + ERA5 + AIS | End-to-end cases; all four needed together |

`zenodo.verify_dataset()` reports whether tiles carry a CRS, so the assumption is tested against the real archives rather than trusted. It was always implicit in the design — Zenodo listed under "Training", CDSE under "Inference" — but nothing in the code said so, and a plan that only lives in a table gets forgotten.

**Lesson** — **Read the data's format description before planning around it.** Five minutes on the Zenodo record answered a question that would otherwise have surfaced after a 96 GB download and a day of confusion. The record also gave the exact band layout, class directories and licence, which is what let the adapter be written correctly without the bytes.

Second lesson: the archives total **96 GB**. "Free" and "small" are different properties, and the plan had quietly conflated them.

---

### [P-22] The 95% region is overconfident, and the harness said which half is wrong
**2026-09-07** · Phase 7 · **Severity:** major (a real limitation, reported not fixed)

**Symptom** — With the anti-cheating rules switched on, the 95% credible region contained the true source **83%** of the time and the 90% region **75%**. Both fall short. In Phase 3, the same code reported **100%** coverage.

**Root cause** — Phase 3's figure was measured without two things the harness now does: it used the true particle cloud rather than the detector's mask, and it generated drift with the *same* distributions the inversion assumes. The second is the one that matters. The harness now generates windage from U(0.005, 0.055) and diffusivity from LogU(0.5, 20), while the inversion assumes the literature ranges U(0.01, 0.04) and LogU(1, 10). When the truth lies outside the assumed prior, the posterior is too tight — which is the correct behaviour of a correct algorithm given a wrong prior, not a defect.

Running both modes decomposes it cleanly:

| Nominal | Matched priors | Mismatched (fair test) |
|---|---|---|
| 50% | 0.50 | 0.50 |
| 68% | 0.83 | 0.67 |
| 90% | **1.00** | **0.75** |
| 95% | **1.00** | **0.83** |

**The inversion machinery calibrates correctly — conservatively, even — when its priors cover reality. The miscalibration is prior misspecification, and it lives entirely in the tails.**

**Fix** — None applied, deliberately. Widening the drift prior would lift the number, but the literature range for windage *is* 1–4%; widening it because our own generator went outside it would be tuning to the test, which is the exact failure this project has caught four times already. What we do instead: report both curves, state that the operational prior should be widened because real oil's windage varies with type, weathering and sea state more than one literature range captures, and record the shortfall in the acceptance criteria rather than quietly meeting them.

**Lesson** — **Measure the thing twice, differently, and the difference tells you where the error lives.** One coverage number would have left "the inversion is overconfident" as an unactionable verdict. Two numbers separate "the algorithm is wrong" from "the prior is wrong", and only the second is survivable.

It also retrospectively devalues the Phase 3 result. That 100% was real, but it answered a narrower question than it appeared to: *machinery correct given correct priors*, not *system calibrated end to end*. Both belong on the slide, labelled.

---

### [P-21] Slick brightness depended on the Monte Carlo particle count
**2026-09-07** · Phase 7 · **Severity:** major

**Symptom** — The first harness run skipped three of four trials: two for "injected slick below the minimum detectable size", one for "no oil confirmed by the detector". Detection rate 25%, which would have looked like a badly underperforming detector.

**Root cause** — Not the detector. `render_damping` computed surface concentration as raw **particle count per scene cell**, so a slick's radar darkness scaled with how many particles the simulation happened to use. Particle count is a Monte Carlo sample size — a numerical parameter with no physical meaning. Doubling N doubled the apparent oil.

The compounding effect is what made it bite: the harness deliberately generates with high diffusivity, so clouds spread wider, so each cell held fewer particles, so the slick faded below the detection threshold *because it was well dispersed*. The harness was reporting its own generator's spread as an undetectable spill.

**Fix** — Damping now depends on the **fraction of the release** in each cell, not the count. Verified invariant: peak damping −7.7/−6.9/−6.9 dB and detectable area within 1.5% across N = 2,000 / 8,000 / 32,000, where previously it scaled with N. The physics is also better this way — mass per unit area genuinely falls as a slick spreads, so a thin sheen damps less than a thick film.

**Lesson** — **A numerical parameter had leaked into the physics.** N controls sampling accuracy and nothing else; the moment a physical output moves when you change it, the model is wrong regardless of how plausible the output looks. The invariance check is two lines and should have been written when the injector was.

The harness earned its cost on its very first run, before producing a single metric.

---

### [P-20] The SAR backdrop never appeared, and nothing reported an error
**2026-09-07** · Phase 6 · **Severity:** major

**Symptom** — The interface rendered detections, posterior, tracks and the animation correctly, but the SAR scene behind them was simply absent. No console error, no failed request, no missing layer. `deckgl.props.layers` listed `scene` every time.

**Root cause** — Three separate things, found in order.

*deck.gl is not published on cdnjs at all.* The page loaded it from `cdnjs.cloudflare.com/ajax/libs/deck.gl/…`, which returns 404 for **every** version — the library metadata endpoint reports "Library not found". The page's own guard caught this and said so, which is the only reason it was not a silent blank map.

*`HeatmapLayer` was broken on this build.* It emitted a wall of luma.gl "unknown binding" warnings and painted over the viewport. Replaced with overlapping translucent `ScatterplotLayer` discs, which are dumber, render correctly, and read the same.

*And the actual cause:* `BitmapLayer` was given the image as a **URL string**, so it began an asynchronous fetch-and-decode each time the layer was constructed. The particle animation reconstructs layers roughly 17 times a second, so the upload was restarted before it ever completed. Clearing the interval made the scene appear instantly — that was the diagnostic that isolated it.

**Fix** — Vendored deck.gl into `web/vendor/` and committed it, decoded the scene bitmap **once** per case into an `Image` and handed deck.gl the decoded object rather than a URL, and swapped the heatmap for discs.

A wrong turn worth recording: the first attempt cached layer *instances* and reused them across frames. That did not help and is a deck.gl anti-pattern — layers are immutable descriptors diffed by `id`, so reusing instances is meaningless at best. Reverted to fresh instances per frame, which is correct once the async load is gone.

**Lesson** — **A missing visual element produces no stack trace.** Three defects stacked up here and not one of them raised: a 404 on a CDN that does not host the library, a layer type quietly painting over everything, and an async upload perpetually restarting. The only reliable way through was bisection — render the layer alone, then stop the animation — rather than reading code.

Also: the vendoring was scheduled as a Day 12 hardening task. Doing it on the day the dependency was introduced turned an NFR-1 compliance chore into the fix for an actual bug, and removed the project's last network dependency two weeks early.

---

### [P-19] The look-alike classifier scored perfectly by learning a shortcut
**2026-09-06** · Phase 5 · **Severity:** major (a metric that would have been quoted)

**Symptom** — The trained LightGBM gate reported **100% in-domain accuracy, 0.000 false-positive rate, and 94% on a cross-region holdout.** Excellent numbers for the hardest problem in SAR oil detection, and exactly what the design hopes for.

**Root cause** — The feature-importance table gave it away. `wind_speed_ms` carried **89.6% of total gain**, `damping_ratio_db` most of the rest, and **11 of 14 features carried none at all.** Every shape and edge feature the design leans on — edge-gradient ratio, solidity, elongation, complexity — contributed exactly zero.

The reason is an artefact of the synthetic world. Injected slicks are seeded along vessel tracks in mid-domain where the wind is ~7 m/s, while the segmenter's false positives cluster at the *rim of the calm pocket*, where wind is 3–4 m/s — just inside the gate window. Wind speed therefore separates the two classes almost perfectly, and the classifier never had to learn anything harder. On real SAR, oil and look-alikes both occur across the whole wind window and no such shortcut exists.

Worth noting the shortcut is not *wrong* — low wind genuinely does raise the odds of an artefact, and the design lists wind as feature #1 deliberately. The problem is that it left nothing for the other thirteen features to do, so the numbers say nothing about the discrimination we actually claim.

**Fix** — The metric was not "corrected"; it was qualified. Added a degeneracy check to `train_gate.py` that prints a **DEGENERATE MODEL WARNING** whenever one feature exceeds 70% of gain or fewer than four features contribute, stating in the output that the accuracy figures describe a shortcut and must not be quoted as evidence the shape features work. The model is still shipped — it does the job on synthetic scenes, giving zero false positives on both clean regions — but its reported performance now carries its own health warning.

**Lesson** — **A number that is better than the state of the art is a bug report, not a result.** Published SOTA for look-alike discrimination is nowhere near 100%, so a student prototype reaching it on the first try meant the task had been made easy, not that the method was good.

The general form: when a model performs implausibly well, *read the feature importances before believing the accuracy*. The importance table diagnosed this in one glance; the confusion matrix never would have.

And the honest consequence, which goes on the slide: **look-alike discrimination cannot be validated on synthetic scenes.** It needs the Zenodo and Krestenitis data, where the classes genuinely overlap. Until then we report the physics gate — which is a rule, not a learned thing, and needs no validation set — and say plainly that the learned half is unproven.

---

### [P-18] A timestamp-resolution error silently deleted 99.6% of the AIS
**2026-09-06** · Phase 4 · **Severity:** critical

**Symptom** — The first attribution run printed a cleaning report showing `dropped_speed_jump: 43130` out of `rows_in: 43309`. The speed filter had discarded **99.6% of every AIS fix in the case**, leaving 179 rows from 179 vessels. Nothing raised. Nothing warned. The report looked like an ordinary summary of a cleaning step.

**Root cause** — Epoch seconds were computed as `df["timestamp"].astype("int64") / 1e9`, the standard pandas idiom. But pandas 3.0 reads this parquet column as `datetime64[us]`, not `[ns]`, so `astype("int64")` yields **microseconds**. Dividing by 1e9 gave seconds ÷ 1000. Every `dt` between consecutive fixes was 1000× too small, every implied speed 1000× too large, and a 12-knot cargo ship looked like it was doing 12,000 knots.

**Fix** — Replaced with a resolution-independent conversion, `(ts - pd.Timestamp(0, tz="UTC")).dt.total_seconds()`. Then added a guard: `clean_and_reconstruct` now **refuses** to return a dataset whose speed filter dropped more than half the fixes, because that is a unit error rather than dirty data. A test reproduces the gutting and asserts the refusal.

**Lesson** — Three things.

*The failure mode was a silent success.* No exception, no NaN, no obviously wrong number — just a filter doing exactly what it was told with corrupted inputs. Had the cleaning report not been printed during a smoke test, attribution would have run on 0.4% of the traffic and produced a confident, plausible, meaningless ranking.

*The idiom was correct until the library changed underneath it.* `astype("int64") / 1e9` is what everyone writes, and it was right for pandas 2.x defaults. **Never encode a unit assumption you have not verified against the actual dtype.**

*Sanity-check the magnitude of what a filter removes.* Dropping 99.6% of anything should be impossible to ship quietly. That guard now lives in the code, not in someone's memory.

---

### [P-17] The posterior was overconfident by treating mask cells as independent
**2026-09-06** · Phase 3 · **Severity:** critical (caught by measurement, not by inspection)

**Symptom** — The first working inversion recovered a source in 1.7 s and produced a 95% credible region of **306 km²** — an order of magnitude tighter than the 2,000–8,000 km² the design anticipates. The true source fell *outside* the 50% region and the true release time *outside* the 95% interval. Effective sample size was 0.77%.

**Root cause** — The Bernoulli likelihood sums one term per mask cell. At a 500 m grid a 128 km² slick is ~500 cells, so the log-likelihood is a sum of 500 terms and any small difference in fit is multiplied by 500. **The cells are not independent observations.** Neighbouring cells are perfectly correlated at scales the drift model cannot resolve, and the binding scale is the forcing resolution: at CMEMS 1/12° the model has no information about structure below ~9 km, so two mask cells 2 km apart say the same thing about a source hypothesis. This is a textbook pseudo-likelihood overconfidence bug.

**Fix** — Temper the likelihood by `n_effective / n_cells`, where `n_effective = slick area / correlation area` and the correlation length defaults to the forcing resolution. Then **measure it**, with `scripts/calibrate_inversion.py`, over 14 synthetic releases:

| Decorrelation length | 95% coverage | t0 coverage | median 95% area | ESS |
|---|---|---|---|---|
| **none (cells independent)** | **7%** | 0% | 180 km² | 0.4% |
| 2 km | 43% | 29% | 284 km² | 1.1% |
| 5 km | 100% | 93% | 563 km² | 24.7% |
| **9 km (forcing resolution, default)** | **100%** | 93% | 630 km² | 38.9% |

**Lesson** — A nominal 95% region that contains the truth 7% of the time is the exact failure this project exists to avoid: it is the mechanism by which a system confidently names an innocent ship. It was invisible to inspection — the code was correct, the algebra was correct, the runtime was good, and the answer looked precise. **Only measuring coverage revealed it.** This is why calibration, not accuracy, is the headline metric: accuracy would have looked excellent here.

---

### [P-16] A convex hull is not a slick
**2026-09-06** · Phase 3 · **Severity:** major

**Symptom** — The first calibration harness built its synthetic observation by taking the convex hull of the drifted particle cloud. Coverage numbers came out strange and inconsistent across settings.

**Root cause** — [testing_strategy.md](testing_strategy.md) specifies thresholding particle *density* into a slick polygon; I used a hull instead, which was faster to write. A hull is far larger than the plume it encloses — it fills in every concave gap a real drifting slick has. Scoring hypotheses against an inflated observation systematically rewards those that over-spread, biasing the posterior away from the true compact source.

**Fix** — Added `ObservedMask.from_points()`, which histograms the particles, smooths, and thresholds to the level containing 85% of the mass. A test now asserts the density-thresholded mask is strictly smaller than the hull of the same cloud.

**Lesson** — The shortcut changed the physics of the test, not just its speed. It is also a reminder that the written strategy said the right thing and the implementation quietly did something else; the deviation was not deliberate and was not recorded until it caused wrong numbers.

---

### [P-15] A "64x shear growth" figure that was actually windage variance
**2026-09-06** · Phase 2 · **Severity:** major (a wrong claim, caught before it reached a slide)

**Symptom** — A first diagnostic of the synthetic ocean reported that a 200 m particle cloud grew **64x over 48 hours**, which we took as evidence that shear dispersion was well represented. It is the kind of number that would have gone onto a slide.

**Root cause** — The diagnostic ran with `TransportParams(diffusion_enabled=False)`, which switches off diffusion but leaves **windage enabled with alpha sampled per particle**. Windage spans 1–4% of a ~6 m/s wind, so different particles drift at velocities differing by ~0.09 m/s; over 48 hours that alone is ~15 km of spread. The measurement was dominated by parameter variance, not by the flow.

Re-measuring with windage *and* diffusion off gave the truth: **1.06–1.32x**, an effective strain of ~1.5×10⁻⁶ s⁻¹ — the very bottom of the realistic range. The synthetic ocean barely sheared at all.

**Fix** — Two parts. First, honesty: the earlier number was withdrawn. Second, physics: added a narrow zonal jet, since fronts and jets are where sub-mesoscale separation actually happens. Re-measured separation is now **2.1–3.1x over 48 h** (~6.5×10⁻⁶ s⁻¹), which is realistic for a shelf sea. The code comment states the measured figure, not the hoped-for one.

**Lesson** — Two of them, and the second is the important one.

*Attribute an effect before believing it.* "Cloud grew 64x, therefore our shear is realistic" conflated four mechanisms into one number. The way to test a mechanism is to switch off everything else.

*A synthetic world that flatters the system is worse than no synthetic world.* Had this stood, every inversion metric through Phase 7 would have been computed against an ocean whose uncertainty came almost entirely from a parameter we control, and we would have discovered it at Phase 8 with real data and no time left.

---

### [P-14] The kernel is reversible -- but only with parameters held fixed
**2026-09-06** · Phase 2 · **Severity:** major (design consequence, not a code bug)

**Symptom** — The reversibility check failed badly: forward 24 h then backward 24 h with advection only left particles **5.1 km** from their origin, median. The claim that advection is time-reversible is the justification for the entire backward proposal pass, so this looked serious.

**Root cause** — Not the integrator. `alpha` (windage) is sampled per particle from U(0.01, 0.04) at the start of each run, so the forward run and the backward run used **different windage for the same particle**. Two draws differ by ~0.09 m/s against a 6 m/s wind, and over 24 hours that is ~7.8 km.

Isolating it confirmed the kernel is sound:

| Configuration | Median round-trip error over 24 h |
|---|---|
| Windage off | **0.0 m** |
| Windage fixed at 0.025 | **13 m** |
| Windage sampled U(0.01, 0.04) | **5,132 m** |

**Fix** — Added `TransportParams.proposal()`, which returns diffusion-off *and* windage-fixed-at-the-mean. The backward proposal pass exists to narrow the search region; sampling windage there would widen it by kilometres of displacement that looks like physical uncertainty and is not. Making the correct configuration a named factory means the inversion cannot get it wrong by omission.

**Lesson** — "Advection is reversible" is true of the *equation* and only conditionally true of an *implementation*: it holds when every per-particle parameter is held fixed across the two runs. That distinction is not in any of the design documents, and it would have quietly widened every proposal region we ever generated. Also worth noting that the 13 m residual with fixed windage is itself explainable — wind is evaluated once per step at the step-start position, which differs between directions — rather than being unexplained slop.

---

### [P-13] The calm pocket was too small for the wind grid to resolve
**2026-09-06** · Phase 1 · **Severity:** minor (found by inspection, not by a test)

**Symptom** — The synthetic wind field was configured with a calm pocket bottoming out at 1.3 m/s, comfortably below the 3 m/s detectability threshold. But sampling the written `wind.nc` gave a **minimum of 2.0 m/s**, and only 1.7% of the domain fell below the gate threshold.

**Root cause** — The pocket had a 22 km radius, and the wind grid is written at the ERA5 spacing of 0.25° — about 28 km in latitude at 57°N. The pocket fell between grid points, so the coarse field never sampled its floor.

**Fix** — Widened the pocket to 35 km. Sampled minimum is now 1.61 m/s and 5.1% of the domain sits below the gate, which lands inside our 5–15% target abstention band.

**Lesson** — Worth keeping rather than "fixing away", because **it is a real effect, not an artefact**: coarse reanalysis genuinely under-resolves small calm patches, and that is one of the reasons look-alike rejection is hard in practice. The physics gate will read 0.25° ERA5 while the SAR scene shows 10 m structure, and that mismatch is a real limitation of the method. We kept the effect and gave the gate a clear signal, rather than pretending the resolution mismatch does not exist. Also a reminder that **the useful check is on the written artefact, not the config value** — the config said 1.3 and the file said 2.0.

---

### [P-12] xarray cannot serialise timezone-aware datetimes
**2026-09-06** · Phase 1 · **Severity:** minor

**Symptom** — `Dataset.to_netcdf()` failed with `unable to infer dtype on variable 'time'` when the time coordinate was built from timezone-aware `datetime` objects.

**Root cause** — NetCDF and CF have no timezone concept; time is stored numerically against a reference epoch. xarray therefore refuses tz-aware Python datetimes rather than silently dropping the offset.

**Fix** — Added `as_naive_utc()`, which converts to UTC and then strips the tzinfo before writing. No information is lost because everything in the project is UTC, enforced at the contract boundary by `require_utc`.

**Lesson** — The important half is the **read** side, not the write side. Because the file now stores naive timestamps, any reader that forgets to re-attach UTC has silently created watch-list item W-04 — the timezone error that shifts every attribution by a constant number of hours. The conversion helper carries that warning in its docstring, and `test_ais_timestamps_are_utc_aware` guards the AIS path.

---

### [P-11] Unanchored .gitignore patterns silently excluded real source
**2026-09-06** · Phase 0 · **Severity:** major (caught at first commit)

**Symptom** — After creating the module skeleton, `git status` listed `src/api/`, `src/detect/`, `src/transport/` and the rest, but **`src/eval/` was missing entirely** — no error, no warning.

**Root cause** — The `.gitignore` contained bare `eval/` and `vendor/`, intended for the root-level evaluation output directory and the vendored CDN assets. Gitignore patterns without a leading slash match a directory of that name **at any depth**, so `eval/` also matched `src/eval/` — the evaluation harness that produces the calibration curve and top-K metrics — and `vendor/` matched `web/vendor/`.

**Fix** — Anchored both: `/eval/` for the root output directory, and `web/vendor/*` with a `!web/vendor/.gitkeep` exception. Verified with `git check-ignore -v`.

**Lesson** — This would have been genuinely expensive. `src/eval/` is M4's deliverable and the source of every number on three slides; it would have been written, worked locally, and simply never appeared for anyone who cloned — with no error at any point. **Read `git status` against the directory listing you expect, at least once, on the first real commit.** An unanchored gitignore pattern is a silent data-loss bug, and `git check-ignore -v <path>` names the exact line responsible.

---

### [P-10] UTM zone arithmetic in a test, not in the code
**2026-09-06** · Phase 0 · **Severity:** minor

**Symptom** — Two parametrised tests of `BoundingBox.utm_epsg()` failed: 69°E returned zone 42 where the test expected 43, and −90° returned zone 16 where the test expected 15.

**Root cause** — The test expectations were wrong, not the implementation. Zone 42 spans 66–72°E, so 69°E is zone 42. And −90° sits exactly on the 15/16 boundary, where the convention assigns the higher zone.

**Fix** — Corrected the expectations and moved the Gulf of Mexico case to −92° so it no longer sits on a zone boundary at all.

**Lesson** — Worth logging because it is the *reassuring* kind of failure: the guard against computing areas in degrees (W-11) fired on the first run and caught a disagreement, which is exactly what it is for. Also a reminder to avoid boundary values in test data unless the boundary itself is what you are testing.

---
