# Technical Design

**Status:** Frozen before coding · **Owners:** M1 (detection), M2 (physics/inversion), M4 (attribution) · **Last updated:** 2026-09-05

This document is the scientific specification. [architecture.md](architecture.md) says how the code is arranged; this says what it computes and why that is the correct thing to compute.

---

## 0. The three claims this project rests on

Everything else is engineering. These three are the reasons we win or lose.

1. **Look-alike rejection is a physics problem, not a capacity problem.** The information separating an oil slick from a low-wind patch is *not in the image*. We inject it from ERA5 and we abstain when physics says no answer is possible.
2. **Backward drift is not the inverse of forward drift.** Advection reverses; turbulent diffusion does not. We only ever run physics forward, and recover the source by conditioning.
3. **A vessel's AIS track is a generative hypothesis, not a set of features.** We do not score vessels with invented weights. We simulate what each vessel *would have produced* and ask which prediction matches the satellite.

---

## 1. Detection

### 1.1 Product and polarisation

**GRD, not SLC.** We need amplitude, not phase — no interferometry, no polarimetric decomposition. GRD is multi-looked, smaller and faster. IW mode, 10 m pixel spacing.

**VV, not VH.** Bragg scattering from the sea surface is a co-pol phenomenon and it is what oil damps. VH sits near the noise-equivalent sigma-0 over low-backscatter water — which is precisely our regime — so the slick-versus-sea contrast collapses into noise. We keep VH only as a noise-floor sanity channel.

### 1.2 Preprocessing (R3; prototype consumes pre-calibrated data)

```
apply orbit file → thermal noise removal → radiometric calibration to sigma-0
→ speckle filter (refined Lee 7x7) → land/coast mask → convert to dB → tile 256x256
```

**Terrain correction is skipped** — the ocean is flat, there is no topographic distortion to correct. **But we still geocode:** Ellipsoid Correction (Range-Doppler at 0 m height) using the GRD GCP grid. Saying "we skip terrain correction" without this second clause invites the question *"then how are your polygons in lat/lon?"*, and it is a fair question.

### 1.3 Segmentation

**U-Net with a ResNet34 ImageNet-pretrained encoder**, via `segmentation_models_pytorch`.

- Not YOLO — slicks are amorphous regions with no meaningful bounding box.
- Not SegFormer — transformers are data-hungry, we have ~2,000 labelled scenes, and we cannot afford the ablation study to prove it underperformed.
- ImageNet pretraining transfers surprisingly well to single-channel SAR. Replicate sigma-0 across three channels, or re-initialise `conv1` by summing the RGB weights.

**Loss: Dice + Focal.** Oil pixels are a tiny fraction of any scene. Plain cross-entropy will happily predict "all sea" and report 99% pixel accuracy. **We report IoU on the oil class specifically and never overall accuracy** — quoting overall accuracy on this task is a tell that you have not understood the problem.

**Augmentation:** flips, rotations, and **multiplicative** speckle noise. SAR speckle is multiplicative; adding Gaussian noise instead is a detail a remote-sensing scientist will notice immediately.

**Split protocol: hold out entire geographic regions, not random tiles.** Adjacent tiles from the same scene share speckle statistics and sea state; a random split leaks and inflates your numbers. Train on regions A+B, test on C, **and report the drop.**

**Target: IoU 0.50–0.65 on the oil class.** Published SOTA on the standard benchmark is U-Net ~0.54, DeepLabv3+ ~0.53. If we report 0.95 we have overfit, and a judge who knows the literature will say so. *"At published state of the art, with a 10-point cross-region drop we also report"* is a stronger claim than a fake 0.95.

Stage 1 is tuned for **recall**, not F1. Precision is Stage 2's job.

### 1.4 The physics gate — differentiator #1

Everyone else will attack look-alikes with a bigger CNN. **The classes are not separable in image space**, because a low-wind zone and an oil slick genuinely look identical on a radar image.

The physics: oil damps short capillary-gravity waves, suppressing Bragg scattering, so it appears dark. But that only produces contrast in a window:

| 10 m wind | Sea state | Consequence |
|---|---|---|
| **< ~3 m/s** | Surface already smooth; everything is dark | **No contrast is physically possible.** A dark patch here is far more likely a low-wind artefact than oil |
| **~3–10 m/s** | Bragg waves present; oil damping produces real contrast | **The detectability window** |
| **> ~10–12 m/s** | Wave action disperses and submerges the slick; roughness rebuilds | Coherent surface slicks unlikely to persist |

**Implementation.** Sample ERA5 10 m wind at each candidate patch. Outside [3, 12] m/s the patch is returned as `undetermined — outside detectability window`, **as a hard override that the classifier cannot overturn**. Inside the window, LightGBM decides.

**Feature set (~14):**

| # | Feature | Rationale |
|---|---|---|
| 1 | ERA5 wind speed at centroid | The gate itself |
| 2 | Wind-speed spatial gradient across the patch | A wind front produces a dark region with a soft wind-aligned edge |
| 3 | Damping ratio: sigma0(slick) − sigma0(background), dB | The direct physical measurement |
| 4 | **Damping ratio normalised for incidence angle** | Backscatter falls steeply across the swath. Failing to normalise makes near-range and far-range patches incomparable — this alone will corrupt a naive classifier |
| 5 | Mean sigma-0 inside | |
| 6 | Std sigma-0 inside | Oil is more homogeneous than a wind shadow |
| 7 | Mean boundary gradient magnitude | Oil boundaries are sharper than low-wind transitions |
| 8 | Inside/outside gradient ratio | Edge sharpness, scale-free |
| 9 | Area (km²) | |
| 10 | Elongation (major/minor axis) | Vessel discharges are linear; biogenic films are amorphous |
| 11 | Complexity, P²/(4πA) | |
| 12 | Solidity, area / convex-hull area | |
| 13 | Component count | Fragmentation |
| 14 | Distance to nearest shipping lane (AIS density raster) | Contextual prior |

**Classifier: LightGBM, three-way — `oil` / `look-alike` / `undetermined`.** GBDT over a neural net here because with 14 features it wins, trains in seconds, and produces feature importances that go straight onto a slide.

**Adding `undetermined` is the maturity signal.** An operational system that abstains under low wind is more trustworthy than one that always answers. Target abstention 5–15%: below that we are not really gating, above that we are useless.

---

## 2. Characterisation

Extract only what feeds a downstream decision. Features computed to pad a slide are cut.

| Feature | Feeds | Why it matters |
|---|---|---|
| Area, perimeter | Volume estimate, ensemble particle count | Scale of release |
| **Elongation + orientation** | **Release-mode inference** | A long thin slick aligned with a track means *continuous discharge under way*, which constrains t0 to an **interval, not an instant**. This is real information that materially narrows the inversion |
| Fragmentation | Weathering state, diffusivity prior | Fragmented slicks are older and more wave-worked |
| Complexity | Turbulence exposure | Feeds the K_h prior |
| Solidity | Shape regularity | Look-alike feature |
| Damping-ratio statistics | Gate features, thickness hint | Direct physical measurement |
| **Centroid + boundary polygon** | **The inversion target** | This *is* the observation the inversion conditions on |

### Release-mode inference

```
if elongation > 4 and |orientation − nearest_track_heading| < 15°:
    mode = "continuous"      # t0 prior = an interval, width from length / vessel speed
elif elongation < 2.5 and n_components == 1:
    mode = "instantaneous"   # t0 prior = a narrow bump
else:
    mode = "indeterminate"   # t0 prior = uniform over the lookback window
```

Crude, interpretable, and it changes the answer. That is the bar for inclusion.

### Age — what we will and will not claim

**We will not output an age in hours from a single SAR scene.** Damping ratio confounds film thickness, oil type, wind, incidence angle and sea state simultaneously, and one image cannot decouple them.

**What we do instead is stronger:** the inversion produces a posterior over **t0** directly. *The width of that posterior is our statement about age, and it comes with error bars.* We do not estimate age and then drift; we drift and thereby infer age.

Prepared answer: *"Not feasible as a direct radiometric inversion from a single scene — so we recover it as a by-product of the source-time posterior, with quantified uncertainty. The PS says 'if feasible'; our answer is that the radiometric route isn't, but the dynamical route is."*

---

## 3. The transport kernel

One implementation, used by the proposal pass, the inversion, the forecast and the attribution engine.

### 3.1 Governing equation

```
dx = [ u_cur(x,t) + alpha * u_wind10(x,t) ] dt  +  sqrt(2 * K_h * dt) * xi
```

- `u_cur` comes from **CMEMS SMOC**, which already merges geostrophic + tidal + **Stokes drift** into `uo`/`vo`. CMEMS explicitly recommends SMOC for Lagrangian drift applications, which is a citable justification for the choice.
- **Therefore we do NOT add a separate Stokes parameterisation.** Doing so double-counts a real physical term and biases every trajectory downwind. This is gated on `ForcingBundle.currents.includes_stokes` so it cannot be got wrong by forgetting.
- `alpha` = windage, the fraction of 10 m wind imparted to surface oil. Literature range 1–4%. **Sampled per particle, not fixed** — treating it as a constant understates uncertainty by a large factor.
- `K_h` = horizontal diffusivity, order 1–10 m²/s. **Sampled per particle**, log-uniform.
- `xi` = standard 2-D Gaussian white noise.

### 3.2 Numerics

- **Operator splitting:** RK4 for the deterministic advection, Euler–Maruyama for the stochastic term. Standard, correct to the order that matters here.
- **Timestep 15 min.** With currents at O(1 m/s) that is ~900 m per step against a ~9 km grid cell — comfortably resolved.
- **Interpolation:** bilinear in space, linear in time. Precomputed into a single float32 array in memory; no repeated NetCDF reads inside the loop.
- **Projection:** integrate in the case's local UTM (metres). Converting m/s to degrees per step inside the loop is a classic source of latitude-dependent error.
- **Land:** particles intersecting the land mask are marked `beached` and frozen. **Never deleted** — deletion silently biases the posterior away from the coast, which is exactly where spills matter most.
- **Vectorised:** all 1e5 particles advance as one NumPy array operation per step. 192 steps × 4 RK4 stages is a few seconds. A per-particle Python loop is minutes, and it is the difference between an interactive demo and a slideshow.

### 3.3 Why Lagrangian and not Eulerian

An Eulerian advection–diffusion solve on a fixed grid suffers numerical diffusion that would **artificially inflate our uncertainty envelope** — we would be reporting solver error as physics. More fundamentally, it cannot carry per-particle origin labels, and the entire inversion depends on knowing where each parcel came from.

### 3.4 Backward mode

Run with `dt < 0`. **Legitimate only with diffusion disabled** (see §4.1). The kernel refuses to run backward with `K_h > 0`; this is an assertion in code, not a convention.

---

## 4. Source inversion — the scientific core

### 4.1 Why naive backward drift is wrong

- **Advection is time-reversible.** It is a deterministic ODE. Integrate with negative dt and you recover the path.
- **Turbulent diffusion is not.** It is entropy-increasing. Running it backwards is ill-posed — mathematically the same error as un-stirring milk from coffee.

Run a particle model backwards with diffusion on and you get a spreading cloud that *looks* like an uncertainty envelope but is a forward diffusion process pointed backwards in time. It is not a posterior. **This is the single most likely way a competent-looking team gets destroyed on this problem statement, and most teams will do exactly this.**

### 4.2 What we do: Bayesian source inversion by forward ensemble

We only ever run physics **forward**, and recover the source by conditioning.

```
1. PROPOSAL    Backward ADVECTION-ONLY pass (diffusion off — legitimate,
               because pure advection is reversible) from the observed mask.
               Yields a coarse candidate region R(t0). Cheap: seconds.

2. HYPOTHESIS  Partition R x [t_obs − T_max, t_obs] into a grid of source
   GRID        hypotheses h = (x0 cell, t0 bin). Typically ~4,096 hypotheses.

3. PROPAGATE   Seed M particles per hypothesis, all tagged with origin_marker = h.
               Run ONE forward simulation with full stochastic physics.

4. CONDITION   For each h, compute p(observed mask | h) — see §4.3.

5. POSTERIOR   p(h | obs) ∝ p(obs | h) * prior(h).  KDE-smooth to a 3-D
               density over (lat, lon, t0). Extract credible regions.
```

> **Backward integration narrows the search. Forward simulation computes the answer.**

That sentence is the intellectual core of the project. It is defensible under expert questioning and it is the reason we survive a panel containing someone who models the ocean for a living.

### 4.3 The observation operator — and a correction to the obvious approach

**The naive version, which we reject.** The tempting implementation is per-particle rejection ABC: *accept particle i if its position at t_obs falls inside the observed slick polygon; the origins of accepted particles are the posterior.*

This is subtly but importantly wrong. It computes `p(x0, t0 | one oil parcel ended up somewhere in the slick)`. That is **not** `p(source | observed slick shape)`. It has two failure modes:

1. **It never penalises false coverage.** A hypothesis whose particles smear across the entire scene gets exactly the same per-particle credit as one producing a compact cloud that matches the slick precisely — because only the hits are counted and the misses cost nothing.
2. **It discards all shape and extent information** — the very quantities Stage 3 just spent effort computing.

**What we do instead: a hypothesis-level likelihood over the whole AOI.**

For each hypothesis `h`, rasterise its particle cloud at `t_obs` onto the AOI grid as a density `rho_h(x)`, then convert to a predicted oil-presence probability:

```
q_h(x) = 1 − exp( −lambda * rho_h(x) )
```

`lambda` is a single detectability constant, calibrated once so that a physically plausible particle density maps to `q ≈ 0.9`. Then, against the observed binary mask `m(x)`:

```
log L(h) = SUM over all AOI pixels [ m(x) * log q_h(x) + (1 − m(x)) * log(1 − q_h(x)) ]
```

This is a Bernoulli likelihood over the whole scene. **Predicted oil where the satellite saw none now costs you**, which is exactly the property the naive version lacks. And critically, it is *the same operator* used for vessel attribution in §5, so blind hypotheses, named vessels and the dark hypothesis are all scored on one comparable scale.

**It also has a closed form that makes it cheap.** Evaluated literally over every AOI pixel for every hypothesis this would be ~4×10⁹ operations per case. It does not need to be. Since `log(1 − q_h(x)) = −lambda * rho_h(x)`, the negative term collapses to a count:

```
SUM over x NOT in mask [ log(1 − q_h(x)) ]  =  −lambda * (particle mass landing outside the mask)
```

so the whole likelihood reduces to

```
log L(h) = SUM over MASK pixels [ log(1 − exp(−lambda * rho_h)) ]  −  lambda * (mass outside mask)
```

which costs O(mask pixels + particles) per hypothesis instead of O(all pixels) — about 2 s instead of 40 s, with **no approximation**; it is algebraically identical.

This also makes the physics legible in one line: **reward for covering the observed slick, minus a linear penalty for every particle predicted where the satellite saw nothing.** See [performance_review.md](performance_review.md) §3.1.

*Cheap fallback if calibration of `lambda` proves fiddly:* soft-IoU between `q_h` and `m`, used as a log-likelihood surrogate. Less principled, same penalisation property, zero tuning.

**Boundary uncertainty.** The mask is not exact. Dilate `m` by the segmentation boundary uncertainty (~1–2 px plus geolocation error) and treat pixels in that annulus as unobserved rather than as negatives, so we do not punish a hypothesis for a one-pixel disagreement.

### 4.4 Efficiency — the real engineering risk

With a blind prior over a 60 km × 60 km × 48 h volume, the fraction of hypotheses with meaningful likelihood is tiny and most compute is wasted. Three mitigations, in order of application:

1. **The backward-advection proposal** (step 1) is the primary one — it concentrates the hypothesis grid where mass actually is. Expect it to raise the useful fraction from <1% to 5–20%.
2. **Sequential refinement (ABC-SMC-lite).** Run a coarse grid, keep the top decile by likelihood, subdivide, re-run. Two rounds is usually enough and costs less than one fine-grained blind pass.
3. **Report `effective_sample_size`.** If it collapses, the posterior is Monte Carlo noise and **the UI must say so** rather than render a confident-looking blob. Detecting our own failure is worth more than hiding it.

### 4.5 Uncertainty model

**Monte Carlo ensemble with parameter perturbation** — not linearised covariance propagation. The dynamics are nonlinear and the posterior is non-Gaussian, so a covariance would be quietly wrong. Monte Carlo assumes nothing and is trivially parallel.

Sampled **per particle**:

| Source | Distribution | Rationale |
|---|---|---|
| Slick boundary | Perturb seed positions by segmentation boundary uncertainty | Mask edges are uncertain |
| Windage `alpha` | U(0.01, 0.04) | Literature range for surface oil |
| Diffusivity `K_h` | LogU(1, 10) m²/s | Order-of-magnitude uncertainty |
| Current field | Correlated noise scaled to CMEMS QUID RMS error | **Model error, not just resolution** — most teams ignore this entirely |
| Release time `t0` | Prior from release-mode inference (§2) | The unknown we solve for |

### 4.6 How uncertainty actually grows backwards

A judge may probe this, so be precise:

1. **Shear dispersion dominates.** Two particles a few hundred metres apart sit in slightly different current cells and separate roughly *exponentially* at short lags, then diffusively at longer ones. Backward uncertainty is therefore **super-linear in lookback time**, not linear.
2. **Current-field resolution sets a floor.** At 1/12° (~9 km) sub-mesoscale eddies are unrepresented. This is irreducible, not a tuning problem.
3. **Windage uncertainty** (1–4%) integrates directly into position error, growing linearly with time.
4. **Time uncertainty couples into space.** ±6 h of t0 uncertainty in a 0.5 m/s current is ±10 km on its own.

**Practical consequence, stated before a judge states it to us:** beyond roughly **48–72 hours** of lookback the envelope becomes too large to be operationally useful. We report the horizon as a system limit. Being the team that publishes its own operating envelope is worth more than five points of IoU.

### 4.7 Reporting — the honest object

Output **credible regions**, never a point and never a bare percentage:

- **50% credible region** — smallest area containing 50% of posterior mass
- **95% credible region** — the operational search envelope
- **t0 marginal** — posterior over release time alone, with its 95% HPD width in hours
- **Area in km² of each region** — the honest measure of how much we actually narrowed things

The defensible statement:

> *"The source lies within this 4,210 km² region, during this 14-hour window, with 95% posterior probability **under our stated model assumptions**."*

That final clause is not weasel wording. It is the difference between a scientific claim and a marketing claim, and an NTRO evaluator will register that we know the difference.

---

## 5. Attribution — vessel tracks as generative hypotheses

### 5.1 Why a weighted score is indefensible

The obvious approach — `score = w1*proximity + w2*gap + w3*anomaly` — dies on one question: **"where did w1 come from?"** There is no answer. No labelled spill-to-vessel attribution dataset exists, so any weights not learned from labels are invented.

### 5.2 Why "track integral through the posterior" is better, but still a proxy

An improvement is to integrate the posterior density along the vessel's path: `L(v) = ∫ p(x_v(t), t) dt`. This has no free weights, which is a genuine advance.

But it is still a **proxy**, not a likelihood. `p(x0, t0 | obs)` is a marginal over point sources; a real discharge from a moving vessel is a **line source in space-time**, and evaluating a point-source marginal along a line does not give you `p(obs | vessel v)`. It also inherits whatever smoothing the KDE applied.

### 5.3 What we do: vessel-conditioned forward simulation

The key realisation: **a continuous discharge from a moving vessel is a line source in space-time whose parameters we already have — from AIS.** We do not need to invert for it. We can simulate it directly.

For each candidate vessel `v`:

```
1. Interpolate its AIS track to the transport timestep over the
   candidate discharge window W.
2. Seed particles ALONG that track, at the vessel's actual positions
   and actual times, tagged origin_marker = v.
3. Forward-simulate with full stochastic physics to t_obs.
4. Rasterise to q_v(x) and evaluate the SAME Bernoulli likelihood
   from §4.3 against the observed mask:
       log L(v) = SUM_x [ m log q_v + (1−m) log(1−q_v) ]
5. Maximise (or marginalise) over the discharge window W, and report
   the best-fitting window as evidence.
```

**All candidate vessels are seeded into a single forward run**, distinguished by `origin_marker`. ~20 vessel hypotheses cost one simulation, not twenty.

Why this is materially better than §5.2:

| | Track integral | Vessel-conditioned simulation |
|---|---|---|
| Object computed | A proxy score | `p(obs \| H_v)` — an actual likelihood |
| Uses slick shape | ✗ discarded | ✓ full spatial match |
| Penalises over-prediction | ✗ | ✓ |
| Explains elongated track-aligned slicks | ✗ awkward | ✓ naturally — that is what a line source produces |
| Comparable to the dark hypothesis | Only loosely | ✓ identical operator, identical scale |
| Cost | Cheap | One extra forward run |

We still compute the blind inversion of §4 — it delivers PS clause (b), it produces the search envelope, and it is what the dark-vessel hypothesis is built from. **Clause (b) is answered by blind inversion; clause (c) is answered by vessel-conditioned simulation.**

### 5.4 The dark-vessel hypothesis — non-negotiable

```
                       L(v) * pi(v)
P(v | obs) = ─────────────────────────────────────────
             SUM_u [ L(u) * pi(u) ]  +  L_dark * pi_dark
```

`H_dark`: *the source was a vessel not present in AIS.* Its likelihood is computed by the identical operator — seed the blind proposal region, **excluding tubes around observed tracks**, forward-simulate, evaluate. `pi_dark` is a prior on AIS non-compliance in the region (we use 0.10–0.20 and **show the sensitivity**, because it is an assumption, not a measurement).

Without this term, a system normalising only over observed vessels will **confidently name an innocent ship** whenever the true polluter had AIS off. With it, the system can output *"most probable explanation: a vessel not transmitting AIS"* — a correct, operationally valuable answer that no competing team will produce.

### 5.5 Behavioural prior `pi(v)`

Few and interpretable. Each factor is displayed individually in the UI; none is hidden inside a sum.

| Factor | Handling | The trap |
|---|---|---|
| **AIS gap** overlapping the posterior time window | Modelled **relative to the local baseline gap rate** in that area | Gaps are extremely common for benign reasons — shore-receiver coverage. Using an absolute threshold manufactures suspicion out of poor coverage. This is the single most likely way to produce a confident wrong answer |
| **Speed anomaly** | Deviation from the vessel's own transit median | Slowing is consistent with discharging, but also with weather, traffic and pilotage |
| **Course change** near the posterior region | Bearing change rate | Weak on its own |
| **Vessel type** (AIS static) | Tanker / bulk carrier prior above passenger | Coarse but real |
| **Track–slick axis alignment** | Angle between slick major axis and vessel heading | Physically motivated and strong: a continuous discharge leaves a slick *along* the track |

### 5.6 Why not learning-to-rank or a GBDT here

Because **there is no training label.** Any supervised ranker would be trained on synthetic data and would learn our own simulator's biases — and a judge will ask exactly that. The Bayesian formulation needs no labels, which converts our data limitation into an architectural advantage.

### 5.7 AIS preprocessing

Real AIS is dirty. Filter hard, **count everything dropped**, and report the counts — an undisclosed filter is a hidden assumption.

| Problem | Handling |
|---|---|
| Duplicate / shared MMSI | Split into separate tracks on implausible jumps |
| Position jumps | Reject fixes implying > 40 kn |
| Irregular intervals | Linear interpolation to a fixed grid; **flag interpolated segments** — never let them silently become evidence |
| Missing static data | Vessel type prior falls back to uninformative |
| Timezone | DMA is UTC. Assert on load, do not assume |

**Spatio-temporal prefilter (FR-13):** keep vessels whose track intersects the 95% credible region's spatial support during the t0 marginal's support, dilated by a safety margin. Report the reduction factor — *214 vessels to 3* is our headline metric, and it is the one number a non-specialist judge will remember.

---

## 6. Forecast (clause b, forward half)

Seed from the observed mask, forward-simulate with the same kernel and the same per-particle parameter sampling, and emit envelopes at +6 / +12 / +24 h. Same uncertainty treatment, same honesty. Cheap: it reuses everything.

---

## 7. Evaluation targets

Full protocol in [testing_strategy.md](testing_strategy.md). Targets:

| Subsystem | Metric | Credible prototype value | Note |
|---|---|---|---|
| Segmentation | **IoU (oil class)** | **0.50–0.65** | Published SOTA ~0.54. Above 0.85 on a fair split is implausible |
| Segmentation | Recall (oil) | > 0.75 | Stage 1 favours recall by design |
| Segmentation | Cross-region drop | report it, ~10 pts | Reporting the drop is worth more than hiding it |
| Physics gate | **False-positive rate** | **< 0.15** | The headline number for Stage 2 |
| Physics gate | Abstention rate | 5–15% | Too low = not gating; too high = useless |
| Transport | Separation vs drifter @ 24 h | 10–25 km | Realistic for a 1/12° field. Anyone claiming < 5 km is not measuring honestly |
| **Inversion** | **True source inside 95% region** | **> 0.90** | *Calibration*, not accuracy. This is the number that proves honesty |
| **Inversion** | 95% region area @ 24 h | 2,000–8,000 km² | The honest measure of narrowing |
| Inversion | t0 marginal width @ 24 h | 8–20 h | |
| Attribution | **Top-3 recall** | **0.60–0.80** | On synthetic ground truth |
| Attribution | Top-1 recall | 0.30–0.50 | **Deliberately modest — top-1 is not the goal** |
| Attribution | **Candidate reduction factor** | **50–100x** | The headline |
| System | Runtime, cached case to ranking | < 60 s | |

### The metric that wins

**Calibration, not accuracy.** If we say 95% and the true source falls inside the 95% region 95% of the time, our uncertainty is *honest*. Almost no student team can make that claim and every serious evaluator respects it. We plot nominal versus empirical coverage and put it on a slide.

---

## 8. Prototype scope boundary

| Component | Prototype (Round 1) | Round 3 |
|---|---|---|
| SAR ingest | Pre-calibrated Zenodo tiles | Full SNAP chain from raw GRD |
| Segmentation | U-Net, ~2k tiles, one GPU | Larger training set, ensembling, cross-sensor |
| Physics gate | Hard wind gate + LightGBM (14 feat.) | Add sea-state, biogenic-film seasonality, SST fronts |
| Transport | Custom RK4 kernel, no weathering | OpenDrift / OpenOil with ADIOS weathering |
| Inversion | 2-round refinement, ~4k hypotheses, 1e5 particles | Adaptive SMC, 1e6+ particles, GPU |
| Attribution | Vessel-conditioned sim + dark term | + repeat-offender aggregation across spills |
| Ship detection | CFAR (Tier B) | Trained detector, AIS-matched dark-vessel flagging |
| Validation | Synthetic truth + segmentation holdout | + GDP drifter trajectories, external oceanographer review |
| Coverage | 3 prepared cases | Regional, then EEZ-scale |

---

## 9. Known open problems

Recording these is part of the design, not an admission of weakness. Each is a candidate Round 3 contribution.

1. **No end-to-end ground truth exists.** There is no public dataset of confirmed spill-to-vessel attributions. We validate detection, drift and inversion *separately* and are explicit that end-to-end validation is an open problem — and that a system like this is how you would start building that dataset.
2. **Synthetic validation is partly circular.** The simulator generates the observation *and* evaluates candidates, so it tests the inversion machinery, not the physics. That is precisely why drift must be validated separately against real drifter trajectories, with no simulator in the loop.
3. **`lambda` and `pi_dark` are calibrated, not derived.** Both are stated assumptions with sensitivity analysis, not measurements.
4. **Single-scene only.** Multi-temporal linking across passes would tighten t0 substantially and is the highest-value Round 3 addition.
5. **Sub-mesoscale is unresolvable** at 1/12°. Higher-resolution regional models (CMEMS NW Shelf at 1.5 km) would help in specific regions but not globally.
