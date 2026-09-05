# SIH26143 — Technical Solution Design
### Oil spill detection + AIS vessel attribution · NTRO · Software
**IIIT Kottayam · 6-member CSE / AI-DS team · v1.0**

---

## 0. SOURCE OF TRUTH AND VERIFICATION LEDGER

The official PS requires an automated pipeline that:
- **(a)** detects and characterises the slick, computing geometric properties and *"age if feasible"*
- **(b)** uses oceanographic and meteorological data to *"trace the slick towards the origin point and time"* and predict future flow
- **(c)** attributes the spill using historic AIS to *"reconstruct vessel traffic around the **origin window in space and time**"*, filter irrelevant traffic, and score suspects on *"proximity, trajectory, behavioural anomalies etc."*
- Delivers a *"suitable visual interface"*

**Read clause (c) again.** The PS itself says **"origin window in space and time"** — not origin point. The uncertainty-envelope framing is not us softening the problem to protect ourselves; it is the literal wording of the requirement. Clause (b) says "point", clause (c) says "window". When those conflict, (c) is the operationally meaningful one, and we implement (c). **This single observation defends our entire design choice against the accusation of under-delivering.**

The PS also explicitly names our data sources: marinecadastre.gov AIS (for format), real AIS *"if available, else synthetic data can be prepared"*, and the Zenodo Sentinel-1 SAR oil spill dataset.

### What I verified this session (Aug 2026)

| Item | Status | Detail |
|---|---|---|
| OpenDrift / OpenOil backtracking | ✅ **VERIFIED** | Native forward *and* backward simulation. OpenOil uses NOAA ADIOS oil database + PyGnome weathering code |
| Sentinel-1 constellation | ✅ **VERIFIED** | **Sentinel-1A terminated 29 June 2026.** Final config = **1C + 1D**, 6-day nominal revisit. 1D open access since 17 Apr 2026 |
| Zenodo SAR oil dataset | ✅ **VERIFIED** | Trujillo-Acatitla et al., 3 parts, openly downloadable, no request needed. Sigma0 dB, TIFF, with explicit look-alike class |
| Krestenitis benchmark | ⚠️ **RESTRICTED** | The standard benchmark (1112 img, 5 classes) is **not open** — requires a request from a supervisor's institutional email. Start this now; it has lead time |
| CMEMS SMOC | ✅ **VERIFIED** | `cmems_mod_glo_phy_anfc_merged-uv_PT1H-i` — currents + tides + Stokes drift, hourly, 1/12°. **CMEMS explicitly recommends it for drift/Lagrangian applications** |
| Free non-US historical AIS | ✅ **VERIFIED** | Danish Maritime Authority (`web.ais.dk/aisdata/`, CSV) and Norwegian Coastal Administration (`ais-public.kystverket.no`, NLOD licence) |
| Published SAR segmentation SOTA | ✅ **VERIFIED** | U-Net **IoU ≈ 0.54**, DeepLabv3+ **≈ 0.53** on the oil class (Krestenitis benchmark) |
| Global Drifter Program buoy data | ⚠️ **UNVERIFIED** | I did not confirm access this session. Verify before relying on it — it underpins our drift validation |

> **Sentinel-1A is dead.** Any team whose slides say "Sentinel-1A and 1B" is displaying knowledge that is 4½ years stale (1B failed Dec 2021) and 2 months stale (1A terminated June 2026). Say **1C/1D**. It costs nothing and signals you actually read current mission documentation.

---

## 1. THE SIXTEEN SUBPROBLEMS — REAL DIFFICULTY

No softening. Difficulty is rated for *our* team, not for a research lab.

| # | Subproblem | Real scientific difficulty | Verdict |
|---|---|---|---|
| 1 | SAR slick detection | **Moderate.** Semantic segmentation on a well-studied benchmark. Published SOTA is only IoU ~0.54 — the task is genuinely hard, but the *engineering* is standard | Tractable |
| 2 | Characterisation | **Low-moderate.** Geometry from a binary mask is classical image analysis. The hard part is choosing features that are physically meaningful downstream | Tractable |
| 3 | Look-alike discrimination | **HIGH — this is the real detection problem.** Low-wind zones, biogenic films, rain cells and wakes are all radar-dark. Pure appearance-based ML plateaus here because the classes are genuinely ambiguous in image space | **Needs physics** |
| 4 | Age estimation | **VERY HIGH — bordering on not-well-posed from a single SAR scene.** Damping ratio depends on film thickness, oil type, wind, incidence angle and sea state simultaneously. The PS says *"if feasible"* — that is the PS telling you it may not be | **Constrain, don't claim** |
| 5 | Ocean/met integration | **Low technically, high in gotchas.** CRS alignment, time interpolation, NetCDF conventions, staggered grids, land masking | Tractable |
| 6 | Forward prediction | **Moderate.** Solved by OpenOil. The difficulty is parameter choice (oil type, release rate), not the solver | Tractable |
| 7 | Backward hindcasting | **HIGH — and the naive approach is scientifically wrong.** See §5. Advection reverses; turbulent diffusion does not | **Core innovation** |
| 8 | Source location/time | **HIGH.** This is an *inverse problem* with a non-unique solution. Many (position, time) pairs explain the same observed slick | **Core innovation** |
| 9 | Uncertainty propagation | **HIGH.** Errors compound multiplicatively backwards. Doing this properly is what separates a defensible system from theatre | **Core innovation** |
| 10 | AIS reconstruction | **Low-moderate.** Volume and messy real-world data (duplicate MMSI, position jumps, irregular intervals) | Tractable |
| 11 | Irrelevant-traffic filtering | **Moderate.** Cheap if done spatially; the subtlety is not over-filtering the actual culprit | Tractable |
| 12 | Trajectory correlation | **Moderate-high.** Requires a principled distance measure between a track and a *probability field*, not between two points | **Design carefully** |
| 13 | Behavioural anomaly | **Moderate.** AIS gaps, speed/course anomalies. Danger: gaps are extremely common for benign reasons (receiver coverage) | **Beware false signal** |
| 14 | Suspect ranking | **HIGH if done honestly.** Trivially easy to produce an arbitrary weighted score. Hard to produce a number that means something | **Core innovation** |
| 15 | Evidence generation | **Low technically, high in judgment.** What constitutes defensible evidence vs. an accusation | Tractable |
| 16 | Visualisation | **Moderate.** Rendering 10⁵ animated particles + AIS tracks in a browser needs GPU-accelerated layers | Tractable |

**The honest summary:** subproblems 1, 2, 5, 6, 10, 11, 15, 16 are engineering. Subproblems 3, 7, 8, 9, 14 are science — and they are where the project is won or lost. **Subproblem 4 (age) is where teams will overclaim and get destroyed.**

---

## 2. THE CENTRAL DESIGN DECISION

### Why naive backward drift is wrong

Oil transport is advection + turbulent diffusion:

```
dx/dt = u_total(x, t) + √(2K) · η(t)
```

where `u_total` = currents + tides + Stokes drift + windage, `K` is horizontal diffusivity, `η` is white noise.

- **Advection is time-reversible.** It's a deterministic ODE; integrate with negative dt and you recover the path.
- **Turbulent diffusion is NOT time-reversible.** It is entropy-increasing. Running it backwards is an ill-posed problem — mathematically the same error as trying to un-stir milk from coffee.

So if you run OpenDrift backwards *with diffusion enabled*, you get a spreading cloud that **looks** like an uncertainty envelope but is not a correct posterior. It is a forward diffusion process pointed backwards in time. A physical oceanographer on the judging panel will know this immediately.

**This is the single most likely way a competent-looking team gets destroyed on this PS.** Most teams will do exactly this.

### What we do instead: Bayesian source inversion by forward ensemble

We only ever run the physics **forward**, and recover the source by conditioning.

```
1. PRIOR      Sample N candidate sources (x₀, t₀) over a region and time window
2. PROPAGATE  Run ONE OpenOil simulation forward to observation time t_obs,
              with full stochastic physics (advection + diffusion + weathering)
3. CONDITION  Keep only particles whose final position lies inside the
              observed slick mask
4. POSTERIOR  The ORIGINS of the accepted particles ARE samples from
              p(x₀, t₀ | observed slick)
```

This is **Approximate Bayesian Computation** by rejection sampling. It is exact up to Monte Carlo error, it needs one forward run, and it is correct precisely *because* it never runs diffusion backwards.

**The efficiency trick:** a blind prior wastes almost every particle. So we use a cheap **advection-only backward run** (diffusion off — legitimate, because pure advection *is* reversible) to find the region worth sampling, then use that as the proposal distribution for the forward ensemble.

> **Backward integration narrows the search. Forward simulation computes the answer.**

That sentence is the intellectual core of this project. It is defensible under expert questioning, it is genuinely novel at student level, and it is the reason we survive a panel that includes someone who models the ocean for a living.

---

## 3. END-TO-END ARCHITECTURE

```
┌─── INGEST ──────────────────────────────────────────────────────┐
│  Sentinel-1 GRD IW (Copernicus Data Space)                      │
│  CMEMS SMOC — currents+tides+Stokes, hourly 1/12°               │
│  ERA5 10m wind (CDS API)                                        │
│  AIS — MarineCadastre / Danish DMA / Norwegian Kystverket       │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─── SAR PREPROCESS (SNAP / pyroSAR) ─────────────────────────────┐
│  orbit file → thermal noise removal → radiometric calibration   │
│  → σ⁰ → speckle filter (refined Lee 7×7) → land mask → dB       │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─── STAGE 1: DARK-PATCH SEGMENTATION ────────────────────────────┐
│  U-Net (ResNet34 encoder, ImageNet pretrained) → dark polygons  │
│  Deliberately HIGH RECALL. Precision is Stage 2's job.          │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─── STAGE 2: PHYSICS-GATED LOOK-ALIKE REJECTION ★ ───────────────┐
│  ERA5 wind at patch → detectability window gate (3–10 m/s)      │
│  + damping ratio normalised by incidence angle                  │
│  + shape/edge-gradient features → GBDT classifier               │
│  OUT: oil | look-alike | undetermined  (+ reason string)        │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─── CHARACTERISATION ────────────────────────────────────────────┐
│  area, perimeter, elongation, orientation, fragmentation,       │
│  complexity, damping stats → release-mode inference             │
│  (continuous discharge vs instantaneous) → constrains t₀ prior  │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─── STAGE 3: SOURCE INVERSION ★★ (the core) ─────────────────────┐
│  (i)  backward advection-only run → proposal region             │
│  (ii) forward OpenOil ensemble, N≈10⁵ particles, full physics   │
│  (iii) reject particles not landing in observed mask            │
│  (iv) KDE over accepted origins → posterior p(x₀,t₀ | obs)      │
│  OUT: 3-D posterior density over (lat, lon, time)               │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─── STAGE 4: AIS ATTRIBUTION ★★★ ────────────────────────────────┐
│  spatio-temporal prefilter → track reconstruction               │
│  → TRACK INTEGRAL through posterior field                       │
│  → behavioural prior (AIS gap, speed/course anomaly, type)      │
│  → normalise WITH explicit "dark vessel" hypothesis             │
│  OUT: ranked candidates + per-vessel evidence                   │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─── EVIDENCE + INTERFACE ────────────────────────────────────────┐
│  FastAPI · PostGIS/DuckDB · React + deck.gl · evidence PDF      │
└─────────────────────────────────────────────────────────────────┘
```

★ = differentiator we implement

### Component decisions — build vs. reuse

| Component | Decision | Why |
|---|---|---|
| SAR preprocessing | **REUSE** — ESA SNAP / pyroSAR | Writing a calibration chain is weeks of work with zero innovation credit |
| Segmentation model | **BUILD** (train) on `segmentation_models_pytorch` | This is our ML contribution. Architecture reused, training ours |
| Look-alike discriminator | **BUILD** ★ | This is differentiator #2. Nobody ships this |
| Ocean/wave/wind models | **REUSE** — CMEMS, ERA5 | Building an ocean model is a PhD, not a hackathon |
| Drift solver | **REUSE** — OpenDrift/OpenOil | Peer-reviewed, MET Norway. Rewriting it is negative value |
| Weathering | **REUSE** — OpenOil/ADIOS | Same |
| **Source inversion** | **BUILD** ★★ | Differentiator #1. This is the project |
| **Attribution scoring** | **BUILD** ★★★ | Differentiator #3 |
| Frontend | **BUILD** on deck.gl | Explicitly required by the PS |

---

## 4. SAR COMPONENT

### Product and polarisation

**Use GRD, not SLC.** SLC retains phase, which we do not need — we are not doing interferometry or polarimetric decomposition. GRD is smaller, faster, and already multi-looked. IW mode, 10 m pixel spacing.

**Use VV, not VH.** Bragg scattering from the sea surface is far stronger in co-pol. VH is dominated by noise over calm water, which is exactly our regime. Oil damping is a VV phenomenon.

> **Prepared answer:** *"VH sits near the noise floor over low-backscatter water, so slick-versus-sea contrast collapses. VV carries the Bragg signal that oil damps. We keep VH only as a noise-floor sanity channel."*

### Preprocessing chain

`apply orbit file → thermal noise removal → radiometric calibration to σ⁰ → speckle filter (refined Lee 7×7) → land/coast mask → convert to dB → tile 256×256`

Terrain correction is **not needed** over open ocean (flat surface, no topographic distortion). Skip it and say why — knowing what to skip is as informative as knowing what to run.

### Architecture choice

**U-Net with a ResNet34 ImageNet-pretrained encoder.** Not SegFormer, not YOLO.

- YOLO is object detection; slicks are amorphous regions with no meaningful bounding box.
- SegFormer/transformers are data-hungry and we have ~2,000 labelled scenes. It will underperform and we cannot afford the ablation study to prove otherwise.
- ImageNet pretraining transfers surprisingly well even to single-channel SAR (replicate σ⁰ across 3 channels or re-initialise conv1 by summing weights).

**Set expectations now: published SOTA on the standard benchmark is IoU ≈ 0.54 on the oil class.** If you report 0.95 you have overfit, and a judge who knows the literature will say so. Target **0.50–0.65** and present it as *at published state of the art* — which is a stronger claim than a fake 0.95.

### Look-alike discrimination — the physics gate ★

Everyone will try to solve this with a bigger CNN. **The classes are not separable in image space alone**, because a low-wind zone and an oil slick genuinely look the same. The information that separates them is *not in the image*.

The physics: oil damps short capillary-gravity waves, suppressing Bragg scattering → dark. But:

| Wind regime | What happens | Implication |
|---|---|---|
| **< ~3 m/s** | Sea surface already smooth — everything is dark | **No contrast is physically possible.** A dark patch here is far more likely a low-wind artefact |
| **~3–10 m/s** | Bragg waves present; oil damping produces genuine contrast | **The detectability window** |
| **> ~10–12 m/s** | Wave action disperses and submerges the slick, roughness rebuilds | Coherent surface slicks are unlikely to persist |

So: pull ERA5 10m wind, compute local wind speed at each candidate patch, and **gate**. A dark patch in a 1.5 m/s wind field is reported as *"undetermined — outside detectability window"*, not as oil.

Stage-2 feature set:
- **ERA5 local wind speed** (the gate) — and its spatial gradient across the patch
- **Damping ratio** σ⁰(slick) / σ⁰(background), normalised for incidence angle (backscatter falls steeply across the swath; failing to normalise makes near-range and far-range patches incomparable)
- **Edge gradient sharpness** — oil boundaries are sharper than low-wind transitions
- **Shape descriptors** — elongation, complexity. Vessel discharges are linear and track-aligned; biogenic films are amorphous
- **Shipping-lane proximity** — contextual prior

Classifier: **gradient-boosted trees (LightGBM)** on these ~15 features, not a neural net. With this few features, GBDT beats a NN, trains in seconds, and — critically — **gives you feature importances that you can put on a slide**. Explainability is a scored asset here.

Three-way output: `oil | look-alike | undetermined`. **Adding "undetermined" is a maturity signal.** An operational system that abstains under low wind is more trustworthy than one that always answers.

---

## 5. CHARACTERISATION

Extract only what feeds a downstream decision. Do not compute features to pad a slide.

| Feature | Feeds | Why it matters |
|---|---|---|
| Area, perimeter | Volume estimate, ensemble particle count | Scale of release |
| **Elongation + orientation** | **Release-mode inference** | A long thin slick aligned with a track ⇒ *continuous discharge under way*, which constrains t₀ to an interval, not an instant. This is real information |
| Fragmentation (component count) | Age proxy, weathering state | Fragmented slicks are older / more wave-worked |
| Complexity (perimeter²/area) | Weathering, turbulence exposure | Feeds ensemble diffusivity choice |
| Damping ratio statistics | Look-alike classifier + thickness hint | Direct physical measurement |
| Centroid + boundary polygon | **Inversion target** | The observation the inversion conditions on |

### Age: what we will and will not claim

**We will not output an age in hours from a single SAR scene.** It is not defensible: damping ratio confounds thickness, oil type, wind, incidence angle and sea state simultaneously, and we cannot decouple them from one image.

**What we will do instead** — and this is stronger:

> The inversion produces a posterior over **t₀** directly. The width of that posterior *is* our statement about age, and it comes with error bars. We do not estimate age and then drift; we drift and thereby infer age.

The PS says *"age if feasible."* Our answer: *"Not feasible as a direct radiometric inversion from a single scene — so we recover it as a by-product of the source-time posterior, with quantified uncertainty."* That converts an admission of limitation into a demonstration of understanding.

---

## 6. DRIFT AND INVERSION — THE CORE

### Transport model

```
dx/dt = u_cur(x,t) + u_tide(x,t) + u_stokes(x,t) + α·u_wind10(x,t) + √(2K_h)·η(t)
```

- `u_cur + u_tide + u_stokes` come **pre-merged** from CMEMS **SMOC** (`cmems_mod_glo_phy_anfc_merged-uv_PT1H-i`). CMEMS explicitly recommends SMOC for Lagrangian drift, which is a citable justification for the choice.
- `α` = windage coefficient, ~1–4% of 10 m wind for surface oil. **Treat α as uncertain, not fixed** — sample it per-particle.
- `K_h` = horizontal diffusivity, order 1–10 m²/s.

Use **Lagrangian particle tracking**, not Eulerian. Eulerian advection-diffusion on a fixed grid suffers numerical diffusion that would artificially inflate our uncertainty envelope, and it cannot carry per-particle origin labels — which our whole inversion depends on.

### Forward simulation
Standard OpenOil: seed at source, run to target time, full weathering on.

### Backward hindcast — two-stage

**Stage A — proposal (advection-only, backwards).** Seed particles on the observed slick, run backwards with `diffusion = 0`, `windage = mean`. Pure advection is deterministic and reversible, so this is legitimate. Output: a coarse candidate region `R(t₀)` for each candidate release time. Cheap — seconds.

**Stage B — inversion (forward ensemble, full physics).** Sample N ≈ 10⁵ candidate sources `(x₀, t₀)` from `R` expanded by a safety margin, spanning t₀ over the plausible window (e.g. 6–72 h before observation). **Seed them all in a single OpenOil run** — OpenDrift supports per-element seed times. Run forward to `t_obs`.

**Conditioning.** Accept particle *i* if its position at `t_obs` falls inside the observed slick polygon. To avoid a hard 0/1 cliff, use a soft likelihood — distance to the mask through a Gaussian kernel with bandwidth set by segmentation boundary uncertainty (~1–2 pixels ≈ 10–20 m, plus geolocation error).

**Posterior.**
```
p(x₀, t₀ | obs) ∝ (1/N) Σᵢ w(xᵢ(t_obs)) · δ(x₀ᵢ, t₀ᵢ)
```
Smooth with a KDE → a 3-D density over (lat, lon, time).

### How uncertainty actually grows backwards

Be precise here, because a judge may probe it:

1. **Shear dispersion dominates.** Two particles a few hundred metres apart sit in slightly different current cells and separate roughly *exponentially* early on, then diffusively at longer lags. Backward uncertainty is therefore **super-linear in lookback time**, not linear.
2. **Current-field resolution sets a floor.** At 1/12° (~8–9 km) the model cannot represent sub-mesoscale eddies. Structure below that scale is unmodelled — an irreducible error source.
3. **Windage coefficient uncertainty** (1–4%) integrates directly into position error and grows linearly with time.
4. **Time uncertainty couples to space.** If t₀ is uncertain by ±6 h and the current is 0.5 m/s, that alone contributes ~±10 km.

**Practical consequence, and say this before a judge says it to you:** beyond roughly **48–72 hours** of lookback, the envelope typically becomes too large to be operationally useful. **Report the lookback horizon as a system limit, don't hide it.** Being the team that states its own operating envelope is worth more than an extra 5% IoU.

---

## 7. UNCERTAINTY MODEL

**Method: Monte Carlo ensemble with parameter perturbation.** Not covariance propagation — the dynamics are nonlinear and non-Gaussian, so a linearised covariance would be quietly wrong. Monte Carlo makes no distributional assumption and is trivially parallel.

Sample **per particle**:

| Source | Distribution | Rationale |
|---|---|---|
| Slick boundary | Perturb seed positions by segmentation uncertainty | Mask edges are uncertain |
| Windage α | Uniform(0.01, 0.04) | Literature range for surface oil |
| Diffusivity K_h | LogUniform(1, 10) m²/s | Order-of-magnitude uncertainty |
| Current field | Perturb with correlated noise scaled to CMEMS QUID-reported RMS error | Model error, not just resolution |
| Release time t₀ | Uniform over candidate window | The unknown we're solving for |
| Oil type | Sample over plausible ADIOS types | Affects weathering and windage |

### Reporting — the honest object

Output **credible regions**, not a point and not a fake percentage:

- **50% credible region** — smallest area containing 50% of posterior mass
- **95% credible region** — the operational search envelope
- **Time marginal** — posterior over t₀ alone
- Report **area in km²** of each region. This is the honest measure of how much we actually narrowed things.

The statement we can defend:

> *"The source lies within this 4,200 km² region, during this 14-hour window, with 95% posterior probability under our stated model assumptions."*

Note the final clause. **"Under our stated model assumptions"** is not weasel wording — it is the difference between a scientific claim and a marketing claim, and an NTRO evaluator will register that you know the difference.

---

## 8. AIS ATTRIBUTION ENGINE

### Why a weighted score is indefensible

The obvious approach — `score = w₁·proximity + w₂·gap + w₃·anomaly` — dies on one question: *"where did w₁ come from?"* There is no answer. Any weights that aren't learned from labelled attributions are invented, and no labelled attribution dataset exists.

### What we do instead: track integral through the posterior

We already have `p(x₀, t₀ | obs)` — a probability density over source space-time. A vessel's track is a curve through that same space. So:

```
L(v) = ∫ p(x_v(t), t) dt
```

**The likelihood that vessel v is the source is the posterior mass its track sweeps through.** This is not an invented score — it is a direct evaluation of the density we computed, along the path the vessel actually took. There are no free weights.

Then apply a behavioural prior and normalise:

```
                    L(v) · π(v)
P(v | obs) = ─────────────────────────────
             Σ_u L(u)·π(u)  +  L_dark·π_dark
```

### The dark-vessel hypothesis — non-negotiable

That final term in the denominator is the most important part of the whole design.

If the true polluter had AIS switched off, **it is not in the candidate set at all**, and any system that normalises only over observed vessels will confidently name an innocent ship. So we carry an explicit hypothesis: *"the source was a vessel not present in AIS."* Its likelihood is the posterior mass not covered by any observed track, weighted by a prior on AIS non-compliance.

Consequence: the system can and will output *"most probable explanation: a vessel not transmitting AIS."* **That is a correct and operationally valuable answer**, and no competing team will produce it.

### Behavioural prior π(v)

Keep it interpretable and few:
- **AIS gap** overlapping the posterior time window. *Handle with care* — gaps are extremely common for benign reasons (shore receiver coverage). Model the gap prior **relative to the local baseline gap rate** in that area, not absolutely, or you will manufacture suspicion out of poor coverage.
- **Speed anomaly** — slowing, which is consistent with discharging
- **Course change** near the posterior region
- **Vessel type** from AIS static data (tankers/bulk carriers vs passenger)
- **Track–slick geometric alignment** — does the slick's major axis align with the vessel's heading? A continuous discharge leaves a slick *along the track*, which is a strong, physically motivated signal

### Why not learning-to-rank / GBDT here?

Because **there is no training label**. We have no dataset of confirmed spill→vessel attributions. Any supervised ranker would be trained on synthetic data and would learn our own simulator's biases — and a judge will ask exactly that. The Bayesian formulation needs no labels, which turns our data limitation into an architectural advantage.

---

## 9. DIFFERENTIATORS

| # | Differentiator | Sci. value | Innov. | Feasible | Judge impact | Defensible | Copycat-resistant |
|---|---|---|---|---|---|---|---|
| 1 | **Bayesian source inversion (forward-ensemble ABC)** | 10 | 9 | 7 | 9 | 10 | 9 |
| 2 | **Physics-gated look-alike rejection (wind window)** | 8 | 7 | 9 | 9 | 10 | 7 |
| 3 | **Track-integral scoring + dark-vessel hypothesis** | 9 | 9 | 8 | 10 | 10 | 9 |
| 4 | Drift validation vs. real drifter buoys | 9 | 6 | 8 | 10 | 10 | 6 |
| 5 | Repeat-offender analysis across many spills | 7 | 9 | 7 | 10 | 9 | 8 |
| 6 | Dark-vessel detection (SAR ships ✗ AIS) | 8 | 7 | 5 | 9 | 8 | 7 |
| 7 | Release-mode inference from slick morphology | 7 | 8 | 7 | 7 | 8 | 8 |
| 8 | Incidence-angle-normalised damping | 6 | 4 | 9 | 4 | 9 | 3 |
| 9 | Multi-scene temporal linking | 6 | 6 | 4 | 6 | 7 | 6 |
| 10 | Court-ready evidence bundle | 4 | 6 | 9 | 8 | 8 | 4 |
| 11 | Abstention under low confidence | 7 | 6 | 9 | 8 | 10 | 5 |
| 12 | Oil-type inference from weathering | 5 | 6 | 3 | 5 | 4 | 7 |
| 13 | Shipping-lane prior from AIS density | 6 | 5 | 8 | 5 | 8 | 5 |

### Implement these three

**#1 Bayesian source inversion.** The scientific core. Turns "we ran particles backwards" into "we solved an inverse problem correctly." Highest defensibility of anything on the list.

**#3 Track-integral scoring with the dark-vessel hypothesis.** Makes ranking mathematically principled and produces the one output no other team will have: *"probably a vessel that wasn't transmitting."*

**#2 Physics-gated look-alike rejection.** Cheapest to build, enormous judge impact, and it directly answers the hardest question in SAR oil detection with physics instead of a bigger model.

**Plus #4 as validation** (it's the evidence for #1) **and #5 as the scaling story on the PPT** — a vessel appearing in the top-3 across six independent spills is intelligence, even when no single attribution is conclusive. That framing is how the system becomes operationally valuable despite honest per-spill uncertainty. It is also the single most memorable idea we have.

---

## 10. DATASET STACK

| Dataset | Provider | Access | Resolution | Use | Status |
|---|---|---|---|---|---|
| **Sentinel-1 GRD IW** | ESA / Copernicus | Copernicus Data Space, free registration | 10 m, 6-day revisit (1C+1D) | Inference | ✅ VERIFIED |
| **Zenodo SAR oil spill (Parts I–III)** | Trujillo-Acatitla et al. | Direct Zenodo download, no request | 2048×2048, σ⁰ dB, TIFF | **Training** | ✅ VERIFIED — named in the PS |
| **Krestenitis benchmark** | MKLab / CERTH | **Request via supervisor's institutional email** | 1250×650, 5 classes | Training + benchmark comparison | ⚠️ RESTRICTED — start now |
| **Deep-SAR (SOS)** | Zhu et al. | Public | 256×256, PALSAR + S-1 | Extra training; **Persian Gulf split is Arabian-Sea-relevant** | ⚠️ VERIFY link |
| **CMEMS SMOC** | Copernicus Marine | `copernicusmarine` toolbox, free account | 1/12°, hourly | Drift forcing | ✅ VERIFIED |
| **CMEMS NW Shelf** | Copernicus Marine | Same | **1.5 km** | High-res validation region | ✅ VERIFIED |
| **ERA5 10m wind** | ECMWF / C3S | CDS API, free | 0.25°, hourly | Look-alike gate + windage | ✅ VERIFIED (re-check CDS endpoint) |
| **MarineCadastre AIS** | NOAA / BOEM | Direct download, public domain | US waters, 1-min | AIS — **named in the PS** | ✅ VERIFIED |
| **Danish DMA AIS** | Danish Maritime Authority | `web.ais.dk/aisdata/`, CSV, free | Danish waters | **Non-US real AIS** | ✅ VERIFIED |
| **Norwegian AIS** | Kystverket | `ais-public.kystverket.no`, NLOD | Norwegian waters | Non-US real AIS | ✅ VERIFIED |
| **Global Drifter Program** | NOAA AOML | — | 6-hourly | **Drift validation** | ⚠️ UNVERIFIED — confirm first |
| Indian waters AIS | — | — | — | — | ❌ **No free bulk historical source found.** Use synthetic, as the PS permits |

### The regional strategy this forces

**Validate in Northern European or US waters. Demonstrate transferability to Indian waters.**

Northern Europe gives us free real AIS (Denmark/Norway) *and* a 1.5 km ocean model. That is the only place we can do end-to-end work on entirely real data. The Gulf of Mexico via MarineCadastre is our second real region.

For the Arabian Sea we use global SMOC + synthetic AIS — **which the PS explicitly authorises**. Say this openly on the Feasibility slide. Disclosed synthetic data is a methodological choice; discovered synthetic data is a credibility collapse.

---

## 11. TRAINING STRATEGY

### What is actually trained
- **U-Net segmentation** — the only deep model we train. Fine-tune from ImageNet weights.
- **LightGBM look-alike classifier** — ~15 physical features, seconds to train.

### What is pretrained / reused
ResNet34 encoder (ImageNet), OpenOil weathering (NOAA ADIOS), all ocean/atmosphere models.

### What is rule-based / physics-based
The wind detectability gate, incidence-angle normalisation, all geometric characterisation, the inversion itself, and the track-integral scoring. **None of these are learned, and that is a strength** — they cannot overfit and they need no labels.

### Class imbalance
Oil pixels are a tiny fraction of any scene. Use **combined Dice + Focal loss**, not plain cross-entropy (which will happily predict "all sea" and score 99% accuracy). Report **IoU on the oil class specifically**, never overall pixel accuracy — quoting overall accuracy on this task is a tell that you don't understand the problem.

### Domain shift and geographic holdout
SAR statistics vary by sea state, region and incidence angle. **Hold out entire geographic regions, not random tiles.** Random tile splits leak — adjacent tiles from the same scene share speckle and sea state, and will inflate your numbers.

Protocol: train on regions A+B, test on region C, and **report the drop**. A team that reports "IoU 0.58 in-domain, 0.44 cross-region" is far more credible than one reporting a single suspiciously round number.

Augmentation: flips, rotations, and **speckle-consistent** noise (multiplicative, not additive Gaussian — SAR speckle is multiplicative, and getting this wrong is a detail an RS scientist will notice).

### Scope discipline

**MUST BUILD** — U-Net segmentation · physics gate + LightGBM · OpenOil forward/backward wrapper · **inversion engine** · AIS ingest + track reconstruction · **track-integral scorer with dark-vessel term** · deck.gl interface · evidence export

**SHOULD BUILD** — GDP drifter validation harness · geographic holdout evaluation · release-mode inference · repeat-offender aggregation

**NICE TO HAVE** — dark-vessel detection from SAR · multi-scene linking · shipping-lane prior · oil-type inference

**DO NOT BUILD** — your own SAR calibration chain · your own ocean model · your own drift solver · your own weathering model · a supervised attribution ranker (no labels exist) · a chatbot · a blockchain evidence ledger · mobile app · anything with "LLM" in it

---

## 12. EVALUATION METRICS

| Subsystem | Metric | Credible student prototype | Notes |
|---|---|---|---|
| Segmentation | **IoU (oil class)** | **0.50–0.65** | Published SOTA ≈ 0.54. Above 0.85 on a fair split is implausible — expect to be challenged |
| Segmentation | Recall (oil) | > 0.75 | Stage 1 must favour recall |
| Look-alike | **False-positive rate** | **< 0.15** | The headline number for Stage 2 |
| Look-alike | Specificity | > 0.85 | |
| Look-alike | Abstention rate | 5–15% | Too low = not gating; too high = useless |
| Drift forward | Separation distance vs. drifter @24 h | **10–25 km** | Realistic for a 1/12° field. Anyone claiming <5 km is not measuring honestly |
| Drift forward | Skill score vs. persistence | > 0.5 | Beat "assume it doesn't move" |
| **Inversion** | **True source inside 95% region** | **> 0.90** | *Calibration*, not accuracy — this is the number that proves honesty |
| **Inversion** | **95% region area @ 24 h** | **2,000–8,000 km²** | The honest measure of narrowing |
| Inversion | Time-marginal width @ 24 h | 8–20 h | |
| Attribution | **Top-3 recall** | **0.60–0.80** | On synthetic ground truth |
| Attribution | Top-1 recall | 0.30–0.50 | **Deliberately modest — top-1 is not the goal** |
| Attribution | Candidate reduction factor | **50–100×** | ~200 vessels → ~3. **This is our headline metric** |
| System | Runtime, scene → ranking | < 10 min | |

### The metric that wins

**Calibration, not accuracy.** If we say 95% and the true source falls inside 95% of the time, our uncertainty is *honest*. That is a claim almost no student team can make and every serious evaluator respects. Plot a **calibration curve** — nominal coverage vs. empirical coverage — and put it on a slide.

---

## 13. THE 90-SECOND DEMO

Fully offline. Pre-computed results, pre-cached tiles, no live API calls. The judge chooses among prepared **real** scenes.

| Time | Screen | Audio |
|---|---|---|
| 0–8s | Single Sentinel-1 scene, no UI chrome. A dark slick. | *"This spill was detected. Nobody was ever charged."* |
| 8–15s | Three case cards. **Judge picks one.** | *"Pick any of these three real scenes."* |
| 15–28s | Segmentation overlay. Two dark patches highlighted. | *"Two dark patches. Both look identical."* |
| 28–40s | **One patch turns grey — REJECTED.** ERA5 wind field overlays: 1.4 m/s. Label: *"outside detectability window."* | *"At one and a half metres per second the sea is already flat — oil cannot produce contrast. That is not a spill, and we say so."* |
| 40–52s | Characterisation panel: area, elongation, orientation. Slick axis vector drawn. | *"Elongated and track-aligned — consistent with a continuous discharge under way."* |
| 52–66s | **Particles flow backwards.** Then a heat map blooms — the posterior. **The 95% envelope is drawn, with its area in km².** | *"We don't run diffusion backwards — that's not reversible. We run a hundred thousand forward simulations and keep the ones that reproduce what the satellite saw."* |
| 66–78s | AIS tracks fade in. Counter: **214 vessels**. Filter runs. **Three remain**, ranked, with per-vessel evidence. | *"Two hundred and fourteen ships were in that window. Three of them have tracks that pass through the probable source region."* |
| 78–86s | Top candidate: track overlaid on envelope, AIS silence window in red. Fourth row visible: **"Vessel not transmitting AIS — 22%."** | *"And we always carry the possibility that the real polluter wasn't transmitting at all."* |
| 86–90s | Calibration curve. Freeze. | ***"We are not telling you who did it. We are telling you which three ships to ask — and how much to trust that."*** |

**The 28–40s beat is the one that wins.** Showing what you correctly *refuse* to flag is more persuasive than anything you do flag, because it proves the system has judgment rather than eagerness.

---

## 14. TECH STACK

| Layer | Choice | Why |
|---|---|---|
| SAR preprocessing | **ESA SNAP** (esa_snappy) or **pyroSAR** | Reference implementation; judges recognise it |
| ML | **PyTorch** + `segmentation_models_pytorch` | U-Net + pretrained encoders in a few lines |
| Tabular ML | **LightGBM** | Fast, interpretable feature importances |
| Ocean data | **`copernicusmarine`** toolbox | Official CMEMS client |
| Weather | **`cdsapi`** | Official ERA5 client |
| Drift | **OpenDrift / OpenOil** | Peer-reviewed, MET Norway |
| Geospatial | rasterio, geopandas, shapely, pyproj | Standard |
| AIS storage | **DuckDB + spatial extension** | No server, columnar, fast on CSV. **PostGIS only if you genuinely need concurrency** — you don't |
| Backend | **FastAPI** | Async, auto OpenAPI docs |
| Job queue | Celery + Redis *(only if needed)* | Inversion runs are long |
| Frontend | **React + deck.gl** | **GPU layers. `TripsLayer` animates AIS, `ScatterplotLayer`/`HeatmapLayer` renders 10⁵ particles at 60fps.** Leaflet will collapse under this |
| Basemap | MapLibre GL + free tiles | No API key, works offline |
| Packaging | Docker + docker-compose | |
| Deployment | **Single VM, or laptop for the demo** | **No Kubernetes. No cloud-native anything.** It adds risk and zero marks |

deck.gl over Leaflet is not a style preference — animating a hundred thousand particles is the visual core of the demo, and only GPU layers will do it smoothly.

---

## 15. SIX-MEMBER TEAM

| # | Role | Owns | Stack | Key deliverable | Depends on |
|---|---|---|---|---|---|
| **1** | **SAR / ML** | Preprocessing chain, U-Net, training, geographic holdout eval | SNAP, PyTorch, smp | Segmentation model + honest metrics table | Data from #3 |
| **2** | **Physics / inversion** ★ | OpenOil wrapper, **the inversion engine**, uncertainty, drifter validation | OpenDrift, xarray, numpy | Posterior field + calibration curve | Masks from #1, forcing from #3 |
| **3** | **Data engineering** | CMEMS/ERA5/AIS pipelines, DuckDB, caching, demo case prep | copernicusmarine, cdsapi, DuckDB, polars | Reproducible data layer + prepared cases | — (starts first) |
| **4** | **Attribution** ★ | Track reconstruction, track-integral scorer, dark-vessel term, behavioural prior, evidence export | pandas, geopandas, scipy | Ranked candidates + evidence PDF | Posterior from #2, AIS from #3 |
| **5** | **Frontend / geo** | deck.gl interface, particle + AIS animation, map UX | React, deck.gl, MapLibre | The interface the PS requires | API from #6 |
| **6** | **Integration + narrative** | FastAPI, Docker, **the six slides**, demo video, rehearsals | FastAPI, Docker, PowerPoint | **Submitted PDF + video** | Everyone |

**Member 6 is not a spare part.** Your documented failure mode is the submission, not the code. This person owns the slides from **day one** and has authority to demand screenshots from anyone.

**Integration discipline:** define the three internal contracts in week 1 and freeze them — `SlickDetection` (polygon + features + confidence), `SourcePosterior` (gridded density + credible regions), `RankedCandidates` (vessels + scores + evidence). With frozen contracts, all six can work against stubs in parallel. Without them, you converge in week 12 or never.

---

## 16. PHASED PLAN

### First, a correction to the deadline framing

**20 September 2026 is the *idea submission* deadline — six slides plus a video. It is not the prototype deadline.** The working prototype is for the Grand Finale in December.

This changes everything about the next three weeks. You are not building the system by 20 September. You are building **enough of a thin vertical slice to prove the idea is real**, and producing a submission. Teams that misread this burn three weeks on model tuning and submit weak slides — which is precisely the failure you have already lived through twice.

### Phase 0 — GO/NO-GO (25 Aug – 1 Sep) 🔴 HARD GATE
- Member 3: Copernicus + CMEMS + CDS accounts; download one S-1 GRD scene; pull SMOC and ERA5 for it
- Member 1: Zenodo Parts I–III downloaded; one U-Net epoch runs end to end
- Member 2: OpenOil installed; toy forward *and* backward run completes
- Member 6: six pointer phrases written and answered in one sentence each
- **Someone emails a faculty supervisor to request the Krestenitis dataset** (long lead time)

> **Biggest risk:** SNAP/snappy installation. It is Java-backed and genuinely painful. **Mitigation: the Zenodo data is already calibrated σ⁰ in dB, so you can train without SNAP at all.** SNAP only becomes necessary for processing raw scenes. Do not let it block Phase 0.

**GO if:** a scene renders, a U-Net epoch runs, an OpenOil run completes. **NO-GO → switch to SIH26034 on 1 September, no debate.**

### Phase 1 — Data + contracts (1–5 Sep)
Freeze the three interface contracts. AIS into DuckDB. Choose 3 demo cases.
🔴 *Risk: AIS volume. Mitigation: subset to one region-month before optimising anything.*

### Phase 2 — Detection (3–10 Sep)
Train U-Net. Geographic holdout. Physics gate + LightGBM.
🔴 *Risk: chasing IoU. **0.55 is the published benchmark — hit it and stop.** The gate matters more than the last 3 points.*

### Phase 3 — Inversion (5–14 Sep) ★ the long pole
Backward proposal → forward ensemble → conditioning → posterior. **Start this in parallel with Phase 2 using a hand-drawn mask.** Do not wait for the model.
🔴 *Risk: the ensemble is too slow. Mitigation: cut N, coarsen the time grid, precompute for demo cases.*

### Phase 4 — AIS + attribution (10–16 Sep)
Track reconstruction, track-integral scoring, dark-vessel term.
🔴 *Risk: messy AIS (duplicate MMSI, position jumps). Mitigation: filter hard, document what you dropped.*

### Phase 5 — Thin slice integration (14–17 Sep)
One case running end to end. Rough UI is fine.
🔴 *Risk: contract drift. Mitigation: contracts were frozen in Phase 1.*

### Phase 6 — Validation (15–18 Sep)
Calibration curve. Top-K on synthetic. Drifter comparison if GDP verified.
🔴 *Risk: numbers are bad. **Mitigation: report them anyway.** Honest weak numbers beat absent numbers — and the calibration curve is the point, not the accuracy.*

### Phase 7 — Submission (17–20 Sep) 🔴
Six slides. Video. PDF. SPOC upload. **Remember: the PS cannot be changed after submission.**
🔴 *Risk: leaving this to the last 48 hours. This is the risk that has already beaten you twice.*

### Phase 8 — Post-submission → December
Full system, real robustness, repeat-offender module, external validation from an oceanographer.

---

## 17. FAILURE MODES

| Risk | P | Impact | Mitigation | Fallback |
|---|---|---|---|---|
| SNAP/snappy install hell | **High** | Med | Zenodo data is pre-calibrated — train without SNAP | Use only pre-processed data; document as a scoping choice |
| Segmentation underperforms | Med | Med | Published SOTA is 0.54; you're at benchmark | Hand-corrected masks for demo cases, disclosed |
| Look-alike FPs remain high | Med | **High** | Physics gate + abstention class | Report abstention rate as a feature |
| Forward ensemble too slow | Med | **High** | Precompute demo cases; reduce N; coarsen grid | Demo runs from cached posteriors |
| Posterior too diffuse to be useful | **Med-High** | **High** | Cap lookback at 24–48 h; state the horizon as a system limit | **Report the envelope honestly — this is not failure, it's the finding** |
| No real attribution ground truth | **Certain** | Med | Synthetic-from-real-AIS validation; be explicit | Frame as an open problem the system helps solve |
| AIS gaps ≠ guilt | **High** | Med | Baseline-relative gap prior | Downweight gaps entirely if unreliable |
| Judge finds backward diffusion error | **Low (we avoid it)** | **Fatal if present** | **Our entire design avoids it** | — |
| **Scientific overclaiming** | **Med** | **Fatal** | Ban "87% confidence" language from every artefact. Credible regions only | — |
| Team can't integrate | Med | **High** | Frozen contracts, week 1 | Demo the strongest single component |
| Weak submission despite good code | **Med** | **Fatal** | Member 6 owns slides from day one | — |

**The two fatal ones are both self-inflicted:** overclaiming, and a weak submission. Neither is a technical problem. Both are the ones that have actually beaten you before.

---

## 18. TWENTY JUDGE QUESTIONS

**1. Why Sentinel-1 and not optical?**
All-weather, day-night. Oil detection depends on surface roughness damping, which is a radar phenomenon — optical needs sun glint geometry that rarely coincides with a spill. We use the 1C/1D constellation at 6-day nominal revisit since 1A was terminated in June.

**2. Why GRD not SLC?**
We need amplitude, not phase. No interferometry, no polarimetric decomposition. GRD is multi-looked and far cheaper to process.

**3. Why VV?**
VH sits near the noise floor over low-backscatter water, so contrast collapses exactly where we need it. Bragg scattering, which oil damps, is a co-pol phenomenon.

**4. How do you distinguish oil from look-alikes?**
Partly we don't — and we say so. Below about 3 m/s the sea is already flat and *no* contrast is physically possible, so we gate on ERA5 wind and return "undetermined" rather than guess. Above that we use incidence-angle-normalised damping ratio, edge sharpness and shape. The gate is physics; the classifier is a 15-feature LightGBM whose importances we can show you.

**5. Why should we trust your backward drift?**
Because we don't do backward drift in the way you're expecting. Turbulent diffusion is not time-reversible — running it backwards is ill-posed. We use backward advection only to *narrow the search*, then run 10⁵ *forward* simulations with full physics and keep the ones that reproduce the observed slick. The physics only ever runs forward. That's Approximate Bayesian Computation, and it's why our envelope is a real posterior.

**6. What happens when the current data is wrong?**
It always is, to some degree. We perturb the current field with correlated noise scaled to the CMEMS-published RMS error, so model error enters the ensemble rather than being ignored. At 1/12° we cannot resolve sub-mesoscale eddies — that's an irreducible floor, and it's why we report regions rather than points.

**7. Why isn't nearest-vessel matching enough?**
Because the nearest vessel to a slick centroid is usually not the source — the slick has drifted for hours. Nearest-vessel answers "who is near the oil now"; the question is "who was at the origin when the release happened." Different place, different time.

**8. What does your confidence number actually mean?**
It's posterior probability under our stated model assumptions — the fraction of accepted ensemble particles whose origin lies along that vessel's track, normalised over all candidates including a dark-vessel hypothesis. It is not a probability of guilt, and we'd resist anyone presenting it as one.

**9. What if several vessels fall inside the envelope?**
That's the normal case, and the system is designed to report it. We return a ranked shortlist, not a name. Reducing 214 vessels to 3 is the deliverable. If the posterior is too diffuse to separate candidates, we say that too.

**10. What if the polluter had AIS off?**
Then it isn't in the candidate set, which is why we carry an explicit dark-vessel hypothesis in the denominator. The system can output "most probable explanation: a vessel not transmitting AIS." Without that term, we'd confidently name an innocent ship.

**11. How do you validate attribution when there's no ground truth?**
We don't claim to. No public dataset of confirmed spill→vessel attributions exists. We validate three things separately: detection on real labelled SAR with geographic holdout; drift against real drifter trajectories; and inversion on synthetic releases seeded from *real* AIS tracks, where ground truth is true by construction. End-to-end validation on real attributed spills is an open problem — and a system like this is how you'd start building that dataset.

**12. Isn't your synthetic validation circular?**
Partly, and here's the boundary. The simulator generates the observation *and* evaluates candidates, so it tests the inversion machinery, not the physics. That's why drift is validated separately against real drifters — real ocean, real trajectories, no simulator in the loop.

**13. What's your IoU?**
0.50–0.65 on the oil class, in line with published benchmarks (U-Net ~0.54 on the standard dataset). Cross-region holdout drops it by roughly 10 points and we report both. If we'd told you 0.95 you should have asked what we'd overfitted to.

**14. Can you estimate slick age?**
Not directly from a single scene — damping ratio confounds thickness, oil type, wind and incidence angle simultaneously. We recover release time as the time-marginal of the source posterior instead, which comes with error bars rather than a false point estimate. The PS says "if feasible"; our answer is that the radiometric route isn't, but the dynamical route is.

**15. How far back can you go?**
Practically 48–72 hours. Beyond that the envelope grows super-linearly — shear dispersion separates neighbouring particles roughly exponentially at first — and stops being operationally useful. We report the horizon as a system limit rather than producing envelopes nobody can act on.

**16. Why can't a commercial provider do this?**
They largely can, for detection — CleanSeaNet and KSAT run operational services. What's missing is the automated closing of the loop to attribution with quantified uncertainty. Detection tells you a crime happened; this tells you who to investigate and how much to trust it.

**17. What's actually innovative here?**
Three things: framing source-finding as Bayesian inversion by forward ensemble rather than naive backtracking; gating look-alike rejection on the physical wind-detectability window rather than a bigger CNN; and scoring vessels by the posterior mass their track sweeps, with an explicit dark-vessel hypothesis. The first is the one we'd defend hardest.

**18. What does deployment cost?**
Every input is free and open — Sentinel-1, CMEMS, ERA5. Compute is one GPU for training and CPU for inference; an inversion runs in minutes. The recurring cost is AIS in regions without an open feed.

**19. How does it scale to the Indian EEZ?**
The inputs are global, so the method is region-agnostic. The binding constraint isn't compute, it's AIS availability — we validated on Danish, Norwegian and US data because those are the open historical feeds. For Indian waters we'd need a national AIS feed, and we use synthetic data in the interim, as the PS permits.

**20. Why should we pick your team?**
Because we can tell you what our system can't do. We won't name a ship, we won't give you an age from one scene, and past 72 hours we'll tell you the answer isn't useful. Everything we do claim has a calibration curve behind it.

---

## FINAL OUTPUT

### 1. The best possible solution — one paragraph

An automated pipeline that ingests Sentinel-1 GRD, segments candidate dark patches with a U-Net, and then — critically — rejects look-alikes using a physical wind-detectability gate rather than more model capacity, abstaining where contrast is physically impossible. Characterised slick geometry constrains a release-mode prior. The system then solves for the source as a **Bayesian inverse problem**: a cheap backward advection-only pass narrows the search region, after which ~10⁵ candidate sources are propagated **forward** through full stochastic drift physics and conditioned on reproducing the observed slick, yielding a genuine posterior density over source position and time with calibrated credible regions. Historic AIS is reconstructed over that window, and each vessel is scored by the **posterior mass its track sweeps**, normalised against an explicit **dark-vessel hypothesis** — producing a ranked shortlist, an evidence bundle, and an honest statement of how much the search space was narrowed, rendered in a GPU-accelerated geospatial interface.

### 2. Final architecture
See §3. One pipeline: **Ingest → SAR preprocess → U-Net → physics gate → characterisation → backward proposal → forward ensemble → posterior → AIS track integral → ranked evidence → interface.**

### 3. Top 3 differentiators
1. **Bayesian source inversion by forward ensemble** — correct where naive backtracking is provably wrong
2. **Track-integral scoring with explicit dark-vessel hypothesis** — principled ranking, no invented weights
3. **Physics-gated look-alike rejection** — answers SAR's hardest question with physics, and abstains honestly

### 4. Dataset stack
Sentinel-1 GRD IW (1C/1D) · Zenodo SAR oil spill Parts I–III · Krestenitis *(request now)* · CMEMS SMOC · ERA5 10m wind · MarineCadastre + Danish DMA + Norwegian Kystverket AIS · GDP drifters *(verify)*

### 5. Model / algorithm stack
U-Net/ResNet34 (Dice+Focal) · LightGBM on 15 physical features · OpenOil Lagrangian transport · ABC rejection sampling · KDE posterior · track-integral Bayesian scoring

### 6. Six-member plan
SAR/ML · Physics-inversion · Data engineering · Attribution · Frontend/geo · Integration+narrative. Three frozen contracts in week 1.

### 7. Three-week execution
GO/NO-GO by **1 Sep** → data+contracts → detection → **inversion (long pole, start early with hand-drawn masks)** → AIS → thin slice → validation → **submission by 20 Sep**. Target a thin vertical slice plus six strong slides, **not** a complete system.

### 8. Biggest technical risk
**The forward ensemble is too slow, or the posterior too diffuse to separate candidates.** Mitigate by capping lookback at 24–48 h, precomputing demo cases, and — if it stays diffuse — **reporting that honestly as a finding about the physics rather than hiding it.**

### 9. GO / NO-GO criteria — 1 September
✅ **GO:** S-1 scene renders · U-Net epoch completes · OpenOil forward+backward runs · CMEMS + ERA5 retrieved · six pointer phrases answered
❌ **NO-GO → switch to SIH26034 immediately.** The switch is free on 1 September and catastrophic in November.

### 10. Grand finale demo
See §13. The winning beat is at **28–40s**: the system *rejects* a dark patch because the wind was 1.4 m/s. Showing what you refuse to flag proves judgment.

### 11. The one sentence that defines the product

> **We don't name the ship. We take two hundred vessels down to three, and we tell you exactly how much to trust that number.**
