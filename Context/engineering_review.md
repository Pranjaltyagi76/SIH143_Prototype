# Engineering Review

**Status:** 🔧 Scheduled for Day 13–14, near the end of implementation. Standards and the pre-agreed debt register recorded now. · **Owner:** M6 · **Last updated:** 2026-09-05

---

## 0. Purpose

Two audiences.

**Round 2 judges** want evidence that a 14-day prototype was engineered rather than hacked — that the shortcuts were chosen, bounded and recorded rather than stumbled into.

**Round 3 (ourselves, in December)** need to know exactly which parts of this code survive and which were always disposable. A shortcut you recorded on the day you took it costs an hour to undo. One you rediscover three months later costs a week.

**Deliberate debt is engineering. Accidental debt is a mess.** This document is the boundary between them, and most of it is written *before* the code so the boundary is real.

---

## 1. Standards in force during the sprint

Deliberately few. Every rule below is one we will actually follow at 1 a.m. on Day 11; anything more elaborate would be aspirational and therefore worthless.

| Standard | Rule | Enforcement |
|---|---|---|
| **Contracts** | The four Pydantic models are the only inter-stage interface. No stage reaches into another's internals | Frozen Day 2; fixtures validated in CI |
| **Formatting** | `black`, line length 100 | Pre-commit hook |
| **Linting** | `ruff`, default rules | Pre-commit hook |
| **Type hints** | On every function signature crossing a module boundary. Not required internally | Review |
| **Tests** | Every physics invariant in [testing_strategy.md](testing_strategy.md) §3.1 has a unit test **before** the code it guards | Review |
| **Determinism** | Every RNG derives from the case seed. No bare `np.random` calls | Grep audit on Day 12 |
| **Units** | SI internally — metres, seconds, m/s. Conversion only at the UI boundary. Variables carrying non-SI units are suffixed (`area_km2`, `speed_kn`) | Naming convention |
| **No magic numbers** | Every physical constant lives in `configs/physics.yaml` with a comment giving its source | Review |
| **Logging** | Structured JSON to `out/log.jsonl`. No `print` in library code | Grep audit |
| **Commits** | Present tense, one logical change. Not enforced beyond readability |

### The one rule that matters most

**Every function that makes a physical assumption states it in its docstring, in words, with a source.** Not `# windage`. Instead:

```python
def apply_windage(u_wind10, alpha):
    """Add the wind-driven component of surface oil drift.

    alpha is the fraction of 10 m wind speed imparted to a surface slick,
    ~1-4% in the literature. Sampled per particle, never fixed - treating it
    as a constant materially understates position uncertainty.

    NOTE: This is the ONLY wind term. Stokes drift is already merged into
    the CMEMS SMOC current field; adding a separate Stokes parameterisation
    here would double-count a real physical term.
    """
```

This costs thirty seconds and it is where the project's scientific credibility physically lives. It is also, directly, the material for the Round 2 write-up.

---

## 2. Technical debt register — accepted before coding

Each item is a shortcut we are taking **on purpose**, with the reason, the cost, and the exit.

| # | Debt | Why accepted | Cost incurred | Exit in Round 3 |
|---|---|---|---|---|
| **D1** | **Custom transport kernel instead of OpenDrift/OpenOil** | OpenDrift needs conda + GDAL, likely WSL2 on our Windows fleet. Unacceptable risk on the critical path with 14 days | No weathering; weaker to cite | Kernel sits behind one interface, `transport.simulate(...)`. Swap is a single adapter class |
| **D2** | **No SAR preprocessing chain** | SNAP is Java-backed and the highest-probability install failure in the project. Zenodo data is already calibrated σ⁰ dB | Cannot ingest a raw scene | Add pyroSAR chain as `ingest/preprocess.py`. Downstream unaffected — it produces the same σ⁰ raster |
| **D3** | **Three hardcoded cases, no general ingest** | Demo needs depth, not breadth | Not a general system | `build_case.py` already exists; generalise its config |
| **D4** | **`lambda` (detectability constant) hand-calibrated** | No principled derivation available in the time | An assumption, not a measurement | Calibrate against controlled-release experiments or literature damping data |
| **D5** | **`pi_dark` (AIS non-compliance prior) assumed at 0.10–0.20** | No regional statistics available to us | An assumption | Estimate empirically from SAR-detected ships without AIS matches, at scale |
| **D6** | **Synthetic AIS for the Arabian Sea case** | No free bulk historical Indian AIS source exists. We looked | Not real validation | National AIS feed, if an operator provides one |
| **D7** | **No weathering** | Follows from D1 | Drift is slightly wrong for volatile oils over long lookbacks | Comes free with D1 |
| **D8** | **KDE via separable Gaussian filter, not a proper kernel estimator** | ~100× faster, adequate at our grid resolution | Slightly smoothed posterior tails | Adaptive-bandwidth KDE |
| **D9** | **Frontend has no build step or component framework** | Removes npm from the critical path entirely | Single large HTML file; will not scale past ~5 panels | React + Vite when the UI grows |
| **D10** | **No CI** | 14 days, 6 people, one integration branch | Regressions caught at the 18:00 integration check instead of on push | GitHub Actions running pytest |
| **D11** | **Only three demo cases are integration-tested** | Time | Unknown behaviour on unusual scenes | Broaden the corpus |
| **D12** | **Particle counts tuned for 60 s, not for convergence** | NFR-2 | Posterior is Monte Carlo-noisy at the tails — **this is why `effective_sample_size` is reported** | Adaptive SMC with a convergence criterion |

**The register is complete before coding starts.** Anything discovered during the sprint is appended with its discovery date, which is how we tell chosen debt from accidental debt.

---

## 3. Design decisions to re-examine in Round 3

Not debt — genuine forks where we picked one branch and should revisit with more time.

| Decision | Chosen | Worth reconsidering because |
|---|---|---|
| U-Net / ResNet34 | Data-appropriate at ~2k scenes | With more labelled data a transformer segmenter may pull ahead. Requires the ablation we could not afford |
| LightGBM for the gate | Interpretable, fast, feature importances for slides | A small NN over raw patches *plus* physics features might beat it. Would need to preserve explainability |
| Hypothesis grid | Regular grid over the proposal region | Adaptive refinement, or a variational posterior, would use the compute better |
| Bernoulli observation operator | Simple, closed-form, penalises over-prediction | Does not use damping-ratio *intensity*, only binary presence. A graded observation model could use more of the image |
| Single-scene inversion | All we have per case | Multi-temporal linking across passes would tighten t0 dramatically. **Highest-value single addition** |
| Behavioural prior as independent factors | Interpretable | Factors are correlated (slow speed and AIS gaps co-occur). A joint model would be better but harder to explain |

---

## 4. Review checklist — to be executed Day 13–14

- [ ] Every module has a docstring stating what it computes and its assumptions
- [ ] Every physical constant is in `configs/physics.yaml` with a source comment
- [ ] No bare `np.random` — all RNG derives from the case seed
- [ ] No `print` in library code
- [ ] All four contracts validate against their fixtures
- [ ] Every physics invariant test from testing_strategy §3.1 exists and passes
- [ ] Dead code removed — especially abandoned experiments
- [ ] No commented-out blocks left behind
- [ ] README quickstart works from a clean clone on a clean machine
- [ ] Debt register updated with everything discovered during the sprint
- [ ] Every `TODO` either resolved or promoted into the debt register with an owner

---

## 5. Retrospective

*Written Day 14, after the demo. Round 2 material — answer honestly; the failures are more useful than the successes.*

### What worked
> *(to be filled)*

### What did not
> *(to be filled)*

### What we would do differently with the same 14 days
> *(to be filled)*

### What surprised us
> *(to be filled)*

### Round 3 handoff — the first five things to do
> *(to be filled)*

---

## 6. Metrics snapshot

*Filled Day 14.*

| Metric | Value |
|---|---|
| Lines of Python (excl. tests) | |
| Lines of tests | |
| Test count / pass rate | |
| Modules | |
| Dependencies (direct) | |
| Days from first commit to first end-to-end run | |
| Open debt items | 12 accepted pre-coding + *(discovered)* |
