# SIH26143 — Oil Spill Detection & AIS Vessel Attribution

**Problem Statement:** Leveraging satellite imagery to determine oil spills at sea along with AIS data correlations to identify the vessel responsible.
**Organisation:** NTRO · **Category:** Software · **Team:** IIIT Kottayam, 6 members (CSE / AI-DS)

---

## What this system does

Given one Sentinel-1 SAR scene containing a suspected oil slick, the system answers three questions and states how much to trust each answer:

1. **Is that dark patch actually oil?** — U-Net segmentation followed by a *physical wind-detectability gate* that abstains where oil–water contrast is physically impossible.
2. **Where and when did it come from?** — Bayesian source inversion. Uncertainty is reported as credible regions in km², never as a point.
3. **Which vessel?** — Each candidate vessel's AIS track is used as a *generative hypothesis*, forward-simulated, and scored by how well it reproduces the observed slick — normalised against an explicit "the polluter wasn't transmitting AIS" hypothesis.

> **The one sentence:** We don't name the ship. We take two hundred vessels down to three, and we tell you exactly how much to trust that number.

---

## Scope: what is built when

| Round | Deliverable | Status |
|---|---|---|
| **Round 1** | Running prototype, end-to-end on pre-cached cases | 🔨 **This build** — 14-day sprint |
| **Round 2** | Approach, architecture, methodology, evidence of rigour | 📄 **These documents** |
| **Round 3** | Production system — raw scene ingest, OpenOil weathering, scale | 🔮 Blueprint only |

Every document in this folder describes the **full system**, then marks a `PROTOTYPE SCOPE` boundary showing what actually gets built now. That boundary is deliberate and disclosed, not a gap.

### Prototype simplifications (all disclosed, all reversible)

| Full system | Prototype | Why it's safe |
|---|---|---|
| ESA SNAP raw GRD ingest | Pre-calibrated σ⁰ dB tiles (Zenodo) | Zero innovation credit in calibration; highest install-failure risk |
| OpenDrift / OpenOil solver | ~150-line Lagrangian RK4 kernel | Differentiator is the *inversion method*, not the solver. No weathering in prototype |
| React + deck.gl SPA | Single HTML page, deck.gl from CDN | Identical visuals, no Node toolchain |
| Live CMEMS / CDS API calls | Pre-cached NetCDF per case | Venue WiFi will fail. Demo must be fully offline |
| Global coverage | 3 prepared cases | Depth over breadth for a 14-day build |

---

## Regional strategy

**Validate where the data is real. Demonstrate transferability to Indian waters.**

- **Primary (real end-to-end):** Danish waters — Kattegat / Skagerrak. Real AIS (Danish Maritime Authority), real CMEMS currents, real ERA5 wind.
- **Secondary (India story):** Arabian Sea / Gulf of Kutch. Real CMEMS + ERA5, **synthetic AIS** — which the problem statement explicitly permits.

Disclosed synthetic data is a methodological choice. Discovered synthetic data is a credibility collapse. It goes on the slide.

---

## Repository layout (target)

```
SIH143/
├── Context/              ← you are here: all planning + review docs
├── data/
│   ├── cases/            ← one folder per prepared demo case
│   ├── forcing/          ← cached CMEMS + ERA5 NetCDF
│   ├── ais/              ← DMA CSV → parquet
│   └── training/         ← Zenodo SAR oil spill Parts I–III
├── src/
│   ├── detect/           ← U-Net, physics gate, characterisation
│   ├── transport/        ← Lagrangian kernel, forcing readers
│   ├── inversion/        ← proposal, ensemble, posterior
│   ├── attribution/      ← AIS ingest, vessel hypotheses, ranking
│   ├── contracts/        ← the four frozen schemas
│   └── api/              ← FastAPI
├── web/                  ← single-page deck.gl UI
├── notebooks/            ← training + evaluation
└── scripts/              ← run_case.py, build_synthetic_truth.py
```

---

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

```bash
python scripts/run_case.py --case data/cases/kattegat_2024_03_11 --out out/
```

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`. The demo runs entirely offline from cached case data.

---

## Document index

| Document | Purpose | Lifecycle |
|---|---|---|
| [requirements.md](requirements.md) | PS clause traceability, acceptance criteria, explicit non-goals | Frozen before coding |
| [architecture.md](architecture.md) | System structure, the `Case` spine, four frozen contracts | Frozen before coding |
| [technical_design.md](technical_design.md) | The science: segmentation, physics gate, inversion, attribution | Frozen before coding |
| [phasewise_roadmap.md](phasewise_roadmap.md) | 14-day sprint, day-by-day, 6 owners, gates | Frozen before coding |
| [testing_strategy.md](testing_strategy.md) | Synthetic ground-truth generator, metrics, calibration protocol | Basic now, expanded during build |
| [deployment.md](deployment.md) | Environment, packaging, demo-day runbook, failure drills | Live during build |
| [security_review.md](security_review.md) | Licensing, AIS sensitivity, defamation risk, secrets | After features complete |
| [performance_review.md](performance_review.md) | Budgets, profiling, hotspots | After it works |
| [engineering_review.md](engineering_review.md) | Code quality, tech debt, Round 3 handoff | Near the end |
| [problems_faced_and_bugs_encountered.md](problems_faced_and_bugs_encountered.md) | Running log of every trap hit and solved | **Continuous — this is Round 2 material** |

---

## Status

**Sprint start:** 6 September 2026 · **Round 1 demo target:** 19 September 2026

Live status is tracked in [phasewise_roadmap.md](phasewise_roadmap.md).
