# Context — planning and review documentation

**Problem Statement:** Leveraging satellite imagery to determine oil spills at sea along with AIS data correlations to identify the vessel responsible.
**Organisation:** NTRO · **Category:** Software · **PS ID:** SIH26143

> The project overview, figures and quickstart live in the [root README](../README.md).
> This folder is the engineering record: what we planned, what we measured, and what went wrong.

---

## What these documents are for

They serve three audiences, and the third is the one that matters most.

1. **Round 2 judges** — depth behind the demo: why the method is what it is, what it cannot do, and how we know.
2. **Round 3 (ourselves, in December)** — the blueprint for the production system, with every accepted shortcut and its exit path recorded.
3. **Us, next week** — a decision written down is a decision that does not get relitigated at 2 a.m.

Each document describes the **full system**, then marks a `PROTOTYPE SCOPE` boundary for what was actually built. That boundary is deliberate and disclosed, not a gap.

---

## The documents

| Document | Purpose | State |
|---|---|---|
| [requirements.md](requirements.md) | Every PS clause traced to a testable acceptance criterion, plus explicit non-goals | Frozen before coding |
| [architecture.md](architecture.md) | The `Case` spine, the four frozen contracts, build-vs-reuse decisions | Frozen before coding |
| [technical_design.md](technical_design.md) | Physics gate, Bayesian source inversion, vessel-conditioned attribution | Frozen before coding |
| [phasewise_roadmap.md](phasewise_roadmap.md) | Ten phases with a completion note and measurements for each | Updated per phase |
| [testing_strategy.md](testing_strategy.md) | The truth harness and its anti-cheating rules | Expanded during build |
| [deployment.md](deployment.md) | Environment, offline runbook, demo-day failure drills | Live |
| [security_review.md](security_review.md) | Wrongful-attribution controls, licensing, AIS sensitivity, executed checklist | ✅ Executed |
| [performance_review.md](performance_review.md) | Budgets against measurements | ✅ Measured |
| [engineering_review.md](engineering_review.md) | Standards and the accepted technical-debt register | Live |
| **[problems_faced_and_bugs_encountered.md](problems_faced_and_bugs_encountered.md)** | **24 findings, including six silent successes** | ✍️ Continuous |

---

## The three claims the project rests on

Everything else is engineering. These are where it is won or lost.

**1 · Look-alike rejection is a physics problem, not a capacity problem.**
A low-wind patch and an oil slick are genuinely inseparable in image space. The information that distinguishes them is not in the image, so we inject it from the wind field and abstain where no answer is physically possible.

**2 · Backward drift is not the inverse of forward drift.**
Advection reverses; turbulent diffusion does not. Backward integration narrows the search; forward simulation computes the answer. The kernel raises rather than integrating backwards with diffusion enabled.

**3 · A vessel's AIS track is a generative hypothesis, not a feature vector.**
A discharge from a moving vessel is a line source in space-time, and AIS already gives us its parameters. We simulate what each vessel *would have produced* and ask which prediction matches the satellite — no invented weights, and the dark-vessel hypothesis competes on the same scale.

---

## What was measured

| | |
|---|---|
| End to end, scene → ranked vessels | **14.5 s** (budget 60 s) |
| Tests | **283 passing** |
| Calibration at 50% / 68% | 0.50 / 0.67 — honest |
| Calibration at 90% / 95% | 0.75 / 0.83 — **below target**, cause isolated (P-22) |
| Culprit in candidate set | **1.00** |
| Dark hypothesis ranked top when culprit removed | **1.00** |
| Top-3 recall | **0.44** — below the 0.60 target |
| Traffic reduction | **179×** |

Two targets are missed. Both are reported as misses, with the cause separated from the symptom rather than tuned away.

---

## The pattern worth taking away

Six of the twenty-four logged findings were **silent successes** — plausible, confident, wrong output with no error anywhere. Not one was caught by a crash; every one was caught by measuring something a second way, or by attributing an effect before believing it.

That is now a rule, and it is enforced in the test suite:

> **Any check whose job is to find something must be shown failing on a planted example before its passing result is believed.**

`tests/test_offline.py` proves its own network guard bites. `tests/test_audit.py` plants an accusatory string and asserts the audit fails. Those exist because [P-24](problems_faced_and_bugs_encountered.md) was an audit that reported twelve clean passes while one of its checks examined nothing at all.

---

## Regenerating the figures

The figures in the root README are drawn from real pipeline output, not illustrated:

```bash
python scripts/make_figures.py
```

A figure that looks wrong means the pipeline is wrong.
