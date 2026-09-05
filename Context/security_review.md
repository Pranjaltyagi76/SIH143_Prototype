# Security Review

**Status:** 🔧 Scheduled for Day 12 (after features complete, before demo). Framework and known findings recorded now. · **Owner:** M6 · **Last updated:** 2026-09-05

---

## 0. The threat model is not what you would expect

This is an offline batch pipeline on a single laptop with no users, no authentication, no persistence layer and no internet exposure. The conventional web-security surface is almost empty.

**The real risk in this system is that it produces accusations.** It names vessels — real vessels, with real owners, operators and crews — as probable polluters, on the basis of a probabilistic model with acknowledged limitations. Every other item in this review is secondary to that.

| Risk class | Severity here | Why |
|---|---|---|
| **Wrongful attribution / defamation** | 🔴 **Critical** | The core output of the system is an accusation |
| Data licensing and attribution | 🟠 High | Multiple sources with different licences; misuse is a real breach |
| AIS data sensitivity | 🟠 High | Vessel tracks are commercially sensitive and can identify individuals |
| Secrets management | 🟡 Medium | API credentials in a shared repo |
| Supply chain | 🟡 Medium | ~40 pinned dependencies |
| Network exposure | 🟢 Low | Loopback-bound, offline at demo |
| Injection / auth / XSS | 🟢 Low | No user input, no auth, no multi-tenancy |

---

## 1. Wrongful attribution — the critical control

### The harm

A false positive here is not a wrong number on a dashboard. It is a named ship, a named operator, and potentially a regulatory investigation or a public accusation founded on a model that we ourselves document as having a 2,000–8,000 km² uncertainty envelope and no end-to-end validation.

The system can be wrong for at least five structural reasons, all documented in [technical_design.md](technical_design.md):

1. The true polluter had AIS off and is **not in the candidate set at all**
2. The posterior is too diffuse to separate candidates, but still ranks them
3. AIS gaps used as behavioural evidence are **overwhelmingly benign** — shore-receiver coverage, not concealment
4. Current-field error at 1/12° is irreducible and unmodelled below ~9 km
5. There is **no validation** that the end-to-end chain produces correct attributions on real spills, because no such dataset exists

### Controls — mandatory, verified on Day 12

| # | Control | Verification |
|---|---|---|
| SC-1 | **The dark-vessel hypothesis is always present in the normalisation and always displayed.** Without it the system will confidently name an innocent ship whenever the true polluter was dark | Unit test: with zero AIS vessels supplied, the dark hypothesis receives probability 1 |
| SC-2 | **Banned vocabulary.** No output, screen, slide, export or spoken line may use: *guilty, responsible, culprit, identified the vessel, the polluter is, confirmed*. Permitted: *candidate, investigative lead, consistent with, under stated model assumptions* | Read every screen aloud on Day 12. Grep the codebase for banned strings |
| SC-3 | **Every displayed probability appears next to its assumption list.** `SourcePosterior.assumptions` is carried to the UI and the export, and is not collapsible away | UI review |
| SC-4 | **Vessel identity is redacted in the demo and in all public artefacts.** MMSI and name are shown as `REDACTED_IN_DEMO` in slides, video and any published screenshot | Screenshot audit before the video is recorded |
| SC-5 | **The evidence export carries an explicit limitations page**, including the absence of end-to-end validation | Export review |
| SC-6 | **The AIS-gap prior is baseline-relative, never absolute** | Unit test: a region with poor receiver coverage does not elevate every vessel in it |
| SC-7 | **The system reports when it cannot separate candidates** rather than ranking noise | If `effective_sample_size` collapses or the top-2 probabilities are within noise, the UI says so |
| SC-8 | **No ranked output leaves the machine.** Nothing is published, posted, emailed or shared outside the team | Process control |

> **The single sentence that carries this:** *"We are not telling you who did it. We are telling you which three ships to ask — and how much to trust that."*

### Why this is a strength, not a caveat

Being the team that has a written control list for wrongful attribution is itself a differentiator. An NTRO evaluator assessing an intelligence product will recognise the difference between a system that outputs a name and a system that outputs a bounded, caveated investigative lead. The second one is the one you can actually deploy.

---

## 2. Data licensing and attribution

**Action required — M3, by Day 10.** Each licence must be read, not assumed, and recorded in the table below. A misattributed dataset in a public submission is a real breach, not a technicality.

| Source | Licence | Attribution required | Redistribution | Verified |
|---|---|---|---|---|
| Sentinel-1 GRD (Copernicus) | Copernicus open licence | ✅ "Contains modified Copernicus Sentinel data [year]" | ✅ permitted with attribution | ⬜ |
| CMEMS SMOC | Copernicus Marine Service licence | ✅ CMEMS credit line required | ⚠️ check bulk redistribution | ⬜ |
| ERA5 (C3S / CDS) | Copernicus C3S licence, **acceptance required in account** | ✅ | ⚠️ check | ⬜ |
| Zenodo SAR oil spill dataset | Per-record (likely CC-BY) | ✅ **cite the authors — Trujillo-Acatitla et al.** | Per licence | ⬜ |
| Krestenitis benchmark | Restricted; institutional request | ✅ cite | ❌ **almost certainly not redistributable** | ⬜ |
| Danish DMA AIS | Free public data | ✅ credit DMA | ✅ | ⬜ |
| Norwegian Kystverket AIS | NLOD | ✅ | ✅ | ⬜ |
| MarineCadastre AIS | US public domain | Courtesy credit | ✅ | ⬜ |
| deck.gl, MapLibre | MIT / BSD-3 | Notice retained | ✅ | ⬜ |
| Python dependencies | Mixed OSS | `THIRD_PARTY_LICENSES.md` | ✅ | ⬜ |

**Rules already in force:**
- **No dataset is committed to the repository.** `data/` is `.gitignore`d in full. Restricted datasets are never redistributed, not even to teammates outside the request's terms.
- Attribution lines appear on the data slide, in the README, and in the evidence export.

---

## 3. AIS data sensitivity

AIS is publicly broadcast, but "public" is not "unrestricted".

| Concern | Assessment | Control |
|---|---|---|
| **Is AIS personal data?** | A vessel is not a person, but MMSI + name + track + timestamp can identify a small crew and their movements — particularly for fishing vessels and small craft. Under GDPR, EU-sourced tracks warrant caution | Treat as sensitive. Redact identity in every external artefact (SC-4) |
| **Commercial sensitivity** | Vessel movements reveal trading patterns and charter relationships | Not published; kept local |
| **Aggregation risk** | Combining AIS with an accusation model creates a *profile* that neither source alone constitutes | The controls in §1 are the mitigation |
| **Retention** | We hold one region-month | Delete after the competition; not committed to the repo |
| **Synthetic AIS** | Arabian Sea vessels are generated | **Must never use a real MMSI or real vessel name.** Generator uses reserved/invalid MMSI ranges — verified by unit test |

---

## 4. Secrets management

| Item | Control | Status |
|---|---|---|
| CDSE / CMEMS / CDS credentials | `.env`, `python-dotenv`, `.gitignore`d **on Day 1 before the first commit** | ⬜ |
| `.env.example` with empty values | Committed, so a new machine knows what it needs | ⬜ |
| Accidental commit | `git log -p | grep` scan for credential patterns before submission. If any secret was ever committed, **rotate it** — removing it from history is not sufficient | ⬜ |
| Credentials in notebooks | Notebook outputs cleared before commit | ⬜ |
| Credentials in logs | `out/log.jsonl` must never contain a request URL with an embedded token | ⬜ |

---

## 5. Supply chain

| Control | Detail |
|---|---|
| **Exact version pinning** | No `>=` anywhere in `requirements.txt` |
| Vulnerability scan | `pip-audit` on Day 12; record findings and decisions below |
| Typosquat check | Manual review of the dependency list — every package name read once, deliberately |
| **Model checkpoints** | `torch.load` executes pickle. We load **only our own weights, from our own drive.** Never a checkpoint from an untrusted source |
| Vendored CDN assets | deck.gl and MapLibre pinned to exact versions and vendored on Day 12 (also required by NFR-1) |
| Wheel cache | `pip download -d vendor/` on the backup drive, so the demo machine is rebuildable offline |

---

## 6. Application surface

| Item | Assessment |
|---|---|
| **Network binding** | `127.0.0.1` only, **enforced in code**, never `0.0.0.0`. There is no auth layer, so the API must not be reachable from conference WiFi |
| Authentication | None, and correctly none — single-user, loopback, offline |
| Input validation | No user-supplied input. Case IDs from the API are validated against a whitelist of existing folders (**path-traversal guard** — the one real injection vector) |
| File parsing | NetCDF / GeoTIFF from official sources only. Not a hostile-input scenario |
| Frontend | Data is our own JSON. Any string rendered from AIS static data (vessel name) is escaped — vessel names are free text from a broadcast and are the only untrusted string in the system |
| Logging | Structured JSON, no credentials, no full file paths in anything shown to an audience |

---

## 7. Review checklist — to be executed Day 12

- [ ] Grep the codebase and UI for every banned word in SC-2
- [ ] Read every screen aloud, hunting for overclaiming
- [ ] Verify the dark-vessel row renders in all three cases
- [ ] Verify assumptions are displayed next to every probability
- [ ] Verify vessel identities are redacted in slides, video and screenshots
- [ ] Confirm `.env` is not in git history
- [ ] Run `pip-audit`; record and triage findings
- [ ] Confirm the API refuses to bind to a non-loopback address
- [ ] Confirm path-traversal guard on the case-ID parameter
- [ ] Confirm the synthetic AIS generator emits no real MMSI
- [ ] Complete the licence verification table in §2
- [ ] Confirm no dataset is committed to the repository

---

## 8. Findings log

*Populated during the Day 12 review.*

| # | Finding | Severity | Status | Resolution |
|---|---|---|---|---|
| | *(none yet — review not run)* | | | |
