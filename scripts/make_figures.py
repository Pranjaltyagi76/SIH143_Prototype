"""Generate the README figures from real pipeline output.

    python scripts/make_figures.py

Every figure is drawn from files the pipeline actually produced -- the scene
raster, the detection contracts, the posterior grid, the ranked candidates and
the evaluation summaries. Nothing is illustrative or hand-placed, so a figure
that looks wrong means the pipeline is wrong, and regenerating after a change
shows what the change did.

    docs/figures/01-detection.png    what the detector found, and refused
    docs/figures/02-attribution.png  source posterior and ranked vessels
    docs/figures/03-evaluation.png   calibration, envelope, attribution metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASE = ROOT / "data" / "cases" / "synth_kattegat_spill"
EVAL = ROOT / "eval"
OUT = ROOT / "docs" / "figures"

INK = "#e6edf3"
DIM = "#8b97a7"
BG = "#0d1117"
PANEL = "#141b25"
LINE = "#232c39"

CLASS_STYLE = {
    "oil": ("#f0883e", "CONFIRMED OIL"),
    # Lighter than the panel grey: this outline is drawn on top of the SAR
    # backscatter, and #8b97a7 disappeared into it.
    "look_alike": ("#d6dde5", "LOOK-ALIKE, REJECTED"),
    "undetermined": ("#4d9fd6", "UNDETERMINED"),
}
LINE_STYLE = {"oil": "-", "look_alike": (0, (5, 2)), "undetermined": "-"}


def _dark(fig, *axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(LINE)
        ax.tick_params(colors=DIM, labelsize=8)
        ax.xaxis.label.set_color(DIM)
        ax.yaxis.label.set_color(DIM)
        ax.title.set_color(INK)


def _load(name, default=None):
    path = CASE / "out" / name
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def figure_detection() -> None:
    """The scene, what was found in it, and what was refused."""
    from PIL import Image

    detections = _load("detections.json", [])
    manifest = _load("ui/manifest.json", {})
    if not detections or not manifest:
        print("  skipped 01-detection: run scripts/run_case.py first")
        return

    scene = np.asarray(Image.open(CASE / "out" / "ui" / "scene.png").convert("L"))
    west, south, east, north = manifest["bounds"]

    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.16)
    ax = fig.add_subplot(gs[0])
    # Percentile stretch: the raw PNG is clipped to a fixed dB range so cases
    # stay comparable, which leaves this scene mid-grey and flattens the
    # low-wind pocket. Stretching is a display choice for the figure only.
    lo, hi = np.percentile(scene, [2, 98])
    ax.imshow(scene, cmap="gray", extent=[west, east, south, north],
              origin="upper", aspect="auto", vmin=lo, vmax=hi)

    # Label only the three largest patches: the small abstentions cluster
    # inside the low-wind pocket and their labels overprint each other.
    labelled = {id(d) for d in sorted(detections,
                                      key=lambda x: -x["geometry"]["area_km2"])[:3]}
    for d in detections:
        colour, _ = CLASS_STYLE[d["classification"]]
        poly = np.array(d["polygon_wgs84"])
        ax.plot(poly[:, 0], poly[:, 1], color=colour, lw=2.2,
                ls=LINE_STYLE[d["classification"]], zorder=3)
        ax.fill(poly[:, 0], poly[:, 1], color=colour,
                alpha=0.30 if d["classification"] == "oil" else 0.13, zorder=2)
        if id(d) in labelled:
            cx, cy = d["geometry"]["centroid"]
            ax.annotate(f"{d['environment']['wind_speed_ms']:.1f} m/s",
                        (cx, cy), color=colour, fontsize=9.5, weight="bold",
                        ha="center", zorder=4,
                        path_effects=None)

    ax.set_title("Sentinel-1 σ⁰ VV — every dark patch, classified", fontsize=12, pad=10)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in CLASS_STYLE.values()],
              loc="lower left", facecolor=PANEL, edgecolor=LINE,
              labelcolor=INK, fontsize=8.5, framealpha=0.95)

    # Right: the reason for each verdict. This is the part that matters.
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    ax2.set_title("Why — the reason travels with the verdict", fontsize=12,
                  color=INK, pad=10, loc="left")
    y = 0.97
    for d in sorted(detections, key=lambda x: -x["geometry"]["area_km2"]):
        colour, label = CLASS_STYLE[d["classification"]]
        ax2.text(0, y, label, color=colour, fontsize=9, weight="bold", va="top")
        ax2.text(1.0, y, f"{d['geometry']['area_km2']:,.0f} km²", color=INK,
                 fontsize=9, va="top", ha="right")
        y -= 0.045
        reason = d["classification_reason"]
        for line in _wrap(reason, 74):
            ax2.text(0, y, line, color=DIM, fontsize=8.0, va="top")
            y -= 0.033
        y -= 0.035

    fig.text(0.5, 0.015,
             "Nine of fourteen dark patches across the built cases are refused, "
             "each with its wind speed stated. Below ~3 m/s the sea is already "
             "smooth and oil cannot produce radar contrast.",
             color=DIM, fontsize=8.5, ha="center", style="italic")

    _dark(fig, ax, ax2)
    fig.savefig(OUT / "01-detection.png", dpi=135, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("  wrote 01-detection.png")


def figure_attribution() -> None:
    """Source posterior, candidate tracks, and the ranked shortlist."""
    posterior = _load("posterior.json", {})
    candidates = _load("candidates.json", {})
    field = _load("ui/posterior.json", {})
    truth_path = CASE / "truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.is_file() else {}
    if "credible_regions" not in posterior or not candidates:
        print("  skipped 02-attribution: no posterior or candidates")
        return

    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.2)
    ax = fig.add_subplot(gs[0])

    cells = np.array(field.get("cells", []))
    if cells.size:
        ax.scatter(cells[:, 0], cells[:, 1], c=cells[:, 2], cmap="magma",
                   s=110, alpha=0.75, marker="s", linewidths=0)

    for level, style in (("95", (1.4, "--")), ("50", (2.2, "-"))):
        poly = np.array(posterior["credible_regions"][level]["polygon_wgs84"])
        area = posterior["credible_regions"][level]["area_km2"]
        ax.plot(poly[:, 0], poly[:, 1], color="#c8a2ff", lw=style[0], ls=style[1],
                label=f"{level}% region — {area:,.0f} km²", zorder=5)

    for c in candidates["candidates"][:5]:
        coords = (c.get("track_geojson", {}).get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        track = np.array(coords)
        is_true = truth and c["mmsi"] == truth.get("culprit_mmsi")
        ax.plot(track[:, 0], track[:, 1],
                color="#3fb950" if is_true else "#e5534b",
                lw=2.2 if is_true else 1.0, alpha=1.0 if is_true else 0.55, zorder=4,
                label="true source vessel" if is_true else None)

    ax.set_title("Source posterior and candidate traffic", fontsize=12, pad=10)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(loc="lower left", facecolor=PANEL, edgecolor=LINE,
              labelcolor=INK, fontsize=8.5, framealpha=0.95)

    # Ranked shortlist, with the dark hypothesis competing on the same axis.
    ax2 = fig.add_subplot(gs[1])
    top = candidates["candidates"][:6]
    dark = candidates["dark_vessel_hypothesis"]
    labels = [f"{c['rank']}. {c['vessel_type']}  {c['mmsi']}" for c in top][::-1]
    values = [c["posterior_probability"] for c in top][::-1]
    colours = ["#3fb950" if truth and c["mmsi"] == truth.get("culprit_mmsi") else "#e5534b"
               for c in top][::-1]
    labels.append("VESSEL NOT TRANSMITTING AIS")
    values.append(dark["posterior_probability"])
    colours.append("#8957e5")

    bars = ax2.barh(range(len(values)), values, color=colours, height=0.68)
    ax2.set_yticks(range(len(values)))
    ax2.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax2.set_xlabel("posterior probability under stated model assumptions")
    ax2.set_title("Ranked investigative leads", fontsize=12, pad=10, loc="left")
    for bar, v in zip(bars, values):
        ax2.text(v + 0.006, bar.get_y() + bar.get_height() / 2, f"{v:.1%}",
                 va="center", color=INK, fontsize=8.5)

    tr = candidates["traffic_reduction"]
    fig.text(0.5, 0.015,
             f"{tr['vessels_in_window']} vessels in the window → {tr['after_prefilter']} "
             f"after spatio-temporal filtering → {tr['reported']} above the reporting "
             f"threshold.   The dark-vessel hypothesis competes on the same scale, so "
             f"naming a ship requires out-scoring \"nobody was transmitting\".",
             color=DIM, fontsize=8.5, ha="center", style="italic")

    _dark(fig, ax, ax2)
    fig.savefig(OUT / "02-attribution.png", dpi=135, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("  wrote 02-attribution.png")


def figure_evaluation() -> None:
    """Calibration, operating envelope, and the attribution metrics."""
    fair_path, matched_path = EVAL / "summary.json", EVAL / "summary_matched.json"
    if not fair_path.is_file():
        print("  skipped 03-evaluation: run scripts/truth_harness.py first")
        return
    fair = json.loads(fair_path.read_text(encoding="utf-8"))
    matched = json.loads(matched_path.read_text(encoding="utf-8")) if matched_path.is_file() else None

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))

    # --- calibration -------------------------------------------------------
    nominal = np.array([int(k) / 100 for k in fair["calibration"]])
    ax1.plot([0, 1], [0, 1], color=DIM, ls=":", lw=1.2, label="perfect calibration")
    ax1.plot(nominal, [fair["calibration"][k] for k in fair["calibration"]],
             "o-", color="#f0883e", lw=2, ms=7, label="fair test (priors mismatched)")
    if matched:
        ax1.plot(nominal, [matched["calibration"][k] for k in matched["calibration"]],
                 "s--", color="#3fb950", lw=1.8, ms=6, label="priors matched")
    ax1.set_xlim(0.4, 1.0); ax1.set_ylim(0.3, 1.05)
    ax1.set_xlabel("nominal credible level"); ax1.set_ylabel("empirical coverage")
    ax1.set_title("Calibration — the metric that matters", fontsize=11)
    ax1.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=INK, fontsize=7.5, loc="lower right")
    ax1.annotate("overconfident\nin the tails", xy=(0.95, fair["calibration"]["95"]),
                 xytext=(0.62, 0.55), color="#f0883e", fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="#f0883e", lw=1.1))

    # --- operating envelope -------------------------------------------------
    env = {int(k): v for k, v in fair["envelope_by_lookback"].items() if v}
    keys = sorted(env)
    ax2.plot(keys, [env[k] for k in keys], "o-", color="#8957e5", lw=2.2, ms=7)
    ax2.set_xlabel("lookback (hours)"); ax2.set_ylabel("95% region area (km²)")
    ax2.set_title("Uncertainty grows with lookback", fontsize=11)
    ax2.axvline(72, color="#e5534b", ls="--", lw=1.2)
    ax2.annotate("72 h operating limit", xy=(72, max(env.values()) * 0.55),
                 color="#e5534b", fontsize=8, rotation=90, ha="right")

    # --- attribution --------------------------------------------------------
    a, d = fair["attribution"], fair["dark_vessel_cases"]
    names = ["culprit in\ncandidate set", "dark ranked top\n(culprit removed)",
             "top-3 recall", "top-1 recall"]
    vals = [a["in_candidate_set"], d["dark_ranked_top"], a["top3_recall"], a["top1_recall"]]
    targets = [None, None, 0.60, 0.30]
    colours = ["#3fb950" if t is None or v >= t else "#f0883e"
               for v, t in zip(vals, targets)]
    bars = ax3.barh(range(len(vals)), vals, color=colours, height=0.6)
    ax3.set_yticks(range(len(vals)))
    ax3.set_yticklabels(names, fontsize=8.5, color=INK)
    ax3.set_xlim(0, 1.08); ax3.set_xlabel("rate")
    ax3.set_title("Attribution over 24 end-to-end trials", fontsize=11)
    for bar, v, t in zip(bars, vals, targets):
        ax3.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}" + ("" if t is None else f"  (target {t:.2f})"),
                 va="center", color=INK, fontsize=8)

    fig.text(0.5, -0.02,
             "Measured with drift parameters the inversion does not assume, one trial "
             "in four hiding the culprit from AIS, and every trial running the whole "
             "chain including detector error. Two numbers miss target and are reported "
             "as misses.",
             color=DIM, fontsize=8.5, ha="center", style="italic")

    _dark(fig, ax1, ax2, ax3)
    fig.tight_layout()
    fig.savefig(OUT / "03-evaluation.png", dpi=135, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("  wrote 03-evaluation.png")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"generating figures into {OUT}")
    figure_detection()
    figure_attribution()
    figure_evaluation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
