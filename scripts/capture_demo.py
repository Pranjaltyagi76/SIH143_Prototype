"""Capture the demo interface for the README.

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000    # in another terminal
    python scripts/capture_demo.py

Drives the real interface in a real browser and saves what it sees, so the
README shows the demo rather than an artist's impression of it. Re-run after any
UI change and the screenshots follow.

Uses the Edge or Chrome already installed on the machine via Playwright's
``channel`` option, so nothing extra is downloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "screenshots"
VIEWPORT = {"width": 1600, "height": 950}

# The animation runs ~17 frames a second over the proposal pass. Waiting for it
# to finish means the trails are fully drawn rather than caught half-way.
ANIMATION_MS = 9000

# Captured at 2x for crisp text, then downscaled. A full-interface shot at 2x is
# ~3.4 MB, which is dead weight in a README that GitHub renders at under 1000 px
# anyway -- and it would bloat the repository on every re-capture.
MAX_WIDTH_PX = 1800


def _switch(page, case_id: str) -> None:
    page.evaluate(
        """(id) => { const s = document.getElementById('case-select');
                     s.value = id; s.onchange(); }""",
        case_id,
    )
    page.wait_for_timeout(ANIMATION_MS)


def capture(channel: str, base_url: str) -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    shots = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(base_url, wait_until="networkidle")
        page.wait_for_selector("#verdict .funnel", timeout=30_000)
        page.wait_for_timeout(ANIMATION_MS)

        # 1. The whole interface on the case that contains a spill.
        page.screenshot(path=OUT / "01-interface.png")
        shots.append("01-interface.png")

        # 2. The result block alone. This is the output that matters most, and
        #    it is now the first thing on the panel rather than three screens
        #    down, so it is worth showing on its own.
        page.locator("#verdict").screenshot(path=OUT / "02-result.png")
        shots.append("02-result.png")

        # 3. An abstention, expanded to show the reason. The system refusing to
        #    judge is the strongest thing it does, so the reason has to be visible.
        page.evaluate(
            """() => { const d = [...document.querySelectorAll('details.det')]
                         .find(x => x.querySelector('.tag.undetermined'));
                       if (d) d.open = true; }"""
        )
        page.wait_for_timeout(400)
        page.locator("#detections").screenshot(path=OUT / "03-abstention.png")
        shots.append("03-abstention.png")

        # 4. A candidate selected, so its AIS track is highlighted on the map.
        page.evaluate(
            """() => { const l = [...document.querySelectorAll('[data-lead]')]
                         .find(x => x.dataset.lead !== '__dark__');
                       if (l) l.click(); }"""
        )
        page.wait_for_timeout(1200)
        page.screenshot(path=OUT / "04-candidate-selected.png")
        shots.append("04-candidate-selected.png")

        # 5. A scene with no oil in it at all. A complete, correct answer.
        _switch(page, "synth_kattegat")
        page.screenshot(path=OUT / "05-clean-scene.png")
        shots.append("05-clean-scene.png")
        page.locator("#verdict").screenshot(path=OUT / "06-clean-result.png")
        shots.append("06-clean-result.png")

        browser.close()

    total_before = sum((OUT / n).stat().st_size for n in shots)
    for name in shots:
        _shrink(OUT / name)
    total_after = sum((OUT / n).stat().st_size for n in shots)

    print(f"captured {len(shots)} screenshots into {OUT}")
    for name in shots:
        size = (OUT / name).stat().st_size / 1024
        print(f"  {name:28s} {size:6.0f} KB")
    print(f"  total {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")
    return 0


def _shrink(path: Path) -> None:
    """Downscale to a sensible display width and re-encode."""
    from PIL import Image

    with Image.open(path) as img:
        if img.width > MAX_WIDTH_PX:
            height = round(img.height * MAX_WIDTH_PX / img.width)
            img = img.resize((MAX_WIDTH_PX, height), Image.LANCZOS)
        img.convert("RGB").save(path, optimize=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="msedge", help="msedge or chrome")
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    args = ap.parse_args(argv)

    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(args.url.rstrip("/") + "/healthz", timeout=5)
    except (urllib.error.URLError, OSError):
        print(f"no server at {args.url}. Start it first:")
        print("  uvicorn src.api.main:app --host 127.0.0.1 --port 8000")
        return 2

    return capture(args.channel, args.url)


if __name__ == "__main__":
    raise SystemExit(main())
