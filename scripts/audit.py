"""Pre-demo audit: overclaiming, secrets, and the offline guarantee.

    python scripts/audit.py

The checklist in Context/security_review.md section 7 was written as something a
person reads and ticks. This runs it instead. A checklist that is executed on
every change catches what a checklist read once under pressure does not.

The most important check is the first one. This system produces **accusations**
-- ranked real vessels, with real MMSI, as probable polluters -- on a model that
has a 95% credible region measured in thousands of square kilometres and no
end-to-end validation on real spills. The language it uses is not decoration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Language that asserts responsibility rather than reporting a lead. Checked in
# code, in the interface, and in generated output.
BANNED = (
    "guilty",
    "culprit was",
    "the culprit is",
    "responsible for the spill",
    "the polluter is",
    "identified the vessel",
    "confirmed spill by",
    "proven",
    "definitely the",
)

# Phrases that must be present in anything a viewer sees.
REQUIRED_UI = (
    "not a determination of responsibility",
    "under stated model assumptions",
)

# Credential-shaped strings that must never reach the repository.
SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]{8,}"),
    re.compile(r"(?i)\bcds_api_key\s*[:=]\s*\S{10,}"),
)

TEXT_SUFFIXES = {".py", ".html", ".md", ".json", ".yaml", ".yml", ".txt"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "data", "models", "eval", "vendor"}


@dataclass
class Audit:
    passed: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    warned: list = field(default_factory=list)

    def ok(self, name: str, detail: str = "") -> None:
        self.passed.append((name, detail))

    def fail(self, name: str, detail: str) -> None:
        self.failed.append((name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.warned.append((name, detail))


def _sources():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


# Files that must contain the banned terms in order to forbid them, plus the
# documentation that explains why they are forbidden.
LANGUAGE_EXEMPT = {
    "src/contracts/candidates.py",   # defines BANNED and rejects it
    "scripts/audit.py",              # this file
}


def _string_literals(source: str):
    """Yield every string literal in a Python source file.

    Parsed with ``ast`` rather than matched with a regex. A regex over quoted
    spans has to reason about escapes, triple quotes, f-strings and raw
    prefixes, and the first attempt here silently matched nothing at all --
    the audit reported a clean pass while checking zero literals (P-24).
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def check_language(a: Audit) -> None:
    """SC-2. The system emits investigative leads, not determinations.

    Scoped to what a **viewer** sees: the interface, and the JSON the pipeline
    produces. Internal vocabulary is not the control -- "culprit" is a perfectly
    good variable name in an evaluation harness, and banning it there would make
    this check cry wolf, which is worse than not running it.
    """
    hits = []

    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8").lower()
    for term in BANNED:
        if term in html:
            hits.append(f"web/index.html: {term!r}")

    for output in ROOT.glob("data/cases/*/out/*.json"):
        text = output.read_text(encoding="utf-8", errors="ignore").lower()
        for term in BANNED:
            if term in text:
                hits.append(f"{output.relative_to(ROOT).as_posix()}: {term!r}")

    # Python string literals only: an identifier or comment is not user-facing.
    for path in (ROOT / "src").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in LANGUAGE_EXEMPT:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        for body in _string_literals(source):
            lowered = body.lower()
            for term in BANNED:
                if term in lowered:
                    hits.append(f"{rel}: {term!r}")

    if hits:
        a.fail("no accusatory language", "; ".join(sorted(set(hits))[:6]))
    else:
        a.ok("no accusatory language",
             f"{len(BANNED)} terms checked in the UI, case output and string literals")


def check_ui_disclaimers(a: Audit) -> None:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8").lower()
    missing = [p for p in REQUIRED_UI if p not in html]
    if missing:
        a.fail("interface states its limits", f"missing: {missing}")
    else:
        a.ok("interface states its limits", "disclaimer and assumptions present")


def check_dark_hypothesis_rendered(a: Audit) -> None:
    """SC-1. Without it the system names an innocent ship whenever the real
    polluter was not transmitting."""
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    if "dark_vessel_hypothesis" not in html or "not transmitting" not in html.lower():
        a.fail("dark-vessel hypothesis is displayed", "not found in the interface")
        return
    shown = 0
    for candidates in ROOT.glob("data/cases/*/out/candidates.json"):
        payload = json.loads(candidates.read_text(encoding="utf-8"))
        if "dark_vessel_hypothesis" not in payload:
            a.fail("dark-vessel hypothesis is displayed", f"absent from {candidates}")
            return
        shown += 1
    a.ok("dark-vessel hypothesis is displayed", f"in the UI and in {shown} case output(s)")


def check_abstention_reasons(a: Audit) -> None:
    """Showing what the system refuses to judge is the strongest thing it does."""
    total = abstained = 0
    for path in ROOT.glob("data/cases/*/out/detections.json"):
        for d in json.loads(path.read_text(encoding="utf-8")):
            total += 1
            if d.get("abstained"):
                abstained += 1
                if "detectability window" not in d.get("classification_reason", ""):
                    a.fail("abstentions state their reason", f"{d['detection_id']} has none")
                    return
    if total == 0:
        a.warn("abstentions state their reason", "no case output to check")
    else:
        a.ok("abstentions state their reason", f"{abstained}/{total} detections abstained")


def check_secrets(a: Audit) -> None:
    if (ROOT / ".env").is_file():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if tracked.returncode == 0:
            a.fail("secrets are not committed", ".env is tracked by git")
            return

    hits = []
    for path in _sources():
        rel = path.relative_to(ROOT).as_posix()
        if rel.endswith((".env.example", "audit.py")):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(rel)
    if hits:
        a.fail("secrets are not committed", f"credential-shaped strings in {set(hits)}")
    else:
        a.ok("secrets are not committed", ".env untracked, no credential patterns")


def check_data_not_committed(a: Audit) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "data/"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    unexpected = [f for f in tracked if not f.endswith(".gitkeep")]
    if unexpected:
        a.fail("no datasets committed", f"{len(unexpected)} file(s) under data/")
    else:
        a.ok("no datasets committed", "data/ holds only .gitkeep")


def check_api_binding(a: Audit) -> None:
    """No auth layer + attribution output = loopback only."""
    from src.api.main import serve

    try:
        serve(host="0.0.0.0")
    except ValueError:
        a.ok("API refuses non-loopback binding", "enforced in serve()")
    except Exception as exc:  # pragma: no cover
        a.fail("API refuses non-loopback binding", f"unexpected: {exc!r}")
    else:
        a.fail("API refuses non-loopback binding", "0.0.0.0 was accepted")


def check_path_traversal(a: Audit) -> None:
    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)
    for attempt in ("../../etc/passwd", "..", "%2e%2e"):
        if client.get(f"/api/cases/{attempt}/detections").status_code == 200:
            a.fail("case id is validated", f"{attempt!r} was served")
            return
    a.ok("case id is validated", "traversal attempts rejected")


def check_synthetic_mmsi(a: Audit) -> None:
    """W-14. A real vessel appearing in an accusation demo would be serious."""
    from src.ingest.synthetic_ais import REAL_MID_MAX, REAL_MID_MIN, synthetic_mmsi

    bad = [
        m for m in (synthetic_mmsi(i) for i in range(1, 500))
        if REAL_MID_MIN <= int(m[:3]) <= REAL_MID_MAX
    ]
    if bad:
        a.fail("synthetic MMSI cannot collide", f"{len(bad)} in the real range")
    else:
        a.ok("synthetic MMSI cannot collide", "all outside MID 201-775")


def check_offline_assets(a: Audit) -> None:
    vendored = list((ROOT / "web" / "vendor").glob("deck.gl*.min.js"))
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    remote = [
        line.strip()[:80] for line in html.splitlines()
        if ("src=" in line or "href=" in line) and "http" in line
    ]
    if not vendored:
        a.fail("no remote assets", "deck.gl is not vendored")
    elif remote:
        a.fail("no remote assets", f"page references {remote[0]}")
    else:
        a.ok("no remote assets", f"deck.gl vendored ({vendored[0].name})")


def check_seed_threading(a: Audit) -> None:
    """NFR-3. Every RNG derives from the case seed; no bare np.random."""
    # np.random.Generator is a TYPE, not a source of randomness -- annotations
    # referencing it are correct usage. Only the legacy global functions, which
    # are lowercase, bypass the seed.
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"np\.random\.([a-z]\w*)", text):
            if match.group(1) == "default_rng":
                continue
            offenders.append(f"{path.relative_to(ROOT).as_posix()}: np.random.{match.group(1)}")
    if offenders:
        a.fail("all randomness is seeded", "; ".join(offenders[:5]))
    else:
        a.ok("all randomness is seeded", "only default_rng, seeded per case")


def check_demo_cases_ready(a: Audit) -> None:
    ready = []
    for manifest in ROOT.glob("data/cases/*/out/ui/manifest.json"):
        ready.append(manifest.parents[2].name)
    if not ready:
        a.fail("demo cases are built", "no case has a UI bundle; run run_case.py")
    elif len(ready) < 2:
        a.warn("demo cases are built", f"only {ready}; the runbook expects a spare")
    else:
        a.ok("demo cases are built", ", ".join(sorted(ready)))


CHECKS = (
    ("SC-2  language", check_language),
    ("SC-3  disclaimers", check_ui_disclaimers),
    ("SC-1  dark hypothesis", check_dark_hypothesis_rendered),
    ("FR-3a abstention reasons", check_abstention_reasons),
    ("SEC   secrets", check_secrets),
    ("SEC   datasets", check_data_not_committed),
    ("SEC   API binding", check_api_binding),
    ("SEC   path traversal", check_path_traversal),
    ("W-14  synthetic MMSI", check_synthetic_mmsi),
    ("NFR-1 offline assets", check_offline_assets),
    ("NFR-3 seeded randomness", check_seed_threading),
    ("DEMO  cases built", check_demo_cases_ready),
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    a = Audit()
    for label, check in CHECKS:
        try:
            check(a)
        except Exception as exc:  # a broken check is itself a finding
            a.fail(label, f"check raised {exc!r}")

    if args.json:
        print(json.dumps({
            "passed": [n for n, _ in a.passed],
            "warned": [{"check": n, "detail": d} for n, d in a.warned],
            "failed": [{"check": n, "detail": d} for n, d in a.failed],
        }, indent=2))
        return 1 if a.failed else 0

    print("PRE-DEMO AUDIT\n" + "=" * 62)
    for name, detail in a.passed:
        print(f"  PASS  {name:34s} {detail}")
    for name, detail in a.warned:
        print(f"  WARN  {name:34s} {detail}")
    for name, detail in a.failed:
        print(f"  FAIL  {name:34s} {detail}")
    print("=" * 62)
    print(f"{len(a.passed)} passed, {len(a.warned)} warnings, {len(a.failed)} failures")
    if a.failed:
        print("\nDo not demo until the failures above are resolved.")
    return 1 if a.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
