"""Fetch the real datasets and report what is ready.

    python scripts/fetch_real_data.py --check          # nothing is downloaded
    python scripts/fetch_real_data.py --zenodo part3   # ~9.9 GB
    python scripts/fetch_real_data.py --cmems --era5 --case-config configs/cases/kattegat.yaml

``--check`` is the default and downloads nothing. Everything here is free, but
free is not the same as small: the Zenodo archives total roughly 96 GB across
the three parts, so no download starts without being asked for by name.

Credentials live in ``.env`` and are never committed. All four services are free
and need registration only -- see ``.env.example``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Zenodo records for the dataset named in the problem statement.
# Trujillo-Acatitla et al., CC-BY-4.0.
ZENODO_PARTS = {
    "part1": {"record": 8253899, "gb": 45.9, "what": "train images + ground truth"},
    "part2": {"record": 8346860, "gb": 40.7, "what": "validation images + ground truth"},
    "part3": {"record": 13761290, "gb": 9.9, "what": "test images + ground truth"},
}

CREDENTIALS = {
    "Copernicus Data Space (Sentinel-1 GRD)": ("CDSE_USERNAME", "CDSE_PASSWORD"),
    "Copernicus Marine (CMEMS currents)": ("CMEMS_USERNAME", "CMEMS_PASSWORD"),
    "Climate Data Store (ERA5 wind)": ("CDS_API_KEY",),
}

OPTIONAL_PACKAGES = {
    "copernicusmarine": "CMEMS currents",
    "cdsapi": "ERA5 wind",
    "py7zr": "extracting the Zenodo .7z archives",
}


@dataclass
class Readiness:
    credentials: dict
    packages: dict
    on_disk: dict

    @property
    def can_build_real_case(self) -> bool:
        return all(self.credentials.values()) and self.packages.get("copernicusmarine", False)


def check() -> Readiness:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    creds = {
        name: all(os.environ.get(k) for k in keys)
        for name, keys in CREDENTIALS.items()
    }
    packages = {}
    for pkg in OPTIONAL_PACKAGES:
        try:
            __import__(pkg)
            packages[pkg] = True
        except ImportError:
            packages[pkg] = False

    raw = ROOT / "data" / "raw"
    on_disk = {
        "zenodo": sorted(p.name for p in (raw / "zenodo").glob("*")) if (raw / "zenodo").is_dir() else [],
        "ais": sorted(p.name for p in (raw / "ais").glob("*")) if (raw / "ais").is_dir() else [],
        "forcing": sorted(p.name for p in (raw / "forcing").glob("*")) if (raw / "forcing").is_dir() else [],
    }
    return Readiness(credentials=creds, packages=packages, on_disk=on_disk)


def report(r: Readiness) -> None:
    print("CREDENTIALS  (all free; registration only -- see .env.example)")
    for name, ok in r.credentials.items():
        print(f"  {'OK ' if ok else '-- '} {name}")

    print("\nOPTIONAL PACKAGES")
    for pkg, why in OPTIONAL_PACKAGES.items():
        print(f"  {'OK ' if r.packages.get(pkg) else '-- '} {pkg:20s} {why}")

    print("\nRAW DATA ON DISK")
    for kind, files in r.on_disk.items():
        print(f"  {kind:10s} {len(files)} item(s)" + (f"  {files[:3]}" if files else ""))

    print("\nZENODO ARCHIVES  (free, CC-BY-4.0, but large)")
    total = sum(p["gb"] for p in ZENODO_PARTS.values())
    for name, p in ZENODO_PARTS.items():
        print(f"  {name}  {p['gb']:5.1f} GB  record {p['record']}  {p['what']}")
    print(f"  total {total:.1f} GB")

    print("\nWHAT EACH SOURCE UNLOCKS")
    print("  Zenodo tiles      segmentation training + IoU benchmarking ONLY.")
    print("                    They are image crops with no CRS, so they cannot be")
    print("                    drifted or correlated with AIS.")
    print("  CDSE + CMEMS +    end-to-end real cases. All three are needed together:")
    print("  ERA5 + AIS        a scene without forcing cannot be inverted, and an")
    print("                    inversion without AIS cannot be attributed.")

    if not r.can_build_real_case:
        print("\nNOT READY for a real end-to-end case. The pipeline runs today on")
        print("synthetic cases; see scripts/build_synthetic_case.py.")


def fetch_zenodo(part: str, dest: Path) -> int:
    """Download one Zenodo part. Large -- asked for by name, never implied."""
    import json
    import urllib.request

    spec = ZENODO_PARTS[part]
    url = f"https://zenodo.org/api/records/{spec['record']}"
    with urllib.request.urlopen(url, timeout=60) as fh:
        record = json.load(fh)

    dest.mkdir(parents=True, exist_ok=True)
    for f in record.get("files", []):
        target = dest / f["key"]
        if target.exists():
            print(f"  already present: {target.name}")
            continue
        link = f["links"]["self"]
        size_gb = f.get("size", 0) / 1e9
        print(f"  downloading {f['key']}  ({size_gb:.1f} GB) ...")
        urllib.request.urlretrieve(link, target)
        print(f"  saved {target}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report readiness, download nothing")
    ap.add_argument("--zenodo", choices=sorted(ZENODO_PARTS), help="download one Zenodo part")
    ap.add_argument("--verify-zenodo", type=Path,
                    help="check an extracted archive against our format assumptions")
    args = ap.parse_args(argv)

    if args.verify_zenodo:
        from src.ingest.zenodo import verify_dataset

        result = verify_dataset(args.verify_zenodo)
        print(f"checked {result['checked']} tiles from {result['root']}")
        print(f"  georeferenced: {result['georeferenced']}/{result['checked']}")
        for problem in result["problems"]:
            print(f"  PROBLEM: {problem}")
        if "note" in result:
            print(f"\n  {result['note']}")
        return 1 if result["problems"] else 0

    if args.zenodo:
        spec = ZENODO_PARTS[args.zenodo]
        print(f"fetching Zenodo {args.zenodo}: {spec['gb']} GB, {spec['what']}")
        return fetch_zenodo(args.zenodo, ROOT / "data" / "raw" / "zenodo" / args.zenodo)

    report(check())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
