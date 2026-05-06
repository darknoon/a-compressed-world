"""
Download the default Monaco PMTiles cutout into data/raw/monaco.pmtiles.

This uses the pmtiles CLI to extract the Monaco bbox from the public Protomaps v4
OpenStreetMap mirror on Source Cooperative.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


PMTILES_VERSION = "1.30.1"
PMTILES_URL = (
    "https://github.com/protomaps/go-pmtiles/releases/download/"
    f"v{PMTILES_VERSION}/go-pmtiles_{PMTILES_VERSION}_Linux_x86_64.tar.gz"
)
SOURCE_URL = "https://data.source.coop/protomaps/openstreetmap/v4.pmtiles"
MONACO_BBOX = "7.4090,43.7247,7.4399,43.7519"
TOOLS_DIR = Path("tools")
PMTILES_BIN = TOOLS_DIR / "pmtiles"
DEFAULT_OUTPUT = Path("data/raw/monaco.pmtiles")


def ensure_pmtiles_cli() -> Path:
    system_pmtiles = shutil.which("pmtiles")
    if system_pmtiles:
        return Path(system_pmtiles)
    if PMTILES_BIN.exists():
        return PMTILES_BIN

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TOOLS_DIR / "go-pmtiles.tar.gz"
    print(f"downloading {PMTILES_URL}")
    urllib.request.urlretrieve(PMTILES_URL, archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(TOOLS_DIR, filter="data")
    archive_path.unlink()
    PMTILES_BIN.chmod(0o755)
    return PMTILES_BIN


def download(output: Path, bbox: str, source_url: str, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        print(f"{output} already exists; use --force to re-download")
        return

    pmtiles = ensure_pmtiles_cli()
    cmd = [
        str(pmtiles),
        "extract",
        source_url,
        str(output),
        f"--bbox={bbox}",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Monaco PMTiles source data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bbox", default=MONACO_BBOX)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.output, args.bbox, args.source_url, args.force)


if __name__ == "__main__":
    main()
