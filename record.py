"""
Parse a redirected train.py log and update records/.

Usage:
  uv run record.py records/20260506-180000.log --description "baseline"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUMMARY_KEYS = {
    "bpb",
    "training_seconds",
    "peak_vram_mb",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
    "dim",
    "heads",
}


def parse_log(path: Path) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {"log": str(path)}
    in_summary = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "---":
            in_summary = True
            continue
        if not in_summary or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key not in SUMMARY_KEYS:
            continue
        if key in {"num_steps", "depth", "dim", "heads"}:
            summary[key] = int(value)
        else:
            summary[key] = float(value)

    missing = sorted(SUMMARY_KEYS - summary.keys())
    if missing:
        raise ValueError(f"{path} is missing summary keys: {', '.join(missing)}")
    return summary


def append_record(log_path: Path, description: str) -> Path:
    records = Path("records")
    records.mkdir(exist_ok=True)
    timestamp = log_path.stem
    summary = parse_log(log_path)
    summary["description"] = description

    json_path = records / f"{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    tsv = records / "results.tsv"
    if not tsv.exists():
        tsv.write_text(
            "timestamp\tbpb\tpeak_vram_mb\ttraining_seconds\ttotal_tokens_M\tnum_steps\t"
            "num_params_M\tdepth\tdim\theads\tlog\tdescription\n",
            encoding="utf-8",
        )
    with tsv.open("a", encoding="utf-8") as f:
        f.write(
            f"{timestamp}\t{summary['bpb']:.6f}\t{summary['peak_vram_mb']:.1f}\t"
            f"{summary['training_seconds']:.1f}\t{summary['total_tokens_M']:.1f}\t"
            f"{summary['num_steps']}\t{summary['num_params_M']:.1f}\t{summary['depth']}\t"
            f"{summary['dim']}\t{summary['heads']}\t{log_path}\t{description}\n"
        )
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a completed train.py log")
    parser.add_argument("log", type=Path)
    parser.add_argument("--description", default="manual run")
    args = parser.parse_args()
    print(append_record(args.log, args.description))


if __name__ == "__main__":
    main()
