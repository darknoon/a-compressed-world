# A Compressed World

KISS autoresearch setup for transformer-based lossless modeling of PMTiles data. The shape
is borrowed from [Karpathy autoresearch](https://github.com/karpathy/autoresearch):
`prepare.py` is fixed, `train.py` is the self-contained file agents edit, and `program.md`
is the human-authored research brief. The GPU data-loading style follows the spirit of
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), while the compression
goal is inspired by [NNCP](https://bellard.org/nncp/).

The first target is the Monaco map subset: every tile intersecting the Monaco bounding box
at every zoom present in the PMTiles archive, including the more zoomed-out tiles above it.
The current metric is bits per byte (`bpb`) on the prepared byte stream.

Current status: `bpb` is a cross-entropy estimate, not an actual compressed artifact yet.
The next compressor milestone is an arithmetic/range coder that can encode
`data/processed/monaco_pmtiles/data.bin`, decode it, and verify byte-for-byte
reconstruction.

## Setup

This project uses `uv` for all Python dependency management.

```bash
uv sync
```

Download the Monaco PMTiles cutout, then prepare it:

```bash
uv run download.py
uv run prepare.py
```

`download.py` extracts the Monaco bbox from the public Protomaps v4 OpenStreetMap mirror
on Source Cooperative into `data/raw/monaco.pmtiles`. Use
`uv run prepare.py --input /path/to/archive.pmtiles` if the archive lives somewhere else.
The default bbox is Monaco.

Prepared data is written to `data/processed/monaco_pmtiles/data.bin`, a contiguous `uint8`
file that is cheap to memmap and copy to GPU.

## Run The Baseline

```bash
uv run train.py
```

For a quick smoke test, limit optimizer steps:

```bash
ACW_EVAL_TOKENS=8192 ACW_BATCH_SIZE=4 ACW_DEPTH=1 ACW_DIM=64 ACW_HEADS=4 ACW_COMPILE=0 uv run train.py --steps 3
```

At the end it prints:

```text
---
bpb: ...
training_seconds: ...
peak_vram_mb: ...
total_tokens_M: ...
num_steps: ...
num_params_M: ...
```

For autoresearch runs, redirect stdout to a timestamped log and summarize it separately:

```bash
stamp=$(date +%Y%m%d-%H%M%S)
uv run train.py > records/$stamp.log 2>&1
uv run record.py records/$stamp.log --description "baseline"
```

`record.py` appends `records/results.tsv` and writes `records/<timestamp>.json`.

## Autoresearch

Point an agent at `program.md`. The intended loop is:

1. Read `README.md`, `prepare.py`, `train.py`, prior records, and relevant saved logs.
2. Run the baseline if no record exists.
3. Edit only `train.py`.
4. Run `train.py` into a timestamped log under `records/`, then record it with `record.py`.
5. Keep changes that improve `bpb` enough to justify their complexity.

## Reference

https://bellard.org/nncp/

## What To Improve Next

- Add a real range coder so checkpoints can produce actual lossless compressed artifacts,
  not just ideal size estimates from cross entropy, and verify exact reconstruction.
- Compare alternate prepared streams: payload-only, PMTiles directory+payload, z/x/y
  metadata-rich records, and raw archive ranges.
