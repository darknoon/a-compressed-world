# A Compressed World Autoresearch

You are running autonomous compression research for PMTiles data.

## Goal

Compress a fixed Monaco map subset: every tile that intersects the Monaco bounding box,
including the low-zoom tiles above it and all more detailed zooms present in the source
PMTiles archive. The current metric is bits per byte (`bpb`) over the exact prepared byte
stream. Lower is better.

Important: the current `bpb` is only the neural model's cross-entropy estimate. It is not
yet an actual compressed file. A major research milestone is to add an arithmetic/range
coder that uses the model probabilities to write a compressed artifact, then decode it and
verify byte-for-byte reconstruction of `data/processed/monaco_pmtiles/data.bin`.

This is inspired by:

- Karpathy `autoresearch`: keep the repo small, use `program.md` as the human-authored
  research instructions, and let agents edit one self-contained `train.py`.
- Keller Jordan `modded-nanogpt`: favor fast GPU-friendly data loading, simple direct
  training scripts, and records that later agents can inspect.
- Bellard NNCP: use a neural predictor as the probability model for lossless compression.

## Files

- `prepare.py`: fixed data prep, PMTiles tile selection, dataloader, and `evaluate_bpb`.
  Do not edit this during experiments unless the human explicitly changes the task.
- `download.py`: downloads the default Monaco PMTiles cutout to `data/raw/monaco.pmtiles`.
  Use it only if the raw file is missing.
- `train.py`: the only file agents edit. It contains the model, optimizer, and training
  loop in one place. It prints progress and final metrics to stdout only.
- `record.py`: parses a redirected `train.py` log and updates `records/results.tsv` plus
  a JSON summary. Do not move this logic into `train.py`.
- `records/`: previous experiment summaries and full training logs. Read
  `records/results.tsv`, recent JSON summaries, and relevant `*.log` files before
  proposing a new change.
- `README.md`: setup and command reference.

## Setup

1. Confirm `uv sync` has run.
2. Confirm data exists by checking `data/processed/monaco_pmtiles/data.bin`.
3. If prepared data does not exist, confirm `data/raw/monaco.pmtiles` exists. If it does
   not, run `uv run download.py`. Then run:

```bash
uv run prepare.py
```

4. Run the baseline before changing `train.py`:

```bash
stamp=$(date +%Y%m%d-%H%M%S)
uv run train.py > records/$stamp.log 2>&1
uv run record.py records/$stamp.log --description "baseline"
```

For setup-only smoke tests, use `uv run train.py --steps 3` with small `ACW_*`
overrides. Do not use smoke-test results as research records.

## Experiment Loop

Repeat:

1. Read prior results in `records/results.tsv`.
2. Make one focused edit to `train.py`.
3. Run `stamp=$(date +%Y%m%d-%H%M%S); uv run train.py > records/$stamp.log 2>&1`.
4. If the run completed, run `uv run record.py records/$stamp.log --description "..."`
   with a short description of the experiment.
5. Extract `bpb` and `peak_vram_mb` from the saved log or `records/results.tsv`.
6. Keep the change only if it improves `bpb` enough to justify the added complexity.
7. Prefer simple changes: learning rate, batch size, depth, width, context use, optimizer,
   activation, normalization, attention variants, and training schedule.

## Compression Correctness

The project is not complete until it can round-trip the prepared byte stream:

1. Encode `data/processed/monaco_pmtiles/data.bin` with the trained model and an
   arithmetic/range coder.
2. Decode the compressed artifact with the same checkpoint.
3. Verify the decoded bytes exactly match the original `data.bin`.
4. Report actual compressed bytes per input byte alongside the estimated `bpb`.

Until that exists, treat `bpb` as a research proxy only. Do not claim actual compression or
reconstruction success from cross-entropy alone.

## Rules

- Do not edit `prepare.py` during normal autoresearch.
- Do not move the model into a package or library. Keep `train.py` self-contained.
- Do not add dependencies unless the human asks.
- Use `uv` for Python commands.
- If a run crashes, inspect the last part of its saved log, fix simple mistakes, and retry.
- If an idea is ugly and only improves by noise, discard it.
