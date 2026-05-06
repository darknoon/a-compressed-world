"""
Fixed data preparation and evaluation harness for PMTiles compression experiments.

Usage:
  uv run prepare.py --input data/raw/monaco.pmtiles

The output is a contiguous byte stream in data/processed/monaco_pmtiles/.
Each selected tile is encoded as:
  b"PMTILE\\0" + tile_id:u64le + payload_len:u32le + payload

This keeps the first target simple: model the exact byte sequence for the Monaco map
subset, including every tile intersecting Monaco at every available zoom.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Fixed constants. Agents should not edit this file during autoresearch runs.
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024
TIME_BUDGET = int(os.environ.get("ACW_TIME_BUDGET", "300"))
EVAL_TOKENS = int(os.environ.get("ACW_EVAL_TOKENS", str(2 * 1024 * 1024)))
VOCAB_SIZE = 256

CACHE_DIR = Path(os.environ.get("ACW_CACHE_DIR", "data/processed")).expanduser()
DATASET_DIR = CACHE_DIR / "monaco_pmtiles"
DATA_BIN = DATASET_DIR / "data.bin"
METADATA_JSON = DATASET_DIR / "metadata.json"

MONACO_BBOX = (7.4090, 43.7247, 7.4399, 43.7519)  # west, south, east, north
DEFAULT_INPUT = Path("data/raw/monaco.pmtiles")
RECORD_MAGIC = b"PMTILE\0"


@dataclass(frozen=True)
class TileEntry:
    tile_id: int
    run_length: int
    offset: int
    length: int
    is_dir: bool


@dataclass(frozen=True)
class Header:
    root_offset: int
    root_length: int
    json_offset: int
    json_length: int
    leaf_offset: int
    tile_data_offset: int
    min_zoom: int
    max_zoom: int
    internal_compression: int
    tile_compression: int


def read_header(f: BinaryIO) -> Header:
    f.seek(0)
    raw = f.read(127)
    if len(raw) != 127 or raw[:7] != b"PMTiles":
        raise ValueError("input is not a PMTiles v3 file")
    if raw[7] != 3:
        raise ValueError(f"unsupported PMTiles version {raw[7]}; expected v3")
    return Header(
        root_offset=struct.unpack_from("<Q", raw, 8)[0],
        root_length=struct.unpack_from("<Q", raw, 16)[0],
        json_offset=struct.unpack_from("<Q", raw, 24)[0],
        json_length=struct.unpack_from("<Q", raw, 32)[0],
        leaf_offset=struct.unpack_from("<Q", raw, 40)[0],
        tile_data_offset=struct.unpack_from("<Q", raw, 56)[0],
        internal_compression=raw[97],
        tile_compression=raw[98],
        min_zoom=raw[100],
        max_zoom=raw[101],
    )


def decompress_pmtiles(data: bytes, compression: int) -> bytes:
    """Apply a PMTiles compression code (1=none, 2=gzip) to a blob.

    Used for both internal directory bytes and tile payloads.
    """
    if compression == 1:
        return data
    if compression == 2:
        return gzip.decompress(data)
    raise ValueError(
        f"unsupported PMTiles compression code {compression}; "
        "this KISS harness currently supports none and gzip"
    )


# Backwards-compatible alias for the directory-only callers.
decompress_directory = decompress_pmtiles


def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if b < 0x80:
            return value, pos
        shift += 7


def deserialize_directory(data: bytes) -> list[TileEntry]:
    num_entries, pos = read_varint(data, 0)
    tile_ids: list[int] = []
    last = 0
    for _ in range(num_entries):
        delta, pos = read_varint(data, pos)
        last += delta
        tile_ids.append(last)

    run_lengths = []
    for _ in range(num_entries):
        value, pos = read_varint(data, pos)
        run_lengths.append(value)

    lengths = []
    for _ in range(num_entries):
        value, pos = read_varint(data, pos)
        lengths.append(value)

    offsets = []
    for i in range(num_entries):
        value, pos = read_varint(data, pos)
        if value == 0 and i > 0:
            offsets.append(offsets[i - 1] + lengths[i - 1])
        else:
            offsets.append(value - 1)

    return [
        TileEntry(tile_ids[i], run_lengths[i], offsets[i], lengths[i], run_lengths[i] == 0)
        for i in range(num_entries)
    ]


def hilbert_xy_to_d(n: int, x: int, y: int) -> int:
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if x & s else 0
        ry = 1 if y & s else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s //= 2
    return d


def tile_id(z: int, x: int, y: int) -> int:
    zoom_offset = (4**z - 1) // 3
    return zoom_offset + hilbert_xy_to_d(1 << z, x, y)


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 1 << z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def selected_tile_ids(min_zoom: int, max_zoom: int, bbox: tuple[float, float, float, float]) -> set[int]:
    west, south, east, north = bbox
    ids: set[int] = set()
    for z in range(min_zoom, max_zoom + 1):
        x0, y0 = lonlat_to_tile(west, north, z)
        x1, y1 = lonlat_to_tile(east, south, z)
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                ids.add(tile_id(z, x, y))
    return ids


def read_directory_at(f: BinaryIO, header: Header, offset: int, length: int, base: int) -> list[TileEntry]:
    f.seek(base + offset)
    return deserialize_directory(decompress_directory(f.read(length), header.internal_compression))


def collect_entries(f: BinaryIO, header: Header) -> list[TileEntry]:
    root = read_directory_at(f, header, header.root_offset, header.root_length, 0)
    entries: list[TileEntry] = []
    leaves = [entry for entry in root if entry.is_dir]
    entries.extend(entry for entry in root if not entry.is_dir)
    for leaf in leaves:
        entries.extend(read_directory_at(f, header, leaf.offset, leaf.length, header.leaf_offset))
    return entries


def tile_records(pmtiles_path: Path, bbox: tuple[float, float, float, float]) -> tuple[list[bytes], dict]:
    with pmtiles_path.open("rb") as f:
        header = read_header(f)
        wanted = selected_tile_ids(header.min_zoom, header.max_zoom, bbox)
        records: list[bytes] = []
        matched_tiles = 0
        stored_payload_bytes = 0
        decoded_payload_bytes = 0
        for entry in collect_entries(f, header):
            ids = range(entry.tile_id, entry.tile_id + max(entry.run_length, 1))
            if not any(tid in wanted for tid in ids):
                continue
            f.seek(header.tile_data_offset + entry.offset)
            stored = f.read(entry.length)
            # Strip the per-tile transport compression so the model sees raw
            # tile bytes (MVT protobuf for tile_type=1) rather than gzipped
            # near-uniform noise. Without this the byte stream is already
            # entropy-coded and bpb floors near 8.
            payload = decompress_pmtiles(stored, header.tile_compression)
            stored_payload_bytes += len(stored)
            decoded_payload_bytes += len(payload)
            for tid in ids:
                if tid in wanted:
                    # Decode is intentionally omitted; the ID in the byte stream is enough for now.
                    matched_tiles += 1
                    records.append(
                        RECORD_MAGIC
                        + struct.pack("<Q", tid)
                        + struct.pack("<I", len(payload))
                        + payload
                    )

    metadata = {
        "source": str(pmtiles_path),
        "bbox": bbox,
        "records": len(records),
        "matched_tiles": matched_tiles,
        "bytes": sum(len(r) for r in records),
        "tile_compression": header.tile_compression,
        "stored_payload_bytes": stored_payload_bytes,
        "decoded_payload_bytes": decoded_payload_bytes,
    }
    return records, metadata


def prepare(input_path: Path, bbox: tuple[float, float, float, float]) -> None:
    t0 = time.time()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    records, metadata = tile_records(input_path, bbox)
    if not records:
        raise RuntimeError("no tiles matched the requested bounding box")

    data = b"".join(records)
    DATA_BIN.write_bytes(data)
    metadata |= {
        "data_path": str(DATA_BIN),
        "data_records": len(records),
        "data_bytes": len(data),
        "created_sec": round(time.time() - t0, 2),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


class ByteLoader:
    def __init__(self, path: Path, batch_size: int, seq_len: int):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        if self.data.size <= seq_len + 1:
            raise ValueError(f"{path} has only {self.data.size} bytes; need more data")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.cpu = torch.empty((2, batch_size, seq_len), dtype=torch.long, pin_memory=True)
        self.gpu = torch.empty((2, batch_size, seq_len), dtype=torch.long, device="cuda")

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = np.random.randint(0, self.data.size - self.seq_len - 1, size=self.batch_size)
        for row, start in enumerate(starts):
            chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
            self.cpu[0, row].copy_(torch.from_numpy(chunk[:-1]))
            self.cpu[1, row].copy_(torch.from_numpy(chunk[1:]))
        self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[0], self.gpu[1]


def make_dataloader(batch_size: int, seq_len: int) -> ByteLoader:
    return ByteLoader(DATA_BIN, batch_size, seq_len)


@torch.no_grad()
def evaluate_bpb(model: torch.nn.Module, batch_size: int) -> float:
    model.eval()
    loader = make_dataloader(batch_size, MAX_SEQ_LEN)
    steps = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y = loader.next_batch()
        loss = model(x, y, reduction="none")
        total_nats += float(loss.sum().item())
        total_bytes += int(y.numel())
    model.train()
    return total_nats / (math.log(2) * total_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Monaco PMTiles byte data")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        type=Path,
        help=f"Path to a PMTiles v3 archive (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--bbox", nargs=4, type=float, default=MONACO_BBOX)
    args = parser.parse_args()
    prepare(args.input, tuple(args.bbox))


if __name__ == "__main__":
    main()
