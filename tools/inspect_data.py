"""
Walk a prepared data.bin and decode the MVT (Mapbox Vector Tile) protobuf
in each tile record so we can sanity-check what we're actually compressing.

Usage:
  uv run tools/inspect_data.py                       # summary of monaco
  uv run tools/inspect_data.py --path data/processed_paris/monaco_pmtiles/data.bin
  uv run tools/inspect_data.py --tile 0              # dump a specific tile
  uv run tools/inspect_data.py --max-records 100     # cap how many records to walk

The MVT proto schema we care about (Mapbox vector_tile.proto):
  Tile {
    repeated Layer layers = 3;            // wire type 2
  }
  Layer {
    required string name = 1;             // wire type 2
    repeated Feature features = 2;        // wire type 2
    repeated string keys = 3;             // wire type 2
    repeated Value values = 4;            // wire type 2
    optional uint32 extent = 5 [default = 4096];
    required uint32 version = 15;
  }
  Feature {
    optional uint64 id = 1;
    repeated uint32 tags = 2 [packed=true];
    optional GeomType type = 3 [default = UNKNOWN];  // 0=UNKNOWN 1=POINT 2=LINESTRING 3=POLYGON
    repeated uint32 geometry = 4 [packed=true];
  }
"""

from __future__ import annotations

import argparse
import struct
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path("data/processed/monaco_pmtiles/data.bin")
RECORD_MAGIC = b"PMTILE\0"

GEOM_TYPES = {0: "UNKNOWN", 1: "POINT", 2: "LINESTRING", 3: "POLYGON"}


# ---------------------------------------------------------------------------
# Minimal protobuf wire reader.
# ---------------------------------------------------------------------------


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if b < 0x80:
            return value, pos
        shift += 7


def read_field(buf: bytes, pos: int) -> tuple[int, int, bytes | int, int]:
    """Read one (field_number, wire_type, value, new_pos)."""
    tag, pos = read_varint(buf, pos)
    field_no = tag >> 3
    wire = tag & 7
    if wire == 0:  # varint
        v, pos = read_varint(buf, pos)
        return field_no, wire, v, pos
    if wire == 2:  # length-delimited
        ln, pos = read_varint(buf, pos)
        v = buf[pos : pos + ln]
        return field_no, wire, v, pos + ln
    if wire == 5:  # 32-bit
        v = buf[pos : pos + 4]
        return field_no, wire, v, pos + 4
    if wire == 1:  # 64-bit
        v = buf[pos : pos + 8]
        return field_no, wire, v, pos + 8
    raise ValueError(f"unsupported wire type {wire} at pos {pos}")


# ---------------------------------------------------------------------------
# MVT decoding.
# ---------------------------------------------------------------------------


def decode_feature(buf: bytes) -> dict:
    feat = {"id": None, "type": 0, "n_tags": 0, "geom_len": 0}
    pos = 0
    while pos < len(buf):
        no, wire, v, pos = read_field(buf, pos)
        if no == 1 and wire == 0:
            feat["id"] = v
        elif no == 2 and wire == 2:
            feat["n_tags"] = sum(1 for _ in iter_varints(v))
        elif no == 3 and wire == 0:
            feat["type"] = v
        elif no == 4 and wire == 2:
            feat["geom_len"] = sum(1 for _ in iter_varints(v))
    return feat


def iter_varints(buf: bytes):
    pos = 0
    while pos < len(buf):
        v, pos = read_varint(buf, pos)
        yield v


def decode_layer(buf: bytes) -> dict:
    layer = {
        "name": None,
        "version": None,
        "extent": 4096,
        "features": [],
        "keys": [],
        "n_values": 0,
        "raw_bytes": len(buf),
    }
    pos = 0
    while pos < len(buf):
        no, wire, v, pos = read_field(buf, pos)
        if no == 1 and wire == 2:
            layer["name"] = v.decode("utf-8", errors="replace")
        elif no == 2 and wire == 2:
            layer["features"].append(decode_feature(v))
        elif no == 3 and wire == 2:
            layer["keys"].append(v.decode("utf-8", errors="replace"))
        elif no == 4 and wire == 2:
            layer["n_values"] += 1
        elif no == 5 and wire == 0:
            layer["extent"] = v
        elif no == 15 and wire == 0:
            layer["version"] = v
    return layer


def decode_tile(buf: bytes) -> list[dict]:
    layers = []
    pos = 0
    while pos < len(buf):
        no, wire, v, pos = read_field(buf, pos)
        if no == 3 and wire == 2:
            layers.append(decode_layer(v))
        # other top-level fields ignored
    return layers


# ---------------------------------------------------------------------------
# data.bin walker.
# ---------------------------------------------------------------------------


def walk_records(path: Path, max_records: int | None = None):
    buf = memoryview(path.read_bytes())
    i = 0
    n = 0
    while i < len(buf):
        if buf[i : i + 7].tobytes() != RECORD_MAGIC:
            raise ValueError(f"bad magic at offset {i}")
        tile_id = struct.unpack_from("<Q", buf, i + 7)[0]
        payload_len = struct.unpack_from("<I", buf, i + 15)[0]
        start = i + 19
        end = start + payload_len
        yield n, tile_id, bytes(buf[start:end])
        n += 1
        i = end
        if max_records is not None and n >= max_records:
            return


def summarize(path: Path, max_records: int | None) -> None:
    print(f"data: {path}  ({path.stat().st_size:,} bytes)")
    total_records = 0
    total_layer_bytes = 0
    total_features = 0
    layer_names: Counter[str] = Counter()
    geom_type_counts: Counter[str] = Counter()
    keys_seen: Counter[str] = Counter()
    largest_tiles: list[tuple[int, int, int]] = []  # (size, idx, tile_id)

    for idx, tile_id, payload in walk_records(path, max_records=max_records):
        total_records += 1
        layers = decode_tile(payload)
        for layer in layers:
            total_layer_bytes += layer["raw_bytes"]
            layer_names[layer["name"] or "<unknown>"] += 1
            total_features += len(layer["features"])
            for feat in layer["features"]:
                geom_type_counts[GEOM_TYPES.get(feat["type"], "UNKNOWN")] += 1
            for key in layer["keys"]:
                keys_seen[key] += 1
        largest_tiles.append((len(payload), idx, tile_id))

    largest_tiles.sort(reverse=True)
    print(f"records walked: {total_records}")
    print(f"unique layer names: {len(layer_names)}")
    print(f"total features: {total_features:,}")
    print(f"total layer raw bytes: {total_layer_bytes:,}")
    print()
    print("layer name frequency (across tiles):")
    for name, count in layer_names.most_common(20):
        print(f"  {count:>5}  {name}")
    print()
    print("feature geometry types:")
    for geom, count in geom_type_counts.most_common():
        print(f"  {count:>7}  {geom}")
    print()
    print("most common attribute keys:")
    for key, count in keys_seen.most_common(15):
        print(f"  {count:>5}  {key}")
    print()
    print("largest 5 tiles by payload size:")
    for size, idx, tile_id in largest_tiles[:5]:
        print(f"  record {idx:>3} tile_id={tile_id:<6} payload={size:,} bytes")


def dump_one(path: Path, target_idx: int) -> None:
    for idx, tile_id, payload in walk_records(path):
        if idx != target_idx:
            continue
        print(f"record {idx} tile_id={tile_id} payload_bytes={len(payload):,}")
        layers = decode_tile(payload)
        print(f"layers: {len(layers)}")
        for li, layer in enumerate(layers):
            n_feat = len(layer["features"])
            geom_counts = Counter(GEOM_TYPES.get(f["type"], "UNKNOWN") for f in layer["features"])
            geom_str = ", ".join(f"{c} {g}" for g, c in geom_counts.most_common())
            print(
                f"  [{li:>2}] name={layer['name']!r:<25} v={layer['version']} extent={layer['extent']} "
                f"keys={len(layer['keys'])} values={layer['n_values']} features={n_feat} "
                f"({geom_str}) raw_bytes={layer['raw_bytes']:,}"
            )
            avg_geom = (
                sum(f["geom_len"] for f in layer["features"]) / max(1, n_feat) if n_feat else 0
            )
            avg_tags = sum(f["n_tags"] for f in layer["features"]) / max(1, n_feat) if n_feat else 0
            print(f"       avg geometry varints / feature: {avg_geom:.1f}, avg tags: {avg_tags:.1f}")
            if layer["keys"]:
                preview = ", ".join(layer["keys"][:8])
                more = f", +{len(layer['keys']) - 8}" if len(layer["keys"]) > 8 else ""
                print(f"       keys: {preview}{more}")
        return
    print(f"no record at index {target_idx}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a prepared MVT data.bin")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--tile", type=int, default=None, help="Dump details of a single record (by index)"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit how many records the summary walks (default: all)",
    )
    args = parser.parse_args()
    if args.tile is not None:
        dump_one(args.path, args.tile)
    else:
        summarize(args.path, args.max_records)


if __name__ == "__main__":
    main()
