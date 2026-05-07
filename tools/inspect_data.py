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
    feat = {"id": None, "type": 0, "tags": [], "geom_raw": []}
    pos = 0
    while pos < len(buf):
        no, wire, v, pos = read_field(buf, pos)
        if no == 1 and wire == 0:
            feat["id"] = v
        elif no == 2 and wire == 2:
            feat["tags"] = list(iter_varints(v))
        elif no == 3 and wire == 0:
            feat["type"] = v
        elif no == 4 and wire == 2:
            feat["geom_raw"] = list(iter_varints(v))
    return feat


def decode_value(buf: bytes):
    """MVT Value oneof: string/float/double/int/uint/sint/bool."""
    pos = 0
    while pos < len(buf):
        no, wire, v, pos = read_field(buf, pos)
        if no == 1 and wire == 2:
            return v.decode("utf-8", errors="replace")
        if no == 2 and wire == 5:  # float (4 bytes)
            return struct.unpack("<f", v)[0]
        if no == 3 and wire == 1:  # double (8 bytes)
            return struct.unpack("<d", v)[0]
        if no == 4 and wire == 0:  # int64
            # zigzag NOT applied for int64 in MVT; treat raw varint.
            return v
        if no == 5 and wire == 0:  # uint64
            return v
        if no == 6 and wire == 0:  # sint64 (zigzag)
            return (v >> 1) ^ -(v & 1)
        if no == 7 and wire == 0:  # bool
            return bool(v)
    return None


def decode_geometry(geom_raw: list[int]) -> list:
    """Decode MVT geometry varints into a list of (cmd, [(x, y), ...])."""
    cmd_names = {1: "MoveTo", 2: "LineTo", 7: "ClosePath"}
    out: list[tuple[str, list[tuple[int, int]]]] = []
    x, y = 0, 0
    i = 0
    while i < len(geom_raw):
        cmd_int = geom_raw[i]
        i += 1
        cid = cmd_int & 0x7
        count = cmd_int >> 3
        coords: list[tuple[int, int]] = []
        if cid in (1, 2):  # MoveTo / LineTo, count parameter pairs
            for _ in range(count):
                if i + 1 >= len(geom_raw):
                    break
                dx_raw = geom_raw[i]
                dy_raw = geom_raw[i + 1]
                i += 2
                dx = (dx_raw >> 1) ^ -(dx_raw & 1)
                dy = (dy_raw >> 1) ^ -(dy_raw & 1)
                x += dx
                y += dy
                coords.append((x, y))
        out.append((cmd_names.get(cid, f"cmd{cid}"), coords))
    return out


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
        "values": [],
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
            layer["values"].append(decode_value(v))
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
    _ = total_layer_bytes  # kept for parity with earlier debug; unused below

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


def _abbrev(seq, head: int = 3, tail: int = 2):
    """numpy-style truncation: head ... tail. Returns the abbreviated list and total."""
    if len(seq) <= head + tail + 1:
        return list(seq), len(seq)
    return list(seq[:head]) + ["..."] + list(seq[-tail:]), len(seq)


def _fmt_value(v) -> str:
    if isinstance(v, str):
        if len(v) > 40:
            return repr(v[:37] + "...")
        return repr(v)
    return repr(v)


def _fmt_geom(geom: list, max_pts: int = 4) -> str:
    """Compact geometry summary: 'MoveTo(x,y); LineTo*N: (x,y)..(x,y); ClosePath'."""
    parts = []
    for cmd, coords in geom:
        if not coords:
            parts.append(cmd)
            continue
        if len(coords) <= max_pts:
            inside = ", ".join(f"({x},{y})" for x, y in coords)
            parts.append(f"{cmd}: {inside}")
        else:
            inside = (
                ", ".join(f"({x},{y})" for x, y in coords[:2])
                + ", ..., "
                + ", ".join(f"({x},{y})" for x, y in coords[-2:])
            )
            parts.append(f"{cmd}*{len(coords)}: {inside}")
    return "; ".join(parts)


def _resolve_attrs(tags: list[int], keys: list[str], values: list) -> dict:
    out = {}
    for i in range(0, len(tags) - 1, 2):
        ki, vi = tags[i], tags[i + 1]
        if 0 <= ki < len(keys) and 0 <= vi < len(values):
            out[keys[ki]] = values[vi]
    return out


def _hilbert_xy_to_d(n: int, x: int, y: int) -> int:
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


def _zxy_to_tile_id(z: int, x: int, y: int) -> int:
    """PMTiles encoding: zoom-major then Hilbert-curve order within a zoom."""
    return (4**z - 1) // 3 + _hilbert_xy_to_d(1 << z, x, y)


def _tile_id_to_zxy(tid: int) -> tuple[int, int, int]:
    z = 0
    while (4 ** (z + 1) - 1) // 3 <= tid:
        z += 1
    d = tid - (4**z - 1) // 3
    n = 1 << z
    x = y = 0
    s = 1
    t = d
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return z, x, y


def dump_one(path: Path, target_idx: int, max_features: int = 5, max_items: int = 6) -> None:
    for idx, tile_id, payload in walk_records(path):
        if idx != target_idx:
            continue
        z, x, y = _tile_id_to_zxy(tile_id)
        print(
            f"record {idx} tile_id={tile_id} z={z} x={x} y={y} payload_bytes={len(payload):,}"
        )
        layers = decode_tile(payload)
        print(f"layers: {len(layers)}")
        for li, layer in enumerate(layers):
            n_feat = len(layer["features"])
            geom_counts = Counter(GEOM_TYPES.get(f["type"], "UNKNOWN") for f in layer["features"])
            geom_str = ", ".join(f"{c} {g}" for g, c in geom_counts.most_common())
            print()
            print(
                f"  [{li}] name={layer['name']!r}  v={layer['version']}  extent={layer['extent']}  "
                f"raw_bytes={layer['raw_bytes']:,}  features={n_feat} ({geom_str})"
            )
            keys_abbr, k_total = _abbrev(layer["keys"], head=4, tail=2)
            vals_abbr, v_total = _abbrev(layer["values"], head=4, tail=2)
            keys_str = ", ".join(repr(k) if k != "..." else "..." for k in keys_abbr)
            vals_str = ", ".join(_fmt_value(x) if x != "..." else "..." for x in vals_abbr)
            print(f"      keys ({k_total}): [{keys_str}]")
            print(f"      values ({v_total}): [{vals_str}]")

            features_to_show = layer["features"][:max_features]
            for fi, feat in enumerate(features_to_show):
                geom_decoded = decode_geometry(feat["geom_raw"])
                attrs = _resolve_attrs(feat["tags"], layer["keys"], layer["values"])
                # Abbreviate attrs dict if huge.
                if len(attrs) > max_items:
                    keep = list(attrs.items())[:max_items]
                    attrs_str = (
                        "{"
                        + ", ".join(f"{k!r}: {_fmt_value(v)}" for k, v in keep)
                        + f", ... +{len(attrs) - max_items} more"
                        + "}"
                    )
                else:
                    attrs_str = (
                        "{"
                        + ", ".join(f"{k!r}: {_fmt_value(v)}" for k, v in attrs.items())
                        + "}"
                    )
                geom_str_f = _fmt_geom(geom_decoded)
                if len(geom_str_f) > 200:
                    geom_str_f = geom_str_f[:197] + "..."
                print(
                    f"      [feat {fi}] id={feat['id']} type={GEOM_TYPES.get(feat['type'], 'UNKNOWN')} "
                    f"attrs={attrs_str}"
                )
                print(f"          geom={geom_str_f}")
            if n_feat > max_features:
                print(f"      ... +{n_feat - max_features} more features")
        return
    print(f"no record at index {target_idx}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a prepared MVT data.bin")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help=(
            "Dump details of a single record by ZERO-BASED INDEX in data.bin "
            "(0 = first record on disk, NOT the PMTiles tile_id). The summary "
            "view shows tile_ids alongside indices."
        ),
    )
    parser.add_argument(
        "--tile-id",
        type=int,
        default=None,
        help="Alternative: dump by PMTiles tile_id (e.g. 0 for the world tile)",
    )
    parser.add_argument(
        "--zxy",
        type=str,
        default=None,
        help=(
            "Alternative: dump by slippy-map z/x/y (the format pmtiles.io shows, "
            "e.g. 11/1068/746). Converts to tile_id internally."
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5,
        help="Cap features shown per layer when dumping a tile (default 5)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=6,
        help="Cap items shown per attrs dict when dumping a tile (default 6)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit how many records the summary walks (default: all)",
    )
    args = parser.parse_args()
    target_tile_id: int | None = None
    if args.zxy is not None:
        z, x, y = (int(part) for part in args.zxy.split("/"))
        target_tile_id = _zxy_to_tile_id(z, x, y)
        print(f"z/x/y {z}/{x}/{y} -> tile_id {target_tile_id}")
    elif args.tile_id is not None:
        target_tile_id = args.tile_id

    if args.tile is not None:
        dump_one(args.path, args.tile, args.max_features, args.max_items)
    elif target_tile_id is not None:
        for idx, tid, _ in walk_records(args.path):
            if tid == target_tile_id:
                dump_one(args.path, idx, args.max_features, args.max_items)
                return
        print(f"no record with tile_id={target_tile_id}")
    else:
        summarize(args.path, args.max_records)


if __name__ == "__main__":
    main()
