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
# Order-preserving structural dumpers.
# ---------------------------------------------------------------------------

# MVT field name hints, indexed as (path_str, field_no) -> human label.
MVT_NAMES = {
    ("Tile", 3): "layers",
    ("Layer", 1): "name",
    ("Layer", 2): "feature",
    ("Layer", 3): "key",
    ("Layer", 4): "value",
    ("Layer", 5): "extent",
    ("Layer", 15): "version",
    ("Feature", 1): "id",
    ("Feature", 2): "tags",
    ("Feature", 3): "type",
    ("Feature", 4): "geometry",
    ("Value", 1): "string_value",
    ("Value", 2): "float_value",
    ("Value", 3): "double_value",
    ("Value", 4): "int_value",
    ("Value", 5): "uint_value",
    ("Value", 6): "sint_value",
    ("Value", 7): "bool_value",
}

# Hint about the message type a length-delim child should be parsed as.
# Path is the parent message type and field number.
CHILD_TYPE = {
    ("Tile", 3): "Layer",
    ("Layer", 2): "Feature",
    ("Layer", 4): "Value",
}

# Length-delim leaf interpretation.
LEAF_KIND = {
    ("Layer", 1): "string",
    ("Layer", 3): "string",  # key
    ("Feature", 2): "packed_varint",
    ("Feature", 4): "packed_varint",
}

WIRE_NAMES = {0: "VARINT", 1: "I64", 2: "LEN", 5: "I32"}


def _read_packed_varints(buf: bytes) -> list[int]:
    out: list[int] = []
    pos = 0
    while pos < len(buf):
        v, pos = read_varint(buf, pos)
        out.append(v)
    return out


def parse_structured(buf: bytes, type_name: str, base_offset: int = 0) -> list[dict]:
    """Walk a protobuf message, returning fields in wire order with byte offsets.

    Each entry: {
      "offset":         byte offset where the field starts (tag byte),
      "tag_offset":     same as `offset`,
      "value_offset":   byte offset where the field value begins,
      "end_offset":     byte offset of the first byte AFTER this field,
      "field_no":       int,
      "wire":           "VARINT" | "I64" | "LEN" | "I32",
      "name":           e.g. "layers" or "<field 17>",
      "value":          for VARINT/I32/I64 the raw int/bytes;
                        for LEN the bytes (raw) plus optional decoded form.
      ...
    }

    For LEN fields, when we know the child message type we recurse and add
    "fields" (list of nested entries) plus "child_type". For string-typed
    leaves we add "string". For packed-varint leaves we add "packed".
    """
    out: list[dict] = []
    pos = 0
    while pos < len(buf):
        tag_off = pos
        tag, pos = read_varint(buf, pos)
        field_no = tag >> 3
        wire = tag & 7
        entry: dict = {
            "offset": base_offset + tag_off,
            "tag_offset": base_offset + tag_off,
            "field_no": field_no,
            "wire": WIRE_NAMES.get(wire, f"wire{wire}"),
            "name": MVT_NAMES.get((type_name, field_no), f"<field {field_no}>"),
        }
        value_off = pos
        if wire == 0:
            v, pos = read_varint(buf, pos)
            entry["value"] = v
            entry["value_offset"] = base_offset + value_off
            entry["end_offset"] = base_offset + pos
        elif wire == 5:
            entry["value"] = buf[pos : pos + 4]
            entry["value_offset"] = base_offset + value_off
            pos += 4
            entry["end_offset"] = base_offset + pos
        elif wire == 1:
            entry["value"] = buf[pos : pos + 8]
            entry["value_offset"] = base_offset + value_off
            pos += 8
            entry["end_offset"] = base_offset + pos
        elif wire == 2:
            ln, pos = read_varint(buf, pos)
            payload_off = pos
            payload = buf[pos : pos + ln]
            pos += ln
            entry["length"] = ln
            entry["value_offset"] = base_offset + payload_off
            entry["end_offset"] = base_offset + pos
            entry["value"] = payload
            child = CHILD_TYPE.get((type_name, field_no))
            kind = LEAF_KIND.get((type_name, field_no))
            if child is not None:
                entry["child_type"] = child
                entry["fields"] = parse_structured(payload, child, base_offset + payload_off)
            elif kind == "string":
                entry["string"] = payload.decode("utf-8", errors="replace")
            elif kind == "packed_varint":
                entry["packed"] = _read_packed_varints(payload)
        else:
            raise ValueError(f"unsupported wire type {wire} at pos {tag_off}")
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# JSON / hex / textproto renderers.
# ---------------------------------------------------------------------------


def _entry_to_json(entry: dict, type_name: str) -> dict:
    """Convert one parsed entry to a JSON-serialisable dict, preserving order
    via the `fields` list for nested messages. Bytes are rendered as hex.
    """
    out = {
        "field_no": entry["field_no"],
        "name": entry["name"],
        "wire": entry["wire"],
        "offset": entry["offset"],
        "value_offset": entry["value_offset"],
        "end_offset": entry["end_offset"],
    }
    if entry["wire"] in ("VARINT",):
        out["value"] = entry["value"]
    elif entry["wire"] in ("I32", "I64"):
        out["bytes_hex"] = entry["value"].hex()
    elif entry["wire"] == "LEN":
        out["length"] = entry["length"]
        if "fields" in entry:
            out["child_type"] = entry["child_type"]
            out["fields"] = [
                _entry_to_json(c, entry["child_type"]) for c in entry["fields"]
            ]
        elif "string" in entry:
            out["string"] = entry["string"]
        elif "packed" in entry:
            out["packed"] = entry["packed"]
        else:
            out["bytes_hex"] = bytes(entry["value"]).hex()
    return out


def dump_json(buf: bytes, type_name: str = "Tile") -> dict:
    fields = parse_structured(buf, type_name)
    return {
        "type": type_name,
        "byte_length": len(buf),
        "fields": [_entry_to_json(e, type_name) for e in fields],
    }


def _format_value(entry: dict, max_len: int = 80) -> str:
    if entry["wire"] == "VARINT":
        return f"varint({entry['value']})"
    if entry["wire"] == "LEN":
        if "string" in entry:
            s = entry["string"]
            if len(s) > max_len:
                s = s[: max_len - 3] + "..."
            return f"string({json_dumps_str(s)})"
        if "packed" in entry:
            arr = entry["packed"]
            if len(arr) > 12:
                preview = ", ".join(str(x) for x in arr[:6]) + ", ..., " + ", ".join(
                    str(x) for x in arr[-3:]
                )
                return f"packed({len(arr)} ints: [{preview}])"
            return f"packed({arr})"
        if "fields" in entry:
            return f"{entry['child_type']}({len(entry['fields'])} fields)"
        return f"bytes({entry['length']})"
    if entry["wire"] in ("I32", "I64"):
        return f"{entry['wire']}(0x{entry['value'].hex()})"
    return "?"


def json_dumps_str(s: str) -> str:
    import json

    return json.dumps(s, ensure_ascii=False)


def render_textproto(entries: list[dict], indent: int = 0) -> str:
    """Render entries as a textproto-ish string with byte offsets prefixed.

    Each line:  <offset>: <indent>field_no(wire) name = value
    """
    lines: list[str] = []
    pad = "  " * indent
    for e in entries:
        head = f"{e['offset']:>7}: {pad}#{e['field_no']:<2} {e['wire']:<6} {e['name']:<14}"
        if e["wire"] == "LEN" and "fields" in e:
            lines.append(f"{head} = {e['child_type']} {{  // len={e['length']}")
            lines.append(render_textproto(e["fields"], indent + 1))
            lines.append(f"       : {pad}}}  // end {e['name']}")
        else:
            lines.append(f"{head} = {_format_value(e)}")
    return "\n".join(lines)


def render_hex_dump(buf: bytes, entries: list[dict], indent: int = 0) -> str:
    """Annotated hex dump: each protobuf field shown with its raw bytes.

    Output per field (length-delim shown with its tag/length bytes; the
    payload bytes follow on subsequent lines, optionally recursed for
    nested messages):
        <offset> <hex bytes>   #<field_no> <wire> <name> = <value summary>
    """
    out: list[str] = []
    pad = "  " * indent

    def hex_chunk(b: bytes, width: int = 16) -> list[str]:
        s = b.hex()
        return [s[i : i + width * 2] for i in range(0, len(s), width * 2)]

    for e in entries:
        # Tag bytes occupy [tag_offset, value_offset).
        tag_bytes = buf[e["tag_offset"] : e["value_offset"]]
        if e["wire"] == "LEN":
            payload = buf[e["value_offset"] : e["end_offset"]]
            label = (
                f"#{e['field_no']} {e['wire']} {e['name']} "
                f"len={e['length']} -> {_format_value(e)}"
            )
            out.append(
                f"{e['tag_offset']:>7}  {tag_bytes.hex():<24}  {pad}{label}"
            )
            if "fields" in e:
                out.append(render_hex_dump(buf, e["fields"], indent + 1))
            else:
                # Show a few hex chunks of the payload so the user can see the bytes.
                preview = hex_chunk(payload[:48])
                for i, chunk in enumerate(preview):
                    off = e["value_offset"] + i * 16
                    out.append(f"{off:>7}  {chunk:<32}  {pad}  ")
                if len(payload) > 48:
                    out.append(
                        f"       ... {len(payload) - 48} more payload bytes ..."
                    )
        else:
            value_bytes = buf[e["value_offset"] : e["end_offset"]]
            label = (
                f"#{e['field_no']} {e['wire']} {e['name']} = {_format_value(e)}"
            )
            out.append(
                f"{e['tag_offset']:>7}  "
                f"{(tag_bytes + value_bytes).hex():<24}  {pad}{label}"
            )
    return "\n".join(out)


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
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Dump the targeted tile as JSON, preserving on-wire field order "
            "and byte offsets. Use with --tile/--tile-id/--zxy."
        ),
    )
    parser.add_argument(
        "--hex",
        action="store_true",
        help=(
            "Annotated hex dump of the targeted tile's protobuf bytes, with "
            "each field labelled inline. Use with --tile/--tile-id/--zxy."
        ),
    )
    parser.add_argument(
        "--textproto",
        action="store_true",
        help=(
            "Compact textproto-ish dump (one line per field with byte "
            "offsets and value summaries). Use with --tile/--tile-id/--zxy."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the dump to this file instead of stdout",
    )
    args = parser.parse_args()
    target_tile_id: int | None = None
    if args.zxy is not None:
        z, x, y = (int(part) for part in args.zxy.split("/"))
        target_tile_id = _zxy_to_tile_id(z, x, y)
        print(f"z/x/y {z}/{x}/{y} -> tile_id {target_tile_id}")
    elif args.tile_id is not None:
        target_tile_id = args.tile_id

    structured_modes = (args.json, args.hex, args.textproto)
    want_structured = any(structured_modes)

    def write_out(text: str) -> None:
        if args.out is not None:
            args.out.write_text(text + "\n", encoding="utf-8")
            print(f"wrote {len(text)} chars to {args.out}")
        else:
            print(text)

    def emit_payload(idx: int, tile_id: int, payload: bytes) -> None:
        z, x, y = _tile_id_to_zxy(tile_id)
        header = f"# record {idx} tile_id={tile_id} z={z} x={x} y={y} bytes={len(payload):,}"
        if args.json:
            import json as _json

            doc = {
                "record": idx,
                "tile_id": tile_id,
                "zxy": [z, x, y],
                "payload": dump_json(payload, "Tile"),
            }
            write_out(_json.dumps(doc, indent=2, ensure_ascii=False))
            return
        entries = parse_structured(payload, "Tile")
        if args.hex:
            text = header + "\n" + render_hex_dump(payload, entries)
            write_out(text)
            return
        if args.textproto:
            text = header + "\n" + render_textproto(entries)
            write_out(text)
            return
        # Fallback: shouldn't reach here when want_structured is True.
        write_out(header)

    if want_structured and args.tile is None and target_tile_id is None:
        print("--json/--hex/--textproto requires --tile, --tile-id, or --zxy")
        return

    if args.tile is not None:
        if want_structured:
            for idx, tid, payload in walk_records(args.path):
                if idx == args.tile:
                    emit_payload(idx, tid, payload)
                    return
            print(f"no record at index {args.tile}")
        else:
            dump_one(args.path, args.tile, args.max_features, args.max_items)
    elif target_tile_id is not None:
        for idx, tid, payload in walk_records(args.path):
            if tid == target_tile_id:
                if want_structured:
                    emit_payload(idx, tid, payload)
                else:
                    dump_one(args.path, idx, args.max_features, args.max_items)
                return
        print(f"no record with tile_id={target_tile_id}")
    else:
        summarize(args.path, args.max_records)


if __name__ == "__main__":
    main()
