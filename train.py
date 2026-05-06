"""
Self-contained training file for PMTiles byte modeling.

Agents may edit this file during autoresearch. Do not move the model into a library.
Run:
  uv run train.py
  uv run train.py --self-test
  uv run train.py --compress --n-bytes 4096
  uv run train.py --decompress records/compressed.bin --n-bytes 4096
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from pathlib import Path

# CUBLAS workspace config must be set before any cuda init for deterministic algorithms.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import DATA_BIN, MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, evaluate_bpb, make_dataloader

CHECKPOINT_PATH = Path(os.environ.get("ACW_CHECKPOINT", "records/checkpoint.pt"))
COMPRESSED_PATH = Path(os.environ.get("ACW_COMPRESSED", "records/compressed.bin"))

# ---------------------------------------------------------------------------
# Hyperparameters. This is the first place agents should try small changes.
# ---------------------------------------------------------------------------

BATCH_SIZE = int(os.environ.get("ACW_BATCH_SIZE", "32"))
DEPTH = int(os.environ.get("ACW_DEPTH", "8"))
DIM = int(os.environ.get("ACW_DIM", "512"))
HEADS = int(os.environ.get("ACW_HEADS", "8"))
MLP_MULT = int(os.environ.get("ACW_MLP_MULT", "4"))
DROPOUT = float(os.environ.get("ACW_DROPOUT", "0.0"))
LR = float(os.environ.get("ACW_LR", "3e-4"))
WEIGHT_DECAY = float(os.environ.get("ACW_WEIGHT_DECAY", "0.1"))
GRAD_CLIP = float(os.environ.get("ACW_GRAD_CLIP", "1.0"))
COMPILE = os.environ.get("ACW_COMPILE", "1") == "1"

# NNCP-style online compression knobs.
NNCP_TRAIN_EVERY = int(os.environ.get("ACW_NNCP_TRAIN_EVERY", "64"))
NNCP_TRAIN_BATCH = int(os.environ.get("ACW_NNCP_TRAIN_BATCH", "8"))
NNCP_TRAIN_SEQ = int(os.environ.get("ACW_NNCP_TRAIN_SEQ", "256"))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, MLP_MULT * dim),
            nn.GELU(),
            nn.Linear(MLP_MULT * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ByteTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(DIM, HEADS) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None, reduction: str = "mean"
    ) -> torch.Tensor:
        _, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.norm(h))
        if y is None:
            return logits
        return F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), reduction=reduction)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def should_continue(train_start: float, step: int, max_steps: int | None) -> bool:
    if max_steps is not None:
        return step < max_steps
    return time.time() - train_start < TIME_BUDGET


# ===========================================================================
# Arithmetic coder primitives (NNCP-style 32-bit, E3 follow-on).
# ===========================================================================

PRECISION = 32
TOP_MASK = 1 << (PRECISION - 1)        # 0x80000000
SECOND_MASK = 1 << (PRECISION - 2)     # 0x40000000
MAX_VAL = (1 << PRECISION) - 1         # 0xFFFFFFFF
FREQ_BITS = 14
TOTAL_FREQ = 1 << FREQ_BITS            # 16384


class BitWriter:
    """Accumulates bits MSB-first into a bytearray."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.cur = 0
        self.nbits = 0  # number of bits currently buffered in self.cur (0..7)

    def write_bit(self, b: int) -> None:
        self.cur = (self.cur << 1) | (b & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.buf.append(self.cur)
            self.cur = 0
            self.nbits = 0

    def write_bits_msb(self, value: int, n: int) -> None:
        for i in range(n - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def finish(self) -> bytes:
        if self.nbits > 0:
            # pad remaining bits with zeros on the right (LSBs of the byte).
            self.buf.append(self.cur << (8 - self.nbits))
            self.cur = 0
            self.nbits = 0
        return bytes(self.buf)


class BitReader:
    """Reads bits MSB-first from a bytes buffer. Returns 0 past end of stream."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0  # 0..7, counting bits already consumed in current byte from MSB

    def read_bit(self) -> int:
        if self.byte_pos >= len(self.data):
            return 0
        b = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return b


class ArithmeticEncoder:
    def __init__(self, writer: BitWriter) -> None:
        self.w = writer
        self.low = 0
        self.high = MAX_VAL
        self.pending = 0
        self.bits_emitted = 0

    def _emit(self, b: int) -> None:
        self.w.write_bit(b)
        self.bits_emitted += 1

    def _emit_with_pending(self, b: int) -> None:
        self._emit(b)
        for _ in range(self.pending):
            self._emit(b ^ 1)
        self.pending = 0

    def encode_symbol(self, cum_low: int, cum_high: int, total: int) -> None:
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_high) // total - 1
        self.low = self.low + (rng * cum_low) // total
        while True:
            if self.high < TOP_MASK:
                self._emit_with_pending(0)
            elif self.low >= TOP_MASK:
                self._emit_with_pending(1)
                self.low -= TOP_MASK
                self.high -= TOP_MASK
            elif self.low >= SECOND_MASK and self.high < (TOP_MASK | SECOND_MASK):
                self.pending += 1
                self.low -= SECOND_MASK
                self.high -= SECOND_MASK
            else:
                break
            self.low = (self.low << 1) & MAX_VAL
            self.high = ((self.high << 1) | 1) & MAX_VAL

    def finish(self) -> None:
        self.pending += 1
        if self.low < SECOND_MASK:
            self._emit(0)
            for _ in range(self.pending):
                self._emit(1)
        else:
            self._emit(1)
            for _ in range(self.pending):
                self._emit(0)
        self.pending = 0


class ArithmeticDecoder:
    def __init__(self, reader: BitReader) -> None:
        self.r = reader
        self.low = 0
        self.high = MAX_VAL
        self.code = 0
        for _ in range(PRECISION):
            self.code = (self.code << 1) | self.r.read_bit()

    def decode_target(self, total: int) -> int:
        rng = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // rng

    def update(self, cum_low: int, cum_high: int, total: int) -> None:
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_high) // total - 1
        self.low = self.low + (rng * cum_low) // total
        while True:
            if self.high < TOP_MASK:
                pass
            elif self.low >= TOP_MASK:
                self.low -= TOP_MASK
                self.high -= TOP_MASK
                self.code -= TOP_MASK
            elif self.low >= SECOND_MASK and self.high < (TOP_MASK | SECOND_MASK):
                self.low -= SECOND_MASK
                self.high -= SECOND_MASK
                self.code -= SECOND_MASK
            else:
                break
            self.low = (self.low << 1) & MAX_VAL
            self.high = ((self.high << 1) | 1) & MAX_VAL
            self.code = ((self.code << 1) | self.r.read_bit()) & MAX_VAL


def quantize_probs(probs: np.ndarray) -> np.ndarray:
    """Deterministically quantize a (256,) float prob vector to integer freqs.

    Returns int64 (256,) summing to TOTAL_FREQ, every entry >= 1.
    """
    assert probs.shape == (VOCAB_SIZE,)
    p = np.asarray(probs, dtype=np.float64)
    # Clip and normalize defensively. Softmax outputs should already be valid,
    # but tiny negative values from numerical noise must not break the floor.
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        p = np.full_like(p, 1.0 / VOCAB_SIZE)
    else:
        p = p / s

    budget = TOTAL_FREQ - VOCAB_SIZE  # 16384 - 256 = 16128
    scaled = p * budget
    floor = np.floor(scaled).astype(np.int64)
    freqs = floor + 1
    remainder = TOTAL_FREQ - int(freqs.sum())
    if remainder > 0:
        frac = scaled - floor.astype(np.float64)
        # Sort by (-fractional_part, symbol_id) stably so ties go to the
        # smaller symbol id. np.lexsort sorts by last key first.
        order = np.lexsort((np.arange(VOCAB_SIZE), -frac))
        freqs[order[:remainder]] += 1
    elif remainder < 0:
        # Shouldn't happen given the +1 floor offset, but guard anyway.
        raise RuntimeError(f"quantize_probs over-budget: remainder={remainder}")
    assert int(freqs.sum()) == TOTAL_FREQ
    assert int(freqs.min()) >= 1
    return freqs


def freqs_to_cum(freqs: np.ndarray) -> np.ndarray:
    """Returns int64 (257,) cumulative table; cum[i+1] - cum[i] == freqs[i]."""
    cum = np.zeros(VOCAB_SIZE + 1, dtype=np.int64)
    np.cumsum(freqs, out=cum[1:])
    return cum


def find_symbol(cum: np.ndarray, target: int) -> int:
    """Binary search: smallest i such that cum[i+1] > target."""
    # cum has length 257, strictly increasing (freqs >= 1).
    # We want the i in [0, 256) with cum[i] <= target < cum[i+1].
    idx = int(np.searchsorted(cum, target, side="right")) - 1
    return idx


# ===========================================================================
# Determinism setup.
# ===========================================================================


def configure_determinism(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Self-test for the arithmetic coder.
# ===========================================================================


def _self_test_uniform() -> None:
    rng = np.random.default_rng(0)
    n = 1024
    src = rng.integers(0, VOCAB_SIZE, size=n, dtype=np.uint8).tobytes()
    uniform_freqs = np.full(VOCAB_SIZE, TOTAL_FREQ // VOCAB_SIZE, dtype=np.int64)
    # 16384 / 256 = 64 exactly, sums to 16384.
    assert int(uniform_freqs.sum()) == TOTAL_FREQ
    cum = freqs_to_cum(uniform_freqs)

    writer = BitWriter()
    enc = ArithmeticEncoder(writer)
    for b in src:
        enc.encode_symbol(int(cum[b]), int(cum[b + 1]), TOTAL_FREQ)
    enc.finish()
    payload = writer.finish()
    print(
        f"self-test uniform: input={n} bytes encoded={len(payload)} bytes "
        f"bpb={enc.bits_emitted / n:.4f}"
    )
    assert len(payload) >= n - 1, "uniform distribution must not compress"

    reader = BitReader(payload)
    dec = ArithmeticDecoder(reader)
    out = bytearray()
    for _ in range(n):
        target = dec.decode_target(TOTAL_FREQ)
        sym = find_symbol(cum, target)
        dec.update(int(cum[sym]), int(cum[sym + 1]), TOTAL_FREQ)
        out.append(sym)
    assert bytes(out) == src, "uniform self-test round-trip mismatch"
    print("self-test uniform: round-trip OK")


def _self_test_skewed() -> None:
    if not DATA_BIN.exists():
        print(f"self-test skewed: skipped, {DATA_BIN} missing")
        return
    data = np.fromfile(DATA_BIN, dtype=np.uint8)[:1024]
    n = len(data)
    # Build distribution from byte frequencies in data.bin (whole file).
    full = np.fromfile(DATA_BIN, dtype=np.uint8)
    counts = np.bincount(full, minlength=VOCAB_SIZE).astype(np.float64)
    probs = counts / counts.sum()
    freqs = quantize_probs(probs.astype(np.float32))
    cum = freqs_to_cum(freqs)
    # Predicted bits per byte under this distribution.
    predicted_bpb = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())

    writer = BitWriter()
    enc = ArithmeticEncoder(writer)
    for b in data:
        enc.encode_symbol(int(cum[int(b)]), int(cum[int(b) + 1]), TOTAL_FREQ)
    enc.finish()
    payload = writer.finish()
    actual_bpb = enc.bits_emitted / n
    print(
        f"self-test skewed: input={n} bytes encoded={len(payload)} bytes "
        f"bpb={actual_bpb:.4f} predicted_bpb={predicted_bpb:.4f}"
    )

    reader = BitReader(payload)
    dec = ArithmeticDecoder(reader)
    out = bytearray()
    for _ in range(n):
        target = dec.decode_target(TOTAL_FREQ)
        sym = find_symbol(cum, target)
        dec.update(int(cum[sym]), int(cum[sym + 1]), TOTAL_FREQ)
        out.append(sym)
    assert bytes(out) == data.tobytes(), "skewed self-test round-trip mismatch"
    print("self-test skewed: round-trip OK")


def run_self_test() -> None:
    _self_test_uniform()
    _self_test_skewed()


# ===========================================================================
# NNCP-style online compress / decompress.
# ===========================================================================


def _nncp_setup_model() -> tuple[ByteTransformer, torch.optim.Optimizer]:
    """Build a fresh model + optimizer with deterministic init."""
    configure_determinism(1337)
    model = ByteTransformer().cuda()
    # Force eval mode side effects (dropout off) but keep params trainable.
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    return model, opt


@torch.no_grad()
def _nncp_predict(model: ByteTransformer, ctx: list[int]) -> np.ndarray:
    """Return (256,) float64 probabilities for the next byte given context.

    Uses a uniform prior when ctx is empty.
    """
    if len(ctx) == 0:
        return np.full(VOCAB_SIZE, 1.0 / VOCAB_SIZE, dtype=np.float64)
    model.eval()
    # Cap to MAX_SEQ_LEN tokens of context.
    ctx_t = torch.tensor(ctx[-MAX_SEQ_LEN:], dtype=torch.long, device="cuda").unsqueeze(0)
    logits = model(ctx_t)  # (1, t, V)
    last = logits[0, -1].float()
    probs = F.softmax(last, dim=-1).double().cpu().numpy()
    return probs


def _nncp_train_step(
    model: ByteTransformer,
    opt: torch.optim.Optimizer,
    seen: np.ndarray,
    n_seen: int,
    rng: np.random.Generator,
    seq_len: int,
    batch_size: int,
) -> float:
    """One AdamW step on random crops sampled from the bytes seen so far.

    `seen` is a fixed-capacity uint8 array; only the first `n_seen` entries are valid.
    """
    if n_seen <= seq_len + 1:
        return float("nan")
    model.train()
    starts = rng.integers(0, n_seen - seq_len - 1, size=batch_size)
    x = np.empty((batch_size, seq_len), dtype=np.int64)
    y = np.empty((batch_size, seq_len), dtype=np.int64)
    for i, s in enumerate(starts):
        chunk = seen[s : s + seq_len + 1].astype(np.int64)
        x[i] = chunk[:-1]
        y[i] = chunk[1:]
    xt = torch.from_numpy(x).cuda()
    yt = torch.from_numpy(y).cuda()
    opt.zero_grad(set_to_none=True)
    loss = model(xt, yt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    return float(loss.item())


def _open_data(n_bytes: int | None) -> np.ndarray:
    data = np.fromfile(DATA_BIN, dtype=np.uint8)
    if n_bytes is not None:
        data = data[:n_bytes]
    return data


def nncp_compress(n_bytes: int | None, out_path: Path) -> None:
    """Encode the first `n_bytes` of data.bin to `out_path`."""
    data = _open_data(n_bytes)
    n = len(data)
    print(f"nncp compress: bytes={n} model dim={DIM} depth={DEPTH} heads={HEADS}")
    model, opt = _nncp_setup_model()
    print(f"num_params_M: {count_params(model) / 1e6:.6f}")

    seen = np.empty(n, dtype=np.uint8)
    n_seen = 0

    writer = BitWriter()
    enc = ArithmeticEncoder(writer)
    train_rng = np.random.default_rng(1337)

    t0 = time.time()
    last_print = 0
    for i in range(n):
        b = int(data[i])
        # Build context as the suffix of seen[:n_seen] (already a bounded slice).
        lo = max(0, n_seen - MAX_SEQ_LEN)
        ctx = seen[lo:n_seen].tolist()
        probs = _nncp_predict(model, ctx)
        freqs = quantize_probs(probs.astype(np.float32))
        cum = freqs_to_cum(freqs)
        enc.encode_symbol(int(cum[b]), int(cum[b + 1]), TOTAL_FREQ)
        seen[n_seen] = b
        n_seen += 1

        if NNCP_TRAIN_EVERY > 0 and n_seen % NNCP_TRAIN_EVERY == 0:
            _nncp_train_step(
                model, opt, seen, n_seen, train_rng, NNCP_TRAIN_SEQ, NNCP_TRAIN_BATCH
            )

        if n_seen - last_print >= 1024 or n_seen == n:
            elapsed = time.time() - t0
            bits = enc.bits_emitted
            bpb = bits / n_seen
            print(
                f"step {n_seen} bytes {n_seen} bits {bits} bpb {bpb:.4f} "
                f"elapsed {elapsed:.1f}s"
            )
            last_print = n_seen

    enc.finish()
    payload = writer.finish()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Header: magic(4) + n_bytes(u64le)
    header = b"ACWC" + struct.pack("<Q", n)
    out_path.write_bytes(header + payload)
    elapsed = time.time() - t0
    print(
        f"compress done: input={n} compressed_payload={len(payload)} "
        f"file={len(header) + len(payload)} bpb={enc.bits_emitted / n:.4f} "
        f"elapsed={elapsed:.1f}s out={out_path}"
    )


def nncp_decompress(in_path: Path, n_bytes: int | None) -> None:
    """Decode `in_path` and verify it equals data.bin[:n_bytes]."""
    raw = in_path.read_bytes()
    if raw[:4] != b"ACWC":
        raise ValueError(f"bad magic in {in_path}: {raw[:4]!r}")
    declared_n = struct.unpack_from("<Q", raw, 4)[0]
    payload = raw[12:]
    if n_bytes is not None and n_bytes != declared_n:
        raise ValueError(
            f"--n-bytes={n_bytes} but artifact declares {declared_n}; "
            "decode the same range you encoded"
        )
    n = declared_n
    print(f"nncp decompress: bytes={n} payload_bytes={len(payload)}")

    truth = _open_data(n)
    assert len(truth) == n, f"truth length {len(truth)} != declared {n}"

    model, opt = _nncp_setup_model()
    print(f"num_params_M: {count_params(model) / 1e6:.6f}")

    reader = BitReader(payload)
    dec = ArithmeticDecoder(reader)

    seen = np.empty(n, dtype=np.uint8)
    n_seen = 0
    train_rng = np.random.default_rng(1337)

    t0 = time.time()
    last_print = 0
    out = bytearray()
    for i in range(n):
        lo = max(0, n_seen - MAX_SEQ_LEN)
        ctx = seen[lo:n_seen].tolist()
        probs = _nncp_predict(model, ctx)
        freqs = quantize_probs(probs.astype(np.float32))
        cum = freqs_to_cum(freqs)
        target = dec.decode_target(TOTAL_FREQ)
        sym = find_symbol(cum, target)
        dec.update(int(cum[sym]), int(cum[sym + 1]), TOTAL_FREQ)
        # Sanity check against truth (which we have here for validation).
        if sym != int(truth[i]):
            raise AssertionError(
                f"mismatch at byte {i}: decoded={sym} truth={int(truth[i])}"
            )
        out.append(sym)
        seen[n_seen] = sym
        n_seen += 1

        if NNCP_TRAIN_EVERY > 0 and n_seen % NNCP_TRAIN_EVERY == 0:
            _nncp_train_step(
                model, opt, seen, n_seen, train_rng, NNCP_TRAIN_SEQ, NNCP_TRAIN_BATCH
            )

        if n_seen - last_print >= 1024 or n_seen == n:
            elapsed = time.time() - t0
            print(f"decode {n_seen}/{n} elapsed {elapsed:.1f}s")
            last_print = n_seen

    decoded = bytes(out)
    assert decoded == truth.tobytes(), "decoded bytes do not match data.bin"
    print(f"decompress done: bytes={n} round-trip OK elapsed={time.time() - t0:.1f}s")


# ===========================================================================
# Original training loop (unchanged behavior aside from ordering imports).
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PMTiles byte model")
    parser.add_argument("--steps", type=int, default=None, help="Limit optimizer steps")
    parser.add_argument(
        "--self-test", action="store_true", help="Run arithmetic-coder self-tests and exit"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Run NNCP-style online compression and write records/compressed.bin",
    )
    parser.add_argument(
        "--decompress",
        type=Path,
        default=None,
        help="Decode the given artifact and verify it matches data.bin",
    )
    parser.add_argument(
        "--n-bytes",
        type=int,
        default=None,
        help="Limit compress/decompress to the first N bytes of data.bin",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=COMPRESSED_PATH,
        help=f"Output artifact path for --compress (default: {COMPRESSED_PATH})",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.compress:
        assert torch.cuda.is_available(), "compress requires CUDA"
        nncp_compress(args.n_bytes, args.out)
        return

    if args.decompress is not None:
        assert torch.cuda.is_available(), "decompress requires CUDA"
        nncp_decompress(args.decompress, args.n_bytes)
        return

    # ----------------------- standard training mode -----------------------
    assert torch.cuda.is_available(), "This research harness expects a local NVIDIA GPU."
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(
        "config: "
        f"batch_size={BATCH_SIZE} depth={DEPTH} dim={DIM} heads={HEADS} "
        f"seq_len={MAX_SEQ_LEN} lr={LR} weight_decay={WEIGHT_DECAY} compile={COMPILE} "
        f"steps={args.steps if args.steps is not None else 'time_budget'}"
    )
    train_loader = make_dataloader(BATCH_SIZE, MAX_SEQ_LEN)
    model = ByteTransformer().cuda()
    raw_model = model
    print(f"num_params_M: {count_params(raw_model) / 1e6:.6f}")
    if COMPILE:
        print("compiling model")
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    train_start = time.time()
    total_tokens = 0
    step = 0
    while should_continue(train_start, step, args.steps):
        x, y = train_loader.next_batch()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        total_tokens += x.numel()
        step += 1
        if step == 1 or step % 50 == 0:
            elapsed = time.time() - train_start
            print(
                f"step {step:05d} loss {loss.item():.4f} "
                f"elapsed {elapsed:.1f}s tokens_M {total_tokens / 1e6:.1f}"
            )

    train_seconds = time.time() - train_start
    bpb = evaluate_bpb(model, BATCH_SIZE)
    torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    summary = {
        "bpb": bpb,
        "training_seconds": train_seconds,
        "peak_vram_mb": peak_vram_mb,
        "total_tokens_M": total_tokens / 1e6,
        "num_steps": step,
        "num_params_M": count_params(raw_model) / 1e6,
        "depth": DEPTH,
        "dim": DIM,
        "heads": HEADS,
    }
    print("---")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": raw_model.state_dict(),
            "config": {
                "depth": DEPTH,
                "dim": DIM,
                "heads": HEADS,
                "mlp_mult": MLP_MULT,
                "dropout": DROPOUT,
                "vocab_size": VOCAB_SIZE,
                "max_seq_len": MAX_SEQ_LEN,
            },
            "summary": summary,
        },
        CHECKPOINT_PATH,
    )
    print(f"checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
