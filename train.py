"""
Self-contained training file for PMTiles byte modeling.

Agents may edit this file during autoresearch. Do not move the model into a library.
Run:
  uv run train.py
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, evaluate_bpb, make_dataloader

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PMTiles byte model")
    parser.add_argument("--steps", type=int, default=None, help="Limit optimizer steps")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
