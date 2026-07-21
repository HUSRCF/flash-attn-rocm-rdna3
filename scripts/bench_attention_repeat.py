#!/usr/bin/env python3
import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_attn import flash_attn_func


def clone_with_grad(q, k, v, dout):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    return q, k, v, dout.detach().clone()


def bench_backward(make_tensors, fn, warmup, iters):
    for _ in range(warmup):
        q, k, v, dout = make_tensors()
        out = fn(q, k, v)
        out.backward(dout)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        q, k, v, dout = make_tensors()
        out = fn(q, k, v)
        out.backward(dout)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def summarize(values):
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--causal", choices=["false", "true", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    causals = [False, True] if args.causal == "both" else [args.causal == "true"]
    device = "cuda"

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    torch.manual_seed(args.seed)

    for causal in causals:
        q = torch.randn(args.batch, args.seqlen, args.heads, args.dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        dout = torch.randn_like(q)

        def make_tensors():
            return clone_with_grad(q, k, v, dout)

        def flash_fn(q_, k_, v_):
            return flash_attn_func(q_, k_, v_, causal=causal, deterministic=False)

        def sdpa_fn(q_, k_, v_):
            return F.scaled_dot_product_attention(
                q_.transpose(1, 2),
                k_.transpose(1, 2),
                v_.transpose(1, 2),
                is_causal=causal,
            ).transpose(1, 2)

        for name, fn in (("flash", flash_fn), ("sdpa", sdpa_fn)):
            vals = [bench_backward(make_tensors, fn, args.warmup, args.iters) for _ in range(args.repeats)]
            row = {
                "backend": name,
                "dtype": args.dtype,
                "causal": int(causal),
                "batch": args.batch,
                "heads": args.heads,
                "seqlen": args.seqlen,
                "dim": args.dim,
                "repeats": args.repeats,
                "warmup": args.warmup,
                "iters": args.iters,
                **summarize(vals),
            }
            rows.append(row)
            print(
                f"{name:5s} causal={int(causal)} mean={row['mean_ms']:.6f} "
                f"std={row['std_ms']:.6f} min={row['min_ms']:.6f} max={row['max_ms']:.6f}"
            )

    fieldnames = list(rows[0].keys())
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
