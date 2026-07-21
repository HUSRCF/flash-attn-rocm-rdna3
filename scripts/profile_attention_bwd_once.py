import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from flash_attn import flash_attn_func


def sdpa(q, k, v, causal):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=causal,
    )
    return out.transpose(1, 2)


def flash(q, k, v, causal):
    return flash_attn_func(q, k, v, causal=causal, deterministic=False)


def zero_grads(*tensors):
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def run_once(fn, q, k, v, dout, causal):
    zero_grads(q, k, v)
    out = fn(q, k, v, causal)
    out.backward(dout)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["flash", "sdpa"], required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda"
    q = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn_like(q)

    fn = flash if args.backend == "flash" else sdpa
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        run_once(fn, q, k, v, dout, args.causal)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        run_once(fn, q, k, v, dout, args.causal)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.iters
    print(
        f"backend={args.backend} dtype={args.dtype} causal={args.causal} "
        f"B={args.batch} H={args.nheads} S={args.seqlen} D={args.headdim} "
        f"avg_fwd_bwd_ms={elapsed_ms:.3f} device={torch.cuda.get_device_name(0)} "
        f"pid={os.getpid()}"
    )


if __name__ == "__main__":
    main()
