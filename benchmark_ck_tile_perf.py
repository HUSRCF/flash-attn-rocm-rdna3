import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

import flash_attn
import flash_attn_2_cuda
from flash_attn import flash_attn_func


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@dataclass
class PerfRow:
    tag: str
    backend: str
    batch: int
    seqlen: int
    nheads: int
    headdim: int
    dtype: str
    causal: bool
    deterministic: str
    warmup: int
    iters: int
    trials: int
    fwd_ms_mean: float
    fwd_ms_std: float
    bwd_ms_mean: float
    bwd_ms_std: float
    total_ms_mean: float
    total_ms_std: float
    ok: bool
    error: str


def make_inputs(batch: int, seqlen: int, nheads: int, headdim: int, dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    dout = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda")
    return q, k, v, dout


def zero_grads(*tensors):
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def sdpa_native(q, k, v, causal: bool):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


def sdpa_math(q, k, v, causal: bool):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    except Exception:
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


def run_backend(q, k, v, causal: bool, backend: str, deterministic: bool):
    if backend == "flash":
        return flash_attn_func(q, k, v, causal=causal, deterministic=deterministic)
    if backend == "sdpa":
        return sdpa_native(q, k, v, causal=causal)
    if backend == "sdpa_math":
        return sdpa_math(q, k, v, causal=causal)
    raise ValueError(f"Unsupported backend: {backend}")


def run_timed_iters(q, k, v, dout, causal: bool, deterministic: bool, backend: str, num_iters: int):
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)

    fwd_ms = 0.0
    bwd_ms = 0.0
    total_ms = 0.0

    for _ in range(num_iters):
        zero_grads(q, k, v)
        torch.cuda.synchronize()

        fwd_start.record()
        out = run_backend(q, k, v, causal=causal, backend=backend, deterministic=deterministic)
        fwd_end.record()
        out.backward(dout)
        bwd_end.record()
        torch.cuda.synchronize()

        fwd_ms += fwd_start.elapsed_time(fwd_end)
        total_ms += fwd_start.elapsed_time(bwd_end)
        bwd_ms += fwd_end.elapsed_time(bwd_end)

    scale = 1.0 / num_iters
    return fwd_ms * scale, bwd_ms * scale, total_ms * scale


def benchmark_case(
    batch: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    dtype: torch.dtype,
    causal: bool,
    deterministic: bool,
    backend: str,
    warmup: int,
    iters: int,
    trials: int,
    seed: int,
):
    q, k, v, dout = make_inputs(batch, seqlen, nheads, headdim, dtype, seed)

    if warmup > 0:
        run_timed_iters(
            q, k, v, dout, causal=causal, deterministic=deterministic, backend=backend, num_iters=warmup
        )

    fwd_trials = []
    bwd_trials = []
    total_trials = []
    for trial_idx in range(trials):
        trial_seed = seed + trial_idx + 1
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)
        fwd_ms, bwd_ms, total_ms = run_timed_iters(
            q, k, v, dout, causal=causal, deterministic=deterministic, backend=backend, num_iters=iters
        )
        fwd_trials.append(fwd_ms)
        bwd_trials.append(bwd_ms)
        total_trials.append(total_ms)

    def mean_std(values: List[float]):
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        return mean, std

    fwd_mean, fwd_std = mean_std(fwd_trials)
    bwd_mean, bwd_std = mean_std(bwd_trials)
    total_mean, total_std = mean_std(total_trials)
    return fwd_mean, fwd_std, bwd_mean, bwd_std, total_mean, total_std


def worker_main(args):
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtypes]
    fwd_mean, fwd_std, bwd_mean, bwd_std, total_mean, total_std = benchmark_case(
        batch=args.batch,
        seqlen=parse_int_list(args.seqlens)[0],
        nheads=args.nheads,
        headdim=parse_int_list(args.dims)[0],
        dtype=dtype,
        causal=args.causal == "true",
        deterministic=args.deterministic == "true",
        backend=args.backend,
        warmup=args.warmup,
        iters=args.iters,
        trials=args.trials,
        seed=args.base_seed,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "fwd_ms_mean": fwd_mean,
                "fwd_ms_std": fwd_std,
                "bwd_ms_mean": bwd_mean,
                "bwd_ms_std": bwd_std,
                "total_ms_mean": total_mean,
                "total_ms_std": total_std,
            }
        )
    )


def run_case_isolated(
    args, dtype_name: str, causal: bool, deterministic: bool, headdim: int, seqlen: int, backend: str
):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--tag",
        args.tag,
        "--batch",
        str(args.batch),
        "--nheads",
        str(args.nheads),
        "--dims",
        str(headdim),
        "--dtypes",
        dtype_name,
        "--backend",
        backend,
        "--causal",
        "true" if causal else "false",
        "--deterministic",
        "true" if deterministic else "false",
        "--seqlens",
        str(seqlen),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--trials",
        str(args.trials),
        "--base-seed",
        str(args.base_seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if proc.returncode != 0:
        merged = []
        if proc.stdout.strip():
            merged.extend([line.rstrip() for line in proc.stdout.splitlines() if line.strip()])
        if proc.stderr.strip():
            merged.extend([line.rstrip() for line in proc.stderr.splitlines() if line.strip()])

        if not merged:
            return {"ok": False, "error": f"worker exited with code {proc.returncode}"}

        runtime_lines = [line for line in merged if "RuntimeError:" in line or "HIP Function Failed" in line]
        if runtime_lines:
            summary = " | ".join(runtime_lines[-2:])
        else:
            summary = " | ".join(merged[-5:])

        return {"ok": False, "error": f"rc={proc.returncode} {summary}"[:1000]}

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "error": "worker produced no output"}

    try:
        payload = json.loads(lines[-1])
    except Exception:
        return {"ok": False, "error": f"worker produced non-JSON tail: {lines[-1][:200]}"}
    return payload


def main():
    parser = argparse.ArgumentParser(description="CK Tile 性能测试（基于 test_ck_tile_probe.py 的输入构造）")
    parser.add_argument("--tag", type=str, default="perf")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--dims", type=str, default="64")
    parser.add_argument("--dtypes", type=str, default="fp16")
    parser.add_argument("--backend", type=str, default="flash", choices=["flash", "sdpa", "sdpa_math", "both"])
    parser.add_argument("--causal", type=str, default="false", choices=["both", "true", "false"])
    parser.add_argument(
        "--deterministic", type=str, default="false", choices=["both", "true", "false"]
    )
    parser.add_argument("--seqlens", type=str, default="64")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无可用 HIP/CUDA 设备")

    if args.worker:
        worker_main(args)
        return

    print("=" * 100)
    print("Attention 性能测试")
    print(f"tag={args.tag}")
    print(f"torch={torch.__version__}, hip={torch.version.hip}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"flash_attn={flash_attn.__file__}")
    print(f"flash_attn_2_cuda={flash_attn_2_cuda.__file__}")
    print("=" * 100)

    dim_list = parse_int_list(args.dims)
    seqlen_list = parse_int_list(args.seqlens)
    causal_list = [False, True] if args.causal == "both" else [args.causal == "true"]
    backend_list = ["flash", "sdpa"] if args.backend == "both" else [args.backend]
    dtype_names = [item.strip().lower() for item in args.dtypes.split(",") if item.strip()]
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}

    rows: List[PerfRow] = []
    for dtype_name in dtype_names:
        if dtype_name not in dtype_map:
            raise ValueError(f"不支持 dtype: {dtype_name}")
        dtype = dtype_map[dtype_name]

        for causal in causal_list:
            for backend in backend_list:
                deterministic_list = (
                    [False, True]
                    if backend == "flash" and args.deterministic == "both"
                    else [args.deterministic == "true"]
                    if backend == "flash"
                    else [False]
                )
                for deterministic in deterministic_list:
                    for headdim in dim_list:
                        for seqlen in seqlen_list:
                            result = run_case_isolated(
                                args=args,
                                dtype_name=dtype_name,
                                causal=causal,
                                deterministic=deterministic,
                                headdim=headdim,
                                seqlen=seqlen,
                                backend=backend,
                            )

                            row = PerfRow(
                                tag=args.tag,
                                backend=backend,
                                batch=args.batch,
                                seqlen=seqlen,
                                nheads=args.nheads,
                                headdim=headdim,
                                dtype=dtype_name,
                                causal=causal,
                                deterministic="n/a" if backend != "flash" else str(deterministic),
                                warmup=args.warmup,
                                iters=args.iters,
                                trials=args.trials,
                                fwd_ms_mean=result.get("fwd_ms_mean", float("nan")),
                                fwd_ms_std=result.get("fwd_ms_std", float("nan")),
                                bwd_ms_mean=result.get("bwd_ms_mean", float("nan")),
                                bwd_ms_std=result.get("bwd_ms_std", float("nan")),
                                total_ms_mean=result.get("total_ms_mean", float("nan")),
                                total_ms_std=result.get("total_ms_std", float("nan")),
                                ok=bool(result.get("ok", False)),
                                error=result.get("error", ""),
                            )
                            rows.append(row)

                            if row.ok:
                                print(
                                    f"backend={backend:<9} dtype={dtype_name:<4} causal={str(causal):<5} det={row.deterministic:<5} "
                                    f"B={args.batch:<3} H={args.nheads:<3} D={headdim:<3} S={seqlen:<5} "
                                    f"fwd={row.fwd_ms_mean:.3f}±{row.fwd_ms_std:.3f} ms "
                                    f"bwd={row.bwd_ms_mean:.3f}±{row.bwd_ms_std:.3f} ms "
                                    f"total={row.total_ms_mean:.3f}±{row.total_ms_std:.3f} ms "
                                    f"(warmup={args.warmup}, iters={args.iters}, trials={args.trials})"
                                )
                            else:
                                print(
                                    f"backend={backend:<9} dtype={dtype_name:<4} causal={str(causal):<5} det={row.deterministic:<5} "
                                    f"B={args.batch:<3} H={args.nheads:<3} D={headdim:<3} S={seqlen:<5} "
                                    f"FAILED error={row.error}"
                                )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow([
                "tag",
                "backend",
                "batch",
                "seqlen",
                "nheads",
                "headdim",
                "dtype",
                "causal",
                "deterministic",
                "warmup",
                "iters",
                "trials",
                "fwd_ms_mean",
                "fwd_ms_std",
                "bwd_ms_mean",
                "bwd_ms_std",
                "total_ms_mean",
                "total_ms_std",
                "ok",
                "error",
            ])
            for row in rows:
                writer.writerow([
                    row.tag,
                    row.backend,
                    row.batch,
                    row.seqlen,
                    row.nheads,
                    row.headdim,
                    row.dtype,
                    int(row.causal),
                    row.deterministic,
                    row.warmup,
                    row.iters,
                    row.trials,
                    f"{row.fwd_ms_mean:.6f}",
                    f"{row.fwd_ms_std:.6f}",
                    f"{row.bwd_ms_mean:.6f}",
                    f"{row.bwd_ms_std:.6f}",
                    f"{row.total_ms_mean:.6f}",
                    f"{row.total_ms_std:.6f}",
                    int(row.ok),
                    row.error,
                ])
        print(f"\n已写出 CSV: {args.csv}")


if __name__ == "__main__":
    main()
