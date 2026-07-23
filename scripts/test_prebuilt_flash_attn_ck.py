#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


def prefer_local_bundle() -> Path | None:
    if "--installed-only" in sys.argv:
        return None
    script = Path(__file__).resolve()
    root = script.parent.parent if script.parent.name == "scripts" else script.parent
    if (root / "flash_attn").is_dir() and list(root.glob("flash_attn_2_cuda*.so")):
        sys.path.insert(0, str(root))
        return root
    return None


LOCAL_ROOT = prefer_local_bundle()

import flash_attn  # noqa: E402
import flash_attn_2_cuda  # noqa: E402
from flash_attn import flash_attn_func  # noqa: E402


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> List[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_causal(value: str) -> List[bool]:
    value = value.lower()
    if value == "both":
        return [False, True]
    if value in ("true", "1", "yes", "causal"):
        return [True]
    if value in ("false", "0", "no", "noncausal"):
        return [False]
    raise ValueError(f"invalid causal selector: {value}")


def force_sdpa_math(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    except Exception:
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def sdpa_default(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def max_abs_value(a: torch.Tensor) -> float:
    return float(a.abs().max().item())


def bf16_ulp_at_max_abs(a: torch.Tensor, b: torch.Tensor) -> int:
    """Return the largest BF16 ULP distance at a maximum-error element."""
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("BF16 ULP distance requires bfloat16 tensors")

    abs_diff = (a.detach().float() - b.detach().float()).abs()
    max_diff = abs_diff.max()
    if float(max_diff.item()) == 0.0:
        return 0

    # Convert sign-magnitude BF16 encodings to a monotonically ordered integer
    # space. Positive and negative zero intentionally map to the same point.
    sign = 1 << 15
    magnitude = sign - 1

    def ordered_bits(value: torch.Tensor) -> torch.Tensor:
        raw = value.detach().view(torch.int16).to(torch.int32) & ((1 << 16) - 1)
        return torch.where(
            (raw & sign) != 0,
            sign - (raw & magnitude),
            raw + sign,
        )

    ulp = (ordered_bits(a) - ordered_bits(b)).abs()
    return int(ulp[abs_diff == max_diff].max().item())


def make_tensors(
    batch: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    grad_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device, generator=gen)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device, generator=gen)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device, generator=gen)
    dout = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device, generator=gen)
    if grad_scale != 1.0:
        dout.mul_(grad_scale)
    return q, k, v, dout


def clone_with_grad(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dout: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        q.detach().clone().requires_grad_(True),
        k.detach().clone().requires_grad_(True),
        v.detach().clone().requires_grad_(True),
        dout.detach().clone(),
    )


def backward_or_raise(out: torch.Tensor, dout: torch.Tensor) -> None:
    out.backward(dout)
    torch.cuda.synchronize()


def run_correctness_once(args, dtype_name: str, causal: bool, seqlen: int, headdim: int, seed: int) -> Dict[str, float | str | int | bool]:
    dtype = DTYPE_MAP[dtype_name]
    q0, k0, v0, dout0 = make_tensors(
        args.batch,
        seqlen,
        args.nheads,
        headdim,
        dtype,
        args.device,
        seed,
        args.grad_scale,
    )

    q, k, v, dout = clone_with_grad(q0, k0, v0, dout0)
    out_fa = flash_attn_func(q, k, v, causal=causal, deterministic=False)
    backward_or_raise(out_fa, dout)
    dq_fa, dk_fa, dv_fa = q.grad.detach(), k.grad.detach(), v.grad.detach()

    q_ref, k_ref, v_ref, dout_ref = clone_with_grad(q0, k0, v0, dout0)
    out_ref_h = force_sdpa_math(
        q_ref.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        causal,
    )
    out_ref = out_ref_h.transpose(1, 2)
    backward_or_raise(out_ref, dout_ref)
    dq_ref, dk_ref, dv_ref = q_ref.grad.detach(), k_ref.grad.detach(), v_ref.grad.detach()

    fwd_abs = max_abs(out_fa, out_ref)
    dq_abs = max_abs(dq_fa, dq_ref)
    dk_abs = max_abs(dk_fa, dk_ref)
    dv_abs = max_abs(dv_fa, dv_ref)
    bwd_abs = max(dq_abs, dk_abs, dv_abs)
    scale = abs(args.grad_scale) if args.grad_scale != 0 else 1.0
    eps = 1.0e-30
    dq_rel = dq_abs / max(max_abs_value(dq_ref), eps)
    dk_rel = dk_abs / max(max_abs_value(dk_ref), eps)
    dv_rel = dv_abs / max(max_abs_value(dv_ref), eps)
    bwd_rel = max(dq_rel, dk_rel, dv_rel)
    bwd_scaled = bwd_abs / scale

    if dtype_name == "bf16":
        gradient_pairs = (
            (dq_fa, dq_ref, dq_abs),
            (dk_fa, dk_ref, dk_abs),
            (dv_fa, dv_ref, dv_abs),
        )
        bwd_ulp_at_max_abs = max(
            bf16_ulp_at_max_abs(actual, reference)
            for actual, reference, tensor_abs in gradient_pairs
            if tensor_abs == bwd_abs
        )
    else:
        bwd_ulp_at_max_abs = -1

    fwd_atol = args.bf16_fwd_atol if dtype_name == "bf16" else args.fp16_fwd_atol
    fwd_ok = fwd_abs <= fwd_atol
    strict_bwd_ok = (
        bwd_scaled <= args.bwd_scaled_atol
        and bwd_rel <= args.bwd_rel_atol
    )
    bf16_boundary_ok = (
        dtype_name == "bf16"
        and bwd_scaled <= args.bf16_boundary_bwd_scaled_atol
        and bwd_rel <= args.bf16_boundary_bwd_rel_atol
        and bwd_ulp_at_max_abs <= args.bf16_boundary_max_ulp
    )
    ok = fwd_ok and (strict_bwd_ok or bf16_boundary_ok)
    if fwd_ok and strict_bwd_ok:
        gate = "strict"
    elif fwd_ok and bf16_boundary_ok:
        gate = "bf16_ulp_boundary"
    else:
        gate = "fail"

    return {
        "dtype": dtype_name,
        "causal": causal,
        "seqlen": seqlen,
        "headdim": headdim,
        "seed": seed,
        "fwd_abs": fwd_abs,
        "bwd_abs": bwd_abs,
        "dq_abs": dq_abs,
        "dk_abs": dk_abs,
        "dv_abs": dv_abs,
        "bwd_scaled": bwd_scaled,
        "dq_rel": dq_rel,
        "dk_rel": dk_rel,
        "dv_rel": dv_rel,
        "bwd_rel": bwd_rel,
        "bwd_ulp_at_max_abs": bwd_ulp_at_max_abs,
        "gate": gate,
        "ok": ok,
        "error": "",
    }


def summarize_correctness(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, bool, int, int], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dtype"]), bool(row["causal"]), int(row["seqlen"]), int(row["headdim"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (dtype_name, causal, seqlen, headdim), values in sorted(grouped.items()):
        numeric_keys = [
            "fwd_abs",
            "bwd_abs",
            "dq_abs",
            "dk_abs",
            "dv_abs",
            "bwd_scaled",
            "bwd_rel",
            "bwd_ulp_at_max_abs",
        ]
        item: Dict[str, object] = {
            "dtype": dtype_name,
            "causal": causal,
            "seqlen": seqlen,
            "headdim": headdim,
            "repeats": len(values),
            "gates": ",".join(sorted({str(value["gate"]) for value in values})),
            "ok": all(bool(v["ok"]) for v in values),
        }
        for key in numeric_keys:
            vals = [float(v[key]) for v in values]
            item[f"{key}_max"] = max(vals)
            item[f"{key}_mean"] = sum(vals) / len(vals)
        summary.append(item)
    return summary


def cuda_time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def run_perf_case(args, dtype_name: str, causal: bool, seqlen: int, headdim: int) -> Dict[str, object]:
    dtype = DTYPE_MAP[dtype_name]
    base = make_tensors(
        args.batch,
        seqlen,
        args.nheads,
        headdim,
        dtype,
        args.device,
        args.base_seed + 100000 + seqlen + headdim,
        1.0,
    )
    q0, k0, v0, dout0 = base

    def fa_fwd():
        with torch.no_grad():
            return flash_attn_func(q0, k0, v0, causal=causal, deterministic=False)

    def torch_default_fwd():
        with torch.no_grad():
            return sdpa_default(q0.transpose(1, 2), k0.transpose(1, 2), v0.transpose(1, 2), causal)

    def torch_math_fwd():
        with torch.no_grad():
            return force_sdpa_math(q0.transpose(1, 2), k0.transpose(1, 2), v0.transpose(1, 2), causal)

    def fa_bwd():
        q, k, v, dout = clone_with_grad(q0, k0, v0, dout0)
        out = flash_attn_func(q, k, v, causal=causal, deterministic=False)
        out.backward(dout)

    def torch_default_bwd():
        q, k, v, dout = clone_with_grad(q0, k0, v0, dout0)
        out = sdpa_default(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal).transpose(1, 2)
        out.backward(dout)

    def torch_math_bwd():
        q, k, v, dout = clone_with_grad(q0, k0, v0, dout0)
        out = force_sdpa_math(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal).transpose(1, 2)
        out.backward(dout)

    fa_fwd_ms = cuda_time_ms(fa_fwd, args.perf_warmup, args.perf_iters)
    torch_default_fwd_ms = cuda_time_ms(torch_default_fwd, args.perf_warmup, args.perf_iters)
    torch_math_fwd_ms = cuda_time_ms(torch_math_fwd, args.perf_warmup, args.perf_iters)
    fa_bwd_ms = cuda_time_ms(fa_bwd, args.perf_warmup, args.perf_iters)
    torch_default_bwd_ms = cuda_time_ms(torch_default_bwd, args.perf_warmup, args.perf_iters)
    torch_math_bwd_ms = cuda_time_ms(torch_math_bwd, args.perf_warmup, args.perf_iters)

    torch_best_fwd_ms = min(torch_default_fwd_ms, torch_math_fwd_ms)
    torch_best_bwd_ms = min(torch_default_bwd_ms, torch_math_bwd_ms)
    return {
        "dtype": dtype_name,
        "causal": causal,
        "seqlen": seqlen,
        "headdim": headdim,
        "flash_fwd_ms": fa_fwd_ms,
        "torch_default_fwd_ms": torch_default_fwd_ms,
        "torch_math_fwd_ms": torch_math_fwd_ms,
        "torch_best_fwd_ms": torch_best_fwd_ms,
        "flash_fwd_speedup_vs_torch_best": torch_best_fwd_ms / fa_fwd_ms,
        "flash_bwd_ms": fa_bwd_ms,
        "torch_default_bwd_ms": torch_default_bwd_ms,
        "torch_math_bwd_ms": torch_math_bwd_ms,
        "torch_best_bwd_ms": torch_best_bwd_ms,
        "flash_bwd_speedup_vs_torch_best": torch_best_bwd_ms / fa_bwd_ms,
    }


def env_info(device: str) -> Dict[str, object]:
    info: Dict[str, object] = {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "flash_attn": str(Path(flash_attn.__file__).resolve()),
        "flash_attn_2_cuda": str(Path(flash_attn_2_cuda.__file__).resolve()),
        "local_root": str(LOCAL_ROOT) if LOCAL_ROOT else "",
    }
    if torch.cuda.is_available():
        index = torch.device(device).index
        index = 0 if index is None else index
        prop = torch.cuda.get_device_properties(index)
        info.update(
            {
                "device_name": prop.name,
                "device_index": index,
                "gcn_arch": getattr(prop, "gcnArchName", ""),
                "total_memory_gb": round(prop.total_memory / (1024**3), 3),
            }
        )
    return info


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prebuilt FA4 CK FlashAttention correctness/perf validation.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--dims", default="64,128,256")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--causal", default="both")
    parser.add_argument("--seqlens", default="768,1024,2048")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--grad-scale", type=float, default=100.0)
    parser.add_argument("--fp16-fwd-atol", type=float, default=2.0e-3)
    parser.add_argument("--bf16-fwd-atol", type=float, default=2.0e-2)
    parser.add_argument("--bwd-scaled-atol", type=float, default=3.0e-2)
    parser.add_argument("--bwd-rel-atol", type=float, default=3.0e-2)
    parser.add_argument("--bf16-boundary-bwd-scaled-atol", type=float, default=4.1e-2)
    parser.add_argument("--bf16-boundary-bwd-rel-atol", type=float, default=8.0e-3)
    parser.add_argument("--bf16-boundary-max-ulp", type=int, default=1)
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--perf-iters", type=int, default=20)
    parser.add_argument("--perf-warmup", type=int, default=5)
    parser.add_argument("--csv", default="testoutput/prebuilt_flash_attn_ck_correctness.csv")
    parser.add_argument("--summary-csv", default="testoutput/prebuilt_flash_attn_ck_summary.csv")
    parser.add_argument("--perf-csv", default="testoutput/prebuilt_flash_attn_ck_perf.csv")
    parser.add_argument("--json", default="testoutput/prebuilt_flash_attn_ck_report.json")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--installed-only", action="store_true", help="Do not prefer the extracted bundle path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA/HIP device is visible")
    torch.cuda.set_device(args.device)

    dims = parse_int_list(args.dims)
    dtypes = parse_str_list(args.dtypes)
    causal_values = parse_causal(args.causal)
    seqlens = parse_int_list(args.seqlens)
    for dtype_name in dtypes:
        if dtype_name not in DTYPE_MAP:
            raise ValueError(f"unsupported dtype: {dtype_name}")

    print("=" * 100)
    print("FA4 CK prebuilt validation")
    for key, value in env_info(args.device).items():
        print(f"{key}={value}")
    print(
        f"cases=dims={dims}, dtypes={dtypes}, causal={causal_values}, "
        f"seqlens={seqlens}, repeats={args.repeats}, grad_scale={args.grad_scale}"
    )
    print("=" * 100)

    correctness_rows: List[Dict[str, object]] = []
    perf_rows: List[Dict[str, object]] = []
    failures = 0
    start_time = time.time()

    for dtype_name in dtypes:
        for causal in causal_values:
            for headdim in dims:
                for seqlen in seqlens:
                    for rep in range(args.repeats):
                        seed = args.base_seed + rep + 1000 * seqlen + 100000 * headdim + (17 if causal else 0)
                        try:
                            row = run_correctness_once(args, dtype_name, causal, seqlen, headdim, seed)
                        except RuntimeError as exc:
                            if "out of memory" in str(exc).lower():
                                torch.cuda.empty_cache()
                            row = {
                                "dtype": dtype_name,
                                "causal": causal,
                                "seqlen": seqlen,
                                "headdim": headdim,
                                "seed": seed,
                                "ok": False,
                                "error": repr(exc),
                            }
                        correctness_rows.append(row)
                        status = "OK" if row.get("ok") else "BAD"
                        if not row.get("ok"):
                            failures += 1
                        print(
                            f"[{status}] dtype={dtype_name} causal={causal} D={headdim:<3} S={seqlen:<5} "
                            f"rep={rep} fwd={float(row.get('fwd_abs', math.nan)):.6f} "
                            f"bwd_scaled={float(row.get('bwd_scaled', math.nan)):.6f} "
                            f"bwd_rel={float(row.get('bwd_rel', math.nan)):.6f} "
                            f"bwd_ulp_at_max_abs={int(row.get('bwd_ulp_at_max_abs', -1))} "
                            f"gate={row.get('gate', 'fail')} "
                            f"err={row.get('error', '')}"
                        )
                        if failures and args.fail_fast:
                            break
                    if failures and args.fail_fast:
                        break

                    if args.perf:
                        try:
                            perf = run_perf_case(args, dtype_name, causal, seqlen, headdim)
                            perf_rows.append(perf)
                            print(
                                f"[PERF] dtype={dtype_name} causal={causal} D={headdim:<3} S={seqlen:<5} "
                                f"fwd={perf['flash_fwd_ms']:.4f}ms "
                                f"fwd_speedup={perf['flash_fwd_speedup_vs_torch_best']:.3f}x "
                                f"bwd={perf['flash_bwd_ms']:.4f}ms "
                                f"bwd_speedup={perf['flash_bwd_speedup_vs_torch_best']:.3f}x"
                            )
                        except RuntimeError as exc:
                            if "out of memory" in str(exc).lower():
                                torch.cuda.empty_cache()
                            perf_rows.append(
                                {
                                    "dtype": dtype_name,
                                    "causal": causal,
                                    "seqlen": seqlen,
                                    "headdim": headdim,
                                    "error": repr(exc),
                                }
                            )
                            print(f"[PERF_BAD] dtype={dtype_name} causal={causal} D={headdim} S={seqlen} err={exc!r}")

    summary_rows = summarize_correctness(correctness_rows)
    write_csv(Path(args.csv), correctness_rows)
    write_csv(Path(args.summary_csv), summary_rows)
    if args.perf:
        write_csv(Path(args.perf_csv), perf_rows)

    report = {
        "env": env_info(args.device),
        "args": vars(args),
        "elapsed_sec": time.time() - start_time,
        "total_runs": len(correctness_rows),
        "failed_runs": failures,
        "strict_runs": sum(row.get("gate") == "strict" for row in correctness_rows),
        "bf16_ulp_boundary_runs": sum(
            row.get("gate") == "bf16_ulp_boundary" for row in correctness_rows
        ),
        "correctness": correctness_rows,
        "summary": summary_rows,
        "perf": perf_rows,
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("-" * 100)
    print(f"total_runs={len(correctness_rows)} failed_runs={failures}")
    print(f"csv={args.csv}")
    print(f"summary_csv={args.summary_csv}")
    if args.perf:
        print(f"perf_csv={args.perf_csv}")
    print(f"json={args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
