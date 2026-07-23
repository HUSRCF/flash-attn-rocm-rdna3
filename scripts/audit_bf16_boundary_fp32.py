#!/usr/bin/env python3
"""Audit the accepted BF16 boundary cases against BF16 and true FP32 SDPA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test_prebuilt_flash_attn_ck import (
    bf16_ulp_at_max_abs,
    clone_with_grad,
    force_sdpa_math,
    make_tensors,
    max_abs,
    max_abs_value,
)
from flash_attn import flash_attn_func


# These are the three rows identified by the frozen 96-case standard matrix.
CASES = (
    (768, 64, 7168017),
    (2048, 64, 8448018),
    (2048, 256, 27648018),
)


FP32_ERROR_MULTIPLIER = 4.0
BF16_FLOOR_MULTIPLIER = 4.0
BF16_FRACTION_BITS = 7


def bf16_local_ulp(reference: torch.Tensor) -> torch.Tensor:
    """Return BF16 spacing at each FP32 reference value."""
    absolute = reference.detach().float().abs()
    _, exponent = torch.frexp(absolute)
    normal_ulp = torch.ldexp(
        torch.ones_like(absolute), exponent - (BF16_FRACTION_BITS + 1)
    )
    info = torch.finfo(torch.bfloat16)
    minimum_subnormal = info.tiny * info.eps
    return torch.where(
        absolute < info.tiny,
        torch.full_like(absolute, minimum_subnormal),
        normal_ulp,
    )


def _finite_max(tensor: torch.Tensor) -> float:
    if not bool(torch.isfinite(tensor).all().item()):
        return float("inf")
    return float(tensor.max().item())


def _finite_mean(tensor: torch.Tensor) -> float:
    if not bool(torch.isfinite(tensor).all().item()):
        return float("inf")
    return float(tensor.mean().item())


def fp32_oracle_envelope(
    candidate: torch.Tensor,
    bf16_math: torch.Tensor,
    fp32_math: torch.Tensor,
) -> dict[str, object]:
    """Gate a BF16 result against a true FP32 oracle and BF16 error budget."""
    candidate_fp32 = candidate.detach().float()
    bf16_math_fp32 = bf16_math.detach().float()
    fp32_reference = fp32_math.detach().float()

    candidate_error = (candidate_fp32 - fp32_reference).abs()
    dtype_error = (bf16_math_fp32 - fp32_reference).abs()
    floor = BF16_FLOOR_MULTIPLIER * bf16_local_ulp(fp32_reference)
    allowance = FP32_ERROR_MULTIPLIER * dtype_error + floor
    ratio = candidate_error / allowance.clamp(min=1.0e-12)
    finite = (
        torch.isfinite(candidate_fp32)
        & torch.isfinite(bf16_math_fp32)
        & torch.isfinite(fp32_reference)
        & torch.isfinite(candidate_error)
        & torch.isfinite(allowance)
        & torch.isfinite(ratio)
    )
    violations = (~finite) | (candidate_error > allowance)

    return {
        "ok": not bool(violations.any().item()),
        "violations": int(violations.sum().item()),
        "elements": candidate.numel(),
        "max_ratio": _finite_max(ratio),
        "candidate_max_abs": _finite_max(candidate_error),
        "candidate_mae": _finite_mean(candidate_error),
        "bf16_math_max_abs": _finite_max(dtype_error),
        "bf16_math_mae": _finite_mean(dtype_error),
        "allowance_max": _finite_max(allowance),
    }


def run_case(
    *,
    device: str,
    seqlen: int,
    headdim: int,
    seed: int,
    grad_scale: float,
    scaled_atol: float,
    relative_atol: float,
    max_ulp: int,
) -> dict[str, object]:
    q0, k0, v0, dout0 = make_tensors(
        batch=2,
        seqlen=seqlen,
        nheads=4,
        headdim=headdim,
        dtype=torch.bfloat16,
        device=device,
        seed=seed,
        grad_scale=grad_scale,
    )

    q, k, v, dout = clone_with_grad(q0, k0, v0, dout0)
    out = flash_attn_func(q, k, v, causal=True, deterministic=False)
    out.backward(dout)
    torch.cuda.synchronize()
    actual = {
        "out": out.detach(),
        "dq": q.grad.detach(),
        "dk": k.grad.detach(),
        "dv": v.grad.detach(),
    }

    q_bf16, k_bf16, v_bf16, dout_bf16 = clone_with_grad(q0, k0, v0, dout0)
    out_bf16 = force_sdpa_math(
        q_bf16.transpose(1, 2),
        k_bf16.transpose(1, 2),
        v_bf16.transpose(1, 2),
        True,
    ).transpose(1, 2)
    out_bf16.backward(dout_bf16)
    torch.cuda.synchronize()
    bf16_math = {
        "out": out_bf16.detach(),
        "dq": q_bf16.grad.detach(),
        "dk": k_bf16.grad.detach(),
        "dv": v_bf16.grad.detach(),
    }

    # This branch deliberately promotes the inputs before the forward pass and
    # retains FP32 gradients, unlike the ordinary BF16 math-SDPA comparison.
    q_fp32 = q0.detach().float().requires_grad_(True)
    k_fp32 = k0.detach().float().requires_grad_(True)
    v_fp32 = v0.detach().float().requires_grad_(True)
    out_fp32 = force_sdpa_math(
        q_fp32.transpose(1, 2),
        k_fp32.transpose(1, 2),
        v_fp32.transpose(1, 2),
        True,
    ).transpose(1, 2)
    out_fp32.backward(dout0.float())
    torch.cuda.synchronize()
    fp32_math = {
        "out": out_fp32.detach(),
        "dq": q_fp32.grad.detach(),
        "dk": k_fp32.grad.detach(),
        "dv": v_fp32.grad.detach(),
    }

    gradient_names = ("dq", "dk", "dv")
    gradient_abs = {
        name: max_abs(actual[name], bf16_math[name]) for name in gradient_names
    }
    gradient_relative = {
        name: gradient_abs[name]
        / max(max_abs_value(bf16_math[name]), 1.0e-30)
        for name in gradient_names
    }
    bwd_abs = max(gradient_abs.values())
    bwd_relative = max(gradient_relative.values())
    scale = abs(grad_scale) if grad_scale else 1.0
    bwd_scaled = bwd_abs / scale
    bwd_ulp_at_max_abs = max(
        bf16_ulp_at_max_abs(actual[name], bf16_math[name])
        for name in gradient_names
        if gradient_abs[name] == bwd_abs
    )
    boundary_ok = (
        bwd_scaled <= scaled_atol
        and bwd_relative <= relative_atol
        and bwd_ulp_at_max_abs <= max_ulp
    )

    oracle = {
        name: fp32_oracle_envelope(actual[name], bf16_math[name], fp32_math[name])
        for name in ("out", *gradient_names)
    }
    fp32_oracle_ok = all(bool(metrics["ok"]) for metrics in oracle.values())
    fp32_oracle_max_ratio = max(float(metrics["max_ratio"]) for metrics in oracle.values())

    dv_bf16_abs = (actual["dv"].float() - bf16_math["dv"].float()).abs()
    dv_max_index = int(dv_bf16_abs.flatten().argmax().item())
    dv_candidate_fp32_error = (actual["dv"].float() - fp32_math["dv"]).abs()
    dv_bf16_fp32_error = (bf16_math["dv"].float() - fp32_math["dv"]).abs()

    result: dict[str, object] = {
        "seqlen": seqlen,
        "headdim": headdim,
        "seed": seed,
        "grad_scale": grad_scale,
        "dq_max_abs_vs_bf16_math": gradient_abs["dq"],
        "dk_max_abs_vs_bf16_math": gradient_abs["dk"],
        "dv_max_abs_vs_bf16_math": gradient_abs["dv"],
        "bwd_max_abs_vs_bf16_math": bwd_abs,
        "bwd_scaled_abs_vs_bf16_math": bwd_scaled,
        "bwd_relative_vs_bf16_math": bwd_relative,
        "bwd_ulp_at_max_abs": bwd_ulp_at_max_abs,
        "dv_candidate_at_max": float(actual["dv"].flatten()[dv_max_index].item()),
        "dv_bf16_math_at_max": float(bf16_math["dv"].flatten()[dv_max_index].item()),
        "dv_fp32_math_at_max": float(fp32_math["dv"].flatten()[dv_max_index].item()),
        "dv_candidate_closer_to_fp32": int(
            (dv_candidate_fp32_error < dv_bf16_fp32_error).sum().item()
        ),
        "dv_bf16_math_closer_to_fp32": int(
            (dv_bf16_fp32_error < dv_candidate_fp32_error).sum().item()
        ),
        "dv_fp32_distance_ties": int(
            (dv_candidate_fp32_error == dv_bf16_fp32_error).sum().item()
        ),
        "bwd_bf16_mismatch_count": sum(
            int((actual[name] != bf16_math[name]).sum().item())
            for name in gradient_names
        ),
        "boundary_ok": boundary_ok,
        "fp32_oracle_ok": fp32_oracle_ok,
        "fp32_oracle_max_ratio": fp32_oracle_max_ratio,
        "ok": boundary_ok and fp32_oracle_ok,
    }
    for name, metrics in oracle.items():
        for metric_name, value in metrics.items():
            result[f"{name}_fp32_{metric_name}"] = value
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grad-scale", type=float, default=100.0)
    parser.add_argument("--scaled-atol", type=float, default=4.1e-2)
    parser.add_argument("--relative-atol", type=float, default=8.0e-3)
    parser.add_argument("--max-ulp", type=int, default=1)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("testoutput/make-release/bf16/fp32_boundary_audit.csv"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("testoutput/make-release/bf16/fp32_boundary_audit.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA/HIP device is visible")
    torch.cuda.set_device(args.device)

    rows = [
        run_case(
            device=args.device,
            seqlen=seqlen,
            headdim=headdim,
            seed=seed,
            grad_scale=args.grad_scale,
            scaled_atol=args.scaled_atol,
            relative_atol=args.relative_atol,
            max_ulp=args.max_ulp,
        )
        for seqlen, headdim, seed in CASES
    ]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "device": args.device,
        "thresholds": {
            "scaled_atol": args.scaled_atol,
            "relative_atol": args.relative_atol,
            "max_ulp": args.max_ulp,
            "fp32_error_multiplier": FP32_ERROR_MULTIPLIER,
            "bf16_floor_multiplier": BF16_FLOOR_MULTIPLIER,
        },
        "passed": all(bool(row["ok"]) for row in rows),
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    for row in rows:
        print(
            f"S={row['seqlen']} D={row['headdim']} ok={row['ok']} "
            f"bwd_scaled={row['bwd_scaled_abs_vs_bf16_math']} "
            f"bwd_relative={row['bwd_relative_vs_bf16_math']} "
            f"bwd_ulp_at_max={row['bwd_ulp_at_max_abs']} "
            f"fp32_gate={row['fp32_oracle_ok']} "
            f"fp32_max_ratio={row['fp32_oracle_max_ratio']}"
        )
    print(f"csv={args.csv}")
    print(f"json={args.json}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
