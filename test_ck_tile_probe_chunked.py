import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

import flash_attn
import flash_attn_2_cuda
from safe_rdna3_chunked_bwd import safe_rdna3_chunked_flash_attn_fwd_bwd


def force_sdpa_math(q, k, v, is_causal: bool):
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    except Exception:
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def is_finite_number(x: float) -> bool:
    return math.isfinite(x)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_boundary_seqlens(mode: str) -> List[int]:
    values = [
        31, 32, 33,
        63, 64, 65,
        95, 96, 97,
        127, 128, 129,
        191, 192, 193,
        255, 256, 257,
        319, 320, 321,
        383, 384, 385,
        511, 512, 513,
        767, 768, 769,
        1023, 1024, 1025,
    ]
    if mode == "full":
        values += [1535, 1536, 1537, 2047, 2048, 2049]
    return sorted(set(values))


@dataclass
class Row:
    tag: str
    seqlen: int
    headdim: int
    dtype: str
    causal: bool
    fwd_err: float
    bwd_err: float
    dq_err: float
    dk_err: float
    dv_err: float
    ok: bool


def run_case(
    batch,
    seqlen,
    nheads,
    headdim,
    dtype,
    causal,
    seed,
    chunk_size,
) -> Tuple[float, float, float, float, float]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    dout = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda")

    out_fa, dq_fa, dk_fa, dv_fa = safe_rdna3_chunked_flash_attn_fwd_bwd(
        q,
        k,
        v,
        dout,
        dropout_p=0.0,
        causal=causal,
        deterministic=False,
        chunk_size=chunk_size,
    )
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out_ref = force_sdpa_math(qh, kh, vh, is_causal=causal).transpose(1, 2)
    out_ref.backward(dout)
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

    fwd = max_abs(out_fa, out_ref)
    dq_err = max_abs(dq_fa, dq_ref)
    dk_err = max_abs(dk_fa, dk_ref)
    dv_err = max_abs(dv_fa, dv_ref)
    bwd = max(dq_err, dk_err, dv_err)
    return fwd, bwd, dq_err, dk_err, dv_err


def detect_transitions(rows: List[Row]) -> List[str]:
    messages: List[str] = []
    grouped = {}
    for row in rows:
        key = (row.dtype, row.causal, row.headdim)
        grouped.setdefault(key, []).append(row)

    for key, values in grouped.items():
        values = sorted(values, key=lambda item: item.seqlen)
        for idx in range(1, len(values)):
            prev_item, curr_item = values[idx - 1], values[idx]
            jump = (curr_item.fwd_err > prev_item.fwd_err * 4 and curr_item.fwd_err > 0.05) or (
                curr_item.bwd_err > prev_item.bwd_err * 4 and curr_item.bwd_err > 0.05
            )
            state_flip = prev_item.ok != curr_item.ok
            if jump or state_flip:
                messages.append(
                    f"dtype={key[0]}, causal={key[1]}, D={key[2]}: "
                    f"S {prev_item.seqlen}->{curr_item.seqlen}, "
                    f"fwd {prev_item.fwd_err:.4f}->{curr_item.fwd_err:.4f}, "
                    f"bwd {prev_item.bwd_err:.4f}->{curr_item.bwd_err:.4f}, "
                    f"ok {prev_item.ok}->{curr_item.ok}"
                )
    return messages


def main():
    parser = argparse.ArgumentParser(description="CK tile 边界探针（chunked backward wrapper vs SDPA-MATH）")
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--dims", type=str, default="64,96,128")
    parser.add_argument("--dtypes", type=str, default="fp16,bf16")
    parser.add_argument("--causal", type=str, default="both", choices=["both", "true", "false"])
    parser.add_argument("--seqlens", type=str, default="")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--fw-atol", type=float, default=2e-3)
    parser.add_argument("--bw-atol", type=float, default=5e-3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无可用 HIP/CUDA 设备")

    print("=" * 100)
    print("CK Tile 边界探针（Chunked BWD Wrapper）")
    print(f"tag={args.tag}")
    print(f"chunk_size={args.chunk_size}")
    print(f"torch={torch.__version__}, hip={torch.version.hip}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"flash_attn={flash_attn.__file__}")
    print(f"flash_attn_2_cuda={flash_attn_2_cuda.__file__}")
    print("=" * 100)

    dim_list = parse_int_list(args.dims)
    dtype_names = [item.strip().lower() for item in args.dtypes.split(",") if item.strip()]
    causal_list = [False, True] if args.causal == "both" else [args.causal == "true"]
    seqlens = parse_int_list(args.seqlens) if args.seqlens else build_boundary_seqlens(args.mode)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    rows: List[Row] = []

    total = 0
    for dtype_name in dtype_names:
        if dtype_name not in dtype_map:
            raise ValueError(f"不支持 dtype: {dtype_name}")
        dtype = dtype_map[dtype_name]

        for causal in causal_list:
            for headdim in dim_list:
                for seqlen in seqlens:
                    fwd_vals = []
                    bwd_vals = []
                    dq_vals = []
                    dk_vals = []
                    dv_vals = []
                    for repeat_idx in range(args.repeats):
                        seed = args.base_seed + repeat_idx
                        torch.cuda.synchronize()
                        fwd, bwd, dq_err, dk_err, dv_err = run_case(
                            args.batch,
                            seqlen,
                            args.nheads,
                            headdim,
                            dtype,
                            causal,
                            seed,
                            args.chunk_size,
                        )
                        fwd_vals.append(fwd)
                        bwd_vals.append(bwd)
                        dq_vals.append(dq_err)
                        dk_vals.append(dk_err)
                        dv_vals.append(dv_err)

                    fwd_mean = sum(fwd_vals) / len(fwd_vals)
                    bwd_mean = sum(bwd_vals) / len(bwd_vals)
                    dq_mean = sum(dq_vals) / len(dq_vals)
                    dk_mean = sum(dk_vals) / len(dk_vals)
                    dv_mean = sum(dv_vals) / len(dv_vals)
                    finite = all(
                        is_finite_number(value)
                        for value in [fwd_mean, bwd_mean, dq_mean, dk_mean, dv_mean]
                    )
                    ok = finite and (fwd_mean <= args.fw_atol) and (bwd_mean <= args.bw_atol)
                    row = Row(
                        tag=args.tag,
                        seqlen=seqlen,
                        headdim=headdim,
                        dtype=dtype_name,
                        causal=causal,
                        fwd_err=fwd_mean,
                        bwd_err=bwd_mean,
                        dq_err=dq_mean,
                        dk_err=dk_mean,
                        dv_err=dv_mean,
                        ok=ok,
                    )
                    rows.append(row)
                    total += 1

                    icon = "OK" if ok else "BAD"
                    print(
                        f"[{icon}] dtype={dtype_name:<4} causal={str(causal):<5} D={headdim:<3} S={seqlen:<4} "
                        f"fwd={fwd_mean:.6f} bwd={bwd_mean:.6f} "
                        f"dq={dq_mean:.6f} dk={dk_mean:.6f} dv={dv_mean:.6f} "
                        f"bucket64={(seqlen + 63)//64:<3} mod64={seqlen % 64:<2}"
                    )

    bad = [row for row in rows if not row.ok]
    print("-" * 100)
    print(f"总 case: {total}, 失败: {len(bad)}, 通过: {total - len(bad)}")

    if bad:
        print("\nTop 20 失败样本（按 bwd_err 降序）:")
        for row in sorted(bad, key=lambda item: item.bwd_err, reverse=True)[:20]:
            print(
                f"dtype={row.dtype}, causal={row.causal}, D={row.headdim}, S={row.seqlen}, "
                f"fwd={row.fwd_err:.6f}, bwd={row.bwd_err:.6f}, "
                f"dq={row.dq_err:.6f}, dk={row.dk_err:.6f}, dv={row.dv_err:.6f}, "
                f"bucket64={(row.seqlen + 63)//64}, mod64={row.seqlen % 64}"
            )

    transitions = detect_transitions(rows)
    print("\n疑似 tile/分块切换跳点:")
    if transitions:
        for message in transitions[:40]:
            print(f"- {message}")
    else:
        print("- 未发现明显跳点（当前扫描范围内）")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow([
                "tag", "seqlen", "headdim", "dtype", "causal",
                "fwd_err", "bwd_err", "dq_err", "dk_err", "dv_err", "ok", "bucket64", "mod64"
            ])
            for row in rows:
                writer.writerow([
                    row.tag,
                    row.seqlen,
                    row.headdim,
                    row.dtype,
                    int(row.causal),
                    f"{row.fwd_err:.8f}",
                    f"{row.bwd_err:.8f}",
                    f"{row.dq_err:.8f}",
                    f"{row.dk_err:.8f}",
                    f"{row.dv_err:.8f}",
                    int(row.ok),
                    (row.seqlen + 63) // 64,
                    row.seqlen % 64,
                ])
        print(f"\n已写出 CSV: {args.csv}")


if __name__ == "__main__":
    main()
