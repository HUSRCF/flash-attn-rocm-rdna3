import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

import flash_attn
import flash_attn_2_cuda
from flash_attn import flash_attn_func


def force_sdpa_math(q, k, v, is_causal: bool):
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    except Exception:
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def check_close(
    a: torch.Tensor, b: torch.Tensor, atol: float = 5e-3, rtol: float = 1e-2
) -> Tuple[float, bool]:
    max_abs_err = (a - b).abs().max().item()
    is_close = torch.allclose(a, b, atol=atol, rtol=rtol)
    return max_abs_err, is_close


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
    fwd_close: bool
    dq_close: bool
    dk_close: bool
    dv_close: bool
    ok: bool


def run_case(
    batch,
    seqlen,
    nheads,
    headdim,
    dtype,
    causal,
    seed,
    fw_atol,
    fw_rtol,
    bw_atol,
    bw_rtol,
) -> Tuple[float, bool, float, bool, float, bool, float, bool]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    dout = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda")

    out_fa = flash_attn_func(q, k, v, causal=causal)
    out_fa.backward(dout)
    dq_fa, dk_fa, dv_fa = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out_ref = force_sdpa_math(qh, kh, vh, is_causal=causal).transpose(1, 2)
    out_ref.backward(dout)
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

    fwd_err, fwd_close = check_close(out_fa, out_ref, atol=fw_atol, rtol=fw_rtol)
    dq_err, dq_close = check_close(dq_fa, dq_ref, atol=bw_atol, rtol=bw_rtol)
    dk_err, dk_close = check_close(dk_fa, dk_ref, atol=bw_atol, rtol=bw_rtol)
    dv_err, dv_close = check_close(dv_fa, dv_ref, atol=bw_atol, rtol=bw_rtol)
    return fwd_err, fwd_close, dq_err, dq_close, dk_err, dk_close, dv_err, dv_close


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
    parser = argparse.ArgumentParser(description="第4步：CK tile 边界误差探针 v3（max abs + torch.allclose）")
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
    parser.add_argument("--fw-rtol", type=float, default=1e-2)
    parser.add_argument("--bw-atol", type=float, default=5e-3)
    parser.add_argument("--bw-rtol", type=float, default=1e-2)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无可用 HIP/CUDA 设备")

    print("=" * 100)
    print("CK Tile 边界探针 v3")
    print(f"tag={args.tag}")
    print(f"torch={torch.__version__}, hip={torch.version.hip}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"flash_attn={flash_attn.__file__}")
    print(f"flash_attn_2_cuda={flash_attn_2_cuda.__file__}")
    print(
        f"close rule: fwd(atol={args.fw_atol}, rtol={args.fw_rtol}), "
        f"bwd(atol={args.bw_atol}, rtol={args.bw_rtol})"
    )
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
                    dq_vals = []
                    dk_vals = []
                    dv_vals = []
                    fwd_ok_all = True
                    dq_ok_all = True
                    dk_ok_all = True
                    dv_ok_all = True

                    for repeat_idx in range(args.repeats):
                        seed = args.base_seed + repeat_idx
                        torch.cuda.synchronize()
                        fwd_err, fwd_close, dq_err, dq_close, dk_err, dk_close, dv_err, dv_close = run_case(
                            args.batch,
                            seqlen,
                            args.nheads,
                            headdim,
                            dtype,
                            causal,
                            seed,
                            args.fw_atol,
                            args.fw_rtol,
                            args.bw_atol,
                            args.bw_rtol,
                        )
                        fwd_vals.append(fwd_err)
                        dq_vals.append(dq_err)
                        dk_vals.append(dk_err)
                        dv_vals.append(dv_err)
                        fwd_ok_all = fwd_ok_all and fwd_close
                        dq_ok_all = dq_ok_all and dq_close
                        dk_ok_all = dk_ok_all and dk_close
                        dv_ok_all = dv_ok_all and dv_close

                    fwd_mean = sum(fwd_vals) / len(fwd_vals)
                    dq_mean = sum(dq_vals) / len(dq_vals)
                    dk_mean = sum(dk_vals) / len(dk_vals)
                    dv_mean = sum(dv_vals) / len(dv_vals)
                    bwd_mean = max(dq_mean, dk_mean, dv_mean)
                    finite = all(
                        is_finite_number(value)
                        for value in [fwd_mean, bwd_mean, dq_mean, dk_mean, dv_mean]
                    )
                    ok = finite and fwd_ok_all and dq_ok_all and dk_ok_all and dv_ok_all
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
                        fwd_close=fwd_ok_all,
                        dq_close=dq_ok_all,
                        dk_close=dk_ok_all,
                        dv_close=dv_ok_all,
                        ok=ok,
                    )
                    rows.append(row)
                    total += 1

                    icon = "OK" if ok else "BAD"
                    print(
                        f"[{icon}] dtype={dtype_name:<4} causal={str(causal):<5} D={headdim:<3} S={seqlen:<4} "
                        f"fwd={fwd_mean:.6f} bwd={bwd_mean:.6f} "
                        f"dq={dq_mean:.6f} dk={dk_mean:.6f} dv={dv_mean:.6f} "
                        f"close=f{int(fwd_ok_all)} dq{int(dq_ok_all)} dk{int(dk_ok_all)} dv{int(dv_ok_all)} "
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
                f"close=f{int(row.fwd_close)} dq{int(row.dq_close)} dk{int(row.dk_close)} dv{int(row.dv_close)}, "
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
                "fwd_err", "bwd_err", "dq_err", "dk_err", "dv_err",
                "fwd_close", "dq_close", "dk_close", "dv_close", "ok", "bucket64", "mod64"
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
                    int(row.fwd_close),
                    int(row.dq_close),
                    int(row.dk_close),
                    int(row.dv_close),
                    int(row.ok),
                    (row.seqlen + 63) // 64,
                    row.seqlen % 64,
                ])


if __name__ == "__main__":
    main()
