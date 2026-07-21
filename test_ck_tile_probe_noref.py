import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

import flash_attn
import flash_attn_2_cuda
from flash_attn import flash_attn_func


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
    out_absmax: float
    dq_absmax: float
    dk_absmax: float
    dv_absmax: float
    ok: bool


def run_case(batch, seqlen, nheads, headdim, dtype, causal, seed) -> Tuple[float, float, float, float]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda", requires_grad=True)
    dout = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda")

    out = flash_attn_func(q, k, v, causal=causal)
    out.backward(dout)

    dq = q.grad
    dk = k.grad
    dv = v.grad

    out_absmax = out.detach().abs().max().item()
    dq_absmax = dq.detach().abs().max().item()
    dk_absmax = dk.detach().abs().max().item()
    dv_absmax = dv.detach().abs().max().item()

    if dk_absmax > 100.0:
        diff = dk.detach().abs()
        max_idx = diff.argmax()
        b, s, h, d = torch.unravel_index(max_idx, diff.shape)
        print(f"\n[💥 GROUND ZERO] dk_absmax={dk_absmax:.6f} 发生在:")
        print(f"Batch={b.item()}, SeqIdx={s.item()}, Head={h.item()}, Dim={d.item()}")
        print(f"对应的 K 维 Block ID (SeqIdx // 64) = {s.item() // 64}")

    return out_absmax, dq_absmax, dk_absmax, dv_absmax


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
            prev_peak = max(prev_item.out_absmax, prev_item.dq_absmax, prev_item.dk_absmax, prev_item.dv_absmax)
            curr_peak = max(curr_item.out_absmax, curr_item.dq_absmax, curr_item.dk_absmax, curr_item.dv_absmax)
            jump = curr_peak > max(prev_peak * 4, 1.0)
            state_flip = prev_item.ok != curr_item.ok
            if jump or state_flip:
                messages.append(
                    f"dtype={key[0]}, causal={key[1]}, D={key[2]}: "
                    f"S {prev_item.seqlen}->{curr_item.seqlen}, "
                    f"peak {prev_peak:.4f}->{curr_peak:.4f}, "
                    f"ok {prev_item.ok}->{curr_item.ok}"
                )
    return messages


def main():
    parser = argparse.ArgumentParser(description="CK-only 边界探针（不跑 reference，适合 rocprof）")
    parser.add_argument("--tag", type=str, default="run_noref")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--dims", type=str, default="64")
    parser.add_argument("--dtypes", type=str, default="fp16")
    parser.add_argument("--causal", type=str, default="false", choices=["both", "true", "false"])
    parser.add_argument("--seqlens", type=str, default="")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--peak-threshold", type=float, default=100.0)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无可用 HIP/CUDA 设备")

    print("=" * 100)
    print("CK-only 边界探针")
    print(f"tag={args.tag}")
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
                    out_vals = []
                    dq_vals = []
                    dk_vals = []
                    dv_vals = []
                    for repeat_idx in range(args.repeats):
                        seed = args.base_seed + repeat_idx
                        torch.cuda.synchronize()
                        out_absmax, dq_absmax, dk_absmax, dv_absmax = run_case(
                            args.batch, seqlen, args.nheads, headdim, dtype, causal, seed
                        )
                        out_vals.append(out_absmax)
                        dq_vals.append(dq_absmax)
                        dk_vals.append(dk_absmax)
                        dv_vals.append(dv_absmax)

                    out_mean = sum(out_vals) / len(out_vals)
                    dq_mean = sum(dq_vals) / len(dq_vals)
                    dk_mean = sum(dk_vals) / len(dk_vals)
                    dv_mean = sum(dv_vals) / len(dv_vals)
                    finite = all(is_finite_number(v) for v in [out_mean, dq_mean, dk_mean, dv_mean])
                    peak = max(out_mean, dq_mean, dk_mean, dv_mean)
                    ok = finite and peak < args.peak_threshold

                    row = Row(
                        tag=args.tag,
                        seqlen=seqlen,
                        headdim=headdim,
                        dtype=dtype_name,
                        causal=causal,
                        out_absmax=out_mean,
                        dq_absmax=dq_mean,
                        dk_absmax=dk_mean,
                        dv_absmax=dv_mean,
                        ok=ok,
                    )
                    rows.append(row)
                    total += 1

                    icon = "OK" if ok else "BAD"
                    print(
                        f"[{icon}] dtype={dtype_name:<4} causal={str(causal):<5} D={headdim:<3} S={seqlen:<4} "
                        f"out={out_mean:.6f} dq={dq_mean:.6f} dk={dk_mean:.6f} dv={dv_mean:.6f} "
                        f"bucket64={(seqlen + 63)//64:<3} mod64={seqlen % 64:<2}"
                    )

    bad = [row for row in rows if not row.ok]
    print("-" * 100)
    print(f"总 case: {total}, 失败: {len(bad)}, 通过: {total - len(bad)}")

    if bad:
        print("\nTop 20 失败样本（按 dk_absmax 降序）:")
        for row in sorted(bad, key=lambda item: item.dk_absmax, reverse=True)[:20]:
            print(
                f"dtype={row.dtype}, causal={row.causal}, D={row.headdim}, S={row.seqlen}, "
                f"out={row.out_absmax:.6f}, dq={row.dq_absmax:.6f}, "
                f"dk={row.dk_absmax:.6f}, dv={row.dv_absmax:.6f}, "
                f"bucket64={(row.seqlen + 63)//64}, mod64={row.seqlen % 64}"
            )

    transitions = detect_transitions(rows)
    print("\n疑似跳点:")
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
                "out_absmax", "dq_absmax", "dk_absmax", "dv_absmax", "ok", "bucket64", "mod64"
            ])
            for row in rows:
                writer.writerow([
                    row.tag,
                    row.seqlen,
                    row.headdim,
                    row.dtype,
                    int(row.causal),
                    f"{row.out_absmax:.8f}",
                    f"{row.dq_absmax:.8f}",
                    f"{row.dk_absmax:.8f}",
                    f"{row.dv_absmax:.8f}",
                    int(row.ok),
                    (row.seqlen + 63) // 64,
                    row.seqlen % 64,
                ])
        print(f"\n已写出 CSV: {args.csv}")


if __name__ == "__main__":
    main()
