import argparse
import csv
import pathlib
import sys
import statistics

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from test_ck_tile_probe import run_case


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid bool: {value}")


def dtype_from_name(name: str) -> torch.dtype:
    name = name.strip().lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def stat(values):
    if len(values) <= 1:
        return values[0], values[0], values[0], 0.0
    return min(values), max(values), statistics.mean(values), statistics.pstdev(values)


def main():
    parser = argparse.ArgumentParser(description="Per-seed accuracy repeat stats for CK probe")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--causal", type=parse_bool, required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--same-seed", action="store_true")
    parser.add_argument("--clip", type=float, default=0.0)
    parser.add_argument("--grad-scale", type=float, default=100.0)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    dtype = dtype_from_name(args.dtype)
    rows = []
    for repeat_idx in range(args.repeats):
        seed = args.base_seed if args.same_seed else args.base_seed + repeat_idx
        torch.cuda.synchronize()
        (
            fwd,
            bwd,
            dq,
            dk,
            dv,
            bwd_scaled,
            dq_scaled,
            dk_scaled,
            dv_scaled,
            bwd_rel,
            dq_rel,
            dk_rel,
            dv_rel,
        ) = run_case(
            args.batch,
            args.seqlen,
            args.nheads,
            args.dim,
            dtype,
            args.causal,
            seed,
            args.clip,
            args.grad_scale,
        )
        row = {
            "seed": seed,
            "fwd": fwd,
            "bwd": bwd,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "bwd_scaled": bwd_scaled,
            "dq_scaled": dq_scaled,
            "dk_scaled": dk_scaled,
            "dv_scaled": dv_scaled,
            "bwd_rel": bwd_rel,
            "dq_rel": dq_rel,
            "dk_rel": dk_rel,
            "dv_rel": dv_rel,
        }
        rows.append(row)
        print(
            f"seed={seed:<4} fwd={fwd:.6f} bwd={bwd:.6f} "
            f"dq={dq:.6f} dk={dk:.6f} dv={dv:.6f} "
            f"bwd/scale={bwd_scaled:.6f} bwd/rel={bwd_rel:.6f}"
        )

    print("-" * 100)
    for key in ["bwd_scaled", "dq_scaled", "dk_scaled", "dv_scaled", "bwd_rel", "dq_rel", "dk_rel", "dv_rel"]:
        values = [row[key] for row in rows]
        min_v, max_v, mean_v, std_v = stat(values)
        print(f"{key:<12} min={min_v:.8f} max={max_v:.8f} mean={mean_v:.8f} std={std_v:.8f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
