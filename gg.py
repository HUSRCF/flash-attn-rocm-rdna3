import traceback
import os
import re

TAG = "O2T"

print(f"[{TAG}_BOOT] start")

try:
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    from test_ck_tile_probe import force_sdpa_math

    print(f"[{TAG}_BOOT] imports_ok")
    print(f"[{TAG}_ENV] torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
    print(f"[{TAG}_ENV] cuda_available={torch.cuda.is_available()}")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    B, S, H, D = 2, 64, 4, 64
    dtype = torch.float16
    causal = False

    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)

    out_fa = flash_attn_func(q, k, v, causal=causal)
    out_ref = force_sdpa_math(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=causal,
    ).transpose(1, 2)

    diff = (out_fa - out_ref).abs().float()
    mean_abs = diff.mean().item()
    max_abs = diff.max().item()
    idx = diff.reshape(-1).argmax().item()

    b = idx // (S * H * D)
    r = idx % (S * H * D)
    s = r // (H * D)
    r = r % (H * D)
    h = r // D
    d = r % D

    print(f"[{TAG}_META] B={B} S={S} H={H} D={D} dtype={dtype} causal={causal}")
    print(f"[{TAG}_STAT] mean_abs={mean_abs:.6f} max_abs={max_abs:.6f}")
    print(
        f"[{TAG}_MAX] b={b} s={s} h={h} d={d} "
        f"fa={out_fa[b, s, h, d].item():.6f} ref={out_ref[b, s, h, d].item():.6f}"
    )

    vals, inds = torch.topk(diff.reshape(-1), k=20)
    stride_s = int(out_fa.stride(1))
    for rank, (v_abs, i_) in enumerate(zip(vals.tolist(), inds.tolist()), 1):
        bb = i_ // (S * H * D)
        rr = i_ % (S * H * D)
        ss = rr // (H * D)
        rr = rr % (H * D)
        hh = rr // D
        dd = rr % D
        print(
            f"[{TAG}_TOP] rank={rank:02d} b={bb} s={ss} h={hh} d={dd} "
            f"abs={v_abs:.6f} fa={out_fa[bb, ss, hh, dd].item():.6f} "
            f"ref={out_ref[bb, ss, hh, dd].item():.6f}"
        )
        print(
            f"[{TAG}_TOP_OFF] rank={rank:02d} b={bb} h={hh} s={ss} d={dd} "
            f"off={ss * stride_s + dd}"
        )

    # 行重排启发式（固定 b=0,h=0）
    fa2 = out_fa[0, :, 0, :].float()
    ref2 = out_ref[0, :, 0, :].float()
    m = (fa2[:, None, :] - ref2[None, :, :]).abs().mean(dim=-1)
    perm_row = m.argmin(dim=1)
    id_hits = (perm_row == torch.arange(S, device=perm_row.device)).sum().item()
    print(f"[{TAG}_ROW] identity_hits={id_hits}/{S}")
    print(f"[{TAG}_ROW_PERM] first16={perm_row[:16].tolist()}")

    # 最大误差点行切片
    fa_row = out_fa[b, s, h, :].float()
    rf_row = out_ref[b, s, h, :].float()
    for i in range(16):
        print(
            f"[{TAG}_SLICE] idx={i:02d} fa={fa_row[i].item():.6f} "
            f"ref={rf_row[i].item():.6f} abs={(fa_row[i]-rf_row[i]).abs().item():.6f}"
        )

    # --- Off -> (s,d) 反查 + torch 对比 ---
    scan_log = os.environ.get("O2T_SCAN_LOG", "")
    target_b = int(os.environ.get("O2T_SCAN_B", "1"))
    target_h = int(os.environ.get("O2T_SCAN_H", "3"))
    scan_max = int(os.environ.get("O2T_SCAN_MAX", "200"))
    offs_env = os.environ.get("O2T_OFFS", "")

    print(
        f"[{TAG}_OFFCFG] log={scan_log or '-'} b={target_b} h={target_h} "
        f"stride_s={stride_s} max={scan_max} offs_env={'yes' if offs_env else 'no'}"
    )

    off_to_scan_v = {}
    if scan_log and os.path.exists(scan_log):
        pattern = re.compile(
            r"^\[O_OWNER_SCAN\] b=\((\d+),(\d+),(\d+)\) t=\d+ off=(\d+) v=([-+0-9.eE]+)"
        )
        with open(scan_log, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.match(line.strip())
                if not m:
                    continue
                hx = int(m.group(1))
                bz = int(m.group(3))
                off = int(m.group(4))
                scan_v = float(m.group(5))
                if bz == target_b and hx == target_h and off not in off_to_scan_v:
                    off_to_scan_v[off] = scan_v

    if offs_env:
        for tok in offs_env.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                off = int(tok)
                off_to_scan_v.setdefault(off, float("nan"))
            except ValueError:
                pass

    if off_to_scan_v:
        valid_rows = []
        invalid_offs = []
        for off, scan_v in off_to_scan_v.items():
            s2 = off // stride_s
            d2 = off % stride_s
            if 0 <= s2 < S and 0 <= d2 < D:
                fa_v = float(out_fa[target_b, s2, target_h, d2].item())
                rf_v = float(out_ref[target_b, s2, target_h, d2].item())
                valid_rows.append((off, s2, d2, scan_v, fa_v, rf_v, abs(fa_v - rf_v)))
            else:
                invalid_offs.append(off)

        valid_rows.sort(key=lambda x: x[6], reverse=True)
        print(f"[{TAG}_OFFN] total={len(off_to_scan_v)} valid={len(valid_rows)} invalid={len(invalid_offs)}")
        for rank, (off, s2, d2, scan_v, fa_v, rf_v, abs_fr) in enumerate(valid_rows[:scan_max], 1):
            scan_s = "nan" if scan_v != scan_v else f"{scan_v:.6f}"
            abs_sr = "nan" if scan_v != scan_v else f"{abs(scan_v - rf_v):.6f}"
            print(
                f"[{TAG}_OFFCMP] rank={rank:03d} b={target_b} h={target_h} "
                f"off={off} s={s2} d={d2} scan={scan_s} fa={fa_v:.6f} ref={rf_v:.6f} "
                f"abs_fa_ref={abs_fr:.6f} abs_scan_ref={abs_sr}"
            )
        if invalid_offs:
            print(f"[{TAG}_OFF_INVALID] first16={invalid_offs[:16]}")
    else:
        print(f"[{TAG}_OFFN] total=0 valid=0 invalid=0")

except Exception as e:
    print(f"[{TAG}_ERR] {type(e).__name__}: {e}")
    traceback.print_exc()
