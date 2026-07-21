#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path


HOST_RE = re.compile(
    r"\[DK_HOST_GLOBAL\] absmax=([-0-9.]+) sum=([-0-9.]+) "
    r"argmax=\(b=(\d+),s=(\d+),h=(\d+),d=(\d+)\) v=([-0-9.]+)"
)
RESULT_RE = re.compile(r"\[(OK|BAD)\].*\bS=(\d+)\b.*\bdk=([0-9.]+)")
GROUND_RE = re.compile(r"\[💥 GROUND ZERO\] dk_err=([-0-9.]+)")
SGRADT_CTX_RE = re.compile(
    r"\[SGRADT_CTX\] cand=([AB]) b=\((\d+),(\d+),(\d+)\) use_lds_remap=(\d+)"
)
SGRADT_PRE_RE = re.compile(
    r"\[SGRADT_PRE\] cand=([AB]) b=\((\d+),(\d+),(\d+)\) "
    r"t=(\d+) q=(\d+) k=(\d+) v=([-0-9.]+)"
)
SGRADT_POST_RE = re.compile(
    r"\[SGRADT_POST\] cand=([AB]) b=\((\d+),(\d+),(\d+)\) "
    r"t=(\d+) k=(\d+) q=(\d+) v=([-0-9.]+)"
)
DK_RETURN_T0_RE = re.compile(
    r"\[DK_RETURN_T0\] b=\((\d+),(\d+),(\d+)\) sz=(\d+) absmax=([-0-9.]+) "
    r"s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
QSHUF_INIT_T0_RE = re.compile(
    r"\[QSHUF_INIT_T0\] b=\((\d+),(\d+),(\d+)\) sz=(\d+) absmax=([-0-9.]+) "
    r"s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
QBLK_HOT_T0_RE = re.compile(
    r"\[QBLK_HOT_T0\] b=\((\d+),(\d+),(\d+)\) sz=(\d+) absmax=([-0-9.]+) "
    r"s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
QTREG_HOT_T0_RE = re.compile(
    r"\[QTREG_HOT_T0\] b=\((\d+),(\d+),(\d+)\) sz=(\d+) absmax=([-0-9.]+) "
    r"s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
QSHUF_HOT_T0_RE = re.compile(
    r"\[QSHUF_HOT_T0\] b=\((\d+),(\d+),(\d+)\) sz=(\d+) absmax=([-0-9.]+) "
    r"s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
QBLK_TAIL_T0_RE = re.compile(
    r"\[QBLK_TAIL_T0\] b=\((\d+),(\d+),(\d+)\) sz=(\d+) absmax=([-0-9.]+) "
    r"s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
QTREG_TAIL_T0_RE = re.compile(
    r"\[QTREG_TAIL_T0\] b=\((\d+),(\d+),(\d+)\) sz=(\d+) absmax=([-0-9.]+) "
    r"s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
DK_ROW_SRC_T0_RE = re.compile(
    r"\[DK_ROW_SRC_T0\] b=\((\d+),(\d+),(\d+)\) count=(\d+) absmax=([-0-9.]+) sum=([-0-9.]+) "
    r"sqsum=([-0-9.]+) s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
DK_ROW_DSTR_T0_RE = re.compile(
    r"\[DK_ROW_DSTR_T0\] b=\((\d+),(\d+),(\d+)\) count=(\d+) sum_y=([-0-9]+) xor_y=([0-9a-fA-F]+) "
    r"y0=([-0-9]+) y1=([-0-9]+) y2=([-0-9]+) y3=([-0-9]+)"
)
DK_ROW_DIRECT_T0_RE = re.compile(
    r"\[DK_ROW_DIRECT_T0\] b=\((\d+),(\d+),(\d+)\) count=(\d+) absmax=([-0-9.]+) sum=([-0-9.]+) "
    r"sqsum=([-0-9.]+) s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
DK_ROW_GLOB_POST_T0_RE = re.compile(
    r"\[DK_ROW_GLOB_POST_T0\] b=\((\d+),(\d+),(\d+)\) count=(\d+) absmax=([-0-9.]+) sum=([-0-9.]+) "
    r"sqsum=([-0-9.]+) s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)
DK_ROW_SLOT_T0_RE = re.compile(
    r"\[DK_ROW_SLOT_T0\] b=\((\d+),(\d+),(\d+)\) mask=([0-9a-fA-F]+) "
    r"y0=([-0-9]+) v0=([-0-9.]+) y1=([-0-9]+) v1=([-0-9.]+) "
    r"y2=([-0-9]+) v2=([-0-9.]+) y3=([-0-9]+) v3=([-0-9.]+)"
)
DK_ROW_GLOB_CMP_T0_RE = re.compile(
    r"\[DK_ROW_GLOB_CMP_T0\] b=\((\d+),(\d+),(\d+)\) "
    r"y0=([-0-9]+) pre0=([-0-9.]+) post0=([-0-9.]+) d0=([-0-9.]+) "
    r"y1=([-0-9]+) pre1=([-0-9.]+) post1=([-0-9.]+) d1=([-0-9.]+) "
    r"y2=([-0-9]+) pre2=([-0-9.]+) post2=([-0-9.]+) d2=([-0-9.]+) "
    r"y3=([-0-9]+) pre3=([-0-9.]+) post3=([-0-9.]+) d3=([-0-9.]+)"
)
DK_ROW_YVAL_T0_RE = re.compile(
    r"\[DK_ROW_YVAL_T0\] b=\((\d+),(\d+),(\d+)\) part=(\d+) "
    r"y0=([-0-9]+) v0=([-0-9.]+) y1=([-0-9]+) v1=([-0-9.]+) "
    r"y2=([-0-9]+) v2=([-0-9.]+) y3=([-0-9]+) v3=([-0-9.]+) "
    r"y4=([-0-9]+) v4=([-0-9.]+) y5=([-0-9]+) v5=([-0-9.]+) "
    r"y6=([-0-9]+) v6=([-0-9.]+) y7=([-0-9]+) v7=([-0-9.]+)"
)
DK_ROW_PATH_T0_RE = re.compile(
    r"\[DK_ROW_PATH_T0\] b=\((\d+),(\d+),(\d+)\) part=(\d+) "
    r"y0=([-0-9]+) m0=([-0-9]+) n0=([-0-9]+) ib0=([-0-9]+) "
    r"y1=([-0-9]+) m1=([-0-9]+) n1=([-0-9]+) ib1=([-0-9]+) "
    r"y2=([-0-9]+) m2=([-0-9]+) n2=([-0-9]+) ib2=([-0-9]+) "
    r"y3=([-0-9]+) m3=([-0-9]+) n3=([-0-9]+) ib3=([-0-9]+) "
    r"y4=([-0-9]+) m4=([-0-9]+) n4=([-0-9]+) ib4=([-0-9]+) "
    r"y5=([-0-9]+) m5=([-0-9]+) n5=([-0-9]+) ib5=([-0-9]+) "
    r"y6=([-0-9]+) m6=([-0-9]+) n6=([-0-9]+) ib6=([-0-9]+) "
    r"y7=([-0-9]+) m7=([-0-9]+) n7=([-0-9]+) ib7=([-0-9]+)"
)
DK_ROW_DI_T0_RE = re.compile(
    r"\[DK_ROW_DI_T0\] b=\((\d+),(\d+),(\d+)\) part=(\d+) vals=([-0-9,;]+)"
)
DK_EPI_CAST_T0_RE = re.compile(
    r"\[DK_EPI_CAST_T0\] b=\((\d+),(\d+),(\d+)\) sz=(\d+) absmax=([-0-9.]+) "
    r"s0=([-0-9.]+) s1=([-0-9.]+) s2=([-0-9.]+) s3=([-0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CK BWD debug log")
    parser.add_argument("log", type=Path, help="Path to run_dbg*.log")
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Only print digest summaries, skip per-section details",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=None,
        help="Only print summaries for one sequence length",
    )
    parser.add_argument(
        "--row-detail",
        type=int,
        default=None,
        metavar="S",
        help="Print representative DK_ROW_DSTR_T0 and DK_ROW_YVAL_T0 details for sequence length S",
    )
    parser.add_argument(
        "--row-compare-a",
        type=int,
        default=None,
        metavar="S",
        help="Compare row-store probes for the first section with sequence length S",
    )
    parser.add_argument(
        "--row-compare-b",
        type=int,
        default=None,
        metavar="S",
        help="Compare row-store probes for the second section with sequence length S",
    )
    parser.add_argument(
        "--tail-detail",
        type=int,
        default=None,
        metavar="S",
        help="Print targeted tail-row probe details for sequence length S",
    )
    return parser.parse_args()


def split_host_sections(lines: list[str]) -> list[tuple[int, int, int, list[str]]]:
    host_positions = [i for i, line in enumerate(lines) if HOST_RE.search(line)]
    sections = []
    prev_end = 0
    for host_idx in host_positions:
        end = host_idx + 1
        sections.append((prev_end, end, host_idx, lines[prev_end:end]))
        prev_end = end
    return sections


def collect_results(lines: list[str]) -> tuple[list[tuple[str, int, str, int]], dict[int, str]]:
    results: list[tuple[str, int, str, int]] = []
    ground_by_line: dict[int, str] = {}
    last_ground = None
    for i, line in enumerate(lines, 1):
        ground_match = GROUND_RE.search(line)
        if ground_match:
            last_ground = ground_match.group(1)
        result_match = RESULT_RE.search(line)
        if result_match:
            status, seqlen, dk = result_match.groups()
            results.append((status, int(seqlen), dk, i))
            if last_ground is not None:
                ground_by_line[i] = last_ground
                last_ground = None
    return results, ground_by_line


def summarize_sgradt(section: list[str]) -> list[str]:
    ctx = {}
    pre: dict[str, Counter[tuple[int, int, float]]] = defaultdict(Counter)
    post: dict[str, Counter[tuple[int, int, float]]] = defaultdict(Counter)

    for line in section:
        match = SGRADT_CTX_RE.search(line)
        if match:
            cand, _, _, _, use_lds_remap = match.groups()
            ctx[cand] = int(use_lds_remap)
            continue
        match = SGRADT_PRE_RE.search(line)
        if match:
            cand, _, _, _, _, q, k, value = match.groups()
            pre[cand][(int(q), int(k), round(float(value), 6))] += 1
            continue
        match = SGRADT_POST_RE.search(line)
        if match:
            cand, _, _, _, _, k, q, value = match.groups()
            post[cand][(int(q), int(k), round(float(value), 6))] += 1

    if not (ctx or pre or post):
        return ["  SGRADT: none"]

    out = []
    for cand in sorted(set(ctx) | set(pre) | set(post)):
        pre_set = set(pre[cand])
        post_set = set(post[cand])
        out.append(
            "  cand {}: use_lds_remap={} set_equal={} uniq_pre={} uniq_post={}".format(
                cand,
                ctx.get(cand, "?"),
                pre_set == post_set,
                len(pre_set),
                len(post_set),
            )
        )
        if pre_set:
            by_q: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for (q, _k, value), count in pre[cand].items():
                by_q[q].append((count, value))
            q_modes = []
            for q in sorted(by_q):
                value = max(by_q[q])[1]
                q_modes.append(f"q{q}={value:.6f}")
            out.append(f"    pre_mode: {' '.join(q_modes)}")
    return out


def collect_dk_return_t0(section: list[str]) -> dict[str, object] | None:
    fingerprints: list[tuple[float, float, float, float, float]] = []
    sizes: Counter[int] = Counter()
    absmax_values: list[float] = []
    for line in section:
        match = DK_RETURN_T0_RE.search(line)
        if not match:
            continue
        _bx, _by, _bz, size, absmax, s0, s1, s2, s3 = match.groups()
        fp = tuple(round(float(v), 6) for v in (absmax, s0, s1, s2, s3))
        fingerprints.append(fp)
        sizes[int(size)] += 1
        absmax_values.append(fp[0])

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]:.6f},{fp[1]:.6f},{fp[2]:.6f},{fp[3]:.6f},{fp[4]:.6f}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "sizes": dict(sorted(sizes.items())),
        "absmax_min": min(absmax_values),
        "absmax_max": max(absmax_values),
        "top": counts.most_common(6),
    }


def summarize_dk_return_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_dk_return_t0(section)
    if summary is None:
        return [], None

    out = [
        "  DK_RETURN_T0 blocks={} uniq={} digest={} absmax_range=[{:.6f},{:.6f}] sz={}".format(
            summary["blocks"],
            summary["uniq"],
            summary["digest"],
            summary["absmax_min"],
            summary["absmax_max"],
            ",".join(f"{size}x{count}" for size, count in summary["sizes"].items()),
        )
    ]
    for fp, count in summary["top"]:
        out.append(
            "    {}x absmax={:.6f} s0={:.6f} s1={:.6f} s2={:.6f} s3={:.6f}".format(
                count, fp[0], fp[1], fp[2], fp[3], fp[4]
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_basic_tile_t0(
    section: list[str], pattern: re.Pattern[str]
) -> dict[str, object] | None:
    fingerprints: list[tuple[float, float, float, float, float]] = []
    sizes: Counter[int] = Counter()
    absmax_values: list[float] = []
    for line in section:
        match = pattern.search(line)
        if not match:
            continue
        _bx, _by, _bz, size, absmax, s0, s1, s2, s3 = match.groups()
        fp = tuple(round(float(v), 6) for v in (absmax, s0, s1, s2, s3))
        fingerprints.append(fp)
        sizes[int(size)] += 1
        absmax_values.append(fp[0])

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]:.6f},{fp[1]:.6f},{fp[2]:.6f},{fp[3]:.6f},{fp[4]:.6f}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "sizes": dict(sorted(sizes.items())),
        "absmax_min": min(absmax_values),
        "absmax_max": max(absmax_values),
        "top": counts.most_common(6),
    }


def summarize_basic_tile_t0(
    section: list[str], pattern: re.Pattern[str], tag: str
) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_basic_tile_t0(section, pattern)
    if summary is None:
        return [], None

    out = [
        "  {} blocks={} uniq={} digest={} absmax_range=[{:.6f},{:.6f}] sz={}".format(
            tag,
            summary["blocks"],
            summary["uniq"],
            summary["digest"],
            summary["absmax_min"],
            summary["absmax_max"],
            ",".join(f"{size}x{count}" for size, count in summary["sizes"].items()),
        )
    ]
    for fp, count in summary["top"]:
        out.append(
            "    {}x absmax={:.6f} s0={:.6f} s1={:.6f} s2={:.6f} s3={:.6f}".format(
                count, fp[0], fp[1], fp[2], fp[3], fp[4]
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_dk_epi_cast_t0(section: list[str]) -> dict[str, object] | None:
    fingerprints: list[tuple[float, float, float, float, float]] = []
    sizes: Counter[int] = Counter()
    absmax_values: list[float] = []
    for line in section:
        match = DK_EPI_CAST_T0_RE.search(line)
        if not match:
            continue
        _bx, _by, _bz, size, absmax, s0, s1, s2, s3 = match.groups()
        fp = tuple(round(float(v), 6) for v in (absmax, s0, s1, s2, s3))
        fingerprints.append(fp)
        sizes[int(size)] += 1
        absmax_values.append(fp[0])

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]:.6f},{fp[1]:.6f},{fp[2]:.6f},{fp[3]:.6f},{fp[4]:.6f}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "sizes": dict(sorted(sizes.items())),
        "absmax_min": min(absmax_values),
        "absmax_max": max(absmax_values),
        "top": counts.most_common(6),
    }


def summarize_dk_epi_cast_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_dk_epi_cast_t0(section)
    if summary is None:
        return [], None

    out = [
        "  DK_EPI_CAST_T0 blocks={} uniq={} digest={} absmax_range=[{:.6f},{:.6f}] sz={}".format(
            summary["blocks"],
            summary["uniq"],
            summary["digest"],
            summary["absmax_min"],
            summary["absmax_max"],
            ",".join(f"{size}x{count}" for size, count in summary["sizes"].items()),
        )
    ]
    for fp, count in summary["top"]:
        out.append(
            "    {}x absmax={:.6f} s0={:.6f} s1={:.6f} s2={:.6f} s3={:.6f}".format(
                count, fp[0], fp[1], fp[2], fp[3], fp[4]
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_window_sig(section: list[str], regex: re.Pattern[str]) -> dict[str, object] | None:
    fingerprints: list[tuple[int, int, int, int, float, float, float, float, float, float, float]] = []
    absmax_values: list[float] = []
    for line in section:
        match = regex.search(line)
        if not match:
            continue
        (_bx, _by, _bz, row0, col0, valid_rows, valid_cols, absmax, total_sum,
         sqsum, s0, s1, s2, s3) = match.groups()
        fp = (
            int(row0),
            int(col0),
            int(valid_rows),
            int(valid_cols),
            round(float(absmax), 6),
            round(float(total_sum), 6),
            round(float(sqsum), 6),
            round(float(s0), 6),
            round(float(s1), 6),
            round(float(s2), 6),
            round(float(s3), 6),
        )
        fingerprints.append(fp)
        absmax_values.append(fp[4])

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]},{fp[1]},{fp[2]},{fp[3]},{fp[4]:.6f},{fp[5]:.6f},{fp[6]:.6f},{fp[7]:.6f},{fp[8]:.6f},{fp[9]:.6f},{fp[10]:.6f}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "absmax_min": min(absmax_values),
        "absmax_max": max(absmax_values),
        "top": counts.most_common(6),
    }


def summarize_window_sig(section: list[str], regex: re.Pattern[str], label: str) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_window_sig(section, regex)
    if summary is None:
        return [], None

    out = [
        f"  {label} blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']} "
        f"absmax_range=[{summary['absmax_min']:.6f},{summary['absmax_max']:.6f}]"
    ]
    for fp, count in summary["top"]:
        out.append(
            "    {}x origin=({}, {}) valid=({}, {}) absmax={:.6f} sum={:.6f} sqsum={:.6f} "
            "s0={:.6f} s1={:.6f} s2={:.6f} s3={:.6f}".format(
                count, fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7], fp[8], fp[9], fp[10]
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_row_src_t0(section: list[str]) -> dict[str, object] | None:
    fingerprints: list[tuple[int, float, float, float, float, float, float, float]] = []
    for line in section:
        match = DK_ROW_SRC_T0_RE.search(line)
        if not match:
            continue
        (_bx, _by, _bz, count, absmax, total_sum, sqsum, s0, s1, s2, s3) = match.groups()
        fp = (
            int(count),
            round(float(absmax), 6),
            round(float(total_sum), 6),
            round(float(sqsum), 6),
            round(float(s0), 6),
            round(float(s1), 6),
            round(float(s2), 6),
            round(float(s3), 6),
        )
        fingerprints.append(fp)

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]},{fp[1]:.6f},{fp[2]:.6f},{fp[3]:.6f},{fp[4]:.6f},{fp[5]:.6f},{fp[6]:.6f},{fp[7]:.6f}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "top": counts.most_common(6),
    }


def summarize_row_src_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_src_t0(section)
    if summary is None:
        return [], None

    out = [f"  DK_ROW_SRC_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']}"]
    for fp, count in summary["top"]:
        out.append(
            "    {}x count={} absmax={:.6f} sum={:.6f} sqsum={:.6f} s0={:.6f} s1={:.6f} s2={:.6f} s3={:.6f}".format(
                count, fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7]
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_row_dstr_t0(section: list[str]) -> dict[str, object] | None:
    fingerprints: list[tuple[int, int, str, int, int, int, int]] = []
    for line in section:
        match = DK_ROW_DSTR_T0_RE.search(line)
        if not match:
            continue
        (_bx, _by, _bz, count, sum_y, xor_y, y0, y1, y2, y3) = match.groups()
        fp = (
            int(count),
            int(sum_y),
            xor_y.lower(),
            int(y0),
            int(y1),
            int(y2),
            int(y3),
        )
        fingerprints.append(fp)

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]},{fp[1]},{fp[2]},{fp[3]},{fp[4]},{fp[5]},{fp[6]}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "top": counts.most_common(6),
    }


def summarize_row_dstr_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_dstr_t0(section)
    if summary is None:
        return [], None

    out = [f"  DK_ROW_DSTR_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']}"]
    for fp, count in summary["top"]:
        out.append(
            "    {}x count={} sum_y={} xor_y={} y0={} y1={} y2={} y3={}".format(
                count, fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6]
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_row_direct_t0(section: list[str]) -> dict[str, object] | None:
    fingerprints: list[tuple[int, float, float, float, float, float, float, float]] = []
    for line in section:
        match = DK_ROW_DIRECT_T0_RE.search(line)
        if not match:
            continue
        (_bx, _by, _bz, count, absmax, total_sum, sqsum, s0, s1, s2, s3) = match.groups()
        fp = (
            int(count),
            round(float(absmax), 6),
            round(float(total_sum), 6),
            round(float(sqsum), 6),
            round(float(s0), 6),
            round(float(s1), 6),
            round(float(s2), 6),
            round(float(s3), 6),
        )
        fingerprints.append(fp)

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]},{fp[1]:.6f},{fp[2]:.6f},{fp[3]:.6f},{fp[4]:.6f},{fp[5]:.6f},{fp[6]:.6f},{fp[7]:.6f}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "top": counts.most_common(6),
    }


def summarize_row_direct_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_direct_t0(section)
    if summary is None:
        return [], None

    out = [f"  DK_ROW_DIRECT_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']}"]
    for fp, count in summary["top"]:
        out.append(
            "    {}x count={} absmax={:.6f} sum={:.6f} sqsum={:.6f} s0={:.6f} s1={:.6f} s2={:.6f} s3={:.6f}".format(
                count, fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7]
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_row_glob_post_t0(section: list[str]) -> dict[str, object] | None:
    fingerprints: list[tuple[int, float, float, float, float, float, float, float]] = []
    for line in section:
        match = DK_ROW_GLOB_POST_T0_RE.search(line)
        if not match:
            continue
        (_bx, _by, _bz, count, absmax, total_sum, sqsum, s0, s1, s2, s3) = match.groups()
        fp = (
            int(count),
            round(float(absmax), 6),
            round(float(total_sum), 6),
            round(float(sqsum), 6),
            round(float(s0), 6),
            round(float(s1), 6),
            round(float(s2), 6),
            round(float(s3), 6),
        )
        fingerprints.append(fp)

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]},{fp[1]:.6f},{fp[2]:.6f},{fp[3]:.6f},{fp[4]:.6f},{fp[5]:.6f},{fp[6]:.6f},{fp[7]:.6f}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "top": counts.most_common(6),
    }


def summarize_row_glob_post_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_glob_post_t0(section)
    if summary is None:
        return [], None

    out = [f"  DK_ROW_GLOB_POST_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']}"]
    for fp, count in summary["top"]:
        out.append(
            "    {}x count={} absmax={:.6f} sum={:.6f} sqsum={:.6f} s0={:.6f} s1={:.6f} s2={:.6f} s3={:.6f}".format(
                count, fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7]
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_row_glob_cmp_t0(section: list[str]) -> dict[str, object] | None:
    fingerprints: list[
        tuple[int, float, float, float, int, float, float, float, int, float, float, float, int, float, float, float]
    ] = []
    max_abs_diff = 0.0
    nonzero_diff = 0
    for line in section:
        match = DK_ROW_GLOB_CMP_T0_RE.search(line)
        if not match:
            continue
        (
            _bx,
            _by,
            _bz,
            y0,
            pre0,
            post0,
            d0,
            y1,
            pre1,
            post1,
            d1,
            y2,
            pre2,
            post2,
            d2,
            y3,
            pre3,
            post3,
            d3,
        ) = match.groups()
        fp = (
            int(y0),
            round(float(pre0), 6),
            round(float(post0), 6),
            round(float(d0), 6),
            int(y1),
            round(float(pre1), 6),
            round(float(post1), 6),
            round(float(d1), 6),
            int(y2),
            round(float(pre2), 6),
            round(float(post2), 6),
            round(float(d2), 6),
            int(y3),
            round(float(pre3), 6),
            round(float(post3), 6),
            round(float(d3), 6),
        )
        fingerprints.append(fp)
        for diff in (float(d0), float(d1), float(d2), float(d3)):
            abs_diff = abs(diff)
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff
            if abs_diff > 5e-7:
                nonzero_diff += 1

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]},{fp[1]:.6f},{fp[2]:.6f},{fp[3]:.6f},{fp[4]},{fp[5]:.6f},{fp[6]:.6f},{fp[7]:.6f},{fp[8]},{fp[9]:.6f},{fp[10]:.6f},{fp[11]:.6f},{fp[12]},{fp[13]:.6f},{fp[14]:.6f},{fp[15]:.6f}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "max_abs_diff": round(max_abs_diff, 6),
        "nonzero_diff": nonzero_diff,
        "top": counts.most_common(6),
    }


def summarize_row_glob_cmp_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_glob_cmp_t0(section)
    if summary is None:
        return [], None

    out = [
        f"  DK_ROW_GLOB_CMP_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']} "
        f"max_abs_diff={summary['max_abs_diff']:.6f} nonzero_diff={summary['nonzero_diff']}"
    ]
    for fp, count in summary["top"]:
        out.append(
            "    {}x y0={} pre0={:.6f} post0={:.6f} d0={:.6f} y1={} pre1={:.6f} post1={:.6f} d1={:.6f} y2={} pre2={:.6f} post2={:.6f} d2={:.6f} y3={} pre3={:.6f} post3={:.6f} d3={:.6f}".format(
                count,
                fp[0],
                fp[1],
                fp[2],
                fp[3],
                fp[4],
                fp[5],
                fp[6],
                fp[7],
                fp[8],
                fp[9],
                fp[10],
                fp[11],
                fp[12],
                fp[13],
                fp[14],
                fp[15],
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_row_slot_t0(section: list[str]) -> dict[str, object] | None:
    fingerprints: list[tuple[str, int, float, int, float, int, float, int, float]] = []
    for line in section:
        match = DK_ROW_SLOT_T0_RE.search(line)
        if not match:
            continue
        (_bx, _by, _bz, mask, y0, v0, y1, v1, y2, v2, y3, v3) = match.groups()
        fp = (
            mask.lower(),
            int(y0),
            round(float(v0), 6),
            int(y1),
            round(float(v1), 6),
            int(y2),
            round(float(v2), 6),
            int(y3),
            round(float(v3), 6),
        )
        fingerprints.append(fp)

    if not fingerprints:
        return None

    counts = Counter(fingerprints)
    canonical = "|".join(
        f"{count}:{fp[0]},{fp[1]},{fp[2]:.6f},{fp[3]},{fp[4]:.6f},{fp[5]},{fp[6]:.6f},{fp[7]},{fp[8]:.6f}"
        for fp, count in sorted(counts.items())
    )
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "top": counts.most_common(6),
    }


def summarize_row_slot_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_slot_t0(section)
    if summary is None:
        return [], None

    out = [f"  DK_ROW_SLOT_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']}"]
    for fp, count in summary["top"]:
        out.append(
            "    {}x mask={} y0={} v0={:.6f} y1={} v1={:.6f} y2={} v2={:.6f} y3={} v3={:.6f}".format(
                count, fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7], fp[8]
            )
        )
    if summary["uniq"] > 6:
        out.append(f"    ... {summary['uniq'] - 6} more signatures")
    return out, summary


def collect_row_yval_t0(section: list[str]) -> dict[str, object] | None:
    blocks: dict[tuple[int, int, int], dict[int, tuple[tuple[int, float], ...]]] = defaultdict(dict)
    for line in section:
        match = DK_ROW_YVAL_T0_RE.search(line)
        if not match:
            continue
        (
            bx,
            by,
            bz,
            part,
            y0,
            v0,
            y1,
            v1,
            y2,
            v2,
            y3,
            v3,
            y4,
            v4,
            y5,
            v5,
            y6,
            v6,
            y7,
            v7,
        ) = match.groups()
        pairs = (
            (int(y0), round(float(v0), 6)),
            (int(y1), round(float(v1), 6)),
            (int(y2), round(float(v2), 6)),
            (int(y3), round(float(v3), 6)),
            (int(y4), round(float(v4), 6)),
            (int(y5), round(float(v5), 6)),
            (int(y6), round(float(v6), 6)),
            (int(y7), round(float(v7), 6)),
        )
        blocks[(int(bx), int(by), int(bz))][int(part)] = pairs

    if not blocks:
        return None

    fingerprints: list[tuple[tuple[int, tuple[tuple[int, float], ...]], ...]] = []
    for key in sorted(blocks):
        part_map = blocks[key]
        fingerprint = tuple((part, part_map[part]) for part in sorted(part_map))
        fingerprints.append(fingerprint)

    counts = Counter(fingerprints)
    canonical_parts = []
    for fingerprint, count in sorted(counts.items()):
        part_text = []
        for part, pairs in fingerprint:
            pair_text = ",".join(f"{y}:{value:.6f}" for y, value in pairs)
            part_text.append(f"p{part}[{pair_text}]")
        canonical_parts.append(f"{count}:{'|'.join(part_text)}")
    canonical = "|".join(canonical_parts)
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "counts": counts,
        "top": counts.most_common(3),
    }


def summarize_row_yval_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_yval_t0(section)
    if summary is None:
        return [], None

    out = [f"  DK_ROW_YVAL_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']}"]
    for fingerprint, count in summary["top"]:
        parts = []
        for part, pairs in fingerprint:
            pair_text = " ".join(f"y{y}={value:.6f}" for y, value in pairs)
            parts.append(f"part={part} {pair_text}")
        out.append(f"    {count}x " + " | ".join(parts))
    if summary["uniq"] > 3:
        out.append(f"    ... {summary['uniq'] - 3} more signatures")
    return out, summary


def collect_row_yseq_t0(section: list[str]) -> dict[str, object] | None:
    blocks: dict[tuple[int, int, int], dict[int, tuple[int, ...]]] = defaultdict(dict)
    for line in section:
        match = DK_ROW_YVAL_T0_RE.search(line)
        if not match:
            continue
        (
            bx,
            by,
            bz,
            part,
            y0,
            _v0,
            y1,
            _v1,
            y2,
            _v2,
            y3,
            _v3,
            y4,
            _v4,
            y5,
            _v5,
            y6,
            _v6,
            y7,
            _v7,
        ) = match.groups()
        ys = (
            int(y0),
            int(y1),
            int(y2),
            int(y3),
            int(y4),
            int(y5),
            int(y6),
            int(y7),
        )
        blocks[(int(bx), int(by), int(bz))][int(part)] = ys

    if not blocks:
        return None

    fingerprints: list[tuple[tuple[int, tuple[int, ...]], ...]] = []
    for key in sorted(blocks):
        part_map = blocks[key]
        fingerprint = tuple((part, part_map[part]) for part in sorted(part_map))
        fingerprints.append(fingerprint)

    counts = Counter(fingerprints)
    canonical_parts = []
    for fingerprint, count in sorted(counts.items()):
        part_text = []
        for part, ys in fingerprint:
            y_text = ",".join(str(y) for y in ys)
            part_text.append(f"p{part}[{y_text}]")
        canonical_parts.append(f"{count}:{'|'.join(part_text)}")
    canonical = "|".join(canonical_parts)
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "counts": counts,
        "top": counts.most_common(3),
    }


def summarize_row_yseq_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_yseq_t0(section)
    if summary is None:
        return [], None

    out = [f"  DK_ROW_YSEQ_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']}"]
    for fingerprint, count in summary["top"]:
        parts = []
        for part, ys in fingerprint:
            y_text = " ".join(f"y{i}={y}" for i, y in enumerate(ys, start=part * 8))
            parts.append(f"part={part} {y_text}")
        out.append(f"    {count}x " + " | ".join(parts))
    if summary["uniq"] > 3:
        out.append(f"    ... {summary['uniq'] - 3} more signatures")
    return out, summary


def collect_row_path_t0(section: list[str]) -> dict[str, object] | None:
    blocks: dict[tuple[int, int, int], dict[int, tuple[tuple[int, int, int, int], ...]]] = defaultdict(dict)
    for line in section:
        match = DK_ROW_PATH_T0_RE.search(line)
        if not match:
            continue
        (
            bx,
            by,
            bz,
            part,
            y0,
            m0,
            n0,
            ib0,
            y1,
            m1,
            n1,
            ib1,
            y2,
            m2,
            n2,
            ib2,
            y3,
            m3,
            n3,
            ib3,
            y4,
            m4,
            n4,
            ib4,
            y5,
            m5,
            n5,
            ib5,
            y6,
            m6,
            n6,
            ib6,
            y7,
            m7,
            n7,
            ib7,
        ) = match.groups()
        entries = (
            (int(y0), int(m0), int(n0), int(ib0)),
            (int(y1), int(m1), int(n1), int(ib1)),
            (int(y2), int(m2), int(n2), int(ib2)),
            (int(y3), int(m3), int(n3), int(ib3)),
            (int(y4), int(m4), int(n4), int(ib4)),
            (int(y5), int(m5), int(n5), int(ib5)),
            (int(y6), int(m6), int(n6), int(ib6)),
            (int(y7), int(m7), int(n7), int(ib7)),
        )
        blocks[(int(bx), int(by), int(bz))][int(part)] = entries

    if not blocks:
        return None

    fingerprints: list[tuple[tuple[int, tuple[tuple[int, int, int, int], ...]], ...]] = []
    for key in sorted(blocks):
        part_map = blocks[key]
        fingerprint = tuple((part, part_map[part]) for part in sorted(part_map))
        fingerprints.append(fingerprint)

    counts = Counter(fingerprints)
    canonical_parts = []
    for fingerprint, count in sorted(counts.items()):
        part_text = []
        for part, entries in fingerprint:
            entry_text = ",".join(f"{y}:{m}:{n}:{ib}" for y, m, n, ib in entries)
            part_text.append(f"p{part}[{entry_text}]")
        canonical_parts.append(f"{count}:{'|'.join(part_text)}")
    canonical = "|".join(canonical_parts)
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "counts": counts,
        "top": counts.most_common(3),
    }


def summarize_row_path_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_path_t0(section)
    if summary is None:
        return [], None

    out = [f"  DK_ROW_PATH_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']}"]
    for fingerprint, count in summary["top"]:
        parts = []
        for part, entries in fingerprint:
            entry_text = " ".join(
                f"s{i}=y{y}/m{m}/n{n}/ib{ib}" for i, (y, m, n, ib) in enumerate(entries, start=part * 8)
            )
            parts.append(f"part={part} {entry_text}")
        out.append(f"    {count}x " + " | ".join(parts))
    if summary["uniq"] > 3:
        out.append(f"    ... {summary['uniq'] - 3} more signatures")
    return out, summary


def collect_row_di_t0(section: list[str]) -> dict[str, object] | None:
    blocks: dict[tuple[int, int, int], dict[int, tuple[tuple[int, int, int, int], ...]]] = defaultdict(dict)
    for line in section:
        match = DK_ROW_DI_T0_RE.search(line)
        if not match:
            continue
        bx, by, bz, part, vals = match.groups()
        entries = []
        for item in vals.split(";"):
            if not item:
                continue
            d0, d10, d11, d12 = (int(x) for x in item.split(","))
            entries.append((d0, d10, d11, d12))
        blocks[(int(bx), int(by), int(bz))][int(part)] = tuple(entries)

    if not blocks:
        return None

    fingerprints: list[tuple[tuple[int, tuple[tuple[int, int, int, int], ...]], ...]] = []
    for key in sorted(blocks):
        part_map = blocks[key]
        fingerprint = tuple((part, part_map[part]) for part in sorted(part_map))
        fingerprints.append(fingerprint)

    counts = Counter(fingerprints)
    canonical_parts = []
    for fingerprint, count in sorted(counts.items()):
        part_text = []
        for part, entries in fingerprint:
            entry_text = ",".join(f"{d0}:{d10}:{d11}:{d12}" for d0, d10, d11, d12 in entries)
            part_text.append(f"p{part}[{entry_text}]")
        canonical_parts.append(f"{count}:{'|'.join(part_text)}")
    canonical = "|".join(canonical_parts)
    return {
        "blocks": len(fingerprints),
        "uniq": len(counts),
        "digest": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12],
        "counts": counts,
        "top": counts.most_common(3),
    }


def summarize_row_di_t0(section: list[str]) -> tuple[list[str], dict[str, object] | None]:
    summary = collect_row_di_t0(section)
    if summary is None:
        return [], None

    out = [f"  DK_ROW_DI_T0 blocks={summary['blocks']} uniq={summary['uniq']} digest={summary['digest']}"]
    for fingerprint, count in summary["top"]:
        parts = []
        for part, entries in fingerprint:
            entry_text = " ".join(
                f"s{i}=({d0},{d10},{d11},{d12})" for i, (d0, d10, d11, d12) in enumerate(entries, start=part * 8)
            )
            parts.append(f"part={part} {entry_text}")
        out.append(f"    {count}x " + " | ".join(parts))
    if summary["uniq"] > 3:
        out.append(f"    ... {summary['uniq'] - 3} more signatures")
    return out, summary


def print_rollup(
    title: str,
    rollup: list[tuple[int, int | None, str, str]],
    seqlen_filter: int | None,
) -> None:
    if not rollup:
        return
    grouped: dict[tuple[int | None, str], list[tuple[int, str]]] = defaultdict(list)
    for idx, seqlen, result_status, digest in rollup:
        if seqlen_filter is not None and seqlen != seqlen_filter:
            continue
        grouped[(seqlen, result_status)].append((idx, digest))
    if not grouped:
        return

    print(title)
    for (seqlen, result_status), items in sorted(
        grouped.items(), key=lambda x: (x[0][0] is None, x[0][0], x[0][1])
    ):
        digest_counts = Counter(digest for _idx, digest in items)
        digest_text = " ".join(f"{digest}x{count}" for digest, count in digest_counts.items())
        section_text = ",".join(str(idx) for idx, _digest in items)
        print(f"  S={seqlen} RESULT={result_status} sections={section_text} digests={digest_text}")


def flatten_row_yval_fingerprint(
    fingerprint: tuple[tuple[int, tuple[tuple[int, float], ...]], ...]
) -> list[tuple[int, float]]:
    flat: list[tuple[int, float]] = []
    for _part, pairs in sorted(fingerprint):
        flat.extend(pairs)
    return flat


def flatten_row_yseq_fingerprint(
    fingerprint: tuple[tuple[int, tuple[int, ...]], ...]
) -> list[int]:
    flat: list[int] = []
    for _part, ys in sorted(fingerprint):
        flat.extend(ys)
    return flat


def flatten_row_path_fingerprint(
    fingerprint: tuple[tuple[int, tuple[tuple[int, int, int, int], ...]], ...]
) -> list[tuple[int, int, int, int]]:
    flat: list[tuple[int, int, int, int]] = []
    for _part, entries in sorted(fingerprint):
        flat.extend(entries)
    return flat


def flatten_row_di_fingerprint(
    fingerprint: tuple[tuple[int, tuple[tuple[int, int, int, int], ...]], ...]
) -> list[tuple[int, int, int, int]]:
    flat: list[tuple[int, int, int, int]] = []
    for _part, entries in sorted(fingerprint):
        flat.extend(entries)
    return flat


def print_row_yval_fingerprint(
    label: str,
    fingerprint: tuple[tuple[int, tuple[tuple[int, float], ...]], ...],
) -> None:
    print(label)
    for part, pairs in sorted(fingerprint):
        pair_text = " ".join(f"y{y}={value:.6f}" for y, value in pairs)
        print(f"  part={part} {pair_text}")


def print_row_yseq_fingerprint(
    label: str,
    fingerprint: tuple[tuple[int, tuple[int, ...]], ...],
) -> None:
    print(label)
    for part, ys in sorted(fingerprint):
        y_text = " ".join(f"y{i}={y}" for i, y in enumerate(ys, start=part * 8))
        print(f"  part={part} {y_text}")


def print_row_path_fingerprint(
    label: str,
    fingerprint: tuple[tuple[int, tuple[tuple[int, int, int, int], ...]], ...],
) -> None:
    print(label)
    for part, entries in sorted(fingerprint):
        entry_text = " ".join(
            f"s{i}=y{y}/m{m}/n{n}/ib{ib}" for i, (y, m, n, ib) in enumerate(entries, start=part * 8)
        )
        print(f"  part={part} {entry_text}")


def print_row_di_fingerprint(
    label: str,
    fingerprint: tuple[tuple[int, tuple[tuple[int, int, int, int], ...]], ...],
) -> None:
    print(label)
    for part, entries in sorted(fingerprint):
        entry_text = " ".join(
            f"s{i}=({d0},{d10},{d11},{d12})" for i, (d0, d10, d11, d12) in enumerate(entries, start=part * 8)
        )
        print(f"  part={part} {entry_text}")


def print_row_detail(
    target_seqlen: int,
    section_records: list[tuple[int, int | None, str, list[str]]],
) -> None:
    print(f"ROW detail S={target_seqlen}:")
    chosen: dict[str, tuple[int, dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]] = {}
    for desired_status in ("OK", "BAD"):
        for idx, seqlen, status, section in section_records:
            if seqlen == target_seqlen and status == desired_status:
                dstr_summary = collect_row_dstr_t0(section)
                yval_summary = collect_row_yval_t0(section)
                yseq_summary = collect_row_yseq_t0(section)
                path_summary = collect_row_path_t0(section)
                di_summary = collect_row_di_t0(section)
                if dstr_summary is not None and yval_summary is not None and yseq_summary is not None and path_summary is not None and di_summary is not None:
                    chosen[desired_status] = (idx, dstr_summary, yval_summary, yseq_summary, path_summary, di_summary)
                    break

    missing = [status for status in ("OK", "BAD") if status not in chosen]
    if missing:
        print(f"  missing statuses: {', '.join(missing)}")
        return

    for status in ("OK", "BAD"):
        idx, dstr_summary, yval_summary, yseq_summary, path_summary, di_summary = chosen[status]
        print(
            f"  {status} section={idx} "
            f"DSTR={dstr_summary['digest']} YSEQ={yseq_summary['digest']} YVAL={yval_summary['digest']} PATH={path_summary['digest']} DI={di_summary['digest']}"
        )
        dstr_fp, dstr_count = dstr_summary["top"][0]
        print(
            "    DSTR top {}x count={} sum_y={} xor_y={} y0={} y1={} y2={} y3={}".format(
                dstr_count, dstr_fp[0], dstr_fp[1], dstr_fp[2], dstr_fp[3], dstr_fp[4], dstr_fp[5], dstr_fp[6]
            )
        )
        yseq_fp, yseq_count = yseq_summary["top"][0]
        print(f"    YSEQ top {yseq_count}x")
        print_row_yseq_fingerprint("    y-sequence:", yseq_fp)
        yval_fp, yval_count = yval_summary["top"][0]
        print(f"    YVAL top {yval_count}x")
        print_row_yval_fingerprint("    representative:", yval_fp)
        path_fp, path_count = path_summary["top"][0]
        print(f"    PATH top {path_count}x")
        print_row_path_fingerprint("    path:", path_fp)
        di_fp, di_count = di_summary["top"][0]
        print(f"    DI top {di_count}x")
        print_row_di_fingerprint("    distributed-index:", di_fp)

    ok_yseq_fp = chosen["OK"][3]["top"][0][0]
    bad_yseq_fp = chosen["BAD"][3]["top"][0][0]
    ok_ys = flatten_row_yseq_fingerprint(ok_yseq_fp)
    bad_ys = flatten_row_yseq_fingerprint(bad_yseq_fp)
    first_y_index = None
    for i, (ok_y, bad_y) in enumerate(zip(ok_ys, bad_ys)):
        if ok_y != bad_y:
            first_y_index = i
            print(f"  first y-linear diff at slot {i}: OK y={ok_y} BAD y={bad_y}")
            break
    if first_y_index is None:
        print("  y-linear list: no difference in representative top fingerprint")

    ok_yval_fp = chosen["OK"][2]["top"][0][0]
    bad_yval_fp = chosen["BAD"][2]["top"][0][0]
    ok_flat = flatten_row_yval_fingerprint(ok_yval_fp)
    bad_flat = flatten_row_yval_fingerprint(bad_yval_fp)

    first_pair_index = None
    for i, (ok_pair, bad_pair) in enumerate(zip(ok_flat, bad_flat)):
        if ok_pair != bad_pair:
            first_pair_index = i
            print(
                "  first (y,value) diff at slot {}: OK (y={}, v={:.6f}) BAD (y={}, v={:.6f})".format(
                    i, ok_pair[0], ok_pair[1], bad_pair[0], bad_pair[1]
                )
            )
            break
    if first_pair_index is None:
        print("  y/value list: no difference in representative top fingerprint")

    ok_path_fp = chosen["OK"][4]["top"][0][0]
    bad_path_fp = chosen["BAD"][4]["top"][0][0]
    ok_path = flatten_row_path_fingerprint(ok_path_fp)
    bad_path = flatten_row_path_fingerprint(bad_path_fp)
    first_path_index = None
    for i, (ok_entry, bad_entry) in enumerate(zip(ok_path, bad_path)):
        if ok_entry != bad_entry:
            first_path_index = i
            print(
                "  first path diff at slot {}: OK (y={}, m={}, n={}, ib={}) BAD (y={}, m={}, n={}, ib={})".format(
                    i,
                    ok_entry[0],
                    ok_entry[1],
                    ok_entry[2],
                    ok_entry[3],
                    bad_entry[0],
                    bad_entry[1],
                    bad_entry[2],
                    bad_entry[3],
                )
            )
            break
    if first_path_index is None:
        print("  path list: no difference in representative top fingerprint")

    ok_di_fp = chosen["OK"][5]["top"][0][0]
    bad_di_fp = chosen["BAD"][5]["top"][0][0]
    ok_di = flatten_row_di_fingerprint(ok_di_fp)
    bad_di = flatten_row_di_fingerprint(bad_di_fp)
    first_di_index = None
    for i, (ok_entry, bad_entry) in enumerate(zip(ok_di, bad_di)):
        if ok_entry != bad_entry:
            first_di_index = i
            print(f"  first distributed-index diff at slot {i}: OK {ok_entry} BAD {bad_entry}")
            break
    if first_di_index is None:
        print("  distributed-index list: no difference in representative top fingerprint")

    print(f"ROW YSEQ modes S={target_seqlen}:")
    for status in ("OK", "BAD"):
        mode_counts: Counter[tuple[tuple[int, tuple[int, ...]], ...]] = Counter()
        for _idx, seqlen, section_status, section in section_records:
            if seqlen != target_seqlen or section_status != status:
                continue
            summary = collect_row_yseq_t0(section)
            if summary is None:
                continue
            for fingerprint, count in summary["counts"].items():
                mode_counts[fingerprint] += count
        print(f"  {status}:")
        if not mode_counts:
            print("    none")
            continue
        for fingerprint, count in mode_counts.most_common():
            print(f"    {count}x")
            for part, ys in sorted(fingerprint):
                y_text = " ".join(f"y{i}={y}" for i, y in enumerate(ys, start=part * 8))
                print(f"      part={part} {y_text}")


def get_section_summaries(section: list[str]) -> dict[str, dict[str, object] | None]:
    return {
        "glob_post": collect_row_glob_post_t0(section),
        "glob_cmp": collect_row_glob_cmp_t0(section),
        "dstr": collect_row_dstr_t0(section),
        "yseq": collect_row_yseq_t0(section),
        "yval": collect_row_yval_t0(section),
        "path": collect_row_path_t0(section),
        "di": collect_row_di_t0(section),
        "qshuf_init": collect_basic_tile_t0(section, QSHUF_INIT_T0_RE),
        "qblk_hot": collect_basic_tile_t0(section, QBLK_HOT_T0_RE),
        "qtreg_hot": collect_basic_tile_t0(section, QTREG_HOT_T0_RE),
        "qshuf_hot": collect_basic_tile_t0(section, QSHUF_HOT_T0_RE),
        "qblk_tail": collect_basic_tile_t0(section, QBLK_TAIL_T0_RE),
        "qtreg_tail": collect_basic_tile_t0(section, QTREG_TAIL_T0_RE),
    }


def pick_first_section(
    target_seqlen: int,
    section_records: list[tuple[int, int | None, str, list[str]]],
) -> tuple[int, int | None, str, list[str]] | None:
    for record in section_records:
        idx, seqlen, _status, _section = record
        if seqlen == target_seqlen:
            return record
    return None


def print_row_compare(
    seqlen_a: int,
    seqlen_b: int,
    section_records: list[tuple[int, int | None, str, list[str]]],
) -> None:
    rec_a = pick_first_section(seqlen_a, section_records)
    rec_b = pick_first_section(seqlen_b, section_records)
    print(f"ROW compare S={seqlen_a} vs S={seqlen_b}:")
    if rec_a is None or rec_b is None:
        missing = []
        if rec_a is None:
            missing.append(str(seqlen_a))
        if rec_b is None:
            missing.append(str(seqlen_b))
        print(f"  missing sequence lengths: {', '.join(missing)}")
        return

    idx_a, _sa, status_a, section_a = rec_a
    idx_b, _sb, status_b, section_b = rec_b
    sums_a = get_section_summaries(section_a)
    sums_b = get_section_summaries(section_b)
    print(f"  A: section={idx_a} status={status_a}")
    print(f"  B: section={idx_b} status={status_b}")

    for key in (
        "glob_post",
        "glob_cmp",
        "dstr",
        "yseq",
        "yval",
        "path",
        "di",
        "qshuf_init",
        "qblk_hot",
        "qtreg_hot",
        "qshuf_hot",
        "qblk_tail",
        "qtreg_tail",
    ):
        summary_a = sums_a[key]
        summary_b = sums_b[key]
        if summary_a is None or summary_b is None:
            print(f"  {key.upper()}: missing")
            continue
        print(
            f"  {key.upper()}: A={summary_a['digest']} B={summary_b['digest']} "
            f"same={summary_a['digest'] == summary_b['digest']}"
        )

    if sums_a["yseq"] is not None and sums_b["yseq"] is not None:
        fp_a = sums_a["yseq"]["top"][0][0]
        fp_b = sums_b["yseq"]["top"][0][0]
        ys_a = flatten_row_yseq_fingerprint(fp_a)
        ys_b = flatten_row_yseq_fingerprint(fp_b)
        for i, (a, b) in enumerate(zip(ys_a, ys_b)):
            if a != b:
                print(f"  first YSEQ diff at slot {i}: A y={a} B y={b}")
                break
        else:
            print("  YSEQ top fingerprint: no slot diff")

    if sums_a["path"] is not None and sums_b["path"] is not None:
        fp_a = sums_a["path"]["top"][0][0]
        fp_b = sums_b["path"]["top"][0][0]
        path_a = flatten_row_path_fingerprint(fp_a)
        path_b = flatten_row_path_fingerprint(fp_b)
        for i, (a, b) in enumerate(zip(path_a, path_b)):
            if a != b:
                print(f"  first PATH diff at slot {i}: A {a} B {b}")
                break
        else:
            print("  PATH top fingerprint: no slot diff")

    if sums_a["di"] is not None and sums_b["di"] is not None:
        fp_a = sums_a["di"]["top"][0][0]
        fp_b = sums_b["di"]["top"][0][0]
        di_a = flatten_row_di_fingerprint(fp_a)
        di_b = flatten_row_di_fingerprint(fp_b)
        for i, (a, b) in enumerate(zip(di_a, di_b)):
            if a != b:
                print(f"  first DI diff at slot {i}: A {a} B {b}")
                break
        else:
            print("  DI top fingerprint: no slot diff")


def print_tail_detail(
    target_seqlen: int,
    section_records: list[tuple[int, int | None, str, list[str]]],
) -> None:
    print(f"TAIL detail S={target_seqlen}:")
    found = False
    for idx, seqlen, status, section in section_records:
        if seqlen != target_seqlen:
            continue
        found = True
        print(f"  section={idx} status={status}")
        print_tail_block_records(collect_tail_block_records(section))

    if not found:
        print("  missing sequence length")


def collect_tail_block_records(section: list[str]) -> dict[tuple[int, int, int], dict[str, list[object]]]:
    records: dict[tuple[int, int, int], dict[str, list[object]]] = defaultdict(lambda: defaultdict(list))
    for line in section:
        match = DK_ROW_SRC_T0_RE.search(line)
        if match:
            bx, by, bz, count, absmax, total_sum, sqsum, s0, s1, s2, s3 = match.groups()
            block = (int(bx), int(by), int(bz))
            records[block]["ROW_SRC"].append({
                "count": int(count),
                "absmax": float(absmax),
                "sum": float(total_sum),
                "sqsum": float(sqsum),
                "samples": [float(s0), float(s1), float(s2), float(s3)],
            })
            continue

        match = DK_ROW_DIRECT_T0_RE.search(line)
        if match:
            bx, by, bz, count, absmax, total_sum, sqsum, s0, s1, s2, s3 = match.groups()
            block = (int(bx), int(by), int(bz))
            records[block]["ROW_DIRECT"].append({
                "count": int(count),
                "absmax": float(absmax),
                "sum": float(total_sum),
                "sqsum": float(sqsum),
                "samples": [float(s0), float(s1), float(s2), float(s3)],
            })
            continue

        match = DK_ROW_GLOB_POST_T0_RE.search(line)
        if match:
            bx, by, bz, count, absmax, total_sum, sqsum, s0, s1, s2, s3 = match.groups()
            block = (int(bx), int(by), int(bz))
            records[block]["ROW_GLOB_POST"].append({
                "count": int(count),
                "absmax": float(absmax),
                "sum": float(total_sum),
                "sqsum": float(sqsum),
                "samples": [float(s0), float(s1), float(s2), float(s3)],
            })
            continue

        match = DK_ROW_GLOB_CMP_T0_RE.search(line)
        if match:
            (
                bx,
                by,
                bz,
                y0,
                pre0,
                post0,
                d0,
                y1,
                pre1,
                post1,
                d1,
                y2,
                pre2,
                post2,
                d2,
                y3,
                pre3,
                post3,
                d3,
            ) = match.groups()
            block = (int(bx), int(by), int(bz))
            diffs = [float(d0), float(d1), float(d2), float(d3)]
            records[block]["ROW_GLOB_CMP"].append({
                "slots": [
                    (int(y0), float(pre0), float(post0), float(d0)),
                    (int(y1), float(pre1), float(post1), float(d1)),
                    (int(y2), float(pre2), float(post2), float(d2)),
                    (int(y3), float(pre3), float(post3), float(d3)),
                ],
                "max_abs_diff": max(abs(x) for x in diffs),
                "nonzero_diff": sum(1 for x in diffs if abs(x) > 5e-7),
            })
            continue

        match = DK_ROW_DSTR_T0_RE.search(line)
        if match:
            bx, by, bz, count, sum_y, xor_y, y0, y1, y2, y3 = match.groups()
            block = (int(bx), int(by), int(bz))
            records[block]["ROW_DSTR"].append({
                "count": int(count),
                "sum_y": int(sum_y),
                "xor_y": xor_y.lower(),
                "slots": [int(y0), int(y1), int(y2), int(y3)],
            })
            continue

        match = DK_ROW_PATH_T0_RE.search(line)
        if match:
            (
                bx,
                by,
                bz,
                part,
                y0,
                m0,
                n0,
                ib0,
                y1,
                m1,
                n1,
                ib1,
                y2,
                m2,
                n2,
                ib2,
                y3,
                m3,
                n3,
                ib3,
                y4,
                m4,
                n4,
                ib4,
                y5,
                m5,
                n5,
                ib5,
                y6,
                m6,
                n6,
                ib6,
                y7,
                m7,
                n7,
                ib7,
            ) = match.groups()
            block = (int(bx), int(by), int(bz))
            records[block]["ROW_PATH"].append(
                (
                    int(part),
                    [
                        (int(y0), int(m0), int(n0), int(ib0)),
                        (int(y1), int(m1), int(n1), int(ib1)),
                        (int(y2), int(m2), int(n2), int(ib2)),
                        (int(y3), int(m3), int(n3), int(ib3)),
                        (int(y4), int(m4), int(n4), int(ib4)),
                        (int(y5), int(m5), int(n5), int(ib5)),
                        (int(y6), int(m6), int(n6), int(ib6)),
                        (int(y7), int(m7), int(n7), int(ib7)),
                    ],
                )
            )
            continue

        match = DK_ROW_DI_T0_RE.search(line)
        if match:
            bx, by, bz, part, vals = match.groups()
            block = (int(bx), int(by), int(bz))
            entries = []
            for item in vals.split(";"):
                if not item:
                    continue
                d0, d10, d11, d12 = (int(x) for x in item.split(","))
                entries.append((d0, d10, d11, d12))
            records[block]["ROW_DI"].append((int(part), entries))

    return records


def print_tail_block_records(records: dict[tuple[int, int, int], dict[str, list[object]]]) -> None:
    interesting = []
    for block, data in sorted(records.items()):
        hit = False
        for label in ("ROW_SRC", "ROW_DIRECT", "ROW_GLOB_POST", "ROW_DSTR"):
            for rec in data.get(label, []):
                if isinstance(rec, dict) and rec.get("count", 0) > 0:
                    hit = True
                    break
            if hit:
                break
        if hit or data.get("ROW_GLOB_CMP") or data.get("ROW_PATH") or data.get("ROW_DI"):
            interesting.append((block, data))

    if not interesting:
        print("    no nonzero-hit tail blocks")
        return

    for block, data in interesting:
        print(f"    block={block}")
        max_occ = max(
            len(data.get("ROW_SRC", [])),
            len(data.get("ROW_DIRECT", [])),
            len(data.get("ROW_GLOB_POST", [])),
            len(data.get("ROW_GLOB_CMP", [])),
            len(data.get("ROW_DSTR", [])),
            1,
        )
        for occ in range(max_occ):
            print(f"      occ={occ}")
            if occ < len(data.get("ROW_SRC", [])):
                rec = data["ROW_SRC"][occ]
                print(
                    "        ROW_SRC count={} absmax={:.6f} sum={:.6f} sqsum={:.6f} samples={}".format(
                        rec["count"], rec["absmax"], rec["sum"], rec["sqsum"], rec["samples"]
                    )
                )
            if occ < len(data.get("ROW_DIRECT", [])):
                rec = data["ROW_DIRECT"][occ]
                print(
                    "        ROW_DIRECT count={} absmax={:.6f} sum={:.6f} sqsum={:.6f} samples={}".format(
                        rec["count"], rec["absmax"], rec["sum"], rec["sqsum"], rec["samples"]
                    )
                )
            if occ < len(data.get("ROW_GLOB_POST", [])):
                rec = data["ROW_GLOB_POST"][occ]
                print(
                    "        ROW_GLOB_POST count={} absmax={:.6f} sum={:.6f} sqsum={:.6f} samples={}".format(
                        rec["count"], rec["absmax"], rec["sum"], rec["sqsum"], rec["samples"]
                    )
                )
            if occ < len(data.get("ROW_GLOB_CMP", [])):
                rec = data["ROW_GLOB_CMP"][occ]
                print(
                    "        ROW_GLOB_CMP max_abs_diff={:.6f} nonzero_diff={} slots={}".format(
                        rec["max_abs_diff"], rec["nonzero_diff"], rec["slots"]
                    )
                )
            if occ < len(data.get("ROW_DSTR", [])):
                rec = data["ROW_DSTR"][occ]
                print(
                    "        ROW_DSTR count={} sum_y={} xor_y={} slots={}".format(
                        rec["count"], rec["sum_y"], rec["xor_y"], rec["slots"]
                    )
                )
        if data.get("ROW_PATH"):
            path_parts = sorted(data["ROW_PATH"], key=lambda x: x[0])
            print(f"      ROW_PATH raw={path_parts}")
        if data.get("ROW_DI"):
            di_parts = sorted(data["ROW_DI"], key=lambda x: x[0])
            print(f"      ROW_DI raw={di_parts}")


def main() -> None:
    args = parse_args()
    lines = args.log.read_text(errors="replace").splitlines()
    sections = split_host_sections(lines)
    results, ground_by_line = collect_results(lines)
    section_records: list[tuple[int, int | None, str, list[str]]] = []
    t0_rollup: list[tuple[int, int | None, str, str]] = []
    epi_cast_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_src_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_dstr_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_direct_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_glob_post_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_glob_cmp_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_slot_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_yval_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_yseq_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_path_t0_rollup: list[tuple[int, int | None, str, str]] = []
    row_di_t0_rollup: list[tuple[int, int | None, str, str]] = []
    qshuf_init_t0_rollup: list[tuple[int, int | None, str, str]] = []
    qblk_hot_t0_rollup: list[tuple[int, int | None, str, str]] = []
    qtreg_hot_t0_rollup: list[tuple[int, int | None, str, str]] = []
    qshuf_hot_t0_rollup: list[tuple[int, int | None, str, str]] = []
    qblk_tail_t0_rollup: list[tuple[int, int | None, str, str]] = []
    qtreg_tail_t0_rollup: list[tuple[int, int | None, str, str]] = []

    if not args.brief:
        print(f"host_sections={len(sections)}")
    for idx, (_start, _end, host_idx, section) in enumerate(sections, 1):
        host_match = HOST_RE.search(lines[host_idx])
        assert host_match is not None
        absmax, total_sum, b, s, h, d, value = host_match.groups()
        result_status = "unmatched"
        result_line = "  RESULT: unmatched"
        result_seqlen: int | None = None
        if idx - 1 < len(results):
            status, result_seqlen, dk, line_no = results[idx - 1]
            ground = ground_by_line.get(line_no, "-")
            result_status = status
            result_line = f"  RESULT: {status} S={result_seqlen} dk={dk} ground={ground} line={line_no}"

        seqlen = result_seqlen
        section_records.append((idx, seqlen, result_status, section))

        if not args.brief:
            print(
                f"SECTION {idx}: S={seqlen} host_argmax=(b={b},s={s},h={h},d={d}) "
                f"absmax={absmax} sum={total_sum} v={value}"
            )
            if seqlen is not None and int(s) >= seqlen:
                print(f"  HOST_ARGMAX_OOB: seq_idx={s} >= seqlen={seqlen}")
            print(result_line)
            for line in summarize_sgradt(section):
                print(line)
        t0_lines, t0_summary = summarize_dk_return_t0(section)
        if not args.brief:
            for line in t0_lines:
                print(line)
        if t0_summary is not None:
            t0_rollup.append((idx, seqlen, result_status, str(t0_summary["digest"])))
        epi_cast_t0_lines, epi_cast_t0_summary = summarize_dk_epi_cast_t0(section)
        if not args.brief:
            for line in epi_cast_t0_lines:
                print(line)
        if epi_cast_t0_summary is not None:
            epi_cast_t0_rollup.append((idx, seqlen, result_status, str(epi_cast_t0_summary["digest"])))
        row_src_t0_lines, row_src_t0_summary = summarize_row_src_t0(section)
        if not args.brief:
            for line in row_src_t0_lines:
                print(line)
        if row_src_t0_summary is not None:
            row_src_t0_rollup.append((idx, seqlen, result_status, str(row_src_t0_summary["digest"])))
        row_direct_t0_lines, row_direct_t0_summary = summarize_row_direct_t0(section)
        if not args.brief:
            for line in row_direct_t0_lines:
                print(line)
        if row_direct_t0_summary is not None:
            row_direct_t0_rollup.append((idx, seqlen, result_status, str(row_direct_t0_summary["digest"])))
        row_glob_post_t0_lines, row_glob_post_t0_summary = summarize_row_glob_post_t0(section)
        if not args.brief:
            for line in row_glob_post_t0_lines:
                print(line)
        if row_glob_post_t0_summary is not None:
            row_glob_post_t0_rollup.append((idx, seqlen, result_status, str(row_glob_post_t0_summary["digest"])))
        row_glob_cmp_t0_lines, row_glob_cmp_t0_summary = summarize_row_glob_cmp_t0(section)
        if not args.brief:
            for line in row_glob_cmp_t0_lines:
                print(line)
        if row_glob_cmp_t0_summary is not None:
            row_glob_cmp_t0_rollup.append((idx, seqlen, result_status, str(row_glob_cmp_t0_summary["digest"])))
        row_slot_t0_lines, row_slot_t0_summary = summarize_row_slot_t0(section)
        if not args.brief:
            for line in row_slot_t0_lines:
                print(line)
        if row_slot_t0_summary is not None:
            row_slot_t0_rollup.append((idx, seqlen, result_status, str(row_slot_t0_summary["digest"])))
        row_yval_t0_lines, row_yval_t0_summary = summarize_row_yval_t0(section)
        if not args.brief:
            for line in row_yval_t0_lines:
                print(line)
        if row_yval_t0_summary is not None:
            row_yval_t0_rollup.append((idx, seqlen, result_status, str(row_yval_t0_summary["digest"])))
        row_yseq_t0_lines, row_yseq_t0_summary = summarize_row_yseq_t0(section)
        if not args.brief:
            for line in row_yseq_t0_lines:
                print(line)
        if row_yseq_t0_summary is not None:
            row_yseq_t0_rollup.append((idx, seqlen, result_status, str(row_yseq_t0_summary["digest"])))
        row_path_t0_lines, row_path_t0_summary = summarize_row_path_t0(section)
        if not args.brief:
            for line in row_path_t0_lines:
                print(line)
        if row_path_t0_summary is not None:
            row_path_t0_rollup.append((idx, seqlen, result_status, str(row_path_t0_summary["digest"])))
        row_di_t0_lines, row_di_t0_summary = summarize_row_di_t0(section)
        if not args.brief:
            for line in row_di_t0_lines:
                print(line)
        if row_di_t0_summary is not None:
            row_di_t0_rollup.append((idx, seqlen, result_status, str(row_di_t0_summary["digest"])))
        qshuf_init_t0_lines, qshuf_init_t0_summary = summarize_basic_tile_t0(section, QSHUF_INIT_T0_RE, "QSHUF_INIT_T0")
        if not args.brief:
            for line in qshuf_init_t0_lines:
                print(line)
        if qshuf_init_t0_summary is not None:
            qshuf_init_t0_rollup.append((idx, seqlen, result_status, str(qshuf_init_t0_summary["digest"])))
        qblk_hot_t0_lines, qblk_hot_t0_summary = summarize_basic_tile_t0(section, QBLK_HOT_T0_RE, "QBLK_HOT_T0")
        if not args.brief:
            for line in qblk_hot_t0_lines:
                print(line)
        if qblk_hot_t0_summary is not None:
            qblk_hot_t0_rollup.append((idx, seqlen, result_status, str(qblk_hot_t0_summary["digest"])))
        qtreg_hot_t0_lines, qtreg_hot_t0_summary = summarize_basic_tile_t0(section, QTREG_HOT_T0_RE, "QTREG_HOT_T0")
        if not args.brief:
            for line in qtreg_hot_t0_lines:
                print(line)
        if qtreg_hot_t0_summary is not None:
            qtreg_hot_t0_rollup.append((idx, seqlen, result_status, str(qtreg_hot_t0_summary["digest"])))
        qshuf_hot_t0_lines, qshuf_hot_t0_summary = summarize_basic_tile_t0(section, QSHUF_HOT_T0_RE, "QSHUF_HOT_T0")
        if not args.brief:
            for line in qshuf_hot_t0_lines:
                print(line)
        if qshuf_hot_t0_summary is not None:
            qshuf_hot_t0_rollup.append((idx, seqlen, result_status, str(qshuf_hot_t0_summary["digest"])))
        qblk_tail_t0_lines, qblk_tail_t0_summary = summarize_basic_tile_t0(section, QBLK_TAIL_T0_RE, "QBLK_TAIL_T0")
        if not args.brief:
            for line in qblk_tail_t0_lines:
                print(line)
        if qblk_tail_t0_summary is not None:
            qblk_tail_t0_rollup.append((idx, seqlen, result_status, str(qblk_tail_t0_summary["digest"])))
        qtreg_tail_t0_lines, qtreg_tail_t0_summary = summarize_basic_tile_t0(section, QTREG_TAIL_T0_RE, "QTREG_TAIL_T0")
        if not args.brief:
            for line in qtreg_tail_t0_lines:
                print(line)
        if qtreg_tail_t0_summary is not None:
            qtreg_tail_t0_rollup.append((idx, seqlen, result_status, str(qtreg_tail_t0_summary["digest"])))
        row_dstr_t0_lines, row_dstr_t0_summary = summarize_row_dstr_t0(section)
        if not args.brief:
            for line in row_dstr_t0_lines:
                print(line)
        if row_dstr_t0_summary is not None:
            row_dstr_t0_rollup.append((idx, seqlen, result_status, str(row_dstr_t0_summary["digest"])))
    print_rollup("DK_RETURN_T0 digest summary:", t0_rollup, args.seqlen)
    print_rollup("DK_EPI_CAST_T0 digest summary:", epi_cast_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_SRC_T0 digest summary:", row_src_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_DIRECT_T0 digest summary:", row_direct_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_GLOB_POST_T0 digest summary:", row_glob_post_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_GLOB_CMP_T0 digest summary:", row_glob_cmp_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_SLOT_T0 digest summary:", row_slot_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_YVAL_T0 digest summary:", row_yval_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_YSEQ_T0 digest summary:", row_yseq_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_PATH_T0 digest summary:", row_path_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_DI_T0 digest summary:", row_di_t0_rollup, args.seqlen)
    print_rollup("QSHUF_INIT_T0 digest summary:", qshuf_init_t0_rollup, args.seqlen)
    print_rollup("QBLK_HOT_T0 digest summary:", qblk_hot_t0_rollup, args.seqlen)
    print_rollup("QTREG_HOT_T0 digest summary:", qtreg_hot_t0_rollup, args.seqlen)
    print_rollup("QSHUF_HOT_T0 digest summary:", qshuf_hot_t0_rollup, args.seqlen)
    print_rollup("QBLK_TAIL_T0 digest summary:", qblk_tail_t0_rollup, args.seqlen)
    print_rollup("QTREG_TAIL_T0 digest summary:", qtreg_tail_t0_rollup, args.seqlen)
    print_rollup("DK_ROW_DSTR_T0 digest summary:", row_dstr_t0_rollup, args.seqlen)
    if args.row_detail is not None:
        print_row_detail(args.row_detail, section_records)
    if args.tail_detail is not None:
        print_tail_detail(args.tail_detail, section_records)
    if args.row_compare_a is not None and args.row_compare_b is not None:
        print_row_compare(args.row_compare_a, args.row_compare_b, section_records)


if __name__ == "__main__":
    main()
