from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_attn.flash_attn_interface import flash_attn_gpu, maybe_contiguous


def _pad_last_dim(x: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, int]:
    head_dim = x.shape[-1]
    pad = (multiple - head_dim % multiple) % multiple
    return (F.pad(x, [0, pad]), head_dim) if pad else (x, head_dim)


def _materialize(x: torch.Tensor) -> torch.Tensor:
    src = maybe_contiguous(x.detach())
    dst = torch.empty(src.shape, dtype=src.dtype, device=src.device)
    dst.copy_(src)
    return dst


def _assert_has_storage(name: str, x: torch.Tensor) -> None:
    try:
        _ = x.data_ptr()
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise RuntimeError(
            f"{name} has no storage: shape={tuple(x.shape)} dtype={x.dtype} "
            f"device={x.device} stride={tuple(x.stride())} type={type(x)}"
        ) from exc


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    dropout_p: float,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
) -> None:
    if dropout_p != 0.0:
        raise NotImplementedError("Chunked exact backward prototype requires dropout_p == 0.0")
    if window_size != (-1, -1):
        raise NotImplementedError("Chunked exact backward prototype only supports dense attention")
    if softcap != 0.0:
        raise NotImplementedError("Chunked exact backward prototype does not support softcap")
    if alibi_slopes is not None:
        raise NotImplementedError("Chunked exact backward prototype does not support ALiBi")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v with shape (B, S, H, D)")
    if k.shape[:2] != v.shape[:2] or k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have matching (B, S, H)")
    if causal and q.shape[1] != k.shape[1]:
        raise NotImplementedError("Current causal chunking prototype requires seqlen_q == seqlen_k")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")


def safe_rdna3_chunked_flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    _validate(q, k, v, chunk_size, dropout_p, causal, window_size, softcap, alibi_slopes)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    v = maybe_contiguous(v)

    q_pad, head_dim_og = _pad_last_dim(q)
    k_pad, _ = _pad_last_dim(k)
    v_pad, _ = _pad_last_dim(v)
    _assert_has_storage("q_pad", q_pad)
    _assert_has_storage("k_pad", k_pad)
    _assert_has_storage("v_pad", v_pad)

    out_pad, softmax_lse, _unused_s, _unused_rng = flash_attn_gpu.fwd(
        _materialize(q_pad),
        _materialize(k_pad),
        _materialize(v_pad),
        None,
        alibi_slopes,
        None,
        None,
        None,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        False,
        None,
    )
    _assert_has_storage("out_pad", out_pad)
    _assert_has_storage("softmax_lse", softmax_lse)
    return out_pad[..., :head_dim_og], _materialize(q_pad), _materialize(k_pad), _materialize(v_pad), _materialize(out_pad), _materialize(softmax_lse), head_dim_og


def safe_rdna3_chunked_flash_attn_fwd_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out, q_pad, k_pad, v_pad, out_pad, softmax_lse, head_dim_og = safe_rdna3_chunked_flash_attn_forward(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        chunk_size=chunk_size,
    )

    dout = _materialize(dout)
    if head_dim_og != q_pad.shape[-1]:
        dout = F.pad(dout, [0, q_pad.shape[-1] - head_dim_og])
    dout = _materialize(dout)

    seqlen_q = q_pad.shape[1]
    seqlen_k = k_pad.shape[1]

    dq = torch.zeros_like(q_pad)
    dk = torch.zeros_like(k_pad)
    dv = torch.zeros_like(v_pad)

    for q_start in range(0, seqlen_q, chunk_size):
        q_end = min(q_start + chunk_size, seqlen_q)
        q_blk = _materialize(q_pad[:, q_start:q_end])
        out_blk = _materialize(out_pad[:, q_start:q_end])
        dout_blk = _materialize(dout[:, q_start:q_end])
        lse_blk = _materialize(softmax_lse[:, :, q_start:q_end])

        for k_start in range(0, seqlen_k, chunk_size):
            k_end = min(k_start + chunk_size, seqlen_k)

            if causal:
                if k_start >= q_end:
                    continue
                local_causal = q_start == k_start
            else:
                local_causal = False

            k_blk = _materialize(k_pad[:, k_start:k_end])
            v_blk = _materialize(v_pad[:, k_start:k_end])

            dq_blk = torch.empty_like(q_blk)
            dk_blk = torch.empty_like(k_blk)
            dv_blk = torch.empty_like(v_blk)

            _assert_has_storage("dout_blk", dout_blk)
            _assert_has_storage("q_blk", q_blk)
            _assert_has_storage("k_blk", k_blk)
            _assert_has_storage("v_blk", v_blk)
            _assert_has_storage("out_blk", out_blk)
            _assert_has_storage("lse_blk", lse_blk)
            _assert_has_storage("dq_blk", dq_blk)
            _assert_has_storage("dk_blk", dk_blk)
            _assert_has_storage("dv_blk", dv_blk)

            flash_attn_gpu.bwd(
                dout_blk,
                q_blk,
                k_blk,
                v_blk,
                out_blk,
                lse_blk,
                dq_blk,
                dk_blk,
                dv_blk,
                alibi_slopes,
                dropout_p,
                q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale,
                local_causal,
                window_size[0],
                window_size[1],
                softcap,
                deterministic,
                None,
                None,
            )

            dq[:, q_start:q_end].add_(dq_blk)
            dk[:, k_start:k_end].add_(dk_blk)
            dv[:, k_start:k_end].add_(dv_blk)

    return out, dq[..., :head_dim_og], dk[..., :head_dim_og], dv[..., :head_dim_og]
