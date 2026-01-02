# attention_triton.py
import math

import torch
import triton
import triton.language as tl


@triton.jit
def regular_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    # strides in elements
    stride_qz,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_km,
    stride_kd,
    stride_vz,
    stride_vm,
    stride_vd,
    stride_oz,
    stride_om,
    stride_od,
    L,  # runtime
    scale,  # runtime
    # compile-time
    D: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashAttention-style forward-only causal attention (no external mask),
    streaming over K/V blocks with online softmax.
    Each program: one (z, query-block).
      z indexes (batch*head).
    Shapes expected by wrapper: Q,K,V in [Z, L, D], output [Z, L, D].
    """

    pid_m = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    z = pid_z

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < L
    offs_m_i32 = offs_m.to(tl.int32)

    offs_d = tl.arange(0, D)

    # Online softmax stats per query row
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Output accumulator
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Stream over key blocks
    for n_start in range(0, L, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < L
        offs_n_i32 = offs_n.to(tl.int32)

        # Compute scores tile: (BLOCK_M, BLOCK_N)
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Dot(Q, K^T) in chunks of BLOCK_D
        for d0 in tl.static_range(0, D, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < D

            q_ptrs = (
                q_ptr
                + z * stride_qz
                + offs_m[:, None] * stride_qm
                + d_offsets[None, :] * stride_qd
            )
            k_ptrs = (
                k_ptr
                + z * stride_kz
                + offs_n[:, None] * stride_km
                + d_offsets[None, :] * stride_kd
            )

            q_block = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            k_block = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            scores += tl.dot(q_block, tl.trans(k_block))

        scores = scores * scale

        if causal:
            causal_mask = offs_n_i32[None, :] > offs_m_i32[:, None]
            scores = tl.where(causal_mask, -float("inf"), scores)

        valid = m_mask[:, None] & n_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))

        # Online softmax update
        block_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, block_max)

        scores_shifted = scores - m_i_new[:, None]
        p = tl.exp(scores_shifted)

        l_old = l_i * tl.exp(m_i - m_i_new)
        l_new = l_old + tl.sum(p, axis=1)

        l_new_safe = tl.where(l_new > 0, l_new, 1.0)
        coeff_old = l_old / l_new_safe

        # p @ V
        v_ptrs = (
            v_ptr
            + z * stride_vz
            + offs_n[None, :] * stride_vm
            + offs_d[:, None] * stride_vd
        )
        v_block = tl.load(
            v_ptrs,
            mask=n_mask[None, :] & (offs_d[:, None] < D),
            other=0.0,
        ).to(tl.float32)  # (D, BLOCK_N)

        o_block = tl.dot(p, tl.trans(v_block))  # (BLOCK_M, D)
        acc = coeff_old[:, None] * acc + o_block / l_new_safe[:, None]

        m_i = m_i_new
        l_i = l_new

    # Write output
    out_ptrs = (
        out_ptr
        + z * stride_oz
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & (offs_d[None, :] < D))


def regular_attention_triton(
    xq: torch.Tensor,  # (B, H, L, D)
    xk: torch.Tensor,
    xv: torch.Tensor,
    causal: bool = True,
    block_m: int = 32,
    block_n: int = 32,
    block_d: int = 32,
):
    """
    Fused regular causal attention forward using Triton (FlashAttention-style).
    No external mask. Assumes Lq = Lk = L and Q/K/V are same shapes.

    Returns: (B, H, L, D), same dtype as input.
    """
    assert xq.is_cuda and xk.is_cuda and xv.is_cuda
    assert xq.shape == xk.shape == xv.shape
    assert xq.dtype == xk.dtype == xv.dtype
    assert xq.dim() == 4

    B, H, L, D = xq.shape

    Z = B * H
    q = xq.contiguous().view(Z, L, D)
    k = xk.contiguous().view(Z, L, D)
    v = xv.contiguous().view(Z, L, D)

    out = torch.empty_like(q)

    stride_qz, stride_qm, stride_qd = q.stride()
    stride_kz, stride_km, stride_kd = k.stride()
    stride_vz, stride_vm, stride_vd = v.stride()
    stride_oz, stride_om, stride_od = out.stride()

    grid = (triton.cdiv(L, block_m), Z)
    scale = 1.0 / math.sqrt(D)

    regular_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        stride_qz,
        stride_qm,
        stride_qd,
        stride_kz,
        stride_km,
        stride_kd,
        stride_vz,
        stride_vm,
        stride_vd,
        stride_oz,
        stride_om,
        stride_od,
        L,
        scale,
        D=D,
        causal=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,  # safe default; you can tune later
    )

    return out.view(B, H, L, D)


@triton.jit
def tapa_fwd_kernel(
    q_amp_ptr,
    q_phase_ptr,  # [Z, L, Da], [Z, L, Dp]
    k_amp_ptr,
    k_phase_ptr,  # [Z, L, Da], [Z, L, Dp]
    v_ptr,  # [Z, L, D]
    out_ptr,  # [Z, L, D]
    dist_lut_ptr,  # [L] float (|d|^alpha)
    # strides (elements)
    stride_qz,
    stride_qm,
    stride_qda,
    stride_qdp,
    stride_kz,
    stride_km,
    stride_kda,
    stride_kdp,
    stride_vz,
    stride_vm,
    stride_vd,
    stride_oz,
    stride_om,
    stride_od,
    # runtime
    L,
    scale_amp,
    scale_phase,
    # compile-time
    D: tl.constexpr,
    Da: tl.constexpr,
    Dp: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Forward-only fused TAPA attention (lean variant):
      scores = (Q_amp K_amp^T / sqrt(Da)) * cos( (Q_phase K_phase^T / sqrt(Dp)) * |i-j|^alpha )

    Causal masking built-in. Assumes Lq = Lk = L.

    Note: This kernel is intentionally minimal and forward-only.
    """

    pid_m = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    z = pid_z

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < L
    offs_m_i32 = offs_m.to(tl.int32)

    offs_d = tl.arange(0, D)

    # Online softmax stats
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for n_start in range(0, L, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < L
        offs_n_i32 = offs_n.to(tl.int32)

        # ---- amplitude scores ----
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.static_range(0, Da, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < Da

            q_ptrs = (
                q_amp_ptr
                + z * stride_qz
                + offs_m[:, None] * stride_qm
                + d_offsets[None, :] * stride_qda
            )
            k_ptrs = (
                k_amp_ptr
                + z * stride_kz
                + offs_n[:, None] * stride_km
                + d_offsets[None, :] * stride_kda
            )

            q_block = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            k_block = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            scores += tl.dot(q_block, tl.trans(k_block))
        scores = scores * scale_amp

        # ---- phase dot ----
        phase_scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.static_range(0, Dp, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < Dp

            q_ptrs = (
                q_phase_ptr
                + z * stride_qz
                + offs_m[:, None] * stride_qm
                + d_offsets[None, :] * stride_qdp
            )
            k_ptrs = (
                k_phase_ptr
                + z * stride_kz
                + offs_n[:, None] * stride_km
                + d_offsets[None, :] * stride_kdp
            )

            q_block = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            k_block = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            phase_scores += tl.dot(q_block, tl.trans(k_block))
        phase_scores = phase_scores * scale_phase

        # ---- distance LUT: |i-j|^alpha ----
        diff = tl.abs(offs_m_i32[:, None] - offs_n_i32[None, :])
        diff = tl.minimum(diff, L - 1)
        dist = tl.load(dist_lut_ptr + diff)  # (BLOCK_M, BLOCK_N)

        phase_scores = phase_scores * dist
        phase_scores = tl.cos(phase_scores)

        scores = scores * phase_scores

        # masking
        if causal:
            causal_mask = offs_n_i32[None, :] > offs_m_i32[:, None]
            scores = tl.where(causal_mask, -float("inf"), scores)

        valid = m_mask[:, None] & n_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))

        # online softmax
        block_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, block_max)

        scores_shifted = scores - m_i_new[:, None]
        p = tl.exp(scores_shifted)

        l_old = l_i * tl.exp(m_i - m_i_new)
        l_new = l_old + tl.sum(p, axis=1)

        l_new_safe = tl.where(l_new > 0, l_new, 1.0)
        coeff_old = l_old / l_new_safe

        # p @ V
        v_ptrs = (
            v_ptr
            + z * stride_vz
            + offs_n[None, :] * stride_vm
            + offs_d[:, None] * stride_vd
        )
        v_block = tl.load(
            v_ptrs,
            mask=n_mask[None, :] & (offs_d[:, None] < D),
            other=0.0,
        ).to(tl.float32)

        o_block = tl.dot(p, tl.trans(v_block))
        acc = coeff_old[:, None] * acc + o_block / l_new_safe[:, None]

        m_i = m_i_new
        l_i = l_new

    # write output
    out_ptrs = (
        out_ptr
        + z * stride_oz
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & (offs_d[None, :] < D))


def tapa_attention_triton(
    xq: torch.Tensor,  # (B, H, L, D)
    xk: torch.Tensor,
    xv: torch.Tensor,
    theta: float,
    alpha: float,
    causal: bool = True,
    block_m: int = 32,
    block_n: int = 32,
    block_d: int = 32,
):
    """
    Fused TAPA forward using Triton kernel (forward-only).
    Splits head dim D into Da=int(theta*D) and Dp=D-Da.

    Returns: (B, H, L, D)
    """
    assert 0.0 < theta < 1.0
    assert alpha > 0.0
    assert xq.is_cuda and xk.is_cuda and xv.is_cuda
    assert xq.shape == xk.shape == xv.shape
    assert xq.dtype == xk.dtype == xv.dtype
    assert xq.dim() == 4

    B, H, L, D = xq.shape
    Z = B * H

    q = xq.contiguous().view(Z, L, D)
    k = xk.contiguous().view(Z, L, D)
    v = xv.contiguous().view(Z, L, D)

    Da = int(theta * D)
    Da = max(1, min(Da, D - 1))
    Dp = D - Da

    q_amp = q[..., :Da].contiguous()
    q_phase = q[..., Da:].contiguous()
    k_amp = k[..., :Da].contiguous()
    k_phase = k[..., Da:].contiguous()

    # dist_lut[d] = d^alpha (float), stored in fp32 then cast to input dtype for compute
    dist_lut = torch.arange(L, device=xq.device, dtype=torch.float32).pow(alpha)
    dist_lut = dist_lut.to(dtype=xq.dtype).contiguous()

    out = torch.empty_like(q)

    # strides
    stride_qz, stride_qm, stride_qda = q_amp.stride()
    stride_kz, stride_km, stride_kda = k_amp.stride()
    stride_vz, stride_vm, stride_vd = v.stride()
    stride_oz, stride_om, stride_od = out.stride()

    # For phase tensors, we only need the last stride
    _, _, stride_qdp = q_phase.stride()
    _, _, stride_kdp = k_phase.stride()

    grid = (triton.cdiv(L, block_m), Z)

    scale_amp = 1.0 / math.sqrt(Da)
    scale_phase = 1.0 / math.sqrt(Dp)

    tapa_fwd_kernel[grid](
        q_amp,
        q_phase,
        k_amp,
        k_phase,
        v,
        out,
        dist_lut,
        stride_qz,
        stride_qm,
        stride_qda,
        stride_qdp,
        stride_kz,
        stride_km,
        stride_kda,
        stride_kdp,
        stride_vz,
        stride_vm,
        stride_vd,
        stride_oz,
        stride_om,
        stride_od,
        L,
        scale_amp,
        scale_phase,
        D=D,
        Da=Da,
        Dp=Dp,
        causal=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )

    return out.view(B, H, L, D)
