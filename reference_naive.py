# reference_naive.py
import math
import torch


def naive_regular_attention(xq, xk, xv, causal: bool = True):
    """
    Reference regular attention (materializes scores).
    xq,xk,xv: (B,H,L,D)
    """
    B, H, L, D = xq.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale  # (B,H,L,L)

    if causal:
        i = torch.arange(L, device=xq.device)
        j = torch.arange(L, device=xq.device)
        causal_mask = j[None, :] > i[:, None]  # (L,L)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, xv)
    return out


def naive_tapa_attention(xq, xk, xv, theta: float, alpha: float, causal: bool = True):
    """
    Reference TAPA/CAP attention for correctness only (materializes scores):
      scores = (qA kA^T / sqrt(Da)) * cos( (qP kP^T / sqrt(Dp)) * |i-j|^alpha )
    """
    assert 0.0 < theta < 1.0
    assert alpha > 0.0

    B, H, L, D = xq.shape
    Da = int(theta * D)
    Da = max(1, min(Da, D - 1))
    Dp = D - Da

    qA, qP = xq[..., :Da], xq[..., Da:]
    kA, kP = xk[..., :Da], xk[..., Da:]

    amp = torch.matmul(qA, kA.transpose(-2, -1)) / math.sqrt(Da)  # (B,H,L,L)
    ph = torch.matmul(qP, kP.transpose(-2, -1)) / math.sqrt(Dp)   # (B,H,L,L)

    # dist = |i-j|^alpha
    idx = torch.arange(L, device=xq.device)
    dist = (idx[:, None] - idx[None, :]).abs().float().pow(alpha)  # (L,L)
    dist = dist.to(dtype=ph.dtype)

    scores = amp * torch.cos(ph * dist)

    if causal:
        causal_mask = idx[None, :] > idx[:, None]
        scores = scores.masked_fill(causal_mask, float("-inf"))

    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, xv)
    return out
