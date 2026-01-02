# tests.py
import torch

from attention_triton import regular_attention_triton, tapa_attention_triton
from reference_naive import naive_regular_attention, naive_tapa_attention


@torch.no_grad()
def run_tests():
    assert torch.cuda.is_available(), "CUDA is required for Triton tests."

    torch.manual_seed(0)

    # Keep L small so naive reference (LxL) is feasible
    B, H, L, D = 2, 4, 128, 64
    dtype = torch.float16  # or torch.bfloat16 if your GPU supports it well

    xq = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
    xk = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
    xv = torch.randn(B, H, L, D, device="cuda", dtype=dtype)

    # --- Regular attention ---
    ref = naive_regular_attention(xq, xk, xv, causal=True)
    out = regular_attention_triton(xq, xk, xv, causal=True)

    max_err = (ref - out).abs().max().item()
    print(f"[regular] max abs err: {max_err:.6f}")
    # Triton uses fp32 accum; expect small fp16 error
    assert max_err < 5e-2, "Regular attention mismatch too large."

    # --- TAPA attention ---
    theta = 0.5
    alpha = 0.5

    ref = naive_tapa_attention(xq, xk, xv, theta=theta, alpha=alpha, causal=True)
    out = tapa_attention_triton(xq, xk, xv, theta=theta, alpha=alpha, causal=True)

    max_err = (ref - out).abs().max().item()
    print(f"[tapa]    max abs err: {max_err:.6f}")
    assert max_err < 8e-2, "TAPA attention mismatch too large."

    print("All tests passed")


if __name__ == "__main__":
    run_tests()
