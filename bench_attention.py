# bench_attention.py
import time
import torch

from attention_triton import regular_attention_triton, tapa_attention_triton


def time_fn(fn, iters=30, warmup=10):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / iters


@torch.no_grad()
def main():
    assert torch.cuda.is_available(), "CUDA required."
    torch.manual_seed(0)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Llama-like head geometry (common choice)
    B, H, D = 1, 32, 128
    theta = 0.5
    alpha = 0.5

    lengths = [2048, 4096, 8192, 16384, 32768]  # add 65536 if you can

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"dtype={dtype}, B={B}, H={H}, D={D}, theta={theta}, alpha={alpha}\n")
    print("L\tbaseline(ms)\tTAPA(ms)\tTAPA/baseline")

    for L in lengths:
        xq = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        xk = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        xv = torch.randn(B, H, L, D, device="cuda", dtype=dtype)

        f_base = lambda: regular_attention_triton(xq, xk, xv, causal=True)
        f_tapa = lambda: tapa_attention_triton(xq, xk, xv, theta=theta, alpha=alpha, causal=True)

        tb = time_fn(f_base)
        tt = time_fn(f_tapa)

        print(f"{L}\t{tb*1e3:.3f}\t\t{tt*1e3:.3f}\t\t{tt/tb:.3f}")


if __name__ == "__main__":
    main()
