# TAPA Attention

This repository provides a minimal, ready-to-run release of the **attention kernels** for:
- **Baseline causal attention** (FlashAttention-style online softmax; forward-only; Triton)
- **TAPA attention** (amplitude–phase split with phase cosine modulation; forward-only; Triton)

The implementation focuses on the kernel-level behavior:
the naive unfused implementation can materialize multiple score matrices, while a fused/streaming kernel
accumulates both dot products in a single streaming pass and avoids `L×L` materialization.

## Scope / Limitations
- **Forward-only** kernels (no backward kernel in this minimal release).
- **Causal attention only**, **no external mask**.
- Assumes `Lq = Lk = L` and input tensors are shaped `(B, H, L, D)`.

## Files
- `attention_triton.py`: Triton kernels + Python wrappers
- `reference_naive.py`: naive PyTorch reference (materializes scores; used for correctness)
- `tests.py`: correctness checks vs naive reference on small sequence lengths
- `bench_attention.py`: kernel-level benchmark on long sequences using random Q/K/V

## Install
```bash
pip install -r requirements.txt
