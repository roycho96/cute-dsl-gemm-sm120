# CuTe DSL GEMM for sm_120 (consumer Blackwell), benchmarked vs cuBLAS

A CuTe DSL (CUTLASS 4.x Python) batched GEMM for NVIDIA sm_120 (RTX 50-series,
consumer Blackwell), tuned and extended to match or beat cuBLAS across a range
of shapes. bf16 in / fp32 accumulate / bf16 out, `C = A · Bᵀ` (nn.Linear / TN
layout).

The kernel is a fork of NVIDIA's official `blackwell_geforce/dense_gemm.py`
CuTe DSL example (`Sm120GemmKernel`, BSD-3). sm_120 has no tcgen05/UMMA/TMEM and
no usable cluster multicast (those are SM100, datacenter Blackwell), so the
instruction path here is TMA loads + warp-specialized producer/consumer +
`ldmatrix.x4` + `mma.sync m16n8k16` + a multi-stage mbarrier pipeline — the same
Ampere-class tensor path cuBLAS itself runs on this card.

This repo adds, on top of the example:

- a **stream-K scheduler** with a cross-CTA fp32 fixup — new code, not in the
  example. Getting the reduction correct on sm_120 surfaced two real
  memory-ordering quirks (relaxed `red.global` unordered by later fences;
  weak loads served stale from L1 across acquire fences); see
  [RESULTS.md](RESULTS.md) §4.
- **tuning knobs** the example hardcodes: MMA atom layout, occupancy,
  pipeline depth, epilogue stages, scheduler raster direction, register
  budgets, MMA instruction shape.
- resolution of the example's two mainloop TODOs (RESULTS.md §3).

## Results (RTX 5060 Ti, 36 SM, CUDA 13, torch 2.11, cutlass-dsl 4.5.0)

11 shapes vs cuBLAS (`torch.matmul`, same tensors, same process, CUDA-graph
captured, median of 3 independent passes, cuBLAS-first per shape):

**0 losses · 7 wins outside clock noise (+1.3% to +5.8%) · 4 compute-bound
squares tie at the tensor-core ceiling.**

The 4 ties are not a tuning gap. ncu head-to-head on 2048³ shows cuBLAS running
the *identical* instruction class (`cutlass_80_tensorop_bf16_s16816gemm_64x64_32x6`)
at the *same* ~48% HMMA-pipe utilization, 0.3% apart in runtime — the remaining
margin is smaller than cuBLAS's own ±1.3% clock-state variance, while this
kernel holds ±0.15%. Claiming a win there would be cherry-picking the baseline's
slow runs. Full table, per-shape scheduler choice, and evidence are in
[RESULTS.md](RESULTS.md).

Every reported config is gated on a rel-err < 2e-2 fp32 reference before timing;
measured max rel-err is 2.6–3.1e-3, identical to cuBLAS's own bf16 error. The
stream-K protocol survived >2,000 verification launches with 0 failures.

## Run

```bash
pip install torch nvidia-cutlass-dsl        # torch 2.11 + cu13, cutlass-dsl 4.5.0

# one large shape, data-parallel vs stream-K:
python bench.py --shapes "8192,8192,8192" --configs "128,128,64;128,128,64:sk"

# small shape, tuned config (64³ tile, atom (4,1,1), epi_stage 2, occupancy 2):
python bench.py --shapes "2048,2048,2048" --configs "64,64,64:4,1,1:e2:o2"
```

Config-string syntax (`sk` = stream-K, `n` = raster-N, `eN`/`oN`/`gN`, etc.) is
documented in [RESULTS.md](RESULTS.md) §7. `profile_one.py` is a standalone
single-kernel runner for ncu.

## Attribution & license

`gemm_sm120.py` is derived from NVIDIA CUTLASS's
`examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py` and retains NVIDIA's
BSD-3-Clause header verbatim. The mainloop/epilogue TMA/MMA/pipeline structure
is NVIDIA's; the stream-K scheduler, the exposed tuning knobs, and the benchmark
harness (`bench.py`, `profile_one.py`) are additions in this repo. Everything is
BSD-3-Clause — see [LICENSE](LICENSE).
