# CuTe DSL GEMM for sm_120

A CuTe DSL (CUTLASS 4.x Python) batched GEMM for NVIDIA sm_120 (RTX 50-series,
consumer Blackwell), tuned and benchmarked against cuBLAS at locked clocks.
bf16 in / fp32 accumulate / bf16 out, `C = A · Bᵀ` (nn.Linear / TN layout).

The kernel is a fork of NVIDIA's `blackwell_geforce/dense_gemm.py`
(`Sm120GemmKernel`, BSD-3). sm_120 has no tcgen05/UMMA/TMEM and no usable cluster
multicast (those are SM100, datacenter Blackwell), so the path is TMA loads +
warp-specialized producer/consumer + `ldmatrix.x4` + `mma.sync m16n8k16` + an
mbarrier pipeline — the same Ampere-class tensor path cuBLAS runs on this card.

Added on top of the example:

- a **stream-K scheduler** with a cross-CTA fp32 fixup (new code). Getting the
  reduction correct on sm_120 surfaced two memory-ordering quirks; see
  [RESULTS.md](RESULTS.md).
- **tuning knobs** the example hardcodes: MMA atom layout, occupancy, pipeline
  depth, epilogue stages, raster direction, register budgets, MMA instruction shape.
- resolution of the example's two mainloop TODOs.

## Results (RTX 5060 Ti, 36 SM, CUDA 13.2, torch 2.11, cutlass-dsl 4.5.0, clock locked 2595 MHz)

11 shapes, 20-pass median, throughput vs cuBLAS and vs the fp32-accumulate
roofline:

**This kernel reaches 93–99% of cuBLAS (46–97% of the roofline). cuBLAS is faster
or equal on every shape.** The large compute-bound shapes come closest — 97–99%
of cuBLAS at 96–97% of the roofline (8192³ and 16384×4096×4096 at 99%). Small and
thin shapes trail more, where cuBLAS uses specialized kernels.

Both kernels run the same Ampere-class tensor path at the same locked clock, near
the fp32-accumulate pipe ceiling; cuBLAS closes the last 1–3% with a
deeper-pipelined smaller tile, per-shape kernel selection, and SASS-level
scheduling. Matching cuBLAS to within a few percent on standard dense bf16 is the
realistic ceiling — beating it needs epilogue fusion or FP8/NVFP4, which cuBLAS
does not cover. The value here is a competitive kernel that can be modified and
fused into.

Separately, the stream-K scheduler recovers **+20–34%** over plain data-parallel
on low-tile-count, deep-K shapes, where the persistent data-parallel kernel
wave-quantizes and leaves SMs idle in the tail wave.

Correctness: every config gated on rel-err < 2e-2 vs fp32 (measured max
2.6–3.1e-3, cuBLAS's own bf16 error). Stream-K survived >2,000 verification
launches with 0 failures.

Full table, per-shape roofline %, and the stream-K data are in
[RESULTS.md](RESULTS.md).

## Run

```bash
pip install torch nvidia-cutlass-dsl        # torch 2.11 + cu13, cutlass-dsl 4.5.0

# large shape, data-parallel vs stream-K:
python bench.py --shapes "8192,8192,8192" --configs "128,128,64;128,128,64:sk"

# low-tile deep-K shape where stream-K recovers wave-quantization:
python bench.py --shapes "768,1024,8192" --configs "128,128,64;128,128,64:sk"

# small shape, tuned config:
python bench.py --shapes "2048,2048,2048" --configs "64,64,64:4,1,1:e2:o2"
```

Config-string syntax and `profile_one.py` (a standalone ncu runner) are
documented in [RESULTS.md](RESULTS.md).

## Attribution & license

`gemm_sm120.py` is derived from NVIDIA CUTLASS's
`examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py` and retains NVIDIA's
BSD-3-Clause header. The mainloop/epilogue TMA/MMA/pipeline structure is
NVIDIA's; the stream-K scheduler, the exposed tuning knobs, and the benchmark
harness (`bench.py`, `profile_one.py`) are additions in this repo. Everything is
BSD-3-Clause — see [LICENSE](LICENSE).
