# CuTe DSL GEMM vs cuBLAS on RTX 5060 Ti (sm_120)

bf16 in / fp32 accumulate / bf16 out, C = A·Bᵀ (nn.Linear / TN layout). cuBLAS
baseline is `torch.matmul(A, B.t())` on the same tensors in the same process.

Environment: RTX 5060 Ti (36 SM, sm_120), CUDA 13.2, driver 595.79,
torch 2.11.0+cu130, nvidia-cutlass-dsl 4.5.0. SM clock locked at 2595 MHz.

## Kernel

Fork of NVIDIA `blackwell_geforce/dense_gemm.py` (`Sm120GemmKernel`, BSD-3).
NVIDIA's TMA/MMA/pipeline mainloop and epilogue, unchanged. Added here: tuning
knobs the example hardcodes (atom layout, occupancy, pipeline depth, epilogue
stages, raster, register budgets, MMA shape), a stream-K scheduler, and
resolution of the two mainloop TODOs. Instruction path: TMA loads,
warp-specialized producer/consumer (1 DMA + 4 MMA warps, `setmaxnreg` 40/232),
`ldmatrix.x4` + `mma.sync m16n8k16`, mbarrier pipeline, TMA-store epilogue — the
Ampere-class tensor path cuBLAS also runs on this card
(`cutlass_80_tensorop_bf16_s16816gemm_64x64_32x6`).

## Method

CUDA-event timing, both sides CUDA-graph captured, 20 passes per shape (median
per pass), cuBLAS measured first. Correctness gated at rel-err < 2e-2 vs an fp32
reference; measured max 2.6–3.1e-3 (cuBLAS's own bf16 error). Throughput is
TFLOP/s = 2·M·N·K / time.

The SM clock is locked at 2595 MHz, so both kernels run at the same fixed clock.
Roofline: on GeForce, bf16 with fp32 accumulate runs at 512 FLOP/SM/clock (half
the fp16-accumulate rate) = **47.8 TFLOP/s** at 36 SM × 2595 MHz.

## Results

| M,N,K | config | TFLOP/s | vs cuBLAS | vs roofline |
|---|---|---|---|---|
| 512,512,512 | 64³ o2 | 21.7 | 93.3% | 46% |
| 1024,1024,1024 | 128³ | 37.3 | 99.9% | 79% |
| 2048,2048,2048 | 64³ o2 | 43.3 | 96.0% | 91% |
| 4096,4096,4096 | 128³ | 45.7 | 97.5% | 96% |
| 8192,8192,8192 | 128³ | 46.0 | 99.3% | 96% |
| 8192,4096,11008 | 128³ N | 46.2 | 98.8% | 97% |
| 2048,8192,8192 | 128³ | 46.0 | 97.6% | 96% |
| 16384,4096,4096 | 128³ N | 46.6 | 99.2% | 97% |
| 4096,4096,512 | 128³ | 42.8 | 97.2% | 90% |
| 8192,1024,1024 | 128³ | 42.7 | 94.6% | 90% |
| 6144,4096,4096 | 128³ N | 46.3 | 98.6% | 97% |

cuBLAS is faster or equal on every shape; this kernel reaches **93–99% of cuBLAS
(46–97% of the roofline)**. The large compute-bound shapes come closest — 97–99%
of cuBLAS at 96–97% of the roofline, with 8192³ and 16384×4096×4096 at 99%. The
small and thin shapes trail more (93–97%), where cuBLAS uses specialized split-K
kernels. Run-to-run σ is 0.1–0.5%.

Why the last few percent goes to cuBLAS: both kernels use the same Ampere-class
tensor path (`ldmatrix.x4` + `mma.sync m16n8k16`, fp32 accumulate — sm_120 has no
tcgen05/UMMA), and both sit near the fp32-accumulate HMMA-pipe ceiling. cuBLAS
closes the gap with a smaller tile and deeper pipeline (its 2048³ kernel is
64×64 / 6-stage, while a 128-tile here fits only 2 stages, so its memory latency
is less hidden), per-shape kernel selection, and SASS-level instruction
scheduling. Matching cuBLAS to within a few percent on standard dense bf16 is the
realistic ceiling; the value is a competitive kernel that can be modified and
fused into (epilogue fusion, FP8/NVFP4), which cuBLAS does not cover.

## Stream-K

Stream-K flattens the persistent schedule's remainder into (tile, k-tile) units
divided evenly over all CTAs, with a cross-CTA fp32 fixup (per-CTA workspace slot
+ release/acquire flag). It matters on one pattern: a low 128-tile count (a small
non-multiple of 36 SMs) with deep K.

| M,N,K | 128-tiles | 128³ DP | 128³ SK | 64³ DP |
|---|---|---|---|---|
| 768,1024,8192 | 48 (1.3 wave) | 65% | 85% | 84% |
| 640,1152,8192 | 45 (1.3 wave) | 62% | 83% | 95% |
| 1152,1152,8192 | 81 (2.3 wave) | 74% | 89% | 83% |

(% of roofline.) Here the 128-tile data-parallel kernel wave-quantizes hard
(62–74%) — the tail wave leaves most SMs idle — and stream-K recovers +20–34%. A
smaller 64-tile is an alternative (more tiles, less quantization) and sometimes
wins outright, so benchmark both `:sk` and a 64-tile. On the main benchmark
shapes (many tiles, near-integer waves) stream-K is within noise of data-parallel
and is not used: the persistent scheduler already absorbs the tail. Stream-K also
needs deep K — with shallow K the fixup overhead outweighs the gain.

Correctness needed three sm_120 memory-ordering facts: relaxed `red.global` is
not ordered by a later fence or release (partials use plain stores under the
release flag); weak loads can return stale from L1 across an acquire fence (the
fold uses `ld.relaxed.gpu`); the flag is a release-increment / acquire-poll.
0 failures over >2,000 verification launches.

## Config key

`64³ o2` = 64×64×64 tile, atom (4,1,1), epilogue stage 2, occupancy 2 (grid 72).
`128³` = 128×128×64 tile, epilogue stage 8, occupancy 1 (grid 36). `N` = raster
along N. bench.py config strings: `64,64,64:4,1,1:e2:o2`, `128,128,64:n`,
`128,128,64:sk` (`sk` = stream-K, `xN` = extra stream-K waves, `k8` = m16n8k8).

## Caveats

- Stream-K enforces one CTA per SM: with two co-resident, the fixup path corrupts
  the sibling CTA's in-flight fragments (reproducible; root cause not established).
  Excluded by construction.
- NVIDIA's example epilogue reuses TMA-store buffers when epi_stage < #subtiles
  (tileM·tileN / (64·32)) and returns wrong results; all configs keep
  epi_stage ≥ #subtiles.
- atom (1,4,1) everywhere and (4,1,1) on 128-wide tiles miscompute in this DSL
  version (excluded); (4,1,1) is correct on 64-wide tiles.

## Files

- `gemm_sm120.py` — kernel (NVIDIA example + knobs + stream-K + resolved TODOs).
- `bench.py` — correctness gate + CUDA-graph benchmark vs `torch.matmul`.
- `profile_one.py` — single-kernel runner for ncu.
- `final.jsonl` — the data behind the tables.
