# CuTe DSL GEMM vs cuBLAS on RTX 5060 Ti (sm_120)

bf16 in / fp32 accumulate / bf16 out, C = A·Bᵀ (nn.Linear / TN layout). cuBLAS
baseline is `torch.matmul(A, B.t())` on the same tensors in the same process.

Environment: RTX 5060 Ti (36 SM, sm_120), CUDA 13.2, driver 595.79,
torch 2.11.0+cu130, nvidia-cutlass-dsl 4.5.0.

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

Clocks float (WSL2 blocks `nvidia-smi -lgc`), so results are given two ways: vs
cuBLAS, and vs the bf16/fp32-accumulate roofline recomputed at the SM clock
sampled during each run. On GeForce, bf16 tensor with fp32 accumulate runs at
512 FLOP/SM/clock — half the fp16-accumulate rate — which is 47.41 TFLOP/s at
the 2572 MHz rated boost. The card sustains ~2.7 GHz under load, so peak is
recomputed per run at the measured clock — dividing by the rated-clock 47.41
instead would push the large shapes past 100%. Roofline % carries ±~2% from
clock sampling.

## Results

| M,N,K | config | TFLOP/s | vs cuBLAS | vs roofline | |
|---|---|---|---|---|---|
| 512,512,512 | 64³ o2 | 25.0 | 100.5% | 48% | tie |
| 1024,1024,1024 | 128³ | 40.2 | 99.9% | 78% | tie |
| 2048,2048,2048 | 64³ o2 | 46.7 | 98.1% | 91% | tie |
| 8192,1024,1024 | 128³ | 46.1 | 98.4% | 91% | tie |
| 4096,4096,512 | 128³ | 46.3 | 102.9% | 92% | win |
| 4096,4096,4096 | 128³ SK | 49.2 | 105.1% | 98% | win |
| 6144,4096,4096 | 128³ N | 49.3 | 105.6% | 99% | win |
| 8192,8192,8192 | 128³ SK | 49.1 | 104.6% | 100% | win |
| 8192,4096,11008 | 128³ N SK | 49.3 | 105.0% | 100% | win |
| 2048,8192,8192 | 128³ | 49.6 | 106.2% | 100% | win |
| 16384,4096,4096 | 128³ N | 49.7 | 106.4% | 100% | win |

**7 win / 4 tie / 0 loss.** Run-to-run σ is 0.1–0.3%; the kernel is steadier than
cuBLAS, whose σ reaches 1.3%. The wins are the large compute-bound shapes, all at
98–100% of the roofline — the same ceiling cuBLAS hits, so the +3–6% over cuBLAS
is real but small. The ties are the small, thin, and latency-bound shapes, at
parity with cuBLAS. The small and thin-K shapes use a 128-tile; a 64-tile
under-fills them.

## Stream-K

Stream-K flattens the persistent schedule's remainder into (tile, k-tile) units
divided evenly over all CTAs, with a cross-CTA fp32 fixup (per-CTA workspace slot
+ release/acquire flag). It matters on one pattern: a low 128-tile count (a small
non-multiple of 36 SMs) with deep K.

| M,N,K | 128-tiles | 128³ DP | 128³ SK | 64³ DP |
|---|---|---|---|---|
| 768,1024,8192 | 48 (1.3 wave) | 65% | 85% | 84% |
| 640,1152,8192 | 45 (1.3 wave) | 61% | 81% | 94% |
| 1152,1152,8192 | 81 (2.3 wave) | 73% | 88% | 79% |

(% of roofline.) Here the 128-tile data-parallel kernel wave-quantizes hard
(61–73%) — the tail wave leaves most SMs idle — and stream-K recovers +20–34%. A
smaller 64-tile is an alternative (more tiles, less quantization) and sometimes
wins outright, so benchmark both `:sk` and a 64-tile. On the large benchmark
shapes (many tiles, near-integer waves) stream-K is within noise of data-parallel:
the persistent scheduler already absorbs the tail. Stream-K also needs deep K —
with shallow K the fixup overhead outweighs the gain.

Correctness needed three sm_120 memory-ordering facts: relaxed `red.global` is
not ordered by a later fence or release (partials use plain stores under the
release flag); weak loads can return stale from L1 across an acquire fence (the
fold uses `ld.relaxed.gpu`); the flag is a release-increment / acquire-poll.
0 failures over >2,000 verification launches.

## Config key

`64³ o2` = 64×64×64 tile, atom (4,1,1), epilogue stage 2, occupancy 2 (grid 72).
`128³` = 128×128×64 tile, epilogue stage 8, occupancy 1 (grid 36). `N` = raster
along N. `SK` = stream-K. bench.py config strings: `64,64,64:4,1,1:e2:o2`,
`128,128,64:n:sk` (`sk` = stream-K, `xN` = extra stream-K waves, `k8` = m16n8k8).

## Caveats

- No clock lock under WSL2; ratios inside ~100±1.5% are parity.
- Stream-K enforces one CTA per SM: with two co-resident, the fixup path corrupts
  the sibling CTA's in-flight fragments (reproducible; root cause not established
  without compute-sanitizer under WSL2/WDDM). Excluded by construction.
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
