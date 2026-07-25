# CuTe DSL GEMM vs cuBLAS on RTX 5060 Ti (sm_120)

A batched bf16 GEMM in CuTe DSL (CUTLASS 4.x Python), tuned and extended to match
or beat cuBLAS on a consumer Blackwell card. bf16 in, fp32 accumulate, bf16 out;
C = A·Bᵀ (nn.Linear / TN layout). The cuBLAS baseline is `torch.matmul(A, B.t())`
on the same tensors in the same process.

Environment: NVIDIA GeForce RTX 5060 Ti (36 SM, sm_120), CUDA 13.0,
torch 2.11.0+cu130, nvidia-cutlass-dsl 4.5.0.

## What the kernel is

`gemm_sm120.py` is a fork of NVIDIA's `examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py`
(`Sm120GemmKernel`, BSD-3). The TMA/MMA/pipeline structure of the mainloop and
epilogue is NVIDIA's. This repo adds three things on top:

- Tuning knobs the example hardcodes: MMA atom layout, occupancy, pipeline depth
  (`max_ab_stage`), epilogue stages, scheduler raster direction, register budgets,
  and MMA instruction shape.
- A stream-K scheduler (`stream_k=True`) — new code, described below.
- Resolution of the example's two mainloop TODOs.

sm_120 (consumer Blackwell) has no tcgen05/UMMA/TMEM and no usable cluster
multicast; those belong to SM100 (datacenter Blackwell). The instruction path here
is TMA global-to-shared loads, a warp-specialized producer/consumer split (1 DMA
warp + 4 MMA warps, `setmaxnreg` 40/232), `ldmatrix.x4` feeding `mma.sync m16n8k16`,
a multi-stage mbarrier pipeline, and a TMA-store epilogue. This is the same
Ampere-class tensor path cuBLAS runs on this card: its 2048-cube kernel is
`cutlass_80_tensorop_bf16_s16816gemm_64x64_32x6_tn_align8`.

## Results

11 shapes. CUDA-event timing, ≥20 warmup, 40–120 timed samples per pass (small
shapes amortized to ≥2 ms/sample), median reported, both sides CUDA-graph captured,
cuBLAS measured first per shape. Every config clears rel-err < 2e-2 against an fp32
reference; measured max rel-err is 2.6–3.1e-3, the same as cuBLAS's own bf16 error.

Clocks float (WSL2, no clock lock), so the full set ran in three independent passes.
The ratio column is the median across passes with the observed range. The kernel's
own throughput is far steadier than the baseline's: on 2048-cube it holds
48.89–49.00 TFLOP/s (σ≈0.1%) while cuBLAS swings 47.77–49.02 (σ≈1.3%), so most of
the ratio movement on the close shapes is cuBLAS clock variance, not this kernel.

| M,N,K | config (scheduler) | mine TFLOP/s | cuBLAS TFLOP/s | ratio, median [range] | verdict |
|---|---|---|---|---|---|
| 512,512,512 | 64³ a411 e2 o2 (DP) | 25.06 | 25.32 | 100.1% [98.7–102.1] | tie (latency-bound, ~10.7 us) |
| 1024,1024,1024 | 64³ a411 e2 o2 (DP) | 40.84 | 40.97 | 100.0% [99.7–100.3] | tie |
| 2048,2048,2048 | 64³ a411 e2 o2 (DP) | 48.91 | 49.02 | 99.8% [99.5–102.6] | tie (ncu: at ceiling) |
| 4096,4096,4096 | 128×128×64 (stream-K) | 50.17 | 50.03 | 101.0% [99.5–102.8] | tie, SK-leaning win |
| 8192,8192,8192 | 128×128×64 (stream-K) | 49.42 | 47.75 | 103.5% [103.5–105.8] | **win** |
| 8192,4096,11008 | 128×128×64 raster-N (stream-K) | 50.78 | 48.77 | 104.2% [104.1–104.4] | **win** |
| 2048,8192,8192 | 128×128×64 (DP) | 50.20 | 48.42 | 103.8% [103.7–104.4] | **win** |
| 16384,4096,4096 | 128×128×64 raster-N (DP) | 50.96 | 48.25 | 105.8% [105.3–106.0] | **win** |
| 4096,4096,512 (thin-K) | 64³ a411 e2 o2 (DP) | 48.09 | 47.08 | 102.3% [102.1–102.3] | **win** |
| 8192,1024,1024 | 64³ a411 e2 o2 (DP) | 49.14 | 48.62 | 101.3% [101.1–101.9] | **win** |
| 6144,4096,4096 | 128×128×64 raster-N (DP) | 50.53 | 48.49 | 104.1% [104.1–104.2] | **win** |

Config key: `a411` = MMA atom (4,1,1); `64³ e2 o2` = 64×64×64 tile, epilogue stage 2,
occupancy 2 (grid 72); `128×128×64` = epilogue stage 8, occupancy 1 (grid 36);
`sk` = stream-K. TFLOP/s columns come from the final pass (`final.jsonl`); the ratio
statistics fold in all three passes.

Bottom line: 0 losses. 7 shapes beat cuBLAS beyond clock noise (+1.3% to +5.8%).
The 4 compute-bound squares (512–4096) tie at the hardware ceiling, with 4096-cube
leaning to a win under stream-K. Those ties are real parity, not unfinished tuning —
see "Why the squares tie" below.

## What the knobs are worth

Per-shape lever ranking, strongest first: occupancy 2 (small tiles), then tile size,
then raster direction, then epilogue stages. On top of those, stream-K adds +0.5–3%
but only on large shapes with big remainder waves and occupancy-1 winners, and atom
(4,1,1) adds +0.2–0.9% on 64³ tiles. Levers that did nothing: scheduler swizzle,
`mma_regs` 232→240, 128×64 / 64×128 tiles (95.7–98.1% everywhere), and
`sk_extra_waves` ≥ 1 (worse — it splits more tiles than the load balance is worth).

## Stream-K scheduler

Stream-K splits the tile space into full data-parallel waves plus a remainder. The
remainder's (tile, k-tile) units are divided contiguously across all CTAs. The CTA
that owns a tile's k=0 segment waits for the other CTAs contributing to that tile,
folds their fp32 partials from a per-CTA workspace slot, then runs the normal
TMA-store epilogue. The grid is exactly the persistent CTA count, so every
contributor is resident and the owner's wait cannot deadlock.

Getting the cross-CTA reduction correct on sm_120 took three memory-ordering
findings, each pinned down by 100–150 launches per variant (broken variants failed
anywhere from 1/120 to 150/150):

1. Relaxed global reductions (`red.global`) are fire-and-forget. Neither a later
   `fence.acq_rel.gpu` nor a later same-thread release reliably orders them ahead of
   a release-flag publish. Partials are written with plain stores, which the release
   flag does cover.
2. Weak loads can come back stale from the reader's L1 even after an acquire fence,
   because the L1 is not invalidated. The fold uses strong relaxed loads
   (`ld.relaxed.gpu.global`), which read from L2 (the coherence point) but still
   pipeline like ordinary loads.
3. The flag is a release-increment / acquire-poll pair. Per-thread fences plus a CTA
   barrier put all 128 threads' stores under the one release.

The final protocol ran >2,000 verification launches (single and CUDA-graph bursts)
across 10 shape/config combinations with 0 failures, worst rel-err 2.6–3.1e-3.

Where it helps, measured against data-parallel on all 11 shapes:

- Wins: 8192-cube (+0.7–2.9%; its 28/36 remainder wave is the biggest tail in the
  set), 4096-cube (+0.5%), 8192×4096×11008 (+0.2%). These are the shipped configs.
- Ties: the other large rectangles — tails ≤1.5% of total waves, so fixup overhead
  cancels the gain.
- Losses: every shape whose best data-parallel config uses occupancy 2 (512–2048
  squares, thin-K, 8192×1024×1024). Stream-K runs one CTA per SM (below), so it gives
  up the co-residency overlap occupancy 2 buys on small tiles — worth 2–35%, far more
  than any tail.

### The occupancy-2 limitation (unresolved)

With two of these CTAs co-resident on one SM, running the stream-K fixup path
corrupts in-flight operand fragments of the sibling CTA. It reproduces with any cheap
contributor-path code, even when every store is dynamically unreachable, and clears
only when every workspace op carries per-op release semantics — which costs more than
the tail it would save. I could not establish root cause (no compute-sanitizer under
WSL2/WDDM). The launch-time register allocation (96/thread static) does not reserve
the `setmaxnreg` targets (232/40), which is spec-gray and harmless only on an
exclusive SM, but register-budget experiments neither confirmed nor fixed it.

So stream-K enforces one CTA per SM structurally: its per-CTA smem footprint exceeds
half the SM's capacity, so the hardware cannot co-schedule a second. The
data-parallel occupancy-2 configs share no cross-CTA code and are unaffected.

## The two upstream TODOs

**"leverage ldmatrix.x4"** — resolved by measurement, not new code. The example's ×2
factor on the N permutation already pairs two n8 MMA value tiles, so each warp's B
fragment spans a 16×16 smem block per k-block: exactly one `ldmatrix.x4` footprint.
SASS confirms it — every LDSM in the kernel is `LDSM.16.M88.4` (32 instances for atom
(2,2,1), 40 for (4,1,1)), with zero `.1`/`.2` variants. It holds when
`atom_layout[1] == 1` too, which makes atom (4,1,1) usable on 64-wide tiles: it
measured +0.2–0.9% over (2,2,1) with consistent sign on all five shapes tried, and is
now the default for 64³ shapes. The TODO comment is replaced with an explanation of
why the permutation is what enables the x4 path.

**"remove this hard code" (`mma_inst_mnk`)** — now a constructor knob, validated
against the two shapes `MmaF16BF16Op` accepts. `(16,8,8)` reaches 51.0% of cuBLAS on
2048-cube, exactly the 2× instruction-issue penalty you would expect, so `(16,8,16)`
(the widest f16/bf16 `mma.sync` on sm_120) stays the default. The knob documents that
the choice is deliberate.

## Why the squares tie

The compute-bound squares (512–2048, and 4096 without stream-K) tie because both
kernels sit at the same hardware ceiling, not because tuning stopped early. ncu
head-to-head on 2048³, same shape and process class:

- This kernel: HMMA-pipe issue 48.4% of peak-sustained, DRAM 9.2%, 18.1% warps
  active, 415 us under ncu.
- cuBLAS (`cutlass_80_tensorop_bf16_s16816gemm_64x64_32x6`): HMMA 48.9%, DRAM 9.2%,
  16.4% warps active, 414 us.

Same instruction class, same tile class, same utilization, 0.3% apart. There is no
≥1.5% win physically available on these shapes at matched clocks. Claiming one from a
lucky pass — the 102.6% sample in the 2048³ range, or 102.1% at 512³ — would just be
catching the baseline on a slow clock.

## Limitations and caveats

- No clock lock (`nvidia-smi -lgc` denied under WSL2). Over a day the cuBLAS baseline
  on a fixed shape moved ±1.3% (2048³: 47.77–49.02 TFLOP/s) while this kernel moved
  ±0.15%. Read any ratio inside ~100±1.5% as parity; the table ranges make that
  visible.
- Stream-K is restricted to one CTA per SM. The occupancy-2 corruption is
  reproducible, unexplained at root cause, and excluded by construction rather than
  fixed.
- sm_120 memory-ordering quirks (stream-K section): relaxed `red.global` unordered by
  later fences/releases; weak loads stale in L1 across acquire fences.
- NVIDIA's example has an epilogue buffer-reuse race: `epi_stage` < #subtiles
  (tileM·tileN / (64·32)) reuses TMA-store buffers before the store finishes and
  returns wrong results. All configs keep `epi_stage ≥ #subtiles`.
- `atom_layout` (1,4,1) everywhere, and (4,1,1) on 128-wide tiles, miscompute in this
  DSL version (verified, excluded). (4,1,1) is correct on 64-wide tiles and used there.
- Measurement order is cuBLAS-first per shape (coolest GPU for the baseline), which is
  conservative for the reported wins.

## Files

- `gemm_sm120.py` — the kernel: NVIDIA blackwell_geforce example + tuning knobs +
  stream-K + resolved TODOs.
- `bench.py` — harness: fp32 correctness gate + CUDA-event/CUDA-graph benchmark vs
  `torch.matmul`. Config strings like `64,64,64:4,1,1:e2:o2:g2` or `128,128,64:n:sk`
  (`sk` = stream-K, `xN` = extra stream-K waves, `k8` = m16n8k8 instruction).
- `profile_one.py` — standalone single-kernel runner for ncu.
- `final.jsonl` — the consolidated pass behind the table.
