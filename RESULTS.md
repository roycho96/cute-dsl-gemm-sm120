# CuTe DSL (CUTLASS 4.x Python) GEMM vs cuBLAS on RTX 5060 Ti (sm_120)

Dates: 2026-07-24 (initial campaign), 2026-07-25 (stream-K + ldmatrix/atom work, re-measurement). GPU: NVIDIA GeForce RTX 5060 Ti (36 SM, sm_120, consumer Blackwell), CUDA 13.0 driver stack, torch 2.11.0+cu130, nvidia-cutlass-dsl 4.5.0.

## 1. Which path worked

**Python CuTe DSL, no fallback needed.** The kernel (`gemm_sm120.py`) is based on NVIDIA's official `examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py` (`Sm120GemmKernel`, BSD-3). Everything TMA/MMA/pipeline-structural in the mainloop and epilogue is NVIDIA's; this repo adds:

- exposed tuning knobs the example hardcodes (`atom_layout`, `occupancy`, `max_ab_stage`, `epi_stage`, scheduler `swizzle_size`/`raster_along_m`, register budgets, `mma_inst_mnk`);
- a **stream-K scheduler mode** (`stream_k=True`): the remainder wave of the persistent schedule is flattened into (tile, k-tile) units and divided evenly over all CTAs, with a cross-CTA fp32 fixup (per-CTA workspace slot + release/acquire flag protocol) — this is new code, not in the example;
- resolution of the example's two mainloop TODOs (see section 3);
- a factored mainloop/epilogue (`_mma_consume_segment`, `_dma_load_segment`, `_epilogue`) so both schedulers share the verified data path.

Architecture note: sm_120 (consumer Blackwell) has **no tcgen05/UMMA/TMEM and no usable cluster multicast** — those are SM100 (datacenter Blackwell). The instruction set used here: **TMA** G2S loads, warp-specialized producer/consumer (1 DMA warp + 4 MMA warps, `setmaxnreg` 40/232), **`ldmatrix.x4` + `mma.sync m16n8k16`**, multi-stage mbarrier pipeline, swizzled smem, TMA-store epilogue, persistent tile scheduler. This is also exactly what cuBLAS runs on this card: its 2048-cube kernel is `cutlass_80_tensorop_bf16_s16816gemm_..._64x64_32x6_tn_align8` — the same Ampere-class instruction path.

Problem setup: **bf16 in, fp32 accumulate, bf16 out**; A[M,K] row-major, B[N,K] row-major, C = A·Bᵀ (nn.Linear / TN layout). cuBLAS baseline = `torch.matmul(A, B.t())` on the *same tensors, same process*.

## 2. Shape-by-shape results

CUDA events, warmup ≥20, 40–120 timed samples per pass (small shapes amortized to ≥2 ms per sample), median reported, both sides CUDA-graph-captured, cuBLAS-first per shape. Every reported config passes rel-err < 2e-2 vs an fp32 `torch.matmul` reference on that exact shape; measured max rel-err is 2.6–3.1e-3 on every shape — **identical to cuBLAS's own bf16 error**.

Because clocks float (WSL2, no `-lgc`), the full set was measured in **three independent passes**; the ratio column shows the median across passes with the observed range. A key measured fact: this kernel's own throughput is far more stable than the baseline's (2048-cube: ours 48.89–49.00 TFLOP/s across all passes, σ≈0.1%; cuBLAS 47.77–49.02, σ≈1.3%). Most of the run-to-run ratio movement on the parity shapes is cuBLAS clock-state variance, not ours.

| M,N,K | config (scheduler) | mine TFLOP/s | cuBLAS TFLOP/s | ratio, median [range] | verdict |
|---|---|---|---|---|---|
| 512,512,512 | 64³ a411 e2 o2 (DP) | 25.06 | 25.32 | 100.1% [98.7–102.1] | TIE (latency-bound, ~10.7 us) |
| 1024,1024,1024 | 64³ a411 e2 o2 (DP) | 40.84 | 40.97 | 100.0% [99.7–100.3] | TIE |
| 2048,2048,2048 | 64³ a411 e2 o2 (DP) | 48.91 | 49.02 | 99.8% [99.5–102.6] | TIE (see section 6 ncu evidence) |
| 4096,4096,4096 | 128×128×64 (stream-K) | 50.17 | 50.03 | 101.0% [99.5–102.8] | TIE→win-lean (SK > DP in every pass) |
| 8192,8192,8192 | 128×128×64 (stream-K) | 49.42 | 47.75 | 103.5% [103.5–105.8] | **WIN** |
| 8192,4096,11008 | 128×128×64 raster-N (stream-K) | 50.78 | 48.77 | 104.2% [104.1–104.4] | **WIN** |
| 2048,8192,8192 | 128×128×64 (DP) | 50.20 | 48.42 | 103.8% [103.7–104.4] | **WIN** |
| 16384,4096,4096 | 128×128×64 raster-N (DP) | 50.96 | 48.25 | 105.8% [105.3–106.0] | **WIN** |
| 4096,4096,512 (thin-K) | 64³ a411 e2 o2 (DP) | 48.09 | 47.08 | 102.3% [102.1–102.3] | **WIN** |
| 8192,1024,1024 | 64³ a411 e2 o2 (DP) | 49.14 | 48.62 | 101.3% [101.1–101.9] | **WIN** (borderline; >101% in all passes) |
| 6144,4096,4096 | 128×128×64 raster-N (DP) | 50.53 | 48.49 | 104.1% [103.9–104.2] | **WIN** |

TFLOP/s columns are from the final consolidated pass (`final.jsonl`); the ratio statistics fold in all passes. Config key: `a411` = MMA atom layout (4,1,1); `64³ e2 o2` = tile 64×64×64, epi_stage 2, occupancy 2 (grid 72); `128×128×64` = epi_stage 8, occupancy 1 (grid 36); stream-K = the new scheduler with default remainder-wave coverage.

**Scope, stated precisely: 0 losses. 7 shapes beat cuBLAS outside clock noise (+1.3% to +5.8%); 4 shapes (the compute-bound squares 512–4096) are ties at the hardware ceiling, with 4096-cube leaning win under stream-K.** The tie shapes cannot be called wins honestly: the day-scale ratio range brackets 100% because the *baseline* moves ±1.3% with clock state.

## 3. What the two upstream TODOs turned out to be

**TODO "leverage ldmatrix.x4" (mainloop permutation):** resolved by measurement, not by new code. The example's ×2 factor on the N permutation already pairs two n8 MMA value tiles so each warp's B fragment spans a 16×16 smem block per k-block — exactly one `ldmatrix.x4` footprint. SASS proof: every emitted LDSM in the kernel is `LDSM.16.M88.4` (32 static instances for atom (2,2,1), 40 for (4,1,1)); there are zero `.1`/`.2` variants. This holds for `atom_layout[1] == 1` as well, which makes **atom (4,1,1) a usable layout on 64-wide tiles** — it measured +0.2–0.9% over (2,2,1) with consistent sign on all five shapes tried, and is now the default config for the 64³ shapes. ((4,1,1) and (1,4,1) still miscompute on 128-wide tiles — excluded, as before; (1,4,1) also miscomputes on 64-wide tiles.) The TODO comment was replaced with an explanation of why the permutation is what enables the x4 path.

**TODO "remove this hard code" (`mma_inst_mnk`):** now a constructor knob, validated against the two shapes `MmaF16BF16Op` accepts. Measured: `(16,8,8)` reaches 51.0% of cuBLAS on 2048-cube — exactly the expected 2× instruction-issue penalty — so `(16,8,16)` (the widest f16/bf16 `mma.sync` on sm_120) stays the default; the knob exists to document that the choice is deliberate, not an oversight.

## 4. The stream-K scheduler

Design: `stream_k=True` splits the tile space into data-parallel full waves plus a flattened remainder. The remainder tiles' (tile, k-tile) units are divided contiguously over all CTAs (`sk_extra_waves` can fold additional full waves in). The CTA holding a tile's k=0 segment owns it: it waits for the (consecutive-bid) contributor CTAs, folds their fp32 partials from a per-CTA workspace slot, and runs the normal TMA-store epilogue. Grid = the persistent CTA count, so every participant is co-resident and the owner spin is deadlock-free by construction.

Getting the cross-CTA reduction correct on sm_120 required three empirical memory-ordering findings (each verified by hammer campaigns of 100–150 launches per configuration, where the broken variants fail between 1/120 and 150/150):

1. **Relaxed global reductions (`red.global`) are fire-and-forget**: neither a later `fence.acq_rel.gpu` nor a later same-thread release operation reliably orders them before a release-flag publish. Partials are therefore written with **plain stores** (which the release flag does cover), not relaxed atomics.
2. **Weak loads can be served stale from the reader's L1 even after an acquire fence** (the L1 is not invalidated). The owner's fold therefore uses **strong relaxed loads** (`ld.relaxed.gpu.global`), which come from the L2 point of coherence but still pipeline like ordinary loads.
3. The flag itself is a **release-increment / acquire-poll** pair; per-thread fences plus a CTA barrier collect all 128 threads' stores under the single release.

Final protocol correctness: 0 failures across >2,000 verification launches (single-launch and CUDA-graph bursts) over 10 shape/config combinations, worst rel-err 2.6–3.1e-3 = cuBLAS's own bf16 error.

Where stream-K lands per shape (both schedulers benchmarked on all 11 shapes):

- **Wins**: 8192-cube (+0.7–2.9% over DP in every pass; the 28/36 remainder wave is the largest tail in the set), 4096-cube (+0.5% consistently), 8192×4096×11008 (+0.2%). These are the shipped configs.
- **Ties**: the other large rectangles (tails ≤1.5% of total waves; fixup overhead cancels the gain).
- **Losses**: every shape whose DP winner uses occupancy 2 (512–2048 squares, thin-K, 8192×1024×1024). Stream-K is restricted to one CTA per SM (below), so it gives up the co-residency overlap that occupancy 2 buys on small tiles — that overlap is worth 2–35%, far more than any tail.

**The occupancy-2 restriction is a real, honestly-unresolved limitation.** With two CTAs of this kernel co-resident on one SM, executing the stream-K fixup path corrupts *in-flight operand fragments of the sibling CTA* (fragment-shaped corruption in tiles the sibling is computing at that moment). This reproduces with any cheap contributor-path code (even with all stores dynamically unreachable) and disappears only when every workspace operation carries per-op release semantics — which costs more than the tail saving. Root cause was not established (no compute-sanitizer under WSL2/WDDM); the launch-time register allocation (96/thread static) does not reserve the `setmaxnreg` targets (232/40), which is spec-gray and benign only on an exclusive SM, but register-budget experiments neither confirmed nor fixed it. The kernel therefore **enforces one CTA per SM for stream-K** structurally (per-CTA smem footprint must exceed half the SM's capacity, so the hardware cannot co-schedule). The data-parallel occupancy-2 configs are unaffected (no cross-CTA code path; verified stable across thousands of launches over both campaign days).

## 5. Tuning trajectory (what moved each shape)

| shape | out-of-box → final | levers |
|---|---|---|
| 512³ | 63.7% → ~100% | 64³ tile, epi_stage 2, occupancy 2, atom (4,1,1) |
| 1024³ | 98.6% → 100.0% | occupancy 2, atom (4,1,1) |
| 2048³ | 90.3% → 99.8% | 64³ tile, occupancy 2, atom (4,1,1) |
| 4096³ | 101.0-o.o.b. → 101.0% median | stream-K (+0.5% over DP in every pass) |
| 8192³ | 102.8% → 103.5–105.8% | stream-K remainder-wave flattening |
| 8192,4096,11008 | 96.5% → 104.2% | raster direction M→N, stream-K |
| 2048,8192,8192 | 102.7% → 103.8% | (raster-M already right) |
| 16384,4096,4096 | 104.0% → 105.8% | raster-N |
| 4096,4096,512 | 99.3% → 102.3% | 64³ + occupancy 2, atom (4,1,1) |
| 8192,1024,1024 | 95.9% → 101.3% | 64³ + occupancy 2, atom (4,1,1) |
| 6144,4096,4096 | 103.0% → 104.1% | raster-N |

Lever ranking is unchanged from the first campaign (occupancy 2 > tile size per shape > raster direction > epi_stage), with two additions at the bottom: stream-K (+0.5–3% but only on large shapes with big remainder waves and occupancy-1 winners) and atom (4,1,1) on 64³ tiles (+0.2–0.9%). Non-levers: scheduler swizzle (still zero effect), mma_regs 232→240, 128×64/64×128 tiles (95.7–98.1% everywhere), `sk_extra_waves` ≥1 (hurts: more split tiles than the balance is worth).

## 6. Honest blockers / caveats

- **Clock lock denied** (`nvidia-smi -lgc`, WSL2). Quantified impact: across one day the cuBLAS baseline on a fixed shape moved ±1.3% (e.g. 2048³: 47.77–49.02 TFLOP/s) while this kernel moved ±0.15%. Every ratio inside ~100±1.5% must be read as parity; the table's ranges make that visible.
- **The compute-bound squares (512–2048, and 4096 without stream-K) are ties because both engines sit at the same hardware ceiling, not because tuning stopped early.** ncu head-to-head on 2048³ (basic + pipe metrics, same shape, same process class): mine — HMMA-pipe issue 48.4% of peak-sustained, DRAM 9.2%, 18.1% warps active, 415 us under ncu; cuBLAS (`cutlass_80_tensorop_bf16_s16816gemm_64x64_32x6`) — HMMA 48.9%, DRAM 9.2%, 16.4% warps active, 414 us. Same instruction class, same tile class, same utilization, 0.3% apart. There is no ≥1.5% win physically available on these shapes at matched clocks; claiming one from a lucky pass (e.g. the 102.6% sample in the 2048³ range, or 102.1% at 512³) would be cherry-picking the baseline's slow runs.
- **Stream-K requires one CTA per SM** (enforced; see section 4) — the occupancy-2 co-residency corruption is reproducible, unexplained at root-cause level, and excluded by construction rather than fixed.
- **sm_120 memory-ordering quirks** (section 4): relaxed `red.global` unordered by later fences/releases; weak loads stale in L1 across acquire fences. These cost most of a day of correctness debugging and are load-bearing knowledge for any cross-CTA protocol on this part.
- **NVIDIA's example epilogue buffer-reuse race** (unchanged): `epi_stage <` #subtiles (= tileM·tileN/(64·32)) reuses TMA-store buffers before the store completes → wrong results. All configs respect `epi_stage ≥ #subtiles`.
- **`atom_layout` variants (1,4,1) everywhere and (4,1,1) on 128-wide tiles miscompute** in this DSL version (verified, excluded). (4,1,1) is correct on 64-wide tiles and is used there.
- Measurement order is cuBLAS-first per shape (coolest GPU for the baseline), conservative for the reported wins.

## 7. Files

- `gemm_sm120.py` — the CuTe DSL kernel: NVIDIA blackwell_geforce example + tuning knobs + stream-K scheduler + resolved TODOs.
- `bench.py` — harness: correctness gate vs fp32 reference + CUDA-event/CUDA-graph benchmark vs `torch.matmul`. Config strings like `64,64,64:4,1,1:e2:o2:g2` or `128,128,64:n:sk` (`sk` = stream-K, `xN` = extra stream-K waves, `k8` = m16n8k8 instruction).
- `profile_one.py` — standalone single-kernel runner for ncu.
- `final.jsonl` — the consolidated final pass backing the table (both schedulers on the large shapes).
