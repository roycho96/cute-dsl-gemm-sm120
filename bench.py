#!/usr/bin/env python
"""Benchmark CuTe DSL sm_120 GEMM vs cuBLAS (torch.matmul, bf16).

Layouts (identical operands for both):
  A [M, K] bf16 row-major, B [N, K] bf16 row-major, C = A @ B^T [M, N] bf16.
This is the nn.Linear / TN layout: a_major=k, b_major=k, c_major=n.

Timing: CUDA events. Both cuBLAS and the DSL kernel are (optionally) captured
into a CUDA graph so small shapes are not dominated by Python launch overhead.
Median over --iters timed samples, warmup first, cuBLAS and DSL measured
back-to-back in the same process (shared thermal/clock state).
"""

import argparse
import json
import math
import statistics
import time

import torch
import cuda.bindings.driver as cuda_drv

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from gemm_sm120 import Sm120GemmKernel


def make_cute_view(t2d: torch.Tensor, dynamic: bool):
    """(rows, cols) row-major torch GPU tensor -> cute tensor (rows, cols, 1)."""
    t3d = t2d.unsqueeze(0).permute(1, 2, 0)  # (rows, cols, 1), stride (cols, 1, rows*cols)
    ct = from_dlpack(t3d, assumed_align=16)
    if dynamic:
        ct = ct.mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(
            mode=1, stride_order=(2, 0, 1), divisibility=8
        )
    return ct


def cuda_stream():
    return cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)


def time_fn(fn, warmup, iters, inner=1):
    """Return list of per-call times (seconds) using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3 / inner)
    return times


def build_graph(fn):
    g = torch.cuda.CUDAGraph()
    # warm up in a side stream first (required before capture)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        fn()
    return g


def bench_shape(M, N, K, configs, warmup, iters, use_graph=True, dynamic=False,
                tol=2e-2, verbose=True):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    A = (torch.randn(M, K, device=dev) / math.sqrt(K)).to(torch.bfloat16)
    B = (torch.randn(N, K, device=dev)).to(torch.bfloat16)
    C = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    C_ref_f32 = torch.matmul(A.float(), B.float().t())
    ref_max = C_ref_f32.abs().max().item()

    flops = 2.0 * M * N * K

    # how many launches per timed sample so one sample >= ~2ms of GPU work
    est_t = flops / 40e12
    inner = max(1, int(2e-3 / max(est_t, 1e-9)))
    inner = min(inner, 200)

    results = {"M": M, "N": N, "K": K}

    # ---- cuBLAS baseline ----
    def cublas_call():
        torch.matmul(A, B.t(), out=C)

    if use_graph:
        g = build_graph(cublas_call)
        fn = g.replay
    else:
        fn = cublas_call
    times = time_fn(fn, warmup, iters, inner)
    t_cublas = statistics.median(times)
    cublas_tflops = flops / t_cublas / 1e12
    # cuBLAS own numeric error (context for tolerance)
    cublas_call()
    torch.cuda.synchronize()
    cublas_err = (C.float() - C_ref_f32).abs().max().item() / ref_max
    results["cublas"] = {"tflops": cublas_tflops, "t_ms": t_cublas * 1e3,
                        "relerr": cublas_err}
    if verbose:
        print(f"  cuBLAS: {cublas_tflops:8.2f} TFLOP/s ({t_cublas*1e6:9.1f} us) relerr={cublas_err:.2e}")

    results["configs"] = []
    for cfg in configs:
        tag = cfg_tag(cfg)
        try:
            compiled, mWS, mFL, _, _ = get_compiled(cfg, A, B, C, dynamic)
        except Exception as e:
            results["configs"].append({"cfg": tag, "error": str(e)[:300]})
            if verbose:
                print(f"  {tag}: COMPILE FAIL: {str(e)[:150]}")
            continue

        mA = make_cute_view(A, dynamic)
        mB = make_cute_view(B, dynamic)
        mC = make_cute_view(C, dynamic)

        def dsl_call():
            compiled(mA, mB, mC, mWS, mFL, cuda_stream())

        # correctness first
        C.zero_()
        torch.cuda.synchronize()
        try:
            dsl_call()
            torch.cuda.synchronize()
        except Exception as e:
            results["configs"].append({"cfg": tag, "error": "RUN: " + str(e)[:300]})
            if verbose:
                print(f"  {tag}: RUN FAIL: {str(e)[:150]}")
            continue
        relerr = (C.float() - C_ref_f32).abs().max().item() / ref_max
        ok = relerr < tol
        if not ok:
            results["configs"].append({"cfg": tag, "relerr": relerr, "verdict": "WRONG"})
            if verbose:
                print(f"  {tag}: INCORRECT relerr={relerr:.3e}")
            continue

        if use_graph:
            try:
                g2 = build_graph(dsl_call)
                fn2 = g2.replay
            except Exception:
                fn2 = dsl_call
        else:
            fn2 = dsl_call
        times = time_fn(fn2, warmup, iters, inner)
        t_dsl = statistics.median(times)
        dsl_tflops = flops / t_dsl / 1e12
        ratio = t_cublas / t_dsl
        results["configs"].append({"cfg": tag, "tflops": dsl_tflops,
                                   "t_ms": t_dsl * 1e3, "relerr": relerr,
                                   "pct_cublas": 100.0 * ratio})
        if verbose:
            print(f"  {tag}: {dsl_tflops:8.2f} TFLOP/s ({t_dsl*1e6:9.1f} us) "
                  f"= {100*ratio:6.1f}% of cuBLAS  relerr={relerr:.2e}")
    return results


def cfg_tag(cfg):
    t = cfg["tile"]
    a = cfg.get("atom", (2, 2, 1))
    s = cfg.get("stages", None)
    o = cfg.get("occupancy", 1)
    e = cfg.get("epi_stage", 8)
    w = cfg.get("swizzle", 1)
    rm = "m" if cfg.get("raster_m", True) else "n"
    g = cfg.get("grid_mult", 1)
    sk = f"_sk{cfg.get('sk_extra', 0)}" if cfg.get("stream_k", False) else ""
    k8 = "_k8" if cfg.get("mma_inst", (16, 8, 16)) == (16, 8, 8) else ""
    return (f"t{t[0]}x{t[1]}x{t[2]}_a{a[0]}{a[1]}{a[2]}_s{s or 'max'}"
            f"_o{o}_e{e}_w{w}{rm}_g{g}{sk}{k8}")


_compile_cache = {}


def get_compiled(cfg, A, B, C, dynamic):
    M, K = A.shape
    N = B.shape[0]
    key = (cfg_tag(cfg),) + (() if dynamic else (M, N, K))
    if key in _compile_cache:
        return _compile_cache[key]
    stream_k = cfg.get("stream_k", False)
    sk_extra = cfg.get("sk_extra", 0)
    tile = cfg["tile"]
    gemm = Sm120GemmKernel(
        cutlass.Float32,
        tile,
        atom_layout=cfg.get("atom", (2, 2, 1)),
        occupancy=cfg.get("occupancy", 1),
        max_ab_stage=cfg.get("stages", None),
        load_regs=cfg.get("load_regs", 40),
        mma_regs=cfg.get("mma_regs", 232),
        epi_stage_req=cfg.get("epi_stage", 8),
        swizzle_size=cfg.get("swizzle", 1),
        raster_along_m=cfg.get("raster_m", True),
        stream_k=stream_k,
        sk_extra_waves=sk_extra,
        mma_inst_mnk=cfg.get("mma_inst", (16, 8, 16)),
    )
    hw = cutlass.utils.HardwareInfo()
    max_active_clusters = hw.get_max_active_clusters(1) * cfg.get("grid_mult", 1)
    dev = A.device
    if stream_k:
        # fp32 partial workspace: one private tile-sized slot per CTA (a CTA
        # publishes at most one partial per launch). int32 arrival counters:
        # one per stream-K tile slot; the kernel returns the counters to zero
        # before it exits, so a single zero-initialized allocation suffices.
        ws = torch.zeros(max_active_clusters * tile[0] * tile[1],
                         dtype=torch.float32, device=dev)
        fl = torch.zeros((1 + sk_extra) * max_active_clusters,
                         dtype=torch.int32, device=dev)
    else:
        ws = torch.zeros(8, dtype=torch.float32, device=dev)
        fl = torch.zeros(8, dtype=torch.int32, device=dev)
    mWS = from_dlpack(ws, assumed_align=16)
    mFL = from_dlpack(fl, assumed_align=16)
    mA = make_cute_view(A, dynamic)
    mB = make_cute_view(B, dynamic)
    mC = make_cute_view(C, dynamic)
    t0 = time.time()
    compiled = cute.compile(gemm, mA, mB, mC, mWS, mFL, max_active_clusters,
                            cuda_stream())
    print(f"    [compiled {cfg_tag(cfg)} in {time.time()-t0:.1f}s "
          f"ab_stage={gemm.ab_stage} epi_stage={gemm.epi_stage} epi_tile={gemm.epi_tile}]")
    # Keep the torch workspaces alive alongside the compiled kernel
    _compile_cache[key] = (compiled, mWS, mFL, ws, fl)
    return _compile_cache[key]


def parse_cfg(s):
    # Format: tile[:atom][:knob]... e.g. "128,128,64:2,2,1:s3:e4:w8:n:sk"
    # Knobs: sN=max ab stages, eN=epi stages, wN=swizzle size, n=raster along N,
    # gN=grid multiplier, oN=occupancy, rL,M=load/mma register budgets,
    # sk=stream-K scheduler, xN=extra full waves folded into the stream-K phase.
    parts = s.split(":")
    cfg = {"tile": tuple(int(x) for x in parts[0].split(","))}
    for p in parts[1:]:
        if p == "sk":
            cfg["stream_k"] = True
        elif p == "k8":
            cfg["mma_inst"] = (16, 8, 8)
        elif p.startswith("x"):
            cfg["sk_extra"] = int(p[1:])
        elif p.startswith("s"):
            cfg["stages"] = int(p[1:])
        elif p.startswith("e"):
            cfg["epi_stage"] = int(p[1:])
        elif p.startswith("w"):
            cfg["swizzle"] = int(p[1:])
        elif p.startswith("n"):
            cfg["raster_m"] = False
        elif p.startswith("g"):
            cfg["grid_mult"] = int(p[1:])
        elif p.startswith("o"):
            cfg["occupancy"] = int(p[1:])
        elif p.startswith("r"):
            lr, mr = p[1:].split(",")
            cfg["load_regs"] = int(lr)
            cfg["mma_regs"] = int(mr)
        else:
            cfg["atom"] = tuple(int(x) for x in p.split(","))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=str, required=True,
                    help="semicolon-separated M,N,K triples")
    ap.add_argument("--configs", type=str, required=True,
                    help="semicolon-separated tile[:atom][:sN] configs")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--dynamic", action="store_true",
                    help="dynamic-layout compile (one kernel for all shapes)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";") if s]
    configs = [parse_cfg(s) for s in args.configs.split(";") if s]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    all_results = []
    for (M, N, K) in shapes:
        print(f"shape M={M} N={N} K={K}")
        r = bench_shape(M, N, K, configs, args.warmup, args.iters,
                        use_graph=not args.no_graph, dynamic=args.dynamic)
        all_results.append(r)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    return all_results


if __name__ == "__main__":
    main()
