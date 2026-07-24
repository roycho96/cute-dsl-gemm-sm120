# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from typing import Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Blackwell Geforce architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell MMA for matrix multiply-accumulate (MMA) operations
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using Blackwell MMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Blackwell MMA instructions operate as follows:
- Read matrix A from registers
- Read matrix B from registers
- Perform MMA operation and store the result in Accumulator(register)

To run this example:

.. code-block:: bash

    python examples/blackwell_geforce/dense_gemm.py                                   \
      --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                 \
      --a_dtype Float16 --b_dtype Float16                                 \
      --c_dtype Float16 --acc_dtype Float32                               \
      --a_major k --b_major k --c_major n

The above example command compute batched gemm with M=8192, N=8192, K=8192,
batch_count=1. The tile shape is 128x256x64 and the cluster shape is (1,1).
The input, mma accumulator and output data type are set as fp16, fp32
and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell_geforce/dense_gemm.py                               \
      --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                 \
      --a_dtype Float16 --b_dtype Float16                                 \
      --c_dtype Float16 --acc_dtype Float32                               \
      --a_major k --b_major k --c_major n

Constraints:
* Supported input data types: fp16, bf16
* For fp16 types, A and B must have the same data type
* Only fp32 accumulation is supported in this example
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, respectively.
* OOB tiles are not allowed when TMA store is disabled
"""


# /////////////////////////////////////////////////////////////////////////////
#  Helpers to parse args
# /////////////////////////////////////////////////////////////////////////////
def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of MxNxKxL GEMM on Blackwell Geforce.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mnk",
        type=parse_comma_separated_ints,
        choices=[
            (64, 64, 64),
            (64, 128, 64),
            (128, 64, 64),
            (128, 128, 64),
            (128, 256, 64),
            (128, 128, 128),
        ],
        default=(64, 64, 64),
        help="CTA tile shape (comma-separated)",
    )
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        default=False,
        help="Skip reference checking",
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    return args


# /////////////////////////////////////////////////////////////////////////////
#  Host setup and device kernel launch
# /////////////////////////////////////////////////////////////////////////////


# Requested epilogue stage count, held in a module-level cell so the static
# _compute_stages helper can read it without changing its signature.
_EPI_STAGE_REQ = [8]


class Sm120GemmKernel:
    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
        atom_layout=(2, 2, 1),  # MMA atom layout (M, N, K); determines the number of MMA warps
        occupancy=1,  # target CTAs per SM; divides the per-CTA shared memory budget
        max_ab_stage=None,  # cap on the A/B pipeline depth (None = fill available smem)
        load_regs=40,  # register budget for the DMA (TMA load) warp
        mma_regs=232,  # register budget for each MMA warp
        epi_stage_req=8,  # requested epilogue smem stages; fewer stages free smem for A/B
        swizzle_size=1,  # tile-scheduler swizzle width, groups CTA tiles for L2 reuse
        raster_along_m=True,  # rasterize CTA tiles along M if True, along N otherwise
        stream_k=False,  # use the stream-K scheduler instead of the data-parallel one
        sk_extra_waves=0,  # extra full waves folded into the stream-K phase
        mma_inst_mnk=(16, 8, 16),  # mma.sync instruction shape; m16n8k16 is the widest
    ):
        self.epi_stage_req = epi_stage_req
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        # Stream-K: distribute the tile/k-tile work units of the last partial wave
        # (plus sk_extra_waves full waves) evenly over all persistent CTAs. Output
        # tiles whose k range is computed by more than one CTA are combined through
        # an fp32 workspace with a per-tile semaphore; the CTA owning the k=0 chunk
        # performs the reduction and the epilogue. This removes the wave-quantization
        # tail that a pure data-parallel persistent schedule suffers whenever the
        # tile count is not a multiple of the resident CTA count.
        self.stream_k = stream_k
        self.sk_extra_waves = sk_extra_waves
        if stream_k and swizzle_size != 1:
            # The stream-K index math relies on the exact (unpadded) tile layout,
            # which the scheduler only guarantees for swizzle_size == 1.
            raise ValueError("stream_k requires swizzle_size == 1")
        if mma_inst_mnk not in ((16, 8, 16), (16, 8, 8)):
            raise ValueError("mma_inst_mnk must be (16,8,16) or (16,8,8)")
        self.mma_inst_mnk = tuple(mma_inst_mnk)
        self.acc_dtype = acc_dtype
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.tiled_mma = None
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = occupancy
        self.max_ab_stage = max_ab_stage
        self.atom_layout = tuple(atom_layout)
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = load_regs
        self.mma_register_requirement = mma_regs

    def _setup_attributes(self):
        # Instruction shape is a constructor knob; m16n8k16 is the widest f16/bf16
        # mma.sync on sm_120 and is strictly preferable (m16n8k8 doubles the issue
        # count for the same math). The knob exists to document that choice, not
        # because a faster alternative is available on this architecture.
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            # The x2 factor on N pairs two n8 MMA value tiles so each warp's B
            # fragment spans a 16x16 smem block per k-block. That is exactly the
            # footprint of one ldmatrix.x4, so the smem-to-register copies below
            # (LdMatrix8x8x16bOp with num_matrices=4) compile to full-width
            # LDSM.16.M88.4 for B as well as A, for any atom_layout N extent
            # (including atom_layout[1] == 1). Verified in SASS: all emitted
            # LDSM instructions are .4, none .1/.2.
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=False
        )

        # Compute stage before compute smem layout
        # Publish the requested epilogue stage count for _compute_stages
        _EPI_STAGE_REQ[0] = self.epi_stage_req
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        # Optionally cap the A/B pipeline depth below what shared memory allows
        if self.max_ab_stage is not None:
            self.ab_stage = min(self.ab_stage, self.max_ab_stage)

        if self.ab_stage <= 0:
            raise RuntimeError("ab_stage == 0: not enough shared memory for this config")

        if self.stream_k:
            # Two structural requirements for stream-K:
            # 1. Deadlock freedom: a tile owner spin-waits on partials from
            #    other CTAs of the same launch, so the grid must not exceed
            #    the number of simultaneously resident CTAs.
            # 2. One CTA per SM: with two CTAs of this kernel co-resident on
            #    an SM, executing the stream-K fixup path corrupts in-flight
            #    operand fragments of the sibling CTA (observed on sm_120;
            #    the launch-time register allocation does not reserve the
            #    setmaxnreg targets, and only fully release-ordered workspace
            #    writes -- which cost more than stream-K saves -- avoid it).
            # Both are guaranteed by requiring the per-CTA shared memory
            # footprint to exceed half the SM's capacity: the hardware then
            # never co-schedules two CTAs, and the grid of one CTA per SM
            # (max_active_clusters at occupancy 1) is always fully resident.
            ab_bytes = (
                cute.size(cute.slice_(self.tile_shape_mnk, (None, 0, None)))
                + cute.size(cute.slice_(self.tile_shape_mnk, (0, None, None)))
            ) * 2  # 16-bit operands
            epi_bytes = (
                cute.size(self.epi_tile) * self.c_dtype.width // 8 * self.epi_stage
            )
            smem_used = self.ab_stage * ab_bytes + epi_bytes + 2048
            if 2 * smem_used <= self.smem_capacity:
                raise ValueError(
                    "stream_k requires one CTA per SM: smem/CTA "
                    f"{smem_used} B must exceed half of {self.smem_capacity} B; "
                    "use more mainloop/epilogue stages"
                )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sk_partials: cute.Tensor,
        sk_flags: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param sk_partials: fp32 workspace for stream-K partial accumulators
            (zero-initialized; one tileM*tileN block per stream-K tile slot).
            A 1-element dummy tensor is accepted when stream_k is disabled.
        :type sk_partials: cute.Tensor
        :param sk_flags: int32 per-tile arrival counters for stream-K reduction
            (zero-initialized). A 1-element dummy when stream_k is disabled.
        :type sk_flags: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")
        if cutlass.const_expr(self.b_dtype.width != 16 and self.b_dtype.width != 8):
            raise TypeError("b_dtype should be float16 or float8")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid_inst(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )
        if cutlass.const_expr(self.stream_k):
            # Stream-K requires every CTA to be co-resident: the tile owner spins
            # on partial results produced by other CTAs of the same launch. Launch
            # exactly the persistent CTA count (max_active_clusters is computed for
            # this kernel's occupancy target), never more.
            grid = (max_active_clusters, 1, 1)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            sk_partials,
            sk_flags,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        sk_partials: cute.Tensor,
        sk_flags: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        # bidx, bidy, bidz = cute.arch.block_idx()
        # bdimx, bdimy, bdimz = cute.arch.grid_dim()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        #  Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        #  Make frangments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # cluster wait for barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Create the tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        if cutlass.const_expr(self.stream_k):
            # Stream-K work decomposition, computed identically by the DMA and
            # MMA warps. The tile space is split into a data-parallel region of
            # full waves followed by a stream-K region covering the remainder
            # wave (plus sk_extra_waves full waves). The stream-K region is
            # flattened into (tile, k-tile) work units and divided evenly over
            # all CTAs, so no CTA idles through a partial tail wave.
            bid_sk = cutlass.Int32(cute.arch.block_idx()[0])
            grid_sk = cutlass.Int32(cute.arch.grid_dim()[0])
            total_tiles = cutlass.Int32(
                cute.size(tile_sched_params.problem_layout_ncluster_mnl)
            )
            full_waves = total_tiles // grid_sk
            extra_waves = cutlass.Int32(self.sk_extra_waves)
            if extra_waves > full_waves:
                extra_waves = full_waves
            sk_tile_cnt = total_tiles - full_waves * grid_sk + extra_waves * grid_sk
            dp_tile_cnt = total_tiles - sk_tile_cnt
            sk_unit_cnt = sk_tile_cnt * k_tile_cnt
            sk_unit_begin = (bid_sk * sk_unit_cnt) // grid_sk
            sk_unit_end = ((bid_sk + 1) * sk_unit_cnt) // grid_sk

        # Create the pipeline states for producer and consumer
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # ///////////////////////////////////////////////////////////////////////////////
            # Copy Atom A/B retiling for TMA load A/B
            # ///////////////////////////////////////////////////////////////////////////////
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)

            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)

            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            if cutlass.const_expr(not self.stream_k):
                # Data-parallel persistent schedule: one CTA computes the full
                # k range of each of its tiles.
                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                    accumulators.fill(0.0)
                    mainloop_consumer_state = self._mma_consume_segment(
                        k_tile_cnt,
                        num_k_blocks,
                        mainloop_pipeline,
                        mainloop_consumer_state,
                        smem_tiled_copy_A,
                        smem_tiled_copy_B,
                        tCsA_copy_view,
                        tCsB_copy_view,
                        tCrA_copy_view,
                        tCrB_copy_view,
                        tCrA,
                        tCrB,
                        tiled_mma,
                        accumulators,
                    )
                    self._epilogue(
                        tidx, warp_idx, accumulators, sC, tma_atom_c, gC_mnl_slice
                    )
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                # End of while loop
            else:
                # Stream-K schedule, phase 1: data-parallel full waves.
                tile_lin = bid_sk
                while tile_lin < dp_tile_cnt:
                    work_info = tile_sched._get_current_work_for_linear_idx(tile_lin)
                    tile_coord_mnl = work_info.tile_idx
                    gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                    accumulators.fill(0.0)
                    mainloop_consumer_state = self._mma_consume_segment(
                        k_tile_cnt,
                        num_k_blocks,
                        mainloop_pipeline,
                        mainloop_consumer_state,
                        smem_tiled_copy_A,
                        smem_tiled_copy_B,
                        tCsA_copy_view,
                        tCsB_copy_view,
                        tCrA_copy_view,
                        tCrB_copy_view,
                        tCrA,
                        tCrB,
                        tiled_mma,
                        accumulators,
                    )
                    self._epilogue(
                        tidx, warp_idx, accumulators, sC, tma_atom_c, gC_mnl_slice
                    )
                    tile_lin = tile_lin + grid_sk

                # Stream-K schedule, phase 2: the remainder tiles are flattened
                # into (tile, k-tile) units and divided evenly over all CTAs.
                # Each CTA walks its contiguous unit range one tile segment at
                # a time. The CTA holding a tile's k=0 segment owns the tile:
                # it merges the other CTAs' partial sums and runs the epilogue.
                unit = sk_unit_begin
                while unit < sk_unit_end:
                    tile_rel = unit // k_tile_cnt
                    k_begin = unit - tile_rel * k_tile_cnt
                    seg_cnt = k_tile_cnt - k_begin
                    units_left = sk_unit_end - unit
                    if seg_cnt > units_left:
                        seg_cnt = units_left

                    work_info = tile_sched._get_current_work_for_linear_idx(
                        dp_tile_cnt + tile_rel
                    )
                    tile_coord_mnl = work_info.tile_idx
                    accumulators.fill(0.0)
                    mainloop_consumer_state = self._mma_consume_segment(
                        seg_cnt,
                        num_k_blocks,
                        mainloop_pipeline,
                        mainloop_consumer_state,
                        smem_tiled_copy_A,
                        smem_tiled_copy_B,
                        tCsA_copy_view,
                        tCsB_copy_view,
                        tCrA_copy_view,
                        tCrB_copy_view,
                        tCrA,
                        tCrB,
                        tiled_mma,
                        accumulators,
                    )
                    if k_begin == 0:
                        if seg_cnt < k_tile_cnt:
                            # Tile owner with an incomplete k range: wait for
                            # the contributing CTAs, fold in their partials.
                            # Contributors occupy the consecutive block ids
                            # after the owner; the last one holds the tile's
                            # final k-tile unit.
                            last_contrib = (
                                grid_sk * ((tile_rel + 1) * k_tile_cnt) - 1
                            ) // sk_unit_cnt
                            self._sk_wait_and_reduce(
                                sk_partials,
                                sk_flags,
                                tile_rel,
                                tidx,
                                accumulators,
                                last_contrib - bid_sk,
                                bid_sk + 1,
                            )
                        gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                        self._epilogue(
                            tidx, warp_idx, accumulators, sC, tma_atom_c, gC_mnl_slice
                        )
                    else:
                        # Contributor: publish this segment's partial sums.
                        self._sk_commit_partials(
                            sk_partials,
                            sk_flags,
                            tile_rel,
                            tidx,
                            accumulators,
                            bid_sk,
                        )
                    unit = unit + seg_cnt
                # End of stream-K unit loop
        # End of MMA warp group
        # Start of DMA warp group
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            if cutlass.const_expr(not self.stream_k):
                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                    mainloop_producer_state = self._dma_load_segment(
                        0,
                        k_tile_cnt,
                        mainloop_pipeline,
                        mainloop_producer_state,
                        tma_atom_a,
                        tma_atom_b,
                        tAgA_mkl,
                        tBgB_nkl,
                        tAsA,
                        tBsB,
                        a_mcast_mask,
                        b_mcast_mask,
                    )
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                # end of while loop
            else:
                # Mirror of the MMA warps' stream-K schedule; both sides derive
                # the identical segment sequence from the same index arithmetic.
                tile_lin = bid_sk
                while tile_lin < dp_tile_cnt:
                    work_info = tile_sched._get_current_work_for_linear_idx(tile_lin)
                    tile_coord_mnl = work_info.tile_idx
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                    mainloop_producer_state = self._dma_load_segment(
                        0,
                        k_tile_cnt,
                        mainloop_pipeline,
                        mainloop_producer_state,
                        tma_atom_a,
                        tma_atom_b,
                        tAgA_mkl,
                        tBgB_nkl,
                        tAsA,
                        tBsB,
                        a_mcast_mask,
                        b_mcast_mask,
                    )
                    tile_lin = tile_lin + grid_sk

                unit = sk_unit_begin
                while unit < sk_unit_end:
                    tile_rel = unit // k_tile_cnt
                    k_begin = unit - tile_rel * k_tile_cnt
                    seg_cnt = k_tile_cnt - k_begin
                    units_left = sk_unit_end - unit
                    if seg_cnt > units_left:
                        seg_cnt = units_left

                    work_info = tile_sched._get_current_work_for_linear_idx(
                        dp_tile_cnt + tile_rel
                    )
                    tile_coord_mnl = work_info.tile_idx
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                    mainloop_producer_state = self._dma_load_segment(
                        k_begin,
                        seg_cnt,
                        mainloop_pipeline,
                        mainloop_producer_state,
                        tma_atom_a,
                        tma_atom_b,
                        tAgA_mkl,
                        tBgB_nkl,
                        tAsA,
                        tBsB,
                        a_mcast_mask,
                        b_mcast_mask,
                    )
                    unit = unit + seg_cnt
                # End of stream-K unit loop

            # Wait A/B buffer empty
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @cute.jit
    def _mma_consume_segment(
        self,
        seg_k_cnt,
        num_k_blocks,
        mainloop_pipeline,
        mainloop_consumer_state,
        smem_tiled_copy_A,
        smem_tiled_copy_B,
        tCsA_copy_view,
        tCsB_copy_view,
        tCrA_copy_view,
        tCrB_copy_view,
        tCrA,
        tCrB,
        tiled_mma,
        accumulators,
    ):
        """Consume ``seg_k_cnt`` k-tiles from the mainloop pipeline into the
        accumulators. The software-pipelined structure (smem-to-register copy of
        the next k-block overlapped with the MMA of the current one) matches the
        original inline mainloop; only the trip count is parameterized so the
        stream-K schedule can consume partial k ranges.

        :return: The advanced consumer pipeline state.
        """
        mainloop_consumer_state.reset_count()

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_state.count < seg_k_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_state
            )

        #  Wait for TMA copies to complete
        mainloop_pipeline.consumer_wait(mainloop_consumer_state, peek_ab_full_status)
        # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsB_p: (MMA, (4, MMA_N / 4), MMA_K)
        tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
        tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
        cute.copy(
            smem_tiled_copy_A,
            tCsA_p[None, None, 0],
            tCrA_copy_view[None, None, 0],
        )
        cute.copy(
            smem_tiled_copy_B,
            tCsB_p[None, None, 0],
            tCrB_copy_view[None, None, 0],
        )

        for k_tile in range(0, seg_k_cnt - 1, 1, unroll=1):
            # unroll the loop
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1

                if k_block_idx == num_k_blocks - 1:
                    mainloop_pipeline.consumer_release(mainloop_consumer_state)
                    mainloop_consumer_state.advance()

                    peek_ab_full_status = cutlass.Boolean(1)
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                    tCsA_p = tCsA_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]
                    tCsB_p = tCsB_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]
                    mainloop_pipeline.consumer_wait(
                        mainloop_consumer_state, peek_ab_full_status
                    )

                # Copy data from smem to tCrA/tCrB for the next k_block
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, k_block_next],
                    tCrA_copy_view[None, None, k_block_next],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, k_block_next],
                    tCrB_copy_view[None, None, k_block_next],
                )
                # Gemm of the current k_block
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA[None, None, k_block_idx],
                    tCrB[None, None, k_block_idx],
                    accumulators,
                )
        # end of for loop
        # Hoist out last k_tile
        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
            k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1

            if k_block_idx == num_k_blocks - 1:
                mainloop_pipeline.consumer_release(mainloop_consumer_state)
                mainloop_consumer_state.advance()

            if k_block_next > 0:
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, k_block_next],
                    tCrA_copy_view[None, None, k_block_next],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, k_block_next],
                    tCrB_copy_view[None, None, k_block_next],
                )
            # Gemm of the current k_block
            cute.gemm(
                tiled_mma,
                accumulators,
                tCrA[None, None, k_block_idx],
                tCrB[None, None, k_block_idx],
                accumulators,
            )
        return mainloop_consumer_state

    @cute.jit
    def _dma_load_segment(
        self,
        k_begin,
        seg_k_cnt,
        mainloop_pipeline,
        mainloop_producer_state,
        tma_atom_a,
        tma_atom_b,
        tAgA_mkl,
        tBgB_nkl,
        tAsA,
        tBsB,
        a_mcast_mask,
        b_mcast_mask,
    ):
        """Issue TMA loads for k-tiles ``[k_begin, k_begin + seg_k_cnt)`` of the
        current output tile into the mainloop pipeline.

        :return: The advanced producer pipeline state.
        """
        for k_tile in range(0, seg_k_cnt, 1, unroll=1):
            # Wait for the A/B buffers to be empty before loading into them;
            # this also sets the transaction barrier for the A/B buffers.
            mainloop_pipeline.producer_acquire(mainloop_producer_state)

            tAgA_k = tAgA_mkl[(None, k_begin + k_tile)]
            tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

            tBgB_k = tBgB_nkl[(None, k_begin + k_tile)]
            tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

            cute.copy(
                tma_atom_a,
                tAgA_k,
                tAsA_pipe,
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                    mainloop_producer_state
                ),
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_b,
                tBgB_k,
                tBsB_pipe,
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                    mainloop_producer_state
                ),
                mcast_mask=b_mcast_mask,
            )
            # Mainloop pipeline's producer commit is a NOP
            mainloop_pipeline.producer_commit(mainloop_producer_state)
            mainloop_producer_state.advance()
        return mainloop_producer_state

    @cute.jit
    def _epilogue(self, tidx, warp_idx, accumulators, sC, tma_atom_c, gC_mnl_slice):
        """Convert the fp32 accumulators to the output type and store the tile:
        stmatrix to smem, then TMA store to global memory, subtile by subtile.
        """
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, self.tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        # Allocate D registers.
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tcgc_for_tma_partition,
        )

        epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
        epi_tile_shape = tcgc_for_tma_partition.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(1, epi_tile_shape[0])
        )

        # Initialize tma store pipeline
        tma_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_mma_warps * self.num_threads_per_warp,
        )
        tma_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=tma_store_producer_group,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy from accumulators to D registers
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            # Type conversion
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            # Register to shared memory
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s,
                tRS_rD_out,
                tRS_sD[(None, None, None, epi_buffer)],
            )

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            # barrier for sync
            self.epilog_sync_barrier.arrive_and_wait()

            # Get the global memory coordinate for the current epi tile.
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                tma_store_pipeline.producer_commit()
                tma_store_pipeline.producer_acquire()

        tma_store_pipeline.producer_tail()

    @cute.jit
    def _sk_commit_partials(
        self, sk_partials, sk_flags, tile_slot, tidx, accumulators, bid
    ):
        """Stream-K contributor path: publish this CTA's partial accumulators
        for one output tile, then signal the tile owner.

        A CTA's contiguous unit range contains at most one segment that starts
        mid-tile, so each CTA writes at most one partial per launch and owns a
        private workspace slot indexed by its block id. The partial is written
        with plain stores in per-thread fragment order (every CTA maps the
        same (tidx, v) pair to the same tile element, so no layout translation
        is needed) and published with a release-ordered flag increment. Plain
        stores are used deliberately: relaxed global reductions are
        fire-and-forget on sm_120 and are not reliably ordered by a later
        fence or release operation, which corrupts the reduction; ordinary
        stores are correctly covered by the release flag.
        """
        frag_size = cute.size(accumulators.shape)
        num_threads = self.num_mma_warps * self.num_threads_per_warp
        base = (bid * num_threads + tidx) * frag_size
        for v in cutlass.range_constexpr(frag_size):
            sk_partials[base + v] = accumulators[v]
        # Release ordering: every thread fences its stores, the barrier
        # collects them CTA-wide, and one release-ordered flag increment
        # publishes the whole partial to the owner.
        cute.arch.fence_acq_rel_gpu()
        self.epilog_sync_barrier.arrive_and_wait()
        if tidx == 0:
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_add(
                sk_flags.iterator + tile_slot,
                cutlass.Int32(1),
                sem="release",
                scope="gpu",
            )

    @cute.jit
    def _sk_wait_and_reduce(
        self,
        sk_partials,
        sk_flags,
        tile_slot,
        tidx,
        accumulators,
        expected_contribs,
        first_contrib_bid,
    ):
        """Stream-K owner path: wait until every contributing CTA has
        published its partial for this tile, then fold the partials into the
        local accumulators.

        Contributors to a tile occupy consecutive block ids directly after the
        owner, so the owner reads ``expected_contribs`` private slots starting
        at ``first_contrib_bid``. Nothing needs to be cleared afterwards: the
        slots are plain-overwritten by their owning CTAs on the next launch,
        and the arrival counter is returned to zero here.
        """
        if tidx == 0:
            arrived = cute.arch.atomic_add(
                sk_flags.iterator + tile_slot,
                cutlass.Int32(0),
                sem="acquire",
                scope="gpu",
            )
            while arrived < expected_contribs:
                arrived = cute.arch.atomic_add(
                    sk_flags.iterator + tile_slot,
                    cutlass.Int32(0),
                    sem="acquire",
                    scope="gpu",
                )
            # Reset the counter for the next launch; all contributions for
            # this tile are complete, so no other CTA touches this slot again.
            cute.arch.atomic_exch(
                sk_flags.iterator + tile_slot, cutlass.Int32(0), scope="gpu"
            )
            cute.arch.fence_acq_rel_gpu()
        self.epilog_sync_barrier.arrive_and_wait()
        # Acquire ordering: the polling thread's fence orders the observed
        # flag ahead of the barrier, and each reading thread fences afterwards
        # so its own loads observe the contributors' stores.
        cute.arch.fence_acq_rel_gpu()

        frag_size = cute.size(accumulators.shape)
        num_threads = self.num_mma_warps * self.num_threads_per_warp
        for c in range(0, expected_contribs, 1, unroll=1):
            cbase = ((first_contrib_bid + c) * num_threads + tidx) * frag_size
            for v in cutlass.range_constexpr(frag_size):
                # Strong relaxed load (ld.relaxed.gpu.global): the value must
                # come from the L2 point of coherence. A weak load can be
                # served stale from the reader's L1 on sm_120 even after an
                # acquire fence, and is also free to be scheduled ahead of the
                # arrival poll; a relaxed strong load has neither problem and,
                # unlike an atomic read-modify-write, still pipelines freely.
                part = cute.arch.load(
                    sk_partials.iterator + (cbase + v),
                    self.acc_dtype,
                    sem="relaxed",
                    scope="gpu",
                )
                accumulators[v] = accumulators[v] + part

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """
        epi_stage = _EPI_STAGE_REQ[0]  # requested epilogue stage count (default 8)
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_dtype: type[cutlass.Numeric],
        b_layout: cute.Layout,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: cute.Layout,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout for matrix A
        :type a_layout: Layout
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout for matrix B
        :type b_layout: Layout
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: leading dimension of the output matrix C
        :type c_layout: Layout
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    def _compute_grid_inst(self, c, tile_shape_mnk, max_active_clusters):
        # Instance-level wrapper forwarding the scheduler knobs to the static helper
        return self._compute_grid(
            c, tile_shape_mnk, max_active_clusters,
            self.swizzle_size, self.raster_along_m,
        )

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
        swizzle_size: int = 1,
        raster_along_m: bool = True,
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param swizzle_size: Number of CTA tiles grouped per swizzle step for L2 reuse.
        :type swizzle_size: int
        :param raster_along_m: Rasterize CTA tiles along M if True, along N otherwise.
        :type raster_along_m: bool

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl,
            swizzle_size=swizzle_size, raster_along_m=raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor


def run(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    tile_shape_mnk: Tuple[int, int, int],
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    import torch
    import cutlass.torch as cutlass_torch

    print("Running Blackwell Geforce Dense GEMM with:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Tile Shape: {tile_shape_mnk}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")

    a_dtype = getattr(cutlass, a_dtype) if isinstance(a_dtype, str) else a_dtype
    b_dtype = getattr(cutlass, b_dtype) if isinstance(b_dtype, str) else b_dtype
    c_dtype = getattr(cutlass, c_dtype) if isinstance(c_dtype, str) else c_dtype
    acc_dtype = getattr(cutlass, acc_dtype) if isinstance(acc_dtype, str) else acc_dtype

    # Unpack parameters
    m, n, k, l = mnkl
    cluster_shape_mnk = (1, 1, 1)

    # Prepare pytorch tensors: A, B (random from 0 to 2) and C (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major, a_dtype)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major, b_dtype)
    c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major, c_dtype)

    def create_cute_tensor(data_ref, cutlass_dtype):
        cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
            data_ref, cutlass_dtype, True, 16
        )

        if cutlass_dtype.is_float and cutlass_dtype.width == 8:
            f32_torch_tensor = data_ref.to(dtype=torch.float32)
            cute_tensor = cutlass_torch.convert_cute_tensor(
                f32_torch_tensor,
                cute_tensor,
                cutlass_dtype,
                is_dynamic_layout=True,
            )
        return cute_tensor, torch_tensor

    a_tensor, a_torch_gpu = create_cute_tensor(a_torch_cpu, a_dtype)
    b_tensor, b_torch_gpu = create_cute_tensor(b_torch_cpu, b_dtype)
    c_tensor, c_torch_gpu = create_cute_tensor(c_torch_cpu, c_dtype)

    # Dummy stream-K workspace (this entry point runs the data-parallel path)
    ws_torch = torch.zeros(8, dtype=torch.float32, device="cuda")
    fl_torch = torch.zeros(8, dtype=torch.int32, device="cuda")
    from cutlass.cute.runtime import from_dlpack

    ws_tensor = from_dlpack(ws_torch, assumed_align=16)
    fl_tensor = from_dlpack(fl_torch, assumed_align=16)

    gemm = Sm120GemmKernel(
        acc_dtype,
        tile_shape_mnk,
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mnk[0] * cluster_shape_mnk[1]
    )

    # Initialize stream
    stream = cutlass_torch.default_stream()
    # compile gemm kernel
    compiled_gemm = cute.compile(
        gemm, a_tensor, b_tensor, c_tensor, ws_tensor, fl_tensor,
        max_active_clusters, stream
    )

    if not skip_ref_check:
        print("Reference checking ...")
        # execution
        compiled_gemm(a_tensor, b_tensor, c_tensor, ws_tensor, fl_tensor, stream)
        torch.cuda.synchronize()

        # Ref check
        ref = torch.einsum(
            "mkl,nkl->mnl",
            a_torch_cpu.to(dtype=torch.float32),
            b_torch_cpu.to(dtype=torch.float32),
        )

        # Copy gpu tensor to cpu
        kernel_result = c_torch_gpu.cpu()

        # Convert ref to c_dtype
        _, ref_torch_gpu = create_cute_tensor(ref, c_dtype)
        ref_result = ref_torch_gpu.cpu()

        # Assert close results
        torch.testing.assert_close(
            kernel_result, ref_result, atol=tolerance, rtol=1e-03
        )

    def generate_tensors():
        a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major, a_dtype)
        b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major, b_dtype)
        c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major, c_dtype)
        mA_workspace, _ = create_cute_tensor(a_torch_cpu, a_dtype)
        mB_workspace, _ = create_cute_tensor(b_torch_cpu, b_dtype)
        mC_workspace, _ = create_cute_tensor(c_torch_cpu, c_dtype)
        return testing.JitArguments(
            mA_workspace, mB_workspace, mC_workspace, ws_tensor, fl_tensor, stream
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch_gpu.numel() * a_torch_gpu.element_size()
            + b_torch_gpu.numel() * b_torch_gpu.element_size()
            + c_torch_gpu.numel() * c_torch_gpu.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    print(f"Execution time: {exec_time} microseconds per iteration")

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":
    args = parse_arguments()
    run(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.tile_shape_mnk,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
