#!/usr/bin/env python
"""Run the DSL GEMM kernel a few times standalone (for ncu profiling)."""
import math
import sys

import torch

from bench import make_cute_view, cuda_stream, get_compiled, parse_cfg

M, N, K = (int(x) for x in sys.argv[1].split(","))
cfg = parse_cfg(sys.argv[2])

dev = torch.device("cuda")
A = (torch.randn(M, K, device=dev) / math.sqrt(K)).to(torch.bfloat16)
B = torch.randn(N, K, device=dev).to(torch.bfloat16)
C = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)

compiled, mWS, mFL, _, _ = get_compiled(cfg, A, B, C, dynamic=False)
mA, mB, mC = (make_cute_view(t, False) for t in (A, B, C))
for _ in range(3):
    compiled(mA, mB, mC, mWS, mFL, cuda_stream())
torch.cuda.synchronize()
print("done")
