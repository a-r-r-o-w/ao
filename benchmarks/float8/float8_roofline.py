# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a script to estimate the benefit from converting a `torch.nn.Linear`
layer to float8, by estimating the difference in e2e GPU kernel time between:
1. bf16 gemms in fwd and bwd, and 
2. float8 gemms in fwd and bwd, and float8 overhead

The gemm times are estimated either from direct measurements via benchmarks,
or with a roofline estimation based on TOPS and peak compute bandwidth of an 
NVIDIA H100.

The float8 overhead times are estimated by counting memory reads and writes
based on the specified float8 scaling, and estimating that we can achieve
a certain % of machine peak memory bandwidth when performing these reads and writes.

Additional context:
1. the formulas for fwd/bwd gemms in a linear layer, with corresponding input
   and output sizes:

  input @ weight_t = output
  MxK @ KxN => MxN

  grad_output @ weight = grad_input
  MxN @ NxK => MxK

  input_t @ grad_output = grad_weight
  KxM @ MxN => KxN

2. we properly model the worst-case of the current torch.compile limitations regarding
   float8 scaling
3. assume for float8 activations/gradients that torch.compile will fuse to the
preceding op. Note that this is not always true in practice.
4. assume no AC (TODO model it)
5. assume no float8 all-gather (TODO model it)
"""

import csv
import copy
import time
from typing import Optional

import fire
import pandas as pd
import sympy

import torch
import torch.utils.benchmark as benchmark

from utils import get_name_to_shapes_iter, get_gpu_kernel_gemm_time_s
from torchao.float8.roofline_utils import (
    get_gemm_time_sympy,
    get_float8_mem_sympy,
)


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(4):
        f(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.blocked_autorange()
    return measurement.mean


# TODO cache it
def get_gemm_times(M, K, N, fast_accum):
    device = torch.device('cuda')

    # bf16 time
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device).t().contiguous().t()
    bf16_time_s = get_gpu_kernel_gemm_time_s(torch.mm, x_bf16, w_bf16)

    # f8 time
    d1, d2, d3 = torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16
    A = torch.zeros(M, K, device=device, dtype=d1)
    B = torch.zeros(K, N, device=device, dtype=d2).t().contiguous().t()
    scale_a = torch.tensor([1.0], device=device)
    scale_b = torch.tensor([1.0], device=device)

    def do_matmul(A, B):
        return torch._scaled_mm(
            A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=fast_accum
        )
    f8_time_s = get_gpu_kernel_gemm_time_s(do_matmul, A, B)

    return bf16_time_s, f8_time_s

def run(
    outfile: str,
    gemm_time_strategy: str = "benchmarks",
    model_torch_compile_limitations: bool = False,
    scaling_type_input: str = "dynamic",
    scaling_type_weight: str = "dynamic",
    scaling_type_grad_output: str = "dynamic",
    shape_gen_name: str = "square",
):
    """
    Args:
    * `gemm_time_strategy`:
      - `benchmarks`: use benchmarks for gemm times (more accurate for all shapes)
      - `roofline`: use roofline model for gemm times (only accurate for large shapes)
    * `model_torch_compile_limitations`: if True, adjust memory traffic estimates based
      on current limitations of torch.compile for float8 scaling/casting kernels.
    * `scaling_type_input`: `dynamic` or `delayed`
    * `scaling_type_weight`: `dynamic` or `delayed`
    * `scaling_type_grad_output`: `dynamic` or `delayed`
    * `shape_gen_name`: `llama`, `square`, or `sweep`
    """

    print(f'gemm_time_strategy: {gemm_time_strategy}')
    print(f'model_torch_compile_limitations: {model_torch_compile_limitations}')
    print(f'scaling_type_input: {scaling_type_input}')
    print(f'scaling_type_weight: {scaling_type_weight}')
    print(f'scaling_type_grad_output: {scaling_type_grad_output}')
    print(f'shape_gen_name: {shape_gen_name}')

    assert gemm_time_strategy in ("benchmarks", "roofline"), \
        "`gemm_time_strategy` must be 'benchmarks' or 'roofline'"

    M, K, N = sympy.symbols('M K N')

    fp8_mem_time_sympy = get_float8_mem_sympy(
        M, 
        K, 
        N, 
        model_torch_compile_limitations,
        scaling_type_input,
        scaling_type_weight,
        scaling_type_grad_output,
    )
    print()
    print('fp8_mem_time_sympy', fp8_mem_time_sympy)

    if gemm_time_strategy == "roofline":
        bf16_gemm_time_sympy = get_gemm_time_sympy(M, K, N, torch.bfloat16)
        print('bf16_gemm_time_sympy', bf16_gemm_time_sympy)
        fp8_gemm_time_sympy = get_gemm_time_sympy(M, K, N, torch.float8_e4m3fn)
        print('fp8_gemm_time_sympy', fp8_gemm_time_sympy)
        print()
    else:
        print()

    headers = [
        'fwd_M', 'fwd_K', 'fwd_N', 
        'bf16_gemm_s', 
        'fp8_gemm_s', 'fp8_overhead_s', 'fp8_total_s', 
        'speedup',
    ]
    results = []
    
    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, None, None, None)

    for name, (M_val, K_val, N_val) in name_to_shapes:
        if gemm_time_strategy == "benchmarks":
            bf16_g1, f8_g1 = get_gemm_times(M_val, K_val, N_val, True)
            bf16_g2, f8_g2 = get_gemm_times(M_val, N_val, K_val, False)
            bf16_g3, f8_g3 = get_gemm_times(K_val, M_val, N_val, False)
            bf16_time_val = bf16_g1 + bf16_g2 + bf16_g3
            fp8_gemm_time_s = f8_g1 + f8_g2 + f8_g3
            fp8_mem_time_s = fp8_mem_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
            fp8_time_val = fp8_gemm_time_s + fp8_mem_time_s
        else:
            assert gemm_time_strategy == "roofline", "unsupported"
            bf16_time_val = bf16_gemm_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
            fp8_gemm_time_s = fp8_gemm_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
            fp8_mem_time_s = fp8_mem_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
            fp8_time_val = fp8_gemm_time_s + fp8_mem_time_s

        results.append([
            M_val, K_val, N_val, 
            bf16_time_val, 
            fp8_gemm_time_s, fp8_mem_time_s, fp8_time_val, 
            bf16_time_val / fp8_time_val,
        ])

    df = pd.DataFrame(results, columns=headers)
    print(df)
    df.to_csv(outfile)
    print('done')

if __name__ == '__main__':
    fire.Fire(run)
