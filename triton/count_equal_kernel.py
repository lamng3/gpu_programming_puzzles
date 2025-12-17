"""
created by: Nathan N
url: https://leetgpu.com/challenges/count-array-element
"""
import torch
import triton
import triton.language as tl

@triton.jit
def count_equal_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # rewrite mask to consider K
    mask = offsets < N

    input = tl.load(input_ptr + offsets, mask=mask)

    # MAP
    match_k = tl.where((input == K), 1, 0)

    # BLOCK REDUCE
    k_count = tl.sum(match_k, 0)

    # using `tl.store(output_ptr, k_count)` will fail
    tl.atomic_add(output_ptr, k_count)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    count_equal_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)