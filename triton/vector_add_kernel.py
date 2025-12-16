"""
created by: Nathan N
url: https://leetgpu.com/challenges/vector-addition
"""
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # important concept in GPU programming: SPMD (Single Program, Multiple Data)

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    output = a + b
    tl.store(c_ptr + offsets, output, mask=mask)
   
# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    # launch with a block size of BLOCK_SIZE
    # this will create N/BLOCK_SIZE programs
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)