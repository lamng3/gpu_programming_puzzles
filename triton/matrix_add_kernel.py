"""
created by: Nathan N
url: https://leetgpu.com/challenges/matrix-addition
"""
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # 2D N * N can be treated as 1D of N * N elements
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    output = a + b
    c = tl.store(c_ptr + offsets, output, mask=mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
