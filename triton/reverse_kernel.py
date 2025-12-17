"""
created by: Nathan N
url: https://leetgpu.com/challenges/reverse-array
"""
import torch
import triton
import triton.language as tl

@triton.jit
def reverse_kernel(
    input_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
): 
    """
    program analysis
        let pid = 2 and N = 6000
        block_start = 2 * 1024 = 2048
        offsets = 2048 + [0, 1, 2, ..., 1023]
                = [2048, 2049, ..., 3071]
        mid = 3000 and N-1 = 5999
        left_offsets = offsets = [2048, 2049, ..., 3071]
        right_offsets = [3951, 3950, ..., 2928]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # swap at midpoint
    # creates a strict boundary at midpoint
    mid = N // 2
    mask = offsets < mid

    # left and right offsets
    left_offsets = offsets
    right_offsets = N - 1 - left_offsets

    # left and right pointers
    left_ptr = input_ptr + left_offsets
    right_ptr = input_ptr + right_offsets

    # load
    L = tl.load(left_ptr, mask=mask)
    R = tl.load(right_ptr, mask=mask)

    # store
    tl.store(left_ptr, R, mask=mask)
    tl.store(right_ptr, L, mask=mask)

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    # cdiv(x,div) computes ceiling division of x by div
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)
    
    reverse_kernel[grid](
        input,
        N,
        BLOCK_SIZE
    ) 