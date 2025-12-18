"""
created by: Nathan N
url: https://leetgpu.com/challenges/matrix-multiplication

[TODO] revisit
In this problem, we need to define BLOCK_SIZE to avoid "Straw" problem: 
    GPU spends 99% on waiting for data and 1% on doing math
To do so, we modify the kernel to process Blocks of data.
Remember, in Triton, we general think in "Blocks of Data" rather than individual threads
"""
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:,None] * stride_am
    b_ptrs = b_ptr + offs_k[None,:] * stride_bk

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(N):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += a * b
        a_ptrs += stride_an
        b_ptrs += stride_bn

    c_ptrs = c_ptr + offs_m[:,None] * stride_cm + offs_k[None,:] * stride_ck
    mask = (offs_m[:,None] < M) & (offs_k[None,:] < K)
    
    tl.store(c_ptrs, acc, mask=mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1 
    stride_bn, stride_bk = K, 1  
    stride_cm, stride_ck = K, 1
    
    BLOCK_SIZE = 32
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(K, BLOCK_SIZE)) 
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_SIZE_M = BLOCK_SIZE,
        BLOCK_SIZE_K = BLOCK_SIZE
    )