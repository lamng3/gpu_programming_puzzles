# Notes

## GPU Programming Concept
Single **most important** concept in GPU programming:
> SPMD: Single Program Multiple Data

Example
```python
pid = tl.program_id(0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```
These 3 lines are needed because Triton launches many identical copies of your Kernel at the same time. 
- `triton.language.program_id(axis, semantic=None)`: Returns the id of the current program instance along the given axis
- `block_start = pid * BLOCK_SIZE`: Kernel skips ahead to find its specific chunk
- `offsets = block_start + tl.arange(0, BLOCK_SIZE)`: Kernel chooses which specific items to grab

## Load and Store
Take a look at an example of adding 2 1D vectors of same size `a` and `b`. In LeetCode, we would do this:
```python
c = [0] * n
for i in range(n):
    c[i] = a[i] + b[i]
```
Now, we have GPU, yay go brrrr. So the idea idea behind the use of `tl.load` and `tl.store` is that we want to process the computation of c in parallel. 

We can think GPU simply with 2 major components: 
- High Bandwidth Memory (HBM)
- Shared Memory (SRAM)
SRAM is blazing fast, while HBM is huge and slow. The goal of FlashAttention and Triton is to load data from HBM **once**, and do tons of math on SRAM before storing it back.

### History Lesson
Now some history lesson folks. Starting with Programming Model.
- CUDA Programming Model is *Scalar Program, Blocked Threads* 
- Triton Programming Model is *Blocked Program, Scalar Threads*

#### So why this change? 
Well, a paper by [M.Lam et al.](https://suif.stanford.edu/papers/lam-asplos91.pdf) in 1991 set the main premise of the whole Triton project:
> Programming paradigms based on **blocked algorithms** can facilitate the construction of high-performance compute kernels for neural networks.

**IMO** Data locality and shared memory:
- Minimize expensive global memory accesses, allowing compilers to aggressively fuse operations
- Enable efficient block-level processing, allowing compilers to aggressively optimize sparse operations by skipping unnecessary computations and maximizing memory bandwidth

### See More
- [Triton Motivations](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- [[LAM1991] The Cache Performance and Optimizations of Blocked Algorithms](https://suif.stanford.edu/papers/lam-asplos91.pdf)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

### Exercise
- [Vector Addition](/triton/vector_add_kernel.py)