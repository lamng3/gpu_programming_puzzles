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

### Exercise
- [Vector Addition](/triton/vector_add_kernel.py)