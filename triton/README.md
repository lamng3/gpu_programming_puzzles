# 📝 GPU Programming Notes: Triton & Concepts

## Table of Contents

1. [References](#1-references)
2. [Exercises](#2-exercises)
3. [The Core Concept: SPMD](#3-the-core-concept-spmd)
4. [Memory Paradigm: Load & Store](#4-memory-paradigm-load--store)
   * [HBM vs. SRAM](#hbm-vs-sram)
5. [The Paradigm Shift: Scalar vs. Blocked](#5-the-paradigm-shift-scalar-vs-blocked)
   * [Why the change?](#why-the-change)
6. [1D vs 2D Operations](#6-1d-vs-2d-operations)
   * [The "Flattening" Trick](#why-this-works-the-flattening-trick)
7. [Reverse Kernel](#7-reverse-kernel)  
8. [Atomic Add vs Store](#8-tlatomic_add-vs-tlstore) 
    * [Map and Block Reduction Paradigm](#map-and-block-reduction-paradigm-maybe-allreduce)
    * [Writing Better Kernel To Avoid Locking](#writing-better-kernel-to-avoid-locking)

---

## 1. References

* [Triton Documentation: Motivations](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
* **[LAM1991]** [The Cache Performance and Optimizations of Blocked Algorithms](https://suif.stanford.edu/papers/lam-asplos91.pdf)
* **[FlashAttention]** [Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

---

## 2. Exercises

### 🧮 Fundamentals

* [x] [Vector Addition Kernel](/triton/vector_add_kernel.py)

### 🧩 2D Operations

* [x] [Matrix Addition Kernel](/triton/matrix_add_kernel.py)

### 🚚 Memory & Data Movement

### 🔽 Reductions & Aggregations

### 🧱 Sliding Windows & Convolutions

### 🧠 Attention & Advanced Kernels

* [ ] [Softmax Attention]()
* [ ] [Linear Self-Attention]()

### Notes
`tl.cdiv(x,div)` computes the ceiling divbison of `x` by `div`

---

## 3. The Core Concept: SPMD

The single **most important** concept in GPU programming is **SPMD** (Single Program, Multiple Data).

> **Definition:** You write **one** program (kernel), and the GPU launches thousands of instances of that program simultaneously, each processing different data.

### ⚡ Triton Boilerplate Example

In Triton, we handle this parallelism by calculating offsets based on the Program ID (`pid`).

```python
# 1. Get the unique ID for this specific program instance
pid = tl.program_id(0)

# 2. Skip ahead to find this instance's specific chunk of data
block_start = pid * BLOCK_SIZE

# 3. Create a vector of indices to grab specific items
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

**Breakdown:**

* `tl.program_id(axis)`: Returns the ID of the current program instance (similar to a block index).
* `offsets`: The list of indices telling this kernel instance exactly which data to touch.

---

## 4. Memory Paradigm: Load & Store

Consider adding two 1D vectors `a` and `b`. In standard CPU Python, we iterate sequentially:

```python
# CPU approach (Sequential)
c = [0] * n
for i in range(n):
    c[i] = a[i] + b[i]
```

On a GPU, we want to process this in parallel (“GPU go brrr”).
However, performance is dominated not by compute, but by **memory hierarchy**.

---

### HBM vs. SRAM

We can simplify the GPU into two major components:

| Component | Type                    | Characteristics                    | Role in Optimization |
| --------- | ----------------------- | ---------------------------------- | -------------------- |
| **HBM**   | High Bandwidth Memory   | Huge capacity, high latency (slow) | Main storage         |
| **SRAM**  | Shared / On-chip Memory | Tiny capacity, very low latency    | Scratchpad / cache   |

**The Golden Rule of Triton & FlashAttention:**

> Load data from **HBM** → do as much math as possible in **SRAM** → store back to **HBM**

Minimizing trips to HBM is how you achieve peak performance.

---

## 5. The Paradigm Shift: Scalar vs. Blocked

### History Lesson 📜

There is a fundamental difference between CUDA and Triton:

| Feature     | **CUDA Model**                  | **Triton Model**                |
| ----------- | ------------------------------- | ------------------------------- |
| Granularity | Scalar Program, Blocked Threads | Blocked Program, Scalar Threads |
| Focus       | Managing threads                | Managing blocks of data         |

---

### Why the change?

A seminal 1991 paper by [M. Lam et al.](https://suif.stanford.edu/papers/lam-asplos91.pdf) established the foundation behind Triton:

> *Programming paradigms based on **blocked algorithms** can facilitate the construction of high-performance compute kernels.*

**Key advantages of blocked algorithms:**

1. **Data locality** — keeps data in fast SRAM
2. **Compiler optimization** — enables aggressive fusion
3. **Sparsity handling** — skip unnecessary work at the block level

---

## 6. 1D vs 2D Operations

For **element-wise** operations, 2D Matrix addition is physically identical to 1D Vector addition.

Even though we mathematically write a matrix as $N \times N$, computer memory is linear—it's just a long strip of bytes.

### Why This Works? The "Flattening" Trick
1. **Contiguous Memory:** Standard PyTorch tensors are stored in "Row-Major" order. 
    - Row 0 is stored first, immediately followed by Row 1, then Row 2, etc.
2. **No Gaps:** Because there are no gaps between the end of Row 0 and the start of Row 1, the computer sees one giant array of length $N^2$.

*By treating it as 1D, you save yourself the headache of calculating row/column indices (pid_m, pid_n) and strides (stride_m, stride_n).*

---

## 7. Reverse Kernel

The task here is to reverse kernel in place. The idea is to select a midpoint and swap left and right portions.

See code here [Reverse Kernel](/triton/reverse_kernel.py)

### Program Analysis

Let `pid = 2` and `N = 6000`
- `block_start` = 2 * 1024 = 2048
- `offsets` = 2048 + [0, 1, 2, ..., 1023] = [2048, 2049, ..., 3071]
- `mask` = [True .. True(2999) False(3000) .. False] 
- `mid = 3000` and `N-1 = 5999`
- `left_offsets` = offsets = [2048, 2049, ..., 3071]
- `right_offsets` = [3951, 3950, ..., 2928]

```python
# left and right pointers
left_ptr = input_ptr + left_offsets
right_ptr = input_ptr + right_offsets

# load
L = tl.load(left_ptr, mask=mask)
R = tl.load(right_ptr, mask=mask)

# store
tl.store(left_ptr, R, mask=mask)
tl.store(right_ptr, L, mask=mask)
```
We can see that the `right_offsets` is already flipping the memory. So when we load R, it will be flipped. And when we store L into `right_ptr`, L will also be flipped.

### Would overlapping memory cause an issue in offset?
As you can see, there might be some overlapping memory spaces between `left_offsets` and `right_offsets` in the example above (e.g. [2928 .. 3071]). So would this cause any issue?

Thanks to safety mechanism in `mask`, it does not cause an issue, as we set `mask = offsets < N // 2`.

---

## 8. `tl.atomic_add()` vs `tl.store()`

The problem of interest in this case is counting the number of elements equal to K inside an array.

See code here [Count Array Element](/triton/count_equal_kernel.py)

The question is how we differentiate between `atomic_add()` and `store()`. Apparently, using `store()` would cause the problem to fail.

### Overwriting and Accumulating
This boils down to **Overwriting (Destructive)** and **Accumulating (Constructive)**. Since the grid launches multiple blocks (e.g. Block 0, Block 1, ...), they all run in parallel and try to write to the **same address** (`output_ptr`).

1. `tl.store()` (The "Overwrite")
- **Behavior:** "Whatever is at this address, delete it and put my value there."
- With this, we create a **Race Condition**. Blocks will fight to write their value, and the "winner" is whoever finish last
- **Scenario**:
    - Block 0 find 5 matches
    - Block 1 find 3 matches
    - Total should be 8
- However, in `tl.store()`, the work done by Block 0 is lost, and the final result is 3

2. `tl.atomic_add()` (The "Accumulate")
- **Behavior:** "Check what is currently at this address, add my value to it, and write the new total back. Do not let anyone else touch this address while I am doing this."
- This locks memory address for a brief moment to ensure the math is correct.

#### Will `tl.atomic_add()` kills parallelism as it requires **locking/serialization**?
No, not significantly because I am doing a Block Reduction first.

However, if used incorrectly, it will destroy the performance. 

In my code, I am putting heavy lifting to summing part. So 99.9% of the work is in calculating. The remaining is to send 1 single request to global memory to add its partial result.

#### The case when parallelism is killed with `tl.atomic_add()`
```
# BAD CODE: Kills Parallelism
# Every single matching thread tries to lock the SAME memory address.
match_k = tl.where((input == K), 1, 0)
if match_k == 1:
    tl.atomic_add(output_ptr, 1)
```
If we skipped the `tl.sum()` and tried to atomic add for every single match, we would kill parallelism.
- **Scenario:** 1,000,000 elements.
- **Total Atomics:** Up to 1,000,000.
- **Result:** All 1 million threads get into a single file line to update that one memory address. The GPU effectively becomes a single-threaded CPU for the duration of those writes. This is called **High Contention**.

#### Is it a norm in practice to use `tl.atomic_add()`?
Yes, `tl.atomic_add()` is absolutely a norm in practice, but it is typically used as the **"Last Mile"** strategy.

In high-performance GPU programming (CUDA/Triton), reduction usually follows a strict hierarchy. `tl.atomic_add ()` sits at the very top of that hierarchy.

### Two-Pass Reduction (Split Accumulation)

Tree Reduction or Two-Pass Reduction

coming soon. avoiding atomic add. especially in flaoting point addition because it is non-associative.

### High Contention (The "Hot Spot" Problem)

coming soon. If grid is massive and they all hitting same address. Again, we use Two-Pass Reduction. More to this later.

### Map and Block Reduction Paradigm (maybe AllReduce?)

coming soon.

### Writing Better Kernel To Avoid Locking

coming soon.