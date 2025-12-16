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

