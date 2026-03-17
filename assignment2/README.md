# SC4064 GPU Programming — Assignment 2: HPC Problem and CUDA Libraries

**Instructor:** Loke Yuan Ren (yrloke@ntu.edu.sg)
**College of Computing and Data Science, Nanyang Technological University**

**Due:** Friday, Week 8: March 17, 2026, 11:59 PM (late submissions will **not** be accepted)

---

## Objectives

1. Implement numerical PDE solvers on GPUs
2. Translate mathematical operators into computational forms
3. Apply CUDA libraries appropriately
4. Evaluate memory hierarchy optimizations
5. Visualize scientific computing results
6. Critically compare computational approaches

## Platform

- NVIDIA GPUs (ASPIRE 2A at NSCC Singapore provides NVIDIA A100 GPUs)
- Single-GPU programming; personal workstation with CUDA driver/toolkit is also acceptable
- Document your computing environment if not using ASPIRE 2A

---

## 1. Introduction

Implement and analyze a GPU-based solver for the **two-dimensional wave equation**:

1. Implement a custom CUDA stencil kernel
2. Implement an alternative version using CUDA libraries
3. Visualization
4. Conduct performance and bandwidth analysis

Focus: correctness + quantitative performance analysis + critical reasoning.

---

## 2. Mathematical Background

### 2D Wave Equation

$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$$

where $u(t, x, y)$ is the wave field, $c$ is the wave speed, $(x, y) \in \Omega = [0,1] \times [0,1]$, $t > 0$.

Solve on a square domain with **Dirichlet boundary conditions**.

### 2.1 Initial and Boundary Conditions

- **Initial displacement** (at $t=0$):
  $$u(0, x, y) = \sin(\pi x) \sin(\pi y)$$

- **Initial velocity** (at $t=0$):
  $$\frac{\partial u}{\partial t}(0, x, y) = 0$$

- **Dirichlet boundary condition**: $u(t, x, y) = 0$ for all points on the domain boundary and all times.

### 2.2 Finite Difference Discretization

5-point stencil scheme:

$$u_{i,j}^{n+1} = 2u_{i,j}^n - u_{i,j}^{n-1} + \frac{c^2 \Delta t^2}{\Delta x \Delta y} \left( u_{i+1,j}^n + u_{i-1,j}^n + u_{i,j+1}^n + u_{i,j-1}^n - 4u_{i,j}^n \right)$$

where $i, j$ are spatial indices, $n$ is the time index. Assume $\Delta x = \Delta y$. Then $\lambda^2 = \frac{c^2 \Delta t^2}{\Delta x \Delta y}$.

**Parameters** (may vary for resolution study):

- Wave speed: $c = 1.0$
- Grid spacing: $\Delta x = \Delta y = 0.01$
- Time step: $\Delta t = 0.005$

**Stability condition** (2D second-order finite difference):

$$\lambda \le \frac{1}{\sqrt{2}}$$

---

## 3. Implementation Requirements

### 3.1 Part A — Custom CUDA Kernel

Implement **two versions**:

#### (A1) Global Memory Version

- Direct implementation of the stencil
- No shared memory optimization

#### (A2) Shared Memory Tiled Version

- Use shared memory tiling
- Implement halo loading
- Handle boundary conditions correctly

**Requirements:**

- Use 2D thread blocks
- Justify your block size choice
- Measure kernel execution time using CUDA events

### 3.2 Part B — CUDA Library Implementation

Implement **at least one** of the following:

#### Option 1 — cuSPARSE (Recommended)

Reformulate the update as:

$$\mathbf{u}^{n+1} = 2\mathbf{u}^n - \mathbf{u}^{n-1} + \lambda^2 L \mathbf{u}^n$$

where $L$ is the discrete Laplacian matrix stored in CSR format.

**Requirements:**

- Construct the sparse Laplacian matrix
- Use `cusparseSpMV`
- Compare performance with the stencil implementation

#### Option 2 — cuBLAS

Construct a dense matrix $L$ (small grids only) and use `cublasDgemv`.

- Must explain why this approach is inefficient.

### 3.3 Wave Field Visualization

Export selected timesteps and generate:

- 2D heatmaps
- Surface plots
- Optional animation

### 3.4 Timing

Measure:

- Kernel time per timestep
- Total simulation time
- Library call time (if applicable)

### 3.5 Effective Memory Bandwidth

For the 5-point stencil:

- 5 reads, 1 write
- In double precision: $6 \times 8 = 48$ bytes per grid update

$$\text{Bandwidth} = \frac{\text{Total bytes transferred}}{\text{Kernel time}}$$

Purpose: evaluate GPU performance independently of numerical accuracy effects.

### 3.6 Baseline Configuration

Reference domain:

$$\Omega_0 = [0, 1] \times [0, 1]$$

with uniform grid spacing $\Delta x = \Delta y = \text{constant}$.

Time step $\Delta t$ must remain fixed to keep CFL number constant:

$$\lambda = \frac{c \Delta t}{\Delta x}$$

### 3.7 Scaling Strategy

For the performance scaling study:

- Keep $\Delta x$, $\Delta y$, and $\Delta t$ fixed
- Increase the physical domain size

Enlarged domains:

$$\Omega_k = [0, L_k] \times [0, L_k], \quad L_k = 1, 2, 4, 8$$

Number of grid points:

$$N_k = \frac{L_k}{\Delta x}$$

Enlarging the domain increases total workload while leaving the numerical scheme unchanged.

### 3.8 Experimental Procedure

For each domain size:

1. Run the solver for a fixed number of time steps
2. Measure total runtime
3. Compute effective memory bandwidth
4. Record GPU occupancy and throughput metrics

### 3.9 Important Remarks

- This is a **performance study**, not a resolution study
- The numerical accuracy does not change because $\Delta x$ is fixed
- The CFL condition remains unchanged
- Only the total computational workload increases

### 3.10 Analysis Questions

Students must discuss:

- How does runtime scale with total grid points?
- Does performance scale linearly with problem size?
- Is the kernel memory-bound or compute-bound?
- How does block size affect performance?

---

## 4. Marking Scheme

| Component                      | Weight |
| ------------------------------ | ------ |
| Correctness                    | 20%    |
| Kernel implementation quality  | 10%    |
| Library implementation         | 10%    |
| Visualization                  | 20%    |
| Performance analysis           | 25%    |
| Critical discussion            | 15%    |

---

## Submission

Submit via NTULearn as a single ZIP archive + a PDF report (report must NOT be in the ZIP).

1. **Source Code and Job Submission Script**
   One CUDA source file (`.cu`) with all implementations in different functions. Include sufficient comments. Also include job submission scripts and Makefiles.

2. **Visualization**
   Images in jpg or video in mp4.

3. **Report**
   A single report ($\le$ 3 pages) documenting analysis and reasoning.
