// wave2d.cu — 2D Wave Equation Solver on GPU
// SC4064 Assignment 2
//
// Implements three approaches:
//   (A1) Global memory stencil kernel
//   (A2) Shared memory tiled stencil kernel
//   (B)  cuSPARSE SpMV-based solver
//
// Compile: nvcc -O2 -std=c++20 -ccbin g++-14 -lcusparse -o wave2d wave2d.cu

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// ── Error-checking macros ──────────────────────────────────────────────────

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUSPARSE(call)                                                   \
    do {                                                                       \
        cusparseStatus_t status = (call);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__,        \
                    __LINE__, (int)status);                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ── Physical parameters ───────────────────────────────────────────────────

static constexpr double C_WAVE = 1.0;
static constexpr double DX     = 0.01;
static constexpr double DY     = 0.01;
static constexpr double DT     = 0.005;
static constexpr double LAMBDA2 = (C_WAVE * C_WAVE * DT * DT) / (DX * DY);
// LAMBDA2 = 0.25, lambda = 0.5  (satisfies lambda <= 1/sqrt(2) ≈ 0.707)

static constexpr int NUM_STEPS = 200;  // total simulation time = 1.0 s

// ══════════════════════════════════════════════════════════════════════════
//  CUDA Kernels
// ══════════════════════════════════════════════════════════════════════════

// Initialize wave field: u(0,x,y) = sin(π x) sin(π y)
__global__ void initWaveField(double *u, int Nx, int Ny, double dx, double dy) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Ny && j < Nx) {
        double x = j * dx;
        double y = i * dy;
        u[i * Nx + j] = sin(M_PI * x) * sin(M_PI * y);
    }
}

// First time step: u1 = u0 + 0.5 λ² Lap(u0)   (since ∂u/∂t(0) = 0)
__global__ void waveFirstStep(double *u1, const double *u0,
                              int Nx, int Ny, double half_lambda2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1) {
        int idx = i * Nx + j;
        double lap = u0[idx - Nx] + u0[idx + Nx]
                   + u0[idx - 1]  + u0[idx + 1]
                   - 4.0 * u0[idx];
        u1[idx] = u0[idx] + half_lambda2 * lap;
    }
}

// ── (A1) Global memory stencil kernel ─────────────────────────────────────

__global__ void waveStepGlobal(double *u_next, const double *u_curr,
                               const double *u_prev,
                               int Nx, int Ny, double lambda2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1) {
        int idx = i * Nx + j;
        double lap = u_curr[idx - Nx] + u_curr[idx + Nx]
                   + u_curr[idx - 1]  + u_curr[idx + 1]
                   - 4.0 * u_curr[idx];
        u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + lambda2 * lap;
    }
}

// ── (A2) Shared memory tiled stencil kernel ───────────────────────────────

__global__ void waveStepShared(double *u_next, const double *u_curr,
                               const double *u_prev,
                               int Nx, int Ny, double lambda2) {
    extern __shared__ double tile[];
    const int tileW = blockDim.x + 2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int j  = blockIdx.x * blockDim.x + tx;
    const int i  = blockIdx.y * blockDim.y + ty;
    const int si = ty + 1;  // shared-memory row
    const int sj = tx + 1;  // shared-memory col

    // Load center element
    double center = 0.0;
    if (i < Ny && j < Nx) center = u_curr[i * Nx + j];
    tile[si * tileW + sj] = center;

    // Top halo
    if (ty == 0)
        tile[0 * tileW + sj] = (i > 0 && j < Nx)
                                    ? u_curr[(i - 1) * Nx + j] : 0.0;
    // Bottom halo
    if (ty == (int)blockDim.y - 1)
        tile[(blockDim.y + 1) * tileW + sj] =
            (i + 1 < Ny && j < Nx) ? u_curr[(i + 1) * Nx + j] : 0.0;
    // Left halo
    if (tx == 0)
        tile[si * tileW + 0] = (j > 0 && i < Ny)
                                    ? u_curr[i * Nx + (j - 1)] : 0.0;
    // Right halo
    if (tx == (int)blockDim.x - 1)
        tile[si * tileW + (int)blockDim.x + 1] =
            (j + 1 < Nx && i < Ny) ? u_curr[i * Nx + (j + 1)] : 0.0;

    __syncthreads();

    if (i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1) {
        int idx = i * Nx + j;
        double lap = tile[(si - 1) * tileW + sj]
                   + tile[(si + 1) * tileW + sj]
                   + tile[si * tileW + (sj - 1)]
                   + tile[si * tileW + (sj + 1)]
                   - 4.0 * tile[si * tileW + sj];
        u_next[idx] = 2.0 * tile[si * tileW + sj] - u_prev[idx]
                    + lambda2 * lap;
    }
}

// ── Kernels for cuSPARSE path ─────────────────────────────────────────────

// u_next = 2·u_curr − u_prev + λ²·lap   (element-wise, interior-only vector)
__global__ void updateFromLap(double *u_next, const double *u_curr,
                              const double *u_prev, const double *lap,
                              int n, double lambda2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + lambda2 * lap[idx];
}

// u1 = u0 + 0.5·λ²·lap   (first time step, interior-only)
__global__ void firstStepFromLap(double *u1, const double *u0,
                                 const double *lap, int n,
                                 double half_lambda2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        u1[idx] = u0[idx] + half_lambda2 * lap[idx];
}

// Extract interior values from full grid → contiguous interior vector
__global__ void extractInterior(const double *full, double *interior,
                                int Nx, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalInterior = M * M;
    if (idx < totalInterior) {
        int ii = idx / M;  // interior row (0-based)
        int jj = idx % M;  // interior col (0-based)
        interior[idx] = full[(ii + 1) * Nx + (jj + 1)];
    }
}

// Scatter interior vector → full grid
__global__ void scatterInterior(const double *interior, double *full,
                                int Nx, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalInterior = M * M;
    if (idx < totalInterior) {
        int ii = idx / M;
        int jj = idx % M;
        full[(ii + 1) * Nx + (jj + 1)] = interior[idx];
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  Host helper functions
// ══════════════════════════════════════════════════════════════════════════

struct TimingResult {
    double totalMs;
    double perStepMs;
    double bandwidthGBs;
    double maxError;
};

// Build the discrete Laplacian in CSR format for interior points.
// M = number of interior points per dimension.
// Matrix size: M² × M²,  up to 5 non-zeros per row.
void buildLaplacianCSR(int M,
                       std::vector<int>    &rowPtr,
                       std::vector<int>    &colInd,
                       std::vector<double> &values) {
    int N = M * M;
    rowPtr.resize(N + 1);
    colInd.clear();
    values.clear();
    colInd.reserve(5 * N);
    values.reserve(5 * N);

    int nnz = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int r = i * M + j;
            rowPtr[r] = nnz;
            // Up neighbor
            if (i > 0) {
                colInd.push_back(r - M);
                values.push_back(1.0);
                nnz++;
            }
            // Left neighbor
            if (j > 0) {
                colInd.push_back(r - 1);
                values.push_back(1.0);
                nnz++;
            }
            // Center
            colInd.push_back(r);
            values.push_back(-4.0);
            nnz++;
            // Right neighbor
            if (j < M - 1) {
                colInd.push_back(r + 1);
                values.push_back(1.0);
                nnz++;
            }
            // Down neighbor
            if (i < M - 1) {
                colInd.push_back(r + M);
                values.push_back(1.0);
                nnz++;
            }
        }
    }
    rowPtr[N] = nnz;
}

// Save wave field snapshot as binary file
// Format: [int32 Nx] [int32 Ny] [Nx*Ny doubles, row-major]
void saveFieldBinary(const double *h_u, int Nx, int Ny,
                     const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&Nx), sizeof(int));
    out.write(reinterpret_cast<const char *>(&Ny), sizeof(int));
    out.write(reinterpret_cast<const char *>(h_u),
              (size_t)Nx * Ny * sizeof(double));
    printf("  Saved %s\n", filename.c_str());
}

// Compute L∞ error vs analytical solution at time step n
double computeMaxError(const double *h_u, int Nx, int Ny,
                       double dx, double dy, int numSteps) {
    double t = numSteps * DT;
    double cosCoeff = cos(C_WAVE * sqrt(2.0) * M_PI * t);
    double maxErr = 0.0;
    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            double x = j * dx;
            double y = i * dy;
            double exact = cosCoeff * sin(M_PI * x) * sin(M_PI * y);
            double err = fabs(h_u[i * Nx + j] - exact);
            if (err > maxErr) maxErr = err;
        }
    }
    return maxErr;
}

// ══════════════════════════════════════════════════════════════════════════
//  Solver drivers
// ══════════════════════════════════════════════════════════════════════════

// ── (A1) Global memory solver ─────────────────────────────────────────────
TimingResult runGlobal(int Nx, int Ny, int numSteps, int bx, int by) {
    size_t bytes = (size_t)Nx * Ny * sizeof(double);
    double *d_u0, *d_u1, *d_u2;
    CHECK_CUDA(cudaMalloc(&d_u0, bytes));
    CHECK_CUDA(cudaMalloc(&d_u1, bytes));
    CHECK_CUDA(cudaMalloc(&d_u2, bytes));
    CHECK_CUDA(cudaMemset(d_u0, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u1, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u2, 0, bytes));

    dim3 block(bx, by);
    dim3 grid((Nx + bx - 1) / bx, (Ny + by - 1) / by);

    initWaveField<<<grid, block>>>(d_u0, Nx, Ny, DX, DY);
    CHECK_CUDA(cudaDeviceSynchronize());

    waveFirstStep<<<grid, block>>>(d_u1, d_u0, Nx, Ny, 0.5 * LAMBDA2);
    CHECK_CUDA(cudaDeviceSynchronize());

    double *u_prev = d_u0, *u_curr = d_u1, *u_next = d_u2;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int step = 2; step <= numSteps; step++) {
        waveStepGlobal<<<grid, block>>>(u_next, u_curr, u_prev,
                                        Nx, Ny, LAMBDA2);
        double *tmp = u_prev; u_prev = u_curr; u_curr = u_next; u_next = tmp;
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, start, stop));

    // Copy result and compute error
    std::vector<double> h_u(Nx * Ny);
    CHECK_CUDA(cudaMemcpy(h_u.data(), u_curr, bytes, cudaMemcpyDeviceToHost));
    double maxErr = computeMaxError(h_u.data(), Nx, Ny, DX, DY, numSteps);

    int timedSteps = numSteps - 1;
    double perStep = totalMs / timedSteps;
    double bytesPerStep = (double)(Nx - 2) * (Ny - 2) * 48.0;
    double bw = bytesPerStep / (perStep * 1e6);  // GB/s

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_u0));
    CHECK_CUDA(cudaFree(d_u1));
    CHECK_CUDA(cudaFree(d_u2));

    return {totalMs, perStep, bw, maxErr};
}

// ── (A2) Shared memory solver ─────────────────────────────────────────────
TimingResult runShared(int Nx, int Ny, int numSteps, int bx, int by) {
    size_t bytes = (size_t)Nx * Ny * sizeof(double);
    double *d_u0, *d_u1, *d_u2;
    CHECK_CUDA(cudaMalloc(&d_u0, bytes));
    CHECK_CUDA(cudaMalloc(&d_u1, bytes));
    CHECK_CUDA(cudaMalloc(&d_u2, bytes));
    CHECK_CUDA(cudaMemset(d_u0, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u1, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u2, 0, bytes));

    dim3 block(bx, by);
    dim3 grid((Nx + bx - 1) / bx, (Ny + by - 1) / by);
    size_t smemBytes = (size_t)(bx + 2) * (by + 2) * sizeof(double);

    initWaveField<<<grid, block>>>(d_u0, Nx, Ny, DX, DY);
    CHECK_CUDA(cudaDeviceSynchronize());

    waveFirstStep<<<grid, block>>>(d_u1, d_u0, Nx, Ny, 0.5 * LAMBDA2);
    CHECK_CUDA(cudaDeviceSynchronize());

    double *u_prev = d_u0, *u_curr = d_u1, *u_next = d_u2;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int step = 2; step <= numSteps; step++) {
        waveStepShared<<<grid, block, smemBytes>>>(u_next, u_curr, u_prev,
                                                    Nx, Ny, LAMBDA2);
        double *tmp = u_prev; u_prev = u_curr; u_curr = u_next; u_next = tmp;
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, start, stop));

    std::vector<double> h_u(Nx * Ny);
    CHECK_CUDA(cudaMemcpy(h_u.data(), u_curr, bytes, cudaMemcpyDeviceToHost));
    double maxErr = computeMaxError(h_u.data(), Nx, Ny, DX, DY, numSteps);

    int timedSteps = numSteps - 1;
    double perStep = totalMs / timedSteps;
    double bytesPerStep = (double)(Nx - 2) * (Ny - 2) * 48.0;
    double bw = bytesPerStep / (perStep * 1e6);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_u0));
    CHECK_CUDA(cudaFree(d_u1));
    CHECK_CUDA(cudaFree(d_u2));

    return {totalMs, perStep, bw, maxErr};
}

// ── (B) cuSPARSE solver ──────────────────────────────────────────────────
TimingResult runCuSPARSE(int Nx, int Ny, int numSteps) {
    int M = Nx - 2;  // interior points per dimension
    int totalInt = M * M;
    size_t fullBytes = (size_t)Nx * Ny * sizeof(double);
    size_t intBytes  = (size_t)totalInt * sizeof(double);

    // Build CSR Laplacian on host
    std::vector<int>    h_rowPtr, h_colInd;
    std::vector<double> h_values;
    buildLaplacianCSR(M, h_rowPtr, h_colInd, h_values);
    int nnz = (int)h_values.size();

    // Transfer CSR to device
    int    *d_rowPtr, *d_colInd;
    double *d_csrVal;
    CHECK_CUDA(cudaMalloc(&d_rowPtr, (totalInt + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csrVal, nnz * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr.data(),
                          (totalInt + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(),
                          nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrVal, h_values.data(),
                          nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize wave field on full grid
    double *d_full0, *d_full1;
    CHECK_CUDA(cudaMalloc(&d_full0, fullBytes));
    CHECK_CUDA(cudaMalloc(&d_full1, fullBytes));
    CHECK_CUDA(cudaMemset(d_full0, 0, fullBytes));
    CHECK_CUDA(cudaMemset(d_full1, 0, fullBytes));

    dim3 block2d(16, 16);
    dim3 grid2d((Nx + 15) / 16, (Ny + 15) / 16);
    initWaveField<<<grid2d, block2d>>>(d_full0, Nx, Ny, DX, DY);
    waveFirstStep<<<grid2d, block2d>>>(d_full1, d_full0, Nx, Ny,
                                       0.5 * LAMBDA2);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Allocate interior vectors
    double *d_prev, *d_curr, *d_next, *d_v;
    CHECK_CUDA(cudaMalloc(&d_prev, intBytes));
    CHECK_CUDA(cudaMalloc(&d_curr, intBytes));
    CHECK_CUDA(cudaMalloc(&d_next, intBytes));
    CHECK_CUDA(cudaMalloc(&d_v,    intBytes));

    int threads1d = 256;
    int blocks1d  = (totalInt + threads1d - 1) / threads1d;

    extractInterior<<<blocks1d, threads1d>>>(d_full0, d_prev, Nx, M);
    extractInterior<<<blocks1d, threads1d>>>(d_full1, d_curr, Nx, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_full0));
    CHECK_CUDA(cudaFree(d_full1));

    // Setup cuSPARSE
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matDescr;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matDescr, totalInt, totalInt, nnz,
        d_rowPtr, d_colInd, d_csrVal,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, totalInt, d_curr, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, totalInt, d_v,    CUDA_R_64F));

    double alpha = 1.0, beta = 0.0;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matDescr, vecX, &beta, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    void *d_buffer = nullptr;
    if (bufferSize > 0) CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    // Time the main loop
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int step = 2; step <= numSteps; step++) {
        // SpMV: v = L · u_curr
        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescr, vecX, &beta, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

        // u_next = 2·u_curr − u_prev + λ²·v
        updateFromLap<<<blocks1d, threads1d>>>(
            d_next, d_curr, d_prev, d_v, totalInt, LAMBDA2);

        // Rotate pointers
        double *tmp = d_prev; d_prev = d_curr; d_curr = d_next; d_next = tmp;
        CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, d_curr));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, start, stop));

    // Compute error: scatter interior → full grid
    double *d_fullOut;
    CHECK_CUDA(cudaMalloc(&d_fullOut, fullBytes));
    CHECK_CUDA(cudaMemset(d_fullOut, 0, fullBytes));
    scatterInterior<<<blocks1d, threads1d>>>(d_curr, d_fullOut, Nx, M);

    std::vector<double> h_u(Nx * Ny);
    CHECK_CUDA(cudaMemcpy(h_u.data(), d_fullOut, fullBytes,
                          cudaMemcpyDeviceToHost));
    double maxErr = computeMaxError(h_u.data(), Nx, Ny, DX, DY, numSteps);

    int timedSteps = numSteps - 1;
    double perStep = totalMs / timedSteps;
    double bytesPerStep = (double)(Nx - 2) * (Ny - 2) * 48.0;
    double bw = bytesPerStep / (perStep * 1e6);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroySpMat(matDescr));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    if (d_buffer) CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaFree(d_rowPtr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_csrVal));
    CHECK_CUDA(cudaFree(d_prev));
    CHECK_CUDA(cudaFree(d_curr));
    CHECK_CUDA(cudaFree(d_next));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_fullOut));

    return {totalMs, perStep, bw, maxErr};
}

// ── Visualization data export (not timed) ─────────────────────────────────
void runVisualization(int Nx, int Ny, int numSteps,
                      const std::vector<int> &saveSteps,
                      const std::string &outDir) {
    size_t bytes = (size_t)Nx * Ny * sizeof(double);
    double *d_u0, *d_u1, *d_u2;
    CHECK_CUDA(cudaMalloc(&d_u0, bytes));
    CHECK_CUDA(cudaMalloc(&d_u1, bytes));
    CHECK_CUDA(cudaMalloc(&d_u2, bytes));
    CHECK_CUDA(cudaMemset(d_u0, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u1, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u2, 0, bytes));

    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);

    initWaveField<<<grid, block>>>(d_u0, Nx, Ny, DX, DY);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> h_u(Nx * Ny);

    auto maybeSave = [&](int step, double *d_u) {
        if (std::find(saveSteps.begin(), saveSteps.end(), step)
            != saveSteps.end()) {
            CHECK_CUDA(cudaMemcpy(h_u.data(), d_u, bytes,
                                  cudaMemcpyDeviceToHost));
            saveFieldBinary(h_u.data(), Nx, Ny,
                            outDir + "field_step" + std::to_string(step)
                            + ".bin");
        }
    };

    maybeSave(0, d_u0);

    waveFirstStep<<<grid, block>>>(d_u1, d_u0, Nx, Ny, 0.5 * LAMBDA2);
    CHECK_CUDA(cudaDeviceSynchronize());
    maybeSave(1, d_u1);

    double *u_prev = d_u0, *u_curr = d_u1, *u_next = d_u2;
    for (int step = 2; step <= numSteps; step++) {
        waveStepGlobal<<<grid, block>>>(u_next, u_curr, u_prev,
                                        Nx, Ny, LAMBDA2);
        double *tmp = u_prev; u_prev = u_curr; u_curr = u_next; u_next = tmp;

        if (std::find(saveSteps.begin(), saveSteps.end(), step)
            != saveSteps.end()) {
            CHECK_CUDA(cudaMemcpy(h_u.data(), u_curr, bytes,
                                  cudaMemcpyDeviceToHost));
            saveFieldBinary(h_u.data(), Nx, Ny,
                            outDir + "field_step" + std::to_string(step)
                            + ".bin");
        }
    }

    CHECK_CUDA(cudaFree(d_u0));
    CHECK_CUDA(cudaFree(d_u1));
    CHECK_CUDA(cudaFree(d_u2));
}

// ══════════════════════════════════════════════════════════════════════════
//  Main
// ══════════════════════════════════════════════════════════════════════════

int main() {
    // Warm up CUDA context
    {
        double *tmp;
        CHECK_CUDA(cudaMalloc(&tmp, 1024));
        CHECK_CUDA(cudaFree(tmp));
    }

    printf("=== 2D Wave Equation Solver ===\n");
    printf("c=%.1f  dx=dy=%.4f  dt=%.4f  lambda^2=%.4f  steps=%d\n\n",
           C_WAVE, DX, DT, LAMBDA2, NUM_STEPS);

    // ── Block size study (L = 1) ──────────────────────────────────────────
    {
        int Nx = (int)(1.0 / DX) + 1;  // 101
        int Ny = Nx;
        printf("--- Block Size Study (L=1, grid %dx%d) ---\n", Nx, Ny);
        printf("%-10s %-14s %-14s %-12s %-12s %-10s %-10s\n",
               "Block", "Global(ms)", "Shared(ms)", "BW_G(GB/s)",
               "BW_S(GB/s)", "Err_G", "Err_S");

        struct { int bx, by; } sizes[] = {{8,8}, {16,16}, {32,8}, {32,32}};
        for (auto &s : sizes) {
            auto rg = runGlobal(Nx, Ny, NUM_STEPS, s.bx, s.by);
            auto rs = runShared(Nx, Ny, NUM_STEPS, s.bx, s.by);
            printf("(%2d,%2d)    %-14.4f %-14.4f %-12.2f %-12.2f %-10.2e %-10.2e\n",
                   s.bx, s.by,
                   rg.totalMs, rs.totalMs,
                   rg.bandwidthGBs, rs.bandwidthGBs,
                   rg.maxError, rs.maxError);
        }
    }

    // ── Scaling study ─────────────────────────────────────────────────────
    printf("\n--- Scaling Study (block 16x16, %d steps) ---\n", NUM_STEPS);
    printf("%-6s %-6s %-10s %-14s %-14s %-14s %-12s %-12s %-10s\n",
           "L", "Nx", "Points", "Global(ms)", "Shared(ms)",
           "cuSPARSE(ms)", "BW_G(GB/s)", "BW_S(GB/s)", "MaxErr");

    double Ls[] = {1.0, 2.0, 4.0, 8.0};
    for (double L : Ls) {
        int Nx = (int)(L / DX) + 1;
        int Ny = Nx;
        auto rg = runGlobal  (Nx, Ny, NUM_STEPS, 16, 16);
        auto rs = runShared  (Nx, Ny, NUM_STEPS, 16, 16);
        auto rc = runCuSPARSE(Nx, Ny, NUM_STEPS);
        printf("%-6.0f %-6d %-10d %-14.4f %-14.4f %-14.4f %-12.2f %-12.2f %-10.2e\n",
               L, Nx, Nx * Ny,
               rg.totalMs, rs.totalMs, rc.totalMs,
               rg.bandwidthGBs, rs.bandwidthGBs,
               rg.maxError);
    }

    // ── Generate visualization snapshots (L = 1) ──────────────────────────
    printf("\n--- Generating Visualization Data (L=1) ---\n");
    int visNx = (int)(1.0 / DX) + 1;
    runVisualization(visNx, visNx, NUM_STEPS,
                     {0, 50, 100, 150, 200}, "out/");

    printf("\nDone. Run 'python3 visualize.py' to generate plots.\n");
    return 0;
}
