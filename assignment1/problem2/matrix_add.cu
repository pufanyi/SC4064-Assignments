#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << std::format("CUDA error at {}:{}: {}", __FILE__,  \
                                     __LINE__, cudaGetErrorString(err))     \
                      << std::endl;                                        \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define ROWS 8192
#define COLS 8192

// ============================================================
// Initialization kernel
// ============================================================
__global__ void initRandom(float *data, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int h = seed ^ (unsigned int)idx;
        h ^= h >> 16;
        h *= 0x45d9f3b;
        h ^= h >> 16;
        data[idx] = (float)(h & 0xFFFFFF) / (float)0xFFFFFF * 100.0f;
    }
}

// ============================================================
// 1D Configuration: 1D grid with 1D blocks
//
// Global thread index:
//     idx = blockIdx.x * blockDim.x + threadIdx.x
//
// Mapping to matrix element (i, j):
//     i = idx / COLS   (row index)
//     j = idx % COLS   (column index)
//
// Each thread processes one element: C[i][j] = A[i][j] + B[i][j]
// The matrix is stored in row-major order, so element (i, j) is
// at linear offset i * COLS + j, which equals idx.
// ============================================================
__global__ void matrixAdd1D(const float *A, const float *B, float *C,
                            int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

// ============================================================
// 2D Configuration: 2D grid with 2D blocks
//
// Global thread index:
//     row = blockIdx.y * blockDim.y + threadIdx.y
//     col = blockIdx.x * blockDim.x + threadIdx.x
//
// Mapping to matrix element (i, j):
//     i = row   (directly corresponds to blockIdx.y / threadIdx.y)
//     j = col   (directly corresponds to blockIdx.x / threadIdx.x)
//
// Each thread processes one element: C[i][j] = A[i][j] + B[i][j]
// The linear memory offset is i * cols + j.
// The 2D indexing naturally maps to matrix coordinates.
// ============================================================
__global__ void matrixAdd2D(const float *A, const float *B, float *C,
                            int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int total = ROWS * COLS;
    size_t bytes = (size_t)total * sizeof(float);

    std::cout << std::format("Matrix Addition: {} x {} (float)\n", ROWS, COLS);
    std::cout << std::format("Total elements: {}\n\n", total);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host memory allocation failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Initialize A and B with random values in [0.0f, 100.0f] on GPU
    int initBlock = 256;
    int initGrid = (total + initBlock - 1) / initBlock;
    initRandom<<<initGrid, initBlock>>>(d_A, total, 42);
    initRandom<<<initGrid, initBlock>>>(d_B, total, 137);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Initialize C to 0.0f
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ============================================================
    // 1D Configuration
    // ============================================================
    std::cout << "=== 1D Configuration (1D grid, 1D blocks) ===" << std::endl;
    std::cout << std::format("{:<15}{:<15}{:<15}{:<15}", "Block Size",
                             "Grid Size", "Time (ms)", "GFLOPS")
              << std::endl;
    std::cout << "-----------------------------------------------------------"
              << std::endl;

    int blockSizes1D[] = {64, 128, 256, 512};
    int numTests1D = sizeof(blockSizes1D) / sizeof(blockSizes1D[0]);

    for (int t = 0; t < numTests1D; t++) {
        int blockSize = blockSizes1D[t];
        int gridSize = (total + blockSize - 1) / blockSize;

        // Warm up
        matrixAdd1D<<<gridSize, blockSize>>>(d_A, d_B, d_C, ROWS, COLS);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Timed run
        CHECK_CUDA(cudaEventRecord(start));
        matrixAdd1D<<<gridSize, blockSize>>>(d_A, d_B, d_C, ROWS, COLS);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        // FLOPS: total additions, 1 FLOP each
        double flops = (double)total;
        double gflops = flops / (ms * 1e6);

        std::cout << std::format("{:<15}{:<15}{:<15.4f}{:<15.4f}", blockSize,
                                 gridSize, ms, gflops)
                  << std::endl;
    }

    // ============================================================
    // 2D Configuration
    // ============================================================
    std::cout << std::endl;
    std::cout << "=== 2D Configuration (2D grid, 2D blocks) ===" << std::endl;
    std::cout << std::format("{:<20}{:<20}{:<15}{:<15}", "Block Size",
                             "Grid Size", "Time (ms)", "GFLOPS")
              << std::endl;
    std::cout << "-------------------------------------------------------------------"
              << std::endl;

    // Test different 2D block sizes
    struct BlockConfig {
        int x, y;
    };
    BlockConfig blockConfigs2D[] = {{16, 16}, {32, 8}, {32, 16}, {32, 32}};
    int numTests2D = sizeof(blockConfigs2D) / sizeof(blockConfigs2D[0]);

    for (int t = 0; t < numTests2D; t++) {
        dim3 blockDim(blockConfigs2D[t].x, blockConfigs2D[t].y);
        dim3 gridDim((COLS + blockDim.x - 1) / blockDim.x,
                     (ROWS + blockDim.y - 1) / blockDim.y);

        // Warm up
        matrixAdd2D<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROWS, COLS);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Timed run
        CHECK_CUDA(cudaEventRecord(start));
        matrixAdd2D<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROWS, COLS);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        double flops = (double)total;
        double gflops = flops / (ms * 1e6);

        std::string blockStr = std::format("({}, {})", blockDim.x, blockDim.y);
        std::string gridStr = std::format("({}, {})", gridDim.x, gridDim.y);

        std::cout << std::format("{:<20}{:<20}{:<15.4f}{:<15.4f}", blockStr,
                                 gridStr, ms, gflops)
                  << std::endl;
    }

    // ============================================================
    // Verification
    // ============================================================
    CHECK_CUDA(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        if (std::fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-3f) {
            correct = false;
            std::cerr << std::format("Mismatch at index {}: {} != {} + {}", i,
                                     h_C[i], h_A[i], h_B[i])
                      << std::endl;
            break;
        }
    }
    std::cout << std::format("\nVerification: {}",
                             correct ? "PASSED" : "FAILED")
              << std::endl;

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
