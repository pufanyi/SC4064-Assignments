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

#define M 8192
#define K 8192
#define N 8192

// ============================================================
// Initialization kernel: random values in [0.0f, 1.0f]
// ============================================================
__global__ void initRandom(float *data, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int h = seed ^ (unsigned int)idx;
        h ^= h >> 16;
        h *= 0x45d9f3b;
        h ^= h >> 16;
        data[idx] = (float)(h & 0xFFFFFF) / (float)0xFFFFFF;
    }
}

// ============================================================
// Matrix Multiplication Kernel: C = A × B
//
// Thread-to-element mapping:
//     row = blockIdx.y * blockDim.y + threadIdx.y
//     col = blockIdx.x * blockDim.x + threadIdx.x
//
// Each thread computes one element C(i, j) by performing the
// inner product of row i of A and column j of B:
//
//     C(i, j) = sum_{k=0}^{K-1} A(i, k) * B(k, j)
//
// All matrices are stored in row-major order:
//     A(i, k) is at A[i * K + k]
//     B(k, j) is at B[k * N + j]
//     C(i, j) is at C[i * N + j]
// ============================================================
__global__ void matrixMul(const float *A, const float *B, float *C,
                          int m, int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ============================================================
// CPU reference (partial) for verification
// ============================================================
void matrixMulCPU(const float *A, const float *B, float *C,
                  int m, int k, int n, int numElements) {
    for (int idx = 0; idx < numElements; idx++) {
        int row = idx / n;
        int col = idx % n;
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[idx] = sum;
    }
}

int main() {
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    std::cout << std::format("Matrix Multiplication: C = A x B\n");
    std::cout << std::format("A: {} x {},  B: {} x {},  C: {} x {}\n\n",
                             M, K, K, N, M, N);

    // Allocate host memory
    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);
    float *h_ref = (float *)malloc(bytesC);
    if (!h_A || !h_B || !h_C || !h_ref) {
        std::cerr << "Host memory allocation failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));

    // Initialize A and B with random values in [0.0f, 1.0f]
    int initBlock = 256;
    int initGridA = ((M * K) + initBlock - 1) / initBlock;
    int initGridB = ((K * N) + initBlock - 1) / initBlock;
    initRandom<<<initGridA, initBlock>>>(d_A, M * K, 42);
    initRandom<<<initGridB, initBlock>>>(d_B, K * N, 137);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Initialize C to 0.0f
    CHECK_CUDA(cudaMemset(d_C, 0, bytesC));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ============================================================
    // Experiment with different 2D block sizes
    // ============================================================
    std::cout << "=== Matrix Multiplication (2D grid, 2D blocks) ===" << std::endl;
    std::cout << std::format("{:<20}{:<20}{:<15}{:<15}", "Block Size",
                             "Grid Size", "Time (ms)", "GFLOPS")
              << std::endl;
    std::cout << "-------------------------------------------------------------------"
              << std::endl;

    struct BlockConfig {
        int x, y;
    };
    BlockConfig configs[] = {{8, 8}, {16, 16}, {32, 32}};
    int numConfigs = sizeof(configs) / sizeof(configs[0]);

    // Total FLOPs: 2 * M * N * K (K multiplications + K additions per element)
    double totalFlops = 2.0 * (double)M * (double)N * (double)K;

    for (int t = 0; t < numConfigs; t++) {
        dim3 blockDim(configs[t].x, configs[t].y);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);

        // Initialize C to 0
        CHECK_CUDA(cudaMemset(d_C, 0, bytesC));

        // Warm up
        matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Timed run
        CHECK_CUDA(cudaMemset(d_C, 0, bytesC));
        CHECK_CUDA(cudaEventRecord(start));
        matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        double gflops = totalFlops / (ms * 1e6);

        std::string blockStr = std::format("({}, {})", blockDim.x, blockDim.y);
        std::string gridStr = std::format("({}, {})", gridDim.x, gridDim.y);

        std::cout << std::format("{:<20}{:<20}{:<15.2f}{:<15.2f}", blockStr,
                                 gridStr, ms, gflops)
                  << std::endl;
    }

    // ============================================================
    // Verification: compare GPU result with CPU for a few elements
    // ============================================================
    CHECK_CUDA(cudaMemcpy(h_A, d_A, bytesA, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B, d_B, bytesB, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    // Verify first few elements on CPU
    int verifyCount = 100;
    matrixMulCPU(h_A, h_B, h_ref, M, K, N, verifyCount);

    bool correct = true;
    for (int i = 0; i < verifyCount; i++) {
        float relErr = std::fabs(h_C[i] - h_ref[i]) /
                       (std::fabs(h_ref[i]) + 1e-7f);
        if (relErr > 1e-3f) {
            correct = false;
            std::cerr << std::format(
                "Mismatch at index {}: GPU={}, CPU={}, relErr={}",
                i, h_C[i], h_ref[i], relErr)
                      << std::endl;
            break;
        }
    }
    std::cout << std::format("\nVerification (first {} elements): {}",
                             verifyCount, correct ? "PASSED" : "FAILED")
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
    free(h_ref);

    return 0;
}
