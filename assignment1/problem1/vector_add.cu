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

#define N (1 << 30)

__global__ void initRandom(float *data, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple hash-based pseudo-random
        unsigned int h = seed ^ (unsigned int)idx;
        h ^= h >> 16;
        h *= 0x45d9f3b;
        h ^= h >> 16;
        data[idx] = (float)(h & 0xFFFFFF) / (float)0xFFFFFF * 100.0f;
    }
}

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    size_t bytes = (size_t)N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        std::cerr << "Host memory allocation failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Initialize on GPU (much faster than CPU rand() loop)
    int initBlock = 256;
    int initGrid = (N + initBlock - 1) / initBlock;
    initRandom<<<initGrid, initBlock>>>(d_a, N, 42);
    initRandom<<<initGrid, initBlock>>>(d_b, N, 137);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Test different block sizes
    int blockSizes[] = {32, 64, 128, 256};
    int numTests = sizeof(blockSizes) / sizeof(blockSizes[0]);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << std::format("Vector Addition: N = {} (2^30)", N) << std::endl;
    std::cout << std::format("{:<15}{:<15}{:<15}{:<15}", "Block Size",
                             "Grid Size", "Time (ms)", "GFLOPS")
              << std::endl;
    std::cout << "-----------------------------------------------------------"
              << std::endl;

    for (int t = 0; t < numTests; t++) {
        int blockSize = blockSizes[t];
        int gridSize = (N + blockSize - 1) / blockSize;

        // Warm up
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Timed run
        CHECK_CUDA(cudaEventRecord(start));
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        // FLOPS: N additions, 1 FLOP each
        double flops = (double)N;
        double gflops = flops / (ms * 1e6);

        std::cout << std::format("{:<15}{:<15}{:<15.3f}{:<15.4f}", blockSize,
                                 gridSize, ms, gflops)
                  << std::endl;
    }

    // Verify result (using the last run)
    CHECK_CUDA(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (long i = 0; i < 1000; i++) {
        if (std::fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-3f) {
            correct = false;
            std::cerr << std::format("Mismatch at index {}: {} != {} + {}", i,
                                     h_c[i], h_a[i], h_b[i])
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
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
