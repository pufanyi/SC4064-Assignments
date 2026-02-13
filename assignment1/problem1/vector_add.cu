#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define N (1 << 30)

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
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize with random values in [0.0f, 100.0f]
    srand(42);
    for (long i = 0; i < N; i++) {
        h_a[i] = ((float)rand() / RAND_MAX) * 100.0f;
        h_b[i] = ((float)rand() / RAND_MAX) * 100.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Test different block sizes
    int blockSizes[] = {32, 64, 128, 256};
    int numTests = sizeof(blockSizes) / sizeof(blockSizes[0]);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("Vector Addition: N = %d (2^30)\n", N);
    printf("%-15s %-15s %-15s %-15s\n", "Block Size", "Grid Size",
           "Time (ms)", "GFLOPS");
    printf("-----------------------------------------------------------\n");

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

        printf("%-15d %-15d %-15.3f %-15.4f\n", blockSize, gridSize, ms,
               gflops);
    }

    // Verify result (using the last run)
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (long i = 0; i < 1000; i++) {
        if (fabsf(h_c[i] - (h_a[i] + h_b[i])) > 1e-3f) {
            correct = false;
            fprintf(stderr, "Mismatch at index %ld: %f != %f + %f\n", i,
                    h_c[i], h_a[i], h_b[i]);
            break;
        }
    }
    printf("\nVerification: %s\n", correct ? "PASSED" : "FAILED");

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
