#include "multigpu.cuh"
#include "pipeline.cuh"
#include "pgm_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

// ─────────────────────────────────────────────────────────────────────────────
// process_batch_on_device
//
// Runs the full three-stage pipeline on a sub-batch of images using a single
// GPU (identified by device_id). Each image is processed on its own CUDA
// stream so that H→D transfers, kernels, and D→H transfers can overlap
// across images.
// ─────────────────────────────────────────────────────────────────────────────
static void process_batch_on_device(std::vector<ImageEntry>& sub_batch, int device_id)
{
    cudaSetDevice(device_id);

    int W = sub_batch[0].width;
    int H = sub_batch[0].height;
    size_t img_bytes = (size_t)W * H * sizeof(uint8_t);
    int    n_images  = (int)sub_batch.size();

    // ── Per-image device buffers ──────────────────────────────────────────
    // d_in   : raw input (reused for equalize output)
    // d_buf1 : blur output / sobel input
    // d_buf2 : sobel output / equalize input
    // d_hist : histogram (256 unsigned ints)
    // d_cdf  : CDF (256 floats)
    std::vector<uint8_t*>      d_in(n_images),  d_buf1(n_images), d_buf2(n_images);
    std::vector<unsigned int*> d_hist(n_images);
    std::vector<float*>        d_cdf(n_images);

    for (int i = 0; i < n_images; i++) {
        cudaMalloc(&d_in[i],   img_bytes);
        cudaMalloc(&d_buf1[i], img_bytes);
        cudaMalloc(&d_buf2[i], img_bytes);
        cudaMalloc(&d_hist[i], 256 * sizeof(unsigned int));
        cudaMalloc(&d_cdf[i],  256 * sizeof(float));
    }

    // ── Per-image CUDA streams ────────────────────────────────────────────
    std::vector<cudaStream_t> streams(n_images);
    for (int i = 0; i < n_images; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // ── Kernel launch configurations ─────────────────────────────────────
    dim3 blk2D(TILE_W, TILE_H);
    dim3 grid2D((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);
    int threads1D = 256;
    int blocks1D  = (W * H + threads1D - 1) / threads1D;

    // ── Submit all images to the GPU ──────────────────────────────────────
    for (int i = 0; i < n_images; i++) {

        // H→D: copy raw input to device
        cudaMemcpyAsync(d_in[i], sub_batch[i].host_in, img_bytes,
                        cudaMemcpyHostToDevice, streams[i]);

        // Stage 1: Gaussian blur
        gaussianBlurKernel<<<grid2D, blk2D, 0, streams[i]>>>(
            d_in[i], d_buf1[i], W, H);

        // Stage 2: Sobel edge detection
        sobelKernel<<<grid2D, blk2D, 0, streams[i]>>>(
            d_buf1[i], d_buf2[i], W, H);

        // Zero-initialise histogram
        cudaMemsetAsync(d_hist[i], 0, 256 * sizeof(unsigned int), streams[i]);

        // Stage 3A: Histogram
        histogramKernel<<<blocks1D, threads1D, 0, streams[i]>>>(
            d_buf2[i], d_hist[i], W, H);

        // Stage 3B: CDF via thrust (thrust uses default stream — sync first)
        cudaStreamSynchronize(streams[i]);

        thrust::device_ptr<unsigned int> hist_ptr(d_hist[i]);
        thrust::device_ptr<float>        cdf_ptr(d_cdf[i]);
        thrust::exclusive_scan(hist_ptr, hist_ptr + 256, cdf_ptr);

        // Find cdf_min on the host
        float h_cdf[256];
        cudaMemcpy(h_cdf, d_cdf[i], 256 * sizeof(float), cudaMemcpyDeviceToHost);
        float cdf_min = 0.f;
        for (int b = 0; b < 256; b++) {
            if (h_cdf[b] > 0.f) { cdf_min = h_cdf[b]; break; }
        }

        // Stage 3C: Equalisation
        equalizeKernel<<<blocks1D, threads1D, 0, streams[i]>>>(
            d_buf2[i], d_in[i], d_cdf[i], cdf_min, W, H);

        // D→H: copy equalised output back to host
        cudaMemcpyAsync(sub_batch[i].host_out, d_in[i], img_bytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // ── Wait for all images to finish ─────────────────────────────────────
    for (int i = 0; i < n_images; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // ── Save results ──────────────────────────────────────────────────────
    for (int i = 0; i < n_images; i++) {
        pgm_save(sub_batch[i].output_path, sub_batch[i].host_out, W, H);
    }

    // ── Clean up ──────────────────────────────────────────────────────────
    for (int i = 0; i < n_images; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_in[i]);
        cudaFree(d_buf1[i]);
        cudaFree(d_buf2[i]);
        cudaFree(d_hist[i]);
        cudaFree(d_cdf[i]);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// run_pipeline_multigpu  (public entry point)
// ─────────────────────────────────────────────────────────────────────────────
void run_pipeline_multigpu(std::vector<ImageEntry>& batch)
{
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 2) {
        printf("[multigpu] Only %d GPU(s) detected — falling back to single GPU.\n",
               num_gpus);
        process_batch_on_device(batch, 0);
        return;
    }

    printf("[multigpu] Using %d GPUs.\n", num_gpus);

    // Split batch: GPU 0 gets first half, GPU 1 gets the rest.
    int n    = (int)batch.size();
    int half = n / 2;

    std::vector<ImageEntry> sub0(batch.begin(), batch.begin() + half);
    std::vector<ImageEntry> sub1(batch.begin() + half, batch.end());

    // Process both sub-batches in parallel using host threads.
    std::thread t0([&sub0]() { process_batch_on_device(sub0, 0); });
    std::thread t1([&sub1]() { process_batch_on_device(sub1, 1); });
    t0.join();
    t1.join();
}

void run_pipeline_singlegpu(std::vector<ImageEntry>& batch)
{
    process_batch_on_device(batch, 0);
}
