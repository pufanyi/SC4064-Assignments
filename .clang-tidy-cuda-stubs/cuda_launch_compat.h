// Stub: cuda_launch_compat.h
// Force-included before every .cu file in CI to provide cudaConfigureCall.
//
// clang-20 translates <<<>>> kernel launches into a call to
// cudaConfigureCall, which was deprecated in CUDA 10 and removed in
// CUDA 12. With -nocudalib the declaration may be missing from
// CUDA 11.8 headers, so we provide it here.

#ifndef CUDA_LAUNCH_COMPAT_H_
#define CUDA_LAUNCH_COMPAT_H_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __cudaConfigureCall_defined
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                              size_t sharedMem = 0,
                              cudaStream_t stream = nullptr);
#endif

#ifdef __cplusplus
}
#endif

#endif // CUDA_LAUNCH_COMPAT_H_
