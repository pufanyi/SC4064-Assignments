// Stub: cusparse.h — minimal declarations for clang-tidy
// This file exists only so that clang-tidy can parse wave2d.cu
// when libcusparse-dev is not installed.

#ifndef CUSPARSE_STUB_H
#define CUSPARSE_STUB_H

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// ── Status / enum types ────────────────────────────────────────────────────
typedef enum {
    CUSPARSE_STATUS_SUCCESS = 0
} cusparseStatus_t;

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0
} cusparseOperation_t;

typedef enum {
    CUSPARSE_INDEX_32I = 1
} cusparseIndexType_t;

typedef enum {
    CUSPARSE_INDEX_BASE_ZERO = 0
} cusparseIndexBase_t;

typedef enum {
    CUSPARSE_SPMV_ALG_DEFAULT = 0
} cusparseSpMVAlg_t;

// ── Opaque handle / descriptor types ───────────────────────────────────────
struct cusparseContext;
typedef cusparseContext *cusparseHandle_t;

struct cusparseSpMatDescr;
typedef cusparseSpMatDescr *cusparseSpMatDescr_t;

struct cusparseDnVecDescr;
typedef cusparseDnVecDescr *cusparseDnVecDescr_t;

// ── API stubs (declarations only) ──────────────────────────────────────────
inline cusparseStatus_t cusparseCreate(cusparseHandle_t *) {
    return CUSPARSE_STATUS_SUCCESS;
}
inline cusparseStatus_t cusparseDestroy(cusparseHandle_t) {
    return CUSPARSE_STATUS_SUCCESS;
}

inline cusparseStatus_t cusparseCreateCsr(
    cusparseSpMatDescr_t *, int64_t, int64_t, int64_t,
    void *, void *, void *,
    cusparseIndexType_t, cusparseIndexType_t,
    cusparseIndexBase_t, cudaDataType) {
    return CUSPARSE_STATUS_SUCCESS;
}
inline cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t) {
    return CUSPARSE_STATUS_SUCCESS;
}

inline cusparseStatus_t cusparseCreateDnVec(
    cusparseDnVecDescr_t *, int64_t, void *, cudaDataType) {
    return CUSPARSE_STATUS_SUCCESS;
}
inline cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t) {
    return CUSPARSE_STATUS_SUCCESS;
}
inline cusparseStatus_t cusparseDnVecSetValues(
    cusparseDnVecDescr_t, void *) {
    return CUSPARSE_STATUS_SUCCESS;
}

inline cusparseStatus_t cusparseSpMV_bufferSize(
    cusparseHandle_t, cusparseOperation_t,
    const void *, cusparseSpMatDescr_t, cusparseDnVecDescr_t,
    const void *, cusparseDnVecDescr_t,
    cudaDataType, cusparseSpMVAlg_t, size_t *) {
    return CUSPARSE_STATUS_SUCCESS;
}
inline cusparseStatus_t cusparseSpMV(
    cusparseHandle_t, cusparseOperation_t,
    const void *, cusparseSpMatDescr_t, cusparseDnVecDescr_t,
    const void *, cusparseDnVecDescr_t,
    cudaDataType, cusparseSpMVAlg_t, void *) {
    return CUSPARSE_STATUS_SUCCESS;
}

#endif // CUSPARSE_STUB_H
