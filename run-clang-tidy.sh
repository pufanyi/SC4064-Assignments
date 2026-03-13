#!/bin/bash
# Run clang-tidy on CUDA files
# Usage: ./run-clang-tidy.sh [file.cu ...]
#        ./run-clang-tidy.sh              (runs on all .cu files)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUBS_DIR="$SCRIPT_DIR/.clang-tidy-cuda-stubs"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
GPU_ARCH="${GPU_ARCH:-sm_70}"

EXTRA_ARGS="-- --cuda-gpu-arch=$GPU_ARCH -std=c++20 -x cuda --cuda-path=$CUDA_PATH -isystem $STUBS_DIR"

if [ $# -eq 0 ]; then
    files=$(find "$SCRIPT_DIR" -name '*.cu' -not -path '*/build/*')
else
    files="$@"
fi

exit_code=0
for f in $files; do
    echo "=== $f ==="
    clang-tidy-20 "$f" $EXTRA_ARGS
    if [ $? -ne 0 ]; then
        exit_code=1
    fi
    echo
done

exit $exit_code
