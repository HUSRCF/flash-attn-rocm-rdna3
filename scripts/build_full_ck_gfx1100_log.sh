#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

mkdir -p build_logs
log_file="build_logs/full_ck_gfx1100_$(date +%Y%m%d_%H%M%S).log"

# Optional AOCC host compiler trial. Leave disabled for the baseline full build
# unless you intentionally want to compare host compiler behavior/compile time.
# export CC=/opt/AMD/aocc-compiler-5.1.0/bin/clang
# export CXX=/opt/AMD/aocc-compiler-5.1.0/bin/clang++

env \
  BUILD_TARGET=rocm \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  FLASH_ATTN_CK_MINIMAL_DEBUG=FALSE \
  GPU_ARCHS=gfx1100 \
  MAX_JOBS=128 \
  CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4=0 \
  CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_KT_ONLY=0 \
  CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY=0 \
  CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_ONE_SLICE=0 \
  CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE=0 \
  python setup.py build_ext --inplace 2>&1 | tee "${log_file}"

echo "build log: ${log_file}"
