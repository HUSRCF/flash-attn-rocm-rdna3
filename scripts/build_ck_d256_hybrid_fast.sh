#!/usr/bin/env bash
set -euo pipefail

# D256 hybrid candidate validated on gfx1100/W7900.
# Key idea:
# - Keep generic DK/DV LDS layout to reduce DK-only LDS pressure under stream overlap.
# - Use old-style DQ scheduling for D256: no late DO/D prefetch and no late reg load.
# - Enable inplace dS in DQ to reduce DQ scratch/codegen pressure.
# - Keep split stream pool enabled for lower end-to-end time.

LOG=${1:-build_logs/build_d256_hybrid_fullmatrix.log}
mkdir -p "$(dirname "$LOG")"

BUILD_TARGET=rocm \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_CK_MINIMAL_DEBUG=TRUE \
FLASH_ATTN_CK_MINIMAL_OPTDIM=256 \
FLASH_ATTN_CK_MINIMAL_DTYPE=fp16,bf16 \
FLASH_ATTN_CK_MINIMAL_BATCH_ONLY=FALSE \
FLASH_ATTN_CK_MINIMAL_NONCAUSAL_ONLY=FALSE \
GPU_ARCHS=gfx1100 \
MAX_JOBS=128 \
CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT_ONLY=1 \
CK_TILE_DEBUG_BWD_SPLIT_PARALLEL_STREAMS=2 \
CK_TILE_DEBUG_BWD_SPLIT_DQ_INPLACE_DS=1 \
CK_TILE_DEBUG_BWD_SPLIT_DQ_LATE_DO_D_PREFETCH=0 \
CK_TILE_DEBUG_BWD_SPLIT_DQ_LATE_REG_LOAD=0 \
CK_TILE_DEBUG_BWD_SPLIT_DKDV_USE_GENERIC_LDS_LAYOUT=1 \
CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_P=1 \
CK_TILE_DEBUG_BWD_SPLIT_LAUNCH_DQ_FIRST=0 \
CK_TILE_DEBUG_BWD_SPLIT_DV_EARLY_Q_LSE_STORE=0 \
CK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM=0 \
python setup.py build_ext --inplace --force > "$LOG" 2>&1
