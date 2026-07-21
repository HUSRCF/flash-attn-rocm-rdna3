#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./dump_fa_isa.sh [--kernel PATTERN] [--so PATH] [--out DIR]
  ./dump_fa_isa.sh [--analyze-bwd] [--dk-pattern REGEX] [--dv-pattern REGEX] [--so PATH] [--out DIR]
  ./dump_fa_isa.sh [--analyze-bwd] --gpu-obj PATH --dv-addr HEX --dk-addr HEX [--dq-addr HEX] [--out DIR]

Examples:
  ./dump_fa_isa.sh
  ./dump_fa_isa.sh --kernel 'split_dk|kr_ktr_vr_iglp'
  ./dump_fa_isa.sh --so ./flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so --kernel 'fmha_bwd'
  ./dump_fa_isa.sh --analyze-bwd
  ./dump_fa_isa.sh --analyze-bwd --dk-pattern '_split_dk' --dv-pattern '_split_dv'
  ./dump_fa_isa.sh --analyze-bwd --gpu-obj testoutput/fa_chunk8_unbundle/gfx1100.o --dv-addr 0xe500 --dk-addr 0x14200 --dq-addr 0x19c00

What it does:
  1. Resolve the loaded flash_attn_2_cuda .so, unless --so is given
  2. Dump:
     - symbols.txt
     - amdhsa.txt
     - disassembly.txt
  3. If --kernel is given, generate:
     - symbol_hits.txt
     - disasm_hits.txt
  4. If --analyze-bwd is given, additionally generate:
     - bwd_isa/dk_labels.txt
     - bwd_isa/dv_labels.txt
     - bwd_isa/dk_kernel_blocks.txt
     - bwd_isa/dv_kernel_blocks.txt
     - bwd_isa/dk_summary.txt
     - bwd_isa/dv_summary.txt
     - bwd_isa/compare_summary.txt
     - fatbin_strings.txt
     - bwd_isa/kernel_name_inventory.txt
     - bwd_isa/dk_name_candidates.txt
     - bwd_isa/dv_name_candidates.txt
     - bwd_isa/dv_addr_window.s
     - bwd_isa/dk_addr_window.s
     - bwd_isa/dv_addr_summary.txt
     - bwd_isa/dk_addr_summary.txt

Notes:
  - This is a base dump script. It does not try to perfectly cut one kernel body out of the
    disassembly. Use disasm_hits.txt to find line numbers, then inspect the surrounding region.
  - --analyze-bwd is a convenience mode for the current CK split_dk vs split_dv investigation.
    It first tries to extract `.hip_fatbin`, unbundle AMDGPU code objects, and analyze the
    AMDGPU ISA disassembly. It then extracts all matching kernel blocks and summarizes:
      scratch_load/store, ds_read/write, s_waitcnt, s_barrier, v_wmma, s_delay_alu
  - Preferred tools:
      llvm-objdump / llvm-readelf / llvm-readobj
    Fallback:
      /opt/rocm/bin/roc-obj
EOF
}

KERNEL_PATTERN=""
SO_PATH=""
OUT_DIR=""
ANALYZE_BWD=0
DK_PATTERN="_split_dk"
DV_PATTERN="_split_dv"
GPU_OBJ_PATH=""
DK_ADDR=""
DV_ADDR=""
DQ_ADDR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel)
      KERNEL_PATTERN="${2:-}"
      shift 2
      ;;
    --so)
      SO_PATH="${2:-}"
      shift 2
      ;;
    --out)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --analyze-bwd)
      ANALYZE_BWD=1
      shift
      ;;
    --dk-pattern)
      DK_PATTERN="${2:-}"
      shift 2
      ;;
    --dv-pattern)
      DV_PATTERN="${2:-}"
      shift 2
      ;;
    --gpu-obj)
      GPU_OBJ_PATH="${2:-}"
      shift 2
      ;;
    --dk-addr)
      DK_ADDR="${2:-}"
      shift 2
      ;;
    --dv-addr)
      DV_ADDR="${2:-}"
      shift 2
      ;;
    --dq-addr)
      DQ_ADDR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SO_PATH" ]]; then
  SO_PATH="$(python - <<'PY'
try:
    import torch  # load PyTorch runtime deps such as libc10 first
    import flash_attn_2_cuda
    print(flash_attn_2_cuda.__file__)
except Exception:
    pass
PY
)"
fi

if [[ -z "$SO_PATH" ]]; then
  SO_PATH="$(find . -maxdepth 3 -type f \( -name 'flash_attn_2_cuda*.so' -o -name 'flash_attn*.so' \) | head -n 1 || true)"
fi

if [[ ! -f "$SO_PATH" ]]; then
  echo "SO not found: $SO_PATH" >&2
  exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="testoutput/isa_${stamp}"
fi

mkdir -p "$OUT_DIR"

extract_kernel_blocks() {
  local regex="$1"
  local input_file="$2"
  local labels_file="$3"
  local blocks_file="$4"

  grep -En "^[[:space:]]*[[:xdigit:]]+[[:space:]]+<.*${regex}.*>:" "$input_file" > "$labels_file" || true

  awk -v re="$regex" '
    function is_label(line) {
      return line ~ /^[[:space:]]*[[:xdigit:]]+[[:space:]]+<.*>:/
    }
    {
      if (is_label($0)) {
        if (capturing) {
          print ""
        }
        capturing = ($0 ~ re)
        if (capturing) {
          count += 1
          print "===== KERNEL " count " ====="
          print $0
        }
        next
      }
      if (capturing) {
        print
      }
    }
    END {
      if (count == 0) {
        exit 1
      }
    }
  ' "$input_file" > "$blocks_file" || true
}

count_pattern() {
  local pattern="$1"
  local file="$2"
  grep -Eic "$pattern" "$file" 2>/dev/null || true
}

sanitize_name() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

write_hit_file() {
  local pattern="$1"
  local input_file="$2"
  local output_file="$3"
  grep -Ein "$pattern" "$input_file" > "$output_file" || true
}

summarize_kernel_blocks() {
  local tag="$1"
  local blocks_file="$2"
  local out_dir="$3"

  local summary_file="${out_dir}/${tag}_summary.txt"
  local scratch_hits="${out_dir}/${tag}_scratch_hits.txt"
  local lds_hits="${out_dir}/${tag}_lds_hits.txt"
  local wait_hits="${out_dir}/${tag}_wait_hits.txt"
  local wmma_hits="${out_dir}/${tag}_wmma_hits.txt"

  if [[ ! -s "$blocks_file" ]]; then
    cat > "$summary_file" <<EOF
tag=$tag
matched_kernels=0
EOF
    : > "$scratch_hits"
    : > "$lds_hits"
    : > "$wait_hits"
    : > "$wmma_hits"
    return 0
  fi

  local matched_kernels
  matched_kernels="$(grep -Ec '^===== KERNEL ' "$blocks_file" || true)"
  if [[ "$matched_kernels" == "0" ]]; then
    matched_kernels=1
  fi
  local scratch_load_count
  local scratch_store_count
  local ds_read_count
  local ds_write_count
  local waitcnt_count
  local barrier_count
  local wmma_count
  local delay_alu_count

  scratch_load_count="$(count_pattern 'scratch_load' "$blocks_file")"
  scratch_store_count="$(count_pattern 'scratch_store' "$blocks_file")"
  ds_read_count="$(count_pattern 'ds_(read|load)' "$blocks_file")"
  ds_write_count="$(count_pattern 'ds_(write|store)' "$blocks_file")"
  waitcnt_count="$(count_pattern 's_waitcnt' "$blocks_file")"
  barrier_count="$(count_pattern 's_barrier' "$blocks_file")"
  wmma_count="$(count_pattern 'v_wmma' "$blocks_file")"
  delay_alu_count="$(count_pattern 's_delay_alu' "$blocks_file")"

  cat > "$summary_file" <<EOF
tag=$tag
matched_kernels=$matched_kernels
scratch_load=$scratch_load_count
scratch_store=$scratch_store_count
ds_read=$ds_read_count
ds_write=$ds_write_count
s_waitcnt=$waitcnt_count
s_barrier=$barrier_count
v_wmma=$wmma_count
s_delay_alu=$delay_alu_count
EOF

  write_hit_file 'scratch_(load|store)' "$blocks_file" "$scratch_hits"
  write_hit_file 'ds_(read|write|load|store)' "$blocks_file" "$lds_hits"
  write_hit_file 's_waitcnt|s_barrier' "$blocks_file" "$wait_hits"
  write_hit_file 'v_wmma|s_delay_alu' "$blocks_file" "$wmma_hits"
}

read_summary_value() {
  local key="$1"
  local file="$2"
  awk -F= -v key="$key" '$1 == key { print $2; found=1; exit } END { if (!found) print 0 }' "$file"
}

write_compare_summary() {
  local dk_summary="$1"
  local dv_summary="$2"
  local output_file="$3"

  local keys=(
    matched_kernels
    scratch_load
    scratch_store
    ds_read
    ds_write
    s_waitcnt
    s_barrier
    v_wmma
    s_delay_alu
  )

  {
    printf 'metric,dk,dv\n'
    local key
    for key in "${keys[@]}"; do
      printf '%s,%s,%s\n' \
        "$key" \
        "$(read_summary_value "$key" "$dk_summary")" \
        "$(read_summary_value "$key" "$dv_summary")"
    done
  } > "$output_file"
}

normalize_hex_addr() {
  local raw="$1"
  raw="${raw#0x}"
  raw="${raw#0X}"
  printf '0x%x\n' "$((16#$raw))"
}

hex_addr_add() {
  local base="$1"
  local delta="$2"
  base="${base#0x}"
  base="${base#0X}"
  printf '0x%x\n' "$((16#$base + delta))"
}

extract_addr_window() {
  local obj_file="$1"
  local start_addr="$2"
  local stop_addr="$3"
  local out_file="$4"
  "$LLVM_OBJDUMP" -d -C --triple=amdgcn-amd-amdhsa \
    --start-address="$start_addr" \
    --stop-address="$stop_addr" \
    "$obj_file" > "$out_file"
}

find_tool() {
  local name
  for name in "$@"; do
    if [[ -x "$name" ]]; then
      printf '%s\n' "$name"
      return 0
    fi
    if command -v "$name" >/dev/null 2>&1; then
      command -v "$name"
      return 0
    fi
  done
  return 1
}

LLVM_OBJDUMP="$(find_tool /opt/rocm/llvm/bin/llvm-objdump llvm-objdump || true)"
LLVM_READELF="$(find_tool /opt/rocm/llvm/bin/llvm-readelf llvm-readelf || true)"
LLVM_READOBJ="$(find_tool /opt/rocm/llvm/bin/llvm-readobj llvm-readobj || true)"
ROC_OBJ="$(find_tool /opt/rocm/bin/roc-obj roc-obj || true)"
OBJCOPY_TOOL="$(find_tool /usr/bin/objcopy objcopy || true)"
CLANG_BUNDLER="$(find_tool /opt/rocm/llvm/bin/clang-offload-bundler clang-offload-bundler || true)"

if [[ -n "$LLVM_OBJDUMP" ]]; then
  OBJDUMP_TOOL="$LLVM_OBJDUMP"
  TOOL_KIND="llvm"
elif [[ -n "$ROC_OBJ" ]]; then
  OBJDUMP_TOOL="$ROC_OBJ"
  TOOL_KIND="roc-obj"
else
  echo "No llvm-objdump or roc-obj found." >&2
  exit 1
fi

printf 'SO_PATH=%s\n' "$SO_PATH" | tee "$OUT_DIR/meta.txt"
printf 'OUT_DIR=%s\n' "$OUT_DIR" | tee -a "$OUT_DIR/meta.txt"
printf 'TOOL=%s\n' "$OBJDUMP_TOOL" | tee -a "$OUT_DIR/meta.txt"
printf 'TOOL_KIND=%s\n' "$TOOL_KIND" | tee -a "$OUT_DIR/meta.txt"
printf 'LLVM_READELF=%s\n' "${LLVM_READELF:-}" | tee -a "$OUT_DIR/meta.txt"
printf 'LLVM_READOBJ=%s\n' "${LLVM_READOBJ:-}" | tee -a "$OUT_DIR/meta.txt"
printf 'ROC_OBJ=%s\n' "${ROC_OBJ:-}" | tee -a "$OUT_DIR/meta.txt"
printf 'OBJCOPY_TOOL=%s\n' "${OBJCOPY_TOOL:-}" | tee -a "$OUT_DIR/meta.txt"
printf 'CLANG_BUNDLER=%s\n' "${CLANG_BUNDLER:-}" | tee -a "$OUT_DIR/meta.txt"
if [[ -n "$KERNEL_PATTERN" ]]; then
  printf 'KERNEL_PATTERN=%s\n' "$KERNEL_PATTERN" | tee -a "$OUT_DIR/meta.txt"
fi
if [[ "$ANALYZE_BWD" == "1" ]]; then
  printf 'ANALYZE_BWD=1\n' | tee -a "$OUT_DIR/meta.txt"
  printf 'DK_PATTERN=%s\n' "$DK_PATTERN" | tee -a "$OUT_DIR/meta.txt"
  printf 'DV_PATTERN=%s\n' "$DV_PATTERN" | tee -a "$OUT_DIR/meta.txt"
  if [[ -n "$GPU_OBJ_PATH" ]]; then
    printf 'GPU_OBJ_PATH=%s\n' "$GPU_OBJ_PATH" | tee -a "$OUT_DIR/meta.txt"
  fi
  if [[ -n "$DV_ADDR" ]]; then
    printf 'DV_ADDR=%s\n' "$DV_ADDR" | tee -a "$OUT_DIR/meta.txt"
  fi
  if [[ -n "$DK_ADDR" ]]; then
    printf 'DK_ADDR=%s\n' "$DK_ADDR" | tee -a "$OUT_DIR/meta.txt"
  fi
  if [[ -n "$DQ_ADDR" ]]; then
    printf 'DQ_ADDR=%s\n' "$DQ_ADDR" | tee -a "$OUT_DIR/meta.txt"
  fi
fi

if [[ "$TOOL_KIND" == "roc-obj" ]]; then
  "$OBJDUMP_TOOL" --symbols "$SO_PATH" > "$OUT_DIR/symbols.txt"
  "$OBJDUMP_TOOL" --amdhsa "$SO_PATH" > "$OUT_DIR/amdhsa.txt"
  "$OBJDUMP_TOOL" --disassemble "$SO_PATH" > "$OUT_DIR/disassembly.txt"
else
  "$OBJDUMP_TOOL" -t "$SO_PATH" > "$OUT_DIR/symbols.txt"
  if [[ -n "$LLVM_READELF" ]]; then
    "$LLVM_READELF" -n "$SO_PATH" > "$OUT_DIR/amdhsa.txt" || true
  elif [[ -n "$LLVM_READOBJ" ]]; then
    "$LLVM_READOBJ" --notes "$SO_PATH" > "$OUT_DIR/amdhsa.txt" || true
  elif [[ -n "$ROC_OBJ" ]]; then
    "$ROC_OBJ" --amdhsa "$SO_PATH" > "$OUT_DIR/amdhsa.txt" || true
  else
    : > "$OUT_DIR/amdhsa.txt"
  fi
  "$OBJDUMP_TOOL" -d "$SO_PATH" > "$OUT_DIR/disassembly.txt"
fi

GPU_DISASM_PATH=""

if [[ "$ANALYZE_BWD" == "1" ]]; then
  FATBIN_SECTION=""
  if [[ -n "$LLVM_READELF" ]]; then
    FATBIN_SECTION="$("$LLVM_READELF" -S "$SO_PATH" 2>/dev/null | awk '/\.hip_fatbin|\.hipFatBinSegment/ {print $2; exit}')"
  else
    FATBIN_SECTION="$(readelf -S "$SO_PATH" 2>/dev/null | awk '/\.hip_fatbin|\.hipFatBinSegment/ {print $2; exit}')"
  fi

  if [[ -n "$FATBIN_SECTION" && -n "$OBJCOPY_TOOL" && -n "$CLANG_BUNDLER" && -n "$LLVM_OBJDUMP" ]]; then
    FATBIN_PATH="$OUT_DIR/hip_fatbin.bin"
    BUNDLES_PATH="$OUT_DIR/bundles.txt"
    GPU_DIR="$OUT_DIR/gpu_code_objects"
    GPU_DISASM_PATH="$OUT_DIR/amdgcn_disassembly.txt"

    mkdir -p "$GPU_DIR"
    "$OBJCOPY_TOOL" --dump-section "${FATBIN_SECTION}=${FATBIN_PATH}" "$SO_PATH"
    "$CLANG_BUNDLER" --type=o --input="$FATBIN_PATH" --list > "$BUNDLES_PATH"
    strings "$FATBIN_PATH" > "$OUT_DIR/fatbin_strings.txt" || true

    : > "$GPU_DISASM_PATH"
    bundle_index=0
    while IFS= read -r bundle_target; do
      [[ -z "$bundle_target" ]] && continue
      [[ "$bundle_target" != *amdgcn* ]] && continue
      bundle_index=$((bundle_index + 1))
      safe_target="$(sanitize_name "$bundle_target")"
      bundle_out="$GPU_DIR/${bundle_index}_${safe_target}.o"
      bundle_disasm="$GPU_DIR/${bundle_index}_${safe_target}.disasm.txt"
      "$CLANG_BUNDLER" --type=o \
        --input="$FATBIN_PATH" \
        --targets="$bundle_target" \
        --output="$bundle_out" \
        --unbundle
      "$LLVM_OBJDUMP" -d -C --triple=amdgcn-amd-amdhsa "$bundle_out" > "$bundle_disasm"
      {
        printf '===== BUNDLE %d: %s =====\n' "$bundle_index" "$bundle_target"
        cat "$bundle_disasm"
        printf '\n'
      } >> "$GPU_DISASM_PATH"
    done < "$BUNDLES_PATH"

    printf 'FATBIN_SECTION=%s\n' "$FATBIN_SECTION" | tee -a "$OUT_DIR/meta.txt"
    printf 'FATBIN_PATH=%s\n' "$FATBIN_PATH" | tee -a "$OUT_DIR/meta.txt"
    printf 'BUNDLES_PATH=%s\n' "$BUNDLES_PATH" | tee -a "$OUT_DIR/meta.txt"
    printf 'GPU_DISASM_PATH=%s\n' "$GPU_DISASM_PATH" | tee -a "$OUT_DIR/meta.txt"
  fi
fi

if [[ -n "$KERNEL_PATTERN" ]]; then
  if command -v rg >/dev/null 2>&1; then
    rg -n "$KERNEL_PATTERN" "$OUT_DIR/symbols.txt" > "$OUT_DIR/symbol_hits.txt" || true
    rg -n "$KERNEL_PATTERN" "$OUT_DIR/disassembly.txt" > "$OUT_DIR/disasm_hits.txt" || true
  else
    grep -En "$KERNEL_PATTERN" "$OUT_DIR/symbols.txt" > "$OUT_DIR/symbol_hits.txt" || true
    grep -En "$KERNEL_PATTERN" "$OUT_DIR/disassembly.txt" > "$OUT_DIR/disasm_hits.txt" || true
  fi
fi

if [[ "$ANALYZE_BWD" == "1" ]]; then
  BWD_DIR="$OUT_DIR/bwd_isa"
  ANALYZE_INPUT="$OUT_DIR/disassembly.txt"
  if [[ -n "$GPU_DISASM_PATH" && -s "$GPU_DISASM_PATH" ]]; then
    ANALYZE_INPUT="$GPU_DISASM_PATH"
  fi
  mkdir -p "$BWD_DIR"

  if [[ -f "$OUT_DIR/fatbin_strings.txt" ]]; then
    grep -E 'FmhaBwdDQDKDVKernel|FmhaBwdConvertQGradKernel|BlockFmhaBwdDKDVPipelineKRKTRVRIGLP|BlockFmhaBwdDQPipelineKRKTRVRIGLP' \
      "$OUT_DIR/fatbin_strings.txt" > "$BWD_DIR/kernel_name_inventory.txt" || true
    grep -E 'BlockFmhaBwdDKDVPipelineKRKTRVRIGLP.*ELb1ELb0' \
      "$OUT_DIR/fatbin_strings.txt" > "$BWD_DIR/dk_name_candidates.txt" || true
    grep -E 'BlockFmhaBwdDKDVPipelineKRKTRVRIGLP.*ELb0ELb1' \
      "$OUT_DIR/fatbin_strings.txt" > "$BWD_DIR/dv_name_candidates.txt" || true
  fi

  extract_kernel_blocks "$DK_PATTERN" "$ANALYZE_INPUT" \
    "$BWD_DIR/dk_labels.txt" "$BWD_DIR/dk_kernel_blocks.txt"
  extract_kernel_blocks "$DV_PATTERN" "$ANALYZE_INPUT" \
    "$BWD_DIR/dv_labels.txt" "$BWD_DIR/dv_kernel_blocks.txt"

  summarize_kernel_blocks "dk" "$BWD_DIR/dk_kernel_blocks.txt" "$BWD_DIR"
  summarize_kernel_blocks "dv" "$BWD_DIR/dv_kernel_blocks.txt" "$BWD_DIR"
  write_compare_summary "$BWD_DIR/dk_summary.txt" "$BWD_DIR/dv_summary.txt" \
    "$BWD_DIR/compare_summary.txt"

  if [[ -n "$GPU_OBJ_PATH" && -n "$DK_ADDR" && -n "$DV_ADDR" ]]; then
    norm_dv_addr="$(normalize_hex_addr "$DV_ADDR")"
    norm_dk_addr="$(normalize_hex_addr "$DK_ADDR")"
    if [[ -n "$DQ_ADDR" ]]; then
      norm_dq_addr="$(normalize_hex_addr "$DQ_ADDR")"
    else
      norm_dq_addr=""
    fi

    dv_stop_addr="$norm_dk_addr"
    dk_stop_addr="$norm_dq_addr"

    if [[ -z "$dk_stop_addr" ]]; then
      dk_stop_addr="$(hex_addr_add "$norm_dk_addr" 0x8000)"
    fi

    extract_addr_window "$GPU_OBJ_PATH" "$norm_dv_addr" "$dv_stop_addr" \
      "$BWD_DIR/dv_addr_window.s"
    extract_addr_window "$GPU_OBJ_PATH" "$norm_dk_addr" "$dk_stop_addr" \
      "$BWD_DIR/dk_addr_window.s"

    summarize_kernel_blocks "dv_addr" "$BWD_DIR/dv_addr_window.s" "$BWD_DIR"
    summarize_kernel_blocks "dk_addr" "$BWD_DIR/dk_addr_window.s" "$BWD_DIR"

    printf 'NORM_DV_ADDR=%s\n' "$norm_dv_addr" | tee -a "$OUT_DIR/meta.txt"
    printf 'NORM_DK_ADDR=%s\n' "$norm_dk_addr" | tee -a "$OUT_DIR/meta.txt"
    printf 'DV_STOP_ADDR=%s\n' "$dv_stop_addr" | tee -a "$OUT_DIR/meta.txt"
    printf 'DK_STOP_ADDR=%s\n' "$dk_stop_addr" | tee -a "$OUT_DIR/meta.txt"
  fi
  printf 'ANALYZE_INPUT=%s\n' "$ANALYZE_INPUT" | tee -a "$OUT_DIR/meta.txt"
fi

cat <<EOF
Done.

Main files:
  $OUT_DIR/meta.txt
  $OUT_DIR/symbols.txt
  $OUT_DIR/amdhsa.txt
  $OUT_DIR/disassembly.txt
EOF

if [[ -n "$KERNEL_PATTERN" ]]; then
  cat <<EOF

Pattern hit files:
  $OUT_DIR/symbol_hits.txt
  $OUT_DIR/disasm_hits.txt

Suggested next step:
  sed -n 'START,ENDp' $OUT_DIR/disassembly.txt
EOF
fi

if [[ "$ANALYZE_BWD" == "1" ]]; then
  cat <<EOF

BWD ISA analysis files:
  $OUT_DIR/bwd_isa/dk_labels.txt
  $OUT_DIR/bwd_isa/dv_labels.txt
  $OUT_DIR/bwd_isa/dk_kernel_blocks.txt
  $OUT_DIR/bwd_isa/dv_kernel_blocks.txt
  $OUT_DIR/bwd_isa/dk_summary.txt
  $OUT_DIR/bwd_isa/dv_summary.txt
  $OUT_DIR/bwd_isa/compare_summary.txt
  $OUT_DIR/bwd_isa/dk_scratch_hits.txt
  $OUT_DIR/bwd_isa/dv_scratch_hits.txt
  $OUT_DIR/bwd_isa/dk_lds_hits.txt
  $OUT_DIR/bwd_isa/dv_lds_hits.txt
  $OUT_DIR/bwd_isa/dk_wait_hits.txt
  $OUT_DIR/bwd_isa/dv_wait_hits.txt
  $OUT_DIR/bwd_isa/dk_wmma_hits.txt
  $OUT_DIR/bwd_isa/dv_wmma_hits.txt
  $OUT_DIR/bwd_isa/kernel_name_inventory.txt
  $OUT_DIR/bwd_isa/dk_name_candidates.txt
  $OUT_DIR/bwd_isa/dv_name_candidates.txt
  $OUT_DIR/bwd_isa/dv_addr_window.s
  $OUT_DIR/bwd_isa/dk_addr_window.s
  $OUT_DIR/bwd_isa/dv_addr_summary.txt
  $OUT_DIR/bwd_isa/dk_addr_summary.txt
EOF
  if [[ -n "$GPU_DISASM_PATH" && -s "$GPU_DISASM_PATH" ]]; then
    cat <<EOF
  $OUT_DIR/hip_fatbin.bin
  $OUT_DIR/fatbin_strings.txt
  $OUT_DIR/bundles.txt
  $OUT_DIR/amdgcn_disassembly.txt
  $OUT_DIR/gpu_code_objects/
EOF
  fi
fi
