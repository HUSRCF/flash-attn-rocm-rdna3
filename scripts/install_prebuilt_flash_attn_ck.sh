#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/install_prebuilt_flash_attn_ck.sh bundle [output.tar.gz]
  ./scripts/install_prebuilt_flash_attn_ck.sh install
  ./scripts/install_prebuilt_flash_attn_ck.sh check

Environment:
  PYTHON=python                         Python interpreter for target env.
  SITE_PACKAGES=/path/to/site-packages  Override install destination.
  ALLOW_ARCH_MISMATCH=1                 Allow non-gfx1100 GPU arch.
  ALLOW_RUNTIME_MISMATCH=1              Allow non-PyTorch-2.12/ROCm-7.2 runtime.
  SKIP_GPU_CHECK=1                      Skip runtime GPU arch check.

Notes:
  This installs the prebuilt FA4 CK package and flash_attn_2_cuda*.so.
  It does not compile anything.
  Import torch before importing flash_attn_2_cuda.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

warn() {
    echo "WARN: $*" >&2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

PYTHON_BIN="${PYTHON:-python}"
MODE="${1:-install}"
shift || true

find_so() {
    shopt -s nullglob
    local candidates=("$ROOT_DIR"/flash_attn_2_cuda*.so)
    shopt -u nullglob
    [[ ${#candidates[@]} -eq 1 ]] || die "expected exactly one flash_attn_2_cuda*.so under $ROOT_DIR, got ${#candidates[@]}"
    printf '%s\n' "${candidates[0]}"
}

python_ext_suffix() {
    "$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or "")
PY
}

python_site_packages() {
    "$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
}

check_python_abi() {
    local so_path="$1"
    local ext_suffix
    ext_suffix="$(python_ext_suffix)"
    [[ -n "$ext_suffix" ]] || die "cannot read Python EXT_SUFFIX from $PYTHON_BIN"
    case "$(basename "$so_path")" in
        *"$ext_suffix") ;;
        *)
            die "SO ABI suffix mismatch: so=$(basename "$so_path"), python_ext_suffix=$ext_suffix. Use the same Python ABI or rebuild."
            ;;
    esac
}

check_torch_and_gpu() {
    "$PYTHON_BIN" - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: cannot import torch in target Python env: {exc!r}")

print(f"torch={torch.__version__}, hip={getattr(torch.version, 'hip', None)}")

torch_series = ".".join(torch.__version__.split("+", 1)[0].split(".")[:2])
hip_version = str(getattr(torch.version, "hip", "") or "")
hip_series = ".".join(hip_version.split(".")[:2])
runtime_mismatches = []
if torch_series != "2.12":
    runtime_mismatches.append(f"PyTorch {torch_series} (expected 2.12)")
if hip_series != "7.2":
    runtime_mismatches.append(f"PyTorch HIP {hip_series or 'none'} (expected 7.2)")
if runtime_mismatches and os.environ.get("ALLOW_RUNTIME_MISMATCH") != "1":
    raise SystemExit(
        "ERROR: prebuilt runtime mismatch: "
        + ", ".join(runtime_mismatches)
        + ". This binary is validated for PyTorch 2.12 / ROCm 7.2. "
        "Set ALLOW_RUNTIME_MISMATCH=1 only after independent validation."
    )

if os.environ.get("SKIP_GPU_CHECK") == "1":
    print("gpu_check=skipped")
    raise SystemExit(0)

if not torch.cuda.is_available():
    print("gpu_check=warning_no_visible_gpu")
    raise SystemExit(0)

prop = torch.cuda.get_device_properties(0)
arch = getattr(prop, "gcnArchName", "") or ""
print(f"device={prop.name}, arch={arch}")
if "gfx1100" not in arch and os.environ.get("ALLOW_ARCH_MISMATCH") != "1":
    raise SystemExit(
        "ERROR: this prebuilt SO was built for gfx1100. "
        f"Visible GPU arch is {arch!r}. Set ALLOW_ARCH_MISMATCH=1 only if you accept the risk."
    )
PY
}

install_package() {
    local so_path="$1"
    local site_packages="${SITE_PACKAGES:-}"
    [[ -d "$ROOT_DIR/flash_attn" ]] || die "missing flash_attn package under $ROOT_DIR"
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "cannot find PYTHON=$PYTHON_BIN"
    command -v rsync >/dev/null 2>&1 || die "rsync is required for install"

    check_python_abi "$so_path"
    check_torch_and_gpu

    if [[ -z "$site_packages" ]]; then
        site_packages="$(python_site_packages)"
    fi
    mkdir -p "$site_packages"

    echo "Installing flash_attn package to $site_packages/flash_attn"
    rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' "$ROOT_DIR/flash_attn/" "$site_packages/flash_attn/"

    echo "Installing extension $(basename "$so_path") to $site_packages"
    install -m 0755 "$so_path" "$site_packages/$(basename "$so_path")"

    local dist_info="$site_packages/flash_attn_fa4_prebuilt-4.0.0.dist-info"
    mkdir -p "$dist_info"
    cat >"$dist_info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: flash-attn-fa4-prebuilt
Version: 4.0.0
Summary: Prebuilt FA4 CK FlashAttention package for ROCm gfx1100.
Requires-Dist: einops
EOF
    cat >"$dist_info/WHEEL" <<'EOF'
Wheel-Version: 1.0
Generator: install_prebuilt_flash_attn_ck.sh
Root-Is-Purelib: false
Tag: py3-none-any
EOF
    printf 'install_prebuilt_flash_attn_ck.sh\n' >"$dist_info/INSTALLER"
    : >"$dist_info/RECORD"

    verify_import installed
}

verify_import() {
    local mode="${1:-installed}"
    (
        cd /tmp
        if [[ "$mode" == "local" ]]; then
            export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
        fi
        "$PYTHON_BIN" - <<'PY'
import torch
import flash_attn
import flash_attn_2_cuda
print(f"import_ok torch={torch.__version__} hip={getattr(torch.version, 'hip', None)} flash_attn={flash_attn.__file__}")
print(f"extension_ok {flash_attn_2_cuda.__file__}")
PY
    )
}

make_bundle() {
    local so_path="$1"
    local output="${2:-}"
    local default_name="flash_attn_fa4_c18_prebuilt_gfx1100_py312_rocm72.tar.gz"
    output="${output:-$ROOT_DIR/$default_name}"
    [[ -d "$ROOT_DIR/flash_attn" ]] || die "missing flash_attn package under $ROOT_DIR"
    command -v rsync >/dev/null 2>&1 || die "rsync is required for bundle"
    command -v tar >/dev/null 2>&1 || die "tar is required for bundle"

    local tmp
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/flash_attn_fa4_prebuilt.XXXXXX")"
    BUNDLE_TMP_DIR="$tmp"
    trap 'if [[ -n "${BUNDLE_TMP_DIR:-}" ]]; then rm -rf "$BUNDLE_TMP_DIR"; fi' EXIT

    local bundle_root="$tmp/flash-attn-fa4-prebuilt"
    mkdir -p "$bundle_root/scripts"
    rsync -a --exclude='__pycache__' --exclude='*.pyc' "$ROOT_DIR/flash_attn/" "$bundle_root/flash_attn/"
    install -m 0755 "$so_path" "$bundle_root/$(basename "$so_path")"
    install -m 0755 "$SCRIPT_DIR/$(basename "$0")" "$bundle_root/scripts/$(basename "$0")"
    if [[ -f "$SCRIPT_DIR/test_prebuilt_flash_attn_ck.py" ]]; then
        install -m 0755 "$SCRIPT_DIR/test_prebuilt_flash_attn_ck.py" "$bundle_root/scripts/test_prebuilt_flash_attn_ck.py"
    fi

    if [[ -f "$ROOT_DIR/PREBUILT_INSTALL.md" ]]; then
        install -m 0644 "$ROOT_DIR/PREBUILT_INSTALL.md" "$bundle_root/PREBUILT_INSTALL.md"
    else
        cat >"$bundle_root/PREBUILT_INSTALL.md" <<'EOF'
# FA4 CK Prebuilt Install

Run:

```bash
PYTHON=python ./scripts/install_prebuilt_flash_attn_ck.sh install
```

The extension was built for Python 3.12, ROCm 7.2, and gfx1100.
EOF
    fi

    cat >"$bundle_root/PREBUILT_MANIFEST.txt" <<EOF
source_root=$ROOT_DIR
so=$(basename "$so_path")
target_arch=gfx1100
python_abi=cpython-312-x86_64-linux-gnu
artifact_runtime=pytorch-2.12_rocm-7.2
host_compiler_note=not_used_by_prebuilt_install
created_at=$(date -Iseconds)
EOF

    tar -C "$tmp" -czf "$output" flash-attn-fa4-prebuilt
    echo "Wrote bundle: $output"
}

SO_PATH="$(find_so)"

case "$MODE" in
    install)
        install_package "$SO_PATH"
        ;;
    bundle)
        make_bundle "$SO_PATH" "${1:-}"
        ;;
    check)
        check_python_abi "$SO_PATH"
        check_torch_and_gpu
        verify_import local
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        die "unknown mode: $MODE"
        ;;
esac
