#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/husrcf/anaconda3/bin/python}"
IFACE_FILE="${IFACE_FILE:-/home/husrcf/anaconda3/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py}"

log() {
  printf '[codex] %s\n' "$*"
}

doctor() {
  log "python=${PYTHON_BIN}"
  log "iface=${IFACE_FILE}"
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

iface = Path(os.environ.get("IFACE_FILE", "/home/husrcf/anaconda3/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py"))
print(f"FLASH_ATTENTION_TRITON_AMD_ENABLE={os.getenv('FLASH_ATTENTION_TRITON_AMD_ENABLE')}")
print(f"iface_exists={iface.exists()}")

if iface.exists():
    s = iface.read_text(encoding="utf-8")
    old = "        None,\n        alibi_slopes,\n        dropout_p,\n"
    new = "        None,\n        alibi_slopes,\n        None,\n        None,\n        None,\n        dropout_p,\n"
    print(f"iface_already_patched={new in s}")
    print(f"iface_legacy_call_pattern={old in s}")

import torch
import flash_attn_2_cuda
doc = (flash_attn_2_cuda.fwd.__doc__ or "").splitlines()[0]
print(f"fwd_signature={doc}")
PY
}

patch_iface() {
  log "patching ${IFACE_FILE}"
  IFACE_FILE="${IFACE_FILE}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

p = Path(os.environ["IFACE_FILE"])
if not p.exists():
    raise SystemExit(f"iface not found: {p}")

s = p.read_text(encoding="utf-8")
old = "        None,\n        alibi_slopes,\n        dropout_p,\n"
new = "        None,\n        alibi_slopes,\n        None,\n        None,\n        None,\n        dropout_p,\n"

if new in s:
    print("already patched")
elif old in s:
    s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    print(f"patched: {p}")
else:
    raise SystemExit("target snippet not found, please check interface file version")
PY
}

run_ck() {
  if [ "$#" -eq 0 ]; then
    log "usage: ./codex.sh run <command...>"
    exit 2
  fi
  log "run with FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE: $*"
  env FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE "$@"
}

usage() {
  cat <<'EOF'
Usage:
  ./codex.sh doctor
  ./codex.sh patch
  ./codex.sh sync
  ./codex.sh run <command...>

Optional env:
  PYTHON_BIN=/path/to/python
  IFACE_FILE=/path/to/flash_attn_interface.py
EOF
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    doctor)
      shift
      doctor "$@"
      ;;
    patch)
      shift
      patch_iface "$@"
      ;;
    sync)
      shift
      doctor
      patch_iface
      doctor
      ;;
    run)
      shift
      run_ck "$@"
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      log "unknown command: ${cmd}"
      usage
      exit 2
      ;;
  esac
}

main "$@"
