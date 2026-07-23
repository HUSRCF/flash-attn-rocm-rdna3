# Vendored dependencies

This source release is a frozen monorepo. It does not require Git submodules or source downloads during a build.

## Composable Kernel

- Upstream: https://github.com/ROCm/composable_kernel.git
- Frozen revision: recorded in `VENDORED_DEPENDENCIES.json`
- Local state: the c18 RDNA3 FMHA/GEMM changes and four new canonical headers are included directly under `csrc/composable_kernel`.
- License: MIT; see `csrc/composable_kernel/LICENSE`.

## CUTLASS

- Upstream: https://github.com/NVIDIA/cutlass.git
- Frozen revision: recorded in `VENDORED_DEPENDENCIES.json`
- Local state: clean upstream snapshot at freeze time.
- License: BSD-3-Clause; see `csrc/cutlass/LICENSE.txt`.
- Additional files under the vendored tree retain their own license notices.

## rocThrust

- Upstream: https://github.com/ROCm/rocThrust.git
- Frozen release tag and version: recorded in `VENDORED_DEPENDENCIES.json`.
- Local state: complete header tree plus the generated version header under
  `csrc/rocthrust`.
- License: Apache-2.0; see `csrc/rocthrust/LICENSE`.
- Bundled third-party notices and additional licenses: see
  `csrc/rocthrust/NOTICES.txt`; individual headers retain their notices.

## rocPRIM

- Upstream: https://github.com/ROCm/rocPRIM.git
- Frozen release tag and version: recorded in `VENDORED_DEPENDENCIES.json`.
- Local state: complete public header tree plus the generated version header
  under `csrc/rocprim`.
- License: MIT; see `csrc/rocprim/LICENSE.md`.
- Bundled third-party notices and additional licenses: see
  `csrc/rocprim/NOTICES.txt`; individual headers retain their notices.

System ROCm, a compatible ROCm PyTorch installation, Python build tools, and
Ninja remain environment dependencies; they are not vendored into this
repository.
