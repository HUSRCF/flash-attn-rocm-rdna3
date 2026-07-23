# Building the frozen ROCm CK release

This repository is a source-complete snapshot for the RDNA3 `gfx1100` target.
Composable Kernel (CK) and CUTLASS are ordinary directories under `csrc/`, not
Git submodules. A clone already contains the native-source build closure: do
not run `git submodule update`, and no CK/CUTLASS source download is part of a
build.

The GPU toolchain is deliberately not vendored. Linux, ROCm, a matching
ROCm-enabled PyTorch installation, Python build tooling, CMake, and Ninja must
already be available in the build environment. The exact environment used for
release validation is recorded in `VENDORED_DEPENDENCIES.json`; other
combinations are not implied to be release-validated.

## Quick start

From a normal clone:

```bash
make doctor
make vendor-check
make test-smoke
```

`make test-smoke` first invokes the minimal build. That build defaults to FP16,
head dimension 64, the standard batch family, and `gfx1100`. It is intended to
establish that the toolchain, vendored source closure, extension import, and a
small forward/backward path work before starting the much larger release build.
Use `make build-minimal` instead when compilation without a GPU test is wanted.

For the complete frozen kernel set and release validation:

```bash
make test-release MAX_JOBS=8 DEVICE=cuda:0
```

`make test-release` first builds the complete closure. Use `make build-full`
when only compilation is wanted. `MAX_JOBS` controls Ninja parallelism; lower
it when host memory is limited. The full build is intentionally large.
`make help` lists all supported targets and override variables.

## Dependencies

Install the ROCm/PyTorch stack using the distribution appropriate for the host.
Then install or verify the small Python dependency set:

```bash
python -m pip install -r requirements-build.txt
python -c "import torch, einops, packaging, psutil, ninja"
```

The requirements file does not vendor or select a ROCm runtime. In particular,
do not replace a working ROCm-enabled PyTorch installation with a generic
PyTorch wheel.

For a machine without package-network access, provision the environment first
or point pip at a local wheelhouse:

```bash
python -m pip install --no-index --find-links /path/to/wheelhouse \
  -r requirements-build.txt
```

Once those external dependencies and this clone are present, the source build
itself is local. The release commands disable remote wheel lookup explicitly.

## Stable build targets

- `make doctor` checks the Python, PyTorch, ROCm, Ninja, and compiler-facing
  environment without compiling kernels.
- `make vendor-check` checks the complete frozen CK/CUTLASS file inventory as
  well as required source and license sentinels.
- `make freeze-check` combines release-tree structural checks.
- `make build-minimal` builds the small diagnostic kernel closure in place.
- `make build-full` builds the full frozen CK kernel closure in place.
- `make wheel` builds a local wheel; it does not fetch an upstream wheel.
- `make sdist` creates the source archive.
- `make verify-sdist` extracts the archive, checks the vendored closure and
  metadata, and by default compiles the minimal closure from that extracted
  tree. Set `VERIFY_SDIST_BUILD=0` only for a quick structure-only check.
- `make test-smoke` builds and validates the minimal extension.
- `make test-bf16-fp32-audit` reruns the three accepted BF16 boundary rows
  against both BF16 math-SDPA and a true FP32-input/backward oracle. It requires
  a content-bound full-build marker and verifies the imported extension comes
  from this checkout and matches that marker.
- `make test-release` builds the full extension and runs the package release
  standard-batch matrix, focused FP32 boundary audit, and frozen official nodes.
- `make test-release-pytest` runs the frozen set of 40 supported official CK
  nodes covering reference-relative standard cases, exact deterministic
  repeats, and varlen cases. Its post-check requires exactly 40 executed tests
  and rejects skipped nodes.
- `make clean` removes reproducible local build products.

Common development overrides are `PYTHON`, `GPU_ARCHS`, `MAX_JOBS`, `DEVICE`,
and the `MINIMAL_*` variables shown by `make help`. The aggregate frozen release
gate accepts only `gfx1100` and the documented matrix, repeat count, gradient
scale, and tolerances; changing those values is rejected before compilation.

## Release correctness policy

The default standard-batch release matrix covers head dimensions 32, 64, 128, and 256;
sequence lengths 768, 1024, and 2048; causal and non-causal attention; and two
repeats. It runs FP16 and BF16 separately so the lower-precision format is not
judged by an accidental one-size-fits-all threshold. Results are written below
`testoutput/make-release/fp16` and `testoutput/make-release/bf16`.

The default gates are:

- FP16: `FP16_FWD_ATOL=2.0e-3`,
  `FP16_BWD_SCALED_ATOL=3.0e-3`, and `FP16_BWD_REL_ATOL=1.0e-3`.
- BF16 strict path: `BF16_FWD_ATOL=2.0e-2`,
  `BF16_BWD_SCALED_ATOL=3.0e-2`, and `BF16_BWD_REL_ATOL=3.0e-2`.
- BF16 boundary fallback: scaled backward error at most `4.1e-2`, relative
  backward error at most `8.0e-3`, and a distance of at most one BF16 ULP at
  the element attaining the maximum absolute backward error.

The BF16 fallback does not assert a whole-tensor one-ULP bound. It is allowed
only for the three scenarios identified by the frozen matrix; the release
target rejects any newly appearing fallback scenario. It then reruns those
rows with promoted FP32 inputs and FP32 backward tensors. Candidate output,
`dQ`, `dK`, and `dV` must all be finite and remain inside an error envelope
formed as four times the BF16-math distance to FP32 plus four times the local
BF16 spacing at the FP32 reference value. The floor therefore shrinks near
zero instead of becoming a unit-scale absolute tolerance. This FP32 comparison
participates in the exit code; it is not report-only. The
aggregate release target also rejects altered matrix or tolerance values before
compilation.

The release target therefore covers standard batch plus the tracked official
standard, deterministic, and varlen node set. The broader accepted strict
coverage for varlen/group layouts, SplitKV, PagedKV, AppendKV, cache mutation,
and feature corners is summarized in `docs/release/RESULTS.md`; those external
matrix results are retained as frozen evidence, and should not be inferred to
have been rerun merely from this Make target.

## Direct local build

The Makefile is the stable entry point. If integration requires invoking the
Python build directly, the equivalent full in-place build contract is:

```bash
env BUILD_TARGET=rocm \
  FLASH_ATTN_CK_PROFILE=release \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
  FLASH_ATTENTION_ALLOW_REMOTE_WHEEL=FALSE \
  GPU_ARCHS=gfx1100 MAX_JOBS=8 \
  python setup.py build_ext --inplace --force
```

Do not carry experimental `CK_TILE_*` overrides into a release-profile build.
The accepted native-O and D64-only backward choices are frozen in source
defaults. A development archive may also contain an ignored `command.md` lab
notebook; its historical commands intentionally use non-release overrides and
are not part of the source release.

To build and install a wheel locally:

```bash
make wheel MAX_JOBS=8
python -m pip install --no-deps dist/*.whl
```

`--no-deps` is appropriate after the external ROCm/PyTorch environment has
already been provisioned and prevents pip from trying to resolve it again.

## Source archive contents

`make sdist` includes:

- FlashAttention Python and native extension sources;
- the vendored CK and CUTLASS source trees, including code generation inputs;
- the exact identity manifest for the full generated `gfx1100` kernel set;
- `VENDORED_FILES.txt`, the complete CK/CUTLASS path inventory checked against
  the unpacked archive;
- the root, CK, and CUTLASS license/notice files;
- release provenance and compact validation documentation; and
- the release helper scripts and tests used by the stable targets.

It excludes generated headers, object/shared-library artifacts, compiler build
trees, profiler captures, tensors, and historical experiment archives. Those
outputs are reproducible or retained outside source Git; they are not required
to compile a fresh clone.

Use `make verify-sdist` for the strongest archive check: it proves that the
unpacked archive can generate and compile the minimal extension without using
the original checkout. `make verify-sdist VERIFY_SDIST_BUILD=0` checks only the
archive structure and setup metadata and therefore is not sufficient evidence
of build closure by itself.

## Troubleshooting

- If `make doctor` reports a non-ROCm PyTorch build, install the matching ROCm
  PyTorch package before continuing.
- If Ninja jobs are killed or the host starts swapping, retry with a smaller
  `MAX_JOBS` value.
- If `make vendor-check` fails, the clone or source archive is incomplete.
  Reacquire the repository; there is no submodule repair step.
- Use `make clean` before comparing source-generated kernel counts across build
  profiles.
