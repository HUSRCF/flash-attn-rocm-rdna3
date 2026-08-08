# Building the frozen ROCm CK release

This repository is a source-complete snapshot for the RDNA3 `gfx1100` target.
Composable Kernel (CK), CUTLASS, rocThrust, and rocPRIM are ordinary
directories under `csrc/`; they are not Git submodules. A clone already
contains the native-source build closure: do
not run `git submodule update`, and no vendored source download is part of a
build.

The GPU toolchain is deliberately not vendored. Linux, ROCm, a matching
ROCm-enabled PyTorch installation, Python build tooling, CMake, and Ninja must
already be available in the build environment. The exact environment used for
release validation is recorded in `VENDORED_DEPENDENCIES.json`; other
combinations are not implied to be release-validated.

Compiler compatibility is performance-sensitive even when compilation and
correctness both succeed. The accepted artifact was linked from the frozen
ROCm 7.2/Clang 22 object closure matching its PyTorch ROCm runtime. A diagnostic
rebuild with the host ROCm 7.14/Clang 23 compiler changed the large legacy FWD
dispatcher and slowed existing short and non-causal routes despite leaving the
device payload of the selected kernel unchanged. Use a compiler matching the
PyTorch ROCm major/minor version for release-comparable performance.

The rocThrust and rocPRIM headers needed by PyTorch HIP extension compilation are
included under `csrc/`. A separate system rocThrust or rocPRIM development
is therefore not required for this frozen build.

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

Install a ROCm-enabled PyTorch build using the distribution appropriate for the
host before installing this repository's dependencies. PyTorch 2.9 or newer is
recommended; versions below 2.9 are not guaranteed. PyTorch is deliberately
absent from the requirements files and wheel metadata so pip cannot replace a
working ROCm build with an incompatible generic package.

Verify PyTorch first, then install the small Python dependency set:

```bash
python -c "import torch; assert torch.version.hip, 'install ROCm-enabled PyTorch first'; print(torch.__version__, torch.version.hip)"
python -m pip install -r requirements-build.txt
python -c "import torch, einops, packaging, psutil, ninja"
```

The requirements file does not install PyTorch or select a ROCm runtime.

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
  environment without compiling kernels. It reports PyTorch ROCm and compiler
  ROCm separately and warns when their major/minor versions differ.
- `make vendor-check` checks the complete frozen vendored file inventory as
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
`dQ`, `dK`, and `dV` must all be finite. For each tensor, candidate MAE must be
at most twice the BF16-math MAE against FP32, and candidate maximum absolute
error must be at most twice the BF16-math maximum plus one BF16 ULP at the
tensor-wide FP32 scale. These aggregate bounds remain meaningful for values
produced by cancellation near zero. These constants are empirical frozen bounds
calibrated on the accepted safe-O and corrected native-O artifacts for these
three rows; they are not a universal BF16 error theorem.

The former pointwise envelope (four times BF16-math error plus four local ULPs)
is retained in the JSON/CSV as `local_envelope_*` diagnostics, but it is not a
hard gate: its allowance collapses near zero and rejects the accepted safe-O
baseline. The audit also verifies that the vendored BF16 round-to-nearest
inline-assembly helper retains its required early-clobber output constraint.
The source guard covers the known register-alias regression. The aggregate FP32
gate is not a substitute for a safe/native bitwise pair and is not claimed to
detect every sparse one-ULP bias. The aggregate FP32 comparison and source
guard participate in the exit code.
The aggregate release target also rejects altered matrix or tolerance values
before compilation.

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

## Conditional gfx11 FP16/D128 causal forward path

The frozen full build includes an additional standard batch FWD specialization
for FP16/backend-D128 on gfx11. It uses the QR row-V pipeline, a `128 x 64`
query/K tile, occupancy 4, sequence padding, exact backend head dimensions,
LSE, no bias, no dropout, and the native-O epilogue. Code generation restricts
the specialization to the validated pure-causal feature domain; it is not
generated for non-causal, BF16, group/varlen, or other feature combinations.
Dispatch rejects unpadded partial backend dimensions. A user-level dimension
such as 127 is padded by the Python interface to backend D128 and is therefore
intentionally treated as a D128 call before the output is sliced back to the
requested width.

In standard batch mode, `max_seqlen_q` is the runtime query-length bound and
equals `seqlen_q` for the fixed-length calls validated here. The runtime gate
requires an unbounded left window, a zero right window,
`max_seqlen_q >= 2048`, `seqlen_k >= 768`, and `batch * nhead_q >= 4`.
Non-causal calls always retain the existing dispatch.

The new trait is deliberately excluded from the generated legacy `fmha_fwd_v2`
API table. The full build links a small GNU-linker wrapper in front of
`fmha_fwd`; it checks the K threshold first, calls the new trait directly only
when every gate matches, and otherwise calls the unmodified legacy dispatcher.
This keeps existing `128 x 32`, short-sequence, non-causal, BF16, varlen, and
feature-rich routes out of the large-dispatcher compiler-layout perturbation
that was observed during development. Minimal debug builds omit both the
wrapper and its linker option.

The accepted GPU1 precision validation covered the causal boundary cases with
an FP32 oracle. All seven focused cases passed, including the padded user-level
D127 case and exact backend D128 routing; focused FP16 D128 FWD+BWD validation
also passed all four non-causal/causal S2048/S2560 cases. The largest observed
focused FWD absolute error was 0.001953125.

The final clean two-round full-vs-full ABBA measurements used 500 warmups, 50
trials of 20 iterations, and a 10 percent trimmed mean. Non-causal Q=2560 at
K=256/1024/2048/4096 measured 1.0014x, 0.9998x, 0.9996x, and 1.0000x versus the
pre-change c18 package. Causal Q=2048 at K=1024/2048/4096 measured 1.2466x,
1.3333x, and 1.1484x. The K256 fallback measured 0.9941x in that sweep; a
separate 100-trial boundary sweep measured 0.9951x at K256, 0.9991x at K384,
and gains from 1.0215x through 1.1947x over K512 through K1024. The production
K gate remains 768 so only the clear-gain region selects the new kernel.

In simplified-mask builds, the wrapper checks both the runtime mask enum and
the decoded window bounds. It accepts only a pure top-left or bottom-right
causal mask with an unbounded left window and a zero right window;
sliding-window and generic masks call the unchanged legacy dispatcher.

`CK_TILE_DISABLE_FWD_D128_FP16_B128X64_O4=1` is a developer-only minimal-build
rollback. The frozen full-release source manifest assumes the accepted default
is enabled, so do not set this override for `make build-full` or release wheels.

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
- the vendored CK and CUTLASS source trees plus rocThrust and rocPRIM header
  trees;
- the exact identity manifest for the full generated `gfx1100` kernel set;
- `VENDORED_FILES.txt`, the complete vendored path inventory checked against
  the unpacked archive;
- the root and all four vendored dependency license/notice files;
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
- If `make doctor` warns that PyTorch ROCm and compiler ROCm differ, the source
  build may still complete, but do not treat it as release-performance-
  equivalent. This host's observed combination is PyTorch ROCm 7.2 with the
  ROCm 7.14 / Clang 23 compiler.
- If Ninja jobs are killed or the host starts swapping, retry with a smaller
  `MAX_JOBS` value.
- If `make vendor-check` fails, the clone or source archive is incomplete.
  Reacquire the repository; there is no submodule repair step.
- Use `make clean` before comparing source-generated kernel counts across build
  profiles.
