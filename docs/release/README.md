# Frozen release evidence

This directory contains compact evidence for the c18 gfx1100 source release.

- `RESULTS.md` is the accepted package-wide build and precision report.
- `VENDORED_DEPENDENCIES.json` records exact source provenance and the validated toolchain.
- Large tensors, profiler captures, ISA dumps, intermediate binaries, and historical candidates remain in the local release2 archive and are intentionally excluded from source Git.
- The validated extension is distributed as a release asset rather than a Git blob.
