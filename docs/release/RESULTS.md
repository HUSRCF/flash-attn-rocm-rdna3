# c18 package-wide gfx1100 build and precision results

Date: 2026-07-21

All valid GPU runs used physical GPU1 through a single visibility variable:

```text
HIP_VISIBLE_DEVICES=1
architecture=gfx1100
```

## Build

The build disabled CK minimal filtering and force-recompiled the package-wide ROCm
integration for gfx1100. It retained the accepted source defaults:

- QR pipeline and the validated native-O per-instance whitelist.
- D64-only BWD PT and SGradT register remaps.
- Default LDS BWD paths for D128/D256.
- FP16 and BF16 integration kernels; FP32 is used as the numerical oracle.

Generated source coverage before compilation:

```text
standard FWD: approximately 1362 files
BWD: 913 files
AppendKV: 97 files
SplitKV/PagedKV: 691 files
```

Build result:

```text
3070/3070 Ninja steps completed
compiler errors: 0
extension size: 401752320 bytes
```

Immutable artifact:

```text
testoutput/c18_full_build_accuracy_20260721/binaries/c18_full_gfx1100_fp16_bf16.so
```

The checksum is stored in `binaries/SHA256SUMS` and is intentionally not repeated here.

## Precision and runtime validation

### Import and deterministic D256 boundary

```text
full extension import on gfx1100: PASS
S128 D256 FP16 deterministic BWD: PASS
S128 D256 BF16 deterministic BWD: PASS
```

### Standard batch SDPA-math matrix

The matrix covered D32/D64/D128/D256, FP16/BF16, causal/noncausal,
S768/S1024/S2048, two seeds, FWD+BWD, and gradient scale 100.

```text
total runs: 96
generic strict gate: 93/96
FP16: 48/48
BF16: 45/48
```

Maximum observed errors:

```text
FP16 max FWD absolute error: 0.001953125
FP16 max scaled BWD absolute error: 0.0025
FP16 max relative BWD error: 0.000949667616334283

BF16 max FWD absolute error: 0.015625
BF16 max scaled BWD absolute error: 0.04
BF16 max relative BWD error: 0.007772020725388601
```

The generic BWD absolute threshold is 0.03. The three BF16 misses were causal
S768/D64, S2048/D64, and S2048/D256. A focused bit-level audit reproduced all
three and showed that the maximum dV discrepancy is exactly one BF16 ULP in each
case. Their maximum relative gradient error remains below 0.8 percent. These are
BF16 quantization-boundary misses of the generic absolute gate. More precisely,
the element attaining the maximum absolute dV error is one ULP away from the
reference in each case; this focused audit does not claim a whole-tensor ULP bound.

Follow-up diagnosis against a true FP32 SDPA-math backward oracle found:

```text
accepted source-default artifact vs current full artifact: dV bitwise equal in all 3 cases
candidate FP32 MAE / PyTorch-BF16 FP32 MAE: approximately 1.59 to 1.60
toward-zero vs away-from-zero mismatches: approximately balanced
maximum absolute error among multi-ULP near-zero differences: 1.0 before scale normalization
```

The larger ULP counts occur only near zero, where ULP distance is not a useful
standalone magnitude metric. Each case has only one element above an absolute
dV difference of 3, and that maximum-error element is the one-ULP boundary already
described above.

Gradient-scale ablation further showed:

```text
scale 1/64/128 normalized maximum: 0.015625 for the D64 cases
scale 1/64/128 normalized maximum: 0.03125 for the D256 case
scale 100 normalized maximum: 0.04 for all three cases
```

Thus the fixed `bwd_scaled <= 0.03` generic gate is not BF16-ULP aware and the
non-power-of-two scale 100 makes the discrete maximum worse. The evidence points
to a gate-calibration false negative and is consistent with implementation or
reduction-order rounding. It rules out a regression introduced by the current
workspace fix/full build, but does not claim CK is identical to the FP32 oracle or
exclude every pre-existing implementation-level numerical difference.

The upstream CK reference-relative tests independently adjudicated these paths:

```text
official standard precision nodes: 22/22 PASS
```

Those nodes include deterministic FP16/BF16 D32/D64/D128/D256, partial head
dimensions, long causal BF16 S2048 cases, and dropout/ALiBi/MQA/GQA feature corners.

### Strict source-default comparison

The full binary was compared against the previously accepted D64-only BWD
source-default artifact over D64/D128/D256, FP16/BF16, causal/local boundaries,
and sequences through S4096:

```text
tensor rows passing pair gates: 200/200
FP32-reference rows passing: 200/200
stable tensor rows bitwise equal: 174
Split-dQ nondeterministic rows inside frozen envelope: 26/26
maximum reference-envelope ratio: 0.3176930546760559
```

### Deterministic repeatability

Eight official nodes covered D32/D64/D128/D256 and FP16/BF16. Each node repeated
BWD 50 times and required exact equality for dQ, dK, and dV:

```text
8/8 PASS
```

### Varlen/group

Official varlen FWD+BWD oracle nodes:

```text
supported combinations: 9/9 PASS
D256 BF16 dropout case rerun with deterministic=False: PASS
```

One deliberately selected combination is rejected by an existing Python guard:

```text
gfx11 D256 deterministic varlen BWD with nonzero dropout: unsupported
```

Its FWD numerical assertions pass before the guard rejects BWD. The same feature
combination passes when deterministic mode is disabled. This is a known capability
gap, not a numerical mismatch from the full build.

The strict independent varlen/group FWD validator covered MHA/GQA, causal/local,
boundary/decode, paged/nonpaged mappings, FP16/BF16, and D64/D128/D256:

```text
unique scenarios: 72/72 PASS
tensor rows repeat-stable: 144/144
oracle values checked: 15456672
```

### SplitKV, PagedKV, and AppendKV

The strict cache validator covered splits 1/2, paged/nonpaged, causal/noncausal,
MHA/GQA/local attention, append, rotary, cache mutation, FP16/BF16, and
D64/D128/D256:

```text
unique scenarios: 108/108 PASS
tensor rows repeat-stable: 288/288
oracle values checked: 1528928
```

## FP16/D128 causal FWD addendum (2026-08-07)

The final package adds one gfx11 FP16/backend-D128 causal FWD specialization
with a `128 x 64` query/K tile and occupancy 4. It is reached through a small
outer wrapper and is excluded from the generated legacy API table. The legacy
API source generated by the final tree is byte-for-byte equal to the accepted
ROCm 7.2 c18 source. The complete identity manifest contains 3064 generated
sources, exactly one of which is the new specialization.

The final extension contains the complete frozen kernel closure and is
401789312 bytes. Validation used PyTorch 2.12.1 with ROCm 7.2 on physical GPU1.
The standard-batch matrix covered FP16 and BF16, D32/D64/D128/D256,
causal/non-causal attention, S768/S1024/S2048, two seeds, and FWD+BWD:

```text
FP16 strict: 48/48 PASS
BF16: 48/48 PASS (45 strict, 3 documented one-ULP boundary gates)
BF16 FP32-oracle boundary audit: 3/3 PASS
frozen official CK nodes: 40/40 PASS
```

Clean two-round full-package ABBA measurements used 500 warmups, 50 trials of
20 iterations, and a 10 percent trimmed mean. Non-causal Q=2560 at
K=256/1024/2048/4096 remained between 0.9996x and 1.0014x of the pre-change c18
package. Causal Q=2048 improved by 1.2466x at K1024, 1.3333x at K2048, and
1.1484x at K4096. A separate 100-trial boundary sweep measured 1.1411x,
1.1867x, and 1.1947x at K768/K896/K1024. K256 and K384 retain the legacy route;
their measured ratios were 0.9951x and 0.9991x.

The performance artifact was linked from the matching ROCm 7.2/Clang 22 object
closure. A ROCm 7.14/Clang 23 diagnostic rebuild changed legacy host-dispatcher
performance despite matching selected device-kernel payloads, so it is not used
as release-comparable evidence.

## Conclusion

The package-wide gfx1100 build is usable for the tested supported domains. The
workspace-size fix is present in the full artifact, and no supported-path precision
regression was identified by the oracle and pair gates used here. The only uncovered
capability is the already guarded D256 deterministic varlen BWD plus nonzero-dropout
combination.
