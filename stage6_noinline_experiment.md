# Stage 6 `noinline` Experiment

## Goal

Try a more aggressive spill-reduction step on the active `dk-only split_dk` path by forcing a hard register-allocation boundary around `Stage 6`.

The intended target was:

- `SGradTFromGemm2CToGemm3A(...)`
- `gemm_3(...)`
- the heavy Stage-6 state around:
  - `dp_acc`
  - `dst_reg_tensor`
  - `qt_reg_tensor`

## Baseline Before This Experiment

This experiment was applied on top of an already improved baseline that had:

- hoisted `dst_reg_tensor` out of the loop for reuse
- hoisted several uniform origins/offsets into scalar locals
- removed `ds_gemm = cast_tile<GemmDataType>(dp_acc)` materialization
- moved cast into the `SGradT` LDS transpose path

That baseline reduced profiler scratch to roughly:

- `dv-only`: `28`
- active `dk-only split_dk`: `780`
- `dq`: `308`

and still preserved correctness.

## Code Change

The experiment introduced a dedicated `[[gnu::noinline]]` helper:

- `RunStage6(...)`

It wrapped:

- `SGradTFromGemm2CToGemm3A(...)`
- `gemm_3(...)`
- the trailing `kHasBiasGrad` sync

Call sites in the active `split_dk` hot loop and tail were changed to route Stage 6 through that helper.

Both:

- `block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp`
- `block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp_hip.hpp`

were updated, so this was not a stale `_hip` sync issue.

## Profiler Result

`Scratch_Per_Workitem` on the active `dk-only split_dk` line improved further:

- `780 -> 640`

So the `noinline` boundary did affect register allocation in the expected direction.

## Correctness Result

Correctness regressed badly:

- `S=64` remained correct
- `S=128` already failed
- larger exact multiples of `64` also failed
- the failure was broad on `dk`, not a tiny numerical drift

Representative user-observed pattern:

- `S=64`: OK
- `S=128`: BAD
- `S=256`: BAD
- `S=512`: BAD
- `S=1024`: BAD
- `S=2048`: BAD

This is important because the failure is not tied to partial tails:

- `128`, `256`, `512`, `1024`, `2048` are all exact multiples of `64`

## Interpretation

The most likely cause is not the earlier `780`-baseline changes.

The strongest new variable was the device-call boundary itself:

- `dk_acc`
- `dst_reg_tensor`
- `dp_acc` / `ds_gemm`
- `qt_reg_tensor`

now had to cross a separate `noinline` device helper boundary.

That likely changed how large `static_distributed_tensor` / `thread_buffer` objects were materialized, passed, or restored in the hot-loop Stage-6 path.

The key signature is:

- scratch improved
- correctness broke immediately once the hot loop was exercised

That is consistent with "register-pressure improved, but ABI/codegen assumptions for Stage-6 state were disturbed".

## Conclusion

This experiment is negative and should stay rolled back.

Durable conclusion:

- do not force Stage 6 across a separate device-call boundary in this path
- especially do not move `static_distributed_tensor` / `thread_buffer` heavy Stage-6 state across such a boundary here

Keep the previous `780`-scratch baseline instead.

## Current Recommended Baseline

Keep:

- `dst_reg_tensor` hoisted out of the loop
- uniform origin/offset scalarization
- fused cast inside the `SGradT` transpose path

Do not keep:

- `RunStage6(...)`
- `[[gnu::noinline]]` Stage-6 split

## Next Direction

If further pressure reduction is needed, prefer:

- more intra-function live-range reduction
- reducing overlap among `dp_acc`, `qt_reg_tensor`, and bias/dbias temporaries
- avoiding extra materialized tiles

Avoid:

- crossing a device-call boundary with large Stage-6 register objects

## Tile-Size Dependency Context

Current active backward instance uses a shape equivalent to:

- `kM0=32`
- `kN0=64`
- `kK0=64`
- `kK1=32`
- `kK2=64`
- `kK3=32`
- `kK4=32`
- `kQKHeaddim=64`
- `kVHeaddim=64`

For the active `split_dk` path, the direct source-level hard constraints are:

- `kQKHeaddim >= kK0`
- `kM0 == kK1`
- `kVHeaddim >= kK2`
- `kM0 == kK3`

Those come directly from the active pipeline body.

There is also a window-shape consistency requirement:

- the `QDram/KDram/VDram/Bias/OGrad/LSE/D/QGrad/BiasGrad` temporary windows must match
  `kM0` / `kN0` exactly

In addition, each GEMM tile must still divide cleanly against:

- `BlockWarps`
- `WarpTile`

With the current generated warp layout, this means:

- `kM0` must stay a multiple of `16`
- `kN0` is effectively pinned to a multiple of `64` under the current block-warps choices
- `kK0`, `kK1`, `kK2`, `kK3`, `kK4` must stay multiples of `16`

Important practical consequence:

- under the current block-warps layout, shrinking `kN0` alone is not a legal "small tweak"
- shrinking `kM0` is much easier, but it forces:
  - `kK1`
  - `kK3`
  to shrink with it

## Practical Smaller Shape Candidates

The following candidates are the most plausible first trials if the goal is to reduce scratch
without rewriting the whole warp/block mapping:

### Candidate A: halve `kM0`

Use:

- `sequence<16,64,64,16,64,16,16,64,64>`

Why:

- this directly targets the `dp_acc` / `dst_reg_tensor` / Stage-6 footprint
- it satisfies the explicit `kM0 == kK1 == kK3` dependency
- it keeps `kN0=64`, so it does not violate the current `BlockWarps * WarpTile` divisibility on
  the N side

Tradeoff:

- more loop iterations along sequence tiles
- lower per-tile reuse
- but it is the cleanest "reduce register peak" knob under the current warp layout

### Candidate B: keep `kM0=32`, shorten head-dim staging

Use:

- `sequence<32,64,32,32,32,32,32,64,64>`

Why:

- leaves the M/N tile geometry unchanged
- reduces `kK0` and `kK2` from `64` to `32`
- may lower pressure in `gemm_0` / `gemm_2` staging without touching the Stage-6 block shape

Tradeoff:

- more inner-K iterations over the full head dimension
- less attractive if the real hotspot is only Stage 6

### Candidate C: combine both

Use:

- `sequence<16,64,32,16,32,16,16,64,64>`

Why:

- this is the most aggressive shape reduction that still fits the current visible hard
  dependencies without changing `kN0`
- it attacks:
  - sequence-tile footprint
  - `gemm_0`
  - `gemm_2`
  - Stage 6

Tradeoff:

- highest risk of performance regression from extra iteration overhead
- but also the strongest chance of materially lowering scratch

## What Not To Try First

- Do not try shrinking only `kN0` under the current block-warps layout.
  With the current generated `BlockWarps` / `WarpTile`, `kN0=32` would break the clean
  `NIterPerWarp` divisibility assumptions.

- Do not change only `kK1` or only `kK3`.
  The active pipeline explicitly requires:
  - `kM0 == kK1`
  - `kM0 == kK3`

- If the same shape is still shared with a `dq` path, also re-check `kK4`.
  For a dedicated `dk-only` shape, `kK4` is less important.
  For a shared shape, it still has to satisfy the `gemm_4` divisibility path.

## Recommended Order

If the goal is scratch reduction with minimal structural change, try in this order:

1. Candidate A
2. Candidate B
3. Candidate C

Rationale:

- Candidate A is the cleanest Stage-6-focused reduction
- Candidate B is the least invasive for overall tile geometry
- Candidate C is the strongest but most likely to move performance in both directions

## How Tile Entries Are Routed

The real source of truth for backward tile selection is:

- `csrc/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`

The `FmhaBwdDQDKDVTileSize` entry is expanded into:

- `fmha_block_tile_{idx}`
- `fmha_bwd_shape_{idx}`
- `fmha_bwd_pipeline_problem_{idx}`
- `fmha_bwd_dq_dk_dv_kernel_{idx}`

Dispatch does not branch directly on every tile field.
Instead, it first matches a generated trait key on:

- dtype
- mode (`batch` / `group`)
- mask
- bias / dbias
- dropout
- deterministic
- `dcheck`
- `dvcheck`
- optional `max_seq_q`

The tile entry then comes along with that generated trait.

Important practical implication:

- replacing an existing tile entry is risky because it silently changes the implementation behind
  an already valid dispatch key
- adding a new entry and letting the generator emit a second kernel is safer for A/B work

## Additional Dependency Notes

The earlier Candidate A failure showed that changing:

- `kM0`
- `kK1`
- `kK3`

is not a local tweak.

Even though it satisfied the explicit hard constraints:

- `kM0 == kK1`
- `kM0 == kK3`

it also changed, at the same time:

- `gemm_1` K iteration count
- `gemm_3` K iteration count
- `QT` LDS shape `(kQKHeaddim, kM0)`
- `SGradT` reg-slice shape `(kN0, kK3)`
- the outer hot-loop count `ceil(seqlen_q / kM0)`

That is why it behaved like a whole-pipeline perturbation rather than a simple scratch-only tweak.

## Recommended Next Candidate

The next safer candidate should avoid changing:

- `kM0`
- `kK1`
- `kK3`
- `kK4`

and instead only reduce the `gemm_2` unroll depth.

Recommended candidate:

- `sequence<32,64,64,32,32,32,32,64,64>`

Compared with the current baseline:

- keep `kM0 = 32`
- keep `kN0 = 64`
- keep `kK0 = 64`
- keep `kK1 = 32`
- change `kK2: 64 -> 32`
- keep `kK3 = 32`
- keep `kK4 = 32`
- keep `kQKHeaddim = 64`
- keep `kVHeaddim = 64`

Why this is safer:

- it does not change loop count along sequence tiles
- it does not change the `QT` / `SGradT` shape pair used by Stage 6
- it does not perturb the `dq`-side `kK4`
- it specifically targets `gemm_2`, which is the producer of `dp_acc`, one of the remaining large
  live tensors on the active `dk-only split_dk` path

Expected effect:

- lower `gemm_2` register pressure
- possibly lower `dp_acc`-related pressure before Stage 6
- lower scratch with much smaller risk of broad functional breakage than Candidate A

## Recommended Trial Order

From here, the best order is:

1. Add a new entry for `sequence<32,64,64,32,32,32,32,64,64>` instead of replacing the baseline
2. Validate correctness first
3. Then compare `Scratch_Per_Workitem`
4. Only if that is stable, consider a later trial on `kK0`

## Update After Candidate B

Candidate B, which changed only:

- `kK2: 64 -> 32`

also failed correctness.

This is an important correction to the earlier planning notes.

Why this happened:

- in the active `split_dk` path, `gemm_2` is executed as a single call on the current tile
- there is no extra outer-K accumulation loop around `gemm_2`
- `gemm_2` shape is built from:
  - `(kM0, kN0, kK2)`
- but the OGrad/V staging still represents the full `kVHeaddim`

So under the current pipeline structure, reducing `kK2` is not a harmless register-tuning knob.
It changes the actual mathematical work done by `gemm_2` unless a matching outer-K loop is added.

This means the earlier "safer next candidate" idea was too optimistic.

## Revised Practical Guidance

For the current monolithic backward pipeline, the following fields should be treated as
structural rather than freely tunable:

- `kK0`
- `kK1`
- `kK2`
- `kK3`

Reason:

- `gemm_0`, `gemm_1`, `gemm_2`, and `gemm_3` are each consumed as single-call stages in the
  active path
- only `gemm_4` has an explicit outer loop structure in the `dq` side

So in practice:

- `kK1` and `kK3` are already explicitly tied to `kM0`
- `kK2` is effectively tied to the full `V` reduction extent of `gemm_2`
- `kK0` should also be treated as structurally coupled to the full Q/K reduction extent unless a
  matching outer-K loop is introduced

## Revised Next Recommendation

Do not continue inventing smaller `TileFmhaBwdShape` tuples inside the current shared pipeline.

The next safer direction is:

1. keep the current working tile tuple
2. continue reducing pressure by code-shape / specialization work
3. if tile-size experimentation is still needed, do it only after introducing a dedicated
   `dk-only` specialized pipeline or explicit extra-K loop structure

In other words:

- current pipeline: structural/code-shape tuning is still the safer lever
- new tile tuples: should wait for a more specialized kernel structure
