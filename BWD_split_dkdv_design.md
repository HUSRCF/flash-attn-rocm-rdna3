# BWD Split DKDV Design

目标：把当前 `split_dkdv` 再拆成两套专门 kernel，而不是继续在同一套 `dkdv-only` pipeline 上抠寄存器生命周期。

## 背景

当前 `gfx11`, `fp16`, `D=64`, `noncausal`, `batch` 主路径上：

- `split_dq` 已稳定在 `Scratch_Per_Workitem = 308`
- `split_dkdv` 经多轮优化后，稳定在 `~792`
- 再尝试用更多 LDS 换 `K/V` 不常驻寄存器，会让 `LDS_Per_Workgroup` 明显膨胀，并导致性能倒退

结论：

- `split_dkdv` 剩余负担已经主要是结构性的，不再是简单的 live-range 问题
- 下一步应该拆 `dV` 和 `dK`，让它们不再共享同一套寄存器 / LDS / 中间转排负担

## 依赖拆解

`dV` 侧：

- 依赖 `gemm_0: Q @ K^T -> P`
- 依赖 softmax / dropout 产生的 `P`
- 依赖 `gemm_1: P^T @ dO^T -> dV`
- 不依赖 `V`
- 不依赖 `gemm_2 / gemm_3`

`dK` 侧：

- 依赖 `gemm_0: Q @ K^T -> P`
- 依赖 softmax / dropout 产生的 `P`
- 依赖 `gemm_2: dO @ V -> dS`
- 依赖 stage5: `dS = P * (dP - D)`
- 依赖 `gemm_3: dS^T @ Q^T -> dK`
- 不依赖 `gemm_1`

共享部分只有：

- `dot_do_o`
- `gemm_0 + bias/mask/softmax/dropout`

## 建议路线

推荐最终形态：

- `dot_do_o`
- `dv-only`
- `dk-only`
- `dq-only`
- `convert_dq`

其中：

- `dv-only` 只保留 `P` 到 `dV` 的链路
- `dk-only` 只保留 `P` 到 `dK` 的链路
- 不要再使用“同一 pipeline + bool 分支关功能”的方式，把未使用路径留在编译器视野里

## 已知正确性约束

- `Q^T` 必须在覆写 `shuffled_q` 之前读取
- `CK_TILE_FMHA_BWD_WMMA_PT_LDS_REMAP=1`
- `CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP=1`
- 只有当 `CWarpDstr` 与 `AWarpDstr` thread-buffer 语义兼容时，才允许 bypass

## 实施顺序

1. 先在 codegen / API 层加二次拆分路由骨架
2. 优先实现 `dv-only` pipeline
3. 再实现 `dk-only` pipeline
4. 最后接入二次 split dispatch，并逐 case 验证

## 验证门槛

- `test_ck_tile_probe.py` 至少覆盖 `S=64/128/256/512`
- kernel name 必须能区分 `split_dv` 与 `split_dk`
- `rocprof` 重点看两件事：
  - `split_dv` 和 `split_dk` 的 `Scratch_Per_Workitem` 都应明显低于当前 `split_dkdv`
  - 不允许再出现通过扩大 LDS 换 scratch 而导致整体性能倒退
- benchmark 以总 `bwd` 时间为准，不以单项 profile 漂亮为准
