# Flash-Attention CK `split_dk` Debug Handoff

最后更新: 2026-03-22

本文档用于把当前 `RDNA3 / gfx1100 / CK backward dK 随机错误` 的排查状态完整交接给下一位继续处理的人。

## 1. 问题摘要

当前问题发生在 AMD RDNA3 `gfx1100` 上的 CK backward 路径，表现为:

- `dK` 随机错误
- `dQ` 和 `dV` 基本稳定
- `S=512` 正常
- `S=1024` 和更大长度会随机 `OK/BAD`
- `512 -> 1024` 与 “一开始直接 1024” 走的是同一套 kernel 路径，不是 kernel 切换问题

当前最准确的定位表述是:

- 问题发生在 CK backward 的 `_split_dk` 路径
- 当前证据已经把主要怀疑从 `gemm_3 / dk_acc` 本体往后推
- 下一步最应该查的是 `dk` 的 epilogue/store 路径

## 2. 当前最重要结论

### 2.1 当前 repro 走的不是 fused `dq_dk_dv`，而是 `_split_dk`

真实运行路径是:

- kernel 名里包含 `_split_dk`
- active pipeline 是 `BlockFmhaBwdDKDVPipelineKRKTRVRIGLP`
- 不是 `BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP`
- active `KGradEpiloguePipeline` 是 codegen 的
  `Default2DEpilogue<Default2DEpilogueProblem<AccDataType, KGradDataType, false, (dpad > 0)>>`

对应代码:

- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp)
- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp)

### 2.2 `S=1024` 的 good/bad 在 return 前指纹一致

已经加入并验证过的 return-stage probe:

- `DK_RETURN_T0`

结论:

- `S=1024` 的 `OK` 和 `BAD` 共享同一个 `DK_RETURN_T0 digest`

这意味着:

- 当前这套跨 block return-stage 指纹没有抓到 `dk_acc` 漂移
- 问题不能再优先怀疑 `gemm_3` 主体计算已经明显分叉

### 2.3 Host 侧 whole-tensor 统计已经证明错误是真实存在于 `dk_expanded`

Host 侧 probe:

- `DK_HOST_GLOBAL`

结论:

- bad run 的 host tensor 统计会明显漂移
- 不是单纯测试 harness 误判
- 不是只在单个 device probe 采样点上“没撞到”

对应代码:

- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp)

### 2.4 当前 repro 不走 host 侧 `sum_out(dk)` merge

当前 case:

- `Hq == Hk == 4`

因此在 host 代码里:

- 不进入 MQA/GQA 的 `sum_out(dk)` merge 路径

这很重要，因为它把“host merge/reduction”从当前主嫌疑中移除了。

对应代码:

- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L629)

## 3. 已排除或基本排除的方向

以下方向已经做过实验，当前不应再作为第一优先级重复投入:

### 3.1 指针对齐问题

已检查:

- Python 层 `data_ptr() % 256`

结论:

- `S=1024` 时输入输出相关指针满足对齐
- 不是显存池对齐导致向量化 load/store 失效

### 3.2 简单同步缺失

已尝试:

- 开启 `CK_TILE_DEBUG_BWD_FORCE_STAGE6_SYNC`
- 插入更强 `block_sync_lds()`

结论:

- 不能消除错误

### 3.3 显式 `s_waitcnt` / waitcnt 补丁

已尝试:

- 在怀疑位置插入 `s_waitcnt vmcnt(0) lgkmcnt(0)`

结论:

- 不能消除错误

### 3.4 VOPD / scheduler 重排

已尝试:

- 禁用 VOPD
- scheduler 相关试验

结论:

- 不能单独解释问题

### 3.5 单点/单 block 追 ground zero

已证明不适合作为主线:

- bad block 会漂
- 静态点位 probe 命中率低
- 对“全图漂移式错误”证据价值弱

### 3.6 `CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP` 作为 perf 开关

结论:

- 它不是单纯 perf 开关
- 在 WMMA 路径上它属于真实数据布局转换路径的一部分
- 关掉后“整体不准”是符合源码逻辑的，不能作为有效 A/B

## 4. 当前代码里保留的主线 probe

### 4.1 Pipeline return 阶段

文件:

- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp)

保留:

- `DK_RETURN_T0`

说明:

- `DK_RETURN_FP` 已从真源码中删除
- `DK_RETURN_TARGET` 已从当前主线源码中删除
- 单点 return 采样没有比 `DK_RETURN_T0` 提供更多证据，还增加编译和日志负担

### 4.2 Kernel epilogue/store 阶段

文件:

- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp)

保留:

- `DK_EPI_CAST_T0`
- `DK_EPI_T0`
- `DK_EPI_MODE`

`DK_EPI_CAST_T0` 设计目的:

- 在 `KGradEpiloguePipeline{}(...)` 之前
- 对 `cast_tile<KGradDataType>(dk_acc_tile)` 做 thread-0 block fingerprint
- 用于判断 cast 是否已经分叉

`DK_EPI_T0` 设计目的:

- 在 `KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr)` 之后
- 从当前 block 对应的 `dk_dram_window` 直接回读 global store 结果
- 由 thread 0 做 block 级 fingerprint
- 用于判断:
  - `cast 后` 是否仍稳定
  - `epilogue/store 后` 是否开始分叉

`DK_EPI_MODE` 设计目的:

- 直接在 `dk` kernel 侧打印 active epilogue store mode
- 输出 `UseRawStore / CK_TILE_DEBUG_NO_RAW_STORE / CK_TILE_DEBUG_FORCE_STORE_PATH / kPadM / kPadN`
- 用于避免把 trait 上的 `UseRawStore=1` 误读成“baseline 实际走 raw-store”

### 4.3 Host 侧

文件:

- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp)

保留:

- `DK_HOST_GLOBAL`
- `FLASH_ATTN_CK_BWD_DEBUG`

## 5. 已删除或降级的 probe

### 5.1 已删除

- `DK_RETURN_FP`
- `DK_RETURN_TARGET`
- `DK_LOGICAL_PRE`
- `DK_LOGICAL_POST`
- `DK_RETURN_CTX`
- `DK_EPI_CTX`
- `DK_HOST_BH`

原因:

- 对当前问题没有超过 `DK_RETURN_T0` / `DK_EPI_CAST_T0` / `DK_EPI_T0` 的新增证据
- 增加编译压力
- 增加日志体积

### 5.2 已保留但不是默认主线

- `SGRADT_CTX`
- `SGRADT_PRE`
- `SGRADT_POST`

当前定位:

- 这些 probe 曾用于证明 sampled remap 点位是自洽的
- 但它们不能证明整个 remap 阶段无问题
- 默认不建议继续作为主线输出

### 5.3 已暂停的路线

- Python chunked backward / chunk policy 方案

状态:

- 已尝试写 Python 原型
- 无论走 custom-op wrapper 还是 raw backend `flash_attn_gpu.bwd`
- 都会在手工调用 chunked `bwd` 时触发:
  - `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage`

当前结论:

- 这条路线不是“数学上不对”
- 而是当前 backend 接口语义/调用约束下无法直接这样外部手工分块
- 现阶段已暂停，不作为主线继续投入

相关文件:

- [safe_rdna3_chunked_bwd.py](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/safe_rdna3_chunked_bwd.py)
- [test_ck_tile_probe_chunked.py](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/test_ck_tile_probe_chunked.py)

## 6. 日志分析脚本现状

文件:

- [analyze_dbg_log.py](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/analyze_dbg_log.py)

当前功能:

- 正确按 host section 切分日志
- 解析 `DK_HOST_GLOBAL`
- 关联对应 case 的 `RESULT`
- 解析 `DK_RETURN_T0`
- 解析 `DK_EPI_CAST_T0`
- 解析 `DK_EPI_T0`
- 输出 digest summary

注意:

- 之前脚本曾有 section 错绑 bug
- 会把一个 case 的 `HOST_GLOBAL` 关联到后一个 case 的 `DK_EPI_CTX/RESULT`
- 这个 bug 已修复

因此:

- 早先出现的 `HOST_ARGMAX_OOB` 结论不应再使用

## 7. 当前最可靠的证据链

可以把当前状态压缩成下面这条逻辑链:

1. `S=1024` bad run 真实存在，且 `dK` 错误明显
2. `DK_HOST_GLOBAL/BH` 证明错误已经体现在 host 看到的 `dk_expanded`
3. `DK_RETURN_T0 digest` 在 `S=1024` 的 good/bad 之间一致
4. `DK_EPI_CAST_T0 digest` 在 `S=1024` 的 good/bad 之间也一致
5. `DK_EPI_T0 digest` 在 `S=1024` 的 good/bad 之间开始分叉
6. 当前 `Hq == Hk`，host 不做 `sum_out(dk)` merge
7. 所以最合理的下一跳不是 host merge，不是 `gemm_3` 本体，也不是 cast，而是:
   - kernel 内 epilogue store/writeback 路径

补充:

- `DK_EPI_MODE` 已证明 baseline 为 `raw=1 no_raw=0 force=0 pad_m=0 pad_n=0`
- 因为 `pad_m == 0 && pad_n == 0`，baseline 实际不走默认 raw-store 分支，而是 normal-store
- `force normal` 与 baseline 基本一致，`force raw` 更差

## 8. 下一步最应该做什么

### 主线优先级 1

重编后收集新日志，重点看:

- `DK_RETURN_T0`
- `DK_EPI_CAST_T0`
- `DK_EPI_T0`
- `DK_HOST_GLOBAL`
- `DK_EPI_MODE`

推荐 grep:

```bash
rg "DK_RETURN_T0|DK_EPI_CAST_T0|DK_EPI_T0|DK_EPI_MODE|DK_HOST_GLOBAL|GROUND ZERO|\\[OK\\]|\\[BAD\\]" run_dbg_kname.log
```

推荐分析:

```bash
python analyze_dbg_log.py run_dbg_kname.log
```

### 判定规则

#### 情况 A

如果:

- `DK_RETURN_T0` good/bad 一致
- `DK_EPI_CAST_T0` good/bad 也一致
- `DK_EPI_T0` good/bad 开始分叉

那么:

- 直接锁定 epilogue store/writeback 路径
- 优先做 `raw store` vs `normal store` A/B

#### 情况 B

如果:

- `DK_RETURN_T0` good/bad 一致
- `DK_EPI_CAST_T0` good/bad 开始分叉

那么:

- cast-to-output dtype 或 cast 前后的 epilogue 输入布局需要优先排查

#### 情况 C

如果:

- `DK_RETURN_T0` good/bad 一致
- `DK_EPI_CAST_T0` 也一致
- `DK_EPI_T0` 也一致
- 但 host `DK_HOST_GLOBAL` 分叉

那么:

- 说明还要继续向后查更外层 kernel store/materialization
- 但当前 repro 不走 host merge，所以应继续查 kernel-side global write visibility / layout interpretation

#### 情况 D

如果:

- `DK_RETURN_T0` 已经分叉

那么:

- 之前“return 前稳定”的结论被新证据推翻
- 需要回退到 `gemm_3 / dk_acc` 一侧重新定位

## 9. 当前不要做什么

下面这些事当前不推荐继续做:

- 不要再追单个漂移 block id
- 不要再用 `DK_RETURN_FP`
- 不要再恢复 `DK_RETURN_TARGET` / `DK_LOGICAL_PRE` / `DK_LOGICAL_POST` 到主线
- 不要把 `CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP` 当成 perf 开关乱关
- 不要优先继续投入 Python chunked backward
- 不要在 `_hip` 文件里手改后当真源码依据

## 9.1 当前最值钱的 A/B

直接针对 epilogue store 分支：

- `CK_TILE_DEBUG_NO_RAW_STORE=1`
- `CK_TILE_DEBUG_FORCE_STORE_PATH=-1`
- `CK_TILE_DEBUG_FORCE_STORE_PATH=1`
- `CK_TILE_DEBUG_FORCE_ROW_STORE=1`

对应源码：

- [default_2d_epilogue.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/epilogue/default_2d_epilogue.hpp#L720)

辅助日志 tag：

- `DK_EPI_MODE`

当前结论:

- baseline 已经是 normal-store
- `force normal` 没修好问题
- `force raw` 更差
- 下一个最值钱的 A/B 是 `row store`，否则就该直接扎进 `store_tile(...)`

## 10. 真源码与自动生成文件的约定

必须遵守:

- `_hip.hpp` / `_hip.cpp` 是自动生成文件
- 真正应修改并作为依据的是非 `_hip` 版本

本次已修改的关键真源码:

- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp)
- [block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp)
- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp)
- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp)
- [analyze_dbg_log.py](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/analyze_dbg_log.py)
- [todo.md](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/todo.md)

## 11. 建议的最短接手流程

下一位接手时，建议只做下面 4 步:

1. 重新编译
2. 跑当前复现脚本生成新的 `run_dbg_kname.log`
3. 执行:

```bash
python analyze_dbg_log.py run_dbg_kname.log
```

4. 先回答两个问题:

- `DK_EPI_CAST_T0` 在 `S=1024` good/bad 之间是否一致
- `DK_EPI_T0` 在 `S=1024` good/bad 之间是否分叉

5. 然后优先做 `row store` A/B；如果仍然失败，继续深入 `store_tile(...)`。

## 12. 附注: 当前仓库根目录里的辅助文件

为当前排障准备的文件:

- [todo.md](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/todo.md)
- [dk_chain.md](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/dk_chain.md)
- [analyze_dbg_log.py](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/analyze_dbg_log.py)
- [HANDOFF_split_dk_debug.md](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/HANDOFF_split_dk_debug.md)

## 13. 一句话交接结论

当前最可信的判断是:

- `S=1024` 的错误不是已经明显出现在 `dk_acc` return 指纹上
- 也不是 cast-to-output dtype 这一步先分叉
- 也不是当前 repro 下的 host merge 问题
- baseline 实际已在 normal-store 路径上，force-raw 更差
- 下一步应直接盯 `dk` 的 kernel epilogue `store_tile(...)` 路径，优先做 `row store` A/B
