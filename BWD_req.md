# BWD Hidden Requirements

这份文档记录 CK Tile FMHA BWD 路径里，调 tile / 合并 fork 时必须同时满足的隐形关联。核心结论是：`fmha_bwd.py` 里的 tile 表不是孤立配置，和下游 kernel、pipeline、WMMA fragment 布局、LDS scratch 容量强耦合。

## 1. 五段 GEMM 不是独立的

BWD 主路径里至少有下面几段：

- `gemm_0`: `Q @ K^T -> P`
- `gemm_1`: `P^T @ dO^T -> dV`
- `gemm_2`: `dO @ V -> dS`
- `gemm_3`: `dS^T @ Q^T -> dK`
- `gemm_4`: `dS @ K^T -> dQ`

关键文件：

- [block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp)
- [block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp)

硬约束：

- `kQKHeaddim >= kK0`
- `kVHeaddim >= kK2`
- `kM0 == kK1`
- `kM0 == kK3`

含义：

- 改 `bm0/bn0/bk*` 时，实际是在同时重配多段 GEMM 的块形状。
- 某一段看起来合法，不代表整个 BWD 链合法。

## 2. dV 和 dK 依赖不同的布局重排链

虽然 `dQ/dK/dV` 都来自同一个 BWD kernel，但后半段写回路径不同：

- `dV` 依赖 `PTFromGemm0CToGemm1A(...)`
- `dK` 依赖 `SGradTFromGemm2CToGemm3A(...)`
- `dQ` 依赖单独的 `remap_dq_acc_for_store(...)`

关键文件：

- [block_fmha_bwd_pipeline_default_policy.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp)
- [block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp)

经验结论：

- `dq/dv` 正常但 `dk` 炸掉，是完全可能且常见的。
- 如果只有 `dk` 出 `nan`，优先怀疑 `gemm_2 C -> gemm_3 A` 的布局兼容性，而不是整个 backward。

## 3. PT / SGradT 的分布必须匹配

当前目录已经内建了两组重要的布局假设：

- `PT input distribution must match gemm_0 C block distribution`
- `PT output distribution must match gemm_1 A block distribution`
- `SGradT input distribution must match gemm_2 C block distribution`
- `SGradT output distribution must match gemm_3 A block distribution`
- `gemm_0 and gemm_2 C block distributions differ` 不能成立
- `gemm_1 and gemm_3 A block distributions differ` 不能成立

这些约束位于：

- [block_fmha_bwd_pipeline_default_policy.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp#L1933)
- [block_fmha_bwd_pipeline_default_policy.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp#L2592)

含义：

- 不能只看 kernel name 里 tile 参数“像是能接起来”。
- 只要 block distribution 或 thread distribution 变了，PT / SGradT 的重解释就可能失效。

## 4. WMMA remap 只在特定 fragment trait 下启用

当前目录的 remap 逻辑不是无条件开启，而是只在特定 WMMA 属性下工作。

触发条件：

- `kRepeat == 2`
- `kABKLane == 1`
- `kABK1PerLane == 16`
- `kCM0PerLane == 8`
- `kCM1PerLane == 1`

关键文件：

- [warp_gemm_attribute_wmma_impl_base_traits.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp)
- [block_fmha_bwd_pipeline_default_policy.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp#L1949)
- [block_fmha_bwd_pipeline_default_policy.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp#L2614)

含义：

- 改 tile 后，可能让某段 GEMM 切到另一种 warp-gemm attribute。
- 一旦不再满足这些条件，原先依赖的 LDS remap 可能不再启用。
- 或者仍然启用，但新 tile 下 remap 假设不再成立。

## 5. LDS scratch 容量必须能覆盖 remap

当前目录已经为了 BWD remap 扩大了共享内存需求，尤其是 `dQ` 的写回 remap。

关键文件：

- [block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp#L79)
- [block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp#L79)
- [block_fmha_bwd_pipeline_default_policy.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp#L1969)
- [block_fmha_bwd_pipeline_default_policy.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp#L2626)

需要满足：

- `kBlockScratchElems <= kScratchCapacity`
- `kWarpCount * kWarpScratchElems <= kM0 * kN0`
- `GetSmemSize()` 必须至少覆盖 base smem 和 remap 额外需求

含义：

- 改 tile 时，即使功能逻辑没错，也可能先死在 LDS 不够。
- “能编译” 也不代表 scratch 没问题，仍然可能出现运行时错误或结果损坏。

## 6. block-gemm 的临时 tensor 不能假设天然兼容

当前目录里：

## 7. 长序列污染先别急着归因 tile，先排除“未写满”和“真越界”

当前仓库已经在 wrapper 侧加了两个专门给 CK BWD 用的调试开关，位置：

- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp)
- [mha_bwd.hip](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.hip)

调试开关：

- `FLASH_ATTN_CK_BWD_POISON_OUTPUTS=1`
  含义：`dq/dk/dv` 缺省输出不再用 `empty_like`，而是先填 `NaN`。如果 kernel 没完整写满，结果里会直接残留 `NaN`。
- `FLASH_ATTN_CK_BWD_GUARD_ROWS=<N>`
  含义：给 `dk_expanded/dv_expanded` 的 sequence 维前后各加 `N` 行 guard，并用 `NaN` 初始化。CK 调完以后会同步检查 guard 是否被破坏。

判读规则：

- `dk_nan=1` 但 `dk_guard_corrupt=0`
  更像“kernel 没把合法输出区域完整写满”。
- `dk_guard_corrupt=1`
  更像 sequence 维发生了真实越界写。
- 先跑 `2048` 坏，再回测 `1024` 也坏
  如果打开 poison 后 `1024` 直接出现 `NaN`，这通常不是同步问题，而是上一次长序列把某些输出块留成了未定义值，随后被 allocator 复用放大了。

- [block_gemm_areg_bsmem_creg_v2.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp)

已经把：

- `auto a_block_tensor = a_block_tensor_tmp;`

改成：

- 先构造目标分布的 `a_block_tensor`
- 再逐元素复制

含义：

- CK 里“看上去都是同形状 tensor”不代表 thread distribution 等价。
- 更换 tile、warp repeat、warp tile 后，这类假设最容易失效。

## 7. 当前目录里 BWD 正确性修复主线

这棵 `_20260319c` 当前目录里，和 BWD 正确性直接相关的修复主要有：

- `PTFromGemm0CToGemm1A(...)` 加入 LDS remap
- `SGradTFromGemm2CToGemm3A(...)` 加入 LDS remap
- `dQ` 写回前先做 `remap_dq_acc_for_store(...)`
- `GetSmemSize()` 扩容以容纳 remap scratch
- 某些 WMMA dispatch / trait 关联修正为使用正确的 `Gemm3WarpTile`

这些修复都不应在合并 BWD tile fork 时被回退。

## 8. 合并 BWD tile fork 的推荐流程

不要直接整块搬另一棵树的 `gfx11/gfx12 get_dq_dk_dv_tiles()`。更稳的流程是：

1. 保持当前目录的 kernel / pipeline / gemm / layout 修复不动。
2. 每次只改一小组 tile，优先从单一 `hdim`、单一 `dtype`、单一 `causal` case 开始。
3. 重新编译后记录实际命中的 kernel name。
4. 分别检查：
   - `fwd`
   - `dq`
   - `dk`
   - `dv`
5. 只要出现“仅 `dk` 坏”或“仅 `dq` 坏”，优先检查布局重排链，而不是继续盲调 tile。

## 9. 推荐诊断开关

这些环境变量对定位很有用：

- `CK_TILE_FMHA_BWD_LAYOUT_DIAG=1`
  用于检查 `PT` 路径
- `CK_TILE_FMHA_BWD_LAYOUT_DIAG=2`
  用于检查 `SGradT` 路径
- `CK_TILE_FMHA_BWD_WMMA_PT_LDS_REMAP=0/1`
  对 `dV` 路径做 A/B
- `CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP=0/1`
  对 `dK` 路径做 A/B
- `FLASH_ATTN_CK_LOG_LEVEL=1`
  打印实际命中的 kernel 名
- `HIP_LAUNCH_BLOCKING=1`
  让错误更靠近真实出错点

## 10. 最重要的经验结论

对 CK Tile FMHA BWD 来说，下面这句话基本成立：

- `tile` 不是单纯的性能参数，而是 kernel 布局契约的一部分。

因此：

- 换 `bn0/bm0/rm/rn` 可能会直接改变 fragment layout 假设
- 只要触碰 `gemm_2 -> gemm_3` 或 `gemm_0 -> gemm_1` 的中间表示，就必须把 remap / distribution / scratch 一起看
- “之前 FWD 正常” 不代表 BWD 的 tile fork 可以安全搬进来

补充：

- 当前目录已经把 `PT` / `SGradT` 的 bypass 路径收紧为编译期约束：
  只有当 `CWarpDstr` 与 `AWarpDstr` 的 thread-buffer 语义完全兼容时，才允许不经过 LDS remap 直接旁路。
- 对当前 `gfx11/gfx12` WMMA BWD 主路径，这等价于 `CK_TILE_FMHA_BWD_WMMA_PT_LDS_REMAP=1` 和
  `CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP=1` 应视为必须条件，而不是可调开关。

## 11. 当前已确认的 BWD limit

对 `gfx11/gfx12`, `fp16/bf16`, `D=64` 这条当前 WMMA BWD 主路径，下面这类 tile 目前应视为不支持：

- `b16x64x64x16x64x16x32x64x64`
- 更一般地说：`kM0=16` 且 `kK1=kK3=16` 的这一档 profile

现象：

- `fwd` 正常
- `dv` 仍正常
- `dk` 与 `dq` 会同时出现严重错误，甚至 `nan`

这说明问题不在 `PT -> gemm1 -> dV`，而在后半链：

- `dS -> gemm3 -> dK`
- `dS -> gemm4 -> dQ`

当前判断：

- 这更像是后半链的 fragment/layout 契约未覆盖该 profile，而不是简单的 `GetSmemSize()` 不足
- 因此不要把 `bm0=16 / bk1=16 / bk3=16` 这一档当成现成可调优化候选

推进优化时的限制：

- 优先保持 `kM0=32, kK1=32, kK3=32`
- 优先只动不会改变后半链 fragment 迭代形状的参数
- 如果必须继续尝试 `kM0=16`，需要先补新的 compile-time layout/fragment guard，再做数值验证

结合你当前代码，我会这样修，优先级从高到低：

  1. 先修成“LDS 每轮全覆盖”，不要依赖“这次 store 应该写满”

  - 最可疑的是 qt_lds 和 SGradT remap scratch
  - 当前 Stage 6 前后顺序在这里：
      - 先 load_tile(qt_lds_read_window)，见 csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/
  block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp:751
      - 然后马上 overwrite 下一轮 shuffled_q LDS，见 csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/
  block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp:755
  - 如果某些 lane/store 因执行掩码、padding、分布映射没有写满，那下一轮就会读到旧值
  - 所以最实用的修法是：
      - 每轮在写 shuffled_q_lds_write_window 前，先把整块 qt_lds 清零
      - 每次 SGradTFromGemm2CToGemm3A(...) 使用的 scratch 也先清零对应 block
  - 这不一定最优，但最符合 ISA 对 LDS “静默越界/静默未定义” 风险的防御思路

  2. 补强 LDS producer/consumer 之间的等待，不只靠 barrier

  - 按 ISA，s_barrier 不是完整的数据可见性保证，关键还是前面的 LDS 操作有没有 drain 完
  - 你这里 Stage 6 很像典型的：
      - DS/LDS write
      - barrier
      - 另一批 load/use
  - 我会检查并倾向于显式保证：
      - lgkmcnt(0) drain
      - 再 s_barrier
  - 也就是说，要确认关键 handoff 不是只有 block_sync_lds()，还要确保编译后真的有足够的 LDS wait
  - 如果没有，就需要在关键点补显式 wait

  3. 不要信任 Stage 6 remap 路径“天然全覆盖”

  - SGradTFromGemm2CToGemm3A(...) 很重，见 csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/
  block_fmha_bwd_pipeline_default_policy.hpp:2168
  - 它内部有：
      - LDS transpose/remap
      - direct bypass
      - WMMA fragment 布局假设
  - 我会偏向保守：
      - 强制走 LDS remap
      - 不要依赖 bypass
      - scratch 区域每轮先初始化
  - 原因是 RDNA3 的 WMMA fragment 约束太硬，任何“看起来没问题的部分写入”都可能在后面放大

  4. 把问题先收敛到 “LDS stale” 而不是 “数值极端”

  - 你现在看到的是长度增长后不稳定，不是固定 cliff
  - 这更像跨迭代状态污染，不像输入幅值本身
  - clip 可以做，但我不认为它是主修法
  - 主修法还是：
      - 清 LDS
      - 强制 full overwrite
      - 强化 wait+barrier

  我认为最值得先试的两个具体补丁是：

  - 补丁 A：
    在 hot loop 每轮 Stage 4/5 之后、重新写 shuffled_q_lds_write_window 之前，显式清零整个 qt_lds 对应 block，再按原逻辑写入下一
  轮 Q^T
  - 补丁 B：
    在 SGradTFromGemm2CToGemm3A(...) 使用的 ds_lds_ptr scratch 上，每次 remap 前显式清零 kN0 * kK3 那块 scratch，再做 transpose/
  remap

  如果这两个补丁能显著改善 >768 的漂移，那基本就能说明：

  - 问题核心不是 tile size
  - 而是 LDS block 在 반복 Stage 6 中没有被完全、确定地重新定义

  所以一句话：

  - 参考 ISA，我觉得该修的是 LDS 生命周期与可见性
  - 不是先修 tile，也不是先修数值 clip
  - 先把 Stage 6 相关 LDS scratch 变成“每轮清零 + 每轮写满 + 显式等待”再看
