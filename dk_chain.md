# dK Execution Chain

这份文档只描述当前仓库里 `dK` 的实际执行链路，不讨论推测。

## 1. Python / Extension Entry

PyTorch 扩展把 backward 入口导出成 `flash_attn_2_cuda.bwd`：

- [flash_api.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/flash_api.cpp#L49)
- [flash_api.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/flash_api.cpp#L117)

关键点：

```cpp
std::vector<at::Tensor>
mha_bwd(...);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bwd", &mha_bwd, "Backward pass");
}
```

## 2. Host Backward Entry

真正的 host 侧 backward 实现在：

- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L350)

这里完成：

- 检查 `q/k/v/out/dout`
- 解析 `batch_size / seqlen_q / seqlen_k / num_heads / num_heads_k / head_size`
- 构造 `dk` / `dv` 输出 tensor
- 构造 `dq_accum`
- 构造 `dk_expanded` / `dv_expanded`
- 组装 CK traits 和 CK args
- 调用 `fmha_bwd(traits, args, stream_config)`

关键段落：

- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L404)
- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L443)
- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L501)
- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L551)
- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L589)

核心代码：

```cpp
auto traits =
    get_ck_fmha_bwd_traits(mask, q_dtype_str, head_size, is_dropout,
                           alibi_slopes_.has_value(), deterministic);

auto args =
    get_ck_fmha_bwd_args(mask, batch_size, seqlen_q, seqlen_k,
                         num_heads, num_heads_k, head_size,
                         q, k, v, alibi_slopes_, out, softmax_lse_kernel,
                         dout, dq_accum, softmax_d, dq, dk_expanded,
                         dv_expanded, softmax_scale, p_dropout,
                         drop_seed_offset);

float t = fmha_bwd(traits, args, stream_config);
```

## 3. CK Traits / Args Packing

traits:

- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L160)

args:

- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L179)

`get_ck_fmha_bwd_args(...)` 把这些东西传进 CK kernel：

- `q_ptr / k_ptr / v_ptr`
- `lse_ptr / do_ptr / d_ptr`
- `dq_acc_ptr / dk_ptr / dv_ptr`
- `seqlen_q / seqlen_k / hdim_q / hdim_v`
- `stride_dk / nhead_stride_dk / batch_stride_dk`
- `raw_scale / scale`

`dK` 的 host 输出地址最终就是 `args.dk_ptr`。

## 4. CK Kernel Shell

CK backward kernel 壳子在：

- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp)

这个 kernel 干的事是：

- 建好 `dk_dram_window`
- 调用 `FmhaPipeline{}(...)` 取回 `dk_acc_tile`
- 调用 `KGradEpiloguePipeline{}(...)` 把 `dk_acc_tile` 写回 global memory

最关键的一段：

- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L1366)
- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L1382)
- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L1384)
- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L1386)

```cpp
const auto dk_origin = dk_dram_window.get_window_origin();
printf("[DK_EPI_CTX] ...");
dump_logical_pre("DK_LOGICAL_PRE", dk_acc_tile);

KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);

dump_logical_post("DK_LOGICAL_POST", dk_ptr, kargs.stride_dk,
                  kargs.hdim_q, dk_dram_window);
```

## 5. Current Device Probe Target

当前 device probe 仍然盯着旧的单点：

- `blockIdx = (4, 1, 1)`
- local `(m, n) = (57, 10)`

定义位置：

- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L20)
- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L43)
- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L14)
- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L41)

```cpp
#define CK_TILE_DEBUG_BWD_TARGET_BLOCK_X 4
#define CK_TILE_DEBUG_BWD_TARGET_BLOCK_Y 1
#define CK_TILE_DEBUG_BWD_TARGET_BLOCK_Z 1
#define CK_TILE_DEBUG_BWD_TARGET_LOCAL_M 57
#define CK_TILE_DEBUG_BWD_TARGET_LOCAL_N 10
```

## 6. dK Inside Pipeline

当前这条 repro 的 `dK` 主体来自 IGLP pipeline：

- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp)

### 6.1 `dk_acc` 初始化

- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L533)
- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L700)

```cpp
auto dk_acc = [&]() {
    if constexpr(kDoDK)
        return decltype(gemm_3.MakeCBlockTile()){};
    else
        return ck_tile::null_type{};
}();

clear_tile(dk_acc);
```

### 6.2 `dK` 的核心计算

`dK = dS^T @ Q^T` 发生在 Stage 6，也就是 `gemm_3(...)`：

- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1271)
- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1282)
- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1299)

```cpp
// STAGE 6, SGrad^T@Q^T Gemm3
auto dst_reg_tensor = ...;
auto qt_reg_tensor = load_tile(qt_lds_read_window);
Policy::template SGradTFromGemm2CToGemm3A(...);
gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
```

也就是：

- `dp_acc` / `ds_gemm` 先经过 `SGradTFromGemm2CToGemm3A(...)`
- 变成 `dst_reg_tensor`
- 然后和 `qt_reg_tensor` 一起喂给 `gemm_3`
- 累加到 `dk_acc`

### 6.2.1 `CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP` 的实际含义

`SGradTFromGemm2CToGemm3A(...)` 的真源在：

- [block_fmha_bwd_pipeline_default_policy.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp#L2638)

这一步不是单纯“优化一下搬运方式”，而是把 `gemm_2.C` 的布局转换成 `gemm_3.A` 需要的布局。

核心逻辑是：

```cpp
constexpr bool kUseLdsRemap =
    (CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP != 0) && ...;

if constexpr(kUseLdsRemap) {
    TransposeBlockTensorThroughLds(dst_out, ds_in, scratch_ptr);
} else {
    static_assert(kDirectBypassCompatible, ...);
    // direct bypass / thread-buffer path
}
```

因此：

- `CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP=1`
  - 走 `TransposeBlockTensorThroughLds(...)`
  - 通过 LDS 做一次 block transpose / remap
- `CK_TILE_FMHA_BWD_WMMA_SGRADT_LDS_REMAP=0`
  - 不是“关闭优化但保持同一算法”
  - 而是改走 `direct bypass` 分支
  - 这个分支还要求 `kDirectBypassCompatible`

所以，之前“把这个宏关掉后完全跑不准”，从源码上看并不奇怪；它不能作为一个干净的性能 A/B 开关来解读。

### 6.2.2 当前 remap 位点上的新探针

现在 probe 已经直接插在 `SGradTFromGemm2CToGemm3A(...)` 内部，用来观察 remap 前后：

- `SGRADT_CTX`
  - 打印候选块是否走 `use_lds_remap`
- `SGRADT_PRE`
  - 取样 `ds_in`，看 remap 前的 `(q, k)` 值
- `SGRADT_POST`
  - 取样 `dst_out`，看 remap 后的 `(k, q)` 值

当前候选块来自 host `DK_HOST_GLOBAL` 抓到过的坏点：

- candidate A: `(block.x, block.y, block.z) = (7, 0, 1)`, `local_k = 43`
- candidate B: `(block.x, block.y, block.z) = (14, 1, 0)`, `local_k = 21`

这些 probe 的目的不是证明“这个 tile 每次都必炸”，而是把观察点从旧的干净块，迁到 host 已经报过异常的 tile 区域，提高命中率。

### 6.3 `dK` scale

loop 结束后，`dk_acc` 还会乘 scale：

- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1325)
- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1336)

无 dropout 时走：

```cpp
tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);
```

有 dropout 时走：

```cpp
tile_elementwise_inout(
    [&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; }, dk_acc);
```

### 6.4 Pipeline Return

pipeline 返回 `dk_acc`：

- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1348)
- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1359)
- [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1363)

```cpp
printf("[DK_RETURN_CTX] ...");
dump_bwd_tile_probe("DK_RETURN_TARGET", dk_acc,
                    CK_TILE_DEBUG_BWD_TARGET_LOCAL_M,
                    CK_TILE_DEBUG_BWD_TARGET_LOCAL_N);

return make_tuple(dk_acc, dv_acc);
```

## 7. Epilogue Store to Global dK

pipeline 返回的 `dk_acc_tile` 在 kernel 壳子里被写入 global `dk_ptr`：

- [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L1384)

```cpp
KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);
```

这里前后你现在已经有两个 probe：

- `DK_LOGICAL_PRE`
- `DK_LOGICAL_POST`

它们对应的是：

- epilogue 前的逻辑 tile 视图
- epilogue 后从 `dk_ptr` 按同一逻辑位置读回

## 8. Host-Side dK Validation Probe

kernel 返回以后，host 现在会强制同步并打印整张 `dk_expanded` 的摘要：

- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L72)
- [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L592)

```cpp
TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, ...);
dump_dk_host_stats(dk_expanded, batch_size, seqlen_k, num_heads, head_size);
```

`dump_dk_host_stats(...)` 会打印：

- `DK_HOST_GLOBAL`
  - 整张 `dk` 的 `absmax`
  - `sum`
  - `argmax=(b,s,h,d)`
- `DK_HOST_BH`
  - 每个 `(batch, head)` slice 的 `absmax`
  - `sum`

## 9. Important Current Fact

当前代码里，device probe 盯的是旧单点：

- block `(4,1,1)`
- local `(57,10)`

但 host probe 已经证明 `S=1024` 的坏点漂移到了别的 global 坐标，所以“当前单点是干净点”这个结论是和代码一致的，不是日志误读。

## 10. Practical Read Order

如果你只想快速审 `dK`，建议按这个顺序看：

1. [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L551)
2. [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L1366)
3. [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1271)
4. [block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1336)
5. [fmha_bwd_kernel.hpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L1384)
6. [mha_bwd.cpp](/home/husrcf/Code/ProtBind/fa4/flash-attention-fa4-v4.0.0.beta4_20260319c/csrc/flash_attn_ck/mha_bwd.cpp#L592)
