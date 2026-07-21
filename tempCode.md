# WMMA `TransposeC` 相关源码摘录

以下内容按调用链整理，便于集中分析。源码已恢复到当前可编译基线，不包含刚才那个危险的 `mode=1` 实验分支。

## 1. `setup.py`：ROCm / CK 主链编译入口
文件：`setup.py`（约 `448-712`）

```python
    if not SKIP_CK_BUILD:
        ck_dir = "csrc/composable_kernel"

        #use codegen get code dispatch
        if not os.path.exists("./build"):
            os.makedirs("build")

        # Detect architecture family for CK code generation
        archs_for_detection = os.getenv("GPU_ARCHS", "native").split(";")
        is_rdna3 = any(a.startswith("gfx11") for a in archs_for_detection if a != "native")
        is_rdna4 = any(a.startswith("gfx12") for a in archs_for_detection if a != "native")
        if archs_for_detection == ["native"]:
            device_arch = torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
            is_rdna3 = device_arch.startswith("gfx11")
            is_rdna4 = device_arch.startswith("gfx12")

        # Set CK code generation targets based on architecture
        # gfx11 = RDNA3 (gfx1100/1101/1102), gfx12 = RDNA4 (gfx1200/1201)
        if is_rdna3:
            ck_targets = "gfx11"
        elif is_rdna4:
            ck_targets = "gfx12"
        else:
            ck_targets = "gfx9,gfx950"

        print(
            f"[CK_MINIMAL_DEBUG] enabled={CK_MINIMAL_DEBUG}, "
            f"dtype={CK_MINIMAL_DTYPE}, optdim={CK_MINIMAL_OPTDIM}, "
            f"noncausal_only={CK_MINIMAL_NONCAUSAL_ONLY}, batch_only={CK_MINIMAL_BATCH_ONLY}"
        )

        optdim = CK_MINIMAL_OPTDIM if CK_MINIMAL_DEBUG else os.getenv("OPT_DIM", "32,64,128,256")
        gen_dirs = ["fwd", "bwd"] if CK_MINIMAL_DEBUG else ["fwd", "fwd_appendkv", "fwd_splitkv", "bwd"]
        optdim_items = [item.strip() for item in CK_MINIMAL_OPTDIM.split(",") if item.strip()]
        can_apply_codegen_filter = CK_MINIMAL_DEBUG and CK_MINIMAL_BATCH_ONLY and len(optdim_items) == 1
        minimal_codegen_pattern = ""
        minimal_codegen_bwd_filter = ""
        if can_apply_codegen_filter:
            dim = optdim_items[0]
            # 保持100+规模：仅限定 d/dtype/batch，不再按 bias/mask/dropout 细切
            minimal_codegen_pattern = f"*d{dim}_{CK_MINIMAL_DTYPE}_batch*"
            minimal_codegen_bwd_filter = (
                f"*fmha_bwd_dot_do_o_d{dim}_{CK_MINIMAL_DTYPE}*batch*"
                f"@*fmha_bwd_convert_dq_d{dim}_{CK_MINIMAL_DTYPE}*batch*"
                f"@*fmha_bwd_d{dim}_{CK_MINIMAL_DTYPE}_batch*"
            )
            print(f"[CK_MINIMAL_DEBUG] codegen filter pattern={minimal_codegen_pattern}")
            print(f"[CK_MINIMAL_DEBUG] codegen bwd filter={minimal_codegen_bwd_filter}")
        for direction in gen_dirs:
            command = [
                sys.executable,
                f"{ck_dir}/example/ck_tile/01_fmha/generate.py",
                "-d",
                direction,
                "--targets",
                ck_targets,
                "--output_dir",
                "build",
                "--receipt",
                "2",
                "--optdim",
                optdim,
            ]

            if can_apply_codegen_filter:
                if direction == "bwd":
                    command.extend(["--filter", minimal_codegen_bwd_filter])
                else:
                    command.extend(["--filter", minimal_codegen_pattern])

            subprocess.run(command, check=True)

        # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
        # See https://github.com/pytorch/pytorch/pull/70650
        generator_flag = []
        torch_dir = torch.__path__[0]
        if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
            generator_flag = ["-DOLD_GENERATOR_PATH"]

        check_if_rocm_home_none("flash_attn")
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        validate_and_update_archs(archs)

        if archs != ['native']:
            cc_flag = [f"--offload-arch={arch}" for arch in archs]
        else:
            arch = torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
            cc_flag = [f"--offload-arch={arch}"]

        # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
        # torch._C._GLIBCXX_USE_CXX11_ABI
        # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
        if FORCE_CXX11_ABI:
            torch._C._GLIBCXX_USE_CXX11_ABI = True

        if CK_MINIMAL_DEBUG:
            base_cpp_sources = [
                "csrc/flash_attn_ck/flash_api.cpp",
                "csrc/flash_attn_ck/flash_common.cpp",
                "csrc/flash_attn_ck/mha_bwd.cpp",
                "csrc/flash_attn_ck/mha_fwd.cpp",
            ]
        else:
            base_cpp_sources = [
                "csrc/flash_attn_ck/flash_api.cpp",
                "csrc/flash_attn_ck/flash_common.cpp",
                "csrc/flash_attn_ck/mha_bwd.cpp",
                "csrc/flash_attn_ck/mha_fwd_kvcache.cpp",
                "csrc/flash_attn_ck/mha_fwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_bwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_fwd.cpp",
            ]
        generated_cpp_sources = glob.glob("build/fmha_*wd*.cpp")
        if CK_MINIMAL_DEBUG:
            before = len(generated_cpp_sources)
            generated_cpp_sources = filter_ck_minimal_generated_sources(generated_cpp_sources)
            after = len(generated_cpp_sources)
            print(
                f"[CK_MINIMAL_DEBUG] keeping generated kernels: {after}/{before} "
                f"(dtype={CK_MINIMAL_DTYPE}, optdim={CK_MINIMAL_OPTDIM}, "
                f"noncausal_only={CK_MINIMAL_NONCAUSAL_ONLY}, batch_only={CK_MINIMAL_BATCH_ONLY})"
            )
            if after == 0:
                raise RuntimeError("[CK_MINIMAL_DEBUG] 过滤后无可用 CK kernel，请检查最小化条件")

        sources = base_cpp_sources + generated_cpp_sources

        # Check if torch is using hipify v2. Until CK is updated with HIPIFY_V2 macro,
        # we must replace the incorrect APIs.
        maybe_hipify_v2_flag = []
        if detect_hipify_v2():
            maybe_hipify_v2_flag = ["-DHIPIFY_V2"]

        rename_cpp_to_cu(sources)

        if CK_MINIMAL_DEBUG:
            base_cu_sources = [
                "csrc/flash_attn_ck/flash_api.cu",
                "csrc/flash_attn_ck/flash_common.cu",
                "csrc/flash_attn_ck/mha_bwd.cu",
                "csrc/flash_attn_ck/mha_fwd.cu",
            ]
        else:
            base_cu_sources = [
                "csrc/flash_attn_ck/flash_api.cu",
                "csrc/flash_attn_ck/flash_common.cu",
                "csrc/flash_attn_ck/mha_bwd.cu",
                "csrc/flash_attn_ck/mha_fwd_kvcache.cu",
                "csrc/flash_attn_ck/mha_fwd.cu",
                "csrc/flash_attn_ck/mha_varlen_bwd.cu",
                "csrc/flash_attn_ck/mha_varlen_fwd.cu",
            ]
        generated_cu_sources = [source[:-4] + ".cu" for source in generated_cpp_sources]
        renamed_sources = base_cu_sources + generated_cu_sources

        if CK_MINIMAL_DEBUG:
            cc_flag += ["-DFLASH_ATTN_CK_MINIMAL_API=1"]

        cc_flag += ["-O3","-std=c++20",
                    "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                    "-fgpu-flush-denormals-to-zero",
                    "-DCK_ENABLE_BF16",
                    "-DCK_ENABLE_BF8",
                    "-DCK_ENABLE_FP16",
                    "-DCK_ENABLE_FP32",
                    "-DCK_ENABLE_FP64",
                    "-DCK_ENABLE_FP8",
                    "-DCK_ENABLE_INT8",
                    "-DUSE_PROF_API=1",
                    # "-DFLASHATTENTION_DISABLE_BACKWARD",
                    "-D__HIP_PLATFORM_HCC__=1"]

        # RDNA3/RDNA4 use WMMA, CDNA uses XDL - only add XDL flag for CDNA
        if not (is_rdna3 or is_rdna4):
            cc_flag += ["-DCK_USE_XDL"]
        else:
            # Suppress occupancy warning for RDNA3/4 - compiler bug reports wrong desired occupancy
            cc_flag += ["-Wno-pass-failed"]

        cc_flag += [f"-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get('CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT', 3)}"]

        ck_tile_dump_a_tmp = os.environ.get("CK_TILE_DEBUG_DUMP_A_TMP")
        if ck_tile_dump_a_tmp is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_DUMP_A_TMP={ck_tile_dump_a_tmp}"]

        ck_tile_no_raw_store = os.environ.get("CK_TILE_DEBUG_NO_RAW_STORE")
        if ck_tile_no_raw_store is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_NO_RAW_STORE={ck_tile_no_raw_store}"]

        ck_tile_force_store_path = os.environ.get("CK_TILE_DEBUG_FORCE_STORE_PATH")
        if ck_tile_force_store_path is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_FORCE_STORE_PATH={ck_tile_force_store_path}"]

        ck_tile_force_row_store = os.environ.get("CK_TILE_DEBUG_FORCE_ROW_STORE")
        if ck_tile_force_row_store is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_FORCE_ROW_STORE={ck_tile_force_row_store}"]

        ck_tile_owner_tag = os.environ.get("CK_TILE_DEBUG_OWNER_TAG")
        if ck_tile_owner_tag is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_OWNER_TAG={ck_tile_owner_tag}"]

        ck_tile_owner_scan = os.environ.get("CK_TILE_DEBUG_OWNER_SCAN")
        if ck_tile_owner_scan is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_OWNER_SCAN={ck_tile_owner_scan}"]

        ck_tile_owner_scan_limit = os.environ.get("CK_TILE_DEBUG_OWNER_SCAN_LIMIT")
        if ck_tile_owner_scan_limit is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_OWNER_SCAN_LIMIT={ck_tile_owner_scan_limit}"]

        ck_tile_owner_scan_abs_min = os.environ.get("CK_TILE_DEBUG_OWNER_SCAN_ABS_MIN")
        if ck_tile_owner_scan_abs_min is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_OWNER_SCAN_ABS_MIN={ck_tile_owner_scan_abs_min}"]

        ck_tile_o_glb_after = os.environ.get("CK_TILE_DEBUG_O_GLB_AFTER")
        if ck_tile_o_glb_after is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_AFTER={ck_tile_o_glb_after}"]

        ck_tile_o_glb_b = os.environ.get("CK_TILE_DEBUG_O_GLB_B")
        if ck_tile_o_glb_b is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_B={ck_tile_o_glb_b}"]

        ck_tile_o_glb_h = os.environ.get("CK_TILE_DEBUG_O_GLB_H")
        if ck_tile_o_glb_h is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_H={ck_tile_o_glb_h}"]

        ck_tile_o_glb_m = os.environ.get("CK_TILE_DEBUG_O_GLB_M")
        if ck_tile_o_glb_m is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_M={ck_tile_o_glb_m}"]

        ck_tile_o_glb_n = os.environ.get("CK_TILE_DEBUG_O_GLB_N")
        if ck_tile_o_glb_n is not None:
            cc_flag += [f"-DCK_TILE_DEBUG_O_GLB_N={ck_tile_o_glb_n}"]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = get_hip_version()
        if hip_version > Version('5.5.00000'):
            cc_flag += ["-mllvm", "--lsr-drop-solution=1"]
        if hip_version > Version('5.7.23302'):
            cc_flag += ["-fno-offload-uniform-block"]
        if hip_version > Version('6.1.40090'):
            cc_flag += ["-mllvm", "-enable-post-misched=0"]
        if hip_version > Version('6.2.41132'):
            cc_flag += ["-mllvm", "-amdgpu-early-inline-all=true",
                        "-mllvm", "-amdgpu-function-calls=false"]
        if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
            cc_flag += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

        extra_compile_args = {
            "cxx": ["-O3", "-std=c++20"] + generator_flag + maybe_hipify_v2_flag,
            "nvcc": cc_flag + generator_flag + maybe_hipify_v2_flag,
        }

        include_dirs = [
            Path(this_dir) / "csrc" / "composable_kernel" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "library" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "example" / "ck_tile" / "01_fmha",
        ]

        ext_modules.append(
            CUDAExtension(
                name="flash_attn_2_cuda",
                sources=renamed_sources,
                extra_compile_args=extra_compile_args,
                include_dirs=include_dirs,
            )
        )
```

## 2. FMHA pipeline：两个 GEMM 都走 `TransposeC=true`
文件：`csrc/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`

```cpp
            }
            else
            {
                constexpr bool SwizzleA =
                    Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}) == 32;
                return WarpGemmDispatcher<typename Problem::QDataType,
                                          typename Problem::KDataType,
                                          typename Problem::SaccDataType,
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                          true, // TransposeC
                                          SwizzleA>{};
            }
        }();

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::SaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 decltype(warp_gemm)>;
```

```cpp
            }
            else
            {
                return WarpGemmDispatcher<typename Problem::PDataType,
                                          typename Problem::VDataType,
                                          typename Problem::OaccDataType,
                                          Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                                          Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                                          Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                                          true>{};
            }
        }();

        using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::PDataType,
                                                 typename Problem::VDataType,
                                                 typename Problem::OaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                 WarpGemm>;
```

## 3. WMMA 分布骨架：`CWarpDstrEncodingTrait` / `CTransposedWarpDstrEncodingTrait`
文件：`csrc/composable_kernel/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp`

```cpp
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"

namespace ck_tile {

// TODO: currently only support 16 bit input, which means only support tr16_b128; will use ADataType
// to determine the layout in the future
template <typename Impl>
struct AWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kAMLane>,
              sequence<Impl::kABK0PerLane, Impl::kABKLane, Impl::kABK1PerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct BWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kBNLane>,
              sequence<Impl::kABK0PerLane, Impl::kABKLane, Impl::kABK1PerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct CWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNLane>>,
        tuple<typename Impl::kCPs2RHssMajor>,
        tuple<typename Impl::kCPs2RHssMinor>,
        typename Impl::kCYs2RHsMajor,
        typename Impl::kCYs2RHsMinor>;
};

template <typename Impl>
struct CTransposedWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
        tuple<typename Impl::kCTPs2RHssMajor>,
        tuple<typename Impl::kCTPs2RHssMinor>,
        typename Impl::kCTYs2RHsMajor,
        typename Impl::kCTYs2RHsMinor>;
};

template <typename WarpGemmAttributeWmmaImpl_, bool kTransC = false>
struct WarpGemmAttributeWmma
{
    using Impl = remove_cvref_t<WarpGemmAttributeWmmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kM;
    static constexpr index_t kN          = Impl::kN;
    static constexpr index_t kK          = Impl::kK;
    static constexpr index_t kCMLane     = Impl::kCMLane;
    static constexpr index_t kKPerThread = Impl::kABK0PerLane * Impl::kABK1PerLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    // 16 bit input, kAMLane = 16, kABK0PerLane = 4, kABKLane = 2, kABK1PerLane = 2
    // 8  bit input, kAMLane = 16, kABK0PerLane = 2, kABKLane = 2, kABK1PerLane = 4
    using AWarpDstrEncoding = typename AWarpDstrEncodingTrait<Impl>::type;
    using BWarpDstrEncoding = typename BWarpDstrEncodingTrait<Impl>::type;

    // kCM0PerLane = 1, kCMLane = 2, kCM1PerLane = 2, kCNLane = 16
    using CWarpDstrEncoding =
        std::conditional_t<kTransC,
                           typename CTransposedWarpDstrEncodingTrait<Impl>::type,
                           typename CWarpDstrEncodingTrait<Impl>::type>;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        if constexpr(kTransC)
        {
            Impl{}(c_vec, b_vec, a_vec, bool_constant<post_nop_>{});
        }
        else
        {
            Impl{}(c_vec, a_vec, b_vec, bool_constant<post_nop_>{});
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(kTransC)
        {
            return Impl{}(b_vec, a_vec);
        }
        else
        {
            return Impl{}(a_vec, b_vec);
        }
    }
```

## 4. WMMA 基础 traits：可疑的 `kCT*`
文件：`csrc/composable_kernel/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp`

```cpp
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
namespace ck_tile {
template <typename Arch, typename ADType, typename BDType, typename CDType>
struct WmmaTraitsBase;

// GFX11 specialization
template <typename ADType, typename BDType, typename CDType>
struct WmmaTraitsBase<gfx11_t, ADType, BDType, CDType>
{
    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    using AVecType = ext_vector_t<ADataType, 16>;
    using BVecType = ext_vector_t<BDataType, 16>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kRepeat      = 2;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 1;
    static constexpr index_t kABK1PerLane = 16;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 8;
    static constexpr index_t kCM1PerLane = 1;

    using kABPs2RHssMajor = sequence<0, 2, 1>;
    using kABPs2RHssMinor = sequence<0, 1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<1, 0>;
    using kCTYs2RHsMajor  = sequence<2, 2>;
    using kCTYs2RHsMinor  = sequence<0, 2>;
};

// GFX12 specialization
template <typename ADType, typename BDType, typename CDType>
struct WmmaTraitsBase<gfx12_t, ADType, BDType, CDType>
{
    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kRepeat      = 1;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 2;
    static constexpr index_t kABK1PerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    using kABPs2RHssMajor = sequence<2, 1>;
    using kABPs2RHssMinor = sequence<1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<1, 0>;
    using kCTYs2RHsMajor  = sequence<2, 2>;
    using kCTYs2RHsMinor  = sequence<0, 2>;
};
} // namespace ck_tile
```

## 5. Block GEMM：把 `WG::CWarpDstrEncoding` 嵌入 block 级分布
文件：`csrc/composable_kernel/include/ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp`

```cpp
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        // constrcut from A-block-tensor from A-Block-tensor-tmp
        // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have equivalent
```

```cpp
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
```

## 6. Epilogue：按 tile distribution 解算全局坐标并写回
文件：`csrc/composable_kernel/include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`

```cpp
                constexpr auto dstr_spans = remove_cvref_t<decltype(o_cast_tile)>::get_distributed_spans();

                index_t owner_log_count = 0;

                sweep_tile_span(dstr_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(dstr_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto distributed_indices = make_tuple(idx0, idx1);

                        const auto x_indices = [&]() {
                            if constexpr(is_partition_index)
                            {
                                return get_x_indices_from_distributed_indices(
                                    o_cast_tile.get_tile_distribution(),
                                    distributed_indices,
                                    ds_dram_windows);
                            }
                            else
                            {
                                return get_x_indices_from_distributed_indices(
                                    o_cast_tile.get_tile_distribution(), distributed_indices);
                            }
                        }();
```

```cpp
            #if CK_TILE_DEBUG_FORCE_STORE_PATH > 0
            if constexpr(true)
#elif CK_TILE_DEBUG_FORCE_STORE_PATH < 0
            if constexpr(false)
#else
            if constexpr(UseRawStore && !CK_TILE_DEBUG_NO_RAW_STORE && (kPadM || kPadN))
#endif
            {
                if constexpr(MemoryOperation == memory_operation_enum::set)
                {
                    if constexpr(is_partition_index)
                    {
                        store_tile_raw(o_dram_window_tmp,
                                       o_cast_tile,
                                       /*partition_index=*/ds_dram_windows);
                    }
                    else
                    {
                        store_tile_raw(o_dram_window_tmp, o_cast_tile);
                    }
                }
                else
                {
                    update_tile_raw(o_dram_window_tmp, o_cast_tile);
                }
                buffer_store_fence();
            }
            else
            {
                if constexpr(MemoryOperation == memory_operation_enum::set)
                {
                    if constexpr(is_partition_index)
                    {
                        store_tile(o_dram_window_tmp,
                                   o_cast_tile,
                                   /*partition_index=*/ds_dram_windows);
                    }
                    else
                    {
                        store_tile(o_dram_window_tmp, o_cast_tile);
                    }
```

## 7. Dropout：说明下游确实假定了更深的索引结构
文件：`csrc/composable_kernel/include/ck_tile/ops/fmha/block/block_dropout_hip.hpp`

```cpp
                if(is_store_randval)
                {
                    const auto randval_store = cast_tile<RandValOutputDataType>(randval);
                    store_tile(randval_dram_window, randval_store);
                }
                move_tile_window(randval_dram_window, {0, kNPerStep});
                // Drop values of P based on the generated probabilities
                constexpr auto randval_spans = decltype(randval)::get_distributed_spans();
                sweep_tile_span(randval_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(randval_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto p_idx0 =
                            tile_distributed_index<i_m0 * MIterPerWarp +
                                                   idx0.impl_.template at<0>()>{};
                        constexpr auto p_idx1 =
                            tile_distributed_index<i_n0,
                                                   idx1.impl_.template at<1>(),
                                                   idx1.impl_.template at<2>()>{};
                        constexpr auto p_idx = ck_tile::make_tuple(p_idx0, p_idx1);
                        constexpr auto r_idx = ck_tile::make_tuple(idx0, idx1);
                        p_compute(p_idx)     = randval[r_idx] <= p_undrop_in_uint8_t
                                                   ? p_compute[p_idx] * rp_undrop
                                                   : PComputeDataType(0);
                    });
```

## 8. 编译报错根：`sequence::at<I>()` 越界静态断言
文件：`csrc/composable_kernel/include/ck_tile/core/container/sequence_hip.hpp`

```cpp
namespace impl {
// static_assert(__has_builtin(__type_pack_element), "can't find __type_pack_element");
template <index_t I, typename... Ts>
using at_index_t = __type_pack_element<I, Ts...>;
} // namespace impl

// we could implement as below, similiar to std. But let's reduce the symbol name...
// template< class T, T... Ints >
// class integer_sequence;

template <index_t... Is>
struct sequence
{
    using type       = sequence;
    using value_type = index_t;

    CK_TILE_HOST_DEVICE static constexpr index_t size() { return sizeof...(Is); }
    CK_TILE_HOST_DEVICE static constexpr bool is_static() { return true; };

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto get()
    {
        static_assert(I < size(), "wrong! I too large");
        return number<impl::at_index_t<I, constant<Is>...>{}>{};
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto get(number<I>)
    {
        static_assert(I < size(), "wrong! I too large");
        return number<get<I>()>{};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t at(index_t I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const index_t mData[size() + 1] = {Is..., 0};
        return mData[I];
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto at()
    {
        static_assert(I < size(), "wrong! I too large");
        return number<impl::at_index_t<I, constant<Is>...>{}>{};
    }
```

## 9. 附录：`WmmaTraitsBase<gfx11_t / gfx12_t>` 原样特化
文件：`csrc/composable_kernel/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp`

```cpp
// GFX11 specialization
template <typename ADType, typename BDType, typename CDType>
struct WmmaTraitsBase<gfx11_t, ADType, BDType, CDType>
{
    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    using AVecType = ext_vector_t<ADataType, 16>;
    using BVecType = ext_vector_t<BDataType, 16>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kRepeat      = 2;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 1;
    static constexpr index_t kABK1PerLane = 16;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 8;
    static constexpr index_t kCM1PerLane = 1;

    using kABPs2RHssMajor = sequence<0, 2, 1>;
    using kABPs2RHssMinor = sequence<0, 1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<1, 0>;
    using kCTYs2RHsMajor  = sequence<2, 2>;
    using kCTYs2RHsMinor  = sequence<0, 2>;
};

// GFX12 specialization
template <typename ADType, typename BDType, typename CDType>
struct WmmaTraitsBase<gfx12_t, ADType, BDType, CDType>
{
    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kRepeat      = 1;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 2;
    static constexpr index_t kABK1PerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    using kABPs2RHssMajor = sequence<2, 1>;
    using kABPs2RHssMinor = sequence<1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<1, 0>;
    using kCTYs2RHsMajor  = sequence<2, 2>;
    using kCTYs2RHsMinor  = sequence<0, 2>;
};
```
