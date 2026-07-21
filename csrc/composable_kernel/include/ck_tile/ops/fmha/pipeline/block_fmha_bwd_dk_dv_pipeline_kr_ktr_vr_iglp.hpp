// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_PRECAST_SGRAD
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_PRECAST_SGRAD 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_DS
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_DS 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_P
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_P 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_INLINE_DS_REMAP
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_INLINE_DS_REMAP 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_DS_LDS_MATERIALIZE
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_DS_LDS_MATERIALIZE 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_USE_GENERIC_LDS_LAYOUT
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_USE_GENERIC_LDS_LAYOUT 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_LATE_QDO_STORE_LAMBDA
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_LATE_QDO_STORE_LAMBDA 0
#endif

#ifndef CK_TILE_ENABLE_D256_HYBRID_FAST
#define CK_TILE_ENABLE_D256_HYBRID_FAST 1
#endif

#ifndef CK_TILE_DEBUG_BWD_DKDV_D256_NONMASK_NO_EDGE_CHECK
#define CK_TILE_DEBUG_BWD_DKDV_D256_NONMASK_NO_EDGE_CHECK 0
#endif

#ifndef CK_TILE_DEBUG_BWD_DKDV_NONMASK_NO_EDGE_CHECK
#define CK_TILE_DEBUG_BWD_DKDV_NONMASK_NO_EDGE_CHECK 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DV_D256_NONMASK_NO_EDGE_CHECK
#define CK_TILE_DEBUG_BWD_SPLIT_DV_D256_NONMASK_NO_EDGE_CHECK 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_KT_ONLY
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_KT_ONLY 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_ONE_SLICE
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_ONE_SLICE 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE 2
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
#define CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DV_EARLY_Q_LSE_STORE
#define CK_TILE_DEBUG_BWD_SPLIT_DV_EARLY_Q_LSE_STORE 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM
#define CK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM 0
#endif

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM_D256
#define CK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM_D256 1
#endif

namespace ck_tile {

template <typename Problem,
          typename Policy = BlockFmhaBwdPipelineDefaultPolicy,
          bool kDoDK_ = true,
          bool kDoDV_ = true>
struct BlockFmhaBwdDKDVPipelineKRKTRVRIGLP
{
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType          = remove_cvref_t<typename Problem::GemmDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using DDataType             = remove_cvref_t<typename Problem::DDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType         = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType         = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType         = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType         = remove_cvref_t<typename Problem::VGradDataType>;
    using BiasGradDataType      = remove_cvref_t<typename Problem::BiasGradDataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using FmhaDropout           = remove_cvref_t<typename Problem::FmhaDropout>;
    using HotLoopScheduler      = typename Policy::template HotLoopScheduler<Problem>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;

    static constexpr index_t kM0        = BlockFmhaShape::kM0;
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t kK0        = BlockFmhaShape::kK0;
    static constexpr index_t kK1        = BlockFmhaShape::kK1;
    static constexpr index_t kK2        = BlockFmhaShape::kK2;
    static constexpr index_t kK3        = BlockFmhaShape::kK3;
    static constexpr index_t kK4        = BlockFmhaShape::kK4;
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kVHeaddim  = BlockFmhaShape::kVHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr index_t kPadHeadDimQ  = Problem::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum         = Problem::BiasEnum;
    static constexpr bool kHasBiasGrad     = Problem::kHasBiasGrad;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;
    static constexpr bool kUseTrLoad       = Problem::kUseTrLoad;
    static constexpr bool kDoDQ            = false;
    static constexpr bool kDoDK            = kDoDK_;
    static constexpr bool kDoDV            = kDoDV_;
    static constexpr bool kDoDKDV          = kDoDK || kDoDV;
    static constexpr bool kIsD256HybridShape =
        CK_TILE_ENABLE_D256_HYBRID_FAST && kQKHeaddim == 256 && kVHeaddim == 256;
    static constexpr bool kDelayDVOnlyQDoPrefetch =
        (CK_TILE_DEBUG_BWD_SPLIT_DV_EARLY_Q_LSE_STORE != 0) && kDoDV && !kDoDK;
    static constexpr bool kUseSplitDvGemm1BSmem =
        kDoDV && !kDoDK &&
        (((CK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM != 0) &&
          Problem::BlockFmhaShape::kVHeaddim == 64) ||
         ((CK_TILE_DEBUG_BWD_SPLIT_DV_GEMM1_BSMEM_D256 != 0) && kIsD256HybridShape &&
          Problem::BlockFmhaShape::kVHeaddim == 256));
    static constexpr bool kIsFusedFullBwd  = false;
    static constexpr bool kUseD256HybridFast =
        kIsD256HybridShape && kDoDK && !kDoDV;
    static constexpr bool kUseDKOnlyFreshP =
        (CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_P != 0) && kUseD256HybridFast;
    static constexpr bool kUseD256InlineDsRemap =
        (CK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_INLINE_DS_REMAP != 0) && kUseD256HybridFast;
    static constexpr bool kUseD256DsLdsMaterialize =
        (CK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_DS_LDS_MATERIALIZE != 0) && kUseD256HybridFast;
    static constexpr bool kUseD256LateQDOStoreLambda =
        (CK_TILE_DEBUG_BWD_SPLIT_DKDV_LATE_QDO_STORE_LAMBDA != 0) && kUseD256HybridFast &&
        !kHasBiasGrad;
    static constexpr bool kUseGenericLDSLayout =
        CK_TILE_DEBUG_BWD_SPLIT_DKDV_USE_GENERIC_LDS_LAYOUT || kUseD256HybridFast;
    // Experimental batch nonmask DK-only fast path. This only skips the score-tile
    // K-tail edge mask; all global store/load validity checks remain intact.
    // D64/D128 trials regressed, so keep this tied to the D256 hybrid layout.
    static constexpr bool kAllowDKOnlyNonMaskNoEdgeCheck =
        ((CK_TILE_DEBUG_BWD_DKDV_NONMASK_NO_EDGE_CHECK != 0) ||
         Problem::kSkipDKDVBwdNonMaskEdgeCheck) &&
        kUseD256HybridFast;
    static constexpr bool kAllowD256NonMaskNoEdgeCheck =
        (CK_TILE_DEBUG_BWD_DKDV_D256_NONMASK_NO_EDGE_CHECK != 0) && kUseD256HybridFast;
    static constexpr bool kSkipDKOnlyNonMaskEdgeCheck =
        (kAllowDKOnlyNonMaskNoEdgeCheck || kAllowD256NonMaskNoEdgeCheck) &&
        !FmhaMask::IsMasking && !kIsGroupMode && !FmhaDropout::IsDropout;
    // Split-DV has the same score-tile edge predicate as DK-only. Keep this
    // separately gated so D256 DV-only can be tested without touching fused DKDV.
    static constexpr bool kSkipDVOnlyNonMaskEdgeCheck =
        ((CK_TILE_DEBUG_BWD_SPLIT_DV_D256_NONMASK_NO_EDGE_CHECK != 0) ||
         Problem::kSkipDVBwdNonMaskEdgeCheck) &&
        kDoDV && !kDoDK &&
        kIsD256HybridShape && !FmhaMask::IsMasking && !kIsGroupMode && !FmhaDropout::IsDropout;
    static_assert(kDoDKDV, "At least one of dk/dv must remain enabled.");
    static_assert(!kUseTrLoad, "This pipeline does not use trload!");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentOGrad<Problem>();
    static constexpr index_t kAlignmentQGrad = 1;
    static constexpr index_t kAlignmentKGrad =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentKGrad<Problem>();
    static constexpr index_t kAlignmentVGrad =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentVGrad<Problem>();
    static constexpr index_t kAlignmentBias = 1;

    static constexpr const char* name =
        kDoDK && kDoDV ? "kr_ktr_vr_iglp_dkdv" : (kDoDK ? "kr_ktr_vr_iglp_dk" : "kr_ktr_vr_iglp_dv");

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr index_t smem_size_q    = Policy::template GetSmemSizeQ<Problem>();
        constexpr index_t smem_size_qt   = kDoDK ? Policy::template GetSmemSizeQT<Problem>() : 0;
        constexpr index_t smem_size_lse  = Policy::template GetSmemSizeLSE<Problem>();
        constexpr index_t smem_size_k    = Policy::template GetSmemSizeK<Problem>();
        constexpr index_t smem_size_kt   = Policy::template GetSmemSizeKT<Problem>();
        constexpr index_t smem_size_v    = kDoDK ? Policy::template GetSmemSizeV<Problem>() : 0;
        constexpr index_t smem_size_do   = kDoDK ? Policy::template GetSmemSizeOGrad<Problem>() : 0;
        constexpr index_t smem_size_dot  =
            (kDoDK || kDoDV) ? Policy::template GetSmemSizeOGradT<Problem>() : 0;
        constexpr index_t smem_size_d    = kDoDK ? Policy::template GetSmemSizeD<Problem>() : 0;
        constexpr index_t smem_size_ds   = Policy::template GetSmemSizeSGrad<Problem>();
        constexpr index_t smem_size_bias = Policy::template GetSmemSizeBias<Problem>();
        constexpr index_t smem_size_stage0_0 = smem_size_k + smem_size_kt;
        constexpr index_t smem_size_stage0_1 = smem_size_v;
        constexpr bool kSeparateDKOnlyScratch = kDoDK && !kDoDV;
        constexpr index_t smem_size_qt_guard = kDoDK ? smem_size_qt : 0;
        if constexpr(kUseGenericLDSLayout)
        {
            constexpr index_t smem_size_stage1 = smem_size_qt + smem_size_q +
                                                 smem_size_dot + smem_size_do + smem_size_lse +
                                                 smem_size_d + max(smem_size_bias, smem_size_ds);
            return max(smem_size_stage0_0, smem_size_stage0_1, smem_size_stage1) +
                   smem_size_qt;
        }
        else
        {
            constexpr index_t smem_size_stage1_shared =
                smem_size_q + smem_size_dot + smem_size_do + smem_size_lse + smem_size_d +
                (kSeparateDKOnlyScratch ? smem_size_bias : max(smem_size_bias, smem_size_ds));
            constexpr index_t smem_size_base =
                max(smem_size_stage0_0, smem_size_stage0_1, smem_size_stage1_shared);
            constexpr index_t smem_size_qt_separate =
                kDoDK ? smem_size_base + smem_size_qt + smem_size_qt_guard : 0;
            constexpr index_t smem_size_ds_separate =
                kSeparateDKOnlyScratch
                    ? smem_size_base + smem_size_qt + smem_size_qt_guard + smem_size_ds
                    : 0;
            return max(smem_size_stage0_0,
                       smem_size_stage0_1,
                       smem_size_stage1_shared,
                       smem_size_qt_separate,
                       smem_size_ds_separate);
        }
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp,
              typename BiasGradDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    operator()(void* smem_ptr,
               const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
               const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
               const DDramBlockWindowTmp& d_dram_block_window_tmp,
               const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               const BiasGradDramBlockWindowTmp& dbias_dram_block_window_tmp,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float raw_scale,
               float scale,
               float rp_undrop,
               float scale_rp_undrop,
               FmhaDropout& dropout) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>> &&
                std::is_same_v<OGradDataType,
                               remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                std::is_same_v<LSEDataType,
                               remove_cvref_t<typename LSEDramBlockWindowTmp::DataType>> &&
                std::is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == OGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == LSEDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == DDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = []() {
            if constexpr(kUseSplitDvGemm1BSmem)
            {
                return Policy::template GetPTOGradTBlockGemmBSmem<Problem>();
            }
            else
            {
                return Policy::template GetPTOGradTBlockGemm<Problem>();
            }
        }();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();
        constexpr index_t kSmemSizeQT   = kDoDK ? Policy::template GetSmemSizeQT<Problem>() : 0;
        constexpr index_t kSmemSizeDo   = kDoDK ? Policy::template GetSmemSizeOGrad<Problem>() : 0;
        constexpr index_t kSmemSizeDoT  =
            (kDoDK || kDoDV) ? Policy::template GetSmemSizeOGradT<Problem>() : 0;
        constexpr index_t kSmemSizeQ    = Policy::template GetSmemSizeQ<Problem>();
        constexpr index_t kSmemSizeLSE  = Policy::template GetSmemSizeLSE<Problem>();
        constexpr index_t kSmemSizeD    = kDoDK ? Policy::template GetSmemSizeD<Problem>() : 0;
        constexpr index_t kSmemSizeK    = Policy::template GetSmemSizeK<Problem>();
        constexpr index_t kSmemSizeKT   = Policy::template GetSmemSizeKT<Problem>();
        constexpr index_t kSmemSizeV    = kDoDK ? Policy::template GetSmemSizeV<Problem>() : 0;
        auto make_zero_return = [&]() {
            if constexpr(kDoDK && kDoDV)
            {
                auto dv_acc = decltype(gemm_1.MakeCBlockTile()){};
                auto dk_acc = decltype(gemm_3.MakeCBlockTile()){};
                clear_tile(dv_acc);
                clear_tile(dk_acc);
                return make_tuple(dk_acc, dv_acc);
            }
            else if constexpr(kDoDK)
            {
                auto dk_acc = decltype(gemm_3.MakeCBlockTile()){};
                clear_tile(dk_acc);
                return make_tuple(dk_acc, ck_tile::null_type{});
            }
            else
            {
                auto dv_acc = decltype(gemm_1.MakeCBlockTile()){};
                clear_tile(dv_acc);
                return make_tuple(ck_tile::null_type{}, dv_acc);
            }
        };

        // K, HBM ->LDS ->Reg
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             k_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeKDramTileDistribution<Problem>());

        KDataType* k_lds_ptr =
            static_cast<KDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        const auto k_origin = k_dram_window.get_window_origin();
        // Early termination
        const auto [seqlen_q_start, seqlen_q_end] =
            mask.GetTileRangeAlongY(k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_q_end - seqlen_q_start, kM0);

        // check early exit if masked and no work to do.
        if constexpr(FmhaMask::IsMasking)
        {
            if(num_total_loop <= 0)
            {
                return make_zero_return();
            }
        }
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsWriteBlockDescriptor<Problem>());

        auto k_lds_write_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKRegBlockDescriptor<Problem>());

        auto k_reg_tensor = make_static_distributed_tensor<KDataType>(
            Policy::template MakeKRegBlockDescriptor<Problem>());

        auto shuffled_k_block_tile = make_static_distributed_tensor<KDataType>(
            Policy::template MakeShuffledKRegWriteBlockDescriptor<Problem>());

        KDataType* kt_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));

        auto shuffled_k_lds_write = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeShuffledKLdsWriteBlockDescriptor<Problem>());

        auto shuffled_k_lds_write_window = make_tile_window(
            shuffled_k_lds_write, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto kt_lds_read = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeKTLdsReadBlockDescriptor<Problem>());

        auto kt_lds_read_window =
            make_tile_window(kt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeKTRegBlockDescriptor<Problem>());

        constexpr index_t kSmemSizeDS   = Policy::template GetSmemSizeSGrad<Problem>();
        constexpr index_t kSmemSizeBias = Policy::template GetSmemSizeBias<Problem>();
        constexpr bool kSeparateDKOnlyScratch = kDoDK && !kDoDV;
        constexpr bool kDebugDKOnlyDummyDVGemm1 =
            CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1 && kDoDK && !kDoDV;
        constexpr bool kDebugDKOnlyDummyDQGemm4 =
            CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4 && kDoDK && !kDoDV;
        constexpr bool kDebugDKOnlyDummyDQKTOnly =
            CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_KT_ONLY && kDoDK && !kDoDV;
        constexpr bool kDebugDKOnlyDummyDQDsLdsOnly =
            CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY && kDoDK && !kDoDV;
        constexpr bool kDebugDKOnlyDummyDQKTRemap =
            kDebugDKOnlyDummyDQGemm4 || kDebugDKOnlyDummyDQKTOnly;
        constexpr bool kDebugDKOnlyCarryLDSRegs =
            CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS && kDoDK && !kDoDV;
        constexpr index_t kGenericQTGuardSize = kDoDK ? kSmemSizeQT : 0;
        constexpr index_t kGenericQTBaseSize =
            kDoDK ? kSmemSizeQT + kGenericQTGuardSize : 0;
        constexpr index_t kGenericDoOffset         = kGenericQTBaseSize;
        constexpr index_t kGenericDoTOffset        = kGenericDoOffset + kSmemSizeDo;
        constexpr index_t kGenericQOffset          = kGenericDoTOffset + kSmemSizeDoT;
        constexpr index_t kGenericLSEOffset        = kGenericQOffset + kSmemSizeQ;
        constexpr index_t kGenericDOffset          = kGenericLSEOffset + kSmemSizeLSE;
        constexpr index_t kGenericSharedTailOffset = kGenericDOffset + kSmemSizeD;
        constexpr index_t kGenericQTOffset         = 0;
        constexpr index_t kGenericDSOffset         = kGenericSharedTailOffset;
        constexpr index_t kGenericBiasOffset       = kGenericSharedTailOffset;

        constexpr index_t kLegacyStage0Size =
            max(Policy::template GetSmemSizeK<Problem>() + Policy::template GetSmemSizeKT<Problem>(),
                kDoDK ? Policy::template GetSmemSizeV<Problem>() : 0);
        constexpr index_t kLegacyStage1SharedSize =
            kSmemSizeDo + kSmemSizeDoT + kSmemSizeQ + kSmemSizeLSE + kSmemSizeD +
            (kSeparateDKOnlyScratch ? kSmemSizeBias : max(kSmemSizeBias, kSmemSizeDS));
        constexpr index_t kLegacySharedBaseSize =
            max(kLegacyStage0Size, kLegacyStage1SharedSize);
        constexpr index_t kLegacyDoOffset         = 0;
        constexpr index_t kLegacyDoTOffset        = kLegacyDoOffset + kSmemSizeDo;
        constexpr index_t kLegacyQOffset          = kLegacyDoTOffset + kSmemSizeDoT;
        constexpr index_t kLegacyLSEOffset        = kLegacyQOffset + kSmemSizeQ;
        constexpr index_t kLegacyDOffset          = kLegacyLSEOffset + kSmemSizeLSE;
        constexpr index_t kLegacySharedTailOffset = kLegacyDOffset + kSmemSizeD;
        constexpr index_t kLegacyQTOffset         = kDoDK ? kLegacySharedBaseSize : 0;
        constexpr index_t kLegacyQTGuardSize      = kDoDK ? kSmemSizeQT : 0;
        constexpr index_t kLegacyDSOffset =
            kSeparateDKOnlyScratch ? (kLegacyQTOffset + kSmemSizeQT + kLegacyQTGuardSize)
                                   : kLegacySharedTailOffset;
        constexpr index_t kLegacyBiasOffset = kLegacySharedTailOffset;

        constexpr index_t kQTGuardSize =
            kUseGenericLDSLayout ? kGenericQTGuardSize : kLegacyQTGuardSize;
        constexpr index_t kDoOffset  = kUseGenericLDSLayout ? kGenericDoOffset : kLegacyDoOffset;
        constexpr index_t kDoTOffset = kUseGenericLDSLayout ? kGenericDoTOffset : kLegacyDoTOffset;
        constexpr index_t kQOffset   = kUseGenericLDSLayout ? kGenericQOffset : kLegacyQOffset;
        constexpr index_t kLSEOffset = kUseGenericLDSLayout ? kGenericLSEOffset : kLegacyLSEOffset;
        constexpr index_t kDOffset   = kUseGenericLDSLayout ? kGenericDOffset : kLegacyDOffset;
        constexpr index_t kSharedTailOffset =
            kUseGenericLDSLayout ? kGenericSharedTailOffset : kLegacySharedTailOffset;
        constexpr index_t kQTOffset = kUseGenericLDSLayout ? kGenericQTOffset : kLegacyQTOffset;
        constexpr index_t kDSOffset = kUseGenericLDSLayout ? kGenericDSOffset : kLegacyDSOffset;
        constexpr index_t kBiasOffset =
            kUseGenericLDSLayout ? kGenericBiasOffset : kLegacyBiasOffset;

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4
        auto kt_reg_tensor = make_static_distributed_tensor<KDataType>(
            Policy::template MakeKTRegBlockDescriptor<Problem>());
#endif

        //------------------------------------------------------------------
        // V, HBM ->LDS ->Reg
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             v_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeVDramTileDistribution<Problem>());

        VDataType* v_lds_ptr =
            static_cast<VDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto v_lds = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kVHeaddim>{}), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kVHeaddim>{}),
                             v_lds_write_window.get_window_origin(),
                             Policy::template MakeVRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // Pre-load K/V into registers. DK-only still writes the shuffled K^T LDS image for
        // debug DQ probes, but the formal DK path must not carry a dead K^T register tile.
        auto k_block_tile = load_tile(k_dram_window);
        auto v_block_tile = make_static_distributed_tensor<VDataType>(
            Policy::template MakeVDramTileDistribution<Problem>());
        if constexpr(kDoDK)
        {
            v_block_tile = load_tile(v_dram_window);
        }

        store_tile(k_lds_write_window, k_block_tile);
        if constexpr(kDoDK && !kDoDV)
        {
            shuffle_tile(shuffled_k_block_tile, k_block_tile);
            store_tile(shuffled_k_lds_write_window, shuffled_k_block_tile);
        }
        block_sync_lds();
        k_reg_tensor = load_tile(k_lds_read_window);
        if constexpr(kDoDK && !kDoDV)
        {
            block_sync_lds();
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4
            kt_reg_tensor = load_tile(kt_lds_read_window);
#endif
        }

        auto v_reg_tensor = make_static_distributed_tensor<VDataType>(
            Policy::template MakeVRegBlockDescriptor<Problem>());
        if constexpr(kDoDK)
        {
            store_tile(v_lds_write_window, v_block_tile);
            block_sync_lds();
            v_reg_tensor = load_tile(v_lds_read_window);
        }
        else
        {
            block_sync_lds();
        }
        //---------------------------- Loop Load in ----------------------------//
        // Q: HBM ->Reg ->LDS
        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeQDramTileDistribution<Problem>());

        QDataType* q_lds_ptr =
            static_cast<QDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr) + kQOffset));

        auto q_lds = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());

        // QT: Reg -> Reg-> LDS
        auto shuffled_q_block_tile = make_static_distributed_tensor<QDataType>(
            Policy::template MakeShuffledQRegWriteBlockDescriptor<Problem>());

        QDataType* qt_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + kQTOffset));

        auto shuffled_q_lds_write = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeShuffledQLdsWriteBlockDescriptor<Problem>());

        auto shuffled_q_lds_write_window = make_tile_window(
            shuffled_q_lds_write, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto qt_lds_read = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeQTLdsReadBlockDescriptor<Problem>());

        auto qt_lds_read_window =
            make_tile_window(qt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeQTRegSliceBlockDescriptor<Problem>());

        // dO: HBM ->Reg ->LDS
        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeOGradDramTileDistribution<Problem>());

        OGradDataType* do_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + kDoOffset));

        auto do_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());

        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto do_lds_read_window =
            make_tile_window(do_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             do_lds_window.get_window_origin(),
                             Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());
        // dOT: Reg ->Reg ->LDS
        auto shuffled_do_block_tile = make_static_distributed_tensor<OGradDataType>(
            Policy::template MakeShuffledOGradRegWriteBlockDescriptor<Problem>());

        OGradDataType* dot_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + kDoTOffset));

        auto shuffled_do_lds_write = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeShuffledOGradLdsWriteBlockDescriptor<Problem>());

        auto shuffled_do_lds_write_window = make_tile_window(
            shuffled_do_lds_write, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto dot_read_lds = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsReadBlockDescriptor<Problem>());

        auto dot_lds_read_window =
            make_tile_window(dot_read_lds,
                             make_tuple(number<kVHeaddim>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeOGradTRegSliceBlockDescriptor<Problem>());

        // dS: Reg -> Reg -> LDS
        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(
            static_cast<void*>(static_cast<char*>(smem_ptr) + kDSOffset));

        auto ds_lds = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());

        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY ||     \
    CK_TILE_DEBUG_BWD_SPLIT_DKDV_D256_DS_LDS_MATERIALIZE ||        \
    CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4 ||                 \
    CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY
        auto ds_lds_read_window =
            make_tile_window(ds_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK4>{}),
                             ds_lds_window.get_window_origin(),
                             Policy::template MakeSGradRegSliceBlockDescriptor<Problem>());
#endif

        auto ds_lds_full_read_window =
            make_tile_window(ds_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kN0>{}),
                             ds_lds_window.get_window_origin(),
                             remove_cvref_t<decltype(gemm_2.MakeCBlockTile())>::
                                 get_tile_distribution());

        static_assert(std::is_same_v<BiasDataType, BiasGradDataType>,
                      "BiasDataType and BiasGradDataType should be the same!");

        // LSE: HBM -> LDS ->Reg
        auto lse_dram_window = make_tile_window(
            lse_dram_block_window_tmp.get_bottom_tensor_view(),
            lse_dram_block_window_tmp.get_window_lengths(),
            {seqlen_q_start},
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        LSEDataType* lse_lds_ptr = static_cast<LSEDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + kLSEOffset));

        auto lse_lds = make_tensor_view<address_space_enum::lds>(
            lse_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto lse_lds_write_window = make_tile_window(lse_lds, make_tuple(number<kM0>{}), {0});

        auto lse_lds_read_window = make_tile_window(
            lse_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        // D: HBM ->Reg
        auto d_dram_window = make_tile_window(
            d_dram_block_window_tmp.get_bottom_tensor_view(),
            d_dram_block_window_tmp.get_window_lengths(),
            {seqlen_q_start},
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        DDataType* d_lds_ptr = static_cast<DDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + kDOffset));

        auto d_lds = make_tensor_view<address_space_enum::lds>(
            d_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto d_lds_write_window = make_tile_window(d_lds, make_tuple(number<kM0>{}), {0});

        auto d_lds_read_window = make_tile_window(
            d_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        // RandVal: HBM ->Reg
        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0), false>(
            randval_dram_block_window_tmp, seqlen_q_start);

        using SPBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType  = decltype(gemm_4.MakeCBlockTile());
        constexpr bool kPNeedsCastToGemm =
            !std::is_same_v<typename SPBlockTileType::DataType, GemmDataType>;
        constexpr bool kSGradNeedsCastToGemm =
            !std::is_same_v<typename SPGradBlockTileType::DataType, GemmDataType>;
        auto dk_acc = [&]() {
            if constexpr(kDoDK)
                return decltype(gemm_3.MakeCBlockTile()){};
            else
                return ck_tile::null_type{};
        }();
        auto dv_acc = [&]() {
            if constexpr(kDoDV)
                return decltype(gemm_1.MakeCBlockTile()){};
            else
                return ck_tile::null_type{};
        }();
        index_t i_total_loops = 0;
        index_t seqlen_q_step = seqlen_q_start;
        const index_t k_origin_col0 = k_origin.at(number<0>{});
        const index_t bias_origin_col0 = bias_dram_block_window_tmp.get_window_origin().at(number<1>{});
        const index_t dbias_origin_col0 =
            dbias_dram_block_window_tmp.get_window_origin().at(number<1>{});
        static_assert(kQKHeaddim >= kK0, "kQKHeaddim should be equal or greater than kK0");
        static_assert(kM0 == kK1, "kM0 should equal to kK1");
        static_assert(kVHeaddim >= kK2, "kVHeaddim should be equal or greater than kK2");
        static_assert(kM0 == kK3, "kM0 should equal to kK3");
        constexpr index_t k4_loops = kN0 / kK4;
        /*
         * Prefetch Q, LSE, dO, D
         */
        auto q_block_tile = load_tile(q_dram_window);
        move_tile_window(q_dram_window, {kM0, 0});
        auto lse_block_tile = load_tile(lse_dram_window);
        move_tile_window(lse_dram_window, {kM0});

        auto do_block_tile = load_tile(do_dram_window);
        move_tile_window(do_dram_window, {kM0, 0});

        auto d_block_tile = [&]() {
            if constexpr(kDoDK)
                return load_tile(d_dram_window);
            else
                return ck_tile::null_type{};
        }();
        if constexpr(kDoDK)
        {
            move_tile_window(d_dram_window, {kM0});
        }

        /*
         * Store prefetched data into LDS
         */
        block_sync_lds();
        store_tile(q_lds_window, q_block_tile);
        if constexpr(kDoDK)
        {
            shuffle_tile(shuffled_q_block_tile, q_block_tile);
            store_tile(shuffled_q_lds_write_window, shuffled_q_block_tile);
        }

        store_tile(lse_lds_write_window, lse_block_tile);

        if constexpr(kDoDK)
        {
            store_tile(do_lds_window, do_block_tile);
        }
        if constexpr(kDoDV || kDebugDKOnlyDummyDVGemm1)
        {
            shuffle_tile(shuffled_do_block_tile, do_block_tile);
            store_tile(shuffled_do_lds_write_window, shuffled_do_block_tile);
        }

        if constexpr(kDoDK)
        {
            store_tile(d_lds_write_window, d_block_tile);
        }
        block_sync_lds();

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
        auto carried_q_reg_tensor = load_tile(q_lds_read_window);
        auto carried_lse          = load_tile(lse_lds_read_window);
        auto carried_do_reg_tensor = [&]() {
            if constexpr(kDebugDKOnlyCarryLDSRegs)
            {
                return load_tile(do_lds_read_window);
            }
            else
            {
                return ck_tile::null_type{};
            }
        }();
        auto carried_d = [&]() {
            if constexpr(kDebugDKOnlyCarryLDSRegs)
            {
                return load_tile(d_lds_read_window);
            }
            else
            {
                return ck_tile::null_type{};
            }
        }();
        auto carried_qt_reg_tensor = [&]() {
            if constexpr(kDebugDKOnlyCarryLDSRegs)
            {
                return load_tile(qt_lds_read_window);
            }
            else
            {
                return ck_tile::null_type{};
            }
        }();
#endif

        if constexpr(kDoDV)
            clear_tile(dv_acc);
        if constexpr(kDoDK)
            clear_tile(dk_acc);

        auto debug_dummy_dq_gemm4 = [&](const auto& ds_like) {
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4 || \
    CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_DS_LDS_ONLY
            if constexpr(kDebugDKOnlyDummyDQGemm4 || kDebugDKOnlyDummyDQDsLdsOnly)
            {
                const auto ds_gemm = cast_tile<GemmDataType>(ds_like);
                store_tile(ds_lds_window, ds_gemm);
                block_sync_lds();

                auto ds_reg_tensor      = load_tile(ds_lds_read_window);
                auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
                move_tile_window(ds_lds_read_window, {0, kK4});

                auto dq_dummy_acc = QGradBlockTileType{};
                clear_tile(dq_dummy_acc);
                if constexpr(kDebugDKOnlyDummyDQGemm4)
                {
                    static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                        if constexpr(CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_ONE_SLICE)
                        {
                            if constexpr(i_k4 > 0)
                            {
                                return;
                            }
                        }
                    if constexpr(i_k4 < k4_loops - 1)
                    {
                        ds_reg_tensor_next = load_tile(ds_lds_read_window);
                        move_tile_window(ds_lds_read_window, {0, kK4});
                    }
                    auto kt_reg_tensor_slice =
                        get_slice_tile(kt_reg_tensor,
                                       sequence<0, i_k4 * kK4>{},
                                       sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});
                    gemm_4(dq_dummy_acc, ds_reg_tensor, kt_reg_tensor_slice);
                    if constexpr(i_k4 < k4_loops - 1)
                    {
                        ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
                    }
                    });
                }
                move_tile_window(ds_lds_read_window, {0, -kN0});

                if constexpr(!kDebugDKOnlyDummyDQGemm4)
                {
                    constexpr auto ds_dummy_buf_size =
                        remove_cvref_t<decltype(ds_reg_tensor)>::get_thread_buffer_size();
                    static_for<0, ds_dummy_buf_size, 1>{}([&](auto i) {
                        auto ds_debug_value =
                            type_convert<float>(ds_reg_tensor.get_thread_buffer().at(i));
                        asm volatile("" : "+v"(ds_debug_value) : : "memory");
                    });
                    return;
                }

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE == 0
                (void)dq_dummy_acc;
#elif CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DQ_GEMM4_SINK_MODE == 1
                auto dq_debug_value =
                    type_convert<float>(dq_dummy_acc.get_thread_buffer().at(number<0>{}));
                asm volatile("" : "+v"(dq_debug_value) : : "memory");
#else
                constexpr auto dq_dummy_buf_size =
                    remove_cvref_t<decltype(dq_dummy_acc)>::get_thread_buffer_size();
                static_for<0, dq_dummy_buf_size, 1>{}([&](auto i) {
                    auto dq_debug_value =
                        type_convert<float>(dq_dummy_acc.get_thread_buffer().at(i));
                    asm volatile("" : "+v"(dq_debug_value) : : "memory");
                });
#endif
            }
#else
            (void)ds_like;
#endif
        };

        __builtin_amdgcn_sched_barrier(0);
        // Hot loop
        while(i_total_loops < (num_total_loop - 1))
        {
            // STAGE 1, Q@K Gemm0
            auto s_acc = SPBlockTileType{};
            const index_t q_step = seqlen_q_step;

            if constexpr(!kDelayDVOnlyQDoPrefetch)
            {
                q_block_tile = load_tile(q_dram_window);
                move_tile_window(q_dram_window, {kM0, 0});

                lse_block_tile = load_tile(lse_dram_window);
                move_tile_window(lse_dram_window, {kM0});

                do_block_tile = load_tile(do_dram_window);
                move_tile_window(do_dram_window, {kM0, 0});
            }

            if constexpr(kDoDK)
            {
                d_block_tile = load_tile(d_dram_window);
                move_tile_window(d_dram_window, {kM0});
            }

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
            if constexpr(kDebugDKOnlyCarryLDSRegs)
            {
                s_acc = gemm_0(carried_q_reg_tensor, k_reg_tensor);
            }
            else
#endif
            {
                auto q_reg_tensor = load_tile(q_lds_read_window);
                s_acc = gemm_0(q_reg_tensor, k_reg_tensor);
            }

            HotLoopScheduler::template GemmStagedScheduler<0>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                auto bias_dram_window =
                    make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                                     bias_dram_block_window_tmp.get_window_lengths(),
                                     {q_step, bias_origin_col0},
                                     Policy::template MakeBiasTileDistribution<Problem>());
                auto* bias_lds_ptr = reinterpret_cast<BiasDataType*>(
                    static_cast<char*>(smem_ptr) + kBiasOffset);
                auto bias_lds = make_tensor_view<address_space_enum::lds>(
                    bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());
                auto bias_lds_write_window =
                    make_tile_window(bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});
                auto bias_s_lds_read_window =
                    make_tile_window(bias_lds_write_window.get_bottom_tensor_view(),
                                     bias_lds_write_window.get_window_lengths(),
                                     bias_lds_write_window.get_window_origin(),
                                     Policy::template MakeBiasSTileDistribution<decltype(gemm_0)>());
                const auto bias_tile    = load_tile(bias_dram_window);
                auto shuffled_bias_tile = make_static_distributed_tensor<BiasDataType>(
                    Policy::template MakeShuffledBiasTileDistribution<Problem>());
                shuffle_tile(shuffled_bias_tile, bias_tile);
                // SGrad and Bias use the same address in LDS, finish loading ds on the previous
                // iteration to reuse LDS.
                block_sync_lds();
                store_tile(bias_lds_write_window, shuffled_bias_tile);
                block_sync_lds();
                auto bias_s_tile = load_tile(bias_s_lds_read_window);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
                        x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                    },
                    s_acc,
                    bias_s_tile);
                move_tile_window(bias_dram_window, {kM0, 0});
                __builtin_amdgcn_sched_barrier(0);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin_col0 + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        s_acc(i_j_idx) *= scale;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }

            {
                bool need_perpixel_check = [&]() {
                    if constexpr(kSkipDKOnlyNonMaskEdgeCheck || kSkipDVOnlyNonMaskEdgeCheck)
                    {
                        return false;
                    }
                    else
                    {
                        return mask.IsEdgeTile(
                            q_step, k_origin_col0, number<kM0>{}, number<kN0>{});
                    }
                }();
                if(need_perpixel_check)
                {
                    set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                        const auto row = q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin_col0 + tile_idx.at(number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
                }
            }

            static const auto get_validated_lse = [](LSEDataType raw_lse) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_lse == -numeric<LSEDataType>::infinity()
                               ? type_convert<LSEDataType>(0.f)
                               : raw_lse;
                }
                else
                {
                    return raw_lse;
                }
            };

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
            auto lse = [&]() {
                if constexpr(kDebugDKOnlyCarryLDSRegs)
                {
                    return carried_lse;
                }
                else
                {
                    return load_tile(lse_lds_read_window);
                }
            }();
#else
            auto lse = load_tile(lse_lds_read_window);
#endif
            decltype(auto) p = [&]() -> decltype(auto) {
                if constexpr(kUseDKOnlyFreshP)
                {
                    return SPBlockTileType{};
                }
                else
                {
                    return (s_acc);
                }
            }();
            constexpr auto p_spans = remove_cvref_t<decltype(p)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p(i_j_idx) = exp2(s_acc[i_j_idx] - row_lse);
                    }
                    else
                    {
                        p(i_j_idx) = exp2(scale * s_acc[i_j_idx] - row_lse);
                    }
                });
            });

            if constexpr(FmhaDropout::IsDropout)
            {
                dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                    q_step, k_origin_col0, p, randval_dram_window);
            }
            auto debug_dummy_dv_gemm1 = [&](const auto& p_tile) {
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1
                if constexpr(kDoDK && !kDoDV)
                {
                    auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                        Policy::template MakePTRegSliceBlockDescriptor<Problem>());
                    const auto p_gemm = [&]() {
                        if constexpr(FmhaDropout::IsDropout)
                        {
                            return tile_elementwise_in(
                                [](const auto& x) {
                                    return type_convert<GemmDataType>(x > 0.f ? x : 0.f);
                                },
                                p_tile);
                        }
                        else
                        {
                            return cast_tile<GemmDataType>(p_tile);
                        }
                    }();
                    Policy::template PTFromGemm0CToGemm1A<Problem,
                                                          decltype(pt_reg_tensor),
                                                          decltype(p_gemm)>(
                        pt_reg_tensor, p_gemm, ds_lds_ptr);
                    auto dot_reg_tensor = load_tile(dot_lds_read_window);
                    auto dv_dummy_acc   = decltype(gemm_1.MakeCBlockTile()){};
                    clear_tile(dv_dummy_acc);
                    gemm_1(dv_dummy_acc, pt_reg_tensor, dot_reg_tensor);
                    constexpr auto dv_dummy_buf_size =
                        remove_cvref_t<decltype(dv_dummy_acc)>::get_thread_buffer_size();
                    static_for<0, dv_dummy_buf_size, 1>{}([&](auto i) {
                        auto dv_debug_value =
                            type_convert<float>(dv_dummy_acc.get_thread_buffer().at(i));
                        asm volatile("" : "+v"(dv_debug_value) : : "memory");
                    });
                }
#else
                (void)p_tile;
#endif
            };

            if constexpr(kDoDV)
            {
                auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                    Policy::template MakePTRegSliceBlockDescriptor<Problem>());
                if constexpr(FmhaDropout::IsDropout || kPNeedsCastToGemm)
                {
                    const auto p_gemm = [&]() {
                        if constexpr(FmhaDropout::IsDropout)
                        {
                            return tile_elementwise_in(
                                [](const auto& x) {
                                    return type_convert<GemmDataType>(x > 0.f ? x : 0.f);
                                },
                                p);
                        }
                        else
                        {
                            return cast_tile<GemmDataType>(p);
                        }
                    }();
                    Policy::template PTFromGemm0CToGemm1A<Problem>(
                        pt_reg_tensor, p_gemm, ds_lds_ptr);
                }
                else
                {
                    Policy::template PTFromGemm0CToGemm1A<Problem>(pt_reg_tensor, p, ds_lds_ptr);
                }
                if constexpr(kUseSplitDvGemm1BSmem)
                {
                    gemm_1(dv_acc, pt_reg_tensor, dot_lds_read_window);
                }
                else
                {
                    auto dot_reg_tensor = load_tile(dot_lds_read_window);
                    gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);
                }
            }
            else
            {
                debug_dummy_dv_gemm1(p);
            }

                HotLoopScheduler::template GemmStagedScheduler<1>();
            __builtin_amdgcn_sched_barrier(0);
            if constexpr(kDoDK)
            {
                // STAGE 4, OGrad@V Gemm2
                auto dp_acc        = SPGradBlockTileType{};
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
                if constexpr(kDebugDKOnlyCarryLDSRegs)
                {
                    dp_acc = gemm_2(carried_do_reg_tensor, v_reg_tensor);
                }
                else
#endif
                {
                    auto do_reg_tensor = load_tile(do_lds_read_window);
                    dp_acc = gemm_2(do_reg_tensor, v_reg_tensor);
                }
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
                auto d = [&]() {
                    if constexpr(kDebugDKOnlyCarryLDSRegs)
                    {
                        return carried_d;
                    }
                    else
                    {
                        return load_tile(d_lds_read_window);
                    }
                }();
#else
                auto d             = load_tile(d_lds_read_window);
#endif
                // Gemm3 must consume the current tile's Q^T before shuffled_q LDS is overwritten
                // by the next iteration's Q. Keep the load callable so DK-only can compute SGrad
                // first and avoid carrying Q^T across Stage 5.
                auto load_qt_reg_tensor = [&]() {
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
                    if constexpr(kDebugDKOnlyCarryLDSRegs)
                    {
                        return carried_qt_reg_tensor;
                    }
                    else
                    {
                        return load_tile(qt_lds_read_window);
                    }
#else
                    return load_tile(qt_lds_read_window);
#endif
                };

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_LATE_QDO_STORE_LAMBDA
#define CK_TILE_DKDV_STORE_NEXT_QDO_TILE()                                                \
    do                                                                                    \
    {                                                                                     \
        block_sync_lds();                                                                 \
        store_tile(q_lds_window, q_block_tile);                                           \
        shuffle_tile(shuffled_q_block_tile, q_block_tile);                                \
        store_tile(shuffled_q_lds_write_window, shuffled_q_block_tile);                    \
        store_tile(lse_lds_write_window, lse_block_tile);                                 \
        store_tile(do_lds_window, do_block_tile);                                         \
        if constexpr(kDoDV || kDebugDKOnlyDummyDVGemm1)                                   \
        {                                                                                 \
            shuffle_tile(shuffled_do_block_tile, do_block_tile);                          \
            store_tile(shuffled_do_lds_write_window, shuffled_do_block_tile);              \
        }                                                                                 \
        store_tile(d_lds_write_window, d_block_tile);                                     \
        HotLoopScheduler::template GemmStagedScheduler<2>();                              \
        __builtin_amdgcn_sched_barrier(0);                                                \
    } while(false)
#else
                auto store_next_qdo_tile = [&]() {
                    block_sync_lds();

                    store_tile(q_lds_window, q_block_tile);
                    shuffle_tile(shuffled_q_block_tile, q_block_tile);
                    store_tile(shuffled_q_lds_write_window, shuffled_q_block_tile);
                    store_tile(lse_lds_write_window, lse_block_tile);
                    store_tile(do_lds_window, do_block_tile);
                    if constexpr(kDoDV || kDebugDKOnlyDummyDVGemm1)
                    {
                        shuffle_tile(shuffled_do_block_tile, do_block_tile);
                        store_tile(shuffled_do_lds_write_window, shuffled_do_block_tile);
                    }
                    store_tile(d_lds_write_window, d_block_tile);

                    HotLoopScheduler::template GemmStagedScheduler<2>();
                    __builtin_amdgcn_sched_barrier(0);
                };
#define CK_TILE_DKDV_STORE_NEXT_QDO_TILE() store_next_qdo_tile()
#endif
                // STAGE 5, P^T(PGrad^T - D)
                if constexpr(kSeparateDKOnlyScratch && !kHasBiasGrad)
                {
                    if constexpr(kUseD256InlineDsRemap || kUseD256DsLdsMaterialize)
                    {
                        auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                            Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
                        {
                            auto ds_gemm = make_static_distributed_tensor<GemmDataType>(
                                SPGradBlockTileType::get_tile_distribution());
                            constexpr auto ds_spans = decltype(ds_gemm)::get_distributed_spans();
                            if constexpr(FmhaDropout::IsDropout)
                            {
                                sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                                    constexpr auto i_idx = make_tuple(idx0);
                                    sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                        bool undrop_flag       = p[i_j_idx] >= 0;
                                        ds_gemm(i_j_idx) = type_convert<GemmDataType>(
                                            p[i_j_idx] *
                                            (undrop_flag ? (dp_acc[i_j_idx] - d[i_idx]) : d[i_idx]));
                                    });
                                });
                            }
                            else
                            {
                                sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                                    constexpr auto i_idx = make_tuple(idx0);
                                    sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                        ds_gemm(i_j_idx) = type_convert<GemmDataType>(
                                            p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]));
                                    });
                                });
                            }
                            if constexpr(kUseD256DsLdsMaterialize)
                            {
                                store_tile(ds_lds_window, ds_gemm);
                                block_sync_lds();
                                auto ds_reloaded = load_tile(ds_lds_full_read_window);
                                block_sync_lds();
                                Policy::template SGradTFromGemm2CToGemm3A<
                                    Problem,
                                    decltype(dst_reg_tensor),
                                    decltype(ds_reloaded)>(
                                    dst_reg_tensor, ds_reloaded, ds_lds_ptr);
                            }
                            else
                            {
                                Policy::template SGradTFromGemm2CToGemm3A<
                                    Problem,
                                    decltype(dst_reg_tensor),
                                    decltype(ds_gemm)>(dst_reg_tensor, ds_gemm, ds_lds_ptr);
                            }
                        }
                        auto qt_reg_tensor = load_qt_reg_tensor();
                        gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
#if defined(CK_TILE_DEBUG_BWD_GEMM3_SCHED_BARRIER)
                        __builtin_amdgcn_sched_barrier(0);
#endif
                        CK_TILE_DKDV_STORE_NEXT_QDO_TILE();
                    }
                    else
                    {
                        auto make_ds_gemm = [&]() {
                            auto ds_gemm = make_static_distributed_tensor<GemmDataType>(
                                SPGradBlockTileType::get_tile_distribution());
                            constexpr auto ds_spans = decltype(ds_gemm)::get_distributed_spans();
                            if constexpr(FmhaDropout::IsDropout)
                            {
                                sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                                    constexpr auto i_idx = make_tuple(idx0);
                                    sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                        bool undrop_flag       = p[i_j_idx] >= 0;
                                        ds_gemm(i_j_idx) = type_convert<GemmDataType>(
                                            p[i_j_idx] *
                                            (undrop_flag ? (dp_acc[i_j_idx] - d[i_idx]) : d[i_idx]));
                                    });
                                });
                            }
                            else
                            {
                                sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                                    constexpr auto i_idx = make_tuple(idx0);
                                    sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                        ds_gemm(i_j_idx) = type_convert<GemmDataType>(
                                            p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]));
                                    });
                                });
                            }
                            return ds_gemm;
                        };

                        auto make_dst_reg_tensor = [&]() {
                            auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                                Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
                            {
                                // STAGE 6 remap. Keep ds_gemm scoped to the remap so it does not
                                // overlap the Q^T register tile and GEMM3 scheduling.
                                const auto ds_gemm = make_ds_gemm();
                                Policy::template SGradTFromGemm2CToGemm3A<
                                    Problem,
                                    decltype(dst_reg_tensor),
                                    remove_cvref_t<decltype(ds_gemm)>>(
                                    dst_reg_tensor, ds_gemm, ds_lds_ptr);
                            }
                            return dst_reg_tensor;
                        };

                        auto run_gemm3 = [&](const auto& dst_reg_tensor, const auto& qt_reg_tensor) {
                            // STAGE 6, SGrad^T@Q^T Gemm3.
                            gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
#if defined(CK_TILE_DEBUG_BWD_GEMM3_SCHED_BARRIER)
                            __builtin_amdgcn_sched_barrier(0);
#endif
                        };

                        auto dst_reg_tensor = make_dst_reg_tensor();
                        auto qt_reg_tensor   = load_qt_reg_tensor();
                        run_gemm3(dst_reg_tensor, qt_reg_tensor);
                        CK_TILE_DKDV_STORE_NEXT_QDO_TILE();
                    }
                }
                else
                {
                    auto qt_reg_tensor = load_qt_reg_tensor();
                    CK_TILE_DKDV_STORE_NEXT_QDO_TILE();
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_DS
                    auto ds_work = SPGradBlockTileType{};
#else
                    auto& ds_work = dp_acc;
#endif
                    constexpr auto ds_spans =
                        remove_cvref_t<decltype(ds_work)>::get_distributed_spans();
                    if constexpr(FmhaDropout::IsDropout)
                    {
                        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                            constexpr auto i_idx = make_tuple(idx0);
                            sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                bool undrop_flag       = p[i_j_idx] >= 0;
                                ds_work(i_j_idx) = p[i_j_idx] *
                                                  (undrop_flag ? (dp_acc[i_j_idx] - d[i_idx])
                                                               : d[i_idx]);
                            });
                        });
                    }
                    else
                    {
                        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                            constexpr auto i_idx = make_tuple(idx0);
                            sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                ds_work(i_j_idx) = p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]);
                            });
                        });
                    }

                    if constexpr(kHasBiasGrad && !kSeparateDKOnlyScratch)
                    {
                        auto* bias_lds_ptr = reinterpret_cast<BiasDataType*>(
                            static_cast<char*>(smem_ptr) + kBiasOffset);
                        auto bias_lds = make_tensor_view<address_space_enum::lds>(
                            bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());
                        auto bias_lds_write_window = make_tile_window(
                            bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});
                        auto dbias_dram_window =
                            make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                                             dbias_dram_block_window_tmp.get_window_lengths(),
                                             {q_step, dbias_origin_col0});
                        auto dbias_lds_read_window = make_tile_window(
                            bias_lds,
                            make_tuple(number<kM0>{}, number<kN0>{}),
                            {0, 0},
                            Policy::template MakeShuffledBiasTileDistribution<Problem>());
                        const auto dbias = [&]() {
                            if constexpr(FmhaDropout::IsDropout)
                            {
                                return tile_elementwise_in(
                                    [&rp_undrop](const auto& x) {
                                        return type_convert<BiasGradDataType>(x * rp_undrop);
                                    },
                                    ds_work);
                            }
                            else
                            {
                                return cast_tile<BiasGradDataType>(ds_work);
                            }
                        }();
                        store_tile(bias_lds_write_window, dbias);
                        block_sync_lds();
                        auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
                        auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                            Policy::template MakeBiasTileDistribution<Problem>());
                        shuffle_tile(dbias_tile, shuffled_dbias_tile);
                        store_tile(dbias_dram_window, dbias_tile);
                        move_tile_window(dbias_dram_window, {kM0, 0});
                        __builtin_amdgcn_sched_barrier(0);
                    }

                    // STAGE 6, SGrad^T@Q^T Gemm3
                    if constexpr(kSGradNeedsCastToGemm)
                    {
                        auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                            Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_PRECAST_SGRAD
                        const auto ds_gemm = cast_tile<GemmDataType>(ds_work);
                        Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                                  decltype(dst_reg_tensor),
                                                                  decltype(ds_gemm)>(
                            dst_reg_tensor, ds_gemm, ds_lds_ptr);
#else
                        Policy::template SGradTFromGemm2CToGemm3A<Problem>(
                            dst_reg_tensor, ds_work, ds_lds_ptr);
#endif
                        gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY
                        const auto ds_gemm_debug = cast_tile<GemmDataType>(ds_work);
                        store_tile(ds_lds_window, ds_gemm_debug);
                        block_sync_lds();
                        auto ds_reg_tensor_debug = load_tile(ds_lds_read_window);
                        constexpr auto ds_debug_buf_size =
                            remove_cvref_t<decltype(ds_reg_tensor_debug)>::get_thread_buffer_size();
                        static_for<0, ds_debug_buf_size, 1>{}([&](auto i) {
                            auto ds_debug_value = type_convert<float>(
                                ds_reg_tensor_debug.get_thread_buffer().at(i));
                            asm volatile("" : "+v"(ds_debug_value) : : "memory");
                        });
#endif
#if defined(CK_TILE_DEBUG_BWD_GEMM3_SCHED_BARRIER)
                        __builtin_amdgcn_sched_barrier(0);
#endif
                        if constexpr(kHasBiasGrad && !kSeparateDKOnlyScratch)
                        {
                            block_sync_lds();
                        }
                    }
                    else
                    {
                        const auto& ds_gemm = ds_work;
                        auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                            Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
                        Policy::template SGradTFromGemm2CToGemm3A<Problem>(
                            dst_reg_tensor, ds_gemm, ds_lds_ptr);
                        gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY
                        const auto ds_gemm_debug = cast_tile<GemmDataType>(ds_work);
                        store_tile(ds_lds_window, ds_gemm_debug);
                        block_sync_lds();
                        auto ds_reg_tensor_debug = load_tile(ds_lds_read_window);
                        constexpr auto ds_debug_buf_size =
                            remove_cvref_t<decltype(ds_reg_tensor_debug)>::get_thread_buffer_size();
                        static_for<0, ds_debug_buf_size, 1>{}([&](auto i) {
                            auto ds_debug_value = type_convert<float>(
                                ds_reg_tensor_debug.get_thread_buffer().at(i));
                            asm volatile("" : "+v"(ds_debug_value) : : "memory");
                        });
#endif
#if defined(CK_TILE_DEBUG_BWD_GEMM3_SCHED_BARRIER)
                        __builtin_amdgcn_sched_barrier(0);
#endif
                        if constexpr(kHasBiasGrad && !kSeparateDKOnlyScratch)
                        {
                            block_sync_lds();
                        }
                    }

                    debug_dummy_dq_gemm4(ds_work);
                }
#undef CK_TILE_DKDV_STORE_NEXT_QDO_TILE

                if constexpr(kHasBiasGrad && kSeparateDKOnlyScratch)
                {
                    auto* bias_lds_ptr = reinterpret_cast<BiasDataType*>(
                        static_cast<char*>(smem_ptr) + kBiasOffset);
                    auto bias_lds = make_tensor_view<address_space_enum::lds>(
                        bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());
                    auto bias_lds_write_window =
                        make_tile_window(bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});
                    auto dbias_dram_window =
                        make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                                         dbias_dram_block_window_tmp.get_window_lengths(),
                                         {q_step, dbias_origin_col0});
                    auto dbias_lds_read_window =
                        make_tile_window(bias_lds,
                                         make_tuple(number<kM0>{}, number<kN0>{}),
                                         {0, 0},
                                         Policy::template MakeShuffledBiasTileDistribution<Problem>());
                    const auto dbias = [&]() {
                        if constexpr(FmhaDropout::IsDropout)
                        {
                            return tile_elementwise_in(
                                [&rp_undrop](const auto& x) {
                                    return type_convert<BiasGradDataType>(x * rp_undrop);
                                },
                                dp_acc);
                        }
                        else
                        {
                            return cast_tile<BiasGradDataType>(dp_acc);
                        }
                    }();
                    store_tile(bias_lds_write_window, dbias);
                    block_sync_lds();
                    auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
                    auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                        Policy::template MakeBiasTileDistribution<Problem>());
                    shuffle_tile(dbias_tile, shuffled_dbias_tile);
                    store_tile(dbias_dram_window, dbias_tile);
                    move_tile_window(dbias_dram_window, {kM0, 0});
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
            else
            {
                if constexpr(kDelayDVOnlyQDoPrefetch)
                {
                    q_block_tile = load_tile(q_dram_window);
                    move_tile_window(q_dram_window, {kM0, 0});

                    lse_block_tile = load_tile(lse_dram_window);
                    move_tile_window(lse_dram_window, {kM0});

                    do_block_tile = load_tile(do_dram_window);
                    move_tile_window(do_dram_window, {kM0, 0});
                }

                block_sync_lds();
                store_tile(q_lds_window, q_block_tile);
                store_tile(lse_lds_write_window, lse_block_tile);
                shuffle_tile(shuffled_do_block_tile, do_block_tile);
                store_tile(shuffled_do_lds_write_window, shuffled_do_block_tile);
                HotLoopScheduler::template GemmStagedScheduler<2>();
                __builtin_amdgcn_sched_barrier(0);
            }

            block_sync_lds();

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
            if constexpr(kDebugDKOnlyCarryLDSRegs)
            {
                carried_q_reg_tensor  = load_tile(q_lds_read_window);
                carried_lse           = load_tile(lse_lds_read_window);
                carried_do_reg_tensor = load_tile(do_lds_read_window);
                carried_d             = load_tile(d_lds_read_window);
                carried_qt_reg_tensor = load_tile(qt_lds_read_window);
            }
#endif

            i_total_loops += 1;
            seqlen_q_step += kM0;
        }
        __builtin_amdgcn_sched_barrier(0);

        // Tail
        auto s_acc = SPBlockTileType{};
        const index_t q_step = seqlen_q_step;

        // STAGE 1, Q@K Gemm0
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
        if constexpr(kDebugDKOnlyCarryLDSRegs)
        {
            s_acc = gemm_0(carried_q_reg_tensor, k_reg_tensor);
        }
        else
#endif
        {
            auto q_reg_tensor = load_tile(q_lds_read_window);
            s_acc = gemm_0(q_reg_tensor, k_reg_tensor);
        }

        // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            auto bias_dram_window =
                make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                                 bias_dram_block_window_tmp.get_window_lengths(),
                                 {q_step, bias_origin_col0},
                                 Policy::template MakeBiasTileDistribution<Problem>());
            auto* bias_lds_ptr = reinterpret_cast<BiasDataType*>(
                static_cast<char*>(smem_ptr) + kBiasOffset);
            auto bias_lds = make_tensor_view<address_space_enum::lds>(
                bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());
            auto bias_lds_write_window =
                make_tile_window(bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});
            auto bias_s_lds_read_window =
                make_tile_window(bias_lds_write_window.get_bottom_tensor_view(),
                                 bias_lds_write_window.get_window_lengths(),
                                 bias_lds_write_window.get_window_origin(),
                                 Policy::template MakeBiasSTileDistribution<decltype(gemm_0)>());
            const auto bias_tile    = load_tile(bias_dram_window);
            auto shuffled_bias_tile = make_static_distributed_tensor<BiasDataType>(
                Policy::template MakeShuffledBiasTileDistribution<Problem>());
            shuffle_tile(shuffled_bias_tile, bias_tile);
            // SGrad and Bias use the same address in LDS, finish loading ds in the hot loop to
            // reuse LDS.
            block_sync_lds();
            store_tile(bias_lds_write_window, shuffled_bias_tile);
            block_sync_lds();
            auto bias_s_tile = load_tile(bias_s_lds_read_window);
            tile_elementwise_inout(
                [&](auto& x, const auto& y) {
                    x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                },
                s_acc,
                bias_s_tile);
            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
            sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                    const auto row         = q_step + tile_idx.at(number<0>{});
                    const auto col         = k_origin_col0 + tile_idx.at(number<1>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    s_acc(i_j_idx) *= scale;
                    position_encoding.update(s_acc(i_j_idx), row, col);
                });
            });
        }

        {
            bool need_perpixel_check = [&]() {
                if constexpr(kSkipDKOnlyNonMaskEdgeCheck || kSkipDVOnlyNonMaskEdgeCheck)
                {
                    return false;
                }
                else
                {
                    return mask.IsEdgeTile(
                        q_step, k_origin_col0, number<kM0>{}, number<kN0>{});
                }
            }();
            if(need_perpixel_check)
            {
                set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                    const auto row = q_step + tile_idx.at(number<0>{});
                    const auto col = k_origin_col0 + tile_idx.at(number<1>{});
                    return mask.IsOutOfBound(row, col);
                });
            }
        }

        static const auto get_validated_lse = [](LSEDataType raw_lse) {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         FmhaMask::IsMasking)
            {
                return raw_lse == -numeric<LSEDataType>::infinity() ? type_convert<LSEDataType>(0.f)
                                                                    : raw_lse;
            }
            else
            {
                return raw_lse;
            }
        };

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
        auto lse = [&]() {
            if constexpr(kDebugDKOnlyCarryLDSRegs)
            {
                return carried_lse;
            }
            else
            {
                return load_tile(lse_lds_read_window);
            }
        }();
#else
        auto lse               = load_tile(lse_lds_read_window);
#endif
        decltype(auto) p = [&]() -> decltype(auto) {
            if constexpr(kUseDKOnlyFreshP)
            {
                return SPBlockTileType{};
            }
            else
            {
                return (s_acc);
            }
        }();
        constexpr auto p_spans = remove_cvref_t<decltype(p)>::get_distributed_spans();
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    p(i_j_idx) = exp2(s_acc(i_j_idx) - row_lse);
                }
                else
                {
                    p(i_j_idx) = exp2(scale * s_acc(i_j_idx) - row_lse);
                }
            });
        });

        if constexpr(FmhaDropout::IsDropout)
        {
            dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                q_step, k_origin_col0, p, randval_dram_window);
        }

        HotLoopScheduler::template GemmStagedScheduler<1>();
        __builtin_amdgcn_sched_barrier(0);

        auto debug_dummy_dv_gemm1_tail = [&](const auto& p_tile) {
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_DUMMY_DV_GEMM1
            if constexpr(kDoDK && !kDoDV)
            {
                auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                    Policy::template MakePTRegSliceBlockDescriptor<Problem>());
                const auto p_gemm = [&]() {
                    if constexpr(FmhaDropout::IsDropout)
                    {
                        return tile_elementwise_in(
                            [](const auto& x) {
                                return type_convert<GemmDataType>(x > 0.f ? x : 0.f);
                            },
                            p_tile);
                    }
                    else
                    {
                        return cast_tile<GemmDataType>(p_tile);
                    }
                }();
                Policy::template PTFromGemm0CToGemm1A<Problem, decltype(pt_reg_tensor), decltype(p_gemm)>(
                    pt_reg_tensor, p_gemm, ds_lds_ptr);
                auto dot_reg_tensor = load_tile(dot_lds_read_window);
                auto dv_dummy_acc   = decltype(gemm_1.MakeCBlockTile()){};
                clear_tile(dv_dummy_acc);
                gemm_1(dv_dummy_acc, pt_reg_tensor, dot_reg_tensor);
                constexpr auto dv_dummy_buf_size =
                    remove_cvref_t<decltype(dv_dummy_acc)>::get_thread_buffer_size();
                static_for<0, dv_dummy_buf_size, 1>{}([&](auto i) {
                    auto dv_debug_value =
                        type_convert<float>(dv_dummy_acc.get_thread_buffer().at(i));
                    asm volatile("" : "+v"(dv_debug_value) : : "memory");
                });
            }
#else
            (void)p_tile;
#endif
        };

        if constexpr(kDoDV)
        {
            auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                Policy::template MakePTRegSliceBlockDescriptor<Problem>());
            if constexpr(FmhaDropout::IsDropout || kPNeedsCastToGemm)
            {
                const auto p_gemm = [&]() {
                    if constexpr(FmhaDropout::IsDropout)
                    {
                        return tile_elementwise_in(
                            [](const auto& x) {
                                return type_convert<GemmDataType>(x > 0.f ? x : 0.f);
                            },
                            p);
                    }
                    else
                    {
                        return cast_tile<GemmDataType>(p);
                    }
                }();
                Policy::template PTFromGemm0CToGemm1A<Problem,
                                                      decltype(pt_reg_tensor),
                                                      decltype(p_gemm)>(pt_reg_tensor,
                                                                        p_gemm,
                                                                        ds_lds_ptr);
            }
            else
            {
                Policy::template PTFromGemm0CToGemm1A<Problem,
                                                      decltype(pt_reg_tensor),
                                                      remove_cvref_t<decltype(p)>>(
                    pt_reg_tensor, p, ds_lds_ptr);
            }
            if constexpr(kUseSplitDvGemm1BSmem)
            {
                gemm_1(dv_acc, pt_reg_tensor, dot_lds_read_window);
            }
            else
            {
                auto dot_reg_tensor = load_tile(dot_lds_read_window);
                gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);
            }
        }
        else
        {
            debug_dummy_dv_gemm1_tail(p);
        }
        HotLoopScheduler::template GemmStagedScheduler<2>();
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(kDoDK)
        {
            // STAGE 4, OGrad@V Gemm2
            auto dp_acc        = SPGradBlockTileType{};
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
            if constexpr(kDebugDKOnlyCarryLDSRegs)
            {
                dp_acc = gemm_2(carried_do_reg_tensor, v_reg_tensor);
            }
            else
#endif
            {
                auto do_reg_tensor = load_tile(do_lds_read_window);
                dp_acc = gemm_2(do_reg_tensor, v_reg_tensor);
            }
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
            auto d = [&]() {
                if constexpr(kDebugDKOnlyCarryLDSRegs)
                {
                    return carried_d;
                }
                else
                {
                    return load_tile(d_lds_read_window);
                }
            }();
#else
            auto d             = load_tile(d_lds_read_window);
#endif

            // STAGE 5, P^T(PGrad^T - D)
            if constexpr(kSeparateDKOnlyScratch && !kHasBiasGrad)
            {
                auto make_ds_gemm = [&]() {
                    auto ds_gemm = make_static_distributed_tensor<GemmDataType>(
                        SPGradBlockTileType::get_tile_distribution());
                    constexpr auto ds_spans = decltype(ds_gemm)::get_distributed_spans();
                    if constexpr(FmhaDropout::IsDropout)
                    {
                        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                            constexpr auto i_idx = make_tuple(idx0);
                            sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                bool undrop_flag       = p[i_j_idx] >= 0;
                                ds_gemm(i_j_idx) = type_convert<GemmDataType>(
                                    p[i_j_idx] *
                                    (undrop_flag ? (dp_acc[i_j_idx] - d[i_idx]) : d[i_idx]));
                            });
                        });
                    }
                    else
                    {
                        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                            constexpr auto i_idx = make_tuple(idx0);
                            sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                ds_gemm(i_j_idx) = type_convert<GemmDataType>(
                                    p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]));
                            });
                        });
                    }
                    return ds_gemm;
                };

                auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                    Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
                {
                    // Scope ds_gemm to the remap so tail GEMM3 does not also keep it live.
                    const auto ds_gemm = make_ds_gemm();
                    Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                              decltype(dst_reg_tensor),
                                                              decltype(ds_gemm)>(
                        dst_reg_tensor, ds_gemm, ds_lds_ptr);
                }

#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
                auto qt_reg_tensor = [&]() {
                    if constexpr(kDebugDKOnlyCarryLDSRegs)
                    {
                        return carried_qt_reg_tensor;
                    }
                    else
                    {
                        return load_tile(qt_lds_read_window);
                    }
                }();
#else
                auto qt_reg_tensor = load_tile(qt_lds_read_window);
#endif
                gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
#if defined(CK_TILE_DEBUG_BWD_GEMM3_SCHED_BARRIER)
                __builtin_amdgcn_sched_barrier(0);
#endif
            }
            else
            {
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_FRESH_DS
                auto ds_work = SPGradBlockTileType{};
#else
                auto& ds_work = dp_acc;
#endif
                constexpr auto ds_spans =
                    remove_cvref_t<decltype(ds_work)>::get_distributed_spans();
                if constexpr(FmhaDropout::IsDropout)
                {
                    sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);
                            bool undrop_flag       = p[i_j_idx] >= 0;
                            ds_work(i_j_idx) =
                                p[i_j_idx] *
                                (undrop_flag ? (dp_acc[i_j_idx] - d[i_idx]) : d[i_idx]);
                        });
                    });
                }
                else
                {
                    sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);
                            ds_work(i_j_idx)       = p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]);
                        });
                    });
                }

                if constexpr(kHasBiasGrad && !kSeparateDKOnlyScratch)
                {
                    auto* bias_lds_ptr = reinterpret_cast<BiasDataType*>(
                        static_cast<char*>(smem_ptr) + kBiasOffset);
                    auto bias_lds = make_tensor_view<address_space_enum::lds>(
                        bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());
                    auto bias_lds_write_window =
                        make_tile_window(bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});
                    auto dbias_dram_window =
                        make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                                         dbias_dram_block_window_tmp.get_window_lengths(),
                                         {q_step, dbias_origin_col0});
                    auto dbias_lds_read_window =
                        make_tile_window(bias_lds,
                                         make_tuple(number<kM0>{}, number<kN0>{}),
                                         {0, 0},
                                         Policy::template MakeShuffledBiasTileDistribution<Problem>());
                    const auto dbias = [&]() {
                        if constexpr(FmhaDropout::IsDropout)
                        {
                            return tile_elementwise_in(
                                [&rp_undrop](const auto& x) {
                                    return type_convert<BiasGradDataType>(x * rp_undrop);
                                },
                                ds_work);
                        }
                        else
                        {
                            return cast_tile<BiasGradDataType>(ds_work);
                        }
                    }();
                    block_sync_lds();
                    store_tile(bias_lds_write_window, dbias);
                    block_sync_lds();
                    auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
                    auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                        Policy::template MakeBiasTileDistribution<Problem>());
                    shuffle_tile(dbias_tile, shuffled_dbias_tile);
                    store_tile(dbias_dram_window, dbias_tile);
                    __builtin_amdgcn_sched_barrier(0);
                }

                // STAGE 6, SGrad^T@Q^T Gemm3
                if constexpr(kSGradNeedsCastToGemm)
                {
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
                    auto qt_reg_tensor = [&]() {
                        if constexpr(kDebugDKOnlyCarryLDSRegs)
                        {
                            return carried_qt_reg_tensor;
                        }
                        else
                        {
                            return load_tile(qt_lds_read_window);
                        }
                    }();
#else
                    auto qt_reg_tensor = load_tile(qt_lds_read_window);
#endif
                    auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                        Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_PRECAST_SGRAD
                    const auto ds_gemm = cast_tile<GemmDataType>(ds_work);
                    Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                              decltype(dst_reg_tensor),
                                                              decltype(ds_gemm)>(
                        dst_reg_tensor, ds_gemm, ds_lds_ptr);
#else
                    Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                              decltype(dst_reg_tensor),
                                                              decltype(ds_work)>(
                        dst_reg_tensor, ds_work, ds_lds_ptr);
#endif
                    gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY
                    const auto ds_gemm_debug = cast_tile<GemmDataType>(ds_work);
                    store_tile(ds_lds_window, ds_gemm_debug);
                    block_sync_lds();
                    auto ds_reg_tensor_debug = load_tile(ds_lds_read_window);
                    constexpr auto ds_debug_buf_size =
                        remove_cvref_t<decltype(ds_reg_tensor_debug)>::get_thread_buffer_size();
                    static_for<0, ds_debug_buf_size, 1>{}([&](auto i) {
                        auto ds_debug_value =
                            type_convert<float>(ds_reg_tensor_debug.get_thread_buffer().at(i));
                        asm volatile("" : "+v"(ds_debug_value) : : "memory");
                    });
#endif
#if defined(CK_TILE_DEBUG_BWD_GEMM3_SCHED_BARRIER)
                    __builtin_amdgcn_sched_barrier(0);
#endif
                    if constexpr(kHasBiasGrad && !kSeparateDKOnlyScratch)
                    {
                        block_sync_lds();
                    }
                }
                else
                {
                    const auto& ds_gemm = ds_work;
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_CARRY_LDS_REGS
                    auto qt_reg_tensor = [&]() {
                        if constexpr(kDebugDKOnlyCarryLDSRegs)
                        {
                            return carried_qt_reg_tensor;
                        }
                        else
                        {
                            return load_tile(qt_lds_read_window);
                        }
                    }();
#else
                    auto qt_reg_tensor = load_tile(qt_lds_read_window);
#endif
                    auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
                        Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
                    Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                              decltype(dst_reg_tensor),
                                                              remove_cvref_t<decltype(ds_gemm)>>(
                        dst_reg_tensor, ds_gemm, ds_lds_ptr);
                    gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
#if CK_TILE_DEBUG_BWD_SPLIT_DKDV_POST_GEMM3_DS_LDS_BOUNDARY
                    const auto ds_gemm_debug = cast_tile<GemmDataType>(ds_work);
                    store_tile(ds_lds_window, ds_gemm_debug);
                    block_sync_lds();
                    auto ds_reg_tensor_debug = load_tile(ds_lds_read_window);
                    constexpr auto ds_debug_buf_size =
                        remove_cvref_t<decltype(ds_reg_tensor_debug)>::get_thread_buffer_size();
                    static_for<0, ds_debug_buf_size, 1>{}([&](auto i) {
                        auto ds_debug_value =
                            type_convert<float>(ds_reg_tensor_debug.get_thread_buffer().at(i));
                        asm volatile("" : "+v"(ds_debug_value) : : "memory");
                    });
#endif
#if defined(CK_TILE_DEBUG_BWD_GEMM3_SCHED_BARRIER)
                    __builtin_amdgcn_sched_barrier(0);
#endif
                    if constexpr(kHasBiasGrad && !kSeparateDKOnlyScratch)
                    {
                        block_sync_lds();
                    }
                }

                debug_dummy_dq_gemm4(ds_work);
            }

            if constexpr(kHasBiasGrad && kSeparateDKOnlyScratch)
            {
                auto* bias_lds_ptr = reinterpret_cast<BiasDataType*>(
                    static_cast<char*>(smem_ptr) + kBiasOffset);
                auto bias_lds = make_tensor_view<address_space_enum::lds>(
                    bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());
                auto bias_lds_write_window =
                    make_tile_window(bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});
                auto dbias_dram_window =
                    make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                                     dbias_dram_block_window_tmp.get_window_lengths(),
                                     {q_step, dbias_origin_col0});
                auto dbias_lds_read_window =
                    make_tile_window(bias_lds,
                                     make_tuple(number<kM0>{}, number<kN0>{}),
                                     {0, 0},
                                     Policy::template MakeShuffledBiasTileDistribution<Problem>());
                const auto dbias = [&]() {
                    if constexpr(FmhaDropout::IsDropout)
                    {
                        return tile_elementwise_in(
                            [&rp_undrop](const auto& x) {
                                return type_convert<BiasGradDataType>(x * rp_undrop);
                            },
                            dp_acc);
                    }
                    else
                    {
                        return cast_tile<BiasGradDataType>(dp_acc);
                    }
                }();
                store_tile(bias_lds_write_window, dbias);
                block_sync_lds();
                auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
                auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                    Policy::template MakeBiasTileDistribution<Problem>());
                shuffle_tile(dbias_tile, shuffled_dbias_tile);
                store_tile(dbias_dram_window, dbias_tile);
                __builtin_amdgcn_sched_barrier(0);
            }
        }

        block_sync_lds();

        if constexpr(kDoDKDV && !kIsFusedFullBwd)
        {
            if constexpr(FmhaDropout::IsDropout)
            {
                if constexpr(kDoDK)
                {
                    tile_elementwise_inout(
                        [&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; }, dk_acc);
                }
                if constexpr(kDoDV)
                {
                    tile_elementwise_inout([&rp_undrop](auto& x) { x = x * rp_undrop; }, dv_acc);
                }
            }
            else
            {
                if constexpr(kDoDK)
                {
                    tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);
                }
            }
        }
        return make_tuple(dk_acc, dv_acc);
    }
};

} // namespace ck_tile
