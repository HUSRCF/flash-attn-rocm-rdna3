// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

#ifndef CK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR_D256
#define CK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR_D256 0
#endif

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdDQQMajorPipelineKRKTRVRIGLP
{
    static constexpr auto is_dq_qmajor_pipeline = true;

    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType          = remove_cvref_t<typename Problem::GemmDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using DDataType             = remove_cvref_t<typename Problem::DDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using OGradDataType         = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType         = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType         = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType         = remove_cvref_t<typename Problem::VGradDataType>;
    using BiasGradDataType      = remove_cvref_t<typename Problem::BiasGradDataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using FmhaDropout           = remove_cvref_t<typename Problem::FmhaDropout>;
    using HotLoopScheduler      = typename Policy::template HotLoopScheduler<Problem>;
    using BlockFmhaShape        = remove_cvref_t<typename Problem::BlockFmhaShape>;

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
    static constexpr bool kDoDQ            = true;
    static constexpr bool kDoDKDV          = false;
    static constexpr bool kIsFusedFullBwd  = false;

    static_assert(!kUseTrLoad, "Q-major split DQ uses the non-trload KR/KTR staging contract.");
    static_assert(!kIsGroupMode, "Q-major split DQ trial supports batch mode only.");
    static_assert(!kIsDeterministic, "Q-major split DQ writes final dQ directly and skips convert_dq.");
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "Q-major split DQ trial does not support bias.");
    static_assert(!kHasBiasGrad, "Q-major split DQ trial does not support dbias.");
    static_assert(!FmhaDropout::IsDropout, "Q-major split DQ trial does not support dropout.");
    static_assert((kQKHeaddim == 64 && kVHeaddim == 64) ||
                      (CK_TILE_DEBUG_BWD_SPLIT_DQ_QMAJOR_D256 != 0 && kQKHeaddim == 256 &&
                       kVHeaddim == 256),
                  "Q-major split DQ trial is restricted to D64, or explicit D256 trial.");
    static_assert(kM0 == kK1 && kM0 == kK3, "Invalid Q-major split DQ GEMM contract.");
    static_assert(kQKHeaddim >= kK0 && kVHeaddim >= kK2,
                  "Invalid Q-major split DQ head-dim contract.");
    static_assert(kN0 % kK4 == 0, "Q-major split DQ requires complete GEMM4 k slices.");

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

    static constexpr const char* name = "kr_ktr_vr_iglp_dq_qmajor";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr index_t kBaseSmemSize    = Policy::template GetSmemSize<Problem>();
        constexpr index_t kDQRemapSmemSize = kM0 * kQKHeaddim * sizeof(AccDataType);
        return kBaseSmemSize > kDQRemapSmemSize ? kBaseSmemSize : kDQRemapSmemSize;
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
              typename QGradEpilogue,
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
               QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               const BiasGradDramBlockWindowTmp& dbias_dram_block_window_tmp,
               const QGradEpilogue& dq_epilogue,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float raw_scale,
               float scale,
               float /*rp_undrop*/,
               float /*scale_rp_undrop*/,
               FmhaDropout& dropout) const
    {
        (void)bias_dram_block_window_tmp;
        (void)randval_dram_block_window_tmp;
        (void)dbias_dram_block_window_tmp;
        (void)position_encoding;
        (void)dropout;

        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();
        const auto i_m0     = q_origin.at(number<0>{});

        const auto [seqlen_kv_start, seqlen_kv_end] =
            mask.GetTileRangeAlongX(i_m0, number<kM0>{}, number<kN0>{});

        const index_t num_total_loop =
            integer_divide_ceil(seqlen_kv_end - seqlen_kv_start, kN0);

        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             q_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeQDramTileDistribution<Problem>());

        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             do_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeOGradDramTileDistribution<Problem>());

        auto lse_dram_window =
            make_tile_window(lse_dram_block_window_tmp.get_bottom_tensor_view(),
                             lse_dram_block_window_tmp.get_window_lengths(),
                             lse_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeLSEDDramTileDistribution<Problem,
                                                                           decltype(gemm_0)>());

        auto d_dram_window =
            make_tile_window(d_dram_block_window_tmp.get_bottom_tensor_view(),
                             d_dram_block_window_tmp.get_window_lengths(),
                             d_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeLSEDDramTileDistribution<Problem,
                                                                           decltype(gemm_0)>());

        KDataType* k_lds_ptr =
            static_cast<KDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        KDataType* kt_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));
        VDataType* v_lds_ptr =
            static_cast<VDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        QDataType* q_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>()));
        OGradDataType* do_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>()));
        LSEDataType* lse_lds_ptr = static_cast<LSEDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>()));
        DDataType* d_lds_ptr = static_cast<DDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>()));
        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto q_lds = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});
        auto q_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());

        auto do_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());
        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});
        auto do_lds_read_window =
            make_tile_window(do_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             do_lds_window.get_window_origin(),
                             Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());

        auto lse_lds = make_tensor_view<address_space_enum::lds>(
            lse_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());
        auto lse_lds_write_window = make_tile_window(lse_lds, make_tuple(number<kM0>{}), {0});
        auto lse_lds_read_window = make_tile_window(
            lse_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        auto d_lds = make_tensor_view<address_space_enum::lds>(
            d_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());
        auto d_lds_write_window = make_tile_window(d_lds, make_tuple(number<kM0>{}), {0});
        auto d_lds_read_window = make_tile_window(
            d_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        auto ds_lds = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());
        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});
        auto ds_lds_read_window =
            make_tile_window(ds_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK4>{}),
                             ds_lds_window.get_window_origin(),
                             Policy::template MakeSGradRegSliceBlockDescriptor<Problem>());
        auto remap_dq_acc_for_store = [&](const auto& dq_tile_like) {
            auto* dq_acc_lds_ptr = reinterpret_cast<AccDataType*>(k_lds_ptr);
            auto dq_acc_lds_window = [&]() {
                constexpr auto dq_acc_lds_desc = make_naive_tensor_descriptor_packed(
                    make_tuple(number<kM0>{}, number<kQKHeaddim>{}));
                auto dq_acc_lds_view = make_tensor_view<address_space_enum::lds>(
                    dq_acc_lds_ptr, dq_acc_lds_desc);
                return make_tile_window(dq_acc_lds_view, dq_acc_lds_desc.get_lengths(), {0, 0});
            }();

            store_tile(dq_acc_lds_window, dq_tile_like);
            block_sync_lds();

            auto dq_acc_result_window = [&]() {
                constexpr auto dq_acc_lds_desc = make_naive_tensor_descriptor_packed(
                    make_tuple(number<1>{}, number<kM0>{}, number<kQKHeaddim>{}));
                auto dq_acc_lds_view = make_tensor_view<address_space_enum::lds>(
                    dq_acc_lds_ptr, dq_acc_lds_desc);
                return make_tile_window(dq_acc_lds_view,
                                        dq_acc_lds_desc.get_lengths(),
                                        {0, 0, 0},
                                        Policy::template MakePostQGradAccDramTileDistribution<
                                            Problem>());
            }();

            auto dq_acc_remapped = load_tile(dq_acc_result_window);
            auto dq_store =
                make_static_distributed_tensor<AccDataType>(
                    Policy::template MakePostQGradDramTileDistribution<Problem>());
            dq_store.get_thread_buffer() = dq_acc_remapped.get_thread_buffer();
            return dq_store;
        };

        const auto q_block_tile   = load_tile(q_dram_window);
        const auto do_block_tile  = load_tile(do_dram_window);
        const auto lse_block_tile = load_tile(lse_dram_window);
        const auto d_block_tile   = load_tile(d_dram_window);

        block_sync_lds();
        store_tile(q_lds_window, q_block_tile);
        store_tile(do_lds_window, do_block_tile);
        store_tile(lse_lds_write_window, lse_block_tile);
        store_tile(d_lds_write_window, d_block_tile);
        block_sync_lds();

        const auto q_reg_tensor  = load_tile(q_lds_read_window);
        const auto do_reg_tensor = load_tile(do_lds_read_window);
        const auto lse           = load_tile(lse_lds_read_window);
        const auto d             = load_tile(d_lds_read_window);

        using SPBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType  = decltype(gemm_4.MakeCBlockTile());

        auto dq_acc = QGradBlockTileType{};
        clear_tile(dq_acc);

        index_t seqlen_kv_step = seqlen_kv_start;
        constexpr index_t k4_loops = kN0 / kK4;

        for(index_t i_loop = 0; i_loop < num_total_loop; ++i_loop)
        {
            auto k_dram_window =
                make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                                 k_dram_block_window_tmp.get_window_lengths(),
                                 {seqlen_kv_step, 0},
                                 Policy::template MakeKDramTileDistribution<Problem>());

            auto v_dram_window =
                make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                                 v_dram_block_window_tmp.get_window_lengths(),
                                 {seqlen_kv_step, 0},
                                 Policy::template MakeVDramTileDistribution<Problem>());

            auto k_lds = make_tensor_view<address_space_enum::lds>(
                k_lds_ptr, Policy::template MakeKLdsWriteBlockDescriptor<Problem>());
            auto k_lds_write_window =
                make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});
            auto k_lds_read_window =
                make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                                 make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                                 k_lds_write_window.get_window_origin(),
                                 Policy::template MakeKRegBlockDescriptor<Problem>());

            auto shuffled_k_lds_write = make_tensor_view<address_space_enum::lds>(
                kt_lds_ptr, Policy::template MakeShuffledKLdsWriteBlockDescriptor<Problem>());
            auto shuffled_k_lds_write_window =
                make_tile_window(shuffled_k_lds_write,
                                 make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                                 {0, 0});
            auto kt_lds_read = make_tensor_view<address_space_enum::lds>(
                kt_lds_ptr, Policy::template MakeKTLdsReadBlockDescriptor<Problem>());
            auto kt_lds_read_window =
                make_tile_window(kt_lds_read,
                                 make_tuple(number<kQKHeaddim>{}, number<kN0>{}),
                                 {0, 0},
                                 Policy::template MakeKTRegBlockDescriptor<Problem>());

            auto v_lds = make_tensor_view<address_space_enum::lds>(
                v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());
            auto v_lds_write_window =
                make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kVHeaddim>{}), {0, 0});
            auto v_lds_read_window =
                make_tile_window(v_lds_write_window.get_bottom_tensor_view(),
                                 make_tuple(number<kN0>{}, number<kVHeaddim>{}),
                                 v_lds_write_window.get_window_origin(),
                                 Policy::template MakeVRegBlockDescriptor<Problem>());

            const auto k_block_tile = load_tile(k_dram_window);
            const auto v_block_tile = load_tile(v_dram_window);
            auto shuffled_k_block_tile = make_static_distributed_tensor<KDataType>(
                Policy::template MakeShuffledKRegWriteBlockDescriptor<Problem>());

            store_tile(k_lds_write_window, k_block_tile);
            shuffle_tile(shuffled_k_block_tile, k_block_tile);
            store_tile(shuffled_k_lds_write_window, shuffled_k_block_tile);
            block_sync_lds();
            const auto k_reg_tensor = load_tile(k_lds_read_window);
            block_sync_lds();
            const auto kt_reg_tensor = load_tile(kt_lds_read_window);

            store_tile(v_lds_write_window, v_block_tile);
            block_sync_lds();
            const auto v_reg_tensor = load_tile(v_lds_read_window);

            auto s_acc = SPBlockTileType{};
            s_acc      = gemm_0(q_reg_tensor, k_reg_tensor);

            if(mask.IsEdgeTile(i_m0, seqlen_kv_step, number<kM0>{}, number<kN0>{}))
            {
                set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                    const auto row = i_m0 + tile_idx.at(number<0>{});
                    const auto col = seqlen_kv_step + tile_idx.at(number<1>{});
                    return mask.IsOutOfBound(row, col);
                });
            }

            auto p                 = SPBlockTileType{};
            constexpr auto p_spans = decltype(p)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto raw_lse   = lse[i_idx];
                const auto row_lse =
                    log2e_v<LSEDataType> *
                    (FmhaMask::IsMasking && raw_lse == -numeric<LSEDataType>::infinity()
                         ? type_convert<LSEDataType>(0.f)
                         : raw_lse);

                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p(i_j_idx)             = exp2(scale * s_acc[i_j_idx] - row_lse);
                });
            });

            auto dp_acc = SPGradBlockTileType{};
            dp_acc      = gemm_2(do_reg_tensor, v_reg_tensor);

            auto ds                 = SPGradBlockTileType{};
            constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
            sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ds(i_j_idx)            = p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]);
                });
            });

            const auto ds_gemm = cast_tile<GemmDataType>(ds);
            block_sync_lds();
            store_tile(ds_lds_window, ds_gemm);
            block_sync_lds();

            auto ds_reg_tensor      = load_tile(ds_lds_read_window);
            auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
            move_tile_window(ds_lds_read_window, {0, kK4});

            static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor_next = load_tile(ds_lds_read_window);
                    move_tile_window(ds_lds_read_window, {0, kK4});
                }
                auto kt_reg_tensor_slice =
                    get_slice_tile(kt_reg_tensor,
                                   sequence<0, i_k4 * kK4>{},
                                   sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});
                gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice);
                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
                }
            });
            move_tile_window(ds_lds_read_window, {0, -kN0});

            seqlen_kv_step += kN0;
        }

        tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
        const auto dq_store_tile = remap_dq_acc_for_store(dq_acc);
        dq_epilogue(dq_dram_block_window_tmp, dq_store_tile, nullptr);
    }
};

template <typename, typename = void>
struct fmha_bwd_dq_qmajor_pipeline : std::false_type
{
};

template <typename T>
struct fmha_bwd_dq_qmajor_pipeline<T, std::void_t<decltype(T::is_dq_qmajor_pipeline)>>
    : std::bool_constant<T::is_dq_qmajor_pipeline>
{
};

} // namespace ck_tile
