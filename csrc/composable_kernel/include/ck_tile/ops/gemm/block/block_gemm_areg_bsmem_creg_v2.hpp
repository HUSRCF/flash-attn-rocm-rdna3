// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_default_policy.hpp"

#ifndef CK_TILE_DEBUG_DUMP_A_TMP
#define CK_TILE_DEBUG_DUMP_A_TMP 0
#endif

namespace ck_tile {

// A is block distributed tensor
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmARegBSmemCRegV2DefaultPolicy>
struct BlockGemmARegBSmemCRegV2
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
                std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
            "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iNWarp = get_warp_id() % NWarp;

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
        // distribution
        auto a_block_tensor = make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(
            MakeABlockTileDistribution());

        constexpr auto dst_spans = decltype(a_block_tensor)::get_distributed_spans();

        sweep_tile_span(dst_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(dst_spans[number<1>{}], [&](auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                a_block_tensor(idx) = a_block_tensor_tmp[idx];
            });
        });

#if CK_TILE_DEBUG_DUMP_A_TMP && defined(__HIP_DEVICE_COMPILE__)
        if(blockIdx.x == 0 && threadIdx.x <1)
        {
            const auto& tmp_buf      = a_block_tensor_tmp.get_thread_buffer();
            const auto& dst_buf      = a_block_tensor.get_thread_buffer();
            constexpr index_t tmp_sz = remove_cvref_t<decltype(tmp_buf)>::size();
            constexpr index_t dst_sz = remove_cvref_t<decltype(dst_buf)>::size();

            printf("[A_DUMP] b=(%d,%d,%d) t=%d M=%d N=%d K=%d tmp_sz=%d dst_sz=%d\n",
                   static_cast<int>(blockIdx.x),
                   static_cast<int>(blockIdx.y),
                   static_cast<int>(blockIdx.z),
                   static_cast<int>(threadIdx.x),
                   static_cast<int>(MPerBlock),
                   static_cast<int>(NPerBlock),
                   static_cast<int>(KPerBlock),
                   static_cast<int>(tmp_sz),
                   static_cast<int>(dst_sz));

            for(index_t i = 0; i < tmp_sz; ++i)
            {
                printf("[A_TMP] b=(%d,%d,%d) t=%d i=%d v=%f\n",
                       static_cast<int>(blockIdx.x),
                       static_cast<int>(blockIdx.y),
                       static_cast<int>(blockIdx.z),
                       static_cast<int>(threadIdx.x),
                       static_cast<int>(i),
                       type_convert<float>(tmp_buf[i]));
            }

            for(index_t i = 0; i < dst_sz; ++i)
            {
                printf("[A_DST] b=(%d,%d,%d) t=%d i=%d v=%f\n",
                       static_cast<int>(blockIdx.x),
                       static_cast<int>(blockIdx.y),
                       static_cast<int>(blockIdx.z),
                       static_cast<int>(threadIdx.x),
                       static_cast<int>(i),
                       type_convert<float>(dst_buf[i]));
            }
        }
#endif

        // construct B-warp-window
        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_block_window_tmp.get_window_origin() + multi_index<2>{iNWarp * WG::kN, 0},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

#if 0 // FIXME: using array will cause register spill
        array<array<decltype(b_warp_window_tmp), KIterPerWarp>, NIterPerWarp> b_warp_windows{
            {b_warp_window_tmp}};

        for(index_t nIter = 0; nIter < NIterPerWarp; nIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        // check C-block-distribution
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "wrong!");

        using AWarpDstr = typename WG::AWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                // read B warp tensor from B Block window
                const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

#if CK_TILE_DEBUG_DUMP_A_TMP && defined(__HIP_DEVICE_COMPILE__)
                    if constexpr(kIter == number<0>{} && nIter == number<0>{} &&
                                 mIter == number<0>{})
                    {
                        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2 &&
                           (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 16 ||
                            threadIdx.x == 17))
                        {
                            constexpr index_t kDbgLimitA = 16;
                            constexpr index_t kDbgLimitB = 16;
                            constexpr index_t kDbgLimitC = 8;

                            const auto& a_warp_buf      = a_warp_tensor.get_thread_buffer();
                            const auto& b_warp_buf      = b_warp_tensor.get_thread_buffer();
                            const auto& c_warp_pre_buf  = c_warp_tensor.get_thread_buffer();
                            constexpr index_t a_warp_sz = remove_cvref_t<decltype(a_warp_buf)>::size();
                            constexpr index_t b_warp_sz = remove_cvref_t<decltype(b_warp_buf)>::size();
                            constexpr index_t c_warp_sz = remove_cvref_t<decltype(c_warp_pre_buf)>::size();

                            printf("[WG_PRE] b=(%d,%d,%d) t=%d M=%d N=%d K=%d kIter=%d nIter=%d "
                                   "mIter=%d a_sz=%d b_sz=%d c_sz=%d\n",
                                   static_cast<int>(blockIdx.x),
                                   static_cast<int>(blockIdx.y),
                                   static_cast<int>(blockIdx.z),
                                   static_cast<int>(threadIdx.x),
                                   static_cast<int>(MPerBlock),
                                   static_cast<int>(NPerBlock),
                                   static_cast<int>(KPerBlock),
                                   static_cast<int>(decltype(kIter)::value),
                                   static_cast<int>(decltype(nIter)::value),
                                   static_cast<int>(decltype(mIter)::value),
                                   static_cast<int>(a_warp_sz),
                                   static_cast<int>(b_warp_sz),
                                   static_cast<int>(c_warp_sz));

                            for(index_t i = 0;
                                i < (a_warp_sz < kDbgLimitA ? a_warp_sz : kDbgLimitA);
                                ++i)
                            {
                                printf("[WG_A] b=(%d,%d,%d) t=%d i=%d v=%f\n",
                                       static_cast<int>(blockIdx.x),
                                       static_cast<int>(blockIdx.y),
                                       static_cast<int>(blockIdx.z),
                                       static_cast<int>(threadIdx.x),
                                       static_cast<int>(i),
                                       type_convert<float>(a_warp_buf[i]));
                            }

                            for(index_t i = 0;
                                i < (b_warp_sz < kDbgLimitB ? b_warp_sz : kDbgLimitB);
                                ++i)
                            {
                                printf("[WG_B] b=(%d,%d,%d) t=%d i=%d v=%f\n",
                                       static_cast<int>(blockIdx.x),
                                       static_cast<int>(blockIdx.y),
                                       static_cast<int>(blockIdx.z),
                                       static_cast<int>(threadIdx.x),
                                       static_cast<int>(i),
                                       type_convert<float>(b_warp_buf[i]));
                            }

                            for(index_t i = 0;
                                i < (c_warp_sz < kDbgLimitC ? c_warp_sz : kDbgLimitC);
                                ++i)
                            {
                                printf("[WG_C_PRE] b=(%d,%d,%d) t=%d i=%d v=%f\n",
                                       static_cast<int>(blockIdx.x),
                                       static_cast<int>(blockIdx.y),
                                       static_cast<int>(blockIdx.z),
                                       static_cast<int>(threadIdx.x),
                                       static_cast<int>(i),
                                       type_convert<float>(c_warp_pre_buf[i]));
                            }
                        }
                    }
#endif

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    // WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_array[nIter]);

#if CK_TILE_DEBUG_DUMP_A_TMP && defined(__HIP_DEVICE_COMPILE__)
                    if constexpr(kIter == number<0>{} && nIter == number<0>{} &&
                                 mIter == number<0>{})
                    {
                        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2 &&
                           (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 16 ||
                            threadIdx.x == 17))
                        {
                            constexpr index_t kDbgLimit = 8;
                            const auto& c_warp_post_buf = c_warp_tensor.get_thread_buffer();
                            constexpr index_t c_warp_sz = remove_cvref_t<decltype(c_warp_post_buf)>::size();

                            printf("[WG_C_POST] b=(%d,%d,%d) t=%d M=%d N=%d K=%d kIter=%d nIter=%d "
                                   "mIter=%d c_sz=%d\n",
                                   static_cast<int>(blockIdx.x),
                                   static_cast<int>(blockIdx.y),
                                   static_cast<int>(blockIdx.z),
                                   static_cast<int>(threadIdx.x),
                                   static_cast<int>(MPerBlock),
                                   static_cast<int>(NPerBlock),
                                   static_cast<int>(KPerBlock),
                                   static_cast<int>(decltype(kIter)::value),
                                   static_cast<int>(decltype(nIter)::value),
                                   static_cast<int>(decltype(mIter)::value),
                                   static_cast<int>(c_warp_sz));

                            for(index_t i = 0; i < (c_warp_sz < kDbgLimit ? c_warp_sz : kDbgLimit); ++i)
                            {
                                printf("[WG_C] b=(%d,%d,%d) t=%d i=%d v=%f\n",
                                       static_cast<int>(blockIdx.x),
                                       static_cast<int>(blockIdx.y),
                                       static_cast<int>(blockIdx.z),
                                       static_cast<int>(threadIdx.x),
                                       static_cast<int>(i),
                                       type_convert<float>(c_warp_post_buf[i]));
                            }
                        }
                    }
#endif

                    // write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());

#if CK_TILE_DEBUG_DUMP_A_TMP && defined(__HIP_DEVICE_COMPILE__)
                    if constexpr(kIter == number<0>{} && nIter == number<0>{} &&
                                 mIter == number<0>{})
                    {
                        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2 &&
                           (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 16 ||
                            threadIdx.x == 17))
                        {
                            constexpr index_t kDbgLimit = 8;
                            const auto c_block_post_buf = c_block_tensor.get_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                            constexpr index_t c_block_post_sz =
                                remove_cvref_t<decltype(c_block_post_buf)>::size();

                            for(index_t i = 0;
                                i < (c_block_post_sz < kDbgLimit ? c_block_post_sz : kDbgLimit);
                                ++i)
                            {
                                printf("[WG_C_BLK] b=(%d,%d,%d) t=%d i=%d v=%f\n",
                                       static_cast<int>(blockIdx.x),
                                       static_cast<int>(blockIdx.y),
                                       static_cast<int>(blockIdx.z),
                                       static_cast<int>(threadIdx.x),
                                       static_cast<int>(i),
                                       type_convert<float>(c_block_post_buf[i]));
                            }
                        }
                    }
#endif
                });
            });
        });
    }

    template <index_t MPerBlock = BlockGemmShape::kM, index_t KPerBlock = BlockGemmShape::kK>
    CK_TILE_DEVICE static constexpr auto MakeABlockTileDistribution()
    {
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        return make_static_tile_distribution(a_block_dstr_encode);
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        // constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

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
    }

    // C = A * B
    template <typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor, a_block_tensor_tmp, b_block_window_tmp);
        return c_block_tensor;
    }
};

} // namespace ck_tile
