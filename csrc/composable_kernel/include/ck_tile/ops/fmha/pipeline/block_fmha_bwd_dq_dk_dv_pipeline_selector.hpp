// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor.hpp"

namespace ck_tile {

template <typename Problem, typename Policy, bool kDoDQ = true, bool kDoDKDV = true>
class BlockFmhaBwdDQDKDVPipelineSelector
{
    static constexpr bool has_dpad1 =
        Problem::Traits::kPadHeadDimQ == 1 || Problem::Traits::kPadHeadDimV == 1;
    static constexpr bool is_decode = Problem::BlockFmhaShape::kMaxSeqLenQ > 0;

    public:
    using default_type =
        std::conditional_t<Problem::kUseTrLoad,
                           std::conditional_t<is_decode,
                                              BlockFmhaBwdDQDKDVPipelineTrLoadQRQTRDOR<
                                                  Problem,
                                                  BlockFmhaBwdPipelineTrLoadDefaultPolicy,
                                                  kDoDQ,
                                                  kDoDKDV>,
                                              BlockFmhaBwdDQDKDVPipelineTrLoadKRKTRVR<
                                                  Problem,
                                                  BlockFmhaBwdPipelineTrLoadDefaultPolicy,
                                                  kDoDQ,
                                                  kDoDKDV>>,
                           std::conditional_t<has_dpad1,
                                              BlockFmhaBwdDQDKDVPipelineKRKTRVR<
                                                  Problem,
                                                  BlockFmhaBwdPipelineDefaultPolicy,
                                                  kDoDQ,
                                                  kDoDKDV>,
                                              BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP<
                                                  Problem,
                                                  BlockFmhaBwdPipelineDefaultPolicy,
                                                  kDoDQ,
                                                  kDoDKDV>>>;
    using custom_policy_type =
        std::conditional_t<Problem::kUseTrLoad,
                           std::conditional_t<is_decode,
                                              BlockFmhaBwdDQDKDVPipelineTrLoadQRQTRDOR<
                                                  Problem,
                                                  Policy,
                                                  kDoDQ,
                                                  kDoDKDV>,
                                              BlockFmhaBwdDQDKDVPipelineTrLoadKRKTRVR<
                                                  Problem,
                                                  Policy,
                                                  kDoDQ,
                                                  kDoDKDV>>,
                           std::conditional_t<has_dpad1,
                                              BlockFmhaBwdDQDKDVPipelineKRKTRVR<
                                                  Problem,
                                                  Policy,
                                                  kDoDQ,
                                                  kDoDKDV>,
                                              BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP<
                                                  Problem,
                                                  Policy,
                                                  kDoDQ,
                                                  kDoDKDV>>>;
    using type = std::conditional_t<std::is_same_v<Policy, void>, default_type, custom_policy_type>;
};

template <typename Problem, typename Policy = void, bool kDoDQ = true, bool kDoDKDV = true>
class BlockFmhaBwdDQDKDVPipeline
    : public BlockFmhaBwdDQDKDVPipelineSelector<Problem, Policy, kDoDQ, kDoDKDV>::type
{
    public:
    static constexpr const char* name = "auto";
};

} // namespace ck_tile
