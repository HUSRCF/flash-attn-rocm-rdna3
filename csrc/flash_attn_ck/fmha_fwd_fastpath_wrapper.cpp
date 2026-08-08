// SPDX-License-Identifier: BSD-3-Clause

#include "fmha_fwd.hpp"

// Keep the large legacy dispatcher in its original translation unit. GNU ld's
// --wrap redirects callers here, while __real_* resolves to the unmodified
// implementation. This prevents a new fast-path trait from perturbing the
// machine code and latency of existing short-sequence dispatches.
extern float fmha_fwd_real(fmha_fwd_traits,
                           fmha_fwd_args,
                           const ck_tile::stream_config&)
    asm("__real__Z8fmha_fwd15fmha_fwd_traits13fmha_fwd_argsRKN7ck_tile13stream_configE");

float fmha_fwd_wrap(fmha_fwd_traits,
                    fmha_fwd_args,
                    const ck_tile::stream_config&)
    asm("__wrap__Z8fmha_fwd15fmha_fwd_traits13fmha_fwd_argsRKN7ck_tile13stream_configE");

float fmha_fwd_wrap(fmha_fwd_traits traits,
                    fmha_fwd_args args,
                    const ck_tile::stream_config& config)
{
    // The legacy D128 dispatcher uses an eight-wave b128x32 tile after its
    // short-sequence cutoff. On the validated long noncausal domain, the
    // four-wave b64x32 tile exposes finer workgroup-level parallelism while
    // retaining the same QR/register-P/native-O arithmetic path.
    const bool use_gfx11_d128_fp16_b64x32_nc =
        (traits.mask_type == mask_enum::no_mask) and
        (args.seqlen_q >= 2560) and (args.seqlen_q % 64 == 0) and
        (args.seqlen_k > 0) and (args.seqlen_k % 32 == 0) and
        (args.cu_seqlen_k_ptr == nullptr) and
        (static_cast<long long>(args.batch) * static_cast<long long>(args.nhead_q) >= 4) and
        (args.hdim_q == 128) and (args.hdim_v == 128) and
        (args.window_size_left < 0) and (args.window_size_right < 0) and
        (ck_tile::get_device_name().compare(0, 5, "gfx11") == 0) and
        (traits.data_type.compare("fp16") == 0) and (not traits.is_group_mode) and
        traits.is_v_rowmajor and (not traits.has_logits_soft_cap) and
        (traits.bias_type == bias_enum::no_bias) and traits.has_lse and
        (not traits.has_dropout) and
        (traits.qscale_type == quant_scale_enum::no_scale) and
        (not traits.skip_min_seqlen_q);

    if(use_gfx11_d128_fp16_b64x32_nc)
    {
        using trait_ = fmha_fwd_traits_<128,
                                            FmhaFwdFp16,
                                            false,
                                            64,
                                            32,
                                            16,
                                            128,
                                            32,
                                            128,
                                            true,
                                            ck_tile::BlockFmhaPipelineEnum::QRKSVS,
                                            false,
                                            ck_tile::SimplifiedGenericAttentionMask<false>,
                                            ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                            true,
                                            false,
                                            ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                            false,
                                            false,
                                            false,
                                            false,
                                            false,
                                            false>;
        return fmha_fwd_<trait_, ck_tile::gfx11_t>(config, args);
    }

    // Put the K threshold first: short-K fallback pays one integer comparison
    // before entering the byte-for-byte legacy dispatcher.
    const bool use_gfx11_d128_fp16_b128x64_o4 =
        (args.seqlen_k >= 768) and
        (traits.mask_type == mask_enum::mask_top_left or
         traits.mask_type == mask_enum::mask_bottom_right) and
        (args.max_seqlen_q >= 2048) and
        (static_cast<long long>(args.batch) * static_cast<long long>(args.nhead_q) >= 4) and
        (args.hdim_q == 128) and (args.hdim_v == 128) and
        (args.window_size_left < 0) and (args.window_size_right == 0) and
        (ck_tile::get_device_name().compare(0, 5, "gfx11") == 0) and
        (traits.data_type.compare("fp16") == 0) and (not traits.is_group_mode) and
        traits.is_v_rowmajor and (not traits.has_logits_soft_cap) and
        (traits.bias_type == bias_enum::no_bias) and traits.has_lse and
        (not traits.has_dropout) and
        (traits.qscale_type == quant_scale_enum::no_scale) and
        (not traits.skip_min_seqlen_q);

    if(use_gfx11_d128_fp16_b128x64_o4)
    {
        using trait_ = fmha_fwd_traits_<128,
                                            FmhaFwdFp16,
                                            false,
                                            128,
                                            64,
                                            32,
                                            128,
                                            32,
                                            128,
                                            true,
                                            ck_tile::BlockFmhaPipelineEnum::QRKSVS,
                                            false,
                                            ck_tile::SimplifiedGenericAttentionMask<true>,
                                            ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                            true,
                                            false,
                                            ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                            true,
                                            true,
                                            false,
                                            false,
                                            false,
                                            false>;
        return fmha_fwd_<trait_, ck_tile::gfx11_t>(config, args);
    }

    return fmha_fwd_real(traits, args, config);
}
