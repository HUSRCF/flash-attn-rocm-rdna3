/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

#include "fmha_bwd.hpp"
#include "mask.hpp"
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) { return false; }
    std::string s(value);
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

int64_t env_int_value(const char* name, int64_t default_value = 0) {
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : static_cast<int64_t>(std::atoll(value));
}

std::pair<at::Tensor, at::Tensor>
make_debug_bwd_output(c10::IntArrayRef shape,
                      const at::TensorOptions& opts,
                      bool poison_outputs,
                      int64_t guard_rows) {
    const bool use_poison = poison_outputs || guard_rows > 0;
    if (guard_rows <= 0) {
        at::Tensor out = use_poison
            ? torch::full(shape, std::numeric_limits<float>::quiet_NaN(), opts)
            : torch::empty(shape, opts);
        return {out, at::Tensor()};
    }

    auto backing_shape = shape.vec();
    TORCH_CHECK(backing_shape.size() >= 2, "guarded CK BWD output expects rank >= 2");
    backing_shape[1] += 2 * guard_rows;
    at::Tensor backing =
        torch::full(backing_shape, std::numeric_limits<float>::quiet_NaN(), opts);
    at::Tensor view = backing.narrow(1, guard_rows, shape[1]);
    return {view, backing};
}

bool tensor_has_nan(const at::Tensor& tensor) {
    return tensor.defined() && tensor.numel() > 0 && torch::isnan(tensor).any().item<bool>();
}

bool guard_rows_clean(const at::Tensor& backing, int64_t guard_rows) {
    if (!backing.defined() || guard_rows <= 0) { return true; }
    auto prefix = backing.narrow(1, 0, guard_rows);
    auto suffix = backing.narrow(1, backing.size(1) - guard_rows, guard_rows);
    return torch::isnan(prefix).all().item<bool>() && torch::isnan(suffix).all().item<bool>();
}

bool guard_last_dim_clean(const at::Tensor& backing, int64_t guard_rows) {
    if (!backing.defined() || guard_rows <= 0) { return true; }
    const int64_t dim = backing.dim() - 1;
    auto prefix = backing.narrow(dim, 0, guard_rows);
    auto suffix = backing.narrow(dim, backing.size(dim) - guard_rows, guard_rows);
    return torch::isnan(prefix).all().item<bool>() && torch::isnan(suffix).all().item<bool>();
}

void dump_dk_host_stats(const at::Tensor& dk,
                        int64_t batch_size,
                        int64_t seqlen_k,
                        int64_t num_heads_k,
                        int64_t head_size) {
    if (!dk.defined() || dk.numel() == 0) { return; }

    at::Tensor dk_f32 = dk.to(at::kFloat);
    at::Tensor abs_flat = dk_f32.abs().reshape({-1});
    const float absmax = abs_flat.max().item<float>();
    const float sum = dk_f32.sum().item<float>();
    const int64_t max_idx = abs_flat.argmax().item<int64_t>();

    int64_t rem = max_idx;
    const int64_t d = rem % head_size;
    rem /= head_size;
    const int64_t h = rem % num_heads_k;
    rem /= num_heads_k;
    const int64_t s = rem % seqlen_k;
    rem /= seqlen_k;
    const int64_t b = rem;

    const float signed_max = dk_f32.select(0, b)
                                 .select(0, s)
                                 .select(0, h)
                                 .select(0, d)
                                 .item<float>();

}

struct DebugLSEDPair {
    at::Tensor lse;
    at::Tensor d;
    at::Tensor lse_backing;
    at::Tensor d_backing;
};

DebugLSEDPair make_debug_lsed_pair(const at::Tensor& softmax_lse,
                                   bool poison_outputs,
                                   int64_t guard_rows) {
    const bool use_poison = poison_outputs || guard_rows > 0;
    if (guard_rows <= 0) {
        at::Tensor d = use_poison
            ? torch::full_like(softmax_lse, std::numeric_limits<float>::quiet_NaN())
            : torch::empty_like(softmax_lse);
        return {softmax_lse, d, at::Tensor(), at::Tensor()};
    }

    auto backing_shape = softmax_lse.sizes().vec();
    TORCH_CHECK(!backing_shape.empty(), "guarded CK BWD LSE/D expects rank >= 1");
    backing_shape.back() += 2 * guard_rows;

    at::Tensor lse_backing =
        torch::full(backing_shape, std::numeric_limits<float>::quiet_NaN(), softmax_lse.options());
    at::Tensor d_backing =
        torch::full(backing_shape, std::numeric_limits<float>::quiet_NaN(), softmax_lse.options());

    const int64_t dim = static_cast<int64_t>(backing_shape.size() - 1);
    at::Tensor lse = lse_backing.narrow(dim, guard_rows, softmax_lse.size(-1));
    at::Tensor d   = d_backing.narrow(dim, guard_rows, softmax_lse.size(-1));
    lse.copy_(softmax_lse);
    return {lse, d, lse_backing, d_backing};
}

} // namespace

fmha_bwd_traits get_ck_fmha_bwd_traits(const mask_info &mask,
                                       std::string dtype,
                                       int head_size,
                                       bool has_dropout,
                                       bool enable_alibi,
                                       bool deterministic)
{
    return fmha_bwd_traits{head_size,
                           head_size,
                           dtype,
                           false, // is_group_mode
                           mask.type,
                           enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
                           false,    // has_dbias
                           has_dropout,
                           false, // s_randval
                           deterministic};
}

fmha_bwd_args get_ck_fmha_bwd_args(const mask_info &mask,
                                   // sizes
                                   const int b,
                                   const int seqlen_q,
                                   const int seqlen_k,
                                   const int h,
                                   const int h_k,
                                   const int hdim,
                                   // device pointers
                                   const at::Tensor q,
                                   const at::Tensor k,
                                   const at::Tensor v,
                                   std::optional<at::Tensor> &alibi_slopes_,
                                   const at::Tensor out,
                                   const at::Tensor softmax_lse,
                                   const at::Tensor dout,
                                   at::Tensor dq_acc,
                                   at::Tensor d,
                                   at::Tensor dq,
                                   at::Tensor dk,
                                   at::Tensor dv,
                                   float softmax_scale,
                                   float p_dropout,
                                   std::pair<uint64_t*, uint64_t*> drop_seed_offset)
{
    // q: (batch_size, seqlen_q, nheads, hdim)
    ck_tile::index_t batch_stride_q = q.stride(0);
    ck_tile::index_t stride_q = q.stride(1);
    ck_tile::index_t nhead_stride_q = q.stride(2);

    // k: (batch_size, seqlen_k, nheads_k, hdim)
    ck_tile::index_t batch_stride_k = k.stride(0);
    ck_tile::index_t stride_k = k.stride(1);
    ck_tile::index_t nhead_stride_k = k.stride(2);

    // v: (batch_size, seqlen_k, nheads_k, hdim)
    ck_tile::index_t batch_stride_v = v.stride(0);
    ck_tile::index_t stride_v = v.stride(1);
    ck_tile::index_t nhead_stride_v = v.stride(2);

    // o: (batch_size, seqlen_q, nheads, hdim)
    ck_tile::index_t batch_stride_o = out.stride(0);
    ck_tile::index_t stride_o = out.stride(1);
    ck_tile::index_t nhead_stride_o = out.stride(2);

    // lse: (batch_size, nheads, seqlen_q)
    ck_tile::index_t batch_stride_lse = softmax_lse.stride(0);
    ck_tile::index_t nhead_stride_lse = softmax_lse.stride(1);

    // do: (batch_size, seqlen_q, nheads, hdim)
    ck_tile::index_t batch_stride_do = dout.stride(0);
    ck_tile::index_t stride_do = dout.stride(1);
    ck_tile::index_t nhead_stride_do = dout.stride(2);

    // d: (batch_size, nheads, seqlen_q)
    // CK assume d share the same stride with lse

    // dq: (batch_size, seqlen_q, nheads, hdim)
    ck_tile::index_t batch_stride_dq = dq.stride(0);
    ck_tile::index_t stride_dq = dq.stride(1);
    ck_tile::index_t nhead_stride_dq = dq.stride(2);

    // dk_expanded: (batch_size, seqlen_k, nheads, hdim)
    ck_tile::index_t batch_stride_dk = dk.stride(0);
    ck_tile::index_t stride_dk = dk.stride(1);
    ck_tile::index_t nhead_stride_dk = dk.stride(2);

    // dv_expanded: (batch_size, seqlen_k, nheads, hdim)
    ck_tile::index_t batch_stride_dv = dv.stride(0);
    ck_tile::index_t stride_dv = dv.stride(1);
    ck_tile::index_t nhead_stride_dv = dv.stride(2);

    // dq_acc: (split, batch_size, seqlen_q, nheads, hdim)
    ck_tile::index_t split_stride_dq_acc = dq_acc.stride(0);
    ck_tile::index_t batch_stride_dq_acc = dq_acc.stride(1);
    ck_tile::index_t stride_dq_acc = dq_acc.stride(2);
    ck_tile::index_t nhead_stride_dq_acc = dq_acc.stride(3);

    float p_undrop = 1.0 - p_dropout;

    void *alibi_slopes_ptr = nullptr;
    ck_tile::index_t stride_alibi_slopes = 0;

    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) || alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        alibi_slopes_ptr = alibi_slopes.data_ptr();
        // alibi_slopes:(batch_size, nheads) or (nhead)
        stride_alibi_slopes = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    return fmha_bwd_args{q.data_ptr(),
                         k.data_ptr(),
                         v.data_ptr(),
                         alibi_slopes_ptr, // bias
                         out.data_ptr(),
                         softmax_lse.data_ptr(),
                         dout.data_ptr(),
                         d.data_ptr(),
                         nullptr, // rand_val
                         dq.data_ptr(),
                         dk.data_ptr(),
                         dv.data_ptr(),
                         nullptr, // dbias
                         dq_acc.data_ptr(), // dq_acc
                         nullptr, // seqstart_q_ptr
                         nullptr, // seqstart_k_ptr
                         nullptr, // seqlen_q_ptr
                         nullptr, // seqlen_k_ptr
                         nullptr, // cu_seqlen_q_ptr
                         nullptr, // cu_seqlen_k_ptr
                         seqlen_q,
                         seqlen_k,
                         b,
                         seqlen_q, // max_seqlen_q
                         seqlen_k, // max_seqlen_k
                         hdim, // hdim_q
                         hdim, // hdim_v
                         h, // nhead
                         h_k, // nhead_k
                         softmax_scale,
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_alibi_slopes,
                         stride_o,
                         0, // stride_randval
                         stride_do,
                         stride_dq_acc,
                         stride_dq,
                         stride_dk,
                         stride_dv,
                         0, // stride_dbias, FA without bias
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         0, // nhead_stride_bias, FA without bias
                         nhead_stride_o,
                         0, // nhead_stride_randval
                         nhead_stride_do,
                         nhead_stride_lse,
                         nhead_stride_dq_acc,
                         nhead_stride_dq,
                         nhead_stride_dk,
                         nhead_stride_dv,
                         0, // nhead_stride_dbias, FA without dbias
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         0  , // batch_stride_bias, FA without bias
                         batch_stride_o,
                         0, // batch_stride_randval
                         batch_stride_do,
                         batch_stride_lse,
                         batch_stride_dq_acc,
                         batch_stride_dq,
                         batch_stride_dk,
                         batch_stride_dv,
                         0  , // batch_stride_dbias, FA without dbias
                         split_stride_dq_acc,
                         mask.left,
                         mask.right,
                         static_cast<ck_tile::index_t>(mask.type),
                         p_dropout,
                         p_undrop,
                         drop_seed_offset};
}

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,                   // batch_size x seqlen_q x num_heads, x multiple_of(head_size, 8)
        const at::Tensor &q,                      // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,                    // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,            // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,           // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,           // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,           // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,                    // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float /*softcap*/,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> &rng_state_)
{
#ifdef FLASHATTENTION_DISABLE_BACKWARD
    TORCH_CHECK(false, "This flash attention build does not support backward.");
#endif
    if (is_causal) { window_size_right = 0; }

    bool is_dropout = p_dropout > 0.0;
#ifdef HIPIFY_V2
    auto stream = at::cuda::getCurrentCUDAStream().stream();
#else
    auto stream = at::cuda::getCurrentHIPStream().stream();
#endif

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "CK FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    mask_info mask;
    if (is_causal) {
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
    }
    else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
    }
    else {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // local
    }

    // q, k, v, out had been padded in mha_fwd
    // dq_, dk_, dv_ are also padded tensor
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);

    const bool debug_poison_outputs = env_flag_enabled("FLASH_ATTN_CK_BWD_POISON_OUTPUTS");
    const int64_t debug_guard_rows = env_int_value("FLASH_ATTN_CK_BWD_GUARD_ROWS", 0);

    at::Tensor dq, dk, dv;
    at::Tensor dk_backing, dv_backing;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = debug_poison_outputs
            ? torch::full_like(q, std::numeric_limits<float>::quiet_NaN())
            : torch::empty_like(q);
    }
    if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    CHECK_DEVICE(dk);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        if (num_heads_k == num_heads) {
            auto dk_debug = make_debug_bwd_output(
                k.sizes(), k.options(), debug_poison_outputs, debug_guard_rows);
            dk = dk_debug.first;
            dk_backing = dk_debug.second;
        } else {
            dk = debug_poison_outputs
                ? torch::full_like(k, std::numeric_limits<float>::quiet_NaN())
                : torch::empty_like(k);
        }
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        if (num_heads_k == num_heads) {
            auto dv_debug = make_debug_bwd_output(
                v.sizes(), v.options(), debug_poison_outputs, debug_guard_rows);
            dv = dv_debug.first;
            dv_backing = dv_debug.second;
        } else {
            dv = debug_poison_outputs
                ? torch::full_like(v, std::numeric_limits<float>::quiet_NaN())
                : torch::empty_like(v);
        }
    }

    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();
    auto lsed_debug = make_debug_lsed_pair(softmax_lse, debug_poison_outputs, debug_guard_rows);
    auto softmax_lse_kernel = lsed_debug.lse;
    auto softmax_d = lsed_debug.d;
    at::Tensor dq_accum;

    if (!deterministic) {
        dq_accum = torch::zeros({1, batch_size, seqlen_q, num_heads, head_size}, opts.dtype(at::kFloat));
    } else {
        const ck_tile::index_t kN0 = flash::get_ck_deterministic_bwd_kn0(head_size);
        const ck_tile::index_t nsplits = ck_tile::integer_divide_ceil(seqlen_k, kN0);
        dq_accum = torch::zeros({nsplits, batch_size, seqlen_q, num_heads, head_size}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded;
    at::Tensor dk_expanded_backing, dv_expanded_backing;
    if (num_heads_k != num_heads) {  // MQA / GQA
        auto dk_debug = make_debug_bwd_output(
            {batch_size, seqlen_k, num_heads, head_size},
            opts,
            debug_poison_outputs,
            debug_guard_rows);
        dk_expanded = dk_debug.first;
        dk_expanded_backing = dk_debug.second;
        auto dv_debug = make_debug_bwd_output(
            {batch_size, seqlen_k, num_heads, head_size},
            opts,
            debug_poison_outputs,
            debug_guard_rows);
        dv_expanded = dv_debug.first;
        dv_expanded_backing = dv_debug.second;
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
        dk_expanded_backing = dk_backing;
        dv_expanded_backing = dv_backing;
    }

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    at::Tensor rng_state;

    if (rng_state_.has_value()) {
        rng_state = rng_state_.value();
    } else if(is_dropout) {
        rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        auto philox_args = gen->philox_cuda_state(counter_offset);
        hipLaunchKernelGGL(
            flash::ParsePhiloxCudaState, dim3(1), dim3(64), 0, 0,
            philox_args, reinterpret_cast<uint64_t*>(rng_state.data_ptr()));
    }

    if (seqlen_q > 0) {
        auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
        ck_tile::stream_config stream_config{stream};
        int ck_log_level = 0;
        if(const char* ck_log_level_env = std::getenv("FLASH_ATTN_CK_LOG_LEVEL")) {
            ck_log_level = std::atoi(ck_log_level_env);
            stream_config.log_level_ = ck_log_level;
        }

        auto traits =
            get_ck_fmha_bwd_traits(mask, q_dtype_str, head_size, is_dropout, alibi_slopes_.has_value(), deterministic);

        auto args =
            get_ck_fmha_bwd_args(
                mask,
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                q,
                k,
                v,
                alibi_slopes_,
                out,
                softmax_lse_kernel,
                dout,
                dq_accum,
                softmax_d,
                dq,
                dk_expanded,
                dv_expanded,
                softmax_scale,
                p_dropout,
                drop_seed_offset);

        float t = fmha_bwd(traits, args, stream_config);
        TORCH_CHECK(t >= 0, "invalid argument for fmha_bwd");

        TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
                    "Failed to synchronize CK BWD debug stream");
        dump_dk_host_stats(
            dk_expanded, batch_size, seqlen_k, num_heads, head_size);
        if (debug_poison_outputs || debug_guard_rows > 0) {
            const bool dq_has_nan = tensor_has_nan(dq);
            const bool d_has_nan = tensor_has_nan(softmax_d);
            const bool dk_has_nan = tensor_has_nan(dk_expanded);
            const bool dv_has_nan = tensor_has_nan(dv_expanded);
            const bool lse_guard_corrupt = !guard_last_dim_clean(lsed_debug.lse_backing, debug_guard_rows);
            const bool d_guard_corrupt = !guard_last_dim_clean(lsed_debug.d_backing, debug_guard_rows);
            const bool dk_guard_corrupt = !guard_rows_clean(dk_expanded_backing, debug_guard_rows);
            const bool dv_guard_corrupt = !guard_rows_clean(dv_expanded_backing, debug_guard_rows);
            if (ck_log_level > 0 || dq_has_nan || d_has_nan || dk_has_nan || dv_has_nan ||
                lse_guard_corrupt || d_guard_corrupt || dk_guard_corrupt || dv_guard_corrupt) {
                std::cerr << "[FLASH_ATTN_CK_BWD_DEBUG] "
                          << "B=" << batch_size
                          << " Hq=" << num_heads
                          << " Hk=" << num_heads_k
                          << " D=" << head_size
                          << " Sq=" << seqlen_q
                          << " Sk=" << seqlen_k
                          << " poison=" << static_cast<int>(debug_poison_outputs)
                          << " guard_rows=" << debug_guard_rows
                          << " dq_nan=" << static_cast<int>(dq_has_nan)
                          << " d_nan=" << static_cast<int>(d_has_nan)
                          << " lse_guard_corrupt=" << static_cast<int>(lse_guard_corrupt)
                          << " d_guard_corrupt=" << static_cast<int>(d_guard_corrupt)
                          << " dk_nan=" << static_cast<int>(dk_has_nan)
                          << " dv_nan=" << static_cast<int>(dv_has_nan)
                          << " dk_guard_corrupt=" << static_cast<int>(dk_guard_corrupt)
                          << " dv_guard_corrupt=" << static_cast<int>(dv_guard_corrupt)
                          << std::endl;
            }
        }
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }

    return { dq, dk, dv, softmax_d };
}
