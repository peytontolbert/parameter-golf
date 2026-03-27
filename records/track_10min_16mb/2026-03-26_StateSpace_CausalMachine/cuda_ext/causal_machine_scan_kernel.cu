#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

constexpr int kNumStates = 128;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kNumStates / kWarpSize;

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr) {
    return static_cast<float>(*ptr);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_from_float(float value) {
    return static_cast<scalar_t>(value);
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(mask, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_max_128(float value, float* shared) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_max(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = (threadIdx.x < kNumWarps) ? shared[lane] : -INFINITY;
    if (warp == 0) {
        value = warp_reduce_max(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_sum_128(float value, float* shared) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = (threadIdx.x < kNumWarps) ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

template <typename scalar_t>
__global__ void causal_machine_forward_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int b = blockIdx.x;
    const int s = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * transition_rank);
    float* stay_shared = dest_shared + (transition_rank * kNumStates);
    float* prev_prob = stay_shared + kNumStates;
    float* latent = prev_prob + kNumStates;
    float* scratch = latent + transition_rank;

    for (int idx = s; idx < kNumStates * transition_rank; idx += blockDim.x) {
        source_shared[idx] = transition_source_probs[idx];
    }
    for (int idx = s; idx < transition_rank * kNumStates; idx += blockDim.x) {
        dest_shared[idx] = transition_dest_probs[idx];
    }
    if (s < kNumStates) {
        stay_shared[s] = transition_stay_probs[s];
        prev_prob[s] = expf(load_as_float(initial_log_belief + (b * kNumStates + s)));
    }
    __syncthreads();

    for (int t = 0; t < chunk_len; ++t) {
        const int pos = chunk_start + t;
        const int base = (b * seq_len + pos) * kNumStates;

        if (s < transition_rank) {
            float latent_val = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < kNumStates; ++i) {
                latent_val += prev_prob[i] * source_shared[i * transition_rank + s];
            }
            latent[s] = latent_val;
        }
        __syncthreads();

        float mix_prob = 0.0f;
        #pragma unroll 4
        for (int r = 0; r < transition_rank; ++r) {
            mix_prob += latent[r] * dest_shared[r * kNumStates + s];
        }
        const float stay_prob = stay_shared[s];
        const float pred_prob = fmaxf(stay_prob * prev_prob[s] + (1.0f - stay_prob) * mix_prob, 1.0e-20f);
        const float pred_log = logf(pred_prob);
        const float obs = load_as_float(local_logits + (base + s)) + transition_gate * (
            pred_log + load_as_float(transition_context + (base + s))
        );

        const float obs_max = block_reduce_max_128(obs, scratch);
        const float obs_exp = expf(obs - obs_max);
        const float obs_sum = block_reduce_sum_128(obs_exp, scratch);
        const float log_norm = logf(fmaxf(obs_sum, 1.0e-20f)) + obs_max;
        const float q_log = obs - log_norm;

        beliefs[base + s] = store_from_float<scalar_t>(q_log);
        prev_prob[s] = expf(q_log);
        __syncthreads();
    }

    final_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(logf(fmaxf(prev_prob[s], 1.0e-20f)));
}

template <typename scalar_t>
__global__ void causal_machine_backward_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_per_batch,
    float* __restrict__ grad_transition_dest_per_batch,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate_per_batch,
    float* __restrict__ grad_transition_stay_per_batch) {
    const int b = blockIdx.x;
    const int s = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * transition_rank);
    float* stay_shared = dest_shared + (transition_rank * kNumStates);
    float* prev_prob = stay_shared + kNumStates;
    float* latent = prev_prob + kNumStates;
    float* mix = latent + transition_rank;
    float* q_prob = mix + kNumStates;
    float* carry = q_prob + kNumStates;
    float* grad_mix = carry + kNumStates;
    float* dlatent = grad_mix + kNumStates;
    float* scratch = dlatent + transition_rank;

    float* grad_source_batch = grad_transition_source_per_batch + b * kNumStates * transition_rank;
    float* grad_dest_batch = grad_transition_dest_per_batch + b * transition_rank * kNumStates;
    float* grad_stay_batch = grad_transition_stay_per_batch + b * kNumStates;

    for (int idx = s; idx < kNumStates * transition_rank; idx += blockDim.x) {
        source_shared[idx] = transition_source_probs[idx];
    }
    for (int idx = s; idx < transition_rank * kNumStates; idx += blockDim.x) {
        dest_shared[idx] = transition_dest_probs[idx];
    }
    if (s < kNumStates) {
        stay_shared[s] = transition_stay_probs[s];
        carry[s] = load_as_float(grad_final_belief + (b * kNumStates + s));
    }
    __syncthreads();

    float gate_grad_accum = 0.0f;

    for (int t = chunk_len - 1; t >= 0; --t) {
        const int pos = chunk_start + t;
        const int base = (b * seq_len + pos) * kNumStates;

        if (pos == 0) {
            prev_prob[s] = expf(load_as_float(initial_log_belief + (b * kNumStates + s)));
        } else {
            prev_prob[s] = expf(load_as_float(beliefs + ((b * seq_len + (pos - 1)) * kNumStates + s)));
        }
        q_prob[s] = expf(load_as_float(beliefs + (base + s)));
        __syncthreads();

        if (s < transition_rank) {
            float latent_val = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < kNumStates; ++i) {
                latent_val += prev_prob[i] * source_shared[i * transition_rank + s];
            }
            latent[s] = latent_val;
        }
        __syncthreads();

        float mix_prob = 0.0f;
        #pragma unroll 4
        for (int r = 0; r < transition_rank; ++r) {
            mix_prob += latent[r] * dest_shared[r * kNumStates + s];
        }
        mix[s] = mix_prob;
        const float stay_prob = stay_shared[s];
        const float pred_prob = fmaxf(stay_prob * prev_prob[s] + (1.0f - stay_prob) * mix_prob, 1.0e-20f);
        const float pred_log = logf(pred_prob);

        const float gq = load_as_float(grad_beliefs + (base + s)) + carry[s];
        const float gq_sum = block_reduce_sum_128(gq, scratch);
        const float ga = gq - q_prob[s] * gq_sum;
        const float grad_pred_log = transition_gate * ga;
        const float grad_pred_prob = grad_pred_log / pred_prob;

        grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
        grad_transition_context[base + s] = store_from_float<scalar_t>(transition_gate * ga);
        grad_mix[s] = grad_pred_prob * (1.0f - stay_prob);
        grad_stay_batch[s] += grad_pred_prob * (prev_prob[s] - mix_prob);
        gate_grad_accum += ga * (pred_log + load_as_float(transition_context + (base + s)));
        const float direct_prev_grad_prob = grad_pred_prob * stay_prob;
        __syncthreads();

        if (s < transition_rank) {
            float dlatent_val = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < kNumStates; ++j) {
                dlatent_val += grad_mix[j] * dest_shared[s * kNumStates + j];
            }
            dlatent[s] = dlatent_val;
            #pragma unroll 4
            for (int j = 0; j < kNumStates; ++j) {
                grad_dest_batch[s * kNumStates + j] += latent[s] * grad_mix[j];
            }
        }
        __syncthreads();

        float prev_grad_prob = direct_prev_grad_prob;
        #pragma unroll 4
        for (int r = 0; r < transition_rank; ++r) {
            prev_grad_prob += dlatent[r] * source_shared[s * transition_rank + r];
            grad_source_batch[s * transition_rank + r] += prev_prob[s] * dlatent[r];
        }
        carry[s] = prev_grad_prob * prev_prob[s];
        __syncthreads();
    }

    grad_initial_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(carry[s]);
    const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
    if (s == 0) {
        grad_transition_gate_per_batch[b] += gate_sum;
    }
}

template <typename scalar_t>
void launch_forward_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int batch_size = static_cast<int>(local_logits.size(0));
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int transition_rank = static_cast<int>(transition_source_probs.size(1));
    const dim3 grid(batch_size);
    const dim3 block(kNumStates);
    const size_t shared_bytes = static_cast<size_t>(
        (2 * kNumStates * transition_rank) + (2 * kNumStates) + transition_rank + kNumWarps
    ) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        causal_machine_forward_chunk_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    causal_machine_forward_chunk_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
        local_logits.data_ptr<scalar_t>(),
        transition_source_probs.data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        beliefs.data_ptr<scalar_t>(),
        final_log_belief.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_per_batch,
    const torch::Tensor& grad_transition_dest_per_batch,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_per_batch,
    const torch::Tensor& grad_transition_stay_per_batch) {
    const int batch_size = static_cast<int>(beliefs.size(0));
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int transition_rank = static_cast<int>(transition_source_probs.size(1));
    const dim3 grid(batch_size);
    const dim3 block(kNumStates);
    const size_t shared_bytes = static_cast<size_t>(
        (2 * kNumStates * transition_rank) + (6 * kNumStates) + (2 * transition_rank) + kNumWarps
    ) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        causal_machine_backward_chunk_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    causal_machine_backward_chunk_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
        grad_beliefs.data_ptr<scalar_t>(),
        grad_final_belief.data_ptr<scalar_t>(),
        transition_source_probs.data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        beliefs.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        grad_local_logits.data_ptr<scalar_t>(),
        grad_transition_source_per_batch.data_ptr<float>(),
        grad_transition_dest_per_batch.data_ptr<float>(),
        grad_transition_context.data_ptr<scalar_t>(),
        grad_initial_log_belief.data_ptr<scalar_t>(),
        grad_transition_gate_per_batch.data_ptr<float>(),
        grad_transition_stay_per_batch.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::vector<torch::Tensor> causal_machine_scan_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto prev = initial_log_belief.contiguous();
    auto final_log_belief = torch::empty_like(initial_log_belief);
    for (int64_t chunk_start = 0; chunk_start < seq_len; chunk_start += chunk_size) {
        const int64_t chunk_len = std::min(chunk_size, seq_len - chunk_start);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            local_logits.scalar_type(),
            "causal_machine_scan_forward_chunk",
            [&] {
                launch_forward_chunk<scalar_t>(
                    local_logits,
                    transition_source_probs,
                    transition_dest_probs,
                    transition_context,
                    prev,
                    transition_gate,
                    transition_stay_probs,
                    chunk_start,
                    chunk_len,
                    beliefs,
                    final_log_belief);
            });
        prev = final_log_belief.contiguous();
    }
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief);
    }
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_backward_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    const auto transition_rank = transition_source_probs.size(1);
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_transition_source_per_batch = torch::zeros(
        {batch_size, kNumStates, transition_rank},
        beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_per_batch = torch::zeros(
        {batch_size, transition_rank, kNumStates},
        beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_per_batch = torch::zeros(
        {batch_size, kNumStates},
        beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_gate_per_batch = torch::zeros(
        {batch_size},
        beliefs.options().dtype(torch::kFloat32));
    auto carry = grad_final_belief.contiguous();
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);

    for (int64_t chunk_end = seq_len; chunk_end > 0; chunk_end -= chunk_size) {
        const int64_t chunk_start = std::max<int64_t>(0, chunk_end - chunk_size);
        const int64_t this_chunk_len = chunk_end - chunk_start;
        auto prev = chunk_start == 0
            ? initial_log_belief.contiguous()
            : beliefs.select(1, chunk_start - 1).contiguous();
        auto chunk_grad_initial = torch::zeros_like(initial_log_belief);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            beliefs.scalar_type(),
            "causal_machine_scan_backward_chunk",
            [&] {
                launch_backward_chunk<scalar_t>(
                    grad_beliefs,
                    carry,
                    transition_source_probs,
                    transition_dest_probs,
                    transition_context,
                    prev,
                    beliefs,
                    transition_gate,
                    transition_stay_probs,
                    chunk_start,
                    this_chunk_len,
                    grad_local_logits,
                    grad_transition_source_per_batch,
                    grad_transition_dest_per_batch,
                    grad_transition_context,
                    chunk_grad_initial,
                    grad_transition_gate_per_batch,
                    grad_transition_stay_per_batch);
            });
        carry = chunk_grad_initial.contiguous();
    }
    if (seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief);
    } else {
        grad_initial_log_belief.copy_(carry);
    }

    auto grad_transition_source_probs = grad_transition_source_per_batch.sum(0);
    auto grad_transition_dest_probs = grad_transition_dest_per_batch.sum(0);
    auto grad_transition_stay_probs = grad_transition_stay_per_batch.sum(0);
    auto grad_transition_gate = grad_transition_gate_per_batch.sum().reshape({1});
    return {
        grad_local_logits,
        grad_transition_source_probs,
        grad_transition_dest_probs,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate,
        grad_transition_stay_probs,
    };
}
