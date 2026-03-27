#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

constexpr int kNumStates = 128;

__device__ float block_reduce_max_128(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = kNumStates / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float block_reduce_sum_128(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = kNumStates / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void causal_machine_forward_chunk_kernel(
    const float* __restrict__ local_logits,
    const float* __restrict__ transition_log_probs,
    const float* __restrict__ transition_context,
    const float* __restrict__ initial_log_belief,
    float transition_gate,
    int seq_len,
    int chunk_start,
    int chunk_len,
    float* __restrict__ beliefs,
    float* __restrict__ final_log_belief) {
    const int b = blockIdx.x;
    const int j = threadIdx.x;

    __shared__ float prev[kNumStates];
    __shared__ float pred[kNumStates];
    __shared__ float scratch[kNumStates];

    prev[j] = initial_log_belief[b * kNumStates + j];
    __syncthreads();

    for (int t = 0; t < chunk_len; ++t) {
        const int pos = chunk_start + t;
        const int base = (b * seq_len + pos) * kNumStates;

        float pred_max = -INFINITY;
        #pragma unroll
        for (int i = 0; i < kNumStates; ++i) {
            pred_max = fmaxf(pred_max, prev[i] + transition_log_probs[i * kNumStates + j]);
        }
        float pred_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < kNumStates; ++i) {
            pred_sum += expf(prev[i] + transition_log_probs[i * kNumStates + j] - pred_max);
        }
        const float pred_j = logf(fmaxf(pred_sum, 1.0e-20f)) + pred_max;
        pred[j] = pred_j;
        const float obs_j = local_logits[base + j] + transition_gate * (pred_j + transition_context[base + j]);

        const float obs_max = block_reduce_max_128(obs_j, scratch);
        const float obs_exp = expf(obs_j - obs_max);
        const float obs_sum = block_reduce_sum_128(obs_exp, scratch);
        const float log_norm = logf(fmaxf(obs_sum, 1.0e-20f)) + obs_max;
        const float q_j = obs_j - log_norm;

        beliefs[base + j] = q_j;
        prev[j] = q_j;
        __syncthreads();
    }

    final_log_belief[b * kNumStates + j] = prev[j];
}

__global__ void causal_machine_backward_chunk_kernel(
    const float* __restrict__ grad_beliefs,
    const float* __restrict__ grad_final_belief,
    const float* __restrict__ local_logits,
    const float* __restrict__ transition_log_probs,
    const float* __restrict__ transition_context,
    const float* __restrict__ initial_log_belief,
    const float* __restrict__ beliefs,
    float transition_gate,
    int seq_len,
    int chunk_start,
    int chunk_len,
    float* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_context,
    float* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_per_batch,
    float* __restrict__ grad_transition_gate_per_batch) {
    const int b = blockIdx.x;
    const int s = threadIdx.x;

    __shared__ float prev[kNumStates];
    __shared__ float pred[kNumStates];
    __shared__ float q[kNumStates];
    __shared__ float carry[kNumStates];
    __shared__ float grad_pred[kNumStates];
    __shared__ float scratch[kNumStates];

    carry[s] = grad_final_belief[b * kNumStates + s];
    __syncthreads();

    float gate_grad_accum = 0.0f;
    float* grad_transition_batch = grad_transition_per_batch + b * kNumStates * kNumStates;

    for (int t = chunk_len - 1; t >= 0; --t) {
        const int pos = chunk_start + t;
        const int base = (b * seq_len + pos) * kNumStates;

        if (pos == 0) {
            prev[s] = initial_log_belief[b * kNumStates + s];
        } else {
            prev[s] = beliefs[(b * seq_len + (pos - 1)) * kNumStates + s];
        }
        q[s] = beliefs[base + s];
        __syncthreads();

        float pred_max = -INFINITY;
        #pragma unroll
        for (int i = 0; i < kNumStates; ++i) {
            pred_max = fmaxf(pred_max, prev[i] + transition_log_probs[i * kNumStates + s]);
        }
        float pred_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < kNumStates; ++i) {
            pred_sum += expf(prev[i] + transition_log_probs[i * kNumStates + s] - pred_max);
        }
        const float pred_j = logf(fmaxf(pred_sum, 1.0e-20f)) + pred_max;
        pred[s] = pred_j;

        const float gq = grad_beliefs[base + s] + carry[s];
        const float gq_sum = block_reduce_sum_128(gq, scratch);
        const float ga = gq - expf(q[s]) * gq_sum;

        grad_local_logits[base + s] = ga;
        grad_transition_context[base + s] = transition_gate * ga;
        grad_pred[s] = transition_gate * ga;
        gate_grad_accum += ga * (pred_j + transition_context[base + s]);
        __syncthreads();

        float prev_grad = 0.0f;
        const float prev_s = prev[s];
        #pragma unroll
        for (int j = 0; j < kNumStates; ++j) {
            const float weight = expf(prev_s + transition_log_probs[s * kNumStates + j] - pred[j]);
            const float contrib = grad_pred[j] * weight;
            grad_transition_batch[s * kNumStates + j] += contrib;
            prev_grad += contrib;
        }
        carry[s] = prev_grad;
        __syncthreads();
    }

    grad_initial_log_belief[b * kNumStates + s] = carry[s];
    const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
    if (s == 0) {
        grad_transition_gate_per_batch[b] += gate_sum;
    }
}

void launch_forward_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_log_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int batch_size = static_cast<int>(local_logits.size(0));
    const int seq_len = static_cast<int>(local_logits.size(1));
    const dim3 grid(batch_size);
    const dim3 block(kNumStates);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    causal_machine_forward_chunk_kernel<<<grid, block, 0, stream>>>(
        local_logits.data_ptr<float>(),
        transition_log_probs.data_ptr<float>(),
        transition_context.data_ptr<float>(),
        initial_log_belief.data_ptr<float>(),
        static_cast<float>(transition_gate),
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        beliefs.data_ptr<float>(),
        final_log_belief.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_backward_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_log_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_per_batch,
    const torch::Tensor& grad_transition_gate_per_batch) {
    const int batch_size = static_cast<int>(local_logits.size(0));
    const int seq_len = static_cast<int>(local_logits.size(1));
    const dim3 grid(batch_size);
    const dim3 block(kNumStates);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    causal_machine_backward_chunk_kernel<<<grid, block, 0, stream>>>(
        grad_beliefs.data_ptr<float>(),
        grad_final_belief.data_ptr<float>(),
        local_logits.data_ptr<float>(),
        transition_log_probs.data_ptr<float>(),
        transition_context.data_ptr<float>(),
        initial_log_belief.data_ptr<float>(),
        beliefs.data_ptr<float>(),
        static_cast<float>(transition_gate),
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        grad_local_logits.data_ptr<float>(),
        grad_transition_context.data_ptr<float>(),
        grad_initial_log_belief.data_ptr<float>(),
        grad_transition_per_batch.data_ptr<float>(),
        grad_transition_gate_per_batch.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::vector<torch::Tensor> causal_machine_scan_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto batch_size = local_logits.size(0);
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto prev = initial_log_belief.contiguous();
    auto final_log_belief = torch::empty_like(initial_log_belief);
    for (int64_t chunk_start = 0; chunk_start < seq_len; chunk_start += chunk_size) {
        const int64_t chunk_len = std::min(chunk_size, seq_len - chunk_start);
        launch_forward_chunk(
            local_logits,
            transition_log_probs,
            transition_context,
            prev,
            transition_gate,
            chunk_start,
            chunk_len,
            beliefs,
            final_log_belief);
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
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor final_belief,
    double transition_gate,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto batch_size = local_logits.size(0);
    const auto seq_len = local_logits.size(1);
    auto grad_local_logits = torch::zeros_like(local_logits);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_transition_per_batch = torch::zeros(
        {batch_size, kNumStates, kNumStates},
        local_logits.options());
    auto grad_transition_gate_per_batch = torch::zeros({batch_size}, local_logits.options());
    auto carry = grad_final_belief.contiguous();
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);

    for (int64_t chunk_end = seq_len; chunk_end > 0; chunk_end -= chunk_size) {
        const int64_t chunk_start = std::max<int64_t>(0, chunk_end - chunk_size);
        const int64_t this_chunk_len = chunk_end - chunk_start;
        auto prev = chunk_start == 0
            ? initial_log_belief.contiguous()
            : beliefs.select(1, chunk_start - 1).contiguous();
        auto chunk_grad_initial = torch::zeros_like(initial_log_belief);
        launch_backward_chunk(
            grad_beliefs,
            carry,
            local_logits,
            transition_log_probs,
            transition_context,
            prev,
            beliefs,
            transition_gate,
            chunk_start,
            this_chunk_len,
            grad_local_logits,
            grad_transition_context,
            chunk_grad_initial,
            grad_transition_per_batch,
            grad_transition_gate_per_batch);
        carry = chunk_grad_initial.contiguous();
    }
    if (seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief);
    } else {
        grad_initial_log_belief.copy_(carry);
    }

    auto grad_transition_log_probs = grad_transition_per_batch.sum(0);
    auto grad_transition_gate = grad_transition_gate_per_batch.sum().reshape({1});
    return {
        grad_local_logits,
        grad_transition_log_probs,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate,
    };
}
