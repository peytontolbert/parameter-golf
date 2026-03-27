#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr) {
    return static_cast<float>(*ptr);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_from_float(float value) {
    return static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void latent_scan_forward_kernel(
    const scalar_t* __restrict__ drive,
    const float* __restrict__ decay,
    const scalar_t* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    scalar_t* __restrict__ states,
    scalar_t* __restrict__ final_state) {
    const int b = blockIdx.x;
    const int r = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rank_dim) {
        return;
    }

    float prev = load_as_float(initial_state + b * rank_dim + r);
    const float decay_r = decay[r];
    for (int t = 0; t < seq_len; ++t) {
        const int idx = (b * seq_len + t) * rank_dim + r;
        prev = decay_r * prev + load_as_float(drive + idx);
        states[idx] = store_from_float<scalar_t>(prev);
    }
    final_state[b * rank_dim + r] = store_from_float<scalar_t>(prev);
}

template <typename scalar_t>
__global__ void latent_scan_backward_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_final_state,
    const scalar_t* __restrict__ states,
    const float* __restrict__ decay,
    const scalar_t* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    scalar_t* __restrict__ grad_drive,
    float* __restrict__ grad_decay,
    scalar_t* __restrict__ grad_initial_state) {
    const int b = blockIdx.x;
    const int r = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rank_dim) {
        return;
    }

    const float decay_r = decay[r];
    float carry = load_as_float(grad_final_state + b * rank_dim + r);
    float decay_grad = 0.0f;
    for (int t = seq_len - 1; t >= 0; --t) {
        const int idx = (b * seq_len + t) * rank_dim + r;
        const float total_grad = load_as_float(grad_states + idx) + carry;
        grad_drive[idx] = store_from_float<scalar_t>(total_grad);
        const float prev_state = (t == 0)
            ? load_as_float(initial_state + b * rank_dim + r)
            : load_as_float(states + ((b * seq_len + (t - 1)) * rank_dim + r));
        decay_grad += total_grad * prev_state;
        carry = total_grad * decay_r;
    }
    grad_initial_state[b * rank_dim + r] = store_from_float<scalar_t>(carry);
    atomicAdd(grad_decay + r, decay_grad);
}

}  // namespace

std::vector<torch::Tensor> causal_machine_latent_scan_forward_cuda(
    torch::Tensor drive,
    torch::Tensor decay,
    torch::Tensor initial_state) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(drive));
    const auto batch_size = static_cast<int>(drive.size(0));
    const auto seq_len = static_cast<int>(drive.size(1));
    const auto rank_dim = static_cast<int>(drive.size(2));
    auto states = torch::zeros_like(drive);
    auto final_state = torch::zeros_like(initial_state);
    const dim3 block(kThreads);
    const dim3 grid(batch_size, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, drive.scalar_type(), "latent_scan_forward_cuda", [&] {
        latent_scan_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
            drive.data_ptr<scalar_t>(),
            decay.data_ptr<float>(),
            initial_state.data_ptr<scalar_t>(),
            seq_len,
            rank_dim,
            states.data_ptr<scalar_t>(),
            final_state.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {states, final_state};
}

std::vector<torch::Tensor> causal_machine_latent_scan_backward_cuda(
    torch::Tensor grad_states,
    torch::Tensor grad_final_state,
    torch::Tensor states,
    torch::Tensor decay,
    torch::Tensor initial_state) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(states));
    const auto batch_size = static_cast<int>(states.size(0));
    const auto seq_len = static_cast<int>(states.size(1));
    const auto rank_dim = static_cast<int>(states.size(2));
    auto grad_drive = torch::zeros_like(states);
    auto grad_decay = torch::zeros_like(decay);
    auto grad_initial_state = torch::zeros_like(initial_state);
    const dim3 block(kThreads);
    const dim3 grid(batch_size, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, states.scalar_type(), "latent_scan_backward_cuda", [&] {
        latent_scan_backward_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_states.data_ptr<scalar_t>(),
            grad_final_state.data_ptr<scalar_t>(),
            states.data_ptr<scalar_t>(),
            decay.data_ptr<float>(),
            initial_state.data_ptr<scalar_t>(),
            seq_len,
            rank_dim,
            grad_drive.data_ptr<scalar_t>(),
            grad_decay.data_ptr<float>(),
            grad_initial_state.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_drive, grad_decay, grad_initial_state};
}
