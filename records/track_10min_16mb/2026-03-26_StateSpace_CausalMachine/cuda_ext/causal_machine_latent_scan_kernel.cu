#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kLatentChunkSize = 64;
constexpr int kChunkScanThreads = 64;

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr) {
    return static_cast<float>(*ptr);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_from_float(float value) {
    return static_cast<scalar_t>(value);
}

template <typename scalar_t, bool StoreStates>
__device__ __forceinline__ float latent_forward_chunk(
    const scalar_t* __restrict__ drive,
    float decay_r,
    int seq_len,
    int rank_dim,
    int b,
    int r,
    int chunk_start,
    int chunk_len,
    float prev,
    scalar_t* __restrict__ prior_states,
    scalar_t* __restrict__ states) {
    #pragma unroll 4
    for (int dt = 0; dt < chunk_len; ++dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        const float prior = decay_r * prev;
        prior_states[idx] = store_from_float<scalar_t>(prior);
        prev = prior + load_as_float(drive + idx);
        if constexpr (StoreStates) {
            states[idx] = store_from_float<scalar_t>(prev);
        }
    }
    return prev;
}

template <typename scalar_t, bool UseStatesPrev>
__device__ __forceinline__ void latent_backward_chunk(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ prior_states,
    const scalar_t* __restrict__ initial_state,
    float decay_r,
    int seq_len,
    int rank_dim,
    int b,
    int r,
    int chunk_start,
    int chunk_len,
    float& carry,
    float& decay_grad,
    scalar_t* __restrict__ grad_drive) {
    for (int dt = chunk_len - 1; dt >= 0; --dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        const float grad_state = (grad_states != nullptr ? load_as_float(grad_states + idx) : 0.0f) + carry;
        grad_drive[idx] = store_from_float<scalar_t>(grad_state);
        const float grad_prior = load_as_float(grad_prior_states + idx) + grad_state;
        float prev_state;
        if (t == 0) {
            prev_state = load_as_float(initial_state + b * rank_dim + r);
        } else if constexpr (UseStatesPrev) {
            prev_state = load_as_float(states + ((b * seq_len + (t - 1)) * rank_dim + r));
        } else {
            prev_state = load_as_float(prior_states + idx) / decay_r;
        }
        decay_grad += grad_prior * prev_state;
        carry = grad_prior * decay_r;
    }
}

template <typename scalar_t>
__global__ void latent_chunk_summary_kernel(
    const scalar_t* __restrict__ drive,
    const float* __restrict__ decay,
    int seq_len,
    int rank_dim,
    int num_chunks,
    float* __restrict__ chunk_mul,
    float* __restrict__ chunk_add) {
    const int b = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks || r >= rank_dim) {
        return;
    }
    const int chunk_start = chunk_id * kLatentChunkSize;
    const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
    const float decay_r = decay[r];
    float add = 0.0f;
    for (int dt = 0; dt < chunk_len; ++dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        add = decay_r * add + load_as_float(drive + idx);
    }
    const int summary_idx = (b * num_chunks + chunk_id) * rank_dim + r;
    chunk_mul[summary_idx] = powf(decay_r, static_cast<float>(chunk_len));
    chunk_add[summary_idx] = add;
}

template <typename scalar_t>
__global__ void latent_chunk_carry_kernel(
    const float* __restrict__ chunk_mul,
    const float* __restrict__ chunk_add,
    const scalar_t* __restrict__ initial_state,
    int num_chunks,
    int rank_dim,
    float* __restrict__ chunk_prev,
    scalar_t* __restrict__ final_state) {
    const int b = blockIdx.x;
    const int r = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rank_dim) {
        return;
    }
    float prev = load_as_float(initial_state + b * rank_dim + r);
    for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        const int summary_idx = (b * num_chunks + chunk_id) * rank_dim + r;
        chunk_prev[summary_idx] = prev;
        prev = chunk_mul[summary_idx] * prev + chunk_add[summary_idx];
    }
    final_state[b * rank_dim + r] = store_from_float<scalar_t>(prev);
}

template <typename scalar_t>
__global__ void latent_chunk_prefix_scan_kernel(
    const float* __restrict__ chunk_mul,
    const float* __restrict__ chunk_add,
    const scalar_t* __restrict__ initial_state,
    int num_chunks,
    int rank_dim,
    float* __restrict__ chunk_prev,
    scalar_t* __restrict__ final_state) {
    const int b = blockIdx.x;
    const int r = blockIdx.y;
    const int tid = threadIdx.x;
    if (r >= rank_dim || tid >= kChunkScanThreads) {
        return;
    }

    __shared__ float mul_shared[kChunkScanThreads];
    __shared__ float add_shared[kChunkScanThreads];
    __shared__ float mul_tmp[kChunkScanThreads];
    __shared__ float add_tmp[kChunkScanThreads];

    if (tid < num_chunks) {
        const int idx = (b * num_chunks + tid) * rank_dim + r;
        mul_shared[tid] = chunk_mul[idx];
        add_shared[tid] = chunk_add[idx];
    } else {
        mul_shared[tid] = 1.0f;
        add_shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int offset = 1; offset < num_chunks; offset <<= 1) {
        if (tid < num_chunks) {
            float cur_mul = mul_shared[tid];
            float cur_add = add_shared[tid];
            if (tid >= offset) {
                const float left_mul = mul_shared[tid - offset];
                const float left_add = add_shared[tid - offset];
                mul_tmp[tid] = cur_mul * left_mul;
                add_tmp[tid] = cur_mul * left_add + cur_add;
            } else {
                mul_tmp[tid] = cur_mul;
                add_tmp[tid] = cur_add;
            }
        }
        __syncthreads();
        if (tid < num_chunks) {
            mul_shared[tid] = mul_tmp[tid];
            add_shared[tid] = add_tmp[tid];
        }
        __syncthreads();
    }

    const float init = load_as_float(initial_state + b * rank_dim + r);
    if (tid < num_chunks) {
        float prev = init;
        if (tid > 0) {
            const float excl_mul = mul_shared[tid - 1];
            const float excl_add = add_shared[tid - 1];
            prev = excl_mul * init + excl_add;
        }
        chunk_prev[(b * num_chunks + tid) * rank_dim + r] = prev;
        if (tid == num_chunks - 1) {
            const float final_val = mul_shared[tid] * init + add_shared[tid];
            final_state[b * rank_dim + r] = store_from_float<scalar_t>(final_val);
        }
    }
}

template <typename scalar_t, bool StoreStates>
__global__ void latent_chunk_finalize_kernel(
    const scalar_t* __restrict__ drive,
    const float* __restrict__ decay,
    const float* __restrict__ chunk_prev,
    int seq_len,
    int rank_dim,
    int num_chunks,
    scalar_t* __restrict__ prior_states,
    scalar_t* __restrict__ states) {
    const int b = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks || r >= rank_dim) {
        return;
    }
    const int chunk_start = chunk_id * kLatentChunkSize;
    const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
    const float decay_r = decay[r];
    float prev = chunk_prev[(b * num_chunks + chunk_id) * rank_dim + r];
    for (int dt = 0; dt < chunk_len; ++dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        const float prior = decay_r * prev;
        prior_states[idx] = store_from_float<scalar_t>(prior);
        prev = prior + load_as_float(drive + idx);
        if constexpr (StoreStates) {
            states[idx] = store_from_float<scalar_t>(prev);
        }
    }
}

template <typename scalar_t, bool UseStateGrad, bool UseStatesPrev>
__global__ void latent_chunk_backward_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ prior_states,
    const scalar_t* __restrict__ initial_state,
    const float* __restrict__ decay,
    int seq_len,
    int rank_dim,
    int chunk_id,
    float* __restrict__ carry_io,
    scalar_t* __restrict__ grad_drive,
    float* __restrict__ grad_decay,
    scalar_t* __restrict__ grad_initial_state) {
    const int b = blockIdx.x;
    const int r = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rank_dim) {
        return;
    }
    const int chunk_start = chunk_id * kLatentChunkSize;
    const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
    const float decay_r = decay[r];
    float carry = carry_io[b * rank_dim + r];
    float decay_grad = 0.0f;
    for (int dt = chunk_len - 1; dt >= 0; --dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        const float grad_state = (UseStateGrad ? load_as_float(grad_states + idx) : 0.0f) + carry;
        grad_drive[idx] = store_from_float<scalar_t>(grad_state);
        const float grad_prior = load_as_float(grad_prior_states + idx) + grad_state;
        float prev_state;
        if (t == 0) {
            prev_state = load_as_float(initial_state + b * rank_dim + r);
        } else if constexpr (UseStatesPrev) {
            prev_state = load_as_float(states + ((b * seq_len + (t - 1)) * rank_dim + r));
        } else {
            prev_state = load_as_float(prior_states + idx) / decay_r;
        }
        decay_grad += grad_prior * prev_state;
        carry = grad_prior * decay_r;
    }
    carry_io[b * rank_dim + r] = carry;
    if (chunk_start == 0) {
        grad_initial_state[b * rank_dim + r] = store_from_float<scalar_t>(carry);
    }
    atomicAdd(grad_decay + r, decay_grad);
}

template <typename scalar_t, bool UseStateGrad>
__global__ void latent_chunk_backward_summary_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const float* __restrict__ decay,
    int seq_len,
    int rank_dim,
    int num_chunks,
    float* __restrict__ chunk_mul,
    float* __restrict__ chunk_add) {
    const int b = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks || r >= rank_dim) {
        return;
    }
    const int chunk_start = chunk_id * kLatentChunkSize;
    const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
    const float decay_r = decay[r];
    float add = 0.0f;
    for (int dt = chunk_len - 1; dt >= 0; --dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        const float local_grad = load_as_float(grad_prior_states + idx)
            + (UseStateGrad ? load_as_float(grad_states + idx) : 0.0f);
        add = decay_r * (local_grad + add);
    }
    const int summary_idx = (b * num_chunks + chunk_id) * rank_dim + r;
    chunk_mul[summary_idx] = powf(decay_r, static_cast<float>(chunk_len));
    chunk_add[summary_idx] = add;
}

template <typename scalar_t>
__global__ void latent_chunk_reverse_prefix_scan_kernel(
    const float* __restrict__ chunk_mul,
    const float* __restrict__ chunk_add,
    const scalar_t* __restrict__ grad_final_state,
    int num_chunks,
    int rank_dim,
    float* __restrict__ chunk_carry) {
    const int b = blockIdx.x;
    const int r = blockIdx.y;
    const int tid = threadIdx.x;
    if (r >= rank_dim || tid >= kChunkScanThreads) {
        return;
    }

    __shared__ float mul_shared[kChunkScanThreads];
    __shared__ float add_shared[kChunkScanThreads];
    __shared__ float mul_tmp[kChunkScanThreads];
    __shared__ float add_tmp[kChunkScanThreads];

    if (tid < num_chunks) {
        const int rev = num_chunks - 1 - tid;
        const int idx = (b * num_chunks + rev) * rank_dim + r;
        mul_shared[tid] = chunk_mul[idx];
        add_shared[tid] = chunk_add[idx];
    } else {
        mul_shared[tid] = 1.0f;
        add_shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int offset = 1; offset < num_chunks; offset <<= 1) {
        if (tid < num_chunks) {
            float cur_mul = mul_shared[tid];
            float cur_add = add_shared[tid];
            if (tid >= offset) {
                const float left_mul = mul_shared[tid - offset];
                const float left_add = add_shared[tid - offset];
                mul_tmp[tid] = cur_mul * left_mul;
                add_tmp[tid] = cur_mul * left_add + cur_add;
            } else {
                mul_tmp[tid] = cur_mul;
                add_tmp[tid] = cur_add;
            }
        }
        __syncthreads();
        if (tid < num_chunks) {
            mul_shared[tid] = mul_tmp[tid];
            add_shared[tid] = add_tmp[tid];
        }
        __syncthreads();
    }

    const float final_grad = load_as_float(grad_final_state + b * rank_dim + r);
    if (tid < num_chunks) {
        const int rev = num_chunks - 1 - tid;
        float carry = final_grad;
        if (tid > 0) {
            const float excl_mul = mul_shared[tid - 1];
            const float excl_add = add_shared[tid - 1];
            carry = excl_mul * final_grad + excl_add;
        }
        chunk_carry[(b * num_chunks + rev) * rank_dim + r] = carry;
    }
}

template <typename scalar_t, bool UseStateGrad, bool UseStatesPrev>
__global__ void latent_chunk_backward_finalize_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ prior_states,
    const scalar_t* __restrict__ initial_state,
    const float* __restrict__ decay,
    const float* __restrict__ chunk_carry,
    int seq_len,
    int rank_dim,
    int num_chunks,
    scalar_t* __restrict__ grad_drive,
    float* __restrict__ grad_decay,
    scalar_t* __restrict__ grad_initial_state) {
    const int b = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks || r >= rank_dim) {
        return;
    }
    const int chunk_start = chunk_id * kLatentChunkSize;
    const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
    const float decay_r = decay[r];
    float carry = chunk_carry[(b * num_chunks + chunk_id) * rank_dim + r];
    float decay_grad_local = 0.0f;
    for (int dt = chunk_len - 1; dt >= 0; --dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        const float grad_state = (UseStateGrad ? load_as_float(grad_states + idx) : 0.0f) + carry;
        grad_drive[idx] = store_from_float<scalar_t>(grad_state);
        const float grad_prior = load_as_float(grad_prior_states + idx) + grad_state;
        float prev_state;
        if (t == 0) {
            prev_state = load_as_float(initial_state + b * rank_dim + r);
        } else if constexpr (UseStatesPrev) {
            prev_state = load_as_float(states + ((b * seq_len + (t - 1)) * rank_dim + r));
        } else {
            prev_state = load_as_float(prior_states + idx) / decay_r;
        }
        decay_grad_local += grad_prior * prev_state;
        carry = grad_prior * decay_r;
    }
    if (chunk_id == 0) {
        grad_initial_state[b * rank_dim + r] = store_from_float<scalar_t>(carry);
    }
    atomicAdd(grad_decay + r, decay_grad_local);
}

template <typename scalar_t>
__global__ void latent_scan_forward_kernel(
    const scalar_t* __restrict__ drive,
    const float* __restrict__ decay,
    const scalar_t* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    scalar_t* __restrict__ prior_states,
    scalar_t* __restrict__ states,
    scalar_t* __restrict__ final_state) {
    const int b = blockIdx.x;
    const int r = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rank_dim) {
        return;
    }

    float prev = load_as_float(initial_state + b * rank_dim + r);
    const float decay_r = decay[r];
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += kLatentChunkSize) {
        const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
        prev = latent_forward_chunk<scalar_t, true>(
            drive,
            decay_r,
            seq_len,
            rank_dim,
            b,
            r,
            chunk_start,
            chunk_len,
            prev,
            prior_states,
            states);
    }
    final_state[b * rank_dim + r] = store_from_float<scalar_t>(prev);
}

template <typename scalar_t>
__global__ void latent_prior_scan_forward_kernel(
    const scalar_t* __restrict__ drive,
    const float* __restrict__ decay,
    const scalar_t* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    scalar_t* __restrict__ prior_states,
    scalar_t* __restrict__ final_state) {
    const int b = blockIdx.x;
    const int r = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rank_dim) {
        return;
    }

    float prev = load_as_float(initial_state + b * rank_dim + r);
    const float decay_r = decay[r];
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += kLatentChunkSize) {
        const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
        prev = latent_forward_chunk<scalar_t, false>(
            drive,
            decay_r,
            seq_len,
            rank_dim,
            b,
            r,
            chunk_start,
            chunk_len,
            prev,
            prior_states,
            nullptr);
    }
    final_state[b * rank_dim + r] = store_from_float<scalar_t>(prev);
}

template <typename scalar_t>
__global__ void latent_scan_backward_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
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
    for (int chunk_start = ((seq_len - 1) / kLatentChunkSize) * kLatentChunkSize; chunk_start >= 0; chunk_start -= kLatentChunkSize) {
        const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
        latent_backward_chunk<scalar_t, true>(
            grad_states,
            grad_prior_states,
            states,
            nullptr,
            initial_state,
            decay_r,
            seq_len,
            rank_dim,
            b,
            r,
            chunk_start,
            chunk_len,
            carry,
            decay_grad,
            grad_drive);
    }
    grad_initial_state[b * rank_dim + r] = store_from_float<scalar_t>(carry);
    atomicAdd(grad_decay + r, decay_grad);
}

template <typename scalar_t>
__global__ void latent_prior_scan_backward_kernel(
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ grad_final_state,
    const scalar_t* __restrict__ prior_states,
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
    for (int chunk_start = ((seq_len - 1) / kLatentChunkSize) * kLatentChunkSize; chunk_start >= 0; chunk_start -= kLatentChunkSize) {
        const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
        latent_backward_chunk<scalar_t, false>(
            nullptr,
            grad_prior_states,
            nullptr,
            prior_states,
            initial_state,
            decay_r,
            seq_len,
            rank_dim,
            b,
            r,
            chunk_start,
            chunk_len,
            carry,
            decay_grad,
            grad_drive);
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
    const auto num_chunks = static_cast<int>((seq_len + kLatentChunkSize - 1) / kLatentChunkSize);
    auto float_opts = drive.options().dtype(torch::kFloat32);
    auto chunk_mul = torch::zeros({batch_size, num_chunks, rank_dim}, float_opts);
    auto chunk_add = torch::zeros({batch_size, num_chunks, rank_dim}, float_opts);
    auto chunk_prev = torch::zeros({batch_size, num_chunks, rank_dim}, float_opts);
    auto prior_states = torch::zeros_like(drive);
    auto states = torch::zeros_like(drive);
    auto final_state = torch::zeros_like(initial_state);
    if (seq_len == 0) {
        final_state.copy_(initial_state);
        return {states, prior_states, final_state};
    }
    const dim3 block(kThreads);
    const dim3 summary_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 carry_grid(batch_size, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 finalize_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 scan_grid(batch_size, rank_dim);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, drive.scalar_type(), "latent_scan_forward_cuda", [&] {
        latent_chunk_summary_kernel<scalar_t><<<summary_grid, block, 0, stream>>>(
            drive.data_ptr<scalar_t>(),
            decay.data_ptr<float>(),
            seq_len,
            rank_dim,
            num_chunks,
            chunk_mul.data_ptr<float>(),
            chunk_add.data_ptr<float>());
        if (num_chunks <= kChunkScanThreads) {
            latent_chunk_prefix_scan_kernel<scalar_t><<<scan_grid, kChunkScanThreads, 0, stream>>>(
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>(),
                initial_state.data_ptr<scalar_t>(),
                num_chunks,
                rank_dim,
                chunk_prev.data_ptr<float>(),
                final_state.data_ptr<scalar_t>());
        } else {
            latent_chunk_carry_kernel<scalar_t><<<carry_grid, block, 0, stream>>>(
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>(),
                initial_state.data_ptr<scalar_t>(),
                num_chunks,
                rank_dim,
                chunk_prev.data_ptr<float>(),
                final_state.data_ptr<scalar_t>());
        }
        latent_chunk_finalize_kernel<scalar_t, true><<<finalize_grid, block, 0, stream>>>(
            drive.data_ptr<scalar_t>(),
            decay.data_ptr<float>(),
            chunk_prev.data_ptr<float>(),
            seq_len,
            rank_dim,
            num_chunks,
            prior_states.data_ptr<scalar_t>(),
            states.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {states, prior_states, final_state};
}

std::vector<torch::Tensor> causal_machine_latent_prior_scan_forward_cuda(
    torch::Tensor drive,
    torch::Tensor decay,
    torch::Tensor initial_state) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(drive));
    const auto batch_size = static_cast<int>(drive.size(0));
    const auto seq_len = static_cast<int>(drive.size(1));
    const auto rank_dim = static_cast<int>(drive.size(2));
    const auto num_chunks = static_cast<int>((seq_len + kLatentChunkSize - 1) / kLatentChunkSize);
    auto float_opts = drive.options().dtype(torch::kFloat32);
    auto chunk_mul = torch::zeros({batch_size, num_chunks, rank_dim}, float_opts);
    auto chunk_add = torch::zeros({batch_size, num_chunks, rank_dim}, float_opts);
    auto chunk_prev = torch::zeros({batch_size, num_chunks, rank_dim}, float_opts);
    auto prior_states = torch::zeros_like(drive);
    auto final_state = torch::zeros_like(initial_state);
    if (seq_len == 0) {
        final_state.copy_(initial_state);
        return {prior_states, final_state};
    }
    const dim3 block(kThreads);
    const dim3 summary_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 carry_grid(batch_size, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 finalize_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 scan_grid(batch_size, rank_dim);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, drive.scalar_type(), "latent_prior_scan_forward_cuda", [&] {
        latent_chunk_summary_kernel<scalar_t><<<summary_grid, block, 0, stream>>>(
            drive.data_ptr<scalar_t>(),
            decay.data_ptr<float>(),
            seq_len,
            rank_dim,
            num_chunks,
            chunk_mul.data_ptr<float>(),
            chunk_add.data_ptr<float>());
        if (num_chunks <= kChunkScanThreads) {
            latent_chunk_prefix_scan_kernel<scalar_t><<<scan_grid, kChunkScanThreads, 0, stream>>>(
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>(),
                initial_state.data_ptr<scalar_t>(),
                num_chunks,
                rank_dim,
                chunk_prev.data_ptr<float>(),
                final_state.data_ptr<scalar_t>());
        } else {
            latent_chunk_carry_kernel<scalar_t><<<carry_grid, block, 0, stream>>>(
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>(),
                initial_state.data_ptr<scalar_t>(),
                num_chunks,
                rank_dim,
                chunk_prev.data_ptr<float>(),
                final_state.data_ptr<scalar_t>());
        }
        latent_chunk_finalize_kernel<scalar_t, false><<<finalize_grid, block, 0, stream>>>(
            drive.data_ptr<scalar_t>(),
            decay.data_ptr<float>(),
            chunk_prev.data_ptr<float>(),
            seq_len,
            rank_dim,
            num_chunks,
            prior_states.data_ptr<scalar_t>(),
            nullptr);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {prior_states, final_state};
}

std::vector<torch::Tensor> causal_machine_latent_scan_backward_cuda(
    torch::Tensor grad_states,
    torch::Tensor grad_prior_states,
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
    if (seq_len == 0) {
        grad_initial_state.copy_(grad_final_state);
        return {grad_drive, grad_decay, grad_initial_state};
    }
    auto chunk_mul = torch::zeros({batch_size, (seq_len + kLatentChunkSize - 1) / kLatentChunkSize, rank_dim}, states.options().dtype(torch::kFloat32));
    auto chunk_add = torch::zeros_like(chunk_mul);
    auto chunk_carry = torch::zeros_like(chunk_mul);
    const dim3 block(kThreads);
    const dim3 grid(batch_size, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const int num_chunks = static_cast<int>((seq_len + kLatentChunkSize - 1) / kLatentChunkSize);
    const dim3 summary_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 finalize_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 scan_grid(batch_size, rank_dim);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, states.scalar_type(), "latent_scan_backward_cuda", [&] {
        latent_chunk_backward_summary_kernel<scalar_t, true><<<summary_grid, block, 0, stream>>>(
            grad_states.data_ptr<scalar_t>(),
            grad_prior_states.data_ptr<scalar_t>(),
            decay.data_ptr<float>(),
            seq_len,
            rank_dim,
            num_chunks,
            chunk_mul.data_ptr<float>(),
            chunk_add.data_ptr<float>());
        if (num_chunks <= kChunkScanThreads) {
            latent_chunk_reverse_prefix_scan_kernel<scalar_t><<<scan_grid, kChunkScanThreads, 0, stream>>>(
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>(),
                grad_final_state.data_ptr<scalar_t>(),
                num_chunks,
                rank_dim,
                chunk_carry.data_ptr<float>());
        } else {
            auto carry = grad_final_state.to(torch::kFloat32).contiguous();
            for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
                latent_chunk_backward_kernel<scalar_t, true, true><<<grid, block, 0, stream>>>(
                    grad_states.data_ptr<scalar_t>(),
                    grad_prior_states.data_ptr<scalar_t>(),
                    states.data_ptr<scalar_t>(),
                    nullptr,
                    initial_state.data_ptr<scalar_t>(),
                    decay.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    chunk_id,
                    carry.data_ptr<float>(),
                    grad_drive.data_ptr<scalar_t>(),
                    grad_decay.data_ptr<float>(),
                    grad_initial_state.data_ptr<scalar_t>());
            }
        }
        if (num_chunks <= kChunkScanThreads) {
            latent_chunk_backward_finalize_kernel<scalar_t, true, true><<<finalize_grid, block, 0, stream>>>(
                grad_states.data_ptr<scalar_t>(),
                grad_prior_states.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                nullptr,
                initial_state.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                chunk_carry.data_ptr<float>(),
                seq_len,
                rank_dim,
                num_chunks,
                grad_drive.data_ptr<scalar_t>(),
                grad_decay.data_ptr<float>(),
                grad_initial_state.data_ptr<scalar_t>());
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_drive, grad_decay, grad_initial_state};
}

std::vector<torch::Tensor> causal_machine_latent_prior_scan_backward_cuda(
    torch::Tensor grad_prior_states,
    torch::Tensor grad_final_state,
    torch::Tensor prior_states,
    torch::Tensor decay,
    torch::Tensor initial_state) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(prior_states));
    const auto batch_size = static_cast<int>(prior_states.size(0));
    const auto seq_len = static_cast<int>(prior_states.size(1));
    const auto rank_dim = static_cast<int>(prior_states.size(2));
    auto grad_drive = torch::zeros_like(prior_states);
    auto grad_decay = torch::zeros_like(decay);
    auto grad_initial_state = torch::zeros_like(initial_state);
    if (seq_len == 0) {
        grad_initial_state.copy_(grad_final_state);
        return {grad_drive, grad_decay, grad_initial_state};
    }
    auto chunk_mul = torch::zeros({batch_size, (seq_len + kLatentChunkSize - 1) / kLatentChunkSize, rank_dim}, prior_states.options().dtype(torch::kFloat32));
    auto chunk_add = torch::zeros_like(chunk_mul);
    auto chunk_carry = torch::zeros_like(chunk_mul);
    const dim3 block(kThreads);
    const dim3 grid(batch_size, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const int num_chunks = static_cast<int>((seq_len + kLatentChunkSize - 1) / kLatentChunkSize);
    const dim3 summary_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 finalize_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    const dim3 scan_grid(batch_size, rank_dim);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, prior_states.scalar_type(), "latent_prior_scan_backward_cuda", [&] {
        latent_chunk_backward_summary_kernel<scalar_t, false><<<summary_grid, block, 0, stream>>>(
            nullptr,
            grad_prior_states.data_ptr<scalar_t>(),
            decay.data_ptr<float>(),
            seq_len,
            rank_dim,
            num_chunks,
            chunk_mul.data_ptr<float>(),
            chunk_add.data_ptr<float>());
        if (num_chunks <= kChunkScanThreads) {
            latent_chunk_reverse_prefix_scan_kernel<scalar_t><<<scan_grid, kChunkScanThreads, 0, stream>>>(
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>(),
                grad_final_state.data_ptr<scalar_t>(),
                num_chunks,
                rank_dim,
                chunk_carry.data_ptr<float>());
        } else {
            auto carry = grad_final_state.to(torch::kFloat32).contiguous();
            for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
                latent_chunk_backward_kernel<scalar_t, false, false><<<grid, block, 0, stream>>>(
                    nullptr,
                    grad_prior_states.data_ptr<scalar_t>(),
                    nullptr,
                    prior_states.data_ptr<scalar_t>(),
                    initial_state.data_ptr<scalar_t>(),
                    decay.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    chunk_id,
                    carry.data_ptr<float>(),
                    grad_drive.data_ptr<scalar_t>(),
                    grad_decay.data_ptr<float>(),
                    grad_initial_state.data_ptr<scalar_t>());
            }
        }
        if (num_chunks <= kChunkScanThreads) {
            latent_chunk_backward_finalize_kernel<scalar_t, false, false><<<finalize_grid, block, 0, stream>>>(
                nullptr,
                grad_prior_states.data_ptr<scalar_t>(),
                nullptr,
                prior_states.data_ptr<scalar_t>(),
                initial_state.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                chunk_carry.data_ptr<float>(),
                seq_len,
                rank_dim,
                num_chunks,
                grad_drive.data_ptr<scalar_t>(),
                grad_decay.data_ptr<float>(),
                grad_initial_state.data_ptr<scalar_t>());
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_drive, grad_decay, grad_initial_state};
}
