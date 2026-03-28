#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kLatentChunkSize = 64;
// Keep the prefix-scan fast path active for substantially longer sequences.
// With chunk size 64, 256 scan threads covers up to 16384 tokens before the
// kernel has to fall back to the sequential carry path.
constexpr int kChunkScanThreads = 256;
// The single-kernel path parallelizes only over batch x latent_rank, so it
// under-fills large GPUs on the competition shape. Favor the chunked path for
// 1024-token runs unless the user explicitly opts back in.
constexpr int kDefaultSingleKernelMaxSeqLen = 512;

int latent_launch_threads(int rank_dim) {
    if (rank_dim <= 64) {
        return 64;
    }
    if (rank_dim <= 128) {
        return 128;
    }
    return kThreads;
}

int latent_single_kernel_max_seq_len() {
    static const int cached = []() {
        const char* raw = std::getenv("CAUSAL_MACHINE_LATENT_SCAN_SINGLE_KERNEL_MAX_SEQ_LEN");
        if (raw == nullptr || raw[0] == '\0') {
            return kDefaultSingleKernelMaxSeqLen;
        }
        const int parsed = std::atoi(raw);
        return parsed >= 0 ? parsed : kDefaultSingleKernelMaxSeqLen;
    }();
    return cached;
}

std::mutex g_latent_workspace_cache_mutex;
std::unordered_map<std::string, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> g_latent_workspace_cache;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_or_create_latent_workspace(
    const char* tag,
    const torch::Tensor& ref,
    int batch_size,
    int num_chunks,
    int rank_dim) {
    const auto device_index = ref.get_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    const std::string key = std::string(tag)
        + ":" + std::to_string(device_index)
        + ":" + std::to_string(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(stream)))
        + ":" + std::to_string(batch_size)
        + ":" + std::to_string(num_chunks)
        + ":" + std::to_string(rank_dim);
    const auto opts = ref.options().dtype(torch::kFloat32);
    const auto sizes = std::vector<int64_t>{
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(num_chunks),
        static_cast<int64_t>(rank_dim),
    };
    std::lock_guard<std::mutex> lock(g_latent_workspace_cache_mutex);
    auto& entry = g_latent_workspace_cache[key];
    auto ensure_tensor = [&](torch::Tensor& t) {
        if (
            !t.defined()
            || t.scalar_type() != torch::kFloat32
            || t.device() != ref.device()
            || t.sizes().vec() != sizes
        ) {
            t = torch::empty(sizes, opts);
        }
    };
    ensure_tensor(std::get<0>(entry));
    ensure_tensor(std::get<1>(entry));
    ensure_tensor(std::get<2>(entry));
    return entry;
}

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

__global__ void latent_group_summary_kernel(
    const float* __restrict__ in_mul,
    const float* __restrict__ in_add,
    int in_chunks,
    int rank_dim,
    int out_chunks,
    float* __restrict__ out_mul,
    float* __restrict__ out_add) {
    const int b = blockIdx.x;
    const int group_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (group_id >= out_chunks || r >= rank_dim) {
        return;
    }
    const int group_start = group_id * kChunkScanThreads;
    const int group_len = min(kChunkScanThreads, in_chunks - group_start);
    float mul = 1.0f;
    float add = 0.0f;
    for (int i = 0; i < group_len; ++i) {
        const int idx = (b * in_chunks + (group_start + i)) * rank_dim + r;
        const float cur_mul = in_mul[idx];
        const float cur_add = in_add[idx];
        add = cur_mul * add + cur_add;
        mul *= cur_mul;
    }
    const int out_idx = (b * out_chunks + group_id) * rank_dim + r;
    out_mul[out_idx] = mul;
    out_add[out_idx] = add;
}

__global__ void latent_chunk_group_prefix_scan_kernel(
    const float* __restrict__ chunk_mul,
    const float* __restrict__ chunk_add,
    const float* __restrict__ group_prev,
    int num_chunks,
    int rank_dim,
    int num_groups,
    float* __restrict__ chunk_prev) {
    const int b = blockIdx.x;
    const int group_id = blockIdx.y;
    const int r = blockIdx.z;
    const int tid = threadIdx.x;
    if (group_id >= num_groups || r >= rank_dim || tid >= kChunkScanThreads) {
        return;
    }

    __shared__ float mul_shared[kChunkScanThreads];
    __shared__ float add_shared[kChunkScanThreads];
    __shared__ float mul_tmp[kChunkScanThreads];
    __shared__ float add_tmp[kChunkScanThreads];

    const int group_start = group_id * kChunkScanThreads;
    const int group_len = min(kChunkScanThreads, num_chunks - group_start);
    if (tid < group_len) {
        const int idx = (b * num_chunks + (group_start + tid)) * rank_dim + r;
        mul_shared[tid] = chunk_mul[idx];
        add_shared[tid] = chunk_add[idx];
    } else {
        mul_shared[tid] = 1.0f;
        add_shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int offset = 1; offset < group_len; offset <<= 1) {
        if (tid < group_len) {
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
        if (tid < group_len) {
            mul_shared[tid] = mul_tmp[tid];
            add_shared[tid] = add_tmp[tid];
        }
        __syncthreads();
    }

    if (tid < group_len) {
        float prev = group_prev[(b * num_groups + group_id) * rank_dim + r];
        if (tid > 0) {
            const float excl_mul = mul_shared[tid - 1];
            const float excl_add = add_shared[tid - 1];
            prev = excl_mul * prev + excl_add;
        }
        chunk_prev[(b * num_chunks + (group_start + tid)) * rank_dim + r] = prev;
    }
}

__global__ void latent_group_summary_reverse_kernel(
    const float* __restrict__ in_mul,
    const float* __restrict__ in_add,
    int in_chunks,
    int rank_dim,
    int out_chunks,
    float* __restrict__ out_mul,
    float* __restrict__ out_add) {
    const int b = blockIdx.x;
    const int group_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (group_id >= out_chunks || r >= rank_dim) {
        return;
    }
    const int group_start = group_id * kChunkScanThreads;
    const int group_len = min(kChunkScanThreads, in_chunks - group_start);
    float mul = 1.0f;
    float add = 0.0f;
    for (int i = group_len - 1; i >= 0; --i) {
        const int idx = (b * in_chunks + (group_start + i)) * rank_dim + r;
        const float cur_mul = in_mul[idx];
        const float cur_add = in_add[idx];
        add = cur_mul * add + cur_add;
        mul *= cur_mul;
    }
    const int out_idx = (b * out_chunks + group_id) * rank_dim + r;
    out_mul[out_idx] = mul;
    out_add[out_idx] = add;
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

__global__ void latent_chunk_group_reverse_prefix_scan_kernel(
    const float* __restrict__ chunk_mul,
    const float* __restrict__ chunk_add,
    const float* __restrict__ group_carry,
    int num_chunks,
    int rank_dim,
    int num_groups,
    float* __restrict__ chunk_carry) {
    const int b = blockIdx.x;
    const int group_id = blockIdx.y;
    const int r = blockIdx.z;
    const int tid = threadIdx.x;
    if (group_id >= num_groups || r >= rank_dim || tid >= kChunkScanThreads) {
        return;
    }

    __shared__ float mul_shared[kChunkScanThreads];
    __shared__ float add_shared[kChunkScanThreads];
    __shared__ float mul_tmp[kChunkScanThreads];
    __shared__ float add_tmp[kChunkScanThreads];

    const int group_start = group_id * kChunkScanThreads;
    const int group_len = min(kChunkScanThreads, num_chunks - group_start);
    if (tid < group_len) {
        const int rev_local = group_len - 1 - tid;
        const int idx = (b * num_chunks + (group_start + rev_local)) * rank_dim + r;
        mul_shared[tid] = chunk_mul[idx];
        add_shared[tid] = chunk_add[idx];
    } else {
        mul_shared[tid] = 1.0f;
        add_shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int offset = 1; offset < group_len; offset <<= 1) {
        if (tid < group_len) {
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
        if (tid < group_len) {
            mul_shared[tid] = mul_tmp[tid];
            add_shared[tid] = add_tmp[tid];
        }
        __syncthreads();
    }

    if (tid < group_len) {
        const int rev_local = group_len - 1 - tid;
        const int actual_chunk = group_start + rev_local;
        float carry = group_carry[(b * num_groups + group_id) * rank_dim + r];
        if (tid > 0) {
            const float excl_mul = mul_shared[tid - 1];
            const float excl_add = add_shared[tid - 1];
            carry = excl_mul * carry + excl_add;
        }
        chunk_carry[(b * num_chunks + actual_chunk) * rank_dim + r] = carry;
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

template <typename scalar_t>
void latent_prefix_scan_dispatch(
    const torch::Tensor& chunk_mul,
    const torch::Tensor& chunk_add,
    const torch::Tensor& initial_state,
    int batch_size,
    int num_chunks,
    int rank_dim,
    cudaStream_t stream,
    torch::Tensor& chunk_prev,
    torch::Tensor& final_state) {
    if (num_chunks <= kChunkScanThreads) {
        const dim3 scan_grid(batch_size, rank_dim);
        latent_chunk_prefix_scan_kernel<scalar_t><<<scan_grid, kChunkScanThreads, 0, stream>>>(
            chunk_mul.data_ptr<float>(),
            chunk_add.data_ptr<float>(),
            initial_state.data_ptr<scalar_t>(),
            num_chunks,
            rank_dim,
            chunk_prev.data_ptr<float>(),
            final_state.data_ptr<scalar_t>());
        return;
    }

    const int num_groups = static_cast<int>((num_chunks + kChunkScanThreads - 1) / kChunkScanThreads);
    auto summary_opts = chunk_mul.options().dtype(torch::kFloat32);
    auto group_mul = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_add = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_prev = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_final = torch::empty_like(initial_state);
    const dim3 block(kThreads);
    const dim3 summary_grid(batch_size, num_groups, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    latent_group_summary_kernel<<<summary_grid, block, 0, stream>>>(
        chunk_mul.data_ptr<float>(),
        chunk_add.data_ptr<float>(),
        num_chunks,
        rank_dim,
        num_groups,
        group_mul.data_ptr<float>(),
        group_add.data_ptr<float>());
    latent_prefix_scan_dispatch<scalar_t>(
        group_mul,
        group_add,
        initial_state,
        batch_size,
        num_groups,
        rank_dim,
        stream,
        group_prev,
        group_final);
    const dim3 group_scan_grid(batch_size, num_groups, rank_dim);
    latent_chunk_group_prefix_scan_kernel<<<group_scan_grid, kChunkScanThreads, 0, stream>>>(
        chunk_mul.data_ptr<float>(),
        chunk_add.data_ptr<float>(),
        group_prev.data_ptr<float>(),
        num_chunks,
        rank_dim,
        num_groups,
        chunk_prev.data_ptr<float>());
    final_state.copy_(group_final);
}

template <typename scalar_t>
void latent_reverse_prefix_scan_dispatch(
    const torch::Tensor& chunk_mul,
    const torch::Tensor& chunk_add,
    const torch::Tensor& grad_final_state,
    int batch_size,
    int num_chunks,
    int rank_dim,
    cudaStream_t stream,
    torch::Tensor& chunk_carry) {
    if (num_chunks <= kChunkScanThreads) {
        const dim3 scan_grid(batch_size, rank_dim);
        latent_chunk_reverse_prefix_scan_kernel<scalar_t><<<scan_grid, kChunkScanThreads, 0, stream>>>(
            chunk_mul.data_ptr<float>(),
            chunk_add.data_ptr<float>(),
            grad_final_state.data_ptr<scalar_t>(),
            num_chunks,
            rank_dim,
            chunk_carry.data_ptr<float>());
        return;
    }

    const int num_groups = static_cast<int>((num_chunks + kChunkScanThreads - 1) / kChunkScanThreads);
    auto summary_opts = chunk_mul.options().dtype(torch::kFloat32);
    auto group_mul = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_add = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_carry = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    const dim3 block(kThreads);
    const dim3 summary_grid(batch_size, num_groups, static_cast<unsigned int>((rank_dim + kThreads - 1) / kThreads));
    latent_group_summary_reverse_kernel<<<summary_grid, block, 0, stream>>>(
        chunk_mul.data_ptr<float>(),
        chunk_add.data_ptr<float>(),
        num_chunks,
        rank_dim,
        num_groups,
        group_mul.data_ptr<float>(),
        group_add.data_ptr<float>());
    latent_reverse_prefix_scan_dispatch<scalar_t>(
        group_mul,
        group_add,
        grad_final_state,
        batch_size,
        num_groups,
        rank_dim,
        stream,
        group_carry);
    const dim3 group_scan_grid(batch_size, num_groups, rank_dim);
    latent_chunk_group_reverse_prefix_scan_kernel<<<group_scan_grid, kChunkScanThreads, 0, stream>>>(
        chunk_mul.data_ptr<float>(),
        chunk_add.data_ptr<float>(),
        group_carry.data_ptr<float>(),
        num_chunks,
        rank_dim,
        num_groups,
        chunk_carry.data_ptr<float>());
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
    auto prior_states = torch::empty_like(drive);
    auto states = torch::empty_like(drive);
    auto final_state = torch::empty_like(initial_state);
    if (batch_size == 0 || rank_dim == 0 || seq_len == 0) {
        final_state.copy_(initial_state);
        return {states, prior_states, final_state};
    }
    const int launch_threads = latent_launch_threads(rank_dim);
    const dim3 block(static_cast<unsigned int>(launch_threads));
    const dim3 rank_grid(batch_size, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    const dim3 summary_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    const dim3 finalize_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int single_kernel_max_seq_len = latent_single_kernel_max_seq_len();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, drive.scalar_type(), "latent_scan_forward_cuda", [&] {
        if (seq_len <= single_kernel_max_seq_len) {
            latent_scan_forward_kernel<scalar_t><<<rank_grid, block, 0, stream>>>(
                drive.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                initial_state.data_ptr<scalar_t>(),
                seq_len,
                rank_dim,
                prior_states.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                final_state.data_ptr<scalar_t>());
        } else {
            auto [chunk_mul, chunk_add, chunk_prev] = get_or_create_latent_workspace("forward", drive, batch_size, num_chunks, rank_dim);
            latent_chunk_summary_kernel<scalar_t><<<summary_grid, block, 0, stream>>>(
                drive.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                seq_len,
                rank_dim,
                num_chunks,
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>());
            latent_prefix_scan_dispatch<scalar_t>(
                chunk_mul,
                chunk_add,
                initial_state,
                batch_size,
                num_chunks,
                rank_dim,
                stream,
                chunk_prev,
                final_state);
            latent_chunk_finalize_kernel<scalar_t, true><<<finalize_grid, block, 0, stream>>>(
                drive.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                chunk_prev.data_ptr<float>(),
                seq_len,
                rank_dim,
                num_chunks,
                prior_states.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>());
        }
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
    auto prior_states = torch::empty_like(drive);
    auto final_state = torch::empty_like(initial_state);
    if (batch_size == 0 || rank_dim == 0 || seq_len == 0) {
        final_state.copy_(initial_state);
        return {prior_states, final_state};
    }
    const int launch_threads = latent_launch_threads(rank_dim);
    const dim3 block(static_cast<unsigned int>(launch_threads));
    const dim3 rank_grid(batch_size, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    const dim3 summary_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    const dim3 finalize_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int single_kernel_max_seq_len = latent_single_kernel_max_seq_len();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, drive.scalar_type(), "latent_prior_scan_forward_cuda", [&] {
        if (seq_len <= single_kernel_max_seq_len) {
            latent_prior_scan_forward_kernel<scalar_t><<<rank_grid, block, 0, stream>>>(
                drive.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                initial_state.data_ptr<scalar_t>(),
                seq_len,
                rank_dim,
                prior_states.data_ptr<scalar_t>(),
                final_state.data_ptr<scalar_t>());
        } else {
            auto [chunk_mul, chunk_add, chunk_prev] = get_or_create_latent_workspace("forward_prior", drive, batch_size, num_chunks, rank_dim);
            latent_chunk_summary_kernel<scalar_t><<<summary_grid, block, 0, stream>>>(
                drive.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                seq_len,
                rank_dim,
                num_chunks,
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>());
            latent_prefix_scan_dispatch<scalar_t>(
                chunk_mul,
                chunk_add,
                initial_state,
                batch_size,
                num_chunks,
                rank_dim,
                stream,
                chunk_prev,
                final_state);
            latent_chunk_finalize_kernel<scalar_t, false><<<finalize_grid, block, 0, stream>>>(
                drive.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                chunk_prev.data_ptr<float>(),
                seq_len,
                rank_dim,
                num_chunks,
                prior_states.data_ptr<scalar_t>(),
                nullptr);
        }
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
    auto grad_drive = torch::empty_like(states);
    auto grad_decay = torch::zeros_like(decay);
    auto grad_initial_state = torch::empty_like(initial_state);
    if (batch_size == 0 || rank_dim == 0 || seq_len == 0) {
        grad_initial_state.copy_(grad_final_state);
        return {grad_drive, grad_decay, grad_initial_state};
    }
    const int launch_threads = latent_launch_threads(rank_dim);
    const dim3 block(static_cast<unsigned int>(launch_threads));
    const int num_chunks = static_cast<int>((seq_len + kLatentChunkSize - 1) / kLatentChunkSize);
    const dim3 rank_grid(batch_size, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    const dim3 summary_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    const dim3 finalize_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int single_kernel_max_seq_len = latent_single_kernel_max_seq_len();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, states.scalar_type(), "latent_scan_backward_cuda", [&] {
        if (seq_len <= single_kernel_max_seq_len) {
            latent_scan_backward_kernel<scalar_t><<<rank_grid, block, 0, stream>>>(
                grad_states.data_ptr<scalar_t>(),
                grad_prior_states.data_ptr<scalar_t>(),
                grad_final_state.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                initial_state.data_ptr<scalar_t>(),
                seq_len,
                rank_dim,
                grad_drive.data_ptr<scalar_t>(),
                grad_decay.data_ptr<float>(),
                grad_initial_state.data_ptr<scalar_t>());
        } else {
            auto [chunk_mul, chunk_add, chunk_carry] = get_or_create_latent_workspace("backward", states, batch_size, num_chunks, rank_dim);
            latent_chunk_backward_summary_kernel<scalar_t, true><<<summary_grid, block, 0, stream>>>(
                grad_states.data_ptr<scalar_t>(),
                grad_prior_states.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                seq_len,
                rank_dim,
                num_chunks,
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>());
            latent_reverse_prefix_scan_dispatch<scalar_t>(
                chunk_mul,
                chunk_add,
                grad_final_state,
                batch_size,
                num_chunks,
                rank_dim,
                stream,
                chunk_carry);
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
    auto grad_drive = torch::empty_like(prior_states);
    auto grad_decay = torch::zeros_like(decay);
    auto grad_initial_state = torch::empty_like(initial_state);
    if (batch_size == 0 || rank_dim == 0 || seq_len == 0) {
        grad_initial_state.copy_(grad_final_state);
        return {grad_drive, grad_decay, grad_initial_state};
    }
    const int launch_threads = latent_launch_threads(rank_dim);
    const dim3 block(static_cast<unsigned int>(launch_threads));
    const int num_chunks = static_cast<int>((seq_len + kLatentChunkSize - 1) / kLatentChunkSize);
    const dim3 rank_grid(batch_size, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    const dim3 summary_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    const dim3 finalize_grid(batch_size, num_chunks, static_cast<unsigned int>((rank_dim + launch_threads - 1) / launch_threads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int single_kernel_max_seq_len = latent_single_kernel_max_seq_len();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, prior_states.scalar_type(), "latent_prior_scan_backward_cuda", [&] {
        if (seq_len <= single_kernel_max_seq_len) {
            latent_prior_scan_backward_kernel<scalar_t><<<rank_grid, block, 0, stream>>>(
                grad_prior_states.data_ptr<scalar_t>(),
                grad_final_state.data_ptr<scalar_t>(),
                prior_states.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                initial_state.data_ptr<scalar_t>(),
                seq_len,
                rank_dim,
                grad_drive.data_ptr<scalar_t>(),
                grad_decay.data_ptr<float>(),
                grad_initial_state.data_ptr<scalar_t>());
        } else {
            auto [chunk_mul, chunk_add, chunk_carry] = get_or_create_latent_workspace("backward_prior", prior_states, batch_size, num_chunks, rank_dim);
            latent_chunk_backward_summary_kernel<scalar_t, false><<<summary_grid, block, 0, stream>>>(
                nullptr,
                grad_prior_states.data_ptr<scalar_t>(),
                decay.data_ptr<float>(),
                seq_len,
                rank_dim,
                num_chunks,
                chunk_mul.data_ptr<float>(),
                chunk_add.data_ptr<float>());
            latent_reverse_prefix_scan_dispatch<scalar_t>(
                chunk_mul,
                chunk_add,
                grad_final_state,
                batch_size,
                num_chunks,
                rank_dim,
                stream,
                chunk_carry);
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
