#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
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

struct AffineScanValue {
    float mul;
    float add;
};

struct AffineScanOp {
    __device__ __forceinline__ AffineScanValue operator()(AffineScanValue lhs, AffineScanValue rhs) const {
        return {
            rhs.mul * lhs.mul,
            rhs.mul * lhs.add + rhs.add,
        };
    }
};

__device__ __forceinline__ AffineScanValue affine_scan_identity() {
    return {1.0f, 0.0f};
}

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

int cached_sm_count(int device_index) {
    static std::mutex mutex;
    static std::unordered_map<int, int> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(device_index);
    if (it != cache.end()) {
        return it->second;
    }
    int value = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &value,
        cudaDevAttrMultiProcessorCount,
        device_index));
    cache.emplace(device_index, value);
    return value;
}

int persistent_worker_blocks(int device_index, int total_tasks) {
    if (total_tasks <= 0) {
        return 0;
    }
    const int sm_count = std::max(cached_sm_count(device_index), 1);
    const int target_blocks = sm_count * 2;
    return std::max(1, std::min(total_tasks, target_blocks));
}

template <typename KernelT>
int occupancy_persistent_worker_blocks(
    KernelT kernel,
    int device_index,
    int total_tasks,
    int block_threads,
    size_t dynamic_smem_bytes = 0) {
    if (total_tasks <= 0) {
        return 0;
    }
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        kernel,
        block_threads,
        dynamic_smem_bytes));
    if (blocks_per_sm <= 0) {
        return persistent_worker_blocks(device_index, total_tasks);
    }
    const int target_blocks = std::max(blocks_per_sm * std::max(cached_sm_count(device_index), 1), 1);
    return std::max(1, std::min(total_tasks, target_blocks));
}

torch::Tensor make_device_work_queue_counter(const torch::Tensor& reference) {
    return torch::zeros({1}, reference.options().dtype(torch::kInt32));
}

std::mutex g_latent_workspace_cache_mutex;

enum class LatentWorkspaceTag : std::uint8_t {
    Forward = 0,
    ForwardPrior = 1,
    Backward = 2,
    BackwardPrior = 3,
};

struct LatentWorkspaceKey {
    std::uintptr_t stream;
    int device_index;
    int batch_size;
    int num_chunks;
    int rank_dim;
    int scalar_type;
    LatentWorkspaceTag tag;

    bool operator==(const LatentWorkspaceKey& other) const {
        return stream == other.stream
            && device_index == other.device_index
            && batch_size == other.batch_size
            && num_chunks == other.num_chunks
            && rank_dim == other.rank_dim
            && scalar_type == other.scalar_type
            && tag == other.tag;
    }
};

struct LatentWorkspaceKeyHash {
    std::size_t operator()(const LatentWorkspaceKey& key) const {
        std::size_t hash = static_cast<std::size_t>(key.stream);
        hash ^= static_cast<std::size_t>(key.device_index) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::size_t>(key.batch_size) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::size_t>(key.num_chunks) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::size_t>(key.rank_dim) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::size_t>(key.scalar_type) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::size_t>(key.tag) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

std::unordered_map<
    LatentWorkspaceKey,
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>,
    LatentWorkspaceKeyHash> g_latent_workspace_cache;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_or_create_latent_workspace(
    LatentWorkspaceTag tag,
    const torch::Tensor& ref,
    int batch_size,
    int num_chunks,
    int rank_dim,
    torch::ScalarType workspace_dtype) {
    const auto device_index = ref.get_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    const auto opts = ref.options().dtype(workspace_dtype);
    const auto key = LatentWorkspaceKey{
        reinterpret_cast<std::uintptr_t>(stream),
        device_index,
        batch_size,
        num_chunks,
        rank_dim,
        static_cast<int>(workspace_dtype),
        tag,
    };
    const auto batch_size_i64 = static_cast<int64_t>(batch_size);
    const auto num_chunks_i64 = static_cast<int64_t>(num_chunks);
    const auto rank_dim_i64 = static_cast<int64_t>(rank_dim);
    std::lock_guard<std::mutex> lock(g_latent_workspace_cache_mutex);
    auto& entry = g_latent_workspace_cache[key];
    auto ensure_tensor = [&](torch::Tensor& t) {
        if (
            !t.defined()
            || t.scalar_type() != workspace_dtype
            || t.device() != ref.device()
            || t.dim() != 3
            || t.size(0) != batch_size_i64
            || t.size(1) != num_chunks_i64
            || t.size(2) != rank_dim_i64
        ) {
            t = torch::empty({batch_size_i64, num_chunks_i64, rank_dim_i64}, opts);
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

template <typename scalar_t>
__device__ __forceinline__ bool latent_vec2_math_enabled_for_scalar() {
    return false;
}

template <>
__device__ __forceinline__ bool latent_vec2_math_enabled_for_scalar<c10::Half>() {
#if __CUDA_ARCH__ >= 530
    return true;
#else
    return false;
#endif
}

template <>
__device__ __forceinline__ bool latent_vec2_math_enabled_for_scalar<c10::BFloat16>() {
#if __CUDA_ARCH__ >= 800
    return true;
#else
    return false;
#endif
}

template <typename scalar_t>
__device__ __forceinline__ float2 load_pair_as_float2(const scalar_t* ptr) {
    return make_float2(load_as_float(ptr), load_as_float(ptr + 1));
}

template <>
__device__ __forceinline__ float2 load_pair_as_float2<c10::Half>(const c10::Half* ptr) {
    return __half22float2(*reinterpret_cast<const __half2*>(ptr));
}

template <>
__device__ __forceinline__ float2 load_pair_as_float2<c10::BFloat16>(const c10::BFloat16* ptr) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(ptr));
}

template <typename scalar_t>
__device__ __forceinline__ void store_pair_from_float2(scalar_t* ptr, float2 value) {
    ptr[0] = store_from_float<scalar_t>(value.x);
    ptr[1] = store_from_float<scalar_t>(value.y);
}

template <>
__device__ __forceinline__ void store_pair_from_float2<c10::Half>(c10::Half* ptr, float2 value) {
    *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(value.x, value.y);
}

template <>
__device__ __forceinline__ void store_pair_from_float2<c10::BFloat16>(c10::BFloat16* ptr, float2 value) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = __floats2bfloat162_rn(value.x, value.y);
}

__device__ __forceinline__ __half2 load_half2_pair(const c10::Half* ptr) {
    return *reinterpret_cast<const __half2*>(ptr);
}

__device__ __forceinline__ void store_half2_pair(c10::Half* ptr, __half2 value) {
    *reinterpret_cast<__half2*>(ptr) = value;
}

__device__ __forceinline__ __nv_bfloat162 load_bfloat162_pair(const c10::BFloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ __forceinline__ void store_bfloat162_pair(c10::BFloat16* ptr, __nv_bfloat162 value) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
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

template <bool StoreStates>
__device__ __forceinline__ __half2 latent_forward_chunk_half2(
    const c10::Half* __restrict__ drive,
    __half2 decay,
    int seq_len,
    int rank_dim,
    int b,
    int r,
    int chunk_start,
    int chunk_len,
    __half2 prev,
    c10::Half* __restrict__ prior_states,
    c10::Half* __restrict__ states) {
    #pragma unroll 4
    for (int dt = 0; dt < chunk_len; ++dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        const __half2 prior = __hmul2(decay, prev);
        store_half2_pair(prior_states + idx, prior);
        prev = __hadd2(prior, load_half2_pair(drive + idx));
        if constexpr (StoreStates) {
            store_half2_pair(states + idx, prev);
        }
    }
    return prev;
}

template <bool UseStateGrad, bool UseStatesPrev>
__device__ __forceinline__ void latent_backward_chunk_half2(
    const c10::Half* __restrict__ grad_states,
    const c10::Half* __restrict__ grad_prior_states,
    const c10::Half* __restrict__ states,
    const c10::Half* __restrict__ prior_states,
    const c10::Half* __restrict__ initial_state,
    __half2 decay,
    int seq_len,
    int rank_dim,
    int b,
    int r,
    int chunk_start,
    int chunk_len,
    __half2& carry,
    float2& decay_grad,
    c10::Half* __restrict__ grad_drive) {
    const float2 decay_f = __half22float2(decay);
    for (int dt = chunk_len - 1; dt >= 0; --dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        __half2 grad_state = carry;
        if constexpr (UseStateGrad) {
            grad_state = __hadd2(grad_state, load_half2_pair(grad_states + idx));
        }
        store_half2_pair(grad_drive + idx, grad_state);
        const __half2 grad_prior = __hadd2(load_half2_pair(grad_prior_states + idx), grad_state);
        const float2 grad_prior_f = __half22float2(grad_prior);
        float2 prev_state_f;
        if (t == 0) {
            prev_state_f = load_pair_as_float2(initial_state + b * rank_dim + r);
        } else if constexpr (UseStatesPrev) {
            prev_state_f = load_pair_as_float2(states + ((b * seq_len + (t - 1)) * rank_dim + r));
        } else {
            const float2 prior_f = load_pair_as_float2(prior_states + idx);
            prev_state_f = make_float2(prior_f.x / decay_f.x, prior_f.y / decay_f.y);
        }
        decay_grad.x += grad_prior_f.x * prev_state_f.x;
        decay_grad.y += grad_prior_f.y * prev_state_f.y;
        carry = __hmul2(grad_prior, decay);
    }
}

template <bool StoreStates>
__device__ __forceinline__ __nv_bfloat162 latent_forward_chunk_bfloat162(
    const c10::BFloat16* __restrict__ drive,
    __nv_bfloat162 decay,
    int seq_len,
    int rank_dim,
    int b,
    int r,
    int chunk_start,
    int chunk_len,
    __nv_bfloat162 prev,
    c10::BFloat16* __restrict__ prior_states,
    c10::BFloat16* __restrict__ states) {
    #pragma unroll 4
    for (int dt = 0; dt < chunk_len; ++dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        const __nv_bfloat162 prior = __hmul2(decay, prev);
        store_bfloat162_pair(prior_states + idx, prior);
        prev = __hadd2(prior, load_bfloat162_pair(drive + idx));
        if constexpr (StoreStates) {
            store_bfloat162_pair(states + idx, prev);
        }
    }
    return prev;
}

template <bool UseStateGrad, bool UseStatesPrev>
__device__ __forceinline__ void latent_backward_chunk_bfloat162(
    const c10::BFloat16* __restrict__ grad_states,
    const c10::BFloat16* __restrict__ grad_prior_states,
    const c10::BFloat16* __restrict__ states,
    const c10::BFloat16* __restrict__ prior_states,
    const c10::BFloat16* __restrict__ initial_state,
    __nv_bfloat162 decay,
    int seq_len,
    int rank_dim,
    int b,
    int r,
    int chunk_start,
    int chunk_len,
    __nv_bfloat162& carry,
    float2& decay_grad,
    c10::BFloat16* __restrict__ grad_drive) {
    const float2 decay_f = __bfloat1622float2(decay);
    for (int dt = chunk_len - 1; dt >= 0; --dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        __nv_bfloat162 grad_state = carry;
        if constexpr (UseStateGrad) {
            grad_state = __hadd2(grad_state, load_bfloat162_pair(grad_states + idx));
        }
        store_bfloat162_pair(grad_drive + idx, grad_state);
        const __nv_bfloat162 grad_prior = __hadd2(load_bfloat162_pair(grad_prior_states + idx), grad_state);
        const float2 grad_prior_f = __bfloat1622float2(grad_prior);
        float2 prev_state_f;
        if (t == 0) {
            prev_state_f = load_pair_as_float2(initial_state + b * rank_dim + r);
        } else if constexpr (UseStatesPrev) {
            prev_state_f = load_pair_as_float2(states + ((b * seq_len + (t - 1)) * rank_dim + r));
        } else {
            const float2 prior_f = load_pair_as_float2(prior_states + idx);
            prev_state_f = make_float2(prior_f.x / decay_f.x, prior_f.y / decay_f.y);
        }
        decay_grad.x += grad_prior_f.x * prev_state_f.x;
        decay_grad.y += grad_prior_f.y * prev_state_f.y;
        carry = __hmul2(grad_prior, decay);
    }
}

template <typename scalar_t, typename work_t>
__global__ __launch_bounds__(kThreads) void latent_chunk_summary_kernel(
    const scalar_t* __restrict__ drive,
    const scalar_t* __restrict__ decay,
    int seq_len,
    int rank_dim,
    int num_chunks,
    work_t* __restrict__ chunk_mul,
    work_t* __restrict__ chunk_add) {
    const int b = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks || r >= rank_dim) {
        return;
    }
    const int chunk_start = chunk_id * kLatentChunkSize;
    const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
    const float decay_r = load_as_float(decay + r);
    float add = 0.0f;
    for (int dt = 0; dt < chunk_len; ++dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        add = decay_r * add + load_as_float(drive + idx);
    }
    const int summary_idx = (b * num_chunks + chunk_id) * rank_dim + r;
    chunk_mul[summary_idx] = store_from_float<work_t>(powf(decay_r, static_cast<float>(chunk_len)));
    chunk_add[summary_idx] = store_from_float<work_t>(add);
}

template <typename scalar_t, typename work_t>
__global__ __launch_bounds__(kThreads) void latent_chunk_summary_persistent_kernel(
    const scalar_t* __restrict__ drive,
    const scalar_t* __restrict__ decay,
    int seq_len,
    int rank_dim,
    int num_chunks,
    int total_tasks,
    int32_t* __restrict__ work_queue_counter,
    work_t* __restrict__ chunk_mul,
    work_t* __restrict__ chunk_add) {
    const int rank_tile = blockIdx.y;
    const int r = rank_tile * blockDim.x + threadIdx.x;
    const bool active = r < rank_dim;
    __shared__ int current_task;
    while (true) {
        if (threadIdx.x == 0) {
            current_task = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_task >= total_tasks) {
            break;
        }
        if (active) {
            const int b = current_task / num_chunks;
            const int chunk_id = current_task - (b * num_chunks);
            const int chunk_start = chunk_id * kLatentChunkSize;
            const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
            const float decay_r = load_as_float(decay + r);
            float add = 0.0f;
            for (int dt = 0; dt < chunk_len; ++dt) {
                const int t = chunk_start + dt;
                const int idx = (b * seq_len + t) * rank_dim + r;
                add = decay_r * add + load_as_float(drive + idx);
            }
            const int summary_idx = (b * num_chunks + chunk_id) * rank_dim + r;
            chunk_mul[summary_idx] = store_from_float<work_t>(powf(decay_r, static_cast<float>(chunk_len)));
            chunk_add[summary_idx] = store_from_float<work_t>(add);
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename work_t>
__global__ __launch_bounds__(kThreads) void latent_chunk_carry_kernel(
    const work_t* __restrict__ chunk_mul,
    const work_t* __restrict__ chunk_add,
    const scalar_t* __restrict__ initial_state,
    int total_batches,
    int num_chunks,
    int rank_dim,
    int32_t* __restrict__ work_queue_counter,
    work_t* __restrict__ chunk_prev,
    scalar_t* __restrict__ final_state) {
    const int rank_tile = blockIdx.y;
    const int r = rank_tile * blockDim.x + threadIdx.x;
    const bool active = r < rank_dim;

    __shared__ int current_batch;
    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }

        if (active) {
            float prev = load_as_float(initial_state + current_batch * rank_dim + r);
            for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
                const int summary_idx = (current_batch * num_chunks + chunk_id) * rank_dim + r;
                chunk_prev[summary_idx] = store_from_float<work_t>(prev);
                prev = load_as_float(chunk_mul + summary_idx) * prev + load_as_float(chunk_add + summary_idx);
            }
            final_state[current_batch * rank_dim + r] = store_from_float<scalar_t>(prev);
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename work_t>
__global__ __launch_bounds__(kChunkScanThreads) void latent_chunk_prefix_scan_kernel(
    const work_t* __restrict__ chunk_mul,
    const work_t* __restrict__ chunk_add,
    const scalar_t* __restrict__ initial_state,
    int total_batches,
    int num_chunks,
    int rank_dim,
    work_t* __restrict__ chunk_prev,
    scalar_t* __restrict__ final_state) {
    const int current_batch = blockIdx.x;
    const int r = blockIdx.y;
    const int tid = threadIdx.x;
    if (current_batch >= total_batches || r >= rank_dim || tid >= kChunkScanThreads) {
        return;
    }

    using BlockScan = cub::BlockScan<AffineScanValue, kChunkScanThreads>;
    __shared__ typename BlockScan::TempStorage scan_storage;

    AffineScanValue current = affine_scan_identity();
    if (tid < num_chunks) {
        const int idx = (current_batch * num_chunks + tid) * rank_dim + r;
        current.mul = load_as_float(chunk_mul + idx);
        current.add = load_as_float(chunk_add + idx);
    }

    AffineScanValue prefix = affine_scan_identity();
    BlockScan(scan_storage).ExclusiveScan(current, prefix, affine_scan_identity(), AffineScanOp{});
    const AffineScanValue inclusive = AffineScanOp{}(prefix, current);

    const float init = load_as_float(initial_state + current_batch * rank_dim + r);
    if (tid < num_chunks) {
        const float prev = prefix.mul * init + prefix.add;
        chunk_prev[(current_batch * num_chunks + tid) * rank_dim + r] = store_from_float<work_t>(prev);
        if (tid == num_chunks - 1) {
            const float final_val = inclusive.mul * init + inclusive.add;
            final_state[current_batch * rank_dim + r] = store_from_float<scalar_t>(final_val);
        }
    }
}

template <typename work_t>
__global__ __launch_bounds__(kThreads) void latent_group_summary_kernel(
    const work_t* __restrict__ in_mul,
    const work_t* __restrict__ in_add,
    int total_batches,
    int in_chunks,
    int rank_dim,
    int out_chunks,
    work_t* __restrict__ out_mul,
    work_t* __restrict__ out_add) {
    const int current_batch = blockIdx.x;
    const int group_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (current_batch >= total_batches || group_id >= out_chunks || r >= rank_dim) {
        return;
    }

    const int group_start = group_id * kChunkScanThreads;
    const int group_len = min(kChunkScanThreads, in_chunks - group_start);
    float mul = 1.0f;
    float add = 0.0f;
    for (int i = 0; i < group_len; ++i) {
        const int idx = (current_batch * in_chunks + (group_start + i)) * rank_dim + r;
        const float cur_mul = load_as_float(in_mul + idx);
        const float cur_add = load_as_float(in_add + idx);
        add = cur_mul * add + cur_add;
        mul *= cur_mul;
    }
    const int out_idx = (current_batch * out_chunks + group_id) * rank_dim + r;
    out_mul[out_idx] = store_from_float<work_t>(mul);
    out_add[out_idx] = store_from_float<work_t>(add);
}

template <typename work_t>
__global__ __launch_bounds__(kChunkScanThreads) void latent_chunk_group_prefix_scan_kernel(
    const work_t* __restrict__ chunk_mul,
    const work_t* __restrict__ chunk_add,
    const work_t* __restrict__ group_prev,
    int total_batches,
    int num_chunks,
    int rank_dim,
    int num_groups,
    work_t* __restrict__ chunk_prev) {
    const int current_batch = blockIdx.x;
    const int group_id = blockIdx.y;
    const int r = blockIdx.z;
    const int tid = threadIdx.x;
    if (current_batch >= total_batches || group_id >= num_groups || r >= rank_dim || tid >= kChunkScanThreads) {
        return;
    }

    using BlockScan = cub::BlockScan<AffineScanValue, kChunkScanThreads>;
    __shared__ typename BlockScan::TempStorage scan_storage;

    const int group_start = group_id * kChunkScanThreads;
    const int group_len = min(kChunkScanThreads, num_chunks - group_start);
    AffineScanValue current = affine_scan_identity();
    if (tid < group_len) {
        const int idx = (current_batch * num_chunks + (group_start + tid)) * rank_dim + r;
        current.mul = load_as_float(chunk_mul + idx);
        current.add = load_as_float(chunk_add + idx);
    }

    AffineScanValue prefix = affine_scan_identity();
    BlockScan(scan_storage).ExclusiveScan(current, prefix, affine_scan_identity(), AffineScanOp{});

    if (tid < group_len) {
        const float group_seed = load_as_float(group_prev + ((current_batch * num_groups + group_id) * rank_dim + r));
        const float prev = prefix.mul * group_seed + prefix.add;
        chunk_prev[(current_batch * num_chunks + (group_start + tid)) * rank_dim + r] = store_from_float<work_t>(prev);
    }
}

template <typename work_t>
__global__ __launch_bounds__(kThreads) void latent_group_summary_reverse_kernel(
    const work_t* __restrict__ in_mul,
    const work_t* __restrict__ in_add,
    int total_batches,
    int in_chunks,
    int rank_dim,
    int out_chunks,
    work_t* __restrict__ out_mul,
    work_t* __restrict__ out_add) {
    const int current_batch = blockIdx.x;
    const int group_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (current_batch >= total_batches || group_id >= out_chunks || r >= rank_dim) {
        return;
    }

    const int group_start = group_id * kChunkScanThreads;
    const int group_len = min(kChunkScanThreads, in_chunks - group_start);
    float mul = 1.0f;
    float add = 0.0f;
    for (int i = group_len - 1; i >= 0; --i) {
        const int idx = (current_batch * in_chunks + (group_start + i)) * rank_dim + r;
        const float cur_mul = load_as_float(in_mul + idx);
        const float cur_add = load_as_float(in_add + idx);
        add = cur_mul * add + cur_add;
        mul *= cur_mul;
    }
    const int out_idx = (current_batch * out_chunks + group_id) * rank_dim + r;
    out_mul[out_idx] = store_from_float<work_t>(mul);
    out_add[out_idx] = store_from_float<work_t>(add);
}

template <typename scalar_t, typename work_t, bool StoreStates>
__global__ __launch_bounds__(kThreads) void latent_chunk_finalize_kernel(
    const scalar_t* __restrict__ drive,
    const scalar_t* __restrict__ decay,
    const work_t* __restrict__ chunk_prev,
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
    const float decay_r = load_as_float(decay + r);
    float prev = load_as_float(chunk_prev + ((b * num_chunks + chunk_id) * rank_dim + r));
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

template <typename scalar_t, typename work_t, bool StoreStates>
__global__ __launch_bounds__(kThreads) void latent_chunk_finalize_persistent_kernel(
    const scalar_t* __restrict__ drive,
    const scalar_t* __restrict__ decay,
    const work_t* __restrict__ chunk_prev,
    int seq_len,
    int rank_dim,
    int num_chunks,
    int total_tasks,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ prior_states,
    scalar_t* __restrict__ states) {
    const int rank_tile = blockIdx.y;
    const int r = rank_tile * blockDim.x + threadIdx.x;
    const bool active = r < rank_dim;
    __shared__ int current_task;
    while (true) {
        if (threadIdx.x == 0) {
            current_task = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_task >= total_tasks) {
            break;
        }
        if (active) {
            const int b = current_task / num_chunks;
            const int chunk_id = current_task - (b * num_chunks);
            const int chunk_start = chunk_id * kLatentChunkSize;
            const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
            const float decay_r = load_as_float(decay + r);
            float prev = load_as_float(chunk_prev + ((b * num_chunks + chunk_id) * rank_dim + r));
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
        __syncthreads();
    }
}

template <typename scalar_t, typename work_t, bool UseStateGrad, bool UseStatesPrev>
__global__ __launch_bounds__(kThreads) void latent_chunk_backward_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ prior_states,
    const scalar_t* __restrict__ initial_state,
    const work_t* __restrict__ decay,
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
    const float decay_r = load_as_float(decay + r);
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

template <typename scalar_t, typename work_t, bool UseStateGrad>
__global__ __launch_bounds__(kThreads) void latent_chunk_backward_summary_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ decay,
    int seq_len,
    int rank_dim,
    int num_chunks,
    work_t* __restrict__ chunk_mul,
    work_t* __restrict__ chunk_add) {
    const int b = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int r = blockIdx.z * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks || r >= rank_dim) {
        return;
    }
    const int chunk_start = chunk_id * kLatentChunkSize;
    const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
    const float decay_r = load_as_float(decay + r);
    float add = 0.0f;
    for (int dt = chunk_len - 1; dt >= 0; --dt) {
        const int t = chunk_start + dt;
        const int idx = (b * seq_len + t) * rank_dim + r;
        const float local_grad = load_as_float(grad_prior_states + idx)
            + (UseStateGrad ? load_as_float(grad_states + idx) : 0.0f);
        add = decay_r * (local_grad + add);
    }
    const int summary_idx = (b * num_chunks + chunk_id) * rank_dim + r;
    chunk_mul[summary_idx] = store_from_float<work_t>(powf(decay_r, static_cast<float>(chunk_len)));
    chunk_add[summary_idx] = store_from_float<work_t>(add);
}

template <typename scalar_t, typename work_t, bool UseStateGrad>
__global__ __launch_bounds__(kThreads) void latent_chunk_backward_summary_persistent_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ decay,
    int seq_len,
    int rank_dim,
    int num_chunks,
    int total_tasks,
    int32_t* __restrict__ work_queue_counter,
    work_t* __restrict__ chunk_mul,
    work_t* __restrict__ chunk_add) {
    const int rank_tile = blockIdx.y;
    const int r = rank_tile * blockDim.x + threadIdx.x;
    const bool active = r < rank_dim;
    __shared__ int current_task;
    while (true) {
        if (threadIdx.x == 0) {
            current_task = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_task >= total_tasks) {
            break;
        }
        if (active) {
            const int b = current_task / num_chunks;
            const int chunk_id = current_task - (b * num_chunks);
            const int chunk_start = chunk_id * kLatentChunkSize;
            const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
            const float decay_r = load_as_float(decay + r);
            float add = 0.0f;
            for (int dt = chunk_len - 1; dt >= 0; --dt) {
                const int t = chunk_start + dt;
                const int idx = (b * seq_len + t) * rank_dim + r;
                const float local_grad = load_as_float(grad_prior_states + idx)
                    + (UseStateGrad ? load_as_float(grad_states + idx) : 0.0f);
                add = decay_r * (local_grad + add);
            }
            const int summary_idx = (b * num_chunks + chunk_id) * rank_dim + r;
            chunk_mul[summary_idx] = store_from_float<work_t>(powf(decay_r, static_cast<float>(chunk_len)));
            chunk_add[summary_idx] = store_from_float<work_t>(add);
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename work_t>
__global__ __launch_bounds__(kChunkScanThreads) void latent_chunk_reverse_prefix_scan_kernel(
    const work_t* __restrict__ chunk_mul,
    const work_t* __restrict__ chunk_add,
    const scalar_t* __restrict__ grad_final_state,
    int total_batches,
    int num_chunks,
    int rank_dim,
    work_t* __restrict__ chunk_carry) {
    const int current_batch = blockIdx.x;
    const int r = blockIdx.y;
    const int tid = threadIdx.x;
    if (current_batch >= total_batches || r >= rank_dim || tid >= kChunkScanThreads) {
        return;
    }

    using BlockScan = cub::BlockScan<AffineScanValue, kChunkScanThreads>;
    __shared__ typename BlockScan::TempStorage scan_storage;

    AffineScanValue current = affine_scan_identity();
    if (tid < num_chunks) {
        const int rev = num_chunks - 1 - tid;
        const int idx = (current_batch * num_chunks + rev) * rank_dim + r;
        current.mul = load_as_float(chunk_mul + idx);
        current.add = load_as_float(chunk_add + idx);
    }

    AffineScanValue prefix = affine_scan_identity();
    BlockScan(scan_storage).ExclusiveScan(current, prefix, affine_scan_identity(), AffineScanOp{});

    const float final_grad = load_as_float(grad_final_state + current_batch * rank_dim + r);
    if (tid < num_chunks) {
        const int rev = num_chunks - 1 - tid;
        const float carry = prefix.mul * final_grad + prefix.add;
        chunk_carry[(current_batch * num_chunks + rev) * rank_dim + r] = store_from_float<work_t>(carry);
    }
}

template <typename work_t>
__global__ __launch_bounds__(kChunkScanThreads) void latent_chunk_group_reverse_prefix_scan_kernel(
    const work_t* __restrict__ chunk_mul,
    const work_t* __restrict__ chunk_add,
    const work_t* __restrict__ group_carry,
    int total_batches,
    int num_chunks,
    int rank_dim,
    int num_groups,
    work_t* __restrict__ chunk_carry) {
    const int current_batch = blockIdx.x;
    const int group_id = blockIdx.y;
    const int r = blockIdx.z;
    const int tid = threadIdx.x;
    if (current_batch >= total_batches || group_id >= num_groups || r >= rank_dim || tid >= kChunkScanThreads) {
        return;
    }

    using BlockScan = cub::BlockScan<AffineScanValue, kChunkScanThreads>;
    __shared__ typename BlockScan::TempStorage scan_storage;

    const int group_start = group_id * kChunkScanThreads;
    const int group_len = min(kChunkScanThreads, num_chunks - group_start);
    AffineScanValue current = affine_scan_identity();
    if (tid < group_len) {
        const int rev_local = group_len - 1 - tid;
        const int idx = (current_batch * num_chunks + (group_start + rev_local)) * rank_dim + r;
        current.mul = load_as_float(chunk_mul + idx);
        current.add = load_as_float(chunk_add + idx);
    }

    AffineScanValue prefix = affine_scan_identity();
    BlockScan(scan_storage).ExclusiveScan(current, prefix, affine_scan_identity(), AffineScanOp{});

    if (tid < group_len) {
        const int rev_local = group_len - 1 - tid;
        const int actual_chunk = group_start + rev_local;
        const float group_seed = load_as_float(group_carry + ((current_batch * num_groups + group_id) * rank_dim + r));
        const float carry = prefix.mul * group_seed + prefix.add;
        chunk_carry[(current_batch * num_chunks + actual_chunk) * rank_dim + r] = store_from_float<work_t>(carry);
    }
}

template <typename scalar_t, typename work_t, bool UseStateGrad, bool UseStatesPrev>
__global__ __launch_bounds__(kThreads) void latent_chunk_backward_finalize_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ prior_states,
    const scalar_t* __restrict__ initial_state,
    const scalar_t* __restrict__ decay,
    const work_t* __restrict__ chunk_carry,
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
    const float decay_r = load_as_float(decay + r);
    float carry = load_as_float(chunk_carry + ((b * num_chunks + chunk_id) * rank_dim + r));
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

template <typename scalar_t, typename work_t, bool UseStateGrad, bool UseStatesPrev>
__global__ __launch_bounds__(kThreads) void latent_chunk_backward_finalize_persistent_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ prior_states,
    const scalar_t* __restrict__ initial_state,
    const scalar_t* __restrict__ decay,
    const work_t* __restrict__ chunk_carry,
    int seq_len,
    int rank_dim,
    int num_chunks,
    int total_tasks,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_drive,
    float* __restrict__ grad_decay,
    scalar_t* __restrict__ grad_initial_state) {
    const int rank_tile = blockIdx.y;
    const int r = rank_tile * blockDim.x + threadIdx.x;
    const bool active = r < rank_dim;
    __shared__ int current_task;
    while (true) {
        if (threadIdx.x == 0) {
            current_task = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_task >= total_tasks) {
            break;
        }
        if (active) {
            const int b = current_task / num_chunks;
            const int chunk_id = current_task - (b * num_chunks);
            const int chunk_start = chunk_id * kLatentChunkSize;
            const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
            const float decay_r = load_as_float(decay + r);
            float carry = load_as_float(chunk_carry + ((b * num_chunks + chunk_id) * rank_dim + r));
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
        __syncthreads();
    }
}

template <bool StoreStates>
__global__ __launch_bounds__(kThreads) void latent_scan_forward_kernel_half2(
    const c10::Half* __restrict__ drive,
    const c10::Half* __restrict__ decay,
    const c10::Half* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    c10::Half* __restrict__ prior_states,
    c10::Half* __restrict__ states,
    c10::Half* __restrict__ final_state) {
    const int rank_pair_tile = blockIdx.y;
    const int r = (rank_pair_tile * blockDim.x + threadIdx.x) * 2;
    const bool active = (r + 1) < rank_dim;

    __shared__ int current_batch;
    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }

        if (active) {
            __half2 prev = load_half2_pair(initial_state + current_batch * rank_dim + r);
            const __half2 decay2 = load_half2_pair(decay + r);
            for (int chunk_start = 0; chunk_start < seq_len; chunk_start += kLatentChunkSize) {
                const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
                prev = latent_forward_chunk_half2<StoreStates>(
                    drive,
                    decay2,
                    seq_len,
                    rank_dim,
                    current_batch,
                    r,
                    chunk_start,
                    chunk_len,
                    prev,
                    prior_states,
                    states);
            }
            store_half2_pair(final_state + current_batch * rank_dim + r, prev);
        }
        __syncthreads();
    }
}

template <bool UseStateGrad, bool UseStatesPrev>
__global__ __launch_bounds__(kThreads) void latent_scan_backward_kernel_half2(
    const c10::Half* __restrict__ grad_states,
    const c10::Half* __restrict__ grad_prior_states,
    const c10::Half* __restrict__ grad_final_state,
    const c10::Half* __restrict__ states,
    const c10::Half* __restrict__ prior_states,
    const c10::Half* __restrict__ decay,
    const c10::Half* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    c10::Half* __restrict__ grad_drive,
    float* __restrict__ grad_decay,
    c10::Half* __restrict__ grad_initial_state) {
    const int rank_pair_tile = blockIdx.y;
    const int r = (rank_pair_tile * blockDim.x + threadIdx.x) * 2;
    const bool active = (r + 1) < rank_dim;

    __shared__ int current_batch;
    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }

        if (active) {
            const __half2 decay2 = load_half2_pair(decay + r);
            __half2 carry = load_half2_pair(grad_final_state + current_batch * rank_dim + r);
            float2 decay_grad = make_float2(0.0f, 0.0f);
            for (int chunk_start = ((seq_len - 1) / kLatentChunkSize) * kLatentChunkSize; chunk_start >= 0; chunk_start -= kLatentChunkSize) {
                const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
                latent_backward_chunk_half2<UseStateGrad, UseStatesPrev>(
                    grad_states,
                    grad_prior_states,
                    states,
                    prior_states,
                    initial_state,
                    decay2,
                    seq_len,
                    rank_dim,
                    current_batch,
                    r,
                    chunk_start,
                    chunk_len,
                    carry,
                    decay_grad,
                    grad_drive);
            }
            store_half2_pair(grad_initial_state + current_batch * rank_dim + r, carry);
            atomicAdd(grad_decay + r, decay_grad.x);
            atomicAdd(grad_decay + r + 1, decay_grad.y);
        }
        __syncthreads();
    }
}

template <bool StoreStates>
__global__ __launch_bounds__(kThreads) void latent_scan_forward_kernel_bfloat162(
    const c10::BFloat16* __restrict__ drive,
    const c10::BFloat16* __restrict__ decay,
    const c10::BFloat16* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    c10::BFloat16* __restrict__ prior_states,
    c10::BFloat16* __restrict__ states,
    c10::BFloat16* __restrict__ final_state) {
    const int rank_pair_tile = blockIdx.y;
    const int r = (rank_pair_tile * blockDim.x + threadIdx.x) * 2;
    const bool active = (r + 1) < rank_dim;

    __shared__ int current_batch;
    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }

        if (active) {
            __nv_bfloat162 prev = load_bfloat162_pair(initial_state + current_batch * rank_dim + r);
            const __nv_bfloat162 decay2 = load_bfloat162_pair(decay + r);
            for (int chunk_start = 0; chunk_start < seq_len; chunk_start += kLatentChunkSize) {
                const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
                prev = latent_forward_chunk_bfloat162<StoreStates>(
                    drive,
                    decay2,
                    seq_len,
                    rank_dim,
                    current_batch,
                    r,
                    chunk_start,
                    chunk_len,
                    prev,
                    prior_states,
                    states);
            }
            store_bfloat162_pair(final_state + current_batch * rank_dim + r, prev);
        }
        __syncthreads();
    }
}

template <bool UseStateGrad, bool UseStatesPrev>
__global__ __launch_bounds__(kThreads) void latent_scan_backward_kernel_bfloat162(
    const c10::BFloat16* __restrict__ grad_states,
    const c10::BFloat16* __restrict__ grad_prior_states,
    const c10::BFloat16* __restrict__ grad_final_state,
    const c10::BFloat16* __restrict__ states,
    const c10::BFloat16* __restrict__ prior_states,
    const c10::BFloat16* __restrict__ decay,
    const c10::BFloat16* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    c10::BFloat16* __restrict__ grad_drive,
    float* __restrict__ grad_decay,
    c10::BFloat16* __restrict__ grad_initial_state) {
    const int rank_pair_tile = blockIdx.y;
    const int r = (rank_pair_tile * blockDim.x + threadIdx.x) * 2;
    const bool active = (r + 1) < rank_dim;

    __shared__ int current_batch;
    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }

        if (active) {
            const __nv_bfloat162 decay2 = load_bfloat162_pair(decay + r);
            __nv_bfloat162 carry = load_bfloat162_pair(grad_final_state + current_batch * rank_dim + r);
            float2 decay_grad = make_float2(0.0f, 0.0f);
            for (int chunk_start = ((seq_len - 1) / kLatentChunkSize) * kLatentChunkSize; chunk_start >= 0; chunk_start -= kLatentChunkSize) {
                const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
                latent_backward_chunk_bfloat162<UseStateGrad, UseStatesPrev>(
                    grad_states,
                    grad_prior_states,
                    states,
                    prior_states,
                    initial_state,
                    decay2,
                    seq_len,
                    rank_dim,
                    current_batch,
                    r,
                    chunk_start,
                    chunk_len,
                    carry,
                    decay_grad,
                    grad_drive);
            }
            store_bfloat162_pair(grad_initial_state + current_batch * rank_dim + r, carry);
            atomicAdd(grad_decay + r, decay_grad.x);
            atomicAdd(grad_decay + r + 1, decay_grad.y);
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kThreads) void latent_chunk_summary_persistent_kernel_half2(
    const c10::Half* __restrict__ drive,
    const c10::Half* __restrict__ decay,
    int seq_len,
    int rank_dim,
    int num_chunks,
    int total_tasks,
    int32_t* __restrict__ work_queue_counter,
    float* __restrict__ chunk_mul,
    float* __restrict__ chunk_add) {
    const int rank_pair_tile = blockIdx.y;
    const int r = (rank_pair_tile * blockDim.x + threadIdx.x) * 2;
    const bool active = (r + 1) < rank_dim;
    __shared__ int current_task;
    while (true) {
        if (threadIdx.x == 0) {
            current_task = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_task >= total_tasks) {
            break;
        }
        if (active) {
            const int b = current_task / num_chunks;
            const int chunk_id = current_task - (b * num_chunks);
            const int chunk_start = chunk_id * kLatentChunkSize;
            const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
            const __half2 decay2 = load_half2_pair(decay + r);
            __half2 add = __floats2half2_rn(0.0f, 0.0f);
            __half2 mul = __floats2half2_rn(1.0f, 1.0f);
            for (int dt = 0; dt < chunk_len; ++dt) {
                const int t = chunk_start + dt;
                const int idx = (b * seq_len + t) * rank_dim + r;
                add = __hadd2(__hmul2(decay2, add), load_half2_pair(drive + idx));
                mul = __hmul2(mul, decay2);
            }
            const int summary_idx = (b * num_chunks + chunk_id) * rank_dim + r;
            store_pair_from_float2(chunk_mul + summary_idx, __half22float2(mul));
            store_pair_from_float2(chunk_add + summary_idx, __half22float2(add));
        }
        __syncthreads();
    }
}

template <bool StoreStates>
__global__ __launch_bounds__(kThreads) void latent_chunk_finalize_persistent_kernel_half2(
    const c10::Half* __restrict__ drive,
    const c10::Half* __restrict__ decay,
    const float* __restrict__ chunk_prev,
    int seq_len,
    int rank_dim,
    int num_chunks,
    int total_tasks,
    int32_t* __restrict__ work_queue_counter,
    c10::Half* __restrict__ prior_states,
    c10::Half* __restrict__ states) {
    const int rank_pair_tile = blockIdx.y;
    const int r = (rank_pair_tile * blockDim.x + threadIdx.x) * 2;
    const bool active = (r + 1) < rank_dim;
    __shared__ int current_task;
    while (true) {
        if (threadIdx.x == 0) {
            current_task = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_task >= total_tasks) {
            break;
        }
        if (active) {
            const int b = current_task / num_chunks;
            const int chunk_id = current_task - (b * num_chunks);
            const int chunk_start = chunk_id * kLatentChunkSize;
            const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
            const __half2 decay2 = load_half2_pair(decay + r);
            const float2 prev_f = load_pair_as_float2(chunk_prev + ((b * num_chunks + chunk_id) * rank_dim + r));
            __half2 prev = __floats2half2_rn(prev_f.x, prev_f.y);
            prev = latent_forward_chunk_half2<StoreStates>(
                drive,
                decay2,
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
        __syncthreads();
    }
}

template <bool UseStateGrad>
__global__ __launch_bounds__(kThreads) void latent_chunk_backward_summary_persistent_kernel_half2(
    const c10::Half* __restrict__ grad_states,
    const c10::Half* __restrict__ grad_prior_states,
    const c10::Half* __restrict__ decay,
    int seq_len,
    int rank_dim,
    int num_chunks,
    int total_tasks,
    int32_t* __restrict__ work_queue_counter,
    float* __restrict__ chunk_mul,
    float* __restrict__ chunk_add) {
    const int rank_pair_tile = blockIdx.y;
    const int r = (rank_pair_tile * blockDim.x + threadIdx.x) * 2;
    const bool active = (r + 1) < rank_dim;
    __shared__ int current_task;
    while (true) {
        if (threadIdx.x == 0) {
            current_task = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_task >= total_tasks) {
            break;
        }
        if (active) {
            const int b = current_task / num_chunks;
            const int chunk_id = current_task - (b * num_chunks);
            const int chunk_start = chunk_id * kLatentChunkSize;
            const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
            const __half2 decay2 = load_half2_pair(decay + r);
            __half2 add = __floats2half2_rn(0.0f, 0.0f);
            __half2 mul = __floats2half2_rn(1.0f, 1.0f);
            for (int dt = chunk_len - 1; dt >= 0; --dt) {
                const int t = chunk_start + dt;
                const int idx = (b * seq_len + t) * rank_dim + r;
                __half2 local_grad = load_half2_pair(grad_prior_states + idx);
                if constexpr (UseStateGrad) {
                    local_grad = __hadd2(local_grad, load_half2_pair(grad_states + idx));
                }
                add = __hmul2(decay2, __hadd2(local_grad, add));
                mul = __hmul2(mul, decay2);
            }
            const int summary_idx = (b * num_chunks + chunk_id) * rank_dim + r;
            store_pair_from_float2(chunk_mul + summary_idx, __half22float2(mul));
            store_pair_from_float2(chunk_add + summary_idx, __half22float2(add));
        }
        __syncthreads();
    }
}

template <bool UseStateGrad, bool UseStatesPrev>
__global__ __launch_bounds__(kThreads) void latent_chunk_backward_finalize_persistent_kernel_half2(
    const c10::Half* __restrict__ grad_states,
    const c10::Half* __restrict__ grad_prior_states,
    const c10::Half* __restrict__ states,
    const c10::Half* __restrict__ prior_states,
    const c10::Half* __restrict__ initial_state,
    const c10::Half* __restrict__ decay,
    const float* __restrict__ chunk_carry,
    int seq_len,
    int rank_dim,
    int num_chunks,
    int total_tasks,
    int32_t* __restrict__ work_queue_counter,
    c10::Half* __restrict__ grad_drive,
    float* __restrict__ grad_decay,
    c10::Half* __restrict__ grad_initial_state) {
    const int rank_pair_tile = blockIdx.y;
    const int r = (rank_pair_tile * blockDim.x + threadIdx.x) * 2;
    const bool active = (r + 1) < rank_dim;
    __shared__ int current_task;
    while (true) {
        if (threadIdx.x == 0) {
            current_task = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_task >= total_tasks) {
            break;
        }
        if (active) {
            const int b = current_task / num_chunks;
            const int chunk_id = current_task - (b * num_chunks);
            const int chunk_start = chunk_id * kLatentChunkSize;
            const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
            const __half2 decay2 = load_half2_pair(decay + r);
            const float2 carry_f = load_pair_as_float2(chunk_carry + ((b * num_chunks + chunk_id) * rank_dim + r));
            __half2 carry = __floats2half2_rn(carry_f.x, carry_f.y);
            float2 decay_grad_local = make_float2(0.0f, 0.0f);
            latent_backward_chunk_half2<UseStateGrad, UseStatesPrev>(
                grad_states,
                grad_prior_states,
                states,
                prior_states,
                initial_state,
                decay2,
                seq_len,
                rank_dim,
                b,
                r,
                chunk_start,
                chunk_len,
                carry,
                decay_grad_local,
                grad_drive);
            if (chunk_id == 0) {
                store_half2_pair(grad_initial_state + b * rank_dim + r, carry);
            }
            atomicAdd(grad_decay + r, decay_grad_local.x);
            atomicAdd(grad_decay + r + 1, decay_grad_local.y);
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename work_t>
__global__ __launch_bounds__(kThreads) void latent_scan_forward_kernel(
    const scalar_t* __restrict__ drive,
    const work_t* __restrict__ decay,
    const scalar_t* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ prior_states,
    scalar_t* __restrict__ states,
    scalar_t* __restrict__ final_state) {
    const int rank_tile = blockIdx.y;
    const int r = rank_tile * blockDim.x + threadIdx.x;
    const bool active = r < rank_dim;

    __shared__ int current_batch;
    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }

        if (active) {
            float prev = load_as_float(initial_state + current_batch * rank_dim + r);
            const float decay_r = load_as_float(decay + r);
            for (int chunk_start = 0; chunk_start < seq_len; chunk_start += kLatentChunkSize) {
                const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
                prev = latent_forward_chunk<scalar_t, true>(
                    drive,
                    decay_r,
                    seq_len,
                    rank_dim,
                    current_batch,
                    r,
                    chunk_start,
                    chunk_len,
                    prev,
                    prior_states,
                    states);
            }
            final_state[current_batch * rank_dim + r] = store_from_float<scalar_t>(prev);
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename work_t>
__global__ __launch_bounds__(kThreads) void latent_prior_scan_forward_kernel(
    const scalar_t* __restrict__ drive,
    const work_t* __restrict__ decay,
    const scalar_t* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ prior_states,
    scalar_t* __restrict__ final_state) {
    const int rank_tile = blockIdx.y;
    const int r = rank_tile * blockDim.x + threadIdx.x;
    const bool active = r < rank_dim;

    __shared__ int current_batch;
    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }

        if (active) {
            float prev = load_as_float(initial_state + current_batch * rank_dim + r);
            const float decay_r = load_as_float(decay + r);
            for (int chunk_start = 0; chunk_start < seq_len; chunk_start += kLatentChunkSize) {
                const int chunk_len = min(kLatentChunkSize, seq_len - chunk_start);
                prev = latent_forward_chunk<scalar_t, false>(
                    drive,
                    decay_r,
                    seq_len,
                    rank_dim,
                    current_batch,
                    r,
                    chunk_start,
                    chunk_len,
                    prev,
                    prior_states,
                    nullptr);
            }
            final_state[current_batch * rank_dim + r] = store_from_float<scalar_t>(prev);
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename work_t>
__global__ __launch_bounds__(kThreads) void latent_scan_backward_kernel(
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ grad_final_state,
    const scalar_t* __restrict__ states,
    const work_t* __restrict__ decay,
    const scalar_t* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_drive,
    float* __restrict__ grad_decay,
    scalar_t* __restrict__ grad_initial_state) {
    const int rank_tile = blockIdx.y;
    const int r = rank_tile * blockDim.x + threadIdx.x;
    const bool active = r < rank_dim;

    __shared__ int current_batch;
    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }

        if (active) {
            const float decay_r = load_as_float(decay + r);
            float carry = load_as_float(grad_final_state + current_batch * rank_dim + r);
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
                    current_batch,
                    r,
                    chunk_start,
                    chunk_len,
                    carry,
                    decay_grad,
                    grad_drive);
            }
            grad_initial_state[current_batch * rank_dim + r] = store_from_float<scalar_t>(carry);
            atomicAdd(grad_decay + r, decay_grad);
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename work_t>
__global__ __launch_bounds__(kThreads) void latent_prior_scan_backward_kernel(
    const scalar_t* __restrict__ grad_prior_states,
    const scalar_t* __restrict__ grad_final_state,
    const scalar_t* __restrict__ prior_states,
    const work_t* __restrict__ decay,
    const scalar_t* __restrict__ initial_state,
    int seq_len,
    int rank_dim,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_drive,
    float* __restrict__ grad_decay,
    scalar_t* __restrict__ grad_initial_state) {
    const int rank_tile = blockIdx.y;
    const int r = rank_tile * blockDim.x + threadIdx.x;
    const bool active = r < rank_dim;

    __shared__ int current_batch;
    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }

        if (active) {
            const float decay_r = load_as_float(decay + r);
            float carry = load_as_float(grad_final_state + current_batch * rank_dim + r);
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
                    current_batch,
                    r,
                    chunk_start,
                    chunk_len,
                    carry,
                    decay_grad,
                    grad_drive);
            }
            grad_initial_state[current_batch * rank_dim + r] = store_from_float<scalar_t>(carry);
            atomicAdd(grad_decay + r, decay_grad);
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename work_t>
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
        const dim3 scan_grid(
            static_cast<unsigned int>(batch_size),
            static_cast<unsigned int>(rank_dim));
        latent_chunk_prefix_scan_kernel<scalar_t, work_t><<<scan_grid, kChunkScanThreads, 0, stream>>>(
            chunk_mul.data_ptr<work_t>(),
            chunk_add.data_ptr<work_t>(),
            initial_state.data_ptr<scalar_t>(),
            batch_size,
            num_chunks,
            rank_dim,
            chunk_prev.data_ptr<work_t>(),
            final_state.data_ptr<scalar_t>());
        return;
    }

    const int num_groups = static_cast<int>((num_chunks + kChunkScanThreads - 1) / kChunkScanThreads);
    auto summary_opts = chunk_mul.options();
    auto group_mul = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_add = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_prev = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_final = torch::empty_like(initial_state);
    const dim3 block(kThreads);
    const int rank_tiles = static_cast<int>((rank_dim + kThreads - 1) / kThreads);
    const dim3 summary_grid(
        static_cast<unsigned int>(batch_size),
        static_cast<unsigned int>(num_groups),
        static_cast<unsigned int>(rank_tiles));
    latent_group_summary_kernel<work_t><<<summary_grid, block, 0, stream>>>(
        chunk_mul.data_ptr<work_t>(),
        chunk_add.data_ptr<work_t>(),
        batch_size,
        num_chunks,
        rank_dim,
        num_groups,
        group_mul.data_ptr<work_t>(),
        group_add.data_ptr<work_t>());
    latent_prefix_scan_dispatch<scalar_t, work_t>(
        group_mul,
        group_add,
        initial_state,
        batch_size,
        num_groups,
        rank_dim,
        stream,
        group_prev,
        group_final);
    const dim3 group_scan_grid(
        static_cast<unsigned int>(batch_size),
        static_cast<unsigned int>(num_groups),
        static_cast<unsigned int>(rank_dim));
    latent_chunk_group_prefix_scan_kernel<work_t><<<group_scan_grid, kChunkScanThreads, 0, stream>>>(
        chunk_mul.data_ptr<work_t>(),
        chunk_add.data_ptr<work_t>(),
        group_prev.data_ptr<work_t>(),
        batch_size,
        num_chunks,
        rank_dim,
        num_groups,
        chunk_prev.data_ptr<work_t>());
    final_state.copy_(group_final);
}

template <typename scalar_t, typename work_t>
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
        const dim3 scan_grid(
            static_cast<unsigned int>(batch_size),
            static_cast<unsigned int>(rank_dim));
        latent_chunk_reverse_prefix_scan_kernel<scalar_t, work_t><<<scan_grid, kChunkScanThreads, 0, stream>>>(
            chunk_mul.data_ptr<work_t>(),
            chunk_add.data_ptr<work_t>(),
            grad_final_state.data_ptr<scalar_t>(),
            batch_size,
            num_chunks,
            rank_dim,
            chunk_carry.data_ptr<work_t>());
        return;
    }

    const int num_groups = static_cast<int>((num_chunks + kChunkScanThreads - 1) / kChunkScanThreads);
    auto summary_opts = chunk_mul.options();
    auto group_mul = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_add = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    auto group_carry = torch::empty({batch_size, num_groups, rank_dim}, summary_opts);
    const dim3 block(kThreads);
    const int rank_tiles = static_cast<int>((rank_dim + kThreads - 1) / kThreads);
    const dim3 summary_grid(
        static_cast<unsigned int>(batch_size),
        static_cast<unsigned int>(num_groups),
        static_cast<unsigned int>(rank_tiles));
    latent_group_summary_reverse_kernel<work_t><<<summary_grid, block, 0, stream>>>(
        chunk_mul.data_ptr<work_t>(),
        chunk_add.data_ptr<work_t>(),
        batch_size,
        num_chunks,
        rank_dim,
        num_groups,
        group_mul.data_ptr<work_t>(),
        group_add.data_ptr<work_t>());
    latent_reverse_prefix_scan_dispatch<scalar_t, work_t>(
        group_mul,
        group_add,
        grad_final_state,
        batch_size,
        num_groups,
        rank_dim,
        stream,
        group_carry);
    const dim3 group_scan_grid(
        static_cast<unsigned int>(batch_size),
        static_cast<unsigned int>(num_groups),
        static_cast<unsigned int>(rank_dim));
    latent_chunk_group_reverse_prefix_scan_kernel<work_t><<<group_scan_grid, kChunkScanThreads, 0, stream>>>(
        chunk_mul.data_ptr<work_t>(),
        chunk_add.data_ptr<work_t>(),
        group_carry.data_ptr<work_t>(),
        batch_size,
        num_chunks,
        rank_dim,
        num_groups,
        chunk_carry.data_ptr<work_t>());
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
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int single_kernel_max_seq_len = latent_single_kernel_max_seq_len();
    const bool allow_half2 = drive.scalar_type() == torch::kHalf && ((rank_dim & 1) == 0);
    const bool allow_bfloat162 = drive.scalar_type() == torch::kBFloat16 && ((rank_dim & 1) == 0);
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, drive.scalar_type(), "latent_scan_forward_cuda", [&] {
        const bool use_half2 = allow_half2;
        const bool use_bfloat162 = allow_bfloat162;
        // Long low-precision recurrent scans are numerically fragile. Route
        // them through the chunked prefix-scan path so the recurrent carries
        // stay bounded to 64-token chunks and chunk seeds are accumulated in
        // float workspace.
        const bool use_single_kernel = seq_len <= single_kernel_max_seq_len;
        const bool use_float_workspace = use_half2 || use_bfloat162;
        const bool use_vec2_launch = use_half2 || (use_bfloat162 && use_single_kernel);
        const int rank_items = use_vec2_launch ? (rank_dim / 2) : rank_dim;
        const int launch_threads = latent_launch_threads(rank_items);
        const dim3 block(static_cast<unsigned int>(launch_threads));
        const int rank_tiles = static_cast<int>((rank_items + launch_threads - 1) / launch_threads);
        if (use_single_kernel) {
            const int worker_blocks = use_half2
                ? occupancy_persistent_worker_blocks(
                    latent_scan_forward_kernel_half2<true>,
                    drive.get_device(),
                    batch_size,
                    launch_threads)
                : use_bfloat162
                ? occupancy_persistent_worker_blocks(
                    latent_scan_forward_kernel_bfloat162<true>,
                    drive.get_device(),
                    batch_size,
                    launch_threads)
                : occupancy_persistent_worker_blocks(
                    latent_scan_forward_kernel<scalar_t, scalar_t>,
                    drive.get_device(),
                    batch_size,
                    launch_threads);
            const dim3 rank_grid(
                static_cast<unsigned int>(worker_blocks),
                static_cast<unsigned int>(rank_tiles));
            auto work_queue_counter = make_device_work_queue_counter(drive);
            if (use_half2) {
                latent_scan_forward_kernel_half2<true><<<rank_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::Half*>(drive.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(initial_state.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::Half*>(prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<c10::Half*>(states.data_ptr<scalar_t>()),
                    reinterpret_cast<c10::Half*>(final_state.data_ptr<scalar_t>()));
            } else if (use_bfloat162) {
                latent_scan_forward_kernel_bfloat162<true><<<rank_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::BFloat16*>(drive.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(decay.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(initial_state.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::BFloat16*>(prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<c10::BFloat16*>(states.data_ptr<scalar_t>()),
                    reinterpret_cast<c10::BFloat16*>(final_state.data_ptr<scalar_t>()));
            } else {
                latent_scan_forward_kernel<scalar_t, scalar_t><<<rank_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    initial_state.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    prior_states.data_ptr<scalar_t>(),
                    states.data_ptr<scalar_t>(),
                    final_state.data_ptr<scalar_t>());
            }
        } else {
            const auto workspace_dtype = use_float_workspace ? torch::kFloat32 : drive.scalar_type();
            auto [chunk_mul, chunk_add, chunk_prev] = get_or_create_latent_workspace(
                LatentWorkspaceTag::Forward, drive, batch_size, num_chunks, rank_dim, workspace_dtype);
            const int total_chunk_tasks = batch_size * num_chunks;
            const int worker_blocks = use_half2
                ? occupancy_persistent_worker_blocks(
                    latent_chunk_summary_persistent_kernel_half2,
                    drive.get_device(),
                    total_chunk_tasks,
                    launch_threads)
                : use_float_workspace
                ? occupancy_persistent_worker_blocks(
                    latent_chunk_summary_persistent_kernel<scalar_t, float>,
                    drive.get_device(),
                    total_chunk_tasks,
                    launch_threads)
                : occupancy_persistent_worker_blocks(
                    latent_chunk_summary_persistent_kernel<scalar_t, scalar_t>,
                    drive.get_device(),
                    total_chunk_tasks,
                    launch_threads);
            auto work_queue_counter = make_device_work_queue_counter(drive);
            const dim3 persistent_grid(
                static_cast<unsigned int>(worker_blocks),
                static_cast<unsigned int>(rank_tiles));
            if (use_half2) {
                latent_chunk_summary_persistent_kernel_half2<<<persistent_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::Half*>(drive.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<float>(),
                    chunk_add.data_ptr<float>());
            } else if (use_float_workspace) {
                latent_chunk_summary_persistent_kernel<scalar_t, float><<<persistent_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<float>(),
                    chunk_add.data_ptr<float>());
            } else {
                latent_chunk_summary_persistent_kernel<scalar_t, scalar_t><<<persistent_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<scalar_t>(),
                    chunk_add.data_ptr<scalar_t>());
            }
            if (use_half2) {
                latent_prefix_scan_dispatch<scalar_t, float>(
                    chunk_mul,
                    chunk_add,
                    initial_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_prev,
                    final_state);
            } else if (use_float_workspace) {
                latent_prefix_scan_dispatch<scalar_t, float>(
                    chunk_mul,
                    chunk_add,
                    initial_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_prev,
                    final_state);
            } else {
                latent_prefix_scan_dispatch<scalar_t, scalar_t>(
                    chunk_mul,
                    chunk_add,
                    initial_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_prev,
                    final_state);
            }
            if (!use_half2 && !use_float_workspace) {
                work_queue_counter.zero_();
                latent_chunk_finalize_persistent_kernel<scalar_t, scalar_t, true><<<persistent_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    chunk_prev.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    prior_states.data_ptr<scalar_t>(),
                    states.data_ptr<scalar_t>());
            } else if (use_half2) {
                work_queue_counter.zero_();
                latent_chunk_finalize_persistent_kernel_half2<true><<<persistent_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::Half*>(drive.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    chunk_prev.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::Half*>(prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<c10::Half*>(states.data_ptr<scalar_t>()));
            } else {
                work_queue_counter.zero_();
                latent_chunk_finalize_persistent_kernel<scalar_t, float, true><<<persistent_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    chunk_prev.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    prior_states.data_ptr<scalar_t>(),
                    states.data_ptr<scalar_t>());
            }
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
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int single_kernel_max_seq_len = latent_single_kernel_max_seq_len();
    const bool allow_half2 = drive.scalar_type() == torch::kHalf && ((rank_dim & 1) == 0);
    const bool allow_bfloat162 = drive.scalar_type() == torch::kBFloat16 && ((rank_dim & 1) == 0);
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, drive.scalar_type(), "latent_prior_scan_forward_cuda", [&] {
        const bool use_half2 = allow_half2;
        const bool use_bfloat162 = allow_bfloat162;
        // Match the main latent scan path: only use the fully sequential
        // kernel on short sequences, even for low-precision activations.
        const bool use_single_kernel = seq_len <= single_kernel_max_seq_len;
        const bool use_float_workspace = use_half2 || use_bfloat162;
        const bool use_vec2_launch = use_half2 || (use_bfloat162 && use_single_kernel);
        const int rank_items = use_vec2_launch ? (rank_dim / 2) : rank_dim;
        const int launch_threads = latent_launch_threads(rank_items);
        const dim3 block(static_cast<unsigned int>(launch_threads));
        const int rank_tiles = static_cast<int>((rank_items + launch_threads - 1) / launch_threads);
        if (use_single_kernel) {
            const int worker_blocks = use_half2
                ? occupancy_persistent_worker_blocks(
                    latent_scan_forward_kernel_half2<false>,
                    drive.get_device(),
                    batch_size,
                    launch_threads)
                : use_bfloat162
                ? occupancy_persistent_worker_blocks(
                    latent_scan_forward_kernel_bfloat162<false>,
                    drive.get_device(),
                    batch_size,
                    launch_threads)
                : occupancy_persistent_worker_blocks(
                    latent_prior_scan_forward_kernel<scalar_t, scalar_t>,
                    drive.get_device(),
                    batch_size,
                    launch_threads);
            const dim3 rank_grid(
                static_cast<unsigned int>(worker_blocks),
                static_cast<unsigned int>(rank_tiles));
            auto work_queue_counter = make_device_work_queue_counter(drive);
            if (use_half2) {
                latent_scan_forward_kernel_half2<false><<<rank_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::Half*>(drive.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(initial_state.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::Half*>(prior_states.data_ptr<scalar_t>()),
                    nullptr,
                    reinterpret_cast<c10::Half*>(final_state.data_ptr<scalar_t>()));
            } else if (use_bfloat162) {
                latent_scan_forward_kernel_bfloat162<false><<<rank_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::BFloat16*>(drive.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(decay.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(initial_state.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::BFloat16*>(prior_states.data_ptr<scalar_t>()),
                    nullptr,
                    reinterpret_cast<c10::BFloat16*>(final_state.data_ptr<scalar_t>()));
            } else {
                latent_prior_scan_forward_kernel<scalar_t, scalar_t><<<rank_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    initial_state.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    prior_states.data_ptr<scalar_t>(),
                    final_state.data_ptr<scalar_t>());
            }
        } else {
            const auto workspace_dtype = use_float_workspace ? torch::kFloat32 : drive.scalar_type();
            auto [chunk_mul, chunk_add, chunk_prev] = get_or_create_latent_workspace(
                LatentWorkspaceTag::ForwardPrior, drive, batch_size, num_chunks, rank_dim, workspace_dtype);
            const int total_chunk_tasks = batch_size * num_chunks;
            const int worker_blocks = use_half2
                ? occupancy_persistent_worker_blocks(
                    latent_chunk_summary_persistent_kernel_half2,
                    drive.get_device(),
                    total_chunk_tasks,
                    launch_threads)
                : use_float_workspace
                ? occupancy_persistent_worker_blocks(
                    latent_chunk_summary_persistent_kernel<scalar_t, float>,
                    drive.get_device(),
                    total_chunk_tasks,
                    launch_threads)
                : occupancy_persistent_worker_blocks(
                    latent_chunk_summary_persistent_kernel<scalar_t, scalar_t>,
                    drive.get_device(),
                    total_chunk_tasks,
                    launch_threads);
            auto work_queue_counter = make_device_work_queue_counter(drive);
            const dim3 persistent_grid(
                static_cast<unsigned int>(worker_blocks),
                static_cast<unsigned int>(rank_tiles));
            if (use_half2) {
                latent_chunk_summary_persistent_kernel_half2<<<persistent_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::Half*>(drive.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<float>(),
                    chunk_add.data_ptr<float>());
            } else if (use_float_workspace) {
                latent_chunk_summary_persistent_kernel<scalar_t, float><<<persistent_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<float>(),
                    chunk_add.data_ptr<float>());
            } else {
                latent_chunk_summary_persistent_kernel<scalar_t, scalar_t><<<persistent_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<scalar_t>(),
                    chunk_add.data_ptr<scalar_t>());
            }
            if (use_half2) {
                latent_prefix_scan_dispatch<scalar_t, float>(
                    chunk_mul,
                    chunk_add,
                    initial_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_prev,
                    final_state);
            } else if (use_float_workspace) {
                latent_prefix_scan_dispatch<scalar_t, float>(
                    chunk_mul,
                    chunk_add,
                    initial_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_prev,
                    final_state);
            } else {
                latent_prefix_scan_dispatch<scalar_t, scalar_t>(
                    chunk_mul,
                    chunk_add,
                    initial_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_prev,
                    final_state);
            }
            if (!use_half2 && !use_float_workspace) {
                work_queue_counter.zero_();
                latent_chunk_finalize_persistent_kernel<scalar_t, scalar_t, false><<<persistent_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    chunk_prev.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    prior_states.data_ptr<scalar_t>(),
                    nullptr);
            } else if (use_half2) {
                work_queue_counter.zero_();
                latent_chunk_finalize_persistent_kernel_half2<false><<<persistent_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::Half*>(drive.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    chunk_prev.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::Half*>(prior_states.data_ptr<scalar_t>()),
                    nullptr);
            } else {
                work_queue_counter.zero_();
                latent_chunk_finalize_persistent_kernel<scalar_t, float, false><<<persistent_grid, block, 0, stream>>>(
                    drive.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    chunk_prev.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    prior_states.data_ptr<scalar_t>(),
                    nullptr);
            }
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
    auto grad_decay = torch::zeros(decay.sizes(), decay.options().dtype(torch::kFloat32));
    auto grad_initial_state = torch::empty_like(initial_state);
    if (batch_size == 0 || rank_dim == 0 || seq_len == 0) {
        grad_initial_state.copy_(grad_final_state);
        return {grad_drive, grad_decay, grad_initial_state};
    }
    const int num_chunks = static_cast<int>((seq_len + kLatentChunkSize - 1) / kLatentChunkSize);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int single_kernel_max_seq_len = latent_single_kernel_max_seq_len();
    const bool allow_half2 = states.scalar_type() == torch::kHalf && ((rank_dim & 1) == 0);
    const bool allow_bfloat162 = states.scalar_type() == torch::kBFloat16 && ((rank_dim & 1) == 0);
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, states.scalar_type(), "latent_scan_backward_cuda", [&] {
        const bool use_half2 = allow_half2;
        const bool use_bfloat162 = allow_bfloat162;
        // The fused backward recurrence is especially sensitive to precision
        // loss; keep long sequences on the chunked float-workspace path.
        const bool use_single_kernel = seq_len <= single_kernel_max_seq_len;
        const bool use_float_workspace = use_half2 || use_bfloat162;
        const bool use_vec2_launch = use_half2 || (use_bfloat162 && use_single_kernel);
        const int rank_items = use_vec2_launch ? (rank_dim / 2) : rank_dim;
        const int launch_threads = latent_launch_threads(rank_items);
        const dim3 block(static_cast<unsigned int>(launch_threads));
        const int rank_tiles = static_cast<int>((rank_items + launch_threads - 1) / launch_threads);
        if (use_single_kernel) {
            const int worker_blocks = use_half2
                ? occupancy_persistent_worker_blocks(
                    latent_scan_backward_kernel_half2<true, true>,
                    states.get_device(),
                    batch_size,
                    launch_threads)
                : use_bfloat162
                ? occupancy_persistent_worker_blocks(
                    latent_scan_backward_kernel_bfloat162<true, true>,
                    states.get_device(),
                    batch_size,
                    launch_threads)
                : occupancy_persistent_worker_blocks(
                    latent_scan_backward_kernel<scalar_t, scalar_t>,
                    states.get_device(),
                    batch_size,
                    launch_threads);
            const dim3 rank_grid(
                static_cast<unsigned int>(worker_blocks),
                static_cast<unsigned int>(rank_tiles));
            auto work_queue_counter = make_device_work_queue_counter(states);
            if (use_half2) {
                latent_scan_backward_kernel_half2<true, true><<<rank_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::Half*>(grad_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(grad_prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(grad_final_state.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(states.data_ptr<scalar_t>()),
                    nullptr,
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(initial_state.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::Half*>(grad_drive.data_ptr<scalar_t>()),
                    grad_decay.data_ptr<float>(),
                    reinterpret_cast<c10::Half*>(grad_initial_state.data_ptr<scalar_t>()));
            } else if (use_bfloat162) {
                latent_scan_backward_kernel_bfloat162<true, true><<<rank_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::BFloat16*>(grad_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(grad_prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(grad_final_state.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(states.data_ptr<scalar_t>()),
                    nullptr,
                    reinterpret_cast<const c10::BFloat16*>(decay.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(initial_state.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::BFloat16*>(grad_drive.data_ptr<scalar_t>()),
                    grad_decay.data_ptr<float>(),
                    reinterpret_cast<c10::BFloat16*>(grad_initial_state.data_ptr<scalar_t>()));
            } else {
                latent_scan_backward_kernel<scalar_t, scalar_t><<<rank_grid, block, 0, stream>>>(
                    grad_states.data_ptr<scalar_t>(),
                    grad_prior_states.data_ptr<scalar_t>(),
                    grad_final_state.data_ptr<scalar_t>(),
                    states.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    initial_state.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    grad_drive.data_ptr<scalar_t>(),
                    grad_decay.data_ptr<float>(),
                    grad_initial_state.data_ptr<scalar_t>());
            }
        } else {
            const auto workspace_dtype = use_float_workspace ? torch::kFloat32 : states.scalar_type();
            auto [chunk_mul, chunk_add, chunk_carry] = get_or_create_latent_workspace(
                LatentWorkspaceTag::Backward, states, batch_size, num_chunks, rank_dim, workspace_dtype);
            const int total_chunk_tasks = batch_size * num_chunks;
            const int worker_blocks = use_half2
                ? occupancy_persistent_worker_blocks(
                    latent_chunk_backward_summary_persistent_kernel_half2<true>,
                    states.get_device(),
                    total_chunk_tasks,
                    launch_threads)
                : use_float_workspace
                ? occupancy_persistent_worker_blocks(
                    latent_chunk_backward_summary_persistent_kernel<scalar_t, float, true>,
                    states.get_device(),
                    total_chunk_tasks,
                    launch_threads)
                : occupancy_persistent_worker_blocks(
                    latent_chunk_backward_summary_persistent_kernel<scalar_t, scalar_t, true>,
                    states.get_device(),
                    total_chunk_tasks,
                    launch_threads);
            auto work_queue_counter = make_device_work_queue_counter(states);
            const dim3 persistent_grid(
                static_cast<unsigned int>(worker_blocks),
                static_cast<unsigned int>(rank_tiles));
            if (use_half2) {
                latent_chunk_backward_summary_persistent_kernel_half2<true><<<persistent_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::Half*>(grad_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(grad_prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<float>(),
                    chunk_add.data_ptr<float>());
            } else if (use_float_workspace) {
                latent_chunk_backward_summary_persistent_kernel<scalar_t, float, true><<<persistent_grid, block, 0, stream>>>(
                    grad_states.data_ptr<scalar_t>(),
                    grad_prior_states.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<float>(),
                    chunk_add.data_ptr<float>());
            } else {
                latent_chunk_backward_summary_persistent_kernel<scalar_t, scalar_t, true><<<persistent_grid, block, 0, stream>>>(
                    grad_states.data_ptr<scalar_t>(),
                    grad_prior_states.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<scalar_t>(),
                    chunk_add.data_ptr<scalar_t>());
            }
            if (use_half2) {
                latent_reverse_prefix_scan_dispatch<scalar_t, float>(
                    chunk_mul,
                    chunk_add,
                    grad_final_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_carry);
            } else if (use_float_workspace) {
                latent_reverse_prefix_scan_dispatch<scalar_t, float>(
                    chunk_mul,
                    chunk_add,
                    grad_final_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_carry);
            } else {
                latent_reverse_prefix_scan_dispatch<scalar_t, scalar_t>(
                    chunk_mul,
                    chunk_add,
                    grad_final_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_carry);
            }
            if (!use_half2 && !use_float_workspace) {
                work_queue_counter.zero_();
                latent_chunk_backward_finalize_persistent_kernel<scalar_t, scalar_t, true, true><<<persistent_grid, block, 0, stream>>>(
                    grad_states.data_ptr<scalar_t>(),
                    grad_prior_states.data_ptr<scalar_t>(),
                    states.data_ptr<scalar_t>(),
                    nullptr,
                    initial_state.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    chunk_carry.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    grad_drive.data_ptr<scalar_t>(),
                    grad_decay.data_ptr<float>(),
                    grad_initial_state.data_ptr<scalar_t>());
            } else if (use_half2) {
                work_queue_counter.zero_();
                latent_chunk_backward_finalize_persistent_kernel_half2<true, true><<<persistent_grid, block, 0, stream>>>(
                    reinterpret_cast<const c10::Half*>(grad_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(grad_prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(states.data_ptr<scalar_t>()),
                    nullptr,
                    reinterpret_cast<const c10::Half*>(initial_state.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    chunk_carry.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::Half*>(grad_drive.data_ptr<scalar_t>()),
                    grad_decay.data_ptr<float>(),
                    reinterpret_cast<c10::Half*>(grad_initial_state.data_ptr<scalar_t>()));
            } else {
                work_queue_counter.zero_();
                latent_chunk_backward_finalize_persistent_kernel<scalar_t, float, true, true><<<persistent_grid, block, 0, stream>>>(
                    grad_states.data_ptr<scalar_t>(),
                    grad_prior_states.data_ptr<scalar_t>(),
                    states.data_ptr<scalar_t>(),
                    nullptr,
                    initial_state.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    chunk_carry.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    grad_drive.data_ptr<scalar_t>(),
                    grad_decay.data_ptr<float>(),
                    grad_initial_state.data_ptr<scalar_t>());
            }
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
    auto grad_decay = torch::zeros(decay.sizes(), decay.options().dtype(torch::kFloat32));
    auto grad_initial_state = torch::empty_like(initial_state);
    if (batch_size == 0 || rank_dim == 0 || seq_len == 0) {
        grad_initial_state.copy_(grad_final_state);
        return {grad_drive, grad_decay, grad_initial_state};
    }
    const int num_chunks = static_cast<int>((seq_len + kLatentChunkSize - 1) / kLatentChunkSize);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int single_kernel_max_seq_len = latent_single_kernel_max_seq_len();
    const bool allow_half2 = prior_states.scalar_type() == torch::kHalf && ((rank_dim & 1) == 0);
    const bool allow_bfloat162 = prior_states.scalar_type() == torch::kBFloat16 && ((rank_dim & 1) == 0);
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, prior_states.scalar_type(), "latent_prior_scan_backward_cuda", [&] {
        const bool use_half2 = allow_half2;
        const bool use_bfloat162 = allow_bfloat162;
        // Keep prior-only backward aligned with the main latent backward path.
        const bool use_single_kernel = seq_len <= single_kernel_max_seq_len;
        const bool use_float_workspace = use_half2 || use_bfloat162;
        const bool use_vec2_launch = use_half2 || (use_bfloat162 && use_single_kernel);
        const int rank_items = use_vec2_launch ? (rank_dim / 2) : rank_dim;
        const int launch_threads = latent_launch_threads(rank_items);
        const dim3 block(static_cast<unsigned int>(launch_threads));
        const int rank_tiles = static_cast<int>((rank_items + launch_threads - 1) / launch_threads);
        if (use_single_kernel) {
            const int worker_blocks = use_half2
                ? occupancy_persistent_worker_blocks(
                    latent_scan_backward_kernel_half2<false, false>,
                    prior_states.get_device(),
                    batch_size,
                    launch_threads)
                : use_bfloat162
                ? occupancy_persistent_worker_blocks(
                    latent_scan_backward_kernel_bfloat162<false, false>,
                    prior_states.get_device(),
                    batch_size,
                    launch_threads)
                : occupancy_persistent_worker_blocks(
                    latent_prior_scan_backward_kernel<scalar_t, scalar_t>,
                    prior_states.get_device(),
                    batch_size,
                    launch_threads);
            const dim3 rank_grid(
                static_cast<unsigned int>(worker_blocks),
                static_cast<unsigned int>(rank_tiles));
            auto work_queue_counter = make_device_work_queue_counter(prior_states);
            if (use_half2) {
                latent_scan_backward_kernel_half2<false, false><<<rank_grid, block, 0, stream>>>(
                    nullptr,
                    reinterpret_cast<const c10::Half*>(grad_prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(grad_final_state.data_ptr<scalar_t>()),
                    nullptr,
                    reinterpret_cast<const c10::Half*>(prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(initial_state.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::Half*>(grad_drive.data_ptr<scalar_t>()),
                    grad_decay.data_ptr<float>(),
                    reinterpret_cast<c10::Half*>(grad_initial_state.data_ptr<scalar_t>()));
            } else if (use_bfloat162) {
                latent_scan_backward_kernel_bfloat162<false, false><<<rank_grid, block, 0, stream>>>(
                    nullptr,
                    reinterpret_cast<const c10::BFloat16*>(grad_prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(grad_final_state.data_ptr<scalar_t>()),
                    nullptr,
                    reinterpret_cast<const c10::BFloat16*>(prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(decay.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::BFloat16*>(initial_state.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::BFloat16*>(grad_drive.data_ptr<scalar_t>()),
                    grad_decay.data_ptr<float>(),
                    reinterpret_cast<c10::BFloat16*>(grad_initial_state.data_ptr<scalar_t>()));
            } else {
                latent_prior_scan_backward_kernel<scalar_t, scalar_t><<<rank_grid, block, 0, stream>>>(
                    grad_prior_states.data_ptr<scalar_t>(),
                    grad_final_state.data_ptr<scalar_t>(),
                    prior_states.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    initial_state.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    batch_size,
                    work_queue_counter.data_ptr<int32_t>(),
                    grad_drive.data_ptr<scalar_t>(),
                    grad_decay.data_ptr<float>(),
                    grad_initial_state.data_ptr<scalar_t>());
            }
        } else {
            const auto workspace_dtype = use_float_workspace ? torch::kFloat32 : prior_states.scalar_type();
            auto [chunk_mul, chunk_add, chunk_carry] = get_or_create_latent_workspace(
                LatentWorkspaceTag::BackwardPrior, prior_states, batch_size, num_chunks, rank_dim, workspace_dtype);
            const int total_chunk_tasks = batch_size * num_chunks;
            const int worker_blocks = use_half2
                ? occupancy_persistent_worker_blocks(
                    latent_chunk_backward_summary_persistent_kernel_half2<false>,
                    prior_states.get_device(),
                    total_chunk_tasks,
                    launch_threads)
                : use_float_workspace
                ? occupancy_persistent_worker_blocks(
                    latent_chunk_backward_summary_persistent_kernel<scalar_t, float, false>,
                    prior_states.get_device(),
                    total_chunk_tasks,
                    launch_threads)
                : occupancy_persistent_worker_blocks(
                    latent_chunk_backward_summary_persistent_kernel<scalar_t, scalar_t, false>,
                    prior_states.get_device(),
                    total_chunk_tasks,
                    launch_threads);
            auto work_queue_counter = make_device_work_queue_counter(prior_states);
            const dim3 persistent_grid(
                static_cast<unsigned int>(worker_blocks),
                static_cast<unsigned int>(rank_tiles));
            if (use_half2) {
                latent_chunk_backward_summary_persistent_kernel_half2<false><<<persistent_grid, block, 0, stream>>>(
                    nullptr,
                    reinterpret_cast<const c10::Half*>(grad_prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<float>(),
                    chunk_add.data_ptr<float>());
            } else if (use_float_workspace) {
                latent_chunk_backward_summary_persistent_kernel<scalar_t, float, false><<<persistent_grid, block, 0, stream>>>(
                    nullptr,
                    grad_prior_states.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<float>(),
                    chunk_add.data_ptr<float>());
            } else {
                latent_chunk_backward_summary_persistent_kernel<scalar_t, scalar_t, false><<<persistent_grid, block, 0, stream>>>(
                    nullptr,
                    grad_prior_states.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    chunk_mul.data_ptr<scalar_t>(),
                    chunk_add.data_ptr<scalar_t>());
            }
            if (use_half2) {
                latent_reverse_prefix_scan_dispatch<scalar_t, float>(
                    chunk_mul,
                    chunk_add,
                    grad_final_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_carry);
            } else if (use_float_workspace) {
                latent_reverse_prefix_scan_dispatch<scalar_t, float>(
                    chunk_mul,
                    chunk_add,
                    grad_final_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_carry);
            } else {
                latent_reverse_prefix_scan_dispatch<scalar_t, scalar_t>(
                    chunk_mul,
                    chunk_add,
                    grad_final_state,
                    batch_size,
                    num_chunks,
                    rank_dim,
                    stream,
                    chunk_carry);
            }
            if (!use_half2 && !use_float_workspace) {
                work_queue_counter.zero_();
                latent_chunk_backward_finalize_persistent_kernel<scalar_t, scalar_t, false, false><<<persistent_grid, block, 0, stream>>>(
                    nullptr,
                    grad_prior_states.data_ptr<scalar_t>(),
                    nullptr,
                    prior_states.data_ptr<scalar_t>(),
                    initial_state.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    chunk_carry.data_ptr<scalar_t>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    grad_drive.data_ptr<scalar_t>(),
                    grad_decay.data_ptr<float>(),
                    grad_initial_state.data_ptr<scalar_t>());
            } else if (use_half2) {
                work_queue_counter.zero_();
                latent_chunk_backward_finalize_persistent_kernel_half2<false, false><<<persistent_grid, block, 0, stream>>>(
                    nullptr,
                    reinterpret_cast<const c10::Half*>(grad_prior_states.data_ptr<scalar_t>()),
                    nullptr,
                    reinterpret_cast<const c10::Half*>(prior_states.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(initial_state.data_ptr<scalar_t>()),
                    reinterpret_cast<const c10::Half*>(decay.data_ptr<scalar_t>()),
                    chunk_carry.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    reinterpret_cast<c10::Half*>(grad_drive.data_ptr<scalar_t>()),
                    grad_decay.data_ptr<float>(),
                    reinterpret_cast<c10::Half*>(grad_initial_state.data_ptr<scalar_t>()));
            } else {
                work_queue_counter.zero_();
                latent_chunk_backward_finalize_persistent_kernel<scalar_t, float, false, false><<<persistent_grid, block, 0, stream>>>(
                    nullptr,
                    grad_prior_states.data_ptr<scalar_t>(),
                    nullptr,
                    prior_states.data_ptr<scalar_t>(),
                    initial_state.data_ptr<scalar_t>(),
                    decay.data_ptr<scalar_t>(),
                    chunk_carry.data_ptr<float>(),
                    seq_len,
                    rank_dim,
                    num_chunks,
                    total_chunk_tasks,
                    work_queue_counter.data_ptr<int32_t>(),
                    grad_drive.data_ptr<scalar_t>(),
                    grad_decay.data_ptr<float>(),
                    grad_initial_state.data_ptr<scalar_t>());
            }
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_drive, grad_decay, grad_initial_state};
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void latent_replace_forward_kernel(
    const scalar_t* __restrict__ local_logits,
    const scalar_t* __restrict__ prior_logits,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ token_gate,
    const scalar_t* __restrict__ pred_scale,
    int64_t rows,
    int64_t num_states,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ prior_log_beliefs) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int64_t base = row * num_states;
    __shared__ float shared[kThreads];
    const float gate = load_as_float(token_gate + row);
    const float scale = fmaxf(load_as_float(pred_scale + row), 1.0e-4f);

    float thread_max_filtered = -std::numeric_limits<float>::infinity();
    float thread_max_prior = -std::numeric_limits<float>::infinity();
    for (int64_t state = tid; state < num_states; state += blockDim.x) {
        const float raw_unscaled = load_as_float(prior_logits + base + state) + load_as_float(transition_context + base + state);
        const float raw = raw_unscaled / scale;
        const float filtered = load_as_float(local_logits + base + state) + gate * raw;
        thread_max_filtered = fmaxf(thread_max_filtered, filtered);
        thread_max_prior = fmaxf(thread_max_prior, raw);
    }
    shared[tid] = thread_max_filtered;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }
    const float row_max_filtered = shared[0];
    shared[tid] = thread_max_prior;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }
    const float row_max_prior = shared[0];

    float thread_sum_filtered = 0.0f;
    float thread_sum_prior = 0.0f;
    for (int64_t state = tid; state < num_states; state += blockDim.x) {
        const float raw_unscaled = load_as_float(prior_logits + base + state) + load_as_float(transition_context + base + state);
        const float raw = raw_unscaled / scale;
        const float filtered = load_as_float(local_logits + base + state) + gate * raw;
        thread_sum_filtered += expf(filtered - row_max_filtered);
        thread_sum_prior += expf(raw - row_max_prior);
    }
    shared[tid] = thread_sum_filtered;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    const float row_logsum_filtered = row_max_filtered + logf(fmaxf(shared[0], 1.0e-20f));
    shared[tid] = thread_sum_prior;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    const float row_logsum_prior = row_max_prior + logf(fmaxf(shared[0], 1.0e-20f));

    for (int64_t state = tid; state < num_states; state += blockDim.x) {
        const float raw_unscaled = load_as_float(prior_logits + base + state) + load_as_float(transition_context + base + state);
        const float raw = raw_unscaled / scale;
        const float filtered = load_as_float(local_logits + base + state) + gate * raw;
        beliefs[base + state] = store_from_float<scalar_t>(filtered - row_logsum_filtered);
        prior_log_beliefs[base + state] = store_from_float<scalar_t>(raw - row_logsum_prior);
    }
}

constexpr int kLatentReplaceExactNumStates = 128;
constexpr int kLatentReplaceExactThreads = 128;

template <int kBlockThreads>
__device__ __forceinline__ float latent_replace_block_reduce_sum(float value, float* shared, int tid) {
    shared[tid] = value;
    __syncthreads();
    for (int offset = kBlockThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    const float reduced = shared[0];
    __syncthreads();
    return reduced;
}

template <int kBlockThreads>
__device__ __forceinline__ float latent_replace_block_reduce_max(float value, float* shared, int tid) {
    shared[tid] = value;
    __syncthreads();
    for (int offset = kBlockThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }
    const float reduced = shared[0];
    __syncthreads();
    return reduced;
}

template <typename scalar_t>
__global__ __launch_bounds__(kLatentReplaceExactThreads) void latent_replace_forward_kernel_128(
    const scalar_t* __restrict__ local_logits,
    const scalar_t* __restrict__ prior_logits,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ token_gate,
    const scalar_t* __restrict__ pred_scale,
    int64_t rows,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ prior_log_beliefs) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    constexpr int kNumStates = kLatentReplaceExactNumStates;
    const int tid = static_cast<int>(threadIdx.x);
    const int64_t base = row * kNumStates;
    __shared__ float shared[kLatentReplaceExactThreads];
    const float gate = load_as_float(token_gate + row);
    const float scale = fmaxf(load_as_float(pred_scale + row), 1.0e-4f);
    const float raw_unscaled = load_as_float(prior_logits + base + tid) + load_as_float(transition_context + base + tid);
    const float raw = raw_unscaled / scale;
    const float filtered = load_as_float(local_logits + base + tid) + gate * raw;
    const float row_max_filtered = latent_replace_block_reduce_max<kLatentReplaceExactThreads>(filtered, shared, tid);
    const float row_max_prior = latent_replace_block_reduce_max<kLatentReplaceExactThreads>(raw, shared, tid);
    const float filtered_exp = expf(filtered - row_max_filtered);
    const float prior_exp = expf(raw - row_max_prior);
    const float filtered_sum = latent_replace_block_reduce_sum<kLatentReplaceExactThreads>(filtered_exp, shared, tid);
    const float prior_sum = latent_replace_block_reduce_sum<kLatentReplaceExactThreads>(prior_exp, shared, tid);
    const float row_logsum_filtered = row_max_filtered + logf(fmaxf(filtered_sum, 1.0e-20f));
    const float row_logsum_prior = row_max_prior + logf(fmaxf(prior_sum, 1.0e-20f));
    beliefs[base + tid] = store_from_float<scalar_t>(filtered - row_logsum_filtered);
    prior_log_beliefs[base + tid] = store_from_float<scalar_t>(raw - row_logsum_prior);
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void latent_replace_backward_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_prior_log_beliefs,
    const scalar_t* __restrict__ prior_logits,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ token_gate,
    const scalar_t* __restrict__ pred_scale,
    const scalar_t* __restrict__ beliefs,
    const scalar_t* __restrict__ prior_log_beliefs,
    int64_t rows,
    int64_t num_states,
    scalar_t* __restrict__ grad_local_logits,
    scalar_t* __restrict__ grad_prior_logits,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_token_gate,
    scalar_t* __restrict__ grad_pred_scale) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int64_t base = row * num_states;
    __shared__ float shared[kThreads];
    const float gate = load_as_float(token_gate + row);
    const float scale = fmaxf(load_as_float(pred_scale + row), 1.0e-4f);
    const float inv_scale = 1.0f / scale;
    const float inv_scale_sq = inv_scale * inv_scale;

    float thread_sum_grad_beliefs = 0.0f;
    float thread_sum_grad_prior = 0.0f;
    for (int64_t state = tid; state < num_states; state += blockDim.x) {
        thread_sum_grad_beliefs += load_as_float(grad_beliefs + base + state);
        thread_sum_grad_prior += load_as_float(grad_prior_log_beliefs + base + state);
    }
    shared[tid] = thread_sum_grad_beliefs;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    const float row_sum_grad_beliefs = shared[0];
    shared[tid] = thread_sum_grad_prior;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    const float row_sum_grad_prior = shared[0];

    float thread_grad_gate = 0.0f;
    float thread_grad_scale = 0.0f;
    for (int64_t state = tid; state < num_states; state += blockDim.x) {
        const float belief_log = load_as_float(beliefs + base + state);
        const float prior_log = load_as_float(prior_log_beliefs + base + state);
        const float grad_belief = load_as_float(grad_beliefs + base + state);
        const float grad_prior_log = load_as_float(grad_prior_log_beliefs + base + state);
        const float raw_unscaled = load_as_float(prior_logits + base + state) + load_as_float(transition_context + base + state);
        const float raw = raw_unscaled * inv_scale;
        const float grad_filtered = grad_belief - expf(belief_log) * row_sum_grad_beliefs;
        const float grad_prior_row = grad_prior_log - expf(prior_log) * row_sum_grad_prior;
        const float grad_raw = gate * grad_filtered + grad_prior_row;
        const float grad_unscaled = grad_raw * inv_scale;
        grad_local_logits[base + state] = store_from_float<scalar_t>(grad_filtered);
        grad_prior_logits[base + state] = store_from_float<scalar_t>(grad_unscaled);
        grad_transition_context[base + state] = store_from_float<scalar_t>(grad_unscaled);
        thread_grad_gate += grad_filtered * raw;
        thread_grad_scale += -grad_raw * raw_unscaled * inv_scale_sq;
    }
    shared[tid] = thread_grad_gate;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        grad_token_gate[row] = store_from_float<scalar_t>(shared[0]);
    }
    shared[tid] = thread_grad_scale;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        grad_pred_scale[row] = store_from_float<scalar_t>(shared[0]);
    }
}

template <typename scalar_t>
__global__ __launch_bounds__(kLatentReplaceExactThreads) void latent_replace_backward_kernel_128(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_prior_log_beliefs,
    const scalar_t* __restrict__ prior_logits,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ token_gate,
    const scalar_t* __restrict__ pred_scale,
    const scalar_t* __restrict__ beliefs,
    const scalar_t* __restrict__ prior_log_beliefs,
    int64_t rows,
    scalar_t* __restrict__ grad_local_logits,
    scalar_t* __restrict__ grad_prior_logits,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_token_gate,
    scalar_t* __restrict__ grad_pred_scale) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    constexpr int kNumStates = kLatentReplaceExactNumStates;
    const int tid = static_cast<int>(threadIdx.x);
    const int64_t base = row * kNumStates;
    __shared__ float shared[kLatentReplaceExactThreads];
    const float gate = load_as_float(token_gate + row);
    const float scale = fmaxf(load_as_float(pred_scale + row), 1.0e-4f);
    const float inv_scale = 1.0f / scale;
    const float inv_scale_sq = inv_scale * inv_scale;
    const float row_sum_grad_beliefs = latent_replace_block_reduce_sum<kLatentReplaceExactThreads>(
        load_as_float(grad_beliefs + base + tid),
        shared,
        tid);
    const float row_sum_grad_prior = latent_replace_block_reduce_sum<kLatentReplaceExactThreads>(
        load_as_float(grad_prior_log_beliefs + base + tid),
        shared,
        tid);
    const float belief_log = load_as_float(beliefs + base + tid);
    const float prior_log = load_as_float(prior_log_beliefs + base + tid);
    const float grad_belief = load_as_float(grad_beliefs + base + tid);
    const float grad_prior_log = load_as_float(grad_prior_log_beliefs + base + tid);
    const float raw_unscaled = load_as_float(prior_logits + base + tid) + load_as_float(transition_context + base + tid);
    const float raw = raw_unscaled * inv_scale;
    const float grad_filtered = grad_belief - expf(belief_log) * row_sum_grad_beliefs;
    const float grad_prior_row = grad_prior_log - expf(prior_log) * row_sum_grad_prior;
    const float grad_raw = gate * grad_filtered + grad_prior_row;
    const float grad_unscaled = grad_raw * inv_scale;
    grad_local_logits[base + tid] = store_from_float<scalar_t>(grad_filtered);
    grad_prior_logits[base + tid] = store_from_float<scalar_t>(grad_unscaled);
    grad_transition_context[base + tid] = store_from_float<scalar_t>(grad_unscaled);
    const float gate_grad = grad_filtered * raw;
    const float scale_grad = -grad_raw * raw_unscaled * inv_scale_sq;
    const float row_gate_grad = latent_replace_block_reduce_sum<kLatentReplaceExactThreads>(gate_grad, shared, tid);
    const float row_scale_grad = latent_replace_block_reduce_sum<kLatentReplaceExactThreads>(scale_grad, shared, tid);
    if (tid == 0) {
        grad_token_gate[row] = store_from_float<scalar_t>(row_gate_grad);
        grad_pred_scale[row] = store_from_float<scalar_t>(row_scale_grad);
    }
}

std::vector<torch::Tensor> causal_machine_latent_replace_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor prior_logits,
    torch::Tensor transition_context,
    torch::Tensor token_gate,
    torch::Tensor pred_scale) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(local_logits));
    const int64_t batch_size = local_logits.size(0);
    const int64_t seq_len = local_logits.size(1);
    const int64_t num_states = local_logits.size(2);
    auto beliefs = torch::empty_like(local_logits);
    auto prior_log_beliefs = torch::empty_like(local_logits);
    if (batch_size == 0 || seq_len == 0 || num_states == 0) {
        return {beliefs, prior_log_beliefs};
    }
    const int64_t rows = batch_size * seq_len;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid(static_cast<unsigned int>(rows));
    const bool use_exact_128 = num_states == kLatentReplaceExactNumStates;
    const dim3 block(static_cast<unsigned int>(use_exact_128 ? kLatentReplaceExactThreads : kThreads));
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, local_logits.scalar_type(), "latent_replace_forward_cuda", [&] {
        if (use_exact_128) {
            latent_replace_forward_kernel_128<scalar_t><<<grid, block, 0, stream>>>(
                local_logits.data_ptr<scalar_t>(),
                prior_logits.data_ptr<scalar_t>(),
                transition_context.data_ptr<scalar_t>(),
                token_gate.data_ptr<scalar_t>(),
                pred_scale.data_ptr<scalar_t>(),
                rows,
                beliefs.data_ptr<scalar_t>(),
                prior_log_beliefs.data_ptr<scalar_t>());
        } else {
            latent_replace_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
                local_logits.data_ptr<scalar_t>(),
                prior_logits.data_ptr<scalar_t>(),
                transition_context.data_ptr<scalar_t>(),
                token_gate.data_ptr<scalar_t>(),
                pred_scale.data_ptr<scalar_t>(),
                rows,
                num_states,
                beliefs.data_ptr<scalar_t>(),
                prior_log_beliefs.data_ptr<scalar_t>());
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {beliefs, prior_log_beliefs};
}

std::vector<torch::Tensor> causal_machine_latent_replace_backward_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_prior_log_beliefs,
    torch::Tensor prior_logits,
    torch::Tensor transition_context,
    torch::Tensor token_gate,
    torch::Tensor pred_scale,
    torch::Tensor beliefs,
    torch::Tensor prior_log_beliefs) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(beliefs));
    const int64_t batch_size = beliefs.size(0);
    const int64_t seq_len = beliefs.size(1);
    const int64_t num_states = beliefs.size(2);
    auto grad_local_logits = torch::empty_like(beliefs);
    auto grad_prior_logits = torch::empty_like(beliefs);
    auto grad_transition_context = torch::empty_like(beliefs);
    auto grad_token_gate = torch::empty_like(token_gate);
    auto grad_pred_scale = torch::empty_like(pred_scale);
    if (batch_size == 0 || seq_len == 0 || num_states == 0) {
        return {
            grad_local_logits,
            grad_prior_logits,
            grad_transition_context,
            grad_token_gate,
            grad_pred_scale};
    }
    const int64_t rows = batch_size * seq_len;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid(static_cast<unsigned int>(rows));
    const bool use_exact_128 = num_states == kLatentReplaceExactNumStates;
    const dim3 block(static_cast<unsigned int>(use_exact_128 ? kLatentReplaceExactThreads : kThreads));
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, beliefs.scalar_type(), "latent_replace_backward_cuda", [&] {
        if (use_exact_128) {
            latent_replace_backward_kernel_128<scalar_t><<<grid, block, 0, stream>>>(
                grad_beliefs.data_ptr<scalar_t>(),
                grad_prior_log_beliefs.data_ptr<scalar_t>(),
                prior_logits.data_ptr<scalar_t>(),
                transition_context.data_ptr<scalar_t>(),
                token_gate.data_ptr<scalar_t>(),
                pred_scale.data_ptr<scalar_t>(),
                beliefs.data_ptr<scalar_t>(),
                prior_log_beliefs.data_ptr<scalar_t>(),
                rows,
                grad_local_logits.data_ptr<scalar_t>(),
                grad_prior_logits.data_ptr<scalar_t>(),
                grad_transition_context.data_ptr<scalar_t>(),
                grad_token_gate.data_ptr<scalar_t>(),
                grad_pred_scale.data_ptr<scalar_t>());
        } else {
            latent_replace_backward_kernel<scalar_t><<<grid, block, 0, stream>>>(
                grad_beliefs.data_ptr<scalar_t>(),
                grad_prior_log_beliefs.data_ptr<scalar_t>(),
                prior_logits.data_ptr<scalar_t>(),
                transition_context.data_ptr<scalar_t>(),
                token_gate.data_ptr<scalar_t>(),
                pred_scale.data_ptr<scalar_t>(),
                beliefs.data_ptr<scalar_t>(),
                prior_log_beliefs.data_ptr<scalar_t>(),
                rows,
                num_states,
                grad_local_logits.data_ptr<scalar_t>(),
                grad_prior_logits.data_ptr<scalar_t>(),
                grad_transition_context.data_ptr<scalar_t>(),
                grad_token_gate.data_ptr<scalar_t>(),
                grad_pred_scale.data_ptr<scalar_t>());
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_local_logits,
        grad_prior_logits,
        grad_transition_context,
        grad_token_gate,
        grad_pred_scale};
}
