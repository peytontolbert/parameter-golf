#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <limits>
#include <mutex>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

namespace {

// Small-state structured-scan fast path: kernels support arbitrary active
// state counts up to 128 via the dynamic launch-width path, with preferred
// tuned widths at 64/96/128.
constexpr int kMaxNumStates = 128;
constexpr int kWarpSize = 32;
constexpr int kMaxNumWarps = kMaxNumStates / kWarpSize;
constexpr int kMaskedBackwardRankChunk = 16;
constexpr int kAsyncTileRankChunk = 16;
constexpr int kSm90AsyncTileRankChunk = 32;
constexpr int kMaxTiledBlockThreads = 256;
constexpr int kAsyncTilePipelineStages = 2;
constexpr int kSm90AsyncTilePipelineStages = 3;
constexpr int kTensorCoreTile = 16;

static_assert(kMaxNumStates % kWarpSize == 0, "small-state kernels assume warp-aligned block sizes");
static_assert(kMaxTiledBlockThreads % kWarpSize == 0, "tiled kernels assume warp-aligned block sizes");
static_assert(kAsyncTilePipelineStages >= 2, "tiled async path expects at least double buffering");
static_assert(kSm90AsyncTilePipelineStages >= kAsyncTilePipelineStages, "sm90 tiled path should not use fewer async stages");

#define CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS __launch_bounds__(kMaxNumStates, 2)
#define CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS __launch_bounds__(kMaxTiledBlockThreads, 1)

int scan_single_launch_max_seq_len() {
    // Keep the structured scan resident on device for the full sequence.
    // The old seq_len>512 fallback relaunched per chunk from the host and
    // reloaded transition tables on every launch.
    return std::numeric_limits<int>::max();
}

int cached_max_optin_bytes(int device_index) {
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
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device_index));
    cache.emplace(device_index, value);
    return value;
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

int cached_capability_major(int device_index) {
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
        cudaDevAttrComputeCapabilityMajor,
        device_index));
    cache.emplace(device_index, value);
    return value;
}

int cached_capability_minor(int device_index) {
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
        cudaDevAttrComputeCapabilityMinor,
        device_index));
    cache.emplace(device_index, value);
    return value;
}

int cached_l2_cache_size(int device_index) {
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
        cudaDevAttrL2CacheSize,
        device_index));
    cache.emplace(device_index, value);
    return value;
}

int cached_persisting_l2_cache_max_size(int device_index) {
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
        cudaDevAttrMaxPersistingL2CacheSize,
        device_index));
    cache.emplace(device_index, value);
    return value;
}

bool supports_persisting_l2_window(int device_index) {
    return cached_capability_major(device_index) >= 8
        && cached_persisting_l2_cache_max_size(device_index) > 0
        && cached_l2_cache_size(device_index) > 0;
}

int persistent_worker_blocks(int device_index, int total_batches) {
    if (total_batches <= 0) {
        return 0;
    }
    const int sm_count = std::max(cached_sm_count(device_index), 1);
    const int target_blocks = sm_count * 2;
    return std::max(1, std::min(total_batches, target_blocks));
}

template <typename KernelT>
int occupancy_persistent_worker_blocks(
    KernelT kernel,
    int device_index,
    int total_tasks,
    int block_threads,
    size_t dynamic_smem_bytes) {
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

size_t backward_chunk_shared_bytes(int num_states, int transition_rank, bool direct_grad_reduce) {
    const int num_warps = (num_states + kWarpSize - 1) / kWarpSize;
    const size_t base_words = static_cast<size_t>(
        (2 * num_states * transition_rank) + (2 * num_states) + (2 * transition_rank) + num_warps
    );
    const size_t direct_words = direct_grad_reduce
        ? static_cast<size_t>((2 * num_states * transition_rank) + num_states)
        : 0;
    return (base_words + direct_words) * sizeof(float);
}

size_t forward_dense_128_rank8_shared_bytes() {
    constexpr int kNumStates = 128;
    constexpr int kTransitionRank = 8;
    constexpr int kNumWarps = kNumStates / kWarpSize;
    return static_cast<size_t>(
        (2 * kNumStates * kTransitionRank) + kTransitionRank + (kNumWarps * kTransitionRank)
    ) * sizeof(float);
}

size_t backward_dense_128_rank8_shared_bytes(bool direct_grad_reduce) {
    constexpr int kNumStates = 128;
    constexpr int kTransitionRank = 8;
    constexpr int kNumWarps = kNumStates / kWarpSize;
    const size_t base_words = static_cast<size_t>(
        (2 * kNumStates * kTransitionRank) + kTransitionRank + kNumStates + kTransitionRank + (kNumWarps * kTransitionRank)
    );
    const size_t direct_words = direct_grad_reduce
        ? static_cast<size_t>((2 * kNumStates * kTransitionRank) + kNumStates)
        : 0;
    return (base_words + direct_words) * sizeof(float);
}

size_t backward_packed_chunk_shared_bytes(int num_states, int transition_rank, bool direct_grad_reduce) {
    const int num_warps = (num_states + kWarpSize - 1) / kWarpSize;
    const size_t base_words = static_cast<size_t>(
        (2 * num_states) + (2 * transition_rank) + num_warps
    );
    const size_t direct_words = direct_grad_reduce
        ? static_cast<size_t>((2 * num_states * transition_rank) + num_states)
        : 0;
    size_t bytes = (base_words + direct_words) * sizeof(float);
    bytes += static_cast<size_t>(2 * kTensorCoreTile * kTensorCoreTile) * sizeof(__half);
    bytes += static_cast<size_t>(2 * kTensorCoreTile * kTensorCoreTile) * sizeof(float);
    bytes += 64;
    return bytes;
}

bool can_use_direct_grad_reduce(int device_index, int num_states, int transition_rank) {
    if (transition_rank <= 0 || transition_rank > num_states) {
        return false;
    }
    const int max_optin_bytes = cached_max_optin_bytes(device_index);
    return backward_chunk_shared_bytes(num_states, transition_rank, true) <= static_cast<size_t>(max_optin_bytes);
}

size_t forward_chunk_shared_bytes(int num_states, int transition_rank) {
    const int num_warps = (num_states + kWarpSize - 1) / kWarpSize;
    return static_cast<size_t>(
        (2 * num_states * transition_rank) + num_states + transition_rank + (num_warps * transition_rank)
    ) * sizeof(float);
}

size_t forward_packed_chunk_shared_bytes(int num_states, int transition_rank) {
    const int num_warps = (num_states + kWarpSize - 1) / kWarpSize;
    size_t bytes = static_cast<size_t>(num_states + transition_rank + num_warps) * sizeof(float);
    bytes += static_cast<size_t>(2 * kTensorCoreTile * kTensorCoreTile) * sizeof(__half);
    bytes += static_cast<size_t>(2 * kTensorCoreTile * kTensorCoreTile) * sizeof(float);
    bytes += 64;
    return bytes;
}

size_t forward_masked_dense_shared_bytes(int num_states) {
    return static_cast<size_t>((num_states * num_states) + num_states + kMaxNumWarps) * sizeof(float);
}

size_t backward_masked_dense_shared_bytes(int num_states) {
    return static_cast<size_t>((num_states * num_states) + (3 * num_states) + kMaxNumWarps) * sizeof(float);
}

struct ScanChunkSchedulerConfig {
    int64_t seq_len;
    int64_t requested_chunk_size;
    int64_t launch_chunk_size;
    int64_t num_launches;
    bool use_single_launch;
    bool reverse_launch_order;
    bool use_persisting_l2_window;
    bool persistent_device_loop;
};

struct ScanKernelLaunchConfig {
    dim3 grid;
    dim3 block;
    int transition_rank;
    size_t shared_bytes;
    int device_index;
    bool direct_grad_reduce;
    int total_tasks;
};

struct SparseScanKernelLaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_bytes;
    int device_index;
    int total_tasks;
};

struct TiledScanKernelLaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_bytes;
    int device_index;
    int total_tasks;
};

struct TiledBackwardRuntimeConfig {
    int64_t launch_batch_size;
    int64_t staging_worker_blocks;
    int64_t staging_budget_bytes;
    int64_t per_worker_bytes;
};

int64_t ceil_div_int64(int64_t numerator, int64_t denominator) {
    TORCH_CHECK(denominator > 0, "denominator must be positive");
    return (numerator + denominator - 1) / denominator;
}

int round_up_pow2(int value) {
    int rounded = 1;
    while (rounded < value) {
        rounded <<= 1;
    }
    return rounded;
}

template <bool Sm90Path>
constexpr int async_tile_pipeline_stages() {
    return Sm90Path ? kSm90AsyncTilePipelineStages : kAsyncTilePipelineStages;
}

ScanChunkSchedulerConfig make_scan_chunk_scheduler(
    int64_t seq_len,
    int64_t chunk_size,
    bool reverse_launch_order = false) {
    TORCH_CHECK(seq_len >= 0, "seq_len must be non-negative");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    const bool use_single_launch = true;
    const int64_t launch_chunk_size = seq_len == 0 ? 0 : std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    const int64_t num_launches = seq_len == 0 ? 0 : 1;
    return {
        seq_len,
        chunk_size,
        launch_chunk_size,
        num_launches,
        use_single_launch,
        reverse_launch_order,
        false,
        seq_len > 0,
    };
}

void validate_dynamic_smem_bytes(
    const char* op_name,
    size_t shared_bytes,
    int device_index) {
    const int max_optin_bytes = cached_max_optin_bytes(device_index);
    TORCH_CHECK(
        shared_bytes <= static_cast<size_t>(max_optin_bytes),
        op_name,
        " requires ",
        shared_bytes,
        " bytes of dynamic shared memory, but device only supports ",
        max_optin_bytes,
        " bytes. Reduce CAUSAL_MACHINE_TRANSITION_RANK, disable USE_CAUSAL_MACHINE_CUDA_SCAN, or use a higher-SMEM GPU."
    );
}

ScanKernelLaunchConfig make_forward_launch_config(
    const torch::Tensor& local_logits,
    int transition_rank) {
    const int num_states = static_cast<int>(local_logits.size(2));
    const int device_index = local_logits.get_device();
    const int total_batches = static_cast<int>(local_logits.size(0));
    return {
        dim3(1),
        dim3(num_states),
        transition_rank,
        forward_chunk_shared_bytes(num_states, transition_rank),
        device_index,
        false,
        total_batches,
    };
}

ScanKernelLaunchConfig make_forward_packed_launch_config(
    const torch::Tensor& local_logits,
    int transition_rank) {
    const int num_states = static_cast<int>(local_logits.size(2));
    const int device_index = local_logits.get_device();
    const int total_batches = static_cast<int>(local_logits.size(0));
    return {
        dim3(1),
        dim3(num_states),
        transition_rank,
        forward_packed_chunk_shared_bytes(num_states, transition_rank),
        device_index,
        false,
        total_batches,
    };
}

ScanKernelLaunchConfig make_forward_masked_launch_config(
    const torch::Tensor& local_logits) {
    const int num_states = static_cast<int>(local_logits.size(2));
    const int device_index = local_logits.get_device();
    const int total_batches = static_cast<int>(local_logits.size(0));
    return {
        dim3(1),
        dim3(num_states),
        0,
        forward_masked_dense_shared_bytes(num_states),
        device_index,
        false,
        total_batches,
    };
}

ScanKernelLaunchConfig make_backward_masked_launch_config(
    const torch::Tensor& beliefs) {
    const int num_states = static_cast<int>(beliefs.size(2));
    const int device_index = beliefs.get_device();
    const int total_batches = static_cast<int>(beliefs.size(0));
    return {
        dim3(1),
        dim3(num_states),
        0,
        backward_masked_dense_shared_bytes(num_states),
        device_index,
        false,
        total_batches,
    };
}

ScanKernelLaunchConfig make_backward_launch_config(
    const torch::Tensor& beliefs,
    int transition_rank,
    bool direct_grad_reduce) {
    const int num_states = static_cast<int>(beliefs.size(2));
    const int device_index = beliefs.get_device();
    const int total_batches = static_cast<int>(beliefs.size(0));
    return {
        dim3(1),
        dim3(num_states),
        transition_rank,
        backward_chunk_shared_bytes(num_states, transition_rank, direct_grad_reduce),
        device_index,
        direct_grad_reduce,
        total_batches,
    };
}

ScanKernelLaunchConfig make_backward_packed_launch_config(
    const torch::Tensor& beliefs,
    int transition_rank,
    bool direct_grad_reduce) {
    const int num_states = static_cast<int>(beliefs.size(2));
    const int device_index = beliefs.get_device();
    const int total_batches = static_cast<int>(beliefs.size(0));
    return {
        dim3(1),
        dim3(num_states),
        transition_rank,
        backward_packed_chunk_shared_bytes(num_states, transition_rank, direct_grad_reduce),
        device_index,
        direct_grad_reduce,
        total_batches,
    };
}

size_t forward_sparse_shared_bytes(int num_states, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    return static_cast<size_t>((2 * num_states) + num_warps + 4) * sizeof(float);
}

size_t backward_sparse_shared_bytes(int num_states, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    return static_cast<size_t>((5 * num_states) + num_warps) * sizeof(float);
}

SparseScanKernelLaunchConfig make_forward_sparse_launch_config(
    const torch::Tensor& local_logits,
    int num_states) {
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(std::min(std::max(num_states, 1), 256))));
    const int device_index = local_logits.get_device();
    const int total_batches = static_cast<int>(local_logits.size(0));
    return {
        dim3(1),
        dim3(static_cast<unsigned int>(block_threads)),
        forward_sparse_shared_bytes(num_states, block_threads),
        device_index,
        total_batches,
    };
}

size_t forward_tiled_shared_bytes(int num_states, int tile_size, int split_size, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    const int staged_rank_chunk = std::max(1, std::min(split_size, kAsyncTileRankChunk));
    size_t bytes = static_cast<size_t>(
        (2 * num_states)
        + staged_rank_chunk
        + (static_cast<int>(kAsyncTilePipelineStages) * staged_rank_chunk * std::max(tile_size, 1))
        + (static_cast<int>(kAsyncTilePipelineStages) * staged_rank_chunk * std::max(num_states, 1))
        + num_warps
        + 4
    ) * sizeof(float);
    bytes += static_cast<size_t>(2 * kTensorCoreTile * kTensorCoreTile) * sizeof(__half);
    bytes += static_cast<size_t>(kTensorCoreTile * kTensorCoreTile) * sizeof(float);
    bytes += 31;
    return bytes;
}

size_t forward_tiled_shared_bytes_sm90(int num_states, int tile_size, int split_size, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    const int staged_rank_chunk = std::max(1, std::min(split_size, kSm90AsyncTileRankChunk));
    size_t bytes = static_cast<size_t>(
        (2 * num_states)
        + staged_rank_chunk
        + (async_tile_pipeline_stages<true>() * staged_rank_chunk * std::max(tile_size, 1))
        + (async_tile_pipeline_stages<true>() * staged_rank_chunk * std::max(num_states, 1))
        + num_warps
        + 4
    ) * sizeof(float);
    bytes += static_cast<size_t>(2 * kTensorCoreTile * kTensorCoreTile) * sizeof(__half);
    bytes += static_cast<size_t>(kTensorCoreTile * kTensorCoreTile) * sizeof(float);
    bytes += 31;
    return bytes;
}

size_t forward_tiled_packed_shared_bytes(int num_states, int tile_size, int split_size, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    const int staged_rank_chunk = std::max(1, std::min(split_size, kAsyncTileRankChunk));
    size_t bytes = static_cast<size_t>(
        (2 * num_states)
        + staged_rank_chunk
        + (async_tile_pipeline_stages<false>() * staged_rank_chunk * std::max(tile_size, 1))
        + num_warps
        + 4
    ) * sizeof(float);
    bytes += static_cast<size_t>(kAsyncTilePipelineStages) * staged_rank_chunk * std::max(tile_size, 1) * sizeof(uint8_t);
    return bytes;
}

size_t forward_tiled_packed_shared_bytes_sm90(int num_states, int tile_size, int split_size, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    const int staged_rank_chunk = std::max(1, std::min(split_size, kSm90AsyncTileRankChunk));
    size_t bytes = static_cast<size_t>(
        (2 * num_states)
        + staged_rank_chunk
        + (async_tile_pipeline_stages<true>() * staged_rank_chunk * std::max(tile_size, 1))
        + num_warps
        + 4
    ) * sizeof(float);
    bytes += static_cast<size_t>(async_tile_pipeline_stages<true>()) * staged_rank_chunk * std::max(tile_size, 1) * sizeof(uint8_t);
    return bytes;
}

size_t forward_masked_tiled_shared_bytes(int num_states, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    return static_cast<size_t>((2 * num_states) + num_warps + 4) * sizeof(float);
}

size_t backward_masked_tiled_shared_bytes(int num_states, int tile_size, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    const int rank_chunk = std::max(1, std::min(tile_size, kMaskedBackwardRankChunk));
    return static_cast<size_t>(
        (5 * num_states) + (rank_chunk * tile_size) + (rank_chunk * num_states) + num_warps + 4
    ) * sizeof(float);
}

size_t backward_tiled_shared_bytes(int num_states, int split_size, int tile_size, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    const int staged_rank_chunk = std::max(1, std::min(split_size, kAsyncTileRankChunk));
    size_t bytes = static_cast<size_t>(
        (3 * num_states)
        + staged_rank_chunk
        + tile_size
        + (static_cast<int>(kAsyncTilePipelineStages) * staged_rank_chunk * std::max(tile_size, 1))
        + (static_cast<int>(kAsyncTilePipelineStages) * staged_rank_chunk * std::max(num_states, 1))
        + num_warps
        + 4
    ) * sizeof(float);
    bytes += static_cast<size_t>(2 * kTensorCoreTile * kTensorCoreTile) * sizeof(__half);
    bytes += static_cast<size_t>(kTensorCoreTile * kTensorCoreTile) * sizeof(float);
    bytes += 31;
    return bytes;
}

size_t backward_tiled_shared_bytes_sm90(int num_states, int split_size, int tile_size, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    const int staged_rank_chunk = std::max(1, std::min(split_size, kSm90AsyncTileRankChunk));
    size_t bytes = static_cast<size_t>(
        (3 * num_states)
        + staged_rank_chunk
        + tile_size
        + (static_cast<int>(kAsyncTilePipelineStages) * staged_rank_chunk * std::max(tile_size, 1))
        + (static_cast<int>(kAsyncTilePipelineStages) * staged_rank_chunk * std::max(num_states, 1))
        + num_warps
        + 4
    ) * sizeof(float);
    bytes += static_cast<size_t>(2 * kTensorCoreTile * kTensorCoreTile) * sizeof(__half);
    bytes += static_cast<size_t>(kTensorCoreTile * kTensorCoreTile) * sizeof(float);
    bytes += 31;
    return bytes;
}

size_t backward_tiled_packed_shared_bytes(int num_states, int split_size, int tile_size, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    const int staged_rank_chunk = std::max(1, std::min(split_size, kAsyncTileRankChunk));
    size_t bytes = static_cast<size_t>(
        (3 * num_states) + staged_rank_chunk + tile_size + (staged_rank_chunk * std::max(tile_size, 1)) + num_warps + 4
    ) * sizeof(float);
    bytes += static_cast<size_t>(kAsyncTilePipelineStages) * staged_rank_chunk * std::max(tile_size, 1) * sizeof(uint8_t);
    return bytes;
}

size_t backward_tiled_packed_shared_bytes_sm90(int num_states, int split_size, int tile_size, int block_threads) {
    const int num_warps = (block_threads + kWarpSize - 1) / kWarpSize;
    const int staged_rank_chunk = std::max(1, std::min(split_size, kSm90AsyncTileRankChunk));
    size_t bytes = static_cast<size_t>(
        (3 * num_states) + staged_rank_chunk + tile_size + (staged_rank_chunk * std::max(tile_size, 1)) + num_warps + 4
    ) * sizeof(float);
    bytes += static_cast<size_t>(kAsyncTilePipelineStages) * staged_rank_chunk * std::max(tile_size, 1) * sizeof(uint8_t);
    return bytes;
}

TiledScanKernelLaunchConfig make_forward_tiled_launch_config(
    const torch::Tensor& local_logits,
    int tile_size,
    int split_size,
    bool sm90_path = false) {
    const int required_threads = std::min(std::max(std::max(tile_size, split_size), 1), 256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    const int device_index = local_logits.get_device();
    const int total_batches = static_cast<int>(local_logits.size(0));
    return {
        dim3(1),
        dim3(static_cast<unsigned int>(block_threads)),
        sm90_path
            ? forward_tiled_shared_bytes_sm90(
                static_cast<int>(local_logits.size(2)),
                tile_size,
                split_size,
                block_threads)
            : forward_tiled_shared_bytes(
                static_cast<int>(local_logits.size(2)),
                tile_size,
                split_size,
                block_threads),
        device_index,
        total_batches,
    };
}

TiledScanKernelLaunchConfig make_forward_tiled_packed_launch_config(
    const torch::Tensor& local_logits,
    int tile_size,
    int split_size,
    bool sm90_path = false) {
    const int required_threads = std::min(std::max(std::max(tile_size, split_size), 1), 256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    const int device_index = local_logits.get_device();
    const int total_batches = static_cast<int>(local_logits.size(0));
    return {
        dim3(1),
        dim3(static_cast<unsigned int>(block_threads)),
        sm90_path
            ? forward_tiled_packed_shared_bytes_sm90(
                static_cast<int>(local_logits.size(2)),
                tile_size,
                split_size,
                block_threads)
            : forward_tiled_packed_shared_bytes(
                static_cast<int>(local_logits.size(2)),
                tile_size,
                split_size,
                block_threads),
        device_index,
        total_batches,
    };
}

TiledScanKernelLaunchConfig make_forward_masked_tiled_launch_config(
    const torch::Tensor& local_logits,
    int tile_size) {
    const int required_threads = std::min(std::max(tile_size, 1), 256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    const int device_index = local_logits.get_device();
    const int total_batches = static_cast<int>(local_logits.size(0));
    return {
        dim3(1),
        dim3(static_cast<unsigned int>(block_threads)),
        forward_masked_tiled_shared_bytes(
            static_cast<int>(local_logits.size(2)),
            block_threads),
        device_index,
        total_batches,
    };
}

TiledScanKernelLaunchConfig make_backward_masked_tiled_launch_config(
    const torch::Tensor& beliefs,
    int tile_size) {
    const int required_threads = std::min(std::max(tile_size, 1), 256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    const int device_index = beliefs.get_device();
    const int total_batches = static_cast<int>(beliefs.size(0));
    return {
        dim3(1),
        dim3(static_cast<unsigned int>(block_threads)),
        backward_masked_tiled_shared_bytes(
            static_cast<int>(beliefs.size(2)),
            tile_size,
            block_threads),
        device_index,
        total_batches,
    };
}

TiledScanKernelLaunchConfig make_backward_tiled_packed_launch_config(
    const torch::Tensor& beliefs,
    int tile_size,
    int split_size,
    int worker_blocks,
    bool sm90_path = false) {
    const int required_threads = std::min(std::max(std::max(tile_size, split_size), 1), 256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    const int device_index = beliefs.get_device();
    const int total_batches = static_cast<int>(beliefs.size(0));
    (void)worker_blocks;
    return {
        dim3(1),
        dim3(static_cast<unsigned int>(block_threads)),
        sm90_path
            ? backward_tiled_packed_shared_bytes_sm90(
                static_cast<int>(beliefs.size(2)),
                split_size,
                tile_size,
                block_threads)
            : backward_tiled_packed_shared_bytes(
                static_cast<int>(beliefs.size(2)),
                split_size,
                tile_size,
                block_threads),
        device_index,
        total_batches,
    };
}

constexpr int kPagedCacheThreads = 256;
constexpr int kCombineStagingThreads = 256;

struct KernelLaunchDiagnostics {
    int64_t block_threads = 0;
    int64_t shared_bytes = 0;
    int64_t active_blocks_per_sm = 0;
    int64_t active_warps_per_sm = 0;
    int64_t max_warps_per_sm = 0;
    int64_t occupancy_pct = 0;
    int64_t registers_per_thread = 0;
    int64_t static_smem_bytes = 0;
    int64_t max_dynamic_smem_bytes = 0;
};

int tiled_block_threads(int64_t tile_size, int64_t split_size) {
    const int required_threads = std::min(
        std::max(static_cast<int>(tile_size), static_cast<int>(split_size)),
        256);
    return std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
}

int masked_tiled_block_threads(int64_t tile_size) {
    const int required_threads = std::min(static_cast<int>(std::max<int64_t>(tile_size, 1)), 256);
    return std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
}

int64_t preferred_float_load_bytes(int64_t tile_size) {
    const int64_t row_bytes = std::max<int64_t>(tile_size, 1) * static_cast<int64_t>(sizeof(float));
    if (row_bytes % 16 == 0 && tile_size >= 64) {
        return 16;
    }
    if (row_bytes % 8 == 0 && tile_size >= 32) {
        return 8;
    }
    return 4;
}

int64_t elements_per_float_load(int64_t tile_size) {
    return std::max<int64_t>(1, preferred_float_load_bytes(tile_size) / static_cast<int64_t>(sizeof(float)));
}

bool can_use_vectorized_float_io(int64_t tile_size) {
    return preferred_float_load_bytes(tile_size) >= 8;
}

bool can_use_async_memcpy(int device_index) {
    return cached_capability_major(device_index) >= 8;
}

bool can_use_half2_path(int device_index) {
    return cached_capability_major(device_index) >= 6;
}

template <typename scalar_t>
bool can_use_dense_128_rank8_pair_path(int device_index) {
    (void)device_index;
    return false;
}

template <>
bool can_use_dense_128_rank8_pair_path<c10::Half>(int device_index) {
    return can_use_half2_path(device_index);
}

template <>
bool can_use_dense_128_rank8_pair_path<c10::BFloat16>(int device_index) {
    return cached_capability_major(device_index) >= 8;
}

bool can_use_tensor_cores(int device_index) {
    const int major = cached_capability_major(device_index);
    const int minor = cached_capability_minor(device_index);
    return major > 7 || (major == 7 && minor >= 0);
}

bool can_use_wmma(int device_index) {
    return can_use_tensor_cores(device_index);
}

bool can_use_tma(int device_index) {
    return cached_capability_major(device_index) >= 9;
}

bool can_use_wgmma(int device_index) {
    return cached_capability_major(device_index) >= 9;
}

bool can_use_tiled_forward_tensor_core_math(
    int device_index,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t split_size) {
    return can_use_wmma(device_index)
        && num_states >= kTensorCoreTile
        && transition_rank >= kTensorCoreTile
        && tile_size >= kTensorCoreTile
        && split_size >= kTensorCoreTile;
}

int64_t estimate_persisting_l2_candidate_bytes_for_tiled_path(
    int64_t num_states,
    int64_t transition_rank) {
    return std::max<int64_t>(
        num_states * transition_rank * static_cast<int64_t>(sizeof(float)),
        0);
}

int64_t estimate_persisting_l2_effective_bytes(
    int device_index,
    int64_t candidate_bytes) {
    if (!supports_persisting_l2_window(device_index) || candidate_bytes <= 0) {
        return 0;
    }
    return std::min<int64_t>(
        candidate_bytes,
        cached_persisting_l2_cache_max_size(device_index));
}

int64_t tiled_backward_staging_per_worker_bytes(
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size);

int64_t estimate_tiled_forward_bytes_moved(
    int64_t total_batches,
    int64_t seq_len,
    int64_t num_states,
    int64_t transition_rank) {
    const int64_t recurrent_bytes =
        total_batches * seq_len * num_states * static_cast<int64_t>(sizeof(float)) * 4;
    const int64_t transition_bytes =
        2 * num_states * transition_rank * static_cast<int64_t>(sizeof(float));
    const int64_t state_bytes =
        total_batches * num_states * static_cast<int64_t>(sizeof(float));
    return recurrent_bytes + transition_bytes + state_bytes;
}

int64_t estimate_masked_tiled_forward_bytes_moved(
    int64_t total_batches,
    int64_t seq_len,
    int64_t num_states,
    int64_t transition_rank) {
    const int64_t recurrent_bytes =
        total_batches * seq_len * num_states * static_cast<int64_t>(sizeof(float)) * 4;
    const int64_t transition_bytes =
        2 * num_states * transition_rank * static_cast<int64_t>(sizeof(float))
        + num_states * num_states * static_cast<int64_t>(sizeof(bool));
    const int64_t state_bytes =
        total_batches * num_states * static_cast<int64_t>(sizeof(float));
    return recurrent_bytes + transition_bytes + state_bytes;
}

int64_t estimate_tiled_backward_bytes_moved(
    int64_t total_batches,
    int64_t seq_len,
    int64_t num_states,
    int64_t transition_rank) {
    const int64_t recurrent_bytes =
        total_batches * seq_len * num_states * static_cast<int64_t>(sizeof(float)) * 6;
    const int64_t transition_bytes =
        2 * num_states * transition_rank * static_cast<int64_t>(sizeof(float)) * 2;
    const int64_t staging_bytes =
        (((2 * num_states * transition_rank) + (2 * transition_rank) + num_states + 1)
            * static_cast<int64_t>(sizeof(float)));
    return recurrent_bytes + transition_bytes + staging_bytes;
}

int64_t estimate_masked_tiled_backward_bytes_moved(
    int64_t total_batches,
    int64_t seq_len,
    int64_t num_states,
    int64_t transition_rank) {
    const int64_t recurrent_bytes =
        total_batches * seq_len * num_states * static_cast<int64_t>(sizeof(float)) * 6;
    const int64_t transition_bytes =
        2 * num_states * transition_rank * static_cast<int64_t>(sizeof(float)) * 2
        + num_states * num_states * static_cast<int64_t>(sizeof(bool));
    return recurrent_bytes + transition_bytes;
}

int64_t estimate_tiled_forward_sync_points(
    int64_t seq_len,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t split_size) {
    const int64_t state_tiles = std::max<int64_t>(1, ceil_div_int64(num_states, tile_size));
    const int64_t rank_tiles = std::max<int64_t>(
        1,
        ceil_div_int64(transition_rank, std::min<int64_t>(split_size, kAsyncTileRankChunk)));
    return std::max<int64_t>(0, seq_len) * (4 + state_tiles * (5 + 2 * rank_tiles));
}

int64_t estimate_masked_tiled_forward_sync_points(
    int64_t seq_len,
    int64_t num_states,
    int64_t tile_size) {
    const int64_t state_tiles = std::max<int64_t>(1, ceil_div_int64(num_states, tile_size));
    return std::max<int64_t>(0, seq_len) * (5 + state_tiles * 6);
}

int64_t estimate_tiled_backward_sync_points(
    int64_t seq_len,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t split_size) {
    const int64_t state_tiles = std::max<int64_t>(1, ceil_div_int64(num_states, tile_size));
    const int64_t rank_tiles = std::max<int64_t>(
        1,
        ceil_div_int64(transition_rank, std::min<int64_t>(split_size, kAsyncTileRankChunk)));
    return std::max<int64_t>(0, seq_len) * (6 + state_tiles * (6 + 2 * rank_tiles));
}

int64_t estimate_masked_tiled_backward_sync_points(
    int64_t seq_len,
    int64_t num_states,
    int64_t tile_size) {
    const int64_t state_tiles = std::max<int64_t>(1, ceil_div_int64(num_states, tile_size));
    return std::max<int64_t>(0, seq_len) * (6 + state_tiles * 7);
}

template <typename KernelT>
KernelLaunchDiagnostics describe_kernel_launch(
    KernelT kernel,
    int device_index,
    int block_threads,
    size_t dynamic_smem_bytes) {
    KernelLaunchDiagnostics info;
    info.block_threads = std::max<int64_t>(block_threads, 0);
    info.shared_bytes = static_cast<int64_t>(dynamic_smem_bytes);
    info.max_dynamic_smem_bytes = static_cast<int64_t>(cached_max_optin_bytes(device_index));
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
    cudaFuncAttributes attr{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    info.registers_per_thread = static_cast<int64_t>(attr.numRegs);
    info.static_smem_bytes = static_cast<int64_t>(attr.sharedSizeBytes);
    int active_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        kernel,
        block_threads,
        dynamic_smem_bytes));
    info.active_blocks_per_sm = std::max<int64_t>(active_blocks_per_sm, 0);
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device_index));
    info.max_warps_per_sm = std::max<int64_t>(1, props.maxThreadsPerMultiProcessor / kWarpSize);
    info.active_warps_per_sm = info.active_blocks_per_sm * ((block_threads + kWarpSize - 1) / kWarpSize);
    info.occupancy_pct = (100 * info.active_warps_per_sm) / info.max_warps_per_sm;
    return info;
}

int64_t cached_total_global_mem(int device_index) {
    static std::mutex mutex;
    static std::unordered_map<int, int64_t> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(device_index);
    if (it != cache.end()) {
        return it->second;
    }
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device_index));
    const int64_t value = static_cast<int64_t>(props.totalGlobalMem);
    cache.emplace(device_index, value);
    return value;
}

int64_t current_free_global_mem() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return static_cast<int64_t>(free_bytes);
}

int64_t tiled_backward_staging_per_worker_bytes(
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size) {
    const int64_t per_worker_words =
        (2 * num_states * transition_rank)
        + (2 * transition_rank)
        + num_states
        + 1;
    return std::max<int64_t>(per_worker_words * static_cast<int64_t>(sizeof(float)), 1);
}

TiledBackwardRuntimeConfig make_tiled_backward_runtime_config(
    int device_index,
    int64_t total_batches,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size) {
    const int64_t target_worker_blocks = std::max<int64_t>(
        1,
        persistent_worker_blocks(device_index, static_cast<int>(total_batches)));
    const int64_t per_worker_bytes = tiled_backward_staging_per_worker_bytes(
        num_states,
        transition_rank,
        tile_size);
    const int64_t total_mem_bytes = std::max<int64_t>(cached_total_global_mem(device_index), 1);
    const int64_t free_mem_bytes = std::max<int64_t>(current_free_global_mem(), per_worker_bytes);
    const int64_t reserve_bytes = std::max<int64_t>(total_mem_bytes / 32, free_mem_bytes / 8);
    const int64_t usable_bytes = free_mem_bytes > reserve_bytes
        ? (free_mem_bytes - reserve_bytes)
        : std::max<int64_t>(free_mem_bytes / 2, per_worker_bytes);
    const int64_t staging_budget_bytes = std::max<int64_t>(
        per_worker_bytes,
        std::min<int64_t>(usable_bytes / 2, per_worker_bytes * target_worker_blocks));
    const int64_t budget_worker_blocks = std::max<int64_t>(1, staging_budget_bytes / per_worker_bytes);
    const int64_t staging_worker_blocks = std::max<int64_t>(
        1,
        std::min<int64_t>(target_worker_blocks, budget_worker_blocks));
    // The tiled backward stages gradients per persistent CTA, not per batch item.
    // A smaller CTA budget should not force narrower batch slices because the
    // device-side work queue can reuse the same staging slot across many batches.
    const int64_t launch_batch_size = std::max<int64_t>(1, total_batches);
    return {
        launch_batch_size,
        staging_worker_blocks,
        staging_budget_bytes,
        per_worker_bytes,
    };
}

int64_t tiled_backward_launch_worker_blocks(
    int device_index,
    int64_t launch_batch_size,
    int64_t staging_worker_blocks) {
    const int64_t launch_blocks = static_cast<int64_t>(persistent_worker_blocks(
        device_index,
        static_cast<int>(launch_batch_size)));
    return std::max<int64_t>(1, std::min<int64_t>(staging_worker_blocks, launch_blocks));
}

int64_t small_state_direct_staging_worker_blocks(
    int device_index,
    int64_t total_batches) {
    return std::max<int64_t>(
        1,
        static_cast<int64_t>(persistent_worker_blocks(
            device_index,
            static_cast<int>(total_batches))));
}

__global__ void combine_block_staging_1d_kernel(
    const float* __restrict__ staging,
    float* __restrict__ output,
    int64_t num_blocks,
    int64_t size) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    float sum = 0.0f;
    for (int64_t block = 0; block < num_blocks; ++block) {
        sum += staging[block * size + idx];
    }
    output[idx] += sum;
}

__global__ void combine_block_staging_atomic_kernel(
    const float* __restrict__ staging,
    float* __restrict__ output,
    int64_t num_blocks,
    int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t staging_total = num_blocks * total;
    if (idx >= staging_total) {
        return;
    }
    const int64_t output_idx = idx % total;
    atomicAdd(output + output_idx, staging[idx]);
}

__global__ void reduce_block_staging_pass_kernel(
    const float* __restrict__ staging,
    float* __restrict__ reduced,
    int64_t num_blocks,
    int64_t total,
    int64_t reduce_factor) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t reduced_block = static_cast<int64_t>(blockIdx.y);
    const int64_t start_block = reduced_block * reduce_factor;
    if (idx >= total || start_block >= num_blocks) {
        return;
    }
    const int64_t end_block = std::min<int64_t>(num_blocks, start_block + reduce_factor);
    float sum = 0.0f;
    for (int64_t block = start_block; block < end_block; ++block) {
        sum += staging[block * total + idx];
    }
    reduced[reduced_block * total + idx] = sum;
}

__global__ void add_reduced_staging_kernel(
    const float* __restrict__ reduced,
    float* __restrict__ output,
    int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    output[idx] += reduced[idx];
}

enum class BlockStagingReduceMode : int {
    DeterministicTree = 0,
    NondeterministicAtomic = 1,
};

BlockStagingReduceMode block_staging_reduce_mode() {
    static std::mutex mutex;
    static bool initialized = false;
    static BlockStagingReduceMode mode = BlockStagingReduceMode::DeterministicTree;
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return mode;
    }
    if (const char* env_value = std::getenv("CAUSAL_MACHINE_STAGING_REDUCTION_MODE")) {
        if (std::strcmp(env_value, "nondeterministic_atomic") == 0
            || std::strcmp(env_value, "atomic") == 0) {
            mode = BlockStagingReduceMode::NondeterministicAtomic;
        } else if (std::strcmp(env_value, "deterministic_tree") == 0
                   || std::strcmp(env_value, "tree") == 0) {
            mode = BlockStagingReduceMode::DeterministicTree;
        }
    }
    initialized = true;
    return mode;
}

void combine_block_staging_cuda_impl(
    const torch::Tensor& staging,
    const torch::Tensor& output,
    int64_t total) {
    if (total == 0) {
        return;
    }
    const int64_t num_blocks = staging.size(0);
    if (num_blocks <= 0) {
        return;
    }
    const int64_t grid = ceil_div_int64(total, static_cast<int64_t>(kCombineStagingThreads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto mode = block_staging_reduce_mode();
    if (mode == BlockStagingReduceMode::NondeterministicAtomic) {
        const int64_t atomic_grid = ceil_div_int64(
            num_blocks * total,
            static_cast<int64_t>(kCombineStagingThreads));
        combine_block_staging_atomic_kernel<<<
            static_cast<unsigned int>(atomic_grid),
            kCombineStagingThreads,
            0,
            stream>>>(
                staging.data_ptr<float>(),
                output.data_ptr<float>(),
                num_blocks,
                total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    if (num_blocks <= 8) {
        combine_block_staging_1d_kernel<<<
            static_cast<unsigned int>(grid),
            kCombineStagingThreads,
            0,
            stream>>>(
                staging.data_ptr<float>(),
                output.data_ptr<float>(),
                num_blocks,
                total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    constexpr int64_t kReduceFactor = 8;
    torch::Tensor current = staging;
    int64_t current_blocks = num_blocks;
    while (current_blocks > 1) {
        const int64_t next_blocks = ceil_div_int64(current_blocks, kReduceFactor);
        auto reduced = torch::zeros(
            {next_blocks, total},
            output.options());
        reduce_block_staging_pass_kernel<<<
            dim3(static_cast<unsigned int>(grid), static_cast<unsigned int>(next_blocks)),
            kCombineStagingThreads,
            0,
            stream>>>(
                current.data_ptr<float>(),
                reduced.data_ptr<float>(),
                current_blocks,
                total,
                kReduceFactor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        current = reduced;
        current_blocks = next_blocks;
    }
    add_reduced_staging_kernel<<<
        static_cast<unsigned int>(grid),
        kCombineStagingThreads,
        0,
        stream>>>(
            current.data_ptr<float>(),
            output.data_ptr<float>(),
            total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void combine_block_staging_2d_kernel(
    const float* __restrict__ staging,
    float* __restrict__ output,
    int64_t num_blocks,
    int64_t rows,
    int64_t cols) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = rows * cols;
    if (idx >= total) {
        return;
    }
    float sum = 0.0f;
    for (int64_t block = 0; block < num_blocks; ++block) {
        sum += staging[block * total + idx];
    }
    output[idx] += sum;
}

void combine_block_staging_1d_cuda(
    const torch::Tensor& staging,
    const torch::Tensor& output) {
    TORCH_CHECK(staging.scalar_type() == torch::kFloat32, "combine_block_staging_1d_cuda expects float staging");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "combine_block_staging_1d_cuda expects float output");
    const int64_t size = output.numel();
    combine_block_staging_cuda_impl(staging.view({staging.size(0), size}), output.view({size}), size);
}

void combine_block_staging_2d_cuda(
    const torch::Tensor& staging,
    const torch::Tensor& output) {
    TORCH_CHECK(staging.scalar_type() == torch::kFloat32, "combine_block_staging_2d_cuda expects float staging");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "combine_block_staging_2d_cuda expects float output");
    TORCH_CHECK(output.dim() == 2, "combine_block_staging_2d_cuda expects a 2D output tensor");
    const int64_t rows = output.size(0);
    const int64_t cols = output.size(1);
    const int64_t total = rows * cols;
    combine_block_staging_cuda_impl(staging.view({staging.size(0), total}), output.view({total}), total);
}

__global__ void reset_tiled_backward_state_kernel(
    float* __restrict__ grad_transition_source_probs_staging,
    int64_t grad_transition_source_probs_size,
    float* __restrict__ grad_transition_dest_probs_staging,
    int64_t grad_transition_dest_probs_size,
    float* __restrict__ grad_transition_gate_staging,
    int64_t grad_transition_gate_size,
    float* __restrict__ grad_transition_stay_staging,
    int64_t grad_transition_stay_size,
    int32_t* __restrict__ work_queue_counter) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < grad_transition_source_probs_size) {
        grad_transition_source_probs_staging[idx] = 0.0f;
    }
    if (idx < grad_transition_dest_probs_size) {
        grad_transition_dest_probs_staging[idx] = 0.0f;
    }
    if (idx < grad_transition_gate_size) {
        grad_transition_gate_staging[idx] = 0.0f;
    }
    if (idx < grad_transition_stay_size) {
        grad_transition_stay_staging[idx] = 0.0f;
    }
    if (idx == 0) {
        work_queue_counter[0] = 0;
        work_queue_counter[1] = 0;
    }
}

void reset_tiled_backward_state_cuda(
    const torch::Tensor& grad_transition_source_probs_staging,
    const torch::Tensor& grad_transition_dest_probs_staging,
    const torch::Tensor& grad_transition_gate_staging,
    const torch::Tensor& grad_transition_stay_staging,
    const torch::Tensor& work_queue_counter) {
    TORCH_CHECK(
        grad_transition_source_probs_staging.scalar_type() == torch::kFloat32,
        "reset_tiled_backward_state_cuda expects float source staging");
    TORCH_CHECK(
        grad_transition_dest_probs_staging.scalar_type() == torch::kFloat32,
        "reset_tiled_backward_state_cuda expects float dest staging");
    TORCH_CHECK(
        grad_transition_gate_staging.scalar_type() == torch::kFloat32,
        "reset_tiled_backward_state_cuda expects float gate staging");
    TORCH_CHECK(
        grad_transition_stay_staging.scalar_type() == torch::kFloat32,
        "reset_tiled_backward_state_cuda expects float stay staging");
    TORCH_CHECK(
        work_queue_counter.scalar_type() == torch::kInt32,
        "reset_tiled_backward_state_cuda expects int32 work queue counter");
    const int64_t source_size = grad_transition_source_probs_staging.numel();
    const int64_t dest_size = grad_transition_dest_probs_staging.numel();
    const int64_t gate_size = grad_transition_gate_staging.numel();
    const int64_t stay_size = grad_transition_stay_staging.numel();
    const int64_t total = std::max(
        std::max(source_size, dest_size),
        std::max(gate_size, stay_size));
    if (total == 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        C10_CUDA_CHECK(cudaMemsetAsync(work_queue_counter.data_ptr<int32_t>(), 0, 2 * sizeof(int32_t), stream));
        return;
    }
    const int64_t grid = ceil_div_int64(total, static_cast<int64_t>(kCombineStagingThreads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    reset_tiled_backward_state_kernel<<<
        static_cast<unsigned int>(grid),
        kCombineStagingThreads,
        0,
        stream>>>(
            grad_transition_source_probs_staging.data_ptr<float>(),
            source_size,
            grad_transition_dest_probs_staging.data_ptr<float>(),
            dest_size,
            grad_transition_gate_staging.data_ptr<float>(),
            gate_size,
            grad_transition_stay_staging.data_ptr<float>(),
            stay_size,
            work_queue_counter.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
__global__ void record_paged_step_tensor_kernel(
    scalar_t* __restrict__ paged_values,
    const scalar_t* __restrict__ values,
    int64_t batch_size,
    int64_t num_slots,
    int64_t max_pages,
    int64_t page_size,
    int64_t feature_dim,
    int64_t page_idx,
    int64_t page_offset) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_size * feature_dim;
    if (idx >= total) {
        return;
    }
    const int64_t b = idx / feature_dim;
    const int64_t f = idx - (b * feature_dim);
    const int64_t slot_idx = b * max_pages + page_idx;
    if (slot_idx < 0 || slot_idx >= num_slots) {
        return;
    }
    const int64_t dst_idx = ((slot_idx * page_size + page_offset) * feature_dim) + f;
    paged_values[dst_idx] = values[idx];
}

__device__ __forceinline__ int64_t resolve_paged_slot_idx(
    const int64_t* __restrict__ paged_page_table,
    int64_t batch_idx,
    int64_t logical_page_idx,
    int64_t max_pages,
    int64_t num_slots) {
    if (logical_page_idx < 0 || logical_page_idx >= max_pages) {
        return -1;
    }
    const int64_t slot_idx = paged_page_table == nullptr
        ? (batch_idx * max_pages + logical_page_idx)
        : paged_page_table[batch_idx * max_pages + logical_page_idx];
    return (slot_idx >= 0 && slot_idx < num_slots) ? slot_idx : -1;
}

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr);

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_from_float(float value);

struct FloatPair {
    float x;
    float y;
};

template <typename scalar_t>
__device__ __forceinline__ FloatPair load_pair_as_float(const scalar_t* ptr);

__device__ __forceinline__ FloatPair load_pair_as_float(const float* ptr);
__device__ __forceinline__ FloatPair load_pair_as_float(const c10::Half* ptr);
__device__ __forceinline__ FloatPair load_pair_as_float(const c10::BFloat16* ptr);

template <typename scalar_t>
__device__ __forceinline__ void store_pair_from_float(scalar_t* ptr, FloatPair value);

template <typename T>
__device__ __forceinline__ void copy_to_shared_async_or_sync(
    T* __restrict__ shared_dst,
    const T* __restrict__ global_src,
    int total_values);

__device__ __forceinline__ float fast_exp(float value);
__device__ __forceinline__ float fast_log(float value);
__device__ __forceinline__ float block_log_softmax_norm_128(float value, float* shared, float& exp_value, float& inv_sum);

__device__ __forceinline__ float block_log_softmax_norm_128_pair(
    FloatPair value,
    float* shared,
    FloatPair& exp_value,
    float& inv_sum);
__device__ __forceinline__ float apply_score_clamp(float value, float clamp_min, float clamp_max);

template <int StaticTransitionRank>
__device__ __forceinline__ void compute_latent_small_rank_128_from_register(
    float prev_prob_value,
    const float* __restrict__ source_shared,
    float* __restrict__ latent,
    float* __restrict__ partial_sums,
    int thread_idx);

template <int StaticTransitionRank>
__device__ __forceinline__ void compute_latent_small_rank_128_from_pair_register(
    FloatPair prev_prob_value,
    const float* __restrict__ source_shared,
    float* __restrict__ latent,
    float* __restrict__ partial_sums,
    int thread_idx);

template <typename scalar_t>
__global__ void record_paged_step_tensor_from_lengths_kernel(
    scalar_t* __restrict__ paged_values,
    const int64_t* __restrict__ paged_page_table,
    const int64_t* __restrict__ paged_lengths,
    const scalar_t* __restrict__ values,
    int64_t batch_size,
    int64_t num_slots,
    int64_t max_pages,
    int64_t page_size,
    int64_t feature_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_size * feature_dim;
    if (idx >= total) {
        return;
    }
    const int64_t b = idx / feature_dim;
    const int64_t f = idx - (b * feature_dim);
    const int64_t length = paged_lengths[b];
    const int64_t capacity = max_pages * page_size;
    if (length < 0 || length >= capacity || max_pages <= 0 || page_size <= 0) {
        return;
    }
    const int64_t page_idx = length / page_size;
    const int64_t page_offset = length % page_size;
    const int64_t slot_idx = resolve_paged_slot_idx(
        paged_page_table,
        b,
        page_idx,
        max_pages,
        num_slots);
    if (slot_idx < 0) {
        return;
    }
    const int64_t dst_idx = ((slot_idx * page_size + page_offset) * feature_dim) + f;
    paged_values[dst_idx] = values[idx];
}

template <typename scalar_t>
__global__ void read_paged_latest_tensor_kernel(
    const scalar_t* __restrict__ paged_values,
    const int64_t* __restrict__ paged_page_table,
    const int64_t* __restrict__ paged_lengths,
    scalar_t* __restrict__ values,
    int64_t batch_size,
    int64_t num_slots,
    int64_t max_pages,
    int64_t page_size,
    int64_t feature_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_size * feature_dim;
    if (idx >= total) {
        return;
    }
    const int64_t b = idx / feature_dim;
    const int64_t f = idx - (b * feature_dim);
    const int64_t length = paged_lengths[b];
    if (length <= 0 || max_pages <= 0 || page_size <= 0) {
        values[idx] = scalar_t(0);
        return;
    }
    const int64_t capacity = max_pages * page_size;
    const int64_t clamped_length = length < capacity ? length : capacity;
    const int64_t last_idx = clamped_length - 1;
    const int64_t page_idx = last_idx / page_size;
    const int64_t page_offset = last_idx % page_size;
    const int64_t slot_idx = resolve_paged_slot_idx(
        paged_page_table,
        b,
        page_idx,
        max_pages,
        num_slots);
    if (slot_idx < 0) {
        values[idx] = scalar_t(0);
        return;
    }
    const int64_t src_idx = ((slot_idx * page_size + page_offset) * feature_dim) + f;
    values[idx] = paged_values[src_idx];
}

template <typename scalar_t>
__global__ void record_paged_sequence_tensor_kernel(
    scalar_t* __restrict__ paged_values,
    const int64_t* __restrict__ paged_page_table,
    const int64_t* __restrict__ paged_lengths,
    const scalar_t* __restrict__ values,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_slots,
    int64_t max_pages,
    int64_t page_size,
    int64_t feature_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_size * seq_len * feature_dim;
    if (idx >= total) {
        return;
    }
    const int64_t sequence_idx = idx / feature_dim;
    const int64_t f = idx - (sequence_idx * feature_dim);
    const int64_t t = sequence_idx % seq_len;
    const int64_t b = sequence_idx / seq_len;
    const int64_t base_length = paged_lengths[b];
    const int64_t capacity = max_pages * page_size;
    if (base_length < 0 || base_length >= capacity || max_pages <= 0 || page_size <= 0) {
        return;
    }
    const int64_t write_idx = base_length + t;
    if (write_idx < 0 || write_idx >= capacity) {
        return;
    }
    const int64_t page_idx = write_idx / page_size;
    const int64_t page_offset = write_idx % page_size;
    const int64_t slot_idx = resolve_paged_slot_idx(
        paged_page_table,
        b,
        page_idx,
        max_pages,
        num_slots);
    if (slot_idx < 0) {
        return;
    }
    const int64_t dst_idx = ((slot_idx * page_size + page_offset) * feature_dim) + f;
    paged_values[dst_idx] = values[idx];
}

__global__ void increment_paged_lengths_kernel(
    int64_t* __restrict__ paged_lengths,
    int64_t batch_size,
    int64_t delta,
    int64_t capacity) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= batch_size) {
        return;
    }
    int64_t updated = paged_lengths[b] + delta;
    if (updated < 0) {
        updated = 0;
    }
    if (updated > capacity) {
        updated = capacity;
    }
    paged_lengths[b] = updated;
}

template <typename scalar_t>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void paged_step_dense_128_rank8_kernel(
    scalar_t* __restrict__ paged_log_beliefs,
    scalar_t* __restrict__ paged_latent_states,
    const int64_t* __restrict__ paged_page_table,
    int64_t* __restrict__ paged_lengths,
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const float* __restrict__ transition_stay_probs,
    float transition_gate,
    float score_clamp_min,
    float score_clamp_max,
    int64_t batch_size,
    int64_t num_slots,
    int64_t max_pages,
    int64_t page_size,
    int64_t latent_feature_dim,
    scalar_t* __restrict__ beliefs,
    float* __restrict__ final_log_belief) {
    constexpr int kNumStates = 128;
    constexpr int kTransitionRank = 8;
    constexpr float kDefaultProb = 1.0f / static_cast<float>(kNumStates);
    const int64_t b = static_cast<int64_t>(blockIdx.x);
    const int s = threadIdx.x;
    if (b >= batch_size) {
        return;
    }

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * kTransitionRank);
    float* latent = dest_shared + (kTransitionRank * kNumStates);
    float* scratch = latent + kTransitionRank;

    copy_float_matrix_pair_to_shared_async_or_sync(
        source_shared,
        transition_source_probs,
        kTransitionRank,
        kNumStates,
        kTransitionRank,
        kTransitionRank,
        dest_shared,
        transition_dest_probs,
        kNumStates,
        kTransitionRank,
        kNumStates,
        kNumStates);
    __syncthreads();

    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;
    const int64_t length = paged_lengths[b];
    const int64_t capacity = max_pages * page_size;
    float prev_log = -fast_log(static_cast<float>(kNumStates));
    float prev_prob_value = kDefaultProb;
    if (length > 0 && max_pages > 0 && page_size > 0) {
        const int64_t clamped_length = length < capacity ? length : capacity;
        const int64_t last_idx = clamped_length - 1;
        const int64_t page_idx = last_idx / page_size;
        const int64_t page_offset = last_idx % page_size;
        const int64_t slot_idx = resolve_paged_slot_idx(
            paged_page_table,
            b,
            page_idx,
            max_pages,
            num_slots);
        if (slot_idx >= 0) {
            const int64_t src_idx = ((slot_idx * page_size + page_offset) * kNumStates) + s;
            prev_log = load_as_float(paged_log_beliefs + src_idx);
            prev_prob_value = fast_exp(prev_log);
        }
    }

    compute_latent_small_rank_128_from_register<kTransitionRank>(
        prev_prob_value,
        source_shared,
        latent,
        scratch,
        s);
    float mix_prob = 0.0f;
    #pragma unroll
    for (int r = 0; r < kTransitionRank; ++r) {
        mix_prob += latent[r] * dest_shared[r * kNumStates + s];
    }
    const int64_t base = b * kNumStates;
    const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
    const float pred_log = fast_log(pred_prob);
    const float prior_value = pred_log + load_as_float(transition_context + base + s);
    const float obs = load_as_float(local_logits + base + s)
        + transition_gate * apply_score_clamp(prior_value, score_clamp_min, score_clamp_max);
    float obs_exp = 0.0f;
    float inv_obs_sum = 0.0f;
    const float log_norm = block_log_softmax_norm_128(obs, scratch, obs_exp, inv_obs_sum);
    const float q_log = obs - log_norm;
    beliefs[base + s] = store_from_float<scalar_t>(q_log);
    final_log_belief[base + s] = q_log;

    const bool can_write_step =
        length >= 0 && length < capacity && max_pages > 0 && page_size > 0;
    if (can_write_step) {
        const int64_t page_idx = length / page_size;
        const int64_t page_offset = length % page_size;
        const int64_t slot_idx = resolve_paged_slot_idx(
            paged_page_table,
            b,
            page_idx,
            max_pages,
            num_slots);
        if (slot_idx >= 0) {
            const int64_t dst_idx = ((slot_idx * page_size + page_offset) * kNumStates) + s;
            paged_log_beliefs[dst_idx] = store_from_float<scalar_t>(q_log);
            if (paged_latent_states != nullptr && s < latent_feature_dim) {
                const int64_t latent_dst_idx = ((slot_idx * page_size + page_offset) * latent_feature_dim) + s;
                paged_latent_states[latent_dst_idx] = scalar_t(0);
            }
        }
    }
    __syncthreads();
    if (s == 0) {
        int64_t updated_length = length + 1;
        if (updated_length < 0) {
            updated_length = 0;
        }
        if (updated_length > capacity) {
            updated_length = capacity;
        }
        paged_lengths[b] = updated_length;
    }
}

template <typename scalar_t>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void paged_step_dense_128_rank8_pair_kernel(
    scalar_t* __restrict__ paged_log_beliefs,
    scalar_t* __restrict__ paged_latent_states,
    const int64_t* __restrict__ paged_page_table,
    int64_t* __restrict__ paged_lengths,
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const float* __restrict__ transition_stay_probs,
    float transition_gate,
    float score_clamp_min,
    float score_clamp_max,
    int64_t batch_size,
    int64_t num_slots,
    int64_t max_pages,
    int64_t page_size,
    int64_t latent_feature_dim,
    scalar_t* __restrict__ beliefs,
    float* __restrict__ final_log_belief) {
    constexpr int kNumStates = 128;
    constexpr int kTransitionRank = 8;
    constexpr int kStatesPerThread = 2;
    constexpr float kDefaultProb = 1.0f / static_cast<float>(kNumStates);
    const int64_t b = static_cast<int64_t>(blockIdx.x);
    const int pair_idx = threadIdx.x;
    const int state0 = pair_idx * kStatesPerThread;
    const int state1 = state0 + 1;
    if (b >= batch_size) {
        return;
    }

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * kTransitionRank);
    float* latent = dest_shared + (kTransitionRank * kNumStates);
    float* scratch = latent + kTransitionRank;

    copy_float_matrix_pair_to_shared_async_or_sync(
        source_shared,
        transition_source_probs,
        kTransitionRank,
        kNumStates,
        kTransitionRank,
        kTransitionRank,
        dest_shared,
        transition_dest_probs,
        kNumStates,
        kTransitionRank,
        kNumStates,
        kNumStates);
    __syncthreads();

    const FloatPair stay_prob{
        transition_stay_probs[state0],
        transition_stay_probs[state1],
    };
    const FloatPair one_minus_stay{
        1.0f - stay_prob.x,
        1.0f - stay_prob.y,
    };
    const int64_t length = paged_lengths[b];
    const int64_t capacity = max_pages * page_size;
    FloatPair prev_log{
        -fast_log(static_cast<float>(kNumStates)),
        -fast_log(static_cast<float>(kNumStates)),
    };
    FloatPair prev_prob_value{kDefaultProb, kDefaultProb};
    if (length > 0 && max_pages > 0 && page_size > 0) {
        const int64_t clamped_length = length < capacity ? length : capacity;
        const int64_t last_idx = clamped_length - 1;
        const int64_t page_idx = last_idx / page_size;
        const int64_t page_offset = last_idx % page_size;
        const int64_t slot_idx = resolve_paged_slot_idx(
            paged_page_table,
            b,
            page_idx,
            max_pages,
            num_slots);
        if (slot_idx >= 0) {
            const int64_t src_idx = ((slot_idx * page_size + page_offset) * kNumStates) + state0;
            prev_log = load_pair_as_float(paged_log_beliefs + src_idx);
            prev_prob_value = {fast_exp(prev_log.x), fast_exp(prev_log.y)};
        }
    }

    compute_latent_small_rank_128_from_pair_register<kTransitionRank>(
        prev_prob_value,
        source_shared,
        latent,
        scratch,
        pair_idx);
    FloatPair mix_prob{0.0f, 0.0f};
    #pragma unroll
    for (int r = 0; r < kTransitionRank; ++r) {
        const int dest_row = r * kNumStates;
        mix_prob.x += latent[r] * dest_shared[dest_row + state0];
        mix_prob.y += latent[r] * dest_shared[dest_row + state1];
    }
    const int64_t base = b * kNumStates;
    const FloatPair pred_prob{
        fmaxf(stay_prob.x * prev_prob_value.x + one_minus_stay.x * mix_prob.x, 1.0e-20f),
        fmaxf(stay_prob.y * prev_prob_value.y + one_minus_stay.y * mix_prob.y, 1.0e-20f),
    };
    const FloatPair pred_log{fast_log(pred_prob.x), fast_log(pred_prob.y)};
    const FloatPair context_pair = load_pair_as_float(transition_context + base + state0);
    const FloatPair logits_pair = load_pair_as_float(local_logits + base + state0);
    const FloatPair prior_value{
        pred_log.x + context_pair.x,
        pred_log.y + context_pair.y,
    };
    const FloatPair obs{
        logits_pair.x + transition_gate * apply_score_clamp(prior_value.x, score_clamp_min, score_clamp_max),
        logits_pair.y + transition_gate * apply_score_clamp(prior_value.y, score_clamp_min, score_clamp_max),
    };
    FloatPair obs_exp{0.0f, 0.0f};
    float inv_obs_sum = 0.0f;
    const float log_norm = block_log_softmax_norm_128_pair(obs, scratch, obs_exp, inv_obs_sum);
    const FloatPair q_log{obs.x - log_norm, obs.y - log_norm};
    store_pair_from_float(beliefs + base + state0, q_log);
    final_log_belief[base + state0] = q_log.x;
    final_log_belief[base + state1] = q_log.y;

    const bool can_write_step = length >= 0 && length < capacity && max_pages > 0 && page_size > 0;
    if (can_write_step) {
        const int64_t page_idx = length / page_size;
        const int64_t page_offset = length % page_size;
        const int64_t slot_idx = resolve_paged_slot_idx(
            paged_page_table,
            b,
            page_idx,
            max_pages,
            num_slots);
        if (slot_idx >= 0) {
            const int64_t dst_idx = ((slot_idx * page_size + page_offset) * kNumStates) + state0;
            store_pair_from_float(paged_log_beliefs + dst_idx, q_log);
            if (paged_latent_states != nullptr) {
                if (state0 < latent_feature_dim) {
                    paged_latent_states[((slot_idx * page_size + page_offset) * latent_feature_dim) + state0] = scalar_t(0);
                }
                if (state1 < latent_feature_dim) {
                    paged_latent_states[((slot_idx * page_size + page_offset) * latent_feature_dim) + state1] = scalar_t(0);
                }
            }
        }
    }
    __syncthreads();
    if (pair_idx == 0) {
        int64_t updated_length = length + 1;
        if (updated_length < 0) {
            updated_length = 0;
        }
        if (updated_length > capacity) {
            updated_length = capacity;
        }
        paged_lengths[b] = updated_length;
    }
}

enum class PackedTransitionFormat : int;

template <typename scalar_t>
struct tensor_core_input_type;

template <typename scalar_t>
using tensor_core_input_type_t = typename tensor_core_input_type<scalar_t>::type;

template <typename scalar_t>
__device__ __forceinline__ bool tensor_core_math_enabled_for_scalar();

template <typename scalar_t>
__device__ __forceinline__ void wmma_replicated_row_times_matrix_16x16(
    const float* __restrict__ vector16,
    const float* __restrict__ matrix16x16,
    int matrix_row_stride,
    tensor_core_input_type_t<scalar_t>* __restrict__ lhs_half,
    tensor_core_input_type_t<scalar_t>* __restrict__ rhs_half,
    float* __restrict__ accum_tile,
    float* __restrict__ output_accum16);

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ float unpack_packed_value(packed_t value, float scale);

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ float packed_column_dot_lowp(
    const float* __restrict__ lhs,
    const packed_t* __restrict__ packed_matrix,
    const float* __restrict__ row_scales,
    int row_count,
    int row_stride,
    int column_idx);

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ void load_packed_matrix_tile_rowmajor_16x16(
    const packed_t* __restrict__ packed_matrix,
    const float* __restrict__ row_scales,
    int row_stride,
    int row_start,
    int col_start,
    int active_rows,
    int active_cols,
    float* __restrict__ matrix_tile);

template <typename scalar_t, typename packed_t, PackedTransitionFormat Format, int StaticTransitionRank = -1>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void paged_step_packed_kernel(
    scalar_t* __restrict__ paged_log_beliefs,
    scalar_t* __restrict__ paged_latent_states,
    const int64_t* __restrict__ paged_page_table,
    int64_t* __restrict__ paged_lengths,
    const scalar_t* __restrict__ local_logits,
    const packed_t* __restrict__ transition_source_packed,
    const float* __restrict__ transition_source_scales,
    const packed_t* __restrict__ transition_dest_packed,
    const float* __restrict__ transition_dest_scales,
    const scalar_t* __restrict__ transition_context,
    const float* __restrict__ transition_stay_probs,
    float transition_gate,
    int transition_rank,
    int64_t batch_size,
    int64_t num_states,
    int64_t num_slots,
    int64_t max_pages,
    int64_t page_size,
    int64_t latent_feature_dim,
    scalar_t* __restrict__ beliefs,
    float* __restrict__ final_log_belief) {
    const int64_t b = static_cast<int64_t>(blockIdx.x);
    const int s = threadIdx.x;
    if (b >= batch_size || s >= num_states) {
        return;
    }
    const int rank = StaticTransitionRank > 0 ? StaticTransitionRank : transition_rank;
    const int kNumWarps = (static_cast<int>(num_states) + kWarpSize - 1) / kWarpSize;
    constexpr float kMinProb = 1.0e-20f;

    extern __shared__ float shared_mem[];
    float* prev_prob = shared_mem;
    float* latent = prev_prob + num_states;
    float* scratch = latent + rank;
    char* tensor_core_bytes = reinterpret_cast<char*>(scratch + kNumWarps);
    auto tensor_core_addr = reinterpret_cast<std::uintptr_t>(tensor_core_bytes);
    tensor_core_addr = (tensor_core_addr + 15u) & ~static_cast<std::uintptr_t>(15u);
    using tensor_core_input_t = tensor_core_input_type_t<scalar_t>;
    tensor_core_input_t* tensor_core_lhs = reinterpret_cast<tensor_core_input_t*>(tensor_core_addr);
    tensor_core_input_t* tensor_core_rhs = tensor_core_lhs + (kTensorCoreTile * kTensorCoreTile);
    float* tensor_core_accum = reinterpret_cast<float*>(tensor_core_rhs + (kTensorCoreTile * kTensorCoreTile));
    float* tensor_core_matrix = tensor_core_accum + (kTensorCoreTile * kTensorCoreTile);
    const bool use_tensor_core_math =
        tensor_core_math_enabled_for_scalar<scalar_t>()
        && (num_states >= kTensorCoreTile)
        && (rank >= kTensorCoreTile);

    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;
    const int64_t length = paged_lengths[b];
    const int64_t capacity = max_pages * page_size;
    float prev_log = -fast_log(static_cast<float>(std::max<int64_t>(num_states, 1)));
    float prev_prob_value = 1.0f / static_cast<float>(std::max<int64_t>(num_states, 1));
    if (length > 0 && max_pages > 0 && page_size > 0) {
        const int64_t clamped_length = length < capacity ? length : capacity;
        const int64_t last_idx = clamped_length - 1;
        const int64_t page_idx = last_idx / page_size;
        const int64_t page_offset = last_idx % page_size;
        const int64_t slot_idx = resolve_paged_slot_idx(
            paged_page_table,
            b,
            page_idx,
            max_pages,
            num_slots);
        if (slot_idx >= 0) {
            const int64_t src_idx = ((slot_idx * page_size + page_offset) * num_states) + s;
            prev_log = load_as_float(paged_log_beliefs + src_idx);
            prev_prob_value = fast_exp(prev_log);
        }
    }
    prev_prob[s] = prev_prob_value;
    __syncthreads();

    if (use_tensor_core_math) {
        for (int r = s; r < rank; r += blockDim.x) {
            latent[r] = 0.0f;
        }
        __syncthreads();
        for (int rank_start = 0; rank_start < rank; rank_start += kTensorCoreTile) {
            const int active_rank = min(kTensorCoreTile, rank - rank_start);
            if (active_rank == kTensorCoreTile) {
                for (int src_start = 0; src_start < num_states; src_start += kTensorCoreTile) {
                    const int active_src = min(kTensorCoreTile, static_cast<int>(num_states) - src_start);
#if __CUDA_ARCH__ >= 700
                    if (active_src == kTensorCoreTile) {
                        load_packed_matrix_tile_rowmajor_16x16<packed_t, Format>(
                            transition_source_packed,
                            transition_source_scales,
                            rank,
                            src_start,
                            rank_start,
                            active_src,
                            active_rank,
                            tensor_core_matrix);
                        wmma_replicated_row_times_matrix_16x16<scalar_t>(
                            prev_prob + src_start,
                            tensor_core_matrix,
                            kTensorCoreTile,
                            tensor_core_lhs,
                            tensor_core_rhs,
                            tensor_core_accum,
                            latent + rank_start);
                    } else
#endif
                    {
                        for (int r = s; r < active_rank; r += blockDim.x) {
                            latent[rank_start + r] += packed_column_dot_lowp<packed_t, Format>(
                                prev_prob + src_start,
                                transition_source_packed + static_cast<int64_t>(src_start) * rank + rank_start + r,
                                transition_source_scales + src_start,
                                active_src,
                                rank,
                                0);
                        }
                        __syncthreads();
                    }
                }
            } else {
                for (int r = s; r < active_rank; r += blockDim.x) {
                    latent[rank_start + r] = packed_column_dot_lowp<packed_t, Format>(
                        prev_prob,
                        transition_source_packed + rank_start + r,
                        transition_source_scales,
                        static_cast<int>(num_states),
                        rank,
                        0);
                }
                __syncthreads();
            }
        }
        for (int dst = s; dst < num_states; dst += blockDim.x) {
            prev_prob[dst] = 0.0f;
        }
        __syncthreads();
        for (int rank_start = 0; rank_start < rank; rank_start += kTensorCoreTile) {
            const int active_rank = min(kTensorCoreTile, rank - rank_start);
            if (active_rank == kTensorCoreTile) {
                for (int dst_start = 0; dst_start < num_states; dst_start += kTensorCoreTile) {
                    const int active_dst = min(kTensorCoreTile, static_cast<int>(num_states) - dst_start);
#if __CUDA_ARCH__ >= 700
                    if (active_dst == kTensorCoreTile) {
                        load_packed_matrix_tile_rowmajor_16x16<packed_t, Format>(
                            transition_dest_packed,
                            transition_dest_scales,
                            static_cast<int>(num_states),
                            rank_start,
                            dst_start,
                            active_rank,
                            active_dst,
                            tensor_core_matrix);
                        wmma_replicated_row_times_matrix_16x16<scalar_t>(
                            latent + rank_start,
                            tensor_core_matrix,
                            kTensorCoreTile,
                            tensor_core_lhs,
                            tensor_core_rhs,
                            tensor_core_accum,
                            prev_prob + dst_start);
                    } else
#endif
                    {
                        for (int dst_local = s; dst_local < active_dst; dst_local += blockDim.x) {
                            float mix_value = prev_prob[dst_start + dst_local];
                            for (int r = 0; r < active_rank; ++r) {
                                mix_value += latent[rank_start + r] * unpack_packed_value<packed_t, Format>(
                                    transition_dest_packed[static_cast<int64_t>(rank_start + r) * num_states + dst_start + dst_local],
                                    transition_dest_scales[rank_start + r]);
                            }
                            prev_prob[dst_start + dst_local] = mix_value;
                        }
                        __syncthreads();
                    }
                }
            } else {
                for (int dst = s; dst < num_states; dst += blockDim.x) {
                    float mix_value = prev_prob[dst];
                    for (int r = 0; r < active_rank; ++r) {
                        mix_value += latent[rank_start + r] * unpack_packed_value<packed_t, Format>(
                            transition_dest_packed[static_cast<int64_t>(rank_start + r) * num_states + dst],
                            transition_dest_scales[rank_start + r]);
                    }
                    prev_prob[dst] = mix_value;
                }
                __syncthreads();
            }
        }
    } else {
        if (s < rank) {
            latent[s] = packed_column_dot_lowp<packed_t, Format>(
                prev_prob,
                transition_source_packed,
                transition_source_scales,
                static_cast<int>(num_states),
                rank,
                s);
        }
        __syncthreads();
        prev_prob[s] = packed_column_dot_lowp<packed_t, Format>(
            latent,
            transition_dest_packed,
            transition_dest_scales,
            rank,
            static_cast<int>(num_states),
            s);
    }
    __syncthreads();

    const float mix_prob = prev_prob[s];
    const int64_t base = b * num_states;
    const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, kMinProb);
    const float pred_log = fast_log(pred_prob);
    const float obs = load_as_float(local_logits + base + s)
        + transition_gate * (pred_log + load_as_float(transition_context + base + s));
    float obs_exp = 0.0f;
    float inv_obs_sum = 0.0f;
    const float log_norm = block_log_softmax_norm_128(obs, scratch, obs_exp, inv_obs_sum);
    const float q_log = obs - log_norm;
    beliefs[base + s] = store_from_float<scalar_t>(q_log);
    final_log_belief[base + s] = q_log;

    const bool can_write_step =
        length >= 0 && length < capacity && max_pages > 0 && page_size > 0;
    if (can_write_step) {
        const int64_t page_idx = length / page_size;
        const int64_t page_offset = length % page_size;
        const int64_t slot_idx = resolve_paged_slot_idx(
            paged_page_table,
            b,
            page_idx,
            max_pages,
            num_slots);
        if (slot_idx >= 0) {
            const int64_t dst_idx = ((slot_idx * page_size + page_offset) * num_states) + s;
            paged_log_beliefs[dst_idx] = store_from_float<scalar_t>(q_log);
            if (paged_latent_states != nullptr && s < latent_feature_dim) {
                const int64_t latent_dst_idx = ((slot_idx * page_size + page_offset) * latent_feature_dim) + s;
                paged_latent_states[latent_dst_idx] = scalar_t(0);
            }
        }
    }
    __syncthreads();
    if (s == 0) {
        int64_t updated_length = length + 1;
        if (updated_length < 0) {
            updated_length = 0;
        }
        if (updated_length > capacity) {
            updated_length = capacity;
        }
        paged_lengths[b] = updated_length;
    }
}

__global__ void reorder_paged_lengths_kernel(
    const int64_t* __restrict__ src_lengths,
    const int64_t* __restrict__ beam_indices,
    int64_t* __restrict__ dst_lengths,
    int64_t batch_size) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= batch_size) {
        return;
    }
    const int64_t src_b = beam_indices[b];
    dst_lengths[b] = (src_b >= 0 && src_b < batch_size) ? src_lengths[src_b] : int64_t{0};
}

__global__ void reorder_paged_page_table_kernel(
    const int64_t* __restrict__ src_page_table,
    const int64_t* __restrict__ beam_indices,
    int64_t* __restrict__ dst_page_table,
    int64_t batch_size,
    int64_t max_pages) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_size * max_pages;
    if (idx >= total) {
        return;
    }
    const int64_t b = idx / max_pages;
    const int64_t page = idx - (b * max_pages);
    const int64_t src_b = beam_indices[b];
    dst_page_table[idx] = (src_b >= 0 && src_b < batch_size)
        ? src_page_table[src_b * max_pages + page]
        : int64_t{-1};
}

TiledScanKernelLaunchConfig make_backward_tiled_launch_config(
    const torch::Tensor& beliefs,
    int tile_size,
    int split_size,
    int worker_blocks,
    bool sm90_path = false) {
    const int required_threads = std::min(std::max(std::max(tile_size, split_size), 1), 256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    const int device_index = beliefs.get_device();
    return {
        dim3(1),
        dim3(static_cast<unsigned int>(block_threads)),
        sm90_path
            ? backward_tiled_shared_bytes_sm90(
                static_cast<int>(beliefs.size(2)),
                split_size,
                tile_size,
                block_threads)
            : backward_tiled_shared_bytes(
                static_cast<int>(beliefs.size(2)),
                split_size,
                tile_size,
                block_threads),
        device_index,
        std::max(worker_blocks, 1),
    };
}

bool use_sm90_tiled_kernel_family(int device_index) {
    return cached_capability_major(device_index) >= 9;
}

SparseScanKernelLaunchConfig make_backward_sparse_launch_config(
    const torch::Tensor& beliefs,
    int num_states) {
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(std::min(std::max(num_states, 1), 256))));
    const int device_index = beliefs.get_device();
    const int total_batches = static_cast<int>(beliefs.size(0));
    return {
        dim3(1),
        dim3(static_cast<unsigned int>(block_threads)),
        backward_sparse_shared_bytes(num_states, block_threads),
        device_index,
        total_batches,
    };
}

enum class PackedTransitionFormat : int {
    Int8 = 0,
    Fp8E4M3 = 1,
    Fp8E5M2 = 2,
};

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ float unpack_packed_value(packed_t value, float scale);

template <typename KernelT>
void ensure_dynamic_smem_configured(KernelT kernel, int device_index, int shared_bytes) {
    static std::mutex mutex;
    static std::unordered_map<std::uint64_t, bool> configured;
    const auto kernel_id = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(kernel));
    const auto key = kernel_id
        ^ (static_cast<std::uint64_t>(device_index & 0xffff) << 48)
        ^ (static_cast<std::uint64_t>(shared_bytes & 0xffffffff) << 8);
    std::lock_guard<std::mutex> lock(mutex);
    if (configured.find(key) != configured.end()) {
        return;
    }
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes));
    configured.emplace(key, true);
}

template <typename LaunchConfigT, typename KernelT>
void finalize_persistent_launch_config(
    const char* op_name,
    LaunchConfigT& launch_config,
    KernelT kernel) {
    validate_dynamic_smem_bytes(op_name, launch_config.shared_bytes, launch_config.device_index);
    ensure_dynamic_smem_configured(
        kernel,
        launch_config.device_index,
        static_cast<int>(launch_config.shared_bytes));
    launch_config.grid = dim3(static_cast<unsigned int>(occupancy_persistent_worker_blocks(
        kernel,
        launch_config.device_index,
        launch_config.total_tasks,
        static_cast<int>(launch_config.block.x),
        launch_config.shared_bytes)));
}

void maybe_configure_persisting_l2_limit(int device_index) {
    if (!supports_persisting_l2_window(device_index)) {
        return;
    }
    static std::mutex mutex;
    static std::unordered_map<int, bool> configured;
    std::lock_guard<std::mutex> lock(mutex);
    if (configured[device_index]) {
        return;
    }
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_index));
    C10_CUDA_CHECK(cudaDeviceSetLimit(
        cudaLimitPersistingL2CacheSize,
        cached_persisting_l2_cache_max_size(device_index)));
    configured[device_index] = true;
}

bool try_set_persisting_l2_window(
    cudaStream_t stream,
    int device_index,
    const void* base_ptr,
    size_t num_bytes) {
    if (!supports_persisting_l2_window(device_index) || base_ptr == nullptr || num_bytes == 0) {
        return false;
    }
    maybe_configure_persisting_l2_limit(device_index);
    cudaStreamAttrValue stream_attr{};
    const size_t window_bytes = std::min<size_t>(
        num_bytes,
        static_cast<size_t>(cached_persisting_l2_cache_max_size(device_index)));
    if (window_bytes == 0) {
        return false;
    }
    stream_attr.accessPolicyWindow.base_ptr = const_cast<void*>(base_ptr);
    stream_attr.accessPolicyWindow.num_bytes = window_bytes;
    stream_attr.accessPolicyWindow.hitRatio = 1.0f;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    C10_CUDA_CHECK(cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &stream_attr));
    return true;
}

void clear_persisting_l2_window(cudaStream_t stream) {
    cudaStreamAttrValue stream_attr{};
    stream_attr.accessPolicyWindow.base_ptr = nullptr;
    stream_attr.accessPolicyWindow.num_bytes = 0;
    stream_attr.accessPolicyWindow.hitRatio = 0.0f;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    C10_CUDA_CHECK(cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &stream_attr));
}

const torch::Tensor* select_persisting_l2_candidate(std::initializer_list<const torch::Tensor*> candidates) {
    const torch::Tensor* best = nullptr;
    size_t best_bytes = 0;
    for (const torch::Tensor* candidate : candidates) {
        if (candidate == nullptr || !candidate->defined() || candidate->numel() == 0) {
            continue;
        }
        const size_t bytes = static_cast<size_t>(candidate->nbytes());
        if (bytes > best_bytes) {
            best = candidate;
            best_bytes = bytes;
        }
    }
    return best;
}

bool try_set_persisting_l2_window_for_tensors(
    cudaStream_t stream,
    int device_index,
    std::initializer_list<const torch::Tensor*> candidates) {
    const torch::Tensor* candidate = select_persisting_l2_candidate(candidates);
    if (candidate == nullptr) {
        return false;
    }
    return try_set_persisting_l2_window(
        stream,
        device_index,
        candidate->data_ptr(),
        static_cast<size_t>(candidate->nbytes()));
}

torch::Tensor make_device_work_queue_counter(const torch::Tensor& reference) {
    return torch::zeros({2}, reference.options().dtype(torch::kInt32));
}

template <typename T>
__device__ __forceinline__ void copy_to_shared_async_or_sync(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int count) {
#if __CUDA_ARCH__ >= 800
    auto pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(T)>(sizeof(T));
    const bool has_work = count > 0;
    if (has_work) {
        pipe.producer_acquire();
    }
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        cuda::memcpy_async(dst + idx, src + idx, shape, pipe);
    }
    if (has_work) {
        pipe.producer_commit();
    }
    pipe.consumer_wait();
    pipe.consumer_release();
#else
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        dst[idx] = src[idx];
    }
#endif
}

__device__ __forceinline__ void copy_float_segment_to_shared_async_or_sync(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int count) {
#if __CUDA_ARCH__ >= 800
    auto pipe = cuda::make_pipeline();
    const bool aligned4 = (
        ((reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src)) & (alignof(float4) - 1)) == 0);
    if (aligned4) {
        const int vec_count = count / 4;
        const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
        auto* dst4 = reinterpret_cast<float4*>(dst);
        auto* src4 = reinterpret_cast<const float4*>(src);
        if (vec_count > 0) {
            pipe.producer_acquire();
        }
        for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
            cuda::memcpy_async(dst4 + idx, src4 + idx, shape4, pipe);
        }
        if (vec_count > 0) {
            pipe.producer_commit();
        }
        pipe.consumer_wait();
        pipe.consumer_release();
        const int tail_start = vec_count * 4;
        for (int idx = tail_start + threadIdx.x; idx < count; idx += blockDim.x) {
            dst[idx] = src[idx];
        }
        return;
    }
    const auto shape1 = cuda::aligned_size_t<alignof(float)>(sizeof(float));
    if (count > 0) {
        pipe.producer_acquire();
    }
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        cuda::memcpy_async(dst + idx, src + idx, shape1, pipe);
    }
    if (count > 0) {
        pipe.producer_commit();
    }
    pipe.consumer_wait();
    pipe.consumer_release();
#else
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        dst[idx] = src[idx];
    }
#endif
}

__device__ __forceinline__ void copy_float_matrix_slice_to_shared_async_or_sync(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int src_stride,
    int rows,
    int cols,
    int dst_stride) {
    for (int row = 0; row < rows; ++row) {
        copy_float_segment_to_shared_async_or_sync(
            dst + static_cast<int64_t>(row) * dst_stride,
            src + static_cast<int64_t>(row) * src_stride,
            cols);
        __syncthreads();
    }
}

__device__ __forceinline__ float pairwise_lowp_product_sum(
    float lhs0,
    float rhs0,
    float lhs1,
    float rhs1) {
#if __CUDA_ARCH__ >= 800
    const __half2 lhs = __halves2half2(__float2half_rn(lhs0), __float2half_rn(lhs1));
    const __half2 rhs = __halves2half2(__float2half_rn(rhs0), __float2half_rn(rhs1));
    const float2 product = __half22float2(__hmul2(lhs, rhs));
    return product.x + product.y;
#else
    return lhs0 * rhs0 + lhs1 * rhs1;
#endif
}

__device__ __forceinline__ float lowp_dot_contiguous(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    int count) {
    float accum = 0.0f;
    int idx = 0;
#if __CUDA_ARCH__ >= 800
    for (; idx + 1 < count; idx += 2) {
        const FloatPair lhs_pair = load_pair_as_float(lhs + idx);
        const FloatPair rhs_pair = load_pair_as_float(rhs + idx);
        accum += pairwise_lowp_product_sum(
            lhs_pair.x,
            rhs_pair.x,
            lhs_pair.y,
            rhs_pair.y);
    }
#endif
    for (; idx < count; ++idx) {
        accum += lhs[idx] * rhs[idx];
    }
    return accum;
}

__device__ __forceinline__ float lowp_dot_rhs_strided(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    int rhs_stride,
    int count) {
    float accum = 0.0f;
    int idx = 0;
#if __CUDA_ARCH__ >= 800
    for (; idx + 1 < count; idx += 2) {
        const FloatPair lhs_pair = load_pair_as_float(lhs + idx);
        accum += pairwise_lowp_product_sum(
            lhs_pair.x,
            rhs[idx * rhs_stride],
            lhs_pair.y,
            rhs[(idx + 1) * rhs_stride]);
    }
#endif
    for (; idx < count; ++idx) {
        accum += lhs[idx] * rhs[idx * rhs_stride];
    }
    return accum;
}

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ float packed_row_dot_lowp(
    const float* __restrict__ lhs,
    const packed_t* __restrict__ packed_row,
    float scale,
    int count) {
    float accum = 0.0f;
    int idx = 0;
#if __CUDA_ARCH__ >= 800
    for (; idx + 1 < count; idx += 2) {
        const FloatPair lhs_pair = load_pair_as_float(lhs + idx);
        const float rhs0 = unpack_packed_value<packed_t, Format>(packed_row[idx], scale);
        const float rhs1 = unpack_packed_value<packed_t, Format>(packed_row[idx + 1], scale);
        accum += pairwise_lowp_product_sum(lhs_pair.x, rhs0, lhs_pair.y, rhs1);
    }
#endif
    for (; idx < count; ++idx) {
        accum += lhs[idx] * unpack_packed_value<packed_t, Format>(packed_row[idx], scale);
    }
    return accum;
}

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ float packed_column_dot_lowp(
    const float* __restrict__ lhs,
    const packed_t* __restrict__ packed_matrix,
    const float* __restrict__ row_scales,
    int row_count,
    int row_stride,
    int column_idx) {
    float accum = 0.0f;
    int row = 0;
#if __CUDA_ARCH__ >= 800
    for (; row + 1 < row_count; row += 2) {
        const FloatPair lhs_pair = load_pair_as_float(lhs + row);
        const float rhs0 = unpack_packed_value<packed_t, Format>(
            packed_matrix[row * row_stride + column_idx],
            row_scales[row]);
        const float rhs1 = unpack_packed_value<packed_t, Format>(
            packed_matrix[(row + 1) * row_stride + column_idx],
            row_scales[row + 1]);
        accum += pairwise_lowp_product_sum(lhs_pair.x, rhs0, lhs_pair.y, rhs1);
    }
#endif
    for (; row < row_count; ++row) {
        accum += lhs[row] * unpack_packed_value<packed_t, Format>(
            packed_matrix[row * row_stride + column_idx],
            row_scales[row]);
    }
    return accum;
}

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ void load_packed_matrix_tile_rowmajor_16x16(
    const packed_t* __restrict__ packed_matrix,
    const float* __restrict__ row_scales,
    int row_stride,
    int row_start,
    int col_start,
    int active_rows,
    int active_cols,
    float* __restrict__ matrix_tile) {
    for (int idx = threadIdx.x; idx < kTensorCoreTile * kTensorCoreTile; idx += blockDim.x) {
        const int row = idx / kTensorCoreTile;
        const int col = idx - row * kTensorCoreTile;
        float value = 0.0f;
        if (row < active_rows && col < active_cols) {
            const int matrix_row = row_start + row;
            const int matrix_col = col_start + col;
            value = unpack_packed_value<packed_t, Format>(
                packed_matrix[static_cast<int64_t>(matrix_row) * row_stride + matrix_col],
                row_scales[matrix_row]);
        }
        matrix_tile[idx] = value;
    }
    __syncthreads();
}

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ void load_packed_matrix_transposed_tile_rowmajor_16x16(
    const packed_t* __restrict__ packed_matrix,
    const float* __restrict__ row_scales,
    int row_stride,
    int original_row_start,
    int original_col_start,
    int active_original_rows,
    int active_original_cols,
    float* __restrict__ matrix_tile) {
    for (int idx = threadIdx.x; idx < kTensorCoreTile * kTensorCoreTile; idx += blockDim.x) {
        const int row = idx / kTensorCoreTile;
        const int col = idx - row * kTensorCoreTile;
        float value = 0.0f;
        if (row < active_original_cols && col < active_original_rows) {
            const int matrix_row = original_row_start + col;
            const int matrix_col = original_col_start + row;
            value = unpack_packed_value<packed_t, Format>(
                packed_matrix[static_cast<int64_t>(matrix_row) * row_stride + matrix_col],
                row_scales[matrix_row]);
        }
        matrix_tile[idx] = value;
    }
    __syncthreads();
}

template <typename Pipe>
__device__ __forceinline__ void enqueue_float_segment_async_uncommitted(
    Pipe& pipe,
    float* __restrict__ dst,
    const float* __restrict__ src,
    int count) {
#if __CUDA_ARCH__ >= 800
    const bool aligned4 = (
        ((reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src)) & (alignof(float4) - 1)) == 0);
    if (aligned4) {
        const int vec_count = count / 4;
        const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
        auto* dst4 = reinterpret_cast<float4*>(dst);
        auto* src4 = reinterpret_cast<const float4*>(src);
        for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
            cuda::memcpy_async(dst4 + idx, src4 + idx, shape4, pipe);
        }
        const int tail_start = vec_count * 4;
        for (int idx = tail_start + threadIdx.x; idx < count; idx += blockDim.x) {
            dst[idx] = src[idx];
        }
        return;
    }
    const auto shape1 = cuda::aligned_size_t<alignof(float)>(sizeof(float));
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        cuda::memcpy_async(dst + idx, src + idx, shape1, pipe);
    }
#else
    (void)pipe;
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        dst[idx] = src[idx];
    }
#endif
}

template <typename Pipe>
__device__ __forceinline__ void enqueue_float_segment_async(
    Pipe& pipe,
    float* __restrict__ dst,
    const float* __restrict__ src,
    int count) {
#if __CUDA_ARCH__ >= 800
    if (count > 0) {
        pipe.producer_acquire();
        enqueue_float_segment_async_uncommitted(pipe, dst, src, count);
        pipe.producer_commit();
    }
#else
    enqueue_float_segment_async_uncommitted(pipe, dst, src, count);
#endif
}

template <typename Pipe>
__device__ __forceinline__ void enqueue_float_matrix_slice_async(
    Pipe& pipe,
    float* __restrict__ dst,
    const float* __restrict__ src,
    int src_stride,
    int rows,
    int cols,
    int dst_stride) {
#if __CUDA_ARCH__ >= 800
    if (rows > 0 && cols > 0) {
        pipe.producer_acquire();
    }
#endif
    for (int row = 0; row < rows; ++row) {
        enqueue_float_segment_async_uncommitted(
            pipe,
            dst + static_cast<int64_t>(row) * dst_stride,
            src + static_cast<int64_t>(row) * src_stride,
            cols);
    }
#if __CUDA_ARCH__ >= 800
    if (rows > 0 && cols > 0) {
        pipe.producer_commit();
    }
#endif
}

template <typename Pipe>
__device__ __forceinline__ void enqueue_float_matrix_pair_slice_async(
    Pipe& pipe,
    float* __restrict__ dst0,
    const float* __restrict__ src0,
    int src0_stride,
    int rows0,
    int cols0,
    int dst0_stride,
    float* __restrict__ dst1,
    const float* __restrict__ src1,
    int src1_stride,
    int rows1,
    int cols1,
    int dst1_stride) {
#if __CUDA_ARCH__ >= 800
    const bool has_work = (rows0 > 0 && cols0 > 0) || (rows1 > 0 && cols1 > 0);
    if (has_work) {
        pipe.producer_acquire();
    }
#endif
    for (int row = 0; row < rows0; ++row) {
        enqueue_float_segment_async_uncommitted(
            pipe,
            dst0 + static_cast<int64_t>(row) * dst0_stride,
            src0 + static_cast<int64_t>(row) * src0_stride,
            cols0);
    }
    for (int row = 0; row < rows1; ++row) {
        enqueue_float_segment_async_uncommitted(
            pipe,
            dst1 + static_cast<int64_t>(row) * dst1_stride,
            src1 + static_cast<int64_t>(row) * src1_stride,
            cols1);
    }
#if __CUDA_ARCH__ >= 800
    if (has_work) {
        pipe.producer_commit();
    }
#endif
}

template <typename Pipe>
__device__ __forceinline__ void wait_for_async_tile(Pipe& pipe) {
#if __CUDA_ARCH__ >= 800
    pipe.consumer_wait();
    pipe.consumer_release();
#else
    (void)pipe;
#endif
}

__device__ __forceinline__ void copy_float_matrix_pair_to_shared_async_or_sync(
    float* __restrict__ dst0,
    const float* __restrict__ src0,
    int src0_stride,
    int rows0,
    int cols0,
    int dst0_stride,
    float* __restrict__ dst1,
    const float* __restrict__ src1,
    int src1_stride,
    int rows1,
    int cols1,
    int dst1_stride) {
#if __CUDA_ARCH__ >= 800
    auto pipe = cuda::make_pipeline();
    enqueue_float_matrix_pair_slice_async(
        pipe,
        dst0,
        src0,
        src0_stride,
        rows0,
        cols0,
        dst0_stride,
        dst1,
        src1,
        src1_stride,
        rows1,
        cols1,
        dst1_stride);
    wait_for_async_tile(pipe);
#else
    for (int row = 0; row < rows0; ++row) {
        float* dst_row = dst0 + static_cast<int64_t>(row) * dst0_stride;
        const float* src_row = src0 + static_cast<int64_t>(row) * src0_stride;
        for (int col = threadIdx.x; col < cols0; col += blockDim.x) {
            dst_row[col] = src_row[col];
        }
    }
    for (int row = 0; row < rows1; ++row) {
        float* dst_row = dst1 + static_cast<int64_t>(row) * dst1_stride;
        const float* src_row = src1 + static_cast<int64_t>(row) * src1_stride;
        for (int col = threadIdx.x; col < cols1; col += blockDim.x) {
            dst_row[col] = src_row[col];
        }
    }
#endif
}

template <typename Pipe, typename packed_t>
__device__ __forceinline__ void enqueue_packed_segment_async_uncommitted(
    Pipe& pipe,
    packed_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    int count) {
#if __CUDA_ARCH__ >= 800
    const auto shape1 = cuda::aligned_size_t<alignof(packed_t)>(sizeof(packed_t));
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        cuda::memcpy_async(dst + idx, src + idx, shape1, pipe);
    }
#else
    (void)pipe;
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        dst[idx] = src[idx];
    }
#endif
}

template <typename Pipe, typename packed_t>
__device__ __forceinline__ void enqueue_packed_matrix_slice_async(
    Pipe& pipe,
    packed_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    int src_stride,
    int rows,
    int cols,
    int dst_stride) {
#if __CUDA_ARCH__ >= 800
    if (rows > 0 && cols > 0) {
        pipe.producer_acquire();
    }
#endif
    for (int row = 0; row < rows; ++row) {
        enqueue_packed_segment_async_uncommitted(
            pipe,
            dst + static_cast<int64_t>(row) * dst_stride,
            src + static_cast<int64_t>(row) * src_stride,
            cols);
    }
#if __CUDA_ARCH__ >= 800
    if (rows > 0 && cols > 0) {
        pipe.producer_commit();
    }
#endif
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
__device__ __forceinline__ FloatPair load_pair_as_float(const scalar_t* ptr) {
    return {load_as_float(ptr), load_as_float(ptr + 1)};
}

__device__ __forceinline__ FloatPair load_pair_as_float(const float* ptr) {
    if ((reinterpret_cast<std::uintptr_t>(ptr) & (alignof(float2) - 1)) == 0) {
        const float2 packed = *reinterpret_cast<const float2*>(ptr);
        return {packed.x, packed.y};
    }
    return {ptr[0], ptr[1]};
}

__device__ __forceinline__ FloatPair load_pair_as_float(const c10::Half* ptr) {
    const auto packed = *reinterpret_cast<const __half2*>(ptr);
    const float2 unpacked = __half22float2(packed);
    return {unpacked.x, unpacked.y};
}

__device__ __forceinline__ FloatPair load_pair_as_float(const c10::BFloat16* ptr) {
    const auto packed = *reinterpret_cast<const __nv_bfloat162*>(ptr);
    const float2 unpacked = __bfloat1622float2(packed);
    return {unpacked.x, unpacked.y};
}

template <typename scalar_t>
__device__ __forceinline__ void store_pair_from_float(scalar_t* ptr, FloatPair value) {
    ptr[0] = store_from_float<scalar_t>(value.x);
    ptr[1] = store_from_float<scalar_t>(value.y);
}

template <>
__device__ __forceinline__ void store_pair_from_float<float>(float* ptr, FloatPair value) {
    if ((reinterpret_cast<std::uintptr_t>(ptr) & (alignof(float2) - 1)) == 0) {
        *reinterpret_cast<float2*>(ptr) = make_float2(value.x, value.y);
        return;
    }
    ptr[0] = value.x;
    ptr[1] = value.y;
}

template <>
__device__ __forceinline__ void store_pair_from_float<c10::Half>(c10::Half* ptr, FloatPair value) {
    *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(value.x, value.y);
}

template <>
__device__ __forceinline__ void store_pair_from_float<c10::BFloat16>(c10::BFloat16* ptr, FloatPair value) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = __floats2bfloat162_rn(value.x, value.y);
}

template <typename scalar_t>
__device__ __forceinline__ bool tensor_core_math_enabled_for_scalar() {
#if __CUDA_ARCH__ >= 800
    return std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>;
#elif __CUDA_ARCH__ >= 700
    return std::is_same_v<scalar_t, c10::Half>;
#else
    return false;
#endif
}

template <typename scalar_t>
struct tensor_core_input_type;

template <>
struct tensor_core_input_type<c10::Half> {
    using type = __half;
};

template <>
struct tensor_core_input_type<float> {
    using type = __half;
};

template <>
struct tensor_core_input_type<double> {
    using type = __half;
};

template <>
struct tensor_core_input_type<c10::BFloat16> {
    using type = __nv_bfloat16;
};

template <typename tc_t>
__device__ __forceinline__ tc_t tensor_core_input_from_float(float value);

template <>
__device__ __forceinline__ __half tensor_core_input_from_float<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 tensor_core_input_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

#if __CUDA_ARCH__ >= 700
template <typename scalar_t>
__device__ __forceinline__ void wmma_replicated_row_times_matrix_16x16(
    const float* __restrict__ vector16,
    const float* __restrict__ matrix16x16,
    int matrix_row_stride,
    tensor_core_input_type_t<scalar_t>* __restrict__ lhs_half,
    tensor_core_input_type_t<scalar_t>* __restrict__ rhs_half,
    float* __restrict__ accum_tile,
    float* __restrict__ output_accum16) {
    using tc_t = tensor_core_input_type_t<scalar_t>;
    if (threadIdx.x < kWarpSize) {
        for (int idx = threadIdx.x; idx < kTensorCoreTile * kTensorCoreTile; idx += kWarpSize) {
            const int row = idx / kTensorCoreTile;
            const int col = idx - row * kTensorCoreTile;
            lhs_half[idx] = tensor_core_input_from_float<tc_t>(vector16[col]);
            rhs_half[col * kTensorCoreTile + row] = tensor_core_input_from_float<tc_t>(matrix16x16[row * matrix_row_stride + col]);
        }
    }
    __syncthreads();
    if (threadIdx.x < kWarpSize) {
        using namespace nvcuda;
        wmma::fragment<wmma::matrix_a, kTensorCoreTile, kTensorCoreTile, kTensorCoreTile, tc_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kTensorCoreTile, kTensorCoreTile, kTensorCoreTile, tc_t, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, kTensorCoreTile, kTensorCoreTile, kTensorCoreTile, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, lhs_half, kTensorCoreTile);
        wmma::load_matrix_sync(b_frag, rhs_half, kTensorCoreTile);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::store_matrix_sync(accum_tile, c_frag, kTensorCoreTile, wmma::mem_row_major);
        for (int col = threadIdx.x; col < kTensorCoreTile; col += kWarpSize) {
            output_accum16[col] += accum_tile[col];
        }
    }
    __syncthreads();
}

template <typename scalar_t>
__device__ __forceinline__ void wmma_matrix_times_replicated_col_vector_16x16(
    const float* __restrict__ matrix16x16,
    int matrix_row_stride,
    const float* __restrict__ vector16,
    tensor_core_input_type_t<scalar_t>* __restrict__ lhs_half,
    tensor_core_input_type_t<scalar_t>* __restrict__ rhs_half,
    float* __restrict__ accum_tile,
    float* __restrict__ output_accum16) {
    using tc_t = tensor_core_input_type_t<scalar_t>;
    if (threadIdx.x < kWarpSize) {
        for (int idx = threadIdx.x; idx < kTensorCoreTile * kTensorCoreTile; idx += kWarpSize) {
            const int row = idx / kTensorCoreTile;
            const int col = idx - row * kTensorCoreTile;
            lhs_half[idx] = tensor_core_input_from_float<tc_t>(matrix16x16[row * matrix_row_stride + col]);
            rhs_half[col * kTensorCoreTile + row] = tensor_core_input_from_float<tc_t>(vector16[row]);
        }
    }
    __syncthreads();
    if (threadIdx.x < kWarpSize) {
        using namespace nvcuda;
        wmma::fragment<wmma::matrix_a, kTensorCoreTile, kTensorCoreTile, kTensorCoreTile, tc_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kTensorCoreTile, kTensorCoreTile, kTensorCoreTile, tc_t, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, kTensorCoreTile, kTensorCoreTile, kTensorCoreTile, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, lhs_half, kTensorCoreTile);
        wmma::load_matrix_sync(b_frag, rhs_half, kTensorCoreTile);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::store_matrix_sync(accum_tile, c_frag, kTensorCoreTile, wmma::mem_row_major);
        for (int row = threadIdx.x; row < kTensorCoreTile; row += kWarpSize) {
            output_accum16[row] += accum_tile[row * kTensorCoreTile];
        }
    }
    __syncthreads();
}
#endif

template <PackedTransitionFormat Format>
__host__ __device__ __forceinline__ float packed_quant_max() {
    if constexpr (Format == PackedTransitionFormat::Int8) {
        return 127.0f;
    } else if constexpr (Format == PackedTransitionFormat::Fp8E4M3) {
        return 448.0f;
    } else {
        return 57344.0f;
    }
}

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ float unpack_packed_value(packed_t value, float scale);

template <>
__device__ __forceinline__ float unpack_packed_value<int8_t, PackedTransitionFormat::Int8>(
    int8_t value,
    float scale) {
    return static_cast<float>(value) * scale;
}

template <>
__device__ __forceinline__ float unpack_packed_value<uint8_t, PackedTransitionFormat::Fp8E4M3>(
    uint8_t value,
    float scale) {
    __nv_fp8_e4m3 packed;
    packed.__x = value;
    return static_cast<float>(packed) * scale;
}

template <>
__device__ __forceinline__ float unpack_packed_value<uint8_t, PackedTransitionFormat::Fp8E5M2>(
    uint8_t value,
    float scale) {
    __nv_fp8_e5m2 packed;
    packed.__x = value;
    return static_cast<float>(packed) * scale;
}

template <typename packed_t, PackedTransitionFormat Format>
__global__ void unpack_transition_table_per_row_kernel(
    const packed_t* __restrict__ packed,
    const float* __restrict__ scales,
    int cols,
    float* __restrict__ output) {
    const int row = blockIdx.x;
    const float scale = scales[row];
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        output[row * cols + col] = unpack_packed_value<packed_t, Format>(
            packed[row * cols + col],
            scale);
    }
}

template <typename packed_t, PackedTransitionFormat Format>
__device__ __forceinline__ packed_t pack_packed_value(float value, float inv_scale);

template <>
__device__ __forceinline__ int8_t pack_packed_value<int8_t, PackedTransitionFormat::Int8>(
    float value,
    float inv_scale) {
    const float scaled = value * inv_scale;
    return static_cast<int8_t>(lrintf(fminf(fmaxf(scaled, -127.0f), 127.0f)));
}

template <>
__device__ __forceinline__ uint8_t pack_packed_value<uint8_t, PackedTransitionFormat::Fp8E4M3>(
    float value,
    float inv_scale) {
    const __nv_fp8_e4m3 packed(value * inv_scale);
    return packed.__x;
}

template <>
__device__ __forceinline__ uint8_t pack_packed_value<uint8_t, PackedTransitionFormat::Fp8E5M2>(
    float value,
    float inv_scale) {
    const __nv_fp8_e5m2 packed(value * inv_scale);
    return packed.__x;
}

__device__ __forceinline__ float fast_exp(float value) {
    return __expf(value);
}

__device__ __forceinline__ float fast_log(float value) {
    return __logf(value);
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
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_max(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = (threadIdx.x < num_warps) ? shared[lane] : -INFINITY;
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
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_log_softmax_norm_128(float value, float* shared, float& exp_value, float& inv_sum) {
    const float value_max = block_reduce_max_128(value, shared);
    exp_value = fast_exp(value - value_max);
    const float value_sum = block_reduce_sum_128(exp_value, shared);
    inv_sum = 1.0f / fmaxf(value_sum, 1.0e-20f);
    return value_max + fast_log(fmaxf(value_sum, 1.0e-20f));
}

__device__ __forceinline__ float block_log_softmax_norm_128_pair(
    FloatPair value,
    float* shared,
    FloatPair& exp_value,
    float& inv_sum) {
    const float value_max = block_reduce_max_128(fmaxf(value.x, value.y), shared);
    exp_value.x = fast_exp(value.x - value_max);
    exp_value.y = fast_exp(value.y - value_max);
    const float value_sum = block_reduce_sum_128(exp_value.x + exp_value.y, shared);
    inv_sum = 1.0f / fmaxf(value_sum, 1.0e-20f);
    return value_max + fast_log(fmaxf(value_sum, 1.0e-20f));
}

__device__ __forceinline__ float apply_score_clamp(
    float value,
    float clamp_min,
    float clamp_max) {
    return fminf(fmaxf(value, clamp_min), clamp_max);
}

__device__ __forceinline__ float score_clamp_grad(
    float value,
    float clamp_min,
    float clamp_max) {
    return (value >= clamp_min && value <= clamp_max) ? 1.0f : 0.0f;
}

constexpr int kMaxNativeScoreTopK = 32;

__device__ __forceinline__ bool native_score_filtering_enabled(
    float score_threshold,
    int score_topk,
    int num_states) {
    return isfinite(score_threshold) || (score_topk > 0 && score_topk < num_states);
}

__device__ __forceinline__ void apply_native_score_filtering_serial(
    float* __restrict__ filtered_cache,
    int num_states,
    float score_threshold,
    int score_topk) {
    const bool use_threshold = isfinite(score_threshold);
    const bool use_topk = score_topk > 0 && score_topk < num_states;
    if (!use_threshold && !use_topk) {
        return;
    }

    float argmax_value = -INFINITY;
    int argmax_index = 0;
    for (int s = 0; s < num_states; ++s) {
        const float value = filtered_cache[s];
        if (value > argmax_value) {
            argmax_value = value;
            argmax_index = s;
        }
        if (use_threshold && value < score_threshold) {
            filtered_cache[s] = -INFINITY;
        }
    }

    bool any_selected = false;
    for (int s = 0; s < num_states; ++s) {
        if (isfinite(filtered_cache[s])) {
            any_selected = true;
            break;
        }
    }
    if (!any_selected) {
        filtered_cache[argmax_index] = argmax_value;
        return;
    }

    if (!use_topk) {
        return;
    }

    float topk_values[kMaxNativeScoreTopK];
    int topk_indices[kMaxNativeScoreTopK];
    int selected = 0;
    for (int i = 0; i < kMaxNativeScoreTopK; ++i) {
        topk_values[i] = -INFINITY;
        topk_indices[i] = -1;
    }

    for (int s = 0; s < num_states; ++s) {
        const float value = filtered_cache[s];
        if (!isfinite(value)) {
            continue;
        }
        int insert_pos = selected;
        const int max_slots = min(score_topk, kMaxNativeScoreTopK);
        if (selected < max_slots) {
            ++selected;
        } else if (value > topk_values[max_slots - 1]) {
            insert_pos = max_slots - 1;
        } else {
            continue;
        }
        while (insert_pos > 0 && value > topk_values[insert_pos - 1]) {
            if (insert_pos < max_slots) {
                topk_values[insert_pos] = topk_values[insert_pos - 1];
                topk_indices[insert_pos] = topk_indices[insert_pos - 1];
            }
            --insert_pos;
        }
        topk_values[insert_pos] = value;
        topk_indices[insert_pos] = s;
    }

    for (int s = 0; s < num_states; ++s) {
        filtered_cache[s] = -INFINITY;
    }
    for (int i = 0; i < min(score_topk, kMaxNativeScoreTopK); ++i) {
        if (topk_indices[i] >= 0) {
            filtered_cache[topk_indices[i]] = topk_values[i];
        }
    }
}

struct BlockArgMax {
    float value;
    int index;
};

__device__ __forceinline__ BlockArgMax select_argmax_pair(BlockArgMax lhs, BlockArgMax rhs) {
    if (rhs.index < 0) {
        return lhs;
    }
    if (lhs.index < 0 || rhs.value > lhs.value || (rhs.value == lhs.value && rhs.index < lhs.index)) {
        return rhs;
    }
    return lhs;
}

__device__ __forceinline__ BlockArgMax warp_reduce_argmax(BlockArgMax value) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        BlockArgMax other{
            __shfl_down_sync(mask, value.value, offset),
            __shfl_down_sync(mask, value.index, offset),
        };
        value = select_argmax_pair(value, other);
    }
    return value;
}

__device__ __forceinline__ BlockArgMax block_reduce_argmax_128(
    BlockArgMax value,
    float* shared_values,
    int* shared_indices) {
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_argmax(value);
    if (lane == 0) {
        shared_values[warp] = value.value;
        shared_indices[warp] = value.index;
    }
    __syncthreads();
    value = (threadIdx.x < num_warps)
        ? BlockArgMax{shared_values[lane], shared_indices[lane]}
        : BlockArgMax{-INFINITY, -1};
    if (warp == 0) {
        value = warp_reduce_argmax(value);
        if (lane == 0) {
            shared_values[0] = value.value;
            shared_indices[0] = value.index;
        }
    }
    __syncthreads();
    return {shared_values[0], shared_indices[0]};
}

__device__ __forceinline__ void apply_native_score_filtering_block(
    float* __restrict__ filtered_cache,
    int num_states,
    float score_threshold,
    int score_topk,
    float* __restrict__ scratch,
    float* __restrict__ topk_values_shared,
    int* __restrict__ topk_indices_shared) {
    const bool use_threshold = isfinite(score_threshold);
    const bool use_topk = score_topk > 0 && score_topk < num_states;
    if (!use_threshold && !use_topk) {
        return;
    }

    BlockArgMax local_argmax{-INFINITY, -1};
    float local_selected = 0.0f;
    for (int s = threadIdx.x; s < num_states; s += blockDim.x) {
        float value = filtered_cache[s];
        local_argmax = select_argmax_pair(local_argmax, {value, s});
        if (use_threshold && value < score_threshold) {
            value = -INFINITY;
            filtered_cache[s] = value;
        }
        if (isfinite(value)) {
            local_selected += 1.0f;
        }
    }
    const BlockArgMax argmax = block_reduce_argmax_128(local_argmax, scratch, topk_indices_shared);
    const float selected = block_reduce_sum_128(local_selected, scratch);
    if (selected == 0.0f) {
        if (threadIdx.x == 0 && argmax.index >= 0) {
            filtered_cache[argmax.index] = argmax.value;
        }
        __syncthreads();
        return;
    }
    if (!use_topk) {
        return;
    }

    const int max_slots = min(score_topk, kMaxNativeScoreTopK);
    for (int slot = threadIdx.x; slot < max_slots; slot += blockDim.x) {
        topk_values_shared[slot] = -INFINITY;
        topk_indices_shared[slot] = -1;
    }
    __syncthreads();

    int selected_slots = 0;
    for (; selected_slots < max_slots; ++selected_slots) {
        BlockArgMax slot_local{-INFINITY, -1};
        for (int s = threadIdx.x; s < num_states; s += blockDim.x) {
            const float value = filtered_cache[s];
            if (isfinite(value)) {
                slot_local = select_argmax_pair(slot_local, {value, s});
            }
        }
        const BlockArgMax slot_best = block_reduce_argmax_128(slot_local, scratch, topk_indices_shared + max_slots);
        if (slot_best.index < 0 || !isfinite(slot_best.value)) {
            break;
        }
        if (threadIdx.x == 0) {
            topk_values_shared[selected_slots] = slot_best.value;
            topk_indices_shared[selected_slots] = slot_best.index;
        }
        __syncthreads();
        for (int s = threadIdx.x; s < num_states; s += blockDim.x) {
            if (s == slot_best.index) {
                filtered_cache[s] = -INFINITY;
            }
        }
        __syncthreads();
    }

    for (int s = threadIdx.x; s < num_states; s += blockDim.x) {
        filtered_cache[s] = -INFINITY;
    }
    __syncthreads();
    for (int slot = threadIdx.x; slot < selected_slots; slot += blockDim.x) {
        const int index = topk_indices_shared[slot];
        if (index >= 0) {
            filtered_cache[index] = topk_values_shared[slot];
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void update_online_logsumexp_stats(
    float tile_max,
    float tile_sum,
    float* __restrict__ tile_stats) {
    const float running_max = tile_stats[0];
    const float running_sum = tile_stats[1];
    if (!isfinite(running_max)) {
        tile_stats[0] = tile_max;
        tile_stats[1] = tile_sum;
        return;
    }
    const float new_max = fmaxf(running_max, tile_max);
    tile_stats[1] = running_sum * fast_exp(running_max - new_max)
        + tile_sum * fast_exp(tile_max - new_max);
    tile_stats[0] = new_max;
}

__device__ __forceinline__ float masked_transition_raw_value(
    const float* __restrict__ transition_source_logits,
    const float* __restrict__ transition_dest_logits,
    const bool* __restrict__ transition_mask,
    int num_states,
    int transition_rank,
    int src,
    int dst) {
    if (!transition_mask[src * num_states + dst]) {
        return 0.0f;
    }
    float raw_value = 0.0f;
    const int64_t source_base = static_cast<int64_t>(src) * transition_rank;
    for (int r = 0; r < transition_rank; ++r) {
        raw_value += transition_source_logits[source_base + r]
            * transition_dest_logits[static_cast<int64_t>(r) * num_states + dst];
    }
    return raw_value;
}

template <int StaticTransitionRank>
__device__ __forceinline__ void compute_latent_small_rank_128(
    const float* __restrict__ prev_prob,
    const float* __restrict__ source_shared,
    float* __restrict__ latent,
    float* __restrict__ partial_sums,
    int thread_idx) {
    static_assert(StaticTransitionRank > 0 && StaticTransitionRank <= kWarpSize);
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    const int lane = thread_idx & (kWarpSize - 1);
    const int warp_id = thread_idx / kWarpSize;
    const float prev = prev_prob[thread_idx];
    float contrib[StaticTransitionRank];
    #pragma unroll
    for (int r = 0; r < StaticTransitionRank; ++r) {
        contrib[r] = prev * source_shared[thread_idx * StaticTransitionRank + r];
    }
    #pragma unroll
    for (int r = 0; r < StaticTransitionRank; ++r) {
        contrib[r] = warp_reduce_sum(contrib[r]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < StaticTransitionRank; ++r) {
            partial_sums[warp_id * StaticTransitionRank + r] = contrib[r];
        }
    }
    __syncthreads();
    if (thread_idx < StaticTransitionRank) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < num_warps; ++w) {
            total += partial_sums[w * StaticTransitionRank + thread_idx];
        }
        latent[thread_idx] = total;
    }
    __syncthreads();
}

template <int StaticTransitionRank>
__device__ __forceinline__ void compute_latent_small_rank_128_from_register(
    float prev_prob_value,
    const float* __restrict__ source_shared,
    float* __restrict__ latent,
    float* __restrict__ partial_sums,
    int thread_idx) {
    static_assert(StaticTransitionRank > 0 && StaticTransitionRank <= kWarpSize);
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    const int lane = thread_idx & (kWarpSize - 1);
    const int warp_id = thread_idx / kWarpSize;
    float contrib[StaticTransitionRank];
    #pragma unroll
    for (int r = 0; r < StaticTransitionRank; ++r) {
        contrib[r] = prev_prob_value * source_shared[thread_idx * StaticTransitionRank + r];
    }
    #pragma unroll
    for (int r = 0; r < StaticTransitionRank; ++r) {
        contrib[r] = warp_reduce_sum(contrib[r]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < StaticTransitionRank; ++r) {
            partial_sums[warp_id * StaticTransitionRank + r] = contrib[r];
        }
    }
    __syncthreads();
    if (thread_idx < StaticTransitionRank) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < num_warps; ++w) {
            total += partial_sums[w * StaticTransitionRank + thread_idx];
        }
        latent[thread_idx] = total;
    }
    __syncthreads();
}

template <int StaticTransitionRank>
__device__ __forceinline__ void compute_latent_small_rank_128_from_pair_register(
    FloatPair prev_prob_value,
    const float* __restrict__ source_shared,
    float* __restrict__ latent,
    float* __restrict__ partial_sums,
    int thread_idx) {
    static_assert(StaticTransitionRank > 0 && StaticTransitionRank <= kWarpSize);
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    const int lane = thread_idx & (kWarpSize - 1);
    const int warp_id = thread_idx / kWarpSize;
    const int state0 = thread_idx * 2;
    const int state1 = state0 + 1;
    float contrib[StaticTransitionRank];
    #pragma unroll
    for (int r = 0; r < StaticTransitionRank; ++r) {
        contrib[r] = prev_prob_value.x * source_shared[state0 * StaticTransitionRank + r]
            + prev_prob_value.y * source_shared[state1 * StaticTransitionRank + r];
    }
    #pragma unroll
    for (int r = 0; r < StaticTransitionRank; ++r) {
        contrib[r] = warp_reduce_sum(contrib[r]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < StaticTransitionRank; ++r) {
            partial_sums[warp_id * StaticTransitionRank + r] = contrib[r];
        }
    }
    __syncthreads();
    if (thread_idx < StaticTransitionRank) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < num_warps; ++w) {
            total += partial_sums[w * StaticTransitionRank + thread_idx];
        }
        latent[thread_idx] = total;
    }
    __syncthreads();
}

__global__ void row_softmax_forward_kernel(
    const float* __restrict__ logits,
    int cols,
    float* __restrict__ probs) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    __shared__ float scratch[kMaxNumWarps];
    const float logit = col < cols ? logits[row * cols + col] : -INFINITY;
    const float row_max = block_reduce_max_128(logit, scratch);
    const float shifted = col < cols ? fast_exp(logit - row_max) : 0.0f;
    const float row_sum = block_reduce_sum_128(shifted, scratch);
    if (col < cols) {
        probs[row * cols + col] = shifted / fmaxf(row_sum, 1.0e-20f);
    }
}

__global__ void row_softmax_backward_kernel(
    const float* __restrict__ grad_probs,
    const float* __restrict__ probs,
    int cols,
    float* __restrict__ grad_logits) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    __shared__ float scratch[kMaxNumWarps];
    const float prob = col < cols ? probs[row * cols + col] : 0.0f;
    const float grad_prob = col < cols ? grad_probs[row * cols + col] : 0.0f;
    const float dot = block_reduce_sum_128(grad_prob * prob, scratch);
    if (col < cols) {
        grad_logits[row * cols + col] = (grad_prob - dot) * prob;
    }
}

__global__ void row_softmax_forward_strided_128_kernel(
    const float* __restrict__ logits,
    int cols,
    float* __restrict__ probs) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float scratch[kMaxNumWarps];
    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, logits[row * cols + col]);
    }
    const float row_max = block_reduce_max_128(local_max, scratch);
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += fast_exp(logits[row * cols + col] - row_max);
    }
    const float row_sum = block_reduce_sum_128(local_sum, scratch);
    const float inv_row_sum = 1.0f / fmaxf(row_sum, 1.0e-20f);
    for (int col = tid; col < cols; col += blockDim.x) {
        probs[row * cols + col] = fast_exp(logits[row * cols + col] - row_max) * inv_row_sum;
    }
}

__global__ void row_softmax_backward_strided_128_kernel(
    const float* __restrict__ grad_probs,
    const float* __restrict__ probs,
    int cols,
    float* __restrict__ grad_logits) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float scratch[kMaxNumWarps];
    float local_dot = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        local_dot += grad_probs[idx] * probs[idx];
    }
    const float dot = block_reduce_sum_128(local_dot, scratch);
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        const float prob = probs[idx];
        grad_logits[idx] = (grad_probs[idx] - dot) * prob;
    }
}

__global__ void row_softmax_stats_strided_128_kernel(
    const float* __restrict__ logits,
    int cols,
    float* __restrict__ row_max,
    float* __restrict__ row_inv_sum) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float scratch[kMaxNumWarps];
    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, logits[row * cols + col]);
    }
    const float max_value = block_reduce_max_128(local_max, scratch);
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += fast_exp(logits[row * cols + col] - max_value);
    }
    const float sum_value = block_reduce_sum_128(local_sum, scratch);
    if (tid == 0) {
        row_max[row] = max_value;
        row_inv_sum[row] = 1.0f / fmaxf(sum_value, 1.0e-20f);
    }
}

__global__ void row_softmax_backward_from_stats_strided_128_kernel(
    const float* __restrict__ grad_probs,
    const float* __restrict__ logits,
    const float* __restrict__ row_max,
    const float* __restrict__ row_inv_sum,
    int cols,
    float* __restrict__ grad_logits) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float scratch[kMaxNumWarps];
    const float max_value = row_max[row];
    const float inv_sum = row_inv_sum[row];
    float local_dot = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        const float prob = fast_exp(logits[idx] - max_value) * inv_sum;
        local_dot += grad_probs[idx] * prob;
    }
    const float dot = block_reduce_sum_128(local_dot, scratch);
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        const float prob = fast_exp(logits[idx] - max_value) * inv_sum;
        grad_logits[idx] = (grad_probs[idx] - dot) * prob;
    }
}

__device__ __forceinline__ float softmax_prob_from_stats_value(
    const float* __restrict__ logits,
    int row_width,
    int row,
    int col,
    const float* __restrict__ row_max,
    const float* __restrict__ row_inv_sum) {
    return fast_exp(logits[static_cast<int64_t>(row) * row_width + col] - row_max[row]) * row_inv_sum[row];
}

__device__ __forceinline__ void compute_row_softmax_stats_from_logits(
    const float* __restrict__ logits,
    int row_width,
    int row,
    float& row_max,
    float& row_inv_sum) {
    row_max = -INFINITY;
    const int64_t row_base = static_cast<int64_t>(row) * row_width;
    for (int col = 0; col < row_width; ++col) {
        row_max = fmaxf(row_max, logits[row_base + col]);
    }
    float row_sum = 0.0f;
    for (int col = 0; col < row_width; ++col) {
        row_sum += fast_exp(logits[row_base + col] - row_max);
    }
    row_inv_sum = 1.0f / fmaxf(row_sum, 1.0e-20f);
}

__device__ __forceinline__ float softmax_prob_from_logits_or_stats_value(
    const float* __restrict__ logits,
    int row_width,
    int row,
    int col,
    const float* __restrict__ row_max,
    const float* __restrict__ row_inv_sum) {
    if (row_max != nullptr && row_inv_sum != nullptr) {
        return softmax_prob_from_stats_value(
            logits,
            row_width,
            row,
            col,
            row_max,
            row_inv_sum);
    }
    float computed_row_max;
    float computed_row_inv_sum;
    compute_row_softmax_stats_from_logits(
        logits,
        row_width,
        row,
        computed_row_max,
        computed_row_inv_sum);
    return fast_exp(logits[static_cast<int64_t>(row) * row_width + col] - computed_row_max) * computed_row_inv_sum;
}

template <bool InputsAreLogits>
__device__ __forceinline__ float sparse_factor_prob_value(
    const float* __restrict__ transition_source_values,
    const float* __restrict__ transition_dest_values,
    const float* __restrict__ source_row_max,
    const float* __restrict__ source_row_inv_sum,
    const float* __restrict__ dest_row_max,
    const float* __restrict__ dest_row_inv_sum,
    int src_state,
    int dst_state,
    int transition_rank,
    int padded_states,
    int num_states) {
    if constexpr (InputsAreLogits) {
        if (src_state >= num_states || dst_state >= num_states) {
            return 0.0f;
        }
    }
    float raw_value = 0.0f;
    #pragma unroll 4
    for (int r = 0; r < transition_rank; ++r) {
        float source_value;
        float dest_value;
        if constexpr (InputsAreLogits) {
            source_value = softmax_prob_from_logits_or_stats_value(
                transition_source_values,
                transition_rank,
                src_state,
                r,
                source_row_max,
                source_row_inv_sum);
            dest_value = softmax_prob_from_logits_or_stats_value(
                transition_dest_values,
                num_states,
                r,
                dst_state,
                dest_row_max,
                dest_row_inv_sum);
        } else {
            source_value = transition_source_values[static_cast<int64_t>(src_state) * transition_rank + r];
            dest_value = transition_dest_values[static_cast<int64_t>(r) * padded_states + dst_state];
        }
        raw_value += source_value * dest_value;
    }
    return raw_value;
}

template <bool InputsAreLogits>
__device__ __forceinline__ float sparse_transition_row_sum_value(
    const float* __restrict__ transition_source_values,
    const float* __restrict__ transition_dest_values,
    const float* __restrict__ source_row_max,
    const float* __restrict__ source_row_inv_sum,
    const float* __restrict__ dest_row_max,
    const float* __restrict__ dest_row_inv_sum,
    const float* __restrict__ row_sums,
    const int32_t* __restrict__ src_row_ptr,
    const int32_t* __restrict__ src_nz_idx,
    const int32_t* __restrict__ block_dst_idx,
    const float* __restrict__ block_mask,
    int src_state,
    int transition_rank,
    int padded_states,
    int num_states,
    int block_size) {
    if constexpr (!InputsAreLogits) {
        return fmaxf(row_sums[src_state], 1.0e-20f);
    } else {
        const int src_block = src_state / block_size;
        const int src_offset = src_state - (src_block * block_size);
        float row_sum = 0.0f;
        for (int entry = src_row_ptr[src_block]; entry < src_row_ptr[src_block + 1]; ++entry) {
            const int nz = src_nz_idx[entry];
            const int dst_block = block_dst_idx[nz];
            const int dst_base = dst_block * block_size;
            const int active_dst = min(block_size, num_states - dst_base);
            const float* mask_row =
                block_mask + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
            for (int dst_offset = 0; dst_offset < active_dst; ++dst_offset) {
                const int dst_state = dst_base + dst_offset;
                const float mask_value = mask_row[dst_offset];
                if (mask_value == 0.0f || dst_state >= padded_states || dst_state >= num_states) {
                    continue;
                }
                row_sum += sparse_factor_prob_value<InputsAreLogits>(
                    transition_source_values,
                    transition_dest_values,
                    source_row_max,
                    source_row_inv_sum,
                    dest_row_max,
                    dest_row_inv_sum,
                    src_state,
                    dst_state,
                    transition_rank,
                    padded_states,
                    num_states) * mask_value;
            }
        }
        return fmaxf(row_sum, 1.0e-20f);
    }
}

__global__ void masked_dense_transition_prepare_kernel(
    const float* __restrict__ transition_source_logits,
    const float* __restrict__ transition_dest_logits,
    const bool* __restrict__ transition_mask,
    int num_states,
    int transition_rank,
    float* __restrict__ transition_matrix,
    float* __restrict__ row_sums) {
    const int row = blockIdx.x;
    const int dst = threadIdx.x;
    __shared__ float scratch[kMaxNumWarps];
    if (row >= num_states || dst >= num_states) {
        return;
    }
    float raw_value = 0.0f;
    if (transition_mask[row * num_states + dst]) {
        for (int r = 0; r < transition_rank; ++r) {
            raw_value += transition_source_logits[row * transition_rank + r]
                * transition_dest_logits[r * num_states + dst];
        }
    }
    const float row_sum = block_reduce_sum_128(raw_value, scratch);
    if (dst == 0) {
        row_sums[row] = row_sum;
    }
    transition_matrix[row * num_states + dst] = raw_value / fmaxf(row_sum, 1.0e-20f);
}

__global__ void masked_transition_row_sums_kernel(
    const float* __restrict__ transition_source_logits,
    const float* __restrict__ transition_dest_logits,
    const bool* __restrict__ transition_mask,
    int num_states,
    int transition_rank,
    float* __restrict__ row_sums) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= num_states) {
        return;
    }
    float row_sum = 0.0f;
    for (int dst = 0; dst < num_states; ++dst) {
        row_sum += masked_transition_raw_value(
            transition_source_logits,
            transition_dest_logits,
            transition_mask,
            num_states,
            transition_rank,
            row,
            dst);
    }
    row_sums[row] = fmaxf(row_sum, 1.0e-20f);
}

__global__ void sparse_transition_raw_blocks_kernel(
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const int32_t* __restrict__ block_col_idx,
    const int32_t* __restrict__ block_dst_idx,
    const float* __restrict__ block_mask,
    int64_t nnz_blocks,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ row_sums) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_rows = nnz_blocks * static_cast<int64_t>(block_size);
    if (row_idx >= total_rows) {
        return;
    }
    const int nz = static_cast<int>(row_idx / block_size);
    const int src_offset = static_cast<int>(row_idx - (static_cast<int64_t>(nz) * block_size));
    const int src_base = block_col_idx[nz] * block_size;
    const int dst_base = block_dst_idx[nz] * block_size;
    const int src_state = src_base + src_offset;
    if (src_state >= padded_states) {
        return;
    }
    const float* src_row = transition_source_probs + static_cast<int64_t>(src_state) * transition_rank;
    const float* mask_row = block_mask + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
    float row_sum = 0.0f;
    for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
        const int dst_state = dst_base + dst_offset;
        const float mask_value = mask_row[dst_offset];
        if (mask_value == 0.0f || dst_state >= padded_states) {
            continue;
        }
        float raw_value = 0.0f;
        #pragma unroll 4
        for (int r = 0; r < transition_rank; ++r) {
            raw_value += src_row[r] * transition_dest_probs[static_cast<int64_t>(r) * padded_states + dst_state];
        }
        row_sum += raw_value * mask_value;
    }
    atomicAdd(row_sums + src_state, row_sum);
}

__global__ void sparse_transition_raw_blocks_kernel(
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const int32_t* __restrict__ block_col_idx,
    const int32_t* __restrict__ block_dst_idx,
    const float* __restrict__ block_mask,
    int64_t nnz_blocks,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ transition_blocks,
    float* __restrict__ row_sums) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_rows = nnz_blocks * static_cast<int64_t>(block_size);
    if (row_idx >= total_rows) {
        return;
    }
    const int nz = static_cast<int>(row_idx / block_size);
    const int src_offset = static_cast<int>(row_idx - (static_cast<int64_t>(nz) * block_size));
    const int src_base = block_col_idx[nz] * block_size;
    const int dst_base = block_dst_idx[nz] * block_size;
    const int src_state = src_base + src_offset;
    if (src_state >= padded_states) {
        return;
    }
    const float* src_row = transition_source_probs + static_cast<int64_t>(src_state) * transition_rank;
    const float* mask_row = block_mask + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
    float* block_row = transition_blocks + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
    float row_sum = 0.0f;
    for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
        const int dst_state = dst_base + dst_offset;
        const float mask_value = mask_row[dst_offset];
        float raw_value = 0.0f;
        if (mask_value != 0.0f && dst_state < padded_states) {
            #pragma unroll 4
            for (int r = 0; r < transition_rank; ++r) {
                raw_value += src_row[r] * transition_dest_probs[static_cast<int64_t>(r) * padded_states + dst_state];
            }
            raw_value *= mask_value;
        }
        block_row[dst_offset] = raw_value;
        row_sum += raw_value;
    }
    atomicAdd(row_sums + src_state, row_sum);
}

__global__ void sparse_transition_row_sums_from_logits_kernel(
    const float* __restrict__ transition_source_logits,
    const float* __restrict__ transition_dest_logits,
    const float* __restrict__ source_row_max,
    const float* __restrict__ source_row_inv_sum,
    const float* __restrict__ dest_row_max,
    const float* __restrict__ dest_row_inv_sum,
    const int32_t* __restrict__ block_col_idx,
    const int32_t* __restrict__ block_dst_idx,
    const float* __restrict__ block_mask,
    int64_t nnz_blocks,
    int num_states,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ row_sums) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_rows = nnz_blocks * static_cast<int64_t>(block_size);
    if (row_idx >= total_rows) {
        return;
    }
    const int nz = static_cast<int>(row_idx / block_size);
    const int src_offset = static_cast<int>(row_idx - (static_cast<int64_t>(nz) * block_size));
    const int src_base = block_col_idx[nz] * block_size;
    const int dst_base = block_dst_idx[nz] * block_size;
    const int src_state = src_base + src_offset;
    if (src_state >= num_states || src_state >= padded_states) {
        return;
    }
    const float* mask_row = block_mask + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
    float row_sum = 0.0f;
    for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
        const int dst_state = dst_base + dst_offset;
        const float mask_value = mask_row[dst_offset];
        if (mask_value == 0.0f || dst_state >= num_states || dst_state >= padded_states) {
            continue;
        }
        float raw_value = 0.0f;
        #pragma unroll 4
        for (int r = 0; r < transition_rank; ++r) {
            raw_value += softmax_prob_from_stats_value(
                transition_source_logits,
                transition_rank,
                src_state,
                r,
                source_row_max,
                source_row_inv_sum)
                * softmax_prob_from_stats_value(
                    transition_dest_logits,
                    num_states,
                    r,
                    dst_state,
                    dest_row_max,
                    dest_row_inv_sum);
        }
        row_sum += raw_value * mask_value;
    }
    atomicAdd(row_sums + src_state, row_sum);
}

template <typename packed_t, PackedTransitionFormat Format>
__global__ void sparse_transition_raw_blocks_packed_kernel(
    const packed_t* __restrict__ transition_source_packed,
    const float* __restrict__ transition_source_scales,
    const packed_t* __restrict__ transition_dest_packed,
    const float* __restrict__ transition_dest_scales,
    const int32_t* __restrict__ block_col_idx,
    const int32_t* __restrict__ block_dst_idx,
    const float* __restrict__ block_mask,
    int64_t nnz_blocks,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ transition_blocks,
    float* __restrict__ row_sums) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_rows = nnz_blocks * static_cast<int64_t>(block_size);
    if (row_idx >= total_rows) {
        return;
    }
    const int nz = static_cast<int>(row_idx / block_size);
    const int src_offset = static_cast<int>(row_idx - (static_cast<int64_t>(nz) * block_size));
    const int src_base = block_col_idx[nz] * block_size;
    const int dst_base = block_dst_idx[nz] * block_size;
    const int src_state = src_base + src_offset;
    if (src_state >= padded_states) {
        return;
    }
    const packed_t* src_row = transition_source_packed + static_cast<int64_t>(src_state) * transition_rank;
    const float src_scale = transition_source_scales[src_state];
    const float* mask_row = block_mask + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
    float* block_row = transition_blocks + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
    float row_sum = 0.0f;
    for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
        const int dst_state = dst_base + dst_offset;
        const float mask_value = mask_row[dst_offset];
        float raw_value = 0.0f;
        if (mask_value != 0.0f && dst_state < padded_states) {
            #pragma unroll 4
            for (int r = 0; r < transition_rank; ++r) {
                raw_value += unpack_packed_value<packed_t, Format>(src_row[r], src_scale)
                    * unpack_packed_value<packed_t, Format>(
                        transition_dest_packed[static_cast<int64_t>(r) * padded_states + dst_state],
                        transition_dest_scales[r]);
            }
            raw_value *= mask_value;
        }
        block_row[dst_offset] = raw_value;
        row_sum += raw_value;
    }
    atomicAdd(row_sums + src_state, row_sum);
}

__global__ void sparse_transition_normalize_blocks_kernel(
    const int32_t* __restrict__ block_col_idx,
    const float* __restrict__ row_sums,
    int64_t nnz_blocks,
    int padded_states,
    int block_size,
    float* __restrict__ transition_blocks) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_rows = nnz_blocks * static_cast<int64_t>(block_size);
    if (row_idx >= total_rows) {
        return;
    }
    const int nz = static_cast<int>(row_idx / block_size);
    const int src_offset = static_cast<int>(row_idx - (static_cast<int64_t>(nz) * block_size));
    const int src_state = block_col_idx[nz] * block_size + src_offset;
    if (src_state >= padded_states) {
        return;
    }
    const float denom = fmaxf(row_sums[src_state], 1.0e-20f);
    float* block_row = transition_blocks + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
    for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
        block_row[dst_offset] /= denom;
    }
}

__global__ void sparse_transition_row_proj_kernel(
    const float* __restrict__ grad_transition_blocks,
    const float* __restrict__ transition_blocks,
    const float* __restrict__ block_mask,
    const int32_t* __restrict__ src_row_ptr,
    const int32_t* __restrict__ src_nz_idx,
    int padded_states,
    int block_size,
    float* __restrict__ row_proj) {
    const int state = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (state >= padded_states) {
        return;
    }
    const int src_block = state / block_size;
    const int src_offset = state - (src_block * block_size);
    float proj = 0.0f;
    for (int entry = src_row_ptr[src_block]; entry < src_row_ptr[src_block + 1]; ++entry) {
        const int nz = src_nz_idx[entry];
        const float* grad_block = grad_transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
        const float* block = transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
        const float* mask = block_mask + static_cast<int64_t>(nz) * block_size * block_size;
        const int row_base = src_offset * block_size;
        for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
            proj += (grad_block[row_base + dst_offset] * mask[row_base + dst_offset]) * block[row_base + dst_offset];
        }
    }
    row_proj[state] = proj;
}

__global__ void sparse_transition_row_proj_grouped_kernel(
    const float* __restrict__ grad_transition_blocks,
    const float* __restrict__ transition_blocks,
    const float* __restrict__ block_mask,
    const int32_t* __restrict__ grouped_src_row_ptr,
    const int32_t* __restrict__ grouped_src_block_idx,
    const int32_t* __restrict__ src_nz_idx,
    int grouped_src_group_count,
    int padded_states,
    int block_size,
    float* __restrict__ row_proj) {
    const int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_grouped_states = grouped_src_group_count * block_size;
    if (linear >= total_grouped_states) {
        return;
    }
    const int group = linear / block_size;
    const int src_offset = linear - (group * block_size);
    const int src_block = grouped_src_block_idx[group];
    const int state = src_block * block_size + src_offset;
    if (state >= padded_states) {
        return;
    }
    float proj = 0.0f;
    for (int ordered_entry = grouped_src_row_ptr[group]; ordered_entry < grouped_src_row_ptr[group + 1]; ++ordered_entry) {
        const int nz = src_nz_idx[ordered_entry];
        const float* grad_block = grad_transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
        const float* block = transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
        const float* mask = block_mask + static_cast<int64_t>(nz) * block_size * block_size;
        const int row_base = src_offset * block_size;
        for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
            proj += (grad_block[row_base + dst_offset] * mask[row_base + dst_offset]) * block[row_base + dst_offset];
        }
    }
    row_proj[state] = proj;
}

__device__ __forceinline__ int find_grouped_src_group_for_block(
    const int32_t* __restrict__ grouped_src_block_idx,
    int grouped_src_group_count,
    int src_block) {
    int lo = 0;
    int hi = grouped_src_group_count - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const int block = grouped_src_block_idx[mid];
        if (block == src_block) {
            return mid;
        }
        if (block < src_block) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

__global__ void sparse_transition_source_grad_kernel(
    const float* __restrict__ grad_transition_blocks,
    const float* __restrict__ block_mask,
    const int32_t* __restrict__ src_row_ptr,
    const int32_t* __restrict__ src_nz_idx,
    const int32_t* __restrict__ block_dst_idx,
    const float* __restrict__ transition_dest_probs,
    const float* __restrict__ row_sums,
    const float* __restrict__ row_proj,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ grad_transition_source_probs) {
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int state = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (r >= transition_rank || state >= padded_states) {
        return;
    }
    const int src_block = state / block_size;
    const int src_offset = state - (src_block * block_size);
    const float denom = fmaxf(row_sums[state], 1.0e-20f);
    const float proj = row_proj[state];
    float grad = 0.0f;
    for (int entry = src_row_ptr[src_block]; entry < src_row_ptr[src_block + 1]; ++entry) {
        const int nz = src_nz_idx[entry];
        const int dst_block = block_dst_idx[nz];
        const int dst_base = dst_block * block_size;
        const float* grad_block = grad_transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
        const float* mask = block_mask + static_cast<int64_t>(nz) * block_size * block_size;
        const int row_base = src_offset * block_size;
        for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
            const float mask_value = mask[row_base + dst_offset];
            if (mask_value == 0.0f) {
                continue;
            }
            const float grad_raw = (((grad_block[row_base + dst_offset] * mask_value) - proj) / denom) * mask_value;
            grad += grad_raw * transition_dest_probs[r * padded_states + (dst_base + dst_offset)];
        }
    }
    grad_transition_source_probs[state * transition_rank + r] = grad;
}

__global__ void sparse_transition_source_grad_grouped_kernel(
    const float* __restrict__ grad_transition_blocks,
    const float* __restrict__ block_mask,
    const int32_t* __restrict__ grouped_src_row_ptr,
    const int32_t* __restrict__ grouped_src_block_idx,
    const int32_t* __restrict__ src_nz_idx,
    const int32_t* __restrict__ block_dst_idx,
    const float* __restrict__ transition_dest_probs,
    const float* __restrict__ row_sums,
    const float* __restrict__ row_proj,
    int grouped_src_group_count,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ grad_transition_source_probs) {
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int linear = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int total_grouped_states = grouped_src_group_count * block_size;
    if (r >= transition_rank || linear >= total_grouped_states) {
        return;
    }
    const int group = linear / block_size;
    const int src_offset = linear - (group * block_size);
    const int src_block = grouped_src_block_idx[group];
    const int state = src_block * block_size + src_offset;
    if (state >= padded_states) {
        return;
    }
    const float denom = fmaxf(row_sums[state], 1.0e-20f);
    const float proj = row_proj[state];
    float grad = 0.0f;
    for (int ordered_entry = grouped_src_row_ptr[group]; ordered_entry < grouped_src_row_ptr[group + 1]; ++ordered_entry) {
        const int nz = src_nz_idx[ordered_entry];
        const int dst_block = block_dst_idx[nz];
        const int dst_base = dst_block * block_size;
        const float* grad_block = grad_transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
        const float* mask = block_mask + static_cast<int64_t>(nz) * block_size * block_size;
        const int row_base = src_offset * block_size;
        for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
            const float mask_value = mask[row_base + dst_offset];
            if (mask_value == 0.0f) {
                continue;
            }
            const float grad_raw = (((grad_block[row_base + dst_offset] * mask_value) - proj) / denom) * mask_value;
            grad += grad_raw * transition_dest_probs[r * padded_states + (dst_base + dst_offset)];
        }
    }
    grad_transition_source_probs[state * transition_rank + r] = grad;
}

__global__ void sparse_transition_source_grad_grouped_compressed_kernel(
    const float* __restrict__ grad_transition_blocks,
    const float* __restrict__ block_mask,
    const int32_t* __restrict__ grouped_src_row_ptr,
    const int32_t* __restrict__ grouped_src_block_idx,
    const int32_t* __restrict__ src_nz_idx,
    const int32_t* __restrict__ block_dst_idx,
    const float* __restrict__ transition_dest_probs,
    const float* __restrict__ row_sums,
    const float* __restrict__ row_proj,
    int grouped_src_group_count,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ grad_transition_source_probs_compressed) {
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int linear = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int total_grouped_states = grouped_src_group_count * block_size;
    if (r >= transition_rank || linear >= total_grouped_states) {
        return;
    }
    const int group = linear / block_size;
    const int src_offset = linear - (group * block_size);
    const int src_block = grouped_src_block_idx[group];
    const int state = src_block * block_size + src_offset;
    if (state >= padded_states) {
        return;
    }
    const float denom = fmaxf(row_sums[state], 1.0e-20f);
    const float proj = row_proj[state];
    float grad = 0.0f;
    for (int ordered_entry = grouped_src_row_ptr[group]; ordered_entry < grouped_src_row_ptr[group + 1]; ++ordered_entry) {
        const int nz = src_nz_idx[ordered_entry];
        const int dst_block = block_dst_idx[nz];
        const int dst_base = dst_block * block_size;
        const float* grad_block = grad_transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
        const float* mask = block_mask + static_cast<int64_t>(nz) * block_size * block_size;
        const int row_base = src_offset * block_size;
        for (int dst_offset = 0; dst_offset < block_size; ++dst_offset) {
            const float mask_value = mask[row_base + dst_offset];
            if (mask_value == 0.0f) {
                continue;
            }
            const float grad_raw = (((grad_block[row_base + dst_offset] * mask_value) - proj) / denom) * mask_value;
            grad += grad_raw * transition_dest_probs[r * padded_states + (dst_base + dst_offset)];
        }
    }
    grad_transition_source_probs_compressed[static_cast<int64_t>(linear) * transition_rank + r] = grad;
}

__global__ void scatter_grouped_source_grad_kernel(
    const float* __restrict__ grad_transition_source_probs_compressed,
    const int32_t* __restrict__ grouped_src_block_idx,
    int grouped_src_group_count,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ grad_transition_source_probs) {
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int linear = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int total_grouped_states = grouped_src_group_count * block_size;
    if (r >= transition_rank || linear >= total_grouped_states) {
        return;
    }
    const int group = linear / block_size;
    const int src_offset = linear - (group * block_size);
    const int src_block = grouped_src_block_idx[group];
    const int state = src_block * block_size + src_offset;
    if (state >= padded_states) {
        return;
    }
    grad_transition_source_probs[state * transition_rank + r] =
        grad_transition_source_probs_compressed[static_cast<int64_t>(linear) * transition_rank + r];
}

__global__ void sparse_transition_dest_grad_kernel(
    const float* __restrict__ grad_transition_blocks,
    const float* __restrict__ block_mask,
    const int32_t* __restrict__ block_row_ptr,
    const int32_t* __restrict__ block_col_idx,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ row_sums,
    const float* __restrict__ row_proj,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ grad_transition_dest_probs) {
    const int dst_state = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (dst_state >= padded_states || r >= transition_rank) {
        return;
    }
    const int dst_block = dst_state / block_size;
    const int dst_offset = dst_state - (dst_block * block_size);
    float grad = 0.0f;
    for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
        const int src_block = block_col_idx[nz];
        const int src_base = src_block * block_size;
        const float* grad_block = grad_transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
        const float* mask = block_mask + static_cast<int64_t>(nz) * block_size * block_size;
        for (int src_offset = 0; src_offset < block_size; ++src_offset) {
            const int src_state = src_base + src_offset;
            if (src_state >= padded_states) {
                break;
            }
            const float mask_value = mask[src_offset * block_size + dst_offset];
            if (mask_value == 0.0f) {
                continue;
            }
            const float denom = fmaxf(row_sums[src_state], 1.0e-20f);
            const float grad_raw = (
                ((grad_block[src_offset * block_size + dst_offset] * mask_value) - row_proj[src_state]) / denom
            ) * mask_value;
            grad += transition_source_probs[src_state * transition_rank + r] * grad_raw;
        }
    }
    grad_transition_dest_probs[r * padded_states + dst_state] = grad;
}

__global__ void sparse_transition_dest_grad_compressed_kernel(
    const float* __restrict__ grad_transition_blocks,
    const float* __restrict__ block_mask,
    const int32_t* __restrict__ active_dst_block_idx,
    const int32_t* __restrict__ block_row_ptr,
    const int32_t* __restrict__ block_col_idx,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ row_sums,
    const float* __restrict__ row_proj,
    int active_dst_block_count,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ grad_transition_dest_probs_compressed) {
    const int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int total_active_dst_states = active_dst_block_count * block_size;
    if (linear >= total_active_dst_states || r >= transition_rank) {
        return;
    }
    const int group = linear / block_size;
    const int dst_offset = linear - (group * block_size);
    const int dst_block = active_dst_block_idx[group];
    const int dst_state = dst_block * block_size + dst_offset;
    if (dst_state >= padded_states) {
        return;
    }
    float grad = 0.0f;
    for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
        const int src_block = block_col_idx[nz];
        const int src_base = src_block * block_size;
        const int active_src = min(block_size, padded_states - src_base);
        const float* grad_block = grad_transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
        const float* mask = block_mask + static_cast<int64_t>(nz) * block_size * block_size;
        for (int src_offset = 0; src_offset < active_src; ++src_offset) {
            const int src_state = src_base + src_offset;
            const float denom = fmaxf(row_sums[src_state], 1.0e-20f);
            const float proj = row_proj[src_state];
            const float mask_value = mask[src_offset * block_size + dst_offset];
            if (mask_value == 0.0f) {
                continue;
            }
            const float grad_raw = (((grad_block[src_offset * block_size + dst_offset] * mask_value) - proj) / denom) * mask_value;
            grad += transition_source_probs[src_state * transition_rank + r] * grad_raw;
        }
    }
    grad_transition_dest_probs_compressed[static_cast<int64_t>(r) * total_active_dst_states + linear] = grad;
}

__global__ void scatter_compressed_dest_grad_kernel(
    const float* __restrict__ grad_transition_dest_probs_compressed,
    const int32_t* __restrict__ active_dst_block_idx,
    int active_dst_block_count,
    int padded_states,
    int transition_rank,
    int block_size,
    float* __restrict__ grad_transition_dest_probs) {
    const int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int total_active_dst_states = active_dst_block_count * block_size;
    if (linear >= total_active_dst_states || r >= transition_rank) {
        return;
    }
    const int group = linear / block_size;
    const int dst_offset = linear - (group * block_size);
    const int dst_block = active_dst_block_idx[group];
    const int dst_state = dst_block * block_size + dst_offset;
    if (dst_state >= padded_states) {
        return;
    }
    grad_transition_dest_probs[r * padded_states + dst_state] =
        grad_transition_dest_probs_compressed[static_cast<int64_t>(r) * total_active_dst_states + linear];
}

template <typename packed_t, PackedTransitionFormat Format>
__global__ void pack_transition_table_per_row_kernel(
    const float* __restrict__ input,
    int cols,
    packed_t* __restrict__ packed,
    float* __restrict__ scales) {
    const int row = blockIdx.x;
    __shared__ float scratch[kMaxNumWarps];
    float row_abs_max_local = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        row_abs_max_local = fmaxf(row_abs_max_local, fabsf(input[row * cols + col]));
    }
    const float row_abs_max = block_reduce_max_128(row_abs_max_local, scratch);
    const float scale = fmaxf(row_abs_max / packed_quant_max<Format>(), 1.0e-12f);
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = input[row * cols + col];
        packed[row * cols + col] = pack_packed_value<packed_t, Format>(value, 1.0f / scale);
    }
}

template <typename scalar_t, int StaticTransitionRank = -1, bool InputsAreLogits = false>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_forward_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    float score_clamp_min,
    float score_clamp_max,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int s = threadIdx.x;
    const int kNumStates = static_cast<int>(blockDim.x);
    const int kNumWarps = (kNumStates + kWarpSize - 1) / kWarpSize;
    const int rank = StaticTransitionRank > 0 ? StaticTransitionRank : transition_rank;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * rank);
    float* prev_prob = dest_shared + (rank * kNumStates);
    float* latent = prev_prob + kNumStates;
    float* scratch = latent + rank;

    if constexpr (InputsAreLogits) {
        for (int row = 0; row < kNumStates; ++row) {
            const float logit = s < rank ? transition_source_probs[row * rank + s] : -INFINITY;
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = s < rank ? fast_exp(logit - row_max) : 0.0f;
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            if (s < rank) {
                source_shared[row * rank + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
            }
        }
        for (int row = 0; row < rank; ++row) {
            const float logit = transition_dest_probs[row * kNumStates + s];
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = fast_exp(logit - row_max);
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            dest_shared[row * kNumStates + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
        }
    } else {
        copy_float_matrix_pair_to_shared_async_or_sync(
            source_shared,
            transition_source_probs,
            rank,
            kNumStates,
            rank,
            rank,
            dest_shared,
            transition_dest_probs,
            kNumStates,
            rank,
            kNumStates,
            kNumStates);
    }
    __syncthreads();
    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float prev_prob_value = fast_exp(load_as_float(initial_log_belief + (b * kNumStates + s)));
        prev_prob[s] = prev_prob_value;
        __syncthreads();

        float last_q_log = load_as_float(initial_log_belief + (b * kNumStates + s));

        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int base = (b * seq_len + pos) * kNumStates;

                if constexpr (StaticTransitionRank == 8) {
                    compute_latent_small_rank_128<8>(prev_prob, source_shared, latent, scratch, s);
                } else if constexpr (StaticTransitionRank == 16) {
                    compute_latent_small_rank_128<16>(prev_prob, source_shared, latent, scratch, s);
                } else if constexpr (StaticTransitionRank == 32) {
                    compute_latent_small_rank_128<32>(prev_prob, source_shared, latent, scratch, s);
                } else {
                    if (s < rank) {
                        float latent_val = 0.0f;
                        #pragma unroll 4
                        for (int i = 0; i < kNumStates; ++i) {
                            latent_val += prev_prob[i] * source_shared[i * rank + s];
                        }
                        latent[s] = latent_val;
                    }
                    __syncthreads();
                }

                float mix_prob = 0.0f;
                if constexpr (StaticTransitionRank > 0) {
                    #pragma unroll
                    for (int r = 0; r < StaticTransitionRank; ++r) {
                        mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                    }
                } else {
                    #pragma unroll 4
                    for (int r = 0; r < rank; ++r) {
                        mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                    }
                }
                const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
                const float pred_log = fast_log(pred_prob);
                const float prior_value = pred_log + load_as_float(transition_context + (base + s));
                const float obs = load_as_float(local_logits + (base + s))
                    + transition_gate * apply_score_clamp(prior_value, score_clamp_min, score_clamp_max);

                float obs_exp = 0.0f;
                float inv_obs_sum = 0.0f;
                const float log_norm = block_log_softmax_norm_128(obs, scratch, obs_exp, inv_obs_sum);
                const float q_log = obs - log_norm;

                beliefs[base + s] = store_from_float<scalar_t>(q_log);
                last_q_log = q_log;
                prev_prob_value = obs_exp * inv_obs_sum;
                prev_prob[s] = prev_prob_value;
                __syncthreads();
            }
        }

        final_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(last_q_log);
        __syncthreads();
    }
}

template <typename scalar_t, bool StoreBeliefs = true>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_forward_chunk_dense_128_rank8_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    float score_clamp_min,
    float score_clamp_max,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    constexpr int kNumStates = 128;
    constexpr int kTransitionRank = 8;
    const int s = threadIdx.x;
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * kTransitionRank);
    float* latent = dest_shared + (kTransitionRank * kNumStates);
    float* scratch = latent + kTransitionRank;

    copy_float_matrix_pair_to_shared_async_or_sync(
        source_shared,
        transition_source_probs,
        kTransitionRank,
        kNumStates,
        kTransitionRank,
        kTransitionRank,
        dest_shared,
        transition_dest_probs,
        kNumStates,
        kTransitionRank,
        kNumStates,
        kNumStates);
    __syncthreads();

    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;
    const int sequence_tile_size = max(chunk_len, 1);

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float prev_prob_value = fast_exp(load_as_float(initial_log_belief + (b * kNumStates + s)));
        float last_q_log = load_as_float(initial_log_belief + (b * kNumStates + s));

        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * kNumStates;
                compute_latent_small_rank_128_from_register<kTransitionRank>(
                    prev_prob_value,
                    source_shared,
                    latent,
                    scratch,
                    s);

                float mix_prob = 0.0f;
                #pragma unroll
                for (int r = 0; r < kTransitionRank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
                const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
                const float pred_log = fast_log(pred_prob);
                const float prior_value = pred_log + load_as_float(transition_context + (base + s));
                const float obs = load_as_float(local_logits + (base + s))
                    + transition_gate * apply_score_clamp(prior_value, score_clamp_min, score_clamp_max);
                float obs_exp = 0.0f;
                float inv_obs_sum = 0.0f;
                const float log_norm = block_log_softmax_norm_128(obs, scratch, obs_exp, inv_obs_sum);
                const float q_log = obs - log_norm;
                if constexpr (StoreBeliefs) {
                    beliefs[base + s] = store_from_float<scalar_t>(q_log);
                }
                last_q_log = q_log;
                prev_prob_value = obs_exp * inv_obs_sum;
            }
        }

        final_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(last_q_log);
        __syncthreads();
    }
}

template <typename scalar_t, bool StoreBeliefs = true>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_forward_chunk_dense_128_rank8_pair_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    float score_clamp_min,
    float score_clamp_max,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    constexpr int kNumStates = 128;
    constexpr int kTransitionRank = 8;
    constexpr int kStatesPerThread = 2;
    const int pair_idx = threadIdx.x;
    const int state0 = pair_idx * kStatesPerThread;
    const int state1 = state0 + 1;
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * kTransitionRank);
    float* latent = dest_shared + (kTransitionRank * kNumStates);
    float* scratch = latent + kTransitionRank;

    copy_float_matrix_pair_to_shared_async_or_sync(
        source_shared,
        transition_source_probs,
        kTransitionRank,
        kNumStates,
        kTransitionRank,
        kTransitionRank,
        dest_shared,
        transition_dest_probs,
        kNumStates,
        kTransitionRank,
        kNumStates,
        kNumStates);
    __syncthreads();

    const FloatPair stay_prob{
        transition_stay_probs[state0],
        transition_stay_probs[state1],
    };
    const FloatPair one_minus_stay{
        1.0f - stay_prob.x,
        1.0f - stay_prob.y,
    };
    const int sequence_tile_size = max(chunk_len, 1);

    while (true) {
        if (pair_idx == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        FloatPair prev_prob_value = load_pair_as_float(initial_log_belief + (b * kNumStates + state0));
        prev_prob_value.x = fast_exp(prev_prob_value.x);
        prev_prob_value.y = fast_exp(prev_prob_value.y);
        FloatPair last_q_log = load_pair_as_float(initial_log_belief + (b * kNumStates + state0));

        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * kNumStates;
                compute_latent_small_rank_128_from_pair_register<kTransitionRank>(
                    prev_prob_value,
                    source_shared,
                    latent,
                    scratch,
                    pair_idx);

                FloatPair mix_prob{0.0f, 0.0f};
                #pragma unroll
                for (int r = 0; r < kTransitionRank; ++r) {
                    const int dest_row = r * kNumStates;
                    mix_prob.x += latent[r] * dest_shared[dest_row + state0];
                    mix_prob.y += latent[r] * dest_shared[dest_row + state1];
                }
                const FloatPair pred_prob{
                    fmaxf(stay_prob.x * prev_prob_value.x + one_minus_stay.x * mix_prob.x, 1.0e-20f),
                    fmaxf(stay_prob.y * prev_prob_value.y + one_minus_stay.y * mix_prob.y, 1.0e-20f),
                };
                const FloatPair pred_log{fast_log(pred_prob.x), fast_log(pred_prob.y)};
                const FloatPair context_pair = load_pair_as_float(transition_context + (base + state0));
                const FloatPair logits_pair = load_pair_as_float(local_logits + (base + state0));
                const FloatPair prior_value{
                    pred_log.x + context_pair.x,
                    pred_log.y + context_pair.y,
                };
                const FloatPair obs{
                    logits_pair.x + transition_gate * apply_score_clamp(prior_value.x, score_clamp_min, score_clamp_max),
                    logits_pair.y + transition_gate * apply_score_clamp(prior_value.y, score_clamp_min, score_clamp_max),
                };
                FloatPair obs_exp{0.0f, 0.0f};
                float inv_obs_sum = 0.0f;
                const float log_norm = block_log_softmax_norm_128_pair(obs, scratch, obs_exp, inv_obs_sum);
                const FloatPair q_log{obs.x - log_norm, obs.y - log_norm};
                if constexpr (StoreBeliefs) {
                    store_pair_from_float(beliefs + (base + state0), q_log);
                }
                last_q_log = q_log;
                prev_prob_value = {obs_exp.x * inv_obs_sum, obs_exp.y * inv_obs_sum};
            }
        }

        store_pair_from_float(final_log_belief + (b * kNumStates + state0), last_q_log);
        __syncthreads();
    }
}

template <typename scalar_t, bool DirectGradReduce = false>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_backward_chunk_dense_128_rank8_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    float score_clamp_min,
    float score_clamp_max,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_per_batch,
    float* __restrict__ grad_transition_dest_per_batch,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate_per_batch,
    float* __restrict__ grad_transition_stay_per_batch) {
    constexpr int kNumStates = 128;
    constexpr int kTransitionRank = 8;
    const int s = threadIdx.x;
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * kTransitionRank);
    float* latent = dest_shared + (kTransitionRank * kNumStates);
    float* grad_mix = latent + kTransitionRank;
    float* dlatent = grad_mix + kNumStates;
    float* scratch = dlatent + kTransitionRank;
    float* grad_source_shared = nullptr;
    float* grad_dest_shared = nullptr;
    float* grad_stay_shared = nullptr;
    if constexpr (DirectGradReduce) {
        grad_source_shared = scratch + ((kNumStates / kWarpSize) * kTransitionRank);
        grad_dest_shared = grad_source_shared + (kNumStates * kTransitionRank);
        grad_stay_shared = grad_dest_shared + (kTransitionRank * kNumStates);
    }
    float* grad_source_batch = nullptr;
    float* grad_dest_batch = nullptr;
    float* grad_stay_batch = nullptr;

    for (int idx = s; idx < kNumStates * kTransitionRank; idx += blockDim.x) {
        source_shared[idx] = transition_source_probs[idx];
    }
    for (int idx = s; idx < kTransitionRank * kNumStates; idx += blockDim.x) {
        dest_shared[idx] = transition_dest_probs[idx];
    }
    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;
    __syncthreads();

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        if constexpr (DirectGradReduce) {
            for (int idx = s; idx < kNumStates * kTransitionRank; idx += blockDim.x) {
                grad_source_shared[idx] = 0.0f;
            }
            for (int idx = s; idx < kTransitionRank * kNumStates; idx += blockDim.x) {
                grad_dest_shared[idx] = 0.0f;
            }
            grad_stay_shared[s] = 0.0f;
        }
        if constexpr (DirectGradReduce) {
            const int worker_slot = static_cast<int>(blockIdx.x);
            grad_source_batch = grad_transition_source_per_batch + worker_slot * kNumStates * kTransitionRank;
            grad_dest_batch = grad_transition_dest_per_batch + worker_slot * kTransitionRank * kNumStates;
            grad_stay_batch = grad_transition_stay_per_batch + worker_slot * kNumStates;
        } else {
            grad_source_batch = grad_transition_source_per_batch + b * kNumStates * kTransitionRank;
            grad_dest_batch = grad_transition_dest_per_batch + b * kTransitionRank * kNumStates;
            grad_stay_batch = grad_transition_stay_per_batch + b * kNumStates;
        }
        __syncthreads();

        float carry_value = load_as_float(grad_final_belief + (b * kNumStates + s));
        float gate_grad_accum = 0.0f;
        float q_prob_value = 0.0f;
        if (seq_len > 0) {
            const int last_pos = seq_len - 1;
            q_prob_value = fast_exp(load_as_float(beliefs + ((static_cast<int64_t>(b) * seq_len + last_pos) * kNumStates + s)));
        }

        const int sequence_tile_size = max(chunk_len, 1);
        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * kNumStates;
                const float prev_prob_value = pos == 0
                    ? fast_exp(load_as_float(initial_log_belief + (b * kNumStates + s)))
                    : fast_exp(load_as_float(beliefs + ((static_cast<int64_t>(b) * seq_len + (pos - 1)) * kNumStates + s)));

                compute_latent_small_rank_128_from_register<kTransitionRank>(
                    prev_prob_value,
                    source_shared,
                    latent,
                    scratch,
                    s);

                float mix_prob = 0.0f;
                #pragma unroll
                for (int r = 0; r < kTransitionRank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
                const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
                const float pred_log = fast_log(pred_prob);
                const float transition_context_value = load_as_float(transition_context + (base + s));
                const float gq = load_as_float(grad_beliefs + (base + s)) + carry_value;
                const float gq_sum = block_reduce_sum_128(gq, scratch);
                const float ga = gq - q_prob_value * gq_sum;
                const float prior_value = pred_log + transition_context_value;
                const float clamped_prior = apply_score_clamp(
                    prior_value,
                    score_clamp_min,
                    score_clamp_max);
                const float grad_prior = (transition_gate * ga) * score_clamp_grad(
                    prior_value,
                    score_clamp_min,
                    score_clamp_max);
                const float grad_pred_prob = grad_prior / pred_prob;

                grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
                grad_transition_context[base + s] = store_from_float<scalar_t>(grad_prior);
                grad_mix[s] = grad_pred_prob * one_minus_stay;
                if constexpr (DirectGradReduce) {
                    grad_stay_shared[s] += grad_pred_prob * (prev_prob_value - mix_prob);
                } else {
                    grad_stay_batch[s] += grad_pred_prob * (prev_prob_value - mix_prob);
                }
                gate_grad_accum += ga * clamped_prior;
                const float direct_prev_grad_prob = grad_pred_prob * stay_prob;
                __syncthreads();

                if (s < kTransitionRank) {
                    float dlatent_val = 0.0f;
                    #pragma unroll 4
                    for (int j = 0; j < kNumStates; ++j) {
                        dlatent_val += grad_mix[j] * dest_shared[s * kNumStates + j];
                    }
                    dlatent[s] = dlatent_val;
                    #pragma unroll 4
                    for (int j = 0; j < kNumStates; ++j) {
                        if constexpr (DirectGradReduce) {
                            grad_dest_shared[s * kNumStates + j] += latent[s] * grad_mix[j];
                        } else {
                            grad_dest_batch[s * kNumStates + j] += latent[s] * grad_mix[j];
                        }
                    }
                }
                __syncthreads();

                float prev_grad_prob = direct_prev_grad_prob;
                #pragma unroll
                for (int r = 0; r < kTransitionRank; ++r) {
                    prev_grad_prob += dlatent[r] * source_shared[s * kTransitionRank + r];
                    if constexpr (DirectGradReduce) {
                        grad_source_shared[s * kTransitionRank + r] += prev_prob_value * dlatent[r];
                    } else {
                        grad_source_batch[s * kTransitionRank + r] += prev_prob_value * dlatent[r];
                    }
                }
                carry_value = prev_grad_prob * prev_prob_value;
                q_prob_value = prev_prob_value;
                __syncthreads();
            }
        }

        grad_initial_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(carry_value);
        const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
        if constexpr (DirectGradReduce) {
            for (int idx = s; idx < kNumStates * kTransitionRank; idx += blockDim.x) {
                grad_source_batch[idx] += grad_source_shared[idx];
            }
            for (int idx = s; idx < kTransitionRank * kNumStates; idx += blockDim.x) {
                grad_dest_batch[idx] += grad_dest_shared[idx];
            }
            grad_stay_batch[s] += grad_stay_shared[s];
            if (s == 0) {
                grad_transition_gate_per_batch[blockIdx.x] += gate_sum;
            }
        } else if (s == 0) {
            grad_transition_gate_per_batch[b] += gate_sum;
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename packed_t, PackedTransitionFormat Format, int StaticTransitionRank = -1>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_forward_chunk_packed_kernel(
    const scalar_t* __restrict__ local_logits,
    const packed_t* __restrict__ transition_source_packed,
    const float* __restrict__ transition_source_scales,
    const packed_t* __restrict__ transition_dest_packed,
    const float* __restrict__ transition_dest_scales,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int s = threadIdx.x;
    const int kNumStates = static_cast<int>(blockDim.x);
    const int rank = StaticTransitionRank > 0 ? StaticTransitionRank : transition_rank;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* prev_prob = shared_mem;
    float* latent = prev_prob + kNumStates;
    const int kNumWarps = (kNumStates + kWarpSize - 1) / kWarpSize;
    float* scratch = latent + rank;
    char* tensor_core_bytes = reinterpret_cast<char*>(scratch + kNumWarps);
    auto tensor_core_addr = reinterpret_cast<std::uintptr_t>(tensor_core_bytes);
    tensor_core_addr = (tensor_core_addr + 15u) & ~static_cast<std::uintptr_t>(15u);
    using tensor_core_input_t = tensor_core_input_type_t<scalar_t>;
    tensor_core_input_t* tensor_core_lhs = reinterpret_cast<tensor_core_input_t*>(tensor_core_addr);
    tensor_core_input_t* tensor_core_rhs = tensor_core_lhs + (kTensorCoreTile * kTensorCoreTile);
    float* tensor_core_accum = reinterpret_cast<float*>(tensor_core_rhs + (kTensorCoreTile * kTensorCoreTile));
    float* tensor_core_matrix = tensor_core_accum + (kTensorCoreTile * kTensorCoreTile);
    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;
    const bool use_tensor_core_math =
        tensor_core_math_enabled_for_scalar<scalar_t>()
        && (kNumStates >= kTensorCoreTile)
        && (rank >= kTensorCoreTile);

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float prev_prob_value = fast_exp(load_as_float(initial_log_belief + (b * kNumStates + s)));
        prev_prob[s] = prev_prob_value;
        __syncthreads();

        float last_q_log = load_as_float(initial_log_belief + (b * kNumStates + s));

        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int base = (b * seq_len + pos) * kNumStates;

                if (use_tensor_core_math) {
                    for (int r = s; r < rank; r += blockDim.x) {
                        latent[r] = 0.0f;
                    }
                    __syncthreads();
                    for (int rank_start = 0; rank_start < rank; rank_start += kTensorCoreTile) {
                        const int active_rank = min(kTensorCoreTile, rank - rank_start);
                        if (active_rank == kTensorCoreTile) {
                            for (int src_start = 0; src_start < kNumStates; src_start += kTensorCoreTile) {
                                const int active_src = min(kTensorCoreTile, kNumStates - src_start);
#if __CUDA_ARCH__ >= 700
                                if (active_src == kTensorCoreTile) {
                                    load_packed_matrix_tile_rowmajor_16x16<packed_t, Format>(
                                        transition_source_packed,
                                        transition_source_scales,
                                        rank,
                                        src_start,
                                        rank_start,
                                        active_src,
                                        active_rank,
                                        tensor_core_matrix);
                                    wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                        prev_prob + src_start,
                                        tensor_core_matrix,
                                        kTensorCoreTile,
                                        tensor_core_lhs,
                                        tensor_core_rhs,
                                        tensor_core_accum,
                                        latent + rank_start);
                                } else
#endif
                                {
                                    for (int r = s; r < active_rank; r += blockDim.x) {
                                        latent[rank_start + r] += packed_column_dot_lowp<packed_t, Format>(
                                            prev_prob + src_start,
                                            transition_source_packed + static_cast<int64_t>(src_start) * rank + rank_start + r,
                                            transition_source_scales + src_start,
                                            active_src,
                                            rank,
                                            0);
                                    }
                                    __syncthreads();
                                }
                            }
                        } else {
                            for (int r = s; r < active_rank; r += blockDim.x) {
                                latent[rank_start + r] = packed_column_dot_lowp<packed_t, Format>(
                                    prev_prob,
                                    transition_source_packed + rank_start + r,
                                    transition_source_scales,
                                    kNumStates,
                                    rank,
                                    0);
                            }
                            __syncthreads();
                        }
                    }
                    for (int dst = s; dst < kNumStates; dst += blockDim.x) {
                        prev_prob[dst] = 0.0f;
                    }
                    __syncthreads();
                    for (int rank_start = 0; rank_start < rank; rank_start += kTensorCoreTile) {
                        const int active_rank = min(kTensorCoreTile, rank - rank_start);
                        if (active_rank == kTensorCoreTile) {
                            for (int dst_start = 0; dst_start < kNumStates; dst_start += kTensorCoreTile) {
                                const int active_dst = min(kTensorCoreTile, kNumStates - dst_start);
#if __CUDA_ARCH__ >= 700
                                if (active_dst == kTensorCoreTile) {
                                    load_packed_matrix_tile_rowmajor_16x16<packed_t, Format>(
                                        transition_dest_packed,
                                        transition_dest_scales,
                                        kNumStates,
                                        rank_start,
                                        dst_start,
                                        active_rank,
                                        active_dst,
                                        tensor_core_matrix);
                                    wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                        latent + rank_start,
                                        tensor_core_matrix,
                                        kTensorCoreTile,
                                        tensor_core_lhs,
                                        tensor_core_rhs,
                                        tensor_core_accum,
                                        prev_prob + dst_start);
                                } else
#endif
                                {
                                    for (int dst_local = s; dst_local < active_dst; dst_local += blockDim.x) {
                                        float mix_value = prev_prob[dst_start + dst_local];
                                        for (int r = 0; r < active_rank; ++r) {
                                            mix_value += latent[rank_start + r] * unpack_packed_value<packed_t, Format>(
                                                transition_dest_packed[static_cast<int64_t>(rank_start + r) * kNumStates + dst_start + dst_local],
                                                transition_dest_scales[rank_start + r]);
                                        }
                                        prev_prob[dst_start + dst_local] = mix_value;
                                    }
                                    __syncthreads();
                                }
                            }
                        } else {
                            for (int dst = s; dst < kNumStates; dst += blockDim.x) {
                                float mix_value = prev_prob[dst];
                                for (int r = 0; r < active_rank; ++r) {
                                    mix_value += latent[rank_start + r] * unpack_packed_value<packed_t, Format>(
                                        transition_dest_packed[static_cast<int64_t>(rank_start + r) * kNumStates + dst],
                                        transition_dest_scales[rank_start + r]);
                                }
                                prev_prob[dst] = mix_value;
                            }
                            __syncthreads();
                        }
                    }
                } else {
                    if (s < rank) {
                        latent[s] = packed_column_dot_lowp<packed_t, Format>(
                            prev_prob,
                            transition_source_packed,
                            transition_source_scales,
                            kNumStates,
                            rank,
                            s);
                    }
                    __syncthreads();
                    prev_prob[s] = packed_column_dot_lowp<packed_t, Format>(
                        latent,
                        transition_dest_packed,
                        transition_dest_scales,
                        rank,
                        kNumStates,
                        s);
                }
                __syncthreads();

                const float mix_prob = prev_prob[s];
                const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
                const float pred_log = fast_log(pred_prob);
                const float obs = load_as_float(local_logits + (base + s)) + transition_gate * (
                    pred_log + load_as_float(transition_context + (base + s))
                );

                float obs_exp = 0.0f;
                float inv_obs_sum = 0.0f;
                const float log_norm = block_log_softmax_norm_128(obs, scratch, obs_exp, inv_obs_sum);
                const float q_log = obs - log_norm;

                beliefs[base + s] = store_from_float<scalar_t>(q_log);
                last_q_log = q_log;
                prev_prob_value = obs_exp * inv_obs_sum;
                prev_prob[s] = prev_prob_value;
                __syncthreads();
            }
        }

        final_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(last_q_log);
        __syncthreads();
    }
}

template <typename scalar_t, int StaticTransitionRank = -1, bool InputsAreLogits = false>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_forward_composable_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const float* __restrict__ transition_stay_probs,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int s = threadIdx.x;
    const int kNumStates = static_cast<int>(blockDim.x);
    const int rank = StaticTransitionRank > 0 ? StaticTransitionRank : transition_rank;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * rank);
    float* prev_prob = dest_shared + (rank * kNumStates);
    float* latent = prev_prob + kNumStates;
    float* scratch = latent + rank;

    if constexpr (InputsAreLogits) {
        for (int row = 0; row < kNumStates; ++row) {
            const float logit = s < rank ? transition_source_probs[row * rank + s] : -INFINITY;
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = s < rank ? fast_exp(logit - row_max) : 0.0f;
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            if (s < rank) {
                source_shared[row * rank + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
            }
        }
        for (int row = 0; row < rank; ++row) {
            const float logit = transition_dest_probs[row * kNumStates + s];
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = fast_exp(logit - row_max);
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            dest_shared[row * kNumStates + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
        }
    } else {
        copy_float_matrix_pair_to_shared_async_or_sync(
            source_shared,
            transition_source_probs,
            rank,
            kNumStates,
            rank,
            rank,
            dest_shared,
            transition_dest_probs,
            kNumStates,
            rank,
            kNumStates,
            kNumStates);
    }
    __syncthreads();
    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float prev_prob_value = fast_exp(load_as_float(initial_log_belief + (b * kNumStates + s)));
        prev_prob[s] = prev_prob_value;
        __syncthreads();

        float last_q_log = load_as_float(initial_log_belief + (b * kNumStates + s));

        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int base = (b * seq_len + pos) * kNumStates;

                if (s < rank) {
                    float latent_val = 0.0f;
                    #pragma unroll 4
                    for (int i = 0; i < kNumStates; ++i) {
                        latent_val += prev_prob[i] * source_shared[i * rank + s];
                    }
                    latent[s] = latent_val;
                }
                __syncthreads();

                float mix_prob = 0.0f;
                if constexpr (StaticTransitionRank > 0) {
                    #pragma unroll
                    for (int r = 0; r < StaticTransitionRank; ++r) {
                        mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                    }
                } else {
                    #pragma unroll 4
                    for (int r = 0; r < rank; ++r) {
                        mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                    }
                }
                const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
                const float obs = load_as_float(local_logits + (base + s)) + load_as_float(transition_context + (base + s));

                const float obs_max = block_reduce_max_128(obs, scratch);
                const float obs_exp = fast_exp(obs - obs_max);
                const float alpha = obs_exp * pred_prob;
                const float alpha_sum = block_reduce_sum_128(alpha, scratch);
                const float q_prob = alpha / fmaxf(alpha_sum, 1.0e-20f);
                const float q_log = fast_log(fmaxf(q_prob, 1.0e-20f));

                beliefs[base + s] = store_from_float<scalar_t>(q_log);
                last_q_log = q_log;
                prev_prob_value = q_prob;
                prev_prob[s] = prev_prob_value;
                __syncthreads();
            }
        }

        final_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(last_q_log);
        __syncthreads();
    }
}

template <typename scalar_t, int StaticTransitionRank = -1, bool InputsAreLogits = false>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_forward_composable_chunk_summary_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const float* __restrict__ transition_stay_probs,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ final_log_belief) {
    const int s = threadIdx.x;
    const int kNumStates = static_cast<int>(blockDim.x);
    const int rank = StaticTransitionRank > 0 ? StaticTransitionRank : transition_rank;
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * rank);
    float* prev_prob = dest_shared + (rank * kNumStates);
    float* latent = prev_prob + kNumStates;
    float* scratch = latent + rank;

    if constexpr (InputsAreLogits) {
        for (int row = 0; row < kNumStates; ++row) {
            const float logit = s < rank ? transition_source_probs[row * rank + s] : -INFINITY;
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = s < rank ? fast_exp(logit - row_max) : 0.0f;
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            if (s < rank) {
                source_shared[row * rank + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
            }
        }
        for (int row = 0; row < rank; ++row) {
            const float logit = transition_dest_probs[row * kNumStates + s];
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = fast_exp(logit - row_max);
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            dest_shared[row * kNumStates + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
        }
    } else {
        copy_float_matrix_pair_to_shared_async_or_sync(
            source_shared,
            transition_source_probs,
            rank,
            kNumStates,
            rank,
            rank,
            dest_shared,
            transition_dest_probs,
            kNumStates,
            rank,
            kNumStates,
            kNumStates);
    }
    __syncthreads();
    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float prev_prob_value = fast_exp(load_as_float(initial_log_belief + (b * kNumStates + s)));
        prev_prob[s] = prev_prob_value;
        __syncthreads();

        float last_q_log = load_as_float(initial_log_belief + (b * kNumStates + s));
        for (int dt = 0; dt < chunk_len; ++dt) {
            const int pos = chunk_start + dt;
            if (pos >= seq_len) {
                break;
            }
            const int base = (b * seq_len + pos) * kNumStates;

            if (s < rank) {
                float latent_val = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < kNumStates; ++i) {
                    latent_val += prev_prob[i] * source_shared[i * rank + s];
                }
                latent[s] = latent_val;
            }
            __syncthreads();

            float mix_prob = 0.0f;
            if constexpr (StaticTransitionRank > 0) {
                #pragma unroll
                for (int r = 0; r < StaticTransitionRank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
            } else {
                #pragma unroll 4
                for (int r = 0; r < rank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
            }
            const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
            const float obs = load_as_float(local_logits + (base + s)) + load_as_float(transition_context + (base + s));

            const float obs_max = block_reduce_max_128(obs, scratch);
            const float obs_exp = fast_exp(obs - obs_max);
            const float alpha = obs_exp * pred_prob;
            const float alpha_sum = block_reduce_sum_128(alpha, scratch);
            const float q_prob = alpha / fmaxf(alpha_sum, 1.0e-20f);
            const float q_log = fast_log(fmaxf(q_prob, 1.0e-20f));

            last_q_log = q_log;
            prev_prob_value = q_prob;
            prev_prob[s] = prev_prob_value;
            __syncthreads();
        }

        final_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(last_q_log);
        __syncthreads();
    }
}

template <typename scalar_t, int StaticTransitionRank = -1, bool InputsAreLogits = false>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_forward_composable_chunk_finalize_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ chunk_initial_log_belief,
    const float* __restrict__ transition_stay_probs,
    int transition_rank,
    int seq_len,
    int total_batches,
    int num_chunks,
    int chunk_size,
    int total_tasks,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ beliefs) {
    const int s = threadIdx.x;
    const int kNumStates = static_cast<int>(blockDim.x);
    const int rank = StaticTransitionRank > 0 ? StaticTransitionRank : transition_rank;
    __shared__ int current_task;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * rank);
    float* prev_prob = dest_shared + (rank * kNumStates);
    float* latent = prev_prob + kNumStates;
    float* scratch = latent + rank;

    if constexpr (InputsAreLogits) {
        for (int row = 0; row < kNumStates; ++row) {
            const float logit = s < rank ? transition_source_probs[row * rank + s] : -INFINITY;
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = s < rank ? fast_exp(logit - row_max) : 0.0f;
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            if (s < rank) {
                source_shared[row * rank + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
            }
        }
        for (int row = 0; row < rank; ++row) {
            const float logit = transition_dest_probs[row * kNumStates + s];
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = fast_exp(logit - row_max);
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            dest_shared[row * kNumStates + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
        }
    } else {
        copy_float_matrix_pair_to_shared_async_or_sync(
            source_shared,
            transition_source_probs,
            rank,
            kNumStates,
            rank,
            rank,
            dest_shared,
            transition_dest_probs,
            kNumStates,
            rank,
            kNumStates,
            kNumStates);
    }
    __syncthreads();
    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;

    while (true) {
        if (s == 0) {
            current_task = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_task >= total_tasks) {
            break;
        }
        const int b = current_task / num_chunks;
        const int chunk_id = current_task - (b * num_chunks);
        const int chunk_start = chunk_id * chunk_size;
        const int chunk_len = min(chunk_size, seq_len - chunk_start);
        const int carry_base = ((chunk_id * total_batches) + b) * kNumStates;

        float prev_prob_value = fast_exp(load_as_float(chunk_initial_log_belief + carry_base + s));
        prev_prob[s] = prev_prob_value;
        __syncthreads();

        for (int dt = 0; dt < chunk_len; ++dt) {
            const int pos = chunk_start + dt;
            const int base = (b * seq_len + pos) * kNumStates;

            if (s < rank) {
                float latent_val = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < kNumStates; ++i) {
                    latent_val += prev_prob[i] * source_shared[i * rank + s];
                }
                latent[s] = latent_val;
            }
            __syncthreads();

            float mix_prob = 0.0f;
            if constexpr (StaticTransitionRank > 0) {
                #pragma unroll
                for (int r = 0; r < StaticTransitionRank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
            } else {
                #pragma unroll 4
                for (int r = 0; r < rank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
            }
            const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
            const float obs = load_as_float(local_logits + (base + s)) + load_as_float(transition_context + (base + s));

            const float obs_max = block_reduce_max_128(obs, scratch);
            const float obs_exp = fast_exp(obs - obs_max);
            const float alpha = obs_exp * pred_prob;
            const float alpha_sum = block_reduce_sum_128(alpha, scratch);
            const float q_prob = alpha / fmaxf(alpha_sum, 1.0e-20f);
            const float q_log = fast_log(fmaxf(q_prob, 1.0e-20f));

            beliefs[base + s] = store_from_float<scalar_t>(q_log);
            prev_prob_value = q_prob;
            prev_prob[s] = prev_prob_value;
            __syncthreads();
        }
    }
}

template <typename scalar_t>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_forward_masked_dense_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_logits,
    const float* __restrict__ transition_dest_logits,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const bool* __restrict__ transition_mask,
    const int64_t* __restrict__ seq_lens,
    float score_clamp_min,
    float score_clamp_max,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int s = threadIdx.x;
    const int num_states = static_cast<int>(blockDim.x);
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* transition_matrix = shared_mem;
    float* prev_prob = transition_matrix + (num_states * num_states);
    float* scratch = prev_prob + num_states;

    for (int row = 0; row < num_states; ++row) {
        float raw_value = 0.0f;
        if (transition_mask[row * num_states + s]) {
            for (int r = 0; r < transition_rank; ++r) {
                raw_value += transition_source_logits[row * transition_rank + r]
                    * transition_dest_logits[r * num_states + s];
            }
        }
        const float row_sum = block_reduce_sum_128(raw_value, scratch);
        transition_matrix[row * num_states + s] = raw_value / fmaxf(row_sum, 1.0e-20f);
        __syncthreads();
    }

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float prev_log_value = load_as_float(initial_log_belief + (b * num_states + s));
        float prev_prob_value = fast_exp(prev_log_value);
        prev_prob[s] = prev_prob_value;
        __syncthreads();

        float last_q_log = prev_log_value;
        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int base = (b * seq_len + pos) * num_states;

                float mix_prob = 0.0f;
                for (int src = 0; src < num_states; ++src) {
                    mix_prob += prev_prob[src] * transition_matrix[src * num_states + s];
                }
                const float pred_prob = fmaxf(
                    (transition_stay_probs[s] * prev_prob_value) + ((1.0f - transition_stay_probs[s]) * mix_prob),
                    1.0e-20f);
                const float pred_log = fast_log(pred_prob);
                const float prior_value = pred_log + load_as_float(transition_context + (base + s));
                const float filtered_value = load_as_float(local_logits + (base + s))
                    + transition_gate * apply_score_clamp(
                    prior_value,
                    score_clamp_min,
                    score_clamp_max);

                float obs_exp = 0.0f;
                float inv_obs_sum = 0.0f;
                const float log_norm = block_log_softmax_norm_128(filtered_value, scratch, obs_exp, inv_obs_sum);
                float next_log_value = filtered_value - log_norm;
                if (has_seq_lens && pos >= static_cast<int>(seq_lens[b])) {
                    next_log_value = prev_log_value;
                }

                beliefs[base + s] = store_from_float<scalar_t>(next_log_value);
                prev_log_value = next_log_value;
                prev_prob_value = fast_exp(next_log_value);
                prev_prob[s] = prev_prob_value;
                last_q_log = next_log_value;
                __syncthreads();
            }
        }

        final_log_belief[b * num_states + s] = store_from_float<scalar_t>(last_q_log);
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_backward_masked_dense_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const float* __restrict__ transition_source_logits,
    const float* __restrict__ transition_dest_logits,
    const bool* __restrict__ transition_mask,
    const float* __restrict__ transition_matrix_global,
    const float* __restrict__ row_sums,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const int64_t* __restrict__ seq_lens,
    float score_clamp_min,
    float score_clamp_max,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_per_batch,
    float* __restrict__ grad_transition_dest_per_batch,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate_per_batch,
    float* __restrict__ grad_transition_stay_per_batch) {
    const int s = threadIdx.x;
    const int num_states = static_cast<int>(blockDim.x);
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* transition_matrix = shared_mem;
    float* prev_prob = transition_matrix + (num_states * num_states);
    float* grad_mix = prev_prob + num_states;
    float* grad_stay_shared = grad_mix + num_states;
    float* scratch = grad_stay_shared + num_states;

    for (int idx = s; idx < num_states * num_states; idx += blockDim.x) {
        transition_matrix[idx] = transition_matrix_global[idx];
    }
    if (s < num_states) {
        grad_stay_shared[s] = 0.0f;
    }
    __syncthreads();

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float* grad_source_batch = grad_transition_source_per_batch + static_cast<int64_t>(b) * num_states * transition_rank;
        float* grad_dest_batch = grad_transition_dest_per_batch + static_cast<int64_t>(b) * transition_rank * num_states;
        float* grad_stay_batch = grad_transition_stay_per_batch + static_cast<int64_t>(b) * num_states;
        if (s < num_states) {
            grad_stay_shared[s] = 0.0f;
        }
        __syncthreads();

        float carry_value = load_as_float(grad_final_belief + (b * num_states + s));
        float gate_grad_accum = 0.0f;
        float q_prob_value = 0.0f;
        if (seq_len > 0) {
            const int last_pos = seq_len - 1;
            q_prob_value = fast_exp(load_as_float(beliefs + ((b * seq_len + last_pos) * num_states + s)));
        }

        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int base = (b * seq_len + pos) * num_states;
                const bool active_step = !has_seq_lens || (pos < static_cast<int>(seq_lens[b]));

                const float prev_log_value = (pos == 0)
                    ? load_as_float(initial_log_belief + (b * num_states + s))
                    : load_as_float(beliefs + ((b * seq_len + (pos - 1)) * num_states + s));
                const float prev_prob_value = fast_exp(prev_log_value);
                prev_prob[s] = prev_prob_value;
                __syncthreads();

                if (!active_step) {
                    carry_value += load_as_float(grad_beliefs + (base + s));
                    q_prob_value = prev_prob_value;
                    __syncthreads();
                    continue;
                }

                float mix_prob = 0.0f;
                for (int src = 0; src < num_states; ++src) {
                    mix_prob += prev_prob[src] * transition_matrix[src * num_states + s];
                }
                const float stay_prob = transition_stay_probs[s];
                const float pred_prob = fmaxf(
                    (stay_prob * prev_prob_value) + ((1.0f - stay_prob) * mix_prob),
                    1.0e-20f);
                const float pred_log = fast_log(pred_prob);
                const float transition_context_value = load_as_float(transition_context + (base + s));
                const float gq = load_as_float(grad_beliefs + (base + s)) + carry_value;
                const float gq_sum = block_reduce_sum_128(gq, scratch);
                const float ga = gq - (q_prob_value * gq_sum);
                const float prior_value = pred_log + transition_context_value;
                const float clamped_prior = apply_score_clamp(
                    prior_value,
                    score_clamp_min,
                    score_clamp_max);
                const float grad_prior = (transition_gate * ga) * score_clamp_grad(
                    prior_value,
                    score_clamp_min,
                    score_clamp_max);
                const float grad_pred_prob = grad_prior / pred_prob;

                grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
                grad_transition_context[base + s] = store_from_float<scalar_t>(grad_prior);
                grad_mix[s] = grad_pred_prob * (1.0f - stay_prob);
                grad_stay_shared[s] += grad_pred_prob * (prev_prob_value - mix_prob);
                gate_grad_accum += ga * clamped_prior;
                carry_value = grad_pred_prob * stay_prob;
                __syncthreads();

                float carry_from_mix = 0.0f;
                for (int dst = 0; dst < num_states; ++dst) {
                    carry_from_mix += grad_mix[dst] * transition_matrix[s * num_states + dst];
                }
                const float row_proj = prev_prob_value * carry_from_mix;
                const float denom = fmaxf(row_sums[s], 1.0e-20f);

                for (int r = 0; r < transition_rank; ++r) {
                    float grad_source_value = 0.0f;
                    const float source_value = transition_source_logits[s * transition_rank + r];
                    for (int dst = 0; dst < num_states; ++dst) {
                        if (!transition_mask[s * num_states + dst]) {
                            continue;
                        }
                        const float grad_transition_value = prev_prob_value * grad_mix[dst];
                        const float grad_raw = (grad_transition_value - row_proj) / denom;
                        grad_source_value += grad_raw * transition_dest_logits[r * num_states + dst];
                        atomicAdd(
                            grad_dest_batch + (static_cast<int64_t>(r) * num_states + dst),
                            source_value * grad_raw);
                    }
                    grad_source_batch[s * transition_rank + r] += grad_source_value;
                }

                carry_value = (carry_value + carry_from_mix) * prev_prob_value;
                q_prob_value = prev_prob_value;
                __syncthreads();
            }
        }

        grad_initial_log_belief[b * num_states + s] = store_from_float<scalar_t>(carry_value);
        grad_stay_batch[s] += grad_stay_shared[s];
        const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
        if (s == 0) {
            grad_transition_gate_per_batch[b] += gate_sum;
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_forward_sparse_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_blocks,
    const int32_t* __restrict__ block_row_ptr,
    const int32_t* __restrict__ block_col_idx,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const int64_t* __restrict__ seq_lens,
    int num_states,
    int block_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int tid = threadIdx.x;
    extern __shared__ float shared_mem[];
    float* prev_log = shared_mem;
    float* prev_prob = prev_log + num_states;
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    float* scratch = prev_prob + num_states;
    float* tile_stats = scratch + num_warps;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    const int state_tile_size = static_cast<int>(blockDim.x);
    __shared__ int current_batch;

    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        for (int s = tid; s < num_states; s += blockDim.x) {
            prev_log[s] = load_as_float(initial_log_belief + (b * num_states + s));
        }
        __syncthreads();
        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * num_states;
                const bool active = !has_seq_lens || (pos < static_cast<int>(seq_lens[b]));
                if (!active) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        beliefs[base + s] = store_from_float<scalar_t>(prev_log[s]);
                    }
                    __syncthreads();
                    continue;
                }
                for (int s = tid; s < num_states; s += blockDim.x) {
                    prev_prob[s] = fast_exp(prev_log[s]);
                }
                __syncthreads();

                if (tid == 0) {
                    tile_stats[0] = -INFINITY;
                    tile_stats[1] = 0.0f;
                }
                __syncthreads();

                for (int state_start = 0; state_start < num_states; state_start += state_tile_size) {
                    const int active_states = min(state_tile_size, num_states - state_start);
                    float filtered_max = -INFINITY;
                    float tile_exp_partial = 0.0f;
                    for (int pass = 0; pass < 2; ++pass) {
                        if (pass == 1) {
                            const float tile_max = block_reduce_max_128(filtered_max, scratch);
                            for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                                const int dst = state_start + dst_local;
                                const int dst_block = dst / block_size;
                                const int dst_offset = dst - (dst_block * block_size);
                                float mix_prob = 0.0f;
                                for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
                                    const int src_block = block_col_idx[nz];
                                    const int src_base = src_block * block_size;
                                    const int active_src = min(block_size, num_states - src_base);
                                    const float* block_ptr = transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
                                    for (int src_offset = 0; src_offset < active_src; ++src_offset) {
                                        mix_prob += prev_prob[src_base + src_offset] * block_ptr[src_offset * block_size + dst_offset];
                                    }
                                }
                                const float stay_prob = transition_stay_probs[dst];
                                const float pred_prob = fmaxf(
                                    stay_prob * prev_prob[dst] + (1.0f - stay_prob) * mix_prob,
                                    1.0e-20f);
                                const float pred_log = fast_log(pred_prob);
                                const float filtered_value = load_as_float(local_logits + base + dst)
                                    + transition_gate * (pred_log + load_as_float(transition_context + base + dst));
                                tile_exp_partial += fast_exp(filtered_value - tile_max);
                            }
                            const float tile_sum = block_reduce_sum_128(tile_exp_partial, scratch);
                            if (tid == 0) {
                                const float running_max = tile_stats[0];
                                const float running_sum = tile_stats[1];
                                if (!isfinite(running_max)) {
                                    tile_stats[0] = tile_max;
                                    tile_stats[1] = tile_sum;
                                } else {
                                    const float new_max = fmaxf(running_max, tile_max);
                                    tile_stats[1] = running_sum * fast_exp(running_max - new_max)
                                        + tile_sum * fast_exp(tile_max - new_max);
                                    tile_stats[0] = new_max;
                                }
                            }
                            __syncthreads();
                            break;
                        }
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int dst = state_start + dst_local;
                            const int dst_block = dst / block_size;
                            const int dst_offset = dst - (dst_block * block_size);
                            float mix_prob = 0.0f;
                            for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
                                const int src_block = block_col_idx[nz];
                                const int src_base = src_block * block_size;
                                const int active_src = min(block_size, num_states - src_base);
                                const float* block_ptr = transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
                                for (int src_offset = 0; src_offset < active_src; ++src_offset) {
                                    mix_prob += prev_prob[src_base + src_offset] * block_ptr[src_offset * block_size + dst_offset];
                                }
                            }
                            const float stay_prob = transition_stay_probs[dst];
                            const float pred_prob = fmaxf(
                                stay_prob * prev_prob[dst] + (1.0f - stay_prob) * mix_prob,
                                1.0e-20f);
                            const float pred_log = fast_log(pred_prob);
                            const float filtered_value = load_as_float(local_logits + base + dst)
                                + transition_gate * (pred_log + load_as_float(transition_context + base + dst));
                            filtered_max = fmaxf(filtered_max, filtered_value);
                        }
                        __syncthreads();
                    }
                }

                const float log_norm = tile_stats[0] + fast_log(fmaxf(tile_stats[1], 1.0e-20f));
                __syncthreads();

                for (int state_start = 0; state_start < num_states; state_start += state_tile_size) {
                    const int active_states = min(state_tile_size, num_states - state_start);
                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        const int dst = state_start + dst_local;
                        const int dst_block = dst / block_size;
                        const int dst_offset = dst - (dst_block * block_size);
                        float mix_prob = 0.0f;
                        for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
                            const int src_block = block_col_idx[nz];
                            const int src_base = src_block * block_size;
                            const int active_src = min(block_size, num_states - src_base);
                            const float* block_ptr = transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
                            for (int src_offset = 0; src_offset < active_src; ++src_offset) {
                                mix_prob += prev_prob[src_base + src_offset] * block_ptr[src_offset * block_size + dst_offset];
                            }
                        }
                        const float stay_prob = transition_stay_probs[dst];
                        const float pred_prob = fmaxf(
                            stay_prob * prev_prob[dst] + (1.0f - stay_prob) * mix_prob,
                            1.0e-20f);
                        const float pred_log = fast_log(pred_prob);
                        const float filtered_value = load_as_float(local_logits + base + dst)
                            + transition_gate * (pred_log + load_as_float(transition_context + base + dst));
                        const float next_log_value = filtered_value - log_norm;
                        prev_log[dst] = next_log_value;
                        beliefs[base + dst] = store_from_float<scalar_t>(next_log_value);
                    }
                    __syncthreads();
                }
            }
        }
        for (int s = tid; s < num_states; s += blockDim.x) {
            final_log_belief[b * num_states + s] = store_from_float<scalar_t>(prev_log[s]);
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_backward_sparse_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const float* __restrict__ transition_blocks,
    const int32_t* __restrict__ block_row_ptr,
    const int32_t* __restrict__ block_col_idx,
    const int32_t* __restrict__ block_dst_idx,
    const int32_t* __restrict__ src_row_ptr,
    const int32_t* __restrict__ src_nz_idx,
    const int32_t* __restrict__ grouped_src_row_ptr,
    const int32_t* __restrict__ grouped_src_block_idx,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const int64_t* __restrict__ seq_lens,
    int num_states,
    int block_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int grouped_src_group_count,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_blocks,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate,
    float* __restrict__ grad_transition_stay) {
    const int tid = threadIdx.x;
    extern __shared__ float shared_mem[];
    float* prev_prob = shared_mem;
    float* grad_mix = prev_prob + num_states;
    float* grad_stay_shared = grad_mix + num_states;
    float* carry_shared = grad_stay_shared + num_states;
    float* scratch = carry_shared + num_states;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        for (int s = tid; s < num_states; s += blockDim.x) {
            grad_stay_shared[s] = 0.0f;
            carry_shared[s] = load_as_float(grad_final_belief + (b * num_states + s));
        }
        __syncthreads();
        float gate_grad_accum = 0.0f;

        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * num_states;
                const bool active_step = !has_seq_lens || (pos < static_cast<int>(seq_lens[b]));
                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float prev_log_value = (pos == 0)
                        ? load_as_float(initial_log_belief + (b * num_states + s))
                        : load_as_float(beliefs + ((static_cast<int64_t>(b) * seq_len + (pos - 1)) * num_states + s));
                    prev_prob[s] = fast_exp(prev_log_value);
                }
                __syncthreads();

                if (!active_step) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        carry_shared[s] += load_as_float(grad_beliefs + (base + s));
                    }
                    __syncthreads();
                    continue;
                }

                float local_gq_sum = 0.0f;
                for (int s = tid; s < num_states; s += blockDim.x) {
                    local_gq_sum += load_as_float(grad_beliefs + (base + s)) + carry_shared[s];
                }
                const float gq_sum = block_reduce_sum_128(local_gq_sum, scratch);
                __syncthreads();

                float gate_grad_local = 0.0f;
                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float prev_prob_value = prev_prob[s];
                    float mix_prob = 0.0f;
                    const int dst_block = s / block_size;
                    const int dst_offset = s - (dst_block * block_size);
                    for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
                        const int src_block = block_col_idx[nz];
                        const int src_base = src_block * block_size;
                        const int active_src = min(block_size, num_states - src_base);
                        const float* block_ptr = transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
                        for (int src_offset = 0; src_offset < active_src; ++src_offset) {
                            mix_prob += prev_prob[src_base + src_offset] * block_ptr[src_offset * block_size + dst_offset];
                        }
                    }
                    const float stay_prob = transition_stay_probs[s];
                    const float pred_prob = fmaxf(
                        stay_prob * prev_prob_value + (1.0f - stay_prob) * mix_prob,
                        1.0e-20f);
                    const float pred_log = fast_log(pred_prob);
                    const float transition_context_value = load_as_float(transition_context + (base + s));
                    const float q_prob_value = fast_exp(load_as_float(beliefs + (base + s)));
                    const float gq = load_as_float(grad_beliefs + (base + s)) + carry_shared[s];
                    const float ga = gq - q_prob_value * gq_sum;
                    const float grad_pred_prob = (transition_gate * ga) / pred_prob;
                    grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
                    grad_transition_context[base + s] = store_from_float<scalar_t>(transition_gate * ga);
                    grad_mix[s] = grad_pred_prob * (1.0f - stay_prob);
                    grad_stay_shared[s] += grad_pred_prob * (prev_prob_value - mix_prob);
                    gate_grad_local += ga * (pred_log + transition_context_value);
                    carry_shared[s] = grad_pred_prob * stay_prob;
                }
                gate_grad_accum += gate_grad_local;
                __syncthreads();

                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float prev_prob_value = prev_prob[s];
                    const int src_block = s / block_size;
                    const int src_offset = s - (src_block * block_size);
                    float carry_from_mix = 0.0f;
                    int row_begin = src_row_ptr[src_block];
                    int row_end = src_row_ptr[src_block + 1];
                    if (grouped_src_group_count > 0) {
                        const int grouped_src = find_grouped_src_group_for_block(
                            grouped_src_block_idx,
                            grouped_src_group_count,
                            src_block);
                        if (grouped_src >= 0) {
                            row_begin = grouped_src_row_ptr[grouped_src];
                            row_end = grouped_src_row_ptr[grouped_src + 1];
                        } else {
                            row_begin = 0;
                            row_end = 0;
                        }
                    }
                    for (int entry = row_begin; entry < row_end; ++entry) {
                        const int nz = src_nz_idx[entry];
                        const int dst_block = block_dst_idx[nz];
                        const int dst_base = dst_block * block_size;
                        const int active_dst = min(block_size, num_states - dst_base);
                        const float* block_ptr = transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
                        float* grad_block_ptr = grad_transition_blocks + static_cast<int64_t>(nz) * block_size * block_size;
                        for (int dst_offset = 0; dst_offset < active_dst; ++dst_offset) {
                            const float grad_mix_value = grad_mix[dst_base + dst_offset];
                            carry_from_mix += grad_mix_value * block_ptr[src_offset * block_size + dst_offset];
                            atomicAdd(
                                grad_block_ptr + src_offset * block_size + dst_offset,
                                prev_prob_value * grad_mix_value);
                        }
                    }
                    carry_shared[s] = (carry_shared[s] + carry_from_mix) * prev_prob_value;
                }
                __syncthreads();
            }
        }

        for (int s = tid; s < num_states; s += blockDim.x) {
            grad_initial_log_belief[b * num_states + s] = store_from_float<scalar_t>(carry_shared[s]);
            atomicAdd(grad_transition_stay + s, grad_stay_shared[s]);
        }
        const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
        if (tid == 0) {
            atomicAdd(grad_transition_gate, gate_sum);
        }
        __syncthreads();
    }
}

template <typename scalar_t, bool InputsAreLogits = false>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_forward_sparse_factor_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_values,
    const float* __restrict__ transition_dest_values,
    const float* __restrict__ source_row_max,
    const float* __restrict__ source_row_inv_sum,
    const float* __restrict__ dest_row_max,
    const float* __restrict__ dest_row_inv_sum,
    const float* __restrict__ row_sums,
    const int32_t* __restrict__ block_row_ptr,
    const int32_t* __restrict__ block_col_idx,
    const int32_t* __restrict__ block_dst_idx,
    const int32_t* __restrict__ src_row_ptr,
    const int32_t* __restrict__ src_nz_idx,
    const float* __restrict__ block_mask,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const int64_t* __restrict__ seq_lens,
    int num_states,
    int padded_states,
    int transition_rank,
    int block_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int tid = threadIdx.x;
    extern __shared__ float shared_mem[];
    float* prev_log = shared_mem;
    float* prev_prob = prev_log + num_states;
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    float* scratch = prev_prob + num_states;
    float* tile_stats = scratch + num_warps;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    const int state_tile_size = static_cast<int>(blockDim.x);
    __shared__ int current_batch;

    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        for (int s = tid; s < num_states; s += blockDim.x) {
            prev_log[s] = load_as_float(initial_log_belief + (b * num_states + s));
        }
        __syncthreads();
        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * num_states;
                const bool active = !has_seq_lens || (pos < static_cast<int>(seq_lens[b]));
                if (!active) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        beliefs[base + s] = store_from_float<scalar_t>(prev_log[s]);
                    }
                    __syncthreads();
                    continue;
                }
                for (int s = tid; s < num_states; s += blockDim.x) {
                    prev_prob[s] = fast_exp(prev_log[s]);
                }
                __syncthreads();

                if (tid == 0) {
                    tile_stats[0] = -INFINITY;
                    tile_stats[1] = 0.0f;
                }
                __syncthreads();

                for (int state_start = 0; state_start < num_states; state_start += state_tile_size) {
                    const int active_states = min(state_tile_size, num_states - state_start);
                    float filtered_max = -INFINITY;
                    float tile_exp_partial = 0.0f;
                    for (int pass = 0; pass < 2; ++pass) {
                        if (pass == 1) {
                            const float tile_max = block_reduce_max_128(filtered_max, scratch);
                            for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                                const int dst = state_start + dst_local;
                                const int dst_block = dst / block_size;
                                const int dst_offset = dst - (dst_block * block_size);
                                float mix_prob = 0.0f;
                                for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
                                    const int src_block = block_col_idx[nz];
                                    const int src_base = src_block * block_size;
                                    const int active_src = min(block_size, num_states - src_base);
                                    const int dst_base = block_dst_idx[nz] * block_size;
                                    const float* mask_row_base =
                                        block_mask + static_cast<int64_t>(nz) * block_size * block_size;
                                    for (int src_offset = 0; src_offset < active_src; ++src_offset) {
                                        const int src_state = src_base + src_offset;
                                        const int dst_state = dst_base + dst_offset;
                                        if (src_state >= padded_states || dst_state >= padded_states) {
                                            continue;
                                        }
                                        const float mask_value = mask_row_base[src_offset * block_size + dst_offset];
                                        if (mask_value == 0.0f) {
                                            continue;
                                        }
                                        const float raw_value = sparse_factor_prob_value<InputsAreLogits>(
                                            transition_source_values,
                                            transition_dest_values,
                                            source_row_max,
                                            source_row_inv_sum,
                                            dest_row_max,
                                            dest_row_inv_sum,
                                            src_state,
                                            dst_state,
                                            transition_rank,
                                            padded_states,
                                            num_states);
                                        const float denom = sparse_transition_row_sum_value<InputsAreLogits>(
                                            transition_source_values,
                                            transition_dest_values,
                                            source_row_max,
                                            source_row_inv_sum,
                                            dest_row_max,
                                            dest_row_inv_sum,
                                            row_sums,
                                            src_row_ptr,
                                            src_nz_idx,
                                            block_dst_idx,
                                            block_mask,
                                            src_state,
                                            transition_rank,
                                            padded_states,
                                            num_states,
                                            block_size);
                                        mix_prob += prev_prob[src_state] * ((raw_value * mask_value) / denom);
                                    }
                                }
                                const float stay_prob = transition_stay_probs[dst];
                                const float pred_prob = fmaxf(
                                    stay_prob * prev_prob[dst] + (1.0f - stay_prob) * mix_prob,
                                    1.0e-20f);
                                const float pred_log = fast_log(pred_prob);
                                const float filtered_value = load_as_float(local_logits + base + dst)
                                    + transition_gate * (pred_log + load_as_float(transition_context + base + dst));
                                tile_exp_partial += fast_exp(filtered_value - tile_max);
                            }
                            const float tile_sum = block_reduce_sum_128(tile_exp_partial, scratch);
                            if (tid == 0) {
                                const float running_max = tile_stats[0];
                                const float running_sum = tile_stats[1];
                                if (!isfinite(running_max)) {
                                    tile_stats[0] = tile_max;
                                    tile_stats[1] = tile_sum;
                                } else {
                                    const float new_max = fmaxf(running_max, tile_max);
                                    tile_stats[1] = running_sum * fast_exp(running_max - new_max)
                                        + tile_sum * fast_exp(tile_max - new_max);
                                    tile_stats[0] = new_max;
                                }
                            }
                            __syncthreads();
                            break;
                        }
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int dst = state_start + dst_local;
                            const int dst_block = dst / block_size;
                            const int dst_offset = dst - (dst_block * block_size);
                            float mix_prob = 0.0f;
                            for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
                                const int src_block = block_col_idx[nz];
                                const int src_base = src_block * block_size;
                                const int active_src = min(block_size, num_states - src_base);
                                const int dst_base = block_dst_idx[nz] * block_size;
                                const float* mask_row_base =
                                    block_mask + static_cast<int64_t>(nz) * block_size * block_size;
                                for (int src_offset = 0; src_offset < active_src; ++src_offset) {
                                    const int src_state = src_base + src_offset;
                                    const int dst_state = dst_base + dst_offset;
                                    if (src_state >= padded_states || dst_state >= padded_states) {
                                        continue;
                                    }
                                    const float mask_value = mask_row_base[src_offset * block_size + dst_offset];
                                    if (mask_value == 0.0f) {
                                        continue;
                                    }
                                    const float raw_value = sparse_factor_prob_value<InputsAreLogits>(
                                        transition_source_values,
                                        transition_dest_values,
                                        source_row_max,
                                        source_row_inv_sum,
                                        dest_row_max,
                                        dest_row_inv_sum,
                                        src_state,
                                        dst_state,
                                        transition_rank,
                                        padded_states,
                                        num_states);
                                    const float denom = sparse_transition_row_sum_value<InputsAreLogits>(
                                        transition_source_values,
                                        transition_dest_values,
                                        source_row_max,
                                        source_row_inv_sum,
                                        dest_row_max,
                                        dest_row_inv_sum,
                                        row_sums,
                                        src_row_ptr,
                                        src_nz_idx,
                                        block_dst_idx,
                                        block_mask,
                                        src_state,
                                        transition_rank,
                                        padded_states,
                                        num_states,
                                        block_size);
                                    mix_prob += prev_prob[src_state] * ((raw_value * mask_value) / denom);
                                }
                            }
                            const float stay_prob = transition_stay_probs[dst];
                            const float pred_prob = fmaxf(
                                stay_prob * prev_prob[dst] + (1.0f - stay_prob) * mix_prob,
                                1.0e-20f);
                            const float pred_log = fast_log(pred_prob);
                            const float filtered_value = load_as_float(local_logits + base + dst)
                                + transition_gate * (pred_log + load_as_float(transition_context + base + dst));
                            filtered_max = fmaxf(filtered_max, filtered_value);
                        }
                        __syncthreads();
                    }
                }

                const float log_norm = tile_stats[0] + fast_log(fmaxf(tile_stats[1], 1.0e-20f));
                __syncthreads();

                for (int state_start = 0; state_start < num_states; state_start += state_tile_size) {
                    const int active_states = min(state_tile_size, num_states - state_start);
                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        const int dst = state_start + dst_local;
                        const int dst_block = dst / block_size;
                        const int dst_offset = dst - (dst_block * block_size);
                        float mix_prob = 0.0f;
                        for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
                            const int src_block = block_col_idx[nz];
                            const int src_base = src_block * block_size;
                            const int active_src = min(block_size, num_states - src_base);
                            const int dst_base = block_dst_idx[nz] * block_size;
                            const float* mask_row_base =
                                block_mask + static_cast<int64_t>(nz) * block_size * block_size;
                            for (int src_offset = 0; src_offset < active_src; ++src_offset) {
                                const int src_state = src_base + src_offset;
                                const int dst_state = dst_base + dst_offset;
                                if (src_state >= padded_states || dst_state >= padded_states) {
                                    continue;
                                }
                                const float mask_value = mask_row_base[src_offset * block_size + dst_offset];
                                if (mask_value == 0.0f) {
                                    continue;
                                }
                                const float raw_value = sparse_factor_prob_value<InputsAreLogits>(
                                    transition_source_values,
                                    transition_dest_values,
                                    source_row_max,
                                    source_row_inv_sum,
                                    dest_row_max,
                                    dest_row_inv_sum,
                                    src_state,
                                    dst_state,
                                    transition_rank,
                                    padded_states,
                                    num_states);
                                const float denom = sparse_transition_row_sum_value<InputsAreLogits>(
                                    transition_source_values,
                                    transition_dest_values,
                                    source_row_max,
                                    source_row_inv_sum,
                                    dest_row_max,
                                    dest_row_inv_sum,
                                    row_sums,
                                    src_row_ptr,
                                    src_nz_idx,
                                    block_dst_idx,
                                    block_mask,
                                    src_state,
                                    transition_rank,
                                    padded_states,
                                    num_states,
                                    block_size);
                                mix_prob += prev_prob[src_state] * ((raw_value * mask_value) / denom);
                            }
                        }
                        const float stay_prob = transition_stay_probs[dst];
                        const float pred_prob = fmaxf(
                            stay_prob * prev_prob[dst] + (1.0f - stay_prob) * mix_prob,
                            1.0e-20f);
                        const float pred_log = fast_log(pred_prob);
                        const float filtered_value = load_as_float(local_logits + base + dst)
                            + transition_gate * (pred_log + load_as_float(transition_context + base + dst));
                        const float next_log_value = filtered_value - log_norm;
                        prev_log[dst] = next_log_value;
                        beliefs[base + dst] = store_from_float<scalar_t>(next_log_value);
                    }
                    __syncthreads();
                }
            }
        }
        for (int s = tid; s < num_states; s += blockDim.x) {
            final_log_belief[b * num_states + s] = store_from_float<scalar_t>(prev_log[s]);
        }
        __syncthreads();
    }
}

template <typename scalar_t, bool InputsAreLogits = false>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_backward_sparse_factor_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const float* __restrict__ transition_source_values,
    const float* __restrict__ transition_dest_values,
    const float* __restrict__ source_row_max,
    const float* __restrict__ source_row_inv_sum,
    const float* __restrict__ dest_row_max,
    const float* __restrict__ dest_row_inv_sum,
    const float* __restrict__ row_sums,
    const int32_t* __restrict__ block_row_ptr,
    const int32_t* __restrict__ block_col_idx,
    const int32_t* __restrict__ block_dst_idx,
    const int32_t* __restrict__ src_row_ptr,
    const int32_t* __restrict__ src_nz_idx,
    const int32_t* __restrict__ grouped_src_row_ptr,
    const int32_t* __restrict__ grouped_src_block_idx,
    const float* __restrict__ block_mask,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const int64_t* __restrict__ seq_lens,
    int num_states,
    int padded_states,
    int transition_rank,
    int block_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int grouped_src_group_count,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_probs,
    float* __restrict__ grad_transition_dest_probs,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate,
    float* __restrict__ grad_transition_stay) {
    const int tid = threadIdx.x;
    extern __shared__ float shared_mem[];
    float* prev_prob = shared_mem;
    float* grad_mix = prev_prob + num_states;
    float* grad_stay_shared = grad_mix + num_states;
    float* carry_shared = grad_stay_shared + num_states;
    float* row_proj_shared = carry_shared + num_states;
    float* scratch = row_proj_shared + num_states;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    while (true) {
        if (threadIdx.x == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        for (int s = tid; s < num_states; s += blockDim.x) {
            grad_stay_shared[s] = 0.0f;
            carry_shared[s] = load_as_float(grad_final_belief + (b * num_states + s));
        }
        __syncthreads();
        float gate_grad_accum = 0.0f;

        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * num_states;
                const bool active_step = !has_seq_lens || (pos < static_cast<int>(seq_lens[b]));
                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float prev_log_value = (pos == 0)
                        ? load_as_float(initial_log_belief + (b * num_states + s))
                        : load_as_float(beliefs + ((static_cast<int64_t>(b) * seq_len + (pos - 1)) * num_states + s));
                    prev_prob[s] = fast_exp(prev_log_value);
                }
                __syncthreads();

                if (!active_step) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        carry_shared[s] += load_as_float(grad_beliefs + (base + s));
                    }
                    __syncthreads();
                    continue;
                }

                float local_gq_sum = 0.0f;
                for (int s = tid; s < num_states; s += blockDim.x) {
                    local_gq_sum += load_as_float(grad_beliefs + (base + s)) + carry_shared[s];
                }
                const float gq_sum = block_reduce_sum_128(local_gq_sum, scratch);
                __syncthreads();

                float gate_grad_local = 0.0f;
                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float prev_prob_value = prev_prob[s];
                    float mix_prob = 0.0f;
                    const int dst_block = s / block_size;
                    const int dst_offset = s - (dst_block * block_size);
                    for (int nz = block_row_ptr[dst_block]; nz < block_row_ptr[dst_block + 1]; ++nz) {
                        const int src_block = block_col_idx[nz];
                        const int src_base = src_block * block_size;
                        const int active_src = min(block_size, num_states - src_base);
                        const int dst_base = block_dst_idx[nz] * block_size;
                        const int dst_state = dst_base + dst_offset;
                        if (dst_state >= padded_states) {
                            continue;
                        }
                        const float* mask_row_base =
                            block_mask + static_cast<int64_t>(nz) * block_size * block_size;
                        for (int src_offset = 0; src_offset < active_src; ++src_offset) {
                            const int src_state = src_base + src_offset;
                            if (src_state >= padded_states) {
                                continue;
                            }
                            const float mask_value = mask_row_base[src_offset * block_size + dst_offset];
                            if (mask_value == 0.0f) {
                                continue;
                            }
                            const float raw_value = sparse_factor_prob_value<InputsAreLogits>(
                                transition_source_values,
                                transition_dest_values,
                                source_row_max,
                                source_row_inv_sum,
                                dest_row_max,
                                dest_row_inv_sum,
                                src_state,
                                dst_state,
                                transition_rank,
                                padded_states,
                                num_states);
                            mix_prob += prev_prob[src_state]
                                * ((raw_value * mask_value) / fmaxf(row_sums[src_state], 1.0e-20f));
                        }
                    }
                    const float stay_prob = transition_stay_probs[s];
                    const float pred_prob = fmaxf(
                        stay_prob * prev_prob_value + (1.0f - stay_prob) * mix_prob,
                        1.0e-20f);
                    const float pred_log = fast_log(pred_prob);
                    const float transition_context_value = load_as_float(transition_context + (base + s));
                    const float q_prob_value = fast_exp(load_as_float(beliefs + (base + s)));
                    const float gq = load_as_float(grad_beliefs + (base + s)) + carry_shared[s];
                    const float ga = gq - q_prob_value * gq_sum;
                    const float grad_pred_prob = (transition_gate * ga) / pred_prob;
                    grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
                    grad_transition_context[base + s] = store_from_float<scalar_t>(transition_gate * ga);
                    grad_mix[s] = grad_pred_prob * (1.0f - stay_prob);
                    grad_stay_shared[s] += grad_pred_prob * (prev_prob_value - mix_prob);
                    gate_grad_local += ga * (pred_log + transition_context_value);
                    carry_shared[s] = grad_pred_prob * stay_prob;
                }
                gate_grad_accum += gate_grad_local;
                __syncthreads();

                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float prev_prob_value = prev_prob[s];
                    const int src_block = s / block_size;
                    const int src_offset = s - (src_block * block_size);
                    const float denom = sparse_transition_row_sum_value<InputsAreLogits>(
                        transition_source_values,
                        transition_dest_values,
                        source_row_max,
                        source_row_inv_sum,
                        dest_row_max,
                        dest_row_inv_sum,
                        row_sums,
                        src_row_ptr,
                        src_nz_idx,
                        block_dst_idx,
                        block_mask,
                        s,
                        transition_rank,
                        padded_states,
                        num_states,
                        block_size);
                    float carry_from_mix = 0.0f;
                    int row_begin = src_row_ptr[src_block];
                    int row_end = src_row_ptr[src_block + 1];
                    if (grouped_src_group_count > 0) {
                        const int grouped_src = find_grouped_src_group_for_block(
                            grouped_src_block_idx,
                            grouped_src_group_count,
                            src_block);
                        if (grouped_src >= 0) {
                            row_begin = grouped_src_row_ptr[grouped_src];
                            row_end = grouped_src_row_ptr[grouped_src + 1];
                        } else {
                            row_begin = 0;
                            row_end = 0;
                        }
                    }
                    for (int entry = row_begin; entry < row_end; ++entry) {
                        const int nz = src_nz_idx[entry];
                        const int dst_block = block_dst_idx[nz];
                        const int dst_base = dst_block * block_size;
                        const int active_dst = min(block_size, num_states - dst_base);
                        const float* mask_row =
                            block_mask + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
                        for (int dst_offset = 0; dst_offset < active_dst; ++dst_offset) {
                            const int dst = dst_base + dst_offset;
                            if (dst >= padded_states) {
                                continue;
                            }
                            const float mask_value = mask_row[dst_offset];
                            if (mask_value == 0.0f) {
                                continue;
                            }
                            const float raw_value = sparse_factor_prob_value<InputsAreLogits>(
                                transition_source_values,
                                transition_dest_values,
                                source_row_max,
                                source_row_inv_sum,
                                dest_row_max,
                                dest_row_inv_sum,
                                s,
                                dst,
                                transition_rank,
                                padded_states,
                                num_states);
                            carry_from_mix += grad_mix[dst] * ((raw_value * mask_value) / denom);
                        }
                    }
                    const float row_proj = prev_prob_value * carry_from_mix;
                    row_proj_shared[s] = row_proj;
                    for (int entry = row_begin; entry < row_end; ++entry) {
                        const int nz = src_nz_idx[entry];
                        const int dst_block = block_dst_idx[nz];
                        const int dst_base = dst_block * block_size;
                        const int active_dst = min(block_size, num_states - dst_base);
                        const float* mask_row =
                            block_mask + (static_cast<int64_t>(nz) * block_size + src_offset) * block_size;
                        for (int dst_offset = 0; dst_offset < active_dst; ++dst_offset) {
                            const int dst = dst_base + dst_offset;
                            if (dst >= padded_states) {
                                continue;
                            }
                            const float mask_value = mask_row[dst_offset];
                            if (mask_value == 0.0f) {
                                continue;
                            }
                            const float grad_raw =
                                ((prev_prob_value * grad_mix[dst]) - row_proj) / denom * mask_value;
                            const int64_t source_row_base = static_cast<int64_t>(s) * transition_rank;
                            for (int r = 0; r < transition_rank; ++r) {
                                float source_value;
                                float dest_value;
                                if constexpr (InputsAreLogits) {
                                    source_value = softmax_prob_from_stats_value(
                                        transition_source_values,
                                        transition_rank,
                                        s,
                                        r,
                                        source_row_max,
                                        source_row_inv_sum);
                                    dest_value = softmax_prob_from_stats_value(
                                        transition_dest_values,
                                        num_states,
                                        r,
                                        dst,
                                        dest_row_max,
                                        dest_row_inv_sum);
                                } else {
                                    source_value = transition_source_values[source_row_base + r];
                                    dest_value =
                                        transition_dest_values[static_cast<int64_t>(r) * padded_states + dst];
                                }
                                atomicAdd(
                                    grad_transition_source_probs + source_row_base + r,
                                    grad_raw * dest_value);
                                atomicAdd(
                                    grad_transition_dest_probs
                                        + static_cast<int64_t>(r) * padded_states + dst,
                                    source_value * grad_raw);
                            }
                        }
                    }
                    carry_shared[s] = (carry_shared[s] + carry_from_mix) * prev_prob_value;
                }
                __syncthreads();
            }
        }

        for (int s = tid; s < num_states; s += blockDim.x) {
            grad_initial_log_belief[b * num_states + s] = store_from_float<scalar_t>(carry_shared[s]);
            atomicAdd(grad_transition_stay + s, grad_stay_shared[s]);
        }
        const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
        if (tid == 0) {
            atomicAdd(grad_transition_gate, gate_sum);
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_forward_masked_tiled_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_logits,
    const float* __restrict__ transition_dest_logits,
    const scalar_t* __restrict__ transition_context,
    const float* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const bool* __restrict__ transition_mask,
    const float* __restrict__ row_sums,
    const int64_t* __restrict__ seq_lens,
    float score_clamp_min,
    float score_clamp_max,
    float score_threshold,
    int score_topk,
    int num_states,
    int transition_rank,
    int tile_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    float* __restrict__ masked_transition_tile_cache,
    float* __restrict__ filtered_value_cache,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int tid = threadIdx.x;
    __shared__ int current_batch;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    const int chunk_end = min(chunk_start + chunk_len, seq_len);
    extern __shared__ float shared_mem[];
    float* prev_log = shared_mem;
    float* prev_prob = prev_log + num_states;
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    float* scratch = prev_prob + num_states;
    float* tile_stats = scratch + num_warps;
    float* topk_values_shared = tile_stats + 2;
    int* topk_indices_shared = reinterpret_cast<int*>(topk_values_shared + kMaxNativeScoreTopK);
    const bool use_global_score_cache = native_score_filtering_enabled(score_threshold, score_topk, num_states);

    while (true) {
        if (tid == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float* tile_cache = masked_transition_tile_cache
            + static_cast<int64_t>(blockIdx.x) * num_states * tile_size;
        float* filtered_cache = filtered_value_cache
            + static_cast<int64_t>(blockIdx.x) * num_states;
        float* score_cache = use_global_score_cache ? filtered_cache : prev_log;
        for (int s = tid; s < num_states; s += blockDim.x) {
            prev_log[s] = initial_log_belief[b * num_states + s];
        }
        __syncthreads();

        for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, chunk_end - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * num_states;
                const bool active = !has_seq_lens || pos < static_cast<int>(seq_lens[b]);

                if (!active) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        beliefs[base + s] = store_from_float<scalar_t>(prev_log[s]);
                    }
                    __syncthreads();
                    continue;
                }

                for (int s = tid; s < num_states; s += blockDim.x) {
                    prev_prob[s] = fast_exp(prev_log[s]);
                }
                __syncthreads();

                if (tid == 0) {
                    tile_stats[0] = -INFINITY;
                    tile_stats[1] = 0.0f;
                }
                __syncthreads();

                for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                    const int active_states = min(tile_size, num_states - state_start);
                    for (int linear = tid; linear < num_states * active_states; linear += blockDim.x) {
                        const int src = linear / active_states;
                        const int dst_local = linear - (src * active_states);
                        const int dst = state_start + dst_local;
                        tile_cache[src * tile_size + dst_local] = masked_transition_raw_value(
                            transition_source_logits,
                            transition_dest_logits,
                            transition_mask,
                            num_states,
                            transition_rank,
                            src,
                            dst) / row_sums[src];
                    }
                    __syncthreads();

                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        const int dst = state_start + dst_local;
                        float mix_prob = 0.0f;
                        for (int src = 0; src < num_states; ++src) {
                            mix_prob += prev_prob[src] * tile_cache[src * tile_size + dst_local];
                        }
                        const float stay_prob = transition_stay_probs[dst];
                        const float pred_prob = fmaxf(
                            stay_prob * prev_prob[dst] + (1.0f - stay_prob) * mix_prob,
                            1.0e-20f);
                        const float pred_log = fast_log(pred_prob);
                        const float prior_value = pred_log + load_as_float(transition_context + base + dst);
                        score_cache[dst] = load_as_float(local_logits + base + dst)
                            + transition_gate * apply_score_clamp(
                                prior_value,
                                score_clamp_min,
                                score_clamp_max);
                    }
                    __syncthreads();

                    if (!use_global_score_cache) {
                        float filtered_max = -INFINITY;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int dst = state_start + dst_local;
                            filtered_max = fmaxf(filtered_max, score_cache[dst]);
                        }
                        __syncthreads();
                        const float tile_max = block_reduce_max_128(filtered_max, scratch);
                        float tile_exp_partial = 0.0f;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int dst = state_start + dst_local;
                            const float filtered_value = score_cache[dst];
                            tile_exp_partial += isfinite(filtered_value)
                                ? fast_exp(filtered_value - tile_max)
                                : 0.0f;
                        }
                        const float tile_sum = block_reduce_sum_128(tile_exp_partial, scratch);
                        if (tid == 0) {
                            update_online_logsumexp_stats(tile_max, tile_sum, tile_stats);
                        }
                        __syncthreads();
                    }
                }

                if (use_global_score_cache) {
                    apply_native_score_filtering_block(
                        filtered_cache,
                        num_states,
                        score_threshold,
                        score_topk,
                        scratch,
                        topk_values_shared,
                        topk_indices_shared);
                    if (tid == 0) {
                        tile_stats[0] = -INFINITY;
                        tile_stats[1] = 0.0f;
                    }
                    __syncthreads();
                    for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                        const int active_states = min(tile_size, num_states - state_start);
                        float filtered_max = -INFINITY;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int dst = state_start + dst_local;
                            filtered_max = fmaxf(filtered_max, filtered_cache[dst]);
                        }
                        __syncthreads();
                        const float tile_max = block_reduce_max_128(filtered_max, scratch);
                        float tile_exp_partial = 0.0f;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int dst = state_start + dst_local;
                            const float filtered_value = filtered_cache[dst];
                            tile_exp_partial += isfinite(filtered_value)
                                ? fast_exp(filtered_value - tile_max)
                                : 0.0f;
                        }
                        const float tile_sum = block_reduce_sum_128(tile_exp_partial, scratch);
                        if (tid == 0) {
                            update_online_logsumexp_stats(tile_max, tile_sum, tile_stats);
                        }
                        __syncthreads();
                    }
                }

                const float log_norm = tile_stats[0] + fast_log(fmaxf(tile_stats[1], 1.0e-20f));
                __syncthreads();

                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float next_log = score_cache[s] - log_norm;
                    prev_log[s] = next_log;
                    beliefs[base + s] = store_from_float<scalar_t>(next_log);
                }
                __syncthreads();
            }
        }

        for (int s = tid; s < num_states; s += blockDim.x) {
            final_log_belief[b * num_states + s] = store_from_float<scalar_t>(prev_log[s]);
        }
        __syncthreads();
    }
}

template <typename scalar_t, bool Sm90Path = false>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_forward_tiled_chunk_kernel(
    const scalar_t* __restrict__ local_logits,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const float* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const int64_t* __restrict__ seq_lens,
    float score_clamp_min,
    float score_clamp_max,
    float score_threshold,
    int score_topk,
    int num_states,
    int transition_rank,
    int tile_size,
    int split_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    float* __restrict__ filtered_value_cache,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int tid = threadIdx.x;
    __shared__ int current_batch;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    const int block_slot = static_cast<int>(blockIdx.x);
    const int staged_rank_chunk = max(1, min(split_size, Sm90Path ? kSm90AsyncTileRankChunk : kAsyncTileRankChunk));
    const int pipeline_stages = async_tile_pipeline_stages<Sm90Path>();
    extern __shared__ float shared_mem[];
    float* prev_log = shared_mem;
    float* prev_prob = prev_log + num_states;
    float* latent = prev_prob + num_states;
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    float* scratch = latent + staged_rank_chunk;
    float* tile_stats = scratch + num_warps;
    float* dest_tile_shared = tile_stats + 2;
    float* source_tile_shared = dest_tile_shared
        + static_cast<int64_t>(pipeline_stages) * staged_rank_chunk * tile_size;
    char* tensor_core_workspace_raw = reinterpret_cast<char*>(
        source_tile_shared + static_cast<int64_t>(pipeline_stages) * num_states * staged_rank_chunk);
    std::uintptr_t tensor_core_workspace_addr = reinterpret_cast<std::uintptr_t>(tensor_core_workspace_raw);
    tensor_core_workspace_addr = (tensor_core_workspace_addr + 31u) & ~std::uintptr_t(31u);
    char* tensor_core_workspace = reinterpret_cast<char*>(tensor_core_workspace_addr);
    using tensor_core_input_t = tensor_core_input_type_t<scalar_t>;
    tensor_core_input_t* tensor_core_lhs = reinterpret_cast<tensor_core_input_t*>(tensor_core_workspace);
    tensor_core_input_t* tensor_core_rhs = tensor_core_lhs + (kTensorCoreTile * kTensorCoreTile);
    float* tensor_core_accum = reinterpret_cast<float*>(tensor_core_rhs + (kTensorCoreTile * kTensorCoreTile));
    auto tile_pipe = cuda::make_pipeline();
    const bool use_tensor_core_math =
        tensor_core_math_enabled_for_scalar<scalar_t>()
        && (transition_rank >= kTensorCoreTile)
        && (tile_size >= kTensorCoreTile)
        && (split_size >= kTensorCoreTile);
    const bool use_global_score_cache = native_score_filtering_enabled(score_threshold, score_topk, num_states);

    while (true) {
        if (tid == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float* filtered_cache = filtered_value_cache
            + static_cast<int64_t>(block_slot) * num_states;
        float* score_cache = use_global_score_cache ? filtered_cache : prev_log;
        for (int s = tid; s < num_states; s += blockDim.x) {
            prev_log[s] = load_as_float(initial_log_belief + (b * num_states + s));
        }
        __syncthreads();

        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int base = (b * seq_len + pos) * num_states;
                const bool active = !has_seq_lens || pos < static_cast<int>(seq_lens[b]);

                if (!active) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        const float carry_log = prev_log[s];
                        beliefs[base + s] = store_from_float<scalar_t>(carry_log);
                    }
                    __syncthreads();
                    continue;
                }

                for (int s = tid; s < num_states; s += blockDim.x) {
                    prev_prob[s] = fast_exp(prev_log[s]);
                }
                __syncthreads();

                if (tid == 0) {
                    tile_stats[0] = -INFINITY;
                    tile_stats[1] = 0.0f;
                }
                __syncthreads();

                for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                    const int active_states = min(tile_size, num_states - state_start);
                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        score_cache[state_start + dst_local] = 0.0f;
                    }
                    __syncthreads();
                    if (transition_rank > 0) {
                        const int total_rank_tiles = (transition_rank + staged_rank_chunk - 1) / staged_rank_chunk;
                        int next_tile_to_launch = 0;
                        const int initial_prefetch = min(pipeline_stages, total_rank_tiles);
                        for (; next_tile_to_launch < initial_prefetch; ++next_tile_to_launch) {
                            const int rank_start = next_tile_to_launch * staged_rank_chunk;
                            const int active_rank = min(staged_rank_chunk, transition_rank - rank_start);
                            const int stage_slot = next_tile_to_launch % pipeline_stages;
                            enqueue_float_matrix_pair_slice_async(
                                tile_pipe,
                                dest_tile_shared + static_cast<int64_t>(stage_slot) * staged_rank_chunk * tile_size,
                                transition_dest_probs + static_cast<int64_t>(rank_start) * num_states + state_start,
                                num_states,
                                active_rank,
                                active_states,
                                static_cast<int>(tile_size),
                                source_tile_shared + static_cast<int64_t>(stage_slot) * num_states * staged_rank_chunk,
                                transition_source_probs + rank_start,
                                transition_rank,
                                num_states,
                                active_rank,
                                staged_rank_chunk);
                        }
                        if (initial_prefetch > 0) {
                            wait_for_async_tile(tile_pipe);
                            __syncthreads();
                        }
                        for (int rank_tile_idx = 0; rank_tile_idx < total_rank_tiles; ++rank_tile_idx) {
                            const int current_stage = rank_tile_idx % pipeline_stages;
                            const int current_rank_start = rank_tile_idx * staged_rank_chunk;
                            const int current_active_rank = min(staged_rank_chunk, transition_rank - current_rank_start);
                            if (next_tile_to_launch < total_rank_tiles) {
                                const int next_rank_start = next_tile_to_launch * staged_rank_chunk;
                                const int next_active_rank = min(staged_rank_chunk, transition_rank - next_rank_start);
                                const int next_stage = next_tile_to_launch % pipeline_stages;
                                enqueue_float_matrix_pair_slice_async(
                                    tile_pipe,
                                    dest_tile_shared + static_cast<int64_t>(next_stage) * staged_rank_chunk * tile_size,
                                    transition_dest_probs + static_cast<int64_t>(next_rank_start) * num_states + state_start,
                                    num_states,
                                    next_active_rank,
                                    active_states,
                                    static_cast<int>(tile_size),
                                    source_tile_shared + static_cast<int64_t>(next_stage) * num_states * staged_rank_chunk,
                                    transition_source_probs + next_rank_start,
                                    transition_rank,
                                    num_states,
                                    next_active_rank,
                                    staged_rank_chunk);
                                ++next_tile_to_launch;
                            }
                            const float* current_source_tile = source_tile_shared
                                + static_cast<int64_t>(current_stage) * num_states * staged_rank_chunk;
                            if (use_tensor_core_math && current_active_rank == kTensorCoreTile) {
                                for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                    latent[r] = 0.0f;
                                }
                                __syncthreads();
                                for (int src_start = 0; src_start < num_states; src_start += kTensorCoreTile) {
                                    const int active_src = min(kTensorCoreTile, num_states - src_start);
#if __CUDA_ARCH__ >= 700
                                    if (active_src == kTensorCoreTile) {
                                        wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                            prev_prob + src_start,
                                            current_source_tile + static_cast<int64_t>(src_start) * staged_rank_chunk,
                                            staged_rank_chunk,
                                            tensor_core_lhs,
                                            tensor_core_rhs,
                                            tensor_core_accum,
                                            latent);
                                    } else
#endif
                                    {
                                        for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                            latent[r] += lowp_dot_rhs_strided(
                                                prev_prob + src_start,
                                                current_source_tile + static_cast<int64_t>(src_start) * staged_rank_chunk + r,
                                                staged_rank_chunk,
                                                active_src);
                                        }
                                        __syncthreads();
                                    }
                                }
                            } else {
                                for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                    latent[r] = lowp_dot_rhs_strided(
                                        prev_prob,
                                        current_source_tile + r,
                                        staged_rank_chunk,
                                        num_states);
                                }
                            }
                            __syncthreads();
                            const float* current_dest_tile = dest_tile_shared
                                + static_cast<int64_t>(current_stage) * staged_rank_chunk * tile_size;
                            if (use_tensor_core_math && current_active_rank == kTensorCoreTile) {
                                for (int dst_subtile = 0; dst_subtile < active_states; dst_subtile += kTensorCoreTile) {
                                    const int active_dst = min(kTensorCoreTile, active_states - dst_subtile);
#if __CUDA_ARCH__ >= 700
                                    if (active_dst == kTensorCoreTile) {
                                        wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                            latent,
                                            current_dest_tile + dst_subtile,
                                            tile_size,
                                            tensor_core_lhs,
                                            tensor_core_rhs,
                                            tensor_core_accum,
                                            score_cache + state_start + dst_subtile);
                                    } else
#endif
                                    {
                                        for (int dst_local = dst_subtile + tid; dst_local < dst_subtile + active_dst; dst_local += blockDim.x) {
                                            score_cache[state_start + dst_local] += lowp_dot_rhs_strided(
                                                latent,
                                                current_dest_tile + dst_local,
                                                tile_size,
                                                current_active_rank);
                                        }
                                        __syncthreads();
                                    }
                                }
                            } else {
                                for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                                    score_cache[state_start + dst_local] += lowp_dot_rhs_strided(
                                        latent,
                                        current_dest_tile + dst_local,
                                        tile_size,
                                        current_active_rank);
                                }
                            }
                            if (rank_tile_idx + 1 < total_rank_tiles) {
                                wait_for_async_tile(tile_pipe);
                            }
                            __syncthreads();
                        }
                    }
                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        const int s = state_start + dst_local;
                        const float stay_prob = transition_stay_probs[s];
                        const float mix_prob = score_cache[s];
                        const float pred_prob = fmaxf(
                            stay_prob * prev_prob[s] + (1.0f - stay_prob) * mix_prob,
                            1.0e-20f);
                        const float pred_log = fast_log(pred_prob);
                        const float prior_value = pred_log + load_as_float(transition_context + base + s);
                        score_cache[s] = load_as_float(local_logits + base + s)
                            + transition_gate * apply_score_clamp(
                                prior_value,
                                score_clamp_min,
                                score_clamp_max);
                    }
                    __syncthreads();

                    if (!use_global_score_cache) {
                        float filtered_max = -INFINITY;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int s = state_start + dst_local;
                            filtered_max = fmaxf(filtered_max, score_cache[s]);
                        }
                        __syncthreads();
                        const float tile_max = block_reduce_max_128(filtered_max, scratch);
                        float tile_exp_partial = 0.0f;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int s = state_start + dst_local;
                            const float filtered_value = score_cache[s];
                            tile_exp_partial += isfinite(filtered_value)
                                ? fast_exp(filtered_value - tile_max)
                                : 0.0f;
                        }
                        const float tile_sum = block_reduce_sum_128(tile_exp_partial, scratch);
                        if (tid == 0) {
                            update_online_logsumexp_stats(tile_max, tile_sum, tile_stats);
                        }
                        __syncthreads();
                    }
                }

                if (use_global_score_cache) {
                    apply_native_score_filtering_block(
                        filtered_cache,
                        num_states,
                        score_threshold,
                        score_topk,
                        scratch,
                        dest_tile_shared,
                        reinterpret_cast<int*>(dest_tile_shared + kMaxNativeScoreTopK));
                    if (tid == 0) {
                        tile_stats[0] = -INFINITY;
                        tile_stats[1] = 0.0f;
                    }
                    __syncthreads();
                    for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                        const int active_states = min(tile_size, num_states - state_start);
                        float filtered_max = -INFINITY;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int s = state_start + dst_local;
                            filtered_max = fmaxf(filtered_max, filtered_cache[s]);
                        }
                        __syncthreads();
                        const float tile_max = block_reduce_max_128(filtered_max, scratch);
                        float tile_exp_partial = 0.0f;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int s = state_start + dst_local;
                            const float filtered_value = filtered_cache[s];
                            tile_exp_partial += isfinite(filtered_value)
                                ? fast_exp(filtered_value - tile_max)
                                : 0.0f;
                        }
                        const float tile_sum = block_reduce_sum_128(tile_exp_partial, scratch);
                        if (tid == 0) {
                            update_online_logsumexp_stats(tile_max, tile_sum, tile_stats);
                        }
                        __syncthreads();
                    }
                }

                const float log_norm = tile_stats[0] + fast_log(fmaxf(tile_stats[1], 1.0e-20f));
                __syncthreads();

                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float next_log = score_cache[s] - log_norm;
                    prev_log[s] = next_log;
                    beliefs[base + s] = store_from_float<scalar_t>(next_log);
                }
                __syncthreads();
            }
        }
        for (int s = tid; s < num_states; s += blockDim.x) {
            final_log_belief[b * num_states + s] = store_from_float<scalar_t>(prev_log[s]);
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_backward_masked_tiled_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const float* __restrict__ transition_source_logits,
    const float* __restrict__ transition_dest_logits,
    const scalar_t* __restrict__ transition_context,
    const float* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const bool* __restrict__ transition_mask,
    const float* __restrict__ row_sums,
    const int64_t* __restrict__ seq_lens,
    float score_clamp_min,
    float score_clamp_max,
    float score_threshold,
    int score_topk,
    int num_states,
    int transition_rank,
    int tile_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    float* __restrict__ masked_transition_tile_cache,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_per_batch,
    float* __restrict__ grad_transition_dest_per_batch,
    scalar_t* __restrict__ grad_transition_context,
    float* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate_per_batch,
    float* __restrict__ grad_transition_stay_per_batch) {
    const int tid = threadIdx.x;
    __shared__ int current_batch;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    extern __shared__ float shared_mem[];
    float* prev_prob = shared_mem;
    float* carry = prev_prob + num_states;
    float* next_carry = carry + num_states;
    float* grad_mix = next_carry + num_states;
    float* row_proj_cache = grad_mix + num_states;
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    float* scratch = row_proj_cache + num_states;
    const int rank_chunk = max(1, min(tile_size, kMaskedBackwardRankChunk));
    float* dest_logits_tile = scratch + num_warps + 4;
    float* source_logits_tile = dest_logits_tile + rank_chunk * tile_size;

    while (true) {
        if (tid == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float* grad_source_batch = grad_transition_source_per_batch + static_cast<int64_t>(b) * num_states * transition_rank;
        float* grad_dest_batch = grad_transition_dest_per_batch + static_cast<int64_t>(b) * transition_rank * num_states;
        float* grad_stay_batch = grad_transition_stay_per_batch + static_cast<int64_t>(b) * num_states;
        float* tile_cache = masked_transition_tile_cache
            + static_cast<int64_t>(blockIdx.x) * num_states * tile_size;

        for (int s = tid; s < num_states; s += blockDim.x) {
            carry[s] = load_as_float(grad_final_belief + (b * num_states + s));
        }
        __syncthreads();

        float gate_grad_accum = 0.0f;
        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * num_states;
                const bool active_step = !has_seq_lens || (pos < static_cast<int>(seq_lens[b]));

                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float prev_log_value = (pos == 0)
                        ? initial_log_belief[b * num_states + s]
                        : load_as_float(beliefs + (((static_cast<int64_t>(b) * seq_len + (pos - 1)) * num_states) + s));
                    prev_prob[s] = fast_exp(prev_log_value);
                    next_carry[s] = 0.0f;
                }
                __syncthreads();

                if (!active_step) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        carry[s] += load_as_float(grad_beliefs + (base + s));
                    }
                    __syncthreads();
                    continue;
                }

                float gq_sum_partial = 0.0f;
                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float q_log_value = load_as_float(beliefs + (base + s));
                    if (isfinite(q_log_value)) {
                        gq_sum_partial += load_as_float(grad_beliefs + (base + s)) + carry[s];
                    }
                }
                const float gq_sum = block_reduce_sum_128(gq_sum_partial, scratch);
                __syncthreads();

                for (int s = tid; s < num_states; s += blockDim.x) {
                    grad_mix[s] = 0.0f;
                    row_proj_cache[s] = 0.0f;
                    next_carry[s] = 0.0f;
                }
                __syncthreads();

                for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                    const int active_states = min(tile_size, num_states - state_start);
                    for (int linear = tid; linear < num_states * active_states; linear += blockDim.x) {
                        const int src = linear / active_states;
                        const int dst_local = linear - (src * active_states);
                        const int dst = state_start + dst_local;
                        tile_cache[src * tile_size + dst_local] = masked_transition_raw_value(
                            transition_source_logits,
                            transition_dest_logits,
                            transition_mask,
                            num_states,
                            transition_rank,
                            src,
                            dst) / row_sums[src];
                    }
                    __syncthreads();

                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        const int dst = state_start + dst_local;
                        const float q_log_value = load_as_float(beliefs + (base + dst));
                        const bool active_state = isfinite(q_log_value);
                        if (!active_state) {
                            grad_local_logits[base + dst] = store_from_float<scalar_t>(0.0f);
                            grad_transition_context[base + dst] = store_from_float<scalar_t>(0.0f);
                            grad_mix[dst] = 0.0f;
                            continue;
                        }
                        float mix_prob = 0.0f;
                        for (int src = 0; src < num_states; ++src) {
                            mix_prob += prev_prob[src] * tile_cache[src * tile_size + dst_local];
                        }
                        const float prev_prob_value = prev_prob[dst];
                        const float q_prob_value = fast_exp(q_log_value);
                        const float gq = load_as_float(grad_beliefs + (base + dst)) + carry[dst];
                        const float ga = gq - q_prob_value * gq_sum;
                        const float stay_prob = transition_stay_probs[dst];
                        const float pred_prob = fmaxf(
                            stay_prob * prev_prob_value + (1.0f - stay_prob) * mix_prob,
                            1.0e-20f);
                        const float pred_log = fast_log(pred_prob);
                        const float context_value = load_as_float(transition_context + (base + dst));
                        const float prior_value = pred_log + context_value;
                        const float clamped_prior = apply_score_clamp(
                            prior_value,
                            score_clamp_min,
                            score_clamp_max);
                        const float grad_prior = (transition_gate * ga) * score_clamp_grad(
                            prior_value,
                            score_clamp_min,
                            score_clamp_max);
                        const float grad_pred_prob = grad_prior / pred_prob;
                        grad_local_logits[base + dst] = store_from_float<scalar_t>(ga);
                        grad_transition_context[base + dst] = store_from_float<scalar_t>(grad_prior);
                        grad_mix[dst] = grad_pred_prob * (1.0f - stay_prob);
                        grad_stay_batch[dst] += grad_pred_prob * (prev_prob_value - mix_prob);
                        next_carry[dst] += grad_pred_prob * stay_prob;
                        gate_grad_accum += ga * clamped_prior;
                    }
                    __syncthreads();
                    for (int src = tid; src < num_states; src += blockDim.x) {
                        float carry_from_mix_tile = 0.0f;
                        for (int dst_local = 0; dst_local < active_states; ++dst_local) {
                            const int dst = state_start + dst_local;
                            carry_from_mix_tile += grad_mix[dst] * tile_cache[src * tile_size + dst_local];
                        }
                        row_proj_cache[src] += carry_from_mix_tile;
                    }
                    __syncthreads();
                }

                for (int src = tid; src < num_states; src += blockDim.x) {
                    const float prev_prob_value = prev_prob[src];
                    next_carry[src] = (next_carry[src] + row_proj_cache[src]) * prev_prob_value;
                    row_proj_cache[src] = prev_prob_value * row_proj_cache[src];
                }
                __syncthreads();

                for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                    const int active_states = min(tile_size, num_states - state_start);
                    for (int linear = tid; linear < num_states * active_states; linear += blockDim.x) {
                        const int src = linear / active_states;
                        const int dst_local = linear - (src * active_states);
                        const int dst = state_start + dst_local;
                        float grad_raw = 0.0f;
                        if (transition_mask[src * num_states + dst]) {
                            grad_raw =
                                ((prev_prob[src] * grad_mix[dst]) - row_proj_cache[src]) / row_sums[src];
                        }
                        tile_cache[src * tile_size + dst_local] = grad_raw;
                    }
                    __syncthreads();

                    for (int src = tid; src < num_states; src += blockDim.x) {
                        const int64_t source_base = static_cast<int64_t>(src) * transition_rank;
                        for (int rank_start = 0; rank_start < transition_rank; rank_start += rank_chunk) {
                            const int active_rank = min(rank_chunk, transition_rank - rank_start);
                            for (int linear = tid; linear < active_rank * active_states; linear += blockDim.x) {
                                const int r_local = linear / active_states;
                                const int dst_local = linear - (r_local * active_states);
                                const int dst = state_start + dst_local;
                                dest_logits_tile[r_local * tile_size + dst_local] =
                                    transition_dest_logits[static_cast<int64_t>(rank_start + r_local) * num_states + dst];
                            }
                            __syncthreads();
                            float grad_source_accum[kMaskedBackwardRankChunk];
                            #pragma unroll
                            for (int r_local = 0; r_local < kMaskedBackwardRankChunk; ++r_local) {
                                grad_source_accum[r_local] = 0.0f;
                            }
                            for (int dst_local = 0; dst_local < active_states; ++dst_local) {
                                const float grad_raw = tile_cache[src * tile_size + dst_local];
                                #pragma unroll
                                for (int r_local = 0; r_local < kMaskedBackwardRankChunk; ++r_local) {
                                    if (r_local < active_rank) {
                                        grad_source_accum[r_local] += grad_raw
                                            * dest_logits_tile[r_local * tile_size + dst_local];
                                    }
                                }
                            }
                            for (int r_local = 0; r_local < active_rank; ++r_local) {
                                grad_source_batch[source_base + rank_start + r_local] += grad_source_accum[r_local];
                            }
                            __syncthreads();
                        }
                    }
                    __syncthreads();

                    for (int rank_start = 0; rank_start < transition_rank; rank_start += rank_chunk) {
                        const int active_rank = min(rank_chunk, transition_rank - rank_start);
                        for (int linear = tid; linear < active_rank * num_states; linear += blockDim.x) {
                            const int r_local = linear / num_states;
                            const int src = linear - (r_local * num_states);
                            source_logits_tile[r_local * num_states + src] =
                                transition_source_logits[static_cast<int64_t>(src) * transition_rank + rank_start + r_local];
                        }
                        __syncthreads();
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const int dst = state_start + dst_local;
                            for (int r_local = 0; r_local < active_rank; ++r_local) {
                                float grad_dest_value = 0.0f;
                                for (int src = 0; src < num_states; ++src) {
                                    grad_dest_value += source_logits_tile[r_local * num_states + src]
                                        * tile_cache[src * tile_size + dst_local];
                                }
                                grad_dest_batch[static_cast<int64_t>(rank_start + r_local) * num_states + dst] += grad_dest_value;
                            }
                        }
                        __syncthreads();
                    }
                    __syncthreads();
                }

                for (int s = tid; s < num_states; s += blockDim.x) {
                    carry[s] = next_carry[s];
                }
                __syncthreads();
            }
        }

        for (int s = tid; s < num_states; s += blockDim.x) {
            grad_initial_log_belief[b * num_states + s] = carry[s];
        }
        const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
        if (tid == 0) {
            grad_transition_gate_per_batch[b] += gate_sum;
        }
        __syncthreads();
    }
}

template <typename scalar_t, bool Sm90Path = false>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_backward_tiled_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const float* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const int64_t* __restrict__ seq_lens,
    float score_clamp_min,
    float score_clamp_max,
    float score_threshold,
    int score_topk,
    int num_states,
    int transition_rank,
    int tile_size,
    int split_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    float* __restrict__ latent_cache_staging,
    float* __restrict__ grad_latent_accum_staging,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_probs_staging,
    float* __restrict__ grad_transition_dest_probs_staging,
    float* __restrict__ grad_transition_source_probs_output,
    float* __restrict__ grad_transition_dest_probs_output,
    scalar_t* __restrict__ grad_transition_context,
    float* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate_staging,
    float* __restrict__ grad_transition_stay_staging,
    float* __restrict__ grad_transition_gate_output,
    float* __restrict__ grad_transition_stay_output) {
    const int tid = threadIdx.x;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    const int staged_rank_chunk = max(1, min(split_size, Sm90Path ? kSm90AsyncTileRankChunk : kAsyncTileRankChunk));
    __shared__ int current_batch;
    const int block_slot = static_cast<int>(blockIdx.x);
    float* latent_cache = latent_cache_staging + static_cast<int64_t>(block_slot) * transition_rank;
    float* grad_latent_accum = grad_latent_accum_staging
        + static_cast<int64_t>(block_slot) * transition_rank;
    float* grad_transition_source_probs = grad_transition_source_probs_staging
        + static_cast<int64_t>(block_slot) * num_states * transition_rank;
    float* grad_transition_dest_probs = grad_transition_dest_probs_staging
        + static_cast<int64_t>(block_slot) * transition_rank * num_states;
    float* grad_transition_gate = grad_transition_gate_staging + block_slot;
    float* grad_transition_stay = grad_transition_stay_staging
        + static_cast<int64_t>(block_slot) * num_states;

    extern __shared__ float shared_mem[];
    float* prev_prob = shared_mem;
    float* carry = prev_prob + num_states;
    float* next_carry = carry + num_states;
    float* latent = next_carry + num_states;
    float* grad_mix_tile = latent + staged_rank_chunk;
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    float* scratch = grad_mix_tile + tile_size;
    float* dest_tile_shared = scratch + num_warps;
    float* source_tile_shared = dest_tile_shared
        + static_cast<int64_t>(kAsyncTilePipelineStages) * staged_rank_chunk * tile_size;
    auto tile_pipe = cuda::make_pipeline();
    char* tensor_core_workspace_raw = reinterpret_cast<char*>(
        source_tile_shared + static_cast<int64_t>(kAsyncTilePipelineStages) * num_states * staged_rank_chunk);
    std::uintptr_t tensor_core_workspace_addr = reinterpret_cast<std::uintptr_t>(tensor_core_workspace_raw);
    tensor_core_workspace_addr = (tensor_core_workspace_addr + 31u) & ~std::uintptr_t(31u);
    char* tensor_core_workspace = reinterpret_cast<char*>(tensor_core_workspace_addr);
    using tensor_core_input_t = tensor_core_input_type_t<scalar_t>;
    tensor_core_input_t* tensor_core_lhs = reinterpret_cast<tensor_core_input_t*>(tensor_core_workspace);
    tensor_core_input_t* tensor_core_rhs = tensor_core_lhs + (kTensorCoreTile * kTensorCoreTile);
    float* tensor_core_accum = reinterpret_cast<float*>(tensor_core_rhs + (kTensorCoreTile * kTensorCoreTile));
    const bool use_tensor_core_math =
        tensor_core_math_enabled_for_scalar<scalar_t>()
        && (transition_rank >= kTensorCoreTile)
        && (tile_size >= kTensorCoreTile)
        && (split_size >= kTensorCoreTile);

    while (true) {
        if (tid == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        for (int s = tid; s < num_states; s += blockDim.x) {
            carry[s] = load_as_float(grad_final_belief + (b * num_states + s));
        }
        __syncthreads();

        float gate_grad_accum = 0.0f;

        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * num_states;
                const bool active_step = !has_seq_lens || (pos < static_cast<int>(seq_lens[b]));

                if (!active_step) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        carry[s] += load_as_float(grad_beliefs + (base + s));
                    }
                    __syncthreads();
                    continue;
                }

                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float prev_log_value = (pos == 0)
                        ? initial_log_belief[b * num_states + s]
                        : load_as_float(beliefs + (((static_cast<int64_t>(b) * seq_len + (pos - 1)) * num_states) + s));
                    prev_prob[s] = fast_exp(prev_log_value);
                    next_carry[s] = 0.0f;
                }
                __syncthreads();

                float gq_sum_partial = 0.0f;
                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float q_log_value = load_as_float(beliefs + (base + s));
                    if (isfinite(q_log_value)) {
                        gq_sum_partial += load_as_float(grad_beliefs + (base + s)) + carry[s];
                    }
                }
                const float gq_sum = block_reduce_sum_128(gq_sum_partial, scratch);
                __syncthreads();

                if (transition_rank > 0) {
                    int current_rank_start = 0;
                    int current_active_rank = min(staged_rank_chunk, transition_rank);
                    int current_stage = 0;
                    enqueue_float_matrix_slice_async(
                        tile_pipe,
                        source_tile_shared,
                        transition_source_probs,
                        transition_rank,
                        num_states,
                        current_active_rank,
                        staged_rank_chunk);
                    wait_for_async_tile(tile_pipe);
                    __syncthreads();
                    while (current_rank_start < transition_rank) {
                        const int next_rank_start = current_rank_start + current_active_rank;
                        const int next_active_rank = next_rank_start < transition_rank
                            ? min(staged_rank_chunk, transition_rank - next_rank_start)
                            : 0;
                        const int next_stage = current_stage ^ 1;
                        if (next_active_rank > 0) {
                            enqueue_float_matrix_slice_async(
                                tile_pipe,
                                source_tile_shared + static_cast<int64_t>(next_stage) * num_states * staged_rank_chunk,
                                transition_source_probs + next_rank_start,
                                transition_rank,
                                num_states,
                                next_active_rank,
                                staged_rank_chunk);
                        }
                        const float* current_source_tile = source_tile_shared
                            + static_cast<int64_t>(current_stage) * num_states * staged_rank_chunk;
                        if (use_tensor_core_math && current_active_rank == kTensorCoreTile) {
                            for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                latent[r] = 0.0f;
                            }
                            __syncthreads();
                            for (int src_start = 0; src_start < num_states; src_start += kTensorCoreTile) {
                                const int active_src = min(kTensorCoreTile, num_states - src_start);
#if __CUDA_ARCH__ >= 700
                                if (active_src == kTensorCoreTile) {
                                    wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                        prev_prob + src_start,
                                        current_source_tile + static_cast<int64_t>(src_start) * staged_rank_chunk,
                                        staged_rank_chunk,
                                        tensor_core_lhs,
                                        tensor_core_rhs,
                                        tensor_core_accum,
                                        latent);
                                } else
#endif
                                {
                                    for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                        latent[r] += lowp_dot_rhs_strided(
                                            prev_prob + src_start,
                                            current_source_tile + static_cast<int64_t>(src_start) * staged_rank_chunk + r,
                                            staged_rank_chunk,
                                            active_src);
                                    }
                                    __syncthreads();
                                }
                            }
                            for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                latent_cache[current_rank_start + r] = latent[r];
                                grad_latent_accum[current_rank_start + r] = 0.0f;
                            }
                        } else {
                            for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                latent_cache[current_rank_start + r] = lowp_dot_rhs_strided(
                                    prev_prob,
                                    current_source_tile + r,
                                    staged_rank_chunk,
                                    num_states);
                                grad_latent_accum[current_rank_start + r] = 0.0f;
                            }
                        }
                        if (next_active_rank > 0) {
                            wait_for_async_tile(tile_pipe);
                        }
                        __syncthreads();
                        current_rank_start = next_rank_start;
                        current_active_rank = next_active_rank;
                        current_stage = next_stage;
                    }
                }

                for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                    const int active_states = min(tile_size, num_states - state_start);
                    for (int dst = tid; dst < active_states; dst += blockDim.x) {
                        grad_mix_tile[dst] = 0.0f;
                    }
                    __syncthreads();

                    if (transition_rank > 0) {
                        int current_rank_start = 0;
                        int current_active_rank = min(staged_rank_chunk, transition_rank);
                        int current_stage = 0;
                        enqueue_float_matrix_slice_async(
                            tile_pipe,
                            dest_tile_shared,
                            transition_dest_probs + state_start,
                            num_states,
                            current_active_rank,
                            active_states,
                            static_cast<int>(tile_size));
                        wait_for_async_tile(tile_pipe);
                        __syncthreads();
                        while (current_rank_start < transition_rank) {
                            const int next_rank_start = current_rank_start + current_active_rank;
                            const int next_active_rank = next_rank_start < transition_rank
                                ? min(staged_rank_chunk, transition_rank - next_rank_start)
                                : 0;
                            const int next_stage = current_stage ^ 1;
                            if (next_active_rank > 0) {
                                enqueue_float_matrix_slice_async(
                                    tile_pipe,
                                    dest_tile_shared + static_cast<int64_t>(next_stage) * staged_rank_chunk * tile_size,
                                    transition_dest_probs + static_cast<int64_t>(next_rank_start) * num_states + state_start,
                                    num_states,
                                    next_active_rank,
                                    active_states,
                                    static_cast<int>(tile_size));
                            }
                            for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                latent[r] = latent_cache[current_rank_start + r];
                            }
                            __syncthreads();
                            const float* current_dest_tile = dest_tile_shared
                                + static_cast<int64_t>(current_stage) * staged_rank_chunk * tile_size;
                            if (use_tensor_core_math && current_active_rank == kTensorCoreTile) {
                                for (int dst_subtile = 0; dst_subtile < active_states; dst_subtile += kTensorCoreTile) {
                                    const int active_dst = min(kTensorCoreTile, active_states - dst_subtile);
#if __CUDA_ARCH__ >= 700
                                    if (active_dst == kTensorCoreTile) {
                                        wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                            latent,
                                            current_dest_tile + dst_subtile,
                                            tile_size,
                                            tensor_core_lhs,
                                            tensor_core_rhs,
                                            tensor_core_accum,
                                            grad_mix_tile + dst_subtile);
                                    } else
#endif
                                    {
                                        for (int dst = dst_subtile + tid; dst < dst_subtile + active_dst; dst += blockDim.x) {
                                            grad_mix_tile[dst] += lowp_dot_rhs_strided(
                                                latent,
                                                current_dest_tile + dst,
                                                tile_size,
                                                current_active_rank);
                                        }
                                        __syncthreads();
                                    }
                                }
                            } else {
                                for (int dst = tid; dst < active_states; dst += blockDim.x) {
                                    grad_mix_tile[dst] += lowp_dot_rhs_strided(
                                        latent,
                                        current_dest_tile + dst,
                                        tile_size,
                                        current_active_rank);
                                }
                            }
                            if (next_active_rank > 0) {
                                wait_for_async_tile(tile_pipe);
                            }
                            __syncthreads();
                            current_rank_start = next_rank_start;
                            current_active_rank = next_active_rank;
                            current_stage = next_stage;
                        }
                    }

                    for (int dst = tid; dst < active_states; dst += blockDim.x) {
                        const int s = state_start + dst;
                        const float q_log_value = load_as_float(beliefs + (base + s));
                        const bool active_state = isfinite(q_log_value);
                        if (!active_state) {
                            grad_local_logits[base + s] = store_from_float<scalar_t>(0.0f);
                            grad_transition_context[base + s] = store_from_float<scalar_t>(0.0f);
                            grad_mix_tile[dst] = 0.0f;
                            continue;
                        }
                        const float prev_prob_value = prev_prob[s];
                        const float q_prob_value = fast_exp(q_log_value);
                        const float gq = load_as_float(grad_beliefs + (base + s)) + carry[s];
                        const float ga = gq - q_prob_value * gq_sum;
                        const float stay_prob = transition_stay_probs[s];
                        const float mix_prob = grad_mix_tile[dst];
                        const float pred_prob = fmaxf(
                            stay_prob * prev_prob_value + (1.0f - stay_prob) * mix_prob,
                            1.0e-20f);
                        const float pred_log = fast_log(pred_prob);
                        const float context_value = load_as_float(transition_context + (base + s));
                        const float prior_value = pred_log + context_value;
                        const float clamped_prior = apply_score_clamp(
                            prior_value,
                            score_clamp_min,
                            score_clamp_max);
                        const float grad_prior = (transition_gate * ga) * score_clamp_grad(
                            prior_value,
                            score_clamp_min,
                            score_clamp_max);
                        const float grad_pred_prob = grad_prior / pred_prob;
                        grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
                        grad_transition_context[base + s] = store_from_float<scalar_t>(grad_prior);
                        gate_grad_accum += ga * clamped_prior;
                        grad_transition_stay[s] += grad_pred_prob * (prev_prob_value - mix_prob);
                        grad_mix_tile[dst] = grad_pred_prob * (1.0f - stay_prob);
                        next_carry[s] += grad_pred_prob * stay_prob;
                    }
                    __syncthreads();

                    if (transition_rank > 0) {
                        int current_rank_start = 0;
                        int current_active_rank = min(staged_rank_chunk, transition_rank);
                        int current_stage = 0;
                        enqueue_float_matrix_slice_async(
                            tile_pipe,
                            dest_tile_shared,
                            transition_dest_probs + state_start,
                            num_states,
                            current_active_rank,
                            active_states,
                            static_cast<int>(tile_size));
                        wait_for_async_tile(tile_pipe);
                        __syncthreads();
                        while (current_rank_start < transition_rank) {
                            const int next_rank_start = current_rank_start + current_active_rank;
                            const int next_active_rank = next_rank_start < transition_rank
                                ? min(staged_rank_chunk, transition_rank - next_rank_start)
                                : 0;
                            const int next_stage = current_stage ^ 1;
                            if (next_active_rank > 0) {
                                enqueue_float_matrix_slice_async(
                                    tile_pipe,
                                    dest_tile_shared + static_cast<int64_t>(next_stage) * staged_rank_chunk * tile_size,
                                    transition_dest_probs + static_cast<int64_t>(next_rank_start) * num_states + state_start,
                                    num_states,
                                    next_active_rank,
                                    active_states,
                                    static_cast<int>(tile_size));
                            }
                            for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                latent[r] = latent_cache[current_rank_start + r];
                            }
                            __syncthreads();

                            const float* current_dest_tile = dest_tile_shared
                                + static_cast<int64_t>(current_stage) * staged_rank_chunk * tile_size;
                            if (use_tensor_core_math && current_active_rank == kTensorCoreTile) {
                                for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                    latent[r] = 0.0f;
                                }
                                __syncthreads();
                                for (int dst_subtile = 0; dst_subtile < active_states; dst_subtile += kTensorCoreTile) {
                                    const int active_dst = min(kTensorCoreTile, active_states - dst_subtile);
#if __CUDA_ARCH__ >= 700
                                    if (active_dst == kTensorCoreTile) {
                                        wmma_matrix_times_replicated_col_vector_16x16<scalar_t>(
                                            current_dest_tile + dst_subtile,
                                            tile_size,
                                            grad_mix_tile + dst_subtile,
                                            tensor_core_lhs,
                                            tensor_core_rhs,
                                            tensor_core_accum,
                                            latent);
                                    } else
#endif
                                    {
                                        for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                            const float* dest_row = current_dest_tile + static_cast<int64_t>(r) * tile_size + dst_subtile;
                                            latent[r] += lowp_dot_contiguous(
                                                grad_mix_tile + dst_subtile,
                                                dest_row,
                                                active_dst);
                                        }
                                        __syncthreads();
                                    }
                                    for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                        const int r_idx = current_rank_start + r;
                                        for (int dst = dst_subtile; dst < dst_subtile + active_dst; ++dst) {
                                            const int s = state_start + dst;
                                            grad_transition_dest_probs[static_cast<int64_t>(r_idx) * num_states + s]
                                                += latent_cache[r_idx] * grad_mix_tile[dst];
                                        }
                                    }
                                    __syncthreads();
                                }
                                for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                    grad_latent_accum[current_rank_start + r] += latent[r];
                                }
                            } else {
                                for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                    const int r_idx = current_rank_start + r;
                                    const float* dest_row = current_dest_tile + static_cast<int64_t>(r) * tile_size;
                                    const float grad_latent_value = lowp_dot_contiguous(
                                        grad_mix_tile,
                                        dest_row,
                                        active_states);
                                    for (int dst = 0; dst < active_states; ++dst) {
                                        const int s = state_start + dst;
                                        grad_transition_dest_probs[static_cast<int64_t>(r_idx) * num_states + s]
                                            += latent_cache[r_idx] * grad_mix_tile[dst];
                                    }
                                    grad_latent_accum[r_idx] += grad_latent_value;
                                }
                            }
                            if (next_active_rank > 0) {
                                wait_for_async_tile(tile_pipe);
                            }
                            __syncthreads();
                            current_rank_start = next_rank_start;
                            current_active_rank = next_active_rank;
                            current_stage = next_stage;
                        }
                    }
                }

                __syncthreads();
                if (transition_rank > 0) {
                    int current_rank_start = 0;
                    int current_active_rank = min(staged_rank_chunk, transition_rank);
                    int current_stage = 0;
                    enqueue_float_matrix_slice_async(
                        tile_pipe,
                        source_tile_shared,
                        transition_source_probs,
                        transition_rank,
                        num_states,
                        current_active_rank,
                        staged_rank_chunk);
                    wait_for_async_tile(tile_pipe);
                    __syncthreads();
                    while (current_rank_start < transition_rank) {
                        const int next_rank_start = current_rank_start + current_active_rank;
                        const int next_active_rank = next_rank_start < transition_rank
                            ? min(staged_rank_chunk, transition_rank - next_rank_start)
                            : 0;
                        const int next_stage = current_stage ^ 1;
                        if (next_active_rank > 0) {
                            enqueue_float_matrix_slice_async(
                                tile_pipe,
                                source_tile_shared + static_cast<int64_t>(next_stage) * num_states * staged_rank_chunk,
                                transition_source_probs + next_rank_start,
                                transition_rank,
                                num_states,
                                next_active_rank,
                                staged_rank_chunk);
                        }
                        for (int r = tid; r < current_active_rank; r += blockDim.x) {
                            latent[r] = grad_latent_accum[current_rank_start + r];
                        }
                        __syncthreads();
                        const float* current_source_tile = source_tile_shared
                            + static_cast<int64_t>(current_stage) * num_states * staged_rank_chunk;
                        if (use_tensor_core_math && current_active_rank == kTensorCoreTile) {
                            for (int src_start = 0; src_start < num_states; src_start += kTensorCoreTile) {
                                const int active_src = min(kTensorCoreTile, num_states - src_start);
                                for (int src_local = tid; src_local < active_src; src_local += blockDim.x) {
                                    grad_mix_tile[src_local] = 0.0f;
                                }
                                __syncthreads();
#if __CUDA_ARCH__ >= 700
                                if (active_src == kTensorCoreTile) {
                                    wmma_matrix_times_replicated_col_vector_16x16<scalar_t>(
                                        current_source_tile + static_cast<int64_t>(src_start) * staged_rank_chunk,
                                        staged_rank_chunk,
                                        latent,
                                        tensor_core_lhs,
                                        tensor_core_rhs,
                                        tensor_core_accum,
                                        grad_mix_tile);
                                } else
#endif
                                {
                                    for (int src_local = tid; src_local < active_src; src_local += blockDim.x) {
                                        grad_mix_tile[src_local] = lowp_dot_contiguous(
                                            latent,
                                            current_source_tile + static_cast<int64_t>(src_start + src_local) * staged_rank_chunk,
                                            current_active_rank);
                                    }
                                    __syncthreads();
                                }
                                for (int src_local = tid; src_local < active_src; src_local += blockDim.x) {
                                    const int src = src_start + src_local;
                                    const float prev_prob_value = prev_prob[src];
                                    const int64_t source_base = static_cast<int64_t>(src) * transition_rank + current_rank_start;
                                    next_carry[src] += grad_mix_tile[src_local];
                                    for (int r = 0; r < current_active_rank; ++r) {
                                        grad_transition_source_probs[source_base + r] += prev_prob_value * latent[r];
                                    }
                                }
                                __syncthreads();
                            }
                        } else {
                            for (int src = tid; src < num_states; src += blockDim.x) {
                                const float prev_prob_value = prev_prob[src];
                                const int64_t source_base = static_cast<int64_t>(src) * transition_rank + current_rank_start;
                                const float carry_contrib = lowp_dot_contiguous(
                                    latent,
                                    current_source_tile + static_cast<int64_t>(src) * staged_rank_chunk,
                                    current_active_rank);
                                for (int r = 0; r < current_active_rank; ++r) {
                                    grad_transition_source_probs[source_base + r] += prev_prob_value * latent[r];
                                }
                                next_carry[src] += carry_contrib;
                            }
                        }
                        if (next_active_rank > 0) {
                            wait_for_async_tile(tile_pipe);
                        }
                        __syncthreads();
                        current_rank_start = next_rank_start;
                        current_active_rank = next_active_rank;
                        current_stage = next_stage;
                    }
                }

                for (int s = tid; s < num_states; s += blockDim.x) {
                    carry[s] = next_carry[s] * prev_prob[s];
                }
                __syncthreads();
            }
        }

        for (int s = tid; s < num_states; s += blockDim.x) {
            grad_initial_log_belief[b * num_states + s] = carry[s];
        }
        const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
        if (tid == 0) {
            grad_transition_gate[0] += gate_sum;
        }
        __syncthreads();
    }

    const int64_t source_total = static_cast<int64_t>(num_states) * transition_rank;
    const int64_t dest_total = static_cast<int64_t>(transition_rank) * num_states;
    const int64_t stay_total = num_states;
    if (gridDim.x == 1) {
        for (int64_t idx = tid; idx < source_total; idx += blockDim.x) {
            grad_transition_source_probs_output[idx] = grad_transition_source_probs[idx];
        }
        for (int64_t idx = tid; idx < dest_total; idx += blockDim.x) {
            grad_transition_dest_probs_output[idx] = grad_transition_dest_probs[idx];
        }
        for (int64_t idx = tid; idx < stay_total; idx += blockDim.x) {
            grad_transition_stay_output[idx] = grad_transition_stay[idx];
        }
        if (tid == 0) {
            grad_transition_gate_output[0] = grad_transition_gate[0];
        }
    } else {
        for (int64_t idx = tid; idx < source_total; idx += blockDim.x) {
            atomicAdd(grad_transition_source_probs_output + idx, grad_transition_source_probs[idx]);
        }
        for (int64_t idx = tid; idx < dest_total; idx += blockDim.x) {
            atomicAdd(grad_transition_dest_probs_output + idx, grad_transition_dest_probs[idx]);
        }
        for (int64_t idx = tid; idx < stay_total; idx += blockDim.x) {
            atomicAdd(grad_transition_stay_output + idx, grad_transition_stay[idx]);
        }
        if (tid == 0) {
            atomicAdd(grad_transition_gate_output, grad_transition_gate[0]);
        }
    }
}

template <typename scalar_t, typename packed_t, PackedTransitionFormat Format, bool Sm90Path = false>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_forward_tiled_chunk_packed_kernel(
    const scalar_t* __restrict__ local_logits,
    const packed_t* __restrict__ transition_source_packed,
    const float* __restrict__ transition_source_scales,
    const packed_t* __restrict__ transition_dest_packed,
    const float* __restrict__ transition_dest_scales,
    const scalar_t* __restrict__ transition_context,
    const float* __restrict__ initial_log_belief,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const int64_t* __restrict__ seq_lens,
    float score_clamp_min,
    float score_clamp_max,
    float score_threshold,
    int score_topk,
    int num_states,
    int transition_rank,
    int tile_size,
    int split_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    float* __restrict__ filtered_value_cache,
    scalar_t* __restrict__ beliefs,
    scalar_t* __restrict__ final_log_belief) {
    const int tid = threadIdx.x;
    __shared__ int current_batch;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    const int block_slot = static_cast<int>(blockIdx.x);
    const int staged_rank_chunk = max(1, min(split_size, Sm90Path ? kSm90AsyncTileRankChunk : kAsyncTileRankChunk));
    const int pipeline_stages = async_tile_pipeline_stages<Sm90Path>();
    extern __shared__ float shared_mem[];
    float* prev_log = shared_mem;
    float* prev_prob = prev_log + num_states;
    float* latent = prev_prob + num_states;
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    float* scratch = latent + staged_rank_chunk;
    float* tile_stats = scratch + num_warps;
    float* dest_tile_shared = tile_stats + 2;
    packed_t* dest_tile_packed_shared = reinterpret_cast<packed_t*>(
        dest_tile_shared + static_cast<int64_t>(pipeline_stages) * staged_rank_chunk * tile_size);
    auto tile_pipe = cuda::make_pipeline();
    const bool use_global_score_cache = native_score_filtering_enabled(score_threshold, score_topk, num_states);

    while (true) {
        if (tid == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        float* filtered_cache = filtered_value_cache + static_cast<int64_t>(block_slot) * num_states;
        float* score_cache = use_global_score_cache ? filtered_cache : prev_log;
        for (int s = tid; s < num_states; s += blockDim.x) {
            prev_log[s] = initial_log_belief[b * num_states + s];
        }
        __syncthreads();

        for (int tile_start = chunk_start; tile_start < seq_len; tile_start += sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = 0; t < current_chunk_len; ++t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * num_states;
                const bool active = !has_seq_lens || pos < static_cast<int>(seq_lens[b]);
                if (!active) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        beliefs[base + s] = store_from_float<scalar_t>(prev_log[s]);
                    }
                    __syncthreads();
                    continue;
                }

                for (int s = tid; s < num_states; s += blockDim.x) {
                    prev_prob[s] = fast_exp(prev_log[s]);
                }
                __syncthreads();

                if (tid == 0) {
                    tile_stats[0] = -INFINITY;
                    tile_stats[1] = 0.0f;
                }
                __syncthreads();

                for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                    const int active_states = min(tile_size, num_states - state_start);
                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        score_cache[state_start + dst_local] = 0.0f;
                    }
                    __syncthreads();

                    {
                        const int total_rank_tiles = (transition_rank + staged_rank_chunk - 1) / staged_rank_chunk;
                        int next_tile_to_launch = 0;
                        const int initial_prefetch = min(pipeline_stages, total_rank_tiles);
                        for (; next_tile_to_launch < initial_prefetch; ++next_tile_to_launch) {
                            const int rank_start = next_tile_to_launch * staged_rank_chunk;
                            const int active_rank = min(staged_rank_chunk, transition_rank - rank_start);
                            const int stage_slot = next_tile_to_launch % pipeline_stages;
                            enqueue_packed_matrix_slice_async(
                                tile_pipe,
                                dest_tile_packed_shared + static_cast<int64_t>(stage_slot) * staged_rank_chunk * tile_size,
                                transition_dest_packed + static_cast<int64_t>(rank_start) * num_states + state_start,
                                num_states,
                                active_rank,
                                active_states,
                                tile_size);
                        }
                        if (initial_prefetch > 0) {
                            wait_for_async_tile(tile_pipe);
                            __syncthreads();
                        }
                        for (int rank_tile_idx = 0; rank_tile_idx < total_rank_tiles; ++rank_tile_idx) {
                            const int current_rank_start = rank_tile_idx * staged_rank_chunk;
                            const int current_active_rank = min(staged_rank_chunk, transition_rank - current_rank_start);
                            const int current_stage = rank_tile_idx % pipeline_stages;
                            if (next_tile_to_launch < total_rank_tiles) {
                                const int next_rank_start = next_tile_to_launch * staged_rank_chunk;
                                const int next_active_rank = min(staged_rank_chunk, transition_rank - next_rank_start);
                                const int next_stage = next_tile_to_launch % pipeline_stages;
                                enqueue_packed_matrix_slice_async(
                                    tile_pipe,
                                    dest_tile_packed_shared + static_cast<int64_t>(next_stage) * staged_rank_chunk * tile_size,
                                    transition_dest_packed + static_cast<int64_t>(next_rank_start) * num_states + state_start,
                                    num_states,
                                    next_active_rank,
                                    active_states,
                                    tile_size);
                                ++next_tile_to_launch;
                            }
                            for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                latent[r] = packed_column_dot_lowp<packed_t, Format>(
                                    prev_prob,
                                    transition_source_packed,
                                    transition_source_scales,
                                    num_states,
                                    transition_rank,
                                    current_rank_start + r);
                            }
                            const packed_t* current_packed_tile = dest_tile_packed_shared
                                + static_cast<int64_t>(current_stage) * staged_rank_chunk * tile_size;
                            for (int r = 0; r < current_active_rank; ++r) {
                                const packed_t* packed_row = current_packed_tile + static_cast<int64_t>(r) * tile_size;
                                float* dest_row = dest_tile_shared + static_cast<int64_t>(r) * tile_size;
                                const float dest_scale = transition_dest_scales[current_rank_start + r];
                                for (int dst_local = 2 * tid; dst_local < active_states; dst_local += 2 * blockDim.x) {
                                    if (dst_local + 1 < active_states) {
                                        store_pair_from_float<float>(
                                            dest_row + dst_local,
                                            {
                                                unpack_packed_value<packed_t, Format>(packed_row[dst_local], dest_scale),
                                                unpack_packed_value<packed_t, Format>(packed_row[dst_local + 1], dest_scale),
                                            });
                                    } else {
                                        dest_row[dst_local] = unpack_packed_value<packed_t, Format>(
                                            packed_row[dst_local],
                                            dest_scale);
                                    }
                                }
                            }
                            __syncthreads();
                            for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                                score_cache[state_start + dst_local] += lowp_dot_rhs_strided(
                                    latent,
                                    dest_tile_shared + dst_local,
                                    tile_size,
                                    current_active_rank);
                            }
                            if (rank_tile_idx + 1 < total_rank_tiles) {
                                wait_for_async_tile(tile_pipe);
                            }
                            __syncthreads();
                        }
                    }

                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        const int s = state_start + dst_local;
                        const float stay_prob = transition_stay_probs[s];
                        const float mix_prob = score_cache[s];
                        const float pred_prob = fmaxf(
                            stay_prob * prev_prob[s] + (1.0f - stay_prob) * mix_prob,
                            1.0e-20f);
                        const float pred_log = fast_log(pred_prob);
                        const float prior_value = pred_log + load_as_float(transition_context + base + s);
                        score_cache[s] = load_as_float(local_logits + base + s)
                            + transition_gate * apply_score_clamp(prior_value, score_clamp_min, score_clamp_max);
                    }
                    __syncthreads();

                    float filtered_max = -INFINITY;
                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        filtered_max = fmaxf(filtered_max, score_cache[state_start + dst_local]);
                    }
                    __syncthreads();
                    const float tile_max = block_reduce_max_128(filtered_max, scratch);
                    float tile_exp_partial = 0.0f;
                    for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                        const float filtered_value = score_cache[state_start + dst_local];
                        tile_exp_partial += isfinite(filtered_value) ? fast_exp(filtered_value - tile_max) : 0.0f;
                    }
                    const float tile_sum = block_reduce_sum_128(tile_exp_partial, scratch);
                    if (tid == 0) {
                        update_online_logsumexp_stats(tile_max, tile_sum, tile_stats);
                    }
                    __syncthreads();
                }

                if (use_global_score_cache) {
                    apply_native_score_filtering_block(
                        filtered_cache,
                        num_states,
                        score_threshold,
                        score_topk,
                        scratch,
                        dest_tile_shared,
                        reinterpret_cast<int*>(dest_tile_shared + kMaxNativeScoreTopK));
                    if (tid == 0) {
                        tile_stats[0] = -INFINITY;
                        tile_stats[1] = 0.0f;
                    }
                    __syncthreads();
                    for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                        const int active_states = min(tile_size, num_states - state_start);
                        float filtered_max = -INFINITY;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            filtered_max = fmaxf(filtered_max, filtered_cache[state_start + dst_local]);
                        }
                        __syncthreads();
                        const float tile_max = block_reduce_max_128(filtered_max, scratch);
                        float tile_exp_partial = 0.0f;
                        for (int dst_local = tid; dst_local < active_states; dst_local += blockDim.x) {
                            const float filtered_value = filtered_cache[state_start + dst_local];
                            tile_exp_partial += isfinite(filtered_value) ? fast_exp(filtered_value - tile_max) : 0.0f;
                        }
                        const float tile_sum = block_reduce_sum_128(tile_exp_partial, scratch);
                        if (tid == 0) {
                            update_online_logsumexp_stats(tile_max, tile_sum, tile_stats);
                        }
                        __syncthreads();
                    }
                }

                const float log_norm = tile_stats[0] + fast_log(fmaxf(tile_stats[1], 1.0e-20f));
                __syncthreads();
                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float next_log = score_cache[s] - log_norm;
                    prev_log[s] = next_log;
                    beliefs[base + s] = store_from_float<scalar_t>(next_log);
                }
                __syncthreads();
            }
        }

        for (int s = tid; s < num_states; s += blockDim.x) {
            final_log_belief[b * num_states + s] = store_from_float<scalar_t>(prev_log[s]);
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename packed_t, PackedTransitionFormat Format, bool Sm90Path = false>
__global__ CAUSAL_MACHINE_TILED_LAUNCH_BOUNDS void causal_machine_backward_tiled_chunk_packed_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const packed_t* __restrict__ transition_source_packed,
    const float* __restrict__ transition_source_scales,
    const packed_t* __restrict__ transition_dest_packed,
    const float* __restrict__ transition_dest_scales,
    const scalar_t* __restrict__ transition_context,
    const float* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    const int64_t* __restrict__ seq_lens,
    float score_clamp_min,
    float score_clamp_max,
    float score_threshold,
    int score_topk,
    int num_states,
    int transition_rank,
    int tile_size,
    int split_size,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    float* __restrict__ latent_cache_staging,
    float* __restrict__ grad_latent_accum_staging,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_probs_staging,
    float* __restrict__ grad_transition_dest_probs_staging,
    float* __restrict__ grad_transition_source_probs_output,
    float* __restrict__ grad_transition_dest_probs_output,
    scalar_t* __restrict__ grad_transition_context,
    float* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate_staging,
    float* __restrict__ grad_transition_stay_staging,
    float* __restrict__ grad_transition_gate_output,
    float* __restrict__ grad_transition_stay_output) {
    (void)score_threshold;
    (void)score_topk;
    const int tid = threadIdx.x;
    const bool has_seq_lens = seq_lens != nullptr;
    const int sequence_tile_size = max(chunk_len, 1);
    const int staged_rank_chunk = max(1, min(split_size, Sm90Path ? kSm90AsyncTileRankChunk : kAsyncTileRankChunk));
    __shared__ int current_batch;
    const int block_slot = static_cast<int>(blockIdx.x);
    float* latent_cache = latent_cache_staging + static_cast<int64_t>(block_slot) * transition_rank;
    float* grad_latent_accum = grad_latent_accum_staging + static_cast<int64_t>(block_slot) * transition_rank;
    float* grad_transition_source_probs = grad_transition_source_probs_staging + static_cast<int64_t>(block_slot) * num_states * transition_rank;
    float* grad_transition_dest_probs = grad_transition_dest_probs_staging + static_cast<int64_t>(block_slot) * transition_rank * num_states;
    float* grad_transition_gate = grad_transition_gate_staging + block_slot;
    float* grad_transition_stay = grad_transition_stay_staging + static_cast<int64_t>(block_slot) * num_states;
    extern __shared__ float shared_mem[];
    float* prev_prob = shared_mem;
    float* carry = prev_prob + num_states;
    float* next_carry = carry + num_states;
    float* latent = next_carry + num_states;
    float* grad_mix_tile = latent + staged_rank_chunk;
    const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
    float* scratch = grad_mix_tile + tile_size;
    float* dest_tile_shared = scratch + num_warps;
    packed_t* dest_tile_packed_shared = reinterpret_cast<packed_t*>(
        dest_tile_shared + static_cast<int64_t>(staged_rank_chunk) * tile_size);
    auto tile_pipe = cuda::make_pipeline();

    while (true) {
        if (tid == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        for (int s = tid; s < num_states; s += blockDim.x) {
            carry[s] = load_as_float(grad_final_belief + (b * num_states + s));
        }
        __syncthreads();
        float gate_grad_accum = 0.0f;

        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int64_t base = (static_cast<int64_t>(b) * seq_len + pos) * num_states;
                const bool active_step = !has_seq_lens || (pos < static_cast<int>(seq_lens[b]));
                if (!active_step) {
                    for (int s = tid; s < num_states; s += blockDim.x) {
                        carry[s] += load_as_float(grad_beliefs + (base + s));
                    }
                    __syncthreads();
                    continue;
                }

                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float prev_log_value = (pos == 0)
                        ? initial_log_belief[b * num_states + s]
                        : load_as_float(beliefs + (((static_cast<int64_t>(b) * seq_len + (pos - 1)) * num_states) + s));
                    prev_prob[s] = fast_exp(prev_log_value);
                    next_carry[s] = 0.0f;
                }
                __syncthreads();

                float gq_sum_partial = 0.0f;
                for (int s = tid; s < num_states; s += blockDim.x) {
                    const float q_log_value = load_as_float(beliefs + (base + s));
                    if (isfinite(q_log_value)) {
                        gq_sum_partial += load_as_float(grad_beliefs + (base + s)) + carry[s];
                    }
                }
                const float gq_sum = block_reduce_sum_128(gq_sum_partial, scratch);
                __syncthreads();

                for (int rank_start = 0; rank_start < transition_rank; rank_start += staged_rank_chunk) {
                    const int active_rank = min(staged_rank_chunk, transition_rank - rank_start);
                    for (int r = tid; r < active_rank; r += blockDim.x) {
                        latent_cache[rank_start + r] = packed_column_dot_lowp<packed_t, Format>(
                            prev_prob,
                            transition_source_packed,
                            transition_source_scales,
                            num_states,
                            transition_rank,
                            rank_start + r);
                        grad_latent_accum[rank_start + r] = 0.0f;
                    }
                    __syncthreads();
                }

                for (int state_start = 0; state_start < num_states; state_start += tile_size) {
                    const int active_states = min(tile_size, num_states - state_start);
                    for (int dst = tid; dst < active_states; dst += blockDim.x) {
                        grad_mix_tile[dst] = 0.0f;
                    }
                    __syncthreads();

                    for (int rank_start = 0; rank_start < transition_rank; rank_start += staged_rank_chunk) {
                        int current_rank_start = rank_start;
                        int current_active_rank = min(staged_rank_chunk, transition_rank - current_rank_start);
                        int current_stage = 0;
                        enqueue_packed_matrix_slice_async(
                            tile_pipe,
                            dest_tile_packed_shared,
                            transition_dest_packed + static_cast<int64_t>(current_rank_start) * num_states + state_start,
                            num_states,
                            current_active_rank,
                            active_states,
                            tile_size);
                        wait_for_async_tile(tile_pipe);
                        __syncthreads();
                        while (current_rank_start < transition_rank) {
                            const int next_rank_start = current_rank_start + current_active_rank;
                            const int next_active_rank = next_rank_start < transition_rank
                                ? min(staged_rank_chunk, transition_rank - next_rank_start)
                                : 0;
                            const int next_stage = current_stage ^ 1;
                            if (next_active_rank > 0) {
                                enqueue_packed_matrix_slice_async(
                                    tile_pipe,
                                    dest_tile_packed_shared + static_cast<int64_t>(next_stage) * staged_rank_chunk * tile_size,
                                    transition_dest_packed + static_cast<int64_t>(next_rank_start) * num_states + state_start,
                                    num_states,
                                    next_active_rank,
                                    active_states,
                                    tile_size);
                            }
                            for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                latent[r] = latent_cache[current_rank_start + r];
                            }
                            const packed_t* current_packed_tile = dest_tile_packed_shared
                                + static_cast<int64_t>(current_stage) * staged_rank_chunk * tile_size;
                            for (int r = 0; r < current_active_rank; ++r) {
                                const packed_t* packed_row = current_packed_tile + static_cast<int64_t>(r) * tile_size;
                                float* dest_row = dest_tile_shared + static_cast<int64_t>(r) * tile_size;
                                const float dest_scale = transition_dest_scales[current_rank_start + r];
                                for (int dst = 2 * tid; dst < active_states; dst += 2 * blockDim.x) {
                                    if (dst + 1 < active_states) {
                                        store_pair_from_float<float>(
                                            dest_row + dst,
                                            {
                                                unpack_packed_value<packed_t, Format>(packed_row[dst], dest_scale),
                                                unpack_packed_value<packed_t, Format>(packed_row[dst + 1], dest_scale),
                                            });
                                    } else {
                                        dest_row[dst] = unpack_packed_value<packed_t, Format>(
                                            packed_row[dst],
                                            dest_scale);
                                    }
                                }
                            }
                            __syncthreads();
                            for (int dst = tid; dst < active_states; dst += blockDim.x) {
                                grad_mix_tile[dst] += lowp_dot_rhs_strided(
                                    latent,
                                    dest_tile_shared + dst,
                                    tile_size,
                                    current_active_rank);
                            }
                            if (next_active_rank > 0) {
                                wait_for_async_tile(tile_pipe);
                            }
                            __syncthreads();
                            current_rank_start = next_rank_start;
                            current_active_rank = next_active_rank;
                            current_stage = next_stage;
                        }
                        break;
                    }

                    for (int dst = tid; dst < active_states; dst += blockDim.x) {
                        const int s = state_start + dst;
                        const float q_log_value = load_as_float(beliefs + (base + s));
                        if (!isfinite(q_log_value)) {
                            grad_local_logits[base + s] = store_from_float<scalar_t>(0.0f);
                            grad_transition_context[base + s] = store_from_float<scalar_t>(0.0f);
                            grad_mix_tile[dst] = 0.0f;
                            continue;
                        }
                        const float prev_prob_value = prev_prob[s];
                        const float q_prob_value = fast_exp(q_log_value);
                        const float gq = load_as_float(grad_beliefs + (base + s)) + carry[s];
                        const float ga = gq - q_prob_value * gq_sum;
                        const float stay_prob = transition_stay_probs[s];
                        const float mix_prob = grad_mix_tile[dst];
                        const float pred_prob = fmaxf(stay_prob * prev_prob_value + (1.0f - stay_prob) * mix_prob, 1.0e-20f);
                        const float pred_log = fast_log(pred_prob);
                        const float context_value = load_as_float(transition_context + (base + s));
                        const float prior_value = pred_log + context_value;
                        const float clamped_prior = apply_score_clamp(prior_value, score_clamp_min, score_clamp_max);
                        const float grad_prior = (transition_gate * ga) * score_clamp_grad(prior_value, score_clamp_min, score_clamp_max);
                        const float grad_pred_prob = grad_prior / pred_prob;
                        grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
                        grad_transition_context[base + s] = store_from_float<scalar_t>(grad_prior);
                        gate_grad_accum += ga * clamped_prior;
                        grad_transition_stay[s] += grad_pred_prob * (prev_prob_value - mix_prob);
                        grad_mix_tile[dst] = grad_pred_prob * (1.0f - stay_prob);
                        next_carry[s] += grad_pred_prob * stay_prob;
                    }
                    __syncthreads();

                    for (int rank_start = 0; rank_start < transition_rank; rank_start += staged_rank_chunk) {
                        int current_rank_start = rank_start;
                        int current_active_rank = min(staged_rank_chunk, transition_rank - current_rank_start);
                        int current_stage = 0;
                        enqueue_packed_matrix_slice_async(
                            tile_pipe,
                            dest_tile_packed_shared,
                            transition_dest_packed + static_cast<int64_t>(current_rank_start) * num_states + state_start,
                            num_states,
                            current_active_rank,
                            active_states,
                            tile_size);
                        wait_for_async_tile(tile_pipe);
                        __syncthreads();
                        while (current_rank_start < transition_rank) {
                            const int next_rank_start = current_rank_start + current_active_rank;
                            const int next_active_rank = next_rank_start < transition_rank
                                ? min(staged_rank_chunk, transition_rank - next_rank_start)
                                : 0;
                            const int next_stage = current_stage ^ 1;
                            if (next_active_rank > 0) {
                                enqueue_packed_matrix_slice_async(
                                    tile_pipe,
                                    dest_tile_packed_shared + static_cast<int64_t>(next_stage) * staged_rank_chunk * tile_size,
                                    transition_dest_packed + static_cast<int64_t>(next_rank_start) * num_states + state_start,
                                    num_states,
                                    next_active_rank,
                                    active_states,
                                    tile_size);
                            }
                            for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                latent[r] = latent_cache[current_rank_start + r];
                            }
                            const packed_t* current_packed_tile = dest_tile_packed_shared
                                + static_cast<int64_t>(current_stage) * staged_rank_chunk * tile_size;
                            for (int r = 0; r < current_active_rank; ++r) {
                                const packed_t* packed_row = current_packed_tile + static_cast<int64_t>(r) * tile_size;
                                float* dest_row = dest_tile_shared + static_cast<int64_t>(r) * tile_size;
                                const float dest_scale = transition_dest_scales[current_rank_start + r];
                                for (int dst = 2 * tid; dst < active_states; dst += 2 * blockDim.x) {
                                    if (dst + 1 < active_states) {
                                        store_pair_from_float<float>(
                                            dest_row + dst,
                                            {
                                                unpack_packed_value<packed_t, Format>(packed_row[dst], dest_scale),
                                                unpack_packed_value<packed_t, Format>(packed_row[dst + 1], dest_scale),
                                            });
                                    } else {
                                        dest_row[dst] = unpack_packed_value<packed_t, Format>(
                                            packed_row[dst],
                                            dest_scale);
                                    }
                                }
                            }
                            __syncthreads();
                            for (int r = tid; r < current_active_rank; r += blockDim.x) {
                                const int r_idx = current_rank_start + r;
                                const float grad_latent_value = lowp_dot_contiguous(
                                    grad_mix_tile,
                                    dest_tile_shared + static_cast<int64_t>(r) * tile_size,
                                    active_states);
                                for (int dst = 0; dst < active_states; ++dst) {
                                    grad_transition_dest_probs[static_cast<int64_t>(r_idx) * num_states + state_start + dst]
                                        += latent_cache[r_idx] * grad_mix_tile[dst];
                                }
                                grad_latent_accum[r_idx] += grad_latent_value;
                            }
                            if (next_active_rank > 0) {
                                wait_for_async_tile(tile_pipe);
                            }
                            __syncthreads();
                            current_rank_start = next_rank_start;
                            current_active_rank = next_active_rank;
                            current_stage = next_stage;
                        }
                        break;
                    }
                }

                for (int rank_start = 0; rank_start < transition_rank; rank_start += staged_rank_chunk) {
                    const int active_rank = min(staged_rank_chunk, transition_rank - rank_start);
                    for (int r = tid; r < active_rank; r += blockDim.x) {
                        latent[r] = grad_latent_accum[rank_start + r];
                    }
                    __syncthreads();
                    for (int src = tid; src < num_states; src += blockDim.x) {
                        const float prev_prob_value = prev_prob[src];
                        const int64_t source_base = static_cast<int64_t>(src) * transition_rank + rank_start;
                        const float carry_contrib = packed_row_dot_lowp<packed_t, Format>(
                            latent,
                            transition_source_packed + source_base,
                            transition_source_scales[src],
                            active_rank);
                        for (int r = 0; r < active_rank; ++r) {
                            grad_transition_source_probs[source_base + r] += prev_prob_value * latent[r];
                        }
                        next_carry[src] += carry_contrib;
                    }
                    __syncthreads();
                }

                for (int s = tid; s < num_states; s += blockDim.x) {
                    carry[s] = next_carry[s] * prev_prob[s];
                }
                __syncthreads();
            }
        }

        for (int s = tid; s < num_states; s += blockDim.x) {
            grad_initial_log_belief[b * num_states + s] = carry[s];
        }
        const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
        if (tid == 0) {
            grad_transition_gate[0] += gate_sum;
        }
        __syncthreads();
    }

    const int64_t source_total = static_cast<int64_t>(num_states) * transition_rank;
    const int64_t dest_total = static_cast<int64_t>(transition_rank) * num_states;
    if (gridDim.x == 1) {
        for (int64_t idx = tid; idx < source_total; idx += blockDim.x) {
            grad_transition_source_probs_output[idx] = grad_transition_source_probs[idx];
        }
        for (int64_t idx = tid; idx < dest_total; idx += blockDim.x) {
            grad_transition_dest_probs_output[idx] = grad_transition_dest_probs[idx];
        }
        for (int64_t idx = tid; idx < num_states; idx += blockDim.x) {
            grad_transition_stay_output[idx] = grad_transition_stay[idx];
        }
        if (tid == 0) {
            grad_transition_gate_output[0] = grad_transition_gate[0];
        }
    } else {
        for (int64_t idx = tid; idx < source_total; idx += blockDim.x) {
            atomicAdd(grad_transition_source_probs_output + idx, grad_transition_source_probs[idx]);
        }
        for (int64_t idx = tid; idx < dest_total; idx += blockDim.x) {
            atomicAdd(grad_transition_dest_probs_output + idx, grad_transition_dest_probs[idx]);
        }
        for (int64_t idx = tid; idx < num_states; idx += blockDim.x) {
            atomicAdd(grad_transition_stay_output + idx, grad_transition_stay[idx]);
        }
        if (tid == 0) {
            atomicAdd(grad_transition_gate_output, grad_transition_gate[0]);
        }
    }
}

template <typename scalar_t, int StaticTransitionRank = -1, bool DirectGradReduce = false, bool InputsAreLogits = false>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_backward_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    float score_clamp_min,
    float score_clamp_max,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_per_batch,
    float* __restrict__ grad_transition_dest_per_batch,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate_per_batch,
    float* __restrict__ grad_transition_stay_per_batch) {
    const int s = threadIdx.x;
    const int kNumStates = static_cast<int>(blockDim.x);
    const int kNumWarps = (kNumStates + kWarpSize - 1) / kWarpSize;
    const int rank = StaticTransitionRank > 0 ? StaticTransitionRank : transition_rank;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * rank);
    float* prev_prob = dest_shared + (rank * kNumStates);
    float* latent = prev_prob + kNumStates;
    float* grad_mix = latent + rank;
    float* dlatent = grad_mix + kNumStates;
    float* scratch = dlatent + rank;
    float* grad_source_shared = nullptr;
    float* grad_dest_shared = nullptr;
    float* grad_stay_shared = nullptr;
    if constexpr (DirectGradReduce) {
        grad_source_shared = scratch + kNumWarps;
        grad_dest_shared = grad_source_shared + (kNumStates * rank);
        grad_stay_shared = grad_dest_shared + (rank * kNumStates);
    }
    float* grad_source_batch = nullptr;
    float* grad_dest_batch = nullptr;
    float* grad_stay_batch = nullptr;

    if constexpr (InputsAreLogits) {
        for (int row = 0; row < kNumStates; ++row) {
            const float logit = s < rank ? transition_source_probs[row * rank + s] : -INFINITY;
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = s < rank ? fast_exp(logit - row_max) : 0.0f;
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            if (s < rank) {
                source_shared[row * rank + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
            }
        }
        for (int row = 0; row < rank; ++row) {
            const float logit = transition_dest_probs[row * kNumStates + s];
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = fast_exp(logit - row_max);
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            dest_shared[row * kNumStates + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
        }
    } else {
        for (int idx = s; idx < kNumStates * rank; idx += blockDim.x) {
            source_shared[idx] = transition_source_probs[idx];
        }
        for (int idx = s; idx < rank * kNumStates; idx += blockDim.x) {
            dest_shared[idx] = transition_dest_probs[idx];
        }
    }
    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;
    __syncthreads();

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        if constexpr (DirectGradReduce) {
            for (int idx = s; idx < kNumStates * rank; idx += blockDim.x) {
                grad_source_shared[idx] = 0.0f;
            }
            for (int idx = s; idx < rank * kNumStates; idx += blockDim.x) {
                grad_dest_shared[idx] = 0.0f;
            }
            if (s < kNumStates) {
                grad_stay_shared[s] = 0.0f;
            }
        }
        if constexpr (DirectGradReduce) {
            const int worker_slot = static_cast<int>(blockIdx.x);
            grad_source_batch = grad_transition_source_per_batch + worker_slot * kNumStates * rank;
            grad_dest_batch = grad_transition_dest_per_batch + worker_slot * rank * kNumStates;
            grad_stay_batch = grad_transition_stay_per_batch + worker_slot * kNumStates;
        } else {
            grad_source_batch = grad_transition_source_per_batch + b * kNumStates * rank;
            grad_dest_batch = grad_transition_dest_per_batch + b * rank * kNumStates;
            grad_stay_batch = grad_transition_stay_per_batch + b * kNumStates;
        }
        __syncthreads();

        float carry_value = load_as_float(grad_final_belief + (b * kNumStates + s));
        float gate_grad_accum = 0.0f;
        float q_prob_value = 0.0f;
        if (seq_len > 0) {
            const int last_pos = seq_len - 1;
            q_prob_value = fast_exp(load_as_float(beliefs + ((b * seq_len + last_pos) * kNumStates + s)));
        }

        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int base = (b * seq_len + pos) * kNumStates;

            float prev_prob_value;
            if (pos == 0) {
                prev_prob_value = fast_exp(load_as_float(initial_log_belief + (b * kNumStates + s)));
            } else {
                prev_prob_value = fast_exp(load_as_float(beliefs + ((b * seq_len + (pos - 1)) * kNumStates + s)));
            }
            prev_prob[s] = prev_prob_value;
            __syncthreads();

            if (s < rank) {
                float latent_val = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < kNumStates; ++i) {
                    latent_val += prev_prob[i] * source_shared[i * rank + s];
                }
                latent[s] = latent_val;
            }
            __syncthreads();

            float mix_prob = 0.0f;
            if constexpr (StaticTransitionRank > 0) {
                #pragma unroll
                for (int r = 0; r < StaticTransitionRank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
            } else {
                #pragma unroll 4
                for (int r = 0; r < rank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
            }
            const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
            const float pred_log = fast_log(pred_prob);
            const float transition_context_value = load_as_float(transition_context + (base + s));

            const float gq = load_as_float(grad_beliefs + (base + s)) + carry_value;
            const float gq_sum = block_reduce_sum_128(gq, scratch);
            const float ga = gq - q_prob_value * gq_sum;
            const float prior_value = pred_log + transition_context_value;
            const float clamped_prior = apply_score_clamp(
                prior_value,
                score_clamp_min,
                score_clamp_max);
            const float grad_prior = (transition_gate * ga) * score_clamp_grad(
                prior_value,
                score_clamp_min,
                score_clamp_max);
            const float grad_pred_log = grad_prior;
            const float grad_pred_prob = grad_pred_log / pred_prob;

            grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
            grad_transition_context[base + s] = store_from_float<scalar_t>(grad_prior);
            grad_mix[s] = grad_pred_prob * one_minus_stay;
            if constexpr (DirectGradReduce) {
                grad_stay_shared[s] += grad_pred_prob * (prev_prob_value - mix_prob);
            } else {
                grad_stay_batch[s] += grad_pred_prob * (prev_prob_value - mix_prob);
            }
            gate_grad_accum += ga * clamped_prior;
            const float direct_prev_grad_prob = grad_pred_prob * stay_prob;
            __syncthreads();

            if (s < rank) {
                float dlatent_val = 0.0f;
                #pragma unroll 4
                for (int j = 0; j < kNumStates; ++j) {
                    dlatent_val += grad_mix[j] * dest_shared[s * kNumStates + j];
                }
                dlatent[s] = dlatent_val;
                #pragma unroll 4
                for (int j = 0; j < kNumStates; ++j) {
                    if constexpr (DirectGradReduce) {
                        grad_dest_shared[s * kNumStates + j] += latent[s] * grad_mix[j];
                    } else {
                        grad_dest_batch[s * kNumStates + j] += latent[s] * grad_mix[j];
                    }
                }
            }
            __syncthreads();

            float prev_grad_prob = direct_prev_grad_prob;
            if constexpr (StaticTransitionRank > 0) {
                #pragma unroll
                for (int r = 0; r < StaticTransitionRank; ++r) {
                    prev_grad_prob += dlatent[r] * source_shared[s * rank + r];
                    if constexpr (DirectGradReduce) {
                        grad_source_shared[s * rank + r] += prev_prob_value * dlatent[r];
                    } else {
                        grad_source_batch[s * rank + r] += prev_prob_value * dlatent[r];
                    }
                }
            } else {
                #pragma unroll 4
                for (int r = 0; r < rank; ++r) {
                    prev_grad_prob += dlatent[r] * source_shared[s * rank + r];
                    if constexpr (DirectGradReduce) {
                        grad_source_shared[s * rank + r] += prev_prob_value * dlatent[r];
                    } else {
                        grad_source_batch[s * rank + r] += prev_prob_value * dlatent[r];
                    }
                }
            }
                carry_value = prev_grad_prob * prev_prob_value;
                q_prob_value = prev_prob_value;
                __syncthreads();
            }
        }

        if constexpr (InputsAreLogits) {
            for (int row = 0; row < kNumStates; ++row) {
                const int idx = row * rank + s;
                const float prob = s < rank ? source_shared[idx] : 0.0f;
                const float grad_prob = s < rank
                    ? (DirectGradReduce ? grad_source_shared[idx] : grad_source_batch[idx])
                    : 0.0f;
                const float row_dot = block_reduce_sum_128(grad_prob * prob, scratch);
                if (s < rank) {
                    const float grad_logit = (grad_prob - row_dot) * prob;
                    if constexpr (DirectGradReduce) {
                        grad_source_shared[idx] = grad_logit;
                    } else {
                        grad_source_batch[idx] = grad_logit;
                    }
                }
            }
            for (int row = 0; row < rank; ++row) {
                const int idx = row * kNumStates + s;
                const float prob = dest_shared[idx];
                const float grad_prob = DirectGradReduce ? grad_dest_shared[idx] : grad_dest_batch[idx];
                const float row_dot = block_reduce_sum_128(grad_prob * prob, scratch);
                const float grad_logit = (grad_prob - row_dot) * prob;
                if constexpr (DirectGradReduce) {
                    grad_dest_shared[idx] = grad_logit;
                } else {
                    grad_dest_batch[idx] = grad_logit;
                }
            }
            __syncthreads();
        }

        grad_initial_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(carry_value);
        const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
        if constexpr (DirectGradReduce) {
            for (int idx = s; idx < kNumStates * rank; idx += blockDim.x) {
                grad_source_batch[idx] += grad_source_shared[idx];
            }
            for (int idx = s; idx < rank * kNumStates; idx += blockDim.x) {
                grad_dest_batch[idx] += grad_dest_shared[idx];
            }
            if (s < kNumStates) {
                grad_stay_batch[s] += grad_stay_shared[s];
            }
            if (s == 0) {
                grad_transition_gate_per_batch[blockIdx.x] += gate_sum;
            }
        } else {
            if (s == 0) {
                grad_transition_gate_per_batch[b] += gate_sum;
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t, int StaticTransitionRank = -1, bool DirectGradReduce = false, bool InputsAreLogits = false>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_backward_composable_chunk_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const float* __restrict__ transition_source_probs,
    const float* __restrict__ transition_dest_probs,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    const float* __restrict__ transition_stay_probs,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_per_batch,
    float* __restrict__ grad_transition_dest_per_batch,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_stay_per_batch) {
    const int s = threadIdx.x;
    const int kNumStates = static_cast<int>(blockDim.x);
    const int kNumWarps = (kNumStates + kWarpSize - 1) / kWarpSize;
    const int rank = StaticTransitionRank > 0 ? StaticTransitionRank : transition_rank;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* source_shared = shared_mem;
    float* dest_shared = source_shared + (kNumStates * rank);
    float* prev_prob = dest_shared + (rank * kNumStates);
    float* latent = prev_prob + kNumStates;
    float* grad_mix = latent + rank;
    float* dlatent = grad_mix + kNumStates;
    float* scratch = dlatent + rank;
    float* grad_source_shared = nullptr;
    float* grad_dest_shared = nullptr;
    float* grad_stay_shared = nullptr;
    if constexpr (DirectGradReduce) {
        grad_source_shared = scratch + kNumWarps;
        grad_dest_shared = grad_source_shared + (kNumStates * rank);
        grad_stay_shared = grad_dest_shared + (rank * kNumStates);
    }
    float* grad_source_batch = nullptr;
    float* grad_dest_batch = nullptr;
    float* grad_stay_batch = nullptr;
    if constexpr (InputsAreLogits) {
        for (int row = 0; row < kNumStates; ++row) {
            const float logit = s < rank ? transition_source_probs[row * rank + s] : -INFINITY;
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = s < rank ? fast_exp(logit - row_max) : 0.0f;
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            if (s < rank) {
                source_shared[row * rank + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
            }
        }
        for (int row = 0; row < rank; ++row) {
            const float logit = transition_dest_probs[row * kNumStates + s];
            const float row_max = block_reduce_max_128(logit, scratch);
            const float row_exp = fast_exp(logit - row_max);
            const float row_sum = block_reduce_sum_128(row_exp, scratch);
            dest_shared[row * kNumStates + s] = row_exp / fmaxf(row_sum, 1.0e-20f);
        }
    } else {
        for (int idx = s; idx < kNumStates * rank; idx += blockDim.x) {
            source_shared[idx] = transition_source_probs[idx];
        }
        for (int idx = s; idx < rank * kNumStates; idx += blockDim.x) {
            dest_shared[idx] = transition_dest_probs[idx];
        }
    }
    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;
    __syncthreads();

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        if constexpr (DirectGradReduce) {
            for (int idx = s; idx < kNumStates * rank; idx += blockDim.x) {
                grad_source_shared[idx] = 0.0f;
            }
            for (int idx = s; idx < rank * kNumStates; idx += blockDim.x) {
                grad_dest_shared[idx] = 0.0f;
            }
            if (s < kNumStates) {
                grad_stay_shared[s] = 0.0f;
            }
        }
        if constexpr (DirectGradReduce) {
            const int worker_slot = static_cast<int>(blockIdx.x);
            grad_source_batch = grad_transition_source_per_batch + worker_slot * kNumStates * rank;
            grad_dest_batch = grad_transition_dest_per_batch + worker_slot * rank * kNumStates;
            grad_stay_batch = grad_transition_stay_per_batch + worker_slot * kNumStates;
        } else {
            grad_source_batch = grad_transition_source_per_batch + b * kNumStates * rank;
            grad_dest_batch = grad_transition_dest_per_batch + b * rank * kNumStates;
            grad_stay_batch = grad_transition_stay_per_batch + b * kNumStates;
        }
        __syncthreads();

        float carry_value = load_as_float(grad_final_belief + (b * kNumStates + s));
        float q_prob_value = 0.0f;
        if (seq_len > 0) {
            const int last_pos = seq_len - 1;
            q_prob_value = fast_exp(load_as_float(beliefs + ((b * seq_len + last_pos) * kNumStates + s)));
        }

        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int base = (b * seq_len + pos) * kNumStates;

            float prev_prob_value;
            if (pos == 0) {
                prev_prob_value = fast_exp(load_as_float(initial_log_belief + (b * kNumStates + s)));
            } else {
                prev_prob_value = fast_exp(load_as_float(beliefs + ((b * seq_len + (pos - 1)) * kNumStates + s)));
            }
            prev_prob[s] = prev_prob_value;
            __syncthreads();

            if (s < rank) {
                float latent_val = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < kNumStates; ++i) {
                    latent_val += prev_prob[i] * source_shared[i * rank + s];
                }
                latent[s] = latent_val;
            }
            __syncthreads();

            float mix_prob = 0.0f;
            if constexpr (StaticTransitionRank > 0) {
                #pragma unroll
                for (int r = 0; r < StaticTransitionRank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
            } else {
                #pragma unroll 4
                for (int r = 0; r < rank; ++r) {
                    mix_prob += latent[r] * dest_shared[r * kNumStates + s];
                }
            }
            const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);

            const float gq = load_as_float(grad_beliefs + (base + s)) + carry_value;
            const float gq_sum = block_reduce_sum_128(gq, scratch);
            const float ga = gq - q_prob_value * gq_sum;
            const float grad_pred_prob = ga / pred_prob;

            grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
            grad_transition_context[base + s] = store_from_float<scalar_t>(ga);
            grad_mix[s] = grad_pred_prob * one_minus_stay;
            if constexpr (DirectGradReduce) {
                grad_stay_shared[s] += grad_pred_prob * (prev_prob_value - mix_prob);
            } else {
                grad_stay_batch[s] += grad_pred_prob * (prev_prob_value - mix_prob);
            }
            const float direct_prev_grad_prob = grad_pred_prob * stay_prob;
            __syncthreads();

            if (s < rank) {
                float dlatent_val = 0.0f;
                #pragma unroll 4
                for (int j = 0; j < kNumStates; ++j) {
                    dlatent_val += grad_mix[j] * dest_shared[s * kNumStates + j];
                }
                dlatent[s] = dlatent_val;
                #pragma unroll 4
                for (int j = 0; j < kNumStates; ++j) {
                    if constexpr (DirectGradReduce) {
                        grad_dest_shared[s * kNumStates + j] += latent[s] * grad_mix[j];
                    } else {
                        grad_dest_batch[s * kNumStates + j] += latent[s] * grad_mix[j];
                    }
                }
            }
            __syncthreads();

            float prev_grad_prob = direct_prev_grad_prob;
            if constexpr (StaticTransitionRank > 0) {
                #pragma unroll
                for (int r = 0; r < StaticTransitionRank; ++r) {
                    prev_grad_prob += dlatent[r] * source_shared[s * rank + r];
                    if constexpr (DirectGradReduce) {
                        grad_source_shared[s * rank + r] += prev_prob_value * dlatent[r];
                    } else {
                        grad_source_batch[s * rank + r] += prev_prob_value * dlatent[r];
                    }
                }
            } else {
                #pragma unroll 4
                for (int r = 0; r < rank; ++r) {
                    prev_grad_prob += dlatent[r] * source_shared[s * rank + r];
                    if constexpr (DirectGradReduce) {
                        grad_source_shared[s * rank + r] += prev_prob_value * dlatent[r];
                    } else {
                        grad_source_batch[s * rank + r] += prev_prob_value * dlatent[r];
                    }
                }
            }
                carry_value = prev_grad_prob * prev_prob_value;
                q_prob_value = prev_prob_value;
                __syncthreads();
            }
        }

        if constexpr (InputsAreLogits) {
            for (int row = 0; row < kNumStates; ++row) {
                const int idx = row * rank + s;
                const float prob = s < rank ? source_shared[idx] : 0.0f;
                const float grad_prob = s < rank
                    ? (DirectGradReduce ? grad_source_shared[idx] : grad_source_batch[idx])
                    : 0.0f;
                const float row_dot = block_reduce_sum_128(grad_prob * prob, scratch);
                if (s < rank) {
                    const float grad_logit = (grad_prob - row_dot) * prob;
                    if constexpr (DirectGradReduce) {
                        grad_source_shared[idx] = grad_logit;
                    } else {
                        grad_source_batch[idx] = grad_logit;
                    }
                }
            }
            for (int row = 0; row < rank; ++row) {
                const int idx = row * kNumStates + s;
                const float prob = dest_shared[idx];
                const float grad_prob = DirectGradReduce ? grad_dest_shared[idx] : grad_dest_batch[idx];
                const float row_dot = block_reduce_sum_128(grad_prob * prob, scratch);
                const float grad_logit = (grad_prob - row_dot) * prob;
                if constexpr (DirectGradReduce) {
                    grad_dest_shared[idx] = grad_logit;
                } else {
                    grad_dest_batch[idx] = grad_logit;
                }
            }
            __syncthreads();
        }

        grad_initial_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(carry_value);
        if constexpr (DirectGradReduce) {
            for (int idx = s; idx < kNumStates * rank; idx += blockDim.x) {
                grad_source_batch[idx] += grad_source_shared[idx];
            }
            for (int idx = s; idx < rank * kNumStates; idx += blockDim.x) {
                grad_dest_batch[idx] += grad_dest_shared[idx];
            }
            if (s < kNumStates) {
                grad_stay_batch[s] += grad_stay_shared[s];
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename packed_t, PackedTransitionFormat Format, int StaticTransitionRank = -1, bool DirectGradReduce = false>
__global__ CAUSAL_MACHINE_SMALL_LAUNCH_BOUNDS void causal_machine_backward_chunk_packed_kernel(
    const scalar_t* __restrict__ grad_beliefs,
    const scalar_t* __restrict__ grad_final_belief,
    const packed_t* __restrict__ transition_source_packed,
    const float* __restrict__ transition_source_scales,
    const packed_t* __restrict__ transition_dest_packed,
    const float* __restrict__ transition_dest_scales,
    const scalar_t* __restrict__ transition_context,
    const scalar_t* __restrict__ initial_log_belief,
    const scalar_t* __restrict__ beliefs,
    float transition_gate,
    const float* __restrict__ transition_stay_probs,
    int transition_rank,
    int seq_len,
    int chunk_start,
    int chunk_len,
    int total_batches,
    int32_t* __restrict__ work_queue_counter,
    scalar_t* __restrict__ grad_local_logits,
    float* __restrict__ grad_transition_source_per_batch,
    float* __restrict__ grad_transition_dest_per_batch,
    scalar_t* __restrict__ grad_transition_context,
    scalar_t* __restrict__ grad_initial_log_belief,
    float* __restrict__ grad_transition_gate_per_batch,
    float* __restrict__ grad_transition_stay_per_batch) {
    const int s = threadIdx.x;
    const int kNumStates = static_cast<int>(blockDim.x);
    const int kNumWarps = (kNumStates + kWarpSize - 1) / kWarpSize;
    const int rank = StaticTransitionRank > 0 ? StaticTransitionRank : transition_rank;
    const int sequence_tile_size = max(chunk_len, 1);
    __shared__ int current_batch;

    extern __shared__ float shared_mem[];
    float* prev_prob = shared_mem;
    float* latent = prev_prob + kNumStates;
    float* grad_mix = latent + rank;
    float* dlatent = grad_mix + kNumStates;
    float* scratch = dlatent + rank;
    float* grad_source_shared = nullptr;
    float* grad_dest_shared = nullptr;
    float* grad_stay_shared = nullptr;
    float* tensor_core_base = scratch + kNumWarps;
    if constexpr (DirectGradReduce) {
        grad_source_shared = tensor_core_base;
        grad_dest_shared = grad_source_shared + (kNumStates * rank);
        grad_stay_shared = grad_dest_shared + (rank * kNumStates);
        tensor_core_base = grad_stay_shared + kNumStates;
    }
    char* tensor_core_bytes = reinterpret_cast<char*>(tensor_core_base);
    auto tensor_core_addr = reinterpret_cast<std::uintptr_t>(tensor_core_bytes);
    tensor_core_addr = (tensor_core_addr + 15u) & ~static_cast<std::uintptr_t>(15u);
    using tensor_core_input_t = tensor_core_input_type_t<scalar_t>;
    tensor_core_input_t* tensor_core_lhs = reinterpret_cast<tensor_core_input_t*>(tensor_core_addr);
    tensor_core_input_t* tensor_core_rhs = tensor_core_lhs + (kTensorCoreTile * kTensorCoreTile);
    float* tensor_core_accum = reinterpret_cast<float*>(tensor_core_rhs + (kTensorCoreTile * kTensorCoreTile));
    float* tensor_core_matrix = tensor_core_accum + (kTensorCoreTile * kTensorCoreTile);
    float* grad_source_batch = nullptr;
    float* grad_dest_batch = nullptr;
    float* grad_stay_batch = nullptr;
    const float stay_prob = transition_stay_probs[s];
    const float one_minus_stay = 1.0f - stay_prob;
    const bool use_tensor_core_math =
        tensor_core_math_enabled_for_scalar<scalar_t>()
        && (kNumStates >= kTensorCoreTile)
        && (rank >= kTensorCoreTile);
    __syncthreads();

    while (true) {
        if (s == 0) {
            current_batch = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();
        if (current_batch >= total_batches) {
            break;
        }
        const int b = current_batch;
        if constexpr (DirectGradReduce) {
            for (int idx = s; idx < kNumStates * rank; idx += blockDim.x) {
                grad_source_shared[idx] = 0.0f;
            }
            for (int idx = s; idx < rank * kNumStates; idx += blockDim.x) {
                grad_dest_shared[idx] = 0.0f;
            }
            if (s < kNumStates) {
                grad_stay_shared[s] = 0.0f;
            }
        }
        if constexpr (DirectGradReduce) {
            const int worker_slot = static_cast<int>(blockIdx.x);
            grad_source_batch = grad_transition_source_per_batch + worker_slot * kNumStates * rank;
            grad_dest_batch = grad_transition_dest_per_batch + worker_slot * rank * kNumStates;
            grad_stay_batch = grad_transition_stay_per_batch + worker_slot * kNumStates;
        } else {
            grad_source_batch = grad_transition_source_per_batch + b * kNumStates * rank;
            grad_dest_batch = grad_transition_dest_per_batch + b * rank * kNumStates;
            grad_stay_batch = grad_transition_stay_per_batch + b * kNumStates;
        }
        __syncthreads();

        float carry_value = load_as_float(grad_final_belief + (b * kNumStates + s));
        float gate_grad_accum = 0.0f;
        float q_prob_value = 0.0f;
        if (seq_len > 0) {
            const int last_pos = seq_len - 1;
            q_prob_value = fast_exp(load_as_float(beliefs + ((b * seq_len + last_pos) * kNumStates + s)));
        }

        for (int tile_start = ((seq_len - 1) / sequence_tile_size) * sequence_tile_size;
             tile_start >= chunk_start;
             tile_start -= sequence_tile_size) {
            const int current_chunk_len = min(sequence_tile_size, seq_len - tile_start);
            for (int t = current_chunk_len - 1; t >= 0; --t) {
                const int pos = tile_start + t;
                const int base = (b * seq_len + pos) * kNumStates;

                float prev_prob_value;
                if (pos == 0) {
                    prev_prob_value = fast_exp(load_as_float(initial_log_belief + (b * kNumStates + s)));
                } else {
                    prev_prob_value = fast_exp(load_as_float(beliefs + ((b * seq_len + (pos - 1)) * kNumStates + s)));
                }
                prev_prob[s] = prev_prob_value;
                __syncthreads();

                if (use_tensor_core_math) {
                    for (int r = s; r < rank; r += blockDim.x) {
                        latent[r] = 0.0f;
                    }
                    __syncthreads();
                    for (int rank_start = 0; rank_start < rank; rank_start += kTensorCoreTile) {
                        const int active_rank = min(kTensorCoreTile, rank - rank_start);
                        if (active_rank == kTensorCoreTile) {
                            for (int src_start = 0; src_start < kNumStates; src_start += kTensorCoreTile) {
                                const int active_src = min(kTensorCoreTile, kNumStates - src_start);
#if __CUDA_ARCH__ >= 700
                                if (active_src == kTensorCoreTile) {
                                    load_packed_matrix_tile_rowmajor_16x16<packed_t, Format>(
                                        transition_source_packed,
                                        transition_source_scales,
                                        rank,
                                        src_start,
                                        rank_start,
                                        active_src,
                                        active_rank,
                                        tensor_core_matrix);
                                    wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                        prev_prob + src_start,
                                        tensor_core_matrix,
                                        kTensorCoreTile,
                                        tensor_core_lhs,
                                        tensor_core_rhs,
                                        tensor_core_accum,
                                        latent + rank_start);
                                } else
#endif
                                {
                                    for (int r = s; r < active_rank; r += blockDim.x) {
                                        latent[rank_start + r] += packed_column_dot_lowp<packed_t, Format>(
                                            prev_prob + src_start,
                                            transition_source_packed + static_cast<int64_t>(src_start) * rank + rank_start + r,
                                            transition_source_scales + src_start,
                                            active_src,
                                            rank,
                                            0);
                                    }
                                    __syncthreads();
                                }
                            }
                        } else {
                            for (int r = s; r < active_rank; r += blockDim.x) {
                                latent[rank_start + r] = packed_column_dot_lowp<packed_t, Format>(
                                    prev_prob,
                                    transition_source_packed + rank_start + r,
                                    transition_source_scales,
                                    kNumStates,
                                    rank,
                                    0);
                            }
                            __syncthreads();
                        }
                    }
                    for (int dst = s; dst < kNumStates; dst += blockDim.x) {
                        grad_mix[dst] = 0.0f;
                    }
                    __syncthreads();
                    for (int rank_start = 0; rank_start < rank; rank_start += kTensorCoreTile) {
                        const int active_rank = min(kTensorCoreTile, rank - rank_start);
                        if (active_rank == kTensorCoreTile) {
                            for (int dst_start = 0; dst_start < kNumStates; dst_start += kTensorCoreTile) {
                                const int active_dst = min(kTensorCoreTile, kNumStates - dst_start);
#if __CUDA_ARCH__ >= 700
                                if (active_dst == kTensorCoreTile) {
                                    load_packed_matrix_tile_rowmajor_16x16<packed_t, Format>(
                                        transition_dest_packed,
                                        transition_dest_scales,
                                        kNumStates,
                                        rank_start,
                                        dst_start,
                                        active_rank,
                                        active_dst,
                                        tensor_core_matrix);
                                    wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                        latent + rank_start,
                                        tensor_core_matrix,
                                        kTensorCoreTile,
                                        tensor_core_lhs,
                                        tensor_core_rhs,
                                        tensor_core_accum,
                                        grad_mix + dst_start);
                                } else
#endif
                                {
                                    for (int dst_local = s; dst_local < active_dst; dst_local += blockDim.x) {
                                        float mix_value = grad_mix[dst_start + dst_local];
                                        for (int r = 0; r < active_rank; ++r) {
                                            mix_value += latent[rank_start + r] * unpack_packed_value<packed_t, Format>(
                                                transition_dest_packed[static_cast<int64_t>(rank_start + r) * kNumStates + dst_start + dst_local],
                                                transition_dest_scales[rank_start + r]);
                                        }
                                        grad_mix[dst_start + dst_local] = mix_value;
                                    }
                                    __syncthreads();
                                }
                            }
                        } else {
                            for (int dst = s; dst < kNumStates; dst += blockDim.x) {
                                float mix_value = grad_mix[dst];
                                for (int r = 0; r < active_rank; ++r) {
                                    mix_value += latent[rank_start + r] * unpack_packed_value<packed_t, Format>(
                                        transition_dest_packed[static_cast<int64_t>(rank_start + r) * kNumStates + dst],
                                        transition_dest_scales[rank_start + r]);
                                }
                                grad_mix[dst] = mix_value;
                            }
                            __syncthreads();
                        }
                    }
                } else {
                    if (s < rank) {
                        latent[s] = packed_column_dot_lowp<packed_t, Format>(
                            prev_prob,
                            transition_source_packed,
                            transition_source_scales,
                            kNumStates,
                            rank,
                            s);
                    }
                    __syncthreads();
                    grad_mix[s] = packed_column_dot_lowp<packed_t, Format>(
                        latent,
                        transition_dest_packed,
                        transition_dest_scales,
                        rank,
                        kNumStates,
                        s);
                }
                __syncthreads();

                const float mix_prob = grad_mix[s];
                const float pred_prob = fmaxf(stay_prob * prev_prob_value + one_minus_stay * mix_prob, 1.0e-20f);
                const float pred_log = fast_log(pred_prob);
                const float transition_context_value = load_as_float(transition_context + (base + s));

                const float gq = load_as_float(grad_beliefs + (base + s)) + carry_value;
                const float gq_sum = block_reduce_sum_128(gq, scratch);
                const float ga = gq - q_prob_value * gq_sum;
                const float grad_pred_log = transition_gate * ga;
                const float grad_pred_prob = grad_pred_log / pred_prob;

                grad_local_logits[base + s] = store_from_float<scalar_t>(ga);
                grad_transition_context[base + s] = store_from_float<scalar_t>(transition_gate * ga);
                grad_mix[s] = grad_pred_prob * one_minus_stay;
                if constexpr (DirectGradReduce) {
                    grad_stay_shared[s] += grad_pred_prob * (prev_prob_value - mix_prob);
                } else {
                    grad_stay_batch[s] += grad_pred_prob * (prev_prob_value - mix_prob);
                }
                gate_grad_accum += ga * (pred_log + transition_context_value);
                const float direct_prev_grad_prob = grad_pred_prob * stay_prob;
                __syncthreads();

                if (use_tensor_core_math) {
                    for (int r = s; r < rank; r += blockDim.x) {
                        dlatent[r] = 0.0f;
                    }
                    __syncthreads();
                    for (int rank_start = 0; rank_start < rank; rank_start += kTensorCoreTile) {
                        const int active_rank = min(kTensorCoreTile, rank - rank_start);
                        if (active_rank == kTensorCoreTile) {
                            for (int dst_start = 0; dst_start < kNumStates; dst_start += kTensorCoreTile) {
                                const int active_dst = min(kTensorCoreTile, kNumStates - dst_start);
#if __CUDA_ARCH__ >= 700
                                if (active_dst == kTensorCoreTile) {
                                    load_packed_matrix_transposed_tile_rowmajor_16x16<packed_t, Format>(
                                        transition_dest_packed,
                                        transition_dest_scales,
                                        kNumStates,
                                        rank_start,
                                        dst_start,
                                        active_rank,
                                        active_dst,
                                        tensor_core_matrix);
                                    wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                        grad_mix + dst_start,
                                        tensor_core_matrix,
                                        kTensorCoreTile,
                                        tensor_core_lhs,
                                        tensor_core_rhs,
                                        tensor_core_accum,
                                        dlatent + rank_start);
                                } else
#endif
                                {
                                    for (int r = s; r < active_rank; r += blockDim.x) {
                                        float dlatent_value = dlatent[rank_start + r];
                                        for (int dst_local = 0; dst_local < active_dst; ++dst_local) {
                                            dlatent_value += grad_mix[dst_start + dst_local] * unpack_packed_value<packed_t, Format>(
                                                transition_dest_packed[static_cast<int64_t>(rank_start + r) * kNumStates + dst_start + dst_local],
                                                transition_dest_scales[rank_start + r]);
                                        }
                                        dlatent[rank_start + r] = dlatent_value;
                                    }
                                    __syncthreads();
                                }
                                for (int r = s; r < active_rank; r += blockDim.x) {
                                    const int r_idx = rank_start + r;
                                    for (int dst_local = 0; dst_local < active_dst; ++dst_local) {
                                        const int dst = dst_start + dst_local;
                                        if constexpr (DirectGradReduce) {
                                            grad_dest_shared[r_idx * kNumStates + dst] += latent[r_idx] * grad_mix[dst];
                                        } else {
                                            grad_dest_batch[r_idx * kNumStates + dst] += latent[r_idx] * grad_mix[dst];
                                        }
                                    }
                                }
                                __syncthreads();
                            }
                        } else {
                            for (int r = s; r < active_rank; r += blockDim.x) {
                                const int r_idx = rank_start + r;
                                dlatent[r_idx] = packed_row_dot_lowp<packed_t, Format>(
                                    grad_mix,
                                    transition_dest_packed + static_cast<int64_t>(r_idx) * kNumStates,
                                    transition_dest_scales[r_idx],
                                    kNumStates);
                                for (int j = 0; j < kNumStates; ++j) {
                                    if constexpr (DirectGradReduce) {
                                        grad_dest_shared[r_idx * kNumStates + j] += latent[r_idx] * grad_mix[j];
                                    } else {
                                        grad_dest_batch[r_idx * kNumStates + j] += latent[r_idx] * grad_mix[j];
                                    }
                                }
                            }
                            __syncthreads();
                        }
                    }
                    for (int src = s; src < kNumStates; src += blockDim.x) {
                        grad_mix[src] = 0.0f;
                    }
                    __syncthreads();
                    for (int rank_start = 0; rank_start < rank; rank_start += kTensorCoreTile) {
                        const int active_rank = min(kTensorCoreTile, rank - rank_start);
                        if (active_rank == kTensorCoreTile) {
                            for (int src_start = 0; src_start < kNumStates; src_start += kTensorCoreTile) {
                                const int active_src = min(kTensorCoreTile, kNumStates - src_start);
#if __CUDA_ARCH__ >= 700
                                if (active_src == kTensorCoreTile) {
                                    load_packed_matrix_transposed_tile_rowmajor_16x16<packed_t, Format>(
                                        transition_source_packed,
                                        transition_source_scales,
                                        rank,
                                        src_start,
                                        rank_start,
                                        active_src,
                                        active_rank,
                                        tensor_core_matrix);
                                    wmma_replicated_row_times_matrix_16x16<scalar_t>(
                                        dlatent + rank_start,
                                        tensor_core_matrix,
                                        kTensorCoreTile,
                                        tensor_core_lhs,
                                        tensor_core_rhs,
                                        tensor_core_accum,
                                        grad_mix + src_start);
                                } else
#endif
                                {
                                    for (int src_local = s; src_local < active_src; src_local += blockDim.x) {
                                        float prev_grad_value = grad_mix[src_start + src_local];
                                        for (int r = 0; r < active_rank; ++r) {
                                            prev_grad_value += dlatent[rank_start + r] * unpack_packed_value<packed_t, Format>(
                                                transition_source_packed[static_cast<int64_t>(src_start + src_local) * rank + rank_start + r],
                                                transition_source_scales[src_start + src_local]);
                                        }
                                        grad_mix[src_start + src_local] = prev_grad_value;
                                    }
                                    __syncthreads();
                                }
                            }
                        } else {
                            for (int src = s; src < kNumStates; src += blockDim.x) {
                                float prev_grad_value = grad_mix[src];
                                for (int r = 0; r < active_rank; ++r) {
                                    prev_grad_value += dlatent[rank_start + r] * unpack_packed_value<packed_t, Format>(
                                        transition_source_packed[static_cast<int64_t>(src) * rank + rank_start + r],
                                        transition_source_scales[src]);
                                }
                                grad_mix[src] = prev_grad_value;
                            }
                            __syncthreads();
                        }
                    }
                } else {
                    if (s < rank) {
                        dlatent[s] = packed_row_dot_lowp<packed_t, Format>(
                            grad_mix,
                            transition_dest_packed + s * kNumStates,
                            transition_dest_scales[s],
                            kNumStates);
                        #pragma unroll 4
                        for (int j = 0; j < kNumStates; ++j) {
                            if constexpr (DirectGradReduce) {
                                grad_dest_shared[s * kNumStates + j] += latent[s] * grad_mix[j];
                            } else {
                                grad_dest_batch[s * kNumStates + j] += latent[s] * grad_mix[j];
                            }
                        }
                    }
                    __syncthreads();
                    grad_mix[s] = packed_row_dot_lowp<packed_t, Format>(
                        dlatent,
                        transition_source_packed + s * rank,
                        transition_source_scales[s],
                        rank);
                }
                __syncthreads();

                float prev_grad_prob = direct_prev_grad_prob + grad_mix[s];
                if constexpr (StaticTransitionRank > 0) {
                    #pragma unroll
                    for (int r = 0; r < StaticTransitionRank; ++r) {
                        if constexpr (DirectGradReduce) {
                            grad_source_shared[s * rank + r] += prev_prob_value * dlatent[r];
                        } else {
                            grad_source_batch[s * rank + r] += prev_prob_value * dlatent[r];
                        }
                    }
                } else {
                    #pragma unroll 4
                    for (int r = 0; r < rank; ++r) {
                        if constexpr (DirectGradReduce) {
                            grad_source_shared[s * rank + r] += prev_prob_value * dlatent[r];
                        } else {
                            grad_source_batch[s * rank + r] += prev_prob_value * dlatent[r];
                        }
                    }
                }
                carry_value = prev_grad_prob * prev_prob_value;
                q_prob_value = prev_prob_value;
                __syncthreads();
            }
        }

        grad_initial_log_belief[b * kNumStates + s] = store_from_float<scalar_t>(carry_value);
        const float gate_sum = block_reduce_sum_128(gate_grad_accum, scratch);
        if constexpr (DirectGradReduce) {
            for (int idx = s; idx < kNumStates * rank; idx += blockDim.x) {
                grad_source_batch[idx] += grad_source_shared[idx];
            }
            for (int idx = s; idx < rank * kNumStates; idx += blockDim.x) {
                grad_dest_batch[idx] += grad_dest_shared[idx];
            }
            if (s < kNumStates) {
                grad_stay_batch[s] += grad_stay_shared[s];
            }
            if (s == 0) {
                grad_transition_gate_per_batch[blockIdx.x] += gate_sum;
            }
        } else {
            if (s == 0) {
                grad_transition_gate_per_batch[b] += gate_sum;
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t, bool StoreBeliefs = true>
void launch_forward_chunk_dense_128_rank8(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    double score_clamp_min,
    double score_clamp_max,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    constexpr int kVectorizedBlockThreads = 64;
    const bool use_pair_path = can_use_dense_128_rank8_pair_path<scalar_t>(local_logits.get_device());
    ScanKernelLaunchConfig launch_config{
        dim3(1),
        dim3(static_cast<unsigned int>(use_pair_path ? kVectorizedBlockThreads : 128)),
        8,
        forward_dense_128_rank8_shared_bytes(),
        local_logits.get_device(),
        false,
        static_cast<int>(local_logits.size(0)),
    };
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (use_pair_path) {
        finalize_persistent_launch_config(
            "causal_machine_scan forward dense_128_rank8",
            launch_config,
            causal_machine_forward_chunk_dense_128_rank8_pair_kernel<scalar_t, StoreBeliefs>);
    } else {
        finalize_persistent_launch_config(
            "causal_machine_scan forward dense_128_rank8",
            launch_config,
            causal_machine_forward_chunk_dense_128_rank8_kernel<scalar_t, StoreBeliefs>);
    }
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    if (use_pair_path) {
        causal_machine_forward_chunk_dense_128_rank8_pair_kernel<scalar_t, StoreBeliefs><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
            local_logits.data_ptr<scalar_t>(),
            transition_source_probs.data_ptr<float>(),
            transition_dest_probs.data_ptr<float>(),
            transition_context.data_ptr<scalar_t>(),
            initial_log_belief.data_ptr<scalar_t>(),
            static_cast<float>(transition_gate),
            transition_stay_probs.data_ptr<float>(),
            static_cast<float>(score_clamp_min),
            static_cast<float>(score_clamp_max),
            static_cast<int>(local_logits.size(1)),
            static_cast<int>(chunk_start),
            static_cast<int>(chunk_len),
            static_cast<int>(local_logits.size(0)),
            work_queue_counter.data_ptr<int32_t>(),
            beliefs.data_ptr<scalar_t>(),
            final_log_belief.data_ptr<scalar_t>());
    } else {
        causal_machine_forward_chunk_dense_128_rank8_kernel<scalar_t, StoreBeliefs><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
            local_logits.data_ptr<scalar_t>(),
            transition_source_probs.data_ptr<float>(),
            transition_dest_probs.data_ptr<float>(),
            transition_context.data_ptr<scalar_t>(),
            initial_log_belief.data_ptr<scalar_t>(),
            static_cast<float>(transition_gate),
            transition_stay_probs.data_ptr<float>(),
            static_cast<float>(score_clamp_min),
            static_cast<float>(score_clamp_max),
            static_cast<int>(local_logits.size(1)),
            static_cast<int>(chunk_start),
            static_cast<int>(chunk_len),
            static_cast<int>(local_logits.size(0)),
            work_queue_counter.data_ptr<int32_t>(),
            beliefs.data_ptr<scalar_t>(),
            final_log_belief.data_ptr<scalar_t>());
    }
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int StaticTransitionRank = -1, bool InputsAreLogits = false>
void launch_forward_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    double score_clamp_min,
    double score_clamp_max,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_probs.size(1));
    auto launch_config = make_forward_launch_config(local_logits, transition_rank);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan forward",
        launch_config,
        causal_machine_forward_chunk_kernel<scalar_t, StaticTransitionRank, InputsAreLogits>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    causal_machine_forward_chunk_kernel<scalar_t, StaticTransitionRank, InputsAreLogits><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        local_logits.data_ptr<scalar_t>(),
        transition_source_probs.data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        static_cast<float>(score_clamp_min),
        static_cast<float>(score_clamp_max),
        launch_config.transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        beliefs.data_ptr<scalar_t>(),
        final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int StaticTransitionRank = -1, bool InputsAreLogits = false>
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
    launch_forward_chunk<scalar_t, StaticTransitionRank, InputsAreLogits>(
        local_logits,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        chunk_start,
        chunk_len,
        beliefs,
        final_log_belief);
}

template <typename scalar_t>
void launch_forward_masked_dense_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& transition_mask,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    const int transition_rank = static_cast<int>(transition_source_logits.size(1));
    auto launch_config = make_forward_masked_launch_config(local_logits);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan masked dense forward",
        launch_config,
        causal_machine_forward_masked_dense_chunk_kernel<scalar_t>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_logits, &transition_dest_logits});
    causal_machine_forward_masked_dense_chunk_kernel<scalar_t><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        local_logits.data_ptr<scalar_t>(),
        transition_source_logits.data_ptr<float>(),
        transition_dest_logits.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        transition_mask.data_ptr<bool>(),
        seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
        static_cast<float>(score_clamp_min),
        static_cast<float>(score_clamp_max),
        transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        beliefs.data_ptr<scalar_t>(),
        final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_forward_masked_dense_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& transition_mask,
    const torch::Tensor& seq_lens,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    launch_forward_masked_dense_chunk<scalar_t>(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        transition_mask,
        seq_lens,
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        chunk_start,
        chunk_len,
        beliefs,
        final_log_belief);
}

void prepare_masked_dense_transition(
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& transition_mask,
    const torch::Tensor& transition_matrix,
    const torch::Tensor& row_sums) {
    const int num_states = static_cast<int>(transition_mask.size(0));
    const int transition_rank = static_cast<int>(transition_source_logits.size(1));
    const dim3 grid(static_cast<unsigned int>(num_states));
    const dim3 block(static_cast<unsigned int>(num_states));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    masked_dense_transition_prepare_kernel<<<grid, block, 0, stream>>>(
        transition_source_logits.data_ptr<float>(),
        transition_dest_logits.data_ptr<float>(),
        transition_mask.data_ptr<bool>(),
        num_states,
        transition_rank,
        transition_matrix.data_ptr<float>(),
        row_sums.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepare_masked_transition_row_sums(
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& transition_mask,
    const torch::Tensor& row_sums) {
    const int num_states = static_cast<int>(transition_mask.size(0));
    const int transition_rank = static_cast<int>(transition_source_logits.size(1));
    constexpr int kRowSumThreads = 256;
    const int blocks = std::max(1, (num_states + kRowSumThreads - 1) / kRowSumThreads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    masked_transition_row_sums_kernel<<<blocks, kRowSumThreads, 0, stream>>>(
        transition_source_logits.data_ptr<float>(),
        transition_dest_logits.data_ptr<float>(),
        transition_mask.data_ptr<bool>(),
        num_states,
        transition_rank,
        row_sums.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_masked_dense_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& transition_mask,
    const torch::Tensor& transition_matrix,
    const torch::Tensor& row_sums,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_per_batch,
    const torch::Tensor& grad_transition_dest_per_batch,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_per_batch,
    const torch::Tensor& grad_transition_stay_per_batch) {
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const int transition_rank = static_cast<int>(transition_source_logits.size(1));
    auto launch_config = make_backward_masked_launch_config(beliefs);
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan masked dense backward",
        launch_config,
        causal_machine_backward_masked_dense_chunk_kernel<scalar_t>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_logits, &transition_dest_logits});
    causal_machine_backward_masked_dense_chunk_kernel<scalar_t><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        grad_beliefs.data_ptr<scalar_t>(),
        grad_final_belief.data_ptr<scalar_t>(),
        transition_source_logits.data_ptr<float>(),
        transition_dest_logits.data_ptr<float>(),
        transition_mask.data_ptr<bool>(),
        transition_matrix.data_ptr<float>(),
        row_sums.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        beliefs.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
        static_cast<float>(score_clamp_min),
        static_cast<float>(score_clamp_max),
        transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        grad_local_logits.data_ptr<scalar_t>(),
        grad_transition_source_per_batch.data_ptr<float>(),
        grad_transition_dest_per_batch.data_ptr<float>(),
        grad_transition_context.data_ptr<scalar_t>(),
        grad_initial_log_belief.data_ptr<scalar_t>(),
        grad_transition_gate_per_batch.data_ptr<float>(),
        grad_transition_stay_per_batch.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_masked_dense_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& transition_mask,
    const torch::Tensor& transition_matrix,
    const torch::Tensor& row_sums,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_per_batch,
    const torch::Tensor& grad_transition_dest_per_batch,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_per_batch,
    const torch::Tensor& grad_transition_stay_per_batch) {
    launch_backward_masked_dense_chunk<scalar_t>(
        grad_beliefs,
        grad_final_belief,
        transition_source_logits,
        transition_dest_logits,
        transition_mask,
        transition_matrix,
        row_sums,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate,
        transition_stay_probs,
        seq_lens,
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        chunk_start,
        chunk_len,
        grad_local_logits,
        grad_transition_source_per_batch,
        grad_transition_dest_per_batch,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate_per_batch,
        grad_transition_stay_per_batch);
}

template <typename scalar_t, int StaticTransitionRank = -1>
void launch_forward_chunk_quantized(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_q,
    const torch::Tensor& transition_source_scales,
    const torch::Tensor& transition_dest_q,
    const torch::Tensor& transition_dest_scales,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_q.size(1));
    auto launch_config = make_forward_packed_launch_config(local_logits, transition_rank);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan quantized forward",
        launch_config,
        causal_machine_forward_chunk_packed_kernel<scalar_t, int8_t, PackedTransitionFormat::Int8, StaticTransitionRank>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_q, &transition_dest_q});
    causal_machine_forward_chunk_packed_kernel<scalar_t, int8_t, PackedTransitionFormat::Int8, StaticTransitionRank><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        local_logits.data_ptr<scalar_t>(),
        transition_source_q.data_ptr<int8_t>(),
        transition_source_scales.data_ptr<float>(),
        transition_dest_q.data_ptr<int8_t>(),
        transition_dest_scales.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        launch_config.transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        beliefs.data_ptr<scalar_t>(),
        final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <PackedTransitionFormat Format, typename scalar_t, int StaticTransitionRank = -1>
void launch_forward_chunk_fp8(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_packed,
    const torch::Tensor& transition_source_scales,
    const torch::Tensor& transition_dest_packed,
    const torch::Tensor& transition_dest_scales,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    static_assert(
        Format == PackedTransitionFormat::Fp8E4M3 || Format == PackedTransitionFormat::Fp8E5M2,
        "launch_forward_chunk_fp8 only supports FP8 packed formats");
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_packed.size(1));
    auto launch_config = make_forward_packed_launch_config(local_logits, transition_rank);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan fp8 forward",
        launch_config,
        causal_machine_forward_chunk_packed_kernel<scalar_t, uint8_t, Format, StaticTransitionRank>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_packed, &transition_dest_packed});
    causal_machine_forward_chunk_packed_kernel<scalar_t, uint8_t, Format, StaticTransitionRank><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        local_logits.data_ptr<scalar_t>(),
        transition_source_packed.data_ptr<uint8_t>(),
        transition_source_scales.data_ptr<float>(),
        transition_dest_packed.data_ptr<uint8_t>(),
        transition_dest_scales.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        launch_config.transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        beliefs.data_ptr<scalar_t>(),
        final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int StaticTransitionRank = -1, bool InputsAreLogits = false>
void launch_forward_composable_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& transition_stay_probs,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_probs.size(1));
    auto launch_config = make_forward_launch_config(local_logits, transition_rank);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan composable forward",
        launch_config,
        causal_machine_forward_composable_chunk_kernel<scalar_t, StaticTransitionRank, InputsAreLogits>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    causal_machine_forward_composable_chunk_kernel<scalar_t, StaticTransitionRank, InputsAreLogits><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        local_logits.data_ptr<scalar_t>(),
        transition_source_probs.data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        transition_stay_probs.data_ptr<float>(),
        launch_config.transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        beliefs.data_ptr<scalar_t>(),
        final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int StaticTransitionRank = -1, bool InputsAreLogits = false>
void launch_forward_composable_chunk_summary(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& transition_stay_probs,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& final_log_belief) {
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_probs.size(1));
    auto launch_config = make_forward_launch_config(local_logits, transition_rank);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan composable forward summary",
        launch_config,
        causal_machine_forward_composable_chunk_summary_kernel<scalar_t, StaticTransitionRank, InputsAreLogits>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    causal_machine_forward_composable_chunk_summary_kernel<scalar_t, StaticTransitionRank, InputsAreLogits><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        local_logits.data_ptr<scalar_t>(),
        transition_source_probs.data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        transition_stay_probs.data_ptr<float>(),
        launch_config.transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int StaticTransitionRank = -1, bool InputsAreLogits = false>
void launch_forward_composable_chunk_finalize(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& chunk_initial_log_belief,
    const torch::Tensor& transition_stay_probs,
    int64_t num_chunks,
    int64_t chunk_size,
    const torch::Tensor& beliefs) {
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_probs.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    const int total_tasks = static_cast<int>(total_batches * num_chunks);
    auto launch_config = make_forward_launch_config(local_logits, transition_rank);
    launch_config.total_tasks = total_tasks;
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan composable forward finalize",
        launch_config,
        causal_machine_forward_composable_chunk_finalize_kernel<scalar_t, StaticTransitionRank, InputsAreLogits>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    causal_machine_forward_composable_chunk_finalize_kernel<scalar_t, StaticTransitionRank, InputsAreLogits><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        local_logits.data_ptr<scalar_t>(),
        transition_source_probs.data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        chunk_initial_log_belief.data_ptr<scalar_t>(),
        transition_stay_probs.data_ptr<float>(),
        launch_config.transition_rank,
        static_cast<int>(local_logits.size(1)),
        total_batches,
        static_cast<int>(num_chunks),
        static_cast<int>(chunk_size),
        total_tasks,
        work_queue_counter.data_ptr<int32_t>(),
        beliefs.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_forward_sparse_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_blocks,
    const torch::Tensor& block_row_ptr,
    const torch::Tensor& block_col_idx,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int num_states = static_cast<int>(local_logits.size(2));
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    auto launch_config = make_forward_sparse_launch_config(local_logits, num_states);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan_forward_sparse_chunk",
        launch_config,
        causal_machine_forward_sparse_chunk_kernel<scalar_t>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_blocks});
    causal_machine_forward_sparse_chunk_kernel<scalar_t><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        local_logits.data_ptr<scalar_t>(),
        transition_blocks.data_ptr<float>(),
        block_row_ptr.data_ptr<int32_t>(),
        block_col_idx.data_ptr<int32_t>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
        num_states,
        static_cast<int>(block_size),
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        beliefs.data_ptr<scalar_t>(),
        final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_forward_sparse_factor_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& row_sums,
    const torch::Tensor& block_row_ptr,
    const torch::Tensor& block_col_idx,
    const torch::Tensor& block_dst_idx,
    const torch::Tensor& src_row_ptr,
    const torch::Tensor& src_nz_idx,
    const torch::Tensor& block_mask,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int num_states = static_cast<int>(local_logits.size(2));
    const int padded_states = static_cast<int>(row_sums.size(0));
    const int transition_rank = static_cast<int>(transition_source_probs.size(1));
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    auto launch_config = make_forward_sparse_launch_config(local_logits, num_states);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan_forward_sparse_factor_chunk",
        launch_config,
        causal_machine_forward_sparse_factor_chunk_kernel<scalar_t, false>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs, &row_sums});
    causal_machine_forward_sparse_factor_chunk_kernel<scalar_t, false><<<
        launch_config.grid,
        launch_config.block,
        launch_config.shared_bytes,
        stream>>>(
            local_logits.data_ptr<scalar_t>(),
            transition_source_probs.data_ptr<float>(),
            transition_dest_probs.data_ptr<float>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            row_sums.data_ptr<float>(),
            block_row_ptr.data_ptr<int32_t>(),
            block_col_idx.data_ptr<int32_t>(),
            block_dst_idx.data_ptr<int32_t>(),
            nullptr,
            nullptr,
            block_mask.data_ptr<float>(),
            transition_context.data_ptr<scalar_t>(),
            initial_log_belief.data_ptr<scalar_t>(),
            static_cast<float>(transition_gate),
            transition_stay_probs.data_ptr<float>(),
            seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
            num_states,
            padded_states,
            transition_rank,
            static_cast<int>(block_size),
            seq_len,
            static_cast<int>(chunk_start),
            static_cast<int>(chunk_len),
            total_batches,
            work_queue_counter.data_ptr<int32_t>(),
            beliefs.data_ptr<scalar_t>(),
            final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_forward_sparse_logits_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& source_row_max,
    const torch::Tensor& source_row_inv_sum,
    const torch::Tensor& dest_row_max,
    const torch::Tensor& dest_row_inv_sum,
    const torch::Tensor& row_sums,
    const torch::Tensor& block_row_ptr,
    const torch::Tensor& block_col_idx,
    const torch::Tensor& block_dst_idx,
    const torch::Tensor& src_row_ptr,
    const torch::Tensor& src_nz_idx,
    const torch::Tensor& block_mask,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int num_states = static_cast<int>(local_logits.size(2));
    const int padded_states = static_cast<int>(row_sums.size(0));
    const int transition_rank = static_cast<int>(transition_source_logits.size(1));
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    auto launch_config = make_forward_sparse_launch_config(local_logits, num_states);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const float* source_row_max_ptr =
        source_row_max.defined() && source_row_max.numel() > 0 ? source_row_max.data_ptr<float>() : nullptr;
    const float* source_row_inv_sum_ptr =
        source_row_inv_sum.defined() && source_row_inv_sum.numel() > 0 ? source_row_inv_sum.data_ptr<float>() : nullptr;
    const float* dest_row_max_ptr =
        dest_row_max.defined() && dest_row_max.numel() > 0 ? dest_row_max.data_ptr<float>() : nullptr;
    const float* dest_row_inv_sum_ptr =
        dest_row_inv_sum.defined() && dest_row_inv_sum.numel() > 0 ? dest_row_inv_sum.data_ptr<float>() : nullptr;
    finalize_persistent_launch_config(
        "causal_machine_scan_forward_sparse_logits_chunk",
        launch_config,
        causal_machine_forward_sparse_factor_chunk_kernel<scalar_t, true>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_logits, &transition_dest_logits, &row_sums});
    causal_machine_forward_sparse_factor_chunk_kernel<scalar_t, true><<<
        launch_config.grid,
        launch_config.block,
        launch_config.shared_bytes,
        stream>>>(
            local_logits.data_ptr<scalar_t>(),
            transition_source_logits.data_ptr<float>(),
            transition_dest_logits.data_ptr<float>(),
            source_row_max_ptr,
            source_row_inv_sum_ptr,
            dest_row_max_ptr,
            dest_row_inv_sum_ptr,
            row_sums.data_ptr<float>(),
            block_row_ptr.data_ptr<int32_t>(),
            block_col_idx.data_ptr<int32_t>(),
            block_dst_idx.data_ptr<int32_t>(),
            src_row_ptr.data_ptr<int32_t>(),
            src_nz_idx.data_ptr<int32_t>(),
            block_mask.data_ptr<float>(),
            transition_context.data_ptr<scalar_t>(),
            initial_log_belief.data_ptr<scalar_t>(),
            static_cast<float>(transition_gate),
            transition_stay_probs.data_ptr<float>(),
            seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
            num_states,
            padded_states,
            transition_rank,
            static_cast<int>(block_size),
            seq_len,
            static_cast<int>(chunk_start),
            static_cast<int>(chunk_len),
            total_batches,
            work_queue_counter.data_ptr<int32_t>(),
            beliefs.data_ptr<scalar_t>(),
            final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KernelT, typename SharedBytesFn>
TiledScanKernelLaunchConfig select_tiled_launch_config_occupancy_driven(
    const torch::Tensor& reference,
    int total_batches,
    int required_threads,
    SharedBytesFn shared_bytes_fn,
    KernelT kernel) {
    const int device_index = reference.get_device();
    const int clamped_required_threads = std::min(std::max(required_threads, 1), 256);
    const int min_candidate_threads = clamped_required_threads <= 32 ? kWarpSize : 64;
    const std::array<int, 4> candidate_threads = {32, 64, 128, 256};
    TiledScanKernelLaunchConfig best_config{};
    int64_t best_score = std::numeric_limits<int64_t>::min();
    bool found_candidate = false;
    for (const int block_threads : candidate_threads) {
        if (block_threads < min_candidate_threads) {
            continue;
        }
        const size_t shared_bytes = shared_bytes_fn(block_threads);
        if (shared_bytes > static_cast<size_t>(cached_max_optin_bytes(device_index))) {
            continue;
        }
        const KernelLaunchDiagnostics diag = describe_kernel_launch(
            kernel,
            device_index,
            block_threads,
            shared_bytes);
        const int64_t useful_threads = std::min<int64_t>(block_threads, clamped_required_threads);
        const int64_t score =
            (diag.occupancy_pct * 1'000'000'000LL)
            + (diag.active_blocks_per_sm * 1'000'000LL)
            + (diag.active_warps_per_sm * 10'000LL)
            + (useful_threads * 100LL)
            - static_cast<int64_t>(shared_bytes / 256);
        if (!found_candidate || score > best_score) {
            best_score = score;
            best_config = {
                dim3(1),
                dim3(static_cast<unsigned int>(block_threads)),
                shared_bytes,
                device_index,
                total_batches,
            };
            found_candidate = true;
        }
    }
    if (found_candidate) {
        return best_config;
    }
    const int fallback_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(clamped_required_threads)));
    return {
        dim3(1),
        dim3(static_cast<unsigned int>(fallback_threads)),
        shared_bytes_fn(fallback_threads),
        device_index,
        total_batches,
    };
}

template <typename scalar_t>
void launch_forward_tiled_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    int64_t tile_size,
    int64_t split_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& work_queue_counter,
    const torch::Tensor& filtered_value_cache,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int num_states = static_cast<int>(local_logits.size(2));
    const int transition_rank = static_cast<int>(transition_source_probs.size(1));
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    const bool sm90_path = use_sm90_tiled_kernel_family(local_logits.get_device());
    auto launch_config = select_tiled_launch_config_occupancy_driven(
        local_logits,
        total_batches,
        std::max(static_cast<int>(tile_size), static_cast<int>(split_size)),
        [&](int block_threads) {
            return sm90_path
                ? forward_tiled_shared_bytes_sm90(
                    static_cast<int>(local_logits.size(2)),
                    static_cast<int>(tile_size),
                    static_cast<int>(split_size),
                    block_threads)
                : forward_tiled_shared_bytes(
                    static_cast<int>(local_logits.size(2)),
                    static_cast<int>(tile_size),
                    static_cast<int>(split_size),
                    block_threads);
        },
        sm90_path
            ? causal_machine_forward_tiled_chunk_kernel<scalar_t, true>
            : causal_machine_forward_tiled_chunk_kernel<scalar_t, false>);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    work_queue_counter.zero_();
    if (sm90_path) {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled forward sm90",
            launch_config,
            causal_machine_forward_tiled_chunk_kernel<scalar_t, true>);
    } else {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled forward",
            launch_config,
            causal_machine_forward_tiled_chunk_kernel<scalar_t, false>);
    }
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    if (sm90_path) {
        causal_machine_forward_tiled_chunk_kernel<scalar_t, true><<<
            launch_config.grid,
            launch_config.block,
            launch_config.shared_bytes,
            stream>>>(
                local_logits.data_ptr<scalar_t>(),
                transition_source_probs.data_ptr<float>(),
                transition_dest_probs.data_ptr<float>(),
                transition_context.data_ptr<scalar_t>(),
                initial_log_belief.data_ptr<float>(),
                static_cast<float>(transition_gate),
                transition_stay_probs.data_ptr<float>(),
                seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
                static_cast<float>(score_clamp_min),
                static_cast<float>(score_clamp_max),
                static_cast<float>(score_threshold),
                static_cast<int>(score_topk),
                num_states,
                transition_rank,
                static_cast<int>(tile_size),
                static_cast<int>(split_size),
                seq_len,
                static_cast<int>(chunk_start),
                static_cast<int>(chunk_len),
                total_batches,
                work_queue_counter.data_ptr<int32_t>(),
                filtered_value_cache.data_ptr<float>(),
                beliefs.data_ptr<scalar_t>(),
                final_log_belief.data_ptr<scalar_t>());
    } else {
        causal_machine_forward_tiled_chunk_kernel<scalar_t, false><<<
            launch_config.grid,
            launch_config.block,
            launch_config.shared_bytes,
            stream>>>(
                local_logits.data_ptr<scalar_t>(),
                transition_source_probs.data_ptr<float>(),
                transition_dest_probs.data_ptr<float>(),
                transition_context.data_ptr<scalar_t>(),
                initial_log_belief.data_ptr<float>(),
                static_cast<float>(transition_gate),
                transition_stay_probs.data_ptr<float>(),
                seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
                static_cast<float>(score_clamp_min),
                static_cast<float>(score_clamp_max),
                static_cast<float>(score_threshold),
                static_cast<int>(score_topk),
                num_states,
                transition_rank,
                static_cast<int>(tile_size),
                static_cast<int>(split_size),
                seq_len,
                static_cast<int>(chunk_start),
                static_cast<int>(chunk_len),
                total_batches,
                work_queue_counter.data_ptr<int32_t>(),
                filtered_value_cache.data_ptr<float>(),
                beliefs.data_ptr<scalar_t>(),
                final_log_belief.data_ptr<scalar_t>());
    }
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_forward_tiled_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    int64_t tile_size,
    int64_t split_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int total_batches = static_cast<int>(local_logits.size(0));
    const bool sm90_path = use_sm90_tiled_kernel_family(local_logits.get_device());
    auto launch_config = select_tiled_launch_config_occupancy_driven(
        local_logits,
        total_batches,
        std::max(static_cast<int>(tile_size), static_cast<int>(split_size)),
        [&](int block_threads) {
            return sm90_path
                ? forward_tiled_shared_bytes_sm90(
                    static_cast<int>(local_logits.size(2)),
                    static_cast<int>(tile_size),
                    static_cast<int>(split_size),
                    block_threads)
                : forward_tiled_shared_bytes(
                    static_cast<int>(local_logits.size(2)),
                    static_cast<int>(tile_size),
                    static_cast<int>(split_size),
                    block_threads);
        },
        sm90_path
            ? causal_machine_forward_tiled_chunk_kernel<scalar_t, true>
            : causal_machine_forward_tiled_chunk_kernel<scalar_t, false>);
    if (sm90_path) {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled forward sm90",
            launch_config,
            causal_machine_forward_tiled_chunk_kernel<scalar_t, true>);
    } else {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled forward",
            launch_config,
            causal_machine_forward_tiled_chunk_kernel<scalar_t, false>);
    }
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto filtered_value_cache = torch::empty(
        {launch_config.grid.x, local_logits.size(2)},
        transition_source_probs.options().dtype(torch::kFloat32));
    launch_forward_tiled_chunk<scalar_t>(
        local_logits,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        seq_lens,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk,
        tile_size,
        split_size,
        chunk_start,
        chunk_len,
        work_queue_counter,
        filtered_value_cache,
        beliefs,
        final_log_belief);
}

template <typename scalar_t>
void launch_forward_masked_tiled_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& row_sums,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& transition_mask,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    int64_t tile_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& work_queue_counter,
    const torch::Tensor& masked_transition_tile_cache,
    const torch::Tensor& filtered_value_cache,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int num_states = static_cast<int>(local_logits.size(2));
    const int transition_rank = static_cast<int>(transition_source_logits.size(1));
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    auto launch_config = select_tiled_launch_config_occupancy_driven(
        local_logits,
        total_batches,
        static_cast<int>(tile_size),
        [&](int block_threads) {
            return forward_masked_tiled_shared_bytes(
                static_cast<int>(local_logits.size(2)),
                block_threads);
        },
        causal_machine_forward_masked_tiled_chunk_kernel<scalar_t>);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    work_queue_counter.zero_();
    finalize_persistent_launch_config(
        "causal_machine_scan masked tiled forward",
        launch_config,
        causal_machine_forward_masked_tiled_chunk_kernel<scalar_t>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_logits, &transition_dest_logits});
    causal_machine_forward_masked_tiled_chunk_kernel<scalar_t><<<
        launch_config.grid,
        launch_config.block,
        launch_config.shared_bytes,
        stream>>>(
            local_logits.data_ptr<scalar_t>(),
            transition_source_logits.data_ptr<float>(),
            transition_dest_logits.data_ptr<float>(),
            transition_context.data_ptr<scalar_t>(),
            initial_log_belief.data_ptr<float>(),
            static_cast<float>(transition_gate),
            transition_stay_probs.data_ptr<float>(),
            transition_mask.data_ptr<bool>(),
            row_sums.data_ptr<float>(),
            seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
            static_cast<float>(score_clamp_min),
            static_cast<float>(score_clamp_max),
            static_cast<float>(score_threshold),
            static_cast<int>(score_topk),
            num_states,
            transition_rank,
            static_cast<int>(tile_size),
            seq_len,
            static_cast<int>(chunk_start),
            static_cast<int>(chunk_len),
            total_batches,
            work_queue_counter.data_ptr<int32_t>(),
            masked_transition_tile_cache.data_ptr<float>(),
            filtered_value_cache.data_ptr<float>(),
            beliefs.data_ptr<scalar_t>(),
            final_log_belief.data_ptr<scalar_t>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_forward_masked_tiled_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& row_sums,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& transition_mask,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    int64_t tile_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    auto launch_config = select_tiled_launch_config_occupancy_driven(
        local_logits,
        static_cast<int>(local_logits.size(0)),
        static_cast<int>(tile_size),
        [&](int block_threads) {
            return forward_masked_tiled_shared_bytes(
                static_cast<int>(local_logits.size(2)),
                block_threads);
        },
        causal_machine_forward_masked_tiled_chunk_kernel<scalar_t>);
    finalize_persistent_launch_config(
        "causal_machine_scan masked tiled forward",
        launch_config,
        causal_machine_forward_masked_tiled_chunk_kernel<scalar_t>);
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto masked_transition_tile_cache = torch::empty(
        {launch_config.grid.x, local_logits.size(2), tile_size},
        transition_source_logits.options().dtype(torch::kFloat32));
    auto filtered_value_cache = torch::empty(
        {launch_config.grid.x, local_logits.size(2)},
        transition_source_logits.options().dtype(torch::kFloat32));
    launch_forward_masked_tiled_chunk<scalar_t>(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        row_sums,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        transition_mask,
        seq_lens,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk,
        tile_size,
        chunk_start,
        chunk_len,
        work_queue_counter,
        masked_transition_tile_cache,
        filtered_value_cache,
        beliefs,
        final_log_belief);
}

template <typename scalar_t>
void launch_forward_masked_tiled_chunk(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& row_sums,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& transition_mask,
    const torch::Tensor& seq_lens,
    int64_t tile_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    launch_forward_masked_tiled_chunk<scalar_t>(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        row_sums,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        transition_mask,
        seq_lens,
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        0,
        tile_size,
        chunk_start,
        chunk_len,
        beliefs,
        final_log_belief);
}

template <typename scalar_t, typename packed_t, PackedTransitionFormat Format>
void launch_forward_tiled_chunk_packed(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_packed,
    const torch::Tensor& transition_source_scales,
    const torch::Tensor& transition_dest_packed,
    const torch::Tensor& transition_dest_scales,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    int64_t tile_size,
    int64_t split_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& work_queue_counter,
    const torch::Tensor& filtered_value_cache,
    const torch::Tensor& beliefs,
    const torch::Tensor& final_log_belief) {
    const int num_states = static_cast<int>(local_logits.size(2));
    const int transition_rank = static_cast<int>(transition_source_packed.size(1));
    const int seq_len = static_cast<int>(local_logits.size(1));
    const int total_batches = static_cast<int>(local_logits.size(0));
    const bool sm90_path = use_sm90_tiled_kernel_family(local_logits.get_device());
    auto launch_config = select_tiled_launch_config_occupancy_driven(
        local_logits,
        total_batches,
        std::max(static_cast<int>(tile_size), static_cast<int>(split_size)),
        [&](int block_threads) {
            return sm90_path
                ? forward_tiled_packed_shared_bytes_sm90(
                    static_cast<int>(local_logits.size(2)),
                    static_cast<int>(tile_size),
                    static_cast<int>(split_size),
                    block_threads)
                : forward_tiled_packed_shared_bytes(
                    static_cast<int>(local_logits.size(2)),
                    static_cast<int>(tile_size),
                    static_cast<int>(split_size),
                    block_threads);
        },
        sm90_path
            ? causal_machine_forward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, true>
            : causal_machine_forward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, false>);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    work_queue_counter.zero_();
    if (sm90_path) {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled forward packed sm90",
            launch_config,
            causal_machine_forward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, true>);
    } else {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled forward packed",
            launch_config,
            causal_machine_forward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, false>);
    }
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_packed, &transition_dest_packed, &transition_source_scales, &transition_dest_scales});
    if (sm90_path) {
        causal_machine_forward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, true><<<
            launch_config.grid,
            launch_config.block,
            launch_config.shared_bytes,
            stream>>>(
                local_logits.data_ptr<scalar_t>(),
                transition_source_packed.data_ptr<packed_t>(),
                transition_source_scales.data_ptr<float>(),
                transition_dest_packed.data_ptr<packed_t>(),
                transition_dest_scales.data_ptr<float>(),
                transition_context.data_ptr<scalar_t>(),
                initial_log_belief.data_ptr<float>(),
                static_cast<float>(transition_gate),
                transition_stay_probs.data_ptr<float>(),
                seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
                static_cast<float>(score_clamp_min),
                static_cast<float>(score_clamp_max),
                static_cast<float>(score_threshold),
                static_cast<int>(score_topk),
                num_states,
                transition_rank,
                static_cast<int>(tile_size),
                static_cast<int>(split_size),
                seq_len,
                static_cast<int>(chunk_start),
                static_cast<int>(chunk_len),
                total_batches,
                work_queue_counter.data_ptr<int32_t>(),
                filtered_value_cache.data_ptr<float>(),
                beliefs.data_ptr<scalar_t>(),
                final_log_belief.data_ptr<scalar_t>());
    } else {
        causal_machine_forward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, false><<<
            launch_config.grid,
            launch_config.block,
            launch_config.shared_bytes,
            stream>>>(
                local_logits.data_ptr<scalar_t>(),
                transition_source_packed.data_ptr<packed_t>(),
                transition_source_scales.data_ptr<float>(),
                transition_dest_packed.data_ptr<packed_t>(),
                transition_dest_scales.data_ptr<float>(),
                transition_context.data_ptr<scalar_t>(),
                initial_log_belief.data_ptr<float>(),
                static_cast<float>(transition_gate),
                transition_stay_probs.data_ptr<float>(),
                seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
                static_cast<float>(score_clamp_min),
                static_cast<float>(score_clamp_max),
                static_cast<float>(score_threshold),
                static_cast<int>(score_topk),
                num_states,
                transition_rank,
                static_cast<int>(tile_size),
                static_cast<int>(split_size),
                seq_len,
                static_cast<int>(chunk_start),
                static_cast<int>(chunk_len),
                total_batches,
                work_queue_counter.data_ptr<int32_t>(),
                filtered_value_cache.data_ptr<float>(),
                beliefs.data_ptr<scalar_t>(),
                final_log_belief.data_ptr<scalar_t>());
    }
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_tiled_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    int64_t tile_size,
    int64_t split_size,
    int64_t chunk_start,
    int64_t chunk_len,
    int64_t worker_blocks,
    const torch::Tensor& work_queue_counter,
    const torch::Tensor& latent_cache_staging,
    const torch::Tensor& grad_latent_accum_staging,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_probs_staging,
    const torch::Tensor& grad_transition_dest_probs_staging,
    const torch::Tensor& grad_transition_source_probs,
    const torch::Tensor& grad_transition_dest_probs,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_staging,
    const torch::Tensor& grad_transition_stay_staging,
    const torch::Tensor& grad_transition_gate,
    const torch::Tensor& grad_transition_stay) {
    const int num_states = static_cast<int>(beliefs.size(2));
    const int transition_rank = static_cast<int>(transition_source_probs.size(1));
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const bool sm90_path = use_sm90_tiled_kernel_family(beliefs.get_device());
    auto launch_config = select_tiled_launch_config_occupancy_driven(
        beliefs,
        std::max(static_cast<int>(worker_blocks), 1),
        std::max(static_cast<int>(tile_size), static_cast<int>(split_size)),
        [&](int block_threads) {
            return sm90_path
                ? backward_tiled_shared_bytes_sm90(
                    static_cast<int>(beliefs.size(2)),
                    static_cast<int>(split_size),
                    static_cast<int>(tile_size),
                    block_threads)
                : backward_tiled_shared_bytes(
                    static_cast<int>(beliefs.size(2)),
                    static_cast<int>(split_size),
                    static_cast<int>(tile_size),
                    block_threads);
        },
        sm90_path
            ? causal_machine_backward_tiled_chunk_kernel<scalar_t, true>
            : causal_machine_backward_tiled_chunk_kernel<scalar_t, false>);
    launch_config.total_tasks = std::max(static_cast<int>(worker_blocks), 1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    work_queue_counter.zero_();
    if (sm90_path) {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled backward sm90",
            launch_config,
            causal_machine_backward_tiled_chunk_kernel<scalar_t, true>);
    } else {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled backward",
            launch_config,
            causal_machine_backward_tiled_chunk_kernel<scalar_t, false>);
    }
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    if (sm90_path) {
        causal_machine_backward_tiled_chunk_kernel<scalar_t, true><<<
            launch_config.grid,
            launch_config.block,
            launch_config.shared_bytes,
            stream>>>(
                grad_beliefs.data_ptr<scalar_t>(),
                grad_final_belief.data_ptr<scalar_t>(),
                transition_source_probs.data_ptr<float>(),
                transition_dest_probs.data_ptr<float>(),
                transition_context.data_ptr<scalar_t>(),
                initial_log_belief.data_ptr<float>(),
                beliefs.data_ptr<scalar_t>(),
                static_cast<float>(transition_gate),
                transition_stay_probs.data_ptr<float>(),
                seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
                static_cast<float>(score_clamp_min),
                static_cast<float>(score_clamp_max),
                static_cast<float>(score_threshold),
                static_cast<int>(score_topk),
                num_states,
                transition_rank,
                static_cast<int>(tile_size),
                static_cast<int>(split_size),
                seq_len,
                static_cast<int>(chunk_start),
                static_cast<int>(chunk_len),
                total_batches,
                work_queue_counter.data_ptr<int32_t>(),
                latent_cache_staging.data_ptr<float>(),
                grad_latent_accum_staging.data_ptr<float>(),
                grad_local_logits.data_ptr<scalar_t>(),
                grad_transition_source_probs_staging.data_ptr<float>(),
                grad_transition_dest_probs_staging.data_ptr<float>(),
                grad_transition_source_probs.data_ptr<float>(),
                grad_transition_dest_probs.data_ptr<float>(),
                grad_transition_context.data_ptr<scalar_t>(),
                grad_initial_log_belief.data_ptr<float>(),
                grad_transition_gate_staging.data_ptr<float>(),
                grad_transition_stay_staging.data_ptr<float>(),
                grad_transition_gate.data_ptr<float>(),
                grad_transition_stay.data_ptr<float>());
    } else {
        causal_machine_backward_tiled_chunk_kernel<scalar_t, false><<<
            launch_config.grid,
            launch_config.block,
            launch_config.shared_bytes,
            stream>>>(
                grad_beliefs.data_ptr<scalar_t>(),
                grad_final_belief.data_ptr<scalar_t>(),
                transition_source_probs.data_ptr<float>(),
                transition_dest_probs.data_ptr<float>(),
                transition_context.data_ptr<scalar_t>(),
                initial_log_belief.data_ptr<float>(),
                beliefs.data_ptr<scalar_t>(),
                static_cast<float>(transition_gate),
                transition_stay_probs.data_ptr<float>(),
                seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
                static_cast<float>(score_clamp_min),
                static_cast<float>(score_clamp_max),
                static_cast<float>(score_threshold),
                static_cast<int>(score_topk),
                num_states,
                transition_rank,
                static_cast<int>(tile_size),
                static_cast<int>(split_size),
                seq_len,
                static_cast<int>(chunk_start),
                static_cast<int>(chunk_len),
                total_batches,
                work_queue_counter.data_ptr<int32_t>(),
                latent_cache_staging.data_ptr<float>(),
                grad_latent_accum_staging.data_ptr<float>(),
                grad_local_logits.data_ptr<scalar_t>(),
                grad_transition_source_probs_staging.data_ptr<float>(),
                grad_transition_dest_probs_staging.data_ptr<float>(),
                grad_transition_source_probs.data_ptr<float>(),
                grad_transition_dest_probs.data_ptr<float>(),
                grad_transition_context.data_ptr<scalar_t>(),
                grad_initial_log_belief.data_ptr<float>(),
                grad_transition_gate_staging.data_ptr<float>(),
                grad_transition_stay_staging.data_ptr<float>(),
                grad_transition_gate.data_ptr<float>(),
                grad_transition_stay.data_ptr<float>());
    }
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename packed_t, PackedTransitionFormat Format>
void launch_backward_tiled_chunk_packed(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_packed,
    const torch::Tensor& transition_source_scales,
    const torch::Tensor& transition_dest_packed,
    const torch::Tensor& transition_dest_scales,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    int64_t tile_size,
    int64_t split_size,
    int64_t chunk_start,
    int64_t chunk_len,
    int64_t worker_blocks,
    const torch::Tensor& work_queue_counter,
    const torch::Tensor& latent_cache_staging,
    const torch::Tensor& grad_latent_accum_staging,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_probs_staging,
    const torch::Tensor& grad_transition_dest_probs_staging,
    const torch::Tensor& grad_transition_source_probs,
    const torch::Tensor& grad_transition_dest_probs,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_staging,
    const torch::Tensor& grad_transition_stay_staging,
    const torch::Tensor& grad_transition_gate,
    const torch::Tensor& grad_transition_stay) {
    const int num_states = static_cast<int>(beliefs.size(2));
    const int transition_rank = static_cast<int>(transition_source_packed.size(1));
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const bool sm90_path = use_sm90_tiled_kernel_family(beliefs.get_device());
    auto launch_config = select_tiled_launch_config_occupancy_driven(
        beliefs,
        std::max(static_cast<int>(worker_blocks), 1),
        std::max(static_cast<int>(tile_size), static_cast<int>(split_size)),
        [&](int block_threads) {
            return sm90_path
                ? backward_tiled_packed_shared_bytes_sm90(
                    static_cast<int>(beliefs.size(2)),
                    static_cast<int>(split_size),
                    static_cast<int>(tile_size),
                    block_threads)
                : backward_tiled_packed_shared_bytes(
                    static_cast<int>(beliefs.size(2)),
                    static_cast<int>(split_size),
                    static_cast<int>(tile_size),
                    block_threads);
        },
        sm90_path
            ? causal_machine_backward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, true>
            : causal_machine_backward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, false>);
    launch_config.total_tasks = std::max(static_cast<int>(worker_blocks), 1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    work_queue_counter.zero_();
    if (sm90_path) {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled backward packed sm90",
            launch_config,
            causal_machine_backward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, true>);
    } else {
        finalize_persistent_launch_config(
            "causal_machine_scan tiled backward packed",
            launch_config,
            causal_machine_backward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, false>);
    }
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_packed, &transition_dest_packed, &transition_source_scales, &transition_dest_scales});
    if (sm90_path) {
        causal_machine_backward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, true><<<
            launch_config.grid,
            launch_config.block,
            launch_config.shared_bytes,
            stream>>>(
                grad_beliefs.data_ptr<scalar_t>(),
                grad_final_belief.data_ptr<scalar_t>(),
                transition_source_packed.data_ptr<packed_t>(),
                transition_source_scales.data_ptr<float>(),
                transition_dest_packed.data_ptr<packed_t>(),
                transition_dest_scales.data_ptr<float>(),
                transition_context.data_ptr<scalar_t>(),
                initial_log_belief.data_ptr<float>(),
                beliefs.data_ptr<scalar_t>(),
                static_cast<float>(transition_gate),
                transition_stay_probs.data_ptr<float>(),
                seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
                static_cast<float>(score_clamp_min),
                static_cast<float>(score_clamp_max),
                static_cast<float>(score_threshold),
                static_cast<int>(score_topk),
                num_states,
                transition_rank,
                static_cast<int>(tile_size),
                static_cast<int>(split_size),
                seq_len,
                static_cast<int>(chunk_start),
                static_cast<int>(chunk_len),
                total_batches,
                work_queue_counter.data_ptr<int32_t>(),
                latent_cache_staging.data_ptr<float>(),
                grad_latent_accum_staging.data_ptr<float>(),
                grad_local_logits.data_ptr<scalar_t>(),
                grad_transition_source_probs_staging.data_ptr<float>(),
                grad_transition_dest_probs_staging.data_ptr<float>(),
                grad_transition_source_probs.data_ptr<float>(),
                grad_transition_dest_probs.data_ptr<float>(),
                grad_transition_context.data_ptr<scalar_t>(),
                grad_initial_log_belief.data_ptr<float>(),
                grad_transition_gate_staging.data_ptr<float>(),
                grad_transition_stay_staging.data_ptr<float>(),
                grad_transition_gate.data_ptr<float>(),
                grad_transition_stay.data_ptr<float>());
    } else {
        causal_machine_backward_tiled_chunk_packed_kernel<scalar_t, packed_t, Format, false><<<
            launch_config.grid,
            launch_config.block,
            launch_config.shared_bytes,
            stream>>>(
                grad_beliefs.data_ptr<scalar_t>(),
                grad_final_belief.data_ptr<scalar_t>(),
                transition_source_packed.data_ptr<packed_t>(),
                transition_source_scales.data_ptr<float>(),
                transition_dest_packed.data_ptr<packed_t>(),
                transition_dest_scales.data_ptr<float>(),
                transition_context.data_ptr<scalar_t>(),
                initial_log_belief.data_ptr<float>(),
                beliefs.data_ptr<scalar_t>(),
                static_cast<float>(transition_gate),
                transition_stay_probs.data_ptr<float>(),
                seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
                static_cast<float>(score_clamp_min),
                static_cast<float>(score_clamp_max),
                static_cast<float>(score_threshold),
                static_cast<int>(score_topk),
                num_states,
                transition_rank,
                static_cast<int>(tile_size),
                static_cast<int>(split_size),
                seq_len,
                static_cast<int>(chunk_start),
                static_cast<int>(chunk_len),
                total_batches,
                work_queue_counter.data_ptr<int32_t>(),
                latent_cache_staging.data_ptr<float>(),
                grad_latent_accum_staging.data_ptr<float>(),
                grad_local_logits.data_ptr<scalar_t>(),
                grad_transition_source_probs_staging.data_ptr<float>(),
                grad_transition_dest_probs_staging.data_ptr<float>(),
                grad_transition_source_probs.data_ptr<float>(),
                grad_transition_dest_probs.data_ptr<float>(),
                grad_transition_context.data_ptr<scalar_t>(),
                grad_initial_log_belief.data_ptr<float>(),
                grad_transition_gate_staging.data_ptr<float>(),
                grad_transition_stay_staging.data_ptr<float>(),
                grad_transition_gate.data_ptr<float>(),
                grad_transition_stay.data_ptr<float>());
    }
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_masked_tiled_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& row_sums,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& transition_mask,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    int64_t tile_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& work_queue_counter,
    const torch::Tensor& masked_transition_tile_cache,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_per_batch,
    const torch::Tensor& grad_transition_dest_per_batch,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_per_batch,
    const torch::Tensor& grad_transition_stay_per_batch) {
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const int transition_rank = static_cast<int>(transition_source_logits.size(1));
    auto launch_config = select_tiled_launch_config_occupancy_driven(
        beliefs,
        total_batches,
        static_cast<int>(tile_size),
        [&](int block_threads) {
            return backward_masked_tiled_shared_bytes(
                static_cast<int>(beliefs.size(2)),
                static_cast<int>(tile_size),
                block_threads);
        },
        causal_machine_backward_masked_tiled_chunk_kernel<scalar_t>);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    work_queue_counter.zero_();
    finalize_persistent_launch_config(
        "causal_machine_scan masked tiled backward",
        launch_config,
        causal_machine_backward_masked_tiled_chunk_kernel<scalar_t>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_logits, &transition_dest_logits});
    causal_machine_backward_masked_tiled_chunk_kernel<scalar_t><<<
        launch_config.grid,
        launch_config.block,
        launch_config.shared_bytes,
        stream>>>(
            grad_beliefs.data_ptr<scalar_t>(),
            grad_final_belief.data_ptr<scalar_t>(),
            transition_source_logits.data_ptr<float>(),
            transition_dest_logits.data_ptr<float>(),
            transition_context.data_ptr<scalar_t>(),
            initial_log_belief.data_ptr<float>(),
            beliefs.data_ptr<scalar_t>(),
            static_cast<float>(transition_gate),
            transition_stay_probs.data_ptr<float>(),
            transition_mask.data_ptr<bool>(),
            row_sums.data_ptr<float>(),
            seq_lens.defined() && seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
            static_cast<float>(score_clamp_min),
            static_cast<float>(score_clamp_max),
            static_cast<float>(score_threshold),
            static_cast<int>(score_topk),
            static_cast<int>(beliefs.size(2)),
            transition_rank,
            static_cast<int>(tile_size),
            seq_len,
            static_cast<int>(chunk_start),
            static_cast<int>(chunk_len),
            total_batches,
            work_queue_counter.data_ptr<int32_t>(),
            masked_transition_tile_cache.data_ptr<float>(),
            grad_local_logits.data_ptr<scalar_t>(),
            grad_transition_source_per_batch.data_ptr<float>(),
            grad_transition_dest_per_batch.data_ptr<float>(),
            grad_transition_context.data_ptr<scalar_t>(),
            grad_initial_log_belief.data_ptr<float>(),
            grad_transition_gate_per_batch.data_ptr<float>(),
            grad_transition_stay_per_batch.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_masked_tiled_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& row_sums,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& transition_mask,
    const torch::Tensor& seq_lens,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    int64_t tile_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_per_batch,
    const torch::Tensor& grad_transition_dest_per_batch,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_per_batch,
    const torch::Tensor& grad_transition_stay_per_batch) {
    auto launch_config = select_tiled_launch_config_occupancy_driven(
        beliefs,
        static_cast<int>(beliefs.size(0)),
        static_cast<int>(tile_size),
        [&](int block_threads) {
            return backward_masked_tiled_shared_bytes(
                static_cast<int>(beliefs.size(2)),
                static_cast<int>(tile_size),
                block_threads);
        },
        causal_machine_backward_masked_tiled_chunk_kernel<scalar_t>);
    finalize_persistent_launch_config(
        "causal_machine_scan masked tiled backward",
        launch_config,
        causal_machine_backward_masked_tiled_chunk_kernel<scalar_t>);
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    auto masked_transition_tile_cache = torch::empty(
        {launch_config.grid.x, beliefs.size(2), tile_size},
        transition_source_logits.options().dtype(torch::kFloat32));
    launch_backward_masked_tiled_chunk<scalar_t>(
        grad_beliefs,
        grad_final_belief,
        transition_source_logits,
        transition_dest_logits,
        row_sums,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate,
        transition_stay_probs,
        transition_mask,
        seq_lens,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk,
        tile_size,
        chunk_start,
        chunk_len,
        work_queue_counter,
        masked_transition_tile_cache,
        grad_local_logits,
        grad_transition_source_per_batch,
        grad_transition_dest_per_batch,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate_per_batch,
        grad_transition_stay_per_batch);
}

template <typename scalar_t>
void launch_backward_masked_tiled_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& row_sums,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& transition_mask,
    const torch::Tensor& seq_lens,
    int64_t tile_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_per_batch,
    const torch::Tensor& grad_transition_dest_per_batch,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_per_batch,
    const torch::Tensor& grad_transition_stay_per_batch) {
    launch_backward_masked_tiled_chunk<scalar_t>(
        grad_beliefs,
        grad_final_belief,
        transition_source_logits,
        transition_dest_logits,
        row_sums,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate,
        transition_stay_probs,
        transition_mask,
        seq_lens,
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        0,
        tile_size,
        chunk_start,
        chunk_len,
        grad_local_logits,
        grad_transition_source_per_batch,
        grad_transition_dest_per_batch,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate_per_batch,
        grad_transition_stay_per_batch);
}

template <typename scalar_t>
void launch_backward_sparse_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_blocks,
    const torch::Tensor& block_row_ptr,
    const torch::Tensor& block_col_idx,
    const torch::Tensor& block_dst_idx,
    const torch::Tensor& src_row_ptr,
    const torch::Tensor& src_nz_idx,
    const torch::Tensor& grouped_src_row_ptr,
    const torch::Tensor& grouped_src_block_idx,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_blocks,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate,
    const torch::Tensor& grad_transition_stay) {
    const int num_states = static_cast<int>(beliefs.size(2));
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const int grouped_src_group_count = static_cast<int>(grouped_src_block_idx.numel());
    const int32_t* grouped_src_row_ptr_ptr =
        grouped_src_row_ptr.numel() > 0 ? grouped_src_row_ptr.data_ptr<int32_t>() : nullptr;
    const int32_t* grouped_src_block_idx_ptr =
        grouped_src_block_idx.numel() > 0 ? grouped_src_block_idx.data_ptr<int32_t>() : nullptr;
    auto launch_config = make_backward_sparse_launch_config(beliefs, num_states);
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan_backward_sparse_chunk",
        launch_config,
        causal_machine_backward_sparse_chunk_kernel<scalar_t>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_blocks});
    causal_machine_backward_sparse_chunk_kernel<scalar_t><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        grad_beliefs.data_ptr<scalar_t>(),
        grad_final_belief.data_ptr<scalar_t>(),
        transition_blocks.data_ptr<float>(),
        block_row_ptr.data_ptr<int32_t>(),
        block_col_idx.data_ptr<int32_t>(),
        block_dst_idx.data_ptr<int32_t>(),
        src_row_ptr.data_ptr<int32_t>(),
        src_nz_idx.data_ptr<int32_t>(),
        grouped_src_row_ptr_ptr,
        grouped_src_block_idx_ptr,
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        beliefs.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
        num_states,
        static_cast<int>(block_size),
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        grouped_src_group_count,
        work_queue_counter.data_ptr<int32_t>(),
        grad_local_logits.data_ptr<scalar_t>(),
        grad_transition_blocks.data_ptr<float>(),
        grad_transition_context.data_ptr<scalar_t>(),
        grad_initial_log_belief.data_ptr<scalar_t>(),
        grad_transition_gate.data_ptr<float>(),
        grad_transition_stay.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_sparse_factor_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& row_sums,
    const torch::Tensor& block_row_ptr,
    const torch::Tensor& block_col_idx,
    const torch::Tensor& block_dst_idx,
    const torch::Tensor& src_row_ptr,
    const torch::Tensor& src_nz_idx,
    const torch::Tensor& grouped_src_row_ptr,
    const torch::Tensor& grouped_src_block_idx,
    const torch::Tensor& block_mask,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_probs,
    const torch::Tensor& grad_transition_dest_probs,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate,
    const torch::Tensor& grad_transition_stay) {
    const int num_states = static_cast<int>(beliefs.size(2));
    const int padded_states = static_cast<int>(row_sums.size(0));
    const int transition_rank = static_cast<int>(transition_source_probs.size(1));
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const int grouped_src_group_count = static_cast<int>(grouped_src_block_idx.numel());
    const int32_t* grouped_src_row_ptr_ptr =
        grouped_src_row_ptr.numel() > 0 ? grouped_src_row_ptr.data_ptr<int32_t>() : nullptr;
    const int32_t* grouped_src_block_idx_ptr =
        grouped_src_block_idx.numel() > 0 ? grouped_src_block_idx.data_ptr<int32_t>() : nullptr;
    auto launch_config = make_backward_sparse_launch_config(beliefs, num_states);
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan_backward_sparse_factor_chunk",
        launch_config,
        causal_machine_backward_sparse_factor_chunk_kernel<scalar_t, false>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs, &row_sums});
    causal_machine_backward_sparse_factor_chunk_kernel<scalar_t, false><<<
        launch_config.grid,
        launch_config.block,
        launch_config.shared_bytes,
        stream>>>(
            grad_beliefs.data_ptr<scalar_t>(),
            grad_final_belief.data_ptr<scalar_t>(),
            transition_source_probs.data_ptr<float>(),
            transition_dest_probs.data_ptr<float>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            row_sums.data_ptr<float>(),
            block_row_ptr.data_ptr<int32_t>(),
            block_col_idx.data_ptr<int32_t>(),
            block_dst_idx.data_ptr<int32_t>(),
            src_row_ptr.data_ptr<int32_t>(),
            src_nz_idx.data_ptr<int32_t>(),
            grouped_src_row_ptr_ptr,
            grouped_src_block_idx_ptr,
            block_mask.data_ptr<float>(),
            transition_context.data_ptr<scalar_t>(),
            initial_log_belief.data_ptr<scalar_t>(),
            beliefs.data_ptr<scalar_t>(),
            static_cast<float>(transition_gate),
            transition_stay_probs.data_ptr<float>(),
            seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
            num_states,
            padded_states,
            transition_rank,
            static_cast<int>(block_size),
            seq_len,
            static_cast<int>(chunk_start),
            static_cast<int>(chunk_len),
            total_batches,
            grouped_src_group_count,
            work_queue_counter.data_ptr<int32_t>(),
            grad_local_logits.data_ptr<scalar_t>(),
            grad_transition_source_probs.data_ptr<float>(),
            grad_transition_dest_probs.data_ptr<float>(),
            grad_transition_context.data_ptr<scalar_t>(),
            grad_initial_log_belief.data_ptr<scalar_t>(),
            grad_transition_gate.data_ptr<float>(),
            grad_transition_stay.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_sparse_logits_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_logits,
    const torch::Tensor& transition_dest_logits,
    const torch::Tensor& source_row_max,
    const torch::Tensor& source_row_inv_sum,
    const torch::Tensor& dest_row_max,
    const torch::Tensor& dest_row_inv_sum,
    const torch::Tensor& row_sums,
    const torch::Tensor& block_row_ptr,
    const torch::Tensor& block_col_idx,
    const torch::Tensor& block_dst_idx,
    const torch::Tensor& src_row_ptr,
    const torch::Tensor& src_nz_idx,
    const torch::Tensor& grouped_src_row_ptr,
    const torch::Tensor& grouped_src_block_idx,
    const torch::Tensor& block_mask,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    const torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_probs,
    const torch::Tensor& grad_transition_dest_probs,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate,
    const torch::Tensor& grad_transition_stay) {
    const int num_states = static_cast<int>(beliefs.size(2));
    const int padded_states = static_cast<int>(row_sums.size(0));
    const int transition_rank = static_cast<int>(transition_source_logits.size(1));
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const int grouped_src_group_count = static_cast<int>(grouped_src_block_idx.numel());
    const int32_t* grouped_src_row_ptr_ptr =
        grouped_src_row_ptr.numel() > 0 ? grouped_src_row_ptr.data_ptr<int32_t>() : nullptr;
    const int32_t* grouped_src_block_idx_ptr =
        grouped_src_block_idx.numel() > 0 ? grouped_src_block_idx.data_ptr<int32_t>() : nullptr;
    auto launch_config = make_backward_sparse_launch_config(beliefs, num_states);
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan_backward_sparse_logits_chunk",
        launch_config,
        causal_machine_backward_sparse_factor_chunk_kernel<scalar_t, true>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_logits, &transition_dest_logits, &row_sums});
    causal_machine_backward_sparse_factor_chunk_kernel<scalar_t, true><<<
        launch_config.grid,
        launch_config.block,
        launch_config.shared_bytes,
        stream>>>(
            grad_beliefs.data_ptr<scalar_t>(),
            grad_final_belief.data_ptr<scalar_t>(),
            transition_source_logits.data_ptr<float>(),
            transition_dest_logits.data_ptr<float>(),
            source_row_max.data_ptr<float>(),
            source_row_inv_sum.data_ptr<float>(),
            dest_row_max.data_ptr<float>(),
            dest_row_inv_sum.data_ptr<float>(),
            row_sums.data_ptr<float>(),
            block_row_ptr.data_ptr<int32_t>(),
            block_col_idx.data_ptr<int32_t>(),
            block_dst_idx.data_ptr<int32_t>(),
            src_row_ptr.data_ptr<int32_t>(),
            src_nz_idx.data_ptr<int32_t>(),
            grouped_src_row_ptr_ptr,
            grouped_src_block_idx_ptr,
            block_mask.data_ptr<float>(),
            transition_context.data_ptr<scalar_t>(),
            initial_log_belief.data_ptr<scalar_t>(),
            beliefs.data_ptr<scalar_t>(),
            static_cast<float>(transition_gate),
            transition_stay_probs.data_ptr<float>(),
            seq_lens.numel() > 0 ? seq_lens.data_ptr<int64_t>() : nullptr,
            num_states,
            padded_states,
            transition_rank,
            static_cast<int>(block_size),
            seq_len,
            static_cast<int>(chunk_start),
            static_cast<int>(chunk_len),
            total_batches,
            grouped_src_group_count,
            work_queue_counter.data_ptr<int32_t>(),
            grad_local_logits.data_ptr<scalar_t>(),
            grad_transition_source_probs.data_ptr<float>(),
            grad_transition_dest_probs.data_ptr<float>(),
            grad_transition_context.data_ptr<scalar_t>(),
            grad_initial_log_belief.data_ptr<scalar_t>(),
            grad_transition_gate.data_ptr<float>(),
            grad_transition_stay.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, bool DirectGradReduce = false>
void launch_backward_chunk_dense_128_rank8(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    double transition_gate,
    const torch::Tensor& transition_stay_probs,
    double score_clamp_min,
    double score_clamp_max,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_per_batch,
    const torch::Tensor& grad_transition_dest_per_batch,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_per_batch,
    const torch::Tensor& grad_transition_stay_per_batch) {
    ScanKernelLaunchConfig launch_config{
        dim3(1),
        dim3(128),
        8,
        backward_dense_128_rank8_shared_bytes(DirectGradReduce),
        beliefs.get_device(),
        DirectGradReduce,
        static_cast<int>(beliefs.size(0)),
    };
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan backward dense_128_rank8",
        launch_config,
        causal_machine_backward_chunk_dense_128_rank8_kernel<scalar_t, DirectGradReduce>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    causal_machine_backward_chunk_dense_128_rank8_kernel<scalar_t, DirectGradReduce><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        grad_beliefs.data_ptr<scalar_t>(),
        grad_final_belief.data_ptr<scalar_t>(),
        transition_source_probs.data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        beliefs.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        static_cast<float>(score_clamp_min),
        static_cast<float>(score_clamp_max),
        static_cast<int>(beliefs.size(1)),
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        static_cast<int>(beliefs.size(0)),
        work_queue_counter.data_ptr<int32_t>(),
        grad_local_logits.data_ptr<scalar_t>(),
        grad_transition_source_per_batch.data_ptr<float>(),
        grad_transition_dest_per_batch.data_ptr<float>(),
        grad_transition_context.data_ptr<scalar_t>(),
        grad_initial_log_belief.data_ptr<scalar_t>(),
        grad_transition_gate_per_batch.data_ptr<float>(),
        grad_transition_stay_per_batch.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int StaticTransitionRank = -1, bool DirectGradReduce = false, bool InputsAreLogits = false>
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
    double score_clamp_min,
    double score_clamp_max,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_per_batch,
    const torch::Tensor& grad_transition_dest_per_batch,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_gate_per_batch,
    const torch::Tensor& grad_transition_stay_per_batch) {
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_probs.size(1));
    auto launch_config = make_backward_launch_config(beliefs, transition_rank, DirectGradReduce);
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan backward",
        launch_config,
        causal_machine_backward_chunk_kernel<scalar_t, StaticTransitionRank, DirectGradReduce, InputsAreLogits>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    causal_machine_backward_chunk_kernel<scalar_t, StaticTransitionRank, DirectGradReduce, InputsAreLogits><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        grad_beliefs.data_ptr<scalar_t>(),
        grad_final_belief.data_ptr<scalar_t>(),
        transition_source_probs.data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        beliefs.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        static_cast<float>(score_clamp_min),
        static_cast<float>(score_clamp_max),
        launch_config.transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        grad_local_logits.data_ptr<scalar_t>(),
        grad_transition_source_per_batch.data_ptr<float>(),
        grad_transition_dest_per_batch.data_ptr<float>(),
        grad_transition_context.data_ptr<scalar_t>(),
        grad_initial_log_belief.data_ptr<scalar_t>(),
        grad_transition_gate_per_batch.data_ptr<float>(),
        grad_transition_stay_per_batch.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int StaticTransitionRank = -1, bool DirectGradReduce = false, bool InputsAreLogits = false>
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
    launch_backward_chunk<scalar_t, StaticTransitionRank, DirectGradReduce, InputsAreLogits>(
        grad_beliefs,
        grad_final_belief,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate,
        transition_stay_probs,
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        chunk_start,
        chunk_len,
        grad_local_logits,
        grad_transition_source_per_batch,
        grad_transition_dest_per_batch,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate_per_batch,
        grad_transition_stay_per_batch);
}

template <typename scalar_t, int StaticTransitionRank = -1, bool DirectGradReduce = false, bool InputsAreLogits = false>
void launch_backward_composable_chunk(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& beliefs,
    const torch::Tensor& transition_stay_probs,
    int64_t chunk_start,
    int64_t chunk_len,
    const torch::Tensor& grad_local_logits,
    const torch::Tensor& grad_transition_source_per_batch,
    const torch::Tensor& grad_transition_dest_per_batch,
    const torch::Tensor& grad_transition_context,
    const torch::Tensor& grad_initial_log_belief,
    const torch::Tensor& grad_transition_stay_per_batch) {
    const int seq_len = static_cast<int>(grad_beliefs.size(1));
    const int total_batches = static_cast<int>(grad_beliefs.size(0));
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_probs.size(1));
    auto launch_config = make_backward_launch_config(grad_beliefs, transition_rank, DirectGradReduce);
    auto work_queue_counter = make_device_work_queue_counter(grad_beliefs);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan composable backward",
        launch_config,
        causal_machine_backward_composable_chunk_kernel<scalar_t, StaticTransitionRank, DirectGradReduce, InputsAreLogits>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_probs, &transition_dest_probs});
    causal_machine_backward_composable_chunk_kernel<scalar_t, StaticTransitionRank, DirectGradReduce, InputsAreLogits><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        grad_beliefs.data_ptr<scalar_t>(),
        grad_final_belief.data_ptr<scalar_t>(),
        transition_source_probs.data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        beliefs.data_ptr<scalar_t>(),
        transition_stay_probs.data_ptr<float>(),
        launch_config.transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        grad_local_logits.data_ptr<scalar_t>(),
        grad_transition_source_per_batch.data_ptr<float>(),
        grad_transition_dest_per_batch.data_ptr<float>(),
        grad_transition_context.data_ptr<scalar_t>(),
        grad_initial_log_belief.data_ptr<scalar_t>(),
        grad_transition_stay_per_batch.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int StaticTransitionRank = -1, bool DirectGradReduce = false>
void launch_backward_chunk_quantized(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_q,
    const torch::Tensor& transition_source_scales,
    const torch::Tensor& transition_dest_q,
    const torch::Tensor& transition_dest_scales,
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
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_q.size(1));
    auto launch_config = make_backward_packed_launch_config(beliefs, transition_rank, DirectGradReduce);
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan quantized backward",
        launch_config,
        causal_machine_backward_chunk_packed_kernel<scalar_t, int8_t, PackedTransitionFormat::Int8, StaticTransitionRank, DirectGradReduce>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_q, &transition_dest_q});
    causal_machine_backward_chunk_packed_kernel<scalar_t, int8_t, PackedTransitionFormat::Int8, StaticTransitionRank, DirectGradReduce><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        grad_beliefs.data_ptr<scalar_t>(),
        grad_final_belief.data_ptr<scalar_t>(),
        transition_source_q.data_ptr<int8_t>(),
        transition_source_scales.data_ptr<float>(),
        transition_dest_q.data_ptr<int8_t>(),
        transition_dest_scales.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        beliefs.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        launch_config.transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        grad_local_logits.data_ptr<scalar_t>(),
        grad_transition_source_per_batch.data_ptr<float>(),
        grad_transition_dest_per_batch.data_ptr<float>(),
        grad_transition_context.data_ptr<scalar_t>(),
        grad_initial_log_belief.data_ptr<scalar_t>(),
        grad_transition_gate_per_batch.data_ptr<float>(),
        grad_transition_stay_per_batch.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <PackedTransitionFormat Format, typename scalar_t, int StaticTransitionRank = -1, bool DirectGradReduce = false>
void launch_backward_chunk_fp8(
    const torch::Tensor& grad_beliefs,
    const torch::Tensor& grad_final_belief,
    const torch::Tensor& transition_source_packed,
    const torch::Tensor& transition_source_scales,
    const torch::Tensor& transition_dest_packed,
    const torch::Tensor& transition_dest_scales,
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
    static_assert(
        Format == PackedTransitionFormat::Fp8E4M3 || Format == PackedTransitionFormat::Fp8E5M2,
        "launch_backward_chunk_fp8 only supports FP8 packed formats");
    const int seq_len = static_cast<int>(beliefs.size(1));
    const int total_batches = static_cast<int>(beliefs.size(0));
    const int transition_rank = StaticTransitionRank > 0 ? StaticTransitionRank : static_cast<int>(transition_source_packed.size(1));
    auto launch_config = make_backward_packed_launch_config(beliefs, transition_rank, DirectGradReduce);
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_persistent_launch_config(
        "causal_machine_scan fp8 backward",
        launch_config,
        causal_machine_backward_chunk_packed_kernel<scalar_t, uint8_t, Format, StaticTransitionRank, DirectGradReduce>);
    const bool persisting_l2_window = try_set_persisting_l2_window_for_tensors(
        stream,
        launch_config.device_index,
        {&transition_source_packed, &transition_dest_packed});
    causal_machine_backward_chunk_packed_kernel<scalar_t, uint8_t, Format, StaticTransitionRank, DirectGradReduce><<<launch_config.grid, launch_config.block, launch_config.shared_bytes, stream>>>(
        grad_beliefs.data_ptr<scalar_t>(),
        grad_final_belief.data_ptr<scalar_t>(),
        transition_source_packed.data_ptr<uint8_t>(),
        transition_source_scales.data_ptr<float>(),
        transition_dest_packed.data_ptr<uint8_t>(),
        transition_dest_scales.data_ptr<float>(),
        transition_context.data_ptr<scalar_t>(),
        initial_log_belief.data_ptr<scalar_t>(),
        beliefs.data_ptr<scalar_t>(),
        static_cast<float>(transition_gate),
        transition_stay_probs.data_ptr<float>(),
        launch_config.transition_rank,
        seq_len,
        static_cast<int>(chunk_start),
        static_cast<int>(chunk_len),
        total_batches,
        work_queue_counter.data_ptr<int32_t>(),
        grad_local_logits.data_ptr<scalar_t>(),
        grad_transition_source_per_batch.data_ptr<float>(),
        grad_transition_dest_per_batch.data_ptr<float>(),
        grad_transition_context.data_ptr<scalar_t>(),
        grad_initial_log_belief.data_ptr<scalar_t>(),
        grad_transition_gate_per_batch.data_ptr<float>(),
        grad_transition_stay_per_batch.data_ptr<float>());
    if (persisting_l2_window) {
        clear_persisting_l2_window(stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

int64_t causal_machine_scan_single_launch_max_seq_len_cuda() {
    return scan_single_launch_max_seq_len();
}

int64_t causal_machine_scan_forward_chunk_shared_bytes_cuda(int64_t transition_rank) {
    return static_cast<int64_t>(forward_chunk_shared_bytes(kMaxNumStates, static_cast<int>(transition_rank)));
}

int64_t causal_machine_scan_forward_tiled_chunk_shared_bytes_cuda(
    int64_t num_states,
    int64_t split_size,
    int64_t tile_size) {
    const int required_threads = std::min(
        std::max(static_cast<int>(tile_size), static_cast<int>(split_size)),
        256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    return static_cast<int64_t>(forward_tiled_shared_bytes(
        static_cast<int>(num_states),
        static_cast<int>(tile_size),
        static_cast<int>(split_size),
        block_threads));
}

int64_t causal_machine_scan_forward_masked_tiled_chunk_shared_bytes_cuda(
    int64_t num_states,
    int64_t tile_size) {
    const int required_threads = std::min(static_cast<int>(std::max<int64_t>(tile_size, 1)), 256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    return static_cast<int64_t>(forward_masked_tiled_shared_bytes(
        static_cast<int>(num_states),
        block_threads));
}

int64_t causal_machine_scan_backward_tiled_chunk_shared_bytes_cuda(
    int64_t num_states,
    int64_t split_size,
    int64_t tile_size) {
    const int required_threads = std::min(
        std::max(static_cast<int>(tile_size), static_cast<int>(split_size)),
        256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    return static_cast<int64_t>(backward_tiled_shared_bytes(
        static_cast<int>(num_states),
        static_cast<int>(split_size),
        static_cast<int>(tile_size),
        block_threads));
}

int64_t causal_machine_scan_backward_masked_tiled_chunk_shared_bytes_cuda(
    int64_t num_states,
    int64_t tile_size) {
    const int required_threads = std::min(static_cast<int>(std::max<int64_t>(tile_size, 1)), 256);
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(required_threads)));
    return static_cast<int64_t>(backward_masked_tiled_shared_bytes(
        static_cast<int>(num_states),
        static_cast<int>(tile_size),
        block_threads));
}

int64_t causal_machine_scan_backward_chunk_shared_bytes_cuda(int64_t num_states, int64_t transition_rank, bool direct_grad_reduce) {
    return static_cast<int64_t>(backward_chunk_shared_bytes(
        static_cast<int>(num_states),
        static_cast<int>(transition_rank),
        direct_grad_reduce));
}

bool causal_machine_scan_can_use_direct_grad_reduce_cuda(int64_t device_index, int64_t num_states, int64_t transition_rank) {
    return can_use_direct_grad_reduce(
        static_cast<int>(device_index),
        static_cast<int>(num_states),
        static_cast<int>(transition_rank));
}

bool causal_machine_scan_can_use_tiled_forward_kernel_cuda(
    int64_t device_index,
    int64_t num_states,
    int64_t tile_size,
    int64_t split_size) {
    const auto shared_bytes = causal_machine_scan_forward_tiled_chunk_shared_bytes_cuda(
        num_states,
        split_size,
        tile_size);
    return shared_bytes <= static_cast<int64_t>(cached_max_optin_bytes(static_cast<int>(device_index)));
}

bool causal_machine_scan_can_use_tiled_forward_tensor_core_math_cuda(
    int64_t device_index,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t split_size) {
    return can_use_tiled_forward_tensor_core_math(
        static_cast<int>(device_index),
        num_states,
        transition_rank,
        tile_size,
        split_size);
}

bool causal_machine_scan_can_use_masked_tiled_forward_kernel_cuda(
    int64_t device_index,
    int64_t num_states,
    int64_t tile_size) {
    const auto shared_bytes = causal_machine_scan_forward_masked_tiled_chunk_shared_bytes_cuda(
        num_states,
        tile_size);
    return shared_bytes <= static_cast<int64_t>(cached_max_optin_bytes(static_cast<int>(device_index)));
}

bool causal_machine_scan_can_use_tiled_backward_kernel_cuda(
    int64_t device_index,
    int64_t num_states,
    int64_t tile_size,
    int64_t split_size) {
    const auto shared_bytes = causal_machine_scan_backward_tiled_chunk_shared_bytes_cuda(
        num_states,
        split_size,
        tile_size);
    return shared_bytes <= static_cast<int64_t>(cached_max_optin_bytes(static_cast<int>(device_index)));
}

bool causal_machine_scan_can_use_masked_tiled_backward_kernel_cuda(
    int64_t device_index,
    int64_t num_states,
    int64_t tile_size) {
    const auto shared_bytes = causal_machine_scan_backward_masked_tiled_chunk_shared_bytes_cuda(
        num_states,
        tile_size);
    return shared_bytes <= static_cast<int64_t>(cached_max_optin_bytes(static_cast<int>(device_index)));
}

int64_t causal_machine_scan_cached_max_optin_bytes_cuda(int64_t device_index) {
    return cached_max_optin_bytes(static_cast<int>(device_index));
}

int64_t causal_machine_scan_cached_capability_major_cuda(int64_t device_index) {
    return cached_capability_major(static_cast<int>(device_index));
}

int64_t causal_machine_scan_cached_capability_minor_cuda(int64_t device_index) {
    return cached_capability_minor(static_cast<int>(device_index));
}

int64_t causal_machine_scan_cached_sm_count_cuda(int64_t device_index) {
    return cached_sm_count(static_cast<int>(device_index));
}

int64_t causal_machine_scan_cached_l2_cache_size_cuda(int64_t device_index) {
    return cached_l2_cache_size(static_cast<int>(device_index));
}

int64_t causal_machine_scan_cached_persisting_l2_cache_max_size_cuda(int64_t device_index) {
    return cached_persisting_l2_cache_max_size(static_cast<int>(device_index));
}

bool causal_machine_scan_supports_persisting_l2_window_cuda(int64_t device_index) {
    return supports_persisting_l2_window(static_cast<int>(device_index));
}

int64_t causal_machine_scan_cached_total_global_mem_cuda(int64_t device_index) {
    return cached_total_global_mem(static_cast<int>(device_index));
}

int64_t causal_machine_scan_persistent_worker_blocks_cuda(int64_t device_index, int64_t total_batches) {
    return static_cast<int64_t>(
        persistent_worker_blocks(static_cast<int>(device_index), static_cast<int>(total_batches)));
}

int64_t causal_machine_scan_preferred_load_bytes_cuda(
    int64_t num_states,
    int64_t tile_size,
    int64_t split_size) {
    (void)num_states;
    return preferred_float_load_bytes(std::max<int64_t>(tile_size, split_size));
}

int64_t causal_machine_scan_elements_per_load_cuda(
    int64_t num_states,
    int64_t tile_size,
    int64_t split_size) {
    (void)num_states;
    return elements_per_float_load(std::max<int64_t>(tile_size, split_size));
}

bool causal_machine_scan_can_use_vectorized_io_cuda(
    int64_t num_states,
    int64_t tile_size,
    int64_t split_size) {
    (void)num_states;
    return can_use_vectorized_float_io(std::max<int64_t>(tile_size, split_size));
}

bool causal_machine_scan_can_use_async_memcpy_cuda(int64_t device_index) {
    return can_use_async_memcpy(static_cast<int>(device_index));
}

bool causal_machine_scan_can_use_tensor_cores_cuda(int64_t device_index) {
    return can_use_tensor_cores(static_cast<int>(device_index));
}

bool causal_machine_scan_can_use_half2_path_cuda(int64_t device_index) {
    return can_use_half2_path(static_cast<int>(device_index));
}

bool causal_machine_scan_can_use_wmma_cuda(int64_t device_index) {
    return can_use_wmma(static_cast<int>(device_index));
}

bool causal_machine_scan_can_use_tma_cuda(int64_t device_index) {
    (void)device_index;
    // Hopper-class hardware support exists architecturally, but the structured
    // scan kernels do not yet implement a real TMA path.
    return false;
}

bool causal_machine_scan_can_use_wgmma_cuda(int64_t device_index) {
    (void)device_index;
    // Hopper-class hardware support exists architecturally, but the structured
    // scan kernels do not yet implement a real WGMMA path.
    return false;
}

std::vector<int64_t> causal_machine_scan_describe_tiled_forward_runtime_cuda(
    int64_t device_index,
    int64_t total_batches,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t split_size,
    int64_t seq_len) {
    const int device = static_cast<int>(device_index);
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
    const int block_threads = tiled_block_threads(tile_size, split_size);
    const bool sm90_path = use_sm90_tiled_kernel_family(device);
    const int64_t shared_bytes = sm90_path
        ? static_cast<int64_t>(forward_tiled_shared_bytes_sm90(
            static_cast<int>(num_states),
            static_cast<int>(tile_size),
            static_cast<int>(split_size),
            block_threads))
        : causal_machine_scan_forward_tiled_chunk_shared_bytes_cuda(
            num_states,
            split_size,
            tile_size);
    const auto diag = sm90_path
        ? describe_kernel_launch(
            causal_machine_forward_tiled_chunk_kernel<float, true>,
            device,
            block_threads,
            static_cast<size_t>(shared_bytes))
        : describe_kernel_launch(
            causal_machine_forward_tiled_chunk_kernel<float, false>,
            device,
            block_threads,
            static_cast<size_t>(shared_bytes));
    const int64_t worker_blocks = persistent_worker_blocks(device, static_cast<int>(total_batches));
    const int64_t preferred_load_bytes = causal_machine_scan_preferred_load_bytes_cuda(
        num_states,
        tile_size,
        split_size);
    const int64_t elements_per_load = causal_machine_scan_elements_per_load_cuda(
        num_states,
        tile_size,
        split_size);
    const int64_t candidate_bytes = estimate_persisting_l2_candidate_bytes_for_tiled_path(
        num_states,
        transition_rank);
    const int64_t effective_bytes = estimate_persisting_l2_effective_bytes(device, candidate_bytes);
    const int64_t free_mem_bytes = std::max<int64_t>(current_free_global_mem(), 1);
    const int64_t total_mem_bytes = std::max<int64_t>(cached_total_global_mem(device), 1);
    const int64_t tensor_core_math = static_cast<int64_t>(
        can_use_tiled_forward_tensor_core_math(
            device,
            num_states,
            transition_rank,
            tile_size,
            split_size));
    return {
        worker_blocks,
        shared_bytes,
        diag.block_threads,
        preferred_load_bytes,
        elements_per_load,
        static_cast<int64_t>(causal_machine_scan_can_use_vectorized_io_cuda(num_states, tile_size, split_size)),
        static_cast<int64_t>(causal_machine_scan_can_use_async_memcpy_cuda(device_index)),
        tensor_core_math,
        diag.active_blocks_per_sm,
        diag.active_warps_per_sm,
        diag.max_warps_per_sm,
        diag.occupancy_pct,
        diag.registers_per_thread,
        diag.static_smem_bytes,
        candidate_bytes,
        effective_bytes,
        static_cast<int64_t>(effective_bytes > 0),
        estimate_tiled_forward_bytes_moved(total_batches, seq_len, num_states, transition_rank),
        estimate_tiled_forward_sync_points(seq_len, num_states, transition_rank, tile_size, split_size),
        free_mem_bytes,
        total_mem_bytes,
    };
}

std::vector<int64_t> causal_machine_scan_describe_masked_tiled_forward_runtime_cuda(
    int64_t device_index,
    int64_t total_batches,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t seq_len) {
    const int device = static_cast<int>(device_index);
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
    const int block_threads = masked_tiled_block_threads(tile_size);
    const int64_t shared_bytes = causal_machine_scan_forward_masked_tiled_chunk_shared_bytes_cuda(
        num_states,
        tile_size);
    const auto diag = describe_kernel_launch(
        causal_machine_forward_masked_tiled_chunk_kernel<float>,
        device,
        block_threads,
        static_cast<size_t>(shared_bytes));
    const int64_t worker_blocks = persistent_worker_blocks(device, static_cast<int>(total_batches));
    const int64_t preferred_load_bytes = preferred_float_load_bytes(tile_size);
    const int64_t elements_per_load = elements_per_float_load(tile_size);
    const int64_t candidate_bytes = estimate_persisting_l2_candidate_bytes_for_tiled_path(
        num_states,
        transition_rank);
    const int64_t effective_bytes = estimate_persisting_l2_effective_bytes(device, candidate_bytes);
    const int64_t free_mem_bytes = std::max<int64_t>(current_free_global_mem(), 1);
    const int64_t total_mem_bytes = std::max<int64_t>(cached_total_global_mem(device), 1);
    return {
        worker_blocks,
        shared_bytes,
        diag.block_threads,
        preferred_load_bytes,
        elements_per_load,
        static_cast<int64_t>(can_use_vectorized_float_io(tile_size)),
        static_cast<int64_t>(can_use_async_memcpy(device)),
        diag.active_blocks_per_sm,
        diag.active_warps_per_sm,
        diag.max_warps_per_sm,
        diag.occupancy_pct,
        diag.registers_per_thread,
        diag.static_smem_bytes,
        candidate_bytes,
        effective_bytes,
        static_cast<int64_t>(effective_bytes > 0),
        estimate_masked_tiled_forward_bytes_moved(total_batches, seq_len, num_states, transition_rank),
        estimate_masked_tiled_forward_sync_points(seq_len, num_states, tile_size),
        free_mem_bytes,
        total_mem_bytes,
    };
}

std::vector<int64_t> causal_machine_scan_describe_tiled_backward_runtime_cuda(
    int64_t device_index,
    int64_t total_batches,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t split_size,
    int64_t seq_len) {
    const int device = static_cast<int>(device_index);
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
    const int64_t total_mem_bytes = std::max<int64_t>(cached_total_global_mem(device), 1);
    const int64_t free_mem_bytes = std::max<int64_t>(current_free_global_mem(), 1);
    const auto runtime_config = make_tiled_backward_runtime_config(
        device,
        total_batches,
        num_states,
        transition_rank,
        tile_size);
    const int64_t launch_worker_blocks = tiled_backward_launch_worker_blocks(
        device,
        runtime_config.launch_batch_size,
        runtime_config.staging_worker_blocks);
    const int block_threads = tiled_block_threads(tile_size, split_size);
    const bool sm90_path = use_sm90_tiled_kernel_family(device);
    const int64_t shared_bytes = sm90_path
        ? static_cast<int64_t>(backward_tiled_shared_bytes_sm90(
            static_cast<int>(num_states),
            static_cast<int>(split_size),
            static_cast<int>(tile_size),
            block_threads))
        : causal_machine_scan_backward_tiled_chunk_shared_bytes_cuda(
            num_states,
            split_size,
            tile_size);
    const auto diag = sm90_path
        ? describe_kernel_launch(
            causal_machine_backward_tiled_chunk_kernel<float, true>,
            device,
            block_threads,
            static_cast<size_t>(shared_bytes))
        : describe_kernel_launch(
            causal_machine_backward_tiled_chunk_kernel<float, false>,
            device,
            block_threads,
            static_cast<size_t>(shared_bytes));
    const int64_t preferred_load_bytes = causal_machine_scan_preferred_load_bytes_cuda(
        num_states,
        tile_size,
        transition_rank);
    const int64_t elements_per_load = causal_machine_scan_elements_per_load_cuda(
        num_states,
        tile_size,
        transition_rank);
    const int64_t candidate_bytes = estimate_persisting_l2_candidate_bytes_for_tiled_path(
        num_states,
        transition_rank);
    const int64_t effective_bytes = estimate_persisting_l2_effective_bytes(device, candidate_bytes);
    const int64_t tensor_core_math = static_cast<int64_t>(
        can_use_tiled_forward_tensor_core_math(
            device,
            num_states,
            transition_rank,
            tile_size,
            split_size));
    return {
        runtime_config.launch_batch_size,
        runtime_config.staging_worker_blocks,
        runtime_config.staging_budget_bytes,
        runtime_config.per_worker_bytes,
        launch_worker_blocks,
        free_mem_bytes,
        total_mem_bytes,
        shared_bytes,
        diag.block_threads,
        diag.active_blocks_per_sm,
        diag.active_warps_per_sm,
        diag.max_warps_per_sm,
        diag.occupancy_pct,
        diag.registers_per_thread,
        diag.static_smem_bytes,
        preferred_load_bytes,
        elements_per_load,
        static_cast<int64_t>(causal_machine_scan_can_use_vectorized_io_cuda(num_states, tile_size, transition_rank)),
        static_cast<int64_t>(causal_machine_scan_can_use_async_memcpy_cuda(device_index)),
        tensor_core_math,
        candidate_bytes,
        effective_bytes,
        static_cast<int64_t>(effective_bytes > 0),
        estimate_tiled_backward_bytes_moved(total_batches, seq_len, num_states, transition_rank),
        estimate_tiled_backward_sync_points(seq_len, num_states, transition_rank, tile_size, split_size),
    };
}

std::vector<int64_t> causal_machine_scan_describe_masked_tiled_backward_runtime_cuda(
    int64_t device_index,
    int64_t total_batches,
    int64_t num_states,
    int64_t tile_size,
    int64_t seq_len) {
    const int device = static_cast<int>(device_index);
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
    const int64_t total_mem_bytes = std::max<int64_t>(cached_total_global_mem(device), 1);
    const int64_t free_mem_bytes = std::max<int64_t>(current_free_global_mem(), 1);
    const int64_t worker_blocks = persistent_worker_blocks(device, static_cast<int>(total_batches));
    const int block_threads = masked_tiled_block_threads(tile_size);
    const int64_t shared_bytes = causal_machine_scan_backward_masked_tiled_chunk_shared_bytes_cuda(
        num_states,
        tile_size);
    const auto diag = describe_kernel_launch(
        causal_machine_backward_masked_tiled_chunk_kernel<float>,
        device,
        block_threads,
        static_cast<size_t>(shared_bytes));
    const int64_t preferred_load_bytes = preferred_float_load_bytes(tile_size);
    const int64_t elements_per_load = elements_per_float_load(tile_size);
    const int64_t candidate_bytes = estimate_persisting_l2_candidate_bytes_for_tiled_path(
        num_states,
        num_states);
    const int64_t effective_bytes = estimate_persisting_l2_effective_bytes(device, candidate_bytes);
    return {
        worker_blocks,
        shared_bytes,
        diag.block_threads,
        diag.active_blocks_per_sm,
        diag.active_warps_per_sm,
        diag.max_warps_per_sm,
        diag.occupancy_pct,
        diag.registers_per_thread,
        diag.static_smem_bytes,
        preferred_load_bytes,
        elements_per_load,
        static_cast<int64_t>(can_use_vectorized_float_io(tile_size)),
        static_cast<int64_t>(can_use_async_memcpy(device)),
        candidate_bytes,
        effective_bytes,
        static_cast<int64_t>(effective_bytes > 0),
        estimate_masked_tiled_backward_bytes_moved(total_batches, seq_len, num_states, num_states),
        estimate_masked_tiled_backward_sync_points(seq_len, num_states, tile_size),
        free_mem_bytes,
        total_mem_bytes,
    };
}

torch::Tensor causal_machine_scan_unpack_int8_cuda(
    torch::Tensor packed,
    torch::Tensor scales);
torch::Tensor causal_machine_scan_unpack_fp8_e4m3_cuda(
    torch::Tensor packed,
    torch::Tensor scales);
torch::Tensor causal_machine_scan_unpack_fp8_e5m2_cuda(
    torch::Tensor packed,
    torch::Tensor scales);

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor filtered_value_cache,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_quantized_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor filtered_value_cache,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_fp8_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor filtered_value_cache,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    torch::Tensor work_queue_counter,
    torch::Tensor masked_transition_tile_cache,
    torch::Tensor filtered_value_cache,
    torch::Tensor row_sums,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks_cuda(
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size);

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor filtered_value_cache,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(
        initial_log_belief.sizes(),
        local_logits.options());
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief.to(final_log_belief.options()));
        return {beliefs, final_log_belief};
    }
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_tiled_logits_kernel_workspace",
        [&] {
            launch_forward_tiled_chunk<scalar_t>(
                local_logits,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                score_threshold,
                score_topk,
                tile_size,
                split_size,
                0,
                launch_chunk_len,
                work_queue_counter,
                filtered_value_cache,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_quantized_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(initial_log_belief.sizes(), local_logits.options());
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief.to(final_log_belief.options()));
        return {beliefs, final_log_belief};
    }
    auto launch_config = make_forward_tiled_launch_config(
        local_logits,
        static_cast<int>(tile_size),
        static_cast<int>(split_size));
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto filtered_value_cache = torch::empty(
        {launch_config.grid.x, local_logits.size(2)},
        transition_source_scales.options().dtype(torch::kFloat32));
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_tiled_quantized_kernel",
        [&] {
            launch_forward_tiled_chunk_packed<scalar_t, int8_t, PackedTransitionFormat::Int8>(
                local_logits,
                transition_source_q,
                transition_source_scales,
                transition_dest_q,
                transition_dest_scales,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                score_threshold,
                score_topk,
                tile_size,
                split_size,
                0,
                launch_chunk_len,
                work_queue_counter,
                filtered_value_cache,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_quantized_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor filtered_value_cache,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(initial_log_belief.sizes(), local_logits.options());
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief.to(final_log_belief.options()));
        return {beliefs, final_log_belief};
    }
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_tiled_quantized_kernel_workspace",
        [&] {
            launch_forward_tiled_chunk_packed<scalar_t, int8_t, PackedTransitionFormat::Int8>(
                local_logits,
                transition_source_q,
                transition_source_scales,
                transition_dest_q,
                transition_dest_scales,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                score_threshold,
                score_topk,
                tile_size,
                split_size,
                0,
                launch_chunk_len,
                work_queue_counter,
                filtered_value_cache,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_fp8_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(initial_log_belief.sizes(), local_logits.options());
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief.to(final_log_belief.options()));
        return {beliefs, final_log_belief};
    }
    auto launch_config = make_forward_tiled_launch_config(
        local_logits,
        static_cast<int>(tile_size),
        static_cast<int>(split_size));
    auto work_queue_counter = make_device_work_queue_counter(local_logits);
    auto filtered_value_cache = torch::empty(
        {launch_config.grid.x, local_logits.size(2)},
        transition_source_scales.options().dtype(torch::kFloat32));
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_tiled_fp8_kernel",
        [&] {
            if (fp8_format == 0) {
                launch_forward_tiled_chunk_packed<scalar_t, uint8_t, PackedTransitionFormat::Fp8E4M3>(
                    local_logits, transition_source_packed, transition_source_scales, transition_dest_packed, transition_dest_scales,
                    transition_context, initial_log_belief, transition_gate, transition_stay_probs, seq_lens,
                    score_clamp_min, score_clamp_max, score_threshold, score_topk,
                    tile_size, split_size, 0, launch_chunk_len, work_queue_counter, filtered_value_cache, beliefs, final_log_belief);
            } else {
                launch_forward_tiled_chunk_packed<scalar_t, uint8_t, PackedTransitionFormat::Fp8E5M2>(
                    local_logits, transition_source_packed, transition_source_scales, transition_dest_packed, transition_dest_scales,
                    transition_context, initial_log_belief, transition_gate, transition_stay_probs, seq_lens,
                    score_clamp_min, score_clamp_max, score_threshold, score_topk,
                    tile_size, split_size, 0, launch_chunk_len, work_queue_counter, filtered_value_cache, beliefs, final_log_belief);
            }
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_fp8_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor filtered_value_cache,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(initial_log_belief.sizes(), local_logits.options());
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief.to(final_log_belief.options()));
        return {beliefs, final_log_belief};
    }
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_tiled_fp8_kernel_workspace",
        [&] {
            if (fp8_format == 0) {
                launch_forward_tiled_chunk_packed<scalar_t, uint8_t, PackedTransitionFormat::Fp8E4M3>(
                    local_logits, transition_source_packed, transition_source_scales, transition_dest_packed, transition_dest_scales,
                    transition_context, initial_log_belief, transition_gate, transition_stay_probs, seq_lens,
                    score_clamp_min, score_clamp_max, score_threshold, score_topk,
                    tile_size, split_size, 0, launch_chunk_len, work_queue_counter, filtered_value_cache, beliefs, final_log_belief);
            } else {
                launch_forward_tiled_chunk_packed<scalar_t, uint8_t, PackedTransitionFormat::Fp8E5M2>(
                    local_logits, transition_source_packed, transition_source_scales, transition_dest_packed, transition_dest_scales,
                    transition_context, initial_log_belief, transition_gate, transition_stay_probs, seq_lens,
                    score_clamp_min, score_clamp_max, score_threshold, score_topk,
                    tile_size, split_size, 0, launch_chunk_len, work_queue_counter, filtered_value_cache, beliefs, final_log_belief);
            }
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    torch::Tensor work_queue_counter,
    torch::Tensor masked_transition_tile_cache,
    torch::Tensor filtered_value_cache,
    torch::Tensor row_sums,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(
        initial_log_belief.sizes(),
        local_logits.options());
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief.to(final_log_belief.options()));
        return {beliefs, final_log_belief};
    }
    prepare_masked_transition_row_sums(
        transition_source_logits,
        transition_dest_logits,
        transition_mask,
        row_sums);
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_masked_tiled_logits_kernel_workspace",
        [&] {
            launch_forward_masked_tiled_chunk<scalar_t>(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                row_sums,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                transition_mask,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                score_threshold,
                score_topk,
                tile_size,
                0,
                launch_chunk_len,
                work_queue_counter,
                masked_transition_tile_cache,
                filtered_value_cache,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(
        initial_log_belief.sizes(),
        local_logits.options());
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief.to(final_log_belief.options()));
        return {beliefs, final_log_belief};
    }
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_tiled_logits_kernel",
        [&] {
            launch_forward_tiled_chunk<scalar_t>(
                local_logits,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                score_threshold,
                score_topk,
                tile_size,
                split_size,
                0,
                launch_chunk_len,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(
        initial_log_belief.sizes(),
        local_logits.options());
    auto row_sums = torch::empty(
        {transition_mask.size(0)},
        transition_source_logits.options().dtype(torch::kFloat32));
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief.to(final_log_belief.options()));
        return {beliefs, final_log_belief};
    }
    prepare_masked_transition_row_sums(
        transition_source_logits,
        transition_dest_logits,
        transition_mask,
        row_sums);
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_masked_tiled_logits_kernel",
        [&] {
            launch_forward_masked_tiled_chunk<scalar_t>(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                row_sums,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                transition_mask,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                score_threshold,
                score_topk,
                tile_size,
                0,
                launch_chunk_len,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

namespace {

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_aten_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max) {
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    const auto num_states = beliefs.size(2);
    const auto transition_rank = transition_source_probs.size(1);
    const bool has_seq_lens = seq_lens.defined() && seq_lens.numel() > 0;

    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_source_probs = torch::zeros_like(transition_source_probs);
    auto grad_transition_dest_probs = torch::zeros_like(transition_dest_probs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, transition_source_probs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros_like(transition_stay_probs);

    if (batch_size == 0 || seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief.to(grad_initial_log_belief.options()));
        return {
            grad_local_logits,
            grad_transition_source_probs,
            grad_transition_dest_probs,
            grad_transition_context,
            grad_initial_log_belief,
            grad_transition_gate,
            grad_transition_stay,
        };
    }

    const auto float_options = transition_source_probs.options().dtype(torch::kFloat32);
    auto grad_beliefs_f32 = grad_beliefs.to(float_options);
    auto grad_final_belief_f32 = grad_final_belief.to(float_options);
    auto transition_context_f32 = transition_context.to(float_options);
    auto initial_log_belief_f32 = initial_log_belief.to(float_options);
    auto beliefs_f32 = beliefs.to(float_options);
    auto stay_probs = transition_stay_probs.to(float_options).view({1, num_states});
    auto one_minus_stay = 1.0f - stay_probs;
    auto carry = grad_final_belief_f32.clone();
    const float gate_value = static_cast<float>(transition_gate);
    const int64_t rank_tile = std::max<int64_t>(1, split_size);
    const int64_t effective_chunk_size = std::max<int64_t>(1, chunk_size);
    const int64_t state_tile_size = std::max<int64_t>(1, tile_size);

    for (int64_t chunk_end = seq_len; chunk_end > 0; chunk_end -= effective_chunk_size) {
        const int64_t chunk_start = std::max<int64_t>(0, chunk_end - effective_chunk_size);
        for (int64_t pos = chunk_end - 1; pos >= chunk_start; --pos) {
            auto q_prob = beliefs_f32.select(1, pos).exp();
            auto prev_log = pos == 0 ? initial_log_belief_f32 : beliefs_f32.select(1, pos - 1);
            auto prev_probs = prev_log.exp();
            auto transition_context_t = transition_context_f32.select(1, pos);
            auto gq = grad_beliefs_f32.select(1, pos) + carry;
            auto gq_sum = gq.sum(-1, true);

            torch::Tensor active_f;
            if (has_seq_lens) {
                active_f = seq_lens.gt(pos).view({batch_size, 1}).to(float_options);
            } else {
                active_f = torch::ones({batch_size, 1}, float_options);
            }

            std::vector<torch::Tensor> latent_splits;
            std::vector<torch::Tensor> grad_latent_splits;
            latent_splits.reserve(static_cast<size_t>((transition_rank + rank_tile - 1) / rank_tile));
            grad_latent_splits.reserve(static_cast<size_t>((transition_rank + rank_tile - 1) / rank_tile));
            for (int64_t rank_start = 0; rank_start < transition_rank; rank_start += rank_tile) {
                const int64_t rank_end = std::min<int64_t>(transition_rank, rank_start + rank_tile);
                auto source_split = transition_source_probs.slice(1, rank_start, rank_end);
                auto latent_split = torch::matmul(prev_probs, source_split);
                latent_splits.push_back(latent_split);
                grad_latent_splits.push_back(torch::zeros_like(latent_split));
            }

            auto grad_prev_probs = torch::zeros_like(prev_probs);
            auto grad_local_t = grad_local_logits.select(1, pos);
            auto grad_context_t = grad_transition_context.select(1, pos);
            for (int64_t state_start = 0; state_start < num_states; state_start += state_tile_size) {
                const int64_t state_end = std::min<int64_t>(num_states, state_start + state_tile_size);
                auto prev_probs_tile = prev_probs.slice(1, state_start, state_end);
                auto stay_tile = stay_probs.slice(1, state_start, state_end);
                auto q_prob_tile = q_prob.slice(1, state_start, state_end);
                auto gq_tile = gq.slice(1, state_start, state_end);
                auto context_tile = transition_context_t.slice(1, state_start, state_end);
                auto mix_probs_tile = torch::zeros({batch_size, state_end - state_start}, float_options);

                for (int64_t rank_start = 0, split_idx = 0; rank_start < transition_rank; rank_start += rank_tile, ++split_idx) {
                    const int64_t rank_end = std::min<int64_t>(transition_rank, rank_start + rank_tile);
                    auto dest_split_tile = transition_dest_probs.slice(0, rank_start, rank_end).slice(1, state_start, state_end);
                    mix_probs_tile.add_(torch::matmul(latent_splits[static_cast<size_t>(split_idx)], dest_split_tile));
                }

                auto pred_probs_tile = stay_tile * prev_probs_tile + (1.0f - stay_tile) * mix_probs_tile;
                pred_probs_tile = pred_probs_tile.clamp_min(1.0e-20f);
                auto pred_log_tile = pred_probs_tile.log();
                auto ga_tile = (gq_tile - q_prob_tile * gq_sum) * active_f;
                auto prior_tile = pred_log_tile + context_tile;
                auto clamped_prior_tile = prior_tile.clamp(score_clamp_min, score_clamp_max);
                auto clamp_grad_tile =
                    prior_tile.ge(score_clamp_min).logical_and(prior_tile.le(score_clamp_max)).to(float_options);

                grad_local_t.slice(1, state_start, state_end).copy_(ga_tile.to(grad_local_t.scalar_type()));
                auto grad_prior_tile = gate_value * ga_tile * clamp_grad_tile;
                grad_context_t.slice(1, state_start, state_end).copy_(grad_prior_tile.to(grad_context_t.scalar_type()));
                grad_transition_gate.add_((ga_tile * clamped_prior_tile).sum().view({1}));

                auto grad_pred_prob_tile = active_f * (grad_prior_tile / pred_probs_tile);
                grad_transition_stay.slice(0, state_start, state_end).add_(
                    (grad_pred_prob_tile * (prev_probs_tile - mix_probs_tile)).sum(0));
                grad_prev_probs.slice(1, state_start, state_end).add_(grad_pred_prob_tile * stay_tile);
                auto grad_mix_tile = grad_pred_prob_tile * (1.0f - stay_tile);

                for (int64_t rank_start = 0, split_idx = 0; rank_start < transition_rank; rank_start += rank_tile, ++split_idx) {
                    const int64_t rank_end = std::min<int64_t>(transition_rank, rank_start + rank_tile);
                    auto dest_split_tile = transition_dest_probs.slice(0, rank_start, rank_end).slice(1, state_start, state_end);
                    auto grad_latent_split = torch::matmul(grad_mix_tile, dest_split_tile.transpose(0, 1));
                    grad_transition_dest_probs.slice(0, rank_start, rank_end).slice(1, state_start, state_end).add_(
                        torch::matmul(latent_splits[static_cast<size_t>(split_idx)].transpose(0, 1), grad_mix_tile));
                    grad_latent_splits[static_cast<size_t>(split_idx)].add_(grad_latent_split);
                }
            }

            for (int64_t rank_start = 0, split_idx = 0; rank_start < transition_rank; rank_start += rank_tile, ++split_idx) {
                const int64_t rank_end = std::min<int64_t>(transition_rank, rank_start + rank_tile);
                auto source_split = transition_source_probs.slice(1, rank_start, rank_end);
                const auto& grad_latent_split = grad_latent_splits[static_cast<size_t>(split_idx)];
                grad_transition_source_probs.slice(1, rank_start, rank_end).add_(
                    torch::matmul(prev_probs.transpose(0, 1), grad_latent_split));
                grad_prev_probs.add_(torch::matmul(grad_latent_split, source_split.transpose(0, 1)));
            }

            auto grad_prev_log = grad_prev_probs * prev_probs;
            carry = active_f * grad_prev_log + (1.0f - active_f) * gq;
        }
    }

    grad_initial_log_belief.copy_(carry.to(grad_initial_log_belief.options()));
    return {
        grad_local_logits,
        grad_transition_source_probs,
        grad_transition_dest_probs,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate,
        grad_transition_stay,
    };
}

}  // namespace

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_kernel_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor latent_cache_staging,
    torch::Tensor grad_latent_accum_staging,
    torch::Tensor grad_transition_source_probs_staging,
    torch::Tensor grad_transition_dest_probs_staging,
    torch::Tensor grad_transition_gate_staging,
    torch::Tensor grad_transition_stay_staging,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_quantized_kernel_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor latent_cache_staging,
    torch::Tensor grad_latent_accum_staging,
    torch::Tensor grad_transition_source_probs_staging,
    torch::Tensor grad_transition_dest_probs_staging,
    torch::Tensor grad_transition_gate_staging,
    torch::Tensor grad_transition_stay_staging,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_fp8_kernel_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor latent_cache_staging,
    torch::Tensor grad_latent_accum_staging,
    torch::Tensor grad_transition_source_probs_staging,
    torch::Tensor grad_transition_dest_probs_staging,
    torch::Tensor grad_transition_gate_staging,
    torch::Tensor grad_transition_stay_staging,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_kernel_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);

    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_source_probs = torch::zeros_like(transition_source_probs);
    auto grad_transition_dest_probs = torch::zeros_like(transition_dest_probs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, transition_source_probs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros_like(transition_stay_probs);

    if (batch_size == 0 || seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief.to(grad_initial_log_belief.options()));
        return {
            grad_local_logits,
            grad_transition_source_probs,
            grad_transition_dest_probs,
            grad_transition_context,
            grad_initial_log_belief,
            grad_transition_gate,
            grad_transition_stay,
        };
    }

    const bool can_use_custom_kernel = causal_machine_scan_can_use_tiled_backward_kernel_cuda(
        beliefs.get_device(),
        beliefs.size(2),
        tile_size,
        split_size);
    TORCH_CHECK(
        can_use_custom_kernel,
        "causal_machine_scan tiled backward custom kernel is not supported for this runtime config");

    const int64_t sequence_tile_size = seq_len == 0
        ? 0
        : std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    const auto runtime_config = make_tiled_backward_runtime_config(
        beliefs.get_device(),
        batch_size,
        beliefs.size(2),
        transition_source_probs.size(1),
        tile_size);
    const int64_t staging_worker_blocks = runtime_config.staging_worker_blocks;
    auto grad_transition_source_probs_staging = torch::zeros(
        {staging_worker_blocks, transition_source_probs.size(0), transition_source_probs.size(1)},
        transition_source_probs.options());
    auto grad_transition_dest_probs_staging = torch::zeros(
        {staging_worker_blocks, transition_dest_probs.size(0), transition_dest_probs.size(1)},
        transition_dest_probs.options());
    auto latent_cache_staging = torch::empty(
        {staging_worker_blocks, transition_source_probs.size(1)},
        transition_source_probs.options());
    auto grad_latent_accum_staging = torch::empty(
        {staging_worker_blocks, transition_source_probs.size(1)},
        transition_source_probs.options());
    auto grad_transition_gate_staging = torch::zeros(
        {staging_worker_blocks},
        transition_source_probs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_staging = torch::zeros(
        {staging_worker_blocks, transition_stay_probs.size(0)},
        transition_stay_probs.options());
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        beliefs.scalar_type(),
        "causal_machine_scan_backward_tiled_probs",
        [&] {
            const int device_index = beliefs.get_device();
            const int64_t launch_worker_blocks = tiled_backward_launch_worker_blocks(
                device_index,
                batch_size,
                staging_worker_blocks);
            auto grad_transition_source_probs_staging_chunk =
                grad_transition_source_probs_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_dest_probs_staging_chunk =
                grad_transition_dest_probs_staging.narrow(0, 0, launch_worker_blocks);
            auto latent_cache_staging_chunk = latent_cache_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_latent_accum_staging_chunk = grad_latent_accum_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_gate_staging_chunk =
                grad_transition_gate_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_stay_staging_chunk =
                grad_transition_stay_staging.narrow(0, 0, launch_worker_blocks);

            reset_tiled_backward_state_cuda(
                grad_transition_source_probs_staging_chunk,
                grad_transition_dest_probs_staging_chunk,
                grad_transition_gate_staging_chunk,
                grad_transition_stay_staging_chunk,
                work_queue_counter);

            launch_backward_tiled_chunk<scalar_t>(
                grad_beliefs,
                grad_final_belief,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                initial_log_belief,
                beliefs,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                score_threshold,
                score_topk,
                tile_size,
                split_size,
                0,
                sequence_tile_size,
                launch_worker_blocks,
                work_queue_counter,
                latent_cache_staging_chunk,
                grad_latent_accum_staging_chunk,
                grad_local_logits,
                grad_transition_source_probs_staging_chunk,
                grad_transition_dest_probs_staging_chunk,
                grad_transition_source_probs,
                grad_transition_dest_probs,
                grad_transition_context,
                grad_initial_log_belief,
                grad_transition_gate_staging_chunk,
                grad_transition_stay_staging_chunk,
                grad_transition_gate,
                grad_transition_stay);
        });

    return {
        grad_local_logits,
        grad_transition_source_probs,
        grad_transition_dest_probs,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate,
        grad_transition_stay,
    };
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_quantized_kernel_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_source_probs = torch::zeros({beliefs.size(2), transition_source_q.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs = torch::zeros({transition_dest_q.size(0), beliefs.size(2)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros_like(transition_stay_probs);
    if (batch_size == 0 || seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief.to(grad_initial_log_belief.options()));
        return {grad_local_logits, grad_transition_source_probs, grad_transition_dest_probs, grad_transition_context, grad_initial_log_belief, grad_transition_gate, grad_transition_stay};
    }
    const auto runtime_config = make_tiled_backward_runtime_config(
        beliefs.get_device(), batch_size, beliefs.size(2), transition_source_q.size(1), tile_size);
    const int64_t staging_worker_blocks = runtime_config.staging_worker_blocks;
    auto grad_transition_source_probs_staging = torch::zeros({staging_worker_blocks, beliefs.size(2), transition_source_q.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs_staging = torch::zeros({staging_worker_blocks, transition_dest_q.size(0), beliefs.size(2)}, beliefs.options().dtype(torch::kFloat32));
    auto latent_cache_staging = torch::empty({staging_worker_blocks, transition_source_q.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_latent_accum_staging = torch::empty({staging_worker_blocks, transition_source_q.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_gate_staging = torch::zeros({staging_worker_blocks}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_staging = torch::zeros({staging_worker_blocks, beliefs.size(2)}, beliefs.options().dtype(torch::kFloat32));
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    const int64_t launch_worker_blocks = tiled_backward_launch_worker_blocks(
        beliefs.get_device(), runtime_config.launch_batch_size, runtime_config.staging_worker_blocks);
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        beliefs.scalar_type(),
        "causal_machine_scan_backward_tiled_quantized_kernel",
        [&] {
            launch_backward_tiled_chunk_packed<scalar_t, int8_t, PackedTransitionFormat::Int8>(
                grad_beliefs, grad_final_belief, transition_source_q, transition_source_scales, transition_dest_q, transition_dest_scales,
                transition_context, initial_log_belief, beliefs, transition_gate, transition_stay_probs, seq_lens,
                score_clamp_min, score_clamp_max, score_threshold, score_topk,
                tile_size, split_size, 0, launch_chunk_len, launch_worker_blocks, work_queue_counter,
                latent_cache_staging, grad_latent_accum_staging, grad_local_logits,
                grad_transition_source_probs_staging, grad_transition_dest_probs_staging,
                grad_transition_source_probs, grad_transition_dest_probs,
                grad_transition_context, grad_initial_log_belief,
                grad_transition_gate_staging, grad_transition_stay_staging,
                grad_transition_gate, grad_transition_stay);
        });
    return {grad_local_logits, grad_transition_source_probs, grad_transition_dest_probs, grad_transition_context, grad_initial_log_belief, grad_transition_gate, grad_transition_stay};
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_quantized_kernel_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor latent_cache_staging,
    torch::Tensor grad_latent_accum_staging,
    torch::Tensor grad_transition_source_probs_staging,
    torch::Tensor grad_transition_dest_probs_staging,
    torch::Tensor grad_transition_gate_staging,
    torch::Tensor grad_transition_stay_staging,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_source_probs = torch::zeros({beliefs.size(2), transition_source_q.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs = torch::zeros({transition_dest_q.size(0), beliefs.size(2)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros_like(transition_stay_probs);
    if (batch_size == 0 || seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief.to(grad_initial_log_belief.options()));
        return {grad_local_logits, grad_transition_source_probs, grad_transition_dest_probs, grad_transition_context, grad_initial_log_belief, grad_transition_gate, grad_transition_stay};
    }
    const auto runtime_config = make_tiled_backward_runtime_config(
        beliefs.get_device(), batch_size, beliefs.size(2), transition_source_q.size(1), tile_size);
    const int64_t requested_worker_blocks = runtime_config.staging_worker_blocks;
    const int64_t available_worker_blocks = std::min<int64_t>({
        std::max<int64_t>(1, latent_cache_staging.size(0)),
        std::max<int64_t>(1, grad_latent_accum_staging.size(0)),
        std::max<int64_t>(1, grad_transition_source_probs_staging.size(0)),
        std::max<int64_t>(1, grad_transition_dest_probs_staging.size(0)),
        std::max<int64_t>(1, grad_transition_gate_staging.size(0)),
        std::max<int64_t>(1, grad_transition_stay_staging.size(0)),
    });
    const int64_t launch_worker_blocks = std::min<int64_t>(
        available_worker_blocks,
        tiled_backward_launch_worker_blocks(
            beliefs.get_device(), batch_size, requested_worker_blocks));
    TORCH_CHECK(launch_worker_blocks > 0, "workspace-backed tiled quantized backward requires at least one worker block");
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        beliefs.scalar_type(),
        "causal_machine_scan_backward_tiled_quantized_kernel_workspace",
        [&] {
            auto grad_transition_source_probs_staging_chunk = grad_transition_source_probs_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_dest_probs_staging_chunk = grad_transition_dest_probs_staging.narrow(0, 0, launch_worker_blocks);
            auto latent_cache_staging_chunk = latent_cache_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_latent_accum_staging_chunk = grad_latent_accum_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_gate_staging_chunk = grad_transition_gate_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_stay_staging_chunk = grad_transition_stay_staging.narrow(0, 0, launch_worker_blocks);
            reset_tiled_backward_state_cuda(
                grad_transition_source_probs_staging_chunk,
                grad_transition_dest_probs_staging_chunk,
                grad_transition_gate_staging_chunk,
                grad_transition_stay_staging_chunk,
                work_queue_counter);
            launch_backward_tiled_chunk_packed<scalar_t, int8_t, PackedTransitionFormat::Int8>(
                grad_beliefs, grad_final_belief, transition_source_q, transition_source_scales, transition_dest_q, transition_dest_scales,
                transition_context, initial_log_belief, beliefs, transition_gate, transition_stay_probs, seq_lens,
                score_clamp_min, score_clamp_max, score_threshold, score_topk,
                tile_size, split_size, 0, launch_chunk_len, launch_worker_blocks, work_queue_counter,
                latent_cache_staging_chunk, grad_latent_accum_staging_chunk, grad_local_logits,
                grad_transition_source_probs_staging_chunk, grad_transition_dest_probs_staging_chunk,
                grad_transition_source_probs, grad_transition_dest_probs,
                grad_transition_context, grad_initial_log_belief,
                grad_transition_gate_staging_chunk, grad_transition_stay_staging_chunk,
                grad_transition_gate, grad_transition_stay);
        });
    return {grad_local_logits, grad_transition_source_probs, grad_transition_dest_probs, grad_transition_context, grad_initial_log_belief, grad_transition_gate, grad_transition_stay};
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_fp8_kernel_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_source_probs = torch::zeros({beliefs.size(2), transition_source_packed.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs = torch::zeros({transition_dest_packed.size(0), beliefs.size(2)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros_like(transition_stay_probs);
    if (batch_size == 0 || seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief.to(grad_initial_log_belief.options()));
        return {grad_local_logits, grad_transition_source_probs, grad_transition_dest_probs, grad_transition_context, grad_initial_log_belief, grad_transition_gate, grad_transition_stay};
    }
    const auto runtime_config = make_tiled_backward_runtime_config(
        beliefs.get_device(), batch_size, beliefs.size(2), transition_source_packed.size(1), tile_size);
    const int64_t staging_worker_blocks = runtime_config.staging_worker_blocks;
    auto grad_transition_source_probs_staging = torch::zeros({staging_worker_blocks, beliefs.size(2), transition_source_packed.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs_staging = torch::zeros({staging_worker_blocks, transition_dest_packed.size(0), beliefs.size(2)}, beliefs.options().dtype(torch::kFloat32));
    auto latent_cache_staging = torch::empty({staging_worker_blocks, transition_source_packed.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_latent_accum_staging = torch::empty({staging_worker_blocks, transition_source_packed.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_gate_staging = torch::zeros({staging_worker_blocks}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_staging = torch::zeros({staging_worker_blocks, beliefs.size(2)}, beliefs.options().dtype(torch::kFloat32));
    auto work_queue_counter = make_device_work_queue_counter(beliefs);
    const int64_t launch_worker_blocks = tiled_backward_launch_worker_blocks(
        beliefs.get_device(), runtime_config.launch_batch_size, runtime_config.staging_worker_blocks);
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        beliefs.scalar_type(),
        "causal_machine_scan_backward_tiled_fp8_kernel",
        [&] {
            if (fp8_format == 0) {
                launch_backward_tiled_chunk_packed<scalar_t, uint8_t, PackedTransitionFormat::Fp8E4M3>(
                    grad_beliefs, grad_final_belief, transition_source_packed, transition_source_scales, transition_dest_packed, transition_dest_scales,
                    transition_context, initial_log_belief, beliefs, transition_gate, transition_stay_probs, seq_lens,
                    score_clamp_min, score_clamp_max, score_threshold, score_topk,
                    tile_size, split_size, 0, launch_chunk_len, launch_worker_blocks, work_queue_counter,
                    latent_cache_staging, grad_latent_accum_staging, grad_local_logits,
                    grad_transition_source_probs_staging, grad_transition_dest_probs_staging,
                    grad_transition_source_probs, grad_transition_dest_probs,
                    grad_transition_context, grad_initial_log_belief,
                    grad_transition_gate_staging, grad_transition_stay_staging,
                    grad_transition_gate, grad_transition_stay);
            } else {
                launch_backward_tiled_chunk_packed<scalar_t, uint8_t, PackedTransitionFormat::Fp8E5M2>(
                    grad_beliefs, grad_final_belief, transition_source_packed, transition_source_scales, transition_dest_packed, transition_dest_scales,
                    transition_context, initial_log_belief, beliefs, transition_gate, transition_stay_probs, seq_lens,
                    score_clamp_min, score_clamp_max, score_threshold, score_topk,
                    tile_size, split_size, 0, launch_chunk_len, launch_worker_blocks, work_queue_counter,
                    latent_cache_staging, grad_latent_accum_staging, grad_local_logits,
                    grad_transition_source_probs_staging, grad_transition_dest_probs_staging,
                    grad_transition_source_probs, grad_transition_dest_probs,
                    grad_transition_context, grad_initial_log_belief,
                    grad_transition_gate_staging, grad_transition_stay_staging,
                    grad_transition_gate, grad_transition_stay);
            }
        });
    return {grad_local_logits, grad_transition_source_probs, grad_transition_dest_probs, grad_transition_context, grad_initial_log_belief, grad_transition_gate, grad_transition_stay};
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_fp8_kernel_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor latent_cache_staging,
    torch::Tensor grad_latent_accum_staging,
    torch::Tensor grad_transition_source_probs_staging,
    torch::Tensor grad_transition_dest_probs_staging,
    torch::Tensor grad_transition_gate_staging,
    torch::Tensor grad_transition_stay_staging,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_source_probs = torch::zeros({beliefs.size(2), transition_source_packed.size(1)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs = torch::zeros({transition_dest_packed.size(0), beliefs.size(2)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros_like(transition_stay_probs);
    if (batch_size == 0 || seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief.to(grad_initial_log_belief.options()));
        return {grad_local_logits, grad_transition_source_probs, grad_transition_dest_probs, grad_transition_context, grad_initial_log_belief, grad_transition_gate, grad_transition_stay};
    }
    const auto runtime_config = make_tiled_backward_runtime_config(
        beliefs.get_device(), batch_size, beliefs.size(2), transition_source_packed.size(1), tile_size);
    const int64_t requested_worker_blocks = runtime_config.staging_worker_blocks;
    const int64_t available_worker_blocks = std::min<int64_t>({
        std::max<int64_t>(1, latent_cache_staging.size(0)),
        std::max<int64_t>(1, grad_latent_accum_staging.size(0)),
        std::max<int64_t>(1, grad_transition_source_probs_staging.size(0)),
        std::max<int64_t>(1, grad_transition_dest_probs_staging.size(0)),
        std::max<int64_t>(1, grad_transition_gate_staging.size(0)),
        std::max<int64_t>(1, grad_transition_stay_staging.size(0)),
    });
    const int64_t launch_worker_blocks = std::min<int64_t>(
        available_worker_blocks,
        tiled_backward_launch_worker_blocks(
            beliefs.get_device(), batch_size, requested_worker_blocks));
    TORCH_CHECK(launch_worker_blocks > 0, "workspace-backed tiled fp8 backward requires at least one worker block");
    const int64_t launch_chunk_len = std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        beliefs.scalar_type(),
        "causal_machine_scan_backward_tiled_fp8_kernel_workspace",
        [&] {
            auto grad_transition_source_probs_staging_chunk = grad_transition_source_probs_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_dest_probs_staging_chunk = grad_transition_dest_probs_staging.narrow(0, 0, launch_worker_blocks);
            auto latent_cache_staging_chunk = latent_cache_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_latent_accum_staging_chunk = grad_latent_accum_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_gate_staging_chunk = grad_transition_gate_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_stay_staging_chunk = grad_transition_stay_staging.narrow(0, 0, launch_worker_blocks);
            reset_tiled_backward_state_cuda(
                grad_transition_source_probs_staging_chunk,
                grad_transition_dest_probs_staging_chunk,
                grad_transition_gate_staging_chunk,
                grad_transition_stay_staging_chunk,
                work_queue_counter);
            if (fp8_format == 0) {
                launch_backward_tiled_chunk_packed<scalar_t, uint8_t, PackedTransitionFormat::Fp8E4M3>(
                    grad_beliefs, grad_final_belief, transition_source_packed, transition_source_scales, transition_dest_packed, transition_dest_scales,
                    transition_context, initial_log_belief, beliefs, transition_gate, transition_stay_probs, seq_lens,
                    score_clamp_min, score_clamp_max, score_threshold, score_topk,
                    tile_size, split_size, 0, launch_chunk_len, launch_worker_blocks, work_queue_counter,
                    latent_cache_staging_chunk, grad_latent_accum_staging_chunk, grad_local_logits,
                    grad_transition_source_probs_staging_chunk, grad_transition_dest_probs_staging_chunk,
                    grad_transition_source_probs, grad_transition_dest_probs,
                    grad_transition_context, grad_initial_log_belief,
                    grad_transition_gate_staging_chunk, grad_transition_stay_staging_chunk,
                    grad_transition_gate, grad_transition_stay);
            } else {
                launch_backward_tiled_chunk_packed<scalar_t, uint8_t, PackedTransitionFormat::Fp8E5M2>(
                    grad_beliefs, grad_final_belief, transition_source_packed, transition_source_scales, transition_dest_packed, transition_dest_scales,
                    transition_context, initial_log_belief, beliefs, transition_gate, transition_stay_probs, seq_lens,
                    score_clamp_min, score_clamp_max, score_threshold, score_topk,
                    tile_size, split_size, 0, launch_chunk_len, launch_worker_blocks, work_queue_counter,
                    latent_cache_staging_chunk, grad_latent_accum_staging_chunk, grad_local_logits,
                    grad_transition_source_probs_staging_chunk, grad_transition_dest_probs_staging_chunk,
                    grad_transition_source_probs, grad_transition_dest_probs,
                    grad_transition_context, grad_initial_log_belief,
                    grad_transition_gate_staging_chunk, grad_transition_stay_staging_chunk,
                    grad_transition_gate, grad_transition_stay);
            }
        });
    return {grad_local_logits, grad_transition_source_probs, grad_transition_dest_probs, grad_transition_context, grad_initial_log_belief, grad_transition_gate, grad_transition_stay};
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_kernel_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor latent_cache_staging,
    torch::Tensor grad_latent_accum_staging,
    torch::Tensor grad_transition_source_probs_staging,
    torch::Tensor grad_transition_dest_probs_staging,
    torch::Tensor grad_transition_gate_staging,
    torch::Tensor grad_transition_stay_staging,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);

    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_source_probs = torch::zeros_like(transition_source_probs);
    auto grad_transition_dest_probs = torch::zeros_like(transition_dest_probs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, transition_source_probs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros_like(transition_stay_probs);

    if (batch_size == 0 || seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief.to(grad_initial_log_belief.options()));
        return {
            grad_local_logits,
            grad_transition_source_probs,
            grad_transition_dest_probs,
            grad_transition_context,
            grad_initial_log_belief,
            grad_transition_gate,
            grad_transition_stay,
        };
    }

    const bool can_use_custom_kernel = causal_machine_scan_can_use_tiled_backward_kernel_cuda(
        beliefs.get_device(),
        beliefs.size(2),
        tile_size,
        split_size);
    TORCH_CHECK(
        can_use_custom_kernel,
        "causal_machine_scan tiled backward custom kernel is not supported for this runtime config");

    const int64_t sequence_tile_size = seq_len == 0
        ? 0
        : std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    const auto runtime_config = make_tiled_backward_runtime_config(
        beliefs.get_device(),
        batch_size,
        beliefs.size(2),
        transition_source_probs.size(1),
        tile_size);
    const int64_t requested_worker_blocks = runtime_config.staging_worker_blocks;
    const int64_t available_worker_blocks = std::min<int64_t>({
        std::max<int64_t>(1, latent_cache_staging.size(0)),
        std::max<int64_t>(1, grad_latent_accum_staging.size(0)),
        std::max<int64_t>(1, grad_transition_source_probs_staging.size(0)),
        std::max<int64_t>(1, grad_transition_dest_probs_staging.size(0)),
        std::max<int64_t>(1, grad_transition_gate_staging.size(0)),
        std::max<int64_t>(1, grad_transition_stay_staging.size(0)),
    });
    const int64_t launch_worker_blocks = std::min<int64_t>(
        available_worker_blocks,
        tiled_backward_launch_worker_blocks(
            beliefs.get_device(),
            batch_size,
            requested_worker_blocks));
    TORCH_CHECK(launch_worker_blocks > 0, "workspace-backed tiled backward requires at least one worker block");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        beliefs.scalar_type(),
        "causal_machine_scan_backward_tiled_probs_workspace",
        [&] {
            auto grad_transition_source_probs_staging_chunk =
                grad_transition_source_probs_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_dest_probs_staging_chunk =
                grad_transition_dest_probs_staging.narrow(0, 0, launch_worker_blocks);
            auto latent_cache_staging_chunk = latent_cache_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_latent_accum_staging_chunk = grad_latent_accum_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_gate_staging_chunk =
                grad_transition_gate_staging.narrow(0, 0, launch_worker_blocks);
            auto grad_transition_stay_staging_chunk =
                grad_transition_stay_staging.narrow(0, 0, launch_worker_blocks);

            reset_tiled_backward_state_cuda(
                grad_transition_source_probs_staging_chunk,
                grad_transition_dest_probs_staging_chunk,
                grad_transition_gate_staging_chunk,
                grad_transition_stay_staging_chunk,
                work_queue_counter);

            launch_backward_tiled_chunk<scalar_t>(
                grad_beliefs,
                grad_final_belief,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                initial_log_belief,
                beliefs,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                score_threshold,
                score_topk,
                tile_size,
                split_size,
                0,
                sequence_tile_size,
                launch_worker_blocks,
                work_queue_counter,
                latent_cache_staging_chunk,
                grad_latent_accum_staging_chunk,
                grad_local_logits,
                grad_transition_source_probs_staging_chunk,
                grad_transition_dest_probs_staging_chunk,
                grad_transition_source_probs,
                grad_transition_dest_probs,
                grad_transition_context,
                grad_initial_log_belief,
                grad_transition_gate_staging_chunk,
                grad_transition_stay_staging_chunk,
                grad_transition_gate,
                grad_transition_stay);
        });

    return {
        grad_local_logits,
        grad_transition_source_probs,
        grad_transition_dest_probs,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate,
        grad_transition_stay,
    };
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    const bool can_use_custom_kernel = causal_machine_scan_can_use_tiled_backward_kernel_cuda(
        beliefs.get_device(),
        beliefs.size(2),
        tile_size,
        split_size);
    if (!can_use_custom_kernel) {
        TORCH_CHECK(
            !std::isfinite(score_threshold) && score_topk <= 0,
            "ATen tiled backward fallback does not support native threshold/topk filtering"
        );
        return causal_machine_scan_backward_tiled_probs_aten_cuda(
            grad_beliefs,
            grad_final_belief,
            transition_source_probs,
            transition_dest_probs,
            transition_context,
            initial_log_belief,
            beliefs,
            transition_gate,
            transition_stay_probs,
            seq_lens,
            chunk_size,
            tile_size,
            split_size,
            score_clamp_min,
            score_clamp_max);
    }
    return causal_machine_scan_backward_tiled_probs_kernel_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate,
        transition_stay_probs,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

template <typename packed_t, PackedTransitionFormat Format>
std::vector<torch::Tensor> causal_machine_scan_pack_transition_table_cuda(torch::Tensor table) {
    c10::cuda::CUDAGuard device_guard(table.device());
    TORCH_CHECK(table.dim() == 2, "packed transition table input must be 2D");
    TORCH_CHECK(table.scalar_type() == torch::kFloat32, "packed transition table input must be float32");
    auto packed = torch::empty(
        table.sizes(),
        table.options().dtype(
            Format == PackedTransitionFormat::Int8 ? torch::kInt8 : torch::kUInt8));
    auto scales = torch::empty({table.size(0)}, table.options().dtype(torch::kFloat32));
    const dim3 grid(static_cast<unsigned int>(table.size(0)));
    const dim3 block(kMaxNumStates);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    pack_transition_table_per_row_kernel<packed_t, Format><<<grid, block, 0, stream>>>(
        table.data_ptr<float>(),
        static_cast<int>(table.size(1)),
        packed.data_ptr<packed_t>(),
        scales.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {packed, scales};
}

std::vector<torch::Tensor> causal_machine_scan_pack_int8_cuda(torch::Tensor table) {
    return causal_machine_scan_pack_transition_table_cuda<int8_t, PackedTransitionFormat::Int8>(table);
}

std::vector<torch::Tensor> causal_machine_scan_pack_fp8_e4m3_cuda(torch::Tensor table) {
    return causal_machine_scan_pack_transition_table_cuda<uint8_t, PackedTransitionFormat::Fp8E4M3>(table);
}

std::vector<torch::Tensor> causal_machine_scan_pack_fp8_e5m2_cuda(torch::Tensor table) {
    return causal_machine_scan_pack_transition_table_cuda<uint8_t, PackedTransitionFormat::Fp8E5M2>(table);
}

template <typename packed_t, PackedTransitionFormat Format>
torch::Tensor causal_machine_scan_unpack_transition_table_cuda(
    torch::Tensor packed,
    torch::Tensor scales) {
    c10::cuda::CUDAGuard device_guard(packed.device());
    TORCH_CHECK(packed.dim() == 2, "packed transition table input must be 2D");
    TORCH_CHECK(scales.dim() == 1, "packed transition table scales must be 1D");
    TORCH_CHECK(packed.is_cuda(), "packed transition table input must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "packed transition table scales must be a CUDA tensor");
    TORCH_CHECK(scales.is_contiguous(), "packed transition table scales must be contiguous");
    TORCH_CHECK(packed.is_contiguous(), "packed transition table input must be contiguous");
    TORCH_CHECK(
        scales.get_device() == packed.get_device(),
        "packed transition table scales must be on the same CUDA device as the packed table");
    TORCH_CHECK(
        scales.size(0) == packed.size(0),
        "packed transition table scales must have one entry per row");
    auto unpacked = torch::empty(
        packed.sizes(),
        packed.options().dtype(torch::kFloat32));
    const dim3 grid(static_cast<unsigned int>(packed.size(0)));
    const dim3 block(kMaxNumStates);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    unpack_transition_table_per_row_kernel<packed_t, Format><<<grid, block, 0, stream>>>(
        packed.data_ptr<packed_t>(),
        scales.data_ptr<float>(),
        static_cast<int>(packed.size(1)),
        unpacked.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return unpacked;
}

torch::Tensor causal_machine_scan_unpack_int8_cuda(
    torch::Tensor packed,
    torch::Tensor scales) {
    TORCH_CHECK(packed.scalar_type() == torch::kInt8, "packed int8 transition table must be int8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "packed int8 transition scales must be float32");
    return causal_machine_scan_unpack_transition_table_cuda<int8_t, PackedTransitionFormat::Int8>(
        packed,
        scales);
}

torch::Tensor causal_machine_scan_unpack_fp8_e4m3_cuda(
    torch::Tensor packed,
    torch::Tensor scales) {
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed FP8 transition table must be uint8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "packed FP8 transition scales must be float32");
    return causal_machine_scan_unpack_transition_table_cuda<uint8_t, PackedTransitionFormat::Fp8E4M3>(
        packed,
        scales);
}

torch::Tensor causal_machine_scan_unpack_fp8_e5m2_cuda(
    torch::Tensor packed,
    torch::Tensor scales) {
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed FP8 transition table must be uint8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "packed FP8 transition scales must be float32");
    return causal_machine_scan_unpack_transition_table_cuda<uint8_t, PackedTransitionFormat::Fp8E5M2>(
        packed,
        scales);
}

std::vector<torch::Tensor> causal_machine_scan_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    const auto transition_rank = transition_source_probs.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty_like(initial_log_belief);
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_chunk",
        [&] {
                switch (transition_rank) {
                    case 8:
                        if (local_logits.size(2) == 128) {
                            launch_forward_chunk_dense_128_rank8<scalar_t>(
                                local_logits,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                beliefs,
                                final_log_belief);
                        } else {
                            launch_forward_chunk<scalar_t, 8>(
                                local_logits,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                beliefs,
                                final_log_belief);
                        }
                        break;
                case 16:
                    launch_forward_chunk<scalar_t, 16>(
                        local_logits,
                        transition_source_probs,
                        transition_dest_probs,
                        transition_context,
                        initial_log_belief,
                        transition_gate,
                        transition_stay_probs,
                        score_clamp_min,
                        score_clamp_max,
                        0,
                        scheduler.launch_chunk_size,
                        beliefs,
                        final_log_belief);
                    break;
                case 32:
                    launch_forward_chunk<scalar_t, 32>(
                        local_logits,
                        transition_source_probs,
                        transition_dest_probs,
                        transition_context,
                        initial_log_belief,
                        transition_gate,
                        transition_stay_probs,
                        score_clamp_min,
                        score_clamp_max,
                        0,
                        scheduler.launch_chunk_size,
                        beliefs,
                        final_log_belief);
                    break;
                case 64:
                    launch_forward_chunk<scalar_t, 64>(
                        local_logits,
                        transition_source_probs,
                        transition_dest_probs,
                        transition_context,
                        initial_log_belief,
                        transition_gate,
                        transition_stay_probs,
                        score_clamp_min,
                        score_clamp_max,
                        0,
                        scheduler.launch_chunk_size,
                        beliefs,
                        final_log_belief);
                    break;
                case 128:
                    launch_forward_chunk<scalar_t, 128>(
                        local_logits,
                        transition_source_probs,
                        transition_dest_probs,
                        transition_context,
                        initial_log_belief,
                        transition_gate,
                        transition_stay_probs,
                        score_clamp_min,
                        score_clamp_max,
                        0,
                        scheduler.launch_chunk_size,
                        beliefs,
                        final_log_belief);
                    break;
                default:
                    launch_forward_chunk<scalar_t>(
                        local_logits,
                        transition_source_probs,
                        transition_dest_probs,
                        transition_context,
                        initial_log_belief,
                        transition_gate,
                        transition_stay_probs,
                        score_clamp_min,
                        score_clamp_max,
                        0,
                        scheduler.launch_chunk_size,
                        beliefs,
                        final_log_belief);
                    break;
            }
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_logits_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    if (seq_len == 0) {
        auto beliefs = torch::empty_like(local_logits);
        auto final_log_belief = torch::empty_like(initial_log_belief);
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty_like(initial_log_belief);
    const auto transition_rank = transition_source_logits.size(1);
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_logits_chunk",
        [&] {
            switch (transition_rank) {
                case 8:
                    launch_forward_chunk<scalar_t, 8, true>(local_logits, transition_source_logits, transition_dest_logits, transition_context, initial_log_belief, transition_gate, transition_stay_probs, score_clamp_min, score_clamp_max, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                    break;
                case 16:
                    launch_forward_chunk<scalar_t, 16, true>(local_logits, transition_source_logits, transition_dest_logits, transition_context, initial_log_belief, transition_gate, transition_stay_probs, score_clamp_min, score_clamp_max, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                    break;
                case 32:
                    launch_forward_chunk<scalar_t, 32, true>(local_logits, transition_source_logits, transition_dest_logits, transition_context, initial_log_belief, transition_gate, transition_stay_probs, score_clamp_min, score_clamp_max, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                    break;
                case 64:
                    launch_forward_chunk<scalar_t, 64, true>(local_logits, transition_source_logits, transition_dest_logits, transition_context, initial_log_belief, transition_gate, transition_stay_probs, score_clamp_min, score_clamp_max, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                    break;
                case 128:
                    launch_forward_chunk<scalar_t, 128, true>(local_logits, transition_source_logits, transition_dest_logits, transition_context, initial_log_belief, transition_gate, transition_stay_probs, score_clamp_min, score_clamp_max, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                    break;
                default:
                    launch_forward_chunk<scalar_t, -1, true>(local_logits, transition_source_logits, transition_dest_logits, transition_context, initial_log_belief, transition_gate, transition_stay_probs, score_clamp_min, score_clamp_max, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                    break;
            }
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_masked_logits_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    TORCH_CHECK(
        !std::isfinite(score_threshold) && score_topk <= 0,
        "small-state masked dense CUDA kernel does not support native threshold/topk filtering"
    );
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty_like(initial_log_belief);
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_masked_logits_chunk",
        [&] {
            launch_forward_masked_dense_chunk<scalar_t>(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                transition_mask,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                0,
                scheduler.launch_chunk_size,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_backward_masked_logits_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    torch::Tensor work_queue_counter,
    torch::Tensor masked_transition_tile_cache,
    torch::Tensor row_sums,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk);

std::vector<torch::Tensor> causal_machine_scan_backward_masked_logits_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto seq_len = beliefs.size(1);
    const auto num_states = beliefs.size(2);
    const bool use_masked_tiled = num_states > kMaxNumStates;
    if (!use_masked_tiled) {
        TORCH_CHECK(
            !std::isfinite(score_threshold) && score_topk <= 0,
            "small-state masked dense CUDA backward kernel does not support native threshold/topk filtering"
        );
    }
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_source_per_batch = torch::zeros(
        {beliefs.size(0), transition_source_logits.size(0), transition_source_logits.size(1)},
        transition_source_logits.options());
    auto grad_transition_dest_per_batch = torch::zeros(
        {beliefs.size(0), transition_dest_logits.size(0), transition_dest_logits.size(1)},
        transition_dest_logits.options());
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = use_masked_tiled
        ? torch::zeros(initial_log_belief.sizes(), initial_log_belief.options().dtype(torch::kFloat32))
        : torch::zeros_like(initial_log_belief);
    auto grad_transition_gate_per_batch = torch::zeros(
        {beliefs.size(0)},
        transition_source_logits.options().dtype(torch::kFloat32));
    auto grad_transition_stay_per_batch = torch::zeros(
        {beliefs.size(0), transition_stay_probs.size(0)},
        transition_stay_probs.options().dtype(torch::kFloat32));
    if (seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief.to(grad_initial_log_belief.options()));
        auto grad_initial_out = use_masked_tiled
            ? grad_initial_log_belief.to(initial_log_belief.options())
            : grad_initial_log_belief;
        return {
            grad_local_logits,
            grad_transition_source_per_batch.sum(0),
            grad_transition_dest_per_batch.sum(0),
            grad_transition_context,
            grad_initial_out,
            grad_transition_gate_per_batch.sum().reshape({1}),
            grad_transition_stay_per_batch.sum(0),
        };
    }

    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size, true);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        beliefs.scalar_type(),
        "causal_machine_scan_backward_masked_logits_chunk",
        [&] {
            auto transition_matrix = torch::empty(
                {transition_mask.size(0), transition_mask.size(1)},
                transition_source_logits.options());
            auto row_sums = torch::empty(
                {transition_mask.size(0)},
                transition_source_logits.options());
            prepare_masked_dense_transition(
                transition_source_logits,
                transition_dest_logits,
                transition_mask,
                transition_matrix,
                row_sums);
            launch_backward_masked_dense_chunk<scalar_t>(
                grad_beliefs,
                grad_final_belief,
                transition_source_logits,
                transition_dest_logits,
                transition_mask,
                transition_matrix,
                row_sums,
                transition_context,
                initial_log_belief,
                beliefs,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                score_clamp_min,
                score_clamp_max,
                0,
                scheduler.launch_chunk_size,
                grad_local_logits,
                grad_transition_source_per_batch,
                grad_transition_dest_per_batch,
                grad_transition_context,
                grad_initial_log_belief,
                grad_transition_gate_per_batch,
                grad_transition_stay_per_batch);
        });
    auto grad_initial_out = use_masked_tiled
        ? grad_initial_log_belief.to(initial_log_belief.options())
        : grad_initial_log_belief;
    return {
        grad_local_logits,
        grad_transition_source_per_batch.sum(0),
        grad_transition_dest_per_batch.sum(0),
        grad_transition_context,
        grad_initial_out,
        grad_transition_gate_per_batch.sum().reshape({1}),
        grad_transition_stay_per_batch.sum(0),
    };
}

std::vector<torch::Tensor> causal_machine_scan_backward_masked_logits_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    torch::Tensor work_queue_counter,
    torch::Tensor masked_transition_tile_cache,
    torch::Tensor row_sums,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto seq_len = beliefs.size(1);
    const auto num_states = beliefs.size(2);
    const bool use_masked_tiled = num_states > kMaxNumStates;
    if (!use_masked_tiled) {
        TORCH_CHECK(
            !std::isfinite(score_threshold) && score_topk <= 0,
            "small-state masked dense CUDA backward kernel does not support native threshold/topk filtering");
    }
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_source_per_batch = torch::zeros(
        {beliefs.size(0), transition_source_logits.size(0), transition_source_logits.size(1)},
        transition_source_logits.options());
    auto grad_transition_dest_per_batch = torch::zeros(
        {beliefs.size(0), transition_dest_logits.size(0), transition_dest_logits.size(1)},
        transition_dest_logits.options());
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = use_masked_tiled
        ? torch::zeros(initial_log_belief.sizes(), initial_log_belief.options().dtype(torch::kFloat32))
        : torch::zeros_like(initial_log_belief);
    auto grad_transition_gate_per_batch = torch::zeros(
        {beliefs.size(0)},
        transition_source_logits.options().dtype(torch::kFloat32));
    auto grad_transition_stay_per_batch = torch::zeros(
        {beliefs.size(0), transition_stay_probs.size(0)},
        transition_stay_probs.options().dtype(torch::kFloat32));
    if (seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief.to(grad_initial_log_belief.options()));
        auto grad_initial_out = use_masked_tiled
            ? grad_initial_log_belief.to(initial_log_belief.options())
            : grad_initial_log_belief;
        return {
            grad_local_logits,
            grad_transition_source_per_batch.sum(0),
            grad_transition_dest_per_batch.sum(0),
            grad_transition_context,
            grad_initial_out,
            grad_transition_gate_per_batch.sum().reshape({1}),
            grad_transition_stay_per_batch.sum(0),
        };
    }

    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size, true);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        beliefs.scalar_type(),
        "causal_machine_scan_backward_masked_logits_chunk",
        [&] {
            if (!use_masked_tiled) {
                auto transition_matrix = torch::empty(
                    {transition_mask.size(0), transition_mask.size(1)},
                    transition_source_logits.options());
                auto dense_row_sums = torch::empty(
                    {transition_mask.size(0)},
                    transition_source_logits.options());
                prepare_masked_dense_transition(
                    transition_source_logits,
                    transition_dest_logits,
                    transition_mask,
                    transition_matrix,
                    dense_row_sums);
                launch_backward_masked_dense_chunk<scalar_t>(
                    grad_beliefs,
                    grad_final_belief,
                    transition_source_logits,
                    transition_dest_logits,
                    transition_mask,
                    transition_matrix,
                    dense_row_sums,
                    transition_context,
                    initial_log_belief,
                    beliefs,
                    transition_gate,
                    transition_stay_probs,
                    seq_lens,
                    score_clamp_min,
                    score_clamp_max,
                    0,
                    scheduler.launch_chunk_size,
                    grad_local_logits,
                    grad_transition_source_per_batch,
                    grad_transition_dest_per_batch,
                    grad_transition_context,
                    grad_initial_log_belief,
                    grad_transition_gate_per_batch,
                    grad_transition_stay_per_batch);
            } else {
                const int64_t tile_size = num_states;
                prepare_masked_transition_row_sums(
                    transition_source_logits,
                    transition_dest_logits,
                    transition_mask,
                    row_sums);
                launch_backward_masked_tiled_chunk<scalar_t>(
                    grad_beliefs,
                    grad_final_belief,
                    transition_source_logits,
                    transition_dest_logits,
                    row_sums,
                    transition_context,
                    initial_log_belief.to(torch::kFloat32),
                    beliefs,
                    transition_gate,
                    transition_stay_probs,
                    transition_mask,
                    seq_lens,
                    score_clamp_min,
                    score_clamp_max,
                    score_threshold,
                    score_topk,
                    tile_size,
                    0,
                    scheduler.launch_chunk_size,
                    work_queue_counter,
                    masked_transition_tile_cache,
                    grad_local_logits,
                    grad_transition_source_per_batch,
                    grad_transition_dest_per_batch,
                    grad_transition_context,
                    grad_initial_log_belief,
                    grad_transition_gate_per_batch,
                    grad_transition_stay_per_batch);
            }
        });
    auto grad_initial_out = use_masked_tiled
        ? grad_initial_log_belief.to(initial_log_belief.options())
        : grad_initial_log_belief;
    return {
        grad_local_logits,
        grad_transition_source_per_batch.sum(0),
        grad_transition_dest_per_batch.sum(0),
        grad_transition_context,
        grad_initial_out,
        grad_transition_gate_per_batch.sum().reshape({1}),
        grad_transition_stay_per_batch.sum(0),
    };
}

std::vector<torch::Tensor> causal_machine_scan_forward_sparse_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_blocks,
    torch::Tensor block_row_ptr,
    torch::Tensor block_col_idx,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty_like(initial_log_belief);
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_sparse_chunk",
        [&] {
            launch_forward_sparse_chunk<scalar_t>(
                local_logits,
                transition_blocks,
                block_row_ptr,
                block_col_idx,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                block_size,
                0,
                scheduler.launch_chunk_size,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

torch::Tensor causal_machine_scan_compute_sparse_row_sums_cuda(
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size) {
    c10::cuda::CUDAGuard device_guard(transition_source_probs.device());
    const auto num_states = transition_source_probs.size(0);
    const auto transition_rank = transition_source_probs.size(1);
    const auto nnz_blocks = block_col_idx.size(0);
    auto padded_source_probs = transition_source_probs.contiguous();
    auto padded_dest_probs = transition_dest_probs.contiguous();
    if (padded_states != num_states) {
        auto source_padded = torch::zeros(
            {padded_states, transition_rank},
            transition_source_probs.options().dtype(torch::kFloat32));
        source_padded.narrow(0, 0, num_states).copy_(transition_source_probs);
        padded_source_probs = source_padded.contiguous();
        auto dest_padded = torch::zeros(
            {transition_rank, padded_states},
            transition_dest_probs.options().dtype(torch::kFloat32));
        dest_padded.narrow(1, 0, num_states).copy_(transition_dest_probs);
        padded_dest_probs = dest_padded.contiguous();
    }
    auto row_sums = torch::zeros(
        {padded_states},
        transition_source_probs.options().dtype(torch::kFloat32));
    if (nnz_blocks == 0 || padded_states == 0 || block_size == 0) {
        return row_sums;
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kSparseMaterializeThreads = 128;
    const int64_t total_rows = nnz_blocks * block_size;
    const dim3 grid(static_cast<unsigned int>(ceil_div_int64(total_rows, kSparseMaterializeThreads)));
    sparse_transition_raw_blocks_kernel<<<grid, kSparseMaterializeThreads, 0, stream>>>(
        padded_source_probs.data_ptr<float>(),
        padded_dest_probs.data_ptr<float>(),
        block_col_idx.data_ptr<int32_t>(),
        block_dst_idx.data_ptr<int32_t>(),
        block_mask.data_ptr<float>(),
        nnz_blocks,
        static_cast<int>(padded_states),
        static_cast<int>(transition_rank),
        static_cast<int>(block_size),
        row_sums.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return row_sums;
}

torch::Tensor causal_machine_scan_compute_sparse_row_sums_from_logits_cuda(
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor source_row_max,
    torch::Tensor source_row_inv_sum,
    torch::Tensor dest_row_max,
    torch::Tensor dest_row_inv_sum,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size) {
    c10::cuda::CUDAGuard device_guard(transition_source_logits.device());
    const auto num_states = transition_source_logits.size(0);
    const auto transition_rank = transition_source_logits.size(1);
    const auto nnz_blocks = block_col_idx.size(0);
    auto row_sums = torch::zeros(
        {padded_states},
        transition_source_logits.options().dtype(torch::kFloat32));
    if (nnz_blocks == 0 || padded_states == 0 || block_size == 0) {
        return row_sums;
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kSparseMaterializeThreads = 128;
    const int64_t total_rows = nnz_blocks * block_size;
    const dim3 grid(static_cast<unsigned int>(ceil_div_int64(total_rows, kSparseMaterializeThreads)));
    sparse_transition_row_sums_from_logits_kernel<<<grid, kSparseMaterializeThreads, 0, stream>>>(
        transition_source_logits.data_ptr<float>(),
        transition_dest_logits.data_ptr<float>(),
        source_row_max.data_ptr<float>(),
        source_row_inv_sum.data_ptr<float>(),
        dest_row_max.data_ptr<float>(),
        dest_row_inv_sum.data_ptr<float>(),
        block_col_idx.data_ptr<int32_t>(),
        block_dst_idx.data_ptr<int32_t>(),
        block_mask.data_ptr<float>(),
        nnz_blocks,
        static_cast<int>(num_states),
        static_cast<int>(padded_states),
        static_cast<int>(transition_rank),
        static_cast<int>(block_size),
        row_sums.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return row_sums;
}

std::vector<torch::Tensor> causal_machine_scan_forward_sparse_factors_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor row_sums,
    torch::Tensor block_row_ptr,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    const auto transition_rank = transition_source_probs.size(1);
    const auto padded_states = row_sums.size(0);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty_like(initial_log_belief);
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    auto padded_source_probs = transition_source_probs.contiguous();
    auto padded_dest_probs = transition_dest_probs.contiguous();
    if (padded_states != transition_source_probs.size(0)) {
        auto source_padded = torch::zeros(
            {padded_states, transition_rank},
            transition_source_probs.options().dtype(torch::kFloat32));
        source_padded.narrow(0, 0, transition_source_probs.size(0)).copy_(transition_source_probs);
        padded_source_probs = source_padded.contiguous();
        auto dest_padded = torch::zeros(
            {transition_rank, padded_states},
            transition_dest_probs.options().dtype(torch::kFloat32));
        dest_padded.narrow(1, 0, transition_dest_probs.size(1)).copy_(transition_dest_probs);
        padded_dest_probs = dest_padded.contiguous();
    }
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_sparse_factor_chunk",
        [&] {
            launch_forward_sparse_factor_chunk<scalar_t>(
                local_logits,
                padded_source_probs,
                padded_dest_probs,
                row_sums,
                block_row_ptr,
                block_col_idx,
                block_dst_idx,
                torch::Tensor(),
                torch::Tensor(),
                block_mask,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                block_size,
                0,
                scheduler.launch_chunk_size,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_sparse_logits_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor block_row_ptr,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor src_row_ptr,
    torch::Tensor src_nz_idx,
    torch::Tensor block_mask,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const int64_t padded_states = static_cast<int64_t>(block_row_ptr.size(0) - 1) * block_size;
    auto row_sums = torch::empty(
        {padded_states},
        transition_source_logits.options().dtype(torch::kFloat32));
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty_like(initial_log_belief);
    if (local_logits.size(1) == 0) {
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto scheduler = make_scan_chunk_scheduler(local_logits.size(1), chunk_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_sparse_logits_chunk",
        [&] {
            launch_forward_sparse_logits_chunk<scalar_t>(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                row_sums,
                block_row_ptr,
                block_col_idx,
                block_dst_idx,
                src_row_ptr,
                src_nz_idx,
                block_mask,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                block_size,
                0,
                scheduler.launch_chunk_size,
                beliefs,
                final_log_belief);
        });
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_backward_sparse_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_blocks,
    torch::Tensor block_row_ptr,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor src_row_ptr,
    torch::Tensor src_nz_idx,
    torch::Tensor grouped_src_row_ptr,
    torch::Tensor grouped_src_block_idx,
    torch::Tensor row_sums,
    torch::Tensor block_mask,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto num_states = beliefs.size(2);
    const auto transition_rank = transition_source_probs.size(1);
    const auto padded_states = row_sums.size(0);
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_blocks = torch::zeros_like(transition_blocks);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros({num_states}, beliefs.options().dtype(torch::kFloat32));
    const auto seq_len = beliefs.size(1);
    if (seq_len > 0) {
        const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size, true);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            beliefs.scalar_type(),
            "causal_machine_scan_backward_sparse_chunk",
            [&] {
                launch_backward_sparse_chunk<scalar_t>(
                    grad_beliefs,
                    grad_final_belief,
                    transition_blocks,
                    block_row_ptr,
                    block_col_idx,
                    block_dst_idx,
                    src_row_ptr,
                    src_nz_idx,
                    grouped_src_row_ptr,
                    grouped_src_block_idx,
                    transition_context,
                    initial_log_belief,
                    beliefs,
                    transition_gate,
                    transition_stay_probs,
                    seq_lens,
                    block_size,
                    0,
                    scheduler.launch_chunk_size,
                    grad_local_logits,
                    grad_transition_blocks,
                    grad_transition_context,
                    grad_initial_log_belief,
                    grad_transition_gate,
                    grad_transition_stay);
            });
    } else {
        grad_initial_log_belief.copy_(grad_final_belief);
    }
    auto padded_source_probs = transition_source_probs.contiguous();
    auto padded_dest_probs = transition_dest_probs.contiguous();
    if (padded_states != num_states) {
        auto source_padded = torch::zeros(
            {padded_states, transition_rank},
            transition_source_probs.options().dtype(torch::kFloat32));
        source_padded.narrow(0, 0, num_states).copy_(transition_source_probs);
        padded_source_probs = source_padded.contiguous();
        auto dest_padded = torch::zeros(
            {transition_rank, padded_states},
            transition_dest_probs.options().dtype(torch::kFloat32));
        dest_padded.narrow(1, 0, num_states).copy_(transition_dest_probs);
        padded_dest_probs = dest_padded.contiguous();
    }
    auto row_proj = torch::zeros({padded_states}, beliefs.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool use_grouped_sparse_backward = (
        grouped_src_row_ptr.numel() > 1
        && grouped_src_block_idx.numel() > 0
        && grouped_src_block_idx.numel() < src_row_ptr.size(0) - 1
    );
    const int grouped_src_group_count = static_cast<int>(grouped_src_block_idx.numel());
    auto active_dst_mask = block_row_ptr.slice(0, 1).ne(block_row_ptr.slice(0, 0, block_row_ptr.size(0) - 1));
    auto active_dst_block_idx = torch::nonzero(active_dst_mask).squeeze(1).to(torch::kInt32).contiguous();
    const bool use_compressed_dest_grad = (
        active_dst_block_idx.numel() > 0
        && active_dst_block_idx.numel() < block_row_ptr.size(0) - 1
    );
    auto grad_transition_source_probs_padded = torch::zeros(
        {padded_states, transition_rank},
        beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs_padded = torch::zeros(
        {transition_rank, padded_states},
        beliefs.options().dtype(torch::kFloat32));
    torch::Tensor grad_transition_source_probs_compressed;
    torch::Tensor grad_transition_dest_probs_compressed;
    if (use_grouped_sparse_backward) {
        grad_transition_source_probs_compressed = torch::zeros(
            {static_cast<int64_t>(grouped_src_group_count) * block_size, transition_rank},
            beliefs.options().dtype(torch::kFloat32));
    }
    if (use_compressed_dest_grad) {
        grad_transition_dest_probs_compressed = torch::zeros(
            {transition_rank, static_cast<int64_t>(active_dst_block_idx.numel()) * block_size},
            beliefs.options().dtype(torch::kFloat32));
    }
    constexpr int kRowProjThreads = 128;
    if (use_grouped_sparse_backward) {
        const int64_t grouped_states = static_cast<int64_t>(grouped_src_group_count) * static_cast<int64_t>(block_size);
        const dim3 row_proj_grid(static_cast<unsigned int>(ceil_div_int64(grouped_states, kRowProjThreads)));
        sparse_transition_row_proj_grouped_kernel<<<row_proj_grid, kRowProjThreads, 0, stream>>>(
            grad_transition_blocks.data_ptr<float>(),
            transition_blocks.data_ptr<float>(),
            block_mask.data_ptr<float>(),
            grouped_src_row_ptr.data_ptr<int32_t>(),
            grouped_src_block_idx.data_ptr<int32_t>(),
            src_nz_idx.data_ptr<int32_t>(),
            grouped_src_group_count,
            static_cast<int>(padded_states),
            static_cast<int>(block_size),
            row_proj.data_ptr<float>());
    } else {
        const dim3 row_proj_grid(static_cast<unsigned int>(ceil_div_int64(padded_states, kRowProjThreads)));
        sparse_transition_row_proj_kernel<<<row_proj_grid, kRowProjThreads, 0, stream>>>(
            grad_transition_blocks.data_ptr<float>(),
            transition_blocks.data_ptr<float>(),
            block_mask.data_ptr<float>(),
            src_row_ptr.data_ptr<int32_t>(),
            src_nz_idx.data_ptr<int32_t>(),
            static_cast<int>(padded_states),
            static_cast<int>(block_size),
            row_proj.data_ptr<float>());
    }
    const dim3 factor_block(16, 16);
    if (use_grouped_sparse_backward) {
        const int64_t grouped_states = static_cast<int64_t>(grouped_src_group_count) * static_cast<int64_t>(block_size);
        const dim3 source_grid(
            static_cast<unsigned int>(ceil_div_int64(transition_rank, static_cast<int64_t>(factor_block.x))),
            static_cast<unsigned int>(ceil_div_int64(grouped_states, static_cast<int64_t>(factor_block.y))));
        sparse_transition_source_grad_grouped_compressed_kernel<<<source_grid, factor_block, 0, stream>>>(
            grad_transition_blocks.data_ptr<float>(),
            block_mask.data_ptr<float>(),
            grouped_src_row_ptr.data_ptr<int32_t>(),
            grouped_src_block_idx.data_ptr<int32_t>(),
            src_nz_idx.data_ptr<int32_t>(),
            block_dst_idx.data_ptr<int32_t>(),
            padded_dest_probs.data_ptr<float>(),
            row_sums.data_ptr<float>(),
            row_proj.data_ptr<float>(),
            grouped_src_group_count,
            static_cast<int>(padded_states),
            static_cast<int>(transition_rank),
            static_cast<int>(block_size),
            grad_transition_source_probs_compressed.data_ptr<float>());
        scatter_grouped_source_grad_kernel<<<source_grid, factor_block, 0, stream>>>(
            grad_transition_source_probs_compressed.data_ptr<float>(),
            grouped_src_block_idx.data_ptr<int32_t>(),
            grouped_src_group_count,
            static_cast<int>(padded_states),
            static_cast<int>(transition_rank),
            static_cast<int>(block_size),
            grad_transition_source_probs_padded.data_ptr<float>());
    } else {
        const dim3 source_grid(
            static_cast<unsigned int>(ceil_div_int64(transition_rank, static_cast<int64_t>(factor_block.x))),
            static_cast<unsigned int>(ceil_div_int64(padded_states, static_cast<int64_t>(factor_block.y))));
        sparse_transition_source_grad_kernel<<<source_grid, factor_block, 0, stream>>>(
            grad_transition_blocks.data_ptr<float>(),
            block_mask.data_ptr<float>(),
            src_row_ptr.data_ptr<int32_t>(),
            src_nz_idx.data_ptr<int32_t>(),
            block_dst_idx.data_ptr<int32_t>(),
            padded_dest_probs.data_ptr<float>(),
            row_sums.data_ptr<float>(),
            row_proj.data_ptr<float>(),
            static_cast<int>(padded_states),
            static_cast<int>(transition_rank),
            static_cast<int>(block_size),
            grad_transition_source_probs_padded.data_ptr<float>());
    }
    if (use_compressed_dest_grad) {
        const int64_t active_dst_states = static_cast<int64_t>(active_dst_block_idx.numel()) * static_cast<int64_t>(block_size);
        const dim3 dest_grid(
            static_cast<unsigned int>(ceil_div_int64(active_dst_states, static_cast<int64_t>(factor_block.x))),
            static_cast<unsigned int>(ceil_div_int64(transition_rank, static_cast<int64_t>(factor_block.y))));
        sparse_transition_dest_grad_compressed_kernel<<<dest_grid, factor_block, 0, stream>>>(
            grad_transition_blocks.data_ptr<float>(),
            block_mask.data_ptr<float>(),
            active_dst_block_idx.data_ptr<int32_t>(),
            block_row_ptr.data_ptr<int32_t>(),
            block_col_idx.data_ptr<int32_t>(),
            padded_source_probs.data_ptr<float>(),
            row_sums.data_ptr<float>(),
            row_proj.data_ptr<float>(),
            static_cast<int>(active_dst_block_idx.numel()),
            static_cast<int>(padded_states),
            static_cast<int>(transition_rank),
            static_cast<int>(block_size),
            grad_transition_dest_probs_compressed.data_ptr<float>());
        scatter_compressed_dest_grad_kernel<<<dest_grid, factor_block, 0, stream>>>(
            grad_transition_dest_probs_compressed.data_ptr<float>(),
            active_dst_block_idx.data_ptr<int32_t>(),
            static_cast<int>(active_dst_block_idx.numel()),
            static_cast<int>(padded_states),
            static_cast<int>(transition_rank),
            static_cast<int>(block_size),
            grad_transition_dest_probs_padded.data_ptr<float>());
    } else {
        const dim3 dest_grid(
            static_cast<unsigned int>(ceil_div_int64(padded_states, static_cast<int64_t>(factor_block.x))),
            static_cast<unsigned int>(ceil_div_int64(transition_rank, static_cast<int64_t>(factor_block.y))));
        sparse_transition_dest_grad_kernel<<<dest_grid, factor_block, 0, stream>>>(
            grad_transition_blocks.data_ptr<float>(),
            block_mask.data_ptr<float>(),
            block_row_ptr.data_ptr<int32_t>(),
            block_col_idx.data_ptr<int32_t>(),
            padded_source_probs.data_ptr<float>(),
            row_sums.data_ptr<float>(),
            row_proj.data_ptr<float>(),
            static_cast<int>(padded_states),
            static_cast<int>(transition_rank),
            static_cast<int>(block_size),
            grad_transition_dest_probs_padded.data_ptr<float>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto grad_transition_source_probs = padded_states == num_states
        ? grad_transition_source_probs_padded
        : grad_transition_source_probs_padded.narrow(0, 0, num_states).contiguous();
    auto grad_transition_dest_probs = padded_states == num_states
        ? grad_transition_dest_probs_padded
        : grad_transition_dest_probs_padded.narrow(1, 0, num_states).contiguous();
    return {
        grad_local_logits,
        grad_transition_source_probs,
        grad_transition_dest_probs,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate,
        grad_transition_stay,
    };
}

std::vector<torch::Tensor> causal_machine_scan_backward_sparse_factors_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor block_row_ptr,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor src_row_ptr,
    torch::Tensor src_nz_idx,
    torch::Tensor grouped_src_row_ptr,
    torch::Tensor grouped_src_block_idx,
    torch::Tensor row_sums,
    torch::Tensor block_mask,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto num_states = beliefs.size(2);
    const auto transition_rank = transition_source_probs.size(1);
    const auto padded_states = row_sums.size(0);
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros({num_states}, beliefs.options().dtype(torch::kFloat32));
    auto padded_source_probs = transition_source_probs.contiguous();
    auto padded_dest_probs = transition_dest_probs.contiguous();
    if (padded_states != num_states) {
        auto source_padded = torch::zeros(
            {padded_states, transition_rank},
            transition_source_probs.options().dtype(torch::kFloat32));
        source_padded.narrow(0, 0, num_states).copy_(transition_source_probs);
        padded_source_probs = source_padded.contiguous();
        auto dest_padded = torch::zeros(
            {transition_rank, padded_states},
            transition_dest_probs.options().dtype(torch::kFloat32));
        dest_padded.narrow(1, 0, num_states).copy_(transition_dest_probs);
        padded_dest_probs = dest_padded.contiguous();
    }
    auto grad_transition_source_probs_padded = torch::zeros(
        {padded_states, transition_rank},
        beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs_padded = torch::zeros(
        {transition_rank, padded_states},
        beliefs.options().dtype(torch::kFloat32));
    const auto seq_len = beliefs.size(1);
    if (seq_len > 0) {
        const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size, true);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            beliefs.scalar_type(),
            "causal_machine_scan_backward_sparse_factor_chunk",
            [&] {
                launch_backward_sparse_factor_chunk<scalar_t>(
                    grad_beliefs,
                    grad_final_belief,
                    padded_source_probs,
                    padded_dest_probs,
                    row_sums,
                    block_row_ptr,
                    block_col_idx,
                    block_dst_idx,
                    src_row_ptr,
                    src_nz_idx,
                    grouped_src_row_ptr,
                    grouped_src_block_idx,
                    block_mask,
                    transition_context,
                    initial_log_belief,
                    beliefs,
                    transition_gate,
                    transition_stay_probs,
                    seq_lens,
                    block_size,
                    0,
                    scheduler.launch_chunk_size,
                    grad_local_logits,
                    grad_transition_source_probs_padded,
                    grad_transition_dest_probs_padded,
                    grad_transition_context,
                    grad_initial_log_belief,
                    grad_transition_gate,
                    grad_transition_stay);
            });
    } else {
        grad_initial_log_belief.copy_(grad_final_belief);
    }
    auto grad_transition_source_probs = padded_states == num_states
        ? grad_transition_source_probs_padded
        : grad_transition_source_probs_padded.narrow(0, 0, num_states).contiguous();
    auto grad_transition_dest_probs = padded_states == num_states
        ? grad_transition_dest_probs_padded
        : grad_transition_dest_probs_padded.narrow(1, 0, num_states).contiguous();
    return {
        grad_local_logits,
        grad_transition_source_probs,
        grad_transition_dest_probs,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate,
        grad_transition_stay,
    };
}

std::vector<torch::Tensor> causal_machine_scan_backward_sparse_logits_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor block_row_ptr,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor src_row_ptr,
    torch::Tensor src_nz_idx,
    torch::Tensor grouped_src_row_ptr,
    torch::Tensor grouped_src_block_idx,
    torch::Tensor block_mask,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    auto source_row_max = torch::empty(
        {transition_source_logits.size(0)},
        transition_source_logits.options().dtype(torch::kFloat32));
    auto source_row_inv_sum = torch::empty_like(source_row_max);
    auto dest_row_max = torch::empty(
        {transition_dest_logits.size(0)},
        transition_dest_logits.options().dtype(torch::kFloat32));
    auto dest_row_inv_sum = torch::empty_like(dest_row_max);
    if (transition_source_logits.numel() > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        const dim3 block(kMaxNumStates);
        const dim3 source_grid(static_cast<unsigned int>(transition_source_logits.size(0)));
        const dim3 dest_grid(static_cast<unsigned int>(transition_dest_logits.size(0)));
        row_softmax_stats_strided_128_kernel<<<source_grid, block, 0, stream>>>(
            transition_source_logits.data_ptr<float>(),
            static_cast<int>(transition_source_logits.size(1)),
            source_row_max.data_ptr<float>(),
            source_row_inv_sum.data_ptr<float>());
        row_softmax_stats_strided_128_kernel<<<dest_grid, block, 0, stream>>>(
            transition_dest_logits.data_ptr<float>(),
            static_cast<int>(transition_dest_logits.size(1)),
            dest_row_max.data_ptr<float>(),
            dest_row_inv_sum.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int64_t padded_states = static_cast<int64_t>(block_row_ptr.size(0) - 1) * block_size;
    auto row_sums = torch::empty(
        {padded_states},
        transition_source_logits.options().dtype(torch::kFloat32));
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay = torch::zeros({beliefs.size(2)}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_source_probs = torch::zeros_like(transition_source_logits);
    auto grad_transition_dest_probs = torch::zeros_like(transition_dest_logits);
    if (beliefs.size(1) > 0) {
        const auto scheduler = make_scan_chunk_scheduler(beliefs.size(1), chunk_size, true);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            beliefs.scalar_type(),
            "causal_machine_scan_backward_sparse_logits_chunk",
            [&] {
                launch_backward_sparse_logits_chunk<scalar_t>(
                    grad_beliefs,
                    grad_final_belief,
                    transition_source_logits,
                    transition_dest_logits,
                    source_row_max,
                    source_row_inv_sum,
                    dest_row_max,
                    dest_row_inv_sum,
                    row_sums,
                    block_row_ptr,
                    block_col_idx,
                    block_dst_idx,
                    src_row_ptr,
                    src_nz_idx,
                    grouped_src_row_ptr,
                    grouped_src_block_idx,
                    block_mask,
                    transition_context,
                    initial_log_belief,
                    beliefs,
                    transition_gate,
                    transition_stay_probs,
                    seq_lens,
                    block_size,
                    0,
                    scheduler.launch_chunk_size,
                    grad_local_logits,
                    grad_transition_source_probs,
                    grad_transition_dest_probs,
                    grad_transition_context,
                    grad_initial_log_belief,
                    grad_transition_gate,
                    grad_transition_stay);
            });
    } else {
        grad_initial_log_belief.copy_(grad_final_belief);
    }
    std::vector<torch::Tensor> grads = {
        grad_local_logits,
        grad_transition_source_probs,
        grad_transition_dest_probs,
        grad_transition_context,
        grad_initial_log_belief,
        grad_transition_gate,
        grad_transition_stay,
    };
    if (transition_source_logits.numel() > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        const dim3 block(kMaxNumStates);
        const dim3 source_grid(static_cast<unsigned int>(transition_source_logits.size(0)));
        const dim3 dest_grid(static_cast<unsigned int>(transition_dest_logits.size(0)));
        auto grad_transition_source_logits = torch::empty_like(transition_source_logits);
        auto grad_transition_dest_logits = torch::empty_like(transition_dest_logits);
        row_softmax_backward_from_stats_strided_128_kernel<<<source_grid, block, 0, stream>>>(
            grad_transition_source_probs.data_ptr<float>(),
            transition_source_logits.data_ptr<float>(),
            source_row_max.data_ptr<float>(),
            source_row_inv_sum.data_ptr<float>(),
            static_cast<int>(transition_source_logits.size(1)),
            grad_transition_source_logits.data_ptr<float>());
        row_softmax_backward_from_stats_strided_128_kernel<<<dest_grid, block, 0, stream>>>(
            grad_transition_dest_probs.data_ptr<float>(),
            transition_dest_logits.data_ptr<float>(),
            dest_row_max.data_ptr<float>(),
            dest_row_inv_sum.data_ptr<float>(),
            static_cast<int>(transition_dest_logits.size(1)),
            grad_transition_dest_logits.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        grads[1] = grad_transition_source_logits;
        grads[2] = grad_transition_dest_logits;
    }
    return grads;
}

std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks_cuda(
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size) {
    c10::cuda::CUDAGuard device_guard(transition_source_probs.device());
    const auto num_states = transition_source_probs.size(0);
    const auto transition_rank = transition_source_probs.size(1);
    const auto nnz_blocks = block_col_idx.size(0);
    auto padded_source_probs = transition_source_probs.contiguous();
    auto padded_dest_probs = transition_dest_probs.contiguous();
    if (padded_states != num_states) {
        auto source_padded = torch::zeros(
            {padded_states, transition_rank},
            transition_source_probs.options().dtype(torch::kFloat32));
        source_padded.narrow(0, 0, num_states).copy_(transition_source_probs);
        padded_source_probs = source_padded.contiguous();
        auto dest_padded = torch::zeros(
            {transition_rank, padded_states},
            transition_dest_probs.options().dtype(torch::kFloat32));
        dest_padded.narrow(1, 0, num_states).copy_(transition_dest_probs);
        padded_dest_probs = dest_padded.contiguous();
    }
    auto transition_blocks = torch::zeros(
        {nnz_blocks, block_size, block_size},
        transition_source_probs.options().dtype(torch::kFloat32));
    auto row_sums = torch::zeros(
        {padded_states},
        transition_source_probs.options().dtype(torch::kFloat32));
    if (nnz_blocks == 0 || padded_states == 0 || block_size == 0) {
        return {transition_blocks, row_sums};
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kSparseMaterializeThreads = 128;
    const int64_t total_rows = nnz_blocks * block_size;
    const dim3 grid(static_cast<unsigned int>(ceil_div_int64(total_rows, kSparseMaterializeThreads)));
    sparse_transition_raw_blocks_kernel<<<grid, kSparseMaterializeThreads, 0, stream>>>(
        padded_source_probs.data_ptr<float>(),
        padded_dest_probs.data_ptr<float>(),
        block_col_idx.data_ptr<int32_t>(),
        block_dst_idx.data_ptr<int32_t>(),
        block_mask.data_ptr<float>(),
        nnz_blocks,
        static_cast<int>(padded_states),
        static_cast<int>(transition_rank),
        static_cast<int>(block_size),
        transition_blocks.data_ptr<float>(),
        row_sums.data_ptr<float>());
    sparse_transition_normalize_blocks_kernel<<<grid, kSparseMaterializeThreads, 0, stream>>>(
        block_col_idx.data_ptr<int32_t>(),
        row_sums.data_ptr<float>(),
        nnz_blocks,
        static_cast<int>(padded_states),
        static_cast<int>(block_size),
        transition_blocks.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        transition_blocks,
        row_sums,
    };
}

template <typename packed_t, PackedTransitionFormat Format>
std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks_packed_cuda_impl(
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size) {
    c10::cuda::CUDAGuard device_guard(transition_source_packed.device());
    const auto num_states = transition_source_packed.size(0);
    const auto transition_rank = transition_source_packed.size(1);
    const auto nnz_blocks = block_col_idx.size(0);
    auto padded_source_packed = transition_source_packed.contiguous();
    auto padded_source_scales = transition_source_scales.contiguous();
    auto padded_dest_packed = transition_dest_packed.contiguous();
    if (padded_states != num_states) {
        auto source_padded = torch::zeros(
            {padded_states, transition_rank},
            transition_source_packed.options());
        source_padded.narrow(0, 0, num_states).copy_(transition_source_packed);
        padded_source_packed = source_padded.contiguous();
        auto source_scales_padded = torch::ones(
            {padded_states},
            transition_source_scales.options().dtype(torch::kFloat32));
        source_scales_padded.narrow(0, 0, num_states).copy_(transition_source_scales);
        padded_source_scales = source_scales_padded.contiguous();
        auto dest_padded = torch::zeros(
            {transition_rank, padded_states},
            transition_dest_packed.options());
        dest_padded.narrow(1, 0, num_states).copy_(transition_dest_packed);
        padded_dest_packed = dest_padded.contiguous();
    }
    auto transition_blocks = torch::zeros(
        {nnz_blocks, block_size, block_size},
        transition_source_scales.options().dtype(torch::kFloat32));
    auto row_sums = torch::zeros(
        {padded_states},
        transition_source_scales.options().dtype(torch::kFloat32));
    if (nnz_blocks == 0 || padded_states == 0 || block_size == 0) {
        return {transition_blocks, row_sums};
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kSparseMaterializeThreads = 128;
    const int64_t total_rows = nnz_blocks * block_size;
    const dim3 grid(static_cast<unsigned int>(ceil_div_int64(total_rows, kSparseMaterializeThreads)));
    sparse_transition_raw_blocks_packed_kernel<packed_t, Format><<<grid, kSparseMaterializeThreads, 0, stream>>>(
        padded_source_packed.data_ptr<packed_t>(),
        padded_source_scales.data_ptr<float>(),
        padded_dest_packed.data_ptr<packed_t>(),
        transition_dest_scales.data_ptr<float>(),
        block_col_idx.data_ptr<int32_t>(),
        block_dst_idx.data_ptr<int32_t>(),
        block_mask.data_ptr<float>(),
        nnz_blocks,
        static_cast<int>(padded_states),
        static_cast<int>(transition_rank),
        static_cast<int>(block_size),
        transition_blocks.data_ptr<float>(),
        row_sums.data_ptr<float>());
    sparse_transition_normalize_blocks_kernel<<<grid, kSparseMaterializeThreads, 0, stream>>>(
        block_col_idx.data_ptr<int32_t>(),
        row_sums.data_ptr<float>(),
        nnz_blocks,
        static_cast<int>(padded_states),
        static_cast<int>(block_size),
        transition_blocks.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        transition_blocks,
        row_sums,
    };
}

std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks_int8_cuda(
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size) {
    return causal_machine_scan_materialize_sparse_blocks_packed_cuda_impl<int8_t, PackedTransitionFormat::Int8>(
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        block_col_idx,
        block_dst_idx,
        block_mask,
        padded_states,
        block_size);
}

std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks_fp8_cuda(
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t fp8_format,
    int64_t padded_states,
    int64_t block_size) {
    if (fp8_format == 0) {
        return causal_machine_scan_materialize_sparse_blocks_packed_cuda_impl<uint8_t, PackedTransitionFormat::Fp8E4M3>(
            transition_source_packed,
            transition_source_scales,
            transition_dest_packed,
            transition_dest_scales,
            block_col_idx,
            block_dst_idx,
            block_mask,
            padded_states,
            block_size);
    }
    TORCH_CHECK(fp8_format == 1, "fp8_format must be 0 (e4m3) or 1 (e5m2)");
    return causal_machine_scan_materialize_sparse_blocks_packed_cuda_impl<uint8_t, PackedTransitionFormat::Fp8E5M2>(
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        block_col_idx,
        block_dst_idx,
        block_mask,
        padded_states,
        block_size);
}

std::vector<torch::Tensor> causal_machine_scan_forward_composable_logits_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    const auto transition_rank = transition_source_logits.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty_like(initial_log_belief);
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }

    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size);
    const int64_t launch_chunk_size = std::max<int64_t>(scheduler.launch_chunk_size, 1);
    const int64_t num_chunks = ceil_div_int64(seq_len, launch_chunk_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_forward_composable_logits_chunk",
        [&] {
            const bool use_chunk_summary_path = num_chunks > 1;
            if (use_chunk_summary_path) {
                auto carry_log_belief_cache = torch::empty(
                    {num_chunks + 1, local_logits.size(0), local_logits.size(2)},
                    initial_log_belief.options());
                carry_log_belief_cache.select(0, 0).copy_(initial_log_belief);
                switch (transition_rank) {
                    case 8:
                        for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                            const int64_t chunk_start = chunk_idx * launch_chunk_size;
                            const int64_t current_chunk_len = std::min<int64_t>(launch_chunk_size, seq_len - chunk_start);
                            launch_forward_composable_chunk_summary<scalar_t, 8, true>(
                                local_logits, transition_source_logits, transition_dest_logits, transition_context,
                                carry_log_belief_cache.select(0, chunk_idx), transition_stay_probs,
                                chunk_start, current_chunk_len, carry_log_belief_cache.select(0, chunk_idx + 1));
                        }
                        launch_forward_composable_chunk_finalize<scalar_t, 8, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            carry_log_belief_cache, transition_stay_probs, num_chunks, launch_chunk_size, beliefs);
                        break;
                    case 16:
                        for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                            const int64_t chunk_start = chunk_idx * launch_chunk_size;
                            const int64_t current_chunk_len = std::min<int64_t>(launch_chunk_size, seq_len - chunk_start);
                            launch_forward_composable_chunk_summary<scalar_t, 16, true>(
                                local_logits, transition_source_logits, transition_dest_logits, transition_context,
                                carry_log_belief_cache.select(0, chunk_idx), transition_stay_probs,
                                chunk_start, current_chunk_len, carry_log_belief_cache.select(0, chunk_idx + 1));
                        }
                        launch_forward_composable_chunk_finalize<scalar_t, 16, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            carry_log_belief_cache, transition_stay_probs, num_chunks, launch_chunk_size, beliefs);
                        break;
                    case 32:
                        for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                            const int64_t chunk_start = chunk_idx * launch_chunk_size;
                            const int64_t current_chunk_len = std::min<int64_t>(launch_chunk_size, seq_len - chunk_start);
                            launch_forward_composable_chunk_summary<scalar_t, 32, true>(
                                local_logits, transition_source_logits, transition_dest_logits, transition_context,
                                carry_log_belief_cache.select(0, chunk_idx), transition_stay_probs,
                                chunk_start, current_chunk_len, carry_log_belief_cache.select(0, chunk_idx + 1));
                        }
                        launch_forward_composable_chunk_finalize<scalar_t, 32, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            carry_log_belief_cache, transition_stay_probs, num_chunks, launch_chunk_size, beliefs);
                        break;
                    case 64:
                        for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                            const int64_t chunk_start = chunk_idx * launch_chunk_size;
                            const int64_t current_chunk_len = std::min<int64_t>(launch_chunk_size, seq_len - chunk_start);
                            launch_forward_composable_chunk_summary<scalar_t, 64, true>(
                                local_logits, transition_source_logits, transition_dest_logits, transition_context,
                                carry_log_belief_cache.select(0, chunk_idx), transition_stay_probs,
                                chunk_start, current_chunk_len, carry_log_belief_cache.select(0, chunk_idx + 1));
                        }
                        launch_forward_composable_chunk_finalize<scalar_t, 64, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            carry_log_belief_cache, transition_stay_probs, num_chunks, launch_chunk_size, beliefs);
                        break;
                    case 128:
                        for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                            const int64_t chunk_start = chunk_idx * launch_chunk_size;
                            const int64_t current_chunk_len = std::min<int64_t>(launch_chunk_size, seq_len - chunk_start);
                            launch_forward_composable_chunk_summary<scalar_t, 128, true>(
                                local_logits, transition_source_logits, transition_dest_logits, transition_context,
                                carry_log_belief_cache.select(0, chunk_idx), transition_stay_probs,
                                chunk_start, current_chunk_len, carry_log_belief_cache.select(0, chunk_idx + 1));
                        }
                        launch_forward_composable_chunk_finalize<scalar_t, 128, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            carry_log_belief_cache, transition_stay_probs, num_chunks, launch_chunk_size, beliefs);
                        break;
                    default:
                        for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                            const int64_t chunk_start = chunk_idx * launch_chunk_size;
                            const int64_t current_chunk_len = std::min<int64_t>(launch_chunk_size, seq_len - chunk_start);
                            launch_forward_composable_chunk_summary<scalar_t, -1, true>(
                                local_logits, transition_source_logits, transition_dest_logits, transition_context,
                                carry_log_belief_cache.select(0, chunk_idx), transition_stay_probs,
                                chunk_start, current_chunk_len, carry_log_belief_cache.select(0, chunk_idx + 1));
                        }
                        launch_forward_composable_chunk_finalize<scalar_t, -1, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            carry_log_belief_cache, transition_stay_probs, num_chunks, launch_chunk_size, beliefs);
                        break;
                }
                final_log_belief.copy_(carry_log_belief_cache.select(0, num_chunks));
            } else {
                switch (transition_rank) {
                    case 8:
                        launch_forward_composable_chunk<scalar_t, 8, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            initial_log_belief, transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 16:
                        launch_forward_composable_chunk<scalar_t, 16, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            initial_log_belief, transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 32:
                        launch_forward_composable_chunk<scalar_t, 32, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            initial_log_belief, transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 64:
                        launch_forward_composable_chunk<scalar_t, 64, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            initial_log_belief, transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 128:
                        launch_forward_composable_chunk<scalar_t, 128, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            initial_log_belief, transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    default:
                        launch_forward_composable_chunk<scalar_t, -1, true>(
                            local_logits, transition_source_logits, transition_dest_logits, transition_context,
                            initial_log_belief, transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                }
            }
        });
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
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    const auto num_states = beliefs.size(2);
    const auto transition_rank = transition_source_probs.size(1);
    // Avoid materializing [B, ...] gradient buffers when the direct in-kernel
    // reduction path fits in shared memory for the active rank/state geometry.
    const bool direct_small_rank_grad = can_use_direct_grad_reduce(
        beliefs.get_device(),
        static_cast<int>(num_states),
        static_cast<int>(transition_rank));
    const int64_t direct_staging_worker_blocks = direct_small_rank_grad
        ? small_state_direct_staging_worker_blocks(beliefs.get_device(), batch_size)
        : 0;
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_transition_source_probs = torch::zeros({num_states, transition_rank}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs = torch::zeros({transition_rank, num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_probs = torch::zeros({num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_source_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, num_states, transition_rank}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, num_states, transition_rank}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, transition_rank, num_states}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, transition_rank, num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, num_states}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_gate_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size}, beliefs.options().dtype(torch::kFloat32));
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto carry = grad_final_belief.contiguous();
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size, true);
    if (seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief);
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            beliefs.scalar_type(),
            "causal_machine_scan_backward_chunk",
            [&] {
                switch (transition_rank) {
                    case 8:
                        if (beliefs.size(2) == 128) {
                            if (direct_small_rank_grad) {
                                launch_backward_chunk_dense_128_rank8<scalar_t, true>(
                                    grad_beliefs,
                                    carry,
                                    transition_source_probs,
                                    transition_dest_probs,
                                    transition_context,
                                    initial_log_belief,
                                    beliefs,
                                    transition_gate,
                                    transition_stay_probs,
                                    score_clamp_min,
                                    score_clamp_max,
                                    0,
                                    scheduler.launch_chunk_size,
                                    grad_local_logits,
                                    grad_transition_source_per_batch,
                                    grad_transition_dest_per_batch,
                                    grad_transition_context,
                                    grad_initial_log_belief,
                                    grad_transition_gate_per_batch,
                                    grad_transition_stay_per_batch);
                            } else {
                                launch_backward_chunk_dense_128_rank8<scalar_t>(
                                    grad_beliefs,
                                    carry,
                                    transition_source_probs,
                                    transition_dest_probs,
                                    transition_context,
                                    initial_log_belief,
                                    beliefs,
                                    transition_gate,
                                    transition_stay_probs,
                                    score_clamp_min,
                                    score_clamp_max,
                                    0,
                                    scheduler.launch_chunk_size,
                                    grad_local_logits,
                                    grad_transition_source_per_batch,
                                    grad_transition_dest_per_batch,
                                    grad_transition_context,
                                    grad_initial_log_belief,
                                    grad_transition_gate_per_batch,
                                    grad_transition_stay_per_batch);
                            }
                        } else {
                            launch_backward_chunk<scalar_t, 8, true>(
                                grad_beliefs,
                                carry,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                beliefs,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                grad_local_logits,
                                grad_transition_source_per_batch,
                                grad_transition_dest_per_batch,
                                grad_transition_context,
                                grad_initial_log_belief,
                                grad_transition_gate_per_batch,
                                grad_transition_stay_per_batch);
                        }
                        break;
                    case 16:
                        launch_backward_chunk<scalar_t, 16, true>(
                            grad_beliefs,
                            carry,
                            transition_source_probs,
                            transition_dest_probs,
                            transition_context,
                            initial_log_belief,
                            beliefs,
                            transition_gate,
                            transition_stay_probs,
                            score_clamp_min,
                            score_clamp_max,
                            0,
                            scheduler.launch_chunk_size,
                            grad_local_logits,
                            grad_transition_source_per_batch,
                            grad_transition_dest_per_batch,
                            grad_transition_context,
                            grad_initial_log_belief,
                            grad_transition_gate_per_batch,
                            grad_transition_stay_per_batch);
                        break;
                    case 32:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk<scalar_t, 32, true>(
                                grad_beliefs,
                                carry,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                beliefs,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                grad_local_logits,
                                grad_transition_source_per_batch,
                                grad_transition_dest_per_batch,
                                grad_transition_context,
                                grad_initial_log_belief,
                                grad_transition_gate_per_batch,
                                grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk<scalar_t, 32>(
                                grad_beliefs,
                                carry,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                beliefs,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                grad_local_logits,
                                grad_transition_source_per_batch,
                                grad_transition_dest_per_batch,
                                grad_transition_context,
                                grad_initial_log_belief,
                                grad_transition_gate_per_batch,
                                grad_transition_stay_per_batch);
                        }
                        break;
                    case 64:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk<scalar_t, 64, true>(
                                grad_beliefs,
                                carry,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                beliefs,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                grad_local_logits,
                                grad_transition_source_per_batch,
                                grad_transition_dest_per_batch,
                                grad_transition_context,
                                grad_initial_log_belief,
                                grad_transition_gate_per_batch,
                                grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk<scalar_t, 64>(
                                grad_beliefs,
                                carry,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                beliefs,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                grad_local_logits,
                                grad_transition_source_per_batch,
                                grad_transition_dest_per_batch,
                                grad_transition_context,
                                grad_initial_log_belief,
                                grad_transition_gate_per_batch,
                                grad_transition_stay_per_batch);
                        }
                        break;
                    case 128:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk<scalar_t, 128, true>(
                                grad_beliefs,
                                carry,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                beliefs,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                grad_local_logits,
                                grad_transition_source_per_batch,
                                grad_transition_dest_per_batch,
                                grad_transition_context,
                                grad_initial_log_belief,
                                grad_transition_gate_per_batch,
                                grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk<scalar_t, 128>(
                                grad_beliefs,
                                carry,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                beliefs,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                grad_local_logits,
                                grad_transition_source_per_batch,
                                grad_transition_dest_per_batch,
                                grad_transition_context,
                                grad_initial_log_belief,
                                grad_transition_gate_per_batch,
                                grad_transition_stay_per_batch);
                        }
                        break;
                    default:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk<scalar_t, -1, true>(
                                grad_beliefs,
                                carry,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                beliefs,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                grad_local_logits,
                                grad_transition_source_per_batch,
                                grad_transition_dest_per_batch,
                                grad_transition_context,
                                grad_initial_log_belief,
                                grad_transition_gate_per_batch,
                                grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk<scalar_t>(
                                grad_beliefs,
                                carry,
                                transition_source_probs,
                                transition_dest_probs,
                                transition_context,
                                initial_log_belief,
                                beliefs,
                                transition_gate,
                                transition_stay_probs,
                                score_clamp_min,
                                score_clamp_max,
                                0,
                                scheduler.launch_chunk_size,
                                grad_local_logits,
                                grad_transition_source_per_batch,
                                grad_transition_dest_per_batch,
                                grad_transition_context,
                                grad_initial_log_belief,
                                grad_transition_gate_per_batch,
                                grad_transition_stay_per_batch);
                        }
                        break;
                }
            });
    }

    grad_transition_source_probs = grad_transition_source_per_batch.sum(0);
    grad_transition_dest_probs = grad_transition_dest_per_batch.sum(0);
    grad_transition_stay_probs = grad_transition_stay_per_batch.sum(0);
    grad_transition_gate = grad_transition_gate_per_batch.sum().reshape({1});
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

std::vector<torch::Tensor> causal_machine_scan_backward_logits_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    auto transition_source_probs = torch::empty_like(transition_source_logits);
    auto transition_dest_probs = torch::empty_like(transition_dest_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 block(kMaxNumStates);
    const dim3 source_grid(static_cast<unsigned int>(transition_source_logits.size(0)));
    const dim3 dest_grid(static_cast<unsigned int>(transition_dest_logits.size(0)));
    row_softmax_forward_kernel<<<source_grid, block, 0, stream>>>(
        transition_source_logits.data_ptr<float>(),
        static_cast<int>(transition_source_logits.size(1)),
        transition_source_probs.data_ptr<float>());
    row_softmax_forward_kernel<<<dest_grid, block, 0, stream>>>(
        transition_dest_logits.data_ptr<float>(),
        static_cast<int>(transition_dest_logits.size(1)),
        transition_dest_probs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto grads = causal_machine_scan_backward_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate,
        transition_stay_probs,
        chunk_size,
        score_clamp_min,
        score_clamp_max);
    auto grad_transition_source_logits = torch::empty_like(transition_source_logits);
    auto grad_transition_dest_logits = torch::empty_like(transition_dest_logits);
    row_softmax_backward_kernel<<<source_grid, block, 0, stream>>>(
        grads[1].data_ptr<float>(),
        transition_source_probs.data_ptr<float>(),
        static_cast<int>(transition_source_probs.size(1)),
        grad_transition_source_logits.data_ptr<float>());
    row_softmax_backward_kernel<<<dest_grid, block, 0, stream>>>(
        grads[2].data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        static_cast<int>(transition_dest_probs.size(1)),
        grad_transition_dest_logits.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    grads[1] = grad_transition_source_logits;
    grads[2] = grad_transition_dest_logits;
    return grads;
}

std::vector<torch::Tensor> causal_machine_scan_backward_composable_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    const auto num_states = beliefs.size(2);
    const auto transition_rank = transition_source_probs.size(1);
    const bool direct_small_rank_grad = can_use_direct_grad_reduce(
        beliefs.get_device(),
        static_cast<int>(num_states),
        static_cast<int>(transition_rank));
    const int64_t direct_staging_worker_blocks = direct_small_rank_grad
        ? small_state_direct_staging_worker_blocks(beliefs.get_device(), batch_size)
        : 0;
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_transition_source_probs = torch::zeros({num_states, transition_rank}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs = torch::zeros({transition_rank, num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_probs = torch::zeros({num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_source_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, num_states, transition_rank}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, num_states, transition_rank}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, transition_rank, num_states}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, transition_rank, num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, num_states}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto carry = grad_final_belief.contiguous();
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size, true);
    if (seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief);
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            beliefs.scalar_type(),
            "causal_machine_scan_backward_composable_logits_chunk",
            [&] {
                switch (transition_rank) {
                    case 8:
                        launch_backward_composable_chunk<scalar_t, 8, true>(
                            grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                            initial_log_belief, beliefs, transition_stay_probs, 0, scheduler.launch_chunk_size,
                            grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                            grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        break;
                    case 16:
                        launch_backward_composable_chunk<scalar_t, 16, true>(
                            grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                            initial_log_belief, beliefs, transition_stay_probs, 0, scheduler.launch_chunk_size,
                            grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                            grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        break;
                    case 32:
                        if (direct_small_rank_grad) {
                            launch_backward_composable_chunk<scalar_t, 32, true>(
                                grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                                initial_log_belief, beliefs, transition_stay_probs, 0, scheduler.launch_chunk_size,
                                grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                                grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_composable_chunk<scalar_t, 32, false>(
                                grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                                initial_log_belief, beliefs, transition_stay_probs, 0, scheduler.launch_chunk_size,
                                grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                                grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        }
                        break;
                    case 64:
                        if (direct_small_rank_grad) {
                            launch_backward_composable_chunk<scalar_t, 64, true>(
                                grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                                initial_log_belief, beliefs, transition_stay_probs, 0, scheduler.launch_chunk_size,
                                grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                                grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_composable_chunk<scalar_t, 64, false>(
                                grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                                initial_log_belief, beliefs, transition_stay_probs, 0, scheduler.launch_chunk_size,
                                grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                                grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        }
                        break;
                    case 128:
                        if (direct_small_rank_grad) {
                            launch_backward_composable_chunk<scalar_t, 128, true>(
                                grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                                initial_log_belief, beliefs, transition_stay_probs, 0, scheduler.launch_chunk_size,
                                grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                                grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_composable_chunk<scalar_t, 128, false>(
                                grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                                initial_log_belief, beliefs, transition_stay_probs, 0, scheduler.launch_chunk_size,
                                grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                                grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        }
                        break;
                    default:
                        if (direct_small_rank_grad) {
                            launch_backward_composable_chunk<scalar_t, -1, true>(
                                grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                                initial_log_belief, beliefs, transition_stay_probs, 0, seq_len,
                                grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                                grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_composable_chunk<scalar_t, -1, false>(
                                grad_beliefs, carry, transition_source_probs, transition_dest_probs, transition_context,
                                initial_log_belief, beliefs, transition_stay_probs, 0, seq_len,
                                grad_local_logits, grad_transition_source_per_batch, grad_transition_dest_per_batch,
                                grad_transition_context, grad_initial_log_belief, grad_transition_stay_per_batch);
                        }
                        break;
                }
            });
    }

    grad_transition_source_probs = grad_transition_source_per_batch.sum(0);
    grad_transition_dest_probs = grad_transition_dest_per_batch.sum(0);
    grad_transition_stay_probs = grad_transition_stay_per_batch.sum(0);
    return {
        grad_local_logits,
        grad_transition_source_probs,
        grad_transition_dest_probs,
        grad_transition_context,
        grad_initial_log_belief,
        torch::zeros({1}, beliefs.options().dtype(torch::kFloat32)),
        grad_transition_stay_probs,
    };
}

std::vector<torch::Tensor> causal_machine_scan_backward_composable_logits_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    auto transition_source_probs = torch::empty_like(transition_source_logits);
    auto transition_dest_probs = torch::empty_like(transition_dest_logits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 block(kMaxNumStates);
    const dim3 source_grid(static_cast<unsigned int>(transition_source_logits.size(0)));
    const dim3 dest_grid(static_cast<unsigned int>(transition_dest_logits.size(0)));
    row_softmax_forward_kernel<<<source_grid, block, 0, stream>>>(
        transition_source_logits.data_ptr<float>(),
        static_cast<int>(transition_source_logits.size(1)),
        transition_source_probs.data_ptr<float>());
    row_softmax_forward_kernel<<<dest_grid, block, 0, stream>>>(
        transition_dest_logits.data_ptr<float>(),
        static_cast<int>(transition_dest_logits.size(1)),
        transition_dest_probs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto grads = causal_machine_scan_backward_composable_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_stay_probs,
        chunk_size);
    auto grad_transition_source_logits = torch::empty_like(transition_source_logits);
    auto grad_transition_dest_logits = torch::empty_like(transition_dest_logits);
    row_softmax_backward_kernel<<<source_grid, block, 0, stream>>>(
        grads[1].data_ptr<float>(),
        transition_source_probs.data_ptr<float>(),
        static_cast<int>(transition_source_probs.size(1)),
        grad_transition_source_logits.data_ptr<float>());
    row_softmax_backward_kernel<<<dest_grid, block, 0, stream>>>(
        grads[2].data_ptr<float>(),
        transition_dest_probs.data_ptr<float>(),
        static_cast<int>(transition_dest_probs.size(1)),
        grad_transition_dest_logits.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    grads[1] = grad_transition_source_logits;
    grads[2] = grad_transition_dest_logits;
    return grads;
}

std::vector<torch::Tensor> causal_machine_scan_forward_quantized_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    const auto transition_rank = transition_source_q.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty_like(initial_log_belief);
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size);
    {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            local_logits.scalar_type(),
            "causal_machine_scan_forward_quantized_chunk",
            [&] {
                switch (transition_rank) {
                    case 8:
                        launch_forward_chunk_quantized<scalar_t, 8>(
                            local_logits, transition_source_q, transition_source_scales, transition_dest_q,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 16:
                        launch_forward_chunk_quantized<scalar_t, 16>(
                            local_logits, transition_source_q, transition_source_scales, transition_dest_q,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 32:
                        launch_forward_chunk_quantized<scalar_t, 32>(
                            local_logits, transition_source_q, transition_source_scales, transition_dest_q,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 64:
                        launch_forward_chunk_quantized<scalar_t, 64>(
                            local_logits, transition_source_q, transition_source_scales, transition_dest_q,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 128:
                        launch_forward_chunk_quantized<scalar_t, 128>(
                            local_logits, transition_source_q, transition_source_scales, transition_dest_q,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    default:
                        launch_forward_chunk_quantized<scalar_t>(
                            local_logits, transition_source_q, transition_source_scales, transition_dest_q,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                }
            });
    }
    return {beliefs, final_log_belief};
}

template <PackedTransitionFormat Format>
std::vector<torch::Tensor> causal_machine_scan_forward_fp8_cuda_impl(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(local_logits.device());
    const auto seq_len = local_logits.size(1);
    const auto transition_rank = transition_source_packed.size(1);
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty_like(initial_log_belief);
    if (seq_len == 0) {
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size);
    {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            local_logits.scalar_type(),
            "causal_machine_scan_forward_fp8_chunk",
            [&] {
                switch (transition_rank) {
                    case 8:
                        launch_forward_chunk_fp8<Format, scalar_t, 8>(
                            local_logits, transition_source_packed, transition_source_scales, transition_dest_packed,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 16:
                        launch_forward_chunk_fp8<Format, scalar_t, 16>(
                            local_logits, transition_source_packed, transition_source_scales, transition_dest_packed,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 32:
                        launch_forward_chunk_fp8<Format, scalar_t, 32>(
                            local_logits, transition_source_packed, transition_source_scales, transition_dest_packed,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 64:
                        launch_forward_chunk_fp8<Format, scalar_t, 64>(
                            local_logits, transition_source_packed, transition_source_scales, transition_dest_packed,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    case 128:
                        launch_forward_chunk_fp8<Format, scalar_t, 128>(
                            local_logits, transition_source_packed, transition_source_scales, transition_dest_packed,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                    default:
                        launch_forward_chunk_fp8<Format, scalar_t>(
                            local_logits, transition_source_packed, transition_source_scales, transition_dest_packed,
                            transition_dest_scales, transition_context, initial_log_belief, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, beliefs, final_log_belief);
                        break;
                }
            });
    }
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_forward_fp8_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    int64_t chunk_size) {
    TORCH_CHECK(fp8_format == 0 || fp8_format == 1, "fp8_format must be 0 (e4m3) or 1 (e5m2)");
    if (fp8_format == 0) {
        return causal_machine_scan_forward_fp8_cuda_impl<PackedTransitionFormat::Fp8E4M3>(
            local_logits,
            transition_source_packed,
            transition_source_scales,
            transition_dest_packed,
            transition_dest_scales,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            chunk_size);
    }
    return causal_machine_scan_forward_fp8_cuda_impl<PackedTransitionFormat::Fp8E5M2>(
        local_logits,
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_backward_quantized_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    const auto num_states = beliefs.size(2);
    const auto transition_rank = transition_source_q.size(1);
    const bool direct_small_rank_grad = can_use_direct_grad_reduce(
        beliefs.get_device(),
        static_cast<int>(num_states),
        static_cast<int>(transition_rank));
    const int64_t direct_staging_worker_blocks = direct_small_rank_grad
        ? small_state_direct_staging_worker_blocks(beliefs.get_device(), batch_size)
        : 0;
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_transition_source_probs = torch::zeros({num_states, transition_rank}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs = torch::zeros({transition_rank, num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_probs = torch::zeros({num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_source_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, num_states, transition_rank}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, num_states, transition_rank}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, transition_rank, num_states}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, transition_rank, num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, num_states}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, num_states}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_gate_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size}, beliefs.options().dtype(torch::kFloat32));
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto carry = grad_final_belief.contiguous();
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size, true);
    if (seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief);
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            beliefs.scalar_type(),
            "causal_machine_scan_backward_quantized_chunk",
            [&] {
                switch (transition_rank) {
                    case 8:
                        launch_backward_chunk_quantized<scalar_t, 8, true>(
                            grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                            transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                            grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                            grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        break;
                    case 16:
                        launch_backward_chunk_quantized<scalar_t, 16, true>(
                            grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                            transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                            grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                            grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        break;
                    case 32:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk_quantized<scalar_t, 32, true>(
                                grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk_quantized<scalar_t, 32>(
                                grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        }
                        break;
                    case 64:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk_quantized<scalar_t, 64, true>(
                                grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk_quantized<scalar_t, 64>(
                                grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        }
                        break;
                    case 128:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk_quantized<scalar_t, 128, true>(
                                grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk_quantized<scalar_t, 128>(
                                grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        }
                        break;
                    default:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk_quantized<scalar_t, -1, true>(
                                grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk_quantized<scalar_t>(
                                grad_beliefs, carry, transition_source_q, transition_source_scales, transition_dest_q,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        }
                        break;
                }
            });
    }

    grad_transition_source_probs = grad_transition_source_per_batch.sum(0);
    grad_transition_dest_probs = grad_transition_dest_per_batch.sum(0);
    grad_transition_stay_probs = grad_transition_stay_per_batch.sum(0);
    grad_transition_gate = grad_transition_gate_per_batch.sum().reshape({1});
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

template <PackedTransitionFormat Format>
std::vector<torch::Tensor> causal_machine_scan_backward_fp8_cuda_impl(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    c10::cuda::CUDAGuard device_guard(beliefs.device());
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    const auto transition_rank = transition_source_packed.size(1);
    const bool direct_small_rank_grad = can_use_direct_grad_reduce(
        beliefs.get_device(),
        kMaxNumStates,
        static_cast<int>(transition_rank));
    const int64_t direct_staging_worker_blocks = direct_small_rank_grad
        ? small_state_direct_staging_worker_blocks(beliefs.get_device(), batch_size)
        : 0;
    auto grad_local_logits = torch::zeros_like(beliefs);
    auto grad_transition_context = torch::zeros_like(transition_context);
    auto grad_transition_source_probs = torch::zeros({kMaxNumStates, transition_rank}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_probs = torch::zeros({transition_rank, kMaxNumStates}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_probs = torch::zeros({kMaxNumStates}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_gate = torch::zeros({1}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_source_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, kMaxNumStates, transition_rank}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, kMaxNumStates, transition_rank}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_dest_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, transition_rank, kMaxNumStates}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, transition_rank, kMaxNumStates}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_stay_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks, kMaxNumStates}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size, kMaxNumStates}, beliefs.options().dtype(torch::kFloat32));
    auto grad_transition_gate_per_batch = direct_small_rank_grad
        ? torch::zeros({direct_staging_worker_blocks}, beliefs.options().dtype(torch::kFloat32))
        : torch::zeros({batch_size}, beliefs.options().dtype(torch::kFloat32));
    auto grad_initial_log_belief = torch::zeros_like(initial_log_belief);
    auto carry = grad_final_belief.contiguous();
    const auto scheduler = make_scan_chunk_scheduler(seq_len, chunk_size, true);
    if (seq_len == 0) {
        grad_initial_log_belief.copy_(grad_final_belief);
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            beliefs.scalar_type(),
            "causal_machine_scan_backward_fp8_chunk",
            ([&] {
                switch (transition_rank) {
                    case 8:
                        launch_backward_chunk_fp8<Format, scalar_t, 8, true>(
                            grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                            transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                            grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                            grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        break;
                    case 16:
                        launch_backward_chunk_fp8<Format, scalar_t, 16, true>(
                            grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                            transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                            transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                            grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                            grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        break;
                    case 32:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk_fp8<Format, scalar_t, 32, true>(
                                grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk_fp8<Format, scalar_t, 32>(
                                grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        }
                        break;
                    case 64:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk_fp8<Format, scalar_t, 64, true>(
                                grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk_fp8<Format, scalar_t, 64>(
                                grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        }
                        break;
                    case 128:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk_fp8<Format, scalar_t, 128, true>(
                                grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk_fp8<Format, scalar_t, 128>(
                                grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        }
                        break;
                    default:
                        if (direct_small_rank_grad) {
                            launch_backward_chunk_fp8<Format, scalar_t, -1, true>(
                                grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        } else {
                            launch_backward_chunk_fp8<Format, scalar_t>(
                                grad_beliefs, carry, transition_source_packed, transition_source_scales, transition_dest_packed,
                                transition_dest_scales, transition_context, initial_log_belief, beliefs, transition_gate,
                                transition_stay_probs, 0, scheduler.launch_chunk_size, grad_local_logits, grad_transition_source_per_batch,
                                grad_transition_dest_per_batch, grad_transition_context, grad_initial_log_belief,
                                grad_transition_gate_per_batch, grad_transition_stay_per_batch);
                        }
                        break;
                }
            }));
    }

    grad_transition_source_probs = grad_transition_source_per_batch.sum(0);
    grad_transition_dest_probs = grad_transition_dest_per_batch.sum(0);
    grad_transition_stay_probs = grad_transition_stay_per_batch.sum(0);
    grad_transition_gate = grad_transition_gate_per_batch.sum().reshape({1});
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

std::vector<torch::Tensor> causal_machine_scan_backward_fp8_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    int64_t chunk_size) {
    TORCH_CHECK(fp8_format == 0 || fp8_format == 1, "fp8_format must be 0 (e4m3) or 1 (e5m2)");
    if (fp8_format == 0) {
        return causal_machine_scan_backward_fp8_cuda_impl<PackedTransitionFormat::Fp8E4M3>(
            grad_beliefs,
            grad_final_belief,
            transition_source_packed,
            transition_source_scales,
            transition_dest_packed,
            transition_dest_scales,
            transition_context,
            initial_log_belief,
            beliefs,
            transition_gate,
            transition_stay_probs,
            chunk_size);
    }
    return causal_machine_scan_backward_fp8_cuda_impl<PackedTransitionFormat::Fp8E5M2>(
        grad_beliefs,
        grad_final_belief,
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate,
        transition_stay_probs,
        chunk_size);
}

void causal_machine_scan_record_paged_step_tensor_cuda(
    torch::Tensor paged_values,
    torch::Tensor values,
    int64_t num_updates) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(paged_values));
    const int64_t num_slots = static_cast<int64_t>(paged_values.size(0));
    const int64_t batch_size = static_cast<int64_t>(values.size(0));
    const int64_t feature_dim = static_cast<int64_t>(paged_values.size(2));
    const int64_t page_size = static_cast<int64_t>(paged_values.size(1));
    const int64_t max_pages = batch_size > 0 ? (num_slots / batch_size) : 0;
    if (batch_size == 0 || feature_dim == 0 || max_pages == 0 || page_size == 0) {
        return;
    }
    const int64_t page_idx = num_updates / page_size;
    if (page_idx >= max_pages) {
        return;
    }
    const int64_t page_offset = num_updates % page_size;
    const int64_t total = batch_size * feature_dim;
    const int blocks = static_cast<int>((total + kPagedCacheThreads - 1) / kPagedCacheThreads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        paged_values.scalar_type(),
        "causal_machine_scan_record_paged_step_tensor_cuda",
        ([&] {
            record_paged_step_tensor_kernel<scalar_t><<<blocks, kPagedCacheThreads, 0, stream>>>(
                paged_values.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                batch_size,
                num_slots,
                max_pages,
                page_size,
                feature_dim,
                page_idx,
                page_offset);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void causal_machine_scan_record_paged_step_tensor_from_lengths_cuda(
    torch::Tensor paged_values,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor values) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(paged_values));
    const int64_t num_slots = static_cast<int64_t>(paged_values.size(0));
    const int64_t batch_size = static_cast<int64_t>(paged_page_table.size(0));
    const int64_t max_pages = static_cast<int64_t>(paged_page_table.size(1));
    const int64_t page_size = static_cast<int64_t>(paged_values.size(1));
    const int64_t feature_dim = static_cast<int64_t>(paged_values.size(2));
    if (batch_size == 0 || feature_dim == 0 || max_pages == 0 || page_size == 0) {
        return;
    }
    const int64_t total = batch_size * feature_dim;
    const int blocks = static_cast<int>((total + kPagedCacheThreads - 1) / kPagedCacheThreads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        paged_values.scalar_type(),
        "causal_machine_scan_record_paged_step_tensor_from_lengths_cuda",
        ([&] {
            record_paged_step_tensor_from_lengths_kernel<scalar_t><<<blocks, kPagedCacheThreads, 0, stream>>>(
                paged_values.data_ptr<scalar_t>(),
                paged_page_table.data_ptr<int64_t>(),
                paged_lengths.data_ptr<int64_t>(),
                values.data_ptr<scalar_t>(),
                batch_size,
                num_slots,
                max_pages,
                page_size,
                feature_dim);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void causal_machine_scan_record_paged_sequence_tensor_cuda(
    torch::Tensor paged_values,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor values,
    int64_t num_updates) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(paged_values));
    (void)num_updates;
    const int64_t num_slots = static_cast<int64_t>(paged_values.size(0));
    const int64_t batch_size = static_cast<int64_t>(paged_page_table.size(0));
    const int64_t seq_len = static_cast<int64_t>(values.size(1));
    const int64_t max_pages = static_cast<int64_t>(paged_page_table.size(1));
    const int64_t page_size = static_cast<int64_t>(paged_values.size(1));
    const int64_t feature_dim = static_cast<int64_t>(paged_values.size(2));
    if (batch_size == 0 || seq_len == 0 || feature_dim == 0 || max_pages == 0 || page_size == 0) {
        return;
    }
    const int64_t total = batch_size * seq_len * feature_dim;
    const int blocks = static_cast<int>((total + kPagedCacheThreads - 1) / kPagedCacheThreads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        paged_values.scalar_type(),
        "causal_machine_scan_record_paged_sequence_tensor_cuda",
        ([&] {
            record_paged_sequence_tensor_kernel<scalar_t><<<blocks, kPagedCacheThreads, 0, stream>>>(
                paged_values.data_ptr<scalar_t>(),
                paged_page_table.data_ptr<int64_t>(),
                paged_lengths.data_ptr<int64_t>(),
                values.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                num_slots,
                max_pages,
                page_size,
                feature_dim);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void causal_machine_scan_increment_paged_lengths_cuda(
    torch::Tensor paged_lengths,
    int64_t delta,
    int64_t capacity) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(paged_lengths));
    const auto batch_size = paged_lengths.size(0);
    if (batch_size == 0) {
        return;
    }
    const int blocks = static_cast<int>((batch_size + kPagedCacheThreads - 1) / kPagedCacheThreads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    increment_paged_lengths_kernel<<<blocks, kPagedCacheThreads, 0, stream>>>(
        paged_lengths.data_ptr<int64_t>(),
        batch_size,
        delta,
        capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void causal_machine_scan_read_paged_latest_tensor_cuda(
    torch::Tensor paged_values,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor values) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(paged_values));
    const int64_t num_slots = static_cast<int64_t>(paged_values.size(0));
    const int64_t batch_size = static_cast<int64_t>(paged_page_table.size(0));
    const int64_t max_pages = static_cast<int64_t>(paged_page_table.size(1));
    const int64_t page_size = static_cast<int64_t>(paged_values.size(1));
    const int64_t feature_dim = static_cast<int64_t>(paged_values.size(2));
    if (batch_size == 0 || feature_dim == 0 || max_pages == 0 || page_size == 0) {
        values.zero_();
        return;
    }
    const int64_t total = batch_size * feature_dim;
    const int blocks = static_cast<int>((total + kPagedCacheThreads - 1) / kPagedCacheThreads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        paged_values.scalar_type(),
        "causal_machine_scan_read_paged_latest_tensor_cuda",
        ([&] {
            read_paged_latest_tensor_kernel<scalar_t><<<blocks, kPagedCacheThreads, 0, stream>>>(
                paged_values.data_ptr<scalar_t>(),
                paged_page_table.data_ptr<int64_t>(),
                paged_lengths.data_ptr<int64_t>(),
                values.data_ptr<scalar_t>(),
                batch_size,
                num_slots,
                max_pages,
                page_size,
                feature_dim);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> causal_machine_scan_paged_step_dense_128_rank8_cuda(
    torch::Tensor paged_log_beliefs,
    torch::Tensor paged_latent_states,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor transition_stay_probs,
    double transition_gate,
    double score_clamp_min,
    double score_clamp_max) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(local_logits));
    const int64_t batch_size = static_cast<int64_t>(local_logits.size(0));
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(
        {local_logits.size(0), local_logits.size(2)},
        local_logits.options().dtype(torch::kFloat32));
    if (batch_size == 0) {
        return {beliefs, final_log_belief};
    }
    const int64_t num_slots = static_cast<int64_t>(paged_log_beliefs.size(0));
    const int64_t max_pages = static_cast<int64_t>(paged_page_table.size(1));
    const int64_t page_size = static_cast<int64_t>(paged_log_beliefs.size(1));
    const bool has_paged_latent_states = paged_latent_states.defined() && paged_latent_states.numel() > 0;
    const int64_t latent_feature_dim = has_paged_latent_states
        ? static_cast<int64_t>(paged_latent_states.size(2))
        : 0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_paged_step_dense_128_rank8_cuda",
        ([&] {
            const bool use_pair_path = can_use_dense_128_rank8_pair_path<scalar_t>(local_logits.get_device());
            if (use_pair_path) {
                paged_step_dense_128_rank8_pair_kernel<scalar_t><<<
                    static_cast<unsigned int>(batch_size),
                    64,
                    forward_dense_128_rank8_shared_bytes(),
                    stream>>>(
                        paged_log_beliefs.data_ptr<scalar_t>(),
                        has_paged_latent_states ? paged_latent_states.data_ptr<scalar_t>() : nullptr,
                        paged_page_table.data_ptr<int64_t>(),
                        paged_lengths.data_ptr<int64_t>(),
                        local_logits.data_ptr<scalar_t>(),
                        transition_source_probs.data_ptr<float>(),
                        transition_dest_probs.data_ptr<float>(),
                        transition_context.data_ptr<scalar_t>(),
                        transition_stay_probs.data_ptr<float>(),
                        static_cast<float>(transition_gate),
                        static_cast<float>(score_clamp_min),
                        static_cast<float>(score_clamp_max),
                        batch_size,
                        num_slots,
                        max_pages,
                        page_size,
                        latent_feature_dim,
                        beliefs.data_ptr<scalar_t>(),
                        final_log_belief.data_ptr<float>());
            } else {
                paged_step_dense_128_rank8_kernel<scalar_t><<<
                    static_cast<unsigned int>(batch_size),
                    128,
                    forward_dense_128_rank8_shared_bytes(),
                    stream>>>(
                        paged_log_beliefs.data_ptr<scalar_t>(),
                        has_paged_latent_states ? paged_latent_states.data_ptr<scalar_t>() : nullptr,
                        paged_page_table.data_ptr<int64_t>(),
                        paged_lengths.data_ptr<int64_t>(),
                        local_logits.data_ptr<scalar_t>(),
                        transition_source_probs.data_ptr<float>(),
                        transition_dest_probs.data_ptr<float>(),
                        transition_context.data_ptr<scalar_t>(),
                        transition_stay_probs.data_ptr<float>(),
                        static_cast<float>(transition_gate),
                        static_cast<float>(score_clamp_min),
                        static_cast<float>(score_clamp_max),
                        batch_size,
                        num_slots,
                        max_pages,
                        page_size,
                        latent_feature_dim,
                        beliefs.data_ptr<scalar_t>(),
                        final_log_belief.data_ptr<float>());
            }
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {beliefs, final_log_belief};
}

template <typename packed_t, PackedTransitionFormat Format>
std::vector<torch::Tensor> causal_machine_scan_paged_step_packed_cuda_impl(
    torch::Tensor paged_log_beliefs,
    torch::Tensor paged_latent_states,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor transition_stay_probs,
    double transition_gate) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(local_logits));
    const int64_t batch_size = static_cast<int64_t>(local_logits.size(0));
    const int64_t num_states = static_cast<int64_t>(local_logits.size(2));
    const int64_t transition_rank = static_cast<int64_t>(transition_source_packed.size(1));
    auto beliefs = torch::empty_like(local_logits);
    auto final_log_belief = torch::empty(
        {local_logits.size(0), local_logits.size(2)},
        local_logits.options().dtype(torch::kFloat32));
    if (batch_size == 0) {
        return {beliefs, final_log_belief};
    }
    const int64_t num_slots = static_cast<int64_t>(paged_log_beliefs.size(0));
    const int64_t max_pages = static_cast<int64_t>(paged_page_table.size(1));
    const int64_t page_size = static_cast<int64_t>(paged_log_beliefs.size(1));
    const bool has_paged_latent_states = paged_latent_states.defined() && paged_latent_states.numel() > 0;
    const int64_t latent_feature_dim = has_paged_latent_states
        ? static_cast<int64_t>(paged_latent_states.size(2))
        : 0;
    const int block_threads = std::max(
        kWarpSize,
        std::min(256, round_up_pow2(static_cast<int>(std::max<int64_t>(num_states, 1)))));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const size_t shared_bytes = forward_packed_chunk_shared_bytes(
        static_cast<int>(num_states),
        static_cast<int>(transition_rank));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        local_logits.scalar_type(),
        "causal_machine_scan_paged_step_packed_cuda_impl",
        ([&] {
            switch (transition_rank) {
                case 8:
                    paged_step_packed_kernel<scalar_t, packed_t, Format, 8><<<
                        static_cast<unsigned int>(batch_size),
                        block_threads,
                        shared_bytes,
                        stream>>>(
                            paged_log_beliefs.data_ptr<scalar_t>(),
                            has_paged_latent_states ? paged_latent_states.data_ptr<scalar_t>() : nullptr,
                            paged_page_table.data_ptr<int64_t>(),
                            paged_lengths.data_ptr<int64_t>(),
                            local_logits.data_ptr<scalar_t>(),
                            transition_source_packed.data_ptr<packed_t>(),
                            transition_source_scales.data_ptr<float>(),
                            transition_dest_packed.data_ptr<packed_t>(),
                            transition_dest_scales.data_ptr<float>(),
                            transition_context.data_ptr<scalar_t>(),
                            transition_stay_probs.data_ptr<float>(),
                            static_cast<float>(transition_gate),
                            static_cast<int>(transition_rank),
                            batch_size,
                            num_states,
                            num_slots,
                            max_pages,
                            page_size,
                            latent_feature_dim,
                            beliefs.data_ptr<scalar_t>(),
                            final_log_belief.data_ptr<float>());
                    break;
                case 16:
                    paged_step_packed_kernel<scalar_t, packed_t, Format, 16><<<
                        static_cast<unsigned int>(batch_size),
                        block_threads,
                        shared_bytes,
                        stream>>>(
                            paged_log_beliefs.data_ptr<scalar_t>(),
                            has_paged_latent_states ? paged_latent_states.data_ptr<scalar_t>() : nullptr,
                            paged_page_table.data_ptr<int64_t>(),
                            paged_lengths.data_ptr<int64_t>(),
                            local_logits.data_ptr<scalar_t>(),
                            transition_source_packed.data_ptr<packed_t>(),
                            transition_source_scales.data_ptr<float>(),
                            transition_dest_packed.data_ptr<packed_t>(),
                            transition_dest_scales.data_ptr<float>(),
                            transition_context.data_ptr<scalar_t>(),
                            transition_stay_probs.data_ptr<float>(),
                            static_cast<float>(transition_gate),
                            static_cast<int>(transition_rank),
                            batch_size,
                            num_states,
                            num_slots,
                            max_pages,
                            page_size,
                            latent_feature_dim,
                            beliefs.data_ptr<scalar_t>(),
                            final_log_belief.data_ptr<float>());
                    break;
                case 32:
                    paged_step_packed_kernel<scalar_t, packed_t, Format, 32><<<
                        static_cast<unsigned int>(batch_size),
                        block_threads,
                        shared_bytes,
                        stream>>>(
                            paged_log_beliefs.data_ptr<scalar_t>(),
                            has_paged_latent_states ? paged_latent_states.data_ptr<scalar_t>() : nullptr,
                            paged_page_table.data_ptr<int64_t>(),
                            paged_lengths.data_ptr<int64_t>(),
                            local_logits.data_ptr<scalar_t>(),
                            transition_source_packed.data_ptr<packed_t>(),
                            transition_source_scales.data_ptr<float>(),
                            transition_dest_packed.data_ptr<packed_t>(),
                            transition_dest_scales.data_ptr<float>(),
                            transition_context.data_ptr<scalar_t>(),
                            transition_stay_probs.data_ptr<float>(),
                            static_cast<float>(transition_gate),
                            static_cast<int>(transition_rank),
                            batch_size,
                            num_states,
                            num_slots,
                            max_pages,
                            page_size,
                            latent_feature_dim,
                            beliefs.data_ptr<scalar_t>(),
                            final_log_belief.data_ptr<float>());
                    break;
                case 64:
                    paged_step_packed_kernel<scalar_t, packed_t, Format, 64><<<
                        static_cast<unsigned int>(batch_size),
                        block_threads,
                        shared_bytes,
                        stream>>>(
                            paged_log_beliefs.data_ptr<scalar_t>(),
                            has_paged_latent_states ? paged_latent_states.data_ptr<scalar_t>() : nullptr,
                            paged_page_table.data_ptr<int64_t>(),
                            paged_lengths.data_ptr<int64_t>(),
                            local_logits.data_ptr<scalar_t>(),
                            transition_source_packed.data_ptr<packed_t>(),
                            transition_source_scales.data_ptr<float>(),
                            transition_dest_packed.data_ptr<packed_t>(),
                            transition_dest_scales.data_ptr<float>(),
                            transition_context.data_ptr<scalar_t>(),
                            transition_stay_probs.data_ptr<float>(),
                            static_cast<float>(transition_gate),
                            static_cast<int>(transition_rank),
                            batch_size,
                            num_states,
                            num_slots,
                            max_pages,
                            page_size,
                            latent_feature_dim,
                            beliefs.data_ptr<scalar_t>(),
                            final_log_belief.data_ptr<float>());
                    break;
                case 128:
                    paged_step_packed_kernel<scalar_t, packed_t, Format, 128><<<
                        static_cast<unsigned int>(batch_size),
                        block_threads,
                        shared_bytes,
                        stream>>>(
                            paged_log_beliefs.data_ptr<scalar_t>(),
                            has_paged_latent_states ? paged_latent_states.data_ptr<scalar_t>() : nullptr,
                            paged_page_table.data_ptr<int64_t>(),
                            paged_lengths.data_ptr<int64_t>(),
                            local_logits.data_ptr<scalar_t>(),
                            transition_source_packed.data_ptr<packed_t>(),
                            transition_source_scales.data_ptr<float>(),
                            transition_dest_packed.data_ptr<packed_t>(),
                            transition_dest_scales.data_ptr<float>(),
                            transition_context.data_ptr<scalar_t>(),
                            transition_stay_probs.data_ptr<float>(),
                            static_cast<float>(transition_gate),
                            static_cast<int>(transition_rank),
                            batch_size,
                            num_states,
                            num_slots,
                            max_pages,
                            page_size,
                            latent_feature_dim,
                            beliefs.data_ptr<scalar_t>(),
                            final_log_belief.data_ptr<float>());
                    break;
                default:
                    paged_step_packed_kernel<scalar_t, packed_t, Format><<<
                        static_cast<unsigned int>(batch_size),
                        block_threads,
                        shared_bytes,
                        stream>>>(
                            paged_log_beliefs.data_ptr<scalar_t>(),
                            has_paged_latent_states ? paged_latent_states.data_ptr<scalar_t>() : nullptr,
                            paged_page_table.data_ptr<int64_t>(),
                            paged_lengths.data_ptr<int64_t>(),
                            local_logits.data_ptr<scalar_t>(),
                            transition_source_packed.data_ptr<packed_t>(),
                            transition_source_scales.data_ptr<float>(),
                            transition_dest_packed.data_ptr<packed_t>(),
                            transition_dest_scales.data_ptr<float>(),
                            transition_context.data_ptr<scalar_t>(),
                            transition_stay_probs.data_ptr<float>(),
                            static_cast<float>(transition_gate),
                            static_cast<int>(transition_rank),
                            batch_size,
                            num_states,
                            num_slots,
                            max_pages,
                            page_size,
                            latent_feature_dim,
                            beliefs.data_ptr<scalar_t>(),
                            final_log_belief.data_ptr<float>());
                    break;
            }
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {beliefs, final_log_belief};
}

std::vector<torch::Tensor> causal_machine_scan_paged_step_quantized_cuda(
    torch::Tensor paged_log_beliefs,
    torch::Tensor paged_latent_states,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor transition_stay_probs,
    double transition_gate) {
    return causal_machine_scan_paged_step_packed_cuda_impl<int8_t, PackedTransitionFormat::Int8>(
        paged_log_beliefs,
        paged_latent_states,
        paged_page_table,
        paged_lengths,
        local_logits,
        transition_source_q,
        transition_source_scales,
        transition_dest_q,
        transition_dest_scales,
        transition_context,
        transition_stay_probs,
        transition_gate);
}

std::vector<torch::Tensor> causal_machine_scan_paged_step_fp8_cuda(
    torch::Tensor paged_log_beliefs,
    torch::Tensor paged_latent_states,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor transition_stay_probs,
    double transition_gate,
    int64_t fp8_format) {
    TORCH_CHECK(fp8_format == 0 || fp8_format == 1, "fp8_format must be 0 (e4m3) or 1 (e5m2)");
    if (fp8_format == 0) {
        return causal_machine_scan_paged_step_packed_cuda_impl<uint8_t, PackedTransitionFormat::Fp8E4M3>(
            paged_log_beliefs,
            paged_latent_states,
            paged_page_table,
            paged_lengths,
            local_logits,
            transition_source_packed,
            transition_source_scales,
            transition_dest_packed,
            transition_dest_scales,
            transition_context,
            transition_stay_probs,
            transition_gate);
    }
    return causal_machine_scan_paged_step_packed_cuda_impl<uint8_t, PackedTransitionFormat::Fp8E5M2>(
        paged_log_beliefs,
        paged_latent_states,
        paged_page_table,
        paged_lengths,
        local_logits,
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        transition_context,
        transition_stay_probs,
        transition_gate);
}

void causal_machine_scan_reorder_paged_cache_cuda(
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor beam_indices) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(paged_page_table));
    const auto batch_size = paged_page_table.size(0);
    const auto max_pages = paged_page_table.size(1);
    if (batch_size == 0 || max_pages == 0) {
        return;
    }
    auto reordered_page_table = torch::empty_like(paged_page_table);
    auto reordered_lengths = torch::empty_like(paged_lengths);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t total_table = batch_size * max_pages;
    const int table_blocks = static_cast<int>((total_table + kPagedCacheThreads - 1) / kPagedCacheThreads);
    const int length_blocks = static_cast<int>((batch_size + kPagedCacheThreads - 1) / kPagedCacheThreads);
    reorder_paged_page_table_kernel<<<table_blocks, kPagedCacheThreads, 0, stream>>>(
        paged_page_table.data_ptr<int64_t>(),
        beam_indices.data_ptr<int64_t>(),
        reordered_page_table.data_ptr<int64_t>(),
        batch_size,
        max_pages);
    reorder_paged_lengths_kernel<<<length_blocks, kPagedCacheThreads, 0, stream>>>(
        paged_lengths.data_ptr<int64_t>(),
        beam_indices.data_ptr<int64_t>(),
        reordered_lengths.data_ptr<int64_t>(),
        batch_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    paged_page_table.copy_(reordered_page_table);
    paged_lengths.copy_(reordered_lengths);
}
