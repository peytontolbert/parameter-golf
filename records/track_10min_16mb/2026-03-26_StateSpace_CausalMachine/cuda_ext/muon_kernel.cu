#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/ops/baddbmm.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/norm.h>

#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int64_t kMuonFamilySquare = 0;
constexpr int64_t kMuonFamilyTall = 1;
constexpr int64_t kMuonFamilyWideSmall = 2;
constexpr int64_t kMuonFamilyWideLarge = 3;

enum class MuonSquareBackend : int {
    kAuto = 0,
    kCublas = 1,
    kCublasLt = 2,
    kHybrid = 3,
};

inline bool muon_family_transposes(int64_t family_code) {
    return family_code == kMuonFamilyTall;
}

inline int muon_family_threads(int64_t family_code) {
    return family_code == kMuonFamilyWideSmall ? 128 : 256;
}

inline void check_cublas_status(cublasStatus_t status, const char* op_name) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, op_name, " failed with cuBLAS status ", static_cast<int>(status));
}

inline void check_cublaslt_status(cublasStatus_t status, const char* op_name) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, op_name, " failed with cuBLAS Lt status ", static_cast<int>(status));
}

constexpr size_t kDefaultMuonCublasLtWorkspaceBytes = 32 * 1024 * 1024;

size_t muon_cublaslt_workspace_bytes() {
    static const size_t workspace_bytes = []() -> size_t {
        const char* env = std::getenv("MUON_CUBLASLT_WORKSPACE_BYTES");
        if (env == nullptr || *env == '\0') {
            return kDefaultMuonCublasLtWorkspaceBytes;
        }
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end == env || (end != nullptr && *end != '\0') || parsed == 0) {
            return kDefaultMuonCublasLtWorkspaceBytes;
        }
        return static_cast<size_t>(parsed);
    }();
    return workspace_bytes;
}

bool muon_can_use_cublaslt_square(const torch::Tensor& x) {
    if (!x.is_cuda() || x.scalar_type() != torch::kBFloat16 || x.dim() != 3) {
        return false;
    }
    if (x.size(1) != x.size(2) || x.size(1) <= 0) {
        return false;
    }
    const auto* props = at::cuda::getCurrentDeviceProperties();
    return props != nullptr && props->major >= 8;
}

MuonSquareBackend muon_square_backend_policy() {
    const char* env = std::getenv("MUON_SQUARE_BACKEND");
    if (env == nullptr || *env == '\0') {
        return MuonSquareBackend::kAuto;
    }
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "auto") {
        return MuonSquareBackend::kAuto;
    }
    if (value == "cublas") {
        return MuonSquareBackend::kCublas;
    }
    if (value == "cublaslt") {
        return MuonSquareBackend::kCublasLt;
    }
    if (value == "hybrid") {
        return MuonSquareBackend::kHybrid;
    }
    return MuonSquareBackend::kAuto;
}

MuonSquareBackend select_muon_square_backend(const torch::Tensor& x) {
    if (!muon_can_use_cublaslt_square(x)) {
        return MuonSquareBackend::kCublas;
    }
    const MuonSquareBackend policy = muon_square_backend_policy();
    if (
        policy == MuonSquareBackend::kCublas
        || policy == MuonSquareBackend::kCublasLt
        || policy == MuonSquareBackend::kHybrid) {
        return policy;
    }
    const int64_t batch = x.size(0);
    const int64_t dim = x.size(1);
    // The competition square buckets are typically 640x640 with bucket_size ~= 12.
    // Prefer the hybrid path there: cuBLAS for Gram products, cuBLASLt for fused update GEMMs.
    if (dim >= 512 && dim % 64 == 0 && batch >= 8) {
        return MuonSquareBackend::kHybrid;
    }
    if (dim >= 256 && dim % 32 == 0 && batch >= 4) {
        return MuonSquareBackend::kCublasLt;
    }
    return MuonSquareBackend::kCublas;
}

struct MuonLtSquarePlanKey {
    int device_index;
    int64_t batch;
    int64_t dim;

    bool operator==(const MuonLtSquarePlanKey& other) const {
        return device_index == other.device_index && batch == other.batch && dim == other.dim;
    }
};

struct MuonLtSquarePlanKeyHash {
    size_t operator()(const MuonLtSquarePlanKey& key) const {
        size_t seed = static_cast<size_t>(key.device_index);
        seed ^= static_cast<size_t>(key.batch) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        seed ^= static_cast<size_t>(key.dim) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct MuonLtSquarePlan {
    cublasLtMatmulDesc_t tn_desc = nullptr;
    cublasLtMatmulDesc_t nn_desc = nullptr;
    cublasLtMatrixLayout_t matrix_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t tn_algo{};
    cublasLtMatmulAlgo_t nn_algo{};
    torch::Tensor workspace;
    size_t workspace_bytes = 0;

    ~MuonLtSquarePlan() {
        if (preference != nullptr) {
            cublasLtMatmulPreferenceDestroy(preference);
        }
        if (matrix_layout != nullptr) {
            cublasLtMatrixLayoutDestroy(matrix_layout);
        }
        if (tn_desc != nullptr) {
            cublasLtMatmulDescDestroy(tn_desc);
        }
        if (nn_desc != nullptr) {
            cublasLtMatmulDescDestroy(nn_desc);
        }
    }
};

std::shared_ptr<MuonLtSquarePlan> get_muon_cublaslt_square_plan(int device_index, int64_t batch, int64_t dim) {
    static std::mutex cache_mutex;
    static std::unordered_map<MuonLtSquarePlanKey, std::shared_ptr<MuonLtSquarePlan>, MuonLtSquarePlanKeyHash> cache;

    const MuonLtSquarePlanKey key{device_index, batch, dim};
    {
        std::lock_guard<std::mutex> guard(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    c10::cuda::CUDAGuard device_guard(device_index);
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    auto plan = std::make_shared<MuonLtSquarePlan>();
    plan->workspace_bytes = muon_cublaslt_workspace_bytes();

    check_cublaslt_status(
        cublasLtMatmulDescCreate(&plan->tn_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "cublasLtMatmulDescCreate(tn)");
    check_cublaslt_status(
        cublasLtMatmulDescCreate(&plan->nn_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "cublasLtMatmulDescCreate(nn)");
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    check_cublaslt_status(
        cublasLtMatmulDescSetAttribute(plan->tn_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)),
        "cublasLtMatmulDescSetAttribute(tn, transa)");
    check_cublaslt_status(
        cublasLtMatmulDescSetAttribute(plan->tn_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)),
        "cublasLtMatmulDescSetAttribute(tn, transb)");
    check_cublaslt_status(
        cublasLtMatmulDescSetAttribute(plan->nn_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)),
        "cublasLtMatmulDescSetAttribute(nn, transa)");
    check_cublaslt_status(
        cublasLtMatmulDescSetAttribute(plan->nn_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)),
        "cublasLtMatmulDescSetAttribute(nn, transb)");

    check_cublaslt_status(
        cublasLtMatrixLayoutCreate(&plan->matrix_layout, CUDA_R_16BF, dim, dim, dim),
        "cublasLtMatrixLayoutCreate");
    TORCH_CHECK(batch <= std::numeric_limits<int32_t>::max(), "Muon cublasLt batch exceeds int32 range");
    const int32_t batch_count = static_cast<int32_t>(batch);
    const int64_t stride = dim * dim;
    check_cublaslt_status(
        cublasLtMatrixLayoutSetAttribute(
            plan->matrix_layout,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count,
            sizeof(batch_count)),
        "cublasLtMatrixLayoutSetAttribute(batch_count)");
    check_cublaslt_status(
        cublasLtMatrixLayoutSetAttribute(
            plan->matrix_layout,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride,
            sizeof(stride)),
        "cublasLtMatrixLayoutSetAttribute(stride)");

    check_cublaslt_status(
        cublasLtMatmulPreferenceCreate(&plan->preference),
        "cublasLtMatmulPreferenceCreate");
    check_cublaslt_status(
        cublasLtMatmulPreferenceSetAttribute(
            plan->preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &plan->workspace_bytes,
            sizeof(plan->workspace_bytes)),
        "cublasLtMatmulPreferenceSetAttribute(workspace)");

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    check_cublaslt_status(
        cublasLtMatmulAlgoGetHeuristic(
            handle,
            plan->tn_desc,
            plan->matrix_layout,
            plan->matrix_layout,
            plan->matrix_layout,
            plan->matrix_layout,
            plan->preference,
            1,
            &heuristic,
            &returned_results),
        "cublasLtMatmulAlgoGetHeuristic(tn)");
    TORCH_CHECK(returned_results > 0, "cublasLtMatmulAlgoGetHeuristic(tn) returned no algorithms");
    plan->tn_algo = heuristic.algo;

    check_cublaslt_status(
        cublasLtMatmulAlgoGetHeuristic(
            handle,
            plan->nn_desc,
            plan->matrix_layout,
            plan->matrix_layout,
            plan->matrix_layout,
            plan->matrix_layout,
            plan->preference,
            1,
            &heuristic,
            &returned_results),
        "cublasLtMatmulAlgoGetHeuristic(nn)");
    TORCH_CHECK(returned_results > 0, "cublasLtMatmulAlgoGetHeuristic(nn) returned no algorithms");
    plan->nn_algo = heuristic.algo;

    if (plan->workspace_bytes > 0) {
        plan->workspace = torch::empty(
            {static_cast<int64_t>(plan->workspace_bytes)},
            torch::TensorOptions().device(torch::kCUDA, device_index).dtype(torch::kUInt8));
    }

    std::lock_guard<std::mutex> guard(cache_mutex);
    auto [it, inserted] = cache.emplace(key, plan);
    return inserted ? plan : it->second;
}

void muon_cublaslt_square_matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor out,
    bool transpose_a,
    float alpha,
    float beta) {
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "muon_cublaslt_square_matmul expects bf16 inputs");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "muon_cublaslt_square_matmul expects bf16 inputs");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "muon_cublaslt_square_matmul expects bf16 outputs");
    const int device_index = a.get_device();
    auto plan = get_muon_cublaslt_square_plan(device_index, a.size(0), a.size(1));
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    const cublasLtMatmulDesc_t desc = transpose_a ? plan->tn_desc : plan->nn_desc;
    const cublasLtMatmulAlgo_t* algo = transpose_a ? &plan->tn_algo : &plan->nn_algo;
    void* workspace_ptr = plan->workspace.defined() ? plan->workspace.data_ptr() : nullptr;
    check_cublaslt_status(
        cublasLtMatmul(
            handle,
            desc,
            &alpha,
            a.data_ptr(),
            plan->matrix_layout,
            b.data_ptr(),
            plan->matrix_layout,
            &beta,
            out.data_ptr(),
            plan->matrix_layout,
            out.data_ptr(),
            plan->matrix_layout,
            algo,
            workspace_ptr,
            plan->workspace_bytes,
            at::cuda::getCurrentCUDAStream()),
        "cublasLtMatmul");
}

template <typename scalar_t>
__global__ void gather_same_shape_tensor_batch_kernel(
    const int64_t* src_ptrs,
    float* dst,
    int64_t bucket_size,
    int64_t rows,
    int64_t cols) {
    const int64_t total = bucket_size * rows * cols;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t matrix_idx = idx / (rows * cols);
    const int64_t elem_idx = idx % (rows * cols);
    const auto* src = reinterpret_cast<const scalar_t*>(static_cast<uintptr_t>(src_ptrs[matrix_idx]));
    dst[idx] = static_cast<float>(src[elem_idx]);
}

template <typename scalar_t>
__global__ void fused_prepare_muon_batch_kernel(
    const int64_t* grad_ptrs,
    float* momentum_batch,
    float* effective_batch,
    float momentum,
    bool nesterov,
    int64_t bucket_size,
    int64_t rows,
    int64_t cols) {
    const int64_t total = bucket_size * rows * cols;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t matrix_idx = idx / (rows * cols);
    const int64_t elem_idx = idx % (rows * cols);
    const auto* grad = reinterpret_cast<const scalar_t*>(static_cast<uintptr_t>(grad_ptrs[matrix_idx]));
    const float grad_value = static_cast<float>(grad[elem_idx]);
    const float updated_momentum = momentum_batch[idx] * momentum + grad_value;
    momentum_batch[idx] = updated_momentum;
    effective_batch[idx] = nesterov ? (grad_value + updated_momentum * momentum) : updated_momentum;
}

template <typename scalar_t>
__global__ void fused_prepare_muon_batch_capturable_kernel(
    const int64_t* grad_ptrs,
    float* momentum_batch,
    float* effective_batch,
    const float* momentum_ptr,
    bool nesterov,
    int64_t bucket_size,
    int64_t rows,
    int64_t cols) {
    const int64_t total = bucket_size * rows * cols;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float momentum = *momentum_ptr;
    const int64_t matrix_idx = idx / (rows * cols);
    const int64_t elem_idx = idx % (rows * cols);
    const auto* grad = reinterpret_cast<const scalar_t*>(static_cast<uintptr_t>(grad_ptrs[matrix_idx]));
    const float grad_value = static_cast<float>(grad[elem_idx]);
    const float updated_momentum = momentum_batch[idx] * momentum + grad_value;
    momentum_batch[idx] = updated_momentum;
    effective_batch[idx] = nesterov ? (grad_value + updated_momentum * momentum) : updated_momentum;
}

template <typename scalar_t>
__global__ void normalize_effective_batch_kernel(
    const float* effective_batch,
    const float* norms,
    scalar_t* ns_input_batch,
    bool transpose_input,
    int64_t bucket_size,
    int64_t rows,
    int64_t cols,
    int64_t ns_rows,
    int64_t ns_cols) {
    const int64_t total = bucket_size * ns_rows * ns_cols;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t matrix_idx = idx / (ns_rows * ns_cols);
    const int64_t local_idx = idx % (ns_rows * ns_cols);
    const int64_t out_row = local_idx / ns_cols;
    const int64_t out_col = local_idx % ns_cols;
    const int64_t src_row = transpose_input ? out_col : out_row;
    const int64_t src_col = transpose_input ? out_row : out_col;
    const int64_t src_idx = matrix_idx * rows * cols + src_row * cols + src_col;
    ns_input_batch[idx] = static_cast<scalar_t>(effective_batch[src_idx] / norms[matrix_idx]);
}

template <typename param_t, typename projected_t>
__global__ void apply_projected_updates_kernel(
    const int64_t* param_ptrs,
    const projected_t* projected,
    float lr,
    float decay_factor,
    float aspect_scale,
    bool transpose_input,
    int64_t bucket_size,
    int64_t rows,
    int64_t cols) {
    const int64_t total = bucket_size * rows * cols;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t matrix_idx = idx / (rows * cols);
    const int64_t elem_idx = idx % (rows * cols);
    const int64_t row = elem_idx / cols;
    const int64_t col = elem_idx % cols;
    const int64_t projected_cols = transpose_input ? rows : cols;
    const int64_t projected_row = transpose_input ? col : row;
    const int64_t projected_col = transpose_input ? row : col;
    const int64_t projected_idx =
        matrix_idx * rows * cols + projected_row * projected_cols + projected_col;
    auto* param = reinterpret_cast<param_t*>(static_cast<uintptr_t>(param_ptrs[matrix_idx]));
    const float value =
        static_cast<float>(param[elem_idx]) * decay_factor
        - lr * aspect_scale * static_cast<float>(projected[projected_idx]);
    param[elem_idx] = static_cast<param_t>(value);
}

template <typename param_t, typename projected_t>
__global__ void apply_projected_updates_capturable_kernel(
    const int64_t* param_ptrs,
    const projected_t* projected,
    const float* lr_ptr,
    const float* weight_decay_ptr,
    float aspect_scale,
    bool transpose_input,
    int64_t bucket_size,
    int64_t rows,
    int64_t cols) {
    const int64_t total = bucket_size * rows * cols;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float lr = *lr_ptr;
    const float weight_decay = *weight_decay_ptr;
    const float decay_factor = weight_decay > 0.0f ? (1.0f - lr * weight_decay) : 1.0f;
    const int64_t matrix_idx = idx / (rows * cols);
    const int64_t elem_idx = idx % (rows * cols);
    const int64_t row = elem_idx / cols;
    const int64_t col = elem_idx % cols;
    const int64_t projected_cols = transpose_input ? rows : cols;
    const int64_t projected_row = transpose_input ? col : row;
    const int64_t projected_col = transpose_input ? row : col;
    const int64_t projected_idx =
        matrix_idx * rows * cols + projected_row * projected_cols + projected_col;
    auto* param = reinterpret_cast<param_t*>(static_cast<uintptr_t>(param_ptrs[matrix_idx]));
    const float value =
        static_cast<float>(param[elem_idx]) * decay_factor
        - lr * aspect_scale * static_cast<float>(projected[projected_idx]);
    param[elem_idx] = static_cast<param_t>(value);
}

torch::Tensor make_device_pointer_tensor(const std::vector<torch::Tensor>& tensors) {
    auto ptrs_cpu = torch::empty(
        {static_cast<int64_t>(tensors.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto* ptrs_data = ptrs_cpu.data_ptr<int64_t>();
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        ptrs_data[idx] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensors[idx].data_ptr()));
    }
    return ptrs_cpu.to(tensors.front().device(), /*non_blocking=*/false, /*copy=*/true);
}

void gather_same_shape_tensor_batch(
    const std::vector<torch::Tensor>& tensors,
    torch::Tensor dst) {
    const auto bucket_size = static_cast<int64_t>(tensors.size());
    TORCH_CHECK(bucket_size > 0, "gather_same_shape_tensor_batch requires non-empty tensors");
    const int64_t rows = tensors.front().size(0);
    const int64_t cols = tensors.front().size(1);
    auto ptrs = make_device_pointer_tensor(tensors);
    const int threads = 256;
    const int64_t total = bucket_size * rows * cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        tensors.front().scalar_type(),
        "gather_same_shape_tensor_batch_kernel",
        [&] {
            gather_same_shape_tensor_batch_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                ptrs.data_ptr<int64_t>(),
                dst.data_ptr<float>(),
                bucket_size,
                rows,
                cols);
        });
    AT_CUDA_CHECK(cudaGetLastError());
}

void fused_prepare_muon_batch(
    const std::vector<torch::Tensor>& grads,
    torch::Tensor momentum_batch,
    torch::Tensor effective_batch,
    double momentum,
    bool nesterov,
    int64_t family_code) {
    const auto bucket_size = static_cast<int64_t>(grads.size());
    TORCH_CHECK(bucket_size > 0, "fused_prepare_muon_batch requires non-empty grads");
    TORCH_CHECK(momentum_batch.scalar_type() == torch::kFloat, "fused_prepare_muon_batch expects float momentum_batch");
    TORCH_CHECK(effective_batch.scalar_type() == torch::kFloat, "fused_prepare_muon_batch expects float effective_batch");
    const int64_t rows = grads.front().size(0);
    const int64_t cols = grads.front().size(1);
    auto ptrs = make_device_pointer_tensor(grads);
    const int threads = muon_family_threads(family_code);
    const int64_t total = bucket_size * rows * cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const float momentum_f = static_cast<float>(momentum);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        grads.front().scalar_type(),
        "fused_prepare_muon_batch_kernel",
        [&] {
            fused_prepare_muon_batch_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                ptrs.data_ptr<int64_t>(),
                momentum_batch.data_ptr<float>(),
                effective_batch.data_ptr<float>(),
                momentum_f,
                nesterov,
                bucket_size,
                rows,
                cols);
        });
    AT_CUDA_CHECK(cudaGetLastError());
}

void fused_prepare_muon_batch_capturable(
    const std::vector<torch::Tensor>& grads,
    torch::Tensor momentum_batch,
    torch::Tensor effective_batch,
    const torch::Tensor& momentum,
    bool nesterov,
    int64_t family_code) {
    const auto bucket_size = static_cast<int64_t>(grads.size());
    TORCH_CHECK(bucket_size > 0, "fused_prepare_muon_batch_capturable requires non-empty grads");
    TORCH_CHECK(momentum_batch.scalar_type() == torch::kFloat, "fused_prepare_muon_batch_capturable expects float momentum_batch");
    TORCH_CHECK(effective_batch.scalar_type() == torch::kFloat, "fused_prepare_muon_batch_capturable expects float effective_batch");
    TORCH_CHECK(momentum.is_cuda(), "fused_prepare_muon_batch_capturable expects CUDA momentum tensor");
    TORCH_CHECK(momentum.scalar_type() == torch::kFloat, "fused_prepare_muon_batch_capturable expects float32 momentum tensor");
    TORCH_CHECK(momentum.numel() == 1, "fused_prepare_muon_batch_capturable expects scalar momentum tensor");
    const int64_t rows = grads.front().size(0);
    const int64_t cols = grads.front().size(1);
    auto ptrs = make_device_pointer_tensor(grads);
    const int threads = muon_family_threads(family_code);
    const int64_t total = bucket_size * rows * cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        grads.front().scalar_type(),
        "fused_prepare_muon_batch_capturable_kernel",
        [&] {
            fused_prepare_muon_batch_capturable_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                ptrs.data_ptr<int64_t>(),
                momentum_batch.data_ptr<float>(),
                effective_batch.data_ptr<float>(),
                momentum.data_ptr<float>(),
                nesterov,
                bucket_size,
                rows,
                cols);
        });
    AT_CUDA_CHECK(cudaGetLastError());
}

void normalize_effective_batch(
    const torch::Tensor& effective_batch,
    torch::Tensor norms,
    torch::Tensor ns_input_batch,
    double eps,
    int64_t family_code) {
    const int64_t bucket_size = effective_batch.size(0);
    const int64_t rows = effective_batch.size(1);
    const int64_t cols = effective_batch.size(2);
    const bool transpose_input = muon_family_transposes(family_code);
    const int64_t ns_rows = transpose_input ? cols : rows;
    const int64_t ns_cols = transpose_input ? rows : cols;
    auto flat_input = effective_batch.flatten(1);
    at::norm_out(norms, flat_input, 2, {1}, true);
    norms.clamp_min_(eps);
    const int threads = muon_family_threads(family_code);
    const int64_t total = bucket_size * ns_rows * ns_cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        ns_input_batch.scalar_type(),
        "normalize_effective_batch_kernel",
        [&] {
            normalize_effective_batch_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                effective_batch.data_ptr<float>(),
                norms.data_ptr<float>(),
                ns_input_batch.data_ptr<scalar_t>(),
                transpose_input,
                bucket_size,
                rows,
                cols,
                ns_rows,
                ns_cols);
        });
    AT_CUDA_CHECK(cudaGetLastError());
}

void apply_projected_updates(
    const std::vector<torch::Tensor>& params,
    const torch::Tensor& projected_batch,
    double lr,
    double weight_decay,
    bool transpose_input,
    double aspect_scale,
    int64_t family_code) {
    const auto bucket_size = static_cast<int64_t>(params.size());
    TORCH_CHECK(bucket_size > 0, "apply_projected_updates requires non-empty params");
    const int64_t rows = params.front().size(0);
    const int64_t cols = params.front().size(1);
    auto ptrs = make_device_pointer_tensor(params);
    const int threads = muon_family_threads(family_code);
    const int64_t total = bucket_size * rows * cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const float lr_f = static_cast<float>(lr);
    const float decay_factor = static_cast<float>(weight_decay > 0.0 ? (1.0 - lr * weight_decay) : 1.0);
    const float aspect_scale_f = static_cast<float>(aspect_scale);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        params.front().scalar_type(),
        "apply_projected_updates_param_dispatch",
        [&] {
            using param_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                projected_batch.scalar_type(),
                "apply_projected_updates_projected_dispatch",
                [&] {
                    using projected_t = scalar_t;
                    apply_projected_updates_kernel<param_t, projected_t><<<blocks, threads, 0, stream>>>(
                        ptrs.data_ptr<int64_t>(),
                        projected_batch.data_ptr<projected_t>(),
                        lr_f,
                        decay_factor,
                        aspect_scale_f,
                        transpose_input,
                        bucket_size,
                        rows,
                        cols);
                });
        });
    AT_CUDA_CHECK(cudaGetLastError());
}

void apply_projected_updates_capturable(
    const std::vector<torch::Tensor>& params,
    const torch::Tensor& projected_batch,
    const torch::Tensor& lr,
    const torch::Tensor& weight_decay,
    bool transpose_input,
    double aspect_scale,
    int64_t family_code) {
    const auto bucket_size = static_cast<int64_t>(params.size());
    TORCH_CHECK(bucket_size > 0, "apply_projected_updates_capturable requires non-empty params");
    TORCH_CHECK(lr.is_cuda(), "apply_projected_updates_capturable expects CUDA lr tensor");
    TORCH_CHECK(weight_decay.is_cuda(), "apply_projected_updates_capturable expects CUDA weight_decay tensor");
    TORCH_CHECK(lr.scalar_type() == torch::kFloat, "apply_projected_updates_capturable expects float32 lr tensor");
    TORCH_CHECK(weight_decay.scalar_type() == torch::kFloat, "apply_projected_updates_capturable expects float32 weight_decay tensor");
    TORCH_CHECK(lr.numel() == 1, "apply_projected_updates_capturable expects scalar lr tensor");
    TORCH_CHECK(weight_decay.numel() == 1, "apply_projected_updates_capturable expects scalar weight_decay tensor");
    const int64_t rows = params.front().size(0);
    const int64_t cols = params.front().size(1);
    auto ptrs = make_device_pointer_tensor(params);
    const int threads = muon_family_threads(family_code);
    const int64_t total = bucket_size * rows * cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const float aspect_scale_f = static_cast<float>(aspect_scale);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        params.front().scalar_type(),
        "apply_projected_updates_capturable_param_dispatch",
        [&] {
            using param_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                projected_batch.scalar_type(),
                "apply_projected_updates_capturable_projected_dispatch",
                [&] {
                    using projected_t = scalar_t;
                    apply_projected_updates_capturable_kernel<param_t, projected_t><<<blocks, threads, 0, stream>>>(
                        ptrs.data_ptr<int64_t>(),
                        projected_batch.data_ptr<projected_t>(),
                        lr.data_ptr<float>(),
                        weight_decay.data_ptr<float>(),
                        aspect_scale_f,
                        transpose_input,
                        bucket_size,
                        rows,
                        cols);
                });
        });
    AT_CUDA_CHECK(cudaGetLastError());
}

void batched_newton_schulz_workspace(
    torch::Tensor x,
    torch::Tensor gram,
    torch::Tensor gram_sq,
    torch::Tensor next_x,
    int64_t steps) {
    constexpr double a = 3.4445;
    constexpr double b = -4.7750;
    constexpr double c = 2.0315;
    const int64_t batch = x.size(0);
    const int64_t rows = x.size(1);
    const int64_t cols = x.size(2);
    const int64_t x_stride = rows * cols;
    const int64_t gram_stride = rows * rows;
    auto* gram_ptr = gram.data_ptr<at::BFloat16>();
    auto* gram_sq_ptr = gram_sq.data_ptr<at::BFloat16>();
    const float one = 1.0f;
    const float zero = 0.0f;
    const float a_f = static_cast<float>(a);
    const float b_f = static_cast<float>(b);
    const float c_f = static_cast<float>(c);
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    const MuonSquareBackend square_backend = select_muon_square_backend(x);
    const bool use_cublaslt_square_gram =
        square_backend == MuonSquareBackend::kCublasLt;
    const bool use_cublaslt_square_update =
        square_backend == MuonSquareBackend::kCublasLt || square_backend == MuonSquareBackend::kHybrid;
    torch::Tensor current = x;
    torch::Tensor scratch = next_x;

    for (int64_t step = 0; step < steps; ++step) {
        auto* current_ptr = current.data_ptr<at::BFloat16>();
        auto* scratch_ptr = scratch.data_ptr<at::BFloat16>();
        if (use_cublaslt_square_gram) {
            muon_cublaslt_square_matmul(current, current, gram, /*transpose_a=*/true, one, zero);
        } else {
            check_cublas_status(
                cublasGemmStridedBatchedEx(
                    handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    static_cast<int>(rows),
                    static_cast<int>(rows),
                    static_cast<int>(cols),
                    &one,
                    current_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(cols),
                    static_cast<long long>(x_stride),
                    current_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(cols),
                    static_cast<long long>(x_stride),
                    &zero,
                    gram_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(rows),
                    static_cast<long long>(gram_stride),
                    static_cast<int>(batch),
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmStridedBatchedEx(xxt)");
        }
        if (use_cublaslt_square_gram) {
            muon_cublaslt_square_matmul(gram, gram, gram_sq, /*transpose_a=*/false, one, zero);
        } else {
            check_cublas_status(
                cublasGemmStridedBatchedEx(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    static_cast<int>(rows),
                    static_cast<int>(rows),
                    static_cast<int>(rows),
                    &one,
                    gram_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(rows),
                    static_cast<long long>(gram_stride),
                    gram_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(rows),
                    static_cast<long long>(gram_stride),
                    &zero,
                    gram_sq_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(rows),
                    static_cast<long long>(gram_stride),
                    static_cast<int>(batch),
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmStridedBatchedEx(gram_sq)");
        }
        scratch.copy_(current);
        if (use_cublaslt_square_update) {
            muon_cublaslt_square_matmul(current, gram, scratch, /*transpose_a=*/false, b_f, a_f);
            muon_cublaslt_square_matmul(current, gram_sq, scratch, /*transpose_a=*/false, c_f, one);
        } else {
            check_cublas_status(
                cublasGemmStridedBatchedEx(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    static_cast<int>(cols),
                    static_cast<int>(rows),
                    static_cast<int>(rows),
                    &b_f,
                    current_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(cols),
                    static_cast<long long>(x_stride),
                    gram_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(rows),
                    static_cast<long long>(gram_stride),
                    &a_f,
                    scratch_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(cols),
                    static_cast<long long>(x_stride),
                    static_cast<int>(batch),
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmStridedBatchedEx(update1)");
            check_cublas_status(
                cublasGemmStridedBatchedEx(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    static_cast<int>(cols),
                    static_cast<int>(rows),
                    static_cast<int>(rows),
                    &c_f,
                    current_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(cols),
                    static_cast<long long>(x_stride),
                    gram_sq_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(rows),
                    static_cast<long long>(gram_stride),
                    &one,
                    scratch_ptr,
                    CUDA_R_16BF,
                    static_cast<int>(cols),
                    static_cast<long long>(x_stride),
                    static_cast<int>(batch),
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmStridedBatchedEx(update2)");
        }
        std::swap(current, scratch);
    }
    if (!current.is_same(x)) {
        x.copy_(current);
    }
}

torch::Tensor batched_newton_schulz(
    const torch::Tensor& input,
    int64_t steps,
    double eps) {
    const bool transposed = input.size(1) > input.size(2);
    auto x = transposed ? input.transpose(1, 2).contiguous().to(torch::kBFloat16) : input.to(torch::kBFloat16);
    auto norms = input.to(torch::kFloat).flatten(1).norm(2, 1, true).clamp_min(eps).view({input.size(0), 1, 1});
    x.div_(norms.to(torch::kBFloat16));
    auto gram = torch::empty({x.size(0), x.size(1), x.size(1)}, x.options());
    auto gram_sq = torch::empty_like(gram);
    auto next_x = torch::empty_like(x);
    batched_newton_schulz_workspace(x, gram, gram_sq, next_x, steps);
    if (transposed) {
        x = x.transpose(1, 2).contiguous();
    }
    return x;
}

}  // namespace

std::vector<torch::Tensor> muon_grouped_step_cuda(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> momentum_bufs,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps) {
    const auto bucket_size = static_cast<int64_t>(params.size());
    TORCH_CHECK(bucket_size > 0, "muon_grouped_step_cuda requires a non-empty bucket");

    std::vector<torch::Tensor> grad_views;
    std::vector<torch::Tensor> mom_views;
    grad_views.reserve(grads.size());
    mom_views.reserve(momentum_bufs.size());
    for (const auto& grad : grads) {
        grad_views.push_back(grad.to(torch::kFloat));
    }
    for (const auto& mom : momentum_bufs) {
        mom_views.push_back(mom.to(torch::kFloat));
    }

    auto grad_batch = torch::stack(grad_views, 0);
    auto momentum_batch = torch::stack(mom_views, 0);
    momentum_batch.mul_(momentum).add_(grad_batch);

    auto effective_batch = nesterov ? grad_batch + momentum_batch * momentum : momentum_batch;
    auto projected_batch = batched_newton_schulz(effective_batch, ns_steps, eps).to(torch::kFloat);

    const double aspect_scale = std::sqrt(
        std::max(1.0, static_cast<double>(params.front().size(0)) / std::max(1.0, static_cast<double>(params.front().size(1)))));
    projected_batch.mul_(aspect_scale);

    for (int64_t idx = 0; idx < bucket_size; ++idx) {
        auto& param = params[idx];
        auto& momentum_buf = momentum_bufs[idx];
        if (weight_decay > 0.0) {
            param.mul_(1.0 - lr * weight_decay);
        }
        momentum_buf.copy_(momentum_batch[idx].to(momentum_buf.scalar_type()));
        param.add_(projected_batch[idx].to(param.scalar_type()), -lr);
    }

    return {momentum_batch, projected_batch};
}

void muon_grouped_step_batched_cuda(
    std::vector<torch::Tensor> params,
    torch::Tensor grad_batch,
    torch::Tensor momentum_batch,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps) {
    const auto bucket_size = static_cast<int64_t>(params.size());
    TORCH_CHECK(bucket_size > 0, "muon_grouped_step_batched_cuda requires a non-empty bucket");

    auto grad_batch_f32 = grad_batch.to(torch::kFloat);
    auto momentum_batch_f32 = momentum_batch.to(torch::kFloat);
    momentum_batch_f32.mul_(momentum).add_(grad_batch_f32);

    auto effective_batch = nesterov ? grad_batch_f32 + momentum_batch_f32 * momentum : momentum_batch_f32;
    auto projected_batch = batched_newton_schulz(effective_batch, ns_steps, eps);
    const double aspect_scale = std::sqrt(
        std::max(1.0, static_cast<double>(params.front().size(0)) / std::max(1.0, static_cast<double>(params.front().size(1)))));
    if (!momentum_batch.is_same(momentum_batch_f32)) {
        momentum_batch.copy_(momentum_batch_f32.to(momentum_batch.scalar_type()));
    }
    apply_projected_updates(
        params,
        projected_batch,
        lr,
        weight_decay,
        /*transpose_input=*/false,
        aspect_scale,
        kMuonFamilySquare);
}

void muon_grouped_step_workspace_cuda(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    torch::Tensor grad_batch,
    torch::Tensor momentum_batch,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps) {
    const auto bucket_size = static_cast<int64_t>(params.size());
    TORCH_CHECK(bucket_size > 0, "muon_grouped_step_workspace_cuda requires a non-empty bucket");
    auto grad_batch_f32 = grad_batch.scalar_type() == torch::kFloat ? grad_batch : grad_batch.to(torch::kFloat);
    auto momentum_batch_f32 = momentum_batch.scalar_type() == torch::kFloat ? momentum_batch : momentum_batch.to(torch::kFloat);
    fused_prepare_muon_batch(grads, momentum_batch_f32, grad_batch_f32, momentum, nesterov, kMuonFamilySquare);
    auto projected_batch = batched_newton_schulz(grad_batch_f32, ns_steps, eps);
    const double aspect_scale = std::sqrt(
        std::max(1.0, static_cast<double>(params.front().size(0)) / std::max(1.0, static_cast<double>(params.front().size(1)))));
    if (!momentum_batch.is_same(momentum_batch_f32)) {
        momentum_batch.copy_(momentum_batch_f32.to(momentum_batch.scalar_type()));
    }
    apply_projected_updates(
        params,
        projected_batch,
        lr,
        weight_decay,
        /*transpose_input=*/false,
        aspect_scale,
        kMuonFamilySquare);
}

void muon_grouped_step_family_workspace_cuda(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    torch::Tensor effective_batch,
    torch::Tensor momentum_batch,
    torch::Tensor norms,
    torch::Tensor ns_input_batch,
    torch::Tensor gram_batch,
    torch::Tensor gram_sq_batch,
    torch::Tensor next_x_batch,
    int64_t family_code,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps) {
    const auto bucket_size = static_cast<int64_t>(params.size());
    TORCH_CHECK(bucket_size > 0, "muon_grouped_step_family_workspace_cuda requires a non-empty bucket");
    fused_prepare_muon_batch(grads, momentum_batch, effective_batch, momentum, nesterov, family_code);
    normalize_effective_batch(effective_batch, norms, ns_input_batch, eps, family_code);
    batched_newton_schulz_workspace(ns_input_batch, gram_batch, gram_sq_batch, next_x_batch, ns_steps);
    const double aspect_scale = std::sqrt(
        std::max(1.0, static_cast<double>(params.front().size(0)) / std::max(1.0, static_cast<double>(params.front().size(1)))));
    apply_projected_updates(
        params,
        ns_input_batch,
        lr,
        weight_decay,
        muon_family_transposes(family_code),
        aspect_scale,
        family_code);
}

void muon_grouped_step_family_workspace_capturable_cuda(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    torch::Tensor effective_batch,
    torch::Tensor momentum_batch,
    torch::Tensor norms,
    torch::Tensor ns_input_batch,
    torch::Tensor gram_batch,
    torch::Tensor gram_sq_batch,
    torch::Tensor next_x_batch,
    int64_t family_code,
    torch::Tensor lr,
    torch::Tensor momentum,
    torch::Tensor weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps) {
    const auto bucket_size = static_cast<int64_t>(params.size());
    TORCH_CHECK(bucket_size > 0, "muon_grouped_step_family_workspace_capturable_cuda requires a non-empty bucket");
    fused_prepare_muon_batch_capturable(grads, momentum_batch, effective_batch, momentum, nesterov, family_code);
    normalize_effective_batch(effective_batch, norms, ns_input_batch, eps, family_code);
    batched_newton_schulz_workspace(ns_input_batch, gram_batch, gram_sq_batch, next_x_batch, ns_steps);
    const double aspect_scale = std::sqrt(
        std::max(1.0, static_cast<double>(params.front().size(0)) / std::max(1.0, static_cast<double>(params.front().size(1)))));
    apply_projected_updates_capturable(
        params,
        ns_input_batch,
        lr,
        weight_decay,
        muon_family_transposes(family_code),
        aspect_scale,
        family_code);
}
