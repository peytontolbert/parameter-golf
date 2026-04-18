#include <torch/extension.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/linear.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cublasLt.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int kBlockThreads = 256;

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float<c10::Half>(c10::Half value) {
    return __half2float(*reinterpret_cast<__half*>(&value));
}

template <>
__device__ inline float scalar_to_float<c10::BFloat16>(c10::BFloat16 value) {
    return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&value));
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float value) {
    return static_cast<scalar_t>(value);
}

template <>
__device__ inline c10::Half float_to_scalar<c10::Half>(float value) {
    c10::Half out;
    *reinterpret_cast<__half*>(&out) = __float2half_rn(value);
    return out;
}

template <>
__device__ inline c10::BFloat16 float_to_scalar<c10::BFloat16>(float value) {
    c10::BFloat16 out;
    *reinterpret_cast<__nv_bfloat16*>(&out) = __float2bfloat16(value);
    return out;
}

__device__ inline float sanitize_value(float value, float clamp_abs) {
    if (!isfinite(value)) {
        value = 0.0f;
    }
    if (clamp_abs > 0.0f) {
        value = fminf(fmaxf(value, -clamp_abs), clamp_abs);
    }
    return value;
}

template <typename scalar_t>
__global__ void branch_prep_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ x0,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ mixed,
    scalar_t* __restrict__ normed,
    int64_t outer,
    int64_t dim,
    float eps,
    float clamp_abs,
    float norm_scale) {
    __shared__ float shmem[kBlockThreads];
    const int64_t row = blockIdx.x;
    if (row >= outer) {
        return;
    }
    const int tid = threadIdx.x;
    const int64_t base = row * dim;
    float sum_sq = 0.0f;
    for (int64_t col = tid; col < dim; col += blockDim.x) {
        const float mix0 = scalar_to_float(mix[col]);
        const float mix1 = scalar_to_float(mix[dim + col]);
        float value = mix0 * scalar_to_float(x[base + col]) + mix1 * scalar_to_float(x0[base + col]);
        value = sanitize_value(value, clamp_abs);
        mixed[base + col] = float_to_scalar<scalar_t>(value);
        sum_sq += value * value;
    }
    shmem[tid] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(shmem[0] / fmaxf(static_cast<float>(dim), 1.0f) + eps) * norm_scale;
    for (int64_t col = tid; col < dim; col += blockDim.x) {
        const float value = scalar_to_float(mixed[base + col]) * inv_rms;
        normed[base + col] = float_to_scalar<scalar_t>(value);
    }
}

template <typename scalar_t>
__global__ void branch_prep_backward_kernel(
    const scalar_t* __restrict__ grad_mixed,
    const scalar_t* __restrict__ grad_normed,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ x0,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_x0,
    float* __restrict__ grad_mix,
    int64_t outer,
    int64_t dim,
    float eps,
    float clamp_abs,
    float norm_scale) {
    __shared__ float shmem_sum_sq[kBlockThreads];
    __shared__ float shmem_dot[kBlockThreads];
    const int64_t row = blockIdx.x;
    if (row >= outer) {
        return;
    }
    const int tid = threadIdx.x;
    const int64_t base = row * dim;
    float sum_sq = 0.0f;
    float dot = 0.0f;
    for (int64_t col = tid; col < dim; col += blockDim.x) {
        const float mix0 = scalar_to_float(mix[col]);
        const float mix1 = scalar_to_float(mix[dim + col]);
        float raw = mix0 * scalar_to_float(x[base + col]) + mix1 * scalar_to_float(x0[base + col]);
        const float mixed = sanitize_value(raw, clamp_abs);
        sum_sq += mixed * mixed;
        dot += scalar_to_float(grad_normed[base + col]) * mixed;
    }
    shmem_sum_sq[tid] = sum_sq;
    shmem_dot[tid] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem_sum_sq[tid] += shmem_sum_sq[tid + stride];
            shmem_dot[tid] += shmem_dot[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(shmem_sum_sq[0] / fmaxf(static_cast<float>(dim), 1.0f) + eps);
    const float coeff = norm_scale * inv_rms;
    const float corr = norm_scale * inv_rms * inv_rms * inv_rms * shmem_dot[0] / fmaxf(static_cast<float>(dim), 1.0f);
    for (int64_t col = tid; col < dim; col += blockDim.x) {
        const float mix0 = scalar_to_float(mix[col]);
        const float mix1 = scalar_to_float(mix[dim + col]);
        const float x_v = scalar_to_float(x[base + col]);
        const float x0_v = scalar_to_float(x0[base + col]);
        const float raw = mix0 * x_v + mix1 * x0_v;
        const bool finite = isfinite(raw);
        const bool inside_clamp = (clamp_abs <= 0.0f) || (raw >= -clamp_abs && raw <= clamp_abs);
        const float active = (finite && inside_clamp) ? 1.0f : 0.0f;
        const float mixed = sanitize_value(raw, clamp_abs);
        const float grad_y =
            scalar_to_float(grad_mixed[base + col])
            + coeff * scalar_to_float(grad_normed[base + col])
            - corr * mixed;
        const float grad_raw = active * grad_y;
        grad_x[base + col] = float_to_scalar<scalar_t>(grad_raw * mix0);
        grad_x0[base + col] = float_to_scalar<scalar_t>(grad_raw * mix1);
        atomicAdd(&grad_mix[col], grad_raw * x_v);
        atomicAdd(&grad_mix[dim + col], grad_raw * x0_v);
    }
}

template <typename scalar_t>
__global__ void branch_update_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ scale,
    scalar_t* __restrict__ out,
    int64_t total,
    int64_t dim,
    float residual_alpha,
    float clamp_abs) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t col = idx % dim;
    float value = scalar_to_float(x[idx]) * residual_alpha + scalar_to_float(scale[col]) * scalar_to_float(update[idx]);
    value = sanitize_value(value, clamp_abs);
    out[idx] = float_to_scalar<scalar_t>(value);
}

template <typename scalar_t>
__global__ void branch_update_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ scale,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_update,
    float* __restrict__ grad_scale,
    int64_t total,
    int64_t dim,
    float residual_alpha,
    float clamp_abs) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t col = idx % dim;
    const float scale_v = scalar_to_float(scale[col]);
    const float pre = scalar_to_float(x[idx]) * residual_alpha + scale_v * scalar_to_float(update[idx]);
    const bool finite = isfinite(pre);
    const bool inside_clamp = (clamp_abs <= 0.0f) || (pre >= -clamp_abs && pre <= clamp_abs);
    const float active = (finite && inside_clamp) ? 1.0f : 0.0f;
    const float go = scalar_to_float(grad_output[idx]) * active;
    grad_x[idx] = float_to_scalar<scalar_t>(go * residual_alpha);
    grad_update[idx] = float_to_scalar<scalar_t>(go * scale_v);
    atomicAdd(&grad_scale[col], go * scalar_to_float(update[idx]));
}

template <typename scalar_t, bool kInverseScale>
__global__ void partial_rotary_pair_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const scalar_t* __restrict__ scale,
    int64_t outer,
    int64_t seq_len,
    int64_t head_dim,
    int64_t rope_half) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = outer * rope_half;
    if (idx >= total) {
        return;
    }
    const int64_t outer_idx = idx / rope_half;
    const int64_t pair_idx = idx % rope_half;
    const int64_t t = outer_idx % seq_len;
    const int64_t base = outer_idx * head_dim;
    const float x1 = scalar_to_float(x[base + pair_idx]);
    const float x2 = scalar_to_float(x[base + rope_half + pair_idx]);
    const int64_t table_idx = t * rope_half + pair_idx;
    const float cos_v = scalar_to_float(cos[table_idx]);
    const float sin_v = scalar_to_float(sin[table_idx]);
    float out1 = x1 * cos_v + x2 * sin_v;
    float out2 = x1 * (-sin_v) + x2 * cos_v;
    if (scale != nullptr) {
        const float scale_v = fmaxf(scalar_to_float(scale[table_idx]), 1.0e-6f);
        if constexpr (kInverseScale) {
            out1 /= scale_v;
            out2 /= scale_v;
        } else {
            out1 *= scale_v;
            out2 *= scale_v;
        }
    }
    out[base + pair_idx] = float_to_scalar<scalar_t>(out1);
    out[base + rope_half + pair_idx] = float_to_scalar<scalar_t>(out2);
}

template <typename scalar_t, bool kInverseScale>
__global__ void partial_rotary_pair_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const scalar_t* __restrict__ scale,
    int64_t outer,
    int64_t seq_len,
    int64_t head_dim,
    int64_t rope_half) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = outer * rope_half;
    if (idx >= total) {
        return;
    }
    const int64_t outer_idx = idx / rope_half;
    const int64_t pair_idx = idx % rope_half;
    const int64_t t = outer_idx % seq_len;
    const int64_t base = outer_idx * head_dim;
    const float go1 = scalar_to_float(grad_output[base + pair_idx]);
    const float go2 = scalar_to_float(grad_output[base + rope_half + pair_idx]);
    const int64_t table_idx = t * rope_half + pair_idx;
    const float cos_v = scalar_to_float(cos[table_idx]);
    const float sin_v = scalar_to_float(sin[table_idx]);
    float grad1 = go1 * cos_v - go2 * sin_v;
    float grad2 = go1 * sin_v + go2 * cos_v;
    if (scale != nullptr) {
        const float scale_v = fmaxf(scalar_to_float(scale[table_idx]), 1.0e-6f);
        if constexpr (kInverseScale) {
            grad1 /= scale_v;
            grad2 /= scale_v;
        } else {
            grad1 *= scale_v;
            grad2 *= scale_v;
        }
    }
    grad_input[base + pair_idx] = float_to_scalar<scalar_t>(grad1);
    grad_input[base + rope_half + pair_idx] = float_to_scalar<scalar_t>(grad2);
}

template <typename scalar_t>
__global__ void relu_square_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float value = scalar_to_float(x[idx]);
    const float relu_v = value > 0.0f ? value : 0.0f;
    out[idx] = float_to_scalar<scalar_t>(relu_v * relu_v);
}

template <typename scalar_t>
__global__ void relu_square_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ grad_input,
    int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float x_v = scalar_to_float(x[idx]);
    const float go = scalar_to_float(grad_output[idx]);
    const float relu_v = x_v > 0.0f ? x_v : 0.0f;
    grad_input[idx] = float_to_scalar<scalar_t>(go * 2.0f * relu_v);
}

const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline void check_cublaslt(cublasStatus_t status, const char* op_name) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, op_name, " failed: ", cublas_status_string(status));
}

bool is_supported_linear_dtype(torch::ScalarType dtype) {
    return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

cudaDataType_t to_cuda_data_type(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32:
            return CUDA_R_32F;
        case torch::kFloat16:
            return CUDA_R_16F;
        case torch::kBFloat16:
            return CUDA_R_16BF;
        default:
            TORCH_CHECK(false, "Unsupported dtype for cuBLASLt linear");
    }
}

torch::Tensor run_cublaslt_linear(
    const torch::Tensor& x_2d,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
    if (!is_supported_linear_dtype(x_2d.scalar_type()) || x_2d.scalar_type() != weight.scalar_type()) {
        return at::linear(x_2d, weight, bias);
    }

    const auto m = x_2d.size(0);
    const auto k = x_2d.size(1);
    const auto n = weight.size(0);
    auto out = torch::empty({m, n}, x_2d.options());
    auto stream = c10::cuda::getCurrentCUDAStream(x_2d.get_device());
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    auto workspace = at::cuda::getCUDABlasLtWorkspace();
    auto workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    const auto scale_type = CUDA_R_32F;
    const auto data_type = to_cuda_data_type(x_2d.scalar_type());
    const auto compute_type = CUBLAS_COMPUTE_32F;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;

    auto cleanup = [&]() {
        if (preference != nullptr) {
            cublasLtMatmulPreferenceDestroy(preference);
        }
        if (c_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(c_desc);
        }
        if (b_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(b_desc);
        }
        if (a_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(a_desc);
        }
        if (op_desc != nullptr) {
            cublasLtMatmulDescDestroy(op_desc);
        }
    };

    check_cublaslt(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type), "cublasLtMatmulDescCreate");
    check_cublaslt(
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
        "cublasLtMatmulDescSetAttribute(TRANSA)");
    check_cublaslt(
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
        "cublasLtMatmulDescSetAttribute(TRANSB)");
    check_cublaslt(cublasLtMatrixLayoutCreate(&a_desc, data_type, k, n, k), "cublasLtMatrixLayoutCreate(A)");
    check_cublaslt(cublasLtMatrixLayoutCreate(&b_desc, data_type, k, m, k), "cublasLtMatrixLayoutCreate(B)");
    check_cublaslt(cublasLtMatrixLayoutCreate(&c_desc, data_type, n, m, n), "cublasLtMatrixLayoutCreate(C)");
    check_cublaslt(cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate");
    check_cublaslt(
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,
            sizeof(workspace_size)),
        "cublasLtMatmulPreferenceSetAttribute(workspace)");

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        op_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        preference,
        1,
        &heuristic,
        &returned_results);
    if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
        cleanup();
        return at::linear(x_2d, weight, bias);
    }

    const auto matmul_status = cublasLtMatmul(
        handle,
        op_desc,
        &alpha,
        weight.data_ptr(),
        a_desc,
        x_2d.data_ptr(),
        b_desc,
        &beta,
        out.data_ptr(),
        c_desc,
        out.data_ptr(),
        c_desc,
        &heuristic.algo,
        workspace,
        workspace_size,
        stream.stream());
    cleanup();
    if (matmul_status != CUBLAS_STATUS_SUCCESS) {
        return at::linear(x_2d, weight, bias);
    }
    if (bias.has_value() && bias.value().defined()) {
        out.add_(bias.value().view({1, n}));
    }
    return out;
}

template <typename scalar_t>
void launch_branch_prep(
    const torch::Tensor& x,
    const torch::Tensor& x0,
    const torch::Tensor& mix,
    torch::Tensor& mixed,
    torch::Tensor& normed,
    float eps,
    float clamp_abs,
    float norm_scale) {
    const int64_t dim = x.size(-1);
    const int64_t outer = x.numel() / dim;
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    branch_prep_kernel<scalar_t><<<static_cast<int>(outer), kBlockThreads, 0, stream.stream()>>>(
        x.data_ptr<scalar_t>(),
        x0.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        mixed.data_ptr<scalar_t>(),
        normed.data_ptr<scalar_t>(),
        outer,
        dim,
        eps,
        clamp_abs,
        norm_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_branch_prep_backward(
    const torch::Tensor& grad_mixed,
    const torch::Tensor& grad_normed,
    const torch::Tensor& x,
    const torch::Tensor& x0,
    const torch::Tensor& mix,
    torch::Tensor& grad_x,
    torch::Tensor& grad_x0,
    torch::Tensor& grad_mix,
    float eps,
    float clamp_abs,
    float norm_scale) {
    const int64_t dim = x.size(-1);
    const int64_t outer = x.numel() / dim;
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    branch_prep_backward_kernel<scalar_t><<<static_cast<int>(outer), kBlockThreads, 0, stream.stream()>>>(
        grad_mixed.data_ptr<scalar_t>(),
        grad_normed.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        x0.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        grad_x.data_ptr<scalar_t>(),
        grad_x0.data_ptr<scalar_t>(),
        grad_mix.data_ptr<float>(),
        outer,
        dim,
        eps,
        clamp_abs,
        norm_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_branch_update(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const torch::Tensor& scale,
    torch::Tensor& out,
    float residual_alpha,
    float clamp_abs) {
    const int64_t total = x.numel();
    const int64_t dim = x.size(-1);
    const int blocks = static_cast<int>((total + kBlockThreads - 1) / kBlockThreads);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    branch_update_kernel<scalar_t><<<blocks, kBlockThreads, 0, stream.stream()>>>(
        x.data_ptr<scalar_t>(),
        update.data_ptr<scalar_t>(),
        scale.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        total,
        dim,
        residual_alpha,
        clamp_abs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_branch_update_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& update,
    const torch::Tensor& scale,
    torch::Tensor& grad_x,
    torch::Tensor& grad_update,
    torch::Tensor& grad_scale,
    float residual_alpha,
    float clamp_abs) {
    const int64_t total = x.numel();
    const int64_t dim = x.size(-1);
    const int blocks = static_cast<int>((total + kBlockThreads - 1) / kBlockThreads);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    branch_update_backward_kernel<scalar_t><<<blocks, kBlockThreads, 0, stream.stream()>>>(
        grad_output.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        update.data_ptr<scalar_t>(),
        scale.data_ptr<scalar_t>(),
        grad_x.data_ptr<scalar_t>(),
        grad_update.data_ptr<scalar_t>(),
        grad_scale.data_ptr<float>(),
        total,
        dim,
        residual_alpha,
        clamp_abs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, bool kInverseScale>
void launch_partial_rotary_pair(
    const torch::Tensor& x,
    torch::Tensor& out,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const c10::optional<torch::Tensor>& scale,
    int64_t rope_dims) {
    const int64_t head_dim = x.size(-1);
    const int64_t rope_half = rope_dims / 2;
    const int64_t outer = x.numel() / head_dim;
    const int64_t total = outer * rope_half;
    const int blocks = static_cast<int>((total + kBlockThreads - 1) / kBlockThreads);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    const auto* scale_ptr = (scale.has_value() && scale.value().defined()) ? scale.value().data_ptr<scalar_t>() : nullptr;
    partial_rotary_pair_kernel<scalar_t, kInverseScale><<<blocks, kBlockThreads, 0, stream.stream()>>>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        scale_ptr,
        outer,
        x.size(2),
        head_dim,
        rope_half);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, bool kInverseScale>
void launch_partial_rotary_pair_backward(
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const c10::optional<torch::Tensor>& scale,
    int64_t rope_dims) {
    const int64_t head_dim = grad_output.size(-1);
    const int64_t rope_half = rope_dims / 2;
    const int64_t outer = grad_output.numel() / head_dim;
    const int64_t total = outer * rope_half;
    const int blocks = static_cast<int>((total + kBlockThreads - 1) / kBlockThreads);
    const auto stream = c10::cuda::getCurrentCUDAStream(grad_output.get_device());
    const auto* scale_ptr = (scale.has_value() && scale.value().defined()) ? scale.value().data_ptr<scalar_t>() : nullptr;
    partial_rotary_pair_backward_kernel<scalar_t, kInverseScale><<<blocks, kBlockThreads, 0, stream.stream()>>>(
        grad_output.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>(),
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        scale_ptr,
        outer,
        grad_output.size(2),
        head_dim,
        rope_half);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_relu_square(const torch::Tensor& x, torch::Tensor& out) {
    const int64_t total = x.numel();
    const int blocks = static_cast<int>((total + kBlockThreads - 1) / kBlockThreads);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    relu_square_kernel<scalar_t><<<blocks, kBlockThreads, 0, stream.stream()>>>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_relu_square_backward(const torch::Tensor& grad_output, const torch::Tensor& x, torch::Tensor& grad_input) {
    const int64_t total = x.numel();
    const int blocks = static_cast<int>((total + kBlockThreads - 1) / kBlockThreads);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    relu_square_backward_kernel<scalar_t><<<blocks, kBlockThreads, 0, stream.stream()>>>(
        grad_output.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>(),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::vector<torch::Tensor> causal_machine_branch_prep_cuda(
    torch::Tensor x,
    torch::Tensor x0,
    torch::Tensor mix_weights,
    double eps,
    double clamp_abs,
    double norm_scale) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto mixed = torch::empty_like(x);
    auto normed = torch::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "causal_machine_branch_prep_cuda",
        [&] {
            launch_branch_prep<scalar_t>(
                x,
                x0,
                mix_weights,
                mixed,
                normed,
                static_cast<float>(eps),
                static_cast<float>(clamp_abs),
                static_cast<float>(norm_scale));
        });
    return {mixed, normed};
}

std::vector<torch::Tensor> causal_machine_branch_prep_backward_cuda(
    torch::Tensor grad_mixed,
    torch::Tensor grad_normed,
    torch::Tensor x,
    torch::Tensor x0,
    torch::Tensor mix_weights,
    double eps,
    double clamp_abs,
    double norm_scale) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto grad_x = torch::empty_like(x);
    auto grad_x0 = torch::empty_like(x0);
    auto grad_mix = torch::zeros({2, x.size(-1)}, x.options().dtype(torch::kFloat32));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "causal_machine_branch_prep_backward_cuda",
        [&] {
            launch_branch_prep_backward<scalar_t>(
                grad_mixed,
                grad_normed,
                x,
                x0,
                mix_weights,
                grad_x,
                grad_x0,
                grad_mix,
                static_cast<float>(eps),
                static_cast<float>(clamp_abs),
                static_cast<float>(norm_scale));
        });
    return {grad_x, grad_x0, grad_mix};
}

torch::Tensor causal_machine_branch_update_cuda(
    torch::Tensor x,
    torch::Tensor update,
    torch::Tensor scale,
    double residual_alpha,
    double clamp_abs) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "causal_machine_branch_update_cuda",
        [&] {
            launch_branch_update<scalar_t>(
                x,
                update,
                scale,
                out,
                static_cast<float>(residual_alpha),
                static_cast<float>(clamp_abs));
        });
    return out;
}

std::vector<torch::Tensor> causal_machine_branch_update_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor update,
    torch::Tensor scale,
    double residual_alpha,
    double clamp_abs) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto grad_x = torch::empty_like(x);
    auto grad_update = torch::empty_like(update);
    auto grad_scale = torch::zeros({x.size(-1)}, x.options().dtype(torch::kFloat32));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "causal_machine_branch_update_backward_cuda",
        [&] {
            launch_branch_update_backward<scalar_t>(
                grad_output,
                x,
                update,
                scale,
                grad_x,
                grad_update,
                grad_scale,
                static_cast<float>(residual_alpha),
                static_cast<float>(clamp_abs));
        });
    return {grad_x, grad_update, grad_scale};
}

std::vector<torch::Tensor> causal_machine_partial_rotary_pair_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    c10::optional<torch::Tensor> scale,
    int64_t rope_dims,
    bool inverse_scale_k) {
    c10::cuda::CUDAGuard device_guard(q.device());
    auto q_out = q.clone();
    auto k_out = k.clone();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "causal_machine_partial_rotary_pair_cuda",
        [&] {
            launch_partial_rotary_pair<scalar_t, false>(q, q_out, cos, sin, scale, rope_dims);
            if (inverse_scale_k) {
                launch_partial_rotary_pair<scalar_t, true>(k, k_out, cos, sin, scale, rope_dims);
            } else {
                launch_partial_rotary_pair<scalar_t, false>(k, k_out, cos, sin, scale, rope_dims);
            }
        });
    return {q_out, k_out};
}

std::vector<torch::Tensor> causal_machine_partial_rotary_pair_backward_cuda(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor cos,
    torch::Tensor sin,
    c10::optional<torch::Tensor> scale,
    int64_t rope_dims,
    bool inverse_scale_k) {
    c10::cuda::CUDAGuard device_guard(grad_q.device());
    auto grad_q_input = grad_q.clone();
    auto grad_k_input = grad_k.clone();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        grad_q.scalar_type(),
        "causal_machine_partial_rotary_pair_backward_cuda",
        [&] {
            launch_partial_rotary_pair_backward<scalar_t, false>(
                grad_q,
                grad_q_input,
                cos,
                sin,
                scale,
                rope_dims);
            if (inverse_scale_k) {
                launch_partial_rotary_pair_backward<scalar_t, true>(
                    grad_k,
                    grad_k_input,
                    cos,
                    sin,
                    scale,
                    rope_dims);
            } else {
                launch_partial_rotary_pair_backward<scalar_t, false>(
                    grad_k,
                    grad_k_input,
                    cos,
                    sin,
                    scale,
                    rope_dims);
            }
        });
    return {grad_q_input, grad_k_input};
}

torch::Tensor causal_machine_relu_square_cuda(torch::Tensor x) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "causal_machine_relu_square_cuda",
        [&] {
            launch_relu_square<scalar_t>(x, out);
        });
    return out;
}

torch::Tensor causal_machine_relu_square_backward_cuda(torch::Tensor grad_output, torch::Tensor x) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto grad_input = torch::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "causal_machine_relu_square_backward_cuda",
        [&] {
            launch_relu_square_backward<scalar_t>(grad_output, x, grad_input);
        });
    return grad_input;
}

torch::Tensor causal_machine_cublaslt_linear_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_2d = x.reshape({-1, x.size(-1)}).contiguous();
    auto out_2d = run_cublaslt_linear(x_2d, weight, bias);
    auto out_sizes = x.sizes().vec();
    out_sizes.back() = weight.size(0);
    return out_2d.view(out_sizes);
}

std::vector<torch::Tensor> causal_machine_qkv_projection_cuda(
    torch::Tensor x,
    torch::Tensor q_weight,
    c10::optional<torch::Tensor> q_bias,
    torch::Tensor k_weight,
    c10::optional<torch::Tensor> k_bias,
    torch::Tensor v_weight,
    c10::optional<torch::Tensor> v_bias) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_2d = x.reshape({-1, x.size(-1)}).contiguous();
    auto packed_weight = at::cat({q_weight, k_weight, v_weight}, 0).contiguous();
    c10::optional<torch::Tensor> packed_bias = c10::nullopt;
    if ((q_bias.has_value() && q_bias.value().defined())
        || (k_bias.has_value() && k_bias.value().defined())
        || (v_bias.has_value() && v_bias.value().defined())) {
        std::vector<torch::Tensor> pieces;
        pieces.reserve(3);
        pieces.push_back(q_bias.has_value() && q_bias.value().defined() ? q_bias.value() : torch::zeros({q_weight.size(0)}, q_weight.options()));
        pieces.push_back(k_bias.has_value() && k_bias.value().defined() ? k_bias.value() : torch::zeros({k_weight.size(0)}, k_weight.options()));
        pieces.push_back(v_bias.has_value() && v_bias.value().defined() ? v_bias.value() : torch::zeros({v_weight.size(0)}, v_weight.options()));
        packed_bias = at::cat(pieces, 0).contiguous();
    }
    auto packed_out = run_cublaslt_linear(x_2d, packed_weight, packed_bias);
    auto base_sizes = x.sizes().vec();
    auto q_sizes = base_sizes;
    auto k_sizes = base_sizes;
    auto v_sizes = base_sizes;
    q_sizes.back() = q_weight.size(0);
    k_sizes.back() = k_weight.size(0);
    v_sizes.back() = v_weight.size(0);
    auto splits = packed_out.split_with_sizes({q_weight.size(0), k_weight.size(0), v_weight.size(0)}, -1);
    return {
        splits[0].view(q_sizes),
        splits[1].view(k_sizes),
        splits[2].view(v_sizes),
    };
}
