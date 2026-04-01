#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

std::vector<torch::Tensor> muon_grouped_step_cuda(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> momentum_bufs,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps);
void muon_grouped_step_batched_cuda(
    std::vector<torch::Tensor> params,
    torch::Tensor grad_batch,
    torch::Tensor momentum_batch,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps);
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
    double eps);
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
    double eps);
void muon_grouped_step_family_workspace_with_ptrs_cuda(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    torch::Tensor param_ptrs,
    torch::Tensor grad_ptrs,
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
    double eps);
void muon_grouped_step_family_workspace_capturable_cuda(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    torch::Tensor param_ptrs,
    torch::Tensor grad_ptrs,
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
    double eps);
int64_t muon_describe_square_backend_cuda(const torch::Tensor& x);
void muon_prewarm_square_backend_cuda(const torch::Tensor& x);

namespace {

bool is_supported_muon_workspace_dtype(torch::ScalarType dtype) {
    return dtype == torch::kFloat || dtype == torch::kBFloat16;
}

void check_pointer_tensor(
    const torch::Tensor& ptrs,
    int64_t expected_size,
    const torch::Device& ref_device,
    const char* name) {
    TORCH_CHECK(ptrs.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(ptrs.device() == ref_device, name, " device mismatch");
    TORCH_CHECK(ptrs.scalar_type() == torch::kInt64, name, " must be int64");
    TORCH_CHECK(ptrs.dim() == 1 && ptrs.size(0) == expected_size, name, " shape mismatch");
    TORCH_CHECK(ptrs.is_contiguous(), name, " must be contiguous");
}

void check_grouped_inputs(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    const std::vector<torch::Tensor>& momentum_bufs) {
    if (params.empty()) {
        throw std::invalid_argument("muon_grouped_step requires at least one parameter tensor");
    }
    if (params.size() != grads.size() || params.size() != momentum_bufs.size()) {
        throw std::invalid_argument("params, grads, and momentum_bufs must have matching lengths");
    }
    const auto ref_sizes = params.front().sizes();
    const auto ref_device = params.front().device();
    for (size_t idx = 0; idx < params.size(); ++idx) {
        const auto& p = params[idx];
        const auto& g = grads[idx];
        const auto& m = momentum_bufs[idx];
        TORCH_CHECK(p.is_cuda(), "muon_grouped_step expects CUDA params");
        TORCH_CHECK(g.is_cuda(), "muon_grouped_step expects CUDA grads");
        TORCH_CHECK(m.is_cuda(), "muon_grouped_step expects CUDA momentum buffers");
        TORCH_CHECK(p.dim() == 2, "muon_grouped_step expects matrix params");
        TORCH_CHECK(g.dim() == 2, "muon_grouped_step expects matrix grads");
        TORCH_CHECK(m.dim() == 2, "muon_grouped_step expects matrix momentum buffers");
        TORCH_CHECK(p.sizes() == ref_sizes, "muon_grouped_step requires same-shape grouped params");
        TORCH_CHECK(g.sizes() == ref_sizes, "muon_grouped_step requires same-shape grouped grads");
        TORCH_CHECK(m.sizes() == ref_sizes, "muon_grouped_step requires same-shape grouped momentum buffers");
        TORCH_CHECK(p.sizes() == g.sizes(), "param and grad shapes must match");
        TORCH_CHECK(p.sizes() == m.sizes(), "param and momentum buffer shapes must match");
        TORCH_CHECK(p.device() == ref_device, "all grouped params must share a device");
        TORCH_CHECK(g.device() == ref_device, "all grouped grads must share a device");
        TORCH_CHECK(m.device() == ref_device, "all grouped momentum buffers must share a device");
        TORCH_CHECK(p.is_contiguous(), "grouped params must be contiguous");
        TORCH_CHECK(g.is_contiguous(), "grouped grads must be contiguous");
        TORCH_CHECK(m.is_contiguous(), "grouped momentum buffers must be contiguous");
        TORCH_CHECK(
            p.scalar_type() == torch::kFloat || p.scalar_type() == torch::kBFloat16 || p.scalar_type() == torch::kHalf,
            "grouped params must use float, bf16, or fp16");
        TORCH_CHECK(
            g.scalar_type() == torch::kFloat || g.scalar_type() == torch::kBFloat16 || g.scalar_type() == torch::kHalf,
            "grouped grads must use float, bf16, or fp16");
        TORCH_CHECK(
            m.scalar_type() == torch::kFloat || m.scalar_type() == p.scalar_type(),
            "momentum buffer dtype must be float32 or match param dtype");
    }
}

}  // namespace

std::vector<torch::Tensor> muon_grouped_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> momentum_bufs,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps = 1.0e-7) {
    check_grouped_inputs(params, grads, momentum_bufs);
    TORCH_CHECK(std::isfinite(lr), "lr must be finite");
    TORCH_CHECK(std::isfinite(momentum), "momentum must be finite");
    TORCH_CHECK(std::isfinite(weight_decay), "weight_decay must be finite");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
    TORCH_CHECK(ns_steps >= 0, "ns_steps must be non-negative");
    return muon_grouped_step_cuda(
        std::move(params),
        std::move(grads),
        std::move(momentum_bufs),
        lr,
        momentum,
        weight_decay,
        nesterov,
        ns_steps,
        eps);
}

void muon_grouped_step_batched(
    std::vector<torch::Tensor> params,
    torch::Tensor grad_batch,
    torch::Tensor momentum_batch,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps = 1.0e-7) {
    TORCH_CHECK(!params.empty(), "muon_grouped_step_batched requires at least one parameter tensor");
    const auto ref_sizes = params.front().sizes();
    const auto ref_device = params.front().device();
    TORCH_CHECK(grad_batch.is_cuda(), "muon_grouped_step_batched expects CUDA grad_batch");
    TORCH_CHECK(momentum_batch.is_cuda(), "muon_grouped_step_batched expects CUDA momentum_batch");
    TORCH_CHECK(grad_batch.dim() == 3, "muon_grouped_step_batched expects grad_batch with shape [B, M, N]");
    TORCH_CHECK(momentum_batch.dim() == 3, "muon_grouped_step_batched expects momentum_batch with shape [B, M, N]");
    TORCH_CHECK(
        grad_batch.size(0) == static_cast<int64_t>(params.size()),
        "grad_batch leading dim must match params size");
    TORCH_CHECK(
        momentum_batch.size(0) == static_cast<int64_t>(params.size()),
        "momentum_batch leading dim must match params size");
    TORCH_CHECK(grad_batch.size(1) == ref_sizes[0] && grad_batch.size(2) == ref_sizes[1], "grad_batch shape mismatch");
    TORCH_CHECK(
        momentum_batch.size(1) == ref_sizes[0] && momentum_batch.size(2) == ref_sizes[1],
        "momentum_batch shape mismatch");
    TORCH_CHECK(grad_batch.device() == ref_device, "grad_batch device mismatch");
    TORCH_CHECK(momentum_batch.device() == ref_device, "momentum_batch device mismatch");
    TORCH_CHECK(grad_batch.is_contiguous(), "grad_batch must be contiguous");
    TORCH_CHECK(momentum_batch.is_contiguous(), "momentum_batch must be contiguous");
    TORCH_CHECK(
        grad_batch.scalar_type() == torch::kFloat || grad_batch.scalar_type() == torch::kBFloat16 || grad_batch.scalar_type() == torch::kHalf,
        "grad_batch must use float, bf16, or fp16");
    TORCH_CHECK(
        momentum_batch.scalar_type() == torch::kFloat || momentum_batch.scalar_type() == torch::kBFloat16 || momentum_batch.scalar_type() == torch::kHalf,
        "momentum_batch must use float, bf16, or fp16");
    for (const auto& p : params) {
        TORCH_CHECK(p.is_cuda(), "muon_grouped_step_batched expects CUDA params");
        TORCH_CHECK(p.dim() == 2, "muon_grouped_step_batched expects matrix params");
        TORCH_CHECK(p.sizes() == ref_sizes, "muon_grouped_step_batched requires same-shape params");
        TORCH_CHECK(p.device() == ref_device, "all grouped params must share a device");
        TORCH_CHECK(p.is_contiguous(), "grouped params must be contiguous");
    }
    TORCH_CHECK(std::isfinite(lr), "lr must be finite");
    TORCH_CHECK(std::isfinite(momentum), "momentum must be finite");
    TORCH_CHECK(std::isfinite(weight_decay), "weight_decay must be finite");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
    TORCH_CHECK(ns_steps >= 0, "ns_steps must be non-negative");
    muon_grouped_step_batched_cuda(
        std::move(params),
        std::move(grad_batch),
        std::move(momentum_batch),
        lr,
        momentum,
        weight_decay,
        nesterov,
        ns_steps,
        eps);
}

void muon_grouped_step_workspace(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    torch::Tensor grad_batch,
    torch::Tensor momentum_batch,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov,
    int64_t ns_steps,
    double eps = 1.0e-7) {
    if (params.empty()) {
        throw std::invalid_argument("muon_grouped_step_workspace requires at least one parameter tensor");
    }
    if (params.size() != grads.size()) {
        throw std::invalid_argument("params and grads must have matching lengths");
    }
    const auto ref_sizes = params.front().sizes();
    const auto ref_device = params.front().device();
    for (size_t idx = 0; idx < params.size(); ++idx) {
        const auto& p = params[idx];
        const auto& g = grads[idx];
        TORCH_CHECK(p.is_cuda(), "muon_grouped_step_workspace expects CUDA params");
        TORCH_CHECK(g.is_cuda(), "muon_grouped_step_workspace expects CUDA grads");
        TORCH_CHECK(p.dim() == 2, "muon_grouped_step_workspace expects matrix params");
        TORCH_CHECK(g.dim() == 2, "muon_grouped_step_workspace expects matrix grads");
        TORCH_CHECK(p.sizes() == ref_sizes, "muon_grouped_step_workspace requires same-shape params");
        TORCH_CHECK(g.sizes() == ref_sizes, "muon_grouped_step_workspace requires same-shape grads");
        TORCH_CHECK(p.device() == ref_device, "all grouped params must share a device");
        TORCH_CHECK(g.device() == ref_device, "all grouped grads must share a device");
        TORCH_CHECK(p.is_contiguous(), "grouped params must be contiguous");
        TORCH_CHECK(g.is_contiguous(), "grouped grads must be contiguous");
    }
    TORCH_CHECK(grad_batch.is_cuda(), "muon_grouped_step_workspace expects CUDA grad_batch");
    TORCH_CHECK(momentum_batch.is_cuda(), "muon_grouped_step_workspace expects CUDA momentum_batch");
    TORCH_CHECK(grad_batch.dim() == 3, "muon_grouped_step_workspace expects grad_batch with shape [B, M, N]");
    TORCH_CHECK(momentum_batch.dim() == 3, "muon_grouped_step_workspace expects momentum_batch with shape [B, M, N]");
    TORCH_CHECK(
        grad_batch.size(0) == static_cast<int64_t>(params.size()) && grad_batch.size(1) == ref_sizes[0] && grad_batch.size(2) == ref_sizes[1],
        "grad_batch shape mismatch");
    TORCH_CHECK(
        momentum_batch.size(0) == static_cast<int64_t>(params.size()) && momentum_batch.size(1) == ref_sizes[0] && momentum_batch.size(2) == ref_sizes[1],
        "momentum_batch shape mismatch");
    TORCH_CHECK(grad_batch.is_contiguous(), "grad_batch must be contiguous");
    TORCH_CHECK(momentum_batch.is_contiguous(), "momentum_batch must be contiguous");
    TORCH_CHECK(std::isfinite(lr), "lr must be finite");
    TORCH_CHECK(std::isfinite(momentum), "momentum must be finite");
    TORCH_CHECK(std::isfinite(weight_decay), "weight_decay must be finite");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
    TORCH_CHECK(ns_steps >= 0, "ns_steps must be non-negative");
    muon_grouped_step_workspace_cuda(
        std::move(params),
        std::move(grads),
        std::move(grad_batch),
        std::move(momentum_batch),
        lr,
        momentum,
        weight_decay,
        nesterov,
        ns_steps,
        eps);
}

void muon_grouped_step_family_workspace(
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
    double eps = 1.0e-7) {
    if (params.empty()) {
        throw std::invalid_argument("muon_grouped_step_family_workspace requires at least one parameter tensor");
    }
    if (params.size() != grads.size()) {
        throw std::invalid_argument("params and grads must have matching lengths");
    }
    TORCH_CHECK(family_code >= 0 && family_code <= 3, "invalid Muon family code");
    const auto ref_sizes = params.front().sizes();
    const auto ref_device = params.front().device();
    const int64_t bucket_size = static_cast<int64_t>(params.size());
    const int64_t rows = ref_sizes[0];
    const int64_t cols = ref_sizes[1];
    const bool transpose_input = family_code == 1;
    const int64_t ns_rows = transpose_input ? cols : rows;
    const int64_t ns_cols = transpose_input ? rows : cols;
    for (size_t idx = 0; idx < params.size(); ++idx) {
        const auto& p = params[idx];
        const auto& g = grads[idx];
        TORCH_CHECK(p.is_cuda(), "muon_grouped_step_family_workspace expects CUDA params");
        TORCH_CHECK(g.is_cuda(), "muon_grouped_step_family_workspace expects CUDA grads");
        TORCH_CHECK(p.dim() == 2, "muon_grouped_step_family_workspace expects matrix params");
        TORCH_CHECK(g.dim() == 2, "muon_grouped_step_family_workspace expects matrix grads");
        TORCH_CHECK(p.sizes() == ref_sizes, "muon_grouped_step_family_workspace requires same-shape params");
        TORCH_CHECK(g.sizes() == ref_sizes, "muon_grouped_step_family_workspace requires same-shape grads");
        TORCH_CHECK(p.device() == ref_device, "all grouped params must share a device");
        TORCH_CHECK(g.device() == ref_device, "all grouped grads must share a device");
        TORCH_CHECK(p.is_contiguous(), "grouped params must be contiguous");
        TORCH_CHECK(g.is_contiguous(), "grouped grads must be contiguous");
    }
    TORCH_CHECK(
        effective_batch.is_cuda() && effective_batch.dim() == 3 &&
            effective_batch.size(0) == bucket_size && effective_batch.size(1) == rows && effective_batch.size(2) == cols,
        "effective_batch shape mismatch");
    TORCH_CHECK(
        momentum_batch.is_cuda() && momentum_batch.dim() == 3 &&
            momentum_batch.size(0) == bucket_size && momentum_batch.size(1) == rows && momentum_batch.size(2) == cols,
        "momentum_batch shape mismatch");
    TORCH_CHECK(
        norms.is_cuda() && norms.dim() == 2 &&
            norms.size(0) == bucket_size && norms.size(1) == 1,
        "norms shape mismatch");
    TORCH_CHECK(
        ns_input_batch.is_cuda() && ns_input_batch.dim() == 3 &&
            ns_input_batch.size(0) == bucket_size && ns_input_batch.size(1) == ns_rows && ns_input_batch.size(2) == ns_cols,
        "ns_input_batch shape mismatch");
    TORCH_CHECK(
        gram_batch.is_cuda() && gram_batch.dim() == 3 &&
            gram_batch.size(0) == bucket_size && gram_batch.size(1) == ns_rows && gram_batch.size(2) == ns_rows,
        "gram_batch shape mismatch");
    TORCH_CHECK(
        gram_sq_batch.is_cuda() && gram_sq_batch.dim() == 3 &&
            gram_sq_batch.size(0) == bucket_size && gram_sq_batch.size(1) == ns_rows && gram_sq_batch.size(2) == ns_rows,
        "gram_sq_batch shape mismatch");
    TORCH_CHECK(
        next_x_batch.is_cuda() && next_x_batch.dim() == 3 &&
            next_x_batch.size(0) == bucket_size && next_x_batch.size(1) == ns_rows && next_x_batch.size(2) == ns_cols,
        "next_x_batch shape mismatch");
    TORCH_CHECK(effective_batch.is_contiguous(), "effective_batch must be contiguous");
    TORCH_CHECK(momentum_batch.is_contiguous(), "momentum_batch must be contiguous");
    TORCH_CHECK(norms.is_contiguous(), "norms must be contiguous");
    TORCH_CHECK(ns_input_batch.is_contiguous(), "ns_input_batch must be contiguous");
    TORCH_CHECK(gram_batch.is_contiguous(), "gram_batch must be contiguous");
    TORCH_CHECK(gram_sq_batch.is_contiguous(), "gram_sq_batch must be contiguous");
    TORCH_CHECK(next_x_batch.is_contiguous(), "next_x_batch must be contiguous");
    TORCH_CHECK(effective_batch.scalar_type() == torch::kFloat, "effective_batch must be float32");
    TORCH_CHECK(momentum_batch.scalar_type() == torch::kFloat, "momentum_batch must be float32");
    TORCH_CHECK(norms.scalar_type() == torch::kFloat, "norms must be float32");
    const auto workspace_dtype = ns_input_batch.scalar_type();
    TORCH_CHECK(
        is_supported_muon_workspace_dtype(workspace_dtype),
        "ns_input_batch must be float32 or bf16");
    TORCH_CHECK(gram_batch.scalar_type() == workspace_dtype, "gram_batch dtype must match ns_input_batch");
    TORCH_CHECK(gram_sq_batch.scalar_type() == workspace_dtype, "gram_sq_batch dtype must match ns_input_batch");
    TORCH_CHECK(next_x_batch.scalar_type() == workspace_dtype, "next_x_batch dtype must match ns_input_batch");
    TORCH_CHECK(std::isfinite(lr), "lr must be finite");
    TORCH_CHECK(std::isfinite(momentum), "momentum must be finite");
    TORCH_CHECK(std::isfinite(weight_decay), "weight_decay must be finite");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
    TORCH_CHECK(ns_steps >= 0, "ns_steps must be non-negative");
    muon_grouped_step_family_workspace_cuda(
        std::move(params),
        std::move(grads),
        std::move(effective_batch),
        std::move(momentum_batch),
        std::move(norms),
        std::move(ns_input_batch),
        std::move(gram_batch),
        std::move(gram_sq_batch),
        std::move(next_x_batch),
        family_code,
        lr,
        momentum,
        weight_decay,
        nesterov,
        ns_steps,
        eps);
}

void muon_grouped_step_family_workspace_with_ptrs(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    torch::Tensor param_ptrs,
    torch::Tensor grad_ptrs,
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
    double eps = 1.0e-7) {
    if (params.empty()) {
        throw std::invalid_argument("muon_grouped_step_family_workspace_with_ptrs requires at least one parameter tensor");
    }
    if (params.size() != grads.size()) {
        throw std::invalid_argument("params and grads must have matching lengths");
    }
    TORCH_CHECK(family_code >= 0 && family_code <= 3, "invalid Muon family code");
    const auto ref_sizes = params.front().sizes();
    const auto ref_device = params.front().device();
    const int64_t bucket_size = static_cast<int64_t>(params.size());
    const int64_t rows = ref_sizes[0];
    const int64_t cols = ref_sizes[1];
    const bool transpose_input = family_code == 1;
    const int64_t ns_rows = transpose_input ? cols : rows;
    const int64_t ns_cols = transpose_input ? rows : cols;
    for (size_t idx = 0; idx < params.size(); ++idx) {
        const auto& p = params[idx];
        const auto& g = grads[idx];
        TORCH_CHECK(p.is_cuda(), "muon_grouped_step_family_workspace_with_ptrs expects CUDA params");
        TORCH_CHECK(g.is_cuda(), "muon_grouped_step_family_workspace_with_ptrs expects CUDA grads");
        TORCH_CHECK(p.dim() == 2, "muon_grouped_step_family_workspace_with_ptrs expects matrix params");
        TORCH_CHECK(g.dim() == 2, "muon_grouped_step_family_workspace_with_ptrs expects matrix grads");
        TORCH_CHECK(p.sizes() == ref_sizes, "muon_grouped_step_family_workspace_with_ptrs requires same-shape params");
        TORCH_CHECK(g.sizes() == ref_sizes, "muon_grouped_step_family_workspace_with_ptrs requires same-shape grads");
        TORCH_CHECK(p.device() == ref_device, "all grouped params must share a device");
        TORCH_CHECK(g.device() == ref_device, "all grouped grads must share a device");
        TORCH_CHECK(p.is_contiguous(), "grouped params must be contiguous");
        TORCH_CHECK(g.is_contiguous(), "grouped grads must be contiguous");
    }
    TORCH_CHECK(
        effective_batch.is_cuda() && effective_batch.dim() == 3 &&
            effective_batch.size(0) == bucket_size && effective_batch.size(1) == rows && effective_batch.size(2) == cols,
        "effective_batch shape mismatch");
    TORCH_CHECK(
        momentum_batch.is_cuda() && momentum_batch.dim() == 3 &&
            momentum_batch.size(0) == bucket_size && momentum_batch.size(1) == rows && momentum_batch.size(2) == cols,
        "momentum_batch shape mismatch");
    TORCH_CHECK(
        norms.is_cuda() && norms.dim() == 2 &&
            norms.size(0) == bucket_size && norms.size(1) == 1,
        "norms shape mismatch");
    TORCH_CHECK(
        ns_input_batch.is_cuda() && ns_input_batch.dim() == 3 &&
            ns_input_batch.size(0) == bucket_size && ns_input_batch.size(1) == ns_rows && ns_input_batch.size(2) == ns_cols,
        "ns_input_batch shape mismatch");
    TORCH_CHECK(
        gram_batch.is_cuda() && gram_batch.dim() == 3 &&
            gram_batch.size(0) == bucket_size && gram_batch.size(1) == ns_rows && gram_batch.size(2) == ns_rows,
        "gram_batch shape mismatch");
    TORCH_CHECK(
        gram_sq_batch.is_cuda() && gram_sq_batch.dim() == 3 &&
            gram_sq_batch.size(0) == bucket_size && gram_sq_batch.size(1) == ns_rows && gram_sq_batch.size(2) == ns_rows,
        "gram_sq_batch shape mismatch");
    TORCH_CHECK(
        next_x_batch.is_cuda() && next_x_batch.dim() == 3 &&
            next_x_batch.size(0) == bucket_size && next_x_batch.size(1) == ns_rows && next_x_batch.size(2) == ns_cols,
        "next_x_batch shape mismatch");
    TORCH_CHECK(effective_batch.scalar_type() == torch::kFloat, "effective_batch must be float32");
    TORCH_CHECK(momentum_batch.scalar_type() == torch::kFloat, "momentum_batch must be float32");
    TORCH_CHECK(norms.scalar_type() == torch::kFloat, "norms must be float32");
    const auto workspace_dtype = ns_input_batch.scalar_type();
    TORCH_CHECK(
        is_supported_muon_workspace_dtype(workspace_dtype),
        "ns_input_batch must be float32 or bf16");
    TORCH_CHECK(gram_batch.scalar_type() == workspace_dtype, "gram_batch dtype must match ns_input_batch");
    TORCH_CHECK(gram_sq_batch.scalar_type() == workspace_dtype, "gram_sq_batch dtype must match ns_input_batch");
    TORCH_CHECK(next_x_batch.scalar_type() == workspace_dtype, "next_x_batch dtype must match ns_input_batch");
    check_pointer_tensor(param_ptrs, bucket_size, ref_device, "param_ptrs");
    check_pointer_tensor(grad_ptrs, bucket_size, ref_device, "grad_ptrs");
    TORCH_CHECK(std::isfinite(lr), "lr must be finite");
    TORCH_CHECK(std::isfinite(momentum), "momentum must be finite");
    TORCH_CHECK(std::isfinite(weight_decay), "weight_decay must be finite");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
    TORCH_CHECK(ns_steps >= 0, "ns_steps must be non-negative");
    muon_grouped_step_family_workspace_with_ptrs_cuda(
        std::move(params),
        std::move(grads),
        std::move(param_ptrs),
        std::move(grad_ptrs),
        std::move(effective_batch),
        std::move(momentum_batch),
        std::move(norms),
        std::move(ns_input_batch),
        std::move(gram_batch),
        std::move(gram_sq_batch),
        std::move(next_x_batch),
        family_code,
        lr,
        momentum,
        weight_decay,
        nesterov,
        ns_steps,
        eps);
}

void muon_grouped_step_family_workspace_capturable(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    torch::Tensor param_ptrs,
    torch::Tensor grad_ptrs,
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
    double eps = 1.0e-7) {
    if (params.empty()) {
        throw std::invalid_argument("muon_grouped_step_family_workspace_capturable requires at least one parameter tensor");
    }
    if (params.size() != grads.size()) {
        throw std::invalid_argument("params and grads must have matching lengths");
    }
    TORCH_CHECK(family_code >= 0 && family_code <= 3, "invalid Muon family code");
    const auto ref_sizes = params.front().sizes();
    const auto ref_device = params.front().device();
    const int64_t bucket_size = static_cast<int64_t>(params.size());
    const int64_t rows = ref_sizes[0];
    const int64_t cols = ref_sizes[1];
    const bool transpose_input = family_code == 1;
    const int64_t ns_rows = transpose_input ? cols : rows;
    const int64_t ns_cols = transpose_input ? rows : cols;
    for (size_t idx = 0; idx < params.size(); ++idx) {
        const auto& p = params[idx];
        const auto& g = grads[idx];
        TORCH_CHECK(p.is_cuda(), "muon_grouped_step_family_workspace_capturable expects CUDA params");
        TORCH_CHECK(g.is_cuda(), "muon_grouped_step_family_workspace_capturable expects CUDA grads");
        TORCH_CHECK(p.dim() == 2, "muon_grouped_step_family_workspace_capturable expects matrix params");
        TORCH_CHECK(g.dim() == 2, "muon_grouped_step_family_workspace_capturable expects matrix grads");
        TORCH_CHECK(p.sizes() == ref_sizes, "muon_grouped_step_family_workspace_capturable requires same-shape params");
        TORCH_CHECK(g.sizes() == ref_sizes, "muon_grouped_step_family_workspace_capturable requires same-shape grads");
        TORCH_CHECK(p.device() == ref_device, "all grouped params must share a device");
        TORCH_CHECK(g.device() == ref_device, "all grouped grads must share a device");
        TORCH_CHECK(p.is_contiguous(), "grouped params must be contiguous");
        TORCH_CHECK(g.is_contiguous(), "grouped grads must be contiguous");
    }
    TORCH_CHECK(
        effective_batch.is_cuda() && effective_batch.dim() == 3 &&
            effective_batch.size(0) == bucket_size && effective_batch.size(1) == rows && effective_batch.size(2) == cols,
        "effective_batch shape mismatch");
    TORCH_CHECK(
        momentum_batch.is_cuda() && momentum_batch.dim() == 3 &&
            momentum_batch.size(0) == bucket_size && momentum_batch.size(1) == rows && momentum_batch.size(2) == cols,
        "momentum_batch shape mismatch");
    TORCH_CHECK(
        norms.is_cuda() && norms.dim() == 2 &&
            norms.size(0) == bucket_size && norms.size(1) == 1,
        "norms shape mismatch");
    TORCH_CHECK(
        ns_input_batch.is_cuda() && ns_input_batch.dim() == 3 &&
            ns_input_batch.size(0) == bucket_size && ns_input_batch.size(1) == ns_rows && ns_input_batch.size(2) == ns_cols,
        "ns_input_batch shape mismatch");
    TORCH_CHECK(
        gram_batch.is_cuda() && gram_batch.dim() == 3 &&
            gram_batch.size(0) == bucket_size && gram_batch.size(1) == ns_rows && gram_batch.size(2) == ns_rows,
        "gram_batch shape mismatch");
    TORCH_CHECK(
        gram_sq_batch.is_cuda() && gram_sq_batch.dim() == 3 &&
            gram_sq_batch.size(0) == bucket_size && gram_sq_batch.size(1) == ns_rows && gram_sq_batch.size(2) == ns_rows,
        "gram_sq_batch shape mismatch");
    TORCH_CHECK(
        next_x_batch.is_cuda() && next_x_batch.dim() == 3 &&
            next_x_batch.size(0) == bucket_size && next_x_batch.size(1) == ns_rows && next_x_batch.size(2) == ns_cols,
        "next_x_batch shape mismatch");
    TORCH_CHECK(effective_batch.scalar_type() == torch::kFloat, "effective_batch must be float32");
    TORCH_CHECK(momentum_batch.scalar_type() == torch::kFloat, "momentum_batch must be float32");
    TORCH_CHECK(norms.scalar_type() == torch::kFloat, "norms must be float32");
    const auto workspace_dtype = ns_input_batch.scalar_type();
    TORCH_CHECK(
        is_supported_muon_workspace_dtype(workspace_dtype),
        "ns_input_batch must be float32 or bf16");
    TORCH_CHECK(gram_batch.scalar_type() == workspace_dtype, "gram_batch dtype must match ns_input_batch");
    TORCH_CHECK(gram_sq_batch.scalar_type() == workspace_dtype, "gram_sq_batch dtype must match ns_input_batch");
    TORCH_CHECK(next_x_batch.scalar_type() == workspace_dtype, "next_x_batch dtype must match ns_input_batch");
    check_pointer_tensor(param_ptrs, bucket_size, ref_device, "param_ptrs");
    check_pointer_tensor(grad_ptrs, bucket_size, ref_device, "grad_ptrs");
    TORCH_CHECK(lr.is_cuda() && lr.scalar_type() == torch::kFloat && lr.numel() == 1, "lr must be a CUDA float32 scalar tensor");
    TORCH_CHECK(momentum.is_cuda() && momentum.scalar_type() == torch::kFloat && momentum.numel() == 1, "momentum must be a CUDA float32 scalar tensor");
    TORCH_CHECK(weight_decay.is_cuda() && weight_decay.scalar_type() == torch::kFloat && weight_decay.numel() == 1, "weight_decay must be a CUDA float32 scalar tensor");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
    TORCH_CHECK(ns_steps >= 0, "ns_steps must be non-negative");
    muon_grouped_step_family_workspace_capturable_cuda(
        std::move(params),
        std::move(grads),
        std::move(param_ptrs),
        std::move(grad_ptrs),
        std::move(effective_batch),
        std::move(momentum_batch),
        std::move(norms),
        std::move(ns_input_batch),
        std::move(gram_batch),
        std::move(gram_sq_batch),
        std::move(next_x_batch),
        family_code,
        std::move(lr),
        std::move(momentum),
        std::move(weight_decay),
        nesterov,
        ns_steps,
        eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_step", &muon_grouped_step, "Grouped Muon step for same-shape matrix buckets (CUDA)");
    m.def("grouped_step_batched", &muon_grouped_step_batched, "Grouped Muon step with pre-batched grads and momentum (CUDA)");
    m.def("grouped_step_workspace", &muon_grouped_step_workspace, "Grouped Muon step with preallocated grad and momentum workspaces (CUDA)");
    m.def(
        "grouped_step_family_workspace",
        &muon_grouped_step_family_workspace,
        "Grouped Muon step with family-specialized workspaces (CUDA)");
    m.def(
        "grouped_step_family_workspace_with_ptrs",
        &muon_grouped_step_family_workspace_with_ptrs,
        "Grouped Muon step with family-specialized workspaces and cached pointer tensors (CUDA)");
    m.def(
        "grouped_step_family_workspace_capturable",
        &muon_grouped_step_family_workspace_capturable,
        "Grouped Muon step with family-specialized workspaces and CUDA scalar tensors (CUDA)");
    m.def("describe_square_backend", &muon_describe_square_backend_cuda, "Describe Muon square backend selection (CUDA)");
    m.def("prewarm_square_backend", &muon_prewarm_square_backend_cuda, "Prewarm Muon square backend policy objects (CUDA)");
}
