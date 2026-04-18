#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <vector>

std::vector<torch::Tensor> causal_machine_branch_prep_cuda(
    torch::Tensor x,
    torch::Tensor x0,
    torch::Tensor mix_weights,
    double eps,
    double clamp_abs,
    double norm_scale);

std::vector<torch::Tensor> causal_machine_branch_prep_backward_cuda(
    torch::Tensor grad_mixed,
    torch::Tensor grad_normed,
    torch::Tensor x,
    torch::Tensor x0,
    torch::Tensor mix_weights,
    double eps,
    double clamp_abs,
    double norm_scale);

torch::Tensor causal_machine_branch_update_cuda(
    torch::Tensor x,
    torch::Tensor update,
    torch::Tensor scale,
    double residual_alpha,
    double clamp_abs);

std::vector<torch::Tensor> causal_machine_branch_update_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor update,
    torch::Tensor scale,
    double residual_alpha,
    double clamp_abs);

std::vector<torch::Tensor> causal_machine_partial_rotary_pair_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    c10::optional<torch::Tensor> scale,
    int64_t rope_dims,
    bool inverse_scale_k);

std::vector<torch::Tensor> causal_machine_partial_rotary_pair_backward_cuda(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor cos,
    torch::Tensor sin,
    c10::optional<torch::Tensor> scale,
    int64_t rope_dims,
    bool inverse_scale_k);

torch::Tensor causal_machine_relu_square_cuda(torch::Tensor x);
torch::Tensor causal_machine_relu_square_backward_cuda(torch::Tensor grad_output, torch::Tensor x);

torch::Tensor causal_machine_cublaslt_linear_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias);

std::vector<torch::Tensor> causal_machine_qkv_projection_cuda(
    torch::Tensor x,
    torch::Tensor q_weight,
    c10::optional<torch::Tensor> q_bias,
    torch::Tensor k_weight,
    c10::optional<torch::Tensor> k_bias,
    torch::Tensor v_weight,
    c10::optional<torch::Tensor> v_bias);

namespace {

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_optional_bias(
    const c10::optional<torch::Tensor>& bias,
    int64_t expected_size,
    const torch::Device& ref_device,
    torch::ScalarType ref_dtype,
    const char* name) {
    if (!bias.has_value() || !bias.value().defined()) {
        return;
    }
    const auto& b = bias.value();
    TORCH_CHECK(b.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(b.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(b.dim() == 1, name, " must be rank-1");
    TORCH_CHECK(b.size(0) == expected_size, name, " size mismatch");
    TORCH_CHECK(b.device() == ref_device, name, " device mismatch");
    TORCH_CHECK(b.scalar_type() == ref_dtype, name, " dtype mismatch");
}

void check_same_shape_dtype_device(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const char* a_name,
    const char* b_name) {
    check_cuda_tensor(a, a_name);
    check_cuda_tensor(b, b_name);
    TORCH_CHECK(a.sizes() == b.sizes(), a_name, " and ", b_name, " shape mismatch");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), a_name, " and ", b_name, " dtype mismatch");
    TORCH_CHECK(a.device() == b.device(), a_name, " and ", b_name, " device mismatch");
}

}  // namespace

std::vector<torch::Tensor> causal_machine_branch_prep(
    torch::Tensor x,
    torch::Tensor x0,
    torch::Tensor mix_weights,
    double eps,
    double clamp_abs,
    double norm_scale) {
    check_same_shape_dtype_device(x, x0, "x", "x0");
    check_cuda_tensor(mix_weights, "mix_weights");
    TORCH_CHECK(x.dim() >= 2, "branch_prep expects x rank >= 2");
    TORCH_CHECK(mix_weights.dim() == 2, "mix_weights must be rank-2");
    TORCH_CHECK(mix_weights.size(0) == 2, "mix_weights leading dim must be 2");
    TORCH_CHECK(mix_weights.size(1) == x.size(-1), "mix_weights trailing dim mismatch");
    TORCH_CHECK(mix_weights.scalar_type() == x.scalar_type(), "mix_weights dtype mismatch");
    TORCH_CHECK(mix_weights.device() == x.device(), "mix_weights device mismatch");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
    TORCH_CHECK(std::isfinite(clamp_abs) && clamp_abs >= 0.0, "clamp_abs must be finite and non-negative");
    TORCH_CHECK(std::isfinite(norm_scale), "norm_scale must be finite");
    return causal_machine_branch_prep_cuda(
        std::move(x),
        std::move(x0),
        std::move(mix_weights),
        eps,
        clamp_abs,
        norm_scale);
}

std::vector<torch::Tensor> causal_machine_branch_prep_backward(
    torch::Tensor grad_mixed,
    torch::Tensor grad_normed,
    torch::Tensor x,
    torch::Tensor x0,
    torch::Tensor mix_weights,
    double eps,
    double clamp_abs,
    double norm_scale) {
    check_same_shape_dtype_device(grad_mixed, x, "grad_mixed", "x");
    check_same_shape_dtype_device(grad_normed, x, "grad_normed", "x");
    check_same_shape_dtype_device(x, x0, "x", "x0");
    check_cuda_tensor(mix_weights, "mix_weights");
    TORCH_CHECK(x.dim() >= 2, "branch_prep_backward expects x rank >= 2");
    TORCH_CHECK(mix_weights.dim() == 2, "mix_weights must be rank-2");
    TORCH_CHECK(mix_weights.size(0) == 2, "mix_weights leading dim must be 2");
    TORCH_CHECK(mix_weights.size(1) == x.size(-1), "mix_weights trailing dim mismatch");
    TORCH_CHECK(mix_weights.scalar_type() == x.scalar_type(), "mix_weights dtype mismatch");
    TORCH_CHECK(mix_weights.device() == x.device(), "mix_weights device mismatch");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
    TORCH_CHECK(std::isfinite(clamp_abs) && clamp_abs >= 0.0, "clamp_abs must be finite and non-negative");
    TORCH_CHECK(std::isfinite(norm_scale), "norm_scale must be finite");
    return causal_machine_branch_prep_backward_cuda(
        std::move(grad_mixed),
        std::move(grad_normed),
        std::move(x),
        std::move(x0),
        std::move(mix_weights),
        eps,
        clamp_abs,
        norm_scale);
}

torch::Tensor causal_machine_branch_update(
    torch::Tensor x,
    torch::Tensor update,
    torch::Tensor scale,
    double residual_alpha,
    double clamp_abs) {
    check_same_shape_dtype_device(x, update, "x", "update");
    check_cuda_tensor(scale, "scale");
    TORCH_CHECK(x.dim() >= 2, "branch_update expects x rank >= 2");
    TORCH_CHECK(scale.dim() == 1, "scale must be rank-1");
    TORCH_CHECK(scale.size(0) == x.size(-1), "scale size mismatch");
    TORCH_CHECK(scale.scalar_type() == x.scalar_type(), "scale dtype mismatch");
    TORCH_CHECK(scale.device() == x.device(), "scale device mismatch");
    TORCH_CHECK(std::isfinite(residual_alpha), "residual_alpha must be finite");
    TORCH_CHECK(std::isfinite(clamp_abs) && clamp_abs >= 0.0, "clamp_abs must be finite and non-negative");
    return causal_machine_branch_update_cuda(
        std::move(x),
        std::move(update),
        std::move(scale),
        residual_alpha,
        clamp_abs);
}

std::vector<torch::Tensor> causal_machine_branch_update_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor update,
    torch::Tensor scale,
    double residual_alpha,
    double clamp_abs) {
    check_same_shape_dtype_device(grad_output, x, "grad_output", "x");
    check_same_shape_dtype_device(x, update, "x", "update");
    check_cuda_tensor(scale, "scale");
    TORCH_CHECK(scale.dim() == 1, "scale must be rank-1");
    TORCH_CHECK(scale.size(0) == x.size(-1), "scale size mismatch");
    TORCH_CHECK(scale.scalar_type() == x.scalar_type(), "scale dtype mismatch");
    TORCH_CHECK(scale.device() == x.device(), "scale device mismatch");
    TORCH_CHECK(std::isfinite(residual_alpha), "residual_alpha must be finite");
    TORCH_CHECK(std::isfinite(clamp_abs) && clamp_abs >= 0.0, "clamp_abs must be finite and non-negative");
    return causal_machine_branch_update_backward_cuda(
        std::move(grad_output),
        std::move(x),
        std::move(update),
        std::move(scale),
        residual_alpha,
        clamp_abs);
}

std::vector<torch::Tensor> causal_machine_partial_rotary_pair(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    c10::optional<torch::Tensor> scale,
    int64_t rope_dims,
    bool inverse_scale_k) {
    check_cuda_tensor(q, "q");
    check_cuda_tensor(k, "k");
    check_cuda_tensor(cos, "cos");
    check_cuda_tensor(sin, "sin");
    TORCH_CHECK(q.dim() == 4, "q must be [B, H, T, D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B, H_kv, T, D]");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k dtype mismatch");
    TORCH_CHECK(q.device() == k.device(), "q and k device mismatch");
    TORCH_CHECK(cos.scalar_type() == q.scalar_type(), "cos dtype mismatch");
    TORCH_CHECK(sin.scalar_type() == q.scalar_type(), "sin dtype mismatch");
    TORCH_CHECK(cos.device() == q.device(), "cos device mismatch");
    TORCH_CHECK(sin.device() == q.device(), "sin device mismatch");
    TORCH_CHECK(q.size(2) == k.size(2), "q and k sequence length mismatch");
    TORCH_CHECK(q.size(3) == k.size(3), "q and k head dim mismatch");
    TORCH_CHECK(cos.dim() == 4 && sin.dim() == 4, "cos and sin must be rank-4");
    TORCH_CHECK(cos.size(2) == q.size(2), "cos sequence length mismatch");
    TORCH_CHECK(sin.size(2) == q.size(2), "sin sequence length mismatch");
    TORCH_CHECK(cos.size(3) == sin.size(3), "cos and sin trailing dim mismatch");
    TORCH_CHECK(rope_dims >= 0 && rope_dims <= q.size(3), "rope_dims out of range");
    TORCH_CHECK(rope_dims % 2 == 0, "rope_dims must be even");
    if (scale.has_value() && scale.value().defined()) {
        check_cuda_tensor(scale.value(), "scale");
        TORCH_CHECK(scale.value().scalar_type() == q.scalar_type(), "scale dtype mismatch");
        TORCH_CHECK(scale.value().device() == q.device(), "scale device mismatch");
        TORCH_CHECK(scale.value().dim() == 4, "scale must be rank-4");
        TORCH_CHECK(scale.value().size(2) == q.size(2), "scale sequence length mismatch");
        TORCH_CHECK(scale.value().size(3) >= rope_dims / 2, "scale trailing dim mismatch");
    }
    return causal_machine_partial_rotary_pair_cuda(
        std::move(q),
        std::move(k),
        std::move(cos),
        std::move(sin),
        std::move(scale),
        rope_dims,
        inverse_scale_k);
}

std::vector<torch::Tensor> causal_machine_partial_rotary_pair_backward(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor cos,
    torch::Tensor sin,
    c10::optional<torch::Tensor> scale,
    int64_t rope_dims,
    bool inverse_scale_k) {
    check_cuda_tensor(grad_q, "grad_q");
    check_cuda_tensor(grad_k, "grad_k");
    check_cuda_tensor(cos, "cos");
    check_cuda_tensor(sin, "sin");
    TORCH_CHECK(grad_q.dim() == 4, "grad_q must be [B, H, T, D]");
    TORCH_CHECK(grad_k.dim() == 4, "grad_k must be [B, H_kv, T, D]");
    TORCH_CHECK(grad_q.scalar_type() == grad_k.scalar_type(), "grad_q and grad_k dtype mismatch");
    TORCH_CHECK(grad_q.device() == grad_k.device(), "grad_q and grad_k device mismatch");
    TORCH_CHECK(cos.scalar_type() == grad_q.scalar_type(), "cos dtype mismatch");
    TORCH_CHECK(sin.scalar_type() == grad_q.scalar_type(), "sin dtype mismatch");
    TORCH_CHECK(cos.device() == grad_q.device(), "cos device mismatch");
    TORCH_CHECK(sin.device() == grad_q.device(), "sin device mismatch");
    TORCH_CHECK(grad_q.size(2) == grad_k.size(2), "grad_q and grad_k sequence length mismatch");
    TORCH_CHECK(grad_q.size(3) == grad_k.size(3), "grad_q and grad_k head dim mismatch");
    TORCH_CHECK(cos.dim() == 4 && sin.dim() == 4, "cos and sin must be rank-4");
    TORCH_CHECK(cos.size(2) == grad_q.size(2), "cos sequence length mismatch");
    TORCH_CHECK(sin.size(2) == grad_q.size(2), "sin sequence length mismatch");
    TORCH_CHECK(cos.size(3) == sin.size(3), "cos and sin trailing dim mismatch");
    TORCH_CHECK(rope_dims >= 0 && rope_dims <= grad_q.size(3), "rope_dims out of range");
    TORCH_CHECK(rope_dims % 2 == 0, "rope_dims must be even");
    if (scale.has_value() && scale.value().defined()) {
        check_cuda_tensor(scale.value(), "scale");
        TORCH_CHECK(scale.value().scalar_type() == grad_q.scalar_type(), "scale dtype mismatch");
        TORCH_CHECK(scale.value().device() == grad_q.device(), "scale device mismatch");
        TORCH_CHECK(scale.value().dim() == 4, "scale must be rank-4");
        TORCH_CHECK(scale.value().size(2) == grad_q.size(2), "scale sequence length mismatch");
        TORCH_CHECK(scale.value().size(3) >= rope_dims / 2, "scale trailing dim mismatch");
    }
    return causal_machine_partial_rotary_pair_backward_cuda(
        std::move(grad_q),
        std::move(grad_k),
        std::move(cos),
        std::move(sin),
        std::move(scale),
        rope_dims,
        inverse_scale_k);
}

torch::Tensor causal_machine_relu_square(torch::Tensor x) {
    check_cuda_tensor(x, "x");
    return causal_machine_relu_square_cuda(std::move(x));
}

torch::Tensor causal_machine_relu_square_backward(torch::Tensor grad_output, torch::Tensor x) {
    check_same_shape_dtype_device(grad_output, x, "grad_output", "x");
    return causal_machine_relu_square_backward_cuda(std::move(grad_output), std::move(x));
}

torch::Tensor causal_machine_cublaslt_linear(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias) {
    check_cuda_tensor(x, "x");
    check_cuda_tensor(weight, "weight");
    TORCH_CHECK(x.dim() >= 2, "linear expects x rank >= 2");
    TORCH_CHECK(weight.dim() == 2, "weight must be rank-2");
    TORCH_CHECK(x.size(-1) == weight.size(1), "input feature size mismatch");
    TORCH_CHECK(x.device() == weight.device(), "x and weight device mismatch");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight dtype mismatch");
    check_optional_bias(bias, weight.size(0), x.device(), x.scalar_type(), "bias");
    return causal_machine_cublaslt_linear_cuda(std::move(x), std::move(weight), std::move(bias));
}

std::vector<torch::Tensor> causal_machine_qkv_projection(
    torch::Tensor x,
    torch::Tensor q_weight,
    c10::optional<torch::Tensor> q_bias,
    torch::Tensor k_weight,
    c10::optional<torch::Tensor> k_bias,
    torch::Tensor v_weight,
    c10::optional<torch::Tensor> v_bias) {
    check_cuda_tensor(x, "x");
    check_cuda_tensor(q_weight, "q_weight");
    check_cuda_tensor(k_weight, "k_weight");
    check_cuda_tensor(v_weight, "v_weight");
    TORCH_CHECK(x.dim() >= 2, "qkv_projection expects x rank >= 2");
    TORCH_CHECK(q_weight.dim() == 2 && k_weight.dim() == 2 && v_weight.dim() == 2, "q/k/v weight must be rank-2");
    TORCH_CHECK(
        x.size(-1) == q_weight.size(1) && x.size(-1) == k_weight.size(1) && x.size(-1) == v_weight.size(1),
        "q/k/v input feature size mismatch");
    TORCH_CHECK(q_weight.device() == x.device(), "q_weight device mismatch");
    TORCH_CHECK(k_weight.device() == x.device(), "k_weight device mismatch");
    TORCH_CHECK(v_weight.device() == x.device(), "v_weight device mismatch");
    TORCH_CHECK(q_weight.scalar_type() == x.scalar_type(), "q_weight dtype mismatch");
    TORCH_CHECK(k_weight.scalar_type() == x.scalar_type(), "k_weight dtype mismatch");
    TORCH_CHECK(v_weight.scalar_type() == x.scalar_type(), "v_weight dtype mismatch");
    check_optional_bias(q_bias, q_weight.size(0), x.device(), x.scalar_type(), "q_bias");
    check_optional_bias(k_bias, k_weight.size(0), x.device(), x.scalar_type(), "k_bias");
    check_optional_bias(v_bias, v_weight.size(0), x.device(), x.scalar_type(), "v_bias");
    return causal_machine_qkv_projection_cuda(
        std::move(x),
        std::move(q_weight),
        std::move(q_bias),
        std::move(k_weight),
        std::move(k_bias),
        std::move(v_weight),
        std::move(v_bias));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("branch_prep", &causal_machine_branch_prep, "Fused residual mix + sanitize + RMSNorm branch preparation (CUDA)");
    m.def(
        "branch_prep_backward",
        &causal_machine_branch_prep_backward,
        "Backward for fused residual mix + sanitize + RMSNorm branch preparation (CUDA)"
    );
    m.def("branch_update", &causal_machine_branch_update, "Fused residual branch update with per-channel scale (CUDA)");
    m.def(
        "branch_update_backward",
        &causal_machine_branch_update_backward,
        "Backward for fused residual branch update with per-channel scale (CUDA)"
    );
    m.def("partial_rotary_pair", &causal_machine_partial_rotary_pair, "Apply partial rotary embedding to q/k pair (CUDA)");
    m.def(
        "partial_rotary_pair_backward",
        &causal_machine_partial_rotary_pair_backward,
        "Backward for partial rotary embedding q/k pair (CUDA)"
    );
    m.def("relu_square", &causal_machine_relu_square, "Apply relu^2 pointwise activation (CUDA)");
    m.def("relu_square_backward", &causal_machine_relu_square_backward, "Backward for relu^2 pointwise activation (CUDA)");
    m.def("linear", &causal_machine_cublaslt_linear, "cuBLASLt-backed dense linear forward (CUDA)");
    m.def("qkv_projection", &causal_machine_qkv_projection, "Packed QKV projection forward via cuBLASLt (CUDA)");
}
