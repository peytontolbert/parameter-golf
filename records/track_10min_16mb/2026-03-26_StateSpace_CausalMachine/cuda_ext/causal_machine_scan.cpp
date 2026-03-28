#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> causal_machine_scan_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size);

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
    int64_t chunk_size);

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
    int64_t chunk_size);

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
    int64_t chunk_size);

namespace {

bool is_supported_activation_dtype(const torch::Tensor& tensor) {
    return tensor.scalar_type() == torch::kFloat32
        || tensor.scalar_type() == torch::kFloat16
        || tensor.scalar_type() == torch::kBFloat16;
}

void check_cuda_activation(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        is_supported_activation_dtype(tensor),
        name,
        " must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_float32(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_int8(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt8, name, " must be int8");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_cuda_device(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name) {
    TORCH_CHECK(
        tensor.get_device() == reference.get_device(),
        name,
        " must be on the same CUDA device as local_logits"
    );
}

void check_structured_shapes(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& transition_stay_probs) {
    TORCH_CHECK(local_logits.dim() == 3, "local_logits must have shape [B, L, 128]");
    TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits shape");
    TORCH_CHECK(local_logits.size(2) == 128, "local_logits last dim must be 128");
    TORCH_CHECK(transition_source_probs.dim() == 2, "transition_source_probs must have shape [128, R]");
    TORCH_CHECK(transition_source_probs.size(0) == 128, "transition_source_probs first dim must be 128");
    TORCH_CHECK(
        transition_source_probs.size(1) <= 128,
        "transition_rank must be <= 128 for causal_machine_scan_cuda"
    );
    TORCH_CHECK(transition_dest_probs.dim() == 2, "transition_dest_probs must have shape [R, 128]");
    TORCH_CHECK(transition_dest_probs.size(1) == 128, "transition_dest_probs last dim must be 128");
    TORCH_CHECK(
        transition_dest_probs.size(0) == transition_source_probs.size(1),
        "transition_source_probs and transition_dest_probs rank must match"
    );
    TORCH_CHECK(transition_stay_probs.dim() == 1, "transition_stay_probs must have shape [128]");
    TORCH_CHECK(transition_stay_probs.size(0) == 128, "transition_stay_probs size must be 128");
    TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape [B, 128]");
    TORCH_CHECK(initial_log_belief.size(0) == local_logits.size(0), "initial_log_belief batch must match local_logits");
    TORCH_CHECK(initial_log_belief.size(1) == 128, "initial_log_belief last dim must be 128");
}

void check_same_cuda_devices(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& transition_stay_probs) {
    check_same_cuda_device(transition_source_probs, local_logits, "transition_source_probs");
    check_same_cuda_device(transition_dest_probs, local_logits, "transition_dest_probs");
    check_same_cuda_device(transition_context, local_logits, "transition_context");
    check_same_cuda_device(initial_log_belief, local_logits, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, local_logits, "transition_stay_probs");
}

void check_structured_quantized_shapes(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_q,
    const torch::Tensor& transition_source_scales,
    const torch::Tensor& transition_dest_q,
    const torch::Tensor& transition_dest_scales,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& transition_stay_probs) {
    check_structured_shapes(
        local_logits,
        transition_source_q,
        transition_dest_q,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    TORCH_CHECK(
        transition_source_scales.dim() == 1 && transition_source_scales.size(0) == 128,
        "transition_source_scales must have shape [128]"
    );
    TORCH_CHECK(
        transition_dest_scales.dim() == 1 && transition_dest_scales.size(0) == transition_dest_q.size(0),
        "transition_dest_scales must have shape [R]"
    );
}

}  // namespace

std::vector<torch::Tensor> causal_machine_scan_forward(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_float32(transition_source_probs, "transition_source_probs");
    check_cuda_float32(transition_dest_probs, "transition_dest_probs");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_shapes(
        local_logits,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_devices(
        local_logits,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    TORCH_CHECK(
        transition_context.scalar_type() == local_logits.scalar_type(),
        "transition_context must match local_logits dtype"
    );
    TORCH_CHECK(
        initial_log_belief.scalar_type() == local_logits.scalar_type(),
        "initial_log_belief must match local_logits dtype"
    );
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (local_logits.size(0) == 0 || local_logits.size(1) == 0) {
        auto beliefs = torch::empty_like(local_logits);
        auto final_log_belief = torch::empty_like(initial_log_belief);
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    return causal_machine_scan_forward_cuda(
        local_logits,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_forward_quantized(
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
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_int8(transition_source_q, "transition_source_q");
    check_cuda_float32(transition_source_scales, "transition_source_scales");
    check_cuda_int8(transition_dest_q, "transition_dest_q");
    check_cuda_float32(transition_dest_scales, "transition_dest_scales");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_quantized_shapes(
        local_logits,
        transition_source_q,
        transition_source_scales,
        transition_dest_q,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_devices(
        local_logits,
        transition_source_q,
        transition_dest_q,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_device(transition_source_scales, local_logits, "transition_source_scales");
    check_same_cuda_device(transition_dest_scales, local_logits, "transition_dest_scales");
    TORCH_CHECK(
        transition_context.scalar_type() == local_logits.scalar_type(),
        "transition_context must match local_logits dtype"
    );
    TORCH_CHECK(
        initial_log_belief.scalar_type() == local_logits.scalar_type(),
        "initial_log_belief must match local_logits dtype"
    );
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (local_logits.size(0) == 0 || local_logits.size(1) == 0) {
        auto beliefs = torch::empty_like(local_logits);
        auto final_log_belief = torch::empty_like(initial_log_belief);
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    return causal_machine_scan_forward_quantized_cuda(
        local_logits,
        transition_source_q,
        transition_source_scales,
        transition_dest_q,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_backward(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    double transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_final_belief, "grad_final_belief");
    check_cuda_float32(transition_source_probs, "transition_source_probs");
    check_cuda_float32(transition_dest_probs, "transition_dest_probs");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_shapes(
        beliefs,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_devices(
        beliefs,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_device(grad_beliefs, beliefs, "grad_beliefs");
    check_same_cuda_device(grad_final_belief, beliefs, "grad_final_belief");
    TORCH_CHECK(grad_beliefs.sizes() == beliefs.sizes(), "grad_beliefs must match beliefs shape");
    TORCH_CHECK(grad_final_belief.sizes() == initial_log_belief.sizes(), "grad_final_belief must match initial_log_belief shape");
    TORCH_CHECK(grad_beliefs.scalar_type() == beliefs.scalar_type(), "grad_beliefs must match beliefs dtype");
    TORCH_CHECK(grad_final_belief.scalar_type() == beliefs.scalar_type(), "grad_final_belief must match beliefs dtype");
    TORCH_CHECK(transition_context.scalar_type() == beliefs.scalar_type(), "transition_context must match beliefs dtype");
    TORCH_CHECK(initial_log_belief.scalar_type() == beliefs.scalar_type(), "initial_log_belief must match beliefs dtype");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (beliefs.size(0) == 0 || beliefs.size(1) == 0) {
        return {
            torch::zeros_like(beliefs),
            torch::zeros_like(transition_source_probs),
            torch::zeros_like(transition_dest_probs),
            torch::zeros_like(transition_context),
            torch::zeros_like(initial_log_belief),
            torch::zeros({1}, beliefs.options().dtype(torch::kFloat32)),
            torch::zeros_like(transition_stay_probs),
        };
    }
    return causal_machine_scan_backward_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate,
        transition_stay_probs,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_backward_quantized(
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
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_final_belief, "grad_final_belief");
    check_cuda_int8(transition_source_q, "transition_source_q");
    check_cuda_float32(transition_source_scales, "transition_source_scales");
    check_cuda_int8(transition_dest_q, "transition_dest_q");
    check_cuda_float32(transition_dest_scales, "transition_dest_scales");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_quantized_shapes(
        beliefs,
        transition_source_q,
        transition_source_scales,
        transition_dest_q,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_devices(
        beliefs,
        transition_source_q,
        transition_dest_q,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_device(transition_source_scales, beliefs, "transition_source_scales");
    check_same_cuda_device(transition_dest_scales, beliefs, "transition_dest_scales");
    check_same_cuda_device(grad_beliefs, beliefs, "grad_beliefs");
    check_same_cuda_device(grad_final_belief, beliefs, "grad_final_belief");
    TORCH_CHECK(grad_beliefs.sizes() == beliefs.sizes(), "grad_beliefs must match beliefs shape");
    TORCH_CHECK(grad_final_belief.sizes() == initial_log_belief.sizes(), "grad_final_belief must match initial_log_belief shape");
    TORCH_CHECK(grad_beliefs.scalar_type() == beliefs.scalar_type(), "grad_beliefs must match beliefs dtype");
    TORCH_CHECK(grad_final_belief.scalar_type() == beliefs.scalar_type(), "grad_final_belief must match beliefs dtype");
    TORCH_CHECK(transition_context.scalar_type() == beliefs.scalar_type(), "transition_context must match beliefs dtype");
    TORCH_CHECK(initial_log_belief.scalar_type() == beliefs.scalar_type(), "initial_log_belief must match beliefs dtype");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (beliefs.size(0) == 0 || beliefs.size(1) == 0) {
        return {
            torch::zeros_like(beliefs),
            torch::zeros({transition_source_q.size(0), transition_source_q.size(1)}, beliefs.options().dtype(torch::kFloat32)),
            torch::zeros({transition_dest_q.size(0), transition_dest_q.size(1)}, beliefs.options().dtype(torch::kFloat32)),
            torch::zeros_like(transition_context),
            torch::zeros_like(initial_log_belief),
            torch::zeros({1}, beliefs.options().dtype(torch::kFloat32)),
            torch::zeros_like(transition_stay_probs),
        };
    }
    return causal_machine_scan_backward_quantized_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_q,
        transition_source_scales,
        transition_dest_q,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate,
        transition_stay_probs,
        chunk_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causal_machine_scan_forward, "Causal machine structured scan forward (CUDA)");
    m.def("forward_quantized", &causal_machine_scan_forward_quantized, "Causal machine structured scan forward quantized (CUDA)");
    m.def("backward", &causal_machine_scan_backward, "Causal machine structured scan backward (CUDA)");
    m.def("backward_quantized", &causal_machine_scan_backward_quantized, "Causal machine structured scan backward quantized (CUDA)");
}
