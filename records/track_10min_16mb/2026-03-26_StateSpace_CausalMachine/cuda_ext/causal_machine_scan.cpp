#include <torch/extension.h>

#include <stdexcept>
#include <vector>

std::vector<torch::Tensor> causal_machine_scan_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    int64_t chunk_size);

std::vector<torch::Tensor> causal_machine_scan_backward_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor final_belief,
    double transition_gate,
    int64_t chunk_size);

namespace {

void check_cuda_float32(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_shapes(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_log_probs,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief) {
    TORCH_CHECK(local_logits.dim() == 3, "local_logits must have shape [B, L, 128]");
    TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits shape");
    TORCH_CHECK(local_logits.size(2) == 128, "local_logits last dim must be 128");
    TORCH_CHECK(transition_log_probs.dim() == 2, "transition_log_probs must have shape [128, 128]");
    TORCH_CHECK(transition_log_probs.size(0) == 128 && transition_log_probs.size(1) == 128, "transition_log_probs must be [128, 128]");
    TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape [B, 128]");
    TORCH_CHECK(initial_log_belief.size(0) == local_logits.size(0), "initial_log_belief batch must match local_logits");
    TORCH_CHECK(initial_log_belief.size(1) == 128, "initial_log_belief last dim must be 128");
}

}  // namespace

std::vector<torch::Tensor> causal_machine_scan_forward(
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    double transition_gate,
    int64_t chunk_size) {
    check_cuda_float32(local_logits, "local_logits");
    check_cuda_float32(transition_log_probs, "transition_log_probs");
    check_cuda_float32(transition_context, "transition_context");
    check_cuda_float32(initial_log_belief, "initial_log_belief");
    check_shapes(local_logits, transition_log_probs, transition_context, initial_log_belief);
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    return causal_machine_scan_forward_cuda(
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        transition_gate,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_backward(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor local_logits,
    torch::Tensor transition_log_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor final_belief,
    double transition_gate,
    int64_t chunk_size) {
    check_cuda_float32(grad_beliefs, "grad_beliefs");
    check_cuda_float32(grad_final_belief, "grad_final_belief");
    check_cuda_float32(local_logits, "local_logits");
    check_cuda_float32(transition_log_probs, "transition_log_probs");
    check_cuda_float32(transition_context, "transition_context");
    check_cuda_float32(initial_log_belief, "initial_log_belief");
    check_cuda_float32(beliefs, "beliefs");
    check_cuda_float32(final_belief, "final_belief");
    check_shapes(local_logits, transition_log_probs, transition_context, initial_log_belief);
    TORCH_CHECK(grad_beliefs.sizes() == beliefs.sizes(), "grad_beliefs must match beliefs shape");
    TORCH_CHECK(final_belief.sizes() == initial_log_belief.sizes(), "final_belief must match initial_log_belief shape");
    TORCH_CHECK(grad_final_belief.sizes() == final_belief.sizes(), "grad_final_belief must match final_belief shape");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    return causal_machine_scan_backward_cuda(
        grad_beliefs,
        grad_final_belief,
        local_logits,
        transition_log_probs,
        transition_context,
        initial_log_belief,
        beliefs,
        final_belief,
        transition_gate,
        chunk_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causal_machine_scan_forward, "Causal machine chunked scan forward (CUDA)");
    m.def("backward", &causal_machine_scan_backward, "Causal machine chunked scan backward (CUDA)");
}
