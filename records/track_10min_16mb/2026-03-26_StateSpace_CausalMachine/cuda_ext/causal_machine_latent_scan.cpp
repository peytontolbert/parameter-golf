#include <torch/extension.h>

#include <cmath>
#include <vector>

std::vector<torch::Tensor> causal_machine_latent_scan_forward_cuda(
    torch::Tensor drive,
    torch::Tensor decay,
    torch::Tensor initial_state);

std::vector<torch::Tensor> causal_machine_latent_prior_scan_forward_cuda(
    torch::Tensor drive,
    torch::Tensor decay,
    torch::Tensor initial_state);

std::vector<torch::Tensor> causal_machine_latent_scan_backward_cuda(
    torch::Tensor grad_states,
    torch::Tensor grad_prior_states,
    torch::Tensor grad_final_state,
    torch::Tensor states,
    torch::Tensor decay,
    torch::Tensor initial_state);

std::vector<torch::Tensor> causal_machine_latent_prior_scan_backward_cuda(
    torch::Tensor grad_prior_states,
    torch::Tensor grad_final_state,
    torch::Tensor prior_states,
    torch::Tensor decay,
    torch::Tensor initial_state);

std::vector<torch::Tensor> causal_machine_latent_replace_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor prior_logits,
    torch::Tensor transition_context,
    torch::Tensor token_gate,
    torch::Tensor pred_scale);

std::vector<torch::Tensor> causal_machine_latent_replace_backward_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_prior_log_beliefs,
    torch::Tensor prior_logits,
    torch::Tensor transition_context,
    torch::Tensor token_gate,
    torch::Tensor pred_scale,
    torch::Tensor beliefs,
    torch::Tensor prior_log_beliefs);

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

void check_matching_activation_dtype(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name,
    const char* reference_name) {
    TORCH_CHECK(
        tensor.scalar_type() == reference.scalar_type(),
        name,
        " must match ",
        reference_name,
        " dtype"
    );
}

void check_same_cuda_device(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name) {
    TORCH_CHECK(
        tensor.get_device() == reference.get_device(),
        name,
        " must be on the same CUDA device as drive/states"
    );
}

void check_shapes(
    const torch::Tensor& drive,
    const torch::Tensor& decay,
    const torch::Tensor& initial_state) {
    TORCH_CHECK(drive.dim() == 3, "drive must have shape [B, L, R]");
    TORCH_CHECK(decay.dim() == 1, "decay must have shape [R]");
    TORCH_CHECK(initial_state.dim() == 2, "initial_state must have shape [B, R]");
    TORCH_CHECK(drive.size(2) == decay.size(0), "drive and decay rank must match");
    TORCH_CHECK(drive.size(0) == initial_state.size(0), "drive and initial_state batch must match");
    TORCH_CHECK(drive.size(2) == initial_state.size(1), "drive and initial_state rank must match");
}

void check_replace_shapes(
    const torch::Tensor& local_logits,
    const torch::Tensor& prior_logits,
    const torch::Tensor& transition_context,
    const torch::Tensor& token_gate,
    const torch::Tensor& pred_scale) {
    TORCH_CHECK(local_logits.dim() == 3, "local_logits must have shape [B, L, S]");
    TORCH_CHECK(prior_logits.sizes() == local_logits.sizes(), "prior_logits must match local_logits shape");
    TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits shape");
    TORCH_CHECK(token_gate.dim() == 3, "token_gate must have shape [B, L, 1]");
    TORCH_CHECK(pred_scale.dim() == 3, "pred_scale must have shape [B, L, 1]");
    TORCH_CHECK(token_gate.size(0) == local_logits.size(0), "token_gate batch must match local_logits");
    TORCH_CHECK(token_gate.size(1) == local_logits.size(1), "token_gate sequence length must match local_logits");
    TORCH_CHECK(token_gate.size(2) == 1, "token_gate trailing dim must be 1");
    TORCH_CHECK(pred_scale.size(0) == local_logits.size(0), "pred_scale batch must match local_logits");
    TORCH_CHECK(pred_scale.size(1) == local_logits.size(1), "pred_scale sequence length must match local_logits");
    TORCH_CHECK(pred_scale.size(2) == 1, "pred_scale trailing dim must be 1");
}

void check_finite_decay(const torch::Tensor& decay, const char* name) {
    auto decay_cpu = decay.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    const float* decay_ptr = decay_cpu.data_ptr<float>();
    const int64_t rank_dim = decay_cpu.size(0);
    for (int64_t i = 0; i < rank_dim; ++i) {
        TORCH_CHECK(
            std::isfinite(static_cast<double>(decay_ptr[i])),
            name,
            " must be finite"
        );
    }
}

void check_positive_finite_decay(const torch::Tensor& decay, const char* name) {
    auto decay_cpu = decay.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    const float* decay_ptr = decay_cpu.data_ptr<float>();
    const int64_t rank_dim = decay_cpu.size(0);
    for (int64_t i = 0; i < rank_dim; ++i) {
        TORCH_CHECK(
            std::isfinite(static_cast<double>(decay_ptr[i])) && decay_ptr[i] > 0.0f,
            name,
            " must be strictly positive and finite for prior-only latent scan"
        );
    }
}

void check_same_cuda_devices(
    const torch::Tensor& drive_or_states,
    const torch::Tensor& decay,
    const torch::Tensor& initial_state) {
    check_same_cuda_device(decay, drive_or_states, "decay");
    check_same_cuda_device(initial_state, drive_or_states, "initial_state");
}

}  // namespace

std::vector<torch::Tensor> causal_machine_latent_scan_forward(
    torch::Tensor drive,
    torch::Tensor decay,
    torch::Tensor initial_state) {
    check_cuda_activation(drive, "drive");
    check_cuda_activation(decay, "decay");
    check_cuda_activation(initial_state, "initial_state");
    check_shapes(drive, decay, initial_state);
    check_finite_decay(decay, "decay");
    check_same_cuda_devices(drive, decay, initial_state);
    check_matching_activation_dtype(decay, drive, "decay", "drive");
    TORCH_CHECK(initial_state.scalar_type() == drive.scalar_type(), "initial_state must match drive dtype");
    return causal_machine_latent_scan_forward_cuda(drive, decay, initial_state);
}

std::vector<torch::Tensor> causal_machine_latent_prior_scan_forward(
    torch::Tensor drive,
    torch::Tensor decay,
    torch::Tensor initial_state) {
    check_cuda_activation(drive, "drive");
    check_cuda_activation(decay, "decay");
    check_cuda_activation(initial_state, "initial_state");
    check_shapes(drive, decay, initial_state);
    check_positive_finite_decay(decay, "decay");
    check_same_cuda_devices(drive, decay, initial_state);
    check_matching_activation_dtype(decay, drive, "decay", "drive");
    TORCH_CHECK(initial_state.scalar_type() == drive.scalar_type(), "initial_state must match drive dtype");
    return causal_machine_latent_prior_scan_forward_cuda(drive, decay, initial_state);
}

std::vector<torch::Tensor> causal_machine_latent_scan_backward(
    torch::Tensor grad_states,
    torch::Tensor grad_prior_states,
    torch::Tensor grad_final_state,
    torch::Tensor states,
    torch::Tensor decay,
    torch::Tensor initial_state) {
    check_cuda_activation(grad_states, "grad_states");
    check_cuda_activation(grad_prior_states, "grad_prior_states");
    check_cuda_activation(grad_final_state, "grad_final_state");
    check_cuda_activation(states, "states");
    check_cuda_activation(decay, "decay");
    check_cuda_activation(initial_state, "initial_state");
    check_shapes(states, decay, initial_state);
    check_finite_decay(decay, "decay");
    check_same_cuda_devices(states, decay, initial_state);
    check_same_cuda_device(grad_states, states, "grad_states");
    check_same_cuda_device(grad_prior_states, states, "grad_prior_states");
    check_same_cuda_device(grad_final_state, states, "grad_final_state");
    TORCH_CHECK(grad_states.sizes() == states.sizes(), "grad_states must match states shape");
    TORCH_CHECK(grad_prior_states.sizes() == states.sizes(), "grad_prior_states must match states shape");
    TORCH_CHECK(grad_final_state.sizes() == initial_state.sizes(), "grad_final_state must match initial_state shape");
    TORCH_CHECK(grad_states.scalar_type() == states.scalar_type(), "grad_states must match states dtype");
    TORCH_CHECK(grad_prior_states.scalar_type() == states.scalar_type(), "grad_prior_states must match states dtype");
    TORCH_CHECK(grad_final_state.scalar_type() == states.scalar_type(), "grad_final_state must match states dtype");
    check_matching_activation_dtype(decay, states, "decay", "states");
    TORCH_CHECK(initial_state.scalar_type() == states.scalar_type(), "initial_state must match states dtype");
    return causal_machine_latent_scan_backward_cuda(
        grad_states,
        grad_prior_states,
        grad_final_state,
        states,
        decay,
        initial_state);
}

std::vector<torch::Tensor> causal_machine_latent_prior_scan_backward(
    torch::Tensor grad_prior_states,
    torch::Tensor grad_final_state,
    torch::Tensor prior_states,
    torch::Tensor decay,
    torch::Tensor initial_state) {
    check_cuda_activation(grad_prior_states, "grad_prior_states");
    check_cuda_activation(grad_final_state, "grad_final_state");
    check_cuda_activation(prior_states, "prior_states");
    check_cuda_activation(decay, "decay");
    check_cuda_activation(initial_state, "initial_state");
    check_shapes(prior_states, decay, initial_state);
    check_positive_finite_decay(decay, "decay");
    check_same_cuda_devices(prior_states, decay, initial_state);
    check_same_cuda_device(grad_prior_states, prior_states, "grad_prior_states");
    check_same_cuda_device(grad_final_state, prior_states, "grad_final_state");
    TORCH_CHECK(grad_prior_states.sizes() == prior_states.sizes(), "grad_prior_states must match prior_states shape");
    TORCH_CHECK(grad_final_state.sizes() == initial_state.sizes(), "grad_final_state must match initial_state shape");
    TORCH_CHECK(grad_prior_states.scalar_type() == prior_states.scalar_type(), "grad_prior_states must match prior_states dtype");
    TORCH_CHECK(grad_final_state.scalar_type() == prior_states.scalar_type(), "grad_final_state must match prior_states dtype");
    check_matching_activation_dtype(decay, prior_states, "decay", "prior_states");
    TORCH_CHECK(initial_state.scalar_type() == prior_states.scalar_type(), "initial_state must match prior_states dtype");
    return causal_machine_latent_prior_scan_backward_cuda(
        grad_prior_states,
        grad_final_state,
        prior_states,
        decay,
        initial_state);
}

std::vector<torch::Tensor> causal_machine_latent_replace_forward(
    torch::Tensor local_logits,
    torch::Tensor prior_logits,
    torch::Tensor transition_context,
    torch::Tensor token_gate,
    torch::Tensor pred_scale) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_activation(prior_logits, "prior_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(token_gate, "token_gate");
    check_cuda_activation(pred_scale, "pred_scale");
    check_replace_shapes(local_logits, prior_logits, transition_context, token_gate, pred_scale);
    check_same_cuda_device(prior_logits, local_logits, "prior_logits");
    check_same_cuda_device(transition_context, local_logits, "transition_context");
    check_same_cuda_device(token_gate, local_logits, "token_gate");
    check_same_cuda_device(pred_scale, local_logits, "pred_scale");
    check_matching_activation_dtype(prior_logits, local_logits, "prior_logits", "local_logits");
    check_matching_activation_dtype(transition_context, local_logits, "transition_context", "local_logits");
    check_matching_activation_dtype(token_gate, local_logits, "token_gate", "local_logits");
    check_matching_activation_dtype(pred_scale, local_logits, "pred_scale", "local_logits");
    return causal_machine_latent_replace_forward_cuda(
        local_logits,
        prior_logits,
        transition_context,
        token_gate,
        pred_scale);
}

std::vector<torch::Tensor> causal_machine_latent_replace_backward(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_prior_log_beliefs,
    torch::Tensor prior_logits,
    torch::Tensor transition_context,
    torch::Tensor token_gate,
    torch::Tensor pred_scale,
    torch::Tensor beliefs,
    torch::Tensor prior_log_beliefs) {
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_prior_log_beliefs, "grad_prior_log_beliefs");
    check_cuda_activation(prior_logits, "prior_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(token_gate, "token_gate");
    check_cuda_activation(pred_scale, "pred_scale");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_activation(prior_log_beliefs, "prior_log_beliefs");
    check_replace_shapes(beliefs, prior_logits, transition_context, token_gate, pred_scale);
    TORCH_CHECK(grad_beliefs.sizes() == beliefs.sizes(), "grad_beliefs must match beliefs shape");
    TORCH_CHECK(grad_prior_log_beliefs.sizes() == beliefs.sizes(), "grad_prior_log_beliefs must match beliefs shape");
    TORCH_CHECK(prior_log_beliefs.sizes() == beliefs.sizes(), "prior_log_beliefs must match beliefs shape");
    check_same_cuda_device(grad_beliefs, beliefs, "grad_beliefs");
    check_same_cuda_device(grad_prior_log_beliefs, beliefs, "grad_prior_log_beliefs");
    check_same_cuda_device(prior_logits, beliefs, "prior_logits");
    check_same_cuda_device(transition_context, beliefs, "transition_context");
    check_same_cuda_device(token_gate, beliefs, "token_gate");
    check_same_cuda_device(pred_scale, beliefs, "pred_scale");
    check_same_cuda_device(prior_log_beliefs, beliefs, "prior_log_beliefs");
    check_matching_activation_dtype(grad_beliefs, beliefs, "grad_beliefs", "beliefs");
    check_matching_activation_dtype(grad_prior_log_beliefs, beliefs, "grad_prior_log_beliefs", "beliefs");
    check_matching_activation_dtype(prior_logits, beliefs, "prior_logits", "beliefs");
    check_matching_activation_dtype(transition_context, beliefs, "transition_context", "beliefs");
    check_matching_activation_dtype(token_gate, beliefs, "token_gate", "beliefs");
    check_matching_activation_dtype(pred_scale, beliefs, "pred_scale", "beliefs");
    check_matching_activation_dtype(prior_log_beliefs, beliefs, "prior_log_beliefs", "beliefs");
    return causal_machine_latent_replace_backward_cuda(
        grad_beliefs,
        grad_prior_log_beliefs,
        prior_logits,
        transition_context,
        token_gate,
        pred_scale,
        beliefs,
        prior_log_beliefs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causal_machine_latent_scan_forward, "Causal machine latent scan forward (CUDA)");
    m.def("forward_prior_only", &causal_machine_latent_prior_scan_forward, "Causal machine latent prior scan forward (CUDA)");
    m.def("backward", &causal_machine_latent_scan_backward, "Causal machine latent scan backward (CUDA)");
    m.def("backward_prior_only", &causal_machine_latent_prior_scan_backward, "Causal machine latent prior scan backward (CUDA)");
    m.def("forward_replace", &causal_machine_latent_replace_forward, "Causal machine latent-replace projection forward (CUDA)");
    m.def("backward_replace", &causal_machine_latent_replace_backward, "Causal machine latent-replace projection backward (CUDA)");
}
