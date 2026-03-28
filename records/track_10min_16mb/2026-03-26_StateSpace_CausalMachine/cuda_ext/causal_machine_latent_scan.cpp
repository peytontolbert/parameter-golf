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
    check_cuda_float32(decay, "decay");
    check_cuda_activation(initial_state, "initial_state");
    check_shapes(drive, decay, initial_state);
    check_same_cuda_devices(drive, decay, initial_state);
    TORCH_CHECK(initial_state.scalar_type() == drive.scalar_type(), "initial_state must match drive dtype");
    return causal_machine_latent_scan_forward_cuda(drive, decay, initial_state);
}

std::vector<torch::Tensor> causal_machine_latent_prior_scan_forward(
    torch::Tensor drive,
    torch::Tensor decay,
    torch::Tensor initial_state) {
    check_cuda_activation(drive, "drive");
    check_cuda_float32(decay, "decay");
    check_cuda_activation(initial_state, "initial_state");
    check_shapes(drive, decay, initial_state);
    check_same_cuda_devices(drive, decay, initial_state);
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
    check_cuda_float32(decay, "decay");
    check_cuda_activation(initial_state, "initial_state");
    check_shapes(states, decay, initial_state);
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
    check_cuda_float32(decay, "decay");
    check_cuda_activation(initial_state, "initial_state");
    check_shapes(prior_states, decay, initial_state);
    check_same_cuda_devices(prior_states, decay, initial_state);
    check_same_cuda_device(grad_prior_states, prior_states, "grad_prior_states");
    check_same_cuda_device(grad_final_state, prior_states, "grad_final_state");
    TORCH_CHECK(grad_prior_states.sizes() == prior_states.sizes(), "grad_prior_states must match prior_states shape");
    TORCH_CHECK(grad_final_state.sizes() == initial_state.sizes(), "grad_final_state must match initial_state shape");
    TORCH_CHECK(grad_prior_states.scalar_type() == prior_states.scalar_type(), "grad_prior_states must match prior_states dtype");
    TORCH_CHECK(grad_final_state.scalar_type() == prior_states.scalar_type(), "grad_final_state must match prior_states dtype");
    TORCH_CHECK(initial_state.scalar_type() == prior_states.scalar_type(), "initial_state must match prior_states dtype");
    return causal_machine_latent_prior_scan_backward_cuda(
        grad_prior_states,
        grad_final_state,
        prior_states,
        decay,
        initial_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causal_machine_latent_scan_forward, "Causal machine latent scan forward (CUDA)");
    m.def("forward_prior_only", &causal_machine_latent_prior_scan_forward, "Causal machine latent prior scan forward (CUDA)");
    m.def("backward", &causal_machine_latent_scan_backward, "Causal machine latent scan backward (CUDA)");
    m.def("backward_prior_only", &causal_machine_latent_prior_scan_backward, "Causal machine latent prior scan backward (CUDA)");
}
