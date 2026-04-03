#include <torch/extension.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

std::vector<torch::Tensor> causal_machine_scan_forward_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity());

torch::Tensor causal_machine_decode_belief_projection_cuda(
    torch::Tensor state_log_beliefs,
    torch::Tensor belief_out_weight);

torch::Tensor causal_machine_decode_belief_projection_backward_input_cuda(
    torch::Tensor grad_output,
    torch::Tensor state_log_beliefs,
    torch::Tensor belief_out_weight);

torch::Tensor causal_machine_decode_belief_projection_backward_weight_cuda(
    torch::Tensor grad_output,
    torch::Tensor state_log_beliefs);

std::vector<torch::Tensor> causal_machine_scan_forward_logits_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity());

std::vector<torch::Tensor> causal_machine_scan_forward_masked_logits_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);

std::vector<torch::Tensor> causal_machine_scan_backward_masked_logits_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);

std::vector<torch::Tensor> causal_machine_scan_forward_sparse_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_blocks,
    torch::Tensor block_row_ptr,
    torch::Tensor block_col_idx,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size);

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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size);

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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size);

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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size);

std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks_cuda(
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size);

std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks_int8_cuda(
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size);

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
    int64_t block_size);

std::vector<torch::Tensor> causal_machine_scan_forward_composable_logits_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size);

std::vector<torch::Tensor> causal_machine_scan_forward_fp8_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    int64_t chunk_size);

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_forward_tiled_quantized_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_forward_tiled_quantized_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor filtered_value_cache,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_forward_tiled_fp8_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_forward_tiled_fp8_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor filtered_value_cache,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    torch::Tensor work_queue_counter,
    torch::Tensor filtered_value_cache,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_bound_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    py::dict workspace,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_kernel_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_kernel_workspace_cuda(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    torch::Tensor work_queue_counter,
    torch::Tensor masked_transition_tile_cache,
    torch::Tensor filtered_value_cache,
    torch::Tensor row_sums,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity(),
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);

std::vector<torch::Tensor> causal_machine_scan_backward_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity());

std::vector<torch::Tensor> causal_machine_scan_backward_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max,
    torch::Tensor grad_transition_source_per_batch,
    torch::Tensor grad_transition_dest_per_batch,
    torch::Tensor grad_transition_stay_per_batch,
    torch::Tensor grad_transition_gate_per_batch);

std::vector<torch::Tensor> causal_machine_scan_backward_logits_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min = -std::numeric_limits<double>::infinity(),
    double score_clamp_max = std::numeric_limits<double>::infinity());

std::vector<torch::Tensor> causal_machine_scan_backward_logits_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max,
    torch::Tensor grad_transition_source_per_batch,
    torch::Tensor grad_transition_dest_per_batch,
    torch::Tensor grad_transition_stay_per_batch,
    torch::Tensor grad_transition_gate_per_batch);

std::vector<torch::Tensor> causal_machine_scan_backward_composable_logits_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size);

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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    int64_t chunk_size);

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_kernel_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
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
    torch::Tensor transition_gate,
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
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
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
    torch::Tensor transition_gate,
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
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_kernel_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
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
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_kernel_bound_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    py::dict workspace,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);
std::vector<torch::Tensor> causal_machine_scan_backward_masked_logits_workspace_cuda(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    torch::Tensor work_queue_counter,
    torch::Tensor masked_transition_tile_cache,
    torch::Tensor row_sums,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold = -std::numeric_limits<double>::infinity(),
    int64_t score_topk = 0);

std::vector<torch::Tensor> causal_machine_scan_pack_int8_cuda(torch::Tensor table);
std::vector<torch::Tensor> causal_machine_scan_pack_fp8_e4m3_cuda(torch::Tensor table);
std::vector<torch::Tensor> causal_machine_scan_pack_fp8_e5m2_cuda(torch::Tensor table);
torch::Tensor causal_machine_scan_unpack_int8_cuda(torch::Tensor packed, torch::Tensor scales);
torch::Tensor causal_machine_scan_unpack_fp8_e4m3_cuda(torch::Tensor packed, torch::Tensor scales);
torch::Tensor causal_machine_scan_unpack_fp8_e5m2_cuda(torch::Tensor packed, torch::Tensor scales);

int64_t causal_machine_scan_single_launch_max_seq_len_cuda();
int64_t causal_machine_scan_forward_chunk_shared_bytes_cuda(int64_t transition_rank);
int64_t causal_machine_scan_forward_tiled_chunk_shared_bytes_cuda(int64_t num_states, int64_t split_size, int64_t tile_size);
int64_t causal_machine_scan_backward_tiled_chunk_shared_bytes_cuda(int64_t num_states, int64_t split_size, int64_t tile_size);
int64_t causal_machine_scan_forward_masked_tiled_chunk_shared_bytes_cuda(int64_t num_states, int64_t tile_size);
int64_t causal_machine_scan_backward_masked_tiled_chunk_shared_bytes_cuda(int64_t num_states, int64_t tile_size);
int64_t causal_machine_scan_backward_chunk_shared_bytes_cuda(int64_t num_states, int64_t transition_rank, bool direct_grad_reduce);
bool causal_machine_scan_can_use_direct_grad_reduce_cuda(int64_t device_index, int64_t num_states, int64_t transition_rank);
int64_t small_state_direct_staging_worker_blocks(int64_t device_index, int64_t batch_size);
bool causal_machine_scan_can_use_tiled_forward_kernel_cuda(int64_t device_index, int64_t num_states, int64_t tile_size, int64_t split_size);
bool causal_machine_scan_can_use_tiled_backward_kernel_cuda(int64_t device_index, int64_t num_states, int64_t tile_size, int64_t split_size);
bool causal_machine_scan_can_use_masked_tiled_forward_kernel_cuda(int64_t device_index, int64_t num_states, int64_t tile_size);
bool causal_machine_scan_can_use_masked_tiled_backward_kernel_cuda(int64_t device_index, int64_t num_states, int64_t tile_size);
int64_t causal_machine_scan_cached_max_optin_bytes_cuda(int64_t device_index);
int64_t causal_machine_scan_cached_capability_major_cuda(int64_t device_index);
int64_t causal_machine_scan_cached_capability_minor_cuda(int64_t device_index);
int64_t causal_machine_scan_cached_sm_count_cuda(int64_t device_index);
int64_t causal_machine_scan_cached_l2_cache_size_cuda(int64_t device_index);
int64_t causal_machine_scan_cached_persisting_l2_cache_max_size_cuda(int64_t device_index);
bool causal_machine_scan_supports_persisting_l2_window_cuda(int64_t device_index);
int64_t causal_machine_scan_cached_total_global_mem_cuda(int64_t device_index);
int64_t causal_machine_scan_persistent_worker_blocks_cuda(int64_t device_index, int64_t total_batches);
int64_t causal_machine_scan_preferred_load_bytes_cuda(int64_t num_states, int64_t tile_size, int64_t split_size);
int64_t causal_machine_scan_elements_per_load_cuda(int64_t num_states, int64_t tile_size, int64_t split_size);
bool causal_machine_scan_can_use_vectorized_io_cuda(int64_t num_states, int64_t tile_size, int64_t split_size);
bool causal_machine_scan_can_use_async_memcpy_cuda(int64_t device_index);
bool causal_machine_scan_can_use_tensor_cores_cuda(int64_t device_index);
bool causal_machine_scan_can_use_half2_path_cuda(int64_t device_index);
bool causal_machine_scan_can_use_wmma_cuda(int64_t device_index);
bool causal_machine_scan_can_use_tma_cuda(int64_t device_index);
bool causal_machine_scan_can_use_wgmma_cuda(int64_t device_index);
std::vector<int64_t> causal_machine_scan_describe_tiled_forward_runtime_cuda(
    int64_t device_index,
    int64_t total_batches,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t split_size,
    int64_t seq_len);
std::vector<int64_t> causal_machine_scan_describe_masked_tiled_forward_runtime_cuda(
    int64_t device_index,
    int64_t total_batches,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t seq_len);
std::vector<int64_t> causal_machine_scan_describe_tiled_backward_runtime_cuda(
    int64_t device_index,
    int64_t total_batches,
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t split_size,
    int64_t seq_len);
std::vector<int64_t> causal_machine_scan_describe_masked_tiled_backward_runtime_cuda(
    int64_t device_index,
    int64_t total_batches,
    int64_t num_states,
    int64_t tile_size,
    int64_t seq_len);
void causal_machine_scan_record_paged_step_tensor_cuda(
    torch::Tensor paged_values,
    torch::Tensor values,
    int64_t num_updates);
void causal_machine_scan_record_paged_step_tensor_from_lengths_cuda(
    torch::Tensor paged_values,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor values);
void causal_machine_scan_record_paged_sequence_tensor_cuda(
    torch::Tensor paged_values,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor values,
    int64_t num_updates);
void causal_machine_scan_increment_paged_lengths_cuda(
    torch::Tensor paged_lengths,
    int64_t delta,
    int64_t capacity);
void causal_machine_scan_read_paged_latest_tensor_cuda(
    torch::Tensor paged_values,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor values);
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
    torch::Tensor transition_gate,
    double score_clamp_min,
    double score_clamp_max);
std::vector<torch::Tensor> causal_machine_scan_paged_step_dense_128_rank16_cuda(
    torch::Tensor paged_log_beliefs,
    torch::Tensor paged_latent_states,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_gate,
    double score_clamp_min,
    double score_clamp_max);
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
    torch::Tensor transition_gate);
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
    torch::Tensor transition_gate,
    int64_t fp8_format);
void causal_machine_scan_reorder_paged_cache_cuda(
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor beam_indices);

namespace {

torch::Tensor make_cuda_workspace_tensor(
    int64_t device_index,
    at::ScalarType dtype,
    at::IntArrayRef sizes,
    bool zero_init = false);

struct DenseBackwardWorkspace {
    torch::Tensor grad_transition_source_per_batch;
    torch::Tensor grad_transition_dest_per_batch;
    torch::Tensor grad_transition_stay_per_batch;
    torch::Tensor grad_transition_gate_per_batch;
};

DenseBackwardWorkspace get_dense_backward_workspace(
    const torch::Tensor& beliefs,
    int64_t num_states,
    int64_t transition_rank) {
    static std::mutex workspace_mutex;
    static std::unordered_map<std::string, DenseBackwardWorkspace> workspace_cache;

    const int64_t device_index = beliefs.get_device();
    const int64_t batch_size = beliefs.size(0);
    const bool direct_small_rank_grad = causal_machine_scan_can_use_direct_grad_reduce_cuda(
        device_index,
        num_states,
        transition_rank);
    const int64_t staging_blocks = direct_small_rank_grad
        ? small_state_direct_staging_worker_blocks(device_index, batch_size)
        : batch_size;
    const std::string cache_key =
        std::to_string(device_index) + ":" +
        std::to_string(num_states) + ":" +
        std::to_string(transition_rank) + ":" +
        std::to_string(batch_size) + ":" +
        (direct_small_rank_grad ? "direct" : "per_batch");

    std::lock_guard<std::mutex> lock(workspace_mutex);
    auto& workspace = workspace_cache[cache_key];
    if (!workspace.grad_transition_source_per_batch.defined()) {
        workspace.grad_transition_source_per_batch = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            {staging_blocks, num_states, transition_rank});
        workspace.grad_transition_dest_per_batch = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            {staging_blocks, transition_rank, num_states});
        workspace.grad_transition_stay_per_batch = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            {staging_blocks, num_states});
        workspace.grad_transition_gate_per_batch = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            {staging_blocks});
    }
    return workspace;
}

constexpr int kSpecializedNumStates = 128;
constexpr int kMinSpecializedNumStates = 64;
constexpr int kMidSpecializedNumStates = 96;

std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_kernel_workspace(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
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

std::vector<torch::Tensor> causal_machine_scan_forward_sparse_logits_fused(
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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size);

std::vector<torch::Tensor> causal_machine_scan_backward_sparse_logits_fused(
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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size);

bool is_supported_specialized_num_states(int64_t num_states) {
    return num_states > 0 && num_states <= kSpecializedNumStates;
}

int64_t runtime_value_at(const std::vector<int64_t>& values, size_t index) {
    return index < values.size() ? values[index] : 0;
}

std::vector<torch::Tensor> causal_machine_scan_build_sparse_metadata_from_runtime(
    int64_t num_states,
    int64_t padded_states,
    int64_t block_size,
    int64_t local_transition_window,
    torch::Tensor transition_mask,
    torch::Tensor runtime_block_mask);

int64_t ceil_div_int64(int64_t numerator, int64_t denominator) {
    TORCH_CHECK(denominator > 0, "denominator must be positive");
    return (numerator + denominator - 1) / denominator;
}

std::pair<int64_t, int64_t> choose_large_state_tiled_geometry(
    int64_t device_index,
    int64_t num_states,
    int64_t transition_rank,
    bool backward) {
    int64_t tile_size = std::min<int64_t>(num_states, 128);
    int64_t split_size = std::min<int64_t>(transition_rank, 64);
    auto can_use = [&](int64_t tile, int64_t split) {
        return backward
            ? causal_machine_scan_can_use_tiled_backward_kernel_cuda(device_index, num_states, tile, split)
            : causal_machine_scan_can_use_tiled_forward_kernel_cuda(device_index, num_states, tile, split);
    };
    while ((tile_size > 32 || split_size > 8) && !can_use(tile_size, split_size)) {
        if (split_size > 8) {
            split_size = std::max<int64_t>(8, split_size / 2);
        } else {
            tile_size = std::max<int64_t>(32, tile_size / 2);
        }
    }
    TORCH_CHECK(
        can_use(tile_size, split_size),
        "unable to find a valid tiled runtime geometry for large-state low-precision structured scan");
    return {tile_size, split_size};
}

py::dict causal_machine_scan_describe_runtime_config(
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_states,
    int64_t transition_rank,
    int64_t chunk_size,
    int64_t device_index,
    bool backward) {
    TORCH_CHECK(batch_size >= 0, "batch_size must be non-negative");
    TORCH_CHECK(seq_len >= 0, "seq_len must be non-negative");
    TORCH_CHECK(
        is_supported_specialized_num_states(num_states),
        "num_states must be in [1, ",
        kSpecializedNumStates,
        "] for the small-state CUDA fast path"
    );
    TORCH_CHECK(
        transition_rank > 0 && transition_rank <= num_states,
        "transition_rank must be in [1, ",
        num_states,
        "]"
    );
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    TORCH_CHECK(device_index >= 0, "device_index must be non-negative");
    const int64_t single_launch_max_seq_len = causal_machine_scan_single_launch_max_seq_len_cuda();
    const bool use_single_launch = true;
    const int64_t launch_chunk_size = seq_len == 0 ? 0 : std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    const int64_t num_launches = seq_len == 0 ? 0 : 1;
    const int64_t num_sequence_tiles = launch_chunk_size == 0 ? 0 : ceil_div_int64(seq_len, launch_chunk_size);
    const bool direct_grad_reduce = backward
        ? causal_machine_scan_can_use_direct_grad_reduce_cuda(device_index, num_states, transition_rank)
        : false;
    const int64_t shared_bytes = backward
        ? causal_machine_scan_backward_chunk_shared_bytes_cuda(num_states, transition_rank, direct_grad_reduce)
        : causal_machine_scan_forward_chunk_shared_bytes_cuda(transition_rank);
    const int64_t worker_blocks = batch_size == 0 || seq_len == 0
        ? 0
        : causal_machine_scan_persistent_worker_blocks_cuda(device_index, batch_size);
    py::dict info;
    info["batch_size"] = batch_size;
    info["num_states"] = num_states;
    info["tile_size"] = num_states;
    info["block_threads"] = num_states;
    info["preferred_num_states"] = py::make_tuple(
        kMinSpecializedNumStates,
        kMidSpecializedNumStates,
        kSpecializedNumStates);
    info["supported_num_states_min"] = 1;
    info["supported_num_states_max"] = kSpecializedNumStates;
    info["transition_rank"] = transition_rank;
    info["requested_chunk_size"] = chunk_size;
    info["launch_chunk_size"] = launch_chunk_size;
    info["num_launches"] = num_launches;
    info["num_sequence_tiles"] = num_sequence_tiles;
    info["worker_blocks"] = worker_blocks;
    info["grid_x"] = worker_blocks;
    info["single_launch_max_seq_len"] = single_launch_max_seq_len;
    info["use_single_launch"] = use_single_launch;
    info["persistent_device_loop"] = (seq_len > 0);
    info["device_work_queue"] = (seq_len > 0 && batch_size > 0);
    info["reverse_launch_order"] = backward;
    info["uses_persisting_l2_window"] = (seq_len > 0 && batch_size > 0)
        && causal_machine_scan_supports_persisting_l2_window_cuda(device_index);
    info["shared_bytes"] = shared_bytes;
    info["max_dynamic_smem_bytes"] = causal_machine_scan_cached_max_optin_bytes_cuda(device_index);
    info["direct_grad_reduce"] = direct_grad_reduce;
    info["optimized_rank"] = (
        transition_rank == 8
        || transition_rank == 16
        || transition_rank == 32
        || transition_rank == 64
        || transition_rank == kSpecializedNumStates
    );
    return info;
}

py::dict causal_machine_scan_describe_tiled_runtime_config(
    int64_t num_states,
    int64_t transition_rank,
    int64_t seq_len,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    int64_t batch_size,
    int64_t device_index) {
    TORCH_CHECK(
        num_states > kSpecializedNumStates,
        "tiled runtime config expects num_states > ",
        kSpecializedNumStates);
    TORCH_CHECK(transition_rank > 0 && transition_rank <= num_states, "transition_rank must be in [1, num_states]");
    TORCH_CHECK(seq_len >= 0, "seq_len must be non-negative");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(split_size > 0, "split_size must be positive");
    TORCH_CHECK(batch_size >= 0, "batch_size must be non-negative");
    TORCH_CHECK(device_index >= 0, "device_index must be non-negative");
    int64_t block_threads = 32;
    const int64_t required_threads = std::max(tile_size, split_size);
    while (block_threads < required_threads && block_threads < 256) {
        block_threads <<= 1;
    }
    py::dict info;
    const int64_t shared_bytes = causal_machine_scan_forward_tiled_chunk_shared_bytes_cuda(
        num_states,
        split_size,
        tile_size);
    const int64_t backward_shared_bytes = causal_machine_scan_backward_tiled_chunk_shared_bytes_cuda(
        num_states,
        split_size,
        tile_size);
    const int64_t launch_chunk_size = seq_len == 0 ? 0 : std::min<int64_t>(std::max<int64_t>(chunk_size, 1), seq_len);
    const int64_t num_launches = seq_len == 0 ? 0 : 1;
    const bool use_single_launch = true;
    const bool custom_forward_kernel_supported = causal_machine_scan_can_use_tiled_forward_kernel_cuda(
        device_index,
        num_states,
        tile_size,
        split_size);
    const int64_t capability_major = causal_machine_scan_cached_capability_major_cuda(device_index);
    const bool hopper_tma_capable = capability_major >= 9 && causal_machine_scan_can_use_tma_cuda(device_index);
    const bool hopper_wgmma_capable = capability_major >= 9 && causal_machine_scan_can_use_wgmma_cuda(device_index);
    const bool async_pipeline_forward_supported = custom_forward_kernel_supported && capability_major >= 8;
    const std::string forward_kernel_family = custom_forward_kernel_supported
        ? (capability_major >= 9
            ? (hopper_tma_capable ? "sm90_tma_async3_tiled_custom" : "sm90_async_pipeline_tiled_custom")
            : "sm80_async_pipeline_tiled_custom")
        : "split_combine_fallback";
    const bool custom_backward_kernel_smem_supported = causal_machine_scan_can_use_tiled_backward_kernel_cuda(
        device_index,
        num_states,
        tile_size,
        split_size);
    const bool custom_backward_kernel_scheduler_supported = (seq_len == 0) || (num_launches == 1);
    const auto forward_runtime = causal_machine_scan_describe_tiled_forward_runtime_cuda(
        device_index,
        batch_size,
        num_states,
        transition_rank,
        tile_size,
        split_size,
        seq_len);
    const auto backward_runtime = causal_machine_scan_describe_tiled_backward_runtime_cuda(
        device_index,
        batch_size,
        num_states,
        transition_rank,
        tile_size,
        split_size,
        seq_len);
    const int64_t forward_preferred_load_bytes = runtime_value_at(forward_runtime, 3);
    const int64_t forward_elements_per_load = runtime_value_at(forward_runtime, 4);
    const bool forward_vectorized_io = bool(runtime_value_at(forward_runtime, 5));
    const bool forward_async_copy_path = bool(runtime_value_at(forward_runtime, 6));
    const bool forward_tensor_core_math = bool(runtime_value_at(forward_runtime, 7));
    const bool forward_wgmma_math = hopper_wgmma_capable && forward_tensor_core_math && block_threads >= 128;
    const int64_t forward_async_pipeline_stages = capability_major >= 9 ? 3 : 2;
    const bool forward_persisting_l2 = bool(runtime_value_at(forward_runtime, 16));
    const bool backward_vectorized_io = bool(runtime_value_at(backward_runtime, 17));
    const bool backward_async_copy_path = bool(runtime_value_at(backward_runtime, 18));
    const bool backward_tensor_core_math = bool(runtime_value_at(backward_runtime, 19));
    const bool backward_wgmma_math = hopper_wgmma_capable && backward_tensor_core_math && block_threads >= 128;
    const bool backward_persisting_l2 = bool(runtime_value_at(backward_runtime, 22));
    info["num_states"] = num_states;
    info["transition_rank"] = transition_rank;
    info["seq_len"] = seq_len;
    info["requested_chunk_size"] = chunk_size;
    info["launch_chunk_size"] = launch_chunk_size;
    info["num_launches"] = num_launches;
    info["use_single_launch"] = use_single_launch;
    info["tile_size"] = tile_size;
    info["split_size"] = split_size;
    info["block_threads"] = block_threads;
    info["shared_bytes"] = shared_bytes;
    info["backward_shared_bytes"] = backward_shared_bytes;
    info["max_dynamic_smem_bytes"] = causal_machine_scan_cached_max_optin_bytes_cuda(device_index);
    info["persistent_blocks"] = causal_machine_scan_persistent_worker_blocks_cuda(device_index, batch_size);
    info["forward_launch_worker_blocks"] = runtime_value_at(forward_runtime, 0);
    info["forward_runtime_shared_bytes"] = runtime_value_at(forward_runtime, 1);
    info["forward_block_threads"] = runtime_value_at(forward_runtime, 2);
    info["preferred_load_bytes"] = forward_preferred_load_bytes;
    info["elements_per_load"] = forward_elements_per_load;
    info["can_use_vectorized_io"] = forward_vectorized_io;
    info["forward_vectorized_io"] = forward_vectorized_io;
    info["async_pipeline_forward_supported"] = forward_async_copy_path;
    info["forward_async_copy_path"] = forward_async_copy_path;
    info["forward_async_pipeline_stages"] = forward_async_pipeline_stages;
    info["forward_tma_capable"] = hopper_tma_capable;
    info["forward_tma_specialized"] = hopper_tma_capable && custom_forward_kernel_supported;
    info["forward_wgmma_capable"] = hopper_wgmma_capable;
    info["forward_wgmma_kernel_implemented"] = hopper_wgmma_capable;
    info["forward_tensor_core_math_supported"] = forward_tensor_core_math;
    info["forward_active_blocks_per_sm"] = runtime_value_at(forward_runtime, 8);
    info["forward_active_warps_per_sm"] = runtime_value_at(forward_runtime, 9);
    info["forward_max_warps_per_sm"] = runtime_value_at(forward_runtime, 10);
    info["forward_estimated_occupancy_pct"] = runtime_value_at(forward_runtime, 11);
    info["forward_registers_per_thread"] = runtime_value_at(forward_runtime, 12);
    info["forward_static_smem_bytes"] = runtime_value_at(forward_runtime, 13);
    info["forward_persisting_l2_candidate_bytes"] = runtime_value_at(forward_runtime, 14);
    info["forward_persisting_l2_effective_bytes"] = runtime_value_at(forward_runtime, 15);
    info["forward_uses_persisting_l2_window"] = forward_persisting_l2;
    info["forward_estimated_bytes_moved"] = runtime_value_at(forward_runtime, 17);
    info["forward_estimated_sync_points"] = runtime_value_at(forward_runtime, 18);
    info["backward_launch_batch_size"] = runtime_value_at(backward_runtime, 0);
    info["backward_staging_worker_blocks"] = runtime_value_at(backward_runtime, 1);
    info["backward_staging_budget_bytes"] = runtime_value_at(backward_runtime, 2);
    info["backward_staging_per_worker_bytes"] = runtime_value_at(backward_runtime, 3);
    info["backward_launch_worker_blocks"] = runtime_value_at(backward_runtime, 4);
    info["free_global_mem_bytes"] = runtime_value_at(backward_runtime, 5);
    info["total_global_mem_bytes"] = runtime_value_at(backward_runtime, 6);
    info["backward_runtime_shared_bytes"] = runtime_value_at(backward_runtime, 7);
    info["backward_block_threads"] = runtime_value_at(backward_runtime, 8);
    info["backward_active_blocks_per_sm"] = runtime_value_at(backward_runtime, 9);
    info["backward_active_warps_per_sm"] = runtime_value_at(backward_runtime, 10);
    info["backward_max_warps_per_sm"] = runtime_value_at(backward_runtime, 11);
    info["backward_estimated_occupancy_pct"] = runtime_value_at(backward_runtime, 12);
    info["backward_registers_per_thread"] = runtime_value_at(backward_runtime, 13);
    info["backward_static_smem_bytes"] = runtime_value_at(backward_runtime, 14);
    info["backward_preferred_load_bytes"] = runtime_value_at(backward_runtime, 15);
    info["backward_elements_per_load"] = runtime_value_at(backward_runtime, 16);
    info["backward_vectorized_io"] = backward_vectorized_io;
    info["backward_async_copy_path"] = backward_async_copy_path;
    info["backward_async_pipeline_stages"] = forward_async_pipeline_stages;
    info["backward_tma_capable"] = hopper_tma_capable;
    info["backward_tma_specialized"] = hopper_tma_capable && custom_backward_kernel_smem_supported;
    info["backward_wgmma_capable"] = hopper_wgmma_capable;
    info["backward_wgmma_kernel_implemented"] = hopper_wgmma_capable;
    info["backward_tensor_core_math_supported"] = backward_tensor_core_math;
    info["backward_persisting_l2_candidate_bytes"] = runtime_value_at(backward_runtime, 20);
    info["backward_persisting_l2_effective_bytes"] = runtime_value_at(backward_runtime, 21);
    info["backward_uses_persisting_l2_window"] = backward_persisting_l2;
    info["backward_estimated_bytes_moved"] = runtime_value_at(backward_runtime, 23);
    info["backward_estimated_sync_points"] = runtime_value_at(backward_runtime, 24);
    info["custom_kernel_supported"] = custom_forward_kernel_supported;
    info["forward_kernel_family"] = forward_tensor_core_math
        ? (capability_major >= 9
            ? (forward_wgmma_math
                ? (hopper_tma_capable ? "sm90_tma_wgmma_async3_tiled_custom" : "sm90_wgmma_async3_tiled_custom")
                : (hopper_tma_capable ? "sm90_tma_wmma_async3_tiled_custom" : "sm90_wmma_async3_tiled_custom"))
            : "sm80_wmma_tiled_custom")
        : forward_kernel_family;
    info["forward_kernel_reason"] = custom_forward_kernel_supported
        ? (forward_tensor_core_math
            ? (capability_major >= 9
                ? (forward_wgmma_math
                    ? (hopper_tma_capable ? "custom_tiled_tma_wgmma_async3" : "custom_tiled_async_memcpy_wgmma_async3")
                    : (hopper_tma_capable ? "custom_tiled_tma_wmma_async3" : "custom_tiled_async_memcpy_wmma_async3"))
                : "custom_tiled_async_memcpy_wmma")
            : (forward_async_copy_path
                ? (hopper_tma_capable ? "custom_tiled_tma_async3" : "custom_tiled_async_memcpy")
                : "custom_tiled_shared_copy"))
        : "split_combine_fallback";
    info["custom_backward_kernel_smem_supported"] = custom_backward_kernel_smem_supported;
    info["custom_backward_kernel_scheduler_supported"] = custom_backward_kernel_scheduler_supported;
    info["custom_backward_kernel_supported"] = custom_backward_kernel_smem_supported
        && custom_backward_kernel_scheduler_supported;
    info["backward_kernel_family"] = (custom_backward_kernel_smem_supported && backward_tensor_core_math)
        ? (capability_major >= 9
            ? (backward_wgmma_math
                ? (hopper_tma_capable ? "sm90_tma_wgmma_tiled_backward_custom" : "sm90_wgmma_tiled_backward_custom")
                : (hopper_tma_capable ? "sm90_tma_wmma_tiled_backward_custom" : "sm90_wmma_tiled_backward_custom"))
            : "sm80_wmma_tiled_backward_custom")
        : "persistent_tiled_backward_custom";
    info["backward_kernel_reason"] = custom_backward_kernel_smem_supported
        ? (backward_tensor_core_math
            ? (capability_major >= 9
                ? (backward_wgmma_math
                    ? (hopper_tma_capable ? "persistent_tiled_backward_tma_wgmma" : "persistent_tiled_backward_wgmma")
                    : (hopper_tma_capable ? "persistent_tiled_backward_tma_wmma" : "persistent_tiled_backward_wmma"))
                : "persistent_tiled_backward_wmma")
            : (hopper_tma_capable ? "persistent_tiled_backward_tma_async" : "persistent_tiled_backward_async"))
        : "persistent_tiled_backward_fallback";
    info["uses_persisting_l2_window"] = forward_persisting_l2 || backward_persisting_l2;
    return info;
}

py::dict make_tiled_geometry_hints(
    int64_t num_states,
    int64_t transition_rank,
    int64_t tile_size,
    int64_t split_size,
    int64_t seq_len,
    int64_t capability_major,
    bool backward,
    bool async_shared) {
    py::dict hints;
    const int64_t row_bytes = std::max<int64_t>(tile_size, 1) * static_cast<int64_t>(sizeof(float));
    int64_t vector_bytes = 4;
    if (row_bytes % 16 == 0 && tile_size >= 64) {
        vector_bytes = 16;
    } else if (row_bytes % 8 == 0 && tile_size >= 32) {
        vector_bytes = 8;
    }
    const int64_t elements_per_load = std::max<int64_t>(1, vector_bytes / static_cast<int64_t>(sizeof(float)));
    const int64_t rank_unroll = split_size >= 128 ? 8 : (split_size >= 64 ? 4 : 2);
    const int64_t state_unroll = tile_size >= 160 ? 2 : 1;
    const int64_t items_per_thread = backward ? 1 : (seq_len >= 1024 ? state_unroll : 1);
    const int64_t block_threads = capability_major >= 9 ? 256 : (tile_size >= 128 ? 128 : 64);
    hints["block_threads"] = block_threads;
    hints["items_per_thread"] = items_per_thread;
    hints["vector_bytes"] = vector_bytes;
    hints["elements_per_load"] = elements_per_load;
    hints["rank_unroll"] = rank_unroll;
    hints["state_unroll"] = state_unroll;
    hints["load_path"] = async_shared ? "async_shared" : "shared";
    hints["workspace_mode"] = backward ? "tiled_backward" : "tiled_forward";
    hints["geometry_score"] = tile_size * split_size * std::max<int64_t>(items_per_thread, 1);
    return hints;
}

std::vector<int64_t> make_tiled_candidate_sizes(
    int64_t limit,
    std::initializer_list<int64_t> preferred) {
    std::vector<int64_t> values;
    const int64_t capped_limit = std::max<int64_t>(limit, 1);
    auto push_unique = [&](int64_t candidate) {
        const int64_t value = std::max<int64_t>(1, std::min<int64_t>(candidate, capped_limit));
        if (std::find(values.begin(), values.end(), value) == values.end()) {
            values.push_back(value);
        }
    };
    push_unique(capped_limit);
    for (const int64_t candidate : preferred) {
        push_unique(candidate);
    }
    return values;
}

int64_t score_tiled_runtime_candidate(const py::dict& info, bool backward) {
    const int64_t tile_size = info[py::str("tile_size")].cast<int64_t>();
    const int64_t split_size = info[py::str("split_size")].cast<int64_t>();
    const int64_t forward_occ = info[py::str("forward_estimated_occupancy_pct")].cast<int64_t>();
    const int64_t backward_occ = backward
        ? info[py::str("backward_estimated_occupancy_pct")].cast<int64_t>()
        : forward_occ;
    const int64_t occupancy = backward ? std::min(forward_occ, backward_occ) : forward_occ;
    const int64_t forward_vector_bytes = info[py::str("preferred_load_bytes")].cast<int64_t>();
    const int64_t backward_vector_bytes = backward
        ? info[py::str("backward_preferred_load_bytes")].cast<int64_t>()
        : forward_vector_bytes;
    const int64_t vector_bytes = std::min(forward_vector_bytes, backward_vector_bytes);
    const int64_t forward_elements_per_load = info[py::str("elements_per_load")].cast<int64_t>();
    const int64_t backward_elements_per_load = backward
        ? info[py::str("backward_elements_per_load")].cast<int64_t>()
        : forward_elements_per_load;
    const int64_t elements_per_load = std::min(forward_elements_per_load, backward_elements_per_load);
    const int64_t forward_bytes_moved = info[py::str("forward_estimated_bytes_moved")].cast<int64_t>();
    const int64_t backward_bytes_moved = backward
        ? info[py::str("backward_estimated_bytes_moved")].cast<int64_t>()
        : 0;
    const int64_t total_bytes_moved = forward_bytes_moved + backward_bytes_moved;
    const int64_t forward_sync_points = info[py::str("forward_estimated_sync_points")].cast<int64_t>();
    const int64_t backward_sync_points = backward
        ? info[py::str("backward_estimated_sync_points")].cast<int64_t>()
        : 0;
    const int64_t total_sync_points = forward_sync_points + backward_sync_points;
    const bool async_forward = info[py::str("async_pipeline_forward_supported")].cast<bool>();
    const bool vectorized_forward = info[py::str("forward_vectorized_io")].cast<bool>();
    const bool vectorized_backward = backward
        ? info[py::str("backward_vectorized_io")].cast<bool>()
        : vectorized_forward;
    const bool uses_persisting_l2 = info[py::str("uses_persisting_l2_window")].cast<bool>();
    const bool tensor_core_math = info[py::str("forward_tensor_core_math_supported")].cast<bool>();

    int64_t score = occupancy * 10000;
    score += std::min<int64_t>(tile_size * split_size, 32768);
    score += vector_bytes * 400;
    score += elements_per_load * 300;
    if (async_forward) {
        score += 12000;
    }
    if (vectorized_forward) {
        score += 6000;
    }
    if (vectorized_backward) {
        score += 4000;
    }
    if (tensor_core_math) {
        score += 3000;
    }
    if (uses_persisting_l2) {
        score += 2000;
    }
    score -= total_sync_points / 64;
    score -= total_bytes_moved / (8 * 1024 * 1024);
    return score;
}

py::dict causal_machine_scan_select_tiled_runtime_policy(
    int64_t num_states,
    int64_t transition_rank,
    int64_t seq_len,
    int64_t chunk_size,
    bool backward,
    int64_t device_index) {
    TORCH_CHECK(
        num_states > kSpecializedNumStates,
        "tiled runtime policy expects num_states > ",
        kSpecializedNumStates);
    TORCH_CHECK(transition_rank > 0 && transition_rank <= num_states, "transition_rank must be in [1, num_states]");
    TORCH_CHECK(seq_len >= 0, "seq_len must be non-negative");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    TORCH_CHECK(device_index >= 0, "device_index must be non-negative");

    const int64_t capability_major = causal_machine_scan_cached_capability_major_cuda(device_index);
    const auto tile_candidates = make_tiled_candidate_sizes(
        num_states,
        capability_major >= 9
            ? std::initializer_list<int64_t>{192, 160, 128, 96, 64, 48, 32}
            : std::initializer_list<int64_t>{160, 128, 96, 64, 48, 32});
    const auto split_candidates = make_tiled_candidate_sizes(
        transition_rank,
        capability_major >= 9
            ? std::initializer_list<int64_t>{128, 96, 64, 48, 32, 24, 16}
            : std::initializer_list<int64_t>{96, 64, 48, 32, 24, 16});

    py::dict best_info;
    int64_t best_score = std::numeric_limits<int64_t>::min();
    py::dict fallback_info;
    int64_t fallback_score = std::numeric_limits<int64_t>::min();
    int64_t candidate_count = 0;
    for (const int64_t raw_tile_size : tile_candidates) {
        const int64_t tile_size = std::max<int64_t>(32, std::min<int64_t>(num_states, raw_tile_size));
        for (const int64_t raw_split_size : split_candidates) {
            const int64_t split_size = std::max<int64_t>(16, std::min<int64_t>(transition_rank, raw_split_size));
            if (split_size > tile_size && tile_size < num_states) {
                continue;
            }
            ++candidate_count;
        auto info = causal_machine_scan_describe_tiled_runtime_config(
            num_states,
            transition_rank,
            seq_len,
            chunk_size,
            tile_size,
            split_size,
            1,
            device_index);
            const bool forward_supported = bool(info["custom_kernel_supported"].cast<bool>());
            const bool backward_supported = bool(info["custom_backward_kernel_supported"].cast<bool>());
            const bool async_shared = bool(info["async_pipeline_forward_supported"].cast<bool>());
            auto hints = make_tiled_geometry_hints(
                num_states,
                transition_rank,
                tile_size,
                split_size,
                seq_len,
                capability_major,
                backward,
                async_shared);
            for (auto item : hints) {
                info[item.first] = item.second;
            }
            info["block_threads"] = backward
                ? info[py::str("backward_block_threads")]
                : info[py::str("forward_block_threads")];
            const int64_t score = score_tiled_runtime_candidate(info, backward);
            info["selection_score"] = score;
            info["candidate_count"] = candidate_count;
            if (fallback_info.empty() || score > fallback_score) {
                fallback_info = info;
                fallback_score = score;
            }
            if (forward_supported && (!backward || backward_supported) && score > best_score) {
                best_info = info;
                best_score = score;
            }
        }
    }

    if (!best_info.empty()) {
        best_info["selected"] = true;
        best_info["selection_reason"] = backward
            ? "runtime_autotune_custom_forward_backward"
            : "runtime_autotune_custom_forward";
        best_info["candidate_count"] = candidate_count;
        return best_info;
    }

    if (!fallback_info.empty()) {
        fallback_info["selected"] = false;
        fallback_info["selection_reason"] = "runtime_autotune_fallback";
        fallback_info["candidate_count"] = candidate_count;
        return fallback_info;
    }
    TORCH_CHECK(false, "no tiled runtime policy candidates available");
}

py::dict causal_machine_scan_select_dense_runtime_policy(
    int64_t num_states,
    int64_t transition_rank,
    int64_t seq_len,
    int64_t chunk_size,
    bool needs_grad,
    int64_t device_index) {
    TORCH_CHECK(
        is_supported_specialized_num_states(num_states),
        "dense runtime policy expects num_states in [1, ",
        kSpecializedNumStates,
        "]");
    TORCH_CHECK(transition_rank > 0 && transition_rank <= num_states, "transition_rank must be in [1, num_states]");
    TORCH_CHECK(seq_len >= 0, "seq_len must be non-negative");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    TORCH_CHECK(device_index >= 0, "device_index must be non-negative");

    auto info = causal_machine_scan_describe_runtime_config(
        1,
        seq_len,
        num_states,
        transition_rank,
        chunk_size,
        device_index,
        needs_grad);
    const int64_t capability_major = causal_machine_scan_cached_capability_major_cuda(device_index);
    const bool exact_dense_128_rank8 = (num_states == 128 && transition_rank == 8);
    const bool exact_dense_128_rank16 = (num_states == 128 && transition_rank == 16);
    const bool optimized_rank = bool(info[py::str("optimized_rank")].cast<bool>());
    const bool direct_grad_reduce = bool(info[py::str("direct_grad_reduce")].cast<bool>());
    const int64_t shared_bytes = info[py::str("shared_bytes")].cast<int64_t>();
    const bool uses_persisting_l2 = bool(info[py::str("uses_persisting_l2_window")].cast<bool>());

    info["selected"] = true;
    info["candidate_count"] = (exact_dense_128_rank8 || exact_dense_128_rank16) ? 3 : (optimized_rank ? 2 : 1);
    info["supports_async_pipeline"] = capability_major >= 8;
    info["block_threads"] = num_states;
    info["items_per_thread"] = (!needs_grad && seq_len >= 1024) ? 2 : 1;
    info["vector_bytes"] = 4;
    info["elements_per_load"] = 1;
    info["tile_size"] = num_states;
    info["split_size"] = transition_rank;
    info["workspace_mode"] = "dense_forward";
    info["workspace_mode_backward"] = "dense_backward";
    info["load_path"] = capability_major >= 8 ? "async_shared" : "shared";
    info["direct_grad_reduce"] = direct_grad_reduce;
    info["uses_persisting_l2_window"] = uses_persisting_l2;

    int64_t score = 100000;
    score += std::min<int64_t>(num_states * transition_rank, 4096);
    score += uses_persisting_l2 ? 4000 : 0;
    score -= shared_bytes / 32;
    if (exact_dense_128_rank8) {
        info["kernel_family"] = needs_grad ? "dense_128_rank8_train" : "dense_128_rank8_eval";
        info["backward_kernel_family"] = "dense_128_rank8_backward";
        info["rank_unroll"] = 8;
        info["state_unroll"] = 4;
        info["selection_reason"] = needs_grad
            ? "exact_shape_dense_128_rank8_train"
            : "exact_shape_dense_128_rank8_eval";
        score += 120000;
        if (needs_grad && direct_grad_reduce) {
            score += 6000;
        }
        if (!needs_grad && seq_len == 1) {
            score += 12000;
        }
    } else if (exact_dense_128_rank16) {
        info["kernel_family"] = needs_grad ? "dense_128_rank16_train" : "dense_128_rank16_eval";
        info["backward_kernel_family"] = "dense_128_rank16_backward";
        info["rank_unroll"] = 16;
        info["state_unroll"] = 4;
        info["selection_reason"] = needs_grad
            ? "exact_shape_dense_128_rank16_train"
            : "exact_shape_dense_128_rank16_eval";
        score += 120000;
        if (needs_grad && direct_grad_reduce) {
            score += 6000;
        }
    } else if (optimized_rank) {
        info["kernel_family"] = "small_state_static_rank";
        info["backward_kernel_family"] = "small_state_static_rank_backward";
        info["rank_unroll"] = transition_rank >= 64 ? 4 : 2;
        info["state_unroll"] = num_states >= 96 ? 2 : 1;
        info["selection_reason"] = "optimized_small_state_static_rank";
        score += 18000;
    } else {
        info["kernel_family"] = "small_state_generic";
        info["backward_kernel_family"] = "small_state_generic_backward";
        info["rank_unroll"] = 1;
        info["state_unroll"] = 1;
        info["selection_reason"] = "generic_small_state_fallback";
    }
    info["selection_score"] = score;
    return info;
}

py::dict causal_machine_scan_describe_masked_tiled_runtime_config(
    int64_t num_states,
    int64_t batch_size,
    int64_t device_index,
    bool backward) {
    TORCH_CHECK(
        num_states > kSpecializedNumStates,
        "masked tiled runtime config expects num_states > ",
        kSpecializedNumStates);
    TORCH_CHECK(batch_size >= 0, "batch_size must be non-negative");
    TORCH_CHECK(device_index >= 0, "device_index must be non-negative");
    const int64_t tile_size = num_states;
    int64_t block_threads = 32;
    const int64_t required_threads = tile_size;
    while (block_threads < required_threads && block_threads < 256) {
        block_threads <<= 1;
    }
    const auto forward_runtime = !backward
        ? causal_machine_scan_describe_masked_tiled_forward_runtime_cuda(
            device_index,
            batch_size,
            num_states,
            num_states,
            tile_size,
            1)
        : std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const auto backward_runtime = backward
        ? causal_machine_scan_describe_masked_tiled_backward_runtime_cuda(
            device_index,
            batch_size,
            num_states,
            tile_size,
            1)
        : std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int64_t shared_bytes = backward
        ? causal_machine_scan_backward_masked_tiled_chunk_shared_bytes_cuda(num_states, tile_size)
        : causal_machine_scan_forward_masked_tiled_chunk_shared_bytes_cuda(num_states, tile_size);
    const bool custom_kernel_smem_supported = backward
        ? causal_machine_scan_can_use_masked_tiled_backward_kernel_cuda(device_index, num_states, tile_size)
        : causal_machine_scan_can_use_masked_tiled_forward_kernel_cuda(device_index, num_states, tile_size);
    const bool custom_kernel_memory_supported = !backward
        || backward_runtime.at(1) <= backward_runtime.at(2);
    const bool custom_kernel_supported = custom_kernel_smem_supported
        && custom_kernel_memory_supported;
    const bool extension_fallback_supported = true;
    const bool supports_tma = causal_machine_scan_can_use_tma_cuda(device_index);
    const bool supports_wgmma = causal_machine_scan_can_use_wgmma_cuda(device_index);
    py::dict info;
    info["num_states"] = num_states;
    info["tile_size"] = tile_size;
    info["block_threads"] = block_threads;
    info["batch_size"] = batch_size;
    info["persistent_blocks"] = causal_machine_scan_persistent_worker_blocks_cuda(device_index, batch_size);
    info["shared_bytes"] = shared_bytes;
    info["max_dynamic_smem_bytes"] = causal_machine_scan_cached_max_optin_bytes_cuda(device_index);
    info["preferred_load_bytes"] = backward ? runtime_value_at(backward_runtime, 8) : runtime_value_at(forward_runtime, 3);
    info["elements_per_load"] = backward ? runtime_value_at(backward_runtime, 9) : runtime_value_at(forward_runtime, 4);
    info["can_use_vectorized_io"] = backward ? bool(runtime_value_at(backward_runtime, 10)) : bool(runtime_value_at(forward_runtime, 5));
    info["async_pipeline_forward_supported"] = !backward && bool(runtime_value_at(forward_runtime, 6));
    info["forward_active_blocks_per_sm"] = runtime_value_at(forward_runtime, 7);
    info["forward_active_warps_per_sm"] = runtime_value_at(forward_runtime, 8);
    info["forward_max_warps_per_sm"] = runtime_value_at(forward_runtime, 9);
    info["forward_estimated_occupancy_pct"] = runtime_value_at(forward_runtime, 10);
    info["forward_registers_per_thread"] = runtime_value_at(forward_runtime, 11);
    info["forward_static_smem_bytes"] = runtime_value_at(forward_runtime, 12);
    info["forward_persisting_l2_candidate_bytes"] = runtime_value_at(forward_runtime, 13);
    info["forward_persisting_l2_effective_bytes"] = runtime_value_at(forward_runtime, 14);
    info["forward_uses_persisting_l2_window"] = bool(runtime_value_at(forward_runtime, 15));
    info["forward_estimated_bytes_moved"] = runtime_value_at(forward_runtime, 16);
    info["forward_estimated_sync_points"] = runtime_value_at(forward_runtime, 17);
    info["custom_kernel_smem_supported"] = custom_kernel_smem_supported;
    info["custom_kernel_memory_supported"] = custom_kernel_memory_supported;
    info["custom_kernel_supported"] = custom_kernel_supported;
    info["extension_fallback_supported"] = extension_fallback_supported;
    info["supports_tma"] = supports_tma;
    info["supports_wgmma"] = supports_wgmma;
    info["runtime_supported"] = custom_kernel_supported || extension_fallback_supported;
    info["uses_persisting_l2_window"] = backward
        ? bool(runtime_value_at(backward_runtime, 13))
        : bool(runtime_value_at(forward_runtime, 15));
    info["backward_runtime_shared_bytes"] = backward ? runtime_value_at(backward_runtime, 1) : 0;
    info["backward_tile_cache_bytes"] = backward ? runtime_value_at(backward_runtime, 1) : 0;
    info["backward_active_blocks_per_sm"] = runtime_value_at(backward_runtime, 3);
    info["backward_active_warps_per_sm"] = runtime_value_at(backward_runtime, 4);
    info["backward_max_warps_per_sm"] = runtime_value_at(backward_runtime, 5);
    info["backward_estimated_occupancy_pct"] = runtime_value_at(backward_runtime, 6);
    info["backward_registers_per_thread"] = runtime_value_at(backward_runtime, 7);
    info["backward_static_smem_bytes"] = runtime_value_at(backward_runtime, 8);
    info["backward_preferred_load_bytes"] = runtime_value_at(backward_runtime, 9);
    info["backward_elements_per_load"] = runtime_value_at(backward_runtime, 10);
    info["backward_vectorized_io"] = bool(runtime_value_at(backward_runtime, 11));
    info["backward_async_copy_path"] = bool(runtime_value_at(backward_runtime, 12));
    info["backward_persisting_l2_candidate_bytes"] = runtime_value_at(backward_runtime, 13);
    info["backward_persisting_l2_effective_bytes"] = runtime_value_at(backward_runtime, 14);
    info["backward_uses_persisting_l2_window"] = bool(runtime_value_at(backward_runtime, 15));
    info["backward_estimated_bytes_moved"] = runtime_value_at(backward_runtime, 16);
    info["backward_estimated_sync_points"] = runtime_value_at(backward_runtime, 17);
    info["free_global_mem_bytes"] = backward ? runtime_value_at(backward_runtime, 18) : runtime_value_at(forward_runtime, 18);
    info["total_global_mem_bytes"] = backward ? runtime_value_at(backward_runtime, 19) : runtime_value_at(forward_runtime, 19);
    info["backward"] = backward;
    info["kernel_choice_reason"] = custom_kernel_supported
        ? (backward
            ? (supports_tma
                ? (supports_wgmma ? "masked_tiled_tma_wgmma_backward" : "masked_tiled_tma_backward")
                : (supports_wgmma ? "masked_tiled_wgmma_backward" : "masked_tiled_custom"))
            : (supports_tma
                ? (supports_wgmma ? "masked_tiled_tma_wgmma_forward" : "masked_tiled_tma_forward")
                : (supports_wgmma ? "masked_tiled_wgmma_forward" : "masked_tiled_custom_forward")))
        : "masked_extension_fallback";
    return info;
}

py::dict causal_machine_scan_describe_device_runtime_config(int64_t device_index) {
    TORCH_CHECK(device_index >= 0, "device_index must be non-negative");
    py::dict info;
    info["device_index"] = device_index;
    info["capability_major"] = causal_machine_scan_cached_capability_major_cuda(device_index);
    info["capability_minor"] = causal_machine_scan_cached_capability_minor_cuda(device_index);
    info["sm_count"] = causal_machine_scan_cached_sm_count_cuda(device_index);
    info["l2_cache_bytes"] = causal_machine_scan_cached_l2_cache_size_cuda(device_index);
    info["persisting_l2_cache_max_bytes"] = causal_machine_scan_cached_persisting_l2_cache_max_size_cuda(device_index);
    info["supports_persisting_l2_window"] = causal_machine_scan_supports_persisting_l2_window_cuda(device_index);
    info["max_dynamic_smem_bytes"] = causal_machine_scan_cached_max_optin_bytes_cuda(device_index);
    info["total_global_mem_bytes"] = causal_machine_scan_cached_total_global_mem_cuda(device_index);
    info["supports_async_memcpy"] = causal_machine_scan_can_use_async_memcpy_cuda(device_index);
    info["supports_tensor_cores"] = causal_machine_scan_can_use_tensor_cores_cuda(device_index);
    info["supports_half2_path"] = causal_machine_scan_can_use_half2_path_cuda(device_index);
    info["supports_wmma"] = causal_machine_scan_can_use_wmma_cuda(device_index);
    info["supports_tma"] = causal_machine_scan_can_use_tma_cuda(device_index);
    info["supports_wgmma"] = causal_machine_scan_can_use_wgmma_cuda(device_index);
    info["structured_scan_tma_kernels_implemented"] = true;
    info["structured_scan_wgmma_kernels_implemented"] = true;
    info["preferred_load_bytes"] = causal_machine_scan_preferred_load_bytes_cuda(0, 64, 64);
    info["elements_per_load"] = causal_machine_scan_elements_per_load_cuda(0, 64, 64);
    return info;
}

void check_same_cuda_device(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name);

torch::Tensor make_cuda_workspace_tensor(
    int64_t device_index,
    at::ScalarType dtype,
    at::IntArrayRef sizes,
    bool zero_init) {
    auto options = torch::TensorOptions().device(torch::kCUDA, device_index).dtype(dtype);
    return zero_init ? torch::zeros(sizes, options) : torch::empty(sizes, options);
}

std::string normalize_scan_workspace_mode(const std::string& mode) {
    if (mode == "tiled_forward" || mode == "tiled_backward"
            || mode == "masked_tiled_forward" || mode == "masked_tiled_backward") {
        return mode;
    }
    TORCH_CHECK(
        false,
        "scan workspace mode must be one of tiled_forward, tiled_backward, masked_tiled_forward, masked_tiled_backward"
    );
}

torch::Tensor get_workspace_tensor(
    const py::dict& workspace,
    const char* key,
    at::ScalarType dtype,
    const torch::Tensor& reference) {
    TORCH_CHECK(workspace.contains(py::str(key)), "workspace is missing required tensor: ", key);
    auto tensor = workspace[py::str(key)].cast<torch::Tensor>();
    TORCH_CHECK(tensor.scalar_type() == dtype, "workspace tensor ", key, " has incorrect dtype");
    check_same_cuda_device(tensor, reference, key);
    return tensor;
}

torch::Tensor get_optional_workspace_tensor(
    const py::dict& workspace,
    const char* key,
    at::ScalarType dtype,
    const torch::Tensor& reference) {
    if (!workspace.contains(py::str(key))) {
        return torch::Tensor();
    }
    auto tensor = workspace[py::str(key)].cast<torch::Tensor>();
    TORCH_CHECK(tensor.scalar_type() == dtype, "workspace tensor ", key, " has incorrect dtype");
    check_same_cuda_device(tensor, reference, key);
    return tensor;
}

void check_workspace_shape_min(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& required_shape,
    const char* name) {
    TORCH_CHECK(
        tensor.dim() == static_cast<int64_t>(required_shape.size()),
        "workspace tensor ",
        name,
        " must have rank ",
        required_shape.size());
    for (size_t dim = 0; dim < required_shape.size(); ++dim) {
        TORCH_CHECK(
            tensor.size(static_cast<int64_t>(dim)) >= required_shape[dim],
            "workspace tensor ",
            name,
            " is too small on dim ",
            dim,
            ": got ",
            tensor.size(static_cast<int64_t>(dim)),
            ", need at least ",
            required_shape[dim]);
    }
}

py::dict causal_machine_scan_describe_workspace_config(
    const std::string& mode,
    int64_t num_states,
    int64_t transition_rank,
    int64_t batch_size,
    int64_t tile_size,
    int64_t split_size,
    int64_t seq_len,
    int64_t chunk_size,
    int64_t device_index) {
    const std::string normalized_mode = normalize_scan_workspace_mode(mode);
    TORCH_CHECK(num_states > 0, "num_states must be positive");
    TORCH_CHECK(transition_rank > 0, "transition_rank must be positive");
    TORCH_CHECK(batch_size >= 0, "batch_size must be non-negative");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(split_size > 0, "split_size must be positive");
    TORCH_CHECK(device_index >= 0, "device_index must be non-negative");

    py::dict info;
    info["mode"] = normalized_mode;
    info["device_index"] = device_index;
    info["num_states"] = num_states;
    info["transition_rank"] = transition_rank;
    info["batch_size"] = batch_size;
    info["tile_size"] = tile_size;
    info["split_size"] = split_size;
    info["seq_len"] = seq_len;
    info["chunk_size"] = chunk_size;

    int64_t total_bytes = 2 * static_cast<int64_t>(sizeof(int32_t));
    info["work_queue_counter_shape"] = std::vector<int64_t>{2};

    if (normalized_mode == "tiled_forward") {
        auto runtime = causal_machine_scan_describe_tiled_runtime_config(
            num_states,
            transition_rank,
            seq_len,
            chunk_size,
            tile_size,
            split_size,
            batch_size,
            device_index);
        const int64_t persistent_blocks = runtime["persistent_blocks"].cast<int64_t>();
        info["persistent_blocks"] = persistent_blocks;
        info["filtered_value_cache_shape"] = std::vector<int64_t>{persistent_blocks, num_states};
        total_bytes += persistent_blocks * num_states * static_cast<int64_t>(sizeof(float));
    } else if (normalized_mode == "masked_tiled_forward") {
        auto runtime = causal_machine_scan_describe_masked_tiled_runtime_config(
            num_states,
            batch_size,
            device_index,
            false);
        const int64_t persistent_blocks = runtime["persistent_blocks"].cast<int64_t>();
        info["persistent_blocks"] = persistent_blocks;
        info["masked_transition_tile_cache_shape"] = std::vector<int64_t>{persistent_blocks, num_states, tile_size};
        info["filtered_value_cache_shape"] = std::vector<int64_t>{persistent_blocks, num_states};
        info["row_sums_shape"] = std::vector<int64_t>{num_states};
        total_bytes += persistent_blocks * num_states * tile_size * static_cast<int64_t>(sizeof(float));
        total_bytes += persistent_blocks * num_states * static_cast<int64_t>(sizeof(float));
        total_bytes += num_states * static_cast<int64_t>(sizeof(float));
    } else if (normalized_mode == "tiled_backward") {
        auto runtime = causal_machine_scan_describe_tiled_runtime_config(
            num_states,
            transition_rank,
            seq_len,
            chunk_size,
            tile_size,
            split_size,
            batch_size,
            device_index);
        const int64_t persistent_blocks = runtime["persistent_blocks"].cast<int64_t>();
        const int64_t staging_worker_blocks = runtime["backward_staging_worker_blocks"].cast<int64_t>();
        info["persistent_blocks"] = persistent_blocks;
        info["staging_worker_blocks"] = staging_worker_blocks;
        info["filtered_value_cache_shape"] = std::vector<int64_t>{persistent_blocks, num_states};
        info["latent_cache_staging_shape"] = std::vector<int64_t>{staging_worker_blocks, transition_rank};
        info["grad_latent_accum_staging_shape"] = std::vector<int64_t>{staging_worker_blocks, transition_rank};
        info["grad_transition_source_probs_staging_shape"] = std::vector<int64_t>{staging_worker_blocks, num_states, transition_rank};
        info["grad_transition_dest_probs_staging_shape"] = std::vector<int64_t>{staging_worker_blocks, transition_rank, num_states};
        info["grad_transition_gate_staging_shape"] = std::vector<int64_t>{staging_worker_blocks};
        info["grad_transition_stay_staging_shape"] = std::vector<int64_t>{staging_worker_blocks, num_states};
        total_bytes += persistent_blocks * num_states * static_cast<int64_t>(sizeof(float));
        total_bytes += staging_worker_blocks * transition_rank * static_cast<int64_t>(sizeof(float));
        total_bytes += staging_worker_blocks * transition_rank * static_cast<int64_t>(sizeof(float));
        total_bytes += staging_worker_blocks * num_states * transition_rank * static_cast<int64_t>(sizeof(float));
        total_bytes += staging_worker_blocks * transition_rank * num_states * static_cast<int64_t>(sizeof(float));
        total_bytes += staging_worker_blocks * static_cast<int64_t>(sizeof(float));
        total_bytes += staging_worker_blocks * num_states * static_cast<int64_t>(sizeof(float));
    } else {
        auto runtime = causal_machine_scan_describe_masked_tiled_runtime_config(
            num_states,
            batch_size,
            device_index,
            true);
        const int64_t persistent_blocks = runtime["persistent_blocks"].cast<int64_t>();
        info["persistent_blocks"] = persistent_blocks;
        info["masked_transition_tile_cache_shape"] = std::vector<int64_t>{persistent_blocks, num_states, tile_size};
        info["filtered_value_cache_shape"] = std::vector<int64_t>{persistent_blocks, num_states};
        info["row_sums_shape"] = std::vector<int64_t>{num_states};
        total_bytes += persistent_blocks * num_states * tile_size * static_cast<int64_t>(sizeof(float));
        total_bytes += persistent_blocks * num_states * static_cast<int64_t>(sizeof(float));
        total_bytes += num_states * static_cast<int64_t>(sizeof(float));
    }

    info["workspace_total_bytes"] = total_bytes;
    return info;
}

py::dict causal_machine_scan_create_workspace(
    const std::string& mode,
    int64_t num_states,
    int64_t transition_rank,
    int64_t batch_size,
    int64_t tile_size,
    int64_t split_size,
    int64_t seq_len,
    int64_t chunk_size,
    int64_t device_index) {
    auto info = causal_machine_scan_describe_workspace_config(
        mode,
        num_states,
        transition_rank,
        batch_size,
        tile_size,
        split_size,
        seq_len,
        chunk_size,
        device_index);
    const std::string normalized_mode = info["mode"].cast<std::string>();
    py::dict workspace;
    workspace["mode"] = normalized_mode;
    workspace["work_queue_counter"] = make_cuda_workspace_tensor(
        device_index,
        torch::kInt32,
        {2},
        true);
    if (normalized_mode == "tiled_forward") {
        workspace["filtered_value_cache"] = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            info["filtered_value_cache_shape"].cast<std::vector<int64_t>>());
    } else if (normalized_mode == "masked_tiled_forward") {
        const auto shape_a = info["masked_transition_tile_cache_shape"].cast<std::vector<int64_t>>();
        const auto shape_b = info["filtered_value_cache_shape"].cast<std::vector<int64_t>>();
        workspace["masked_transition_tile_cache"] = make_cuda_workspace_tensor(device_index, torch::kFloat32, shape_a);
        workspace["filtered_value_cache"] = make_cuda_workspace_tensor(device_index, torch::kFloat32, shape_b);
        workspace["row_sums"] = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            info["row_sums_shape"].cast<std::vector<int64_t>>());
    } else if (normalized_mode == "tiled_backward") {
        workspace["filtered_value_cache"] = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            info["filtered_value_cache_shape"].cast<std::vector<int64_t>>());
        workspace["latent_cache_staging"] = make_cuda_workspace_tensor(
            device_index, torch::kFloat32, info["latent_cache_staging_shape"].cast<std::vector<int64_t>>());
        workspace["grad_latent_accum_staging"] = make_cuda_workspace_tensor(
            device_index, torch::kFloat32, info["grad_latent_accum_staging_shape"].cast<std::vector<int64_t>>());
        workspace["grad_transition_source_probs_staging"] = make_cuda_workspace_tensor(
            device_index, torch::kFloat32, info["grad_transition_source_probs_staging_shape"].cast<std::vector<int64_t>>(), true);
        workspace["grad_transition_dest_probs_staging"] = make_cuda_workspace_tensor(
            device_index, torch::kFloat32, info["grad_transition_dest_probs_staging_shape"].cast<std::vector<int64_t>>(), true);
        workspace["grad_transition_gate_staging"] = make_cuda_workspace_tensor(
            device_index, torch::kFloat32, info["grad_transition_gate_staging_shape"].cast<std::vector<int64_t>>(), true);
        workspace["grad_transition_stay_staging"] = make_cuda_workspace_tensor(
            device_index, torch::kFloat32, info["grad_transition_stay_staging_shape"].cast<std::vector<int64_t>>(), true);
    } else {
        workspace["masked_transition_tile_cache"] = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            info["masked_transition_tile_cache_shape"].cast<std::vector<int64_t>>());
        workspace["filtered_value_cache"] = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            info["filtered_value_cache_shape"].cast<std::vector<int64_t>>());
        workspace["row_sums"] = make_cuda_workspace_tensor(
            device_index,
            torch::kFloat32,
            info["row_sums_shape"].cast<std::vector<int64_t>>());
    }
    workspace["config"] = info;
    return workspace;
}

torch::Tensor pad_first_dim(const torch::Tensor& tensor, int64_t target_size, double fill_value) {
    if (tensor.size(0) == target_size) {
        return tensor;
    }
    auto sizes = tensor.sizes().vec();
    sizes[0] = target_size;
    auto padded = torch::full(sizes, fill_value, tensor.options());
    padded.narrow(0, 0, tensor.size(0)).copy_(tensor);
    return padded.contiguous();
}

torch::Tensor pad_last_dim(const torch::Tensor& tensor, int64_t target_size, double fill_value) {
    if (tensor.size(-1) == target_size) {
        return tensor;
    }
    auto sizes = tensor.sizes().vec();
    sizes.back() = target_size;
    auto padded = torch::full(sizes, fill_value, tensor.options());
    padded.narrow(tensor.dim() - 1, 0, tensor.size(-1)).copy_(tensor);
    return padded.contiguous();
}

torch::Tensor slice_first_dim(const torch::Tensor& tensor, int64_t size) {
    return tensor.narrow(0, 0, size).contiguous();
}

torch::Tensor slice_last_dim(const torch::Tensor& tensor, int64_t size) {
    return tensor.narrow(tensor.dim() - 1, 0, size).contiguous();
}

bool needs_specialized_rank_padding(int64_t num_states, int64_t transition_rank) {
    return transition_rank > num_states;
}

std::vector<torch::Tensor> slice_specialized_forward_outputs(
    const std::vector<torch::Tensor>& outputs,
    int64_t num_states) {
    return {
        slice_last_dim(outputs[0], num_states),
        slice_last_dim(outputs[1], num_states),
    };
}

std::vector<torch::Tensor> slice_specialized_backward_outputs(
    const std::vector<torch::Tensor>& outputs,
    int64_t num_states) {
    return {
        slice_last_dim(outputs[0], num_states),
        slice_first_dim(outputs[1], num_states),
        slice_last_dim(outputs[2], num_states),
        slice_last_dim(outputs[3], num_states),
        slice_last_dim(outputs[4], num_states),
        outputs[5],
        slice_last_dim(outputs[6], num_states),
    };
}

template <typename PaddedFn, typename DirectFn>
std::vector<torch::Tensor> dispatch_specialized_forward_wrapper(
    int64_t num_states,
    int64_t transition_rank,
    PaddedFn&& padded_fn,
    DirectFn&& direct_fn) {
    if (needs_specialized_rank_padding(num_states, transition_rank)) {
        return slice_specialized_forward_outputs(padded_fn(), num_states);
    }
    return direct_fn();
}

template <typename PaddedFn, typename DirectFn>
std::vector<torch::Tensor> dispatch_specialized_backward_wrapper(
    int64_t num_states,
    int64_t transition_rank,
    PaddedFn&& padded_fn,
    DirectFn&& direct_fn) {
    if (needs_specialized_rank_padding(num_states, transition_rank)) {
        return slice_specialized_backward_outputs(padded_fn(), num_states);
    }
    return direct_fn();
}

double neg_inf_fill() {
    return -std::numeric_limits<float>::infinity();
}

bool is_supported_activation_dtype(const torch::Tensor& tensor) {
    return tensor.scalar_type() == torch::kFloat32
        || tensor.scalar_type() == torch::kFloat16
        || tensor.scalar_type() == torch::kBFloat16;
}

bool is_optional_tensor_defined(const torch::Tensor& tensor) {
    return tensor.defined() && tensor.numel() > 0;
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

torch::Tensor normalize_transition_gate_tensor(
    const torch::Tensor& transition_gate,
    const torch::Tensor& reference,
    const char* name = "transition_gate") {
    check_cuda_float32(transition_gate, name);
    check_same_cuda_device(transition_gate, reference, name);
    TORCH_CHECK(transition_gate.numel() == 1, name, " must be a scalar tensor");
    return transition_gate.contiguous().reshape({});
}

double transition_gate_value_from_tensor(
    const torch::Tensor& transition_gate,
    const torch::Tensor& reference,
    const char* name = "transition_gate") {
    return static_cast<double>(normalize_transition_gate_tensor(transition_gate, reference, name).item<float>());
}

void check_cuda_int8(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt8, name, " must be int8");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_uint8(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, name, " must be uint8");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_bool(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kBool, name, " must be bool");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_int64(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_int32(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
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
    TORCH_CHECK(
        local_logits.dim() == 3,
        "local_logits must have shape [B, L, N] for the specialized CUDA fast path"
    );
    const auto num_states = local_logits.size(2);
    TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits shape");
    TORCH_CHECK(
        is_supported_specialized_num_states(num_states),
        "local_logits last dim must be in [1, 128] for the small-state CUDA fast path"
    );
    TORCH_CHECK(
        transition_source_probs.dim() == 2,
        "transition_source_probs must have shape [N, R]"
    );
    TORCH_CHECK(
        transition_source_probs.size(0) == num_states,
        "transition_source_probs first dim must match local_logits last dim"
    );
    TORCH_CHECK(
        transition_source_probs.size(1) <= kSpecializedNumStates,
        "transition_rank must be <= ",
        kSpecializedNumStates,
        " for the specialized causal_machine_scan CUDA fast path"
    );
    TORCH_CHECK(
        transition_dest_probs.dim() == 2,
        "transition_dest_probs must have shape [R, N]"
    );
    TORCH_CHECK(
        transition_dest_probs.size(1) == num_states,
        "transition_dest_probs last dim must match local_logits last dim"
    );
    TORCH_CHECK(
        transition_dest_probs.size(0) == transition_source_probs.size(1),
        "transition_source_probs and transition_dest_probs rank must match"
    );
    TORCH_CHECK(
        transition_stay_probs.dim() == 1,
        "transition_stay_probs must have shape [N]"
    );
    TORCH_CHECK(
        transition_stay_probs.size(0) == num_states,
        "transition_stay_probs size must match local_logits last dim"
    );
    TORCH_CHECK(
        initial_log_belief.dim() == 2,
        "initial_log_belief must have shape [B, N]"
    );
    TORCH_CHECK(initial_log_belief.size(0) == local_logits.size(0), "initial_log_belief batch must match local_logits");
    TORCH_CHECK(
        initial_log_belief.size(1) == num_states,
        "initial_log_belief last dim must match local_logits last dim"
    );
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
        transition_source_scales.dim() == 1 && transition_source_scales.size(0) == local_logits.size(2),
        "transition_source_scales must have shape [N]"
    );
    TORCH_CHECK(
        transition_dest_scales.dim() == 1 && transition_dest_scales.size(0) == transition_dest_q.size(0),
        "transition_dest_scales must have shape [R]"
    );
}

void check_structured_fp8_shapes(
    const torch::Tensor& local_logits,
    const torch::Tensor& transition_source_packed,
    const torch::Tensor& transition_source_scales,
    const torch::Tensor& transition_dest_packed,
    const torch::Tensor& transition_dest_scales,
    const torch::Tensor& transition_context,
    const torch::Tensor& initial_log_belief,
    const torch::Tensor& transition_stay_probs) {
    check_structured_shapes(
        local_logits,
        transition_source_packed,
        transition_dest_packed,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    TORCH_CHECK(
        transition_source_scales.dim() == 1 && transition_source_scales.size(0) == local_logits.size(2),
        "transition_source_scales must have shape [N]"
    );
    TORCH_CHECK(
        transition_dest_scales.dim() == 1 && transition_dest_scales.size(0) == transition_dest_packed.size(0),
        "transition_dest_scales must have shape [R]"
    );
}

torch::Tensor empty_cuda_bool_tensor_like(const torch::Tensor& reference) {
    return torch::empty({0, 0}, reference.options().dtype(torch::kBool));
}

torch::Tensor empty_cuda_int64_tensor_like(const torch::Tensor& reference) {
    return torch::empty({0}, reference.options().dtype(torch::kInt64));
}

namespace {

std::tuple<torch::Tensor, torch::Tensor> build_masked_transition_matrix_from_factors(
    const torch::Tensor& transition_source_probs,
    const torch::Tensor& transition_dest_probs,
    const torch::Tensor& transition_mask) {
    const auto float_options = transition_source_probs.options().dtype(torch::kFloat32);
    auto raw_transition = torch::matmul(
        transition_source_probs.to(float_options),
        transition_dest_probs.to(float_options));
    auto masked_transition = raw_transition * transition_mask.to(float_options);
    auto row_sums = masked_transition.sum(1, true).clamp_min(1.0e-20f);
    return {
        masked_transition / row_sums,
        row_sums,
    };
}

struct MaskedSparseFallbackMetadata {
    torch::Tensor row_ptr;
    torch::Tensor col_idx;
    torch::Tensor dst_idx;
    torch::Tensor src_row_ptr;
    torch::Tensor src_nz_idx;
    torch::Tensor grouped_src_row_ptr;
    torch::Tensor grouped_src_block_idx;
    torch::Tensor grouped_src_group_ids;
    torch::Tensor block_mask;
    int64_t block_size;
};

int64_t default_masked_sparse_block_size(int64_t num_states) {
    if (num_states <= kSpecializedNumStates) {
        return 32;
    }
    if (num_states <= 256) {
        return 64;
    }
    return kSpecializedNumStates;
}

std::vector<torch::Tensor> build_grouped_sparse_backward_metadata_local(
    torch::Tensor col_idx,
    torch::Tensor src_nz_idx) {
    check_cuda_int32(col_idx, "col_idx");
    check_cuda_int32(src_nz_idx, "src_nz_idx");
    check_same_cuda_device(src_nz_idx, col_idx, "src_nz_idx");
    TORCH_CHECK(col_idx.dim() == 1, "col_idx must be 1D");
    TORCH_CHECK(src_nz_idx.dim() == 1, "src_nz_idx must be 1D");
    auto options_i32 = col_idx.options().dtype(torch::kInt32);
    auto empty_i32 = torch::empty({0}, options_i32);
    if (src_nz_idx.numel() == 0) {
        return {
            torch::zeros({1}, options_i32),
            empty_i32,
            empty_i32,
        };
    }
    auto ordered_src_blocks = col_idx.index_select(0, src_nz_idx.to(torch::kInt64)).to(torch::kInt32).contiguous();
    auto starts = torch::ones_like(ordered_src_blocks, options_i32);
    if (ordered_src_blocks.size(0) > 1) {
        auto curr = ordered_src_blocks.slice(0, 1);
        auto prev = ordered_src_blocks.slice(0, 0, ordered_src_blocks.size(0) - 1);
        starts.slice(0, 1).copy_(curr.ne(prev).to(torch::kInt32));
    }
    auto group_ids = starts.cumsum(0).sub_(1).to(torch::kInt32).contiguous();
    const int64_t num_groups = static_cast<int64_t>(group_ids[-1].item<int>()) + 1;
    auto counts = torch::bincount(group_ids.to(torch::kInt64), torch::Tensor(), num_groups).to(torch::kInt32);
    auto row_ptr = torch::zeros({num_groups + 1}, options_i32);
    if (num_groups > 0) {
        row_ptr.narrow(0, 1, num_groups).copy_(counts.cumsum(0).to(torch::kInt32));
    }
    auto group_start_idx = torch::nonzero(starts).squeeze(1).to(torch::kInt64);
    auto grouped_src_block_idx = ordered_src_blocks.index_select(0, group_start_idx).to(torch::kInt32).contiguous();
    return {
        row_ptr.contiguous(),
        grouped_src_block_idx,
        group_ids,
    };
}

MaskedSparseFallbackMetadata build_masked_sparse_fallback_metadata(
    int64_t num_states,
    const torch::Tensor& transition_mask) {
    const auto block_size = default_masked_sparse_block_size(num_states);
    const auto num_state_blocks = ceil_div_int64(num_states, block_size);
    const auto padded_states = num_state_blocks * block_size;
    auto empty_block_mask = torch::empty({0, 0}, transition_mask.options().dtype(torch::kBool));
    auto sparse_meta = causal_machine_scan_build_sparse_metadata_from_runtime(
        num_states,
        padded_states,
        block_size,
        -1,
        transition_mask,
        empty_block_mask);
    auto row_ptr = sparse_meta[0];
    auto col_idx = sparse_meta[1];
    auto dst_idx = sparse_meta[2];
    auto src_row_ptr = sparse_meta[3];
    auto src_nz_idx = sparse_meta[4];
    auto block_mask = sparse_meta[5];
    auto grouped_sparse_meta = build_grouped_sparse_backward_metadata_local(col_idx, src_nz_idx);
    return {
        row_ptr,
        col_idx,
        dst_idx,
        src_row_ptr,
        src_nz_idx,
        grouped_sparse_meta[0],
        grouped_sparse_meta[1],
        grouped_sparse_meta[2],
        block_mask,
        block_size,
    };
}

std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_aten(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size) {
    const auto batch_size = local_logits.size(0);
    const auto seq_len = local_logits.size(1);
    const auto num_states = local_logits.size(2);
    const bool has_seq_lens = seq_lens.defined() && seq_lens.numel() > 0;
    const auto float_options = transition_source_probs.options().dtype(torch::kFloat32);
    const auto transition_gate_value = static_cast<float>(transition_gate_value_from_tensor(transition_gate, local_logits));
    const auto effective_chunk_size = std::max<int64_t>(1, chunk_size);
    auto transition_matrix_and_sums = build_masked_transition_matrix_from_factors(
        transition_source_probs,
        transition_dest_probs,
        transition_mask);
    auto transition_matrix = std::get<0>(transition_matrix_and_sums);
    auto prev_log_belief = initial_log_belief.to(float_options);
    auto prev_probs = prev_log_belief.exp();
    auto stay_probs = transition_stay_probs.to(float_options).view({1, num_states});
    std::vector<torch::Tensor> belief_steps;
    belief_steps.reserve(static_cast<size_t>(seq_len));

    for (int64_t chunk_start = 0; chunk_start < seq_len; chunk_start += effective_chunk_size) {
        const int64_t chunk_end = std::min<int64_t>(seq_len, chunk_start + effective_chunk_size);
        for (int64_t pos = chunk_start; pos < chunk_end; ++pos) {
            auto local_logits_t = local_logits.select(1, pos).to(float_options);
            auto transition_context_t = transition_context.select(1, pos).to(float_options);
            auto mix_probs = torch::matmul(prev_probs, transition_matrix);
            auto pred_probs = (
                stay_probs * prev_probs + (1.0f - stay_probs) * mix_probs).clamp_min(1.0e-20f);
            auto pred_log = pred_probs.log();
            auto filtered_logits = local_logits_t
                + transition_gate_value * (pred_log + transition_context_t);
            auto next_log_belief = torch::log_softmax(filtered_logits, -1);
            if (has_seq_lens) {
                auto active = seq_lens.gt(pos).view({batch_size, 1});
                next_log_belief = torch::where(active, next_log_belief, prev_log_belief);
            }
            belief_steps.push_back(next_log_belief.to(local_logits.options()));
            prev_log_belief = next_log_belief;
            prev_probs = prev_log_belief.exp();
        }
    }

    auto beliefs = belief_steps.empty()
        ? torch::empty_like(local_logits)
        : torch::stack(belief_steps, 1);
    return {
        beliefs,
        prev_log_belief.to(local_logits.options()),
    };
}

std::vector<torch::Tensor> causal_machine_scan_backward_masked_tiled_probs_aten(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size) {
    const auto batch_size = beliefs.size(0);
    const auto seq_len = beliefs.size(1);
    const auto num_states = beliefs.size(2);
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
    const auto transition_gate_value = static_cast<float>(transition_gate_value_from_tensor(transition_gate, beliefs));
    const auto effective_chunk_size = std::max<int64_t>(1, chunk_size);
    auto transition_matrix_and_sums = build_masked_transition_matrix_from_factors(
        transition_source_probs,
        transition_dest_probs,
        transition_mask);
    auto transition_matrix = std::get<0>(transition_matrix_and_sums);
    auto row_sums = std::get<1>(transition_matrix_and_sums);
    auto transition_mask_f32 = transition_mask.to(float_options);
    auto grad_beliefs_f32 = grad_beliefs.to(float_options);
    auto grad_final_belief_f32 = grad_final_belief.to(float_options);
    auto transition_context_f32 = transition_context.to(float_options);
    auto initial_log_belief_f32 = initial_log_belief.to(float_options);
    auto beliefs_f32 = beliefs.to(float_options);
    auto stay_probs = transition_stay_probs.to(float_options).view({1, num_states});
    auto carry = grad_final_belief_f32.clone();
    auto grad_transition_matrix = torch::zeros_like(transition_matrix);

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
            auto mix_probs = torch::matmul(prev_probs, transition_matrix);
            auto pred_probs = (
                stay_probs * prev_probs + (1.0f - stay_probs) * mix_probs).clamp_min(1.0e-20f);
            auto pred_log = pred_probs.log();
            auto ga = (gq - q_prob * gq_sum) * active_f;

            grad_local_logits.select(1, pos).copy_(ga.to(grad_local_logits.scalar_type()));
            grad_transition_context.select(1, pos).copy_(
                (transition_gate_value * ga).to(grad_transition_context.scalar_type()));
            grad_transition_gate.add_((ga * (pred_log + transition_context_t)).sum().view({1}));

            auto grad_pred_prob = active_f * ((transition_gate_value * ga) / pred_probs);
            grad_transition_stay.add_((grad_pred_prob * (prev_probs - mix_probs)).sum(0));

            auto grad_prev_probs = grad_pred_prob * stay_probs;
            auto grad_mix_probs = grad_pred_prob * (1.0f - stay_probs);
            grad_transition_matrix.add_(torch::matmul(prev_probs.transpose(0, 1), grad_mix_probs));
            grad_prev_probs.add_(torch::matmul(grad_mix_probs, transition_matrix.transpose(0, 1)));

            auto grad_prev_log = grad_prev_probs * prev_probs;
            carry = active_f * grad_prev_log + (1.0f - active_f) * gq;
        }
    }

    auto row_dot = (grad_transition_matrix * transition_matrix).sum(1, true);
    auto grad_raw_masked = (grad_transition_matrix - row_dot) / row_sums;
    auto grad_raw = grad_raw_masked * transition_mask_f32;
    grad_transition_source_probs.copy_(
        torch::matmul(grad_raw, transition_dest_probs.to(float_options).transpose(0, 1)));
    grad_transition_dest_probs.copy_(
        torch::matmul(transition_source_probs.to(float_options).transpose(0, 1), grad_raw));
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

std::vector<torch::Tensor> causal_machine_scan_forward_masked_logits(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_float32(transition_source_logits, "transition_source_logits");
    check_cuda_float32(transition_dest_logits, "transition_dest_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    TORCH_CHECK(local_logits.dim() == 3, "local_logits must have shape [B, L, N]");
    TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits shape");
    TORCH_CHECK(transition_source_logits.dim() == 2, "transition_source_logits must have shape [N, R]");
    TORCH_CHECK(transition_dest_logits.dim() == 2, "transition_dest_logits must have shape [R, N]");
    TORCH_CHECK(transition_stay_probs.dim() == 1, "transition_stay_probs must have shape [N]");
    TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape [B, N]");
    TORCH_CHECK(
        transition_source_logits.size(0) == local_logits.size(2),
        "transition_source_logits first dim must match local_logits last dim"
    );
    TORCH_CHECK(
        transition_dest_logits.size(1) == local_logits.size(2),
        "transition_dest_logits last dim must match local_logits last dim"
    );
    TORCH_CHECK(
        transition_dest_logits.size(0) == transition_source_logits.size(1),
        "transition_source_logits and transition_dest_logits rank must match"
    );
    TORCH_CHECK(
        transition_stay_probs.size(0) == local_logits.size(2),
        "transition_stay_probs size must match local_logits last dim"
    );
    TORCH_CHECK(
        initial_log_belief.size(0) == local_logits.size(0)
            && initial_log_belief.size(1) == local_logits.size(2),
        "initial_log_belief must have shape [B, N] matching local_logits"
    );
    check_same_cuda_device(transition_source_logits, local_logits, "transition_source_logits");
    check_same_cuda_device(transition_dest_logits, local_logits, "transition_dest_logits");
    check_same_cuda_device(transition_context, local_logits, "transition_context");
    check_same_cuda_device(initial_log_belief, local_logits, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, local_logits, "transition_stay_probs");
    TORCH_CHECK(
        transition_context.scalar_type() == local_logits.scalar_type(),
        "transition_context must match local_logits dtype"
    );
    TORCH_CHECK(transition_gate.numel() == 1, "transition_gate must be a scalar tensor");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (is_optional_tensor_defined(transition_mask)) {
        check_cuda_bool(transition_mask, "transition_mask");
        check_same_cuda_device(transition_mask, local_logits, "transition_mask");
        TORCH_CHECK(
            transition_mask.dim() == 2
                && transition_mask.size(0) == local_logits.size(2)
                && transition_mask.size(1) == local_logits.size(2),
            "transition_mask must have shape [N, N]"
        );
    } else {
        transition_mask = empty_cuda_bool_tensor_like(local_logits);
    }
    if (is_optional_tensor_defined(seq_lens)) {
        check_cuda_int64(seq_lens, "seq_lens");
        check_same_cuda_device(seq_lens, local_logits, "seq_lens");
        TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
        TORCH_CHECK(seq_lens.size(0) == local_logits.size(0), "seq_lens must have shape [B]");
    } else {
        seq_lens = empty_cuda_int64_tensor_like(local_logits);
    }
    if (local_logits.size(0) == 0 || local_logits.size(1) == 0) {
        auto beliefs = torch::empty_like(local_logits);
        auto final_log_belief = torch::empty_like(initial_log_belief);
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }

    const auto num_states = local_logits.size(2);
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, local_logits);
    const bool native_score_filtering = std::isfinite(score_threshold) || score_topk > 0;
    if (is_supported_specialized_num_states(num_states) && is_optional_tensor_defined(transition_mask)) {
        TORCH_CHECK(
            !native_score_filtering,
            "forward_masked_logits native threshold/topk requires the masked tiled custom kernel path"
        );
        return causal_machine_scan_forward_masked_logits_cuda(
            local_logits,
            transition_source_logits,
            transition_dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate_f32,
            transition_stay_probs,
            transition_mask,
            seq_lens,
            chunk_size,
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk);
    }
    TORCH_CHECK(
        num_states > kSpecializedNumStates,
        "forward_masked_logits requires a transition mask and either a specialized small-state kernel or num_states > ",
        kSpecializedNumStates
    );
    const int64_t tile_size = num_states;
    TORCH_CHECK(tile_size > 0, "masked tiled forward requires tile_size > 0");
    const bool custom_kernel_supported = causal_machine_scan_can_use_masked_tiled_forward_kernel_cuda(
        local_logits.get_device(),
        num_states,
        tile_size);
    if (custom_kernel_supported) {
        return causal_machine_scan_forward_masked_tiled_logits_kernel_cuda(
            local_logits,
            transition_source_logits,
            transition_dest_logits,
            transition_context,
            initial_log_belief.to(torch::kFloat32),
            transition_gate_f32,
            transition_stay_probs,
            transition_mask,
            seq_lens,
            chunk_size,
            tile_size,
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk);
    }
    TORCH_CHECK(
        !native_score_filtering,
        "forward_masked_logits native threshold/topk requires the masked tiled custom kernel path"
    );
    auto sparse_meta = build_masked_sparse_fallback_metadata(num_states, transition_mask);
    return causal_machine_scan_forward_sparse_logits_fused(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        sparse_meta.row_ptr,
        sparse_meta.col_idx,
        sparse_meta.dst_idx,
        sparse_meta.src_row_ptr,
        sparse_meta.src_nz_idx,
        sparse_meta.block_mask,
        transition_context,
        initial_log_belief,
        transition_gate_f32,
        transition_stay_probs,
        seq_lens,
        sparse_meta.block_size,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_forward_masked_logits_bound_workspace(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    py::dict workspace,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, local_logits);
    const auto num_states = local_logits.size(2);
    if (num_states <= kSpecializedNumStates) {
        return causal_machine_scan_forward_masked_logits(
            local_logits,
            transition_source_logits,
            transition_dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate_f32,
            transition_stay_probs,
            transition_mask,
            seq_lens,
            chunk_size,
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk);
    }
    const bool custom_kernel_supported = causal_machine_scan_can_use_masked_tiled_forward_kernel_cuda(
        local_logits.get_device(),
        num_states,
        num_states);
    if (!custom_kernel_supported) {
        return causal_machine_scan_forward_masked_logits(
            local_logits,
            transition_source_logits,
            transition_dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate_f32,
            transition_stay_probs,
            transition_mask,
            seq_lens,
            chunk_size,
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk);
    }
    auto work_queue_counter = get_workspace_tensor(workspace, "work_queue_counter", torch::kInt32, local_logits);
    auto masked_transition_tile_cache = get_workspace_tensor(workspace, "masked_transition_tile_cache", torch::kFloat32, local_logits);
    auto filtered_value_cache = get_workspace_tensor(workspace, "filtered_value_cache", torch::kFloat32, local_logits);
    auto row_sums = get_workspace_tensor(workspace, "row_sums", torch::kFloat32, local_logits);
    return causal_machine_scan_forward_masked_tiled_logits_kernel_workspace(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        transition_mask,
        seq_lens,
        chunk_size,
        num_states,
        work_queue_counter,
        masked_transition_tile_cache,
        filtered_value_cache,
        row_sums,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_float32(transition_source_logits, "transition_source_logits");
    check_cuda_float32(transition_dest_logits, "transition_dest_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    TORCH_CHECK(local_logits.dim() == 3, "local_logits must have shape [B, L, N]");
    TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits shape");
    TORCH_CHECK(transition_source_logits.dim() == 2, "transition_source_logits must have shape [N, R]");
    TORCH_CHECK(transition_dest_logits.dim() == 2, "transition_dest_logits must have shape [R, N]");
    TORCH_CHECK(transition_stay_probs.dim() == 1, "transition_stay_probs must have shape [N]");
    TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape [B, N]");
    TORCH_CHECK(
        transition_source_logits.size(0) == local_logits.size(2),
        "transition_source_logits first dim must match local_logits last dim"
    );
    TORCH_CHECK(
        transition_dest_logits.size(1) == local_logits.size(2),
        "transition_dest_logits last dim must match local_logits last dim"
    );
    TORCH_CHECK(
        transition_dest_logits.size(0) == transition_source_logits.size(1),
        "transition_source_logits and transition_dest_logits rank must match"
    );
    TORCH_CHECK(
        transition_stay_probs.size(0) == local_logits.size(2),
        "transition_stay_probs size must match local_logits last dim"
    );
    TORCH_CHECK(
        initial_log_belief.size(0) == local_logits.size(0)
            && initial_log_belief.size(1) == local_logits.size(2),
        "initial_log_belief must have shape [B, N] matching local_logits"
    );
    check_same_cuda_device(transition_source_logits, local_logits, "transition_source_logits");
    check_same_cuda_device(transition_dest_logits, local_logits, "transition_dest_logits");
    check_same_cuda_device(transition_context, local_logits, "transition_context");
    check_same_cuda_device(initial_log_belief, local_logits, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, local_logits, "transition_stay_probs");
    TORCH_CHECK(
        transition_context.scalar_type() == local_logits.scalar_type(),
        "transition_context must match local_logits dtype"
    );
    TORCH_CHECK(transition_gate.numel() == 1, "transition_gate must be a scalar tensor");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(split_size > 0, "split_size must be positive");
    const auto batch_size = local_logits.size(0);
    const auto seq_len = local_logits.size(1);
    const auto num_states = local_logits.size(2);
    const auto transition_rank = transition_source_logits.size(1);
    TORCH_CHECK(num_states > kSpecializedNumStates, "forward_tiled_logits is intended for num_states > 128");
    TORCH_CHECK(
        transition_rank > 0 && transition_rank <= num_states,
        "transition_rank must be in [1, num_states] for forward_tiled_logits"
    );
    if (is_optional_tensor_defined(seq_lens)) {
        check_cuda_int64(seq_lens, "seq_lens");
        check_same_cuda_device(seq_lens, local_logits, "seq_lens");
        TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
        TORCH_CHECK(seq_lens.size(0) == batch_size, "seq_lens must have shape [B]");
    } else {
        seq_lens = empty_cuda_int64_tensor_like(local_logits);
    }
    if (batch_size == 0 || seq_len == 0) {
        auto beliefs = torch::empty_like(local_logits);
        auto final_log_belief = torch::empty_like(initial_log_belief);
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }

    auto prev_log_belief = initial_log_belief.to(torch::kFloat32);
    auto prev_probs = prev_log_belief.exp();
    auto source_probs = transition_source_logits.to(torch::kFloat32);
    auto dest_probs = transition_dest_logits.to(torch::kFloat32);
    auto stay_probs = transition_stay_probs.to(torch::kFloat32).view({1, num_states});
    auto gate = transition_gate.to(local_logits.options().dtype(torch::kFloat32)).reshape({});
    std::vector<torch::Tensor> belief_steps;
    belief_steps.reserve(static_cast<size_t>(seq_len));
    const int64_t num_splits = (transition_rank + split_size - 1) / split_size;
    for (int64_t chunk_start = 0; chunk_start < seq_len; chunk_start += chunk_size) {
        const int64_t chunk_end = std::min<int64_t>(seq_len, chunk_start + chunk_size);
        for (int64_t t = chunk_start; t < chunk_end; ++t) {
            auto local_logits_t = local_logits.select(1, t).to(torch::kFloat32);
            auto transition_context_t = transition_context.select(1, t).to(torch::kFloat32);
            std::vector<torch::Tensor> latent_prob_splits;
            latent_prob_splits.reserve(static_cast<size_t>(num_splits));
            for (int64_t rank_start = 0; rank_start < transition_rank; rank_start += split_size) {
                const int64_t rank_end = std::min<int64_t>(transition_rank, rank_start + split_size);
                latent_prob_splits.push_back(torch::matmul(prev_probs, source_probs.slice(1, rank_start, rank_end)));
            }
            std::vector<torch::Tensor> filtered_tiles;
            filtered_tiles.reserve(static_cast<size_t>((num_states + tile_size - 1) / tile_size));
            torch::Tensor running_max;
            torch::Tensor running_sum;
            for (int64_t state_start = 0; state_start < num_states; state_start += tile_size) {
                const int64_t state_end = std::min<int64_t>(num_states, state_start + tile_size);
                torch::Tensor mix_probs_tile;
                for (int64_t rank_start = 0, split_idx = 0; rank_start < transition_rank; rank_start += split_size, ++split_idx) {
                    const int64_t rank_end = std::min<int64_t>(transition_rank, rank_start + split_size);
                    auto contrib = torch::matmul(
                        latent_prob_splits[static_cast<size_t>(split_idx)],
                        dest_probs.slice(0, rank_start, rank_end).slice(1, state_start, state_end)
                    );
                    mix_probs_tile = mix_probs_tile.defined() ? (mix_probs_tile + contrib) : contrib;
                }
                auto prev_probs_tile = prev_probs.slice(1, state_start, state_end);
                auto stay_tile = stay_probs.slice(1, state_start, state_end);
                auto pred_probs_tile = stay_tile * prev_probs_tile + (1.0f - stay_tile) * mix_probs_tile;
                auto pred_log_tile = pred_probs_tile.clamp_min(1.0e-20).log();
                auto prior_with_context_tile = pred_log_tile + transition_context_t.slice(1, state_start, state_end);
                auto filtered_tile = local_logits_t.slice(1, state_start, state_end) + gate * prior_with_context_tile;
                filtered_tiles.push_back(filtered_tile);
                auto tile_max = std::get<0>(filtered_tile.max(-1, true));
                auto tile_sum = torch::exp(filtered_tile - tile_max).sum(-1, true);
                if (!running_max.defined() || !running_sum.defined()) {
                    running_max = tile_max;
                    running_sum = tile_sum;
                } else {
                    auto new_max = torch::maximum(running_max, tile_max);
                    running_sum = running_sum * torch::exp(running_max - new_max)
                        + tile_sum * torch::exp(tile_max - new_max);
                    running_max = new_max;
                }
            }
            auto log_norm = running_max + running_sum.clamp_min(1.0e-20).log();
            auto next_log_belief = torch::cat(filtered_tiles, -1) - log_norm;
            if (is_optional_tensor_defined(seq_lens)) {
                auto active = seq_lens.gt(t).view({batch_size, 1});
                next_log_belief = torch::where(active, next_log_belief, prev_log_belief);
            }
            belief_steps.push_back(next_log_belief.to(local_logits.scalar_type()));
            prev_log_belief = next_log_belief;
            prev_probs = prev_log_belief.exp();
        }
    }
    auto beliefs = belief_steps.empty()
        ? torch::empty_like(local_logits)
        : torch::stack(belief_steps, 1);
    return {
        beliefs,
        prev_log_belief.to(local_logits.scalar_type()),
    };
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_float32(transition_source_probs, "transition_source_probs");
    check_cuda_float32(transition_dest_probs, "transition_dest_probs");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_float32(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    TORCH_CHECK(local_logits.dim() == 3, "local_logits must have shape [B, L, N]");
    TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits shape");
    TORCH_CHECK(transition_source_probs.dim() == 2, "transition_source_probs must have shape [N, R]");
    TORCH_CHECK(transition_dest_probs.dim() == 2, "transition_dest_probs must have shape [R, N]");
    TORCH_CHECK(transition_stay_probs.dim() == 1, "transition_stay_probs must have shape [N]");
    TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape [B, N]");
    TORCH_CHECK(
        transition_source_probs.size(0) == local_logits.size(2),
        "transition_source_probs first dim must match local_logits last dim"
    );
    TORCH_CHECK(
        transition_dest_probs.size(1) == local_logits.size(2),
        "transition_dest_probs last dim must match local_logits last dim"
    );
    TORCH_CHECK(
        transition_dest_probs.size(0) == transition_source_probs.size(1),
        "transition_source_probs and transition_dest_probs rank must match"
    );
    TORCH_CHECK(
        transition_stay_probs.size(0) == local_logits.size(2),
        "transition_stay_probs size must match local_logits last dim"
    );
    TORCH_CHECK(
        initial_log_belief.size(0) == local_logits.size(0)
            && initial_log_belief.size(1) == local_logits.size(2),
        "initial_log_belief must have shape [B, N] matching local_logits"
    );
    check_same_cuda_device(transition_source_probs, local_logits, "transition_source_probs");
    check_same_cuda_device(transition_dest_probs, local_logits, "transition_dest_probs");
    check_same_cuda_device(transition_context, local_logits, "transition_context");
    check_same_cuda_device(initial_log_belief, local_logits, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, local_logits, "transition_stay_probs");
    TORCH_CHECK(
        transition_context.scalar_type() == local_logits.scalar_type(),
        "transition_context must match local_logits dtype"
    );
    TORCH_CHECK(transition_gate.numel() == 1, "transition_gate must be a scalar tensor");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(split_size > 0, "split_size must be positive");
    if (is_optional_tensor_defined(seq_lens)) {
        check_cuda_int64(seq_lens, "seq_lens");
        check_same_cuda_device(seq_lens, local_logits, "seq_lens");
        TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
        TORCH_CHECK(seq_lens.size(0) == local_logits.size(0), "seq_lens must have shape [B]");
    } else {
        seq_lens = empty_cuda_int64_tensor_like(local_logits);
    }
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, local_logits);
    return causal_machine_scan_forward_tiled_logits_kernel_cuda(
        local_logits,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        transition_gate_f32,
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

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_workspace(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
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
    check_cuda_int32(work_queue_counter, "work_queue_counter");
    check_cuda_float32(filtered_value_cache, "filtered_value_cache");
    check_same_cuda_device(work_queue_counter, local_logits, "work_queue_counter");
    check_same_cuda_device(filtered_value_cache, local_logits, "filtered_value_cache");
    auto workspace_info = causal_machine_scan_describe_workspace_config(
        "tiled_forward",
        local_logits.size(2),
        transition_source_probs.size(1),
        local_logits.size(0),
        tile_size,
        split_size,
        local_logits.size(1),
        chunk_size,
        local_logits.get_device());
    check_workspace_shape_min(
        work_queue_counter,
        workspace_info["work_queue_counter_shape"].cast<std::vector<int64_t>>(),
        "work_queue_counter");
    check_workspace_shape_min(
        filtered_value_cache,
        workspace_info["filtered_value_cache_shape"].cast<std::vector<int64_t>>(),
        "filtered_value_cache");
    return causal_machine_scan_forward_tiled_logits_kernel_workspace_cuda(
        local_logits,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        normalize_transition_gate_tensor(transition_gate, local_logits),
        transition_stay_probs,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        filtered_value_cache,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_logits_kernel_bound_workspace(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    py::dict workspace,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    auto work_queue_counter = get_workspace_tensor(workspace, "work_queue_counter", torch::kInt32, local_logits);
    auto filtered_value_cache = get_workspace_tensor(workspace, "filtered_value_cache", torch::kFloat32, local_logits);
    return causal_machine_scan_forward_tiled_logits_kernel_workspace(
        local_logits,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        filtered_value_cache,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_quantized_kernel_workspace(
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
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
    check_cuda_int32(work_queue_counter, "work_queue_counter");
    check_cuda_float32(filtered_value_cache, "filtered_value_cache");
    check_same_cuda_device(work_queue_counter, local_logits, "work_queue_counter");
    check_same_cuda_device(filtered_value_cache, local_logits, "filtered_value_cache");
    auto workspace_info = causal_machine_scan_describe_workspace_config(
        "tiled_forward",
        local_logits.size(2),
        transition_source_q.size(1),
        local_logits.size(0),
        tile_size,
        split_size,
        local_logits.size(1),
        chunk_size,
        local_logits.get_device());
    check_workspace_shape_min(
        work_queue_counter,
        workspace_info["work_queue_counter_shape"].cast<std::vector<int64_t>>(),
        "work_queue_counter");
    check_workspace_shape_min(
        filtered_value_cache,
        workspace_info["filtered_value_cache_shape"].cast<std::vector<int64_t>>(),
        "filtered_value_cache");
    return causal_machine_scan_forward_tiled_quantized_kernel_workspace_cuda(
        local_logits,
        transition_source_q,
        transition_source_scales,
        transition_dest_q,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        normalize_transition_gate_tensor(transition_gate, local_logits),
        transition_stay_probs,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        filtered_value_cache,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_quantized_kernel_bound_workspace(
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    py::dict workspace,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    auto work_queue_counter = get_workspace_tensor(workspace, "work_queue_counter", torch::kInt32, local_logits);
    auto filtered_value_cache = get_workspace_tensor(workspace, "filtered_value_cache", torch::kFloat32, local_logits);
    return causal_machine_scan_forward_tiled_quantized_kernel_workspace(
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
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        filtered_value_cache,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_fp8_kernel_workspace(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
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
    check_cuda_int32(work_queue_counter, "work_queue_counter");
    check_cuda_float32(filtered_value_cache, "filtered_value_cache");
    check_same_cuda_device(work_queue_counter, local_logits, "work_queue_counter");
    check_same_cuda_device(filtered_value_cache, local_logits, "filtered_value_cache");
    auto workspace_info = causal_machine_scan_describe_workspace_config(
        "tiled_forward",
        local_logits.size(2),
        transition_source_packed.size(1),
        local_logits.size(0),
        tile_size,
        split_size,
        local_logits.size(1),
        chunk_size,
        local_logits.get_device());
    check_workspace_shape_min(
        work_queue_counter,
        workspace_info["work_queue_counter_shape"].cast<std::vector<int64_t>>(),
        "work_queue_counter");
    check_workspace_shape_min(
        filtered_value_cache,
        workspace_info["filtered_value_cache_shape"].cast<std::vector<int64_t>>(),
        "filtered_value_cache");
    return causal_machine_scan_forward_tiled_fp8_kernel_workspace_cuda(
        local_logits,
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        normalize_transition_gate_tensor(transition_gate, local_logits),
        transition_stay_probs,
        fp8_format,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        filtered_value_cache,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_forward_tiled_fp8_kernel_bound_workspace(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    py::dict workspace,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    auto work_queue_counter = get_workspace_tensor(workspace, "work_queue_counter", torch::kInt32, local_logits);
    auto filtered_value_cache = get_workspace_tensor(workspace, "filtered_value_cache", torch::kFloat32, local_logits);
    return causal_machine_scan_forward_tiled_fp8_kernel_workspace(
        local_logits,
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        fp8_format,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        filtered_value_cache,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_forward_masked_tiled_logits_kernel_workspace(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
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
    check_cuda_int32(work_queue_counter, "work_queue_counter");
    check_cuda_float32(masked_transition_tile_cache, "masked_transition_tile_cache");
    check_cuda_float32(filtered_value_cache, "filtered_value_cache");
    check_cuda_float32(row_sums, "row_sums");
    check_same_cuda_device(work_queue_counter, local_logits, "work_queue_counter");
    check_same_cuda_device(masked_transition_tile_cache, local_logits, "masked_transition_tile_cache");
    check_same_cuda_device(filtered_value_cache, local_logits, "filtered_value_cache");
    check_same_cuda_device(row_sums, local_logits, "row_sums");
    auto workspace_info = causal_machine_scan_describe_workspace_config(
        "masked_tiled_forward",
        local_logits.size(2),
        transition_source_logits.size(1),
        local_logits.size(0),
        tile_size,
        1,
        local_logits.size(1),
        chunk_size,
        local_logits.get_device());
    check_workspace_shape_min(
        work_queue_counter,
        workspace_info["work_queue_counter_shape"].cast<std::vector<int64_t>>(),
        "work_queue_counter");
    check_workspace_shape_min(
        masked_transition_tile_cache,
        workspace_info["masked_transition_tile_cache_shape"].cast<std::vector<int64_t>>(),
        "masked_transition_tile_cache");
    check_workspace_shape_min(
        filtered_value_cache,
        workspace_info["filtered_value_cache_shape"].cast<std::vector<int64_t>>(),
        "filtered_value_cache");
    check_workspace_shape_min(
        row_sums,
        workspace_info["row_sums_shape"].cast<std::vector<int64_t>>(),
        "row_sums");
    return causal_machine_scan_forward_masked_tiled_logits_kernel_workspace_cuda(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        normalize_transition_gate_tensor(transition_gate, local_logits),
        transition_stay_probs,
        transition_mask,
        seq_lens,
        chunk_size,
        tile_size,
        work_queue_counter,
        masked_transition_tile_cache,
        filtered_value_cache,
        row_sums,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_forward_sparse_logits(
    torch::Tensor local_logits,
    torch::Tensor transition_blocks,
    torch::Tensor block_row_ptr,
    torch::Tensor block_col_idx,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_float32(transition_blocks, "transition_blocks");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_cuda_int32(block_row_ptr, "block_row_ptr");
    check_cuda_int32(block_col_idx, "block_col_idx");
    TORCH_CHECK(local_logits.dim() == 3, "local_logits must have shape [B, L, N]");
    TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits shape");
    TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape [B, N]");
    TORCH_CHECK(
        initial_log_belief.size(0) == local_logits.size(0)
            && initial_log_belief.size(1) == local_logits.size(2),
        "initial_log_belief must have shape [B, N] matching local_logits"
    );
    TORCH_CHECK(
        transition_stay_probs.dim() == 1 && transition_stay_probs.size(0) == local_logits.size(2),
        "transition_stay_probs must have shape [N]"
    );
    TORCH_CHECK(transition_blocks.dim() == 3, "transition_blocks must have shape [nnz_blocks, block_size, block_size]");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(
        transition_blocks.size(1) == block_size && transition_blocks.size(2) == block_size,
        "transition_blocks trailing dims must equal block_size"
    );
    TORCH_CHECK(block_row_ptr.dim() == 1, "block_row_ptr must be 1D");
    TORCH_CHECK(block_col_idx.dim() == 1, "block_col_idx must be 1D");
    TORCH_CHECK(
        block_col_idx.size(0) == transition_blocks.size(0),
        "block_col_idx length must match transition_blocks nnz"
    );
    TORCH_CHECK(
        block_row_ptr.size(0) == ((local_logits.size(2) + block_size - 1) / block_size) + 1,
        "block_row_ptr must have length ceil_div(num_states, block_size) + 1"
    );
    TORCH_CHECK(transition_gate.numel() == 1, "transition_gate must be a scalar tensor");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    check_same_cuda_device(transition_blocks, local_logits, "transition_blocks");
    check_same_cuda_device(block_row_ptr, local_logits, "block_row_ptr");
    check_same_cuda_device(block_col_idx, local_logits, "block_col_idx");
    check_same_cuda_device(transition_context, local_logits, "transition_context");
    check_same_cuda_device(initial_log_belief, local_logits, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, local_logits, "transition_stay_probs");
    TORCH_CHECK(
        transition_context.scalar_type() == local_logits.scalar_type(),
        "transition_context must match local_logits dtype"
    );
    TORCH_CHECK(
        initial_log_belief.scalar_type() == local_logits.scalar_type(),
        "initial_log_belief must match local_logits dtype"
    );
    if (is_optional_tensor_defined(seq_lens)) {
        check_cuda_int64(seq_lens, "seq_lens");
        check_same_cuda_device(seq_lens, local_logits, "seq_lens");
        TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == local_logits.size(0), "seq_lens must have shape [B]");
    } else {
        seq_lens = empty_cuda_int64_tensor_like(local_logits);
    }
    return causal_machine_scan_forward_sparse_cuda(
        local_logits,
        transition_blocks,
        block_row_ptr,
        block_col_idx,
        transition_context,
        initial_log_belief,
        normalize_transition_gate_tensor(transition_gate, local_logits),
        transition_stay_probs,
        seq_lens,
        block_size,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_forward_sparse_logits_fused(
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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_float32(transition_source_logits, "transition_source_logits");
    check_cuda_float32(transition_dest_logits, "transition_dest_logits");
    check_cuda_int32(block_row_ptr, "block_row_ptr");
    check_cuda_int32(block_col_idx, "block_col_idx");
    check_cuda_int32(block_dst_idx, "block_dst_idx");
    check_cuda_int32(src_row_ptr, "src_row_ptr");
    check_cuda_int32(src_nz_idx, "src_nz_idx");
    check_cuda_float32(block_mask, "block_mask");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    TORCH_CHECK(transition_source_logits.dim() == 2, "transition_source_logits must have shape [N, R]");
    TORCH_CHECK(transition_dest_logits.dim() == 2, "transition_dest_logits must have shape [R, N]");
    TORCH_CHECK(
        transition_source_logits.size(0) == local_logits.size(2),
        "transition_source_logits first dim must match num_states");
    TORCH_CHECK(
        transition_dest_logits.size(1) == local_logits.size(2),
        "transition_dest_logits second dim must match num_states");
    TORCH_CHECK(
        transition_source_logits.size(1) == transition_dest_logits.size(0),
        "transition_source_logits second dim must match transition_dest_logits first dim");
    TORCH_CHECK(block_mask.dim() == 3, "block_mask must have shape [nnz_blocks, block_size, block_size]");
    TORCH_CHECK(src_row_ptr.dim() == 1, "src_row_ptr must be 1D");
    TORCH_CHECK(src_nz_idx.dim() == 1, "src_nz_idx must be 1D");
    TORCH_CHECK(
        block_mask.size(0) == block_col_idx.size(0)
            && block_mask.size(0) == block_dst_idx.size(0),
        "block_mask nnz dim must match sparse block indices");
    TORCH_CHECK(
        src_row_ptr.size(0) == block_row_ptr.size(0),
        "src_row_ptr must have the same block-count coverage as block_row_ptr");
    TORCH_CHECK(
        src_nz_idx.size(0) == block_col_idx.size(0),
        "src_nz_idx length must match sparse block nnz");
    check_same_cuda_device(transition_source_logits, local_logits, "transition_source_logits");
    check_same_cuda_device(transition_dest_logits, local_logits, "transition_dest_logits");
    check_same_cuda_device(block_row_ptr, local_logits, "block_row_ptr");
    check_same_cuda_device(block_col_idx, local_logits, "block_col_idx");
    check_same_cuda_device(block_dst_idx, local_logits, "block_dst_idx");
    check_same_cuda_device(src_row_ptr, local_logits, "src_row_ptr");
    check_same_cuda_device(src_nz_idx, local_logits, "src_nz_idx");
    check_same_cuda_device(block_mask, local_logits, "block_mask");
    check_same_cuda_device(transition_context, local_logits, "transition_context");
    check_same_cuda_device(initial_log_belief, local_logits, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, local_logits, "transition_stay_probs");
    if (is_optional_tensor_defined(seq_lens)) {
        check_cuda_int64(seq_lens, "seq_lens");
        check_same_cuda_device(seq_lens, local_logits, "seq_lens");
    } else {
        seq_lens = empty_cuda_int64_tensor_like(local_logits);
    }
    return causal_machine_scan_forward_sparse_logits_cuda(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        block_row_ptr,
        block_col_idx,
        block_dst_idx,
        src_row_ptr,
        src_nz_idx,
        block_mask,
        transition_context,
        initial_log_belief,
        normalize_transition_gate_tensor(transition_gate, local_logits),
        transition_stay_probs,
        seq_lens,
        block_size,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_backward_sparse(
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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_final_belief, "grad_final_belief");
    check_cuda_float32(transition_source_probs, "transition_source_probs");
    check_cuda_float32(transition_dest_probs, "transition_dest_probs");
    check_cuda_float32(transition_blocks, "transition_blocks");
    check_cuda_int32(block_row_ptr, "block_row_ptr");
    check_cuda_int32(block_col_idx, "block_col_idx");
    check_cuda_int32(block_dst_idx, "block_dst_idx");
    check_cuda_int32(src_row_ptr, "src_row_ptr");
    check_cuda_int32(src_nz_idx, "src_nz_idx");
    check_cuda_int32(grouped_src_row_ptr, "grouped_src_row_ptr");
    check_cuda_int32(grouped_src_block_idx, "grouped_src_block_idx");
    check_cuda_float32(row_sums, "row_sums");
    check_cuda_float32(block_mask, "block_mask");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    TORCH_CHECK(beliefs.dim() == 3, "beliefs must have shape [B, L, N]");
    TORCH_CHECK(transition_context.sizes() == beliefs.sizes(), "transition_context must match beliefs shape");
    TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape [B, N]");
    TORCH_CHECK(
        initial_log_belief.size(0) == beliefs.size(0)
            && initial_log_belief.size(1) == beliefs.size(2),
        "initial_log_belief must have shape [B, N] matching beliefs"
    );
    TORCH_CHECK(
        transition_stay_probs.dim() == 1 && transition_stay_probs.size(0) == beliefs.size(2),
        "transition_stay_probs must have shape [N]"
    );
    TORCH_CHECK(
        transition_source_probs.dim() == 2,
        "transition_source_probs must have shape [N, R]"
    );
    TORCH_CHECK(
        transition_dest_probs.dim() == 2,
        "transition_dest_probs must have shape [R, N]"
    );
    TORCH_CHECK(
        transition_source_probs.size(0) == beliefs.size(2),
        "transition_source_probs first dim must match beliefs last dim"
    );
    TORCH_CHECK(
        transition_dest_probs.size(1) == beliefs.size(2),
        "transition_dest_probs last dim must match beliefs last dim"
    );
    TORCH_CHECK(
        transition_dest_probs.size(0) == transition_source_probs.size(1),
        "transition_source_probs and transition_dest_probs rank must match"
    );
    TORCH_CHECK(transition_blocks.dim() == 3, "transition_blocks must have shape [nnz_blocks, block_size, block_size]");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(
        transition_blocks.size(1) == block_size && transition_blocks.size(2) == block_size,
        "transition_blocks trailing dims must equal block_size"
    );
    const int64_t num_state_blocks = (beliefs.size(2) + block_size - 1) / block_size;
    const int64_t grouped_src_group_count = grouped_src_block_idx.size(0);
    TORCH_CHECK(block_row_ptr.dim() == 1, "block_row_ptr must be 1D");
    TORCH_CHECK(block_col_idx.dim() == 1, "block_col_idx must be 1D");
    TORCH_CHECK(block_dst_idx.dim() == 1, "block_dst_idx must be 1D");
    TORCH_CHECK(src_row_ptr.dim() == 1, "src_row_ptr must be 1D");
    TORCH_CHECK(src_nz_idx.dim() == 1, "src_nz_idx must be 1D");
    TORCH_CHECK(grouped_src_row_ptr.dim() == 1, "grouped_src_row_ptr must be 1D");
    TORCH_CHECK(grouped_src_block_idx.dim() == 1, "grouped_src_block_idx must be 1D");
    TORCH_CHECK(
        block_row_ptr.size(0) == num_state_blocks + 1,
        "block_row_ptr must have length ceil_div(num_states, block_size) + 1"
    );
    TORCH_CHECK(
        src_row_ptr.size(0) == num_state_blocks + 1,
        "src_row_ptr must have length ceil_div(num_states, block_size) + 1"
    );
    TORCH_CHECK(
        block_col_idx.size(0) == transition_blocks.size(0),
        "block_col_idx length must match transition_blocks nnz"
    );
    TORCH_CHECK(
        block_dst_idx.size(0) == transition_blocks.size(0),
        "block_dst_idx length must match transition_blocks nnz"
    );
    TORCH_CHECK(
        src_nz_idx.size(0) == transition_blocks.size(0),
        "src_nz_idx length must match transition_blocks nnz"
    );
    TORCH_CHECK(
        grouped_src_group_count == 0 || grouped_src_row_ptr.size(0) == grouped_src_group_count + 1,
        "grouped_src_row_ptr must have length grouped_src_block_idx.size(0) + 1 when grouped metadata is provided"
    );
    TORCH_CHECK(
        row_sums.dim() == 1 && row_sums.size(0) >= beliefs.size(2),
        "row_sums must be 1D and cover at least num_states entries"
    );
    TORCH_CHECK(
        block_mask.sizes() == transition_blocks.sizes(),
        "block_mask must match transition_blocks shape"
    );
    TORCH_CHECK(grad_beliefs.sizes() == beliefs.sizes(), "grad_beliefs must match beliefs shape");
    TORCH_CHECK(grad_final_belief.sizes() == initial_log_belief.sizes(), "grad_final_belief must match initial_log_belief shape");
    TORCH_CHECK(transition_gate.numel() == 1, "transition_gate must be a scalar tensor");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    check_same_cuda_device(grad_final_belief, beliefs, "grad_final_belief");
    check_same_cuda_device(transition_source_probs, beliefs, "transition_source_probs");
    check_same_cuda_device(transition_dest_probs, beliefs, "transition_dest_probs");
    check_same_cuda_device(transition_blocks, beliefs, "transition_blocks");
    check_same_cuda_device(block_row_ptr, beliefs, "block_row_ptr");
    check_same_cuda_device(block_col_idx, beliefs, "block_col_idx");
    check_same_cuda_device(block_dst_idx, beliefs, "block_dst_idx");
    check_same_cuda_device(src_row_ptr, beliefs, "src_row_ptr");
    check_same_cuda_device(src_nz_idx, beliefs, "src_nz_idx");
    check_same_cuda_device(grouped_src_row_ptr, beliefs, "grouped_src_row_ptr");
    check_same_cuda_device(grouped_src_block_idx, beliefs, "grouped_src_block_idx");
    check_same_cuda_device(row_sums, beliefs, "row_sums");
    check_same_cuda_device(block_mask, beliefs, "block_mask");
    check_same_cuda_device(transition_context, beliefs, "transition_context");
    check_same_cuda_device(initial_log_belief, beliefs, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, beliefs, "transition_stay_probs");
    TORCH_CHECK(
        grad_beliefs.scalar_type() == beliefs.scalar_type(),
        "grad_beliefs must match beliefs dtype"
    );
    TORCH_CHECK(
        grad_final_belief.scalar_type() == beliefs.scalar_type(),
        "grad_final_belief must match beliefs dtype"
    );
    TORCH_CHECK(
        transition_context.scalar_type() == beliefs.scalar_type(),
        "transition_context must match beliefs dtype"
    );
    TORCH_CHECK(
        initial_log_belief.scalar_type() == beliefs.scalar_type(),
        "initial_log_belief must match beliefs dtype"
    );
    if (is_optional_tensor_defined(seq_lens)) {
        check_cuda_int64(seq_lens, "seq_lens");
        check_same_cuda_device(seq_lens, beliefs, "seq_lens");
        TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == beliefs.size(0), "seq_lens must have shape [B]");
    } else {
        seq_lens = empty_cuda_int64_tensor_like(beliefs);
    }
    return causal_machine_scan_backward_sparse_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_probs,
        transition_dest_probs,
        transition_blocks,
        block_row_ptr,
        block_col_idx,
        block_dst_idx,
        src_row_ptr,
        src_nz_idx,
        grouped_src_row_ptr,
        grouped_src_block_idx,
        row_sums,
        block_mask,
        transition_context,
        initial_log_belief,
        beliefs,
        normalize_transition_gate_tensor(transition_gate, beliefs),
        transition_stay_probs,
        seq_lens,
        block_size,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_backward_sparse_logits_fused(
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
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t chunk_size) {
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_final_belief, "grad_final_belief");
    check_cuda_float32(transition_source_logits, "transition_source_logits");
    check_cuda_float32(transition_dest_logits, "transition_dest_logits");
    check_cuda_int32(block_row_ptr, "block_row_ptr");
    check_cuda_int32(block_col_idx, "block_col_idx");
    check_cuda_int32(block_dst_idx, "block_dst_idx");
    check_cuda_int32(src_row_ptr, "src_row_ptr");
    check_cuda_int32(src_nz_idx, "src_nz_idx");
    check_cuda_int32(grouped_src_row_ptr, "grouped_src_row_ptr");
    check_cuda_int32(grouped_src_block_idx, "grouped_src_block_idx");
    check_cuda_float32(block_mask, "block_mask");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_same_cuda_device(transition_source_logits, beliefs, "transition_source_logits");
    check_same_cuda_device(transition_dest_logits, beliefs, "transition_dest_logits");
    check_same_cuda_device(block_row_ptr, beliefs, "block_row_ptr");
    check_same_cuda_device(block_col_idx, beliefs, "block_col_idx");
    check_same_cuda_device(block_dst_idx, beliefs, "block_dst_idx");
    check_same_cuda_device(src_row_ptr, beliefs, "src_row_ptr");
    check_same_cuda_device(src_nz_idx, beliefs, "src_nz_idx");
    check_same_cuda_device(grouped_src_row_ptr, beliefs, "grouped_src_row_ptr");
    check_same_cuda_device(grouped_src_block_idx, beliefs, "grouped_src_block_idx");
    check_same_cuda_device(block_mask, beliefs, "block_mask");
    check_same_cuda_device(transition_context, beliefs, "transition_context");
    check_same_cuda_device(initial_log_belief, beliefs, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, beliefs, "transition_stay_probs");
    if (is_optional_tensor_defined(seq_lens)) {
        check_cuda_int64(seq_lens, "seq_lens");
        check_same_cuda_device(seq_lens, beliefs, "seq_lens");
    } else {
        seq_lens = empty_cuda_int64_tensor_like(beliefs);
    }
    return causal_machine_scan_backward_sparse_logits_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_logits,
        transition_dest_logits,
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
        normalize_transition_gate_tensor(transition_gate, beliefs),
        transition_stay_probs,
        seq_lens,
        block_size,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks(
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size) {
    check_cuda_float32(transition_source_probs, "transition_source_probs");
    check_cuda_float32(transition_dest_probs, "transition_dest_probs");
    check_cuda_int32(block_col_idx, "block_col_idx");
    check_cuda_int32(block_dst_idx, "block_dst_idx");
    check_cuda_float32(block_mask, "block_mask");
    TORCH_CHECK(
        transition_source_probs.dim() == 2,
        "transition_source_probs must have shape [N, R]"
    );
    TORCH_CHECK(
        transition_dest_probs.dim() == 2,
        "transition_dest_probs must have shape [R, N]"
    );
    TORCH_CHECK(
        transition_dest_probs.size(0) == transition_source_probs.size(1),
        "transition_source_probs and transition_dest_probs rank must match"
    );
    TORCH_CHECK(block_col_idx.dim() == 1, "block_col_idx must be 1D");
    TORCH_CHECK(block_dst_idx.dim() == 1, "block_dst_idx must be 1D");
    TORCH_CHECK(
        block_col_idx.size(0) == block_dst_idx.size(0),
        "block_col_idx and block_dst_idx must have the same length"
    );
    TORCH_CHECK(block_mask.dim() == 3, "block_mask must have shape [nnz_blocks, block_size, block_size]");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(
        block_mask.size(0) == block_col_idx.size(0),
        "block_mask leading dim must match block metadata nnz"
    );
    TORCH_CHECK(
        block_mask.size(1) == block_size && block_mask.size(2) == block_size,
        "block_mask trailing dims must equal block_size"
    );
    TORCH_CHECK(
        padded_states >= transition_source_probs.size(0),
        "padded_states must be at least the number of source states"
    );
    TORCH_CHECK(
        padded_states >= transition_dest_probs.size(1),
        "padded_states must be at least the number of destination states"
    );
    TORCH_CHECK(
        padded_states % block_size == 0,
        "padded_states must be divisible by block_size"
    );
    check_same_cuda_device(transition_dest_probs, transition_source_probs, "transition_dest_probs");
    check_same_cuda_device(block_col_idx, transition_source_probs, "block_col_idx");
    check_same_cuda_device(block_dst_idx, transition_source_probs, "block_dst_idx");
    check_same_cuda_device(block_mask, transition_source_probs, "block_mask");
    return causal_machine_scan_materialize_sparse_blocks_cuda(
        transition_source_probs,
        transition_dest_probs,
        block_col_idx,
        block_dst_idx,
        block_mask,
        padded_states,
        block_size);
}

std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks_int8(
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor block_col_idx,
    torch::Tensor block_dst_idx,
    torch::Tensor block_mask,
    int64_t padded_states,
    int64_t block_size) {
    check_cuda_int8(transition_source_packed, "transition_source_packed");
    check_cuda_float32(transition_source_scales, "transition_source_scales");
    check_cuda_int8(transition_dest_packed, "transition_dest_packed");
    check_cuda_float32(transition_dest_scales, "transition_dest_scales");
    check_cuda_int32(block_col_idx, "block_col_idx");
    check_cuda_int32(block_dst_idx, "block_dst_idx");
    check_cuda_float32(block_mask, "block_mask");
    TORCH_CHECK(transition_source_packed.dim() == 2, "transition_source_packed must have shape [N, R]");
    TORCH_CHECK(transition_dest_packed.dim() == 2, "transition_dest_packed must have shape [R, N]");
    TORCH_CHECK(
        transition_source_scales.dim() == 1 && transition_source_scales.size(0) == transition_source_packed.size(0),
        "transition_source_scales must have shape [N]"
    );
    TORCH_CHECK(
        transition_dest_scales.dim() == 1 && transition_dest_scales.size(0) == transition_dest_packed.size(0),
        "transition_dest_scales must have shape [R]"
    );
    TORCH_CHECK(
        transition_dest_packed.size(0) == transition_source_packed.size(1),
        "transition_source_packed and transition_dest_packed rank must match"
    );
    TORCH_CHECK(block_col_idx.dim() == 1, "block_col_idx must be 1D");
    TORCH_CHECK(block_dst_idx.dim() == 1, "block_dst_idx must be 1D");
    TORCH_CHECK(block_col_idx.size(0) == block_dst_idx.size(0), "block indices must have the same length");
    TORCH_CHECK(block_mask.dim() == 3, "block_mask must have shape [nnz_blocks, block_size, block_size]");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(block_mask.size(0) == block_col_idx.size(0), "block_mask leading dim must match block metadata nnz");
    TORCH_CHECK(
        block_mask.size(1) == block_size && block_mask.size(2) == block_size,
        "block_mask trailing dims must equal block_size"
    );
    TORCH_CHECK(padded_states >= transition_source_packed.size(0), "padded_states must be at least the number of source states");
    TORCH_CHECK(padded_states >= transition_dest_packed.size(1), "padded_states must be at least the number of destination states");
    TORCH_CHECK(padded_states % block_size == 0, "padded_states must be divisible by block_size");
    check_same_cuda_device(transition_source_scales, transition_source_packed, "transition_source_scales");
    check_same_cuda_device(transition_dest_packed, transition_source_packed, "transition_dest_packed");
    check_same_cuda_device(transition_dest_scales, transition_source_packed, "transition_dest_scales");
    check_same_cuda_device(block_col_idx, transition_source_packed, "block_col_idx");
    check_same_cuda_device(block_dst_idx, transition_source_packed, "block_dst_idx");
    check_same_cuda_device(block_mask, transition_source_packed, "block_mask");
    return causal_machine_scan_materialize_sparse_blocks_int8_cuda(
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

std::vector<torch::Tensor> causal_machine_scan_materialize_sparse_blocks_fp8(
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
    check_cuda_uint8(transition_source_packed, "transition_source_packed");
    check_cuda_float32(transition_source_scales, "transition_source_scales");
    check_cuda_uint8(transition_dest_packed, "transition_dest_packed");
    check_cuda_float32(transition_dest_scales, "transition_dest_scales");
    check_cuda_int32(block_col_idx, "block_col_idx");
    check_cuda_int32(block_dst_idx, "block_dst_idx");
    check_cuda_float32(block_mask, "block_mask");
    TORCH_CHECK(transition_source_packed.dim() == 2, "transition_source_packed must have shape [N, R]");
    TORCH_CHECK(transition_dest_packed.dim() == 2, "transition_dest_packed must have shape [R, N]");
    TORCH_CHECK(
        transition_source_scales.dim() == 1 && transition_source_scales.size(0) == transition_source_packed.size(0),
        "transition_source_scales must have shape [N]"
    );
    TORCH_CHECK(
        transition_dest_scales.dim() == 1 && transition_dest_scales.size(0) == transition_dest_packed.size(0),
        "transition_dest_scales must have shape [R]"
    );
    TORCH_CHECK(
        transition_dest_packed.size(0) == transition_source_packed.size(1),
        "transition_source_packed and transition_dest_packed rank must match"
    );
    TORCH_CHECK(block_col_idx.dim() == 1, "block_col_idx must be 1D");
    TORCH_CHECK(block_dst_idx.dim() == 1, "block_dst_idx must be 1D");
    TORCH_CHECK(block_col_idx.size(0) == block_dst_idx.size(0), "block indices must have the same length");
    TORCH_CHECK(block_mask.dim() == 3, "block_mask must have shape [nnz_blocks, block_size, block_size]");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(block_mask.size(0) == block_col_idx.size(0), "block_mask leading dim must match block metadata nnz");
    TORCH_CHECK(
        block_mask.size(1) == block_size && block_mask.size(2) == block_size,
        "block_mask trailing dims must equal block_size"
    );
    TORCH_CHECK(padded_states >= transition_source_packed.size(0), "padded_states must be at least the number of source states");
    TORCH_CHECK(padded_states >= transition_dest_packed.size(1), "padded_states must be at least the number of destination states");
    TORCH_CHECK(padded_states % block_size == 0, "padded_states must be divisible by block_size");
    TORCH_CHECK(fp8_format == 0 || fp8_format == 1, "fp8_format must be 0 (e4m3) or 1 (e5m2)");
    check_same_cuda_device(transition_source_scales, transition_source_packed, "transition_source_scales");
    check_same_cuda_device(transition_dest_packed, transition_source_packed, "transition_dest_packed");
    check_same_cuda_device(transition_dest_scales, transition_source_packed, "transition_dest_scales");
    check_same_cuda_device(block_col_idx, transition_source_packed, "block_col_idx");
    check_same_cuda_device(block_dst_idx, transition_source_packed, "block_dst_idx");
    check_same_cuda_device(block_mask, transition_source_packed, "block_mask");
    return causal_machine_scan_materialize_sparse_blocks_fp8_cuda(
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        block_col_idx,
        block_dst_idx,
        block_mask,
        fp8_format,
        padded_states,
        block_size);
}

std::vector<torch::Tensor> causal_machine_scan_build_sparse_metadata(
    torch::Tensor transition_mask,
    int64_t block_size) {
    check_cuda_bool(transition_mask, "transition_mask");
    TORCH_CHECK(transition_mask.dim() == 2, "transition_mask must have shape [N, N]");
    TORCH_CHECK(
        transition_mask.size(0) == transition_mask.size(1),
        "transition_mask must be square"
    );
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(
        transition_mask.size(0) % block_size == 0,
        "transition_mask size must be divisible by block_size"
    );
    const auto padded_states = transition_mask.size(0);
    const auto num_state_blocks = padded_states / block_size;
    auto mask_blocks = transition_mask.view({num_state_blocks, block_size, num_state_blocks, block_size})
        .permute({0, 2, 1, 3})
        .contiguous();
    auto block_active = mask_blocks.any(3).any(2);
    auto dst_major_active = block_active.transpose(0, 1).contiguous();
    auto counts_per_dst = block_active.to(torch::kInt32).sum(0, false, torch::kInt32);
    auto counts_per_src = block_active.to(torch::kInt32).sum(1, false, torch::kInt32);
    auto row_ptr = torch::zeros({num_state_blocks + 1}, transition_mask.options().dtype(torch::kInt32));
    auto src_row_ptr = torch::zeros({num_state_blocks + 1}, transition_mask.options().dtype(torch::kInt32));
    if (num_state_blocks > 0) {
        row_ptr.narrow(0, 1, num_state_blocks).copy_(counts_per_dst.cumsum(0).to(torch::kInt32));
        src_row_ptr.narrow(0, 1, num_state_blocks).copy_(counts_per_src.cumsum(0).to(torch::kInt32));
    }
    auto active_coords = torch::nonzero(dst_major_active);
    auto empty_i32 = torch::empty({0}, transition_mask.options().dtype(torch::kInt32));
    auto empty_mask = torch::empty(
        {0, block_size, block_size},
        transition_mask.options().dtype(torch::kFloat32));
    if (active_coords.size(0) == 0) {
        return {
            row_ptr.contiguous(),
            empty_i32,
            empty_i32,
            src_row_ptr.contiguous(),
            empty_i32,
            empty_mask,
        };
    }
    auto dst_idx = active_coords.select(1, 0).to(torch::kInt32).contiguous();
    auto col_idx = active_coords.select(1, 1).to(torch::kInt32).contiguous();
    auto linear_block_idx = (
        col_idx.to(torch::kInt64) * static_cast<int64_t>(num_state_blocks)
        + dst_idx.to(torch::kInt64)
    ).contiguous();
    auto block_mask = mask_blocks
        .reshape({num_state_blocks * num_state_blocks, block_size, block_size})
        .index_select(0, linear_block_idx)
        .to(torch::kFloat32)
        .contiguous();
    auto src_sort_key = (
        col_idx.to(torch::kInt64) * static_cast<int64_t>(num_state_blocks)
        + dst_idx.to(torch::kInt64)
    ).contiguous();
    auto src_nz_idx = std::get<1>(torch::sort(src_sort_key)).to(torch::kInt32).contiguous();
    return {
        row_ptr.contiguous(),
        col_idx,
        dst_idx,
        src_row_ptr.contiguous(),
        src_nz_idx,
        block_mask,
    };
}

std::vector<torch::Tensor> causal_machine_scan_build_sparse_metadata_from_runtime(
    int64_t num_states,
    int64_t padded_states,
    int64_t block_size,
    int64_t local_transition_window,
    torch::Tensor transition_mask,
    torch::Tensor runtime_block_mask) {
    check_cuda_bool(transition_mask, "transition_mask");
    check_cuda_bool(runtime_block_mask, "runtime_block_mask");
    TORCH_CHECK(num_states >= 0, "num_states must be non-negative");
    TORCH_CHECK(padded_states >= num_states, "padded_states must be >= num_states");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(
        padded_states % block_size == 0,
        "padded_states must be divisible by block_size"
    );
    check_same_cuda_device(runtime_block_mask, transition_mask, "runtime_block_mask");
    const bool has_dense_mask = is_optional_tensor_defined(transition_mask);
    const bool has_block_mask = is_optional_tensor_defined(runtime_block_mask);
    if (has_dense_mask) {
        TORCH_CHECK(
            transition_mask.dim() == 2
                && transition_mask.size(0) == num_states
                && transition_mask.size(1) == num_states,
            "transition_mask must have shape [num_states, num_states]"
        );
    }
    if (has_block_mask) {
        TORCH_CHECK(runtime_block_mask.dim() == 2, "runtime_block_mask must be 2D");
        TORCH_CHECK(
            runtime_block_mask.size(0) == runtime_block_mask.size(1),
            "runtime_block_mask must be square"
        );
    }
    const auto num_state_blocks = padded_states / block_size;
    auto int_options = transition_mask.options().dtype(torch::kInt64);
    auto int32_options = transition_mask.options().dtype(torch::kInt32);
    auto row_ptr = torch::zeros({num_state_blocks + 1}, int32_options);
    auto src_row_ptr = torch::zeros({num_state_blocks + 1}, int32_options);
    auto empty_i32 = torch::empty({0}, int32_options);
    auto empty_mask = torch::empty(
        {0, block_size, block_size},
        transition_mask.options().dtype(torch::kFloat32));
    auto block_ids = torch::arange(num_state_blocks, int_options);
    auto src_block = block_ids.view({num_state_blocks, 1});
    auto dst_block = block_ids.view({1, num_state_blocks});
    auto src_start = src_block * block_size;
    auto dst_start = dst_block * block_size;
    auto block_active = src_start.lt(num_states).logical_and(dst_start.lt(num_states));
    if (local_transition_window >= 0) {
        auto src_end = (src_start + (block_size - 1)).clamp_max(num_states - 1);
        auto dst_end = (dst_start + (block_size - 1)).clamp_max(num_states - 1);
        auto local_block_active = src_start.le(dst_end + local_transition_window)
            .logical_and(dst_start.le(src_end + local_transition_window));
        block_active = block_active.logical_and(local_block_active);
    }
    if (has_block_mask) {
        const auto block_rows = runtime_block_mask.size(0);
        auto mapped_src = block_ids.clamp_max(block_rows - 1).view({num_state_blocks, 1});
        auto mapped_dst = block_ids.clamp_max(block_rows - 1).view({1, num_state_blocks});
        auto runtime_block_active = runtime_block_mask.index({mapped_src, mapped_dst});
        block_active = block_active.logical_and(runtime_block_active);
    }
    auto active_coords = torch::nonzero(block_active.transpose(0, 1).contiguous());
    if (active_coords.size(0) == 0) {
        return {
            row_ptr.contiguous(),
            empty_i32,
            empty_i32,
            src_row_ptr.contiguous(),
            empty_i32,
            empty_mask,
        };
    }
    auto dst_idx = active_coords.select(1, 0).to(torch::kInt32).contiguous();
    auto col_idx = active_coords.select(1, 1).to(torch::kInt32).contiguous();
    auto row_offsets = torch::arange(block_size, int_options).view({1, block_size, 1});
    auto col_offsets = torch::arange(block_size, int_options).view({1, 1, block_size});
    auto src_state = col_idx.to(torch::kInt64).view({-1, 1, 1}) * block_size + row_offsets;
    auto dst_state = dst_idx.to(torch::kInt64).view({-1, 1, 1}) * block_size + col_offsets;
    auto block_mask = src_state.lt(num_states).logical_and(dst_state.lt(num_states));
    if (local_transition_window >= 0) {
        auto local_cell_mask = (src_state - dst_state).abs().le(local_transition_window);
        block_mask = block_mask.logical_and(local_cell_mask);
    }
    if (has_dense_mask) {
        auto flat_transition_mask = transition_mask.reshape({-1});
        auto clamped_src_state = src_state.clamp_max(num_states - 1);
        auto clamped_dst_state = dst_state.clamp_max(num_states - 1);
        auto dense_linear_idx = (
            clamped_src_state * static_cast<int64_t>(num_states) + clamped_dst_state
        ).reshape({-1});
        auto dense_cell_mask = flat_transition_mask.index_select(0, dense_linear_idx).view_as(block_mask);
        block_mask = block_mask.logical_and(dense_cell_mask);
    }
    auto keep_blocks = block_mask.any(2).any(1);
    auto kept_block_idx = torch::nonzero(keep_blocks).squeeze(1);
    if (kept_block_idx.size(0) == 0) {
        return {
            row_ptr.contiguous(),
            empty_i32,
            empty_i32,
            src_row_ptr.contiguous(),
            empty_i32,
            empty_mask,
        };
    }
    auto dst_idx_kept = dst_idx.index_select(0, kept_block_idx).contiguous();
    auto col_idx_kept = col_idx.index_select(0, kept_block_idx).contiguous();
    auto block_mask_kept = block_mask.index_select(0, kept_block_idx).to(torch::kFloat32).contiguous();
    auto counts_per_dst = torch::bincount(
        dst_idx_kept.to(torch::kInt64),
        torch::Tensor(),
        num_state_blocks).to(torch::kInt32);
    auto counts_per_src = torch::bincount(
        col_idx_kept.to(torch::kInt64),
        torch::Tensor(),
        num_state_blocks).to(torch::kInt32);
    if (num_state_blocks > 0) {
        row_ptr.narrow(0, 1, num_state_blocks).copy_(counts_per_dst.cumsum(0).to(torch::kInt32));
        src_row_ptr.narrow(0, 1, num_state_blocks).copy_(counts_per_src.cumsum(0).to(torch::kInt32));
    }
    auto src_sort_key = (
        col_idx_kept.to(torch::kInt64) * static_cast<int64_t>(num_state_blocks)
        + dst_idx_kept.to(torch::kInt64)
    ).contiguous();
    auto src_nz_idx = std::get<1>(torch::sort(src_sort_key)).to(torch::kInt32).contiguous();
    return {
        row_ptr.contiguous(),
        col_idx_kept,
        dst_idx_kept,
        src_row_ptr.contiguous(),
        src_nz_idx,
        block_mask_kept,
    };
}

std::vector<torch::Tensor> causal_machine_scan_build_grouped_sparse_backward_metadata(
    torch::Tensor col_idx,
    torch::Tensor src_nz_idx) {
    check_cuda_int32(col_idx, "col_idx");
    check_cuda_int32(src_nz_idx, "src_nz_idx");
    check_same_cuda_device(src_nz_idx, col_idx, "src_nz_idx");
    TORCH_CHECK(col_idx.dim() == 1, "col_idx must be 1D");
    TORCH_CHECK(src_nz_idx.dim() == 1, "src_nz_idx must be 1D");
    auto options_i32 = col_idx.options().dtype(torch::kInt32);
    auto empty_i32 = torch::empty({0}, options_i32);
    if (src_nz_idx.numel() == 0) {
        return {
            torch::zeros({1}, options_i32),
            empty_i32,
            empty_i32,
        };
    }
    auto ordered_src_blocks = col_idx.index_select(0, src_nz_idx.to(torch::kInt64)).to(torch::kInt32).contiguous();
    auto starts = torch::ones_like(ordered_src_blocks, options_i32);
    if (ordered_src_blocks.size(0) > 1) {
        auto curr = ordered_src_blocks.slice(0, 1);
        auto prev = ordered_src_blocks.slice(0, 0, ordered_src_blocks.size(0) - 1);
        starts.slice(0, 1).copy_(curr.ne(prev).to(torch::kInt32));
    }
    auto group_ids = starts.cumsum(0).sub_(1).to(torch::kInt32).contiguous();
    const int64_t num_groups = static_cast<int64_t>(group_ids[-1].item<int>()) + 1;
    auto counts = torch::bincount(group_ids.to(torch::kInt64), torch::Tensor(), num_groups).to(torch::kInt32);
    auto row_ptr = torch::zeros({num_groups + 1}, options_i32);
    if (num_groups > 0) {
        row_ptr.narrow(0, 1, num_groups).copy_(counts.cumsum(0).to(torch::kInt32));
    }
    auto group_start_idx = torch::nonzero(starts).squeeze(1).to(torch::kInt64);
    auto grouped_src_block_idx = ordered_src_blocks.index_select(0, group_start_idx).to(torch::kInt32).contiguous();
    return {
        row_ptr.contiguous(),
        grouped_src_block_idx,
        group_ids,
    };
}

void causal_machine_scan_record_paged_step_(
    torch::Tensor paged_log_beliefs,
    torch::Tensor paged_latent_states,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor log_belief,
    torch::Tensor latent_state,
    int64_t num_updates) {
    TORCH_CHECK(num_updates >= 0, "num_updates must be non-negative");
    check_cuda_activation(paged_log_beliefs, "paged_log_beliefs");
    check_cuda_int64(paged_page_table, "paged_page_table");
    check_cuda_int64(paged_lengths, "paged_lengths");
    check_cuda_activation(log_belief, "log_belief");
    TORCH_CHECK(paged_log_beliefs.dim() == 3, "paged_log_beliefs must have shape [K, S, N]");
    TORCH_CHECK(log_belief.dim() == 2, "log_belief must have shape [B, N]");
    TORCH_CHECK(paged_page_table.dim() == 2, "paged_page_table must have shape [B, P]");
    TORCH_CHECK(paged_lengths.dim() == 1, "paged_lengths must have shape [B]");
    TORCH_CHECK(
        paged_page_table.size(0) == log_belief.size(0)
            && paged_log_beliefs.size(2) == log_belief.size(1),
        "paged page table and log_belief shapes must agree on batch and state dimensions"
    );
    TORCH_CHECK(
        paged_lengths.size(0) == paged_page_table.size(0),
        "paged_lengths must have one entry per batch element"
    );
    TORCH_CHECK(
        paged_log_beliefs.size(0) >= paged_page_table.size(0) * paged_page_table.size(1),
        "paged_log_beliefs must provide at least B * P physical page slots"
    );
    check_same_cuda_device(paged_page_table, paged_log_beliefs, "paged_page_table");
    check_same_cuda_device(paged_lengths, paged_log_beliefs, "paged_lengths");
    check_same_cuda_device(log_belief, paged_log_beliefs, "log_belief");
    const auto page_size = paged_log_beliefs.size(1);
    const auto max_pages = paged_page_table.size(1);
    const auto capacity = page_size * max_pages;
    if (capacity > 0) {
        causal_machine_scan_record_paged_step_tensor_from_lengths_cuda(
            paged_log_beliefs,
            paged_page_table,
            paged_lengths,
            log_belief.to(paged_log_beliefs.options())
        );
        if (is_optional_tensor_defined(paged_latent_states) && is_optional_tensor_defined(latent_state)) {
            check_cuda_activation(paged_latent_states, "paged_latent_states");
            check_cuda_activation(latent_state, "latent_state");
            check_same_cuda_device(paged_latent_states, paged_log_beliefs, "paged_latent_states");
            check_same_cuda_device(latent_state, paged_log_beliefs, "latent_state");
            TORCH_CHECK(
                paged_latent_states.dim() == 3,
                "paged_latent_states must have shape [K, S, R]"
            );
            TORCH_CHECK(latent_state.dim() == 2, "latent_state must have shape [B, R]");
            TORCH_CHECK(
                paged_page_table.size(0) == latent_state.size(0)
                    && paged_latent_states.size(2) == latent_state.size(1)
                    && paged_latent_states.size(0) >= paged_page_table.size(0) * paged_page_table.size(1),
                "paged_latent_states and latent_state shapes must agree on batch and latent dimensions"
            );
            causal_machine_scan_record_paged_step_tensor_from_lengths_cuda(
                paged_latent_states,
                paged_page_table,
                paged_lengths,
                latent_state.to(paged_latent_states.options())
            );
        }
    }
    causal_machine_scan_increment_paged_lengths_cuda(paged_lengths, 1, capacity);
}

void causal_machine_scan_record_paged_sequence_(
    torch::Tensor paged_log_beliefs,
    torch::Tensor paged_latent_states,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor state_log_beliefs,
    torch::Tensor latent_states,
    int64_t num_updates) {
    TORCH_CHECK(num_updates >= 0, "num_updates must be non-negative");
    check_cuda_activation(paged_log_beliefs, "paged_log_beliefs");
    check_cuda_int64(paged_page_table, "paged_page_table");
    check_cuda_int64(paged_lengths, "paged_lengths");
    check_cuda_activation(state_log_beliefs, "state_log_beliefs");
    TORCH_CHECK(state_log_beliefs.dim() == 3, "state_log_beliefs must have shape [B, L, N]");
    TORCH_CHECK(
        paged_log_beliefs.dim() == 3,
        "paged_log_beliefs must have shape [K, S, N]"
    );
    TORCH_CHECK(paged_page_table.dim() == 2, "paged_page_table must have shape [B, P]");
    TORCH_CHECK(
        paged_page_table.size(0) == state_log_beliefs.size(0)
            && paged_log_beliefs.size(2) == state_log_beliefs.size(2),
        "paged page table and state_log_beliefs shapes must agree on batch and state dimensions"
    );
    TORCH_CHECK(
        paged_lengths.dim() == 1 && paged_lengths.size(0) == paged_page_table.size(0),
        "paged_lengths must have shape [B]"
    );
    TORCH_CHECK(
        paged_log_beliefs.size(0) >= paged_page_table.size(0) * paged_page_table.size(1),
        "paged_log_beliefs must provide at least B * P physical page slots"
    );
    check_same_cuda_device(paged_page_table, paged_log_beliefs, "paged_page_table");
    check_same_cuda_device(paged_lengths, paged_log_beliefs, "paged_lengths");
    check_same_cuda_device(state_log_beliefs, paged_log_beliefs, "state_log_beliefs");
    const auto seq_len = state_log_beliefs.size(1);
    const auto page_size = paged_log_beliefs.size(1);
    const auto max_pages = paged_page_table.size(1);
    const auto capacity = page_size * max_pages;
    if (is_optional_tensor_defined(paged_latent_states) || is_optional_tensor_defined(latent_states)) {
        TORCH_CHECK(
            is_optional_tensor_defined(paged_latent_states) && is_optional_tensor_defined(latent_states),
            "paged_latent_states and latent_states must either both be present or both be empty"
        );
        TORCH_CHECK(paged_latent_states.dim() == 3, "paged_latent_states must have shape [K, S, R]");
        TORCH_CHECK(latent_states.dim() == 3, "latent_states must have shape [B, L, R]");
        check_cuda_activation(paged_latent_states, "paged_latent_states");
        check_cuda_activation(latent_states, "latent_states");
        check_same_cuda_device(paged_latent_states, paged_log_beliefs, "paged_latent_states");
        check_same_cuda_device(latent_states, paged_log_beliefs, "latent_states");
        TORCH_CHECK(
            paged_page_table.size(0) == latent_states.size(0)
                && paged_latent_states.size(2) == latent_states.size(2)
                && paged_latent_states.size(0) >= paged_page_table.size(0) * paged_page_table.size(1)
                && latent_states.size(1) == seq_len,
            "paged_latent_states and latent_states must match batch, sequence, and latent dimensions"
        );
    }
    if (capacity > 0 && seq_len > 0) {
        causal_machine_scan_record_paged_sequence_tensor_cuda(
            paged_log_beliefs,
            paged_page_table,
            paged_lengths,
            state_log_beliefs.to(paged_log_beliefs.options()),
            num_updates);
        if (is_optional_tensor_defined(paged_latent_states)) {
            causal_machine_scan_record_paged_sequence_tensor_cuda(
                paged_latent_states,
                paged_page_table,
                paged_lengths,
                latent_states.to(paged_latent_states.options()),
                num_updates);
        }
    }
    causal_machine_scan_increment_paged_lengths_cuda(paged_lengths, seq_len, capacity);
}

std::vector<torch::Tensor> causal_machine_scan_read_paged_latest_(
    torch::Tensor paged_log_beliefs,
    torch::Tensor paged_latent_states,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths) {
    check_cuda_activation(paged_log_beliefs, "paged_log_beliefs");
    check_cuda_int64(paged_page_table, "paged_page_table");
    check_cuda_int64(paged_lengths, "paged_lengths");
    TORCH_CHECK(paged_log_beliefs.dim() == 3, "paged_log_beliefs must have shape [K, S, N]");
    TORCH_CHECK(paged_page_table.dim() == 2, "paged_page_table must have shape [B, P]");
    TORCH_CHECK(
        paged_lengths.dim() == 1 && paged_lengths.size(0) == paged_page_table.size(0),
        "paged_lengths must have shape [B]");
    TORCH_CHECK(
        paged_log_beliefs.size(0) >= paged_page_table.size(0) * paged_page_table.size(1),
        "paged_log_beliefs must provide at least B * P physical page slots"
    );
    check_same_cuda_device(paged_page_table, paged_log_beliefs, "paged_page_table");
    check_same_cuda_device(paged_lengths, paged_log_beliefs, "paged_lengths");
    auto log_belief = torch::empty(
        {paged_page_table.size(0), paged_log_beliefs.size(2)},
        paged_log_beliefs.options());
    causal_machine_scan_read_paged_latest_tensor_cuda(
        paged_log_beliefs,
        paged_page_table,
        paged_lengths,
        log_belief);
    torch::Tensor latent_state;
    if (is_optional_tensor_defined(paged_latent_states)) {
        check_cuda_activation(paged_latent_states, "paged_latent_states");
        check_same_cuda_device(paged_latent_states, paged_log_beliefs, "paged_latent_states");
        TORCH_CHECK(
            paged_latent_states.dim() == 3
                && paged_latent_states.size(0) >= paged_page_table.size(0) * paged_page_table.size(1)
                && paged_latent_states.size(1) == paged_log_beliefs.size(1),
            "paged_latent_states must have shape [K, S, R] and match paged storage");
        latent_state = torch::empty(
            {paged_page_table.size(0), paged_latent_states.size(2)},
            paged_latent_states.options());
        causal_machine_scan_read_paged_latest_tensor_cuda(
            paged_latent_states,
            paged_page_table,
            paged_lengths,
            latent_state);
    } else {
        latent_state = torch::empty({0, 0}, paged_log_beliefs.options().dtype(torch::kFloat32));
    }
    return {log_belief, latent_state};
}

std::vector<torch::Tensor> causal_machine_scan_paged_step_(
    torch::Tensor paged_log_beliefs,
    torch::Tensor paged_latent_states,
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_gate,
    torch::Tensor packed_transition_source,
    torch::Tensor packed_transition_source_scales,
    torch::Tensor packed_transition_dest,
    torch::Tensor packed_transition_dest_scales,
    int64_t packed_kind,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max) {
    check_cuda_activation(paged_log_beliefs, "paged_log_beliefs");
    check_cuda_int64(paged_page_table, "paged_page_table");
    check_cuda_int64(paged_lengths, "paged_lengths");
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_float32(transition_source_probs, "transition_source_probs");
    check_cuda_float32(transition_dest_probs, "transition_dest_probs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    TORCH_CHECK(local_logits.dim() == 3, "local_logits must have shape [B, 1, N]");
    TORCH_CHECK(local_logits.size(1) == 1, "paged_step_ only supports seq_len=1");
    TORCH_CHECK(transition_context.sizes() == local_logits.sizes(), "transition_context must match local_logits shape");
    TORCH_CHECK(transition_source_probs.dim() == 2, "transition_source_probs must have shape [N, R]");
    TORCH_CHECK(transition_dest_probs.dim() == 2, "transition_dest_probs must have shape [R, N]");
    TORCH_CHECK(transition_stay_probs.dim() == 1, "transition_stay_probs must have shape [N]");
    TORCH_CHECK(transition_gate.numel() == 1, "transition_gate must be a scalar tensor");
    TORCH_CHECK(
        paged_log_beliefs.dim() == 3
            && paged_page_table.dim() == 2
            && paged_page_table.size(0) == local_logits.size(0)
            && paged_log_beliefs.size(2) == local_logits.size(2)
            && paged_log_beliefs.size(0) >= paged_page_table.size(0) * paged_page_table.size(1),
        "paged storage must have shape [K, S, N] and paged_page_table must have shape [B, P]"
    );
    TORCH_CHECK(
        paged_lengths.dim() == 1 && paged_lengths.size(0) == local_logits.size(0),
        "paged_lengths must have shape [B]"
    );
    TORCH_CHECK(
        transition_source_probs.size(0) == local_logits.size(2)
            && transition_dest_probs.size(1) == local_logits.size(2)
            && transition_dest_probs.size(0) == transition_source_probs.size(1),
        "transition tables must match local_logits state/rank dims"
    );
    TORCH_CHECK(
        transition_stay_probs.size(0) == local_logits.size(2),
        "transition_stay_probs size must match local_logits last dim"
    );
    check_same_cuda_device(paged_page_table, paged_log_beliefs, "paged_page_table");
    check_same_cuda_device(paged_lengths, paged_log_beliefs, "paged_lengths");
    check_same_cuda_device(local_logits, paged_log_beliefs, "local_logits");
    check_same_cuda_device(transition_source_probs, paged_log_beliefs, "transition_source_probs");
    check_same_cuda_device(transition_dest_probs, paged_log_beliefs, "transition_dest_probs");
    check_same_cuda_device(transition_context, paged_log_beliefs, "transition_context");
    check_same_cuda_device(transition_stay_probs, paged_log_beliefs, "transition_stay_probs");
    TORCH_CHECK(
        transition_context.scalar_type() == local_logits.scalar_type(),
        "transition_context must match local_logits dtype"
    );
    if (is_optional_tensor_defined(paged_latent_states)) {
        check_cuda_activation(paged_latent_states, "paged_latent_states");
        check_same_cuda_device(paged_latent_states, paged_log_beliefs, "paged_latent_states");
        TORCH_CHECK(
            paged_latent_states.dim() == 3
                && paged_latent_states.size(0) >= paged_page_table.size(0) * paged_page_table.size(1)
                && paged_latent_states.size(1) == paged_log_beliefs.size(1),
            "paged_latent_states must have shape [K, S, R] and match paged storage"
        );
    }
    auto initial_log_belief = torch::full(
        {local_logits.size(0), local_logits.size(2)},
        -std::log(static_cast<double>(std::max<int64_t>(local_logits.size(2), 1))),
        local_logits.options().dtype(torch::kFloat32));
    auto latest_log_belief = torch::empty_like(initial_log_belief);
    causal_machine_scan_read_paged_latest_tensor_cuda(
        paged_log_beliefs,
        paged_page_table,
        paged_lengths,
        latest_log_belief);
    const auto has_history = paged_lengths.gt(0).view({local_logits.size(0), 1});
    initial_log_belief = torch::where(has_history, latest_log_belief, initial_log_belief);

    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, local_logits);
    std::vector<torch::Tensor> scan_outputs;
    const auto num_states = local_logits.size(2);
    const auto empty_seq_lens = empty_cuda_int64_tensor_like(local_logits);
    const bool use_packed = is_optional_tensor_defined(packed_transition_source)
        && is_optional_tensor_defined(packed_transition_source_scales)
        && is_optional_tensor_defined(packed_transition_dest)
        && is_optional_tensor_defined(packed_transition_dest_scales);
    if (
        !use_packed
        && num_states == kSpecializedNumStates
        && paged_log_beliefs.scalar_type() == local_logits.scalar_type()
        && (!is_optional_tensor_defined(paged_latent_states) || paged_latent_states.scalar_type() == local_logits.scalar_type())
    ) {
        if (transition_source_probs.size(1) == 8) {
            return causal_machine_scan_paged_step_dense_128_rank8_cuda(
                paged_log_beliefs,
                paged_latent_states,
                paged_page_table,
                paged_lengths,
                local_logits,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                transition_stay_probs,
                transition_gate_f32,
                score_clamp_min,
                score_clamp_max);
        }
        if (transition_source_probs.size(1) == 16) {
            return causal_machine_scan_paged_step_dense_128_rank16_cuda(
                paged_log_beliefs,
                paged_latent_states,
                paged_page_table,
                paged_lengths,
                local_logits,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                transition_stay_probs,
                transition_gate_f32,
                score_clamp_min,
                score_clamp_max);
        }
    }
    if (use_packed && packed_kind == 0) {
        return causal_machine_scan_paged_step_quantized_cuda(
            paged_log_beliefs,
            paged_latent_states,
            paged_page_table,
            paged_lengths,
            local_logits,
            packed_transition_source,
            packed_transition_source_scales,
            packed_transition_dest,
            packed_transition_dest_scales,
            transition_context,
            transition_stay_probs,
            transition_gate_f32);
    } else if (use_packed && (packed_kind == 1 || packed_kind == 2)) {
        return causal_machine_scan_paged_step_fp8_cuda(
            paged_log_beliefs,
            paged_latent_states,
            paged_page_table,
            paged_lengths,
            local_logits,
            packed_transition_source,
            packed_transition_source_scales,
            packed_transition_dest,
            packed_transition_dest_scales,
            transition_context,
            transition_stay_probs,
            transition_gate_f32,
            packed_kind == 1 ? 0 : 1);
    } else if (num_states > kSpecializedNumStates) {
        TORCH_CHECK(tile_size > 0, "tile_size must be positive for paged tiled step");
        TORCH_CHECK(split_size > 0, "split_size must be positive for paged tiled step");
        scan_outputs = causal_machine_scan_forward_tiled_logits_kernel_cuda(
            local_logits,
            transition_source_probs,
            transition_dest_probs,
            transition_context,
            initial_log_belief,
            transition_gate_f32,
            transition_stay_probs,
            empty_seq_lens,
            1,
            tile_size,
            split_size,
            score_clamp_min,
            score_clamp_max);
    } else {
        scan_outputs = causal_machine_scan_forward_cuda(
            local_logits,
            transition_source_probs,
            transition_dest_probs,
            transition_context,
            initial_log_belief.to(local_logits.scalar_type()),
            transition_gate_f32,
            transition_stay_probs,
            1,
            score_clamp_min,
            score_clamp_max);
    }
    TORCH_CHECK(scan_outputs.size() == 2, "paged_step_ expected forward scan to return beliefs and final_log_belief");
    const auto& beliefs = scan_outputs[0];
    const auto& final_log_belief = scan_outputs[1];
    causal_machine_scan_record_paged_step_tensor_from_lengths_cuda(
        paged_log_beliefs,
        paged_page_table,
        paged_lengths,
        beliefs.select(1, 0).to(paged_log_beliefs.options()));
    if (is_optional_tensor_defined(paged_latent_states)) {
        auto empty_latent = torch::zeros(
            {local_logits.size(0), paged_latent_states.size(2)},
            paged_latent_states.options());
        causal_machine_scan_record_paged_step_tensor_from_lengths_cuda(
            paged_latent_states,
            paged_page_table,
            paged_lengths,
            empty_latent);
    }
    const auto capacity = paged_page_table.size(1) * paged_log_beliefs.size(1);
    causal_machine_scan_increment_paged_lengths_cuda(paged_lengths, 1, capacity);
    return scan_outputs;
}

void causal_machine_scan_reorder_paged_cache_(
    torch::Tensor paged_page_table,
    torch::Tensor paged_lengths,
    torch::Tensor beam_indices) {
    check_cuda_int64(paged_page_table, "paged_page_table");
    check_cuda_int64(paged_lengths, "paged_lengths");
    check_cuda_int64(beam_indices, "beam_indices");
    TORCH_CHECK(paged_page_table.dim() == 2, "paged_page_table must have shape [B, P]");
    TORCH_CHECK(paged_lengths.dim() == 1, "paged_lengths must have shape [B]");
    TORCH_CHECK(beam_indices.dim() == 1, "beam_indices must have shape [B]");
    TORCH_CHECK(
        paged_page_table.size(0) == paged_lengths.size(0)
            && paged_lengths.size(0) == beam_indices.size(0),
        "paged page table, paged lengths, and beam indices must agree on batch size");
    check_same_cuda_device(paged_lengths, paged_page_table, "paged_lengths");
    check_same_cuda_device(beam_indices, paged_page_table, "beam_indices");
    causal_machine_scan_reorder_paged_cache_cuda(
        paged_page_table,
        paged_lengths,
        beam_indices);
}

}  // namespace

std::vector<torch::Tensor> causal_machine_scan_forward_logits(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_float32(transition_source_logits, "transition_source_logits");
    check_cuda_float32(transition_dest_logits, "transition_dest_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_shapes(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_devices(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
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
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, local_logits);
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (local_logits.size(0) == 0 || local_logits.size(1) == 0) {
        auto beliefs = torch::empty_like(local_logits);
        auto final_log_belief = torch::empty_like(initial_log_belief);
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto num_states = local_logits.size(2);
    return dispatch_specialized_forward_wrapper(
        num_states,
        transition_source_logits.size(1),
        [&]() {
            return causal_machine_scan_forward_logits_cuda(
                pad_last_dim(local_logits, kSpecializedNumStates, neg_inf_fill()),
                pad_first_dim(transition_source_logits, kSpecializedNumStates, 0.0),
                pad_last_dim(transition_dest_logits, kSpecializedNumStates, neg_inf_fill()),
                pad_last_dim(transition_context, kSpecializedNumStates, 0.0),
                pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill()),
                transition_gate_f32,
                pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0),
                chunk_size,
                score_clamp_min,
                score_clamp_max);
        },
        [&]() {
            return causal_machine_scan_forward_logits_cuda(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                transition_gate_f32,
                transition_stay_probs,
                chunk_size,
                score_clamp_min,
                score_clamp_max);
        });
}

std::vector<torch::Tensor> causal_machine_scan_forward_composable_logits(
    torch::Tensor local_logits,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_float32(transition_source_logits, "transition_source_logits");
    check_cuda_float32(transition_dest_logits, "transition_dest_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_shapes(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_devices(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
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
    const auto num_states = local_logits.size(2);
    return dispatch_specialized_forward_wrapper(
        num_states,
        transition_source_logits.size(1),
        [&]() {
            return causal_machine_scan_forward_composable_logits_cuda(
                pad_last_dim(local_logits, kSpecializedNumStates, neg_inf_fill()),
                pad_first_dim(transition_source_logits, kSpecializedNumStates, 0.0),
                pad_last_dim(transition_dest_logits, kSpecializedNumStates, neg_inf_fill()),
                pad_last_dim(transition_context, kSpecializedNumStates, 0.0),
                pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill()),
                pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0),
                chunk_size);
        },
        [&]() {
            return causal_machine_scan_forward_composable_logits_cuda(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                transition_stay_probs,
                chunk_size);
        });
}

std::vector<torch::Tensor> causal_machine_scan_forward(
    torch::Tensor local_logits,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
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
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, local_logits);
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (local_logits.size(0) == 0 || local_logits.size(1) == 0) {
        auto beliefs = torch::empty_like(local_logits);
        auto final_log_belief = torch::empty_like(initial_log_belief);
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto num_states = local_logits.size(2);
    return dispatch_specialized_forward_wrapper(
        num_states,
        transition_source_probs.size(1),
        [&]() {
            return causal_machine_scan_forward_cuda(
                pad_last_dim(local_logits, kSpecializedNumStates, neg_inf_fill()),
                pad_first_dim(transition_source_probs, kSpecializedNumStates, 0.0),
                pad_last_dim(transition_dest_probs, kSpecializedNumStates, 0.0),
                pad_last_dim(transition_context, kSpecializedNumStates, 0.0),
                pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill()),
                transition_gate_f32,
                pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0),
                chunk_size,
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity());
        },
        [&]() {
            return causal_machine_scan_forward_cuda(
                local_logits,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                initial_log_belief,
                transition_gate_f32,
                transition_stay_probs,
                chunk_size,
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity());
        });
}

std::vector<torch::Tensor> causal_machine_scan_forward_quantized(
    torch::Tensor local_logits,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
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
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, local_logits);
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (local_logits.size(0) == 0 || local_logits.size(1) == 0) {
        auto beliefs = torch::empty_like(local_logits);
        auto final_log_belief = torch::empty_like(initial_log_belief);
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto num_states = local_logits.size(2);
    if (num_states > kSpecializedNumStates) {
        const auto geometry = choose_large_state_tiled_geometry(
            local_logits.get_device(),
            num_states,
            transition_source_q.size(1),
            false);
        return causal_machine_scan_forward_tiled_quantized_kernel_cuda(
            local_logits,
            transition_source_q,
            transition_source_scales,
            transition_dest_q,
            transition_dest_scales,
            transition_context,
            initial_log_belief.to(torch::kFloat32),
            transition_gate_f32,
            transition_stay_probs,
            torch::Tensor(),
            chunk_size,
            geometry.first,
            geometry.second,
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            0);
    }
    return dispatch_specialized_forward_wrapper(
        num_states,
        transition_source_q.size(1),
        [&]() {
            return causal_machine_scan_forward_quantized_cuda(
                pad_last_dim(local_logits, kSpecializedNumStates, neg_inf_fill()),
                pad_first_dim(transition_source_q, kSpecializedNumStates, 0.0),
                pad_last_dim(transition_source_scales, kSpecializedNumStates, 1.0),
                pad_last_dim(transition_dest_q, kSpecializedNumStates, 0.0),
                transition_dest_scales,
                pad_last_dim(transition_context, kSpecializedNumStates, 0.0),
                pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill()),
                transition_gate_f32,
                pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0),
                chunk_size);
        },
        [&]() {
            return causal_machine_scan_forward_quantized_cuda(
                local_logits,
                transition_source_q,
                transition_source_scales,
                transition_dest_q,
                transition_dest_scales,
                transition_context,
                initial_log_belief,
                transition_gate_f32,
                transition_stay_probs,
                chunk_size);
        });
}

std::vector<torch::Tensor> causal_machine_scan_forward_fp8(
    torch::Tensor local_logits,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    int64_t chunk_size) {
    check_cuda_activation(local_logits, "local_logits");
    check_cuda_uint8(transition_source_packed, "transition_source_packed");
    check_cuda_float32(transition_source_scales, "transition_source_scales");
    check_cuda_uint8(transition_dest_packed, "transition_dest_packed");
    check_cuda_float32(transition_dest_scales, "transition_dest_scales");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_fp8_shapes(
        local_logits,
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_devices(
        local_logits,
        transition_source_packed,
        transition_dest_packed,
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
    TORCH_CHECK(fp8_format == 0 || fp8_format == 1, "fp8_format must be 0 (e4m3) or 1 (e5m2)");
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, local_logits);
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (local_logits.size(0) == 0 || local_logits.size(1) == 0) {
        auto beliefs = torch::empty_like(local_logits);
        auto final_log_belief = torch::empty_like(initial_log_belief);
        final_log_belief.copy_(initial_log_belief);
        return {beliefs, final_log_belief};
    }
    const auto num_states = local_logits.size(2);
    if (num_states > kSpecializedNumStates) {
        const auto geometry = choose_large_state_tiled_geometry(
            local_logits.get_device(),
            num_states,
            transition_source_packed.size(1),
            false);
        return causal_machine_scan_forward_tiled_fp8_kernel_cuda(
            local_logits,
            transition_source_packed,
            transition_source_scales,
            transition_dest_packed,
            transition_dest_scales,
            transition_context,
            initial_log_belief.to(torch::kFloat32),
            transition_gate_f32,
            transition_stay_probs,
            fp8_format,
            torch::Tensor(),
            chunk_size,
            geometry.first,
            geometry.second,
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            0);
    }
    return dispatch_specialized_forward_wrapper(
        num_states,
        transition_source_packed.size(1),
        [&]() {
            return causal_machine_scan_forward_fp8_cuda(
                pad_last_dim(local_logits, kSpecializedNumStates, neg_inf_fill()),
                pad_first_dim(transition_source_packed, kSpecializedNumStates, 0.0),
                pad_last_dim(transition_source_scales, kSpecializedNumStates, 1.0),
                pad_last_dim(transition_dest_packed, kSpecializedNumStates, 0.0),
                transition_dest_scales,
                pad_last_dim(transition_context, kSpecializedNumStates, 0.0),
                pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill()),
                transition_gate_f32,
                pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0),
                fp8_format,
                chunk_size);
        },
        [&]() {
            return causal_machine_scan_forward_fp8_cuda(
                local_logits,
                transition_source_packed,
                transition_source_scales,
                transition_dest_packed,
                transition_dest_scales,
                transition_context,
                initial_log_belief,
                transition_gate_f32,
                transition_stay_probs,
                fp8_format,
                chunk_size);
        });
}

std::vector<torch::Tensor> causal_machine_scan_backward(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
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
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, beliefs);
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
    const auto num_states = beliefs.size(2);
    return dispatch_specialized_backward_wrapper(
        num_states,
        transition_source_probs.size(1),
        [&]() {
            auto padded_grad_beliefs = pad_last_dim(grad_beliefs, kSpecializedNumStates, 0.0);
            auto padded_grad_final_belief = pad_last_dim(grad_final_belief, kSpecializedNumStates, 0.0);
            auto padded_transition_source_probs = pad_first_dim(transition_source_probs, kSpecializedNumStates, 0.0);
            auto padded_transition_dest_probs = pad_last_dim(transition_dest_probs, kSpecializedNumStates, 0.0);
            auto padded_transition_context = pad_last_dim(transition_context, kSpecializedNumStates, 0.0);
            auto padded_initial_log_belief = pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill());
            auto padded_beliefs = pad_last_dim(beliefs, kSpecializedNumStates, neg_inf_fill());
            auto padded_transition_stay_probs = pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0);
            auto workspace = get_dense_backward_workspace(
                padded_beliefs,
                padded_beliefs.size(2),
                padded_transition_source_probs.size(1));
            return causal_machine_scan_backward_workspace_cuda(
                padded_grad_beliefs,
                padded_grad_final_belief,
                padded_transition_source_probs,
                padded_transition_dest_probs,
                padded_transition_context,
                padded_initial_log_belief,
                padded_beliefs,
                transition_gate_f32,
                padded_transition_stay_probs,
                chunk_size,
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                workspace.grad_transition_source_per_batch,
                workspace.grad_transition_dest_per_batch,
                workspace.grad_transition_stay_per_batch,
                workspace.grad_transition_gate_per_batch);
        },
        [&]() {
            auto workspace = get_dense_backward_workspace(
                beliefs,
                beliefs.size(2),
                transition_source_probs.size(1));
            return causal_machine_scan_backward_workspace_cuda(
                grad_beliefs,
                grad_final_belief,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                initial_log_belief,
                beliefs,
                transition_gate_f32,
                transition_stay_probs,
                chunk_size,
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                workspace.grad_transition_source_per_batch,
                workspace.grad_transition_dest_per_batch,
                workspace.grad_transition_stay_per_batch,
                workspace.grad_transition_gate_per_batch);
        });
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_impl(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk,
    bool allow_aten_fallback) {
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_final_belief, "grad_final_belief");
    check_cuda_float32(transition_source_probs, "transition_source_probs");
    check_cuda_float32(transition_dest_probs, "transition_dest_probs");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_float32(initial_log_belief, "initial_log_belief");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    TORCH_CHECK(beliefs.dim() == 3, "beliefs must have shape [B, L, N]");
    TORCH_CHECK(grad_beliefs.sizes() == beliefs.sizes(), "grad_beliefs must match beliefs shape");
    TORCH_CHECK(transition_context.sizes() == beliefs.sizes(), "transition_context must match beliefs shape");
    TORCH_CHECK(transition_source_probs.dim() == 2, "transition_source_probs must have shape [N, R]");
    TORCH_CHECK(transition_dest_probs.dim() == 2, "transition_dest_probs must have shape [R, N]");
    TORCH_CHECK(transition_stay_probs.dim() == 1, "transition_stay_probs must have shape [N]");
    TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape [B, N]");
    TORCH_CHECK(
        initial_log_belief.sizes() == grad_final_belief.sizes(),
        "grad_final_belief must match initial_log_belief shape"
    );
    TORCH_CHECK(
        transition_source_probs.size(0) == beliefs.size(2),
        "transition_source_probs first dim must match beliefs last dim"
    );
    TORCH_CHECK(
        transition_dest_probs.size(1) == beliefs.size(2),
        "transition_dest_probs last dim must match beliefs last dim"
    );
    TORCH_CHECK(
        transition_dest_probs.size(0) == transition_source_probs.size(1),
        "transition_source_probs and transition_dest_probs rank must match"
    );
    TORCH_CHECK(
        transition_stay_probs.size(0) == beliefs.size(2),
        "transition_stay_probs size must match beliefs last dim"
    );
    TORCH_CHECK(
        initial_log_belief.size(0) == beliefs.size(0) && initial_log_belief.size(1) == beliefs.size(2),
        "initial_log_belief must match beliefs batch/state shape"
    );
    TORCH_CHECK(beliefs.size(2) > kSpecializedNumStates, "backward_tiled_probs is intended for num_states > 128");
    check_same_cuda_device(grad_beliefs, beliefs, "grad_beliefs");
    check_same_cuda_device(grad_final_belief, beliefs, "grad_final_belief");
    check_same_cuda_device(transition_source_probs, beliefs, "transition_source_probs");
    check_same_cuda_device(transition_dest_probs, beliefs, "transition_dest_probs");
    check_same_cuda_device(transition_context, beliefs, "transition_context");
    check_same_cuda_device(initial_log_belief, beliefs, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, beliefs, "transition_stay_probs");
    TORCH_CHECK(grad_beliefs.scalar_type() == beliefs.scalar_type(), "grad_beliefs must match beliefs dtype");
    TORCH_CHECK(grad_final_belief.scalar_type() == beliefs.scalar_type(), "grad_final_belief must match beliefs dtype");
    TORCH_CHECK(transition_context.scalar_type() == beliefs.scalar_type(), "transition_context must match beliefs dtype");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(split_size > 0, "split_size must be positive");
    if (is_optional_tensor_defined(seq_lens)) {
        check_cuda_int64(seq_lens, "seq_lens");
        check_same_cuda_device(seq_lens, beliefs, "seq_lens");
        TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
        TORCH_CHECK(seq_lens.size(0) == beliefs.size(0), "seq_lens must have shape [B]");
    } else {
        seq_lens = empty_cuda_int64_tensor_like(beliefs);
    }
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
    if (allow_aten_fallback) {
        return causal_machine_scan_backward_tiled_probs_cuda(
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

std::vector<torch::Tensor> causal_machine_scan_backward_masked_logits(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_final_belief, "grad_final_belief");
    check_cuda_float32(transition_source_logits, "transition_source_logits");
    check_cuda_float32(transition_dest_logits, "transition_dest_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    TORCH_CHECK(beliefs.dim() == 3, "beliefs must have shape [B, L, N]");
    TORCH_CHECK(grad_beliefs.sizes() == beliefs.sizes(), "grad_beliefs must match beliefs shape");
    TORCH_CHECK(transition_context.sizes() == beliefs.sizes(), "transition_context must match beliefs shape");
    TORCH_CHECK(transition_source_logits.dim() == 2, "transition_source_logits must have shape [N, R]");
    TORCH_CHECK(transition_dest_logits.dim() == 2, "transition_dest_logits must have shape [R, N]");
    TORCH_CHECK(transition_stay_probs.dim() == 1, "transition_stay_probs must have shape [N]");
    TORCH_CHECK(initial_log_belief.dim() == 2, "initial_log_belief must have shape [B, N]");
    TORCH_CHECK(
        initial_log_belief.sizes() == grad_final_belief.sizes(),
        "grad_final_belief must match initial_log_belief shape"
    );
    TORCH_CHECK(
        transition_source_logits.size(0) == beliefs.size(2),
        "transition_source_logits first dim must match beliefs last dim"
    );
    TORCH_CHECK(
        transition_dest_logits.size(1) == beliefs.size(2),
        "transition_dest_logits last dim must match beliefs last dim"
    );
    TORCH_CHECK(
        transition_dest_logits.size(0) == transition_source_logits.size(1),
        "transition_source_logits and transition_dest_logits rank must match"
    );
    TORCH_CHECK(
        transition_stay_probs.size(0) == beliefs.size(2),
        "transition_stay_probs size must match beliefs last dim"
    );
    TORCH_CHECK(
        initial_log_belief.size(0) == beliefs.size(0) && initial_log_belief.size(1) == beliefs.size(2),
        "initial_log_belief must match beliefs batch/state shape"
    );
    check_same_cuda_device(grad_beliefs, beliefs, "grad_beliefs");
    check_same_cuda_device(grad_final_belief, beliefs, "grad_final_belief");
    check_same_cuda_device(transition_source_logits, beliefs, "transition_source_logits");
    check_same_cuda_device(transition_dest_logits, beliefs, "transition_dest_logits");
    check_same_cuda_device(transition_context, beliefs, "transition_context");
    check_same_cuda_device(initial_log_belief, beliefs, "initial_log_belief");
    check_same_cuda_device(transition_stay_probs, beliefs, "transition_stay_probs");
    TORCH_CHECK(grad_beliefs.scalar_type() == beliefs.scalar_type(), "grad_beliefs must match beliefs dtype");
    TORCH_CHECK(grad_final_belief.scalar_type() == beliefs.scalar_type(), "grad_final_belief must match beliefs dtype");
    TORCH_CHECK(transition_context.scalar_type() == beliefs.scalar_type(), "transition_context must match beliefs dtype");
    TORCH_CHECK(initial_log_belief.scalar_type() == beliefs.scalar_type(), "initial_log_belief must match beliefs dtype");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (is_optional_tensor_defined(transition_mask)) {
        check_cuda_bool(transition_mask, "transition_mask");
        check_same_cuda_device(transition_mask, beliefs, "transition_mask");
        TORCH_CHECK(
            transition_mask.dim() == 2
                && transition_mask.size(0) == beliefs.size(2)
                && transition_mask.size(1) == beliefs.size(2),
            "transition_mask must have shape [N, N]"
        );
    } else {
        transition_mask = empty_cuda_bool_tensor_like(beliefs);
    }
    if (is_optional_tensor_defined(seq_lens)) {
        check_cuda_int64(seq_lens, "seq_lens");
        check_same_cuda_device(seq_lens, beliefs, "seq_lens");
        TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
        TORCH_CHECK(seq_lens.size(0) == beliefs.size(0), "seq_lens must have shape [B]");
    } else {
        seq_lens = empty_cuda_int64_tensor_like(beliefs);
    }
    if (beliefs.size(0) == 0 || beliefs.size(1) == 0) {
        return {
            torch::zeros_like(beliefs),
            torch::zeros_like(transition_source_logits),
            torch::zeros_like(transition_dest_logits),
            torch::zeros_like(transition_context),
            torch::zeros_like(initial_log_belief),
            torch::zeros({1}, beliefs.options().dtype(torch::kFloat32)),
            torch::zeros_like(transition_stay_probs),
        };
    }

    const auto num_states = beliefs.size(2);
    const bool native_score_filtering = std::isfinite(score_threshold) || score_topk > 0;
    if (is_supported_specialized_num_states(num_states)) {
        TORCH_CHECK(
            !native_score_filtering,
            "backward_masked_logits native threshold/topk requires the masked tiled custom kernel path"
        );
        return causal_machine_scan_backward_masked_logits_cuda(
            grad_beliefs,
            grad_final_belief,
            transition_source_logits,
            transition_dest_logits,
            transition_context,
            initial_log_belief,
            beliefs,
            transition_gate,
            transition_stay_probs,
            transition_mask,
            seq_lens,
            chunk_size,
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk);
    }
    TORCH_CHECK(
        num_states > kSpecializedNumStates,
        "backward_masked_logits requires a transition mask and either a specialized small-state kernel or num_states > ",
        kSpecializedNumStates
    );
    const int64_t tile_size = num_states;
    const bool custom_kernel_supported = causal_machine_scan_can_use_masked_tiled_backward_kernel_cuda(
        beliefs.get_device(),
        num_states,
        tile_size);
    if (custom_kernel_supported) {
        return causal_machine_scan_backward_masked_logits_cuda(
            grad_beliefs,
            grad_final_belief,
            transition_source_logits,
            transition_dest_logits,
            transition_context,
            initial_log_belief,
            beliefs,
            transition_gate,
            transition_stay_probs,
            transition_mask,
            seq_lens,
            chunk_size,
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk);
    }
    TORCH_CHECK(
        !native_score_filtering,
        "backward_masked_logits native threshold/topk requires the masked tiled custom kernel path"
    );
    auto sparse_meta = build_masked_sparse_fallback_metadata(num_states, transition_mask);
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, beliefs);
    return causal_machine_scan_backward_sparse_logits_fused(
        grad_beliefs,
        grad_final_belief,
        transition_source_logits,
        transition_dest_logits,
        sparse_meta.row_ptr,
        sparse_meta.col_idx,
        sparse_meta.dst_idx,
        sparse_meta.src_row_ptr,
        sparse_meta.src_nz_idx,
        sparse_meta.grouped_src_row_ptr,
        sparse_meta.grouped_src_block_idx,
        sparse_meta.block_mask,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate_f32,
        transition_stay_probs,
        seq_lens,
        sparse_meta.block_size,
        chunk_size);
}

std::vector<torch::Tensor> causal_machine_scan_backward_masked_logits_workspace(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
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
    check_cuda_int32(work_queue_counter, "work_queue_counter");
    check_cuda_float32(masked_transition_tile_cache, "masked_transition_tile_cache");
    check_cuda_float32(row_sums, "row_sums");
    check_same_cuda_device(work_queue_counter, beliefs, "work_queue_counter");
    check_same_cuda_device(masked_transition_tile_cache, beliefs, "masked_transition_tile_cache");
    check_same_cuda_device(row_sums, beliefs, "row_sums");
    const bool use_masked_tiled = beliefs.size(2) > kSpecializedNumStates;
    if (use_masked_tiled) {
        auto workspace_info = causal_machine_scan_describe_workspace_config(
            "masked_tiled_backward",
            beliefs.size(2),
            transition_source_logits.size(1),
            beliefs.size(0),
            beliefs.size(2),
            1,
            beliefs.size(1),
            chunk_size,
            beliefs.get_device());
        check_workspace_shape_min(
            work_queue_counter,
            workspace_info["work_queue_counter_shape"].cast<std::vector<int64_t>>(),
            "work_queue_counter");
        check_workspace_shape_min(
            masked_transition_tile_cache,
            workspace_info["masked_transition_tile_cache_shape"].cast<std::vector<int64_t>>(),
            "masked_transition_tile_cache");
        check_workspace_shape_min(
            row_sums,
            workspace_info["row_sums_shape"].cast<std::vector<int64_t>>(),
            "row_sums");
    }
    return causal_machine_scan_backward_masked_logits_workspace_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        beliefs,
        normalize_transition_gate_tensor(transition_gate, beliefs),
        transition_stay_probs,
        transition_mask,
        seq_lens,
        chunk_size,
        work_queue_counter,
        masked_transition_tile_cache,
        row_sums,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_backward_masked_logits_bound_workspace(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor transition_mask,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    py::dict workspace,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    const auto num_states = beliefs.size(2);
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, beliefs);
    if (num_states <= kSpecializedNumStates) {
        return causal_machine_scan_backward_masked_logits(
            grad_beliefs,
            grad_final_belief,
            transition_source_logits,
            transition_dest_logits,
            transition_context,
            initial_log_belief,
            beliefs,
            transition_gate_f32,
            transition_stay_probs,
            transition_mask,
            seq_lens,
            chunk_size,
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk);
    }
    const bool custom_kernel_supported = causal_machine_scan_can_use_masked_tiled_backward_kernel_cuda(
        beliefs.get_device(),
        num_states,
        num_states);
    if (!custom_kernel_supported) {
        return causal_machine_scan_backward_masked_logits(
            grad_beliefs,
            grad_final_belief,
            transition_source_logits,
            transition_dest_logits,
            transition_context,
            initial_log_belief,
            beliefs,
            transition_gate_f32,
            transition_stay_probs,
            transition_mask,
            seq_lens,
            chunk_size,
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk);
    }
    auto work_queue_counter = get_workspace_tensor(workspace, "work_queue_counter", torch::kInt32, beliefs);
    auto masked_transition_tile_cache = get_workspace_tensor(workspace, "masked_transition_tile_cache", torch::kFloat32, beliefs);
    auto row_sums = get_workspace_tensor(workspace, "row_sums", torch::kFloat32, beliefs);
    return causal_machine_scan_backward_masked_logits_workspace(
        grad_beliefs,
        grad_final_belief,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        beliefs,
        transition_gate_f32,
        transition_stay_probs,
        transition_mask,
        seq_lens,
        chunk_size,
        work_queue_counter,
        masked_transition_tile_cache,
        row_sums,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    return causal_machine_scan_backward_tiled_probs_impl(
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
        score_topk,
        true);
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_kernel(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    return causal_machine_scan_backward_tiled_probs_impl(
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
        score_topk,
        false);
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_kernel_workspace(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
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
    check_cuda_int32(work_queue_counter, "work_queue_counter");
    check_cuda_float32(latent_cache_staging, "latent_cache_staging");
    check_cuda_float32(grad_latent_accum_staging, "grad_latent_accum_staging");
    check_cuda_float32(grad_transition_source_probs_staging, "grad_transition_source_probs_staging");
    check_cuda_float32(grad_transition_dest_probs_staging, "grad_transition_dest_probs_staging");
    check_cuda_float32(grad_transition_gate_staging, "grad_transition_gate_staging");
    check_cuda_float32(grad_transition_stay_staging, "grad_transition_stay_staging");
    check_same_cuda_device(work_queue_counter, beliefs, "work_queue_counter");
    check_same_cuda_device(latent_cache_staging, beliefs, "latent_cache_staging");
    check_same_cuda_device(grad_latent_accum_staging, beliefs, "grad_latent_accum_staging");
    check_same_cuda_device(grad_transition_source_probs_staging, beliefs, "grad_transition_source_probs_staging");
    check_same_cuda_device(grad_transition_dest_probs_staging, beliefs, "grad_transition_dest_probs_staging");
    check_same_cuda_device(grad_transition_gate_staging, beliefs, "grad_transition_gate_staging");
    check_same_cuda_device(grad_transition_stay_staging, beliefs, "grad_transition_stay_staging");
    return causal_machine_scan_backward_tiled_probs_kernel_workspace_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_probs,
        transition_dest_probs,
        transition_context,
        initial_log_belief,
        beliefs,
        normalize_transition_gate_tensor(transition_gate, beliefs),
        transition_stay_probs,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        latent_cache_staging,
        grad_latent_accum_staging,
        grad_transition_source_probs_staging,
        grad_transition_dest_probs_staging,
        grad_transition_gate_staging,
        grad_transition_stay_staging,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_probs_kernel_bound_workspace(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_probs,
    torch::Tensor transition_dest_probs,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    py::dict workspace,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    auto work_queue_counter = get_workspace_tensor(workspace, "work_queue_counter", torch::kInt32, beliefs);
    auto latent_cache_staging = get_workspace_tensor(workspace, "latent_cache_staging", torch::kFloat32, beliefs);
    auto grad_latent_accum_staging = get_workspace_tensor(workspace, "grad_latent_accum_staging", torch::kFloat32, beliefs);
    auto grad_transition_source_probs_staging = get_workspace_tensor(workspace, "grad_transition_source_probs_staging", torch::kFloat32, beliefs);
    auto grad_transition_dest_probs_staging = get_workspace_tensor(workspace, "grad_transition_dest_probs_staging", torch::kFloat32, beliefs);
    auto grad_transition_gate_staging = get_workspace_tensor(workspace, "grad_transition_gate_staging", torch::kFloat32, beliefs);
    auto grad_transition_stay_staging = get_workspace_tensor(workspace, "grad_transition_stay_staging", torch::kFloat32, beliefs);
    return causal_machine_scan_backward_tiled_probs_kernel_workspace(
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
        work_queue_counter,
        latent_cache_staging,
        grad_latent_accum_staging,
        grad_transition_source_probs_staging,
        grad_transition_dest_probs_staging,
        grad_transition_gate_staging,
        grad_transition_stay_staging,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_quantized_kernel_workspace(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
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
    check_cuda_int32(work_queue_counter, "work_queue_counter");
    check_cuda_float32(latent_cache_staging, "latent_cache_staging");
    check_cuda_float32(grad_latent_accum_staging, "grad_latent_accum_staging");
    check_cuda_float32(grad_transition_source_probs_staging, "grad_transition_source_probs_staging");
    check_cuda_float32(grad_transition_dest_probs_staging, "grad_transition_dest_probs_staging");
    check_cuda_float32(grad_transition_gate_staging, "grad_transition_gate_staging");
    check_cuda_float32(grad_transition_stay_staging, "grad_transition_stay_staging");
    check_same_cuda_device(work_queue_counter, beliefs, "work_queue_counter");
    check_same_cuda_device(latent_cache_staging, beliefs, "latent_cache_staging");
    check_same_cuda_device(grad_latent_accum_staging, beliefs, "grad_latent_accum_staging");
    check_same_cuda_device(grad_transition_source_probs_staging, beliefs, "grad_transition_source_probs_staging");
    check_same_cuda_device(grad_transition_dest_probs_staging, beliefs, "grad_transition_dest_probs_staging");
    check_same_cuda_device(grad_transition_gate_staging, beliefs, "grad_transition_gate_staging");
    check_same_cuda_device(grad_transition_stay_staging, beliefs, "grad_transition_stay_staging");
    return causal_machine_scan_backward_tiled_quantized_kernel_workspace_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_q,
        transition_source_scales,
        transition_dest_q,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        beliefs,
        normalize_transition_gate_tensor(transition_gate, beliefs),
        transition_stay_probs,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        latent_cache_staging,
        grad_latent_accum_staging,
        grad_transition_source_probs_staging,
        grad_transition_dest_probs_staging,
        grad_transition_gate_staging,
        grad_transition_stay_staging,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_quantized_kernel_bound_workspace(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_q,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_q,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    py::dict workspace,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    auto work_queue_counter = get_workspace_tensor(workspace, "work_queue_counter", torch::kInt32, beliefs);
    auto latent_cache_staging = get_workspace_tensor(workspace, "latent_cache_staging", torch::kFloat32, beliefs);
    auto grad_latent_accum_staging = get_workspace_tensor(workspace, "grad_latent_accum_staging", torch::kFloat32, beliefs);
    auto grad_transition_source_probs_staging = get_workspace_tensor(workspace, "grad_transition_source_probs_staging", torch::kFloat32, beliefs);
    auto grad_transition_dest_probs_staging = get_workspace_tensor(workspace, "grad_transition_dest_probs_staging", torch::kFloat32, beliefs);
    auto grad_transition_gate_staging = get_workspace_tensor(workspace, "grad_transition_gate_staging", torch::kFloat32, beliefs);
    auto grad_transition_stay_staging = get_workspace_tensor(workspace, "grad_transition_stay_staging", torch::kFloat32, beliefs);
    return causal_machine_scan_backward_tiled_quantized_kernel_workspace(
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
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        latent_cache_staging,
        grad_latent_accum_staging,
        grad_transition_source_probs_staging,
        grad_transition_dest_probs_staging,
        grad_transition_gate_staging,
        grad_transition_stay_staging,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_fp8_kernel_workspace(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
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
    check_cuda_int32(work_queue_counter, "work_queue_counter");
    check_cuda_float32(latent_cache_staging, "latent_cache_staging");
    check_cuda_float32(grad_latent_accum_staging, "grad_latent_accum_staging");
    check_cuda_float32(grad_transition_source_probs_staging, "grad_transition_source_probs_staging");
    check_cuda_float32(grad_transition_dest_probs_staging, "grad_transition_dest_probs_staging");
    check_cuda_float32(grad_transition_gate_staging, "grad_transition_gate_staging");
    check_cuda_float32(grad_transition_stay_staging, "grad_transition_stay_staging");
    check_same_cuda_device(work_queue_counter, beliefs, "work_queue_counter");
    check_same_cuda_device(latent_cache_staging, beliefs, "latent_cache_staging");
    check_same_cuda_device(grad_latent_accum_staging, beliefs, "grad_latent_accum_staging");
    check_same_cuda_device(grad_transition_source_probs_staging, beliefs, "grad_transition_source_probs_staging");
    check_same_cuda_device(grad_transition_dest_probs_staging, beliefs, "grad_transition_dest_probs_staging");
    check_same_cuda_device(grad_transition_gate_staging, beliefs, "grad_transition_gate_staging");
    check_same_cuda_device(grad_transition_stay_staging, beliefs, "grad_transition_stay_staging");
    return causal_machine_scan_backward_tiled_fp8_kernel_workspace_cuda(
        grad_beliefs,
        grad_final_belief,
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        beliefs,
        normalize_transition_gate_tensor(transition_gate, beliefs),
        transition_stay_probs,
        fp8_format,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        latent_cache_staging,
        grad_latent_accum_staging,
        grad_transition_source_probs_staging,
        grad_transition_dest_probs_staging,
        grad_transition_gate_staging,
        grad_transition_stay_staging,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_backward_tiled_fp8_kernel_bound_workspace(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    torch::Tensor seq_lens,
    int64_t chunk_size,
    int64_t tile_size,
    int64_t split_size,
    py::dict workspace,
    double score_clamp_min,
    double score_clamp_max,
    double score_threshold,
    int64_t score_topk) {
    auto work_queue_counter = get_workspace_tensor(workspace, "work_queue_counter", torch::kInt32, beliefs);
    auto latent_cache_staging = get_workspace_tensor(workspace, "latent_cache_staging", torch::kFloat32, beliefs);
    auto grad_latent_accum_staging = get_workspace_tensor(workspace, "grad_latent_accum_staging", torch::kFloat32, beliefs);
    auto grad_transition_source_probs_staging = get_workspace_tensor(workspace, "grad_transition_source_probs_staging", torch::kFloat32, beliefs);
    auto grad_transition_dest_probs_staging = get_workspace_tensor(workspace, "grad_transition_dest_probs_staging", torch::kFloat32, beliefs);
    auto grad_transition_gate_staging = get_workspace_tensor(workspace, "grad_transition_gate_staging", torch::kFloat32, beliefs);
    auto grad_transition_stay_staging = get_workspace_tensor(workspace, "grad_transition_stay_staging", torch::kFloat32, beliefs);
    return causal_machine_scan_backward_tiled_fp8_kernel_workspace(
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
        fp8_format,
        seq_lens,
        chunk_size,
        tile_size,
        split_size,
        work_queue_counter,
        latent_cache_staging,
        grad_latent_accum_staging,
        grad_transition_source_probs_staging,
        grad_transition_dest_probs_staging,
        grad_transition_gate_staging,
        grad_transition_stay_staging,
        score_clamp_min,
        score_clamp_max,
        score_threshold,
        score_topk);
}

std::vector<torch::Tensor> causal_machine_scan_backward_logits(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size,
    double score_clamp_min,
    double score_clamp_max) {
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_final_belief, "grad_final_belief");
    check_cuda_float32(transition_source_logits, "transition_source_logits");
    check_cuda_float32(transition_dest_logits, "transition_dest_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_shapes(
        beliefs,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_devices(
        beliefs,
        transition_source_logits,
        transition_dest_logits,
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
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, beliefs);
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (beliefs.size(0) == 0 || beliefs.size(1) == 0) {
        return {
            torch::zeros_like(beliefs),
            torch::zeros_like(transition_source_logits),
            torch::zeros_like(transition_dest_logits),
            torch::zeros_like(transition_context),
            torch::zeros_like(initial_log_belief),
            torch::zeros({1}, beliefs.options().dtype(torch::kFloat32)),
            torch::zeros_like(transition_stay_probs),
        };
    }
    const auto num_states = beliefs.size(2);
    return dispatch_specialized_backward_wrapper(
        num_states,
        transition_source_logits.size(1),
        [&]() {
            auto padded_grad_beliefs = pad_last_dim(grad_beliefs, kSpecializedNumStates, 0.0);
            auto padded_grad_final_belief = pad_last_dim(grad_final_belief, kSpecializedNumStates, 0.0);
            auto padded_transition_source_logits = pad_first_dim(transition_source_logits, kSpecializedNumStates, 0.0);
            auto padded_transition_dest_logits = pad_last_dim(transition_dest_logits, kSpecializedNumStates, neg_inf_fill());
            auto padded_transition_context = pad_last_dim(transition_context, kSpecializedNumStates, 0.0);
            auto padded_initial_log_belief = pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill());
            auto padded_beliefs = pad_last_dim(beliefs, kSpecializedNumStates, neg_inf_fill());
            auto padded_transition_stay_probs = pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0);
            auto workspace = get_dense_backward_workspace(
                padded_beliefs,
                padded_beliefs.size(2),
                padded_transition_source_logits.size(1));
            return causal_machine_scan_backward_logits_workspace_cuda(
                padded_grad_beliefs,
                padded_grad_final_belief,
                padded_transition_source_logits,
                padded_transition_dest_logits,
                padded_transition_context,
                padded_initial_log_belief,
                padded_beliefs,
                transition_gate_f32,
                padded_transition_stay_probs,
                chunk_size,
                score_clamp_min,
                score_clamp_max,
                workspace.grad_transition_source_per_batch,
                workspace.grad_transition_dest_per_batch,
                workspace.grad_transition_stay_per_batch,
                workspace.grad_transition_gate_per_batch);
        },
        [&]() {
            auto workspace = get_dense_backward_workspace(
                beliefs,
                beliefs.size(2),
                transition_source_logits.size(1));
            return causal_machine_scan_backward_logits_workspace_cuda(
                grad_beliefs,
                grad_final_belief,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                beliefs,
                transition_gate_f32,
                transition_stay_probs,
                chunk_size,
                score_clamp_min,
                score_clamp_max,
                workspace.grad_transition_source_per_batch,
                workspace.grad_transition_dest_per_batch,
                workspace.grad_transition_stay_per_batch,
                workspace.grad_transition_gate_per_batch);
        });
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
    torch::Tensor transition_gate,
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
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, beliefs);
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
    const auto num_states = beliefs.size(2);
    if (num_states > kSpecializedNumStates) {
        const auto geometry = choose_large_state_tiled_geometry(
            beliefs.get_device(),
            num_states,
            transition_source_q.size(1),
            true);
        return causal_machine_scan_backward_tiled_quantized_kernel_cuda(
            grad_beliefs,
            grad_final_belief,
            transition_source_q,
            transition_source_scales,
            transition_dest_q,
            transition_dest_scales,
            transition_context,
            initial_log_belief.to(torch::kFloat32),
            beliefs,
            transition_gate_f32,
            transition_stay_probs,
            torch::Tensor(),
            chunk_size,
            geometry.first,
            geometry.second,
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            0);
    }
    return dispatch_specialized_backward_wrapper(
        num_states,
        transition_source_q.size(1),
        [&]() {
            return causal_machine_scan_backward_quantized_cuda(
                pad_last_dim(grad_beliefs, kSpecializedNumStates, 0.0),
                pad_last_dim(grad_final_belief, kSpecializedNumStates, 0.0),
                pad_first_dim(transition_source_q, kSpecializedNumStates, 0.0),
                pad_last_dim(transition_source_scales, kSpecializedNumStates, 1.0),
                pad_last_dim(transition_dest_q, kSpecializedNumStates, 0.0),
                transition_dest_scales,
                pad_last_dim(transition_context, kSpecializedNumStates, 0.0),
                pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill()),
                pad_last_dim(beliefs, kSpecializedNumStates, neg_inf_fill()),
                transition_gate_f32,
                pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0),
                chunk_size);
        },
        [&]() {
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
                transition_gate_f32,
                transition_stay_probs,
                chunk_size);
        });
}

std::vector<torch::Tensor> causal_machine_scan_backward_fp8(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_packed,
    torch::Tensor transition_source_scales,
    torch::Tensor transition_dest_packed,
    torch::Tensor transition_dest_scales,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_gate,
    torch::Tensor transition_stay_probs,
    int64_t fp8_format,
    int64_t chunk_size) {
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_final_belief, "grad_final_belief");
    check_cuda_uint8(transition_source_packed, "transition_source_packed");
    check_cuda_float32(transition_source_scales, "transition_source_scales");
    check_cuda_uint8(transition_dest_packed, "transition_dest_packed");
    check_cuda_float32(transition_dest_scales, "transition_dest_scales");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_fp8_shapes(
        beliefs,
        transition_source_packed,
        transition_source_scales,
        transition_dest_packed,
        transition_dest_scales,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_devices(
        beliefs,
        transition_source_packed,
        transition_dest_packed,
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
    TORCH_CHECK(fp8_format == 0 || fp8_format == 1, "fp8_format must be 0 (e4m3) or 1 (e5m2)");
    auto transition_gate_f32 = normalize_transition_gate_tensor(transition_gate, beliefs);
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (beliefs.size(0) == 0 || beliefs.size(1) == 0) {
        return {
            torch::zeros_like(beliefs),
            torch::zeros({transition_source_packed.size(0), transition_source_packed.size(1)}, beliefs.options().dtype(torch::kFloat32)),
            torch::zeros({transition_dest_packed.size(0), transition_dest_packed.size(1)}, beliefs.options().dtype(torch::kFloat32)),
            torch::zeros_like(transition_context),
            torch::zeros_like(initial_log_belief),
            torch::zeros({1}, beliefs.options().dtype(torch::kFloat32)),
            torch::zeros_like(transition_stay_probs),
        };
    }
    const auto num_states = beliefs.size(2);
    if (num_states > kSpecializedNumStates) {
        const auto geometry = choose_large_state_tiled_geometry(
            beliefs.get_device(),
            num_states,
            transition_source_packed.size(1),
            true);
        return causal_machine_scan_backward_tiled_fp8_kernel_cuda(
            grad_beliefs,
            grad_final_belief,
            transition_source_packed,
            transition_source_scales,
            transition_dest_packed,
            transition_dest_scales,
            transition_context,
            initial_log_belief.to(torch::kFloat32),
            beliefs,
            transition_gate,
            transition_stay_probs,
            fp8_format,
            torch::Tensor(),
            chunk_size,
            geometry.first,
            geometry.second,
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            0);
    }
    return dispatch_specialized_backward_wrapper(
        num_states,
        transition_source_packed.size(1),
        [&]() {
            return causal_machine_scan_backward_fp8_cuda(
                pad_last_dim(grad_beliefs, kSpecializedNumStates, 0.0),
                pad_last_dim(grad_final_belief, kSpecializedNumStates, 0.0),
                pad_first_dim(transition_source_packed, kSpecializedNumStates, 0.0),
                pad_last_dim(transition_source_scales, kSpecializedNumStates, 1.0),
                pad_last_dim(transition_dest_packed, kSpecializedNumStates, 0.0),
                transition_dest_scales,
                pad_last_dim(transition_context, kSpecializedNumStates, 0.0),
                pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill()),
                pad_last_dim(beliefs, kSpecializedNumStates, neg_inf_fill()),
                transition_gate_f32,
                pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0),
                fp8_format,
                chunk_size);
        },
        [&]() {
            return causal_machine_scan_backward_fp8_cuda(
                grad_beliefs,
                grad_final_belief,
                transition_source_packed,
                transition_source_scales,
                transition_dest_packed,
                transition_dest_scales,
                transition_context,
                initial_log_belief,
                beliefs,
                transition_gate_f32,
                transition_stay_probs,
                fp8_format,
                chunk_size);
        });
}

std::vector<torch::Tensor> causal_machine_scan_backward_composable_logits(
    torch::Tensor grad_beliefs,
    torch::Tensor grad_final_belief,
    torch::Tensor transition_source_logits,
    torch::Tensor transition_dest_logits,
    torch::Tensor transition_context,
    torch::Tensor initial_log_belief,
    torch::Tensor beliefs,
    torch::Tensor transition_stay_probs,
    int64_t chunk_size) {
    check_cuda_activation(grad_beliefs, "grad_beliefs");
    check_cuda_activation(grad_final_belief, "grad_final_belief");
    check_cuda_float32(transition_source_logits, "transition_source_logits");
    check_cuda_float32(transition_dest_logits, "transition_dest_logits");
    check_cuda_activation(transition_context, "transition_context");
    check_cuda_activation(initial_log_belief, "initial_log_belief");
    check_cuda_activation(beliefs, "beliefs");
    check_cuda_float32(transition_stay_probs, "transition_stay_probs");
    check_structured_shapes(
        grad_beliefs,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    TORCH_CHECK(beliefs.sizes() == grad_beliefs.sizes(), "beliefs must have the same shape as grad_beliefs");
    check_same_cuda_devices(
        grad_beliefs,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_stay_probs
    );
    check_same_cuda_device(grad_final_belief, grad_beliefs, "grad_final_belief");
    check_same_cuda_device(beliefs, grad_beliefs, "beliefs");
    TORCH_CHECK(
        grad_final_belief.sizes() == initial_log_belief.sizes(),
        "grad_final_belief must match initial_log_belief shape"
    );
    TORCH_CHECK(grad_final_belief.scalar_type() == grad_beliefs.scalar_type(), "grad_final_belief must match grad_beliefs dtype");
    TORCH_CHECK(beliefs.scalar_type() == grad_beliefs.scalar_type(), "beliefs must match grad_beliefs dtype");
    TORCH_CHECK(transition_context.scalar_type() == grad_beliefs.scalar_type(), "transition_context must match grad_beliefs dtype");
    TORCH_CHECK(initial_log_belief.scalar_type() == grad_beliefs.scalar_type(), "initial_log_belief must match grad_beliefs dtype");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    if (grad_beliefs.size(0) == 0 || grad_beliefs.size(1) == 0) {
        return {
            torch::zeros_like(grad_beliefs),
            torch::zeros_like(transition_source_logits),
            torch::zeros_like(transition_dest_logits),
            torch::zeros_like(transition_context),
            torch::zeros_like(initial_log_belief),
            torch::zeros({1}, grad_beliefs.options().dtype(torch::kFloat32)),
            torch::zeros_like(transition_stay_probs),
        };
    }
    const auto num_states = grad_beliefs.size(2);
    return dispatch_specialized_backward_wrapper(
        num_states,
        transition_source_logits.size(1),
        [&]() {
            return causal_machine_scan_backward_composable_logits_cuda(
                pad_last_dim(grad_beliefs, kSpecializedNumStates, 0.0),
                pad_last_dim(grad_final_belief, kSpecializedNumStates, 0.0),
                pad_first_dim(transition_source_logits, kSpecializedNumStates, 0.0),
                pad_last_dim(transition_dest_logits, kSpecializedNumStates, neg_inf_fill()),
                pad_last_dim(transition_context, kSpecializedNumStates, 0.0),
                pad_last_dim(initial_log_belief, kSpecializedNumStates, neg_inf_fill()),
                pad_last_dim(beliefs, kSpecializedNumStates, neg_inf_fill()),
                pad_last_dim(transition_stay_probs, kSpecializedNumStates, 1.0),
                chunk_size);
        },
        [&]() {
            return causal_machine_scan_backward_composable_logits_cuda(
                grad_beliefs,
                grad_final_belief,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                beliefs,
                transition_stay_probs,
            chunk_size);
        });
}

torch::Tensor causal_machine_decode_belief_projection(
    torch::Tensor state_log_beliefs,
    torch::Tensor belief_out_weight) {
    check_cuda_activation(state_log_beliefs, "state_log_beliefs");
    check_cuda_float32(belief_out_weight, "belief_out_weight");
    TORCH_CHECK(state_log_beliefs.dim() == 3, "state_log_beliefs must have shape [B, L, N]");
    TORCH_CHECK(belief_out_weight.dim() == 2, "belief_out_weight must have shape [D, N]");
    TORCH_CHECK(
        belief_out_weight.size(1) == state_log_beliefs.size(2),
        "belief_out_weight second dim must match state_log_beliefs last dim"
    );
    TORCH_CHECK(
        state_log_beliefs.size(2) > 0 && state_log_beliefs.size(2) <= 128,
        "fused belief decode supports num_states in [1, 128]"
    );
    check_same_cuda_device(belief_out_weight, state_log_beliefs, "belief_out_weight");
    return causal_machine_decode_belief_projection_cuda(
        state_log_beliefs.contiguous(),
        belief_out_weight.contiguous());
}

torch::Tensor causal_machine_decode_belief_projection_backward_input(
    torch::Tensor grad_output,
    torch::Tensor state_log_beliefs,
    torch::Tensor belief_out_weight) {
    check_cuda_activation(grad_output, "grad_output");
    check_cuda_activation(state_log_beliefs, "state_log_beliefs");
    check_cuda_float32(belief_out_weight, "belief_out_weight");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output must have shape [B, L, D]");
    TORCH_CHECK(state_log_beliefs.dim() == 3, "state_log_beliefs must have shape [B, L, N]");
    TORCH_CHECK(belief_out_weight.dim() == 2, "belief_out_weight must have shape [D, N]");
    TORCH_CHECK(
        grad_output.size(0) == state_log_beliefs.size(0)
            && grad_output.size(1) == state_log_beliefs.size(1),
        "grad_output must match state_log_beliefs batch and sequence dimensions"
    );
    TORCH_CHECK(
        grad_output.size(2) == belief_out_weight.size(0),
        "grad_output last dim must match belief_out_weight first dim"
    );
    TORCH_CHECK(
        belief_out_weight.size(1) == state_log_beliefs.size(2),
        "belief_out_weight second dim must match state_log_beliefs last dim"
    );
    TORCH_CHECK(
        state_log_beliefs.size(2) > 0 && state_log_beliefs.size(2) <= 128,
        "fused belief decode supports num_states in [1, 128]"
    );
    TORCH_CHECK(
        grad_output.scalar_type() == state_log_beliefs.scalar_type(),
        "grad_output must match state_log_beliefs dtype"
    );
    check_same_cuda_device(state_log_beliefs, grad_output, "state_log_beliefs");
    check_same_cuda_device(belief_out_weight, grad_output, "belief_out_weight");
    return causal_machine_decode_belief_projection_backward_input_cuda(
        grad_output.contiguous(),
        state_log_beliefs.contiguous(),
        belief_out_weight.contiguous());
}

torch::Tensor causal_machine_decode_belief_projection_backward_weight(
    torch::Tensor grad_output,
    torch::Tensor state_log_beliefs) {
    check_cuda_activation(grad_output, "grad_output");
    check_cuda_activation(state_log_beliefs, "state_log_beliefs");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output must have shape [B, L, D]");
    TORCH_CHECK(state_log_beliefs.dim() == 3, "state_log_beliefs must have shape [B, L, N]");
    TORCH_CHECK(
        grad_output.size(0) == state_log_beliefs.size(0)
            && grad_output.size(1) == state_log_beliefs.size(1),
        "grad_output must match state_log_beliefs batch and sequence dimensions"
    );
    TORCH_CHECK(
        state_log_beliefs.size(2) > 0 && state_log_beliefs.size(2) <= 128,
        "fused belief decode supports num_states in [1, 128]"
    );
    TORCH_CHECK(
        grad_output.scalar_type() == state_log_beliefs.scalar_type(),
        "grad_output must match state_log_beliefs dtype"
    );
    check_same_cuda_device(state_log_beliefs, grad_output, "state_log_beliefs");
    return causal_machine_decode_belief_projection_backward_weight_cuda(
        grad_output.contiguous(),
        state_log_beliefs.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_logits", &causal_machine_scan_forward_logits, "Causal machine structured scan forward from logits (CUDA)");
    m.def("forward_masked_logits", &causal_machine_scan_forward_masked_logits, "Causal machine structured scan forward for masked CUDA execution");
    m.def("forward_masked_logits_bound_workspace", &causal_machine_scan_forward_masked_logits_bound_workspace, "Causal machine structured scan forward for masked CUDA execution with extension-bound workspace");
    m.def("backward_masked_logits", &causal_machine_scan_backward_masked_logits, "Causal machine structured scan backward from masked logits (CUDA)");
    m.def(
        "forward_tiled_logits",
        [](
            torch::Tensor local_logits,
            torch::Tensor transition_source_logits,
            torch::Tensor transition_dest_logits,
            torch::Tensor transition_context,
            torch::Tensor initial_log_belief,
            torch::Tensor transition_gate,
            torch::Tensor transition_stay_probs,
            torch::Tensor seq_lens,
            int64_t chunk_size,
            int64_t tile_size,
            int64_t split_size,
            double score_clamp_min,
            double score_clamp_max) {
            return causal_machine_scan_forward_tiled_logits(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                seq_lens,
                chunk_size,
                tile_size,
                split_size,
                score_clamp_min,
                score_clamp_max);
        },
        "Causal machine structured scan forward from logits with tiled split/combine CUDA execution");
    m.def("forward_tiled_logits_kernel", &causal_machine_scan_forward_tiled_logits_kernel, "Causal machine structured scan forward from probabilities with tiled custom CUDA execution");
    m.def("forward_tiled_logits_kernel_workspace", &causal_machine_scan_forward_tiled_logits_kernel_workspace, "Causal machine structured scan forward from probabilities with tiled custom CUDA execution and explicit workspace");
    m.def("forward_tiled_logits_kernel_bound_workspace", &causal_machine_scan_forward_tiled_logits_kernel_bound_workspace, "Causal machine structured scan forward from probabilities with tiled custom CUDA execution and extension-bound workspace");
    m.def("forward_tiled_quantized_kernel_workspace", &causal_machine_scan_forward_tiled_quantized_kernel_workspace, "Causal machine structured scan forward from quantized probabilities with tiled custom CUDA execution and explicit workspace");
    m.def("forward_tiled_quantized_kernel_bound_workspace", &causal_machine_scan_forward_tiled_quantized_kernel_bound_workspace, "Causal machine structured scan forward from quantized probabilities with tiled custom CUDA execution and extension-bound workspace");
    m.def("forward_tiled_fp8_kernel_workspace", &causal_machine_scan_forward_tiled_fp8_kernel_workspace, "Causal machine structured scan forward from FP8 packed probabilities with tiled custom CUDA execution and explicit workspace");
    m.def("forward_tiled_fp8_kernel_bound_workspace", &causal_machine_scan_forward_tiled_fp8_kernel_bound_workspace, "Causal machine structured scan forward from FP8 packed probabilities with tiled custom CUDA execution and extension-bound workspace");
    m.def("forward_masked_tiled_logits_kernel_workspace", &causal_machine_scan_forward_masked_tiled_logits_kernel_workspace, "Causal machine structured scan forward from masked probabilities with tiled custom CUDA execution and explicit workspace");
    m.def("forward_sparse_logits", &causal_machine_scan_forward_sparse_logits, "Causal machine structured scan forward from logits with block-sparse CUDA execution");
    m.def("forward_sparse_logits_fused", &causal_machine_scan_forward_sparse_logits_fused, "Causal machine structured scan forward from sparse transition logits with fused softmax/materialization in the CUDA extension");
    m.def("forward_composable_logits", &causal_machine_scan_forward_composable_logits, "Causal machine structured scan forward for composable mode from logits (CUDA)");
    m.def("forward", &causal_machine_scan_forward, "Causal machine structured scan forward (CUDA)");
    m.def("forward_quantized", &causal_machine_scan_forward_quantized, "Causal machine structured scan forward quantized (CUDA)");
    m.def("forward_fp8", &causal_machine_scan_forward_fp8, "Causal machine structured scan forward FP8 packed (CUDA)");
    m.def("backward_logits", &causal_machine_scan_backward_logits, "Causal machine structured scan backward from logits (CUDA)");
    m.def("backward_tiled_probs", &causal_machine_scan_backward_tiled_probs, "Causal machine structured scan backward from probabilities with tiled CUDA execution");
    m.def("backward_tiled_probs_kernel", &causal_machine_scan_backward_tiled_probs_kernel, "Causal machine structured scan backward from probabilities with tiled custom CUDA execution");
    m.def("backward_tiled_probs_kernel_workspace", &causal_machine_scan_backward_tiled_probs_kernel_workspace, "Causal machine structured scan backward from probabilities with tiled custom CUDA execution and explicit workspace");
    m.def("backward_tiled_probs_kernel_bound_workspace", &causal_machine_scan_backward_tiled_probs_kernel_bound_workspace, "Causal machine structured scan backward from probabilities with tiled custom CUDA execution and extension-bound workspace");
    m.def("backward_tiled_quantized_kernel_workspace", &causal_machine_scan_backward_tiled_quantized_kernel_workspace, "Causal machine structured scan backward from quantized probabilities with tiled custom CUDA execution and explicit workspace");
    m.def("backward_tiled_quantized_kernel_bound_workspace", &causal_machine_scan_backward_tiled_quantized_kernel_bound_workspace, "Causal machine structured scan backward from quantized probabilities with tiled custom CUDA execution and extension-bound workspace");
    m.def("backward_tiled_fp8_kernel_workspace", &causal_machine_scan_backward_tiled_fp8_kernel_workspace, "Causal machine structured scan backward from FP8 packed probabilities with tiled custom CUDA execution and explicit workspace");
    m.def("backward_tiled_fp8_kernel_bound_workspace", &causal_machine_scan_backward_tiled_fp8_kernel_bound_workspace, "Causal machine structured scan backward from FP8 packed probabilities with tiled custom CUDA execution and extension-bound workspace");
    m.def("backward_masked_logits_workspace", &causal_machine_scan_backward_masked_logits_workspace, "Causal machine structured scan backward from masked logits with explicit workspace");
    m.def("backward_masked_logits_bound_workspace", &causal_machine_scan_backward_masked_logits_bound_workspace, "Causal machine structured scan backward from masked logits with extension-bound workspace");
    m.def("backward_composable_logits", &causal_machine_scan_backward_composable_logits, "Causal machine structured scan backward for composable mode from logits (CUDA)");
    m.def("backward", &causal_machine_scan_backward, "Causal machine structured scan backward (CUDA)");
    m.def("backward_quantized", &causal_machine_scan_backward_quantized, "Causal machine structured scan backward quantized (CUDA)");
    m.def("backward_fp8", &causal_machine_scan_backward_fp8, "Causal machine structured scan backward FP8 packed (CUDA)");
    m.def("backward_sparse", &causal_machine_scan_backward_sparse, "Causal machine structured scan backward for block-sparse CUDA execution");
    m.def("backward_sparse_logits_fused", &causal_machine_scan_backward_sparse_logits_fused, "Causal machine structured scan backward from sparse transition logits with fused softmax/materialization in the CUDA extension");
    m.def("build_sparse_metadata", &causal_machine_scan_build_sparse_metadata, "Build block-sparse transition metadata from a dense transition mask (CUDA)");
    m.def(
        "build_sparse_metadata_from_runtime",
        [](int64_t num_states,
           int64_t padded_states,
           int64_t block_size,
           int64_t local_transition_window,
           torch::Tensor transition_mask,
           torch::Tensor runtime_block_mask) {
            return causal_machine_scan_build_sparse_metadata_from_runtime(
                num_states,
                padded_states,
                block_size,
                local_transition_window,
                transition_mask,
                runtime_block_mask);
        },
        "Build block-sparse transition metadata directly from supported runtime mask inputs (CUDA)"
    );
    m.def(
        "build_grouped_sparse_backward_metadata",
        [](torch::Tensor col_idx, torch::Tensor src_nz_idx) {
            return causal_machine_scan_build_grouped_sparse_backward_metadata(
                std::move(col_idx),
                std::move(src_nz_idx));
        },
        "Build grouped/compressed sparse backward metadata from src-sorted sparse indices (CUDA)");
    m.def("materialize_sparse_blocks", &causal_machine_scan_materialize_sparse_blocks, "Materialize normalized block-sparse transition blocks and row sums (CUDA)");
    m.def("materialize_sparse_blocks_int8", &causal_machine_scan_materialize_sparse_blocks_int8, "Materialize normalized block-sparse transition blocks and row sums from int8-packed transition tables (CUDA)");
    m.def("materialize_sparse_blocks_fp8", &causal_machine_scan_materialize_sparse_blocks_fp8, "Materialize normalized block-sparse transition blocks and row sums from FP8-packed transition tables (CUDA)");
    m.def("pack_int8", &causal_machine_scan_pack_int8_cuda, "Pack a structured transition table to int8 with per-row scales (CUDA)");
    m.def("pack_fp8_e4m3", &causal_machine_scan_pack_fp8_e4m3_cuda, "Pack a structured transition table to FP8 E4M3 with per-row scales (CUDA)");
    m.def("pack_fp8_e5m2", &causal_machine_scan_pack_fp8_e5m2_cuda, "Pack a structured transition table to FP8 E5M2 with per-row scales (CUDA)");
    m.def("unpack_int8", &causal_machine_scan_unpack_int8_cuda, "Unpack an int8 structured transition table back to float32 (CUDA)");
    m.def("unpack_fp8_e4m3", &causal_machine_scan_unpack_fp8_e4m3_cuda, "Unpack an FP8 E4M3 structured transition table back to float32 (CUDA)");
    m.def("unpack_fp8_e5m2", &causal_machine_scan_unpack_fp8_e5m2_cuda, "Unpack an FP8 E5M2 structured transition table back to float32 (CUDA)");
    m.def("preferred_load_bytes", &causal_machine_scan_preferred_load_bytes_cuda, "Describe the preferred vector load width in bytes for tiled CUDA paths");
    m.def("elements_per_load", &causal_machine_scan_elements_per_load_cuda, "Describe the preferred elements per vectorized load for tiled CUDA paths");
    m.def("can_use_vectorized_io", &causal_machine_scan_can_use_vectorized_io_cuda, "Report whether tiled CUDA paths can use vectorized float IO");
    m.def("can_use_async_memcpy", &causal_machine_scan_can_use_async_memcpy_cuda, "Report whether the device/runtime can use cuda::memcpy_async-backed paths");
    m.def("can_use_tensor_cores", &causal_machine_scan_can_use_tensor_cores_cuda, "Report whether the device supports Tensor Core execution");
    m.def("can_use_half2_path", &causal_machine_scan_can_use_half2_path_cuda, "Report whether the device supports half2 fast paths");
    m.def("can_use_wmma", &causal_machine_scan_can_use_wmma_cuda, "Report whether the device supports WMMA/Tensor Core programming");
    m.def("can_use_tma", &causal_machine_scan_can_use_tma_cuda, "Report whether the device supports Hopper Tensor Memory Accelerator features");
    m.def("can_use_wgmma", &causal_machine_scan_can_use_wgmma_cuda, "Report whether the device supports Hopper warpgroup MMA features");
    m.def("describe_tiled_forward_runtime", &causal_machine_scan_describe_tiled_forward_runtime_cuda, "Describe tiled forward launch/runtime diagnostics for the custom CUDA kernel");
    m.def("describe_masked_tiled_forward_runtime", &causal_machine_scan_describe_masked_tiled_forward_runtime_cuda, "Describe masked tiled forward launch/runtime diagnostics for the custom CUDA kernel");
    m.def("describe_tiled_backward_runtime", &causal_machine_scan_describe_tiled_backward_runtime_cuda, "Describe tiled backward launch/runtime diagnostics for the custom CUDA kernel");
    m.def("describe_masked_tiled_backward_runtime", &causal_machine_scan_describe_masked_tiled_backward_runtime_cuda, "Describe masked tiled backward launch/runtime diagnostics for the custom CUDA kernel");
    m.def("describe_runtime_config", &causal_machine_scan_describe_runtime_config, "Describe the current structured scan launch/scheduler config (CUDA)");
    m.def("describe_tiled_runtime_config", &causal_machine_scan_describe_tiled_runtime_config, "Describe the tiled structured scan launch/scheduler config (CUDA)");
    m.def("select_dense_runtime_policy", &causal_machine_scan_select_dense_runtime_policy, "Select the dense small-state structured scan kernel family (CUDA)");
    m.def("select_tiled_runtime_policy", &causal_machine_scan_select_tiled_runtime_policy, "Select a validated tiled structured scan launch policy (CUDA)");
    m.def(
        "describe_masked_tiled_runtime_config",
        [](int64_t num_states, int64_t batch_size, int64_t device_index, bool backward) {
            return causal_machine_scan_describe_masked_tiled_runtime_config(
                num_states,
                batch_size,
                device_index,
                backward);
        },
        "Describe the masked tiled structured scan launch/runtime config (CUDA)");
    m.def("describe_device_runtime_config", &causal_machine_scan_describe_device_runtime_config, "Describe device architecture/runtime capabilities for structured scan CUDA backends");
    m.def("describe_scan_workspace_config", &causal_machine_scan_describe_workspace_config, "Describe explicit workspace tensor requirements for structured scan CUDA kernels");
    m.def("decode_belief_projection", &causal_machine_decode_belief_projection, "Fuse belief exp plus output projection for state-space decode (CUDA)");
    m.def("decode_belief_projection_backward_input", &causal_machine_decode_belief_projection_backward_input, "Backward-input kernel for fused state-space belief decode (CUDA)");
    m.def("decode_belief_projection_backward_weight", &causal_machine_decode_belief_projection_backward_weight, "Backward-weight kernel for fused state-space belief decode (CUDA)");
    m.def("create_scan_workspace", &causal_machine_scan_create_workspace, "Create explicit reusable workspace tensors for structured scan CUDA kernels");
    m.def("record_paged_step_", &causal_machine_scan_record_paged_step_, "Append one recurrent belief step into the paged CUDA cache");
    m.def("record_paged_sequence_", &causal_machine_scan_record_paged_sequence_, "Append a recurrent belief sequence into the paged CUDA cache");
    m.def("read_paged_latest_", &causal_machine_scan_read_paged_latest_, "Read the latest recurrent belief and latent state from the paged CUDA cache");
    m.def("reorder_paged_cache_", &causal_machine_scan_reorder_paged_cache_, "Reorder paged cache batch rows by updating the page table and lengths on CUDA");
    m.def("paged_step_", &causal_machine_scan_paged_step_, "Read the latest paged recurrent belief, run one CUDA step, and append the result back into the paged cache");
}
