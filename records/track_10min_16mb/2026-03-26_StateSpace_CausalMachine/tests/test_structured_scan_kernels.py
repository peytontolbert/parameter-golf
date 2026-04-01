import importlib.util
import math
import os
import pathlib
import sys
import unittest
from unittest import mock

import torch


def _load_train_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    train_path = root / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("state_space_causal_machine_train_gpt_kernel_tests", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec for {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_gpt = _load_train_module()


def _dense_scan_from_probs(
    local_logits: torch.Tensor,
    transition_source_probs: torch.Tensor,
    transition_dest_probs: torch.Tensor,
    transition_context: torch.Tensor,
    initial_log_belief: torch.Tensor,
    transition_gate: torch.Tensor,
    transition_stay_probs: torch.Tensor,
    transition_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = local_logits.shape
    prev_log_belief = initial_log_belief
    stay_probs = transition_stay_probs.view(1, -1)
    dense_transition = None
    if transition_mask is not None:
        dense_transition = transition_source_probs @ transition_dest_probs
        dense_transition = dense_transition * transition_mask.to(dtype=dense_transition.dtype)
        dense_transition = dense_transition / dense_transition.sum(dim=1, keepdim=True).clamp_min(1.0e-20)
    belief_steps = []
    for pos in range(seq_len):
        prev_probs = prev_log_belief.exp()
        if dense_transition is None:
            latent = prev_probs @ transition_source_probs
            mix_probs = latent @ transition_dest_probs
        else:
            mix_probs = prev_probs @ dense_transition
        pred_probs = (stay_probs * prev_probs + (1.0 - stay_probs) * mix_probs).clamp_min(1.0e-20)
        filtered_logits = local_logits[:, pos, :] + transition_gate * (pred_probs.log() + transition_context[:, pos, :])
        next_log_belief = torch.log_softmax(filtered_logits, dim=-1)
        belief_steps.append(next_log_belief)
        prev_log_belief = next_log_belief
    beliefs = torch.stack(belief_steps, dim=1)
    return beliefs, prev_log_belief


def _clone_param(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(tensor.requires_grad)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for structured scan kernel tests")
class StructuredScanKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.device = torch.device("cuda")
        cls.ext = train_gpt.load_causal_machine_scan_cuda()

    def _assert_close(self, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float, msg: str):
        self.assertTrue(
            torch.allclose(actual.detach().float().cpu(), expected.detach().float().cpu(), atol=atol, rtol=rtol),
            msg,
        )

    def test_dense_cuda_backward_matches_autograd_reference(self):
        b, l, n, r = 2, 4, 64, 32
        local_logits = torch.randn((b, l, n), device=self.device, dtype=torch.float32, requires_grad=True)
        source_logits = torch.randn((n, r), device=self.device, dtype=torch.float32, requires_grad=True)
        dest_logits = torch.randn((r, n), device=self.device, dtype=torch.float32, requires_grad=True)
        transition_context = torch.randn((b, l, n), device=self.device, dtype=torch.float32, requires_grad=True)
        initial_log_belief = torch.log_softmax(
            torch.randn((b, n), device=self.device, dtype=torch.float32),
            dim=-1,
        ).requires_grad_()
        transition_gate = torch.tensor(0.35, device=self.device, dtype=torch.float32, requires_grad=True)
        transition_stay_probs = torch.sigmoid(
            torch.randn((n,), device=self.device, dtype=torch.float32)
        ).requires_grad_()

        beliefs, final_belief = train_gpt.causal_machine_scan_cuda(
            local_logits,
            source_logits,
            dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            chunk_size=64,
            runtime_config=None,
        )
        loss = beliefs.sum() + final_belief.sum()
        loss.backward()
        test_grads = [t.grad.detach().clone() for t in (
            local_logits,
            source_logits,
            dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
        )]

        ref_local = _clone_param(local_logits)
        ref_source = _clone_param(source_logits)
        ref_dest = _clone_param(dest_logits)
        ref_context = _clone_param(transition_context)
        ref_initial = _clone_param(initial_log_belief)
        ref_gate = _clone_param(transition_gate)
        ref_stay = _clone_param(transition_stay_probs)
        ref_beliefs, ref_final = _dense_scan_from_probs(
            ref_local,
            torch.softmax(ref_source, dim=-1),
            torch.softmax(ref_dest, dim=-1),
            ref_context,
            ref_initial,
            ref_gate,
            ref_stay,
        )
        ref_loss = ref_beliefs.sum() + ref_final.sum()
        ref_loss.backward()
        ref_grads = [t.grad.detach().clone() for t in (
            ref_local,
            ref_source,
            ref_dest,
            ref_context,
            ref_initial,
            ref_gate,
            ref_stay,
        )]

        self._assert_close(beliefs, ref_beliefs, atol=2.0e-4, rtol=2.0e-4, msg="dense beliefs mismatch")
        self._assert_close(final_belief, ref_final, atol=2.0e-4, rtol=2.0e-4, msg="dense final belief mismatch")
        for idx, (actual, expected) in enumerate(zip(test_grads, ref_grads)):
            self._assert_close(actual, expected, atol=3.0e-4, rtol=3.0e-4, msg=f"dense grad {idx} mismatch")

    def test_tiled_cuda_backward_matches_dense_reference(self):
        b, l, n, r = 2, 4, 160, 24
        runtime_config = train_gpt.StructuredScanRuntimeConfig()
        kernel_config = train_gpt.autotune_structured_scan_kernel_config(
            num_states=n,
            transition_rank=r,
            seq_len=l,
            device=self.device,
            default_chunk_size=64,
            needs_grad=True,
            runtime_config=runtime_config,
        )
        local_logits = torch.randn((b, l, n), device=self.device, dtype=torch.float32, requires_grad=True)
        source_probs = torch.softmax(torch.randn((n, r), device=self.device, dtype=torch.float32), dim=-1).requires_grad_()
        dest_probs = torch.softmax(torch.randn((r, n), device=self.device, dtype=torch.float32), dim=-1).requires_grad_()
        transition_context = torch.randn((b, l, n), device=self.device, dtype=torch.float32, requires_grad=True)
        initial_log_belief = torch.log_softmax(
            torch.randn((b, n), device=self.device, dtype=torch.float32),
            dim=-1,
        ).requires_grad_()
        transition_gate = torch.tensor(0.2, device=self.device, dtype=torch.float32, requires_grad=True)
        transition_stay_probs = torch.sigmoid(
            torch.randn((n,), device=self.device, dtype=torch.float32)
        ).requires_grad_()

        beliefs, final_belief = train_gpt.causal_machine_scan_tiled_cuda(
            local_logits,
            source_probs,
            dest_probs,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            runtime_config=runtime_config,
            chunk_size=kernel_config.chunk_size,
            tile_size=kernel_config.tile_size,
            split_size=kernel_config.split_size,
        )
        loss = beliefs.sum() + final_belief.sum()
        loss.backward()
        test_grads = [t.grad.detach().clone() for t in (
            local_logits,
            source_probs,
            dest_probs,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
        )]

        ref_local = _clone_param(local_logits)
        ref_source = _clone_param(source_probs)
        ref_dest = _clone_param(dest_probs)
        ref_context = _clone_param(transition_context)
        ref_initial = _clone_param(initial_log_belief)
        ref_gate = _clone_param(transition_gate)
        ref_stay = _clone_param(transition_stay_probs)
        ref_beliefs, ref_final = _dense_scan_from_probs(
            ref_local,
            ref_source,
            ref_dest,
            ref_context,
            ref_initial,
            ref_gate,
            ref_stay,
        )
        ref_loss = ref_beliefs.sum() + ref_final.sum()
        ref_loss.backward()
        ref_grads = [t.grad.detach().clone() for t in (
            ref_local,
            ref_source,
            ref_dest,
            ref_context,
            ref_initial,
            ref_gate,
            ref_stay,
        )]

        self._assert_close(beliefs, ref_beliefs, atol=4.0e-4, rtol=4.0e-4, msg="tiled beliefs mismatch")
        self._assert_close(final_belief, ref_final, atol=4.0e-4, rtol=4.0e-4, msg="tiled final belief mismatch")
        for idx, (actual, expected) in enumerate(zip(test_grads, ref_grads)):
            self._assert_close(actual, expected, atol=6.0e-4, rtol=6.0e-4, msg=f"tiled grad {idx} mismatch")

    def test_composable_cuda_chunked_forward_matches_dense_reference(self):
        b, l, n, r = 2, 7, 64, 16
        chunk_size = 3
        local_logits = torch.randn((b, l, n), device=self.device, dtype=torch.float32)
        source_logits = torch.randn((n, r), device=self.device, dtype=torch.float32)
        dest_logits = torch.randn((r, n), device=self.device, dtype=torch.float32)
        transition_context = torch.randn((b, l, n), device=self.device, dtype=torch.float32)
        initial_log_belief = torch.log_softmax(
            torch.randn((b, n), device=self.device, dtype=torch.float32),
            dim=-1,
        )
        transition_stay_probs = torch.sigmoid(
            torch.randn((n,), device=self.device, dtype=torch.float32)
        )

        beliefs, final_belief = self.ext.forward_composable_logits(
            local_logits,
            source_logits,
            dest_logits,
            transition_context,
            initial_log_belief,
            transition_stay_probs,
            chunk_size,
        )

        ref_beliefs, ref_final = _dense_scan_from_probs(
            local_logits,
            torch.softmax(source_logits, dim=-1),
            torch.softmax(dest_logits, dim=-1),
            transition_context,
            initial_log_belief,
            torch.tensor(1.0, device=self.device, dtype=torch.float32),
            transition_stay_probs,
        )

        self._assert_close(beliefs, ref_beliefs, atol=3.0e-4, rtol=3.0e-4, msg="composable chunked beliefs mismatch")
        self._assert_close(final_belief, ref_final, atol=3.0e-4, rtol=3.0e-4, msg="composable chunked final belief mismatch")

    def test_masked_cuda_matches_dense_masked_reference(self):
        b, l, n, r = 2, 4, 160, 24
        mask = torch.rand((n, n), device=self.device) > 0.55
        runtime_config = train_gpt.StructuredScanRuntimeConfig(transition_mask=mask)
        local_logits = torch.randn((b, l, n), device=self.device, dtype=torch.float32, requires_grad=True)
        source_logits = torch.randn((n, r), device=self.device, dtype=torch.float32, requires_grad=True)
        dest_logits = torch.randn((r, n), device=self.device, dtype=torch.float32, requires_grad=True)
        transition_context = torch.randn((b, l, n), device=self.device, dtype=torch.float32, requires_grad=True)
        initial_log_belief = torch.log_softmax(
            torch.randn((b, n), device=self.device, dtype=torch.float32),
            dim=-1,
        ).requires_grad_()
        transition_gate = torch.tensor(0.3, device=self.device, dtype=torch.float32, requires_grad=True)
        transition_stay_probs = torch.sigmoid(
            torch.randn((n,), device=self.device, dtype=torch.float32)
        ).requires_grad_()

        beliefs, final_belief = train_gpt.causal_machine_scan_masked_cuda(
            local_logits,
            source_logits,
            dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            runtime_config=runtime_config,
            chunk_size=64,
        )
        loss = beliefs.sum() + final_belief.sum()
        loss.backward()
        test_grads = [t.grad.detach().clone() for t in (
            local_logits,
            source_logits,
            dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
        )]

        ref_local = _clone_param(local_logits)
        ref_source = _clone_param(source_logits)
        ref_dest = _clone_param(dest_logits)
        ref_context = _clone_param(transition_context)
        ref_initial = _clone_param(initial_log_belief)
        ref_gate = _clone_param(transition_gate)
        ref_stay = _clone_param(transition_stay_probs)
        ref_beliefs, ref_final = _dense_scan_from_probs(
            ref_local,
            torch.softmax(ref_source, dim=-1),
            torch.softmax(ref_dest, dim=-1),
            ref_context,
            ref_initial,
            ref_gate,
            ref_stay,
            transition_mask=mask,
        )
        ref_loss = ref_beliefs.sum() + ref_final.sum()
        ref_loss.backward()
        ref_grads = [t.grad.detach().clone() for t in (
            ref_local,
            ref_source,
            ref_dest,
            ref_context,
            ref_initial,
            ref_gate,
            ref_stay,
        )]

        self._assert_close(beliefs, ref_beliefs, atol=1.0e-3, rtol=1.0e-3, msg="masked beliefs mismatch")
        self._assert_close(final_belief, ref_final, atol=1.0e-3, rtol=1.0e-3, msg="masked final belief mismatch")
        for idx, (actual, expected) in enumerate(zip(test_grads, ref_grads)):
            self._assert_close(actual, expected, atol=2.0e-3, rtol=2.0e-3, msg=f"masked grad {idx} mismatch")

    def test_packed_cuda_matches_dense_reference(self):
        b, l, n, r = 2, 4, 128, 32
        local_logits = torch.randn((b, l, n), device=self.device, dtype=torch.float32, requires_grad=True)
        source_logits = torch.randn((n, r), device=self.device, dtype=torch.float32, requires_grad=True)
        dest_logits = torch.randn((r, n), device=self.device, dtype=torch.float32, requires_grad=True)
        transition_context = torch.randn((b, l, n), device=self.device, dtype=torch.float32, requires_grad=True)
        initial_log_belief = torch.log_softmax(
            torch.randn((b, n), device=self.device, dtype=torch.float32),
            dim=-1,
        ).requires_grad_()
        transition_gate = torch.tensor(0.25, device=self.device, dtype=torch.float32, requires_grad=True)
        transition_stay_probs = torch.sigmoid(
            torch.randn((n,), device=self.device, dtype=torch.float32)
        ).requires_grad_()

        for packed_dtype, atol, rtol in (("int8", 2.5e-2, 2.5e-2), ("fp8_e4m3", 4.0e-2, 4.0e-2)):
            cache: dict[str, object] = {}
            train_gpt._cached_env_str.cache_clear()
            with mock.patch.dict(os.environ, {"CAUSAL_MACHINE_SCAN_PACKED_DTYPE": packed_dtype}, clear=False):
                packed = train_gpt.get_or_update_scan_transition_prepack(
                    cache,
                    source_logits,
                    dest_logits,
                    self.device,
                )
                self.assertIsNotNone(packed, f"failed to build packed tables for {packed_dtype}")
                beliefs, final_belief = train_gpt.causal_machine_scan_cuda(
                    local_logits,
                    source_logits,
                    dest_logits,
                    transition_context,
                    initial_log_belief,
                    transition_gate,
                    transition_stay_probs,
                    packed_transition_tables=packed,
                    chunk_size=64,
                    runtime_config=None,
                )
                ref_beliefs, ref_final = _dense_scan_from_probs(
                    local_logits,
                    torch.softmax(source_logits, dim=-1),
                    torch.softmax(dest_logits, dim=-1),
                    transition_context,
                    initial_log_belief,
                    transition_gate,
                    transition_stay_probs,
                )
                self._assert_close(beliefs, ref_beliefs, atol=atol, rtol=rtol, msg=f"{packed_dtype} beliefs mismatch")
                self._assert_close(final_belief, ref_final, atol=atol, rtol=rtol, msg=f"{packed_dtype} final belief mismatch")


if __name__ == "__main__":
    unittest.main()
