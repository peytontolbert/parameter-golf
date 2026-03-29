import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


def _load_train_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    train_path = root / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("state_space_causal_machine_train_gpt", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec for {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_gpt = _load_train_module()


class StructuredScanConfigTests(unittest.TestCase):
    def test_scan_extension_build_dirs_prefer_dedicated_dir_with_legacy_fallback(self):
        source_dir = pathlib.Path(train_gpt.__file__).resolve().parent / "cuda_ext"
        build_dirs = train_gpt._extension_candidate_build_dirs_multi(
            source_dir,
            ("build/causal_machine_scan_cuda", "build"),
        )
        self.assertGreaterEqual(len(build_dirs), 2)
        self.assertEqual(build_dirs[0], (source_dir / "build/causal_machine_scan_cuda").resolve())
        self.assertIn((source_dir / "build").resolve(), build_dirs)

    def test_latent_extension_build_dirs_prefer_dedicated_dir_with_legacy_fallback(self):
        source_dir = pathlib.Path(train_gpt.__file__).resolve().parent / "cuda_ext"
        build_dirs = train_gpt._extension_candidate_build_dirs_multi(
            source_dir,
            ("build/causal_machine_latent_scan_cuda", "build"),
        )
        self.assertGreaterEqual(len(build_dirs), 2)
        self.assertEqual(build_dirs[0], (source_dir / "build/causal_machine_latent_scan_cuda").resolve())
        self.assertIn((source_dir / "build").resolve(), build_dirs)

    def test_paged_history_uses_int64_lengths_for_cuda_extension_compat(self):
        cache = train_gpt.CausalMachineCache()
        cache.enable_paged_history(
            batch_size=2,
            num_states=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            latent_rank=4,
            page_size=16,
            max_pages=2,
        )
        self.assertIsNotNone(cache.paged_lengths)
        self.assertEqual(cache.paged_lengths.dtype, torch.int64)
        self.assertEqual(cache.last_paged_write_backend, "ready")

    def test_structured_scan_kernel_info_reports_packed_kind_and_paged_backend(self):
        kernel_config = train_gpt.StructuredScanKernelConfig(
            num_states=128,
            transition_rank=32,
            chunk_size=64,
            tile_size=128,
            split_size=32,
            backend="cuda",
            allow_cuda=True,
            allow_tiled_cuda=False,
            allow_quantized_tables=True,
        )
        cache = train_gpt.CausalMachineCache(last_paged_write_backend="cuda")
        packed_tables = (
            train_gpt._PACKED_TRANSITION_INT8,
            torch.empty((0,), dtype=torch.int8),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int8),
            torch.empty((0,), dtype=torch.float32),
        )
        info = train_gpt._structured_scan_kernel_info(
            path="cuda_dense",
            kernel_config=kernel_config,
            runtime_config=train_gpt.StructuredScanRuntimeConfig(use_paged_cache=True),
            packed_transition_tables=packed_tables,
            cache=cache,
        )
        self.assertEqual(info["path"], "cuda_dense")
        self.assertEqual(info["packed_transition_kind"], "int8")
        self.assertEqual(info["paged_cache_write_backend"], "cuda")
        self.assertTrue(info["uses_paged_cache"])

    def test_structured_scan_kernel_info_reports_arch_and_low_precision_contract(self):
        kernel_config = train_gpt.StructuredScanKernelConfig(
            num_states=256,
            transition_rank=64,
            chunk_size=128,
            backend="cuda_tiled",
            policy_name="sm100_tiled",
            arch_family="sm100+",
            kernel_family="sm100_tiled_async_proto",
            supports_async_pipeline=True,
            supports_tensor_memory_accel=True,
            supports_cluster_launch_control=True,
            supports_tma=True,
            supports_wgmma=True,
            supports_tcgen05=True,
            sparse_reorder_mode="grouped_src_blocks",
            benchmark_family="tiled_large_state",
        )
        low_precision_metadata = train_gpt.StructuredScanLowPrecisionMetadata(
            packed_kind=train_gpt._PACKED_TRANSITION_FP8_E4M3,
            source_amax=1.25,
            dest_amax=0.75,
            source_scale=0.01,
            dest_scale=0.02,
            source_scale_inv=100.0,
            dest_scale_inv=50.0,
            step=7,
        )
        cache = train_gpt.CausalMachineCache()
        cache._packed_transition_cache = {"low_precision_metadata": low_precision_metadata}
        info = train_gpt._structured_scan_kernel_info(
            path="cuda_tiled",
            kernel_config=kernel_config,
            runtime_config=train_gpt.StructuredScanRuntimeConfig(),
            cache=cache,
        )
        self.assertEqual(info["arch_family"], "sm100+")
        self.assertEqual(info["kernel_family"], "sm100_tiled_async_proto")
        self.assertEqual(info["sparse_reorder_mode"], "grouped_src_blocks")
        self.assertTrue(info["supports_wgmma"])
        self.assertEqual(info["low_precision_source_scale_inv"], 100.0)

    def test_supports_generic_rank_up_to_128(self):
        self.assertTrue(train_gpt.supports_structured_scan_cuda_config(64, 8))
        self.assertTrue(train_gpt.supports_structured_scan_cuda_config(96, 8))
        self.assertTrue(train_gpt.supports_structured_scan_cuda_config(128, 1))
        self.assertTrue(train_gpt.supports_structured_scan_cuda_config(128, 7))
        self.assertTrue(train_gpt.supports_structured_scan_cuda_config(128, 96))
        self.assertTrue(train_gpt.supports_structured_scan_cuda_config(128, 127))
        self.assertTrue(train_gpt.supports_structured_scan_cuda_config(128, 128))

    def test_rejects_invalid_state_count_and_rank(self):
        self.assertFalse(train_gpt.supports_structured_scan_cuda_config(129, 8))
        self.assertFalse(train_gpt.supports_structured_scan_cuda_config(128, 0))
        self.assertFalse(train_gpt.supports_structured_scan_cuda_config(128, 129))

    def test_cuda_capability_check_accepts_generic_rank_when_smem_fits(self):
        props = SimpleNamespace(
            shared_memory_per_block_optin=train_gpt._causal_machine_scan_shared_bytes(96),
            shared_memory_per_block=0,
        )
        with mock.patch.object(train_gpt.torch.cuda, "get_device_properties", return_value=props):
            self.assertTrue(train_gpt._can_use_causal_machine_scan_cuda(torch.device("cuda"), 96))

    def test_cuda_capability_check_rejects_rank_above_kernel_limit(self):
        props = SimpleNamespace(shared_memory_per_block_optin=1 << 20, shared_memory_per_block=0)
        with mock.patch.object(train_gpt.torch.cuda, "get_device_properties", return_value=props):
            self.assertFalse(train_gpt._can_use_causal_machine_scan_cuda(torch.device("cuda"), 129))

    def test_masked_cuda_capability_requires_specialized_backward_when_grad_needed(self):
        ext = SimpleNamespace(forward_masked_logits=mock.Mock())
        with mock.patch.object(train_gpt, "load_causal_machine_scan_cuda", return_value=ext):
            self.assertFalse(
                train_gpt._can_use_causal_machine_masked_scan_cuda(
                    torch.device("cuda"),
                    num_states=64,
                    transition_rank=8,
                    needs_grad=True,
                )
            )

    def test_masked_cuda_capability_uses_masked_kernel_support_for_generic_states(self):
        ext = SimpleNamespace(forward_masked_logits=mock.Mock(), backward_masked_logits=mock.Mock())
        props = SimpleNamespace(shared_memory_per_block_optin=1 << 20, shared_memory_per_block=0)
        with (
            mock.patch.object(train_gpt, "load_causal_machine_scan_cuda", return_value=ext),
            mock.patch.object(train_gpt.torch.cuda, "get_device_properties", return_value=props),
        ):
            self.assertTrue(
                train_gpt._can_use_causal_machine_masked_scan_cuda(
                    torch.device("cuda"),
                    num_states=160,
                    transition_rank=24,
                    needs_grad=True,
                )
            )

    def test_tiled_cuda_capability_uses_extension_fallback_when_custom_kernel_is_unavailable(self):
        ext = SimpleNamespace(forward_tiled_logits=mock.Mock(), backward_tiled_probs=mock.Mock())
        with mock.patch.object(train_gpt, "load_causal_machine_scan_cuda", return_value=ext):
            self.assertTrue(
                train_gpt._can_use_causal_machine_tiled_scan_cuda(
                    torch.device("cuda"),
                    num_states=160,
                    transition_rank=24,
                    needs_grad=True,
                )
            )

    def test_sparse_cuda_capability_accepts_large_state_count_when_smem_fits(self):
        props = SimpleNamespace(
            shared_memory_per_block_optin=train_gpt._causal_machine_sparse_scan_shared_bytes(4096),
            shared_memory_per_block=0,
        )
        with mock.patch.object(train_gpt.torch.cuda, "get_device_properties", return_value=props):
            self.assertTrue(
                train_gpt._can_use_causal_machine_sparse_scan_cuda(
                    torch.device("cuda"),
                    num_states=4096,
                    transition_rank=32,
                )
            )

    def test_arch_spec_reports_sm100_features(self):
        with mock.patch.object(train_gpt.torch.cuda, "get_device_capability", return_value=(10, 0)):
            spec = train_gpt._describe_structured_scan_arch_spec(torch.device("cuda"))
        self.assertEqual(spec.arch_family, "sm100+")
        self.assertTrue(spec.supports_tcgen05)
        self.assertFalse(spec.supports_tma)
        self.assertFalse(spec.supports_wgmma)

    def test_describe_sparse_transition_tables_reports_grouping(self):
        tables = train_gpt.StructuredSparseTransitionTables(
            blocks=torch.empty((3, 4, 4)),
            row_ptr=torch.tensor([0, 1, 3], dtype=torch.int32),
            col_idx=torch.tensor([0, 0, 1], dtype=torch.int32),
            dst_idx=torch.tensor([0, 1, 1], dtype=torch.int32),
            src_row_ptr=torch.tensor([0, 2, 3], dtype=torch.int32),
            src_nz_idx=torch.tensor([0, 1, 2], dtype=torch.int32),
            row_sums=torch.ones((8,), dtype=torch.float32),
            block_mask=torch.ones((3, 4, 4), dtype=torch.float32),
            block_size=4,
            density=0.375,
            grouped_src_row_ptr=torch.tensor([0, 2, 3], dtype=torch.int32),
            grouped_src_block_idx=torch.tensor([0, 1], dtype=torch.int32),
            grouped_src_group_ids=torch.tensor([0, 0, 1], dtype=torch.int32),
            grouped_src_group_count=2,
        )
        info = train_gpt.describe_structured_sparse_transition_tables(tables)
        self.assertEqual(info["nnz_blocks"], 3)
        self.assertEqual(info["grouped_src_group_count"], 2)
        self.assertTrue(info["has_grouped_backward_metadata"])

    def test_autotune_can_select_cuda_for_generic_rank(self):
        runtime_config = train_gpt.StructuredScanRuntimeConfig()
        with (
            mock.patch.object(train_gpt.torch.cuda, "get_device_capability", return_value=(8, 0)),
            mock.patch.object(train_gpt, "_can_use_causal_machine_scan_cuda", return_value=True),
        ):
            config = train_gpt.autotune_structured_scan_kernel_config(
                num_states=128,
                transition_rank=96,
                seq_len=2048,
                device=torch.device("cuda"),
                default_chunk_size=64,
                runtime_config=runtime_config,
            )
        self.assertEqual(config.backend, "cuda")
        self.assertTrue(config.allow_cuda)
        self.assertEqual(config.transition_rank, 96)

    def test_autotune_keeps_cuda_tiled_when_only_extension_fallback_is_available(self):
        runtime_config = train_gpt.StructuredScanRuntimeConfig(allow_cuda=True)
        with (
            mock.patch.object(train_gpt, "_can_use_causal_machine_tiled_scan_cuda", return_value=True),
            mock.patch.object(train_gpt, "_resolve_structured_scan_tiled_kernel_config", return_value=None),
        ):
            config = train_gpt.autotune_structured_scan_kernel_config(
                num_states=160,
                transition_rank=24,
                seq_len=2048,
                device=torch.device("cuda"),
                default_chunk_size=64,
                needs_grad=True,
                runtime_config=runtime_config,
            )
        self.assertEqual(config.backend, "cuda_tiled")
        self.assertTrue(config.allow_tiled_cuda)

    def test_masked_runtime_features_still_disable_cuda_backend(self):
        runtime_config = train_gpt.StructuredScanRuntimeConfig(seq_lens=[128])
        with (
            mock.patch.object(train_gpt.torch.cuda, "get_device_capability", return_value=(9, 0)),
            mock.patch.object(train_gpt, "_can_use_causal_machine_scan_cuda", return_value=True),
        ):
            config = train_gpt.autotune_structured_scan_kernel_config(
                num_states=128,
                transition_rank=16,
                seq_len=2048,
                device=torch.device("cuda"),
                default_chunk_size=64,
                runtime_config=runtime_config,
            )
        self.assertEqual(config.backend, "python")
        self.assertFalse(config.allow_cuda)

    def test_describe_runtime_config_uses_extension_metadata(self):
        ext = SimpleNamespace(
            describe_runtime_config=mock.Mock(
                return_value={
                    "batch_size": 11,
                    "launch_chunk_size": 256,
                    "num_launches": 1,
                    "num_sequence_tiles": 4,
                    "worker_blocks": 6,
                    "grid_x": 6,
                    "use_single_launch": True,
                    "persistent_device_loop": True,
                    "device_work_queue": True,
                    "shared_bytes": 40960,
                    "direct_grad_reduce": True,
                }
            )
        )
        with (
            mock.patch.object(train_gpt, "load_causal_machine_scan_cuda", return_value=ext),
            mock.patch.object(train_gpt.torch.cuda, "current_device", return_value=3),
        ):
            config = train_gpt.describe_structured_scan_cuda_runtime_config(
                batch_size=11,
                seq_len=1024,
                transition_rank=32,
                chunk_size=256,
                device=torch.device("cuda"),
                backward=True,
            )
        self.assertEqual(config["batch_size"], 11)
        self.assertEqual(config["launch_chunk_size"], 256)
        self.assertEqual(config["num_launches"], 1)
        self.assertEqual(config["num_sequence_tiles"], 4)
        self.assertEqual(config["worker_blocks"], 6)
        self.assertEqual(config["grid_x"], 6)
        self.assertTrue(config["persistent_device_loop"])
        self.assertTrue(config["device_work_queue"])
        self.assertTrue(config["direct_grad_reduce"])
        ext.describe_runtime_config.assert_called_once_with(11, 1024, 128, 32, 256, 3, True)

    def test_describe_runtime_config_requires_cuda_device(self):
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            train_gpt.describe_structured_scan_cuda_runtime_config(
                seq_len=32,
                transition_rank=8,
                chunk_size=32,
                device=torch.device("cpu"),
            )

    def test_masked_cuda_specialized_path_uses_explicit_backward_hook(self):
        num_states = 64
        transition_rank = 8
        batch_size = 2
        seq_len = 3
        ext = SimpleNamespace()

        def _forward_masked_logits(
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
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk,
        ):
            return local_logits.clone(), initial_log_belief.clone()

        ext.forward_masked_logits = mock.Mock(side_effect=_forward_masked_logits)
        ext.backward_masked_logits = mock.Mock(
            return_value=(
                torch.ones((batch_size, seq_len, num_states), dtype=torch.float32),
                torch.ones((num_states, transition_rank), dtype=torch.float32),
                torch.ones((transition_rank, num_states), dtype=torch.float32),
                torch.ones((batch_size, seq_len, num_states), dtype=torch.float32),
                torch.ones((batch_size, num_states), dtype=torch.float32),
                torch.ones((1,), dtype=torch.float32),
                torch.ones((num_states,), dtype=torch.float32),
            )
        )

        runtime_config = train_gpt.StructuredScanRuntimeConfig(
            allow_cuda=True,
            transition_mask=torch.ones((num_states, num_states), dtype=torch.bool),
        )
        local_logits = torch.zeros((batch_size, seq_len, num_states), dtype=torch.float32, requires_grad=True)
        transition_source_logits = torch.zeros((num_states, transition_rank), dtype=torch.float32, requires_grad=True)
        transition_dest_logits = torch.zeros((transition_rank, num_states), dtype=torch.float32, requires_grad=True)
        transition_context = torch.zeros((batch_size, seq_len, num_states), dtype=torch.float32, requires_grad=True)
        initial_log_belief = torch.zeros((batch_size, num_states), dtype=torch.float32, requires_grad=True)
        transition_gate = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
        transition_stay_probs = torch.zeros((num_states,), dtype=torch.float32, requires_grad=True)

        with mock.patch.object(train_gpt, "load_causal_machine_scan_cuda", return_value=ext):
            beliefs, _final_belief = train_gpt.causal_machine_scan_masked_cuda(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                runtime_config=runtime_config,
                chunk_size=32,
            )
            beliefs.sum().backward()

        ext.forward_masked_logits.assert_called_once()
        ext.backward_masked_logits.assert_called_once()

    def test_masked_cuda_generic_path_uses_masked_kernel(self):
        num_states = 160
        transition_rank = 24
        batch_size = 2
        seq_len = 3
        ext = SimpleNamespace()

        def _forward_masked_logits(
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
            score_clamp_min,
            score_clamp_max,
            score_threshold,
            score_topk,
        ):
            return local_logits.clone(), initial_log_belief.clone()

        ext.forward_masked_logits = mock.Mock(side_effect=_forward_masked_logits)
        ext.backward_masked_logits = mock.Mock(
            return_value=(
                torch.ones((batch_size, seq_len, num_states), dtype=torch.float32),
                torch.ones((num_states, transition_rank), dtype=torch.float32),
                torch.ones((transition_rank, num_states), dtype=torch.float32),
                torch.ones((batch_size, seq_len, num_states), dtype=torch.float32),
                torch.ones((batch_size, num_states), dtype=torch.float32),
                torch.ones((1,), dtype=torch.float32),
                torch.ones((num_states,), dtype=torch.float32),
            )
        )

        runtime_config = train_gpt.StructuredScanRuntimeConfig(
            allow_cuda=True,
            transition_mask=torch.ones((num_states, num_states), dtype=torch.bool),
        )
        local_logits = torch.zeros((batch_size, seq_len, num_states), dtype=torch.float32, requires_grad=True)
        transition_source_logits = torch.zeros((num_states, transition_rank), dtype=torch.float32, requires_grad=True)
        transition_dest_logits = torch.zeros((transition_rank, num_states), dtype=torch.float32, requires_grad=True)
        transition_context = torch.zeros((batch_size, seq_len, num_states), dtype=torch.float32, requires_grad=True)
        initial_log_belief = torch.zeros((batch_size, num_states), dtype=torch.float32, requires_grad=True)
        transition_gate = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
        transition_stay_probs = torch.zeros((num_states,), dtype=torch.float32, requires_grad=True)

        with mock.patch.object(train_gpt, "load_causal_machine_scan_cuda", return_value=ext):
            beliefs, _final_belief = train_gpt.causal_machine_scan_masked_cuda(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                runtime_config=runtime_config,
                chunk_size=32,
            )
            beliefs.sum().backward()

        ext.forward_masked_logits.assert_called_once()
        ext.backward_masked_logits.assert_called_once()
        self.assertIsNotNone(transition_source_logits.grad)
        self.assertIsNotNone(transition_dest_logits.grad)

    def test_masked_cuda_generic_path_requires_masked_entrypoints(self):
        num_states = 160
        transition_rank = 24
        runtime_config = train_gpt.StructuredScanRuntimeConfig(
            allow_cuda=True,
            transition_mask=torch.ones((num_states, num_states), dtype=torch.bool),
        )
        local_logits = torch.zeros((1, 2, num_states), dtype=torch.float32)
        transition_source_logits = torch.zeros((num_states, transition_rank), dtype=torch.float32)
        transition_dest_logits = torch.zeros((transition_rank, num_states), dtype=torch.float32)
        transition_context = torch.zeros((1, 2, num_states), dtype=torch.float32)
        initial_log_belief = torch.zeros((1, num_states), dtype=torch.float32)
        transition_gate = torch.zeros((1,), dtype=torch.float32)
        transition_stay_probs = torch.zeros((num_states,), dtype=torch.float32)

        with (
            mock.patch.object(train_gpt, "load_causal_machine_scan_cuda", return_value=SimpleNamespace()),
            self.assertRaisesRegex(RuntimeError, "requires forward_masked_logits and backward_masked_logits"),
        ):
            train_gpt.causal_machine_scan_masked_cuda(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                runtime_config=runtime_config,
                chunk_size=32,
            )

    def test_sparse_runtime_reroutes_to_masked_cuda_when_sparse_kernel_is_unavailable(self):
        runtime_config = train_gpt.StructuredScanRuntimeConfig(
            allow_cuda=True,
            transition_mask=torch.ones((1600, 1600), dtype=torch.bool),
        )
        sparse_tables = train_gpt.StructuredSparseTransitionTables(
            blocks=torch.empty((0, 32, 32), dtype=torch.float32),
            row_ptr=torch.zeros((1,), dtype=torch.int32),
            col_idx=torch.zeros((0,), dtype=torch.int32),
            dst_idx=torch.zeros((0,), dtype=torch.int32),
            src_row_ptr=torch.zeros((1,), dtype=torch.int32),
            src_nz_idx=torch.zeros((0,), dtype=torch.int32),
            row_sums=torch.zeros((1600,), dtype=torch.float32),
            block_mask=torch.empty((0, 32, 32), dtype=torch.float32),
            block_size=32,
            density=0.25,
        )
        local_logits = torch.zeros((1, 2, 1600), dtype=torch.float32)
        transition_source_logits = torch.zeros((1600, 32), dtype=torch.float32)
        transition_dest_logits = torch.zeros((32, 1600), dtype=torch.float32)
        transition_context = torch.zeros((1, 2, 1600), dtype=torch.float32)
        initial_log_belief = torch.zeros((1, 1600), dtype=torch.float32)
        transition_gate = torch.zeros((1,), dtype=torch.float32)
        transition_stay_probs = torch.zeros((1600,), dtype=torch.float32)

        with (
            mock.patch.object(train_gpt, "_can_use_causal_machine_sparse_scan_cuda", return_value=False),
            mock.patch.object(train_gpt, "_can_use_causal_machine_masked_scan_cuda", return_value=True),
            mock.patch.object(
                train_gpt,
                "causal_machine_scan_masked_cuda",
                return_value=(local_logits.clone(), initial_log_belief.clone()),
            ) as masked_cuda,
        ):
            result = train_gpt._execute_structured_sparse_runtime_cuda(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                sparse_tables,
                runtime_config=runtime_config,
                chunk_size=64,
                needs_grad=False,
            )

        self.assertIsNotNone(result)
        masked_cuda.assert_called_once()

    def test_tiled_cuda_path_falls_back_to_extension_forward_and_backward(self):
        num_states = 160
        transition_rank = 24
        batch_size = 2
        seq_len = 3
        ext = SimpleNamespace()

        def _forward_tiled_logits(
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
            score_clamp_min,
            score_clamp_max,
        ):
            return local_logits.clone(), initial_log_belief.clone()

        ext.forward_tiled_logits = mock.Mock(side_effect=_forward_tiled_logits)
        ext.backward_tiled_probs = mock.Mock(
            return_value=(
                torch.ones((batch_size, seq_len, num_states), dtype=torch.float32),
                torch.ones((num_states, transition_rank), dtype=torch.float32),
                torch.ones((transition_rank, num_states), dtype=torch.float32),
                torch.ones((batch_size, seq_len, num_states), dtype=torch.float32),
                torch.ones((batch_size, num_states), dtype=torch.float32),
                torch.ones((1,), dtype=torch.float32),
                torch.ones((num_states,), dtype=torch.float32),
            )
        )

        runtime_config = train_gpt.StructuredScanRuntimeConfig(allow_cuda=True)
        local_logits = torch.zeros((batch_size, seq_len, num_states), dtype=torch.float32, requires_grad=True)
        transition_source_probs = torch.zeros((num_states, transition_rank), dtype=torch.float32, requires_grad=True)
        transition_dest_probs = torch.zeros((transition_rank, num_states), dtype=torch.float32, requires_grad=True)
        transition_context = torch.zeros((batch_size, seq_len, num_states), dtype=torch.float32, requires_grad=True)
        initial_log_belief = torch.zeros((batch_size, num_states), dtype=torch.float32, requires_grad=True)
        transition_gate = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
        transition_stay_probs = torch.zeros((num_states,), dtype=torch.float32, requires_grad=True)

        with mock.patch.object(train_gpt, "load_causal_machine_scan_cuda", return_value=ext):
            beliefs, _final_belief = train_gpt.causal_machine_scan_tiled_cuda(
                local_logits,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                runtime_config=runtime_config,
                chunk_size=32,
                tile_size=128,
                split_size=64,
            )
            beliefs.sum().backward()

        ext.forward_tiled_logits.assert_called_once()
        ext.backward_tiled_probs.assert_called_once()
        self.assertIsNotNone(transition_source_probs.grad)
        self.assertIsNotNone(transition_dest_probs.grad)

    def test_tiled_cuda_routes_masked_runtime_into_masked_probs_backend(self):
        num_states = 160
        transition_rank = 24
        batch_size = 2
        seq_len = 3
        runtime_config = train_gpt.StructuredScanRuntimeConfig(
            allow_cuda=True,
            backend="cuda_tiled",
            local_transition_window=32,
        )
        local_logits = torch.zeros((batch_size, seq_len, num_states), dtype=torch.float32)
        transition_source_probs = torch.zeros((num_states, transition_rank), dtype=torch.float32)
        transition_dest_probs = torch.zeros((transition_rank, num_states), dtype=torch.float32)
        transition_context = torch.zeros((batch_size, seq_len, num_states), dtype=torch.float32)
        initial_log_belief = torch.zeros((batch_size, num_states), dtype=torch.float32)
        transition_gate = torch.zeros((1,), dtype=torch.float32)
        transition_stay_probs = torch.zeros((num_states,), dtype=torch.float32)

        with mock.patch.object(
            train_gpt,
            "causal_machine_scan_masked_probs_cuda",
            return_value=(local_logits.clone(), initial_log_belief.clone()),
        ) as masked_probs_cuda:
            beliefs, final_belief = train_gpt.causal_machine_scan_tiled_cuda(
                local_logits,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                runtime_config=runtime_config,
                chunk_size=32,
                tile_size=128,
                split_size=64,
            )

        masked_probs_cuda.assert_called_once()
        self.assertEqual(beliefs.shape, local_logits.shape)
        self.assertEqual(final_belief.shape, initial_log_belief.shape)


if __name__ == "__main__":
    unittest.main()
