import importlib.util
import os
import pathlib
import sys
import unittest

import torch


def _load_train_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    train_path = root / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("state_space_causal_machine_train_gpt_muon_tests", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec for {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_gpt = _load_train_module()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Muon CUDA tests")
class MuonCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.device = torch.device("cuda")
        cls.ext = train_gpt.load_muon_cuda()

    def _assert_close(self, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float, msg: str):
        self.assertTrue(
            torch.allclose(actual.detach().float().cpu(), expected.detach().float().cpu(), atol=atol, rtol=rtol),
            msg,
        )

    def _make_family_workspace(
        self,
        shape: tuple[int, int],
        bucket_size: int,
        *,
        device: torch.device | None = None,
        workspace_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor | int]:
        target_device = self.device if device is None else device
        family_code = train_gpt._muon_bucket_family_code(shape)
        transpose_input = family_code == 1
        ns_rows = shape[1] if transpose_input else shape[0]
        ns_cols = shape[0] if transpose_input else shape[1]
        if workspace_dtype is None:
            workspace_dtype = train_gpt._muon_family_workspace_dtype(shape, family_code)
        effective = torch.empty((bucket_size, *shape), device=target_device, dtype=torch.float32)
        norms = torch.empty((bucket_size, 1), device=target_device, dtype=torch.float32)
        ns_input = torch.empty((bucket_size, ns_rows, ns_cols), device=target_device, dtype=workspace_dtype)
        gram = torch.empty((bucket_size, ns_rows, ns_rows), device=target_device, dtype=workspace_dtype)
        return {
            "family_code": family_code,
            "effective": effective,
            "norms": norms,
            "ns_input": ns_input,
            "gram": gram,
            "gram_sq": torch.empty_like(gram),
            "next_x": torch.empty_like(ns_input),
        }

    def test_grouped_step_matches_python_reference(self):
        shape = (64, 32)
        bucket_size = 3
        lr = 0.0623
        momentum = 0.96
        weight_decay = 0.03
        backend_steps = 5
        params = [torch.randn(shape, device=self.device, dtype=torch.float32) for _ in range(bucket_size)]
        grads = [torch.randn(shape, device=self.device, dtype=torch.float32) for _ in range(bucket_size)]
        moms = [torch.randn(shape, device=self.device, dtype=torch.float32) for _ in range(bucket_size)]

        ref_params = [p.detach().clone() for p in params]
        ref_moms = [m.detach().clone() for m in moms]
        test_params = [p.detach().clone() for p in params]
        test_grads = [g.detach().clone() for g in grads]
        test_moms = [m.detach().clone() for m in moms]

        for idx in range(bucket_size):
            state = {"momentum_buffer": ref_moms[idx]}
            train_gpt._muon_python_step_param(
                ref_params[idx],
                grads[idx],
                state,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True,
                backend_steps=backend_steps,
            )

        self.ext.grouped_step(
            test_params,
            test_grads,
            test_moms,
            lr,
            momentum,
            weight_decay,
            True,
            backend_steps,
            1.0e-7,
        )

        for idx, (actual, expected) in enumerate(zip(test_params, ref_params, strict=True)):
            self._assert_close(actual, expected, atol=3.0e-3, rtol=3.0e-3, msg=f"param {idx} mismatch")
        for idx, (actual, expected) in enumerate(zip(test_moms, ref_moms, strict=True)):
            self._assert_close(actual, expected, atol=1.0e-5, rtol=1.0e-5, msg=f"momentum {idx} mismatch")

        workspace_grad_batch = torch.empty((bucket_size, *shape), device=self.device, dtype=torch.float32)
        workspace_momentum = torch.stack([m.detach().clone() for m in moms], dim=0)
        workspace_params = [p.detach().clone() for p in params]
        workspace_grads = [g.detach().clone() for g in grads]
        self.ext.grouped_step_workspace(
            workspace_params,
            workspace_grads,
            workspace_grad_batch,
            workspace_momentum,
            lr,
            momentum,
            weight_decay,
            True,
            backend_steps,
            1.0e-7,
        )
        for idx, (actual, expected) in enumerate(zip(workspace_params, ref_params, strict=True)):
            self._assert_close(actual, expected, atol=3.0e-3, rtol=3.0e-3, msg=f"workspace param {idx} mismatch")
        for idx, (actual, expected) in enumerate(zip(workspace_momentum, ref_moms, strict=True)):
            self._assert_close(actual, expected, atol=1.0e-5, rtol=1.0e-5, msg=f"workspace momentum {idx} mismatch")

        family_params = [p.detach().clone() for p in params]
        family_grads = [g.detach().clone() for g in grads]
        family_momentum = torch.stack([m.detach().clone() for m in moms], dim=0)
        family_workspace = self._make_family_workspace(shape, bucket_size)
        self.ext.grouped_step_family_workspace(
            family_params,
            family_grads,
            family_workspace["effective"],
            family_momentum,
            family_workspace["norms"],
            family_workspace["ns_input"],
            family_workspace["gram"],
            family_workspace["gram_sq"],
            family_workspace["next_x"],
            int(family_workspace["family_code"]),
            lr,
            momentum,
            weight_decay,
            True,
            backend_steps,
            1.0e-7,
        )
        for idx, (actual, expected) in enumerate(zip(family_params, ref_params, strict=True)):
            self._assert_close(actual, expected, atol=3.0e-3, rtol=3.0e-3, msg=f"family param {idx} mismatch")
        for idx, (actual, expected) in enumerate(zip(family_momentum, ref_moms, strict=True)):
            self._assert_close(actual, expected, atol=1.0e-5, rtol=1.0e-5, msg=f"family momentum {idx} mismatch")

    def test_optimizer_auto_bucket_policy_routes_competition_square_buckets(self):
        previous_policy = train_gpt.MUON_CUDA_BUCKET_POLICY
        train_gpt.MUON_CUDA_BUCKET_POLICY = "auto"
        try:
            square_params = [torch.randn((640, 640), device=self.device, dtype=torch.float32) for _ in range(12)]
            rect_params = [torch.randn((320, 640), device=self.device, dtype=torch.float32) for _ in range(4)]
            all_params = square_params + rect_params
            optimizer = train_gpt.Muon(
                all_params,
                lr=0.05,
                momentum=0.95,
                backend_steps=5,
                backend_steps_light=5,
                backend_refresh_interval=1,
                weight_decay=0.01,
                nesterov=True,
            )
            grads = [torch.randn_like(param) for param in all_params]
            for param, grad in zip(all_params, grads, strict=True):
                param.grad = grad
            optimizer.step()
            stats = optimizer.last_step_stats
            self.assertEqual(int(stats.get("cuda_tensor_count", 0)), len(all_params))
            self.assertEqual(int(stats.get("fallback_tensor_count", 0)), 0)
            self.assertEqual(int(stats.get("bucket_count", 0)), 2)
        finally:
            train_gpt.MUON_CUDA_BUCKET_POLICY = previous_policy

    def test_optimizer_auto_bucket_policy_keeps_small_square_buckets_on_python(self):
        previous_policy = train_gpt.MUON_CUDA_BUCKET_POLICY
        train_gpt.MUON_CUDA_BUCKET_POLICY = "auto"
        try:
            square_params = [torch.randn((64, 64), device=self.device, dtype=torch.float32) for _ in range(4)]
            rect_params = [torch.randn((32, 64), device=self.device, dtype=torch.float32) for _ in range(4)]
            all_params = square_params + rect_params
            optimizer = train_gpt.Muon(
                all_params,
                lr=0.05,
                momentum=0.95,
                backend_steps=5,
                backend_steps_light=5,
                backend_refresh_interval=1,
                weight_decay=0.01,
                nesterov=True,
            )
            grads = [torch.randn_like(param) for param in all_params]
            for param, grad in zip(all_params, grads, strict=True):
                param.grad = grad
            optimizer.step()
            stats = optimizer.last_step_stats
            self.assertEqual(int(stats.get("cuda_tensor_count", 0)), len(rect_params))
            self.assertEqual(int(stats.get("fallback_tensor_count", 0)), len(square_params))
            self.assertEqual(int(stats.get("bucket_count", 0)), 2)
        finally:
            train_gpt.MUON_CUDA_BUCKET_POLICY = previous_policy

    def test_square_backend_policy_variants_match_reference(self):
        shape = (128, 128)
        bucket_size = 4
        lr = 0.05
        momentum = 0.95
        weight_decay = 0.01
        backend_steps = 5
        params = [torch.randn(shape, device=self.device, dtype=torch.float32) for _ in range(bucket_size)]
        grads = [torch.randn(shape, device=self.device, dtype=torch.float32) for _ in range(bucket_size)]
        moms = [torch.randn(shape, device=self.device, dtype=torch.float32) for _ in range(bucket_size)]

        ref_params = [p.detach().clone() for p in params]
        ref_moms = [m.detach().clone() for m in moms]
        for idx in range(bucket_size):
            state = {"momentum_buffer": ref_moms[idx]}
            train_gpt._muon_python_step_param(
                ref_params[idx],
                grads[idx],
                state,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True,
                backend_steps=backend_steps,
            )

        previous_backend = os.environ.get("MUON_SQUARE_BACKEND")
        previous_workspace_dtype = os.environ.get("MUON_FAMILY_WORKSPACE_DTYPE")
        try:
            for square_backend in ("cublas", "cublaslt", "hybrid"):
                os.environ["MUON_SQUARE_BACKEND"] = square_backend
                os.environ["MUON_FAMILY_WORKSPACE_DTYPE"] = "bf16"
                family_params = [p.detach().clone() for p in params]
                family_grads = [g.detach().clone() for g in grads]
                family_momentum = torch.stack([m.detach().clone() for m in moms], dim=0)
                family_workspace = self._make_family_workspace(shape, bucket_size)
                self.assertEqual(family_workspace["ns_input"].dtype, torch.bfloat16)
                self.ext.grouped_step_family_workspace(
                    family_params,
                    family_grads,
                    family_workspace["effective"],
                    family_momentum,
                    family_workspace["norms"],
                    family_workspace["ns_input"],
                    family_workspace["gram"],
                    family_workspace["gram_sq"],
                    family_workspace["next_x"],
                    int(family_workspace["family_code"]),
                    lr,
                    momentum,
                    weight_decay,
                    True,
                    backend_steps,
                    1.0e-7,
                )
                for idx, (actual, expected) in enumerate(zip(family_params, ref_params, strict=True)):
                    self._assert_close(
                        actual,
                        expected,
                        atol=3.0e-3,
                        rtol=3.0e-3,
                        msg=f"{square_backend} square param {idx} mismatch",
                    )
                for idx, (actual, expected) in enumerate(zip(family_momentum, ref_moms, strict=True)):
                    self._assert_close(
                        actual,
                        expected,
                        atol=1.0e-5,
                        rtol=1.0e-5,
                        msg=f"{square_backend} square momentum {idx} mismatch",
                    )
        finally:
            if previous_backend is None:
                os.environ.pop("MUON_SQUARE_BACKEND", None)
            else:
                os.environ["MUON_SQUARE_BACKEND"] = previous_backend
            if previous_workspace_dtype is None:
                os.environ.pop("MUON_FAMILY_WORKSPACE_DTYPE", None)
            else:
                os.environ["MUON_FAMILY_WORKSPACE_DTYPE"] = previous_workspace_dtype

    def test_square_backend_api_and_stats_are_reported(self):
        shape = (128, 128)
        bucket_size = 8
        previous_policy = train_gpt.MUON_CUDA_BUCKET_POLICY
        previous_backend = os.environ.get("MUON_SQUARE_BACKEND")
        previous_workspace_dtype = os.environ.get("MUON_FAMILY_WORKSPACE_DTYPE")
        try:
            train_gpt.MUON_CUDA_BUCKET_POLICY = "auto"
            os.environ["MUON_SQUARE_BACKEND"] = "cublaslt"
            square_batch = torch.empty((bucket_size, *shape), device=self.device, dtype=torch.bfloat16)
            backend_code = int(self.ext.describe_square_backend(square_batch))
            self.assertEqual(backend_code, 2)
            self.ext.prewarm_square_backend(square_batch)
            os.environ.pop("MUON_FAMILY_WORKSPACE_DTYPE", None)

            params = [torch.randn(shape, device=self.device, dtype=torch.float32) for _ in range(bucket_size)]
            optimizer = train_gpt.Muon(
                params,
                lr=0.05,
                momentum=0.95,
                backend_steps=5,
                backend_steps_light=5,
                backend_refresh_interval=1,
                weight_decay=0.01,
                nesterov=True,
            )
            for param in params:
                param.grad = torch.randn_like(param)
            optimizer.step()
            stats = optimizer.last_step_stats
            workspace = next(iter(optimizer._cuda_bucket_workspaces.values()))
            self.assertEqual(workspace.get("workspace_dtype"), "float32")
            self.assertEqual(workspace.get("square_backend"), "cublas")
            self.assertEqual(stats.get("square_backend_modes"), "cublas")
            self.assertEqual(int(stats.get("square_backend_cublas_bucket_count", 0)), 1)
            self.assertEqual(int(stats.get("square_backend_cublas_tensor_count", 0)), bucket_size)
        finally:
            train_gpt.MUON_CUDA_BUCKET_POLICY = previous_policy
            if previous_backend is None:
                os.environ.pop("MUON_SQUARE_BACKEND", None)
            else:
                os.environ["MUON_SQUARE_BACKEND"] = previous_backend
            if previous_workspace_dtype is None:
                os.environ.pop("MUON_FAMILY_WORKSPACE_DTYPE", None)
            else:
                os.environ["MUON_FAMILY_WORKSPACE_DTYPE"] = previous_workspace_dtype

    def test_prepare_cuda_graph_capture_caches_pointer_tensors(self):
        shape = (128, 128)
        bucket_size = 8
        previous_policy = train_gpt.MUON_CUDA_BUCKET_POLICY
        previous_backend = os.environ.get("MUON_SQUARE_BACKEND")
        previous_workspace_dtype = os.environ.get("MUON_FAMILY_WORKSPACE_DTYPE")
        try:
            train_gpt.MUON_CUDA_BUCKET_POLICY = "auto"
            os.environ["MUON_SQUARE_BACKEND"] = "cublaslt"
            os.environ.pop("MUON_FAMILY_WORKSPACE_DTYPE", None)
            params = [torch.randn(shape, device=self.device, dtype=torch.float32) for _ in range(bucket_size)]
            optimizer = train_gpt.Muon(
                params,
                lr=0.05,
                momentum=0.95,
                backend_steps=5,
                backend_steps_light=5,
                backend_refresh_interval=1,
                weight_decay=0.01,
                nesterov=True,
                capturable=True,
            )
            ready = optimizer.prepare_cuda_graph_capture()
            self.assertTrue(ready)
            self.assertTrue(optimizer.supports_full_step_cuda_graph())
            workspace = next(iter(optimizer._cuda_bucket_workspaces.values()))
            self.assertTrue(isinstance(workspace.get("param_ptrs"), torch.Tensor))
            self.assertTrue(isinstance(workspace.get("grad_ptrs"), torch.Tensor))
            self.assertEqual(workspace["param_ptrs"].dtype, torch.int64)
            self.assertEqual(workspace["grad_ptrs"].dtype, torch.int64)
            self.assertTrue(workspace["param_ptrs"].is_cuda)
            self.assertTrue(workspace["grad_ptrs"].is_cuda)
            self.assertEqual(workspace.get("workspace_dtype"), "float32")
            self.assertEqual(workspace.get("square_backend"), "cublas")
        finally:
            train_gpt.MUON_CUDA_BUCKET_POLICY = previous_policy
            if previous_backend is None:
                os.environ.pop("MUON_SQUARE_BACKEND", None)
            else:
                os.environ["MUON_SQUARE_BACKEND"] = previous_backend
            if previous_workspace_dtype is None:
                os.environ.pop("MUON_FAMILY_WORKSPACE_DTYPE", None)
            else:
                os.environ["MUON_FAMILY_WORKSPACE_DTYPE"] = previous_workspace_dtype

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "multiple CUDA devices are required")
    def test_family_workspace_succeeds_with_non_current_target_device(self):
        previous_device = torch.cuda.current_device()
        target_device = torch.device("cuda:1")
        try:
            torch.cuda.set_device(0)
            shape = (128, 128)
            bucket_size = 4
            lr = 0.05
            momentum = 0.95
            weight_decay = 0.01
            backend_steps = 5
            params = [torch.randn(shape, device=target_device, dtype=torch.float32) for _ in range(bucket_size)]
            grads = [torch.randn(shape, device=target_device, dtype=torch.float32) for _ in range(bucket_size)]
            family_momentum = torch.zeros((bucket_size, *shape), device=target_device, dtype=torch.float32)
            family_workspace = self._make_family_workspace(shape, bucket_size, device=target_device)

            self.ext.grouped_step_family_workspace(
                params,
                grads,
                family_workspace["effective"],
                family_momentum,
                family_workspace["norms"],
                family_workspace["ns_input"],
                family_workspace["gram"],
                family_workspace["gram_sq"],
                family_workspace["next_x"],
                int(family_workspace["family_code"]),
                lr,
                momentum,
                weight_decay,
                True,
                backend_steps,
                1.0e-7,
            )

            self.assertTrue(torch.isfinite(family_momentum).all().item())
            self.assertEqual(family_workspace["ns_input"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(family_workspace["ns_input"].float()).all().item())
        finally:
            torch.cuda.set_device(previous_device)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "multiple CUDA devices are required")
    def test_capturable_family_workspace_rejects_cross_device_scalars(self):
        previous_device = torch.cuda.current_device()
        target_device = torch.device("cuda:1")
        try:
            torch.cuda.set_device(0)
            shape = (128, 128)
            bucket_size = 4
            params = [torch.randn(shape, device=target_device, dtype=torch.float32) for _ in range(bucket_size)]
            grads = [torch.randn(shape, device=target_device, dtype=torch.float32) for _ in range(bucket_size)]
            param_ptrs = torch.tensor(
                [p.data_ptr() for p in params], device=target_device, dtype=torch.int64
            )
            grad_ptrs = torch.tensor(
                [g.data_ptr() for g in grads], device=target_device, dtype=torch.int64
            )
            family_momentum = torch.zeros((bucket_size, *shape), device=target_device, dtype=torch.float32)
            family_workspace = self._make_family_workspace(shape, bucket_size, device=target_device)
            lr = torch.tensor(0.05, device="cuda:0", dtype=torch.float32)
            momentum = torch.tensor(0.95, device=target_device, dtype=torch.float32)
            weight_decay = torch.tensor(0.01, device=target_device, dtype=torch.float32)

            with self.assertRaisesRegex(RuntimeError, "expects lr on cuda:1"):
                self.ext.grouped_step_family_workspace_capturable(
                    params,
                    grads,
                    param_ptrs,
                    grad_ptrs,
                    family_workspace["effective"],
                    family_momentum,
                    family_workspace["norms"],
                    family_workspace["ns_input"],
                    family_workspace["gram"],
                    family_workspace["gram_sq"],
                    family_workspace["next_x"],
                    int(family_workspace["family_code"]),
                    lr,
                    momentum,
                    weight_decay,
                    True,
                    5,
                    1.0e-7,
                )
        finally:
            torch.cuda.set_device(previous_device)


if __name__ == "__main__":
    unittest.main()
