#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass

import torch


def _load_train_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    train_path = root / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("state_space_causal_machine_train_gpt_muon_bench", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec for {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_gpt = _load_train_module()


@dataclass(frozen=True)
class BucketSpec:
    name: str
    shape: tuple[int, int]
    bucket_size: int


@dataclass(frozen=True)
class BenchCase:
    name: str
    buckets: tuple[BucketSpec, ...]

    @property
    def total_elements(self) -> int:
        return sum(int(spec.bucket_size) * int(spec.shape[0]) * int(spec.shape[1]) for spec in self.buckets)


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name}") from exc


def _build_cases(
    *,
    model_dim: int,
    mlp_hidden: int,
    num_heads: int,
    num_kv_heads: int,
    attn_blocks: int,
    mlp_blocks: int,
) -> dict[str, BenchCase]:
    if model_dim <= 0 or mlp_hidden <= 0 or num_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("model_dim, mlp_hidden, num_heads, and num_kv_heads must be positive")
    if model_dim % num_heads != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")
    kv_dim = model_dim * num_kv_heads // num_heads
    return {
        "attn_square": BenchCase(
            name="attn_square",
            buckets=(BucketSpec("attn_square", (model_dim, model_dim), 2 * max(attn_blocks, 0)),),
        ),
        "attn_kv": BenchCase(
            name="attn_kv",
            buckets=(BucketSpec("attn_kv", (kv_dim, model_dim), 2 * max(attn_blocks, 0)),),
        ),
        "mlp_up": BenchCase(
            name="mlp_up",
            buckets=(BucketSpec("mlp_up", (mlp_hidden, model_dim), max(mlp_blocks, 0)),),
        ),
        "mlp_down": BenchCase(
            name="mlp_down",
            buckets=(BucketSpec("mlp_down", (model_dim, mlp_hidden), max(mlp_blocks, 0)),),
        ),
        "competition_full": BenchCase(
            name="competition_full",
            buckets=(
                BucketSpec("attn_square", (model_dim, model_dim), 2 * max(attn_blocks, 0)),
                BucketSpec("attn_kv", (kv_dim, model_dim), 2 * max(attn_blocks, 0)),
                BucketSpec("mlp_up", (mlp_hidden, model_dim), max(mlp_blocks, 0)),
                BucketSpec("mlp_down", (model_dim, mlp_hidden), max(mlp_blocks, 0)),
            ),
        ),
    }


def _build_bucket_tensors(
    spec: BucketSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    params = [torch.randn(spec.shape, device=device, dtype=dtype) for _ in range(spec.bucket_size)]
    grads = [torch.randn(spec.shape, device=device, dtype=dtype) for _ in range(spec.bucket_size)]
    moms = [torch.randn(spec.shape, device=device, dtype=dtype) for _ in range(spec.bucket_size)]
    return params, grads, moms


def _make_python_runner(
    case: BenchCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    backend_steps: int,
):
    state: list[tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, torch.Tensor]]]] = []
    for spec in case.buckets:
        params, grads, moms = _build_bucket_tensors(spec, device=device, dtype=dtype)
        states = [{"momentum_buffer": mom} for mom in moms]
        state.append((params, grads, states))

    def run_once() -> None:
        for params, grads, states in state:
            for param, grad, param_state in zip(params, grads, states, strict=True):
                train_gpt._muon_python_step_param(
                    param,
                    grad,
                    param_state,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    nesterov=nesterov,
                    backend_steps=backend_steps,
                )

    return run_once


def _make_cuda_runner(
    case: BenchCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    backend_steps: int,
    square_backend: str,
):
    os.environ["MUON_SQUARE_BACKEND"] = str(square_backend)
    ext = train_gpt.load_muon_cuda()
    state: list[tuple[list[torch.Tensor], list[torch.Tensor], dict[str, torch.Tensor | int]]] = []
    for spec in case.buckets:
        params, grads, moms = _build_bucket_tensors(spec, device=device, dtype=dtype)
        family_code = train_gpt._muon_bucket_family_code(spec.shape)
        transpose_input = family_code == 1
        ns_rows = spec.shape[1] if transpose_input else spec.shape[0]
        ns_cols = spec.shape[0] if transpose_input else spec.shape[1]
        momentum_batch = torch.stack([mom.to(torch.float32) for mom in moms], dim=0)
        workspace: dict[str, torch.Tensor | int] = {
            "family_code": family_code,
            "effective_batch": torch.empty((spec.bucket_size, *spec.shape), device=device, dtype=torch.float32),
            "momentum_batch": momentum_batch,
            "norms": torch.empty((spec.bucket_size, 1), device=device, dtype=torch.float32),
            "ns_input_batch": torch.empty((spec.bucket_size, ns_rows, ns_cols), device=device, dtype=torch.bfloat16),
            "gram_batch": torch.empty((spec.bucket_size, ns_rows, ns_rows), device=device, dtype=torch.bfloat16),
        }
        workspace["gram_sq_batch"] = torch.empty_like(workspace["gram_batch"])
        workspace["next_x_batch"] = torch.empty_like(workspace["ns_input_batch"])
        state.append((params, grads, workspace))

    def run_once() -> None:
        for params, grads, workspace in state:
            ext.grouped_step_family_workspace(
                params,
                grads,
                workspace["effective_batch"],
                workspace["momentum_batch"],
                workspace["norms"],
                workspace["ns_input_batch"],
                workspace["gram_batch"],
                workspace["gram_sq_batch"],
                workspace["next_x_batch"],
                int(workspace["family_code"]),
                float(lr),
                float(momentum),
                float(weight_decay),
                bool(nesterov),
                int(backend_steps),
                1.0e-7,
            )

    return run_once


def _make_optimizer_runner(
    case: BenchCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    backend_steps: int,
    bucket_policy: str,
    square_backend: str,
):
    params: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    previous_policy = train_gpt.MUON_CUDA_BUCKET_POLICY
    train_gpt.MUON_CUDA_BUCKET_POLICY = str(bucket_policy)
    for spec in case.buckets:
        bucket_params, bucket_grads, _ = _build_bucket_tensors(spec, device=device, dtype=dtype)
        params.extend(bucket_params)
        grads.extend(bucket_grads)
    optimizer = train_gpt.Muon(
        params,
        lr=lr,
        momentum=momentum,
        backend_steps=backend_steps,
        backend_steps_light=backend_steps,
        backend_refresh_interval=1,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

    def run_once() -> None:
        train_gpt.MUON_CUDA_BUCKET_POLICY = str(bucket_policy)
        os.environ["MUON_SQUARE_BACKEND"] = str(square_backend)
        for param, grad in zip(params, grads, strict=True):
            param.grad = grad
        optimizer.step()

    run_once._restore_bucket_policy = previous_policy  # type: ignore[attr-defined]
    return run_once


def _benchmark(run_once, *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(max(int(warmup), 0)):
        run_once()
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(max(int(iters), 1)):
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started_at) * 1000.0)
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the grouped Muon CUDA optimizer path.")
    parser.add_argument(
        "--case",
        action="append",
        default=["competition_full"],
        choices=["competition_full", "attn_square", "attn_kv", "mlp_up", "mlp_down", "all"],
        help="Benchmark case to run. Repeat to select multiple cases.",
    )
    parser.add_argument("--backend", choices=["python", "cuda", "auto", "both"], default="both")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--model-dim", type=int, default=640)
    parser.add_argument("--mlp-hidden", type=int, default=1280)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument(
        "--attn-blocks",
        type=int,
        default=6,
        help="Attention blocks in the competition recipe. Used to size Muon attention buckets.",
    )
    parser.add_argument(
        "--mlp-blocks",
        type=int,
        default=10,
        help="Blocks with MLP matrices in the competition recipe. Used to size Muon MLP buckets.",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--backend-steps", type=int, default=5)
    parser.add_argument(
        "--square-backend",
        choices=["auto", "cublas", "cublaslt", "hybrid"],
        default="auto",
        help="Square-family Newton-Schulz backend policy inside the Muon CUDA extension.",
    )
    parser.add_argument("--print-ncu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Muon CUDA benchmarks")
    device = torch.device("cuda", int(args.device))
    torch.cuda.set_device(device)
    dtype = _dtype_from_name(str(args.dtype))
    all_cases = _build_cases(
        model_dim=int(args.model_dim),
        mlp_hidden=int(args.mlp_hidden),
        num_heads=int(args.num_heads),
        num_kv_heads=int(args.num_kv_heads),
        attn_blocks=int(args.attn_blocks),
        mlp_blocks=int(args.mlp_blocks),
    )
    selected = list(dict.fromkeys(args.case))
    if "all" in selected:
        selected = [name for name in all_cases if name != "competition_full"] + ["competition_full"]
    print(
        f"device=cuda:{int(args.device)} dtype={args.dtype} warmup={int(args.warmup)} "
        f"iters={int(args.iters)} backend={args.backend} square_backend={args.square_backend}"
    )
    for case_name in selected:
        case = all_cases[case_name]
        backends = ("python", "cuda") if args.backend == "both" else (str(args.backend),)
        stats_by_backend: dict[str, dict[str, float]] = {}
        restore_bucket_policy = None
        for backend in backends:
            if backend == "python":
                runner = _make_python_runner(
                    case,
                    device=device,
                    dtype=dtype,
                    lr=float(args.lr),
                    momentum=float(args.momentum),
                    weight_decay=float(args.weight_decay),
                    nesterov=True,
                    backend_steps=int(args.backend_steps),
                )
            elif backend == "auto":
                runner = _make_optimizer_runner(
                    case,
                    device=device,
                    dtype=dtype,
                    lr=float(args.lr),
                    momentum=float(args.momentum),
                    weight_decay=float(args.weight_decay),
                    nesterov=True,
                    backend_steps=int(args.backend_steps),
                    bucket_policy="auto",
                    square_backend=str(args.square_backend),
                )
                restore_bucket_policy = getattr(runner, "_restore_bucket_policy", None)
            else:
                runner = _make_cuda_runner(
                    case,
                    device=device,
                    dtype=dtype,
                    lr=float(args.lr),
                    momentum=float(args.momentum),
                    weight_decay=float(args.weight_decay),
                    nesterov=True,
                    backend_steps=int(args.backend_steps),
                    square_backend=str(args.square_backend),
                )
            stats_by_backend[backend] = _benchmark(runner, warmup=int(args.warmup), iters=int(args.iters))
        if restore_bucket_policy is not None:
            train_gpt.MUON_CUDA_BUCKET_POLICY = restore_bucket_policy

        line = f"{case.name:18s} elements={case.total_elements:9d}"
        if "python" in stats_by_backend:
            line += (
                f" python_median_ms={stats_by_backend['python']['median_ms']:.3f}"
                f" python_mean_ms={stats_by_backend['python']['mean_ms']:.3f}"
            )
        if "cuda" in stats_by_backend:
            line += (
                f" cuda_median_ms={stats_by_backend['cuda']['median_ms']:.3f}"
                f" cuda_mean_ms={stats_by_backend['cuda']['mean_ms']:.3f}"
            )
        if "auto" in stats_by_backend:
            line += (
                f" auto_median_ms={stats_by_backend['auto']['median_ms']:.3f}"
                f" auto_mean_ms={stats_by_backend['auto']['mean_ms']:.3f}"
            )
        if "python" in stats_by_backend and "cuda" in stats_by_backend:
            speedup = stats_by_backend["python"]["median_ms"] / max(stats_by_backend["cuda"]["median_ms"], 1.0e-9)
            line += f" speedup={speedup:.3f}x"
        print(line)

    if args.print_ncu:
        script_path = pathlib.Path(__file__).resolve()
        print(
            "ncu example: "
            f"ncu --set full --target-processes all python {script_path} "
            f"--device {int(args.device)} --backend cuda --case competition_full"
        )


if __name__ == "__main__":
    main()
