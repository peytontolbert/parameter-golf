import argparse
import importlib.util
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


def _load_train_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    train_path = root / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("state_space_causal_machine_train_gpt_bench", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec for {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_gpt = _load_train_module()


@dataclass
class BenchCase:
    name: str
    batch_size: int
    seq_len: int
    run_once: Callable[[], None]


def _set_packed_dtype(name: str) -> None:
    os.environ["CAUSAL_MACHINE_SCAN_PACKED_DTYPE"] = str(name)
    train_gpt._cached_env_str.cache_clear()


def _zero_grads(tensors: list[torch.Tensor]) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad = None


def _benchmark_case(case: BenchCase, *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(max(warmup, 0)):
        case.run_once()
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(max(iters, 1)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        case.run_once()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    median_ms = statistics.median(samples_ms)
    mean_ms = statistics.fmean(samples_ms)
    tokens = int(case.batch_size) * int(case.seq_len)
    tokens_per_s = float(tokens) / (median_ms / 1000.0)
    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "tokens_per_s": tokens_per_s,
    }


def _make_dense_case(device: torch.device, *, mode: str) -> BenchCase:
    b, l, n, r = 8, 128, 128, 64
    requires_grad = mode == "forward_backward"
    local_logits = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    source_logits = torch.randn((n, r), device=device, dtype=torch.float32, requires_grad=requires_grad)
    dest_logits = torch.randn((r, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    transition_context = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    initial_log_belief = torch.log_softmax(torch.randn((b, n), device=device, dtype=torch.float32), dim=-1)
    initial_log_belief.requires_grad_(requires_grad)
    transition_gate = torch.tensor(0.35, device=device, dtype=torch.float32, requires_grad=requires_grad)
    transition_stay_probs = torch.sigmoid(torch.randn((n,), device=device, dtype=torch.float32))
    transition_stay_probs.requires_grad_(requires_grad)
    grad_tensors = [
        local_logits,
        source_logits,
        dest_logits,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
    ]

    def run_once() -> None:
        _zero_grads(grad_tensors)
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
        if mode == "forward_backward":
            (beliefs.sum() + final_belief.sum()).backward()

    return BenchCase("dense", b, l, run_once)


def _make_packed_case(device: torch.device, *, mode: str, packed_dtype: str) -> BenchCase:
    b, l, n, r = 8, 128, 128, 64
    requires_grad = mode == "forward_backward"
    local_logits = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    source_logits = torch.randn((n, r), device=device, dtype=torch.float32, requires_grad=requires_grad)
    dest_logits = torch.randn((r, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    transition_context = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    initial_log_belief = torch.log_softmax(torch.randn((b, n), device=device, dtype=torch.float32), dim=-1)
    initial_log_belief.requires_grad_(requires_grad)
    transition_gate = torch.tensor(0.35, device=device, dtype=torch.float32, requires_grad=requires_grad)
    transition_stay_probs = torch.sigmoid(torch.randn((n,), device=device, dtype=torch.float32))
    transition_stay_probs.requires_grad_(requires_grad)
    grad_tensors = [
        local_logits,
        source_logits,
        dest_logits,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
    ]
    _set_packed_dtype(packed_dtype)
    packed_cache: dict[str, object] = {}
    packed_tables = train_gpt.get_or_update_scan_transition_prepack(
        packed_cache,
        source_logits.detach(),
        dest_logits.detach(),
        device,
    )
    if packed_tables is None:
        raise RuntimeError(f"failed to build packed tables for {packed_dtype}")

    def run_once() -> None:
        _zero_grads(grad_tensors)
        beliefs, final_belief = train_gpt.causal_machine_scan_cuda(
            local_logits,
            source_logits,
            dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            packed_transition_tables=packed_tables,
            chunk_size=64,
            runtime_config=None,
        )
        if mode == "forward_backward":
            (beliefs.sum() + final_belief.sum()).backward()

    return BenchCase(f"packed_{packed_dtype}", b, l, run_once)


def _make_tiled_case(device: torch.device, *, mode: str) -> BenchCase:
    b, l, n, r = 4, 128, 256, 64
    requires_grad = mode == "forward_backward"
    runtime_config = train_gpt.StructuredScanRuntimeConfig()
    kernel_config = train_gpt.autotune_structured_scan_kernel_config(
        num_states=n,
        transition_rank=r,
        seq_len=l,
        device=device,
        default_chunk_size=64,
        needs_grad=requires_grad,
        runtime_config=runtime_config,
    )
    local_logits = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    source_probs = torch.softmax(torch.randn((n, r), device=device, dtype=torch.float32), dim=-1)
    source_probs.requires_grad_(requires_grad)
    dest_probs = torch.softmax(torch.randn((r, n), device=device, dtype=torch.float32), dim=-1)
    dest_probs.requires_grad_(requires_grad)
    transition_context = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    initial_log_belief = torch.log_softmax(torch.randn((b, n), device=device, dtype=torch.float32), dim=-1)
    initial_log_belief.requires_grad_(requires_grad)
    transition_gate = torch.tensor(0.25, device=device, dtype=torch.float32, requires_grad=requires_grad)
    transition_stay_probs = torch.sigmoid(torch.randn((n,), device=device, dtype=torch.float32))
    transition_stay_probs.requires_grad_(requires_grad)
    grad_tensors = [
        local_logits,
        source_probs,
        dest_probs,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
    ]

    def run_once() -> None:
        _zero_grads(grad_tensors)
        beliefs, final_belief = train_gpt.causal_machine_scan_tiled_cuda(
            local_logits,
            source_probs,
            dest_probs,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            runtime_config=runtime_config,
            chunk_size=int(kernel_config.chunk_size),
            tile_size=int(kernel_config.tile_size),
            split_size=int(kernel_config.split_size),
        )
        if mode == "forward_backward":
            (beliefs.sum() + final_belief.sum()).backward()

    return BenchCase("tiled", b, l, run_once)


def _make_masked_case(device: torch.device, *, mode: str) -> BenchCase:
    b, l, n, r = 4, 128, 256, 64
    requires_grad = mode == "forward_backward"
    transition_mask = (torch.rand((n, n), device=device, dtype=torch.float32) > 0.8).contiguous()
    runtime_config = train_gpt.StructuredScanRuntimeConfig(transition_mask=transition_mask)
    local_logits = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    source_logits = torch.randn((n, r), device=device, dtype=torch.float32, requires_grad=requires_grad)
    dest_logits = torch.randn((r, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    transition_context = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    initial_log_belief = torch.log_softmax(torch.randn((b, n), device=device, dtype=torch.float32), dim=-1)
    initial_log_belief.requires_grad_(requires_grad)
    transition_gate = torch.tensor(0.25, device=device, dtype=torch.float32, requires_grad=requires_grad)
    transition_stay_probs = torch.sigmoid(torch.randn((n,), device=device, dtype=torch.float32))
    transition_stay_probs.requires_grad_(requires_grad)
    grad_tensors = [
        local_logits,
        source_logits,
        dest_logits,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
    ]

    def run_once() -> None:
        _zero_grads(grad_tensors)
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
        if mode == "forward_backward":
            (beliefs.sum() + final_belief.sum()).backward()

    return BenchCase("masked", b, l, run_once)


def _make_sparse_case(device: torch.device, *, mode: str, packed_dtype: str | None) -> BenchCase:
    b, l, n, r = 4, 128, 256, 64
    requires_grad = mode == "forward_backward"
    transition_mask = (torch.rand((n, n), device=device, dtype=torch.float32) > 0.9).contiguous()
    runtime_config = train_gpt.StructuredScanRuntimeConfig(transition_mask=transition_mask)
    local_logits = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    source_logits = torch.randn((n, r), device=device, dtype=torch.float32, requires_grad=requires_grad)
    dest_logits = torch.randn((r, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    transition_context = torch.randn((b, l, n), device=device, dtype=torch.float32, requires_grad=requires_grad)
    initial_log_belief = torch.log_softmax(torch.randn((b, n), device=device, dtype=torch.float32), dim=-1)
    initial_log_belief.requires_grad_(requires_grad)
    transition_gate = torch.tensor(0.25, device=device, dtype=torch.float32, requires_grad=requires_grad)
    transition_stay_probs = torch.sigmoid(torch.randn((n,), device=device, dtype=torch.float32))
    transition_stay_probs.requires_grad_(requires_grad)
    grad_tensors = [
        local_logits,
        source_logits,
        dest_logits,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
    ]
    packed_tables = None
    if packed_dtype is not None:
        _set_packed_dtype(packed_dtype)
        packed_tables = train_gpt.get_or_update_scan_transition_prepack(
            {},
            source_logits.detach(),
            dest_logits.detach(),
            device,
        )
        if packed_tables is None:
            raise RuntimeError(f"failed to build sparse packed tables for {packed_dtype}")
    sparse_tables = train_gpt.get_or_update_scan_transition_sparse_blocks(
        {},
        source_logits.detach(),
        dest_logits.detach(),
        device,
        runtime_config,
        packed_transition_tables=packed_tables,
    )
    if sparse_tables is None:
        raise RuntimeError("failed to build sparse transition tables")

    def run_once() -> None:
        _zero_grads(grad_tensors)
        beliefs, final_belief = train_gpt.causal_machine_scan_sparse_cuda_autograd(
            local_logits,
            source_logits,
            dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            sparse_tables,
            runtime_config=runtime_config,
            chunk_size=64,
        )
        if mode == "forward_backward":
            (beliefs.sum() + final_belief.sum()).backward()

    name = "sparse" if packed_dtype is None else f"sparse_packed_{packed_dtype}"
    return BenchCase(name, b, l, run_once)


def _build_cases(device: torch.device, *, mode: str, selected: list[str]) -> list[BenchCase]:
    builders = {
        "dense": lambda: _make_dense_case(device, mode=mode),
        "packed_int8": lambda: _make_packed_case(device, mode=mode, packed_dtype="int8"),
        "packed_fp8_e4m3": lambda: _make_packed_case(device, mode=mode, packed_dtype="fp8_e4m3"),
        "packed_fp8_e5m2": lambda: _make_packed_case(device, mode=mode, packed_dtype="fp8_e5m2"),
        "tiled": lambda: _make_tiled_case(device, mode=mode),
        "masked": lambda: _make_masked_case(device, mode=mode),
        "sparse": lambda: _make_sparse_case(device, mode=mode, packed_dtype=None),
        "sparse_packed_int8": lambda: _make_sparse_case(device, mode=mode, packed_dtype="int8"),
        "sparse_packed_fp8_e4m3": lambda: _make_sparse_case(device, mode=mode, packed_dtype="fp8_e4m3"),
    }
    names = list(builders.keys()) if selected == ["all"] else selected
    return [builders[name]() for name in names]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark structured scan CUDA kernel paths.")
    parser.add_argument(
        "--case",
        action="append",
        default=["all"],
        choices=[
            "all",
            "dense",
            "packed_int8",
            "packed_fp8_e4m3",
            "packed_fp8_e5m2",
            "tiled",
            "masked",
            "sparse",
            "sparse_packed_int8",
            "sparse_packed_fp8_e4m3",
        ],
        help="Benchmark case to run. Repeat to select multiple cases.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=20, help="Number of timed iterations per case.")
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help="Benchmark forward only or forward+backward.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--print-ncu",
        action="store_true",
        help="Print an example Nsight Compute command for the chosen benchmark script.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for structured scan kernel benchmarks")
    device = torch.device("cuda", int(args.device))
    torch.cuda.set_device(device)
    train_gpt.load_causal_machine_scan_cuda()
    selected_cases = list(dict.fromkeys(args.case))
    if "all" in selected_cases and len(selected_cases) > 1:
        selected_cases = [name for name in selected_cases if name != "all"]
    cases = _build_cases(device, mode=str(args.mode), selected=selected_cases)
    print(f"device=cuda:{int(args.device)} mode={args.mode} warmup={int(args.warmup)} iters={int(args.iters)}")
    for case in cases:
        stats = _benchmark_case(case, warmup=int(args.warmup), iters=int(args.iters))
        print(
            f"{case.name:24s} "
            f"median_ms={stats['median_ms']:.3f} "
            f"mean_ms={stats['mean_ms']:.3f} "
            f"tokens_per_s={stats['tokens_per_s']:.1f}"
        )
    if args.print_ncu:
        script_path = pathlib.Path(__file__).resolve()
        print(
            "ncu example: "
            f"ncu --set full --target-processes all python {script_path} "
            f"--device {int(args.device)} --mode {args.mode} --case tiled"
        )


if __name__ == "__main__":
    main()
