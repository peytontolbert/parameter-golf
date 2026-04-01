#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

import torch


def _load_train_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    train_path = root / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("state_space_causal_machine_train_gpt_perf", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec for {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_gpt = _load_train_module()
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _default_matrix_for_arch(arch_family: str) -> dict[str, str]:
    matrix = {
        "batch_sizes": "1,4,8",
        "seq_lens": "512,2048",
        "num_states": "160,256",
        "transition_ranks": "64,96",
        "tile_sizes": "128,160",
        "split_sizes": "64,96",
        "dtype": "bf16",
    }
    if arch_family == "sm90+":
        matrix.update(
            seq_lens="1024,4096",
            num_states="192,256,384",
            transition_ranks="64,96,128",
            tile_sizes="128,192",
            split_sizes="64,96,128",
        )
    elif arch_family == "sm100+":
        matrix.update(
            seq_lens="1024,4096,8192",
            num_states="192,256,384,512",
            transition_ranks="64,96,128,160",
            tile_sizes="128,192,256",
            split_sizes="64,96,128,160",
        )
    return matrix


def _run_and_capture(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return completed.stdout


def _parse_csv_output(raw: str) -> list[dict[str, str]]:
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    return [dict(row) for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run structured scan perf sweeps with arch-aware defaults")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--reuse-workspace", action="store_true")
    parser.add_argument("--include-kernel-bench", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    arch_spec = train_gpt._describe_structured_scan_arch_spec(device)
    matrix = _default_matrix_for_arch(str(arch_spec.arch_family))
    structured_scan_bench = ROOT / "benchmarks" / "structured_scan_bench.py"
    kernel_bench = ROOT / "benchmarks" / "bench_structured_scan_kernels.py"
    cmd = [
        sys.executable,
        str(structured_scan_bench),
        "--batch-sizes",
        matrix["batch_sizes"],
        "--seq-lens",
        matrix["seq_lens"],
        "--num-states",
        matrix["num_states"],
        "--transition-ranks",
        matrix["transition_ranks"],
        "--tile-sizes",
        matrix["tile_sizes"],
        "--split-sizes",
        matrix["split_sizes"],
        "--dtype",
        matrix["dtype"],
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    if args.reuse_workspace:
        cmd.append("--reuse-workspace")
    raw_csv = _run_and_capture(cmd)
    matrix_rows = _parse_csv_output(raw_csv)
    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "arch_family": str(arch_spec.arch_family),
        "capability_major": int(arch_spec.capability_major),
        "capability_minor": int(arch_spec.capability_minor),
        "supports_async_pipeline": bool(arch_spec.supports_async_pipeline),
        "supports_tensor_memory_accel": bool(arch_spec.supports_tensor_memory_accel),
        "supports_cluster_launch_control": bool(arch_spec.supports_cluster_launch_control),
        "sweep_matrix": matrix,
        "structured_scan_bench": matrix_rows,
    }
    if args.include_kernel_bench:
        raw_kernel = _run_and_capture([sys.executable, str(kernel_bench), "--mode", "forward"])
        output["kernel_bench_stdout"] = raw_kernel
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        pathlib.Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
