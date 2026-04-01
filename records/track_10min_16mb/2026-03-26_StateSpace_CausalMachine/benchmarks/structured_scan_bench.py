#!/usr/bin/env python3

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_gpt import (  # noqa: E402
    StructuredScanRuntimeConfig,
    causal_machine_scan_tiled_cuda,
    create_structured_scan_workspace,
)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _bench_once(
    *,
    batch_size: int,
    seq_len: int,
    num_states: int,
    transition_rank: int,
    tile_size: int,
    split_size: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    reuse_workspace: bool,
) -> dict[str, float | int | str]:
    device = torch.device("cuda", 0)
    local_logits = torch.randn(batch_size, seq_len, num_states, device=device, dtype=dtype)
    transition_source = torch.softmax(
        torch.randn(num_states, transition_rank, device=device, dtype=torch.float32), dim=-1
    ).contiguous()
    transition_dest = torch.softmax(
        torch.randn(transition_rank, num_states, device=device, dtype=torch.float32), dim=-1
    ).contiguous()
    transition_context = torch.randn_like(local_logits)
    initial_log_belief = torch.log_softmax(
        torch.randn(batch_size, num_states, device=device, dtype=torch.float32), dim=-1
    ).contiguous()
    transition_gate = torch.ones((), device=device, dtype=torch.float32)
    transition_stay = torch.full((num_states,), 0.1, device=device, dtype=torch.float32)
    runtime_config = StructuredScanRuntimeConfig()
    workspace = None
    if reuse_workspace:
        workspace = create_structured_scan_workspace(
            mode="tiled_forward",
            device=device,
            num_states=num_states,
            transition_rank=transition_rank,
            batch_size=batch_size,
            seq_len=seq_len,
            chunk_size=seq_len,
            tile_size=tile_size,
            split_size=split_size,
        )
    torch.cuda.synchronize(device)
    for _ in range(max(warmup, 0)):
        causal_machine_scan_tiled_cuda(
            local_logits,
            transition_source,
            transition_dest,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay,
            runtime_config=runtime_config,
            chunk_size=seq_len,
            tile_size=tile_size,
            split_size=split_size,
            workspace=workspace,
        )
    torch.cuda.synchronize(device)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(max(iters, 1))]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(max(iters, 1))]
    for idx in range(max(iters, 1)):
        start_events[idx].record()
        causal_machine_scan_tiled_cuda(
            local_logits,
            transition_source,
            transition_dest,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay,
            runtime_config=runtime_config,
            chunk_size=seq_len,
            tile_size=tile_size,
            split_size=split_size,
            workspace=workspace,
        )
        end_events[idx].record()
    torch.cuda.synchronize(device)
    elapsed_ms = [float(start.elapsed_time(end)) for start, end in zip(start_events, end_events)]
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_states": num_states,
        "transition_rank": transition_rank,
        "tile_size": tile_size,
        "split_size": split_size,
        "dtype": str(dtype).replace("torch.", ""),
        "workspace": "reused" if reuse_workspace else "ephemeral",
        "mean_ms": statistics.fmean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone structured scan benchmark sweeps")
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--seq-lens", default="512,2048")
    parser.add_argument("--num-states", default="160,256")
    parser.add_argument("--transition-ranks", default="64,96")
    parser.add_argument("--tile-sizes", default="128,160")
    parser.add_argument("--split-sizes", default="64,96")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--reuse-workspace", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for structured_scan_bench.py")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(
        "batch_size,seq_len,num_states,transition_rank,tile_size,split_size,dtype,workspace,mean_ms,median_ms,min_ms,max_ms"
    )
    for batch_size in _parse_csv_ints(args.batch_sizes):
        for seq_len in _parse_csv_ints(args.seq_lens):
            for num_states in _parse_csv_ints(args.num_states):
                for transition_rank in _parse_csv_ints(args.transition_ranks):
                    if transition_rank > num_states:
                        continue
                    for tile_size in _parse_csv_ints(args.tile_sizes):
                        if tile_size > num_states:
                            continue
                        for split_size in _parse_csv_ints(args.split_sizes):
                            if split_size > transition_rank:
                                continue
                            result = _bench_once(
                                batch_size=batch_size,
                                seq_len=seq_len,
                                num_states=num_states,
                                transition_rank=transition_rank,
                                tile_size=tile_size,
                                split_size=split_size,
                                dtype=dtype,
                                warmup=args.warmup,
                                iters=args.iters,
                                reuse_workspace=args.reuse_workspace,
                            )
                            print(
                                ",".join(
                                    str(result[key])
                                    for key in (
                                        "batch_size",
                                        "seq_len",
                                        "num_states",
                                        "transition_rank",
                                        "tile_size",
                                        "split_size",
                                        "dtype",
                                        "workspace",
                                        "mean_ms",
                                        "median_ms",
                                        "min_ms",
                                        "max_ms",
                                    )
                                )
                            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
