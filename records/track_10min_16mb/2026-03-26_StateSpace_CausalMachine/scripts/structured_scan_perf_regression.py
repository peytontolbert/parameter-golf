#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two structured scan perf matrix JSON reports")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--max-slower-ratio", type=float, default=1.05)
    args = parser.parse_args()

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    base_rows = {
        (
            row["batch_size"],
            row["seq_len"],
            row["num_states"],
            row["transition_rank"],
            row["tile_size"],
            row["split_size"],
            row["dtype"],
            row["workspace"],
        ): row
        for row in baseline.get("structured_scan_bench", [])
    }
    cand_rows = {
        (
            row["batch_size"],
            row["seq_len"],
            row["num_states"],
            row["transition_rank"],
            row["tile_size"],
            row["split_size"],
            row["dtype"],
            row["workspace"],
        ): row
        for row in candidate.get("structured_scan_bench", [])
    }
    failures: list[str] = []
    for key, base in base_rows.items():
        cand = cand_rows.get(key)
        if cand is None:
            failures.append(f"missing candidate row for {key}")
            continue
        base_ms = float(base["median_ms"])
        cand_ms = float(cand["median_ms"])
        if base_ms <= 0.0:
            continue
        ratio = cand_ms / base_ms
        if ratio > float(args.max_slower_ratio):
            failures.append(f"{key}: candidate median_ms {cand_ms:.4f} > {base_ms:.4f} * {args.max_slower_ratio:.3f}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("structured scan perf regression check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
