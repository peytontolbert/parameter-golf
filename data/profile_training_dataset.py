#!/usr/bin/env python3
"""
Profile token-stream structure for parameter-golf tuning.

This script is meant for local training profiling, not leaderboard submission.
It reads the existing binary shard format and summarizes statistics that help
choose between long-context eval, local lexical features, tokenizer variants,
and compression-aware training heuristics.

The shard format does not preserve document boundaries, so the profiler focuses
on stream-visible signals:
- token entropy and frequency concentration
- lagged token dependence across short/medium/long ranges
- context-window novelty / reuse
- tokenizer byte allocation if a tokenizer is provided
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover
    spm = None

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover
    Tokenizer = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def load_data_shard(file: Path) -> np.ndarray:
    header = np.fromfile(file, dtype="<i4", count=HEADER_INTS)
    header_bytes = HEADER_INTS * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return tokens.astype(np.int32, copy=False)


def resolve_files(pattern: str, max_shards: int) -> list[Path]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    if max_shards > 0:
        files = files[:max_shards]
    return files


def iter_tokens(files: Iterable[Path], max_tokens: int) -> np.ndarray:
    arrays: list[np.ndarray] = []
    remaining = max_tokens if max_tokens > 0 else None
    for file in files:
        arr = load_data_shard(file)
        if remaining is not None:
            if remaining <= 0:
                break
            if arr.size > remaining:
                arr = arr[:remaining]
            remaining -= int(arr.size)
        arrays.append(arr)
        if remaining == 0:
            break
    if not arrays:
        raise ValueError("No tokens loaded")
    return np.ascontiguousarray(np.concatenate(arrays, axis=0))


def sample_tokens_from_shard(file: Path, sample_tokens: int) -> np.ndarray:
    tokens = load_data_shard(file)
    if sample_tokens <= 0 or sample_tokens >= tokens.size:
        return tokens
    idx = np.linspace(0, tokens.size - 1, num=sample_tokens, dtype=np.int64)
    return np.ascontiguousarray(tokens[idx])


def sample_tokens_across_shards(files: list[Path], total_tokens: int) -> np.ndarray:
    if total_tokens <= 0:
        return iter_tokens(files, total_tokens)
    per_shard = max(total_tokens // max(len(files), 1), 1)
    arrays = [sample_tokens_from_shard(file, per_shard) for file in files]
    merged = np.ascontiguousarray(np.concatenate(arrays, axis=0))
    return merged[:total_tokens]


def shard_token_totals(files: list[Path]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    total_tokens = 0
    for file in files:
        header = np.fromfile(file, dtype="<i4", count=HEADER_INTS)
        if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
            raise ValueError(f"Unexpected shard header for {file}")
        num_tokens = int(header[2])
        rows.append({"file": str(file), "num_tokens": num_tokens})
        total_tokens += num_tokens
    return {
        "total_tokens": int(total_tokens),
        "num_shards": int(len(files)),
        "shards": rows,
    }


def build_base_bytes(tokenizer_path: str, vocab_size: int) -> np.ndarray | None:
    if not tokenizer_path:
        return None
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    table = np.zeros((vocab_size,), dtype=np.int32)
    if path.suffix == ".model":
        if spm is None:
            raise RuntimeError("sentencepiece is not installed")
        sp = spm.SentencePieceProcessor(model_file=str(path))
        for token_id in range(min(vocab_size, int(sp.vocab_size()))):
            if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
                continue
            if sp.is_byte(token_id):
                table[token_id] = 1
                continue
            piece = sp.id_to_piece(token_id)
            if piece.startswith("▁"):
                piece = piece[1:]
            table[token_id] = len(piece.encode("utf-8"))
        return table

    if path.suffix == ".json":
        if Tokenizer is None:
            raise RuntimeError("tokenizers is not installed")
        tok = Tokenizer.from_file(str(path))
        vocab = tok.get_vocab(with_added_tokens=True)
        inverse_vocab = {idx: token for token, idx in vocab.items()}
        for token_id in range(vocab_size):
            token = inverse_vocab.get(token_id)
            if token is None:
                continue
            cleaned = token.replace("Ġ", "").replace("▁", "")
            table[token_id] = len(cleaned.encode("utf-8")) if cleaned else 0
        return table

    raise ValueError(f"Unsupported tokenizer format: {tokenizer_path}")


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-(probs * np.log2(probs)).sum())


def gini_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    probs = np.sort(counts.astype(np.float64) / total)
    n = probs.size
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (index * probs).sum() / n) - (n + 1) / n)


def js_divergence_bits(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    p = p_counts.astype(np.float64)
    q = q_counts.astype(np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    kl_pm = float((p[p_mask] * (np.log2(p[p_mask]) - np.log2(m[p_mask]))).sum())
    kl_qm = float((q[q_mask] * (np.log2(q[q_mask]) - np.log2(m[q_mask]))).sum())
    return 0.5 * (kl_pm + kl_qm)


def top_tokens(counts: np.ndarray, k: int) -> list[dict[str, int]]:
    if counts.size == 0:
        return []
    top_idx = np.argsort(counts)[::-1][:k]
    return [
        {"token_id": int(idx), "count": int(counts[idx])}
        for idx in top_idx
        if int(counts[idx]) > 0
    ]


@dataclass
class LagMetric:
    lag: int
    match_rate: float
    mutual_information_bits: float
    normalized_mi: float
    conditional_entropy_bits: float
    support_pairs: int


@dataclass
class RunLogSummary:
    path: str
    train_seq_len: int | None
    eval_seq_len: int | None
    train_batch_tokens: int | None
    model_params: int | None
    steps_completed: int
    final_step_avg_ms: float | None
    best_val_bpb: float | None
    final_val_bpb: float | None
    best_val_step: int | None
    final_val_step: int | None
    marginal_bpb_gain_last_1000: float | None
    stop_recommendation: str
    train_time_ms: float | None


@dataclass
class MissingRunLog:
    path: str
    reason: str


def lag_metrics(tokens: np.ndarray, vocab_size: int, lags: list[int]) -> list[LagMetric]:
    counts = np.bincount(tokens, minlength=vocab_size).astype(np.int64)
    entropy = entropy_from_counts(counts)
    metrics: list[LagMetric] = []
    for lag in lags:
        if lag <= 0 or lag >= tokens.size:
            continue
        left = tokens[:-lag]
        right = tokens[lag:]
        pair_ids = left.astype(np.int64) * vocab_size + right.astype(np.int64)
        joint = np.bincount(pair_ids, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size).astype(np.float64)
        total = joint.sum()
        if total <= 0:
            continue
        joint /= total
        py = joint.sum(axis=0)
        px_lag = joint.sum(axis=1)
        nz = joint > 0
        px_py = np.outer(px_lag, py)
        mi = float((joint[nz] * (np.log2(joint[nz]) - np.log2(px_py[nz]))).sum())
        cond_entropy = max(entropy - mi, 0.0)
        normalized = mi / entropy if entropy > 0 else 0.0
        match_rate = float(np.mean(left == right))
        metrics.append(
            LagMetric(
                lag=lag,
                match_rate=match_rate,
                mutual_information_bits=mi,
                normalized_mi=normalized,
                conditional_entropy_bits=cond_entropy,
                support_pairs=int(np.count_nonzero(joint)),
            )
        )
    return metrics


def context_reuse_profile(tokens: np.ndarray, windows: list[int]) -> list[dict[str, float]]:
    profiles: list[dict[str, float]] = []
    for window in windows:
        if window <= 0:
            continue
        seen = Counter()
        q: deque[int] = deque()
        reuse_hits = 0
        valid = 0
        for token in map(int, tokens):
            if q:
                valid += 1
                if seen[token] > 0:
                    reuse_hits += 1
            q.append(token)
            seen[token] += 1
            if len(q) > window:
                old = q.popleft()
                seen[old] -= 1
                if seen[old] <= 0:
                    del seen[old]
        profiles.append(
            {
                "window": int(window),
                "reuse_rate": float(reuse_hits / max(valid, 1)),
                "novelty_rate": float(1.0 - (reuse_hits / max(valid, 1))),
            }
        )
    return profiles


def top_bigrams(tokens: np.ndarray, vocab_size: int, k: int) -> list[dict[str, int]]:
    if tokens.size < 2:
        return []
    pair_ids = tokens[:-1].astype(np.int64) * vocab_size + tokens[1:].astype(np.int64)
    counts = np.bincount(pair_ids, minlength=vocab_size * vocab_size)
    idx = np.argsort(counts)[::-1][:k]
    out: list[dict[str, int]] = []
    for flat in idx:
        count = int(counts[flat])
        if count <= 0:
            continue
        out.append(
            {
                "left": int(flat // vocab_size),
                "right": int(flat % vocab_size),
                "count": count,
            }
        )
    return out


def shard_summaries(
    files: list[Path],
    vocab_size: int,
    sample_tokens_per_shard: int,
    base_bytes: np.ndarray | None,
) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    summaries: list[dict[str, float | int | str]] = []
    sampled_counts: list[np.ndarray] = []
    for file in files:
        tokens = sample_tokens_from_shard(file, sample_tokens_per_shard)
        counts = np.bincount(tokens, minlength=vocab_size).astype(np.int64)
        sampled_counts.append(counts)
        bytes_per_token = None if base_bytes is None else float(base_bytes[tokens].mean())
        summaries.append(
            {
                "file": str(file),
                "sample_tokens": int(tokens.size),
                "entropy_bits": entropy_from_counts(counts),
                "effective_vocab": float(2.0**entropy_from_counts(counts)),
                "gini": gini_from_counts(counts),
                "bytes_per_token": bytes_per_token,
            }
        )

    global_counts = np.sum(sampled_counts, axis=0) if sampled_counts else np.zeros((vocab_size,), dtype=np.int64)
    js_values = [js_divergence_bits(counts, global_counts) for counts in sampled_counts] if sampled_counts else [0.0]
    entropy_values = [float(item["entropy_bits"]) for item in summaries] if summaries else [0.0]
    bytes_values = [float(item["bytes_per_token"]) for item in summaries if item["bytes_per_token"] is not None]
    aggregate = {
        "num_shards_profiled": len(summaries),
        "sample_tokens_per_shard": int(sample_tokens_per_shard),
        "mean_entropy_bits": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "std_entropy_bits": float(np.std(entropy_values)) if entropy_values else 0.0,
        "mean_js_divergence_bits": float(np.mean(js_values)) if js_values else 0.0,
        "max_js_divergence_bits": float(np.max(js_values)) if js_values else 0.0,
        "mean_bytes_per_token": float(np.mean(bytes_values)) if bytes_values else 0.0,
        "std_bytes_per_token": float(np.std(bytes_values)) if bytes_values else 0.0,
    }
    for item, js in zip(summaries, js_values, strict=True):
        item["js_divergence_to_global_bits"] = float(js)
    return summaries, aggregate


def eval_length_candidates(tokens: np.ndarray, lengths: list[int]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for length in lengths:
        if length <= 0:
            continue
        if tokens.size <= length:
            continue
        x = tokens[:-length]
        y = tokens[length:]
        match_rate = float(np.mean(x == y))
        seen = Counter()
        q: deque[int] = deque()
        reuse_hits = 0
        valid = 0
        for token in map(int, tokens):
            if q:
                valid += 1
                if seen[token] > 0:
                    reuse_hits += 1
            q.append(token)
            seen[token] += 1
            if len(q) > length:
                old = q.popleft()
                seen[old] -= 1
                if seen[old] <= 0:
                    del seen[old]
        out.append(
            {
                "seq_len": int(length),
                "lag_match_rate": match_rate,
                "context_reuse_rate": float(reuse_hits / max(valid, 1)),
            }
        )
    return out


def marginal_eval_context_gain(eval_candidates: list[dict[str, float]]) -> list[dict[str, float | int]]:
    ordered = sorted(eval_candidates, key=lambda item: int(item["seq_len"]))
    out: list[dict[str, float | int]] = []
    prev: dict[str, float] | None = None
    for item in ordered:
        seq_len = int(item["seq_len"])
        reuse = float(item["context_reuse_rate"])
        match_rate = float(item["lag_match_rate"])
        if prev is None:
            out.append(
                {
                    "seq_len": seq_len,
                    "delta_from_prev_seq_len": 0,
                    "delta_reuse_rate": 0.0,
                    "delta_match_rate": 0.0,
                    "reuse_gain_per_256_tokens": 0.0,
                }
            )
        else:
            delta_seq = max(seq_len - int(prev["seq_len"]), 1)
            delta_reuse = reuse - float(prev["context_reuse_rate"])
            delta_match = match_rate - float(prev["lag_match_rate"])
            out.append(
                {
                    "seq_len": seq_len,
                    "delta_from_prev_seq_len": int(delta_seq),
                    "delta_reuse_rate": float(delta_reuse),
                    "delta_match_rate": float(delta_match),
                    "reuse_gain_per_256_tokens": float(delta_reuse * (256.0 / delta_seq)),
                }
            )
        prev = item
    return out


def context_gain_by_distance(lag_stats: list[LagMetric], bands: list[tuple[int, int]]) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for start, end in bands:
        band = [m for m in lag_stats if start <= m.lag <= end]
        if not band:
            continue
        out.append(
            {
                "lag_start": int(start),
                "lag_end": int(end),
                "count": int(len(band)),
                "avg_normalized_mi": float(np.mean([m.normalized_mi for m in band])),
                "max_normalized_mi": float(np.max([m.normalized_mi for m in band])),
                "avg_match_rate": float(np.mean([m.match_rate for m in band])),
            }
        )
    return out


def recurrence_burst_profile_by_frequency(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    max_gap: int = 2048,
) -> list[dict[str, object]]:
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return []
    q50 = float(np.quantile(nonzero, 0.5))
    q90 = float(np.quantile(nonzero, 0.9))
    bucket_ids = {
        "rare": np.flatnonzero((counts > 0) & (counts <= q50)),
        "mid": np.flatnonzero((counts > q50) & (counts <= q90)),
        "common": np.flatnonzero(counts > q90),
    }
    bucket_name_by_token = np.full((vocab_size,), fill_value=-1, dtype=np.int8)
    bucket_order = ["rare", "mid", "common"]
    for idx, name in enumerate(bucket_order):
        ids = bucket_ids[name]
        if ids.size > 0:
            bucket_name_by_token[ids] = idx

    last_seen_by_bucket: list[dict[int, int]] = [dict() for _ in bucket_order]
    gaps_by_bucket: list[list[int]] = [[] for _ in bucket_order]
    for pos, token in enumerate(map(int, tokens)):
        bucket_idx = int(bucket_name_by_token[token]) if 0 <= token < vocab_size else -1
        if bucket_idx < 0:
            continue
        last_seen = last_seen_by_bucket[bucket_idx]
        prev = last_seen.get(token)
        if prev is not None:
            gap = pos - prev
            if 1 <= gap <= max_gap:
                gaps_by_bucket[bucket_idx].append(gap)
        last_seen[token] = pos

    out: list[dict[str, object]] = []
    for bucket_idx, name in enumerate(bucket_order):
        gaps = gaps_by_bucket[bucket_idx]
        token_ids = bucket_ids[name]
        if not gaps:
            out.append(
                {
                    "bucket": name,
                    "num_tokens_in_bucket": int(token_ids.size),
                    "num_recurrences": 0,
                    "mean_gap": None,
                    "median_gap": None,
                    "share_gap_le_32": 0.0,
                    "share_gap_le_128": 0.0,
                    "share_gap_le_512": 0.0,
                }
            )
            continue
        gaps_np = np.asarray(gaps, dtype=np.int32)
        out.append(
            {
                "bucket": name,
                "num_tokens_in_bucket": int(token_ids.size),
                "num_recurrences": int(gaps_np.size),
                "mean_gap": float(gaps_np.mean()),
                "median_gap": float(np.median(gaps_np)),
                "share_gap_le_32": float(np.mean(gaps_np <= 32)),
                "share_gap_le_128": float(np.mean(gaps_np <= 128)),
                "share_gap_le_512": float(np.mean(gaps_np <= 512)),
            }
        )
    return out


def transition_geometry_profile(tokens: np.ndarray, vocab_size: int, counts: np.ndarray, top_k: int) -> dict[str, object]:
    if tokens.size < 2:
        return {
            "available": False,
            "reason": "not_enough_tokens",
        }
    pair_ids = tokens[:-1].astype(np.int64) * vocab_size + tokens[1:].astype(np.int64)
    pair_counts = np.bincount(pair_ids, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size).astype(np.int64)
    active_rows = np.flatnonzero(pair_counts.sum(axis=1) > 0)
    row_branching = np.count_nonzero(pair_counts > 0, axis=1)[active_rows]
    row_entropies: list[float] = []
    for row in active_rows:
        row_counts = pair_counts[row]
        total = int(row_counts.sum())
        if total <= 0:
            continue
        probs = row_counts[row_counts > 0].astype(np.float64) / total
        row_entropies.append(float(-(probs * np.log2(probs)).sum()))

    top_source_ids = np.argsort(counts)[::-1][: max(top_k, 1)]
    top_source_rows = []
    for token_id in top_source_ids:
        row = pair_counts[int(token_id)]
        total = int(row.sum())
        if total <= 0:
            continue
        next_ids = np.argsort(row)[::-1][: min(top_k, vocab_size)]
        top_source_rows.append(
            {
                "token_id": int(token_id),
                "source_count": int(counts[int(token_id)]),
                "out_degree": int(np.count_nonzero(row)),
                "top_next_tokens": [
                    {"token_id": int(next_id), "count": int(row[int(next_id)])}
                    for next_id in next_ids
                    if int(row[int(next_id)]) > 0
                ],
            }
        )

    return {
        "available": True,
        "active_source_tokens": int(active_rows.size),
        "mean_out_degree": float(np.mean(row_branching)) if row_branching.size else 0.0,
        "median_out_degree": float(np.median(row_branching)) if row_branching.size else 0.0,
        "mean_row_entropy_bits": float(np.mean(row_entropies)) if row_entropies else 0.0,
        "median_row_entropy_bits": float(np.median(row_entropies)) if row_entropies else 0.0,
        "top_source_rows": top_source_rows,
    }


def parse_run_log(path: str) -> RunLogSummary:
    text = Path(path).read_text(encoding="utf-8")
    train_seq_len = None
    eval_seq_len = None
    train_batch_tokens = None
    model_params = None
    steps_completed = 0
    final_step_avg_ms = None
    best_val_bpb = None
    final_val_bpb = None
    best_val_step = None
    final_val_step = None
    val_history: list[tuple[int, float]] = []

    for line in text.splitlines():
        if train_seq_len is None:
            m = re.search(r"train_batch_tokens:(\d+)\s+train_seq_len:(\d+)", line)
            if m:
                train_batch_tokens = int(m.group(1))
                train_seq_len = int(m.group(2))
        if eval_seq_len is None:
            m = re.search(r"eval_mode:.*eval_seq_len:(\d+)", line)
            if m:
                eval_seq_len = int(m.group(1))
        if eval_seq_len is None:
            m = re.search(r"final_eval_tokens:\d+\s+eval_seq_len:(\d+)", line)
            if m:
                eval_seq_len = int(m.group(1))
        if model_params is None:
            m = re.search(r"model_params:(\d+)", line)
            if m:
                model_params = int(m.group(1))
        m = re.search(r"step:(\d+)/\d+.*step_avg:([0-9.]+)ms", line)
        if m:
            steps_completed = max(steps_completed, int(m.group(1)))
            final_step_avg_ms = float(m.group(2))
        m = re.search(r"step:(\d+)/\d+.*val_bpb:([0-9.]+)", line)
        if m:
            step = int(m.group(1))
            val_bpb = float(m.group(2))
            val_history.append((step, val_bpb))
            final_val_bpb = val_bpb
            final_val_step = step
            if best_val_bpb is None or val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                best_val_step = step
        m = re.search(r"stopping_early:.*step:(\d+)/\d+", line)
        if m:
            steps_completed = max(steps_completed, int(m.group(1)))

    marginal_bpb_gain_last_1000 = None
    stop_recommendation = "insufficient_validation_history"
    train_time_ms = None
    if len(val_history) >= 2:
        tail_step, tail_bpb = val_history[-1]
        prev_candidates = [item for item in val_history[:-1] if tail_step - item[0] >= 1000]
        if prev_candidates:
            prev_step, prev_bpb = prev_candidates[-1]
            marginal_bpb_gain_last_1000 = prev_bpb - tail_bpb
        else:
            prev_step, prev_bpb = val_history[-2]
            span = max(tail_step - prev_step, 1)
            marginal_bpb_gain_last_1000 = (prev_bpb - tail_bpb) * (1000.0 / span)
        if marginal_bpb_gain_last_1000 < 0.002:
            stop_recommendation = "stop_early_candidate"
        elif marginal_bpb_gain_last_1000 < 0.006:
            stop_recommendation = "close_monitoring"
        else:
            stop_recommendation = "keep_training"
    for line in text.splitlines():
        m = re.search(r"train_time:(\d+)ms", line)
        if m:
            train_time_ms = float(m.group(1))

    return RunLogSummary(
        path=path,
        train_seq_len=train_seq_len,
        eval_seq_len=eval_seq_len,
        train_batch_tokens=train_batch_tokens,
        model_params=model_params,
        steps_completed=steps_completed,
        final_step_avg_ms=final_step_avg_ms,
        best_val_bpb=best_val_bpb,
        final_val_bpb=final_val_bpb,
        best_val_step=best_val_step,
        final_val_step=final_val_step,
        marginal_bpb_gain_last_1000=marginal_bpb_gain_last_1000,
        stop_recommendation=stop_recommendation,
        train_time_ms=train_time_ms,
    )


def load_run_logs(paths: list[str]) -> tuple[list[RunLogSummary], list[MissingRunLog]]:
    runs: list[RunLogSummary] = []
    missing: list[MissingRunLog] = []
    for path in paths:
        p = path.strip()
        if not p:
            continue
        file_path = Path(p)
        if not file_path.exists():
            missing.append(MissingRunLog(path=p, reason="file_not_found"))
            continue
        try:
            runs.append(parse_run_log(p))
        except Exception as exc:
            missing.append(MissingRunLog(path=p, reason=f"parse_error:{type(exc).__name__}"))
    return runs, missing


def parse_candidate_models(text: str, default_vocab_size: int, default_num_heads: int, default_num_kv_heads: int) -> list[dict[str, int | str]]:
    candidates: list[dict[str, int | str]] = []
    if not text.strip():
        return candidates
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) < 4:
            raise ValueError(
                "Expected candidate model spec name:num_layers:model_dim:mlp_hidden[:num_heads:num_kv_heads[:vocab_size]], "
                f"got: {item}"
            )
        candidates.append(
            {
                "name": fields[0],
                "num_layers": int(fields[1]),
                "model_dim": int(fields[2]),
                "mlp_hidden": int(fields[3]),
                "num_heads": int(fields[4]) if len(fields) >= 5 else default_num_heads,
                "num_kv_heads": int(fields[5]) if len(fields) >= 6 else default_num_kv_heads,
                "vocab_size": int(fields[6]) if len(fields) >= 7 else default_vocab_size,
            }
        )
    return candidates


def estimate_transformer_param_count(
    vocab_size: int,
    num_layers: int,
    model_dim: int,
    mlp_hidden: int,
    num_kv_heads: int,
    num_heads: int,
    tie_embeddings: bool = True,
) -> int:
    head_dim = model_dim // max(num_heads, 1)
    kv_dim = num_kv_heads * head_dim
    embed_params = vocab_size * model_dim
    lm_head_params = 0 if tie_embeddings else vocab_size * model_dim
    block_params = num_layers * (
        (model_dim * model_dim) + 2 * (model_dim * kv_dim) + (model_dim * model_dim) + 2 * (model_dim * mlp_hidden)
    )
    control_params = num_layers * (9 * model_dim + num_heads)
    skip_params = (num_layers // 2) * model_dim
    final_params = model_dim
    return int(embed_params + lm_head_params + block_params + control_params + skip_params + final_params)


def estimate_model_bytes_q6(param_count: int, fp16_fraction: float = 0.12) -> int:
    q6_bytes = param_count * 0.75
    fp16_bytes = param_count * fp16_fraction * 2.0
    return int(q6_bytes + fp16_bytes)


def budgeted_model_candidates(
    candidates: list[dict[str, int | str]],
    anchor_run: RunLogSummary | None,
    max_wallclock_seconds: float,
    total_train_tokens_available: int,
    current_model_params: int | None,
    current_best_val_bpb: float | None,
) -> list[dict[str, object]]:
    if not candidates:
        return []
    if anchor_run is None or anchor_run.final_step_avg_ms is None or current_model_params is None or current_model_params <= 0:
        return [
            {
                "name": str(cand["name"]),
                "param_count_estimate": estimate_transformer_param_count(
                    vocab_size=int(cand["vocab_size"]),
                    num_layers=int(cand["num_layers"]),
                    model_dim=int(cand["model_dim"]),
                    mlp_hidden=int(cand["mlp_hidden"]),
                    num_kv_heads=int(cand["num_kv_heads"]),
                    num_heads=int(cand["num_heads"]),
                ),
                "available": False,
                "reason": "anchor_run_missing",
            }
            for cand in candidates
        ]

    rows: list[dict[str, object]] = []
    anchor_params = float(current_model_params)
    anchor_step_ms = float(anchor_run.final_step_avg_ms)
    anchor_steps = max(anchor_run.steps_completed, 1)
    batch_tokens = int(anchor_run.train_batch_tokens or 0)
    max_wallclock_ms = max_wallclock_seconds * 1000.0
    for cand in candidates:
        param_count = estimate_transformer_param_count(
            vocab_size=int(cand["vocab_size"]),
            num_layers=int(cand["num_layers"]),
            model_dim=int(cand["model_dim"]),
            mlp_hidden=int(cand["mlp_hidden"]),
            num_kv_heads=int(cand["num_kv_heads"]),
            num_heads=int(cand["num_heads"]),
        )
        rel_scale = param_count / max(anchor_params, 1.0)
        projected_step_ms = anchor_step_ms * (rel_scale ** 1.08)
        est_steps = int(max_wallclock_ms // projected_step_ms)
        est_train_tokens = est_steps * batch_tokens
        coverage = min(est_train_tokens / max(total_train_tokens_available, 1), 1.0)
        bytes_est = estimate_model_bytes_q6(param_count)
        artifact_ratio = bytes_est / 16_000_000.0
        capacity_gain = math.log(max(rel_scale, 1e-9), 2.0)
        projected_bpb = None
        if current_best_val_bpb is not None:
            bpb_delta_proxy = (
                -0.012 * capacity_gain
                + 0.000015 * max(anchor_steps - est_steps, 0)
                + 0.03 * max(artifact_ratio - 1.0, 0.0)
            )
            projected_bpb = current_best_val_bpb + bpb_delta_proxy
        rows.append(
            {
                "name": str(cand["name"]),
                "num_layers": int(cand["num_layers"]),
                "model_dim": int(cand["model_dim"]),
                "mlp_hidden": int(cand["mlp_hidden"]),
                "param_count_estimate": int(param_count),
                "relative_param_scale": float(rel_scale),
                "projected_step_ms": float(projected_step_ms),
                "estimated_steps_in_budget": int(est_steps),
                "estimated_train_tokens": int(est_train_tokens),
                "estimated_train_stream_coverage": float(coverage),
                "artifact_bytes_q6_proxy": int(bytes_est),
                "artifact_limit_ratio": float(artifact_ratio),
                "projected_bpb_proxy": projected_bpb,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["projected_bpb_proxy"]) if row["projected_bpb_proxy"] is not None else float("inf"),
            -int(row["estimated_steps_in_budget"]),
        ),
    )


def observed_runtime_frontier(runs: list[RunLogSummary]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[RunLogSummary]] = {}
    for run in runs:
        if run.train_seq_len is None or run.train_batch_tokens is None or run.final_step_avg_ms is None:
            continue
        grouped.setdefault((int(run.train_seq_len), int(run.train_batch_tokens)), []).append(run)

    rows: list[dict[str, object]] = []
    for (train_seq_len, train_batch_tokens), group in grouped.items():
        fastest = min(group, key=lambda item: float(item.final_step_avg_ms or float("inf")))
        best_bpb_run = min(
            [item for item in group if item.best_val_bpb is not None],
            key=lambda item: float(item.best_val_bpb),
            default=None,
        )
        rows.append(
            {
                "train_seq_len": int(train_seq_len),
                "train_batch_tokens": int(train_batch_tokens),
                "num_runs": int(len(group)),
                "fastest_step_avg_ms": float(fastest.final_step_avg_ms),
                "fastest_run_path": fastest.path,
                "fastest_eval_seq_len": fastest.eval_seq_len,
                "best_val_bpb": None if best_bpb_run is None else float(best_bpb_run.best_val_bpb),
                "best_bpb_run_path": None if best_bpb_run is None else best_bpb_run.path,
                "best_bpb_eval_seq_len": None if best_bpb_run is None else best_bpb_run.eval_seq_len,
                "max_steps_completed": int(max(item.steps_completed for item in group)),
            }
        )
    return sorted(rows, key=lambda row: (float(row["fastest_step_avg_ms"]), -int(row["train_seq_len"])))


def infer_component_overheads_from_runs(runs: list[RunLogSummary]) -> dict[str, float]:
    candidates = [run for run in runs if run.train_seq_len is not None and run.train_batch_tokens is not None and run.final_step_avg_ms is not None]
    if not candidates:
        return {}

    by_key: dict[tuple[int, int], list[RunLogSummary]] = {}
    for run in candidates:
        by_key.setdefault((int(run.train_seq_len), int(run.train_batch_tokens)), []).append(run)

    def best_step_ms(seq_len: int, batch_tokens: int) -> float | None:
        group = by_key.get((seq_len, batch_tokens), [])
        if not group:
            return None
        return min(float(item.final_step_avg_ms) for item in group if item.final_step_avg_ms is not None)

    observed: dict[str, float] = {}
    candidate_batches = sorted({int(run.train_batch_tokens) for run in candidates})
    overheads_2048: list[float] = []
    for batch_tokens in candidate_batches:
        base_1024 = best_step_ms(1024, batch_tokens)
        base_2048 = best_step_ms(2048, batch_tokens)
        if base_1024 is None or base_2048 is None or base_1024 <= 0.0:
            continue
        overheads_2048.append(max(base_2048 / base_1024 - 1.0, 0.0))
    if overheads_2048:
        observed["train_seq_len_2048"] = float(min(overheads_2048))
    return observed


def estimate_real_eval_budget_frontier(
    total_val_tokens: int,
    eval_lengths: list[int],
    eval_stride: int,
    eval_batch_seqs: int,
    max_wallclock_seconds: float,
    base_eval_seq_len: int,
    base_eval_batch_seqs: int,
    base_eval_step_ms: float,
    ttt_multipliers: list[float],
) -> list[dict[str, object]]:
    if total_val_tokens <= 1:
        return []
    rows: list[dict[str, object]] = []
    base_work = max(base_eval_seq_len * base_eval_batch_seqs, 1)
    max_wallclock_ms = max_wallclock_seconds * 1000.0
    for seq_len in eval_lengths:
        if seq_len <= 0 or total_val_tokens <= seq_len:
            continue
        stride = eval_stride if eval_stride > 0 else seq_len
        num_windows = 1 + max((total_val_tokens - 1 - seq_len) // max(stride, 1), 0)
        num_batches = math.ceil(num_windows / max(eval_batch_seqs, 1))
        rel_cost = (seq_len * eval_batch_seqs) / base_work
        per_batch_ms = base_eval_step_ms * rel_cost
        total_eval_ms = per_batch_ms * num_batches
        row = {
            "seq_len": int(seq_len),
            "stride": int(stride),
            "eval_batch_seqs": int(eval_batch_seqs),
            "num_windows": int(num_windows),
            "num_batches": int(num_batches),
            "estimated_per_batch_ms": float(per_batch_ms),
            "estimated_total_eval_ms": float(total_eval_ms),
            "fits_in_budget": bool(total_eval_ms <= max_wallclock_ms),
            "ttt_variants": [],
        }
        for mult in ttt_multipliers:
            total_ttt_ms = total_eval_ms * mult
            row["ttt_variants"].append(
                {
                    "multiplier": float(mult),
                    "estimated_total_eval_ms": float(total_ttt_ms),
                    "fits_in_budget": bool(total_ttt_ms <= max_wallclock_ms),
                }
            )
        rows.append(row)
    return rows


def compare_runs_equal_wallclock(runs: list[RunLogSummary], wallclock_points_seconds: list[int]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for run in runs:
        point_rows = []
        for sec in wallclock_points_seconds:
            if run.final_step_avg_ms is None or run.final_val_step is None or run.final_val_bpb is None:
                point_rows.append({"seconds": int(sec), "estimated_step": None, "projected_val_bpb": None})
                continue
            est_step = int((sec * 1000.0) // max(run.final_step_avg_ms, 1e-9))
            if run.best_val_step is not None and est_step >= run.best_val_step and run.best_val_bpb is not None:
                projected_bpb = run.best_val_bpb
            elif run.final_val_step > 0:
                frac = min(est_step / max(run.final_val_step, 1), 1.0)
                projected_bpb = run.final_val_bpb + (1.0 - frac) * 0.02
            else:
                projected_bpb = run.final_val_bpb
            point_rows.append(
                {
                    "seconds": int(sec),
                    "estimated_step": int(est_step),
                    "projected_val_bpb": None if projected_bpb is None else float(projected_bpb),
                }
            )
        out.append(
            {
                "path": run.path,
                "best_val_bpb": run.best_val_bpb,
                "final_val_bpb": run.final_val_bpb,
                "final_step_avg_ms": run.final_step_avg_ms,
                "points": point_rows,
            }
        )
    return out


def recipe_frontier(
    focused_analysis: list[dict[str, object]],
    real_eval_frontier: list[dict[str, object]],
    training_budget: dict[str, object],
    training_coverage: dict[str, object],
) -> list[dict[str, object]]:
    focused_by_name = {str(item["name"]): item for item in focused_analysis}
    eval_1408 = next((row for row in real_eval_frontier if int(row["seq_len"]) == 1408), None)
    eval_2048 = next((row for row in real_eval_frontier if int(row["seq_len"]) == 2048), None)
    coverage_min = float(training_coverage.get("coverage_fraction_min", 0.0))
    base_steps = int(training_budget.get("estimated_steps", 0))
    candidates = [
        ("smear_only_fast", ["smear_gate"], eval_1408),
        ("smear_eval2048", ["smear_gate", "eval_seq_len_2048"], eval_2048),
        ("peak_ttt_10l", ["smear_gate", "eval_seq_len_2048", "ttt_lora"], eval_2048),
        ("bigram1024_eval2048", ["bigram_1024", "eval_seq_len_2048"], eval_2048),
        ("train2048_heavy", ["train_seq_len_2048"], eval_2048),
    ]
    rows: list[dict[str, object]] = []
    for name, component_names, eval_row in candidates:
        score = 0.0
        lost_steps = 0
        for component_name in component_names:
            item = focused_by_name.get(component_name)
            if item is None:
                continue
            score += float(item["priority_score"])
            budget = item.get("budget_sensitivity", {})
            if isinstance(budget, dict) and budget.get("lost_steps_max") is not None:
                lost_steps += int(budget["lost_steps_max"])
        eval_total_ms = float(eval_row["estimated_total_eval_ms"]) if eval_row is not None else None
        fits_eval = bool(eval_row["fits_in_budget"]) if eval_row is not None else None
        rows.append(
            {
                "name": name,
                "components": component_names,
                "composite_priority_score": float(score),
                "estimated_train_steps": max(base_steps - lost_steps, 0),
                "estimated_train_coverage_floor": float(max(coverage_min - (lost_steps / max(base_steps, 1)) * coverage_min, 0.0)),
                "estimated_total_eval_ms": eval_total_ms,
                "eval_fits_budget": fits_eval,
            }
        )
    return sorted(rows, key=lambda row: float(row["composite_priority_score"]), reverse=True)


def leaderboard_consistent_analysis(
    lag_stats: list[LagMetric],
    reuse_stats: list[dict[str, float]],
    eval_candidates: list[dict[str, float]],
    focused_analysis: list[dict[str, object]],
) -> list[dict[str, object]]:
    focused_by_name = {str(item["name"]): item for item in focused_analysis}
    reuse_128 = _lookup_reuse(reuse_stats, 128) or 0.0
    reuse_2048 = _lookup_eval_reuse(eval_candidates, 2048) or _lookup_reuse(reuse_stats, 2048) or 0.0
    long_nmi = max((m.normalized_mi for m in lag_stats if m.lag >= 512), default=0.0)

    candidates = [
        {
            "name": "mixed_quant_export",
            "leaderboard_prior": 0.95,
            "dataset_fit": 0.25,
            "rationale": "Current top runs are dominated by mixed int5/int6 or selective mixed-precision export policies.",
            "depends_on": ["fake_quant_tail"],
        },
        {
            "name": "bigram_hash_large",
            "leaderboard_prior": 0.9,
            "dataset_fit": 0.55 * reuse_128,
            "rationale": "Current winning runs repeatedly use large BigramHash tables, consistent with strong local reuse.",
            "depends_on": ["bigram_4096", "smear_gate"],
        },
        {
            "name": "swa_late",
            "leaderboard_prior": 0.85,
            "dataset_fit": 0.15,
            "rationale": "Late-only SWA appears across top runs and is especially plausible when export quantization is the scored metric.",
            "depends_on": [],
        },
        {
            "name": "fake_quant_tail",
            "leaderboard_prior": 0.8,
            "dataset_fit": 0.2,
            "rationale": "Compression-aware tail finetuning is explicitly aligned with the scored post-quant artifact.",
            "depends_on": ["fake_quant_tail"],
        },
        {
            "name": "train_seq_len_2048",
            "leaderboard_prior": 0.75,
            "dataset_fit": max(reuse_2048 - (_lookup_eval_reuse(eval_candidates, 1024) or 0.0), 0.0) + 0.5 * long_nmi,
            "rationale": "The leaderboard now shows true-2048 lanes can win when the kernel is fast enough.",
            "depends_on": ["train_seq_len_2048"],
        },
        {
            "name": "ttt_lora",
            "leaderboard_prior": 0.15,
            "dataset_fit": 0.5 * reuse_2048 + 0.5 * long_nmi,
            "rationale": "TTT had profiler support, but leaderboard evidence is weak relative to lexical and compression-aware lanes.",
            "depends_on": ["ttt_lora"],
        },
    ]

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        focused_support = 0.0
        focused_budget = 0.0
        for dep in candidate["depends_on"]:
            item = focused_by_name.get(dep)
            if item is None:
                continue
            focused_support += max(float(item["priority_score"]), 0.0) / 1000.0
            budget = item.get("budget_sensitivity", {})
            if isinstance(budget, dict) and budget.get("lost_steps_max") is not None:
                focused_budget += float(budget["lost_steps_max"]) / 1000.0
        composite = 0.65 * float(candidate["leaderboard_prior"]) + 0.35 * (float(candidate["dataset_fit"]) + focused_support) - 0.1 * focused_budget
        rows.append(
            {
                "name": str(candidate["name"]),
                "leaderboard_prior": float(candidate["leaderboard_prior"]),
                "dataset_fit": float(candidate["dataset_fit"]),
                "focused_support": float(focused_support),
                "budget_penalty_proxy": float(focused_budget),
                "composite_score": float(composite),
                "depends_on": list(candidate["depends_on"]),
                "rationale": str(candidate["rationale"]),
            }
        )
    return sorted(rows, key=lambda row: float(row["composite_score"]), reverse=True)


def recurrence_burst_profile(tokens: np.ndarray, max_gap: int = 2048, top_k: int = 16) -> dict[str, object]:
    last_seen: dict[int, int] = {}
    gaps: list[int] = []
    for idx, token in enumerate(map(int, tokens)):
        prev = last_seen.get(token)
        if prev is not None:
            gap = idx - prev
            if 1 <= gap <= max_gap:
                gaps.append(gap)
        last_seen[token] = idx
    if not gaps:
        return {
            "max_gap_profiled": int(max_gap),
            "num_recurrences": 0,
            "mean_gap": None,
            "median_gap": None,
            "share_gap_le_8": 0.0,
            "share_gap_le_32": 0.0,
            "share_gap_le_128": 0.0,
            "share_gap_le_512": 0.0,
            "top_gap_counts": [],
        }
    gaps_np = np.asarray(gaps, dtype=np.int32)
    counts = Counter(gaps)
    return {
        "max_gap_profiled": int(max_gap),
        "num_recurrences": int(gaps_np.size),
        "mean_gap": float(gaps_np.mean()),
        "median_gap": float(np.median(gaps_np)),
        "share_gap_le_8": float(np.mean(gaps_np <= 8)),
        "share_gap_le_32": float(np.mean(gaps_np <= 32)),
        "share_gap_le_128": float(np.mean(gaps_np <= 128)),
        "share_gap_le_512": float(np.mean(gaps_np <= 512)),
        "top_gap_counts": [{"gap": int(g), "count": int(c)} for g, c in counts.most_common(top_k)],
    }


def estimate_eval_budget_frontier(
    eval_lengths: list[int],
    eval_batch_seqs: int,
    max_wallclock_seconds: float,
    base_eval_seq_len: int,
    base_eval_step_ms: float,
    ttt_multipliers: list[float],
) -> list[dict[str, object]]:
    if base_eval_step_ms <= 0:
        raise ValueError(f"base_eval_step_ms must be positive, got {base_eval_step_ms}")
    out: list[dict[str, object]] = []
    base_work = max(base_eval_seq_len * eval_batch_seqs, 1)
    for seq_len in eval_lengths:
        if seq_len <= 0:
            continue
        rel_cost = (seq_len * eval_batch_seqs) / base_work
        eval_ms = base_eval_step_ms * rel_cost
        row = {
            "seq_len": int(seq_len),
            "eval_batch_seqs": int(eval_batch_seqs),
            "relative_cost_vs_base": float(rel_cost),
            "estimated_eval_forward_ms": float(eval_ms),
            "estimated_eval_forwards_in_budget": int((max_wallclock_seconds * 1000.0) // eval_ms),
            "ttt_variants": [],
        }
        for mult in ttt_multipliers:
            ttt_ms = eval_ms * mult
            row["ttt_variants"].append(
                {
                    "multiplier": float(mult),
                    "estimated_eval_forward_ms": float(ttt_ms),
                    "estimated_eval_forwards_in_budget": int((max_wallclock_seconds * 1000.0) // ttt_ms),
                }
            )
        out.append(row)
    return out


def estimate_training_budget(
    max_wallclock_seconds: float,
    avg_step_ms: float,
    train_batch_tokens: int,
    train_seq_len: int,
    world_size: int,
    grad_accum_steps: int,
) -> dict[str, float | int]:
    if avg_step_ms <= 0:
        raise ValueError(f"avg_step_ms must be positive, got {avg_step_ms}")
    max_wallclock_ms = max_wallclock_seconds * 1000.0
    estimated_steps = int(max_wallclock_ms // avg_step_ms)
    estimated_train_tokens = int(estimated_steps * train_batch_tokens)
    local_batch_tokens = train_batch_tokens // max(world_size * grad_accum_steps, 1)
    sequences_per_rank_per_micro = local_batch_tokens // max(train_seq_len, 1)
    return {
        "max_wallclock_seconds": float(max_wallclock_seconds),
        "avg_step_ms": float(avg_step_ms),
        "estimated_steps": estimated_steps,
        "estimated_train_tokens": estimated_train_tokens,
        "train_batch_tokens": int(train_batch_tokens),
        "train_seq_len": int(train_seq_len),
        "world_size": int(world_size),
        "grad_accum_steps": int(grad_accum_steps),
        "local_batch_tokens": int(local_batch_tokens),
        "sequences_per_rank_per_microstep": int(sequences_per_rank_per_micro),
    }


def estimate_training_budget_range(
    max_wallclock_seconds: float,
    avg_step_ms_values: list[float],
    train_batch_tokens: int,
    train_seq_len: int,
    world_size: int,
    grad_accum_steps: int,
) -> dict[str, object]:
    values = [float(v) for v in avg_step_ms_values if float(v) > 0]
    if not values:
        raise ValueError("avg_step_ms_values must contain at least one positive value")
    points = [
        estimate_training_budget(
            max_wallclock_seconds=max_wallclock_seconds,
            avg_step_ms=v,
            train_batch_tokens=train_batch_tokens,
            train_seq_len=train_seq_len,
            world_size=world_size,
            grad_accum_steps=grad_accum_steps,
        )
        for v in values
    ]
    steps = [int(point["estimated_steps"]) for point in points]
    tokens = [int(point["estimated_train_tokens"]) for point in points]
    return {
        "avg_step_ms_values": values,
        "estimated_steps_min": int(min(steps)),
        "estimated_steps_max": int(max(steps)),
        "estimated_train_tokens_min": int(min(tokens)),
        "estimated_train_tokens_max": int(max(tokens)),
        "points": points,
    }


def estimate_training_coverage(
    total_train_tokens_available: int,
    training_budget: dict[str, object],
) -> dict[str, object]:
    if total_train_tokens_available <= 0:
        raise ValueError("total_train_tokens_available must be positive")
    point_rows = []
    range_info = training_budget.get("range") if isinstance(training_budget, dict) else None
    source_points = range_info.get("points", []) if isinstance(range_info, dict) else [training_budget]
    for point in source_points:
        estimated_train_tokens = int(point["estimated_train_tokens"])
        wraps = estimated_train_tokens / total_train_tokens_available
        point_rows.append(
            {
                "avg_step_ms": float(point["avg_step_ms"]),
                "estimated_train_tokens": estimated_train_tokens,
                "coverage_fraction": float(min(estimated_train_tokens / total_train_tokens_available, 1.0)),
                "expected_wraps": float(wraps),
                "wraps_dataset": bool(estimated_train_tokens >= total_train_tokens_available),
            }
        )
    coverage_values = [float(row["coverage_fraction"]) for row in point_rows]
    wrap_values = [float(row["expected_wraps"]) for row in point_rows]
    return {
        "total_train_tokens_available": int(total_train_tokens_available),
        "coverage_fraction_min": float(min(coverage_values)),
        "coverage_fraction_max": float(max(coverage_values)),
        "expected_wraps_min": float(min(wrap_values)),
        "expected_wraps_max": float(max(wrap_values)),
        "points": point_rows,
    }


def loss_geometry_surface(
    lag_stats: list[LagMetric],
    reuse_stats: list[dict[str, float]],
    eval_candidates: list[dict[str, float]],
    recurrence_profile: dict[str, object],
    recurrence_by_bucket: list[dict[str, object]],
    transition_geometry: dict[str, object],
    training_budget: dict[str, object],
    training_coverage: dict[str, object],
    eval_budget_frontier: list[dict[str, object]],
) -> dict[str, object]:
    reuse_128 = _lookup_reuse(reuse_stats, 128) or 0.0
    reuse_512 = _lookup_reuse(reuse_stats, 512) or 0.0
    reuse_2048 = _lookup_reuse(reuse_stats, 2048) or 0.0
    long_nmi = max((m.normalized_mi for m in lag_stats if m.lag >= 512), default=0.0)
    short_nmi = max((m.normalized_mi for m in lag_stats if m.lag <= 32), default=0.0)
    medium_nmi = max((m.normalized_mi for m in lag_stats if 64 <= m.lag < 512), default=0.0)
    coverage_min = float(training_coverage.get("coverage_fraction_min", 0.0))
    coverage_max = float(training_coverage.get("coverage_fraction_max", 0.0))
    est_steps = int(training_budget.get("estimated_steps", 0))
    mean_gap = float(recurrence_profile["mean_gap"]) if recurrence_profile.get("mean_gap") is not None else 0.0
    share_gap_128 = float(recurrence_profile.get("share_gap_le_128", 0.0))
    common_bucket = next((row for row in recurrence_by_bucket if row["bucket"] == "common"), None)
    rare_bucket = next((row for row in recurrence_by_bucket if row["bucket"] == "rare"), None)
    common_short_recur = float(common_bucket["share_gap_le_128"]) if common_bucket else 0.0
    rare_short_recur = float(rare_bucket["share_gap_le_128"]) if rare_bucket else 0.0
    mean_out_degree = float(transition_geometry.get("mean_out_degree", 0.0))
    mean_row_entropy = float(transition_geometry.get("mean_row_entropy_bits", 0.0))

    best_eval = max(eval_candidates, key=lambda row: row["context_reuse_rate"]) if eval_candidates else {"seq_len": 0, "context_reuse_rate": 0.0}
    best_eval_frontier = max(eval_budget_frontier, key=lambda row: row["estimated_eval_forwards_in_budget"]) if eval_budget_frontier else None
    eval_2048 = next((row for row in eval_budget_frontier if int(row["seq_len"]) == 2048), None)
    eval_headroom_ratio = 0.0
    if best_eval_frontier and eval_2048:
        denom = max(float(best_eval_frontier["estimated_eval_forwards_in_budget"]), 1.0)
        eval_headroom_ratio = float(eval_2048["estimated_eval_forwards_in_budget"]) / denom

    axes = {
        "locality_axis": float(min(max(short_nmi + 0.75 * reuse_128, 0.0), 1.0)),
        "medium_context_axis": float(min(max(medium_nmi + 0.5 * reuse_512, 0.0), 1.0)),
        "long_context_axis": float(min(max(long_nmi + 0.5 * (reuse_2048 - reuse_512), 0.0), 1.0)),
        "recurrence_axis": float(min(max(0.5 * share_gap_128 + 0.5 * (1.0 / max(mean_gap / 128.0, 1.0)), 0.0), 1.0)),
        "lexical_shortcut_axis": float(min(max(0.6 * common_short_recur + 0.2 * max(1.0 - mean_row_entropy / 8.0, 0.0) + 0.2 * max(1.0 - mean_out_degree / 128.0, 0.0), 0.0), 1.0)),
        "adaptation_axis": float(min(max(0.5 * (1.0 - rare_short_recur) + 0.5 * (reuse_2048 - reuse_128), 0.0), 1.0)),
        "train_kernel_pressure": float(min(max(1.0 - 0.5 * (coverage_min + coverage_max), 0.0), 1.0)),
        "eval_kernel_headroom": float(min(max(eval_headroom_ratio, 0.0), 1.0)),
    }

    if axes["adaptation_axis"] >= 0.25 and axes["train_kernel_pressure"] >= 0.4:
        shape_name = "eval_adaptation_dominant"
    elif axes["long_context_axis"] >= axes["lexical_shortcut_axis"]:
        shape_name = "long_context_dominant"
    else:
        shape_name = "local_lexical_dominant"

    return {
        "shape_name": shape_name,
        "best_eval_seq_len_by_reuse": int(best_eval["seq_len"]),
        "estimated_steps_point": est_steps,
        "train_stream_coverage_range": [coverage_min, coverage_max],
        "axes": axes,
    }


def parse_component_overheads(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not text.strip():
        return out
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"Expected name=value in component overhead spec, got: {item}")
        out[key.strip()] = float(value.strip())
    return out


def _lookup_reuse(reuse_stats: list[dict[str, float]], window: int) -> float | None:
    return next((float(item["reuse_rate"]) for item in reuse_stats if int(item["window"]) == window), None)


def _lookup_eval_reuse(eval_candidates: list[dict[str, float]], seq_len: int) -> float | None:
    return next((float(item["context_reuse_rate"]) for item in eval_candidates if int(item["seq_len"]) == seq_len), None)


def focused_upgrade_analysis(
    lag_stats: list[LagMetric],
    reuse_stats: list[dict[str, float]],
    eval_candidates: list[dict[str, float]],
    training_budget: dict[str, object],
    component_overheads: dict[str, float],
    observed_component_overrides: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    reuse_128 = _lookup_reuse(reuse_stats, 128) or 0.0
    reuse_1024 = _lookup_eval_reuse(eval_candidates, 1024) or _lookup_reuse(reuse_stats, 1024) or 0.0
    reuse_1408 = _lookup_eval_reuse(eval_candidates, 1408) or 0.0
    reuse_2048 = _lookup_eval_reuse(eval_candidates, 2048) or _lookup_reuse(reuse_stats, 2048) or 0.0
    long_nmi = max((m.normalized_mi for m in lag_stats if m.lag >= 512), default=0.0)
    range_points = training_budget.get("range", {}).get("points", []) if isinstance(training_budget, dict) else []
    base_step_values = [float(point["avg_step_ms"]) for point in range_points] or [float(training_budget["avg_step_ms"])]

    default_overheads = {
        "eval_seq_len_1408": 0.0,
        "eval_seq_len_2048": 0.0,
        "smear_gate": 0.03,
        "bigram_1024": 0.08,
        "bigram_2048": 0.12,
        "bigram_4096": 0.18,
        "train_seq_len_2048": 0.35,
        "ttt_lora": 0.0,
        "tokenizer_redesign": 0.0,
        "fake_quant_tail": 0.02,
    }
    merged_overheads = {**default_overheads, **component_overheads}
    if observed_component_overrides:
        merged_overheads.update(observed_component_overrides)

    candidates = [
        {
            "name": "eval_seq_len_1408",
            "kind": "eval_only",
            "rationale": "Longer final eval only; preserves training steps while capturing more context reuse.",
            "dataset_signal": max(reuse_1408 - reuse_1024, 0.0) + 0.5 * long_nmi,
        },
        {
            "name": "eval_seq_len_2048",
            "kind": "eval_only",
            "rationale": "Longest profiled final eval; highest measured context reuse.",
            "dataset_signal": max(reuse_2048 - reuse_1024, 0.0) + 0.75 * long_nmi,
        },
        {
            "name": "smear_gate",
            "kind": "train_and_eval",
            "rationale": "Light lexical shortcut with likely small training-kernel tax.",
            "dataset_signal": 0.6 * reuse_128,
        },
        {
            "name": "bigram_1024",
            "kind": "train_and_eval",
            "rationale": "Moderate lexical shortcut capacity with lower collision rate than smear-only.",
            "dataset_signal": 0.8 * reuse_128,
        },
        {
            "name": "bigram_2048",
            "kind": "train_and_eval",
            "rationale": "Stronger lexical shortcut lane if the step-time increase stays acceptable.",
            "dataset_signal": 0.95 * reuse_128,
        },
        {
            "name": "bigram_4096",
            "kind": "train_and_eval",
            "rationale": "Largest lexical shortcut lane; highest potential lexical capacity but expensive kernel risk.",
            "dataset_signal": 1.0 * reuse_128,
        },
        {
            "name": "train_seq_len_2048",
            "kind": "train_and_eval",
            "rationale": "Longer training context can exploit very high long-window reuse, but it directly taxes the training kernel.",
            "dataset_signal": max(reuse_2048 - reuse_1024, 0.0) + long_nmi,
        },
        {
            "name": "ttt_lora",
            "kind": "eval_only",
            "rationale": "Eval-time adaptation can exploit high long-window reuse without changing train steps.",
            "dataset_signal": 0.5 * reuse_2048 + 0.5 * long_nmi,
        },
        {
            "name": "tokenizer_redesign",
            "kind": "data_model",
            "rationale": "Tokenizer compression already looks decent; this remains a later-stage, higher-disruption lane.",
            "dataset_signal": max(0.0, 2.4 - 2.03539625) * 0.1,
        },
        {
            "name": "fake_quant_tail",
            "kind": "train_and_export",
            "rationale": "Cheap post-quant robustness lane aligned to the scored artifact.",
            "dataset_signal": 0.08,
        },
    ]

    results: list[dict[str, object]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        overhead = float(merged_overheads.get(name, 0.0))
        step_points = []
        for base_ms in base_step_values:
            adjusted_ms = base_ms * (1.0 + overhead)
            est_steps = int((float(training_budget["max_wallclock_seconds"]) * 1000.0) // adjusted_ms)
            base_steps = int((float(training_budget["max_wallclock_seconds"]) * 1000.0) // base_ms)
            step_points.append(
                {
                    "base_step_ms": base_ms,
                    "adjusted_step_ms": adjusted_ms,
                    "estimated_steps": est_steps,
                    "lost_steps_vs_base": base_steps - est_steps,
                }
            )
        avg_lost = float(np.mean([p["lost_steps_vs_base"] for p in step_points])) if step_points else 0.0
        dataset_signal = float(candidate["dataset_signal"])
        if candidate["kind"] == "eval_only":
            priority_score = dataset_signal * 1000.0
        else:
            priority_score = dataset_signal * 1000.0 - avg_lost
        results.append(
            {
                "name": name,
                "kind": candidate["kind"],
                "dataset_signal": dataset_signal,
                "assumed_train_slowdown_fraction": overhead,
                "rationale": candidate["rationale"],
                "budget_sensitivity": {
                    "estimated_steps_min": int(min(p["estimated_steps"] for p in step_points)) if step_points else None,
                    "estimated_steps_max": int(max(p["estimated_steps"] for p in step_points)) if step_points else None,
                    "lost_steps_min": int(min(p["lost_steps_vs_base"] for p in step_points)) if step_points else None,
                    "lost_steps_max": int(max(p["lost_steps_vs_base"] for p in step_points)) if step_points else None,
                    "points": step_points,
                },
                "priority_score": priority_score,
            }
        )
    return sorted(results, key=lambda item: float(item["priority_score"]), reverse=True)


def load_upgrade_module() -> object | None:
    if torch is None:
        return None
    root = Path(__file__).resolve().parents[1]
    module_path = root / "train_gpt_upgrade.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("train_gpt_upgrade_profile_import", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_token_enrichment_kernels(
    vocab_size: int,
    train_seq_len: int,
    train_batch_tokens: int,
    eval_seq_len: int,
    eval_batch_seqs: int,
    benchmark_bigrams: list[int],
    benchmark_steps: int,
    benchmark_warmup_steps: int,
    device_name: str,
    max_wallclock_seconds: float,
) -> dict[str, object]:
    if torch is None:
        return {"available": False, "reason": "torch_not_installed"}

    module = load_upgrade_module()
    if module is None:
        return {"available": False, "reason": "train_gpt_upgrade_import_failed"}

    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        return {"available": False, "reason": f"requested_device_unavailable:{device_name}"}

    per_step_seqs = max(train_batch_tokens // max(train_seq_len, 1), 1)
    variants: list[tuple[str, bool, int]] = [("baseline", False, 0), ("smear_only", True, 0)]
    for size in benchmark_bigrams:
        if size <= 0:
            continue
        variants.append((f"bigram_{size}", False, size))
        variants.append((f"smear_bigram_{size}", True, size))

    results: list[dict[str, object]] = []
    base_train_ms: float | None = None
    base_eval_ms: float | None = None

    for name, use_smear_gate, bigram_vocab_size in variants:
        model = module.GPT(
            vocab_size=vocab_size,
            num_layers=10,
            model_dim=512,
            num_heads=8,
            num_kv_heads=4,
            mlp_mult=2,
            mlp_hidden=992,
            num_register_tokens=0,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            overtone_embed_init=False,
            overtone_embed_power=0.5,
            overtone_embed_scale=1.0,
            resid_mix_phase_init=False,
            resid_mix_phase_sharpness=8.0,
            use_alibi=False,
            token_shift=False,
            use_xpos=False,
            xpos_scale_base=512.0,
            num_shared_layers=0,
            shared_layer_repeats=0,
            shared_loop_gate_init=-2.0,
            use_residual_attention=False,
            residual_attention_gain_init=0.5,
            memory_kv_slots=0,
            use_relative_bias=False,
            relative_bias_num_buckets=32,
            relative_bias_max_distance=128,
            macaron_layout=False,
            retention_layers=0,
            retention_decay_init=1.5,
            retention_output_gate=True,
            dynamic_head_layers=0,
            dynamic_head_top_k=0,
            dynamic_head_temperature=1.0,
            use_subln=False,
            use_deepnorm=False,
            moe_every_n_layers=0,
            moe_num_experts=4,
            moe_top_k=1,
            moe_hidden=0,
            use_hyper_connections=False,
            hyper_connection_layers=0,
            hyper_connection_gate_init=-2.0,
            use_smear_gate=use_smear_gate,
            bigram_vocab_size=bigram_vocab_size,
            bigram_dim=128,
            orthogonal_init=False,
            mup_proj_init=False,
        ).to(device)
        if device.type == "cuda":
            model = model.bfloat16()
            for submodule in model.modules():
                if isinstance(submodule, module.CastedLinear):
                    submodule.float()
            module.restore_low_dim_params_to_fp32(model)
            amp_enabled = True
        else:
            amp_enabled = False

        model.train()
        x = torch.randint(0, vocab_size, (per_step_seqs, train_seq_len), device=device, dtype=torch.int64)
        y = torch.randint(0, vocab_size, (per_step_seqs, train_seq_len), device=device, dtype=torch.int64)
        train_durations: list[float] = []

        for step_idx in range(benchmark_warmup_steps + benchmark_steps):
            for param in model.parameters():
                param.grad = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                loss = model(x, y)
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            if step_idx >= benchmark_warmup_steps:
                train_durations.append(elapsed_ms)

        model.eval()
        eval_x = torch.randint(0, vocab_size, (eval_batch_seqs, eval_seq_len), device=device, dtype=torch.int64)
        eval_durations: list[float] = []
        with torch.inference_mode():
            for step_idx in range(benchmark_warmup_steps + benchmark_steps):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    _ = model.forward_logits(eval_x)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = 1000.0 * (time.perf_counter() - t0)
                if step_idx >= benchmark_warmup_steps:
                    eval_durations.append(elapsed_ms)

        train_step_ms = float(np.median(train_durations))
        eval_step_ms = float(np.median(eval_durations))
        if base_train_ms is None:
            base_train_ms = train_step_ms
            base_eval_ms = eval_step_ms
        estimated_train_steps = int((max_wallclock_seconds * 1000.0) // train_step_ms)
        estimated_eval_passes = int((max_wallclock_seconds * 1000.0) // eval_step_ms)
        results.append(
            {
                "variant": name,
                "use_smear_gate": bool(use_smear_gate),
                "bigram_vocab_size": int(bigram_vocab_size),
                "train_step_ms_median": train_step_ms,
                "eval_step_ms_median": eval_step_ms,
                "train_slowdown_vs_baseline": (train_step_ms / base_train_ms) if base_train_ms else 1.0,
                "eval_slowdown_vs_baseline": (eval_step_ms / base_eval_ms) if base_eval_ms else 1.0,
                "estimated_train_steps_in_budget": estimated_train_steps,
                "estimated_eval_forwards_in_budget": estimated_eval_passes,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "available": True,
        "device": str(device),
        "train_batch_tokens": int(train_batch_tokens),
        "train_seq_len": int(train_seq_len),
        "train_batch_sequences": int(per_step_seqs),
        "eval_seq_len": int(eval_seq_len),
        "eval_batch_seqs": int(eval_batch_seqs),
        "benchmark_steps": int(benchmark_steps),
        "benchmark_warmup_steps": int(benchmark_warmup_steps),
        "variants": results,
    }


def summarize_recommendations(
    lag_stats: list[LagMetric],
    reuse_stats: list[dict[str, float]],
    entropy_bits: float,
    bytes_per_token: float | None,
    shard_aggregate: dict[str, float] | None = None,
    eval_candidates: list[dict[str, float]] | None = None,
    training_budget: dict[str, float | int] | None = None,
    training_coverage: dict[str, object] | None = None,
    loss_shape: dict[str, object] | None = None,
    observed_frontier: list[dict[str, object]] | None = None,
    leaderboard_frontier: list[dict[str, object]] | None = None,
) -> list[str]:
    recs: list[str] = []
    long_lag_nmi = [m.normalized_mi for m in lag_stats if m.lag >= 512]
    medium_lag_nmi = [m.normalized_mi for m in lag_stats if 64 <= m.lag < 512]
    short_reuse = next((x["reuse_rate"] for x in reuse_stats if x["window"] == 128), None)
    long_reuse = next((x["reuse_rate"] for x in reuse_stats if x["window"] == 2048), None)

    if long_lag_nmi and max(long_lag_nmi) >= 0.01:
        recs.append("Long-context eval is worth testing: the stream retains measurable dependence beyond 512 tokens.")
    elif medium_lag_nmi and max(medium_lag_nmi) >= 0.01:
        recs.append("Signal is mostly medium-range: prioritize 512-1408 eval sweeps before pushing to very long contexts.")
    else:
        recs.append("Long-range token dependence looks weak in the sampled stream; local lexical features may pay off more than bigger eval context.")

    if short_reuse is not None and short_reuse >= 0.45:
        recs.append("High local reuse suggests BigramHash / SmearGate style lexical shortcuts are plausible high-ROI additions.")
    if long_reuse is not None and long_reuse >= 0.70:
        recs.append("Very high long-window reuse suggests TTT or longer eval context could exploit document-specific repetition.")

    if entropy_bits <= 6.5:
        recs.append("Token entropy is fairly concentrated; compression-aware training and selective precision are likely more important than adding capacity.")
    else:
        recs.append("Token entropy is relatively high; tokenizer/data pairing shifts may still buy meaningful headroom.")

    if bytes_per_token is not None:
        if bytes_per_token >= 3.6:
            recs.append("Average bytes per token are high; tokenizer redesign may be a strong lever.")
        elif bytes_per_token <= 2.8:
            recs.append("Tokenizer compression already looks decent; focus on modeling and post-quant export interaction first.")
    if shard_aggregate is not None and shard_aggregate.get("max_js_divergence_bits", 0.0) >= 0.02:
        recs.append("Shard drift is non-trivial; TTT or mild domain-adaptive evaluation may benefit more than a one-size-fits-all context setting.")
    if eval_candidates:
        best = max(eval_candidates, key=lambda item: item["context_reuse_rate"])
        if best["seq_len"] >= 1408:
            recs.append(f"Longer final eval looks justified; sampled reuse is strongest by seq_len={best['seq_len']}.")
    if training_budget is not None:
        estimated_steps = int(training_budget["estimated_steps"])
        train_seq_len = int(training_budget["train_seq_len"])
        if estimated_steps < 6000:
            recs.append(
                f"Throughput is tight at TRAIN_SEQ_LEN={train_seq_len}: estimated {estimated_steps} steps in 10 minutes. "
                "Prefer longer eval over longer training context unless the modeling gain is overwhelming."
            )
        else:
            recs.append(
                f"Estimated {estimated_steps} steps fit in the 10-minute budget at TRAIN_SEQ_LEN={train_seq_len}; "
                "this training context is still throughput-viable."
            )
        range_info = training_budget.get("range") if isinstance(training_budget, dict) else None
        if isinstance(range_info, dict):
            recs.append(
                f"Using the supplied step-time range, a more realistic 10-minute budget is "
                f"{int(range_info['estimated_steps_min'])}-{int(range_info['estimated_steps_max'])} steps."
            )
    if observed_frontier:
        true_2048 = next((row for row in observed_frontier if int(row["train_seq_len"]) == 2048), None)
        true_1024 = next((row for row in observed_frontier if int(row["train_seq_len"]) == 1024), None)
        if true_2048 is not None and float(true_2048["fastest_step_avg_ms"]) <= 300.0:
            recs.append(
                f"Observed cluster runs show TRAIN_SEQ_LEN=2048 is throughput-viable at "
                f"{float(true_2048['fastest_step_avg_ms']):.2f} ms/step with batch_tokens={int(true_2048['train_batch_tokens'])}; "
                "override generic slowdown heuristics accordingly."
            )
        if (
            true_2048 is not None
            and true_1024 is not None
            and float(true_2048["fastest_step_avg_ms"]) <= 300.0
            and float(true_1024["fastest_step_avg_ms"]) <= 300.0
        ):
            ratio = float(true_2048["fastest_step_avg_ms"]) / max(float(true_1024["fastest_step_avg_ms"]), 1e-9)
            if 0.5 <= ratio <= 2.0:
                recs.append(
                    f"Observed seq-len runtime ratio 2048/1024 is {ratio:.3f} for the fastest logged lanes, "
                    "which is the most relevant budget signal for this cluster."
                )
    if leaderboard_frontier:
        top = leaderboard_frontier[0]
        recs.append(
            f"Leaderboard-consistent ranking currently favors {top['name']} "
            f"(score={float(top['composite_score']):.3f}); use this as an external prior, separate from pure dataset signal."
        )
    if training_coverage is not None:
        recs.append(
            f"Estimated train-stream coverage is {training_coverage['coverage_fraction_min']:.3f}-"
            f"{training_coverage['coverage_fraction_max']:.3f} of the available shard tokens, with "
            f"{training_coverage['expected_wraps_min']:.3f}-{training_coverage['expected_wraps_max']:.3f} expected passes."
        )
    if loss_shape is not None:
        shape_name = str(loss_shape.get("shape_name", "unknown"))
        axes = loss_shape.get("axes", {})
        recs.append(
            f"Loss-shape regime looks {shape_name}; axes long_context={axes.get('long_context_axis', 0.0):.3f} "
            f"lexical={axes.get('lexical_shortcut_axis', 0.0):.3f} adaptation={axes.get('adaptation_axis', 0.0):.3f} "
            f"train_pressure={axes.get('train_kernel_pressure', 0.0):.3f}."
        )
    return recs


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile parameter-golf training shards for tuning decisions.")
    parser.add_argument("--train-glob", default="./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
    parser.add_argument("--val-glob", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--max-shards", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4_000_000)
    parser.add_argument("--sample-across-shards", action="store_true")
    parser.add_argument("--lags", default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--reuse-windows", default="128,512,1024,2048")
    parser.add_argument("--eval-lengths", default="512,1024,1408,2048")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--sample-tokens-per-shard", type=int, default=250_000)
    parser.add_argument("--include-shard-summaries", action="store_true")
    parser.add_argument("--max-wallclock-seconds", type=float, default=600.0)
    parser.add_argument("--avg-step-ms", type=float, default=43.5)
    parser.add_argument("--avg-step-ms-range", default="")
    parser.add_argument(
        "--component-overheads",
        default="",
        help="Comma-separated assumed training slowdown fractions, e.g. smear_gate=0.03,bigram_2048=0.12",
    )
    parser.add_argument("--base-eval-step-ms", type=float, default=120.0)
    parser.add_argument("--base-eval-seq-len", type=int, default=1024)
    parser.add_argument("--base-eval-batch-seqs", type=int, default=512)
    parser.add_argument("--ttt-multipliers", default="1.5,2.0,3.0")
    parser.add_argument("--train-batch-tokens", type=int, default=524_288)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--benchmark-kernels", action="store_true")
    parser.add_argument("--benchmark-device", default="")
    parser.add_argument("--benchmark-steps", type=int, default=10)
    parser.add_argument("--benchmark-warmup-steps", type=int, default=3)
    parser.add_argument("--benchmark-bigram-sizes", default="1024,2048,4096")
    parser.add_argument("--benchmark-eval-batch-seqs", type=int, default=256)
    parser.add_argument("--run-logs", default="")
    parser.add_argument("--wallclock-points-seconds", default="300,450,600")
    parser.add_argument(
        "--candidate-models",
        default="",
        help="Comma-separated candidate model specs: name:num_layers:model_dim:mlp_hidden[:num_heads:num_kv_heads[:vocab_size]]",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    files = resolve_files(args.train_glob, args.max_shards)
    val_files = resolve_files(args.val_glob, 0)
    train_stream = shard_token_totals(files)
    val_stream = shard_token_totals(val_files)
    sequential_tokens = iter_tokens(files, args.max_tokens)
    distribution_tokens = sample_tokens_across_shards(files, args.max_tokens) if args.sample_across_shards else sequential_tokens
    token_counts = np.bincount(distribution_tokens, minlength=args.vocab_size).astype(np.int64)
    entropy_bits = entropy_from_counts(token_counts)
    effective_vocab = float(2.0**entropy_bits)
    lag_stats = lag_metrics(sequential_tokens, args.vocab_size, parse_int_list(args.lags))
    reuse_stats = context_reuse_profile(sequential_tokens, parse_int_list(args.reuse_windows))

    base_bytes = build_base_bytes(args.tokenizer_path, args.vocab_size) if args.tokenizer_path else None
    bytes_per_token = None if base_bytes is None else float(base_bytes[distribution_tokens].mean())
    shard_summary_rows, shard_aggregate = shard_summaries(files, args.vocab_size, args.sample_tokens_per_shard, base_bytes)
    eval_candidates = eval_length_candidates(sequential_tokens, parse_int_list(args.eval_lengths))
    marginal_context = marginal_eval_context_gain(eval_candidates)
    context_bands = context_gain_by_distance(lag_stats, [(1, 8), (16, 64), (128, 512), (1024, 2048)])
    recurrence_profile = recurrence_burst_profile(sequential_tokens, max_gap=max(parse_int_list(args.eval_lengths), default=2048))
    recurrence_by_bucket = recurrence_burst_profile_by_frequency(
        sequential_tokens,
        vocab_size=args.vocab_size,
        counts=token_counts,
        max_gap=max(parse_int_list(args.eval_lengths), default=2048),
    )
    training_budget = estimate_training_budget(
        max_wallclock_seconds=args.max_wallclock_seconds,
        avg_step_ms=args.avg_step_ms,
        train_batch_tokens=args.train_batch_tokens,
        train_seq_len=args.train_seq_len,
        world_size=args.world_size,
        grad_accum_steps=args.grad_accum_steps,
    )
    if args.avg_step_ms_range.strip():
        training_budget["range"] = estimate_training_budget_range(
            max_wallclock_seconds=args.max_wallclock_seconds,
            avg_step_ms_values=parse_float_list(args.avg_step_ms_range),
            train_batch_tokens=args.train_batch_tokens,
            train_seq_len=args.train_seq_len,
            world_size=args.world_size,
            grad_accum_steps=args.grad_accum_steps,
        )
    training_coverage = estimate_training_coverage(
        total_train_tokens_available=int(train_stream["total_tokens"]),
        training_budget=training_budget,
    )
    run_log_summaries, missing_run_logs = load_run_logs([path.strip() for path in args.run_logs.split(",") if path.strip()])
    observed_frontier = observed_runtime_frontier(run_log_summaries)
    observed_overrides = infer_component_overheads_from_runs(run_log_summaries)
    focused_analysis = focused_upgrade_analysis(
        lag_stats=lag_stats,
        reuse_stats=reuse_stats,
        eval_candidates=eval_candidates,
        training_budget=training_budget,
        component_overheads=parse_component_overheads(args.component_overheads),
        observed_component_overrides=observed_overrides,
    )
    leaderboard_analysis = leaderboard_consistent_analysis(
        lag_stats=lag_stats,
        reuse_stats=reuse_stats,
        eval_candidates=eval_candidates,
        focused_analysis=focused_analysis,
    )
    eval_budget_frontier = estimate_eval_budget_frontier(
        eval_lengths=parse_int_list(args.eval_lengths),
        eval_batch_seqs=args.base_eval_batch_seqs,
        max_wallclock_seconds=args.max_wallclock_seconds,
        base_eval_seq_len=args.base_eval_seq_len,
        base_eval_step_ms=args.base_eval_step_ms,
        ttt_multipliers=parse_float_list(args.ttt_multipliers),
    )
    real_eval_frontier = estimate_real_eval_budget_frontier(
        total_val_tokens=int(val_stream["total_tokens"]),
        eval_lengths=parse_int_list(args.eval_lengths),
        eval_stride=64,
        eval_batch_seqs=args.base_eval_batch_seqs,
        max_wallclock_seconds=args.max_wallclock_seconds,
        base_eval_seq_len=args.base_eval_seq_len,
        base_eval_batch_seqs=args.base_eval_batch_seqs,
        base_eval_step_ms=args.base_eval_step_ms,
        ttt_multipliers=parse_float_list(args.ttt_multipliers),
    )
    transition_geometry = transition_geometry_profile(sequential_tokens, args.vocab_size, token_counts, args.top_k)
    loss_shape = loss_geometry_surface(
        lag_stats=lag_stats,
        reuse_stats=reuse_stats,
        eval_candidates=eval_candidates,
        recurrence_profile=recurrence_profile,
        recurrence_by_bucket=recurrence_by_bucket,
        transition_geometry=transition_geometry,
        training_budget=training_budget,
        training_coverage=training_coverage,
        eval_budget_frontier=eval_budget_frontier,
    )
    anchor_run = min(
        [item for item in run_log_summaries if item.best_val_bpb is not None and item.final_step_avg_ms is not None],
        key=lambda item: float(item.best_val_bpb),
        default=None,
    )
    current_model_params = anchor_run.model_params if anchor_run is not None else None
    current_best_val_bpb = anchor_run.best_val_bpb if anchor_run is not None else None
    candidate_models = parse_candidate_models(
        args.candidate_models,
        default_vocab_size=args.vocab_size,
        default_num_heads=8,
        default_num_kv_heads=4,
    )
    model_frontier = budgeted_model_candidates(
        candidates=candidate_models,
        anchor_run=anchor_run,
        max_wallclock_seconds=args.max_wallclock_seconds,
        total_train_tokens_available=int(train_stream["total_tokens"]),
        current_model_params=current_model_params,
        current_best_val_bpb=current_best_val_bpb,
    )
    equal_wallclock = compare_runs_equal_wallclock(run_log_summaries, parse_int_list(args.wallclock_points_seconds))
    recipes = recipe_frontier(
        focused_analysis=focused_analysis,
        real_eval_frontier=real_eval_frontier,
        training_budget=training_budget,
        training_coverage=training_coverage,
    )
    kernel_efficiency = (
        benchmark_token_enrichment_kernels(
            vocab_size=args.vocab_size,
            train_seq_len=args.train_seq_len,
            train_batch_tokens=args.train_batch_tokens,
            eval_seq_len=max(parse_int_list(args.eval_lengths), default=args.train_seq_len),
            eval_batch_seqs=args.benchmark_eval_batch_seqs,
            benchmark_bigrams=parse_int_list(args.benchmark_bigram_sizes),
            benchmark_steps=args.benchmark_steps,
            benchmark_warmup_steps=args.benchmark_warmup_steps,
            device_name=args.benchmark_device,
            max_wallclock_seconds=args.max_wallclock_seconds,
        )
        if args.benchmark_kernels
        else {"available": False, "reason": "benchmark_kernels_disabled"}
    )

    profile = {
        "schema_version": 1,
        "competition_grounding": {
            "objective": "Minimize post-roundtrip validation BPB under the 16,000,000-byte submission limit.",
            "evaluation_constraints": [
                "Evaluation may use any sequence length, but the total evaluation procedure must stay under 10 minutes on 8xH100.",
                "No training-data access is allowed during evaluation unless those bits are included in the artifact.",
            ],
            "profiling_use": "This profile is for offline tuning and sweep prioritization, not for submission-time evaluation.",
        },
        "train_glob": args.train_glob,
        "tokenizer_path": args.tokenizer_path,
        "vocab_size": args.vocab_size,
        "sample": {
            "num_files": len(files),
            "files": [str(path) for path in files],
            "sequential_tokens_profiled": int(sequential_tokens.size),
            "distribution_tokens_profiled": int(distribution_tokens.size),
            "distribution_mode": "cross_shard_sample" if args.sample_across_shards else "sequential_prefix",
        },
        "train_stream": train_stream,
        "val_stream": val_stream,
        "token_distribution": {
            "entropy_bits": entropy_bits,
            "effective_vocab": effective_vocab,
            "gini": gini_from_counts(token_counts),
            "top_tokens": top_tokens(token_counts, args.top_k),
        },
        "lag_dependence": [
            {
                "lag": m.lag,
                "match_rate": m.match_rate,
                "mutual_information_bits": m.mutual_information_bits,
                "normalized_mi": m.normalized_mi,
                "conditional_entropy_bits": m.conditional_entropy_bits,
                "support_pairs": m.support_pairs,
            }
            for m in lag_stats
        ],
        "context_gain_by_distance": context_bands,
        "context_reuse": reuse_stats,
        "marginal_eval_context_gain": marginal_context,
        "recurrence_burst_profile": recurrence_profile,
        "recurrence_burst_profile_by_frequency": recurrence_by_bucket,
        "top_bigrams": top_bigrams(sequential_tokens, args.vocab_size, args.top_k),
        "transition_geometry": transition_geometry,
        "eval_length_candidates": eval_candidates,
        "eval_budget_frontier": eval_budget_frontier,
        "real_eval_budget_frontier": real_eval_frontier,
        "training_budget_estimate": training_budget,
        "training_coverage_estimate": training_coverage,
        "focused_upgrade_analysis": focused_analysis,
        "leaderboard_consistent_analysis": leaderboard_analysis,
        "loss_geometry_surface": loss_shape,
        "run_log_frontier": {
            "anchor_run": None
            if anchor_run is None
            else {
                "path": anchor_run.path,
                "train_seq_len": anchor_run.train_seq_len,
                "eval_seq_len": anchor_run.eval_seq_len,
                "train_batch_tokens": anchor_run.train_batch_tokens,
                "model_params": anchor_run.model_params,
                "steps_completed": anchor_run.steps_completed,
                "final_step_avg_ms": anchor_run.final_step_avg_ms,
                "best_val_bpb": anchor_run.best_val_bpb,
                "final_val_bpb": anchor_run.final_val_bpb,
                "best_val_step": anchor_run.best_val_step,
                "final_val_step": anchor_run.final_val_step,
                "marginal_bpb_gain_last_1000": anchor_run.marginal_bpb_gain_last_1000,
                "stop_recommendation": anchor_run.stop_recommendation,
                "train_time_ms": anchor_run.train_time_ms,
            },
            "runs": [
                {
                    "path": item.path,
                    "train_seq_len": item.train_seq_len,
                    "eval_seq_len": item.eval_seq_len,
                    "train_batch_tokens": item.train_batch_tokens,
                    "model_params": item.model_params,
                    "steps_completed": item.steps_completed,
                    "final_step_avg_ms": item.final_step_avg_ms,
                    "best_val_bpb": item.best_val_bpb,
                    "final_val_bpb": item.final_val_bpb,
                    "best_val_step": item.best_val_step,
                    "final_val_step": item.final_val_step,
                    "marginal_bpb_gain_last_1000": item.marginal_bpb_gain_last_1000,
                    "stop_recommendation": item.stop_recommendation,
                    "train_time_ms": item.train_time_ms,
                }
                for item in run_log_summaries
            ],
            "missing": [{"path": item.path, "reason": item.reason} for item in missing_run_logs],
            "observed_runtime_frontier": observed_frontier,
            "observed_component_overrides": observed_overrides,
        },
        "equal_wallclock_run_comparison": equal_wallclock,
        "budgeted_model_candidates": model_frontier,
        "recipe_frontier": recipes,
        "kernel_efficiency": kernel_efficiency,
        "shard_profile": {
            "aggregate": shard_aggregate,
            "summaries": shard_summary_rows if args.include_shard_summaries else [],
        },
        "tokenizer_profile": {
            "bytes_per_token": bytes_per_token,
        },
        "recommendations": summarize_recommendations(
            lag_stats,
            reuse_stats,
            entropy_bits,
            bytes_per_token,
            shard_aggregate=shard_aggregate,
            eval_candidates=eval_candidates,
            training_budget=training_budget,
            training_coverage=training_coverage,
            loss_shape=loss_shape,
            observed_frontier=observed_frontier,
            leaderboard_frontier=leaderboard_analysis,
        ),
        "limitations": [
            "Binary shards do not preserve explicit document boundaries, so document-level TTT fit is inferred indirectly from stream statistics.",
            "Mutual information is estimated on the sampled token stream, not the full corpus.",
            "Shard-level drift is estimated from evenly spaced token samples within each shard rather than exhaustive full-shard histograms.",
            "Kernel efficiency benchmarks are synthetic random-token timing estimates on the current hardware; they predict relative cost, not final BPB.",
        ],
    }

    output = json.dumps(profile, indent=2, sort_keys=True)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
        print(out_path)
    else:
        print(output)


if __name__ == "__main__":
    main()
