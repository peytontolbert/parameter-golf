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
import hashlib
import glob
import importlib.util
import json
import math
import os
import pickle
import re
import tempfile
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
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
PROFILE_CACHE_SCHEMA_VERSION = 3


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


def shard_num_tokens(file: Path) -> int:
    header = np.fromfile(file, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    return int(header[2])


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


def token_count_for_files(files: Iterable[Path], max_tokens: int) -> int:
    total = 0
    remaining = max_tokens if max_tokens > 0 else None
    for file in files:
        if remaining is not None and remaining <= 0:
            break
        num_tokens = shard_num_tokens(file)
        if remaining is not None and num_tokens > remaining:
            num_tokens = remaining
        total += int(num_tokens)
        if remaining is not None:
            remaining -= int(num_tokens)
    return int(total)


def _path_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _normalize_for_cache(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha1": hashlib.sha1(np.ascontiguousarray(value).view(np.uint8)).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _normalize_for_cache(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_cache(v) for v in value]
    return value


def build_cache_key(stage_name: str, payload: dict[str, object]) -> str:
    normalized = {
        "cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "stage_name": stage_name,
        "payload": _normalize_for_cache(payload),
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def load_cached_stage(cache_dir: Path, stage_name: str, key: str) -> object | None:
    cache_path = cache_dir / stage_name / f"{key}.pkl"
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as f:
        return pickle.load(f)


def save_cached_stage(cache_dir: Path, stage_name: str, key: str, value: object) -> Path:
    stage_dir = cache_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    final_path = stage_dir / f"{key}.pkl"
    fd, tmp_name = tempfile.mkstemp(prefix=f"{stage_name}_{key}_", suffix=".tmp", dir=str(stage_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return final_path


def materialize_token_stream_memmap(
    files: list[Path],
    max_tokens: int,
    cache_dir: Path,
    namespace: str,
    *,
    sample_across_shards_mode: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    token_cache_dir = cache_dir / "token_streams"
    token_cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "namespace": namespace,
        "max_tokens": int(max_tokens),
        "sample_across_shards_mode": bool(sample_across_shards_mode),
        "files": [_path_fingerprint(path) for path in files],
    }
    key = build_cache_key(f"token_stream_{namespace}", fingerprint)
    final_path = token_cache_dir / f"{namespace}_{key}.npy"
    cache_hit = final_path.exists()
    if not cache_hit:
        if sample_across_shards_mode:
            target_tokens = int(max_tokens) if int(max_tokens) > 0 else token_count_for_files(files, max_tokens)
            per_shard = max(target_tokens // max(len(files), 1), 1)
            sample_counts = [min(per_shard, shard_num_tokens(file)) for file in files]
            total_tokens = max(min(int(sum(sample_counts)), target_tokens), 1)
        else:
            total_tokens = token_count_for_files(files, max_tokens)
        memmap = np.lib.format.open_memmap(str(final_path), mode="w+", dtype=np.int32, shape=(int(total_tokens),))
        if sample_across_shards_mode:
            offset = 0
            for file in files:
                sampled = sample_tokens_from_shard(file, per_shard)
                if offset >= memmap.shape[0]:
                    break
                take = min(int(sampled.size), int(memmap.shape[0] - offset))
                memmap[offset : offset + take] = sampled[:take]
                offset += take
        else:
            offset = 0
            remaining = int(max_tokens) if int(max_tokens) > 0 else None
            for file in files:
                if remaining is not None and remaining <= 0:
                    break
                arr = load_data_shard(file)
                if remaining is not None and arr.size > remaining:
                    arr = arr[:remaining]
                take = int(arr.size)
                memmap[offset : offset + take] = arr
                offset += take
                if remaining is not None:
                    remaining -= take
        memmap.flush()
        del memmap
    return (
        np.load(final_path, mmap_mode="r"),
        {
            "cache_key": key,
            "path": str(final_path),
            "cache_hit": bool(cache_hit),
            "sample_across_shards_mode": bool(sample_across_shards_mode),
        },
    )


def sample_tokens_from_shard(file: Path, sample_tokens: int) -> np.ndarray:
    tokens = load_data_shard(file)
    if sample_tokens <= 0 or sample_tokens >= tokens.size:
        return tokens
    idx = np.linspace(0, tokens.size - 1, num=sample_tokens, dtype=np.int64)
    return np.ascontiguousarray(tokens[idx])


def read_shard_prefix_tokens(file: Path, prefix_tokens: int) -> np.ndarray:
    header = np.fromfile(file, dtype="<i4", count=HEADER_INTS)
    header_bytes = HEADER_INTS * np.dtype("<i4").itemsize
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    if prefix_tokens <= 0 or prefix_tokens >= num_tokens:
        return load_data_shard(file)
    tokens = np.fromfile(file, dtype="<u2", count=prefix_tokens, offset=header_bytes)
    if tokens.size != prefix_tokens:
        raise ValueError(f"Short read for {file}")
    return tokens.astype(np.int32, copy=False)


def read_shard_window_tokens(file: Path, start_token: int, window_tokens: int) -> np.ndarray:
    header = np.fromfile(file, dtype="<i4", count=HEADER_INTS)
    header_bytes = HEADER_INTS * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    if num_tokens <= 0:
        return np.zeros((0,), dtype=np.int32)
    if window_tokens <= 0 or window_tokens >= num_tokens:
        return load_data_shard(file)
    start = min(max(int(start_token), 0), max(num_tokens - int(window_tokens), 0))
    tokens = np.fromfile(file, dtype="<u2", count=int(window_tokens), offset=header_bytes + start * token_bytes)
    if tokens.size != int(window_tokens):
        raise ValueError(f"Short read for {file}")
    return tokens.astype(np.int32, copy=False)


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


def build_tokenizer_metadata(tokenizer_path: str, vocab_size: int) -> list[dict[str, object]] | None:
    if not tokenizer_path:
        return None
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    meta: list[dict[str, object]] = [
        {"piece": None, "rendered": None, "byte_len": 0, "usable": False, "leading_space": False}
        for _ in range(vocab_size)
    ]
    if path.suffix == ".model":
        if spm is None:
            raise RuntimeError("sentencepiece is not installed")
        sp = spm.SentencePieceProcessor(model_file=str(path))
        for token_id in range(min(vocab_size, int(sp.vocab_size()))):
            if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
                continue
            piece = sp.id_to_piece(token_id)
            leading_space = piece.startswith("▁")
            rendered = (" " + piece[1:]) if leading_space else piece
            meta[token_id] = {
                "piece": piece,
                "rendered": rendered,
                "byte_len": len(rendered.encode("utf-8")),
                "usable": True,
                "leading_space": leading_space,
            }
        return meta

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
            rendered = token.replace("Ġ", " ").replace("▁", " ")
            meta[token_id] = {
                "piece": token,
                "rendered": rendered,
                "byte_len": len(rendered.encode("utf-8")),
                "usable": True,
                "leading_space": token.startswith(("Ġ", "▁")),
            }
        return meta

    raise ValueError(f"Unsupported tokenizer format: {tokenizer_path}")


def reconstruct_text_from_tokens(tokens: np.ndarray, tokenizer_meta: list[dict[str, object]] | None) -> str | None:
    if tokenizer_meta is None:
        return None
    pieces: list[str] = []
    for token_id in map(int, tokens):
        if token_id < 0 or token_id >= len(tokenizer_meta):
            continue
        meta = tokenizer_meta[token_id]
        rendered = meta.get("rendered")
        if not isinstance(rendered, str):
            continue
        pieces.append(rendered)
    if not pieces:
        return None
    return "".join(pieces)


def evaluate_tokenizer_on_text(tokenizer_path: Path, text: str, trainer_family: str) -> dict[str, object]:
    encoded_ids: list[int]
    vocab_size: int
    if tokenizer_path.suffix == ".model":
        if spm is None:
            raise RuntimeError("sentencepiece is not installed")
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        encoded_ids = list(sp.encode(text, out_type=int))
        vocab_size = int(sp.vocab_size())
    elif tokenizer_path.suffix == ".json":
        if Tokenizer is None:
            raise RuntimeError("tokenizers is not installed")
        tok = Tokenizer.from_file(str(tokenizer_path))
        encoding = tok.encode(text)
        encoded_ids = list(encoding.ids)
        vocab_size = int(tok.get_vocab_size(with_added_tokens=True))
    else:
        raise ValueError(f"Unsupported tokenizer format: {tokenizer_path}")
    byte_count = len(text.encode("utf-8"))
    token_count = len(encoded_ids)
    tokens_per_byte = float(token_count / max(byte_count, 1))
    bytes_per_token = float(byte_count / max(token_count, 1))
    counts = np.bincount(np.asarray(encoded_ids, dtype=np.int32), minlength=max(vocab_size, 1)).astype(np.int64)
    entropy_bits = entropy_from_counts(counts)
    return {
        "tokenizer_path": str(tokenizer_path),
        "family": trainer_family,
        "vocab_size": vocab_size,
        "byte_count": int(byte_count),
        "token_count": int(token_count),
        "bytes_per_token": bytes_per_token,
        "tokens_per_byte": tokens_per_byte,
        "entropy_bits": float(entropy_bits),
        "effective_vocab": float(2.0**entropy_bits),
    }


def train_local_tokenizer_candidates(
    text: str | None,
    output_dir: Path,
    vocab_sizes: list[int],
    trainer_families: list[str],
    current_vocab_size: int,
    model_dim: int,
    current_bytes_per_token: float | None,
) -> dict[str, object]:
    if not text:
        return {"available": False, "reason": "reconstructed_text_unavailable"}
    if len(text.strip()) < 1024:
        return {"available": False, "reason": "reconstructed_text_too_small"}

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "tokenizer_profile_sample.txt"
    wrapped_text = "\n".join(text[idx : idx + 2048] for idx in range(0, len(text), 2048))
    sample_path.write_text(wrapped_text, encoding="utf-8")
    results: list[dict[str, object]] = []

    for family in trainer_families:
        normalized_family = family.strip().lower()
        if not normalized_family:
            continue
        for vocab_size in vocab_sizes:
            if vocab_size <= 0:
                continue
            prefix = output_dir / f"{normalized_family}_{vocab_size}"
            try:
                if normalized_family in {"sp_bpe", "sp_unigram"}:
                    if spm is None:
                        raise RuntimeError("sentencepiece_not_installed")
                    model_type = "bpe" if normalized_family == "sp_bpe" else "unigram"
                    spm.SentencePieceTrainer.train(
                        input=str(sample_path),
                        model_prefix=str(prefix),
                        vocab_size=int(vocab_size),
                        model_type=model_type,
                        character_coverage=1.0,
                        bos_id=-1,
                        eos_id=-1,
                        pad_id=-1,
                        unk_id=0,
                        train_extremely_large_corpus=False,
                    )
                    artifact_path = prefix.with_suffix(".model")
                elif normalized_family == "hf_bpe":
                    if Tokenizer is None:
                        raise RuntimeError("tokenizers_not_installed")
                    from tokenizers import Tokenizer as HFTokenizer
                    from tokenizers import models, pre_tokenizers, trainers

                    tokenizer = HFTokenizer(models.BPE(unk_token="[UNK]"))
                    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
                    trainer = trainers.BpeTrainer(
                        vocab_size=int(vocab_size),
                        min_frequency=2,
                        special_tokens=["[UNK]"],
                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                    )
                    tokenizer.train([str(sample_path)], trainer)
                    artifact_path = prefix.with_suffix(".json")
                    tokenizer.save(str(artifact_path))
                else:
                    continue

                metrics = evaluate_tokenizer_on_text(artifact_path, text, normalized_family)
                token_count = int(metrics["token_count"])
                baseline_token_count = None
                estimated_token_change_frac = None
                if current_bytes_per_token is not None and current_bytes_per_token > 0:
                    baseline_token_count = len(text.encode("utf-8")) / current_bytes_per_token
                    estimated_token_change_frac = (token_count / max(baseline_token_count, 1.0)) - 1.0
                embedding_param_delta = (int(vocab_size) - int(current_vocab_size)) * int(model_dim)
                metrics.update(
                    {
                        "estimated_token_change_fraction_vs_current": None
                        if estimated_token_change_frac is None
                        else float(estimated_token_change_frac),
                        "embedding_param_delta": int(embedding_param_delta),
                        "artifact_bytes": int(artifact_path.stat().st_size) if artifact_path.exists() else None,
                    }
                )
                results.append(metrics)
            except Exception as exc:
                results.append(
                    {
                        "family": normalized_family,
                        "vocab_size": int(vocab_size),
                        "available": False,
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
                )

    viable = [row for row in results if row.get("bytes_per_token") is not None]
    for row in viable:
        token_reduction_fraction = max(0.0, -float(row.get("estimated_token_change_fraction_vs_current") or 0.0))
        embedding_penalty = max(int(row.get("embedding_param_delta") or 0), 0) / max(float(model_dim * 1024 * 1024), 1.0)
        artifact_penalty = max(int(row.get("artifact_bytes") or 0) - 100_000, 0) / 1_000_000.0
        conservative_bonus = 0.02 if int(row["vocab_size"]) == int(current_vocab_size) else 0.0
        if int(row["vocab_size"]) == int(current_vocab_size) + 1024:
            conservative_bonus -= 0.01
        row["token_reduction_fraction_vs_current"] = float(token_reduction_fraction)
        row["training_usefulness_score"] = float(token_reduction_fraction - 0.25 * embedding_penalty - 0.10 * artifact_penalty)
        row["conservative_score"] = float(token_reduction_fraction - 0.50 * embedding_penalty - 0.20 * artifact_penalty + conservative_bonus)

    best_by_training = sorted(
        viable,
        key=lambda row: (
            -float(row["training_usefulness_score"]),
            int(row["token_count"]),
            int(row["artifact_bytes"] or 0),
        ),
    )
    best_conservative = sorted(
        viable,
        key=lambda row: (
            -float(row["conservative_score"]),
            abs(int(row["vocab_size"]) - int(current_vocab_size)),
            int(row["artifact_bytes"] or 0),
        ),
    )
    best_by_bytes = sorted(
        viable,
        key=lambda row: (
            float(row["bytes_per_token"]),
            abs(int(row["vocab_size"]) - int(current_vocab_size)),
        ),
    )
    return {
        "available": bool(viable),
        "sample_text_bytes": len(text.encode("utf-8")),
        "sample_text_chars": len(text),
        "output_dir": str(output_dir),
        "current_vocab_size": int(current_vocab_size),
        "current_bytes_per_token": current_bytes_per_token,
        "candidates": results,
        "best_candidates": best_by_training[:8],
        "best_by_training_usefulness": best_by_training[:8],
        "best_conservative_candidates": best_conservative[:8],
        "best_by_bytes_per_token": best_by_bytes[:8],
    }


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
    run_id: str | None
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
    config: dict[str, object]
    artifact_summary: dict[str, object]
    final_eval_summary: dict[str, object]
    metrics_path: str | None


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


def tokenizer_merge_candidates(
    tokens: np.ndarray,
    vocab_size: int,
    tokenizer_meta: list[dict[str, object]] | None,
    k: int,
) -> list[dict[str, object]]:
    if tokenizer_meta is None or tokens.size < 2:
        return []
    pair_ids = tokens[:-1].astype(np.int64) * vocab_size + tokens[1:].astype(np.int64)
    counts = np.bincount(pair_ids, minlength=vocab_size * vocab_size)
    ranked = np.argsort(counts)[::-1]
    out: list[dict[str, object]] = []
    seen_merged: set[str] = set()
    for flat in ranked:
        count = int(counts[flat])
        if count <= 1:
            break
        left = int(flat // vocab_size)
        right = int(flat % vocab_size)
        left_meta = tokenizer_meta[left]
        right_meta = tokenizer_meta[right]
        if not bool(left_meta["usable"]) or not bool(right_meta["usable"]):
            continue
        if bool(right_meta["leading_space"]):
            continue
        left_rendered = str(left_meta["rendered"])
        right_rendered = str(right_meta["rendered"])
        if not left_rendered or not right_rendered:
            continue
        merged_rendered = left_rendered + right_rendered
        if merged_rendered in seen_merged:
            continue
        seen_merged.add(merged_rendered)
        merged_byte_len = len(merged_rendered.encode("utf-8"))
        out.append(
            {
                "left": left,
                "right": right,
                "count": count,
                "left_piece": str(left_meta["piece"]),
                "right_piece": str(right_meta["piece"]),
                "left_rendered": left_rendered,
                "right_rendered": right_rendered,
                "merged_rendered": merged_rendered,
                "merged_byte_len": merged_byte_len,
                "estimated_token_reduction": count,
                "byte_mass": int(count * merged_byte_len),
            }
        )
        if len(out) >= k:
            break
    return out


def tokenizer_budget_analysis(
    candidates: list[dict[str, object]],
    budget_options: list[int],
    model_dim: int,
    current_vocab_size: int,
    submission_limit_bytes: int,
    code_size_bytes: int,
    current_artifact_bytes: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ranked = sorted(candidates, key=lambda item: int(item["byte_mass"]), reverse=True)
    for add_tokens in budget_options:
        if add_tokens <= 0:
            continue
        chosen = ranked[:add_tokens]
        extra_embedding_bytes = int(add_tokens * model_dim * 2)
        projected_artifact = None if current_artifact_bytes is None else int(current_artifact_bytes + extra_embedding_bytes)
        projected_headroom = None if projected_artifact is None else int(submission_limit_bytes - code_size_bytes - projected_artifact)
        rows.append(
            {
                "extra_vocab_tokens": int(add_tokens),
                "projected_vocab_size": int(current_vocab_size + add_tokens),
                "candidate_count_used": int(len(chosen)),
                "estimated_token_reduction": int(sum(int(item["estimated_token_reduction"]) for item in chosen)),
                "estimated_byte_mass_covered": int(sum(int(item["byte_mass"]) for item in chosen)),
                "extra_embedding_bytes_fp16_tied": extra_embedding_bytes,
                "projected_artifact_bytes_excluding_code": projected_artifact,
                "projected_submission_headroom_after_code": projected_headroom,
                "fits_submission_limit_estimate": None if projected_headroom is None else bool(projected_headroom >= 0),
                "top_merged_rendered": [str(item["merged_rendered"]) for item in chosen[: min(8, len(chosen))]],
            }
        )
    return rows


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


def _fit_histogram_kmeans(features: np.ndarray, num_clusters: int, seed: int = 1337, num_iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array")
    rows = features.shape[0]
    k = max(1, min(int(num_clusters), rows))
    if rows == 1:
        return np.zeros((1,), dtype=np.int64), features.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(rows, size=k, replace=False)
    centroids = features[init_idx].astype(np.float32, copy=True)
    assignments = np.zeros((rows,), dtype=np.int64)
    for _ in range(max(int(num_iters), 1)):
        distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1).astype(np.int64, copy=False)
        next_centroids = np.zeros_like(centroids)
        for cluster_idx in range(k):
            mask = assignments == cluster_idx
            if np.any(mask):
                next_centroids[cluster_idx] = features[mask].mean(axis=0)
            else:
                next_centroids[cluster_idx] = features[int(rng.integers(rows))]
        denom = np.clip(next_centroids.sum(axis=1, keepdims=True), 1e-9, None)
        centroids = (next_centroids / denom).astype(np.float32, copy=False)
    return assignments.astype(np.int64, copy=False), centroids.astype(np.float32, copy=False)


def shard_cluster_profile(
    files: list[Path],
    vocab_size: int,
    sample_tokens_per_shard: int,
    prefix_tokens: int,
    cluster_ks: list[int],
    neighbor_topk: int,
    top_k: int,
) -> dict[str, object]:
    if not files:
        return {
            "aggregate": {
                "num_shards_profiled": 0,
                "sample_tokens_per_shard": int(sample_tokens_per_shard),
                "prefix_tokens": int(prefix_tokens),
            },
            "summaries": [],
            "nearest_pairs": [],
            "farthest_pairs": [],
            "cluster_sweeps": [],
        }

    summary_rows: list[dict[str, object]] = []
    sampled_counts_rows: list[np.ndarray] = []
    sampled_hist_rows: list[np.ndarray] = []
    prefix_counts_rows: list[np.ndarray] = []
    prefix_hist_rows: list[np.ndarray] = []
    entropy_values: list[float] = []
    prefix_js_values: list[float] = []
    pair_rows: list[tuple[float, int, int]] = []
    js_matrix = np.zeros((len(files), len(files)), dtype=np.float32)

    for file in files:
        sampled = sample_tokens_from_shard(file, sample_tokens_per_shard)
        prefix = read_shard_prefix_tokens(file, prefix_tokens)
        sampled_counts = np.bincount(sampled, minlength=vocab_size).astype(np.int64)
        prefix_counts = np.bincount(prefix, minlength=vocab_size).astype(np.int64)
        sampled_hist = sampled_counts.astype(np.float32)
        sampled_hist /= max(float(sampled_hist.sum()), 1.0)
        prefix_hist = prefix_counts.astype(np.float32)
        prefix_hist /= max(float(prefix_hist.sum()), 1.0)
        entropy_bits = entropy_from_counts(sampled_counts)
        prefix_js = js_divergence_bits(prefix_counts, sampled_counts)
        entropy_values.append(float(entropy_bits))
        prefix_js_values.append(float(prefix_js))
        sampled_counts_rows.append(sampled_counts)
        sampled_hist_rows.append(sampled_hist)
        prefix_counts_rows.append(prefix_counts)
        prefix_hist_rows.append(prefix_hist)
        summary_rows.append(
            {
                "file": str(file),
                "sample_tokens": int(sampled.size),
                "prefix_tokens": int(prefix.size),
                "entropy_bits": float(entropy_bits),
                "gini": float(gini_from_counts(sampled_counts)),
                "prefix_js_to_sampled_bits": float(prefix_js),
                "top_tokens": top_tokens(sampled_counts, top_k),
            }
        )

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            js = float(js_divergence_bits(sampled_counts_rows[i], sampled_counts_rows[j]))
            js_matrix[i, j] = js
            js_matrix[j, i] = js
            pair_rows.append((js, i, j))

    nearest_values: list[float] = []
    for i, row in enumerate(summary_rows):
        if len(files) <= 1:
            row["nearest_neighbor_file"] = None
            row["nearest_neighbor_js_bits"] = None
            continue
        order = np.argsort(js_matrix[i])
        neighbor_idx = next((int(idx) for idx in order if int(idx) != i), None)
        if neighbor_idx is None:
            row["nearest_neighbor_file"] = None
            row["nearest_neighbor_js_bits"] = None
            continue
        nearest_js = float(js_matrix[i, neighbor_idx])
        nearest_values.append(nearest_js)
        row["nearest_neighbor_file"] = str(files[neighbor_idx])
        row["nearest_neighbor_js_bits"] = nearest_js

    pair_rows_sorted = sorted(pair_rows, key=lambda item: item[0])
    nearest_pairs = [
        {
            "js_divergence_bits": float(js),
            "left_file": str(files[i]),
            "right_file": str(files[j]),
        }
        for js, i, j in pair_rows_sorted[: max(int(neighbor_topk), 0)]
    ]
    farthest_pairs = [
        {
            "js_divergence_bits": float(js),
            "left_file": str(files[i]),
            "right_file": str(files[j]),
        }
        for js, i, j in reversed(pair_rows_sorted[-max(int(neighbor_topk), 0) :])
    ]

    sampled_hists = np.stack(sampled_hist_rows, axis=0).astype(np.float32, copy=False)
    prefix_hists = np.stack(prefix_hist_rows, axis=0).astype(np.float32, copy=False)
    cluster_sweeps: list[dict[str, object]] = []
    for cluster_k in cluster_ks:
        if cluster_k <= 0:
            continue
        assignments, centroids = _fit_histogram_kmeans(sampled_hists, cluster_k)
        actual_k = int(centroids.shape[0])
        prefix_scores = prefix_hists @ centroids.T
        prefix_assignments = prefix_scores.argmax(axis=1).astype(np.int64, copy=False)
        if actual_k > 1:
            top2 = np.partition(prefix_scores, kth=actual_k - 2, axis=1)[:, -2:]
            prefix_margins = top2[:, 1] - top2[:, 0]
        else:
            prefix_margins = np.ones((prefix_scores.shape[0],), dtype=np.float32)
        within_cluster_js = [
            float(js_divergence_bits(sampled_hists[idx], centroids[int(assignments[idx])])) for idx in range(sampled_hists.shape[0])
        ]
        centroid_pair_js: list[float] = []
        for i in range(actual_k):
            for j in range(i + 1, actual_k):
                centroid_pair_js.append(float(js_divergence_bits(centroids[i], centroids[j])))
        cluster_rows = []
        counts_by_cluster = np.bincount(assignments, minlength=actual_k).astype(np.int64)
        for cluster_idx in range(actual_k):
            members = np.flatnonzero(assignments == cluster_idx)
            cluster_rows.append(
                {
                    "cluster_id": int(cluster_idx),
                    "num_shards": int(members.size),
                    "member_files_preview": [str(files[int(member)]) for member in members[: min(6, members.size)]],
                    "mean_prefix_js_to_sampled_bits": float(np.mean([prefix_js_values[int(member)] for member in members]))
                    if members.size
                    else 0.0,
                    "mean_entropy_bits": float(np.mean([entropy_values[int(member)] for member in members])) if members.size else 0.0,
                }
            )
        cluster_sweeps.append(
            {
                "requested_k": int(cluster_k),
                "actual_k": int(actual_k),
                "cluster_counts": [int(v) for v in counts_by_cluster.tolist()],
                "prefix_assignment_match_rate": float(np.mean(prefix_assignments == assignments)),
                "mean_prefix_assignment_margin": float(np.mean(prefix_margins)) if prefix_margins.size else 0.0,
                "mean_within_cluster_js_bits": float(np.mean(within_cluster_js)) if within_cluster_js else 0.0,
                "max_within_cluster_js_bits": float(np.max(within_cluster_js)) if within_cluster_js else 0.0,
                "mean_centroid_separation_bits": float(np.mean(centroid_pair_js)) if centroid_pair_js else 0.0,
                "min_centroid_separation_bits": float(np.min(centroid_pair_js)) if centroid_pair_js else 0.0,
                "clusters": cluster_rows,
            }
        )

    aggregate = {
        "num_shards_profiled": int(len(files)),
        "sample_tokens_per_shard": int(sample_tokens_per_shard),
        "prefix_tokens": int(prefix_tokens),
        "mean_entropy_bits": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "std_entropy_bits": float(np.std(entropy_values)) if entropy_values else 0.0,
        "mean_prefix_js_to_sampled_bits": float(np.mean(prefix_js_values)) if prefix_js_values else 0.0,
        "max_prefix_js_to_sampled_bits": float(np.max(prefix_js_values)) if prefix_js_values else 0.0,
        "mean_nearest_neighbor_js_bits": float(np.mean(nearest_values)) if nearest_values else 0.0,
        "max_nearest_neighbor_js_bits": float(np.max(nearest_values)) if nearest_values else 0.0,
        "min_pairwise_js_bits": float(pair_rows_sorted[0][0]) if pair_rows_sorted else 0.0,
        "max_pairwise_js_bits": float(pair_rows_sorted[-1][0]) if pair_rows_sorted else 0.0,
    }
    return {
        "aggregate": aggregate,
        "summaries": summary_rows,
        "nearest_pairs": nearest_pairs,
        "farthest_pairs": farthest_pairs,
        "cluster_sweeps": cluster_sweeps,
    }


def shard_phase_drift_profile(
    files: list[Path],
    vocab_size: int,
    sample_tokens_per_shard: int,
    prefix_tokens: int,
    phase_segments: int,
) -> dict[str, object]:
    if not files:
        return {
            "aggregate": {
                "num_shards_profiled": 0,
                "phase_segments": int(max(phase_segments, 0)),
                "window_tokens": 0,
            },
            "summaries": [],
        }

    phase_segments = max(int(phase_segments), 2)
    window_tokens = max(
        int(prefix_tokens),
        min(max(int(sample_tokens_per_shard) // phase_segments, 1), max(int(prefix_tokens), 2048)),
    )
    summary_rows: list[dict[str, object]] = []
    prefix_hists: list[np.ndarray] = []
    late_hists: list[np.ndarray] = []
    own_late_js_values: list[float] = []
    consecutive_js_values: list[float] = []
    pairwise_js_values: list[float] = []
    entropy_std_values: list[float] = []

    for file in files:
        header = np.fromfile(file, dtype="<i4", count=HEADER_INTS)
        if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
            raise ValueError(f"Unexpected shard header for {file}")
        num_tokens = int(header[2])
        if num_tokens <= 0:
            continue
        actual_window = min(window_tokens, num_tokens)
        max_start = max(num_tokens - actual_window, 0)
        starts = np.linspace(0, max_start, num=phase_segments, dtype=np.int64)
        segment_hists: list[np.ndarray] = []
        segment_entropies: list[float] = []
        for start in starts:
            segment = read_shard_window_tokens(file, int(start), actual_window)
            counts = np.bincount(segment, minlength=vocab_size).astype(np.int64)
            hist = counts.astype(np.float32)
            hist /= max(float(hist.sum()), 1.0)
            segment_hists.append(hist)
            segment_entropies.append(float(entropy_from_counts(counts)))

        if not segment_hists:
            continue
        prefix_hist = segment_hists[0]
        late_hist = segment_hists[-1]
        prefix_hists.append(prefix_hist)
        late_hists.append(late_hist)
        own_late_js = float(js_divergence_bits(prefix_hist, late_hist))
        own_late_js_values.append(own_late_js)

        consecutive = [
            float(js_divergence_bits(segment_hists[idx], segment_hists[idx + 1]))
            for idx in range(len(segment_hists) - 1)
        ]
        pairwise = [
            float(js_divergence_bits(segment_hists[left], segment_hists[right]))
            for left in range(len(segment_hists))
            for right in range(left + 1, len(segment_hists))
        ]
        consecutive_js_values.extend(consecutive)
        pairwise_js_values.extend(pairwise)
        entropy_std = float(np.std(segment_entropies)) if segment_entropies else 0.0
        entropy_std_values.append(entropy_std)
        summary_rows.append(
            {
                "file": str(file),
                "num_tokens": int(num_tokens),
                "window_tokens": int(actual_window),
                "segment_starts": [int(v) for v in starts.tolist()],
                "segment_entropy_bits": [float(v) for v in segment_entropies],
                "prefix_to_late_js_bits": own_late_js,
                "mean_consecutive_segment_js_bits": float(np.mean(consecutive)) if consecutive else 0.0,
                "mean_pairwise_segment_js_bits": float(np.mean(pairwise)) if pairwise else 0.0,
                "segment_entropy_std_bits": entropy_std,
            }
        )

    nearest_other_late_js_values: list[float] = []
    self_advantage_values: list[float] = []
    for idx, row in enumerate(summary_rows):
        if len(late_hists) <= 1:
            row["nearest_other_late_file"] = None
            row["nearest_other_late_js_bits"] = None
            row["prefix_self_advantage_bits"] = None
            continue
        distances = [
            (float(js_divergence_bits(prefix_hists[idx], late_hists[other_idx])), other_idx)
            for other_idx in range(len(late_hists))
            if other_idx != idx
        ]
        nearest_js, nearest_idx = min(distances, key=lambda item: item[0])
        own_late_js = own_late_js_values[idx]
        advantage = float(nearest_js - own_late_js)
        nearest_other_late_js_values.append(float(nearest_js))
        self_advantage_values.append(advantage)
        row["nearest_other_late_file"] = summary_rows[nearest_idx]["file"]
        row["nearest_other_late_js_bits"] = float(nearest_js)
        row["prefix_self_advantage_bits"] = advantage

    aggregate = {
        "num_shards_profiled": int(len(summary_rows)),
        "phase_segments": int(phase_segments),
        "window_tokens": int(window_tokens),
        "mean_prefix_to_late_js_bits": float(np.mean(own_late_js_values)) if own_late_js_values else 0.0,
        "max_prefix_to_late_js_bits": float(np.max(own_late_js_values)) if own_late_js_values else 0.0,
        "mean_consecutive_segment_js_bits": float(np.mean(consecutive_js_values)) if consecutive_js_values else 0.0,
        "mean_pairwise_segment_js_bits": float(np.mean(pairwise_js_values)) if pairwise_js_values else 0.0,
        "mean_segment_entropy_std_bits": float(np.mean(entropy_std_values)) if entropy_std_values else 0.0,
        "mean_nearest_other_late_js_bits": float(np.mean(nearest_other_late_js_values)) if nearest_other_late_js_values else 0.0,
        "mean_prefix_self_advantage_bits": float(np.mean(self_advantage_values)) if self_advantage_values else 0.0,
        "positive_self_advantage_rate": float(np.mean(np.asarray(self_advantage_values) > 0.0))
        if self_advantage_values
        else 0.0,
    }
    return {
        "aggregate": aggregate,
        "summaries": summary_rows,
    }


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


def lagged_pair_counts(tokens: np.ndarray, vocab_size: int, lag: int) -> np.ndarray | None:
    lag = max(int(lag), 1)
    if tokens.size <= lag:
        return None
    pair_ids = tokens[:-lag].astype(np.int64) * vocab_size + tokens[lag:].astype(np.int64)
    return np.bincount(pair_ids, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size).astype(np.float64)


def _component_token_entries(
    component: np.ndarray,
    token_ids: np.ndarray,
    top_k: int,
    tokenizer_meta: list[dict[str, object]] | None,
    descending: bool,
) -> list[dict[str, object]]:
    if component.size == 0 or token_ids.size == 0 or top_k <= 0:
        return []
    order = np.argsort(component)
    if descending:
        order = order[::-1]
    entries: list[dict[str, object]] = []
    for local_idx in order[:top_k]:
        token_id = int(token_ids[int(local_idx)])
        entry: dict[str, object] = {
            "token_id": token_id,
            "weight": float(component[int(local_idx)]),
        }
        if tokenizer_meta is not None and 0 <= token_id < len(tokenizer_meta):
            meta = tokenizer_meta[token_id]
            piece = meta.get("piece")
            rendered = meta.get("rendered")
            if isinstance(piece, str):
                entry["piece"] = piece
            if isinstance(rendered, str):
                entry["rendered"] = rendered
        entries.append(entry)
    return entries


def _orient_component_sign(component: np.ndarray) -> np.ndarray:
    if component.size == 0:
        return component
    max_idx = int(np.argmax(np.abs(component)))
    if component[max_idx] < 0:
        return -component
    return component


def _build_spectral_basis_summary(
    basis_name: str,
    full_basis: np.ndarray,
    eigenvalues: np.ndarray,
    active_token_ids: np.ndarray,
    top_k: int,
    tokenizer_meta: list[dict[str, object]] | None,
) -> dict[str, object]:
    components: list[dict[str, object]] = []
    for comp_idx in range(full_basis.shape[1]):
        component = _orient_component_sign(full_basis[:, comp_idx].astype(np.float64, copy=False))
        components.append(
            {
                "component_index": int(comp_idx),
                "eigenvalue": float(eigenvalues[comp_idx]),
                "top_positive_tokens": _component_token_entries(
                    component[active_token_ids],
                    active_token_ids,
                    top_k,
                    tokenizer_meta,
                    descending=True,
                ),
                "top_negative_tokens": _component_token_entries(
                    component[active_token_ids],
                    active_token_ids,
                    top_k,
                    tokenizer_meta,
                    descending=False,
                ),
            }
        )
    positive_eigs = np.clip(eigenvalues, a_min=0.0, a_max=None)
    energy = float(positive_eigs.sum())
    return {
        "name": basis_name,
        "rank": int(full_basis.shape[1]),
        "active_tokens": int(active_token_ids.size),
        "eigenvalues": [float(v) for v in eigenvalues.tolist()],
        "positive_eigenvalue_energy": energy,
        "components": components,
    }


def _spectral_eigenbasis_from_pair_counts(
    pair_counts: np.ndarray,
    vocab_size: int,
    rank: int,
    top_k: int,
    tokenizer_meta: list[dict[str, object]] | None,
    *,
    basis_prefix: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sym_counts = pair_counts + pair_counts.T
    active_token_ids = np.flatnonzero(sym_counts.sum(axis=1) > 0)
    if active_token_ids.size < 2:
        return ({"available": False, "reason": "insufficient_active_tokens", "rank": int(rank)}, {})

    active_sym_counts = sym_counts[np.ix_(active_token_ids, active_token_ids)]
    target_rank = min(rank, int(active_token_ids.size))
    if target_rank <= 0:
        return ({"available": False, "reason": "zero_target_rank", "rank": int(rank)}, {})

    degree = active_sym_counts.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    normalized_transition = inv_sqrt_degree[:, None] * active_sym_counts * inv_sqrt_degree[None, :]
    norm_eigs, norm_vecs = np.linalg.eigh(normalized_transition)
    norm_order = np.argsort(norm_eigs)[::-1][:target_rank]
    norm_eigs = norm_eigs[norm_order].astype(np.float32, copy=False)
    norm_vecs = norm_vecs[:, norm_order].astype(np.float32, copy=False)

    total_mass = float(active_sym_counts.sum())
    p_i = degree / max(total_mass, 1e-12)
    expected = np.outer(p_i, p_i)
    observed = active_sym_counts / max(total_mass, 1e-12)
    with np.errstate(divide="ignore"):
        ppmi = np.log(np.clip(observed, 1e-12, None) / np.clip(expected, 1e-12, None))
    ppmi[~np.isfinite(ppmi)] = 0.0
    ppmi = np.maximum(ppmi, 0.0).astype(np.float64, copy=False)
    ppmi_eigs, ppmi_vecs = np.linalg.eigh(ppmi)
    ppmi_order = np.argsort(ppmi_eigs)[::-1][:target_rank]
    ppmi_eigs = ppmi_eigs[ppmi_order].astype(np.float32, copy=False)
    ppmi_vecs = ppmi_vecs[:, ppmi_order].astype(np.float32, copy=False)

    full_norm_basis = np.zeros((vocab_size, target_rank), dtype=np.float32)
    full_norm_basis[active_token_ids] = norm_vecs
    full_ppmi_basis = np.zeros((vocab_size, target_rank), dtype=np.float32)
    full_ppmi_basis[active_token_ids] = ppmi_vecs

    summary = {
        "available": True,
        "rank": int(target_rank),
        "active_tokens": int(active_token_ids.size),
        "symmetric_transition": _build_spectral_basis_summary(
            f"{basis_prefix}_symmetric_transition",
            full_norm_basis,
            norm_eigs,
            active_token_ids,
            top_k,
            tokenizer_meta,
        ),
        "ppmi": _build_spectral_basis_summary(
            f"{basis_prefix}_ppmi",
            full_ppmi_basis,
            ppmi_eigs,
            active_token_ids,
            top_k,
            tokenizer_meta,
        ),
    }
    arrays = {
        "active_token_ids": active_token_ids.astype(np.int32, copy=False),
        f"{basis_prefix}_symmetric_transition_basis": full_norm_basis,
        f"{basis_prefix}_symmetric_transition_eigenvalues": norm_eigs,
        f"{basis_prefix}_ppmi_basis": full_ppmi_basis,
        f"{basis_prefix}_ppmi_eigenvalues": ppmi_eigs,
    }
    return summary, arrays


def spectral_eigenbasis_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    rank: int,
    top_k: int,
    tokenizer_meta: list[dict[str, object]] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if tokens.size < 2:
        return ({"available": False, "reason": "not_enough_tokens", "rank": int(rank)}, {})
    pair_counts = lagged_pair_counts(tokens, vocab_size, lag=1)
    if pair_counts is None:
        return ({"available": False, "reason": "not_enough_tokens", "rank": int(rank)}, {})
    return _spectral_eigenbasis_from_pair_counts(
        pair_counts,
        vocab_size,
        rank,
        top_k,
        tokenizer_meta,
        basis_prefix="lag1",
    )


def lagged_spectral_eigenbasis_profiles(
    tokens: np.ndarray,
    vocab_size: int,
    rank: int,
    top_k: int,
    lags: list[int],
    tokenizer_meta: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    seen_active_ids = False
    for lag in lags:
        lag = max(int(lag), 1)
        pair_counts = lagged_pair_counts(tokens, vocab_size, lag=lag)
        if pair_counts is None:
            rows.append({"available": False, "lag": int(lag), "rank": int(rank), "reason": "not_enough_tokens"})
            continue
        summary, local_arrays = _spectral_eigenbasis_from_pair_counts(
            pair_counts,
            vocab_size,
            rank,
            top_k,
            tokenizer_meta,
            basis_prefix=f"lag{lag}",
        )
        summary["lag"] = int(lag)
        rows.append(summary)
        for key, value in local_arrays.items():
            if key == "active_token_ids":
                if not seen_active_ids:
                    arrays[key] = value
                    seen_active_ids = True
                continue
            arrays[key] = value
    return rows, arrays


def spectral_basis_recommendations(
    base_profile: dict[str, object],
    lagged_profiles: list[dict[str, object]],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    seen_keys: set[tuple[int, str]] = set()
    for item in [base_profile, *lagged_profiles]:
        if not bool(item.get("available")):
            continue
        lag = int(item.get("lag", 1))
        rank = int(item.get("rank", 0))
        for basis_key, basis_name in (("symmetric_transition", "symmetric_transition"), ("ppmi", "ppmi")):
            dedupe_key = (lag, basis_name)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            sub = item.get(basis_key)
            if not isinstance(sub, dict):
                continue
            eigs = [float(v) for v in sub.get("eigenvalues", [])]
            if not eigs:
                continue
            eig0 = abs(eigs[0])
            eig1 = abs(eigs[1]) if len(eigs) > 1 else 0.0
            eigengap = eig0 - eig1
            energy = float(sub.get("positive_eigenvalue_energy", 0.0))
            score = eigengap + 0.02 * energy - 0.01 * max(lag - 1, 0)
            candidates.append(
                {
                    "lag": int(lag),
                    "basis": basis_name,
                    "rank": int(rank),
                    "eigengap": float(eigengap),
                    "positive_energy": energy,
                    "score": float(score),
                }
            )
    candidates.sort(key=lambda row: (-float(row["score"]), int(row["lag"]), row["basis"]))
    best = candidates[0] if candidates else None
    recommendation = None
    if best is not None:
        recommendation = {
            "basis": str(best["basis"]),
            "lag": int(best["lag"]),
            "rank": int(best["rank"]),
            "trainer_hint": {
                "use_fixed_spectral_logit_adapter": True,
                "fixed_spectral_logit_rank": int(best["rank"]),
                "fixed_spectral_logit_map": "pmi_eigen" if str(best["basis"]) == "ppmi" else "transition_eigen",
                "spectral_profile_lag": int(best["lag"]),
            },
        }
    return {
        "candidates": candidates[:8],
        "recommended_basis": recommendation,
    }


def _build_context_distribution_lookup(
    context_ids: np.ndarray,
    next_tokens: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if context_ids.size <= 0 or next_tokens.size <= 0:
        return None
    sort_idx = np.lexsort((next_tokens, context_ids))
    ctx_sorted = context_ids[sort_idx]
    next_sorted = next_tokens[sort_idx]
    pair_change = np.ones((ctx_sorted.size,), dtype=bool)
    pair_change[1:] = (ctx_sorted[1:] != ctx_sorted[:-1]) | (next_sorted[1:] != next_sorted[:-1])
    pair_starts = np.flatnonzero(pair_change)
    pair_ends = np.append(pair_starts[1:], ctx_sorted.size)
    pair_ctx = ctx_sorted[pair_starts]
    pair_counts = (pair_ends - pair_starts).astype(np.int64, copy=False)
    pair_next = next_sorted[pair_starts]
    ctx_change = np.ones((pair_ctx.size,), dtype=bool)
    ctx_change[1:] = pair_ctx[1:] != pair_ctx[:-1]
    ctx_starts = np.flatnonzero(ctx_change)
    ctx_ends = np.append(ctx_starts[1:], pair_ctx.size)
    return {
        "context_ids": pair_ctx[ctx_starts].astype(np.int64, copy=False),
        "pair_next": pair_next.astype(np.int64, copy=False),
        "pair_counts": pair_counts.astype(np.int64, copy=False),
        "ctx_starts": ctx_starts.astype(np.int64, copy=False),
        "ctx_ends": ctx_ends.astype(np.int64, copy=False),
        "ctx_totals": np.add.reduceat(pair_counts, ctx_starts).astype(np.int64, copy=False),
    }


def _lookup_target_probabilities(
    lookup: dict[str, np.ndarray],
    context_ids: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    probs = np.zeros((context_ids.size,), dtype=np.float64)
    if context_ids.size <= 0 or targets.size <= 0:
        return probs
    unique_ctx = lookup["context_ids"]
    pair_next = lookup["pair_next"]
    pair_counts = lookup["pair_counts"]
    ctx_starts = lookup["ctx_starts"]
    ctx_ends = lookup["ctx_ends"]
    ctx_totals = lookup["ctx_totals"]
    pos = np.searchsorted(unique_ctx, context_ids)
    seen = (pos < unique_ctx.size) & (unique_ctx[pos] == context_ids)
    for idx in np.flatnonzero(seen):
        ctx_idx = int(pos[idx])
        start = int(ctx_starts[ctx_idx])
        end = int(ctx_ends[ctx_idx])
        local_next = pair_next[start:end]
        local_counts = pair_counts[start:end]
        target = int(targets[idx])
        j = int(np.searchsorted(local_next, target))
        if j < local_next.size and int(local_next[j]) == target:
            probs[idx] = float(local_counts[j] / max(int(ctx_totals[ctx_idx]), 1))
    return probs


def _lookup_context_posteriors(
    lookup: dict[str, np.ndarray],
    context_ids: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    matrix = np.zeros((context_ids.size, vocab_size), dtype=np.float32)
    if context_ids.size <= 0:
        return matrix
    unique_ctx = lookup["context_ids"]
    pair_next = lookup["pair_next"]
    pair_counts = lookup["pair_counts"].astype(np.float32, copy=False)
    ctx_starts = lookup["ctx_starts"]
    ctx_ends = lookup["ctx_ends"]
    ctx_totals = lookup["ctx_totals"].astype(np.float32, copy=False)
    pos = np.searchsorted(unique_ctx, context_ids)
    seen = (pos < unique_ctx.size) & (unique_ctx[pos] == context_ids)
    for row_idx in np.flatnonzero(seen):
        ctx_idx = int(pos[row_idx])
        start = int(ctx_starts[ctx_idx])
        end = int(ctx_ends[ctx_idx])
        total = float(ctx_totals[ctx_idx])
        if total <= 0.0:
            continue
        matrix[row_idx, pair_next[start:end]] = pair_counts[start:end] / total
    return matrix


def oracle_backoff_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    base_bytes: np.ndarray | None = None,
    orders: list[int] | None = None,
    max_eval_tokens: int = 65536,
) -> dict[str, object]:
    requested_orders = sorted({int(v) for v in (orders or [2, 3, 4, 5, 6]) if 0 < int(v) <= 6})
    if tokens.size < 4 or not requested_orders:
        return {"available": False, "reason": "not_enough_tokens_or_orders"}
    total_count = float(np.maximum(counts.astype(np.float64, copy=False), 0.0).sum())
    if total_count <= 0.0:
        return {"available": False, "reason": "empty_unigram_counts"}
    unigram_probs = np.maximum(counts.astype(np.float64, copy=False), 0.0) / total_count
    max_order = max(requested_orders)
    if tokens.size <= max_order:
        return {"available": False, "reason": "not_enough_tokens_for_max_order"}
    eval_positions = np.arange(max_order, tokens.size, dtype=np.int64)
    sample_count = min(int(max_eval_tokens), int(eval_positions.size))
    sample_positions = (
        np.linspace(max_order, tokens.size - 1, num=sample_count, dtype=np.int64)
        if sample_count < eval_positions.size
        else eval_positions
    )
    target_tokens = tokens[sample_positions].astype(np.int64, copy=False)
    order_cache: dict[int, dict[str, np.ndarray]] = {}
    for order in requested_orders:
        transitions = _build_context_transitions(tokens, vocab_size, order)
        if transitions is None:
            continue
        order_context_ids, order_next_tokens = transitions
        lookup = _build_context_distribution_lookup(order_context_ids, order_next_tokens)
        if lookup is None:
            continue
        aligned_context_ids = order_context_ids[(sample_positions - order).astype(np.int64, copy=False)]
        target_probs = _lookup_target_probabilities(lookup, aligned_context_ids, target_tokens)
        order_cache[order] = {
            "context_ids": aligned_context_ids.astype(np.int64, copy=False),
            "target_probs": target_probs.astype(np.float64, copy=False),
        }
    if not order_cache:
        return {"available": False, "reason": "no_order_cache"}

    rows: list[dict[str, object]] = []
    for max_backoff_order in requested_orders:
        probs = np.clip(unigram_probs[target_tokens].astype(np.float64, copy=False), 1e-12, None)
        selected_orders = np.zeros((target_tokens.size,), dtype=np.int64)
        for order in sorted([o for o in requested_orders if o <= max_backoff_order], reverse=True):
            cached = order_cache.get(order)
            if cached is None:
                continue
            target_probs = cached["target_probs"]
            use_mask = (selected_orders == 0) & (target_probs > 0.0)
            if np.any(use_mask):
                probs[use_mask] = np.clip(target_probs[use_mask], 1e-12, None)
                selected_orders[use_mask] = int(order)
        surprisals = -np.log2(np.clip(probs, 1e-12, None))
        bits_per_byte = _bits_per_byte_from_surprisals(surprisals, target_tokens, base_bytes)
        order_usage = {
            f"order_{order}_rate": float(np.mean(selected_orders == order))
            for order in sorted([o for o in requested_orders if o <= max_backoff_order], reverse=True)
        }
        rows.append(
            {
                "max_order": int(max_backoff_order),
                "sampled_eval_tokens": int(target_tokens.size),
                "entropy_bits": float(surprisals.mean()) if surprisals.size > 0 else 0.0,
                "bits_per_byte": bits_per_byte,
                "exact_order_usage_rate": float(np.mean(selected_orders > 0)),
                "fallback_unigram_rate": float(np.mean(selected_orders == 0)),
                **order_usage,
            }
        )
    best_entropy = min(rows, key=lambda row: float(row["entropy_bits"]), default=None)
    best_bpb = min(
        [row for row in rows if row.get("bits_per_byte") is not None],
        key=lambda row: float(row["bits_per_byte"]),
        default=None,
    )
    return {
        "available": True,
        "rows": rows,
        "best_by_entropy_bits": best_entropy,
        "best_by_bits_per_byte": best_bpb,
        "max_sampled_eval_tokens": int(target_tokens.size),
        "notes": [
            "This backoff profile is an optimistic offline oracle on the sampled stream: context counts are built from the full profiled prefix, so it is stronger than a strict online evaluator.",
            "Use it as an upper bound on how much exact local context memory is present before deciding whether to distill it into a train-time bias head.",
        ],
    }


def _sketch_token_tables(vocab_size: int, sketch_dim: int) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.arange(vocab_size, dtype=np.int64)
    bucket_ids = ((token_ids * np.int64(1103515245) + np.int64(12345)) % np.int64(max(sketch_dim, 1))).astype(
        np.int64, copy=False
    )
    sign_bits = ((token_ids * np.int64(214013) + np.int64(2531011)) >> np.int64(4)) & np.int64(1)
    signs = np.where(sign_bits == 0, 1.0, -1.0).astype(np.float32, copy=False)
    return bucket_ids, signs


def _window_sketch_matrix(
    tokens: np.ndarray,
    positions: np.ndarray,
    window: int,
    sketch_dim: int,
    bucket_ids: np.ndarray,
    signs: np.ndarray,
    direction: str,
) -> np.ndarray:
    matrix = np.zeros((positions.size, sketch_dim), dtype=np.float32)
    if positions.size <= 0 or window <= 0 or sketch_dim <= 0:
        return matrix
    for row_idx, pos in enumerate(positions.astype(np.int64, copy=False)):
        if direction == "past":
            start = max(int(pos) - window, 0)
            end = int(pos)
        else:
            start = int(pos)
            end = min(int(pos) + window, int(tokens.size))
        window_tokens = tokens[start:end].astype(np.int64, copy=False)
        if window_tokens.size <= 0:
            continue
        local_buckets = bucket_ids[window_tokens]
        local_weights = signs[window_tokens]
        sketch = np.bincount(local_buckets, weights=local_weights, minlength=sketch_dim).astype(np.float32, copy=False)
        matrix[row_idx] = sketch / max(int(window_tokens.size), 1)
    return matrix


def _compute_cca(
    x: np.ndarray,
    y: np.ndarray,
    ranks: list[int],
    regularization: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if x.size <= 0 or y.size <= 0 or x.shape[0] != y.shape[0]:
        return {"available": False, "reason": "invalid_cca_inputs"}, {}
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    n = max(int(x.shape[0]), 1)
    cxx = (xc.T @ xc) / n
    cyy = (yc.T @ yc) / n
    cxy = (xc.T @ yc) / n
    reg_x = cxx + np.eye(cxx.shape[0], dtype=np.float32) * float(regularization)
    reg_y = cyy + np.eye(cyy.shape[0], dtype=np.float32) * float(regularization)
    eig_x, vec_x = np.linalg.eigh(reg_x.astype(np.float64, copy=False))
    eig_y, vec_y = np.linalg.eigh(reg_y.astype(np.float64, copy=False))
    keep_x = eig_x > 1e-9
    keep_y = eig_y > 1e-9
    eig_x = eig_x[keep_x]
    eig_y = eig_y[keep_y]
    vec_x = vec_x[:, keep_x]
    vec_y = vec_y[:, keep_y]
    if eig_x.size <= 0 or eig_y.size <= 0:
        return {"available": False, "reason": "degenerate_covariance"}, {}
    invsqrt_x = (vec_x / np.sqrt(eig_x)[None, :]) @ vec_x.T
    invsqrt_y = (vec_y / np.sqrt(eig_y)[None, :]) @ vec_y.T
    tmat = invsqrt_x @ cxy @ invsqrt_y
    u, s, vt = np.linalg.svd(tmat.astype(np.float64, copy=False), full_matrices=False)
    canonical_corrs = np.clip(s.astype(np.float32, copy=False), 0.0, 1.0)
    past_basis = (invsqrt_x @ u).astype(np.float32, copy=False)
    future_basis = (invsqrt_y @ vt.T).astype(np.float32, copy=False)
    safe_ranks = sorted({int(r) for r in ranks if int(r) > 0})
    rows: list[dict[str, object]] = []
    cumulative = np.cumsum(canonical_corrs.astype(np.float64, copy=False))
    max_cumulative = float(cumulative[min(max(safe_ranks or [1]) - 1, cumulative.size - 1)]) if cumulative.size > 0 else 0.0
    best_rank = None
    if canonical_corrs.size > 0 and safe_ranks:
        target = 0.95 * max_cumulative
        for rank in safe_ranks:
            capped = min(rank, int(canonical_corrs.size))
            if capped <= 0:
                continue
            cumulative_corr = float(cumulative[capped - 1])
            mean_corr = float(canonical_corrs[:capped].mean())
            row = {
                "rank": int(capped),
                "mean_canonical_correlation": mean_corr,
                "cumulative_canonical_correlation": cumulative_corr,
                "fraction_of_max_cumulative": float(cumulative_corr / max(max_cumulative, 1e-12)),
            }
            rows.append(row)
            if best_rank is None and cumulative_corr >= target:
                best_rank = row
    if best_rank is None and rows:
        best_rank = rows[-1]
    return (
        {
            "available": True,
            "rows": rows,
            "best_rank_by_cumulative_correlation": best_rank,
            "top_canonical_correlations": [float(v) for v in canonical_corrs[: min(16, canonical_corrs.size)].tolist()],
        },
        {
            "past_future_cca_canonical_correlations": canonical_corrs[: max(safe_ranks or [0], default=0)].astype(np.float32, copy=False),
            "past_future_cca_past_basis": past_basis[:, : max(safe_ranks or [0], default=0)].astype(np.float32, copy=False),
            "past_future_cca_future_basis": future_basis[:, : max(safe_ranks or [0], default=0)].astype(np.float32, copy=False),
            "past_future_cca_past_mean": x_mean.astype(np.float32, copy=False),
            "past_future_cca_future_mean": y_mean.astype(np.float32, copy=False),
        },
    )


def past_future_cca_profile(
    tokens: np.ndarray,
    vocab_size: int,
    past_window: int,
    future_window: int,
    sketch_dim: int,
    ranks: list[int],
    max_prefixes: int,
    regularization: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    max_offset = max(int(past_window), int(future_window))
    if tokens.size <= max_offset + 1:
        return {"available": False, "reason": "not_enough_tokens"}, {}
    start = int(past_window)
    end = int(tokens.size - future_window)
    if end <= start:
        return {"available": False, "reason": "invalid_window_geometry"}, {}
    total_positions = end - start
    sample_count = min(max(int(max_prefixes), 1), total_positions)
    positions = np.linspace(start, end - 1, num=sample_count, dtype=np.int64)
    bucket_ids, signs = _sketch_token_tables(vocab_size, sketch_dim)
    x = _window_sketch_matrix(tokens, positions, past_window, sketch_dim, bucket_ids, signs, direction="past")
    y = _window_sketch_matrix(tokens, positions, future_window, sketch_dim, bucket_ids, signs, direction="future")
    cca_summary, cca_arrays = _compute_cca(x, y, ranks, regularization)
    if not bool(cca_summary.get("available")):
        return cca_summary, cca_arrays
    cca_summary.update(
        {
            "past_window": int(past_window),
            "future_window": int(future_window),
            "sketch_dim": int(sketch_dim),
            "sampled_prefixes": int(positions.size),
        }
    )
    return cca_summary, cca_arrays


def _simple_kmeans(points: np.ndarray, k: int, iters: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, points.shape[1]), dtype=np.float32)
    k = min(max(int(k), 1), int(points.shape[0]))
    init_idx = np.linspace(0, points.shape[0] - 1, num=k, dtype=np.int64)
    centroids = points[init_idx].astype(np.float32, copy=True)
    labels = np.zeros((points.shape[0],), dtype=np.int64)
    for _ in range(max(int(iters), 1)):
        dists = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1).astype(np.int64, copy=False)
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                centroids[idx] = points[mask].mean(axis=0)
    return labels, centroids


def _assign_kmeans(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 0 or centroids.shape[0] <= 0:
        return np.zeros((points.shape[0],), dtype=np.int64)
    dists = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1).astype(np.int64, copy=False)


def predictive_state_transfer_spectrum(
    tokens: np.ndarray,
    vocab_size: int,
    past_window: int,
    future_window: int,
    sketch_dim: int,
    ranks: list[int],
    max_prefixes: int,
    regularization: float,
    taus: list[int],
    num_clusters: int,
    kmeans_iters: int,
) -> dict[str, object]:
    valid_taus = sorted({int(t) for t in taus if int(t) > 0})
    if not valid_taus:
        return {"available": False, "reason": "no_taus"}
    max_tau = max(valid_taus)
    start = int(past_window)
    end = int(tokens.size - future_window - max_tau)
    if end <= start:
        return {"available": False, "reason": "not_enough_tokens"}
    sample_count = min(max(int(max_prefixes), 2), end - start)
    positions = np.linspace(start, end - 1, num=sample_count, dtype=np.int64)
    bucket_ids, signs = _sketch_token_tables(vocab_size, sketch_dim)
    x = _window_sketch_matrix(tokens, positions, past_window, sketch_dim, bucket_ids, signs, direction="past")
    y = _window_sketch_matrix(tokens, positions, future_window, sketch_dim, bucket_ids, signs, direction="future")
    cca_summary, cca_arrays = _compute_cca(x, y, ranks, regularization)
    if not bool(cca_summary.get("available")):
        return {"available": False, "reason": str(cca_summary.get("reason", "cca_failed"))}
    best_rank = cca_summary.get("best_rank_by_cumulative_correlation")
    if not isinstance(best_rank, dict):
        return {"available": False, "reason": "no_cca_rank"}
    rank = int(best_rank.get("rank", 0))
    if rank <= 0:
        return {"available": False, "reason": "invalid_cca_rank"}
    past_basis = cca_arrays["past_future_cca_past_basis"][:, :rank]
    past_mean = cca_arrays["past_future_cca_past_mean"]
    state_t = (x - past_mean) @ past_basis
    labels_t, centroids = _simple_kmeans(state_t.astype(np.float32, copy=False), num_clusters, kmeans_iters)
    rows: list[dict[str, object]] = []
    for tau in valid_taus:
        next_positions = positions + int(tau)
        x_next = _window_sketch_matrix(tokens, next_positions, past_window, sketch_dim, bucket_ids, signs, direction="past")
        state_next = (x_next - past_mean) @ past_basis
        labels_next = _assign_kmeans(state_next.astype(np.float32, copy=False), centroids)
        cluster_count = int(centroids.shape[0])
        trans_counts = np.zeros((cluster_count, cluster_count), dtype=np.float64)
        for src, dst in zip(labels_t.tolist(), labels_next.tolist(), strict=True):
            trans_counts[int(src), int(dst)] += 1.0
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        transition = np.divide(trans_counts, np.maximum(row_sums, 1e-12), out=np.zeros_like(trans_counts), where=row_sums > 0)
        eigvals = np.linalg.eigvals(transition.T)
        eig_abs = np.sort(np.abs(eigvals))[::-1]
        lambda2 = float(eig_abs[1]) if eig_abs.size > 1 else 0.0
        spectral_gap = float(max(1.0 - lambda2, 0.0))
        stationary = trans_counts.sum(axis=1)
        stationary = stationary / max(float(stationary.sum()), 1e-12)
        sorted_mass = np.sort(stationary)[::-1]
        rows.append(
            {
                "tau": int(tau),
                "state_rank": int(rank),
                "num_clusters": int(cluster_count),
                "second_eigenvalue_abs": lambda2,
                "spectral_gap": spectral_gap,
                "mixing_time_proxy": float(1.0 / max(spectral_gap, 1e-6)),
                "stationary_top1_mass": float(sorted_mass[0]) if sorted_mass.size > 0 else 0.0,
                "stationary_top2_mass": float(sorted_mass[:2].sum()) if sorted_mass.size > 1 else (float(sorted_mass[0]) if sorted_mass.size > 0 else 0.0),
            }
        )
    best = max(
        rows,
        key=lambda row: (float(row["second_eigenvalue_abs"]) - 0.25 * float(row["stationary_top1_mass"]), -int(row["tau"])),
        default=None,
    )
    return {
        "available": True,
        "past_window": int(past_window),
        "future_window": int(future_window),
        "sketch_dim": int(sketch_dim),
        "sampled_prefixes": int(positions.size),
        "rows": rows,
        "best_tau_by_slow_mode": best,
    }


def _chunk_start_positions(total_tokens: int, chunk_tokens: int, chunk_stride: int) -> np.ndarray:
    if total_tokens <= 0:
        return np.zeros((0,), dtype=np.int64)
    safe_chunk = max(int(chunk_tokens), 1)
    safe_stride = max(int(chunk_stride), 1)
    if total_tokens <= safe_chunk:
        return np.asarray([0], dtype=np.int64)
    starts = np.arange(0, total_tokens - safe_chunk + 1, safe_stride, dtype=np.int64)
    last_start = int(total_tokens - safe_chunk)
    if starts.size <= 0 or int(starts[-1]) != last_start:
        starts = np.append(starts, np.int64(last_start))
    return starts.astype(np.int64, copy=False)


def _transition_summary_from_labels(labels: np.ndarray, num_states: int) -> tuple[np.ndarray, dict[str, float]]:
    cluster_count = max(int(num_states), 1)
    if labels.size <= 1:
        transition = np.eye(cluster_count, dtype=np.float64)
        return transition, {
            "self_transition_mass": 1.0,
            "second_eigenvalue_abs": 0.0,
            "spectral_gap": 1.0,
            "mixing_time_proxy": 1.0,
            "stationary_top1_mass": 1.0,
        }
    trans_counts = np.zeros((cluster_count, cluster_count), dtype=np.float64)
    for src, dst in zip(labels[:-1].tolist(), labels[1:].tolist(), strict=True):
        trans_counts[int(src), int(dst)] += 1.0
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    transition = np.divide(trans_counts, np.maximum(row_sums, 1e-12), out=np.zeros_like(trans_counts), where=row_sums > 0)
    eigvals = np.linalg.eigvals(transition.T)
    eig_abs = np.sort(np.abs(eigvals))[::-1]
    lambda2 = float(eig_abs[1]) if eig_abs.size > 1 else 0.0
    spectral_gap = float(max(1.0 - lambda2, 0.0))
    stationary = trans_counts.sum(axis=1)
    stationary = stationary / max(float(stationary.sum()), 1e-12)
    sorted_mass = np.sort(stationary)[::-1]
    return transition, {
        "self_transition_mass": float(np.mean(labels[:-1] == labels[1:])),
        "second_eigenvalue_abs": lambda2,
        "spectral_gap": spectral_gap,
        "mixing_time_proxy": float(1.0 / max(spectral_gap, 1e-6)),
        "stationary_top1_mass": float(sorted_mass[0]) if sorted_mass.size > 0 else 0.0,
    }


def dataset_world_model_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    past_window: int,
    future_window: int,
    sketch_dim: int,
    ranks: list[int],
    max_prefixes: int,
    regularization: float,
    chunk_tokens: int,
    chunk_stride: int,
    prefix_tokens: int,
    regime_counts: list[int],
    kmeans_iters: int,
    top_tokens: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    valid_regime_counts = sorted({int(v) for v in regime_counts if int(v) > 0})
    if not valid_regime_counts:
        return {"available": False, "reason": "no_regime_counts"}, {}
    start = int(past_window)
    end = int(tokens.size - future_window)
    if end <= start:
        return {"available": False, "reason": "not_enough_tokens"}, {}
    sample_count = min(max(int(max_prefixes), 2), end - start)
    positions = np.linspace(start, end - 1, num=sample_count, dtype=np.int64)
    bucket_ids, signs = _sketch_token_tables(vocab_size, sketch_dim)
    x = _window_sketch_matrix(tokens, positions, past_window, sketch_dim, bucket_ids, signs, direction="past")
    y = _window_sketch_matrix(tokens, positions, future_window, sketch_dim, bucket_ids, signs, direction="future")
    cca_summary, cca_arrays = _compute_cca(x, y, ranks, regularization)
    if not bool(cca_summary.get("available")):
        return {"available": False, "reason": str(cca_summary.get("reason", "cca_failed"))}, {}
    best_rank = cca_summary.get("best_rank_by_cumulative_correlation")
    if not isinstance(best_rank, dict):
        return {"available": False, "reason": "no_cca_rank"}, {}
    rank = int(best_rank.get("rank", 0))
    if rank <= 0:
        return {"available": False, "reason": "invalid_cca_rank"}, {}

    past_basis = cca_arrays["past_future_cca_past_basis"][:, :rank]
    past_mean = cca_arrays["past_future_cca_past_mean"]
    state_vectors = (x - past_mean) @ past_basis

    chunk_starts = _chunk_start_positions(int(tokens.size), int(chunk_tokens), int(chunk_stride))
    chunk_rows: list[dict[str, object]] = []
    chunk_means: list[np.ndarray] = []
    chunk_stds: list[np.ndarray] = []
    chunk_count_rows: list[np.ndarray] = []
    chunk_prefix_hist_rows: list[np.ndarray] = []
    kept_chunk_starts: list[int] = []
    global_probs = np.maximum(counts.astype(np.float64, copy=False), 0.0)
    global_probs = global_probs / max(float(global_probs.sum()), 1e-12)
    safe_top_tokens = max(int(top_tokens), 1)
    safe_prefix = max(1, min(int(prefix_tokens), int(chunk_tokens)))

    for chunk_idx, chunk_start in enumerate(chunk_starts.tolist()):
        chunk_end = min(int(chunk_start) + int(chunk_tokens), int(tokens.size))
        mask = (positions >= int(chunk_start)) & (positions < int(chunk_end))
        if not np.any(mask):
            continue
        local_states = state_vectors[mask]
        local_counts = np.bincount(tokens[int(chunk_start) : int(chunk_end)], minlength=vocab_size).astype(np.float64, copy=False)
        total = float(local_counts.sum())
        if total <= 0.0:
            continue
        local_probs = local_counts / total
        prefix_counts = np.bincount(
            tokens[int(chunk_start) : min(int(chunk_start) + safe_prefix, int(chunk_end))],
            minlength=vocab_size,
        ).astype(np.float64, copy=False)
        prefix_total = float(prefix_counts.sum())
        prefix_hist = (prefix_counts / max(prefix_total, 1.0)).astype(np.float32, copy=False)
        lift = np.divide(local_probs, np.maximum(global_probs, 1e-12), out=np.zeros_like(local_probs), where=global_probs > 0.0)
        score = local_probs * np.log2(np.maximum(lift, 1e-12))
        top_ids = np.argsort(score)[::-1][:safe_top_tokens]
        kept_chunk_starts.append(int(chunk_start))
        chunk_means.append(local_states.mean(axis=0).astype(np.float32, copy=False))
        chunk_stds.append(local_states.std(axis=0).astype(np.float32, copy=False))
        chunk_count_rows.append(local_counts.astype(np.float32, copy=False))
        chunk_prefix_hist_rows.append(prefix_hist)
        chunk_rows.append(
            {
                "chunk_index": int(chunk_idx),
                "chunk_start": int(chunk_start),
                "chunk_end": int(chunk_end),
                "positions_covered": int(local_states.shape[0]),
                "state_norm": float(np.linalg.norm(local_states.mean(axis=0))),
                "state_std_mean": float(local_states.std(axis=0).mean()) if local_states.shape[0] > 0 else 0.0,
                "token_entropy_bits": entropy_from_counts(local_counts.astype(np.int64, copy=False)),
                "top_token_mass": float(local_probs.max()) if local_probs.size > 0 else 0.0,
                "prefix_tokens_for_classifier": int(safe_prefix),
                "top_tokens_by_regime_score": [
                    {
                        "token_id": int(token_id),
                        "chunk_prob": float(local_probs[int(token_id)]),
                        "global_prob": float(global_probs[int(token_id)]),
                        "lift": float(lift[int(token_id)]),
                    }
                    for token_id in top_ids.tolist()
                    if local_probs[int(token_id)] > 0.0
                ],
            }
        )

    if not chunk_rows:
        return {"available": False, "reason": "no_chunk_states"}, {}

    chunk_mean_array = np.stack(chunk_means, axis=0).astype(np.float32, copy=False)
    chunk_std_array = np.stack(chunk_stds, axis=0).astype(np.float32, copy=False)
    chunk_count_array = np.stack(chunk_count_rows, axis=0).astype(np.float32, copy=False)
    chunk_prefix_hist_array = np.stack(chunk_prefix_hist_rows, axis=0).astype(np.float32, copy=False)
    chunk_embeddings = np.concatenate([chunk_mean_array, chunk_std_array], axis=1).astype(np.float32, copy=False)

    regime_rows: list[dict[str, object]] = []
    best_world = None
    best_labels = None
    best_centroids = None
    best_transition = None
    best_metrics = None
    for regime_count in valid_regime_counts:
        labels, centroids = _simple_kmeans(chunk_embeddings, regime_count, kmeans_iters)
        if centroids.shape[0] <= 0:
            continue
        residual = chunk_embeddings - centroids[labels]
        within_mse = float(np.square(residual).mean())
        centroid_centered = centroids - centroids.mean(axis=0, keepdims=True)
        centroid_spread = float(np.square(centroid_centered).mean())
        transition, metrics = _transition_summary_from_labels(labels, int(centroids.shape[0]))
        score = float(
            centroid_spread / max(within_mse, 1e-9)
            + 0.5 * metrics["second_eigenvalue_abs"]
            + 0.25 * metrics["self_transition_mass"]
            - 0.25 * metrics["stationary_top1_mass"]
        )
        row = {
            "num_regimes": int(centroids.shape[0]),
            "within_regime_mse": within_mse,
            "centroid_spread": centroid_spread,
            "self_transition_mass": float(metrics["self_transition_mass"]),
            "second_eigenvalue_abs": float(metrics["second_eigenvalue_abs"]),
            "spectral_gap": float(metrics["spectral_gap"]),
            "mixing_time_proxy": float(metrics["mixing_time_proxy"]),
            "stationary_top1_mass": float(metrics["stationary_top1_mass"]),
            "selection_score": score,
        }
        regime_rows.append(row)
        if best_world is None or score > float(best_world["selection_score"]):
            best_world = row
            best_labels = labels.astype(np.int64, copy=False)
            best_centroids = centroids.astype(np.float32, copy=False)
            best_transition = transition.astype(np.float32, copy=False)
            best_metrics = metrics

    if best_world is None or best_labels is None or best_centroids is None or best_transition is None or best_metrics is None:
        return {"available": False, "reason": "no_regime_fit"}, {}

    occupancy = np.bincount(best_labels, minlength=int(best_centroids.shape[0])).astype(np.float64, copy=False)
    occupancy = occupancy / max(float(occupancy.sum()), 1e-12)
    chunk_entropy = np.asarray([float(row["token_entropy_bits"]) for row in chunk_rows], dtype=np.float64)
    chunk_top_mass = np.asarray([float(row["top_token_mass"]) for row in chunk_rows], dtype=np.float64)
    regime_summaries: list[dict[str, object]] = []
    curriculum_weights: list[dict[str, object]] = []
    route_priors: list[dict[str, object]] = []
    for regime_id in range(int(best_centroids.shape[0])):
        mask = best_labels == regime_id
        regime_chunk_count = int(np.sum(mask))
        if regime_chunk_count <= 0:
            continue
        regime_counts_sum = chunk_count_array[mask].sum(axis=0).astype(np.float64, copy=False)
        regime_probs = regime_counts_sum / max(float(regime_counts_sum.sum()), 1e-12)
        regime_lift = np.divide(regime_probs, np.maximum(global_probs, 1e-12), out=np.zeros_like(regime_probs), where=global_probs > 0.0)
        regime_score = regime_probs * np.log2(np.maximum(regime_lift, 1e-12))
        top_ids = np.argsort(regime_score)[::-1][:safe_top_tokens]
        mean_entropy = float(chunk_entropy[mask].mean()) if np.any(mask) else 0.0
        mean_top_mass = float(chunk_top_mass[mask].mean()) if np.any(mask) else 0.0
        rarity = float(1.0 - occupancy[regime_id])
        entropy_center = float(chunk_entropy.mean()) if chunk_entropy.size > 0 else 0.0
        difficulty = mean_entropy - entropy_center
        curriculum_weight = float(np.clip(1.0 + 0.20 * rarity + 0.08 * difficulty, 0.75, 1.35))
        route_temp = float(np.clip(1.0 + 0.15 * difficulty - 0.10 * mean_top_mass, 0.85, 1.25))
        curriculum_weights.append(
            {
                "regime_id": int(regime_id),
                "curriculum_weight": curriculum_weight,
                "coverage_fraction": float(occupancy[regime_id]),
            }
        )
        route_priors.append(
            {
                "regime_id": int(regime_id),
                "route_temperature_multiplier": route_temp,
                "regime_confidence_proxy": float(np.clip(mean_top_mass, 0.0, 1.0)),
            }
        )
        regime_summaries.append(
            {
                "regime_id": int(regime_id),
                "chunk_count": regime_chunk_count,
                "coverage_fraction": float(occupancy[regime_id]),
                "mean_chunk_entropy_bits": mean_entropy,
                "mean_chunk_top_token_mass": mean_top_mass,
                "mean_state_norm": float(np.linalg.norm(chunk_mean_array[mask].mean(axis=0))),
                "top_tokens_by_lift": [
                    {
                        "token_id": int(token_id),
                        "regime_prob": float(regime_probs[int(token_id)]),
                        "global_prob": float(global_probs[int(token_id)]),
                        "lift": float(regime_lift[int(token_id)]),
                    }
                    for token_id in top_ids.tolist()
                    if regime_probs[int(token_id)] > 0.0
                ],
            }
        )

    arrays = {
        "dataset_world_model_chunk_state_means": chunk_mean_array,
        "dataset_world_model_chunk_state_stds": chunk_std_array,
        "dataset_world_model_chunk_embeddings": chunk_embeddings.astype(np.float32, copy=False),
        "dataset_world_model_chunk_starts": np.asarray(kept_chunk_starts, dtype=np.int64),
        "dataset_world_model_chunk_token_counts": chunk_count_array.astype(np.float32, copy=False),
        "dataset_world_model_chunk_prefix_hists": chunk_prefix_hist_array.astype(np.float32, copy=False),
        "dataset_world_model_regime_assignments": best_labels.astype(np.int64, copy=False),
        "dataset_world_model_regime_centroids": best_centroids.astype(np.float32, copy=False),
        "dataset_world_model_regime_transition": best_transition.astype(np.float32, copy=False),
    }
    return (
        {
            "available": True,
            "chunk_predictive_state_encoder": {
                "source": "past_future_cca",
                "past_window": int(past_window),
                "future_window": int(future_window),
                "sketch_dim": int(sketch_dim),
                "state_rank": int(rank),
                "sampled_prefixes": int(positions.size),
                "chunk_tokens": int(chunk_tokens),
                "chunk_stride": int(chunk_stride),
                "prefix_tokens": int(safe_prefix),
                "num_chunks": int(chunk_embeddings.shape[0]),
                "mean_chunk_state_norm": float(np.linalg.norm(chunk_mean_array, axis=1).mean()) if chunk_mean_array.size > 0 else 0.0,
            },
            "global_regime_model": {
                "regime_sweep": regime_rows,
                "best_regime_model": {
                    **best_world,
                    "num_chunks": int(chunk_embeddings.shape[0]),
                    "transition_matrix_shape": [int(best_transition.shape[0]), int(best_transition.shape[1])],
                },
                "regime_summaries": regime_summaries,
            },
            "trainer_distilled_hints": {
                "prefix_causal_only": True,
                "source": "offline_chunk_predictive_state_world_model",
                "trainer_hint": {
                    "use_dataset_regime_prefix_classifier": True,
                    "dataset_regime_prefix_tokens": int(safe_prefix),
                    "dataset_regime_chunk_tokens": int(chunk_tokens),
                    "dataset_regime_chunk_stride": int(chunk_stride),
                    "dataset_regime_state_rank": int(rank),
                    "dataset_regime_count": int(best_centroids.shape[0]),
                    "dataset_regime_source": "offline_chunk_predictive_state_world_model",
                },
                "curriculum_weights": curriculum_weights,
                "route_priors": route_priors,
            },
            "chunk_summaries": chunk_rows,
            "notes": [
                "This world-model layer is an offline universal coarse-graining over chunked predictive-state summaries.",
                "Use the emitted trainer hints only through prefix-causal chunk classifiers or regime-conditioned priors; the full offline regime assignment is clairvoyant.",
            ],
            "selection_metrics": {
                "second_eigenvalue_abs": float(best_metrics["second_eigenvalue_abs"]),
                "spectral_gap": float(best_metrics["spectral_gap"]),
                "self_transition_mass": float(best_metrics["self_transition_mass"]),
            },
        },
        arrays,
    )


def ppm_oracle_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    base_bytes: np.ndarray | None = None,
    orders: list[int] | None = None,
    max_eval_tokens: int = 65536,
) -> dict[str, object]:
    requested_orders = sorted({int(v) for v in (orders or [2, 3, 4, 5, 6]) if int(v) > 0})
    if tokens.size < 4 or not requested_orders:
        return {"available": False, "reason": "not_enough_tokens_or_orders"}
    max_order = max(requested_orders)
    if tokens.size <= max_order:
        return {"available": False, "reason": "not_enough_tokens_for_max_order"}
    total_count = float(np.maximum(counts.astype(np.float64, copy=False), 0.0).sum())
    if total_count <= 0.0:
        return {"available": False, "reason": "empty_unigram_counts"}
    unigram_probs = np.maximum(counts.astype(np.float64, copy=False), 0.0) / total_count
    eval_positions = np.arange(max_order, tokens.size, dtype=np.int64)
    sample_count = min(int(max_eval_tokens), int(eval_positions.size))
    sample_positions = (
        np.linspace(max_order, tokens.size - 1, num=sample_count, dtype=np.int64)
        if sample_count < eval_positions.size
        else eval_positions
    )
    target_tokens = tokens[sample_positions].astype(np.int64, copy=False)
    lookup_by_order: dict[int, dict[str, np.ndarray]] = {}
    aligned_context_ids: dict[int, np.ndarray] = {}
    for order in range(1, max_order + 1):
        transitions = _build_context_transitions(tokens, vocab_size, order)
        if transitions is None:
            continue
        context_ids, next_tokens = transitions
        lookup = _build_context_distribution_lookup(context_ids, next_tokens)
        if lookup is None:
            continue
        lookup_by_order[order] = lookup
        aligned_context_ids[order] = context_ids[(sample_positions - order).astype(np.int64, copy=False)]
    if 1 not in lookup_by_order:
        return {"available": False, "reason": "missing_order1_lookup"}
    rows: list[dict[str, object]] = []
    for max_backoff_order in requested_orders:
        probs = np.zeros((target_tokens.size,), dtype=np.float64)
        for idx in range(target_tokens.size):
            target = int(target_tokens[idx])
            p = float(np.clip(unigram_probs[target], 1e-12, None))
            for order in range(1, max_backoff_order + 1):
                lookup = lookup_by_order.get(order)
                if lookup is None:
                    continue
                ctx = int(aligned_context_ids[order][idx])
                unique_ctx = lookup["context_ids"]
                pos = int(np.searchsorted(unique_ctx, ctx))
                if pos >= unique_ctx.size or int(unique_ctx[pos]) != ctx:
                    continue
                start = int(lookup["ctx_starts"][pos])
                end = int(lookup["ctx_ends"][pos])
                local_next = lookup["pair_next"][start:end]
                local_counts = lookup["pair_counts"][start:end]
                total = int(lookup["ctx_totals"][pos])
                unique_children = max(int(end - start), 1)
                j = int(np.searchsorted(local_next, target))
                if j < local_next.size and int(local_next[j]) == target:
                    p = float(local_counts[j] / max(total + unique_children, 1))
                else:
                    p = float(unique_children / max(total + unique_children, 1) * p)
            probs[idx] = max(p, 1e-12)
        surprisals = -np.log2(np.clip(probs, 1e-12, None))
        rows.append(
            {
                "max_order": int(max_backoff_order),
                "sampled_eval_tokens": int(target_tokens.size),
                "entropy_bits": float(surprisals.mean()) if surprisals.size > 0 else 0.0,
                "bits_per_byte": _bits_per_byte_from_surprisals(surprisals, target_tokens, base_bytes),
            }
        )
    best_entropy = min(rows, key=lambda row: float(row["entropy_bits"]), default=None)
    best_bpb = min(
        [row for row in rows if row.get("bits_per_byte") is not None],
        key=lambda row: float(row["bits_per_byte"]),
        default=None,
    )
    return {
        "available": True,
        "rows": rows,
        "best_by_entropy_bits": best_entropy,
        "best_by_bits_per_byte": best_bpb,
        "max_sampled_eval_tokens": int(target_tokens.size),
        "notes": [
            "This is a bounded-order PPM-style Witten-Bell backoff oracle over the sampled stream.",
            "It is a stronger causal compression proxy than exact n-gram lookup, but it is still an offline profiler object rather than a submission-time evaluator.",
        ],
    }


def _fit_ridge_token_readout(
    states: np.ndarray,
    targets: np.ndarray,
    vocab_size: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_mean = states.mean(axis=0, keepdims=True)
    centered_states = states - state_mean
    target_probs = np.bincount(targets.astype(np.int64, copy=False), minlength=vocab_size).astype(np.float64, copy=False)
    target_probs /= max(float(target_probs.sum()), 1.0)
    centered_x = centered_states.astype(np.float64, copy=False)
    xtx = centered_x.T @ centered_x
    rank = int(states.shape[1])
    xty = np.zeros((rank, vocab_size), dtype=np.float64)
    for dim in range(rank):
        np.add.at(xty[dim], targets.astype(np.int64, copy=False), centered_x[:, dim])
    weights = np.linalg.solve(xtx + np.eye(rank, dtype=np.float64) * float(ridge), xty)
    bias_logits = np.log(np.clip(target_probs, 1e-12, None))
    return weights.astype(np.float32, copy=False), bias_logits.astype(np.float32, copy=False), state_mean.astype(np.float32, copy=False)


def _cross_entropy_bits_from_scores(
    scores: np.ndarray,
    targets: np.ndarray,
    base_bytes: np.ndarray | None = None,
) -> tuple[float, float | None]:
    if scores.size <= 0 or targets.size <= 0:
        return 0.0, None
    row_max = np.max(scores, axis=1, keepdims=True)
    stable = scores - row_max
    logsumexp = np.log(np.exp(stable).sum(axis=1)) + row_max[:, 0]
    target_scores = scores[np.arange(scores.shape[0]), targets.astype(np.int64, copy=False)]
    surprisals = (logsumexp - target_scores) / math.log(2.0)
    bits = float(np.mean(surprisals)) if surprisals.size > 0 else 0.0
    bpb = _bits_per_byte_from_surprisals(surprisals, targets, base_bytes)
    return bits, bpb


def minimal_causal_state_profile(
    tokens: np.ndarray,
    vocab_size: int,
    base_bytes: np.ndarray | None,
    past_window: int,
    future_window: int,
    sketch_dim: int,
    ranks: list[int],
    max_prefixes: int,
    regularization: float,
    holdout_fraction: float,
    ridge: float,
    near_best_bpb_tol: float,
    near_best_bits_tol: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    cca_summary, cca_arrays = past_future_cca_profile(
        tokens,
        vocab_size,
        past_window=past_window,
        future_window=future_window,
        sketch_dim=sketch_dim,
        ranks=ranks,
        max_prefixes=max_prefixes,
        regularization=regularization,
    )
    if not bool(cca_summary.get("available")):
        return {"available": False, "reason": str(cca_summary.get("reason", "cca_unavailable"))}, {}
    max_offset = max(int(past_window), int(future_window))
    start = int(past_window)
    end = int(tokens.size - future_window)
    if end <= start:
        return {"available": False, "reason": "invalid_window_geometry"}, {}
    sample_count = min(max(int(max_prefixes), 8), end - start)
    positions = np.linspace(start, end - 1, num=sample_count, dtype=np.int64)
    bucket_ids, signs = _sketch_token_tables(vocab_size, sketch_dim)
    past_sketches = _window_sketch_matrix(tokens, positions, past_window, sketch_dim, bucket_ids, signs, direction="past")
    targets = tokens[positions].astype(np.int64, copy=False)
    split = int(max(1, min(sample_count - 1, round(sample_count * (1.0 - holdout_fraction)))))
    train_x = past_sketches[:split]
    train_y = targets[:split]
    hold_x = past_sketches[split:]
    hold_y = targets[split:]
    if hold_x.shape[0] <= 0 or train_x.shape[0] <= 0:
        return {"available": False, "reason": "invalid_train_holdout_split"}, {}

    unigram_probs = np.bincount(train_y.astype(np.int64, copy=False), minlength=vocab_size).astype(np.float64, copy=False)
    unigram_probs /= max(float(unigram_probs.sum()), 1.0)
    unigram_scores = np.broadcast_to(np.log(np.clip(unigram_probs, 1e-12, None))[None, :], (hold_y.size, vocab_size))
    unigram_bits, unigram_bpb = _cross_entropy_bits_from_scores(unigram_scores, hold_y, base_bytes)

    raw_weights, raw_bias, raw_mean = _fit_ridge_token_readout(train_x, train_y, vocab_size, ridge)
    raw_scores = (hold_x - raw_mean) @ raw_weights + raw_bias
    raw_bits, raw_bpb = _cross_entropy_bits_from_scores(raw_scores, hold_y, base_bytes)

    past_basis = cca_arrays["past_future_cca_past_basis"]
    past_mean = cca_arrays["past_future_cca_past_mean"]
    safe_ranks = sorted({int(r) for r in ranks if int(r) > 0})
    rows: list[dict[str, object]] = []
    artifacts_by_rank: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    best_bpb_value = math.inf
    best_bits_value = math.inf
    for rank in safe_ranks:
        rank = min(rank, int(past_basis.shape[1]))
        if rank <= 0:
            continue
        basis = past_basis[:, :rank]
        train_states = (train_x - past_mean) @ basis
        hold_states = (hold_x - past_mean) @ basis
        weights, bias, state_mean = _fit_ridge_token_readout(train_states, train_y, vocab_size, ridge)
        hold_scores = (hold_states - state_mean) @ weights + bias
        hold_bits, hold_bpb = _cross_entropy_bits_from_scores(hold_scores, hold_y, base_bytes)
        estimated_fp16_bytes = int(sketch_dim * rank * 2 + vocab_size * rank * 2 + vocab_size * 2)
        row = {
            "rank": int(rank),
            "heldout_cross_entropy_bits": float(hold_bits),
            "heldout_bits_per_byte": hold_bpb,
            "heldout_delta_bits_vs_unigram": float(unigram_bits - hold_bits),
            "heldout_delta_bpb_vs_unigram": None if unigram_bpb is None or hold_bpb is None else float(unigram_bpb - hold_bpb),
            "estimated_fp16_state_bytes": estimated_fp16_bytes,
        }
        rows.append(row)
        artifacts_by_rank[int(rank)] = (weights, bias, state_mean)
        current_bpb = float(hold_bpb) if hold_bpb is not None else math.inf
        if current_bpb < best_bpb_value or (math.isinf(best_bpb_value) and hold_bits < best_bits_value):
            best_bpb_value = current_bpb
            best_bits_value = float(hold_bits)
    best = None
    smallest_near_best = None
    if rows:
        best = min(
            rows,
            key=lambda row: (
                math.inf if row.get("heldout_bits_per_byte") is None else float(row["heldout_bits_per_byte"]),
                float(row["heldout_cross_entropy_bits"]),
                int(row["rank"]),
            ),
        )
        best_bpb = None if best.get("heldout_bits_per_byte") is None else float(best["heldout_bits_per_byte"])
        best_bits = float(best["heldout_cross_entropy_bits"])
        near_rows = []
        for row in rows:
            bpb_ok = (
                best_bpb is None
                or row.get("heldout_bits_per_byte") is None
                or float(row["heldout_bits_per_byte"]) <= best_bpb + float(near_best_bpb_tol)
            )
            bits_ok = float(row["heldout_cross_entropy_bits"]) <= best_bits + float(near_best_bits_tol)
            if bpb_ok and bits_ok:
                near_rows.append(row)
        if near_rows:
            smallest_near_best = min(near_rows, key=lambda row: int(row["rank"]))

    summary = {
        "available": True,
        "past_window": int(past_window),
        "future_window": int(future_window),
        "sketch_dim": int(sketch_dim),
        "sampled_prefixes": int(sample_count),
        "train_prefixes": int(train_x.shape[0]),
        "holdout_prefixes": int(hold_x.shape[0]),
        "unigram_holdout_bits": float(unigram_bits),
        "unigram_holdout_bits_per_byte": unigram_bpb,
        "raw_past_sketch_holdout_bits": float(raw_bits),
        "raw_past_sketch_holdout_bits_per_byte": raw_bpb,
        "rows": rows,
        "best_rank_by_holdout_bpb": best,
        "smallest_near_best_rank": smallest_near_best,
    }
    arrays: dict[str, np.ndarray] = {}
    export_row = smallest_near_best if isinstance(smallest_near_best, dict) else best
    export_rank = None if not isinstance(export_row, dict) else int(export_row.get("rank", 0))
    if export_rank is not None and export_rank in artifacts_by_rank and export_rank > 0:
        best_weights, best_bias, best_state_mean = artifacts_by_rank[export_rank]
        arrays = {
            "minimal_causal_state_past_basis": past_basis[:, :export_rank].astype(np.float32, copy=False),
            "minimal_causal_state_past_mean": past_mean.astype(np.float32, copy=False),
            "minimal_causal_state_readout": best_weights.astype(np.float32, copy=False),
            "minimal_causal_state_bias": best_bias.astype(np.float32, copy=False),
            "minimal_causal_state_state_mean": best_state_mean.astype(np.float32, copy=False),
        }
    return summary, arrays


def _strict_online_minimal_state_batch(
    tokens: np.ndarray,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    minimal_causal_state_arrays: dict[str, np.ndarray],
    max_eval_tokens: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    required = {
        "minimal_causal_state_past_basis",
        "minimal_causal_state_past_mean",
        "minimal_causal_state_readout",
        "minimal_causal_state_bias",
        "minimal_causal_state_state_mean",
    }
    if not required.issubset(minimal_causal_state_arrays.keys()):
        return None
    max_h = 8
    start = int(max(past_window, 1))
    end = int(tokens.size - max_h)
    if end <= start:
        return None
    total_positions = end - start
    sample_count = min(max(int(max_eval_tokens), 1), total_positions)
    positions = (
        np.arange(end - sample_count, end, dtype=np.int64)
        if sample_count < total_positions
        else np.arange(start, end, dtype=np.int64)
    )
    bucket_ids, signs = _sketch_token_tables(vocab_size, sketch_dim)
    past_sketches = _window_sketch_matrix(tokens, positions, past_window, sketch_dim, bucket_ids, signs, direction="past")
    past_basis = minimal_causal_state_arrays["minimal_causal_state_past_basis"]
    past_mean = minimal_causal_state_arrays["minimal_causal_state_past_mean"]
    state_mean = minimal_causal_state_arrays["minimal_causal_state_state_mean"]
    readout = minimal_causal_state_arrays["minimal_causal_state_readout"]
    bias = minimal_causal_state_arrays["minimal_causal_state_bias"]
    states = (past_sketches - past_mean) @ past_basis
    scores = (states - state_mean) @ readout + bias
    targets = tokens[positions].astype(np.int64, copy=False)
    return scores.astype(np.float32, copy=False), targets


def _causal_machine_decoder_bytes(
    decodability: dict[str, object] | None,
    decodability_arrays: dict[str, np.ndarray],
    past_future_cca_arrays: dict[str, np.ndarray],
) -> int:
    weights = np.asarray(decodability_arrays.get("causal_state_decoder_weights"))
    bias = np.asarray(decodability_arrays.get("causal_state_decoder_bias"))
    state_mean = np.asarray(decodability_arrays.get("causal_state_decoder_state_mean"))
    if weights.size <= 0 or bias.size <= 0 or state_mean.size <= 0:
        return 0
    total = int(weights.size + bias.size + state_mean.size) * 2
    best_trainable = None if decodability is None else decodability.get("best_trainable_feature_family")
    if isinstance(best_trainable, dict) and str(best_trainable.get("feature_family")) == "past_cca_linear":
        feature_dim = int(best_trainable.get("feature_dim", 0))
        if feature_dim > 0:
            past_basis = np.asarray(past_future_cca_arrays.get("past_future_cca_past_basis"))
            past_mean = np.asarray(past_future_cca_arrays.get("past_future_cca_past_mean"))
            total += int(past_basis[:, :feature_dim].size + past_mean.size) * 2
    return total


def _strict_online_causal_machine_batch(
    tokens: np.ndarray,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    causal_state_arrays: dict[str, np.ndarray],
    causal_state_decodability: dict[str, object] | None,
    causal_state_decodability_arrays: dict[str, np.ndarray],
    past_future_cca_arrays: dict[str, np.ndarray],
    max_eval_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    required = {
        "causal_machine_signature_centroids",
        "causal_machine_log_probs",
        "causal_state_decoder_weights",
        "causal_state_decoder_bias",
        "causal_state_decoder_state_mean",
    }
    if not required.issubset(set(causal_state_arrays.keys()) | set(causal_state_decodability_arrays.keys())):
        return None
    max_h = 8
    start = int(max(past_window, 1))
    end = int(tokens.size - max_h)
    if end <= start:
        return None
    total_positions = end - start
    sample_count = min(max(int(max_eval_tokens), 1), total_positions)
    positions = (
        np.arange(end - sample_count, end, dtype=np.int64)
        if sample_count < total_positions
        else np.arange(start, end, dtype=np.int64)
    )
    bucket_ids, signs = _sketch_token_tables(vocab_size, sketch_dim)
    past_sketches = _window_sketch_matrix(tokens, positions, past_window, sketch_dim, bucket_ids, signs, direction="past").astype(
        np.float32, copy=False
    )
    best_trainable = None if causal_state_decodability is None else causal_state_decodability.get("best_trainable_feature_family")
    feature_family = str(best_trainable.get("feature_family")) if isinstance(best_trainable, dict) else "past_sketch_linear"
    feature_dim = int(best_trainable.get("feature_dim", 0)) if isinstance(best_trainable, dict) else 0
    decoder_inputs = past_sketches
    if feature_family == "past_cca_linear":
        past_basis = np.asarray(past_future_cca_arrays.get("past_future_cca_past_basis"))
        past_mean = np.asarray(past_future_cca_arrays.get("past_future_cca_past_mean"))
        if past_basis.size <= 0 or past_mean.size <= 0 or feature_dim <= 0:
            return None
        decoder_inputs = (past_sketches - past_mean.astype(np.float32, copy=False)) @ past_basis[:, :feature_dim].astype(
            np.float32, copy=False
        )
    weights = np.asarray(causal_state_decodability_arrays.get("causal_state_decoder_weights")).astype(np.float64, copy=False)
    bias = np.asarray(causal_state_decodability_arrays.get("causal_state_decoder_bias")).astype(np.float64, copy=False)
    state_mean = np.asarray(causal_state_decodability_arrays.get("causal_state_decoder_state_mean")).astype(np.float64, copy=False)
    state_scores = (
        decoder_inputs.astype(np.float64, copy=False) - state_mean
    ) @ weights + bias
    row_max = np.max(state_scores, axis=1, keepdims=True)
    state_probs = np.exp(state_scores - row_max)
    state_probs /= np.maximum(state_probs.sum(axis=1, keepdims=True), 1e-12)
    log_probs = np.asarray(causal_state_arrays.get("causal_machine_log_probs")).astype(np.float64, copy=False)
    token_scores = state_probs @ log_probs
    targets = tokens[positions].astype(np.int64, copy=False)
    return token_scores.astype(np.float32, copy=False), targets, state_probs.astype(np.float32, copy=False)


def strict_online_state_eval_profile(
    tokens: np.ndarray,
    vocab_size: int,
    base_bytes: np.ndarray | None,
    past_window: int,
    sketch_dim: int,
    causal_state_arrays: dict[str, np.ndarray],
    causal_state_decodability: dict[str, object] | None,
    causal_state_decodability_arrays: dict[str, np.ndarray],
    past_future_cca_arrays: dict[str, np.ndarray],
    max_eval_tokens: int,
    horizons: list[int] | None = None,
) -> dict[str, object]:
    batch = _strict_online_causal_machine_batch(
        tokens,
        vocab_size,
        past_window,
        sketch_dim,
        causal_state_arrays,
        causal_state_decodability,
        causal_state_decodability_arrays,
        past_future_cca_arrays,
        max_eval_tokens,
    )
    if batch is None:
        return {"available": False, "reason": "missing_causal_machine_decoder_or_arrays"}
    scores, targets, state_probs = batch
    bits, bpb = _cross_entropy_bits_from_scores(scores, targets, base_bytes)
    probs = state_probs @ np.exp(np.asarray(causal_state_arrays.get("causal_machine_log_probs")).astype(np.float64, copy=False))
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    valid_horizons = sorted({int(h) for h in (horizons or [1, 2, 4, 8]) if int(h) > 0})
    start = int(max(past_window, 1))
    end = int(tokens.size - max(valid_horizons, default=1))
    total_positions = end - start
    sample_count = min(max(int(max_eval_tokens), 1), total_positions)
    positions = (
        np.arange(end - sample_count, end, dtype=np.int64)
        if sample_count < total_positions
        else np.arange(start, end, dtype=np.int64)
    )
    horizon_rows: list[dict[str, object]] = []
    for horizon in valid_horizons:
        future_ce = 0.0
        future_entropy = 0.0
        for row_idx, pos in enumerate(positions):
            window = tokens[pos : pos + horizon].astype(np.int64, copy=False)
            hist = np.bincount(window, minlength=vocab_size).astype(np.float64, copy=False)
            hist /= max(float(hist.sum()), 1.0)
            pred = np.clip(probs[row_idx].astype(np.float64, copy=False), 1e-12, None)
            future_ce += float(-(hist * np.log2(pred)).sum())
            nonzero = hist[hist > 0.0]
            future_entropy += float(-(nonzero * np.log2(nonzero)).sum()) if nonzero.size > 0 else 0.0
        denom = max(int(positions.size), 1)
        horizon_rows.append(
            {
                "horizon": int(horizon),
                "future_sketch_cross_entropy_bits": float(future_ce / denom),
                "future_sketch_entropy_bits": float(future_entropy / denom),
            }
        )
    return {
        "available": True,
        "sampled_eval_tokens": int(targets.size),
        "next_token_cross_entropy_bits": float(bits),
        "next_token_bits_per_byte": bpb,
        "future_horizons": horizon_rows,
    }


def _tt_svd_3(tensor: np.ndarray, bond_dim: int) -> tuple[np.ndarray, int]:
    n0, n1, n2 = tensor.shape
    unfold1 = tensor.reshape(n0, n1 * n2)
    u1, s1, vt1 = np.linalg.svd(unfold1.astype(np.float64, copy=False), full_matrices=False)
    r1 = min(int(bond_dim), int(s1.size))
    core1 = u1[:, :r1]
    tmp = (s1[:r1, None] * vt1[:r1]).reshape(r1 * n1, n2)
    u2, s2, vt2 = np.linalg.svd(tmp.astype(np.float64, copy=False), full_matrices=False)
    r2 = min(int(bond_dim), int(s2.size))
    core2 = u2[:, :r2].reshape(r1, n1, r2)
    core3 = (s2[:r2, None] * vt2[:r2]).reshape(r2, n2)
    recon = np.einsum("ar,rbs,sc->abc", core1, core2, core3, optimize=True)
    est_bytes = int((n0 * r1 + r1 * n1 * r2 + r2 * n2) * 2)
    return recon.astype(np.float32, copy=False), est_bytes


def tensor_network_state_frontier_profile(
    tokens: np.ndarray,
    vocab_size: int,
    base_bytes: np.ndarray | None,
    past_window: int,
    sketch_dim: int,
    causal_state_arrays: dict[str, np.ndarray],
    causal_state_decodability: dict[str, object] | None,
    causal_state_decodability_arrays: dict[str, np.ndarray],
    past_future_cca_arrays: dict[str, np.ndarray],
    max_eval_tokens: int,
    bond_dims: list[int] | None = None,
) -> dict[str, object]:
    batch = _strict_online_causal_machine_batch(
        tokens,
        vocab_size,
        past_window,
        sketch_dim,
        causal_state_arrays,
        causal_state_decodability,
        causal_state_decodability_arrays,
        past_future_cca_arrays,
        max_eval_tokens,
    )
    if batch is None:
        return {"available": False, "reason": "missing_causal_machine_decoder_or_arrays"}
    log_probs = np.asarray(causal_state_arrays.get("causal_machine_log_probs"))
    if log_probs.size <= 0:
        return {"available": False, "reason": "missing_causal_machine_log_probs"}
    side = int(round(math.sqrt(vocab_size)))
    if side * side != vocab_size:
        return {"available": False, "reason": "non_square_vocab_for_tensor_frontier"}
    _, targets, state_probs = batch
    weight_tensor = log_probs.reshape(log_probs.shape[0], side, side)
    decoder_bytes = _causal_machine_decoder_bytes(
        causal_state_decodability,
        causal_state_decodability_arrays,
        past_future_cca_arrays,
    )
    rows: list[dict[str, object]] = []
    for bond_dim in sorted({int(v) for v in (bond_dims or [2, 4, 8, 16]) if int(v) > 0}):
        recon_tensor, est_bytes = _tt_svd_3(weight_tensor, bond_dim)
        recon_readout = recon_tensor.reshape(log_probs.shape[0], vocab_size)
        scores = state_probs.astype(np.float64, copy=False) @ recon_readout.astype(np.float64, copy=False)
        bits, bpb = _cross_entropy_bits_from_scores(scores, targets, base_bytes)
        rows.append(
            {
                "bond_dim": int(bond_dim),
                "heldout_cross_entropy_bits": float(bits),
                "heldout_bits_per_byte": bpb,
                "estimated_fp16_state_bytes": int(est_bytes + decoder_bytes),
            }
        )
    best = min(
        [row for row in rows if row.get("heldout_bits_per_byte") is not None],
        key=lambda row: (float(row["heldout_bits_per_byte"]), int(row["bond_dim"])),
        default=None,
    )
    return {
        "available": True,
        "rows": rows,
        "best_bond_dim_by_holdout_bpb": best,
    }


def candidate_state_frontier_profile(
    anchor_model_bpb: float | None,
    strict_online_state_eval: dict[str, object] | None,
    minimal_causal_state: dict[str, object] | None,
    minimal_causal_state_reco: dict[str, object] | None,
    causal_state_reconstruction: dict[str, object] | None,
    tensor_network_state_frontier: dict[str, object] | None,
    ppm_oracle: dict[str, object] | None,
    oracle_backoff: dict[str, object] | None,
    regime_conditioned_bpb: dict[str, object] | None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if isinstance(strict_online_state_eval, dict) and strict_online_state_eval.get("available"):
        strict_estimated_bytes = None
        if isinstance(causal_state_reconstruction, dict):
            rec = causal_state_reconstruction.get("smallest_near_best_state_count") or causal_state_reconstruction.get(
                "best_state_count_by_holdout_bpb"
            )
            if isinstance(rec, dict):
                strict_estimated_bytes = int(rec.get("estimated_fp16_state_bytes", 0))
        rows.append(
            {
                "component": "causal_machine_strict_online",
                "family": "causal_machine",
                "heldout_bits_per_byte": strict_online_state_eval.get("next_token_bits_per_byte"),
                "estimated_bytes": strict_estimated_bytes,
                "trainable": True,
                "strict_online": True,
            }
        )
    if isinstance(minimal_causal_state_reco, dict) and isinstance(minimal_causal_state_reco.get("recommended_state"), dict):
        rec = minimal_causal_state_reco["recommended_state"]
        rows.append(
            {
                "component": "minimal_causal_state_frontier",
                "family": "causal_state",
                "heldout_bits_per_byte": rec.get("heldout_bits_per_byte"),
                "estimated_bytes": int(rec.get("estimated_fp16_state_bytes", 0)),
                "trainable": True,
                "strict_online": False,
            }
        )
    if isinstance(causal_state_reconstruction, dict) and causal_state_reconstruction.get("available"):
        rec = causal_state_reconstruction.get("smallest_near_best_state_count") or causal_state_reconstruction.get("best_state_count_by_holdout_bpb")
        if isinstance(rec, dict):
            rows.append(
                {
                    "component": f"causal_machine_states_{int(rec.get('num_states', 0))}",
                    "family": "causal_machine",
                    "heldout_bits_per_byte": rec.get("heldout_bits_per_byte"),
                    "estimated_bytes": int(rec.get("estimated_fp16_state_bytes", 0)),
                    "trainable": True,
                    "strict_online": False,
                }
            )
    if isinstance(tensor_network_state_frontier, dict) and tensor_network_state_frontier.get("available"):
        best = tensor_network_state_frontier.get("best_bond_dim_by_holdout_bpb")
        if isinstance(best, dict):
            rows.append(
                {
                    "component": f"causal_machine_tensor_network_bond_{int(best.get('bond_dim', 0))}",
                    "family": "tensor_network",
                    "heldout_bits_per_byte": best.get("heldout_bits_per_byte"),
                    "estimated_bytes": int(best.get("estimated_fp16_state_bytes", 0)),
                    "trainable": True,
                    "strict_online": True,
                }
            )
    if isinstance(regime_conditioned_bpb, dict) and regime_conditioned_bpb.get("available"):
        ablation = regime_conditioned_bpb.get("bpb_ablation") or {}
        rows.append(
            {
                "component": "prefix_regime_prior",
                "family": "world_model",
                "heldout_bits_per_byte": ablation.get("prefix_regime_bits_per_byte"),
                "estimated_bytes": None,
                "trainable": True,
                "strict_online": True,
            }
        )
    if isinstance(oracle_backoff, dict) and oracle_backoff.get("available"):
        best = oracle_backoff.get("best_by_bits_per_byte") or oracle_backoff.get("best_by_entropy_bits")
        if isinstance(best, dict):
            rows.append(
                {
                    "component": f"oracle_backoff_order_{int(best.get('max_order', 0))}",
                    "family": "oracle_backoff",
                    "heldout_bits_per_byte": best.get("bits_per_byte"),
                    "estimated_bytes": None,
                    "trainable": False,
                    "strict_online": False,
                }
            )
    if isinstance(ppm_oracle, dict) and ppm_oracle.get("available"):
        best = ppm_oracle.get("best_by_bits_per_byte") or ppm_oracle.get("best_by_entropy_bits")
        if isinstance(best, dict):
            rows.append(
                {
                    "component": f"ppm_oracle_order_{int(best.get('max_order', 0))}",
                    "family": "ppm_oracle",
                    "heldout_bits_per_byte": best.get("bits_per_byte"),
                    "estimated_bytes": None,
                    "trainable": False,
                    "strict_online": False,
                }
            )
    clean_rows = [row for row in rows if row.get("heldout_bits_per_byte") is not None]
    for row in clean_rows:
        row["delta_bpb_vs_anchor_proxy"] = None if anchor_model_bpb is None else float(anchor_model_bpb - float(row["heldout_bits_per_byte"]))
    clean_rows.sort(
        key=lambda row: (
            float(row["heldout_bits_per_byte"]),
            math.inf if row.get("estimated_bytes") is None else float(row["estimated_bytes"]),
        )
    )
    frontier: list[dict[str, object]] = []
    best_bytes = math.inf
    for row in clean_rows:
        est_bytes = math.inf if row.get("estimated_bytes") is None else float(row["estimated_bytes"])
        if est_bytes < best_bytes:
            frontier.append(row)
            best_bytes = est_bytes
    return {
        "available": True,
        "anchor_model_bpb": anchor_model_bpb,
        "rows": clean_rows,
        "pareto_frontier": frontier,
        "best_trainable": min(
            [row for row in clean_rows if bool(row.get("trainable"))],
            key=lambda row: float(row["heldout_bits_per_byte"]),
            default=None,
        ),
        "best_oracle": min(
            [row for row in clean_rows if not bool(row.get("trainable"))],
            key=lambda row: float(row["heldout_bits_per_byte"]),
            default=None,
        ),
    }


def future_signature_profile(
    tokens: np.ndarray,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    horizons: list[int],
    max_prefixes: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    valid_horizons = sorted({int(h) for h in horizons if int(h) > 0})
    if not valid_horizons:
        return {"available": False, "reason": "no_horizons"}, {}
    max_horizon = max(valid_horizons)
    start = int(max(past_window, 1))
    end = int(tokens.size - max_horizon)
    if end <= start:
        return {"available": False, "reason": "not_enough_tokens"}, {}
    total_positions = end - start
    sample_count = min(max(int(max_prefixes), 1), total_positions)
    positions = (
        np.arange(end - sample_count, end, dtype=np.int64)
        if sample_count < total_positions
        else np.arange(start, end, dtype=np.int64)
    )
    bucket_ids, signs = _sketch_token_tables(vocab_size, sketch_dim)
    signature_parts: list[np.ndarray] = []
    horizon_entropy_cols: list[np.ndarray] = []
    horizon_rows: list[dict[str, object]] = []
    for horizon in valid_horizons:
        future_sketch = _window_sketch_matrix(tokens, positions, horizon, sketch_dim, bucket_ids, signs, direction="future")
        signature_parts.append(future_sketch.astype(np.float32, copy=False))
        entropies = np.zeros((positions.size, 1), dtype=np.float32)
        for row_idx, pos in enumerate(positions.tolist()):
            window = tokens[int(pos) : int(pos) + int(horizon)].astype(np.int64, copy=False)
            counts = np.bincount(window, minlength=vocab_size).astype(np.float64, copy=False)
            total = float(counts.sum())
            if total <= 0.0:
                continue
            probs = counts / total
            nz = probs[probs > 0.0]
            entropies[row_idx, 0] = float(-(nz * np.log2(nz)).sum()) if nz.size > 0 else 0.0
        signature_parts.append(entropies)
        horizon_entropy_cols.append(entropies.astype(np.float32, copy=False))
        horizon_rows.append(
            {
                "horizon": int(horizon),
                "mean_future_entropy_bits": float(entropies.mean()) if entropies.size > 0 else 0.0,
                "max_future_entropy_bits": float(entropies.max()) if entropies.size > 0 else 0.0,
            }
        )
    targets = tokens[positions].astype(np.int64, copy=False)
    signatures = np.concatenate(signature_parts, axis=1) if signature_parts else np.zeros((positions.size, 0), dtype=np.float32)
    entropy_matrix = (
        np.concatenate(horizon_entropy_cols, axis=1).astype(np.float32, copy=False)
        if horizon_entropy_cols
        else np.zeros((positions.size, 0), dtype=np.float32)
    )
    return (
        {
            "available": True,
            "sampled_prefixes": int(positions.size),
            "signature_dim": int(signatures.shape[1]),
            "horizons": [int(h) for h in valid_horizons],
            "horizon_rows": horizon_rows,
        },
        {
            "future_signature_matrix": signatures.astype(np.float32, copy=False),
            "future_signature_positions": positions.astype(np.int64, copy=False),
            "future_signature_targets": targets.astype(np.int64, copy=False),
            "future_signature_horizon_entropies": entropy_matrix.astype(np.float32, copy=False),
            "future_signature_horizons": np.asarray(valid_horizons, dtype=np.int64),
        },
    )


def _causal_state_sample_views(
    tokens: np.ndarray,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    signatures = np.asarray(future_signature_arrays.get("future_signature_matrix"))
    positions = np.asarray(future_signature_arrays.get("future_signature_positions"))
    targets = np.asarray(future_signature_arrays.get("future_signature_targets"))
    centroids = np.asarray(causal_state_arrays.get("causal_machine_signature_centroids"))
    if (
        signatures.size <= 0
        or positions.size <= 0
        or targets.size <= 0
        or centroids.size <= 0
        or signatures.shape[0] != positions.shape[0]
        or signatures.shape[0] != targets.shape[0]
    ):
        return None
    bucket_ids, signs = _sketch_token_tables(vocab_size, sketch_dim)
    past_sketch = _window_sketch_matrix(
        tokens,
        positions.astype(np.int64, copy=False),
        past_window,
        sketch_dim,
        bucket_ids,
        signs,
        direction="past",
    ).astype(np.float32, copy=False)
    labels = _assign_kmeans(
        signatures.astype(np.float32, copy=False),
        centroids.astype(np.float32, copy=False),
    )
    return (
        positions.astype(np.int64, copy=False),
        past_sketch,
        labels.astype(np.int64, copy=False),
        targets.astype(np.int64, copy=False),
        signatures.astype(np.float32, copy=False),
    )


def causal_state_reconstruction_profile(
    future_signature_arrays: dict[str, np.ndarray],
    vocab_size: int,
    base_bytes: np.ndarray | None,
    state_counts: list[int],
    holdout_fraction: float,
    kmeans_iters: int,
    near_best_bpb_tol: float,
    near_best_bits_tol: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    signatures = np.asarray(future_signature_arrays.get("future_signature_matrix"))
    targets = np.asarray(future_signature_arrays.get("future_signature_targets"))
    horizon_entropies = np.asarray(future_signature_arrays.get("future_signature_horizon_entropies"))
    if signatures.size <= 0 or targets.size <= 0 or signatures.shape[0] != targets.shape[0]:
        return {"available": False, "reason": "missing_future_signatures"}, {}
    sample_count = int(signatures.shape[0])
    split = int(max(1, min(sample_count - 1, round(sample_count * (1.0 - holdout_fraction)))))
    train_x = signatures[:split].astype(np.float32, copy=False)
    train_y = targets[:split].astype(np.int64, copy=False)
    hold_x = signatures[split:].astype(np.float32, copy=False)
    hold_y = targets[split:].astype(np.int64, copy=False)
    train_h = horizon_entropies[:split].astype(np.float32, copy=False) if horizon_entropies.size > 0 else np.zeros((train_x.shape[0], 0), dtype=np.float32)
    if hold_x.shape[0] <= 0 or train_x.shape[0] <= 0:
        return {"available": False, "reason": "invalid_train_holdout_split"}, {}
    unigram_probs = np.bincount(train_y, minlength=vocab_size).astype(np.float64, copy=False)
    unigram_probs /= max(float(unigram_probs.sum()), 1.0)
    unigram_surprisals = -np.log2(np.clip(unigram_probs[hold_y], 1e-12, None))
    unigram_bpb = _bits_per_byte_from_surprisals(unigram_surprisals, hold_y, base_bytes)
    rows: list[dict[str, object]] = []
    artifacts_by_count: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    valid_counts = sorted({int(v) for v in state_counts if int(v) > 0})
    for num_states in valid_counts:
        labels_train, centroids = _simple_kmeans(train_x, num_states, kmeans_iters)
        labels_hold = _assign_kmeans(hold_x, centroids)
        state_token_counts = np.zeros((centroids.shape[0], vocab_size), dtype=np.float64)
        for label, target in zip(labels_train.tolist(), train_y.tolist(), strict=True):
            state_token_counts[int(label), int(target)] += 1.0
        smoothed = state_token_counts + 1e-3
        state_probs = smoothed / np.maximum(smoothed.sum(axis=1, keepdims=True), 1e-12)
        hold_probs = state_probs[labels_hold, hold_y]
        hold_surprisals = -np.log2(np.clip(hold_probs, 1e-12, None))
        hold_bits = float(hold_surprisals.mean()) if hold_surprisals.size > 0 else 0.0
        hold_bpb = _bits_per_byte_from_surprisals(hold_surprisals, hold_y, base_bytes)
        state_mass = np.bincount(labels_train, minlength=centroids.shape[0]).astype(np.float64, copy=False)
        state_mass /= max(float(state_mass.sum()), 1e-12)
        state_entropies = np.zeros((centroids.shape[0],), dtype=np.float64)
        mean_future_entropy = np.zeros((centroids.shape[0],), dtype=np.float64)
        for idx in range(centroids.shape[0]):
            probs = state_probs[idx]
            nz = probs[probs > 0.0]
            state_entropies[idx] = float(-(nz * np.log2(nz)).sum()) if nz.size > 0 else 0.0
            mask = labels_train == idx
            if np.any(mask) and train_h.shape[1] > 0:
                mean_future_entropy[idx] = float(train_h[mask].mean())
        estimated_bytes = int(centroids.size * 2 + state_probs.size * 2 + state_mass.size * 2)
        row = {
            "num_states": int(centroids.shape[0]),
            "heldout_cross_entropy_bits": float(hold_bits),
            "heldout_bits_per_byte": hold_bpb,
            "heldout_delta_bpb_vs_unigram": None if unigram_bpb is None or hold_bpb is None else float(unigram_bpb - hold_bpb),
            "conditional_next_token_entropy_bits": float(np.dot(state_mass, state_entropies)),
            "mean_future_entropy_bits": float(np.dot(state_mass, mean_future_entropy)) if mean_future_entropy.size > 0 else 0.0,
            "mean_state_top1_mass": float(np.dot(state_mass, np.max(state_probs, axis=1))),
            "estimated_fp16_state_bytes": estimated_bytes,
        }
        rows.append(row)
        artifacts_by_count[int(centroids.shape[0])] = (
            centroids.astype(np.float32, copy=False),
            np.log(np.clip(state_probs, 1e-12, None)).astype(np.float32, copy=False),
            state_mass.astype(np.float32, copy=False),
        )
    best = min(
        rows,
        key=lambda row: (
            math.inf if row.get("heldout_bits_per_byte") is None else float(row["heldout_bits_per_byte"]),
            float(row["heldout_cross_entropy_bits"]),
            int(row["num_states"]),
        ),
        default=None,
    )
    smallest_near_best = None
    if isinstance(best, dict):
        best_bpb = None if best.get("heldout_bits_per_byte") is None else float(best["heldout_bits_per_byte"])
        best_bits = float(best["heldout_cross_entropy_bits"])
        near_rows = []
        for row in rows:
            bpb_ok = best_bpb is None or row.get("heldout_bits_per_byte") is None or float(row["heldout_bits_per_byte"]) <= best_bpb + float(near_best_bpb_tol)
            bits_ok = float(row["heldout_cross_entropy_bits"]) <= best_bits + float(near_best_bits_tol)
            if bpb_ok and bits_ok:
                near_rows.append(row)
        if near_rows:
            smallest_near_best = min(near_rows, key=lambda row: int(row["num_states"]))
    export_row = smallest_near_best if isinstance(smallest_near_best, dict) else best
    export_arrays: dict[str, np.ndarray] = {}
    if isinstance(export_row, dict):
        export_count = int(export_row.get("num_states", 0))
        if export_count in artifacts_by_count:
            centroids, log_probs, state_mass = artifacts_by_count[export_count]
            export_arrays = {
                "causal_machine_signature_centroids": centroids.astype(np.float32, copy=False),
                "causal_machine_log_probs": log_probs.astype(np.float32, copy=False),
                "causal_machine_state_masses": state_mass.astype(np.float32, copy=False),
            }
    return (
        {
            "available": True,
            "sampled_prefixes": int(sample_count),
            "train_prefixes": int(train_x.shape[0]),
            "holdout_prefixes": int(hold_x.shape[0]),
            "unigram_holdout_bits_per_byte": unigram_bpb,
            "rows": rows,
            "best_state_count_by_holdout_bpb": best,
            "smallest_near_best_state_count": smallest_near_best,
        },
        export_arrays,
    )


def state_transition_determinism_profile(
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    signatures = np.asarray(future_signature_arrays.get("future_signature_matrix"))
    centroids = np.asarray(causal_state_arrays.get("causal_machine_signature_centroids"))
    if signatures.size <= 0 or centroids.size <= 0:
        return {"available": False, "reason": "missing_causal_machine_arrays"}
    labels = _assign_kmeans(signatures.astype(np.float32, copy=False), centroids.astype(np.float32, copy=False))
    transition, summary = _transition_summary_from_labels(labels, int(centroids.shape[0]))
    row_entropies = []
    row_top1 = []
    for row in transition:
        nz = row[row > 0.0]
        row_entropies.append(float(-(nz * np.log2(nz)).sum()) if nz.size > 0 else 0.0)
        row_top1.append(float(row.max()) if row.size > 0 else 0.0)
    row_sums = transition.sum(axis=1)
    weights = row_sums / max(float(row_sums.sum()), 1e-12)
    return {
        "available": True,
        "num_states": int(centroids.shape[0]),
        "transition_entropy_bits": float(np.dot(weights, np.asarray(row_entropies, dtype=np.float64))),
        "mean_top1_transition_mass": float(np.dot(weights, np.asarray(row_top1, dtype=np.float64))),
        **summary,
    }


def state_entropy_floor_profile(
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    base_bytes: np.ndarray | None,
) -> dict[str, object]:
    signatures = np.asarray(future_signature_arrays.get("future_signature_matrix"))
    targets = np.asarray(future_signature_arrays.get("future_signature_targets"))
    horizon_entropies = np.asarray(future_signature_arrays.get("future_signature_horizon_entropies"))
    horizons = np.asarray(future_signature_arrays.get("future_signature_horizons"))
    centroids = np.asarray(causal_state_arrays.get("causal_machine_signature_centroids"))
    log_probs = np.asarray(causal_state_arrays.get("causal_machine_log_probs"))
    if signatures.size <= 0 or targets.size <= 0 or centroids.size <= 0 or log_probs.size <= 0:
        return {"available": False, "reason": "missing_causal_machine_arrays"}
    labels = _assign_kmeans(signatures.astype(np.float32, copy=False), centroids.astype(np.float32, copy=False))
    target_log_probs = log_probs[labels, targets.astype(np.int64, copy=False)]
    surprisals = -target_log_probs.astype(np.float64, copy=False) / math.log(2.0)
    bpb = _bits_per_byte_from_surprisals(surprisals, targets.astype(np.int64, copy=False), base_bytes)
    state_mass = np.bincount(labels, minlength=centroids.shape[0]).astype(np.float64, copy=False)
    state_mass /= max(float(state_mass.sum()), 1e-12)
    state_next_entropy = np.zeros((centroids.shape[0],), dtype=np.float64)
    for idx in range(centroids.shape[0]):
        probs = np.exp(log_probs[idx].astype(np.float64, copy=False))
        probs /= max(float(probs.sum()), 1e-12)
        nz = probs[probs > 0.0]
        state_next_entropy[idx] = float(-(nz * np.log2(nz)).sum()) if nz.size > 0 else 0.0
    horizon_rows: list[dict[str, object]] = []
    if horizon_entropies.size > 0 and horizons.size == horizon_entropies.shape[1]:
        for col_idx, horizon in enumerate(horizons.tolist()):
            state_means = np.zeros((centroids.shape[0],), dtype=np.float64)
            for state_idx in range(centroids.shape[0]):
                mask = labels == state_idx
                if np.any(mask):
                    state_means[state_idx] = float(horizon_entropies[mask, col_idx].mean())
            horizon_rows.append(
                {
                    "horizon": int(horizon),
                    "conditional_future_entropy_bits": float(np.dot(state_mass, state_means)),
                }
            )
    return {
        "available": True,
        "num_states": int(centroids.shape[0]),
        "conditional_next_token_entropy_bits": float(np.dot(state_mass, state_next_entropy)),
        "heldout_bits_per_byte": bpb,
        "future_entropy_rows": horizon_rows,
    }


def causal_state_decodability_profile(
    tokens: np.ndarray,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    past_future_cca_arrays: dict[str, np.ndarray],
    holdout_fraction: float,
    ridge: float,
    base_bytes: np.ndarray | None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sample_views = _causal_state_sample_views(
        tokens,
        vocab_size,
        past_window,
        sketch_dim,
        future_signature_arrays,
        causal_state_arrays,
    )
    if sample_views is None:
        return {"available": False, "reason": "missing_causal_state_views"}, {}
    _, past_sketch, labels, targets, _ = sample_views
    state_log_probs = np.asarray(causal_state_arrays.get("causal_machine_log_probs"))
    if state_log_probs.size <= 0:
        return {"available": False, "reason": "missing_state_log_probs"}, {}
    sample_count = int(past_sketch.shape[0])
    split = int(max(1, min(sample_count - 1, round(sample_count * (1.0 - holdout_fraction)))))
    train_x = past_sketch[:split].astype(np.float32, copy=False)
    hold_x = past_sketch[split:].astype(np.float32, copy=False)
    train_labels = labels[:split].astype(np.int64, copy=False)
    hold_labels = labels[split:].astype(np.int64, copy=False)
    hold_targets = targets[split:].astype(np.int64, copy=False)
    if train_x.shape[0] <= 0 or hold_x.shape[0] <= 0:
        return {"available": False, "reason": "invalid_train_holdout_split"}, {}

    rows: list[dict[str, object]] = []
    artifacts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    state_prior = np.bincount(train_labels, minlength=state_log_probs.shape[0]).astype(np.float64, copy=False)
    state_prior /= max(float(state_prior.sum()), 1.0)
    state_prior_scores = np.log(np.clip(state_prior, 1e-12, None))[None, :].repeat(hold_labels.shape[0], axis=0)
    state_bits, _ = _cross_entropy_bits_from_scores(state_prior_scores, hold_labels, None)
    state_probs = np.exp(state_prior_scores - state_prior_scores.max(axis=1, keepdims=True))
    state_probs /= np.maximum(state_probs.sum(axis=1, keepdims=True), 1e-12)
    token_scores = state_probs @ state_log_probs.astype(np.float64, copy=False)
    token_bits, token_bpb = _cross_entropy_bits_from_scores(token_scores, hold_targets, base_bytes)
    rows.append(
        {
            "feature_family": "state_prior",
            "state_cross_entropy_bits": float(state_bits),
            "token_cross_entropy_bits": float(token_bits),
            "token_bits_per_byte": token_bpb,
            "state_top1_accuracy": float(np.mean(np.argmax(state_prior_scores, axis=1) == hold_labels)),
            "feature_dim": 0,
        }
    )

    weights, bias, state_mean = _fit_ridge_token_readout(train_x, train_labels, int(state_log_probs.shape[0]), ridge)
    artifacts["past_sketch_linear"] = (
        weights.astype(np.float32, copy=False),
        bias.astype(np.float32, copy=False),
        state_mean.astype(np.float32, copy=False),
    )
    hold_scores = (hold_x.astype(np.float64, copy=False) - state_mean.astype(np.float64, copy=False)) @ weights.astype(
        np.float64, copy=False
    ) + bias.astype(np.float64, copy=False)
    state_bits, _ = _cross_entropy_bits_from_scores(hold_scores, hold_labels, None)
    state_probs = np.exp(hold_scores - hold_scores.max(axis=1, keepdims=True))
    state_probs /= np.maximum(state_probs.sum(axis=1, keepdims=True), 1e-12)
    token_scores = state_probs @ state_log_probs.astype(np.float64, copy=False)
    token_bits, token_bpb = _cross_entropy_bits_from_scores(token_scores, hold_targets, base_bytes)
    rows.append(
        {
            "feature_family": "past_sketch_linear",
            "state_cross_entropy_bits": float(state_bits),
            "token_cross_entropy_bits": float(token_bits),
            "token_bits_per_byte": token_bpb,
            "state_top1_accuracy": float(np.mean(np.argmax(hold_scores, axis=1) == hold_labels)),
            "feature_dim": int(train_x.shape[1]),
        }
    )

    cca_basis = np.asarray(past_future_cca_arrays.get("past_future_cca_past_basis"))
    cca_mean = np.asarray(past_future_cca_arrays.get("past_future_cca_past_mean"))
    if cca_basis.size > 0 and cca_mean.size > 0:
        max_rank = min(int(cca_basis.shape[1]), 32)
        if max_rank > 0:
            train_cca = (train_x - cca_mean.astype(np.float32, copy=False)) @ cca_basis[:, :max_rank].astype(np.float32, copy=False)
            hold_cca = (hold_x - cca_mean.astype(np.float32, copy=False)) @ cca_basis[:, :max_rank].astype(np.float32, copy=False)
            weights, bias, state_mean = _fit_ridge_token_readout(
                train_cca,
                train_labels,
                int(state_log_probs.shape[0]),
                ridge,
            )
            artifacts["past_cca_linear"] = (
                weights.astype(np.float32, copy=False),
                bias.astype(np.float32, copy=False),
                state_mean.astype(np.float32, copy=False),
            )
            hold_scores = (
                hold_cca.astype(np.float64, copy=False) - state_mean.astype(np.float64, copy=False)
            ) @ weights.astype(np.float64, copy=False) + bias.astype(np.float64, copy=False)
            state_bits, _ = _cross_entropy_bits_from_scores(hold_scores, hold_labels, None)
            state_probs = np.exp(hold_scores - hold_scores.max(axis=1, keepdims=True))
            state_probs /= np.maximum(state_probs.sum(axis=1, keepdims=True), 1e-12)
            token_scores = state_probs @ state_log_probs.astype(np.float64, copy=False)
            token_bits, token_bpb = _cross_entropy_bits_from_scores(token_scores, hold_targets, base_bytes)
            rows.append(
                {
                    "feature_family": "past_cca_linear",
                    "state_cross_entropy_bits": float(state_bits),
                    "token_cross_entropy_bits": float(token_bits),
                    "token_bits_per_byte": token_bpb,
                    "state_top1_accuracy": float(np.mean(np.argmax(hold_scores, axis=1) == hold_labels)),
                    "feature_dim": int(max_rank),
                }
            )
    best = min(
        rows,
        key=lambda row: (
            math.inf if row.get("token_bits_per_byte") is None else float(row["token_bits_per_byte"]),
            float(row["state_cross_entropy_bits"]),
        ),
        default=None,
    )
    trainable_rows = [row for row in rows if str(row.get("feature_family")) != "state_prior"]
    best_trainable = min(
        trainable_rows,
        key=lambda row: (
            math.inf if row.get("token_bits_per_byte") is None else float(row["token_bits_per_byte"]),
            float(row["state_cross_entropy_bits"]),
            int(row.get("feature_dim", 0)),
        ),
        default=None,
    )
    export_arrays: dict[str, np.ndarray] = {}
    if isinstance(best_trainable, dict):
        family = str(best_trainable.get("feature_family", ""))
        artifact = artifacts.get(family)
        if artifact is not None:
            weights, bias, state_mean = artifact
            export_arrays = {
                "causal_state_decoder_weights": weights.astype(np.float32, copy=False),
                "causal_state_decoder_bias": bias.astype(np.float32, copy=False),
                "causal_state_decoder_state_mean": state_mean.astype(np.float32, copy=False),
            }
    return (
        {
            "available": True,
            "train_prefixes": int(train_x.shape[0]),
            "holdout_prefixes": int(hold_x.shape[0]),
            "rows": rows,
            "best_feature_family": best,
            "best_trainable_feature_family": best_trainable,
        },
        export_arrays,
    )


def causal_state_transition_learnability_profile(
    tokens: np.ndarray,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    holdout_fraction: float,
    ridge: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sample_views = _causal_state_sample_views(
        tokens,
        vocab_size,
        past_window,
        sketch_dim,
        future_signature_arrays,
        causal_state_arrays,
    )
    if sample_views is None:
        return {"available": False, "reason": "missing_causal_state_views"}, {}
    positions, past_sketch, labels, _, _ = sample_views
    if positions.size < 2:
        return {"available": False, "reason": "not_enough_positions"}, {}
    valid = np.diff(positions) == 1
    if not np.any(valid):
        return {"available": False, "reason": "no_consecutive_prefix_pairs"}, {}
    curr_labels = labels[:-1][valid].astype(np.int64, copy=False)
    next_labels = labels[1:][valid].astype(np.int64, copy=False)
    curr_features = past_sketch[:-1][valid].astype(np.float32, copy=False)
    curr_tokens = tokens[positions[:-1][valid]].astype(np.int64, copy=False)
    sample_count = int(curr_labels.shape[0])
    split = int(max(1, min(sample_count - 1, round(sample_count * (1.0 - holdout_fraction)))))
    train_labels = curr_labels[:split]
    hold_labels = curr_labels[split:]
    train_next = next_labels[:split]
    hold_next = next_labels[split:]
    train_features = curr_features[:split]
    hold_features = curr_features[split:]
    train_tokens = curr_tokens[:split]
    hold_tokens = curr_tokens[split:]
    num_states = int(max(labels.max(initial=0), next_labels.max(initial=0)) + 1)

    rows: list[dict[str, object]] = []
    export_arrays: dict[str, np.ndarray] = {}
    transition = np.zeros((num_states, num_states), dtype=np.float64)
    for src, dst in zip(train_labels.tolist(), train_next.tolist(), strict=True):
        transition[int(src), int(dst)] += 1.0
    transition += 1e-3
    transition /= np.maximum(transition.sum(axis=1, keepdims=True), 1e-12)
    state_scores = np.log(np.clip(transition[hold_labels], 1e-12, None))
    bits, _ = _cross_entropy_bits_from_scores(state_scores, hold_next, None)
    rows.append(
        {
            "predictor_family": "current_state_markov",
            "next_state_cross_entropy_bits": float(bits),
            "next_state_top1_accuracy": float(np.mean(np.argmax(state_scores, axis=1) == hold_next)),
        }
    )

    pair_counts: dict[tuple[int, int], np.ndarray] = {}
    for src, tok, dst in zip(train_labels.tolist(), train_tokens.tolist(), train_next.tolist(), strict=True):
        key = (int(src), int(tok))
        arr = pair_counts.get(key)
        if arr is None:
            arr = np.full((num_states,), 1e-3, dtype=np.float64)
            pair_counts[key] = arr
        arr[int(dst)] += 1.0
    token_scores = np.zeros((hold_next.shape[0], num_states), dtype=np.float64)
    backoff = np.log(np.clip(transition, 1e-12, None))
    for row_idx, (src, tok) in enumerate(zip(hold_labels.tolist(), hold_tokens.tolist(), strict=True)):
        arr = pair_counts.get((int(src), int(tok)))
        if arr is None:
            token_scores[row_idx] = backoff[int(src)]
        else:
            probs = arr / max(float(arr.sum()), 1e-12)
            token_scores[row_idx] = np.log(np.clip(probs, 1e-12, None))
    bits, _ = _cross_entropy_bits_from_scores(token_scores, hold_next, None)
    rows.append(
        {
            "predictor_family": "state_plus_token_table",
            "next_state_cross_entropy_bits": float(bits),
            "next_state_top1_accuracy": float(np.mean(np.argmax(token_scores, axis=1) == hold_next)),
        }
    )

    weights, bias, feature_mean = _fit_ridge_token_readout(train_features, train_next, num_states, ridge)
    feature_scores = (
        hold_features.astype(np.float64, copy=False) - feature_mean.astype(np.float64, copy=False)
    ) @ weights.astype(np.float64, copy=False) + bias.astype(np.float64, copy=False)
    bits, _ = _cross_entropy_bits_from_scores(feature_scores, hold_next, None)
    rows.append(
        {
            "predictor_family": "past_sketch_linear",
            "next_state_cross_entropy_bits": float(bits),
            "next_state_top1_accuracy": float(np.mean(np.argmax(feature_scores, axis=1) == hold_next)),
        }
    )
    best = min(rows, key=lambda row: float(row["next_state_cross_entropy_bits"]), default=None)
    if isinstance(best, dict) and str(best.get("predictor_family")) == "past_sketch_linear":
        export_arrays = {
            "causal_state_transition_weights": weights.astype(np.float32, copy=False),
            "causal_state_transition_bias": bias.astype(np.float32, copy=False),
            "causal_state_transition_feature_mean": feature_mean.astype(np.float32, copy=False),
        }
    return (
        {
            "available": True,
            "train_transitions": int(train_next.shape[0]),
            "holdout_transitions": int(hold_next.shape[0]),
            "rows": rows,
            "best_predictor_family": best,
        },
        export_arrays,
    )


def causal_state_multi_horizon_sufficiency_profile(
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    base_bytes: np.ndarray | None,
) -> dict[str, object]:
    signatures = np.asarray(future_signature_arrays.get("future_signature_matrix"))
    targets = np.asarray(future_signature_arrays.get("future_signature_targets"))
    horizon_entropies = np.asarray(future_signature_arrays.get("future_signature_horizon_entropies"))
    horizons = np.asarray(future_signature_arrays.get("future_signature_horizons"))
    centroids = np.asarray(causal_state_arrays.get("causal_machine_signature_centroids"))
    log_probs = np.asarray(causal_state_arrays.get("causal_machine_log_probs"))
    if signatures.size <= 0 or targets.size <= 0 or centroids.size <= 0 or log_probs.size <= 0:
        return {"available": False, "reason": "missing_causal_state_arrays"}
    labels = _assign_kmeans(signatures.astype(np.float32, copy=False), centroids.astype(np.float32, copy=False))
    target_log_probs = log_probs[labels, targets.astype(np.int64, copy=False)]
    surprisals = -target_log_probs.astype(np.float64, copy=False) / math.log(2.0)
    conditional_next_bits = float(surprisals.mean()) if surprisals.size > 0 else 0.0
    conditional_next_bpb = _bits_per_byte_from_surprisals(surprisals, targets.astype(np.int64, copy=False), base_bytes)
    unigram = np.bincount(targets.astype(np.int64, copy=False), minlength=log_probs.shape[1]).astype(np.float64, copy=False)
    unigram /= max(float(unigram.sum()), 1.0)
    unconditional_next_bits = float(
        -np.mean(np.log2(np.clip(unigram[targets.astype(np.int64, copy=False)], 1e-12, None)))
    ) if targets.size > 0 else 0.0
    rows: list[dict[str, object]] = [
        {
            "horizon": 1,
            "unconditional_entropy_bits": float(unconditional_next_bits),
            "conditional_entropy_bits": float(conditional_next_bits),
            "entropy_reduction_bits": float(unconditional_next_bits - conditional_next_bits),
            "heldout_bits_per_byte": conditional_next_bpb,
        }
    ]
    if horizon_entropies.size > 0 and horizons.size == horizon_entropies.shape[1]:
        state_mass = np.bincount(labels, minlength=centroids.shape[0]).astype(np.float64, copy=False)
        state_mass /= max(float(state_mass.sum()), 1e-12)
        for col_idx, horizon in enumerate(horizons.tolist()):
            unconditional = float(horizon_entropies[:, col_idx].mean())
            state_means = np.zeros((centroids.shape[0],), dtype=np.float64)
            for state_idx in range(centroids.shape[0]):
                mask = labels == state_idx
                if np.any(mask):
                    state_means[state_idx] = float(horizon_entropies[mask, col_idx].mean())
            conditional = float(np.dot(state_mass, state_means))
            rows.append(
                {
                    "horizon": int(horizon),
                    "unconditional_entropy_bits": unconditional,
                    "conditional_entropy_bits": conditional,
                    "entropy_reduction_bits": float(unconditional - conditional),
                    "heldout_bits_per_byte": None,
                }
            )
    best = max(rows, key=lambda row: float(row["entropy_reduction_bits"]), default=None)
    return {
        "available": True,
        "rows": rows,
        "best_horizon_by_entropy_reduction": best,
    }


def causal_state_merge_error_profile(
    causal_state_reconstruction: dict[str, object] | None,
) -> dict[str, object]:
    if causal_state_reconstruction is None or not bool(causal_state_reconstruction.get("available")):
        return {"available": False, "reason": "missing_causal_state_reconstruction"}
    rows = causal_state_reconstruction.get("rows")
    if not isinstance(rows, list) or not rows:
        return {"available": False, "reason": "missing_rows"}
    best = causal_state_reconstruction.get("best_state_count_by_holdout_bpb")
    if not isinstance(best, dict):
        return {"available": False, "reason": "missing_best_state"}
    best_states = int(best.get("num_states", 0))
    best_bpb = best.get("heldout_bits_per_byte")
    best_bits = float(best.get("heldout_cross_entropy_bits", 0.0))
    merge_rows: list[dict[str, object]] = []
    for row in rows:
        num_states = int(row.get("num_states", 0))
        if num_states >= best_states:
            continue
        row_bpb = row.get("heldout_bits_per_byte")
        merge_rows.append(
            {
                "merged_to_num_states": num_states,
                "merge_penalty_bits": float(row.get("heldout_cross_entropy_bits", 0.0)) - best_bits,
                "merge_penalty_bpb": None
                if best_bpb is None or row_bpb is None
                else float(row_bpb) - float(best_bpb),
                "bytes_saved_vs_best": int(best.get("estimated_fp16_state_bytes", 0)) - int(row.get("estimated_fp16_state_bytes", 0)),
            }
        )
    smallest_low_penalty = None
    if merge_rows:
        smallest_low_penalty = min(
            [row for row in merge_rows if float(row["merge_penalty_bits"]) <= 0.05] or merge_rows,
            key=lambda row: int(row["merged_to_num_states"]),
        )
    return {
        "available": True,
        "best_state_count": best_states,
        "merge_rows": merge_rows,
        "smallest_low_penalty_merge": smallest_low_penalty,
    }


def causal_state_residual_geometry_profile(
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    ranks: list[int] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    signatures = np.asarray(future_signature_arrays.get("future_signature_matrix"))
    centroids = np.asarray(causal_state_arrays.get("causal_machine_signature_centroids"))
    if signatures.size <= 0 or centroids.size <= 0:
        return {"available": False, "reason": "missing_causal_state_arrays"}, {}
    labels = _assign_kmeans(signatures.astype(np.float32, copy=False), centroids.astype(np.float32, copy=False))
    residual = signatures.astype(np.float32, copy=False) - centroids[labels].astype(np.float32, copy=False)
    residual_rms = float(np.sqrt(np.square(residual.astype(np.float64, copy=False)).mean())) if residual.size > 0 else 0.0
    if residual.shape[0] <= 1 or residual.shape[1] <= 0:
        return {"available": False, "reason": "degenerate_residual_matrix"}, {}
    centered = residual - residual.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered.astype(np.float64, copy=False), full_matrices=False)
    energy = singular_values * singular_values
    total_energy = float(energy.sum())
    safe_ranks = sorted({int(r) for r in (ranks or [4, 8, 16, 32]) if int(r) > 0})
    rows: list[dict[str, object]] = []
    cumulative = np.cumsum(energy)
    for rank in safe_ranks:
        capped = min(rank, int(cumulative.size))
        if capped <= 0:
            continue
        rows.append(
            {
                "rank": int(capped),
                "residual_energy_fraction": float(cumulative[capped - 1] / max(total_energy, 1e-12)),
            }
        )
    state_mass = np.bincount(labels, minlength=centroids.shape[0]).astype(np.float64, copy=False)
    state_mass /= max(float(state_mass.sum()), 1e-12)
    state_residual_norms = np.zeros((centroids.shape[0],), dtype=np.float64)
    for state_idx in range(centroids.shape[0]):
        mask = labels == state_idx
        if np.any(mask):
            state_residual_norms[state_idx] = float(np.sqrt(np.square(residual[mask].astype(np.float64, copy=False)).mean()))
    best = min(rows, key=lambda row: abs(float(row["residual_energy_fraction"]) - 0.95), default=None)
    export_rank = 0 if not isinstance(best, dict) else int(best.get("rank", 0))
    export_arrays: dict[str, np.ndarray] = {}
    if export_rank > 0:
        capped = min(export_rank, int(vt.shape[0]))
        export_arrays = {
            "causal_state_residual_basis": vt[:capped].transpose(1, 0).astype(np.float32, copy=False),
            "causal_state_residual_mean": residual.mean(axis=0, keepdims=True).astype(np.float32, copy=False),
            "causal_state_residual_singular_values": singular_values[:capped].astype(np.float32, copy=False),
        }
    return (
        {
            "available": True,
            "residual_rms": residual_rms,
            "weighted_state_residual_rms": float(np.dot(state_mass, state_residual_norms)),
            "rows": rows,
            "best_rank_near_95pct_energy": best,
        },
        export_arrays,
    )


def predictive_state_compression_profile(
    tokens: np.ndarray,
    vocab_size: int,
    order: int,
    ranks: list[int],
    max_prefixes: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    transitions = _build_context_transitions(tokens, vocab_size, order)
    if transitions is None:
        return (
            {"available": False, "order": int(order), "reason": "not_enough_tokens"},
            {},
        )
    context_ids, next_tokens = transitions
    lookup = _build_context_distribution_lookup(context_ids, next_tokens)
    if lookup is None:
        return (
            {"available": False, "order": int(order), "reason": "no_context_lookup"},
            {},
        )
    total_positions = int(context_ids.size)
    sample_count = min(max(int(max_prefixes), 1), total_positions)
    sample_idx = np.linspace(0, total_positions - 1, num=sample_count, dtype=np.int64)
    sampled_context_ids = context_ids[sample_idx].astype(np.int64, copy=False)
    sampled_targets = next_tokens[sample_idx].astype(np.int64, copy=False)
    unique_ctx = lookup["context_ids"]
    pos = np.searchsorted(unique_ctx, sampled_context_ids)
    seen = (pos < unique_ctx.size) & (unique_ctx[pos] == sampled_context_ids)
    if not np.any(seen):
        return (
            {"available": False, "order": int(order), "reason": "no_seen_contexts"},
            {},
        )
    sample_idx = sample_idx[seen]
    sampled_context_ids = sampled_context_ids[seen]
    sampled_targets = sampled_targets[seen]
    pos = pos[seen]

    matrix = np.zeros((sampled_context_ids.size, vocab_size), dtype=np.float32)
    for row_idx, ctx_idx in enumerate(pos):
        start = int(lookup["ctx_starts"][ctx_idx])
        end = int(lookup["ctx_ends"][ctx_idx])
        local_next = lookup["pair_next"][start:end]
        local_counts = lookup["pair_counts"][start:end].astype(np.float32, copy=False)
        total = float(local_counts.sum())
        if total <= 0.0:
            continue
        matrix[row_idx, local_next] = local_counts / total

    col_mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - col_mean
    cov = centered.T @ centered / max(centered.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64, copy=False))
    order_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order_idx].astype(np.float32, copy=False)
    eigvecs = eigvecs[:, order_idx].astype(np.float32, copy=False)
    positive_eigs = np.clip(eigvals, a_min=0.0, a_max=None)
    total_energy = float(positive_eigs.sum())

    safe_ranks = sorted({int(r) for r in ranks if int(r) > 0})
    rows: list[dict[str, object]] = []
    true_token_probs = matrix[np.arange(matrix.shape[0]), sampled_targets]
    base_surprisal = -np.log2(np.clip(true_token_probs.astype(np.float64, copy=False), 1e-12, None))
    base_cross_entropy = float(base_surprisal.mean()) if base_surprisal.size > 0 else 0.0
    for rank in safe_ranks:
        rank = min(rank, int(eigvecs.shape[1]))
        if rank <= 0:
            continue
        basis = eigvecs[:, :rank]
        projected = centered @ basis
        reconstructed = projected @ basis.T + col_mean
        reconstructed = np.clip(reconstructed, a_min=1e-9, a_max=None)
        reconstructed /= reconstructed.sum(axis=1, keepdims=True).clip(min=1e-9)
        recon_true = reconstructed[np.arange(reconstructed.shape[0]), sampled_targets]
        recon_surprisal = -np.log2(np.clip(recon_true.astype(np.float64, copy=False), 1e-12, None))
        explained = float(np.clip(positive_eigs[:rank].sum() / max(total_energy, 1e-12), 0.0, 1.0))
        rows.append(
            {
                "rank": int(rank),
                "variance_explained": explained,
                "oracle_cross_entropy_bits": base_cross_entropy,
                "reconstructed_cross_entropy_bits": float(recon_surprisal.mean()) if recon_surprisal.size > 0 else 0.0,
                "cross_entropy_gap_bits": float(recon_surprisal.mean() - base_cross_entropy) if recon_surprisal.size > 0 else 0.0,
                "mean_l2_reconstruction_error": float(np.square(matrix - reconstructed).mean()),
            }
        )
    best = min(rows, key=lambda row: (float(row["cross_entropy_gap_bits"]), int(row["rank"])), default=None)
    summary = {
        "available": True,
        "order": int(order),
        "sampled_prefixes": int(matrix.shape[0]),
        "target_rank_candidates": safe_ranks,
        "positive_eigenvalue_energy": total_energy,
        "top_eigenvalues": [float(v) for v in eigvals[: min(16, eigvals.size)].tolist()],
        "rows": rows,
        "best_rank_by_cross_entropy_gap": best,
    }
    arrays = {
        "predictive_state_eigenvalues": eigvals[: max(safe_ranks or [0], default=0)].astype(np.float32, copy=False),
        "predictive_state_basis": eigvecs[:, : max(safe_ranks or [0], default=0)].astype(np.float32, copy=False),
        "predictive_state_column_mean": col_mean.astype(np.float32, copy=False),
    }
    return summary, arrays


def predictive_state_transition_profile(
    tokens: np.ndarray,
    vocab_size: int,
    order: int,
    ranks: list[int],
    max_prefixes: int,
) -> dict[str, object]:
    transitions = _build_context_transitions(tokens, vocab_size, order)
    if transitions is None:
        return {"available": False, "order": int(order), "reason": "not_enough_tokens"}
    context_ids, _next_tokens = transitions
    if context_ids.size < 2:
        return {"available": False, "order": int(order), "reason": "not_enough_context_pairs"}
    lookup = _build_context_distribution_lookup(context_ids, tokens[order:].astype(np.int64, copy=False))
    if lookup is None:
        return {"available": False, "order": int(order), "reason": "no_context_lookup"}
    total_pairs = int(context_ids.size - 1)
    sample_count = min(max(int(max_prefixes), 2), total_pairs)
    pair_idx = np.linspace(0, total_pairs - 1, num=sample_count, dtype=np.int64)
    curr_contexts = context_ids[pair_idx].astype(np.int64, copy=False)
    next_contexts = context_ids[pair_idx + 1].astype(np.int64, copy=False)
    curr_matrix = _lookup_context_posteriors(lookup, curr_contexts, vocab_size)
    next_matrix = _lookup_context_posteriors(lookup, next_contexts, vocab_size)
    col_mean = curr_matrix.mean(axis=0, keepdims=True)
    curr_centered = curr_matrix - col_mean
    next_centered = next_matrix - col_mean
    cov = curr_centered.T @ curr_centered / max(curr_centered.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64, copy=False))
    order_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order_idx].astype(np.float32, copy=False)
    safe_ranks = sorted({int(r) for r in ranks if int(r) > 0})
    rows: list[dict[str, object]] = []
    for rank in safe_ranks:
        rank = min(rank, int(eigvecs.shape[1]))
        if rank <= 0:
            continue
        basis = eigvecs[:, :rank]
        state_t = curr_centered @ basis
        state_next = next_centered @ basis
        if state_t.shape[0] < rank:
            continue
        transition_matrix, *_ = np.linalg.lstsq(state_t.astype(np.float64, copy=False), state_next.astype(np.float64, copy=False), rcond=None)
        predicted = state_t @ transition_matrix.astype(np.float32, copy=False)
        residual = state_next - predicted
        denom = float(np.square(state_next).mean())
        explained = float(1.0 - np.square(residual).mean() / max(denom, 1e-12))
        op_eigs = np.linalg.eigvals(transition_matrix)
        rows.append(
            {
                "rank": int(rank),
                "state_transition_mse": float(np.square(residual).mean()),
                "state_transition_r2": explained,
                "operator_spectral_radius": float(np.max(np.abs(op_eigs))) if op_eigs.size > 0 else 0.0,
            }
        )
    best = max(rows, key=lambda row: (float(row["state_transition_r2"]), -float(row["state_transition_mse"])), default=None)
    return {
        "available": True,
        "order": int(order),
        "sampled_transition_pairs": int(curr_contexts.size),
        "rows": rows,
        "best_rank_by_transition_r2": best,
    }


def predictive_state_recommendations(
    predictive_state_compression: dict[str, object] | None,
) -> dict[str, object]:
    if predictive_state_compression is None or not bool(predictive_state_compression.get("available")):
        return {"recommended_state": None}
    best = predictive_state_compression.get("best_rank_by_cross_entropy_gap")
    if not isinstance(best, dict):
        return {"recommended_state": None}
    order = int(predictive_state_compression.get("order", 0))
    rank = int(best.get("rank", 0))
    gap_bits = float(best.get("cross_entropy_gap_bits", 0.0))
    explained = float(best.get("variance_explained", 0.0))
    return {
        "recommended_state": {
            "order": order,
            "rank": rank,
            "cross_entropy_gap_bits": gap_bits,
            "variance_explained": explained,
            "trainer_hint": {
                "use_predictive_state_bias": True,
                "predictive_state_order": order,
                "predictive_state_rank": rank,
                "predictive_state_source": "offline_prefix_future_svd",
            },
        }
    }


def predictive_state_transition_recommendations(
    predictive_state_transition: dict[str, object] | None,
) -> dict[str, object]:
    if predictive_state_transition is None or not bool(predictive_state_transition.get("available")):
        return {"recommended_operator": None}
    best = predictive_state_transition.get("best_rank_by_transition_r2")
    if not isinstance(best, dict):
        return {"recommended_operator": None}
    order = int(predictive_state_transition.get("order", 0))
    rank = int(best.get("rank", 0))
    return {
        "recommended_operator": {
            "order": order,
            "rank": rank,
            "state_transition_r2": float(best.get("state_transition_r2", 0.0)),
            "operator_spectral_radius": float(best.get("operator_spectral_radius", 0.0)),
            "trainer_hint": {
                "use_predictive_state_operator": True,
                "predictive_state_order": order,
                "predictive_state_rank": rank,
                "predictive_state_operator_source": "offline_linear_state_dynamics",
            },
        }
    }


def past_future_cca_recommendations(
    past_future_cca: dict[str, object] | None,
) -> dict[str, object]:
    if past_future_cca is None or not bool(past_future_cca.get("available")):
        return {"recommended_state": None}
    best = past_future_cca.get("best_rank_by_cumulative_correlation")
    if not isinstance(best, dict):
        return {"recommended_state": None}
    return {
        "recommended_state": {
            "rank": int(best.get("rank", 0)),
            "mean_canonical_correlation": float(best.get("mean_canonical_correlation", 0.0)),
            "cumulative_canonical_correlation": float(best.get("cumulative_canonical_correlation", 0.0)),
            "past_window": int(past_future_cca.get("past_window", 0)),
            "future_window": int(past_future_cca.get("future_window", 0)),
            "trainer_hint": {
                "use_past_future_state_bias": True,
                "past_future_state_rank": int(best.get("rank", 0)),
                "past_future_window_past": int(past_future_cca.get("past_window", 0)),
                "past_future_window_future": int(past_future_cca.get("future_window", 0)),
            },
        }
    }


def ppm_oracle_recommendations(
    ppm_oracle: dict[str, object] | None,
) -> dict[str, object]:
    if ppm_oracle is None or not bool(ppm_oracle.get("available")):
        return {"recommended_order": None}
    best = ppm_oracle.get("best_by_bits_per_byte") or ppm_oracle.get("best_by_entropy_bits")
    if not isinstance(best, dict):
        return {"recommended_order": None}
    return {
        "recommended_order": {
            "max_order": int(best.get("max_order", 0)),
            "entropy_bits": float(best.get("entropy_bits", 0.0)),
            "bits_per_byte": None if best.get("bits_per_byte") is None else float(best.get("bits_per_byte")),
        }
    }


def minimal_causal_state_recommendations(
    minimal_causal_state: dict[str, object] | None,
) -> dict[str, object]:
    if minimal_causal_state is None or not bool(minimal_causal_state.get("available")):
        return {"recommended_state": None}
    best = minimal_causal_state.get("best_rank_by_holdout_bpb")
    smallest = minimal_causal_state.get("smallest_near_best_rank")
    if not isinstance(best, dict):
        return {"recommended_state": None}
    chosen = smallest if isinstance(smallest, dict) else best
    return {
        "recommended_state": {
            "rank": int(chosen.get("rank", 0)),
            "heldout_bits_per_byte": None
            if chosen.get("heldout_bits_per_byte") is None
            else float(chosen.get("heldout_bits_per_byte")),
            "heldout_cross_entropy_bits": float(chosen.get("heldout_cross_entropy_bits", 0.0)),
            "heldout_delta_bpb_vs_unigram": None
            if chosen.get("heldout_delta_bpb_vs_unigram") is None
            else float(chosen.get("heldout_delta_bpb_vs_unigram")),
            "estimated_fp16_state_bytes": int(chosen.get("estimated_fp16_state_bytes", 0)),
            "trainer_hint": {
                "use_minimal_causal_state_bias": True,
                "minimal_causal_state_rank": int(chosen.get("rank", 0)),
                "minimal_causal_state_source": "heldout_state_frontier",
            },
        },
        "best_state": best,
        "smallest_near_best_state": smallest,
    }


def dataset_world_model_recommendations(
    dataset_world_model: dict[str, object] | None,
) -> dict[str, object]:
    if dataset_world_model is None or not bool(dataset_world_model.get("available")):
        return {"recommended_world_model": None}
    encoder = dataset_world_model.get("chunk_predictive_state_encoder") or {}
    regime_model = dataset_world_model.get("global_regime_model") or {}
    trainer_hints = dataset_world_model.get("trainer_distilled_hints") or {}
    best = regime_model.get("best_regime_model")
    hint = trainer_hints.get("trainer_hint")
    if not isinstance(best, dict) or not isinstance(hint, dict):
        return {"recommended_world_model": None}
    return {
        "recommended_world_model": {
            "state_rank": int(encoder.get("state_rank", 0)),
            "chunk_tokens": int(encoder.get("chunk_tokens", 0)),
            "num_regimes": int(best.get("num_regimes", 0)),
            "second_eigenvalue_abs": float(best.get("second_eigenvalue_abs", 0.0)),
            "self_transition_mass": float(best.get("self_transition_mass", 0.0)),
            "trainer_hint": hint,
        }
    }


def regime_conditioned_bpb_profile(
    counts: np.ndarray,
    base_bytes: np.ndarray | None,
    dataset_world_model: dict[str, object] | None,
    dataset_world_model_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if dataset_world_model is None or not bool(dataset_world_model.get("available")):
        return {"available": False, "reason": "missing_dataset_world_model"}, {}
    chunk_counts = dataset_world_model_arrays.get("dataset_world_model_chunk_token_counts")
    prefix_hists = dataset_world_model_arrays.get("dataset_world_model_chunk_prefix_hists")
    labels = dataset_world_model_arrays.get("dataset_world_model_regime_assignments")
    centroids = dataset_world_model_arrays.get("dataset_world_model_regime_centroids")
    if chunk_counts is None or prefix_hists is None or labels is None or centroids is None:
        return {"available": False, "reason": "missing_world_model_arrays"}, {}
    if chunk_counts.shape[0] <= 0 or labels.size <= 0:
        return {"available": False, "reason": "empty_world_model_arrays"}, {}

    chunk_counts_f64 = chunk_counts.astype(np.float64, copy=False)
    prefix_hists_f64 = prefix_hists.astype(np.float64, copy=False)
    labels_i64 = labels.astype(np.int64, copy=False)
    num_regimes = int(max(int(centroids.shape[0]), int(labels_i64.max()) + 1 if labels_i64.size > 0 else 0))
    if num_regimes <= 0:
        return {"available": False, "reason": "invalid_regime_count"}, {}

    global_probs = np.maximum(counts.astype(np.float64, copy=False), 0.0)
    global_probs = global_probs / max(float(global_probs.sum()), 1e-12)
    global_surprisal = -np.log2(np.clip(global_probs, 1e-12, None))

    regime_token_counts = np.zeros((num_regimes, chunk_counts_f64.shape[1]), dtype=np.float64)
    occupancy = np.bincount(labels_i64, minlength=num_regimes).astype(np.float64, copy=False)
    regime_prefix_centroids = np.zeros((num_regimes, prefix_hists_f64.shape[1]), dtype=np.float64)
    for regime_id in range(num_regimes):
        mask = labels_i64 == regime_id
        if not np.any(mask):
            regime_token_counts[regime_id] = np.maximum(counts.astype(np.float64, copy=False), 1.0)
            regime_prefix_centroids[regime_id] = global_probs
            continue
        regime_token_counts[regime_id] = chunk_counts_f64[mask].sum(axis=0)
        regime_prefix_centroids[regime_id] = prefix_hists_f64[mask].mean(axis=0)
    regime_token_probs = np.divide(
        np.maximum(regime_token_counts, 1e-12),
        np.maximum(regime_token_counts.sum(axis=1, keepdims=True), 1e-12),
        out=np.zeros_like(regime_token_counts),
        where=np.maximum(regime_token_counts.sum(axis=1, keepdims=True), 1e-12) > 0,
    )
    prefix_scores = prefix_hists_f64 @ regime_prefix_centroids.T
    prefix_labels = np.argmax(prefix_scores, axis=1).astype(np.int64, copy=False)
    if num_regimes > 1:
        top2 = np.partition(prefix_scores, kth=num_regimes - 2, axis=1)[:, -2:]
        prefix_margin = top2[:, 1] - top2[:, 0]
    else:
        prefix_margin = np.ones((prefix_scores.shape[0],), dtype=np.float64)
    prefix_accuracy = float(np.mean(prefix_labels == labels_i64)) if labels_i64.size > 0 else 0.0

    if base_bytes is not None:
        byte_table = np.maximum(base_bytes.astype(np.float64, copy=False), 0.0)
        chunk_bytes = chunk_counts_f64 @ byte_table
        total_bytes = float(chunk_bytes.sum())
    else:
        byte_table = None
        chunk_bytes = None
        total_bytes = 0.0

    global_bit_sum = 0.0
    oracle_regime_bit_sum = 0.0
    prefix_regime_bit_sum = 0.0
    chunk_rows: list[dict[str, object]] = []
    for idx in range(chunk_counts_f64.shape[0]):
        local_counts = chunk_counts_f64[idx]
        oracle_regime = int(labels_i64[idx])
        prefix_regime = int(prefix_labels[idx])
        global_bits = float((local_counts * global_surprisal).sum())
        oracle_bits = float((local_counts * (-np.log2(np.clip(regime_token_probs[oracle_regime], 1e-12, None)))).sum())
        prefix_bits = float((local_counts * (-np.log2(np.clip(regime_token_probs[prefix_regime], 1e-12, None)))).sum())
        global_bit_sum += global_bits
        oracle_regime_bit_sum += oracle_bits
        prefix_regime_bit_sum += prefix_bits
        row = {
            "chunk_index": int(idx),
            "oracle_regime_id": oracle_regime,
            "prefix_regime_id": prefix_regime,
            "prefix_regime_correct": bool(prefix_regime == oracle_regime),
            "prefix_assignment_margin": float(prefix_margin[idx]),
            "global_cross_entropy_bits": float(global_bits / max(float(local_counts.sum()), 1e-12)),
            "oracle_regime_cross_entropy_bits": float(oracle_bits / max(float(local_counts.sum()), 1e-12)),
            "prefix_regime_cross_entropy_bits": float(prefix_bits / max(float(local_counts.sum()), 1e-12)),
        }
        if chunk_bytes is not None and chunk_bytes[idx] > 0.0:
            row["global_bits_per_byte"] = float(global_bits / chunk_bytes[idx])
            row["oracle_regime_bits_per_byte"] = float(oracle_bits / chunk_bytes[idx])
            row["prefix_regime_bits_per_byte"] = float(prefix_bits / chunk_bytes[idx])
        else:
            row["global_bits_per_byte"] = None
            row["oracle_regime_bits_per_byte"] = None
            row["prefix_regime_bits_per_byte"] = None
        chunk_rows.append(row)

    macro_k = max(1, min(num_regimes, int(round(math.sqrt(num_regimes))))) if num_regimes > 1 else 1
    macro_labels, macro_centroids = _simple_kmeans(centroids.astype(np.float32, copy=False), macro_k, 8)
    fine_to_macro = {int(regime_id): int(macro_labels[regime_id]) for regime_id in range(num_regimes)}
    macro_rows: list[dict[str, object]] = []
    for macro_id in range(int(macro_centroids.shape[0])):
        member_regimes = [regime_id for regime_id, mapped in fine_to_macro.items() if int(mapped) == macro_id]
        macro_rows.append(
            {
                "macro_regime_id": int(macro_id),
                "member_regime_ids": member_regimes,
                "coverage_fraction": float(sum(float(occupancy[r]) for r in member_regimes) / max(float(occupancy.sum()), 1e-12)),
            }
        )

    regime_rows: list[dict[str, object]] = []
    safe_topk = 8
    for regime_id in range(num_regimes):
        probs = regime_token_probs[regime_id]
        lift = np.divide(probs, np.maximum(global_probs, 1e-12), out=np.zeros_like(probs), where=global_probs > 0.0)
        score = probs * np.log2(np.maximum(lift, 1e-12))
        top_ids = np.argsort(score)[::-1][:safe_topk]
        regime_rows.append(
            {
                "regime_id": int(regime_id),
                "macro_regime_id": int(fine_to_macro[int(regime_id)]),
                "coverage_fraction": float(occupancy[regime_id] / max(float(occupancy.sum()), 1e-12)),
                "top_tokens_by_lift": [
                    {
                        "token_id": int(token_id),
                        "regime_prob": float(probs[int(token_id)]),
                        "global_prob": float(global_probs[int(token_id)]),
                        "lift": float(lift[int(token_id)]),
                    }
                    for token_id in top_ids.tolist()
                    if probs[int(token_id)] > 0.0
                ],
            }
        )

    global_entropy_bits = float(global_bit_sum / max(float(chunk_counts_f64.sum()), 1e-12))
    oracle_entropy_bits = float(oracle_regime_bit_sum / max(float(chunk_counts_f64.sum()), 1e-12))
    prefix_entropy_bits = float(prefix_regime_bit_sum / max(float(chunk_counts_f64.sum()), 1e-12))
    global_bpb = float(global_bit_sum / total_bytes) if total_bytes > 0.0 else None
    oracle_bpb = float(oracle_regime_bit_sum / total_bytes) if total_bytes > 0.0 else None
    prefix_bpb = float(prefix_regime_bit_sum / total_bytes) if total_bytes > 0.0 else None

    trainer_hints = (dataset_world_model.get("trainer_distilled_hints") or {})
    trainer_hint = dict(trainer_hints.get("trainer_hint") or {})
    trainer_export = {
        "source": "dataset_world_model",
        "prefix_classifier": {
            "accuracy": prefix_accuracy,
            "mean_assignment_margin": float(np.mean(prefix_margin)) if prefix_margin.size > 0 else 0.0,
            "num_regimes": num_regimes,
        },
        "trainer_hint": trainer_hint,
        "curriculum_weights": list(trainer_hints.get("curriculum_weights") or []),
        "route_priors": list(trainer_hints.get("route_priors") or []),
        "macro_regime_hierarchy": macro_rows,
        "regime_token_priors": regime_rows,
    }

    return (
        {
            "available": True,
            "regime_hierarchy": {
                "macro_regimes": macro_rows,
                "fine_to_macro": [{"regime_id": int(k), "macro_regime_id": int(v)} for k, v in fine_to_macro.items()],
            },
            "prefix_classifier": {
                "accuracy": prefix_accuracy,
                "mean_assignment_margin": float(np.mean(prefix_margin)) if prefix_margin.size > 0 else 0.0,
                "chunk_count": int(labels_i64.size),
            },
            "regime_conditioned_priors": regime_rows,
            "bpb_ablation": {
                "global_unigram_entropy_bits": global_entropy_bits,
                "oracle_regime_entropy_bits": oracle_entropy_bits,
                "prefix_regime_entropy_bits": prefix_entropy_bits,
                "global_unigram_bits_per_byte": global_bpb,
                "oracle_regime_bits_per_byte": oracle_bpb,
                "prefix_regime_bits_per_byte": prefix_bpb,
                "oracle_gain_vs_global_bpb": None if global_bpb is None or oracle_bpb is None else float(global_bpb - oracle_bpb),
                "prefix_gain_vs_global_bpb": None if global_bpb is None or prefix_bpb is None else float(global_bpb - prefix_bpb),
            },
            "trainer_export": trainer_export,
            "chunk_rows": chunk_rows,
            "notes": [
                "Regime-conditioned BPB here is a prior-only coarse predictor over world-model chunks, not the full next-token model.",
                "The oracle-regime row is clairvoyant because it uses offline chunk regime labels; the prefix-regime row is the trainer-legal causal version.",
            ],
        },
        {
            "regime_conditioned_bpb_regime_token_probs": regime_token_probs.astype(np.float32, copy=False),
            "regime_conditioned_bpb_prefix_centroids": regime_prefix_centroids.astype(np.float32, copy=False),
            "regime_conditioned_bpb_prefix_labels": prefix_labels.astype(np.int64, copy=False),
            "regime_conditioned_bpb_macro_labels": macro_labels.astype(np.int64, copy=False),
        },
    )


def next_token_loss_decomposition_profile(
    tokens: np.ndarray,
    counts: np.ndarray,
    base_bytes: np.ndarray | None,
    lexical_profile: dict[str, object] | None,
    recent_copy_profile: dict[str, object] | None,
    ppm_oracle: dict[str, object] | None,
    regime_conditioned_bpb: dict[str, object] | None,
) -> dict[str, object]:
    if tokens.size < 2:
        return {"available": False, "reason": "not_enough_tokens"}
    total = float(np.maximum(counts.astype(np.float64, copy=False), 0.0).sum())
    if total <= 0.0:
        return {"available": False, "reason": "empty_counts"}
    global_probs = np.maximum(counts.astype(np.float64, copy=False), 0.0) / total
    target_tokens = tokens[1:].astype(np.int64, copy=False)
    unigram_surprisals = -np.log2(np.clip(global_probs[target_tokens], 1e-12, None))
    unigram_entropy_bits = float(unigram_surprisals.mean()) if unigram_surprisals.size > 0 else 0.0
    unigram_bpb = _bits_per_byte_from_surprisals(unigram_surprisals, target_tokens, base_bytes)

    rows: list[dict[str, object]] = [
        {
            "component": "global_unigram",
            "entropy_bits": unigram_entropy_bits,
            "bits_per_byte": unigram_bpb,
            "causal_trainable": True,
            "clairvoyant": False,
        }
    ]
    if isinstance(lexical_profile, dict) and lexical_profile.get("available"):
        order_rows = [row for row in lexical_profile.get("context_orders", []) if isinstance(row, dict) and row.get("available")]
        order1 = next((row for row in order_rows if int(row.get("order", 0)) == 1), None)
        best = min(order_rows, key=lambda row: float(row.get("conditional_entropy_bits", float("inf"))), default=None)
        if isinstance(order1, dict):
            rows.append(
                {
                    "component": "exact_order1_lexical",
                    "entropy_bits": float(order1.get("conditional_entropy_bits", 0.0)),
                    "bits_per_byte": None if order1.get("conditional_bits_per_byte") is None else float(order1.get("conditional_bits_per_byte")),
                    "causal_trainable": True,
                    "clairvoyant": False,
                }
            )
        if isinstance(best, dict):
            rows.append(
                {
                    "component": f"best_exact_lexical_order_{int(best.get('order', 0))}",
                    "entropy_bits": float(best.get("conditional_entropy_bits", 0.0)),
                    "bits_per_byte": None if best.get("conditional_bits_per_byte") is None else float(best.get("conditional_bits_per_byte")),
                    "causal_trainable": True,
                    "clairvoyant": False,
                }
            )
    if isinstance(recent_copy_profile, dict) and recent_copy_profile.get("available"):
        best = recent_copy_profile.get("best_by_bits_per_byte") or recent_copy_profile.get("best_by_entropy")
        if isinstance(best, dict):
            rows.append(
                {
                    "component": f"best_recent_copy_order_{int(best.get('context_order', 0))}_window_{int(best.get('window', 0))}",
                    "entropy_bits": float(best.get("combined_cross_entropy_bits", 0.0)),
                    "bits_per_byte": None if best.get("combined_bits_per_byte") is None else float(best.get("combined_bits_per_byte")),
                    "causal_trainable": True,
                    "clairvoyant": False,
                }
            )
    if isinstance(regime_conditioned_bpb, dict) and regime_conditioned_bpb.get("available"):
        ablation = regime_conditioned_bpb.get("bpb_ablation") or {}
        rows.append(
            {
                "component": "regime_prior_prefix_inferred",
                "entropy_bits": float(ablation.get("prefix_regime_entropy_bits", 0.0)),
                "bits_per_byte": None if ablation.get("prefix_regime_bits_per_byte") is None else float(ablation.get("prefix_regime_bits_per_byte")),
                "causal_trainable": True,
                "clairvoyant": False,
            }
        )
        rows.append(
            {
                "component": "regime_prior_oracle_known",
                "entropy_bits": float(ablation.get("oracle_regime_entropy_bits", 0.0)),
                "bits_per_byte": None if ablation.get("oracle_regime_bits_per_byte") is None else float(ablation.get("oracle_regime_bits_per_byte")),
                "causal_trainable": False,
                "clairvoyant": True,
            }
        )
    if isinstance(ppm_oracle, dict) and ppm_oracle.get("available"):
        best = ppm_oracle.get("best_by_bits_per_byte") or ppm_oracle.get("best_by_entropy_bits")
        if isinstance(best, dict):
            rows.append(
                {
                    "component": f"ppm_oracle_order_{int(best.get('max_order', 0))}",
                    "entropy_bits": float(best.get("entropy_bits", 0.0)),
                    "bits_per_byte": None if best.get("bits_per_byte") is None else float(best.get("bits_per_byte")),
                    "causal_trainable": False,
                    "clairvoyant": True,
                }
            )
    ppm_row = next((row for row in rows if str(row["component"]).startswith("ppm_oracle_")), None)
    for row in rows:
        row["gain_vs_unigram_bits"] = float(unigram_entropy_bits - float(row["entropy_bits"]))
        if unigram_bpb is not None and row.get("bits_per_byte") is not None:
            row["gain_vs_unigram_bpb"] = float(unigram_bpb - float(row["bits_per_byte"]))
        else:
            row["gain_vs_unigram_bpb"] = None
        if ppm_row is not None:
            row["residual_to_ppm_bits"] = float(float(row["entropy_bits"]) - float(ppm_row["entropy_bits"]))
            if row.get("bits_per_byte") is not None and ppm_row.get("bits_per_byte") is not None:
                row["residual_to_ppm_bpb"] = float(float(row["bits_per_byte"]) - float(ppm_row["bits_per_byte"]))
            else:
                row["residual_to_ppm_bpb"] = None
    best_causal = min(
        [row for row in rows if bool(row.get("causal_trainable")) and row.get("bits_per_byte") is not None],
        key=lambda row: float(row["bits_per_byte"]),
        default=None,
    )
    return {
        "available": True,
        "rows": rows,
        "best_causal_component": best_causal,
        "best_oracle_component": ppm_row,
    }


def posterior_error_taxonomy_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    base_bytes: np.ndarray | None,
    recent_copy_profile: dict[str, object] | None,
    dataset_world_model: dict[str, object] | None,
    dataset_world_model_arrays: dict[str, np.ndarray],
    regime_conditioned_bpb_arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    if dataset_world_model is None or not bool(dataset_world_model.get("available")):
        return {"available": False, "reason": "missing_dataset_world_model"}
    transitions = _build_context_transitions(tokens, vocab_size, 1)
    if transitions is None:
        return {"available": False, "reason": "not_enough_tokens"}
    context_ids, next_tokens = transitions
    lookup = _build_context_distribution_lookup(context_ids, next_tokens)
    top1_lookup = _top1_lookup_from_context_ids(context_ids, next_tokens)
    regime_token_probs = regime_conditioned_bpb_arrays.get("regime_conditioned_bpb_regime_token_probs")
    prefix_labels = regime_conditioned_bpb_arrays.get("regime_conditioned_bpb_prefix_labels")
    chunk_starts = dataset_world_model_arrays.get("dataset_world_model_chunk_starts")
    if lookup is None or top1_lookup is None or regime_token_probs is None or prefix_labels is None or chunk_starts is None:
        return {"available": False, "reason": "missing_lookup_or_regime_arrays"}
    target_probs = _lookup_target_probabilities(lookup, context_ids, next_tokens)
    pos = np.searchsorted(top1_lookup["context_ids"], context_ids)
    seen = (pos < top1_lookup["context_ids"].size) & (top1_lookup["context_ids"][pos] == context_ids)
    top1_prob = np.where(seen, top1_lookup["top1_prob"][pos], 0.0)
    top1_token = np.where(seen, top1_lookup["top1_token"][pos], -1)
    global_probs = np.maximum(counts.astype(np.float64, copy=False), 0.0)
    global_probs = global_probs / max(float(global_probs.sum()), 1e-12)
    rare_threshold = float(np.quantile(global_probs[global_probs > 0.0], 0.2)) if np.any(global_probs > 0.0) else 0.0
    chunk_tokens = int((dataset_world_model.get("chunk_predictive_state_encoder") or {}).get("chunk_tokens", 0))
    prefix_tokens = int((dataset_world_model.get("chunk_predictive_state_encoder") or {}).get("prefix_tokens", 0))

    copy_hit_mask = None
    if isinstance(recent_copy_profile, dict) and recent_copy_profile.get("available"):
        best = recent_copy_profile.get("best_by_bits_per_byte") or recent_copy_profile.get("best_by_entropy")
        if isinstance(best, dict):
            copy_profile = _recent_copy_oracle_profile(tokens, vocab_size, int(best.get("window", 128)), -np.log2(np.clip(global_probs, 1e-12, None)))
            if copy_profile.get("available"):
                copy_hit_mask = copy_profile["hit_mask"]

    if base_bytes is not None:
        token_bytes = np.maximum(base_bytes[next_tokens].astype(np.float64, copy=False), 0.0)
    else:
        token_bytes = np.ones((next_tokens.size,), dtype=np.float64)

    rows_by_name: dict[str, dict[str, object]] = {}
    analyzed = 0.0
    for idx in range(next_tokens.size):
        target_pos = idx + 1
        chunk_idx = int(np.searchsorted(chunk_starts, target_pos, side="right") - 1)
        if chunk_idx < 0 or chunk_idx >= prefix_labels.size:
            continue
        chunk_start = int(chunk_starts[chunk_idx])
        if chunk_tokens > 0 and target_pos >= chunk_start + chunk_tokens:
            continue
        if prefix_tokens > 0 and target_pos < chunk_start + prefix_tokens:
            continue
        target = int(next_tokens[idx])
        conf = float(top1_prob[idx])
        correct = bool(int(top1_token[idx]) == target)
        lexical_prob = float(np.clip(target_probs[idx], 1e-12, None))
        lexical_surprisal = float(-math.log2(lexical_prob))
        regime_id = int(prefix_labels[chunk_idx])
        regime_prob = float(np.clip(regime_token_probs[regime_id, target], 1e-12, None))
        global_prob = float(np.clip(global_probs[target], 1e-12, None))
        copy_hit = bool(copy_hit_mask[idx]) if copy_hit_mask is not None and idx < copy_hit_mask.size else False
        if conf >= 0.75 and correct:
            name = "high_confidence_correct"
        elif conf >= 0.75 and not correct:
            name = "high_confidence_wrong"
        elif copy_hit:
            name = "copy_recoverable"
        elif regime_prob >= 1.5 * global_prob:
            name = "regime_recoverable"
        elif global_prob <= rare_threshold:
            name = "rare_tail"
        else:
            name = "diffuse_uncertainty"
        row = rows_by_name.setdefault(
            name,
            {
                "bucket": name,
                "token_mass": 0.0,
                "byte_mass": 0.0,
                "lexical_surprisal_sum_bits": 0.0,
                "lexical_top1_correct_mass": 0.0,
            },
        )
        row["token_mass"] += 1.0
        row["byte_mass"] += float(token_bytes[idx])
        row["lexical_surprisal_sum_bits"] += lexical_surprisal
        row["lexical_top1_correct_mass"] += 1.0 if correct else 0.0
        analyzed += 1.0

    if analyzed <= 0:
        return {"available": False, "reason": "no_positions_after_prefix_filter"}
    total_bytes = float(sum(float(row["byte_mass"]) for row in rows_by_name.values()))
    rows = []
    for row in rows_by_name.values():
        token_mass = float(row["token_mass"])
        byte_mass = float(row["byte_mass"])
        rows.append(
            {
                "bucket": str(row["bucket"]),
                "token_share": float(token_mass / analyzed),
                "byte_share": float(byte_mass / max(total_bytes, 1e-12)),
                "mean_lexical_surprisal_bits": float(row["lexical_surprisal_sum_bits"] / max(token_mass, 1e-12)),
                "lexical_top1_accuracy": float(row["lexical_top1_correct_mass"] / max(token_mass, 1e-12)),
            }
        )
    rows = sorted(rows, key=lambda item: float(item["byte_share"]), reverse=True)
    return {
        "available": True,
        "posterior_source": "exact_order1_proxy",
        "analyzed_target_positions": int(analyzed),
        "rows": rows,
        "notes": [
            "This taxonomy is built from the exact order-1 proxy posterior, not the trained transformer posterior.",
            "It is meant to separate local lexical misses from copy-recoverable, regime-recoverable, and rare-tail uncertainty.",
        ],
    }


def residual_floor_dashboard_profile(
    next_token_loss_decomposition: dict[str, object] | None,
    posterior_error_taxonomy: dict[str, object] | None,
) -> dict[str, object]:
    if next_token_loss_decomposition is None or not bool(next_token_loss_decomposition.get("available")):
        return {"available": False, "reason": "missing_next_token_loss_decomposition"}
    rows = list(next_token_loss_decomposition.get("rows") or [])
    if not rows:
        return {"available": False, "reason": "no_decomposition_rows"}
    unigram = next((row for row in rows if str(row.get("component")) == "global_unigram"), None)
    best_causal = next_token_loss_decomposition.get("best_causal_component")
    best_oracle = next_token_loss_decomposition.get("best_oracle_component")
    if not isinstance(unigram, dict):
        return {"available": False, "reason": "missing_unigram_row"}
    taxonomy_top = []
    if isinstance(posterior_error_taxonomy, dict) and posterior_error_taxonomy.get("available"):
        taxonomy_top = list((posterior_error_taxonomy.get("rows") or [])[:3])
    return {
        "available": True,
        "baseline_bpb": unigram.get("bits_per_byte"),
        "best_causal_bpb": None if not isinstance(best_causal, dict) else best_causal.get("bits_per_byte"),
        "best_oracle_bpb": None if not isinstance(best_oracle, dict) else best_oracle.get("bits_per_byte"),
        "remaining_gap_best_causal_to_oracle_bpb": (
            None
            if not isinstance(best_causal, dict) or not isinstance(best_oracle, dict) or best_causal.get("bits_per_byte") is None or best_oracle.get("bits_per_byte") is None
            else float(best_causal["bits_per_byte"]) - float(best_oracle["bits_per_byte"])
        ),
        "largest_error_buckets": taxonomy_top,
        "path_summary": [
            "Use the best causal component as the current floor proxy and the oracle component as the structural ceiling.",
            "Use the largest posterior-error buckets to decide whether the next bit should come from lexical, copy, regime, or residual modeling work.",
        ],
    }


def order1_proxy_peak_potential_profile(
    lexical_profile: dict[str, object] | None,
    recent_copy_profile: dict[str, object] | None,
    proxy_calibration: dict[str, object] | None,
    confidence_route_budget: dict[str, object] | None,
    regime_conditioned_bpb: dict[str, object] | None,
    ppm_oracle: dict[str, object] | None,
) -> dict[str, object]:
    if lexical_profile is None or not bool(lexical_profile.get("available")):
        return {"available": False, "reason": "missing_lexical_profile"}
    order1 = next(
        (
            row
            for row in lexical_profile.get("context_orders", [])
            if isinstance(row, dict) and row.get("available") and int(row.get("order", 0)) == 1
        ),
        None,
    )
    if not isinstance(order1, dict):
        return {"available": False, "reason": "missing_order1_profile"}

    def _component_row(
        name: str,
        entropy_bits: float | None,
        bits_per_byte: float | None,
        *,
        causal_trainable: bool,
        clairvoyant: bool,
        source: str,
    ) -> dict[str, object]:
        row = {
            "component": name,
            "source": source,
            "entropy_bits": None if entropy_bits is None else float(entropy_bits),
            "bits_per_byte": None if bits_per_byte is None else float(bits_per_byte),
            "causal_trainable": bool(causal_trainable),
            "clairvoyant": bool(clairvoyant),
        }
        if entropy_bits is not None:
            row["gain_vs_order1_bits"] = float(max(float(order1["conditional_entropy_bits"]) - float(entropy_bits), 0.0))
        else:
            row["gain_vs_order1_bits"] = None
        if order1.get("conditional_bits_per_byte") is not None and bits_per_byte is not None:
            row["gain_vs_order1_bpb"] = float(max(float(order1["conditional_bits_per_byte"]) - float(bits_per_byte), 0.0))
        else:
            row["gain_vs_order1_bpb"] = None
        return row

    rows: list[dict[str, object]] = [
        _component_row(
            "exact_order1_proxy",
            float(order1["conditional_entropy_bits"]),
            None if order1.get("conditional_bits_per_byte") is None else float(order1["conditional_bits_per_byte"]),
            causal_trainable=True,
            clairvoyant=False,
            source="lexical_entropy_profile",
        )
    ]

    if isinstance(recent_copy_profile, dict) and recent_copy_profile.get("available"):
        best_copy = recent_copy_profile.get("best_by_bits_per_byte") or recent_copy_profile.get("best_by_entropy")
        if isinstance(best_copy, dict):
            rows.append(
                _component_row(
                    f"recent_copy_order_{int(best_copy.get('context_order', 0))}_window_{int(best_copy.get('window', 0))}",
                    None
                    if best_copy.get("combined_cross_entropy_bits") is None
                    else float(best_copy.get("combined_cross_entropy_bits", 0.0)),
                    None if best_copy.get("combined_bits_per_byte") is None else float(best_copy.get("combined_bits_per_byte")),
                    causal_trainable=True,
                    clairvoyant=False,
                    source="recent_copy_window_profile",
                )
            )

    ablation = regime_conditioned_bpb.get("bpb_ablation") if isinstance(regime_conditioned_bpb, dict) else None
    if isinstance(ablation, dict):
        rows.append(
            _component_row(
                "prefix_regime_prior",
                None if ablation.get("prefix_regime_entropy_bits") is None else float(ablation.get("prefix_regime_entropy_bits")),
                None if ablation.get("prefix_regime_bits_per_byte") is None else float(ablation.get("prefix_regime_bits_per_byte")),
                causal_trainable=True,
                clairvoyant=False,
                source="regime_conditioned_bpb",
            )
        )
        rows.append(
            _component_row(
                "oracle_regime_prior",
                None if ablation.get("oracle_regime_entropy_bits") is None else float(ablation.get("oracle_regime_entropy_bits")),
                None if ablation.get("oracle_regime_bits_per_byte") is None else float(ablation.get("oracle_regime_bits_per_byte")),
                causal_trainable=False,
                clairvoyant=True,
                source="regime_conditioned_bpb",
            )
        )

    if isinstance(ppm_oracle, dict) and ppm_oracle.get("available"):
        best_ppm = ppm_oracle.get("best_by_bits_per_byte") or ppm_oracle.get("best_by_entropy_bits")
        if isinstance(best_ppm, dict):
            rows.append(
                _component_row(
                    f"ppm_oracle_order_{int(best_ppm.get('max_order', 0))}",
                    None if best_ppm.get("entropy_bits") is None else float(best_ppm.get("entropy_bits")),
                    None if best_ppm.get("bits_per_byte") is None else float(best_ppm.get("bits_per_byte")),
                    causal_trainable=False,
                    clairvoyant=True,
                    source="ppm_oracle_profile",
                )
            )

    calib_row = None
    if isinstance(proxy_calibration, dict) and proxy_calibration.get("available"):
        calib_row = next(
            (
                row
                for row in proxy_calibration.get("exact_profiles", [])
                if isinstance(row, dict) and int(row.get("order", 0)) == 1
            ),
            None,
        )

    safe_route = None
    oracle_route = None
    if isinstance(confidence_route_budget, dict) and confidence_route_budget.get("available"):
        order1_rows = [
            row
            for row in confidence_route_budget.get("rows", [])
            if isinstance(row, dict) and str(row.get("source")) == "exact" and int(row.get("order", 0)) == 1
        ]
        if order1_rows:
            oracle_route = max(order1_rows, key=lambda row: float(row.get("oracle_mixed_top1_upper_bound", 0.0)))
            safe_candidates = [
                row
                for row in order1_rows
                if row.get("false_trust_rate") is not None and float(row.get("false_trust_rate", 1.0)) <= 0.10
            ]
            safe_route = max(safe_candidates, key=lambda row: float(row.get("shortcut_trust_mass", 0.0)), default=None)

    best_trainable = min(
        [row for row in rows if bool(row.get("causal_trainable")) and row.get("bits_per_byte") is not None],
        key=lambda row: float(row["bits_per_byte"]),
        default=None,
    )
    best_clairvoyant = min(
        [row for row in rows if row.get("bits_per_byte") is not None],
        key=lambda row: float(row["bits_per_byte"]),
        default=None,
    )
    ppm_row = next((row for row in rows if str(row.get("component")).startswith("ppm_oracle_")), None)
    return {
        "available": True,
        "order1_proxy_posterior": {
            "conditional_entropy_bits": float(order1["conditional_entropy_bits"]),
            "conditional_bits_per_byte": None
            if order1.get("conditional_bits_per_byte") is None
            else float(order1["conditional_bits_per_byte"]),
            "top1_accuracy": (
                float(calib_row["top1_accuracy"])
                if isinstance(calib_row, dict) and calib_row.get("top1_accuracy") is not None
                else float(order1.get("top1_next_token_coverage", 0.0))
            ),
            "expected_calibration_error": (
                None
                if not isinstance(calib_row, dict) or calib_row.get("expected_calibration_error") is None
                else float(calib_row["expected_calibration_error"])
            ),
            "effective_branching_factor": float(order1.get("effective_branching_factor", 0.0)),
            "dominant_next_ge_50_rate": float(order1.get("dominant_next_ge_50_rate", 0.0)),
            "top4_next_token_coverage": float(order1.get("top4_next_token_coverage", 0.0)),
        },
        "confidence_routing": {
            "best_safe_exact_order1": None
            if safe_route is None
            else {
                "confidence_threshold": float(safe_route.get("confidence_threshold", 0.0)),
                "shortcut_trust_mass": float(safe_route.get("shortcut_trust_mass", 0.0)),
                "trusted_shortcut_accuracy": None
                if safe_route.get("trusted_shortcut_accuracy") is None
                else float(safe_route["trusted_shortcut_accuracy"]),
                "false_trust_rate": None
                if safe_route.get("false_trust_rate") is None
                else float(safe_route["false_trust_rate"]),
                "oracle_mixed_top1_upper_bound": float(safe_route.get("oracle_mixed_top1_upper_bound", 0.0)),
            },
            "best_oracle_exact_order1": None
            if oracle_route is None
            else {
                "confidence_threshold": float(oracle_route.get("confidence_threshold", 0.0)),
                "shortcut_trust_mass": float(oracle_route.get("shortcut_trust_mass", 0.0)),
                "trusted_shortcut_accuracy": None
                if oracle_route.get("trusted_shortcut_accuracy") is None
                else float(oracle_route["trusted_shortcut_accuracy"]),
                "false_trust_rate": None
                if oracle_route.get("false_trust_rate") is None
                else float(oracle_route["false_trust_rate"]),
                "oracle_mixed_top1_upper_bound": float(oracle_route.get("oracle_mixed_top1_upper_bound", 0.0)),
            },
        },
        "component_rows": rows,
        "peak_potential": {
            "best_trainable_component": best_trainable,
            "best_clairvoyant_component": best_clairvoyant,
            "trainable_gain_vs_order1_bits": None
            if not isinstance(best_trainable, dict) or best_trainable.get("gain_vs_order1_bits") is None
            else float(best_trainable["gain_vs_order1_bits"]),
            "trainable_gain_vs_order1_bpb": None
            if not isinstance(best_trainable, dict) or best_trainable.get("gain_vs_order1_bpb") is None
            else float(best_trainable["gain_vs_order1_bpb"]),
            "clairvoyant_gain_vs_order1_bits": None
            if not isinstance(best_clairvoyant, dict) or best_clairvoyant.get("gain_vs_order1_bits") is None
            else float(best_clairvoyant["gain_vs_order1_bits"]),
            "clairvoyant_gain_vs_order1_bpb": None
            if not isinstance(best_clairvoyant, dict) or best_clairvoyant.get("gain_vs_order1_bpb") is None
            else float(best_clairvoyant["gain_vs_order1_bpb"]),
            "remaining_trainable_to_ppm_bpb": (
                None
                if not isinstance(best_trainable, dict)
                or not isinstance(ppm_row, dict)
                or best_trainable.get("bits_per_byte") is None
                or ppm_row.get("bits_per_byte") is None
                else float(best_trainable["bits_per_byte"]) - float(ppm_row["bits_per_byte"])
            ),
        },
        "notes": [
            "This section treats the exact order-1 proxy posterior as the starting point and asks how far copy, regime, and oracle structure can lower the floor from there.",
            "Component gains overlap; the best trainable and clairvoyant rows are shortest-path reference points, not additive reductions.",
        ],
    }


def _load_tokens_memmap(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _compute_basic_signal_bundle(payload: dict[str, object]) -> dict[str, object]:
    sequential_tokens = _load_tokens_memmap(str(payload["sequential_tokens_path"]))
    vocab_size = int(payload["vocab_size"])
    top_k = int(payload["top_k"])
    token_counts = np.asarray(payload["token_counts"], dtype=np.int64)
    eval_lengths = [int(v) for v in payload["eval_lengths"]]
    lag_stats = lag_metrics(sequential_tokens, vocab_size, [int(v) for v in payload["lags"]])
    reuse_stats = context_reuse_profile(sequential_tokens, [int(v) for v in payload["reuse_windows"]])
    eval_candidates = eval_length_candidates(sequential_tokens, eval_lengths)
    marginal_context = marginal_eval_context_gain(eval_candidates)
    context_bands = context_gain_by_distance(lag_stats, [(1, 8), (16, 64), (128, 512), (1024, 2048)])
    transition_geometry = transition_geometry_profile(sequential_tokens, vocab_size, token_counts, top_k)
    recurrence_profile = recurrence_burst_profile(sequential_tokens, max_gap=max(eval_lengths, default=2048))
    recurrence_by_bucket = recurrence_burst_profile_by_frequency(
        sequential_tokens,
        vocab_size=vocab_size,
        counts=token_counts,
        max_gap=max(eval_lengths, default=2048),
    )
    return {
        "lag_stats": lag_stats,
        "reuse_stats": reuse_stats,
        "eval_candidates": eval_candidates,
        "marginal_context": marginal_context,
        "context_bands": context_bands,
        "transition_geometry": transition_geometry,
        "recurrence_profile": recurrence_profile,
        "recurrence_by_bucket": recurrence_by_bucket,
        "top_bigrams": top_bigrams(sequential_tokens, vocab_size, top_k),
    }


def _compute_lexical_bundle(payload: dict[str, object]) -> dict[str, object]:
    sequential_tokens = _load_tokens_memmap(str(payload["sequential_tokens_path"]))
    vocab_size = int(payload["vocab_size"])
    token_counts = np.asarray(payload["token_counts"], dtype=np.int64)
    base_bytes = None if payload.get("base_bytes") is None else np.asarray(payload["base_bytes"], dtype=np.int32)
    lexical_profile = lexical_entropy_profile(
        sequential_tokens,
        vocab_size,
        token_counts,
        base_bytes=base_bytes,
        context_orders=[int(v) for v in payload["lexical_context_orders"]],
    )
    hashed_lexical_profile = hashed_lexical_collision_profile(
        sequential_tokens,
        vocab_size,
        token_counts,
        lexical_profile=lexical_profile,
        base_bytes=base_bytes,
        orders=[int(v) for v in payload["hashed_lexical_orders"]],
        bucket_sizes=[int(v) for v in payload["hashed_lexical_buckets"]],
    )
    recent_copy = recent_copy_window_profile(
        sequential_tokens,
        vocab_size,
        token_counts,
        lexical_profile=lexical_profile,
        base_bytes=base_bytes,
        windows=[int(v) for v in payload["recent_copy_windows"]],
        context_orders=[int(v) for v in payload["recent_copy_orders"]],
    )
    unconstrained_collapse = unconstrained_entropy_collapse_profile(
        sequential_tokens,
        vocab_size,
        token_counts,
        base_bytes=base_bytes,
        context_orders=[int(v) for v in payload["unconstrained_context_orders"]],
        recent_copy_windows=[int(v) for v in payload["unconstrained_recent_copy_windows"]],
    )
    proxy_calibration = proxy_calibration_profile(
        sequential_tokens,
        vocab_size,
        token_counts,
        base_bytes=base_bytes,
        exact_orders=[int(v) for v in payload["lexical_context_orders"]],
        hashed_orders=[int(v) for v in payload["hashed_lexical_orders"]],
        hashed_buckets=[int(v) for v in payload["hashed_lexical_buckets"]],
        n_bins=int(payload["proxy_calibration_bins"]),
    )
    lexical_control = lexical_control_profile(
        sequential_tokens,
        vocab_size,
        base_bytes=base_bytes,
        orders=[int(v) for v in payload["lexical_control_orders"]],
        bucket_sizes=[int(v) for v in payload["lexical_control_buckets"]],
        shortcut_scales=[float(v) for v in payload["lexical_control_shortcut_scales"]],
        smear_gate_inits=[float(v) for v in payload["lexical_control_smear_gates"]],
        route_scales=[float(v) for v in payload["lexical_control_route_scales"]],
        n_bins=int(payload["proxy_calibration_bins"]),
    )
    early_budget_coverage = early_budget_coverage_profile(
        sequential_tokens,
        vocab_size,
        budget_token_points=[int(v) for v in payload["budget_coverage_token_points"]],
        orders=[int(v) for v in payload["budget_coverage_orders"]],
        bucket_sizes=[int(v) for v in payload["budget_coverage_buckets"]],
        top_k_buckets=int(payload["budget_coverage_topk"]),
    )
    confidence_route_budget = confidence_route_budget_profile(
        sequential_tokens,
        vocab_size,
        base_bytes=base_bytes,
        exact_orders=[int(v) for v in payload["confidence_route_exact_orders"]],
        hashed_orders=[int(v) for v in payload["confidence_route_hashed_orders"]],
        hashed_buckets=[int(v) for v in payload["confidence_route_buckets"]],
        confidence_thresholds=[float(v) for v in payload["confidence_route_thresholds"]],
        n_bins=int(payload["proxy_calibration_bins"]),
    )
    higher_order_retention = higher_order_lexical_retention_profile(
        hashed_lexical_profile,
        target_order=3,
        collision_cap=0.65,
        retention_floor=0.60,
    )
    route_calibration = route_calibration_selectivity_profile(
        confidence_route_budget,
        proxy_calibration=proxy_calibration,
        target_hashed_order=3,
        false_trust_cap=0.10,
    )
    return {
        "lexical_profile": lexical_profile,
        "hashed_lexical_profile": hashed_lexical_profile,
        "recent_copy": recent_copy,
        "unconstrained_collapse": unconstrained_collapse,
        "proxy_calibration": proxy_calibration,
        "lexical_control": lexical_control,
        "early_budget_coverage": early_budget_coverage,
        "confidence_route_budget": confidence_route_budget,
        "higher_order_retention": higher_order_retention,
        "route_calibration": route_calibration,
    }


def _compute_transfer_shard_bundle(payload: dict[str, object]) -> dict[str, object]:
    sequential_tokens = _load_tokens_memmap(str(payload["sequential_tokens_path"]))
    val_sequential_tokens = _load_tokens_memmap(str(payload["val_sequential_tokens_path"]))
    vocab_size = int(payload["vocab_size"])
    base_bytes = None if payload.get("base_bytes") is None else np.asarray(payload["base_bytes"], dtype=np.int32)
    files = [Path(path) for path in payload["files"]]
    train_val_transfer = train_val_transfer_profile(
        sequential_tokens,
        val_sequential_tokens,
        vocab_size,
        base_bytes=base_bytes,
        exact_orders=[int(v) for v in payload["transfer_exact_orders"]],
        hashed_orders=[int(v) for v in payload["transfer_hashed_orders"]],
        hashed_buckets=[int(v) for v in payload["transfer_hashed_buckets"]],
    )
    shard_summary_rows, shard_aggregate = shard_summaries(files, vocab_size, int(payload["sample_tokens_per_shard"]), base_bytes)
    shard_clustering = shard_cluster_profile(
        files,
        vocab_size=vocab_size,
        sample_tokens_per_shard=int(payload["sample_tokens_per_shard"]),
        prefix_tokens=int(payload["shard_prefix_tokens"]),
        cluster_ks=[int(v) for v in payload["shard_cluster_k_values"]],
        neighbor_topk=int(payload["shard_neighbor_topk"]),
        top_k=int(payload["top_k"]),
    )
    shard_phase = shard_phase_drift_profile(
        files,
        vocab_size=vocab_size,
        sample_tokens_per_shard=int(payload["sample_tokens_per_shard"]),
        prefix_tokens=int(payload["shard_prefix_tokens"]),
        phase_segments=int(payload["shard_phase_segments"]),
    )
    return {
        "train_val_transfer": train_val_transfer,
        "shard_summary_rows": shard_summary_rows,
        "shard_aggregate": shard_aggregate,
        "shard_clustering": shard_clustering,
        "shard_phase": shard_phase,
    }


def _compute_state_bundle(payload: dict[str, object]) -> dict[str, object]:
    sequential_tokens = _load_tokens_memmap(str(payload["sequential_tokens_path"]))
    vocab_size = int(payload["vocab_size"])
    token_counts = np.asarray(payload["token_counts"], dtype=np.int64)
    base_bytes = None if payload.get("base_bytes") is None else np.asarray(payload["base_bytes"], dtype=np.int32)
    tokenizer_meta = payload.get("tokenizer_meta")
    spectral_eigenbases, spectral_eigenbasis_arrays = spectral_eigenbasis_profile(
        sequential_tokens,
        vocab_size,
        token_counts,
        rank=int(payload["spectral_eigen_rank"]),
        top_k=int(payload["top_k"]),
        tokenizer_meta=tokenizer_meta,
    )
    lagged_spectral_eigenbases, lagged_spectral_eigenbasis_arrays = lagged_spectral_eigenbasis_profiles(
        sequential_tokens,
        vocab_size,
        rank=int(payload["spectral_eigen_rank"]),
        top_k=int(payload["top_k"]),
        lags=[int(v) for v in payload["spectral_lags"]],
        tokenizer_meta=tokenizer_meta,
    )
    spectral_recommendation = spectral_basis_recommendations(
        spectral_eigenbases,
        lagged_spectral_eigenbases,
    )
    past_future_cca, past_future_cca_arrays = past_future_cca_profile(
        sequential_tokens,
        vocab_size,
        past_window=int(payload["cca_past_window"]),
        future_window=int(payload["cca_future_window"]),
        sketch_dim=int(payload["cca_sketch_dim"]),
        ranks=[int(v) for v in payload["cca_ranks"]],
        max_prefixes=int(payload["cca_max_prefixes"]),
        regularization=float(payload["cca_regularization"]),
    )
    predictive_state_transfer = predictive_state_transfer_spectrum(
        sequential_tokens,
        vocab_size,
        past_window=int(payload["cca_past_window"]),
        future_window=int(payload["cca_future_window"]),
        sketch_dim=int(payload["cca_sketch_dim"]),
        ranks=[int(v) for v in payload["cca_ranks"]],
        max_prefixes=int(payload["cca_max_prefixes"]),
        regularization=float(payload["cca_regularization"]),
        taus=[int(v) for v in payload["transfer_taus"]],
        num_clusters=int(payload["transfer_clusters"]),
        kmeans_iters=int(payload["transfer_kmeans_iters"]),
    )
    minimal_causal_state, minimal_causal_state_arrays = minimal_causal_state_profile(
        sequential_tokens,
        vocab_size,
        base_bytes=base_bytes,
        past_window=int(payload["cca_past_window"]),
        future_window=int(payload["cca_future_window"]),
        sketch_dim=int(payload["cca_sketch_dim"]),
        ranks=[int(v) for v in payload["causal_state_ranks"]],
        max_prefixes=int(payload["causal_state_max_prefixes"]),
        regularization=float(payload["cca_regularization"]),
        holdout_fraction=float(payload["causal_state_holdout_fraction"]),
        ridge=float(payload["causal_state_ridge"]),
        near_best_bpb_tol=float(payload["causal_state_near_best_bpb_tol"]),
        near_best_bits_tol=float(payload["causal_state_near_best_bits_tol"]),
    )
    future_signature, future_signature_arrays = future_signature_profile(
        sequential_tokens,
        vocab_size,
        past_window=int(payload["cca_past_window"]),
        sketch_dim=int(payload["cca_sketch_dim"]),
        horizons=[int(v) for v in payload["causal_machine_future_horizons"]],
        max_prefixes=int(payload["causal_state_max_prefixes"]),
    )
    causal_state_reconstruction, causal_state_arrays = causal_state_reconstruction_profile(
        future_signature_arrays,
        vocab_size,
        base_bytes=base_bytes,
        state_counts=[int(v) for v in payload["causal_machine_state_counts"]],
        holdout_fraction=float(payload["causal_state_holdout_fraction"]),
        kmeans_iters=int(payload["causal_machine_kmeans_iters"]),
        near_best_bpb_tol=float(payload["causal_state_near_best_bpb_tol"]),
        near_best_bits_tol=float(payload["causal_state_near_best_bits_tol"]),
    )
    state_transition_determinism = state_transition_determinism_profile(
        future_signature_arrays,
        causal_state_arrays,
    )
    state_entropy_floor = state_entropy_floor_profile(
        future_signature_arrays,
        causal_state_arrays,
        base_bytes=base_bytes,
    )
    causal_state_decodability, causal_state_decodability_arrays = causal_state_decodability_profile(
        sequential_tokens,
        vocab_size,
        past_window=int(payload["cca_past_window"]),
        sketch_dim=int(payload["cca_sketch_dim"]),
        future_signature_arrays=future_signature_arrays,
        causal_state_arrays=causal_state_arrays,
        past_future_cca_arrays=past_future_cca_arrays,
        holdout_fraction=float(payload["causal_state_holdout_fraction"]),
        ridge=float(payload["causal_state_ridge"]),
        base_bytes=base_bytes,
    )
    causal_state_transition_learnability, causal_state_transition_learnability_arrays = causal_state_transition_learnability_profile(
        sequential_tokens,
        vocab_size,
        past_window=int(payload["cca_past_window"]),
        sketch_dim=int(payload["cca_sketch_dim"]),
        future_signature_arrays=future_signature_arrays,
        causal_state_arrays=causal_state_arrays,
        holdout_fraction=float(payload["causal_state_holdout_fraction"]),
        ridge=float(payload["causal_state_ridge"]),
    )
    causal_state_multi_horizon_sufficiency = causal_state_multi_horizon_sufficiency_profile(
        future_signature_arrays,
        causal_state_arrays,
        base_bytes=base_bytes,
    )
    causal_state_merge_error = causal_state_merge_error_profile(causal_state_reconstruction)
    causal_state_residual_geometry, causal_state_residual_geometry_arrays = causal_state_residual_geometry_profile(
        future_signature_arrays,
        causal_state_arrays,
        ranks=[int(v) for v in payload["causal_state_ranks"]],
    )
    strict_online_state_eval = strict_online_state_eval_profile(
        sequential_tokens,
        vocab_size,
        base_bytes=base_bytes,
        past_window=int(payload["cca_past_window"]),
        sketch_dim=int(payload["cca_sketch_dim"]),
        causal_state_arrays=causal_state_arrays,
        causal_state_decodability=causal_state_decodability,
        causal_state_decodability_arrays=causal_state_decodability_arrays,
        past_future_cca_arrays=past_future_cca_arrays,
        max_eval_tokens=int(payload["strict_online_max_eval_tokens"]),
        horizons=[int(v) for v in payload["strict_online_horizons"]],
    )
    tensor_network_state_frontier = tensor_network_state_frontier_profile(
        sequential_tokens,
        vocab_size,
        base_bytes=base_bytes,
        past_window=int(payload["cca_past_window"]),
        sketch_dim=int(payload["cca_sketch_dim"]),
        causal_state_arrays=causal_state_arrays,
        causal_state_decodability=causal_state_decodability,
        causal_state_decodability_arrays=causal_state_decodability_arrays,
        past_future_cca_arrays=past_future_cca_arrays,
        max_eval_tokens=int(payload["strict_online_max_eval_tokens"]),
        bond_dims=[int(v) for v in payload["tensor_bond_dims"]],
    )
    predictive_state_compression, predictive_state_arrays = predictive_state_compression_profile(
        sequential_tokens,
        vocab_size,
        order=int(payload["predictive_state_order"]),
        ranks=[int(v) for v in payload["predictive_state_ranks"]],
        max_prefixes=int(payload["predictive_state_max_prefixes"]),
    )
    predictive_state_recommendation = predictive_state_recommendations(
        predictive_state_compression,
    )
    predictive_state_transition = predictive_state_transition_profile(
        sequential_tokens,
        vocab_size,
        order=int(payload["predictive_state_order"]),
        ranks=[int(v) for v in payload["predictive_state_ranks"]],
        max_prefixes=int(payload["predictive_state_max_prefixes"]),
    )
    predictive_state_transition_recommendation = predictive_state_transition_recommendations(
        predictive_state_transition,
    )
    oracle_backoff = oracle_backoff_profile(
        sequential_tokens,
        vocab_size,
        token_counts,
        base_bytes=base_bytes,
        orders=[int(v) for v in payload["oracle_backoff_orders"]],
        max_eval_tokens=int(payload["oracle_backoff_max_eval_tokens"]),
    )
    ppm_oracle = ppm_oracle_profile(
        sequential_tokens,
        vocab_size,
        token_counts,
        base_bytes=base_bytes,
        orders=[int(v) for v in payload["ppm_orders"]],
        max_eval_tokens=int(payload["ppm_max_eval_tokens"]),
    )
    ppm_oracle_reco = ppm_oracle_recommendations(ppm_oracle)
    dataset_world_model, dataset_world_model_arrays = dataset_world_model_profile(
        sequential_tokens,
        vocab_size,
        token_counts,
        past_window=int(payload["cca_past_window"]),
        future_window=int(payload["cca_future_window"]),
        sketch_dim=int(payload["cca_sketch_dim"]),
        ranks=[int(v) for v in payload["cca_ranks"]],
        max_prefixes=int(payload["cca_max_prefixes"]),
        regularization=float(payload["cca_regularization"]),
        chunk_tokens=int(payload["world_model_chunk_tokens"]),
        chunk_stride=int(payload["world_model_chunk_stride"]),
        prefix_tokens=int(payload["world_model_prefix_tokens"]),
        regime_counts=[int(v) for v in payload["world_model_regime_counts"]],
        kmeans_iters=int(payload["world_model_kmeans_iters"]),
        top_tokens=int(payload["world_model_top_tokens"]),
    )
    dataset_world_model_reco = dataset_world_model_recommendations(dataset_world_model)
    regime_conditioned_bpb, regime_conditioned_bpb_arrays = regime_conditioned_bpb_profile(
        token_counts,
        base_bytes,
        dataset_world_model,
        dataset_world_model_arrays,
    )
    return {
        "spectral_eigenbases": spectral_eigenbases,
        "spectral_eigenbasis_arrays": spectral_eigenbasis_arrays,
        "lagged_spectral_eigenbases": lagged_spectral_eigenbases,
        "lagged_spectral_eigenbasis_arrays": lagged_spectral_eigenbasis_arrays,
        "spectral_recommendation": spectral_recommendation,
        "past_future_cca": past_future_cca,
        "past_future_cca_arrays": past_future_cca_arrays,
        "past_future_cca_reco": past_future_cca_recommendations(past_future_cca),
        "predictive_state_transfer": predictive_state_transfer,
        "minimal_causal_state": minimal_causal_state,
        "minimal_causal_state_arrays": minimal_causal_state_arrays,
        "minimal_causal_state_reco": minimal_causal_state_recommendations(minimal_causal_state),
        "future_signature": future_signature,
        "future_signature_arrays": future_signature_arrays,
        "causal_state_reconstruction": causal_state_reconstruction,
        "causal_state_arrays": causal_state_arrays,
        "state_transition_determinism": state_transition_determinism,
        "state_entropy_floor": state_entropy_floor,
        "causal_state_decodability": causal_state_decodability,
        "causal_state_decodability_arrays": causal_state_decodability_arrays,
        "causal_state_transition_learnability": causal_state_transition_learnability,
        "causal_state_transition_learnability_arrays": causal_state_transition_learnability_arrays,
        "causal_state_multi_horizon_sufficiency": causal_state_multi_horizon_sufficiency,
        "causal_state_merge_error": causal_state_merge_error,
        "causal_state_residual_geometry": causal_state_residual_geometry,
        "causal_state_residual_geometry_arrays": causal_state_residual_geometry_arrays,
        "strict_online_state_eval": strict_online_state_eval,
        "tensor_network_state_frontier": tensor_network_state_frontier,
        "predictive_state_compression": predictive_state_compression,
        "predictive_state_arrays": predictive_state_arrays,
        "predictive_state_recommendation": predictive_state_recommendation,
        "predictive_state_transition": predictive_state_transition,
        "predictive_state_transition_recommendation": predictive_state_transition_recommendation,
        "oracle_backoff": oracle_backoff,
        "ppm_oracle": ppm_oracle,
        "ppm_oracle_reco": ppm_oracle_reco,
        "dataset_world_model": dataset_world_model,
        "dataset_world_model_arrays": dataset_world_model_arrays,
        "dataset_world_model_reco": dataset_world_model_reco,
        "regime_conditioned_bpb": regime_conditioned_bpb,
        "regime_conditioned_bpb_arrays": regime_conditioned_bpb_arrays,
    }


def _run_profile_bundle(bundle_name: str, payload: dict[str, object]) -> dict[str, object]:
    if bundle_name == "basic_signal":
        return _compute_basic_signal_bundle(payload)
    if bundle_name == "lexical":
        return _compute_lexical_bundle(payload)
    if bundle_name == "transfer_shard":
        return _compute_transfer_shard_bundle(payload)
    if bundle_name == "state":
        return _compute_state_bundle(payload)
    raise ValueError(f"Unknown profile bundle: {bundle_name}")


def _build_context_transitions(tokens: np.ndarray, vocab_size: int, order: int) -> tuple[np.ndarray, np.ndarray] | None:
    if order <= 0 or tokens.size <= order:
        return None
    context_ids = np.zeros((tokens.size - order,), dtype=np.int64)
    for offset in range(order):
        context_ids *= np.int64(vocab_size)
        context_ids += tokens[offset : tokens.size - order + offset].astype(np.int64, copy=False)
    next_tokens = tokens[order:].astype(np.int64, copy=False)
    if context_ids.size <= 0 or next_tokens.size <= 0:
        return None
    return context_ids, next_tokens


def _top1_lookup_from_context_ids(context_ids: np.ndarray, next_tokens: np.ndarray) -> dict[str, np.ndarray] | None:
    if context_ids.size <= 0 or next_tokens.size <= 0:
        return None
    sort_idx = np.lexsort((next_tokens, context_ids))
    ctx_sorted = context_ids[sort_idx]
    next_sorted = next_tokens[sort_idx]
    pair_change = np.ones((ctx_sorted.size,), dtype=bool)
    pair_change[1:] = (ctx_sorted[1:] != ctx_sorted[:-1]) | (next_sorted[1:] != next_sorted[:-1])
    pair_starts = np.flatnonzero(pair_change)
    pair_ends = np.append(pair_starts[1:], ctx_sorted.size)
    pair_ctx = ctx_sorted[pair_starts]
    pair_counts = (pair_ends - pair_starts).astype(np.int64, copy=False)
    pair_next = next_sorted[pair_starts]

    ctx_change = np.ones((pair_ctx.size,), dtype=bool)
    ctx_change[1:] = pair_ctx[1:] != pair_ctx[:-1]
    ctx_starts = np.flatnonzero(ctx_change)
    ctx_ends = np.append(ctx_starts[1:], pair_ctx.size)

    unique_ctx = pair_ctx[ctx_starts].astype(np.int64, copy=False)
    top1_token = np.zeros((unique_ctx.size,), dtype=np.int64)
    top1_prob = np.zeros((unique_ctx.size,), dtype=np.float64)
    total_count = np.zeros((unique_ctx.size,), dtype=np.int64)
    branching = np.zeros((unique_ctx.size,), dtype=np.int64)

    for row_idx, (start, end) in enumerate(zip(ctx_starts, ctx_ends, strict=True)):
        local_counts = pair_counts[start:end]
        local_next = pair_next[start:end]
        total = int(local_counts.sum())
        if total <= 0:
            continue
        order_idx = np.argsort(local_counts)[::-1]
        top1_token[row_idx] = int(local_next[order_idx[0]])
        top1_prob[row_idx] = float(local_counts[order_idx[0]] / total)
        total_count[row_idx] = total
        branching[row_idx] = int(local_counts.size)
    return {
        "context_ids": unique_ctx,
        "top1_token": top1_token,
        "top1_prob": top1_prob,
        "total_count": total_count,
        "branching": branching,
    }


def _byte_weighted_rate(mask: np.ndarray, token_bytes: np.ndarray | None) -> float | None:
    if token_bytes is None or token_bytes.size <= 0:
        return None
    total = float(token_bytes.sum())
    if total <= 0.0:
        return None
    return float(token_bytes[mask].sum() / total)


def _byte_weighted_accuracy(correct_mask: np.ndarray, token_bytes: np.ndarray | None) -> float | None:
    if token_bytes is None or token_bytes.size <= 0:
        return None
    total = float(token_bytes.sum())
    if total <= 0.0:
        return None
    return float(token_bytes[correct_mask].sum() / total)


def _conditional_stats_from_context_ids(
    context_ids: np.ndarray,
    next_tokens: np.ndarray,
    unigram_entropy: float,
    base_bytes: np.ndarray | None = None,
) -> dict[str, object]:
    if context_ids.size <= 0 or next_tokens.size <= 0:
        return {
            "available": False,
            "reason": "no_transitions",
        }
    sort_idx = np.lexsort((next_tokens, context_ids))
    ctx_sorted = context_ids[sort_idx]
    next_sorted = next_tokens[sort_idx]
    pair_change = np.ones((ctx_sorted.size,), dtype=bool)
    pair_change[1:] = (ctx_sorted[1:] != ctx_sorted[:-1]) | (next_sorted[1:] != next_sorted[:-1])
    pair_starts = np.flatnonzero(pair_change)
    pair_ends = np.append(pair_starts[1:], ctx_sorted.size)
    pair_ctx = ctx_sorted[pair_starts]
    pair_counts = (pair_ends - pair_starts).astype(np.int64, copy=False)
    pair_next = next_sorted[pair_starts]

    ctx_change = np.ones((pair_ctx.size,), dtype=bool)
    ctx_change[1:] = pair_ctx[1:] != pair_ctx[:-1]
    ctx_starts = np.flatnonzero(ctx_change)
    ctx_ends = np.append(ctx_starts[1:], pair_ctx.size)

    total_pairs = int(pair_counts.sum())
    weighted_row_entropy_sum = 0.0
    top1_mass = 0
    top4_mass = 0
    top8_mass = 0
    top1_byte_mass = 0.0
    top4_byte_mass = 0.0
    top8_byte_mass = 0.0
    byte_weighted_row_surprisal_sum = 0.0
    deterministic_contexts = 0
    strongly_biased_contexts = 0
    continuation_counts: list[int] = []
    target_byte_total = 0.0
    if base_bytes is not None:
        safe_base_bytes = np.maximum(base_bytes.astype(np.float64, copy=False), 0.0)
    else:
        safe_base_bytes = None

    for start, end in zip(ctx_starts, ctx_ends, strict=True):
        local_counts = pair_counts[start:end]
        local_next_tokens = pair_next[start:end]
        total = int(local_counts.sum())
        if total <= 0:
            continue
        probs = local_counts.astype(np.float64) / total
        surprisals = -np.log2(np.clip(probs, 1e-12, None))
        weighted_row_entropy_sum += float((probs * surprisals).sum()) * total
        if safe_base_bytes is not None:
            local_bytes = safe_base_bytes[local_next_tokens]
            byte_mass = local_counts.astype(np.float64) * local_bytes
            target_byte_total += float(byte_mass.sum())
            byte_weighted_row_surprisal_sum += float((byte_mass * surprisals).sum())
        else:
            local_bytes = None
        order_idx = np.argsort(local_counts)[::-1]
        sorted_counts = local_counts[order_idx]
        top1 = int(sorted_counts[0]) if sorted_counts.size else 0
        top1_mass += top1
        top4_mass += int(sorted_counts[:4].sum())
        top8_mass += int(sorted_counts[:8].sum())
        if local_bytes is not None:
            sorted_byte_mass = byte_mass[order_idx]
            top1_byte_mass += float(sorted_byte_mass[:1].sum())
            top4_byte_mass += float(sorted_byte_mass[:4].sum())
            top8_byte_mass += float(sorted_byte_mass[:8].sum())
        continuation_counts.append(int(local_counts.size))
        if local_counts.size <= 1:
            deterministic_contexts += 1
        if total > 0 and (top1 / total) >= 0.5:
            strongly_biased_contexts += 1

    active_contexts = max(int(len(continuation_counts)), 1)
    conditional_entropy = float(weighted_row_entropy_sum / max(total_pairs, 1))
    mutual_information = max(unigram_entropy - conditional_entropy, 0.0)
    mean_target_bytes = float(target_byte_total / max(total_pairs, 1)) if safe_base_bytes is not None else None
    conditional_bits_per_byte = (
        float(conditional_entropy / max(mean_target_bytes, 1e-12))
        if mean_target_bytes is not None and mean_target_bytes > 0.0
        else None
    )
    byte_weighted_conditional_entropy = (
        float(byte_weighted_row_surprisal_sum / max(target_byte_total, 1e-12)) if target_byte_total > 0.0 else None
    )
    return {
        "available": True,
        "active_contexts": int(len(continuation_counts)),
        "avg_continuations_per_context": float(np.mean(continuation_counts)) if continuation_counts else 0.0,
        "median_continuations_per_context": float(np.median(continuation_counts)) if continuation_counts else 0.0,
        "conditional_entropy_bits": conditional_entropy,
        "mutual_information_bits": mutual_information,
        "normalized_mi": float(mutual_information / unigram_entropy) if unigram_entropy > 0 else 0.0,
        "effective_branching_factor": float(2.0**conditional_entropy),
        "top1_next_token_coverage": float(top1_mass / max(total_pairs, 1)),
        "top4_next_token_coverage": float(top4_mass / max(total_pairs, 1)),
        "top8_next_token_coverage": float(top8_mass / max(total_pairs, 1)),
        "top1_next_byte_coverage": float(top1_byte_mass / max(target_byte_total, 1e-12)) if target_byte_total > 0.0 else None,
        "top4_next_byte_coverage": float(top4_byte_mass / max(target_byte_total, 1e-12)) if target_byte_total > 0.0 else None,
        "top8_next_byte_coverage": float(top8_byte_mass / max(target_byte_total, 1e-12)) if target_byte_total > 0.0 else None,
        "mean_target_bytes": mean_target_bytes,
        "conditional_bits_per_byte": conditional_bits_per_byte,
        "byte_weighted_conditional_entropy_bits": byte_weighted_conditional_entropy,
        "deterministic_context_rate": float(deterministic_contexts / active_contexts),
        "dominant_next_ge_50_rate": float(strongly_biased_contexts / active_contexts),
        "total_transition_pairs": int(total_pairs),
    }


def _hash_context_ids(context_ids: np.ndarray, bucket_count: int) -> np.ndarray:
    x = context_ids.astype(np.uint64, copy=False)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x % np.uint64(max(bucket_count, 1))).astype(np.int64, copy=False)


def _context_entropy_stats(
    tokens: np.ndarray,
    vocab_size: int,
    order: int,
    unigram_entropy: float,
    base_bytes: np.ndarray | None = None,
) -> dict[str, object]:
    transitions = _build_context_transitions(tokens, vocab_size, order)
    if transitions is None:
        return {
            "available": False,
            "order": int(order),
            "reason": "not_enough_tokens",
        }
    context_ids, next_tokens = transitions
    stats = _conditional_stats_from_context_ids(context_ids, next_tokens, unigram_entropy, base_bytes=base_bytes)
    return {
        **stats,
        "available": bool(stats.get("available", False)),
        "order": int(order),
    }


def _safe_exact_context_order_limit(vocab_size: int) -> int:
    if vocab_size <= 1:
        return 1
    return max(int(math.floor(math.log((2**63) - 1) / math.log(max(vocab_size, 2)))), 1)


def _exact_context_target_surprisal_profile(
    tokens: np.ndarray,
    vocab_size: int,
    order: int,
) -> dict[str, object]:
    if order > _safe_exact_context_order_limit(vocab_size):
        return {
            "available": False,
            "order": int(order),
            "reason": "order_exceeds_exact_int64_context_limit",
            "max_exact_order": int(_safe_exact_context_order_limit(vocab_size)),
        }
    transitions = _build_context_transitions(tokens, vocab_size, order)
    if transitions is None:
        return {
            "available": False,
            "order": int(order),
            "reason": "not_enough_tokens",
        }
    context_ids, next_tokens = transitions
    sort_idx = np.lexsort((next_tokens, context_ids))
    ctx_sorted = context_ids[sort_idx]
    next_sorted = next_tokens[sort_idx]
    pair_change = np.ones((ctx_sorted.size,), dtype=bool)
    pair_change[1:] = (ctx_sorted[1:] != ctx_sorted[:-1]) | (next_sorted[1:] != next_sorted[:-1])
    pair_starts = np.flatnonzero(pair_change)
    pair_ends = np.append(pair_starts[1:], ctx_sorted.size)
    pair_ctx = ctx_sorted[pair_starts]
    pair_counts = (pair_ends - pair_starts).astype(np.int64, copy=False)
    ctx_change = np.ones((pair_ctx.size,), dtype=bool)
    ctx_change[1:] = pair_ctx[1:] != pair_ctx[:-1]
    ctx_starts = np.flatnonzero(ctx_change)
    ctx_ends = np.append(ctx_starts[1:], pair_ctx.size)
    context_totals = np.add.reduceat(pair_counts, ctx_starts).astype(np.float64, copy=False)
    pair_context_totals = np.repeat(context_totals, ctx_ends - ctx_starts)
    pair_probs = pair_counts.astype(np.float64, copy=False) / np.clip(pair_context_totals, 1.0, None)
    pair_surprisals = -np.log2(np.clip(pair_probs, 1e-12, None))
    sorted_surprisals = np.repeat(pair_surprisals, pair_counts)
    surprisals = np.empty((next_tokens.size,), dtype=np.float64)
    surprisals[sort_idx] = sorted_surprisals
    return {
        "available": True,
        "order": int(order),
        "target_positions": np.arange(order, tokens.size, dtype=np.int64),
        "target_tokens": next_tokens.astype(np.int64, copy=False),
        "surprisal_bits": surprisals,
        "active_contexts": int(ctx_starts.size),
    }


def _recent_copy_oracle_profile(
    tokens: np.ndarray,
    vocab_size: int,
    window: int,
    unigram_surprisal_by_token: np.ndarray,
) -> dict[str, object]:
    if tokens.size < 2 or window <= 0:
        return {
            "available": False,
            "window": int(window),
            "reason": "not_enough_tokens",
        }
    history_counts = np.zeros((vocab_size,), dtype=np.int32)
    q: deque[int] = deque()
    hit_mask = np.zeros((tokens.size,), dtype=bool)
    for pos, token in enumerate(map(int, tokens)):
        if pos > 0:
            hit_mask[pos] = history_counts[token] > 0
        q.append(token)
        history_counts[token] += 1
        if len(q) > window:
            old = q.popleft()
            history_counts[old] -= 1
    target_positions = np.arange(1, tokens.size, dtype=np.int64)
    target_tokens = tokens[1:].astype(np.int64, copy=False)
    target_hit_mask = hit_mask[1:]
    surprisal_bits = np.where(target_hit_mask, 0.0, unigram_surprisal_by_token[target_tokens])
    return {
        "available": True,
        "window": int(window),
        "target_positions": target_positions,
        "target_tokens": target_tokens,
        "hit_mask": target_hit_mask,
        "surprisal_bits": surprisal_bits.astype(np.float64, copy=False),
    }


def _bits_per_byte_from_surprisals(
    surprisals: np.ndarray,
    target_tokens: np.ndarray,
    base_bytes: np.ndarray | None,
) -> float | None:
    if base_bytes is None or surprisals.size <= 0 or target_tokens.size <= 0:
        return None
    token_bytes = np.maximum(base_bytes[target_tokens].astype(np.float64, copy=False), 0.0)
    total_bytes = float(token_bytes.sum())
    if total_bytes <= 0.0:
        return None
    return float(surprisals.sum() / total_bytes)


def _collapse_fraction(unigram_entropy: float, remaining_entropy: float) -> float:
    if unigram_entropy <= 0.0:
        return 0.0
    return float(max(unigram_entropy - remaining_entropy, 0.0) / unigram_entropy)


def unconstrained_entropy_collapse_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    base_bytes: np.ndarray | None = None,
    context_orders: list[int] | None = None,
    recent_copy_windows: list[int] | None = None,
) -> dict[str, object]:
    if tokens.size < 2:
        return {"available": False, "reason": "not_enough_tokens"}
    unigram_entropy = entropy_from_counts(counts)
    total_count = max(int(counts.sum()), 1)
    unigram_probs = counts.astype(np.float64, copy=False) / total_count
    unigram_surprisal_by_token = -np.log2(np.clip(unigram_probs, 1e-12, None))
    requested_orders = sorted({int(v) for v in (context_orders or [1, 2, 3, 4, 5]) if int(v) > 0})
    requested_windows = sorted({int(v) for v in (recent_copy_windows or [128, 256, 512, 1024, 2048]) if int(v) > 0})
    max_exact_order = _safe_exact_context_order_limit(vocab_size)

    exact_rows: list[dict[str, object]] = []
    exact_loss_profiles: dict[int, dict[str, object]] = {}
    available_exact_orders: list[int] = []
    prev_exact_entropy = unigram_entropy
    for order in requested_orders:
        loss_profile = _exact_context_target_surprisal_profile(tokens, vocab_size, order)
        if not loss_profile.get("available"):
            exact_rows.append(
                {
                    "available": False,
                    "order": int(order),
                    "reason": loss_profile.get("reason", "unavailable"),
                    "max_exact_order": loss_profile.get("max_exact_order"),
                }
            )
            continue
        target_tokens = loss_profile["target_tokens"]
        surprisal_bits = loss_profile["surprisal_bits"]
        conditional_entropy_bits = float(np.mean(surprisal_bits)) if surprisal_bits.size > 0 else 0.0
        conditional_bits_per_byte = _bits_per_byte_from_surprisals(surprisal_bits, target_tokens, base_bytes)
        row = {
            "available": True,
            "order": int(order),
            "active_target_positions": int(target_tokens.size),
            "active_contexts": int(loss_profile["active_contexts"]),
            "conditional_entropy_bits": conditional_entropy_bits,
            "conditional_bits_per_byte": conditional_bits_per_byte,
            "incremental_entropy_reduction_bits": float(max(prev_exact_entropy - conditional_entropy_bits, 0.0)),
            "collapse_fraction_relative_to_unigram": _collapse_fraction(unigram_entropy, conditional_entropy_bits),
            "remaining_entropy_fraction_relative_to_unigram": (
                float(conditional_entropy_bits / unigram_entropy) if unigram_entropy > 0.0 else 0.0
            ),
        }
        exact_rows.append(row)
        exact_loss_profiles[int(order)] = loss_profile
        available_exact_orders.append(int(order))
        prev_exact_entropy = conditional_entropy_bits

    recent_copy_rows: list[dict[str, object]] = []
    recent_copy_profiles: dict[int, dict[str, object]] = {}
    prev_recent_copy_entropy = unigram_entropy
    for window in requested_windows:
        copy_profile = _recent_copy_oracle_profile(tokens, vocab_size, window, unigram_surprisal_by_token)
        if not copy_profile.get("available"):
            recent_copy_rows.append(
                {
                    "available": False,
                    "window": int(window),
                    "reason": copy_profile.get("reason", "unavailable"),
                }
            )
            continue
        target_tokens = copy_profile["target_tokens"]
        hit_mask = copy_profile["hit_mask"]
        surprisal_bits = copy_profile["surprisal_bits"]
        oracle_entropy_bits = float(np.mean(surprisal_bits)) if surprisal_bits.size > 0 else 0.0
        oracle_bits_per_byte = _bits_per_byte_from_surprisals(surprisal_bits, target_tokens, base_bytes)
        row = {
            "available": True,
            "window": int(window),
            "active_target_positions": int(target_tokens.size),
            "copy_hit_rate": float(np.mean(hit_mask)) if hit_mask.size > 0 else 0.0,
            "oracle_floor_entropy_bits": oracle_entropy_bits,
            "oracle_floor_bits_per_byte": oracle_bits_per_byte,
            "incremental_entropy_reduction_bits": float(max(prev_recent_copy_entropy - oracle_entropy_bits, 0.0)),
            "collapse_fraction_relative_to_unigram": _collapse_fraction(unigram_entropy, oracle_entropy_bits),
            "remaining_entropy_fraction_relative_to_unigram": (
                float(oracle_entropy_bits / unigram_entropy) if unigram_entropy > 0.0 else 0.0
            ),
        }
        recent_copy_rows.append(row)
        recent_copy_profiles[int(window)] = copy_profile
        prev_recent_copy_entropy = oracle_entropy_bits

    hybrid_rows: list[dict[str, object]] = []
    for order in available_exact_orders:
        exact_profile = exact_loss_profiles[order]
        exact_positions = exact_profile["target_positions"]
        exact_target_tokens = exact_profile["target_tokens"]
        exact_surprisals = exact_profile["surprisal_bits"]
        exact_row = next(item for item in exact_rows if item.get("available") and int(item["order"]) == order)
        for window in requested_windows:
            copy_profile = recent_copy_profiles.get(int(window))
            if copy_profile is None or not copy_profile.get("available"):
                continue
            copy_surprisals = copy_profile["surprisal_bits"][exact_positions - 1]
            hybrid_surprisals = np.minimum(exact_surprisals, copy_surprisals)
            hybrid_entropy_bits = float(np.mean(hybrid_surprisals)) if hybrid_surprisals.size > 0 else 0.0
            hybrid_bits_per_byte = _bits_per_byte_from_surprisals(hybrid_surprisals, exact_target_tokens, base_bytes)
            copy_only_entropy_bits = float(np.mean(copy_surprisals)) if copy_surprisals.size > 0 else 0.0
            row = {
                "available": True,
                "context_order": int(order),
                "window": int(window),
                "active_target_positions": int(exact_target_tokens.size),
                "hybrid_oracle_floor_entropy_bits": hybrid_entropy_bits,
                "hybrid_oracle_floor_bits_per_byte": hybrid_bits_per_byte,
                "incremental_entropy_reduction_vs_exact_bits": float(
                    max(float(exact_row["conditional_entropy_bits"]) - hybrid_entropy_bits, 0.0)
                ),
                "incremental_entropy_reduction_vs_recent_copy_bits": float(
                    max(copy_only_entropy_bits - hybrid_entropy_bits, 0.0)
                ),
                "collapse_fraction_relative_to_unigram": _collapse_fraction(unigram_entropy, hybrid_entropy_bits),
                "remaining_entropy_fraction_relative_to_unigram": (
                    float(hybrid_entropy_bits / unigram_entropy) if unigram_entropy > 0.0 else 0.0
                ),
            }
            hybrid_rows.append(row)

    exact_available_rows = [row for row in exact_rows if row.get("available")]
    recent_copy_available_rows = [row for row in recent_copy_rows if row.get("available")]
    hybrid_available_rows = [row for row in hybrid_rows if row.get("available")]
    best_exact = min(exact_available_rows, key=lambda row: float(row["conditional_entropy_bits"]), default=None)
    best_recent_copy = min(
        recent_copy_available_rows,
        key=lambda row: float(row["oracle_floor_entropy_bits"]),
        default=None,
    )
    best_hybrid = min(
        hybrid_available_rows,
        key=lambda row: float(row["hybrid_oracle_floor_entropy_bits"]),
        default=None,
    )
    best_hybrid_bpb = min(
        [row for row in hybrid_available_rows if row.get("hybrid_oracle_floor_bits_per_byte") is not None],
        key=lambda row: float(row["hybrid_oracle_floor_bits_per_byte"]),
        default=None,
    )

    running_best_exact = float("inf")
    exact_curve: list[dict[str, object]] = []
    for row in exact_available_rows:
        running_best_exact = min(running_best_exact, float(row["conditional_entropy_bits"]))
        exact_curve.append(
            {
                "max_context_order": int(row["order"]),
                "remaining_entropy_bits": float(running_best_exact),
                "collapse_fraction_relative_to_unigram": _collapse_fraction(unigram_entropy, float(running_best_exact)),
            }
        )

    running_best_recent_copy = float("inf")
    recent_copy_curve: list[dict[str, object]] = []
    for row in recent_copy_available_rows:
        running_best_recent_copy = min(running_best_recent_copy, float(row["oracle_floor_entropy_bits"]))
        recent_copy_curve.append(
            {
                "max_window": int(row["window"]),
                "remaining_entropy_bits": float(running_best_recent_copy),
                "collapse_fraction_relative_to_unigram": _collapse_fraction(unigram_entropy, float(running_best_recent_copy)),
            }
        )

    hybrid_curve_by_order: list[dict[str, object]] = []
    for order in available_exact_orders:
        eligible = [row for row in hybrid_available_rows if int(row["context_order"]) <= order]
        if not eligible:
            continue
        best = min(eligible, key=lambda row: float(row["hybrid_oracle_floor_entropy_bits"]))
        hybrid_curve_by_order.append(
            {
                "max_context_order": int(order),
                "best_window": int(best["window"]),
                "remaining_entropy_bits": float(best["hybrid_oracle_floor_entropy_bits"]),
                "collapse_fraction_relative_to_unigram": _collapse_fraction(
                    unigram_entropy, float(best["hybrid_oracle_floor_entropy_bits"])
                ),
            }
        )

    hybrid_curve_by_window: list[dict[str, object]] = []
    for window in requested_windows:
        eligible = [row for row in hybrid_available_rows if int(row["window"]) <= window]
        if not eligible:
            continue
        best = min(eligible, key=lambda row: float(row["hybrid_oracle_floor_entropy_bits"]))
        hybrid_curve_by_window.append(
            {
                "max_window": int(window),
                "best_context_order": int(best["context_order"]),
                "remaining_entropy_bits": float(best["hybrid_oracle_floor_entropy_bits"]),
                "collapse_fraction_relative_to_unigram": _collapse_fraction(
                    unigram_entropy, float(best["hybrid_oracle_floor_entropy_bits"])
                ),
            }
        )

    best_remaining_entropy_bits = min(
        [float(unigram_entropy)]
        + ([float(best_exact["conditional_entropy_bits"])] if best_exact is not None else [])
        + ([float(best_recent_copy["oracle_floor_entropy_bits"])] if best_recent_copy is not None else [])
        + ([float(best_hybrid["hybrid_oracle_floor_entropy_bits"])] if best_hybrid is not None else [])
    )
    best_collapsible_fraction_toward_zero = _collapse_fraction(unigram_entropy, best_remaining_entropy_bits)
    gap_to_zero_explained_by_exact_lexical = (
        _collapse_fraction(unigram_entropy, float(best_exact["conditional_entropy_bits"])) if best_exact is not None else None
    )
    gap_to_zero_explained_by_recent_copy = (
        _collapse_fraction(unigram_entropy, float(best_recent_copy["oracle_floor_entropy_bits"]))
        if best_recent_copy is not None
        else None
    )
    gap_to_zero_explained_by_hybrid_oracle = (
        _collapse_fraction(unigram_entropy, float(best_hybrid["hybrid_oracle_floor_entropy_bits"]))
        if best_hybrid is not None
        else None
    )

    return {
        "available": bool(exact_available_rows or recent_copy_available_rows),
        "purpose": (
            "Estimate how much validation BPB is structurally collapsible toward zero under perfect stream-structure "
            "capture, separating exact lexical and recent-copy ceilings from budget-constrained model design."
        ),
        "unigram_entropy_bits": float(unigram_entropy),
        "best_collapsible_fraction_toward_zero": float(best_collapsible_fraction_toward_zero),
        "best_remaining_entropy_bits": float(best_remaining_entropy_bits),
        "gap_to_zero_explained_by_exact_lexical": gap_to_zero_explained_by_exact_lexical,
        "gap_to_zero_explained_by_recent_copy": gap_to_zero_explained_by_recent_copy,
        "gap_to_zero_explained_by_hybrid_oracle": gap_to_zero_explained_by_hybrid_oracle,
        "max_exact_context_order_without_int64_overflow": int(max_exact_order),
        "requested_context_orders": requested_orders,
        "requested_recent_copy_windows": requested_windows,
        "exact_lexical_by_order": exact_rows,
        "recent_copy_oracle_by_window": recent_copy_rows,
        "hybrid_oracle_floor": {
            "rows": hybrid_rows,
            "best_by_entropy": best_hybrid,
            "best_by_bits_per_byte": best_hybrid_bpb,
        },
        "best_exact_lexical_floor": best_exact,
        "best_recent_copy_floor": best_recent_copy,
        "asymptotic_remaining_entropy_curve": {
            "exact_lexical_by_max_order": exact_curve,
            "recent_copy_by_max_window": recent_copy_curve,
            "hybrid_by_max_order": hybrid_curve_by_order,
            "hybrid_by_max_window": hybrid_curve_by_window,
        },
        "realizable_caveat_notes": [
            "These are stream-structure lower bounds, not model-achieved expectations.",
            "The gap-to-zero fields report the fraction of unigram entropy that would be removed if the corresponding signal were captured perfectly.",
            "The exact lexical rows use empirical next-token distributions for each observed context order, so they assume perfect memorization of observed continuation structure.",
            "The recent-copy oracle rows assume zero surprise whenever the next token already appears inside the chosen recent window; this is stronger than a practical copy mechanism.",
            "The hybrid oracle floor takes the better per-token loss of exact lexical and recent-copy signals, which is intentionally optimistic and not directly realizable by a normalized model.",
        ],
    }


def lexical_entropy_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    base_bytes: np.ndarray | None = None,
    context_orders: list[int] | None = None,
) -> dict[str, object]:
    if tokens.size < 2:
        return {
            "available": False,
            "reason": "not_enough_tokens",
        }
    unigram_entropy = entropy_from_counts(counts)
    requested_orders = sorted({int(v) for v in (context_orders or [1, 2, 4]) if int(v) > 0})
    if 1 not in requested_orders:
        requested_orders.insert(0, 1)
    context_profiles = [_context_entropy_stats(tokens, vocab_size, order, unigram_entropy, base_bytes=base_bytes) for order in requested_orders]
    available_profiles = [item for item in context_profiles if item.get("available")]
    if not available_profiles:
        return {
            "available": False,
            "reason": "no_transitions",
        }
    for idx, item in enumerate(available_profiles):
        prev = available_profiles[idx - 1] if idx > 0 else None
        if prev is None:
            item["incremental_entropy_reduction_bits"] = float(unigram_entropy - float(item["conditional_entropy_bits"]))
            item["incremental_top4_coverage_gain"] = float(item["top4_next_token_coverage"])
            if item.get("conditional_bits_per_byte") is not None:
                item["incremental_bits_per_byte_reduction"] = float(unigram_entropy / max(float(item.get("mean_target_bytes", 0.0)), 1e-12)) - float(item["conditional_bits_per_byte"])
            else:
                item["incremental_bits_per_byte_reduction"] = None
        else:
            item["incremental_entropy_reduction_bits"] = float(prev["conditional_entropy_bits"]) - float(item["conditional_entropy_bits"])
            item["incremental_top4_coverage_gain"] = float(item["top4_next_token_coverage"]) - float(prev["top4_next_token_coverage"])
            if item.get("conditional_bits_per_byte") is not None and prev.get("conditional_bits_per_byte") is not None:
                item["incremental_bits_per_byte_reduction"] = float(prev["conditional_bits_per_byte"]) - float(item["conditional_bits_per_byte"])
            else:
                item["incremental_bits_per_byte_reduction"] = None

    order1 = next((item for item in available_profiles if int(item["order"]) == 1), available_profiles[0])
    best = min(available_profiles, key=lambda item: float(item["conditional_entropy_bits"]))
    best_bpb = min(
        [item for item in available_profiles if item.get("conditional_bits_per_byte") is not None],
        key=lambda item: float(item["conditional_bits_per_byte"]),
        default=None,
    )
    return {
        "available": True,
        "active_source_tokens": int(order1["active_contexts"]),
        "conditional_entropy_bits": float(order1["conditional_entropy_bits"]),
        "conditional_bits_per_byte": order1.get("conditional_bits_per_byte"),
        "byte_weighted_conditional_entropy_bits": order1.get("byte_weighted_conditional_entropy_bits"),
        "deterministic_source_rate": float(order1["deterministic_context_rate"]),
        "dominant_next_ge_50_rate": float(order1["dominant_next_ge_50_rate"]),
        "effective_branching_factor": float(order1["effective_branching_factor"]),
        "mean_target_bytes": order1.get("mean_target_bytes"),
        "mutual_information_bits": float(order1["mutual_information_bits"]),
        "normalized_mi": float(order1["normalized_mi"]),
        "repeat_next_token_rate": float(np.mean(tokens[:-1] == tokens[1:])) if tokens.size >= 2 else 0.0,
        "top1_next_token_coverage": float(order1["top1_next_token_coverage"]),
        "top4_next_token_coverage": float(order1["top4_next_token_coverage"]),
        "top8_next_token_coverage": float(order1["top8_next_token_coverage"]),
        "top1_next_byte_coverage": order1.get("top1_next_byte_coverage"),
        "top4_next_byte_coverage": order1.get("top4_next_byte_coverage"),
        "top8_next_byte_coverage": order1.get("top8_next_byte_coverage"),
        "total_transition_pairs": int(order1["total_transition_pairs"]),
        "context_orders": available_profiles,
        "best_context_order": int(best["order"]),
        "best_conditional_entropy_bits": float(best["conditional_entropy_bits"]),
        "best_total_mutual_information_bits": float(best["mutual_information_bits"]),
        "best_bits_per_byte_context_order": None if best_bpb is None else int(best_bpb["order"]),
        "best_conditional_bits_per_byte": None if best_bpb is None else float(best_bpb["conditional_bits_per_byte"]),
    }


def hashed_lexical_collision_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    lexical_profile: dict[str, object] | None = None,
    base_bytes: np.ndarray | None = None,
    orders: list[int] | None = None,
    bucket_sizes: list[int] | None = None,
) -> dict[str, object]:
    if tokens.size < 3:
        return {
            "available": False,
            "reason": "not_enough_tokens",
        }
    requested_orders = sorted({int(v) for v in (orders or [2, 3, 4]) if int(v) > 1})
    requested_buckets = sorted({int(v) for v in (bucket_sizes or [1024, 2048, 4096, 8192, 16384]) if int(v) > 0})
    if not requested_orders or not requested_buckets:
        return {
            "available": False,
            "reason": "no_requested_orders_or_buckets",
        }
    unigram_entropy = entropy_from_counts(counts)
    exact_profiles: dict[int, dict[str, object]] = {}
    if lexical_profile is not None and lexical_profile.get("available"):
        for item in lexical_profile.get("context_orders", []):
            if isinstance(item, dict) and item.get("available"):
                exact_profiles[int(item.get("order", 0))] = item
    if 1 not in exact_profiles:
        exact_profiles[1] = _context_entropy_stats(tokens, vocab_size, 1, unigram_entropy, base_bytes=base_bytes)
    order1 = exact_profiles.get(1)
    if not isinstance(order1, dict) or not order1.get("available"):
        return {
            "available": False,
            "reason": "missing_order1_profile",
        }

    per_order: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    order1_entropy = float(order1["conditional_entropy_bits"])
    order1_bpb = order1.get("conditional_bits_per_byte")

    for order in requested_orders:
        exact_profile = exact_profiles.get(order)
        if exact_profile is None or not exact_profile.get("available"):
            exact_profile = _context_entropy_stats(tokens, vocab_size, order, unigram_entropy, base_bytes=base_bytes)
            exact_profiles[order] = exact_profile
        transitions = _build_context_transitions(tokens, vocab_size, order)
        if transitions is None or not exact_profile.get("available"):
            per_order.append(
                {
                    "available": False,
                    "order": int(order),
                    "reason": "not_enough_tokens",
                }
            )
            continue

        context_ids, next_tokens = transitions
        unique_contexts = int(np.unique(context_ids).size)
        exact_entropy = float(exact_profile["conditional_entropy_bits"])
        exact_bpb = exact_profile.get("conditional_bits_per_byte")
        exact_incremental_gain = max(order1_entropy - exact_entropy, 0.0)
        exact_incremental_bpb_gain = (
            max(float(order1_bpb) - float(exact_bpb), 0.0)
            if order1_bpb is not None and exact_bpb is not None
            else None
        )
        sweep_rows: list[dict[str, object]] = []

        for bucket_count in requested_buckets:
            bucket_ids = _hash_context_ids(context_ids, bucket_count)
            bucket_stats = _conditional_stats_from_context_ids(bucket_ids, next_tokens, unigram_entropy, base_bytes=base_bytes)
            occupied_buckets = int(np.unique(bucket_ids).size)
            hashed_entropy = float(bucket_stats["conditional_entropy_bits"])
            hashed_bpb = bucket_stats.get("conditional_bits_per_byte")
            exact_gain = max(unigram_entropy - exact_entropy, 0.0)
            hashed_gain = max(unigram_entropy - hashed_entropy, 0.0)
            realized_incremental_gain = max(order1_entropy - hashed_entropy, 0.0)
            realized_incremental_bpb_gain = (
                max(float(order1_bpb) - float(hashed_bpb), 0.0)
                if order1_bpb is not None and hashed_bpb is not None
                else None
            )
            row = {
                **bucket_stats,
                "available": True,
                "order": int(order),
                "bucket_count": int(bucket_count),
                "unique_exact_contexts": unique_contexts,
                "occupied_buckets": occupied_buckets,
                "hash_load_factor": float(unique_contexts / max(bucket_count, 1)),
                "mean_exact_contexts_per_bucket": float(unique_contexts / max(occupied_buckets, 1)),
                "collision_context_fraction": float(1.0 - (occupied_buckets / max(unique_contexts, 1))),
                "exact_conditional_entropy_bits": exact_entropy,
                "exact_conditional_bits_per_byte": exact_bpb,
                "one_token_conditional_entropy_bits": order1_entropy,
                "one_token_conditional_bits_per_byte": order1_bpb,
                "entropy_regret_vs_exact_bits": float(max(hashed_entropy - exact_entropy, 0.0)),
                "bits_per_byte_regret_vs_exact": (
                    float(max(float(hashed_bpb) - float(exact_bpb), 0.0))
                    if hashed_bpb is not None and exact_bpb is not None
                    else None
                ),
                "retained_exact_entropy_gain_fraction": float(hashed_gain / max(exact_gain, 1e-12)) if exact_gain > 0.0 else None,
                "incremental_entropy_reduction_vs_order1_bits": float(realized_incremental_gain),
                "retained_incremental_entropy_gain_fraction": (
                    float(realized_incremental_gain / max(exact_incremental_gain, 1e-12)) if exact_incremental_gain > 0.0 else None
                ),
                "incremental_bits_per_byte_reduction_vs_order1": realized_incremental_bpb_gain,
                "retained_incremental_bits_per_byte_gain_fraction": (
                    float(realized_incremental_bpb_gain / max(float(exact_incremental_bpb_gain), 1e-12))
                    if realized_incremental_bpb_gain is not None and exact_incremental_bpb_gain is not None and exact_incremental_bpb_gain > 0.0
                    else None
                ),
            }
            sweep_rows.append(row)
            all_rows.append(row)

        best_entropy = min(sweep_rows, key=lambda item: float(item["conditional_entropy_bits"]))
        best_bpb = min(
            [item for item in sweep_rows if item.get("conditional_bits_per_byte") is not None],
            key=lambda item: float(item["conditional_bits_per_byte"]),
            default=None,
        )
        best_incremental = max(
            [item for item in sweep_rows if item.get("retained_incremental_entropy_gain_fraction") is not None],
            key=lambda item: float(item["retained_incremental_entropy_gain_fraction"]),
            default=None,
        )
        per_order.append(
            {
                "available": True,
                "order": int(order),
                "unique_exact_contexts": unique_contexts,
                "exact_conditional_entropy_bits": exact_entropy,
                "exact_conditional_bits_per_byte": exact_bpb,
                "one_token_conditional_entropy_bits": order1_entropy,
                "one_token_conditional_bits_per_byte": order1_bpb,
                "bucket_sweeps": sweep_rows,
                "best_bucket_by_entropy": {
                    "bucket_count": int(best_entropy["bucket_count"]),
                    "conditional_entropy_bits": float(best_entropy["conditional_entropy_bits"]),
                    "entropy_regret_vs_exact_bits": float(best_entropy["entropy_regret_vs_exact_bits"]),
                    "retained_incremental_entropy_gain_fraction": best_entropy.get("retained_incremental_entropy_gain_fraction"),
                },
                "best_bucket_by_bits_per_byte": None
                if best_bpb is None
                else {
                    "bucket_count": int(best_bpb["bucket_count"]),
                    "conditional_bits_per_byte": float(best_bpb["conditional_bits_per_byte"]),
                    "bits_per_byte_regret_vs_exact": best_bpb.get("bits_per_byte_regret_vs_exact"),
                    "retained_incremental_bits_per_byte_gain_fraction": best_bpb.get(
                        "retained_incremental_bits_per_byte_gain_fraction"
                    ),
                },
                "best_bucket_by_incremental_retention": None
                if best_incremental is None
                else {
                    "bucket_count": int(best_incremental["bucket_count"]),
                    "retained_incremental_entropy_gain_fraction": float(best_incremental["retained_incremental_entropy_gain_fraction"]),
                    "incremental_entropy_reduction_vs_order1_bits": float(best_incremental["incremental_entropy_reduction_vs_order1_bits"]),
                },
            }
        )

    available_rows = [item for item in all_rows if item.get("available")]
    if not available_rows:
        return {
            "available": False,
            "reason": "no_hashed_profiles",
        }
    best_overall_entropy = min(available_rows, key=lambda item: float(item["conditional_entropy_bits"]))
    best_overall_bpb = min(
        [item for item in available_rows if item.get("conditional_bits_per_byte") is not None],
        key=lambda item: float(item["conditional_bits_per_byte"]),
        default=None,
    )
    return {
        "available": True,
        "orders": per_order,
        "best_overall_by_entropy": {
            "order": int(best_overall_entropy["order"]),
            "bucket_count": int(best_overall_entropy["bucket_count"]),
            "conditional_entropy_bits": float(best_overall_entropy["conditional_entropy_bits"]),
            "entropy_regret_vs_exact_bits": float(best_overall_entropy["entropy_regret_vs_exact_bits"]),
            "retained_incremental_entropy_gain_fraction": best_overall_entropy.get("retained_incremental_entropy_gain_fraction"),
        },
        "best_overall_by_bits_per_byte": None
        if best_overall_bpb is None
        else {
            "order": int(best_overall_bpb["order"]),
            "bucket_count": int(best_overall_bpb["bucket_count"]),
            "conditional_bits_per_byte": float(best_overall_bpb["conditional_bits_per_byte"]),
            "bits_per_byte_regret_vs_exact": best_overall_bpb.get("bits_per_byte_regret_vs_exact"),
            "retained_incremental_bits_per_byte_gain_fraction": best_overall_bpb.get(
                "retained_incremental_bits_per_byte_gain_fraction"
            ),
        },
    }


def higher_order_lexical_retention_profile(
    hashed_lexical_profile: dict[str, object] | None,
    target_order: int = 3,
    collision_cap: float = 0.65,
    retention_floor: float = 0.60,
) -> dict[str, object]:
    if not isinstance(hashed_lexical_profile, dict) or not hashed_lexical_profile.get("available"):
        return {"available": False, "reason": "missing_hashed_lexical_profile"}

    def _to_float(value: object, default: float = 0.0) -> float:
        return default if value is None else float(value)

    def _row_score(row: dict[str, object]) -> float:
        retention = _to_float(row.get("retained_incremental_entropy_gain_fraction"))
        retention_bpb = _to_float(row.get("retained_incremental_bits_per_byte_gain_fraction"))
        collision = min(max(_to_float(row.get("collision_context_fraction"), 1.0), 0.0), 1.0)
        one_token_entropy = _to_float(row.get("one_token_conditional_entropy_bits"))
        exact_entropy = _to_float(row.get("exact_conditional_entropy_bits"))
        regret = max(_to_float(row.get("entropy_regret_vs_exact_bits")), 0.0)
        exact_incremental = max(one_token_entropy - exact_entropy, 1e-12)
        regret_ratio = min(max(regret / exact_incremental, 0.0), 1.0)
        return float(
            0.50 * retention
            + 0.15 * retention_bpb
            + 0.20 * (1.0 - collision)
            + 0.15 * (1.0 - regret_ratio)
        )

    def _compact_row(row: dict[str, object], score: float) -> dict[str, object]:
        return {
            "bucket_count": int(row["bucket_count"]),
            "conditional_entropy_bits": float(row["conditional_entropy_bits"]),
            "conditional_bits_per_byte": None
            if row.get("conditional_bits_per_byte") is None
            else float(row["conditional_bits_per_byte"]),
            "collision_context_fraction": float(row["collision_context_fraction"]),
            "hash_load_factor": float(row["hash_load_factor"]),
            "retained_incremental_entropy_gain_fraction": None
            if row.get("retained_incremental_entropy_gain_fraction") is None
            else float(row["retained_incremental_entropy_gain_fraction"]),
            "retained_incremental_bits_per_byte_gain_fraction": None
            if row.get("retained_incremental_bits_per_byte_gain_fraction") is None
            else float(row["retained_incremental_bits_per_byte_gain_fraction"]),
            "entropy_regret_vs_exact_bits": float(row["entropy_regret_vs_exact_bits"]),
            "balanced_retention_score": float(score),
        }

    per_order: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    target_summary: dict[str, object] | None = None

    for order_entry in hashed_lexical_profile.get("orders", []):
        if not isinstance(order_entry, dict) or not order_entry.get("available"):
            continue
        bucket_rows = [
            row
            for row in order_entry.get("bucket_sweeps", [])
            if isinstance(row, dict) and row.get("available")
        ]
        if not bucket_rows:
            continue
        scored_rows = [(row, _row_score(row)) for row in bucket_rows]
        best_balanced_row, best_balanced_score = max(scored_rows, key=lambda item: float(item[1]))
        safe_rows = [
            (row, score)
            for row, score in scored_rows
            if float(row.get("collision_context_fraction", 1.0)) <= collision_cap
        ]
        best_safe = max(
            safe_rows,
            key=lambda item: (
                _to_float(item[0].get("retained_incremental_entropy_gain_fraction")),
                float(item[1]),
                -float(item[0].get("conditional_entropy_bits", float("inf"))),
            ),
            default=None,
        )
        retention_rows = [
            (row, score)
            for row, score in scored_rows
            if _to_float(row.get("retained_incremental_entropy_gain_fraction")) >= retention_floor
        ]
        safe_retention_rows = [
            (row, score)
            for row, score in retention_rows
            if float(row.get("collision_context_fraction", 1.0)) <= collision_cap
        ]
        min_bucket_for_retention = min(retention_rows, key=lambda item: int(item[0]["bucket_count"]), default=None)
        min_bucket_for_safe_retention = min(
            safe_retention_rows, key=lambda item: int(item[0]["bucket_count"]), default=None
        )
        summary = {
            "order": int(order_entry["order"]),
            "unique_exact_contexts": int(order_entry.get("unique_exact_contexts", 0)),
            "exact_conditional_entropy_bits": float(order_entry.get("exact_conditional_entropy_bits", 0.0)),
            "exact_conditional_bits_per_byte": None
            if order_entry.get("exact_conditional_bits_per_byte") is None
            else float(order_entry["exact_conditional_bits_per_byte"]),
            "exact_incremental_entropy_reduction_vs_order1_bits": float(
                max(
                    _to_float(order_entry.get("one_token_conditional_entropy_bits"))
                    - _to_float(order_entry.get("exact_conditional_entropy_bits")),
                    0.0,
                )
            ),
            "best_balanced_bucket": _compact_row(best_balanced_row, best_balanced_score),
            "best_under_collision_cap": None
            if best_safe is None
            else _compact_row(best_safe[0], float(best_safe[1])),
            "smallest_bucket_meeting_retention_floor": None
            if min_bucket_for_retention is None
            else _compact_row(min_bucket_for_retention[0], float(min_bucket_for_retention[1])),
            "smallest_bucket_meeting_retention_and_collision_caps": None
            if min_bucket_for_safe_retention is None
            else _compact_row(min_bucket_for_safe_retention[0], float(min_bucket_for_safe_retention[1])),
            "frontier": [_compact_row(row, score) for row, score in scored_rows],
        }
        per_order.append(summary)
        comparison_rows.append(
            {
                "order": int(summary["order"]),
                "exact_conditional_entropy_bits": float(summary["exact_conditional_entropy_bits"]),
                "exact_incremental_entropy_reduction_vs_order1_bits": float(
                    summary["exact_incremental_entropy_reduction_vs_order1_bits"]
                ),
                "best_balanced_bucket_count": int(summary["best_balanced_bucket"]["bucket_count"]),
                "best_balanced_conditional_entropy_bits": float(
                    summary["best_balanced_bucket"]["conditional_entropy_bits"]
                ),
                "best_balanced_retention_fraction": summary["best_balanced_bucket"][
                    "retained_incremental_entropy_gain_fraction"
                ],
                "best_balanced_collision_fraction": float(
                    summary["best_balanced_bucket"]["collision_context_fraction"]
                ),
                "best_balanced_score": float(summary["best_balanced_bucket"]["balanced_retention_score"]),
                "best_under_collision_cap_bucket_count": None
                if summary["best_under_collision_cap"] is None
                else int(summary["best_under_collision_cap"]["bucket_count"]),
                "best_under_collision_cap_entropy_bits": None
                if summary["best_under_collision_cap"] is None
                else float(summary["best_under_collision_cap"]["conditional_entropy_bits"]),
                "best_under_collision_cap_retention_fraction": None
                if summary["best_under_collision_cap"] is None
                else summary["best_under_collision_cap"]["retained_incremental_entropy_gain_fraction"],
            }
        )
        if int(summary["order"]) == int(target_order):
            target_summary = summary

    if not per_order:
        return {"available": False, "reason": "no_order_summaries"}

    preferred_safe = max(
        [row for row in comparison_rows if row.get("best_under_collision_cap_bucket_count") is not None],
        key=lambda row: (
            _to_float(row.get("best_under_collision_cap_retention_fraction")),
            _to_float(row.get("best_balanced_score")),
        ),
        default=None,
    )
    order4_summary = next((row for row in per_order if int(row["order"]) == 4), None)
    order4_upgrade_case = None
    if target_summary is not None and order4_summary is not None and int(target_order) != 4:
        target_safe = target_summary.get("best_under_collision_cap")
        order4_safe = order4_summary.get("best_under_collision_cap")
        target_entropy = (
            float(target_safe["conditional_entropy_bits"])
            if isinstance(target_safe, dict)
            else float(target_summary["best_balanced_bucket"]["conditional_entropy_bits"])
        )
        order4_entropy = (
            float(order4_safe["conditional_entropy_bits"])
            if isinstance(order4_safe, dict)
            else float(order4_summary["best_balanced_bucket"]["conditional_entropy_bits"])
        )
        target_retention = (
            _to_float(target_safe.get("retained_incremental_entropy_gain_fraction"))
            if isinstance(target_safe, dict)
            else _to_float(target_summary["best_balanced_bucket"].get("retained_incremental_entropy_gain_fraction"))
        )
        order4_retention = (
            _to_float(order4_safe.get("retained_incremental_entropy_gain_fraction"))
            if isinstance(order4_safe, dict)
            else _to_float(order4_summary["best_balanced_bucket"].get("retained_incremental_entropy_gain_fraction"))
        )
        order4_upgrade_case = {
            "target_order": int(target_order),
            "challenger_order": 4,
            "exact_entropy_gain_vs_target_bits": float(
                max(float(target_summary["exact_conditional_entropy_bits"]) - float(order4_summary["exact_conditional_entropy_bits"]), 0.0)
            ),
            "hashed_entropy_gain_vs_target_bits": float(max(target_entropy - order4_entropy, 0.0)),
            "hashed_retention_delta_vs_target": float(order4_retention - target_retention),
            "order4_beats_target_under_collision_budget": bool(
                order4_entropy + 0.05 < target_entropy and order4_retention >= target_retention - 0.05
            ),
        }

    return {
        "available": True,
        "target_order": int(target_order),
        "collision_cap": float(collision_cap),
        "retention_floor": float(retention_floor),
        "preferred_order_under_collision_budget": None
        if preferred_safe is None
        else {
            "order": int(preferred_safe["order"]),
            "bucket_count": int(preferred_safe["best_under_collision_cap_bucket_count"]),
            "retention_fraction": preferred_safe["best_under_collision_cap_retention_fraction"],
            "entropy_bits": preferred_safe["best_under_collision_cap_entropy_bits"],
        },
        "target_order_summary": target_summary,
        "order_comparison": comparison_rows,
        "order4_upgrade_case": order4_upgrade_case,
        "orders": per_order,
    }


def _prediction_profile_from_context_ids(
    context_ids: np.ndarray,
    next_tokens: np.ndarray,
    base_bytes: np.ndarray | None = None,
    n_bins: int = 15,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if context_ids.size <= 0 or next_tokens.size <= 0:
        return {"available": False, "reason": "no_transitions"}, {}
    sort_idx = np.lexsort((next_tokens, context_ids))
    ctx_sorted = context_ids[sort_idx]
    next_sorted = next_tokens[sort_idx]
    pair_change = np.ones((ctx_sorted.size,), dtype=bool)
    pair_change[1:] = (ctx_sorted[1:] != ctx_sorted[:-1]) | (next_sorted[1:] != next_sorted[:-1])
    pair_starts = np.flatnonzero(pair_change)
    pair_ends = np.append(pair_starts[1:], ctx_sorted.size)
    pair_ctx = ctx_sorted[pair_starts]
    pair_counts = (pair_ends - pair_starts).astype(np.int64, copy=False)
    pair_next = next_sorted[pair_starts]

    ctx_change = np.ones((pair_ctx.size,), dtype=bool)
    ctx_change[1:] = pair_ctx[1:] != pair_ctx[:-1]
    ctx_starts = np.flatnonzero(ctx_change)
    ctx_ends = np.append(ctx_starts[1:], pair_ctx.size)

    total_pairs = int(pair_counts.sum())
    byte_table = np.maximum(base_bytes.astype(np.float64, copy=False), 0.0) if base_bytes is not None else None
    target_byte_total = 0.0
    top1_byte_mass = 0.0
    weighted_entropy_sum = 0.0
    weighted_conf_sum = 0.0
    weighted_margin_sum = 0.0
    weighted_accuracy_sum = 0.0
    bin_count = np.zeros((max(int(n_bins), 1),), dtype=np.float64)
    bin_conf_sum = np.zeros_like(bin_count)
    bin_acc_sum = np.zeros_like(bin_count)
    confidence_samples: list[float] = []
    margin_samples: list[float] = []
    accuracy_samples: list[float] = []
    entropy_samples: list[float] = []
    weight_samples: list[float] = []

    for start, end in zip(ctx_starts, ctx_ends, strict=True):
        local_counts = pair_counts[start:end]
        local_next_tokens = pair_next[start:end]
        total = int(local_counts.sum())
        if total <= 0:
            continue
        probs = local_counts.astype(np.float64) / total
        surprisals = -np.log2(np.clip(probs, 1e-12, None))
        row_entropy = float((probs * surprisals).sum())
        order_idx = np.argsort(local_counts)[::-1]
        sorted_counts = local_counts[order_idx]
        sorted_tokens = local_next_tokens[order_idx]
        top1_prob = float(sorted_counts[0] / total)
        top1_acc = top1_prob
        top2_prob = float(sorted_counts[1] / total) if sorted_counts.size > 1 else 0.0
        margin_bits = float(np.log2(max(top1_prob, 1e-12) / max(top2_prob, 1e-12)))
        weighted_entropy_sum += row_entropy * total
        weighted_conf_sum += top1_prob * total
        weighted_margin_sum += margin_bits * total
        weighted_accuracy_sum += top1_acc * total
        if byte_table is not None:
            local_bytes = byte_table[local_next_tokens]
            byte_mass = local_counts.astype(np.float64) * local_bytes
            target_byte_total += float(byte_mass.sum())
            top1_byte_mass += float(byte_mass[order_idx][:1].sum())
        bin_idx = min(int(top1_prob * bin_count.size), bin_count.size - 1)
        bin_count[bin_idx] += total
        bin_conf_sum[bin_idx] += top1_prob * total
        bin_acc_sum[bin_idx] += top1_acc * total
        confidence_samples.append(top1_prob)
        margin_samples.append(margin_bits)
        accuracy_samples.append(top1_acc)
        entropy_samples.append(row_entropy)
        weight_samples.append(float(total))

    if total_pairs <= 0:
        return {"available": False, "reason": "empty_after_grouping"}, {}
    ece = 0.0
    confidence_bins: list[dict[str, float]] = []
    for idx in range(bin_count.size):
        if bin_count[idx] <= 0:
            continue
        frac = float(bin_count[idx] / total_pairs)
        mean_conf = float(bin_conf_sum[idx] / bin_count[idx])
        mean_acc = float(bin_acc_sum[idx] / bin_count[idx])
        lower = float(idx / bin_count.size)
        upper = float((idx + 1) / bin_count.size)
        ece += frac * abs(mean_conf - mean_acc)
        confidence_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "mass": frac,
                "mean_confidence": mean_conf,
                "mean_accuracy": mean_acc,
            }
        )
    summary = {
        "available": True,
        "active_contexts": int(len(weight_samples)),
        "total_transition_pairs": total_pairs,
        "conditional_entropy_bits": float(weighted_entropy_sum / max(total_pairs, 1)),
        "top1_accuracy": float(weighted_accuracy_sum / max(total_pairs, 1)),
        "top1_next_byte_coverage": float(top1_byte_mass / max(target_byte_total, 1e-12)) if target_byte_total > 0.0 else None,
        "mean_confidence": float(weighted_conf_sum / max(total_pairs, 1)),
        "mean_margin_bits": float(weighted_margin_sum / max(total_pairs, 1)),
        "mean_row_entropy_bits": float(weighted_entropy_sum / max(total_pairs, 1)),
        "expected_calibration_error": float(ece),
        "confidence_bins": confidence_bins,
    }
    samples = {
        "confidence": np.asarray(confidence_samples, dtype=np.float64),
        "margin_bits": np.asarray(margin_samples, dtype=np.float64),
        "accuracy": np.asarray(accuracy_samples, dtype=np.float64),
        "entropy_bits": np.asarray(entropy_samples, dtype=np.float64),
        "weight": np.asarray(weight_samples, dtype=np.float64),
    }
    return summary, samples


def proxy_calibration_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    base_bytes: np.ndarray | None = None,
    exact_orders: list[int] | None = None,
    hashed_orders: list[int] | None = None,
    hashed_buckets: list[int] | None = None,
    n_bins: int = 15,
) -> dict[str, object]:
    if tokens.size < 2:
        return {"available": False, "reason": "not_enough_tokens"}
    exact_rows: list[dict[str, object]] = []
    hashed_rows: list[dict[str, object]] = []
    for order in sorted({int(v) for v in (exact_orders or [1, 2, 4]) if int(v) > 0}):
        transitions = _build_context_transitions(tokens, vocab_size, order)
        if transitions is None:
            continue
        context_ids, next_tokens = transitions
        summary, _ = _prediction_profile_from_context_ids(context_ids, next_tokens, base_bytes=base_bytes, n_bins=n_bins)
        if summary.get("available"):
            exact_rows.append({"source": "exact", "order": int(order), **summary})
    for order in sorted({int(v) for v in (hashed_orders or [2]) if int(v) > 1}):
        transitions = _build_context_transitions(tokens, vocab_size, order)
        if transitions is None:
            continue
        context_ids, next_tokens = transitions
        for bucket_count in sorted({int(v) for v in (hashed_buckets or [8192, 16384, 65536]) if int(v) > 0}):
            bucket_ids = _hash_context_ids(context_ids, bucket_count)
            summary, _ = _prediction_profile_from_context_ids(bucket_ids, next_tokens, base_bytes=base_bytes, n_bins=n_bins)
            if summary.get("available"):
                hashed_rows.append(
                    {
                        "source": "hashed",
                        "order": int(order),
                        "bucket_count": int(bucket_count),
                        **summary,
                    }
                )
    all_rows = exact_rows + hashed_rows
    if not all_rows:
        return {"available": False, "reason": "no_profiles"}
    best_ece = min(all_rows, key=lambda item: float(item["expected_calibration_error"]))
    best_accuracy = max(all_rows, key=lambda item: float(item["top1_accuracy"]))
    return {
        "available": True,
        "exact_profiles": exact_rows,
        "hashed_profiles": hashed_rows,
        "best_calibrated": {
            "source": str(best_ece["source"]),
            "order": int(best_ece["order"]),
            "bucket_count": None if "bucket_count" not in best_ece else int(best_ece["bucket_count"]),
            "expected_calibration_error": float(best_ece["expected_calibration_error"]),
            "top1_accuracy": float(best_ece["top1_accuracy"]),
        },
        "best_accuracy": {
            "source": str(best_accuracy["source"]),
            "order": int(best_accuracy["order"]),
            "bucket_count": None if "bucket_count" not in best_accuracy else int(best_accuracy["bucket_count"]),
            "expected_calibration_error": float(best_accuracy["expected_calibration_error"]),
            "top1_accuracy": float(best_accuracy["top1_accuracy"]),
        },
    }


def lexical_control_profile(
    tokens: np.ndarray,
    vocab_size: int,
    base_bytes: np.ndarray | None = None,
    orders: list[int] | None = None,
    bucket_sizes: list[int] | None = None,
    shortcut_scales: list[float] | None = None,
    smear_gate_inits: list[float] | None = None,
    route_scales: list[float] | None = None,
    n_bins: int = 15,
) -> dict[str, object]:
    if tokens.size < 3:
        return {"available": False, "reason": "not_enough_tokens"}
    rows: list[dict[str, object]] = []
    requested_orders = sorted({int(v) for v in (orders or [2]) if int(v) > 0})
    requested_buckets = sorted({int(v) for v in (bucket_sizes or [8192, 16384, 65536]) if int(v) > 0})
    shortcut_grid = [float(v) for v in (shortcut_scales or [0.0, 0.01, 0.02, 0.05])]
    smear_grid = [float(v) for v in (smear_gate_inits or [-2.0, -1.0, -0.5])]
    route_grid = [float(v) for v in (route_scales or [0.02, 0.05, 0.08])]

    for order in requested_orders:
        transitions = _build_context_transitions(tokens, vocab_size, order)
        if transitions is None:
            continue
        context_ids, next_tokens = transitions
        for bucket_count in requested_buckets:
            source_ids = _hash_context_ids(context_ids, bucket_count)
            summary, samples = _prediction_profile_from_context_ids(source_ids, next_tokens, base_bytes=base_bytes, n_bins=n_bins)
            if not summary.get("available"):
                continue
            weights = samples.get("weight")
            margins = samples.get("margin_bits")
            accuracies = samples.get("accuracy")
            if weights is None or margins is None or accuracies is None or weights.size <= 0:
                continue
            total_weight = float(np.sum(weights))
            for shortcut_scale in shortcut_grid:
                confidence = 1.0 / (1.0 + np.exp(-(float(shortcut_scale) * margins)))
                mean_conf = float(np.sum(confidence * weights) / max(total_weight, 1e-12))
                cold_start_deviation = abs(mean_conf - 0.5)
                trusted = confidence >= 0.60
                trusted_mass = float(np.sum(weights[trusted]) / max(total_weight, 1e-12))
                trusted_acc = float(np.sum(accuracies[trusted] * weights[trusted]) / max(np.sum(weights[trusted]), 1e-12)) if np.any(trusted) else None
                confidence_weighted_accuracy = float(np.sum(confidence * accuracies * weights) / max(total_weight, 1e-12))
                centered_confidence = 2.0 * confidence - 1.0
                for route_scale in route_grid:
                    route_signal = (1.0 - 2.0 * confidence) * float(route_scale)
                    route_abs_mean = float(np.sum(np.abs(route_signal) * weights) / max(total_weight, 1e-12))
                    route_signed_mean = float(np.sum(route_signal * weights) / max(total_weight, 1e-12))
                    for smear_gate_init in smear_grid:
                        initial_smear_gate = float(1.0 / (1.0 + math.exp(-float(smear_gate_init))))
                        conditioned_smear_gate = float(
                            np.sum((1.0 / (1.0 + np.exp(-(float(smear_gate_init) + centered_confidence)))) * weights)
                            / max(total_weight, 1e-12)
                        )
                        stability_adjusted_shortcut_score = (
                            confidence_weighted_accuracy
                            - cold_start_deviation
                            - 0.25 * route_abs_mean
                            - 0.10 * abs(conditioned_smear_gate - initial_smear_gate)
                        )
                        rows.append(
                            {
                                "source_order": int(order),
                                "bucket_count": int(bucket_count),
                                "source_top1_accuracy": float(summary["top1_accuracy"]),
                                "source_ece": float(summary["expected_calibration_error"]),
                                "shortcut_scale_init": float(shortcut_scale),
                                "smear_gate_init": float(smear_gate_init),
                                "lexical_route_scale_init": float(route_scale),
                                "mean_confidence": mean_conf,
                                "cold_start_confidence_deviation": cold_start_deviation,
                                "trusted_shortcut_mass": trusted_mass,
                                "trusted_shortcut_accuracy": trusted_acc,
                                "confidence_weighted_accuracy": confidence_weighted_accuracy,
                                "route_bias_abs_mean": route_abs_mean,
                                "route_bias_signed_mean": route_signed_mean,
                                "initial_smear_gate": initial_smear_gate,
                                "conditioned_smear_gate_mean": conditioned_smear_gate,
                                "stability_adjusted_shortcut_score": stability_adjusted_shortcut_score,
                            }
                        )
    if not rows:
        return {"available": False, "reason": "no_candidate_rows"}
    best_stable = max(rows, key=lambda item: float(item["stability_adjusted_shortcut_score"]))
    best_trusted = max(rows, key=lambda item: float(item["trusted_shortcut_mass"]) * float(item.get("trusted_shortcut_accuracy") or 0.0))
    return {
        "available": True,
        "candidate_rows": rows,
        "best_stability_adjusted": best_stable,
        "best_trusted_shortcut": best_trusted,
    }


def recent_copy_window_profile(
    tokens: np.ndarray,
    vocab_size: int,
    counts: np.ndarray,
    lexical_profile: dict[str, object] | None = None,
    base_bytes: np.ndarray | None = None,
    windows: list[int] | None = None,
    context_orders: list[int] | None = None,
) -> dict[str, object]:
    if tokens.size < 3:
        return {"available": False, "reason": "not_enough_tokens"}
    unigram_entropy = entropy_from_counts(counts)
    order1 = None
    if lexical_profile is not None and lexical_profile.get("available"):
        order1 = next(
            (item for item in lexical_profile.get("context_orders", []) if isinstance(item, dict) and int(item.get("order", 0)) == 1),
            None,
        )
    if order1 is None:
        order1 = _context_entropy_stats(tokens, vocab_size, 1, unigram_entropy, base_bytes=base_bytes)
    if not isinstance(order1, dict) or not order1.get("available"):
        return {"available": False, "reason": "missing_order1_profile"}
    order1_entropy = float(order1["conditional_entropy_bits"])
    order1_bpb = order1.get("conditional_bits_per_byte")
    byte_table = np.maximum(base_bytes.astype(np.float64, copy=False), 0.0) if base_bytes is not None else None
    requested_windows = sorted({int(v) for v in (windows or [128, 256, 512]) if int(v) > 0})
    requested_orders = sorted({int(v) for v in (context_orders or [1, 2, 4]) if int(v) > 0})
    rows: list[dict[str, object]] = []

    for order in requested_orders:
        transitions = _build_context_transitions(tokens, vocab_size, order)
        if transitions is None:
            continue
        context_ids, next_tokens = transitions
        total_positions = int(next_tokens.size)
        all_target_bytes = float(byte_table[next_tokens].sum()) if byte_table is not None else 0.0
        for window in requested_windows:
            history: dict[int, deque[tuple[int, int]]] = {}
            covered = 0
            top1_correct = 0
            top4_correct = 0
            nll_sum = 0.0
            byte_nll_sum = 0.0
            covered_byte_total = 0.0
            top1_byte_mass = 0.0
            match_counts: list[int] = []
            for idx, (ctx, target) in enumerate(zip(context_ids, next_tokens, strict=True)):
                pos = order + idx
                key = int(ctx)
                bucket = history.get(key)
                if bucket is None:
                    bucket = deque()
                    history[key] = bucket
                while bucket and (pos - bucket[0][0]) > window:
                    bucket.popleft()
                if bucket:
                    counts_local = Counter(int(token_id) for _, token_id in bucket)
                    total = int(sum(counts_local.values()))
                    if total > 0:
                        covered += 1
                        match_counts.append(total)
                        sorted_items = counts_local.most_common(8)
                        prob = float(counts_local.get(int(target), 0) / total)
                        surprisal = float(-math.log2(max(prob, 1e-12)))
                        nll_sum += surprisal
                        if int(sorted_items[0][0]) == int(target):
                            top1_correct += 1
                        if any(int(token_id) == int(target) for token_id, _ in sorted_items[:4]):
                            top4_correct += 1
                        if byte_table is not None:
                            token_bytes = float(byte_table[int(target)])
                            covered_byte_total += token_bytes
                            byte_nll_sum += token_bytes * surprisal
                            if int(sorted_items[0][0]) == int(target):
                                top1_byte_mass += token_bytes
                bucket.append((pos, int(target)))
            coverage_rate = float(covered / max(total_positions, 1))
            covered_entropy = float(nll_sum / max(covered, 1)) if covered > 0 else None
            combined_entropy = float(coverage_rate * (covered_entropy if covered_entropy is not None else order1_entropy) + (1.0 - coverage_rate) * order1_entropy)
            combined_bpb = None
            if byte_table is not None and order1_bpb is not None and all_target_bytes > 0.0:
                uncovered_bytes = max(all_target_bytes - covered_byte_total, 0.0)
                combined_bpb = float((byte_nll_sum + uncovered_bytes * float(order1_bpb)) / all_target_bytes)
            row = {
                "available": True,
                "context_order": int(order),
                "window": int(window),
                "coverage_rate": coverage_rate,
                "uncovered_rate": float(1.0 - coverage_rate),
                "mean_matches_per_covered_position": float(np.mean(match_counts)) if match_counts else 0.0,
                "median_matches_per_covered_position": float(np.median(match_counts)) if match_counts else 0.0,
                "covered_cross_entropy_bits": covered_entropy,
                "combined_cross_entropy_bits": combined_entropy,
                "incremental_entropy_reduction_vs_order1_bits": float(max(order1_entropy - combined_entropy, 0.0)),
                "top1_accuracy_on_covered": float(top1_correct / max(covered, 1)) if covered > 0 else None,
                "top4_accuracy_on_covered": float(top4_correct / max(covered, 1)) if covered > 0 else None,
                "top1_byte_coverage_on_covered": float(top1_byte_mass / max(covered_byte_total, 1e-12)) if covered_byte_total > 0.0 else None,
                "combined_bits_per_byte": combined_bpb,
                "incremental_bits_per_byte_reduction_vs_order1": (
                    float(max(float(order1_bpb) - float(combined_bpb), 0.0))
                    if order1_bpb is not None and combined_bpb is not None
                    else None
                ),
            }
            rows.append(row)
    if not rows:
        return {"available": False, "reason": "no_windows"}
    best_entropy = min(rows, key=lambda item: float(item["combined_cross_entropy_bits"]))
    best_bpb = min(
        [item for item in rows if item.get("combined_bits_per_byte") is not None],
        key=lambda item: float(item["combined_bits_per_byte"]),
        default=None,
    )
    return {
        "available": True,
        "rows": rows,
        "best_by_entropy": best_entropy,
        "best_by_bits_per_byte": best_bpb,
    }


def early_budget_coverage_profile(
    tokens: np.ndarray,
    vocab_size: int,
    budget_token_points: list[int] | None = None,
    orders: list[int] | None = None,
    bucket_sizes: list[int] | None = None,
    top_k_buckets: int = 128,
) -> dict[str, object]:
    if tokens.size < 3:
        return {"available": False, "reason": "not_enough_tokens"}
    requested_budgets = sorted(
        {max(int(v), 0) for v in (budget_token_points or [1_000_000, 4_000_000, 16_000_000, 64_000_000]) if int(v) > 0}
    )
    requested_orders = sorted({int(v) for v in (orders or [2]) if int(v) > 0})
    requested_buckets = sorted({int(v) for v in (bucket_sizes or [8192, 16384, 65536]) if int(v) > 0})
    effective_budgets = sorted({min(int(tokens.size), int(v)) for v in requested_budgets if min(int(tokens.size), int(v)) > 0})
    rows: list[dict[str, object]] = []

    for budget_tokens in effective_budgets:
        clipped_from = max(v for v in requested_budgets if min(int(tokens.size), int(v)) == budget_tokens)
        budget_prefix = np.ascontiguousarray(tokens[:budget_tokens])
        for order in requested_orders:
            transitions = _build_context_transitions(budget_prefix, vocab_size, order)
            if transitions is None:
                continue
            context_ids, next_tokens = transitions
            lookup = _top1_lookup_from_context_ids(context_ids, next_tokens)
            if lookup is None:
                continue
            exact_counts = lookup["total_count"]
            total_pairs = int(exact_counts.sum())
            if total_pairs <= 0:
                continue
            unique_exact = int(exact_counts.size)
            seen_once = exact_counts == 1
            repeated = exact_counts >= 2
            dominant = lookup["top1_prob"] >= 0.5
            for bucket_count in requested_buckets:
                bucket_ids = _hash_context_ids(context_ids, bucket_count)
                bucket_lookup = _top1_lookup_from_context_ids(bucket_ids, next_tokens)
                if bucket_lookup is None:
                    continue
                bucket_counts = bucket_lookup["total_count"]
                occupied = int(bucket_counts.size)
                sorted_bucket_counts = np.sort(bucket_counts)[::-1]
                bucket_entropy = float(entropy_from_counts(bucket_counts.astype(np.int64, copy=False)))
                rows.append(
                    {
                        "budget_tokens": int(budget_tokens),
                        "requested_budget_tokens": int(clipped_from),
                        "budget_clipped_to_profiled_tokens": bool(clipped_from > budget_tokens),
                        "order": int(order),
                        "bucket_count": int(bucket_count),
                        "total_transition_pairs": total_pairs,
                        "unique_exact_contexts": unique_exact,
                        "exact_seen_once_rate": float(np.mean(seen_once)) if unique_exact > 0 else 0.0,
                        "exact_repeated_context_rate": float(np.mean(repeated)) if unique_exact > 0 else 0.0,
                        "repeated_exact_transition_mass": float(exact_counts[repeated].sum() / max(total_pairs, 1)),
                        "dominant_exact_context_rate": float(np.mean(dominant)) if unique_exact > 0 else 0.0,
                        "occupied_bucket_fraction": float(occupied / max(bucket_count, 1)),
                        "mean_updates_per_occupied_bucket": float(total_pairs / max(occupied, 1)),
                        "top_bucket_mass_fraction": float(sorted_bucket_counts[: max(int(top_k_buckets), 1)].sum() / max(total_pairs, 1)),
                        "bucket_update_gini": float(gini_from_counts(bucket_counts.astype(np.int64, copy=False))),
                        "bucket_update_entropy_bits": bucket_entropy,
                        "effective_bucket_fraction": float((2.0**bucket_entropy) / max(bucket_count, 1)),
                    }
                )
    if not rows:
        return {"available": False, "reason": "no_budget_rows"}
    best = max(rows, key=lambda item: float(item["repeated_exact_transition_mass"]) * float(item["occupied_bucket_fraction"]))
    return {
        "available": True,
        "profiled_tokens": int(tokens.size),
        "rows": rows,
        "best_budget_fit": best,
    }


def train_val_transfer_profile(
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    vocab_size: int,
    base_bytes: np.ndarray | None = None,
    exact_orders: list[int] | None = None,
    hashed_orders: list[int] | None = None,
    hashed_buckets: list[int] | None = None,
) -> dict[str, object]:
    if train_tokens.size < 3 or val_tokens.size < 3:
        return {"available": False, "reason": "not_enough_tokens"}
    exact_rows: list[dict[str, object]] = []
    hashed_rows: list[dict[str, object]] = []
    byte_table = np.maximum(base_bytes.astype(np.float64, copy=False), 0.0) if base_bytes is not None else None

    def _transfer_row(
        train_ids: np.ndarray,
        train_top1_token: np.ndarray,
        train_top1_prob: np.ndarray,
        val_ids: np.ndarray,
        val_next: np.ndarray,
    ) -> dict[str, object]:
        pos = np.searchsorted(train_ids, val_ids)
        safe_pos = np.clip(pos, 0, max(train_ids.size - 1, 0))
        seen = (pos < train_ids.size) & (train_ids[safe_pos] == val_ids)
        token_bytes = byte_table[val_next] if byte_table is not None else None
        seen_rate = float(np.mean(seen)) if seen.size > 0 else 0.0
        unseen_rate = float(1.0 - seen_rate)
        row = {
            "val_token_seen_rate": seen_rate,
            "val_token_seen_byte_rate": _byte_weighted_rate(seen, token_bytes),
            "val_token_unseen_rate": unseen_rate,
            "seen_top1_accuracy": None,
            "seen_top1_byte_coverage": None,
            "seen_mean_confidence": None,
            "seen_calibration_gap": None,
        }
        if not np.any(seen):
            return row
        seen_pos = pos[seen]
        pred = train_top1_token[seen_pos]
        conf = train_top1_prob[seen_pos]
        correct = pred == val_next[seen]
        seen_bytes = token_bytes[seen] if token_bytes is not None else None
        seen_acc = float(np.mean(correct))
        mean_conf = float(np.mean(conf))
        row["seen_top1_accuracy"] = seen_acc
        row["seen_top1_byte_coverage"] = _byte_weighted_accuracy(correct, seen_bytes)
        row["seen_mean_confidence"] = mean_conf
        row["seen_calibration_gap"] = float(mean_conf - seen_acc)
        return row

    for order in sorted({int(v) for v in (exact_orders or [1, 2, 4]) if int(v) > 0}):
        train_transitions = _build_context_transitions(train_tokens, vocab_size, order)
        val_transitions = _build_context_transitions(val_tokens, vocab_size, order)
        if train_transitions is None or val_transitions is None:
            continue
        train_ctx, train_next = train_transitions
        val_ctx, val_next = val_transitions
        lookup = _top1_lookup_from_context_ids(train_ctx, train_next)
        if lookup is None:
            continue
        row = _transfer_row(lookup["context_ids"], lookup["top1_token"], lookup["top1_prob"], val_ctx, val_next)
        exact_rows.append(
            {
                "source": "exact",
                "order": int(order),
                "train_unique_contexts": int(lookup["context_ids"].size),
                **row,
                "transfer_score": float(row["val_token_seen_rate"]) * float(row.get("seen_top1_accuracy") or 0.0),
            }
        )
    for order in sorted({int(v) for v in (hashed_orders or [2]) if int(v) > 1}):
        train_transitions = _build_context_transitions(train_tokens, vocab_size, order)
        val_transitions = _build_context_transitions(val_tokens, vocab_size, order)
        if train_transitions is None or val_transitions is None:
            continue
        train_ctx, train_next = train_transitions
        val_ctx, val_next = val_transitions
        for bucket_count in sorted({int(v) for v in (hashed_buckets or [8192, 16384, 65536]) if int(v) > 0}):
            lookup = _top1_lookup_from_context_ids(_hash_context_ids(train_ctx, bucket_count), train_next)
            if lookup is None:
                continue
            row = _transfer_row(lookup["context_ids"], lookup["top1_token"], lookup["top1_prob"], _hash_context_ids(val_ctx, bucket_count), val_next)
            hashed_rows.append(
                {
                    "source": "hashed",
                    "order": int(order),
                    "bucket_count": int(bucket_count),
                    "train_active_buckets": int(lookup["context_ids"].size),
                    **row,
                    "transfer_score": float(row["val_token_seen_rate"]) * float(row.get("seen_top1_accuracy") or 0.0),
                }
            )
    all_rows = exact_rows + hashed_rows
    if not all_rows:
        return {"available": False, "reason": "no_transfer_rows"}
    best = max(all_rows, key=lambda item: float(item.get("transfer_score", 0.0)))
    return {
        "available": True,
        "exact_rows": exact_rows,
        "hashed_rows": hashed_rows,
        "best_transfer": best,
    }


def confidence_route_budget_profile(
    tokens: np.ndarray,
    vocab_size: int,
    base_bytes: np.ndarray | None = None,
    exact_orders: list[int] | None = None,
    hashed_orders: list[int] | None = None,
    hashed_buckets: list[int] | None = None,
    confidence_thresholds: list[float] | None = None,
    n_bins: int = 15,
) -> dict[str, object]:
    if tokens.size < 3:
        return {"available": False, "reason": "not_enough_tokens"}
    thresholds = sorted({float(v) for v in (confidence_thresholds or [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]) if 0.0 < float(v) < 1.0})
    rows: list[dict[str, object]] = []

    def _append_rows(source: str, order: int, bucket_count: int | None, context_ids: np.ndarray, next_tokens: np.ndarray) -> None:
        summary, samples = _prediction_profile_from_context_ids(context_ids, next_tokens, base_bytes=base_bytes, n_bins=n_bins)
        if not summary.get("available"):
            return
        confidence = samples.get("confidence")
        accuracy = samples.get("accuracy")
        weight = samples.get("weight")
        if confidence is None or accuracy is None or weight is None or weight.size <= 0:
            return
        total_weight = float(np.sum(weight))
        route_prob = np.clip(1.0 - confidence, 1e-9, 1.0 - 1e-9)
        binary_entropy = -(route_prob * np.log2(route_prob) + (1.0 - route_prob) * np.log2(1.0 - route_prob))
        mean_route_entropy = float(np.sum(binary_entropy * weight) / max(total_weight, 1e-12))
        for threshold in thresholds:
            trust_mask = confidence >= float(threshold)
            trust_mass = float(np.sum(weight[trust_mask]) / max(total_weight, 1e-12))
            late_mass = float(1.0 - trust_mass)
            trusted_acc = (
                float(np.sum(accuracy[trust_mask] * weight[trust_mask]) / max(np.sum(weight[trust_mask]), 1e-12))
                if np.any(trust_mask)
                else None
            )
            false_trust = (
                float(np.sum((1.0 - accuracy[trust_mask]) * weight[trust_mask]) / max(np.sum(weight[trust_mask]), 1e-12))
                if np.any(trust_mask)
                else None
            )
            trusted_conf = (
                float(np.sum(confidence[trust_mask] * weight[trust_mask]) / max(np.sum(weight[trust_mask]), 1e-12))
                if np.any(trust_mask)
                else None
            )
            rows.append(
                {
                    "source": source,
                    "order": int(order),
                    "bucket_count": bucket_count,
                    "confidence_threshold": float(threshold),
                    "mean_route_entropy_bits": mean_route_entropy,
                    "shortcut_trust_mass": trust_mass,
                    "late_compute_mass": late_mass,
                    "trusted_shortcut_accuracy": trusted_acc,
                    "trusted_shortcut_confidence": trusted_conf,
                    "false_trust_rate": false_trust,
                    "oracle_mixed_top1_upper_bound": float(trust_mass * float(trusted_acc or 0.0) + late_mass),
                }
            )

    for order in sorted({int(v) for v in (exact_orders or [1, 2]) if int(v) > 0}):
        transitions = _build_context_transitions(tokens, vocab_size, order)
        if transitions is None:
            continue
        context_ids, next_tokens = transitions
        _append_rows("exact", order, None, context_ids, next_tokens)
    for order in sorted({int(v) for v in (hashed_orders or [2]) if int(v) > 1}):
        transitions = _build_context_transitions(tokens, vocab_size, order)
        if transitions is None:
            continue
        context_ids, next_tokens = transitions
        for bucket_count in sorted({int(v) for v in (hashed_buckets or [8192, 16384, 65536]) if int(v) > 0}):
            _append_rows("hashed", order, int(bucket_count), _hash_context_ids(context_ids, bucket_count), next_tokens)
    if not rows:
        return {"available": False, "reason": "no_threshold_rows"}
    best_upper = max(rows, key=lambda item: float(item["oracle_mixed_top1_upper_bound"]))
    safe_candidates = [item for item in rows if item.get("false_trust_rate") is not None and float(item["false_trust_rate"]) <= 0.10]
    best_savings = max(safe_candidates, key=lambda item: float(item["shortcut_trust_mass"]), default=None)
    return {
        "available": True,
        "rows": rows,
        "best_oracle_upper_bound": best_upper,
        "best_compute_saver_under_false_trust_10pct": best_savings,
    }


def route_calibration_selectivity_profile(
    confidence_route_budget: dict[str, object] | None,
    proxy_calibration: dict[str, object] | None = None,
    target_hashed_order: int = 3,
    false_trust_cap: float = 0.10,
) -> dict[str, object]:
    if not isinstance(confidence_route_budget, dict) or not confidence_route_budget.get("available"):
        return {"available": False, "reason": "missing_confidence_route_budget"}

    def _to_float(value: object, default: float = 0.0) -> float:
        return default if value is None else float(value)

    def _row_summary(row: dict[str, object], score: float) -> dict[str, object]:
        return {
            "source": str(row["source"]),
            "order": int(row["order"]),
            "bucket_count": None if row.get("bucket_count") is None else int(row["bucket_count"]),
            "confidence_threshold": float(row["confidence_threshold"]),
            "shortcut_trust_mass": float(row["shortcut_trust_mass"]),
            "late_compute_mass": float(row["late_compute_mass"]),
            "mean_route_entropy_bits": float(row["mean_route_entropy_bits"]),
            "trusted_shortcut_accuracy": None
            if row.get("trusted_shortcut_accuracy") is None
            else float(row["trusted_shortcut_accuracy"]),
            "trusted_shortcut_confidence": None
            if row.get("trusted_shortcut_confidence") is None
            else float(row["trusted_shortcut_confidence"]),
            "confidence_accuracy_gap": None
            if row.get("trusted_shortcut_accuracy") is None or row.get("trusted_shortcut_confidence") is None
            else float(abs(float(row["trusted_shortcut_confidence"]) - float(row["trusted_shortcut_accuracy"]))),
            "false_trust_rate": None if row.get("false_trust_rate") is None else float(row["false_trust_rate"]),
            "oracle_mixed_top1_upper_bound": float(row["oracle_mixed_top1_upper_bound"]),
            "selective_routing_score": float(score),
        }

    def _row_score(row: dict[str, object]) -> float:
        trust_mass = min(max(_to_float(row.get("shortcut_trust_mass")), 0.0), 1.0)
        false_trust = min(max(_to_float(row.get("false_trust_rate"), 1.0), 0.0), 1.0)
        route_entropy = min(max(_to_float(row.get("mean_route_entropy_bits"), 1.0), 0.0), 1.0)
        conf = row.get("trusted_shortcut_confidence")
        acc = row.get("trusted_shortcut_accuracy")
        calibration_gap = 1.0
        if conf is not None and acc is not None:
            calibration_gap = min(abs(float(conf) - float(acc)) / 0.20, 1.0)
        return float(
            0.40 * trust_mass
            + 0.25 * (1.0 - false_trust)
            + 0.20 * (1.0 - route_entropy)
            + 0.15 * (1.0 - calibration_gap)
        )

    rows = [row for row in confidence_route_budget.get("rows", []) if isinstance(row, dict)]
    if not rows:
        return {"available": False, "reason": "no_route_rows"}

    scored_rows = [(row, _row_score(row)) for row in rows]
    safe_rows = [
        (row, score)
        for row, score in scored_rows
        if row.get("false_trust_rate") is not None and float(row["false_trust_rate"]) <= false_trust_cap
    ]
    meaningful_safe_rows = [
        (row, score)
        for row, score in safe_rows
        if float(row.get("shortcut_trust_mass", 0.0)) >= 0.05
    ]
    best_safe = max(safe_rows, key=lambda item: float(item[1]), default=None)
    lowest_entropy_safe = min(
        meaningful_safe_rows,
        key=lambda item: (
            float(item[0].get("mean_route_entropy_bits", float("inf"))),
            -float(item[0].get("shortcut_trust_mass", 0.0)),
        ),
        default=None,
    )

    grouped_rows: dict[tuple[str, int, int | None], list[tuple[dict[str, object], float]]] = {}
    for row, score in scored_rows:
        key = (str(row["source"]), int(row["order"]), None if row.get("bucket_count") is None else int(row["bucket_count"]))
        grouped_rows.setdefault(key, []).append((row, score))
    grouped_summaries = []
    for (source, order, bucket_count), items in grouped_rows.items():
        safe_items = [
            (row, score)
            for row, score in items
            if row.get("false_trust_rate") is not None and float(row["false_trust_rate"]) <= false_trust_cap
        ]
        best_item = max(safe_items if safe_items else items, key=lambda item: float(item[1]))
        grouped_summaries.append(_row_summary(best_item[0], float(best_item[1])))
    grouped_summaries.sort(
        key=lambda row: (
            float(row["selective_routing_score"]),
            -float(row["shortcut_trust_mass"]),
        ),
        reverse=True,
    )

    target_hashed = next(
        (
            row
            for row in grouped_summaries
            if row["source"] == "hashed"
            and int(row["order"]) == int(target_hashed_order)
        ),
        None,
    )
    proxy_best = None
    if isinstance(proxy_calibration, dict) and proxy_calibration.get("available"):
        best_calibrated = proxy_calibration.get("best_calibrated")
        if isinstance(best_calibrated, dict):
            proxy_best = {
                "source": str(best_calibrated["source"]),
                "order": int(best_calibrated["order"]),
                "bucket_count": None
                if best_calibrated.get("bucket_count") is None
                else int(best_calibrated["bucket_count"]),
                "expected_calibration_error": float(best_calibrated["expected_calibration_error"]),
                "top1_accuracy": float(best_calibrated["top1_accuracy"]),
            }

    return {
        "available": True,
        "false_trust_cap": float(false_trust_cap),
        "target_hashed_order": int(target_hashed_order),
        "proxy_best_calibrated_reference": proxy_best,
        "best_safe_selective_row": None if best_safe is None else _row_summary(best_safe[0], float(best_safe[1])),
        "lowest_entropy_meaningful_safe_row": None
        if lowest_entropy_safe is None
        else _row_summary(lowest_entropy_safe[0], float(lowest_entropy_safe[1])),
        "target_hashed_order_best_row": target_hashed,
        "source_order_frontier": grouped_summaries,
    }


def parse_run_log(path: str) -> RunLogSummary:
    def _metrics_candidates(log_path: Path) -> list[Path]:
        candidates: list[Path] = []
        if log_path.name == "train.log":
            candidates.append(log_path.with_name("metrics.jsonl"))
        candidates.append(log_path.parent / "metrics.jsonl")
        candidates.append(log_path.parent / "runs" / log_path.stem / "metrics.jsonl")
        candidates.append(log_path.parent.parent / "runs" / log_path.stem / "metrics.jsonl")
        seen: set[Path] = set()
        out: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(candidate)
        return out

    def _parse_metrics_sidecar(log_path: Path) -> tuple[str | None, dict[str, object], dict[str, object], dict[str, object], str | None]:
        for candidate in _metrics_candidates(log_path):
            if not candidate.exists():
                continue
            run_id = None
            config: dict[str, object] = {}
            artifact: dict[str, object] = {}
            final_eval: dict[str, object] = {}
            try:
                for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if run_id is None and isinstance(payload.get("run_id"), str):
                        run_id = str(payload["run_id"])
                    event = str(payload.get("event", ""))
                    if event == "config":
                        config.update({key: value for key, value in payload.items() if key not in {"event", "timestamp_utc"}})
                    elif event == "artifact":
                        artifact.update({key: value for key, value in payload.items() if key not in {"event", "timestamp_utc"}})
                    elif event in {"final_eval", "final_eval_ttt"}:
                        final_eval.update({key: value for key, value in payload.items() if key not in {"event", "timestamp_utc"}})
                        final_eval["event"] = event
                    elif event == "ema_export":
                        config["ema_exported"] = True
            except OSError:
                continue
            return run_id, config, artifact, final_eval, str(candidate)
        return None, {}, {}, {}, None

    def _coerce_bool(text: str) -> bool | None:
        normalized = text.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return None

    def _maybe_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    def _infer_int_from_text(patterns: list[str], *texts: str | None) -> int | None:
        for text in texts:
            if not text:
                continue
            for pattern in patterns:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m:
                    for group in m.groups():
                        if group is not None:
                            try:
                                return int(group)
                            except ValueError:
                                continue
        return None

    log_path = Path(path)
    run_id, config, artifact_summary, final_eval_summary, metrics_path = _parse_metrics_sidecar(log_path)
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
    raw_final_val_bpb = None
    raw_final_val_loss = None
    roundtrip_delta_bpb = None
    roundtrip_exact_bpb = None
    roundtrip_eval_ms = None
    total_submission_bytes = _maybe_float(artifact_summary.get("total_submission_bytes"))
    quant_zlib_bytes = _maybe_float(artifact_summary.get("quant_zlib_bytes"))
    code_bytes = _maybe_float(artifact_summary.get("code_bytes"))
    raw_model_bytes = _maybe_float(artifact_summary.get("raw_model_bytes"))

    for line in text.splitlines():
        if run_id is None:
            m = re.search(r"run_id=([^\s]+)", line)
            if m:
                run_id = str(m.group(1))
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
        m = re.search(r"training_preset:([^\s]+)", line)
        if m and "training_preset" not in config:
            config["training_preset"] = str(m.group(1))
        m = re.search(
            r"policy_schema:recipe_family:([^\s]+)\s+curriculum_policy:([^\s]+)\s+data_policy:([^\s]+)\s+"
            r"optimizer_policy:([^\s]+)\s+precision_policy:([^\s]+)\s+eval_policy:([^\s]+)\s+runtime_policy:([^\s]+)",
            line,
        )
        if m:
            config.setdefault("recipe_family", str(m.group(1)))
            config.setdefault("curriculum_policy", str(m.group(2)))
            config.setdefault("data_policy", str(m.group(3)))
            config.setdefault("optimizer_policy", str(m.group(4)))
            config.setdefault("precision_policy", str(m.group(5)))
            config.setdefault("eval_policy", str(m.group(6)))
            config.setdefault("runtime_policy", str(m.group(7)))
        m = re.search(r"packed_batches:train:(True|False)\s+eval_windows:(True|False)", line)
        if m:
            packed_train = _coerce_bool(m.group(1))
            packed_eval = _coerce_bool(m.group(2))
            if packed_train is not None:
                config.setdefault("packed_train_batching", packed_train)
            if packed_eval is not None:
                config.setdefault("packed_eval_windows", packed_eval)
        m = re.search(r"regularization:.*ema_decay:([0-9.eE+-]+)", line)
        if m and "ema_decay" not in config:
            config["ema_decay"] = float(m.group(1))
        m = re.search(r"fake_quant:.*late_qat:(True|False)", line)
        if m and "late_qat" not in config:
            late_qat = _coerce_bool(m.group(1))
            if late_qat is not None:
                config["late_qat"] = late_qat
        m = re.search(r"context_ext:use_xpos:(True|False)", line)
        if m and "use_xpos" not in config:
            use_xpos = _coerce_bool(m.group(1))
            if use_xpos is not None:
                config["use_xpos"] = use_xpos
        m = re.search(r"attn_ext:.*retention_layers:(\d+)", line)
        if m and "retention_layers" not in config:
            config["retention_layers"] = int(m.group(1))
        m = re.search(
            r"token_enrich:use_smear_gate:(True|False)\s+bigram_vocab_size:(\d+)\s+bigram_dim:(\d+)\s+hashed_ngram_order:(\d+)",
            line,
        )
        if m:
            use_smear_gate = _coerce_bool(m.group(1))
            if use_smear_gate is not None:
                config.setdefault("use_smear_gate", use_smear_gate)
            config.setdefault("bigram_vocab_size", int(m.group(2)))
            config.setdefault("bigram_dim", int(m.group(3)))
            config.setdefault("hashed_ngram_order", int(m.group(4)))
        m = re.search(r"late_compute:xsa_last_n:(\d+)", line)
        if m and "xsa_last_n" not in config:
            config["xsa_last_n"] = int(m.group(1))
        m = re.search(
            r"init_export:.*export_quant_bits:(\d+)\s+export_mlp_bits:(\d+)\s+export_attn_bits:(\d+).*"
            r"export_calibration_method:([^\s]+)",
            line,
        )
        if m:
            config.setdefault("export_quant_bits", int(m.group(1)))
            config.setdefault("export_mlp_bits", int(m.group(2)))
            config.setdefault("export_attn_bits", int(m.group(3)))
            config.setdefault("export_calibration_method", str(m.group(4)))
        m = re.search(r"export_block_prune:enabled:(True|False)", line)
        if m and "export_block_prune" not in config:
            export_block_prune = _coerce_bool(m.group(1))
            if export_block_prune is not None:
                config["export_block_prune"] = export_block_prune
        m = re.search(r"Serialized model:\s+(\d+)\s+bytes", line)
        if m:
            raw_model_bytes = float(m.group(1))
        m = re.search(r"Code size:\s+(\d+)\s+bytes", line)
        if m:
            code_bytes = float(m.group(1))
        m = re.search(r"Serialized model q\d+\+\w+:\s+(\d+)\s+bytes", line)
        if m:
            quant_zlib_bytes = float(m.group(1))
        m = re.search(r"Total submission size q\d+\+\w+:\s+(\d+)\s+bytes", line)
        if m:
            total_submission_bytes = float(m.group(1))
        m = re.search(r"final_q\d+_\w+_roundtrip val_loss:[0-9.]+\s+val_bpb:([0-9.]+)\s+eval_time:(\d+)ms", line)
        if m:
            final_eval_summary.setdefault("val_bpb", float(m.group(1)))
            final_eval_summary.setdefault("eval_time_ms", float(m.group(2)))
            roundtrip_eval_ms = float(m.group(2))
        m = re.search(r"final_q\d+_\w+_roundtrip_exact val_loss:[0-9.]+\s+val_bpb:([0-9.]+)", line)
        if m:
            roundtrip_exact_bpb = float(m.group(1))
            final_eval_summary["val_bpb"] = float(m.group(1))
            final_eval_summary["exact"] = True
        m = re.search(r"final_q\d+_\w+_roundtrip_proxy val_loss:[0-9.]+\s+val_bpb:([0-9.]+)", line)
        if m and "val_bpb" not in final_eval_summary:
            final_eval_summary["val_bpb"] = float(m.group(1))
            final_eval_summary["exact"] = False
        m = re.search(
            r"final_roundtrip_gap raw_val_loss:([0-9.]+)\s+raw_val_bpb:([0-9.]+)\s+delta_loss:[+-0-9.eE]+\s+delta_bpb:([+-0-9.eE]+)",
            line,
        )
        if m:
            raw_final_val_loss = float(m.group(1))
            raw_final_val_bpb = float(m.group(2))
            roundtrip_delta_bpb = float(m.group(3))
        m = re.search(r"artifact_margin_bytes:(-?\d+)", line)
        if m:
            artifact_summary["artifact_margin_bytes"] = int(m.group(1))

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

    if train_seq_len is None and config.get("train_seq_len") is not None:
        train_seq_len = int(config["train_seq_len"])
    if eval_seq_len is None and config.get("eval_seq_len") is not None:
        eval_seq_len = int(config["eval_seq_len"])
    if config.get("mlp_hidden") is None:
        inferred = _infer_int_from_text([r"mlp[_-]?(\d+)", r"ffn[_-]?(\d+)"], run_id, str(log_path))
        if inferred is not None:
            config["mlp_hidden"] = inferred
    if config.get("num_layers") is None:
        inferred = _infer_int_from_text([r"(?:^|[_-])(\d{1,2})L(?:[_-]|$)", r"layers?[_-]?(\d+)"], run_id, str(log_path))
        if inferred is not None:
            config["num_layers"] = inferred
    if config.get("model_dim") is None:
        inferred = _infer_int_from_text([r"dim[_-]?(\d+)", r"d[_-]?(\d+)", r"width[_-]?(\d+)"], run_id, str(log_path))
        if inferred is not None:
            config["model_dim"] = inferred
    if train_batch_tokens is None and config.get("train_batch_tokens") is not None:
        train_batch_tokens = int(config["train_batch_tokens"])
    if model_params is None and config.get("model_params") is not None:
        model_params = int(config["model_params"])
    if raw_model_bytes is not None:
        artifact_summary.setdefault("raw_model_bytes", int(raw_model_bytes))
    if quant_zlib_bytes is not None:
        artifact_summary.setdefault("quant_zlib_bytes", int(quant_zlib_bytes))
    if code_bytes is not None:
        artifact_summary.setdefault("code_bytes", int(code_bytes))
    if total_submission_bytes is not None:
        artifact_summary.setdefault("total_submission_bytes", int(total_submission_bytes))
    if roundtrip_eval_ms is not None:
        final_eval_summary.setdefault("eval_time_ms", float(roundtrip_eval_ms))
    if raw_final_val_bpb is not None:
        final_eval_summary.setdefault("raw_val_bpb", float(raw_final_val_bpb))
    if raw_final_val_loss is not None:
        final_eval_summary.setdefault("raw_val_loss", float(raw_final_val_loss))
    if roundtrip_delta_bpb is not None:
        final_eval_summary.setdefault("roundtrip_delta_bpb", float(roundtrip_delta_bpb))
    if roundtrip_exact_bpb is not None:
        final_eval_summary.setdefault("roundtrip_exact_bpb", float(roundtrip_exact_bpb))

    return RunLogSummary(
        path=path,
        run_id=run_id,
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
        config=config,
        artifact_summary=artifact_summary,
        final_eval_summary=final_eval_summary,
        metrics_path=metrics_path,
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


def _round_to_multiple(value: int, multiple: int) -> int:
    multiple = max(int(multiple), 1)
    return max(multiple, int(round(value / multiple) * multiple))


def default_candidate_models_from_anchor(
    anchor_run: RunLogSummary | None,
    default_vocab_size: int,
) -> list[dict[str, int | str]]:
    if anchor_run is None:
        return []
    config = anchor_run.config or {}
    try:
        base_layers = int(config.get("num_layers") or 0)
        base_dim = int(config.get("model_dim") or 0)
        base_mlp = int(config.get("mlp_hidden") or 0)
        base_heads = int(config.get("num_heads") or 8)
        base_kv_heads = int(config.get("num_kv_heads") or 4)
        base_vocab = int(config.get("vocab_size") or default_vocab_size)
    except (TypeError, ValueError):
        return []
    if base_layers <= 0 or base_dim <= 0 or base_mlp <= 0:
        return []
    layer_options = sorted({max(base_layers - 2, 4), base_layers, base_layers + 2})
    mlp_options = sorted(
        {
            _round_to_multiple(int(base_mlp * 3 / 4), 64),
            _round_to_multiple(base_mlp, 64),
            _round_to_multiple(int(base_mlp * 5 / 4), 64),
        }
    )
    candidates: list[dict[str, int | str]] = []
    for num_layers in layer_options:
        for mlp_hidden in mlp_options:
            candidates.append(
                {
                    "name": f"auto_L{num_layers}_D{base_dim}_M{mlp_hidden}",
                    "num_layers": int(num_layers),
                    "model_dim": int(base_dim),
                    "mlp_hidden": int(mlp_hidden),
                    "num_heads": int(base_heads),
                    "num_kv_heads": int(base_kv_heads),
                    "vocab_size": int(base_vocab),
                }
            )
    return candidates


def competition_transformer_recommendation(
    anchor_run: RunLogSummary | None,
    model_frontier: list[dict[str, object]],
    observed_frontier: list[dict[str, object]],
    real_eval_frontier: list[dict[str, object]],
    training_budget: dict[str, object] | None,
    fallback_model_dim: int,
) -> dict[str, object]:
    train_choice = None
    if observed_frontier:
        eligible = [
            row
            for row in observed_frontier
            if float(row.get("fastest_step_avg_ms", float("inf"))) <= 300.0
        ]
        source = eligible if eligible else observed_frontier
        train_choice = min(
            source,
            key=lambda row: (
                float("inf") if row.get("best_val_bpb") is None else float(row["best_val_bpb"]),
                float(row.get("fastest_step_avg_ms", float("inf"))),
                -int(row.get("train_seq_len", 0)),
            ),
            default=None,
        )
    if train_choice is None and isinstance(training_budget, dict):
        train_choice = {
            "train_seq_len": int(training_budget.get("train_seq_len", 0) or 0),
            "train_batch_tokens": int(training_budget.get("train_batch_tokens", 0) or 0),
            "fastest_step_avg_ms": float(training_budget.get("avg_step_ms", 0.0) or 0.0),
            "best_val_bpb": None,
        }

    eval_choice = None
    if real_eval_frontier:
        eligible_eval = [
            row
            for row in real_eval_frontier
            if bool(row.get("fits_budget", False))
        ]
        source = eligible_eval if eligible_eval else real_eval_frontier
        eval_choice = max(
            source,
            key=lambda row: (
                float(row.get("context_reuse_rate", 0.0)),
                -float(row.get("relative_eval_cost", float("inf"))),
                int(row.get("seq_len", 0)),
            ),
            default=None,
        )

    model_choice = None
    if model_frontier:
        artifact_eligible = [
            row for row in model_frontier if float(row.get("artifact_limit_ratio", float("inf"))) <= 1.0
        ]
        budget_eligible = [
            row for row in artifact_eligible if int(row.get("estimated_steps_in_budget", 0)) >= 4000
        ]
        source = budget_eligible or artifact_eligible or model_frontier
        model_choice = min(
            source,
            key=lambda row: (
                float(row["projected_bpb_proxy"]) if row.get("projected_bpb_proxy") is not None else float("inf"),
                -int(row.get("estimated_steps_in_budget", 0)),
                int(row.get("artifact_bytes_q6_proxy", 0)),
            ),
            default=None,
        )
    if model_choice is None and anchor_run is not None:
        config = anchor_run.config or {}
        num_layers = int(config.get("num_layers", 0) or 0)
        model_dim = int(config.get("model_dim", 0) or fallback_model_dim)
        mlp_hidden = int(config.get("mlp_hidden", 0) or 0)
        model_choice = {
            "name": "anchor_run",
            "num_layers": num_layers if num_layers > 0 else None,
            "model_dim": model_dim if model_dim > 0 else None,
            "mlp_hidden": mlp_hidden if mlp_hidden > 0 else None,
            "param_count_estimate": anchor_run.model_params,
            "projected_bpb_proxy": anchor_run.best_val_bpb,
        }

    if train_choice is None and model_choice is None and eval_choice is None:
        return {"available": False, "reason": "no_model_or_runtime_evidence"}

    rationale: list[str] = []
    if isinstance(model_choice, dict):
        rationale.append(
            f"Model shape recommendation prioritizes projected BPB proxy and training steps under the 10-minute budget."
        )
    if isinstance(train_choice, dict):
        rationale.append(
            f"Train seq recommendation prefers the best observed BPB among throughput-viable seq lengths, falling back to the configured budget lane."
        )
    if isinstance(eval_choice, dict):
        rationale.append(
            f"Eval seq recommendation prefers the longest budget-fitting length with the strongest sampled context reuse."
        )
    return {
        "available": True,
        "recommended_model": model_choice,
        "recommended_train_seq": train_choice,
        "recommended_eval_seq": eval_choice,
        "constraints": {
            "max_wallclock_seconds": None if training_budget is None else float(training_budget.get("max_wallclock_seconds", 0.0) or 0.0),
            "competition_target": "10-minute training/eval regime on 8xH100-equivalent hardware",
        },
        "rationale": rationale,
    }


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


def _safe_numeric(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rankdata_average(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and arr[order[j]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman_rank_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    xr = _rankdata_average(xs)
    yr = _rankdata_average(ys)
    x_centered = xr - xr.mean()
    y_centered = yr - yr.mean()
    x_norm = float(np.linalg.norm(x_centered))
    y_norm = float(np.linalg.norm(y_centered))
    if x_norm <= 0.0 or y_norm <= 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / (x_norm * y_norm))


def _metric_preference(metric_name: str) -> str:
    return "higher" if metric_name in {"roundtrip_gain_per_100kb"} else "lower"


def _component_feature_rows(run: RunLogSummary) -> dict[str, object]:
    config = dict(run.config)
    train_seq_len = int(config.get("train_seq_len", run.train_seq_len or 0) or 0)
    eval_seq_len = int(config.get("eval_seq_len", run.eval_seq_len or 0) or 0)
    train_batch_tokens = int(config.get("train_batch_tokens", run.train_batch_tokens or 0) or 0)
    model_params = int(config.get("model_params", run.model_params or 0) or 0)
    best_val_bpb = run.best_val_bpb
    final_step_avg_ms = run.final_step_avg_ms
    artifact = dict(run.artifact_summary)
    final_eval = dict(run.final_eval_summary)
    roundtrip_val_bpb = _safe_numeric(final_eval.get("roundtrip_exact_bpb"))
    if roundtrip_val_bpb is None:
        roundtrip_val_bpb = _safe_numeric(final_eval.get("val_bpb"))
    final_eval_ms = _safe_numeric(final_eval.get("eval_time_ms"))
    submission_bytes = _safe_numeric(artifact.get("total_submission_bytes"))
    if submission_bytes is None and artifact.get("quant_zlib_bytes") is not None and artifact.get("code_bytes") is not None:
        submission_bytes = float(artifact["quant_zlib_bytes"]) + float(artifact["code_bytes"])
    bigram_vocab_size = int(config.get("bigram_vocab_size", 0) or 0)
    hashed_ngram_order = int(config.get("hashed_ngram_order", 1) or 1)
    retention_layers = int(config.get("retention_layers", 0) or 0)
    xsa_last_n = int(config.get("xsa_last_n", 0) or 0)
    export_quant_bits = int(config.get("export_quant_bits", 0) or 0)
    export_mlp_bits = int(config.get("export_mlp_bits", 0) or 0)
    export_attn_bits = int(config.get("export_attn_bits", 0) or 0)
    ema_decay = _safe_numeric(config.get("ema_decay"))
    row = {
        "path": run.path,
        "run_id": run.run_id or Path(run.path).stem,
        "training_preset": config.get("training_preset"),
        "recipe_family": config.get("recipe_family"),
        "precision_policy": config.get("precision_policy"),
        "runtime_policy": config.get("runtime_policy"),
        "train_seq_len": train_seq_len,
        "eval_seq_len": eval_seq_len,
        "train_batch_tokens": train_batch_tokens,
        "model_params": model_params,
        "steps_completed": int(run.steps_completed),
        "best_val_bpb": best_val_bpb,
        "final_val_bpb": run.final_val_bpb,
        "roundtrip_val_bpb": roundtrip_val_bpb,
        "roundtrip_delta_bpb": _safe_numeric(final_eval.get("roundtrip_delta_bpb")),
        "final_step_avg_ms": final_step_avg_ms,
        "final_eval_ms": final_eval_ms,
        "submission_bytes": submission_bytes,
        "quant_zlib_bytes": _safe_numeric(artifact.get("quant_zlib_bytes")),
        "code_bytes": _safe_numeric(artifact.get("code_bytes")),
        "artifact_margin_bytes": _safe_numeric(artifact.get("artifact_margin_bytes")),
        "bigram_vocab_size": bigram_vocab_size,
        "hashed_ngram_order": hashed_ngram_order,
        "retention_layers": retention_layers,
        "xsa_last_n": xsa_last_n,
        "export_quant_bits": export_quant_bits,
        "export_mlp_bits": export_mlp_bits,
        "export_attn_bits": export_attn_bits,
        "ema_decay": ema_decay,
        "use_smear_gate": bool(config.get("use_smear_gate", False)),
        "packed_train_batching": bool(config.get("packed_train_batching", False)),
        "packed_eval_windows": bool(config.get("packed_eval_windows", False)),
        "late_qat": bool(config.get("late_qat", False)),
        "use_xpos": bool(config.get("use_xpos", False)),
        "export_block_prune": bool(config.get("export_block_prune", False)),
        "ema_exported": bool(config.get("ema_exported", False)) or (ema_decay is not None and ema_decay > 0.0),
    }
    row["train_seq_len_2048"] = bool(train_seq_len >= 2048)
    row["eval_seq_len_2048"] = bool(eval_seq_len >= 2048)
    row["bigram_enabled"] = bool(bigram_vocab_size > 0)
    row["bigram_large"] = bool(bigram_vocab_size >= 4096)
    row["bigram_very_large"] = bool(bigram_vocab_size >= 16384)
    row["hashed_order3_plus"] = bool(hashed_ngram_order >= 3)
    row["retention_enabled"] = bool(retention_layers > 0)
    row["xsa_enabled"] = bool(xsa_last_n > 0)
    return row


def realized_run_correlation_analysis(
    runs: list[RunLogSummary],
    focused_analysis: list[dict[str, object]] | None = None,
    objective_lanes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if not runs:
        return {"available": False, "reason": "no_run_logs"}

    run_rows = [_component_feature_rows(run) for run in runs]
    available_roundtrip = [row for row in run_rows if _safe_numeric(row.get("roundtrip_val_bpb")) is not None]
    if available_roundtrip:
        worst_roundtrip = max(float(row["roundtrip_val_bpb"]) for row in available_roundtrip)
        for row in run_rows:
            roundtrip_val_bpb = _safe_numeric(row.get("roundtrip_val_bpb"))
            submission_bytes = _safe_numeric(row.get("submission_bytes"))
            if roundtrip_val_bpb is None or submission_bytes is None or submission_bytes <= 0.0:
                row["roundtrip_gain_per_100kb"] = None
            else:
                gain = worst_roundtrip - roundtrip_val_bpb
                row["roundtrip_gain_per_100kb"] = float(gain / (submission_bytes / 100_000.0))
    else:
        for row in run_rows:
            row["roundtrip_gain_per_100kb"] = None

    feature_names = [
        "train_seq_len",
        "eval_seq_len",
        "train_batch_tokens",
        "model_params",
        "steps_completed",
        "final_step_avg_ms",
        "submission_bytes",
        "bigram_vocab_size",
        "hashed_ngram_order",
        "retention_layers",
        "xsa_last_n",
        "export_quant_bits",
        "export_mlp_bits",
        "export_attn_bits",
        "ema_decay",
        "use_smear_gate",
        "packed_train_batching",
        "packed_eval_windows",
        "late_qat",
        "use_xpos",
        "export_block_prune",
        "ema_exported",
        "train_seq_len_2048",
        "eval_seq_len_2048",
        "bigram_large",
        "bigram_very_large",
        "hashed_order3_plus",
        "retention_enabled",
        "xsa_enabled",
    ]
    outcome_names = [
        "best_val_bpb",
        "roundtrip_val_bpb",
        "final_step_avg_ms",
        "final_eval_ms",
        "roundtrip_gain_per_100kb",
    ]
    correlations: list[dict[str, object]] = []
    for feature_name in feature_names:
        for outcome_name in outcome_names:
            if feature_name == outcome_name:
                continue
            xs: list[float] = []
            ys: list[float] = []
            for row in run_rows:
                x = _safe_numeric(row.get(feature_name))
                y = _safe_numeric(row.get(outcome_name))
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
            corr = _spearman_rank_correlation(xs, ys)
            if corr is None:
                continue
            correlations.append(
                {
                    "feature": feature_name,
                    "outcome": outcome_name,
                    "samples": int(len(xs)),
                    "spearman_rho": float(corr),
                    "preferred_direction": _metric_preference(outcome_name),
                }
            )
    correlations.sort(key=lambda row: (abs(float(row["spearman_rho"])), int(row["samples"])), reverse=True)

    component_names = [
        "use_smear_gate",
        "bigram_large",
        "bigram_very_large",
        "hashed_order3_plus",
        "train_seq_len_2048",
        "eval_seq_len_2048",
        "packed_train_batching",
        "packed_eval_windows",
        "retention_enabled",
        "xsa_enabled",
        "late_qat",
        "ema_exported",
        "export_block_prune",
        "use_xpos",
    ]
    primary_outcome = "roundtrip_val_bpb" if any(_safe_numeric(row.get("roundtrip_val_bpb")) is not None for row in run_rows) else "best_val_bpb"
    component_uplifts: list[dict[str, object]] = []
    for component_name in component_names:
        with_rows = [row for row in run_rows if bool(row.get(component_name))]
        without_rows = [row for row in run_rows if not bool(row.get(component_name))]
        with_vals = [_safe_numeric(row.get(primary_outcome)) for row in with_rows]
        without_vals = [_safe_numeric(row.get(primary_outcome)) for row in without_rows]
        with_clean = [float(value) for value in with_vals if value is not None]
        without_clean = [float(value) for value in without_vals if value is not None]
        if not with_clean or not without_clean:
            continue
        with_mean = float(np.mean(with_clean))
        without_mean = float(np.mean(without_clean))
        beneficial_delta = without_mean - with_mean if _metric_preference(primary_outcome) == "lower" else with_mean - without_mean
        component_uplifts.append(
            {
                "component": component_name,
                "outcome": primary_outcome,
                "with_count": int(len(with_clean)),
                "without_count": int(len(without_clean)),
                "with_mean": float(with_mean),
                "without_mean": float(without_mean),
                "beneficial_delta": float(beneficial_delta),
            }
        )
    component_uplifts.sort(key=lambda row: float(row["beneficial_delta"]), reverse=True)

    pairwise_interactions: list[dict[str, object]] = []
    for idx, first in enumerate(component_names):
        for second in component_names[idx + 1 :]:
            groups = {
                "none": [],
                first: [],
                second: [],
                "both": [],
            }
            for row in run_rows:
                value = _safe_numeric(row.get(primary_outcome))
                if value is None:
                    continue
                has_first = bool(row.get(first))
                has_second = bool(row.get(second))
                if has_first and has_second:
                    groups["both"].append(float(value))
                elif has_first:
                    groups[first].append(float(value))
                elif has_second:
                    groups[second].append(float(value))
                else:
                    groups["none"].append(float(value))
            if any(len(group) == 0 for group in groups.values()):
                continue
            base = float(np.mean(groups["none"]))
            gain_first = base - float(np.mean(groups[first]))
            gain_second = base - float(np.mean(groups[second]))
            gain_both = base - float(np.mean(groups["both"]))
            pairwise_interactions.append(
                {
                    "component_a": first,
                    "component_b": second,
                    "outcome": primary_outcome,
                    "baseline_mean": float(base),
                    "only_a_mean": float(np.mean(groups[first])),
                    "only_b_mean": float(np.mean(groups[second])),
                    "both_mean": float(np.mean(groups["both"])),
                    "additive_expectation_gain": float(gain_first + gain_second),
                    "realized_joint_gain": float(gain_both),
                    "synergy_gain": float(gain_both - gain_first - gain_second),
                }
            )
    pairwise_interactions.sort(key=lambda row: abs(float(row["synergy_gain"])), reverse=True)

    focused_by_name = {str(item["name"]): item for item in (focused_analysis or []) if isinstance(item, dict) and item.get("name")}
    objective_by_name = {str(item["name"]): item for item in (objective_lanes or []) if isinstance(item, dict) and item.get("name")}
    component_mapping = {
        "use_smear_gate": "smear_gate",
        "train_seq_len_2048": "train_seq_len_2048",
        "eval_seq_len_2048": "packed_eval_windows",
        "bigram_large": "bigram_4096",
        "export_block_prune": "export_block_prune",
        "late_qat": "fake_quant_tail",
        "packed_train_batching": "packed_train_batching",
        "packed_eval_windows": "packed_eval_windows",
    }
    uplift_by_component = {str(item["component"]): item for item in component_uplifts}
    profile_alignment: list[dict[str, object]] = []
    for component_name, profile_name in component_mapping.items():
        uplift_row = uplift_by_component.get(component_name)
        focused_row = focused_by_name.get(profile_name)
        objective_row = objective_by_name.get(profile_name)
        if uplift_row is None and focused_row is None and objective_row is None:
            continue
        profile_alignment.append(
            {
                "component": component_name,
                "profile_name": profile_name,
                "realized_beneficial_delta": None if uplift_row is None else float(uplift_row["beneficial_delta"]),
                "focused_priority_score": None if focused_row is None else _safe_numeric(focused_row.get("priority_score")),
                "objective_alignment_score": None
                if objective_row is None
                else _safe_numeric(objective_row.get("objective_alignment_score")),
            }
        )
    profile_alignment.sort(
        key=lambda row: abs(float(row["realized_beneficial_delta"])) if row.get("realized_beneficial_delta") is not None else -1.0,
        reverse=True,
    )

    return {
        "available": True,
        "num_runs": int(len(run_rows)),
        "primary_outcome": primary_outcome,
        "runs": run_rows,
        "top_spearman_correlations": correlations[:24],
        "component_uplifts": component_uplifts[:16],
        "pairwise_interactions": pairwise_interactions[:16],
        "profile_alignment": profile_alignment[:12],
    }


def artifact_byte_attribution_profile(
    tokenizer_path: str,
    assumed_code_bytes: int,
    current_artifact_bytes: int,
    submission_limit_bytes: int,
    anchor_run: RunLogSummary | None,
) -> dict[str, object]:
    tokenizer_bytes = None
    tokenizer_exists = False
    if tokenizer_path:
        tokenizer_file = Path(tokenizer_path)
        if tokenizer_file.exists():
            tokenizer_exists = True
            tokenizer_bytes = int(tokenizer_file.stat().st_size)

    config = {} if anchor_run is None else dict(anchor_run.config)
    artifact = {} if anchor_run is None else dict(anchor_run.artifact_summary)
    observed_code_bytes = int(artifact.get("code_bytes", assumed_code_bytes) or assumed_code_bytes)
    observed_submission_bytes = artifact.get("total_submission_bytes")
    if observed_submission_bytes is None and current_artifact_bytes > 0:
        observed_submission_bytes = int(current_artifact_bytes + observed_code_bytes)
    observed_model_artifact_bytes = artifact.get("quant_zlib_bytes")
    if observed_model_artifact_bytes is None and current_artifact_bytes > 0:
        observed_model_artifact_bytes = int(current_artifact_bytes)
    elif observed_model_artifact_bytes is None and observed_submission_bytes is not None:
        observed_model_artifact_bytes = int(max(int(observed_submission_bytes) - observed_code_bytes, 0))
    if observed_submission_bytes is None or observed_model_artifact_bytes is None:
        return {
            "available": False,
            "reason": "missing_artifact_bytes",
            "tokenizer_exists": tokenizer_exists,
            "tokenizer_bytes_if_bundled": tokenizer_bytes,
        }

    export_quant_bits = int(config.get("export_quant_bits", 6) or 6)
    lexical_buckets = int(config.get("bigram_vocab_size", 0) or 0)
    lexical_dim = int(config.get("bigram_dim", 0) or 0)
    lexical_hash_bytes_proxy = int(round(lexical_buckets * lexical_dim * max(export_quant_bits, 1) / 8.0))
    model_artifact_bytes = int(observed_model_artifact_bytes)
    remaining_after_lexical = max(model_artifact_bytes - lexical_hash_bytes_proxy, 0)
    attn_proxy = int(round(remaining_after_lexical * 0.40))
    mlp_proxy = int(round(remaining_after_lexical * 0.52))
    routing_proxy = max(remaining_after_lexical - attn_proxy - mlp_proxy, 0)
    rows = [
        {
            "name": "attention_weights_proxy",
            "estimated_bytes": int(attn_proxy),
            "basis": "proxy_split_of_remaining_model_artifact",
        },
        {
            "name": "mlp_weights_proxy",
            "estimated_bytes": int(mlp_proxy),
            "basis": "proxy_split_of_remaining_model_artifact",
        },
        {
            "name": "bigram_hash_table_proxy",
            "estimated_bytes": int(lexical_hash_bytes_proxy),
            "basis": "bigram_vocab_size * bigram_dim * export_quant_bits",
        },
        {
            "name": "routing_and_other_proxy",
            "estimated_bytes": int(routing_proxy),
            "basis": "remainder_after_attention_mlp_lexical",
        },
        {
            "name": "code_bytes_observed",
            "estimated_bytes": int(observed_code_bytes),
            "basis": "observed_or_assumed_code_bytes",
        },
    ]
    if tokenizer_bytes is not None:
        rows.append(
            {
                "name": "tokenizer_bytes_if_bundled",
                "estimated_bytes": int(tokenizer_bytes),
                "basis": "observed_tokenizer_file_size",
            }
        )
    rows.sort(key=lambda row: int(row["estimated_bytes"]), reverse=True)
    headroom = int(submission_limit_bytes - int(observed_submission_bytes))
    return {
        "available": True,
        "submission_limit_bytes": int(submission_limit_bytes),
        "observed_submission_bytes": int(observed_submission_bytes),
        "observed_model_artifact_bytes": int(observed_model_artifact_bytes),
        "observed_code_bytes": int(observed_code_bytes),
        "tokenizer_bytes_if_bundled": tokenizer_bytes,
        "tokenizer_exists": tokenizer_exists,
        "submission_headroom_bytes": int(headroom),
        "rows": rows,
        "largest_rows": rows[:8],
        "notes": [
            "Attention/MLP/routing rows are proxy allocations of the observed model artifact after subtracting the lexical hash table estimate.",
            "Code bytes and total submission bytes are observed when available; tokenizer bytes are reported as a bundling reference.",
        ],
    }


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
            "leaderboard_prior": 0.05,
            "dataset_fit": 0.2 * reuse_2048 + 0.2 * long_nmi,
            "rationale": "TTT is now strongly downweighted: dataset signal alone previously overstated it, while empirical runs and leaderboard evidence remain weak.",
            "depends_on": ["ttt_lora"],
        },
        {
            "name": "hyper_connections",
            "leaderboard_prior": 0.02,
            "dataset_fit": 0.05 * reuse_2048,
            "rationale": "Broad hyper-connection mechanisms are currently weakly supported: the dataset looks lexical-dominant and selective late compute is more plausible.",
            "depends_on": ["hyper_connections"],
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


def objective_lane_analysis(
    entropy_bits: float,
    bytes_per_token: float | None,
    lexical_profile: dict[str, object] | None,
    confidence_route_budget: dict[str, object] | None,
    loss_shape: dict[str, object] | None,
    leaderboard_analysis: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    axes = loss_shape.get("axes", {}) if isinstance(loss_shape, dict) else {}
    lexical_axis = float(axes.get("lexical_shortcut_axis", 0.0))
    long_context_axis = float(axes.get("long_context_axis", 0.0))
    train_pressure = float(axes.get("train_kernel_pressure", 0.0))
    eval_headroom = float(axes.get("eval_kernel_headroom", 0.0))
    eval_pressure = float(min(max(1.0 - eval_headroom, 0.0), 1.0))

    top4_next_byte_coverage = 0.0
    best_conditional_bits_per_byte = None
    if isinstance(lexical_profile, dict) and lexical_profile.get("available"):
        top4_next_byte_coverage = float(
            lexical_profile.get("top4_next_byte_coverage")
            or lexical_profile.get("top4_next_token_coverage")
            or 0.0
        )
        if lexical_profile.get("best_conditional_bits_per_byte") is not None:
            best_conditional_bits_per_byte = float(lexical_profile["best_conditional_bits_per_byte"])

    confidence_skip_mass = 0.0
    if isinstance(confidence_route_budget, dict) and confidence_route_budget.get("available"):
        best_safe = confidence_route_budget.get("best_compute_saver_under_false_trust_10pct")
        if isinstance(best_safe, dict):
            confidence_skip_mass = float(best_safe.get("shortcut_trust_mass", 0.0))

    tokenizer_done = 0.5 if bytes_per_token is None else float(min(max((3.4 - bytes_per_token) / 1.0, 0.0), 1.0))
    entropy_concentration = float(min(max((6.8 - entropy_bits) / 1.4, 0.0), 1.0))
    byte_predictability = float(min(max(top4_next_byte_coverage, 0.0), 1.0))
    compression_alignment = float(
        min(max(0.40 * tokenizer_done + 0.30 * entropy_concentration + 0.30 * byte_predictability, 0.0), 1.0)
    )
    if best_conditional_bits_per_byte is not None:
        compression_alignment = float(
            min(max(compression_alignment + 0.10 * min(max((1.6 - best_conditional_bits_per_byte) / 0.8, 0.0), 1.0), 0.0), 1.0)
        )

    leaderboard_by_name = {
        str(item["name"]): float(item.get("composite_score", 0.0))
        for item in (leaderboard_analysis or [])
        if isinstance(item, dict) and item.get("name") is not None
    }

    rows = [
        {
            "name": "packed_train_batching",
            "kind": "runtime",
            "dataset_fit": float(0.55 * train_pressure + 0.25 * lexical_axis + 0.20 * confidence_skip_mass),
            "leaderboard_prior": 0.45,
            "rationale": "When the train budget is tight, runtime/utilization wins are directly objective-aligned even if they do not change model capacity.",
            "suggested_knobs": ["USE_PACKED_TRAIN_BATCHING=1"],
        },
        {
            "name": "packed_eval_windows",
            "kind": "eval_runtime",
            "dataset_fit": float(0.50 * eval_pressure + 0.35 * long_context_axis + 0.15 * confidence_skip_mass),
            "leaderboard_prior": 0.40,
            "rationale": "Packed eval matters most when longer eval context looks useful but evaluation headroom is scarce.",
            "suggested_knobs": ["USE_PACKED_EVAL_WINDOWS=1"],
        },
        {
            "name": "export_calibration_percentile_mse",
            "kind": "export",
            "dataset_fit": float(0.60 * compression_alignment + 0.20 * lexical_axis + 0.20 * entropy_concentration),
            "leaderboard_prior": 0.80,
            "rationale": "Post-quant calibration is tightly aligned to the scored roundtrip artifact, especially on byte-predictable streams.",
            "suggested_knobs": ["EXPORT_CALIBRATION_METHOD=percentile_mse"],
        },
        {
            "name": "export_block_prune",
            "kind": "export",
            "dataset_fit": float(0.45 * compression_alignment + 0.25 * train_pressure + 0.20 * (1.0 - long_context_axis) + 0.10 * lexical_axis),
            "leaderboard_prior": 0.55,
            "rationale": "Block pruning is most plausible when compression pressure is high and the data looks more locally predictable than globally context-hungry.",
            "suggested_knobs": ["EXPORT_BLOCK_PRUNE=1", "EXPORT_BLOCK_PRUNE_DENSITIES=0.98,0.95,0.90"],
        },
        {
            "name": "mixed_quant_export",
            "kind": "export",
            "dataset_fit": float(0.50 * compression_alignment + 0.25 * entropy_concentration + 0.25 * tokenizer_done),
            "leaderboard_prior": max(0.85, leaderboard_by_name.get("mixed_quant_export", 0.0)),
            "rationale": "Selective mixed-precision export remains one of the most objective-aligned lanes whenever tokenizer compression already looks decent.",
            "suggested_knobs": ["EXPORT_QUANT_BITS=6", "EXPORT_MLP_BITS=5", "EXPORT_ATTN_BITS=6"],
        },
        {
            "name": "fake_quant_tail",
            "kind": "train_and_export",
            "dataset_fit": float(0.45 * compression_alignment + 0.20 * lexical_axis + 0.20 * entropy_concentration + 0.15 * train_pressure),
            "leaderboard_prior": max(0.70, leaderboard_by_name.get("fake_quant_tail", 0.0)),
            "rationale": "Cheap quantization-robustness finetuning is a strong lane whenever the final score is measured after roundtrip export.",
            "suggested_knobs": ["LATE_QAT=1", "FAKE_QUANT_TAIL_STEPS=1024"],
        },
        {
            "name": "swa_late",
            "kind": "train_and_export",
            "dataset_fit": float(0.35 * compression_alignment + 0.20 * lexical_axis + 0.15 * entropy_concentration),
            "leaderboard_prior": max(0.75, leaderboard_by_name.get("swa_late", 0.0)),
            "rationale": "Late-only averaging is not a dataset-structure lane, but it often helps the exact exported checkpoint quality that gets scored.",
            "suggested_knobs": ["EMA_DECAY=0.9999"],
        },
    ]
    for row in rows:
        row["objective_alignment_score"] = float(0.60 * float(row["leaderboard_prior"]) + 0.40 * float(row["dataset_fit"]))
        row["axes"] = {
            "compression_alignment": compression_alignment,
            "train_pressure": train_pressure,
            "eval_pressure": eval_pressure,
            "lexical_axis": lexical_axis,
            "long_context_axis": long_context_axis,
            "confidence_skip_mass": confidence_skip_mass,
        }
    return sorted(rows, key=lambda row: float(row["objective_alignment_score"]), reverse=True)


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

    nonstandard_penalties = {
        "ttt_lora_penalty": float(0.35 + 0.35 * axes["lexical_shortcut_axis"] - 0.2 * axes["long_context_axis"]),
        "hyper_connections_penalty": float(0.45 + 0.35 * axes["lexical_shortcut_axis"] - 0.15 * axes["long_context_axis"]),
    }

    return {
        "shape_name": shape_name,
        "best_eval_seq_len_by_reuse": int(best_eval["seq_len"]),
        "estimated_steps_point": est_steps,
        "train_stream_coverage_range": [coverage_min, coverage_max],
        "axes": axes,
        "nonstandard_penalties": nonstandard_penalties,
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
        "hyper_connections": 0.15,
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
            "rationale": "Eval-time adaptation had theoretical support, but should now be treated as a weak lane unless real runs prove otherwise.",
            "dataset_signal": 0.2 * reuse_2048 + 0.2 * long_nmi,
        },
        {
            "name": "hyper_connections",
            "kind": "train_and_eval",
            "rationale": "Broad hyper-connections are likely too global and too expensive for a lexical-dominant stream unless a minimal late-layer variant proves otherwise.",
            "dataset_signal": 0.05 * reuse_2048 + 0.05 * long_nmi,
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
    lexical_profile: dict[str, object] | None = None,
    hashed_lexical_profile: dict[str, object] | None = None,
    higher_order_retention: dict[str, object] | None = None,
    shard_phase_aggregate: dict[str, float] | None = None,
    eval_candidates: list[dict[str, float]] | None = None,
    training_budget: dict[str, float | int] | None = None,
    training_coverage: dict[str, object] | None = None,
    loss_shape: dict[str, object] | None = None,
    recent_copy_profile: dict[str, object] | None = None,
    lexical_control: dict[str, object] | None = None,
    proxy_calibration: dict[str, object] | None = None,
    early_budget_coverage: dict[str, object] | None = None,
    train_val_transfer: dict[str, object] | None = None,
    confidence_route_budget: dict[str, object] | None = None,
    route_calibration: dict[str, object] | None = None,
    observed_frontier: list[dict[str, object]] | None = None,
    leaderboard_frontier: list[dict[str, object]] | None = None,
    objective_lanes: list[dict[str, object]] | None = None,
    realized_run_analysis: dict[str, object] | None = None,
    artifact_byte_attribution: dict[str, object] | None = None,
    tokenizer_merge_summary: dict[str, object] | None = None,
    tokenizer_research: dict[str, object] | None = None,
    spectral_basis_recommendation: dict[str, object] | None = None,
    predictive_state_compression: dict[str, object] | None = None,
    predictive_state_recommendation: dict[str, object] | None = None,
    predictive_state_transition: dict[str, object] | None = None,
    oracle_backoff: dict[str, object] | None = None,
    past_future_cca: dict[str, object] | None = None,
    past_future_cca_recommendation: dict[str, object] | None = None,
    predictive_state_transfer: dict[str, object] | None = None,
    ppm_oracle: dict[str, object] | None = None,
    ppm_oracle_recommendation: dict[str, object] | None = None,
    minimal_causal_state: dict[str, object] | None = None,
    minimal_causal_state_recommendation: dict[str, object] | None = None,
    causal_state_reconstruction: dict[str, object] | None = None,
    state_transition_determinism: dict[str, object] | None = None,
    state_entropy_floor: dict[str, object] | None = None,
    strict_online_state_eval: dict[str, object] | None = None,
    tensor_network_state_frontier: dict[str, object] | None = None,
    candidate_state_frontier: dict[str, object] | None = None,
    dataset_world_model: dict[str, object] | None = None,
    dataset_world_model_recommendation: dict[str, object] | None = None,
    order1_proxy_peak_potential: dict[str, object] | None = None,
    next_token_loss_decomposition: dict[str, object] | None = None,
    residual_floor_dashboard: dict[str, object] | None = None,
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

    if spectral_basis_recommendation is not None:
        recommended_basis = spectral_basis_recommendation.get("recommended_basis")
        if isinstance(recommended_basis, dict):
            basis = str(recommended_basis.get("basis", "unknown"))
            lag = int(recommended_basis.get("lag", 1))
            rank = int(recommended_basis.get("rank", 0))
            if basis == "ppmi":
                recs.append(
                    f"Spectral profiling favors a lag-{lag} PPMI eigenbasis at rank {rank}; this is the strongest fixed-map candidate for a frozen output-side bias head."
                )
            else:
                recs.append(
                    f"Spectral profiling favors a lag-{lag} symmetric-transition eigenbasis at rank {rank}; test it as the first trainer-loaded fixed spectral map."
                )
    causal_state_source = minimal_causal_state_recommendation or minimal_causal_state
    if causal_state_source is not None:
        causal_best = (
            causal_state_source.get("recommended_state")
            if isinstance(causal_state_source, dict) and "recommended_state" in causal_state_source
            else causal_state_source.get("smallest_near_best_rank")
        )
        if isinstance(causal_best, dict):
            bpb = causal_best.get("heldout_bits_per_byte")
            bpb_text = "" if bpb is None else f", held-out bpb={float(bpb):.3f}"
            recs.append(
                f"Minimal causal-state profiling says the smallest near-best state is rank={int(causal_best.get('rank', 0))} with about {int(causal_best.get('estimated_fp16_state_bytes', 0))} fp16 bytes{bpb_text}."
            )
            recs.append(
                "This is the cleanest model-centric target: make the next token nearly deterministic with the smallest held-out-sufficient latent state before adding more specialized structure."
            )
    if causal_state_reconstruction is not None and bool(causal_state_reconstruction.get("available")):
        rec = causal_state_reconstruction.get("smallest_near_best_state_count") or causal_state_reconstruction.get("best_state_count_by_holdout_bpb")
        if isinstance(rec, dict):
            bpb = rec.get("heldout_bits_per_byte")
            bpb_text = "" if bpb is None else f", held-out bpb={float(bpb):.3f}"
            recs.append(
                f"Approximate causal-state reconstruction says you can collapse the stream into about {int(rec.get('num_states', 0))} predictive states for the smallest near-best machine{bpb_text}."
            )
    if state_transition_determinism is not None and bool(state_transition_determinism.get("available")):
        recs.append(
            f"Recovered causal-state transitions have entropy {float(state_transition_determinism.get('transition_entropy_bits', 0.0)):.3f} bits with mean top-1 transition mass {float(state_transition_determinism.get('mean_top1_transition_mass', 0.0)):.3f}; this is the automaton sharpness to distill into the model."
        )
    if state_entropy_floor is not None and bool(state_entropy_floor.get("available")):
        recs.append(
            f"Once the reconstructed causal state is known, the next-token entropy floor drops to about {float(state_entropy_floor.get('conditional_next_token_entropy_bits', 0.0)):.3f} bits; this is the closest current proxy to a shortest-path sufficient state."
        )
    if strict_online_state_eval is not None and bool(strict_online_state_eval.get("available")):
        online_bpb = strict_online_state_eval.get("next_token_bits_per_byte")
        if online_bpb is not None:
            recs.append(
                f"Strict online scoring says the current minimal-state probe still holds up causally at bpb={float(online_bpb):.3f}; this is the deployment-style number to optimize, not only the offline frontier."
            )
    cca_source = past_future_cca_recommendation or past_future_cca
    if cca_source is not None:
        cca_best = (
            cca_source.get("recommended_state")
            if isinstance(cca_source, dict) and "recommended_state" in cca_source
            else cca_source.get("best_rank_by_cumulative_correlation")
        )
        if isinstance(cca_best, dict):
            rank = int(cca_best.get("rank", 0))
            mean_corr = float(cca_best.get("mean_canonical_correlation", 0.0))
            past_window = int(
                cca_best.get("past_window", 0)
                if "past_window" in cca_best
                else (cca_source.get("past_window", 0) if isinstance(cca_source, dict) else 0)
            )
            future_window = int(
                cca_best.get("future_window", 0)
                if "future_window" in cca_best
                else (cca_source.get("future_window", 0) if isinstance(cca_source, dict) else 0)
            )
            recs.append(
                f"Past/future CCA is the first proper predictive-state basis here: rank={rank} keeps mean canonical correlation {mean_corr:.3f} across past_window={past_window} and future_window={future_window}."
            )
            recs.append(
                f"If you want the shortest path beyond token spectra, start from a rank-{rank} past/future state head before adding heavier operator machinery."
            )
    predictive_state_source = predictive_state_recommendation or predictive_state_compression
    if predictive_state_source is not None:
        best_rank = (
            predictive_state_source.get("recommended_state")
            if isinstance(predictive_state_source, dict) and "recommended_state" in predictive_state_source
            else predictive_state_source.get("best_rank_by_cross_entropy_gap")
        )
        if isinstance(best_rank, dict):
            rank = int(best_rank.get("rank", 0))
            gap_bits = float(best_rank.get("cross_entropy_gap_bits", 0.0))
            explained = float(best_rank.get("variance_explained", 0.0))
            order = int(
                best_rank.get("order", 0)
                if "order" in best_rank
                else (predictive_state_source.get("order", 0) if isinstance(predictive_state_source, dict) else 0)
            )
            recs.append(
                f"Predictive-state compression looks promising at order={order}, rank={rank}: the best bottleneck leaves only {gap_bits:.4f} bits of next-token cross-entropy gap while preserving {explained:.1%} of state variance."
            )
            recs.append(
                f"The shortest path above token spectra is a rank-{rank} predictive-state bias head: compress prefix-to-future posteriors offline, then learn a tiny state-conditioned logit correction online."
            )
    if predictive_state_transition is not None and bool(predictive_state_transition.get("available")):
        best_operator = predictive_state_transition.get("best_rank_by_transition_r2")
        if isinstance(best_operator, dict):
            recs.append(
                f"Predictive-state dynamics stay compressible at rank={int(best_operator.get('rank', 0))}: linear state transitions explain {float(best_operator.get('state_transition_r2', 0.0)):.1%} of next-state variance, so a tiny recurrent bias module is plausible."
            )
    if predictive_state_transfer is not None and bool(predictive_state_transfer.get("available")):
        best_tau = predictive_state_transfer.get("best_tau_by_slow_mode")
        if isinstance(best_tau, dict):
            recs.append(
                f"MSM-style transfer analysis finds the slowest predictive mode at tau={int(best_tau.get('tau', 0))}: |lambda2|={float(best_tau.get('second_eigenvalue_abs', 0.0)):.3f}, gap={float(best_tau.get('spectral_gap', 0.0)):.3f}. This is the right object for a slow-mode bias/state controller."
            )
    if tensor_network_state_frontier is not None and bool(tensor_network_state_frontier.get("available")):
        best_bond = tensor_network_state_frontier.get("best_bond_dim_by_holdout_bpb")
        if isinstance(best_bond, dict):
            bpb = best_bond.get("heldout_bits_per_byte")
            bpb_text = "" if bpb is None else f", held-out bpb={float(bpb):.3f}"
            recs.append(
                f"Tensor-network compression of the minimal state already has a viable frontier: bond_dim={int(best_bond.get('bond_dim', 0))} at about {int(best_bond.get('estimated_fp16_state_bytes', 0))} bytes{bpb_text}."
            )
    if candidate_state_frontier is not None and bool(candidate_state_frontier.get("available")):
        best_trainable = candidate_state_frontier.get("best_trainable")
        best_oracle = candidate_state_frontier.get("best_oracle")
        if isinstance(best_trainable, dict):
            recs.append(
                f"Best current trainable state mechanism is {best_trainable['component']} "
                f"at bpb={float(best_trainable.get('heldout_bits_per_byte', 0.0)):.3f}; keep the next trainer patch aligned to this frontier."
            )
        if isinstance(best_oracle, dict):
            recs.append(
                f"Best current oracle ceiling is {best_oracle['component']} "
                f"at bpb={float(best_oracle.get('heldout_bits_per_byte', 0.0)):.3f}; this is the gap the model-side state still needs to compress away."
            )
    world_source = dataset_world_model_recommendation or dataset_world_model
    if world_source is not None:
        best_world = (
            world_source.get("recommended_world_model")
            if isinstance(world_source, dict) and "recommended_world_model" in world_source
            else ((world_source.get("global_regime_model") or {}).get("best_regime_model"))
        )
        encoder = (
            (dataset_world_model or {}).get("chunk_predictive_state_encoder")
            if isinstance(dataset_world_model, dict)
            else {}
        )
        if isinstance(best_world, dict):
            chunk_tokens = int(best_world.get("chunk_tokens", encoder.get("chunk_tokens", 0)))
            num_regimes = int(best_world.get("num_regimes", 0))
            rank = int(best_world.get("state_rank", encoder.get("state_rank", 0)))
            recs.append(
                f"The dataset world model now sits above predictive-state dynamics: chunked rank-{rank} predictive states coarse-grain into {num_regimes} persistent regimes over {chunk_tokens}-token windows."
            )
            recs.append(
                f"Treat it as the universal offline prior for training: distill it into a prefix-only chunk regime classifier, regime-conditioned routing priors, and curriculum weights rather than using clairvoyant regime labels directly."
            )
    if order1_proxy_peak_potential is not None and bool(order1_proxy_peak_potential.get("available")):
        peak = order1_proxy_peak_potential.get("peak_potential")
        if isinstance(peak, dict):
            best_trainable = peak.get("best_trainable_component")
            best_clairvoyant = peak.get("best_clairvoyant_component")
            if isinstance(best_trainable, dict):
                recs.append(
                    f"Order-1 proxy peak-potential says the shortest causal drop below the lexical floor comes from {best_trainable['component']}: it trims about {float(best_trainable.get('gain_vs_order1_bits') or 0.0):.3f} bits from the exact order-1 proxy."
                )
            if isinstance(best_clairvoyant, dict) and best_clairvoyant is not best_trainable:
                recs.append(
                    f"The current clairvoyant ceiling above order-1 is {best_clairvoyant['component']}, leaving about {float(peak.get('clairvoyant_gain_vs_order1_bpb') or 0.0):.3f} bpb of profiled headroom relative to the order-1 proxy."
                )
    if next_token_loss_decomposition is not None and bool(next_token_loss_decomposition.get("available")):
        best_causal = next_token_loss_decomposition.get("best_causal_component")
        if isinstance(best_causal, dict):
            recs.append(
                f"Shortest-path next-token analysis currently favors {best_causal['component']}: it is the strongest causal component floor seen in the decomposition at {float(best_causal.get('entropy_bits', 0.0)):.3f} bits."
            )
    if residual_floor_dashboard is not None and bool(residual_floor_dashboard.get("available")):
        gap = residual_floor_dashboard.get("remaining_gap_best_causal_to_oracle_bpb")
        if gap is not None:
            recs.append(
                f"Residual-floor dashboard says the best current causal proxy still sits about {float(gap):.3f} bpb above the current oracle floor; use the largest taxonomy buckets to decide whether the next gain should come from lexical, copy, regime, or residual modeling."
            )
    if oracle_backoff is not None and bool(oracle_backoff.get("available")):
        best_entropy = oracle_backoff.get("best_by_entropy_bits")
        if isinstance(best_entropy, dict):
            max_order = int(best_entropy.get("max_order", 0))
            entropy_bits = float(best_entropy.get("entropy_bits", 0.0))
            fallback_rate = float(best_entropy.get("fallback_unigram_rate", 0.0))
            recs.append(
                f"Exact backoff oracle remains strong through order={max_order}: optimistic sampled-stream entropy falls to {entropy_bits:.3f} bits with only {fallback_rate:.1%} unigram fallback, so local causal memory is still a major untapped signal."
            )
    ppm_source = ppm_oracle_recommendation or ppm_oracle
    if ppm_source is not None:
        best_ppm = (
            ppm_source.get("recommended_order")
            if isinstance(ppm_source, dict) and "recommended_order" in ppm_source
            else (ppm_source.get("best_by_bits_per_byte") or ppm_source.get("best_by_entropy_bits"))
        )
        if isinstance(best_ppm, dict):
            bpb = best_ppm.get("bits_per_byte")
            bpb_text = "" if bpb is None else f", {float(bpb):.3f} bits/byte"
            recs.append(
                f"PPM-style causal compression is a stronger oracle than exact backoff here: best profiled order={int(best_ppm.get('max_order', 0))} reaches {float(best_ppm.get('entropy_bits', 0.0)):.3f} bits{bpb_text} on the sampled stream."
            )

    if short_reuse is not None and short_reuse >= 0.45:
        recs.append("High local reuse suggests BigramHash / SmearGate style lexical shortcuts are plausible high-ROI additions.")
    if long_reuse is not None and long_reuse >= 0.70:
        recs.append("Very high long-window reuse suggests TTT or longer eval context could exploit document-specific repetition.")
    if lexical_profile is not None and lexical_profile.get("available"):
        lexical_mi = float(lexical_profile.get("mutual_information_bits", 0.0))
        top4 = float(lexical_profile.get("top4_next_token_coverage", 0.0))
        dominant = float(lexical_profile.get("dominant_next_ge_50_rate", 0.0))
        context_orders = lexical_profile.get("context_orders")
        if lexical_mi >= 1.0 or top4 >= 0.45:
            recs.append(
                "One-token lexical context removes a large chunk of entropy in the sampled stream; prioritize more bigram capacity and smear-style local carry before adding heavier global mechanisms."
            )
        elif lexical_mi >= 0.5 or dominant >= 0.20:
            recs.append(
                "Short-context continuation structure is meaningful; soft lexical modulators should stay central even if shard-conditioned controls remain conservative."
            )
        if isinstance(context_orders, list) and len(context_orders) >= 2:
            available_orders = [item for item in context_orders if isinstance(item, dict) and item.get("available")]
            available_orders = sorted(available_orders, key=lambda item: int(item.get("order", 0)))
            if len(available_orders) >= 2:
                one = next((item for item in available_orders if int(item.get("order", 0)) == 1), available_orders[0])
                best = min(available_orders, key=lambda item: float(item.get("conditional_entropy_bits", float("inf"))))
                extra_bits = float(one.get("conditional_entropy_bits", 0.0)) - float(best.get("conditional_entropy_bits", 0.0))
                if int(best.get("order", 0)) >= 2 and extra_bits >= 0.4:
                    recs.append(
                        f"Higher-order lexical context still removes {extra_bits:.2f} bits beyond one-token context; a hashed trigram/4-gram lane is worth testing if the kernel cost stays controlled."
                    )
                elif int(best.get("order", 0)) >= 2 and extra_bits <= 0.15:
                    recs.append(
                        "Most lexical entropy collapse is already available from one-token context; prefer scaling bigram/smear before adding more complex higher-order lexical features."
                    )
    if hashed_lexical_profile is not None and hashed_lexical_profile.get("available"):
        best_entropy = hashed_lexical_profile.get("best_overall_by_entropy")
        best_bpb = hashed_lexical_profile.get("best_overall_by_bits_per_byte")
        if isinstance(best_entropy, dict):
            retained = best_entropy.get("retained_incremental_entropy_gain_fraction")
            if retained is not None and float(retained) >= 0.75:
                recs.append(
                    f"A hashed higher-order lexical lane looks robust to collisions: order={int(best_entropy['order'])} "
                    f"at {int(best_entropy['bucket_count'])} buckets retains about {100.0 * float(retained):.1f}% "
                    "of the entropy reduction beyond one-token context."
                )
            elif retained is not None and float(retained) <= 0.35:
                recs.append(
                    f"Hashed higher-order context collapses badly under collisions at the best tested point "
                    f"(order={int(best_entropy['order'])}, buckets={int(best_entropy['bucket_count'])}); "
                    "prefer larger bigram capacity or a wider hash table before adding this lane."
                )
        if isinstance(best_bpb, dict):
            retained_bpb = best_bpb.get("retained_incremental_bits_per_byte_gain_fraction")
            if retained_bpb is not None and float(retained_bpb) >= 0.75:
                recs.append(
                    f"Byte-weighted results agree: order={int(best_bpb['order'])} with {int(best_bpb['bucket_count'])} "
                    f"buckets preserves about {100.0 * float(retained_bpb):.1f}% of the bits-per-byte gain beyond one-token context."
                )
    if higher_order_retention is not None and higher_order_retention.get("available"):
        target_summary = higher_order_retention.get("target_order_summary")
        preferred_order = higher_order_retention.get("preferred_order_under_collision_budget")
        upgrade_case = higher_order_retention.get("order4_upgrade_case")
        if isinstance(target_summary, dict):
            safe_target = target_summary.get("best_under_collision_cap")
            if isinstance(safe_target, dict):
                recs.append(
                    f"Order={int(target_summary['order'])} is the current higher-order lexical sweet spot under the collision cap: "
                    f"{int(safe_target['bucket_count'])} buckets retain about "
                    f"{100.0 * float(safe_target.get('retained_incremental_entropy_gain_fraction') or 0.0):.1f}% "
                    f"of the gain beyond one-token context at collision fraction "
                    f"{100.0 * float(safe_target['collision_context_fraction']):.1f}%."
                )
            else:
                best_balanced = target_summary.get("best_balanced_bucket")
                if isinstance(best_balanced, dict):
                    recs.append(
                        f"Order={int(target_summary['order'])} still looks like the best higher-order lane, but every tested bucket setting exceeds "
                        f"the collision cap; the best-balanced point is {int(best_balanced['bucket_count'])} buckets with "
                        f"{100.0 * float(best_balanced.get('retained_incremental_entropy_gain_fraction') or 0.0):.1f}% retention."
                    )
        if isinstance(preferred_order, dict) and int(preferred_order.get("order", -1)) != int(higher_order_retention.get("target_order", 3)):
            recs.append(
                f"Under the configured collision cap, order={int(preferred_order['order'])} currently beats the target order with "
                f"{int(preferred_order['bucket_count'])} buckets; do not lock order=3 without checking that larger bucket budget."
            )
        if isinstance(upgrade_case, dict) and not bool(upgrade_case.get("order4_beats_target_under_collision_budget", False)):
            recs.append(
                f"Order=4 does not yet clear the hashed upgrade bar over order={int(upgrade_case['target_order'])}: "
                f"exact gain is {float(upgrade_case['exact_entropy_gain_vs_target_bits']):.3f} bits, but the collision-budgeted "
                f"hashed gain is only {float(upgrade_case['hashed_entropy_gain_vs_target_bits']):.3f} bits."
            )
    if recent_copy_profile is not None and recent_copy_profile.get("available"):
        best_entropy = recent_copy_profile.get("best_by_entropy")
        best_bpb = recent_copy_profile.get("best_by_bits_per_byte")
        if isinstance(best_entropy, dict):
            gain = float(best_entropy.get("incremental_entropy_reduction_vs_order1_bits", 0.0))
            coverage = float(best_entropy.get("coverage_rate", 0.0))
            if gain >= 0.10 and coverage >= 0.10:
                recs.append(
                    f"Recent exact-match copy within a local window looks promising: order={int(best_entropy['context_order'])} "
                    f"at window={int(best_entropy['window'])} covers {100.0 * coverage:.1f}% of positions and removes "
                    f"{gain:.2f} bits beyond one-token context."
                )
        if isinstance(best_bpb, dict) and best_bpb.get("incremental_bits_per_byte_reduction_vs_order1") is not None:
            gain_bpb = float(best_bpb["incremental_bits_per_byte_reduction_vs_order1"])
            if gain_bpb >= 0.03:
                recs.append(
                    f"Byte-weighted local copy also looks non-trivial: window={int(best_bpb['window'])}, "
                    f"order={int(best_bpb['context_order'])}, incremental gain={gain_bpb:.3f} bits/byte over one-token context."
                )
    if proxy_calibration is not None and proxy_calibration.get("available"):
        best_calibrated = proxy_calibration.get("best_calibrated")
        if isinstance(best_calibrated, dict):
            ece = float(best_calibrated.get("expected_calibration_error", 0.0))
            top1 = float(best_calibrated.get("top1_accuracy", 0.0))
            if ece <= 0.05:
                recs.append(
                    f"Lexical predictor calibration looks usable for control signals at "
                    f"{best_calibrated['source']} order={int(best_calibrated['order'])} "
                    f"(ECE={ece:.3f}, top1={top1:.3f})."
                )
            elif ece >= 0.10:
                recs.append(
                    f"Lexical predictor calibration is still loose (best ECE={ece:.3f}); keep confidence-driven smear and routing conservative at initialization."
                )
    if lexical_control is not None and lexical_control.get("available"):
        best_stable = lexical_control.get("best_stability_adjusted")
        if isinstance(best_stable, dict):
            shortcut_scale = float(best_stable.get("shortcut_scale_init", 0.0))
            route_scale = float(best_stable.get("lexical_route_scale_init", 0.0))
            smear_init = float(best_stable.get("smear_gate_init", 0.0))
            cold_dev = float(best_stable.get("cold_start_confidence_deviation", 0.0))
            if shortcut_scale <= 0.01 and cold_dev <= 0.05:
                recs.append(
                    f"Lexical-control sweep favors a very cold shortcut start (shortcut_scale_init={shortcut_scale:.3f}, "
                    f"smear_gate_init={smear_init:.2f}, lexical_route_scale_init={route_scale:.3f}); let the shortcut earn trust before it drives control."
                )
            else:
                recs.append(
                    f"Lexical-control sweep suggests shortcut_scale_init={shortcut_scale:.3f}, smear_gate_init={smear_init:.2f}, "
                    f"lexical_route_scale_init={route_scale:.3f} as a stable starting point."
                )
    if early_budget_coverage is not None and early_budget_coverage.get("available"):
        best_budget = early_budget_coverage.get("best_budget_fit")
        if isinstance(best_budget, dict):
            repeated_mass = float(best_budget.get("repeated_exact_transition_mass", 0.0))
            top_bucket_mass = float(best_budget.get("top_bucket_mass_fraction", 0.0))
            if repeated_mass >= 0.35:
                recs.append(
                    f"Early-budget lexical learning looks viable: by {int(best_budget['budget_tokens'])} profiled tokens, "
                    f"about {100.0 * repeated_mass:.1f}% of transition mass already comes from repeated exact contexts."
                )
            if top_bucket_mass >= 0.20:
                recs.append(
                    f"Hashed updates are concentrated early (best top-bucket mass={100.0 * top_bucket_mass:.1f}% at "
                    f"order={int(best_budget['order'])}, buckets={int(best_budget['bucket_count'])}); a finite lexical table should learn quickly if optimizer pressure stays high."
                )
    if train_val_transfer is not None and train_val_transfer.get("available"):
        best_transfer = train_val_transfer.get("best_transfer")
        if isinstance(best_transfer, dict):
            seen_rate = float(best_transfer.get("val_token_seen_rate", 0.0))
            seen_acc = float(best_transfer.get("seen_top1_accuracy") or 0.0)
            if seen_rate >= 0.50:
                recs.append(
                    f"Train/val lexical overlap is substantial for {best_transfer['source']} order={int(best_transfer['order'])}: "
                    f"{100.0 * seen_rate:.1f}% of val tokens land in seen contexts with seen-top1 accuracy {seen_acc:.3f}."
                )
            elif seen_rate <= 0.20:
                recs.append(
                    f"Validation overlap is limited for the best lexical transfer lane ({best_transfer['source']} order={int(best_transfer['order'])}); "
                    "favor robust low-order shortcuts over brittle memorization-heavy controls."
                )
    if confidence_route_budget is not None and confidence_route_budget.get("available"):
        best_safe = confidence_route_budget.get("best_compute_saver_under_false_trust_10pct")
        if isinstance(best_safe, dict):
            save_mass = float(best_safe.get("shortcut_trust_mass", 0.0))
            thr = float(best_safe.get("confidence_threshold", 0.0))
            if save_mass >= 0.20:
                recs.append(
                    f"Confidence-gated late routing looks budget-relevant: at threshold={thr:.2f}, about {100.0 * save_mass:.1f}% "
                    "of tokens could skip extra compute while keeping false-trust under 10% in the lexical proxy."
                )
            else:
                recs.append(
                    f"Confidence routing still looks narrow in the lexical proxy (best safe skip mass={100.0 * save_mass:.1f}%); "
                    "keep late-route control soft until shortcut confidence sharpens."
                )
    if route_calibration is not None and route_calibration.get("available"):
        target_route = route_calibration.get("target_hashed_order_best_row")
        low_entropy = route_calibration.get("lowest_entropy_meaningful_safe_row")
        best_safe_row = route_calibration.get("best_safe_selective_row")
        if isinstance(target_route, dict):
            recs.append(
                f"Hashed order={int(target_route['order'])} route calibration under the false-trust cap peaks around "
                f"threshold={float(target_route['confidence_threshold']):.2f}: trust mass={100.0 * float(target_route['shortcut_trust_mass']):.1f}%, "
                f"route_entropy={float(target_route['mean_route_entropy_bits']):.3f}, false_trust={100.0 * float(target_route.get('false_trust_rate') or 0.0):.1f}%."
            )
        if isinstance(low_entropy, dict) and float(low_entropy.get("mean_route_entropy_bits", 1.0)) <= 0.60:
            recs.append(
                f"Lower-entropy late routing looks achievable with {low_entropy['source']} order={int(low_entropy['order'])} "
                f"at threshold={float(low_entropy['confidence_threshold']):.2f}; route entropy drops to "
                f"{float(low_entropy['mean_route_entropy_bits']):.3f} while keeping meaningful trust mass "
                f"({100.0 * float(low_entropy['shortcut_trust_mass']):.1f}%)."
            )
        elif isinstance(best_safe_row, dict):
            recs.append(
                f"Late-route calibration is still fairly soft at the best safe point "
                f"({best_safe_row['source']} order={int(best_safe_row['order'])}, threshold={float(best_safe_row['confidence_threshold']):.2f}, "
                f"route_entropy={float(best_safe_row['mean_route_entropy_bits']):.3f}); bias routing toward higher confidence or colder route scales before opening it wider."
            )
    if objective_lanes:
        top_export = next((row for row in objective_lanes if row.get("kind") in {"export", "train_and_export"}), None)
        top_runtime = next((row for row in objective_lanes if row.get("kind") in {"runtime", "eval_runtime"}), None)
        if isinstance(top_export, dict):
            recs.append(
                f"Objective-aligned export lane: {top_export['name']} "
                f"(score={float(top_export['objective_alignment_score']):.3f}); "
                f"{top_export['rationale']}"
            )
        if isinstance(top_runtime, dict):
            recs.append(
                f"Objective-aligned runtime lane: {top_runtime['name']} "
                f"(score={float(top_runtime['objective_alignment_score']):.3f}); "
                f"{top_runtime['rationale']}"
            )

    if isinstance(realized_run_analysis, dict) and realized_run_analysis.get("available"):
        top_uplift = next(iter(realized_run_analysis.get("component_uplifts", [])), None)
        top_corr = next(iter(realized_run_analysis.get("top_spearman_correlations", [])), None)
        if isinstance(top_uplift, dict):
            delta = float(top_uplift.get("beneficial_delta", 0.0))
            outcome = str(top_uplift.get("outcome", "best_val_bpb"))
            if delta > 0.0:
                recs.append(
                    f"Realized runs currently reward {top_uplift['component']}: mean {outcome} improves by about "
                    f"{delta:.4f} across the available run logs."
                )
        if isinstance(top_corr, dict):
            recs.append(
                f"Strongest realized rank-signal so far is {top_corr['feature']} vs {top_corr['outcome']} "
                f"(Spearman rho={float(top_corr['spearman_rho']):+.3f}, n={int(top_corr['samples'])})."
            )

    if isinstance(artifact_byte_attribution, dict) and artifact_byte_attribution.get("available"):
        headroom = int(artifact_byte_attribution.get("submission_headroom_bytes", 0))
        largest_rows = artifact_byte_attribution.get("largest_rows", [])
        top_row = next(iter(largest_rows), None)
        if isinstance(top_row, dict):
            recs.append(
                f"Byte budget check: current observed submission uses {int(artifact_byte_attribution['observed_submission_bytes'])} bytes "
                f"with headroom {headroom:+d}; largest tracked bucket is {top_row['name']} "
                f"at about {int(top_row['estimated_bytes'])} bytes."
            )

    if entropy_bits <= 6.5:
        recs.append("Token entropy is fairly concentrated; compression-aware training and selective precision are likely more important than adding capacity.")
    else:
        recs.append("Token entropy is relatively high; tokenizer/data pairing shifts may still buy meaningful headroom.")

    if bytes_per_token is not None:
        if bytes_per_token >= 3.6:
            recs.append("Average bytes per token are high; tokenizer redesign may be a strong lever.")
        elif bytes_per_token <= 2.8:
            recs.append("Tokenizer compression already looks decent; focus on modeling and post-quant export interaction first.")
    if tokenizer_merge_summary is not None:
        best_small_budget = tokenizer_merge_summary.get("best_small_budget")
        if isinstance(best_small_budget, dict):
            add_tokens = int(best_small_budget.get("extra_vocab_tokens", 0))
            token_reduction = int(best_small_budget.get("estimated_token_reduction", 0))
            fits = best_small_budget.get("fits_submission_limit_estimate")
            if add_tokens > 0 and token_reduction > 0:
                fit_text = "and stays within the estimated artifact budget." if fits is True else "but needs artifact-size verification."
                recs.append(
                    f"A tiny tokenizer bump looks plausible: +{add_tokens} merges targets high-mass recurrent substrings "
                    f"(proxy {token_reduction} token events) {fit_text}"
                )
    if tokenizer_research is not None and tokenizer_research.get("available"):
        best = next(iter(tokenizer_research.get("best_by_training_usefulness", [])), None)
        conservative = next(iter(tokenizer_research.get("best_conservative_candidates", [])), None)
        if isinstance(best, dict):
            delta = best.get("estimated_token_change_fraction_vs_current")
            delta_text = ""
            if delta is not None:
                delta_text = f", token_count_change={100.0 * float(delta):+.1f}%"
            recs.append(
                f"Locally trained tokenizer probe favors {best['family']} vocab={int(best['vocab_size'])} "
                f"for training usefulness with bytes/token={float(best['bytes_per_token']):.3f}{delta_text}; use this as research guidance, not a submission dependency."
            )
        if isinstance(conservative, dict):
            delta = conservative.get("estimated_token_change_fraction_vs_current")
            delta_text = ""
            if delta is not None:
                delta_text = f", token_count_change={100.0 * float(delta):+.1f}%"
            recs.append(
                f"Conservative tokenizer lane: {conservative['family']} vocab={int(conservative['vocab_size'])} "
                f"with bytes/token={float(conservative['bytes_per_token']):.3f}{delta_text}."
            )
    if shard_aggregate is not None and shard_aggregate.get("max_js_divergence_bits", 0.0) >= 0.02:
        recs.append("Shard drift is non-trivial; TTT or mild domain-adaptive evaluation may benefit more than a one-size-fits-all context setting.")
    if shard_aggregate is not None:
        prefix_js = float(shard_aggregate.get("mean_prefix_js_to_sampled_bits", 0.0))
        nearest_js = float(shard_aggregate.get("mean_nearest_neighbor_js_bits", 0.0))
        if nearest_js >= 0.02 and prefix_js <= max(0.5 * nearest_js, 0.01):
            recs.append(
                "Short shard prefixes look discriminative relative to inter-shard drift; shard-conditioned lexical or late-route controls are plausible."
            )
        elif prefix_js > max(nearest_js, 0.02):
            recs.append(
                "Short prefixes do not identify shard state cleanly yet; rely more on lexical shortcuts than early shard-conditioned routing."
            )
    if shard_phase_aggregate is not None:
        self_advantage = float(shard_phase_aggregate.get("mean_prefix_self_advantage_bits", 0.0))
        positive_advantage = float(shard_phase_aggregate.get("positive_self_advantage_rate", 0.0))
        if self_advantage >= 0.01 and positive_advantage >= 0.65:
            recs.append(
                "Early-shard evidence stays closer to its own later phase than to other shards; shard-conditioned modulation can remain useful deeper in the sequence if it stays soft."
            )
        elif self_advantage <= 0.0:
            recs.append(
                "Within-shard phase drift is washing out the early shard signal; avoid turning shard identity into a hard controller and lean more on local lexical evidence."
            )
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
    if training_budget is not None:
        estimated_tokens = int(training_budget.get("estimated_train_tokens", 0) or 0)
        sample_fraction = None
        if estimated_tokens > 0 and shard_aggregate is not None:
            profiled_tokens = int(shard_aggregate.get("sample_tokens_per_shard", 0) or 0) * int(
                shard_aggregate.get("num_shards_profiled", 0) or 0
            )
            if profiled_tokens > 0:
                sample_fraction = float(profiled_tokens / estimated_tokens)
        if sample_fraction is not None and sample_fraction < 0.01:
            recs.append(
                f"Current profiling only samples about {100.0 * sample_fraction:.2f}% of the effective training-token budget; spectral basis rankings may shift if you push max_tokens closer to the real 10-minute training regime."
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
    parser.add_argument(
        "--max-tokens-fraction-of-training-budget",
        type=float,
        default=0.0,
        help="If > 0, derive the profiling token cap as this fraction of the estimated training-token budget.",
    )
    parser.add_argument("--sample-across-shards", action="store_true")
    parser.add_argument("--lags", default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--reuse-windows", default="128,512,1024,2048")
    parser.add_argument("--eval-lengths", default="512,1024,1408,2048")
    parser.add_argument("--lexical-context-orders", default="1,2,4")
    parser.add_argument("--hashed-lexical-orders", default="2,3,4")
    parser.add_argument("--hashed-lexical-buckets", default="1024,2048,4096,8192,16384")
    parser.add_argument("--recent-copy-windows", default="128,256,512")
    parser.add_argument("--recent-copy-orders", default="1,2,4")
    parser.add_argument("--unconstrained-context-orders", default="1,2,3,4,5")
    parser.add_argument("--unconstrained-recent-copy-windows", default="128,256,512,1024,2048")
    parser.add_argument("--proxy-calibration-bins", type=int, default=15)
    parser.add_argument("--lexical-control-orders", default="2,3")
    parser.add_argument("--lexical-control-buckets", default="8192,16384,65536")
    parser.add_argument("--lexical-control-shortcut-scales", default="0.0,0.01,0.02,0.05")
    parser.add_argument("--lexical-control-smear-gates", default="-2.0,-1.0,-0.5")
    parser.add_argument("--lexical-control-route-scales", default="0.02,0.05,0.08")
    parser.add_argument("--budget-coverage-token-points", default="1000000,4000000,16000000,64000000")
    parser.add_argument("--budget-coverage-orders", default="2")
    parser.add_argument("--budget-coverage-buckets", default="8192,16384,65536")
    parser.add_argument("--budget-coverage-topk", type=int, default=128)
    parser.add_argument("--transfer-exact-orders", default="1,2,4")
    parser.add_argument("--transfer-hashed-orders", default="2")
    parser.add_argument("--transfer-hashed-buckets", default="8192,16384,65536")
    parser.add_argument("--confidence-route-exact-orders", default="1,2")
    parser.add_argument("--confidence-route-hashed-orders", default="2,3")
    parser.add_argument("--confidence-route-buckets", default="8192,16384,65536")
    parser.add_argument("--confidence-route-thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--sample-tokens-per-shard", type=int, default=250_000)
    parser.add_argument("--shard-prefix-tokens", type=int, default=512)
    parser.add_argument("--shard-phase-segments", type=int, default=4)
    parser.add_argument("--shard-cluster-k-values", default="4,8,16")
    parser.add_argument("--shard-neighbor-topk", type=int, default=8)
    parser.add_argument("--include-shard-summaries", action="store_true")
    parser.add_argument("--max-wallclock-seconds", type=float, default=600.0)
    parser.add_argument("--avg-step-ms", type=float, default=85.0)
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
    parser.add_argument("--tokenizer-candidate-count", type=int, default=64)
    parser.add_argument("--tokenizer-budget-options", default="16,32,64")
    parser.add_argument("--train-local-tokenizers", action="store_true")
    parser.add_argument("--tokenizer-trainer-families", default="sp_bpe,sp_unigram,hf_bpe")
    parser.add_argument("--tokenizer-train-vocab-sizes", default="768,1024,1280,1536,2048")
    parser.add_argument("--tokenizer-output-dir", default="")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--current-artifact-bytes", type=int, default=0)
    parser.add_argument("--submission-limit-bytes", type=int, default=16_000_000)
    parser.add_argument("--assumed-code-bytes", type=int, default=200_000)
    parser.add_argument("--spectral-eigen-rank", type=int, default=16)
    parser.add_argument("--spectral-lags", default="1,2,4,8")
    parser.add_argument("--cca-past-window", type=int, default=16)
    parser.add_argument("--cca-future-window", type=int, default=8)
    parser.add_argument("--cca-sketch-dim", type=int, default=128)
    parser.add_argument("--cca-ranks", default="4,8,16,32")
    parser.add_argument("--cca-max-prefixes", type=int, default=32768)
    parser.add_argument("--cca-regularization", type=float, default=1e-3)
    parser.add_argument("--causal-state-ranks", default="")
    parser.add_argument("--causal-state-max-prefixes", type=int, default=32768)
    parser.add_argument("--causal-state-holdout-fraction", type=float, default=0.2)
    parser.add_argument("--causal-state-ridge", type=float, default=1e-2)
    parser.add_argument("--causal-state-near-best-bpb-tol", type=float, default=0.01)
    parser.add_argument("--causal-state-near-best-bits-tol", type=float, default=0.05)
    parser.add_argument("--causal-machine-state-counts", default="8,16,32,64,128")
    parser.add_argument("--causal-machine-kmeans-iters", type=int, default=12)
    parser.add_argument("--causal-machine-future-horizons", default="1,2,4,8")
    parser.add_argument("--strict-online-max-eval-tokens", type=int, default=65536)
    parser.add_argument("--strict-online-horizons", default="1,2,4,8")
    parser.add_argument("--tensor-bond-dims", default="2,4,8,16")
    parser.add_argument("--transfer-taus", default="1,2,4,8")
    parser.add_argument("--transfer-clusters", type=int, default=32)
    parser.add_argument("--transfer-kmeans-iters", type=int, default=12)
    parser.add_argument("--oracle-backoff-orders", default="2,3,4,5,6")
    parser.add_argument("--oracle-backoff-max-eval-tokens", type=int, default=65536)
    parser.add_argument("--ppm-orders", default="2,3,4,5,6")
    parser.add_argument("--ppm-max-eval-tokens", type=int, default=65536)
    parser.add_argument("--predictive-state-order", type=int, default=4)
    parser.add_argument("--predictive-state-ranks", default="4,8,16,32")
    parser.add_argument("--predictive-state-max-prefixes", type=int, default=32768)
    parser.add_argument("--world-model-chunk-tokens", type=int, default=65536)
    parser.add_argument("--world-model-chunk-stride", type=int, default=32768)
    parser.add_argument("--world-model-prefix-tokens", type=int, default=1024)
    parser.add_argument("--world-model-regime-counts", default="4,8,16")
    parser.add_argument("--world-model-kmeans-iters", type=int, default=16)
    parser.add_argument("--world-model-top-tokens", type=int, default=8)
    parser.add_argument("--cache-dir", default="runs/profile_cache")
    parser.add_argument("--no-resume-cache", action="store_true")
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    files = resolve_files(args.train_glob, args.max_shards)
    val_files = resolve_files(args.val_glob, 0)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resume_cache = not bool(args.no_resume_cache)
    profile_workers = max(int(args.profile_workers), 1)
    parsed_lags = parse_int_list(args.lags)
    parsed_reuse_windows = parse_int_list(args.reuse_windows)
    parsed_eval_lengths = parse_int_list(args.eval_lengths)
    parsed_lexical_context_orders = parse_int_list(args.lexical_context_orders)
    parsed_hashed_lexical_orders = parse_int_list(args.hashed_lexical_orders)
    parsed_hashed_lexical_buckets = parse_int_list(args.hashed_lexical_buckets)
    parsed_recent_copy_windows = parse_int_list(args.recent_copy_windows)
    parsed_recent_copy_orders = parse_int_list(args.recent_copy_orders)
    parsed_unconstrained_context_orders = parse_int_list(args.unconstrained_context_orders)
    parsed_unconstrained_recent_copy_windows = parse_int_list(args.unconstrained_recent_copy_windows)
    parsed_lexical_control_orders = parse_int_list(args.lexical_control_orders)
    parsed_lexical_control_buckets = parse_int_list(args.lexical_control_buckets)
    parsed_lexical_control_shortcut_scales = parse_float_list(args.lexical_control_shortcut_scales)
    parsed_lexical_control_smear_gates = parse_float_list(args.lexical_control_smear_gates)
    parsed_lexical_control_route_scales = parse_float_list(args.lexical_control_route_scales)
    parsed_budget_coverage_token_points = parse_int_list(args.budget_coverage_token_points)
    parsed_budget_coverage_orders = parse_int_list(args.budget_coverage_orders)
    parsed_budget_coverage_buckets = parse_int_list(args.budget_coverage_buckets)
    parsed_transfer_exact_orders = parse_int_list(args.transfer_exact_orders)
    parsed_transfer_hashed_orders = parse_int_list(args.transfer_hashed_orders)
    parsed_transfer_hashed_buckets = parse_int_list(args.transfer_hashed_buckets)
    parsed_confidence_route_exact_orders = parse_int_list(args.confidence_route_exact_orders)
    parsed_confidence_route_hashed_orders = parse_int_list(args.confidence_route_hashed_orders)
    parsed_confidence_route_buckets = parse_int_list(args.confidence_route_buckets)
    parsed_confidence_route_thresholds = parse_float_list(args.confidence_route_thresholds)
    parsed_spectral_lags = parse_int_list(args.spectral_lags)
    parsed_cca_ranks = parse_int_list(args.cca_ranks)
    parsed_causal_state_ranks = parse_int_list(args.causal_state_ranks) if args.causal_state_ranks.strip() else parsed_cca_ranks
    parsed_causal_machine_state_counts = parse_int_list(args.causal_machine_state_counts)
    parsed_causal_machine_future_horizons = parse_int_list(args.causal_machine_future_horizons)
    parsed_strict_online_horizons = parse_int_list(args.strict_online_horizons)
    parsed_tensor_bond_dims = parse_int_list(args.tensor_bond_dims)
    parsed_transfer_taus = parse_int_list(args.transfer_taus)
    parsed_predictive_state_ranks = parse_int_list(args.predictive_state_ranks)
    parsed_oracle_backoff_orders = parse_int_list(args.oracle_backoff_orders)
    parsed_ppm_orders = parse_int_list(args.ppm_orders)
    parsed_world_model_regime_counts = parse_int_list(args.world_model_regime_counts)
    train_stream = shard_token_totals(files)
    val_stream = shard_token_totals(val_files)
    requested_max_tokens = int(args.max_tokens)
    training_budget_for_sampling = estimate_training_budget(
        max_wallclock_seconds=args.max_wallclock_seconds,
        avg_step_ms=args.avg_step_ms,
        train_batch_tokens=args.train_batch_tokens,
        train_seq_len=args.train_seq_len,
        world_size=args.world_size,
        grad_accum_steps=args.grad_accum_steps,
    )
    derived_budget_tokens = None
    effective_max_tokens = requested_max_tokens
    if float(args.max_tokens_fraction_of_training_budget) > 0.0:
        derived_budget_tokens = int(
            max(
                1,
                round(
                    float(training_budget_for_sampling["estimated_train_tokens"])
                    * float(args.max_tokens_fraction_of_training_budget)
                ),
            )
        )
        if requested_max_tokens > 0:
            effective_max_tokens = max(requested_max_tokens, derived_budget_tokens)
        else:
            effective_max_tokens = derived_budget_tokens
    if effective_max_tokens > 0:
        effective_max_tokens = min(int(effective_max_tokens), int(train_stream["total_tokens"]))
    sequential_tokens, sequential_tokens_cache = materialize_token_stream_memmap(
        files,
        effective_max_tokens,
        cache_dir,
        "train_sequential",
        sample_across_shards_mode=False,
    )
    val_sequential_tokens, val_sequential_tokens_cache = materialize_token_stream_memmap(
        val_files,
        effective_max_tokens,
        cache_dir,
        "val_sequential",
        sample_across_shards_mode=False,
    )
    if args.sample_across_shards:
        distribution_tokens, distribution_tokens_cache = materialize_token_stream_memmap(
            files,
            effective_max_tokens,
            cache_dir,
            "train_distribution",
            sample_across_shards_mode=True,
        )
    else:
        distribution_tokens = sequential_tokens
        distribution_tokens_cache = {
            "cache_key": sequential_tokens_cache["cache_key"],
            "path": sequential_tokens_cache["path"],
            "cache_hit": sequential_tokens_cache["cache_hit"],
            "sample_across_shards_mode": False,
        }
    token_counts = np.bincount(distribution_tokens, minlength=args.vocab_size).astype(np.int64)
    entropy_bits = entropy_from_counts(token_counts)
    effective_vocab = float(2.0**entropy_bits)
    base_bytes = build_base_bytes(args.tokenizer_path, args.vocab_size) if args.tokenizer_path else None
    tokenizer_meta = build_tokenizer_metadata(args.tokenizer_path, args.vocab_size) if args.tokenizer_path else None
    bytes_per_token = None if base_bytes is None else float(base_bytes[distribution_tokens].mean())
    cache_summary = {
        "cache_dir": str(cache_dir),
        "resume_cache": bool(resume_cache),
        "profile_workers_requested": int(profile_workers),
        "token_streams": {
            "train_sequential": sequential_tokens_cache,
            "val_sequential": val_sequential_tokens_cache,
            "distribution": distribution_tokens_cache,
        },
        "bundle_hits": [],
        "bundle_misses": [],
    }
    bundle_specs = {
        "basic_signal": {
            "sequential_tokens_path": sequential_tokens_cache["path"],
            "vocab_size": args.vocab_size,
            "top_k": args.top_k,
            "token_counts": token_counts,
            "lags": parsed_lags,
            "reuse_windows": parsed_reuse_windows,
            "eval_lengths": parsed_eval_lengths,
        },
        "lexical": {
            "sequential_tokens_path": sequential_tokens_cache["path"],
            "vocab_size": args.vocab_size,
            "token_counts": token_counts,
            "base_bytes": base_bytes,
            "lexical_context_orders": parsed_lexical_context_orders,
            "hashed_lexical_orders": parsed_hashed_lexical_orders,
            "hashed_lexical_buckets": parsed_hashed_lexical_buckets,
            "recent_copy_windows": parsed_recent_copy_windows,
            "recent_copy_orders": parsed_recent_copy_orders,
            "unconstrained_context_orders": parsed_unconstrained_context_orders,
            "unconstrained_recent_copy_windows": parsed_unconstrained_recent_copy_windows,
            "proxy_calibration_bins": args.proxy_calibration_bins,
            "lexical_control_orders": parsed_lexical_control_orders,
            "lexical_control_buckets": parsed_lexical_control_buckets,
            "lexical_control_shortcut_scales": parsed_lexical_control_shortcut_scales,
            "lexical_control_smear_gates": parsed_lexical_control_smear_gates,
            "lexical_control_route_scales": parsed_lexical_control_route_scales,
            "budget_coverage_token_points": parsed_budget_coverage_token_points,
            "budget_coverage_orders": parsed_budget_coverage_orders,
            "budget_coverage_buckets": parsed_budget_coverage_buckets,
            "budget_coverage_topk": args.budget_coverage_topk,
            "confidence_route_exact_orders": parsed_confidence_route_exact_orders,
            "confidence_route_hashed_orders": parsed_confidence_route_hashed_orders,
            "confidence_route_buckets": parsed_confidence_route_buckets,
            "confidence_route_thresholds": parsed_confidence_route_thresholds,
        },
        "transfer_shard": {
            "sequential_tokens_path": sequential_tokens_cache["path"],
            "val_sequential_tokens_path": val_sequential_tokens_cache["path"],
            "files": [str(path) for path in files],
            "vocab_size": args.vocab_size,
            "base_bytes": base_bytes,
            "transfer_exact_orders": parsed_transfer_exact_orders,
            "transfer_hashed_orders": parsed_transfer_hashed_orders,
            "transfer_hashed_buckets": parsed_transfer_hashed_buckets,
            "sample_tokens_per_shard": args.sample_tokens_per_shard,
            "shard_prefix_tokens": args.shard_prefix_tokens,
            "shard_cluster_k_values": parse_int_list(args.shard_cluster_k_values),
            "shard_neighbor_topk": args.shard_neighbor_topk,
            "shard_phase_segments": args.shard_phase_segments,
            "top_k": args.top_k,
        },
        "state": {
            "sequential_tokens_path": sequential_tokens_cache["path"],
            "vocab_size": args.vocab_size,
            "token_counts": token_counts,
            "base_bytes": base_bytes,
            "tokenizer_meta": tokenizer_meta,
            "spectral_eigen_rank": args.spectral_eigen_rank,
            "spectral_lags": parsed_spectral_lags,
            "top_k": args.top_k,
            "cca_past_window": args.cca_past_window,
            "cca_future_window": args.cca_future_window,
            "cca_sketch_dim": args.cca_sketch_dim,
            "cca_ranks": parsed_cca_ranks,
            "cca_max_prefixes": args.cca_max_prefixes,
            "cca_regularization": args.cca_regularization,
            "causal_state_ranks": parsed_causal_state_ranks,
            "causal_state_max_prefixes": args.causal_state_max_prefixes,
            "causal_state_holdout_fraction": args.causal_state_holdout_fraction,
            "causal_state_ridge": args.causal_state_ridge,
            "causal_state_near_best_bpb_tol": args.causal_state_near_best_bpb_tol,
            "causal_state_near_best_bits_tol": args.causal_state_near_best_bits_tol,
            "causal_machine_state_counts": parsed_causal_machine_state_counts,
            "causal_machine_kmeans_iters": args.causal_machine_kmeans_iters,
            "causal_machine_future_horizons": parsed_causal_machine_future_horizons,
            "strict_online_max_eval_tokens": args.strict_online_max_eval_tokens,
            "strict_online_horizons": parsed_strict_online_horizons,
            "tensor_bond_dims": parsed_tensor_bond_dims,
            "transfer_taus": parsed_transfer_taus,
            "transfer_clusters": args.transfer_clusters,
            "transfer_kmeans_iters": args.transfer_kmeans_iters,
            "predictive_state_order": args.predictive_state_order,
            "predictive_state_ranks": parsed_predictive_state_ranks,
            "predictive_state_max_prefixes": args.predictive_state_max_prefixes,
            "oracle_backoff_orders": parsed_oracle_backoff_orders,
            "oracle_backoff_max_eval_tokens": args.oracle_backoff_max_eval_tokens,
            "ppm_orders": parsed_ppm_orders,
            "ppm_max_eval_tokens": args.ppm_max_eval_tokens,
            "world_model_chunk_tokens": args.world_model_chunk_tokens,
            "world_model_chunk_stride": args.world_model_chunk_stride,
            "world_model_prefix_tokens": args.world_model_prefix_tokens,
            "world_model_regime_counts": parsed_world_model_regime_counts,
            "world_model_kmeans_iters": args.world_model_kmeans_iters,
            "world_model_top_tokens": args.world_model_top_tokens,
        },
    }
    bundle_results: dict[str, dict[str, object]] = {}
    pending_futures: dict[object, tuple[str, str]] = {}
    executor: ProcessPoolExecutor | None = None
    if profile_workers > 1:
        executor = ProcessPoolExecutor(max_workers=profile_workers)
    try:
        for bundle_name, payload in bundle_specs.items():
            cache_key = build_cache_key(f"profile_bundle_{bundle_name}", payload)
            cached = load_cached_stage(cache_dir, f"profile_bundle_{bundle_name}", cache_key) if resume_cache else None
            if isinstance(cached, dict):
                bundle_results[bundle_name] = cached
                cache_summary["bundle_hits"].append(bundle_name)
                continue
            cache_summary["bundle_misses"].append(bundle_name)
            if executor is None:
                result = _run_profile_bundle(bundle_name, payload)
                save_cached_stage(cache_dir, f"profile_bundle_{bundle_name}", cache_key, result)
                bundle_results[bundle_name] = result
            else:
                pending_futures[executor.submit(_run_profile_bundle, bundle_name, payload)] = (bundle_name, cache_key)
        for future in as_completed(list(pending_futures.keys())):
            bundle_name, cache_key = pending_futures[future]
            result = future.result()
            save_cached_stage(cache_dir, f"profile_bundle_{bundle_name}", cache_key, result)
            bundle_results[bundle_name] = result
    finally:
        if executor is not None:
            executor.shutdown()

    basic_bundle = bundle_results["basic_signal"]
    lag_stats = basic_bundle["lag_stats"]
    reuse_stats = basic_bundle["reuse_stats"]
    eval_candidates = basic_bundle["eval_candidates"]
    marginal_context = basic_bundle["marginal_context"]
    context_bands = basic_bundle["context_bands"]
    transition_geometry = basic_bundle["transition_geometry"]
    recurrence_profile = basic_bundle["recurrence_profile"]
    recurrence_by_bucket = basic_bundle["recurrence_by_bucket"]
    top_bigrams_rows = basic_bundle["top_bigrams"]

    lexical_bundle = bundle_results["lexical"]
    lexical_profile = lexical_bundle["lexical_profile"]
    hashed_lexical_profile = lexical_bundle["hashed_lexical_profile"]
    recent_copy = lexical_bundle["recent_copy"]
    unconstrained_collapse = lexical_bundle["unconstrained_collapse"]
    proxy_calibration = lexical_bundle["proxy_calibration"]
    lexical_control = lexical_bundle["lexical_control"]
    early_budget_coverage = lexical_bundle["early_budget_coverage"]
    confidence_route_budget = lexical_bundle["confidence_route_budget"]
    higher_order_retention = lexical_bundle["higher_order_retention"]
    route_calibration = lexical_bundle["route_calibration"]

    transfer_shard_bundle = bundle_results["transfer_shard"]
    train_val_transfer = transfer_shard_bundle["train_val_transfer"]
    shard_summary_rows = transfer_shard_bundle["shard_summary_rows"]
    shard_aggregate = transfer_shard_bundle["shard_aggregate"]
    shard_clustering = transfer_shard_bundle["shard_clustering"]
    shard_phase = transfer_shard_bundle["shard_phase"]
    shard_aggregate = {
        **shard_aggregate,
        **{str(key): value for key, value in shard_clustering["aggregate"].items()},
    }
    shard_aggregate = {
        **shard_aggregate,
        **{f"phase_{key}": value for key, value in shard_phase["aggregate"].items()},
    }

    state_bundle = bundle_results["state"]
    spectral_eigenbases = state_bundle["spectral_eigenbases"]
    spectral_eigenbasis_arrays = state_bundle["spectral_eigenbasis_arrays"]
    lagged_spectral_eigenbases = state_bundle["lagged_spectral_eigenbases"]
    lagged_spectral_eigenbasis_arrays = state_bundle["lagged_spectral_eigenbasis_arrays"]
    spectral_recommendation = state_bundle["spectral_recommendation"]
    past_future_cca = state_bundle["past_future_cca"]
    past_future_cca_arrays = state_bundle["past_future_cca_arrays"]
    past_future_cca_reco = state_bundle["past_future_cca_reco"]
    predictive_state_transfer = state_bundle["predictive_state_transfer"]
    minimal_causal_state = state_bundle["minimal_causal_state"]
    minimal_causal_state_arrays = state_bundle["minimal_causal_state_arrays"]
    minimal_causal_state_reco = state_bundle["minimal_causal_state_reco"]
    future_signature = state_bundle["future_signature"]
    causal_state_reconstruction = state_bundle["causal_state_reconstruction"]
    causal_state_arrays = state_bundle["causal_state_arrays"]
    state_transition_determinism = state_bundle["state_transition_determinism"]
    state_entropy_floor = state_bundle["state_entropy_floor"]
    causal_state_decodability = state_bundle.get("causal_state_decodability", {"available": False, "reason": "missing_state_bundle_key"})
    causal_state_decodability_arrays = state_bundle.get("causal_state_decodability_arrays", {})
    causal_state_transition_learnability = state_bundle.get(
        "causal_state_transition_learnability",
        {"available": False, "reason": "missing_state_bundle_key"},
    )
    causal_state_transition_learnability_arrays = state_bundle.get("causal_state_transition_learnability_arrays", {})
    causal_state_multi_horizon_sufficiency = state_bundle.get(
        "causal_state_multi_horizon_sufficiency",
        {"available": False, "reason": "missing_state_bundle_key"},
    )
    causal_state_merge_error = state_bundle.get("causal_state_merge_error", {"available": False, "reason": "missing_state_bundle_key"})
    causal_state_residual_geometry = state_bundle.get(
        "causal_state_residual_geometry",
        {"available": False, "reason": "missing_state_bundle_key"},
    )
    causal_state_residual_geometry_arrays = state_bundle.get("causal_state_residual_geometry_arrays", {})
    strict_online_state_eval = state_bundle.get("strict_online_state_eval", {"available": False, "reason": "missing_state_bundle_key"})
    tensor_network_state_frontier = state_bundle.get(
        "tensor_network_state_frontier",
        {"available": False, "reason": "missing_state_bundle_key"},
    )
    predictive_state_compression = state_bundle["predictive_state_compression"]
    predictive_state_arrays = state_bundle["predictive_state_arrays"]
    predictive_state_recommendation = state_bundle["predictive_state_recommendation"]
    predictive_state_transition = state_bundle["predictive_state_transition"]
    predictive_state_transition_recommendation = state_bundle["predictive_state_transition_recommendation"]
    oracle_backoff = state_bundle["oracle_backoff"]
    ppm_oracle = state_bundle["ppm_oracle"]
    ppm_oracle_reco = state_bundle["ppm_oracle_reco"]
    dataset_world_model = state_bundle["dataset_world_model"]
    dataset_world_model_arrays = state_bundle["dataset_world_model_arrays"]
    dataset_world_model_reco = state_bundle["dataset_world_model_reco"]
    regime_conditioned_bpb = state_bundle["regime_conditioned_bpb"]
    regime_conditioned_bpb_arrays = state_bundle["regime_conditioned_bpb_arrays"]

    reconstructed_text = reconstruct_text_from_tokens(sequential_tokens, tokenizer_meta) if args.train_local_tokenizers else None
    training_budget = training_budget_for_sampling
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
        eval_lengths=parsed_eval_lengths,
        eval_batch_seqs=args.base_eval_batch_seqs,
        max_wallclock_seconds=args.max_wallclock_seconds,
        base_eval_seq_len=args.base_eval_seq_len,
        base_eval_step_ms=args.base_eval_step_ms,
        ttt_multipliers=parse_float_list(args.ttt_multipliers),
    )
    real_eval_frontier = estimate_real_eval_budget_frontier(
        total_val_tokens=int(val_stream["total_tokens"]),
        eval_lengths=parsed_eval_lengths,
        eval_stride=64,
        eval_batch_seqs=args.base_eval_batch_seqs,
        max_wallclock_seconds=args.max_wallclock_seconds,
        base_eval_seq_len=args.base_eval_seq_len,
        base_eval_batch_seqs=args.base_eval_batch_seqs,
        base_eval_step_ms=args.base_eval_step_ms,
        ttt_multipliers=parse_float_list(args.ttt_multipliers),
    )
    next_token_loss_decomposition = next_token_loss_decomposition_profile(
        sequential_tokens,
        token_counts,
        base_bytes,
        lexical_profile,
        recent_copy,
        ppm_oracle,
        regime_conditioned_bpb,
    )
    posterior_error_taxonomy = posterior_error_taxonomy_profile(
        sequential_tokens,
        args.vocab_size,
        token_counts,
        base_bytes,
        recent_copy,
        dataset_world_model,
        dataset_world_model_arrays,
        regime_conditioned_bpb_arrays,
    )
    residual_floor_dashboard = residual_floor_dashboard_profile(
        next_token_loss_decomposition,
        posterior_error_taxonomy,
    )
    order1_proxy_peak_potential = order1_proxy_peak_potential_profile(
        lexical_profile,
        recent_copy,
        proxy_calibration,
        confidence_route_budget,
        regime_conditioned_bpb,
        ppm_oracle,
    )
    tokenizer_candidates = tokenizer_merge_candidates(
        sequential_tokens,
        args.vocab_size,
        tokenizer_meta,
        args.tokenizer_candidate_count,
    )
    tokenizer_budget = tokenizer_budget_analysis(
        tokenizer_candidates,
        parse_int_list(args.tokenizer_budget_options),
        model_dim=args.model_dim,
        current_vocab_size=args.vocab_size,
        submission_limit_bytes=args.submission_limit_bytes,
        code_size_bytes=args.assumed_code_bytes,
        current_artifact_bytes=(args.current_artifact_bytes if args.current_artifact_bytes > 0 else None),
    )
    tokenizer_merge_summary = {
        "candidate_count": int(len(tokenizer_candidates)),
        "best_small_budget": tokenizer_budget[0] if tokenizer_budget else None,
    }
    tokenizer_research = (
        train_local_tokenizer_candidates(
            text=reconstructed_text,
            output_dir=Path(args.tokenizer_output_dir) if args.tokenizer_output_dir else Path("runs/tokenizer_research"),
            vocab_sizes=parse_int_list(args.tokenizer_train_vocab_sizes),
            trainer_families=[part.strip() for part in args.tokenizer_trainer_families.split(",") if part.strip()],
            current_vocab_size=args.vocab_size,
            model_dim=args.model_dim,
            current_bytes_per_token=bytes_per_token,
        )
        if args.train_local_tokenizers
        else {"available": False, "reason": "train_local_tokenizers_disabled"}
    )
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
    objective_analysis = objective_lane_analysis(
        entropy_bits=entropy_bits,
        bytes_per_token=bytes_per_token,
        lexical_profile=lexical_profile,
        confidence_route_budget=confidence_route_budget,
        loss_shape=loss_shape,
        leaderboard_analysis=leaderboard_analysis,
    )
    realized_run_analysis = realized_run_correlation_analysis(
        run_log_summaries,
        focused_analysis=focused_analysis,
        objective_lanes=objective_analysis,
    )
    anchor_run = min(
        [item for item in run_log_summaries if item.best_val_bpb is not None and item.final_step_avg_ms is not None],
        key=lambda item: float(item.best_val_bpb),
        default=None,
    )
    candidate_state_frontier = candidate_state_frontier_profile(
        anchor_run.best_val_bpb if anchor_run is not None else None,
        strict_online_state_eval,
        minimal_causal_state,
        minimal_causal_state_reco,
        causal_state_reconstruction,
        tensor_network_state_frontier,
        ppm_oracle,
        oracle_backoff,
        regime_conditioned_bpb,
    )
    artifact_byte_attribution = artifact_byte_attribution_profile(
        tokenizer_path=args.tokenizer_path,
        assumed_code_bytes=args.assumed_code_bytes,
        current_artifact_bytes=args.current_artifact_bytes,
        submission_limit_bytes=args.submission_limit_bytes,
        anchor_run=anchor_run,
    )
    current_model_params = anchor_run.model_params if anchor_run is not None else None
    current_best_val_bpb = anchor_run.best_val_bpb if anchor_run is not None else None
    candidate_models = parse_candidate_models(
        args.candidate_models,
        default_vocab_size=args.vocab_size,
        default_num_heads=8,
        default_num_kv_heads=4,
    )
    if not candidate_models:
        candidate_models = default_candidate_models_from_anchor(
            anchor_run,
            default_vocab_size=args.vocab_size,
        )
    model_frontier = budgeted_model_candidates(
        candidates=candidate_models,
        anchor_run=anchor_run,
        max_wallclock_seconds=args.max_wallclock_seconds,
        total_train_tokens_available=int(train_stream["total_tokens"]),
        current_model_params=current_model_params,
        current_best_val_bpb=current_best_val_bpb,
    )
    competition_transformer = competition_transformer_recommendation(
        anchor_run=anchor_run,
        model_frontier=model_frontier,
        observed_frontier=observed_frontier,
        real_eval_frontier=real_eval_frontier,
        training_budget=training_budget,
        fallback_model_dim=args.model_dim,
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
        "schema_version": 23,
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
            "requested_max_tokens": int(requested_max_tokens),
            "effective_max_tokens": int(effective_max_tokens),
            "estimated_training_tokens_budget": int(training_budget["estimated_train_tokens"]),
            "derived_budget_tokens_from_fraction": None if derived_budget_tokens is None else int(derived_budget_tokens),
            "max_tokens_fraction_of_training_budget": float(args.max_tokens_fraction_of_training_budget),
            "profiled_fraction_of_training_budget": float(
                int(sequential_tokens.size) / max(int(training_budget["estimated_train_tokens"]), 1)
            ),
        },
        "cache_summary": cache_summary,
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
        "lexical_entropy_profile": lexical_profile,
        "unconstrained_entropy_collapse_profile": unconstrained_collapse,
        "hashed_lexical_collision_profile": hashed_lexical_profile,
        "higher_order_lexical_retention_profile": higher_order_retention,
        "recent_copy_window_profile": recent_copy,
        "proxy_calibration_profile": proxy_calibration,
        "lexical_control_profile": lexical_control,
        "early_budget_coverage_profile": early_budget_coverage,
        "train_val_transfer_profile": train_val_transfer,
        "confidence_route_budget_profile": confidence_route_budget,
        "route_calibration_selectivity_profile": route_calibration,
        "recurrence_burst_profile": recurrence_profile,
        "recurrence_burst_profile_by_frequency": recurrence_by_bucket,
        "top_bigrams": top_bigrams_rows,
        "transition_geometry": transition_geometry,
        "spectral_eigenbases": spectral_eigenbases,
        "lagged_spectral_eigenbases": lagged_spectral_eigenbases,
        "spectral_basis_recommendations": spectral_recommendation,
        "minimal_causal_state_profile": minimal_causal_state,
        "minimal_causal_state_recommendations": minimal_causal_state_reco,
        "future_signature_profile": future_signature,
        "causal_state_reconstruction_profile": causal_state_reconstruction,
        "state_transition_determinism_profile": state_transition_determinism,
        "state_entropy_floor_profile": state_entropy_floor,
        "causal_state_decodability_profile": causal_state_decodability,
        "causal_state_transition_learnability_profile": causal_state_transition_learnability,
        "causal_state_multi_horizon_sufficiency_profile": causal_state_multi_horizon_sufficiency,
        "causal_state_merge_error_profile": causal_state_merge_error,
        "causal_state_residual_geometry_profile": causal_state_residual_geometry,
        "strict_online_state_eval": strict_online_state_eval,
        "strict_online_state_eval_profile": strict_online_state_eval,
        "past_future_cca_profile": past_future_cca,
        "past_future_cca_recommendations": past_future_cca_reco,
        "oracle_backoff_profile": oracle_backoff,
        "ppm_oracle_profile": ppm_oracle,
        "ppm_oracle_recommendations": ppm_oracle_reco,
        "tensor_network_state_frontier": tensor_network_state_frontier,
        "tensor_network_state_frontier_profile": tensor_network_state_frontier,
        "candidate_state_frontier": candidate_state_frontier,
        "predictive_state_compression": predictive_state_compression,
        "predictive_state_recommendations": predictive_state_recommendation,
        "predictive_state_transition_profile": predictive_state_transition,
        "predictive_state_transition_recommendations": predictive_state_transition_recommendation,
        "predictive_state_transfer_spectrum": predictive_state_transfer,
        "dataset_world_model": dataset_world_model,
        "dataset_world_model_recommendations": dataset_world_model_reco,
        "regime_conditioned_bpb": regime_conditioned_bpb,
        "order1_proxy_peak_potential": order1_proxy_peak_potential,
        "next_token_loss_decomposition": next_token_loss_decomposition,
        "posterior_error_taxonomy": posterior_error_taxonomy,
        "residual_floor_dashboard": residual_floor_dashboard,
        "eval_length_candidates": eval_candidates,
        "eval_budget_frontier": eval_budget_frontier,
        "real_eval_budget_frontier": real_eval_frontier,
        "training_budget_estimate": training_budget,
        "training_coverage_estimate": training_coverage,
        "focused_upgrade_analysis": focused_analysis,
        "leaderboard_consistent_analysis": leaderboard_analysis,
        "objective_lane_analysis": objective_analysis,
        "realized_run_correlation_analysis": realized_run_analysis,
        "artifact_byte_attribution": artifact_byte_attribution,
        "loss_geometry_surface": loss_shape,
        "run_log_frontier": {
            "anchor_run": None
            if anchor_run is None
            else {
                "path": anchor_run.path,
                "run_id": anchor_run.run_id,
                "metrics_path": anchor_run.metrics_path,
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
                    "run_id": item.run_id,
                    "metrics_path": item.metrics_path,
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
        "competition_transformer_recommendation": competition_transformer,
        "recipe_frontier": recipes,
        "kernel_efficiency": kernel_efficiency,
        "shard_profile": {
            "aggregate": shard_aggregate,
            "cluster_conditioning": {
                "aggregate": shard_clustering["aggregate"],
                "nearest_pairs": shard_clustering["nearest_pairs"],
                "farthest_pairs": shard_clustering["farthest_pairs"],
                "cluster_sweeps": shard_clustering["cluster_sweeps"],
            },
            "phase_drift": {
                "aggregate": shard_phase["aggregate"],
                "summaries": shard_phase["summaries"] if args.include_shard_summaries else [],
            },
            "summaries": shard_summary_rows if args.include_shard_summaries else [],
            "cluster_conditioning_summaries": shard_clustering["summaries"] if args.include_shard_summaries else [],
        },
        "tokenizer_profile": {
            "bytes_per_token": bytes_per_token,
            "candidate_merges": tokenizer_candidates,
            "budget_analysis": tokenizer_budget,
        },
        "tokenizer_research": tokenizer_research,
        "recommendations": summarize_recommendations(
            lag_stats,
            reuse_stats,
            entropy_bits,
            bytes_per_token,
            shard_aggregate=shard_aggregate,
            lexical_profile=lexical_profile,
            hashed_lexical_profile=hashed_lexical_profile,
            higher_order_retention=higher_order_retention,
            shard_phase_aggregate=shard_phase["aggregate"],
            eval_candidates=eval_candidates,
            training_budget=training_budget,
            training_coverage=training_coverage,
            loss_shape=loss_shape,
            recent_copy_profile=recent_copy,
            lexical_control=lexical_control,
            proxy_calibration=proxy_calibration,
            early_budget_coverage=early_budget_coverage,
            train_val_transfer=train_val_transfer,
            confidence_route_budget=confidence_route_budget,
            route_calibration=route_calibration,
            observed_frontier=observed_frontier,
            leaderboard_frontier=leaderboard_analysis,
            objective_lanes=objective_analysis,
            realized_run_analysis=realized_run_analysis,
            artifact_byte_attribution=artifact_byte_attribution,
            tokenizer_merge_summary=tokenizer_merge_summary,
            tokenizer_research=tokenizer_research,
            spectral_basis_recommendation=spectral_recommendation,
            predictive_state_compression=predictive_state_compression,
            predictive_state_recommendation=predictive_state_recommendation,
            predictive_state_transition=predictive_state_transition,
            oracle_backoff=oracle_backoff,
            past_future_cca=past_future_cca,
            past_future_cca_recommendation=past_future_cca_reco,
            predictive_state_transfer=predictive_state_transfer,
            ppm_oracle=ppm_oracle,
            ppm_oracle_recommendation=ppm_oracle_reco,
            minimal_causal_state=minimal_causal_state,
            minimal_causal_state_recommendation=minimal_causal_state_reco,
            causal_state_reconstruction=causal_state_reconstruction,
            state_transition_determinism=state_transition_determinism,
            state_entropy_floor=state_entropy_floor,
            strict_online_state_eval=strict_online_state_eval,
            tensor_network_state_frontier=tensor_network_state_frontier,
            candidate_state_frontier=candidate_state_frontier,
            dataset_world_model=dataset_world_model,
            dataset_world_model_recommendation=dataset_world_model_reco,
            order1_proxy_peak_potential=order1_proxy_peak_potential,
            next_token_loss_decomposition=next_token_loss_decomposition,
            residual_floor_dashboard=residual_floor_dashboard,
        ),
        "limitations": [
            "Binary shards do not preserve explicit document boundaries, so document-level TTT fit is inferred indirectly from stream statistics.",
            "Mutual information is estimated on the sampled token stream, not the full corpus.",
            "Shard-level drift is estimated from evenly spaced token samples within each shard rather than exhaustive full-shard histograms.",
            "Lexical entropy gains come from empirical one-token contexts on the sampled stream; they measure available local structure, not guaranteed model realization.",
            "Higher-order lexical context statistics are exact for the requested low-order contexts on the sampled stream, but they still summarize stream structure rather than full-model realizability.",
            "Hashed lexical collision sweeps model context-table aliasing only; they do not include embedding optimization effects, training dynamics, or kernel overhead.",
            "Higher-order lexical retention profiling picks balanced hashed sweet spots around a target order, but the collision caps and retention floors are heuristic filters rather than guaranteed training optima.",
            "Recent-copy window analysis measures exact repeated local context matches in the sampled stream; it is a proxy for copy-style experts, not a full learned retrieval model.",
            "Proxy calibration and lexical-control sweeps use empirical n-gram predictors as stand-ins for learned lexical experts, so they estimate control stability rather than guaranteeing model behavior.",
            "Early budget coverage is measured on the profiled sequential prefix only; if max_tokens is small, later budget points may clip to the same observed token window.",
            "Train/val transfer uses exact or hashed context overlap between sampled train and validation prefixes; it estimates lexical carryover, not full model generalization.",
            "Confidence-route budget profile treats lexical confidence as a routing proxy and reports an oracle upper bound for deferred low-confidence tokens; it is not a realized model score.",
            "Route-calibration selectivity profiling scores low-entropy, low-false-trust routing frontiers from lexical proxies; it helps set routing priors but does not guarantee realized gate entropy in the trained model.",
            "Oracle backoff profiling uses full sampled-prefix context counts as an optimistic upper bound; it is intentionally stronger than a strict online evaluator and should be read as available local-memory signal rather than a ready-to-submit method.",
            "Predictive-state transition profiling fits linear dynamics in a compressed posterior space; a good transition R^2 suggests a tiny recurrent bias module is plausible, not that the full future law is linear.",
            "Past/future CCA uses hashed sketch features over bounded windows rather than an exact Hankel object; it is the practical predictive-state factorization for this profiler, not a full spectral PSR implementation.",
            "Minimal causal-state profiling fits held-out next-token loss directly from compressed past/future state ranks; it is the most objective-aligned state frontier in the profiler, but still a linear probe over offline sketches rather than the final deployed state module.",
            "Future-signature profiling builds epsilon-machine-style future fingerprints from bounded future sketches and entropies; it is the right approximation for causal-state merging, not a full infinite-horizon future law.",
            "Causal-state reconstruction clusters prefixes by future signatures and reads out next-token distributions from those states; it is an approximate predictive automaton over sampled prefixes, not an exact recovered epsilon-machine.",
            "State-transition determinism is measured on reconstructed causal states from sampled contiguous prefixes; it estimates how machine-like the recovered states are, but it is still a finite-sample transition proxy.",
            "State-entropy floors report how deterministic next-token and short-horizon futures become once the reconstructed state is known; they are the best current sufficiency proxy in this profiler, but they still sit on bounded-horizon signatures.",
            "Strict online state evaluation reuses the offline-fit minimal causal-state readout but scores it in a past-only streaming setup; this is the correct deployment-style evaluator for the current state family, but it is still a probe rather than a jointly trained recurrent module.",
            "Predictive-state transfer spectrum clusters CCA states into an MSM-style transition model; this is the right dynamical approximation for sweep guidance, not a formal guarantee of metastable process structure.",
            "Tensor-network state frontier compresses the linear minimal-state readout with a TT-SVD-style factorization; it is a bond-dimension frontier over the current state head, not a full tensor-network language model.",
            "Dataset world-model profiling is a chunk-level coarse-graining over predictive-state embeddings; it is useful as a universal offline prior, not as proof that the corpus is generated by a single identifiable latent mechanism.",
            "The full world-model regime assignment is clairvoyant because it is fit offline over long chunks; only prefix-causal regime classifiers and distilled priors should be promoted into the trainer.",
            "Regime-conditioned BPB and the loss decomposition are component floors, not additive guarantees; local lexical, copy, and regime gains overlap.",
            "Order-1 proxy peak-potential treats the exact one-token posterior as the baseline object and reports shortest-path component drops from there; those drops overlap and should not be summed.",
            "Posterior error taxonomy is currently built from an exact order-1 proxy posterior rather than the trained model posterior; it should be read as a routing/debugging lens, not a faithful transformer diagnosis.",
            "PPM-style oracle profiling uses bounded-order Witten-Bell backoff on the sampled stream; it is a stronger causal baseline than raw n-gram lookup, but still an offline ceiling rather than a submission-time algorithm.",
            "Candidate-state frontier compares trainable and oracle state mechanisms on held-out proxy BPB, bytes, and causal-evaluation status; treat it as sweep-ordering guidance rather than a leaderboard guarantee.",
            "Objective-lane analysis turns measured dataset/runtime signals into heuristic priorities for current parameter-golf lanes; it is meant for sweep ordering, not as a guaranteed predictor of leaderboard rank.",
            "Realized run correlation analysis is only as strong as the supplied run logs; sparse sweeps and missing sidecar metrics can make component deltas unstable.",
            "Artifact byte attribution mixes observed submission totals with proxy allocation inside the model artifact unless a completed run exposes structured artifact metrics.",
            "Kernel efficiency benchmarks are synthetic random-token timing estimates on the current hardware; they predict relative cost, not final BPB.",
        ],
    }

    output = json.dumps(profile, indent=2, sort_keys=True)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_spectral_arrays = {
            **spectral_eigenbasis_arrays,
            **lagged_spectral_eigenbasis_arrays,
            **past_future_cca_arrays,
            **predictive_state_arrays,
            **minimal_causal_state_arrays,
            **causal_state_arrays,
            **causal_state_decodability_arrays,
            **causal_state_transition_learnability_arrays,
            **causal_state_residual_geometry_arrays,
            **dataset_world_model_arrays,
            **regime_conditioned_bpb_arrays,
        }
        if all_spectral_arrays:
            sidecar_path = out_path.with_name(f"{out_path.stem}_spectral_eigenbases.npz")
            np.savez_compressed(sidecar_path, **all_spectral_arrays)
            profile["spectral_eigenbases"]["sidecar_npz"] = str(sidecar_path)
            output = json.dumps(profile, indent=2, sort_keys=True)
        world_model_export = None
        if isinstance(regime_conditioned_bpb, dict) and regime_conditioned_bpb.get("available"):
            world_model_export = regime_conditioned_bpb.get("trainer_export")
        if isinstance(world_model_export, dict):
            export_path = out_path.with_name(f"{out_path.stem}_world_model_trainer_export.json")
            export_path.write_text(json.dumps(world_model_export, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            profile["world_model_trainer_export_path"] = str(export_path)
            output = json.dumps(profile, indent=2, sort_keys=True)
        out_path.write_text(output + "\n", encoding="utf-8")
        print(out_path)
    else:
        print(output)


if __name__ == "__main__":
    main()
