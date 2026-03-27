#!/usr/bin/env python3
"""
Causal-focused dataset profiler for parameter-golf tuning.

This is a derived profiler surface built from `profile_training_dataset.py`,
but it intentionally omits the lexical / bigram / hashed-context tuning
surfaces and disables n-gram oracle ceilings by default (using the sentinel
order `0`) so runs stay faster
and the output is centered on:

- next-token causal state structure
- predictive-state compression
- world-model / regime structure
- spectral fixed-map candidates
- training/eval budget implications

Use this when the question is "what should the backbone or causal heads learn?"
rather than "how should we tune lexical shortcut modules?"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import profile_training_dataset as base


SCHEMA_VERSION = 4
REPO_ROOT = Path(__file__).resolve().parents[1]

# Old cache pickles from `profile_training_dataset.py` often reference these
# dataclasses through `__main__` because that script is commonly executed
# directly. Register them here so the causal profiler can reuse the same cache.
for _name in ("LagMetric", "RunLogSummary", "MissingRunLog"):
    if hasattr(base, _name):
        setattr(sys.modules[__name__], _name, getattr(base, _name))


def _resolve_repo_relative_path(text: str) -> str:
    path = Path(text)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _resolve_repo_relative_glob(text: str) -> str:
    if not text:
        return text
    path = Path(text)
    if path.is_absolute():
        return str(path)
    if list(Path().glob(text)):
        return text
    return str(REPO_ROOT / text)


def _bundle_specs(args: argparse.Namespace, sequential_cache: dict[str, object], token_counts: np.ndarray, base_bytes, tokenizer_meta) -> dict[str, dict[str, object]]:
    parsed_eval_lengths = base.parse_int_list(args.eval_lengths)
    parsed_cca_ranks = base.parse_int_list(args.cca_ranks)
    parsed_causal_state_ranks = (
        base.parse_int_list(args.causal_state_ranks) if args.causal_state_ranks.strip() else parsed_cca_ranks
    )
    return {
        "basic_signal": {
            "sequential_tokens_path": sequential_cache["path"],
            "vocab_size": args.vocab_size,
            "top_k": args.top_k,
            "token_counts": token_counts,
            "lags": base.parse_int_list(args.lags),
            "reuse_windows": base.parse_int_list(args.reuse_windows),
            "eval_lengths": parsed_eval_lengths,
        },
        "state": {
            "sequential_tokens_path": sequential_cache["path"],
            "vocab_size": args.vocab_size,
            "token_counts": token_counts,
            "base_bytes": base_bytes,
            "tokenizer_meta": tokenizer_meta,
            "spectral_eigen_rank": args.spectral_eigen_rank,
            "spectral_lags": base.parse_int_list(args.spectral_lags),
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
            "causal_machine_state_counts": base.parse_int_list(args.causal_machine_state_counts),
            "causal_machine_kmeans_iters": args.causal_machine_kmeans_iters,
            "causal_machine_future_horizons": base.parse_int_list(args.causal_machine_future_horizons),
            "strict_online_max_eval_tokens": args.strict_online_max_eval_tokens,
            "strict_online_horizons": base.parse_int_list(args.strict_online_horizons),
            "tensor_bond_dims": base.parse_int_list(args.tensor_bond_dims),
            "transfer_taus": base.parse_int_list(args.transfer_taus),
            "transfer_clusters": args.transfer_clusters,
            "transfer_kmeans_iters": args.transfer_kmeans_iters,
            "predictive_state_order": args.predictive_state_order,
            "predictive_state_ranks": base.parse_int_list(args.predictive_state_ranks),
            "predictive_state_max_prefixes": args.predictive_state_max_prefixes,
            "oracle_backoff_orders": base.parse_int_list(args.oracle_backoff_orders),
            "oracle_backoff_max_eval_tokens": args.oracle_backoff_max_eval_tokens,
            "ppm_orders": base.parse_int_list(args.ppm_orders),
            "ppm_max_eval_tokens": args.ppm_max_eval_tokens,
            "world_model_chunk_tokens": args.world_model_chunk_tokens,
            "world_model_chunk_stride": args.world_model_chunk_stride,
            "world_model_prefix_tokens": args.world_model_prefix_tokens,
            "world_model_regime_counts": base.parse_int_list(args.world_model_regime_counts),
            "world_model_kmeans_iters": args.world_model_kmeans_iters,
            "world_model_top_tokens": args.world_model_top_tokens,
        },
    }


def _run_bundles(bundle_specs: dict[str, dict[str, object]], cache_dir: Path, resume_cache: bool, profile_workers: int) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    cache_summary = {
        "cache_dir": str(cache_dir),
        "resume_cache": bool(resume_cache),
        "profile_workers_requested": int(profile_workers),
        "bundle_hits": [],
        "bundle_misses": [],
    }
    bundle_results: dict[str, dict[str, object]] = {}
    pending_futures: dict[object, tuple[str, str]] = {}
    executor: ProcessPoolExecutor | None = None
    if profile_workers > 1:
        executor = ProcessPoolExecutor(max_workers=profile_workers)
    try:
        for bundle_name, payload in bundle_specs.items():
            cache_key = base.build_cache_key(f"profile_bundle_{bundle_name}", payload)
            cached = base.load_cached_stage(cache_dir, f"profile_bundle_{bundle_name}", cache_key) if resume_cache else None
            if isinstance(cached, dict):
                bundle_results[bundle_name] = cached
                cache_summary["bundle_hits"].append(bundle_name)
                continue
            cache_summary["bundle_misses"].append(bundle_name)
            if executor is None:
                result = base._run_profile_bundle(bundle_name, payload)
                base.save_cached_stage(cache_dir, f"profile_bundle_{bundle_name}", cache_key, result)
                bundle_results[bundle_name] = result
            else:
                pending_futures[executor.submit(base._run_profile_bundle, bundle_name, payload)] = (bundle_name, cache_key)
        for future in as_completed(list(pending_futures.keys())):
            bundle_name, cache_key = pending_futures[future]
            result = future.result()
            base.save_cached_stage(cache_dir, f"profile_bundle_{bundle_name}", cache_key, result)
            bundle_results[bundle_name] = result
    finally:
        if executor is not None:
            executor.shutdown()
    return bundle_results, cache_summary


def _normalized_mi_map(lag_stats: list[object]) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in lag_stats:
        out[int(row.lag)] = float(row.normalized_mi)
    return out


def _safe_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def _bounded_fractions(text: str, *, extra: list[float] | None = None) -> list[float]:
    vals = [v for v in _safe_float_list(text) if 0.0 < float(v) <= 1.0]
    if extra:
        vals.extend(float(v) for v in extra if 0.0 < float(v) <= 1.0)
    dedup = sorted({round(float(v), 6) for v in vals})
    return dedup


def _safe_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(float(part)))
    return values


def build_causal_capacity_inference(
    lag_stats: list[object],
    minimal_causal_state_reco: dict[str, object] | None,
    predictive_state_recommendation: dict[str, object] | None,
    training_budget: dict[str, object],
) -> dict[str, object]:
    lag_map = _normalized_mi_map(lag_stats)
    local_signal = float(lag_map.get(1, 0.0) + lag_map.get(2, 0.0))
    medium_signal = float(lag_map.get(4, 0.0) + lag_map.get(8, 0.0))
    long_signal = float(sum(v for k, v in lag_map.items() if k >= 16))
    local_dominance = float(local_signal / max(local_signal + medium_signal + long_signal, 1e-9))
    rank4_state = None
    if isinstance(minimal_causal_state_reco, dict):
        rank4_state = ((minimal_causal_state_reco.get("recommended_state") or {}).get("rank"))
    predictive_rank = None
    if isinstance(predictive_state_recommendation, dict):
        predictive_rank = ((predictive_state_recommendation.get("recommended_state") or {}).get("rank"))
    estimated_steps = int(training_budget.get("estimated_steps", 0))
    depth_pressure = "low"
    if medium_signal >= 0.10 or long_signal >= 0.03:
        depth_pressure = "medium"
    if medium_signal >= 0.20 or long_signal >= 0.08:
        depth_pressure = "high"
    width_pressure = "low"
    if isinstance(predictive_rank, int) and predictive_rank >= 128:
        width_pressure = "medium"
    if isinstance(predictive_rank, int) and predictive_rank >= 256:
        width_pressure = "high"
    preferred_backbone_band = {
        "num_layers": [8, 10] if depth_pressure == "low" else [10, 12],
        "mlp_hidden": [896, 1024] if width_pressure == "low" else [1024, 1280],
    }
    return {
        "depth_pressure": depth_pressure,
        "width_pressure": width_pressure,
        "local_signal_mass": local_signal,
        "medium_signal_mass": medium_signal,
        "long_signal_mass": long_signal,
        "local_dominance": local_dominance,
        "minimal_causal_state_rank": rank4_state,
        "predictive_state_rank": predictive_rank,
        "estimated_train_steps": estimated_steps,
        "preferred_backbone_band": preferred_backbone_band,
        "recommendation": (
            "Prefer small causal/state heads over backbone growth."
            if depth_pressure == "low" and width_pressure == "low"
            else "Backbone growth is plausible, but only after cheap causal heads are exhausted."
        ),
    }


def build_reveal_sweep_analysis(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    base_bytes: np.ndarray | None,
    training_budget: dict[str, object],
    expected_reveal_fraction: float,
    reveal_fractions: list[float],
    lags: list[int],
    cca_past_window: int,
    cca_future_window: int,
    cca_sketch_dim: int,
    cca_regularization: float,
    causal_state_ranks: list[int],
    causal_state_holdout_fraction: float,
    causal_state_ridge: float,
    causal_state_near_best_bpb_tol: float,
    causal_state_near_best_bits_tol: float,
    predictive_state_order: int,
    predictive_state_ranks: list[int],
    reveal_max_prefixes: int,
) -> dict[str, object]:
    total_tokens = int(tokens.size)
    if total_tokens <= 0:
        return {"available": False, "reason": "no_tokens"}
    rows: list[dict[str, object]] = []
    required_min_tokens = max(
        int(cca_past_window + cca_future_window + 32),
        int(predictive_state_order + 32),
    )
    for frac in reveal_fractions:
        prefix_tokens = max(required_min_tokens, min(total_tokens, int(round(total_tokens * float(frac)))))
        prefix = np.ascontiguousarray(tokens[:prefix_tokens])
        lag_stats = base.lag_metrics(prefix, vocab_size, lags)
        minimal_state, _ = base.minimal_causal_state_profile(
            prefix,
            vocab_size,
            base_bytes=base_bytes,
            past_window=int(cca_past_window),
            future_window=int(cca_future_window),
            sketch_dim=int(cca_sketch_dim),
            ranks=causal_state_ranks,
            max_prefixes=int(reveal_max_prefixes),
            regularization=float(cca_regularization),
            holdout_fraction=float(causal_state_holdout_fraction),
            ridge=float(causal_state_ridge),
            near_best_bpb_tol=float(causal_state_near_best_bpb_tol),
            near_best_bits_tol=float(causal_state_near_best_bits_tol),
        )
        minimal_reco = base.minimal_causal_state_recommendations(minimal_state)
        predictive_state, _ = base.predictive_state_compression_profile(
            prefix,
            vocab_size,
            order=int(predictive_state_order),
            ranks=predictive_state_ranks,
            max_prefixes=int(reveal_max_prefixes),
        )
        predictive_reco = base.predictive_state_recommendations(predictive_state)
        capacity = build_causal_capacity_inference(
            lag_stats=lag_stats,
            minimal_causal_state_reco=minimal_reco,
            predictive_state_recommendation=predictive_reco,
            training_budget=training_budget,
        )
        lag_map = _normalized_mi_map(lag_stats)
        minimal_row = minimal_reco.get("recommended_state") if isinstance(minimal_reco, dict) else None
        predictive_row = predictive_reco.get("recommended_state") if isinstance(predictive_reco, dict) else None
        rows.append(
            {
                "reveal_fraction": float(frac),
                "prefix_tokens_profiled": int(prefix_tokens),
                "lag1_normalized_mi": float(lag_map.get(1, 0.0)),
                "lag2_normalized_mi": float(lag_map.get(2, 0.0)),
                "lag4_normalized_mi": float(lag_map.get(4, 0.0)),
                "lag8_normalized_mi": float(lag_map.get(8, 0.0)),
                "local_signal_mass": float(capacity["local_signal_mass"]),
                "medium_signal_mass": float(capacity["medium_signal_mass"]),
                "long_signal_mass": float(capacity["long_signal_mass"]),
                "preferred_backbone_band": capacity["preferred_backbone_band"],
                "minimal_causal_state_rank": None if not isinstance(minimal_row, dict) else int(minimal_row.get("rank", 0)),
                "minimal_causal_state_bpb": None
                if not isinstance(minimal_row, dict) or minimal_row.get("heldout_bits_per_byte") is None
                else float(minimal_row.get("heldout_bits_per_byte")),
                "predictive_state_rank": None if not isinstance(predictive_row, dict) else int(predictive_row.get("rank", 0)),
                "predictive_state_gap_bits": None
                if not isinstance(predictive_row, dict)
                else float(predictive_row.get("cross_entropy_gap_bits", 0.0)),
            }
        )

    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        cur = rows[idx]
        prev_min = prev.get("minimal_causal_state_bpb")
        cur_min = cur.get("minimal_causal_state_bpb")
        prev_pred = prev.get("predictive_state_gap_bits")
        cur_pred = cur.get("predictive_state_gap_bits")
        cur["delta_minimal_state_bpb_vs_prev"] = (
            None if prev_min is None or cur_min is None else float(prev_min) - float(cur_min)
        )
        cur["delta_predictive_gap_bits_vs_prev"] = (
            None if prev_pred is None or cur_pred is None else float(prev_pred) - float(cur_pred)
        )

    expected_row = min(rows, key=lambda row: abs(float(row["reveal_fraction"]) - float(expected_reveal_fraction)))
    full_row = max(rows, key=lambda row: float(row["reveal_fraction"]))
    smallest_near_full = None
    full_min = full_row.get("minimal_causal_state_bpb")
    full_pred = full_row.get("predictive_state_gap_bits")
    if full_min is not None or full_pred is not None:
        candidates = []
        for row in rows:
            min_ok = (
                full_min is None
                or row.get("minimal_causal_state_bpb") is None
                or float(row["minimal_causal_state_bpb"]) <= float(full_min) + 0.005
            )
            pred_ok = (
                full_pred is None
                or row.get("predictive_state_gap_bits") is None
                or float(row["predictive_state_gap_bits"]) <= float(full_pred) + 0.10
            )
            if min_ok and pred_ok:
                candidates.append(row)
        if candidates:
            smallest_near_full = min(candidates, key=lambda row: float(row["reveal_fraction"]))

    expected_to_full_min_bpb = (
        None
        if expected_row.get("minimal_causal_state_bpb") is None or full_min is None
        else float(expected_row["minimal_causal_state_bpb"]) - float(full_min)
    )
    expected_to_full_pred_gap = (
        None
        if expected_row.get("predictive_state_gap_bits") is None or full_pred is None
        else float(expected_row["predictive_state_gap_bits"]) - float(full_pred)
    )
    prev_before_expected = None
    for row in rows:
        if float(row["reveal_fraction"]) < float(expected_row["reveal_fraction"]):
            prev_before_expected = row
    lower_to_expected_min_bpb = (
        None
        if prev_before_expected is None or prev_before_expected.get("minimal_causal_state_bpb") is None or expected_row.get("minimal_causal_state_bpb") is None
        else float(prev_before_expected["minimal_causal_state_bpb"]) - float(expected_row["minimal_causal_state_bpb"])
    )
    lower_to_expected_pred_gap = (
        None
        if prev_before_expected is None or prev_before_expected.get("predictive_state_gap_bits") is None or expected_row.get("predictive_state_gap_bits") is None
        else float(prev_before_expected["predictive_state_gap_bits"]) - float(expected_row["predictive_state_gap_bits"])
    )

    if expected_to_full_min_bpb is not None and expected_to_full_pred_gap is not None:
        if expected_to_full_min_bpb <= 0.003 and expected_to_full_pred_gap <= 0.05:
            stream_recommendation = "Expected competition reveal is already near the full-profile causal floor; chasing more reveal looks low ROI."
        elif expected_to_full_min_bpb > 0.01 or expected_to_full_pred_gap > 0.20:
            stream_recommendation = "Full-stream structure still improves causal-head proxies materially; seeing more of the stream could be worthwhile if runtime permits."
        else:
            stream_recommendation = "Expected reveal leaves some causal headroom to the full stream, but the gain looks moderate rather than decisive."
    else:
        stream_recommendation = "Reveal sweep could not estimate expected-vs-full causal headroom cleanly."

    if lower_to_expected_min_bpb is not None and lower_to_expected_pred_gap is not None:
        if lower_to_expected_min_bpb <= 0.003 and lower_to_expected_pred_gap <= 0.05:
            lower_reveal_note = "A slightly smaller reveal than the current competition baseline may still preserve most causal-head quality if it buys more optimization steps."
        else:
            lower_reveal_note = "Dropping below the current competition reveal baseline appears to cost meaningful causal-head quality."
    else:
        lower_reveal_note = None

    return {
        "available": True,
        "expected_reveal_fraction": float(expected_reveal_fraction),
        "rows": rows,
        "expected_reveal_row": expected_row,
        "full_reveal_row": full_row,
        "smallest_fraction_near_full": smallest_near_full,
        "expected_to_full_minimal_state_bpb_gap": expected_to_full_min_bpb,
        "expected_to_full_predictive_gap_bits": expected_to_full_pred_gap,
        "lower_to_expected_minimal_state_bpb_gain": lower_to_expected_min_bpb,
        "lower_to_expected_predictive_gap_bits_gain": lower_to_expected_pred_gap,
        "stream_recommendation": stream_recommendation,
        "lower_reveal_note": lower_reveal_note,
    }


def build_token_seen_ladder_analysis(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    base_bytes: np.ndarray | None,
    training_budget: dict[str, object],
    total_train_tokens_available: int,
    token_seen_points: list[int],
    lags: list[int],
    cca_past_window: int,
    cca_future_window: int,
    cca_sketch_dim: int,
    cca_regularization: float,
    causal_state_ranks: list[int],
    causal_state_holdout_fraction: float,
    causal_state_ridge: float,
    causal_state_near_best_bpb_tol: float,
    causal_state_near_best_bits_tol: float,
    predictive_state_order: int,
    predictive_state_ranks: list[int],
    reveal_max_prefixes: int,
) -> dict[str, object]:
    total_profiled_tokens = int(tokens.size)
    if total_profiled_tokens <= 0:
        return {"available": False, "reason": "no_tokens"}
    estimated_train_tokens = int(training_budget.get("estimated_train_tokens", 0))
    train_batch_tokens = int(training_budget.get("train_batch_tokens", 0))
    required_min_tokens = max(
        int(cca_past_window + cca_future_window + 32),
        int(predictive_state_order + 32),
    )
    rows: list[dict[str, object]] = []
    for point in sorted({int(p) for p in token_seen_points if int(p) > 0}):
        prefix_tokens = max(required_min_tokens, min(total_profiled_tokens, int(point)))
        prefix = np.ascontiguousarray(tokens[:prefix_tokens])
        lag_stats = base.lag_metrics(prefix, vocab_size, lags)
        minimal_state, _ = base.minimal_causal_state_profile(
            prefix,
            vocab_size,
            base_bytes=base_bytes,
            past_window=int(cca_past_window),
            future_window=int(cca_future_window),
            sketch_dim=int(cca_sketch_dim),
            ranks=causal_state_ranks,
            max_prefixes=int(reveal_max_prefixes),
            regularization=float(cca_regularization),
            holdout_fraction=float(causal_state_holdout_fraction),
            ridge=float(causal_state_ridge),
            near_best_bpb_tol=float(causal_state_near_best_bpb_tol),
            near_best_bits_tol=float(causal_state_near_best_bits_tol),
        )
        minimal_reco = base.minimal_causal_state_recommendations(minimal_state)
        predictive_state, _ = base.predictive_state_compression_profile(
            prefix,
            vocab_size,
            order=int(predictive_state_order),
            ranks=predictive_state_ranks,
            max_prefixes=int(reveal_max_prefixes),
        )
        predictive_reco = base.predictive_state_recommendations(predictive_state)
        capacity = build_causal_capacity_inference(
            lag_stats=lag_stats,
            minimal_causal_state_reco=minimal_reco,
            predictive_state_recommendation=predictive_reco,
            training_budget=training_budget,
        )
        lag_map = _normalized_mi_map(lag_stats)
        minimal_row = minimal_reco.get("recommended_state") if isinstance(minimal_reco, dict) else None
        predictive_row = predictive_reco.get("recommended_state") if isinstance(predictive_reco, dict) else None
        approx_steps = None if train_batch_tokens <= 0 else float(point) / float(train_batch_tokens)
        dataset_reveal_fraction = float(point) / max(float(total_train_tokens_available), 1.0)
        run_budget_fraction = None if estimated_train_tokens <= 0 else float(point) / float(estimated_train_tokens)
        rows.append(
            {
                "tokens_seen": int(point),
                "prefix_tokens_profiled": int(prefix_tokens),
                "dataset_reveal_fraction": dataset_reveal_fraction,
                "run_budget_fraction": run_budget_fraction,
                "approx_train_steps": approx_steps,
                "lag1_normalized_mi": float(lag_map.get(1, 0.0)),
                "lag2_normalized_mi": float(lag_map.get(2, 0.0)),
                "lag4_normalized_mi": float(lag_map.get(4, 0.0)),
                "lag8_normalized_mi": float(lag_map.get(8, 0.0)),
                "preferred_backbone_band": capacity["preferred_backbone_band"],
                "minimal_causal_state_rank": None if not isinstance(minimal_row, dict) else int(minimal_row.get("rank", 0)),
                "minimal_causal_state_bpb": None
                if not isinstance(minimal_row, dict) or minimal_row.get("heldout_bits_per_byte") is None
                else float(minimal_row.get("heldout_bits_per_byte")),
                "predictive_state_rank": None if not isinstance(predictive_row, dict) else int(predictive_row.get("rank", 0)),
                "predictive_state_gap_bits": None
                if not isinstance(predictive_row, dict)
                else float(predictive_row.get("cross_entropy_gap_bits", 0.0)),
            }
        )

    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        cur = rows[idx]
        prev_min = prev.get("minimal_causal_state_bpb")
        cur_min = cur.get("minimal_causal_state_bpb")
        prev_pred = prev.get("predictive_state_gap_bits")
        cur_pred = cur.get("predictive_state_gap_bits")
        cur["delta_minimal_state_bpb_vs_prev"] = (
            None if prev_min is None or cur_min is None else float(prev_min) - float(cur_min)
        )
        cur["delta_predictive_gap_bits_vs_prev"] = (
            None if prev_pred is None or cur_pred is None else float(prev_pred) - float(cur_pred)
        )

    return {
        "available": True,
        "rows": rows,
        "recommended_early_view": (
            "Early log-scale token ladder shows where causal-head structure stabilizes before the competition reveal point."
        ),
    }


def build_reveal_conditioned_student_recipe_table(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    base_bytes: np.ndarray | None,
    training_budget: dict[str, object],
    tokens_seen_target: int,
    reveal_fractions: list[float],
    past_window: int,
    sketch_dim: int,
    holdout_fraction: float,
    init_hidden_ranks: list[int],
    init_seeds: list[int],
    init_nonlinearities: list[str],
    jacobian_max_samples: int,
    teacher_fit_max_prefixes: int,
    teacher_fit_top_inits: int,
    teacher_fit_steps: list[int],
    teacher_fit_learning_rates: list[float],
    teacher_fit_weight_decays: list[float],
    teacher_fit_state_loss_coeffs: list[float],
    teacher_fit_teacher_loss_coeffs: list[float],
    profile_cache_dir: Path,
) -> dict[str, object]:
    total_tokens = int(tokens.size)
    if total_tokens <= 0:
        return {"available": False, "reason": "no_tokens"}
    effective_target = int(tokens_seen_target) if int(tokens_seen_target) > 0 else int(training_budget.get("estimated_train_tokens", total_tokens))
    effective_target = max(1, min(effective_target, total_tokens))
    rows: list[dict[str, object]] = []
    best_row = None
    for frac in sorted({float(v) for v in reveal_fractions if 0.0 < float(v) <= 1.0}):
        prefix_tokens = max(int(past_window + 32), min(effective_target, int(round(effective_target * frac))))
        prefix = np.ascontiguousarray(tokens[:prefix_tokens])
        tmp_future_signature, tmp_future_signature_arrays = base.future_signature_profile(
            prefix,
            vocab_size=vocab_size,
            past_window=past_window,
            sketch_dim=sketch_dim,
            horizons=[1, 2, 4, 8],
            max_prefixes=max(int(teacher_fit_max_prefixes), 4096),
        )
        tmp_causal_reco, tmp_causal_arrays = base.causal_state_reconstruction_profile(
            future_signature_arrays=tmp_future_signature_arrays,
            vocab_size=vocab_size,
            base_bytes=base_bytes,
            state_counts=[128],
            holdout_fraction=holdout_fraction,
            kmeans_iters=12,
            near_best_bpb_tol=0.01,
            near_best_bits_tol=0.05,
        )
        if not bool(tmp_causal_reco.get("available")):
            rows.append(
                {
                    "reveal_fraction": float(frac),
                    "prefix_tokens_profiled": int(prefix_tokens),
                    "available": False,
                    "reason": str(tmp_causal_reco.get("reason", "causal_reconstruction_unavailable")),
                }
            )
            continue
        rand_dec, _ = random_init_state_decodability_profile(
            prefix,
            vocab_size=vocab_size,
            past_window=past_window,
            sketch_dim=sketch_dim,
            future_signature_arrays=tmp_future_signature_arrays,
            causal_state_arrays=tmp_causal_arrays,
            holdout_fraction=holdout_fraction,
            hidden_ranks=init_hidden_ranks,
            seeds=init_seeds,
            nonlinearities=init_nonlinearities,
        )
        teacher_fit, _ = student_teacher_fit_profile(
            prefix,
            vocab_size=vocab_size,
            base_bytes=base_bytes,
            past_window=past_window,
            sketch_dim=sketch_dim,
            future_signature_arrays=tmp_future_signature_arrays,
            causal_state_arrays=tmp_causal_arrays,
            holdout_fraction=holdout_fraction,
            random_init_state_decodability=rand_dec,
            max_prefixes=teacher_fit_max_prefixes,
            top_inits=teacher_fit_top_inits,
            fit_steps=teacher_fit_steps,
            learning_rates=teacher_fit_learning_rates,
            weight_decays=teacher_fit_weight_decays,
            state_loss_coeffs=teacher_fit_state_loss_coeffs,
            teacher_loss_coeffs=teacher_fit_teacher_loss_coeffs,
            training_budget=training_budget,
        )
        recipe = build_student_teacher_recipe(
            student_teacher_fit=teacher_fit,
            training_budget=training_budget,
        )
        offline_best = tmp_causal_reco.get("best_state_count_by_holdout_bpb") if isinstance(tmp_causal_reco, dict) else None
        row = {
            "reveal_fraction": float(frac),
            "prefix_tokens_profiled": int(prefix_tokens),
            "dataset_tokens_seen_target": int(effective_target),
            "available": bool(recipe.get("available")),
            "offline_causal_machine_bpb": None if not isinstance(offline_best, dict) else offline_best.get("heldout_bits_per_byte"),
            "student_recipe": recipe,
            "student_best_fit": None if not isinstance(teacher_fit, dict) else teacher_fit.get("best_by_holdout_bpb"),
        }
        rows.append(row)
        best_fit = row.get("student_best_fit")
        if isinstance(best_fit, dict) and best_fit.get("holdout_token_bits_per_byte") is not None:
            if best_row is None or float(best_fit["holdout_token_bits_per_byte"]) < float(best_row["student_best_fit"]["holdout_token_bits_per_byte"]):
                best_row = row
    return {
        "available": bool(rows),
        "tokens_seen_target": int(effective_target),
        "target_truncated_to_profiled_tokens": bool(effective_target < int(tokens_seen_target)) if int(tokens_seen_target) > 0 else False,
        "rows": rows,
        "best_reveal_recipe": best_row,
        "note": "This table fixes the tokens-seen target and compares reveal-conditioned student recipes against reveal-specific causal teachers.",
    }


def build_competition_target_recommendation(
    reveal_analysis: dict[str, object] | None,
    training_budget: dict[str, object],
    eval_candidates: list[dict[str, object]],
) -> dict[str, object]:
    expected_row = None if not isinstance(reveal_analysis, dict) else reveal_analysis.get("expected_reveal_row")
    eval_seq = max((int(row.get("seq_len", 0)) for row in eval_candidates), default=int(training_budget.get("train_seq_len", 1024)))
    out = {
        "recommended_train_seq_len": int(training_budget.get("train_seq_len", 1024)),
        "recommended_eval_seq_len": int(eval_seq),
        "recommended_num_layers_band": None,
        "recommended_mlp_hidden_band": None,
        "expected_reveal_fraction": None if not isinstance(reveal_analysis, dict) else reveal_analysis.get("expected_reveal_fraction"),
        "stream_recommendation": None if not isinstance(reveal_analysis, dict) else reveal_analysis.get("stream_recommendation"),
        "lower_reveal_note": None if not isinstance(reveal_analysis, dict) else reveal_analysis.get("lower_reveal_note"),
    }
    if isinstance(expected_row, dict):
        band = expected_row.get("preferred_backbone_band") or {}
        out["recommended_num_layers_band"] = band.get("num_layers")
        out["recommended_mlp_hidden_band"] = band.get("mlp_hidden")
        out["expected_reveal_minimal_causal_state_rank"] = expected_row.get("minimal_causal_state_rank")
        out["expected_reveal_predictive_state_rank"] = expected_row.get("predictive_state_rank")
    return out


def _nearest_tokens_seen_row(rows: list[dict[str, object]], target_tokens_seen: int) -> dict[str, object] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(int(row.get("tokens_seen", 0)) - int(target_tokens_seen)))


def build_full_stream_reference(
    *,
    lag_stats: list[object],
    capacity_inference: dict[str, object],
    minimal_causal_state_reco: dict[str, object] | None,
    predictive_state_recommendation: dict[str, object] | None,
    predictive_state_transition_recommendation: dict[str, object] | None,
    spectral_recommendation: dict[str, object] | None,
    dataset_world_model_reco: dict[str, object] | None,
    candidate_state_frontier: dict[str, object] | None,
    total_train_tokens_available: int,
    profiled_tokens: int,
) -> dict[str, object]:
    lag_map = _normalized_mi_map(lag_stats)
    return {
        "total_train_tokens_available": int(total_train_tokens_available),
        "profiled_tokens": int(profiled_tokens),
        "profiled_fraction_of_full_stream": float(int(profiled_tokens) / max(int(total_train_tokens_available), 1)),
        "lag_normalized_mi": {
            "lag1": float(lag_map.get(1, 0.0)),
            "lag2": float(lag_map.get(2, 0.0)),
            "lag4": float(lag_map.get(4, 0.0)),
            "lag8": float(lag_map.get(8, 0.0)),
        },
        "local_signal_mass": float(capacity_inference.get("local_signal_mass", 0.0)),
        "medium_signal_mass": float(capacity_inference.get("medium_signal_mass", 0.0)),
        "long_signal_mass": float(capacity_inference.get("long_signal_mass", 0.0)),
        "local_dominance": float(capacity_inference.get("local_dominance", 0.0)),
        "preferred_backbone_band": capacity_inference.get("preferred_backbone_band"),
        "minimal_causal_state_recommendation": None
        if not isinstance(minimal_causal_state_reco, dict)
        else minimal_causal_state_reco.get("recommended_state"),
        "predictive_state_recommendation": None
        if not isinstance(predictive_state_recommendation, dict)
        else predictive_state_recommendation.get("recommended_state"),
        "predictive_state_transition_recommendation": None
        if not isinstance(predictive_state_transition_recommendation, dict)
        else predictive_state_transition_recommendation.get("recommended_operator"),
        "spectral_basis_recommendation": None
        if not isinstance(spectral_recommendation, dict)
        else spectral_recommendation.get("recommended_basis"),
        "dataset_world_model_recommendation": None
        if not isinstance(dataset_world_model_reco, dict)
        else dataset_world_model_reco.get("recommended_world_model"),
        "candidate_state_frontier_best_trainable": None
        if not isinstance(candidate_state_frontier, dict)
        else candidate_state_frontier.get("best_trainable"),
    }


def build_delta_to_full_stream(
    *,
    token_seen_ladder: dict[str, object] | None,
    full_stream_reference: dict[str, object],
) -> dict[str, object]:
    rows = None if not isinstance(token_seen_ladder, dict) else token_seen_ladder.get("rows")
    if not isinstance(rows, list) or not rows:
        return {"available": False}
    full_lags = full_stream_reference.get("lag_normalized_mi") or {}
    full_state = full_stream_reference.get("minimal_causal_state_recommendation") or {}
    full_pred = full_stream_reference.get("predictive_state_recommendation") or {}
    out_rows: list[dict[str, object]] = []
    for row in rows:
        out_rows.append(
            {
                "tokens_seen": int(row.get("tokens_seen", 0)),
                "prefix_tokens_profiled": int(row.get("prefix_tokens_profiled", 0)),
                "dataset_reveal_fraction": float(row.get("dataset_reveal_fraction", 0.0)),
                "run_budget_fraction": float(row.get("run_budget_fraction", 0.0)),
                "approx_train_steps": float(row.get("approx_train_steps", 0.0)),
                "delta_lag1_normalized_mi_vs_full": float(row.get("lag1_normalized_mi", 0.0)) - float(full_lags.get("lag1", 0.0)),
                "delta_lag2_normalized_mi_vs_full": float(row.get("lag2_normalized_mi", 0.0)) - float(full_lags.get("lag2", 0.0)),
                "delta_lag4_normalized_mi_vs_full": float(row.get("lag4_normalized_mi", 0.0)) - float(full_lags.get("lag4", 0.0)),
                "delta_lag8_normalized_mi_vs_full": float(row.get("lag8_normalized_mi", 0.0)) - float(full_lags.get("lag8", 0.0)),
                "delta_minimal_causal_state_bpb_vs_full": float(row.get("minimal_causal_state_bpb", 0.0))
                - float(full_state.get("heldout_bits_per_byte", 0.0)),
                "delta_predictive_state_gap_bits_vs_full": float(row.get("predictive_state_gap_bits", 0.0))
                - float(full_pred.get("cross_entropy_gap_bits", 0.0)),
            }
        )
    return {"available": True, "rows": out_rows}


def build_competition_architecture_tradeoff(
    *,
    training_budget: dict[str, object],
    expected_reveal_fraction: float,
    token_seen_ladder: dict[str, object] | None,
    delta_to_full_stream: dict[str, object] | None,
    competition_target: dict[str, object] | None,
) -> dict[str, object]:
    ladder_rows = None if not isinstance(token_seen_ladder, dict) else token_seen_ladder.get("rows")
    delta_rows = None if not isinstance(delta_to_full_stream, dict) else delta_to_full_stream.get("rows")
    if not isinstance(ladder_rows, list) or not ladder_rows or not isinstance(delta_rows, list) or not delta_rows:
        return {"available": False}
    baseline_tokens_seen = int(training_budget.get("estimated_train_tokens", 0))
    baseline_row = _nearest_tokens_seen_row(ladder_rows, baseline_tokens_seen)
    baseline_delta = _nearest_tokens_seen_row(delta_rows, baseline_tokens_seen)
    sensitivity_rows: list[dict[str, object]] = []
    for scale in (0.9, 0.95, 1.0, 1.05, 1.1):
        target = int(round(baseline_tokens_seen * scale))
        ladder_row = _nearest_tokens_seen_row(ladder_rows, target)
        delta_row = _nearest_tokens_seen_row(delta_rows, target)
        if ladder_row is None or delta_row is None:
            continue
        sensitivity_rows.append(
            {
                "relative_tokens_seen_scale": float(scale),
                "tokens_seen": int(ladder_row.get("tokens_seen", 0)),
                "approx_train_steps": float(ladder_row.get("approx_train_steps", 0.0)),
                "dataset_reveal_fraction": float(ladder_row.get("dataset_reveal_fraction", 0.0)),
                "preferred_backbone_band": ladder_row.get("preferred_backbone_band"),
                "delta_minimal_causal_state_bpb_vs_full": float(delta_row.get("delta_minimal_causal_state_bpb_vs_full", 0.0)),
                "delta_predictive_state_gap_bits_vs_full": float(delta_row.get("delta_predictive_state_gap_bits_vs_full", 0.0)),
            }
        )
    return {
        "available": True,
        "baseline_expected_reveal_fraction": float(expected_reveal_fraction),
        "baseline_expected_tokens_seen": int(baseline_tokens_seen),
        "baseline_row": baseline_row,
        "baseline_delta_to_full_stream": baseline_delta,
        "throughput_sensitivity_table": sensitivity_rows,
        "decision_note": None if not isinstance(competition_target, dict) else competition_target.get("stream_recommendation"),
    }


def build_causal_recommendations(
    lag_stats: list[object],
    spectral_recommendation: dict[str, object] | None,
    minimal_causal_state_reco: dict[str, object] | None,
    predictive_state_recommendation: dict[str, object] | None,
    predictive_state_transition_recommendation: dict[str, object] | None,
    dataset_world_model_reco: dict[str, object] | None,
    candidate_state_frontier: dict[str, object] | None,
) -> list[str]:
    notes: list[str] = []
    lag_map = _normalized_mi_map(lag_stats)
    if float(lag_map.get(4, 0.0) + lag_map.get(8, 0.0)) < 0.05:
        notes.append("Long-range token dependence is weak in the profiled stream; favor cheap causal heads before increasing backbone depth.")
    else:
        notes.append("Medium-range dependence is non-trivial; a light recurrent/state bias head is worth testing before major backbone scaling.")
    if isinstance(spectral_recommendation, dict) and isinstance(spectral_recommendation.get("recommended_basis"), dict):
        basis = spectral_recommendation["recommended_basis"]
        notes.append(
            f"Best fixed causal map is {basis.get('basis')} at lag={int(basis.get('lag', 0))}, rank={int(basis.get('rank', 0))}; this is the cleanest output-side bias candidate."
        )
    if isinstance(minimal_causal_state_reco, dict) and isinstance(minimal_causal_state_reco.get("recommended_state"), dict):
        rec = minimal_causal_state_reco["recommended_state"]
        notes.append(
            f"Minimal causal-state recommendation is rank={int(rec.get('rank', 0))}; the next gain should come from a tiny state bias, not a large new backbone."
        )
    if isinstance(predictive_state_recommendation, dict) and isinstance(predictive_state_recommendation.get("recommended_state"), dict):
        rec = predictive_state_recommendation["recommended_state"]
        notes.append(
            f"Predictive-state compression prefers order={int(rec.get('order', 0))}, rank={int(rec.get('rank', 0))}; this is the best compact learned next-token head candidate."
        )
    if isinstance(predictive_state_transition_recommendation, dict) and isinstance(
        predictive_state_transition_recommendation.get("recommended_operator"), dict
    ):
        rec = predictive_state_transition_recommendation["recommended_operator"]
        notes.append(
            f"Predictive-state dynamics remain compressible at rank={int(rec.get('rank', 0))}; a tiny recurrent operator is plausible."
        )
    if isinstance(dataset_world_model_reco, dict) and isinstance(dataset_world_model_reco.get("recommended_world_model"), dict):
        rec = dataset_world_model_reco["recommended_world_model"]
        notes.append(
            f"Chunked world-model structure coarse-grains into {int(rec.get('num_regimes', 0))} regimes over {int(rec.get('chunk_tokens', 0))} tokens; use this as an offline training prior, not a runtime controller."
        )
    if isinstance(candidate_state_frontier, dict) and isinstance(candidate_state_frontier.get("best_trainable"), dict):
        row = candidate_state_frontier["best_trainable"]
        notes.append(
            f"Best trainable state mechanism on the current frontier is {row.get('component')} at held-out bpb={float(row.get('heldout_bits_per_byte', 0.0)):.3f}; align trainer patches with this frontier before scaling FFN width."
        )
    return notes


def _softmax_np(scores: np.ndarray) -> np.ndarray:
    scores64 = scores.astype(np.float64, copy=False)
    row_max = np.max(scores64, axis=1, keepdims=True)
    stable = scores64 - row_max
    probs = np.exp(stable)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs


def _log_softmax_np(scores: np.ndarray) -> np.ndarray:
    scores64 = scores.astype(np.float64, copy=False)
    row_max = np.max(scores64, axis=1, keepdims=True)
    stable = scores64 - row_max
    logsumexp = np.log(np.exp(stable).sum(axis=1, keepdims=True)) + row_max
    return scores64 - logsumexp


def _teacher_state_scores(signatures: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    feat_norm = np.square(signatures.astype(np.float64, copy=False)).sum(axis=1, keepdims=True)
    centroid_norm = np.square(centroids.astype(np.float64, copy=False)).sum(axis=1)[None, :]
    dists = feat_norm - 2.0 * signatures.astype(np.float64, copy=False) @ centroids.astype(np.float64, copy=False).T + centroid_norm
    temperature = max(float(np.std(dists)), 1e-6)
    return -dists / temperature


def _random_mlp_init(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    nonlinearity: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    fan_in = max(int(input_dim), 1)
    fan_hidden = max(int(hidden_dim), 1)
    if str(nonlinearity).lower() == "relu":
        w1_scale = math.sqrt(2.0 / fan_in)
        w2_scale = math.sqrt(2.0 / fan_hidden)
    else:
        w1_scale = math.sqrt(1.0 / fan_in)
        w2_scale = math.sqrt(1.0 / fan_hidden)
    w1 = rng.normal(0.0, w1_scale, size=(input_dim, hidden_dim)).astype(np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = rng.normal(0.0, w2_scale, size=(hidden_dim, output_dim)).astype(np.float32)
    b2 = np.zeros((output_dim,), dtype=np.float32)
    return w1, b1, w2, b2


def _apply_random_mlp(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, nonlinearity: str) -> tuple[np.ndarray, np.ndarray]:
    pre = x.astype(np.float64, copy=False) @ w1.astype(np.float64, copy=False) + b1.astype(np.float64, copy=False)
    nl = str(nonlinearity).lower()
    if nl == "relu":
        hidden = np.maximum(pre, 0.0)
    elif nl == "linear":
        hidden = pre
    else:
        hidden = np.tanh(pre)
    logits = hidden @ w2.astype(np.float64, copy=False) + b2.astype(np.float64, copy=False)
    return logits, pre


def _rowwise_topk_overlap(student_scores: np.ndarray, teacher_labels: np.ndarray, k: int) -> float:
    if student_scores.size <= 0:
        return 0.0
    k = max(1, min(int(k), int(student_scores.shape[1])))
    topk = np.argpartition(student_scores, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == teacher_labels[:, None], axis=1)))


def _pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64, copy=False).reshape(-1)
    bb = b.astype(np.float64, copy=False).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = math.sqrt(max(float((aa * aa).sum()), 1e-12) * max(float((bb * bb).sum()), 1e-12))
    return float((aa @ bb) / denom) if denom > 0.0 else 0.0


def _stable_state_probs(scores: np.ndarray) -> np.ndarray:
    probs = _softmax_np(scores.astype(np.float64, copy=False))
    return np.clip(probs, 1e-12, 1.0)


def _bounded_condition_stats(singular_values: np.ndarray) -> tuple[float, float, float, float, float]:
    s = np.asarray(singular_values, dtype=np.float64)
    if s.size <= 0:
        return 6.0, 0.0, 1.0, 0.0, 0.0
    smax = float(np.max(s))
    if smax <= 0.0:
        return 6.0, 0.0, 1.0, 0.0, 0.0
    keep = s[s > max(smax * 1e-4, 1e-8)]
    if keep.size <= 0:
        keep = np.asarray([smax], dtype=np.float64)
    upper = float(np.percentile(keep, 90))
    lower = float(np.percentile(keep, 10))
    log10_cond = float(np.log10(max(upper / max(lower, 1e-8), 1.0)))
    stable_rank = float((keep @ keep) ** 2 / max(float(np.square(keep * keep).sum()), 1e-12))
    stable_rank_fraction = float(stable_rank / max(float(keep.size), 1.0))
    dead_fraction = float(np.mean(s <= max(smax * 1e-3, 1e-8)))
    gain_ratio = float(smax / max(float(np.mean(keep)), 1e-8))
    conditioning_score = float(1.0 / (1.0 + log10_cond + 1.5 * dead_fraction + 0.25 * max(gain_ratio - 1.0, 0.0)))
    return log10_cond, stable_rank_fraction, dead_fraction, gain_ratio, conditioning_score


def random_init_state_decodability_profile(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    holdout_fraction: float,
    hidden_ranks: list[int],
    seeds: list[int],
    nonlinearities: list[str],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sample_views = base._causal_state_sample_views(
        tokens,
        vocab_size,
        past_window,
        sketch_dim,
        future_signature_arrays,
        causal_state_arrays,
    )
    if sample_views is None:
        return {"available": False, "reason": "missing_causal_state_views"}, {}
    _, past_sketch, labels, _, signatures = sample_views
    centroids = np.asarray(causal_state_arrays.get("causal_machine_signature_centroids"))
    if centroids.size <= 0:
        return {"available": False, "reason": "missing_causal_machine_centroids"}, {}
    sample_count = int(past_sketch.shape[0])
    split = int(max(1, min(sample_count - 1, round(sample_count * (1.0 - holdout_fraction)))))
    hold_x = past_sketch[split:].astype(np.float32, copy=False)
    hold_labels = labels[split:].astype(np.int64, copy=False)
    hold_signatures = signatures[split:].astype(np.float32, copy=False)
    teacher_scores = _teacher_state_scores(hold_signatures, centroids.astype(np.float32, copy=False))
    teacher_log_probs = _log_softmax_np(teacher_scores)
    teacher_probs = np.exp(teacher_log_probs)
    rows: list[dict[str, object]] = []
    best_row = None
    best_export: dict[str, np.ndarray] = {}
    for nonlinearity in nonlinearities:
        for hidden_rank in hidden_ranks:
            for seed in seeds:
                w1, b1, w2, b2 = _random_mlp_init(
                    int(hold_x.shape[1]),
                    int(hidden_rank),
                    int(centroids.shape[0]),
                    nonlinearity=nonlinearity,
                    seed=seed,
                )
                student_scores, _ = _apply_random_mlp(hold_x, w1, b1, w2, b2, nonlinearity)
                student_log_probs = _log_softmax_np(student_scores)
                teacher_kl = float(np.mean(np.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=1)))
                row = {
                    "module_family": f"{nonlinearity}_mlp",
                    "hidden_rank": int(hidden_rank),
                    "seed": int(seed),
                    "teacher_kl_bits": teacher_kl / math.log(2.0),
                    "teacher_logit_correlation": _pearson_flat(student_scores, teacher_scores),
                    "teacher_top1_overlap": float(np.mean(np.argmax(student_scores, axis=1) == hold_labels)),
                    "teacher_top4_overlap": _rowwise_topk_overlap(student_scores, hold_labels, 4),
                    "teacher_top8_overlap": _rowwise_topk_overlap(student_scores, hold_labels, 8),
                    "estimated_fp16_state_bytes": int((w1.size + b1.size + w2.size + b2.size) * 2),
                }
                rows.append(row)
                if best_row is None or (
                    float(row["teacher_kl_bits"]),
                    -float(row["teacher_top4_overlap"]),
                    int(row["estimated_fp16_state_bytes"]),
                ) < (
                    float(best_row["teacher_kl_bits"]),
                    -float(best_row["teacher_top4_overlap"]),
                    int(best_row["estimated_fp16_state_bytes"]),
                ):
                    best_row = row
                    best_export = {
                        "random_init_state_decoder_w1": w1.astype(np.float32, copy=False),
                        "random_init_state_decoder_b1": b1.astype(np.float32, copy=False),
                        "random_init_state_decoder_w2": w2.astype(np.float32, copy=False),
                        "random_init_state_decoder_b2": b2.astype(np.float32, copy=False),
                    }
    return {
        "available": True,
        "holdout_prefixes": int(hold_x.shape[0]),
        "rows": rows,
        "best_candidate": best_row,
    }, best_export


def _fit_ridge_matrix_readout(features: np.ndarray, targets: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = features.mean(axis=0, keepdims=True).astype(np.float64, copy=False)
    centered = features.astype(np.float64, copy=False) - feature_mean
    xtx = centered.T @ centered
    weights = np.linalg.solve(
        xtx + np.eye(centered.shape[1], dtype=np.float64) * float(ridge),
        centered.T @ targets.astype(np.float64, copy=False),
    )
    bias = targets.astype(np.float64, copy=False).mean(axis=0)
    return weights.astype(np.float32, copy=False), bias.astype(np.float32, copy=False), feature_mean.astype(np.float32, copy=False)


def random_init_transition_alignment_profile(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    holdout_fraction: float,
    hidden_ranks: list[int],
    seeds: list[int],
    ridge: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sample_views = base._causal_state_sample_views(
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
    valid = np.diff(positions) == 1
    if not np.any(valid):
        return {"available": False, "reason": "no_consecutive_prefix_pairs"}, {}
    curr_labels = labels[:-1][valid].astype(np.int64, copy=False)
    next_labels = labels[1:][valid].astype(np.int64, copy=False)
    curr_features = past_sketch[:-1][valid].astype(np.float32, copy=False)
    num_states = int(max(labels.max(initial=0), next_labels.max(initial=0)) + 1)
    teacher_next = np.eye(num_states, dtype=np.float32)[next_labels]
    sample_count = int(curr_features.shape[0])
    split = int(max(1, min(sample_count - 1, round(sample_count * (1.0 - holdout_fraction)))))
    train_x = curr_features[:split]
    hold_x = curr_features[split:]
    train_curr = curr_labels[:split]
    hold_curr = curr_labels[split:]
    train_teacher = teacher_next[:split]
    hold_teacher = teacher_next[split:]
    hold_next = next_labels[split:]
    rows: list[dict[str, object]] = []
    best_row = None
    best_export: dict[str, np.ndarray] = {}
    # Random candidate families
    for hidden_rank in hidden_ranks:
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            # x_only linear
            wx = rng.normal(0.0, math.sqrt(1.0 / max(int(train_x.shape[1]), 1)), size=(train_x.shape[1], num_states)).astype(np.float32)
            bx = np.zeros((num_states,), dtype=np.float32)
            x_scores = hold_x.astype(np.float64, copy=False) @ wx.astype(np.float64, copy=False) + bx.astype(np.float64, copy=False)
            bits, _ = base._cross_entropy_bits_from_scores(x_scores, hold_next, None)
            rows.append(
                {
                    "module_family": "x_linear",
                    "hidden_rank": 0,
                    "seed": int(seed),
                    "teacher_transition_ce_bits": float(bits),
                    "teacher_top1_accuracy": float(np.mean(np.argmax(x_scores, axis=1) == hold_next)),
                    "spectral_radius": 0.0,
                    "stability_score": 1.0,
                    "gradient_flow_score": float(np.linalg.svd(wx.astype(np.float64), compute_uv=False).mean()),
                    "estimated_fp16_state_bytes": int((wx.size + bx.size) * 2),
                }
            )
            # state_x_tanh
            w1 = rng.normal(0.0, math.sqrt(1.0 / max(int(train_x.shape[1]), 1)), size=(train_x.shape[1], hidden_rank)).astype(np.float32)
            u1 = rng.normal(0.0, math.sqrt(1.0 / max(num_states, 1)), size=(num_states, hidden_rank)).astype(np.float32)
            b1 = np.zeros((hidden_rank,), dtype=np.float32)
            w2 = rng.normal(0.0, math.sqrt(1.0 / max(hidden_rank, 1)), size=(hidden_rank, num_states)).astype(np.float32)
            b2 = np.zeros((num_states,), dtype=np.float32)
            hold_curr_oh = np.eye(num_states, dtype=np.float32)[hold_curr]
            pre = hold_x.astype(np.float64, copy=False) @ w1.astype(np.float64, copy=False) + hold_curr_oh.astype(np.float64, copy=False) @ u1.astype(np.float64, copy=False) + b1.astype(np.float64, copy=False)
            hidden = np.tanh(pre)
            scores = hidden @ w2.astype(np.float64, copy=False) + b2.astype(np.float64, copy=False)
            bits, _ = base._cross_entropy_bits_from_scores(scores, hold_next, None)
            eigvals = np.linalg.eigvals(u1.astype(np.float64) @ w2.astype(np.float64))
            spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 0.0
            stability_score = float(1.0 / (1.0 + max(spectral_radius - 1.0, 0.0)))
            grad_scale = float(np.mean(1.0 - np.square(np.tanh(pre))))
            row = {
                "module_family": "state_x_tanh",
                "hidden_rank": int(hidden_rank),
                "seed": int(seed),
                "teacher_transition_ce_bits": float(bits),
                "teacher_top1_accuracy": float(np.mean(np.argmax(scores, axis=1) == hold_next)),
                "spectral_radius": spectral_radius,
                "stability_score": stability_score,
                "gradient_flow_score": grad_scale,
                "estimated_fp16_state_bytes": int((w1.size + u1.size + b1.size + w2.size + b2.size) * 2),
            }
            rows.append(row)
            if best_row is None or (
                float(row["teacher_transition_ce_bits"]),
                -float(row["teacher_top1_accuracy"]),
                int(row["estimated_fp16_state_bytes"]),
            ) < (
                float(best_row["teacher_transition_ce_bits"]),
                -float(best_row["teacher_top1_accuracy"]),
                int(best_row["estimated_fp16_state_bytes"]),
            ):
                best_row = row
                best_export = {
                    "random_init_transition_w1": w1.astype(np.float32, copy=False),
                    "random_init_transition_u1": u1.astype(np.float32, copy=False),
                    "random_init_transition_b1": b1.astype(np.float32, copy=False),
                    "random_init_transition_w2": w2.astype(np.float32, copy=False),
                    "random_init_transition_b2": b2.astype(np.float32, copy=False),
                }
    # Fitted linear transition teacher projection for comparison/export
    fit_w, fit_b, fit_mean = _fit_ridge_matrix_readout(train_x, train_teacher, ridge=float(ridge))
    best_export.update(
        {
            "causal_state_transition_target_weights": fit_w.astype(np.float32, copy=False),
            "causal_state_transition_target_bias": fit_b.astype(np.float32, copy=False),
            "causal_state_transition_target_mean": fit_mean.astype(np.float32, copy=False),
        }
    )
    return {
        "available": True,
        "train_transitions": int(train_x.shape[0]),
        "holdout_transitions": int(hold_x.shape[0]),
        "rows": rows,
        "best_candidate": best_row,
    }, best_export


def teacher_jacobian_condition_profile(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    state_init_arrays: dict[str, np.ndarray],
    transition_init_arrays: dict[str, np.ndarray],
    max_samples: int,
) -> dict[str, object]:
    sample_views = base._causal_state_sample_views(
        tokens,
        vocab_size,
        past_window,
        sketch_dim,
        future_signature_arrays,
        causal_state_arrays,
    )
    if sample_views is None:
        return {"available": False, "reason": "missing_causal_state_views"}
    _, past_sketch, labels, _, _ = sample_views
    x = past_sketch[: max(1, min(int(max_samples), int(past_sketch.shape[0])) )].astype(np.float64, copy=False)
    curr_labels = labels[: x.shape[0]].astype(np.int64, copy=False)
    rows: list[dict[str, object]] = []

    def _summarize_jacobians(name: str, jacobians: list[np.ndarray]) -> dict[str, object]:
        conds = []
        stable_ranks = []
        dead = []
        gain_ratios = []
        conditioning_scores = []
        for J in jacobians:
            s = np.linalg.svd(J, compute_uv=False)
            log10_cond, stable_rank_fraction, dead_fraction, gain_ratio, conditioning_score = _bounded_condition_stats(s)
            conds.append(log10_cond)
            stable_ranks.append(stable_rank_fraction)
            dead.append(dead_fraction)
            gain_ratios.append(gain_ratio)
            conditioning_scores.append(conditioning_score)
        return {
            "module_family": name,
            "mean_condition_number": float(10.0 ** np.mean(conds)),
            "mean_log10_condition_number": float(np.mean(conds)),
            "mean_stable_rank_fraction": float(np.mean(stable_ranks)),
            "dead_direction_fraction": float(np.mean(dead)),
            "mean_gain_ratio": float(np.mean(gain_ratios)),
            "conditioning_score": float(np.mean(conditioning_scores)),
        }

    if all(k in state_init_arrays for k in ("random_init_state_decoder_w1", "random_init_state_decoder_b1", "random_init_state_decoder_w2", "random_init_state_decoder_b2")):
        w1 = state_init_arrays["random_init_state_decoder_w1"].astype(np.float64, copy=False)
        b1 = state_init_arrays["random_init_state_decoder_b1"].astype(np.float64, copy=False)
        w2 = state_init_arrays["random_init_state_decoder_w2"].astype(np.float64, copy=False)
        jacobians = []
        for xi in x:
            pre = xi @ w1 + b1
            deriv = 1.0 - np.square(np.tanh(pre))
            jacobians.append(w1 @ (deriv[:, None] * w2))
        rows.append(_summarize_jacobians("random_init_state_decoder", jacobians))

    if all(k in transition_init_arrays for k in ("random_init_transition_w1", "random_init_transition_u1", "random_init_transition_b1", "random_init_transition_w2", "random_init_transition_b2")):
        w1 = transition_init_arrays["random_init_transition_w1"].astype(np.float64, copy=False)
        u1 = transition_init_arrays["random_init_transition_u1"].astype(np.float64, copy=False)
        b1 = transition_init_arrays["random_init_transition_b1"].astype(np.float64, copy=False)
        w2 = transition_init_arrays["random_init_transition_w2"].astype(np.float64, copy=False)
        jacobians = []
        eye_states = np.eye(u1.shape[0], dtype=np.float64)
        for xi, li in zip(x, curr_labels.tolist(), strict=True):
            pre = xi @ w1 + eye_states[int(li)] @ u1 + b1
            deriv = 1.0 - np.square(np.tanh(pre))
            jacobians.append(w1 @ (deriv[:, None] * w2))
        rows.append(_summarize_jacobians("random_init_transition", jacobians))

    best = max(
        rows,
        key=lambda row: (
            float(row.get("conditioning_score", 0.0)),
            float(row.get("mean_stable_rank_fraction", 0.0)),
            -float(row.get("dead_direction_fraction", 1.0)),
        ),
        default=None,
    )
    return {
        "available": bool(rows),
        "jacobian_samples": int(x.shape[0]),
        "rows": rows,
        "best_conditioned_module": best,
    }


def optimizer_reachability_profile(
    *,
    training_budget: dict[str, object],
    random_init_state_decodability: dict[str, object],
    random_init_transition_alignment: dict[str, object],
    teacher_jacobian_condition: dict[str, object],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    estimated_steps = int(training_budget.get("estimated_steps", 0))
    jacobian_rows = {str(row.get("module_family")): row for row in teacher_jacobian_condition.get("rows", [])} if isinstance(teacher_jacobian_condition, dict) else {}
    for source_name, profile, metric_key, module_family in (
        ("state_decoder", random_init_state_decodability, "teacher_kl_bits", "random_init_state_decoder"),
        ("transition", random_init_transition_alignment, "teacher_transition_ce_bits", "random_init_transition"),
    ):
        for row in profile.get("rows", []) if isinstance(profile, dict) else []:
            jac = jacobian_rows.get(module_family, {})
            conditioning_score = float(jac.get("conditioning_score", 0.2))
            stable_rank_fraction = float(jac.get("mean_stable_rank_fraction", 0.0))
            dead_fraction = float(jac.get("dead_direction_fraction", 1.0))
            params_bytes = int(row.get("estimated_fp16_state_bytes", 0))
            bytes_penalty = 1.0 + math.log1p(max(params_bytes, 1)) / 16.0
            difficulty = float(row.get(metric_key, 0.0)) * bytes_penalty / max(0.2 + conditioning_score + 0.25 * stable_rank_fraction, 1e-6)
            difficulty *= 1.0 + 0.5 * dead_fraction
            budget_scale = max(math.log1p(max(estimated_steps, 1)), 1.0)
            reachability = float(1.0 / (1.0 + difficulty / budget_scale))
            if conditioning_score < 0.18:
                lr_scale = 0.5
            elif conditioning_score > 0.38 and dead_fraction < 0.35:
                lr_scale = 1.25
            else:
                lr_scale = 1.0
            rows.append(
                {
                    "source": source_name,
                    "module_family": row.get("module_family"),
                    "hidden_rank": int(row.get("hidden_rank", 0)),
                    "seed": int(row.get("seed", 0)),
                    "estimated_fp16_state_bytes": params_bytes,
                    "difficulty_score": difficulty,
                    "reachability_score": reachability,
                    "suggested_lr_scale": lr_scale,
                    "budget_fit": "high" if reachability >= 0.4 else ("medium" if reachability >= 0.22 else "low"),
                }
            )
    best = max(rows, key=lambda row: float(row["reachability_score"]), default=None)
    return {
        "available": bool(rows),
        "estimated_train_steps": estimated_steps,
        "rows": rows,
        "best_candidate": best,
    }


def state_teacher_curriculum_profile(
    *,
    causal_state_decodability: dict[str, object],
    causal_state_transition_learnability: dict[str, object],
    causal_state_residual_geometry: dict[str, object],
    random_init_state_decodability: dict[str, object],
    random_init_transition_alignment: dict[str, object],
    num_states: int,
) -> dict[str, object]:
    log_states = math.log2(max(int(num_states), 2))
    tasks = []
    best_dec = (causal_state_decodability.get("best_trainable_feature_family") or {}) if isinstance(causal_state_decodability, dict) else {}
    best_trans = (causal_state_transition_learnability.get("best_predictor_family") or {}) if isinstance(causal_state_transition_learnability, dict) else {}
    best_init_dec = (random_init_state_decodability.get("best_candidate") or {}) if isinstance(random_init_state_decodability, dict) else {}
    best_init_trans = (random_init_transition_alignment.get("best_candidate") or {}) if isinstance(random_init_transition_alignment, dict) else {}
    resid_rows = causal_state_residual_geometry.get("rows", []) if isinstance(causal_state_residual_geometry, dict) else []
    rank32_resid = next((row for row in resid_rows if int(row.get("rank", 0)) == 32), None)
    residual_gap = 1.0 - float((rank32_resid or {}).get("residual_energy_fraction", 0.0))
    tasks.append({"task": "state_ids", "difficulty": float(best_dec.get("state_cross_entropy_bits", log_states)) / max(log_states, 1e-6)})
    tasks.append({"task": "state_logits", "difficulty": float(best_init_dec.get("teacher_kl_bits", log_states)) / max(log_states, 1e-6)})
    tasks.append({"task": "transitions", "difficulty": float(best_trans.get("next_state_cross_entropy_bits", log_states)) / max(log_states, 1e-6)})
    tasks.append({"task": "transition_init", "difficulty": float(best_init_trans.get("teacher_transition_ce_bits", log_states)) / max(log_states, 1e-6)})
    tasks.append({"task": "residual_basis", "difficulty": residual_gap})
    ordered = sorted(tasks, key=lambda row: float(row["difficulty"]))
    return {
        "available": True,
        "rows": tasks,
        "recommended_order": [row["task"] for row in ordered],
        "first_focus": None if not ordered else ordered[0]["task"],
        "last_focus": None if not ordered else ordered[-1]["task"],
    }


def online_recovery_gap_profile(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    past_window: int,
    sketch_dim: int,
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    causal_state_decodability_arrays: dict[str, np.ndarray],
    causal_state_transition_learnability_arrays: dict[str, np.ndarray],
    causal_state_residual_geometry_arrays: dict[str, np.ndarray],
    strict_online_state_eval: dict[str, object],
    causal_state_reconstruction: dict[str, object],
    state_entropy_floor: dict[str, object],
    holdout_fraction: float,
    ridge: float,
    base_bytes: np.ndarray | None,
) -> dict[str, object]:
    sample_views = base._causal_state_sample_views(
        tokens,
        vocab_size,
        past_window,
        sketch_dim,
        future_signature_arrays,
        causal_state_arrays,
    )
    if sample_views is None:
        return {"available": False, "reason": "missing_causal_state_views"}
    positions, past_sketch, labels, targets, signatures = sample_views
    centroids = np.asarray(causal_state_arrays.get("causal_machine_signature_centroids")).astype(np.float64, copy=False)
    state_log_probs = np.asarray(causal_state_arrays.get("causal_machine_log_probs")).astype(np.float64, copy=False)
    valid = np.diff(positions) == 1
    if not np.any(valid):
        return {"available": False, "reason": "no_consecutive_prefix_pairs"}
    curr_x = past_sketch[:-1][valid].astype(np.float64, copy=False)
    next_x = past_sketch[1:][valid].astype(np.float64, copy=False)
    curr_labels = labels[:-1][valid].astype(np.int64, copy=False)
    next_labels = labels[1:][valid].astype(np.int64, copy=False)
    next_targets = targets[1:][valid].astype(np.int64, copy=False)
    next_signatures = signatures[1:][valid].astype(np.float64, copy=False)
    sample_count = int(curr_x.shape[0])
    split = int(max(1, min(sample_count - 1, round(sample_count * (1.0 - holdout_fraction)))))
    hold_curr_x = curr_x[split:]
    hold_next_x = next_x[split:]
    hold_curr_labels = curr_labels[split:]
    hold_next_labels = next_labels[split:]
    hold_next_targets = next_targets[split:]
    hold_next_signatures = next_signatures[split:]
    rows: list[dict[str, object]] = []
    offline_bpb = float(((causal_state_reconstruction.get("best_state_count_by_holdout_bpb") or {}).get("heldout_bits_per_byte", 0.0)))
    rows.append({"stage": "offline_causal_machine", "heldout_bits_per_byte": offline_bpb})
    strict_bpb = None if not isinstance(strict_online_state_eval, dict) else strict_online_state_eval.get("next_token_bits_per_byte")
    if all(k in causal_state_decodability_arrays for k in ("causal_state_decoder_weights", "causal_state_decoder_bias", "causal_state_decoder_state_mean")):
        dw = causal_state_decodability_arrays["causal_state_decoder_weights"].astype(np.float64, copy=False)
        db = causal_state_decodability_arrays["causal_state_decoder_bias"].astype(np.float64, copy=False)
        dm = causal_state_decodability_arrays["causal_state_decoder_state_mean"].astype(np.float64, copy=False)
        decoder_scores = (hold_next_x - dm) @ dw + db
        decoder_probs = _stable_state_probs(decoder_scores)
        bits, decoder_bpb = base._cross_entropy_bits_from_scores(decoder_probs @ state_log_probs, hold_next_targets, base_bytes)
        rows.append(
            {
                "stage": "decoder_only_online",
                "heldout_bits_per_byte": decoder_bpb,
                "heldout_cross_entropy_bits": float(bits),
                "strict_online_reference_bpb": None if strict_bpb is None else float(strict_bpb),
                "next_state_top1_accuracy": float(np.mean(np.argmax(decoder_scores, axis=1) == hold_next_labels)),
            }
        )
    elif strict_bpb is not None:
        rows.append({"stage": "decoder_only_online", "heldout_bits_per_byte": float(strict_bpb)})
    transition_keys = {"causal_state_transition_target_weights", "causal_state_transition_target_bias", "causal_state_transition_target_mean"}
    fallback_keys = {"causal_state_transition_weights", "causal_state_transition_bias", "causal_state_transition_feature_mean"}
    if transition_keys.issubset(causal_state_transition_learnability_arrays):
        tw = causal_state_transition_learnability_arrays["causal_state_transition_target_weights"].astype(np.float64, copy=False)
        tb = causal_state_transition_learnability_arrays["causal_state_transition_target_bias"].astype(np.float64, copy=False)
        tm = causal_state_transition_learnability_arrays["causal_state_transition_target_mean"].astype(np.float64, copy=False)
    elif fallback_keys.issubset(causal_state_transition_learnability_arrays):
        tw = causal_state_transition_learnability_arrays["causal_state_transition_weights"].astype(np.float64, copy=False)
        tb = causal_state_transition_learnability_arrays["causal_state_transition_bias"].astype(np.float64, copy=False)
        tm = causal_state_transition_learnability_arrays["causal_state_transition_feature_mean"].astype(np.float64, copy=False)
    else:
        tw = tb = tm = None
    if tw is not None and tb is not None and tm is not None:
        trans_scores = (hold_curr_x - tm) @ tw + tb
        trans_probs = _stable_state_probs(trans_scores)
        bits, bpb = base._cross_entropy_bits_from_scores(trans_probs @ state_log_probs, hold_next_targets, base_bytes)
        rows.append(
            {
                "stage": "decoder_plus_transition",
                "heldout_bits_per_byte": bpb,
                "heldout_cross_entropy_bits": float(bits),
                "next_state_top1_accuracy": float(np.mean(np.argmax(trans_scores, axis=1) == hold_next_labels)),
            }
        )
        if all(k in causal_state_residual_geometry_arrays for k in ("causal_state_residual_basis", "causal_state_residual_mean")):
            basis = causal_state_residual_geometry_arrays["causal_state_residual_basis"].astype(np.float64, copy=False)
            resid_mean = causal_state_residual_geometry_arrays["causal_state_residual_mean"].astype(np.float64, copy=False)
            train_curr_x = curr_x[:split].astype(np.float64, copy=False)
            train_next_signatures = next_signatures[:split].astype(np.float64, copy=False)
            train_trans_scores = (train_curr_x - tm) @ tw + tb
            train_trans_probs = _stable_state_probs(train_trans_scores)
            train_base_sig = train_trans_probs @ centroids
            train_resid = train_next_signatures - train_base_sig
            train_coeff = (train_resid - resid_mean) @ basis
            rw, rb, rm = _fit_ridge_matrix_readout(curr_x[:split].astype(np.float32, copy=False), train_coeff.astype(np.float32, copy=False), float(ridge))
            pred_coeff = (hold_curr_x - rm.astype(np.float64, copy=False)) @ rw.astype(np.float64, copy=False) + rb.astype(np.float64, copy=False)
            base_sig = trans_probs @ centroids
            pred_sig = base_sig + resid_mean + pred_coeff @ basis.T
            refined_state_scores = _teacher_state_scores(pred_sig.astype(np.float32, copy=False), centroids.astype(np.float32, copy=False))
            refined_state_probs = _stable_state_probs(refined_state_scores)
            bits, bpb = base._cross_entropy_bits_from_scores(refined_state_probs @ state_log_probs, hold_next_targets, base_bytes)
            rows.append(
                {
                    "stage": "decoder_plus_transition_plus_residual",
                    "heldout_bits_per_byte": bpb,
                    "heldout_cross_entropy_bits": float(bits),
                }
            )
    floor_bpb = None if not isinstance(state_entropy_floor, dict) else state_entropy_floor.get("heldout_bits_per_byte")
    rows.append({"stage": "state_entropy_floor", "heldout_bits_per_byte": None if floor_bpb is None else float(floor_bpb)})
    best = min([row for row in rows if row.get("heldout_bits_per_byte") is not None], key=lambda row: float(row["heldout_bits_per_byte"]), default=None)
    return {
        "available": True,
        "rows": rows,
        "best_stage": best,
    }


def init_frontier_profile(
    *,
    random_init_state_decodability: dict[str, object],
    random_init_transition_alignment: dict[str, object],
    teacher_jacobian_condition: dict[str, object],
    optimizer_reachability: dict[str, object],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    jac_map = {str(row.get("module_family")): row for row in teacher_jacobian_condition.get("rows", [])} if isinstance(teacher_jacobian_condition, dict) else {}
    reach_rows = optimizer_reachability.get("rows", []) if isinstance(optimizer_reachability, dict) else []
    for row in random_init_state_decodability.get("rows", []) if isinstance(random_init_state_decodability, dict) else []:
        reach = next((r for r in reach_rows if r.get("source") == "state_decoder" and r.get("hidden_rank") == row.get("hidden_rank") and r.get("seed") == row.get("seed")), None)
        jac = jac_map.get("random_init_state_decoder", {})
        rows.append(
            {
                "module_type": "state_decoder",
                "module_family": row.get("module_family"),
                "hidden_rank": int(row.get("hidden_rank", 0)),
                "seed": int(row.get("seed", 0)),
                "init_teacher_metric": float(row.get("teacher_kl_bits", 0.0)),
                "init_top4_overlap": float(row.get("teacher_top4_overlap", 0.0)),
                "jacobian_condition": float(jac.get("mean_condition_number", 0.0)),
                "reachability_score": None if reach is None else float(reach.get("reachability_score", 0.0)),
                "estimated_fp16_state_bytes": int(row.get("estimated_fp16_state_bytes", 0)),
            }
        )
    for row in random_init_transition_alignment.get("rows", []) if isinstance(random_init_transition_alignment, dict) else []:
        reach = next((r for r in reach_rows if r.get("source") == "transition" and r.get("hidden_rank") == row.get("hidden_rank") and r.get("seed") == row.get("seed")), None)
        jac = jac_map.get("random_init_transition", {})
        rows.append(
            {
                "module_type": "transition",
                "module_family": row.get("module_family"),
                "hidden_rank": int(row.get("hidden_rank", 0)),
                "seed": int(row.get("seed", 0)),
                "init_teacher_metric": float(row.get("teacher_transition_ce_bits", 0.0)),
                "init_top4_overlap": float(row.get("teacher_top1_accuracy", 0.0)),
                "jacobian_condition": float(jac.get("mean_condition_number", 0.0)),
                "reachability_score": None if reach is None else float(reach.get("reachability_score", 0.0)),
                "estimated_fp16_state_bytes": int(row.get("estimated_fp16_state_bytes", 0)),
            }
        )
    pareto = []
    sorted_rows = sorted(
        [row for row in rows if row.get("reachability_score") is not None],
        key=lambda row: (-float(row["reachability_score"]), float(row["estimated_fp16_state_bytes"]), float(row["init_teacher_metric"])),
    )
    for row in sorted_rows:
        if not pareto or float(row["estimated_fp16_state_bytes"]) < float(pareto[-1]["estimated_fp16_state_bytes"]):
            pareto.append(row)
    best = sorted_rows[0] if sorted_rows else None
    return {
        "available": bool(rows),
        "rows": rows,
        "pareto_frontier": pareto,
        "best_candidate": best,
    }


def _onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels.astype(np.int64, copy=False)] = 1.0
    return out


def _activation_forward(pre: np.ndarray, nonlinearity: str) -> np.ndarray:
    nl = str(nonlinearity).strip().lower()
    if nl == "relu":
        return np.maximum(pre, 0.0)
    if nl == "linear":
        return pre
    return np.tanh(pre)


def _activation_backward(hidden: np.ndarray, pre: np.ndarray, nonlinearity: str) -> np.ndarray:
    nl = str(nonlinearity).strip().lower()
    if nl == "relu":
        return (pre > 0.0).astype(np.float64, copy=False)
    if nl == "linear":
        return np.ones_like(pre, dtype=np.float64)
    return 1.0 - np.square(hidden)


def _adamw_step(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    *,
    step: int,
    lr: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad * grad)
    mhat = m / max(1.0 - beta1**step, 1e-12)
    vhat = v / max(1.0 - beta2**step, 1e-12)
    param = param - lr * (mhat / (np.sqrt(vhat) + eps) + weight_decay * param)
    return param, m, v


def student_teacher_fit_profile(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    base_bytes: np.ndarray | None,
    past_window: int,
    sketch_dim: int,
    future_signature_arrays: dict[str, np.ndarray],
    causal_state_arrays: dict[str, np.ndarray],
    holdout_fraction: float,
    random_init_state_decodability: dict[str, object],
    max_prefixes: int,
    top_inits: int,
    fit_steps: list[int],
    learning_rates: list[float],
    weight_decays: list[float],
    state_loss_coeffs: list[float],
    teacher_loss_coeffs: list[float],
    training_budget: dict[str, object],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sample_views = base._causal_state_sample_views(
        tokens,
        vocab_size,
        past_window,
        sketch_dim,
        future_signature_arrays,
        causal_state_arrays,
    )
    if sample_views is None:
        return {"available": False, "reason": "missing_causal_state_views"}, {}
    _, past_sketch, labels, targets, signatures = sample_views
    centroids = np.asarray(causal_state_arrays.get("causal_machine_signature_centroids")).astype(np.float32, copy=False)
    state_log_probs = np.asarray(causal_state_arrays.get("causal_machine_log_probs")).astype(np.float64, copy=False)
    if centroids.size <= 0 or state_log_probs.size <= 0:
        return {"available": False, "reason": "missing_causal_machine_arrays"}, {}
    sample_count = int(past_sketch.shape[0])
    split = int(max(1, min(sample_count - 1, round(sample_count * (1.0 - holdout_fraction)))))
    train_x = past_sketch[:split].astype(np.float64, copy=False)
    hold_x = past_sketch[split:].astype(np.float64, copy=False)
    train_labels = labels[:split].astype(np.int64, copy=False)
    hold_labels = labels[split:].astype(np.int64, copy=False)
    hold_targets = targets[split:].astype(np.int64, copy=False)
    train_signatures = signatures[:split].astype(np.float32, copy=False)
    hold_signatures = signatures[split:].astype(np.float32, copy=False)
    if max_prefixes > 0:
        train_take = min(int(max_prefixes), int(train_x.shape[0]))
        hold_take = min(max(int(max_prefixes // 4), 1024), int(hold_x.shape[0]))
        train_x = train_x[-train_take:]
        train_labels = train_labels[-train_take:]
        train_signatures = train_signatures[-train_take:]
        hold_x = hold_x[-hold_take:]
        hold_labels = hold_labels[-hold_take:]
        hold_targets = hold_targets[-hold_take:]
        hold_signatures = hold_signatures[-hold_take:]
    teacher_train_scores = _teacher_state_scores(train_signatures, centroids)
    teacher_hold_scores = _teacher_state_scores(hold_signatures, centroids)
    teacher_train_probs = _stable_state_probs(teacher_train_scores)
    teacher_hold_probs = _stable_state_probs(teacher_hold_scores)
    train_onehot = _onehot(train_labels, int(centroids.shape[0]))
    candidate_rows = []
    for row in random_init_state_decodability.get("rows", []) if isinstance(random_init_state_decodability, dict) else []:
        if "mlp" not in str(row.get("module_family", "")):
            continue
        candidate_rows.append(row)
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            float(row.get("teacher_kl_bits", math.inf)),
            -float(row.get("teacher_top4_overlap", 0.0)),
            int(row.get("estimated_fp16_state_bytes", 0)),
        ),
    )[: max(int(top_inits), 1)]
    if not candidate_rows:
        return {"available": False, "reason": "no_random_init_decoder_candidates"}, {}

    rows: list[dict[str, object]] = []
    best_row = None
    best_export: dict[str, np.ndarray] = {}
    for init_row in candidate_rows:
        module_family = str(init_row.get("module_family", "tanh_mlp"))
        nonlinearity = module_family.replace("_mlp", "")
        hidden_rank = int(init_row.get("hidden_rank", 0))
        seed = int(init_row.get("seed", 0))
        for fit_step_count in sorted({max(int(v), 1) for v in fit_steps}):
            for lr in sorted({float(v) for v in learning_rates if float(v) > 0.0}):
                for weight_decay in sorted({max(float(v), 0.0) for v in weight_decays}):
                    for state_loss_coeff in sorted({max(float(v), 0.0) for v in state_loss_coeffs}):
                        for teacher_loss_coeff in sorted({max(float(v), 0.0) for v in teacher_loss_coeffs}):
                            if state_loss_coeff <= 0.0 and teacher_loss_coeff <= 0.0:
                                continue
                            w1, b1, w2, b2 = _random_mlp_init(
                                int(train_x.shape[1]),
                                hidden_rank,
                                int(centroids.shape[0]),
                                nonlinearity=nonlinearity,
                                seed=seed,
                            )
                            w1 = w1.astype(np.float64, copy=False)
                            b1 = b1.astype(np.float64, copy=False)
                            w2 = w2.astype(np.float64, copy=False)
                            b2 = b2.astype(np.float64, copy=False)
                            m_w1 = np.zeros_like(w1)
                            v_w1 = np.zeros_like(w1)
                            m_b1 = np.zeros_like(b1)
                            v_b1 = np.zeros_like(b1)
                            m_w2 = np.zeros_like(w2)
                            v_w2 = np.zeros_like(w2)
                            m_b2 = np.zeros_like(b2)
                            v_b2 = np.zeros_like(b2)
                            for step_idx in range(1, fit_step_count + 1):
                                pre = train_x @ w1 + b1
                                hidden = _activation_forward(pre, nonlinearity)
                                logits = hidden @ w2 + b2
                                probs = _stable_state_probs(logits)
                                grad_logits = (
                                    state_loss_coeff * (probs - train_onehot)
                                    + teacher_loss_coeff * (probs - teacher_train_probs)
                                ) / max(float(train_x.shape[0]), 1.0)
                                grad_w2 = hidden.T @ grad_logits
                                grad_b2 = grad_logits.sum(axis=0)
                                grad_hidden = grad_logits @ w2.T
                                grad_pre = grad_hidden * _activation_backward(hidden, pre, nonlinearity)
                                grad_w1 = train_x.T @ grad_pre
                                grad_b1 = grad_pre.sum(axis=0)
                                w1, m_w1, v_w1 = _adamw_step(w1, grad_w1, m_w1, v_w1, step=step_idx, lr=lr, weight_decay=weight_decay)
                                b1, m_b1, v_b1 = _adamw_step(b1, grad_b1, m_b1, v_b1, step=step_idx, lr=lr, weight_decay=0.0)
                                w2, m_w2, v_w2 = _adamw_step(w2, grad_w2, m_w2, v_w2, step=step_idx, lr=lr, weight_decay=weight_decay)
                                b2, m_b2, v_b2 = _adamw_step(b2, grad_b2, m_b2, v_b2, step=step_idx, lr=lr, weight_decay=0.0)
                            hold_pre = hold_x @ w1 + b1
                            hold_hidden = _activation_forward(hold_pre, nonlinearity)
                            hold_logits = hold_hidden @ w2 + b2
                            hold_probs = _stable_state_probs(hold_logits)
                            hold_log_probs = np.log(np.clip(hold_probs, 1e-12, None))
                            teacher_hold_log_probs = np.log(np.clip(teacher_hold_probs, 1e-12, None))
                            teacher_kl_bits = float(
                                np.mean(np.sum(teacher_hold_probs * (teacher_hold_log_probs - hold_log_probs), axis=1))
                                / math.log(2.0)
                            )
                            state_bits, _ = base._cross_entropy_bits_from_scores(hold_logits, hold_labels, None)
                            token_bits, token_bpb = base._cross_entropy_bits_from_scores(hold_probs @ state_log_probs, hold_targets, base_bytes)
                            row = {
                                "module_family": module_family,
                                "hidden_rank": hidden_rank,
                                "seed": seed,
                                "teacher_fit_steps": int(fit_step_count),
                                "learning_rate": float(lr),
                                "weight_decay": float(weight_decay),
                                "state_loss_coeff": float(state_loss_coeff),
                                "teacher_loss_coeff": float(teacher_loss_coeff),
                                "holdout_teacher_kl_bits": teacher_kl_bits,
                                "holdout_state_cross_entropy_bits": float(state_bits),
                                "holdout_token_cross_entropy_bits": float(token_bits),
                                "holdout_token_bits_per_byte": token_bpb,
                                "holdout_state_top1_accuracy": float(np.mean(np.argmax(hold_logits, axis=1) == hold_labels)),
                                "holdout_teacher_top4_overlap": _rowwise_topk_overlap(hold_logits, hold_labels, 4),
                                "estimated_fp16_state_bytes": int((w1.size + b1.size + w2.size + b2.size) * 2),
                            }
                            rows.append(row)
                            if best_row is None or (
                                math.inf if row["holdout_token_bits_per_byte"] is None else float(row["holdout_token_bits_per_byte"]),
                                float(row["holdout_teacher_kl_bits"]),
                                int(row["estimated_fp16_state_bytes"]),
                            ) < (
                                math.inf if best_row["holdout_token_bits_per_byte"] is None else float(best_row["holdout_token_bits_per_byte"]),
                                float(best_row["holdout_teacher_kl_bits"]),
                                int(best_row["estimated_fp16_state_bytes"]),
                            ):
                                best_row = row
                                best_export = {
                                    "teacher_fit_state_decoder_w1": w1.astype(np.float32, copy=False),
                                    "teacher_fit_state_decoder_b1": b1.astype(np.float32, copy=False),
                                    "teacher_fit_state_decoder_w2": w2.astype(np.float32, copy=False),
                                    "teacher_fit_state_decoder_b2": b2.astype(np.float32, copy=False),
                                }
    best_by_bpb = min(
        [row for row in rows if row.get("holdout_token_bits_per_byte") is not None],
        key=lambda row: (float(row["holdout_token_bits_per_byte"]), float(row["holdout_teacher_kl_bits"])),
        default=None,
    )
    return {
        "available": bool(rows),
        "train_prefixes": int(train_x.shape[0]),
        "holdout_prefixes": int(hold_x.shape[0]),
        "candidate_inits_considered": candidate_rows,
        "training_budget_estimated_steps": int(training_budget.get("estimated_steps", 0)),
        "rows": rows,
        "best_candidate": best_row,
        "best_by_holdout_bpb": best_by_bpb,
    }, best_export


def build_student_teacher_recipe(
    *,
    student_teacher_fit: dict[str, object],
    training_budget: dict[str, object],
) -> dict[str, object]:
    best = (student_teacher_fit.get("best_by_holdout_bpb") or student_teacher_fit.get("best_candidate")) if isinstance(student_teacher_fit, dict) else None
    if not isinstance(best, dict):
        return {"available": False, "reason": "missing_teacher_fit_candidate"}
    estimated_steps = int(training_budget.get("estimated_steps", 0))
    fit_steps = int(best.get("teacher_fit_steps", 0))
    return {
        "available": True,
        "decoder_module_family": str(best.get("module_family")),
        "decoder_hidden_rank": int(best.get("hidden_rank", 0)),
        "decoder_init_seed": int(best.get("seed", 0)),
        "teacher_fit_steps": fit_steps,
        "teacher_fit_fraction_of_competition_budget": float(fit_steps / max(estimated_steps, 1)),
        "learning_rate": float(best.get("learning_rate", 0.0)),
        "weight_decay": float(best.get("weight_decay", 0.0)),
        "state_loss_coeff": float(best.get("state_loss_coeff", 0.0)),
        "teacher_loss_coeff": float(best.get("teacher_loss_coeff", 0.0)),
        "estimated_fp16_state_bytes": int(best.get("estimated_fp16_state_bytes", 0)),
        "expected_holdout_teacher_kl_bits": float(best.get("holdout_teacher_kl_bits", 0.0)),
        "expected_holdout_token_bits_per_byte": None if best.get("holdout_token_bits_per_byte") is None else float(best.get("holdout_token_bits_per_byte")),
        "trainer_note": (
            "Use this as the causal student warm-start recipe: initialize the decoder with the profiled family/seed, "
            "train it first against teacher state ids and teacher state logits at the recommended loss mix, "
            "then add transition/residual losses only after decoder recovery is stable."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile parameter-golf training shards for causal/state tuning decisions.")
    parser.add_argument("--train-glob", default="./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
    parser.add_argument("--val-glob", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--max-shards", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4_000_000)
    parser.add_argument("--max-tokens-fraction-of-training-budget", type=float, default=0.0)
    parser.add_argument("--sample-across-shards", action="store_true")
    parser.add_argument("--lags", default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--reuse-windows", default="128,512,1024,2048")
    parser.add_argument("--eval-lengths", default="512,1024,1408,2048")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-wallclock-seconds", type=float, default=600.0)
    parser.add_argument("--avg-step-ms", type=float, default=85.0)
    parser.add_argument("--base-eval-step-ms", type=float, default=120.0)
    parser.add_argument("--base-eval-seq-len", type=int, default=1024)
    parser.add_argument("--base-eval-batch-seqs", type=int, default=512)
    parser.add_argument("--ttt-multipliers", default="1.5,2.0,3.0")
    parser.add_argument("--train-batch-tokens", type=int, default=524_288)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--expected-reveal-fraction", type=float, default=0.54132736)
    parser.add_argument("--reveal-sweep-fractions", default="0.45,0.50,0.52,0.54132736,0.56,0.58,0.60,0.75,1.0")
    parser.add_argument("--reveal-sweep-max-prefixes", type=int, default=8192)
    parser.add_argument(
        "--token-seen-ladder",
        default="524288,1048576,2621440,5242880,10485760,26214400,52428800,104857600,262144000,524288000,1048576000,4331667456",
    )
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
    parser.add_argument("--oracle-backoff-orders", default="0")
    parser.add_argument("--oracle-backoff-max-eval-tokens", type=int, default=65536)
    parser.add_argument("--ppm-orders", default="0")
    parser.add_argument("--ppm-max-eval-tokens", type=int, default=65536)
    parser.add_argument("--predictive-state-order", type=int, default=4)
    parser.add_argument("--predictive-state-ranks", default="4,8,16,32,64,128")
    parser.add_argument("--predictive-state-max-prefixes", type=int, default=32768)
    parser.add_argument("--world-model-chunk-tokens", type=int, default=65536)
    parser.add_argument("--world-model-chunk-stride", type=int, default=32768)
    parser.add_argument("--world-model-prefix-tokens", type=int, default=1024)
    parser.add_argument("--world-model-regime-counts", default="4,8,16")
    parser.add_argument("--world-model-kmeans-iters", type=int, default=16)
    parser.add_argument("--world-model-top-tokens", type=int, default=8)
    parser.add_argument("--init-hidden-ranks", default="32,64,128")
    parser.add_argument("--init-seeds", default="0,1,2,3")
    parser.add_argument("--init-nonlinearities", default="tanh,relu")
    parser.add_argument("--jacobian-max-samples", type=int, default=64)
    parser.add_argument("--teacher-fit-max-prefixes", type=int, default=16384)
    parser.add_argument("--teacher-fit-top-inits", type=int, default=2)
    parser.add_argument("--teacher-fit-steps", default="32,64")
    parser.add_argument("--teacher-fit-learning-rates", default="0.001,0.003")
    parser.add_argument("--teacher-fit-weight-decays", default="0.0,0.0001")
    parser.add_argument("--teacher-fit-state-loss-coeffs", default="0.5,1.0")
    parser.add_argument("--teacher-fit-teacher-loss-coeffs", default="1.0,2.0")
    parser.add_argument("--tokens-seen-target", type=int, default=0)
    parser.add_argument("--reveal-condition-fractions", default="0.50,0.52,0.54132736,0.56,0.58,1.0")
    parser.add_argument("--cache-dir", default="runs/profile_cache")
    parser.add_argument("--no-resume-cache", action="store_true")
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    args.train_glob = _resolve_repo_relative_glob(args.train_glob)
    args.val_glob = _resolve_repo_relative_glob(args.val_glob)
    if args.tokenizer_path:
        args.tokenizer_path = _resolve_repo_relative_path(args.tokenizer_path)
    args.cache_dir = _resolve_repo_relative_path(args.cache_dir)
    if args.output_json:
        args.output_json = _resolve_repo_relative_path(args.output_json)

    files = base.resolve_files(args.train_glob, args.max_shards)
    val_files = base.resolve_files(args.val_glob, 0)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resume_cache = not bool(args.no_resume_cache)
    profile_workers = max(int(args.profile_workers), 1)

    train_stream = base.shard_token_totals(files)
    val_stream = base.shard_token_totals(val_files)
    training_budget = base.estimate_training_budget(
        max_wallclock_seconds=args.max_wallclock_seconds,
        avg_step_ms=args.avg_step_ms,
        train_batch_tokens=args.train_batch_tokens,
        train_seq_len=args.train_seq_len,
        world_size=args.world_size,
        grad_accum_steps=args.grad_accum_steps,
    )
    requested_max_tokens = int(args.max_tokens)
    effective_max_tokens = requested_max_tokens
    derived_budget_tokens = None
    if float(args.max_tokens_fraction_of_training_budget) > 0.0:
        derived_budget_tokens = int(
            max(
                1,
                round(
                    float(training_budget["estimated_train_tokens"])
                    * float(args.max_tokens_fraction_of_training_budget)
                ),
            )
        )
        effective_max_tokens = max(requested_max_tokens, derived_budget_tokens) if requested_max_tokens > 0 else derived_budget_tokens
    if effective_max_tokens > 0:
        effective_max_tokens = min(int(effective_max_tokens), int(train_stream["total_tokens"]))

    sequential_tokens, sequential_cache = base.materialize_token_stream_memmap(
        files,
        effective_max_tokens,
        cache_dir,
        "train_sequential",
        sample_across_shards_mode=False,
    )
    val_sequential_tokens, val_cache = base.materialize_token_stream_memmap(
        val_files,
        effective_max_tokens,
        cache_dir,
        "val_sequential",
        sample_across_shards_mode=False,
    )
    if args.sample_across_shards:
        distribution_tokens, distribution_cache = base.materialize_token_stream_memmap(
            files,
            effective_max_tokens,
            cache_dir,
            "train_distribution",
            sample_across_shards_mode=True,
        )
    else:
        distribution_tokens = sequential_tokens
        distribution_cache = {
            "cache_key": sequential_cache["cache_key"],
            "path": sequential_cache["path"],
            "cache_hit": sequential_cache["cache_hit"],
            "sample_across_shards_mode": False,
        }

    token_counts = np.bincount(distribution_tokens, minlength=args.vocab_size).astype(np.int64)
    entropy_bits = base.entropy_from_counts(token_counts)
    effective_vocab = float(2.0**entropy_bits)
    base_bytes = base.build_base_bytes(args.tokenizer_path, args.vocab_size) if args.tokenizer_path else None
    tokenizer_meta = base.build_tokenizer_metadata(args.tokenizer_path, args.vocab_size) if args.tokenizer_path else None
    bytes_per_token = None if base_bytes is None else float(base_bytes[distribution_tokens].mean())

    bundle_results, cache_summary = _run_bundles(
        _bundle_specs(args, sequential_cache, token_counts, base_bytes, tokenizer_meta),
        cache_dir=cache_dir,
        resume_cache=resume_cache,
        profile_workers=profile_workers,
    )
    cache_summary["token_streams"] = {
        "train_sequential": sequential_cache,
        "val_sequential": val_cache,
        "distribution": distribution_cache,
    }

    basic_bundle = bundle_results["basic_signal"]
    state_bundle = bundle_results["state"]

    lag_stats = basic_bundle["lag_stats"]
    reuse_stats = basic_bundle["reuse_stats"]
    eval_candidates = basic_bundle["eval_candidates"]
    marginal_context = basic_bundle["marginal_context"]
    context_bands = basic_bundle["context_bands"]
    transition_geometry = basic_bundle["transition_geometry"]
    recurrence_profile = basic_bundle["recurrence_profile"]
    recurrence_by_bucket = basic_bundle["recurrence_by_bucket"]

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
    causal_state_decodability = state_bundle["causal_state_decodability"]
    causal_state_decodability_arrays = state_bundle.get("causal_state_decodability_arrays", {})
    causal_state_transition_learnability = state_bundle["causal_state_transition_learnability"]
    causal_state_transition_learnability_arrays = state_bundle.get("causal_state_transition_learnability_arrays", {})
    causal_state_multi_horizon_sufficiency = state_bundle["causal_state_multi_horizon_sufficiency"]
    causal_state_merge_error = state_bundle["causal_state_merge_error"]
    causal_state_residual_geometry = state_bundle["causal_state_residual_geometry"]
    causal_state_residual_geometry_arrays = state_bundle.get("causal_state_residual_geometry_arrays", {})
    strict_online_state_eval = state_bundle["strict_online_state_eval"]
    tensor_network_state_frontier = state_bundle["tensor_network_state_frontier"]
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
    init_hidden_ranks = _safe_int_list(args.init_hidden_ranks)
    init_seeds = _safe_int_list(args.init_seeds)
    init_nonlinearities = [part.strip().lower() for part in str(args.init_nonlinearities).split(",") if part.strip()]
    teacher_fit_steps = _safe_int_list(args.teacher_fit_steps)
    teacher_fit_learning_rates = _safe_float_list(args.teacher_fit_learning_rates)
    teacher_fit_weight_decays = _safe_float_list(args.teacher_fit_weight_decays)
    teacher_fit_state_loss_coeffs = _safe_float_list(args.teacher_fit_state_loss_coeffs)
    teacher_fit_teacher_loss_coeffs = _safe_float_list(args.teacher_fit_teacher_loss_coeffs)
    reveal_condition_fractions = _bounded_fractions(args.reveal_condition_fractions, extra=[1.0])

    random_init_state_decodability, random_init_state_arrays = random_init_state_decodability_profile(
        sequential_tokens,
        vocab_size=int(args.vocab_size),
        past_window=int(args.cca_past_window),
        sketch_dim=int(args.cca_sketch_dim),
        future_signature_arrays=state_bundle["future_signature_arrays"],
        causal_state_arrays=causal_state_arrays,
        holdout_fraction=float(args.causal_state_holdout_fraction),
        hidden_ranks=init_hidden_ranks,
        seeds=init_seeds,
        nonlinearities=init_nonlinearities,
    )
    random_init_transition_alignment, random_init_transition_arrays = random_init_transition_alignment_profile(
        sequential_tokens,
        vocab_size=int(args.vocab_size),
        past_window=int(args.cca_past_window),
        sketch_dim=int(args.cca_sketch_dim),
        future_signature_arrays=state_bundle["future_signature_arrays"],
        causal_state_arrays=causal_state_arrays,
        holdout_fraction=float(args.causal_state_holdout_fraction),
        hidden_ranks=init_hidden_ranks,
        seeds=init_seeds,
        ridge=float(args.causal_state_ridge),
    )
    teacher_jacobian_condition = teacher_jacobian_condition_profile(
        sequential_tokens,
        vocab_size=int(args.vocab_size),
        past_window=int(args.cca_past_window),
        sketch_dim=int(args.cca_sketch_dim),
        future_signature_arrays=state_bundle["future_signature_arrays"],
        causal_state_arrays=causal_state_arrays,
        state_init_arrays=random_init_state_arrays,
        transition_init_arrays=random_init_transition_arrays,
        max_samples=int(args.jacobian_max_samples),
    )
    optimizer_reachability = optimizer_reachability_profile(
        training_budget=training_budget,
        random_init_state_decodability=random_init_state_decodability,
        random_init_transition_alignment=random_init_transition_alignment,
        teacher_jacobian_condition=teacher_jacobian_condition,
    )
    state_teacher_curriculum = state_teacher_curriculum_profile(
        causal_state_decodability=causal_state_decodability,
        causal_state_transition_learnability=causal_state_transition_learnability,
        causal_state_residual_geometry=causal_state_residual_geometry,
        random_init_state_decodability=random_init_state_decodability,
        random_init_transition_alignment=random_init_transition_alignment,
        num_states=int(np.asarray(causal_state_arrays.get("causal_machine_log_probs")).shape[0]) if np.asarray(causal_state_arrays.get("causal_machine_log_probs")).size > 0 else 0,
    )
    online_recovery_gap = online_recovery_gap_profile(
        sequential_tokens,
        vocab_size=int(args.vocab_size),
        past_window=int(args.cca_past_window),
        sketch_dim=int(args.cca_sketch_dim),
        future_signature_arrays=state_bundle["future_signature_arrays"],
        causal_state_arrays=causal_state_arrays,
        causal_state_decodability_arrays=causal_state_decodability_arrays,
        causal_state_transition_learnability_arrays=causal_state_transition_learnability_arrays,
        causal_state_residual_geometry_arrays=causal_state_residual_geometry_arrays,
        strict_online_state_eval=strict_online_state_eval,
        causal_state_reconstruction=causal_state_reconstruction,
        state_entropy_floor=state_entropy_floor,
        holdout_fraction=float(args.causal_state_holdout_fraction),
        ridge=float(args.causal_state_ridge),
        base_bytes=base_bytes,
    )
    init_frontier = init_frontier_profile(
        random_init_state_decodability=random_init_state_decodability,
        random_init_transition_alignment=random_init_transition_alignment,
        teacher_jacobian_condition=teacher_jacobian_condition,
        optimizer_reachability=optimizer_reachability,
    )
    student_teacher_fit, student_teacher_fit_arrays = student_teacher_fit_profile(
        sequential_tokens,
        vocab_size=int(args.vocab_size),
        base_bytes=base_bytes,
        past_window=int(args.cca_past_window),
        sketch_dim=int(args.cca_sketch_dim),
        future_signature_arrays=state_bundle["future_signature_arrays"],
        causal_state_arrays=causal_state_arrays,
        holdout_fraction=float(args.causal_state_holdout_fraction),
        random_init_state_decodability=random_init_state_decodability,
        max_prefixes=int(args.teacher_fit_max_prefixes),
        top_inits=int(args.teacher_fit_top_inits),
        fit_steps=teacher_fit_steps,
        learning_rates=teacher_fit_learning_rates,
        weight_decays=teacher_fit_weight_decays,
        state_loss_coeffs=teacher_fit_state_loss_coeffs,
        teacher_loss_coeffs=teacher_fit_teacher_loss_coeffs,
        training_budget=training_budget,
    )
    student_teacher_recipe = build_student_teacher_recipe(
        student_teacher_fit=student_teacher_fit,
        training_budget=training_budget,
    )
    reveal_conditioned_student_recipes = build_reveal_conditioned_student_recipe_table(
        sequential_tokens,
        vocab_size=int(args.vocab_size),
        base_bytes=base_bytes,
        training_budget=training_budget,
        tokens_seen_target=int(args.tokens_seen_target),
        reveal_fractions=reveal_condition_fractions,
        past_window=int(args.cca_past_window),
        sketch_dim=int(args.cca_sketch_dim),
        holdout_fraction=float(args.causal_state_holdout_fraction),
        init_hidden_ranks=init_hidden_ranks,
        init_seeds=init_seeds,
        init_nonlinearities=init_nonlinearities,
        jacobian_max_samples=int(args.jacobian_max_samples),
        teacher_fit_max_prefixes=int(args.teacher_fit_max_prefixes),
        teacher_fit_top_inits=int(args.teacher_fit_top_inits),
        teacher_fit_steps=teacher_fit_steps,
        teacher_fit_learning_rates=teacher_fit_learning_rates,
        teacher_fit_weight_decays=teacher_fit_weight_decays,
        teacher_fit_state_loss_coeffs=teacher_fit_state_loss_coeffs,
        teacher_fit_teacher_loss_coeffs=teacher_fit_teacher_loss_coeffs,
        profile_cache_dir=cache_dir,
    )

    training_coverage = base.estimate_training_coverage(
        total_train_tokens_available=int(train_stream["total_tokens"]),
        training_budget=training_budget,
    )
    eval_budget_frontier = base.estimate_eval_budget_frontier(
        eval_lengths=base.parse_int_list(args.eval_lengths),
        eval_batch_seqs=args.base_eval_batch_seqs,
        max_wallclock_seconds=args.max_wallclock_seconds,
        base_eval_seq_len=args.base_eval_seq_len,
        base_eval_step_ms=args.base_eval_step_ms,
        ttt_multipliers=base.parse_float_list(args.ttt_multipliers),
    )
    real_eval_frontier = base.estimate_real_eval_budget_frontier(
        total_val_tokens=int(val_stream["total_tokens"]),
        eval_lengths=base.parse_int_list(args.eval_lengths),
        eval_stride=64,
        eval_batch_seqs=args.base_eval_batch_seqs,
        max_wallclock_seconds=args.max_wallclock_seconds,
        base_eval_seq_len=args.base_eval_seq_len,
        base_eval_batch_seqs=args.base_eval_batch_seqs,
        base_eval_step_ms=args.base_eval_step_ms,
        ttt_multipliers=base.parse_float_list(args.ttt_multipliers),
    )
    candidate_state_frontier = base.candidate_state_frontier_profile(
        None,
        strict_online_state_eval,
        minimal_causal_state,
        minimal_causal_state_reco,
        causal_state_reconstruction,
        tensor_network_state_frontier,
        ppm_oracle,
        oracle_backoff,
        regime_conditioned_bpb,
    )
    loss_shape = base.loss_geometry_surface(
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
    capacity_inference = build_causal_capacity_inference(
        lag_stats=lag_stats,
        minimal_causal_state_reco=minimal_causal_state_reco,
        predictive_state_recommendation=predictive_state_recommendation,
        training_budget=training_budget,
    )
    reveal_analysis = build_reveal_sweep_analysis(
        sequential_tokens,
        vocab_size=int(args.vocab_size),
        base_bytes=base_bytes,
        training_budget=training_budget,
        expected_reveal_fraction=float(args.expected_reveal_fraction),
        reveal_fractions=_bounded_fractions(
            args.reveal_sweep_fractions,
            extra=[float(args.expected_reveal_fraction), 1.0],
        ),
        lags=base.parse_int_list(args.lags),
        cca_past_window=int(args.cca_past_window),
        cca_future_window=int(args.cca_future_window),
        cca_sketch_dim=int(args.cca_sketch_dim),
        cca_regularization=float(args.cca_regularization),
        causal_state_ranks=(
            base.parse_int_list(args.causal_state_ranks)
            if args.causal_state_ranks.strip()
            else base.parse_int_list(args.cca_ranks)
        ),
        causal_state_holdout_fraction=float(args.causal_state_holdout_fraction),
        causal_state_ridge=float(args.causal_state_ridge),
        causal_state_near_best_bpb_tol=float(args.causal_state_near_best_bpb_tol),
        causal_state_near_best_bits_tol=float(args.causal_state_near_best_bits_tol),
        predictive_state_order=int(args.predictive_state_order),
        predictive_state_ranks=base.parse_int_list(args.predictive_state_ranks),
        reveal_max_prefixes=int(args.reveal_sweep_max_prefixes),
    )
    token_seen_ladder = build_token_seen_ladder_analysis(
        sequential_tokens,
        vocab_size=int(args.vocab_size),
        base_bytes=base_bytes,
        training_budget=training_budget,
        total_train_tokens_available=int(train_stream["total_tokens"]),
        token_seen_points=_safe_int_list(args.token_seen_ladder),
        lags=base.parse_int_list(args.lags),
        cca_past_window=int(args.cca_past_window),
        cca_future_window=int(args.cca_future_window),
        cca_sketch_dim=int(args.cca_sketch_dim),
        cca_regularization=float(args.cca_regularization),
        causal_state_ranks=(
            base.parse_int_list(args.causal_state_ranks)
            if args.causal_state_ranks.strip()
            else base.parse_int_list(args.cca_ranks)
        ),
        causal_state_holdout_fraction=float(args.causal_state_holdout_fraction),
        causal_state_ridge=float(args.causal_state_ridge),
        causal_state_near_best_bpb_tol=float(args.causal_state_near_best_bpb_tol),
        causal_state_near_best_bits_tol=float(args.causal_state_near_best_bits_tol),
        predictive_state_order=int(args.predictive_state_order),
        predictive_state_ranks=base.parse_int_list(args.predictive_state_ranks),
        reveal_max_prefixes=int(args.reveal_sweep_max_prefixes),
    )
    competition_target = build_competition_target_recommendation(
        reveal_analysis=reveal_analysis,
        training_budget=training_budget,
        eval_candidates=eval_candidates,
    )
    full_stream_reference = build_full_stream_reference(
        lag_stats=lag_stats,
        capacity_inference=capacity_inference,
        minimal_causal_state_reco=minimal_causal_state_reco,
        predictive_state_recommendation=predictive_state_recommendation,
        predictive_state_transition_recommendation=predictive_state_transition_recommendation,
        spectral_recommendation=spectral_recommendation,
        dataset_world_model_reco=dataset_world_model_reco,
        candidate_state_frontier=candidate_state_frontier,
        total_train_tokens_available=int(train_stream["total_tokens"]),
        profiled_tokens=int(sequential_tokens.size),
    )
    delta_to_full_stream = build_delta_to_full_stream(
        token_seen_ladder=token_seen_ladder,
        full_stream_reference=full_stream_reference,
    )
    competition_architecture_tradeoff = build_competition_architecture_tradeoff(
        training_budget=training_budget,
        expected_reveal_fraction=float(args.expected_reveal_fraction),
        token_seen_ladder=token_seen_ladder,
        delta_to_full_stream=delta_to_full_stream,
        competition_target=competition_target,
    )
    recommendations = build_causal_recommendations(
        lag_stats=lag_stats,
        spectral_recommendation=spectral_recommendation,
        minimal_causal_state_reco=minimal_causal_state_reco,
        predictive_state_recommendation=predictive_state_recommendation,
        predictive_state_transition_recommendation=predictive_state_transition_recommendation,
        dataset_world_model_reco=dataset_world_model_reco,
        candidate_state_frontier=candidate_state_frontier,
    )

    profile = {
        "schema_version": SCHEMA_VERSION,
        "source_script": str(Path(__file__).resolve()),
        "derived_from": str((Path(__file__).with_name("profile_training_dataset.py")).resolve()),
        "focus": "causal_state_and_next_token_optimization",
        "omitted_sections": [
            "lexical_entropy_profile",
            "hashed_lexical_collision_profile",
            "recent_copy_window_profile",
            "lexical_control_profile",
            "train_val_transfer_profile",
            "confidence_route_budget_profile",
            "route_calibration_selectivity_profile",
            "objective_lane_analysis",
            "top_bigrams",
        ],
        "train_glob": args.train_glob,
        "val_glob": args.val_glob,
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
            "gini": base.gini_from_counts(token_counts),
            "top_tokens": base.top_tokens(token_counts, args.top_k),
            "bytes_per_token": bytes_per_token,
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
        "transition_geometry": transition_geometry,
        "recurrence_burst_profile": recurrence_profile,
        "recurrence_burst_profile_by_frequency": recurrence_by_bucket,
        "spectral_eigenbases": spectral_eigenbases,
        "lagged_spectral_eigenbases": lagged_spectral_eigenbases,
        "spectral_basis_recommendations": spectral_recommendation,
        "past_future_cca_profile": past_future_cca,
        "past_future_cca_recommendations": past_future_cca_reco,
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
        "random_init_state_decodability_profile": random_init_state_decodability,
        "random_init_transition_alignment_profile": random_init_transition_alignment,
        "teacher_jacobian_condition_profile": teacher_jacobian_condition,
        "optimizer_reachability_profile": optimizer_reachability,
        "state_teacher_curriculum_profile": state_teacher_curriculum,
        "online_recovery_gap_profile": online_recovery_gap,
        "init_frontier": init_frontier,
        "student_teacher_fit_profile": student_teacher_fit,
        "student_teacher_recipe": student_teacher_recipe,
        "reveal_conditioned_student_recipe_table": reveal_conditioned_student_recipes,
        "strict_online_state_eval": strict_online_state_eval,
        "tensor_network_state_frontier": tensor_network_state_frontier,
        "candidate_state_frontier": candidate_state_frontier,
        "predictive_state_compression": predictive_state_compression,
        "predictive_state_recommendations": predictive_state_recommendation,
        "predictive_state_transition_profile": predictive_state_transition,
        "predictive_state_transition_recommendations": predictive_state_transition_recommendation,
        "predictive_state_transfer_spectrum": predictive_state_transfer,
        "oracle_backoff_profile": oracle_backoff,
        "ppm_oracle_profile": ppm_oracle,
        "ppm_oracle_recommendations": ppm_oracle_reco,
        "dataset_world_model": dataset_world_model,
        "dataset_world_model_recommendations": dataset_world_model_reco,
        "regime_conditioned_bpb": regime_conditioned_bpb,
        "eval_length_candidates": eval_candidates,
        "eval_budget_frontier": eval_budget_frontier,
        "real_eval_budget_frontier": real_eval_frontier,
        "training_budget_estimate": training_budget,
        "training_coverage_estimate": training_coverage,
        "loss_geometry_surface": loss_shape,
        "causal_capacity_inference": capacity_inference,
        "full_stream_reference": full_stream_reference,
        "budget_prefix_trajectory": token_seen_ladder,
        "delta_to_full_stream": delta_to_full_stream,
        "competition_architecture_tradeoff": competition_architecture_tradeoff,
        "token_seen_ladder_analysis": token_seen_ladder,
        "competition_reveal_analysis": reveal_analysis,
        "competition_target_recommendation": competition_target,
        "recommendations": recommendations,
    }

    out_path = Path(args.output_json) if args.output_json else Path("runs/dataset_profile_causal.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2))

    sidecar_arrays = {
        **spectral_eigenbasis_arrays,
        **lagged_spectral_eigenbasis_arrays,
        **past_future_cca_arrays,
        **minimal_causal_state_arrays,
        **causal_state_arrays,
        **causal_state_decodability_arrays,
        **causal_state_transition_learnability_arrays,
        **causal_state_residual_geometry_arrays,
        **random_init_state_arrays,
        **random_init_transition_arrays,
        **student_teacher_fit_arrays,
        **predictive_state_arrays,
        **dataset_world_model_arrays,
        **regime_conditioned_bpb_arrays,
    }
    if sidecar_arrays:
        sidecar_path = out_path.with_name(f"{out_path.stem}_causal_state_sidecar.npz")
        np.savez_compressed(sidecar_path, **sidecar_arrays)
        profile["causal_state_sidecar_npz"] = str(sidecar_path)
        out_path.write_text(json.dumps(profile, indent=2))

    print(f"Causal dataset profile written to {out_path}")
    if "causal_state_sidecar_npz" in profile:
        print(f"Causal sidecar written to {profile['causal_state_sidecar_npz']}")


if __name__ == "__main__":
    main()
