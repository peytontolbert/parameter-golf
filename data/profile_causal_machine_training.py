#!/usr/bin/env python3
"""
Budgeted training trajectory profiler for the causal state-space trainer.

This script launches a small ladder of short wallclock-capped training runs,
parses the trainer logs, and estimates which configuration is most promising at
the target time budget. It is intentionally separate from the offline dataset
profilers: this one measures actual learning/runtime trajectory, while the
offline profilers measure dataset structure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+)\s+"
    r"(?:(?:val_mode:\S+\s+)?val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?)\s+val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?)|"
    r"train_loss:(?P<train_loss>[-+]?\d+(?:\.\d+)?))\s+"
    r"train_time:(?P<train_time_ms>\d+)ms\s+step_avg:(?P<step_avg_ms>[-+]?\d+(?:\.\d+)?)ms"
)


@dataclass
class StepPoint:
    step: int
    total_steps: int
    train_time_ms: int
    step_avg_ms: float
    train_loss: float | None = None
    val_loss: float | None = None
    val_bpb: float | None = None


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def parse_assignment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Missing key in assignment: {raw}")
    return key, value


def parse_sweep_assignment(raw: str) -> tuple[str, list[str]]:
    key, value = parse_assignment(raw)
    sep = "|" if "|" in value else ","
    values = [item.strip() for item in value.split(sep) if item.strip()]
    if not values:
        raise ValueError(f"Missing values in sweep assignment: {raw}")
    return key, values


def infer_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for parent in resolved.parents:
        if (parent / "data").is_dir() and (parent / "records").is_dir():
            return parent
    raise RuntimeError(f"Unable to infer repo root from trainer path: {script_path}")


def parse_log_steps(path: Path) -> list[StepPoint]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    steps: list[StepPoint] = []
    current_segment: list[StepPoint] = []
    best_segment: list[StepPoint] = []
    for line in lines:
        match = STEP_RE.match(line)
        if not match:
            continue
        point = StepPoint(
            step=int(match.group("step")),
            total_steps=int(match.group("total")),
            train_time_ms=int(match.group("train_time_ms")),
            step_avg_ms=float(match.group("step_avg_ms")),
            train_loss=float(match.group("train_loss")) if match.group("train_loss") else None,
            val_loss=float(match.group("val_loss")) if match.group("val_loss") else None,
            val_bpb=float(match.group("val_bpb")) if match.group("val_bpb") else None,
        )
        # A train line and a val line can legitimately share the same step index.
        # Only treat strictly decreasing step numbers as a new run segment.
        if current_segment and point.step < current_segment[-1].step:
            if len(current_segment) > len(best_segment):
                best_segment = current_segment
            current_segment = []
        current_segment.append(point)
        steps.append(point)
    if len(current_segment) > len(best_segment):
        best_segment = current_segment
    return best_segment or steps


def fit_linear(xs: list[float], ys: list[float]) -> dict[str, float] | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx <= 0.0:
        return None
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
    rmse = math.sqrt(sum(res * res for res in residuals) / len(residuals))
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "rmse": float(rmse),
    }


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def safe_log10(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= 0.0:
        return None
    return math.log10(value)


def steady_step_ms(train_points: list[StepPoint]) -> float | None:
    if len(train_points) < 3:
        return None
    tail = train_points[max(1, len(train_points) // 3) :]
    deltas = [
        float(curr.train_time_ms - prev.train_time_ms)
        for prev, curr in zip(tail[:-1], tail[1:], strict=True)
        if curr.train_time_ms > prev.train_time_ms
    ]
    if not deltas:
        return None
    return float(statistics.median(deltas))


def project_bpb(points: list[tuple[float, float]], target_tokens: float) -> dict[str, float] | None:
    points = sorted(points, key=lambda item: item[0])
    if len(points) >= 3:
        floor_projection = project_bpb_with_floor(points, target_tokens)
        if floor_projection is not None:
            return floor_projection
    xs: list[float] = []
    ys: list[float] = []
    for tokens_seen, val_bpb in points:
        x = safe_log10(tokens_seen)
        if x is None:
            continue
        xs.append(x)
        ys.append(float(val_bpb))
    fit = fit_linear(xs, ys)
    if fit is None:
        return None
    target_x = safe_log10(target_tokens)
    max_tokens_seen = max(tokens for tokens, _ in points) if points else 0.0
    seen_x = safe_log10(max_tokens_seen)
    if target_x is None or seen_x is None:
        return None
    projected = fit["intercept"] + fit["slope"] * target_x
    observed = fit["intercept"] + fit["slope"] * seen_x
    improvement = max(0.0, observed - projected)
    capped_improvement = min(improvement, 0.12)
    center = float(ys[-1] - capped_improvement)
    spread = max(0.02, fit["rmse"] * 1.5 + 0.04 * max(0.0, target_x - seen_x))
    return {
        "mid": center,
        "low": center - spread,
        "high": center + spread,
        "spread": spread,
        "fit_kind": "log_linear",
    }


def project_bpb_with_floor(points: list[tuple[float, float]], target_tokens: float) -> dict[str, float] | None:
    if len(points) < 3:
        return None
    sorted_points = sorted(points, key=lambda item: item[0])
    ys = [float(val_bpb) for _, val_bpb in sorted_points]
    min_y = min(ys)
    max_y = max(ys)
    if max_y - min_y <= 1e-6:
        return {
            "mid": min_y,
            "low": min_y,
            "high": min_y,
            "spread": 0.0,
            "fit_kind": "flat_floor",
            "floor": min_y,
            "alpha": 0.0,
            "amplitude": 0.0,
        }

    best_fit: dict[str, float] | None = None
    floor_candidates: list[float] = []
    floor_low = max(min_y - 0.25, 0.0)
    floor_high = min_y - 0.01
    if floor_high > floor_low:
        for i in range(24):
            frac = i / 23.0 if 23 > 0 else 0.0
            floor_candidates.append(floor_low + frac * (floor_high - floor_low))
    floor_candidates.append(max(min_y - 0.005, 0.0))

    for floor in floor_candidates:
        transformed_xs: list[float] = []
        transformed_ys: list[float] = []
        valid = True
        for tokens_seen, val_bpb in sorted_points:
            margin = float(val_bpb) - floor
            if margin <= 1e-6:
                valid = False
                break
            x = safe_log10(tokens_seen)
            y = safe_log10(margin)
            if x is None or y is None:
                valid = False
                break
            transformed_xs.append(x)
            transformed_ys.append(y)
        if not valid:
            continue
        fit = fit_linear(transformed_xs, transformed_ys)
        if fit is None:
            continue
        alpha = -fit["slope"]
        if alpha <= 0.0:
            continue
        amplitude_log10 = fit["intercept"]
        amplitude = 10.0 ** amplitude_log10
        residual_rmse = fit["rmse"]
        score = residual_rmse + 0.02 * max(floor - floor_low, 0.0)
        if best_fit is None or score < best_fit["score"]:
            best_fit = {
                "score": score,
                "floor": floor,
                "alpha": alpha,
                "amplitude": amplitude,
                "rmse": residual_rmse,
            }

    if best_fit is None:
        return None

    projected = best_fit["floor"] + best_fit["amplitude"] * (float(target_tokens) ** (-best_fit["alpha"]))
    max_tokens_seen = max(tokens for tokens, _ in sorted_points)
    seen_projected = best_fit["floor"] + best_fit["amplitude"] * (float(max_tokens_seen) ** (-best_fit["alpha"]))
    observed_best = min(ys)
    center = min(float(projected), observed_best, seen_projected)
    spread = max(0.02, best_fit["rmse"] * 2.0)
    low = max(best_fit["floor"], center - spread)
    high = center + spread
    return {
        "mid": center,
        "low": low,
        "high": high,
        "spread": spread,
        "fit_kind": "power_law_floor",
        "floor": best_fit["floor"],
        "alpha": best_fit["alpha"],
        "amplitude": best_fit["amplitude"],
    }


def make_variants(base_env: dict[str, str], sweep_assignments: list[tuple[str, list[str]]]) -> list[tuple[str, dict[str, str]]]:
    if not sweep_assignments:
        return [("base", dict(base_env))]
    keys = [key for key, _ in sweep_assignments]
    values_product = itertools.product(*(values for _, values in sweep_assignments))
    variants: list[tuple[str, dict[str, str]]] = []
    for combo in values_product:
        env = dict(base_env)
        label_parts: list[str] = []
        for key, value in zip(keys, combo, strict=True):
            env[key] = value
            label_parts.append(f"{key.lower()}={value}")
        variants.append(("__".join(label_parts), env))
    return variants


def expand_variants_with_seeds(
    variants: list[tuple[str, dict[str, str]]],
    seed_env_name: str,
    seed_list: list[int],
) -> list[tuple[str, str, int | None, dict[str, str]]]:
    expanded: list[tuple[str, str, int | None, dict[str, str]]] = []
    if not seed_list:
        for variant_name, variant_env in variants:
            expanded.append((variant_name, variant_name, None, dict(variant_env)))
        return expanded
    for variant_name, variant_env in variants:
        family_name = variant_name
        for seed in seed_list:
            env = dict(variant_env)
            env[seed_env_name] = str(seed)
            expanded.append((family_name, f"{family_name}__seed={seed}", int(seed), env))
    return expanded


def summarize_run(steps: list[StepPoint], train_batch_tokens: int) -> dict[str, Any]:
    train_points = [point for point in steps if point.train_loss is not None]
    val_points = [point for point in steps if point.val_bpb is not None]
    logged_train_time_ms = int(steps[-1].train_time_ms) if steps else 0
    observed_train_tokens = int(train_points[-1].step * train_batch_tokens) if train_points else 0
    final_val = val_points[-1] if val_points else None
    best_val = min(val_points, key=lambda point: float(point.val_bpb)) if val_points else None
    trajectory_points = [
        {
            "step": point.step,
            "train_time_ms": point.train_time_ms,
            "tokens_seen": int(point.step * train_batch_tokens),
            "val_loss": point.val_loss,
            "val_bpb": point.val_bpb,
        }
        for point in val_points
    ]
    return {
        "steps_completed": int(train_points[-1].step) if train_points else 0,
        "logged_train_time_ms": logged_train_time_ms,
        "observed_train_tokens": observed_train_tokens,
        "steady_step_ms": steady_step_ms(train_points),
        "final_train_loss": train_points[-1].train_loss if train_points else None,
        "final_val_loss": final_val.val_loss if final_val else None,
        "final_val_bpb": final_val.val_bpb if final_val else None,
        "best_val_loss": best_val.val_loss if best_val else None,
        "best_val_bpb": best_val.val_bpb if best_val else None,
        "trajectory_points": trajectory_points,
    }


def build_command(torchrun_bin: str, nproc_per_node: int, trainer_path: Path) -> list[str]:
    return [
        torchrun_bin,
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(trainer_path),
    ]


def build_family_summaries(summary_variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for variant in summary_variants:
        family_name = str(variant.get("family_name") or variant.get("variant_name"))
        families.setdefault(family_name, []).append(variant)
    family_rows: list[dict[str, Any]] = []
    for family_name, variants in families.items():
        projected_mids = [
            float((variant.get("projected_target_bpb") or {}).get("mid"))
            for variant in variants
            if (variant.get("projected_target_bpb") or {}).get("mid") is not None
        ]
        health_scores = [
            float((variant.get("health_metrics") or {}).get("health_score"))
            for variant in variants
            if (variant.get("health_metrics") or {}).get("health_score") is not None
        ]
        projection_confidences = [
            float((variant.get("health_metrics") or {}).get("projection_confidence"))
            for variant in variants
            if (variant.get("health_metrics") or {}).get("projection_confidence") is not None
        ]
        best_variant = min(
            variants,
            key=lambda item: float((item.get("health_metrics") or {}).get("health_score") or math.inf),
        )
        family_rows.append(
            {
                "family_name": family_name,
                "seed_count": len(variants),
                "mean_projected_target_bpb": float(statistics.fmean(projected_mids)) if projected_mids else None,
                "std_projected_target_bpb": float(statistics.pstdev(projected_mids)) if len(projected_mids) >= 2 else 0.0 if projected_mids else None,
                "mean_health_score": float(statistics.fmean(health_scores)) if health_scores else None,
                "std_health_score": float(statistics.pstdev(health_scores)) if len(health_scores) >= 2 else 0.0 if health_scores else None,
                "mean_projection_confidence": float(statistics.fmean(projection_confidences)) if projection_confidences else None,
                "best_variant_name": best_variant.get("variant_name"),
                "best_variant_health_score": (best_variant.get("health_metrics") or {}).get("health_score"),
                "best_variant_projected_target_bpb": (best_variant.get("projected_target_bpb") or {}).get("mid"),
            }
        )
    family_rows.sort(
        key=lambda row: (
            float(row["mean_health_score"]) if row.get("mean_health_score") is not None else math.inf,
            float(row["mean_projected_target_bpb"]) if row.get("mean_projected_target_bpb") is not None else math.inf,
        )
    )
    return family_rows


def build_overall_state(summary: dict[str, Any]) -> dict[str, Any]:
    variants = summary.get("variants") or []
    families = build_family_summaries(variants)
    sorted_variants = sorted(
        variants,
        key=lambda row: float((row.get("health_metrics") or {}).get("health_score") or math.inf),
    )
    top_variants = []
    for row in sorted_variants[:5]:
        top_variants.append(
            {
                "variant_name": row.get("variant_name"),
                "family_name": row.get("family_name"),
                "seed": row.get("seed"),
                "health_score": (row.get("health_metrics") or {}).get("health_score"),
                "projected_target_bpb": (row.get("projected_target_bpb") or {}).get("mid"),
                "latest_curve_tokens": (row.get("health_metrics") or {}).get("latest_curve_tokens"),
                "projection_confidence": (row.get("health_metrics") or {}).get("projection_confidence"),
            }
        )
    important_trajectories = []
    for row in sorted_variants[:3]:
        curve = row.get("trajectory_curve") or []
        important_trajectories.append(
            {
                "variant_name": row.get("variant_name"),
                "family_name": row.get("family_name"),
                "seed": row.get("seed"),
                "health_score": (row.get("health_metrics") or {}).get("health_score"),
                "projected_target_bpb": (row.get("projected_target_bpb") or {}).get("mid"),
                "fit_kind": (row.get("projected_target_bpb") or {}).get("fit_kind"),
                "trajectory_tail": curve[-8:],
            }
        )
    completed_run_count = 0
    failed_run_count = 0
    running_run_count = 0
    for row in variants:
        for run in row.get("runs") or []:
            status = run.get("status")
            if status == "completed":
                completed_run_count += 1
            elif status == "failed":
                failed_run_count += 1
            elif status == "running":
                running_run_count += 1
    return {
        "family_count": len(families),
        "variant_count": len(variants),
        "completed_run_count": completed_run_count,
        "failed_run_count": failed_run_count,
        "running_run_count": running_run_count,
        "top_variants": top_variants,
        "top_families": families[:5],
        "important_trajectories": important_trajectories,
    }


def write_summary(output_path: Path, summary: dict[str, Any]) -> None:
    summary["variant_families"] = build_family_summaries(summary.get("variants") or [])
    summary["state"] = build_overall_state(summary)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=False), encoding="utf-8")


def load_existing_summary(output_path: Path) -> dict[str, Any] | None:
    if not output_path.is_file():
        return None
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def extract_curve_points(variant_rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    by_tokens: dict[int, float] = {}
    for row in variant_rows:
        if row.get("status") != "completed":
            continue
        trajectory_points = row.get("trajectory_points")
        if isinstance(trajectory_points, list) and trajectory_points:
            for point in trajectory_points:
                if not isinstance(point, dict):
                    continue
                tokens_seen = int(point.get("tokens_seen", 0) or 0)
                val_bpb = point.get("val_bpb")
                if tokens_seen <= 0 or val_bpb is None:
                    continue
                prev = by_tokens.get(tokens_seen)
                val_bpb_f = float(val_bpb)
                by_tokens[tokens_seen] = val_bpb_f if prev is None else min(prev, val_bpb_f)
            continue
        tokens_seen = int(row.get("observed_train_tokens", 0) or 0)
        val_bpb = row.get("final_val_bpb")
        if tokens_seen <= 0 or val_bpb is None:
            continue
        prev = by_tokens.get(tokens_seen)
        val_bpb_f = float(val_bpb)
        by_tokens[tokens_seen] = val_bpb_f if prev is None else min(prev, val_bpb_f)
    if not by_tokens:
        return []
    ordered_tokens = sorted(by_tokens)
    curve_points: list[tuple[float, float]] = []
    running_best = math.inf
    for tokens_seen in ordered_tokens:
        running_best = min(running_best, float(by_tokens[tokens_seen]))
        curve_points.append((float(tokens_seen), running_best))
    return curve_points


def curve_fit_diagnostics(points: list[tuple[float, float]], target_tokens: float | None) -> dict[str, float | int | None]:
    if not points:
        return {
            "curve_point_count": 0,
            "latest_curve_tokens": 0,
            "global_slope_per_log10_token": None,
            "recent_slope_per_log10_token": None,
            "recent_curvature": None,
            "extrapolation_ratio": None,
            "log10_token_gap": None,
            "projection_confidence": 0.0,
        }
    ordered = sorted(points, key=lambda item: item[0])
    xs = [safe_log10(tokens_seen) for tokens_seen, _ in ordered]
    ys = [float(val_bpb) for _, val_bpb in ordered]
    valid_pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None]
    global_fit = None
    if len(valid_pairs) >= 2:
        global_fit = fit_linear([x for x, _ in valid_pairs], [y for _, y in valid_pairs])
    recent_pairs = valid_pairs[-4:]
    recent_fit = None
    if len(recent_pairs) >= 2:
        recent_fit = fit_linear([x for x, _ in recent_pairs], [y for _, y in recent_pairs])
    local_slopes: list[float] = []
    for (tokens_a, y_a), (tokens_b, y_b) in zip(ordered[:-1], ordered[1:], strict=True):
        x_a = safe_log10(tokens_a)
        x_b = safe_log10(tokens_b)
        if x_a is None or x_b is None or x_b <= x_a:
            continue
        local_slopes.append((float(y_b) - float(y_a)) / (x_b - x_a))
    recent_curvature = None
    if len(local_slopes) >= 2:
        recent_curvature = float(local_slopes[-1] - local_slopes[-2])
    latest_tokens = float(ordered[-1][0])
    extrapolation_ratio = None
    log10_token_gap = None
    projection_confidence = 0.0
    if target_tokens is not None and target_tokens > 0 and latest_tokens > 0:
        extrapolation_ratio = float(target_tokens) / latest_tokens
        x_latest = safe_log10(latest_tokens)
        x_target = safe_log10(float(target_tokens))
        if x_latest is not None and x_target is not None:
            log10_token_gap = max(x_target - x_latest, 0.0)
            point_factor = min(len(ordered) / 8.0, 1.0)
            gap_factor = 1.0 / (1.0 + log10_token_gap)
            projection_confidence = max(0.0, min(point_factor * gap_factor, 1.0))
    return {
        "curve_point_count": len(ordered),
        "latest_curve_tokens": int(latest_tokens),
        "global_slope_per_log10_token": float(global_fit["slope"]) if global_fit else None,
        "recent_slope_per_log10_token": float(recent_fit["slope"]) if recent_fit else None,
        "recent_curvature": recent_curvature,
        "extrapolation_ratio": extrapolation_ratio,
        "log10_token_gap": log10_token_gap,
        "projection_confidence": projection_confidence,
    }


def compute_health_metrics(
    variant_rows: list[dict[str, Any]],
    projection: dict[str, float] | None,
    target_tokens: float | None,
    curve_points: list[tuple[float, float]],
) -> dict[str, float | int | None]:
    failure_count = sum(1 for row in variant_rows if row.get("status") == "failed")
    completed_runs = [row for row in variant_rows if row.get("status") == "completed"]
    trajectory_regressions = 0
    last_val_bpb: float | None = None
    for row in completed_runs:
        trajectory_points = row.get("trajectory_points")
        if not isinstance(trajectory_points, list):
            continue
        for point in trajectory_points:
            if not isinstance(point, dict):
                continue
            val_bpb = point.get("val_bpb")
            if val_bpb is None:
                continue
            val_bpb_f = float(val_bpb)
            if last_val_bpb is not None and val_bpb_f > last_val_bpb + 0.02:
                trajectory_regressions += 1
            last_val_bpb = val_bpb_f
    latest_completed_tokens = max(
        (int(row.get("observed_train_tokens", 0) or 0) for row in completed_runs),
        default=0,
    )
    latest_final_val_bpb = None
    if completed_runs:
        best_latest_row = max(completed_runs, key=lambda row: int(row.get("observed_train_tokens", 0) or 0))
        if best_latest_row.get("final_val_bpb") is not None:
            latest_final_val_bpb = float(best_latest_row["final_val_bpb"])
    projected_mid = float(projection["mid"]) if projection else None
    projected_spread = float(projection["spread"]) if projection else None
    fit_kind = str(projection.get("fit_kind")) if projection else None
    diagnostics = curve_fit_diagnostics(curve_points, target_tokens)
    recent_slope = diagnostics.get("recent_slope_per_log10_token")
    recent_curvature = diagnostics.get("recent_curvature")
    projection_confidence = float(diagnostics.get("projection_confidence") or 0.0)
    if projected_mid is not None:
        health_score = projected_mid
    elif latest_final_val_bpb is not None:
        health_score = latest_final_val_bpb
    else:
        health_score = math.inf
    health_score += 0.05 * float(failure_count)
    health_score += 0.01 * float(trajectory_regressions)
    if projected_spread is not None:
        health_score += 0.5 * projected_spread
    if recent_slope is not None:
        health_score += 0.02 * max(float(recent_slope), -5.0)
    if recent_curvature is not None and recent_curvature > 0.0:
        health_score += 0.05 * recent_curvature
    health_score += 0.15 * (1.0 - projection_confidence)
    return {
        "health_score": health_score,
        "failure_count": failure_count,
        "trajectory_regressions": trajectory_regressions,
        "latest_completed_tokens": latest_completed_tokens,
        "latest_final_val_bpb": latest_final_val_bpb,
        "projected_mid": projected_mid,
        "projected_spread": projected_spread,
        "fit_kind": fit_kind,
        **diagnostics,
    }


def recompute_variant_metrics(
    variant_rows: list[dict[str, Any]],
    budgets: list[int],
    target_budget_seconds: int,
    target_train_tokens: int | None,
) -> tuple[float, dict[str, float] | None, float | None, list[dict[str, float]], dict[str, float | int | None]]:
    curve_points = extract_curve_points(variant_rows)
    target_tokens = 0.0
    if target_train_tokens is not None and target_train_tokens > 0:
        target_tokens = float(target_train_tokens)
    elif curve_points:
        best_row = max(variant_rows, key=lambda row: int(row.get("observed_train_tokens", 0)))
        max_budget = max(budgets)
        if max_budget > 0:
            target_tokens = float(best_row.get("observed_train_tokens", 0)) * (
                float(target_budget_seconds) / float(max_budget)
            )
    projection = project_bpb(curve_points, target_tokens) if target_tokens > 0.0 else None
    final_rank_metric = None
    final_rows = [
        row
        for row in variant_rows
        if row.get("status") == "completed" and row.get("final_val_bpb") is not None
    ]
    if final_rows:
        final_rank_metric = min(float(row["final_val_bpb"]) for row in final_rows)
    trajectory_curve = [
        {
            "tokens_seen": float(tokens_seen),
            "best_val_bpb": float(val_bpb),
        }
        for tokens_seen, val_bpb in curve_points
    ]
    health_metrics = compute_health_metrics(variant_rows, projection, target_tokens if target_tokens > 0.0 else None, curve_points)
    return target_tokens, projection, final_rank_metric, trajectory_curve, health_metrics


def update_variant_summary(
    summary: dict[str, Any],
    family_name: str,
    seed: int | None,
    variant_name: str,
    variant_env: dict[str, str],
    variant_rows: list[dict[str, Any]],
    budgets: list[int],
    target_budget_seconds: int,
    target_train_tokens: int | None,
) -> None:
    target_tokens, projection, final_rank_metric, trajectory_curve, health_metrics = recompute_variant_metrics(
        variant_rows=variant_rows,
        budgets=budgets,
        target_budget_seconds=target_budget_seconds,
        target_train_tokens=target_train_tokens,
    )
    payload = {
        "family_name": family_name,
        "seed": seed,
        "variant_name": variant_name,
        "env": dict(variant_env),
        "runs": variant_rows,
        "projection_target_tokens": target_tokens,
        "projected_target_bpb": projection,
        "best_observed_final_val_bpb": final_rank_metric,
        "trajectory_curve": trajectory_curve,
        "health_metrics": health_metrics,
    }
    variants = summary.setdefault("variants", [])
    for index, existing in enumerate(variants):
        if existing.get("variant_name") == variant_name:
            variants[index] = payload
            break
    else:
        variants.append(payload)


def find_existing_variant(existing_summary: dict[str, Any] | None, variant_name: str) -> dict[str, Any] | None:
    if not existing_summary:
        return None
    variants = existing_summary.get("variants")
    if not isinstance(variants, list):
        return None
    for variant in variants:
        if isinstance(variant, dict) and variant.get("variant_name") == variant_name:
            return variant
    return None


def launch_run(
    repo_root: Path,
    trainer_path: Path,
    torchrun_bin: str,
    nproc_per_node: int,
    run_id: str,
    budget_seconds: int,
    env_overrides: dict[str, str],
    dry_run: bool,
) -> tuple[int, Path, list[str]]:
    env = os.environ.copy()
    env.update(env_overrides)
    env["RUN_ID"] = run_id
    env["MAX_WALLCLOCK_SECONDS"] = str(budget_seconds)
    cmd = build_command(torchrun_bin, nproc_per_node, trainer_path)
    log_path = repo_root / "logs" / f"{run_id}.txt"
    if dry_run:
        return 0, log_path, cmd
    started_at = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
    elapsed_s = time.perf_counter() - started_at
    print(f"[trajectory] run_id={run_id} budget={budget_seconds}s exit={result.returncode} elapsed={elapsed_s:.1f}s")
    return result.returncode, log_path, cmd


def parse_args() -> argparse.Namespace:
    default_trainer = (
        Path(__file__).resolve().parents[1]
        / "records"
        / "track_10min_16mb"
        / "2026-03-26_StateSpace_CausalMachine"
        / "train_gpt.py"
    )
    parser = argparse.ArgumentParser(description="Profile causal-machine learning trajectory over a wallclock ladder.")
    parser.add_argument("--trainer", default=str(default_trainer))
    parser.add_argument("--torchrun-bin", default="torchrun")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--budget-seconds", default="120,240,360,600")
    parser.add_argument("--target-budget-seconds", type=int, default=600)
    parser.add_argument("--target-train-tokens", type=int, default=None)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--seed-env-name", default="SEED")
    parser.add_argument("--run-prefix", default="causal_traj")
    parser.add_argument("--set-env", action="append", default=[])
    parser.add_argument("--sweep-env", action="append", default=[])
    parser.add_argument("--output-json", default="runs/causal_machine_training_profile.json")
    parser.add_argument("--auto-prune-frac", type=float, default=0.5)
    parser.add_argument("--min-survivors", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer_path = Path(args.trainer).resolve()
    repo_root = infer_repo_root(trainer_path)
    budgets = parse_int_list(args.budget_seconds)
    if not budgets:
        raise SystemExit("No budget points provided.")

    base_env = dict(parse_assignment(item) for item in args.set_env)
    base_env.setdefault("ITERATIONS", "1000000")
    base_env.setdefault("VAL_LOSS_EVERY", "0")
    base_env.setdefault("SAVE_BEST_VAL_BPB_CHECKPOINT", "0")
    base_env.setdefault("REGRESSION_STOP_PATIENCE", "0")

    base_variants = make_variants(base_env, [parse_sweep_assignment(item) for item in args.sweep_env])
    seed_list = parse_int_list(args.seed_list) if str(args.seed_list).strip() else []
    variants = expand_variants_with_seeds(base_variants, args.seed_env_name, seed_list)
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_summary = None if args.dry_run else load_existing_summary(output_path)

    summary: dict[str, Any] = {
        "generated_at_unix": time.time(),
        "trainer": str(trainer_path),
        "repo_root": str(repo_root),
        "torchrun_bin": args.torchrun_bin,
        "nproc_per_node": int(args.nproc_per_node),
        "budget_seconds": budgets,
        "target_budget_seconds": int(args.target_budget_seconds),
        "target_train_tokens": args.target_train_tokens,
        "seed_env_name": args.seed_env_name,
        "seed_list": seed_list,
        "variants": [],
    }

    variant_states: list[dict[str, Any]] = []
    for variant_index, (family_name, variant_name, seed, variant_env) in enumerate(variants):
        existing_variant = find_existing_variant(existing_summary, variant_name)
        existing_rows_by_run_id: dict[str, dict[str, Any]] = {}
        if existing_variant:
            for row in existing_variant.get("runs", []):
                if isinstance(row, dict) and row.get("run_id"):
                    existing_rows_by_run_id[str(row["run_id"])] = dict(row)
        variant_rows: list[dict[str, Any]] = []
        print(f"[trajectory] variant={variant_name} env={variant_env}")
        update_variant_summary(
            summary=summary,
            family_name=family_name,
            seed=seed,
            variant_name=variant_name,
            variant_env=variant_env,
            variant_rows=variant_rows,
            budgets=budgets,
            target_budget_seconds=int(args.target_budget_seconds),
            target_train_tokens=args.target_train_tokens,
        )
        if not args.dry_run:
            write_summary(output_path, summary)
        variant_states.append(
            {
                "variant_index": variant_index,
                "family_name": family_name,
                "seed": seed,
                "variant_name": variant_name,
                "variant_env": variant_env,
                "variant_rows": variant_rows,
                "existing_rows_by_run_id": existing_rows_by_run_id,
                "pruned_after_budget": None,
            }
        )

    active_variant_names = {state["variant_name"] for state in variant_states}
    for budget_idx, budget_seconds in enumerate(budgets):
        for state in variant_states:
            variant_name = state["variant_name"]
            family_name = state["family_name"]
            seed = state["seed"]
            variant_env = state["variant_env"]
            variant_rows = state["variant_rows"]
            existing_rows_by_run_id = state["existing_rows_by_run_id"]
            if variant_name not in active_variant_names:
                continue
            run_id = f"{args.run_prefix}_{state['variant_index']:02d}_{budget_seconds:04d}s"
            existing_row = existing_rows_by_run_id.get(run_id)
            if existing_row and existing_row.get("status") == "completed":
                log_path_raw = existing_row.get("log_path")
                if not args.dry_run and log_path_raw:
                    log_path = Path(str(log_path_raw))
                    if log_path.is_file():
                        steps = parse_log_steps(log_path)
                        refreshed = dict(existing_row)
                        refreshed.update(
                            summarize_run(
                                steps,
                                int(variant_env.get("TRAIN_BATCH_TOKENS", os.environ.get("TRAIN_BATCH_TOKENS", "0")) or 0),
                            )
                        )
                        existing_row = refreshed
                variant_rows.append(existing_row)
                update_variant_summary(
                    summary=summary,
                    family_name=family_name,
                    seed=seed,
                    variant_name=variant_name,
                    variant_env=variant_env,
                    variant_rows=variant_rows,
                    budgets=budgets,
                    target_budget_seconds=int(args.target_budget_seconds),
                    target_train_tokens=args.target_train_tokens,
                )
                if not args.dry_run:
                    write_summary(output_path, summary)
                print(f"[trajectory] reusing completed run_id={run_id} from {output_path}")
                continue
            row: dict[str, Any] = {
                "run_id": run_id,
                "budget_seconds": int(budget_seconds),
                "status": "running",
                "started_at_unix": time.time(),
            }
            variant_rows.append(row)
            update_variant_summary(
                summary=summary,
                family_name=family_name,
                seed=seed,
                variant_name=variant_name,
                variant_env=variant_env,
                variant_rows=variant_rows,
                budgets=budgets,
                target_budget_seconds=int(args.target_budget_seconds),
                target_train_tokens=args.target_train_tokens,
            )
            if not args.dry_run:
                write_summary(output_path, summary)
            exit_code, log_path, cmd = launch_run(
                repo_root=repo_root,
                trainer_path=trainer_path,
                torchrun_bin=args.torchrun_bin,
                nproc_per_node=int(args.nproc_per_node),
                run_id=run_id,
                budget_seconds=int(budget_seconds),
                env_overrides=variant_env,
                dry_run=bool(args.dry_run),
            )
            row.update(
                {
                "run_id": run_id,
                "budget_seconds": int(budget_seconds),
                "command": cmd,
                "log_path": str(log_path),
                "exit_code": int(exit_code),
                "status": "completed" if exit_code == 0 else "failed",
                "finished_at_unix": time.time(),
                }
            )
            if not args.dry_run and log_path.is_file():
                steps = parse_log_steps(log_path)
                row.update(summarize_run(steps, int(variant_env.get("TRAIN_BATCH_TOKENS", os.environ.get("TRAIN_BATCH_TOKENS", "0")) or 0)))
            update_variant_summary(
                summary=summary,
                family_name=family_name,
                seed=seed,
                variant_name=variant_name,
                variant_env=variant_env,
                variant_rows=variant_rows,
                budgets=budgets,
                target_budget_seconds=int(args.target_budget_seconds),
                target_train_tokens=args.target_train_tokens,
            )
            if not args.dry_run:
                write_summary(output_path, summary)
        if budget_idx >= len(budgets) - 1:
            continue
        ranked_active: list[tuple[float, str]] = []
        for row in summary["variants"]:
            variant_name = row.get("variant_name")
            if variant_name not in active_variant_names:
                continue
            health = row.get("health_metrics") or {}
            latest_tokens = int(health.get("latest_completed_tokens") or 0)
            if latest_tokens <= 0:
                continue
            ranked_active.append((float(health.get("health_score") or math.inf), str(variant_name)))
        if len(ranked_active) <= max(int(args.min_survivors), 1):
            continue
        ranked_active.sort(key=lambda item: item[0])
        keep_count = max(int(args.min_survivors), math.ceil(len(ranked_active) * (1.0 - float(args.auto_prune_frac))))
        survivors = {name for _, name in ranked_active[:keep_count]}
        pruned = active_variant_names - survivors
        if pruned:
            print(
                f"[trajectory] pruning after {budget_seconds}s: "
                f"keeping {keep_count}/{len(ranked_active)} variants -> {sorted(survivors)}"
            )
            active_variant_names = survivors

    if not args.dry_run:
        summary["variants"].sort(
            key=lambda row: (
                float((row.get("health_metrics") or {}).get("health_score") or math.inf),
                float(row["projected_target_bpb"]["mid"]) if row.get("projected_target_bpb") else math.inf,
                float(row["best_observed_final_val_bpb"]) if row.get("best_observed_final_val_bpb") is not None else math.inf,
            )
        )
        write_summary(output_path, summary)
        print(f"[trajectory] wrote {output_path}")
        for row in summary["variants"]:
            proj = row.get("projected_target_bpb")
            health = row.get("health_metrics") or {}
            if proj is None:
                print(
                    f"[trajectory] {row['variant_name']}: "
                    f"health_score={float(health.get('health_score') or math.inf):.4f} "
                    f"best_final_val_bpb={row['best_observed_final_val_bpb']}"
                )
            else:
                print(
                    f"[trajectory] {row['variant_name']}: "
                    f"health_score={float(health.get('health_score') or math.inf):.4f} "
                    f"projected_{int(row['projection_target_tokens'])}_tokens_bpb={proj['mid']:.4f} "
                    f"[{proj['low']:.4f}, {proj['high']:.4f}] "
                    f"best_final_val_bpb={row['best_observed_final_val_bpb']:.4f}"
                )


if __name__ == "__main__":
    main()
