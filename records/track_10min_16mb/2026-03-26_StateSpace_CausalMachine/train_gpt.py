"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import functools
import glob
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import uuid
import zlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.cpp_extension import load as load_cpp_extension

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func

    HAS_FLASH_ATTN_3 = True
except Exception:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_3_func

        HAS_FLASH_ATTN_3 = True
    except Exception:
        flash_attn_3_func = None
        HAS_FLASH_ATTN_3 = False

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    SDPBackend = None
    sdpa_kernel = None

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    zstandard = None
    HAS_ZSTD = False

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default winner-thin run:
# - 10 transformer blocks at width 448
# - 8 attention heads with 4 KV heads (GQA), moderate hashed 4-gram lane, partial RoPE, XSA
# - tiny shared-loop tail for recurrence-like bias without broad runtime tax
# - true 2048 train/eval, EMA, late QAT, mixed export, and FA3 enabled
DEFAULT_DATA_PATH = "./data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
# Competition submissions must be self-contained. Leave the profiler path empty
# by default so the record script does not depend on external JSON/NPZ assets.
DEFAULT_CAUSAL_MACHINE_PROFILE_JSON = ""
DEFAULT_CAUSAL_MACHINE_NUM_STATES = 128
DEFAULT_CAUSAL_MACHINE_ONLINE_TEACHER_EMA = 0.95
DEFAULT_VOCAB_SIZE = 1024
USE_CAUSAL_MACHINE_CUDA_SCAN = bool(int(os.environ.get("USE_CAUSAL_MACHINE_CUDA_SCAN", "1")))
USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN = bool(int(os.environ.get("USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN", "1")))
USE_MUON_CUDA = bool(int(os.environ.get("USE_MUON_CUDA", "1")))
PROFILE_MUON_STEP = bool(int(os.environ.get("PROFILE_MUON_STEP", "0")))
MUON_CUDA_BUCKET_POLICY = os.environ.get("MUON_CUDA_BUCKET_POLICY", "auto").strip().lower()
# Historical env name retained for compatibility. This gates the prepacked
# transition-table fast path, which now runs dedicated int8 / FP8 CUDA kernels
# instead of dequantizing back to the float path.
USE_CAUSAL_MACHINE_SCAN_FUSED_DEQUANT = bool(int(os.environ.get("USE_CAUSAL_MACHINE_SCAN_FUSED_DEQUANT", "1")))

_PACKED_TRANSITION_INT8 = 0
_PACKED_TRANSITION_FP8_E4M3 = 1
_PACKED_TRANSITION_FP8_E5M2 = 2

BOS_ID = -1

_CAUSAL_MACHINE_SCAN_CUDA = None
_CAUSAL_MACHINE_SCAN_CUDA_ERROR: Exception | None = None
_CAUSAL_MACHINE_LATENT_SCAN_CUDA = None
_CAUSAL_MACHINE_LATENT_SCAN_CUDA_ERROR: Exception | None = None
_MUON_CUDA = None
_MUON_CUDA_ERROR: Exception | None = None


def _torch_dynamo_disable_if_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None:
        return fn
    disable = getattr(dynamo, "disable", None)
    if disable is None:
        return fn
    return disable(fn)


def _parse_cuda_arch_list(raw: str) -> list[str]:
    archs: list[str] = []
    for item in raw.replace(";", ",").split(","):
        token = item.strip().lower().replace("sm_", "").replace(".", "")
        if not token:
            continue
        if token.endswith("a"):
            if not token[:-1].isdigit():
                continue
        elif not token.isdigit():
            continue
        archs.append(token)
    return archs


@_torch_dynamo_disable_if_available
@functools.lru_cache(maxsize=1)
def _causal_machine_cuda_arch_flags() -> tuple[str, ...]:
    raw = os.environ.get("CAUSAL_MACHINE_CUDA_ARCHS", "").strip()
    archs = _parse_cuda_arch_list(raw)
    if not archs and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            archs = [f"{major}{minor}"]
        except Exception:
            archs = []
    if not raw:
        cuda_version = str(getattr(torch.version, "cuda", "") or "")
        version_tokens = cuda_version.split(".")
        try:
            cuda_major = int(version_tokens[0]) if version_tokens else 0
            cuda_minor = int(version_tokens[1]) if len(version_tokens) > 1 else 0
        except ValueError:
            cuda_major = 0
            cuda_minor = 0
        if (cuda_major, cuda_minor) >= (12, 0):
            if "90" not in archs:
                archs.append("90")
        if (cuda_major, cuda_minor) >= (12, 3):
            if "90a" not in archs:
                archs.append("90a")
    flags: list[str] = []
    for arch in archs:
        if arch.endswith("a"):
            base_arch = arch[:-1]
            if base_arch.isdigit():
                flags.extend([
                    f"-gencode=arch=compute_{base_arch},code=compute_{base_arch}",
                    f"-gencode=arch=compute_{arch},code=sm_{arch}",
                ])
            continue
        flags.extend([
            f"-gencode=arch=compute_{arch},code=compute_{arch}",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ])
    return tuple(flags)


def _load_causal_machine_scan_cuda_extension(source_dir: Path, build_dir: Path):
    return load_cpp_extension(
        name="causal_machine_scan_cuda_ext",
        sources=[
            str(source_dir / "causal_machine_scan.cpp"),
            str(source_dir / "causal_machine_scan_kernel.cu"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", *_causal_machine_cuda_arch_flags()],
        build_directory=str(build_dir),
        verbose=False,
    )


def _extension_candidate_build_dirs(source_dir: Path, build_subdir: str) -> list[Path]:
    dirs: list[Path] = []

    def add_candidate(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in dirs:
            dirs.append(resolved)

    add_candidate(source_dir / build_subdir)

    repo_root = None
    relative_source_dir = None
    for parent in source_dir.parents:
        if parent.name == "records":
            repo_root = parent.parent
            relative_source_dir = source_dir.relative_to(repo_root)
            break
    if relative_source_dir is not None:
        cwd = Path.cwd().resolve()
        for base in (
            cwd,
            cwd.parent,
            Path("/data/parametergolf"),
            Path(os.environ.get("PARAMETER_GOLF_ROOT", "")).resolve() if os.environ.get("PARAMETER_GOLF_ROOT") else None,
        ):
            if base is None:
                continue
            try:
                add_candidate(base / relative_source_dir / build_subdir)
            except Exception:
                continue

    return dirs


def _extension_candidate_build_dirs_multi(source_dir: Path, build_subdirs: Sequence[str]) -> list[Path]:
    dirs: list[Path] = []
    for build_subdir in build_subdirs:
        for candidate in _extension_candidate_build_dirs(source_dir, build_subdir):
            if candidate not in dirs:
                dirs.append(candidate)
    return dirs


@_torch_dynamo_disable_if_available
def _load_existing_cuda_extension_module(
    module_name: str,
    build_dir: Path,
    source_paths: Sequence[Path],
):
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    module_path = build_dir / f"{module_name}.so"
    if not module_path.exists():
        return None
    try:
        module_mtime = module_path.stat().st_mtime
        if any(path.exists() and path.stat().st_mtime > module_mtime for path in source_paths):
            return None
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(module_name, None)
        return None


@_torch_dynamo_disable_if_available
def _stale_prebuilt_cuda_extension_sources(module_path: Path, source_paths: Sequence[Path]) -> tuple[Path, ...]:
    if not module_path.exists():
        return ()
    try:
        module_mtime = module_path.stat().st_mtime
    except Exception:
        return ()
    stale = [
        path
        for path in source_paths
        if path.exists() and path.stat().st_mtime > module_mtime
    ]
    return tuple(stale)


@_torch_dynamo_disable_if_available
def _load_prebuilt_cuda_extension_or_none(
    module_name: str,
    display_name: str,
    build_dirs: Sequence[Path],
    source_paths: Sequence[Path],
):
    checked_paths: list[Path] = []
    for build_dir in build_dirs:
        existing = _load_existing_cuda_extension_module(module_name, build_dir, source_paths)
        if existing is not None:
            return existing
        checked_paths.append(build_dir / f"{module_name}.so")
    if not _require_prebuilt_cuda_extensions():
        return None
    module_path = checked_paths[0]
    stale_sources = _stale_prebuilt_cuda_extension_sources(module_path, source_paths)
    if not any(path.exists() for path in checked_paths):
        checked = ", ".join(str(path) for path in checked_paths)
        raise RuntimeError(
            f"{display_name} requires a prebuilt CUDA extension in competition mode; "
            f"missing one of: {checked}. Prebuild it before starting the timed run."
        )
    if stale_sources:
        stale_names = ", ".join(path.name for path in stale_sources)
        raise RuntimeError(
            f"{display_name} requires an up-to-date prebuilt CUDA extension in competition mode; "
            f"{module_path} is older than: {stale_names}. Rebuild it before starting the timed run."
        )
    raise RuntimeError(
        f"{display_name} requires a loadable prebuilt CUDA extension in competition mode; "
        f"failed to import {module_path}. Rebuild it before starting the timed run."
    )


def _load_causal_machine_latent_scan_cuda_extension(source_dir: Path, build_dir: Path):
    return load_cpp_extension(
        name="causal_machine_latent_scan_cuda_ext",
        sources=[
            str(source_dir / "causal_machine_latent_scan.cpp"),
            str(source_dir / "causal_machine_latent_scan_kernel.cu"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", *_causal_machine_cuda_arch_flags()],
        build_directory=str(build_dir),
        verbose=False,
    )


def _load_muon_cuda_extension(source_dir: Path, build_dir: Path):
    return load_cpp_extension(
        name="muon_cuda_ext",
        sources=[
            str(source_dir / "muon.cpp"),
            str(source_dir / "muon_kernel.cu"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", *_causal_machine_cuda_arch_flags()],
        extra_ldflags=["-lcublasLt"],
        build_directory=str(build_dir),
        verbose=False,
    )


def load_causal_machine_scan_cuda():
    global _CAUSAL_MACHINE_SCAN_CUDA
    global _CAUSAL_MACHINE_SCAN_CUDA_ERROR
    if _CAUSAL_MACHINE_SCAN_CUDA is not None:
        return _CAUSAL_MACHINE_SCAN_CUDA
    if _CAUSAL_MACHINE_SCAN_CUDA_ERROR is not None:
        raise RuntimeError("causal_machine_scan_cuda is unavailable") from _CAUSAL_MACHINE_SCAN_CUDA_ERROR
    source_dir = Path(__file__).resolve().parent / "cuda_ext"
    build_dirs = _extension_candidate_build_dirs_multi(
        source_dir,
        (
            "build/causal_machine_scan_cuda",
            "build",
        ),
    )
    build_dir = build_dirs[0]
    build_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_prebuilt_cuda_extension_or_none(
        "causal_machine_scan_cuda_ext",
        "causal_machine_scan_cuda",
        build_dirs,
        (
            source_dir / "causal_machine_scan.cpp",
            source_dir / "causal_machine_scan_kernel.cu",
        ),
    )
    if existing is not None:
        _CAUSAL_MACHINE_SCAN_CUDA = existing
        return _CAUSAL_MACHINE_SCAN_CUDA
    last_exc: Exception | None = None
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    attempts = 4 if distributed else 2
    if not distributed:
        try:
            _CAUSAL_MACHINE_SCAN_CUDA = _load_causal_machine_scan_cuda_extension(source_dir, build_dir)
            return _CAUSAL_MACHINE_SCAN_CUDA
        except Exception as exc:
            _CAUSAL_MACHINE_SCAN_CUDA_ERROR = exc
            raise RuntimeError("causal_machine_scan_cuda is unavailable") from exc

    sync_device = torch.device("cuda", torch.cuda.current_device())
    for attempt in range(attempts):
        local_exc: Exception | None = None
        leader_success = torch.zeros(1, device=sync_device, dtype=torch.int32)
        try:
            if rank == 0:
                if _CAUSAL_MACHINE_SCAN_CUDA is None:
                    _CAUSAL_MACHINE_SCAN_CUDA = _load_causal_machine_scan_cuda_extension(source_dir, build_dir)
                leader_success.fill_(1)
        except Exception as exc:
            local_exc = exc
            if rank == 0:
                _CAUSAL_MACHINE_SCAN_CUDA = None
            leader_success.zero_()

        dist.broadcast(leader_success, src=0)

        if int(leader_success.item()) == 1:
            try:
                if rank != 0 and _CAUSAL_MACHINE_SCAN_CUDA is None:
                    _CAUSAL_MACHINE_SCAN_CUDA = _load_causal_machine_scan_cuda_extension(source_dir, build_dir)
            except Exception as exc:
                local_exc = exc
                _CAUSAL_MACHINE_SCAN_CUDA = None

            load_ok = torch.tensor([0 if local_exc is not None else 1], device=sync_device, dtype=torch.int32)
            dist.all_reduce(load_ok, op=dist.ReduceOp.MIN)
            if int(load_ok.item()) == 1:
                return _CAUSAL_MACHINE_SCAN_CUDA

        if local_exc is None:
            local_exc = RuntimeError(
                f"causal_machine_scan_cuda leader build failed on rank 0 during attempt {attempt + 1}/{attempts}"
            )
        last_exc = local_exc
        _CAUSAL_MACHINE_SCAN_CUDA = None
        if attempt + 1 < attempts:
            time.sleep(0.5 * (attempt + 1))
            continue

    _CAUSAL_MACHINE_SCAN_CUDA_ERROR = last_exc
    raise RuntimeError("causal_machine_scan_cuda is unavailable after retries") from last_exc


def load_causal_machine_latent_scan_cuda():
    global _CAUSAL_MACHINE_LATENT_SCAN_CUDA
    global _CAUSAL_MACHINE_LATENT_SCAN_CUDA_ERROR
    if _CAUSAL_MACHINE_LATENT_SCAN_CUDA is not None:
        return _CAUSAL_MACHINE_LATENT_SCAN_CUDA
    if _CAUSAL_MACHINE_LATENT_SCAN_CUDA_ERROR is not None:
        raise RuntimeError("causal_machine_latent_scan_cuda is unavailable") from _CAUSAL_MACHINE_LATENT_SCAN_CUDA_ERROR
    source_dir = Path(__file__).resolve().parent / "cuda_ext"
    build_dirs = _extension_candidate_build_dirs_multi(
        source_dir,
        (
            "build/causal_machine_latent_scan_cuda",
            "build",
        ),
    )
    build_dir = build_dirs[0]
    build_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_prebuilt_cuda_extension_or_none(
        "causal_machine_latent_scan_cuda_ext",
        "causal_machine_latent_scan_cuda",
        build_dirs,
        (
            source_dir / "causal_machine_latent_scan.cpp",
            source_dir / "causal_machine_latent_scan_kernel.cu",
        ),
    )
    if existing is not None:
        _CAUSAL_MACHINE_LATENT_SCAN_CUDA = existing
        return _CAUSAL_MACHINE_LATENT_SCAN_CUDA
    last_exc: Exception | None = None
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    attempts = 4 if distributed else 2
    if not distributed:
        try:
            _CAUSAL_MACHINE_LATENT_SCAN_CUDA = _load_causal_machine_latent_scan_cuda_extension(source_dir, build_dir)
            return _CAUSAL_MACHINE_LATENT_SCAN_CUDA
        except Exception as exc:
            _CAUSAL_MACHINE_LATENT_SCAN_CUDA_ERROR = exc
            raise RuntimeError("causal_machine_latent_scan_cuda is unavailable") from exc

    sync_device = torch.device("cuda", torch.cuda.current_device())
    for attempt in range(attempts):
        local_exc: Exception | None = None
        leader_success = torch.zeros(1, device=sync_device, dtype=torch.int32)
        try:
            if rank == 0:
                if _CAUSAL_MACHINE_LATENT_SCAN_CUDA is None:
                    _CAUSAL_MACHINE_LATENT_SCAN_CUDA = _load_causal_machine_latent_scan_cuda_extension(source_dir, build_dir)
                leader_success.fill_(1)
        except Exception as exc:
            local_exc = exc
            if rank == 0:
                _CAUSAL_MACHINE_LATENT_SCAN_CUDA = None
            leader_success.zero_()

        dist.broadcast(leader_success, src=0)

        if int(leader_success.item()) == 1:
            try:
                if rank != 0 and _CAUSAL_MACHINE_LATENT_SCAN_CUDA is None:
                    _CAUSAL_MACHINE_LATENT_SCAN_CUDA = _load_causal_machine_latent_scan_cuda_extension(source_dir, build_dir)
            except Exception as exc:
                local_exc = exc
                _CAUSAL_MACHINE_LATENT_SCAN_CUDA = None

            load_ok = torch.tensor([0 if local_exc is not None else 1], device=sync_device, dtype=torch.int32)
            dist.all_reduce(load_ok, op=dist.ReduceOp.MIN)
            if int(load_ok.item()) == 1:
                return _CAUSAL_MACHINE_LATENT_SCAN_CUDA

        if local_exc is None:
            local_exc = RuntimeError(
                f"causal_machine_latent_scan_cuda leader build failed on rank 0 during attempt {attempt + 1}/{attempts}"
            )
        last_exc = local_exc
        _CAUSAL_MACHINE_LATENT_SCAN_CUDA = None
        if attempt + 1 < attempts:
            time.sleep(0.5 * (attempt + 1))
            continue

    _CAUSAL_MACHINE_LATENT_SCAN_CUDA_ERROR = last_exc
    raise RuntimeError("causal_machine_latent_scan_cuda is unavailable after retries") from last_exc


def load_muon_cuda():
    global _MUON_CUDA
    global _MUON_CUDA_ERROR
    if _MUON_CUDA is not None:
        return _MUON_CUDA
    if _MUON_CUDA_ERROR is not None:
        raise RuntimeError("muon_cuda is unavailable") from _MUON_CUDA_ERROR
    source_dir = Path(__file__).resolve().parent / "cuda_ext"
    build_dirs = _extension_candidate_build_dirs(source_dir, "build/muon_cuda")
    build_dir = build_dirs[0]
    build_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_prebuilt_cuda_extension_or_none(
        "muon_cuda_ext",
        "muon_cuda",
        build_dirs,
        (
            source_dir / "muon.cpp",
            source_dir / "muon_kernel.cu",
        ),
    )
    if existing is not None:
        _MUON_CUDA = existing
        return _MUON_CUDA
    last_exc: Exception | None = None
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    attempts = 4 if distributed else 2
    if not distributed:
        try:
            _MUON_CUDA = _load_muon_cuda_extension(source_dir, build_dir)
            return _MUON_CUDA
        except Exception as exc:
            _MUON_CUDA_ERROR = exc
            raise RuntimeError("muon_cuda is unavailable") from exc

    sync_device = torch.device("cuda", torch.cuda.current_device())
    for attempt in range(attempts):
        local_exc: Exception | None = None
        leader_success = torch.zeros(1, device=sync_device, dtype=torch.int32)
        try:
            if rank == 0:
                if _MUON_CUDA is None:
                    _MUON_CUDA = _load_muon_cuda_extension(source_dir, build_dir)
                leader_success.fill_(1)
        except Exception as exc:
            local_exc = exc
            if rank == 0:
                _MUON_CUDA = None
            leader_success.zero_()

        dist.broadcast(leader_success, src=0)

        if int(leader_success.item()) == 1:
            try:
                if rank != 0 and _MUON_CUDA is None:
                    _MUON_CUDA = _load_muon_cuda_extension(source_dir, build_dir)
            except Exception as exc:
                local_exc = exc
                _MUON_CUDA = None

            load_ok = torch.tensor([0 if local_exc is not None else 1], device=sync_device, dtype=torch.int32)
            dist.all_reduce(load_ok, op=dist.ReduceOp.MIN)
            if int(load_ok.item()) == 1:
                return _MUON_CUDA

        if local_exc is None:
            local_exc = RuntimeError(
                f"muon_cuda leader build failed on rank 0 during attempt {attempt + 1}/{attempts}"
            )
        last_exc = local_exc
        _MUON_CUDA = None
        if attempt + 1 < attempts:
            time.sleep(0.5 * (attempt + 1))
            continue

    _MUON_CUDA_ERROR = last_exc
    raise RuntimeError("muon_cuda is unavailable after retries") from last_exc


class _CausalMachineScanCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_logits: Tensor,
        transition_source_logits: Tensor,
        transition_dest_logits: Tensor,
        transition_context: Tensor,
        initial_log_belief: Tensor,
        transition_gate: Tensor,
        transition_stay_probs: Tensor,
        packed_source_q: Tensor,
        packed_source_scales: Tensor,
        packed_dest_q: Tensor,
        packed_dest_scales: Tensor,
        packed_kind: int,
        use_packed: bool,
        chunk_size: int,
        score_clamp_min: float,
        score_clamp_max: float,
        composable: bool,
    ) -> tuple[Tensor, Tensor]:
        ext = load_causal_machine_scan_cuda()
        local_logits_in = local_logits.contiguous()
        transition_context_in = transition_context.contiguous()
        initial_log_belief_in = initial_log_belief.contiguous()
        transition_stay_probs_f32 = transition_stay_probs.contiguous().float()
        transition_source_logits_f32 = transition_source_logits.contiguous().float()
        transition_dest_logits_f32 = transition_dest_logits.contiguous().float()
        transition_source_probs_f32 = F.softmax(transition_source_logits_f32, dim=-1).contiguous()
        transition_dest_probs_f32 = F.softmax(transition_dest_logits_f32, dim=-1).contiguous()
        gate_value = float(transition_gate.detach().float().item())
        ctx.composable = bool(composable)
        ctx.use_packed = bool(use_packed)
        ctx.packed_kind = int(packed_kind)
        if ctx.use_packed:
            packed_source_q = packed_source_q.contiguous()
            packed_source_scales = packed_source_scales.contiguous()
            packed_dest_q = packed_dest_q.contiguous()
            packed_dest_scales = packed_dest_scales.contiguous()
            if ctx.packed_kind == _PACKED_TRANSITION_INT8:
                beliefs, final_belief = ext.forward_quantized(
                    local_logits_in,
                    packed_source_q,
                    packed_source_scales,
                    packed_dest_q,
                    packed_dest_scales,
                    transition_context_in,
                    initial_log_belief_in,
                    gate_value,
                    transition_stay_probs_f32,
                    int(chunk_size),
                )
            elif ctx.packed_kind == _PACKED_TRANSITION_FP8_E4M3:
                beliefs, final_belief = ext.forward_fp8(
                    local_logits_in,
                    packed_source_q,
                    packed_source_scales,
                    packed_dest_q,
                    packed_dest_scales,
                    transition_context_in,
                    initial_log_belief_in,
                    gate_value,
                    transition_stay_probs_f32,
                    0,
                    int(chunk_size),
                )
            elif ctx.packed_kind == _PACKED_TRANSITION_FP8_E5M2:
                beliefs, final_belief = ext.forward_fp8(
                    local_logits_in,
                    packed_source_q,
                    packed_source_scales,
                    packed_dest_q,
                    packed_dest_scales,
                    transition_context_in,
                    initial_log_belief_in,
                    gate_value,
                    transition_stay_probs_f32,
                    1,
                    int(chunk_size),
                )
            else:
                raise ValueError(f"unsupported packed transition kind: {ctx.packed_kind}")
            packed_source_probs_f32, packed_dest_probs_f32 = _unpack_scan_transition_tables(
                ext,
                int(ctx.packed_kind),
                packed_source_q,
                packed_source_scales,
                packed_dest_q,
                packed_dest_scales,
            )
            ctx.save_for_backward(
                packed_source_q,
                packed_source_scales,
                packed_dest_q,
                packed_dest_scales,
                transition_context_in,
                initial_log_belief_in,
                beliefs.contiguous(),
                packed_source_probs_f32,
                packed_dest_probs_f32,
                transition_gate.float(),
                transition_stay_probs_f32,
            )
        elif ctx.composable:
            beliefs, final_belief = ext.forward_composable_logits(
                local_logits_in,
                transition_source_logits_f32,
                transition_dest_logits_f32,
                transition_context_in,
                initial_log_belief_in,
                transition_stay_probs_f32,
                int(chunk_size),
            )
            ctx.save_for_backward(
                transition_context_in,
                initial_log_belief_in,
                beliefs.contiguous(),
                transition_source_logits_f32,
                transition_dest_logits_f32,
                transition_gate.float(),
                transition_stay_probs_f32,
            )
        else:
            beliefs, final_belief = ext.forward_logits(
                local_logits_in,
                transition_source_logits_f32,
                transition_dest_logits_f32,
                transition_context_in,
                initial_log_belief_in,
                gate_value,
                transition_stay_probs_f32,
                int(chunk_size),
                float(score_clamp_min),
                float(score_clamp_max),
            )
            ctx.save_for_backward(
                transition_context_in,
                initial_log_belief_in,
                beliefs.contiguous(),
                transition_source_logits_f32,
                transition_dest_logits_f32,
                transition_gate.float(),
                transition_stay_probs_f32,
            )
        ctx.chunk_size = int(chunk_size)
        ctx.score_clamp_min = float(score_clamp_min)
        ctx.score_clamp_max = float(score_clamp_max)
        return beliefs, final_belief

    @staticmethod
    def backward(ctx, grad_beliefs: Tensor, grad_final_belief: Tensor):
        ext = load_causal_machine_scan_cuda()
        if bool(getattr(ctx, "use_packed", False)):
            (
                packed_source_q,
                packed_source_scales,
                packed_dest_q,
                packed_dest_scales,
                transition_context_saved,
                initial_log_belief_saved,
                beliefs_saved,
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
            ) = ctx.saved_tensors
            packed_source_q = packed_source_q.contiguous()
            packed_source_scales = packed_source_scales.contiguous()
            packed_dest_q = packed_dest_q.contiguous()
            packed_dest_scales = packed_dest_scales.contiguous()
            common_args = (
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                packed_source_q,
                packed_source_scales,
                packed_dest_q,
                packed_dest_scales,
                transition_context_saved.contiguous(),
                initial_log_belief_saved.contiguous(),
                beliefs_saved.contiguous(),
                float(transition_gate_f32.detach().float().item()),
                transition_stay_probs_f32,
            )
            if int(getattr(ctx, "packed_kind", _PACKED_TRANSITION_INT8)) == _PACKED_TRANSITION_INT8:
                grads = ext.backward_quantized(*common_args, int(ctx.chunk_size))
            elif int(ctx.packed_kind) == _PACKED_TRANSITION_FP8_E4M3:
                grads = ext.backward_fp8(*common_args, 0, int(ctx.chunk_size))
            elif int(ctx.packed_kind) == _PACKED_TRANSITION_FP8_E5M2:
                grads = ext.backward_fp8(*common_args, 1, int(ctx.chunk_size))
            else:
                raise ValueError(f"unsupported packed transition kind: {ctx.packed_kind}")
            grad_local, grad_source_probs, grad_dest_probs, grad_context, grad_initial, grad_gate, grad_stay = grads
            grad_source = (
                _softmax_backward_rows(grad_source_probs, transition_source_probs_f32)
                if ctx.needs_input_grad[1]
                else None
            )
            grad_dest = (
                _softmax_backward_rows(grad_dest_probs, transition_dest_probs_f32)
                if ctx.needs_input_grad[2]
                else None
            )
            if bool(getattr(ctx, "composable", False)):
                grad_gate = torch.zeros_like(grad_gate)
        else:
            (
                transition_context_saved,
                initial_log_belief_saved,
                beliefs_saved,
                transition_source_logits_f32,
                transition_dest_logits_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
            ) = ctx.saved_tensors
            if bool(getattr(ctx, "composable", False)):
                grads = ext.backward_composable_logits(
                    grad_beliefs.contiguous(),
                    grad_final_belief.contiguous(),
                    transition_source_logits_f32,
                    transition_dest_logits_f32,
                    transition_context_saved.contiguous(),
                    initial_log_belief_saved.contiguous(),
                    beliefs_saved.contiguous(),
                    transition_stay_probs_f32,
                    int(ctx.chunk_size),
                )
            else:
                grads = ext.backward_logits(
                    grad_beliefs.contiguous(),
                    grad_final_belief.contiguous(),
                    transition_source_logits_f32,
                    transition_dest_logits_f32,
                    transition_context_saved.contiguous(),
                    initial_log_belief_saved.contiguous(),
                    beliefs_saved.contiguous(),
                    float(transition_gate_f32.detach().float().item()),
                    transition_stay_probs_f32,
                    int(ctx.chunk_size),
                    float(ctx.score_clamp_min),
                    float(ctx.score_clamp_max),
                )
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = grads
        return (
            grad_local,
            grad_source,
            grad_dest,
            grad_context,
            grad_initial,
            grad_gate.reshape_as(transition_gate_f32),
            grad_stay.reshape_as(transition_stay_probs_f32),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _CausalMachineTiledScanCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_logits: Tensor,
        transition_source_probs: Tensor,
        transition_dest_probs: Tensor,
        transition_context: Tensor,
        initial_log_belief: Tensor,
        transition_gate: Tensor,
        transition_stay_probs: Tensor,
        packed_source_q: Tensor,
        packed_source_scales: Tensor,
        packed_dest_q: Tensor,
        packed_dest_scales: Tensor,
        packed_kind: int,
        use_packed: bool,
        seq_lens: Tensor,
        chunk_size: int,
        tile_size: int,
        split_size: int,
        score_clamp_min: float,
        score_clamp_max: float,
        score_threshold: float,
        score_topk: int,
        use_custom_forward_kernel: bool,
        use_custom_backward_kernel: bool,
        workspace: dict[str, Any] | None,
    ) -> tuple[Tensor, Tensor]:
        ext = load_causal_machine_scan_cuda()
        local_logits_in = local_logits.contiguous()
        transition_source_probs_f32 = transition_source_probs.contiguous().float()
        transition_dest_probs_f32 = transition_dest_probs.contiguous().float()
        transition_context_in = transition_context.contiguous()
        initial_log_belief_f32 = initial_log_belief.contiguous().float()
        transition_gate_f32 = transition_gate.contiguous().float().reshape(())
        transition_stay_probs_f32 = transition_stay_probs.contiguous().float()
        packed_source_q_in = packed_source_q.contiguous()
        packed_source_scales_in = packed_source_scales.contiguous()
        packed_dest_q_in = packed_dest_q.contiguous()
        packed_dest_scales_in = packed_dest_scales.contiguous()
        seq_lens_in = seq_lens.contiguous()
        workspace_dict = workspace if isinstance(workspace, dict) else None
        ctx.use_packed = bool(use_packed)
        ctx.packed_kind = int(packed_kind)
        if (
            ctx.use_packed
            and ctx.packed_kind == _PACKED_TRANSITION_INT8
            and workspace_dict is not None
            and hasattr(ext, "forward_tiled_quantized_kernel_bound_workspace")
        ):
            beliefs, final_belief = ext.forward_tiled_quantized_kernel_bound_workspace(
                local_logits_in,
                packed_source_q_in,
                packed_source_scales_in,
                packed_dest_q_in,
                packed_dest_scales_in,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                workspace_dict,
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif (
            ctx.use_packed
            and ctx.packed_kind == _PACKED_TRANSITION_INT8
            and workspace_dict is not None
            and hasattr(ext, "forward_tiled_quantized_kernel_workspace")
        ):
            beliefs, final_belief = ext.forward_tiled_quantized_kernel_workspace(
                local_logits_in,
                packed_source_q_in,
                packed_source_scales_in,
                packed_dest_q_in,
                packed_dest_scales_in,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                workspace_dict["work_queue_counter"],
                workspace_dict["filtered_value_cache"],
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif ctx.use_packed and ctx.packed_kind == _PACKED_TRANSITION_INT8:
            beliefs, final_belief = ext.forward_tiled_quantized_kernel(
                local_logits_in,
                packed_source_q_in,
                packed_source_scales_in,
                packed_dest_q_in,
                packed_dest_scales_in,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif (
            ctx.use_packed
            and ctx.packed_kind in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}
            and workspace_dict is not None
            and hasattr(ext, "forward_tiled_fp8_kernel_bound_workspace")
        ):
            beliefs, final_belief = ext.forward_tiled_fp8_kernel_bound_workspace(
                local_logits_in,
                packed_source_q_in,
                packed_source_scales_in,
                packed_dest_q_in,
                packed_dest_scales_in,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                0 if ctx.packed_kind == _PACKED_TRANSITION_FP8_E4M3 else 1,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                workspace_dict,
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif (
            ctx.use_packed
            and ctx.packed_kind in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}
            and workspace_dict is not None
            and hasattr(ext, "forward_tiled_fp8_kernel_workspace")
        ):
            beliefs, final_belief = ext.forward_tiled_fp8_kernel_workspace(
                local_logits_in,
                packed_source_q_in,
                packed_source_scales_in,
                packed_dest_q_in,
                packed_dest_scales_in,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                0 if ctx.packed_kind == _PACKED_TRANSITION_FP8_E4M3 else 1,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                workspace_dict["work_queue_counter"],
                workspace_dict["filtered_value_cache"],
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif ctx.use_packed and ctx.packed_kind in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}:
            beliefs, final_belief = ext.forward_tiled_fp8_kernel(
                local_logits_in,
                packed_source_q_in,
                packed_source_scales_in,
                packed_dest_q_in,
                packed_dest_scales_in,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                0 if ctx.packed_kind == _PACKED_TRANSITION_FP8_E4M3 else 1,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif (
            bool(use_custom_forward_kernel)
            and workspace_dict is not None
            and hasattr(ext, "forward_tiled_logits_kernel_bound_workspace")
        ):
            beliefs, final_belief = ext.forward_tiled_logits_kernel_bound_workspace(
                local_logits_in,
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                workspace_dict,
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif (
            bool(use_custom_forward_kernel)
            and workspace_dict is not None
            and hasattr(ext, "forward_tiled_logits_kernel_workspace")
        ):
            beliefs, final_belief = ext.forward_tiled_logits_kernel_workspace(
                local_logits_in,
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                workspace_dict["work_queue_counter"],
                workspace_dict["filtered_value_cache"],
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif bool(use_custom_forward_kernel) and hasattr(ext, "forward_tiled_logits_kernel"):
            beliefs, final_belief = ext.forward_tiled_logits_kernel(
                local_logits_in,
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif hasattr(ext, "forward_tiled_logits"):
            beliefs, final_belief = ext.forward_tiled_logits(
                local_logits_in,
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_in,
                initial_log_belief_f32,
                transition_gate_f32,
                transition_stay_probs_f32,
                seq_lens_in,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                float(score_clamp_min),
                float(score_clamp_max),
            )
        else:
            raise RuntimeError("cuda_tiled requires forward_tiled_logits in the CUDA extension")
        if ctx.use_packed:
            ctx.save_for_backward(
                packed_source_q_in,
                packed_source_scales_in,
                packed_dest_q_in,
                packed_dest_scales_in,
                transition_context_in,
                initial_log_belief_f32,
                beliefs.contiguous(),
                transition_gate.float(),
                transition_stay_probs_f32,
                seq_lens_in,
            )
        else:
            ctx.save_for_backward(
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_in,
                initial_log_belief_f32,
                beliefs.contiguous(),
                transition_gate.float(),
                transition_stay_probs_f32,
                seq_lens_in,
            )
        ctx.chunk_size = int(chunk_size)
        ctx.tile_size = int(tile_size)
        ctx.split_size = int(split_size)
        ctx.score_clamp_min = float(score_clamp_min)
        ctx.score_clamp_max = float(score_clamp_max)
        ctx.score_threshold = float(score_threshold)
        ctx.score_topk = int(score_topk)
        ctx.use_custom_backward_kernel = bool(use_custom_backward_kernel) and hasattr(ext, "backward_tiled_probs_kernel")
        ctx.workspace = workspace_dict
        return beliefs, final_belief

    @staticmethod
    def backward(ctx, grad_beliefs: Tensor, grad_final_belief: Tensor):
        ext = load_causal_machine_scan_cuda()
        if bool(getattr(ctx, "use_packed", False)):
            (
                packed_source_q,
                packed_source_scales,
                packed_dest_q,
                packed_dest_scales,
                transition_context_saved,
                initial_log_belief_f32,
                beliefs_saved,
                transition_gate_f32,
                transition_stay_probs_f32,
                seq_lens_saved,
            ) = ctx.saved_tensors
        else:
            (
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_saved,
                initial_log_belief_f32,
                beliefs_saved,
                transition_gate_f32,
                transition_stay_probs_f32,
                seq_lens_saved,
            ) = ctx.saved_tensors
        workspace_dict = getattr(ctx, "workspace", None)
        if (
            bool(getattr(ctx, "use_packed", False))
            and int(getattr(ctx, "packed_kind", -1)) == _PACKED_TRANSITION_INT8
            and workspace_dict is not None
            and hasattr(ext, "backward_tiled_quantized_kernel_bound_workspace")
        ):
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_quantized_kernel_bound_workspace(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                packed_source_q.contiguous(),
                packed_source_scales.contiguous(),
                packed_dest_q.contiguous(),
                packed_dest_scales.contiguous(),
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                transition_gate_f32.detach().float().reshape(()),
                transition_stay_probs_f32,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                workspace_dict,
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        elif (
            bool(getattr(ctx, "use_packed", False))
            and int(getattr(ctx, "packed_kind", -1)) == _PACKED_TRANSITION_INT8
            and workspace_dict is not None
            and hasattr(ext, "backward_tiled_quantized_kernel_workspace")
        ):
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_quantized_kernel_workspace(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                packed_source_q.contiguous(),
                packed_source_scales.contiguous(),
                packed_dest_q.contiguous(),
                packed_dest_scales.contiguous(),
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                transition_gate_f32.detach().float().reshape(()),
                transition_stay_probs_f32,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                workspace_dict["work_queue_counter"],
                workspace_dict["latent_cache_staging"],
                workspace_dict["grad_latent_accum_staging"],
                workspace_dict["grad_transition_source_probs_staging"],
                workspace_dict["grad_transition_dest_probs_staging"],
                workspace_dict["grad_transition_gate_staging"],
                workspace_dict["grad_transition_stay_staging"],
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        elif bool(getattr(ctx, "use_packed", False)) and int(getattr(ctx, "packed_kind", -1)) == _PACKED_TRANSITION_INT8:
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_quantized_kernel(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                packed_source_q.contiguous(),
                packed_source_scales.contiguous(),
                packed_dest_q.contiguous(),
                packed_dest_scales.contiguous(),
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                float(transition_gate_f32.detach().float().item()),
                transition_stay_probs_f32,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        elif (
            bool(getattr(ctx, "use_packed", False))
            and int(getattr(ctx, "packed_kind", -1)) in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}
            and workspace_dict is not None
            and hasattr(ext, "backward_tiled_fp8_kernel_bound_workspace")
        ):
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_fp8_kernel_bound_workspace(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                packed_source_q.contiguous(),
                packed_source_scales.contiguous(),
                packed_dest_q.contiguous(),
                packed_dest_scales.contiguous(),
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                transition_gate_f32.detach().float().reshape(()),
                transition_stay_probs_f32,
                0 if int(ctx.packed_kind) == _PACKED_TRANSITION_FP8_E4M3 else 1,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                workspace_dict,
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        elif (
            bool(getattr(ctx, "use_packed", False))
            and int(getattr(ctx, "packed_kind", -1)) in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}
            and workspace_dict is not None
            and hasattr(ext, "backward_tiled_fp8_kernel_workspace")
        ):
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_fp8_kernel_workspace(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                packed_source_q.contiguous(),
                packed_source_scales.contiguous(),
                packed_dest_q.contiguous(),
                packed_dest_scales.contiguous(),
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                transition_gate_f32.detach().float().reshape(()),
                transition_stay_probs_f32,
                0 if int(ctx.packed_kind) == _PACKED_TRANSITION_FP8_E4M3 else 1,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                workspace_dict["work_queue_counter"],
                workspace_dict["latent_cache_staging"],
                workspace_dict["grad_latent_accum_staging"],
                workspace_dict["grad_transition_source_probs_staging"],
                workspace_dict["grad_transition_dest_probs_staging"],
                workspace_dict["grad_transition_gate_staging"],
                workspace_dict["grad_transition_stay_staging"],
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        elif bool(getattr(ctx, "use_packed", False)) and int(getattr(ctx, "packed_kind", -1)) in {
            _PACKED_TRANSITION_FP8_E4M3,
            _PACKED_TRANSITION_FP8_E5M2,
        }:
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_fp8_kernel(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                packed_source_q.contiguous(),
                packed_source_scales.contiguous(),
                packed_dest_q.contiguous(),
                packed_dest_scales.contiguous(),
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                float(transition_gate_f32.detach().float().item()),
                transition_stay_probs_f32,
                0 if int(ctx.packed_kind) == _PACKED_TRANSITION_FP8_E4M3 else 1,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        elif (
            bool(getattr(ctx, "use_custom_backward_kernel", False))
            and workspace_dict is not None
            and hasattr(ext, "backward_tiled_probs_kernel_bound_workspace")
        ):
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_probs_kernel_bound_workspace(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                transition_gate_f32.detach().float().reshape(()),
                transition_stay_probs_f32,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                workspace_dict,
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        elif (
            bool(getattr(ctx, "use_custom_backward_kernel", False))
            and workspace_dict is not None
            and hasattr(ext, "backward_tiled_probs_kernel_workspace")
        ):
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_probs_kernel_workspace(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                transition_gate_f32.detach().float().reshape(()),
                transition_stay_probs_f32,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                workspace_dict["work_queue_counter"],
                workspace_dict["latent_cache_staging"],
                workspace_dict["grad_latent_accum_staging"],
                workspace_dict["grad_transition_source_probs_staging"],
                workspace_dict["grad_transition_dest_probs_staging"],
                workspace_dict["grad_transition_gate_staging"],
                workspace_dict["grad_transition_stay_staging"],
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        elif bool(getattr(ctx, "use_custom_backward_kernel", False)) and hasattr(ext, "backward_tiled_probs_kernel"):
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_probs_kernel(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                float(transition_gate_f32.detach().float().item()),
                transition_stay_probs_f32,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        elif hasattr(ext, "backward_tiled_probs"):
            grad_local, grad_source, grad_dest, grad_context, grad_initial, grad_gate, grad_stay = ext.backward_tiled_probs(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_saved.contiguous(),
                initial_log_belief_f32.contiguous(),
                beliefs_saved.contiguous(),
                float(transition_gate_f32.detach().float().item()),
                transition_stay_probs_f32,
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                int(ctx.tile_size),
                int(ctx.split_size),
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
            )
        else:
            raise RuntimeError("cuda_tiled requires backward_tiled_probs in the CUDA extension when gradients are enabled")
        return (
            grad_local,
            grad_source,
            grad_dest,
            grad_context,
            grad_initial,
            grad_gate.reshape_as(transition_gate_f32),
            grad_stay.reshape_as(transition_stay_probs_f32),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _materialize_structured_sparse_transition_values(
    transition_source_probs: Tensor,
    transition_dest_probs: Tensor,
    sparse_transition_tables: StructuredSparseTransitionTables,
) -> tuple[Tensor, Tensor]:
    num_states = int(transition_source_probs.size(0))
    padded_states = int(sparse_transition_tables.row_sums.numel())
    block_size = int(sparse_transition_tables.block_size)
    if transition_source_probs.is_cuda and transition_dest_probs.is_cuda:
        ext = load_causal_machine_scan_cuda()
        return ext.materialize_sparse_blocks(
            transition_source_probs.contiguous().float(),
            transition_dest_probs.contiguous().float(),
            sparse_transition_tables.col_idx.contiguous(),
            sparse_transition_tables.dst_idx.contiguous(),
            sparse_transition_tables.block_mask.contiguous().float(),
            int(padded_states),
            int(block_size),
        )
    transition_rank = int(transition_source_probs.size(1))
    source_probs = transition_source_probs.float()
    dest_probs = transition_dest_probs.float()
    if padded_states != num_states:
        source_pad = torch.zeros((padded_states, transition_rank), device=source_probs.device, dtype=torch.float32)
        source_pad[:num_states, :] = source_probs
        source_probs = source_pad
        dest_pad = torch.zeros((transition_rank, padded_states), device=dest_probs.device, dtype=torch.float32)
        dest_pad[:, :num_states] = dest_probs
        dest_probs = dest_pad
    raw_blocks: list[Tensor] = []
    row_sums = torch.zeros((padded_states,), device=source_probs.device, dtype=torch.float32)
    for nz in range(int(sparse_transition_tables.col_idx.numel())):
        src_block = int(sparse_transition_tables.col_idx[nz].item())
        dst_block = int(sparse_transition_tables.dst_idx[nz].item())
        src_base = src_block * block_size
        dst_base = dst_block * block_size
        raw_block = torch.matmul(
            source_probs[src_base : src_base + block_size, :],
            dest_probs[:, dst_base : dst_base + block_size],
        )
        raw_block = raw_block * sparse_transition_tables.block_mask[nz]
        row_sums[src_base : src_base + block_size] += raw_block.sum(dim=1)
        raw_blocks.append(raw_block)
    if raw_blocks:
        normalized_blocks = [
            raw_block
            / row_sums[
                int(sparse_transition_tables.col_idx[nz].item()) * block_size
                : (int(sparse_transition_tables.col_idx[nz].item()) + 1) * block_size
            ].clamp_min(1.0e-20).unsqueeze(1)
            for nz, raw_block in enumerate(raw_blocks)
        ]
        transition_blocks = torch.stack(normalized_blocks, dim=0).contiguous()
    else:
        transition_blocks = torch.empty((0, block_size, block_size), device=source_probs.device, dtype=torch.float32)
    return transition_blocks, row_sums.contiguous()


class _CausalMachineSparseScanCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_logits: Tensor,
        transition_source_logits: Tensor,
        transition_dest_logits: Tensor,
        transition_context: Tensor,
        initial_log_belief: Tensor,
        transition_gate: Tensor,
        transition_stay_probs: Tensor,
        block_row_ptr: Tensor,
        block_col_idx: Tensor,
        block_dst_idx: Tensor,
        src_row_ptr: Tensor,
        src_nz_idx: Tensor,
        grouped_src_row_ptr: Tensor,
        grouped_src_block_idx: Tensor,
        block_mask: Tensor,
        seq_lens: Tensor,
        block_size: int,
        chunk_size: int,
    ) -> tuple[Tensor, Tensor]:
        ext = load_causal_machine_scan_cuda()
        beliefs, final_belief = ext.forward_sparse_logits_fused(
            local_logits.contiguous(),
            transition_source_logits.contiguous().float(),
            transition_dest_logits.contiguous().float(),
            block_row_ptr.contiguous(),
            block_col_idx.contiguous(),
            block_dst_idx.contiguous(),
            src_row_ptr.contiguous(),
            src_nz_idx.contiguous(),
            block_mask.contiguous().float(),
            transition_context.contiguous(),
            initial_log_belief.contiguous(),
            transition_gate.reshape(()).contiguous().float(),
            transition_stay_probs.contiguous().float(),
            seq_lens.contiguous(),
            int(block_size),
            int(chunk_size),
        )
        ctx.block_size = int(block_size)
        ctx.chunk_size = int(chunk_size)
        ctx.save_for_backward(
            transition_source_logits.contiguous().float(),
            transition_dest_logits.contiguous().float(),
            transition_context.contiguous(),
            initial_log_belief.contiguous(),
            beliefs.contiguous(),
            transition_gate.contiguous().float(),
            transition_stay_probs.contiguous().float(),
            block_row_ptr.contiguous(),
            block_col_idx.contiguous(),
            block_dst_idx.contiguous(),
            src_row_ptr.contiguous(),
            src_nz_idx.contiguous(),
            grouped_src_row_ptr.contiguous(),
            grouped_src_block_idx.contiguous(),
            block_mask.contiguous().float(),
            seq_lens.contiguous(),
        )
        return beliefs, final_belief

    @staticmethod
    def backward(ctx, grad_beliefs: Tensor, grad_final_belief: Tensor):
        ext = load_causal_machine_scan_cuda()
        (
            transition_source_logits_f32,
            transition_dest_logits_f32,
            transition_context_saved,
            initial_log_belief_saved,
            beliefs_saved,
            transition_gate_f32,
            transition_stay_probs_f32,
            block_row_ptr,
            block_col_idx,
            block_dst_idx,
            src_row_ptr,
            src_nz_idx,
            grouped_src_row_ptr,
            grouped_src_block_idx,
            block_mask,
            seq_lens,
        ) = ctx.saved_tensors
        grads = ext.backward_sparse_logits_fused(
            grad_beliefs.contiguous(),
            grad_final_belief.contiguous(),
            transition_source_logits_f32.contiguous(),
            transition_dest_logits_f32.contiguous(),
            block_row_ptr.contiguous(),
            block_col_idx.contiguous(),
            block_dst_idx.contiguous(),
            src_row_ptr.contiguous(),
            src_nz_idx.contiguous(),
            grouped_src_row_ptr.contiguous(),
            grouped_src_block_idx.contiguous(),
            block_mask.contiguous(),
            transition_context_saved.contiguous(),
            initial_log_belief_saved.contiguous(),
            beliefs_saved.contiguous(),
            transition_gate_f32.reshape(()),
            transition_stay_probs_f32,
            seq_lens,
            int(ctx.block_size),
            int(ctx.chunk_size),
        )
        grad_local, grad_source_logits, grad_dest_logits, grad_context, grad_initial, grad_gate, grad_stay = grads
        return (
            grad_local,
            grad_source_logits,
            grad_dest_logits,
            grad_context,
            grad_initial,
            grad_gate.reshape_as(transition_gate_f32),
            grad_stay.reshape_as(transition_stay_probs_f32),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_PREFERRED_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES = (64, 96, 128)
_MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES = max(_PREFERRED_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES)


def _causal_machine_scan_shared_bytes(
    transition_rank: int,
    num_states: int = _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES,
) -> int:
    transition_rank = max(int(transition_rank), 1)
    num_states = max(int(num_states), 1)
    num_warps = (num_states + 31) // 32
    forward_bytes = ((2 * num_states * transition_rank) + num_states + transition_rank + (num_warps * transition_rank)) * 4
    backward_base = ((2 * num_states * transition_rank) + (2 * num_states) + (2 * transition_rank) + num_warps) * 4
    backward_direct = ((2 * num_states * transition_rank) + num_states) * 4
    backward_bytes = backward_base + backward_direct
    return max(forward_bytes, backward_bytes)


def _causal_machine_sparse_scan_shared_bytes(num_states: int) -> int:
    states = max(int(num_states), 1)
    block_threads = max(32, min(256, 1 << (min(states, 256) - 1).bit_length()))
    num_warps = (block_threads + 31) // 32
    forward_bytes = ((2 * states) + num_warps + 4) * 4
    backward_bytes = ((4 * states) + num_warps) * 4
    return max(forward_bytes, backward_bytes)


def _supports_structured_scan_num_states(num_states: int) -> bool:
    states = int(num_states)
    return 1 <= states <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES


def _supports_structured_scan_transition_rank(transition_rank: int) -> bool:
    rank = int(transition_rank)
    return 1 <= rank <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES


def _default_structured_scan_tile_size(num_states: int) -> int:
    states = max(int(num_states), 1)
    if states <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        return _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES
    return _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES if states >= 256 else 96


def _default_structured_scan_split_size(num_states: int, transition_rank: int) -> int:
    states = max(int(num_states), 1)
    rank = max(int(transition_rank), 1)
    if states <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES or rank <= 64:
        return rank
    if states >= 512:
        return min(rank, 64)
    if states >= 384:
        return min(rank, 96)
    return min(rank, _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES)


def _structured_scan_tiled_candidate_sizes(limit: int, preferred: int) -> tuple[int, ...]:
    capped_limit = max(min(int(limit), 256), 1)
    values: list[int] = []
    for candidate in (
        preferred,
        capped_limit,
        256,
        192,
        160,
        128,
        96,
        64,
        48,
        32,
        24,
        16,
        8,
        4,
        2,
        1,
    ):
        value = min(max(int(candidate), 1), capped_limit)
        if value not in values:
            values.append(value)
    return tuple(values)


@_torch_dynamo_disable_if_available
@functools.lru_cache(maxsize=None)
def _resolve_structured_scan_tiled_kernel_config_cached(
    device_index: int,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    chunk_size: int,
    preferred_tile_size: int,
    preferred_split_size: int,
    needs_grad: bool,
    ) -> tuple[int, int] | None:
    try:
        ext = load_causal_machine_scan_cuda()
    except Exception:
        ext = None
    if ext is not None and hasattr(ext, "select_tiled_runtime_policy"):
        try:
            info = ext.select_tiled_runtime_policy(
                int(num_states),
                int(transition_rank),
                int(seq_len),
                int(chunk_size),
                bool(needs_grad),
                int(device_index),
            )
            if not bool(info.get("selected", False)):
                return None
            if not bool(info.get("custom_kernel_supported", False)):
                return None
            if bool(needs_grad) and not bool(info.get("custom_backward_kernel_supported", False)):
                return None
            tile_size = int(info.get("tile_size", 0))
            split_size = int(info.get("split_size", 0))
            if tile_size > 0 and split_size > 0:
                return tile_size, split_size
        except Exception:
            pass
    device = torch.device("cuda", int(device_index))
    tile_candidates = _structured_scan_tiled_candidate_sizes(num_states, preferred_tile_size)
    split_candidates = _structured_scan_tiled_candidate_sizes(transition_rank, preferred_split_size)
    for tile_size in tile_candidates:
        for split_size in split_candidates:
            if not _can_use_causal_machine_tiled_forward_kernel(
                device,
                num_states=int(num_states),
                transition_rank=int(transition_rank),
                seq_len=int(seq_len),
                chunk_size=int(chunk_size),
                tile_size=int(tile_size),
                split_size=int(split_size),
            ):
                continue
            if bool(needs_grad) and not _can_use_causal_machine_tiled_backward_kernel(
                device,
                num_states=int(num_states),
                transition_rank=int(transition_rank),
                seq_len=int(seq_len),
                chunk_size=int(chunk_size),
                tile_size=int(tile_size),
                split_size=int(split_size),
            ):
                continue
            return int(tile_size), int(split_size)
    return None


@_torch_dynamo_disable_if_available
@functools.lru_cache(maxsize=None)
def _select_structured_scan_tiled_policy_cached(
    device_index: int,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    chunk_size: int,
    needs_grad: bool,
) -> dict[str, Any] | None:
    try:
        ext = load_causal_machine_scan_cuda()
    except Exception:
        return None
    if not hasattr(ext, "select_tiled_runtime_policy"):
        return None
    try:
        info = ext.select_tiled_runtime_policy(
            int(num_states),
            int(transition_rank),
            int(seq_len),
            int(chunk_size),
            bool(needs_grad),
            int(device_index),
        )
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    if not bool(info.get("selected", False)):
        return None
    if not bool(info.get("custom_kernel_supported", False)):
        return None
    if bool(needs_grad) and not bool(info.get("custom_backward_kernel_supported", False)):
        return None
    return dict(info)


def _resolve_structured_scan_tiled_kernel_config(
    device: torch.device,
    *,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    chunk_size: int,
    preferred_tile_size: int,
    preferred_split_size: int,
    needs_grad: bool,
    runtime_config: StructuredScanRuntimeConfig | None = None,
) -> tuple[int, int] | None:
    if device.type != "cuda":
        return None
    if int(num_states) <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        return None
    if int(transition_rank) <= 0 or int(transition_rank) > int(num_states):
        return None
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    if _has_native_structured_score_filtering(runtime_config, num_states=int(num_states)):
        topk = _resolve_native_structured_score_topk(runtime_config, num_states=int(num_states))
        if topk > 32:
            return None
        tile_candidates = _structured_scan_tiled_candidate_sizes(num_states, preferred_tile_size)
        split_candidates = _structured_scan_tiled_candidate_sizes(transition_rank, preferred_split_size)
        for tile_size in tile_candidates:
            for split_size in split_candidates:
                if not _can_use_causal_machine_tiled_forward_kernel(
                    device,
                    num_states=int(num_states),
                    transition_rank=int(transition_rank),
                    seq_len=int(seq_len),
                    chunk_size=int(chunk_size),
                    tile_size=int(tile_size),
                    split_size=int(split_size),
                    runtime_config=runtime_config,
                ):
                    continue
                if bool(needs_grad) and not _can_use_causal_machine_tiled_backward_kernel(
                    device,
                    num_states=int(num_states),
                    transition_rank=int(transition_rank),
                    seq_len=int(seq_len),
                    chunk_size=int(chunk_size),
                    tile_size=int(tile_size),
                    split_size=int(split_size),
                    runtime_config=runtime_config,
                ):
                    continue
                return int(tile_size), int(split_size)
        return None
    return _resolve_structured_scan_tiled_kernel_config_cached(
        int(device_index),
        int(num_states),
        int(transition_rank),
        int(seq_len),
        int(chunk_size),
        int(preferred_tile_size),
        int(preferred_split_size),
        bool(needs_grad),
    )


def _select_structured_scan_tiled_policy(
    device: torch.device,
    *,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    chunk_size: int,
    needs_grad: bool,
) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    return _select_structured_scan_tiled_policy_cached(
        int(device_index),
        int(num_states),
        int(transition_rank),
        int(seq_len),
        int(chunk_size),
        bool(needs_grad),
    )


@_torch_dynamo_disable_if_available
@functools.lru_cache(maxsize=None)
def _select_structured_scan_dense_policy_cached(
    device_index: int,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    chunk_size: int,
    needs_grad: bool,
) -> dict[str, Any] | None:
    try:
        ext = load_causal_machine_scan_cuda()
    except Exception:
        return None
    if not hasattr(ext, "select_dense_runtime_policy"):
        return None
    try:
        info = ext.select_dense_runtime_policy(
            int(num_states),
            int(transition_rank),
            int(seq_len),
            int(chunk_size),
            bool(needs_grad),
            int(device_index),
        )
    except Exception:
        return None
    return dict(info) if isinstance(info, dict) else None


def _select_structured_scan_dense_policy(
    device: torch.device,
    *,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    chunk_size: int,
    needs_grad: bool,
) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    return _select_structured_scan_dense_policy_cached(
        int(device_index),
        int(num_states),
        int(transition_rank),
        int(seq_len),
        int(chunk_size),
        bool(needs_grad),
    )


def _is_optimized_structured_scan_transition_rank(transition_rank: int) -> bool:
    return int(transition_rank) in (8, 16, 32, 64, _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES)


def _can_use_causal_machine_scan_cuda(
    device: torch.device,
    transition_rank: int,
    num_states: int = _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES,
) -> bool:
    if not _supports_structured_scan_transition_rank(transition_rank):
        return False
    if device.type != "cuda":
        return False
    effective_num_states = (
        _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES
        if int(num_states) < _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES and int(transition_rank) > int(num_states)
        else int(num_states)
    )
    required_bytes = _causal_machine_scan_shared_bytes(transition_rank, num_states=effective_num_states)
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return False
    max_optin = getattr(props, "shared_memory_per_block_optin", 0)
    if max_optin <= 0:
        max_optin = getattr(props, "shared_memory_per_block", 0)
    return required_bytes <= int(max_optin)


def _can_use_causal_machine_masked_scan_cuda(
    device: torch.device,
    *,
    num_states: int,
    transition_rank: int,
    needs_grad: bool = False,
    runtime_config: StructuredScanRuntimeConfig | None = None,
) -> bool:
    if device.type != "cuda":
        return False
    states = int(num_states)
    rank = int(transition_rank)
    if states <= 0 or rank <= 0 or rank > states:
        return False
    if _has_native_structured_score_filtering(runtime_config, num_states=states):
        if states <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
            return False
        topk = _resolve_native_structured_score_topk(runtime_config, num_states=states)
        if topk > 32:
            return False
    try:
        ext = load_causal_machine_scan_cuda()
    except Exception:
        return False
    if supports_structured_scan_cuda_config(states, rank):
        if not hasattr(ext, "forward_masked_logits"):
            return False
        if bool(needs_grad) and not hasattr(ext, "backward_masked_logits"):
            return False
        return _can_use_causal_machine_scan_cuda(device, rank, num_states=states)
    if not hasattr(ext, "forward_masked_logits"):
        return False
    if bool(needs_grad) and not hasattr(ext, "backward_masked_logits"):
        return False
    if not hasattr(ext, "describe_masked_tiled_runtime_config"):
        try:
            props = torch.cuda.get_device_properties(device)
        except Exception:
            return False
        max_optin = int(getattr(props, "shared_memory_per_block_optin", 0) or getattr(props, "shared_memory_per_block", 0))
        if max_optin <= 0:
            return False
        required_threads = min(max(states, 1), 256)
        block_threads = max(32, min(256, 1 << (required_threads - 1).bit_length()))
        num_warps = (block_threads + 31) // 32
        forward_bytes = ((3 * states) + num_warps + 4) * 4
        backward_bytes = ((6 * states) + num_warps + 4) * 4
        return max(forward_bytes, backward_bytes) <= max_optin
    try:
        device_index = int(device.index if device.index is not None else torch.cuda.current_device())
        info = ext.describe_masked_tiled_runtime_config(
            int(states),
            1,
            device_index,
            bool(needs_grad),
        )
    except Exception:
        return False
    return bool(
        info.get(
            "runtime_supported",
            bool(info.get("custom_kernel_supported", False))
            or bool(info.get("extension_fallback_supported", False)),
        )
    )


def _can_use_causal_machine_sparse_scan_cuda(
    device: torch.device,
    *,
    num_states: int,
    transition_rank: int,
) -> bool:
    if device.type != "cuda":
        return False
    states = int(num_states)
    rank = int(transition_rank)
    if states <= 0 or rank <= 0 or rank > states:
        return False
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return False
    max_optin = getattr(props, "shared_memory_per_block_optin", 0)
    if max_optin <= 0:
        max_optin = getattr(props, "shared_memory_per_block", 0)
    return _causal_machine_sparse_scan_shared_bytes(states) <= int(max_optin)


def _can_use_causal_machine_tiled_scan_cuda(
    device: torch.device,
    *,
    num_states: int,
    transition_rank: int,
    needs_grad: bool = False,
    runtime_config: StructuredScanRuntimeConfig | None = None,
) -> bool:
    if device.type != "cuda":
        return False
    if int(num_states) <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        return False
    if int(transition_rank) <= 0 or int(transition_rank) > int(num_states):
        return False
    if _has_native_structured_score_filtering(runtime_config, num_states=int(num_states)):
        topk = _resolve_native_structured_score_topk(runtime_config, num_states=int(num_states))
        if topk > 32:
            return False
    try:
        ext = load_causal_machine_scan_cuda()
    except Exception:
        return False
    if not hasattr(ext, "forward_tiled_logits"):
        return False
    if bool(needs_grad) and not hasattr(ext, "backward_tiled_probs"):
        return False
    return True


def _can_use_causal_machine_tiled_forward_kernel(
    device: torch.device,
    *,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    chunk_size: int,
    tile_size: int,
    split_size: int,
    runtime_config: StructuredScanRuntimeConfig | None = None,
) -> bool:
    if device.type != "cuda":
        return False
    if int(num_states) <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        return False
    if int(transition_rank) <= 0 or int(transition_rank) > int(num_states):
        return False
    if int(seq_len) < 0 or int(chunk_size) <= 0:
        return False
    if int(tile_size) <= 0 or int(split_size) <= 0:
        return False
    if _has_native_structured_score_filtering(runtime_config, num_states=int(num_states)):
        topk = _resolve_native_structured_score_topk(runtime_config, num_states=int(num_states))
        if topk > 32:
            return False
    try:
        ext = load_causal_machine_scan_cuda()
    except Exception:
        return False
    if not hasattr(ext, "describe_tiled_runtime_config"):
        return int(max(tile_size, split_size)) <= 256
    try:
        device_index = int(device.index if device.index is not None else torch.cuda.current_device())
        info = ext.describe_tiled_runtime_config(
            int(num_states),
            int(transition_rank),
            int(seq_len),
            int(chunk_size),
            int(tile_size),
            int(split_size),
            1,
            device_index,
        )
    except Exception:
        return False
    return bool(info.get("custom_kernel_supported", False))


def _can_use_causal_machine_tiled_backward_kernel(
    device: torch.device,
    *,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    chunk_size: int,
    tile_size: int,
    split_size: int,
    runtime_config: StructuredScanRuntimeConfig | None = None,
) -> bool:
    if device.type != "cuda":
        return False
    if int(num_states) <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        return False
    if int(transition_rank) <= 0 or int(transition_rank) > int(num_states):
        return False
    if int(seq_len) < 0 or int(chunk_size) <= 0:
        return False
    if int(tile_size) <= 0 or int(split_size) <= 0:
        return False
    if _has_native_structured_score_filtering(runtime_config, num_states=int(num_states)):
        topk = _resolve_native_structured_score_topk(runtime_config, num_states=int(num_states))
        if topk > 32:
            return False
    try:
        ext = load_causal_machine_scan_cuda()
    except Exception:
        return False
    if not hasattr(ext, "describe_tiled_runtime_config"):
        return int(max(tile_size, split_size)) <= 256
    try:
        device_index = int(device.index if device.index is not None else torch.cuda.current_device())
        info = ext.describe_tiled_runtime_config(
            int(num_states),
            int(transition_rank),
            int(seq_len),
            int(chunk_size),
            int(tile_size),
            int(split_size),
            1,
            device_index,
        )
    except Exception:
        return False
    return bool(info.get("custom_backward_kernel_supported", False))


def create_structured_scan_workspace(
    *,
    mode: str,
    device: torch.device,
    num_states: int,
    transition_rank: int,
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    tile_size: int,
    split_size: int,
) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    try:
        ext = load_causal_machine_scan_cuda()
    except Exception:
        return None
    if not hasattr(ext, "create_scan_workspace"):
        return None
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    return ext.create_scan_workspace(
        str(mode),
        int(num_states),
        int(transition_rank),
        int(batch_size),
        int(tile_size),
        int(split_size),
        int(seq_len),
        int(chunk_size),
        int(device_index),
    )


@_torch_dynamo_disable_if_available
@functools.lru_cache(maxsize=None)
def _describe_structured_scan_device_runtime_cached(device_index: int) -> dict[str, int]:
    device = torch.device("cuda", int(device_index))
    info: dict[str, int] = {
        "device_index": int(device_index),
        "capability_major": 0,
        "capability_minor": 0,
        "sm_count": 0,
        "max_dynamic_smem_bytes": 0,
        "total_global_mem_bytes": 0,
        "supports_tma": 0,
        "supports_wgmma": 0,
    }
    try:
        # Do not trigger a build/load just to answer autotune heuristics.
        # If the extension is already resident, use its richer metadata.
        ext = _CAUSAL_MACHINE_SCAN_CUDA
        if hasattr(ext, "describe_device_runtime_config"):
            raw = ext.describe_device_runtime_config(int(device_index))
            info.update({str(k): int(v) for k, v in raw.items()})
            return info
    except Exception:
        pass
    try:
        major, minor = torch.cuda.get_device_capability(device)
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "capability_major": int(major),
                "capability_minor": int(minor),
                "sm_count": int(getattr(props, "multi_processor_count", 0)),
                "max_dynamic_smem_bytes": int(
                    getattr(props, "shared_memory_per_block_optin", 0)
                    or getattr(props, "shared_memory_per_block", 0)
                ),
                "total_global_mem_bytes": int(getattr(props, "total_memory", 0)),
                "supports_tma": 0,
                "supports_wgmma": 0,
            }
        )
    except Exception:
        pass
    return info


def _describe_structured_scan_device_runtime(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "device_index": -1,
            "capability_major": 0,
            "capability_minor": 0,
            "sm_count": 0,
            "max_dynamic_smem_bytes": 0,
            "total_global_mem_bytes": 0,
        }
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    return _describe_structured_scan_device_runtime_cached(device_index)


def _estimate_structured_scan_cost(
    seq_len: int,
    transition_rank: int,
    num_states: int = _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES,
) -> float:
    seq_len = max(int(seq_len), 1)
    transition_rank = max(int(transition_rank), 1)
    num_states = max(int(num_states), 1)
    transition_work = 2.0 * seq_len * num_states * transition_rank
    norm_work = 4.0 * seq_len * num_states
    return transition_work + norm_work


def _estimate_latent_scan_cost(seq_len: int, latent_rank: int, chunk_size: int = 64) -> float:
    seq_len = max(int(seq_len), 1)
    latent_rank = max(int(latent_rank), 1)
    chunk_size = max(int(chunk_size), 1)
    num_chunks = max((seq_len + chunk_size - 1) // chunk_size, 1)
    scan_work = 4.0 * seq_len * latent_rank
    summary_work = 2.0 * num_chunks * latent_rank
    finalize_work = 2.0 * seq_len * latent_rank
    return scan_work + summary_work + finalize_work


@_torch_dynamo_disable_if_available
@functools.lru_cache(maxsize=None)
def _cached_env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(float(default))))


@_torch_dynamo_disable_if_available
@functools.lru_cache(maxsize=None)
def _cached_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(int(default))).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


@_torch_dynamo_disable_if_available
@functools.lru_cache(maxsize=None)
def _cached_env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip().lower()


def _bounded_latent_decay(logits: Tensor) -> Tensor:
    min_decay = _cached_env_float("CAUSAL_MACHINE_LATENT_DECAY_MIN", 0.99)
    max_decay = _cached_env_float("CAUSAL_MACHINE_LATENT_DECAY_MAX", 0.9995)
    min_decay = min(max(min_decay, 1.0e-6), 0.999999)
    max_decay = min(max(max_decay, min_decay + 1e-6), 0.999999)
    logits_f32 = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    decay = torch.sigmoid(logits_f32)
    decay = min_decay + (max_decay - min_decay) * decay
    return torch.nan_to_num(decay, nan=min_decay, posinf=max_decay, neginf=min_decay)


def _bounded_gate(logits: Tensor, min_env: str, max_env: str, *, default_min: float = 0.10, default_max: float = 0.99) -> Tensor:
    min_gate = _cached_env_float(min_env, default_min)
    max_gate = _cached_env_float(max_env, default_max)
    min_gate = min(max(min_gate, 0.0), 0.999999)
    max_gate = min(max(max_gate, min_gate + 1e-6), 0.999999)
    sig = torch.sigmoid(logits.float())
    return min_gate + (max_gate - min_gate) * sig


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _competition_mode_enabled() -> bool:
    return _env_enabled("COMPETITION_MODE", True)


def _require_prebuilt_cuda_extensions() -> bool:
    return _env_enabled("CAUSAL_MACHINE_REQUIRE_PREBUILT_EXTENSIONS", _competition_mode_enabled())


def _require_fused_cuda_path_contract() -> bool:
    return _env_enabled("CAUSAL_MACHINE_REQUIRE_FUSED_CUDA_PATH", _competition_mode_enabled())


def _structured_filter_mode() -> str:
    return _cached_env_str("CAUSAL_MACHINE_STRUCTURED_FILTER_MODE", "coupled")


def _structured_scan_packed_kind() -> int:
    packed_dtype = _cached_env_str("CAUSAL_MACHINE_SCAN_PACKED_DTYPE", "int8")
    mapping = {
        "int8": _PACKED_TRANSITION_INT8,
        "fp8_e4m3": _PACKED_TRANSITION_FP8_E4M3,
        "e4m3": _PACKED_TRANSITION_FP8_E4M3,
        "fp8_e5m2": _PACKED_TRANSITION_FP8_E5M2,
        "e5m2": _PACKED_TRANSITION_FP8_E5M2,
    }
    if packed_dtype not in mapping:
        raise ValueError(
            "CAUSAL_MACHINE_SCAN_PACKED_DTYPE must be one of "
            "'int8', 'fp8_e4m3', or 'fp8_e5m2'"
        )
    return mapping[packed_dtype]


def _softmax_backward_rows(grad_probs: Tensor, probs: Tensor) -> Tensor:
    grad_probs_f32 = grad_probs.float()
    probs_f32 = probs.float()
    proj = (grad_probs_f32 * probs_f32).sum(dim=-1, keepdim=True)
    return (probs_f32 * (grad_probs_f32 - proj)).contiguous()


def _resolve_structured_scan_chunk_size(
    train_seq_len: int,
    transition_rank: int,
    *,
    env_name: str = "CAUSAL_MACHINE_FILTER_CHUNK_SIZE",
) -> int:
    env_value = _cached_env_int(env_name, -1)
    if env_value > 0:
        return max(8, env_value)
    seq_len = max(int(train_seq_len), 16)
    rank = max(int(transition_rank), 1)
    if rank <= 16:
        target = min(64, seq_len // 8 if seq_len >= 512 else seq_len // 4)
    elif rank <= 32:
        target = min(96, seq_len // 8 if seq_len >= 512 else seq_len // 4)
    else:
        target = min(128, seq_len // 8 if seq_len >= 512 else seq_len // 4)
    return max(16, target)


def _prepare_structured_filter_inputs(
    transition_context: Tensor,
    transition_gate: Tensor,
) -> tuple[Tensor, Tensor]:
    if _structured_filter_mode() != "composable":
        return transition_context, transition_gate
    effective_context = transition_context * transition_gate.to(
        device=transition_context.device,
        dtype=transition_context.dtype,
    )
    effective_gate = torch.ones_like(transition_gate, dtype=torch.float32)
    return effective_context, effective_gate


def causal_machine_scan_cuda(
    local_logits: Tensor,
    transition_source_logits: Tensor,
    transition_dest_logits: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_gate: Tensor,
    transition_stay_probs: Tensor,
    packed_transition_tables: tuple[int, Tensor, Tensor, Tensor, Tensor] | None = None,
    chunk_size: int = 64,
    runtime_config: StructuredScanRuntimeConfig | None = None,
) -> tuple[Tensor, Tensor]:
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=local_logits.device,
        dtype=local_logits.dtype,
        backend_family="cuda",
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        num_states=int(local_logits.size(2)),
    )
    score_clamp_min, score_clamp_max = _resolve_native_structured_score_clamp_bounds(runtime_config)
    clamp_active = math.isfinite(score_clamp_min) or math.isfinite(score_clamp_max)
    use_packed_tables = packed_transition_tables is not None
    if clamp_active:
        use_packed_tables = False
    if not use_packed_tables:
        empty_q = torch.empty((0, 0), device=transition_source_logits.device, dtype=torch.int8)
        empty_s = torch.empty((0,), device=transition_source_logits.device, dtype=torch.float32)
        packed_kind = _PACKED_TRANSITION_INT8
        packed_source_q, packed_source_scales, packed_dest_q, packed_dest_scales = empty_q, empty_s, empty_q, empty_s
        use_packed = False
    else:
        assert packed_transition_tables is not None
        packed_kind, packed_source_q, packed_source_scales, packed_dest_q, packed_dest_scales = packed_transition_tables
        use_packed = True
    return _CausalMachineScanCudaFn.apply(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        packed_source_q,
        packed_source_scales,
        packed_dest_q,
        packed_dest_scales,
        packed_kind,
        use_packed,
        int(chunk_size),
        float(score_clamp_min),
        float(score_clamp_max),
        False if clamp_active else (_structured_filter_mode() == "composable"),
        )


class _CausalMachineMaskedScanCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_logits: Tensor,
        transition_source_logits: Tensor,
        transition_dest_logits: Tensor,
        transition_context: Tensor,
        initial_log_belief: Tensor,
        transition_gate: Tensor,
        transition_stay_probs: Tensor,
        transition_mask: Tensor,
        seq_lens: Tensor,
        chunk_size: int,
        score_clamp_min: float,
        score_clamp_max: float,
        score_threshold: float,
        score_topk: int,
        workspace: dict[str, Any] | None,
    ) -> tuple[Tensor, Tensor]:
        ext = load_causal_machine_scan_cuda()
        local_logits_in = local_logits.contiguous()
        transition_source_logits_f32 = transition_source_logits.contiguous().float()
        transition_dest_logits_f32 = transition_dest_logits.contiguous().float()
        transition_source_probs_f32 = F.softmax(transition_source_logits_f32, dim=-1).contiguous()
        transition_dest_probs_f32 = F.softmax(transition_dest_logits_f32, dim=-1).contiguous()
        transition_context_in = transition_context.contiguous()
        initial_log_belief_in = initial_log_belief.contiguous()
        transition_stay_probs_f32 = transition_stay_probs.contiguous().float()
        transition_mask_in = transition_mask.contiguous()
        seq_lens_in = seq_lens.contiguous()
        workspace_dict = workspace if isinstance(workspace, dict) else None
        if workspace_dict is not None and hasattr(ext, "forward_masked_logits_bound_workspace"):
            beliefs, final_belief = ext.forward_masked_logits_bound_workspace(
                local_logits_in,
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_in,
                initial_log_belief_in,
                transition_gate.reshape(()).contiguous().float(),
                transition_stay_probs_f32,
                transition_mask_in,
                seq_lens_in,
                int(chunk_size),
                workspace_dict,
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        else:
            beliefs, final_belief = ext.forward_masked_logits(
                local_logits_in,
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_in,
                initial_log_belief_in,
                transition_gate.reshape(()).contiguous().float(),
                transition_stay_probs_f32,
                transition_mask_in,
                seq_lens_in,
                int(chunk_size),
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        ctx.save_for_backward(
            transition_source_probs_f32,
            transition_dest_probs_f32,
            transition_context_in,
            initial_log_belief_in,
            beliefs.contiguous(),
            transition_gate.float(),
            transition_stay_probs_f32,
            transition_mask_in,
            seq_lens_in,
        )
        ctx.chunk_size = int(chunk_size)
        ctx.score_clamp_min = float(score_clamp_min)
        ctx.score_clamp_max = float(score_clamp_max)
        ctx.score_threshold = float(score_threshold)
        ctx.score_topk = int(score_topk)
        ctx.workspace = workspace_dict
        return beliefs, final_belief

    @staticmethod
    def backward(ctx, grad_beliefs: Tensor, grad_final_belief: Tensor):
        ext = load_causal_machine_scan_cuda()
        (
            transition_source_probs_f32,
            transition_dest_probs_f32,
            transition_context_saved,
            initial_log_belief_saved,
            beliefs_saved,
            transition_gate_f32,
            transition_stay_probs_f32,
            transition_mask_saved,
            seq_lens_saved,
        ) = ctx.saved_tensors
        workspace_dict = getattr(ctx, "workspace", None)
        if workspace_dict is not None and hasattr(ext, "backward_masked_logits_bound_workspace"):
            grads = ext.backward_masked_logits_bound_workspace(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_saved.contiguous(),
                initial_log_belief_saved.contiguous(),
                beliefs_saved.contiguous(),
                transition_gate_f32.detach().float().reshape(()),
                transition_stay_probs_f32,
                transition_mask_saved.contiguous(),
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                workspace_dict,
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        else:
            grads = ext.backward_masked_logits(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_saved.contiguous(),
                initial_log_belief_saved.contiguous(),
                beliefs_saved.contiguous(),
                float(transition_gate_f32.detach().float().item()),
                transition_stay_probs_f32,
                transition_mask_saved.contiguous(),
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        grad_local, grad_source_probs, grad_dest_probs, grad_context, grad_initial, grad_gate, grad_stay = grads
        return (
            grad_local,
            _softmax_backward_rows(grad_source_probs, transition_source_probs_f32),
            _softmax_backward_rows(grad_dest_probs, transition_dest_probs_f32),
            grad_context,
            grad_initial,
            grad_gate.reshape_as(transition_gate_f32),
            grad_stay.reshape_as(transition_stay_probs_f32),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _CausalMachineMaskedScanCudaProbFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_logits: Tensor,
        transition_source_probs: Tensor,
        transition_dest_probs: Tensor,
        transition_context: Tensor,
        initial_log_belief: Tensor,
        transition_gate: Tensor,
        transition_stay_probs: Tensor,
        transition_mask: Tensor,
        seq_lens: Tensor,
        chunk_size: int,
        score_clamp_min: float,
        score_clamp_max: float,
        score_threshold: float,
        score_topk: int,
        workspace: dict[str, Any] | None,
    ) -> tuple[Tensor, Tensor]:
        ext = load_causal_machine_scan_cuda()
        local_logits_in = local_logits.contiguous()
        transition_source_probs_f32 = transition_source_probs.contiguous().float()
        transition_dest_probs_f32 = transition_dest_probs.contiguous().float()
        transition_context_in = transition_context.contiguous()
        initial_log_belief_in = initial_log_belief.contiguous()
        transition_stay_probs_f32 = transition_stay_probs.contiguous().float()
        transition_mask_in = transition_mask.contiguous()
        seq_lens_in = seq_lens.contiguous()
        workspace_dict = workspace if isinstance(workspace, dict) else None
        if workspace_dict is not None and hasattr(ext, "forward_masked_logits_bound_workspace"):
            beliefs, final_belief = ext.forward_masked_logits_bound_workspace(
                local_logits_in,
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_in,
                initial_log_belief_in,
                transition_gate.reshape(()).contiguous().float(),
                transition_stay_probs_f32,
                transition_mask_in,
                seq_lens_in,
                int(chunk_size),
                workspace_dict,
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        else:
            beliefs, final_belief = ext.forward_masked_logits(
                local_logits_in,
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_in,
                initial_log_belief_in,
                transition_gate.reshape(()).contiguous().float(),
                transition_stay_probs_f32,
                transition_mask_in,
                seq_lens_in,
                int(chunk_size),
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        ctx.save_for_backward(
            transition_source_probs_f32,
            transition_dest_probs_f32,
            transition_context_in,
            initial_log_belief_in,
            beliefs.contiguous(),
            transition_gate.float(),
            transition_stay_probs_f32,
            transition_mask_in,
            seq_lens_in,
        )
        ctx.chunk_size = int(chunk_size)
        ctx.score_clamp_min = float(score_clamp_min)
        ctx.score_clamp_max = float(score_clamp_max)
        ctx.score_threshold = float(score_threshold)
        ctx.score_topk = int(score_topk)
        ctx.workspace = workspace_dict
        return beliefs, final_belief

    @staticmethod
    def backward(ctx, grad_beliefs: Tensor, grad_final_belief: Tensor):
        ext = load_causal_machine_scan_cuda()
        (
            transition_source_probs_f32,
            transition_dest_probs_f32,
            transition_context_saved,
            initial_log_belief_saved,
            beliefs_saved,
            transition_gate_f32,
            transition_stay_probs_f32,
            transition_mask_saved,
            seq_lens_saved,
        ) = ctx.saved_tensors
        workspace_dict = getattr(ctx, "workspace", None)
        if workspace_dict is not None and hasattr(ext, "backward_masked_logits_bound_workspace"):
            grads = ext.backward_masked_logits_bound_workspace(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_saved.contiguous(),
                initial_log_belief_saved.contiguous(),
                beliefs_saved.contiguous(),
                transition_gate_f32.detach().float().reshape(()),
                transition_stay_probs_f32,
                transition_mask_saved.contiguous(),
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                workspace_dict,
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        else:
            grads = ext.backward_masked_logits(
                grad_beliefs.contiguous(),
                grad_final_belief.contiguous(),
                transition_source_probs_f32,
                transition_dest_probs_f32,
                transition_context_saved.contiguous(),
                initial_log_belief_saved.contiguous(),
                beliefs_saved.contiguous(),
                float(transition_gate_f32.detach().float().item()),
                transition_stay_probs_f32,
                transition_mask_saved.contiguous(),
                seq_lens_saved.contiguous(),
                int(ctx.chunk_size),
                float(ctx.score_clamp_min),
                float(ctx.score_clamp_max),
                float(ctx.score_threshold),
                int(ctx.score_topk),
            )
        grad_local, grad_source_probs, grad_dest_probs, grad_context, grad_initial, grad_gate, grad_stay = grads
        return (
            grad_local,
            grad_source_probs,
            grad_dest_probs,
            grad_context,
            grad_initial,
            grad_gate.reshape_as(transition_gate_f32),
            grad_stay.reshape_as(transition_stay_probs_f32),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def causal_machine_scan_masked_cuda(
    local_logits: Tensor,
    transition_source_logits: Tensor,
    transition_dest_logits: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_gate: Tensor,
    transition_stay_probs: Tensor,
    *,
    runtime_config: StructuredScanRuntimeConfig | None,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor]:
    if runtime_config is None:
        raise ValueError("runtime_config is required for masked CUDA structured scan")
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=local_logits.device,
        dtype=local_logits.dtype,
        backend_family="masked_cuda",
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        num_states=int(local_logits.size(2)),
    )
    score_clamp_min, score_clamp_max = _resolve_native_structured_score_clamp_bounds(runtime_config)
    score_threshold = _resolve_native_structured_score_threshold(runtime_config)
    score_topk = _resolve_native_structured_score_topk(runtime_config, num_states=int(local_logits.size(-1)))
    transition_mask = _build_structured_transition_mask(
        int(local_logits.size(-1)),
        device=local_logits.device,
        runtime_config=runtime_config,
    )
    if transition_mask is None:
        raise ValueError("masked CUDA structured scan requires a transition mask")
    seq_lens = _canonicalize_structured_seq_lens(
        runtime_config.seq_lens,
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        device=local_logits.device,
    )
    ext = load_causal_machine_scan_cuda()
    empty_seq_lens = torch.empty((0,), device=local_logits.device, dtype=torch.int64)
    seq_lens_in = seq_lens.contiguous() if seq_lens is not None else empty_seq_lens
    transition_rank = int(transition_source_logits.size(1))
    num_states = int(local_logits.size(-1))
    if _has_native_structured_score_filtering(runtime_config, num_states=num_states) and num_states <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        raise RuntimeError("masked CUDA native threshold/topk requires the large-state masked tiled custom kernel path")
    needs_grad = (
        torch.is_grad_enabled()
        and (
            local_logits.requires_grad
            or transition_source_logits.requires_grad
            or transition_dest_logits.requires_grad
            or transition_context.requires_grad
            or initial_log_belief.requires_grad
            or transition_gate.requires_grad
            or transition_stay_probs.requires_grad
        )
    )
    masked_cuda_autograd = hasattr(ext, "backward_masked_logits")
    workspace_dict: dict[str, Any] | None = None
    if runtime_config is not None and hasattr(ext, "create_scan_workspace"):
        workspace_mode = "masked_tiled_backward" if needs_grad else "masked_tiled_forward"
        if runtime_config.backend_policy is not None:
            workspace_mode = (
                str(runtime_config.backend_policy.workspace_mode_backward)
                if needs_grad
                else str(runtime_config.backend_policy.workspace_mode)
            )
        workspace_dict = create_structured_scan_workspace(
            mode=workspace_mode,
            device=local_logits.device,
            num_states=int(local_logits.size(-1)),
            transition_rank=int(transition_source_logits.size(-1)),
            batch_size=int(local_logits.size(0)),
            seq_len=int(local_logits.size(1)),
            chunk_size=int(chunk_size),
            tile_size=int(local_logits.size(-1)),
            split_size=1,
        )
    if not hasattr(ext, "forward_masked_logits"):
        raise RuntimeError(
            "masked CUDA structured scan requires forward_masked_logits and backward_masked_logits in the CUDA extension"
        )
    if needs_grad and masked_cuda_autograd:
        beliefs, final_belief = _CausalMachineMaskedScanCudaFn.apply(
            local_logits,
            transition_source_logits,
            transition_dest_logits,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            transition_mask,
            seq_lens_in,
            int(chunk_size),
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
            workspace_dict,
        )
    elif not needs_grad:
        transition_source_probs = F.softmax(
            transition_source_logits.contiguous().float(),
            dim=-1,
        ).contiguous()
        transition_dest_probs = F.softmax(
            transition_dest_logits.contiguous().float(),
            dim=-1,
        ).contiguous()
        if workspace_dict is not None and hasattr(ext, "forward_masked_logits_bound_workspace"):
            beliefs, final_belief = ext.forward_masked_logits_bound_workspace(
                local_logits.contiguous(),
                transition_source_probs,
                transition_dest_probs,
                transition_context.contiguous(),
                initial_log_belief.contiguous(),
                transition_gate.reshape(()).contiguous().float(),
                transition_stay_probs.contiguous().float(),
                transition_mask.contiguous(),
                seq_lens_in,
                int(chunk_size),
                workspace_dict,
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        else:
            beliefs, final_belief = ext.forward_masked_logits(
                local_logits.contiguous(),
                transition_source_probs,
                transition_dest_probs,
                transition_context.contiguous(),
                initial_log_belief.contiguous(),
                transition_gate.reshape(()).contiguous().float(),
                transition_stay_probs.contiguous().float(),
                transition_mask.contiguous(),
                seq_lens_in,
                int(chunk_size),
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
    else:
        raise RuntimeError(
            "masked CUDA structured scan requires forward_masked_logits and backward_masked_logits in the CUDA extension"
        )
    return beliefs, final_belief


def causal_machine_scan_masked_probs_cuda(
    local_logits: Tensor,
    transition_source_probs: Tensor,
    transition_dest_probs: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_gate: Tensor,
    transition_stay_probs: Tensor,
    *,
    runtime_config: StructuredScanRuntimeConfig | None,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor]:
    if runtime_config is None:
        raise ValueError("runtime_config is required for masked CUDA structured scan")
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=local_logits.device,
        dtype=local_logits.dtype,
        backend_family="masked_cuda",
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        num_states=int(local_logits.size(2)),
    )
    score_clamp_min, score_clamp_max = _resolve_native_structured_score_clamp_bounds(runtime_config)
    score_threshold = _resolve_native_structured_score_threshold(runtime_config)
    score_topk = _resolve_native_structured_score_topk(runtime_config, num_states=int(local_logits.size(-1)))
    transition_mask = _build_structured_transition_mask(
        int(local_logits.size(-1)),
        device=local_logits.device,
        runtime_config=runtime_config,
    )
    if transition_mask is None:
        raise ValueError("masked CUDA structured scan requires a transition mask")
    seq_lens = _canonicalize_structured_seq_lens(
        runtime_config.seq_lens,
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        device=local_logits.device,
    )
    ext = load_causal_machine_scan_cuda()
    empty_seq_lens = torch.empty((0,), device=local_logits.device, dtype=torch.int64)
    seq_lens_in = seq_lens.contiguous() if seq_lens is not None else empty_seq_lens
    if _has_native_structured_score_filtering(runtime_config, num_states=int(local_logits.size(-1))) and int(local_logits.size(-1)) <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        raise RuntimeError("masked CUDA native threshold/topk requires the large-state masked tiled custom kernel path")
    needs_grad = (
        torch.is_grad_enabled()
        and (
            local_logits.requires_grad
            or transition_source_probs.requires_grad
            or transition_dest_probs.requires_grad
            or transition_context.requires_grad
            or initial_log_belief.requires_grad
            or transition_gate.requires_grad
            or transition_stay_probs.requires_grad
        )
    )
    if not hasattr(ext, "forward_masked_logits"):
        raise RuntimeError(
            "masked CUDA structured scan requires forward_masked_logits and backward_masked_logits in the CUDA extension"
        )
    workspace_dict: dict[str, Any] | None = None
    if runtime_config is not None and hasattr(ext, "create_scan_workspace"):
        workspace_mode = "masked_tiled_backward" if needs_grad else "masked_tiled_forward"
        if runtime_config.backend_policy is not None:
            workspace_mode = (
                str(runtime_config.backend_policy.workspace_mode_backward)
                if needs_grad
                else str(runtime_config.backend_policy.workspace_mode)
            )
        workspace_dict = create_structured_scan_workspace(
            mode=workspace_mode,
            device=local_logits.device,
            num_states=int(local_logits.size(-1)),
            transition_rank=int(transition_source_probs.size(-1)),
            batch_size=int(local_logits.size(0)),
            seq_len=int(local_logits.size(1)),
            chunk_size=int(chunk_size),
            tile_size=int(local_logits.size(-1)),
            split_size=1,
        )
    if needs_grad and hasattr(ext, "backward_masked_logits"):
        return _CausalMachineMaskedScanCudaProbFn.apply(
            local_logits,
            transition_source_probs,
            transition_dest_probs,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            transition_mask,
            seq_lens_in,
            int(chunk_size),
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
            workspace_dict,
        )
    if needs_grad:
        raise RuntimeError(
            "masked CUDA structured scan requires forward_masked_logits and backward_masked_logits in the CUDA extension"
        )
    if workspace_dict is not None and hasattr(ext, "forward_masked_logits_bound_workspace"):
        return ext.forward_masked_logits_bound_workspace(
            local_logits.contiguous(),
            transition_source_probs.contiguous().float(),
            transition_dest_probs.contiguous().float(),
            transition_context.contiguous(),
            initial_log_belief.contiguous(),
            transition_gate.reshape(()).contiguous().float(),
            transition_stay_probs.contiguous().float(),
            transition_mask.contiguous(),
            seq_lens_in,
            int(chunk_size),
            workspace_dict,
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
        )
    return ext.forward_masked_logits(
        local_logits.contiguous(),
        transition_source_probs.contiguous().float(),
        transition_dest_probs.contiguous().float(),
        transition_context.contiguous(),
        initial_log_belief.contiguous(),
        transition_gate.reshape(()).contiguous().float(),
        transition_stay_probs.contiguous().float(),
        transition_mask.contiguous(),
        seq_lens_in,
        int(chunk_size),
        float(score_clamp_min),
        float(score_clamp_max),
        float(score_threshold),
        int(score_topk),
    )


def _execute_structured_sparse_runtime_cuda(
    local_logits: Tensor,
    transition_source_logits: Tensor,
    transition_dest_logits: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_gate: Tensor,
    transition_stay_probs: Tensor,
    sparse_transition_tables: StructuredSparseTransitionTables | None,
    *,
    runtime_config: StructuredScanRuntimeConfig | None,
    chunk_size: int,
    needs_grad: bool,
) -> tuple[Tensor, Tensor] | None:
    if sparse_transition_tables is None:
        return None
    grouped_block_count = max(int(sparse_transition_tables.grouped_src_group_count), 0)
    grouped_sparse_gain = (
        grouped_block_count > 0
        and grouped_block_count < int(sparse_transition_tables.col_idx.numel())
    )
    if not (
        float(sparse_transition_tables.density) < 1.0
        or _structured_runtime_prefers_sparse_cuda(
            runtime_config,
            num_states=int(local_logits.size(-1)),
        )
        or grouped_sparse_gain
    ):
        return None
    if _can_use_causal_machine_sparse_scan_cuda(
        local_logits.device,
        num_states=int(local_logits.size(-1)),
        transition_rank=int(transition_source_logits.size(-1)),
    ):
        if needs_grad:
            return causal_machine_scan_sparse_cuda_autograd(
                local_logits,
                transition_source_logits,
                transition_dest_logits,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                sparse_transition_tables,
                runtime_config=runtime_config,
                chunk_size=int(chunk_size),
            )
        return causal_machine_scan_sparse_cuda(
            local_logits,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            sparse_transition_tables,
            runtime_config=runtime_config,
            chunk_size=int(chunk_size),
        )
    if not _structured_runtime_supports_masked_cuda(runtime_config):
        return None
    if not _can_use_causal_machine_masked_scan_cuda(
        local_logits.device,
        num_states=int(local_logits.size(-1)),
        transition_rank=int(transition_source_logits.size(-1)),
        needs_grad=bool(needs_grad),
        runtime_config=runtime_config,
    ):
        return None
    return causal_machine_scan_masked_cuda(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        runtime_config=runtime_config,
        chunk_size=int(chunk_size),
    )


def causal_machine_scan_tiled_cuda(
    local_logits: Tensor,
    transition_source_probs: Tensor,
    transition_dest_probs: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_gate: Tensor,
    transition_stay_probs: Tensor,
    *,
    runtime_config: StructuredScanRuntimeConfig | None,
    chunk_size: int,
    tile_size: int,
    split_size: int,
    packed_transition_tables: tuple[int, Tensor, Tensor, Tensor, Tensor] | None = None,
    workspace: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor]:
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=local_logits.device,
        dtype=local_logits.dtype,
        backend_family="cuda_tiled",
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        num_states=int(local_logits.size(2)),
    )
    if runtime_config is not None:
        if _structured_runtime_score_mod_callback(runtime_config) is not None:
            raise ValueError("tiled CUDA structured scan does not support callback-based score_mod")
        if _structured_runtime_uses_transition_masking(runtime_config):
            return causal_machine_scan_masked_probs_cuda(
                local_logits,
                transition_source_probs,
                transition_dest_probs,
                transition_context,
                initial_log_belief,
                transition_gate,
                transition_stay_probs,
                runtime_config=runtime_config,
                chunk_size=int(chunk_size),
            )
    seq_lens = _canonicalize_structured_seq_lens(
        runtime_config.seq_lens if runtime_config is not None else None,
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        device=local_logits.device,
    )
    ext = load_causal_machine_scan_cuda()
    empty_seq_lens = torch.empty((0,), device=local_logits.device, dtype=torch.int64)
    score_clamp_min, score_clamp_max = _resolve_native_structured_score_clamp_bounds(runtime_config)
    score_threshold = _resolve_native_structured_score_threshold(runtime_config)
    score_topk = _resolve_native_structured_score_topk(runtime_config, num_states=int(local_logits.size(-1)))
    use_packed_tables = packed_transition_tables is not None
    if use_packed_tables:
        packed_kind, packed_source_q, packed_source_scales, packed_dest_q, packed_dest_scales = packed_transition_tables
        if packed_kind == _PACKED_TRANSITION_INT8:
            use_packed_tables = hasattr(ext, "forward_tiled_quantized_kernel") and (
                not torch.is_grad_enabled() or hasattr(ext, "backward_tiled_quantized_kernel")
            )
        elif packed_kind in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}:
            use_packed_tables = hasattr(ext, "forward_tiled_fp8_kernel") and (
                not torch.is_grad_enabled() or hasattr(ext, "backward_tiled_fp8_kernel")
            )
        else:
            use_packed_tables = False
    if not use_packed_tables:
        packed_kind = _PACKED_TRANSITION_INT8
        packed_source_q = torch.empty((0, 0), device=local_logits.device, dtype=torch.int8)
        packed_source_scales = torch.empty((0,), device=local_logits.device, dtype=torch.float32)
        packed_dest_q = torch.empty((0, 0), device=local_logits.device, dtype=torch.int8)
        packed_dest_scales = torch.empty((0,), device=local_logits.device, dtype=torch.float32)
    needs_grad = (
        torch.is_grad_enabled()
        and (
            local_logits.requires_grad
            or transition_source_probs.requires_grad
            or transition_dest_probs.requires_grad
            or transition_context.requires_grad
            or initial_log_belief.requires_grad
            or transition_gate.requires_grad
            or transition_stay_probs.requires_grad
        )
    )
    can_use_custom_tiled_forward_kernel = (
        hasattr(ext, "forward_tiled_logits_kernel")
        and _can_use_causal_machine_tiled_forward_kernel(
            local_logits.device,
            num_states=int(local_logits.size(-1)),
            transition_rank=int(transition_source_probs.size(-1)),
            seq_len=int(local_logits.size(1)),
            chunk_size=int(chunk_size),
            tile_size=int(tile_size),
            split_size=int(split_size),
            runtime_config=runtime_config,
        )
    )
    can_use_custom_tiled_backward_kernel = (
        needs_grad
        and hasattr(ext, "backward_tiled_probs_kernel")
        and _can_use_causal_machine_tiled_backward_kernel(
            local_logits.device,
            num_states=int(local_logits.size(-1)),
            transition_rank=int(transition_source_probs.size(-1)),
            seq_len=int(local_logits.size(1)),
            chunk_size=int(chunk_size),
            tile_size=int(tile_size),
            split_size=int(split_size),
            runtime_config=runtime_config,
        )
    )
    if use_packed_tables and (
        not can_use_custom_tiled_forward_kernel
        or (needs_grad and not can_use_custom_tiled_backward_kernel)
    ):
        use_packed_tables = False
        packed_kind = _PACKED_TRANSITION_INT8
        packed_source_q = torch.empty((0, 0), device=local_logits.device, dtype=torch.int8)
        packed_source_scales = torch.empty((0,), device=local_logits.device, dtype=torch.float32)
        packed_dest_q = torch.empty((0, 0), device=local_logits.device, dtype=torch.int8)
        packed_dest_scales = torch.empty((0,), device=local_logits.device, dtype=torch.float32)
    native_filtering_active = _has_native_structured_score_filtering(runtime_config, num_states=int(local_logits.size(-1)))
    if native_filtering_active and not can_use_custom_tiled_forward_kernel:
        raise RuntimeError("cuda_tiled native threshold/topk requires the custom tiled forward kernel")
    if native_filtering_active and needs_grad and not can_use_custom_tiled_backward_kernel:
        raise RuntimeError("cuda_tiled native threshold/topk requires the custom tiled backward kernel when gradients are enabled")
    if not can_use_custom_tiled_forward_kernel and not hasattr(ext, "forward_tiled_logits"):
        raise RuntimeError("cuda_tiled requires forward_tiled_logits in the CUDA extension")
    if needs_grad and not can_use_custom_tiled_backward_kernel and not hasattr(ext, "backward_tiled_probs"):
        raise RuntimeError("cuda_tiled requires backward_tiled_probs in the CUDA extension when gradients are enabled")
    workspace_dict = workspace
    if (
        workspace_dict is None
        and (can_use_custom_tiled_forward_kernel or can_use_custom_tiled_backward_kernel)
    ):
        workspace_mode = "tiled_backward" if needs_grad and can_use_custom_tiled_backward_kernel else "tiled_forward"
        if runtime_config is not None and runtime_config.backend_policy is not None:
            if needs_grad and can_use_custom_tiled_backward_kernel:
                workspace_mode = str(runtime_config.backend_policy.workspace_mode_backward)
            else:
                workspace_mode = str(runtime_config.backend_policy.workspace_mode)
        workspace_dict = create_structured_scan_workspace(
            mode=workspace_mode,
            device=local_logits.device,
            num_states=int(local_logits.size(-1)),
            transition_rank=int(transition_source_probs.size(-1)),
            batch_size=int(local_logits.size(0)),
            seq_len=int(local_logits.size(1)),
            chunk_size=int(chunk_size),
            tile_size=int(tile_size),
            split_size=int(split_size),
        )
    if needs_grad:
        beliefs, final_belief = _CausalMachineTiledScanCudaFn.apply(
            local_logits,
            transition_source_probs,
            transition_dest_probs,
            transition_context,
            initial_log_belief,
            transition_gate.reshape(()),
            transition_stay_probs,
            packed_source_q,
            packed_source_scales,
            packed_dest_q,
            packed_dest_scales,
            int(packed_kind),
            bool(use_packed_tables),
            seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
            int(chunk_size),
            int(tile_size),
            int(split_size),
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
            bool(can_use_custom_tiled_forward_kernel),
            bool(can_use_custom_tiled_backward_kernel),
            workspace_dict,
        )
    elif (
        use_packed_tables
        and int(packed_kind) == _PACKED_TRANSITION_INT8
        and workspace_dict is not None
        and hasattr(ext, "forward_tiled_quantized_kernel_bound_workspace")
    ):
        beliefs, final_belief = ext.forward_tiled_quantized_kernel_bound_workspace(
            local_logits.contiguous(),
            packed_source_q.contiguous(),
            packed_source_scales.contiguous(),
            packed_dest_q.contiguous(),
            packed_dest_scales.contiguous(),
            transition_context.contiguous(),
            initial_log_belief.contiguous().float(),
            transition_gate.reshape(()).float(),
            transition_stay_probs.contiguous().float(),
            seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
            int(chunk_size),
            int(tile_size),
            int(split_size),
            workspace_dict,
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
        )
    elif (
        use_packed_tables
        and int(packed_kind) == _PACKED_TRANSITION_INT8
        and workspace_dict is not None
        and hasattr(ext, "forward_tiled_quantized_kernel_workspace")
    ):
        beliefs, final_belief = ext.forward_tiled_quantized_kernel_workspace(
            local_logits.contiguous(),
            packed_source_q.contiguous(),
            packed_source_scales.contiguous(),
            packed_dest_q.contiguous(),
            packed_dest_scales.contiguous(),
            transition_context.contiguous(),
            initial_log_belief.contiguous().float(),
            transition_gate.reshape(()).float(),
            transition_stay_probs.contiguous().float(),
            seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
            int(chunk_size),
            int(tile_size),
            int(split_size),
            workspace_dict["work_queue_counter"],
            workspace_dict["filtered_value_cache"],
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
        )
    elif use_packed_tables and int(packed_kind) == _PACKED_TRANSITION_INT8:
        beliefs, final_belief = ext.forward_tiled_quantized_kernel(
            local_logits.contiguous(),
            packed_source_q.contiguous(),
            packed_source_scales.contiguous(),
            packed_dest_q.contiguous(),
            packed_dest_scales.contiguous(),
            transition_context.contiguous(),
            initial_log_belief.contiguous().float(),
            transition_gate.reshape(()).float(),
            transition_stay_probs.contiguous().float(),
            seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
            int(chunk_size),
            int(tile_size),
            int(split_size),
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
        )
    elif (
        use_packed_tables
        and int(packed_kind) in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}
        and workspace_dict is not None
        and hasattr(ext, "forward_tiled_fp8_kernel_bound_workspace")
    ):
        beliefs, final_belief = ext.forward_tiled_fp8_kernel_bound_workspace(
            local_logits.contiguous(),
            packed_source_q.contiguous(),
            packed_source_scales.contiguous(),
            packed_dest_q.contiguous(),
            packed_dest_scales.contiguous(),
            transition_context.contiguous(),
            initial_log_belief.contiguous().float(),
            transition_gate.reshape(()).float(),
            transition_stay_probs.contiguous().float(),
            0 if int(packed_kind) == _PACKED_TRANSITION_FP8_E4M3 else 1,
            seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
            int(chunk_size),
            int(tile_size),
            int(split_size),
            workspace_dict,
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
        )
    elif (
        use_packed_tables
        and int(packed_kind) in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}
        and workspace_dict is not None
        and hasattr(ext, "forward_tiled_fp8_kernel_workspace")
    ):
        beliefs, final_belief = ext.forward_tiled_fp8_kernel_workspace(
            local_logits.contiguous(),
            packed_source_q.contiguous(),
            packed_source_scales.contiguous(),
            packed_dest_q.contiguous(),
            packed_dest_scales.contiguous(),
            transition_context.contiguous(),
            initial_log_belief.contiguous().float(),
            transition_gate.reshape(()).float(),
            transition_stay_probs.contiguous().float(),
            0 if int(packed_kind) == _PACKED_TRANSITION_FP8_E4M3 else 1,
            seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
            int(chunk_size),
            int(tile_size),
            int(split_size),
            workspace_dict["work_queue_counter"],
            workspace_dict["filtered_value_cache"],
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
        )
    elif use_packed_tables and int(packed_kind) in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}:
        beliefs, final_belief = ext.forward_tiled_fp8_kernel(
            local_logits.contiguous(),
            packed_source_q.contiguous(),
            packed_source_scales.contiguous(),
            packed_dest_q.contiguous(),
            packed_dest_scales.contiguous(),
            transition_context.contiguous(),
            initial_log_belief.contiguous().float(),
            transition_gate.reshape(()).float(),
            transition_stay_probs.contiguous().float(),
            0 if int(packed_kind) == _PACKED_TRANSITION_FP8_E4M3 else 1,
            seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
            int(chunk_size),
            int(tile_size),
            int(split_size),
            float(score_clamp_min),
            float(score_clamp_max),
            float(score_threshold),
            int(score_topk),
        )
    elif can_use_custom_tiled_forward_kernel:
        if workspace_dict is not None and hasattr(ext, "forward_tiled_logits_kernel_bound_workspace"):
            beliefs, final_belief = ext.forward_tiled_logits_kernel_bound_workspace(
                local_logits.contiguous(),
                transition_source_probs.contiguous().float(),
                transition_dest_probs.contiguous().float(),
                transition_context.contiguous(),
                initial_log_belief.contiguous().float(),
                transition_gate.reshape(()).float(),
                transition_stay_probs.contiguous().float(),
                seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                workspace_dict,
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        elif workspace_dict is not None and hasattr(ext, "forward_tiled_logits_kernel_workspace"):
            beliefs, final_belief = ext.forward_tiled_logits_kernel_workspace(
                local_logits.contiguous(),
                transition_source_probs.contiguous().float(),
                transition_dest_probs.contiguous().float(),
                transition_context.contiguous(),
                initial_log_belief.contiguous().float(),
                transition_gate.reshape(()).float(),
                transition_stay_probs.contiguous().float(),
                seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                workspace_dict["work_queue_counter"],
                workspace_dict["filtered_value_cache"],
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
        else:
            beliefs, final_belief = ext.forward_tiled_logits_kernel(
                local_logits.contiguous(),
                transition_source_probs.contiguous().float(),
                transition_dest_probs.contiguous().float(),
                transition_context.contiguous(),
                initial_log_belief.contiguous().float(),
                transition_gate.reshape(()).float(),
                transition_stay_probs.contiguous().float(),
                seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
                int(chunk_size),
                int(tile_size),
                int(split_size),
                float(score_clamp_min),
                float(score_clamp_max),
                float(score_threshold),
                int(score_topk),
            )
    else:
        beliefs, final_belief = ext.forward_tiled_logits(
            local_logits.contiguous(),
            transition_source_probs.contiguous().float(),
            transition_dest_probs.contiguous().float(),
            transition_context.contiguous(),
            initial_log_belief.contiguous().float(),
            transition_gate.reshape(()).float(),
            transition_stay_probs.contiguous().float(),
            seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
            int(chunk_size),
            int(tile_size),
            int(split_size),
            float(score_clamp_min),
            float(score_clamp_max),
        )
    return beliefs, final_belief


@dataclass(frozen=True)
class StructuredSparseTransitionTables:
    blocks: Tensor
    row_ptr: Tensor
    col_idx: Tensor
    dst_idx: Tensor
    src_row_ptr: Tensor
    src_nz_idx: Tensor
    row_sums: Tensor
    block_mask: Tensor
    block_size: int
    density: float
    grouped_src_row_ptr: Tensor | None = None
    grouped_src_block_idx: Tensor | None = None
    grouped_src_group_ids: Tensor | None = None
    grouped_src_group_count: int = 0


def describe_structured_sparse_transition_tables(
    sparse_transition_tables: StructuredSparseTransitionTables,
) -> dict[str, int | float | bool]:
    nnz_blocks = int(sparse_transition_tables.col_idx.numel())
    grouped_src_group_count = int(sparse_transition_tables.grouped_src_group_count)
    avg_blocks_per_group = (
        float(nnz_blocks) / float(max(grouped_src_group_count, 1))
        if grouped_src_group_count > 0
        else 0.0
    )
    return {
        "nnz_blocks": nnz_blocks,
        "block_size": int(sparse_transition_tables.block_size),
        "density": float(sparse_transition_tables.density),
        "has_grouped_backward_metadata": sparse_transition_tables.grouped_src_row_ptr is not None,
        "grouped_src_group_count": grouped_src_group_count,
        "avg_blocks_per_group": avg_blocks_per_group,
    }


def _default_structured_sparse_block_size(num_states: int) -> int:
    states = max(int(num_states), 1)
    if states <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        return 32
    if states <= 256:
        return 64
    return _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES


def _structured_sparse_cache_tensor_key(tensor: Tensor | None) -> tuple[int, int, tuple[int, ...]] | None:
    if tensor is None:
        return None
    return (
        int(tensor.data_ptr()),
        int(getattr(tensor, "_version", -1)),
        tuple(int(dim) for dim in tensor.shape),
    )


def _build_grouped_sparse_backward_metadata_cpu(
    col_idx: Tensor,
    src_nz_idx: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if int(src_nz_idx.numel()) == 0:
        empty = torch.empty((0,), device=col_idx.device, dtype=torch.int32)
        return torch.zeros((1,), device=col_idx.device, dtype=torch.int32), empty, empty
    ordered_src_blocks = col_idx.index_select(0, src_nz_idx.to(torch.int64)).to(torch.int32).contiguous()
    starts = torch.ones_like(ordered_src_blocks, dtype=torch.int32)
    if int(ordered_src_blocks.numel()) > 1:
        starts[1:] = ordered_src_blocks[1:].ne(ordered_src_blocks[:-1]).to(torch.int32)
    group_ids = starts.cumsum(0).sub_(1).to(torch.int32).contiguous()
    num_groups = int(group_ids[-1].item()) + 1
    counts = torch.bincount(group_ids.to(torch.int64), minlength=num_groups).to(torch.int32)
    row_ptr = torch.zeros((num_groups + 1,), device=col_idx.device, dtype=torch.int32)
    if num_groups > 0:
        row_ptr[1:] = counts.cumsum(0).to(torch.int32)
    group_start_idx = torch.nonzero(starts, as_tuple=False).flatten().to(torch.int64)
    grouped_src_block_idx = ordered_src_blocks.index_select(0, group_start_idx).to(torch.int32).contiguous()
    return row_ptr.contiguous(), grouped_src_block_idx, group_ids


def _build_grouped_sparse_backward_metadata_python(
    col_idx: Tensor,
    src_nz_idx: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    return _build_grouped_sparse_backward_metadata_cpu(col_idx, src_nz_idx)


def get_or_update_scan_transition_sparse_blocks(
    cache: dict[str, object],
    source_logits: Tensor,
    dest_logits: Tensor,
    device: torch.device,
    runtime_config: StructuredScanRuntimeConfig | None,
    packed_transition_tables: tuple[int, Tensor, Tensor, Tensor, Tensor] | None = None,
) -> StructuredSparseTransitionTables | None:
    if runtime_config is None or not _structured_runtime_supports_sparse_cuda(runtime_config):
        return None
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=device,
        dtype=torch.float32,
        backend_family="sparse_metadata",
        batch_size=1,
        seq_len=1,
        num_states=max(int(source_logits.size(0)), 1),
    )
    num_states = int(source_logits.size(0))
    transition_rank = int(source_logits.size(1))
    if num_states <= 0 or transition_rank <= 0:
        return None
    block_size = max(1, min(
        int(runtime_config.block_size or _default_structured_sparse_block_size(num_states)),
        num_states,
    ))
    device_index = int(device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else -1))
    cache_key = (
        device_index,
        int(num_states),
        int(transition_rank),
        int(getattr(source_logits, "_version", -1)),
        int(getattr(dest_logits, "_version", -1)),
        int(block_size),
        int(packed_transition_tables[0]) if packed_transition_tables is not None else -1,
        int(runtime_config.local_transition_window) if runtime_config.local_transition_window is not None else -1,
        _structured_sparse_cache_tensor_key(runtime_config.transition_mask),
        _structured_sparse_cache_tensor_key(runtime_config.block_mask),
        id(runtime_config.transition_mask_mod) if runtime_config.transition_mask_mod is not None else 0,
    )
    cached_key = cache.get("key")
    cached_tables = cache.get("tables")
    if (
        runtime_config.transition_mask_mod is None
        and cached_key == cache_key
        and isinstance(cached_tables, StructuredSparseTransitionTables)
    ):
        return cached_tables

    with torch.no_grad():
        if runtime_config.grouped_launch_pack is not None:
            runtime_config.grouped_launch_pack.record_row_sum_prep()
        num_state_blocks = math.ceil(num_states / block_size)
        padded_states = num_state_blocks * block_size
        source_probs = F.softmax(source_logits.detach().to(device=device, dtype=torch.float32), dim=-1).contiguous()
        dest_probs = F.softmax(dest_logits.detach().to(device=device, dtype=torch.float32), dim=-1).contiguous()
        if device.type == "cuda":
            ext = load_causal_machine_scan_cuda()
            transition_mask: Tensor | None = (
                runtime_config.transition_mask.to(device=device, dtype=torch.bool).contiguous()
                if runtime_config is not None and runtime_config.transition_mask is not None
                else None
            )
            mod_transition_mask = _evaluate_structured_transition_mask_mod(
                num_states,
                device=device,
                runtime_config=runtime_config,
            )
            if mod_transition_mask is not None:
                transition_mask = (
                    mod_transition_mask
                    if transition_mask is None
                    else (transition_mask & mod_transition_mask).contiguous()
                )
            transition_mask_tensor = (
                transition_mask
                if transition_mask is not None
                else torch.empty((0, 0), device=device, dtype=torch.bool)
            )
            runtime_block_mask = (
                runtime_config.block_mask.to(device=device, dtype=torch.bool).contiguous()
                if runtime_config is not None and runtime_config.block_mask is not None
                else torch.empty((0, 0), device=device, dtype=torch.bool)
            )
            row_ptr, col_idx, dst_idx, src_row_ptr, src_nz_idx, block_mask = ext.build_sparse_metadata_from_runtime(
                int(num_states),
                int(padded_states),
                int(block_size),
                int(runtime_config.local_transition_window) if runtime_config is not None and runtime_config.local_transition_window is not None else -1,
                transition_mask_tensor,
                runtime_block_mask,
            )
            if int(col_idx.numel()) > 0:
                if packed_transition_tables is not None:
                    packed_kind, packed_source_q, packed_source_scales, packed_dest_q, packed_dest_scales = packed_transition_tables
                    if packed_kind == _PACKED_TRANSITION_INT8:
                        transition_blocks, row_sums = ext.materialize_sparse_blocks_int8(
                            packed_source_q.contiguous(),
                            packed_source_scales.contiguous(),
                            packed_dest_q.contiguous(),
                            packed_dest_scales.contiguous(),
                            col_idx,
                            dst_idx,
                            block_mask,
                            int(padded_states),
                            int(block_size),
                        )
                    elif packed_kind in {_PACKED_TRANSITION_FP8_E4M3, _PACKED_TRANSITION_FP8_E5M2}:
                        transition_blocks, row_sums = ext.materialize_sparse_blocks_fp8(
                            packed_source_q.contiguous(),
                            packed_source_scales.contiguous(),
                            packed_dest_q.contiguous(),
                            packed_dest_scales.contiguous(),
                            col_idx,
                            dst_idx,
                            block_mask,
                            0 if packed_kind == _PACKED_TRANSITION_FP8_E4M3 else 1,
                            int(padded_states),
                            int(block_size),
                        )
                    else:
                        raise ValueError(f"unsupported packed transition kind: {packed_kind}")
                else:
                    transition_blocks, row_sums = ext.materialize_sparse_blocks(
                        source_probs.contiguous(),
                        dest_probs.contiguous(),
                        col_idx,
                        dst_idx,
                        block_mask,
                        int(padded_states),
                        int(block_size),
                    )
            else:
                transition_blocks = torch.empty((0, block_size, block_size), device=device, dtype=torch.float32)
                row_sums = torch.zeros((padded_states,), device=device, dtype=torch.float32)
        else:
            mask = _build_structured_transition_mask(
                num_states,
                device=device,
                runtime_config=runtime_config,
            )
            if mask is None:
                return None
            if padded_states != num_states:
                source_pad = torch.zeros((padded_states, transition_rank), device=device, dtype=torch.float32)
                source_pad[:num_states, :] = source_probs
                source_probs = source_pad
                dest_pad = torch.zeros((transition_rank, padded_states), device=device, dtype=torch.float32)
                dest_pad[:, :num_states] = dest_probs
                dest_probs = dest_pad
                mask_pad = torch.zeros((padded_states, padded_states), device=device, dtype=torch.bool)
                mask_pad[:num_states, :num_states] = mask
                mask = mask_pad
            block_active = (
                mask.view(num_state_blocks, block_size, num_state_blocks, block_size)
                .permute(0, 2, 1, 3)
                .any(dim=3)
                .any(dim=2)
            )
            row_ptr_list: list[int] = [0]
            col_idx_list: list[int] = []
            dst_idx_list: list[int] = []
            for dst_block in range(num_state_blocks):
                active_src = torch.nonzero(block_active[:, dst_block], as_tuple=False).flatten()
                row_ptr_list.append(row_ptr_list[-1] + int(active_src.numel()))
                for src_block in active_src.tolist():
                    col_idx_list.append(int(src_block))
                    dst_idx_list.append(int(dst_block))
            row_ptr = torch.tensor(row_ptr_list, device=device, dtype=torch.int32).contiguous()
            col_idx = torch.tensor(col_idx_list, device=device, dtype=torch.int32).contiguous()
            dst_idx = torch.tensor(dst_idx_list, device=device, dtype=torch.int32).contiguous()
            if col_idx_list:
                block_mask = torch.stack(
                    [
                        mask[
                            src_block * block_size : (src_block + 1) * block_size,
                            dst_block * block_size : (dst_block + 1) * block_size,
                        ]
                        for src_block, dst_block in zip(col_idx_list, dst_idx_list)
                    ],
                    dim=0,
                ).to(dtype=torch.float32).contiguous()
                raw_blocks: list[Tensor] = []
                row_sums = torch.zeros((padded_states,), device=device, dtype=torch.float32)
                for src_block, dst_block in zip(col_idx_list, dst_idx_list):
                    src_base = src_block * block_size
                    dst_base = dst_block * block_size
                    raw_block = torch.matmul(
                        source_probs[src_base : src_base + block_size, :],
                        dest_probs[:, dst_base : dst_base + block_size],
                    )
                    raw_block = raw_block * mask[
                        src_base : src_base + block_size,
                        dst_base : dst_base + block_size,
                    ].to(torch.float32)
                    row_sums[src_base : src_base + block_size] += raw_block.sum(dim=1)
                    raw_blocks.append(raw_block)
                transition_blocks = torch.stack(
                    [
                        raw_block / row_sums[src_block * block_size : (src_block + 1) * block_size].clamp_min(1.0e-20).unsqueeze(1)
                        for raw_block, src_block in zip(raw_blocks, col_idx_list)
                    ],
                    dim=0,
                ).contiguous()
            else:
                transition_blocks = torch.empty((0, block_size, block_size), device=device, dtype=torch.float32)
                row_sums = torch.zeros((padded_states,), device=device, dtype=torch.float32)
                block_mask = torch.empty((0, block_size, block_size), device=device, dtype=torch.float32)
            src_row_ptr_list: list[int] = [0]
            src_nz_idx_list: list[int] = []
            for src_block in range(num_state_blocks):
                nz_indices = [idx for idx, block_idx in enumerate(col_idx_list) if block_idx == src_block]
                src_nz_idx_list.extend(nz_indices)
                src_row_ptr_list.append(len(src_nz_idx_list))
            src_row_ptr = torch.tensor(src_row_ptr_list, device=device, dtype=torch.int32).contiguous()
            src_nz_idx = torch.tensor(src_nz_idx_list, device=device, dtype=torch.int32).contiguous()
        density = float(int(col_idx.numel())) / float(max(num_state_blocks * num_state_blocks, 1))
        if device.type == "cuda":
            ext = load_causal_machine_scan_cuda()
            if hasattr(ext, "build_grouped_sparse_backward_metadata"):
                grouped_src_row_ptr, grouped_src_block_idx, grouped_src_group_ids = ext.build_grouped_sparse_backward_metadata(
                    col_idx.contiguous(),
                    src_nz_idx.contiguous(),
                )
            else:
                grouped_src_row_ptr, grouped_src_block_idx, grouped_src_group_ids = _build_grouped_sparse_backward_metadata_python(
                    col_idx.contiguous(),
                    src_nz_idx.contiguous(),
                )
        else:
            grouped_src_row_ptr, grouped_src_block_idx, grouped_src_group_ids = _build_grouped_sparse_backward_metadata_python(
                col_idx.contiguous(),
                src_nz_idx.contiguous(),
            )
        grouped_src_group_count = int(grouped_src_block_idx.numel())

    tables = StructuredSparseTransitionTables(
        blocks=transition_blocks,
        row_ptr=row_ptr,
        col_idx=col_idx,
        dst_idx=dst_idx,
        src_row_ptr=src_row_ptr,
        src_nz_idx=src_nz_idx,
        row_sums=row_sums.contiguous(),
        block_mask=block_mask,
        block_size=int(block_size),
        density=float(density),
        grouped_src_row_ptr=grouped_src_row_ptr.contiguous(),
        grouped_src_block_idx=grouped_src_block_idx.contiguous(),
        grouped_src_group_ids=grouped_src_group_ids.contiguous(),
        grouped_src_group_count=int(grouped_src_group_count),
    )
    if runtime_config.transition_mask_mod is None:
        cache.clear()
        cache["key"] = cache_key
        cache["tables"] = tables
    return tables


def causal_machine_scan_sparse_cuda(
    local_logits: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_gate: Tensor,
    transition_stay_probs: Tensor,
    sparse_transition_tables: StructuredSparseTransitionTables,
    *,
    runtime_config: StructuredScanRuntimeConfig | None,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor]:
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=local_logits.device,
        dtype=local_logits.dtype,
        backend_family="sparse_cuda",
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        num_states=int(local_logits.size(2)),
    )
    seq_lens = _canonicalize_structured_seq_lens(
        runtime_config.seq_lens if runtime_config is not None else None,
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        device=local_logits.device,
    )
    ext = load_causal_machine_scan_cuda()
    empty_seq_lens = torch.empty((0,), device=local_logits.device, dtype=torch.int64)
    beliefs, final_belief = ext.forward_sparse_logits(
        local_logits.contiguous(),
        sparse_transition_tables.blocks.contiguous(),
        sparse_transition_tables.row_ptr.contiguous(),
        sparse_transition_tables.col_idx.contiguous(),
        transition_context.contiguous(),
        initial_log_belief.contiguous(),
        transition_gate.reshape(()).contiguous().float(),
        transition_stay_probs.contiguous().float(),
        seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
        int(sparse_transition_tables.block_size),
        int(chunk_size),
    )
    return beliefs, final_belief


def causal_machine_scan_sparse_cuda_autograd(
    local_logits: Tensor,
    transition_source_logits: Tensor,
    transition_dest_logits: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_gate: Tensor,
    transition_stay_probs: Tensor,
    sparse_transition_tables: StructuredSparseTransitionTables,
    *,
    runtime_config: StructuredScanRuntimeConfig | None,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor]:
    seq_lens = _canonicalize_structured_seq_lens(
        runtime_config.seq_lens if runtime_config is not None else None,
        batch_size=int(local_logits.size(0)),
        seq_len=int(local_logits.size(1)),
        device=local_logits.device,
    )
    empty_seq_lens = torch.empty((0,), device=local_logits.device, dtype=torch.int64)
    return _CausalMachineSparseScanCudaFn.apply(
        local_logits,
        transition_source_logits,
        transition_dest_logits,
        transition_context,
        initial_log_belief,
        transition_gate,
        transition_stay_probs,
        sparse_transition_tables.row_ptr,
        sparse_transition_tables.col_idx,
        sparse_transition_tables.dst_idx,
        sparse_transition_tables.src_row_ptr,
        sparse_transition_tables.src_nz_idx,
        (
            sparse_transition_tables.grouped_src_row_ptr
            if sparse_transition_tables.grouped_src_row_ptr is not None
            else torch.empty((0,), device=local_logits.device, dtype=torch.int32)
        ),
        (
            sparse_transition_tables.grouped_src_block_idx
            if sparse_transition_tables.grouped_src_block_idx is not None
            else torch.empty((0,), device=local_logits.device, dtype=torch.int32)
        ),
        sparse_transition_tables.block_mask,
        seq_lens.contiguous() if seq_lens is not None else empty_seq_lens,
        int(sparse_transition_tables.block_size),
        int(chunk_size),
    )


def supports_structured_scan_cuda_rank(transition_rank: int) -> bool:
    return supports_structured_scan_cuda_config(_MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES, transition_rank)


def quantize_scan_transition_table_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float().contiguous()
    if t32.ndim != 2:
        raise ValueError(f"expected 2D transition table, got shape={tuple(t32.shape)}")
    clip_abs = t32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    q = torch.clamp(torch.round(t32 / (clip_abs / 127.0)), -127.0, 127.0).to(torch.int8).contiguous()
    scales = (clip_abs / 127.0).reshape(-1).to(dtype=torch.float32).contiguous()
    return q, scales


def get_or_update_structured_reduced_transition_cache(
    cache: dict[str, object],
    source_logits: Tensor,
    dest_logits: Tensor,
    device: torch.device,
    runtime_config: StructuredScanRuntimeConfig | None,
    *,
    num_states: int,
    tile_size: int,
    split_size: int,
) -> StructuredReducedTransitionCache | None:
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=device,
        dtype=torch.float32,
        backend_family="reduced_transition",
        batch_size=1,
        seq_len=1,
        num_states=max(int(num_states), 1),
    )
    source_version = int(getattr(source_logits, "_version", -1))
    dest_version = int(getattr(dest_logits, "_version", -1))
    cache_key = (
        int(device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else -1)),
        source_version,
        dest_version,
        int(num_states),
        int(tile_size),
        int(split_size),
        _structured_sparse_cache_tensor_key(runtime_config.transition_mask),
        _structured_sparse_cache_tensor_key(runtime_config.block_mask),
        int(runtime_config.local_transition_window) if runtime_config.local_transition_window is not None else -1,
        id(runtime_config.transition_mask_mod) if runtime_config.transition_mask_mod is not None else 0,
    )
    cached_key = cache.get("key")
    cached_value = cache.get("reduced")
    if cached_key == cache_key and isinstance(cached_value, StructuredReducedTransitionCache):
        runtime_config.reduced_transition_cache = cached_value
        return cached_value
    with torch.no_grad():
        source_probs = F.softmax(source_logits.detach().to(device=device, dtype=torch.float32), dim=-1).contiguous()
        dest_probs = F.softmax(dest_logits.detach().to(device=device, dtype=torch.float32), dim=-1).contiguous()
        dense_transition_matrix = None
        row_sums = None
        if int(num_states) <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES or _structured_runtime_uses_transition_masking(runtime_config):
            dense_transition_matrix = _build_structured_transition_matrix(
                source_probs,
                dest_probs,
                runtime_config=runtime_config,
            )
            if dense_transition_matrix is not None:
                row_sums = dense_transition_matrix.sum(dim=-1).contiguous()
        split_size = max(int(split_size), 1)
        source_splits = tuple(
            source_probs[:, start : min(start + split_size, int(source_probs.size(1)))].contiguous()
            for start in range(0, int(source_probs.size(1)), split_size)
        )
        dest_splits = tuple(
            dest_probs[start : min(start + split_size, int(dest_probs.size(0))), :].contiguous()
            for start in range(0, int(dest_probs.size(0)), split_size)
        )
        reduced = StructuredReducedTransitionCache(
            dense_transition_matrix=dense_transition_matrix,
            source_prob_splits=source_splits,
            dest_prob_splits=dest_splits,
            row_sums=row_sums,
            split_size=int(split_size),
            tile_size=int(tile_size),
        )
    cache.clear()
    cache["key"] = cache_key
    cache["reduced"] = reduced
    runtime_config.reduced_transition_cache = reduced
    return reduced


def get_or_update_scan_transition_prepack(
    cache: dict[str, object],
    source_logits: Tensor,
    dest_logits: Tensor,
    device: torch.device,
) -> tuple[int, Tensor, Tensor, Tensor, Tensor] | None:
    num_states = int(source_logits.size(0))
    transition_rank = int(source_logits.size(1))
    if (
        not USE_CAUSAL_MACHINE_SCAN_FUSED_DEQUANT
        or device.type != "cuda"
        or num_states <= 0
        or transition_rank <= 0
    ):
        return None
    device_runtime = _describe_structured_scan_device_runtime(device)
    if int(device_runtime.get("capability_major", 0)) < 8:
        return None
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    packed_kind = _structured_scan_packed_kind()
    low_precision_recipe = _default_structured_scan_low_precision_recipe()
    gradients_enabled = torch.is_grad_enabled()
    source_version = int(getattr(source_logits, "_version", -1))
    dest_version = int(getattr(dest_logits, "_version", -1))
    if (
        cache.get("device_index") == device_index
        and cache.get("packed_kind") == packed_kind
        and cache.get("source_version") == source_version
        and cache.get("dest_version") == dest_version
    ):
        packed = cache.get("packed")
        if isinstance(packed, tuple) and len(packed) == 5:
            return packed  # type: ignore[return-value]
    with torch.no_grad():
        ext = load_causal_machine_scan_cuda()
        if gradients_enabled:
            if packed_kind == _PACKED_TRANSITION_INT8 and not hasattr(ext, "unpack_int8"):
                return None
            if packed_kind == _PACKED_TRANSITION_FP8_E4M3 and not hasattr(ext, "unpack_fp8_e4m3"):
                return None
            if packed_kind == _PACKED_TRANSITION_FP8_E5M2 and not hasattr(ext, "unpack_fp8_e5m2"):
                return None
        source_probs = F.softmax(source_logits.detach().to(device=device, dtype=torch.float32), dim=-1).contiguous()
        dest_probs = F.softmax(dest_logits.detach().to(device=device, dtype=torch.float32), dim=-1).contiguous()
        low_precision_metadata = _materialize_structured_scan_low_precision_metadata(
            cache=cache,
            recipe=low_precision_recipe,
            source_probs=source_probs,
            dest_probs=dest_probs,
        )
        if packed_kind == _PACKED_TRANSITION_INT8:
            packed_source_q, packed_source_scales = ext.pack_int8(source_probs)
            packed_dest_q, packed_dest_scales = ext.pack_int8(dest_probs)
        elif packed_kind == _PACKED_TRANSITION_FP8_E4M3:
            packed_source_q, packed_source_scales = ext.pack_fp8_e4m3(source_probs)
            packed_dest_q, packed_dest_scales = ext.pack_fp8_e4m3(dest_probs)
        elif packed_kind == _PACKED_TRANSITION_FP8_E5M2:
            packed_source_q, packed_source_scales = ext.pack_fp8_e5m2(source_probs)
            packed_dest_q, packed_dest_scales = ext.pack_fp8_e5m2(dest_probs)
        else:
            raise ValueError(f"unsupported packed transition kind: {packed_kind}")
        packed = (
            packed_kind,
            packed_source_q.contiguous(),
            packed_source_scales.contiguous(),
            packed_dest_q.contiguous(),
            packed_dest_scales.contiguous(),
        )
    cache.clear()
    cache["device_index"] = device_index
    cache["packed_kind"] = packed_kind
    cache["source_version"] = source_version
    cache["dest_version"] = dest_version
    cache["packed"] = packed
    cache["low_precision_recipe"] = low_precision_recipe
    cache["low_precision_source_scale_estimate"] = float(low_precision_metadata.source_scale)
    cache["low_precision_dest_scale_estimate"] = float(low_precision_metadata.dest_scale)
    cache["low_precision_source_scale_inv"] = float(low_precision_metadata.source_scale_inv)
    cache["low_precision_dest_scale_inv"] = float(low_precision_metadata.dest_scale_inv)
    return packed


def _unpack_scan_transition_tables(
    ext,
    packed_kind: int,
    packed_source_q: Tensor,
    packed_source_scales: Tensor,
    packed_dest_q: Tensor,
    packed_dest_scales: Tensor,
) -> tuple[Tensor, Tensor]:
    if packed_kind == _PACKED_TRANSITION_INT8:
        return (
            ext.unpack_int8(packed_source_q, packed_source_scales).contiguous(),
            ext.unpack_int8(packed_dest_q, packed_dest_scales).contiguous(),
        )
    if packed_kind == _PACKED_TRANSITION_FP8_E4M3:
        return (
            ext.unpack_fp8_e4m3(packed_source_q, packed_source_scales).contiguous(),
            ext.unpack_fp8_e4m3(packed_dest_q, packed_dest_scales).contiguous(),
        )
    if packed_kind == _PACKED_TRANSITION_FP8_E5M2:
        return (
            ext.unpack_fp8_e5m2(packed_source_q, packed_source_scales).contiguous(),
            ext.unpack_fp8_e5m2(packed_dest_q, packed_dest_scales).contiguous(),
        )
    raise ValueError(f"unsupported packed transition kind: {packed_kind}")


class _CausalMachineLatentScanCudaFn(torch.autograd.Function):
    @staticmethod
    @_torch_dynamo_disable_if_available
    def forward(
        ctx,
        drive: Tensor,
        decay: Tensor,
        initial_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ext = load_causal_machine_latent_scan_cuda()
        drive_in = drive.contiguous()
        decay_in = decay.contiguous().to(dtype=drive_in.dtype)
        initial_state_in = initial_state.contiguous()
        states, prior_states, final_state = ext.forward(drive_in, decay_in, initial_state_in)
        ctx.save_for_backward(states.contiguous(), decay_in, initial_state_in)
        return states, prior_states, final_state

    @staticmethod
    @_torch_dynamo_disable_if_available
    def backward(ctx, grad_states: Tensor, grad_prior_states: Tensor, grad_final_state: Tensor):
        ext = load_causal_machine_latent_scan_cuda()
        states, decay_in, initial_state_in = ctx.saved_tensors
        grad_drive, grad_decay, grad_initial = ext.backward(
            grad_states.contiguous(),
            grad_prior_states.contiguous(),
            grad_final_state.contiguous(),
            states,
            decay_in,
            initial_state_in,
        )
        return grad_drive, grad_decay.reshape_as(decay_in).to(decay_in.dtype), grad_initial


@_torch_dynamo_disable_if_available
def causal_machine_latent_scan_cuda(
    drive: Tensor,
    decay: Tensor,
    initial_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    return _CausalMachineLatentScanCudaFn.apply(drive, decay, initial_state)


class _CausalMachineLatentPriorScanCudaFn(torch.autograd.Function):
    @staticmethod
    @_torch_dynamo_disable_if_available
    def forward(
        ctx,
        drive: Tensor,
        decay: Tensor,
        initial_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ext = load_causal_machine_latent_scan_cuda()
        drive_in = drive.contiguous()
        decay_in = decay.contiguous().to(dtype=drive_in.dtype)
        initial_state_in = initial_state.contiguous()
        prior_states, final_state = ext.forward_prior_only(drive_in, decay_in, initial_state_in)
        ctx.save_for_backward(prior_states.contiguous(), decay_in, initial_state_in)
        return prior_states, final_state

    @staticmethod
    @_torch_dynamo_disable_if_available
    def backward(ctx, grad_prior_states: Tensor, grad_final_state: Tensor):
        ext = load_causal_machine_latent_scan_cuda()
        prior_states, decay_in, initial_state_in = ctx.saved_tensors
        grad_drive, grad_decay, grad_initial = ext.backward_prior_only(
            grad_prior_states.contiguous(),
            grad_final_state.contiguous(),
            prior_states,
            decay_in,
            initial_state_in,
        )
        return grad_drive, grad_decay.reshape_as(decay_in).to(decay_in.dtype), grad_initial


@_torch_dynamo_disable_if_available
def causal_machine_latent_prior_scan_cuda(
    drive: Tensor,
    decay: Tensor,
    initial_state: Tensor,
) -> tuple[Tensor, Tensor]:
    return _CausalMachineLatentPriorScanCudaFn.apply(drive, decay, initial_state)


class _CausalMachineLatentReplaceCudaFn(torch.autograd.Function):
    @staticmethod
    @_torch_dynamo_disable_if_available
    def forward(
        ctx,
        local_logits: Tensor,
        prior_logits: Tensor,
        transition_context: Tensor,
        token_gate: Tensor,
        pred_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ext = load_causal_machine_latent_scan_cuda()
        local_logits_in = local_logits.contiguous()
        prior_logits_in = prior_logits.contiguous()
        transition_context_in = transition_context.contiguous()
        token_gate_in = token_gate.contiguous()
        pred_scale_in = pred_scale.contiguous()
        beliefs, prior_log_beliefs = ext.forward_replace(
            local_logits_in,
            prior_logits_in,
            transition_context_in,
            token_gate_in,
            pred_scale_in,
        )
        ctx.save_for_backward(
            prior_logits_in,
            transition_context_in,
            token_gate_in,
            pred_scale_in,
            beliefs.contiguous(),
            prior_log_beliefs.contiguous(),
        )
        return beliefs, prior_log_beliefs

    @staticmethod
    @_torch_dynamo_disable_if_available
    def backward(ctx, grad_beliefs: Tensor, grad_prior_log_beliefs: Tensor | None):
        ext = load_causal_machine_latent_scan_cuda()
        (
            prior_logits_in,
            transition_context_in,
            token_gate_in,
            pred_scale_in,
            beliefs,
            prior_log_beliefs,
        ) = ctx.saved_tensors
        if grad_prior_log_beliefs is None:
            grad_prior_log_beliefs = torch.zeros_like(prior_log_beliefs)
        (
            grad_local_logits,
            grad_prior_logits,
            grad_transition_context,
            grad_token_gate,
            grad_pred_scale,
        ) = ext.backward_replace(
            grad_beliefs.contiguous(),
            grad_prior_log_beliefs.contiguous(),
            prior_logits_in,
            transition_context_in,
            token_gate_in,
            pred_scale_in,
            beliefs,
            prior_log_beliefs,
        )
        return (
            grad_local_logits,
            grad_prior_logits,
            grad_transition_context,
            grad_token_gate,
            grad_pred_scale,
        )


@_torch_dynamo_disable_if_available
def causal_machine_latent_replace_cuda(
    local_logits: Tensor,
    prior_logits: Tensor,
    transition_context: Tensor,
    token_gate: Tensor,
    pred_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    return _CausalMachineLatentReplaceCudaFn.apply(
        local_logits,
        prior_logits,
        transition_context,
        token_gate,
        pred_scale,
    )


def _latent_replace_kernel_family(local_logits: Tensor) -> str:
    if not (USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN and local_logits.is_cuda):
        return "latent_replace_python"
    if int(local_logits.size(-1)) == 128:
        return "latent_replace_step_cache_128" if int(local_logits.size(1)) == 1 else "latent_replace_seq_128"
    return "latent_replace_generic"


def structured_transition_predict_log_belief(
    prev_log_belief: Tensor,
    transition_source_probs: Tensor,
    transition_dest_probs: Tensor,
    transition_stay_probs: Tensor,
) -> Tensor:
    prev_probs = prev_log_belief.float().exp()
    latent_probs = prev_probs @ transition_source_probs.float()
    mix_probs = latent_probs @ transition_dest_probs.float()
    stay_probs = transition_stay_probs.float().unsqueeze(0)
    pred_probs = stay_probs * prev_probs + (1.0 - stay_probs) * mix_probs
    return pred_probs.clamp_min(1.0e-20).log()


def _structured_running_logsumexp_update(
    running_max: Tensor | None,
    running_sum: Tensor | None,
    value_tile: Tensor,
) -> tuple[Tensor, Tensor]:
    tile_max = value_tile.max(dim=-1, keepdim=True).values
    tile_sum = torch.exp(value_tile - tile_max).sum(dim=-1, keepdim=True)
    if running_max is None or running_sum is None:
        return tile_max, tile_sum
    new_max = torch.maximum(running_max, tile_max)
    new_sum = (
        running_sum * torch.exp(running_max - new_max)
        + tile_sum * torch.exp(tile_max - new_max)
    )
    return new_max, new_sum


def structured_scan_fallback_tiled(
    local_logits: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_source_probs: Tensor,
    transition_dest_probs: Tensor,
    transition_stay_probs: Tensor,
    transition_gate: Tensor,
    *,
    chunk_size: int,
    tile_size: int,
    split_size: int | None = None,
    backend: str = "python_tiled",
    track_transition_kl: bool = False,
    runtime_config: StructuredScanRuntimeConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    batch_size, seq_len, num_states = local_logits.shape
    if seq_len == 0:
        empty_beliefs = torch.empty_like(local_logits)
        return empty_beliefs, initial_log_belief.float(), None
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=local_logits.device,
        dtype=local_logits.dtype,
        backend_family=str(backend),
        batch_size=batch_size,
        seq_len=seq_len,
        num_states=num_states,
    )
    seq_lens = _canonicalize_structured_seq_lens(
        runtime_config.seq_lens if runtime_config is not None else None,
        batch_size=batch_size,
        seq_len=seq_len,
        device=local_logits.device,
    )
    carry_plan = runtime_config.prefix_carry_plan
    assert carry_plan is not None
    carry_plan.chunk_size = max(int(chunk_size), 1)
    carry_plan.total_seq_len = int(seq_len)
    carry_plan.seq_lens = seq_lens
    belief_steps: list[Tensor] = []
    prior_steps: list[Tensor] = []
    prev_log_belief = initial_log_belief.float()
    prev_probs = prev_log_belief.exp()
    chunk_size = max(int(chunk_size), 1)
    tile_size = max(int(tile_size), 1)
    transition_rank = int(transition_source_probs.size(1))
    split_size = _default_structured_scan_split_size(num_states, transition_rank) if split_size is None else max(int(split_size), 1)
    transition_gate_f = transition_gate.to(device=local_logits.device, dtype=torch.float32)
    source_probs_f = transition_source_probs.float()
    dest_probs_f = transition_dest_probs.float()
    stay_probs_f = transition_stay_probs.float()
    use_split_combine = backend == "cuda_tiled" and local_logits.is_cuda
    reduced_cache = runtime_config.reduced_transition_cache if runtime_config is not None else None
    rank_ranges = [
        (rank_start, min(rank_start + split_size, transition_rank))
        for rank_start in range(0, transition_rank, split_size)
    ]
    source_prob_splits = (
        list(reduced_cache.source_prob_splits)
        if reduced_cache is not None and reduced_cache.split_size == split_size and len(reduced_cache.source_prob_splits) == len(rank_ranges)
        else [source_probs_f[:, rank_start:rank_end].contiguous() for rank_start, rank_end in rank_ranges]
        if use_split_combine
        else []
    )
    dest_prob_splits = (
        list(reduced_cache.dest_prob_splits)
        if reduced_cache is not None and reduced_cache.split_size == split_size and len(reduced_cache.dest_prob_splits) == len(rank_ranges)
        else [dest_probs_f[rank_start:rank_end, :].contiguous() for rank_start, rank_end in rank_ranges]
        if use_split_combine
        else []
    )
    for chunk_index, chunk_start in enumerate(range(0, seq_len, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        prev_log_belief = carry_plan.resolve_chunk_carry(chunk_index, prev_log_belief)
        prev_probs = prev_log_belief.exp()
        for t in range(chunk_start, chunk_end):
            local_logits_t = local_logits[:, t, :].float()
            transition_context_t = transition_context[:, t, :].float()
            latent_probs = None if use_split_combine else (prev_probs @ source_probs_f)
            latent_prob_splits = (
                [prev_probs @ source_prob_split for source_prob_split in source_prob_splits]
                if use_split_combine
                else []
            )
            filtered_tiles: list[Tensor] = []
            prior_tiles: list[Tensor] = []
            running_max: Tensor | None = None
            running_sum: Tensor | None = None
            for state_start in range(0, num_states, tile_size):
                state_end = min(state_start + tile_size, num_states)
                prev_probs_tile = prev_probs[:, state_start:state_end]
                stay_tile = stay_probs_f[state_start:state_end].unsqueeze(0)
                if use_split_combine:
                    mix_probs_tile: Tensor | None = None
                    for latent_probs_split, dest_prob_split in zip(latent_prob_splits, dest_prob_splits):
                        mix_probs_contrib = latent_probs_split @ dest_prob_split[:, state_start:state_end]
                        mix_probs_tile = mix_probs_contrib if mix_probs_tile is None else (mix_probs_tile + mix_probs_contrib)
                    assert mix_probs_tile is not None
                else:
                    assert latent_probs is not None
                    mix_probs_tile = latent_probs @ dest_probs_f[:, state_start:state_end]
                pred_probs_tile = stay_tile * prev_probs_tile + (1.0 - stay_tile) * mix_probs_tile
                pred_log_tile = pred_probs_tile.clamp_min(1.0e-20).log()
                prior_with_context_tile = pred_log_tile + transition_context_t[:, state_start:state_end]
                filtered_tile = local_logits_t[:, state_start:state_end] + transition_gate_f * prior_with_context_tile
                filtered_tiles.append(filtered_tile)
                if track_transition_kl and _structured_scan_save_all(runtime_config):
                    prior_tiles.append(prior_with_context_tile)
                running_max, running_sum = _structured_running_logsumexp_update(
                    running_max,
                    running_sum,
                    filtered_tile,
                )
            assert running_max is not None and running_sum is not None
            log_norm = running_max + running_sum.clamp_min(1.0e-20).log()
            next_log_belief = torch.cat([tile - log_norm for tile in filtered_tiles], dim=-1)
            prior_with_context = torch.cat(prior_tiles, dim=-1) if prior_tiles else None
            next_prev_log_belief = next_log_belief
            if seq_lens is not None:
                active = (seq_lens > t).unsqueeze(1)
                if prior_with_context is not None:
                    prior_with_context = torch.where(active, prior_with_context, prev_log_belief)
                next_log_belief = torch.where(active, next_log_belief, prev_log_belief)
                next_prev_log_belief = next_log_belief
            if track_transition_kl and _structured_scan_save_all(runtime_config) and prior_with_context is not None:
                prior_steps.append(prior_with_context.to(dtype=local_logits.dtype))
            belief_steps.append(next_log_belief.to(dtype=local_logits.dtype))
            prev_log_belief = next_prev_log_belief
            prev_probs = prev_log_belief.exp()
        carry_plan.record_chunk_summary(chunk_index, chunk_start, chunk_end, prev_log_belief)
        if runtime_config.grouped_launch_pack is not None:
            runtime_config.grouped_launch_pack.record_grouped_small_scan()
    prior_log_beliefs = torch.stack(prior_steps, dim=1) if prior_steps else None
    return torch.stack(belief_steps, dim=1), prev_log_belief, prior_log_beliefs


def structured_scan_fallback(
    local_logits: Tensor,
    transition_context: Tensor,
    initial_log_belief: Tensor,
    transition_source_probs: Tensor,
    transition_dest_probs: Tensor,
    transition_stay_probs: Tensor,
    transition_gate: Tensor,
    *,
    chunk_size: int,
    tile_size: int = 128,
    split_size: int | None = None,
    backend: str = "python",
    track_transition_kl: bool = False,
    runtime_config: StructuredScanRuntimeConfig | None = None,
    packed_transition_tables: tuple[int, Tensor, Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    batch_size, seq_len, num_states = local_logits.shape
    if seq_len == 0:
        empty_beliefs = torch.empty_like(local_logits)
        return empty_beliefs, initial_log_belief.float(), None
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=local_logits.device,
        dtype=local_logits.dtype,
        backend_family=str(backend),
        batch_size=batch_size,
        seq_len=seq_len,
        num_states=num_states,
    )
    seq_lens = _canonicalize_structured_seq_lens(
        runtime_config.seq_lens if runtime_config is not None else None,
        batch_size=batch_size,
        seq_len=seq_len,
        device=local_logits.device,
    )
    supports_tiled_backend = _structured_runtime_supports_tiled_backend(runtime_config)
    use_masked_cuda_tiled_backend = (
        backend == "cuda_tiled"
        and USE_CAUSAL_MACHINE_CUDA_SCAN
        and local_logits.is_cuda
        and (runtime_config is None or runtime_config.allow_cuda)
        and not track_transition_kl
        and runtime_config is not None
        and _structured_runtime_uses_transition_masking(runtime_config)
        and supports_tiled_backend
    )
    if use_masked_cuda_tiled_backend:
        beliefs, final_log_belief = causal_machine_scan_tiled_cuda(
            local_logits,
            transition_source_probs,
            transition_dest_probs,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            runtime_config=runtime_config,
            chunk_size=chunk_size,
            tile_size=tile_size,
            split_size=_default_structured_scan_split_size(num_states, int(transition_source_probs.size(1)))
            if split_size is None
            else int(split_size),
            packed_transition_tables=packed_transition_tables,
        )
        return beliefs, final_log_belief, None
    reduced_cache = runtime_config.reduced_transition_cache if runtime_config is not None else None
    transition_matrix = (
        reduced_cache.dense_transition_matrix
        if reduced_cache is not None and reduced_cache.dense_transition_matrix is not None
        else _build_structured_transition_matrix(
            transition_source_probs,
            transition_dest_probs,
            runtime_config=runtime_config,
        )
    )
    use_cuda_tiled_backend = (
        backend == "cuda_tiled"
        and USE_CAUSAL_MACHINE_CUDA_SCAN
        and local_logits.is_cuda
        and (runtime_config is None or runtime_config.allow_cuda)
        and not track_transition_kl
        and transition_matrix is None
        and supports_tiled_backend
    )
    if use_cuda_tiled_backend:
        beliefs, final_log_belief = causal_machine_scan_tiled_cuda(
            local_logits,
            transition_source_probs,
            transition_dest_probs,
            transition_context,
            initial_log_belief,
            transition_gate,
            transition_stay_probs,
            runtime_config=runtime_config,
            chunk_size=chunk_size,
            tile_size=tile_size,
            split_size=_default_structured_scan_split_size(num_states, int(transition_source_probs.size(1)))
            if split_size is None
            else int(split_size),
            packed_transition_tables=packed_transition_tables,
        )
        return beliefs, final_log_belief, None
    use_tiled_backend = (
        backend in {"python_tiled", "cuda_tiled"}
        and transition_matrix is None
        and supports_tiled_backend
    )
    if use_tiled_backend:
        return structured_scan_fallback_tiled(
            local_logits,
            transition_context,
            initial_log_belief,
            transition_source_probs,
            transition_dest_probs,
            transition_stay_probs,
            transition_gate,
            chunk_size=chunk_size,
            tile_size=tile_size,
            split_size=split_size,
            backend=backend,
            track_transition_kl=track_transition_kl,
            runtime_config=runtime_config,
        )
    belief_steps: list[Tensor] = []
    prior_steps: list[Tensor] = []
    prev_log_belief = initial_log_belief.float()
    chunk_size = max(int(chunk_size), 1)
    transition_gate_f = transition_gate.to(device=local_logits.device, dtype=torch.float32)
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        for t in range(chunk_start, chunk_end):
            if transition_matrix is not None:
                pred_log_belief = structured_transition_predict_log_belief_dense(
                    prev_log_belief,
                    transition_matrix,
                    transition_stay_probs,
                )
            else:
                pred_log_belief = structured_transition_predict_log_belief(
                    prev_log_belief,
                    transition_source_probs,
                    transition_dest_probs,
                    transition_stay_probs,
                )
            pred_log_belief = _apply_structured_score_mod(
                pred_log_belief,
                time_idx=t,
                runtime_config=runtime_config,
            )
            prior_with_context = pred_log_belief + transition_context[:, t, :].float()
            filtered_logits = local_logits[:, t, :].float() + transition_gate_f * prior_with_context
            next_log_belief = F.log_softmax(filtered_logits, dim=-1)
            if seq_lens is not None:
                active = (seq_lens > t).unsqueeze(1)
                prior_with_context = torch.where(active, prior_with_context, prev_log_belief)
                next_log_belief = torch.where(active, next_log_belief, prev_log_belief)
            if track_transition_kl and _structured_scan_save_all(runtime_config):
                prior_steps.append(prior_with_context.to(dtype=local_logits.dtype))
            belief_steps.append(next_log_belief.to(dtype=local_logits.dtype))
            prev_log_belief = next_log_belief
        if runtime_config.grouped_launch_pack is not None:
            runtime_config.grouped_launch_pack.record_grouped_small_scan()
    prior_log_beliefs = torch.stack(prior_steps, dim=1) if prior_steps else None
    return torch.stack(belief_steps, dim=1), prev_log_belief, prior_log_beliefs


def inverse_softplus_scalar(value: float) -> float:
    value = max(float(value), 1e-6)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def inverse_sigmoid_scalar(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def _structured_scan_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _structured_scan_shape_bucket(*sizes: int, quantum: int = 32) -> tuple[int, ...]:
    bucket: list[int] = []
    for size in sizes:
        value = max(int(size), 1)
        rounded = ((value + quantum - 1) // quantum) * quantum
        bucket.append(int(rounded))
    return tuple(bucket)


def _narrow_tensor_prefix(tensor: Tensor, shape: Sequence[int]) -> Tensor:
    if tensor.dim() != len(shape):
        raise ValueError(f"cannot narrow tensor with dim={tensor.dim()} to shape={tuple(shape)}")
    view = tensor
    for dim, size in enumerate(shape):
        if int(view.size(dim)) < int(size):
            raise ValueError(f"tensor shape {tuple(tensor.shape)} cannot satisfy requested shape {tuple(shape)}")
        view = view.narrow(dim, 0, int(size))
    return view


@dataclass(frozen=True)
class StructuredScanWorkspaceKey:
    device_type: str
    device_index: int
    dtype_name: str
    backend_family: str
    shape_bucket: tuple[int, ...]


@dataclass
class StructuredScanWorkspace:
    key: StructuredScanWorkspaceKey
    buffers: dict[str, Tensor] = field(default_factory=dict)
    high_water_bytes: int = 0

    def reserve_tensor(
        self,
        name: str,
        shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        zero: bool = False,
    ) -> Tensor:
        requested_shape = tuple(int(dim) for dim in shape)
        existing = self.buffers.get(name)
        if (
            existing is None
            or existing.device != device
            or existing.dtype != dtype
            or existing.dim() != len(requested_shape)
            or any(int(existing.size(dim)) < requested_shape[dim] for dim in range(len(requested_shape)))
        ):
            buffer = torch.empty(requested_shape, device=device, dtype=dtype)
            self.buffers[name] = buffer
            self.high_water_bytes = max(self.high_water_bytes, int(buffer.numel()) * int(buffer.element_size()))
            existing = buffer
        view = _narrow_tensor_prefix(existing, requested_shape)
        if zero:
            view.zero_()
        return view


@dataclass
class StructuredScanScratchArena:
    workspace: StructuredScanWorkspace

    def empty(
        self,
        name: str,
        shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return self.workspace.reserve_tensor(name, shape, device=device, dtype=dtype, zero=False)

    def zeros(
        self,
        name: str,
        shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return self.workspace.reserve_tensor(name, shape, device=device, dtype=dtype, zero=True)


_STRUCTURED_SCAN_WORKSPACE_REGISTRY: dict[StructuredScanWorkspaceKey, StructuredScanWorkspace] = {}


def get_structured_scan_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    backend_family: str,
    shape_bucket: Sequence[int],
) -> StructuredScanWorkspace:
    device_index = int(device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else -1))
    key = StructuredScanWorkspaceKey(
        device_type=str(device.type),
        device_index=device_index,
        dtype_name=_structured_scan_dtype_name(dtype),
        backend_family=str(backend_family),
        shape_bucket=tuple(int(dim) for dim in shape_bucket),
    )
    workspace = _STRUCTURED_SCAN_WORKSPACE_REGISTRY.get(key)
    if workspace is None:
        workspace = StructuredScanWorkspace(key=key)
        _STRUCTURED_SCAN_WORKSPACE_REGISTRY[key] = workspace
    return workspace


@dataclass(frozen=True)
class StructuredScanBackendPolicy:
    name: str
    backend: str
    arch_family: str
    kernel_family: str
    block_threads: int
    items_per_thread: int
    tile_size: int
    split_size: int
    chunk_size: int
    load_path: str
    supports_async_pipeline: bool = False
    supports_tensor_memory_accel: bool = False
    supports_cluster_launch_control: bool = False
    supports_tma: bool = False
    supports_wgmma: bool = False
    supports_tcgen05: bool = False
    use_virtual_shared_fallback: bool = False
    grouped_launch_packing: bool = False
    vector_bytes: int = 4
    elements_per_load: int = 1
    rank_unroll: int = 1
    state_unroll: int = 1
    workspace_mode: str = "tiled_forward"
    workspace_mode_backward: str = "tiled_backward"
    sparse_reorder_mode: str = "none"
    benchmark_family: str = "structured_scan"


@dataclass
class StructuredReducedTransitionCache:
    dense_transition_matrix: Tensor | None = None
    source_prob_splits: tuple[Tensor, ...] = ()
    dest_prob_splits: tuple[Tensor, ...] = ()
    row_sums: Tensor | None = None
    split_size: int = 0
    tile_size: int = 0


@dataclass
class StructuredScanGroupedLaunchPack:
    row_sum_preps: int = 0
    paged_cache_ops: int = 0
    small_decode_updates: int = 0
    grouped_small_scans: int = 0

    def record_row_sum_prep(self) -> None:
        self.row_sum_preps += 1

    def record_paged_cache_op(self, count: int = 1) -> None:
        self.paged_cache_ops += max(int(count), 0)

    def record_small_decode_update(self, count: int = 1) -> None:
        self.small_decode_updates += max(int(count), 0)

    def record_grouped_small_scan(self, count: int = 1) -> None:
        self.grouped_small_scans += max(int(count), 0)


@dataclass
class StructuredScanSegmentPrefixState:
    chunk_index: int
    chunk_start: int
    chunk_end: int
    carry_log_belief: Tensor | None = None
    active_mask: Tensor | None = None


@dataclass
class StructuredScanSegmentedCarryPlan:
    chunk_size: int
    total_seq_len: int
    seq_lens: Tensor | None = None
    prefix_states: list[StructuredScanSegmentPrefixState] = field(default_factory=list)

    def resolve_chunk_carry(self, chunk_index: int, current: Tensor) -> Tensor:
        if chunk_index == 0 or chunk_index - 1 >= len(self.prefix_states):
            return current
        prefix_state = self.prefix_states[chunk_index - 1]
        if prefix_state.carry_log_belief is None:
            return current
        if prefix_state.active_mask is None:
            return prefix_state.carry_log_belief
        return torch.where(prefix_state.active_mask, prefix_state.carry_log_belief, current)

    def record_chunk_summary(
        self,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
        carry_log_belief: Tensor,
    ) -> None:
        active_mask = None
        if self.seq_lens is not None:
            active_mask = (self.seq_lens > int(chunk_end - 1)).unsqueeze(1)
        state = StructuredScanSegmentPrefixState(
            chunk_index=int(chunk_index),
            chunk_start=int(chunk_start),
            chunk_end=int(chunk_end),
            carry_log_belief=carry_log_belief.detach(),
            active_mask=active_mask.detach() if active_mask is not None else None,
        )
        if chunk_index < len(self.prefix_states):
            self.prefix_states[chunk_index] = state
        else:
            self.prefix_states.append(state)


@dataclass(frozen=True)
class StructuredScanLowPrecisionRecipe:
    packed_kind: int
    amax_history_len: int = 16
    scale_momentum: float = 0.95
    update_interval: int = 1
    allow_graph_capture: bool = True


@dataclass
class StructuredScanLowPrecisionState:
    source_amax_history: deque[float]
    dest_amax_history: deque[float]
    source_scale_estimate: float = 1.0
    dest_scale_estimate: float = 1.0
    step: int = 0


@dataclass(frozen=True)
class StructuredScanLowPrecisionMetadata:
    packed_kind: int
    source_amax: float
    dest_amax: float
    source_scale: float
    dest_scale: float
    source_scale_inv: float
    dest_scale_inv: float
    step: int


@dataclass(frozen=True)
class StructuredScanArchSpec:
    arch_family: str
    capability_major: int
    capability_minor: int
    supports_async_pipeline: bool
    supports_tensor_memory_accel: bool
    supports_cluster_launch_control: bool
    supports_tma: bool
    supports_wgmma: bool
    supports_tcgen05: bool


def _make_structured_scan_low_precision_state(
    recipe: StructuredScanLowPrecisionRecipe,
    cache: dict[str, object],
) -> StructuredScanLowPrecisionState:
    state = cache.get("low_precision_state")
    if isinstance(state, StructuredScanLowPrecisionState):
        return state
    created = StructuredScanLowPrecisionState(
        source_amax_history=deque(maxlen=max(int(recipe.amax_history_len), 1)),
        dest_amax_history=deque(maxlen=max(int(recipe.amax_history_len), 1)),
    )
    cache["low_precision_state"] = created
    return created


def _update_structured_scan_low_precision_state(
    *,
    cache: dict[str, object],
    recipe: StructuredScanLowPrecisionRecipe,
    source_probs: Tensor,
    dest_probs: Tensor,
) -> StructuredScanLowPrecisionState:
    state = _make_structured_scan_low_precision_state(recipe, cache)
    source_amax = float(source_probs.detach().abs().amax().item())
    dest_amax = float(dest_probs.detach().abs().amax().item())
    state.source_amax_history.append(source_amax)
    state.dest_amax_history.append(dest_amax)
    state.step += 1
    if state.step % max(int(recipe.update_interval), 1) == 0:
        source_target = max(state.source_amax_history) if state.source_amax_history else source_amax
        dest_target = max(state.dest_amax_history) if state.dest_amax_history else dest_amax
        momentum = min(max(float(recipe.scale_momentum), 0.0), 0.9999)
        state.source_scale_estimate = momentum * state.source_scale_estimate + (1.0 - momentum) * max(source_target, 1.0e-12)
        state.dest_scale_estimate = momentum * state.dest_scale_estimate + (1.0 - momentum) * max(dest_target, 1.0e-12)
    cache["low_precision_state"] = state
    return state


def _materialize_structured_scan_low_precision_metadata(
    *,
    cache: dict[str, object],
    recipe: StructuredScanLowPrecisionRecipe,
    source_probs: Tensor,
    dest_probs: Tensor,
) -> StructuredScanLowPrecisionMetadata:
    state = _update_structured_scan_low_precision_state(
        cache=cache,
        recipe=recipe,
        source_probs=source_probs,
        dest_probs=dest_probs,
    )
    source_amax = float(source_probs.detach().abs().amax().item())
    dest_amax = float(dest_probs.detach().abs().amax().item())
    source_scale = max(float(state.source_scale_estimate), 1.0e-12)
    dest_scale = max(float(state.dest_scale_estimate), 1.0e-12)
    metadata = StructuredScanLowPrecisionMetadata(
        packed_kind=int(recipe.packed_kind),
        source_amax=source_amax,
        dest_amax=dest_amax,
        source_scale=source_scale,
        dest_scale=dest_scale,
        source_scale_inv=1.0 / source_scale,
        dest_scale_inv=1.0 / dest_scale,
        step=int(state.step),
    )
    cache["low_precision_metadata"] = metadata
    return metadata


def _describe_structured_scan_arch_spec(device: torch.device) -> StructuredScanArchSpec:
    if device.type != "cuda":
        return StructuredScanArchSpec(
            arch_family="cpu",
            capability_major=0,
            capability_minor=0,
            supports_async_pipeline=False,
            supports_tensor_memory_accel=False,
            supports_cluster_launch_control=False,
            supports_tma=False,
            supports_wgmma=False,
            supports_tcgen05=False,
        )
    capability_major, capability_minor = torch.cuda.get_device_capability(device)
    if capability_major >= 10:
        return StructuredScanArchSpec(
            arch_family="sm100+",
            capability_major=int(capability_major),
            capability_minor=int(capability_minor),
            supports_async_pipeline=True,
            supports_tensor_memory_accel=True,
            supports_cluster_launch_control=True,
            supports_tma=False,
            supports_wgmma=False,
            supports_tcgen05=True,
        )
    if capability_major >= 9:
        return StructuredScanArchSpec(
            arch_family="sm90+",
            capability_major=int(capability_major),
            capability_minor=int(capability_minor),
            supports_async_pipeline=True,
            supports_tensor_memory_accel=True,
            supports_cluster_launch_control=True,
            supports_tma=False,
            supports_wgmma=False,
            supports_tcgen05=False,
        )
    if capability_major >= 8:
        return StructuredScanArchSpec(
            arch_family="sm80+",
            capability_major=int(capability_major),
            capability_minor=int(capability_minor),
            supports_async_pipeline=True,
            supports_tensor_memory_accel=False,
            supports_cluster_launch_control=False,
            supports_tma=False,
            supports_wgmma=False,
            supports_tcgen05=False,
        )
    return StructuredScanArchSpec(
        arch_family="legacy",
        capability_major=int(capability_major),
        capability_minor=int(capability_minor),
        supports_async_pipeline=False,
        supports_tensor_memory_accel=False,
        supports_cluster_launch_control=False,
        supports_tma=False,
        supports_wgmma=False,
        supports_tcgen05=False,
    )


@dataclass
class StructuredScanGraphRuntime:
    name: str
    device: torch.device
    static_buffers: dict[str, Tensor] = field(default_factory=dict)
    shape_bucket: tuple[int, ...] = field(default_factory=tuple)
    stable_pool_backed_memory: bool = True
    graph_safe_rng_state: bool = True
    capture_count: int = 0

    def reserve_like(self, name: str, tensor: Tensor) -> Tensor:
        existing = self.static_buffers.get(name)
        if existing is None or existing.shape != tensor.shape or existing.dtype != tensor.dtype or existing.device != tensor.device:
            existing = torch.empty_like(tensor)
            self.static_buffers[name] = existing
        return existing


_STRUCTURED_SCAN_GRAPH_RUNTIME_REGISTRY: dict[tuple[str, str, int, tuple[int, ...]], StructuredScanGraphRuntime] = {}


def get_structured_scan_graph_runtime(
    *,
    name: str,
    device: torch.device,
    shape_bucket: Sequence[int],
) -> StructuredScanGraphRuntime:
    device_index = int(device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else -1))
    key = (str(name), str(device.type), device_index, tuple(int(dim) for dim in shape_bucket))
    runtime = _STRUCTURED_SCAN_GRAPH_RUNTIME_REGISTRY.get(key)
    if runtime is None:
        runtime = StructuredScanGraphRuntime(
            name=str(name),
            device=device,
            shape_bucket=tuple(int(dim) for dim in shape_bucket),
        )
        _STRUCTURED_SCAN_GRAPH_RUNTIME_REGISTRY[key] = runtime
    return runtime


def _default_structured_scan_low_precision_recipe() -> StructuredScanLowPrecisionRecipe:
    return StructuredScanLowPrecisionRecipe(
        packed_kind=_structured_scan_packed_kind(),
        amax_history_len=max(_cached_env_int("CAUSAL_MACHINE_SCAN_AMAX_HISTORY", 16), 1),
        scale_momentum=min(max(_cached_env_float("CAUSAL_MACHINE_SCAN_SCALE_MOMENTUM", 0.95), 0.0), 0.9999),
        update_interval=max(_cached_env_int("CAUSAL_MACHINE_SCAN_SCALE_UPDATE_INTERVAL", 1), 1),
        allow_graph_capture=not _env_enabled("CAUSAL_MACHINE_SCAN_DISABLE_GRAPH_SAFE_PACKING", False),
    )


def _supports_virtual_shared_fallback(
    *,
    device_runtime: dict[str, int],
    num_states: int,
    transition_rank: int,
) -> bool:
    total_mem = int(device_runtime.get("total_global_mem_bytes", 0))
    max_dynamic_smem_bytes = int(device_runtime.get("max_dynamic_smem_bytes", 0))
    if total_mem <= 0:
        return False
    if max_dynamic_smem_bytes >= 96 * 1024:
        return False
    return int(num_states) > _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES and int(transition_rank) > 64


def _select_structured_scan_backend_policy(
    *,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    device: torch.device,
    default_chunk_size: int,
    needs_grad: bool,
    runtime_config: "StructuredScanRuntimeConfig" | None,
) -> StructuredScanBackendPolicy:
    device_runtime = _describe_structured_scan_device_runtime(device)
    arch_spec = _describe_structured_scan_arch_spec(device)
    major = int(arch_spec.capability_major)
    max_dynamic_smem_bytes = int(device_runtime.get("max_dynamic_smem_bytes", 0))
    ext_dense_policy = _select_structured_scan_dense_policy(
        device,
        num_states=int(num_states),
        transition_rank=int(transition_rank),
        seq_len=int(seq_len),
        chunk_size=int(default_chunk_size),
        needs_grad=bool(needs_grad),
    ) if int(num_states) <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES else None
    ext_tiled_policy = _select_structured_scan_tiled_policy(
        device,
        num_states=int(num_states),
        transition_rank=int(transition_rank),
        seq_len=int(seq_len),
        chunk_size=int(default_chunk_size),
        needs_grad=bool(needs_grad),
    ) if int(num_states) > _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES else None
    use_virtual_shared = _supports_virtual_shared_fallback(
        device_runtime=device_runtime,
        num_states=int(num_states),
        transition_rank=int(transition_rank),
    )
    backend = "python"
    if device.type == "cuda" and supports_structured_scan_cuda_config(num_states, transition_rank):
        backend = "cuda"
    if int(num_states) > _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        backend = "cuda_tiled" if device.type == "cuda" else "python_tiled"
    tile_size = _default_structured_scan_tile_size(num_states)
    split_size = _default_structured_scan_split_size(num_states, transition_rank)
    chunk_size = max(int(default_chunk_size), 1)
    block_threads = 128
    items_per_thread = 1
    vector_bytes = 4
    elements_per_load = 1
    rank_unroll = 1
    state_unroll = 1
    load_path = "direct"
    kernel_family = "scalar_fallback"
    workspace_mode = "tiled_forward"
    workspace_mode_backward = "tiled_backward"
    sparse_reorder_mode = "none"
    benchmark_family = "structured_scan"
    grouped = bool(runtime_config.grouped_launch_packing) if runtime_config is not None else True
    if int(num_states) <= _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        name = "sm80_small_state"
        if major >= 9:
            name = "sm90_small_state"
            block_threads = 128
            items_per_thread = 2 if seq_len >= 1024 else 1
            chunk_size = max(chunk_size, 128 if transition_rank <= 32 else 96)
            load_path = "async_shared"
            kernel_family = "small_state_persistent_async"
        elif major >= 8:
            block_threads = 128
            items_per_thread = 1 if transition_rank > 32 else 2
            chunk_size = max(chunk_size, 96 if transition_rank <= 32 else 80)
            load_path = "shared"
            kernel_family = "small_state_persistent_shared"
        else:
            name = "legacy_small_state"
            block_threads = 64
            load_path = "direct"
            kernel_family = "small_state_legacy"
        tile_size = min(int(num_states), _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES)
        split_size = min(int(transition_rank), tile_size)
        workspace_mode = "dense_forward"
        workspace_mode_backward = "dense_backward"
        benchmark_family = "small_state"
        if ext_dense_policy is not None:
            name = str(ext_dense_policy.get("selection_reason", name))
            kernel_family = str(ext_dense_policy.get("kernel_family", kernel_family))
            block_threads = int(ext_dense_policy.get("block_threads", block_threads))
            items_per_thread = int(ext_dense_policy.get("items_per_thread", items_per_thread))
            vector_bytes = int(ext_dense_policy.get("vector_bytes", vector_bytes))
            elements_per_load = int(ext_dense_policy.get("elements_per_load", elements_per_load))
            rank_unroll = int(ext_dense_policy.get("rank_unroll", rank_unroll))
            state_unroll = int(ext_dense_policy.get("state_unroll", state_unroll))
            load_path = str(ext_dense_policy.get("load_path", load_path))
            workspace_mode = str(ext_dense_policy.get("workspace_mode", workspace_mode))
            workspace_mode_backward = str(ext_dense_policy.get("workspace_mode_backward", workspace_mode_backward))
            tile_size = int(ext_dense_policy.get("tile_size", tile_size))
            split_size = int(ext_dense_policy.get("split_size", split_size))
    elif _structured_runtime_uses_transition_masking(runtime_config):
        name = "masked_tiled"
        block_threads = 256 if major >= 9 else 128
        items_per_thread = 1
        tile_size = min(int(num_states), 192 if major >= 9 and max_dynamic_smem_bytes >= 160 * 1024 else 128)
        split_size = min(int(transition_rank), 96 if major >= 8 else 64)
        chunk_size = max(chunk_size, 128 if seq_len >= 1024 else 96)
        load_path = "async_shared" if major >= 9 else ("virtual_shared" if use_virtual_shared else "shared")
        kernel_family = (
            "sm100_masked_tiled_async_proto"
            if major >= 10
            else "sm90_masked_tiled_async"
            if major >= 9
            else "sm80_masked_tiled_shared"
        )
        workspace_mode = "masked_tiled_forward"
        workspace_mode_backward = "masked_tiled_backward"
        benchmark_family = "masked_tiled"
    elif backend == "cuda_tiled":
        name = "sm90_tiled" if major >= 9 else "sm80_tiled"
        block_threads = 256 if major >= 9 else 128
        items_per_thread = 1 if needs_grad else 2
        tile_size = min(int(num_states), 192 if major >= 9 and max_dynamic_smem_bytes >= 160 * 1024 else 160 if major >= 8 else 128)
        split_size = min(int(transition_rank), 128 if major >= 9 else 96 if major >= 8 else 64)
        chunk_size = max(chunk_size, 160 if seq_len >= 2048 else 128 if seq_len >= 1024 else 96)
        load_path = "async_shared" if major >= 9 else ("virtual_shared" if use_virtual_shared else "shared")
        workspace_mode = "tiled_backward" if needs_grad else "tiled_forward"
        workspace_mode_backward = "tiled_backward"
        kernel_family = (
            "sm100_tiled_async_proto"
            if major >= 10
            else "sm90_tiled_async"
            if major >= 9
            else "sm80_tiled_shared"
        )
        benchmark_family = "tiled_large_state"
    else:
        name = "python_fallback"
        tile_size = min(int(num_states), max(int(tile_size), 128))
        split_size = min(int(transition_rank), max(int(split_size), 64))
        load_path = "virtual_shared" if use_virtual_shared else "direct"
        benchmark_family = "python_fallback"
    if ext_tiled_policy is not None:
        block_threads = int(ext_tiled_policy.get("block_threads", block_threads))
        items_per_thread = int(ext_tiled_policy.get("items_per_thread", items_per_thread))
        vector_bytes = int(ext_tiled_policy.get("vector_bytes", vector_bytes))
        elements_per_load = int(ext_tiled_policy.get("elements_per_load", elements_per_load))
        rank_unroll = int(ext_tiled_policy.get("rank_unroll", rank_unroll))
        state_unroll = int(ext_tiled_policy.get("state_unroll", state_unroll))
        load_path = str(ext_tiled_policy.get("load_path", load_path))
        workspace_mode = str(ext_tiled_policy.get("workspace_mode", workspace_mode))
        tile_size = int(ext_tiled_policy.get("tile_size", tile_size))
        split_size = int(ext_tiled_policy.get("split_size", split_size))
    if runtime_config is not None and _structured_runtime_prefers_sparse_cuda(runtime_config, num_states=int(num_states)):
        sparse_reorder_mode = "grouped_src_blocks"
    if runtime_config is not None and runtime_config.backend not in {"", "auto"}:
        backend = str(runtime_config.backend).strip().lower()
    return StructuredScanBackendPolicy(
        name=name,
        backend=backend,
        arch_family=str(arch_spec.arch_family),
        kernel_family=kernel_family,
        block_threads=int(block_threads),
        items_per_thread=int(items_per_thread),
        tile_size=max(int(tile_size), 1),
        split_size=max(int(split_size), 1),
        chunk_size=max(int(chunk_size), 1),
        load_path=load_path,
        supports_async_pipeline=bool(arch_spec.supports_async_pipeline and load_path == "async_shared"),
        supports_tensor_memory_accel=bool(arch_spec.supports_tensor_memory_accel),
        supports_cluster_launch_control=bool(arch_spec.supports_cluster_launch_control),
        supports_tma=bool(arch_spec.supports_tma),
        supports_wgmma=bool(arch_spec.supports_wgmma),
        supports_tcgen05=bool(arch_spec.supports_tcgen05),
        use_virtual_shared_fallback=bool(use_virtual_shared),
        grouped_launch_packing=bool(grouped),
        vector_bytes=max(int(vector_bytes), 4),
        elements_per_load=max(int(elements_per_load), 1),
        rank_unroll=max(int(rank_unroll), 1),
        state_unroll=max(int(state_unroll), 1),
        workspace_mode=workspace_mode,
        workspace_mode_backward=workspace_mode_backward,
        sparse_reorder_mode=sparse_reorder_mode,
        benchmark_family=benchmark_family,
    )


@dataclass
class CausalMachineCache:
    log_belief: Tensor | None = None
    latent_state: Tensor | None = None
    num_updates: int = 0
    workspace: StructuredScanWorkspace | None = None
    grouped_launch_pack: StructuredScanGroupedLaunchPack | None = None
    quant_transition_cache: dict[str, Any] = field(default_factory=dict)
    save_mode: str = "all"
    page_size: int | None = None
    max_pages: int = 0
    paged_layout: str = "belief_major"
    page_table_owner: str = "cache"
    paged_log_beliefs: Tensor | None = None
    paged_latent_states: Tensor | None = None
    paged_lengths: Tensor | None = None
    paged_page_table: Tensor | None = None
    resident_state_valid: bool = True
    last_paged_write_backend: str = "disabled"
    last_paged_write_error: str | None = None
    last_paged_read_backend: str = "disabled"
    last_paged_read_error: str | None = None
    step_graph_runner: dict[str, object] | None = None

    def enable_paged_history(
        self,
        *,
        batch_size: int,
        num_states: int,
        device: torch.device,
        dtype: torch.dtype,
        latent_rank: int = 0,
        page_size: int = 128,
        max_pages: int = 32,
        workspace: StructuredScanWorkspace | None = None,
        grouped_launch_pack: StructuredScanGroupedLaunchPack | None = None,
        save_mode: str = "all",
        paged_layout: str = "belief_major",
    ) -> None:
        page_size = max(int(page_size), 1)
        max_pages = max(int(max_pages), 1)
        self.workspace = workspace
        self.grouped_launch_pack = grouped_launch_pack
        self.save_mode = str(save_mode)
        self.page_size = page_size
        self.max_pages = max_pages
        self.paged_layout = str(paged_layout)
        num_slots = batch_size * max_pages
        if workspace is None:
            workspace = get_structured_scan_workspace(
                device=device,
                dtype=dtype,
                backend_family="paged_cache",
                shape_bucket=_structured_scan_shape_bucket(batch_size, max_pages, page_size, num_states, quantum=16),
            )
            self.workspace = workspace
        self.paged_log_beliefs = workspace.reserve_tensor(
            "paged_log_beliefs",
            (num_slots, page_size, num_states),
            device=device,
            dtype=dtype,
            zero=True,
        )
        self.paged_latent_states = (
            workspace.reserve_tensor(
                "paged_latent_states",
                (num_slots, page_size, latent_rank),
                device=device,
                dtype=torch.float32,
                zero=True,
            )
            if latent_rank > 0
            else None
        )
        self.paged_lengths = workspace.reserve_tensor(
            "paged_lengths",
            (batch_size,),
            device=device,
            dtype=torch.int64,
            zero=True,
        )
        self.paged_page_table = workspace.reserve_tensor(
            "paged_page_table",
            (batch_size, max_pages),
            device=device,
            dtype=torch.int64,
            zero=False,
        )
        self.paged_page_table.copy_(
            torch.arange(num_slots, device=device, dtype=torch.int64).view(batch_size, max_pages)
        )
        self.page_table_owner = "runtime"
        self.resident_state_valid = self.log_belief is not None
        self.last_paged_write_backend = "ready"
        self.last_paged_write_error = None
        self.last_paged_read_backend = "ready"
        self.last_paged_read_error = None
        self.step_graph_runner = None

    def restore_latest_from_paged(self) -> None:
        if (
            self.paged_log_beliefs is None
            or self.paged_lengths is None
            or int(self.paged_log_beliefs.numel()) == 0
            or int(self.paged_lengths.numel()) == 0
        ):
            return
        restored = False
        if self.paged_log_beliefs.is_cuda and self.paged_lengths.is_cuda:
            try:
                ext = load_causal_machine_scan_cuda()
                paged_latent_states = (
                    self.paged_latent_states
                    if self.paged_latent_states is not None
                    else torch.empty((0, 0, 0, 0), device=self.paged_log_beliefs.device, dtype=torch.float32)
                )
                log_belief, latent_state = ext.read_paged_latest_(
                    self.paged_log_beliefs,
                    paged_latent_states,
                    self.paged_page_table,
                    self.paged_lengths,
                )
                self.log_belief = log_belief.detach()
                if latent_state.numel() > 0:
                    self.latent_state = latent_state.detach().float()
                else:
                    self.latent_state = None
                self.last_paged_read_backend = "cuda"
                self.last_paged_read_error = None
                self.resident_state_valid = True
                restored = True
            except Exception:
                self.last_paged_read_backend = "python"
                self.last_paged_read_error = "read_paged_latest_cuda_failed"
                restored = False
        if restored:
            return
        if not bool((self.paged_lengths > 0).any().item()):
            return
        lengths = self.paged_lengths.to(device=self.paged_log_beliefs.device, dtype=torch.int64)
        clamped = lengths.clamp(min=1, max=self.max_pages * max(int(self.page_size or 1), 1)) - 1
        page_idx = torch.div(clamped, int(self.page_size or 1), rounding_mode="floor")
        page_offset = torch.remainder(clamped, int(self.page_size or 1))
        if self.paged_page_table is not None:
            page_idx = self.paged_page_table[torch.arange(self.paged_lengths.size(0), device=self.paged_log_beliefs.device), page_idx]
        self.log_belief = self.paged_log_beliefs[page_idx, page_offset, :].detach()
        if self.paged_latent_states is not None:
            self.latent_state = self.paged_latent_states[page_idx, page_offset, :].detach().float()
        else:
            self.latent_state = None
        self.resident_state_valid = True
        self.last_paged_read_backend = "python"
        self.last_paged_read_error = None

    def drop_resident_state(self) -> None:
        self.log_belief = None
        self.latent_state = None
        self.resident_state_valid = False

    def ensure_resident_state(self) -> None:
        if self.log_belief is None and self.paged_log_beliefs is not None:
            self.restore_latest_from_paged()

    def record_step(self, log_belief: Tensor, latent_state: Tensor | None = None) -> None:
        self.log_belief = log_belief.detach()
        self.resident_state_valid = True
        if latent_state is not None:
            self.latent_state = latent_state.detach().float()
        elif self.latent_state is not None:
            self.latent_state = self.latent_state.detach().float()
        if (
            self.page_size is not None
            and self.page_size > 0
            and self.paged_log_beliefs is not None
            and self.paged_lengths is not None
            and self.save_mode != "none"
        ):
            if self.grouped_launch_pack is not None:
                self.grouped_launch_pack.record_paged_cache_op()
            paged_written = False
            if (
                self.log_belief.is_cuda
                and self.paged_log_beliefs.is_cuda
                and self.paged_lengths.is_cuda
            ):
                try:
                    ext = load_causal_machine_scan_cuda()
                    paged_latent_states = (
                        self.paged_latent_states
                        if self.paged_latent_states is not None
                        else torch.empty((0, 0, 0, 0), device=self.paged_log_beliefs.device, dtype=torch.float32)
                    )
                    latent_to_store = (
                        self.latent_state
                        if self.latent_state is not None and self.paged_latent_states is not None
                        else torch.empty((0, 0), device=self.paged_log_beliefs.device, dtype=torch.float32)
                    )
                    ext.record_paged_step_(
                        self.paged_log_beliefs,
                        paged_latent_states,
                        self.paged_page_table,
                        self.paged_lengths,
                        self.log_belief.to(device=self.paged_log_beliefs.device, dtype=self.paged_log_beliefs.dtype),
                        latent_to_store,
                        int(self.num_updates),
                    )
                    paged_written = True
                    self.last_paged_write_backend = "cuda"
                    self.last_paged_write_error = None
                except Exception:
                    paged_written = False
                    self.last_paged_write_backend = "python"
                    self.last_paged_write_error = "record_paged_step_cuda_failed"
            if not paged_written:
                page_idx = self.num_updates // self.page_size
                page_offset = self.num_updates % self.page_size
                if page_idx < self.max_pages:
                    if self.paged_page_table is not None:
                        slot_idx = self.paged_page_table[:, page_idx]
                    else:
                        slot_idx = torch.arange(self.paged_lengths.size(0), device=self.paged_log_beliefs.device, dtype=torch.int64) * self.max_pages + int(page_idx)
                    belief_to_store = self.log_belief.to(
                        device=self.paged_log_beliefs.device,
                        dtype=self.paged_log_beliefs.dtype,
                    )
                    self.paged_log_beliefs[slot_idx, page_offset, :].copy_(belief_to_store)
                    if self.paged_latent_states is not None and self.latent_state is not None:
                        self.paged_latent_states[slot_idx, page_offset, :].copy_(
                            self.latent_state.to(
                                device=self.paged_latent_states.device,
                                dtype=self.paged_latent_states.dtype,
                            )
                        )
                self.paged_lengths.add_(1).clamp_(max=self.max_pages * self.page_size)
                if self.last_paged_write_backend != "cuda":
                    self.last_paged_write_backend = "python"
        self.num_updates += 1

    def record_sequence(self, state_log_beliefs: Tensor, latent_states: Tensor | None = None) -> None:
        if state_log_beliefs.numel() == 0:
            return
        if self.save_mode == "final":
            latent_final = latent_states[:, -1, :] if latent_states is not None and latent_states.numel() > 0 else None
            self.record_step(state_log_beliefs[:, -1, :], latent_final)
            return
        self.log_belief = state_log_beliefs[:, -1, :].detach()
        self.resident_state_valid = True
        last_latent = None
        if latent_states is not None and latent_states.numel() > 0:
            last_latent = latent_states[:, -1, :]
        elif self.latent_state is not None:
            self.latent_state = self.latent_state.detach().float()
        if last_latent is not None:
            self.latent_state = last_latent.detach().float()
        if (
            self.page_size is not None
            and self.page_size > 0
            and self.paged_log_beliefs is not None
            and self.paged_lengths is not None
            and self.save_mode != "none"
        ):
            if self.grouped_launch_pack is not None:
                self.grouped_launch_pack.record_paged_cache_op(int(state_log_beliefs.size(1)))
            paged_written = False
            if (
                state_log_beliefs.is_cuda
                and self.paged_log_beliefs.is_cuda
                and self.paged_lengths.is_cuda
            ):
                try:
                    ext = load_causal_machine_scan_cuda()
                    paged_latent_states = (
                        self.paged_latent_states
                        if self.paged_latent_states is not None
                        else torch.empty((0, 0, 0, 0), device=self.paged_log_beliefs.device, dtype=torch.float32)
                    )
                    latent_states_in = (
                        latent_states
                        if latent_states is not None and self.paged_latent_states is not None
                        else torch.empty((0, 0, 0), device=self.paged_log_beliefs.device, dtype=torch.float32)
                    )
                    ext.record_paged_sequence_(
                        self.paged_log_beliefs,
                        paged_latent_states,
                        self.paged_page_table,
                        self.paged_lengths,
                        state_log_beliefs.to(device=self.paged_log_beliefs.device, dtype=self.paged_log_beliefs.dtype),
                        latent_states_in,
                        int(self.num_updates),
                    )
                    paged_written = True
                    self.last_paged_write_backend = "cuda"
                    self.last_paged_write_error = None
                except Exception:
                    paged_written = False
                    self.last_paged_write_backend = "python"
                    self.last_paged_write_error = "record_paged_sequence_cuda_failed"
            if paged_written:
                self.num_updates += int(state_log_beliefs.size(1))
                return
        for t in range(int(state_log_beliefs.size(1))):
            latent_t = latent_states[:, t, :] if latent_states is not None else None
            self.record_step(state_log_beliefs[:, t, :], latent_t)

    def reset(self) -> None:
        self.log_belief = None
        self.latent_state = None
        self.num_updates = 0
        self.resident_state_valid = False
        if self.paged_log_beliefs is not None:
            self.paged_log_beliefs.zero_()
        if self.paged_latent_states is not None:
            self.paged_latent_states.zero_()
        if self.paged_lengths is not None:
            self.paged_lengths.zero_()
        if self.paged_page_table is not None:
            self.paged_page_table.copy_(
                torch.arange(
                    self.paged_page_table.size(0) * self.max_pages,
                    device=self.paged_page_table.device,
                    dtype=self.paged_page_table.dtype,
                ).view_as(self.paged_page_table)
            )
        self.last_paged_write_backend = "disabled" if self.paged_log_beliefs is None else "ready"
        self.last_paged_write_error = None
        self.last_paged_read_backend = "disabled" if self.paged_log_beliefs is None else "ready"
        self.last_paged_read_error = None
        self.step_graph_runner = None

    def reorder_batch(self, beam_indices: Tensor) -> None:
        if self.paged_page_table is None or self.paged_lengths is None:
            return
        order = beam_indices.to(device=self.paged_page_table.device, dtype=torch.int64).contiguous()
        reordered = False
        if self.paged_page_table.is_cuda and self.paged_lengths.is_cuda:
            try:
                ext = load_causal_machine_scan_cuda()
                ext.reorder_paged_cache_(self.paged_page_table, self.paged_lengths, order)
                reordered = True
                self.last_paged_read_backend = "cuda"
                self.last_paged_read_error = None
            except Exception:
                reordered = False
        if not reordered:
            self.paged_page_table.copy_(self.paged_page_table.index_select(0, order))
            self.paged_lengths.copy_(self.paged_lengths.index_select(0, order))
        if self.log_belief is not None:
            self.log_belief = self.log_belief.index_select(0, order.to(device=self.log_belief.device))
        if self.latent_state is not None:
            self.latent_state = self.latent_state.index_select(0, order.to(device=self.latent_state.device))


@dataclass(frozen=True)
class StructuredScanKernelConfig:
    num_states: int
    transition_rank: int
    chunk_size: int
    tile_size: int = 128
    split_size: int = 128
    backend: str = "auto"
    policy_name: str = "default"
    arch_family: str = "legacy"
    kernel_family: str = "scalar_fallback"
    block_threads: int = 128
    items_per_thread: int = 1
    load_path: str = "direct"
    vector_bytes: int = 4
    elements_per_load: int = 1
    rank_unroll: int = 1
    state_unroll: int = 1
    supports_async_pipeline: bool = False
    supports_tensor_memory_accel: bool = False
    supports_cluster_launch_control: bool = False
    supports_tma: bool = False
    supports_wgmma: bool = False
    supports_tcgen05: bool = False
    workspace_mode: str = "tiled_forward"
    workspace_mode_backward: str = "tiled_backward"
    sparse_reorder_mode: str = "none"
    benchmark_family: str = "structured_scan"
    use_virtual_shared_fallback: bool = False
    grouped_launch_packing: bool = False
    allow_cuda: bool = True
    allow_tiled_cuda: bool = False
    allow_quantized_tables: bool = True


@dataclass
class StructuredScoreModSpec:
    additive_bias: Tensor | None = None
    state_mask: Tensor | None = None
    pred_scale: float | Tensor | None = None
    threshold: float | Tensor | None = None
    topk: int | None = None
    clamp_min: float | None = None
    clamp_max: float | None = None
    masked_bias_value: float = -1.0e4


def _structured_runtime_score_mod_spec(
    runtime_config: "StructuredScanRuntimeConfig" | None,
) -> StructuredScoreModSpec | None:
    if runtime_config is None:
        return None
    score_mod = runtime_config.score_mod
    return score_mod if isinstance(score_mod, StructuredScoreModSpec) else None


def _structured_runtime_score_mod_callback(
    runtime_config: "StructuredScanRuntimeConfig" | None,
) -> Callable[[Tensor, int, "StructuredScanRuntimeConfig"], Tensor] | None:
    if runtime_config is None:
        return None
    score_mod = runtime_config.score_mod
    if score_mod is None or isinstance(score_mod, StructuredScoreModSpec):
        return None
    if not callable(score_mod):
        raise TypeError("score_mod must be callable or StructuredScoreModSpec")
    return score_mod


@dataclass
class StructuredScanRuntimeConfig:
    seq_lens: Tensor | Sequence[int] | None = None
    local_transition_window: int | None = None
    transition_mask: Tensor | None = None
    block_mask: Tensor | None = None
    block_size: int | None = None
    score_bias: Tensor | None = None
    transition_mask_mod: Callable[[Tensor, Tensor, StructuredScanRuntimeConfig], Tensor] | None = None
    score_mod: Callable[[Tensor, int, "StructuredScanRuntimeConfig"], Tensor] | StructuredScoreModSpec | None = None
    backend: str = "auto"
    allow_cuda: bool = True
    workspace: StructuredScanWorkspace | None = None
    scratch_arena: StructuredScanScratchArena | None = None
    backend_policy: StructuredScanBackendPolicy | None = None
    grouped_launch_pack: StructuredScanGroupedLaunchPack | None = None
    prefix_carry_plan: StructuredScanSegmentedCarryPlan | None = None
    reduced_transition_cache: StructuredReducedTransitionCache | None = None
    low_precision_recipe: StructuredScanLowPrecisionRecipe | None = None
    low_precision_metadata: StructuredScanLowPrecisionMetadata | None = None
    graph_runtime: StructuredScanGraphRuntime | None = None
    grouped_launch_packing: bool = True
    allow_virtual_shared_fallback: bool = True
    batch_mode: str = "auto"
    save_mode: str = "all"
    use_paged_cache: bool = False
    paged_resident_only: bool = False
    use_cuda_graphs: bool = True
    page_size: int = 128
    max_pages: int = 32
    paged_layout: str = "belief_major"


def _resolve_structured_scan_runtime_config(
    runtime_config: StructuredScanRuntimeConfig | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    backend_family: str,
    batch_size: int,
    seq_len: int,
    num_states: int,
) -> StructuredScanRuntimeConfig:
    runtime = StructuredScanRuntimeConfig() if runtime_config is None else runtime_config
    if runtime.workspace is None:
        runtime.workspace = get_structured_scan_workspace(
            device=device,
            dtype=dtype,
            backend_family=backend_family,
            shape_bucket=_structured_scan_shape_bucket(batch_size, seq_len, num_states, quantum=32),
        )
    if runtime.scratch_arena is None:
        runtime.scratch_arena = StructuredScanScratchArena(runtime.workspace)
    if runtime.grouped_launch_pack is None:
        runtime.grouped_launch_pack = StructuredScanGroupedLaunchPack()
    if str(runtime.batch_mode or "auto").strip().lower() in {"", "auto"}:
        runtime.batch_mode = "single" if int(batch_size) == 1 else "batched"
    if runtime.prefix_carry_plan is None:
        runtime.prefix_carry_plan = StructuredScanSegmentedCarryPlan(
            chunk_size=1,
            total_seq_len=max(int(seq_len), 0),
            seq_lens=_canonicalize_structured_seq_lens(runtime.seq_lens, batch_size=batch_size, seq_len=seq_len, device=device)
            if runtime.seq_lens is not None
            else None,
        )
    if runtime.low_precision_recipe is None:
        runtime.low_precision_recipe = _default_structured_scan_low_precision_recipe()
    if runtime.graph_runtime is None and device.type == "cuda":
        runtime.graph_runtime = get_structured_scan_graph_runtime(
            name=f"structured_scan:{backend_family}",
            device=device,
            shape_bucket=_structured_scan_shape_bucket(batch_size, seq_len, num_states, quantum=32),
        )
    return runtime


def _structured_scan_save_mode(runtime_config: StructuredScanRuntimeConfig | None) -> str:
    raw = "all" if runtime_config is None else str(runtime_config.save_mode or "all").strip().lower()
    if raw not in {"all", "final", "none"}:
        raise ValueError(f"unsupported structured scan save_mode={raw!r}")
    return raw


def _structured_scan_save_all(runtime_config: StructuredScanRuntimeConfig | None) -> bool:
    return _structured_scan_save_mode(runtime_config) == "all"


def supports_structured_scan_cuda_config(num_states: int, transition_rank: int) -> bool:
    return (
        _supports_structured_scan_num_states(num_states)
        and _supports_structured_scan_transition_rank(transition_rank)
    )


def _structured_runtime_uses_generalized_features(runtime_config: StructuredScanRuntimeConfig | None) -> bool:
    if runtime_config is None:
        return False
    return bool(
        runtime_config.seq_lens is not None
        or runtime_config.local_transition_window is not None
        or runtime_config.transition_mask is not None
        or runtime_config.block_mask is not None
        or runtime_config.transition_mask_mod is not None
        or _structured_runtime_score_mod_callback(runtime_config) is not None
        or runtime_config.backend not in {"", "auto", "cuda", "cuda_tiled", "python_tiled", "python"}
        or not runtime_config.allow_cuda
    )


def _structured_runtime_uses_transition_masking(runtime_config: StructuredScanRuntimeConfig | None) -> bool:
    if runtime_config is None:
        return False
    return bool(
        runtime_config.local_transition_window is not None
        or runtime_config.transition_mask is not None
        or runtime_config.block_mask is not None
        or runtime_config.transition_mask_mod is not None
    )


def _structured_runtime_supports_masked_cuda(runtime_config: StructuredScanRuntimeConfig | None) -> bool:
    if runtime_config is None:
        return False
    if not runtime_config.allow_cuda:
        return False
    # The masked CUDA path can consume a fully materialized dense boolean mask,
    # so transition_mask_mod is allowed here after Python lowers it to a mask.
    # Callback-based score_mod still mutates the predicted belief itself and has
    # no kernel contract. Lowered StructuredScoreModSpec values stay eligible.
    if _structured_runtime_score_mod_callback(runtime_config) is not None:
        return False
    backend = str(runtime_config.backend or "").strip().lower()
    if backend not in {"", "auto", "cuda", "cuda_tiled"}:
        return False
    return _structured_runtime_uses_transition_masking(runtime_config)


def _structured_runtime_supports_sparse_cuda(runtime_config: StructuredScanRuntimeConfig | None) -> bool:
    if runtime_config is None:
        return False
    if not runtime_config.allow_cuda:
        return False
    if _has_native_structured_score_clamp(runtime_config) or _has_native_structured_score_filtering(runtime_config):
        return False
    if _structured_runtime_score_mod_callback(runtime_config) is not None:
        return False
    backend = str(runtime_config.backend or "").strip().lower()
    if backend not in {"", "auto", "cuda"}:
        return False
    return _structured_runtime_uses_transition_masking(runtime_config)


def _structured_runtime_supports_tiled_backend(runtime_config: StructuredScanRuntimeConfig | None) -> bool:
    if runtime_config is None:
        return True
    if _structured_runtime_score_mod_callback(runtime_config) is not None:
        return False
    return True


def _structured_runtime_prefers_sparse_cuda(
    runtime_config: StructuredScanRuntimeConfig | None,
    *,
    num_states: int,
) -> bool:
    if runtime_config is None or _structured_runtime_score_mod_callback(runtime_config) is not None:
        return False
    if not _structured_runtime_uses_transition_masking(runtime_config):
        return False
    return bool(int(num_states) > _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES)


def _packed_transition_kind_name(packed_kind: int | None) -> str | None:
    if packed_kind is None:
        return None
    if int(packed_kind) == _PACKED_TRANSITION_INT8:
        return "int8"
    if int(packed_kind) == _PACKED_TRANSITION_FP8_E4M3:
        return "fp8_e4m3"
    if int(packed_kind) == _PACKED_TRANSITION_FP8_E5M2:
        return "fp8_e5m2"
    return f"unknown_{int(packed_kind)}"


def _structured_scan_uses_lowp_tensor_core_path(
    *,
    device: torch.device,
    dtype: torch.dtype,
    kernel_config: StructuredScanKernelConfig,
    packed_transition_tables: tuple[int, Tensor, Tensor, Tensor, Tensor] | None,
) -> bool:
    if packed_transition_tables is None:
        return False
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return False
    if int(kernel_config.num_states) < 16 or int(kernel_config.transition_rank) < 16:
        return False
    device_runtime = _describe_structured_scan_device_runtime(device)
    if dtype == torch.bfloat16 and int(device_runtime.get("capability_major", 0)) < 8:
        return False
    return bool(int(device_runtime.get("supports_wmma", 0)) > 0)


def _structured_scan_kernel_info(
    *,
    path: str,
    kernel_config: StructuredScanKernelConfig,
    runtime_config: StructuredScanRuntimeConfig | None,
    packed_transition_tables: tuple[int, Tensor, Tensor, Tensor, Tensor] | None = None,
    cache: CausalMachineCache | None = None,
) -> dict[str, object]:
    packed_kind = None if packed_transition_tables is None else int(packed_transition_tables[0])
    info: dict[str, object] = {
        "path": str(path),
        "backend": str(kernel_config.backend),
        "policy_name": str(kernel_config.policy_name),
        "arch_family": str(getattr(kernel_config, "arch_family", "legacy")),
        "kernel_family": str(getattr(kernel_config, "kernel_family", "scalar_fallback")),
        "chunk_size": int(kernel_config.chunk_size),
        "tile_size": int(kernel_config.tile_size),
        "split_size": int(kernel_config.split_size),
        "block_threads": int(kernel_config.block_threads),
        "items_per_thread": int(kernel_config.items_per_thread),
        "load_path": str(kernel_config.load_path),
        "vector_bytes": int(getattr(kernel_config, "vector_bytes", 4)),
        "elements_per_load": int(getattr(kernel_config, "elements_per_load", 1)),
        "rank_unroll": int(getattr(kernel_config, "rank_unroll", 1)),
        "state_unroll": int(getattr(kernel_config, "state_unroll", 1)),
        "workspace_mode": str(getattr(kernel_config, "workspace_mode", "tiled_forward")),
        "workspace_mode_backward": str(getattr(kernel_config, "workspace_mode_backward", "tiled_backward")),
        "sparse_reorder_mode": str(getattr(kernel_config, "sparse_reorder_mode", "none")),
        "benchmark_family": str(getattr(kernel_config, "benchmark_family", "structured_scan")),
        "supports_async_pipeline": bool(getattr(kernel_config, "supports_async_pipeline", False)),
        "supports_tensor_memory_accel": bool(getattr(kernel_config, "supports_tensor_memory_accel", False)),
        "supports_cluster_launch_control": bool(getattr(kernel_config, "supports_cluster_launch_control", False)),
        "supports_tma": bool(getattr(kernel_config, "supports_tma", False)),
        "supports_wgmma": bool(getattr(kernel_config, "supports_wgmma", False)),
        "supports_tcgen05": bool(getattr(kernel_config, "supports_tcgen05", False)),
        "use_virtual_shared_fallback": bool(kernel_config.use_virtual_shared_fallback),
        "grouped_launch_packing": bool(kernel_config.grouped_launch_packing),
        "num_states": int(kernel_config.num_states),
        "transition_rank": int(kernel_config.transition_rank),
        "allow_cuda": bool(kernel_config.allow_cuda),
        "allow_tiled_cuda": bool(kernel_config.allow_tiled_cuda),
        "allow_quantized_tables": bool(kernel_config.allow_quantized_tables),
        "uses_transition_masking": _structured_runtime_uses_transition_masking(runtime_config),
        "uses_paged_cache": bool(runtime_config.use_paged_cache) if runtime_config is not None else False,
        "packed_transition_kind": _packed_transition_kind_name(packed_kind),
    }
    if "lowp_tensor_core" in str(path):
        arch_family = str(getattr(kernel_config, "arch_family", "legacy"))
        if arch_family.startswith("sm100"):
            info["selected_low_precision_kernel_family"] = "sm100_packed_wmma"
        elif arch_family.startswith("sm90"):
            info["selected_low_precision_kernel_family"] = "sm90_packed_wmma"
        elif arch_family.startswith("sm80"):
            info["selected_low_precision_kernel_family"] = "sm80_packed_wmma"
        else:
            info["selected_low_precision_kernel_family"] = "packed_wmma"
        info["tensor_core_trained_path"] = True
    if runtime_config is not None:
        info["paged_layout"] = str(runtime_config.paged_layout)
        info["allow_virtual_shared_runtime"] = bool(runtime_config.allow_virtual_shared_fallback)
        info["batch_mode"] = str(runtime_config.batch_mode)
        info["save_mode"] = _structured_scan_save_mode(runtime_config)
        if runtime_config.workspace is not None:
            info["workspace_backend_family"] = str(runtime_config.workspace.key.backend_family)
            info["workspace_high_water_bytes"] = int(runtime_config.workspace.high_water_bytes)
        if runtime_config.scratch_arena is not None:
            info["scratch_allocator"] = "persistent_workspace"
        if runtime_config.backend_policy is not None:
            info["backend_policy_runtime"] = str(runtime_config.backend_policy.name)
        if runtime_config.reduced_transition_cache is not None:
            info["has_reduced_transition_cache"] = True
            info["reduced_transition_split_size"] = int(runtime_config.reduced_transition_cache.split_size)
            info["reduced_transition_tile_size"] = int(runtime_config.reduced_transition_cache.tile_size)
        if runtime_config.grouped_launch_pack is not None:
            info["grouped_row_sum_preps"] = int(runtime_config.grouped_launch_pack.row_sum_preps)
            info["grouped_paged_cache_ops"] = int(runtime_config.grouped_launch_pack.paged_cache_ops)
            info["grouped_small_decode_updates"] = int(runtime_config.grouped_launch_pack.small_decode_updates)
            info["grouped_small_scans"] = int(runtime_config.grouped_launch_pack.grouped_small_scans)
        if runtime_config.low_precision_recipe is not None:
            info["low_precision_recipe"] = _packed_transition_kind_name(runtime_config.low_precision_recipe.packed_kind)
            info["low_precision_amax_history_len"] = int(runtime_config.low_precision_recipe.amax_history_len)
        low_precision_metadata = getattr(runtime_config, "low_precision_metadata", None)
        if isinstance(low_precision_metadata, StructuredScanLowPrecisionMetadata):
            info["low_precision_source_amax"] = float(low_precision_metadata.source_amax)
            info["low_precision_dest_amax"] = float(low_precision_metadata.dest_amax)
            info["low_precision_source_scale"] = float(low_precision_metadata.source_scale)
            info["low_precision_dest_scale"] = float(low_precision_metadata.dest_scale)
            info["low_precision_source_scale_inv"] = float(low_precision_metadata.source_scale_inv)
            info["low_precision_dest_scale_inv"] = float(low_precision_metadata.dest_scale_inv)
            info["low_precision_step"] = int(low_precision_metadata.step)
        if runtime_config.graph_runtime is not None:
            info["graph_runtime_name"] = str(runtime_config.graph_runtime.name)
            info["graph_runtime_captures"] = int(runtime_config.graph_runtime.capture_count)
    if cache is not None:
        info["paged_cache_write_backend"] = str(cache.last_paged_write_backend)
        if cache.last_paged_write_error is not None:
            info["paged_cache_write_error"] = str(cache.last_paged_write_error)
        info["paged_cache_read_backend"] = str(cache.last_paged_read_backend)
        if cache.last_paged_read_error is not None:
            info["paged_cache_read_error"] = str(cache.last_paged_read_error)
        info["resident_state_valid"] = bool(cache.resident_state_valid)
        info["page_table_owner"] = str(cache.page_table_owner)
    if cache is not None:
        packed_transition_cache = getattr(cache, "_packed_transition_cache", None)
        cache_low_precision = packed_transition_cache.get("low_precision_metadata") if isinstance(packed_transition_cache, dict) else None
        if isinstance(cache_low_precision, StructuredScanLowPrecisionMetadata):
            info["low_precision_source_amax"] = float(cache_low_precision.source_amax)
            info["low_precision_dest_amax"] = float(cache_low_precision.dest_amax)
            info["low_precision_source_scale"] = float(cache_low_precision.source_scale)
            info["low_precision_dest_scale"] = float(cache_low_precision.dest_scale)
            info["low_precision_source_scale_inv"] = float(cache_low_precision.source_scale_inv)
            info["low_precision_dest_scale_inv"] = float(cache_low_precision.dest_scale_inv)
            info["low_precision_step"] = int(cache_low_precision.step)
    return info


_COMPETITION_STRUCTURED_SCAN_PATHS = frozenset(
    {
        "cuda_dense",
        "cuda_dense_lowp_tensor_core",
        "cuda_masked",
        "cuda_sparse",
    }
)


def _enforce_structured_scan_cuda_contract(info: dict[str, object], *, context: str) -> None:
    if not _require_fused_cuda_path_contract():
        return
    path = str(info.get("path", ""))
    if path in _COMPETITION_STRUCTURED_SCAN_PATHS:
        return
    backend = str(info.get("backend", ""))
    policy_name = str(info.get("policy_name", ""))
    kernel_family = str(info.get("kernel_family", ""))
    raise RuntimeError(
        f"{context} requires the fused CUDA structured-scan path in competition mode; "
        f"got path={path!r}, backend={backend!r}, policy={policy_name!r}, kernel_family={kernel_family!r}. "
        f"Expected one of: {', '.join(sorted(_COMPETITION_STRUCTURED_SCAN_PATHS))}."
    )


def _structured_runtime_supports_fused_paged_step(
    runtime_config: StructuredScanRuntimeConfig | None,
) -> bool:
    if runtime_config is None:
        return False
    if not runtime_config.use_paged_cache or not runtime_config.allow_cuda:
        return False
    if _structured_runtime_uses_transition_masking(runtime_config):
        return False
    if _structured_runtime_score_mod_callback(runtime_config) is not None:
        return False
    return True


def _structured_runtime_supports_step_cuda_graph(
    runtime_config: StructuredScanRuntimeConfig | None,
) -> bool:
    return _structured_runtime_supports_fused_paged_step(runtime_config) and bool(
        runtime_config.use_cuda_graphs if runtime_config is not None else False
    )


def _empty_packed_transition_tensors(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty((0, 0), device=device, dtype=torch.int8),
        torch.empty((0,), device=device, dtype=torch.float32),
        torch.empty((0, 0), device=device, dtype=torch.int8),
        torch.empty((0,), device=device, dtype=torch.float32),
    )


def _update_cache_after_fused_paged_step(
    cache: CausalMachineCache,
    final_log_belief: Tensor,
) -> None:
    cache.log_belief = final_log_belief.detach()
    cache.last_paged_read_backend = "cuda"
    cache.last_paged_read_error = None
    cache.last_paged_write_backend = "cuda"
    cache.last_paged_write_error = None
    cache.resident_state_valid = True
    cache.num_updates += 1


def _structured_fused_paged_step_cuda(
    *,
    local_logits: Tensor,
    transition_source_logits: Tensor,
    transition_dest_logits: Tensor,
    transition_context: Tensor,
    transition_gate: Tensor,
    transition_stay_probs: Tensor,
    cache: CausalMachineCache,
    kernel_config: StructuredScanKernelConfig,
    runtime_config: StructuredScanRuntimeConfig | None,
    packed_transition_tables: tuple[int, Tensor, Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, str]:
    if cache.paged_log_beliefs is None or cache.paged_lengths is None:
        raise RuntimeError("fused paged step requires paged_log_beliefs and paged_lengths")
    ext = load_causal_machine_scan_cuda()
    score_clamp_min, score_clamp_max = _resolve_native_structured_score_clamp_bounds(runtime_config)
    clamp_active = math.isfinite(score_clamp_min) or math.isfinite(score_clamp_max)
    packed_kind = -1
    if packed_transition_tables is None or clamp_active:
        packed_source, packed_source_scales, packed_dest, packed_dest_scales = _empty_packed_transition_tensors(
            local_logits.device
        )
    else:
        packed_kind, packed_source, packed_source_scales, packed_dest, packed_dest_scales = packed_transition_tables
    transition_source_probs = F.softmax(transition_source_logits.contiguous().float(), dim=-1).contiguous()
    transition_dest_probs = F.softmax(transition_dest_logits.contiguous().float(), dim=-1).contiguous()
    paged_latent_states = (
        cache.paged_latent_states
        if cache.paged_latent_states is not None
        else torch.empty((0, 0, 0, 0), device=local_logits.device, dtype=torch.float32)
    )
    beliefs, final_log_belief = ext.paged_step_(
        cache.paged_log_beliefs,
        paged_latent_states,
        cache.paged_page_table,
        cache.paged_lengths,
        local_logits.contiguous(),
        transition_source_probs,
        transition_dest_probs,
        transition_context.contiguous(),
        transition_stay_probs.contiguous().float(),
        transition_gate.reshape(()).contiguous().float(),
        packed_source,
        packed_source_scales,
        packed_dest,
        packed_dest_scales,
        int(packed_kind),
        int(kernel_config.tile_size),
        int(kernel_config.split_size),
        float(score_clamp_min),
        float(score_clamp_max),
    )
    if int(kernel_config.num_states) > _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
        path = "cuda_tiled_paged"
    elif packed_kind >= 0:
        path = (
            "cuda_dense_paged_decode_lowp_tensor_core"
            if _structured_scan_uses_lowp_tensor_core_path(
                device=local_logits.device,
                dtype=local_logits.dtype,
                kernel_config=kernel_config,
                packed_transition_tables=packed_transition_tables,
            )
            else "cuda_dense_paged_decode_lowp"
        )
    else:
        path = (
            f"cuda_{kernel_config.kernel_family}_paged"
            if str(kernel_config.kernel_family).startswith("dense_128_rank8")
            else "cuda_dense_paged"
        )
    return beliefs, final_log_belief, path


def autotune_structured_scan_kernel_config(
    *,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    device: torch.device,
    default_chunk_size: int,
    needs_grad: bool = False,
    runtime_config: StructuredScanRuntimeConfig | None = None,
) -> StructuredScanKernelConfig:
    runtime_config = _resolve_structured_scan_runtime_config(
        runtime_config,
        device=device,
        dtype=torch.float32,
        backend_family="structured_scan",
        batch_size=1,
        seq_len=max(int(seq_len), 1),
        num_states=max(int(num_states), 1),
    )
    policy = _select_structured_scan_backend_policy(
        num_states=int(num_states),
        transition_rank=int(transition_rank),
        seq_len=int(seq_len),
        device=device,
        default_chunk_size=int(default_chunk_size),
        needs_grad=bool(needs_grad),
        runtime_config=runtime_config,
    )
    runtime_config.backend_policy = policy
    chunk_size = max(int(policy.chunk_size), 1)
    tile_size = max(int(policy.tile_size), 1)
    split_size = max(int(policy.split_size), 1)
    device_runtime = _describe_structured_scan_device_runtime(device)
    major = int(device_runtime.get("capability_major", 0))
    max_dynamic_smem_bytes = int(device_runtime.get("max_dynamic_smem_bytes", 0))
    sm_count = int(device_runtime.get("sm_count", 0))
    if device.type == "cuda":
        if major >= 9 and transition_rank <= 32 and seq_len >= 1024:
            chunk_size = max(chunk_size, 128)
        elif major >= 8 and transition_rank <= 16 and seq_len >= 512:
            chunk_size = max(chunk_size, 96)
        elif (
            major >= 8
            and _is_optimized_structured_scan_transition_rank(transition_rank)
            and transition_rank <= 64
            and seq_len >= 512
        ):
            chunk_size = max(chunk_size, 80)
        if sm_count >= 96 and seq_len >= 2048:
            chunk_size = max(chunk_size, 160 if transition_rank <= 64 else 128)
        if num_states > _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES:
            if major >= 9:
                tile_size = min(
                    int(num_states),
                    192 if max_dynamic_smem_bytes >= 160 * 1024 and transition_rank <= 192 else 160,
                )
                split_size = min(int(transition_rank), 128 if max_dynamic_smem_bytes >= 160 * 1024 else 96)
            elif major >= 8:
                tile_size = min(
                    int(num_states),
                    160 if max_dynamic_smem_bytes >= 128 * 1024 and transition_rank <= 160 else 128,
                )
                split_size = min(int(transition_rank), 96 if max_dynamic_smem_bytes >= 96 * 1024 else 64)
            else:
                tile_size = min(int(tile_size), 128)
                split_size = min(int(split_size), 64)
            tile_size = max(32, int(tile_size))
            split_size = max(16, int(split_size))
    allow_cuda = (
        device.type == "cuda"
        and supports_structured_scan_cuda_config(num_states, transition_rank)
        and not _structured_runtime_uses_generalized_features(runtime_config)
        and not _has_native_structured_score_filtering(runtime_config, num_states=int(num_states))
        and _can_use_causal_machine_scan_cuda(device, transition_rank, num_states=num_states)
    )
    allow_cuda_tiled = (
        device.type == "cuda"
        and (runtime_config is None or runtime_config.allow_cuda)
        and num_states > _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES
        and transition_rank > 0
        and transition_rank <= num_states
        and _structured_runtime_supports_tiled_backend(runtime_config)
        and _can_use_causal_machine_tiled_scan_cuda(
            device,
            num_states=int(num_states),
            transition_rank=int(transition_rank),
            needs_grad=bool(needs_grad),
            runtime_config=runtime_config,
        )
    )
    if allow_cuda_tiled:
        tiled_kernel_config = _resolve_structured_scan_tiled_kernel_config(
            device,
            num_states=int(num_states),
            transition_rank=int(transition_rank),
            seq_len=int(seq_len),
            chunk_size=int(chunk_size),
            preferred_tile_size=int(tile_size),
            preferred_split_size=int(split_size),
            needs_grad=bool(needs_grad),
            runtime_config=runtime_config,
        )
        if tiled_kernel_config is not None:
            tile_size, split_size = tiled_kernel_config
    allow_python_tiled = (
        num_states > _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES
        and transition_rank > 0
        and transition_rank <= num_states
        and _structured_runtime_supports_tiled_backend(runtime_config)
    )
    backend = "cuda" if allow_cuda else ("cuda_tiled" if allow_cuda_tiled else ("python_tiled" if allow_python_tiled else "python"))
    if runtime_config is not None and runtime_config.backend not in {"", "auto"}:
        backend = str(runtime_config.backend).strip().lower()
        if backend == "cuda" and not allow_cuda:
            backend = "cuda_tiled" if allow_cuda_tiled else ("python_tiled" if allow_python_tiled else "python")
        if backend == "cuda_tiled" and not allow_cuda_tiled:
            backend = "python_tiled" if allow_python_tiled else "python"
        if backend == "python_tiled" and not allow_python_tiled:
            backend = "python"
    return StructuredScanKernelConfig(
        num_states=int(num_states),
        transition_rank=int(transition_rank),
        chunk_size=chunk_size,
        tile_size=tile_size,
        split_size=split_size,
        backend=backend,
        policy_name=str(policy.name),
        arch_family=str(policy.arch_family),
        kernel_family=str(policy.kernel_family),
        block_threads=int(policy.block_threads),
        items_per_thread=int(policy.items_per_thread),
        load_path=str(policy.load_path),
        vector_bytes=int(policy.vector_bytes),
        elements_per_load=int(policy.elements_per_load),
        rank_unroll=int(policy.rank_unroll),
        state_unroll=int(policy.state_unroll),
        supports_async_pipeline=bool(policy.supports_async_pipeline),
        supports_tensor_memory_accel=bool(policy.supports_tensor_memory_accel),
        supports_cluster_launch_control=bool(policy.supports_cluster_launch_control),
        supports_tma=bool(policy.supports_tma),
        supports_wgmma=bool(policy.supports_wgmma),
        supports_tcgen05=bool(policy.supports_tcgen05),
        workspace_mode=str(policy.workspace_mode),
        workspace_mode_backward=str(policy.workspace_mode_backward),
        sparse_reorder_mode=str(policy.sparse_reorder_mode),
        benchmark_family=str(policy.benchmark_family),
        use_virtual_shared_fallback=bool(policy.use_virtual_shared_fallback and runtime_config.allow_virtual_shared_fallback),
        grouped_launch_packing=bool(policy.grouped_launch_packing and runtime_config.grouped_launch_packing),
        allow_cuda=allow_cuda,
        allow_tiled_cuda=allow_cuda_tiled,
        allow_quantized_tables=allow_cuda and major >= 8,
    )


def _structured_scan_kernel_config_cache_key(
    device: torch.device,
    *,
    seq_len: int,
    needs_grad: bool,
) -> tuple[str, int, int, bool]:
    device_index = int(device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else -1))
    return (str(device.type), device_index, int(seq_len), bool(needs_grad))


def _structured_scan_runtime_allows_precomputed_kernel_config(
    runtime_config: StructuredScanRuntimeConfig | None,
) -> bool:
    return runtime_config is None


def _get_precomputed_structured_scan_kernel_config(
    owner: object,
    *,
    cache_attr: str,
    num_states: int,
    transition_rank: int,
    seq_len: int,
    device: torch.device,
    default_chunk_size: int,
    needs_grad: bool,
    runtime_config: StructuredScanRuntimeConfig | None,
) -> StructuredScanKernelConfig:
    cache_key = _structured_scan_kernel_config_cache_key(
        device,
        seq_len=int(seq_len),
        needs_grad=bool(needs_grad),
    )
    if _structured_scan_runtime_allows_precomputed_kernel_config(runtime_config):
        cache = getattr(owner, cache_attr, None)
        if isinstance(cache, dict):
            cached = cache.get(cache_key)
            if isinstance(cached, StructuredScanKernelConfig):
                return cached
    kernel_config = autotune_structured_scan_kernel_config(
        num_states=int(num_states),
        transition_rank=int(transition_rank),
        seq_len=int(seq_len),
        device=device,
        default_chunk_size=int(default_chunk_size),
        needs_grad=bool(needs_grad),
        runtime_config=runtime_config,
    )
    if _structured_scan_runtime_allows_precomputed_kernel_config(runtime_config):
        cache = getattr(owner, cache_attr, None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(owner, cache_attr, cache)
        cache[cache_key] = kernel_config
    return kernel_config


def _preload_compiled_runtime_extensions(args: Any) -> None:
    if not bool(getattr(args, "enable_torch_compile", False)):
        return
    if USE_CAUSAL_MACHINE_CUDA_SCAN:
        load_causal_machine_scan_cuda()
    if USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN and int(getattr(args, "causal_machine_latent_rank", 0)) > 0:
        load_causal_machine_latent_scan_cuda()
    if bool(getattr(args, "use_muon", False)) and USE_MUON_CUDA:
        load_muon_cuda()


def _prime_compiled_structured_scan_kernel_configs(
    model: nn.Module,
    *,
    train_seq_len: int,
    eval_seq_len: int,
    device: torch.device,
) -> None:
    if device.type != "cuda":
        return
    seq_specs = (
        (max(int(train_seq_len), 1), True),
        (1, False),
        (max(int(eval_seq_len), 1), False),
    )
    for module in model.modules():
        if all(hasattr(module, name) for name in ("num_states", "transition_rank", "filter_chunk_size")):
            num_states = int(getattr(module, "num_states"))
            transition_rank = int(getattr(module, "transition_rank"))
            default_chunk_size = int(getattr(module, "filter_chunk_size"))
            if num_states > 0 and transition_rank > 0:
                for seq_len, needs_grad in seq_specs:
                    _get_precomputed_structured_scan_kernel_config(
                        module,
                        cache_attr="_compiled_kernel_config_cache",
                        num_states=num_states,
                        transition_rank=transition_rank,
                        seq_len=seq_len,
                        device=device,
                        default_chunk_size=default_chunk_size,
                        needs_grad=needs_grad,
                        runtime_config=None,
                    )
        if all(
            hasattr(module, name)
            for name in ("causal_machine_num_states", "causal_machine_transition_rank", "causal_machine_filter_chunk_size")
        ):
            num_states = int(getattr(module, "causal_machine_num_states"))
            transition_rank = int(getattr(module, "causal_machine_transition_rank"))
            default_chunk_size = int(getattr(module, "causal_machine_filter_chunk_size"))
            if num_states > 0 and transition_rank > 0:
                for seq_len, needs_grad in seq_specs:
                    _get_precomputed_structured_scan_kernel_config(
                        module,
                        cache_attr="_compiled_global_kernel_config_cache",
                        num_states=num_states,
                        transition_rank=transition_rank,
                        seq_len=seq_len,
                        device=device,
                        default_chunk_size=default_chunk_size,
                        needs_grad=needs_grad,
                        runtime_config=None,
                    )


def describe_structured_scan_cuda_runtime_config(
    *,
    batch_size: int = 1,
    num_states: int = _MAX_SPECIALIZED_STRUCTURED_SCAN_NUM_STATES,
    seq_len: int,
    transition_rank: int,
    chunk_size: int,
    device: torch.device,
    backward: bool = False,
) -> dict[str, object]:
    if device.type != "cuda":
        raise ValueError("describe_structured_scan_cuda_runtime_config requires a CUDA device")
    ext = load_causal_machine_scan_cuda()
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    info = dict(
        ext.describe_runtime_config(
            int(batch_size),
            int(seq_len),
            int(num_states),
            int(transition_rank),
            int(chunk_size),
            device_index,
            bool(backward),
        )
    )
    info["device_runtime"] = _describe_structured_scan_device_runtime(device)
    return info


def _canonicalize_structured_seq_lens(
    seq_lens: Tensor | Sequence[int] | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tensor | None:
    if seq_lens is None:
        return None
    if isinstance(seq_lens, Tensor):
        seq_lens_t = seq_lens.to(device=device, dtype=torch.int64).reshape(-1)
    else:
        seq_lens_t = torch.tensor(list(seq_lens), device=device, dtype=torch.int64)
    if seq_lens_t.numel() != batch_size:
        raise ValueError(f"seq_lens must have {batch_size} elements, got {seq_lens_t.numel()}")
    return seq_lens_t.clamp_(min=0, max=max(int(seq_len), 0))


def _canonicalize_structured_score_bias(
    score_bias: Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    num_states: int,
    device: torch.device,
) -> Tensor | None:
    if score_bias is None:
        return None
    bias = score_bias.to(device=device, dtype=torch.float32)
    if bias.dim() == 1:
        if bias.size(0) != num_states:
            raise ValueError(f"score_bias must have shape [{num_states}], got {tuple(bias.shape)}")
        return bias.view(1, 1, num_states).expand(batch_size, seq_len, num_states).contiguous()
    if bias.dim() == 2:
        if bias.size(0) != seq_len or bias.size(1) != num_states:
            raise ValueError(f"score_bias must have shape [{seq_len}, {num_states}], got {tuple(bias.shape)}")
        return bias.view(1, seq_len, num_states).expand(batch_size, seq_len, num_states).contiguous()
    if bias.dim() == 3:
        if bias.size(2) != num_states:
            raise ValueError(
                f"score_bias last dim must equal num_states={num_states}, got {tuple(bias.shape)}"
            )
        if bias.size(0) not in {1, batch_size} or bias.size(1) not in {1, seq_len}:
            raise ValueError(
                f"score_bias must broadcast to [{batch_size}, {seq_len}, {num_states}], got {tuple(bias.shape)}"
            )
        return bias.expand(batch_size, seq_len, num_states).contiguous()
    raise ValueError(
        f"score_bias must be 1D [N], 2D [L, N], or 3D [B, L, N]-broadcastable, got {tuple(bias.shape)}"
    )


def _canonicalize_structured_score_mask(
    score_mask: Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    num_states: int,
    device: torch.device,
) -> Tensor | None:
    if score_mask is None:
        return None
    mask = score_mask.to(device=device, dtype=torch.bool)
    if mask.dim() == 1:
        if mask.size(0) != num_states:
            raise ValueError(f"state_mask must have shape [{num_states}], got {tuple(mask.shape)}")
        return mask.view(1, 1, num_states).expand(batch_size, seq_len, num_states).contiguous()
    if mask.dim() == 2:
        if mask.size(0) != seq_len or mask.size(1) != num_states:
            raise ValueError(f"state_mask must have shape [{seq_len}, {num_states}], got {tuple(mask.shape)}")
        return mask.view(1, seq_len, num_states).expand(batch_size, seq_len, num_states).contiguous()
    if mask.dim() == 3:
        if mask.size(2) != num_states:
            raise ValueError(
                f"state_mask last dim must equal num_states={num_states}, got {tuple(mask.shape)}"
            )
        if mask.size(0) not in {1, batch_size} or mask.size(1) not in {1, seq_len}:
            raise ValueError(
                f"state_mask must broadcast to [{batch_size}, {seq_len}, {num_states}], got {tuple(mask.shape)}"
            )
        return mask.expand(batch_size, seq_len, num_states).contiguous()
    raise ValueError(
        f"state_mask must be 1D [N], 2D [L, N], or 3D [B, L, N]-broadcastable, got {tuple(mask.shape)}"
    )


def _materialize_native_structured_score_bias(
    *,
    batch_size: int,
    seq_len: int,
    num_states: int,
    device: torch.device,
    runtime_config: StructuredScanRuntimeConfig | None,
) -> Tensor | None:
    if runtime_config is None:
        return None
    combined_bias = _canonicalize_structured_score_bias(
        runtime_config.score_bias,
        batch_size=batch_size,
        seq_len=seq_len,
        num_states=num_states,
        device=device,
    )
    score_mod_spec = _structured_runtime_score_mod_spec(runtime_config)
    if score_mod_spec is None:
        return combined_bias
    if score_mod_spec.additive_bias is not None:
        additive_bias = _canonicalize_structured_score_bias(
            score_mod_spec.additive_bias,
            batch_size=batch_size,
            seq_len=seq_len,
            num_states=num_states,
            device=device,
        )
        combined_bias = additive_bias if combined_bias is None else (combined_bias + additive_bias)
    if score_mod_spec.state_mask is not None:
        state_mask = _canonicalize_structured_score_mask(
            score_mod_spec.state_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            num_states=num_states,
            device=device,
        )
        assert state_mask is not None
        masked_bias = torch.where(
            state_mask,
            torch.zeros((), device=device, dtype=torch.float32),
            torch.full(
                (),
                float(score_mod_spec.masked_bias_value),
                device=device,
                dtype=torch.float32,
            ),
        ).expand(batch_size, seq_len, num_states)
        combined_bias = masked_bias if combined_bias is None else (combined_bias + masked_bias)
    return combined_bias.contiguous() if combined_bias is not None else None


def _resolve_native_structured_score_pred_scale(
    runtime_config: StructuredScanRuntimeConfig | None,
) -> float:
    score_mod_spec = _structured_runtime_score_mod_spec(runtime_config)
    if score_mod_spec is None or score_mod_spec.pred_scale is None:
        return 1.0
    pred_scale = score_mod_spec.pred_scale
    if isinstance(pred_scale, Tensor):
        if pred_scale.numel() != 1:
            raise ValueError("StructuredScoreModSpec.pred_scale tensor must be scalar")
        scale_value = float(pred_scale.detach().cpu().item())
    else:
        scale_value = float(pred_scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("StructuredScoreModSpec.pred_scale must be a finite positive scalar")
    return scale_value


def _resolve_native_structured_score_threshold(
    runtime_config: StructuredScanRuntimeConfig | None,
) -> float:
    score_mod_spec = _structured_runtime_score_mod_spec(runtime_config)
    if score_mod_spec is None or score_mod_spec.threshold is None:
        return -math.inf
    threshold = score_mod_spec.threshold
    if isinstance(threshold, Tensor):
        if threshold.numel() != 1:
            raise ValueError("StructuredScoreModSpec.threshold tensor must be scalar")
        threshold_value = float(threshold.detach().cpu().item())
    else:
        threshold_value = float(threshold)
    if math.isnan(threshold_value):
        raise ValueError("StructuredScoreModSpec.threshold must not be NaN")
    return threshold_value


def _resolve_native_structured_score_topk(
    runtime_config: StructuredScanRuntimeConfig | None,
    *,
    num_states: int | None = None,
) -> int:
    score_mod_spec = _structured_runtime_score_mod_spec(runtime_config)
    if score_mod_spec is None or score_mod_spec.topk is None:
        return 0
    topk_value = int(score_mod_spec.topk)
    if topk_value <= 0:
        raise ValueError("StructuredScoreModSpec.topk must be a positive integer")
    if num_states is not None and topk_value >= int(num_states):
        return 0
    return topk_value


def _resolve_native_structured_score_clamp_bounds(
    runtime_config: StructuredScanRuntimeConfig | None,
) -> tuple[float, float]:
    score_mod_spec = _structured_runtime_score_mod_spec(runtime_config)
    if score_mod_spec is None:
        return -math.inf, math.inf
    clamp_min = float(score_mod_spec.clamp_min) if score_mod_spec.clamp_min is not None else -math.inf
    clamp_max = float(score_mod_spec.clamp_max) if score_mod_spec.clamp_max is not None else math.inf
    if math.isnan(clamp_min) or math.isnan(clamp_max):
        raise ValueError("StructuredScoreModSpec clamp bounds must not be NaN")
    if clamp_min > clamp_max:
        raise ValueError("StructuredScoreModSpec.clamp_min must be <= clamp_max")
    return clamp_min, clamp_max


def _has_native_structured_score_clamp(
    runtime_config: StructuredScanRuntimeConfig | None,
) -> bool:
    clamp_min, clamp_max = _resolve_native_structured_score_clamp_bounds(runtime_config)
    return math.isfinite(clamp_min) or math.isfinite(clamp_max)


def _has_native_structured_score_filtering(
    runtime_config: StructuredScanRuntimeConfig | None,
    *,
    num_states: int | None = None,
) -> bool:
    threshold = _resolve_native_structured_score_threshold(runtime_config)
    topk = _resolve_native_structured_score_topk(runtime_config, num_states=num_states)
    return math.isfinite(threshold) or topk > 0


def _apply_native_structured_score_mod_inputs(
    transition_context: Tensor,
    transition_gate: Tensor,
    runtime_config: StructuredScanRuntimeConfig | None,
) -> tuple[Tensor, Tensor]:
    score_bias = _materialize_native_structured_score_bias(
        batch_size=int(transition_context.size(0)),
        seq_len=int(transition_context.size(1)),
        num_states=int(transition_context.size(2)),
        device=transition_context.device,
        runtime_config=runtime_config,
    )
    pred_scale = _resolve_native_structured_score_pred_scale(runtime_config)
    context_out = transition_context.float()
    if score_bias is not None:
        context_out = context_out + score_bias
    gate_out = transition_gate.to(device=transition_context.device, dtype=torch.float32)
    if pred_scale != 1.0:
        context_out = context_out / pred_scale
        gate_out = gate_out * pred_scale
    return context_out.to(dtype=transition_context.dtype), gate_out


def _apply_native_structured_score_bias(
    transition_context: Tensor,
    runtime_config: StructuredScanRuntimeConfig | None,
) -> Tensor:
    score_bias = _materialize_native_structured_score_bias(
        batch_size=int(transition_context.size(0)),
        seq_len=int(transition_context.size(1)),
        num_states=int(transition_context.size(2)),
        device=transition_context.device,
        runtime_config=runtime_config,
    )
    if score_bias is None:
        return transition_context
    return (transition_context.float() + score_bias).to(dtype=transition_context.dtype)


def _build_structured_transition_mask(
    num_states: int,
    *,
    device: torch.device,
    runtime_config: StructuredScanRuntimeConfig | None,
) -> Tensor | None:
    if runtime_config is None:
        return None
    row_idx = torch.arange(num_states, device=device, dtype=torch.int64).view(num_states, 1)
    col_idx = torch.arange(num_states, device=device, dtype=torch.int64).view(1, num_states)
    mask: Tensor | None = None
    if runtime_config.transition_mask is not None:
        mask = runtime_config.transition_mask.to(device=device, dtype=torch.bool)
        if mask.shape != (num_states, num_states):
            raise ValueError(
                f"transition_mask must have shape {(num_states, num_states)}, got {tuple(mask.shape)}"
            )
    if runtime_config.local_transition_window is not None:
        local_mask = (row_idx - col_idx).abs() <= int(runtime_config.local_transition_window)
        mask = local_mask if mask is None else (mask & local_mask)
    if runtime_config.block_mask is not None:
        block_mask = runtime_config.block_mask.to(device=device, dtype=torch.bool)
        if block_mask.dim() != 2:
            raise ValueError(f"block_mask must be 2D, got shape={tuple(block_mask.shape)}")
        block_rows, block_cols = block_mask.shape
        if block_rows != block_cols:
            raise ValueError(f"block_mask must be square, got shape={tuple(block_mask.shape)}")
        block_size = int(runtime_config.block_size or math.ceil(num_states / max(block_rows, 1)))
        row_block = torch.div(row_idx, block_size, rounding_mode="floor").clamp(max=block_rows - 1)
        col_block = torch.div(col_idx, block_size, rounding_mode="floor").clamp(max=block_cols - 1)
        block_sparse_mask = block_mask[row_block, col_block]
        mask = block_sparse_mask if mask is None else (mask & block_sparse_mask)
    if runtime_config.transition_mask_mod is not None:
        mod_mask = runtime_config.transition_mask_mod(row_idx.expand(-1, num_states), col_idx.expand(num_states, -1), runtime_config)
        mod_mask = mod_mask.to(device=device, dtype=torch.bool)
        if mod_mask.shape != (num_states, num_states):
            raise ValueError(
                f"transition_mask_mod must return shape {(num_states, num_states)}, got {tuple(mod_mask.shape)}"
            )
        mask = mod_mask if mask is None else (mask & mod_mask)
    return mask


def _evaluate_structured_transition_mask_mod(
    num_states: int,
    *,
    device: torch.device,
    runtime_config: StructuredScanRuntimeConfig | None,
) -> Tensor | None:
    if runtime_config is None or runtime_config.transition_mask_mod is None:
        return None
    row_idx = torch.arange(num_states, device=device, dtype=torch.int64).view(num_states, 1)
    col_idx = torch.arange(num_states, device=device, dtype=torch.int64).view(1, num_states)
    mod_mask = runtime_config.transition_mask_mod(
        row_idx.expand(-1, num_states),
        col_idx.expand(num_states, -1),
        runtime_config,
    )
    mod_mask = mod_mask.to(device=device, dtype=torch.bool)
    if mod_mask.shape != (num_states, num_states):
        raise ValueError(
            f"transition_mask_mod must return shape {(num_states, num_states)}, got {tuple(mod_mask.shape)}"
        )
    return mod_mask.contiguous()


def _build_structured_transition_matrix(
    transition_source_probs: Tensor,
    transition_dest_probs: Tensor,
    *,
    runtime_config: StructuredScanRuntimeConfig | None,
) -> Tensor | None:
    mask = _build_structured_transition_mask(
        int(transition_source_probs.size(0)),
        device=transition_source_probs.device,
        runtime_config=runtime_config,
    )
    if mask is None:
        return None
    raw_transition_matrix = transition_source_probs.float() @ transition_dest_probs.float()
    masked_transition_matrix = raw_transition_matrix.masked_fill(~mask, 0.0)
    denom = masked_transition_matrix.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    normalized = masked_transition_matrix / denom
    if runtime_config is None or runtime_config.scratch_arena is None:
        return normalized
    arena = runtime_config.scratch_arena
    scratch = arena.empty(
        "dense_transition_matrix",
        tuple(int(dim) for dim in normalized.shape),
        device=normalized.device,
        dtype=normalized.dtype,
    )
    scratch.copy_(normalized)
    return scratch


def structured_transition_predict_log_belief_dense(
    prev_log_belief: Tensor,
    transition_matrix: Tensor,
    transition_stay_probs: Tensor,
) -> Tensor:
    prev_probs = prev_log_belief.float().exp()
    mix_probs = prev_probs @ transition_matrix.float()
    stay_probs = transition_stay_probs.float().unsqueeze(0)
    pred_probs = stay_probs * prev_probs + (1.0 - stay_probs) * mix_probs
    return pred_probs.clamp_min(1.0e-20).log()


def _apply_structured_score_mod(
    pred_log_belief: Tensor,
    *,
    time_idx: int,
    runtime_config: StructuredScanRuntimeConfig | None,
) -> Tensor:
    score_mod = _structured_runtime_score_mod_callback(runtime_config)
    if score_mod is None:
        return pred_log_belief
    assert runtime_config is not None
    updated = score_mod(pred_log_belief, int(time_idx), runtime_config)
    if not isinstance(updated, Tensor):
        raise TypeError("score_mod must return a Tensor")
    if updated.shape != pred_log_belief.shape:
        raise ValueError(
            f"score_mod must preserve shape {tuple(pred_log_belief.shape)}, got {tuple(updated.shape)}"
        )
    return updated


@dataclass
class AttentionStepCache:
    k: Tensor | None = None
    v: Tensor | None = None
    max_len: int | None = None

    def reset(self) -> None:
        self.k = None
        self.v = None


@dataclass
class BlockStepCache:
    attention_cache: AttentionStepCache | None = None
    state_cache: CausalMachineCache | None = None

    def reset(self) -> None:
        if self.attention_cache is not None:
            self.attention_cache.reset()
        if self.state_cache is not None:
            self.state_cache.reset()


@dataclass
class BackboneStepCache:
    layers: list[BlockStepCache]
    position: int = 0

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self.position = 0

# Keep local proxy specialist-lane knobs in one place so the frontier and
# promotion recipes do not drift when we tweak early-training behavior.
LOCAL_PROXY_RECIPE_COMMON: dict[str, object] = {
    "train_seq_len": 1024,
    "eval_seq_len": 2048,
    "num_layers": 10,
    "num_kv_heads": 4,
    "mlp_hidden": 896,
    "use_causal_machine_bias": False,
    "use_causal_machine_output_bias": False,
    "use_causal_machine_backbone": True,
    "block_pattern": "attn,attn,attn,attn,ssm,ssm,ssm,ssm,ssm,ssm",
    "causal_machine_num_states": 128,
    "causal_machine_hidden_rank": 64,
    "causal_machine_transition_rank": 8,
    "causal_machine_latent_rank": 16,
    "causal_machine_latent_mode": "replace",
    "causal_machine_scale_init": 0.35,
    "causal_machine_gate_init": -1.5,
    "causal_machine_latent_gate_init": -0.5,
    "causal_machine_latent_decay_init": 0.995,
    "causal_machine_teacher_loss_coeff": 0.0,
    # Competition-safe default: keep extra state-space objectives off unless
    # a launch explicitly opts into them. Profile-backed teacher supervision is
    # not allowed by default, and online-teacher wiring should not inherit
    # hidden nonzero losses from this recipe.
    "causal_machine_state_loss_coeff": 0.0,
    "causal_machine_next_token_loss_coeff": 0.0,
    "causal_machine_transition_kl_coeff": 0.0,
    "causal_machine_future_sketch_loss_coeff": 0.0,
    "causal_machine_transition_gate_init": -0.25,
    "causal_machine_transition_stickiness_init": 2.5,
    "causal_machine_emit_delta_scale_init": 0.10,
    "orthogonal_init": True,
    "mup_proj_init": True,
    "overtone_embed_init": True,
    "resid_mix_phase_init": True,
    "matrix_lr": 0.0623,
    "other_matrix_lr_mult": 0.5,
    "scalar_lr": 0.035,
    "tied_embed_lr": 0.004,
    "force_fp16_tied_embed_export": True,
    "shared_tail_output_gate": False,
    "shared_tail_enable_step": 0,
    "shared_tail_ramp_steps": 400,
    "shared_tail_max_mult": 1.0,
    "use_adaptive_rmsnorm": True,
    "adaptive_rmsnorm_gate_init": -2.0,
    "num_shared_layers": 0,
    "shared_layer_repeats": 0,
    "mid_aux_loss_coeff": 0.0,
    "mid_aux_enable_step": 0,
    "mid_aux_ramp_steps": 120,
    "mid_aux_decay_start_step": 280,
    "mid_aux_decay_end_step": 960,
    "mlp_matrix_lr_mult": 0.575,
    "token_weight_decay": 0.005,
    "head_weight_decay": 0.0,
}

DEFAULT_CAUSAL_MACHINE_SKETCH_HORIZONS: tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_CAUSAL_MACHINE_SKETCH_DIM = 16

    # Keep local proxy launch presets aligned with the shared recipe defaults and
    # enable a cheap early validation window so we can inspect the real trajectory
    # in steps 0-10 without mutating the configured full validation budget.
LOCAL_PROXY_PRESET_COMMON: dict[str, object] = {
    "curriculum_policy": "winner_int6",
    "optimizer_policy": "local_proxy_faststart",
    "eval_policy": "sliding64_2048",
    "runtime_policy": "compiled",
    "lr_warmup_steps": 100,
    "lr_warmup_init_scale": 0.08,
    "lr_warmup_power": 1.75,
    "tied_embed_init_std": 0.0075,
    "tied_embed_warmup_mult": 0.2,
    "tied_embed_warmup_steps": 80,
    "early_val_max_seqs": 256,
    "val_loss_every": 100,
}


def default_block_pattern(num_layers: int) -> str:
    if num_layers == 10:
        return "attn,attn,attn,attn,ssm,ssm,ssm,ssm,ssm,ssm"
    if num_layers <= 0:
        return ""
    front_attn = min(num_layers, max(1, 4 if num_layers >= 8 else num_layers // 2))
    return ",".join("attn" if idx < front_attn else "ssm" for idx in range(num_layers))


def normalize_block_pattern_spec(spec: str, num_layers: int) -> str:
    raw = str(spec or "").strip().lower()
    if not raw:
        return ""
    tokens = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
    aliases = {
        "a": "attn",
        "attn": "attn",
        "attention": "attn",
        "sa": "attn",
        "s": "ssm",
        "ssm": "ssm",
        "state": "ssm",
        "state_space": "ssm",
        "statespace": "ssm",
        "cm": "ssm",
        "causal_machine": "ssm",
    }
    normalized = [aliases.get(token) for token in tokens]
    if any(token is None for token in normalized):
        bad = next(tokens[idx] for idx, token in enumerate(normalized) if token is None)
        raise ValueError(f"BLOCK_PATTERN token {bad!r} is invalid; expected attn or ssm")
    if num_layers > 0 and len(normalized) != num_layers:
        raise ValueError(
            f"BLOCK_PATTERN must provide exactly NUM_LAYERS={num_layers} entries, got {len(normalized)}"
        )
    return ",".join(normalized)

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", DEFAULT_TOKENIZER_PATH)
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 100))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))
    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 2000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 0))
    lr_warmup_steps = int(os.environ.get("LR_WARMUP_STEPS", "0"))
    lr_warmup_init_scale = float(os.environ.get("LR_WARMUP_INIT_SCALE", "0.2"))
    lr_warmup_power = float(os.environ.get("LR_WARMUP_POWER", "1.5"))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", "2048"))
    competition_mode = _competition_mode_enabled()
    training_preset = os.environ.get("TRAINING_PRESET", "local_proxy_promotion").strip().lower()
    recipe_family = os.environ.get("RECIPE_FAMILY", "").strip().lower()
    curriculum_policy = os.environ.get("CURRICULUM_POLICY", "").strip().lower()
    data_policy = os.environ.get("DATA_POLICY", "").strip().lower()
    optimizer_policy = os.environ.get("OPTIMIZER_POLICY", "local_proxy_faststart").strip().lower()
    precision_policy = os.environ.get("PRECISION_POLICY", "").strip().lower()
    eval_policy = os.environ.get("EVAL_POLICY", "").strip().lower()
    runtime_policy = os.environ.get("RUNTIME_POLICY", "compiled").strip().lower()
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    wallclock_finalization_reserve_ms = float(
        os.environ.get("WALLCLOCK_FINALIZATION_RESERVE_MS", "30000" if competition_mode else "0")
    )
    wallclock_validation_reserve_ms = float(
        os.environ.get("WALLCLOCK_VALIDATION_RESERVE_MS", "120000" if competition_mode else "0")
    )
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    rope_dims = int(os.environ.get("ROPE_DIMS", "16"))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    enable_torch_compile = bool(int(os.environ.get("ENABLE_TORCH_COMPILE", "1")))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 256))
    val_max_seqs = int(os.environ.get("VAL_MAX_SEQS", "0"))
    early_val_max_seqs = int(os.environ.get("EARLY_VAL_MAX_SEQS", "256"))
    milestone_val_max_seqs = int(os.environ.get("MILESTONE_VAL_MAX_SEQS", "256"))
    final_eval_max_seqs = int(os.environ.get("FINAL_EVAL_MAX_SEQS", "0"))
    use_strict_streaming_eval = bool(int(os.environ.get("USE_STRICT_STREAMING_EVAL", "0")))
    submission_size_limit_bytes = int(os.environ.get("SUBMISSION_SIZE_LIMIT_BYTES", 16_000_000))
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", 0.0))
    z_loss_coeff = float(os.environ.get("Z_LOSS_COEFF", 0.00005))
    row_norm_loss_coeff = float(os.environ.get("ROW_NORM_LOSS_COEFF", "0.0"))
    outlier_loss_coeff = float(os.environ.get("OUTLIER_LOSS_COEFF", "0.0"))
    outlier_loss_threshold = float(os.environ.get("OUTLIER_LOSS_THRESHOLD", "3.0"))
    logit_var_loss_coeff = float(os.environ.get("LOGIT_VAR_LOSS_COEFF", "0.0005"))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9999))
    ema_start_step = int(os.environ.get("EMA_START_STEP", 200))
    mid_aux_loss_coeff = float(
        os.environ.get("MID_AUX_LOSS_COEFF", str(float(LOCAL_PROXY_RECIPE_COMMON["mid_aux_loss_coeff"])))
    )
    force_fp16_tied_embed_export = bool(int(os.environ.get("FORCE_FP16_TIED_EMBED_EXPORT", "1")))
    int8_auto_keep_budget_bytes = int(os.environ.get("INT8_AUTO_KEEP_BUDGET_BYTES", "0"))
    int8_auto_keep_max_tensors = int(os.environ.get("INT8_AUTO_KEEP_MAX_TENSORS", "0"))
    int8_auto_keep_min_numel = int(os.environ.get("INT8_AUTO_KEEP_MIN_NUMEL", "65536"))
    int8_sensitivity_log_topk = int(os.environ.get("INT8_SENSITIVITY_LOG_TOPK", "8"))
    int8_auto_keep_row_budget_bytes = int(os.environ.get("INT8_AUTO_KEEP_ROW_BUDGET_BYTES", "0"))
    int8_auto_keep_row_group_size = int(os.environ.get("INT8_AUTO_KEEP_ROW_GROUP_SIZE", "32"))
    objective_health_train_shock_ratio = float(os.environ.get("OBJECTIVE_HEALTH_TRAIN_SHOCK_RATIO", "1.40"))
    objective_health_train_shock_window = int(os.environ.get("OBJECTIVE_HEALTH_TRAIN_SHOCK_WINDOW", "5"))
    aux_release_min_step = int(os.environ.get("AUX_RELEASE_MIN_STEP", "30"))
    aux_release_max_mid_top1_agreement = float(os.environ.get("AUX_RELEASE_MAX_MID_TOP1_AGREEMENT", "0.10"))
    aux_release_min_backbone_gain_share = float(os.environ.get("AUX_RELEASE_MIN_BACKBONE_GAIN_SHARE", "0.75"))
    aux_release_train_loss_tol = float(os.environ.get("AUX_RELEASE_TRAIN_LOSS_TOL", "1.002"))
    objective_bootstrap_steps = int(os.environ.get("OBJECTIVE_BOOTSTRAP_STEPS", "200"))
    objective_bootstrap_tokens = int(os.environ.get("OBJECTIVE_BOOTSTRAP_TOKENS", "0"))
    objective_bootstrap_train_log_every = int(os.environ.get("OBJECTIVE_BOOTSTRAP_TRAIN_LOG_EVERY", "10"))
    save_best_val_bpb_checkpoint = bool(int(os.environ.get("SAVE_BEST_VAL_BPB_CHECKPOINT", "1")))
    best_val_bpb_checkpoint_path = os.environ.get("BEST_VAL_BPB_CHECKPOINT_PATH", "best_val_bpb_model.pt").strip()
    init_model_path = os.environ.get("INIT_MODEL_PATH", "").strip()
    init_model_strict = bool(int(os.environ.get("INIT_MODEL_STRICT", "1")))
    regression_stop_min_step = int(os.environ.get("REGRESSION_STOP_MIN_STEP", "0"))
    regression_stop_patience = int(os.environ.get("REGRESSION_STOP_PATIENCE", "0"))
    regression_stop_min_delta = float(os.environ.get("REGRESSION_STOP_MIN_DELTA", "0.0"))
    train_random_offset_tokens = int(os.environ.get("TRAIN_RANDOM_OFFSET_TOKENS", "0"))
    use_packed_eval_windows = bool(int(os.environ.get("USE_PACKED_EVAL_WINDOWS", "1")))
    use_cuda_graphs = bool(int(os.environ.get("USE_CUDA_GRAPHS", "1")))
    cuda_graph_warmup_steps = int(os.environ.get("CUDA_GRAPH_WARMUP_STEPS", "0"))
    debug_static_shapes = bool(int(os.environ.get("DEBUG_STATIC_SHAPES", "0")))
    enable_cudnn_sdp = bool(int(os.environ.get("ENABLE_CUDNN_SDP", "0")))
    enable_flash_sdp = bool(int(os.environ.get("ENABLE_FLASH_SDP", "1")))
    enable_mem_efficient_sdp = bool(int(os.environ.get("ENABLE_MEM_EFFICIENT_SDP", "0")))
    enable_math_sdp = bool(int(os.environ.get("ENABLE_MATH_SDP", "1")))
    overtone_embed_init = bool(int(os.environ.get("OVERTONE_EMBED_INIT", "1")))
    overtone_embed_power = float(os.environ.get("OVERTONE_EMBED_POWER", "0.5"))
    overtone_embed_scale = float(os.environ.get("OVERTONE_EMBED_SCALE", "1.0"))
    resid_mix_phase_init = bool(int(os.environ.get("RESID_MIX_PHASE_INIT", "1")))
    resid_mix_phase_sharpness = float(os.environ.get("RESID_MIX_PHASE_SHARPNESS", "8.0"))
    resid_mix_phase_center = float(os.environ.get("RESID_MIX_PHASE_CENTER", "0.5"))
    use_adaptive_rmsnorm = bool(int(os.environ.get("USE_ADAPTIVE_RMSNORM", "0")))
    adaptive_rmsnorm_gate_init = float(os.environ.get("ADAPTIVE_RMSNORM_GATE_INIT", "-2.0"))
    fake_quant_tail_steps = int(os.environ.get("FAKE_QUANT_TAIL_STEPS", "0"))
    fake_quant_bits = int(os.environ.get("FAKE_QUANT_BITS", "6"))
    fake_quant_full_run = bool(int(os.environ.get("FAKE_QUANT_FULL_RUN", "0")))
    late_qat = bool(int(os.environ.get("LATE_QAT", "1")))
    qat_threshold = float(os.environ.get("QAT_THRESHOLD", "0.15"))
    warmup_prior_init = bool(int(os.environ.get("WARMUP_PRIOR_INIT", "0")))
    warmup_prior_init_mode = os.environ.get("WARMUP_PRIOR_INIT_MODE", "bounded_delta").strip().lower()
    warmup_prior_init_blend = float(os.environ.get("WARMUP_PRIOR_INIT_BLEND", "0.10"))
    warmup_prior_init_delta_rms_mult = float(os.environ.get("WARMUP_PRIOR_INIT_DELTA_RMS_MULT", "0.35"))
    warmup_prior_init_delta_rms_floor = float(os.environ.get("WARMUP_PRIOR_INIT_DELTA_RMS_FLOOR", "0.0005"))
    warmup_prior_init_groups = tuple(
        item.strip()
        for item in os.environ.get("WARMUP_PRIOR_INIT_GROUPS", "other").split(",")
        if item.strip()
    )
    shared_tail_output_gate = bool(
        int(os.environ.get("SHARED_TAIL_OUTPUT_GATE", "1" if LOCAL_PROXY_RECIPE_COMMON["shared_tail_output_gate"] else "0"))
    )
    shared_tail_output_init = float(os.environ.get("SHARED_TAIL_OUTPUT_INIT", "-1.5"))
    shared_tail_enable_step = int(
        os.environ.get("SHARED_TAIL_ENABLE_STEP", str(int(LOCAL_PROXY_RECIPE_COMMON["shared_tail_enable_step"])))
    )
    shared_tail_ramp_steps = int(
        os.environ.get("SHARED_TAIL_RAMP_STEPS", str(int(LOCAL_PROXY_RECIPE_COMMON["shared_tail_ramp_steps"])))
    )
    shared_tail_max_mult = float(
        os.environ.get("SHARED_TAIL_MAX_MULT", str(float(LOCAL_PROXY_RECIPE_COMMON["shared_tail_max_mult"])))
    )
    signed_skip_weights = bool(int(os.environ.get("SIGNED_SKIP_WEIGHTS", "1")))
    mid_aux_enable_step = int(
        os.environ.get("MID_AUX_ENABLE_STEP", str(int(LOCAL_PROXY_RECIPE_COMMON["mid_aux_enable_step"])))
    )
    mid_aux_ramp_steps = int(
        os.environ.get("MID_AUX_RAMP_STEPS", str(int(LOCAL_PROXY_RECIPE_COMMON["mid_aux_ramp_steps"])))
    )
    mid_aux_decay_start_step = int(
        os.environ.get(
            "MID_AUX_DECAY_START_STEP",
            str(int(LOCAL_PROXY_RECIPE_COMMON["mid_aux_decay_start_step"])),
        )
    )
    mid_aux_decay_end_step = int(
        os.environ.get(
            "MID_AUX_DECAY_END_STEP",
            str(int(LOCAL_PROXY_RECIPE_COMMON["mid_aux_decay_end_step"])),
        )
    )
    orthogonal_init = bool(int(os.environ.get("ORTHOGONAL_INIT", "1")))
    mup_proj_init = bool(int(os.environ.get("MUP_PROJ_INIT", "1")))
    export_quant_bits = int(os.environ.get("EXPORT_QUANT_BITS", "5"))
    export_codec = os.environ.get("EXPORT_CODEC", "zstd").strip().lower()
    export_zstd_level = int(os.environ.get("EXPORT_ZSTD_LEVEL", "22"))
    export_high_precision_bits = int(os.environ.get("EXPORT_HIGH_PRECISION_BITS", "8"))
    save_raw_debug_model = bool(int(os.environ.get("SAVE_RAW_DEBUG_MODEL", "0" if competition_mode else "1")))
    run_final_quant_eval = bool(int(os.environ.get("RUN_FINAL_QUANT_EVAL", "0" if competition_mode else "1")))
    export_high_precision_budget_bytes = int(os.environ.get("EXPORT_HIGH_PRECISION_BUDGET_BYTES", "300000"))
    export_high_precision_max_tensors = int(os.environ.get("EXPORT_HIGH_PRECISION_MAX_TENSORS", "4"))
    export_high_precision_min_numel = int(os.environ.get("EXPORT_HIGH_PRECISION_MIN_NUMEL", "65536"))
    use_flash_attn_3 = bool(int(os.environ.get("USE_FLASH_ATTN_3", "1")))
    attention_kv_mode = os.environ.get("ATTENTION_KV_MODE", "").strip().lower()
    block_pattern = os.environ.get(
        "BLOCK_PATTERN",
        str(LOCAL_PROXY_RECIPE_COMMON["block_pattern"]),
    ).strip().lower()
    use_mqa = bool(int(os.environ.get("USE_MQA", "0")))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", DEFAULT_VOCAB_SIZE))
    num_layers = int(os.environ.get("NUM_LAYERS", 10))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 448))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", "896"))
    num_shared_layers = int(os.environ.get("NUM_SHARED_LAYERS", "0"))
    shared_layer_repeats = int(os.environ.get("SHARED_LAYER_REPEATS", "0"))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    use_output_logit_bias = bool(int(os.environ.get("USE_OUTPUT_LOGIT_BIAS", "0")))
    use_causal_machine_bias = bool(
        int(
            os.environ.get(
                "USE_CAUSAL_MACHINE_BIAS",
                "1" if LOCAL_PROXY_RECIPE_COMMON["use_causal_machine_bias"] else "0",
            )
        )
    )
    use_causal_machine_output_bias = bool(
        int(
            os.environ.get(
                "USE_CAUSAL_MACHINE_OUTPUT_BIAS",
                "1" if LOCAL_PROXY_RECIPE_COMMON["use_causal_machine_output_bias"] else "0",
            )
        )
    )
    use_causal_machine_backbone = bool(
        int(
            os.environ.get(
                "USE_CAUSAL_MACHINE_BACKBONE",
                "1" if LOCAL_PROXY_RECIPE_COMMON["use_causal_machine_backbone"] else "0",
            )
        )
    )
    causal_machine_profile_json = os.environ.get(
        "CAUSAL_MACHINE_PROFILE_JSON",
        DEFAULT_CAUSAL_MACHINE_PROFILE_JSON,
    ).strip()
    causal_machine_allow_offline_teacher = bool(
        int(os.environ.get("CAUSAL_MACHINE_ALLOW_OFFLINE_TEACHER", "0"))
    )
    causal_machine_hidden_rank = int(
        os.environ.get(
            "CAUSAL_MACHINE_HIDDEN_RANK",
            str(int(LOCAL_PROXY_RECIPE_COMMON["causal_machine_hidden_rank"])),
        )
    )
    causal_machine_transition_rank = int(
        os.environ.get(
            "CAUSAL_MACHINE_TRANSITION_RANK",
            str(int(LOCAL_PROXY_RECIPE_COMMON["causal_machine_transition_rank"])),
        )
    )
    causal_machine_latent_rank = int(
        os.environ.get(
            "CAUSAL_MACHINE_LATENT_RANK",
            str(int(LOCAL_PROXY_RECIPE_COMMON["causal_machine_latent_rank"])),
        )
    )
    causal_machine_latent_mode = os.environ.get(
        "CAUSAL_MACHINE_LATENT_MODE",
        str(LOCAL_PROXY_RECIPE_COMMON["causal_machine_latent_mode"]),
    ).strip().lower()
    causal_machine_num_states = int(
        os.environ.get(
            "CAUSAL_MACHINE_NUM_STATES",
            str(int(LOCAL_PROXY_RECIPE_COMMON["causal_machine_num_states"])),
        )
    )
    causal_machine_scale_init = float(
        os.environ.get(
            "CAUSAL_MACHINE_SCALE_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_scale_init"])),
        )
    )
    causal_machine_gate_init = float(
        os.environ.get(
            "CAUSAL_MACHINE_GATE_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_gate_init"])),
        )
    )
    causal_machine_latent_gate_init = float(
        os.environ.get(
            "CAUSAL_MACHINE_LATENT_GATE_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_latent_gate_init"])),
        )
    )
    causal_machine_latent_decay_init = float(
        os.environ.get(
            "CAUSAL_MACHINE_LATENT_DECAY_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_latent_decay_init"])),
        )
    )
    causal_machine_teacher_loss_coeff = float(
        os.environ.get(
            "CAUSAL_MACHINE_TEACHER_LOSS_COEFF",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_teacher_loss_coeff"])),
        )
    )
    causal_machine_state_loss_coeff = float(
        os.environ.get(
            "CAUSAL_MACHINE_STATE_LOSS_COEFF",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_state_loss_coeff"])),
        )
    )
    causal_machine_next_token_loss_coeff = float(
        os.environ.get(
            "CAUSAL_MACHINE_NEXT_TOKEN_LOSS_COEFF",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_next_token_loss_coeff"])),
        )
    )
    causal_machine_transition_kl_coeff = float(
        os.environ.get(
            "CAUSAL_MACHINE_TRANSITION_KL_COEFF",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_transition_kl_coeff"])),
        )
    )
    causal_machine_future_sketch_loss_coeff = float(
        os.environ.get(
            "CAUSAL_MACHINE_FUTURE_SKETCH_LOSS_COEFF",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_future_sketch_loss_coeff"])),
        )
    )
    causal_machine_transition_gate_init = float(
        os.environ.get(
            "CAUSAL_MACHINE_TRANSITION_GATE_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_transition_gate_init"])),
        )
    )
    causal_machine_transition_stickiness_init = float(
        os.environ.get(
            "CAUSAL_MACHINE_TRANSITION_STICKINESS_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_transition_stickiness_init"])),
        )
    )
    causal_machine_emit_delta_scale_init = float(
        os.environ.get(
            "CAUSAL_MACHINE_EMIT_DELTA_SCALE_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["causal_machine_emit_delta_scale_init"])),
        )
    )
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", str(float(LOCAL_PROXY_RECIPE_COMMON["tied_embed_lr"]))))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    tied_embed_warmup_mult = float(os.environ.get("TIED_EMBED_WARMUP_MULT", "0.3"))
    tied_embed_warmup_steps = int(os.environ.get("TIED_EMBED_WARMUP_STEPS", "40"))
    token_weight_decay = float(os.environ.get("TOKEN_WEIGHT_DECAY", str(float(LOCAL_PROXY_RECIPE_COMMON["token_weight_decay"]))))
    matrix_lr = float(os.environ.get("MATRIX_LR", str(float(LOCAL_PROXY_RECIPE_COMMON["matrix_lr"]))))
    mlp_matrix_lr_mult = float(os.environ.get("MLP_MATRIX_LR_MULT", str(float(LOCAL_PROXY_RECIPE_COMMON["mlp_matrix_lr_mult"]))))
    other_matrix_lr_mult = float(
        os.environ.get("OTHER_MATRIX_LR_MULT", str(float(LOCAL_PROXY_RECIPE_COMMON["other_matrix_lr_mult"])))
    )
    scalar_lr = float(os.environ.get("SCALAR_LR", str(float(LOCAL_PROXY_RECIPE_COMMON["scalar_lr"]))))
    use_muon = bool(int(os.environ.get("USE_MUON", "1")))
    muon_cuda_graph_mode = os.environ.get("MUON_CUDA_GRAPH_MODE", "auto").strip().lower()
    head_weight_decay = float(os.environ.get("HEAD_WEIGHT_DECAY", str(float(LOCAL_PROXY_RECIPE_COMMON["head_weight_decay"]))))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", "0.03"))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_backend_steps_light = int(os.environ.get("MUON_BACKEND_STEPS_LIGHT", "0"))
    muon_backend_refresh_interval = int(os.environ.get("MUON_BACKEND_REFRESH_INTERVAL", "1"))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    early_phase_steps = int(os.environ.get("EARLY_PHASE_STEPS", "0"))
    early_muon_attn_lr_scale = float(os.environ.get("EARLY_MUON_ATTN_LR_SCALE", "1.0"))
    early_muon_mlp_lr_scale = float(os.environ.get("EARLY_MUON_MLP_LR_SCALE", "1.0"))
    early_muon_other_lr_scale = float(os.environ.get("EARLY_MUON_OTHER_LR_SCALE", "1.0"))
    early_muon_wd_scale = float(os.environ.get("EARLY_MUON_WD_SCALE", "1.0"))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.25))

    def __init__(self):
        if self.muon_cuda_graph_mode not in {"auto", "on", "off"}:
            raise ValueError(
                f"MUON_CUDA_GRAPH_MODE={self.muon_cuda_graph_mode!r} must be one of: auto, on, off"
            )
        self._apply_policy_schema()
        self._apply_training_preset()
        self._resolve_attention_config()
        self._resolve_block_pattern_config()

    def _env_override_map(self) -> dict[str, str]:
        return {
            "train_seq_len": "TRAIN_SEQ_LEN",
            "eval_seq_len": "EVAL_SEQ_LEN",
            "eval_stride": "EVAL_STRIDE",
            "eval_batch_seqs": "EVAL_BATCH_SEQS",
            "use_strict_streaming_eval": "USE_STRICT_STREAMING_EVAL",
            "train_batch_tokens": "TRAIN_BATCH_TOKENS",
            "early_val_max_seqs": "EARLY_VAL_MAX_SEQS",
            "lr_warmup_steps": "LR_WARMUP_STEPS",
            "lr_warmup_init_scale": "LR_WARMUP_INIT_SCALE",
            "num_layers": "NUM_LAYERS",
            "mlp_hidden": "MLP_HIDDEN",
            "rope_dims": "ROPE_DIMS",
            "ln_scale": "LN_SCALE",
            "tied_embed_lr": "TIED_EMBED_LR",
            "tied_embed_warmup_mult": "TIED_EMBED_WARMUP_MULT",
            "tied_embed_warmup_steps": "TIED_EMBED_WARMUP_STEPS",
            "matrix_lr": "MATRIX_LR",
            "other_matrix_lr_mult": "OTHER_MATRIX_LR_MULT",
            "scalar_lr": "SCALAR_LR",
            "use_muon": "USE_MUON",
            "muon_cuda_graph_mode": "MUON_CUDA_GRAPH_MODE",
            "muon_momentum": "MUON_MOMENTUM",
            "muon_momentum_warmup_start": "MUON_MOMENTUM_WARMUP_START",
            "muon_momentum_warmup_steps": "MUON_MOMENTUM_WARMUP_STEPS",
            "muon_backend_steps_light": "MUON_BACKEND_STEPS_LIGHT",
            "muon_backend_refresh_interval": "MUON_BACKEND_REFRESH_INTERVAL",
            "early_phase_steps": "EARLY_PHASE_STEPS",
            "early_muon_attn_lr_scale": "EARLY_MUON_ATTN_LR_SCALE",
            "early_muon_mlp_lr_scale": "EARLY_MUON_MLP_LR_SCALE",
            "early_muon_other_lr_scale": "EARLY_MUON_OTHER_LR_SCALE",
            "early_muon_wd_scale": "EARLY_MUON_WD_SCALE",
            "warmdown_iters": "WARMDOWN_ITERS",
            "grad_clip_norm": "GRAD_CLIP_NORM",
            "muon_weight_decay": "MUON_WEIGHT_DECAY",
            "mid_aux_loss_coeff": "MID_AUX_LOSS_COEFF",
            "mid_aux_enable_step": "MID_AUX_ENABLE_STEP",
            "mid_aux_ramp_steps": "MID_AUX_RAMP_STEPS",
            "mid_aux_decay_start_step": "MID_AUX_DECAY_START_STEP",
            "mid_aux_decay_end_step": "MID_AUX_DECAY_END_STEP",
            "overtone_embed_init": "OVERTONE_EMBED_INIT",
            "resid_mix_phase_init": "RESID_MIX_PHASE_INIT",
            "resid_mix_phase_center": "RESID_MIX_PHASE_CENTER",
            "force_fp16_tied_embed_export": "FORCE_FP16_TIED_EMBED_EXPORT",
            "fake_quant_tail_steps": "FAKE_QUANT_TAIL_STEPS",
            "fake_quant_bits": "FAKE_QUANT_BITS",
            "fake_quant_full_run": "FAKE_QUANT_FULL_RUN",
            "late_qat": "LATE_QAT",
            "qat_threshold": "QAT_THRESHOLD",
            "int8_auto_keep_budget_bytes": "INT8_AUTO_KEEP_BUDGET_BYTES",
            "int8_auto_keep_max_tensors": "INT8_AUTO_KEEP_MAX_TENSORS",
            "int8_auto_keep_row_budget_bytes": "INT8_AUTO_KEEP_ROW_BUDGET_BYTES",
            "int8_auto_keep_row_group_size": "INT8_AUTO_KEEP_ROW_GROUP_SIZE",
            "comp_reference_train_tokens": "COMP_REFERENCE_TRAIN_TOKENS",
            "use_packed_eval_windows": "USE_PACKED_EVAL_WINDOWS",
            "warmup_prior_init": "WARMUP_PRIOR_INIT",
            "warmup_prior_init_mode": "WARMUP_PRIOR_INIT_MODE",
            "warmup_prior_init_blend": "WARMUP_PRIOR_INIT_BLEND",
            "warmup_prior_init_delta_rms_mult": "WARMUP_PRIOR_INIT_DELTA_RMS_MULT",
            "warmup_prior_init_delta_rms_floor": "WARMUP_PRIOR_INIT_DELTA_RMS_FLOOR",
            "warmup_prior_init_groups": "WARMUP_PRIOR_INIT_GROUPS",
            "shared_tail_output_gate": "SHARED_TAIL_OUTPUT_GATE",
            "shared_tail_output_init": "SHARED_TAIL_OUTPUT_INIT",
            "shared_tail_enable_step": "SHARED_TAIL_ENABLE_STEP",
            "shared_tail_ramp_steps": "SHARED_TAIL_RAMP_STEPS",
            "shared_tail_max_mult": "SHARED_TAIL_MAX_MULT",
            "signed_skip_weights": "SIGNED_SKIP_WEIGHTS",
            "use_causal_machine_bias": "USE_CAUSAL_MACHINE_BIAS",
            "causal_machine_profile_json": "CAUSAL_MACHINE_PROFILE_JSON",
            "causal_machine_num_states": "CAUSAL_MACHINE_NUM_STATES",
            "causal_machine_hidden_rank": "CAUSAL_MACHINE_HIDDEN_RANK",
            "causal_machine_transition_rank": "CAUSAL_MACHINE_TRANSITION_RANK",
            "causal_machine_latent_rank": "CAUSAL_MACHINE_LATENT_RANK",
            "causal_machine_latent_mode": "CAUSAL_MACHINE_LATENT_MODE",
            "causal_machine_scale_init": "CAUSAL_MACHINE_SCALE_INIT",
            "causal_machine_gate_init": "CAUSAL_MACHINE_GATE_INIT",
            "causal_machine_latent_gate_init": "CAUSAL_MACHINE_LATENT_GATE_INIT",
            "causal_machine_latent_decay_init": "CAUSAL_MACHINE_LATENT_DECAY_INIT",
            "causal_machine_teacher_loss_coeff": "CAUSAL_MACHINE_TEACHER_LOSS_COEFF",
            "causal_machine_state_loss_coeff": "CAUSAL_MACHINE_STATE_LOSS_COEFF",
            "causal_machine_next_token_loss_coeff": "CAUSAL_MACHINE_NEXT_TOKEN_LOSS_COEFF",
            "causal_machine_transition_kl_coeff": "CAUSAL_MACHINE_TRANSITION_KL_COEFF",
            "causal_machine_future_sketch_loss_coeff": "CAUSAL_MACHINE_FUTURE_SKETCH_LOSS_COEFF",
            "causal_machine_transition_gate_init": "CAUSAL_MACHINE_TRANSITION_GATE_INIT",
            "causal_machine_transition_stickiness_init": "CAUSAL_MACHINE_TRANSITION_STICKINESS_INIT",
            "causal_machine_emit_delta_scale_init": "CAUSAL_MACHINE_EMIT_DELTA_SCALE_INIT",
            "use_adaptive_rmsnorm": "USE_ADAPTIVE_RMSNORM",
            "adaptive_rmsnorm_gate_init": "ADAPTIVE_RMSNORM_GATE_INIT",
            "orthogonal_init": "ORTHOGONAL_INIT",
            "mup_proj_init": "MUP_PROJ_INIT",
            "export_quant_bits": "EXPORT_QUANT_BITS",
            "export_codec": "EXPORT_CODEC",
            "export_high_precision_bits": "EXPORT_HIGH_PRECISION_BITS",
            "export_high_precision_budget_bytes": "EXPORT_HIGH_PRECISION_BUDGET_BYTES",
            "export_high_precision_max_tensors": "EXPORT_HIGH_PRECISION_MAX_TENSORS",
            "export_high_precision_min_numel": "EXPORT_HIGH_PRECISION_MIN_NUMEL",
            "train_random_offset_tokens": "TRAIN_RANDOM_OFFSET_TOKENS",
            "use_cuda_graphs": "USE_CUDA_GRAPHS",
            "cuda_graph_warmup_steps": "CUDA_GRAPH_WARMUP_STEPS",
            "debug_static_shapes": "DEBUG_STATIC_SHAPES",
            "attention_kv_mode": "ATTENTION_KV_MODE",
            "block_pattern": "BLOCK_PATTERN",
            "use_mqa": "USE_MQA",
            "use_causal_machine_output_bias": "USE_CAUSAL_MACHINE_OUTPUT_BIAS",
            "use_causal_machine_backbone": "USE_CAUSAL_MACHINE_BACKBONE",
            "init_model_path": "INIT_MODEL_PATH",
            "init_model_strict": "INIT_MODEL_STRICT",
        }

    def _apply_schema_values(self, values: dict[str, object]) -> None:
        env_overrides = self._env_override_map()
        for attr, value in values.items():
            env_name = env_overrides.get(attr)
            if env_name is not None and env_name in os.environ:
                continue
            setattr(self, attr, value)

    def _apply_policy_schema(self) -> None:
        recipe_families: dict[str, dict[str, object]] = {
            "local_proxy_frontier": {
                **LOCAL_PROXY_RECIPE_COMMON,
                "ema_decay": 0.0,
                "ema_start_step": 0,
            },
            "local_proxy_frontier_promotion": dict(LOCAL_PROXY_RECIPE_COMMON),
        }
        curriculum_policies: dict[str, dict[str, object]] = {
            "winner_int6": {"warmdown_iters": 3000, "grad_clip_norm": 0.3},
        }
        data_policies: dict[str, dict[str, object]] = {
            "sequential": {},
        }
        precision_policies: dict[str, dict[str, object]] = {
            "export_plus": {
                "export_quant_bits": 5,
                "fake_quant_tail_steps": 1024,
                "fake_quant_bits": 8,
                "int8_auto_keep_budget_bytes": 32768,
                "int8_auto_keep_max_tensors": 2,
                "int8_auto_keep_row_budget_bytes": 131072,
                "int8_auto_keep_row_group_size": 32,
            },
        }
        eval_policies: dict[str, dict[str, object]] = {
            "standard": {"eval_stride": 0},
            "sliding64_2048": {"eval_stride": 64, "eval_batch_seqs": 256},
        }
        optimizer_policies: dict[str, dict[str, object]] = {
            "local_proxy_faststart": {
                "muon_momentum": 0.96,
                "muon_momentum_warmup_start": 0.80,
                "muon_momentum_warmup_steps": 250,
                "muon_weight_decay": 0.03,
                "other_matrix_lr_mult": 0.5,
                "early_phase_steps": 600,
                "early_muon_other_lr_scale": 0.5,
            },
            "graph_first": {
                "use_muon": False,
            },
        }
        runtime_policies: dict[str, dict[str, object]] = {
            "compiled": {"enable_torch_compile": True},
            "eager": {"enable_torch_compile": False},
        }

        for value, table, label in (
            (self.recipe_family, recipe_families, "RECIPE_FAMILY"),
            (self.curriculum_policy, curriculum_policies, "CURRICULUM_POLICY"),
            (self.data_policy, data_policies, "DATA_POLICY"),
            (self.optimizer_policy, optimizer_policies, "OPTIMIZER_POLICY"),
            (self.precision_policy, precision_policies, "PRECISION_POLICY"),
            (self.eval_policy, eval_policies, "EVAL_POLICY"),
            (self.runtime_policy, runtime_policies, "RUNTIME_POLICY"),
        ):
            if not value:
                continue
            if value not in table:
                valid = ", ".join(sorted(table))
                raise ValueError(f"Unknown {label}={value!r}; expected one of: {valid}")
            self._apply_schema_values(table[value])

    def _apply_training_preset(self) -> None:
        if not self.training_preset:
            return
        preset_aliases: dict[str, dict[str, str]] = {
            "local_proxy_frontier": {
                **LOCAL_PROXY_PRESET_COMMON,
                "recipe_family": "local_proxy_frontier",
                "fake_quant_tail_steps": 640,
                "fake_quant_bits": 6,
                "fake_quant_full_run": False,
            },
            "local_proxy_promotion": {
                **LOCAL_PROXY_PRESET_COMMON,
                "recipe_family": "local_proxy_frontier_promotion",
                "data_policy": "sequential",
                "precision_policy": "export_plus",
            },
        }
        if self.training_preset not in preset_aliases:
            valid = ", ".join(sorted(preset_aliases))
            raise ValueError(f"Unknown TRAINING_PRESET={self.training_preset!r}; expected one of: {valid}")
        # Explicit schema env vars win over legacy preset aliases.
        for attr, value in preset_aliases[self.training_preset].items():
            env_name = attr.upper()
            if env_name not in os.environ and not getattr(self, attr):
                setattr(self, attr, value)
        self._apply_policy_schema()

    def _resolve_attention_config(self) -> None:
        kv_mode = str(self.attention_kv_mode or "").strip().lower()
        if self.use_mqa:
            kv_mode = "mqa"
        if not kv_mode:
            if int(self.num_kv_heads) <= 1:
                kv_mode = "mqa"
            elif int(self.num_kv_heads) >= int(self.num_heads):
                kv_mode = "mha"
            else:
                kv_mode = "gqa"
        if kv_mode == "mqa":
            self.num_kv_heads = 1
        elif kv_mode == "mha":
            self.num_kv_heads = int(self.num_heads)
        elif kv_mode == "gqa":
            self.num_kv_heads = max(min(int(self.num_kv_heads), int(self.num_heads)), 1)
        else:
            raise ValueError(f"ATTENTION_KV_MODE={kv_mode!r} must be one of: mha, gqa, mqa")
        if int(self.num_heads) % int(self.num_kv_heads) != 0:
            raise ValueError(
                f"Resolved KV mode {kv_mode!r} is incompatible with NUM_HEADS={self.num_heads} "
                f"and NUM_KV_HEADS={self.num_kv_heads}"
            )
        self.attention_kv_mode = kv_mode

    def _resolve_block_pattern_config(self) -> None:
        if "USE_CAUSAL_MACHINE_OUTPUT_BIAS" not in os.environ:
            self.use_causal_machine_output_bias = bool(self.use_causal_machine_bias)
        normalized = normalize_block_pattern_spec(self.block_pattern, int(self.num_layers))
        if normalized:
            self.block_pattern = normalized
            if "ssm" in normalized.split(","):
                self.use_causal_machine_backbone = True
        elif self.use_causal_machine_backbone:
            self.block_pattern = default_block_pattern(int(self.num_layers))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def _muon_python_step_param(
    p: Tensor,
    g: Tensor,
    state: dict[str, Tensor],
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    backend_steps: int,
) -> None:
    if weight_decay > 0.0:
        p.mul_(1.0 - lr * weight_decay)
    buf = state.get("momentum_buffer")
    if buf is None:
        buf = torch.zeros_like(g)
        state["momentum_buffer"] = buf
    buf.mul_(momentum).add_(g)
    effective_g = g.add(buf, alpha=momentum) if nesterov else buf
    update = zeropower_via_newtonschulz5(effective_g, steps=backend_steps)
    update *= max(1, update.size(0) / update.size(1)) ** 0.5
    p.add_(update.to(dtype=p.dtype), alpha=-lr)


def _muon_prefers_cuda_bucket(
    shape: tuple[int, int],
    *,
    bucket_size: int,
) -> bool:
    policy = MUON_CUDA_BUCKET_POLICY
    if policy in {"0", "false", "off", "none"}:
        return False
    if policy in {"1", "true", "on", "all", "force"}:
        return True
    rows, cols = int(shape[0]), int(shape[1])
    elements = rows * cols
    bucket_elements = elements * int(bucket_size)
    if bucket_size < 4:
        return False
    # Large square attention buckets now win on CUDA, but the small square
    # family still tends to lose to launch and gather overhead.
    if rows == cols:
        return bucket_size >= 8 or bucket_elements >= 1_000_000
    return rows < cols or elements >= 524288


def _muon_prefers_cuda_bucket_for_graph_capture(
    shape: tuple[int, int],
    *,
    bucket_size: int,
) -> bool:
    if _muon_prefers_cuda_bucket(shape, bucket_size=bucket_size):
        return True
    rows, cols = int(shape[0]), int(shape[1])
    elements = rows * cols
    bucket_elements = elements * int(bucket_size)
    if bucket_size < 2:
        return False
    if rows == cols:
        return bucket_size >= 4 and bucket_elements >= 262144
    if rows > cols:
        return bucket_elements >= 262144
    return bucket_elements >= 131072


def _muon_bucket_family_code(shape: tuple[int, int]) -> int:
    rows, cols = int(shape[0]), int(shape[1])
    if rows == cols:
        return 0
    if rows > cols:
        return 1
    if rows <= 384:
        return 2
    return 3


def _make_cuda_pointer_tensor(tensors: list[Tensor], device: torch.device) -> Tensor:
    ptrs_cpu = torch.tensor([int(t.data_ptr()) for t in tensors], dtype=torch.int64, device="cpu")
    return ptrs_cpu.to(device=device, dtype=torch.int64, non_blocking=False)


def _muon_square_backend_name(code: int) -> str:
    mapping = {
        1: "cublas",
        2: "cublaslt",
        3: "hybrid",
    }
    return mapping.get(int(code), "cublas")


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        backend_steps_light: int = 0,
        backend_refresh_interval: int = 1,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        capturable: bool = False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                backend_steps_light=backend_steps_light,
                backend_refresh_interval=backend_refresh_interval,
                weight_decay=weight_decay,
                nesterov=nesterov,
            ),
        )
        self.last_step_stats: dict[str, Any] = {}
        self._cuda_bucket_workspaces: dict[tuple[int, ...], dict[str, Any]] = {}
        self._cuda_ext: Any | None = None
        self._capturable_requested = bool(capturable)
        self._graph_capture_ready = False
        self._graph_capture_disable_reason = ""
        self._static_group_buckets: dict[int, list[dict[str, Any]]] = {}
        if self._capturable_requested:
            self._init_capturable_group_tensors()

    def _init_capturable_group_tensors(self) -> None:
        for group in self.param_groups:
            params = list(group.get("params", []))
            first_param = next((p for p in params if isinstance(p, Tensor)), None)
            if first_param is None or first_param.device.type != "cuda":
                continue
            device = first_param.device
            if not isinstance(group.get("lr_tensor"), Tensor):
                lr_tensor = torch.tensor(float(group["lr"]), device=device, dtype=torch.float32)
                group["lr_tensor"] = lr_tensor
                group["lr"] = lr_tensor
            if not isinstance(group.get("momentum_tensor"), Tensor):
                group["momentum_tensor"] = torch.tensor(float(group["momentum"]), device=device, dtype=torch.float32)
            if not isinstance(group.get("weight_decay_tensor"), Tensor):
                group["weight_decay_tensor"] = torch.tensor(float(group["weight_decay"]), device=device, dtype=torch.float32)

    def _group_tensor_or_float(self, group: dict[str, Any], name: str, fallback: float = 0.0) -> float | Tensor:
        tensor_name = f"{name}_tensor"
        value = group.get(tensor_name)
        if isinstance(value, Tensor):
            return value
        return group.get(name, fallback)

    def _load_cuda_ext_or_none(self):
        if self._cuda_ext is not None:
            return self._cuda_ext
        try:
            self._cuda_ext = load_muon_cuda()
        except Exception:
            self._cuda_ext = None
        return self._cuda_ext

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._graph_capture_ready and set_to_none:
            set_to_none = False
        super().zero_grad(set_to_none=set_to_none)

    def supports_full_step_cuda_graph(self) -> bool:
        return bool(self._graph_capture_ready)

    def graph_capture_disable_reason(self) -> str:
        return str(self._graph_capture_disable_reason)

    def prepare_cuda_graph_capture(self) -> bool:
        self._graph_capture_ready = False
        self._graph_capture_disable_reason = ""
        self._static_group_buckets = {}
        if not self._capturable_requested:
            self._graph_capture_disable_reason = "capturable_disabled"
            return False
        ext = self._load_cuda_ext_or_none()
        if ext is None or not hasattr(ext, "grouped_step_family_workspace_capturable"):
            self._graph_capture_disable_reason = "capturable_extension_unavailable"
            return False
        for group_idx, group in enumerate(self.param_groups):
            backend_refresh_interval = max(int(group.get("backend_refresh_interval", 1)), 1)
            backend_steps = int(group.get("backend_steps", 0))
            backend_steps_light = int(group.get("backend_steps_light", 0))
            if backend_refresh_interval > 1 and backend_steps != backend_steps_light:
                self._graph_capture_disable_reason = "dynamic_backend_steps"
                self._static_group_buckets = {}
                return False
            bucket_map: dict[tuple[tuple[int, int], torch.dtype, torch.device], list[tuple[Tensor, Tensor, dict[str, Tensor]]]] = {}
            for p in group["params"]:
                if p.ndim != 2:
                    continue
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(p, dtype=torch.float32 if p.dtype != torch.float32 else p.dtype)
                    state["momentum_buffer"] = buf
                grad = p.grad
                if grad is None:
                    grad = torch.zeros_like(p)
                    p.grad = grad
                if (
                    p.device.type != "cuda"
                    or grad.device.type != "cuda"
                    or not p.is_contiguous()
                    or not grad.is_contiguous()
                    or not buf.is_contiguous()
                ):
                    self._graph_capture_disable_reason = "non_cuda_or_noncontiguous_param"
                    self._static_group_buckets = {}
                    return False
                bucket_key = (tuple(int(v) for v in p.shape), p.dtype, p.device)
                bucket_map.setdefault(bucket_key, []).append((p, grad, state))
            group_buckets: list[dict[str, Any]] = []
            for (shape, _dtype, _device), items in bucket_map.items():
                if not _muon_prefers_cuda_bucket_for_graph_capture(shape, bucket_size=len(items)):
                    self._graph_capture_disable_reason = "python_fallback_bucket"
                    self._static_group_buckets = {}
                    return False
                params_bucket = [item[0] for item in items]
                grads_bucket = [item[1] for item in items]
                states_bucket = [item[2] for item in items]
                workspace = self._get_cuda_bucket_workspace(items)
                group_buckets.append(
                    {
                        "params": params_bucket,
                        "grads": grads_bucket,
                        "states": states_bucket,
                        "family_code": int(workspace["family_code"]),
                        "square_backend": str(workspace.get("square_backend", "")),
                        "workspace": workspace,
                        "shape": tuple(int(v) for v in shape),
                    }
                )
            self._static_group_buckets[group_idx] = group_buckets
        self._graph_capture_ready = True
        return True

    def _get_cuda_bucket_workspace(
        self,
        bucket_items: list[tuple[Tensor, Tensor, dict[str, Tensor]]],
    ) -> dict[str, Any]:
        params_bucket = [item[0] for item in bucket_items]
        states_bucket = [item[2] for item in bucket_items]
        shape = tuple(int(v) for v in params_bucket[0].shape)
        device = params_bucket[0].device
        key = (
            device.index if device.index is not None else -1,
            params_bucket[0].dtype,
            len(bucket_items),
            shape[0],
            shape[1],
            *(id(param) for param in params_bucket),
        )
        workspace = self._cuda_bucket_workspaces.get(key)
        grads_bucket = [item[1] for item in bucket_items]
        if workspace is None:
            family_code = _muon_bucket_family_code(shape)
            transpose_input = family_code == 1
            ns_rows = shape[1] if transpose_input else shape[0]
            ns_cols = shape[0] if transpose_input else shape[1]
            ext = self._load_cuda_ext_or_none()
            effective_batch = torch.empty(
                (len(bucket_items), shape[0], shape[1]),
                device=device,
                dtype=torch.float32,
            )
            momentum_batch = torch.empty(
                (len(bucket_items), shape[0], shape[1]),
                device=device,
                dtype=torch.float32,
            )
            norms = torch.empty(
                (len(bucket_items), 1),
                device=device,
                dtype=torch.float32,
            )
            ns_input_batch = torch.empty(
                (len(bucket_items), ns_rows, ns_cols),
                device=device,
                dtype=torch.bfloat16,
            )
            gram_batch = torch.empty(
                (len(bucket_items), ns_rows, ns_rows),
                device=device,
                dtype=torch.bfloat16,
            )
            gram_sq_batch = torch.empty_like(gram_batch)
            next_x_batch = torch.empty_like(ns_input_batch)
            param_ptrs = _make_cuda_pointer_tensor(params_bucket, device) if self._capturable_requested else None
            grad_ptrs = _make_cuda_pointer_tensor(grads_bucket, device) if self._capturable_requested else None
            square_backend = ""
            if family_code == 0 and ext is not None and hasattr(ext, "describe_square_backend"):
                square_backend = _muon_square_backend_name(int(ext.describe_square_backend(ns_input_batch)))
                if hasattr(ext, "prewarm_square_backend"):
                    ext.prewarm_square_backend(ns_input_batch)
            momentum_views = [momentum_batch[idx] for idx in range(len(bucket_items))]
            for state, view in zip(states_bucket, momentum_views, strict=True):
                existing = state.get("momentum_buffer")
                if existing is None:
                    view.zero_()
                else:
                    view.copy_(existing.to(dtype=view.dtype))
                state["momentum_buffer"] = view
            workspace = {
                "family_code": family_code,
                "effective_batch": effective_batch,
                "momentum_batch": momentum_batch,
                "norms": norms,
                "ns_input_batch": ns_input_batch,
                "gram_batch": gram_batch,
                "gram_sq_batch": gram_sq_batch,
                "next_x_batch": next_x_batch,
                "param_ptrs": param_ptrs,
                "grad_ptrs": grad_ptrs,
                "grad_tensor_ids": tuple(id(grad) for grad in grads_bucket),
                "grad_data_ptrs": tuple(int(grad.data_ptr()) for grad in grads_bucket),
                "square_backend": square_backend,
                "momentum_views": momentum_views,
            }
            self._cuda_bucket_workspaces[key] = workspace
        elif self._capturable_requested:
            grad_tensor_ids = tuple(id(grad) for grad in grads_bucket)
            grad_data_ptrs = tuple(int(grad.data_ptr()) for grad in grads_bucket)
            if (
                workspace.get("grad_tensor_ids") != grad_tensor_ids
                or workspace.get("grad_data_ptrs") != grad_data_ptrs
            ):
                workspace["grad_ptrs"] = _make_cuda_pointer_tensor(grads_bucket, device)
                workspace["grad_tensor_ids"] = grad_tensor_ids
                workspace["grad_data_ptrs"] = grad_data_ptrs
        return workspace

    def _record_square_backend_stats(
        self,
        stats: dict[str, Any],
        workspace: dict[str, Any],
        tensor_count: int,
        element_count: int,
        seen_backends: set[str],
    ) -> None:
        backend = str(workspace.get("square_backend", ""))
        if backend not in {"cublas", "cublaslt", "hybrid"}:
            return
        seen_backends.add(backend)
        bucket_key = f"square_backend_{backend}_bucket_count"
        tensor_key = f"square_backend_{backend}_tensor_count"
        element_key = f"square_backend_{backend}_element_count"
        stats[bucket_key] = int(stats.get(bucket_key, 0)) + 1
        stats[tensor_key] = int(stats.get(tensor_key, 0)) + int(tensor_count)
        stats[element_key] = int(stats.get(element_key, 0)) + int(element_count)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_stats: dict[str, Any] = {
            "group_count": 0,
            "tensor_count": 0,
            "bucket_count": 0,
            "cuda_tensor_count": 0,
            "fallback_tensor_count": 0,
            "cuda_element_count": 0,
            "fallback_element_count": 0,
            "cuda_failure_count": 0,
            "extension_available": False,
            "total_ms": 0.0,
            "cuda_ms": 0.0,
            "fallback_ms": 0.0,
            "square_backend_cublas_bucket_count": 0,
            "square_backend_cublas_tensor_count": 0,
            "square_backend_cublas_element_count": 0,
            "square_backend_cublaslt_bucket_count": 0,
            "square_backend_cublaslt_tensor_count": 0,
            "square_backend_cublaslt_element_count": 0,
            "square_backend_hybrid_bucket_count": 0,
            "square_backend_hybrid_tensor_count": 0,
            "square_backend_hybrid_element_count": 0,
        }
        seen_square_backends: set[str] = set()
        step_started_at = time.perf_counter() if PROFILE_MUON_STEP else 0.0
        if self._graph_capture_ready:
            muon_ext = self._load_cuda_ext_or_none()
            if muon_ext is None or not hasattr(muon_ext, "grouped_step_family_workspace_capturable"):
                raise RuntimeError("graph-safe Muon step requested without capturable muon_cuda extension")
            for group_idx, group in enumerate(self.param_groups):
                params = group["params"]
                if not params:
                    continue
                total_stats["group_count"] = int(total_stats["group_count"]) + 1
                total_stats["extension_available"] = True
                nesterov = bool(group["nesterov"])
                backend_refresh_interval = max(int(group["backend_refresh_interval"]), 1)
                step_index = int(group.get("step_index", 0)) + 1
                group["step_index"] = step_index
                use_full_backend = backend_refresh_interval <= 1 or step_index % backend_refresh_interval == 0
                current_backend_steps = int(group["backend_steps"] if use_full_backend else group["backend_steps_light"])
                lr_tensor = group.get("lr_tensor")
                momentum_tensor = group.get("momentum_tensor")
                weight_decay_tensor = group.get("weight_decay_tensor")
                if not isinstance(lr_tensor, Tensor) or not isinstance(momentum_tensor, Tensor) or not isinstance(weight_decay_tensor, Tensor):
                    raise RuntimeError("graph-safe Muon step requires tensor-backed lr/momentum/weight_decay")
                for bucket in self._static_group_buckets.get(group_idx, []):
                    params_bucket = bucket["params"]
                    grads_bucket = bucket["grads"]
                    workspace = bucket["workspace"]
                    if not isinstance(workspace.get("param_ptrs"), Tensor) or not isinstance(workspace.get("grad_ptrs"), Tensor):
                        raise RuntimeError("graph-safe Muon step requires cached CUDA pointer tensors")
                    bucket_element_count = sum(int(param.numel()) for param in params_bucket)
                    muon_ext.grouped_step_family_workspace_capturable(
                        params_bucket,
                        grads_bucket,
                        workspace["param_ptrs"],
                        workspace["grad_ptrs"],
                        workspace["effective_batch"],
                        workspace["momentum_batch"],
                        workspace["norms"],
                        workspace["ns_input_batch"],
                        workspace["gram_batch"],
                        workspace["gram_sq_batch"],
                        workspace["next_x_batch"],
                        int(bucket["family_code"]),
                        lr_tensor,
                        momentum_tensor,
                        weight_decay_tensor,
                        nesterov,
                        current_backend_steps,
                        1.0e-7,
                    )
                    total_stats["bucket_count"] = int(total_stats["bucket_count"]) + 1
                    total_stats["tensor_count"] = int(total_stats["tensor_count"]) + len(params_bucket)
                    total_stats["cuda_tensor_count"] = int(total_stats["cuda_tensor_count"]) + len(params_bucket)
                    total_stats["cuda_element_count"] = int(total_stats["cuda_element_count"]) + bucket_element_count
                    self._record_square_backend_stats(
                        total_stats,
                        workspace,
                        len(params_bucket),
                        bucket_element_count,
                        seen_square_backends,
                    )
            if PROFILE_MUON_STEP:
                total_stats["total_ms"] = (time.perf_counter() - step_started_at) * 1000.0
            total_stats["square_backend_modes"] = ",".join(sorted(seen_square_backends)) if seen_square_backends else "none"
            self.last_step_stats = total_stats
            return loss
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            total_stats["group_count"] = int(total_stats["group_count"]) + 1
            lr_value = self._group_tensor_or_float(group, "lr", 0.0)
            momentum_value = self._group_tensor_or_float(group, "momentum", 0.0)
            weight_decay_value = self._group_tensor_or_float(group, "weight_decay", 0.0)
            lr = float(lr_value.detach().item()) if isinstance(lr_value, Tensor) else float(lr_value)
            momentum = float(momentum_value.detach().item()) if isinstance(momentum_value, Tensor) else float(momentum_value)
            backend_steps = group["backend_steps"]
            backend_steps_light = group["backend_steps_light"]
            backend_refresh_interval = max(int(group["backend_refresh_interval"]), 1)
            weight_decay = (
                float(weight_decay_value.detach().item())
                if isinstance(weight_decay_value, Tensor)
                else float(weight_decay_value)
            )
            nesterov = group["nesterov"]
            step_index = int(group.get("step_index", 0)) + 1
            group["step_index"] = step_index
            use_full_backend = backend_refresh_interval <= 1 or step_index % backend_refresh_interval == 0
            current_backend_steps = backend_steps if use_full_backend else backend_steps_light
            grouped_buckets: dict[tuple[tuple[int, ...], torch.dtype, torch.device], list[tuple[Tensor, Tensor, dict[str, Tensor]]]] = {}
            fallback_items: list[tuple[Tensor, Tensor, dict[str, Tensor]]] = []
            muon_ext = None
            if USE_MUON_CUDA:
                try:
                    muon_ext = load_muon_cuda()
                except Exception:
                    muon_ext = None
            if muon_ext is not None:
                total_stats["extension_available"] = True
            for p in params:
                if p.grad is None:
                    continue
                total_stats["tensor_count"] = int(total_stats["tensor_count"]) + 1
                g = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                item = (p, g, state)
                if (
                    muon_ext is None
                    or p.device.type != "cuda"
                    or p.ndim != 2
                    or g.ndim != 2
                    or not p.is_contiguous()
                    or not g.is_contiguous()
                    or not buf.is_contiguous()
                ):
                    fallback_items.append(item)
                    continue
                bucket_key = (tuple(int(v) for v in p.shape), p.dtype, p.device)
                grouped_buckets.setdefault(bucket_key, []).append(item)

            for bucket_items in grouped_buckets.values():
                total_stats["bucket_count"] = int(total_stats["bucket_count"]) + 1
                params_bucket = [item[0] for item in bucket_items]
                grads_bucket = [item[1] for item in bucket_items]
                bucket_element_count = sum(int(param.numel()) for param in params_bucket)
                bucket_shape = tuple(int(v) for v in params_bucket[0].shape)
                if not _muon_prefers_cuda_bucket(bucket_shape, bucket_size=len(bucket_items)):
                    fallback_items.extend(bucket_items)
                    continue
                workspace = self._get_cuda_bucket_workspace(bucket_items)
                bucket_started_at = time.perf_counter() if PROFILE_MUON_STEP else 0.0
                try:
                    assert muon_ext is not None
                    muon_ext.grouped_step_family_workspace(
                        params_bucket,
                        grads_bucket,
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
                        int(current_backend_steps),
                        1.0e-7,
                    )
                    total_stats["cuda_tensor_count"] = int(total_stats["cuda_tensor_count"]) + len(bucket_items)
                    total_stats["cuda_element_count"] = int(total_stats["cuda_element_count"]) + bucket_element_count
                    self._record_square_backend_stats(
                        total_stats,
                        workspace,
                        len(bucket_items),
                        bucket_element_count,
                        seen_square_backends,
                    )
                    if PROFILE_MUON_STEP:
                        total_stats["cuda_ms"] = float(total_stats["cuda_ms"]) + (
                            (time.perf_counter() - bucket_started_at) * 1000.0
                        )
                except Exception:
                    total_stats["cuda_failure_count"] = int(total_stats["cuda_failure_count"]) + 1
                    fallback_items.extend(bucket_items)

            for p, g, state in fallback_items:
                total_stats["fallback_tensor_count"] = int(total_stats["fallback_tensor_count"]) + 1
                total_stats["fallback_element_count"] = int(total_stats["fallback_element_count"]) + int(p.numel())
                fallback_started_at = time.perf_counter() if PROFILE_MUON_STEP else 0.0
                _muon_python_step_param(
                    p,
                    g,
                    state,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    nesterov=nesterov,
                    backend_steps=current_backend_steps,
                )
                if PROFILE_MUON_STEP:
                    total_stats["fallback_ms"] = float(total_stats["fallback_ms"]) + (
                        (time.perf_counter() - fallback_started_at) * 1000.0
                    )

        if PROFILE_MUON_STEP:
            total_stats["total_ms"] = (time.perf_counter() - step_started_at) * 1000.0
        total_stats["square_backend_modes"] = ",".join(sorted(seen_square_backends)) if seen_square_backends else "none"
        self.last_step_stats = total_stats
        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def build_tokenizers_json_luts(tokenizer_path: str, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    tok_vocab_size = int(tok.get_vocab_size())
    table_size = max(tok_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    special_tokens = {"[PAD]", "[BOS]", "[EOS]", "[UNK]"}
    for token_id in range(tok_vocab_size):
        piece = tok.id_to_token(token_id)
        if piece is None or piece in special_tokens:
            continue
        decoded = tok.decode([token_id], skip_special_tokens=False)
        if not decoded:
            continue
        base_bytes_np[token_id] = len(decoded.encode("utf-8"))
        is_boundary_token_np[token_id] = False
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def get_eval_seq_len(args: Hyperparameters) -> int:
    return args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len


def trace_shape(x: Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in x.shape)


def enforce_static_shape(x: Tensor, spec: tuple[int, ...], name: str = "tensor") -> Tensor:
    shape = trace_shape(x)
    if len(shape) != len(spec):
        raise ValueError(f"{name} rank mismatch: actual={shape} expected={spec}")
    for actual_dim, expected_dim in zip(shape, spec, strict=True):
        if expected_dim > 0 and actual_dim != expected_dim:
            raise ValueError(f"{name} shape mismatch: actual={shape} expected={spec}")
    return x


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def count_total_train_tokens(pattern: str) -> int:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return sum(_read_shard_num_tokens(file) for file in files)


def pad_token_pair_batch(
    x_seqs: list[Tensor], y_seqs: list[Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    if not x_seqs or len(x_seqs) != len(y_seqs):
        raise ValueError("Expected non-empty matching x/y sequence lists")
    lengths = [int(x.numel()) for x in x_seqs]
    max_len = max(lengths)
    batch_size = len(x_seqs)
    x_batch = torch.zeros((batch_size, max_len), dtype=torch.int64, device=device)
    y_batch = torch.zeros((batch_size, max_len), dtype=torch.int64, device=device)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    for i, (x_seq, y_seq) in enumerate(zip(x_seqs, y_seqs, strict=True)):
        seq_len = int(x_seq.numel())
        x_batch[i, :seq_len] = x_seq.to(device=device, dtype=torch.int64, non_blocking=True)
        y_batch[i, :seq_len] = y_seq.to(device=device, dtype=torch.int64, non_blocking=True)
        loss_mask[i, :seq_len] = True
    return x_batch, y_batch, loss_mask, lengths


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len_override: int = 0,
    max_seqs: int = 0,
) -> tuple[float, float, float, float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    seq_len = seq_len_override if seq_len_override > 0 else args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, EVAL_SEQ_LEN={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    if max_seqs > 0:
        total_seqs = min(total_seqs, max_seqs)
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    # Rotary caches persist across eval and training; inference_mode would cache
    # inference tensors here and then autograd would reject them on the next step.
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            if seq_len == args.train_seq_len:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    batch_loss = model(x, y).detach()
                val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            else:
                logits_model = model.module if hasattr(model, "module") else model
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = logits_model.forward_logits(x)
                batch_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                    reduction="sum",
                )
                val_loss_sum += batch_loss.to(torch.float64)
            batch_token_count = float(y.numel())
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    bytes_per_token = val_byte_count.item() / val_token_count.item()
    model.train()
    return (
        float(val_loss.item()),
        float(bits_per_token * tokens_per_byte),
        float(bits_per_token),
        float(tokens_per_byte),
        float(bytes_per_token),
    )


def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int,
    seq_len_override: int = 0,
    max_seqs: int = 0,
) -> tuple[float, float, float, float, float]:
    if batch_seqs <= 0:
        raise ValueError(f"EVAL_BATCH_SEQS must be positive for sliding eval, got {batch_seqs}")
    seq_len = seq_len_override if seq_len_override > 0 else args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    if max_seqs > 0:
        total_tokens = min(total_tokens, max_seqs * seq_len)
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= stride]
    total_windows = len(window_starts)
    my_start = (total_windows * rank) // world_size
    my_end = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_start:my_end]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    # Keep cached RoPE tensors compatible with the subsequent training phase.
    with torch.no_grad():
        for batch_idx in range(0, len(my_windows), batch_seqs):
            batch_windows = my_windows[batch_idx : batch_idx + batch_seqs]
            x_seqs: list[Tensor] = []
            y_seqs: list[Tensor] = []
            window_lengths: list[int] = []

            for ws in batch_windows:
                end = min(ws + seq_len, total_tokens)
                window_len = end - ws
                window_lengths.append(window_len)
                chunk = val_tokens[ws : end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
                x_seqs.append(chunk[:-1])
                y_seqs.append(chunk[1:])

            if args.use_packed_eval_windows:
                x_batch, y_batch, _, _ = pad_token_pair_batch(x_seqs, y_seqs, device)
            else:
                batch_size = len(batch_windows)
                x_batch = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
                y_batch = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
                for i, (x_seq, y_seq) in enumerate(zip(x_seqs, y_seqs, strict=True)):
                    window_len = int(x_seq.numel())
                    x_batch[i, :window_len] = x_seq
                    y_batch[i, :window_len] = y_seq

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(x_batch)

            batch_size, batch_width = y_batch.shape
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(batch_size, batch_width)

            for i, ws in enumerate(batch_windows):
                window_len = window_lengths[i]
                score_start = 0 if ws == 0 else window_len - stride
                scored_nll = nll[i, score_start:window_len].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(window_len - score_start)
                tgt_ids = y_batch[i, score_start:window_len]
                prev_ids = x_batch[i, score_start:window_len]
                token_bytes = base_bytes_lut[tgt_ids].to(torch.float64)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                byte_count += token_bytes.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    bytes_per_token = byte_count.item() / token_count.item()
    base_model.train()
    return (
        float(val_loss.item()),
        float(bits_per_token * tokens_per_byte),
        float(bits_per_token),
        float(tokens_per_byte),
        float(bytes_per_token),
    )


def eval_val_sliding_streaming_strict(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len_override: int = 0,
    max_seqs: int = 0,
) -> tuple[float, float, float, float, float]:
    seq_len = seq_len_override if seq_len_override > 0 else args.train_seq_len
    total_targets = val_tokens.numel() - 1
    if max_seqs > 0:
        total_targets = min(total_targets, max_seqs * seq_len)
    target_start = (total_targets * rank) // world_size
    target_end = (total_targets * (rank + 1)) // world_size
    warm_start = max(target_start - seq_len, 0)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    backbone_cache = base_model.init_backbone_step_cache(max_len=seq_len)
    causal_machine_cache = (
        base_model.init_causal_machine_cache(batch_size=1, device=device, dtype=torch.float32)
        if hasattr(base_model, "init_causal_machine_cache")
        else None
    )
    with torch.no_grad():
        for pos in range(warm_start, target_end):
            token_in = val_tokens[pos : pos + 1].to(device=device, dtype=torch.int64, non_blocking=True).view(1, 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(
                    token_in,
                    backbone_cache=backbone_cache,
                    causal_machine_cache=causal_machine_cache,
                    update_causal_machine_cache=True,
                )
            if pos < target_start:
                continue
            target = val_tokens[pos + 1 : pos + 2].to(device=device, dtype=torch.int64, non_blocking=True).view(1)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target, reduction="sum")
            loss_sum += nll.to(torch.float64)
            token_count += 1.0
            prev_id = token_in.reshape(-1)
            tgt_id = target.reshape(-1)
            token_bytes = base_bytes_lut[tgt_id].to(torch.float64)
            token_bytes += (has_leading_space_lut[tgt_id] & ~is_boundary_token_lut[prev_id]).to(torch.float64)
            byte_count += token_bytes.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    bytes_per_token = byte_count.item() / token_count.item()
    base_model.train()
    return (
        float(val_loss.item()),
        float(bits_per_token * tokens_per_byte),
        float(bits_per_token),
        float(tokens_per_byte),
        float(bytes_per_token),
    )


def uses_sliding_eval(args: Hyperparameters) -> bool:
    return 0 < args.eval_stride < get_eval_seq_len(args)


def can_use_strict_streaming_caches(base_model: nn.Module) -> bool:
    return bool(getattr(base_model, "supports_incremental_backbone_cache", lambda: False)())


def run_validation(
    args: Hyperparameters,
    model: nn.Module,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len_override: int = 0,
    max_seqs: int = 0,
) -> tuple[float, float, str, float, float, float]:
    eval_seq_len = seq_len_override if seq_len_override > 0 else get_eval_seq_len(args)
    if uses_sliding_eval(args):
        if args.use_strict_streaming_eval and can_use_strict_streaming_caches(base_model):
            val_loss, val_bpb, bits_per_token, tokens_per_byte, bytes_per_token = eval_val_sliding_streaming_strict(
                args,
                base_model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                seq_len_override=eval_seq_len,
                max_seqs=max_seqs,
            )
            return val_loss, val_bpb, "sliding_streaming", bits_per_token, tokens_per_byte, bytes_per_token
        val_loss, val_bpb, bits_per_token, tokens_per_byte, bytes_per_token = eval_val_sliding(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            seq_len_override=eval_seq_len,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
            max_seqs=max_seqs,
        )
        return val_loss, val_bpb, "sliding", bits_per_token, tokens_per_byte, bytes_per_token

    val_loss, val_bpb, bits_per_token, tokens_per_byte, bytes_per_token = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        seq_len_override=eval_seq_len,
        max_seqs=max_seqs,
    )
    return val_loss, val_bpb, "standard", bits_per_token, tokens_per_byte, bytes_per_token


def objective_module_group(name: str) -> str:
    if name == "tok_emb.weight":
        return "embed_head_shared"
    if "lm_head" in name:
        return "head"
    if ".attn." in name:
        return "attn"
    if ".mlp." in name or "mlp_" in name:
        return "mlp"
    if "norm" in name:
        return "norm"
    return "other"


def warmup_prior_init_param_names(model: nn.Module, groups: tuple[str, ...]) -> list[str]:
    return warmup_prior_init_param_names_for_mode(
        model,
        groups,
        getattr(Hyperparameters, "warmup_prior_init_mode", "safe"),
    )


def warmup_prior_init_param_names_for_mode(
    model: nn.Module,
    groups: tuple[str, ...],
    mode: str,
) -> list[str]:
    allowed_groups = {item.strip() for item in groups if item.strip()}
    if not allowed_groups:
        return []
    mode = str(mode or "safe").strip().lower()
    safe_name_patterns = ("cond_gate",)
    selected: list[str] = []
    for name, param in model.named_parameters():
        if objective_module_group(name) not in allowed_groups:
            continue
        is_control = param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        if param.dtype != torch.float32:
            continue
        if mode == "safe" and not any(pattern in name for pattern in safe_name_patterns):
            continue
        if mode in {"bounded_delta", "delta"} and not (is_control or param.ndim >= 2):
            continue
        selected.append(name)
    return selected


def blend_state_tensor(initial: Tensor, warmed: Tensor, blend: float) -> Tensor:
    blend = min(max(float(blend), 0.0), 1.0)
    if blend <= 0.0:
        return initial.clone()
    warmed = warmed.to(device=initial.device, dtype=initial.dtype)
    if blend >= 1.0:
        return warmed.clone()
    mixed = initial.float().lerp(warmed.float(), blend)
    return mixed.to(dtype=initial.dtype)


def transfer_warmup_state_tensor(
    initial: Tensor,
    warmed: Tensor,
    blend: float,
    mode: str,
    delta_rms_mult: float,
    delta_rms_floor: float,
) -> Tensor:
    mode = str(mode or "safe").strip().lower()
    if mode not in {"bounded_delta", "delta"}:
        return blend_state_tensor(initial, warmed, blend)
    blend = min(max(float(blend), 0.0), 1.0)
    if blend <= 0.0:
        return initial.clone()
    initial_f = initial.float()
    warmed_f = warmed.to(device=initial.device, dtype=torch.float32)
    delta = (warmed_f - initial_f) * blend
    delta_rms = delta.pow(2).mean().sqrt()
    initial_rms = initial_f.pow(2).mean().sqrt()
    max_delta_rms = max(float(initial_rms.item()) * max(float(delta_rms_mult), 0.0), float(delta_rms_floor))
    if float(delta_rms.item()) > max_delta_rms > 0.0:
        delta.mul_(max_delta_rms / max(float(delta_rms.item()), 1e-12))
    return (initial_f + delta).to(dtype=initial.dtype)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_NAME_PATTERNS = tuple(
    pattern for pattern in os.environ.get("INT8_KEEP_FLOAT_NAME_PATTERNS", "").split(",") if pattern
)
MIXED_PRECISION_KEEP_FP16_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "MIXED_PRECISION_KEEP_FP16_NAME_PATTERNS",
        "tok_emb.weight,blocks.7.attn.c_k.weight,blocks.8.attn.c_k.weight",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
EXPORT_CALIBRATION_METHOD = os.environ.get("EXPORT_CALIBRATION_METHOD", "percentile_mse").strip().lower()
EXPORT_CALIBRATION_PERCENTILES_RAW = os.environ.get(
    "EXPORT_CALIBRATION_PERCENTILES",
    f"{INT8_CLIP_Q:.8f},0.999,0.9995,0.9999,0.99999,1.0",
)
MIXED_PRECISION_ROW_CLIP_SEARCH = bool(int(os.environ.get("MIXED_PRECISION_ROW_CLIP_SEARCH", "1")))
MIXED_PRECISION_ROW_CLIP_CANDIDATES = (0.999, 0.9995, 0.9999, 0.99999, 1.0)
FORCE_FP16_TIED_EMBED_EXPORT = bool(int(os.environ.get("FORCE_FP16_TIED_EMBED_EXPORT", "1")))
LATE_K_FP16_LAYERS = int(os.environ.get("LATE_K_FP16_LAYERS", "2"))
LARGE_MATRIX_QUANT_BITS = int(os.environ.get("LARGE_MATRIX_QUANT_BITS", "6"))
MIXED_PRECISION_EXPORT_MLP_BITS = int(os.environ.get("EXPORT_MLP_BITS", "0"))
MIXED_PRECISION_EXPORT_ATTN_BITS = int(os.environ.get("EXPORT_ATTN_BITS", "0"))
EXPORT_TENSOR_CANDIDATE_BITS_RAW = os.environ.get("EXPORT_TENSOR_CANDIDATE_BITS", "4,5,6,8")
EXPORT_TENSOR_SWEEP_OBJECTIVE = os.environ.get("EXPORT_TENSOR_SWEEP_OBJECTIVE", "mse_per_compressed_byte").strip().lower()
EXPORT_TENSOR_SWEEP_MIN_NUMEL = int((os.environ.get("EXPORT_TENSOR_SWEEP_MIN_NUMEL", "65536") or "65536").strip())
EXPORT_TENSOR_SWEEP_MAX_EXTRA_REL_MSE = float(
    (os.environ.get("EXPORT_TENSOR_SWEEP_MAX_EXTRA_REL_MSE", "0.0005") or "0.0005").strip()
)
EXPORT_TENSOR_SWEEP_LOG_TOPK = int((os.environ.get("EXPORT_TENSOR_SWEEP_LOG_TOPK", "8") or "8").strip())
EXPORT_BLOCK_PRUNE_ENABLED = bool(int(os.environ.get("EXPORT_BLOCK_PRUNE", "0")))
EXPORT_BLOCK_PRUNE_DENSITIES_RAW = os.environ.get("EXPORT_BLOCK_PRUNE_DENSITIES", "0.98,0.95,0.90")
EXPORT_BLOCK_PRUNE_BLOCK_RAW = os.environ.get("EXPORT_BLOCK_PRUNE_BLOCK", "16,16")
EXPORT_BLOCK_PRUNE_MIN_NUMEL = int(os.environ.get("EXPORT_BLOCK_PRUNE_MIN_NUMEL", "65536"))
EXPORT_BLOCK_PRUNE_MAX_EXTRA_REL_MSE = float(os.environ.get("EXPORT_BLOCK_PRUNE_MAX_EXTRA_REL_MSE", "0.0005"))
EXPORT_BLOCK_PRUNE_LOG_TOPK = int(os.environ.get("EXPORT_BLOCK_PRUNE_LOG_TOPK", "8"))
EXPORT_CHANNEL_EQUALIZATION = bool(int(os.environ.get("EXPORT_CHANNEL_EQUALIZATION", "1")))
EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN = max(float(os.environ.get("EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN", "4.0")), 1.0)

ACTIVE_EXPORT_CONFIG: dict[str, object] = {
    "num_layers": Hyperparameters.num_layers,
    "force_fp16_tied_embed_export": FORCE_FP16_TIED_EMBED_EXPORT,
    "late_k_fp16_layers": LATE_K_FP16_LAYERS,
    "int8_auto_keep_budget_bytes": Hyperparameters.int8_auto_keep_budget_bytes,
    "int8_auto_keep_max_tensors": Hyperparameters.int8_auto_keep_max_tensors,
    "int8_auto_keep_min_numel": Hyperparameters.int8_auto_keep_min_numel,
    "int8_sensitivity_log_topk": Hyperparameters.int8_sensitivity_log_topk,
    "int8_auto_keep_row_budget_bytes": Hyperparameters.int8_auto_keep_row_budget_bytes,
    "int8_auto_keep_row_group_size": Hyperparameters.int8_auto_keep_row_group_size,
    "export_codec": Hyperparameters.export_codec,
    "export_zstd_level": Hyperparameters.export_zstd_level,
    "export_high_precision_bits": Hyperparameters.export_high_precision_bits,
    "export_high_precision_budget_bytes": Hyperparameters.export_high_precision_budget_bytes,
    "export_high_precision_max_tensors": Hyperparameters.export_high_precision_max_tensors,
    "export_high_precision_min_numel": Hyperparameters.export_high_precision_min_numel,
}


def configure_runtime_export(args: Hyperparameters) -> None:
    ACTIVE_EXPORT_CONFIG.update(
        {
            "num_layers": int(args.num_layers),
            "force_fp16_tied_embed_export": bool(args.force_fp16_tied_embed_export),
            "late_k_fp16_layers": int(LATE_K_FP16_LAYERS),
            "int8_auto_keep_budget_bytes": int(args.int8_auto_keep_budget_bytes),
            "int8_auto_keep_max_tensors": int(args.int8_auto_keep_max_tensors),
            "int8_auto_keep_min_numel": int(args.int8_auto_keep_min_numel),
            "int8_sensitivity_log_topk": int(args.int8_sensitivity_log_topk),
            "int8_auto_keep_row_budget_bytes": int(args.int8_auto_keep_row_budget_bytes),
            "int8_auto_keep_row_group_size": int(args.int8_auto_keep_row_group_size),
            "export_codec": str(args.export_codec),
            "export_zstd_level": int(args.export_zstd_level),
            "export_high_precision_bits": int(args.export_high_precision_bits),
            "export_high_precision_budget_bytes": int(args.export_high_precision_budget_bytes),
            "export_high_precision_max_tensors": int(args.export_high_precision_max_tensors),
            "export_high_precision_min_numel": int(args.export_high_precision_min_numel),
        }
    )

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def should_keep_float_tensor(name: str, t: Tensor) -> bool:
    if bool(ACTIVE_EXPORT_CONFIG["force_fp16_tied_embed_export"]) and name == "tok_emb.weight":
        return True
    if (
        int(ACTIVE_EXPORT_CONFIG["late_k_fp16_layers"]) > 0
        and name.endswith("c_k.weight")
        and name.startswith("blocks.")
        and t.ndim == 2
    ):
        try:
            block_idx = int(name.split(".")[1])
            num_layers = int(ACTIVE_EXPORT_CONFIG["num_layers"])
            late_k_fp16_layers = int(ACTIVE_EXPORT_CONFIG["late_k_fp16_layers"])
            if block_idx >= max(num_layers - late_k_fp16_layers, 0):
                return True
        except (IndexError, ValueError):
            pass
    return t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or any(pattern in name for pattern in INT8_KEEP_FLOAT_NAME_PATTERNS)

def parse_export_calibration_percentiles(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = float(piece)
        if value > 1.0:
            value /= 100.0
        values.append(min(max(value, 1e-6), 1.0))
    if not values:
        values = [INT8_CLIP_Q, 1.0]
    values.append(1.0)
    return tuple(sorted(set(values)))


EXPORT_CALIBRATION_PERCENTILES = parse_export_calibration_percentiles(EXPORT_CALIBRATION_PERCENTILES_RAW)


def parse_export_block_prune_densities(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            continue
        if 0.0 < value <= 1.0:
            values.append(value)
    if not values:
        values = [0.98, 0.95, 0.90]
    return tuple(sorted(set(values), reverse=True))


def parse_export_block_prune_block(raw: str) -> tuple[int, int]:
    pieces = [piece.strip().lower() for piece in raw.replace("x", ",").split(",") if piece.strip()]
    if len(pieces) != 2:
        return (16, 16)
    try:
        block_h = max(int(pieces[0]), 1)
        block_w = max(int(pieces[1]), 1)
    except ValueError:
        return (16, 16)
    return block_h, block_w


def parse_export_tensor_candidate_bits(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for piece in raw.replace("/", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            bits = int(piece)
        except ValueError:
            continue
        if 2 <= bits <= 8:
            values.append(bits)
    if not values:
        values = [4, 5, 6, 8]
    return tuple(sorted(set(values)))

EXPORT_BLOCK_PRUNE_DENSITIES = parse_export_block_prune_densities(EXPORT_BLOCK_PRUNE_DENSITIES_RAW)
EXPORT_BLOCK_PRUNE_BLOCK = parse_export_block_prune_block(EXPORT_BLOCK_PRUNE_BLOCK_RAW)
EXPORT_TENSOR_CANDIDATE_BITS = parse_export_tensor_candidate_bits(EXPORT_TENSOR_CANDIDATE_BITS_RAW)


def block_magnitude_prune_2d(t: Tensor, density: float, block: tuple[int, int]) -> Tensor:
    if t.ndim != 2:
        raise ValueError("block_magnitude_prune_2d expects a 2D tensor")
    keep_density = min(max(float(density), 0.0), 1.0)
    if keep_density >= 1.0:
        return t.contiguous()
    block_h, block_w = max(int(block[0]), 1), max(int(block[1]), 1)
    height, width = int(t.shape[0]), int(t.shape[1])
    tiles_h = (height + block_h - 1) // block_h
    tiles_w = (width + block_w - 1) // block_w
    pad_h = tiles_h * block_h - height
    pad_w = tiles_w * block_w - width
    padded = F.pad(t, (0, pad_w, 0, pad_h))
    blocks = padded.view(tiles_h, block_h, tiles_w, block_w).permute(0, 2, 1, 3)
    norms = blocks.reshape(tiles_h, tiles_w, -1).abs().sum(dim=-1)
    keep_blocks = max(1, int(round(keep_density * norms.numel())))
    threshold = norms.reshape(-1).kthvalue(norms.numel() - keep_blocks + 1).values
    keep = norms >= threshold
    keep_mask = keep.repeat_interleave(block_h, dim=0).repeat_interleave(block_w, dim=1)
    pruned = padded * keep_mask.to(dtype=padded.dtype)
    return pruned[:height, :width].contiguous()


def _reshape_tensor_into_blocks_2d(t: Tensor, block: tuple[int, int]) -> tuple[Tensor, Tensor, dict[str, int]]:
    if t.ndim != 2:
        raise ValueError("_reshape_tensor_into_blocks_2d expects a 2D tensor")
    block_h, block_w = max(int(block[0]), 1), max(int(block[1]), 1)
    height, width = int(t.shape[0]), int(t.shape[1])
    tiles_h = (height + block_h - 1) // block_h
    tiles_w = (width + block_w - 1) // block_w
    pad_h = tiles_h * block_h - height
    pad_w = tiles_w * block_w - width
    padded = F.pad(t, (0, pad_w, 0, pad_h))
    blocks = padded.view(tiles_h, block_h, tiles_w, block_w).permute(0, 2, 1, 3).contiguous()
    meta = {
        "height": height,
        "width": width,
        "block_h": block_h,
        "block_w": block_w,
        "tiles_h": tiles_h,
        "tiles_w": tiles_w,
        "padded_h": tiles_h * block_h,
        "padded_w": tiles_w * block_w,
    }
    return padded.contiguous(), blocks, meta


def _merge_blocks_into_tensor_2d(blocks: Tensor, meta: dict[str, int], dtype: torch.dtype) -> Tensor:
    padded = (
        blocks.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(meta["padded_h"]), int(meta["padded_w"]))
    )
    return padded[: int(meta["height"]), : int(meta["width"])].to(dtype=dtype).contiguous()


def _quantize_blocks_with_clip(blocks: Tensor, clip_abs: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    qmax = float(127 if bits == 8 else (2 ** (bits - 1)) - 1)
    qmin = -float(qmax if bits == 8 else (2 ** (bits - 1)))
    clip_abs = clip_abs.clamp_min(1e-12)
    clipped = torch.maximum(torch.minimum(blocks, clip_abs[..., None, None]), -clip_abs[..., None, None])
    scale = (clip_abs / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(clipped / scale[..., None, None]), qmin, qmax).to(torch.int8).contiguous()
    return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()


def quantize_mixed_precision_blockwise_2d(t: Tensor, bits: int, block: tuple[int, int]) -> tuple[Tensor, Tensor]:
    t32 = t.float().contiguous()
    _, blocks, meta = _reshape_tensor_into_blocks_2d(t32, block)
    block_abs = blocks.abs().reshape(int(meta["tiles_h"]), int(meta["tiles_w"]), -1)
    if EXPORT_CALIBRATION_METHOD == "absmax":
        q_blocks, scales = _quantize_blocks_with_clip(blocks, block_abs.amax(dim=-1), bits)
        return _merge_blocks_into_tensor_2d(q_blocks, meta, torch.int8), scales
    if EXPORT_CALIBRATION_METHOD == "percentile":
        q_blocks, scales = _quantize_blocks_with_clip(blocks, torch.quantile(block_abs, INT8_CLIP_Q, dim=-1), bits)
        return _merge_blocks_into_tensor_2d(q_blocks, meta, torch.int8), scales

    best_q, best_s = _quantize_blocks_with_clip(blocks, block_abs.amax(dim=-1), bits)
    best_mse = (best_q.float() * best_s[..., None, None].float() - blocks).square().mean(dim=(-1, -2))
    for clip_q in EXPORT_CALIBRATION_PERCENTILES:
        if clip_q >= 1.0:
            continue
        q_blocks, scales = _quantize_blocks_with_clip(blocks, torch.quantile(block_abs, clip_q, dim=-1), bits)
        mse = (q_blocks.float() * scales[..., None, None].float() - blocks).square().mean(dim=(-1, -2))
        better = mse < best_mse
        if bool(better.any().item()):
            best_mse = torch.where(better, mse, best_mse)
            best_s = torch.where(better, scales, best_s)
            best_q = torch.where(better[..., None, None], q_blocks, best_q)
    return _merge_blocks_into_tensor_2d(best_q, meta, torch.int8), best_s.contiguous()


def extract_blockwise_outliers_2d(
    t: Tensor, keep_density: float, block: tuple[int, int]
) -> tuple[Tensor, dict[str, Tensor] | None, float]:
    t32 = t.float().contiguous()
    keep_density = min(max(float(keep_density), 0.0), 1.0)
    if keep_density >= 1.0 or t32.numel() == 0:
        return t32, None, 1.0
    _, blocks, meta = _reshape_tensor_into_blocks_2d(t32, block)
    block_elems = max(int(meta["block_h"]) * int(meta["block_w"]), 1)
    outliers_per_block = int(round((1.0 - keep_density) * block_elems))
    outliers_per_block = min(max(outliers_per_block, 0), max(block_elems - 1, 0))
    if outliers_per_block <= 0:
        return t32, None, 1.0
    block_abs = blocks.abs().reshape(int(meta["tiles_h"]), int(meta["tiles_w"]), -1)
    topk = torch.topk(block_abs, k=outliers_per_block, dim=-1, largest=True, sorted=False).indices
    mask_flat = torch.zeros_like(block_abs, dtype=torch.bool)
    mask_flat.scatter_(-1, topk, True)
    mask_blocks = mask_flat.view(
        int(meta["tiles_h"]), int(meta["tiles_w"]), int(meta["block_h"]), int(meta["block_w"])
    )
    mask = _merge_blocks_into_tensor_2d(mask_blocks.to(dtype=torch.float32), meta, torch.bool)
    indices = mask.reshape(-1).nonzero(as_tuple=True)[0].to(dtype=torch.int32).contiguous()
    if indices.numel() == 0:
        return t32, None, 1.0
    values = t32.reshape(-1)[indices.to(dtype=torch.int64)].to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    masked = t32.masked_fill(mask, 0.0).contiguous()
    actual_density = 1.0 - (float(indices.numel()) / max(float(t32.numel()), 1.0))
    return masked, {"indices": indices, "values": values}, actual_density


def _quantize_per_row_with_clip(t32: Tensor, clip_abs: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    qmax = float(127 if bits == 8 else (2 ** (bits - 1)) - 1)
    qmin = -float(qmax if bits == 8 else (2 ** (bits - 1)))
    clip_abs = clip_abs.clamp_min(1e-12)
    clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
    scale = (clip_abs / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(clipped / scale[:, None]), qmin, qmax).to(torch.int8).contiguous()
    return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()


def _quantize_per_tensor_with_clip(t32: Tensor, clip_abs: float, bits: int) -> tuple[Tensor, Tensor]:
    qmax = float(127 if bits == 8 else (2 ** (bits - 1)) - 1)
    qmin = -float(qmax if bits == 8 else (2 ** (bits - 1)))
    clip_abs = max(float(clip_abs), 1e-12)
    scale = torch.tensor(clip_abs / qmax, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), qmin, qmax).to(torch.int8).contiguous()
    return q, scale


def quantize_tensor_calibrated(t: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.numel() == 0:
        return _quantize_per_tensor_with_clip(t32, 1.0, bits)

    if t32.ndim == 2:
        row_abs = t32.abs()
        if EXPORT_CALIBRATION_METHOD == "absmax":
            return _quantize_per_row_with_clip(t32, row_abs.amax(dim=1), bits)
        if EXPORT_CALIBRATION_METHOD == "percentile":
            return _quantize_per_row_with_clip(t32, torch.quantile(row_abs, INT8_CLIP_Q, dim=1), bits)

        best_q, best_s = _quantize_per_row_with_clip(t32, row_abs.amax(dim=1), bits)
        best_mse = (best_q.float() * best_s[:, None].float() - t32).square().mean(dim=1)
        for clip_q in EXPORT_CALIBRATION_PERCENTILES:
            if clip_q >= 1.0:
                continue
            q, s = _quantize_per_row_with_clip(t32, torch.quantile(row_abs, clip_q, dim=1), bits)
            mse = (q.float() * s[:, None].float() - t32).square().mean(dim=1)
            better = mse < best_mse
            if bool(better.any().item()):
                best_mse = torch.where(better, mse, best_mse)
                best_s = torch.where(better, s, best_s)
                best_q[better] = q[better]
        return best_q.contiguous(), best_s.contiguous()

    flat_abs = t32.abs().flatten()
    if EXPORT_CALIBRATION_METHOD == "absmax":
        return _quantize_per_tensor_with_clip(t32, float(flat_abs.amax().item()), bits)
    if EXPORT_CALIBRATION_METHOD == "percentile":
        return _quantize_per_tensor_with_clip(t32, float(torch.quantile(flat_abs, INT8_CLIP_Q).item()), bits)

    best_q, best_s = _quantize_per_tensor_with_clip(t32, float(flat_abs.amax().item()), bits)
    best_mse = float((best_q.float() * float(best_s.item()) - t32).square().mean().item())
    for clip_q in EXPORT_CALIBRATION_PERCENTILES:
        if clip_q >= 1.0:
            continue
        q, s = _quantize_per_tensor_with_clip(t32, float(torch.quantile(flat_abs, clip_q).item()), bits)
        mse = float((q.float() * float(s.item()) - t32).square().mean().item())
        if mse < best_mse:
            best_q, best_s, best_mse = q, s, mse
    return best_q, best_s


def quantize_float_tensor(t: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    return quantize_tensor_calibrated(t, bits=bits)


def quantize_tensor_row_groups(
    name: str,
    t: Tensor,
    row_group_size: int,
    passthrough_orig_dtypes: dict[str, str],
) -> list[dict[str, object]]:
    if t.ndim != 2 or row_group_size <= 0:
        return []
    groups: list[dict[str, object]] = []
    for row_start in range(0, t.shape[0], row_group_size):
        row_end = min(row_start + row_group_size, t.shape[0])
        chunk = t[row_start:row_end].contiguous()
        q, s = quantize_float_tensor(chunk)
        deq = dequantize_int8_tensor(q, s, t.dtype)
        diff = deq.float() - chunk.float()
        mse = float(diff.square().mean().item()) if diff.numel() else 0.0
        ref = float(chunk.float().square().mean().item()) if chunk.numel() else 0.0
        rel_mse = mse / max(ref, 1e-12)
        quant_bytes = tensor_nbytes(q) + tensor_nbytes(s)
        keep_bytes = tensor_nbytes(keep_float_tensor(name, chunk, passthrough_orig_dtypes))
        extra_bytes = max(keep_bytes - quant_bytes, 0)
        benefit = rel_mse * float(chunk.numel())
        groups.append(
            {
                "name": name,
                "row_start": row_start,
                "row_end": row_end,
                "extra_bytes": int(extra_bytes),
                "rel_mse": rel_mse,
                "benefit_per_extra_byte": benefit / max(extra_bytes, 1),
            }
        )
    return groups


def dequantize_int8_tensor(q: Tensor, s: Tensor, dtype: torch.dtype) -> Tensor:
    if s.ndim > 0:
        return (q.float() * s.to(dtype=torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
    return (q.float() * float(s.item())).to(dtype=dtype).contiguous()


def restore_nested_scale_tensor(s: Tensor, qmeta_entry: dict[str, object] | None) -> Tensor:
    if not isinstance(qmeta_entry, dict):
        return s.to(dtype=torch.float32).contiguous()
    nested_scale_scales = qmeta_entry.get("nested_scale_scales")
    if isinstance(nested_scale_scales, Tensor):
        return dequantize_int8_tensor(s, nested_scale_scales, torch.float32)
    return s.to(dtype=torch.float32).contiguous()


def dequantize_blockwise_int8_tensor(
    q: Tensor, s: Tensor, dtype: torch.dtype, block: tuple[int, int], qmeta_entry: dict[str, object] | None = None
) -> Tensor:
    _, q_blocks, meta = _reshape_tensor_into_blocks_2d(q.float(), block)
    scales = restore_nested_scale_tensor(s, qmeta_entry).view(int(meta["tiles_h"]), int(meta["tiles_w"]))
    deq_blocks = q_blocks.float() * scales[..., None, None].to(dtype=torch.float32)
    return _merge_blocks_into_tensor_2d(deq_blocks, meta, dtype)


def apply_outlier_passthrough(t: Tensor, outlier_entry: dict[str, Tensor] | None) -> Tensor:
    if not isinstance(outlier_entry, dict):
        return t
    indices = outlier_entry.get("indices")
    values = outlier_entry.get("values")
    if not isinstance(indices, Tensor) or not isinstance(values, Tensor) or indices.numel() == 0:
        return t
    out = t.contiguous()
    flat = out.reshape(-1)
    flat[indices.to(dtype=torch.int64)] = values.to(dtype=out.dtype)
    return flat.view_as(out).contiguous()


def dequantize_quantized_tensor(q: Tensor, s: Tensor, dtype: torch.dtype, qmeta_entry: dict[str, object] | None = None) -> Tensor:
    scheme = str((qmeta_entry or {}).get("scheme", "")).strip().lower()
    if scheme == "blockwise":
        block_raw = (qmeta_entry or {}).get("block", (16, 16))
        if isinstance(block_raw, (list, tuple)) and len(block_raw) == 2:
            block = (max(int(block_raw[0]), 1), max(int(block_raw[1]), 1))
        else:
            block = (16, 16)
        return dequantize_blockwise_int8_tensor(q, s, dtype, block, qmeta_entry)
    return dequantize_int8_tensor(q, s, dtype)


def quantize_mixed_precision_matrix_rows(t32: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    return quantize_tensor_calibrated(t32, bits=bits)


def quantize_mixed_precision_tensor(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    if t.ndim == 2:
        return quantize_mixed_precision_matrix_rows(t.float(), bits)
    return quantize_tensor_calibrated(t, bits=bits)


def resolve_mixed_precision_target_bits(name: str, bits: int, high_bits: int, high_precision_names: set[str]) -> int:
    target_class = _classify_quant_target(name)
    if target_class == "mlp":
        target_bits = MIXED_PRECISION_EXPORT_MLP_BITS if MIXED_PRECISION_EXPORT_MLP_BITS > 0 else bits
    elif target_class == "attn":
        target_bits = MIXED_PRECISION_EXPORT_ATTN_BITS if MIXED_PRECISION_EXPORT_ATTN_BITS > 0 else bits
    elif target_class == "embed":
        target_bits = 8
    else:
        target_bits = 8
    if name in high_precision_names:
        target_bits = high_bits
    return int(target_bits)


def _classify_quant_target(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or ".proj." in name:
        return "attn"
    return "other"


def compressed_quantized_tensor_nbytes(
    q: Tensor, s: Tensor, codec: str, zstd_level: int, extra_payload: dict[str, object] | None = None
) -> int:
    buf = io.BytesIO()
    payload: dict[str, object] = {"q": q, "s": s}
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, buf)
    blob, _ = compress_blob(buf.getvalue(), codec, zstd_level)
    return int(len(blob))


def compressed_serialized_payload_nbytes(payload: dict[str, object], codec: str, zstd_level: int) -> int:
    buf = io.BytesIO()
    torch.save(payload, buf)
    blob, _ = compress_blob(buf.getvalue(), codec, zstd_level)
    return int(len(blob))


def relative_mse(reference: Tensor, approx: Tensor) -> float:
    diff = approx.float() - reference.float()
    mse = float(diff.square().mean().item()) if diff.numel() else 0.0
    ref = float(reference.float().square().mean().item()) if reference.numel() else 0.0
    return mse / max(ref, 1e-12)


def export_candidate_proxy_gain(name: str, rel_mse_gain: float, numel: int) -> float:
    base_gain = max(float(rel_mse_gain), 0.0) * max(int(numel), 1)
    if EXPORT_TENSOR_SWEEP_OBJECTIVE != "proxy_score_per_compressed_byte":
        return base_gain
    target_class = _classify_quant_target(name)
    class_weight = {"embed": 1.15, "attn": 1.0, "mlp": 0.9}.get(target_class, 0.8)
    return base_gain * class_weight


def summarize_export_candidate(candidate: dict[str, object]) -> dict[str, object]:
    summary = {
        "name": str(candidate["name"]),
        "numel": int(candidate["numel"]),
        "export_kind": str(candidate.get("export_kind", "quantized")),
        "scheme": str(candidate.get("scheme", "per_tensor")),
        "bits": int(candidate.get("bits", 0)),
        "rel_mse": float(candidate.get("rel_mse", 0.0)),
        "compressed_nbytes": int(candidate.get("compressed_nbytes", 0)),
        "payload_bytes": int(candidate.get("payload_bytes", 0)),
    }
    if "scale_scheme" in candidate:
        summary["scale_scheme"] = str(candidate["scale_scheme"])
    if "requested_density" in candidate:
        summary["requested_density"] = float(candidate["requested_density"])
    if "kept_density" in candidate:
        summary["kept_density"] = float(candidate["kept_density"])
    if "outlier_count" in candidate:
        summary["outlier_count"] = int(candidate["outlier_count"])
    return summary


def candidate_delta_metrics(name: str, baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, float]:
    baseline_rel_mse = float(baseline.get("rel_mse", 0.0))
    candidate_rel_mse = float(candidate.get("rel_mse", 0.0))
    delta_rel_mse = baseline_rel_mse - candidate_rel_mse
    compressed_delta = int(candidate.get("compressed_nbytes", 0)) - int(baseline.get("compressed_nbytes", 0))
    extra_bytes = max(compressed_delta, 0)
    bytes_saved = max(-compressed_delta, 0)
    proxy_gain = export_candidate_proxy_gain(name, delta_rel_mse, int(candidate.get("numel", baseline.get("numel", 0))))
    return {
        "delta_rel_mse": float(delta_rel_mse),
        "extra_rel_mse": float(max(candidate_rel_mse - baseline_rel_mse, 0.0)),
        "compressed_byte_delta": int(compressed_delta),
        "extra_bytes": int(extra_bytes),
        "bytes_saved": int(bytes_saved),
        "proxy_gain": float(proxy_gain),
        "benefit_per_extra_byte": float(proxy_gain / max(extra_bytes, 1)) if extra_bytes > 0 else 0.0,
        "loss_per_byte_saved": float(max(-proxy_gain, 0.0) / max(bytes_saved, 1)) if bytes_saved > 0 else 0.0,
    }


def build_quantized_export_candidate(
    name: str,
    t: Tensor,
    bits: int,
    codec: str,
    zstd_level: int,
    *,
    block: tuple[int, int] | None = None,
    keep_density: float = 1.0,
) -> dict[str, object]:
    t_cpu = t.detach().to("cpu").contiguous()
    if block is None:
        q, s = quantize_mixed_precision_tensor(t_cpu, bits)
        qmeta = {"scheme": "per_row" if s.ndim > 0 else "per_tensor", "axis": 0, "bits": int(bits)}
        deq = dequantize_quantized_tensor(q, s, t_cpu.dtype, qmeta)
        return {
            "name": str(name),
            "numel": int(t_cpu.numel()),
            "export_kind": "quantized",
            "scheme": str(qmeta["scheme"]),
            "bits": int(bits),
            "rel_mse": float(relative_mse(t_cpu, deq)),
            "compressed_nbytes": int(compressed_quantized_tensor_nbytes(q, s, codec, zstd_level)),
            "payload_bytes": int(tensor_nbytes(q) + tensor_nbytes(s)),
            "q": q,
            "s": s,
            "qmeta": qmeta,
            "outliers": None,
        }

    quant_input, outlier_passthrough, actual_density = extract_blockwise_outliers_2d(t_cpu, keep_density, block)
    q_blockwise, s_blockwise_plain = quantize_mixed_precision_blockwise_2d(quant_input, bits, block)
    scale_payloads: list[tuple[Tensor, dict[str, object]]] = [(s_blockwise_plain, {})]
    if s_blockwise_plain.ndim == 2 and s_blockwise_plain.numel() > 0:
        s_nested_q, s_nested_s = quantize_tensor_calibrated(s_blockwise_plain.float(), bits=8)
        scale_payloads.append(
            (
                s_nested_q.contiguous(),
                {
                    "scale_scheme": "nested_per_row",
                    "nested_scale_bits": 8,
                    "nested_scale_scales": s_nested_s.contiguous(),
                },
            )
        )
    best_candidate: dict[str, object] | None = None
    for scale_payload, scale_meta in scale_payloads:
        qmeta_entry = {
            "scheme": "blockwise",
            "axis": 0,
            "bits": int(bits),
            "block": [int(block[0]), int(block[1])],
            **scale_meta,
        }
        extra_payload: dict[str, object] = {}
        payload_bytes = tensor_nbytes(q_blockwise) + tensor_nbytes(scale_payload)
        if "nested_scale_scales" in scale_meta:
            extra_payload["nested_scale_scales"] = scale_meta["nested_scale_scales"]
            payload_bytes += tensor_nbytes(scale_meta["nested_scale_scales"])
        if outlier_passthrough is not None:
            extra_payload["outliers"] = outlier_passthrough
            qmeta_entry["outlier_count"] = int(outlier_passthrough["indices"].numel())
            payload_bytes += tensor_nbytes(outlier_passthrough["indices"]) + tensor_nbytes(outlier_passthrough["values"])
        compressed_candidate = compressed_quantized_tensor_nbytes(
            q_blockwise, scale_payload, codec, zstd_level, extra_payload=extra_payload
        )
        deq_candidate = dequantize_quantized_tensor(q_blockwise, scale_payload, t_cpu.dtype, qmeta_entry)
        deq_candidate = apply_outlier_passthrough(deq_candidate, outlier_passthrough)
        candidate = {
            "name": str(name),
            "numel": int(t_cpu.numel()),
            "export_kind": "quantized",
            "scheme": "blockwise",
            "bits": int(bits),
            "scale_scheme": str(scale_meta.get("scale_scheme", "fp16_block")),
            "requested_density": float(keep_density),
            "kept_density": float(actual_density),
            "outlier_count": int(0 if outlier_passthrough is None else outlier_passthrough["indices"].numel()),
            "rel_mse": float(relative_mse(t_cpu, deq_candidate)),
            "compressed_nbytes": int(compressed_candidate),
            "payload_bytes": int(payload_bytes),
            "q": q_blockwise,
            "s": scale_payload,
            "qmeta": qmeta_entry,
            "outliers": outlier_passthrough,
        }
        if best_candidate is None or (
            int(candidate["compressed_nbytes"]),
            float(candidate["rel_mse"]),
        ) < (
            int(best_candidate["compressed_nbytes"]),
            float(best_candidate["rel_mse"]),
        ):
            best_candidate = candidate
    if best_candidate is None:
        raise RuntimeError(f"failed to build blockwise candidate for {name}")
    return best_candidate


def build_float_export_candidate(name: str, t: Tensor, codec: str, zstd_level: int) -> dict[str, object]:
    t_cpu = t.detach().to("cpu").contiguous()
    passthrough_orig_dtypes: dict[str, str] = {}
    kept = keep_float_tensor(name, t_cpu, passthrough_orig_dtypes)
    restored = kept.detach().to("cpu").contiguous()
    orig_dtype = passthrough_orig_dtypes.get(name)
    if isinstance(orig_dtype, str):
        restored = restored.to(dtype=getattr(torch, orig_dtype)).contiguous()
    payload: dict[str, object] = {"value": kept}
    if isinstance(orig_dtype, str):
        payload["orig_dtype"] = orig_dtype
    return {
        "name": str(name),
        "numel": int(t_cpu.numel()),
        "export_kind": "float",
        "scheme": "float",
        "bits": int(16 if kept.dtype == torch.float16 else 32),
        "rel_mse": float(relative_mse(t_cpu, restored)),
        "compressed_nbytes": int(compressed_serialized_payload_nbytes(payload, codec, zstd_level)),
        "payload_bytes": int(tensor_nbytes(kept)),
        "kept": kept,
        "passthrough_orig_dtype": orig_dtype,
    }


def build_tensor_export_candidates(
    name: str,
    t: Tensor,
    default_bits: int,
    codec: str,
    zstd_level: int,
    *,
    extra_bits: tuple[int, ...] = (),
    include_float: bool = False,
    include_blockwise: bool = True,
) -> list[dict[str, object]]:
    candidate_bits = tuple(sorted(set((int(default_bits), *EXPORT_TENSOR_CANDIDATE_BITS, *extra_bits))))
    candidates: list[dict[str, object]] = []
    for bits in candidate_bits:
        candidates.append(build_quantized_export_candidate(name, t, bits, codec, zstd_level))
        blockwise_allowed = (
            include_blockwise
            and EXPORT_BLOCK_PRUNE_ENABLED
            and bool(EXPORT_BLOCK_PRUNE_DENSITIES)
            and t.ndim == 2
            and t.numel() >= max(EXPORT_BLOCK_PRUNE_MIN_NUMEL, EXPORT_TENSOR_SWEEP_MIN_NUMEL)
            and _classify_quant_target(name) in {"mlp", "attn"}
        )
        if blockwise_allowed:
            for keep_density in tuple(sorted(set((1.0, *EXPORT_BLOCK_PRUNE_DENSITIES)), reverse=True)):
                candidates.append(
                    build_quantized_export_candidate(
                        name,
                        t,
                        bits,
                        codec,
                        zstd_level,
                        block=EXPORT_BLOCK_PRUNE_BLOCK,
                        keep_density=keep_density,
                    )
                )
    if include_float:
        candidates.append(build_float_export_candidate(name, t, codec, zstd_level))
    deduped: dict[tuple[object, ...], dict[str, object]] = {}
    for candidate in candidates:
        key = (
            candidate.get("export_kind"),
            candidate.get("scheme"),
            int(candidate.get("bits", 0)),
            float(candidate.get("requested_density", 1.0)),
            str(candidate.get("scale_scheme", "")),
        )
        existing = deduped.get(key)
        if existing is None or (
            int(candidate["compressed_nbytes"]),
            float(candidate["rel_mse"]),
        ) < (
            int(existing["compressed_nbytes"]),
            float(existing["rel_mse"]),
        ):
            deduped[key] = candidate
    return list(deduped.values())


def select_default_export_candidate(
    name: str, baseline: dict[str, object], candidates: list[dict[str, object]]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    ranked: list[dict[str, object]] = []
    best = baseline
    for candidate in candidates:
        if candidate is baseline or candidate.get("export_kind") != "quantized":
            continue
        delta = candidate_delta_metrics(name, baseline, candidate)
        if int(delta["bytes_saved"]) <= 0 or float(delta["extra_rel_mse"]) > EXPORT_TENSOR_SWEEP_MAX_EXTRA_REL_MSE:
            continue
        entry = {**summarize_export_candidate(candidate), **delta}
        ranked.append(entry)
        if best is baseline or (
            int(entry["bytes_saved"]),
            -float(entry["extra_rel_mse"]),
            -float(entry["rel_mse"]),
        ) > (
            int(candidate_delta_metrics(name, baseline, best)["bytes_saved"]),
            -float(candidate_delta_metrics(name, baseline, best)["extra_rel_mse"]),
            -float(best["rel_mse"]),
        ):
            best = candidate
    ranked.sort(
        key=lambda item: (
            int(item["bytes_saved"]),
            -float(item["extra_rel_mse"]),
            -float(item["rel_mse"]),
            int(item["numel"]),
        ),
        reverse=True,
    )
    return best, ranked


def compress_blob(payload: bytes, codec: str, zstd_level: int) -> tuple[bytes, str]:
    normalized = codec.strip().lower()
    if normalized == "zstd" and HAS_ZSTD:
        return zstandard.ZstdCompressor(level=zstd_level).compress(payload), "zstd"
    return zlib.compress(payload, level=9), "zlib"


def decompress_blob(payload: bytes, codec: str) -> bytes:
    if codec == "zstd":
        if not HAS_ZSTD:
            raise RuntimeError("EXPORT_CODEC=zstd requested but zstandard is not installed")
        return zstandard.ZstdDecompressor().decompress(payload)
    return zlib.decompress(payload)


def select_auto_keep_float_tensors(state_dict: dict[str, Tensor]) -> tuple[set[str], dict[str, object]]:
    budget_bytes = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_budget_bytes"])
    max_tensors = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_max_tensors"])
    min_numel = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_min_numel"])
    log_topk = int(ACTIVE_EXPORT_CONFIG["int8_sensitivity_log_topk"])
    enabled = budget_bytes > 0 or max_tensors > 0
    info: dict[str, object] = {
        "enabled": enabled,
        "budget_bytes": int(budget_bytes),
        "max_tensors": int(max_tensors),
        "min_numel": int(min_numel),
        "selected_names": [],
        "selected_count": 0,
        "selected_extra_bytes": 0,
        "top_candidates": [],
    }
    if not enabled:
        return set(), info

    candidates: list[dict[str, object]] = []
    codec = str(ACTIVE_EXPORT_CONFIG["export_codec"])
    zstd_level = int(ACTIVE_EXPORT_CONFIG["export_zstd_level"])
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if not t.is_floating_point() or should_keep_float_tensor(name, t) or t.numel() < min_numel:
            continue
        q_bits = LARGE_MATRIX_QUANT_BITS if t.ndim == 2 else 8
        baseline = build_quantized_export_candidate(name, t, q_bits, codec, zstd_level)
        float_candidate = build_float_export_candidate(name, t, codec, zstd_level)
        delta = candidate_delta_metrics(name, baseline, float_candidate)
        if int(delta["extra_bytes"]) <= 0 or float(delta["proxy_gain"]) <= 0.0:
            continue
        max_abs_diff = float((float_candidate["kept"].float() - t.float()).abs().max().item()) if t.numel() else 0.0
        candidates.append(
            {
                "name": str(name),
                "numel": int(t.numel()),
                "keep_bytes": int(float_candidate["payload_bytes"]),
                "quant_bytes": int(baseline["payload_bytes"]),
                "extra_bytes": int(delta["extra_bytes"]),
                "compressed_byte_delta": int(delta["compressed_byte_delta"]),
                "rel_mse": float(baseline["rel_mse"]),
                "rel_mse_gain": float(delta["delta_rel_mse"]),
                "max_abs_diff": max_abs_diff,
                "benefit": float(delta["proxy_gain"]),
                "benefit_per_extra_byte": float(delta["benefit_per_extra_byte"]),
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item["benefit_per_extra_byte"]),
            float(item["benefit"]),
            float(item["rel_mse"]),
            int(item["numel"]),
        ),
        reverse=True,
    )
    selected_names: list[str] = []
    used_extra_bytes = 0
    for item in ranked:
        if max_tensors > 0 and len(selected_names) >= max_tensors:
            break
        extra_bytes = int(item["extra_bytes"])
        if budget_bytes > 0 and used_extra_bytes + extra_bytes > budget_bytes:
            continue
        selected_names.append(str(item["name"]))
        used_extra_bytes += extra_bytes

    info["selected_names"] = selected_names
    info["selected_count"] = len(selected_names)
    info["selected_extra_bytes"] = int(used_extra_bytes)
    info["top_candidates"] = [
        {
            "name": str(item["name"]),
            "numel": int(item["numel"]),
            "extra_bytes": int(item["extra_bytes"]),
            "rel_mse": float(item["rel_mse"]),
            "max_abs_diff": float(item["max_abs_diff"]),
            "benefit_per_extra_byte": float(item["benefit_per_extra_byte"]),
        }
        for item in ranked[: max(log_topk, 0)]
    ]
    return set(selected_names), info


def select_auto_keep_float_row_groups(
    state_dict: dict[str, Tensor], passthrough_orig_dtypes: dict[str, str]
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, object]]:
    budget_bytes = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_row_budget_bytes"])
    group_size = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_row_group_size"])
    enabled = budget_bytes > 0 and group_size > 0
    info: dict[str, object] = {
        "enabled": enabled,
        "budget_bytes": int(budget_bytes),
        "row_group_size": int(group_size),
        "selected_count": 0,
        "selected_extra_bytes": 0,
        "selected_groups": [],
    }
    if not enabled:
        return {}, info

    candidates: list[dict[str, object]] = []
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if (
            not t.is_floating_point()
            or should_keep_float_tensor(name, t)
            or t.ndim != 2
            or t.shape[0] < group_size
        ):
            continue
        candidates.extend(quantize_tensor_row_groups(name, t, group_size, passthrough_orig_dtypes))

    ranked = sorted(
        candidates,
        key=lambda item: (float(item["benefit_per_extra_byte"]), float(item["rel_mse"])),
        reverse=True,
    )
    selected: dict[str, list[tuple[int, int]]] = {}
    used_bytes = 0
    seen: set[tuple[str, int]] = set()
    for item in ranked:
        key = (str(item["name"]), int(item["row_start"]))
        if key in seen:
            continue
        extra = int(item["extra_bytes"])
        if used_bytes + extra > budget_bytes:
            continue
        seen.add(key)
        used_bytes += extra
        selected.setdefault(str(item["name"]), []).append((int(item["row_start"]), int(item["row_end"])))
    for name in selected:
        selected[name].sort()
    info["selected_count"] = sum(len(v) for v in selected.values())
    info["selected_extra_bytes"] = int(used_bytes)
    info["selected_groups"] = [
        {"name": name, "row_start": row_start, "row_end": row_end}
        for name, groups in selected.items()
        for row_start, row_end in groups
    ]
    return selected, info


def _equalization_channel_metric(t: Tensor, dim: int) -> Tensor:
    tf = t.detach().to(dtype=torch.float32)
    rms = tf.square().mean(dim=dim).add_(1e-12).sqrt()
    peak = tf.abs().amax(dim=dim).clamp_min_(1e-12)
    return (rms * peak).sqrt()


def _scaled_rows(weight: Tensor, scales: Tensor) -> Tensor:
    scaled = weight.detach().to(dtype=torch.float32, device="cpu") * scales[:, None].to(dtype=torch.float32, device="cpu")
    return scaled.to(dtype=weight.dtype).contiguous()


def _scaled_cols(weight: Tensor, scales: Tensor) -> Tensor:
    scaled = weight.detach().to(dtype=torch.float32, device="cpu") * scales[None, :].to(dtype=torch.float32, device="cpu")
    return scaled.to(dtype=weight.dtype).contiguous()


def _attention_equalization_layout(
    prefix: str, state_dict: dict[str, Tensor], v_channels: int, proj_in_dim: int
) -> tuple[int, int, int, int] | None:
    q_gain = state_dict.get(f"{prefix}.q_gain")
    if isinstance(q_gain, Tensor) and q_gain.ndim == 1:
        num_heads = int(q_gain.numel())
        if num_heads > 0 and proj_in_dim % num_heads == 0:
            head_dim = proj_in_dim // num_heads
            if head_dim > 0 and v_channels % head_dim == 0:
                num_kv_heads = v_channels // head_dim
                if num_kv_heads > 0 and num_heads % num_kv_heads == 0:
                    return num_heads, num_kv_heads, head_dim, num_heads // num_kv_heads
    if v_channels == proj_in_dim:
        return v_channels, v_channels, 1, 1
    return None


def _attention_equalization_scales(prefix: str, state_dict: dict[str, Tensor], v_weight: Tensor, proj_weight: Tensor) -> Tensor | None:
    if v_weight.ndim != 2 or proj_weight.ndim != 2:
        return None
    v_channels = int(v_weight.shape[0])
    proj_in_dim = int(proj_weight.shape[1])
    layout = _attention_equalization_layout(prefix, state_dict, v_channels, proj_in_dim)
    if layout is None:
        return None
    num_heads, num_kv_heads, head_dim, repeat = layout
    if v_channels != num_kv_heads * head_dim or proj_in_dim != num_heads * head_dim:
        return None
    producer = _equalization_channel_metric(v_weight, dim=1).view(num_kv_heads, head_dim)
    proj_metric = _equalization_channel_metric(proj_weight, dim=0).view(num_heads, head_dim)
    if repeat > 1:
        consumer = proj_metric.view(num_kv_heads, repeat, head_dim).square().mean(dim=1).sqrt()
    else:
        consumer = proj_metric.view(num_kv_heads, head_dim)
    gain = (consumer / producer.clamp_min(1e-12)).clamp_min(1e-12).sqrt()
    min_gain = 1.0 / EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN
    return gain.clamp(min=min_gain, max=EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN).reshape(-1)


def _expand_attention_proj_scales(
    prefix: str, state_dict: dict[str, Tensor], base_scales: Tensor, proj_in_dim: int
) -> Tensor | None:
    layout = _attention_equalization_layout(prefix, state_dict, int(base_scales.numel()), proj_in_dim)
    if layout is None:
        return None
    num_heads, num_kv_heads, head_dim, repeat = layout
    if int(base_scales.numel()) != num_kv_heads * head_dim or proj_in_dim != num_heads * head_dim:
        return None
    return base_scales.view(num_kv_heads, head_dim).repeat_interleave(repeat, dim=0).reshape(-1)


def canonicalize_state_dict_for_export(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    if not EXPORT_CHANNEL_EQUALIZATION:
        return state_dict
    canonicalized = dict(state_dict)
    for name, tensor in state_dict.items():
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point() or tensor.ndim != 2:
            continue
        if name.endswith(".c_v.weight"):
            prefix = name[: -len(".c_v.weight")]
            proj_name = f"{prefix}.proj.weight"
            if (
                proj_name not in state_dict
                or f"{prefix}.inner_attn_norm.weight" in state_dict
                or f"{prefix}.inner_attn_norm.bias" in state_dict
            ):
                continue
            scales = _attention_equalization_scales(prefix, state_dict, tensor, state_dict[proj_name])
            proj_scales = None if scales is None else _expand_attention_proj_scales(prefix, state_dict, scales, int(state_dict[proj_name].shape[1]))
            if scales is None or proj_scales is None:
                continue
            canonicalized[name] = _scaled_rows(tensor, scales)
            canonicalized[proj_name] = _scaled_cols(state_dict[proj_name], proj_scales.reciprocal())
        elif name.endswith(".fc.weight"):
            prefix = name[: -len(".fc.weight")]
            proj_name = f"{prefix}.proj.weight"
            if (
                proj_name not in state_dict
                or f"{prefix}.inner_norm.weight" in state_dict
                or f"{prefix}.inner_norm.bias" in state_dict
            ):
                continue
            proj_weight = state_dict[proj_name]
            if proj_weight.ndim != 2 or int(proj_weight.shape[1]) != int(tensor.shape[0]):
                continue
            producer = _equalization_channel_metric(tensor, dim=1)
            consumer = _equalization_channel_metric(proj_weight, dim=0)
            gain = (consumer / producer.clamp_min(1e-12)).clamp_min(1e-12).sqrt()
            min_gain = 1.0 / EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN
            gain = gain.clamp(min=min_gain, max=EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN)
            canonicalized[name] = _scaled_rows(tensor, gain)
            canonicalized[proj_name] = _scaled_cols(proj_weight, gain.square().reciprocal())
    return canonicalized


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    auto_keep_names, auto_keep_info = select_auto_keep_float_tensors(state_dict)
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    row_group_passthrough: dict[str, list[dict[str, object]]] = {}
    qmeta: dict[str, dict[str, object]] = {}
    row_keep_groups, row_keep_info = select_auto_keep_float_row_groups(state_dict, passthrough_orig_dtypes)
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
            "auto_keep_count",
            "auto_keep_extra_bytes",
            "auto_keep_row_group_count",
            "auto_keep_row_group_extra_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if name in auto_keep_names:
            stats["auto_keep_count"] += 1
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if should_keep_float_tensor(name, t):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if name in row_keep_groups:
            entries: list[dict[str, object]] = []
            for row_start, row_end in row_keep_groups[name]:
                kept = keep_float_tensor(name, t[row_start:row_end].contiguous(), passthrough_orig_dtypes)
                entries.append({"row_start": row_start, "row_end": row_end, "value": kept})
                stats["auto_keep_row_group_count"] += 1
                stats["int8_payload_bytes"] += tensor_nbytes(kept)
            row_group_passthrough[name] = entries
            qmeta[name] = {
                "scheme": "per_row",
                "axis": 0,
                "row_passthrough": [(entry["row_start"], entry["row_end"]) for entry in entries],
            }
        if s.ndim > 0:
            qmeta.setdefault(name, {"scheme": "per_row", "axis": 0})
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if row_group_passthrough:
        obj["row_group_passthrough"] = row_group_passthrough
    stats["auto_keep_extra_bytes"] = int(auto_keep_info["selected_extra_bytes"])
    stats["auto_keep_row_group_extra_bytes"] = int(row_keep_info["selected_extra_bytes"])
    return obj, stats, {**auto_keep_info, "row_groups": row_keep_info}

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    row_group_passthrough = obj.get("row_group_passthrough", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        out[name] = dequantize_quantized_tensor(q, s, dtype, qmeta.get(name, {}))
        if name in row_group_passthrough:
            for entry in row_group_passthrough[name]:
                row_start = int(entry["row_start"])
                row_end = int(entry["row_end"])
                patch = entry["value"].detach().to("cpu").contiguous()
                orig_dtype = passthrough_orig_dtypes.get(name)
                if isinstance(orig_dtype, str):
                    patch = patch.to(dtype=getattr(torch, orig_dtype)).contiguous()
                out[name][row_start:row_end] = patch.to(dtype=out[name].dtype)
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def quantize_state_dict_mixed_precision(state_dict: dict[str, Tensor], bits: int):
    high_bits = min(max(int(ACTIVE_EXPORT_CONFIG["export_high_precision_bits"]), bits), 8)
    codec = str(ACTIVE_EXPORT_CONFIG["export_codec"])
    zstd_level = int(ACTIVE_EXPORT_CONFIG["export_zstd_level"])
    high_precision_budget_bytes = int(ACTIVE_EXPORT_CONFIG["export_high_precision_budget_bytes"])
    high_precision_max_tensors = int(ACTIVE_EXPORT_CONFIG["export_high_precision_max_tensors"])
    high_precision_min_numel = int(ACTIVE_EXPORT_CONFIG["export_high_precision_min_numel"])
    target_bits_by_name: dict[str, int] = {}
    selected_base_candidates: dict[str, dict[str, object]] = {}
    selected_upgrade_candidates: dict[str, dict[str, object]] = {}
    tensor_sweep_selected: list[dict[str, object]] = []
    upgrade_ranked: list[tuple[dict[str, object], dict[str, object]]] = []
    tensor_sweep_info: dict[str, object] = {
        "enabled": bool(EXPORT_TENSOR_CANDIDATE_BITS),
        "objective": EXPORT_TENSOR_SWEEP_OBJECTIVE,
        "candidate_bits": [int(candidate_bits) for candidate_bits in EXPORT_TENSOR_CANDIDATE_BITS],
        "min_numel": int(EXPORT_TENSOR_SWEEP_MIN_NUMEL),
        "max_extra_rel_mse": float(EXPORT_TENSOR_SWEEP_MAX_EXTRA_REL_MSE),
        "selected_names": [],
        "selected_count": 0,
        "compressed_bytes_saved": 0,
        "top_candidates": [],
    }
    high_precision_info: dict[str, object] = {
        "enabled": high_bits > 0 and (high_precision_budget_bytes > 0 or high_precision_max_tensors > 0),
        "low_bits": int(bits),
        "high_bits": int(high_bits),
        "objective": EXPORT_TENSOR_SWEEP_OBJECTIVE,
        "candidate_bits": [int(candidate_bits) for candidate_bits in sorted(set((*EXPORT_TENSOR_CANDIDATE_BITS, high_bits)))],
        "budget_bytes": int(high_precision_budget_bytes),
        "max_tensors": int(high_precision_max_tensors),
        "min_numel": int(high_precision_min_numel),
        "selected_names": [],
        "selected_count": 0,
        "selected_extra_bytes": 0,
        "top_candidates": [],
    }
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if not t.is_floating_point():
            continue
        if (
            should_keep_float_tensor(name, t)
            or any(pattern in name for pattern in MIXED_PRECISION_KEEP_FP16_NAME_PATTERNS)
            or (_classify_quant_target(name) == "embed" and FORCE_FP16_TIED_EMBED_EXPORT)
        ):
            continue
        target_bits = resolve_mixed_precision_target_bits(name, bits, high_bits, set())
        target_bits_by_name[name] = target_bits
        if (
            _classify_quant_target(name) not in {"mlp", "attn"}
            or t.numel() < EXPORT_TENSOR_SWEEP_MIN_NUMEL
        ):
            continue
        baseline = build_quantized_export_candidate(name, t, target_bits, codec, zstd_level)
        candidates = build_tensor_export_candidates(
            name,
            t,
            target_bits,
            codec,
            zstd_level,
            extra_bits=(high_bits,),
            include_float=False,
            include_blockwise=True,
        )
        selected_base, ranked_defaults = select_default_export_candidate(name, baseline, candidates)
        if selected_base is not baseline:
            selected_base_candidates[name] = selected_base
            target_bits_by_name[name] = int(selected_base["bits"])
            selected_entry = {**summarize_export_candidate(selected_base), **candidate_delta_metrics(name, baseline, selected_base)}
            tensor_sweep_selected.append(selected_entry)
        if high_precision_info["enabled"] and t.numel() >= high_precision_min_numel:
            for candidate in candidates:
                if candidate is selected_base:
                    continue
                delta = candidate_delta_metrics(name, selected_base, candidate)
                if int(delta["extra_bytes"]) <= 0 or float(delta["proxy_gain"]) <= 0.0:
                    continue
                entry = {**summarize_export_candidate(candidate), **delta}
                upgrade_ranked.append((entry, candidate))
        if ranked_defaults:
            tensor_sweep_info["top_candidates"].extend(ranked_defaults[:1])

    upgrade_ranked.sort(
        key=lambda item: (
            float(item[0]["benefit_per_extra_byte"]),
            float(item[0]["delta_rel_mse"]),
            -int(item[0]["extra_bytes"]),
            int(item[0]["numel"]),
        ),
        reverse=True,
    )
    selected_upgrade_names: list[str] = []
    used_extra_bytes = 0
    if high_precision_info["enabled"]:
        for entry, candidate in upgrade_ranked:
            name = str(entry["name"])
            if name in selected_upgrade_candidates:
                continue
            if high_precision_max_tensors > 0 and len(selected_upgrade_names) >= high_precision_max_tensors:
                break
            extra_bytes = int(entry["extra_bytes"])
            if high_precision_budget_bytes > 0 and used_extra_bytes + extra_bytes > high_precision_budget_bytes:
                continue
            selected_upgrade_candidates[name] = candidate
            selected_upgrade_names.append(name)
            used_extra_bytes += extra_bytes

    tensor_sweep_selected.sort(
        key=lambda item: (
            int(item["bytes_saved"]),
            -float(item["extra_rel_mse"]),
            -float(item["rel_mse"]),
            int(item["numel"]),
        ),
        reverse=True,
    )
    tensor_sweep_info["selected_names"] = [str(item["name"]) for item in tensor_sweep_selected]
    tensor_sweep_info["selected_count"] = len(tensor_sweep_selected)
    tensor_sweep_info["compressed_bytes_saved"] = int(sum(int(item["bytes_saved"]) for item in tensor_sweep_selected))
    tensor_sweep_info["top_candidates"] = tensor_sweep_selected[: max(EXPORT_TENSOR_SWEEP_LOG_TOPK, 0)]
    high_precision_info["selected_names"] = selected_upgrade_names
    high_precision_info["selected_count"] = len(selected_upgrade_names)
    high_precision_info["selected_extra_bytes"] = int(used_extra_bytes)
    high_precision_info["top_candidates"] = [entry for entry, _candidate in upgrade_ranked[: max(EXPORT_TENSOR_SWEEP_LOG_TOPK, 0)]]

    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    outlier_passthrough: dict[str, dict[str, Tensor]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "payload_bytes",
            "high_precision_tensor_count",
            "high_precision_extra_bytes",
            "block_pruned_tensor_count",
            "block_prune_estimated_bytes_saved",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["payload_bytes"] += tensor_nbytes(t)
            continue
        if (
            should_keep_float_tensor(name, t)
            or any(pattern in name for pattern in MIXED_PRECISION_KEEP_FP16_NAME_PATTERNS)
            or (_classify_quant_target(name) == "embed" and FORCE_FP16_TIED_EMBED_EXPORT)
        ):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        target_bits = int(target_bits_by_name.get(name, resolve_mixed_precision_target_bits(name, bits, high_bits, set())))
        selected_candidate = selected_upgrade_candidates.get(name) or selected_base_candidates.get(name)
        if name in selected_upgrade_candidates:
            stats["high_precision_tensor_count"] += 1
        if selected_candidate is not None and str(selected_candidate.get("scheme", "")) == "blockwise":
            stats["block_pruned_tensor_count"] += 1
        if selected_candidate is not None and selected_candidate.get("export_kind") == "quantized":
            q = selected_candidate["q"]
            s = selected_candidate["s"]
            qmeta[name] = dict(selected_candidate["qmeta"])
            if selected_candidate.get("outliers") is not None:
                outlier_passthrough[name] = selected_candidate["outliers"]
        else:
            q, s = quantize_mixed_precision_tensor(t, target_bits)
            qmeta[name] = {"scheme": "per_row" if s.ndim > 0 else "per_tensor", "axis": 0, "bits": int(target_bits)}
        if selected_candidate is not None and str(selected_candidate.get("scheme", "")) == "blockwise":
            qmeta[name]["block_prune"] = {
                "scheme": str(selected_candidate.get("scheme", "blockwise")),
                "scale_scheme": str(selected_candidate.get("scale_scheme", "fp16_block")),
                "block": [int(EXPORT_BLOCK_PRUNE_BLOCK[0]), int(EXPORT_BLOCK_PRUNE_BLOCK[1])],
                "requested_density": float(selected_candidate.get("requested_density", 1.0)),
                "kept_density": float(selected_candidate.get("kept_density", 1.0)),
                "outlier_count": int(selected_candidate.get("outlier_count", 0)),
            }
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
        nested_scale_scales = qmeta[name].get("nested_scale_scales")
        if isinstance(nested_scale_scales, Tensor):
            stats["payload_bytes"] += tensor_nbytes(nested_scale_scales)
        if name in outlier_passthrough:
            stats["payload_bytes"] += tensor_nbytes(outlier_passthrough[name]["indices"])
            stats["payload_bytes"] += tensor_nbytes(outlier_passthrough[name]["values"])

    stats["high_precision_extra_bytes"] = int(high_precision_info["selected_extra_bytes"])
    stats["block_prune_estimated_bytes_saved"] = int(tensor_sweep_info["compressed_bytes_saved"])
    obj = {
        "__quant_format__": f"mixed_precision_{bits}bit_v2",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
        "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    if outlier_passthrough:
        obj["outlier_passthrough"] = outlier_passthrough
    return obj, stats, {**high_precision_info, "tensor_sweep": tensor_sweep_info, "block_prune": tensor_sweep_info}


def dequantize_state_dict_mixed_precision(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    qmeta = obj.get("qmeta", {})
    outlier_passthrough = obj.get("outlier_passthrough", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        out[name] = dequantize_quantized_tensor(q, s, dtype, qmeta.get(name, {}))
        out[name] = apply_outlier_passthrough(out[name], outlier_passthrough.get(name))
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

SHARD_HEADER_INTS = 256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_BYTES = SHARD_HEADER_INTS * np.dtype("<i4").itemsize
SHARD_TOKEN_BYTES = np.dtype("<u2").itemsize


def _read_shard_num_tokens(file: Path) -> int:
    header = np.fromfile(file, dtype="<i4", count=SHARD_HEADER_INTS)
    if header.size != SHARD_HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = SHARD_HEADER_BYTES + num_tokens * SHARD_TOKEN_BYTES
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    return num_tokens


def load_data_shard(file: Path) -> Tensor:
    num_tokens = _read_shard_num_tokens(file)
    tokens_np = np.memmap(file, dtype="<u2", mode="c", offset=SHARD_HEADER_BYTES, shape=(num_tokens,))
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np)


def _available_loader_workers() -> int:
    override = int(os.environ.get("TOKEN_LOADER_WORKERS", "0"))
    if override > 0:
        return override
    try:
        available = len(os.sched_getaffinity(0))
    except Exception:
        available = os.cpu_count() or 1
    return max(int(available), 1)


def _resolve_loader_worker_count(env_name: str) -> int:
    override = int(os.environ.get(env_name, "0"))
    if override > 0:
        return override
    return _available_loader_workers()


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str, random_offset_tokens: int = 0, seed: int = 1337):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.random_offset_tokens = max(int(random_offset_tokens), 0)
        self.rng = random.Random(seed)
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
        preload_workers = _resolve_loader_worker_count("TOKEN_LOADER_PRELOAD_WORKERS")
        self._preload_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=preload_workers) if len(self.files) > 1 else None
        )
        self._next_tokens_future: Future[Tensor] | None = None
        self._next_file_idx: int | None = None
        self.last_take_info: dict[str, object] = {
            "mode": "init",
            "file_idx": 0,
            "file": str(self.files[0]),
            "start_offset": 0,
            "requested_tokens": 0,
        }
        self._schedule_next_shard_preload()

    def __del__(self) -> None:
        if self._preload_executor is not None:
            self._preload_executor.shutdown(wait=False, cancel_futures=True)

    def state_dict(self) -> dict[str, object]:
        return {
            "file_idx": int(self.file_idx),
            "pos": int(self.pos),
            "rng_state": self.rng.getstate(),
            "last_take_info": copy.deepcopy(self.last_take_info),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.file_idx = int(state["file_idx"])
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = int(state["pos"])
        self.rng.setstate(state["rng_state"])  # type: ignore[arg-type]
        self.last_take_info = copy.deepcopy(state.get("last_take_info", self.last_take_info))
        self._next_tokens_future = None
        self._next_file_idx = None
        self._schedule_next_shard_preload()

    def _schedule_next_shard_preload(self) -> None:
        if self._preload_executor is None:
            return
        next_file_idx = (self.file_idx + 1) % len(self.files)
        if self._next_tokens_future is not None and self._next_file_idx == next_file_idx:
            return
        self._next_file_idx = next_file_idx
        self._next_tokens_future = self._preload_executor.submit(load_data_shard, self.files[next_file_idx])

    def _advance_file(self) -> None:
        next_file_idx = (self.file_idx + 1) % len(self.files)
        if self._next_tokens_future is not None and self._next_file_idx == next_file_idx:
            self.tokens = self._next_tokens_future.result()
        else:
            self.tokens = load_data_shard(self.files[next_file_idx])
        self.file_idx = next_file_idx
        self.pos = 0
        self._next_tokens_future = None
        self._next_file_idx = None
        self._schedule_next_shard_preload()

    def _skip_sequential(self, n: int) -> int:
        remaining = n
        files_touched = 0
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            self.pos += k
            remaining -= k
            files_touched += 1
        return files_touched

    def _collect_sequential(self, n: int) -> tuple[Tensor, int]:
        chunks: list[Tensor] = []
        remaining = n
        files_touched = 0
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
            files_touched += 1
        return (chunks[0] if len(chunks) == 1 else torch.cat(chunks), files_touched)

    def _take_sequential(self, n: int) -> Tensor:
        start_file_idx = self.file_idx
        start_file = str(self.files[self.file_idx])
        start_offset = self.pos
        chunk, files_touched = self._collect_sequential(n)
        self.last_take_info = {
            "mode": "sequential",
            "file_idx": int(start_file_idx),
            "file": start_file,
            "start_offset": int(start_offset),
            "requested_tokens": int(n),
            "files_touched": int(files_touched),
        }
        return chunk

    def _take_sequential_distributed(self, local_n: int, rank: int, world_size: int) -> Tensor:
        total_n = local_n * world_size
        start_file_idx = self.file_idx
        start_file = str(self.files[self.file_idx])
        start_offset = self.pos
        rank_offset = rank * local_n
        files_touched = self._skip_sequential(rank_offset)
        chunk, collected_files = self._collect_sequential(local_n)
        files_touched += collected_files
        files_touched += self._skip_sequential(total_n - rank_offset - local_n)
        self.last_take_info = {
            "mode": "sequential_distributed",
            "file_idx": int(start_file_idx),
            "file": start_file,
            "start_offset": int(start_offset),
            "requested_tokens": int(total_n),
            "returned_tokens": int(local_n),
            "rank_offset": int(rank_offset),
            "world_size": int(world_size),
            "files_touched": int(files_touched),
        }
        return chunk

    def take(self, n: int) -> Tensor:
        if self.random_offset_tokens <= 0:
            return self._take_sequential(n)

        while True:
            max_start = self.tokens.numel() - n
            if max_start < 0:
                self._advance_file()
                continue
            start_low = min(self.pos, max_start)
            start_high = min(max_start, start_low + self.random_offset_tokens)
            start = self.rng.randint(start_low, start_high) if start_high > start_low else start_low
            self.pos = start + n
            self.last_take_info = {
                "mode": "random_offset",
                "file_idx": int(self.file_idx),
                "file": str(self.files[self.file_idx]),
                "start_offset": int(start),
                "requested_tokens": int(n),
                "start_low": int(start_low),
                "start_high": int(start_high),
            }
            return self.tokens[start : start + n]

    def take_distributed(self, local_n: int, rank: int, world_size: int) -> Tensor:
        if local_n <= 0:
            raise ValueError(f"local_n must be positive, got {local_n}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must satisfy 0 <= rank < world_size, got rank={rank}, world_size={world_size}")
        if self.random_offset_tokens <= 0:
            return self._take_sequential_distributed(local_n, rank, world_size)

        total_n = local_n * world_size
        rank_offset = rank * local_n
        while True:
            max_start = self.tokens.numel() - total_n
            if max_start < 0:
                self._advance_file()
                continue
            start_low = min(self.pos, max_start)
            start_high = min(max_start, start_low + self.random_offset_tokens)
            start = self.rng.randint(start_low, start_high) if start_high > start_low else start_low
            local_start = start + rank_offset
            self.pos = start + total_n
            self.last_take_info = {
                "mode": "random_offset_distributed",
                "file_idx": int(self.file_idx),
                "file": str(self.files[self.file_idx]),
                "start_offset": int(start),
                "local_start_offset": int(local_start),
                "requested_tokens": int(total_n),
                "returned_tokens": int(local_n),
                "rank_offset": int(rank_offset),
                "world_size": int(world_size),
                "start_low": int(start_low),
                "start_high": int(start_high),
            }
            return self.tokens[local_start : local_start + local_n]


class DistributedTokenLoader:
    # Each call advances the shared logical token stream by one global chunk, but only
    # materializes this rank's disjoint local span. The extra "+1" token lets us build
    # (x, y) by shifting without reading every other rank's tokens first.
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        bos_token_id: int = -1,
        random_offset_tokens: int = 0,
        seed: int = 1337,
        debug_static_shapes: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.bos_token_id = int(bos_token_id)
        self.debug_static_shapes = bool(debug_static_shapes)
        self.stream = TokenStream(pattern, random_offset_tokens=random_offset_tokens, seed=seed + rank)
        self._prepare_lock = threading.Lock()
        self.last_batch_info: dict[str, object] = {}
        self.prefetch_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.prefetch_depth = max(int(os.environ.get("TOKEN_LOADER_PREFETCH_DEPTH", "2")), 1)
        self._prefetched_batches: deque[dict[str, object]] = deque()
        batch_prep_workers = _resolve_loader_worker_count("TOKEN_LOADER_BATCH_PREP_WORKERS")
        self._batch_prep_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=batch_prep_workers)
        self._prepared_batches: deque[tuple[str, tuple[int, int, int], Future[dict[str, object]]]] = deque()

    def __del__(self) -> None:
        self._batch_prep_executor.shutdown(wait=False, cancel_futures=True)

    def state_dict(self) -> dict[str, object]:
        return {
            "stream": self.stream.state_dict(),
            "last_batch_info": copy.deepcopy(self.last_batch_info),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.stream.load_state_dict(state["stream"])  # type: ignore[arg-type]
        self.last_batch_info = copy.deepcopy(state.get("last_batch_info", {}))
        self._prefetched_batches.clear()
        for _kind, _request, future in self._prepared_batches:
            future.cancel()
        self._prepared_batches.clear()

    def _prepare_batch(
        self,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
        packed: bool,
    ) -> dict[str, object]:
        with self._prepare_lock:
            local_tokens = global_tokens // (self.world_size * grad_accum_steps)
            per_rank_span = local_tokens + 1
            local = self.stream.take_distributed(per_rank_span, self.rank, self.world_size).to(dtype=torch.int64)
            stream_take_info = copy.deepcopy(self.stream.last_take_info)
        batch_info: dict[str, object] = {
            **stream_take_info,
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "local_span_start": int(self.rank * per_rank_span),
            "local_span_end": int((self.rank + 1) * per_rank_span),
            "local_tokens": int(local_tokens),
            "seq_len": int(seq_len),
            "grad_accum_steps": int(grad_accum_steps),
        }
        if packed:
            # Keep packed training on compile-stable shapes by using the same fixed
            # grid as the standard loader and masking only cross-document targets.
            # This preserves BOS-delimited document boundaries without creating
            # variable numbers of packed rows from short documents.
            valid_mask = local[1:] != self.bos_token_id
            batch_info.update(
                {
                    "batch_mode": "packed",
                    "packed_sequences": int(local_tokens // seq_len),
                    "packed_max_seq_len": int(seq_len),
                    "packed_valid_tokens": int(valid_mask.sum().item()),
                    "packed_boundary_drop_tokens": int(max(local_tokens - int(valid_mask.sum().item()), 0)),
                    "packed_fixed_shape": True,
                }
            )
        if self.debug_static_shapes:
            expected_rows = local_tokens // seq_len
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            enforce_static_shape(x, (expected_rows, seq_len), name="packed_x" if packed else "x")
            enforce_static_shape(y, (expected_rows, seq_len), name="packed_y" if packed else "y")
            if packed:
                loss_mask = y != self.bos_token_id
                enforce_static_shape(loss_mask, (expected_rows, seq_len), name="packed_loss_mask")
                batch_info["packed_expected_shape"] = [int(expected_rows), int(seq_len)]
                batch_info["packed_shape_trace"] = {
                    "x": list(trace_shape(x)),
                    "y": list(trace_shape(y)),
                    "loss_mask": list(trace_shape(loss_mask)),
                }
        return {
            "token_span_cpu": local.contiguous(),
            "seq_len": int(seq_len),
            "packed": bool(packed),
            "batch_info": batch_info,
        }

    def _copy_prepared_batch_to_device(self, prepared: dict[str, object]) -> dict[str, object]:
        token_span_cpu = prepared["token_span_cpu"]
        seq_len = prepared["seq_len"]
        packed = prepared["packed"]
        assert isinstance(token_span_cpu, Tensor)
        assert isinstance(seq_len, int)
        assert isinstance(packed, bool)

        token_span_host = torch.empty_like(token_span_cpu, pin_memory=True)
        token_span_host.copy_(token_span_cpu)

        if self.prefetch_stream is None:
            token_span_dev = token_span_host.to(self.device, non_blocking=False)
        else:
            with torch.cuda.stream(self.prefetch_stream):
                token_span_dev = token_span_host.to(self.device, non_blocking=True)

        x_dev = token_span_dev[:-1].view(-1, seq_len)
        y_dev = token_span_dev[1:].view(-1, seq_len)
        loss_mask_dev = (y_dev != self.bos_token_id) if packed else None

        return {
            "kind": prepared.get("kind", "standard"),
            "request": prepared.get("request"),
            "token_span_host": token_span_host,
            "token_span": token_span_dev,
            "x": x_dev,
            "y": y_dev,
            "loss_mask": loss_mask_dev,
            "batch_info": prepared["batch_info"],
        }

    def _clear_prefetch_pipeline(self) -> None:
        self._prefetched_batches.clear()
        for _kind, _request, future in self._prepared_batches:
            future.cancel()
        self._prepared_batches.clear()

    def _prefetch_signature(
        self,
        kind: str,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
    ) -> tuple[str, tuple[int, int, int]]:
        return kind, (global_tokens, seq_len, grad_accum_steps)

    def _ensure_prefetch_signature(
        self,
        kind: str,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
    ) -> None:
        expected_kind, expected_request = self._prefetch_signature(kind, global_tokens, seq_len, grad_accum_steps)
        if self._prefetched_batches:
            front = self._prefetched_batches[0]
            if front.get("kind") != expected_kind or front.get("request") != expected_request:
                self._clear_prefetch_pipeline()
                return
        if self._prepared_batches:
            queued_kind, queued_request, _future = self._prepared_batches[0]
            if queued_kind != expected_kind or queued_request != expected_request:
                self._clear_prefetch_pipeline()

    def _submit_prepared_batch(
        self,
        kind: str,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
    ) -> None:
        request = (global_tokens, seq_len, grad_accum_steps)
        packed = kind == "packed"
        while len(self._prepared_batches) + len(self._prefetched_batches) < self.prefetch_depth:
            future = self._batch_prep_executor.submit(
                self._prepare_batch, global_tokens, seq_len, grad_accum_steps, packed
            )
            self._prepared_batches.append((kind, request, future))

    def _try_launch_prefetch_copy(self) -> None:
        while self._prepared_batches and len(self._prefetched_batches) < self.prefetch_depth:
            kind, request, future = self._prepared_batches[0]
            if not future.done():
                break
            prepared = future.result()
            prepared["kind"] = kind
            prepared["request"] = request
            self._prepared_batches.popleft()
            self._prefetched_batches.append(self._copy_prepared_batch_to_device(prepared))

    def _get_prefetched_batch(
        self,
        kind: str,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        self._ensure_prefetch_signature(kind, global_tokens, seq_len, grad_accum_steps)
        self._submit_prepared_batch(kind, global_tokens, seq_len, grad_accum_steps)
        self._try_launch_prefetch_copy()
        if not self._prefetched_batches:
            queued_kind, request, future = self._prepared_batches.popleft()
            prepared = future.result()
            prepared["kind"] = queued_kind
            prepared["request"] = request
            self._prefetched_batches.append(self._copy_prepared_batch_to_device(prepared))
        batch = self._prefetched_batches.popleft()
        if self.prefetch_stream is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(self.prefetch_stream)
            batch["token_span"].record_stream(current_stream)
            batch["x"].record_stream(current_stream)
            batch["y"].record_stream(current_stream)
            if batch["loss_mask"] is not None:
                batch["loss_mask"].record_stream(current_stream)
        self.last_batch_info = batch["batch_info"]  # type: ignore[assignment]
        self._submit_prepared_batch(kind, global_tokens, seq_len, grad_accum_steps)
        self._try_launch_prefetch_copy()
        return batch["x"], batch["y"], batch["loss_mask"]  # type: ignore[return-value]

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        x, y, _ = self._get_prefetched_batch("standard", global_tokens, seq_len, grad_accum_steps)
        return x, y

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

def fake_quantize_tensor(t: Tensor, bits: int) -> Tensor:
    if bits <= 0:
        return t
    t32 = t.float()
    if t32.numel() == 0:
        return t
    qmax = float((1 << (bits - 1)) - 1)
    if qmax <= 0:
        return t
    if t32.ndim == 2:
        clip_abs = t32.abs().amax(dim=1, keepdim=True)
    else:
        clip_abs = t32.abs().amax()
    # Preserve genuinely small tensors (for example tied embeddings at init)
    # instead of forcing an absolute quant step of 1/qmax, which can zero the
    # entire tensor when its dynamic range is tiny.
    scale = torch.where(clip_abs > 0, clip_abs / qmax, torch.ones_like(clip_abs))
    clipped = torch.clamp(t32, -clip_abs, clip_abs)
    quantized = torch.clamp(torch.round(clipped / scale), -qmax, qmax) * scale
    return (t32 + (quantized - t32).detach()).to(dtype=t.dtype)


def row_norm_regularization(module: nn.Module) -> Tensor:
    penalty = None
    for param in module.parameters():
        if param.ndim != 2:
            continue
        rows = param.float()
        row_rms = rows.square().mean(dim=1).add_(1e-8).sqrt()
        log_row_rms = row_rms.log()
        value = (log_row_rms - log_row_rms.mean()).square().mean()
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return penalty


def outlier_regularization(module: nn.Module, threshold: float) -> Tensor:
    penalty = None
    clamped_threshold = max(float(threshold), 1e-4)
    for param in module.parameters():
        if param.ndim < 2:
            continue
        w = param.float()
        scale = w.square().mean().add_(1e-8).sqrt()
        excess = (w.abs() / scale - clamped_threshold).clamp_min(0.0)
        value = excess.square().mean()
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return penalty


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_quant_bits = 0

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        if self.fake_quant_bits > 0:
            weight = fake_quantize_tensor(weight, self.fake_quant_bits)
        bias = self.bias
        # Let autocast handle compute dtype promotion so we do not allocate a fresh
        # casted weight tensor on every forward and fragment CUDA memory over time.
        if torch.is_autocast_enabled():
            return F.linear(x, weight, bias)
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
            if bias is not None:
                bias = bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    # Supports NTK-aware scaling when evaluated beyond the training context.
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        train_seq_len: int = 1024,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim
        self.base = base
        self.train_seq_len = max(int(train_seq_len), 1)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, None]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                adjusted_base = self.base * (scale ** (self.dim / max(self.dim - 2, 1)))
                inv_freq = 1.0 / (
                    adjusted_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
                )
            else:
                inv_freq = self.inv_freq.to(device)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype), None


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, scale: Tensor | None = None, inverse_scale: bool = False) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    if scale is None:
        return rot
    full_scale = torch.cat((scale, scale), dim=-1)
    return rot / full_scale if inverse_scale else rot * full_scale


def apply_partial_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rope_dims: int,
    scale: Tensor | None = None,
    inverse_scale: bool = False,
) -> Tensor:
    rope_dims = max(int(rope_dims), 0)
    if rope_dims <= 0:
        return x
    if rope_dims >= x.size(-1):
        return apply_rotary_emb(x, cos, sin, scale=scale, inverse_scale=inverse_scale)
    rot_x = x[..., :rope_dims]
    tail_x = x[..., rope_dims:]
    rot = apply_rotary_emb(rot_x, cos[..., : rope_dims // 2], sin[..., : rope_dims // 2], scale=scale, inverse_scale=inverse_scale)
    return torch.cat((rot, tail_x), dim=-1)

def init_q_gain_vector(num_heads: int, base_gain: float) -> Tensor:
    if num_heads <= 1 or base_gain <= 0.0:
        return torch.full((max(int(num_heads), 1),), float(base_gain), dtype=torch.float32)
    head_pos = torch.linspace(-1.0, 1.0, steps=int(num_heads), dtype=torch.float32)
    multipliers = torch.exp(0.12 * head_pos)
    return float(base_gain) * multipliers


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim: int, condition_dim: int = 0, eps: float = 1e-5, gate_init: float = -2.0):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.base_log_scale = nn.Parameter(torch.zeros(self.dim, dtype=torch.float32))
        self.cond_proj = CastedLinear(condition_dim, self.dim, bias=False) if condition_dim > 0 else None
        if self.cond_proj is not None:
            self.cond_proj._zero_init = True
        self.cond_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32)) if self.cond_proj is not None else None

    def forward(self, x: Tensor, condition: Tensor | None = None) -> Tensor:
        x_norm = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        log_scale = 0.5 * torch.tanh(self.base_log_scale.to(dtype=x.dtype))[None, None, :]
        if (
            condition is not None
            and self.cond_proj is not None
            and condition.ndim == x.ndim
            and condition.size(0) == x.size(0)
            and condition.size(1) == x.size(1)
        ):
            cond_delta = torch.tanh(self.cond_proj(condition).to(dtype=x.dtype))
            cond_gain = torch.sigmoid(self.cond_gate.to(dtype=x.dtype))
            log_scale = log_scale + cond_gain * cond_delta
        return x_norm * torch.exp(log_scale)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int = 1024,
        use_flash_attn_3: bool = False,
        rope_dims: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_flash_attn_3 = bool(use_flash_attn_3 and HAS_FLASH_ATTN_3)
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_dims = self.head_dim if rope_dims <= 0 else min(int(rope_dims), self.head_dim)
        if self.rope_dims % 2 != 0:
            raise ValueError("rope_dims must be even")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(init_q_gain_vector(num_heads, qk_gain_init))
        self.rotary = Rotary(
            self.rope_dims,
            base=rope_base,
            train_seq_len=train_seq_len,
        )
        self.fast_path = True

    @torch._dynamo.disable
    def _flash_attn_3(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if flash_attn_3_func is None:
            raise RuntimeError("FA3 path requested but flash_attn_interface is unavailable")
        return flash_attn_3_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            causal=True,
        ).transpose(1, 2).contiguous()

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = True,
    ) -> Tensor:
        use_gqa = self.num_kv_heads != self.num_heads and k.size(1) != q.size(1)
        out_dtype = q.dtype

        def run_sdpa() -> Tensor:
            q_sdpa = q
            k_sdpa = k
            v_sdpa = v
            if use_gqa:
                repeat = self.num_heads // self.num_kv_heads
                try:
                    return F.scaled_dot_product_attention(
                        q_sdpa,
                        k_sdpa,
                        v_sdpa,
                        attn_mask=None,
                        is_causal=is_causal,
                        enable_gqa=True,
                    ).to(dtype=out_dtype)
                except TypeError:
                    k_sdpa = k_sdpa.repeat_interleave(repeat, dim=1)
                    v_sdpa = v_sdpa.repeat_interleave(repeat, dim=1)
            return F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=None,
                is_causal=is_causal,
            ).to(dtype=out_dtype)

        try:
            return run_sdpa()
        except RuntimeError as exc:
            if "Invalid backend" not in str(exc) or sdpa_kernel is None or SDPBackend is None:
                raise
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                return run_sdpa()

    def _project_qkv(self, x: Tensor, q_gain_delta: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        bsz, seqlen, _dim = x.shape
        q_proj = self.c_q(x)
        q = q_proj.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.c_v(x)
        v = v_proj.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin, scale = self.rotary(seqlen, x.device, q.dtype)
        q = apply_partial_rotary_emb(q, cos, sin, self.rope_dims, scale=scale)
        k = apply_partial_rotary_emb(k, cos, sin, self.rope_dims, scale=scale, inverse_scale=True)
        q_gain = self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if q_gain_delta is not None:
            q_gain_delta = q_gain_delta.to(dtype=q.dtype)
            if q_gain_delta.ndim == 2:
                q_gain = q_gain * (1.0 + q_gain_delta[:, :, None, None])
            else:
                q_gain = q_gain * (1.0 + q_gain_delta.permute(0, 2, 1)[:, :, :, None])
        q = q * q_gain
        return q, k, v

    def _project_qkv_step(
        self,
        x: Tensor,
        position: int,
        q_gain_delta: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.size(1) != 1:
            raise ValueError(f"_project_qkv_step expects seq_len=1, got {tuple(x.shape)}")
        bsz, _seqlen, dim = x.shape
        q_proj = self.c_q(x)
        q = q_proj.reshape(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.c_v(x)
        v = v_proj.reshape(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin, scale = self.rotary(position + 1, x.device, q.dtype)
        cos = cos[..., position : position + 1, :]
        sin = sin[..., position : position + 1, :]
        q = apply_partial_rotary_emb(q, cos, sin, self.rope_dims, scale=scale)
        k = apply_partial_rotary_emb(k, cos, sin, self.rope_dims, scale=scale, inverse_scale=True)
        q_gain = self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if q_gain_delta is not None:
            q_gain_delta = q_gain_delta.to(dtype=q.dtype)
            if q_gain_delta.ndim == 2:
                q_gain = q_gain * (1.0 + q_gain_delta[:, :, None, None])
            else:
                q_gain = q_gain * (1.0 + q_gain_delta.permute(0, 2, 1)[:, :, :, None])
        q = q * q_gain
        return q, k, v

    def _repeat_gqa_kv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.num_kv_heads == self.num_heads:
            return k, v
        repeat = self.num_heads // self.num_kv_heads
        return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)

    def forward_dense(self, q: Tensor, k: Tensor, v: Tensor, dtype: torch.dtype) -> Tensor:
        if self.use_flash_attn_3:
            return self._flash_attn_3(q, k, v)
        return self._scaled_dot_product_attention(q, k, v, is_causal=True)

    def forward(
        self,
        x: Tensor,
        q_gain_delta: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        bsz, seqlen, dim = x.shape
        q, k, v = self._project_qkv(x, q_gain_delta=q_gain_delta)
        y = self.forward_dense(q, k, v, q.dtype)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y), None

    def forward_simple(
        self,
        x: Tensor,
        q_gain_delta: Tensor | None = None,
    ) -> Tensor:
        bsz, seqlen, dim = x.shape
        q, k, v = self._project_qkv(x, q_gain_delta=q_gain_delta)
        y = self.forward_dense(q, k, v, q.dtype)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)

    def forward_step(
        self,
        x: Tensor,
        cache: AttentionStepCache,
        position: int,
        q_gain_delta: Tensor | None = None,
    ) -> Tensor:
        bsz, seqlen, dim = x.shape
        if seqlen != 1:
            raise ValueError(f"forward_step expects seq_len=1, got {tuple(x.shape)}")
        q, k_new, v_new = self._project_qkv_step(x, position=position, q_gain_delta=q_gain_delta)
        k_total = k_new if cache.k is None else torch.cat((cache.k, k_new), dim=-2)
        v_total = v_new if cache.v is None else torch.cat((cache.v, v_new), dim=-2)
        if cache.max_len is not None and cache.max_len > 0:
            k_total = k_total[:, :, -cache.max_len :, :]
            v_total = v_total[:, :, -cache.max_len :, :]
        cache.k = k_total.detach()
        cache.v = v_total.detach()
        k_attn, v_attn = self._repeat_gqa_kv(k_total, v_total)
        y = self._scaled_dot_product_attention(q, k_attn, v_attn, is_causal=False)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int, hidden_dim: int = 0):
        super().__init__()
        inner_dim = hidden_dim if hidden_dim > 0 else mlp_mult * dim
        self.fc = CastedLinear(dim, inner_dim, bias=False)
        self.proj = CastedLinear(inner_dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int,
        rope_base: float,
        qk_gain_init: float,
        residual_alpha: float,
        train_seq_len: int,
        use_flash_attn_3: bool,
        rope_dims: int,
        ln_scale: bool,
        use_adaptive_rmsnorm: bool,
        adaptive_rmsnorm_gate_init: float,
        norm_condition_dim: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.residual_alpha = float(residual_alpha)
        norm_scale = (1.0 / math.sqrt(layer_idx + 1)) if ln_scale else 1.0
        self.register_buffer("norm_scale_buffer", torch.tensor(norm_scale, dtype=torch.float32), persistent=False)
        if use_adaptive_rmsnorm:
            self.attn_norm = AdaptiveRMSNorm(dim, condition_dim=norm_condition_dim, gate_init=adaptive_rmsnorm_gate_init)
            self.mlp_norm = AdaptiveRMSNorm(dim, condition_dim=norm_condition_dim, gate_init=adaptive_rmsnorm_gate_init)
        else:
            self.attn_norm = nn.RMSNorm(dim, elementwise_affine=False)
            self.mlp_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            train_seq_len=train_seq_len,
            use_flash_attn_3=use_flash_attn_3,
            rope_dims=rope_dims,
        )
        self.mlp = MLP(dim, mlp_mult, hidden_dim=mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.last_aux: dict[str, Tensor | None] = {}
        self.fast_path = True

    def forward_simple(
        self,
        x: Tensor,
        x0: Tensor,
        q_gain_delta: Tensor | None = None,
        norm_condition: Tensor | None = None,
    ) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        norm_scale = self.norm_scale_buffer.to(device=x.device, dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_normed = self.attn_norm(x, condition=norm_condition) if isinstance(self.attn_norm, AdaptiveRMSNorm) else self.attn_norm(x)
        attn_normed = attn_normed * norm_scale
        attn_out = self.attn.forward_simple(attn_normed, q_gain_delta=q_gain_delta)
        x = x * self.residual_alpha + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_normed = self.mlp_norm(x, condition=norm_condition) if isinstance(self.mlp_norm, AdaptiveRMSNorm) else self.mlp_norm(x)
        mlp_normed = mlp_normed * norm_scale
        x = x * self.residual_alpha + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_normed)
        return x

    def forward_simple_step(
        self,
        x: Tensor,
        x0: Tensor,
        cache: BlockStepCache,
        position: int,
        q_gain_delta: Tensor | None = None,
        norm_condition: Tensor | None = None,
    ) -> Tensor:
        if x.size(1) != 1:
            raise ValueError(f"forward_simple_step expects seq_len=1, got {tuple(x.shape)}")
        if cache.attention_cache is None:
            cache.attention_cache = AttentionStepCache()
        mix = self.resid_mix.to(dtype=x.dtype)
        norm_scale = self.norm_scale_buffer.to(device=x.device, dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_normed = self.attn_norm(x, condition=norm_condition) if isinstance(self.attn_norm, AdaptiveRMSNorm) else self.attn_norm(x)
        attn_normed = attn_normed * norm_scale
        attn_out = self.attn.forward_step(
            attn_normed,
            cache=cache.attention_cache,
            position=position,
            q_gain_delta=q_gain_delta,
        )
        x = x * self.residual_alpha + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_normed = self.mlp_norm(x, condition=norm_condition) if isinstance(self.mlp_norm, AdaptiveRMSNorm) else self.mlp_norm(x)
        mlp_normed = mlp_normed * norm_scale
        x = x * self.residual_alpha + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_normed)
        return x

    def init_step_cache(self, max_len: int | None = None) -> BlockStepCache:
        return BlockStepCache(attention_cache=AttentionStepCache(max_len=max_len))

    def reset_step_cache(self, cache: BlockStepCache) -> None:
        cache.reset()


AttentionBlock = Block


class CausalStateMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_rank: int,
        num_states: int,
        transition_rank: int,
        latent_rank: int,
        latent_mode: str,
        scale_init: float,
        gate_init: float,
        latent_gate_init: float,
        latent_decay_init: float,
        transition_gate_init: float,
        transition_stickiness_init: float,
        emit_delta_scale_init: float,
        train_seq_len: int,
        track_state_ce: bool,
        track_transition_kl: bool,
        track_future_sketch: bool,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_rank = hidden_rank
        self.num_states = num_states
        self.transition_rank = max(1, int(transition_rank))
        self.latent_rank = max(int(latent_rank), 0)
        latent_mode = str(latent_mode or "off").strip().lower()
        if latent_mode not in {"off", "additive", "replace", "auto"}:
            raise ValueError(f"Unsupported latent_mode={latent_mode!r}; expected off, additive, replace, or auto")
        if self.latent_rank <= 0:
            latent_mode = "off"
        self.latent_mode = latent_mode
        self.uses_structured_transition_params = not (self.latent_mode == "replace" and self.latent_rank > 0)
        self.filter_chunk_size = _resolve_structured_scan_chunk_size(train_seq_len, transition_rank)
        self.transition_context_rank = max(8, min(16, hidden_rank))
        self.track_state_ce = bool(track_state_ce)
        self.track_transition_kl = bool(track_transition_kl)
        self.track_future_sketch = bool(track_future_sketch)
        self.prefix_down = CastedLinear(dim, hidden_rank, bias=False)
        self.decoder_hidden = CastedLinear(hidden_rank, hidden_rank, bias=True)
        self.state_head = CastedLinear(hidden_rank, num_states, bias=False)
        self.transition_context_in = CastedLinear(hidden_rank, self.transition_context_rank, bias=False)
        self.transition_context_out = CastedLinear(self.transition_context_rank, num_states, bias=False)
        self.transition_context_out._zero_init = True
        self.latent_in = CastedLinear(hidden_rank, self.latent_rank, bias=False) if self.latent_rank > 0 else None
        self.latent_gate_in = CastedLinear(hidden_rank, self.latent_rank, bias=True) if self.latent_rank > 0 else None
        self.latent_out = CastedLinear(self.latent_rank, num_states, bias=False) if self.latent_rank > 0 else None
        self.hidden_gate = CastedLinear(hidden_rank, 1, bias=True)
        self.transition_gate_proj = CastedLinear(hidden_rank, 1, bias=False)
        self.transition_gate_proj._zero_init = True
        self.transition_pred_scale_proj = CastedLinear(hidden_rank, 1, bias=False)
        self.transition_pred_scale_proj._zero_init = True
        self.belief_out = CastedLinear(num_states, dim, bias=False)
        self.output_scale = nn.Parameter(torch.tensor(inverse_softplus_scalar(scale_init), dtype=torch.float32))
        self.output_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.latent_output_gate = (
            nn.Parameter(torch.tensor(latent_gate_init, dtype=torch.float32)) if self.latent_rank > 0 else None
        )
        self.latent_decay_logits = (
            nn.Parameter(
                torch.full(
                    (self.latent_rank,),
                    fill_value=inverse_sigmoid_scalar(latent_decay_init),
                    dtype=torch.float32,
                )
            )
            if self.latent_rank > 0
            else None
        )
        self.transition_gate = nn.Parameter(torch.tensor(transition_gate_init, dtype=torch.float32))
        self.transition_source_logits = (
            nn.Parameter(torch.zeros((num_states, self.transition_rank), dtype=torch.float32))
            if self.uses_structured_transition_params
            else None
        )
        self.transition_dest_logits = (
            nn.Parameter(torch.zeros((self.transition_rank, num_states), dtype=torch.float32))
            if self.uses_structured_transition_params
            else None
        )
        self.transition_stay_logits = (
            nn.Parameter(torch.full((num_states,), fill_value=float(transition_stickiness_init), dtype=torch.float32))
            if self.uses_structured_transition_params
            else None
        )
        self.emit_delta_scale = nn.Parameter(
            torch.tensor(inverse_softplus_scalar(max(float(emit_delta_scale_init), 1e-6)), dtype=torch.float32)
        )
        self.register_buffer(
            "log_state_priors",
            torch.full((num_states,), fill_value=-math.log(max(num_states, 1)), dtype=torch.float32),
            persistent=False,
        )
        self.future_sketch_head: CastedLinear | None = None
        self.future_sketch_dim = 0
        self.last_aux: dict[str, Tensor | None] = {}
        self.last_kernel_info: dict[str, object] = {}
        self.fast_path = True
        self._auto_mode_cache: dict[tuple[int, int, int, int, str], str] = {}
        self._packed_transition_cache: dict[str, object] = {}
        self._sparse_transition_cache: dict[str, object] = {}
        self._reduced_transition_cache: dict[str, object] = {}
        self._compiled_kernel_config_cache: dict[tuple[str, int, int, bool], StructuredScanKernelConfig] = {}

    def _uses_latent_additive(self) -> bool:
        return self.latent_mode == "additive" and self.latent_rank > 0

    def _uses_latent_replace(self) -> bool:
        return self.latent_mode == "replace" and self.latent_rank > 0

    def _resolved_latent_mode(self, seq_len: int, device: torch.device) -> str:
        if self.latent_rank <= 0:
            return "off"
        if self.latent_mode in {"off", "additive", "replace"}:
            return self.latent_mode
        # Keep "auto" aligned with the intended competition posture: latent replace
        # is the primary engine whenever latent state is enabled. Structured exact
        # remains available only through explicit "off".
        device_key = f"{device.type}:{device.index if device.index is not None else -1}"
        cache_key = (int(seq_len), int(self.transition_rank), int(self.latent_rank), int(self.num_states), device_key)
        cached = self._auto_mode_cache.get(cache_key)
        if cached is not None:
            return cached
        mode = "replace"
        self._auto_mode_cache[cache_key] = mode
        return mode

    def configure_future_sketch(self, total_sketch_dim: int) -> None:
        total_sketch_dim = max(int(total_sketch_dim), 0)
        if total_sketch_dim <= 0:
            self.future_sketch_dim = 0
            self.future_sketch_head = None
            return
        if self.future_sketch_head is not None and self.future_sketch_dim == total_sketch_dim:
            return
        device = self.prefix_down.weight.device
        head = CastedLinear(self.hidden_rank, total_sketch_dim, bias=False).to(device)
        head.float()
        nn.init.normal_(head.weight, mean=0.0, std=0.02)
        self.future_sketch_head = head
        self.future_sketch_dim = total_sketch_dim

    def _initial_log_belief(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return self.log_state_priors.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1).contiguous()

    def _project_hidden(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        prev_x = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
        delta = x - prev_x
        state_features = x + 0.5 * delta
        state_hidden = self.prefix_down(state_features)
        state_hidden = torch.tanh(self.decoder_hidden(state_hidden))
        local_logits = self.state_head(state_hidden)
        transition_context = self.transition_context_out(self.transition_context_in(state_hidden))
        return state_hidden, local_logits, transition_context

    def _compute_latent_sequence_states(
        self,
        state_hidden: Tensor,
        *,
        prior_only: bool = False,
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor] | None:
        if (
            self.latent_in is None
            or self.latent_gate_in is None
            or self.latent_decay_logits is None
        ):
            return None
        drive_hidden = self.latent_in(state_hidden)
        gate_hidden = torch.sigmoid(self.latent_gate_in(state_hidden))
        decay = _bounded_latent_decay(self.latent_decay_logits)
        use_cuda_latent_scan = bool(
            USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN
            and state_hidden.is_cuda
            and drive_hidden.size(-1) > 0
        )
        if use_cuda_latent_scan:
            drive = (drive_hidden * gate_hidden).contiguous()
            initial_state = torch.zeros((drive.size(0), self.latent_rank), device=drive.device, dtype=state_hidden.dtype)
            if prior_only:
                prior_states, final_state = causal_machine_latent_prior_scan_cuda(
                    drive,
                    decay,
                    initial_state,
                )
                latent_states = None
            else:
                latent_states, prior_states, final_state = causal_machine_latent_scan_cuda(
                    drive,
                    decay,
                    initial_state,
                )
                latent_states = latent_states.to(dtype=state_hidden.dtype)
            prior_states = prior_states.to(dtype=state_hidden.dtype)
            final_state = final_state.to(dtype=state_hidden.dtype)
        else:
            drive = drive_hidden.float() * gate_hidden.float()
            prev = torch.zeros((drive.size(0), self.latent_rank), device=drive.device, dtype=torch.float32)
            latent_chunks: list[Tensor] = []
            prior_chunks: list[Tensor] = []
            chunk_size = max(int(self.filter_chunk_size), 1)
            decay_b = decay.view(1, 1, self.latent_rank)
            for chunk_start in range(0, int(drive.size(1)), chunk_size):
                chunk = drive[:, chunk_start : chunk_start + chunk_size, :]
                chunk_len = int(chunk.size(1))
                positions = torch.arange(chunk_len, device=drive.device, dtype=torch.float32).view(1, chunk_len, 1)
                powers = torch.pow(decay_b, positions)
                scaled = chunk / powers.clamp_min(1.0e-8)
                prefix = torch.cumsum(scaled, dim=1)
                chunk_states = powers * (decay_b * prev.unsqueeze(1) + prefix)
                chunk_prior = torch.cat(
                    [
                        (decay_b[:, :1, :] * prev.unsqueeze(1)),
                        decay_b * chunk_states[:, :-1, :],
                    ],
                    dim=1,
                )
                prior_chunks.append(chunk_prior)
                if not prior_only:
                    latent_chunks.append(chunk_states)
                prev = chunk_states[:, -1, :]
            latent_states = (
                torch.cat(latent_chunks, dim=1).to(dtype=state_hidden.dtype)
                if latent_chunks
                else None
            )
            prior_states = torch.cat(prior_chunks, dim=1).to(dtype=state_hidden.dtype)
            final_state = prev.to(dtype=state_hidden.dtype)
        return latent_states, prior_states, final_state, decay.to(device=state_hidden.device, dtype=state_hidden.dtype)

    def _decode_latent_state_logits(self, latent_states: Tensor, *, out_dtype: torch.dtype) -> Tensor | None:
        if self.latent_out is None or self.latent_output_gate is None:
            return None
        latent_hidden = torch.tanh(latent_states)
        gate = _bounded_gate(
            self.latent_output_gate,
            "CAUSAL_MACHINE_LATENT_OUTPUT_GATE_MIN",
            "CAUSAL_MACHINE_LATENT_OUTPUT_GATE_MAX",
            default_min=0.10,
            default_max=0.995,
        ).to(device=latent_states.device, dtype=out_dtype)
        return gate * self.latent_out(latent_hidden.to(dtype=out_dtype))

    def _latent_replace_token_trust(
        self,
        state_hidden: Tensor,
        *,
        out_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        base_gate = _bounded_gate(
            self.transition_gate,
            "CAUSAL_MACHINE_TRANSITION_GATE_MIN",
            "CAUSAL_MACHINE_TRANSITION_GATE_MAX",
            default_min=0.01,
            default_max=0.995,
        ).to(device=state_hidden.device, dtype=out_dtype)
        token_gate_delta = torch.tanh(self.transition_gate_proj(state_hidden).float()).to(
            device=state_hidden.device,
            dtype=out_dtype,
        )
        token_gate = torch.where(
            token_gate_delta >= 0,
            base_gate + (1.0 - base_gate) * token_gate_delta,
            base_gate * (1.0 + token_gate_delta),
        )
        pred_scale = _bounded_gate(
            self.transition_pred_scale_proj(state_hidden),
            "CAUSAL_MACHINE_TRANSITION_PRED_SCALE_MIN",
            "CAUSAL_MACHINE_TRANSITION_PRED_SCALE_MAX",
            default_min=0.75,
            default_max=1.25,
        ).to(device=state_hidden.device, dtype=out_dtype)
        return token_gate, pred_scale

    def _compute_latent_logits_sequence(self, state_hidden: Tensor) -> Tensor | None:
        latent_scan = self._compute_latent_sequence_states(state_hidden, prior_only=False)
        if latent_scan is None:
            return None
        latent_states, _prior_states, _final_state, _decay = latent_scan
        if latent_states is None:
            return None
        return self._decode_latent_state_logits(latent_states, out_dtype=state_hidden.dtype)

    def _compute_latent_step_states(self, state_hidden: Tensor, cache: CausalMachineCache) -> tuple[Tensor, Tensor, Tensor] | None:
        if (
            self.latent_in is None
            or self.latent_gate_in is None
            or self.latent_decay_logits is None
        ):
            return None
        drive = (self.latent_in(state_hidden) * torch.sigmoid(self.latent_gate_in(state_hidden))).float()
        prev = cache.latent_state
        if prev is None:
            prev = torch.zeros((drive.size(0), self.latent_rank), device=drive.device, dtype=torch.float32)
        decay = _bounded_latent_decay(self.latent_decay_logits).view(1, -1)
        prior_state = (decay * prev).to(dtype=state_hidden.dtype)
        next_state = prior_state.float() + drive[:, 0, :]
        cache.latent_state = next_state.detach()
        return prior_state, next_state.to(dtype=state_hidden.dtype), decay.to(device=state_hidden.device, dtype=state_hidden.dtype)

    def _compute_latent_logits_step(self, state_hidden: Tensor, cache: CausalMachineCache) -> Tensor | None:
        latent_step = self._compute_latent_step_states(state_hidden, cache)
        if latent_step is None:
            return None
        _prior_state, next_state, _decay = latent_step
        return self._decode_latent_state_logits(next_state.unsqueeze(1), out_dtype=state_hidden.dtype)

    def _latent_only_sequence_beliefs(self, local_logits: Tensor) -> tuple[Tensor, Tensor, None]:
        state_log_beliefs = F.log_softmax(local_logits.float(), dim=-1).to(dtype=local_logits.dtype)
        return state_log_beliefs, state_log_beliefs[:, -1, :].float(), None

    def _latent_only_step_beliefs(self, local_logits: Tensor, cache: CausalMachineCache) -> Tensor:
        next_log_belief = F.log_softmax(local_logits[:, 0, :].float(), dim=-1)
        cache.record_step(next_log_belief.to(dtype=local_logits.dtype))
        return next_log_belief.to(dtype=local_logits.dtype).unsqueeze(1)

    def _latent_replace_sequence_beliefs(
        self,
        local_logits: Tensor,
        transition_context: Tensor,
        state_hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        latent_scan = self._compute_latent_sequence_states(state_hidden, prior_only=True)
        if latent_scan is None:
            return self._latent_only_sequence_beliefs(local_logits)
        _latent_states, latent_prior_states, final_latent_state, _decay = latent_scan
        prior_logits = self._decode_latent_state_logits(latent_prior_states, out_dtype=local_logits.dtype)
        if prior_logits is None:
            return self._latent_only_sequence_beliefs(local_logits)
        transition_gate, pred_scale = self._latent_replace_token_trust(
            state_hidden,
            out_dtype=local_logits.dtype,
        )
        transition_context = transition_context.to(dtype=local_logits.dtype)
        use_cuda_latent_replace = bool(
            USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN
            and local_logits.is_cuda
            and prior_logits.is_cuda
            and transition_context.is_cuda
            and transition_gate.is_cuda
            and pred_scale.is_cuda
        )
        if use_cuda_latent_replace:
            state_log_beliefs, prior_log_beliefs_full = causal_machine_latent_replace_cuda(
                local_logits,
                prior_logits,
                transition_context,
                transition_gate,
                pred_scale,
            )
            prior_log_beliefs = prior_log_beliefs_full if self.track_transition_kl else None
        else:
            prior_with_context = prior_logits + transition_context
            prior_with_context = prior_with_context / pred_scale.clamp_min(1.0e-4)
            prior_log_beliefs = (
                F.log_softmax(prior_with_context.float(), dim=-1).to(dtype=local_logits.dtype)
                if self.track_transition_kl
                else None
            )
            filtered_logits = local_logits + transition_gate * prior_with_context
            state_log_beliefs = F.log_softmax(filtered_logits.float(), dim=-1).to(dtype=local_logits.dtype)
        return state_log_beliefs, final_latent_state.float(), prior_log_beliefs

    def _latent_replace_step_beliefs(
        self,
        local_logits: Tensor,
        transition_context: Tensor,
        state_hidden: Tensor,
        cache: CausalMachineCache,
    ) -> tuple[Tensor, Tensor | None]:
        latent_step = self._compute_latent_step_states(state_hidden, cache)
        if latent_step is None:
            return self._latent_only_step_beliefs(local_logits, cache), None
        prior_state, _next_state, _decay = latent_step
        prior_logits = self._decode_latent_state_logits(prior_state.unsqueeze(1), out_dtype=local_logits.dtype)
        if prior_logits is None:
            return self._latent_only_step_beliefs(local_logits, cache), None
        transition_gate, pred_scale = self._latent_replace_token_trust(
            state_hidden,
            out_dtype=local_logits.dtype,
        )
        transition_context = transition_context.to(dtype=local_logits.dtype)
        use_cuda_latent_replace = bool(
            USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN
            and local_logits.is_cuda
            and prior_logits.is_cuda
            and transition_context.is_cuda
            and transition_gate.is_cuda
            and pred_scale.is_cuda
        )
        if use_cuda_latent_replace:
            state_log_beliefs, prior_log_beliefs_full = causal_machine_latent_replace_cuda(
                local_logits,
                prior_logits,
                transition_context,
                transition_gate,
                pred_scale,
            )
            prior_log_beliefs = prior_log_beliefs_full if self.track_transition_kl else None
        else:
            prior_with_context = prior_logits + transition_context
            prior_with_context = prior_with_context / pred_scale.clamp_min(1.0e-4)
            prior_log_beliefs = (
                F.log_softmax(prior_with_context[:, 0, :].float(), dim=-1).to(dtype=local_logits.dtype).unsqueeze(1)
                if self.track_transition_kl
                else None
            )
            filtered_logits = local_logits + transition_gate * prior_with_context
            state_log_beliefs = F.log_softmax(filtered_logits[:, 0, :].float(), dim=-1).to(dtype=local_logits.dtype).unsqueeze(1)
        cache.record_step(state_log_beliefs[:, 0, :])
        return state_log_beliefs, prior_log_beliefs

    def _structured_transition_params(self) -> tuple[Tensor, Tensor, Tensor]:
        transition_source_probs = F.softmax(self.transition_source_logits.float(), dim=-1)
        transition_dest_probs = F.softmax(self.transition_dest_logits.float(), dim=-1)
        transition_stay_probs = torch.sigmoid(self.transition_stay_logits.float())
        return transition_source_probs, transition_dest_probs, transition_stay_probs

    def _get_packed_transition_tables(self, device: torch.device) -> tuple[int, Tensor, Tensor, Tensor, Tensor] | None:
        return get_or_update_scan_transition_prepack(
            self._packed_transition_cache,
            self.transition_source_logits,
            self.transition_dest_logits,
            device,
        )

    def _get_sparse_transition_tables(
        self,
        device: torch.device,
        runtime_config: StructuredScanRuntimeConfig | None,
        packed_transition_tables: tuple[int, Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> StructuredSparseTransitionTables | None:
        if _structured_filter_mode() == "composable":
            return None
        return get_or_update_scan_transition_sparse_blocks(
            self._sparse_transition_cache,
            self.transition_source_logits,
            self.transition_dest_logits,
            device,
            runtime_config,
            packed_transition_tables,
        )

    def _get_reduced_transition_cache(
        self,
        device: torch.device,
        runtime_config: StructuredScanRuntimeConfig | None,
        *,
        tile_size: int,
        split_size: int,
    ) -> StructuredReducedTransitionCache | None:
        return get_or_update_structured_reduced_transition_cache(
            self._reduced_transition_cache,
            self.transition_source_logits,
            self.transition_dest_logits,
            device,
            runtime_config,
            num_states=self.num_states,
            tile_size=int(tile_size),
            split_size=int(split_size),
        )

    def _filter_sequence(
        self,
        local_logits: Tensor,
        transition_context: Tensor,
        initial_log_belief: Tensor,
        runtime_config: StructuredScanRuntimeConfig | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        transition_stay_probs = torch.sigmoid(self.transition_stay_logits.float())
        transition_gate = _bounded_gate(
            self.transition_gate,
            "CAUSAL_MACHINE_TRANSITION_GATE_MIN",
            "CAUSAL_MACHINE_TRANSITION_GATE_MAX",
            default_min=0.01,
            default_max=0.995,
        )
        transition_context, transition_gate = _prepare_structured_filter_inputs(
            transition_context,
            transition_gate,
        )
        transition_context, transition_gate = _apply_native_structured_score_mod_inputs(
            transition_context,
            transition_gate,
            runtime_config,
        )
        tiled_needs_grad = (
            torch.is_grad_enabled()
            and (
                local_logits.requires_grad
                or self.transition_source_logits.requires_grad
                or self.transition_dest_logits.requires_grad
                or transition_context.requires_grad
                or initial_log_belief.requires_grad
                or transition_gate.requires_grad
                or transition_stay_probs.requires_grad
            )
        )
        packed_transition_tables = self._get_packed_transition_tables(local_logits.device)
        kernel_config = _get_precomputed_structured_scan_kernel_config(
            self,
            cache_attr="_compiled_kernel_config_cache",
            num_states=self.num_states,
            transition_rank=self.transition_rank,
            seq_len=int(local_logits.size(1)),
            device=local_logits.device,
            default_chunk_size=self.filter_chunk_size,
            needs_grad=tiled_needs_grad,
            runtime_config=runtime_config,
        )
        if runtime_config is not None:
            runtime_config.reduced_transition_cache = self._get_reduced_transition_cache(
                local_logits.device,
                runtime_config,
                tile_size=int(kernel_config.tile_size),
                split_size=int(kernel_config.split_size),
            )
        sparse_transition_tables = (
            self._get_sparse_transition_tables(
                local_logits.device,
                runtime_config,
                packed_transition_tables,
            )
            if (
                USE_CAUSAL_MACHINE_CUDA_SCAN
                and _structured_runtime_supports_sparse_cuda(runtime_config)
            )
            else None
        )
        sparse_needs_grad = (
            torch.is_grad_enabled()
            and (
                local_logits.requires_grad
                or self.transition_source_logits.requires_grad
                or self.transition_dest_logits.requires_grad
                or transition_context.requires_grad
                or initial_log_belief.requires_grad
                or transition_gate.requires_grad
                or transition_stay_probs.requires_grad
            )
        )
        sparse_runtime_result = _execute_structured_sparse_runtime_cuda(
            local_logits,
            self.transition_source_logits,
            self.transition_dest_logits,
            transition_context,
            initial_log_belief.to(dtype=local_logits.dtype),
            transition_gate.reshape(()),
            transition_stay_probs,
            sparse_transition_tables,
            runtime_config=runtime_config,
            chunk_size=int(kernel_config.chunk_size),
            needs_grad=sparse_needs_grad,
        )
        if sparse_runtime_result is not None:
            state_log_beliefs, final_log_belief = sparse_runtime_result
            self.last_kernel_info = _structured_scan_kernel_info(
                path="cuda_sparse",
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
            )
            _enforce_structured_scan_cuda_contract(self.last_kernel_info, context="state mixer structured scan")
            return state_log_beliefs.to(dtype=local_logits.dtype), final_log_belief, None
        masked_needs_grad = (
            torch.is_grad_enabled()
            and (
                local_logits.requires_grad
                or self.transition_source_logits.requires_grad
                or self.transition_dest_logits.requires_grad
                or transition_context.requires_grad
                or initial_log_belief.requires_grad
                or transition_gate.requires_grad
                or transition_stay_probs.requires_grad
            )
        )
        use_masked_cuda = (
            USE_CAUSAL_MACHINE_CUDA_SCAN
            and _structured_runtime_supports_masked_cuda(runtime_config)
            and _can_use_causal_machine_masked_scan_cuda(
                local_logits.device,
                num_states=self.num_states,
                transition_rank=self.transition_rank,
                needs_grad=masked_needs_grad,
                runtime_config=runtime_config,
            )
        )
        if use_masked_cuda:
            state_log_beliefs, final_log_belief = causal_machine_scan_masked_cuda(
                local_logits,
                self.transition_source_logits,
                self.transition_dest_logits,
                transition_context,
                initial_log_belief.to(dtype=local_logits.dtype),
                transition_gate.reshape(()),
                transition_stay_probs,
                runtime_config=runtime_config,
                chunk_size=int(kernel_config.chunk_size),
            )
            self.last_kernel_info = _structured_scan_kernel_info(
                path="cuda_masked",
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
            )
            _enforce_structured_scan_cuda_contract(self.last_kernel_info, context="state mixer structured scan")
            return state_log_beliefs.to(dtype=local_logits.dtype), final_log_belief, None
        if USE_CAUSAL_MACHINE_CUDA_SCAN and kernel_config.backend == "cuda":
            state_log_beliefs, final_log_belief = causal_machine_scan_cuda(
                local_logits,
                self.transition_source_logits,
                self.transition_dest_logits,
                transition_context,
                initial_log_belief.to(dtype=local_logits.dtype),
                transition_gate.reshape(()),
                transition_stay_probs,
                packed_transition_tables=packed_transition_tables,
                chunk_size=int(kernel_config.chunk_size),
                runtime_config=runtime_config,
            )
            # Keep the CUDA-scan competition path to a single fused recurrent pass.
            # Transition-KL priors are not reconstructed in Python here because that
            # would add a second token-by-token recurrence for every SSM block.
            prior_log_beliefs = None
            dense_path = (
                "cuda_dense_lowp_tensor_core"
                if _structured_scan_uses_lowp_tensor_core_path(
                    device=local_logits.device,
                    dtype=local_logits.dtype,
                    kernel_config=kernel_config,
                    packed_transition_tables=packed_transition_tables,
                )
                else "cuda_dense"
            )
            self.last_kernel_info = _structured_scan_kernel_info(
                path=dense_path,
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
            )
            _enforce_structured_scan_cuda_contract(self.last_kernel_info, context="state mixer structured scan")
            return state_log_beliefs.to(dtype=local_logits.dtype), final_log_belief, prior_log_beliefs
        transition_source_probs, transition_dest_probs, _ = self._structured_transition_params()
        self.last_kernel_info = _structured_scan_kernel_info(
            path=f"fallback_{kernel_config.backend}",
            kernel_config=kernel_config,
            runtime_config=runtime_config,
            packed_transition_tables=packed_transition_tables,
        )
        _enforce_structured_scan_cuda_contract(self.last_kernel_info, context="state mixer structured scan")
        return structured_scan_fallback(
            local_logits,
            transition_context,
            initial_log_belief,
            transition_source_probs,
            transition_dest_probs,
            transition_stay_probs,
            transition_gate,
            chunk_size=int(kernel_config.chunk_size),
            tile_size=int(kernel_config.tile_size),
            split_size=int(kernel_config.split_size),
            backend=str(kernel_config.backend),
            track_transition_kl=self.track_transition_kl and _structured_scan_save_all(runtime_config),
            runtime_config=runtime_config,
            packed_transition_tables=packed_transition_tables,
        )

    def _filter_step(
        self,
        local_logits: Tensor,
        transition_context: Tensor,
        cache: CausalMachineCache,
        runtime_config: StructuredScanRuntimeConfig | None = None,
    ) -> Tensor:
        batch_size = int(local_logits.size(0))
        prev_log_belief = (
            cache.log_belief.to(device=local_logits.device, dtype=torch.float32)
            if cache.log_belief is not None
            else self._initial_log_belief(batch_size, local_logits.device, torch.float32)
        )
        transition_gate = _bounded_gate(
            self.transition_gate,
            "CAUSAL_MACHINE_TRANSITION_GATE_MIN",
            "CAUSAL_MACHINE_TRANSITION_GATE_MAX",
            default_min=0.01,
            default_max=0.995,
        )
        transition_context, transition_gate = _prepare_structured_filter_inputs(
            transition_context,
            transition_gate,
        )
        transition_context, transition_gate = _apply_native_structured_score_mod_inputs(
            transition_context,
            transition_gate,
            runtime_config,
        )
        tiled_needs_grad = (
            torch.is_grad_enabled()
            and (
                local_logits.requires_grad
                or self.transition_source_logits.requires_grad
                or self.transition_dest_logits.requires_grad
                or transition_context.requires_grad
                or prev_log_belief.requires_grad
                or transition_gate.requires_grad
                or self.transition_stay_logits.requires_grad
            )
        )
        packed_transition_tables = self._get_packed_transition_tables(local_logits.device)
        kernel_config = _get_precomputed_structured_scan_kernel_config(
            self,
            cache_attr="_compiled_kernel_config_cache",
            num_states=self.num_states,
            transition_rank=self.transition_rank,
            seq_len=1,
            device=local_logits.device,
            default_chunk_size=1,
            needs_grad=tiled_needs_grad,
            runtime_config=runtime_config,
        )
        transition_stay_probs = torch.sigmoid(self.transition_stay_logits.float())
        sparse_transition_tables = (
            self._get_sparse_transition_tables(
                local_logits.device,
                runtime_config,
                packed_transition_tables,
            )
            if (
                USE_CAUSAL_MACHINE_CUDA_SCAN
                and _structured_runtime_supports_sparse_cuda(runtime_config)
            )
            else None
        )
        sparse_needs_grad = (
            torch.is_grad_enabled()
            and (
                local_logits.requires_grad
                or self.transition_source_logits.requires_grad
                or self.transition_dest_logits.requires_grad
                or transition_context.requires_grad
                or prev_log_belief.requires_grad
                or transition_gate.requires_grad
                or transition_stay_probs.requires_grad
            )
        )
        sparse_runtime_result = _execute_structured_sparse_runtime_cuda(
            local_logits,
            self.transition_source_logits,
            self.transition_dest_logits,
            transition_context,
            prev_log_belief.to(dtype=local_logits.dtype),
            transition_gate.reshape(()),
            transition_stay_probs,
            sparse_transition_tables,
            runtime_config=runtime_config,
            chunk_size=1,
            needs_grad=sparse_needs_grad,
        )
        if sparse_runtime_result is not None:
            state_log_beliefs, _final_log_belief = sparse_runtime_result
            cache.record_step(state_log_beliefs[:, 0, :])
            self.last_kernel_info = _structured_scan_kernel_info(
                path="cuda_sparse",
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
                cache=cache,
            )
            return state_log_beliefs
        masked_needs_grad = (
            torch.is_grad_enabled()
            and (
                local_logits.requires_grad
                or self.transition_source_logits.requires_grad
                or self.transition_dest_logits.requires_grad
                or transition_context.requires_grad
                or prev_log_belief.requires_grad
                or transition_gate.requires_grad
                or transition_stay_probs.requires_grad
            )
        )
        use_masked_cuda = (
            USE_CAUSAL_MACHINE_CUDA_SCAN
            and _structured_runtime_supports_masked_cuda(runtime_config)
            and _can_use_causal_machine_masked_scan_cuda(
                local_logits.device,
                num_states=self.num_states,
                transition_rank=self.transition_rank,
                needs_grad=masked_needs_grad,
                runtime_config=runtime_config,
            )
        )
        can_use_fused_paged_step = (
            USE_CAUSAL_MACHINE_CUDA_SCAN
            and kernel_config.backend in {"cuda", "cuda_tiled"}
            and self.latent_rank == 0
            and _structured_runtime_supports_fused_paged_step(runtime_config)
            and cache.paged_log_beliefs is not None
            and cache.paged_lengths is not None
            and local_logits.is_cuda
            and int(local_logits.size(1)) == 1
            and not torch.is_grad_enabled()
            and not use_masked_cuda
        )
        if sparse_runtime_result is None and can_use_fused_paged_step:
            graph_used = False
            if _structured_runtime_supports_step_cuda_graph(runtime_config):
                packed_kind = -1 if packed_transition_tables is None else int(packed_transition_tables[0])
                packed_source_ptr = 0 if packed_transition_tables is None else int(packed_transition_tables[1].data_ptr())
                packed_dest_ptr = 0 if packed_transition_tables is None else int(packed_transition_tables[3].data_ptr())
                graph_signature = (
                    int(id(self)),
                    tuple(int(dim) for dim in local_logits.shape),
                    str(local_logits.dtype),
                    int(kernel_config.num_states),
                    int(kernel_config.transition_rank),
                    str(kernel_config.backend),
                    int(kernel_config.tile_size),
                    int(kernel_config.split_size),
                    int(packed_kind),
                    int(cache.paged_log_beliefs.data_ptr()),
                    int(cache.paged_lengths.data_ptr()),
                    int(packed_source_ptr),
                    int(packed_dest_ptr),
                )
                runner = cache.step_graph_runner if isinstance(cache.step_graph_runner, dict) else None
                if runner is None or runner.get("signature") != graph_signature:
                    try:
                        static_local_logits = torch.empty_like(local_logits)
                        static_transition_context = torch.empty_like(transition_context)
                        static_local_logits.copy_(local_logits, non_blocking=True)
                        static_transition_context.copy_(transition_context, non_blocking=True)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            static_state_log_beliefs, static_final_log_belief, path = _structured_fused_paged_step_cuda(
                                local_logits=static_local_logits,
                                transition_source_logits=self.transition_source_logits,
                                transition_dest_logits=self.transition_dest_logits,
                                transition_context=static_transition_context,
                                transition_gate=transition_gate.reshape(()),
                                transition_stay_probs=transition_stay_probs,
                                cache=cache,
                                kernel_config=kernel_config,
                                runtime_config=runtime_config,
                                packed_transition_tables=packed_transition_tables,
                            )
                        runner = {
                            "signature": graph_signature,
                            "graph": graph,
                            "static_local_logits": static_local_logits,
                            "static_transition_context": static_transition_context,
                            "static_state_log_beliefs": static_state_log_beliefs,
                            "static_final_log_belief": static_final_log_belief,
                            "path": path,
                        }
                        cache.step_graph_runner = runner
                        graph_used = True
                    except Exception:
                        cache.step_graph_runner = None
                        runner = None
                else:
                    static_local_logits = runner["static_local_logits"]
                    static_transition_context = runner["static_transition_context"]
                    static_local_logits.copy_(local_logits, non_blocking=True)
                    static_transition_context.copy_(transition_context, non_blocking=True)
                    runner["graph"].replay()
                    graph_used = True
                if runner is not None and graph_used:
                    state_log_beliefs = runner["static_state_log_beliefs"]
                    final_log_belief = runner["static_final_log_belief"]
                    path = str(runner["path"])
                    _update_cache_after_fused_paged_step(cache, final_log_belief)
                    if runtime_config is not None and runtime_config.grouped_launch_pack is not None:
                        runtime_config.grouped_launch_pack.record_paged_cache_op()
                        runtime_config.grouped_launch_pack.record_small_decode_update()
                    self.last_kernel_info = _structured_scan_kernel_info(
                        path=path,
                        kernel_config=kernel_config,
                        runtime_config=runtime_config,
                        packed_transition_tables=packed_transition_tables,
                        cache=cache,
                    )
                    self.last_kernel_info["graph_replay"] = True
                    return state_log_beliefs
            state_log_beliefs, final_log_belief, path = _structured_fused_paged_step_cuda(
                local_logits=local_logits,
                transition_source_logits=self.transition_source_logits,
                transition_dest_logits=self.transition_dest_logits,
                transition_context=transition_context,
                transition_gate=transition_gate.reshape(()),
                transition_stay_probs=transition_stay_probs,
                cache=cache,
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
            )
            _update_cache_after_fused_paged_step(cache, final_log_belief)
            if runtime_config is not None and runtime_config.grouped_launch_pack is not None:
                runtime_config.grouped_launch_pack.record_paged_cache_op()
                runtime_config.grouped_launch_pack.record_small_decode_update()
            self.last_kernel_info = _structured_scan_kernel_info(
                path=path,
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
                cache=cache,
            )
            return state_log_beliefs
        if use_masked_cuda:
            state_log_beliefs, _final_log_belief = causal_machine_scan_masked_cuda(
                local_logits,
                self.transition_source_logits,
                self.transition_dest_logits,
                transition_context,
                prev_log_belief.to(dtype=local_logits.dtype),
                transition_gate.reshape(()),
                transition_stay_probs,
                runtime_config=runtime_config,
                chunk_size=1,
            )
            cache.record_step(state_log_beliefs[:, 0, :])
            self.last_kernel_info = _structured_scan_kernel_info(
                path="cuda_masked",
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
                cache=cache,
            )
            return state_log_beliefs
        packed_transition_tables = self._get_packed_transition_tables(local_logits.device)
        if USE_CAUSAL_MACHINE_CUDA_SCAN and kernel_config.backend == "cuda":
            state_log_beliefs, _final_log_belief = causal_machine_scan_cuda(
                local_logits,
                self.transition_source_logits,
                self.transition_dest_logits,
                transition_context,
                prev_log_belief.to(dtype=local_logits.dtype),
                transition_gate.reshape(()),
                torch.sigmoid(self.transition_stay_logits.float()),
                packed_transition_tables=packed_transition_tables,
                chunk_size=1,
                runtime_config=runtime_config,
            )
            cache.record_step(state_log_beliefs[:, 0, :])
            dense_path = (
                "cuda_dense_lowp_tensor_core"
                if _structured_scan_uses_lowp_tensor_core_path(
                    device=local_logits.device,
                    dtype=local_logits.dtype,
                    kernel_config=kernel_config,
                    packed_transition_tables=packed_transition_tables,
                )
                else "cuda_dense"
            )
            self.last_kernel_info = _structured_scan_kernel_info(
                path=dense_path,
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
                cache=cache,
            )
            return state_log_beliefs
        transition_source_probs, transition_dest_probs, transition_stay_probs = self._structured_transition_params()
        state_log_beliefs, _final_log_belief, _ = structured_scan_fallback(
            local_logits,
            transition_context,
            prev_log_belief,
            transition_source_probs,
            transition_dest_probs,
            transition_stay_probs,
            transition_gate,
            chunk_size=1,
            tile_size=int(kernel_config.tile_size),
            split_size=int(kernel_config.split_size),
            backend=str(kernel_config.backend),
            track_transition_kl=False,
            runtime_config=runtime_config,
            packed_transition_tables=packed_transition_tables,
        )
        cache.record_step(state_log_beliefs[:, 0, :])
        self.last_kernel_info = _structured_scan_kernel_info(
            path=f"fallback_{kernel_config.backend}",
            kernel_config=kernel_config,
            runtime_config=runtime_config,
            packed_transition_tables=packed_transition_tables,
            cache=cache,
        )
        return state_log_beliefs

    def _decode(self, state_hidden: Tensor, state_log_beliefs: Tensor) -> Tensor:
        scale = F.softplus(self.output_scale).to(device=state_hidden.device, dtype=state_hidden.dtype)
        gate = torch.sigmoid(self.output_gate).to(device=state_hidden.device, dtype=state_hidden.dtype)
        belief_features = self.belief_out(state_log_beliefs.exp().to(dtype=state_hidden.dtype))
        hidden_gate = torch.tanh(self.hidden_gate(state_hidden))
        emit_scale = F.softplus(self.emit_delta_scale).to(device=state_hidden.device, dtype=state_hidden.dtype)
        return scale * gate * belief_features * (1.0 + emit_scale * hidden_gate)

    def forward_simple(self, x: Tensor, runtime_config: StructuredScanRuntimeConfig | None = None) -> Tensor:
        batch_size = int(x.size(0))
        state_hidden, local_logits, transition_context = self._project_hidden(x)
        latent_mode = self._resolved_latent_mode(int(x.size(1)), x.device)
        self.last_kernel_info = {}
        if latent_mode == "replace":
            state_log_beliefs, _, prior_log_beliefs = self._latent_replace_sequence_beliefs(
                local_logits,
                transition_context,
                state_hidden,
            )
            self.last_kernel_info = {
                "path": "latent_replace",
                "backend": "cuda" if bool(USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN and x.is_cuda and self.latent_rank > 0) else "python",
                "family": _latent_replace_kernel_family(local_logits),
                "uses_paged_cache": False,
            }
        else:
            if latent_mode == "additive":
                latent_logits = self._compute_latent_logits_sequence(state_hidden)
                if latent_logits is not None:
                    local_logits = local_logits + latent_logits.to(dtype=local_logits.dtype)
            initial_log_belief = self._initial_log_belief(batch_size, x.device, torch.float32)
            state_log_beliefs, _, prior_log_beliefs = self._filter_sequence(
                local_logits,
                transition_context,
                initial_log_belief,
                runtime_config=runtime_config,
            )
        future_sketch_pred = (
            self.future_sketch_head(state_hidden)
            if self.track_future_sketch and self.future_sketch_head is not None
            else None
        )
        self.last_aux = {
            "local_state_log_probs": F.log_softmax(local_logits.float(), dim=-1).to(dtype=x.dtype) if self.track_state_ce else None,
            "state_log_beliefs": state_log_beliefs if self.track_transition_kl else None,
            "prior_state_log_beliefs": prior_log_beliefs,
            "future_sketch_pred": future_sketch_pred,
            "block_hidden": None,
        }
        return self._decode(state_hidden, state_log_beliefs)

    def forward_step(
        self,
        x: Tensor,
        cache: CausalMachineCache,
        runtime_config: StructuredScanRuntimeConfig | None = None,
    ) -> Tensor:
        if x.size(1) != 1:
            raise ValueError(f"forward_step expects seq_len=1, got {tuple(x.shape)}")
        state_hidden, local_logits, transition_context = self._project_hidden(x)
        latent_mode = self._resolved_latent_mode(1, x.device)
        self.last_kernel_info = {}
        if runtime_config is not None and runtime_config.use_paged_cache and cache.paged_log_beliefs is None:
            runtime_config = _resolve_structured_scan_runtime_config(
                runtime_config,
                device=x.device,
                dtype=x.dtype,
                backend_family="decode_paged_cache",
                batch_size=int(x.size(0)),
                seq_len=1,
                num_states=self.num_states,
            )
            cache.enable_paged_history(
                batch_size=int(x.size(0)),
                num_states=self.num_states,
                device=x.device,
                dtype=x.dtype,
                latent_rank=self.latent_rank,
                page_size=runtime_config.page_size,
                max_pages=runtime_config.max_pages,
                workspace=runtime_config.workspace,
                grouped_launch_pack=runtime_config.grouped_launch_pack,
                save_mode=runtime_config.save_mode,
                paged_layout=runtime_config.paged_layout,
            )
        if (
            runtime_config is not None
            and runtime_config.use_paged_cache
            and cache.log_belief is None
            and cache.paged_log_beliefs is not None
            and not _structured_runtime_supports_fused_paged_step(runtime_config)
        ):
            cache.restore_latest_from_paged()
        if latent_mode == "replace":
            state_log_beliefs, _prior_log_beliefs = self._latent_replace_step_beliefs(
                local_logits,
                transition_context,
                state_hidden,
                cache,
            )
            self.last_kernel_info = {
                "path": "latent_replace",
                "backend": "cuda" if bool(USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN and x.is_cuda and self.latent_rank > 0) else "python",
                "family": _latent_replace_kernel_family(local_logits),
                "uses_paged_cache": bool(runtime_config.use_paged_cache) if runtime_config is not None else False,
                "paged_cache_write_backend": str(cache.last_paged_write_backend),
            }
        else:
            if latent_mode == "additive":
                latent_logits = self._compute_latent_logits_step(state_hidden, cache)
                if latent_logits is not None:
                    local_logits = local_logits + latent_logits.to(dtype=local_logits.dtype)
            state_log_beliefs = self._filter_step(
                local_logits,
                transition_context,
                cache,
                runtime_config=runtime_config,
            )
        decoded = self._decode(state_hidden, state_log_beliefs)
        if runtime_config is not None and runtime_config.use_paged_cache and runtime_config.paged_resident_only:
            cache.drop_resident_state()
        return decoded


class StateSpaceBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        mlp_mult: int,
        mlp_hidden: int,
        residual_alpha: float,
        train_seq_len: int,
        ln_scale: bool,
        use_adaptive_rmsnorm: bool,
        adaptive_rmsnorm_gate_init: float,
        norm_condition_dim: int,
        causal_machine_num_states: int,
        causal_machine_hidden_rank: int,
        causal_machine_transition_rank: int,
        causal_machine_latent_rank: int,
        causal_machine_latent_mode: str,
        causal_machine_scale_init: float,
        causal_machine_gate_init: float,
        causal_machine_latent_gate_init: float,
        causal_machine_latent_decay_init: float,
        causal_machine_transition_gate_init: float,
        causal_machine_transition_stickiness_init: float,
        causal_machine_emit_delta_scale_init: float,
        track_state_ce: bool,
        track_transition_kl: bool,
        track_future_sketch: bool,
    ):
        super().__init__()
        if causal_machine_num_states <= 0 or causal_machine_hidden_rank <= 0:
            raise ValueError("StateSpaceBlock requires positive causal-machine state and hidden sizes")
        self.layer_idx = layer_idx
        self.residual_alpha = float(residual_alpha)
        norm_scale = (1.0 / math.sqrt(layer_idx + 1)) if ln_scale else 1.0
        self.register_buffer("norm_scale_buffer", torch.tensor(norm_scale, dtype=torch.float32), persistent=False)
        if use_adaptive_rmsnorm:
            self.attn_norm = AdaptiveRMSNorm(dim, condition_dim=norm_condition_dim, gate_init=adaptive_rmsnorm_gate_init)
            self.mlp_norm = AdaptiveRMSNorm(dim, condition_dim=norm_condition_dim, gate_init=adaptive_rmsnorm_gate_init)
        else:
            self.attn_norm = nn.RMSNorm(dim, elementwise_affine=False)
            self.mlp_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = CausalStateMixer(
            dim=dim,
                hidden_rank=causal_machine_hidden_rank,
                num_states=causal_machine_num_states,
                transition_rank=causal_machine_transition_rank,
                latent_rank=causal_machine_latent_rank,
                latent_mode=causal_machine_latent_mode,
                scale_init=causal_machine_scale_init,
            gate_init=causal_machine_gate_init,
            latent_gate_init=causal_machine_latent_gate_init,
            latent_decay_init=causal_machine_latent_decay_init,
            transition_gate_init=causal_machine_transition_gate_init,
            transition_stickiness_init=causal_machine_transition_stickiness_init,
            emit_delta_scale_init=causal_machine_emit_delta_scale_init,
            train_seq_len=train_seq_len,
            track_state_ce=track_state_ce,
            track_transition_kl=track_transition_kl,
            track_future_sketch=track_future_sketch,
        )
        self.mlp = MLP(dim, mlp_mult, hidden_dim=mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.fast_path = True

    def forward_simple(
        self,
        x: Tensor,
        x0: Tensor,
        q_gain_delta: Tensor | None = None,
        norm_condition: Tensor | None = None,
    ) -> Tensor:
        del q_gain_delta
        mix = self.resid_mix.to(dtype=x.dtype)
        norm_scale = self.norm_scale_buffer.to(device=x.device, dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_normed = self.attn_norm(x, condition=norm_condition) if isinstance(self.attn_norm, AdaptiveRMSNorm) else self.attn_norm(x)
        attn_normed = attn_normed * norm_scale
        attn_out = self.attn.forward_simple(attn_normed)
        self.last_aux = dict(self.attn.last_aux)
        x = x * self.residual_alpha + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_normed = self.mlp_norm(x, condition=norm_condition) if isinstance(self.mlp_norm, AdaptiveRMSNorm) else self.mlp_norm(x)
        mlp_normed = mlp_normed * norm_scale
        x = x * self.residual_alpha + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_normed)
        self.last_aux["block_hidden"] = x
        return x

    def forward_simple_step(
        self,
        x: Tensor,
        x0: Tensor,
        cache: BlockStepCache,
        position: int,
        q_gain_delta: Tensor | None = None,
        norm_condition: Tensor | None = None,
    ) -> Tensor:
        del position, q_gain_delta
        if x.size(1) != 1:
            raise ValueError(f"forward_simple_step expects seq_len=1, got {tuple(x.shape)}")
        if cache.state_cache is None:
            cache.state_cache = CausalMachineCache()
        mix = self.resid_mix.to(dtype=x.dtype)
        norm_scale = self.norm_scale_buffer.to(device=x.device, dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_normed = self.attn_norm(x, condition=norm_condition) if isinstance(self.attn_norm, AdaptiveRMSNorm) else self.attn_norm(x)
        attn_normed = attn_normed * norm_scale
        attn_out = self.attn.forward_step(attn_normed, cache.state_cache)
        self.last_aux = {}
        x = x * self.residual_alpha + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_normed = self.mlp_norm(x, condition=norm_condition) if isinstance(self.mlp_norm, AdaptiveRMSNorm) else self.mlp_norm(x)
        mlp_normed = mlp_normed * norm_scale
        x = x * self.residual_alpha + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_normed)
        return x

    def init_step_cache(self, max_len: int | None = None) -> BlockStepCache:
        del max_len
        return BlockStepCache(state_cache=CausalMachineCache())

    def reset_step_cache(self, cache: BlockStepCache) -> None:
        cache.reset()


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int,
        tie_embeddings: bool,
        use_output_logit_bias: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        overtone_embed_init: bool,
        overtone_embed_power: float,
        overtone_embed_scale: float,
        resid_mix_phase_init: bool,
        resid_mix_phase_sharpness: float,
        resid_mix_phase_center: float,
        num_shared_layers: int,
        shared_layer_repeats: int,
        attention_kv_mode: str,
        use_causal_machine_bias: bool,
        use_causal_machine_output_bias: bool,
        use_causal_machine_backbone: bool,
        block_pattern: str,
        causal_machine_profile_json: str,
        causal_machine_num_states: int,
        causal_machine_hidden_rank: int,
        causal_machine_transition_rank: int,
        causal_machine_latent_rank: int,
        causal_machine_latent_mode: str,
        causal_machine_scale_init: float,
        causal_machine_gate_init: float,
        causal_machine_latent_gate_init: float,
        causal_machine_latent_decay_init: float,
        causal_machine_teacher_loss_coeff: float,
        causal_machine_state_loss_coeff: float,
        causal_machine_next_token_loss_coeff: float,
        causal_machine_transition_kl_coeff: float,
        causal_machine_future_sketch_loss_coeff: float,
        causal_machine_transition_gate_init: float,
        causal_machine_transition_stickiness_init: float,
        causal_machine_emit_delta_scale_init: float,
        shared_tail_output_gate: bool,
        shared_tail_output_init: float,
        shared_tail_enable_step: int,
        shared_tail_ramp_steps: int,
        shared_tail_max_mult: float,
        signed_skip_weights: bool,
        orthogonal_init: bool,
        mup_proj_init: bool,
        train_seq_len: int,
        use_flash_attn_3: bool,
        rope_dims: int,
        ln_scale: bool,
        use_adaptive_rmsnorm: bool,
        adaptive_rmsnorm_gate_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.vocab_size = max(int(vocab_size), 1)
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.overtone_embed_init = overtone_embed_init
        self.overtone_embed_power = overtone_embed_power
        self.overtone_embed_scale = overtone_embed_scale
        self.resid_mix_phase_init = resid_mix_phase_init
        self.resid_mix_phase_sharpness = resid_mix_phase_sharpness
        self.resid_mix_phase_center = float(resid_mix_phase_center)
        self.num_layers = max(int(num_layers), 0)
        self.use_adaptive_rmsnorm = bool(use_adaptive_rmsnorm)
        self.adaptive_rmsnorm_gate_init = float(adaptive_rmsnorm_gate_init)
        self.num_heads = max(int(num_heads), 1)
        self.num_kv_heads = max(int(num_kv_heads), 1)
        self.attention_kv_mode = str(attention_kv_mode or "").strip().lower() or "gqa"
        self.use_causal_machine_output_bias = bool(use_causal_machine_output_bias or use_causal_machine_bias)
        self.use_causal_machine_bias = self.use_causal_machine_output_bias
        self.use_causal_machine_backbone = bool(use_causal_machine_backbone)
        self.block_pattern = normalize_block_pattern_spec(block_pattern, self.num_layers)
        if self.block_pattern:
            self.block_types = self.block_pattern.split(",")
            if any(block_type == "ssm" for block_type in self.block_types):
                self.use_causal_machine_backbone = True
        else:
            self.block_types = (
                default_block_pattern(self.num_layers).split(",")
                if self.use_causal_machine_backbone
                else ["attn"] * self.num_layers
            )
        self.block_pattern = ",".join(self.block_types)
        self.causal_machine_profile_json = str(causal_machine_profile_json or "").strip()
        self.causal_machine_num_states = max(int(causal_machine_num_states), 0)
        self.causal_machine_hidden_rank = max(int(causal_machine_hidden_rank), 0)
        self.causal_machine_transition_rank = max(int(causal_machine_transition_rank), 1)
        self.causal_machine_latent_rank = max(int(causal_machine_latent_rank), 0)
        self.causal_machine_latent_mode = str(causal_machine_latent_mode or "off").strip().lower()
        if self.causal_machine_latent_rank <= 0:
            self.causal_machine_latent_mode = "off"
        self.causal_machine_replace_uses_structured = _env_enabled(
            "CAUSAL_MACHINE_GLOBAL_REPLACE_STRUCTURED",
            default=_competition_mode_enabled(),
        )
        self.causal_machine_uses_structured_transition_params = (
            self.causal_machine_num_states > 0
            and (
                self.causal_machine_latent_mode != "replace"
                or self.causal_machine_replace_uses_structured
            )
        )
        self.causal_machine_scale_init = max(float(causal_machine_scale_init), 0.0)
        self.causal_machine_gate_init = float(causal_machine_gate_init)
        self.causal_machine_latent_gate_init = float(causal_machine_latent_gate_init)
        self.causal_machine_latent_decay_init = float(causal_machine_latent_decay_init)
        self.causal_machine_teacher_loss_coeff = max(float(causal_machine_teacher_loss_coeff), 0.0)
        self.causal_machine_state_loss_coeff = max(float(causal_machine_state_loss_coeff), 0.0)
        self.causal_machine_next_token_loss_coeff = max(float(causal_machine_next_token_loss_coeff), 0.0)
        self.causal_machine_transition_kl_coeff = max(float(causal_machine_transition_kl_coeff), 0.0)
        self.causal_machine_future_sketch_loss_coeff = max(float(causal_machine_future_sketch_loss_coeff), 0.0)
        self.causal_machine_transition_gate_init = float(causal_machine_transition_gate_init)
        self.causal_machine_transition_stickiness_init = float(causal_machine_transition_stickiness_init)
        self.causal_machine_emit_delta_scale_init = max(float(causal_machine_emit_delta_scale_init), 0.0)
        self.causal_machine_online_teacher_ema = float(DEFAULT_CAUSAL_MACHINE_ONLINE_TEACHER_EMA)
        self.causal_machine_filter_chunk_size = _resolve_structured_scan_chunk_size(
            train_seq_len,
            self.causal_machine_transition_rank,
        )
        self._global_causal_auto_mode_cache: dict[tuple[int, int, int, int, str], str] = {}
        self._global_packed_transition_cache: dict[str, object] = {}
        self._global_sparse_transition_cache: dict[str, object] = {}
        self._global_reduced_transition_cache: dict[str, object] = {}
        self._compiled_global_kernel_config_cache: dict[tuple[str, int, int, bool], StructuredScanKernelConfig] = {}
        self.shared_tail_output_gate = bool(shared_tail_output_gate)
        self.shared_tail_output_init = float(shared_tail_output_init)
        self.shared_tail_enable_step = max(int(shared_tail_enable_step), 0)
        self.shared_tail_ramp_steps = max(int(shared_tail_ramp_steps), 0)
        self.shared_tail_max_mult = min(max(float(shared_tail_max_mult), 0.0), 1.0)
        self.signed_skip_weights = bool(signed_skip_weights)
        self.rope_dims = max(int(rope_dims), 0)
        self.ln_scale = bool(ln_scale)
        self.train_seq_len = max(int(train_seq_len), 1)
        self.current_training_step = 0
        self.orthogonal_init = orthogonal_init
        self.mup_proj_init = mup_proj_init
        self.norm_condition_dim = 0 if not self.use_adaptive_rmsnorm else 2
        self.fake_quant_bits = 0
        self.num_shared_layers = max(int(num_shared_layers), 0)
        self.shared_layer_repeats = max(int(shared_layer_repeats), 0)
        residual_alpha = 1.0
        self.use_output_logit_bias = bool(use_output_logit_bias)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.shared_tail_gate = (
            nn.Parameter(torch.tensor(self.shared_tail_output_init, dtype=torch.float32))
            if self.shared_tail_output_gate
            else None
        )
        self.register_buffer("shared_tail_schedule_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.blocks = nn.ModuleList(
            [
                self._build_backbone_block(
                    block_type=self.block_types[i],
                    layer_idx=i,
                    model_dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_mult=mlp_mult,
                    mlp_hidden=mlp_hidden,
                    rope_base=rope_base,
                    qk_gain_init=qk_gain_init,
                    residual_alpha=residual_alpha,
                    train_seq_len=train_seq_len,
                    use_flash_attn_3=use_flash_attn_3,
                )
                for i in range(num_layers)
            ]
        )
        self.shared_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    num_layers + i,
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    mlp_hidden,
                    rope_base,
                    qk_gain_init,
                    residual_alpha=residual_alpha,
                    train_seq_len=train_seq_len,
                    use_flash_attn_3=use_flash_attn_3,
                    rope_dims=self.rope_dims,
                    ln_scale=self.ln_scale,
                    use_adaptive_rmsnorm=self.use_adaptive_rmsnorm,
                    adaptive_rmsnorm_gate_init=self.adaptive_rmsnorm_gate_init,
                    norm_condition_dim=self.norm_condition_dim,
                )
                for i in range(self.num_shared_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(model_dim, elementwise_affine=False)
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.output_logit_bias = (
            nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32)) if self.use_output_logit_bias else None
        )
        causal_machine_enabled = (
            self.use_causal_machine_output_bias
            and self.causal_machine_hidden_rank > 0
            and self.causal_machine_num_states > 0
        )
        self.causal_machine_sketch_dim = 0
        self.causal_machine_horizons = []
        self.causal_machine_prefix_down = (
            CastedLinear(model_dim, self.causal_machine_hidden_rank, bias=False)
            if causal_machine_enabled
            else None
        )
        self.causal_machine_decoder_hidden = (
            CastedLinear(self.causal_machine_hidden_rank, self.causal_machine_hidden_rank, bias=True)
            if causal_machine_enabled
            else None
        )
        self.causal_machine_state_head = (
            CastedLinear(self.causal_machine_hidden_rank, self.causal_machine_num_states, bias=False)
            if causal_machine_enabled
            else None
        )
        self.causal_machine_transition_context = (
            CastedLinear(self.causal_machine_hidden_rank, self.causal_machine_num_states, bias=False)
            if causal_machine_enabled
            else None
        )
        self.causal_machine_latent_in = (
            CastedLinear(self.causal_machine_hidden_rank, self.causal_machine_latent_rank, bias=False)
            if causal_machine_enabled and self.causal_machine_latent_rank > 0
            else None
        )
        self.causal_machine_latent_gate_in = (
            CastedLinear(self.causal_machine_hidden_rank, self.causal_machine_latent_rank, bias=True)
            if causal_machine_enabled and self.causal_machine_latent_rank > 0
            else None
        )
        self.causal_machine_latent_out = (
            CastedLinear(self.causal_machine_latent_rank, self.causal_machine_num_states, bias=False)
            if causal_machine_enabled and self.causal_machine_latent_rank > 0
            else None
        )
        self.causal_machine_scale = (
            nn.Parameter(torch.tensor(inverse_softplus_scalar(self.causal_machine_scale_init), dtype=torch.float32))
            if causal_machine_enabled
            else None
        )
        self.causal_machine_gate = (
            nn.Parameter(torch.tensor(self.causal_machine_gate_init, dtype=torch.float32))
            if causal_machine_enabled
            else None
        )
        self.causal_machine_latent_output_gate = (
            nn.Parameter(torch.tensor(self.causal_machine_latent_gate_init, dtype=torch.float32))
            if causal_machine_enabled and self.causal_machine_latent_rank > 0
            else None
        )
        self.causal_machine_latent_decay_logits = (
            nn.Parameter(
                torch.full(
                    (self.causal_machine_latent_rank,),
                    fill_value=inverse_sigmoid_scalar(self.causal_machine_latent_decay_init),
                    dtype=torch.float32,
                )
            )
            if causal_machine_enabled and self.causal_machine_latent_rank > 0
            else None
        )
        self.causal_machine_transition_gate = (
            nn.Parameter(torch.tensor(self.causal_machine_transition_gate_init, dtype=torch.float32))
            if causal_machine_enabled
            else None
        )
        self.causal_machine_emit_delta_scale = (
            nn.Parameter(
                torch.tensor(inverse_softplus_scalar(self.causal_machine_emit_delta_scale_init), dtype=torch.float32)
            )
            if causal_machine_enabled
            else None
        )
        self.use_causal_machine_cuda_scan = bool(
            USE_CAUSAL_MACHINE_CUDA_SCAN
            and causal_machine_enabled
            and supports_structured_scan_cuda_config(
                self.causal_machine_num_states,
                self.causal_machine_transition_rank,
            )
        )
        if causal_machine_enabled:
            causal_machine_log_probs = torch.full(
                (self.causal_machine_num_states, vocab_size),
                fill_value=-math.log(max(vocab_size, 1)),
                dtype=torch.float32,
            )
            causal_machine_centroids = torch.empty((0, 0), dtype=torch.float32)
            causal_machine_horizons = torch.empty((0,), dtype=torch.int64)
            causal_machine_log_state_priors = torch.full(
                (self.causal_machine_num_states,),
                fill_value=-math.log(max(self.causal_machine_num_states, 1)),
                dtype=torch.float32,
            )
            if self.causal_machine_uses_structured_transition_params:
                transition_source_init = torch.zeros(
                    (self.causal_machine_num_states, self.causal_machine_transition_rank),
                    dtype=torch.float32,
                )
                transition_dest_init = torch.zeros(
                    (self.causal_machine_transition_rank, self.causal_machine_num_states),
                    dtype=torch.float32,
                )
                transition_stay_init = torch.full(
                    (self.causal_machine_num_states,),
                    fill_value=float(self.causal_machine_transition_stickiness_init),
                    dtype=torch.float32,
                )
            else:
                transition_source_init = torch.empty((0, 0), dtype=torch.float32)
                transition_dest_init = torch.empty((0, 0), dtype=torch.float32)
                transition_stay_init = torch.empty((0,), dtype=torch.float32)
            causal_machine_emit_delta = torch.empty_like(causal_machine_log_probs, dtype=torch.float32)
            bucket_ids_t = torch.empty((0,), dtype=torch.int64)
            signs_t = torch.empty((0,), dtype=torch.float32)
            online_state_counts = torch.zeros((self.causal_machine_num_states,), dtype=torch.float32)
        else:
            causal_machine_log_probs = torch.empty((0, 0), dtype=torch.float32)
            causal_machine_centroids = torch.empty((0, 0), dtype=torch.float32)
            causal_machine_horizons = torch.empty((0,), dtype=torch.int64)
            causal_machine_log_state_priors = torch.empty((0,), dtype=torch.float32)
            transition_source_init = torch.empty((0, 0), dtype=torch.float32)
            transition_dest_init = torch.empty((0, 0), dtype=torch.float32)
            transition_stay_init = torch.empty((0,), dtype=torch.float32)
            causal_machine_emit_delta = torch.empty((0, 0), dtype=torch.float32)
            bucket_ids_t = torch.empty((0,), dtype=torch.int64)
            signs_t = torch.empty((0,), dtype=torch.float32)
            online_state_counts = torch.empty((0,), dtype=torch.float32)
        self.register_buffer("causal_machine_log_probs", causal_machine_log_probs, persistent=True)
        self.register_buffer("causal_machine_log_state_priors", causal_machine_log_state_priors, persistent=True)
        self.causal_machine_transition_source_logits = (
            nn.Parameter(transition_source_init) if causal_machine_enabled and self.causal_machine_uses_structured_transition_params else None
        )
        self.causal_machine_transition_dest_logits = (
            nn.Parameter(transition_dest_init) if causal_machine_enabled and self.causal_machine_uses_structured_transition_params else None
        )
        self.causal_machine_transition_stay_logits = (
            nn.Parameter(transition_stay_init) if causal_machine_enabled and self.causal_machine_uses_structured_transition_params else None
        )
        self.causal_machine_emit_delta = (
            nn.Parameter(causal_machine_emit_delta) if causal_machine_enabled else None
        )
        self.register_buffer("causal_machine_signature_centroids", causal_machine_centroids, persistent=False)
        self.register_buffer("causal_machine_horizon_tensor", causal_machine_horizons, persistent=False)
        self.register_buffer("causal_machine_bucket_ids", bucket_ids_t, persistent=False)
        self.register_buffer("causal_machine_signs", signs_t, persistent=False)
        self.register_buffer("causal_machine_online_state_counts", online_state_counts, persistent=False)
        self.register_buffer("causal_machine_online_teacher_ready", torch.zeros((), dtype=torch.bool), persistent=False)
        self.causal_machine_profile_loaded = False
        self.last_causal_machine_kernel_info: dict[str, object] = {}
        needs_default_online_sketch = (
            self.causal_machine_num_states > 0
            and self.vocab_size > 0
            and (
                self.causal_machine_future_sketch_loss_coeff > 0.0
                or self.causal_machine_state_loss_coeff > 0.0
                or self.causal_machine_teacher_loss_coeff > 0.0
            )
        )
        if needs_default_online_sketch:
            default_sketch_dim = max(4, min(DEFAULT_CAUSAL_MACHINE_SKETCH_DIM, self.vocab_size))
            default_horizons = list(DEFAULT_CAUSAL_MACHINE_SKETCH_HORIZONS)
            bucket_ids_np, signs_np = _sketch_token_tables(self.vocab_size, default_sketch_dim)
            self.causal_machine_horizon_tensor = torch.tensor(default_horizons, dtype=torch.int64)
            self.causal_machine_bucket_ids = torch.from_numpy(bucket_ids_np.astype(np.int64, copy=False))
            self.causal_machine_signs = torch.from_numpy(signs_np.astype(np.float32, copy=False))
            self.causal_machine_sketch_dim = default_sketch_dim
            self.causal_machine_horizons = default_horizons
            total_sketch_dim = default_sketch_dim * len(default_horizons)
            self.causal_machine_signature_centroids = torch.zeros(
                (self.causal_machine_num_states, total_sketch_dim), dtype=torch.float32
            )
            self.causal_machine_online_state_counts = torch.zeros(
                (self.causal_machine_num_states,), dtype=torch.float32
            )
            for block in self.blocks:
                if isinstance(block, StateSpaceBlock):
                    block.attn.configure_future_sketch(total_sketch_dim)
        self.mid_aux_head = CastedLinear(model_dim, vocab_size, bias=False)
        self.mid_aux_head._zero_init = True
        self.fast_features_path = all(block.fast_path for block in self.blocks)
        self._init_weights()

    def _build_backbone_block(
        self,
        block_type: str,
        layer_idx: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int,
        rope_base: float,
        qk_gain_init: float,
        residual_alpha: float,
        train_seq_len: int,
        use_flash_attn_3: bool,
    ) -> nn.Module:
        if block_type == "attn":
            return AttentionBlock(
                layer_idx,
                model_dim,
                num_heads,
                num_kv_heads,
                mlp_mult,
                mlp_hidden,
                rope_base,
                qk_gain_init,
                residual_alpha=residual_alpha,
                train_seq_len=train_seq_len,
                use_flash_attn_3=use_flash_attn_3,
                rope_dims=self.rope_dims,
                ln_scale=self.ln_scale,
                use_adaptive_rmsnorm=self.use_adaptive_rmsnorm,
                adaptive_rmsnorm_gate_init=self.adaptive_rmsnorm_gate_init,
                norm_condition_dim=self.norm_condition_dim,
            )
        if block_type == "ssm":
            return StateSpaceBlock(
                layer_idx=layer_idx,
                dim=model_dim,
                mlp_mult=mlp_mult,
                mlp_hidden=mlp_hidden,
                residual_alpha=residual_alpha,
                train_seq_len=train_seq_len,
                ln_scale=self.ln_scale,
                use_adaptive_rmsnorm=self.use_adaptive_rmsnorm,
                adaptive_rmsnorm_gate_init=self.adaptive_rmsnorm_gate_init,
                norm_condition_dim=self.norm_condition_dim,
                causal_machine_num_states=self.causal_machine_num_states,
                causal_machine_hidden_rank=self.causal_machine_hidden_rank,
                causal_machine_transition_rank=self.causal_machine_transition_rank,
                causal_machine_latent_rank=self.causal_machine_latent_rank,
                causal_machine_latent_mode=self.causal_machine_latent_mode,
                causal_machine_scale_init=self.causal_machine_scale_init,
                causal_machine_gate_init=self.causal_machine_gate_init,
                causal_machine_latent_gate_init=self.causal_machine_latent_gate_init,
                causal_machine_latent_decay_init=self.causal_machine_latent_decay_init,
                causal_machine_transition_gate_init=self.causal_machine_transition_gate_init,
                causal_machine_transition_stickiness_init=self.causal_machine_transition_stickiness_init,
                causal_machine_emit_delta_scale_init=self.causal_machine_emit_delta_scale_init,
                track_state_ce=self.causal_machine_state_loss_coeff > 0.0,
                track_transition_kl=self.causal_machine_transition_kl_coeff > 0.0,
                track_future_sketch=self.causal_machine_future_sketch_loss_coeff > 0.0,
            )
        raise ValueError(f"Unknown block type {block_type!r}")

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            if self.overtone_embed_init:
                self._init_overtone_embedding()
            else:
                nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.causal_machine_prefix_down is not None:
            nn.init.normal_(self.causal_machine_prefix_down.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.causal_machine_decoder_hidden is not None:
            nn.init.normal_(self.causal_machine_decoder_hidden.weight, mean=0.0, std=self.tied_embed_init_std)
            if self.causal_machine_decoder_hidden.bias is not None:
                nn.init.zeros_(self.causal_machine_decoder_hidden.bias)
        if self.causal_machine_latent_in is not None:
            nn.init.normal_(self.causal_machine_latent_in.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.causal_machine_latent_gate_in is not None:
            nn.init.normal_(self.causal_machine_latent_gate_in.weight, mean=0.0, std=self.tied_embed_init_std)
            if self.causal_machine_latent_gate_in.bias is not None:
                nn.init.zeros_(self.causal_machine_latent_gate_in.bias)
        if self.causal_machine_latent_out is not None:
            nn.init.normal_(self.causal_machine_latent_out.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.causal_machine_state_head is not None:
            nn.init.normal_(self.causal_machine_state_head.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.causal_machine_transition_context is not None:
            nn.init.zeros_(self.causal_machine_transition_context.weight)
        if self.causal_machine_emit_delta is not None:
            nn.init.normal_(self.causal_machine_emit_delta, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
        if self.orthogonal_init:
            self._apply_orthogonal_init()
        if self.resid_mix_phase_init:
            self._init_resid_mix_phase_schedule()

    def _apply_orthogonal_init(self) -> None:
        proj_scale = 1.0 / math.sqrt(max(2 * len(self.blocks), 1)) if self.mup_proj_init else 1.0
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if getattr(module, "_zero_init", False):
                continue
            if module.weight.ndim != 2 or min(module.weight.shape) < 64:
                continue
            nn.init.orthogonal_(module.weight, gain=1.0)
            if self.mup_proj_init and (
                name.endswith(".proj")
                or name.endswith(".attn.proj")
                or name.endswith(".mlp.proj")
            ):
                with torch.no_grad():
                    module.weight.mul_(proj_scale)

    def _init_overtone_embedding(self) -> None:
        rows, cols = self.tok_emb.weight.shape
        with torch.no_grad():
            basis = torch.randn((rows, cols), dtype=torch.float32, device=self.tok_emb.weight.device)
            q, _ = torch.linalg.qr(basis, mode="reduced")
            ranks = torch.arange(1, cols + 1, dtype=torch.float32, device=q.device)
            spectrum = ranks.pow(-self.overtone_embed_power)
            spectrum = spectrum / spectrum.norm() * (cols**0.5) * self.tied_embed_init_std * self.overtone_embed_scale
            self.tok_emb.weight.copy_(q * spectrum.unsqueeze(0))

    def _init_resid_mix_phase_schedule(self) -> None:
        total = max(len(self.blocks) - 1, 1)
        with torch.no_grad():
            for idx, block in enumerate(self.blocks):
                depth = idx / total
                alpha = torch.sigmoid(
                    torch.tensor((depth - self.resid_mix_phase_center) * self.resid_mix_phase_sharpness, dtype=torch.float32)
                ).item()
                block.resid_mix[0].fill_(alpha)
                block.resid_mix[1].fill_(1.0 - alpha)

    def _skip_scale(self, idx: int, dtype: torch.dtype) -> Tensor:
        scale = self.skip_weights[idx].to(dtype=dtype)
        return torch.tanh(scale) if self.signed_skip_weights else scale

    def set_fake_quant(self, bits: int) -> None:
        self.fake_quant_bits = max(int(bits), 0)
        for module in self.modules():
            if isinstance(module, CastedLinear):
                module.fake_quant_bits = self.fake_quant_bits

    def _shared_tail_schedule_mult(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.shared_tail_schedule_mult_buffer.to(device=device, dtype=dtype)

    def _embedding_weight(self) -> Tensor:
        weight = self.tok_emb.weight
        if self.fake_quant_bits > 0:
            weight = fake_quantize_tensor(weight, self.fake_quant_bits)
        return weight

    def _apply_output_head(self, head: nn.Module, x: Tensor) -> Tensor:
        if isinstance(head, CastedLinear):
            return head(x)
        weight = getattr(head, "weight", None)
        if isinstance(weight, Tensor):
            return head(x.to(dtype=weight.dtype))
        return head(x)

    def _mid_aux_logits(self, x: Tensor) -> Tensor:
        x = self.final_norm(x, condition=None) if isinstance(self.final_norm, AdaptiveRMSNorm) else self.final_norm(x)
        logits_proj = self._apply_output_head(self.mid_aux_head, x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def _compute_causal_machine_state_features(self, x: Tensor) -> Tensor:
        prefix_cumsum = x.cumsum(dim=1)
        prefix_sum = torch.cat([torch.zeros_like(prefix_cumsum[:, :1, :]), prefix_cumsum[:, :-1, :]], dim=1)
        seq_len = int(x.size(1))
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype).view(1, seq_len, 1)
        denom = positions.clamp_min(1.0)
        prefix_mean = prefix_sum / denom
        return x + prefix_mean

    def init_causal_machine_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        runtime_config: StructuredScanRuntimeConfig | None = None,
    ) -> CausalMachineCache:
        if self.causal_machine_num_states <= 0:
            return CausalMachineCache(log_belief=None, num_updates=0)
        if runtime_config is not None:
            runtime_config = _resolve_structured_scan_runtime_config(
                runtime_config,
                device=device,
                dtype=dtype,
                backend_family="paged_cache",
                batch_size=batch_size,
                seq_len=1,
                num_states=self.causal_machine_num_states,
            )
        if self.causal_machine_log_state_priors.numel() == self.causal_machine_num_states:
            log_belief = self.causal_machine_log_state_priors.to(device=device, dtype=dtype).unsqueeze(0).expand(
                batch_size, -1
            ).contiguous()
        else:
            log_belief = torch.full(
                (batch_size, self.causal_machine_num_states),
                fill_value=-math.log(max(self.causal_machine_num_states, 1)),
                device=device,
                dtype=dtype,
            )
        cache = CausalMachineCache(log_belief=log_belief.clone(), num_updates=0)
        if runtime_config is not None and runtime_config.use_paged_cache:
            cache.enable_paged_history(
                batch_size=batch_size,
                num_states=self.causal_machine_num_states,
                device=device,
                dtype=dtype,
                latent_rank=self.causal_machine_latent_rank,
                page_size=runtime_config.page_size,
                max_pages=runtime_config.max_pages,
                workspace=runtime_config.workspace,
                grouped_launch_pack=runtime_config.grouped_launch_pack,
                save_mode=runtime_config.save_mode,
                paged_layout=runtime_config.paged_layout,
            )
        return cache

    def _compute_causal_machine_emit_log_probs(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        emit_logits = self.causal_machine_log_probs.to(device=device, dtype=torch.float32)
        if self.causal_machine_emit_delta is not None and self.causal_machine_emit_delta_scale is not None:
            emit_delta_scale = F.softplus(self.causal_machine_emit_delta_scale.float())
            emit_logits = emit_logits + emit_delta_scale * self.causal_machine_emit_delta.float()
        return F.log_softmax(emit_logits, dim=-1).to(dtype=dtype)

    def _compute_causal_machine_local_logits_and_transition_context(
        self,
        state_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.causal_machine_decoder_hidden is not None:
            state_hidden = torch.tanh(self._apply_output_head(self.causal_machine_decoder_hidden, state_hidden))
        if self.causal_machine_state_head is None:
            return state_hidden, torch.zeros_like(state_hidden)
        local_logits = self._apply_output_head(self.causal_machine_state_head, state_hidden)
        if self.causal_machine_transition_context is None:
            transition_context = torch.zeros_like(local_logits)
        else:
            transition_context = self._apply_output_head(self.causal_machine_transition_context, state_hidden)
        return local_logits, transition_context

    def _uses_global_latent_additive(self) -> bool:
        return self.causal_machine_latent_mode == "additive" and self.causal_machine_latent_rank > 0

    def _uses_global_latent_replace(self) -> bool:
        return self.causal_machine_latent_mode == "replace" and self.causal_machine_latent_rank > 0

    def _resolved_global_latent_mode(self, seq_len: int, device: torch.device) -> str:
        if self.causal_machine_latent_rank <= 0:
            return "off"
        if self.causal_machine_latent_mode in {"off", "additive", "replace"}:
            return self.causal_machine_latent_mode
        # Keep "auto" aligned with the intended competition posture: latent replace
        # is the primary engine whenever latent state is enabled. Structured exact
        # remains available only through explicit "off".
        device_key = f"{device.type}:{device.index if device.index is not None else -1}"
        cache_key = (
            int(seq_len),
            int(self.causal_machine_transition_rank),
            int(self.causal_machine_latent_rank),
            int(self.causal_machine_num_states),
            device_key,
        )
        cached = self._global_causal_auto_mode_cache.get(cache_key)
        if cached is not None:
            return cached
        mode = "replace"
        self._global_causal_auto_mode_cache[cache_key] = mode
        return mode

    def _compute_global_latent_logits_sequence(self, state_hidden: Tensor) -> Tensor | None:
        if (
            self.causal_machine_latent_in is None
            or self.causal_machine_latent_gate_in is None
            or self.causal_machine_latent_out is None
            or self.causal_machine_latent_output_gate is None
            or self.causal_machine_latent_decay_logits is None
        ):
            return None
        drive = self._apply_output_head(self.causal_machine_latent_in, state_hidden).float()
        drive = drive * torch.sigmoid(self._apply_output_head(self.causal_machine_latent_gate_in, state_hidden).float())
        decay = _bounded_latent_decay(self.causal_machine_latent_decay_logits)
        initial_state = torch.zeros((drive.size(0), self.causal_machine_latent_rank), device=drive.device, dtype=torch.float32)
        use_cuda_latent_scan = bool(
            USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN
            and drive.is_cuda
            and drive.size(-1) > 0
        )
        if use_cuda_latent_scan:
            latent_states, _prior_states, _final_state = causal_machine_latent_scan_cuda(
                drive.to(dtype=state_hidden.dtype),
                decay,
                initial_state.to(dtype=state_hidden.dtype),
            )
            latent_hidden = torch.tanh(latent_states).to(dtype=state_hidden.dtype)
        else:
            prev = torch.zeros((drive.size(0), self.causal_machine_latent_rank), device=drive.device, dtype=torch.float32)
            latent_chunks: list[Tensor] = []
            chunk_size = max(int(self.causal_machine_filter_chunk_size), 1)
            decay_b = decay.view(1, 1, self.causal_machine_latent_rank)
            for chunk_start in range(0, int(drive.size(1)), chunk_size):
                chunk = drive[:, chunk_start : chunk_start + chunk_size, :]
                chunk_len = int(chunk.size(1))
                positions = torch.arange(chunk_len, device=drive.device, dtype=torch.float32).view(1, chunk_len, 1)
                powers = torch.pow(decay_b, positions)
                scaled = chunk / powers.clamp_min(1.0e-8)
                prefix = torch.cumsum(scaled, dim=1)
                chunk_states = powers * (decay_b * prev.unsqueeze(1) + prefix)
                latent_chunks.append(chunk_states)
                prev = chunk_states[:, -1, :]
            latent_hidden = torch.tanh(torch.cat(latent_chunks, dim=1)).to(dtype=state_hidden.dtype)
        gate = _bounded_gate(
            self.causal_machine_latent_output_gate,
            "CAUSAL_MACHINE_LATENT_OUTPUT_GATE_MIN",
            "CAUSAL_MACHINE_LATENT_OUTPUT_GATE_MAX",
            default_min=0.10,
            default_max=0.995,
        ).to(device=state_hidden.device, dtype=state_hidden.dtype)
        return gate * self._apply_output_head(self.causal_machine_latent_out, latent_hidden)

    def _compute_global_latent_logits_step(
        self,
        state_hidden: Tensor,
        cache: CausalMachineCache | None,
        update_cache: bool,
    ) -> Tensor | None:
        if (
            self.causal_machine_latent_in is None
            or self.causal_machine_latent_gate_in is None
            or self.causal_machine_latent_out is None
            or self.causal_machine_latent_output_gate is None
            or self.causal_machine_latent_decay_logits is None
        ):
            return None
        drive = self._apply_output_head(self.causal_machine_latent_in, state_hidden).float()
        drive = drive * torch.sigmoid(self._apply_output_head(self.causal_machine_latent_gate_in, state_hidden).float())
        if cache is not None and cache.latent_state is not None:
            prev = cache.latent_state.to(device=state_hidden.device, dtype=torch.float32)
        else:
            prev = torch.zeros((drive.size(0), self.causal_machine_latent_rank), device=drive.device, dtype=torch.float32)
        decay = _bounded_latent_decay(self.causal_machine_latent_decay_logits).view(1, -1)
        next_state = decay * prev + drive[:, 0, :]
        if update_cache and cache is not None:
            cache.latent_state = next_state.detach()
        latent_hidden = torch.tanh(next_state).to(dtype=state_hidden.dtype).unsqueeze(1)
        gate = _bounded_gate(
            self.causal_machine_latent_output_gate,
            "CAUSAL_MACHINE_LATENT_OUTPUT_GATE_MIN",
            "CAUSAL_MACHINE_LATENT_OUTPUT_GATE_MAX",
            default_min=0.10,
            default_max=0.995,
        ).to(device=state_hidden.device, dtype=state_hidden.dtype)
        return gate * self._apply_output_head(self.causal_machine_latent_out, latent_hidden)

    def _structured_causal_machine_transition_params(self) -> tuple[Tensor, Tensor, Tensor]:
        if (
            self.causal_machine_transition_source_logits is None
            or self.causal_machine_transition_dest_logits is None
            or self.causal_machine_transition_stay_logits is None
        ):
            raise RuntimeError("structured causal-machine transition parameters are unavailable")
        transition_source_probs = F.softmax(self.causal_machine_transition_source_logits.float(), dim=-1)
        transition_dest_probs = F.softmax(self.causal_machine_transition_dest_logits.float(), dim=-1)
        transition_stay_probs = torch.sigmoid(self.causal_machine_transition_stay_logits.float())
        return transition_source_probs, transition_dest_probs, transition_stay_probs

    def _get_global_packed_transition_tables(self, device: torch.device) -> tuple[int, Tensor, Tensor, Tensor, Tensor] | None:
        if (
            self.causal_machine_transition_source_logits is None
            or self.causal_machine_transition_dest_logits is None
        ):
            return None
        return get_or_update_scan_transition_prepack(
            self._global_packed_transition_cache,
            self.causal_machine_transition_source_logits,
            self.causal_machine_transition_dest_logits,
            device,
        )

    def _get_global_sparse_transition_tables(
        self,
        device: torch.device,
        runtime_config: StructuredScanRuntimeConfig | None,
        packed_transition_tables: tuple[int, Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> StructuredSparseTransitionTables | None:
        if _structured_filter_mode() == "composable":
            return None
        if (
            self.causal_machine_transition_source_logits is None
            or self.causal_machine_transition_dest_logits is None
        ):
            return None
        return get_or_update_scan_transition_sparse_blocks(
            self._global_sparse_transition_cache,
            self.causal_machine_transition_source_logits,
            self.causal_machine_transition_dest_logits,
            device,
            runtime_config,
            packed_transition_tables,
        )

    def _get_global_reduced_transition_cache(
        self,
        device: torch.device,
        runtime_config: StructuredScanRuntimeConfig | None,
        *,
        tile_size: int,
        split_size: int,
    ) -> StructuredReducedTransitionCache | None:
        if (
            self.causal_machine_transition_source_logits is None
            or self.causal_machine_transition_dest_logits is None
        ):
            return None
        return get_or_update_structured_reduced_transition_cache(
            self._global_reduced_transition_cache,
            self.causal_machine_transition_source_logits,
            self.causal_machine_transition_dest_logits,
            device,
            runtime_config,
            num_states=self.causal_machine_num_states,
            tile_size=int(tile_size),
            split_size=int(split_size),
        )

    def _filter_causal_machine_beliefs(
        self,
        state_hidden: Tensor,
        cache: CausalMachineCache | None = None,
        update_cache: bool = False,
        runtime_config: StructuredScanRuntimeConfig | None = None,
    ) -> tuple[Tensor, Tensor]:
        self.last_causal_machine_kernel_info = {}
        if (
            self.causal_machine_state_head is None
            or self.causal_machine_transition_source_logits is None
            or self.causal_machine_transition_dest_logits is None
            or self.causal_machine_transition_stay_logits is None
            or self.causal_machine_transition_gate is None
        ):
            state_logits = self._apply_output_head(self.causal_machine_state_head, state_hidden) if self.causal_machine_state_head is not None else state_hidden
            state_log_beliefs = F.log_softmax(state_logits.float(), dim=-1)
            self.last_causal_machine_kernel_info = {
                "path": "softmax_only",
                "backend": "python",
                "uses_paged_cache": bool(runtime_config.use_paged_cache) if runtime_config is not None else False,
            }
            return state_logits, state_log_beliefs.to(dtype=state_hidden.dtype)
        batch_size, seq_len, _ = state_hidden.shape
        local_logits, transition_context = self._compute_causal_machine_local_logits_and_transition_context(state_hidden)
        latent_mode = self._resolved_global_latent_mode(seq_len, state_hidden.device)
        latent_logits = None
        if latent_mode in {"additive", "replace"}:
            if seq_len == 1:
                latent_logits = self._compute_global_latent_logits_step(state_hidden, cache=cache, update_cache=update_cache)
            else:
                latent_logits = self._compute_global_latent_logits_sequence(state_hidden)
            if latent_logits is not None:
                local_logits = local_logits + latent_logits.to(dtype=local_logits.dtype)
        if latent_mode == "replace" and not self.causal_machine_replace_uses_structured:
            state_log_beliefs = F.log_softmax(local_logits.float(), dim=-1)
            self.last_causal_machine_kernel_info = {
                "path": "latent_replace",
                "backend": "cuda" if bool(USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN and state_hidden.is_cuda and self.causal_machine_latent_rank > 0) else "python",
                "uses_paged_cache": bool(runtime_config.use_paged_cache) if runtime_config is not None else False,
            }
            if update_cache and cache is not None:
                if runtime_config is not None and runtime_config.use_paged_cache and cache.paged_log_beliefs is None:
                    cache.enable_paged_history(
                        batch_size=batch_size,
                        num_states=self.causal_machine_num_states,
                        device=state_hidden.device,
                        dtype=state_hidden.dtype,
                        latent_rank=self.causal_machine_latent_rank,
                        page_size=runtime_config.page_size,
                        max_pages=runtime_config.max_pages,
                        workspace=runtime_config.workspace,
                        grouped_launch_pack=runtime_config.grouped_launch_pack,
                        save_mode=runtime_config.save_mode,
                        paged_layout=runtime_config.paged_layout,
                    )
                cache.record_sequence(state_log_beliefs.to(dtype=state_hidden.dtype))
                if runtime_config is not None and runtime_config.use_paged_cache and runtime_config.paged_resident_only:
                    cache.drop_resident_state()
                self.last_causal_machine_kernel_info = dict(self.last_causal_machine_kernel_info)
                self.last_causal_machine_kernel_info["paged_cache_write_backend"] = str(cache.last_paged_write_backend)
                if cache.last_paged_write_error is not None:
                    self.last_causal_machine_kernel_info["paged_cache_write_error"] = str(cache.last_paged_write_error)
            return local_logits, state_log_beliefs.to(dtype=state_hidden.dtype)
        transition_stay_probs = torch.sigmoid(self.causal_machine_transition_stay_logits.float())
        transition_gate = _bounded_gate(
            self.causal_machine_transition_gate,
            "CAUSAL_MACHINE_TRANSITION_GATE_MIN",
            "CAUSAL_MACHINE_TRANSITION_GATE_MAX",
            default_min=0.01,
            default_max=0.995,
        )
        transition_context, transition_gate = _prepare_structured_filter_inputs(
            transition_context,
            transition_gate,
        )
        transition_context, transition_gate = _apply_native_structured_score_mod_inputs(
            transition_context,
            transition_gate,
            runtime_config,
        )
        packed_transition_tables = self._get_global_packed_transition_tables(state_hidden.device)
        if cache is not None and cache.log_belief is None and cache.paged_log_beliefs is not None:
            cache.ensure_resident_state()
        if cache is not None and cache.log_belief is not None:
            initial_log_belief = cache.log_belief.to(device=state_hidden.device, dtype=torch.float32)
        else:
            initial_log_belief = self.init_causal_machine_cache(
                batch_size=batch_size,
                device=state_hidden.device,
                dtype=torch.float32,
            ).log_belief
        if initial_log_belief is None:
            initial_log_belief = torch.full(
                (batch_size, self.causal_machine_num_states),
                fill_value=-math.log(max(self.causal_machine_num_states, 1)),
                device=state_hidden.device,
                dtype=torch.float32,
            )
        tiled_needs_grad = (
            torch.is_grad_enabled()
            and (
                local_logits.requires_grad
                or self.causal_machine_transition_source_logits.requires_grad
                or self.causal_machine_transition_dest_logits.requires_grad
                or transition_context.requires_grad
                or initial_log_belief.requires_grad
                or transition_gate.requires_grad
                or transition_stay_probs.requires_grad
            )
        )
        kernel_config = _get_precomputed_structured_scan_kernel_config(
            self,
            cache_attr="_compiled_global_kernel_config_cache",
            num_states=self.causal_machine_num_states,
            transition_rank=self.causal_machine_transition_rank,
            seq_len=int(seq_len),
            device=state_hidden.device,
            default_chunk_size=self.causal_machine_filter_chunk_size,
            needs_grad=tiled_needs_grad,
            runtime_config=runtime_config,
        )
        if runtime_config is not None:
            runtime_config.reduced_transition_cache = self._get_global_reduced_transition_cache(
                state_hidden.device,
                runtime_config,
                tile_size=int(kernel_config.tile_size),
                split_size=int(kernel_config.split_size),
            )
        sparse_transition_tables = (
            self._get_global_sparse_transition_tables(
                state_hidden.device,
                runtime_config,
                packed_transition_tables,
            )
            if (
                self.use_causal_machine_cuda_scan
                and self.supports_incremental_backbone_cache()
                and _structured_runtime_supports_sparse_cuda(runtime_config)
            )
            else None
        )
        sparse_needs_grad = (
            torch.is_grad_enabled()
            and (
                local_logits.requires_grad
                or self.causal_machine_transition_source_logits.requires_grad
                or self.causal_machine_transition_dest_logits.requires_grad
                or transition_context.requires_grad
                or initial_log_belief.requires_grad
                or transition_gate.requires_grad
                or transition_stay_probs.requires_grad
            )
        )
        sparse_runtime_result = _execute_structured_sparse_runtime_cuda(
            local_logits,
            self.causal_machine_transition_source_logits,
            self.causal_machine_transition_dest_logits,
            transition_context,
            initial_log_belief.to(dtype=local_logits.dtype),
            transition_gate.reshape(()),
            transition_stay_probs,
            sparse_transition_tables,
            runtime_config=runtime_config,
            chunk_size=int(kernel_config.chunk_size),
            needs_grad=sparse_needs_grad,
        )
        masked_needs_grad = (
            torch.is_grad_enabled()
            and (
                local_logits.requires_grad
                or self.causal_machine_transition_source_logits.requires_grad
                or self.causal_machine_transition_dest_logits.requires_grad
                or transition_context.requires_grad
                or initial_log_belief.requires_grad
                or transition_gate.requires_grad
                or transition_stay_probs.requires_grad
            )
        )
        use_masked_cuda = (
            self.use_causal_machine_cuda_scan
            and self.supports_incremental_backbone_cache()
            and _structured_runtime_supports_masked_cuda(runtime_config)
            and _can_use_causal_machine_masked_scan_cuda(
                state_hidden.device,
                num_states=self.causal_machine_num_states,
                transition_rank=self.causal_machine_transition_rank,
                needs_grad=masked_needs_grad,
                runtime_config=runtime_config,
            )
        )
        if sparse_runtime_result is not None:
            state_log_beliefs, final_log_belief = sparse_runtime_result
            self.last_causal_machine_kernel_info = _structured_scan_kernel_info(
                path="cuda_sparse",
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
                cache=cache,
            )
            _enforce_structured_scan_cuda_contract(
                self.last_causal_machine_kernel_info,
                context="global causal machine structured scan",
            )
        elif use_masked_cuda:
            state_log_beliefs, final_log_belief = causal_machine_scan_masked_cuda(
                local_logits,
                self.causal_machine_transition_source_logits,
                self.causal_machine_transition_dest_logits,
                transition_context,
                initial_log_belief.to(dtype=local_logits.dtype),
                transition_gate.reshape(()),
                transition_stay_probs,
                runtime_config=runtime_config,
                chunk_size=int(kernel_config.chunk_size),
            )
            self.last_causal_machine_kernel_info = _structured_scan_kernel_info(
                path="cuda_masked",
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
                cache=cache,
            )
            _enforce_structured_scan_cuda_contract(
                self.last_causal_machine_kernel_info,
                context="global causal machine structured scan",
            )
        elif (
            self.use_causal_machine_cuda_scan
            and self.supports_incremental_backbone_cache()
            and kernel_config.backend == "cuda"
        ):
            state_log_beliefs, final_log_belief = causal_machine_scan_cuda(
                local_logits,
                self.causal_machine_transition_source_logits,
                self.causal_machine_transition_dest_logits,
                transition_context,
                initial_log_belief.to(dtype=local_logits.dtype),
                transition_gate.reshape(()),
                transition_stay_probs,
                packed_transition_tables=packed_transition_tables,
                chunk_size=int(kernel_config.chunk_size),
                runtime_config=runtime_config,
            )
            dense_path = (
                "cuda_dense_lowp_tensor_core"
                if _structured_scan_uses_lowp_tensor_core_path(
                    device=local_logits.device,
                    dtype=local_logits.dtype,
                    kernel_config=kernel_config,
                    packed_transition_tables=packed_transition_tables,
                )
                else "cuda_dense"
            )
            self.last_causal_machine_kernel_info = _structured_scan_kernel_info(
                path=dense_path,
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
                cache=cache,
            )
            _enforce_structured_scan_cuda_contract(
                self.last_causal_machine_kernel_info,
                context="global causal machine structured scan",
            )
        else:
            transition_source_probs, transition_dest_probs, _ = self._structured_causal_machine_transition_params()
            state_log_beliefs, final_log_belief, _ = structured_scan_fallback(
                local_logits,
                transition_context,
                initial_log_belief,
                transition_source_probs,
                transition_dest_probs,
                transition_stay_probs,
                transition_gate,
                chunk_size=int(kernel_config.chunk_size),
                tile_size=int(kernel_config.tile_size),
                split_size=int(kernel_config.split_size),
                backend=str(kernel_config.backend),
                track_transition_kl=False,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
            )
            self.last_causal_machine_kernel_info = _structured_scan_kernel_info(
                path=f"fallback_{kernel_config.backend}",
                kernel_config=kernel_config,
                runtime_config=runtime_config,
                packed_transition_tables=packed_transition_tables,
                cache=cache,
            )
            _enforce_structured_scan_cuda_contract(
                self.last_causal_machine_kernel_info,
                context="global causal machine structured scan",
            )
        if update_cache and cache is not None:
            if runtime_config is not None and runtime_config.use_paged_cache and cache.paged_log_beliefs is None:
                cache.enable_paged_history(
                    batch_size=batch_size,
                    num_states=self.causal_machine_num_states,
                    device=state_hidden.device,
                    dtype=state_hidden.dtype,
                    latent_rank=self.causal_machine_latent_rank,
                    page_size=runtime_config.page_size,
                    max_pages=runtime_config.max_pages,
                    workspace=runtime_config.workspace,
                    grouped_launch_pack=runtime_config.grouped_launch_pack,
                    save_mode=runtime_config.save_mode,
                    paged_layout=runtime_config.paged_layout,
                )
            cache.record_sequence(state_log_beliefs)
            cache.log_belief = final_log_belief.detach().to(dtype=state_hidden.dtype)
            if runtime_config is not None and runtime_config.use_paged_cache and runtime_config.paged_resident_only:
                cache.drop_resident_state()
            self.last_causal_machine_kernel_info = dict(self.last_causal_machine_kernel_info)
            self.last_causal_machine_kernel_info["paged_cache_write_backend"] = str(cache.last_paged_write_backend)
            if cache.last_paged_write_error is not None:
                self.last_causal_machine_kernel_info["paged_cache_write_error"] = str(cache.last_paged_write_error)
        return local_logits, state_log_beliefs.to(dtype=state_hidden.dtype)

    def _compute_causal_machine_outputs(
        self,
        x: Tensor,
        cache: CausalMachineCache | None = None,
        update_cache: bool = False,
        runtime_config: StructuredScanRuntimeConfig | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if (
            self.causal_machine_prefix_down is None
            or self.causal_machine_state_head is None
            or self.causal_machine_scale is None
            or self.causal_machine_gate is None
            or self.causal_machine_log_probs.numel() == 0
        ):
            return None, None, None, None
        state_features = self._compute_causal_machine_state_features(x)
        state_hidden = self._apply_output_head(self.causal_machine_prefix_down, state_features)
        state_logits, state_log_beliefs = self._filter_causal_machine_beliefs(
            state_hidden,
            cache=cache,
            update_cache=update_cache,
            runtime_config=runtime_config,
        )
        local_state_log_probs = F.log_softmax(state_logits.float(), dim=-1).to(dtype=x.dtype)
        emit_logits = self._compute_causal_machine_emit_log_probs(device=x.device, dtype=torch.float32)
        state_probs = state_log_beliefs.float().exp()
        emit_probs = emit_logits.exp()
        mixed_probs = torch.matmul(
            state_probs.reshape(-1, state_probs.size(-1)),
            emit_probs,
        ).reshape(state_probs.size(0), state_probs.size(1), emit_probs.size(-1))
        raw_machine_logits = mixed_probs.clamp_min(1e-30).log().to(dtype=x.dtype)
        scale = F.softplus(self.causal_machine_scale).to(device=x.device, dtype=x.dtype)
        gate = torch.sigmoid(self.causal_machine_gate).to(device=x.device, dtype=x.dtype)
        return scale * gate * raw_machine_logits, state_log_beliefs, raw_machine_logits, local_state_log_probs

    def apply_causal_machine_profile(self, profile: dict[str, object], vocab_size: int) -> None:
        num_states = int(profile["num_states"])
        if num_states != self.causal_machine_num_states:
            raise ValueError(
                f"CAUSAL_MACHINE_PROFILE_JSON num_states={num_states} does not match CAUSAL_MACHINE_NUM_STATES={self.causal_machine_num_states}"
            )
        device = self.tok_emb.weight.device
        sketch_dim = int(profile["sketch_dim"])
        horizons = [int(v) for v in profile["horizons"]]
        total_sketch_dim = sketch_dim * len(horizons)
        log_probs = torch.from_numpy(np.asarray(profile["log_probs"], dtype=np.float32)).to(device=device)
        state_masses = torch.from_numpy(np.asarray(profile["state_masses"], dtype=np.float32)).to(device=device)
        centroids = torch.from_numpy(np.asarray(profile["centroids_sketch"], dtype=np.float32)).to(device=device)
        bucket_ids_np, signs_np = _sketch_token_tables(vocab_size, sketch_dim)
        bucket_ids_t = torch.from_numpy(bucket_ids_np.astype(np.int64, copy=False)).to(device=device)
        signs_t = torch.from_numpy(signs_np.astype(np.float32, copy=False)).to(device=device)
        self.causal_machine_log_probs = log_probs
        mass_sum = state_masses.sum()
        default_prior = torch.full(
            (num_states,),
            fill_value=-math.log(max(num_states, 1)),
            dtype=torch.float32,
            device=device,
        )
        priors = torch.where(
            mass_sum > 0.0,
            torch.log(state_masses / mass_sum.clamp_min(1.0e-12)).clamp_min(-30.0),
            default_prior,
        )
        self.causal_machine_log_state_priors = priors
        self.causal_machine_signature_centroids = centroids
        self.causal_machine_horizon_tensor = torch.tensor(horizons, dtype=torch.int64, device=device)
        self.causal_machine_bucket_ids = bucket_ids_t
        self.causal_machine_signs = signs_t
        self.causal_machine_sketch_dim = sketch_dim
        self.causal_machine_horizons = horizons
        self.causal_machine_profile_loaded = True
        self.causal_machine_online_teacher_ready.fill_(True)
        self.causal_machine_online_state_counts = state_masses.to(device=device, dtype=torch.float32).clone()
        for block in self.blocks:
            if isinstance(block, StateSpaceBlock):
                block.attn.configure_future_sketch(total_sketch_dim)

    def _compute_causal_machine_future_sketch(self, target_ids: Tensor) -> Tensor | None:
        if (
            self.causal_machine_bucket_ids.numel() == 0
            or self.causal_machine_signs.numel() == 0
            or self.causal_machine_horizon_tensor.numel() == 0
            or self.causal_machine_sketch_dim <= 0
        ):
            return None
        batch_size, seq_len = target_ids.shape
        flat_size = batch_size * seq_len
        bucket_ids = self.causal_machine_bucket_ids.to(device=target_ids.device)
        signs = self.causal_machine_signs.to(device=target_ids.device)
        blocks: list[Tensor] = []
        for horizon in self.causal_machine_horizons:
            accum = torch.zeros((flat_size, self.causal_machine_sketch_dim), device=target_ids.device, dtype=torch.float32)
            denom = torch.zeros((flat_size, 1), device=target_ids.device, dtype=torch.float32)
            # Future-sketch targets should only summarize tokens strictly after t.
            for offset in range(1, int(horizon) + 1):
                shifted = torch.zeros_like(target_ids)
                valid = torch.zeros_like(target_ids, dtype=torch.float32)
                if offset < seq_len:
                    shifted[:, : seq_len - offset] = target_ids[:, offset:]
                    valid[:, : seq_len - offset] = 1.0
                flat_bucket = bucket_ids[shifted].reshape(-1, 1)
                flat_weight = (signs[shifted] * valid).reshape(-1, 1)
                accum.scatter_add_(1, flat_bucket, flat_weight)
                denom = denom + valid.reshape(-1, 1)
            accum = accum / denom.clamp_min(1.0)
            blocks.append(accum)
        return torch.cat(blocks, dim=1).reshape(batch_size, seq_len, -1)

    @torch.no_grad()
    def _update_online_causal_teacher(
        self,
        target_ids: Tensor,
        state_log_beliefs: Tensor | None,
        features_3d: Tensor | None = None,
    ) -> None:
        return

    def _compute_causal_machine_teacher_state_ids(
        self,
        target_ids: Tensor,
        features_3d: Tensor | None = None,
    ) -> Tensor | None:
        if not self.causal_machine_profile_loaded:
            return None
        if (
            self.causal_machine_signature_centroids.numel() == 0
            or self.causal_machine_log_probs.numel() == 0
            or self.causal_machine_bucket_ids.numel() == 0
            or self.causal_machine_signs.numel() == 0
            or self.causal_machine_horizon_tensor.numel() == 0
            or self.causal_machine_sketch_dim <= 0
        ):
            return None
        if features_3d is None:
            features_3d = self._compute_causal_machine_future_sketch(target_ids)
        if features_3d is None:
            return None
        features = features_3d.reshape(-1, features_3d.size(-1))
        centroids = self.causal_machine_signature_centroids.to(device=target_ids.device, dtype=features.dtype)
        feat_norm = (features * features).sum(dim=1, keepdim=True)
        centroid_norm = (centroids * centroids).sum(dim=1).unsqueeze(0)
        dists = feat_norm - 2.0 * (features @ centroids.transpose(0, 1)) + centroid_norm
        return torch.argmin(dists, dim=1).reshape(target_ids.size(0), target_ids.size(1))

    def _compute_state_space_backbone_loss(
        self,
        target_ids: Tensor,
        loss_mask: Tensor | None,
        teacher_state_ids: Tensor | None = None,
        teacher_future_sketch: Tensor | None = None,
    ) -> Tensor | None:
        state_blocks = [block for block in self.blocks if isinstance(block, StateSpaceBlock) and block.last_aux]
        if not state_blocks:
            return None
        mask = None if loss_mask is None else loss_mask.to(dtype=torch.float32)
        accum_terms: list[Tensor] = []
        for block in state_blocks:
            aux = block.last_aux
            if self.causal_machine_next_token_loss_coeff > 0.0:
                block_hidden = aux.get("block_hidden")
                if block_hidden is not None:
                    block_features = (
                        self.final_norm(block_hidden, condition=None)
                        if isinstance(self.final_norm, AdaptiveRMSNorm)
                        else self.final_norm(block_hidden)
                    )
                    block_logits = self._project_logits(block_features)
                    block_ce = F.cross_entropy(
                        block_logits.reshape(-1, block_logits.size(-1)).float(),
                        target_ids.reshape(-1),
                        reduction="none",
                    ).view_as(target_ids)
                    if mask is None:
                        accum_terms.append(self.causal_machine_next_token_loss_coeff * block_ce.mean())
                    else:
                        accum_terms.append(
                            self.causal_machine_next_token_loss_coeff
                            * (block_ce * mask).sum()
                            / mask.sum().clamp_min(1.0)
                        )
            if teacher_state_ids is not None and self.causal_machine_state_loss_coeff > 0.0:
                local_state_log_probs = aux.get("local_state_log_probs")
                if local_state_log_probs is not None:
                    state_ce = F.nll_loss(
                        local_state_log_probs.reshape(-1, local_state_log_probs.size(-1)).float(),
                        teacher_state_ids.reshape(-1),
                        reduction="none",
                    ).view_as(target_ids)
                    if mask is None:
                        accum_terms.append(self.causal_machine_state_loss_coeff * state_ce.mean())
                    else:
                        accum_terms.append(
                            self.causal_machine_state_loss_coeff
                            * (state_ce * mask).sum()
                            / mask.sum().clamp_min(1.0)
                        )
            if self.causal_machine_transition_kl_coeff > 0.0:
                state_log_beliefs = aux.get("state_log_beliefs")
                prior_log_beliefs = aux.get("prior_state_log_beliefs")
                if state_log_beliefs is not None and prior_log_beliefs is not None:
                    target_probs = state_log_beliefs.detach().float().exp()
                    transition_kl = (target_probs * (state_log_beliefs.detach().float() - prior_log_beliefs.float())).sum(dim=-1)
                    if mask is None:
                        accum_terms.append(self.causal_machine_transition_kl_coeff * transition_kl.mean())
                    else:
                        accum_terms.append(
                            self.causal_machine_transition_kl_coeff
                            * (transition_kl * mask).sum()
                            / mask.sum().clamp_min(1.0)
                        )
            if teacher_future_sketch is not None and self.causal_machine_future_sketch_loss_coeff > 0.0:
                pred_sketch = aux.get("future_sketch_pred")
                if pred_sketch is not None:
                    sketch_loss = F.smooth_l1_loss(pred_sketch.float(), teacher_future_sketch.float(), reduction="none").mean(dim=-1)
                    if mask is None:
                        accum_terms.append(self.causal_machine_future_sketch_loss_coeff * sketch_loss.mean())
                    else:
                        accum_terms.append(
                            self.causal_machine_future_sketch_loss_coeff
                            * (sketch_loss * mask).sum()
                            / mask.sum().clamp_min(1.0)
                        )
        if not accum_terms:
            return None
        return sum(accum_terms) / float(max(len(state_blocks), 1))

    def _select_online_teacher_beliefs(self, fallback: Tensor | None) -> Tensor | None:
        if fallback is not None:
            return fallback
        for block in reversed(self.blocks):
            if not isinstance(block, StateSpaceBlock) or not block.last_aux:
                continue
            state_log_beliefs = block.last_aux.get("state_log_beliefs")
            if state_log_beliefs is not None:
                return state_log_beliefs
        return None

    def _staged_mult(self, step: int, enable_step: int, ramp_steps: int) -> float:
        step = max(int(step), 0)
        enable_step = max(int(enable_step), 0)
        ramp_steps = max(int(ramp_steps), 0)
        if step < enable_step:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return min(max((step - enable_step) / max(ramp_steps, 1), 0.0), 1.0)

    def set_training_step(self, step: int) -> None:
        self.current_training_step = max(int(step), 0)
        step = self.current_training_step
        shared_tail_mult = self._staged_mult(
            step,
            self.shared_tail_enable_step,
            self.shared_tail_ramp_steps,
        ) * self.shared_tail_max_mult
        self.shared_tail_schedule_mult_buffer.fill_(float(shared_tail_mult))

    def _loss_from_logits(
        self,
        logits: Tensor,
        target_ids: Tensor,
        loss_mask: Tensor | None = None,
        label_smoothing: float = 0.0,
        z_loss_coeff: float = 0.0,
        logit_var_loss_coeff: float = 0.0,
    ) -> Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        targets = target_ids.reshape(-1)
        mask = None if loss_mask is None else loss_mask.reshape(-1).to(dtype=torch.float32)
        ce = F.cross_entropy(flat_logits.float(), targets, reduction="none", label_smoothing=label_smoothing)
        if mask is None:
            loss = ce.mean()
        else:
            loss = (ce * mask).sum() / mask.sum().clamp_min(1.0)
        if z_loss_coeff > 0.0:
            z_term = torch.logsumexp(flat_logits.float(), dim=-1).square()
            z_loss = z_term.mean() if mask is None else (z_term * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + z_loss_coeff * z_loss
        if logit_var_loss_coeff > 0.0:
            # Keep per-token logit spread from exploding in short runs, which tends
            # to help the later quantized softcapped head behave more smoothly.
            logit_std = flat_logits.float().std(dim=-1)
            target_std = float(self.logit_softcap) / 4.0
            logit_var = (logit_std - target_std).square()
            if mask is None:
                loss = loss + logit_var_loss_coeff * logit_var.mean()
            else:
                loss = loss + logit_var_loss_coeff * (logit_var * mask).sum() / mask.sum().clamp_min(1.0)
        return loss

    def _apply_shared_tail(self, x: Tensor, x0: Tensor, norm_condition: Tensor | None = None) -> Tensor:
        if self.num_shared_layers <= 0 or self.shared_layer_repeats <= 0:
            return x
        tail_base = x
        tail_mult = self._shared_tail_schedule_mult(device=x.device, dtype=x.dtype)
        if self.shared_tail_gate is not None:
            tail_mult = tail_mult * torch.sigmoid(self.shared_tail_gate.to(dtype=x.dtype))
        for _ in range(self.shared_layer_repeats):
            for block in self.shared_blocks:
                prev_x = x
                x = block.forward_simple(x, x0, norm_condition=norm_condition)
                x = prev_x + tail_mult * (x - prev_x)
        return tail_base + tail_mult * (x - tail_base)

    def _embed_token_ids_unique(self, token_ids: Tensor) -> Tensor:
        embed_weight = self._embedding_weight()
        return F.embedding(token_ids, embed_weight)

    def _build_norm_condition(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        start_position: int = 0,
    ) -> Tensor | None:
        if not self.use_adaptive_rmsnorm:
            return None
        denom = max(self.train_seq_len - 1, 1)
        positions = torch.arange(
            start_position,
            start_position + seq_len,
            device=device,
            dtype=dtype,
        )
        conf = (positions / float(denom)).clamp(0.0, 1.0)
        conf = 2.0 * conf - 1.0
        conf = conf.view(1, seq_len, 1).expand(batch_size, -1, -1)
        mask = torch.ones((batch_size, seq_len, 1), device=device, dtype=dtype)
        return torch.cat((conf, mask), dim=-1)

    def _forward_features(
        self,
        input_ids: Tensor,
        return_pre_final: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]:
        if self.fast_features_path:
            return self._forward_features_fast(input_ids, return_pre_final=return_pre_final)
        x = self._embed_token_ids_unique(input_ids)
        norm_condition = self._build_norm_condition(
            seq_len=int(input_ids.size(1)),
            batch_size=int(input_ids.size(0)),
            device=x.device,
            dtype=x.dtype,
            start_position=0,
        )
        x = F.rms_norm(x, (x.size(-1),))
        # Keep the block-level residual reference non-aliased so torch.compile
        # does not recompile `_run_block` when the first block sees x is x0.
        x0 = x.clone()
        skips: list[Tensor] = []
        h_mid: Tensor | None = None
        mid_idx = len(self.blocks) // 2

        for i in range(self.num_encoder_layers):
            x = self.blocks[i].forward_simple(x, x0, norm_condition=norm_condition)
            if i == mid_idx:
                h_mid = x
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self._skip_scale(i, x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            x = self.blocks[bi].forward_simple(x, x0, norm_condition=norm_condition)
            if bi == mid_idx:
                h_mid = x
        x = self._apply_shared_tail(x, x0, norm_condition=norm_condition)
        pre_final = x
        x = self.final_norm(x, condition=norm_condition) if isinstance(self.final_norm, AdaptiveRMSNorm) else self.final_norm(x)
        if return_pre_final:
            return x, h_mid, pre_final
        return x, h_mid

    def supports_incremental_backbone_cache(self) -> bool:
        return (
            self.num_shared_layers <= 0
            and self.shared_layer_repeats <= 0
            and all(block.fast_path for block in self.blocks)
        )

    def init_backbone_step_cache(self, max_len: int | None = None) -> BackboneStepCache:
        return BackboneStepCache(
            layers=[block.init_step_cache(max_len=max_len) for block in self.blocks],
            position=0,
        )

    def _forward_features_step_strict(
        self,
        input_ids: Tensor,
        backbone_cache: BackboneStepCache,
    ) -> tuple[Tensor, Tensor | None]:
        if not self.supports_incremental_backbone_cache():
            raise RuntimeError("incremental backbone cache is only supported for the strict helper-free state-machine path")
        if input_ids.ndim != 2 or input_ids.size(1) != 1:
            raise ValueError(f"_forward_features_step_strict expects [batch, 1] input ids, got {tuple(input_ids.shape)}")
        x = self.tok_emb(input_ids)
        norm_condition = self._build_norm_condition(
            seq_len=1,
            batch_size=int(input_ids.size(0)),
            device=x.device,
            dtype=x.dtype,
            start_position=backbone_cache.position,
        )
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x.clone()
        h_mid: Tensor | None = None
        mid_idx = len(self.blocks) // 2
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i].forward_simple_step(
                x,
                x0,
                cache=backbone_cache.layers[i],
                position=backbone_cache.position,
                norm_condition=norm_condition,
            )
            if i == mid_idx:
                h_mid = x
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self._skip_scale(i, x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[bi].forward_simple_step(
                x,
                x0,
                cache=backbone_cache.layers[bi],
                position=backbone_cache.position,
                norm_condition=norm_condition,
            )
            if bi == mid_idx:
                h_mid = x
        x = self._apply_shared_tail(x, x0, norm_condition=norm_condition)
        x = self.final_norm(x, condition=norm_condition) if isinstance(self.final_norm, AdaptiveRMSNorm) else self.final_norm(x)
        backbone_cache.position += 1
        return x, h_mid

    def _forward_features_fast(
        self,
        input_ids: Tensor,
        return_pre_final: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]:
        x = self._embed_token_ids_unique(input_ids)
        norm_condition = self._build_norm_condition(
            seq_len=int(input_ids.size(1)),
            batch_size=int(input_ids.size(0)),
            device=x.device,
            dtype=x.dtype,
            start_position=0,
        )
        x = F.rms_norm(x, (x.size(-1),))
        # Keep the block-level residual reference non-aliased so torch.compile
        # does not recompile `forward_simple` when the first block sees x is x0.
        x0 = x.clone()
        skips: list[Tensor] = []
        h_mid: Tensor | None = None
        mid_idx = len(self.blocks) // 2
        for i in range(self.num_encoder_layers):
            x = self.blocks[i].forward_simple(x, x0, norm_condition=norm_condition)
            if i == mid_idx:
                h_mid = x
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self._skip_scale(i, x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            x = self.blocks[bi].forward_simple(x, x0, norm_condition=norm_condition)
            if bi == mid_idx:
                h_mid = x
        x = self._apply_shared_tail(x, x0, norm_condition=norm_condition)
        pre_final = x
        x = self.final_norm(x, condition=norm_condition) if isinstance(self.final_norm, AdaptiveRMSNorm) else self.final_norm(x)
        if return_pre_final:
            return x, h_mid, pre_final
        return x, h_mid

    def _project_logits(
        self,
        x: Tensor,
        causal_machine_logits: Tensor | None = None,
    ) -> Tensor:
        if self.tie_embeddings:
            embed_weight = self._embedding_weight()
            logits_proj = F.linear(x.to(dtype=embed_weight.dtype), embed_weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self._apply_output_head(self.lm_head, x)
        if self.output_logit_bias is not None:
            logits_proj = logits_proj + self.output_logit_bias.to(device=logits_proj.device, dtype=logits_proj.dtype)
        if causal_machine_logits is not None:
            logits_proj = logits_proj + causal_machine_logits.to(dtype=logits_proj.dtype)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def _needs_global_causal_machine_teacher(self) -> bool:
        return self.causal_machine_profile_loaded and (
            self.causal_machine_teacher_loss_coeff > 0.0 or self.causal_machine_state_loss_coeff > 0.0
        )

    def _needs_global_causal_machine_outputs(self) -> bool:
        return (
            self.use_causal_machine_output_bias
            or self.causal_machine_next_token_loss_coeff > 0.0
            or self._needs_global_causal_machine_teacher()
        )

    def _needs_teacher_future_sketch(self) -> bool:
        return self.causal_machine_future_sketch_loss_coeff > 0.0 or self._needs_global_causal_machine_teacher()

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        loss_mask: Tensor | None = None,
        label_smoothing: float = 0.0,
        z_loss_coeff: float = 0.0,
        logit_var_loss_coeff: float = 0.0,
        mid_aux_loss_coeff: float | Tensor = 0.01,
    ) -> Tensor:
        features, h_mid = self._forward_features(input_ids)
        if self._needs_global_causal_machine_outputs():
            causal_machine_logits, causal_machine_state_log_beliefs, causal_machine_raw_logits, causal_machine_local_state_log_probs = self._compute_causal_machine_outputs(
                features
            )
        else:
            causal_machine_logits = None
            causal_machine_state_log_beliefs = None
            causal_machine_raw_logits = None
            causal_machine_local_state_log_probs = None
        logits = self._project_logits(
            features,
            causal_machine_logits=causal_machine_logits,
        )
        loss = self._loss_from_logits(
            logits,
            target_ids,
            loss_mask=loss_mask,
            label_smoothing=label_smoothing,
            z_loss_coeff=z_loss_coeff,
            logit_var_loss_coeff=logit_var_loss_coeff,
        )
        teacher_future_sketch = (
            self._compute_causal_machine_future_sketch(target_ids) if self._needs_teacher_future_sketch() else None
        )
        teacher_state_ids = (
            self._compute_causal_machine_teacher_state_ids(
                target_ids,
                features_3d=teacher_future_sketch,
            )
            if self._needs_global_causal_machine_teacher()
            else None
        )
        if causal_machine_raw_logits is not None:
            if self.causal_machine_next_token_loss_coeff > 0.0:
                state_ce = F.cross_entropy(
                    causal_machine_raw_logits.reshape(-1, causal_machine_raw_logits.size(-1)).float(),
                    target_ids.reshape(-1),
                    reduction="none",
                    label_smoothing=label_smoothing,
                ).view_as(target_ids)
                if loss_mask is None:
                    state_next_token_loss = state_ce.mean()
                else:
                    state_next_token_loss = (state_ce * loss_mask.to(dtype=state_ce.dtype)).sum() / loss_mask.sum().clamp_min(1.0)
                loss = loss + self.causal_machine_next_token_loss_coeff * state_next_token_loss.to(dtype=loss.dtype)
            if teacher_state_ids is not None:
                mask = None if loss_mask is None else loss_mask.to(dtype=torch.float32)
                teacher_log_probs = self.causal_machine_log_probs[teacher_state_ids].to(device=logits.device, dtype=torch.float32)
                student_log_probs = causal_machine_raw_logits.float()
                teacher_probs = teacher_log_probs.exp()
                teacher_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
                if mask is None:
                    teacher_loss = teacher_kl.mean()
                else:
                    teacher_loss = (teacher_kl * mask).sum() / mask.sum().clamp_min(1.0)
                loss = loss + self.causal_machine_teacher_loss_coeff * teacher_loss.to(dtype=loss.dtype)
                if causal_machine_local_state_log_probs is not None:
                    state_ce = F.nll_loss(
                        causal_machine_local_state_log_probs.reshape(-1, causal_machine_local_state_log_probs.size(-1)).float(),
                        teacher_state_ids.reshape(-1),
                        reduction="none",
                    ).view_as(target_ids)
                    if mask is None:
                        state_loss = state_ce.mean()
                    else:
                        state_loss = (state_ce * mask).sum() / mask.sum().clamp_min(1.0)
                    loss = loss + self.causal_machine_state_loss_coeff * state_loss.to(dtype=loss.dtype)
        state_space_backbone_loss = self._compute_state_space_backbone_loss(
            target_ids,
            loss_mask,
            teacher_state_ids=teacher_state_ids,
            teacher_future_sketch=teacher_future_sketch,
        )
        if state_space_backbone_loss is not None:
            loss = loss + state_space_backbone_loss.to(dtype=loss.dtype)
        mid_aux_coeff_tensor: Tensor | None = None
        mid_aux_coeff_value: float | None = None
        if torch.is_tensor(mid_aux_loss_coeff):
            mid_aux_coeff_tensor = mid_aux_loss_coeff
        else:
            mid_aux_coeff_value = float(mid_aux_loss_coeff)
        if h_mid is not None and (mid_aux_coeff_tensor is not None or abs(mid_aux_coeff_value) > 0.0):
            if h_mid.size(1) == target_ids.size(1):
                mid_logits = self._mid_aux_logits(h_mid)
                mid_ce = F.cross_entropy(
                    mid_logits.reshape(-1, mid_logits.size(-1)).float(),
                    target_ids.reshape(-1),
                    reduction="none",
                    label_smoothing=label_smoothing,
                )
                if loss_mask is None:
                    mid_loss = mid_ce.mean()
                else:
                    mid_mask = loss_mask.reshape(-1).to(dtype=mid_ce.dtype)
                    mid_loss = (mid_ce * mid_mask).sum() / mid_mask.sum().clamp_min(1.0)
                if mid_aux_coeff_tensor is None:
                    mid_aux_coeff_tensor = torch.tensor(
                        float(mid_aux_coeff_value),
                        device=mid_loss.device,
                        dtype=mid_loss.dtype,
                    )
                else:
                    mid_aux_coeff_tensor = mid_aux_coeff_tensor.to(device=mid_loss.device, dtype=mid_loss.dtype)
                loss = loss + mid_aux_coeff_tensor * mid_loss
        return loss

    def forward_logits(
        self,
        input_ids: Tensor,
        backbone_cache: BackboneStepCache | None = None,
        causal_machine_cache: CausalMachineCache | None = None,
        update_causal_machine_cache: bool = False,
    ) -> Tensor:
        if backbone_cache is not None:
            x, _ = self._forward_features_step_strict(
                input_ids,
                backbone_cache=backbone_cache,
            )
        else:
            x, _ = self._forward_features(input_ids)
        if self.use_causal_machine_output_bias:
            causal_machine_logits, _, _, _ = self._compute_causal_machine_outputs(
                x,
                cache=causal_machine_cache,
                update_cache=update_causal_machine_cache,
            )
        else:
            causal_machine_logits = None
        return self._project_logits(
            x,
            causal_machine_logits=causal_machine_logits,
        )

def init_ema_state(module: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone().float() for name, tensor in module.state_dict().items()}


@torch.no_grad()
def update_ema_state(ema_state: dict[str, Tensor], module: nn.Module, decay: float) -> None:
    one_minus_decay = 1.0 - decay
    state_items = list(module.state_dict().items())
    if not state_items:
        return
    ema_tensors = [ema_state[name] for name, _tensor in state_items]
    current_tensors = [tensor.detach().float() for _name, tensor in state_items]
    try:
        torch._foreach_mul_(ema_tensors, decay)
        torch._foreach_add_(ema_tensors, current_tensors, alpha=one_minus_decay)
    except Exception:
        for name, tensor in state_items:
            ema_state[name].mul_(decay).add_(tensor.detach().float(), alpha=one_minus_decay)


def fake_quant_active(args: Hyperparameters, step: int, elapsed_ms: float, lr_scale: float) -> bool:
    if args.fake_quant_bits <= 0:
        return False
    if args.fake_quant_full_run:
        return True
    if args.late_qat and lr_scale <= args.qat_threshold:
        return True
    if args.fake_quant_tail_steps <= 0:
        return False
    if step >= max(args.iterations - args.fake_quant_tail_steps, 0):
        return True
    if args.max_wallclock_seconds > 0 and step > 0:
        step_ms = elapsed_ms / step
        remaining_ms = 1000.0 * args.max_wallclock_seconds - elapsed_ms
        return remaining_ms <= args.fake_quant_tail_steps * step_ms
    return False

def load_initial_model_state(model: nn.Module, path: str, strict: bool) -> tuple[list[str], list[str]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported INIT_MODEL_PATH payload type: {type(payload).__name__}")
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(getattr(incompatible, "missing_keys", [])), list(getattr(incompatible, "unexpected_keys", []))


def _sketch_token_tables(vocab_size: int, sketch_dim: int) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.arange(vocab_size, dtype=np.int64)
    bucket_ids = ((token_ids * np.int64(1103515245) + np.int64(12345)) % np.int64(max(sketch_dim, 1))).astype(
        np.int64, copy=False
    )
    sign_bits = ((token_ids * np.int64(214013) + np.int64(2531011)) >> np.int64(4)) & np.int64(1)
    signs = np.where(sign_bits == 0, 1.0, -1.0).astype(np.float32, copy=False)
    return bucket_ids, signs


def load_causal_machine_profile(profile_json_path: str) -> dict[str, object]:
    profile_path = Path(profile_json_path).expanduser()
    if not profile_path.is_file():
        raise FileNotFoundError(f"CAUSAL_MACHINE_PROFILE_JSON not found: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    future_profile = profile.get("future_signature_profile")
    if not isinstance(future_profile, dict) or not bool(future_profile.get("available")):
        raise ValueError(f"{profile_path} does not contain an available future_signature_profile")
    horizons = [int(v) for v in future_profile.get("horizons", []) if int(v) > 0]
    if not horizons:
        raise ValueError(f"{profile_path} does not contain valid future_signature_profile horizons")
    signature_dim = int(future_profile.get("signature_dim", 0))
    if signature_dim <= 0 or signature_dim % len(horizons) != 0:
        raise ValueError(f"{profile_path} has invalid future signature dimension {signature_dim} for horizons={horizons}")
    per_horizon_dim = signature_dim // len(horizons)
    sketch_dim = per_horizon_dim - 1
    if sketch_dim <= 0:
        raise ValueError(f"{profile_path} implies non-positive causal machine sketch dim {sketch_dim}")
    spectral_block = profile.get("spectral_eigenbases")
    if not isinstance(spectral_block, dict):
        raise ValueError(f"{profile_path} is missing spectral_eigenbases metadata")
    sidecar_path = spectral_block.get("sidecar_npz")
    if not isinstance(sidecar_path, str) or not sidecar_path.strip():
        candidate = profile_path.with_name(f"{profile_path.stem}_spectral_eigenbases.npz")
        sidecar = candidate
    else:
        sidecar = Path(sidecar_path)
    if not sidecar.is_absolute():
        sidecar = (profile_path.parent / sidecar).resolve()
    if not sidecar.is_file():
        raise FileNotFoundError(f"Causal machine sidecar not found: {sidecar}")
    with np.load(sidecar) as arrays:
        if "causal_machine_signature_centroids" not in arrays or "causal_machine_log_probs" not in arrays:
            raise ValueError(f"{sidecar} is missing causal machine arrays")
        centroids = arrays["causal_machine_signature_centroids"].astype(np.float32, copy=False)
        log_probs = arrays["causal_machine_log_probs"].astype(np.float32, copy=False)
        state_masses = (
            arrays["causal_machine_state_masses"].astype(np.float32, copy=False)
            if "causal_machine_state_masses" in arrays
            else np.zeros((log_probs.shape[0],), dtype=np.float32)
        )
    keep_cols: list[int] = []
    for idx in range(len(horizons)):
        base_idx = idx * per_horizon_dim
        keep_cols.extend(range(base_idx, base_idx + sketch_dim))
    centroids_sketch = centroids[:, keep_cols].astype(np.float32, copy=False)
    return {
        "num_states": int(log_probs.shape[0]),
        "horizons": horizons,
        "sketch_dim": int(sketch_dim),
        "log_probs": log_probs,
        "state_masses": state_masses,
        "centroids_sketch": centroids_sketch,
    }


def validate_causal_machine_objective_config(args: Hyperparameters) -> None:
    args.causal_machine_teacher_loss_coeff = max(float(args.causal_machine_teacher_loss_coeff), 0.0)
    args.causal_machine_state_loss_coeff = max(float(args.causal_machine_state_loss_coeff), 0.0)
    args.causal_machine_next_token_loss_coeff = max(float(args.causal_machine_next_token_loss_coeff), 0.0)
    args.causal_machine_transition_kl_coeff = max(float(args.causal_machine_transition_kl_coeff), 0.0)
    args.causal_machine_future_sketch_loss_coeff = max(float(args.causal_machine_future_sketch_loss_coeff), 0.0)
    uses_offline_teacher = args.causal_machine_teacher_loss_coeff > 0.0 or args.causal_machine_state_loss_coeff > 0.0
    has_profile_teacher_artifact = bool(str(args.causal_machine_profile_json or "").strip())
    if has_profile_teacher_artifact and not args.causal_machine_allow_offline_teacher:
        raise ValueError(
            "CAUSAL_MACHINE_PROFILE_JSON is an offline teacher artifact. "
            "Set CAUSAL_MACHINE_ALLOW_OFFLINE_TEACHER=1 to opt into that path explicitly."
        )
    if uses_offline_teacher and not args.causal_machine_allow_offline_teacher:
        raise ValueError(
            "CAUSAL_MACHINE_TEACHER_LOSS_COEFF and CAUSAL_MACHINE_STATE_LOSS_COEFF "
            "require CAUSAL_MACHINE_ALLOW_OFFLINE_TEACHER=1."
        )
    if uses_offline_teacher and not has_profile_teacher_artifact:
        raise ValueError(
            "Offline causal-machine teacher losses require CAUSAL_MACHINE_PROFILE_JSON."
        )


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global BOS_ID, zeropower_via_newtonschulz5

    process_started_at = time.perf_counter()
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    validate_causal_machine_objective_config(args)
    configure_runtime_export(args)
    if args.enable_torch_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    launched_with_dist_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = launched_with_dist_env and world_size > 1
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps_override = os.environ.get("GRAD_ACCUM_STEPS")
    if grad_accum_steps_override is None:
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
        grad_accum_steps = 8 // world_size
    else:
        grad_accum_steps = int(grad_accum_steps_override)
        if grad_accum_steps <= 0:
            raise ValueError(f"GRAD_ACCUM_STEPS must be positive, got {grad_accum_steps}")
    grad_scale = 1.0 / grad_accum_steps
    if args.train_batch_tokens % (world_size * grad_accum_steps * args.train_seq_len) != 0:
        raise ValueError(
            "TRAIN_BATCH_TOKENS must be divisible by WORLD_SIZE * GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN; "
            f"got TRAIN_BATCH_TOKENS={args.train_batch_tokens}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(args.enable_cudnn_sdp)
    enable_flash_sdp(args.enable_flash_sdp)
    enable_mem_efficient_sdp(args.enable_mem_efficient_sdp)
    enable_math_sdp(args.enable_math_sdp)

    raw_model_path = "final_model.pt"
    quant_model_path = "final_model.int8.ptz"
    best_raw_model_path = args.best_val_bpb_checkpoint_path or "best_val_bpb_model.pt"
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = os.path.join("logs", f"{args.run_id}.txt")
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    def end_to_end_wallclock_ms() -> float:
        return 1000.0 * (time.perf_counter() - process_started_at)

    log0(code, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer_kind = ""
    if args.tokenizer_path.endswith(".model"):
        tokenizer_kind = "sentencepiece"
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        tokenizer_vocab_size = int(sp.vocab_size())
        bos_token_id = int(sp.bos_id())
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            sp, args.vocab_size, device
        )
    elif args.tokenizer_path.endswith(".json"):
        from tokenizers import Tokenizer

        tokenizer_kind = "tokenizers_json"
        tok = Tokenizer.from_file(args.tokenizer_path)
        tokenizer_vocab_size = int(tok.get_vocab_size())
        bos_token_id = tok.token_to_id("<s>")
        if bos_token_id is None:
            bos_token_id = tok.token_to_id("<bos>")
        if bos_token_id is None:
            bos_token_id = -1
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_tokenizers_json_luts(
            args.tokenizer_path, args.vocab_size, device
        )
    else:
        raise ValueError(f"Unsupported tokenizer format for TOKENIZER_PATH={args.tokenizer_path!r}; expected .model or .json")
    if tokenizer_vocab_size != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={tokenizer_vocab_size}"
        )
    BOS_ID = int(bos_token_id)
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    total_train_tokens = count_total_train_tokens(args.train_files)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    eval_seq_len = get_eval_seq_len(args)
    val_tokens_eval = val_tokens if eval_seq_len == args.train_seq_len else load_validation_tokens(args.val_files, eval_seq_len)
    log0(f"tokenizer:{tokenizer_kind} dataset:{dataset_name} train_tokens:{total_train_tokens}")
    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    budget_started_at = process_started_at

    if USE_CAUSAL_MACHINE_CUDA_SCAN:
        load_causal_machine_scan_cuda()
        log0("cuda_ext:preloaded causal_machine_scan_cuda", console=False)
    if USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN:
        load_causal_machine_latent_scan_cuda()
        log0("cuda_ext:preloaded causal_machine_latent_scan_cuda", console=False)
    if args.use_muon and USE_MUON_CUDA:
        load_muon_cuda()
        log0("cuda_ext:preloaded muon_cuda", console=False)

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden,
        tie_embeddings=args.tie_embeddings,
        use_output_logit_bias=args.use_output_logit_bias,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        overtone_embed_init=args.overtone_embed_init,
        overtone_embed_power=args.overtone_embed_power,
        overtone_embed_scale=args.overtone_embed_scale,
        resid_mix_phase_init=args.resid_mix_phase_init,
        resid_mix_phase_sharpness=args.resid_mix_phase_sharpness,
        resid_mix_phase_center=args.resid_mix_phase_center,
        num_shared_layers=args.num_shared_layers,
        shared_layer_repeats=args.shared_layer_repeats,
        attention_kv_mode=args.attention_kv_mode,
        use_causal_machine_bias=args.use_causal_machine_bias,
        use_causal_machine_output_bias=args.use_causal_machine_output_bias,
        use_causal_machine_backbone=args.use_causal_machine_backbone,
        block_pattern=args.block_pattern,
        causal_machine_profile_json=args.causal_machine_profile_json,
        causal_machine_num_states=args.causal_machine_num_states,
        causal_machine_hidden_rank=args.causal_machine_hidden_rank,
        causal_machine_transition_rank=args.causal_machine_transition_rank,
        causal_machine_latent_rank=args.causal_machine_latent_rank,
        causal_machine_latent_mode=args.causal_machine_latent_mode,
        causal_machine_scale_init=args.causal_machine_scale_init,
        causal_machine_gate_init=args.causal_machine_gate_init,
        causal_machine_latent_gate_init=args.causal_machine_latent_gate_init,
        causal_machine_latent_decay_init=args.causal_machine_latent_decay_init,
        causal_machine_teacher_loss_coeff=args.causal_machine_teacher_loss_coeff,
        causal_machine_state_loss_coeff=args.causal_machine_state_loss_coeff,
        causal_machine_next_token_loss_coeff=args.causal_machine_next_token_loss_coeff,
        causal_machine_transition_kl_coeff=args.causal_machine_transition_kl_coeff,
        causal_machine_future_sketch_loss_coeff=args.causal_machine_future_sketch_loss_coeff,
        causal_machine_transition_gate_init=args.causal_machine_transition_gate_init,
        causal_machine_transition_stickiness_init=args.causal_machine_transition_stickiness_init,
        causal_machine_emit_delta_scale_init=args.causal_machine_emit_delta_scale_init,
        shared_tail_output_gate=args.shared_tail_output_gate,
        shared_tail_output_init=args.shared_tail_output_init,
        shared_tail_enable_step=args.shared_tail_enable_step,
        shared_tail_ramp_steps=args.shared_tail_ramp_steps,
        shared_tail_max_mult=args.shared_tail_max_mult,
        signed_skip_weights=args.signed_skip_weights,
        orthogonal_init=args.orthogonal_init,
        mup_proj_init=args.mup_proj_init,
        train_seq_len=args.train_seq_len,
        use_flash_attn_3=args.use_flash_attn_3,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        use_adaptive_rmsnorm=args.use_adaptive_rmsnorm,
        adaptive_rmsnorm_gate_init=args.adaptive_rmsnorm_gate_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if args.causal_machine_profile_json:
        causal_machine_profile = load_causal_machine_profile(args.causal_machine_profile_json)
        base_model.apply_causal_machine_profile(causal_machine_profile, vocab_size=args.vocab_size)
        log0(
            f"causal_machine_profile:loaded states:{int(causal_machine_profile['num_states'])} "
            f"horizons:{','.join(str(int(v)) for v in causal_machine_profile['horizons'])} "
            f"sketch_dim:{int(causal_machine_profile['sketch_dim'])}"
        )
    if args.mid_aux_loss_coeff <= 0.0 and getattr(base_model, "mid_aux_head", None) is not None:
        for param in base_model.mid_aux_head.parameters():
            param.requires_grad_(False)
    if args.init_model_path:
        missing_keys, unexpected_keys = load_initial_model_state(
            base_model,
            args.init_model_path,
            strict=args.init_model_strict,
        )
        log0(
            f"init_model:loaded path:{args.init_model_path} strict:{int(args.init_model_strict)} "
            f"missing:{len(missing_keys)} unexpected:{len(unexpected_keys)}"
        )
        if missing_keys:
            log0(f"init_model_missing:{','.join(missing_keys[:12])}")
        if unexpected_keys:
            log0(f"init_model_unexpected:{','.join(unexpected_keys[:12])}")
    enable_model_compile = args.enable_torch_compile
    if enable_model_compile:
        _preload_compiled_runtime_extensions(args)
        _prime_compiled_structured_scan_kernel_configs(
            base_model,
            train_seq_len=int(args.train_seq_len),
            eval_seq_len=int(args.eval_seq_len),
            device=device,
        )
    if enable_model_compile:
        pass
    compiled_model = (
        torch.compile(base_model, dynamic=False, fullgraph=False)
        if enable_model_compile
        else base_model
    )
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    ema_state = init_ema_state(base_model) if args.ema_decay > 0.0 else None
    ema_state_updated = False
    cuda_graph_disable_reason = "pending_optimizer_setup"
    cuda_graph_eligible = False

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    embed_params = [base_model.tok_emb.weight]
    block_named_params = (
        [(f"blocks.{name}", p) for name, p in base_model.blocks.named_parameters()]
        + [(f"shared_blocks.{name}", p) for name, p in base_model.shared_blocks.named_parameters()]
    )
    attn_matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        and ".attn." in name
    ]
    mlp_matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        and ".mlp." in name
    ]
    other_matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        and ".attn." not in name
        and ".mlp." not in name
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.shared_tail_gate is not None:
        scalar_params.append(base_model.shared_tail_gate)
    if getattr(base_model, "output_logit_bias", None) is not None:
        scalar_params.append(base_model.output_logit_bias)
    if isinstance(base_model.final_norm, AdaptiveRMSNorm):
        scalar_params.append(base_model.final_norm.base_log_scale)
        if base_model.final_norm.cond_gate is not None:
            scalar_params.append(base_model.final_norm.cond_gate)
        if base_model.final_norm.cond_proj is not None:
            other_matrix_params.append(base_model.final_norm.cond_proj.weight)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_muon_attn: torch.optim.Optimizer | None = None
    optimizer_muon_mlp: torch.optim.Optimizer | None = None
    optimizer_muon_other: torch.optim.Optimizer | None = None
    optimizer_tok = torch.optim.Adam(
        [{"params": embed_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.token_weight_decay,
        fused=True,
        capturable=bool(args.use_cuda_graphs and not distributed),
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok]
    muon_optimizers: list[torch.optim.Optimizer] = []
    graph_capturable = bool(args.use_cuda_graphs and not distributed)
    if args.use_muon and attn_matrix_params:
        optimizer_muon_attn = Muon(
            attn_matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            backend_steps_light=args.muon_backend_steps_light,
            backend_refresh_interval=args.muon_backend_refresh_interval,
            weight_decay=args.muon_weight_decay,
            capturable=graph_capturable,
        )
        for group in optimizer_muon_attn.param_groups:
            group["base_lr"] = args.matrix_lr
        muon_optimizers.append(optimizer_muon_attn)
        optimizers.append(optimizer_muon_attn)
    if args.use_muon and mlp_matrix_params:
        mlp_matrix_lr = args.matrix_lr * args.mlp_matrix_lr_mult
        optimizer_muon_mlp = Muon(
            mlp_matrix_params,
            lr=mlp_matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            backend_steps_light=args.muon_backend_steps_light,
            backend_refresh_interval=args.muon_backend_refresh_interval,
            weight_decay=args.muon_weight_decay,
            capturable=graph_capturable,
        )
        for group in optimizer_muon_mlp.param_groups:
            group["base_lr"] = mlp_matrix_lr
        muon_optimizers.append(optimizer_muon_mlp)
        optimizers.append(optimizer_muon_mlp)
    if args.use_muon and other_matrix_params:
        other_matrix_lr = args.matrix_lr * args.other_matrix_lr_mult
        optimizer_muon_other = Muon(
            other_matrix_params,
            lr=other_matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            backend_steps_light=args.muon_backend_steps_light,
            backend_refresh_interval=args.muon_backend_refresh_interval,
            weight_decay=args.muon_weight_decay,
            capturable=graph_capturable,
        )
        for group in optimizer_muon_other.param_groups:
            group["base_lr"] = other_matrix_lr
        muon_optimizers.append(optimizer_muon_other)
        optimizers.append(optimizer_muon_other)
    if not args.use_muon and attn_matrix_params:
        optimizer_attn_matrix = torch.optim.Adam(
            [{"params": attn_matrix_params, "lr": args.matrix_lr, "base_lr": args.matrix_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.muon_weight_decay,
            fused=True,
            capturable=graph_capturable,
        )
        optimizers.append(optimizer_attn_matrix)
    if not args.use_muon and mlp_matrix_params:
        mlp_matrix_lr = args.matrix_lr * args.mlp_matrix_lr_mult
        optimizer_mlp_matrix = torch.optim.Adam(
            [{"params": mlp_matrix_params, "lr": mlp_matrix_lr, "base_lr": mlp_matrix_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.muon_weight_decay,
            fused=True,
            capturable=graph_capturable,
        )
        optimizers.append(optimizer_mlp_matrix)
    if not args.use_muon and other_matrix_params:
        optimizer_other_matrix = torch.optim.Adam(
            [{"params": other_matrix_params, "lr": args.matrix_lr, "base_lr": args.matrix_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.muon_weight_decay,
            fused=True,
            capturable=graph_capturable,
        )
        optimizers.append(optimizer_other_matrix)
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
        capturable=graph_capturable,
    )
    optimizers.append(optimizer_scalar)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.head_weight_decay,
            fused=True,
            capturable=graph_capturable,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    positional_mode = "rope_only"
    log0(f"positional_mode:{positional_mode} train_seq_len:{args.train_seq_len} iterations:{args.iterations}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(
        args.train_files,
        rank,
        world_size,
        device,
        bos_token_id=bos_token_id,
        random_offset_tokens=args.train_random_offset_tokens,
        seed=args.seed,
        debug_static_shapes=args.debug_static_shapes,
    )

    def zero_grad_all(set_to_none: bool = True) -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    muon_graph_capture_ready = all(opt.prepare_cuda_graph_capture() for opt in muon_optimizers) if muon_optimizers else False
    muon_graph_capture_forced_off = bool(muon_optimizers) and args.muon_cuda_graph_mode == "off"
    muon_graph_capture_forced_on = bool(muon_optimizers) and args.muon_cuda_graph_mode == "on"
    graph_full_step_supported = (not distributed) and (
        not muon_optimizers or (muon_graph_capture_ready and not muon_graph_capture_forced_off)
    )
    graph_step_optimizers = list(optimizers) if graph_full_step_supported else [opt for opt in optimizers if not isinstance(opt, Muon)]

    def _prepare_graphable_optimizers() -> None:
        if not graph_full_step_supported:
            return
        for opt in graph_step_optimizers:
            if isinstance(opt, Muon):
                continue
            for group in opt.param_groups:
                lr_value = group.get("lr", 0.0)
                if isinstance(lr_value, Tensor):
                    continue
                lr_tensor = torch.tensor(float(lr_value), device=device, dtype=torch.float32)
                group["lr_tensor"] = lr_tensor
                group["lr"] = lr_tensor

    def _set_group_lr(group: dict[str, object], value: float) -> None:
        lr_tensor = group.get("lr_tensor")
        if isinstance(lr_tensor, Tensor):
            lr_tensor.fill_(float(value))
            group["lr"] = lr_tensor
        else:
            group["lr"] = float(value)

    def _set_group_scalar(group: dict[str, object], name: str, value: float) -> None:
        tensor_value = group.get(f"{name}_tensor")
        if isinstance(tensor_value, Tensor):
            tensor_value.fill_(float(value))
            if name in group:
                group[name] = tensor_value
        else:
            group[name] = float(value)

    _prepare_graphable_optimizers()
    if not args.use_cuda_graphs:
        cuda_graph_disable_reason = "disabled"
    elif distributed:
        cuda_graph_disable_reason = "distributed_full_step_unsupported"
    elif muon_graph_capture_forced_off:
        cuda_graph_disable_reason = "muon_full_step_forced_off"
    elif muon_graph_capture_forced_on and not muon_graph_capture_ready:
        muon_reasons = sorted({opt.graph_capture_disable_reason() for opt in muon_optimizers if opt.graph_capture_disable_reason()})
        suffix = f":{','.join(muon_reasons)}" if muon_reasons else ""
        cuda_graph_disable_reason = f"muon_full_step_forced_on_but_unsupported{suffix}"
    elif muon_optimizers and not muon_graph_capture_ready:
        muon_reasons = sorted({opt.graph_capture_disable_reason() for opt in muon_optimizers if opt.graph_capture_disable_reason()})
        suffix = f":{','.join(muon_reasons)}" if muon_reasons else ""
        cuda_graph_disable_reason = f"muon_full_step_unsupported{suffix}"
    else:
        cuda_graph_disable_reason = "eligible"
    cuda_graph_eligible = cuda_graph_disable_reason == "eligible"
    log0(
        f"optimizer_stack:{'muon' if args.use_muon else 'adam_only'} "
        f"muon_cuda_graph_mode:{args.muon_cuda_graph_mode} "
        f"cuda_graph_full_step:{int(graph_full_step_supported and args.use_cuda_graphs)} "
        f"cuda_graph_reason:{cuda_graph_disable_reason}"
    )

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    finalization_reserve_ms = max(float(args.wallclock_finalization_reserve_ms), 0.0)
    validation_reserve_floor_ms = max(float(args.wallclock_validation_reserve_ms), 0.0)

    def remaining_wallclock_ms() -> float:
        if max_wallclock_ms is None:
            return math.inf
        return max(max_wallclock_ms - end_to_end_wallclock_ms(), 0.0)

    def validation_reserve_ms(last_validation_time_ms: float) -> float:
        if max_wallclock_ms is None:
            return 0.0
        measured_reserve = last_validation_time_ms * 1.25 if last_validation_time_ms > 0.0 else 0.0
        return max(validation_reserve_floor_ms, measured_reserve)

    def lr_mul(step: int, elapsed_ms: float) -> float:
        scale = 1.0
        if args.lr_warmup_steps > 0:
            warmup_progress = min(max(step, 0) / max(args.lr_warmup_steps, 1), 1.0)
            warmup_progress = warmup_progress ** max(float(args.lr_warmup_power), 1e-6)
            start_scale = min(max(args.lr_warmup_init_scale, 0.0), 1.0)
            scale *= start_scale + (1.0 - start_scale) * warmup_progress
        if args.warmdown_iters <= 0:
            return scale
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                scale *= max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return scale
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            scale *= remaining_ms / max(warmdown_ms, 1e-9)
        return scale

    def tied_embed_lr_mul(step: int) -> float:
        warmup_mult = min(max(args.tied_embed_warmup_mult, 0.0), 1.0)
        warmup_steps = max(int(args.tied_embed_warmup_steps), 0)
        if warmup_steps <= 0:
            return 1.0
        progress = min(max(step, 0) / max(warmup_steps, 1), 1.0)
        return warmup_mult + (1.0 - warmup_mult) * progress

    def early_phase_frac(step: int) -> float:
        if args.early_phase_steps <= 0:
            return 0.0
        return 1.0 - min(max(step, 0) / max(args.early_phase_steps, 1), 1.0)

    def next_train_microbatch() -> tuple[Tensor, Tensor, Tensor | None]:
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
        loss_mask = None
        return x, y, loss_mask

    current_train_step = 0
    mid_aux_health_cap = 1.0
    active_mid_aux_loss_coeff = float(args.mid_aux_loss_coeff)
    def compute_training_loss(
        x: Tensor,
        y: Tensor,
        loss_mask: Tensor | None,
    ) -> Tensor:
        mid_aux_sched = 1.0
        if args.mid_aux_ramp_steps > 0:
            mid_aux_sched = min(
                max(float(current_train_step - args.mid_aux_enable_step) / float(args.mid_aux_ramp_steps), 0.0),
                1.0,
            )
        elif current_train_step < args.mid_aux_enable_step:
            mid_aux_sched = 0.0
        if args.mid_aux_decay_end_step > args.mid_aux_decay_start_step:
            if current_train_step >= args.mid_aux_decay_end_step:
                mid_aux_sched = 0.0
            elif current_train_step > args.mid_aux_decay_start_step:
                decay_progress = (
                    float(current_train_step - args.mid_aux_decay_start_step)
                    / float(args.mid_aux_decay_end_step - args.mid_aux_decay_start_step)
                )
                mid_aux_sched *= max(1.0 - decay_progress, 0.0)
        current_mid_aux_coeff = active_mid_aux_loss_coeff * mid_aux_sched * mid_aux_health_cap
        current_mid_aux_coeff_t = torch.tensor(current_mid_aux_coeff, device=x.device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            base_model.set_training_step(current_train_step)
            loss = model(
                x,
                y,
                loss_mask=loss_mask,
                label_smoothing=args.label_smoothing,
                z_loss_coeff=args.z_loss_coeff,
                logit_var_loss_coeff=args.logit_var_loss_coeff,
                mid_aux_loss_coeff=current_mid_aux_coeff_t,
            )
            if args.row_norm_loss_coeff > 0.0:
                loss = loss + args.row_norm_loss_coeff * row_norm_regularization(base_model)
            if args.outlier_loss_coeff > 0.0:
                loss = loss + args.outlier_loss_coeff * outlier_regularization(
                    base_model, args.outlier_loss_threshold
                )
        return loss

    def build_train_cuda_graph(
        sample_batches: Sequence[tuple[Tensor, Tensor, Tensor | None]],
        *,
        ema_active: bool,
    ) -> dict[str, object]:
        if not sample_batches:
            raise ValueError("build_train_cuda_graph requires at least one microbatch")
        sample_x = sample_batches[0][0]
        graph_runtime = get_structured_scan_graph_runtime(
            name="train_step",
            device=device,
            shape_bucket=_structured_scan_shape_bucket(*sample_x.shape, quantum=32),
        )
        static_batches: list[tuple[Tensor, Tensor, Tensor | None]] = []
        for micro_step, (sample_x, sample_y, sample_loss_mask) in enumerate(sample_batches):
            static_x = graph_runtime.reserve_like(f"train_x_{micro_step}", sample_x)
            static_y = graph_runtime.reserve_like(f"train_y_{micro_step}", sample_y)
            static_loss_mask = (
                graph_runtime.reserve_like(f"train_loss_mask_{micro_step}", sample_loss_mask)
                if sample_loss_mask is not None
                else None
            )
            static_x.copy_(sample_x, non_blocking=True)
            static_y.copy_(sample_y, non_blocking=True)
            if static_loss_mask is not None and sample_loss_mask is not None:
                static_loss_mask.copy_(sample_loss_mask, non_blocking=True)
            static_batches.append((static_x, static_y, static_loss_mask))
        static_loss = torch.zeros((), device=device, dtype=torch.float32)
        static_grad_norm = torch.zeros((), device=device, dtype=torch.float32)
        zero_grad_all(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            total_loss = torch.zeros((), device=device, dtype=torch.float32)
            for static_x, static_y, static_loss_mask in static_batches:
                captured_loss = compute_training_loss(static_x, static_y, static_loss_mask)
                total_loss.add_(captured_loss.detach().to(dtype=torch.float32))
                (captured_loss * grad_scale).backward()
            total_loss.div_(float(len(static_batches)))
            static_loss.copy_(total_loss)
            if args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
                static_grad_norm.copy_(grad_norm.detach().to(dtype=static_grad_norm.dtype))
            for opt in graph_step_optimizers:
                opt.step()
            if ema_active and ema_state is not None:
                update_ema_state(ema_state, base_model, args.ema_decay)
            for opt in graph_step_optimizers:
                opt.zero_grad(set_to_none=False)
        graph_runtime.capture_count += 1
        return {
            "graph": graph,
            "static_batches": static_batches,
            "static_loss": static_loss,
            "static_grad_norm": static_grad_norm,
            "graph_runtime": graph_runtime,
            "ema_active": bool(ema_active),
            "captures_full_step": True,
            "num_microbatches": int(len(static_batches)),
        }

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    warmup_prior_names = (
        warmup_prior_init_param_names(base_model, args.warmup_prior_init_groups)
        if args.warmup_prior_init and args.warmup_steps > 0
        else []
    )
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        warmed_prior_state: dict[str, Tensor] = {}
        model.train()
        base_model.set_fake_quant(0)
        base_model.set_training_step(0)
        for warmup_step in range(args.warmup_steps):
            current_train_step = warmup_step
            warmup_scale = lr_mul(warmup_step, 0.0)
            token_scale = warmup_scale * tied_embed_lr_mul(warmup_step)
            for opt in optimizers:
                scaled_lr = token_scale if opt is optimizer_tok else warmup_scale
                for group in opt.param_groups:
                    _set_group_lr(group, float(group["base_lr"]) * scaled_lr)
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y, loss_mask = next_train_microbatch()
                warmup_loss = compute_training_loss(x, y, loss_mask)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        if warmup_prior_names:
            current_state = base_model.state_dict()
            warmed_prior_state = {
                name: current_state[name].detach().cpu().clone()
                for name in warmup_prior_names
                if name in current_state
            }
        restored_model_state = {name: tensor.detach().clone() for name, tensor in initial_model_state.items()}
        if warmed_prior_state:
            blend = min(max(float(args.warmup_prior_init_blend), 0.0), 1.0)
            transferred_param_count = 0
            transferred_element_count = 0
            for name, warmed_tensor in warmed_prior_state.items():
                if name not in restored_model_state:
                    continue
                restored_model_state[name] = transfer_warmup_state_tensor(
                    restored_model_state[name],
                    warmed_tensor,
                    blend,
                    args.warmup_prior_init_mode,
                    args.warmup_prior_init_delta_rms_mult,
                    args.warmup_prior_init_delta_rms_floor,
                )
                transferred_param_count += 1
                transferred_element_count += int(restored_model_state[name].numel())
            preview = ",".join(warmup_prior_names[:6])
            if len(warmup_prior_names) > 6:
                preview = f"{preview},..."
            log0(
                f"warmup_prior_init:enabled groups:{','.join(args.warmup_prior_init_groups)} "
                f"mode:{args.warmup_prior_init_mode} blend:{blend:.3f} "
                f"params:{transferred_param_count} elements:{transferred_element_count} names:{preview}"
            )
        base_model.load_state_dict(restored_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        current_train_step = 0
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        base_model.set_fake_quant(0)
        train_loader = DistributedTokenLoader(
            args.train_files,
            rank,
            world_size,
            device,
            bos_token_id=bos_token_id,
            random_offset_tokens=args.train_random_offset_tokens,
            seed=args.seed,
            debug_static_shapes=args.debug_static_shapes,
        )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 1000.0 * (time.perf_counter() - budget_started_at)
    stop_after_step: int | None = None
    raw_final_val_loss = math.nan
    raw_final_val_bpb = math.nan
    best_val_bpb = math.inf
    best_val_loss = math.nan
    best_val_step = -1
    best_val_train_tokens = 0
    best_bootstrap_val_bpb = math.inf
    regression_bad_validations = 0
    scale = lr_mul(0, end_to_end_wallclock_ms())
    cuda_graph_runner: dict[str, object] | None = None
    cuda_graph_runner_fake_quant_bits: int | None = None
    cuda_graph_retry_step = max(args.cuda_graph_warmup_steps, 0)
    graph_captures_grad_clip = args.grad_clip_norm > 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    optimizer_step_time_ms_total = 0.0
    muon_step_time_ms_total = 0.0
    muon_cuda_time_ms_total = 0.0
    muon_fallback_time_ms_total = 0.0
    muon_cuda_tensor_count_total = 0
    muon_fallback_tensor_count_total = 0
    muon_cuda_failure_count_total = 0
    last_validation_time_ms = 0.0

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        train_tokens_seen = int(step) * int(args.train_batch_tokens)
        bootstrap_active = (
            step < max(int(args.objective_bootstrap_steps), 0)
            or train_tokens_seen < max(int(args.objective_bootstrap_tokens), 0)
        )
        scheduled_validation = args.val_loss_every > 0 and step % args.val_loss_every == 0
        force_terminal_validation = last_step and stop_after_step is None
        should_validate = force_terminal_validation or (scheduled_validation and not last_step)
        if should_validate and max_wallclock_ms is not None:
            reserve_needed_ms = finalization_reserve_ms + validation_reserve_ms(last_validation_time_ms)
            if remaining_wallclock_ms() <= reserve_needed_ms:
                should_validate = False
                if stop_after_step is None:
                    stop_after_step = step
        if should_validate:
            current_train_step = step
            base_model.set_training_step(current_train_step)
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            validation_started_at = time.perf_counter()
            validation_max_seqs = args.val_max_seqs
            val_loss, val_bpb, _val_mode, _val_bits_per_token, _val_tokens_per_byte, _val_bytes_per_token = run_validation(
                args,
                model,
                base_model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                max_seqs=validation_max_seqs,
            )
            last_validation_time_ms = (time.perf_counter() - validation_started_at) * 1000.0
            if bootstrap_active and math.isfinite(float(val_bpb)):
                best_bootstrap_val_bpb = min(best_bootstrap_val_bpb, float(val_bpb))
            wallclock_ms = end_to_end_wallclock_ms()
            log0(
                f"step:{step}/{args.iterations} "
                f"val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"wallclock:{wallclock_ms:.0f}ms step_avg:{wallclock_ms / max(step, 1):.2f}ms"
            )
            raw_final_val_loss = val_loss
            raw_final_val_bpb = val_bpb
            improved_best = math.isfinite(float(val_bpb)) and (
                best_val_step < 0 or float(val_bpb) < best_val_bpb - args.regression_stop_min_delta
            )
            if improved_best:
                best_val_bpb = float(val_bpb)
                best_val_loss = float(val_loss)
                best_val_step = int(step)
                best_val_train_tokens = int(train_tokens_seen)
                regression_bad_validations = 0
                if master_process and args.save_best_val_bpb_checkpoint:
                    checkpoint_payload = {
                        "model_state_dict": base_model.state_dict(),
                        "step": best_val_step,
                        "train_tokens_seen": best_val_train_tokens,
                        "val_bpb": best_val_bpb,
                        "val_loss": best_val_loss,
                        "run_id": args.run_id,
                    }
                    torch.save(checkpoint_payload, best_raw_model_path)
            elif (
                args.regression_stop_patience > 0
                and step >= args.regression_stop_min_step
                and best_val_step >= 0
                and float(val_bpb) > best_val_bpb + args.regression_stop_min_delta
            ):
                regression_bad_validations += 1
                if regression_bad_validations >= args.regression_stop_patience and stop_after_step is None:
                    stop_after_step = int(step)
            else:
                regression_bad_validations = 0
            # Keep the captured graph resident across validation so the training
            # path does not pay repeated recapture and cache-flush costs.
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        current_train_step = step
        elapsed_ms = end_to_end_wallclock_ms()
        scale = lr_mul(step, elapsed_ms)
        fake_quant_bits = args.fake_quant_bits if fake_quant_active(args, step, elapsed_ms, scale) else 0
        base_model.set_fake_quant(fake_quant_bits)
        graph_step_active = cuda_graph_eligible and step >= max(args.cuda_graph_warmup_steps, 0) and step >= cuda_graph_retry_step
        if cuda_graph_runner is not None and cuda_graph_runner_fake_quant_bits != fake_quant_bits:
            cuda_graph_runner = None
            cuda_graph_runner_fake_quant_bits = None
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in muon_optimizers:
            for group in opt.param_groups:
                _set_group_scalar(group, "momentum", muon_momentum)

        early_frac = early_phase_frac(step)
        attn_lr_scale = 1.0 + early_frac * (args.early_muon_attn_lr_scale - 1.0)
        mlp_lr_scale = 1.0 + early_frac * (args.early_muon_mlp_lr_scale - 1.0)
        other_lr_scale = 1.0 + early_frac * (args.early_muon_other_lr_scale - 1.0)
        muon_wd_scale = 1.0 + early_frac * (args.early_muon_wd_scale - 1.0)

        token_scale = scale * tied_embed_lr_mul(step)
        for opt in optimizers:
            scaled_lr = token_scale if opt is optimizer_tok else scale
            if opt is optimizer_muon_attn:
                scaled_lr *= attn_lr_scale
            elif opt is optimizer_muon_mlp:
                scaled_lr *= mlp_lr_scale
            elif opt is optimizer_muon_other:
                scaled_lr *= other_lr_scale
            for group in opt.param_groups:
                _set_group_lr(group, float(group["base_lr"]) * scaled_lr)
                if opt is optimizer_muon_attn or opt is optimizer_muon_mlp or opt is optimizer_muon_other:
                    _set_group_scalar(group, "weight_decay", args.muon_weight_decay * muon_wd_scale)

        step_batches = [next_train_microbatch() for _ in range(grad_accum_steps)]
        used_cuda_graph = False
        train_loss = torch.zeros((), device=device)
        ema_active = ema_state is not None and step + 1 >= args.ema_start_step
        graph_runner_invalid = (
            cuda_graph_runner is None
            or bool(cuda_graph_runner.get("ema_active", False)) != bool(ema_active)
            or int(cuda_graph_runner.get("num_microbatches", 0)) != len(step_batches)
        )
        if graph_step_active:
            if graph_runner_invalid:
                try:
                    cuda_graph_runner = build_train_cuda_graph(step_batches, ema_active=ema_active)
                    cuda_graph_runner_fake_quant_bits = fake_quant_bits
                    cuda_graph_retry_step = step
                    used_cuda_graph = True
                    train_loss = cuda_graph_runner["static_loss"]  # type: ignore[assignment]
                    ema_state_updated = ema_state_updated or ema_active
                except Exception as exc:
                    cuda_graph_disable_reason = f"capture_failed:{type(exc).__name__}"
                    cuda_graph_retry_step = step + max(args.val_loss_every, 16, 1)
                    cuda_graph_runner = None
                    cuda_graph_runner_fake_quant_bits = None
                    zero_grad_all()
            else:
                static_batches = cuda_graph_runner["static_batches"]
                for (x, y, loss_mask), (static_x, static_y, static_loss_mask) in zip(step_batches, static_batches, strict=True):
                    static_x.copy_(x, non_blocking=True)
                    static_y.copy_(y, non_blocking=True)
                    if static_loss_mask is not None and loss_mask is not None:
                        static_loss_mask.copy_(loss_mask, non_blocking=True)
                cuda_graph_runner["graph"].replay()
                used_cuda_graph = True
                train_loss = cuda_graph_runner["static_loss"]  # type: ignore[assignment]
                ema_state_updated = ema_state_updated or ema_active

        if not used_cuda_graph:
            zero_grad_all(set_to_none=True)
            for micro_step, (x, y, loss_mask) in enumerate(step_batches):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                loss = compute_training_loss(x, y, loss_mask)
                train_loss += loss.detach()
                (loss * grad_scale).backward()
            train_loss /= grad_accum_steps

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
            optimizer_step_started_at = time.perf_counter()
            for opt in optimizers:
                opt.step()
            optimizer_step_time_ms = (time.perf_counter() - optimizer_step_started_at) * 1000.0
            optimizer_step_time_ms_total += optimizer_step_time_ms
            if PROFILE_MUON_STEP:
                for opt in muon_optimizers:
                    stats = getattr(opt, "last_step_stats", None)
                    if not isinstance(stats, dict):
                        continue
                    muon_step_time_ms_total += float(stats.get("total_ms", 0.0))
                    muon_cuda_time_ms_total += float(stats.get("cuda_ms", 0.0))
                    muon_fallback_time_ms_total += float(stats.get("fallback_ms", 0.0))
                    muon_cuda_tensor_count_total += int(stats.get("cuda_tensor_count", 0))
                    muon_fallback_tensor_count_total += int(stats.get("fallback_tensor_count", 0))
                    muon_cuda_failure_count_total += int(stats.get("cuda_failure_count", 0))
            zero_grad_all(set_to_none=True)
            if ema_active:
                update_ema_state(ema_state, base_model, args.ema_decay)  # type: ignore[arg-type]
                ema_state_updated = True

        train_loss_value = float(train_loss.item())

        step += 1
        approx_wallclock_ms = end_to_end_wallclock_ms()
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            muon_log_suffix = ""
            if PROFILE_MUON_STEP:
                muon_tensor_total = muon_cuda_tensor_count_total + muon_fallback_tensor_count_total
                muon_fallback_frac = (
                    float(muon_fallback_tensor_count_total) / float(muon_tensor_total)
                    if muon_tensor_total > 0
                    else 0.0
                )
                muon_log_suffix = (
                    f" opt_step_avg:{optimizer_step_time_ms_total / step:.2f}ms"
                    f" muon_avg:{muon_step_time_ms_total / step:.2f}ms"
                    f" muon_cuda_avg:{muon_cuda_time_ms_total / step:.2f}ms"
                    f" muon_fallback_avg:{muon_fallback_time_ms_total / step:.2f}ms"
                    f" muon_fallback_frac:{muon_fallback_frac:.3f}"
                    f" muon_cuda_failures:{muon_cuda_failure_count_total}"
                )
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"wallclock:{approx_wallclock_ms:.0f}ms step_avg:{approx_wallclock_ms / step:.2f}ms"
                f"{muon_log_suffix}"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = (
            max_wallclock_ms is not None
            and approx_wallclock_ms >= max(max_wallclock_ms - finalization_reserve_ms, 0.0)
        )
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if ema_state is not None and ema_state_updated:
        base_model.load_state_dict(ema_state, strict=True)

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process and args.save_raw_debug_model:
        torch.save(base_model.state_dict(), raw_model_path)
        model_bytes = os.path.getsize(raw_model_path)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Raw debug size: {model_bytes + code_bytes} bytes")

    export_state_dict = canonicalize_state_dict_for_export(base_model.state_dict())
    if args.export_quant_bits < 8:
        quant_obj, mixed_stats, mixed_precision_info = quantize_state_dict_mixed_precision(
            export_state_dict, bits=args.export_quant_bits
        )
        quant_stats = {
            "param_count": mixed_stats["param_count"],
            "num_tensors": mixed_stats["num_tensors"],
            "num_float_tensors": mixed_stats["num_float_tensors"],
            "num_nonfloat_tensors": mixed_stats["num_nonfloat_tensors"],
            "baseline_tensor_bytes": mixed_stats["baseline_tensor_bytes"],
            "int8_payload_bytes": mixed_stats["payload_bytes"],
            "auto_keep_count": mixed_stats["high_precision_tensor_count"],
            "auto_keep_extra_bytes": mixed_stats["high_precision_extra_bytes"],
            "auto_keep_row_group_count": 0,
            "auto_keep_row_group_extra_bytes": 0,
            "block_pruned_tensor_count": mixed_stats["block_pruned_tensor_count"],
            "block_prune_estimated_bytes_saved": mixed_stats["block_prune_estimated_bytes_saved"],
        }
        auto_keep_info = {**mixed_precision_info, "row_groups": {}}
    else:
        quant_obj, quant_stats, auto_keep_info = quantize_state_dict_int8(export_state_dict)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob, used_codec = compress_blob(quant_raw, args.export_codec, args.export_zstd_level)
    quant_raw_bytes = len(quant_raw)
    submission_over_limit = False
    quant_file_bytes = None
    total_submission_bytes = None
    if master_process:
        with open(quant_model_path, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(quant_model_path)
        code_bytes = len(code.encode("utf-8"))
        total_submission_bytes = quant_file_bytes + code_bytes
        log0(f"Total submission size: {total_submission_bytes} bytes")
        submission_over_limit = total_submission_bytes > args.submission_size_limit_bytes

    if distributed:
        over_limit_tensor = torch.tensor(int(submission_over_limit), device=device)
        dist.all_reduce(over_limit_tensor, op=dist.ReduceOp.MAX)
        submission_over_limit = bool(over_limit_tensor.item())
    if submission_over_limit:
        raise RuntimeError(
            "Final int8+zlib artifact exceeds SUBMISSION_SIZE_LIMIT_BYTES; "
            "adjust model/code size or quantization settings."
        )

    if distributed:
        dist.barrier()
    with open(quant_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(decompress_blob(quant_blob_disk, used_codec)), map_location="cpu")
    if args.export_quant_bits < 8:
        base_model.load_state_dict(dequantize_state_dict_mixed_precision(quant_state), strict=True)
    else:
        base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    if args.run_final_quant_eval:
        torch.cuda.synchronize()
        if uses_sliding_eval(args):
            _q_val_loss, _q_val_bpb, _, _, _ = eval_val_sliding(
                args,
                base_model,
                rank,
                world_size,
                device,
                val_tokens_eval,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                seq_len_override=eval_seq_len,
                stride=args.eval_stride,
                batch_seqs=args.eval_batch_seqs,
                max_seqs=args.final_eval_max_seqs,
            )
        else:
            _q_val_loss, _q_val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens_eval,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                seq_len_override=eval_seq_len,
                max_seqs=args.final_eval_max_seqs,
            )
        torch.cuda.synchronize()
    final_wallclock_ms = end_to_end_wallclock_ms()
    if max_wallclock_ms is not None and final_wallclock_ms > max_wallclock_ms:
        raise RuntimeError(
            f"End-to-end wallclock budget exceeded: final_wallclock_ms={final_wallclock_ms:.0f} "
            f"> max_wallclock_ms={max_wallclock_ms:.0f}. Reduce startup, training, or finalization work."
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
