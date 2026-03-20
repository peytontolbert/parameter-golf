from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import zlib
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

import train_gpt_upgrade as tgu


def default_args() -> tgu.Hyperparameters:
    return tgu.Hyperparameters()


def build_model(args: tgu.Hyperparameters) -> tgu.GPT:
    return tgu.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden,
        num_register_tokens=args.num_register_tokens,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        overtone_embed_init=args.overtone_embed_init,
        overtone_embed_power=args.overtone_embed_power,
        overtone_embed_scale=args.overtone_embed_scale,
        resid_mix_phase_init=args.resid_mix_phase_init,
        resid_mix_phase_sharpness=args.resid_mix_phase_sharpness,
        use_alibi=args.use_alibi,
        token_shift=args.token_shift,
        use_xpos=args.use_xpos,
        xpos_scale_base=args.xpos_scale_base,
        num_shared_layers=args.num_shared_layers,
        shared_layer_repeats=args.shared_layer_repeats,
        shared_loop_gate_init=args.shared_loop_gate_init,
        use_residual_attention=args.use_residual_attention,
        residual_attention_gain_init=args.residual_attention_gain_init,
        memory_kv_slots=args.memory_kv_slots,
        use_relative_bias=args.use_relative_bias,
        relative_bias_num_buckets=args.relative_bias_num_buckets,
        relative_bias_max_distance=args.relative_bias_max_distance,
        macaron_layout=args.macaron_layout,
        retention_layers=args.retention_layers,
        retention_decay_init=args.retention_decay_init,
        retention_output_gate=args.retention_output_gate,
        moe_every_n_layers=args.moe_every_n_layers,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_hidden=args.moe_hidden,
        use_hyper_connections=args.use_hyper_connections,
        hyper_connection_layers=args.hyper_connection_layers,
        hyper_connection_gate_init=args.hyper_connection_gate_init,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(device_name)


def autocast_for(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return contextlib.nullcontext()


def maybe_load_checkpoint(model: torch.nn.Module, checkpoint_path: str | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
        candidate = checkpoint / "final_model.pt"
        if not candidate.is_file():
            raise FileNotFoundError(f"No final_model.pt found under run directory: {checkpoint}")
        checkpoint = candidate
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)


def count_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def model_artifact_stats(model: torch.nn.Module, code_path: Path, limit_bytes: int) -> dict[str, int | float | bool]:
    quant_obj, quant_stats, auto_keep_info = tgu.quantize_state_dict_int8(model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    code_bytes = len(code_path.read_text(encoding="utf-8").encode("utf-8"))
    quant_bytes = len(quant_blob)
    total_bytes = code_bytes + quant_bytes
    return {
        "param_count": int(quant_stats["param_count"]),
        "baseline_tensor_bytes": int(quant_stats["baseline_tensor_bytes"]),
        "int8_payload_bytes": int(quant_stats["int8_payload_bytes"]),
        "quant_zlib_bytes": quant_bytes,
        "code_bytes": code_bytes,
        "total_submission_bytes": total_bytes,
        "under_limit": total_bytes <= limit_bytes,
        "auto_keep_count": int(quant_stats["auto_keep_count"]),
        "auto_keep_extra_bytes": int(quant_stats["auto_keep_extra_bytes"]),
        "auto_keep_enabled": bool(auto_keep_info["enabled"]),
    }


def load_metric_inputs(args: tgu.Hyperparameters, device: torch.device):
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    val_tokens = tgu.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tgu.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    return val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def eval_val_local(
    args: tgu.Hyperparameters,
    model: torch.nn.Module,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    max_seqs: int,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size
    local_batch_seqs = max(1, local_batch_tokens // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if max_seqs > 0:
        total_seqs = min(total_seqs, max_seqs)
    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(0, total_seqs, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, total_seqs)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with autocast_for(device):
                batch_loss = model(x, y).detach().float().item()
            batch_token_count = float(y.numel())
            loss_sum += batch_loss * batch_token_count
            token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += token_bytes.to(torch.float64).sum().item()

    val_loss = loss_sum / token_count
    bits_per_token = val_loss / torch.log(torch.tensor(2.0)).item()
    tokens_per_byte = token_count / byte_count
    model.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)


def eval_val_sliding_local(
    args: tgu.Hyperparameters,
    model: torch.nn.Module,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    max_windows: int,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, args.eval_stride) if min(ws + seq_len, total_tokens) - ws >= args.eval_stride]
    if max_windows > 0:
        window_starts = window_starts[:max_windows]

    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0

    model.eval()
    with torch.inference_mode():
        for batch_idx in range(0, len(window_starts), args.eval_batch_seqs):
            batch_windows = window_starts[batch_idx : batch_idx + args.eval_batch_seqs]
            batch_size = len(batch_windows)
            x_batch = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
            y_batch = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
            window_lengths: list[int] = []

            for i, ws in enumerate(batch_windows):
                end = min(ws + seq_len, total_tokens)
                window_len = end - ws
                window_lengths.append(window_len)
                chunk = val_tokens[ws : end + 1].to(device=device, dtype=torch.int64)
                x_batch[i, :window_len] = chunk[:-1]
                y_batch[i, :window_len] = chunk[1:]

            with autocast_for(device):
                logits = model.forward_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(batch_size, seq_len)

            for i, ws in enumerate(batch_windows):
                window_len = window_lengths[i]
                score_start = 0 if ws == 0 else window_len - args.eval_stride
                scored_nll = nll[i, score_start:window_len].to(torch.float64)
                loss_sum += scored_nll.sum().item()
                token_count += float(window_len - score_start)
                tgt_ids = y_batch[i, score_start:window_len]
                prev_ids = x_batch[i, score_start:window_len]
                token_bytes = base_bytes_lut[tgt_ids].to(torch.float64)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                byte_count += token_bytes.sum().item()

    val_loss = loss_sum / token_count
    bits_per_token = val_loss / torch.log(torch.tensor(2.0)).item()
    tokens_per_byte = token_count / byte_count
    model.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)


def print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_smoke(cli_args: argparse.Namespace) -> None:
    args = default_args()
    device = resolve_device(cli_args.device)
    model = build_model(args).to(device)
    x = torch.randint(0, args.vocab_size, (cli_args.batch_size, args.train_seq_len), device=device)
    y = torch.randint(0, args.vocab_size, (cli_args.batch_size, args.train_seq_len), device=device)
    start = time.perf_counter()
    with autocast_for(device):
        loss = model(x, y)
    if cli_args.backward:
        loss.backward()
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    logits = model.forward_logits(x[:1]).detach()
    print_json(
        {
            "check": "smoke",
            "device": str(device),
            "loss": float(loss.detach().float().item()),
            "logits_shape": list(logits.shape),
            "elapsed_ms": elapsed_ms,
            "param_count": count_params(model),
        }
    )


def cmd_artifact(cli_args: argparse.Namespace) -> None:
    args = default_args()
    model = build_model(args)
    maybe_load_checkpoint(model, cli_args.checkpoint)
    stats = model_artifact_stats(model, Path(tgu.__file__), args.submission_size_limit_bytes)
    stats["check"] = "artifact"
    stats["checkpoint"] = cli_args.checkpoint
    print_json(stats)


def cmd_eval_probe(cli_args: argparse.Namespace) -> None:
    args = default_args()
    device = resolve_device(cli_args.device)
    model = build_model(args).to(device)
    maybe_load_checkpoint(model, cli_args.checkpoint)
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = load_metric_inputs(args, device)

    start = time.perf_counter()
    standard_loss, standard_bpb = eval_val_local(
        args,
        model,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        max_seqs=cli_args.max_seqs,
    )
    standard_ms = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    sliding_loss, sliding_bpb = eval_val_sliding_local(
        args,
        model,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        max_windows=cli_args.max_windows,
    )
    sliding_ms = 1000.0 * (time.perf_counter() - start)

    print_json(
        {
            "check": "eval_probe",
            "device": str(device),
            "checkpoint": cli_args.checkpoint,
            "standard": {"val_loss": standard_loss, "val_bpb": standard_bpb, "elapsed_ms": standard_ms},
            "sliding": {"val_loss": sliding_loss, "val_bpb": sliding_bpb, "elapsed_ms": sliding_ms},
            "delta_bpb_sliding_minus_standard": sliding_bpb - standard_bpb,
        }
    )


def cmd_roundtrip(cli_args: argparse.Namespace) -> None:
    args = default_args()
    device = resolve_device(cli_args.device)
    model = build_model(args).to(device)
    maybe_load_checkpoint(model, cli_args.checkpoint)
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    quant_obj, _, _ = tgu.quantize_state_dict_int8(original_state)
    restored_state = tgu.dequantize_state_dict_int8(quant_obj)
    model.load_state_dict(restored_state, strict=True)

    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    total_tensors = 0
    for name, orig in original_state.items():
        diff = (orig.float() - restored_state[name].float()).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().item()))
        mean_abs_diff += float(diff.mean().item())
        total_tensors += 1
    mean_abs_diff /= max(total_tensors, 1)

    payload: dict[str, object] = {
        "check": "roundtrip",
        "checkpoint": cli_args.checkpoint,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff_per_tensor": mean_abs_diff,
    }

    if cli_args.with_eval:
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = load_metric_inputs(args, device)
        val_loss, val_bpb = eval_val_local(
            args,
            model,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            max_seqs=cli_args.max_seqs,
        )
        payload["roundtrip_eval"] = {"val_loss": val_loss, "val_bpb": val_bpb}

    print_json(payload)


def cmd_throughput(cli_args: argparse.Namespace) -> None:
    args = default_args()
    device = resolve_device(cli_args.device)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, args.vocab_size, (cli_args.batch_size, args.train_seq_len), device=device)
    y = torch.randint(0, args.vocab_size, (cli_args.batch_size, args.train_seq_len), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(cli_args.steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device):
            loss = model(x, y)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mem_mib = torch.cuda.max_memory_allocated(device) // 1024 // 1024
    else:
        peak_mem_mib = 0
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    print_json(
        {
            "check": "throughput",
            "device": str(device),
            "steps": cli_args.steps,
            "batch_size": cli_args.batch_size,
            "train_seq_len": args.train_seq_len,
            "elapsed_ms": elapsed_ms,
            "step_avg_ms": elapsed_ms / cli_args.steps,
            "peak_mem_mib": int(peak_mem_mib),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight standalone checks for train_gpt_upgrade.py")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Random forward or forward+backward smoke test")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--backward", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    artifact = sub.add_parser("artifact", help="Estimate artifact size and budget fit")
    artifact.add_argument("--checkpoint", default=None)
    artifact.set_defaults(func=cmd_artifact)

    eval_probe = sub.add_parser("eval-probe", help="Compare standard and sliding eval on a small subset")
    eval_probe.add_argument("--device", default="cpu")
    eval_probe.add_argument("--checkpoint", default=None)
    eval_probe.add_argument("--max-seqs", type=int, default=32)
    eval_probe.add_argument("--max-windows", type=int, default=128)
    eval_probe.set_defaults(func=cmd_eval_probe)

    roundtrip = sub.add_parser("roundtrip", help="Check quantize/dequantize reconstruction and optional eval")
    roundtrip.add_argument("--device", default="cpu")
    roundtrip.add_argument("--checkpoint", default=None)
    roundtrip.add_argument("--with-eval", action="store_true")
    roundtrip.add_argument("--max-seqs", type=int, default=32)
    roundtrip.set_defaults(func=cmd_roundtrip)

    throughput = sub.add_parser("throughput", help="Run a tiny train-step microbenchmark")
    throughput.add_argument("--device", default="cpu")
    throughput.add_argument("--batch-size", type=int, default=2)
    throughput.add_argument("--steps", type=int, default=5)
    throughput.set_defaults(func=cmd_throughput)

    return parser


def main() -> None:
    parser = build_parser()
    cli_args = parser.parse_args()
    cli_args.func(cli_args)


if __name__ == "__main__":
    main()
