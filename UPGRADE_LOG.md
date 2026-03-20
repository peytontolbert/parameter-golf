# Upgrade Log

This file tracks the evolution of `train_gpt_upgrade.py` from the original `train_gpt.py`.

Use it as the single running source of truth for:
- what we changed
- why we changed it
- whether it worked
- what still needs validation

Methodologies that have not yet been implemented are tracked separately in `METHODOLOGY_BACKLOG.md`.

## Goal

Build a stronger competition-ready trainer while keeping the original `train_gpt.py` untouched.

Primary objective:
- improve final `final_int8_zlib_roundtrip_exact val_bpb`

Hard constraints:
- stay under `16,000,000` total submission bytes
- keep training reproducible
- preserve a clean baseline path for comparison

## Current Base

- Baseline source preserved in `train_gpt.py`
- Alternative submitter strategy reference preserved in `train_gptv2.py`
- Working upgrade branch lives in `train_gpt_upgrade.py`

## Status Legend

- `implemented`: code change is in place
- `validated`: tested and confirmed helpful
- `neutral`: tested but no clear gain
- `regression`: tested and harmful
- `pending`: not yet tested

## Upgrade Summary

| Date | Area | Change | Status | Notes |
|---|---|---|---|---|
| 2026-03-19 | Forking | Created `train_gpt_upgrade.py` from original `train_gpt.py` | implemented | Keeps baseline untouched |
| 2026-03-19 | Safety | Added manifest-based dataset/tokenizer pairing checks | implemented | Prevents silent mismatches in `val_bpb` setup |
| 2026-03-19 | Safety | Added hard submission-size limit check | implemented | Fails fast if final artifact exceeds `16,000,000` bytes |
| 2026-03-19 | Eval | Added optional sliding-window final eval via `EVAL_STRIDE` and `EVAL_BATCH_SEQS` | implemented | Inspired by the in-repo sliding eval submission |
| 2026-03-19 | Compression | Added `INT8_KEEP_FLOAT_NAME_PATTERNS` override | implemented | Lets us preserve sensitive tensors without hardcoding them |
| 2026-03-19 | Model sizing | Added `MLP_HIDDEN` override | implemented | Makes byte-budget tuning easier than only using `MLP_MULT` |
| 2026-03-19 | Runtime | Added `ENABLE_TORCH_COMPILE` toggle | implemented | Useful for debugging and environments where compile is unstable |
| 2026-03-19 | Local benchmarking | Added `VAL_MAX_SEQS` and `FINAL_EVAL_MAX_SEQS` proxy-eval caps | implemented | Speeds up local iteration while keeping full-eval behavior as the default |
| 2026-03-19 | Compression | Added tensor-sensitivity-based auto float passthrough at export | implemented | Can keep the highest-damage tensors in fp16 under a byte budget instead of relying only on manual name patterns |
| 2026-03-19 | Data loading | Added deterministic random-offset shard sampling via `TRAIN_RANDOM_OFFSET_TOKENS` | implemented | Gives short runs more local context diversity without changing the standalone dataset format |
| 2026-03-19 | Model | Added optional learned register tokens via `NUM_REGISTER_TOKENS` | implemented | Gives the tiny model cheap scratch-space capacity without changing the tokenizer or export format |
| 2026-03-19 | Optimizer | Added decoupled Muon weight decay via `MUON_WEIGHT_DECAY` | implemented | Matches the current top recipe more directly and may improve both generalization and quantization robustness |
| 2026-03-19 | Initialization | Added overtone spectral embedding init and phase-transition `resid_mix` init | implemented | Matches the current top recipe's init choices without requiring external dependencies |
| 2026-03-19 | Model | Added optional token shift via `TOKEN_SHIFT` | implemented | Cheap inductive-bias lane from the roadmap with almost no parameter cost |
| 2026-03-19 | Attention | Added optional ALiBi attention bias via `USE_ALIBI` | implemented | Gives a lightweight positional alternative to pure RoPE for short-run comparison |
| 2026-03-19 | Compression | Added row-group auto float passthrough via `INT8_AUTO_KEEP_ROW_BUDGET_BYTES` | implemented | Extends auto-keep from whole tensors to row groups so export bytes can be spent more finely |
| 2026-03-19 | Attention | Added XPos-style rotary scaling via `USE_XPOS` | implemented | Gives a lightweight context-extension companion to the existing RoPE path |
| 2026-03-19 | Depth | Added shared-tail recurrence with learned loop gates | implemented | Provides a minimal shared-block recurrence lane without replacing the main stack |
| 2026-03-20 | Attention | Added optional residual attention via `USE_RESIDUAL_ATTENTION` | implemented | Lets each layer reuse the previous layer's attention logits with a learned gate |
| 2026-03-20 | Memory | Added learned memory K/V slots via `MEMORY_KV_SLOTS` | implemented | Adds a lightweight persistent-memory lane inside attention |
| 2026-03-20 | Model | Added optional macaron block layout via `MACARON_LAYOUT` | implemented | Splits FFN capacity around attention with modest extra cost |
| 2026-03-20 | Attention | Added bucketed relative bias via `USE_RELATIVE_BIAS` | implemented | Gives a learned low-parameter alternative to ALiBi and pure RoPE |
| 2026-03-20 | Training | Added local distillation lane via `DISTILL_*` knobs | implemented | Enables teacher-logit KL for local single-process exploration without changing inference architecture |
| 2026-03-20 | Training | Added fake-quant tail finetune via `FAKE_QUANT_TAIL_*` | implemented | Lets the trainer spend the tail end of optimization adapting to quantized weights |
| 2026-03-20 | Context | Added `TRAINING_PRESET` long-context profiles | implemented | Makes `2048`-token context a first-class script preset with standard vs sliding eval variants |
| 2026-03-20 | Optimization | Fixed optimizer coverage for shared-tail and register-token params | implemented | `shared_blocks`, `shared_loop_gates`, and `register_tokens` now train instead of being left out of optimizer groups |
| 2026-03-20 | Training | Added row-norm regularization via `ROW_NORM_LOSS_COEFF` | implemented | Encourages more uniform per-row scales for matrices, which is a good fit for per-row int8 export |
| 2026-03-20 | Attention | Added retention-style upper layers via `RETENTION_*` | implemented | Lets upper blocks use exponentially decayed causal attention as a standalone first cut toward the retention roadmap |
| 2026-03-20 | Capacity | Added sparse MoE MLP subset via `MOE_*` | implemented | Lets every Nth layer switch to a tiny top-k expert FFN with optional load-balancing loss |
| 2026-03-20 | Memory | Removed late-layer PKM branch from active script | implemented | First cut was runnable but too large for the current competition path, so it was backed out to keep the trainer focused |
| 2026-03-20 | Depth | Added minimal hyper-connections around shared tail via `USE_HYPER_CONNECTIONS` | implemented | Introduces a two-stream residual mix around repeated shared blocks as a lightweight recurrence stabilizer |
| 2026-03-20 | Training | Added outlier regularization via `OUTLIER_LOSS_*` | implemented | Penalizes oversized matrix entries relative to tensor RMS to reduce quantization-sensitive outliers |
| 2026-03-20 | Training | Added logit-variance smoothing via `LOGIT_VAR_LOSS_COEFF` | implemented | Keeps logit spread from growing too sharply in short runs and may smooth the post-quant head |
| 2026-03-19 | Runtime | First `sp1024` baseline launch intended for `CUDA_VISIBLE_DEVICES=2` | regression | Blocked by shell command formatting; resulting OOM appears to have landed on a default 24 GiB GPU instead of physical GPU `2` |

## What We Know Works

- `train_gpt_upgrade.py` exists as a clean fork from the original trainer.
- The new file passed a linter check after edits.
- The original `train_gpt.py` remains unchanged.
- `train_gptv2.py` remains a reference script and is not part of the upgrade branch.

## What Is Not Yet Validated

- Whether sliding-window final eval is the best default for our run profile.
- Whether capped local proxy eval tracks full final eval closely enough to use as the default inner-loop metric.
- Which tensors should be kept in float for the best post-quant score/size tradeoff.
- Whether `MLP_HIDDEN` can buy better compression-quality tradeoffs than `MLP_MULT` alone.
- Whether `TRAIN_SEQ_LEN=2048` style tuning should become the default starting point.
- Whether compile on/off changes total 10-minute throughput enough to matter on our target hardware.

Current short-run policy:
- use `60s` runs as the default local comparison budget
- treat the original `train_gpt.py` as the true baseline
- treat `train_gpt_upgrade.py` with no extra knobs as the upgrade-branch baseline
- define a "60s run" as `60s` of training plus capped proxy eval, not full uncapped final eval
- rerun a matching `60s` baseline whenever the comparison frame changes
- keep uncapped final eval and `600s` runs for candidates that survive the `60s` screen

## What Has Not Worked

- The first `sp1024` baseline launch intended for `CUDA_VISIBLE_DEVICES=2` did not complete.
- Splitting the environment assignment across lines after `DATA_PATH=./data/datasets/` caused the shell to execute `fineweb10B_sp1024` as a separate command.
- The subsequent Python launch appears not to have inherited the intended `CUDA_VISIBLE_DEVICES=2` pin, so model initialization failed with `torch.OutOfMemoryError` on a 24 GiB GPU.
- Hardware check on this machine shows physical GPU `2` is a `12 GiB` RTX 3060, so the traceback reporting a `23.57 GiB` device cannot correspond to physical GPU `2`.

Important:
- Do not add guesses here after an experiment unless we have logs or a clear observation.
- If something is only an idea, put it under `Next Candidates`, not under failure notes.

## Experiment Log

### 2026-03-19

Starting point:
- created a fresh upgrade fork instead of modifying `train_gpt.py`
- avoided using `train_gptv2.py` as the base implementation

Observed result:
- infrastructure changes only
- no training run executed yet
- no `val_bpb` delta measured yet

### 2026-03-19 Local Run: `NUM_KV_HEADS=1` on 2 GPUs

Hypothesis:
- More aggressive KV sharing might reduce artifact bytes enough to be a strong next record-track candidate while still being trainable locally.

Config or command:
- `CUDA_VISIBLE_DEVICES=0,1,2`
- `torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py`
- `NUM_KV_HEADS=1 ENABLE_TORCH_COMPILE=0 MAX_WALLCLOCK_SECONDS=600`

Observed result:
- The run completed locally on `world_size:2` with `grad_accum_steps:4`.
- It reached only `505` steps before the `600s` wallclock cap.
- Final local metric at stop: `val_loss:2.5609`, `val_bpb:1.5167`.
- Peak memory: `13183 MiB allocated`, `13452 MiB reserved`.
- Final int8+zlib artifact: `9776669` bytes total including code.
- Final roundtrip sliding eval: `val_loss:2.54132017`, `val_bpb:1.50511488`.
- Final eval time: `790317ms`.

Judgment:
- neutral

Notes:
- This confirms `NUM_KV_HEADS=1` is locally trainable on a 2-GPU setup and remains comfortably under the artifact limit.
- But the local run is far too slow to be a faithful proxy for the competition target, with `step_avg:1189.38ms` and only `505` steps in `600s`.
- The final eval itself also took `790317ms`, which is another sign that this local setup should be treated as a feasibility rig, not a competition-quality timing proxy.
- Keep this as local feasibility evidence, not as a ranking-quality comparison against `8xH100` record-track runs.

### 2026-03-19 Baseline Run Attempt: intended `sp1024` run on `CUDA_VISIBLE_DEVICES=2`

Hypothesis:
- Use the cached `fineweb10B_sp1024` dataset and tokenizer to establish a first upgrade-script baseline on physical GPU `2`.

Config or command:
- Intended runtime pinning: `CUDA_VISIBLE_DEVICES=2`
- Intended dataset path: `DATA_PATH=./data/datasets/fineweb10B_sp1024`
- Tokenizer path: `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`
- Launch attempt captured in terminal history was split across lines:
  `CUDA_VISIBLE_DEVICES=2 RUN_ID=upgrade_baseline_gpu2 DATA_PATH=./data/datasets/`
  `fineweb10B_sp1024 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=200`
  `VAL_LOSS_EVERY=1000 python train_gpt_upgrade.py | tee baseline_gpu2.log`

Observed result:
- `train_gpt_upgrade.py` detected the cached `sp1024` tokenizer and saw a subset dataset with `train_shards:80/195`.
- The multiline shell invocation was malformed, so Bash first emitted `fineweb10B_sp1024: command not found`.
- The training process still reached model setup, then failed at `.to(device).bfloat16()` with `torch.OutOfMemoryError`.
- A later hardware check showed GPUs `0` and `1` are `24 GiB` RTX 3090s, while physical GPU `2` is a `12 GiB` RTX 3060.
- Because the traceback reported a `23.57 GiB` device, this failure could not have been on physical GPU `2`; the broken multiline command likely meant `python` ran without the intended `CUDA_VISIBLE_DEVICES=2` pin.

Judgment:
- regression

Notes:
- Keep documenting this run as an intended `CUDA_VISIBLE_DEVICES=2` attempt, not a confirmed successful pin to physical GPU `2`.
- Next rerun should keep the full environment assignment on one logical command line and confirm GPU `2` is actually free before model initialization.

### 2026-03-19 Local Run: `INT8_AUTO_KEEP_BUDGET_BYTES=1000000` on 2 GPUs

Hypothesis:
- Tensor-sensitivity-based auto float passthrough may recover post-quant quality more efficiently than manual tensor name rules.

Config or command:
- `CUDA_VISIBLE_DEVICES=0,1,2`
- `RUN_ID=auto_keep_1mb_60s_2gpu`
- `torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py`
- `MAX_WALLCLOCK_SECONDS=60 VAL_MAX_SEQS=64 FINAL_EVAL_MAX_SEQS=64`
- `INT8_AUTO_KEEP_BUDGET_BYTES=1000000 INT8_SENSITIVITY_LOG_TOPK=12`

Observed result:
- The run completed locally on `world_size:2` with `grad_accum_steps:4`.
- It reached `47` steps before the `60s` wallclock cap.
- Proxy validation at stop: `val_loss:6.1285`, `val_bpb:3.6235`.
- Peak memory: `13863 MiB allocated`, `14200 MiB reserved`.
- Final int8+zlib artifact: `7379915` bytes total including code.
- Auto-keep selected `3` tensors for `914944` extra payload bytes, led by `blocks.{4,1,3}.mlp.proj.weight`.
- Final roundtrip sliding eval: `val_loss:6.39365198`, `val_bpb:3.78668569`.
- Final eval time: `851720ms`.

Judgment:
- pending

Notes:
- The selected tensors look plausible, which is the main positive read from this first export-side run.
- The current `1MB` budget effectively buys about `3` full tensors, so the next clean sweep is `500k`, `1.5MB`, and `2MB`.
- Do not rank this run yet against the roadmap without a matching `60s` upgrade baseline or baseline-equivalent comparator.

## Execution Roadmap

Use this as the default order unless a run result suggests reordering:

1. Capture the first exact `60s` local baseline for the original `train_gpt.py`.
2. Capture the matching `60s` upgrade-branch baseline for `train_gpt_upgrade.py` with no extra knobs enabled.
3. Validate whether the `60s` ranking is stable enough to prune weak ideas before `600s` runs.
4. Sweep lower KV sharing options with `NUM_KV_HEADS=1,2,4` against that same `60s` upgrade baseline.
5. Add tensor sensitivity mapping and heterogeneous precision allocation so export bytes are spent where they recover the most post-quant `val_bpb`.
6. Sweep post-quant heuristics next, especially `INT8_KEEP_FLOAT_NAME_PATTERNS`, `MLP_HIDDEN`, and selective lower-bit allocation once sensitivity scores exist.
7. Test compression-aware regularization and quantization-grid alignment, starting with cheap options such as z-loss and ending with brief fake-quant tail training only if the cheap knobs help.
8. Compare standard final eval vs sliding-window final eval at fixed weights.
9. Test a `LongContextSeq2048`-style profile in `train_gpt_upgrade.py`.
10. Test stability cleanups such as QK norm / RMSNorm adjustments, EMA export, and label smoothing.
11. Implement shared-block recurrence only after the lower-risk sweeps have a clean baseline.
12. Implement learned memory/register tokens only after recurrence has an isolated read.
13. Revisit tokenizer-model sweeps after the trainer-side upgrades stabilize.

## One-Minute Commands

True baseline `60s` local comparison run using the original script and capped proxy eval:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

Matching `60s` upgrade-branch baseline run with capped proxy eval:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=upgrade_baseline_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

First two `60s` comparison candidates:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=kv1_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_KV_HEADS=1 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=kv1_keepemb_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_KV_HEADS=1 \
INT8_KEEP_FLOAT_NAME_PATTERNS=tok_emb.weight \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

First tensor-sensitivity export candidate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=auto_keep_1mb_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
INT8_AUTO_KEEP_BUDGET_BYTES=1000000 \
INT8_SENSITIVITY_LOG_TOPK=12 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

First random-offset training candidate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=random_offset_32k_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
TRAIN_RANDOM_OFFSET_TOKENS=32768 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

First register-token candidate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=register2_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
NUM_REGISTER_TOKENS=2 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

Top-recipe-aligned `60s` candidate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=top_recipe_10l_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=10 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
MUON_WEIGHT_DECAY=0.02 \
OVERTONE_EMBED_INIT=1 \
OVERTONE_EMBED_POWER=0.5 \
RESID_MIX_PHASE_INIT=1 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

Cheap-bias `60s` candidates:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=token_shift_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
TOKEN_SHIFT=1 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

New aggressive `60s` candidates:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=rowkeep_512k_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
INT8_AUTO_KEEP_ROW_BUDGET_BYTES=512000 \
INT8_AUTO_KEEP_ROW_GROUP_SIZE=32 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=xpos_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
USE_XPOS=1 \
XPOS_SCALE_BASE=512 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=sharedtail_1x2_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
NUM_SHARED_LAYERS=1 \
SHARED_LAYER_REPEATS=2 \
SHARED_LOOP_GATE_INIT=-2.0 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=resattn_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
USE_RESIDUAL_ATTENTION=1 \
RESIDUAL_ATTENTION_GAIN_INIT=0.5 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=memkv4_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
MEMORY_KV_SLOTS=4 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=macaron_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
MACARON_LAYOUT=1 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=relbias_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
USE_RELATIVE_BIAS=1 \
RELATIVE_BIAS_NUM_BUCKETS=32 \
RELATIVE_BIAS_MAX_DISTANCE=128 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

```bash
RUN_ID=distill_60s_local \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
DISTILL_TEACHER_CHECKPOINT=/abs/path/to/teacher/final_model.pt \
DISTILL_WEIGHT=0.25 \
DISTILL_TEMPERATURE=2.0 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
python train_gpt_upgrade.py
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=fakequant_tail_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
FAKE_QUANT_TAIL_STEPS=16 \
FAKE_QUANT_BITS=8 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=alibi_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
USE_ALIBI=1 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

Promotion rule:
- use capped proxy eval for the default `60s` loop
- rerun promising candidates without `FINAL_EVAL_MAX_SEQS` when promoting them to a more trustworthy read

## Next Candidates

- `NUM_KV_HEADS=1` and `NUM_KV_HEADS=2` against the exact `120s` upgrade baseline
- tensor sensitivity scorer for major matrices and row groups, then `fp16` / `int8` / `int6` allocation on the most sensitive weights
- `INT8_KEEP_FLOAT_NAME_PATTERNS=tok_emb.weight`
- compression-aware regularization and grid alignment, starting with z-loss and then brief fake-quant tail training if warranted
- `VAL_MAX_SEQS` / `FINAL_EVAL_MAX_SEQS` settings that shorten local turnaround without scrambling rankings
- `LongContextSeq2048`-style profile in `train_gpt_upgrade.py`
- `MLP_HIDDEN` sweep around the artifact limit instead of only changing `MLP_MULT`
- EMA export or eval path and mild label smoothing as cheap short-run stabilizers
- random-offset or strided fixed-length sampling if data coverage becomes the bottleneck
- token shift, macaron layout, and ALiBi or relative-bias family as low-parameter architecture alternatives
- Compile on/off timing on the actual target machine

## Helpful Repos Review

### 2026-03-19

High-signal references from `helpful_repos/`:
- `x-transformers`: strong source for compact-transformer ideas and tokenization experiments
- `model-stack`: useful for quantization helpers, loss variants, and sequence packing utilities
- `distill-kit`: simple teacher-student distillation reference for fixed-size students
- `minbpe`: minimal tokenizer experimentation reference if we decide to revisit tokenizer design

Most promising upgrade candidates from review:
- Try more aggressive KV sharing such as `NUM_KV_HEADS=1` to test MQA-style size-quality tradeoffs
- Tensor sensitivity mapping plus heterogeneous precision allocation looks like the best tensor-math fit for the competition because it directly optimizes byte spend against post-quant `val_bpb`
- Compression-aware regularization and grid alignment look like the second-best tensor-math lane because they shape the export path during training instead of after the fact
- Low-rank residual patches on a few sensitive tensors look plausible, but only after the sensitivity scorer exists and only as a second-wave experiment
- Spectrum-aware nonuniform layer allocation looks more promising than full tensor factorization because it reallocates bytes instead of paying heavy runtime overhead
- Consider adding optional z-loss regularization during training to improve logit stability
- Consider EMA shadow weights for final export or eval on short-budget runs
- Consider mild label smoothing and random-offset or strided fixed-length sampling as cheap trainer-side axes
- Consider token shift, macaron layout, and ALiBi or bucketed relative bias as low-parameter architecture lanes
- Consider packed or ragged sequence batching to reduce wasted padding and buy more useful tokens inside the wall-clock budget
- Consider a distillation path only if we can afford the extra training complexity and teacher cost
- Add a regex-aware byte-BPE branch to the tokenizer sweep instead of treating all BPE variants as one bucket
- Add experiment-ledger tooling so proxy/full-eval comparisons are easier to trust and weaker branches get pruned sooner
- Expand tokenizer sweeps to include `unigram` vs `bpe`, normalization family, `byte_fallback`, `split_digits`, and tokenizer-train-doc count
- Expand export work to include outlier-aware int8 and compressed scale-stat heuristics instead of only simple float-keep rules

Lower-priority or high-risk ideas:
- Entropy-based adaptive tokenization from `x-transformers` looks interesting but would require a much larger data and evaluation pipeline change
- Full TT/Tucker/Kronecker-style main-path tensor factorization still looks like the wrong first tensor-math bet here because code size, runtime overhead, and optimization stability all work against it
- MoE or multi-path FFN ideas are likely poor fits for a strict `16,000,000`-byte artifact cap
- Large architectural departures like latent-KV attention or hyper-connections are likely too risky before we validate simpler wins

Expanded review after the wider `helpful_repos/` refresh:
- Highest-value standalone ideas now look like:
  - RetNet-style retention in only the upper `3-4` blocks, borrowing from `torchscale`
  - Tiny sparse MLP experts on a subset of layers, borrowing routing patterns from `ModuleFormer`, `simple-moe`, and `distill-kit`
  - Product-key memory only in the final `1-2` layers, borrowing from `llm_memory_modules_at_scale`
  - XPos or related RoPE scaling from `torchscale` as a lower-risk context extension companion
- Repos that still look directly useful:
  - `torchscale`: retention, XPos, and dilated attention are the strongest architecture references
  - `x-transformers`: still a strong source for compact-transformer stabilization ideas
  - `bitsandbytes`, `QLoRA`: useful mainly for quantization heuristics and selective float preservation, not for importing a training path
  - `tokenizers`, `sentencepiece`, `minbpe`: still the best tokenizer-design references
  - `distill-kit`: useful as a compact reference for tiny MoE, PKM, and KD wiring patterns
- Repos that are probably not worth direct modeling time for Parameter Golf:
  - `ConfidenceTransformer`, `TOLBERT`, `Digital-World-Model`, `Program_Conditioned_Adapter`, `minimal-agent-kernel`, `q-transformer`, `selfplay-arena`
  - `faiss`: better as offline analysis or retrieval infra than as a core model idea here
  - `genetic-algorithm-pytorch`: maybe useful later for outer-loop hyperparameter search, not as a model change
- Why `ConfidenceTransformer` still ranks low:
  - confidence or entropy heads mainly help uncertainty estimation, OOD detection, or selective fallback behavior
  - this competition scores post-quant FineWeb compression on a fixed validation set, so extra confidence machinery has no obvious direct path to better `val_bpb`
  - the only plausible revisit would be a tiny entropy-aware eval policy, not a full auxiliary confidence head
- Deeper useful details from later repo review:
  - `bitsandbytes`: blockwise/nested quantization statistics and outlier-thresholded int8 are the highest-value extra quantization references
  - `sentencepiece` + `tokenizers`: normalization family, `byte_fallback`, `split_digits`, and regex-aware byte segmentation are real tokenizer design knobs
  - `torchscale`: if we implement retention, keep the gated retained-value path and per-head normalization structure in the first cut
  - `blockwise-parallel-transformer`: the reusable part is not the full JAX stack, but chunked attention and chunked cross-entropy as long-context systems knobs; `q_chunk_size` and `k_chunk_size` should be treated as tunable memory-throughput controls if `TRAIN_SEQ_LEN=2048+` starts hitting limits, and the repo reinforces pairing that with aggressive rematerialization plus optional float32 logits for stability
  - `hyper-connections`: the useful takeaway is a very small variant only, namely `2` residual streams around repeated/shared blocks with RMS-normalized dynamic alpha/beta mixing initialized near identity; full multi-stream trunks or frac-connections look too expensive for the record path
  - `lookahead-keys-attention`: its current shape uses six projections (`qu/ku/vu/qc/kc/vc`) plus an extra cached summary state `U`, so it is more byte- and cache-heavy than a normal attention block; if revisited at all, it should be an upper-layer-only or eval-oriented experiment rather than a near-term baseline candidate

Creative next-step order from repo review:
1. Retention upper layers + XPos
2. Sparse MLP experts on a subset of layers
3. Product-key memory in late layers
4. Minimal hyper-connections only if recurrence comes back onto the critical path

Competition-grounded ranking after checking the challenge rules and public wins:
1. Longer-context work remains the best training-side direction because `LongContextSeq2048` is already a proven record-level win and still fits the standalone artifact rules.
2. More aggressive KV sharing is the cleanest byte-budget lever because the challenge counts compressed model bytes and code bytes together.
3. Post-quant heuristics remain first-order because the submitted metric is the post-roundtrip score, not the pre-quant score.
4. Retention in only the upper layers is the best new architecture bet from `helpful_repos`, but it still ranks below simpler byte-budget and context levers.
5. Learned memory/register tokens rank above PKM and MoE for the record path because they are cheaper in code, bytes, and tuning complexity.
6. Tiny sparse MLP experts and PKM look better as non-record creative explorations than as the immediate record-track next step.

Practical split after grounding against the rules:
- Record-track next order:
  - longer context plus XPos
  - `NUM_KV_HEADS=1/2`
  - post-quant float-keep and export heuristics
  - retention in upper layers
  - memory/register tokens
  - regex-aware byte-BPE variants once trainer-side baselines stabilize
- Non-record creative order:
  - retention in upper layers
  - PKM late layers
  - sparse MLP experts
  - hyper-connected recurrence
  - bounded teacher-student distillation

Additional future implementation items from the newer repo review:
- regex-aware byte-BPE tokenizer sweep
- minimal teacher-student distillation lane with optional KL-on-logits
- experiment ledger and pairwise run-ranking tooling inspired by `selfplay-arena`
- tokenizer sweeps over `unigram/bpe`, normalization, `byte_fallback`, `split_digits`, and tokenizer-train-doc count
- outlier-aware int8 export and compressed scale-stat heuristics
- retention implementation detail: preserve the gate-and-norm structure instead of over-simplifying the block
- a separate family of "paid-in-artifact" ideas: tiny learned memory banks, compressed continuation tables, rare-token exception tables, or retrieval sketches that carry training-derived information only if those bytes fit inside the `<16MB` artifact
- local multi-GPU runs on small cards are useful for feasibility and crash-finding, but not as trustworthy ranking proxies for the `8xH100` competition path
- long-context systems lane: chunked attention and chunked cross-entropy controls inspired by `blockwise-parallel-transformer`, especially `q_chunk_size/k_chunk_size` style knobs if `2048+` context becomes memory-bound
- minimal hyper-connections shape: only `2` streams, no frac-connections, and only around repeated/shared blocks with identity-biased residual mixing
- explicit caution on lookahead-keys attention: six projections plus extra cache state make it a poor first-line fit for a strict artifact budget

## Update Rules

When we make a change, update this file with:

1. the code change
2. the hypothesis
3. the command or config used
4. the result
5. the judgment: `validated`, `neutral`, or `regression`

If a run fails for non-ML reasons, record that too. Those failures still save time later.

## New `60s` Commands

Row-norm regularization:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=row_norm_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
ROW_NORM_LOSS_COEFF=0.003 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

Outlier regularization:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=outlier_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
OUTLIER_LOSS_COEFF=0.001 \
OUTLIER_LOSS_THRESHOLD=3.0 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

Logit-variance smoothing:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=logit_var_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
LOGIT_VAR_LOSS_COEFF=0.001 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

Retention-style upper layers:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=retention3_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
RETENTION_LAYERS=3 \
RETENTION_DECAY_INIT=1.5 \
RETENTION_OUTPUT_GATE=1 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

Sparse MoE subset:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=moe_every3_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
MOE_EVERY_N_LAYERS=3 \
MOE_NUM_EXPERTS=4 \
MOE_TOP_K=1 \
MOE_HIDDEN=384 \
MOE_AUX_LOSS_COEFF=0.01 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```

Shared-tail hyper-connections:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_ID=hyper_shared_60s_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=60 \
VAL_MAX_SEQS=64 \
FINAL_EVAL_MAX_SEQS=64 \
NUM_SHARED_LAYERS=1 \
SHARED_LAYER_REPEATS=2 \
USE_HYPER_CONNECTIONS=1 \
HYPER_CONNECTION_LAYERS=1 \
HYPER_CONNECTION_GATE_INIT=-2.0 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=2 train_gpt_upgrade.py
```
