# State Space Causal Machine

Submission record for the state-space/local-objective trainer in this tree.

## Submission Status

This record now assumes the current CUDA-backed runtime, not the older Python-only scaffold:

- fused structured scan CUDA extension for the main causal-machine path
- fused latent scan CUDA extension for the latent recurrence path
- Muon CUDA optimizer extension for grouped matrix buckets
- self-contained causal-machine submission path

The trainer defaults are already aligned to the intended submission posture:

- `COMPETITION_MODE=1` by default
- `TRAINING_PRESET=local_proxy_promotion`
- `OPTIMIZER_POLICY=local_proxy_faststart`
- `RUNTIME_POLICY=compiled`
- `ENABLE_TORCH_COMPILE=1`

## Competition Contract

- `CAUSAL_MACHINE_PROFILE_JSON` must be left unset or empty for submission runs.
- `CAUSAL_MACHINE_ALLOW_OFFLINE_TEACHER` should remain `0`.
- competition mode requires prebuilt, loadable CUDA extensions
- competition mode also expects the fused CUDA structured-scan path to be available

If a timed run starts without fresh prebuilt extensions, the trainer will fail instead of compiling them in-budget.

## Prebuild

Prebuild all required CUDA extensions before the timed run:

```bash
cd /data/parametergolf/peytontolbert-parameter-golf/records/track_10min_16mb/2026-03-26_StateSpace_CausalMachine

/home/peyton/miniconda3/envs/ai/bin/python prebuild_cuda_extensions.py
```

That script builds:

- `causal_machine_scan_cuda`
- `causal_machine_latent_scan_cuda`
- `muon_cuda`

## Runtime Notes

- default data/tokenizer paths in `train_gpt.py` are relative:
  - `./data/datasets/fineweb10B_sp1024`
  - `./data/tokenizers/fineweb_1024_bpe.model`
- when launching from this record directory, pass explicit absolute `DATA_PATH` and `TOKENIZER_PATH`
- the structured scan packed dtype defaults to `int8`
- accepted packed dtypes are:
  - `int8`
  - `fp8_e4m3`
  - `fp8_e5m2`
- `USE_CAUSAL_MACHINE_CUDA_SCAN=1`, `USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN=1`, and `USE_MUON_CUDA=1` are the active defaults

Kernel/backend inspection hooks:

- block SSM path: `StateSpaceBlock.attn.last_kernel_info`
- global causal-machine path: `GPT.last_causal_machine_kernel_info`
- paged recurrent-cache writes: `CausalMachineCache.last_paged_write_backend`
- Muon optimizer step routing: `Muon.last_step_stats`

## Recommended Launches

### 2-GPU local smoke

```bash
cd /data/parametergolf/peytontolbert-parameter-golf/records/track_10min_16mb/2026-03-26_StateSpace_CausalMachine

CUDA_VISIBLE_DEVICES=0,1 \
RUN_ID=causal_ssm_smoke_2gpu \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
USE_CUDA_GRAPHS=0 \
MAX_WALLCLOCK_SECONDS=90 \
ITERATIONS=40 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=20 \
VAL_MAX_SEQS=16 \
FINAL_EVAL_MAX_SEQS=16 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

### Competition-shaped launch

Use explicit paths and keep the submission self-contained:

```bash
cd /data/parametergolf/peytontolbert-parameter-golf

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUN_ID=causal_ssm_1280 \
DATA_PATH=/data/parametergolf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parametergolf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=10 \
MODEL_DIM=640 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_HIDDEN=1280 \
ITERATIONS=1000000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=1 \
VAL_LOSS_EVERY=0 \
VAL_MAX_SEQS=0 \
FINAL_EVAL_MAX_SEQS=32 \
TRAIN_LOG_EVERY=50 \
USE_OUTPUT_LOGIT_BIAS=0 \
USE_CAUSAL_MACHINE_BACKBONE=1 \
USE_CAUSAL_MACHINE_OUTPUT_BIAS=0 \
BLOCK_PATTERN=attn,attn,attn,attn,ssm,ssm,ssm,ssm,attn,attn \
CAUSAL_MACHINE_NUM_STATES=128 \
CAUSAL_MACHINE_HIDDEN_RANK=64 \
CAUSAL_MACHINE_LATENT_MODE=replace \
CAUSAL_MACHINE_LATENT_RANK=16 \
CAUSAL_MACHINE_LATENT_DECAY_MIN=0.995 \
CAUSAL_MACHINE_LATENT_DECAY_MAX=0.9999 \
CAUSAL_MACHINE_TRANSITION_GATE_MIN=0.125 \
CAUSAL_MACHINE_LATENT_OUTPUT_GATE_MIN=0.125 \
CAUSAL_MACHINE_TEACHER_LOSS_COEFF=0.0 \
CAUSAL_MACHINE_STATE_LOSS_COEFF=0.0 \
CAUSAL_MACHINE_NEXT_TOKEN_LOSS_COEFF=0.0 \
CAUSAL_MACHINE_TRANSITION_KL_COEFF=0.0 \
CAUSAL_MACHINE_FUTURE_SKETCH_LOSS_COEFF=0.0 \
USE_CAUSAL_MACHINE_CUDA_SCAN=1 \
USE_CAUSAL_MACHINE_LATENT_CUDA_SCAN=1 \
USE_MUON_CUDA=1 \
SAVE_BEST_VAL_BPB_CHECKPOINT=0 \
REGRESSION_STOP_PATIENCE=0 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-26_StateSpace_CausalMachine/train_gpt.py
```

Optional packed scan setting:

```bash
CAUSAL_MACHINE_SCAN_PACKED_DTYPE=fp8_e4m3
```

Leave that unset to stay on the default `int8` packed-table mode.

## Files

- submission runtime surface:
  - `train_gpt.py`
  - `cuda_ext/`
  - `prebuild_cuda_extensions.py`
- record artifacts:
  - `train.log`
  - `submission.json`

Local-only benchmark and test helpers should stay out of the submission payload.
