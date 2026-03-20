This submission stages the current upgraded trainer for a 10-minute `8xH100` leaderboard run.

The main changes relative to the naive baseline are:
- forced `fp16` export for `tok_emb.weight` to reduce post-quant degradation on the tied embedding / output head
- budget rebalance via `MLP_HIDDEN=992`
- longer warmdown and higher matrix LR following the proven `FP16Embed_WD3600` recipe
- optional cheap training-side stability knobs enabled in this script: `z-loss`, `label smoothing`, and EMA export

## Planned Command

```bash
RUN_ID=upgrade_fp16embed_wd3600 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_HIDDEN=992 \
WARMDOWN_ITERS=3600 \
MATRIX_LR=0.06 \
Z_LOSS_COEFF=1e-4 \
EMA_DECAY=0.9999 \
EMA_START_STEP=200 \
FORCE_FP16_TIED_EMBED_EXPORT=1 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Intended Files For This Record Folder

- `train_gpt.py`: standalone trainer snapshot used for the run
- `train.log`: exact H100 training log from the winning attempt
- `submission.json`: leaderboard metadata for the final run

## Status

This folder is prepared for the upcoming H100 run, but the final metrics and log have not been copied in yet.
