# State Space Causal Machine

Submission scaffold for the state-space variant built on top of the local-objective trainer.

## Recipe

1. Sliding-window validation / final evaluation
2. Shared recurrent tail with explicit diagnostics
3. Training-only mid auxiliary objective with ramp + decay
4. Stronger bigram defaults
5. Lexical shortcut controlled by soft `attn_gain_share` gating
6. Causal-machine state bias loaded from dataset profile v20
7. Dedicated `state` objective bucket and counterfactual ablation
8. Experimental packed-train batching removed from the submission copy

## Files

- `train_gpt.py`: exact trainer for the submission run
- `train.log`: copy the full run log here after the record run completes
- `submission.json`: fill in final metrics after the run completes

## Notes

- Source trainer copied from `2026-03-24_LocalObjective_SharedTail_SoftLexicalGate/train_gpt.py`
- Default causal-machine profile:
  - `/data/parametergolf/parameter-golf/runs/dataset_profile_80shards_v20.json`
- Default state-space settings:
  - `USE_CAUSAL_MACHINE_BIAS=1`
  - `CAUSAL_MACHINE_HIDDEN_RANK=128`
  - `OBJECTIVE_COUNTERFACTUAL_GROUPS=lexical,state`
- Train-side packed batching was intentionally removed here.
  - The prior implementation was boundary masking on a fixed grid, not true document packing, so it was left out of the competition script and should only be revisited experimentally.
- Current state-space trainer hash:
  - `1eb0da9b08a8f66cfdb1cfc3bb1990c228341f6f83585bd308664c25f3b88fdd`
- Recommended launch pattern:
  - `USE_CUDA_GRAPHS=0 CAUSAL_MACHINE_PROFILE_JSON=/data/parametergolf/parameter-golf/runs/dataset_profile_80shards_v20.json torchrun --standalone --nproc_per_node=2 train_gpt.py`
