# Methodology Backlog

This file tracks candidate methodologies for improving `train_gpt_upgrade.py`.

It is intentionally separate from `UPGRADE_LOG.md`:
- `UPGRADE_LOG.md` records changes already made and run results.
- `METHODOLOGY_BACKLOG.md` records ideas before implementation so we can apply them systematically.

Important:
- Do not import code from `helpful_repos`.
- Use external repos only as inspiration for methodologies, design patterns, and experiment directions.
- When an idea is implemented, move the implementation details and result into `UPGRADE_LOG.md` as well.

## Goal

Build a structured queue of model, tokenizer, evaluation, and compression ideas that can be:
- applied one at a time
- measured cleanly
- confirmed, kept neutral, or rejected

Primary objective:
- improve `final_int8_zlib_roundtrip_exact val_bpb`

Hard constraints:
- total submission bytes must stay under `16,000,000`
- the run must remain reproducible
- leaderboard-path runs must fit the challenge time limits

## Status Legend

- `pending`: not implemented yet
- `ready`: well-scoped enough to implement next
- `implemented`: code landed, not yet judged
- `validated`: tested and helpful
- `neutral`: tested, no clear gain
- `regression`: tested and harmful
- `defer`: possible, but not worth current time or risk

## Evaluation Rules

Before calling something `validated`, check:
- `val_bpb` improved in the post-quant final metric we actually submit
- total bytes stayed under the artifact limit
- eval or train time did not become impractical for the target track
- the gain still holds against a clean nearby baseline

Current iteration policy:
- use `60s` runs as the default inner-loop comparison budget
- treat the original `train_gpt.py` as the true baseline
- treat `train_gpt_upgrade.py` with no extra knobs as the upgrade-branch baseline
- define a "60s run" as `60s` of training plus capped proxy eval, not full uncapped final eval
- rerun a matching `60s` baseline whenever the comparison frame changes
- only scale promising ideas back up to `600s` once the short-run evidence is directionally clean
- keep `RUN_ID` unique so each short experiment preserves its own `runs/<RUN_ID>/` history

Default result fields to record for each experiment:
- command or config
- train time
- eval time
- final artifact bytes
- pre-quant `val_bpb`
- post-quant `val_bpb`
- judgment

## Priority Tiers

### Tier 1: Highest Expected Value

These ideas are the best fit for Parameter Golf and the current script.

| ID | Area | Methodology | Inspiration | Why It Fits | Implementation Shape In `train_gpt_upgrade.py` | Main Risks | Status |
|---|---|---|---|---|---|---|---|
| T1 | Tokenizer | Joint tokenizer-model sweep | `sentencepiece`, `tokenizers`, `minbpe` | Tokenization changes both compression and parameter allocation; likely one of the biggest available levers | Add documented experiment presets for `unigram` vs `bpe`, vocab sizes like `512/768/1024/1536`, normalization variants such as `nmt_nfkc/nfkc/identity`, `byte_fallback` on/off, `split_digits` on/off, tokenizer-train-doc sweeps, and regex-aware byte-level BPE in data prep | Easy to waste time on invalid tokenizer/data pairings; must preserve correct BPB accounting | ready |
| T2 | Depth efficiency | Shared-block recurrence | `TinyRecursiveModels`, `x-transformers` | Reusing a small set of blocks can buy more effective depth without paying full parameter cost | Add looped/shared blocks or `num_unique_layers < num_total_steps`; optionally per-loop learned scales | Can hurt optimization or reduce diversity too much | ready |
| T3 | Attention efficiency | More aggressive KV sharing | `x-transformers` | GQA is already present; pushing to `1-2` KV heads may free useful bytes with modest quality loss | Sweep `NUM_KV_HEADS=1,2,4`; consider making lower KV counts the default search axis | Too much sharing may hurt quality more than bytes saved | ready |
| T4 | Working memory | Learned memory/register tokens | `x-transformers` | Small models often benefit from persistent scratch space; parameter cost is small | First cut now implemented as `NUM_REGISTER_TOKENS`, which prepends a small learned set of tokens through the stack and strips them before logits so metric accounting stays on the original token stream | Can slow attention and complicate eval byte accounting if overused | implemented |
| T5 | Stability | QK norm / RMSNorm cleanup | `x-transformers` | Cheap stability improvement, low code complexity, likely helpful in small fast runs | QK RMS normalization is already present; remaining work is only if a further attention-stability knob becomes necessary | Might do little if current setup is already stable | implemented |
| T6 | Compression | Tensor sensitivity mapping and heterogeneous precision allocation | public runs, `bitsandbytes`, `QLoRA` | Final metric depends on post-quant model; the cleanest byte-for-quality trade is often to keep only the most sensitive tensors, channels, or row groups at higher precision | First cut implemented with whole-tensor auto keep, and now extended to row-group passthrough under `INT8_AUTO_KEEP_ROW_BUDGET_BYTES` and `INT8_AUTO_KEEP_ROW_GROUP_SIZE` | Easy to overfit to local proxy eval or create a complicated export policy that is hard to reproduce cleanly | implemented |
| T7 | Compression-aware training | Regularization and quantization-grid alignment | `model-stack`, `QLoRA`, local research notes | Shaping the weight and logit distributions during training may reduce the post-export hit more efficiently than a fancier runtime architecture | Implemented first cut with z-loss, row-norm penalties, outlier penalties, brief fake-quant tail finetuning, and logit-variance smoothing, all off by default and sweepable one at a time | Can cost too much wall-clock or improve proxy metrics without helping the final roundtrip score | implemented |
| T9 | Optimizer | Decoupled Muon weight decay | current top recipe | Directly matches the current leading configuration and is cheap to expose as a standalone knob | Implemented as `MUON_WEIGHT_DECAY`; first target value is `0.02` | Too much decay can slow early learning in short runs | implemented |
| T10 | Initialization | Overtone spectral embedding init plus phase-transition `resid_mix` init | current top recipe | Directly matches the current leading configuration with minimal code footprint and no dependency on external repos | Implemented as `OVERTONE_EMBED_INIT`, `OVERTONE_EMBED_POWER`, `OVERTONE_EMBED_SCALE`, `RESID_MIX_PHASE_INIT`, and `RESID_MIX_PHASE_SHARPNESS` | Can change short-run optimization noticeably, so it needs a clean comparator | implemented |
| T8 | Data coverage | Deterministic random-offset shard sampling | local trainer work | Short runs can overfit the first contiguous shard regions they see; bounded random skips should improve context diversity without changing the dataset format | Now implemented as `TRAIN_RANDOM_OFFSET_TOKENS`, which lets each batch start up to a bounded number of tokens ahead within the current shard while remaining deterministic under the seed | Too much skipping may hurt coverage or make comparisons noisy if the offset is too large | implemented |

### Tier 2: Strong But More Conditional

These are promising, but they need tighter scoping or depend on other decisions first.

| ID | Area | Methodology | Inspiration | Why It Might Help | Implementation Shape In `train_gpt_upgrade.py` | Main Risks | Status |
|---|---|---|---|---|---|---|---|
| C1 | Context | Longer train/eval context | in-repo `LongContextSeq2048`, `x-transformers`, `torchscale` | Already validated in the repo leaderboard; may combine well with stronger small-model design | Added `TRAINING_PRESET=long_context_2048`, `long_context_2048_standard_eval`, and `long_context_2048_sliding_eval` to make `TRAIN_SEQ_LEN=2048` and related eval sizing first-class while still allowing explicit env overrides | Throughput loss may offset quality gains | implemented |
| C2 | Eval | Sliding-window as a tunable family, not one default | in-repo `SlidingWindowEval` | Already proved leaderboard impact; still needs stride tuning vs eval-time budget | Sweep `EVAL_STRIDE` and `EVAL_BATCH_SEQS`; compare final-only sliding eval vs wider use | Too slow or over-optimized for eval-only gain | implemented |
| C3 | Residual path | Lightweight loop gating or residual scaling | `TinyRecursiveModels`, `hyper-connections`, `x-transformers` | Could stabilize repeated shared blocks and improve effective depth | First cut implemented as shared-tail recurrence with `NUM_SHARED_LAYERS`, `SHARED_LAYER_REPEATS`, and learned sigmoid loop gates | Too many knobs, limited gain | implemented |
| C4 | Positional scheme | XPos or alternative RoPE scaling | `x-transformers`, `torchscale` | May help if we extend context without paying for learned position params | First cut implemented as `USE_XPOS` and `XPOS_SCALE_BASE` on top of the current rotary path | Gains may be too small to justify complexity | implemented |
| C5 | Attention biasing | Residual attention | `x-transformers` | Parameter-cheap way to improve information flow across layers | First cut implemented as `USE_RESIDUAL_ATTENTION`, carrying the previous layer's attention logits forward with a learned gate | Training can become finicky; may need LR retuning | implemented |
| C6 | Memory | Learned memory KV slots | `x-transformers` persistent memory | Lower-overhead alternative to full memory modules | First cut implemented as `MEMORY_KV_SLOTS`, adding a small learned K/V bank inside attention | Might overlap with memory tokens and not justify added logic | implemented |
| C7 | Distillation | Bounded teacher-student distillation lane | `distill-kit` | May improve a fixed-size student without changing inference architecture if we keep the implementation minimal | First cut implemented as a local-only `DISTILL_*` lane with teacher-logit KL on top of CE; distributed distillation remains intentionally disabled until a proper DDP-safe path is added | Extra training complexity and teacher cost may not be worth it for record-track work | implemented |
| C8 | Short-run stability | Z-loss, EMA export, label smoothing, and fake-quant tail finetune | `model-stack` | These are cheap training-side knobs that may improve short-budget training and final export quality without a big architectural rewrite | Z-loss, EMA, and label smoothing were already present; first cut fake-quant tail is now implemented as `FAKE_QUANT_TAIL_STEPS` and `FAKE_QUANT_BITS` | Easy to add too many overlapping knobs and lose attribution | implemented |
| C9 | Data coverage | Random-offset or strided fixed-length window sampling | `model-stack` data loaders | Can increase effective document coverage per wall-clock even when `TRAIN_SEQ_LEN` stays fixed | Implemented as deterministic random-offset sampling via `TRAIN_RANDOM_OFFSET_TOKENS`, which keeps reproducibility and token accounting simple while broadening local shard coverage | Can complicate loader logic or make comparisons noisier if sampling is not controlled | implemented |
| C10 | Cheap inductive bias | Token shift, macaron layout, and ALiBi or relative bias family | `x-transformers`, `torchscale` | These are low-parameter alternatives to a full architecture replacement and may help tiny models more than another width tweak | First cut now implemented as `TOKEN_SHIFT`, `USE_ALIBI`, `USE_RELATIVE_BIAS`, and `MACARON_LAYOUT` | Many small knobs can sprawl and each may give only a modest gain | implemented |

### Tier 3: Interesting, But High Risk

These ideas are real, but they are more speculative for this challenge and script.

| ID | Area | Methodology | Inspiration | Why It Is Interesting | Why It Is Risky Here | Status |
|---|---|---|---|---|---|---|
| H1 | Sparse capacity | Tiny MoE or expert FFN | `ModuleFormer`, `simple-moe`, `torchscale` X-MoE | Can increase capacity per token-compute | First cut now exists as `MOE_EVERY_N_LAYERS`, `MOE_NUM_EXPERTS`, `MOE_TOP_K`, `MOE_HIDDEN`, and `MOE_AUX_LOSS_COEFF`; still needs experimental judgment | implemented |
| H2 | External memory | Product-key memory or memory-augmented layers | `llm_memory_modules_at_scale` | Might add capacity without dense FFN growth | First cut was implemented and then removed from the active script after a smoke test showed it pushed params too far for the current competition path; revisit later only in a much smaller form | defer |
| H3 | New attention family | Lookahead-keys attention | `lookahead-keys-attention` | Novel causal attention idea with possible quality/context benefits | Very new, unproven here, high implementation risk | defer |
| H4 | Multi-stream residuals | Full hyper-connections | `hyper-connections` | Could improve depth utilization with limited extra params | First cut now exists as `USE_HYPER_CONNECTIONS`, `HYPER_CONNECTION_LAYERS`, and `HYPER_CONNECTION_GATE_INIT` around the shared tail only; still needs experimental judgment | implemented |

### Tier 4: Likely Not Worth Current Time

These ideas are mostly out of scope for the current trainer.

| ID | Area | Methodology | Inspiration | Reason To Skip For Now | Status |
|---|---|---|---|---|---|
| L1 | Adapters | LoRA/PEFT-style training path | `peft`, `Program_Conditioned_Adapter`, `QLoRA` | Better for finetuning than for this from-scratch tiny LM competition | defer |
| L2 | Agentic memory systems | Repo/program/world-model adaptation | `Digital-World-Model`, `Program_Conditioned_Adapter` | Not aligned with FineWeb compression training | defer |
| L3 | Non-LM side repos | Miscellaneous unrelated repos in `helpful_repos` | various | No clear path to a better Parameter Golf model | defer |
| L4 | Confidence heads / entropy prediction | ConfidenceTransformer-style uncertainty modeling | `ConfidenceTransformer` | Confidence or OOD estimation may help selective decoding or analysis, but it does not directly target `val_bpb` on a fixed validation set | Extra heads, losses, and eval logic spend bytes and tuning budget without a clear path to better compression | defer |

## Creative Ideas

These are the best standalone architectural ideas from `helpful_repos/` that still fit the competition constraints.

They should be treated as self-contained implementation targets inside `train_gpt_upgrade.py` or a record-folder `train_gpt.py`, not as imported dependencies.

### Best Creative Targets

| ID | Idea | Source Repos | Minimal Standalone Spec | Why It Is Attractive | Main Risk | Status |
|---|---|---|---|---|---|---|
| X1 | Retention upper layers | `torchscale` | Replace only the top `3-4` attention blocks with a RetNet-style retention module; keep the rest standard. First cut now exists as `RETENTION_LAYERS`, `RETENTION_DECAY_INIT`, and `RETENTION_OUTPUT_GATE`, using exponentially decayed causal attention in the upper layers while preserving the rest of the stack | Best long-context candidate that may also reduce eval-time cost for richer-context scoring | New attention family may need LR and norm retuning | implemented |
| X2 | Sparse MLP experts on a subset of layers | `ModuleFormer`, `simple-moe`, `distill-kit` | Convert every `2nd` or `3rd` MLP block to a tiny MoE FFN with `num_experts=4`, `top_k=1`, linear router, light aux loss, and smaller per-expert hidden size such as `384-512`; first cut now exists as `MOE_*` | Buys token-conditional capacity without fully paying dense compute | Routing overhead and instability can erase the gain in a `10` minute run | implemented |
| X3 | Product-key memory in late layers | `llm_memory_modules_at_scale`, `distill-kit` | Add PKM only in the last `1-2` blocks with `num_heads=4`, `key_dim=32`, `value_dim=32 or 64`, `num_keys_per_head=64-128`, `top_k=4-8`, and shared keys if needed | Could add useful memory capacity at low parameter cost and compress well | First cut proved too large for the current competition path, so this stays deferred unless redesigned much smaller | defer |
| X4 | Dilated attention only in upper layers | `torchscale` | Keep normal attention in lower layers and use a small dilated pattern near the top, e.g. `segment_length=[256,1024]` and `dilated_ratio=[1,4]` when running `TRAIN_SEQ_LEN=2048` | A more conservative long-context alternative than replacing the whole stack | More implementation complexity than retention for unclear extra gain | pending |
| X5 | Minimal hyper-connections around repeated blocks | `hyper-connections` | Use `num_streams=2` only around shared or repeated blocks; start with no fractions and a simple learned residual mix; first cut now exists around the shared tail only via `USE_HYPER_CONNECTIONS` | Best way to retry recurrence with better gradient flow than plain looping | Too many knobs if we do the full paper instead of a tiny variant | implemented |

### Deprioritized Creative Ideas

- `lookahead-keys-attention`: novel, but too new and high-risk for the current budget.
- `blockwise-parallel-transformer`: more of a systems path for extreme context than a near-term win for a tiny standalone artifact.
- `ConfidenceTransformer`: confidence and OOD heads do not target the competition metric.
- `TOLBERT`, `tolbert-brain`, `Program_Conditioned_Adapter`, `Digital-World-Model`, `minimal-agent-kernel`, `selfplay-arena`, `q-transformer`: interesting for agents or structured reasoning, but not a good fit for FineWeb compression training.
- `genetic-algorithm-pytorch`: more useful for outer-loop search than for the model itself.

### Creative Implementation Order

1. Retention upper layers plus XPos.
2. Sparse MLP experts on a subset of layers.
3. Product-key memory in late layers.
4. Minimal hyper-connections only if recurrence returns to the queue.

## Competition-Grounded Ranking

This ranking is grounded in the actual challenge rules and the current winning submissions.

The main ordering criteria are:
- optimize the submitted metric, which is post-quant `final_int8_zlib_roundtrip_exact val_bpb`, not just pre-quant loss
- stay under the `16,000,000` byte total artifact cap, including code bytes
- preserve a standalone record-folder submission with no dependency on `helpful_repos`
- fit both the training and evaluation wall-clock budgets on the target `8xH100` setup
- prefer ideas that are already adjacent to proven wins in the public leaderboard

### Ranked Next Items For Record-Track Work

| Rank | Item | Why It Ranks Here | Rule Pressure | Status |
|---|---|---|---|---|
| R1 | Longer context plus XPos-style positional scaling | `LongContextSeq2048` is already the strongest training-side public win, and XPos is a low-overhead companion for pushing context further without learned-position growth | Must keep throughput acceptable within `600s`; code change is small and standalone-friendly | ready |
| R2 | More aggressive KV sharing such as `NUM_KV_HEADS=1` or `2` | Directly trades parameter bytes for either quality-preserving compression headroom or more useful capacity elsewhere; especially attractive under a hard artifact cap | Very low code risk, but must verify quality does not fall faster than bytes improve | ready |
| R3 | Better post-quant heuristics and float-keep rules | The competition is scored after quantization and compression, so export policy is first-order, not polish; this is also already supported by in-repo evidence from float-preserved embeddings | Must watch both score and bytes together; easy to overfit to size only | ready |
| R4 | Retention only in upper layers | Strongest new architectural idea from `helpful_repos` that still matches the challenge shape; may preserve context benefits while controlling eval cost | Bigger training-risk than KV or export sweeps; still must fit as one standalone script | pending |
| R5 | Learned memory/register tokens | Cheap way to give a tiny model scratch space with small code and parameter overhead | Adds attention cost and may complicate eval-time behavior if overused | pending |
| R6 | Tiny sparse MLP experts on a subset of layers | Potentially adds token-conditional capacity without paying dense compute everywhere | Routing overhead, aux loss, and code size make this weaker for strict record-track work | pending |
| R7 | Product-key memory in late layers | Could buy compressed capacity in a targeted way | More moving parts and tuning burden than register tokens; weaker first record-track bet | pending |
| R8 | Minimal hyper-connections around repeated blocks | Mostly useful as a stabilizer if recurrence becomes the main architecture path again | Extra complexity without a direct leaderboard-proven analogue yet | pending |

### Ranked Next Items For Non-Record Creative Work

| Rank | Item | Why It Fits The Non-Record Track Better |
|---|---|---|
| N1 | Retention upper layers plus XPos | Most interesting architecture departure with a plausible path back to record-track usefulness |
| N2 | Product-key memory in late layers | Creative and self-contained, but easier to justify when record-track risk tolerance is lower |
| N3 | Sparse MLP experts on a subset of layers | High upside but more tuning-heavy, so better explored outside the immediate record path |
| N4 | Minimal hyper-connections with shared blocks | Best used when exploring recurrence more aggressively than the record path currently allows |

### What To Avoid Ranking Too High

- Full tokenizer redesign as the immediate next step: high upside, but the rules explicitly make tokenizer changes harder to validate correctly.
- Full MoE as an early bet: too much risk against the `16,000,000` byte cap and `600s` train budget.
- Very new attention families such as lookahead-keys: too much novelty risk before the stronger, better-grounded paths are exhausted.
- Confidence or entropy prediction heads: useful for uncertainty estimation, but weakly connected to the submitted compression metric unless they unlock a very specific eval-time policy.
- Full TT/Tucker/Kronecker-style main-path tensor factorization: likely the wrong first tensor-math bet here because code size, runtime overhead, and optimization risk all rise before we finish the much simpler post-quant allocation path.
- Low-rank residual patches on quantized tensors as the immediate next step: interesting and in-bounds, but better treated as a second-wave experiment after we have tensor sensitivity mapping and compression-aware training in place.

Tokenizer-specific caution:
- the tokenizer file itself is relatively small compared with model bytes, but tokenizer changes still require full retokenized exports and more careful metric validation than model-only changes.

## Paid-In-Artifact Training Information

Competition rule reminder:
- evaluation cannot access training data for free
- but we are allowed to carry training-derived information into evaluation if those bytes are stored inside the artifact and the total submission stays under `16,000,000`

So the real design question is:
- what training-derived information is worth paying for with artifact bytes
- and does it beat spending those same bytes on the base model, context, or float-preserved tensors

### Byte-Budget Ranking

| Budget | Most Plausible Ideas | Why They Fit | Main Risk | Status |
|---|---|---|---|---|
| `<=256KB` | Rare-token exception tables, tiny continuation backoff tables, hashed context heuristics | Cheap enough to test without disrupting the model budget much | May be too weak to move final `val_bpb` | pending |
| `~1MB` | Small learned memory bank, compressed n-gram stats, tiny phrase or continuation cache | Big enough to hold meaningful distilled corpus structure while staying model-heavy | Can lose to simply spending the same bytes on model weights | pending |
| `~4MB` | Product-key memory, retrieval sketch with compressed keys and values, richer continuation datastore | Large enough for a serious explicit memory path | Must clearly outperform using those bytes for a stronger dense model | pending |
| `~8MB` | Aggressive explicit datastore or exemplar cache | Can hold much richer training-derived information | Very likely to be dominated by spending the same bytes on the model unless retrieval is extremely efficient | defer |

### Best Paid-For Ideas

| ID | Idea | Byte Range | Why It Is Worth Remembering | First Test Shape | Status |
|---|---|---|---|---|---|
| P1 | Tiny learned memory bank | `256KB-1MB` | Closest to "store training information in the artifact" without a bulky explicit datastore | Compare a very small learned memory path against spending the same bytes on the base model | pending |
| P2 | Compressed n-gram or continuation cache | `256KB-1MB` | Simplest explicit corpus-summary idea with a clear byte budget | Start with top contexts or hashed byte or token continuations in a non-record lane | pending |
| P3 | Rare-token or rare-pattern exception table | `<=256KB` | Targets long-tail cases where tiny models may fail disproportionately | Build a tiny table only for especially costly tokens or patterns | pending |
| P4 | Product-key or hashed retrieval sketch | `1MB-4MB` | Most realistic way to carry more corpus-derived structure without raw text | Start with shared small keys and tiny values, not a full datastore | pending |

### What Is Probably Not Worth Paying For

- Raw or lightly compressed training text chunks
- Large nearest-neighbor index over the corpus
- Multi-megabyte exemplar stores unless they clearly beat the same bytes spent on model parameters

### Decision Rule

Before investing in a paid-for memory idea, ask:
1. would these bytes help more if spent on model weights, context length, or quantization policy?
2. does the mechanism remain fully standalone and reproducible inside the record folder?
3. can we explain clearly in a submission README exactly which bytes are paying for the training-derived information?

## Tensor-Math Priority Lane

This is the best way to channel tensor-heavy ideas into Parameter Golf without drifting into elegant-but-costly factorization work.

### Ranking

| Rank | Item | Why It Fits This Competition | Status |
|---|---|---|---|
| TM1 | Tensor sensitivity mapping and heterogeneous precision allocation | Directly targets the submitted post-quant metric by spending bytes only on the tensors, channels, or row groups that buy the most `val_bpb` back | ready |
| TM2 | Compression-aware regularization and quantization-grid alignment | Lets training shape the final export path instead of hoping post-hoc quantization works out | ready |
| TM3 | Tiny low-rank residual patches on a few sensitive tensors | Legitimate second-wave tensor idea once we know which tensors are sensitive enough to deserve a correction | pending |
| TM4 | Spectrum-aware nonuniform layer allocation | More realistic than full factorization because it just reallocates width, MLP, or precision budget toward the layers that seem to matter most | pending |
| TM5 | Full TT/Tucker/Kronecker-style factorization in the main path | The most mathematically elegant option, but currently the least attractive for the challenge because of runtime, code bytes, and stability risk | defer |

## Implementation Queue

Recommended order:
1. Exact `60s` rerun of the original `train_gpt.py` baseline
2. Matching `60s` rerun of the `train_gpt_upgrade.py` baseline with no extra knobs
3. `60s` single-change sweeps such as lower KV heads and float-keep rules
4. Tensor sensitivity mapping and heterogeneous precision allocation on the strongest `60s` candidates
5. Compression-aware regularization and grid-alignment sweeps
6. Longer-context presets once a local-fit short-run shape is chosen
7. Stability cleanups such as QK norm / RMSNorm adjustments
8. Shared-block recurrence
9. Memory/register tokens
10. Tokenizer-model sweep framework, including regex-aware byte-BPE variants
11. Bounded teacher-student distillation lane
12. Higher-risk sparse or memory modules
13. Experiment search infrastructure for automatic candidate ranking and ledgering

## One-Minute Experiment Loop

Use this as the default workflow until a candidate looks strong enough to justify a `600s` run.

### Baseline First

Always keep two current short-run anchors on the same dataset, tokenizer, and hardware class as the candidate runs:
- the true baseline from the original `train_gpt.py`
- the upgrade-branch baseline from `train_gpt_upgrade.py` with no extra knobs enabled

Canonical true `60s` baseline command with capped proxy eval:

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

Matching `60s` upgrade-branch baseline command with capped proxy eval:

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

Short-run comparison candidates should differ by one main variable at a time:

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

### What To Compare At `60s`

Record these fields for each short run:
- step reached at wallclock stop
- latest pre-quant `val_bpb` if any validation fired
- final `final_int8_zlib_roundtrip_exact val_bpb`
- total int8+zlib submission bytes
- peak memory
- eval time

Interpretation rule:
- treat `60s` runs as ranking proxies for nearby candidates, not as submission-quality evidence
- if a candidate loses clearly at `60s`, prune it
- if a candidate wins or stays close at `60s`, escalate it to a `600s` run
- when escalating, rerun without `FINAL_EVAL_MAX_SEQS` so the candidate gets a more trustworthy final read

## Search Infrastructure

These items are not model changes, but they can accelerate clean iteration and reduce experiment waste.

| ID | Area | Methodology | Inspiration | Why It Fits | Implementation Shape | Main Risks | Status |
|---|---|---|---|---|---|---|---|
| S1 | Experiment ops | Persistent experiment ledger and auto-ranking | `selfplay-arena` | Faster iteration only matters if rankings remain trustworthy; a structured ledger helps us compare runs cleanly under the real metric | Extend local run metadata so every experiment records config, artifact bytes, proxy eval, full eval, and judgment in a machine-readable table or JSONL | Easy to overbuild tooling before we have enough runs to justify it | pending |
| S2 | Experiment ops | Pairwise config comparisons and pruning | `selfplay-arena` | Helps stop weak branches early and reserve GPU time for better candidates | Add a lightweight script that compares candidate runs against the current baseline and marks dominated settings for pruning | Can bias us toward noisy short-run winners if proxy eval is not calibrated well | pending |

## Deeper Repo Notes

These are implementation-level details uncovered during the deeper `helpful_repos/` review that should shape future work.

- `bitsandbytes`: outlier-thresholded int8 export and nested/compressed quantization statistics are better matches for this challenge than generic 4-bit training paths.
- `sentencepiece`: normalization family, `byte_fallback`, `split_digits`, and tokenizer-train-doc count are real sweep axes, not incidental preprocessing knobs.
- `tokenizers`: regex-aware byte-level segmentation and digit policy are the strongest extra tokenizer-design references.
- `torchscale`: if retention is implemented, preserve the gated retained-value path and per-head norm structure in the first version.

## Parallel Checks

Use `upgrade_checks.py` while a main training job is already running. These checks are meant to be cheap and separable from full training:

- `smoke`: random forward or forward+backward sanity check for shape, loss, and obvious breakage
- `artifact`: quantize and compress the current architecture or checkpoint to estimate whether it still fits the artifact budget
- `eval-probe`: compare standard eval and sliding eval on a small subset to understand eval-time deltas without a full run
- `roundtrip`: measure quantize/dequantize reconstruction error, with optional small eval after roundtrip
- `throughput`: tiny train-step microbenchmark for step time and peak memory on a chosen device

Suggested usage pattern:
1. run `smoke` after any architecture edit
2. run `artifact` before starting expensive training
3. run `throughput` on the candidate device to estimate speed regressions
4. run `eval-probe` and `roundtrip --with-eval` on existing checkpoints while another long run is training

Important:
- local full runs on small GPUs are still worth doing for crash-finding, memory checks, and artifact validation
- but they should not be treated as high-confidence ranking proxies for the `8xH100` competition target when the achieved step count is dramatically lower than record-track runs

Example commands:

```bash
python parameter-golf/upgrade_checks.py smoke --device cpu
python parameter-golf/upgrade_checks.py artifact
python parameter-golf/upgrade_checks.py throughput --device cuda:1 --steps 10 --batch-size 2
python parameter-golf/upgrade_checks.py eval-probe --device cuda:1 --checkpoint final_model.pt --max-seqs 32 --max-windows 128
python parameter-golf/upgrade_checks.py roundtrip --device cuda:1 --checkpoint final_model.pt --with-eval --max-seqs 32
```

## Experiment Template

Copy this section when starting a new candidate:

```md
### Experiment: <ID> <short name>

Status:
- pending

Hypothesis:
- 

Implementation plan:
- 

Config or command:
- 

Acceptance criteria:
- improves final post-quant `val_bpb`
- remains under `16,000,000` bytes
- does not create unacceptable runtime cost

Result:
- train time:
- eval time:
- artifact bytes:
- pre-quant `val_bpb`:
- post-quant `val_bpb`:

Judgment:
- pending

Notes:
- 
```

## Structural Notes

Use this backlog to avoid mixing too many changes at once:
- first test one variable at a time when possible
- only combine ideas after each one has an individual read
- if two ideas interact strongly, record both the solo runs and the combined run

When an idea becomes code:
- add the implementation to `UPGRADE_LOG.md`
- record whether it was `validated`, `neutral`, or `regression`
- leave the backlog item here, but update its status and short note

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
