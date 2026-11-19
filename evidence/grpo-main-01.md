# 20-step GRPO and final paired evaluation

**PASS, with limits:** `post-01` operationally restores the native `train-01/global_step_20` checkpoint and completes the frozen 64-case evaluation. The existing `paired-01` comparison is available and independently reconciled: baseline **27/64**, post **35/64**, **11 wins / 3 losses / 50 ties**. These are outcomes under the unchanged answer extractor and reward, not an independent semantic grading of every answer.

The in-training final evaluation remains **34/64**. One case changes reward on reload: both runs calculate 6, but the training response quotes an answer-format placeholder that the first-match extractor selects instead of its later numeric answer. The scoring mechanism is explained below; the cause of the model's different wording is unverified. Neither result was discarded or rescored.

This closeout used existing Modal Volume artifacts and local offline analysis. No Modal action (including summarize/export), inference, training, checkpoint rewrite, infrastructure change, or tensor download was performed. The earlier [smoke evidence](grpo-smoke-01.md) and historical verification records remain intact.

## Acceptance gates

| Gate | Status | Evidence and scope |
|---|---|---|
| Canonical source and provenance | PASS | Current fetched `main` is `7cfddf58a0dece6a63ca67d18b103db2940e4aca`; base/train/post receipts record that clean commit; all 12 submission source-file hashes per run match local source. No pre-existing final-results PR was found. |
| Post checkpoint restoration | PASS, operational | Explicit step-20 source path, resumed global step 20, subsequent completed evaluation, and pinned source ordering of actor load and rollout refresh. Requested-path metadata alone is not the proof. |
| Post evaluation completeness | PASS | Exactly the frozen 64 IDs once each; 64 real, completed validation episodes with posted binary rewards, 64 successful event POSTs, and trainer receipt of all 64 rewards. |
| Post evaluation-only behavior | PASS | `val_only=true`; no training-route traces, actor-update metrics, or new checkpoint directory/files; result checkpoint list empty. |
| Existing paired summary | PASS | `comparison.available=true`, `paired_cases=64`; independent per-case recomputation and aggregate checks match the saved JSON. |
| Twenty training iterations | PASS | Exactly steps 1–20, 640 rewarded training attempts, complete requested metric records, finite numeric metrics, positive update/refresh stage times. |
| Nonzero signal every iteration | FAIL | Steps 2, 5, 6, 9, 11 and 15 have zero advantages and gradients; their groups have constant rewards. |
| Error-free tools / trace merging / dependency closure | FAIL | Tool errors, parser fallbacks, split trajectories and six reviewed metadata conflicts remain. |
| Bitwise replay, tensor identity, causal or significant learning | UNVERIFIED | No tensor comparison or repeated experimental controls; temperature zero does not establish bitwise reproducibility. |

## Sources and frozen inputs

All remote paths below are relative to Volume **`cs329z-agent-lightning`**, mounted at `/data`; the runtime resolves this to `/__modal/volumes/vo-lCOCqrVwdlYV46hXMoHHws`. For each run, inspect `submissions/NAME/receipt.json`, `submissions/NAME/source-sha256.json`, and `runs/NAME/{result,manifest,model_manifest,data_manifest,resolved_config,resolved_packages,observability,run_plan}.json`, `logs/trainer.log`, and `traces/trajectory-*.jsonl`.

| Run | Phase | W&B run ID | Receipt interval, 2026-11-13 UTC | Whole-job duration |
|---|---|---|---|---:|
| `base-01` | baseline-eval | [e04fc17e5116b9f0](https://wandb.ai/amulyap-stanford-university/cs329z-agent-lightning/runs/e04fc17e5116b9f0) | 06:02:54–06:10:09 | 435.446 s |
| `train-01` | main | [fec3f64b9a6be201](https://wandb.ai/amulyap-stanford-university/cs329z-agent-lightning/runs/fec3f64b9a6be201) | 06:22:02–06:56:21 | 2,058.977 s |
| `post-01` | reload-eval | [ccdc8a022413f917](https://wandb.ai/amulyap-stanford-university/cs329z-agent-lightning/runs/ccdc8a022413f917) | 14:51:05–14:56:22 | 316.494 s |

All three receipts and results say `completed`, exit 0. `paired-01` has a completed clean-source receipt at `submissions/paired-01/receipt.json`, 15:24:10–15:24:50 UTC, recording `gpu_used=false` and `evidence/paired-01/summary.json`. These existing jobs were not invoked during review.

| Identity | Value |
|---|---|
| Lab source | `7cfddf58a0dece6a63ca67d18b103db2940e4aca` |
| Agent Lightning | v1.0.3, `cbc56e41e0eab643aa71d195e92e5bba652fa683` |
| Model | `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Harness SHA256 | `e802a2324dab779ffaa49af678196a7cc5f02e3bb89936bd0ed1f1ed8c3a02fc` |
| Launcher SHA256 | `207f2b2514f533e44bf92c5bdecc9c63f0d813e36f3117e716835776b561d09d` |
| Prepared tokenizer SHA256 | `7fe399d0739ac28d562a04f0d294093d5d38ad3f00e6ea0e624f932ccf397116` |
| Fixed-date chat template SHA256 | `a718674808440b537a553c15e0557b5b45d6d76e49ad9ddbc1a9e92447843e2c` |
| `data-small/train_main.parquet`, 256 cases | `6010acaa4d4be314101a537d8fa3b3d0f7f3811ddc53b0162ff8a61300620178` |
| `data-small/eval_main.parquet`, 64 cases | `e90950dcb3daedea284091a54d40de5f2f59a8530e28bdd8e4e4f8ee53ecc0ba` |
| `data-small/eval_main.jsonl`, 64 cases | `27445758e42fd93c062ce297515869af2a372e1edd66cd7e1f853c37fbc96850` |

Model/data/package manifests are identical across base/train/post. Current frozen dataset, prepared tokenizer and template hashes were checked read-only; the JSONL case list matches the manifest. Observed training IDs stay within the training split and do not overlap held-out IDs. The 64 initial model requests have identical serialized messages/tool schemas across baseline, train-initial, train-final and reload. This is input-record identity, not proof of token-level or numerical identity inside the backend.

The common package records include VERL 0.8.0, vLLM 0.22.0, Torch 2.11.0+cu130, FlashAttention 2.8.3, Transformers 5.22.2, AutoGen 0.7.5, MCP calculator 0.2.1, W&B 0.31.0 and Weave 0.54.1. The CUDA image digest is `1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d`. No hosted W&B readback was attempted in this closeout. All three runs have `weave=false`; hosted Weave delivery is not claimed.

The resolved protocol remains GRPO with rollout-level, standard-deviation-normalized advantages; 8 questions × 4 rollouts per iteration; full-weight FSDP with CPU parameter/optimizer offload; AdamW LR `1e-6`, weight decay 0.01, constant schedule; actor mini-batch setting 8, microbatch 1, one PPO epoch, clip low/high 0.2/0.3, gradient clip 1.0; `per_rollout_mean` policy loss, no KL/entropy penalty. vLLM uses BF16, TP 1, eager mode, max context 4096, `llama3_json`, and the frozen template. Agent bounds remain two tool iterations and 1024 generated tokens/request.

Validation uses one rollout/case, `do_sample=false`, temperature 0, top-p 1 and top-k -1; the gateway also explicitly sets validation temperature 0. Train/post resolved configuration differences are limited to run labels/output paths, gateway port, `val_only`, and checkpoint-resume settings. No model, tokenizer, parser, sampling-limit, package or reward change was found. The generic dataset paths in resolved configuration are upstream defaults; `run_plan.json` and the hash-matched launcher pass the frozen datasets directly.

## Post restoration and native checkpoint

The source is exactly `runs/train-01/checkpoints/global_step_20`. The completed marker contains `20`; `train-01/result.json` identifies that checkpoint as latest. Read-only metadata inspection found:

| File relative to `global_step_20/` | Bytes |
|---|---:|
| `actor/model_world_size_1_rank_0.pt` | 14,427,143,047 |
| `actor/optim_world_size_1_rank_0.pt` | 25,702,025,404 |
| `actor/extra_state_world_size_1_rank_0.pt` | 15,141 |
| `data.pt` | 7,316 |
| `actor/fsdp_config.json` | 46 |
| `actor/huggingface/config.json` | 863 |
| `actor/huggingface/generation_config.json` | 184 |
| `actor/huggingface/tokenizer.json` | 17,209,920 |
| `actor/huggingface/tokenizer_config.json` | 353 |
| `actor/huggingface/chat_template.jinja` | 5,294 |

FSDP metadata records version 1/world size 1; model metadata identifies `LlamaForCausalLM`. The checkpoint chat template hash matches the prepared template. Actor files remain at steps 15 and 20; step-5/10 directories retain only `data.pt`, consistent with keeping two actor checkpoints. No tensor files were read or hashed.

`post-01/logs/trainer.log` identifies the exact checkpoint folder at line 1811, sets global step 20 at 1812 and resumes at 1813; line 1909 reports 64/64 rollouts succeeded, and line 1917 reports step-20 validation reward 0.546875 with 64 traces/rewards. Line numbers count lines after stripping ANSI colors. There are no actor-update metrics, training-route episodes, or new checkpoint directory. `result.json` has an empty checkpoint list and no latest checkpoint.

**Inferred from pinned source:** [VERL `_load_checkpoint`](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/ppo/ray_trainer.py#L1043) loads the actor before returning; [Agent Lightning `fit`](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/trainer.py#L680) calls checkpoint loading, then `update_weights`, then validation, and returns for `val_only`. Progress through this path to completed validation corroborates restoration and actor-to-inference refresh beyond the requested-path field. There is no independently recorded tensor fingerprint, nor a separate success receipt for each internal load/refresh operation.

## Final pairing and behavior

The existing summary has `comparison.available=true`, `paired_cases=64`. Local analysis independently validates JSONL schemas, one episode/trace, unique rollout/attempt identities, real held-out validation metadata, exact expected case multiplicity and completed posted binary rewards. Its paired counts and aggregate rewards, calls, errors, usage and durations match the saved summary (allowing only floating-point summation roundoff).

| Baseline → post outcome | Cases |
|---|---:|
| Incorrect → correct (wins) | 11 |
| Correct → incorrect (losses) | 3 |
| Correct → correct | 24 |
| Incorrect → incorrect | 26 |
| Ties | 50 |

Checks: `11 - 3 = 35 - 27 = 8`; `11 + 3 + 50 = 64`. The descriptive reward increase is **8/64 = 12.5 percentage points**. These are post-01 results, not the distinct train-final counts of 11 wins / 4 losses / 49 ties.

| Recorded measure | `base-01` | `post-01` |
|---|---:|---:|
| Correct / expected cases | 27/64 (42.1875%) | 35/64 (54.6875%) |
| Completed / expected cases | 64/64 | 64/64 |
| Posted / trainer-received rewards | 64/64 | 64/64 |
| Missing rewards / incomplete episodes | 0 / 0 | 0 / 0 |
| Model calls / recorded model exceptions | 140 / 0 | 134 / 0 |
| Tool calls | 76 | 70 |
| Tool-reported errors / calls | 23/76 (30.26%) | 12/70 (17.14%) |
| Episodes affected by tool errors | 13/64 (20.31%) | 10/64 (15.63%) |
| Model calls with reported usage | 140/140 | 134/134 |
| Reported prompt tokens | 44,546 | 42,765 |
| Reported completion tokens | 5,468 | 5,633 |
| Mean / median episode duration | 5.193 / 3.834 s | 2.689 / 2.131 s |
| Maximum episode duration | 23.204 s | 9.916 s |
| Whole-job receipt duration | 435.446 s | 316.494 s |
| Model responses finishing by length limit | 1 | 0 |

Every baseline/post episode used a tool. Tool errors are returned MCP errors, distinct from model transport exceptions or parser fallbacks. Counts describe recorded calls, not hidden transport retries. Usage is server-reported through AutoGen, not independent token billing; all calls have usage records here, so no missing values were zero-filled. Inclusive episode times overlap across concurrent episodes; their sum is not wall-clock runtime. Whole-job times include setup/model loading and shutdown. Different startup/cache/scheduling conditions prevent interpreting this as a controlled speed benchmark.

## Why train-final was 34/64 and reload was 35/64

The 128 training-run validation trajectories were separated by timestamps against trainer step windows: initial episodes precede the first training step; final episodes follow the step-20 rollout window. There are exactly 64 expected IDs once in each set, with no unclassified validation episodes.

| Evaluation | Episode interval, 2026-11-13 UTC | Correct | Rewards received |
|---|---|---:|---:|
| Train step 0 | 06:26:22–06:27:38 | 27/64 | 64 |
| Train step 20 | 06:54:04–06:55:11 | 34/64 | 64 |
| Independent reload | 14:55:13–14:56:14 | 35/64 | 64 |

Train-initial rewards equal base-01 case by case. Train-final versus reload has **one 0→1 flip, zero 1→0 flips, 63 ties** (34 both-correct, 29 both-incorrect). The only reward flip is `svamp__chal-520`:

| Observation | Train-final | Post reload |
|---|---|---|
| Trace ID | `2d5c721fdd644b8182b62e2bb6361970` | `03a222e52eb0480d9b8d257961a99d80` |
| Calculator expression / result | `14 - 8` / `6`, no error | Same |
| Model calls / tool calls | 2 / 1 | 2 / 1 |
| Final response includes | Quoted `### ANSWER: <answer> ###`, then `### ANSWER: 6 ###` | `### ANSWER: 6 ###`, without the quoted placeholder |
| Extracted answer | `<answer>` | `6` |
| Posted reward | 0 | 1 |
| Final finish reason | `stop` | `stop` |
| First / second model usage, prompt/completion | 292/19; 327/44 | 292/19; 327/29 |

The unchanged [harness extractor](../experiment/observed_calc_agent.py) uses the **first** regex match of `### ANSWER: ...`. Applying that exact expression offline to the two recorded outputs reproduces their stored extracted answers. Thus this flip reflects output-format sensitivity; both responses contain the numeric answer 6. There was no tool failure, parser-fallback event within this case's execution interval, or length-limit finish. Initial inputs match exactly; the continuation input records match after removing differing generated tool-call IDs, and reported prompt lengths agree. Exact internal token buffers were not audited.

Across all 64 cases, 12 raw final answers and 6 extracted answer strings differ, but only the case above changes reward. Tool/model-call counts also differ on three unchanged-reward cases: `ape210k__00566786` uses one fewer tool call on reload; `ape210k__00999729` uses one fewer errored tool call; `gsm8k__RBn8bGGNuDjKawVx` uses one additional tool call. Therefore identical aggregate outcomes would not have implied identical trajectories either.

**Bounded explanation:** no decoding, package, tokenizer/template, harness or data drift was found. Temperature zero does not guarantee identical output after process restart/reload. Scheduling or numerical variability is a hypothesis, not an established cause; existing evidence cannot determine why the wording changed or independently prove tensor identity. Preserve 34/64 and 35/64 as distinct measured results.

**Concrete defect and separate proposed work:** the first-match answer extractor can grade a quoted instructional placeholder instead of the later intended answer. A future bounded change should define unambiguous final-answer extraction and add fixtures for quoted placeholders, repeated markers and missing markers, under a separately versioned reward/harness protocol. No extractor change, new test, retroactive rescoring or rerun is part of this PR. The same frozen scorer qualifies this comparison, but its scores must not be represented as a complete audit of mathematical correctness.

## Illustrative paired traces

These selected public-task examples illustrate behavior; they are not representative estimates. Full paths follow `runs/RUN/traces/trajectory-ID.jsonl`; the case CSV maps every case to its run-specific trace IDs.

| Category / case | Baseline → post observation | Reward |
|---|---|---|
| Win, `ape210k__00219289` | Calculator expression changes from `126 + 126 * 15` (2016) to `126 + 15 * 126 / 6` (441); post emits 441. | 0→1 |
| Formatting win, `asdiv_a__nluds-1935` | Both calculate `90 - 87 = 3`; baseline extracted answer retains `### 3 ###`, while post extracts `3`. | 0→1 |
| Regression, `ape210k__00163219` | `(3*4) + (3*4)` returns 24 initially; post calls `3 * 4` and answers 12. Both tools execute successfully. | 1→0 |
| Unchanged failure, `ape210k__00075882` | Both tools return 100000 for `0.05 * 2000000`, but final answers are 1000000. | 0→0 |

## Retained all-20-step training audit

Exactly 20 trainer iterations produced 640 completed rewarded training episodes. All numeric step metrics are finite and requested fields are present. Each step has 32 rollouts, 32 with received rewards and 8 GRPO groups. Below, **Z** is zero-variance groups; **A/W** is actor-update/weight-refresh seconds; **C/T/D** is collected/trained/dropped rows. Values are rounded; all drops are `same_reward`, with marked/random drops zero at every step.

| Step | Z / 8 | Advantage min / max | Grad norm | PG loss | Actor loss | A/W seconds | C/T/D |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7/8 | -0.499999 / 1.499997 | 1.219003 | 0.0002025266 | 0.0002025268 | 27.304/2.653 | 71/64/7 |
| 2 | 8/8 | 0.000000 / 0.000000 | 0.000000 | 0 | 0 | 15.789/3.460 | 71/64/7 |
| 3 | 6/8 | -0.499999 / 1.499997 | 0.947244 | 0.0001578569 | 0.0001578564 | 16.236/3.950 | 74/64/10 |
| 4 | 6/8 | -0.866024 / 0.866024 | 2.108149 | -4.454283e-05 | -4.45433e-05 | 16.391/3.659 | 68/64/4 |
| 5 | 8/8 | 0.000000 / 0.000000 | 0.000000 | 0 | 0 | 15.372/3.339 | 71/64/7 |
| 6 | 8/8 | 0.000000 / 0.000000 | 0.000000 | 0 | 0 | 18.236/4.095 | 70/64/6 |
| 7 | 7/8 | -1.499997 / 0.499999 | 1.156745 | -0.0001697233 | -0.0001697233 | 16.755/3.651 | 64/64/0 |
| 8 | 6/8 | -0.499999 / 1.499997 | 0.853050 | -0.0001128259 | -0.000112826 | 16.007/3.686 | 67/64/3 |
| 9 | 8/8 | 0.000000 / 0.000000 | 0.000000 | 0 | 0 | 15.628/3.688 | 69/64/5 |
| 10 | 7/8 | -0.499999 / 1.499997 | 0.360958 | -3.005576e-05 | -3.005611e-05 | 15.517/3.187 | 74/64/10 |
| 11 | 8/8 | 0.000000 / 0.000000 | 0.000000 | 0 | 0 | 17.772/3.690 | 72/64/8 |
| 12 | 2/8 | -1.499997 / 1.499997 | 1.764121 | -7.093156e-05 | -7.093325e-05 | 15.520/3.478 | 71/64/7 |
| 13 | 6/8 | -0.866024 / 0.866024 | 1.012655 | -5.21121e-05 | -5.211215e-05 | 15.584/3.901 | 69/64/5 |
| 14 | 4/8 | -0.499999 / 1.499997 | 1.298675 | 7.378959e-05 | 7.378869e-05 | 17.419/3.427 | 73/64/9 |
| 15 | 8/8 | 0.000000 / 0.000000 | 0.000000 | 0 | 0 | 15.118/3.394 | 68/64/4 |
| 16 | 6/8 | -0.866024 / 1.499997 | 1.315361 | -1.63237e-05 | -1.632422e-05 | 17.781/3.586 | 71/64/7 |
| 17 | 5/8 | -1.499997 / 1.499997 | 1.367509 | 6.948685e-06 | 6.948598e-06 | 15.915/3.453 | 67/64/3 |
| 18 | 6/8 | -0.866024 / 1.499997 | 0.641405 | -5.168346e-06 | -5.167909e-06 | 16.555/3.385 | 66/64/2 |
| 19 | 7/8 | -1.499997 / 0.499999 | 1.116045 | 4.923192e-05 | 4.923181e-05 | 16.176/3.673 | 70/64/6 |
| 20 | 7/8 | -0.499999 / 1.499997 | 0.480757 | -2.52648e-05 | -2.526492e-05 | 15.969/3.339 | 68/64/4 |

Totals: **30/160 informative groups (18.75%)**, 130 constant-reward groups, **1,394 collected / 1,280 trained / 114 dropped rows**. Independent trace grouping yields 8 groups of 4 attempts per step and reproduces all zero-variance counts. Training has 95 rewarded successes out of 640 attempts across changing questions; this is not a paired learning curve.

Steps **2, 5, 6, 9, 11, 15** have zero advantages, losses and gradient norms. All update/refresh stage timings are positive, with no logged skipped update stage or missing metric. These are 20 completed trainer iterations, not 20 demonstrated nonzero learning updates. Zero current gradients do not establish unchanged AdamW parameters: optimizer state and weight decay may still matter. Tensor changes were not measured independently.

The trainer reports **617/640 unmerged rollouts**, **754 trace-merge mismatch rows**, zero truncated samples and zero skipped-empty rows. The pinned [rollout adapter](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/rollout_adapter.py#L635) retains separate rows when token prefixes cannot merge. [Batch flooring](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/trainer.py#L514) preferentially drops constant-reward-group rows to fit the mini-batch multiple. Five steps (3, 5, 10, 12, 20) retain only 31 distinct rollouts in the advantage batch despite all 32 rollout rewards arriving. Advantages are reported before additional per-rollout loss normalization. Mismatch causes and learning impact remain unverified.

Training trajectories contain 1,396 model calls, 756 tool calls and 324 tool-reported errors affecting 239/640 episodes; no model exceptions or incomplete episodes were recorded. Train-initial has 23 tool errors/76 calls; train-final has 13/71. Trainer logs contain **48 parser fallbacks** (47 during training, 1 in final validation), accounting for its 48 tracebacks. Post has one parser fallback and one traceback. Parser fallback returns text rather than a structured call; zero model exceptions does not mean every generated tool request was valid.

## Dependency and interpretation limits

Both retained training and reload image-build checks report the same six reviewed incompatibilities; clean dependency closure is **failed**, not repaired by successful execution:

| Package | Requirement | Build-audited installed dependency |
|---|---|---|
| ml-dtypes 0.6.0 | `numpy>=2.0.0` | 1.26.4 |
| opencv-python-headless 5.0.0.93 | `numpy>=2 ; python_full_version >= '3.9'` | 1.26.4 |
| google-api-core 2.44.0 | `protobuf>=6.33.5,<8.0.0` | 5.29.6 |
| tensorboard 2.21.0 | `protobuf>=6.31.1,<8.0.0` | 5.29.6 |
| xgrammar 0.2.5.post1 | `transformers>=4.38.0,<5` | 5.22.2 |
| proto-plus 1.29.0 | `protobuf>=6.33.5,<8.0.0` | 5.29.6 |

The unscoped runtime package dictionaries instead report Protobuf 6.31.1, consistent with the previously documented ambient Modal metadata ambiguity. The actual imported Protobuf version was not recorded by these training/evaluation processes. Matching package dictionaries do not resolve that ambiguity.

This is one small run on one fixed 64-case set, with sparse reward discrimination, tool errors, answer-format sensitivity, row dropping, unclassified trace mismatches and a bounded replay difference. It does not establish population generalization, statistical significance, convergence, production suitability, or superiority to another framework. Final reload is independent execution on the same cases, not a fresh test population. The smaller latency/error counts are descriptive observations.

## Reconciliation and review record

The sanitized [case-level CSV](grpo-main-01-cases.csv) contains 64 unique IDs, all four rewards (base, train-step-0, train-step-20, post), base/post call/error/usage/duration measures and trace IDs. `base_trace_id`, `train_step0_trace_id`, `train_step20_trace_id`, and `post_trace_id` resolve using the path convention above. Durations are milliseconds; token counts sum only available recorded usage. No prompts, full model outputs, ground-truth dataset rows, raw logs, tensors or credentials are included in the CSV.

Artifact identity anchors (SHA256):

- `evidence/paired-01/summary.json`: `8cba9abbe20f5200f93af9944fbfea6de32c0672c95850d6b7f6ba5595fce1ed`
- `runs/train-01/logs/trainer.log`: `a838e402a92f38ec7ced0c7dfd2443cf8c20beb64975dafa6d86a9efe89213ca`
- `runs/post-01/logs/trainer.log`: `c7b2a9a878c8b2f1d5bf040e328b90ea407d0b4b1a0d5422f5832f22fd95a0fc`
- Changed train-final trajectory: `989f6cbe5066b307bda882f238c63eadc6f5af03e8162adfbd63ad419165f1a9`
- Changed post trajectory: `fd2d4c309d6bd4edbdeacc4508dd34d174184d817a6211b75e5b4779c52fa2a6`

Offline evidence checks executed in this closeout: schema/completeness/identity assertions, frozen-file hashes, timestamp partitioning, per-case pairing, saved-summary reconciliation, all-step metric checks, and replay of the existing extraction regex on the changed case. These are local analysis of historical data, not new model runs. Relevant summarizer/launcher paths and existing offline request/persistence tests were reviewed. No unit-test suite was run for this documentation-only change; earlier test counts are not presented as new execution. The final documentation/CSV diff is checked with `git diff --check` and reviewed for sensitive material before commit.

No comparison/provenance blocker remains for documenting the frozen-protocol result. The exact cause of output variability, tensor identity and broader learning claims remain unverified. Any extractor repair, further experiment, model export or Hugging Face publication requires a separate scope; none is performed here.
