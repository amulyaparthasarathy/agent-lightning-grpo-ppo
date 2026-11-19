# GRPO smoke and checkpoint reload evidence

**Qualification:** the existing `smoke-01` and `smoke-reload-01` establish operational inference → MCP continuation → reward → actor update → checkpoint → reload, with sparse nonzero GRPO signal. They do not establish general learning improvement. This review used read-only persisted evidence and local offline tests; no new Modal job, inference, training, or checkpoint rewrite was performed.

Classification throughout: **verified** means corroborated by persisted execution records or direct read-only checks; **inferred from source code** identifies implementation interpretation; **failed** identifies an observed unsuccessful check or operation; **unverified** means evidence is absent or insufficient. Verification is operational, not an independent numerical audit of model tensors.

## Provenance and evidence locations

**Verified:** GitHub `main` and the clean local `feat/grpo-smoke-validation` branch both started at `ee120371429a4bc23d11b9da53d166d2fa4843fd`. Both submission receipts record this source commit with `tracked_dirty: false`. All 12 recorded submission source-file hashes match the local files. Historical verification documents remain unchanged; their earlier “not run” statements describe earlier milestones.

All paths below are relative to the existing Modal Volume **`cs329z-agent-lightning`**, mounted at `/data`. Runtime logs resolve that mount to `/__modal/volumes/vo-lCOCqrVwdlYV46hXMoHHws`; these are the same artifact locations.

| Evidence | `smoke-01` | `smoke-reload-01` |
|---|---|---|
| Submission | `submissions/smoke-01/receipt.json` | `submissions/smoke-reload-01/receipt.json` |
| Run directory | `runs/smoke-01/` | `runs/smoke-reload-01/` |
| Receipt interval, UTC | Nov 12, 2026, 20:40:21–20:49:49 | Nov 13, 2026, 05:12:52–05:18:59 |
| Result | `completed`, exit 0 | `completed`, exit 0 |
| W&B run ID | `f9ea39c7eeb7ef7a` | `d40b044bf11f4ae7` |
| Trainer log SHA256 | `88d88b5b23b00a25f7f3a6054d76ec3145385bb292d7bc681b365e001777b946` | `b77c4b3ea785fc4b41c7544d7fcc93c88340974b0dd028b3d21449af1120f5c9` |

**Verified:** each run directory contains `result.json`, `manifest.json`, `model_manifest.json`, `data_manifest.json`, `resolved_config.json`, `resolved_packages.json`, `observability.json`, `logs/trainer.log`, and `traces/trajectory-*.jsonl`. Submission directories also retain `source-sha256.json` and `image-build/`. The trainer logs are 191,475 and 168,265 bytes respectively. Log line references below count lines after stripping ANSI color sequences.

**Unverified:** independent W&B history readback. An unauthenticated read-only GraphQL request returned `project: null`; no credentials were opened or copied to obtain access. Persisted console history supplies the metrics below. The recorded links are [smoke-01](https://wandb.ai/amulyap-stanford-university/cs329z-agent-lightning/runs/f9ea39c7eeb7ef7a) and [smoke-reload-01](https://wandb.ai/amulyap-stanford-university/cs329z-agent-lightning/runs/d40b044bf11f4ae7). Hosted Weave ingestion is also unverified: both receipts explicitly have `weave: false`; local traces are available.

**Verified:** Agent Lightning v1.0.3 at `cbc56e41e0eab643aa71d195e92e5bba652fa683`; VERL 0.8.0; vLLM 0.22.0; Torch 2.11.0+cu130; FlashAttention 2.8.3; AutoGen agentchat/ext 0.7.5; MCP calculator 0.2.1; Transformers 5.22.2; W&B 0.31.0; Weave 0.54.1. The run package manifests are identical. The training interpreter is `/opt/agent-lightning/.venv/bin/python`; the image is CUDA 13.0.0 devel Ubuntu 24.04, digest `1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d`. Protobuf has the inventory limitation described below.

**Verified:** both model manifests are identical: `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, architecture `LlamaForCausalLM`. Prepared tokenizer SHA256 is `7fe399d0739ac28d562a04f0d294093d5d38ad3f00e6ea0e624f932ccf397116`; fixed-date Llama JSON template SHA256 is `a718674808440b537a553c15e0557b5b45d6d76e49ad9ddbc1a9e92447843e2c`. The harness SHA256 is `e802a2324dab779ffaa49af678196a7cc5f02e3bb89936bd0ed1f1ed8c3a02fc`. Weights were neither downloaded nor inspected during this review.

**Verified:** data manifests match across runs; all six current frozen files were independently SHA256-checked through read-only Volume streams, without retaining dataset contents. Selection seed is 329. Manifest case IDs confirm train/evaluation disjointness and smoke/final evaluation disjointness; smoke training is nested in main training.

| Frozen file under `data-small/` | Rows | SHA256 |
|---|---:|---|
| `train_smoke.parquet` | 16 | `065b218b8b2bb16f80edf7b4721e2cd98fc6188f603af02146182591ccba6182` |
| `eval_smoke.parquet` | 8 | `d0ceb9f4daea9af580e09b932493ee35ade4ad945127603da2c0f272d54c074e` |
| `eval_smoke.jsonl` | 8 | `703e6203df93c3741f0de843f22a22cf66237b44fb3267193484f79a7ad0f960` |
| `train_main.parquet` | 256 | `6010acaa4d4be314101a537d8fa3b3d0f7f3811ddc53b0162ff8a61300620178` |
| `eval_main.parquet` | 64 | `e90950dcb3daedea284091a54d40de5f2f59a8530e28bdd8e4e4f8ee53ecc0ba` |
| `eval_main.jsonl` | 64 | `27445758e42fd93c062ce297515869af2a372e1edd66cd7e1f853c37fbc96850` |

## Resolved training configuration

**Verified:** these values come from the saved resolved configuration, not the proposed runbook defaults.

| Setting | Actual value |
|---|---|
| Algorithm | GRPO; rollout-level advantage enabled; standard-deviation normalization enabled; gamma/lambda 1 |
| Training batch / group | 8 questions × 4 sampled rollouts = 32 attempts per step |
| Actor | Full weights, FSDP, parameter/optimizer CPU offload; LoRA rank 0; gradient checkpointing enabled |
| Optimizer | AdamW, LR `1e-6`, constant schedule, weight decay 0.01, betas (0.9, 0.999), gradient clip 1.0 |
| PPO | Mini-batch setting 8; microbatch 1/GPU; 1 epoch; clip low/high 0.2/0.3 |
| Loss | `per_rollout_mean`; `loss_agg_mode=token-mean`; entropy coefficient 0; KL reward/loss disabled |
| Rollout | vLLM BF16; TP 1; max model length 4096; max sequences 32; GPU memory fraction 0.5; eager mode |
| Sampling | Train temperature 1, top-p 1, top-k -1; validation temperature 0, no sampling, one rollout/case |
| Length bounds | Data prompt/response 2048/1024; trajectory aggregation prompt/response 2048/2048 |
| Agent | AutoGen + MCP calculator; `llama3_json`; 2 tool iterations; 1024 generated tokens/request; 300-second rollout timeout |
| Smoke schedule | 2 training steps; validate initially and each step; save each step; retain 2 actor checkpoints |
| Reload | `val_only=true`, `resume_mode=resume_path`, explicit smoke `global_step_2`; no new training |

**Inferred from source code:** the agent bounds and gateway sampling overrides are fixed in the hash-matched launcher/harness. The numerical reward is the unchanged Calc-X comparator with tolerance 0.01; it does not require a calculator call. The generic `data.train_files`/`val_files` strings in the resolved config are unused defaults: the [pinned Calc-X launcher](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/examples/calc_x/train_calc_agent.py) passes separately loaded datasets to training. **Verified:** the trainer prints 16 training and 8 validation samples, and trace case IDs match the frozen smoke manifests.

## Smoke execution and signal

**Verified:** the trainer reports `LlamaForCausalLM contains 3.21B parameters` (line 1700), starts the vLLM server against the prepared Llama path with `llama3_json` (1712–1754), and records actual model/tool continuations. The resolved `load_format=dummy` is not evidence of synthetic generation. **Inferred from source code:** the [pinned trainer](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/trainer.py#L680) pushes actor weights to the rollout backend before initial evaluation.

**Verified:** all 88 smoke trajectory files parse, contain exactly one successful episode, have distinct rollout/attempt pairs, identify `synthetic_fixture=false`, and confirm a posted binary reward. There are 64 train attempts and 24 validation attempts. Trainer rollout/trace/reward counts independently agree: 32/32/32 at each training step and 8/8/8 for each validation batch. There are no missing rewards, failed episodes, or incomplete episodes in the reviewed traces. “Successful episode” means execution completed, not that the answer was correct.

| Verified metric | Step 1 | Step 2 |
|---|---:|---:|
| Training reward, successful answers / attempts | 1/32 = 0.03125 | 6/32 = 0.1875 |
| GRPO groups / zero-variance groups | 8 / 7 | 8 / 6 |
| Groups with nonzero signal | 1/8 | 2/8 |
| Advantage min / max | -0.4999989867210388 / 1.4999970197677612 | -0.4999989867210388 / 1.4999970197677612 |
| Advantage mean | -0.07792410999536514 | 0.009711864404380322 |
| `actor/pg_loss` | -0.00007706764154136181 | 0.000010282150469720364 |
| `actor/loss` | -0.00007706787437200546 | 0.000010281801223754883 |
| `actor/grad_norm` | 0.7503504902124405 | 0.7750576883554459 |
| Actor update seconds | 28.650470405 | 17.504815868 |
| Rollout weight refresh seconds | 2.953770664 | 3.519694015 |
| Collected rows / trained rows | 68 / 64 | 73 / 64 |
| Same-reward rows dropped / random rows dropped | 4 / 0 | 9 / 0 |
| Distinct rollouts in advantage batch | 32 | 31 |
| Post-drop `critic/rewards/mean` | 0.03125 | 0.203125 |
| Validation reward | 5/8 = 0.625 | 5/8 = 0.625 |

Metric source: `runs/smoke-01/logs/trainer.log`, step 0 line 1810, step 1 line 1918, step 2 line 2007. Initial validation was **3/8 = 0.375**. The actor losses and gradient norms are finite, with positive gradient norms and completed update/refresh timings. Peak reported actor allocated/reserved GPU memory was approximately 59.905/63.217 GiB; this is smoke evidence, not a 20-step memory guarantee.

**Verified:** independent trace grouping by case and chronological rollout batch reproduces the reward distributions: step 1 has seven all-zero groups and one `[0,0,0,1]` group; step 2 has five all-zero groups, one all-one group, and two groups with one success among four attempts. This establishes useful but sparse reward discrimination: only 3 of 16 groups across the two steps supply nonzero signal. The training batches contain different questions, so rising training reward is not a paired improvement measure.

**Inferred from source code:** `training/n_zero_adv_groups` counts constant-reward groups in the trained batch. Upstream rounds row counts down to a multiple of `ppo_mini_batch_size × rollout.n = 32`, preferentially dropping constant-reward-group rows. This explains why 32 completed rewarded rollouts need not remain 32 distinct rollouts in the trained batch. The step-2 post-drop row reward mean of 0.203125 must not replace the actual rollout reward 6/32. The reported advantage statistics precede the additional per-rollout loss normalization. See the [pinned trainer's batching and update path](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/trainer.py#L514).

**Verified operationally:** both actor-update and rollout-weight-refresh stages completed, followed by validation and native checkpoints. **Unverified:** an independent tensor-difference measurement or an exact per-rollout weight fingerprint; neither is recorded, and model weights were not read for this review.

## Real MCP tool-result continuation

**Verified:** `runs/smoke-01/traces/trajectory-ff133885988e4bc19a01adc196396f2f.jsonl`, case `gsm8k__GpgxXeLS6Q14g7zD`, rollout `f07bc6b6edd94c7299c45280bf23ca5c`, attempt 0, validation route:

1. The first model response requests `calculate` with expression `(100 - 3 * 14) - (100 - 4 * 14)` and call ID `chatcmpl-tool-8afee7dcbe67ad72`.
2. The MCP tool returns text `14`, `is_error=false`.
3. The next model request contains a `FunctionExecutionResultMessage` with content `14` and the same call ID.
4. The model returns `### ANSWER: 14 ###`; the episode completes with reward 1 and `reward_posted=true`.

Reported prompt/completion token counts are 293/33 and 342/8 for the two model calls. This verifies observation continuation, not just tool dispatch. The example is a selected public-task excerpt; the raw trajectory is not committed. Its SHA256 is `5a9081875bb0a4d04be6853f317768af896a310f5045a3685d8cbe74d51505b8`.

## Native checkpoint and reload

**Verified:** `runs/smoke-01/checkpoints/latest_checkpointed_iteration.txt` contains `2`; both `global_step_1` and `global_step_2` exist. Read-only metadata inspection found these nonempty native step-2 files (no tensor files were downloaded):

| File relative to `global_step_2/` | Bytes |
|---|---:|
| `actor/model_world_size_1_rank_0.pt` | 14,427,143,047 |
| `actor/optim_world_size_1_rank_0.pt` | 25,702,025,404 |
| `actor/extra_state_world_size_1_rank_0.pt` | 15,141 |
| `data.pt` | 7,316 |
| `actor/fsdp_config.json` | 46 |

The directory also contains `actor/huggingface/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja}`. **Unverified:** independent checkpoint checksums/tensor integrity; successful reload provides operational evidence instead.

**Verified:** the reload request, resolved config and result identify exactly `runs/smoke-01/checkpoints/global_step_2`; model, data and harness provenance match. The reload trainer log identifies that folder (1817), sets global step to 2 (1818), resumes it (1819), then reaches the dataloader epoch-boundary message (1833) and completes evaluation (1865). **Inferred from source code:** [VERL 0.8.0 `_load_checkpoint`](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/ppo/ray_trainer.py#L1043) loads the actor before reaching that dataloader message; the pinned Agent Lightning `fit` refreshes rollout weights before validation. Thus restoration has execution-path corroboration beyond the launcher's `restored_checkpoint` field, which by itself merely echoes the requested path.

**Verified:** all eight reload trajectories complete with posted binary rewards; exactly the eight frozen smoke-evaluation case IDs appear once. The trainer confirms 8 rollouts, 8 traces and 8 received rewards. Five answers receive reward 1 and three receive reward 0: **5/8 = 0.625, denominator 8**. There are 16 recorded model calls and 8 tool calls. All eight case-level rewards match smoke step-2 validation; this does not imply byte-identical responses (mean total response length is 45.625 versus 46 tokens).

**Verified:** reload has only validation metrics at step 2, zero training-route trajectories, an empty checkpoint list, no latest checkpoint, and no checkpoint directory in its Volume listing. **Inferred from source code:** `val_only=true` returns immediately after initial validation, before actor updates or checkpoint saves. Together these support no additional actor update or checkpoint creation. The log's missing tracker warning refers to the new reload output directory; explicit source-checkpoint restoration follows it. Skipping exhausted dataloader state at the epoch boundary is not skipping actor restoration.

## Failures and limits

| Classification | Finding and consequence |
|---|---|
| Failed | Perfect tool execution: training records 13/36 tool-call errors at step 1 and 19/41 at step 2; smoke validation records 2/9 initially and 1/8 at each later validation; reload records 1/8. These errors coexist with completed episodes and posted rewards. |
| Failed | Error-free tool parsing: three smoke trainer messages report extraction failure with fallback to text/empty tool calls (1847, 1930, 1945). Model/tool transport retries are not counted by the recorded span totals. |
| Failed | Fully merged training trajectories: 31/32 and 32/32 rollouts are unmerged, with 36 and 41 mismatch rows. No samples are reported truncated or skipped as empty. |
| Inferred from source code | The [pinned rollout adapter](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/rollout_adapter.py#L635) retains separate training rows when token prefixes cannot merge. Rollout-level advantage and per-rollout loss are enabled. This is a supported fallback, not proof of an exact single-row trajectory representation. |
| Unverified | Root cause and learning impact of trace merge mismatches: classification was disabled. No raw token/logprob audit or causal diagnosis was performed. |
| Failed | Clean dependency closure: both retained image-build checks exit 1 with the six exact reviewed conflicts below. Operational smoke success does not repair package metadata or establish production suitability. |
| Unverified | General learning, significance, final 64-case performance, long-run stability, and transfer to production tasks. The 3/8 → 5/8 smoke change is descriptive and reuses a tiny development set. |

**Verified:** both `submissions/NAME/image-build/pip-check.txt` records contain the same six conflicts, retained without changing pins:

| Requiring package | Requirement | Installed dependency in build audit |
|---|---|---|
| ml-dtypes 0.6.0 | `numpy>=2.0.0` | 1.26.4 |
| opencv-python-headless 5.0.0.93 | `numpy>=2 ; python_full_version >= '3.9'` | 1.26.4 |
| google-api-core 2.44.0 | `protobuf>=6.33.5,<8.0.0` | 5.29.6 |
| tensorboard 2.21.0 | `protobuf>=6.31.1,<8.0.0` | 5.29.6 |
| xgrammar 0.2.5.post1 | `transformers>=4.38.0,<5` | 5.22.2 |
| proto-plus 1.29.0 | `protobuf>=6.33.5,<8.0.0` | 5.29.6 |

**Verified:** both runtime `resolved_packages.json` files instead report Protobuf 6.31.1, while both retained scoped build inventories report 5.29.6. **Inferred from source code:** the launcher's unscoped distribution dictionary can select ambient Modal metadata, the ambiguity already documented in [VALIDATION.md](../VALIDATION.md). **Unverified:** the actual Protobuf module imported by these two training processes; they did not persist the scoped import audit used by the earlier validation jobs. Do not silently present either inventory as a new runtime qualification.

## Review and next action

**Verified:** `uv run --locked python -m unittest discover -s tests -v` passed **74 tests** in 5.410 seconds; `git diff --check` passed. These are local offline checks, not a replay of the experiments. No implementation or dependency changes are needed for this evidence report; no new regression test was added. Only this reviewed summary is intended for Git, with no credentials, raw trajectories, weights or large logs.

**Engineering assessment:** the two-step execution and reload milestone is qualified with the limitations above. A bounded 20-step academic main experiment is technically justified as a measurement attempt because real reward discrimination, finite updates, rollout refresh and native reload have been demonstrated. Improvement is not promised; sparse signal, tool failures, row dropping and unresolved dependency conflicts remain material limitations.

**Recommended next action:** review the evidence-only draft PR. After a separate decision to authorize billable work, follow the existing frozen-data sequence: base-model evaluation on the 64 final cases, the bounded 20-step main run, then checkpoint-reload evaluation on those same cases. Preserve the current model, data, reward, harness, configuration and run names. This review does not submit or authorize that work.
