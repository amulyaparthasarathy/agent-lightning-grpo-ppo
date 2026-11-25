# Phase 4C offline verification

Development calibration used historical execution source `bc81c1fb0def27b15c56cdd8abb01098e7e1f8d5`. This work branched from PR #6's evidence head `79b534f5fad7b11ca32a89a239a74bb53e4d7b00` while that PR was open and draft. Main was `92b276516e93c305cd0a29c61983cb0f798042a9`. The operator's private handoff, ledger, receipts and trace files were available for read-only inspection. All new plans and raw test output stayed in ignored storage in the isolated Phase 4C worktree.

The final matched and task-local gates ran on `f64d050b6deeb9d1b7a8b08df2342e98c2896610`. This includes the admission and loader work, consecutive CPU optimizer steps, and full exposure retention for a lost submission acknowledgement. The final reader gate and proposal validator also cover the prospective evaluation-routing declaration at study source `4ac797ef3da54b651621abdb1a71048355ac7f1a`. The verification report has a subsequent documentation-only commit. All three route selections were independently planned and validated on the clean study source under the offline transport guard.

## Reproducing the gates

Use the existing Unix Workbench with separate locked client, reader-harness and matched CPU environments. The machine used Linux aarch64 and Python 3.12.3. Reader-harness pins include AutoGen 0.7.5, MCP 1.26.0, OpenAI SDK 2.54.0, PyArrow 25.0.1, Jinja2 3.1.6 and tokenizers 0.23.3. The numerical environment used Torch CPU 2.11.0, tensordict 0.10.0, NumPy 1.26.4, Ray 2.60.0, OmegaConf 2.3.0, Hydra 1.3.2, torchdata 0.11.0, datasets 5.2.0 and PyArrow 25.0.1. The CPU preparation path acquires code and dependencies only. Reused historical wheel tests require local copies of the reviewed vLLM and Transformers wheels. No model weights were acquired.

```bash
bash scripts/test_reader_offline.sh
TASK_TEST_PYTHON="$READER_HARNESS_PYTHON" bash scripts/test_task_local.sh
AGL_PINNED_SOURCES="$PINNED_SOURCE_DIR" AGL_VLLM_WHEEL="$PINNED_VLLM_WHEEL" \
  AGL_TRANSFORMERS_WHEELS="$RETAINED_WHEEL_DIR" OMP_NUM_THREADS=1 \
  bash scripts/test_matched_training.sh
python scripts/recovery_study.py
```

| Gate | Pass | Fail | Skip | Scope |
|---|---:|---:|---:|---|
| Reader offline | 39 + 15 + 135 | 0 | 0 | Credential-cleared transport guard, historical result/accounting scripts, contracts, Parquet membership, mock submissions, source and package guards |
| Task local | 106 | 0 | 0 | Actual stdio MCP and production harness, local reward transport, scripted inference, complete four-stage PPO-MC fixture and failed-evidence review |
| Matched training | 73, plus 8 nested native config checks | 0 | 0 | Source-guarded adapter, loader, Hydra/FSDP configurations, Gloo backward, optimizer, local Ray, IPC and wheel regression |
| Study declaration and three fixture plan/validate pairs | All validated | 0 | 0 | All three selections produced source-bound, training-inadmissible plans under the offline transport guard |

The final GRPO, PPO-MC and paired fixture plan SHA-256 values were respectively `a6c6bf52f5c0e9c97c07c9c6afd2e10451307869db9b299de8ff8288be3b63d4`, `c0ae378dfdc0c3bc2a121a5ba60191d2ed569efc0051f3fc3a68be49990310fa` and `c8213ee0ce8aab882d9185d10a74012c4d81416a279d0040e5b75166a55b3292`. They carry fixture identities and do not admit training. A future attempt needs a newly authorized source-bound plan.

The full task gate took 768.128 seconds. The matched gate reported 64.388 seconds, including its 19.375-second nested configuration run.

The initial matched run exposed missing local historical wheel inputs and a skipped wheel regression. Pointing the gate at the retained, hash-checked code wheels allowed both IPC regressions and the wheel guard to run. The CPU test prep adds pinned loader dependencies and FlashAttention's pure-Python padding helpers. No CUDA training dependencies were installed in the reader client. An expected IPC bind failure and deliberately broken MCP transports are negative fixtures. Passing assertions and process exit codes determine the results.

The 144-episode collected-evidence fixture runs the production recovery agent with scripted wrong answers on training and development. It simulates optimizer/audit, provider, W&B and checkpoint artifacts. Its independent collector accepts confirmed zeros and rejects duplicate, missing, corrupted or cross-stage records. Other harness tests cover repaired typed errors, policy termination, malformed answers, truncated or absent completions, raw gateway messages, tool observations and token identity. Tensor tests separately exercise the numerical boundaries. There is no final case in any qualification or model-facing harness invocation.

The actual pinned example reads both Parquet inputs. Its train function runs through both eager Hugging Face reads in tests, with engine launch intercepted. Mode-000 final files trap accidental access. Tests materialize each new stage and inspect row IDs in order. Actual sync fit and iterator methods compare the recorded schedules for seeds 9001, 2001, 2002 and 2003. Final block metadata and synthetic numeric vectors alone exercise the proposed paired analysis. A known-effect fixture yields seed effects 0, 0.5, 0 and a constant replicate mean of 1/6 despite variation among sampled blocks. Missing rewards fail the analysis. [Loader test](../tests/recovery_loader.py) · [Analysis and file tests](../tests/test_recovery_learning.py)

## What the training representation contains

The frozen Llama configuration declares 131072 maximum positions. The pinned serving engine is capped at 4096 active tokens with at most 1024 generated per call. The trajectory adapter independently allocates 3072 prompt and 2048 response slots. The model configuration, serving budget, allocated tensor width and active attended lengths measure different things.

The adapter consumes the original model-request prompt IDs, sampled action IDs and behavior log probabilities. When the next prompt extends the preceding sampled prefix, it can merge turns and mask the intervening context. Divergent prefixes create separate fragment rows. Each fragment retains its original prompt. The adapter left-pads the prompt, right-pads the response and creates attention, position and response-target masks. Admission checks every sampled turn, position, target, probability, reward and group. It rejects missing or truncated context before computing advantages. Only original sampled actions enter the loss. The raw gateway exchange binds the exact wire messages, served prompt IDs, action IDs and behavior probabilities to the same attempt. A served continuation containing malformed JSON tool arguments also keeps those original IDs. The local template adapter cannot independently render that particular continuation. The scripted test verifies this branch and rejects subsequent history corruption.

The six-generation fixture has eight groups of four rollouts. Sixteen episodes merge into one row each, sixteen fragment into six rows each. Each collection has 112 fragment rows, 384 generated tokens, 384 targets, 160 attended but masked response-context tokens and 224 attended prompt tokens. The complete packed collection has 768 tokens. Sixteen rows attend 24 positions each and 96 attend four each. Maximum active position ID is 23 (zero-based).

| Adapter output | Shape | dtype | Bytes per tensor |
|---|---|---|---:|
| input_ids, attention_mask, position_ids | 112 × 5120 | int64 | 4,587,520 |
| prompts | 112 × 3072 | int64 | 2,752,512 |
| responses, response_mask | 112 × 2048 | int64 | 1,835,008 |
| rollout_log_probs, token_level_scores | 112 × 2048 | float32 | 917,504 |
| is_drop_mask | 112 | bool | 112 |

The pinned left_right_2_no_padding conversion gathers active IDs and positions into a jagged nested batch. _compute_old_log_prob calls the worker on that representation and no_padding_2_padding restores response alignment. The fixture supplies synthetic log probabilities of −0.25 against original −0.5 and −0.75. Its mean signed delta is 0.375 and maximum absolute delta is 0.5. Nonfinite, missing or misaligned values reject admission. Serving and training implementations may legitimately differ.

Actor optimization and, for PPO-MC, critic forward/backward each use one row per GPU microbatch. Effective critic FSDP training and inference microbatch sizes are explicitly one, including the engine-specific fields. The test sends four or 24 packed tokens per microbatch through native minibatch splitting and the actual pinned loss, backward, gradient clipping and AdamW paths. The packed input and position value buffers each contain 768 int64 entries (6,144 bytes). Input offsets have 113 int32 entries (452 bytes), while position offsets have 113 int64 entries (904 bytes). The response loss mask remains 112 × 2048 int64 (1,835,008 bytes). The synthetic packed old-log-probability output is 768 float32 entries (3,072 bytes), restored to 112 × 2048 float32 (917,504 bytes) before comparison on the 384 targets. These buffer sizes exclude allocator overhead and model state. The test substitutes a small differentiable CPU module and a synthetic old-log-probability worker. Two calls advance each required optimizer's step counter to two. The probe exits before any such operation. A separate fixture admits a 3072 + 1024 active-token row, checks position 4095, then rejects one additional prompt token without modifying the trajectory. [Pinned numerical test and printed inventory](../tests/test_matched_training_pinned.py)

## Historical replay and limits

The read-only replay checks published file commitments for all eight stages, including raw traces, audit, configuration, receipts, completion and observability, and checks the local tokenizer bytes against the immutable lock. Rendered prompt lengths matched reported usage in all 694 GRPO and 700 PPO-MC generations. The verified historical conversations had prompt lengths 984–1900. Per-call prompt plus completion lengths were 1009–2032 for GRPO and 1007–1987 for PPO-MC. The three sampled packets had 550 GRPO and 556 PPO-MC fragment rows, 28,841 and 28,249 generated targets respectively, and zero masked response-context tokens in both routes. The replay measures retained histories and reported token counts. Synthetic tensor fixtures supply original IDs and per-token probability evidence for adapter/backward tests. No sampled historical action was reconstructed from decoded text. [Committed calibration measurements](../evidence/phase4b-live-20261125/calibration.json)

```bash
"$READER_HARNESS_PYTHON" scripts/replay_recovery_development.py \
  "$READ_ONLY_PHASE4B_COLLECTION" "$LOCKED_TOKENIZER_JSON"
```

The tests exercise no paid work, live inference, pretrained-model training, final-panel evaluation or cloud operation. They supply no H100 or cgroup peak measurement, recovery throughput, measured checkpoint fit, native model reload or live weight-refresh result. [Outstanding capacity measurements](PHASE4C_RECOVERY_LEARNING.md#future-operation-and-capacity)
