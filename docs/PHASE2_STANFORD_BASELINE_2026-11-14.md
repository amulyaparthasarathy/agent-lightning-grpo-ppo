# Stanford Modal bootstrap baseline, 2026-11-14

**Status:** OPERATIONAL PASS for the frozen 64-case development evaluation. This is a historical evidence record from owner-supplied terminal output, **not** a fresh independent read of Modal, W&B or individual trajectory files. It does not qualify a matched PPO/GRPO experiment or establish bitwise replay.

## Execution identity

| Field | Recorded value |
|---|---|
| Workspace/profile | `amulyasp` / Modal environment `main` |
| Working Volume | `cs329z-agent-lightning-working-stanford-v2` |
| Modal Volume ID (from launcher output) | `vo-SfQGusN1IqkAw5A3o0fiE7` |
| Run / action | `stanford-v2-base-01` / `baseline-eval` |
| Algorithm identity | `grpo`, `historical-grpo-v1`; **evaluation-only**, not GRPO training |
| Model profile | `llama3b` |
| Source commit | `0d84e57e2f72573a2b57fd3526f2e9d823cfb6fb`, tracked clean according to receipt |
| Modal App | `ap-Br2pitYIme71fgSY2sxOBE` |
| Modal FunctionCall | `fc-01M7E4SB0JG5N6YFDMF6YE8X7Y` |
| W&B entity/project | `amulyap-stanford-university/cs329z-agent-lightning` |
| W&B run ID | `c4d590253b711105` |
| Start / finish | 2026-11-14 23:03:01.643207 UTC / 23:10:04.376001 UTC |
| Result | `completed`, `exit_code=0`; `resolved_config_saved=true` |
| Checkpoints | None created; none restored |

The supplied `runs/stanford-v2-base-01/result.json`, `submissions/stanford-v2-base-01/receipt.json`, and `runs/stanford-v2-base-01/observability.json` agree on the run identity. The owner also inspected `runs/stanford-v2-base-01/logs/trainer.log` and supplied these excerpts:

```text
AglRolloutManager: completed=64/64 succeeded=64 failed=0
Initial validation metrics: {'val/reward': 0.40625, 'val/n_rollouts': 64, ...}
step:0 - val/reward:0.40625 - val/n_rollouts:64 - val/n_rollouts_w_trace:64 - val/n_rollouts_w_reward:64
wandb: val/reward 0.40625
```

Therefore the observed binary total is **26/64 = 40.625%** with 64 reported traces and 64 reported rewards. The log reports W&B logging and trajectory-artifact uploads, but this closeout did not independently read the hosted W&B run or its finalized artifact manifests.

## Comparison with historical baseline

The preserved Phase 1 `base-01` score was **27/64 = 42.1875%**, compared with **26/64 = 40.625%** in Stanford bootstrap. The difference is **one case / -1.5625 percentage points**, not evidence by itself of drift or a regression. No case-by-case alignment, tensor-level identity, hosted history reconciliation, or independent semantic rescoring was performed here. Temperature-zero behavior alone is not a bitwise-replay guarantee.

Do not replace the original baseline or combine the two scores. Preserve both observations and investigate individual discordant cases only if needed for a specified research question. The 64-case panel is **development/legacy**. It must not be silently reused as the new, untouched campaign final holdout.

## Stanford environment qualification and limits

The owner reported:
- The new workspace Secret `agent-lightning-credentials` exists, with required key names `HF_TOKEN` and `WANDB_API_KEY`; values were neither disclosed nor inspected here.
- The malformed Stanford Volume v1 was superseded by **v2**. v2 listings show actual directories with two Llama `.safetensors` shards, tokenizer/model metadata, six dataset files and the dataset manifest. Local metadata, shard-size, tokenizer-hash and dataset-file-hash checks passed. Independent post-upload SHA256 of **both remote weight shards** was not yet demonstrated.
- The repaired cold CUDA/VERL image build completed under PR #23, image `im-qPPicZgz64ekqF9AR9UKz1`, app `ap-hFyWMIDVXhgYXjQFMm21tv`. Successful build does not by itself establish final runtime/hosted-image digest, but this evaluation subsequently completed.
- The original failed build image `im-fc2RdjhSAGjfhNX3zuDS5W` and malformed Volume v1 remain distinct historical failures, not fixes to overwrite or erase.

The evaluation completed end-to-end with no training updates or checkpoint creation. This is **not** proof that the common PPO/GRPO retention and loss-normalization protocol, deterministic run seeds, 20/60-step PPO, parallel supervision, 256-case final holdout or W&B grouping are implemented. Those are separate PR B gates.

## Disposition

**Do not repeat `stanford-v2-base-01` merely to create a new experiment baseline.** Future matched pilots may include their own fresh-base, no-update evaluations because their training/evaluation protocol will differ; preserve the original Stanford bootstrap as the infrastructure reference, not the study-level control. Continue with the versioned matched-study protocol and separately authorized jobs. No cloud work, secret modifications, volume deletions or training were performed to create this documentation record.
