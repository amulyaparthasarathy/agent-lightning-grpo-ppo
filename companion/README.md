# One hosted-Tinker GRPO update, twice

An asset lookup takes two model calls. First the assistant writes the arguments for `lookup_asset`. The harness executes the existing lookup and appends its result to the conversation. Then the assistant writes its final JSON answer. The independent asset verifier compares that answer with the requested asset's record. A valid lookup can still lead to a reward of zero. A malformed call produces a typed error observation. A truncated generation or missing probability ends collection without assigning a reward. [Task and verifier](../experiment/asset_task.py) · [Companion harness](hosted.py)

## See one batch locally

Run these commands from the repository root with Python 3.12 and uv. The first command downloads four public tokenizer/configuration files. The second uses the cached renderer, the real task and SDK tensors, and local service doubles. The locked `reader-harness` group supplies Jinja for the model's chat template.

```bash
uv run --locked --inexact --group tinker --group reader-harness python -m companion.hosted prepare-renderer
env -u TINKER_API_KEY HF_HUB_OFFLINE=1 uv run --locked --inexact --group tinker --group reader-harness python -m unittest -v companion.test_hosted
```

Expect `eos=248046` and nine passing tests. Predict how much weight an episode gets if it has two sampled tokens in one fragment and one in another. Inspect the synthetic four-attempt fixture locally.

```bash
uv run --locked --inexact --group tinker --group reader-harness python - <<'PY'
from companion.grpo import training_batch
from companion.test_hosted import group
_, rows, _ = training_batch(group(), "case", "v0")
for r in rows:
    print(r["sequence_id"], sum(r["mask"]), round(r["weight_per_generated_token"], 6))
PY
```

| Attempt / fragment | Generated targets | Weight per generated token | Fragment's share |
|---|---:|---:|---:|
| 0/0 | 2 | 1/8 | 1/4 |
| 1/0 | 2 | 1/12 | 1/6 |
| 1/1 | 1 | 1/12 | 1/12 |
| 2/0 | 2 | 1/8 | 1/4 |
| 3/0 | 2 | 1/12 | 1/6 |
| 3/1 | 1 | 1/12 | 1/12 |

The two fragments of attempt 1 sum to 1/4. Each attempt's final reward supplies one advantage, computed by subtracting the group's mean and dividing by its sample standard deviation plus `1e-6`. For the fixture's rewards `[0, 0, 1, 1]`, the first two advantages are approximately −0.866 and the last two are +0.866. As an optional alteration, give all four attempts reward 1. The batch then has zero advantage at every position and the runner stops before backward. [Batch construction](grpo.py)

For a prompt `[31, 32, 33]` and sampled tokens `[42, 43]`, the model input is `[31, 32, 33, 42]`. The next-token targets are `[32, 33, 42, 43]` and the mask is `[0, 0, 1, 1]`. The first action is predicted at the last prompt position. On the next turn the tool observation enters the new prompt. Its targets get zero advantage. Each action keeps its original token ID and behavior log probability, even if rendering the next prompt changes the previous turn's serialization. Each served model call produces its own fragment.

The trainer sums the token-wise clipped probability-ratio objective. Multiplying each generated token's advantage by its episode weight implements the reduction in the table. Tinker's loss name is `ppo`. The group supplies the baseline here, with no value model or reference-policy term. The runner waits for backward and Adam before saving the next sampler snapshot. [Tinker loss](https://tinker-docs.thinkingmachines.ai/tinker/losses/ppo/) · [Historical update lifecycle](../docs/learning/06-one-complete-training-update.md)

## What the comparison measures

The historical asset run used Llama-3.2-3B-Instruct, the AutoGen/MCP harness, the VERL/FSDP actor update and vLLM on Modal. Both its baseline and reload scored 64/64. This companion reuses the asset registry, data generator, lookup and strict terminal verifier. Its two-call text protocol fixes one lookup slot and returns the observation in a user message. It uses hosted Qwen3.5-4B with rank-8 attention LoRA, a different tokenizer, indexed sampling seeds, the hosted AdamW implementation, and a two-case development comparison. The numerical kernels, trainable parameters, action format and case counts differ. A successful run would qualify the collection-to-reload interface. Two development scores cannot establish improvement, reproduce historical trajectories, or rank the backends. [Historical qualification](../docs/PHASE3_QUALIFICATION_2026-11-24.md) · [Model and rendering example](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)

Local tests use scripted model outputs, including a deliberately wrong final answer. They exercise the real renderer, verifier, tensors and a subprocess boundary. The hosted optimizer, checkpoint service, served probabilities and model behavior await a separate decision. [Exact execution proposal](PROPOSAL.md)
