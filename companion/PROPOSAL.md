# Proposed hosted execution for asset-status-v1

**Decision requested: one attempt with a maximum gross exposure of USD 0.02 before credits.** The public rate calculation at the token limits is USD 0.013541 rounded up. Reserve the full two cents through reconciliation, including on rejection, interruption or early stop. Current execution authority is zero.

## Exact source and command

Use a clean checkout of `111ed8518ce28a26bf641376e09ab343b6aa17ce` on branch `hosted-tinker-grpo`, based on main `6228222015f2595e220f7248c8825e28448f720d`. Later proposal-only commits do not identify a different executable. Prepare the renderer using the [credential-free instructions](README.md#see-one-batch-locally). In the existing Workbench, set `UV_PROJECT_ENVIRONMENT` to the compatible reader CPU environment. Use the existing locked groups. Supply `TINKER_API_KEY` through the environment's secret mechanism.

After the separate execution decision, run the following once from that exact checkout. The command requires the local tokenizer cache and a fresh output directory. The deadline covers the producer, fresh interpreter and SDK shutdown.

```bash
timeout --kill-after=5s 20m env -u TINKER_LOG HF_HUB_OFFLINE=1 \
  uv run --locked --inexact --group tinker --group reader-harness \
  python -m companion.hosted run --source 111ed8518ce28a26bf641376e09ab343b6aa17ce
```

## Frozen design

| Setting | Value |
|---|---|
| Service and model | Thinking Machines hosted Tinker, `Qwen/Qwen3.5-4B` standard 64K catalog ID. SDK 0.40.1, Transformers 5.24.1. The server controls the served weights revision. |
| Renderer | Model publisher's `chat_template.jinja` and tokenizer at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Text only, generation prompt on, thinking off, stop token 248046. |
| Task | `asset-status-v1`, data recipe `asset-status-v1/2026-11-20/329`, task source Git blob `2f8e37305e11b8911eba45cf66dcbd420ea1db42`, environment SHA-256 `d683b4ca993f4d014de260fe8ba04afa901b1488be036a633c37b4fec7ef4089`. |
| Training panel | First two ordered train cases. One question per collection, asset-011 at v0 and asset-041 at v1. Four independent attempts per question. |
| Development panel | First two ordered smoke-eval cases, asset-304 and asset-231. One greedy attempt each at frozen initialization v0 and again at reloaded v2. |
| Sampling | One sequence per call, two model turns and one local lookup per episode. Temperature 1 for training and 0 for evaluation, top-p 1, top-k −1. Each prompt ≤512 tokens, each generation ≤96. Oversize prompts stop before submission. Length stops reject the collection. |
| Seed | One experiment seed, 329, also used for LoRA initialization. Per-call seed is 329 + 2 × stream + turn index (0 or 1). Evaluation streams 0–1 repeat at reload. Training streams 2–5 then 6–9 distinguish attempts on the same prompt. |
| Update | Two collections, at most two backward calls and two completed Adam steps. One pass per collected batch, no microbatch replay. Group sample-SD normalization, equal episode weight, clip ratio [0.8, 1.2]. Attention LoRA rank 8, MLP and unembedding disabled. AdamW LR 1e−5, betas (0.9, 0.95), epsilon 1e−12, weight decay 0, gradient clipping disabled. |

The SDK catalog and renderer example identify the selected model. The tokenizer revision pins the client-side text-to-token mapping. [Hosted catalog](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/) · [Non-thinking renderer example](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/) · [Pinned publisher files](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)

## Limits and accounting

| Maximum for the entire attempt | Bound |
|---|---:|
| Episodes and model turns | 2 baseline + 8 training + 2 reload = 12 episodes, 24 turns |
| Uncached prefill, full price even for repeated prompts | 24 × 512 = 12,288 tokens at $0.33/M |
| Sampled output, including rejected results | 24 × 96 = 2,304 tokens at $1.005/M |
| Training, counting whole contexts and conservatively including the shifted position | 16 fragments × (512 + 96) = 9,728 tokens at $0.737/M |
| Model operations | 24 sample, 2 forward/backward, 2 optimizer, 3 sampler-save submissions |
| Client creation | 1 trainer, 3 sampler clients and 2 service sessions across producer and reload |
| Local storage | One exclusive ignored directory, `.local/hosted-tinker-asset-v1`. Keep its small JSON traces, receipts and manifest through billing reconciliation, review within 7 days. No model weight download. |

At published rates the initial all-equal stop costs at most $0.003186 in tokens, counting both baseline episodes and all four rejected training attempts. If only the second group is uniform the bound is $0.008894, including one update. A failure partway counts every submitted operation at its maximum until usage settles. Any further attempt requires a new decision. [Published token prices](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)

One model operation may be outstanding. Application retries, restarts, replacement groups, extra seeds and evaluation expansions are zero. SDK sampling retries are disabled. A process-scoped guard counts the pinned SDK's HTTPX sends before forwarding them, including control traffic and SDK-internal retries. Producer cap 400 sends, reload cap 100. Model-operation caps above include attempted submissions. A transport failure or unsuccessful HTTP response on a model or creation submission closes the gate to further such requests. Internal metadata polls share the total limit. Each awaited model/metadata operation has a 120-second timeout. On timeout, an accepted remote operation can still settle. Hold the entire $0.02 and reconcile the journal with posted usage before releasing it. [Submission guard](limits.py)

Use the existing free checkpoint storage plan only. Budget three private sampler snapshots and 3 GB in aggregate. Request a 3,600-second TTL at each save and read back type, source/step metadata, size and expiry. A single save over 1 GB stops progression. The SDK only reports size after creation. Before dispatch, confirm sufficient free-plan headroom for this allocation, including provider checkpoint behavior at quota. Review expiry one hour after the last save. Historical retention decisions are independent. [Storage limits and billing](https://tinker-docs.thinkingmachines.ai/administration/checkpoint-storage-pricing/) · [Sampler checkpoint TTL](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/)

The owner reports a $250 grant that may expire December 15, 2026. Balance, expiry, eligible operations, effective rates and storage plan have not been verified. Confirm those terms, model access and that this envelope fits the gross ceiling before submitting the first model request. If they cannot be established, leave the allocation unstarted. The earlier $50 discussion grants no authority.

## Acceptance and stop evidence

1. Read the frozen baseline from two complete, independently scored episodes on v0. For each training group require four distinct episode identities on the scheduled case, two completed turns each, exact prompts, served sequence IDs, generated IDs, finite aligned behavior log probabilities, lookup observations, rewards and one server-confirmed sampler path. Invalid evidence ends the attempt. A completed wrong answer earns zero.
2. If a group has four identical rewards, preserve every attempt and return `constant_group` with its completed update count. At the first collection this means zero backward, zero optimizer steps and no reload. This is a valid bounded stop, short of functional qualification.
3. For full qualification require two informative groups, two finite backward results with aligned output rows, two successful optimizer acknowledgments and sampler saves v0 → v1 → v2 from the same trainer. Check the server-reported sampler path before generation. Use each policy only for its assigned collection.
4. The producer writes the final sampler URI, metadata, baseline scores, source, renderer, version and process ID, then closes its session. A new interpreter checks the manifest digest and reads the same checkpoint metadata from the service. It creates a new sampling client from that URI and evaluates the identical two cases. Require a distinct PID, path readback, two final verifier results and the paired per-case scores. This qualifies inference reload. Resuming optimizer state would require a training-state checkpoint.
5. Inspect `events.jsonl` for submissions, actual tool calls and outcomes, per-fragment masks/weights and update acknowledgments. Report how many lookups succeeded and whether either answer changed. Use `result.json` for the bounded stop or paired comparison. On exception or deadline, retain the partial journal and report the last acknowledged operation as incomplete. Behavioral improvement is not an acceptance threshold.

Nine offline tests exercised the scheduled cases, target shift, prompt/observation masking, separate episode weights, group advantages, rejection paths, stale sampler, early stop, request cap, checkpoint identity and subprocess handoff. The optimizer and service returned scripted responses. Hosted sampling, server loss execution, model access, storage size, checkpoint metadata/expiry and actual fresh-client reload remain unmeasured.
