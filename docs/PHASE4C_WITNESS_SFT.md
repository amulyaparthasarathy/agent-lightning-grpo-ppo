# Can one supervised pass improve recovery witness selection?

P02’s terminal greedy policy rejected all 24 development requests. During training it produced 67 strict ready successes, including 59 with positive relative advantage. All 28 decision-correct ready failures cited stale support. Final-answer supervision directly trains the decision, the observed identifiers and the inclusion of one eligible run. The three public tool exchanges remain conditioning context. A fresh baseline matters because P01 and P02 agreed on baseline decisions but differed on one witness. [Pilot diagnosis](PHASE4C_PILOT_LEARNING_DIAGNOSIS.md)

| Stage in order | Original base or artifact | Data | Result |
|---|---|---|---|
| Fresh-base greedy evaluation | Verified original Llama-3.2-3B-Instruct | 24 development requests in frozen panel order | Independent traces, strict rewards and diagnostic rows |
| Supervised pass | New load of that same original base | 32 demonstrations, one epoch | Terminal projection weights and completion receipt |
| Terminal-reload greedy evaluation | Distinct worker verifies and overlays terminal weights on original base | Same 24 development requests in the same order | Per-class strict successes and assessability |

**The terminal policy passes when** at least 8 of 12 actual ready cases and at least 8 of 12 actual not-ready cases must earn strict reward 1, and all 24 post-SFT answers must be assessable. An assessable answer has a decision correctness judgment under the unchanged verifier, including a valid target and evaluation time. Baseline performance, score delta, family results and tool diagnostics are descriptive. Report the terminal result even if it misses the gate. There is one seed, one terminal artifact and one post-test. This experiment can measure behavior on this development panel. Generalization, the mechanism behind P02 and persistence under RL remain open.

## Demonstrations

The committed recovery split manifest has canonical JSON SHA-256 `c7d07942dda1ee81b8797ce9a59255e7f26bca712e668c18cd1a936d83d6e8ab`. Selection verifies the ordered training transport rows against its content hash, matches all 128 training IDs and excludes every ID in the other split manifests. Only training cases supply selection attributes. Within each family, cycle through lexically ordered variants, ranking members by ascending SHA-256 of compact sorted JSON `[2001,"selection",case_id]`. Sort the resulting 32 cases by the corresponding `"order"` rank. No trajectory outcomes enter either ranking. [Split manifest](../evidence/witness-sft/source-manifest.json) · [Selection and strict replay](../witness_sft/data.py)

| Actual class | Training family | Count |
|---|---|---:|
| ready | recent usable | 8 |
| ready | newer failed, older usable | 8 |
| not_ready | stale success | 4 |
| not_ready | recent incomplete | 4 |
| not_ready | recent integrity failure | 4 |
| not_ready | empty history | 4 |

The set includes five exact-boundary positives, five multi-witness positives, a just-stale negative and two pending-integrity negatives. The solver observes the asset, policy and complete as-of inventory. The demonstration adapter keeps the qualifying run with the largest `(recovery_point_at, run_id)` on positive cases and emits an empty list on negatives. Strict replay of each target against its observed ledger earned reward 1. The model receives the ordinary wire messages, recovery tool definitions and final target. Case IDs, families, verifier labels and selection metadata live in the manifest.

| Commitment | SHA-256 |
|---|---|
| Ordered JSONL training bytes | `15bcd21eefc2e8f7a8bfdc77209f6be3f333f5b67285ced35161531be9db903b` |
| Canonical demonstration manifest | `02dc7c7607201a75613a78d14e940a4b94713fc1131d6d80948575bc1bd404f2` |
| Pinned tokenizer.json | `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4` |

All 32 case IDs, in optimizer input order, and both class totals are available in the manifest. [Manifest](../evidence/witness-sft/manifest.json) · [Training bytes](../evidence/witness-sft/demonstrations.jsonl)

## Fixed update and reload

Use the nine-file prepared model lock and original revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`. The production renderer adds the original fixed-date tool preamble and a single continuation header after the third observation. Encode without automatic special tokens, check full-string tokenization against prefix plus target, and supervise the structured JSON and one `<|eot_id|>` (128009). Prompt lengths were 1,224–1,564 tokens, target lengths 72–87 and complete lengths 1,299–1,645. The 2,540 supervised tokens include 32 termination tokens. Reject sequences over 2,048. Use right padding with attention zero, ignore label −100, no packing or truncation.

Only the last decoder block’s query and value projection weights train (12,582,912 parameters). This limits trainable state while allowing the last attention layer to change how it reads the inventory. Frozen base weights are BF16, updated parameters and AdamW state are FP32, and GPU forwards use BF16 autocast and SDPA. No activation checkpointing or KV cache during training. The frozen lower layers have no trainable inputs. Context in the trainable block can still influence gradients. The LM head projects only positions needed for the shifted targets.

The loss averages causal negative log probabilities within each example, then weights examples equally. Four successive single-example backward passes each contribute one quarter of their own mean. AdamW uses constant LR 0.0001, betas (0.9, 0.999), epsilon 1e−8, zero weight decay, and global gradient-norm clip 1. One epoch gives **32 ÷ (1 × 4) = 8 optimizer updates**. PyTorch, the greedy serving engine and Python hash seed use 2001. Selection and order use the named SHA-256 domains above. There is no sampler shuffle, dropout in the pinned model or checkpoint selection. [Objective and schedule](../witness_sft/training.py)

The artifact contains FP32 replacement tensors for exactly those two projections in Safetensors and a completion manifest written last. The separate stage receipt anchors the manifest digest. The binding includes class, source commit, model revision, all prepared-file commitments, tokenizer, demonstration bytes and manifest, recipe, producer, tensor names/shapes/dtypes, file length/hash and completed update statistics. The consumer re-verifies the base, external commitment and tensor inventory before applying either tensor. It casts the final projections to the serving model’s BF16, exports a temporary full HF model in worker scratch and supplies original decoding metadata, tokenizer and template. The producer and consumer must have distinct provider call **and** container IDs. Fixture bindings additionally fingerprint the unchanged random base. [Artifact format](../witness_sft/artifact.py) · [Worker](../witness_sft/worker.py)

Both evaluation stages run the unchanged recovery agent, AutoGen loop, MCP tools and strict verifier. The serving change for both arms is one loopback vLLM server on the stage worker, alias `auto`, eager BF16, context 4,096, max generation 1,024, one sequence at a time, Llama JSON tool parser, temperature 0, top-p 1 and top-k −1. An authenticated per-case loopback sink confirms the one reward post. Trace replay checks the served completions, model-visible history, observations, classifier output and strict reward. Only the ordinary request enters each episode. The training transcript is never passed to the evaluation agent. [Serving and replay](../witness_sft/evaluation.py)

## Local reproduction

Work from a clean checkout. Point `WITNESS_TOKENIZER` at a local copy of the pinned tokenizer.json. These commands install CPU packages and use locally initialized weights. The two environments keep the Modal client and GPU training stack separate.

```bash
UV_PROJECT_ENVIRONMENT=.venv-reader-cpu uv sync --locked --only-group reader-harness
uv venv --python 3.12 .venv-sft
uv pip install --python .venv-sft/bin/python 'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-sft/bin/python 'transformers==5.22.2' 'tokenizers==0.23.3' 'jinja2==3.1.6'
export WITNESS_TOKENIZER=/path/to/pinned/tokenizer.json
export HF_HUB_OFFLINE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1
.venv-reader-cpu/bin/python -m witness_sft.data --reproduce .local/reproduced-witness-sft
cmp evidence/witness-sft/demonstrations.jsonl .local/reproduced-witness-sft/demonstrations.jsonl
cmp evidence/witness-sft/manifest.json .local/reproduced-witness-sft/manifest.json
.venv-reader-cpu/bin/python -m unittest -v test_witness_sft
.venv-sft/bin/python -m unittest -v test_witness_sft_cpu
python3 -m witness_sft.dispatch plan
```

Use a fresh output directory for each reproduction. The CPU regression runs a full 32-example pass through a tiny Llama with the true vocabulary. It compares the shifted loss and gradients to dense log-softmax on an unequal-length padded batch, verifies that only the two projections changed, and checks exact state and logits after reload. Negative tests cover incomplete identities, wrong frozen bases, manifests, missing weights, split overlap, altered user inputs and fixture admission. Independent review also reconstructed all 32 requests through the production harness with scripted local responses and compared the captured final conditioning against the saved demonstrations. [CPU tests](../tests/test_witness_sft_cpu.py) · [Contract tests](../tests/test_witness_sft.py)

### Measured local results

On implementation commit `eeafcc10099a3a8cc2aef575e959f4f391864677`, all three new pinned CPU tests passed. Preclip gradient norms over eight actual optimizer steps were 0.013290269–0.016530192. Both projection weights changed. All model tensors and test logits matched after restore. A separate run saved a fixture and reproduced its logits in a fresh CPU interpreter. These measurements use a one-layer, width-16 random Llama with the full tokenizer vocabulary.

The seven SFT contract tests, seven pilot diagnosis tests, four scripted historical study tests, 202 reader offline tests and 74 pinned CPU regressions passed with zero skips in their completed runs. An initial invocation of the 74-test gate had two failures, one error and one skip because the IPC wheel, Transformers comparison wheels and padding fixture were missing from its test paths. Restoring the existing local fixture paths resolved all four. Source comparison against main found no changes in the historical evidence, numerical runtime or study declaration.

## Proposed new reservation, currently disabled

Reserve **one lane on one H100 80 GB, 16 CPU cores and 192 GiB host RAM**, sequential single-use workers with both CPU and memory capped. The starting image must be an identified prebuilt reviewed reader training image. Mount the identified original-base volume read-only and a separate dedicated writable SFT volume. Source files are attached as a deferred runtime mount. Cold image builds, new base downloads and model-provider service calls have a zero allocation. If the image or original base is unavailable, close the attempt.

The model contains about 6.0 GiB of BF16 weights. The 48 MiB of FP32 projections need another 48 MiB for gradients and 96 MiB for Adam moments. At maximum measured length, one BF16 intermediate of shape 1,645 × 8,192 takes about 26 MiB. At 87 target positions, one FP32 logits buffer takes about 43 MiB. Allow several such buffers, SDPA workspaces, driver and allocator overhead. The estimate assumes a frozen prefix, microbatch one, no optimizer state for frozen weights and selected-position output projection. vLLM separately requests a 0.5 device-memory fraction. The host stages the base and merges weights in temporary scratch (allow 20 GiB scratch). Record the actual trainer allocator peaks, base-load, artifact-save and merge times from the stage receipts and provider counters; GPU fit, effective throughput and serving compatibility require measurement on this path.

| Phase | Estimated active time | Deadline |
|---|---:|---:|
| Startup per worker | 2 min | 7 min per worker |
| Fresh-base evaluation, including server startup | 12 min | 30 min |
| Training and saving | 8 min | 30 min |
| Reload and evaluation, including server startup | 13 min | 30 min |
| Total | 39 min including three startups | 111 min of worker/startup exposure, 120 min allocation window |

Each server has a 7-minute readiness limit inside its stage. Dispatch deadlines allow 30 seconds for clock differences and can only shorten the reservation. Per-stage execution, worker startup and the allocation clock have separate limits. These times are engineering estimates. The pilot’s GRPO memory and elapsed times do not qualify this SFT configuration.

On November 26, 2026 the public H100 SXM5 rate was $0.001097/s, CPU $0.0000131/core-s, memory $0.00000222/GiB-s, volume storage $0.09/GiB-month and egress $0.04/GiB. At full allocation the worker costs $6.238224/h. Estimated compute is **$4.05**, with expected gross about **$4.10** including small storage and transfer. At 111 minutes the full-rate compute bound is **$11.55 rounded up**. The proposed new cumulative gross ceiling is **$25**, before credits. Reserve $12.50 for compute and shutdown rounding, $1 for storage, $1 for collection/egress and $10.50 for delayed billing and reconciliation. The supervisor runs on the local client. A cold build would require a separate proposal. [Modal public pricing](https://modal.com/pricing) · [Volume billing and delayed storage charges](https://modal.com/docs/guide/volumes)

Retain SFT evidence and the roughly 48 MiB terminal artifact until **seven days after the actual UTC allocation opening**. The opening, completion and artifact receipts share the review instant. The worker checks allocation output against 1 GiB. At 1 GiB the seven-day storage estimate is $0.021 using a 30-day month. Retention continues to incur charges until a separate disposition decision; delayed storage billing can continue for four days after a later deletion. Review the dated horizon and reconcile gross posted charges, retained bytes, transfer, delayed exposure and residual holds before closing the ledger. Provider billing and the final retention disposition determine actual cumulative spend.

**Future operator commands.** The checked-in authorization gate rejects `run` before reading private files or contacting Modal. Activation requires a separately reviewed owner public key, a clean tested source and an expiring off-tree Ed25519 signature over the exact plan and owner profile. A change to source or owner invalidates the signature. The private owner JSON has top-level fields `modal, base_path, image_id, registry_id, output_id, base_id`; its `modal` object contains `profile, workspace, environment, volume, base_volume, registry, app`. Names and IDs come from the new owner reservation. Use a dedicated output volume and registry, an existing prebuilt image and an existing original base directory. The client performs authenticated workspace and exact handle checks and opens its durable one-use reservation before the first submission.

```bash
# Offline, from the reviewed activation checkout
uv sync --locked
.venv/bin/python -m witness_sft.dispatch plan > "$PRIVATE_DIR/plan.json"

# After separate spending and owner authorization, on the signing workbench.
# SFT_EXPIRES_EPOCH is the authorized UTC expiry in Unix seconds.
.venv/bin/python - "$PRIVATE_DIR" "$SFT_EXPIRES_EPOCH" <<'PYTHON'
import json, pathlib, sys
from experiment.recovery_task import digest, dumps
root = pathlib.Path(sys.argv[1])
read = lambda name: json.loads((root / name).read_text())
body = dict(plan_sha256=digest(read('plan.json')), owner_sha256=digest(read('owner.json')),
            expires_epoch=int(sys.argv[2]), ceiling_usd=25)
(root / 'reservation-body.json').write_text(dumps(body))
PYTHON
openssl pkeyutl -sign -rawin -inkey "$OWNER_SIGNING_KEY" -in "$PRIVATE_DIR/reservation-body.json" -out "$PRIVATE_DIR/reservation.sig"
.venv/bin/python - "$PRIVATE_DIR" <<'PYTHON'
import base64, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
permit = dict(body=json.loads((root / 'reservation-body.json').read_text()),
              signature=base64.b64encode((root / 'reservation.sig').read_bytes()).decode())
(root / 'permit.json').write_text(json.dumps(permit))
PYTHON
# On the owning client, with the authenticated, signed owner profile.
MODAL_PROFILE="$SFT_PROFILE" .venv/bin/python -m witness_sft.dispatch run \
  --owner "$PRIVATE_DIR/owner.json" --permit "$PRIVATE_DIR/permit.json" \
  --out "$PRIVATE_DIR/receipts"
```

Archive the call IDs and receipt files on every exit. On timeout, preemption, incomplete evidence, failed identity or lost submission acknowledgement, retain the reservation and reconcile the exact calls and containers. The owner client records intent before spawn, signals the stop registry on submission/result failure and cancels the known call. A lost acknowledgement remains uncertain. There is no resubmission, replacement, additional evaluation or automatic deletion. Close after the third stage regardless of the behavioral result. A later RL experiment requires its own decision.

The closed pilot still has $9.50754026 posted gross, $52.43245974 in conservative holds, $88.06 unencumbered under its old ceiling, and **zero further dispatch authority**. Historical review remains December 2, 2026 UTC; pilot review remains December 3, 2026 UTC. P02 ACTUAL source remains `17889f37651d57653cfc0dbe12d06304ac27371a`. [Pilot closeout](PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md)
