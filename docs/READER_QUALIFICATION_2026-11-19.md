# Reference qualification — 2026-11-19

**The reference Llama/calculator workflow completed real baseline evaluation, two-update training, native checkpoint saving and fresh-worker reload evaluation for both GRPO and PPO-MC.** Same-source CPU preflight and real model/tool integration passed. GRPO observed **zero informative reward groups**; this is operational qualification, not evidence of informative GRPO policy learning or reliable accuracy improvement. PR #3 remains draft and unmerged. The alternate-model and alternate-task milestone rows remain open.

## Executed source and allocation

The successful fourth allocation used public source `5daff64cefd348fb731094cf465b223ddceb5fb2`, frozen throughout execution. Later report commits document that execution; they did not execute it. Base: `ac67aa030b3ba693d626f8ab0ef51e795f5e48c3`. Schema: `reader-reference-v1`; numerical protocol: `matched-rollout-mean-v2`.

| Binding | SHA-256 |
|---|---|
| Source inventory | `2339fb4309e19c579d58821108d4dd7ba1f466c1bd1fdbe7ec7504d21c0306c9` |
| Packaged files | `3096cf38b1744f39f11484482dc28892f773cf18197dc5d2028d219dc1437527` |
| Configuration | `7067463b82cbea680ca9120f2ffed43e80ddc2bf36f60f3a1d4c7497abb66734` |
| Preparation plan | `64b2f70f652a3ad5aa3a66d55f411e2ea37516a86a37b9892d97b8d52baa15e4` |
| Asset-bound training/preflight plan | `a3128740ba4c2133e049bd80e9bdb471aa1ad56df3158479d1594b4e0ab23cc3` |
| Asset manifest, canonical | `037d4af442b7c2137b2211c92ba44a277213027a2c0845c36f28885e72be1706` |
| Model manifest file | `d236d3a67b232fced62586cfe2967e21c5cc15089e14574bd9ca979943062677` |
| Data manifest, canonical | `aef2a3934a6d44f5fb62e16ec10cf5be1abdeba36cd11009e3d6087f8b96d6f7` |
| Task identity | `495398f731e5961f45e1a36b18ff0f3895e1a2e9d1d14dfa946d4acb29306684` |

Authenticated Modal ownership: `amulyasp`, environment `main`. Experiment `reader-reference-20261119-r04`, attempt 4; Volume `reader-reference-20261119-r04-volume` (`vo-DrCbZBe9wn0qPL2MDhaXi0`); registry `reader-reference-20261119-r04-registry` (`di-lBs3ZElyLUOsi03cA6vW6o`). Secret `reader-reference-credentials` contains only `HF_TOKEN` and `WANDB_API_KEY`. W&B project `amulyap-stanford-university/reader-owned-reference` is private. Only those intended credentials/project were created when previously absent; no unrelated or historical resource was modified.

Owned model/data paths are `/data/reader-assets/llama3b-20261119-r04` and `/data/reader-assets/calculator-20261119-r04`; acquisition staging is `/data/reader-staging/preparation-20261119-a04`. Results live under `/data/reader-runs/reader-reference-20261119-r04/a04`. Existing successful preparation was inspected, but the committed contract requires fresh Volume/staging bindings and has no verified transfer interface. This allocation used fresh acquisition rather than reassigning prior manifests or adding a storage design.

## Acquisition and CPU preflight

Model: `meta-llama/Llama-3.2-3B-Instruct@0cb88a4f764b7a12671c53f0838cd831a0843b95`. Both weight shards (6,425,529,048 bytes), complete file inventory, configuration, architecture, tokenizer/token IDs and fixed chat template passed locked content checks. Read-only gated Hugging Face access and remote W&B project access passed.

Calculator acquisition source hashes:

- Train Parquet: `d3cfeb2b869e76500f5734c4a7d3ded97ac14299a6636d30d0d461574bf10c16`.
- Test Parquet: `7bad3039690001c01d73c79a585e07ea3ef98c1698e667540e9a419eb81caf86`.

Deterministic panels contain 256 training cases, 16 nested training-smoke cases, 64 final evaluation cases and eight separate evaluation-smoke cases. Schema, ordered IDs, duplicate/conflict checks, split isolation and privileged-label separation passed. The three historically locked main output files match byte-for-byte. Historical raw acquisition hashes are absent from the old lock; that comparison is unavailable. Fresh manifests have distinct paths, timestamps and source bindings. This is a methodology rerun, not a reassignment of historical results.

Preparation call `fc-01M7TXAE5N01K0Y9GXEMK4DVN8` and CPU preflight call `fc-01M7TXEE8ZP06XYT7D4DCNTW5Y` completed. The actual preflight receipt says `status: passed` for the exact asset-bound plan above. It independently checked acquired bytes and canonical Volume paths, package/source identity, tokenizer/template, model access and W&B access. Real stdio calculator execution checked the advertised `calculate` schema (`expression: string`), `(2 + 3) * 4` → `20`, and `unknown_reader_symbol + 1` → the expected tool error. Bounded workbench shutdown and owned process-group cleanup passed.

CPU/build bounds remained: heavy builder 8 CPUs/64 GiB/7200 seconds; finalizer 4 CPUs/16 GiB/1800 seconds; overlay 4 CPUs/16 GiB/900 seconds; preparation 4 CPUs/16 GiB/3600 seconds; preflight 4 CPUs/16 GiB/900 seconds. The CPU campaign stayed within its 4.5-hour operator stop point. Provider-default build layers were not assumed hard-bounded by those settings.

Image `im-HJDb9JmQmsbLZ7zUUxjFwu` was used for preparation, preflight and GPU stages. Provider build/package receipts were retained; cache-hit status remains unverified. Backend source guards verified 36 files against Agent Lightning `cbc56e41e0eab643aa71d195e92e5bba652fa683` and VERL `7aed6b230776f963fa09509c10d9c3a767d1102c`. Actual runtime: Python 3.12, Agent Lightning 1.0.3, VERL 0.8.0, vLLM 0.22.0, Torch 2.11.0+cu130, FlashAttention 2.8.3, Transformers 5.22.2, AutoGen 0.7.5, MCP 1.30.0, calculator 0.2.1 and PyArrow 25.0.1. Dependency qualification remains `diagnostic_with_reviewed_dependency_exceptions`: exactly six previously reviewed conflicts, no identity drift, and pip-check exit 1. This is not a clean dependency closure or a guarantee of future cold-build compatibility.

## Live GPU outcomes

All stages used one H100 80GB HBM3, 16 CPUs and 196608 MiB host memory, concurrency one. Every worker passed real CUDA/Torch/FlashAttention checks. The 600-second startup, 3600-second stage and 22200-second supervisor bounds were preserved. No GPU retry, cancellation or overlapping stage was needed.

| Algorithm / stage | Outcome | Reward / correct | Valid model → tool → observation → terminal reward chains | Duration, seconds |
|---|---|---|---|---:|
| GRPO baseline | Completed; zero optimizer calls | 27/64 (42.1875%) | 53/64 episodes | 466.69 |
| GRPO training | Completed; two actor calls | Two complete 32-rollout collections | 43/64 episodes | 733.58 |
| GRPO fresh reload | Completed; zero optimizer calls | 25/64 (39.0625%) | 53/64 episodes | 490.36 |
| PPO-MC baseline | Completed; zero optimizer calls | 27/64 (42.1875%) | 52/64 episodes | 476.26 |
| PPO-MC training | Completed; two actor and two critic calls | Two complete 32-rollout collections | 39/64 episodes | 888.98 |
| PPO-MC fresh reload | Completed; zero optimizer calls | 28/64 (43.75%) | 53/64 episodes | 564.57 |

Durations are supervisor submission-to-completion intervals, including startup, validation and checkpoint handling; they are not exact billed GPU seconds. The six intervals total 3620.44 seconds, and the supervisor campaign lasted 3632.26 seconds. All evaluation panels completed all 64 declared cases with confirmed terminal rewards and no failed/incomplete episodes. Tool errors remain in the traces and do not automatically imply failed episodes; successful integration and task correctness are separate measures.

| Training evidence | GRPO | PPO-MC |
|---|---|---|
| Actor / critic optimizer calls | 2 / not applicable | 2 / 2 |
| Weight refresh steps | 0, 1, 2 | 0, 1, 2 |
| Rollouts per update | 32, 32 | 32, 32 |
| Fragment rows per update | 78, 69 | 71, 69 |
| Valid generated tokens per update | 3500, 2929 | 3649, 2826 |
| Actor losses at steps 1, 2 | 0, 0 | −0.0242688654, −0.247704068 |
| Critic losses at steps 1, 2 | Not applicable | 1.88318260, 2.99223987 |
| Actor preclip gradient norms | 0, 0 | 10.3562679, 13.6775064 |
| Critic preclip gradient norms | Not applicable | 93.0971451, 269.9049072 |
| Varying-reward groups | 0/16 | 1/16; diagnostic, not a GRPO advantage claim |

Each collection used eight question presentations and four distinct complete rollouts per presentation. Runtime checks retained generated-token conservation across fragmentation, observation masking, equal-rollout weighting and valid contexts/rewards. Finite zero GRPO losses/gradients are valid execution evidence. **Informative group-relative task-advantage evidence was not observed for GRPO.** No additional run was selected to replace that unfavorable result. PPO-MC produced finite nonzero actor/critic updates under the unchanged terminal Monte Carlo target/residual-advantage calculation; this does not establish stable critic convergence or learning generalization from two updates.

PPO critic pre-update MSE was 3.51375 then 4.26625. The first target population was constant zero, so explained variance was explicitly invalid; the second had target variance 0.0311751 and explained variance −132.146. On that second collection, the earlier-reward constant predictor had MSE 0.032201. These diagnostics do not show a useful calibrated critic after this short run, despite valid optimizer execution.

## Checkpoints, reload and hosted evidence

| Algorithm | Complete checkpoint inventory | Inventory SHA-256 | Training → reload container |
|---|---|---|---|
| GRPO | Actor; 40,146,400,251 bytes | `674d38fcdef20dcdd2f34717bbb0f734d07884600e4368ce7bef4614012c938c` | `ta-01M7TY14NVSXH7FDEKHWHJ7C1R` → `ta-01M7TYQSBE095PFNFJ7SAZBEMR` |
| PPO-MC | Actor + critic; 78,716,827,965 bytes | `23b5842ac75eb5a9072aebf151eb9911ed01d93446e3af67434fdb92bc3b5faa` | `ta-01M7TZN5S7KGA997GXRX6YXSZR` → `ta-01M7V0GCPYBPXHBXKE4YM73KBR` |

Every role includes native model, optimizer and extra-state files with sizes and SHA-256 hashes; actor inventories also retain serving configuration/tokenizer/template files. Training completion verified exact source/configuration/plan/checkpoint identity, and separate fresh workers loaded and reverified those same final step-2 checkpoints. Evaluation made zero optimizer calls, refreshed step 2 and created no new checkpoints. Provider container identity plus launcher/trainer PID distinguishes processes; repeated hostname `modal` or PID values across containers do not imply reuse. This qualifies **evaluation-only reload**, not resumed training or arbitrary weight export.

<details><summary>Native checkpoint tensor inventories (single rank)</summary>

Names below end in `_world_size_1_rank_0.pt`. Complete role inventories also include the configuration/tokenizer files committed by the canonical inventory hashes above.

| Algorithm / role | File prefix | Bytes | SHA-256 |
|---|---|---:|---|
| GRPO / actor | `model` | 14,427,143,047 | `41dff7e9933e61da9d24beeb9a29a16da9b5c3519d390d111e419fe6000d16b7` |
| GRPO / actor | `optim` | 25,702,025,404 | `dc2438706d5cbe7569987af2055e8d34a1be6c425e8fd71de24b336d873bb641` |
| GRPO / actor | `extra_state` | 15,141 | `c0bc5ad77c5df9f109b679d1c4ca8587d9b42ae44b75263357f38e31c59fd908` |
| PPO-MC / actor | `model` | 14,427,143,047 | `5639f73a413ee1fdd7d78150217417c3ac6e65bc3d6f977c545bbae858e8a5e3` |
| PPO-MC / actor | `optim` | 25,702,025,404 | `87da52c126293996f1134218f76376448026cd31b80034fba75cde7e805bb9a6` |
| PPO-MC / actor | `extra_state` | 15,141 | `4acbc2018acfa79f25690ca72e1735e5657c7ce71ffc5b3e88507c77a8541c49` |
| PPO-MC / critic | `model` | 12,851,145,955 | `3bd0622c9f8eed8e813058a1521c1524848b1b924da83b3f16cd997e0e270a89` |
| PPO-MC / critic | `optim` | 25,702,049,980 | `da50b5cd7b69fe9b5c39a4c577cffcc701985df493e383f2a22ed93c72e2bfc9` |
| PPO-MC / critic | `extra_state` | 15,141 | `ab526a9c1cab88eb9e644a66a5939b3e9a3bd9d10a343304cec2acfc9f721397` |

</details>

Hosted W&B readback found all six runs `finished`: GRPO baseline `6cef67d429bba511318d23b7`, training `bc8b6d90d9edfe0e3ef0d26d`, reload `3b226a2084b8db1a19933964`; PPO-MC baseline `883806cf450f9a8395120131`, training `1e4715dd312373cba98a6e15`, reload `8e73c5be93a6286d77e38d83`. Training history contains both updates and required losses; evaluation history contains the declared single step and score. An initial GRPO hosted read showed only update 1; later readback included update 2 without another execution. Local finish receipts and hosted readback are retained separately.

## Recovery provenance

| Allocation | Public execution source | Observed outcome |
|---|---|---|
| Initial | `e2d12edb87300ec97896c5931a10af2efe91c1b5` | CPU entry failed before acquisition: a physical Modal Volume path could not be made relative to literal `/data`. |
| Recovery 1 | `3fa11d169bbec4fb2b75ff3bcf18c639ef8bc007` | Preparation completed; strict absolute-deadline admission failed before scientific preflight. Exact failure-time clock offset was not retained. |
| Recovery 2 | `d920e0e9ec1fbb6c0b5dfc7c4e18ea6c844368bf` | Preparation completed; preflight rejected physical-versus-logical acquired-source path identity. |
| Recovery 3 / resumed attempt 4 | `5daff64cefd348fb731094cf465b223ddceb5fb2` | Canonical-path correction exercised remotely; CPU preparation/preflight and all six GPU stages completed. |

A bounded earlier CPU diagnostic established nonunique hostnames and motivated provider container IDs. Dispatch reserves 30 seconds inside existing deadlines; strict admission remains. The final path correction uses the same owned canonical staging paths for preparation, preflight and stage checks, retaining symlink/corruption rejection. No scientific guard was relaxed. The owner extended the recovery allowance by three attempts within the unchanged $250 ceiling; only one additional allocation was needed. Failed manifests, calls and logs retain their original source identities.

Recovery-2 bindings remain preserved: source inventory `24ee7a5b7e86970a23b793761be21690c187d4a74e89dd6dfde3410bc419a0ae`, configuration `8a057837dc32e67850b6dd5b9c3493fd06f0519370b8ae89f3cfdbd7b2bf0481`, preparation plan `109df1b08ce46364bdbd5d3cc6611d777bcc77dda8f4fd2ba8ea0b52809ddeab`, training plan `52f2277e60f9f46b712cfe459a7a1e61743883c986f8e669601f746fd24c2d0f`, asset manifest `8829d0fe3de249e49a1841aef3cc8ade637f1751f3f6e3484506f50348fdab74`, model manifest `348c506e93573a74f0c3275ddc97afed3343a73bf009457e5a12649125739519`, and data manifest `7c3a77b9f411530e95bbd8df663a1dcb43e815037e4d3971c69f773788653915`. Its terminal failed preflight was `fc-01M7TWF3PHD95B5QQG9TET422A`.

## Evidence reconciliation, limits and costs

Canonical collection completed with exit 0, **70,824,229 bytes**, 715 local files including the collection report, and no collection errors. Local reconciliation reran exact receipt/source/configuration/asset/checkpoint joins and independently reproduced trace hashes and model/tool chains for all **384 episodes**. It confirmed actual optimizer/refresh/finish counts, finite required losses at both updates, and native role inventories. Hosted numeric history independently matched all required losses and evaluation scores. `evidence_complete: true` describes operational completion; `live_qualification: gaps remain` correctly retains the GRPO informative-signal gap.

The bounded tensor audit contains only the first four fragment rows per collection, not full population tensors. Local checks verified finite values, zeros outside response masks, positive valid-token rollout weights and PPO terminal-return-minus-value residuals in those samples (GRPO: 70 and 88 valid tokens; PPO-MC: 99 and 100). Full-population conservation, grouping and terminal-policy checks, together with the whitening calculation, ran in the frozen numerical runtime; offline numerical tests remain their independent formula checks.

All 384 episode inputs exposed only the question, and every recorded model user message matched the frozen question-plus-output-format prompt. Prepared split isolation and source checks passed; privileged labels enter only the unchanged independent verifier. No label injection, missing delivered reward, failed model request or incomplete episode was found in this collected evidence. Calculator errors and malformed tool output remain recorded; no timeout was converted into a completed terminal-zero example. This does not rule out public-dataset exposure during the base model's pretraining or establish a reward-hacking robustness guarantee.

GRPO training rewards were 4/64 (all within constant groups), PPO-MC 3/64. Tool-error counts by baseline/training/reload were GRPO 23/40/23 and PPO-MC 25/35/21; those returned errors are retained, with integration qualification based on observed successful chains. Paired evaluation: GRPO gained zero cases, lost two, retained 25 correct and 37 incorrect; PPO-MC gained three, lost two, retained 25 correct and 34 incorrect. No statistical/generalization claim follows. Zero GRPO policy-gradient signal also does not prove byte-identical post-optimizer parameters: the resolved optimizer retains weight decay 0.01. The observed score differences are not attributed to task-directed GRPO learning.

`bash scripts/test_reader_offline.sh` passed at the executed correction and again after live execution: CPU accounting demo, historical checks (12 artifact hashes, 24 stage summaries, six runs, three contrasts), 39 PPO-MC tests, 15 matched-study tests and 71 reader/shared tests: **125 tests, no skips**. The suite denies network/credentials and explicitly labels mocked submissions and synthetic assets. A real-filesystem regression covers physical/logical mount aliases while still rejecting corrupt acquired bytes and file symlinks. Separate real local and remote calculator checks passed; the actual GPU evidence above supplies qualification beyond fixtures.

At 23:11 UTC all current preparation, preflight, supervisor and six stage calls returned completed; no owned campaign app was active. Closed-hour provider attribution for owned apps was **$5.46372948 gross**, and the lagged whole-workspace delta was **$6.42587482**. Neither is a final campaign invoice: closed hours omit current-hour usage, billing may arrive late, and the workspace delta includes unrelated resources and can be revised. No credits are counted as authorization.

The ledger retains the user's **$70 historical hold plus the full $148 current allocation = $218 conservative total**, leaving **$32** under the $250 ceiling. There is no active compute exposure. Supervisor accounting used a conservative $10/hour worker ceiling and also charged supervision at that ceiling: worker $10.0568, aggregate $20.1464; these are admission estimates, not provider charges. The requested worker shape costs about $6.238224/hour at checked list rates, or about $6.2736 for the measured stage intervals, before preparation/build/supervision/storage. The original six-hour GPU allocation was $125, not a spending target.

Retained Volumes total about 128.744 GiB; the successful Volume contains 116.751 GiB, including native checkpoints. Another seven days at $0.09/GiB-month is about **$2.704**, already within the retained storage allowance. Review retention on **2026-11-26**; no data was deleted automatically. Costs beyond that review need continuing accounting. Pricing: [Modal list prices](https://modal.com/pricing); provider estimates and app limits are not guaranteed billing caps.

Private plans, receipts, raw traces, manifests, W&B readbacks, billing snapshots and the ledger remain under ignored `.local/reader-live-20261119/`; tensors and acquired data remain in owned Volumes. The successful supervisor call is `fc-01M7TXJGWB00GJ3FASK9THYY8J`, app `ap-4Vpp3dPovirZDajJimqcyb`. Stage calls, in order: `fc-01M7TXJTEAMBEK52ECPMQ80X6Y`, `fc-01M7TY12YCZ087KSQEF16VCNN1`, `fc-01M7TYQHK9FX2ZF0P4KP59BPVS`, `fc-01M7TZ6J5SC4C6WYB97DKZ729K`, `fc-01M7TZN4YTRP11P1GRXN2WJ41E`, `fc-01M7V0GB63Y658QPEDJW14Q95C`. These are terminal; do not resubmit the allocation.

No raw model data, checkpoint tensors, credentials or raw traces are published. The private repository, historical publication, source identities, source/data locks, licenses, calculator scorer and numerical semantics are unchanged. The full three-row milestone is **not complete**. This PR is ready for implementation/results review with the zero-signal GRPO limitation explicit; merge remains the owner's subsequent decision.
