# Reference qualification attempt — 2026-11-19

**Live qualification stopped before GPU execution.** Real acquisition and preparation succeeded; same-source CPU preflight did not pass. Neither GRPO nor PPO-MC is qualified. The owner authorized $250 gross and an initial attempt plus two diagnosed recovery attempts. Those three allocations were used; no fourth attempt was submitted. PR #3 remains draft and unmerged.

## Execution provenance and failures

| Allocation | Public execution source | Observed outcome |
|---|---|---|
| Initial | `e2d12edb87300ec97896c5931a10af2efe91c1b5` | CPU entry failed before acquisition: a resolved Modal Volume path could not be made relative to literal `/data`. |
| Recovery 1 | `3fa11d169bbec4fb2b75ff3bcf18c639ef8bc007` | Acquisition/preparation completed. Preflight failed at the strict absolute-deadline gate before scientific checks. The failure did not retain clock samples, so its exact offset is unknown. |
| Recovery 2 | `d920e0e9ec1fbb6c0b5dfc7c4e18ea6c844368bf` | Acquisition/preparation completed. Preflight passed the deadline gate, then rejected a source-path alias mismatch: preparation recorded the physical Volume path, while preflight supplied `/data`. |

A bounded CPU diagnostic found that distinct Modal containers both report hostname `modal`; provider-assigned `MODAL_TASK_ID` is now the reader container identity. Reader dispatch reserves 30 seconds inside the original time limit; strict worker deadline checks remain. After all calls terminated, a further offline correction made preflight and stage input verification use the same owned canonical staging paths as preparation. This correction has **not** run remotely. No historical receipt was reassigned to corrected source.

Final attempted execution bindings:

| Binding | Value |
|---|---|
| Source inventory SHA-256 | `24ee7a5b7e86970a23b793761be21690c187d4a74e89dd6dfde3410bc419a0ae` |
| Configuration SHA-256 | `8a057837dc32e67850b6dd5b9c3493fd06f0519370b8ae89f3cfdbd7b2bf0481` |
| Preparation plan SHA-256 | `109df1b08ce46364bdbd5d3cc6611d777bcc77dda8f4fd2ba8ea0b52809ddeab` |
| Asset-bound plan SHA-256 | `52f2277e60f9f46b712cfe459a7a1e61743883c986f8e669601f746fd24c2d0f` |
| Asset manifest canonical SHA-256 | `8829d0fe3de249e49a1841aef3cc8ade637f1751f3f6e3484506f50348fdab74` |
| Model manifest file SHA-256 | `348c506e93573a74f0c3275ddc97afed3343a73bf009457e5a12649125739519` |
| Data manifest canonical SHA-256 | `7c3a77b9f411530e95bbd8df663a1dcb43e815037e4d3971c69f773788653915` |

## Verified inputs and account prerequisites

Authenticated SDK ownership was `amulyasp`, environment `main`. Existing private credentials were parsed without execution or disclosure. Read-only Hugging Face checks verified access to `meta-llama/Llama-3.2-3B-Instruct@0cb88a4f764b7a12671c53f0838cd831a0843b95`. Only the intended private W&B project `amulyap-stanford-university/reader-owned-reference` and named Modal Secret `reader-reference-credentials` were created when absent. The Secret contains only the two required keys; existence does not prove remote W&B write/finish capability. No unrelated or historical resource was changed.

Actual model acquisition verified both weight shards (6,425,529,048 bytes total), full locked file inventory, config, tokenizer and fixed template hashes. Preparation verified schemas, ordered cases, split isolation, label separation and deterministic selection. Acquired source hashes:

- Train Parquet: `d3cfeb2b869e76500f5734c4a7d3ded97ac14299a6636d30d0d461574bf10c16`.
- Test Parquet: `7bad3039690001c01d73c79a585e07ea3ef98c1698e667540e9a419eb81caf86`.

Prepared panels contain 256 training cases, 16 nested training-smoke cases, 64 final evaluation cases and eight separate evaluation-smoke cases. The three historically locked main output files match byte-for-byte. Historical raw acquisition hashes are absent from the old lock, so that comparison remains unavailable. New manifests have their own timestamps, paths and identities. This is a methodology rerun, not a reassignment of historical results.

The actual final image was `im-qQ9AdnJhnxyW1PlyYEjUxL`. Build/package artifacts were collected; cache-hit status is unverified. Runtime compatibility gates, remote calculator operation, W&B access from the final image and GPU kernels were not reached. A separate real local calculator-only check passes the advertised schema, valid arithmetic and expected invalid-request error; it does not qualify model/tool integration.

## Qualification dimensions

| Dimension | GRPO | PPO-MC |
|---|---|---|
| Baseline / training / reload stages | All not started | All not started |
| Actual actor / critic optimizer calls | 0 / not applicable | 0 / 0 |
| Valid model trajectories, losses, gradients | No live evidence | No live evidence |
| Native checkpoint / fresh-worker reload | No checkpoint or reload | No checkpoint or reload |
| Model → tool → observation → terminal reward chain | Unqualified | Unqualified |
| Informative reward groups | Not measured; no training collection | Not applicable |
| Evaluation accuracy / improvement | Not measured | Not measured |

No zero-signal learning conclusion is inferred from absent training. Task correctness, tool integration, operational completion and learning remain separate claims. The alternate-model and alternate-task milestone rows remain subsequent implementation and qualification work.

## Checks, costs and retained evidence

`bash scripts/test_reader_offline.sh` passed: CPU accounting demo, historical result checks (12 artifact hashes, 24 stage summaries, six runs, three contrasts), 39 PPO-MC tests, 15 matched-study tests and 71 reader/shared tests. No tests were skipped. Network and credentials are denied in this suite; remote submissions and synthetic model/data evidence are explicitly mocked. The added real-filesystem regression reproduces the mount alias mismatch, verifies canonical-path success, and still rejects corrupted acquired bytes and file symlinks. Numerical code, scorer, locks, source identities, licenses and historical evidence remain unchanged.

At 21:50 UTC no owned campaign app was active and all submitted calls were reconciled. The provider's gross workspace delta was about $0.022, but billing was lagged and no completed-hour attribution was available; this is **not a final campaign charge**. The ledger retains a conservative $70 hold for all CPU attempts, the diagnostic and retention, leaving $180 of budget headroom. No GPU allocation was spent. About 11.993 GiB is retained in the two prepared Volumes; at the checked $0.09/GiB-month rate, another seven days is about $0.252 before any other charges. Review retention on 2026-11-26; nothing is automatically deleted. Pricing source: [Modal pricing](https://modal.com/pricing).

Private local plans, receipts, logs, manifests, billing snapshots and the spending ledger remain under ignored `.local/reader-live-20261119/`; model/data bytes remain in owned Volumes. The final failed call is `fc-01M7TWF3PHD95B5QQG9TET422A`. Its registry receipt, provider failure status and collected `preflight/action.log` agree. No raw dataset, model weights, credentials or raw model traces are published here.

A further attempt requires an explicit extension of the recovery-count authorization, a fresh allocation and plans bound to corrected committed source. Available dollars alone do not override the three-attempt limit. Actual preparation and same-plan CPU preflight must pass before the six-stage GPU workflow can start.
