# Existing workflow closeout — November 14, 2026

The subsequent [GitHub-only review](GITHUB_CLOSEOUT_REVIEW_2026-11-14.md) records evidence checks, client-helper hardening and fresh offline tests. The execution results and immutable archive references below remain unchanged.

Workflow `preserve-20261114-01` is **complete**, backed by the immutable [terminal ledger](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/b19f5308aba912a7381310b87295fddbaa738399/workflows/preserve-20261114-01/ledger.json). This closeout executed only `replication_publish` and `replication_retrieve`. No training, scientific evaluation, conversion, backup or completed verification phase was repeated. The original release and all failed attempts remain preserved. HF links below require access to the two existing private repositories.

**Measured results under the unchanged scorer**

| Observation | Original | Replication |
|---|---:|---:|
| Completed GRPO iterations | 20 | 20 |
| Baseline / training initial | 27/64 | 27/64 |
| Training final | 34/64 | 33/64 |
| Native step-20 evaluation | 35/64 | 32/64 |
| Export evaluation | 34/64 | 33/64 |
| Baseline → native wins / losses / ties | 11 / 3 / 50 | 10 / 5 / 49 |
| Native → export wins / losses / ties | 0 / 1 / 63 | 1 / 0 / 63 |
| Native/export raw-answer differences | 15 | 9 |

These are descriptive outcomes on the same frozen 64 cases. The replication native gain over its baseline is 5/64 (7.8125 percentage points). The original historical result remains in [its report](../evidence/grpo-main-01.md). Replication training initial/final values are retained in the verified backup trainer log (step 0 and step 20); its combined 60/128 receipt is not a standalone 64-case evaluation. The replication log records 36 informative groups out of 157, three zero-signal iterations and 1,248 trained rows; the historical original records 30/160, six and 1,280 respectively. These observations do not establish general improvement or convergence.

**CPU reconciliation and resource accounting**

The [append-only audit](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/22157221ad45fb37d08145f4426fbdfaa4d18238/workflows/preserve-20261114-01/closeout-20261114/cpu-reconciliation.json) preserves the original ledger, selected Modal terminal timestamps/startup durations, implementation, tests and failed-attempt history. Ten successful unretried CPU calls each retain `ceil(max(worker elapsed, Modal execution + startup)) + 120` seconds. This releases **4,720.989769976 seconds** from **34,545.212912658**, leaving **29,824.223142682** before launch. The two unchanged 2,400-second reservations project to **34,624.223142682**, below 36,000. No ceiling, reservation guard, phase receipt, prior accounting entry or approved plan was changed.

The ten calls originally accounted for 7,538.989769976 seconds: 1,538.989769976 worker seconds plus 6,000 startup-allowance seconds. Modal reports 1,584.453373909 execution seconds and 27.811681136 startup seconds. Upward rounding plus the retained 1,200-second safety margin yields 2,818 accounted seconds for these calls. The initial build/control allowance was historically reduced from 14,100 to 7,800 seconds by an earlier evidenced reconciliation. All 7,800 seconds remain here, along with diagnostic and failed-attempt allowances. USD, GPU and byte accounting received no credits. The new app reused the existing worker implementation; its log reports no image build steps. CPU seconds follow the existing CPU-worker elapsed-time metric, not requested-core seconds.

| Resource | Before closeout | Reserved projection for both phases | Terminal accounted | Ceiling |
|---|---:|---:|---:|
| cost_usd | 52.24 | 55.39 | 53.91 | 100.00 |
| cpu_seconds | 34,545.21 | 34,624.22 | 31,456.86 | 36,000.00 |
| gpu_seconds | 8,853.62 | 8,853.62 | 8,853.62 | 28,800.00 |
| egress_bytes | 469,275,508,736 | 489,676,603,392 | 489,676,603,392 | 536,870,912,000 |
| hf_bytes | 239,799,894,016 | 256,996,540,416 | 256,996,540,416 | 322,122,547,200 |
| modal_bytes | 175,039,840,256 | 175,039,840,256 | 175,039,840,256 | 214,748,364,800 |
| monthly_usd | 9.00 | 9.00 | 9.00 | 30.00 |

The cost totals are conservative rate estimates with startup/transfer allowances and contingency, not provider invoices. Remaining account credits and actual billed recurring charges were not measured. The $9/month ledger entry retains the previously approved capacity/subscription basis. Byte totals retain reservations where actual usage is unknown; they are not measured storage bills.

**Replication difference investigation**

The immutable [Codex review](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/22157221ad45fb37d08145f4426fbdfaa4d18238/workflows/preserve-20261114-01/reviews/replication-differences.json) identifies the reviewer as the OpenAI Codex assistant, not an independent human reviewer. It binds comparison SHA256 `21777f574c27c7e81af1d4684664913793f98d007148bff1220b60d7970d80cd` and evaluated export-manifest SHA256 `e85d0bc827bb38d97b2894a458d282cc332bb78869be38c955c104d066b5463f`. No conversion or loading defect was identified within these checks; the root cause of changed greedy generations remains unresolved.

Both runs configure validation temperature 0, do_sample=false, n=1, top_p=1 and top_k=-1, with gateway val temperature 0. Training defaults at temperature 1 do not explain validation differences. Package/data/harness hashes match. Nine resolved-config differences concern intended artifact/template paths, run names, checkpoint resume and local gateway port. Native logging names the replication step-20 checkpoint; export logging loads 255/255 weights from the trained export with resume disabled. Both are val_only and have 64 completed rewarded cases. Full preserved inventories contain no evaluation checkpoint files or excluded checkpoint entries.

The exact 255-tensor comparison remains valid: FP32 bytes, shapes, dtypes and tied aliases match. Prepared tokenizer/special-token/template bytes match the export; native checkpoint config and generation bytes were copied unchanged with validated semantics. All nine changed cases have matching first model inputs. Three generate different calculator expressions; six change responses after matching tool arguments/results. Four extracted answers change; five changes are wording only. The sole score gain is a changed numeric calculation, not the placeholder-extraction mechanism found in the original run.

| Changed case | Tool arguments changed | Extracted answer changed | Recorded score changed |
|---|---|---|---|
| `ape210k__00007237` | no | no | no |
| `ape210k__00562105` | yes | yes | no |
| `ape210k__01104423` | no | yes | no |
| `ape210k__01175153` | no | no | no |
| `aqua_rat__CLf74qM1zHeqQhnZ` | no | no | no |
| `aqua_rat__NvfvzSFRovmxZLlM` | no | no | no |
| `gsm8k__XWa4gGg1MN1Xc87R` | yes | yes | 0 → 1 |
| `gsm8k__g7xi7YZzwH9PW4jH` | yes | yes | no |
| `mawps__f9zo88ce7YOnCC8b` | no | no | no |

[Private case-level observations](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/22157221ad45fb37d08145f4426fbdfaa4d18238/workflows/preserve-20261114-01/closeout-20261114/replication-difference-observations.json) retain the exact expression/answer differences and trace hashes. Full raw answers remain at the comparison receipt and original immutable snapshots. The frozen first-match extractor was inspected and recorded answers/rewards cross-checked; there was no rescoring. BF16 numerical execution, batching and request-history differences are possible explanations, not proven causes. Later requests contain generated tool-call IDs. Token/logit replay and in-memory loaded-weight identity are unavailable. Storage tensor equality does not imply deterministic generation equality.

**Immutable artifact revisions and identities**

| Artifact | Immutable HF revision |
|---|---|
| backup | [04b0e8abd04b8f95edbd80436ee269cedacd1e57](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/tree/04b0e8abd04b8f95edbd80436ee269cedacd1e57/snapshots/preserve-20261114-01-original) |
| original_evidence_backup | [987af8d788d4a73c32e15fae13ec09bddb6c6bbf](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/tree/987af8d788d4a73c32e15fae13ec09bddb6c6bbf/snapshots/preserve-20261114-01-original-evaluation) |
| replication_backup | [a43afaa681d5f03f294ceb5b99fbc4f686a2c1d1](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/tree/a43afaa681d5f03f294ceb5b99fbc4f686a2c1d1/snapshots/preserve-20261114-01-replication) |
| replication_evidence_backup | [98ccbe95580dd7310e6015e3a726540903c16832](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/tree/98ccbe95580dd7310e6015e3a726540903c16832/snapshots/preserve-20261114-01-replication-evaluation) |
| publish (main) | [ca9f4e3030854e81cf91f8e5211e709dea322a1a](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-GRPO/tree/ca9f4e3030854e81cf91f8e5211e709dea322a1a/) |
| replication_publish (replication-preserve-20261114-01) | [29eec61c7c804c1e030097b4219ac6e38a88d414](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-GRPO/tree/29eec61c7c804c1e030097b4219ac6e38a88d414/) |
| CPU reconciliation and replication review | `22157221ad45fb37d08145f4426fbdfaa4d18238` |
| Original difference review | `611f38729ef96ce7a3bd5ea2be76b4d9d01ad912` |
| Terminal workflow ledger | `b19f5308aba912a7381310b87295fddbaa738399` |

| SHA256 identity | Original | Replication |
|---|---|---|
| Native actor model | `759da43707212ac4a52bd790babe65b84841cca1c1ea9f2800baece32d5e8a33` | `342e715699a340ac82c2ae10c371d33a389aa3a1a5971ff1253d55bc34904e52` |
| Verified source manifest | `392b6171e05d3e7753b0427a9bf0764d942262d2606751576323a13b518636e1` | `70c98db2a9acda085802a35a831a5f31d78208faccb9c2e08a2422aa90e4e8f8` |
| Evaluated export manifest | `06218b1df7bf03c176fe8160f6da06a8092ace8ff8b6a97f6a985b5cfa433b91` | `e85d0bc827bb38d97b2894a458d282cc332bb78869be38c955c104d066b5463f` |
| Published export manifest | `f42e0bc2ab45e082a768febcccc330dbdee6c1e1f9e3adae34ba66a9d5c20049` | `1de8a98f21ab2a690728f5acb9d5eaa238a70be89302a4599350972b08e18968` |
| Export file-map digest | `c9f2712b4261fe1d5422cddd8d622b93b15ec30412913cae7fd2d991f309995e` | `b46ea487c22e89a1389fcd0f2c73ceab8900076cb0d39baac2c8bb90584800ef` |

Both exports retain all 255 FP32 tensors and the verified `lm_head.weight → model.embed_tokens.weight` alias. Per-file sizes and SHA256 values, including all eight Safetensors shards, are in the [machine-readable closeout index](../evidence/preservation-closeout-2026-11-14.json). Base revision is `0cb88a4f764b7a12671c53f0838cd831a0843b95`; frozen harness SHA256 is `e802a2324dab779ffaa49af678196a7cc5f02e3bb89936bd0ed1f1ed8c3a02fc`.

**Execution provenance and receipt index**

Original baseline/train/post execution: `7cfddf58a0dece6a63ca67d18b103db2940e4aca`; original smoke/reload: `ee120371429a4bc23d11b9da53d166d2fa4843fd`. Replication training, native/export evaluations and this closeout’s remote workers use `423245806c51d8af0c5f0546609ecb8689db4b88`, the inspected GitHub main. The client-only reconciliation helper is archived by SHA256 with its tests at the audit revision; it is not represented as part of that remote execution commit.

| Phase | Actual execution commit | Immutable receipt revision |
|---|---|---|
| inventory (attempt 0) | `bdbf4aaeaf9e3a0bc8d03c986b8a93234b91045e` | [2b8f199ffaea67c7d1c77a95f216a287e4357bca](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/2b8f199ffaea67c7d1c77a95f216a287e4357bca/workflows/preserve-20261114-01/receipts/inventory-0.json) |
| backup (attempt 2) | `e5a4d12e6163e6dff68fc6d6bcd38ee9e630960f` | [fdf57a10ca91773c23c6126f34cb26602ec55dc7](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/fdf57a10ca91773c23c6126f34cb26602ec55dc7/workflows/preserve-20261114-01/receipts/backup-2.json) |
| verify_backup (attempt 0) | `e5a4d12e6163e6dff68fc6d6bcd38ee9e630960f` | [de8e29e050b10fcbe2dd995100abc4e5fea9afd1](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/de8e29e050b10fcbe2dd995100abc4e5fea9afd1/workflows/preserve-20261114-01/receipts/verify_backup-0.json) |
| restore (attempt 0) | `e5a4d12e6163e6dff68fc6d6bcd38ee9e630960f` | [6ad9ead7ee6baddf887a0d9e605c16baa0580688](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/6ad9ead7ee6baddf887a0d9e605c16baa0580688/workflows/preserve-20261114-01/receipts/restore-0.json) |
| native_eval (attempt 1) | `0c43f0c17c4a4082d3c390910183237988b06c42` | [990d2e4f19c2c85fac7309fee4e075d49f573ad5](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/990d2e4f19c2c85fac7309fee4e075d49f573ad5/workflows/preserve-20261114-01/receipts/native_eval-1.json) |
| convert (attempt 1) | `7367380d77e86b15dfce8f0c30df0592d2b56024` | [7edb845dafa0f1a708d1437c08916da511525969](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/7edb845dafa0f1a708d1437c08916da511525969/workflows/preserve-20261114-01/receipts/convert-1.json) |
| export_eval (attempt 0) | `7367380d77e86b15dfce8f0c30df0592d2b56024` | [8d2e9b9edc49fcc574b63843dd93b0452f45c45a](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/8d2e9b9edc49fcc574b63843dd93b0452f45c45a/workflows/preserve-20261114-01/receipts/export_eval-0.json) |
| compare (attempt 0) | `7367380d77e86b15dfce8f0c30df0592d2b56024` | [0d0afe1c8aaf9d6d688692d441d067beab149215](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/0d0afe1c8aaf9d6d688692d441d067beab149215/workflows/preserve-20261114-01/receipts/compare-0.json) |
| original_evidence_backup (attempt 1) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [68b0199c8e5e6980c2d68f26b59a5a5ffb72e05f](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/68b0199c8e5e6980c2d68f26b59a5a5ffb72e05f/workflows/preserve-20261114-01/receipts/original_evidence_backup-1.json) |
| original_evidence_verify (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [ee9bf1fc1a1a2591c72624efce086c6fd9e8f39b](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/ee9bf1fc1a1a2591c72624efce086c6fd9e8f39b/workflows/preserve-20261114-01/receipts/original_evidence_verify-0.json) |
| publish (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [6dc306a1031ad324ad32ce55624fbf1048b4df05](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/6dc306a1031ad324ad32ce55624fbf1048b4df05/workflows/preserve-20261114-01/receipts/publish-0.json) |
| retrieve (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [8cfba1ea1de4fe7e691f129f3bd73dc8b06b8540](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/8cfba1ea1de4fe7e691f129f3bd73dc8b06b8540/workflows/preserve-20261114-01/receipts/retrieve-0.json) |
| replication_baseline (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [9a4021e5183c1f7def027fb99da80be0a3c47a55](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/9a4021e5183c1f7def027fb99da80be0a3c47a55/workflows/preserve-20261114-01/receipts/replication_baseline-0.json) |
| replication_train (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [e67d63c9f676775ad53921677c987e78c9d12db7](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/e67d63c9f676775ad53921677c987e78c9d12db7/workflows/preserve-20261114-01/receipts/replication_train-0.json) |
| replication_native (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [5ad58b18538bc58c4f960b71d2e6677580685695](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/5ad58b18538bc58c4f960b71d2e6677580685695/workflows/preserve-20261114-01/receipts/replication_native-0.json) |
| replication_compare (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [6eac20f6c5a67330e2bc9f1d0e9e3c1b3d1709a7](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/6eac20f6c5a67330e2bc9f1d0e9e3c1b3d1709a7/workflows/preserve-20261114-01/receipts/replication_compare-0.json) |
| replication_backup (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [41adae952fa1ffd75474d317294021c8789d99a3](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/41adae952fa1ffd75474d317294021c8789d99a3/workflows/preserve-20261114-01/receipts/replication_backup-0.json) |
| replication_verify (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [1bdd94b86bc3ae0f3a576ad0d428e5ab4d74e7a2](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/1bdd94b86bc3ae0f3a576ad0d428e5ab4d74e7a2/workflows/preserve-20261114-01/receipts/replication_verify-0.json) |
| replication_convert (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [5fc86787d8ecc2a3d7dc0d118568b976d84dd352](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/5fc86787d8ecc2a3d7dc0d118568b976d84dd352/workflows/preserve-20261114-01/receipts/replication_convert-0.json) |
| replication_export_eval (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [4a64511671dcafc056d1ff640e02f51e81a9b01f](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/4a64511671dcafc056d1ff640e02f51e81a9b01f/workflows/preserve-20261114-01/receipts/replication_export_eval-0.json) |
| replication_export_compare (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [d3de857e10873e7b7014b25248974cab68afbfe2](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/d3de857e10873e7b7014b25248974cab68afbfe2/workflows/preserve-20261114-01/receipts/replication_export_compare-0.json) |
| replication_evidence_backup (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [59252484de6a9dd0e093150ef83726bf108b187b](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/59252484de6a9dd0e093150ef83726bf108b187b/workflows/preserve-20261114-01/receipts/replication_evidence_backup-0.json) |
| replication_evidence_verify (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [9a60b7b44e4bc31fb1fa342734be24d1cd65877a](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/9a60b7b44e4bc31fb1fa342734be24d1cd65877a/workflows/preserve-20261114-01/receipts/replication_evidence_verify-0.json) |
| replication_publish (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [88c2c126df9370803d6ab5378afcf65ef80c5bc9](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/88c2c126df9370803d6ab5378afcf65ef80c5bc9/workflows/preserve-20261114-01/receipts/replication_publish-0.json) |
| replication_retrieve (attempt 0) | `423245806c51d8af0c5f0546609ecb8689db4b88` | [006c8bcc095f252a6b41d85a07e075543cfb4700](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/006c8bcc095f252a6b41d85a07e075543cfb4700/workflows/preserve-20261114-01/receipts/replication_retrieve-0.json) |

Earlier phase commits missing from the old phase schema were reconciled from retained terminal fallbacks for backup/verification/restore and the historical inventory reconciliation. Failed attempts are retained separately:

| Failed phase / attempt | Immutable receipt revision |
|---|---|
| backup / 0 | [cd9529d50251131b94c62ebcb29759df3474649a](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/cd9529d50251131b94c62ebcb29759df3474649a/workflows/preserve-20261114-01/receipts/backup-0-reconciliation-final.json) |
| backup / 1 | [fbb2f383c0a4e5e74ce611acf761236f456b83eb](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/fbb2f383c0a4e5e74ce611acf761236f456b83eb/workflows/preserve-20261114-01/receipts/backup-1.json) |
| convert / 0 | [46f70907249adbff6a8c407da91e5610510fbbfc](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/46f70907249adbff6a8c407da91e5610510fbbfc/workflows/preserve-20261114-01/receipts/convert-0.json) |
| native_eval / 0 | [70efa6fbf653a6b96843ff8ac31e9737701f787a](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/70efa6fbf653a6b96843ff8ac31e9737701f787a/workflows/preserve-20261114-01/receipts/native_eval-0.json) |
| original_evidence_backup / 0 | [7368690087424f703d3123e7c88dedbaa8ec0ba2](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/7368690087424f703d3123e7c88dedbaa8ec0ba2/workflows/preserve-20261114-01/receipts/original_evidence_backup-0.json) |
| diagnostic: conversion-metadata-0 | [24136d63394756b311092bd0b6e5c15fe33d1883](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/24136d63394756b311092bd0b6e5c15fe33d1883/workflows/preserve-20261114-01/receipts/conversion-metadata-0.json) |
| diagnostic: evidence-inventory-0 | [1f89dd0c3042dce675312a1d57e90891fe2c09e4](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/1f89dd0c3042dce675312a1d57e90891fe2c09e4/workflows/preserve-20261114-01/receipts/evidence-inventory-0.json) |
| diagnostic: native-eval-startup-0 | [d5d1dfccc4354b341445234783811cc53f75999c](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/d5d1dfccc4354b341445234783811cc53f75999c/workflows/preserve-20261114-01/receipts/native-eval-startup-0.json) |

Recovery Git bundles remain at these immutable backup revisions:

- `738ce0da2c68b152157a49fa9ccf1e2247d5d589.bundle`: [4793f46aea6f61f0467e48b96c947a2bc6d7deef](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/4793f46aea6f61f0467e48b96c947a2bc6d7deef/workflows/preserve-20261114-01/recovery-history/738ce0da2c68b152157a49fa9ccf1e2247d5d589.bundle)
- `e5a4d12e6163e6dff68fc6d6bcd38ee9e630960f.bundle`: [ef4c7092ce4cdf20e262be439e238a07f23f77a7](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/ef4c7092ce4cdf20e262be439e238a07f23f77a7/workflows/preserve-20261114-01/recovery-history/e5a4d12e6163e6dff68fc6d6bcd38ee9e630960f.bundle)
- `0c43f0c17c4a4082d3c390910183237988b06c42.bundle`: [94261a1ba93d79c722f063105309f39207152283](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/94261a1ba93d79c722f063105309f39207152283/workflows/preserve-20261114-01/recovery-history/0c43f0c17c4a4082d3c390910183237988b06c42.bundle)
- `7367380d77e86b15dfce8f0c30df0592d2b56024.bundle`: [ea6102b98af8a15e9f5edd594246ba522aad3416](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/ea6102b98af8a15e9f5edd594246ba522aad3416/workflows/preserve-20261114-01/recovery-history/7367380d77e86b15dfce8f0c30df0592d2b56024.bundle)
- `423245806c51d8af0c5f0546609ecb8689db4b88.bundle`: [1b93c8c5444aa556a138d39f9e7ae439ce3e2071](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/blob/1b93c8c5444aa556a138d39f9e7ae439ce3e2071/workflows/preserve-20261114-01/recovery-history/423245806c51d8af0c5f0546609ecb8689db4b88.bundle)

**Final independent retrieval and limitations**

Publication call `fc-01M7D13862MZ8TYCMCMM06T6R9` and retrieval call `fc-01M7D1AW6DQ8KZVRKDX45WSN7D` completed under app `ap-FLIHJjDMhxix6JlaNOHOVL`. Retrieval used a fresh CPU worker with no Volume mounts, downloaded the immutable release into fresh scratch/cache, verified every listed file and manifest checksum, then loaded the tokenizer/model with local files only and remote code disabled. The fixed load smoke check succeeded; it is not a new scientific evaluation. The original main release remains `ca9f4e3030854e81cf91f8e5211e709dea322a1a`. All 23 previously passed phase records are unchanged.

Historical execution-closeout validation: **145 tests passed in 6.950 seconds**, including five reconciliation tests covering nonfinite/ambiguous observations, wrong calls, retries/failures/GPU exclusion, duplicate reconciliation, preservation of history, claimed phases and budget guards. The first new-test attempt exposed a shared fixture dictionary and was corrected in the fixture only. A read-only Modal metadata query hit a rate limit; it launched no work and was replaced with bounded spaced reads. No automatic scientific or transfer retry occurred.

Remaining limitations: generation discrepancies are unresolved; no independent mathematical grading, significance/causality claim, validated optimizer continuation or hosted W&B readback. Sparse signal, dropped rows, tool/parser errors and the qualified runtime’s dependency conflicts remain. Final checksum/loading verification does not retroactively establish historical pre-preservation tensor identity. No source, historical artifact, failed receipt or original release was deleted or overwritten.
