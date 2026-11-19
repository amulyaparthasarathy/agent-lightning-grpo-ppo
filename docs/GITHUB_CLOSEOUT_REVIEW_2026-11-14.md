# GitHub-only closeout review — November 14, 2026

The existing `preserve-20261114-01` workflow was already complete before this review. This change publishes its [closeout report](PRESERVATION_CLOSEOUT_2026-11-14.md), [revision/identity index](../evidence/preservation-closeout-2026-11-14.json), [archive locator](../evidence/preservation-closeout-archive-2026-11-14.json), and the client-only accounting helper with regression tests. The requested PR remains subject to owner review; opening it does not authorize a merge.

GitHub main was inspected at `423245806c51d8af0c5f0546609ecb8689db4b88`. Review used 21 read-only immutable HF metadata downloads, the already preserved records, and local offline checks. There were no Modal API calls or execution, GPU jobs, model publication, artifact writes, training, scientific evaluation, conversion, or workflow creation during this GitHub closeout.

The final archive is [revision `b04fe483b526d289f1f9e47a7c97f2b6d520ce55`](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/tree/b04fe483b526d289f1f9e47a7c97f2b6d520ce55/workflows/preserve-20261114-01/closeout-20261114/final). Its manifest/checksums, terminal ledger, all 25 phase references, original/replication comparison receipts, conversion receipts, published export-manifest byte hashes, independently verified checkpoint identities, and baseline/training records were checked. The repository's revision index and archive locator remain unchanged from the completed closeout. The report adds review navigation and labels the earlier test result as historical; its archived source checksum continues to identify the earlier document snapshot.

**Code review**

`reconcile_cpu` is a pure calculation with no cloud client, credential access, file write, CLI, or launch path. It copies the ledger, preserves its plan and phase records, and returns an append-only audit. Both remaining reservations pass through the existing guard. It refuses already completed workflows, duplicate closeout reconciliation, claimed phases, retried/failed/GPU phases, and ambiguous or nonfinite timing evidence. Its caller must authenticate the evidence and use an expected-parent write guard; this helper alone does not authenticate supplied records.

This GitHub review adds validation for complete, finite, nonnegative accounting within every existing limit and a lowercase hexadecimal immutable revision. Tests also cover empty evidence, completed workflows, a larger worker-elapsed observation, no releasable allowance, and a case where only the second remaining reservation exceeds the limit. These are client-side hardening changes after execution, not the implementation used by the historical remote workers. The executed helper and tests remain immutable in the [CPU audit archive](https://huggingface.co/amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup/tree/22157221ad45fb37d08145f4426fbdfaa4d18238/workflows/preserve-20261114-01/closeout-20261114).

Replaying the hardened helper locally against the immutable pre-adjustment ledger and ten recorded Modal timing observations reproduces every original calculated audit field exactly: a **4,720.989769976-second release**, **29,824.223142682 seconds before launch**, and **34,624.223142682 seconds with both reservations**. No live ledger or allowance was changed during this review.

**Offline execution evidence**

The following precommit run tested the proposed helper and tests on a working tree based on the main SHA above. It is separate from the historical 145-test execution-closeout receipt. The final committed tree is tested again before the PR is opened; the PR description records that exact head SHA and run result.

```bash
MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v
```

| Field | Observation |
|---|---|
| Started (UTC) | `2026-11-14T13:11:17.158448+00:00` |
| Finished (UTC) | `2026-11-14T13:11:24.688186+00:00` |
| Result | **148 tests passed**, exit code 0 |
| Suite duration | 7.008 seconds |
| Python / architecture | CPython 3.12.3 / aarch64 |
| Platform | `Linux-5.15.167.4-microsoft-standard-WSL2-aarch64-with-glibc2.39` |
| Invocation | Repository `.venv/bin/python`; executable resolves to `/usr/bin/python3.12` |
| Test log SHA256 | `3f89a3ef3ca2499fb008616e51882415f480802368bd5cf27756ac7a49733c5e` |
| Helper SHA256 | `4621ecec0bc17890bdcdabfa5aa5f10a616d356059c01e7c950cacc6f88e018d` |
| Test module SHA256 | `c81f46e8db254aac824573c6adbe35308d941a08b35452d61dca3cc3eec08914` |

The focused reconciliation suite has eight tests. The full suite uses synthetic fixtures/mocks and offline configuration; it does not qualify a new cloud runtime or revalidate model behavior.

**Publication review and limitations**

The proposed diff is limited to nine explicitly reviewed text files: documentation/navigation, the two existing closeout evidence indexes, and the helper/tests. JSON parsing, local Markdown links, whitespace, an exact file allowlist, credential-pattern checks, and comparisons against preserved raw-answer text are checked before commit. The JSON indexes contain hashes, case identifiers, aggregate observations and receipt references, with no raw question/prompt/answer fields. No `.env`, ignored local review files, raw trajectories, model files, binaries or symlinks are staged. Experiment, scorer, configuration, dependency, worker and workflow files remain unchanged. Pattern checks supplement manual review; they do not prove the absence of every possible unknown secret.

Native/export scores remain **35/64 → 34/64** for the original and **32/64 → 33/64** for the replication. The replication has **nine raw-answer differences**, **four extracted-answer changes**, and **one score gain**. Three tool expressions differ; six responses differ after matching recorded tool arguments/results. The causes remain unresolved despite exact 255-tensor storage equality. The report makes no deterministic reproduction, statistical significance, causal/general improvement, independent mathematical grading, validated optimizer continuation, or hosted W&B-readback claim. Accounted costs remain estimates with reservations, not provider invoices.
