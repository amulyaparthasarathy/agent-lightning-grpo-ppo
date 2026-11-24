# Phase 3 correction and qualification evidence, 2026-11-24

Both algorithms are operationally qualified on `asset-status-v1` under the
prospective `asset-episode-v2` contract. See
[the dated report](../../docs/PHASE3_QUALIFICATION_2026-11-24.md). The earlier
failed campaign is preserved unchanged in
[`phase3-qualification-20261122`](../phase3-qualification-20261122/README.md).

Execution source for both live attempts: `7f1e13d7bacb477c9e5e95a16868a2b0f2484ee6`.
This directory was published by a later documentation commit.

- `identities.json`: planned and executed commits, canonical plan, source,
  configuration, task, environment and data hashes, and the episode policy.
- `a05/reconciliation.json`: the earlier session's interrupted, never-executed
  submission, the provider findings and its stop marker.
- `a06`: the collection-only development run. `development-acceptance.json` holds
  the declared panel, seeds, bounds and reviewed results. It is not qualification
  evidence.
- `a07`: the six-stage qualification. One record per algorithm and stage with the
  supervisor receipt, completion proof, audit-event counts, optimizer and gradient
  events, collection accounting, native checkpoint manifest, recomputed trace joins
  and the hosted W&B readback with every metric comparison.
- `a06` and `a07` also hold source, task, data, prepared-asset and package
  inventories, submission receipts, the collection validator's report,
  `final-local-verification.json` and `collected-file-manifest.json`. The manifest
  hashes retained originals whose contents are deliberately not published.
- `stage-summary.json`: all eight executed algorithm/stage records.
- `trajectory-diagnostics.json`: format, reward, policy-error and group aggregates
  with trajectory shapes (tool names and key sets only).
- `hosted-reconciliation-summary.json`: independent hosted enumeration of all ten
  planned identities.
- `provider-reconciliation.json`, `cost-closeout.json`, `spending-ledger.json`:
  final call states, observed completed-hour charges by attempt, conservative
  exposure bounds and the full reservation history. The US$74 of holds includes
  the US$12.31 observed; it is not additional spending.
- `retained-resources.json`: owned volumes, registries and apps, with the two
  retention review deadlines. No deletion has been scheduled or performed.
- `closeout`: the later [closeout](../../docs/PHASE3_CLOSEOUT_2026-11-24.md)
  records. `preservation-inventory.json` lists local and remote evidence
  locations, checkpoint inventories and backup status.
  `provider-reconciliation-final.json` is the last read-only provider query,
  `campaign-closeout.json` separates observed charges, unresolved billing,
  continuing storage and unused authorization, and `spending-ledger-final.json`
  is the ledger with execution recorded as closed.

Raw traces, prompts with observations, runtime requests and configuration
contents, model weights and checkpoint tensors are not published.
