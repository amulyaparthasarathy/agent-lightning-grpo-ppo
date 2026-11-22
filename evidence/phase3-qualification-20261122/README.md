# Phase 3 qualification evidence, 2026-11-22

The campaign is not fully qualified. Both final baselines passed; both first
training collections were rejected before optimization. No checkpoint was saved.
See [the dated report](../../docs/PHASE3_QUALIFICATION_2026-11-22.md).

- `identities.json`: exact executed commits and canonical plan/source/configuration/task/data hashes.
- `a01`–`a04`: per-attempt source/task/data inventories, actual package inventories,
  submission receipts, stage summaries, trace hashes and safe integration joins.
- `collected-file-manifest.json`: hashes of retained originals, including logs and
  configurations whose contents are deliberately not published.
- `stage-summary.json`: all 24 logical algorithm/stage pairs across four attempts.
- `hosted-reconciliation-summary.json`: independent hosted run enumeration and comparison.
  The a04 training-history explanation preserves the difference between collection
  artifact step 1 and zero actual optimizer updates.
- `provider-reconciliation.json`, `cost-closeout.json`, `spending-ledger.json`:
  owned-call final states, observed partial charges, conservative exposure and
  reservation history. The $40 closeout hold includes observed charges; it is not
  $40 of additional spending. $260 remains unreserved from the new $300 ceiling.
- `retained-resources.json`: explicit resources and the November 29 UTC retention
  review deadline. No automatic deletion has been scheduled.

Raw traces, live runtime requests/configuration contents, model weights and
checkpoint tensors are not published. Every source inventory identifies the
execution commit; this documentation/evidence publication is a later commit.
