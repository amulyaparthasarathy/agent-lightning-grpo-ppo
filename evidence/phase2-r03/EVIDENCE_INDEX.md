# Selected r03 evidence bundle

This is a new, scoped publication index inside the existing private repository. The four CSVs are publication copies normalized from CRLF to LF for Git storage; their fields and values are unchanged, and both byte hashes are recorded. It is not a transplanted copy of the original audit manifest. Read the [closeout](../../docs/PHASE2_CLOSEOUT_2026-11-15.md) first.

- [Verification ledger](VERIFICATION.md) and [scientific analysis](SCIENTIFIC_ANALYSIS.md): explicitly identified publication copies of the completed audit, with transformations and both input/output SHA256 hashes in [manifest.json](manifest.json).
- [Per-seed results](per_seed_results.csv): all six study runs, qualification excluded.
- [Paired differences](paired_differences.csv): all three signed contrasts.
- [Stage receipts](stage_receipts.csv): 24 derived summaries, including qualification. `evidence_path` resolves from the original worktree root into private `.local` collections. `checkpoint_identifier` preserves the original audit's relative or absolute collected-checkpoint locator. These are locators, not included raw files. Call/W&B IDs and receipt/checkpoint hashes are retained for provenance and are not credentials.
- [Failed attempts](failed_attempts.csv): six pre-r03 failed baseline workers; not study outcomes or replacement seeds.
- [Figure](figures/INDEX.md): one selected scalar-results SVG.
- [Integration checks](INTEGRATION_CHECKS.md): closeout checks, separate from scientific verification.

The manifest's `artifacts[].path` is relative to this directory; its hashes cover only those explicitly listed bundle files. `input_path` and `omitted_local_artifacts` are locators relative to `/workspace/projects/repos/agent-lightning-exploratory`, not promises of Git-hosted content. The manifest excludes itself. Original audit output bytes remain in `.local/recovery-r03/scientific-review-20261115-02/final/`. The original EVIDENCE_INDEX, input/artifact hash inventories, 18,231-row training_metrics_long.csv and full 49-SVG atlas remain there; input hashes for the principal omitted files are recorded here without copying their out-of-scope relative manifest paths.

## Private retention locators

- Executed source: `5fb6adfd48ee245509f7dff3ab1d849fcf707d86` in this Git repository. Documentation/merge commits are not executed identities.
- Plans and runner: `.local/recovery-r03/{qualification-plan.json,study-plan.json,phase2_overnight.py,overnight.log}`. No plan or digest was regenerated during closeout.
- Successful local collections: `.local/recovery-r03/{qualification-collected,study-collected}/`; exact stage suffixes appear in the stage CSV.
- Full local audit: `.local/recovery-r03/scientific-review-20261115-02/`, including final reports and previously collected provider/W&B snapshots. The earlier provisional audit is retained separately, with the final audit's corrections explained in its original index.
- Native checkpoints: existing Modal Volume `cs329z-agent-lightning-working-stanford-v2`, under `phase2/studies/agl-explore-20-v1-r03/{phase}/{job}/training/checkpoints/global_step_{2|20}`. Exact job names and checkpoint identities are in the stage CSV. The suffix `v2` names the staging Volume, not its backend version. No tensors are included in Git.
- Hosted records: W&B project `amulyap-stanford-university/cs329z-agent-lightning`, with the exact 24 run IDs in the stage CSV. This is a locator to retained records, not a new API readback.

Raw trajectories, dataset archives/Parquets, model/optimizer tensors, credentials, environment dumps and the paper PDF are excluded. Unknown redistribution rights are not resolved by private integration. This bundle is not a backup of all raw traces or weights; no additional backup or storage cleanup was performed.
