> Publication copy of the 2026-11-15 local audit. Historical audit verdicts are retained; links resolve within this selected evidence bundle. This closeout performed no new cloud audit. See [bundle scope](EVIDENCE_INDEX.md).

# Independent r03 verification — 2026-11-15

**PASS for completed execution and recorded scientific endpoints.** All six study jobs have 20 actual actor optimizer calls; all three PPO-MC jobs also have 20 critic calls. All six separate final-checkpoint evaluations pass the frozen completion checks. This is a three-pair exploratory study on a reused 64-case development panel, not evidence of algorithmic superiority or generalization.

The authoritative deliverables are in this `final/` directory. The earlier audit work files are preserved; their provisional strict config-equality failures are explained in EVIDENCE_INDEX.md.

| Claim | Verdict | Evidence and boundary |
|---|---|---|
| Qualification and study supervisors | PASS | Both local durable indices and collected provider reconciliation report completed; 6 + 18 stage receipts independently checked. |
| All 24 expected stages | PASS | Exact plan identities, call IDs, completed receipts, exit 0, terminal events, resources and stage boundaries; no missing r03 stage. |
| 20 actor updates × six study jobs | PASS | Post-hook `optimizer_step` counts exactly 1..20, one per step; terminal global_step=20. |
| 20 critic updates × three PPO-MC jobs | PASS | Counts exactly 1..20; learned Llama token-classification critic. GRPO has no critic updates. |
| Gradients and weight refresh | PASS | Finite gradient receipts and optimizer_called=true; training refresh 0..20. Eight zero-gradient GRPO calls are retained, not reclassified as skipped. |
| One final checkpoint per study job | PASS | Single global_step_20, native save event, correct actor/critic inventory, byte sum and manifest hash. |
| Six independent checkpoint reload evaluations | PASS | Separate worker IDs, matching checkpoint manifest hashes, resume_path, trainer_returned=20, validation_only=true, zero optimizer calls. |
| All evaluation panels | PASS | 16 evaluations including qualification; each has the expected 64 unique IDs, 64 complete real episodes and 64 posted binary rewards. No partial/synthetic episode counted as success. |
| Source / data / model identity | PASS (recorded evidence) | 44 frozen source file hashes verified against Git; same plans/manifests, guarded model/data validation, resolved configs and runtime registration. No downloaded weight or parquet rehash was attempted. |
| Hosted W&B identity, scalar history and summaries | PASS | Exact 24 planned IDs; 140 history rows; all local scalar rows and final summaries agree within 1e-12 tolerance. No missing final training row. |
| Literal full W&B config equality | FAIL, explained | Hosted actor/critic optimizer total_training_steps resolves local -1 to 2 or 20; nonempty flattened fields otherwise agree. Protocol/source/data/seed/stage metadata matches exactly. This is not a training failure. |
| Provider app metered charges | PASS, as-of snapshot | Qualification $5.76491115; study $34.33868457; total $40.10359572 at 14:12:18 UTC. |
| Final r03 invoice / future storage | UNVERIFIED | Workspace billed amount is not an r03 invoice. Future storage and full study-attributed non-app charges are unavailable. Workspace breakdown also has a $0.00499327 additive discrepancy, preserved below. |
| Fresh holdout / superiority | UNVERIFIED by this design | Repeated legacy development panel; three pairs; algorithm-specific RNG streams. No significance test or superiority claim. |

| Algorithm | Seed | Baseline | Final | Baseline % | Final % | Δ pp | Actor / critic updates | Wins / losses / ties |
|---|---|---|---|---|---|---|---|---|
| GRPO | 2001 | 27/64 | 36/64 | 42.1875 | 56.2500 | 14.0625 | 20 / 0 | 11 / 2 / 51 |
| PPO-MC | 2001 | 26/64 | 30/64 | 40.6250 | 46.8750 | 6.25 | 20 / 20 | 7 / 3 / 54 |
| GRPO | 2002 | 27/64 | 30/64 | 42.1875 | 46.8750 | 4.6875 | 20 / 0 | 7 / 4 / 53 |
| PPO-MC | 2002 | 25/64 | 34/64 | 39.0625 | 53.1250 | 14.0625 | 20 / 20 | 12 / 3 / 49 |
| GRPO | 2003 | 26/64 | 34/64 | 40.6250 | 53.1250 | 12.5 | 20 / 0 | 13 / 5 / 46 |
| PPO-MC | 2003 | 26/64 | 34/64 | 40.6250 | 53.1250 | 12.5 | 20 / 20 | 10 / 2 / 52 |

| Seed | GRPO − PPO final (pp) | GRPO − PPO improvement (pp) | GRPO-only / PPO-only correct | Both correct / both wrong |
|---|---|---|---|---|
| 2001 | 9.375 | 7.8125 | 9 / 3 | 27 / 25 |
| 2002 | -6.25 | -9.375 | 4 / 8 | 26 / 26 |
| 2003 | 0.0 | 0.0 | 7 / 7 | 27 / 23 |

Measured app charges are distinct from $50 qualification / $175 study allocations and the locked $75 reserve. The prelaunch $3.15125017 and $9.08142174 gates are historical workspace observations, not r03 charges.

See [SCIENTIFIC_ANALYSIS.md](SCIENTIFIC_ANALYSIS.md), [stage_receipts.csv](stage_receipts.csv), [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md) and [figures](figures/INDEX.md). Existing source and evidence were not changed; no GPU work, collection, deployment or cloud mutation was performed.
