# Evidence to retain

**Phase 4A offline verification, November 24, 2026:** [`phase4a-offline-20261124/`](phase4a-offline-20261124/README.md) holds prompt-budget measurements, a replay of recorded prompt token counts, candidate plan identities and a scripted calibration report. It contains no live model result; see the [verification report](../docs/PHASE4A_VERIFICATION.md).

**Completed Phase 2 r03, November 15, 2026:** [closeout](../docs/PHASE2_CLOSEOUT_2026-11-15.md), [selected evidence index](phase2-r03/EVIDENCE_INDEX.md), [six-run CSV](phase2-r03/per_seed_results.csv), and [publication manifest](phase2-r03/manifest.json). Run `python3 scripts/check_phase2_results.py` from the repository root for offline arithmetic/hash checks. All six study seeds/arms are retained; qualification and prior failures are separate. The older records below remain history, with their original source and retention scopes.

The completed `preserve-20261114-01` workflow has a [closeout report](../docs/PRESERVATION_CLOSEOUT_2026-11-14.md) and [machine-readable revision/identity index](preservation-closeout-2026-11-14.json). These add original/replication backup, conversion, publication and independent retrieval evidence without replacing the historical experiment records.

The [final archive locator](preservation-closeout-archive-2026-11-14.json) records the immutable private HF revision and checksums for the historical report snapshot, index, terminal readback and stopped-app evidence. The subsequent [GitHub-only review](../docs/GITHUB_CLOSEOUT_REVIEW_2026-11-14.md) distinguishes fresh repository checks from that completed execution.

This directory contains small, reviewed records from actual runs: [GRPO smoke and reload](grpo-smoke-01.md), and [20-step GRPO with final paired evaluation](grpo-main-01.md) with its [64-case comparison](grpo-main-01-cases.csv). Earlier verification records remain historical accounts of their respective milestones.

After each useful experiment, preserve the source commit, resolved configuration, model revision, dataset-selection hashes, actual completed step count, environment manifest, evaluation summary, selected trajectories, failures, and checkpoint identity. Record whether an output is synthetic, local-only, or from an actual model run. Missing results remain missing.

Modal keeps run records under `/data/runs/NAME` and submission/build receipts under `/data/submissions/NAME` in the `cs329z-agent-lightning` Volume. Use the runbook's `export` action and `modal volume get` to retain an evidence archive. The archive excludes native checkpoints and model weights; those remain in the Volume unless you export them separately.

Keep a separate durable copy of useful evidence and any checkpoints you need. A local directory, cloud Volume, W&B view, or ignored file is not by itself a backup. Confirm the export is readable before deleting runtime storage. Never commit credentials or company/customer data.

The original CPU illustration is under `../experiment/example_outputs/cpu-accounting/` and is explicitly synthetic. It is not an RL result. Code can be recreated from Git; a stochastic historical run should be preserved as evidence rather than assumed reproducible byte for byte.
