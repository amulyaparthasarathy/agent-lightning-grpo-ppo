# Phase 4A offline evidence (2026-11-24)

Small reviewed records from local CPU runs at tested commit
`664a5a45e6648b298449c7a2a70c205ee683de39`. **None is a live model result.** No
model was called, no GPU was used and no cloud resource was created. See the
[verification report](../../docs/PHASE4A_VERIFICATION.md).

| File | What it is | Evidence class |
|---|---|---|
| [`prompt-budget.json`](prompt-budget.json) | Token counts of the rendered recovery prompts against the declared context limits. A test checks that the recorded prompt-set hash matches the current prompts | Offline measurement with the pinned tokenizer |
| [`prompt-replay.json`](prompt-replay.json) | The renderer and tokenizer reproduce the server-reported prompt token count of 1,343 of 1,343 recorded Phase 3 calls | Replay of recorded live calls; no new inference; asset task |
| [`candidate-identities.json`](candidate-identities.json) | Hashes of the calibration plans generated offline from the clean tested commit | Offline planning; nothing dispatched |
| [`scripted-calibration-report.json`](scripted-calibration-report.json), [`scripted-calibration-summary.md`](scripted-calibration-summary.md) | The calibration report built from 240 episodes run through the production harness with scripted model responses and simulated adapter events | Scripted inference; exercises reporting only; describes no policy |
| [`verification.json`](verification.json) | Gate results before and after the change | Local test execution |

The pinned tokenizer is a gated model file and is not committed. Raw traces from
the scripted demonstration stay in ignored `outputs/`. The Phase 3 traces used for
the replay are private and are not committed; only counts are recorded here.

Plans are bound to the commit they were generated from. The identities here are
for the tested commit; regenerate plans from the clean approved head before use.
