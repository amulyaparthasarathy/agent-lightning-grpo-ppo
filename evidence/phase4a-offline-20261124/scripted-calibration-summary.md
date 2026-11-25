# Recovery readiness development calibration (scripted-inference)

Panel `rr1-dev-ac7fd952343b0c7d`, split `development`, 240 expected episodes, zero optimizer updates declared.

Provenance: supervisor index, receipts, completions and observability joined per stage; traces and audit events recomputed from the collected directory.

**Not a live policy measurement.** The evidence is scripted, simulated, incomplete or unverified; nothing below is a statement about how a model performs.

Plan problems: the plan is not executable (fixture profile, unbound assets or uncommitted source), so it cannot have produced live evidence

## grpo-matched-v2 (scripted-inference)

- Operational validity: **mechanics_only_not_live**. Scientific suitability: **not_assessed**.
- Episodes: 120 expected; 120 admitted; 120 attempts observed, 120 completed, 120 rewards confirmed, 0 missing; 0 unattempted; 0 unknown.
- Records: 120 trace files, 0 rejected as untrusted, 0 beyond the expected slots (trace evidence for 4 of 4 stages).
- Packets: greedy-development admitted, sampled-packet-1 admitted, sampled-packet-2 admitted, sampled-packet-3 admitted.
- Reasons: the plan is not executable (fixture profile, unbound assets or uncommitted source), so it cannot have produced live evidence; evidence is scripted or simulated, not a live model.
- Greedy: 17/24 rewarded (95% Wilson [0.5083, 0.8509]); contract-valid 24/24; decision correct 17/24 assessable; evidence correct 20/24.
- Sampled: 55/96 rewarded (95% Wilson [0.473, 0.6672]); contract-valid 86/96; decision correct 60/86 assessable; evidence correct 65/86.
- Groups (24): 1 constant-zero, 3 constant-one, 18 mixed with semantic/evidence failures, 2 mixed by format/action only.
- Zero rewards: 41 (10 interface, 31 substantive).
- Typed action errors 15, recovered episodes 11, terminal policy failures 2 (sampled).
- Optimization events: {'optimizer_step': 0, 'gradient_step': 0, 'matched_advantages': 0, 'native_checkpoint_saved': 0}; refresh steps {'greedy-development': [0], 'sampled-packet-1': [0], 'sampled-packet-2': [0], 'sampled-packet-3': [0]}.
- Screen applied to non-live evidence, for mechanics only: screen_passed (18/24 informative groups).

## ppo-mc-matched-v2 (scripted-inference)

- Operational validity: **mechanics_only_not_live**. Scientific suitability: **not_assessed**.
- Episodes: 120 expected; 120 admitted; 120 attempts observed, 120 completed, 120 rewards confirmed, 0 missing; 0 unattempted; 0 unknown.
- Records: 120 trace files, 0 rejected as untrusted, 0 beyond the expected slots (trace evidence for 4 of 4 stages).
- Packets: greedy-development admitted, sampled-packet-1 admitted, sampled-packet-2 admitted, sampled-packet-3 admitted.
- Reasons: the plan is not executable (fixture profile, unbound assets or uncommitted source), so it cannot have produced live evidence; evidence is scripted or simulated, not a live model.
- Greedy: 18/24 rewarded (95% Wilson [0.551, 0.88]); contract-valid 24/24; decision correct 18/24 assessable; evidence correct 22/24.
- Sampled: 45/96 rewarded (95% Wilson [0.3721, 0.5678]); contract-valid 77/96; decision correct 52/77 assessable; evidence correct 53/77.
- Groups (24): 1 constant-zero, 1 constant-one, 19 mixed with semantic/evidence failures, 3 mixed by format/action only.
- Zero rewards: 51 (19 interface, 32 substantive).
- Typed action errors 16, recovered episodes 10, terminal policy failures 3 (sampled).
- Optimization events: {'optimizer_step': 0, 'gradient_step': 0, 'matched_advantages': 0, 'native_checkpoint_saved': 0}; refresh steps {'greedy-development': [0], 'sampled-packet-1': [0], 'sampled-packet-2': [0], 'sampled-packet-3': [0]}.
- Screen applied to non-live evidence, for mechanics only: screen_passed (19/24 informative groups).

## Screen

At least 6 of 24 sampled groups with semantic/evidence variation, across at least 2 case families, with interface failures at most 50% of zero rewards, per algorithm route. This is an engineering heuristic, not a research threshold or proof of learning. It is applied as a decision only to admitted live evidence.
