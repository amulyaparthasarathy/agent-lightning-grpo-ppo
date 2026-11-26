# Seed 2001 functional pilot

The closed scope `seed-2001-development-only` runs a 24-episode greedy base evaluation, 512 sampled training episodes over sixteen updates from fresh base, and a 24-episode development evaluation in a fresh worker from the terminal step-16 checkpoint. The two evaluations use seed 9001. Training uses the frozen 128-case order, eight questions per update and four rollouts per question. The supervisor persists the first-seed technical admission on completion. Independent review requires the gate and recomputes all three admissions. The pilot has no final-release operation.

The synthetic profile disables dispatch. A live pilot requires matching scope and authority, fresh owned resources, actual asset binding and durable same-plan CPU preflight. The full eleven-stage profile requires its own distinct authority. [Pilot template](../examples/reader-profile.recovery-pilot.synthetic.json) · [Study protocol](PHASE4C_LEARNING_STUDY_PROPOSAL.md)

One H100 lane has 16 CPUs and 192 GiB. The training deadline is 14,400 seconds and each evaluation has 1,800 seconds. At $7/hour over 18,000 worker and 18,600 supervisor seconds, compute exposure is $71.17, rounded up to $72. Adding $15 build, $5 preparation/preflight, $7 storage, $2 collection and $2 delayed billing yields a $103 reservation. The cumulative pilot ceiling is $150 before credits. Any engineering replacement needs its own full reservation after reconciling prior exposure. The independent qualification and Phase 4B ledgers retain their original liabilities.

## Pre-dispatch verification

Tested source `9d12ce32ff05b22bf7a3fd5295bb09819f5a6d8d` passed 202 reader offline tests, 74 pinned CPU tests and four scripted execution tests, with zero failures or skips. The four tests include the complete pilot, retained eleven-stage flow, interrupted submission and gross billing. The pilot fixture admitted 560 confirmed outcomes after removing the provider mount and found exactly three worker calls. It verified the terminal technical gate, optimizer calls 1–16 and refreshes 0–16. The pinned gate replays the upstream loader and exercises sixteen actual AdamW updates on a tiny CPU model. Provider calls, checkpoint payloads and inference were simulated in the local flow.

```bash
bash scripts/test_reader_offline.sh
OMP_NUM_THREADS=1 bash scripts/test_matched_training.sh
PYTHONPATH=.:tests "$TASK_TEST_PYTHON" -m unittest -v test_study_execution
```

The pinned command used the existing `MATCHED_TEST_PYTHON`, `AGL_PINNED_SOURCES`, `AGL_VLLM_WHEEL` and `AGL_TRANSFORMERS_WHEELS` fixtures. The original study's tested source remains `b3912e9cb8382f3121ab069d370d23e829fa4d93`. A03's execution source remains `7fdb2b1e0ce2a05f03980cfd3a1e07a31937ca6a`.
