# Offline integration checks — 2026-11-15

Checks use the existing aarch64 client and pinned CPU environments, Python 3.12.3 (GCC 13.3.0). They do not run models or qualify a new training source.

The client interpreter is `/workspace/projects/repos/agent-lightning-exploratory/.venv/bin/python`, Modal 1.6.3. The pinned interpreter is `/workspace/projects/repos/agent-lightning-lab/.local/matched-test-venv/bin/python`, Torch 2.11.0+cpu, NumPy 1.26.4, TensorDict 0.10.0, OmegaConf 2.3.0, Hydra 1.3.2, Ray 2.60.0 and pyzmq 27.3.0. The established suite verifies these versions and the retained upstream source hashes. No environment was created for the suite and no package was installed/upgraded.

Results and final command details are recorded below. Historical scientific audit verdicts are in VERIFICATION.md, not inferred from these integration tests.

## Completed checks

| Check | Actual result |
|---|---|
| Client suite | 286 discovered, **273 passed**, 13 expected skips, exit 0 (19.174 s). |
| Final pinned CPU suite | 61 discovered, **60 passed**, one expected wheel-install fixture skip, exit 0 (33.015 s). All numerical, config and real Ray/ZeroMQ IPC tests passed. |
| Fresh wheel-install fixture | **UNVERIFIED in this closeout**: deliberately not enabled because it installs wheels in a temporary venv. No install was performed. |
| Retained wheel regression | Earlier PASS retained in [a01 tests](../phase2-exploratory/failure-a01/pinned-tests.txt) and [r02 tests](../phase2-exploratory/failure-r02/pinned-tests.txt). Its test, source-identity implementation, source lock and image declaration are byte-identical to `608a3e5` and the executed source. Both existing wheel hashes match the tested lock/fixture. See [retained scope hashes](retained-wheel-test-scope.json). This is retained evidence, not a fresh pass. |
| Derived result command | Six runs, all 24 stage summaries, three paired contrasts and manifest SHA256/size checks pass. Means reproduce the closeout table; malformed denominators and out-of-range counts are rejected. |
| Documentation and preservation | Relative file links and Markdown heading targets pass; selected SVG parses as XML. All **63** files in the original local audit `final/` tree retain their pre-closeout SHA256 hashes. |
| Changed-file review | Explicit documentation/evidence/one-analysis-script allowlist; credential/token/signed-URL pattern scan and manual content review. No raw trajectories, tensors, dataset archives, environment dumps or PDF added. This is not a history-wide publication clearance. |
| Git whitespace / source | `git diff --check` and staged diff check pass. Existing runtime, build, dependency, test, numerical-contract and safety-gate files are unchanged relative to executed `5fb6adf`. Only documented Markdown changes, this evidence bundle and the standard-library checker are allowed. |

The 13 client skips comprise nine numerical pinned-training tests, one pinned-config test, two IPC tests, and the wheel-install fixture. The first 12 are exercised by the final pinned run; the last has the retained evidence above. GPU/model execution, independent tensor rehash, fresh holdout, paid cold build and formal ten-job admission/campaign are UNVERIFIED by these offline checks and were not attempted.

Two setup failures are retained honestly in local logs: the first pinned invocation selected a nonexistent vLLM wheel path and lacked Ray's hostname address inside its isolated namespace (61 tests, two failures, one skip; exit 1). Correcting the wheel path alone left one IPC timeout because Ray selected `172.17.0.2`, which still was not local to that namespace (61 tests, one failure, one skip; exit 1). Adding that address **only to the isolated namespace's loopback** resolved the setup failure. The final run had no failures. No test or runtime code was changed to obtain a pass.

## Actual commands and network boundary

From `/workspace/projects/repos/agent-lightning-exploratory`, the client suite ran:

```bash
unshare -Urn bash -c 'ip link set lo up && env -i PATH="$PATH" HOME="$HOME" LANG=C.UTF-8 MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline RAY_USAGE_STATS_ENABLED=0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v' > .local/closeout-20261115/client-tests.txt 2>&1
```

The successful pinned invocation was:

```bash
unshare -Urn bash -c 'ip link set lo up && ip addr add 172.17.0.2/32 dev lo && ip -brief address && ip route && env -i PATH="$PATH" HOME="$HOME" LANG=C.UTF-8 MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline RAY_USAGE_STATS_ENABLED=0 PYTHONDONTWRITEBYTECODE=1 AGL_PINNED_SOURCES=/workspace/projects/repos/agent-lightning-lab/.local/ppo-review AGL_VLLM_WHEEL=/workspace/projects/repos/agent-lightning-exploratory/.local/failure-r02/vllm-0.22.0-cp38-abi3-manylinux_2_28_x86_64.whl MATCHED_TEST_PYTHON=/workspace/projects/repos/agent-lightning-lab/.local/matched-test-venv/bin/python bash scripts/test_matched_training.sh' > .local/closeout-20261115/pinned-tests-final.txt 2>&1
```

`unshare -Urn` gives the test processes a separate user/network namespace. Only loopback is enabled, with no external interface/default route; the added address is local within that namespace, not a connection to the host/container network. Ray's external address-discovery attempts fail there and its hostname fallback stays local. Subprocesses inherit this boundary, including native Ray and ZeroMQ. Existing tests use local fixture servers/processes only. Credentials are not loaded into the clean environment, Modal config points to `/dev/null`, hosted telemetry is disabled, and HF/Transformers are offline. The `--prepare` path was not used.

The first two pinned commands used the same environment/entrypoint, without `ip addr add`, `ip -brief address` or `ip route`. The first used the erroneous `AGL_VLLM_WHEEL=/workspace/projects/repos/agent-lightning-lab/.local/ppo-review/vllm-0.22.0-cp38-abi3-manylinux_2_31_x86_64.whl`; the second used the final correct wheel path. Logs remain at `.local/closeout-20261115/{pinned-tests,pinned-tests-corrected}.txt`; their hashes and counts are in [offline_checks.json](offline_checks.json). No failed log was overwritten.

Additional commands actually performed:

```bash
python3 scripts/check_phase2_results.py
python3 .local/closeout-20261115/check_bundle.py
git diff --check
git diff --cached --check
```

`check_bundle.py` is a retained local closeout helper: changed-path allowlist, relative-link/anchor checks, original-audit hashes, sensitive-value patterns and SVG XML checks. The durable result checker is the committed script; neither script invokes Modal/W&B/model access. A direct Python call to its `count()` function rejected `27/63`, `65/64`, and `-1/64`.

## GitHub integration boundary

The initial refresh matched all three supplied PR heads and main. GitHub reported private visibility, ordinary merge commits enabled, automatic branch deletion disabled, no rulesets, and no branch protection for main or the three feature branches. All three PRs had no review submissions or inline review threads. Their author discussion concerns are dispositioned in the closeout; author comments are not independent approval.

No `.github` files exist at any of the four inspected heads and the Actions workflows API returned zero workflows. Check-runs and legacy status contexts were empty at each execution head. The legacy combined-status API says `pending` for an empty list; there is no actual pending check or required context in that result. Workflow, branch identity and review/check requirements are rechecked immediately before each authorized merge; no protection or required review is bypassed. Actual merge SHAs and final-main verification are recorded in the PR descriptions and closeout receipt after integration.
