# 7. Read the experiment results

**Opening question: What do the six runs establish, and what do they leave unanswered?**

First identify the measurement and its denominator. Here, accuracy is the fraction of 64 development questions receiving recorded reward 1 under the frozen extractor and numeric scorer. A higher score can reflect numerical answers, output formatting or both. It is not an independent semantic grade.

## Which experiment are we reading?

Study **`agl-explore-20-v1-r03`** ran at source **`5fb6adfd48ee245509f7dff3ab1d849fcf707d86`**. Each arm starts from the prepared `meta-llama/Llama-3.2-3B-Instruct` model before this lab's RL. Baseline means that instruction-tuned model running the same tool harness, not a no-tool model or the upstream Qwen starter.

| Quantity | Correct denominator |
|---|---|
| Available training pool | 256 questions |
| One study collection | Eight question presentations, four real rollouts each: 32 trajectories |
| One study run | 20 actor calls, 160 question presentations, 640 trajectories |
| One PPO-MC study run | Additionally 20 critic calls |
| Six study runs | 3,840 trajectories; 120 actor calls and 60 critic calls |
| Study stages | 18: baseline, training and checkpoint-eval for each run |
| Qualification | Seed 9101, two steps per arm, six additional stages; excluded from study results |
| Evaluation | Reused, non-fresh 64-question development panel |

The 640 trajectories are not 640 unique questions. Twenty updates are not 20 dataset epochs. Reusing the panel across six runs does not create 384 independent test questions. Phase 1 results and earlier failed attempts are separate records.

Study seeds **2001, 2002, 2003** pair question-batch exposure across arms. Python/NumPy/Torch, sampling, optimizer and row-batching streams are algorithm-specific; PPO also has a critic initialization stream. Pairing does not promise identical continuations, token counts, compute budgets or bitwise results.

## Preserve every baseline and paired outcome

| Algorithm | Seed | Baseline correct | Final reload correct | Improvement, percentage points |
|---|---:|---:|---:|---:|
| GRPO | 2001 | 27/64 | 36/64 | +14.062500 |
| PPO-MC | 2001 | 26/64 | 30/64 | +6.250000 |
| GRPO | 2002 | 27/64 | 30/64 | +4.687500 |
| PPO-MC | 2002 | 25/64 | 34/64 | +14.062500 |
| GRPO | 2003 | 26/64 | 34/64 | +12.500000 |
| PPO-MC | 2003 | 26/64 | 34/64 | +12.500000 |

| Algorithm | Mean baseline | Mean final | Mean improvement |
|---|---:|---:|---:|
| GRPO | 41.666667% | 52.083333% | +10.416667 pp |
| PPO-MC | 40.104167% | 51.041667% | +10.937500 pp |

![Recorded paired baseline and final accuracy](../../evidence/phase2-r03/figures/paired_baseline_final_accuracy.svg)

This is the **existing committed SVG**, using the [per-seed CSV](../../evidence/phase2-r03/per_seed_results.csv). It shows baseline and independent final-reload endpoints, not accuracy after every training step. Some endpoint labels crowd together at reading width; use the table above for exact counts, including the overlapping seed 2003 endpoints. Each endpoint is a count out of the same 64 questions; each run retains its own baseline. The [figure index](../../evidence/phase2-r03/figures/INDEX.md) and [manifest](../../evidence/phase2-r03/manifest.json) describe provenance and byte identity.

Let $B_{a,s}$ and $F_{a,s}$ be baseline and final accuracy fractions for algorithm $a$ and seed $s$. With $G$ for GRPO and $P$ for PPO-MC,

$$
D_{F,s}=100(F_{G,s}-F_{P,s}),\qquad
D_{\Delta,s}=100[(F_{G,s}-B_{G,s})-(F_{P,s}-B_{P,s})].
$$

Both differences are in percentage points. They answer different questions.

| Seed | GRPO minus PPO final, pp | Difference in improvements, pp |
|---|---:|---:|
| 2001 | +9.375000 | +7.812500 |
| 2002 | -6.250000 | -9.375000 |
| 2003 | 0.000000 | 0.000000 |

These [paired contrasts](../../evidence/phase2-r03/paired_differences.csv) change sign. Mean final contrast is +1.041667 pp; mean improvement contrast is -0.520833 pp. Baseline differences explain their opposite directions arithmetically; their underlying cause was not established. Do not replace the observed baselines with a shared value. Neither superiority, equivalence, statistical significance nor fresh-holdout generalization was established.

## Read mechanism diagnostics in their own units

| Recorded observation | Supported interpretation |
|---|---|
| Eight GRPO steps had all-constant groups, zero current gradients and actual optimizer calls | Relative reward signal can vanish while optimizer calls still occur |
| Near-zero GRPO scalar losses often accompanied nonzero gradients | Centered scalar contributions need not cancel vector derivatives |
| PPO critic MSE decreased but lost to the earlier-data constant on all 57 comparable steps | Initial prediction error fell without demonstrating advantage over that baseline |
| All 51 defined EV observations were negative; nine were undefined | Predictive diagnostics remained weak; constant-target EV stays undefined |
| Actor clip fractions and measured pre-update PPO-style KL were zero | This one-update-per-collection regime did not exercise repeated-epoch clipping benefits |

For every row, the limitation is the same kind of distinction: a mechanism diagnostic is not a causal explanation of final task performance. In particular, these observations do not prove that the critic caused a worse actor, that zero pre-update KL means no later drift, or that every optimizer call moved parameters.

Tool diagnostics also need their own denominator. Tool errors count recorded call spans, not necessarily failed tasks; an episode can recover after an error. Error-free calculation can evaluate the wrong expression. Across three baseline-to-final evaluations, missing markers fall from 77 to 2 for GRPO and 81 to 16 for PPO-MC, out of 192 repeated-panel episodes per endpoint. The audit reports zero reward for every missing-marker episode. This is consistent with a formatting contribution but does not quantify a causal share of improvement. Recorded scoring was not replaced by alternative grading.

## Sources and unavailable measurements

The [scientific analysis](../../evidence/phase2-r03/SCIENTIFIC_ANALYSIS.md) and [verification ledger](../../evidence/phase2-r03/VERIFICATION.md) qualify these findings. [`matched_exploratory.SEEDS` and `STEPS`](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/matched_exploratory.py#L6), [`matched_contract.streams`](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/matched_contract.py#L37) and [`complete_stage`](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/matched_launch.py#L183) establish design and completion rules. The [frozen scorer](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/observed_calc_agent.py#L301) defines the endpoint. [`check_phase2_results.main`](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/scripts/check_phase2_results.py#L33) checks committed derived results and was added **after** execution.

The private local audit's 18,231-row `training_metrics_long.csv` and 49-SVG collection are absent from both this selected Git bundle and the teaching ZIP. No new empirical curves are drawn. Full-batch raw token advantages are unavailable beyond bounded samples; entropy and reference-policy KL are unavailable. Do not fill missing values with zeros. The guide does not repeat raw trace, remote tensor, model or data verification.

## Offline exercise and answer

**Exercise:** For seed 2001, calculate the final-score contrast and the improvement contrast. Why are they unequal? Does seed 2003's tie mean identical case-level behavior?

**Answer:** Final contrast is $(36-30)/64\times100=9.375$ pp. Improvement contrast is $[(36-27)-(30-26)]/64\times100=7.8125$ pp. The baselines differ by one correct answer. Seed 2003's final totals tie at 34/64, but the paired CSV records seven cases only GRPO solves and seven only PPO solves. Equal totals do not mean identical behavior.

Next: [reproduce and operate within scope](08-reproduce-and-operate.md).
