# 5. PPO-MC step by step

The calculator has returned `36` for `(7+5)*3`. The actor still has to write an answer that the verifier will accept. At this point, how much reward should we expect? A **critic** learns to answer that question from the history the actor can see. If the return is already predictable from the history, the critic can subtract that predictable part. The actor can then give more credit to an unexpectedly good outcome than to an equally good outcome it could already expect.

In [Chapter 4](04-grpo-step-by-step.md), GRPO compared four attempts using the same group average at every action. Here we work through the lab's version of proximal policy optimization with Monte Carlo returns (**PPO-MC**). Keep the same four synthetic attempts. A, B and C earn 0. D earns 1. PPO-MC can assign different advantages even to two actions in D, because calling the calculator and writing the answer happen with different information.

## What could the critic know before token 40?

The actor predicts which token to generate next. The critic reads the same preceding history and predicts the **return**, the total future reward under the collecting policy. With a single binary terminal reward, the expected return is the chance of eventual scored success given that history. The critic emits a scalar from a linear head, so its estimate can fall outside the interval from 0 to 1. The verifier supplies the actual score once the attempt ends.

Use D's invented IDs from [Chapter 3](03-trajectories-tokens-and-fragments.md#line-up-each-prediction-with-its-target). At the input ending in observation `30`, the actor scores target `40` and the critic supplies its value for that same decision. Both use the output **one position before the target**. During training the complete sequence is available, but the causal model can see only the prefix through that input position. [Pinned response slice for log probabilities and values](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/workers/utils/padding.py#L137)

**Predict.** Suppose the critic estimates 0.4 just before D writes its first answer token. D goes on to earn 1. What return should the critic learn to predict at that position? By how much did this particular outcome exceed its prediction?

| History ends in | Next response ID | What happens | Old value | Realized return target | Return minus old value |
|---|---:|---|---:|---:|---:|
| Question and tool instructions, `11` | 20 | Start the tool request | 0.1 | 1 | +0.9 |
| Request token `20` | 21 | Finish the request | 0.2 | 1 | +0.8 |
| Request token `21` | 30 | Harness inserts the observation | excluded | excluded | excluded |
| Tool observation `30` | 40 | Start the final answer | 0.4 | 1 | +0.6 |
| Answer token `40` | 41 | Finish the answer | 0.8 | 1 | +0.2 |

The target at `40` is **1**. We observed the rest of this attempt, so we can use its actual outcome as a **Monte Carlo** sample of the return. Its **raw advantage** is **+0.6**. The final token gets the same realized return and a smaller residual because success was already more expected. These are illustrative predictions. One successful attempt supplies one training example for a history where other continuations could fail.

## How a terminal score reaches an earlier action

Follow D backward. After the last action there are no further rewards. Before the last action there is one reward still to come. Each time we move one action earlier, we add its immediate reward (zero here) to what follows. With no discount, all four actions have return 1. A tool observation changes the next history. Splitting the stored episode at a context change leaves its single scored outcome intact.

For the general case, index **generated actions** by $t=1,\ldots,T$. Write $r_t$ for the reward after action $t$, $V_t$ for the cached critic prediction from the history before it, and $\gamma$ for the discount applied per action. The sampled return at a true terminal endpoint uses zero future value, $V_{T+1}=0$.

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k.
$$

A one-step prediction error compares the old prediction with the reward just received plus the discounted prediction of what remains. **Generalized advantage estimation** (GAE) combines these errors. Its parameter $\lambda$ controls how much weight later errors receive.

$$
\delta_t=r_t+\gamma V_{t+1}-V_t,\qquad
\widehat A_t^{\mathrm{GAE}}=\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}.
$$

For D, set $\gamma=\lambda=1$. Starting before target `40`, the next two one-step errors are $0+0.8-0.4=0.4$ and $1+0-0.8=0.2$. Their sum is 0.6. The intermediate 0.8 occurs once with each sign and cancels. The same cancellation works from every action in rollout $\rho$. All earlier rewards are zero, leaving its terminal score $R_\rho$.

$$
\widehat A_t=\sum_{k=t}^{T} r_k+V_{T+1}-V_t=R_\rho-V_t,\qquad
\widehat G_t=\widehat A_t+V_t=R_\rho.
$$

Watch what the list indexes. Chapter 4’s `[0, 0, 0, 1]` listed the final scores of **four attempts** at one question. In the next command those same digits list the immediate rewards for **four generated tokens within D**. Only its last token is followed by the terminal score. The later whitening exercise switches back to one final score per attempt.

Check the cancellation on a CPU. Paste this and the later commands into a Unix shell at the repository root with Python 3.12. The examples use the standard library.

```bash
python3 - <<'PY'
rewards_after_token = [0, 0, 0, 1]  # one completed attempt, four actions
old_values = [0.1, 0.2, 0.4, 0.8]
next_value = advantage = 0.0     # true terminal endpoint
backward = []
for reward, old in reversed(list(zip(rewards_after_token, old_values, strict=True))):
    delta = reward + next_value - old
    advantage = delta + advantage
    backward.append((round(advantage, 2), round(advantage + old, 2)))
    next_value = old
print("D (raw advantage, return):", backward[::-1])
PY
```

```text
D (raw advantage, return): [(0.9, 1.0), (0.8, 1.0), (0.6, 1.0), (0.2, 1.0)]
```

The matched configuration selects `adv_estimator=gae`, `gamma=1`, `lam=1`. Its executed override looks up the terminal reward by real rollout ID, broadcasts it across the fragments and **directly subtracts the old values**. The backward walk explains the result. The runtime computes it without running a recurrence or adding a reward per fragment. This relies on a complete, scored episode. A prematurely cut trace could need a nonzero estimate of its remaining return. The matched collector rejects incomplete token data, and the harness defines the bounded task's scored endpoint. [Matched advantage calculation](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/matched_runtime.py#L117) · [PPO configuration](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/ppo_contract.py#L69) · [Pinned generic GAE](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/trainer/ppo/core_algos.py#L216) · [GAE paper](https://arxiv.org/abs/1506.02438v6)

## Which positions set the scale? Which attempts set the loss?

First, **whiten** the raw advantages. Subtract the mean and divide by the spread across all valid generated tokens in the collection. This puts the numbers the actor receives on a common scale. Second, **weight** each token's loss so that a long attempt and a short one each get the same total influence in the reduction. The two operations answer different counting questions.

Here is the complete arithmetic fixture. It keeps Chapter 4's masks and lengths. The first token in each attempt has the same prompt and the same old value, 0.1. Later histories can diverge. D's masked observation is marked with a dash.

| Attempt | Terminal score | Old values in response order | Raw advantages at generated positions | Weight per generated token |
|---|---:|---|---|---:|
| A | 0 | 0.1, 0.1 | −0.1, −0.1 | 1/8 |
| B | 0 | 0.1, 0.3, 0.2 | −0.1, −0.3, −0.2 | 1/12 |
| C | 0 | 0.1, 0.6 | −0.1, −0.6 | 1/8 |
| D | 1 | 0.1, 0.2, —, 0.4, 0.8 | +0.9, +0.8, +0.6, +0.2 | 1/16 |

There are $M=11$ valid positions here. Each contributes once to the mean and sample variance. In the study these statistics span **all 32 real rollouts** across eight questions. For a generated position $i$, let $a_i=R_{\rho(i)}-V_i$. With the sums over valid positions, the matched calculation is

$$
\mu=\frac{1}{M}\sum_i a_i,\qquad
s^2=\frac{1}{M-1}\sum_i(a_i-\mu)^2,\qquad
\widetilde A_i=\frac{a_i-\mu}{\sqrt{s^2+10^{-8}}}.
$$

The fixture gives $\mu=1/11$ and $s^2\approx0.228909$. At `40`, whitening takes the raw +0.6 to about +1.064054. The runtime requires at least two valid tokens. It excludes observation and padding positions from both statistics and zeros their policy advantages. Critic return targets keep their original reward scale.

For the loss, let $N$ count real rollouts, $L_\rho$ count all generated targets in rollout $\rho$ across its fragments, and $m_i$ be the target mask. The weight is $w_i=m_i/(N L_{\rho(i)})$. D contributes four samples to whitening. Its four loss weights sum to $4/16=1/4$, exactly the share A gets from its two tokens. Centering under a token average need not give a zero mean under these rollout weights. [Matched weights and actor normalization](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/matched_runtime.py#L102)

## What does each optimizer minimize?

| Training question | Actor | Critic |
|---|---|---|
| Output being trained | Probability of the sampled token given its history | Predicted return from that history |
| Fixed information for this update | Old token log probability and whitened advantage | Cached old value and unwhitened return |
| What clipping compares | Current/old token probability ratio in the Chapter 4 surrogate | Change in the predicted value from its old value |
| Parameters updated | Actor's language model | Separate critic backbone and scalar head |

The actor uses the [Chapter 4 clipped surrogate](04-grpo-step-by-step.md#how-clipping-changes-the-objective), including the asymmetric ratio bounds and dual clipping. The trainer weights each advantage once before summing the masked surrogate. When every ratio equals 1, the actor loss simplifies to $-\sum_i w_i\widetilde A_i$. At `40` its contribution is about $-1.064054/16=-0.066503$.

The critic starts from a separate copy of the prepared Llama backbone, the layers that turn input tokens into numerical representations of their histories. Its token-classification head (`score.weight`, `score.bias`) produces one value per position. The loading check permits the unused language-model output weight. The new head uses a normally initialized weight and zero bias. A dedicated random seed sets the sequence of draws for the critic’s initialization. The code restores the previous random-generator state afterward. Its optimizer trains both the backbone and head. [Critic construction](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/ppo_runtime.py#L379) · [Matched seed and optimizer setup](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/matched_runtime.py#L297)

The critic tries to reduce squared prediction error. For each target it scores both a candidate prediction $v_i$ and a version with the change from $V_i$ clipped to ±0.5. Taking the larger squared error limits the improvement credited to a large jump. On the study's single GPU the objective is

$$
v_i^{\mathrm{clip}}=V_i+\mathrm{clip}(v_i-V_i,-0.5,0.5),
$$

$$
\mathcal L_V=\frac12\sum_i w_i\max\bigl((v_i-R_{\rho(i)})^2,(v_i^{\mathrm{clip}}-R_{\rho(i)})^2\bigr).
$$

Take C's last action. Its old prediction was 0.6 and its target is 0. A candidate value of 0 has squared error 0. The clipped candidate is 0.1 with squared error 0.01. This position contributes $0.5\times(1/8)\times0.01=0.000625$. The critic uses its own rollout-weighted reduction and optimizer. In particular the target for C is still 0 after actor whitening. The backend also scales this loss by the data-parallel size, which was one here. [Matched clipped value loss](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/matched_runtime.py#L167)

## Check the two losses on a CPU

Predict D's total weight if its request and answer occupy separate rows. Then run the calculation with all actor ratios equal to 1 and the critic still at its old values. The last line checks the separate hypothetical change to C.

```bash
python3 - <<'PY'
from math import sqrt
from statistics import mean, variance

# Terminal reward for each attempt. None marks the tool observation.
rewards = [0, 0, 0, 1]
values = [[.1, .1], [.1, .3, .2], [.1, .6], [.1, .2, None, .4, .8]]
raw = [[r - v for v in episode if v is not None]
       for r, episode in zip(rewards, values, strict=True)]
valid = [a for episode in raw for a in episode]
center, spread = mean(valid), sqrt(variance(valid) + 1e-8)
print(f"valid tokens {len(valid)}, mean {center:.6f}, sample variance {variance(valid):.6f}")

actor_loss = critic_loss = 0.0
for name, residuals in zip("ABCD", raw, strict=True):
    weight = 1 / (len(raw) * len(residuals))
    whitened = [(a - center) / spread for a in residuals]
    actor_loss -= weight * sum(whitened)
    critic_loss += 0.5 * weight * sum(a*a for a in residuals)
    print(f"{name} total weight {weight * len(residuals):.2f}")
    if name == "D":
        print(f"token 40 whitened advantage {whitened[2]:.6f}")
print(f"actor loss {actor_loss:.6f}, critic loss at old values {critic_loss:.6f}")

old, candidate, target = values[2][-1], 0.0, rewards[2]
clipped = old + max(-.5, min(.5, candidate - old))
weight = 1 / (len(raw) * len(raw[2]))
loss = .5 * weight * max((candidate - target)**2, (clipped - target)**2)
print(f"C last action with candidate zero: clipped {clipped:.1f}, contribution {loss:.6f}")
PY
```

```text
valid tokens 11, mean 0.090909, sample variance 0.228909
A total weight 0.25
B total weight 0.25
C total weight 0.25
D total weight 0.25
token 40 whitened advantage 1.064054
actor loss 0.203073, critic loss at old values 0.088021
C last action with candidate zero: clipped 0.1, contribution 0.000625
```

D keeps a quarter of the weight if it splits, provided all four generated targets and their original histories are preserved and the two fragments carry D's ID. The actor loss is nonzero at ratio 1 because the negative and positive whitened advantages receive different weights. Each of A's two negative coefficients gets twice the weight of each of D's four positive ones.

**Optional alteration.** In this last command, change D's terminal score to 0, leaving the old predictions fixed. All four returns are now zero. Predict whether the actor still gets a signal. At `40` the raw advantage becomes −0.4. The new token mean is −0.272727 and the whitened advantage is −0.536954. Across attempts the actor loss becomes −0.069516. Values varied across histories, so even a constant-reward collection can yield different actor coefficients. How informative those differences are depends on the critic.

## Did the trained critic predict well?

In the recorded calculator r03 study, each of three PPO-MC runs completed 20 actor updates and 20 critic updates. To test prediction quality, the audit examined the critic **before** each update on newly collected rollouts. It measured mean squared error (MSE) over valid generated tokens. Each token counts once here. The training loss above also has a factor of one half, clipping and equal-rollout weights. [Recorded update counts](../../evidence/phase2-r03/per_seed_results.csv) · [Prediction metric definitions](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/ppo_math.py#L101)

| Recorded PPO-MC seed | Pre-update MSE, first → last collection | Steps with defined explained variance | Steps beating the earlier-data constant on MSE |
|---|---:|---:|---:|
| 2001 | 4.073710 → 0.189495 | 14/20 | 0/19 |
| 2002 | 2.447244 → 0.151124 | 19/20 | 0/19 |
| 2003 | 3.163390 → 0.376402 | 18/20 | 0/19 |

For seed 2001 the first-to-last error fell substantially. Would you choose this critic over always predicting one number? The audit used the mean reward of all earlier real rollouts in that run, a number it could compute *before* the new collection. It scored that constant and the critic on the same current tokens. The constant had smaller MSE on all 57 comparable steps. Each run's first collection lacked earlier data. [Recorded per-seed critic comparison](../../evidence/phase2-r03/SCIENTIFIC_ANALYSIS.md#training-signal-and-critic-behavior)

**Explained variance** asks how much of the variation in returns the predictions account for. The audit computes $1-\mathrm{Var}(V-R)/(\mathrm{Var}(R)+10^{-5})$ using sample variances over valid tokens. A negative value means the residuals varied even more than the targets. All 51 defined values were negative. On nine collections every target was the same, so there was no variation to explain. The audit leaves this metric undefined when target variance is at or below $10^{-8}$. Those nine collections are excluded from the count of 51. [Prediction metrics](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/ppo_math.py#L101)

The optimizer ran, and final pre-update MSE was smaller than initial MSE in each run. The constant comparison shows that the learned critic was a worse predictor of return by MSE on every comparable collection. The study did not isolate the critic's effect on final answer accuracy. The numerical exercises above are synthetic, and the historical [PPO smoke contract](../PHASE2_PPO_PROTOCOL.md) documents an earlier reduction and collection path. The matched override and the [r03 analysis](../../evidence/phase2-r03/SCIENTIFIC_ANALYSIS.md) describe the study used here.

Follow one collection through the two optimizers, policy refresh and checkpoint in [Chapter 6 · One complete training update](06-one-complete-training-update.md).
