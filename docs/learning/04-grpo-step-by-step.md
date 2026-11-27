# 4. GRPO step by step

How can a correct calculator answer teach the actor what to do next time? Follow four attempts at `(7+5)*3` from their final scores to one actor loss. You can check the whole calculation with Python on a CPU.

In [Chapter 3](03-trajectories-tokens-and-fragments.md), the actor requested the calculation, saw the tool return `36`, and wrote the correct final answer. Call that episode D. Now imagine sampling three other attempts at the **same question** from the same policy. Keeping the question fixed means a high score cannot come from drawing an easier problem. Different choices can still lead to different final answers. Suppose A, B and C each earn 0 and D earns 1. These four synthetic episodes form one comparison group.

**Predict.** If all four had succeeded, what could their rewards tell us about which attempt to favor? If only D succeeded, should it get twice as much influence just because the adapter stored it in two fragments?

## Compute the group-relative advantage

Group relative policy optimization (GRPO) uses the mean of all four outcomes on this question as a baseline. It needs no separate model to predict the reward. Start by subtracting the group's mean reward from each episode's reward. This **centering** gives a positive sign to an above-average outcome and a negative sign to a below-average one.

| Episode | Terminal reward | Subtract the mean (0.25) | Divide by sample standard deviation plus epsilon |
|---|---:|---:|---:|
| A | 0 | −0.25 | −0.499999 |
| B | 0 | −0.25 | −0.499999 |
| C | 0 | −0.25 | −0.499999 |
| D | 1 | +0.75 | +1.499997 |

The last column is the **advantage**, rounded to six places. It measures how far this attempt finished above or below the group average on a scale set by the group's spread. For a group $q$ of four distinct rollouts, let $R_\rho$ be the reward of rollout $\rho$. The mean $\bar R_q$ and sample standard deviation $s_q$ are

$$
\bar R_q=\frac{0+0+0+1}{4}=0.25,\qquad
s_q=\sqrt{\frac{3(0-0.25)^2+(1-0.25)^2}{4-1}}=0.5.
$$

The implementation uses `std(unbiased=True)`. After centering, the four deviations must sum to zero. Once you know three, the fourth is fixed, which is why the sample variance uses a denominator of three. The correction applies to the variance before taking its square root. A small constant, epsilon, keeps the division defined when the spread is zero.

$$
A_\rho=\frac{R_\rho-\bar R_q}{s_q+10^{-6}}.
$$

For four successes, every centered reward is $1-1=0$. For four failures, each is $0-0=0$. Both groups yield four zero advantages. The matched trainer retains these groups. Their outcomes count toward performance, but they supply no relative preference for their actions in this update. [Matched advantage calculation](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/matched_runtime.py#L117)

## Which decisions receive D's advantage?

Bring back D's response IDs and mask. Each generated token inherits the episode's advantage, including the tokens that requested the tool before the reward was known. The verifier gave one score for the whole attempt. It did not identify which individual choice caused success.

| Response target ID | 20 | 21 | 30 | 40 | 41 |
|---|---:|---:|---:|---:|---:|
| Source | request | request | tool observation | final answer | final answer |
| Action mask | 1 | 1 | 0 | 1 | 1 |
| Advantage before weighting (approx.) | +1.5 | +1.5 | 0 | +1.5 | +1.5 |
| Weight in this four-episode example | 1/16 | 1/16 | 0 | 1/16 | 1/16 |

Observation `30` is visible when the actor predicts `40`. A request repeated in that second call's prompt supplies history. Its sampled tokens already count on the first call. If D needs two fragments, its four action positions still share one rollout ID, one reward and one quarter of this example's total weight.

## From advantage to a differentiable loss

Take target `40` at the history ending in `30`. Write $a$ for that sampled token and $h$ for its full preceding history. The actor assigns it probability $\pi_\theta(a\mid h)$, where $\theta$ denotes the actor's current weights. Read the vertical bar as *given*. The same token can have a different probability after a different observation. The **importance ratio** compares the current probability of the observed action with its probability under the old policy used for collection. A ratio of 1 means equal probability, and a ratio above 1 means the token has become more likely at that history.

The sampler records `rollout_log_probs`. In this matched run the trainer recomputes `old_log_probs` on the captured histories before updating. It holds those old values fixed and evaluates the current actor on the same targets. The denominator in the loss uses those recomputed old values. Sampling-versus-recomputation differences have their own diagnostics. [Trainer's old-policy pass](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/trainer.py#L583) · [Matched run settings](../../experiment/ppo_contract.py)

A log probability is the natural logarithm of a probability. Subtracting two logs and exponentiating gives a ratio. The implementation first bounds the log difference for numerical stability.

$$
d=\mathrm{clip}(\log\pi_\theta(a\mid h)-\log\pi_{\rm old}(a\mid h),-20,20),\qquad u=\exp(d).
$$

Suppose the old probability of target `40` is 0.25 and its current probability is 0.40. Then $u=0.40/0.25=1.6$. Without objective clipping, minimizing the loss $-A_\rho u$ would favor increasing this ratio for D's positive advantage and decreasing it for a negative advantage.

### How clipping changes the objective

The lab compares that direct loss with a second version using the ratio clipped to $[0.8,1.3]$. It takes the larger loss, so moving farther in a favorable direction eventually brings no further benefit. For negative advantages it also caps an unusually large loss, a rule called **dual clipping**. For a single unweighted token with advantage $A$, the surrogate (the loss used to guide the update) is

$$
\ell^+=\max\bigl(-Au,-A\cdot\mathrm{clip}(u,0.8,1.3)\bigr),\qquad
\ell=\begin{cases}
\min(\ell^+,-3A),& A<0,\\
\ell^+,& A\ge 0.
\end{cases}
$$

Unit advantages make the three clipping cases easy to check by hand.

| $A$ | Ratio $u$ | Direct loss | Clipped-ratio loss | Larger of the two | Final loss after dual clip |
|---:|---:|---:|---:|---:|---:|
| +1 | 1.6 | −1.6 | −1.3 | −1.3 | −1.3 |
| −1 | 0.5 | 0.5 | 0.8 | 0.8 | 0.8 |
| −1 | 4.0 | 4.0 | 1.3 | 4.0 | 3.0 |

For the last row the sampled action did poorly, yet became four times as likely. The ordinary rule keeps the larger penalty of 4. The dual clip limits it to 3. These bounds shape the loss. Ratios such as 1.6 and 4 remain possible. [Pinned actor loss](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/per_rollout_loss.py#L46) · [Clip settings](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/ppo_contract.py#L105)

## Combine the token losses

Average the generated-token losses **within each rollout**, then average those rollout means. With $N$ rollouts and $L_\rho$ generated targets across all fragments of rollout $\rho$, the [Chapter 3 weight](03-trajectories-tokens-and-fragments.md#equal-weight-for-each-real-rollout) for position $t$ of fragment $j$ is $w_{\rho,j,t}=m_{\rho,j,t}/(N L_\rho)$, where $m$ is the action mask. The total loss is

$$
\mathcal L_{\rm actor}=\sum_{\rho,j,t} w_{\rho,j,t}\cdot\ell_{\rho,j,t}.
$$

The actual matched code scales the advantages by these weights **once** and sums the masked losses. This gives the same scalar because a nonnegative weight factors through both clipping comparisons. The full study used $N=32$ across eight question groups. Our example contains one group, so $N=4$. [Rollout weighting](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/89fe32f92f29209803c13264baed951713741d42/experiment/matched_runtime.py#L102)

When every ratio is 1, each rollout's mean loss is its negative advantage. Those four contributions cancel. Now change only target `40` to ratio 1.6. Its clipped loss changes from approximately $-1.5$ to $-1.5\times1.3=-1.95$. It has weight $1/16$, so the whole example's loss changes by about $-0.45/16=-0.028125$.

## Compute the four-episode loss in Python

From the repository root, paste the following into a Unix shell with Python 3.12. It needs only the standard library. A, B and C have illustrative response lengths. D uses the mask from Chapter 3. Predict which printed value will change when only D's first answer token becomes more likely.

```bash
python3 - <<'PY'
from math import exp, log
from statistics import mean, stdev

rewards = [0, 0, 0, 1]
center, spread = mean(rewards), stdev(rewards)
advantages = [(reward - center) / (spread + 1e-6) for reward in rewards]
masks = [[1, 1], [1, 1, 1], [1, 1], [1, 1, 0, 1, 1]]

def token_loss(advantage, old_logp, current_logp):
    delta = min(20, max(-20, current_logp - old_logp))
    ratio = exp(delta)
    clipped_ratio = min(1.3, max(0.8, ratio))
    loss = max(-advantage * ratio, -advantage * clipped_ratio)
    return min(loss, -3 * advantage) if advantage < 0 else loss

def group_loss(probabilities):
    episode_means = []
    for advantage, mask, current in zip(advantages, masks, probabilities, strict=True):
        losses = [token_loss(advantage, log(0.25), log(p))
                  for keep, p in zip(mask, current, strict=True) if keep]
        episode_means.append(mean(losses))
    return mean(episode_means)

current = [[0.25] * len(mask) for mask in masks]
print("mean, sample std:", center, spread)
print("advantages:", [round(a, 6) for a in advantages])
print("loss at old probabilities:", round(group_loss(current), 6))
current[3][3] = 0.40  # D's token 40, immediately after the tool observation
print("loss after changing token 40:", round(group_loss(current), 6))
PY
```

For these four attempts, Python prints the following values.

```text
mean, sample std: 0.25 0.5
advantages: [-0.499999, -0.499999, -0.499999, 1.499997]
loss at old probabilities: 0.0
loss after changing token 40: -0.028125
```

## Why near-zero loss can still yield a gradient

A loss value tells you the height of the loss at the current weights. A **gradient** tells the optimizer which way that height changes if it nudges the weights. At ratio 1 the four centered scalar contributions cancel. The slopes can differ because the actor predicts different tokens at different histories.

To isolate that distinction, consider a separate toy with two possible actions and one adjustable number $\theta$. Let the first action have probability $p=\sigma(\theta)=1/(1+e^{-\theta})$ and the second probability $1-p$. Give them fixed advantages +1 and −1, equal weight, and old probability $1/2$ each. Near $\theta=0$, clipping is inactive.

$$
\mathcal L(\theta)=-\tfrac12\left[\frac{p}{1/2}(1)+\frac{1-p}{1/2}(-1)\right]=1-2\sigma(\theta).
$$

At zero, the scalar is zero. A small increase in $\theta$ raises the probability of the favored action and lowers the probability of the other. Both changes reduce the loss. Estimate its slope by evaluating the loss a small distance $h$ on either side of zero, in the same shell.

```bash
python3 - <<'PY'
from math import exp

def loss(theta):
    return 1 - 2 / (1 + exp(-theta))

h = 1e-5
print("loss, slope:", loss(0), round((loss(h) - loss(-h)) / (2*h), 6))
PY
```

The output is `loss, slope: 0.0 -0.5`. Analytically, $\sigma'(0)=1/4$, so the derivative is $-2/4=-1/2$. Substituting the constant 1 for the ratio *before* differentiating would erase this dependence on the weights.

**Optional alteration.** In the four-attempt command, change only `rewards` to `[1, 1, 1, 1]`. The mean becomes 1 and the spread becomes 0. Every advantage and both losses become zero, even after the probability change. Here the current group-relative policy gradient is zero too, because every sampled action has coefficient zero.

## What did the calculator study measure?

The three GRPO study runs had **110 informative groups out of 480**. An informative group contained different rewards for the same question. On eight of the 60 GRPO steps, all eight groups were constant. The current actor gradient was zero on those steps and the optimizer was called. On other steps, near-zero scalar losses coexisted with nonzero gradients. An optimizer with momentum or weight decay can still move weights on a zero-current-gradient step. The audit did not measure every parameter change. [Recorded calculator analysis](../../evidence/phase2-r03/SCIENTIFIC_ANALYSIS.md)

The study recorded zero actor clipping fractions with one optimizer update per collection. The changed probabilities and active clipping above are synthetic arithmetic to show how the objective works. They add no observed model behavior or evidence of improved performance. [Study diagnostics](../../evidence/phase2-r03/SCIENTIFIC_ANALYSIS.md) · [GRPO method reference](https://arxiv.org/abs/2402.03300v3)

Continue to [Chapter 5 · PPO-MC step by step](05-ppo-mc-step-by-step.md) to see what changes when a critic predicts the eventual return.
