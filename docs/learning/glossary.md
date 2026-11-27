# Glossary and notation

## Shared notation

| Symbol | Meaning |
|---|---|
| $q$ | Question presentation/group; four sampled rollouts in this protocol |
| $\rho$ (`rho`) | One complete sampled rollout with its own identity |
| $j$ | Fragment/training-row index within a rollout |
| $t$ | Position within a fragment’s response in the mask and weight formulas (Chapter 3). In the return formula (Chapter 5), it counts only generated actions in time order. |
| $R_\rho$ | Recorded terminal binary reward for a complete rollout |
| $h_t$, $a_t$ | Exact conditioning history and sampled action token |
| $\pi_\theta$ | Actor's token probability distribution, with parameters $\theta$ |
| $V_\phi$ | Separate learned critic prediction, with parameters $\phi$ |
| $m_{\rho,j,t}$ | Generated-token mask: 1 for actions, 0 for context/padding |
| $L_\rho$ | Generated-token count across the complete rollout |
| $N$ | Real rollouts in the collection; 32 in a study iteration |
| $w_{\rho,j,t}$ | Token weight $m_{\rho,j,t}/(N L_\rho)$ |
| $G_t$, $A_t$ | Return target and advantage at an action position |
| $\gamma$, $\lambda$ | Discount and GAE mixing parameters; both 1 here |
| $u$ | Current/old policy probability ratio; never a rollout identifier |
| $M$ | Number of valid generated tokens in the whitening population |

## Terms used in the lessons

| Term | Meaning and common confusion |
|---|---|
| Actor / policy | Model producing action tokens, including tool requests. It is not the calculator program. |
| Harness | Message/context management, tools, execution limits and control flow around the model. |
| Critic / value | Learned prediction of return from a history; not the reward scorer or an independent verifier. |
| Reward | Recorded endpoint feedback, binary in this study. |
| Return | Reward accumulated from an action onward; here terminal reward on every valid action. |
| Advantage | Return relative to a baseline. Raw, whitened and loss-weighted advantages differ. |
| Whitening | Subtract the mean raw advantage over generated tokens, then divide by the square root of the sample variance plus the numerical stabilizer. |
| Weighting | Assigning each complete rollout equal total loss weight after advantage construction. |
| Group | Four sampled attempts for one question presentation, used for GRPO statistics. |
| Rollout / trajectory | One complete sampled agent attempt; may contain several model/tool calls. |
| Fragment / training row | Representation of part of a rollout with a valid token context. It is not another independent attempt. |
| Response mask | Selects generated action terms. Observations can be inside a response-shaped tensor while masked out. |
| Microbatch | Memory-sized subset for forward/backward computation. |
| Minibatch | Set whose accumulated gradient feeds one optimizer call; the full collection here. |
| PPO epoch | One pass over a collected batch; one here. It is not a pass over the entire question dataset. |
| Optimizer call | Actual step invocation, distinguished from a gradient or a measured parameter change. |
| Weight refresh | Copy/synchronize actor parameters into inference before further sampling/evaluation. |
| Native checkpoint | Saved model, optimizer and extra state; not automatically a deployment export. |
| Independent reload | Separate evaluation-only worker loading the recorded final checkpoint, with zero optimizer calls. |
| Baseline evaluation | Prepared instruction-tuned actor with the tool harness, before this run's RL. |
| Development panel | A set of cases used to examine behavior during development. The calculator study reused 64 questions and recovery P02 reused 24 cases. |
| Percentage point (pp) | Difference between percentages, distinct from percentage relative improvement. |

Definitions follow the [source assignments](source-map.md). Return to the [learning index](README.md).
