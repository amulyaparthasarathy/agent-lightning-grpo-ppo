"""One sampled turn per fragment. The harness owns episode and policy identity."""

from dataclasses import dataclass
import math
from statistics import mean, stdev

import tinker


GROUP_SIZE = 4
PROMPT_LIMIT = 512
OUTPUT_LIMIT = 96


@dataclass(frozen=True)
class Turn:
    policy: str
    sequence_id: str
    prompt: list[int]
    tokens: list[int]
    logprobs: list[float]
    stop: str

    def validate(self):
        if self.stop != "stop" or not self.sequence_id:
            raise ValueError("Incomplete sampled turn")
        if not 0 < len(self.prompt) <= PROMPT_LIMIT or not 0 < len(self.tokens) <= OUTPUT_LIMIT:
            raise ValueError("Invalid prompt or generation length")
        if any(type(t) is not int or t < 0 for t in self.prompt + self.tokens):
            raise ValueError("Invalid token ID")
        if len(self.logprobs) != len(self.tokens) or any(
            not math.isfinite(x) or x > 0 for x in self.logprobs
        ):
            raise ValueError("Missing or invalid behavior log probabilities")


@dataclass(frozen=True)
class Episode:
    episode_id: str
    case_id: str
    reward: float
    turns: list[Turn]


def training_batch(episodes, case_id, policy):
    """One question group per collection. Each real episode owns 1/G of the mass."""
    if len(episodes) != GROUP_SIZE or len({e.episode_id for e in episodes}) != GROUP_SIZE:
        raise ValueError("Expected four distinct attempts")
    rewards = [e.reward for e in episodes]
    if any(r not in (0, 1) for r in rewards):
        raise ValueError("Expected binary verified rewards")
    center, spread = mean(rewards), stdev(rewards)
    rows, evidence, sequences = [], [], set()
    for episode in episodes:
        if episode.case_id != case_id or not episode.turns:
            raise ValueError("Incomplete or mixed question group")
        for turn in episode.turns:
            turn.validate()
            if turn.policy != policy or turn.sequence_id in sequences:
                raise ValueError("Stale policy or reused sampled sequence")
            sequences.add(turn.sequence_id)
        advantage = (episode.reward - center) / (spread + 1e-6) if spread else 0.0
        generated = sum(len(t.tokens) for t in episode.turns)
        weight = 1 / (len(episodes) * generated)
        for turn in episode.turns:
            ids = turn.prompt + turn.tokens
            context = len(turn.prompt) - 1
            mask = [0] * context + [1] * len(turn.tokens)
            inputs = {
                "target_tokens": ids[1:],
                "logprobs": [0.0] * context + turn.logprobs,
                "advantages": [m * advantage * weight for m in mask],
            }
            rows.append(tinker.Datum(tinker.ModelInput.from_ints(ids[:-1]), inputs))
            evidence.append(dict(episode_id=episode.episode_id, sequence_id=turn.sequence_id,
                                 policy=policy, reward=episode.reward, advantage=advantage,
                                 weight_per_generated_token=weight, mask=mask, **inputs))
    return rows, evidence, spread > 0
