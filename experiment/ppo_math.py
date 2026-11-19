# VERL-derived gae/whiten ports:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Agent Lightning-derived normalize_actor portion:
# Copyright (c) Microsoft. All rights reserved.
# Microsoft MIT terms: experiment/LICENSE-AgentLightning.txt.
# Modified for Agent Lightning Lab: standard-library ports of masked GAE,
# whitening and rollout normalization; adds raw-return and diagnostic helpers.
# VERL 7aed6b230776f963fa09509c10d9c3a767d1102c:
# verl/trainer/ppo/core_algos.py (compute_gae_advantage_return; TRL ancestry),
# verl/utils/torch_functional.py (masked_var, masked_whiten).
# Agent Lightning cbc56e41e0eab643aa71d195e92e5bba652fa683:
# agentlightning/verl/per_rollout_loss.py (normalize_advantages_by_rollout).
# Licensed under the Apache License, Version 2.0 (the "License") for the
# identified VERL-derived portions; you may not use those portions except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 and LICENSES/Apache-2.0.txt.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Original lab contributions: Copyright (c) 2026 Agent Lightning Lab
# contributors, MIT (root LICENSE). See LICENSING.md and NOTICE for scope.
# These comments were added for release preparation; the historical executed
# source remains 5fb6adfd48ee245509f7dff3ab1d849fcf707d86.

"""Inspected stdlib mathematical ports. Not historical replay or runtime integration."""
from __future__ import annotations

import math
import statistics as stats
from collections import defaultdict


def summary(values):
    values = list(values)
    if not values:
        return {'n': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
    return {'n': len(values), 'mean': stats.mean(values),
            'std': stats.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values), 'max': max(values)}


def whiten(rows, masks):
    valid = [x for row, mask in zip(rows, masks, strict=True) for x, m in zip(row, mask, strict=True) if m]
    if len(valid) < 2:
        raise ValueError('Whitening needs at least two valid generated tokens')
    mean, var = stats.mean(valid), stats.variance(valid)
    return [[(x - mean) / math.sqrt(var + 1e-8) for x in row] for row in rows]


def gae(rewards, values, masks, gamma=1.0, lam=1.0):
    """Port of pinned VERL recurrence, including masked gap skipping and zero bootstrap."""
    raw, returns = [], []
    for reward, value, mask in zip(rewards, values, masks, strict=True):
        if not len(reward) == len(value) == len(mask):
            raise ValueError('Shape mismatch')
        next_value = last = 0.0
        row = [0.0] * len(reward)
        for t in reversed(range(len(reward))):
            delta = reward[t] + gamma * next_value - value[t]
            candidate = delta + gamma * lam * last
            next_value = value[t] * mask[t] + (1-mask[t]) * next_value
            last = candidate * mask[t] + (1-mask[t]) * last
            row[t] = last
        raw.append(row)
        returns.append([a+v for a, v in zip(row, value, strict=True)])
    return raw, returns, whiten(raw, masks)


def normalize_actor(advantages, masks, rollout_ids):
    counts = defaultdict(int)
    for mask, rid in zip(masks, rollout_ids, strict=True):
        counts[rid] += sum(mask)
    coefficients = [1/max(1, counts[rid]*len(masks)) for rid in rollout_ids]
    return [[a*c for a in row] for row, c in zip(advantages, coefficients, strict=True)], coefficients


def grpo_diagnostic(rows):
    """Read-only, once per real rollout. Caller declares pre/post selection scope."""
    rollouts = {}
    for row in rows:
        value = (row['group'], row['reward'])
        if row['rollout'] in rollouts and rollouts[row['rollout']] != value:
            raise ValueError('Inconsistent rollout group/reward')
        rollouts[row['rollout']] = value
    groups = defaultdict(list)
    for group, reward in rollouts.values():
        groups[group].append(reward)
    result = {}
    for rid, (group, reward) in rollouts.items():
        rewards = groups[group]
        # Pinned GRPO special-cases singleton groups with mean=0, std=1.
        mean = stats.mean(rewards) if len(rewards) > 1 else 0.0
        sd = stats.stdev(rewards) if len(rewards) > 1 else 1.0
        result[rid] = (reward-mean)/(sd+1e-6)
    return result


def prediction_metrics(values, returns, earlier_constant=None, variance_floor=1e-8):
    errors = [v-r for v, r in zip(values, returns, strict=True)]
    if len(errors) < 2:
        raise ValueError('Prediction metrics need two tokens')
    target_var = stats.variance(returns)
    raw_ev = 1-stats.variance(errors)/(target_var+1e-5)
    return {'prediction_time': 'fresh-rollout pre-update', 'mse': stats.mean(e*e for e in errors),
            'mae': stats.mean(abs(e) for e in errors), 'bias': stats.mean(errors),
            'value': summary(values), 'return': summary(returns), 'target_variance': target_var,
            'vf_explained_var_raw': raw_ev, 'ev_valid': target_var > variance_floor,
            'vf_explained_var_qualified': raw_ev if target_var > variance_floor else None,
            'out_of_unit_interval': sum(v < 0 or v > 1 for v in values),
            'earlier_training_constant': earlier_constant,
            'earlier_constant_mse': None if earlier_constant is None else stats.mean((earlier_constant-r)**2 for r in returns)}


def stage_summaries(advantages, masks, rows):
    valid = [[x for x, m in zip(a, mask, strict=True) if m] for a, mask in zip(advantages, masks, strict=True)]
    by_rollout, by_group, by_reward, by_turn = (defaultdict(list) for _ in range(4))
    for values, row in zip(valid, rows, strict=True):
        by_rollout[row['rollout']].extend(values)
        by_group[row['group']].extend(values)
        by_reward[str(row['reward'])].extend(values)
        for value, turn in zip(values, row.get('generated_turns', [None]*len(values)), strict=True):
            by_turn[str(turn)].append(value)
    return {'token_weighted': summary(x for row in valid for x in row),
            'rollout_weighted': summary(stats.mean(x) for x in by_rollout.values() if x),
            'prompt_group_weighted': summary(stats.mean(x) for x in by_group.values() if x),
            'by_prompt_group_token_weighted': {k: summary(v) for k, v in by_group.items()},
            'by_reward_token_weighted': {k: summary(v) for k, v in by_reward.items()},
            'by_turn_token_weighted': {k: summary(v) for k, v in by_turn.items()}}
