"""Pre-optimization checks for the frozen recovery route; valid tensors pass through untouched."""
from collections import Counter, defaultdict
import math

try:
    from . import recovery_calibration as calibration, recovery_learning as learning, recovery_task as task, recovery_prompt
except ImportError:
    import recovery_calibration as calibration
    import recovery_learning as learning
    import recovery_task as task
    import recovery_prompt


def reviewed_episodes(run, request, rollouts, tokenizer):
    original = {r.rollout_id: r for r in rollouts}
    episodes = []
    for _, records in calibration.read_traces(run):
        if records is None:
            raise ValueError('Unreadable recovery trace')
        episode = calibration.episode_row(records, request)
        episodes.append(episode)
        rollout = original.get(episode['rollout_id'])
        if rollout is None:
            continue  # prior collection; cumulative membership is checked by collection_inputs
        inputs = [r['inputs'] for r in records if r['kind'] == 'llm']
        for view, turn in zip(inputs, rollout.triplets or [], strict=True):
            rendered = recovery_prompt.render(*recovery_prompt.from_trace(view))
            if tokenizer.encode(rendered, add_special_tokens=False) != turn.prompt['token_ids']:
                raise ValueError('Original sampled prompt differs from the recorded conversation')
    return episodes


def collection_inputs(request, iteration, rollouts, episodes):
    spec = learning.stage_spec(request)
    if not 1 <= iteration <= len(spec['batches']):
        raise ValueError('Undeclared collection')
    batches = spec['batches'][:iteration]
    cases = batches[-1]
    # Stage traces are cumulative. Reject extras, incomplete attempts and reused IDs.
    reviewed = {e['rollout_id']: e for e in episodes}
    if (len(reviewed) != len(episodes) or len({e['trace_id'] for e in episodes}) != len(episodes)
            or Counter(e['case_id'] for e in episodes) != Counter({cid: 4 for batch in batches for cid in batch})):
        raise ValueError('Cumulative traces differ from declared completed collections')
    ids = [r.rollout_id for r in rollouts]
    if len(ids) != 32 or set(ids) != {rid for rid, e in reviewed.items() if e['case_id'] in cases}:
        raise ValueError('Rollout IDs do not match the confirmed episode set')
    trusted = {r['case_id']: r for r in task.rows('train')}
    groups = defaultdict(set)
    for r in rollouts:
        i = r.sample_idx_in_step
        if type(i) != int or not 0 <= i < len(cases) or r.step != iteration:
            raise ValueError('Unexpected question slot or policy iteration')
        cid = cases[i]
        if any(r.input.get(k) != v for k, v in trusted[cid].items()):
            raise ValueError('Incorrect training split, row or batch order')
        e = reviewed[r.rollout_id]
        if (e['case_id'] != cid or e['reward_posted'] is not True or type(r.final_reward) not in (float, int)
                or r.final_reward not in (0, 1) or e['reward'] != r.final_reward):
            raise ValueError('Unconfirmed or inconsistent trajectory reward')
        if r.rollout_state != 'succeeded' or r.error_message:
            raise ValueError('Controller did not confirm a successful execution')
        events = r.triplet_events
        if any(event['rollout_id'] != r.rollout_id or event['attempt_id'] != e['attempt_id'] for event in events):
            raise ValueError('Controller evidence crosses rollout attempts')
        rewards = [event['data']['value'] for event in events if event['event_type'] == 'reward']
        generations = [event['data'] for event in events if event['event_type'] == 'model_request']
        if rewards != [r.final_reward] or len(generations) != e['llm_calls']:
            raise ValueError('Missing, duplicate or unconfirmed controller reward/generation')
        turns = r.triplets or []
        if len(turns) != e['llm_calls'] or len(turns) != len(e['completion_tokens']) or not 1 <= len(turns) <= 6:
            raise ValueError('Generation count differs from submitted trajectory')
        for t, n, p, raw in zip(turns, e['completion_tokens'], e['prompt_tokens'], generations, strict=True):
            prompt, action = t.prompt['token_ids'], t.response['token_ids']
            probs = t.response.get('log_probs')
            if (raw.get('status') == 'error' or raw.get('http_status', 200) >= 400 or raw.get('error')
                    or raw.get('prompt_token_ids') != prompt or raw.get('response_token_ids') != action
                    or raw.get('response_log_probs') != probs):
                raise ValueError('Controller sampled token evidence differs from adapter input')
            if (p != len(prompt) or n != len(action) or not prompt or not action or n > 1024 or p+n > 4096
                    or not isinstance(probs, list) or len(probs) != n
                    or any(type(x) not in (float, int) or not math.isfinite(x) or x > 0 for x in probs)):
                raise ValueError('Incomplete, over-bound or unaligned served tokens/probabilities')
        groups[r.data_id].add(cid)
    if len(groups) != 8 or any(len(g) != 1 for g in groups.values()):
        raise ValueError('Expected eight distinct complete question groups')


def representation(out, rollouts, metrics):
    import torch
    b = out.batch
    width = b['prompts'].shape[1]
    mask = b['response_mask']
    attention = b['attention_mask']
    probs = b.get('rollout_log_probs')
    if (width != 3072 or mask.shape[1] != 2048 or attention.shape[1] != 5120 or probs is None
            or probs.shape != mask.shape or not torch.isfinite(probs).all() or b['is_drop_mask'].any()
            or metrics['training/n_truncated_sample'] or metrics['training/n_skipped_empty_rows']
            or not torch.equal(b['input_ids'], torch.cat((b['prompts'], b['responses']), dim=1))):
        raise ValueError('Dropped, truncated or invalid padded training representation')
    by_id = {r.rollout_id: r for r in rollouts}
    next_turn = Counter()
    fragment_counts = Counter()
    lengths = []
    for i, rid in enumerate(out.non_tensor_batch['rollout_id_list']):
        r = by_id[rid]
        att = attention[i].bool()
        if (not torch.equal(b['position_ids'][i, att], torch.arange(int(att.sum()), device=att.device))
                or int(att.sum()) > 4096 or b['token_level_scores'][i].sum().item() != r.final_reward
                or out.non_tensor_batch['data_id_list'][i] != r.data_id):
            raise ValueError('Active positions, group or row reward differ')
        prompt = b['prompts'][i, att[:width]].tolist()
        response_length = int(att[width:].sum())
        if (not att[:width].tolist() == [False]*(width-len(prompt)) + [True]*len(prompt)
                or not att[width:].tolist() == [True]*response_length + [False]*(mask.shape[1]-response_length)):
            raise ValueError('Invalid attention boundaries')
        seq = prompt + b['responses'][i, :response_length].tolist()
        expected_mask = torch.zeros_like(mask[i])
        j = next_turn[rid]
        if j >= len(r.triplets) or r.triplets[j].prompt['token_ids'] != prompt:
            raise ValueError('Fragment prompt differs from original generation')
        while j < len(r.triplets):
            t = r.triplets[j]
            p, a = t.prompt['token_ids'], t.response['token_ids']
            stop = len(p)+len(a)
            if seq[:stop] != p+a:
                raise ValueError('Original context or sampled action differs')
            sl = slice(len(p)-len(prompt), stop-len(prompt))
            expected_mask[sl] = 1
            expected_probs = probs.new_tensor(t.response['log_probs'])
            if not torch.equal(probs[i, sl], expected_probs):
                raise ValueError('Target log probabilities differ from original samples')
            j += 1
            if stop == len(seq):
                break
            if j == len(r.triplets) or r.triplets[j].prompt['token_ids'][:stop] != seq[:stop]:
                raise ValueError('Unaccounted fragment context')
        if not torch.equal(mask[i], expected_mask):
            raise ValueError('Policy targets include context/padding or omit sampled actions')
        next_turn[rid] = j
        fragment_counts[rid] += 1
        lengths.append(len(seq))
    if any(next_turn[rid] != len(r.triplets) for rid, r in by_id.items()):
        raise ValueError('Missing sampled turn')
    generated = sum(len(t.response['token_ids']) for r in rollouts for t in r.triplets)
    if generated != int(mask.sum()):
        raise ValueError('Sampled-token conservation failed')
    return {'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'bytes': v.numel()*v.element_size()}
                        for k, v in b.items()}, 'active_lengths': lengths, 'max_active_position': max(lengths)-1,
            'fragment_rows': len(out), 'fragments_per_rollout': dict(fragment_counts),
            'generated_tokens': generated, 'target_tokens': int(mask.sum()),
            'masked_response_context_tokens': int(attention[:, width:].sum()-mask.sum())}


def old_probabilities(batch, critic):
    import torch
    b = batch.batch
    mask = b['response_mask'].bool()
    for key in ('rollout_log_probs', 'old_log_probs', *(('values',) if critic else ())):
        if key not in b or b[key].shape != mask.shape or not torch.isfinite(b[key]).all():
            raise ValueError('Missing, misaligned or nonfinite '+key)
    delta = (b['old_log_probs'][mask] - b['rollout_log_probs'][mask]).double()
    if not delta.numel() or not torch.isfinite(delta).all():
        raise ValueError('Invalid probability comparison')
    return {'target_tokens': delta.numel(), 'mean_delta': delta.mean().item(), 'max_abs_delta': delta.abs().max().item()}
