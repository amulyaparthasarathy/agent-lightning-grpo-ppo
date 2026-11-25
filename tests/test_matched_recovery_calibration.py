"""Recovery calibration through the real pinned collection adapter on CPU.

The production instrumentation (experiment/matched_runtime.instrument_adapter)
wraps the hash-verified upstream RolloutAdapter. Sampled rollouts here are
synthetic token sequences; trainer-level events (return, refresh, W&B finish) are
simulated because no trainer process runs. No model, vLLM, Ray or Modal is used.
"""
import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import zipfile

from experiment import matched_runtime as runtime, reader_contract as c, reader_modes as modes
from experiment import recovery_calibration as calibration, recovery_task as task
import recovery_fixtures as fx
from test_reader_training import source
try:
    from pinned_training_support import load, Proto, SOURCES
except ImportError:
    from tests.pinned_training_support import load, Proto, SOURCES

HAS_TORCH = importlib.util.find_spec('torch') is not None
# SIMULATED trainer-level events: no trainer process runs in this gate.
TRAINER_EVENTS = [{'event': 'weight_refresh', 'step': 0}, {'event': 'wandb_finished', 'id': 'simulated'},
                  {'event': 'trainer_returned', 'global_step': 1, 'validation_only': False, 'collection_only': True, 'optimizer_updates': 0}]


@unittest.skipUnless(HAS_TORCH and (SOURCES/'verl/verl').is_dir(), 'requires explicit pinned CPU test environment')
class CalibrationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict('os.environ', AGL_PPO_AUDIT_DIR=self.tmp.name)
        env.start(); self.addCleanup(env.stop)
        runtime.audit._STATE.clear()
        runtime.audit._STATE['step'] = 1
        self.plan = fx.calibration_plan(source())

    def adapter(self, request, prompt=256, response=256, **flags):
        import torch
        import numpy as np
        from tensordict import TensorDict
        module = load('agentlightning', 'verl/rollout_adapter.py', dict(torch=torch, np=np, TensorDict=TensorDict,
                      DataProto=Proto, json=json, io=io, zipfile=zipfile, base64=base64))
        if request is None:
            runtime.instrument_adapter(module, **flags)
        else:
            # The same mode selection register_in_worker performs for this request.
            runtime.instrument_adapter(module, collection_only=modes.collection_only(request['plan'], request['stage']),
                                       calibration=modes.phase(request['plan']) == modes.CALIBRATION)
        return module.RolloutAdapter(max_prompt_length=prompt, max_response_length=response, device=torch.device('cpu'),
                                     pad_token_id=0, trace_aggregator_level='trajectory')

    def request(self, name='sampled-packet-1', algorithm=0):
        job = self.plan['jobs'][algorithm]
        return c.stage_request(self.plan, job, next(s for s in job['stages'] if s['name'] == name))

    def collection(self, request, behavior, fragmented=False):
        """Reviewed-episode rows and matching sampled rollouts for one packet."""
        declared = self.plan['calibration']['stages'][request['stage']['name']]
        episodes = fx.rows(declared, behavior)
        rollouts = []
        for index, e in enumerate(episodes):
            # One model call per generation the episode reports, each with exactly the number of
            # sampled tokens the trace's server-reported usage gives, and observation tokens between.
            prompt, turns, context = [1, 2, 3], [], [1, 2, 3]
            for call, tokens in enumerate(e['completion_tokens']):
                response = [1000 * (call + 1) + t for t in range(tokens)]
                if fragmented:
                    prompt = [4 + call, 5 + call, 6 + call, 7 + call][:2 + call % 3]
                else:
                    prompt = list(context)
                turns.append((prompt, response))
                context = prompt + response + [90 + call, 91 + call]  # observation, or the reminder
            rollouts.append(NS(rollout_id=e['rollout_id'], data_id='group-'+e['case_id'], step=1, sample_idx_in_step=index % 4,
                               final_reward=e['reward'], input={'case_id': e['case_id']},
                               triplets=[NS(prompt={'token_ids': p}, response={'token_ids': r, 'log_probs': [-.5]*len(r)}, image_urls=None)
                                         for p, r in turns]))
        return episodes, rollouts

    def events(self):
        return [json.loads(line) for p in sorted(Path(self.tmp.name).glob('*.jsonl')) for line in p.read_text().splitlines()]

    def collect(self, request, rollouts, **limits):
        for path in Path(self.tmp.name).glob('*.jsonl'):
            path.unlink()
        with self.assertRaises(runtime.CollectionOnlyComplete):
            self.adapter(request, **limits).get_train_data_batch(rollouts, global_steps=1)
        return self.events()

    def test_both_routes_admit_through_production_instrumentation_and_stop_before_any_update(self):
        mixed = lambda case_id, sample: ('correct', 'semantic', 'format', 'recovered')[sample]
        for algorithm in (0, 1):
            for name in ('sampled-packet-1', 'sampled-packet-2', 'sampled-packet-3'):
                for fragmented in (False, True):
                    with self.subTest(algorithm=algorithm, packet=name, fragmented=fragmented):
                        request = self.request(name, algorithm)
                        episodes, rollouts = self.collection(request, mixed, fragmented)
                        events = self.collect(request, rollouts)
                        kinds = [e['event'] for e in events]
                        for forbidden in calibration.FORBIDDEN_EVENTS:
                            self.assertNotIn(forbidden, kinds)
                        self.assertEqual([kinds.count(k) for k in ('collection_manifest', 'development_collection_admitted', 'calibration_collection_tokens')], [1, 1, 1])
                        admitted = next(e for e in events if e['event'] == 'development_collection_admitted')
                        tokens = next(e for e in events if e['event'] == 'calibration_collection_tokens')
                        sampled = {e['rollout_id']: sum(e['completion_tokens']) for e in episodes}
                        generations = {e['rollout_id']: len(e['completion_tokens']) for e in episodes}
                        self.assertEqual((admitted['real_rollouts'], admitted['groups'], admitted['generated_tokens']), (32, 8, sum(sampled.values())))
                        self.assertEqual(admitted['fragment_rows'], sum(generations.values()) if fragmented else 32)
                        self.assertEqual((tokens['truncated_rows'], tokens['dropped_prompt_rows']), (0, 0))
                        for measured in tokens['rollouts']:
                            rollout = measured['rollout']
                            # Sampled tokens are conserved; observations and the reminder are context, never targets.
                            self.assertEqual(measured['mask_tokens'], sampled[rollout])
                            self.assertEqual(measured['context_tokens'], 0 if fragmented else 2 * (generations[rollout] - 1))
                            self.assertEqual(len(measured['prompt_tokens']), generations[rollout] if fragmented else 1)
                        done = calibration.assemble(request, events + TRAINER_EVENTS, episodes)
                        report = calibration.validate_stage(self.plan, self.plan['jobs'][algorithm], request['stage'], done)
                        self.assertEqual(done['optimizer_updates'], {'actor': 0, 'critic': 0})
                        self.assertEqual((len(report['episodes']), report['collection']['generated_tokens']), (32, sum(sampled.values())))
                        self.assertTrue(all(e['sampled_tokens'] == e['mask_tokens'] == sum(e['completion_tokens']) for e in report['episodes']))
                        self.assertEqual(report['collection']['limits'], {'max_prompt_length': 256, 'max_response_length': 256})
                        # Measured by the adapter, not assumed: coverage is every rollout of the packet.
                        self.assertEqual((report['collection']['rollouts_with_token_evidence'], report['collection']['truncated_rows'],
                                          report['collection']['dropped_prompt_rows']), (32, 0, 0))
                        coverage = calibration.token_coverage(self.plan['calibration']['stages'][name], events, None)
                        self.assertEqual((coverage['rollouts_with_token_evidence'], coverage['sampled_tokens'], coverage['mask_tokens'],
                                          coverage['truncated_rows']), (32, sum(sampled.values()), sum(sampled.values()), 0))
                        # A generation lost or duplicated upstream of the adapter breaks the join with the traces.
                        lost = copy.deepcopy(episodes); lost[0]['completion_tokens'][0] += 1
                        with self.assertRaises(ValueError): calibration.assemble(request, events + TRAINER_EVENTS, lost)

    def test_constant_groups_are_retained_and_reported(self):
        request = self.request()
        for behavior, constant in ((lambda *_: 'correct', 8), (lambda *_: 'semantic', 8),
                                   (lambda case_id, sample: 'correct' if sample else 'evidence', 0)):
            episodes, rollouts = self.collection(request, behavior)
            events = self.collect(request, rollouts)
            done = calibration.assemble(request, events + TRAINER_EVENTS, episodes)
            self.assertEqual(done['calibration']['collection']['adapter_constant_groups'], constant)
            self.assertEqual(len(done['calibration']['episodes']), 32)

    def test_adapter_rejects_incomplete_duplicate_truncated_and_miscounted_collections(self):
        request = self.request()
        episodes, good = self.collection(request, lambda case_id, sample: ('correct', 'semantic')[sample % 2])
        edits = {'missing reward': lambda r: setattr(r[0], 'final_reward', None),
                 'nonbinary reward': lambda r: setattr(r[0], 'final_reward', 0.5),
                 'no trace': lambda r: setattr(r[0], 'triplets', []),
                 'duplicate rollout id': lambda r: setattr(r[1], 'rollout_id', r[0].rollout_id),
                 'incomplete group': lambda r: r.pop(),
                 'whole group missing': lambda r: r.__delitem__(slice(0, 4)),
                 'altered counts': lambda r: setattr(r[4], 'data_id', r[0].data_id)}
        for label, edit in edits.items():
            bad = copy.deepcopy(good); edit(bad)
            with self.subTest(edit=label), self.assertRaises(ValueError):
                self.adapter(request).get_train_data_batch(bad, global_steps=1)
        # Limits that cannot hold the trajectory reject the collection; nothing is cut silently.
        stage = self.plan['calibration']['stages']['sampled-packet-1']
        # Events of several rejected collections together are ambiguous: no figure is taken from them.
        coverage = calibration.token_coverage(stage, self.events(), None)
        self.assertEqual((coverage['rollouts_with_token_evidence'], coverage['truncated_rows'], coverage['dropped_prompt_rows']), (None, None, None))
        self.assertIn('ambiguous', coverage['adapter_evidence'])
        for path in Path(self.tmp.name).glob('*.jsonl'):
            path.unlink()
        with self.assertRaisesRegex(ValueError, 'truncated'): self.adapter(request, response=8).get_train_data_batch(copy.deepcopy(good), global_steps=1)
        # One rejected collection: the adapter measured nothing, so nothing is invented and nothing reads as zero.
        coverage = calibration.token_coverage(stage, self.events(), None)
        self.assertEqual((coverage['rollouts_with_token_evidence'], coverage['truncated_rows'], coverage['dropped_prompt_rows'], coverage['mask_tokens']),
                         (0, None, None, None))
        self.assertIn('unknown', coverage['adapter_evidence'])
        # No audit events at all is not a statement about the run either.
        self.assertIsNone(calibration.token_coverage(stage, None, None)['rollouts_with_token_evidence'])
        with self.assertRaisesRegex(ValueError, 'Invalid prompt context'): self.adapter(request, prompt=2).get_train_data_batch(copy.deepcopy(good), global_steps=1)
        for path in Path(self.tmp.name).glob('*.jsonl'):
            self.assertNotIn('development_collection_admitted', path.read_text())

    def test_wrong_packet_cross_source_and_reward_disagreement_fail_the_join(self):
        request, other = self.request('sampled-packet-1'), self.request('sampled-packet-2')
        episodes, rollouts = self.collection(request, lambda case_id, sample: ('correct', 'semantic')[sample % 2])
        events = self.collect(request, rollouts)
        calibration.assemble(request, events + TRAINER_EVENTS, episodes)
        # A well-formed collection of another packet is admitted by the adapter but not by this stage.
        other_episodes, other_rollouts = self.collection(other, lambda *_: 'correct')
        other_events = self.collect(other, other_rollouts)
        with self.assertRaises(ValueError): calibration.assemble(request, other_events + TRAINER_EVENTS, other_episodes)
        with self.assertRaises(ValueError): calibration.assemble(request, other_events + TRAINER_EVENTS, episodes)
        with self.assertRaises(ValueError): calibration.assemble(request, events + TRAINER_EVENTS, other_episodes)
        flipped = copy.deepcopy(episodes); flipped[0]['reward'] = 1.0 - flipped[0]['reward']
        with self.assertRaises(ValueError): calibration.assemble(request, events + TRAINER_EVENTS, flipped)
        renamed = copy.deepcopy(episodes); renamed[0]['rollout_id'] = 'another-attempt'
        with self.assertRaises(ValueError): calibration.assemble(request, events + TRAINER_EVENTS, renamed)
        for extra in ({'event': 'matched_advantages'}, {'event': 'optimizer_step'}, {'event': 'gradient_step'},
                      {'event': 'native_checkpoint_saved', 'step': 1}, {'event': 'weight_refresh', 'step': 1}):
            with self.assertRaises(ValueError): calibration.assemble(request, events + TRAINER_EVENTS + [extra], episodes)

    def test_mode_selection_preserves_asset_development_and_full_training_paths(self):
        from experiment import reader_assets
        asset = reader_assets.read(Path(__file__).resolve().parents[1]/'examples/reader-profile.asset.synthetic.json')
        development = c.make_plan(dict(asset, purpose='collection-only'), source())
        qualification = c.make_plan(asset, source())
        request = self.request()
        _, rollouts = self.collection(request, lambda *_: 'correct')
        # Asset collection-only: the qualified a06 evidence shape, with no calibration event.
        job = development['jobs'][0]
        asset_request = c.stage_request(development, job, job['stages'][0])
        for path in Path(self.tmp.name).glob('*.jsonl'):
            path.unlink()
        with self.assertRaises(runtime.CollectionOnlyComplete):
            self.adapter(asset_request).get_train_data_batch(copy.deepcopy(rollouts), global_steps=1)
        events = self.events()
        self.assertNotIn('calibration_collection_tokens', [e['event'] for e in events])
        admitted = next(e for e in events if e['event'] == 'development_collection_admitted')
        payload = {'iteration', 'real_rollouts', 'groups', 'generated_tokens', 'fragment_rows', 'constant_groups'}
        calibrated = next(e for e in self.collect(request, copy.deepcopy(rollouts)) if e['event'] == 'development_collection_admitted')
        # Exactly the fields the archived a06 evidence has, in both modes.
        self.assertTrue(payload <= set(admitted))
        self.assertEqual(set(admitted), set(calibrated))
        self.assertEqual({k: admitted[k] for k in payload}, {k: calibrated[k] for k in payload})
        # Full training and evaluation stages return the batch to the unchanged numerical path.
        job = qualification['jobs'][0]
        for stage in job['stages']:
            train_request = c.stage_request(qualification, job, stage)
            self.assertFalse(modes.collection_only(qualification, stage))
            batch, _ = self.adapter(train_request).get_train_data_batch(copy.deepcopy(rollouts), global_steps=1)
            self.assertEqual(batch.batch['response_mask'].sum().item(), sum(sum(len(t.response['token_ids']) for t in r.triplets) for r in rollouts))
        # The greedy calibration stage never enters the training collection branch.
        self.assertFalse(modes.collection_only(self.plan, 'greedy-development'))


if __name__ == '__main__':
    unittest.main()
