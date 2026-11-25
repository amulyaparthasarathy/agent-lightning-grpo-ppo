"""Production recovery harness with real subprocesses and real stdio MCP; inference is scripted.

Every episode here runs recovery_agent.Agent, AutoGen 0.7.5 request construction
and response parsing, the observed client and workbench, and the real server.
Only the OpenAI SDK call and the reward receiver are test boundaries.
"""
import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from experiment import recovery_agent, recovery_calibration as calibration, recovery_probe, recovery_prompt
from experiment import recovery_task as task, reader_integration
from recovery_harness_support import (ANSWER, ASSET, CALL, FAIL, FINAL, INVENTORY, POLICY, RAW, SOLVE, TEXT, completion,
                                      calibration_request, connection_error, demonstration, episode, run_episode)

ROOT = Path(__file__).resolve().parents[1]
ROW = task.rows('development')[0]
TRUTH = task.truth(*task.target(ROW['question']))


def scripted(result, request):
    return reader_integration.chains(result['records'], request, evidence_class='scripted-inference')


def bad_extra(st):
    return CALL('get_asset', asset_id=st.asset_id, as_of=st.as_of)(st)


class HarnessTests(unittest.TestCase):
    def run_case(self, script=None, **options):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        result = run_episode(tmp.name, script, **options)
        self.assertTrue(result['cleanup']['complete'])
        for child in result['children']:
            self.assertEqual(child['environment_keys'], ['PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED'])
            self.assertFalse(Path(f"/proc/{child['pid']}").exists())
        return result

    def complete(self, result, reward, outcome='complete_final'):
        self.assertIsNone(result['error'], result['message'])
        self.assertEqual(result['posts'], [{'event_type': 'reward', 'data': {'value': reward}}])
        out = episode(result)['outputs']
        self.assertEqual((out['outcome'], out['reward'], out['reward_posted']), (outcome, reward, True))
        self.assertEqual(reader_integration.episode_outcome(result['records']), outcome)
        return out

    def incomplete(self, result, reason, posts=0, requests=None):
        # The episode must fail for the stated reason, not because a script ran out.
        self.assertIn(reason, f"{result['error']}: {result['message']}")
        self.assertEqual(len(result['posts']), posts)
        if requests is not None:
            self.assertEqual(len(result['requests']), requests)
        record = episode(result)
        self.assertEqual(record['execution_status'], 'exception')
        self.assertFalse((record['outputs'] or {}).get('reward_posted'))
        self.assertEqual(reader_integration.episode_outcome(result['records']), 'incomplete')
        with self.assertRaises((ValueError, KeyError, TypeError)):
            calibration.episode_row(result['records'], calibration_request(), 'scripted-inference')

    def test_three_sequential_calls_final_answer_inputs_joins_and_cleanup(self):
        request = calibration_request()
        result = self.run_case(request=request)
        out = self.complete(result, 1.0)
        self.assertEqual((out['tool_rounds'], out['successful_tool_calls'], out['policy_errors'], out['llm_calls'], out['tool_calls']), (3, 3, 0, 4, 3))
        requests = result['requests']
        self.assertEqual([len(r['messages']) for r in requests], [2, 4, 6, 8])
        for earlier, later in zip(requests, requests[1:]):
            self.assertEqual(later['messages'][:len(earlier['messages'])], earlier['messages'])
        for sent in requests:
            self.assertEqual({k: v for k, v in sent.items() if k not in ('messages', 'tools')},
                             {'stream': False, 'model': 'auto', 'parallel_tool_calls': False, 'max_tokens': 1024, 'tool_choice': 'auto'})
            self.assertEqual([(t['function']['name'], t['function']['parameters']) for t in sent['tools']],
                             [(n, task.TOOL_SCHEMAS[n]) for n in task.TOOLS])
            self.assertEqual([m['role'] for m in sent['messages'][:2]], ['system', 'user'])
            self.assertEqual(sent['messages'][0]['content'], recovery_agent.SYSTEM_MESSAGE)
            self.assertEqual(sent['messages'][1]['content'], ROW['question']+'\n'+task.PROMPT)
            # No label, diagnostic, family, credential or split information reaches the model.
            text = json.dumps(sent)
            for hidden in (ROW['result'], 'test-secret-canary', 'local-test-only', ROW['case_id'], 'qualifying_run_ids',
                           *task.FAMILIES, 'failure_class', 'development', TRUTH['policy_record_id'] if sent is requests[0] else 'never-present'):
                self.assertNotIn(hidden, text)
        self.assertEqual([m['content'] for m in requests[-1]['messages'] if m['role'] == 'tool'],
                         [task.dumps(task.call(n, a)) for n, a in (
                             ('get_asset', {'asset_id': TRUTH['asset_id']}), ('get_policy', {'policy_id': TRUTH['policy_id']}),
                             ('list_backup_runs', {'asset_id': TRUTH['asset_id'], 'as_of': TRUTH['as_of']}))])
        self.assertEqual(len(result['children']), 1)
        kinds = [r['kind'] for r in result['records']]
        self.assertEqual({k: kinds.count(k) for k in set(kinds)},
                         {'llm': 4, 'served_response': 4, 'tool': 3, 'action_outcome': 3, 'episode': 1, 'trace_status': 1})
        self.assertTrue(all(r['evidence_class'] == 'scripted-inference' and r['synthetic_fixture'] for r in result['records']))
        self.assertEqual(episode(result)['episode_policy'], 'recovery-episode-v1')
        chains = scripted(result, request)
        self.assertEqual([c['tool_call_id'] for c in chains], ['scripted-call-1-0', 'scripted-call-2-0', 'scripted-call-3-0'])
        # Scripted traces can never be read as live evidence, and identities must match.
        with self.assertRaises(ValueError): reader_integration.chains(result['records'], request)
        report = reader_integration.report(Path(result['root']), request)
        self.assertEqual((report['trace_files'], report['qualified'], report['coverage'][0]['invalid_trace']), (1, False, 'ValueError'))
        for field in ('task_id', 'task_sha256', 'environment_sha256', 'configuration_sha256', 'source_sha256', 'wandb_run_id'):
            bad = copy.deepcopy(result['records']); bad[0][field] = 'different'
            with self.assertRaises(ValueError): scripted({'records': bad}, request)
        with self.assertRaises(ValueError): scripted(result, calibration_request(stage=1))
        with self.assertRaises(ValueError): scripted(result, calibration_request(algorithm=1))

    def test_five_rounds_then_toolless_reflection_with_reminder(self):
        for final, reward in ((FINAL, 1.0), (TEXT('The asset is ready.'), 0.0)):
            with self.subTest(reward=reward):
                result = self.run_case([SOLVE, SOLVE, SOLVE, INVENTORY, INVENTORY, final])
                out = self.complete(result, reward)
                self.assertEqual((out['tool_rounds'], out['successful_tool_calls'], out['llm_calls']), (5, 5, 6))
                requests = result['requests']
                self.assertEqual([len(r['messages']) for r in requests], [2, 4, 6, 8, 10, 13])
                self.assertTrue(all('tools' in r and r['tool_choice'] == 'auto' for r in requests[:5]))
                # The reflection request advertises no tools and carries the whole context plus one reminder.
                self.assertEqual(set(requests[5]), {'messages', 'stream', 'model', 'parallel_tool_calls', 'max_tokens'})
                self.assertEqual(requests[5]['messages'][:10], requests[4]['messages'])
                self.assertEqual(requests[5]['messages'][11]['role'], 'tool')
                self.assertEqual(requests[5]['messages'][12], {'role': 'user', 'name': 'user', 'content': recovery_agent.FINAL_INSTRUCTION})
                self.assertEqual(out['diagnostics']['distinct_successful_observations'], {'get_asset': 1, 'get_policy': 1, 'list_backup_runs': 1})
                for value in (TRUTH['decision'], *TRUTH['qualifying_run_ids']):
                    self.assertNotIn(value, recovery_agent.FINAL_INSTRUCTION + recovery_agent.SYSTEM_MESSAGE)
        # A sixth call cannot be executed: tools have ended and a call there is not an answer.
        result = self.run_case([SOLVE, SOLVE, SOLVE, INVENTORY, INVENTORY, INVENTORY, FINAL])
        self.incomplete(result, 'Reflect on tool use produced no valid text response', requests=6)

    def test_invalid_initial_or_middle_call_then_recovery_keeps_action_error_and_reward(self):
        asset = TRUTH['asset_id']
        cases = {'invalid_arguments': (bad_extra, True),
                 'invalid_asset_id': (CALL('get_asset', asset_id=asset.upper()), True),
                 'unknown_asset': (CALL('get_asset', asset_id='ast-00000000'), True),
                 'invalid_policy_id': (CALL('get_policy', policy_id=asset), True),
                 'unknown_policy': (CALL('get_policy', policy_id='pol-00000000'), True),
                 'invalid_as_of': (CALL('list_backup_runs', asset_id=asset, as_of='2026-02-30T00:00Z'), True),
                 'invalid_argument_json': (RAW('get_asset', '{'), False),
                 'unknown_tool': (CALL('lookup_asset', asset_id=asset), False),
                 'non_object_arguments': (RAW('get_asset', json.dumps(asset)), True)}
        self.assertEqual(sorted(cases), sorted(task.EPISODE_POLICY['policy_error_codes']))
        for position in (0, 2):
            for code, (action, reaches_workbench) in cases.items():
                with self.subTest(code=code, position=position):
                    script = [SOLVE, SOLVE, SOLVE, SOLVE]
                    script.insert(position, action)
                    request = calibration_request()
                    result = self.run_case(script, request=request)
                    out = self.complete(result, 1.0, 'complete_recovered')
                    self.assertEqual((out['policy_errors'], out['recovered'], out['tool_rounds'], out['successful_tool_calls']), (1, True, 4, 3))
                    actions = [r for r in result['records'] if r['kind'] == 'action_outcome']
                    self.assertEqual([a['outputs']['policy_error'] for a in actions], [code if i == position else None for i in range(4)])
                    failed = actions[position]
                    # The sampled action is retained unchanged; the typed error is an observation.
                    sent = result['requests'][position]
                    self.assertTrue(failed['inputs']['observation']['is_error'])
                    observed = result['requests'][position+1]['messages'][-1]
                    self.assertEqual(observed['content'], failed['inputs']['observation']['content'])
                    if code not in ('invalid_argument_json', 'unknown_tool'):
                        self.assertEqual(json.loads(observed['content']), task.policy_error(code))
                    tools = [r for r in result['records'] if r['kind'] == 'tool']
                    self.assertEqual(len(tools), 4 if reaches_workbench else 3)
                    self.assertEqual(out['tool_errors'], 1 if reaches_workbench else 0)
                    self.assertEqual(len(scripted(result, request)), 3)
                    row = calibration.episode_row(result['records'], request, 'scripted-inference')
                    self.assertEqual((row['policy_error_codes'], row['recovered'], row['reward']), ([code], True, 1.0))
                    del sent
        # A value that cannot cross the stdio transport gets the server's own typed validation error
        # immediately, instead of hanging the MCP client until its read timeout.
        for raw, code in (('{"asset_id": "ast-\\ud800"}', 'invalid_asset_id'), ('{"asset_id": "ok", "x\\udfff": "y"}', 'invalid_arguments')):
            with self.subTest(raw=raw):
                result = self.run_case([RAW('get_asset', raw), SOLVE, SOLVE, SOLVE, SOLVE])
                self.assertEqual(self.complete(result, 1.0, 'complete_recovered')['policy_errors'], 1)
                first = next(r for r in result['records'] if r['kind'] == 'action_outcome')
                self.assertEqual((first['inputs']['call']['arguments'], first['outputs']['policy_error']), (raw, code))
        # Non-object JSON arguments of every shape are typed without contacting the server or repair.
        for raw in ('[]', json.dumps([asset]), 'null', '7', 'true', '""'):
            with self.subTest(raw=raw):
                result = self.run_case([RAW('get_asset', raw), SOLVE, SOLVE, SOLVE, SOLVE])
                self.complete(result, 1.0, 'complete_recovered')
                first = next(r for r in result['records'] if r['kind'] == 'action_outcome')
                self.assertEqual((first['inputs']['call']['arguments'], first['outputs']['policy_error']), (raw, 'non_object_arguments'))

    def test_second_invalid_action_is_a_declared_terminal_zero(self):
        scripts = {'two initial': [bad_extra, RAW('get_policy', '{')],
                   'error, recovery, error': [bad_extra, SOLVE, CALL('missing_tool')],
                   'late': [SOLVE, SOLVE, SOLVE, CALL('get_policy', policy_id='pol-00000000'), RAW('list_backup_runs', '[]')]}
        for name, script in scripts.items():
            with self.subTest(name=name):
                request = calibration_request()
                result = self.run_case(script, request=request)
                out = self.complete(result, 0.0, 'terminal_policy_failure')
                self.assertEqual((out['policy_errors'], out['raw_answer'], out['termination']), (2, None, 'policy_error_limit'))
                self.assertEqual(len(result['requests']), len(script))  # nothing is generated after the limit
                self.assertEqual(out['diagnostics']['failure_class'], 'action')
                self.assertIsNone(out['diagnostics']['format_compliant']); self.assertIsNone(out['diagnostics']['decision_correct'])
                terminations = [r for r in result['records'] if r['kind'] == 'termination']
                self.assertEqual([t['outputs'] for t in terminations], [
                    {'episode_policy': 'recovery-episode-v1', 'policy_errors': 2, 'declared_terminal_task_failure': True}])
                self.assertEqual(scripted(result, request), [])
                row = calibration.episode_row(result['records'], request, 'scripted-inference')
                self.assertEqual((row['outcome'], row['reward'], row['failure_class'], row['joined_observation_chains']),
                                 ('terminal_policy_failure', 0.0, 'action', 0))

    def test_early_or_ungrounded_answers_are_complete_policy_zeros(self):
        truth = {k: TRUTH[k] for k in ('asset_id', 'as_of', 'decision', 'asset_record_id', 'policy_record_id', 'inventory_id')}
        lucky = task.dumps({**truth, 'supporting_run_ids': TRUTH['qualifying_run_ids'][:1]})
        scripts = {'no tool call': ([TEXT(lucky)], {'asset': False, 'policy': False, 'inventory': False}),
                   'asset only': ([ASSET, TEXT(lucky)], {'asset': True, 'policy': False, 'inventory': False}),
                   'policy skipped': ([ASSET, INVENTORY, TEXT(lucky)], {'asset': True, 'policy': False, 'inventory': True}),
                   'inventory skipped': ([ASSET, POLICY, TEXT(lucky)], {'asset': True, 'policy': True, 'inventory': False}),
                   'inventory at another time': ([ASSET, POLICY, CALL('list_backup_runs', asset_id=TRUTH['asset_id'], as_of='2026-11-11T00:00Z'),
                                                  TEXT(lucky)], {'asset': True, 'policy': True, 'inventory': False})}
        for name, (script, seen) in scripts.items():
            with self.subTest(name=name):
                result = self.run_case(script)
                out = self.complete(result, 0.0)
                d = out['diagnostics']
                # The decision is right and the identifiers are real, but the evidence never reached the model.
                self.assertEqual((d['format_compliant'], d['decision_correct'], d['evidence_correct'], d['failure_class']),
                                 (True, True, False, 'evidence'))
                self.assertEqual((d['observed'], d['required_evidence_observed']), (seen, False))

    def test_completed_wrong_or_malformed_final_answers_receive_zero(self):
        flipped = 'ready' if TRUTH['decision'] == 'not_ready' else 'not_ready'
        finals = {'wrong decision': (ANSWER(decision=flipped, supporting_run_ids=['run-00000000'] if flipped == 'ready' else []), 'semantic'),
                  'template date as as_of': (ANSWER(as_of='2026-11-11T00:00Z'), 'semantic'),
                  'fabricated inventory': (ANSWER(inventory_id='inv-000000000000'), 'evidence'),
                  'prose': (TEXT('The asset is not ready because the newest backup is still running.'), 'format'),
                  'fenced': (lambda st: TEXT('```json\n'+st.answer()+'\n```')(st), 'format'),
                  'call-shaped': (lambda st: TEXT(json.dumps({'name': 'final_answer', 'parameters': json.loads(st.answer())}))(st), 'format'),
                  'extra key': (lambda st: TEXT(st.answer()[:-1]+',"confidence":0.9}')(st), 'format')}
        for name, (final, failure) in finals.items():
            with self.subTest(name=name):
                result = self.run_case([SOLVE, SOLVE, SOLVE, final])
                out = self.complete(result, 0.0)
                self.assertEqual(out['diagnostics']['failure_class'], failure)
                self.assertEqual(out['successful_tool_calls'], 3)
                self.assertTrue(out['diagnostics']['required_evidence_observed'])

    def test_truncation_backend_cancellation_and_malformed_generations_are_incomplete(self):
        two_calls = lambda st: completion(st.step, calls=[('get_asset', task.dumps({'asset_id': st.asset_id})),
                                                          ('get_policy', task.dumps({'policy_id': TRUTH['policy_id']}))])
        truncated_action = lambda st: completion(st.step, calls=[('get_asset', task.dumps({'asset_id': st.asset_id}))], completion_tokens=1024)
        rest = [SOLVE, SOLVE, SOLVE, SOLVE]  # what a healthy episode would have done next
        scripts = {
            'truncated final': ([SOLVE, SOLVE, SOLVE, lambda st: TEXT(st.answer(), finish='length')(st)], 'Incomplete model generation (truncated)', 4),
            'ambiguous action at the token cap': ([truncated_action] + rest, 'Incomplete model generation (ambiguous)', 1),
            'truncated reflection': ([SOLVE, SOLVE, SOLVE, INVENTORY, INVENTORY, lambda st: TEXT(st.answer(), finish='length')(st)],
                                     'Incomplete model generation (truncated)', 6),
            'content filter': ([SOLVE, TEXT('x', finish='content_filter')] + rest, 'Incomplete model generation (unestablished)', 2),
            'two calls in one generation': ([two_calls] + rest, 'more than one tool call in a generation', 1),
            'backend failure first': ([FAIL(connection_error())] + rest, 'APIConnectionError', 1),
            'backend failure after tools': ([SOLVE, SOLVE, FAIL(connection_error())] + rest, 'APIConnectionError', 3),
            'backend failure at reflection': ([SOLVE, SOLVE, SOLVE, INVENTORY, INVENTORY, FAIL(connection_error())], 'APIConnectionError', 6),
            'timeout': ([SOLVE, FAIL(asyncio.TimeoutError)] + rest, 'TimeoutError', 2),
            'cancelled first': ([FAIL(asyncio.CancelledError)] + rest, 'CancelledError', 1),
            'cancelled after tools': ([SOLVE, SOLVE, SOLVE, FAIL(asyncio.CancelledError)], 'CancelledError', 4)}
        for name, (script, reason, requests) in scripts.items():
            with self.subTest(name=name):
                self.incomplete(self.run_case(script), reason, requests=requests)

    def test_original_sampled_tool_call_identity_is_preserved_through_the_retained_chain(self):
        import re
        asset = TRUTH['asset_id']
        normalize = lambda name: re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:64]  # what the pinned client does
        request = calibration_request()
        # A valid call is untouched and nothing is reported as rewritten.
        valid = self.run_case(request=request)
        served = [r['outputs'] for r in valid['records'] if r['kind'] == 'served_response']
        self.assertEqual([s['tool_calls'][0]['name'] for s in served[:3]], list(task.TOOLS))
        self.assertFalse(any(s['framework_normalized']['tool_names_differ'] for s in served))
        self.assertEqual(calibration.episode_row(valid['records'], request, 'scripted-inference')['tool_names_rewritten_by_framework'], 0)
        # Names the pinned client would rewrite. Two of them would become real tools.
        names = ('get asset', 'functions.get_asset', 'get.asset', 'list backup runs', 'get_asset\n', 'gét_asset', 'x' * 70)
        self.assertEqual([normalize(n) in task.TOOLS for n in names], [True, False, True, True, False, False, False])
        for name in names:
            with self.subTest(name=name):
                raw_arguments = json.dumps({'asset_id': asset}, indent=1)  # not the canonical form: kept byte for byte
                result = self.run_case([RAW(name, raw_arguments), SOLVE, SOLVE, SOLVE, SOLVE], request=request)
                out = self.complete(result, 1.0, 'complete_recovered')
                self.assertEqual((out['policy_errors'], out['successful_tool_calls'], out['tool_calls']), (1, 3, 3))
                records = result['records']
                # 1. The served response, recorded below the pinned client.
                first = next(r['outputs'] for r in records if r['kind'] == 'served_response')
                self.assertEqual(first['tool_calls'], [{'id': 'scripted-call-1-0', 'name': name, 'arguments': raw_arguments}])
                self.assertEqual(first['framework_normalized'], {'finish_reason': 'function_calls', 'tool_names': [normalize(name)],
                                                                  'tool_names_differ': True})
                # 2. The generation record: the call as sampled, not as normalized.
                generation = next(r for r in records if r['kind'] == 'llm')['outputs']['content']
                self.assertEqual(generation, [{'id': 'scripted-call-1-0', 'arguments': raw_arguments, 'name': name}])
                # 3. No tool was executed for it: normalization never manufactures a valid action.
                executed = [r['name'] for r in records if r['kind'] == 'tool']
                self.assertEqual(executed, list(task.TOOLS))
                # 4. The action record, its observation and its classification.
                action = next(r for r in records if r['kind'] == 'action_outcome')
                self.assertEqual(action['name'], name)
                self.assertEqual(action['inputs']['call'], {'id': 'scripted-call-1-0', 'arguments': raw_arguments, 'name': name})
                observation = action['inputs']['observation']
                self.assertEqual((observation['name'], observation['call_id'], observation['is_error'], observation['content']),
                                 (name, 'scripted-call-1-0', True, f"Error: tool '{name}' not found in any workbench"))
                self.assertEqual(action['outputs'], {'policy_error': 'unknown_tool', 'successful_tool_call': False})
                # 5. The next model context: the same identity goes back to the model.
                sent = result['requests'][1]['messages']
                self.assertEqual(sent[2]['tool_calls'], [{'id': 'scripted-call-1-0', 'type': 'function',
                                                          'function': {'arguments': raw_arguments, 'name': name}}])
                self.assertEqual((sent[3]['tool_call_id'], sent[3]['content']), ('scripted-call-1-0', observation['content']))
                self.assertIn('{"name": "' + name + '", "parameters": ', recovery_prompt.render(sent, result['requests'][1]['tools']))
                # 6. The reviewed row, and a recorded name cannot be swapped afterwards.
                row = calibration.episode_row(records, request, 'scripted-inference')
                self.assertEqual((row['sampled_calls'][0]['name'], row['sampled_calls'][0]['id'], row['sampled_calls'][0]['policy_error'],
                                  row['policy_error_codes'], row['tool_names_rewritten_by_framework']),
                                 (name, 'scripted-call-1-0', 'unknown_tool', ['unknown_tool'], 1))
                self.assertEqual([c['name'] for c in row['sampled_calls'][1:]], list(task.TOOLS))
                for edit in (lambda rs: next(r for r in rs if r['kind'] == 'llm')['outputs']['content'][0].update(name=normalize(name)),
                             lambda rs: next(r for r in rs if r['kind'] == 'served_response')['outputs']['tool_calls'][0].update(name=normalize(name)),
                             lambda rs: next(r for r in rs if r['kind'] == 'action_outcome')['inputs']['call'].update(name=normalize(name)),
                             lambda rs: next(r for r in rs if r['kind'] == 'action_outcome')['inputs']['observation'].update(name=normalize(name)),
                             lambda rs: rs.remove(next(r for r in rs if r['kind'] == 'served_response'))):
                    bad = copy.deepcopy(records); edit(bad)
                    with self.assertRaises((ValueError, KeyError)): calibration.episode_row(bad, request, 'scripted-inference')
        # The declared terminal limit applies to original identities, and nothing is executed.
        result = self.run_case([CALL('get asset', asset_id=asset), CALL('functions.get_asset', asset_id=asset)], request=request)
        out = self.complete(result, 0.0, 'terminal_policy_failure')
        self.assertEqual((out['policy_errors'], out['tool_calls'], len(result['requests'])), (2, 0, 2))
        self.assertEqual([a['inputs']['call']['name'] for a in result['records'] if a['kind'] == 'action_outcome'], ['get asset', 'functions.get_asset'])
        row = calibration.episode_row(result['records'], request, 'scripted-inference')
        self.assertEqual(([c['name'] for c in row['sampled_calls']], row['failure_class']), (['get asset', 'functions.get_asset'], 'action'))
        # Missing or inconsistent raw metadata fails closed: no reward, no admitted episode.
        for mode, reason in (('served-not-captured', 'Served response was not captured'),
                             ('parsed-arguments-differ', 'Parsed response differs from the served response'),
                             ('parsed-id-differs', 'Parsed response differs from the served response')):
            with self.subTest(mode=mode):
                self.incomplete(self.run_case(mode=mode), reason, requests=1)
        duplicate = lambda st: completion(st.step, calls=[('get_policy', task.dumps({'policy_id': TRUTH['policy_id']}))], call_ids=['scripted-call-1-0'])
        self.incomplete(self.run_case([SOLVE, duplicate, SOLVE, SOLVE]), 'Tool call id repeats an earlier call', requests=2)
        empty_id = lambda st: completion(st.step, calls=[('get_asset', task.dumps({'asset_id': st.asset_id}))], call_ids=[''])
        self.incomplete(self.run_case([empty_id, SOLVE, SOLVE, SOLVE]), 'malformed served tool call', requests=1)

    def test_completion_is_established_only_from_served_protocol_evidence(self):
        call = lambda st, **options: completion(st.step, calls=[('get_asset', task.dumps({'asset_id': st.asset_id}))], **options)
        rest = [SOLVE, SOLVE, SOLVE, SOLVE]
        # Complete answers that are malformed: scored zero as format failures.
        for name, final, kind in (('whitespace', TEXT(' \n\t'), 'whitespace'), ('whitespace without usage', TEXT('  ', usage=False), 'whitespace'),
                                  ('empty with reported completion tokens', TEXT('', completion_tokens=1), 'empty')):
            with self.subTest(name=name):
                request = calibration_request()
                result = self.run_case([SOLVE, SOLVE, SOLVE, final], request=request)
                out = self.complete(result, 0.0)
                self.assertEqual((out['diagnostics']['failure_class'], out['diagnostics']['json_syntax'], out['raw_answer'].strip()), ('format', False, ''))
                row = calibration.episode_row(result['records'], request, 'scripted-inference')
                self.assertEqual((row['answer_content'], row['failure_class'], row['decision_correct']), (kind, 'format', None))
        # Everything else stays an untrusted execution, with its own state.
        cases = {
            'empty answer without reported tokens': ([SOLVE, SOLVE, SOLVE, TEXT('', usage=False)], '(unestablished): empty answer without reported completion tokens', 4),
            'empty answer with zero tokens': ([SOLVE, SOLVE, SOLVE, TEXT('', completion_tokens=0)], '(unestablished): empty answer without reported completion tokens', 4),
            'absent content': ([SOLVE, SOLVE, SOLVE, lambda st: completion(st.step, content=None)], '(unestablished): answer content is absent', 4),
            'confirmed truncated answer': ([SOLVE, SOLVE, SOLVE, lambda st: TEXT(st.answer(), finish='length')(st)], '(truncated): the server reported a length finish', 4),
            'confirmed truncated call': ([lambda st: call(st, finish='length')] + rest, '(truncated): the server reported a length finish', 1),
            'ambiguous exact-cap call': ([lambda st: call(st, completion_tokens=1024)] + rest, '(ambiguous): completion tokens equal the cap', 1),
            'call above the cap': ([lambda st: call(st, completion_tokens=1025)] + rest, '(unestablished): reported completion tokens exceed the cap', 1),
            'call without reported usage': ([lambda st: call(st, usage=False)] + rest, '(unestablished): completion tokens were not reported', 1),
            'call with zero reported tokens': ([lambda st: call(st, completion_tokens=0)] + rest, '(unestablished): completion tokens were not reported', 1),
            'content filter': ([SOLVE, TEXT('x', finish='content_filter')] + rest, '(unestablished): answer did not finish with stop', 2)}
        for name, (script, reason, requests) in cases.items():
            with self.subTest(name=name):
                result = self.run_case(script)
                self.incomplete(result, reason, requests=requests)
                last = [r['outputs'] for r in result['records'] if r['kind'] == 'served_response'][-1]
                self.assertIn(last['completion']['state'], reason)
                self.assertNotEqual(last['completion']['state'], 'complete')
        # An answer that stops exactly at the cap is complete: `stop` is explicit for text.
        result = self.run_case([SOLVE, SOLVE, SOLVE, lambda st: TEXT(st.answer(), completion_tokens=1024)(st)])
        self.complete(result, 1.0)
        # Raw termination metadata and raw content presence are retained as served.
        result = self.run_case([SOLVE, SOLVE, SOLVE, lambda st: TEXT(st.answer(), stop_reason=128009)(st)])
        self.complete(result, 1.0)
        served = [r['outputs'] for r in result['records'] if r['kind'] == 'served_response']
        self.assertEqual([(s['finish_reason'], s['content_present'], s['stop_reason']) for s in served],
                         [('tool_calls', False, None)] * 3 + [('stop', True, 128009)])
        self.assertEqual([s['framework_normalized']['finish_reason'] for s in served], ['function_calls'] * 3 + ['stop'])
        self.assertEqual(served[0]['usage'], {'prompt_tokens': 0, 'completion_tokens': 12})
        # The pure rule, including what the pinned client would have hidden.
        status = task.generation_status
        self.assertEqual(status({'finish_reason': 'stop', 'content': ' ', 'tool_calls': [], 'usage': None}, 1024), ('complete', None))
        self.assertEqual(status({'finish_reason': 'stop', 'content': '', 'tool_calls': [], 'usage': {'completion_tokens': 1}}, 1024), ('complete', None))
        self.assertEqual(status({'finish_reason': 'stop', 'content': None, 'tool_calls': [], 'usage': {'completion_tokens': 5}}, 1024)[0], 'unestablished')
        self.assertEqual(status({'finish_reason': 'length', 'content': 'x', 'tool_calls': [], 'usage': {'completion_tokens': 1024}}, 1024)[0], 'truncated')

    def test_lost_or_altered_observations_and_failed_reward_delivery_are_incomplete(self):
        reasons = {'lost-observation': 'Missing or ambiguous action observation',
                   'tool-backend-error': 'Unclassified tool failure; terminal reward forbidden',
                   'corrupt-observation': 'Observation differs from the trusted environment; terminal reward forbidden'}
        for mode, step in (('lost-observation', 1), ('lost-observation', 3), ('tool-backend-error', 2),
                           ('corrupt-observation', 1), ('corrupt-observation', 3)):
            with self.subTest(mode=mode, step=step):
                result = self.run_case(mode=mode, fault_step=step)
                # Detected when the next generation is requested, before it is sent.
                self.incomplete(result, reasons[mode], requests=step)
                # The real tool call happened; its result did not reach the model intact.
                self.assertGreaterEqual(sum(r['kind'] == 'tool' for r in result['records']), step)
        result = self.run_case(mode='reward-failure')
        self.incomplete(result, 'HTTPStatusError', posts=1, requests=4)
        self.assertEqual(episode(result)['outputs']['reward'], 1.0)  # computed, never confirmed
        # A typed-looking error whose sampled call was valid is not a policy mistake.
        fake = type('Observation', (), {'content': task.dumps(task.policy_error('unknown_asset')), 'is_error': True})
        call = type('Call', (), {'name': 'get_asset', 'arguments': task.dumps({'asset_id': TRUTH['asset_id']})})
        self.assertIsNone(recovery_agent.classify(call, fake))
        fake.content = 'Error: backend unavailable'
        self.assertIsNone(recovery_agent.classify(call, fake))
        call.arguments = '{'
        self.assertIsNone(recovery_agent.classify(call, fake))

    def test_stage_rows_from_real_traces_and_scripted_evidence_is_never_live(self):
        greedy, packet = calibration_request(0), calibration_request(1)
        in_packet = next(r for r in task.rows('development') if r['case_id'] == packet['plan']['calibration']['stages']['sampled-packet-1']['case_ids'][0])
        outside = next(r for r in task.rows('development') if r['case_id'] not in packet['plan']['calibration']['stages']['sampled-packet-1']['case_ids'])
        result = self.run_case(request=packet, row=in_packet, route='train', rollout='ro-1', attempt='at-1')
        row = calibration.episode_row(result['records'], packet, 'scripted-inference')
        self.assertEqual((row['case_id'], row['rollout_id'], row['attempt_id'], row['route'], row['reward'], row['failure_class']),
                         (in_packet['case_id'], 'ro-1', 'at-1', 'train', 1.0, 'correct'))
        self.assertEqual((row['family'], row['variant']), task.family(in_packet['case_id']))
        self.assertEqual((row['successful_calls_by_tool'], row['joined_observation_chains'], row['llm_calls']),
                         ({'get_asset': 1, 'get_policy': 1, 'list_backup_runs': 1}, 3, 4))
        self.assertNotIn(episode(result)['outputs']['raw_answer'], json.dumps(row))
        # The pure fixtures used by the offline and pinned gates have exactly this schema.
        import recovery_fixtures
        self.assertEqual(set(row), set(recovery_fixtures.row(in_packet['case_id'], 'correct', 'train', 'ro-1')))
        self.assertEqual(set(row['sampled_calls'][0]), set(recovery_fixtures.row(in_packet['case_id'], 'correct', 'train', 'ro-1')['sampled_calls'][0]))
        with self.assertRaisesRegex(ValueError, 'Synthetic|live'): calibration.episode_row(result['records'], packet)
        with self.assertRaises(ValueError): calibration.episode_row(result['records'], greedy, 'scripted-inference')
        with self.assertRaises(ValueError): calibration.episode_row(
            self.run_case(request=packet, row=outside, route='train')['records'], packet, 'scripted-inference')
        with self.assertRaises(ValueError): calibration.episode_row(
            self.run_case(request=packet, row=in_packet, route='val')['records'], packet, 'scripted-inference')
        for edit in (lambda e: e['outputs'].update(reward=0.0), lambda e: e['outputs']['diagnostics'].update(decision_correct=False),
                     lambda e: e['outputs'].update(successful_tool_calls=2), lambda e: e.update(inputs={'question': outside['question']}),
                     lambda e: e.update(eval_split='final'), lambda e: e.update(episode_policy='asset-episode-v2')):
            bad = copy.deepcopy(result['records']); edit(next(r for r in bad if r['kind'] == 'episode'))
            with self.assertRaises(ValueError): calibration.episode_row(bad, packet, 'scripted-inference')
        # Recorded classifications are re-derived: an error observation that is not typed
        # evidence for its call cannot stay a policy error, and a recorded code cannot be changed.
        failed = self.run_case([bad_extra, RAW('get_policy', '{')], request=packet, row=in_packet, route='train')
        self.assertEqual(calibration.episode_row(failed['records'], packet, 'scripted-inference')['outcome'], 'terminal_policy_failure')
        for edit in (lambda a: a['inputs']['observation'].update(content='backend unavailable'),
                     lambda a: a['outputs'].update(policy_error='unknown_asset'),
                     lambda a: a['inputs']['call'].update(arguments=task.dumps({'asset_id': in_packet['question'].split()[3]}))):
            bad = copy.deepcopy(failed['records']); edit(next(r for r in bad if r['kind'] == 'action_outcome'))
            with self.assertRaises(ValueError): calibration.episode_row(bad, packet, 'scripted-inference')
        # Every successful call must be joined to the model input that saw its observation.
        bad = copy.deepcopy(result['records'])
        for record in bad:
            if record['kind'] == 'llm':
                for message in record['inputs']['messages']:
                    if message['type'] == 'FunctionExecutionResultMessage':
                        message['content'][0]['content'] = '{}'
        with self.assertRaisesRegex(ValueError, 'not joined'): calibration.episode_row(bad, packet, 'scripted-inference')
        # Another episode's observation cannot be spliced in as evidence.
        bad = copy.deepcopy(result['records'])
        other = self.run_case(request=packet, row=outside, route='train')
        donor = [r for r in other['records'] if r['kind'] == 'action_outcome'][2]
        target = [r for r in bad if r['kind'] == 'action_outcome'][2]
        target['inputs']['observation']['content'] = donor['inputs']['observation']['content']
        with self.assertRaises((ValueError, task.EvidenceIntegrityError)): calibration.episode_row(bad, packet, 'scripted-inference')

    def test_real_mcp_probe_and_scripted_demonstration(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = recovery_probe.check(Path(tmp)/'probe.json')
            self.assertEqual((receipt['status'], receipt['reference_solver']['verified'], len(receipt['checks'])), ('passed', 24, 13))
            self.assertTrue(receipt['owned_process_cleanup_completed'])
            with self.assertRaises(FileExistsError): recovery_probe.check(Path(tmp)/'probe.json')
            with self.assertRaises(ValueError): recovery_probe.check(Path(tmp)/'timeout.json', timeout=.001)
            self.assertEqual(json.loads((Path(tmp)/'timeout.json').read_text())['status'], 'failed')
            report = demonstration(Path(tmp)/'scripted.json')
            self.assertEqual((report['status'], report['evidence_class'], report['live_qualification'], len(report['chains'])),
                             ('passed', 'scripted-inference', False, 3))


class PreflightTests(unittest.TestCase):
    def test_remote_preflight_task_check_probes_and_measures_with_the_served_tokenizer(self):
        # Mocked provider boundary: the real probe and prompt rendering run locally; the
        # tokenizer stands in for the image's. The paid remote preflight itself is not executed.
        from types import SimpleNamespace
        from experiment import reader_preflight, prepare_model
        plan = calibration_request()['plan']
        template = prepare_model.effective_template()
        def tokenizer(scale, chat_template=template):
            return SimpleNamespace(chat_template=chat_template,
                                   encode=lambda text, add_special_tokens: [0] * (len(text) // scale))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for name in ('ok', 'long', 'template', 'unknown'):
                (run/name).mkdir()  # the remote action creates its run directory first
            receipt, budget = reader_preflight.task_probe(plan, run/'ok', tokenizer(5))
            self.assertEqual((receipt['status'], receipt['reference_solver']['verified'], budget['within_limits']), ('passed', 24, True))
            self.assertEqual(json.loads((run/'ok/prompt-budget.json').read_text()), budget)
            self.assertIs(budget['exhaustive_bound'], False)
            with self.assertRaisesRegex(ValueError, 'exceed the declared context limits'):
                reader_preflight.task_probe(plan, run/'long', tokenizer(2))
            self.assertFalse(json.loads((run/'long/prompt-budget.json').read_text())['within_limits'])  # failure evidence is kept
            with self.assertRaisesRegex(ValueError, 'template differs'):
                reader_preflight.task_probe(plan, run/'template', tokenizer(5, template.replace('11 Nov 2026', '12 Nov 2026')))
            unknown = {'profile': {'task': 'unknown-task'}}
            with self.assertRaisesRegex(ValueError, 'Unknown task'): reader_preflight.task_probe(unknown, run/'unknown', tokenizer(5))


class PromptTests(unittest.TestCase):
    def test_rendered_prompts_equal_what_the_harness_sends(self):
        asset, policy, as_of = TRUTH['asset_id'], TRUTH['policy_id'], TRUTH['as_of']
        inventory = ('list_backup_runs', {'asset_id': asset, 'as_of': as_of})
        paths = {'reference': ([SOLVE, SOLVE, SOLVE, FINAL], [('get_asset', {'asset_id': asset}), ('get_policy', {'policy_id': policy}), inventory], False),
                 'typed error and recovery': ([bad_extra, SOLVE, SOLVE, SOLVE, FINAL],
                                              [('get_asset', {'asset_id': asset, 'as_of': as_of}), ('get_asset', {'asset_id': asset}),
                                               ('get_policy', {'policy_id': policy}), inventory], False),
                 'five rounds and reflection': ([SOLVE, SOLVE, SOLVE, INVENTORY, INVENTORY, FINAL],
                                                [('get_asset', {'asset_id': asset}), ('get_policy', {'policy_id': policy}), inventory, inventory, inventory], True)}
        for name, (script, actions, reflect) in paths.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                result = run_episode(tmp, script)
                self.assertIsNone(result['error'])
                views = recovery_prompt.episode(ROW['question'], actions, reflect=reflect)
                self.assertEqual(len(views), len(result['requests']))
                for view, sent in zip(views, result['requests']):
                    # Call IDs are not rendered by the served template, so the prompts are identical.
                    self.assertEqual(view['prompt'], recovery_prompt.render(sent['messages'], sent.get('tools')))
                    self.assertEqual(view['tools_advertised'], 'tools' in sent)
                self.assertEqual([v['kind'] for v in views], ['initial'] + ['continuation'] * (len(views)-2) + ['reflection' if reflect else 'continuation'])
                first, last = views[0]['prompt'], views[-1]['prompt']
                self.assertTrue(first.startswith('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\n'))
                # The template's fixed date is present; the request's as_of is stated as authoritative.
                self.assertIn('Today Date: 11 Nov 2026\n\n', first)
                self.assertIn(as_of, first); self.assertNotIn('2026-11-11', first)
                self.assertIn('It is the current time for this task, even if another date appears elsewhere.', first)
                self.assertIn('That as_of is the only current time; ignore any other date.', first)
                self.assertIn('Given the following functions, please respond with a JSON for a function call', first)
                self.assertEqual(first.count('"type": "function"'), 3)
                self.assertTrue(last.endswith('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'))
                self.assertIn('<|start_header_id|>ipython<|end_header_id|>\n\n{"output": "{\\"asset_id\\"', last)
                self.assertIn('{"name": "get_asset", "parameters": {"asset_id": "'+asset+'"}}<|eot_id|>', last)
                for hidden in (ROW['result'], *task.FAMILIES, 'qualifying'):
                    self.assertNotIn(hidden, last)
                if reflect:
                    # Without tools the function-call preamble and the ipython environment line disappear.
                    self.assertNotIn('Given the following functions', last); self.assertNotIn('Environment: ipython', last)
                    self.assertTrue(last.endswith(recovery_agent.FINAL_INSTRUCTION+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'))
                    self.assertEqual(views[-1]['tool_choice'], 'none (no tools sent)')
                if name.startswith('typed'):
                    self.assertIn(json.dumps({'output': task.dumps(task.policy_error('invalid_arguments'))}), views[1]['prompt'])
        # Recorded trace inputs render to the same prompt as the wire request.
        with tempfile.TemporaryDirectory() as tmp:
            result = run_episode(tmp)
            for record, sent in zip([r for r in result['records'] if r['kind'] == 'llm'], result['requests']):
                messages, tools = recovery_prompt.from_trace(record['inputs'])
                self.assertEqual(recovery_prompt.render(messages, tools), recovery_prompt.render(sent['messages'], sent.get('tools')))

    def test_committed_token_measurement_is_current_and_states_its_assumptions(self):
        record = json.loads((ROOT/'evidence/phase4a-offline-20261124/prompt-budget.json').read_text())
        budget = record['budget']
        lock = json.loads((ROOT/'experiment/ppo_source_lock.json').read_text())['prepared_assets']['tokenizer.json']
        self.assertEqual(record['tokenizer_sha256'], lock['sha256'])
        # The measurement binds the exact prompts it counted; any prompt change invalidates it.
        selection = budget['selection']
        question = next(r['question'] for r in task.rows('development') if r['case_id'] == selection['longest_request_case_id'])
        self.assertEqual(budget['measured_prompts_sha256'],
                         recovery_prompt.prompt_set_sha256(selection['largest_observation_asset_id'], question))
        self.assertEqual(budget['context_limits'], task.CONTEXT_LIMITS)
        self.assertTrue(budget['within_limits'] and all(budget['checks'].values()))
        # It is a measurement of scenarios with stated assumptions, never described as a bound.
        self.assertIs(budget['exhaustive_bound'], False)
        self.assertEqual(set(budget['assumptions']), {'scenario', 'call_size', 'answer_size', 'fragmentation'})
        self.assertNotIn('worst_case_bounded', budget)
        self.assertEqual(selection['assets_compared'], len(task.registry()['runs']))
        self.assertIn('no character-length proxy', selection['method'])
        # The overall maximum covers every measured array.
        arrays = (budget['development_reference_path']['prompt_tokens_by_call_max'], budget['development_five_round_reflection']['prompt_tokens_by_call_max'],
                  budget['heavy_scenario']['prompt_tokens_by_call'], budget['heavy_scenario']['with_initial_typed_error'])
        self.assertEqual(budget['largest_measured_prompt_tokens'], max(max(a) for a in arrays))
        self.assertEqual(len(budget['largest_measured_prompt_covers']), 4)
        largest = budget['largest_measured_prompt_tokens']
        self.assertLessEqual(largest, task.CONTEXT_LIMITS['trajectory_max_prompt_length'])
        self.assertLessEqual(largest + task.CONTEXT_LIMITS['max_tokens'], task.CONTEXT_LIMITS['max_model_len'])
        self.assertLessEqual(budget['heavy_scenario']['merged_response_tokens'], task.CONTEXT_LIMITS['trajectory_max_response_length'])
        self.assertEqual(budget['headroom']['prompt_limit_minus_largest_measured_prompt'], task.CONTEXT_LIMITS['trajectory_max_prompt_length'] - largest)
        # The shared launcher value would not hold the heavy scenario: the task limit is necessary.
        self.assertGreater(largest, 2048)
        self.assertLessEqual(max(budget['development_reference_path']['prompt_tokens_by_call_max']), 2048)
        # Maxima are selected by token count. A counter that inverts the ordering selects differently.
        asset, tokens, compared = recovery_prompt.largest_inventory(len)
        other, _, _ = recovery_prompt.largest_inventory(lambda text: -len(text))
        self.assertNotEqual(asset, other)
        self.assertEqual(compared, 203)
        if os.environ.get('AGL_PINNED_TOKENIZER'):
            count = recovery_prompt.tokenizer(os.environ['AGL_PINNED_TOKENIZER'])
            self.assertEqual(recovery_prompt.budget(count), budget)
            self.assertEqual(recovery_prompt.largest_inventory(count)[:2], (selection['largest_observation_asset_id'], selection['largest_observation_tokens']))


if __name__ == '__main__':
    unittest.main()
